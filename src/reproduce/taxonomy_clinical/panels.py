"""Panel emitters wiring the three engines to the 15 published-CSV schemas.

Each function is zero-argument (called as `fn()`) and returns a
pandas DataFrame whose COLUMN NAMES match the published source-data CSV. Row values are
synthetic-cohort-derived and are NOT expected to numerically match the published
figures (see module docstrings in tree_engine/cox_engine/psm_engine for what is
reimplemented vs. reused, and PROVENANCE in __init__.py for the per-panel summary).

All cohort-wide fits (the two trees, the Cox models, the PS/matching pipeline) are
computed once and cached (`functools.lru_cache` on zero-arg helpers) so calling every
panel function does not refit anything 15 times.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from . import (
    cox_engine,
    data,
    pca as pca_mod,
    programs as programs_mod,
    psm_engine,
    ssgsea as ssgsea_mod,
    tree_engine,
)


def _msi_adjust(score: pd.Series, msi_high: pd.Series) -> pd.Series:
    """Residualize `score` against binary `msi_high` via simple OLS, re-centered on
    the original mean (a plain within-covariate adjustment, not a Cox model)."""
    x = msi_high.reindex(score.index).astype(float).to_numpy()
    y = score.to_numpy()
    var_x = np.var(x)
    b = np.cov(x, y, ddof=0)[0, 1] / var_x if var_x > 0 else 0.0
    a = y.mean() - b * x.mean()
    resid = y - (a + b * x)
    return pd.Series(resid + y.mean(), index=score.index)


@lru_cache(maxsize=1)
def _state() -> dict:
    taxonomy = data.load_taxonomy()
    mutations = data.load_mutations()
    focal_cnv = data.load_focal_cnv()
    arm_fractions = data.load_arm_fractions()
    he_features = data.load_he_features()
    factor_scores = data.load_factor_scores()
    expression = data.load_expression()
    clinical = data.load_clinical()
    survival = data.load_survival()
    treatment = data.load_treatment()
    loadings = data.load_loadings()

    l1 = tree_engine.fit_l1_context(factor_scores, arm_fractions, he_features)
    l2 = tree_engine.fit_l2_tree(
        factor_scores, mutations, focal_cnv, l1.context, taxonomy["mega_cluster"]
    )

    program_meta = programs_mod.program_metadata(loadings)
    prog_scores = programs_mod.program_scores(factor_scores, loadings, z=True)
    pcs = pca_mod.factor_pca(factor_scores, n_components=2)
    wnt = ssgsea_mod.ssgsea_score(expression, ssgsea_mod.WNT_GENES)
    wnt_msiadj = _msi_adjust(wnt, mutations["MSI_High"])

    per_sample_mc = l2.labels.map(l2.mega_cluster)

    return dict(
        taxonomy=taxonomy,
        mutations=mutations,
        focal_cnv=focal_cnv,
        arm_fractions=arm_fractions,
        he_features=he_features,
        factor_scores=factor_scores,
        expression=expression,
        clinical=clinical,
        survival=survival,
        treatment=treatment,
        loadings=loadings,
        l1=l1,
        l2=l2,
        program_meta=program_meta,
        prog_scores=prog_scores,
        pcs=pcs,
        wnt_msiadj=wnt_msiadj,
        per_sample_mc=per_sample_mc,
    )


@lru_cache(maxsize=1)
def _psm_state() -> dict:
    """Pooled PS model + pooled match (any regimen vs `none`) for SFig10a/b(pooled)/d,
    plus one independent 1:1 match per regimen for the arm-stratified panels
    (SFig10b's 3 arm rows, SFig10c, SFig10f, SFig10g, SFig10h)."""
    s = _state()
    clinical, mutations, treatment = s["clinical"], s["mutations"], s["treatment"]
    regimen = treatment["regimen"]
    covariates = psm_engine.build_ps_covariates(clinical, mutations, s["per_sample_mc"])
    treated = treatment["treated"]
    ps = psm_engine.fit_propensity(covariates, treated)
    match = psm_engine.match_1to1(ps, treated)
    matched_ids = set(match.treated_ids) | set(match.control_ids)
    matched_mask = pd.Series(covariates.index.isin(matched_ids), index=covariates.index)

    arm_matches = {
        arm: psm_engine.match_arm(covariates, regimen, arm) for arm in psm_engine.ARMS
    }

    return dict(
        covariates=covariates,
        treated=treated,
        regimen=regimen,
        ps=ps,
        match=match,
        matched_mask=matched_mask,
        arm_matches=arm_matches,
    )


@lru_cache(maxsize=1)
def _sfig8_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    s = _state()
    cohort = cox_engine.build_base_cohort(s["survival"], s["clinical"], s["mutations"])
    return cox_engine.fit_program_forests(cohort, s["prog_scores"], s["program_meta"])


def _matched_pooled_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    """Pooled matched paired deltas over `frame` (treated-of-any-regimen minus its
    matched control) -- feeds SFig10e, which has no `arm` column in its schema."""
    match = _psm_state()["match"]
    treated_vals = frame.loc[match.treated_ids].reset_index(drop=True)
    control_vals = frame.loc[match.control_ids].reset_index(drop=True)
    return treated_vals - control_vals


@lru_cache(maxsize=1)
def _sfig10e_features() -> pd.DataFrame:
    """The 17 programs joined with PC1-PC3, as one feature matrix.

    Both SFig10e estimators apply to every column of this single frame uniformly,
    with no per-feature-type branch. This join is what yields the published
    panel's 20 rows.
    """
    s = _state()
    pcs3 = pca_mod.factor_pca(s["factor_scores"], n_components=3)
    return s["prog_scores"].join(pcs3, how="inner")


def _matched_deltas_for_arm(arm: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Matched paired deltas (treated_arm minus its matched control) for `frame`
    (program scores or PCs), using that arm's own independent 1:1 match."""
    match = _psm_state()["arm_matches"][arm]
    treated_vals = frame.loc[match.treated_ids].reset_index(drop=True)
    control_vals = frame.loc[match.control_ids].reset_index(drop=True)
    return treated_vals - control_vals


def _paired_delta_rows(delta: pd.DataFrame, arm: str) -> tuple[list[dict], list[float]]:
    rows, pvals = [], []
    for col in delta.columns:
        d = delta[col].to_numpy()
        n = len(d)
        mean_delta = float(d.mean()) if n > 0 else float("nan")
        se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        ci_low, ci_high = mean_delta - 1.96 * se, mean_delta + 1.96 * se
        tstat = mean_delta / se if se and np.isfinite(se) and se > 0 else float("nan")
        p = (
            2 * (1 - stats.t.cdf(abs(tstat), df=n - 1))
            if np.isfinite(tstat) and n > 1
            else float("nan")
        )
        pvals.append(p)
        rows.append(
            {
                "feature": col,
                "arm": arm,
                "n_treated": n,
                "mean_delta": mean_delta,
                "se_arm": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return rows, pvals


# ---------------------------------------------------------------------------
# Engine 1 panels: SFig6, SFig7, Fig4b
# ---------------------------------------------------------------------------


def sfig6() -> pd.DataFrame:
    l1 = _state()["l1"]
    rows = [
        {"l1_context": ctx, "rule": l1.rule[ctx], "n": int(n)}
        for ctx, n in l1.counts.items()
    ]
    return pd.DataFrame(rows).sort_values("l1_context").reset_index(drop=True)


def sfig7() -> pd.DataFrame:
    l2 = _state()["l2"]
    rows = [
        {
            "group_short": group,
            "group_full": l2.group_full.get(group, ""),
            "mega_cluster": l2.mega_cluster.get(group, "NA"),
            "n": int(n),
        }
        for group, n in l2.counts.items()
    ]
    return pd.DataFrame(rows).sort_values("group_short").reset_index(drop=True)


def fig4b() -> pd.DataFrame:
    s = _state()
    l2 = s["l2"]
    prog_scores = s["prog_scores"]
    common = prog_scores.index.intersection(l2.labels.index)
    df = prog_scores.loc[common].copy()
    df["group"] = l2.labels.loc[common].to_numpy()
    group_means = df.groupby("group").mean()

    # Re-z-score across the group axis for each program row.
    std = group_means.std(axis=0, ddof=0).replace(0, 1)
    rez = (group_means - group_means.mean(axis=0)) / std

    mat = rez.T
    mat.index.name = "program"
    mat = mat.reset_index()

    meta = s["program_meta"].rename(columns={"program_concise": "annotation_label"})
    return mat.merge(
        meta[["program", "annotation_label", "mega_program", "taxonomy"]],
        on="program",
        how="left",
    )


# ---------------------------------------------------------------------------
# Engine 2 panels: Fig4c, SFig8a, SFig8b
# ---------------------------------------------------------------------------


def fig4c() -> pd.DataFrame:
    s = _state()
    l2 = s["l2"]
    cohort = cox_engine.build_base_cohort(s["survival"], s["clinical"], s["mutations"])
    outcomes = cox_engine.fit_group_outcomes(
        cohort,
        l2.labels,
        s["treatment"]["received_1L"],
        s["wnt_msiadj"],
        tau=cox_engine.RMST_TAU,
    )
    outcomes["mutation_group_id"] = outcomes["mutation_group"].map(l2.mutation_group_id)
    outcomes["mega_cluster"] = outcomes["mutation_group"].map(l2.mega_cluster)
    cols = [
        "mutation_group_id",
        "mega_cluster",
        "n",
        "n_events",
        "wnt_msiadj_mean",
        "rmst_36_adjusted",
        "rmst_36_se",
        "km_estimator",
    ]
    return outcomes[cols].sort_values("mutation_group_id").reset_index(drop=True)


def sfig8a() -> pd.DataFrame:
    return _sfig8_pair()[0]


def sfig8b() -> pd.DataFrame:
    return _sfig8_pair()[1]


# ---------------------------------------------------------------------------
# Engine 3 panels: Fig4d, SFig10a-h
# ---------------------------------------------------------------------------


def fig4d() -> pd.DataFrame:
    s = _state()
    pcs, per_sample_mc, treatment = s["pcs"], s["per_sample_mc"], s["treatment"]
    common = pcs.index.intersection(per_sample_mc.index).intersection(treatment.index)
    df = pcs.loc[common].copy()
    df["mega_cluster"] = per_sample_mc.loc[common].to_numpy()
    df["treated"] = treatment.loc[common, "treated"].to_numpy()

    rows = []
    for mc, g in df.groupby("mega_cluster"):
        naive, treated_g = g[~g["treated"]], g[g["treated"]]
        if len(naive) == 0 or len(treated_g) == 0:
            continue
        rows.append(
            {
                "mega_cluster": mc,
                "PC1_naive": naive["PC1"].mean(),
                "PC2_naive": naive["PC2"].mean(),
                "PC1_treated": treated_g["PC1"].mean(),
                "PC2_treated": treated_g["PC2"].mean(),
                "dPC1": treated_g["PC1"].mean() - naive["PC1"].mean(),
                "dPC2": treated_g["PC2"].mean() - naive["PC2"].mean(),
                "n_naive": len(naive),
                "n_treated": len(treated_g),
            }
        )
    return pd.DataFrame(rows).sort_values("mega_cluster").reset_index(drop=True)


def sfig10a() -> pd.DataFrame:
    s = _psm_state()
    covariates, treated, match = s["covariates"], s["treated"].astype(bool), s["match"]
    matched_t = covariates.loc[match.treated_ids]
    matched_c = covariates.loc[match.control_ids]
    raw_t, raw_c = covariates.loc[treated], covariates.loc[~treated]

    rows = [
        {
            "covariate": col,
            "mean_abs_smd_raw": psm_engine.smd(
                raw_t[col].to_numpy(), raw_c[col].to_numpy()
            ),
            "mean_abs_smd_matched": psm_engine.smd(
                matched_t[col].to_numpy(), matched_c[col].to_numpy()
            ),
        }
        for col in covariates.columns
    ]
    return pd.DataFrame(rows)


def sfig10b() -> pd.DataFrame:
    """4 rows: pooled scope + one per regimen arm (published schema keeps the pooled
    'Tier1_pooled' row alongside the three arms)."""
    s = _psm_state()
    covariates, regimen = s["covariates"], s["regimen"]
    mask = s["matched_mask"]
    auc_pooled = psm_engine.cross_validated_auc(
        covariates.loc[mask], s["treated"].loc[mask]
    )
    rows = [{"scope": "Tier1_pooled", "label": "Pooled", "c2st_auc": auc_pooled}]
    for arm in psm_engine.ARMS:
        match = s["arm_matches"][arm]
        matched_ids = match.treated_ids + match.control_ids
        if len(matched_ids) < 10:  # too few pairs for a stable 5-fold AUC
            rows.append({"scope": arm, "label": arm, "c2st_auc": float("nan")})
            continue
        arm_treated = regimen.reindex(matched_ids) == arm
        auc = psm_engine.cross_validated_auc(covariates.loc[matched_ids], arm_treated)
        rows.append({"scope": arm, "label": arm, "c2st_auc": auc})
    return pd.DataFrame(rows)


def sfig10c() -> pd.DataFrame:
    """3 rows, one per regimen arm (no pooled row in the published schema)."""
    s = _psm_state()
    regimen = s["regimen"]
    rows = []
    for arm in psm_engine.ARMS:
        n_pre = int((regimen == arm).sum())
        n_matched = len(s["arm_matches"][arm].treated_ids)
        frac = n_matched / n_pre if n_pre else float("nan")
        rows.append(
            {
                "arm": arm,
                "n_treated_prematch": n_pre,
                "n_matched": n_matched,
                "frac_matched": frac,
            }
        )
    return pd.DataFrame(rows)


def sfig10d() -> pd.DataFrame:
    s = _psm_state()
    ps, treated = s["ps"], s["treated"].astype(bool)
    bins = np.linspace(0, 1, 21)
    centers = (bins[:-1] + bins[1:]) / 2
    treated_density, _ = np.histogram(ps.loc[treated], bins=bins, density=True)
    naive_density, _ = np.histogram(ps.loc[~treated], bins=bins, density=True)
    return pd.DataFrame(
        {
            "bin_center": centers,
            "treated_density": treated_density,
            "naive_density": naive_density,
        }
    )


def sfig10e() -> pd.DataFrame:
    """20 rows: the 17 programs plus PC1-PC3, matching the published panel.

    `real_pair_mean_delta` is NaN for EVERY row, not just some: it needs longitudinal
    pre/post biopsy pairs, which the synthetic cohort does not have (one row per sample). `psm_mean_delta` is computable for all 20 and
    is emitted for all 20 -- the two columns are different ESTIMATORS applied to the
    same feature matrix, so the unavailability of one says nothing about the other.
    """
    features = _sfig10e_features()
    psm_mean = _matched_pooled_deltas(features).mean(axis=0)
    pcs = [c for c in features.columns if c.startswith("PC")]
    rows = [
        {
            "feature": feature,
            "feature_type": "principal_component" if feature in pcs else "program",
            "real_pair_mean_delta": np.nan,
            "psm_mean_delta": float(psm_mean[feature]),
        }
        for feature in features.columns
    ]
    return pd.DataFrame(rows).reset_index(drop=True)


def sfig10f() -> pd.DataFrame:
    """17 programs x 3 arms = 51 rows (unless an arm's match is too small to yield a
    program's delta, in which case that arm/program combination is dropped and
    reported, not padded)."""
    s = _state()
    prog = s["prog_scores"]
    meta = s["program_meta"].rename(columns={"program_concise": "annotation_label"})
    all_rows = []
    for arm in psm_engine.ARMS:
        delta = _matched_deltas_for_arm(arm, prog)
        rows, pvals = _paired_delta_rows(delta, arm)
        # BH-FDR is computed WITHIN each arm ("fdr_per_arm"), not pooled across arms.
        fdr = psm_engine.bh_fdr(np.array(pvals))
        for row, q in zip(rows, fdr):
            row["fdr_per_arm"] = q
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df = df.merge(
        meta[["program", "annotation_label", "mega_program", "taxonomy"]],
        left_on="feature",
        right_on="program",
        how="left",
    ).drop(columns=["program"])
    return df


def sfig10g() -> pd.DataFrame:
    """2 PCs x 3 arms = 6 rows."""
    s = _state()
    pcs = s["pcs"]
    all_rows = []
    for arm in psm_engine.ARMS:
        delta = _matched_deltas_for_arm(arm, pcs)
        rows, pvals = _paired_delta_rows(delta, arm)
        fdr = psm_engine.bh_fdr(np.array(pvals))
        for row, q in zip(rows, fdr):
            row["fdr_per_arm"] = q
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)


def sfig10h() -> pd.DataFrame:
    """3 rows, one per regimen arm."""
    s = _state()
    pcs = s["pcs"]
    arm_matches = _psm_state()["arm_matches"]
    rows = []
    for arm in psm_engine.ARMS:
        match = arm_matches[arm]
        n = len(match.treated_ids)
        if n == 0:
            rows.append(
                {
                    "group_label": arm,
                    "PC1_naive": np.nan,
                    "PC2_naive": np.nan,
                    "PC1_treated": np.nan,
                    "PC2_treated": np.nan,
                    "n_weight": 0,
                    "n_naive": 0,
                    "n_treated": 0,
                }
            )
            continue
        treated_pcs = pcs.loc[match.treated_ids]
        control_pcs = pcs.loc[match.control_ids]
        rows.append(
            {
                "group_label": arm,
                "PC1_naive": control_pcs["PC1"].mean(),
                "PC2_naive": control_pcs["PC2"].mean(),
                "PC1_treated": treated_pcs["PC1"].mean(),
                "PC2_treated": treated_pcs["PC2"].mean(),
                "n_weight": n,
                "n_naive": n,
                "n_treated": n,
            }
        )
    return pd.DataFrame(rows)
