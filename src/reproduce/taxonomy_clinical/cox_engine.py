"""Engine 2 -- Cox PH survival, left-truncated on `time_from_entry_dx_months`.

SFig8a/b follow the study's analysis code (`lifelines.CoxPHFitter`, `entry_col`,
penalizers, covariate set, BH-FDR, forest selection). Fig4c's g-computation is a
REIMPLEMENTATION: the study's fitting code is not part of this package. The design
choice made here -- one joint Cox model with `mutation_group` as `strata`
(group-specific baseline hazard, shared covariate effects) rather than 127 dummy
covariates -- is a standard direct-standardization recipe; documented in PROVENANCE.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time

from ..methods.rmst import rmst_se_greenwood
from .programs import zscore

logger = logging.getLogger(__name__)

PER_PROGRAM_PENALIZER = 0.01
GLOBAL_PENALIZER = 0.05
GROUP_STRATA_PENALIZER = 0.05
FOREST_N = 35
RMST_TAU = 36.0

# `np.trapz` is deprecated in NumPy 2 in favour of `np.trapezoid`.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def bh_fdr(pvals: pd.Series) -> pd.Series:
    p = pd.Series(pvals, dtype=float)
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    if valid.empty:
        return out
    m = len(valid)
    ranks = np.arange(1, m + 1)
    q = (valid.to_numpy() * m) / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out.loc[valid.index] = q
    return out


def build_base_cohort(
    survival: pd.DataFrame, clinical: pd.DataFrame, mutations: pd.DataFrame
) -> pd.DataFrame:
    """time/event/entry + the 5-covariate clinical adjustment set used by SFig8a/b."""
    df = pd.DataFrame(index=survival.index)
    df["time"] = survival["os_time"].astype(float)
    df["event"] = survival["os_event"].astype(int)
    df["entry"] = survival["time_from_entry_dx_months"].astype(float)
    df["age"] = clinical["age"].astype(float)
    df["purity"] = clinical["purity"].astype(float)
    df["sex_male"] = (clinical["sex"] == "male").astype(int)
    df["msi_high"] = mutations["MSI_High"].astype(int)
    df["metastatic"] = clinical["metastatic"].astype(int)
    df = df[(df["time"] > 0) & (df["entry"] < df["time"])].copy()
    df["age_z"] = zscore(df["age"])
    df["purity_z"] = zscore(df["purity"])
    return df


def fit_cox(
    df: pd.DataFrame, covariates: list[str], *, penalizer: float
) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(
        df[["time", "event", "entry"] + covariates],
        duration_col="time",
        event_col="event",
        entry_col="entry",
        show_progress=False,
    )
    return cph


def fit_program_forests(
    cohort: pd.DataFrame, program_scores_z: pd.DataFrame, program_meta: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SFig8a (per-program adjusted forest) + SFig8b (joint multivariable model)."""
    clinical_covariates = ["age_z", "purity_z", "sex_male", "msi_high", "metastatic"]
    common = cohort.index.intersection(program_scores_z.index)
    cohort = cohort.loc[common].join(program_scores_z.loc[common])
    program_names = program_scores_z.columns.tolist()

    rows = []
    for program in program_names:
        adj_cols = ["time", "event", "entry", program] + clinical_covariates
        adj_df = cohort[adj_cols].dropna()
        if adj_df[program].std() == 0 or len(adj_df) == 0:
            continue
        cph_adj = fit_cox(
            adj_df, [program] + clinical_covariates, penalizer=PER_PROGRAM_PENALIZER
        )
        s1 = cph_adj.summary.loc[program]
        rows.append(
            {
                "program": program,
                "hazard_ratio": float(np.exp(s1["coef"])),
                "hr_lower": float(np.exp(s1["coef lower 95%"])),
                "hr_upper": float(np.exp(s1["coef upper 95%"])),
                "p_value": float(s1["p"]),
            }
        )
    forest = pd.DataFrame(rows)
    forest["fdr"] = bh_fdr(forest["p_value"])
    forest["abs_log_hr"] = np.abs(np.log(forest["hazard_ratio"]))
    forest = forest.merge(program_meta, on="program", how="left")

    plot_df = forest.sort_values(
        ["fdr", "p_value", "abs_log_hr"], ascending=[True, True, False]
    )
    plot_df = plot_df.head(FOREST_N).sort_values("hazard_ratio").copy()
    plot_df["significance_category"] = np.where(
        plot_df["fdr"] < 0.10, "FDR < 0.10", "not significant"
    )

    sfig8a = plot_df[
        [
            "program",
            "program_concise",
            "mega_program",
            "taxonomy",
            "hazard_ratio",
            "hr_lower",
            "hr_upper",
            "p_value",
            "fdr",
            "significance_category",
        ]
    ].reset_index(drop=True)

    # ---- Joint model ----
    covariates = program_names + clinical_covariates
    df_model = cohort[["time", "event", "entry"] + covariates].dropna()
    cph = CoxPHFitter(penalizer=GLOBAL_PENALIZER)
    cph.fit(
        df_model,
        duration_col="time",
        event_col="event",
        entry_col="entry",
        show_progress=False,
    )

    coef = cph.summary.reset_index().rename(columns={"covariate": "feature"})
    coef["hazard_ratio"] = np.exp(coef["coef"])
    coef["abs_coef"] = coef["coef"].abs()
    coef["feature_type"] = np.where(
        coef["feature"].isin(program_names), "program", "clinical"
    )
    coef = coef.merge(
        program_meta.rename(columns={"program": "feature"}), on="feature", how="left"
    )
    coef = (
        coef.sort_values("abs_coef", ascending=False).sort_values("hazard_ratio").copy()
    )
    coef["significance_category"] = np.where(
        coef["p"] < 0.05, "p < 0.05", "not significant"
    )

    sfig8b = coef[
        [
            "feature",
            "feature_type",
            "mega_program",
            "program_concise",
            "taxonomy",
            "coef",
            "hazard_ratio",
            "exp(coef) lower 95%",
            "exp(coef) upper 95%",
            "p",
            "abs_coef",
            "significance_category",
        ]
    ].reset_index(drop=True)

    return sfig8a, sfig8b


def _unadjusted_rmst(df_group: pd.DataFrame, *, tau: float = RMST_TAU):
    """Unadjusted KM RMST(tau) + Greenwood SE of the RMST *estimator*, under left truncation.

    The SE comes from `methods.rmst.rmst_se_greenwood` (Klein &
    Moeschberger 1997 eq. 4.5.5). We do NOT use
    `restricted_mean_survival_time(..., return_variance=True)`'s variance: despite
    the docstring calling it "Greenwood-type", it returns
    `E[T ∧ tau²] - E[T ∧ tau]²` -- the variance of the restricted
    lifetime *random variable*, a population dispersion measure, not the sampling
    variance of the RMST estimator (and it is frequently non-finite under left
    truncation).
    """
    if len(df_group) == 0 or df_group["event"].sum() == 0:
        return np.nan, np.nan, "not_estimable"
    kmf = KaplanMeierFitter()
    try:
        kmf.fit(
            df_group["time"], event_observed=df_group["event"], entry=df_group["entry"]
        )
    except Exception as exc:
        logger.warning(
            "KM fit failed for group (n=%d, events=%d): %s: %s",
            len(df_group),
            int(df_group["event"].sum()),
            type(exc).__name__,
            exc,
        )
        return np.nan, np.nan, "not_estimable"
    try:
        rmst = float(restricted_mean_survival_time(kmf, t=tau))
    except Exception as exc:
        logger.warning(
            "RMST(tau=%s) not estimable for group (n=%d, events=%d): %s: %s",
            tau,
            len(df_group),
            int(df_group["event"].sum()),
            type(exc).__name__,
            exc,
        )
        return np.nan, np.nan, "not_estimable"
    se = rmst_se_greenwood(kmf, tau)
    if not np.isfinite(rmst) or not np.isfinite(se):
        return np.nan, np.nan, "not_estimable"
    return rmst, se, "KM"


def fit_group_outcomes(
    cohort: pd.DataFrame,
    group_labels: pd.Series,
    received_1l: pd.Series,
    wnt_msiadj: pd.Series,
    *,
    tau: float = RMST_TAU,
) -> pd.DataFrame:
    """Per-L2-group outcome table for Fig4c: n, n_events, wnt_msiadj_mean,
    rmst_36_adjusted (direct-standardized Cox marginal), rmst_36_se + km_estimator
    (unadjusted Greenwood, left-truncated).
    """
    common = cohort.index.intersection(group_labels.index)
    df = cohort.loc[common].copy()
    df["mutation_group"] = group_labels.loc[common].to_numpy()
    df["received_1L"] = received_1l.reindex(common).astype(int).to_numpy()

    covariates = [
        "age_z",
        "purity_z",
        "sex_male",
        "msi_high",
        "metastatic",
        "received_1L",
    ]
    model_df = (
        df[["time", "event", "entry", "mutation_group"] + covariates].dropna().copy()
    )

    group_counts = model_df["mutation_group"].value_counts()
    fittable_groups = group_counts[group_counts >= 5].index.tolist()

    cph = None
    if len(fittable_groups) >= 2:
        strat_df = model_df[model_df["mutation_group"].isin(fittable_groups)]
        try:
            cph = CoxPHFitter(penalizer=GROUP_STRATA_PENALIZER)
            cph.fit(
                strat_df,
                duration_col="time",
                event_col="event",
                entry_col="entry",
                strata="mutation_group",
                show_progress=False,
            )
        except Exception as exc:  # reported below; groups fall back to unadjusted RMST
            cph = None
            fit_error = f"{type(exc).__name__}: {exc}"
        else:
            fit_error = None
    else:
        fit_error = None

    rows = []
    errors: dict[str, str] = {}
    for group, df_g in df.groupby("mutation_group"):
        n = len(df_g)
        n_events = int(df_g["event"].sum())
        rmst_unadj, se, estimator = _unadjusted_rmst(df_g, tau=tau)

        rmst_adjusted = np.nan
        if cph is not None and group in fittable_groups:
            standardize_df = model_df.copy()
            standardize_df["mutation_group"] = group
            try:
                surv = cph.predict_survival_function(
                    standardize_df, times=np.linspace(0, tau, 50)
                )
                mean_surv = surv.mean(axis=1)
                rmst_adjusted = float(
                    _trapezoid(mean_surv.to_numpy(), mean_surv.index.to_numpy())
                )
            except Exception as exc:  # reported below
                rmst_adjusted = np.nan
                errors[str(group)] = f"{type(exc).__name__}: {exc}"
        if not np.isfinite(rmst_adjusted):
            rmst_adjusted = rmst_unadj  # falls back to the unadjusted marginal

        rows.append(
            {
                "mutation_group": group,
                "n": n,
                "n_events": n_events,
                "wnt_msiadj_mean": float(wnt_msiadj.reindex(df_g.index).mean()),
                "rmst_36_adjusted": rmst_adjusted,
                "rmst_36_se": se,
                "km_estimator": estimator,
            }
        )
    if fit_error is not None:
        warnings.warn(
            f"stratified Cox fit failed ({fit_error}); rmst_36_adjusted is the "
            "unadjusted RMST for every group",
            stacklevel=2,
        )
    if errors:
        warnings.warn(
            f"adjusted RMST failed for {len(errors)} group(s), which use the unadjusted "
            f"RMST instead: {errors}",
            stacklevel=2,
        )
    return pd.DataFrame(rows)
