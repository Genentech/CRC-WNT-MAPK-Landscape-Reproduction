"""Panel emitters: Fig2f, Fig2h, Fig3a, Fig3b, Fig3c, Fig3d, Fig3e, Fig3f, Fig3g.

Each function returns the panel DataFrame, schema-matched to the published source-data
CSV header of that panel. Column-name parity is the goal; values are not expected to
match (synthetic cohort).
"""

from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import enet_genotype, feature_ladder as ladder_engine
from .data import (
    factor_annotation,
    load_arm_calls,
    load_arm_fractions,
    load_clinical,
    load_expression,
    load_factor_scores,
    load_focal_cnv,
    load_he_features,
    load_mutations,
    retained_factors,
)
from .signatures import mapk_ssgsea, wnt_ssgsea
from .variance_partition import genomic_blocks, genomic_variance_partition

_CACHE: dict[str, pd.DataFrame] = {}


def _factor_pca(n_components: int = 2) -> pd.DataFrame:
    """PCA of the retained (143) factor_scores.

    Design decision (recorded): factors are StandardScaler'd before PCA. MuVI factor
    scores in this synthetic block are not already zero-mean/unit-variance (they are
    non-negative, heterogeneous-scale floats), so an unscaled PCA would be dominated
    by the highest-variance factors rather than reflecting shared structure. The study's
    factor PCA is a precomputed artifact whose exact preprocessing is not available here,
    so this is a documented reimplementation choice, not a recovered fact.
    """
    key = f"pca_{n_components}"
    if key not in _CACHE:
        fs = load_factor_scores()[retained_factors()]
        X = StandardScaler().fit_transform(fs)
        pcs = PCA(n_components=n_components, random_state=42).fit_transform(X)
        cols = [f"PC{i + 1}" for i in range(n_components)]
        _CACHE[key] = pd.DataFrame(pcs, index=fs.index, columns=cols)
    return _CACHE[key]


def fig2f() -> pd.DataFrame:
    """Variance partition (partial R2) of the factor-PCA PC1/PC2 over mut, cnv_arm and HE.

    Target: PC1/PC2 of the factor PCA (`_factor_pca`). Variance partition via
    variance_partition.genomic_variance_partition -> methods.run_variance_partition.
    """
    pcs = _factor_pca(n_components=2)
    mut = load_mutations()
    he = load_he_features()
    clinical = load_clinical()
    # `HE_partial_R2` is HE's conditional partial R2 against the FULL ten-modality
    # joint model -- i.e. what H&E adds over genomics AND clinical AND purity AND
    # demographics. The genomic blocks are binary-encoded (amp/del and gain/loss, MSI
    # excluded from `mut`) -- see variance_partition's docstring.
    part = genomic_variance_partition(
        genomic_blocks(mut, load_focal_cnv(), load_arm_calls()),
        he,
        pcs,
        biopsy=clinical["biopsy_site"],
        msi=mut["MSI_High"],
        side=clinical["sidedness"],
        purity=clinical["purity"],
        age=clinical["age"],
        sex=clinical["sex"],
    )
    out = part.reset_index().rename(columns={"target": "pc"})
    return out[
        [
            "pc",
            "mutations_unique_partial_R2",
            "mut_arm_shared_partial_R2",
            "arm_cnv_unique_partial_R2",
            "HE_partial_R2",
        ]
    ]


def _signature_correlation_frame() -> pd.DataFrame:
    """factor, r_wnt, r_mapk, program, group, taxonomy, annotation_label -- 143 rows.

    `group` <- loadings reference's `mega_program` column, renamed to `group`.
    """
    if "sig_corr" in _CACHE:
        return _CACHE["sig_corr"]
    expr = load_expression()
    wnt = wnt_ssgsea(expr)
    mapk = mapk_ssgsea(expr)
    factors = retained_factors()
    fs = load_factor_scores()[factors]

    corr = pd.DataFrame({"r_wnt": fs.corrwith(wnt), "r_mapk": fs.corrwith(mapk)})
    corr.index.name = "factor"
    corr = corr.reset_index()

    ann = factor_annotation().set_index("factor").loc[factors]
    out = corr.merge(
        ann[["program", "mega_program", "taxonomy", "annotation_label"]].rename(
            columns={"mega_program": "group"}
        ),
        left_on="factor",
        right_index=True,
        how="left",
    )
    _CACHE["sig_corr"] = out
    return out


def fig2h() -> pd.DataFrame:
    """Per-factor Pearson correlation with the WNT and MAPK ssGSEA scores.

    MAPK is scored on the 10-gene `erk_activity_10` panel.
    """
    return _signature_correlation_frame()[
        [
            "factor",
            "r_wnt",
            "r_mapk",
            "program",
            "group",
            "taxonomy",
            "annotation_label",
        ]
    ]


def _combined_predictor_matrix() -> pd.DataFrame:
    """Inner-joined so H&E's missingness restricts the cohort instead of NaN-poisoning it.

    `pd.concat(..., axis=1)` defaults to an outer join, which would silently introduce
    NaN cells for the 1,108 samples without H&E rather than dropping them -- ElasticNetCV
    cannot fit through NaN, so the join must be explicit here. Focal-CNV columns get a
    `_cnv` suffix because they share gene names with mutation columns.
    """
    X = pd.concat(
        [
            load_mutations(),
            load_focal_cnv().add_suffix("_cnv"),
            load_arm_fractions(),
            load_he_features(),
        ],
        axis=1,
        join="inner",
    )
    if X.columns.duplicated().any():
        raise ValueError(
            f"duplicate predictor names: {X.columns[X.columns.duplicated()].tolist()}"
        )
    return X


def _factor_enet_table() -> pd.DataFrame:
    if "enet" in _CACHE:
        return _CACHE["enet"]
    X = _combined_predictor_matrix()
    fs = load_factor_scores()
    factors = retained_factors()
    table = enet_genotype.factor_enet_r2(X, fs, factors)
    _CACHE["enet"] = table
    return table


def fig3a() -> pd.DataFrame:
    """Per-factor cross-validated ElasticNet R2 (`r2_cv_mean`).

    REIMPLEMENTED per enet_genotype.py's docstring.
    """
    table = _factor_enet_table()
    return table[["factor", "r2_cv_mean"]]


def fig3b() -> pd.DataFrame:
    """Fig2h's correlation table joined with the ENET R2 table (inner merge on `factor`).

    A plain merge of the engines' outputs above.
    """
    corr = _signature_correlation_frame()
    r2 = _factor_enet_table()[["factor", "r2_cv_mean", "r2_cv_std"]]
    out = corr.merge(r2, on="factor", how="inner")
    return out[
        [
            "factor",
            "r_wnt",
            "r_mapk",
            "program",
            "group",
            "taxonomy",
            "annotation_label",
            "r2_cv_mean",
            "r2_cv_std",
        ]
    ]


def fig3c() -> pd.DataFrame:
    """Variance partition of the raw WNT ssGSEA score over the full cohort.

    Full-cohort partition with no signature residualisation, so purity is not a required
    input. Full cohort = all 5000 synthetic samples (no biopsy/treatment filtering),
    labelled "full_projected_sensitivity".
    """
    expr = load_expression()
    wnt = wnt_ssgsea(expr).to_frame("WNT")
    part = genomic_variance_partition(
        genomic_blocks(load_mutations(), load_focal_cnv(), load_arm_calls()),
        he=None,
        target=wnt,
    )
    row = part.loc["WNT"]
    return pd.DataFrame(
        {
            "component": [
                "mutations (unique)",
                "mut ∩ arm (shared)",
                "arm CNV (unique)",
            ],
            "partial_R2": [
                row["mutations_unique_partial_R2"],
                row["mut_arm_shared_partial_R2"],
                row["arm_cnv_unique_partial_R2"],
            ],
            "cohort": "full_projected_sensitivity",
            "n_samples_partition": int(row["n_samples_genomic"]),
        }
    )


def _wnt_ladders() -> pd.DataFrame:
    if "ladders" not in _CACHE:
        _CACHE["ladders"] = ladder_engine.wnt_ladders()
    return _CACHE["ladders"]


_LADDER_COLS = [
    "target",
    "space",
    "step",
    "feature_added",
    "cv_r2_mean",
    "cv_r2_std",
    "delta_r2",
    "n_samples",
    "n_candidates",
]


def _ladder_panel(space: str) -> pd.DataFrame:
    ladders = _wnt_ladders()
    df = (
        ladders.loc[ladders["space"] == space, _LADDER_COLS]
        .sort_values("step")
        .reset_index(drop=True)
    )
    return df


def fig3d() -> pd.DataFrame:
    """WNT ladder, space='combined'. See feature_ladder.py docstring for target/cohort."""
    return _ladder_panel("combined")


def fig3e() -> pd.DataFrame:
    """WNT ladder, space='arm'."""
    return _ladder_panel("arm")


def fig3f() -> pd.DataFrame:
    """WNT ladder, space='mut_cnv'."""
    return _ladder_panel("mut_cnv")


def fig3g() -> pd.DataFrame:
    """WNT ladder, space='he'."""
    return _ladder_panel("he")
