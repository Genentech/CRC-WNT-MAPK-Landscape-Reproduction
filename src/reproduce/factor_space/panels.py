"""Panel emitters: Fig2b, Fig2d, SFig3a, SFig3b, SFig3c.

Each function returns the panel DataFrame, schema-matched to the published
source-data header. Values are not expected to match (synthetic cohort).
"""

import pandas as pd

from . import pca, programs, reconstruction
from .data import load_factor_scores


def _order_by_mega_program(items: list[str], meta: pd.DataFrame) -> list[str]:
    """Stable order by `mega_program`, then by the raw label string.

    Unassigned/blank groups sort last under "Unassigned".
    """

    def key(p: str):
        val = meta.at[p, "mega_program"] if p in meta.index else None
        if pd.isna(val) or not str(val).strip():
            return (1, "Unassigned", p)
        return (0, str(val), p)

    return sorted(items, key=key)


def fig2b() -> pd.DataFrame:
    """17x17 Pearson correlation matrix of the variance-weighted program scores."""
    scores = programs.program_score_frame(weight="variance")
    corr = scores.corr(method="pearson")
    meta = programs.program_metadata()

    # Rows and columns carry the SAME mega-program-grouped order as the published
    # panel. The order is a function of the ANNOTATION, not of the correlation
    # values, so it reproduces exactly on synthetic data.
    # Note the secondary key is the raw program string, which sorts "P10_" before
    # "P1_"; that matches the published panel, it is not a bug.
    ordered = _order_by_mega_program(list(corr.index), meta)
    corr = corr.reindex(index=ordered, columns=ordered)
    meta = meta.reindex(ordered)

    out = corr.reset_index().rename(columns={"index": "program"})
    out.insert(1, "group", meta.loc[out["program"], "mega_program"].to_numpy())
    out.insert(2, "taxonomy", meta.loc[out["program"], "taxonomy"].to_numpy())
    out["annotation_label"] = meta.loc[out["program"], "annotation_label"].to_numpy()
    return out


def fig2d() -> pd.DataFrame:
    """17-program PCA biplot coordinates (PC1, PC2), variance-weighted."""
    agg = pca.program_aggregate_loadings(n_pcs=2, weight="variance")
    return agg[
        [
            "program",
            "mega_program",
            "group",
            "taxonomy",
            "annotation_label",
            "n_factors",
            "PC1",
            "PC2",
        ]
    ]


def sfig3a() -> pd.DataFrame:
    """Top-20 (of the 144 available) factors by reconstruction R^2.

    `dense_0` is excluded from the reported rows (as in the published table,
    whose ranks start at 2) but keeps its rank in `r2_rank_all_factors`.
    """
    table = reconstruction.reconstruction_table()
    non_dense = table[table["factor"] != "dense_0"]
    top20 = non_dense.head(20).copy()
    top20 = top20.rename(columns={"rank": "r2_rank_all_factors"})
    return top20[
        [
            "r2_rank_all_factors",
            "factor",
            "reconstruction_r2",
            "r2_share_pct",
            "total_factor_r2",
        ]
    ].reset_index(drop=True)


def sfig3b() -> pd.DataFrame:
    """Full available factor set (144 of the published 270), ranked by reconstruction R^2.

    `in_shaded_retained_region` is True for every row here: the 126 non-retained
    pre-filter factors the published 270-row table also carries are not in the
    synthetic cohort's 144-column `factor_scores` block, so only the
    always-retained portion of the curve is reproducible (see the
    reconstruction.py docstring).
    """
    table = reconstruction.reconstruction_table()
    total = table["total_factor_r2"].iloc[0]
    table = table.copy()
    table["cumulative_share_of_total_factor_r2_pct"] = (
        table["reconstruction_r2"].cumsum() / total * 100
    )
    table["in_shaded_retained_region"] = True
    return table[
        [
            "rank",
            "factor",
            "reconstruction_r2",
            "cumulative_share_of_total_factor_r2_pct",
            "in_shaded_retained_region",
            "total_factor_r2",
        ]
    ]


def sfig3c() -> pd.DataFrame:
    """143x143 raw factor correlation matrix.

    Plain `.corr()` of the retained factor_scores columns plus the annotation join.
    """
    ann = programs.factor_annotation()
    retained = ann.loc[ann["program"].notna(), "factor"].tolist()
    scores = load_factor_scores()[retained]
    corr = scores.corr(method="pearson")
    meta = ann.set_index("factor").loc[retained]

    out = corr.reset_index().rename(columns={"index": "factor"})
    out.insert(1, "mega_program", meta["mega_program"].to_numpy())
    out.insert(2, "program", meta["program"].to_numpy())
    out.insert(3, "taxonomy", meta["taxonomy"].to_numpy())
    out["annotation_label"] = meta["annotation_label"].to_numpy()
    return out
