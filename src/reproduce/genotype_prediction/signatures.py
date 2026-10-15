"""Engine 1: WNT / MAPK ssGSEA signature scoring.

Gene panels, `RESID_COVARIATES` and `ols_residualize` come from `reproduce.methods`.
`_ssgsea` below calls `gp.ssgsea(...)` on the expression DataFrame (full-panel
background ranking, sample_norm_method="rank", min_size=1). All 10 genes of the
default `erk_activity_10` MAPK panel and all 7 WNT genes are in the expression table.
"""

from __future__ import annotations

import pandas as pd

from ..methods.gene_panels import (
    DEFAULT_MAPK_GENESET,
    MAPK_GENESETS,
    RESID_COVARIATES,
    WNT_GENES,
)
from ..methods.residualize import ols_residualize


def _ssgsea(
    expr: pd.DataFrame,
    genes: list[str],
    *,
    sample_norm_method: str = "rank",
    min_size: int = 1,
    threads: int = 4,
) -> pd.Series:
    """Rank-based single-sample ssGSEA (gseapy) for one gene set over a DataFrame.

    `expr` is samples x genes. The full gene panel is used as the ranking
    background (ssGSEA ranks each sample's whole transcriptome); only `genes`
    found in `expr.columns` form the scored set.
    """
    import gseapy as gp

    found = [g for g in genes if g in expr.columns]
    if not found:
        raise ValueError(f"none of the requested genes are in expr.columns: {genes}")
    data = expr.T  # genes x samples, full background for ranking
    res = gp.ssgsea(
        data=data,
        gene_sets={"SIG": found},
        sample_norm_method=sample_norm_method,
        min_size=min_size,
        outdir=None,
        no_plot=True,
        threads=threads,
    )
    nes = res.res2d.copy()
    nes["NES"] = pd.to_numeric(nes["NES"], errors="coerce")
    s = nes.set_index("Name")["NES"]
    s.index.name = None
    return s.reindex(expr.index)


def wnt_ssgsea(expr: pd.DataFrame) -> pd.Series:
    """Raw `wnt_ssgsea` score -- fixed 7-gene panel, all 7 present in the synthetic panel."""
    return _ssgsea(expr, list(WNT_GENES)).rename("wnt_ssgsea")


def mapk_ssgsea(
    expr: pd.DataFrame, *, geneset: str = DEFAULT_MAPK_GENESET
) -> pd.Series:
    """Raw `mapk_ssgsea` score on the `erk_activity_10` panel."""
    return _ssgsea(expr, list(MAPK_GENESETS[geneset])).rename("mapk_ssgsea")


def residualization_metadata(clinical: pd.DataFrame) -> pd.DataFrame:
    """Build the `RESID_COVARIATES` frame (`Purity`, `biopsy`) from the synthetic `clinical` block.

    Column-name mapping: synthetic `purity` -> `Purity`, `biopsy_site` -> `biopsy`
    (the two names `RESID_COVARIATES` expects).
    """
    assert RESID_COVARIATES == ("Purity", "biopsy")
    return pd.DataFrame(
        {"Purity": clinical["purity"], "biopsy": clinical["biopsy_site"]}
    )


def wnt_ssgsea_resid(expr: pd.DataFrame, clinical: pd.DataFrame) -> pd.Series:
    """`wnt_ssgsea_resid` -- raw score OLS-residualised on Purity + biopsy."""
    raw = wnt_ssgsea(expr)
    return ols_residualize(
        raw, residualization_metadata(clinical), RESID_COVARIATES
    ).rename("wnt_ssgsea_resid")


def mapk_ssgsea_resid(
    expr: pd.DataFrame, clinical: pd.DataFrame, *, geneset: str = DEFAULT_MAPK_GENESET
) -> pd.Series:
    """`mapk_ssgsea_resid` -- raw score OLS-residualised on Purity + biopsy."""
    raw = mapk_ssgsea(expr, geneset=geneset)
    return ols_residualize(
        raw, residualization_metadata(clinical), RESID_COVARIATES
    ).rename("mapk_ssgsea_resid")
