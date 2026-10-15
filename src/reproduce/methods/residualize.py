"""Ordinary-least-squares residualization of a score against covariates (`ols_residualize`)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ols_residualize(
    series: pd.Series,
    metadata: pd.DataFrame,
    covariates: list[str] | tuple[str, ...],
) -> pd.Series:
    """Post-hoc OLS residualisation of a scalar score against covariates.

    Regresses ``series`` on ``covariates`` (numeric as-is; categorical/object
    one-hot with ``drop_first=True`` plus an explicit intercept) and returns the
    residuals, centred at zero. Used to build the sample-level ``wnt_ssgsea_resid``
    column (the ssGSEA score itself residualised against Purity + biopsy). The
    residualisation is applied post-hoc to the aggregated score, not per gene.

    Rows of ``series`` that are NaN, or whose covariates contain any NaN, are
    dropped from the fit and returned as NaN (the result is reindexed to
    ``series.index``).

    Parameters
    ----------
    series : pd.Series
        Scalar score per sample (e.g. raw ssGSEA), indexed by sample id.
    metadata : pd.DataFrame
        Per-sample metadata indexed like ``series``.
    covariates : Sequence[str]
        Metadata columns to residualise against (e.g. ``["Purity", "biopsy"]``).

    Returns
    -------
    pd.Series
        Residuals (mean ~0), reindexed to ``series.index``; NaN where a sample
        was dropped for missing score or covariates.
    """
    cov = metadata.reindex(series.index)[list(covariates)]
    keep = series.notna() & cov.notna().all(axis=1)
    idx = series.index[keep]
    if len(idx) == 0:
        raise ValueError("No rows with a non-NaN score and complete covariates")
    design = pd.get_dummies(cov.loc[idx], drop_first=True, dummy_na=False).astype(float)
    design.insert(0, "_intercept", 1.0)
    X = design.to_numpy()
    y = series.loc[idx].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return pd.Series(resid, index=idx).reindex(series.index)


__all__ = ["ols_residualize"]
