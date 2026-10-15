"""Engine 3: per-factor reconstruction R^2.

For a linear factor model X ~ Z W^T (Z: sample x factor scores, W: factor x gene
loadings), the variance of expression reconstructed by factor f alone is
Var(z_f) * ||w_f||^2. Normalizing by the total variance of the `expression`
block gives a reconstruction R^2 per factor. Serves SFig3a (top factors) and
SFig3b (cumulative-variance curve).

Scope caveat: this is computed over the 144 factors present in `factor_scores`
(143 retained + dense_0), not the published 270-factor pre-filter space -- the
synthetic cohort does not include the other 126 factors. `total_factor_r2` and any
rank built from it are therefore internally consistent but not numerically
comparable to the published constant (0.886...).
"""

import numpy as np
import pandas as pd

from .data import (
    load_expression,
    load_factor_scores,
    load_loadings,
    loading_gene_columns,
)


def reconstruction_table() -> pd.DataFrame:
    """One row per available factor (144), ranked by reconstruction_r2 descending."""
    scores = load_factor_scores()
    loadings = load_loadings().set_index("factor")
    gene_cols = loading_gene_columns(load_loadings())

    factors = [f for f in scores.columns if f in loadings.index]
    var_f = scores[factors].var(axis=0)
    sq_norm = (loadings.loc[factors, gene_cols].to_numpy() ** 2).sum(axis=1)
    recon_var = var_f.to_numpy() * sq_norm

    total_gene_var = float(load_expression().var(axis=0).sum())
    reconstruction_r2 = recon_var / total_gene_var
    total_factor_r2 = float(reconstruction_r2.sum())
    r2_share_pct = reconstruction_r2 / total_factor_r2 * 100

    out = (
        pd.DataFrame(
            {
                "factor": factors,
                "reconstruction_r2": reconstruction_r2,
                "r2_share_pct": r2_share_pct,
                "total_factor_r2": total_factor_r2,
            }
        )
        .sort_values("reconstruction_r2", ascending=False)
        .reset_index(drop=True)
    )
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out
