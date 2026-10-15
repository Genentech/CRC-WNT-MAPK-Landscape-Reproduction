"""Single-sample GSEA (Barbie et al. 2009 rank-weighted running-sum statistic).

Reimplemented from the published algorithm, vectorized across samples. Used for the
7-gene WNT signature (AXIN2/ASCL2/LGR5/WIF1/NKD1/SMOC2/SLC12A2); all 7 genes are
present in the synthetic 3,074-gene expression panel, so there is no gene-set coverage
loss for this signature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..methods.gene_panels import WNT_GENES  # noqa: F401


def ssgsea_score(
    expression: pd.DataFrame, gene_set, *, alpha: float = 0.25
) -> pd.Series:
    genes_present = [g for g in gene_set if g in expression.columns]
    if not genes_present:
        raise ValueError(
            "none of the requested gene-set genes are in the expression panel"
        )

    ranks = expression.rank(axis=1, method="average").to_numpy()
    in_set = expression.columns.isin(genes_present).astype(float)

    order = np.argsort(ranks, axis=1)
    sorted_in_set = np.take_along_axis(
        np.broadcast_to(in_set, ranks.shape), order, axis=1
    )
    sorted_ranks = np.take_along_axis(ranks, order, axis=1)

    hit_weight = np.where(sorted_in_set > 0, sorted_ranks**alpha, 0.0)
    hit_cumsum = np.cumsum(hit_weight, axis=1)
    p_hit = hit_cumsum / hit_cumsum[:, -1:]

    miss_cumsum = np.cumsum(1.0 - sorted_in_set, axis=1)
    p_miss = miss_cumsum / miss_cumsum[:, -1:]

    scores = (p_hit - p_miss).sum(axis=1)
    return pd.Series(scores, index=expression.index, name="wnt_ssgsea")
