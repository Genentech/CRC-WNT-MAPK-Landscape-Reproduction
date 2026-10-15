"""Megaclusters of tree leaves: Ward clustering of z-scored per-leaf mean factor scores.

`select_factors` picks one representative per group of highly correlated factors and keeps
the most variable across leaves; `compute_megaclusters` clusters the leaves on those factors
(Ward linkage with optimal leaf ordering), cuts into `N_MEGA` clusters and numbers them
MC1.. from left to right in the dendrogram, so every leaf belongs to exactly one
megacluster. Adapted from the study's analysis code; the settings are the study's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import (
    fcluster,
    leaves_list,
    linkage,
    optimal_leaf_ordering,
)
from scipy.spatial.distance import squareform

#: Megacluster settings used in the study.
N_MEGA = 9
N_TOP_FACTORS = 50
FACTOR_CORR_THRESHOLD = 0.8


def select_factors(
    labels: pd.Series,
    Y: pd.DataFrame,
    *,
    corr_threshold: float = FACTOR_CORR_THRESHOLD,
    n_top: int = N_TOP_FACTORS,
) -> pd.Index:
    """Representative factors for megacluster formation.

    Factors are grouped by correlation (average linkage on 1 - |r|, cut at
    1 - `corr_threshold`); each group is represented by its factor with the largest
    across-leaf variance, and the `n_top` representatives with the largest across-leaf
    variance are kept.
    """
    group_profiles = Y.groupby(labels).mean()
    fac_dist = 1 - Y.corr(method="pearson").abs()
    np.fill_diagonal(fac_dist.values, 0)
    fac_dist.values[:] = np.clip(fac_dist.values, 0, None)
    clusters = fcluster(
        linkage(squareform(fac_dist, checks=False), method="average"),
        t=1 - corr_threshold,
        criterion="distance",
    )
    cmap = pd.Series(clusters, index=Y.columns)
    gvar = group_profiles.var(axis=0)
    reps = [gvar[cmap[cmap == cid].index].idxmax() for cid in np.unique(clusters)]
    rep_var = group_profiles[reps].var().sort_values(ascending=False)
    return pd.Index(rep_var.head(min(n_top, len(reps))).index)


def compute_megaclusters(
    labels: pd.Series, Y: pd.DataFrame, *, n_mega: int = N_MEGA
) -> dict:
    """Ward + optimal-leaf-ordering megaclusters of the leaves.

    Clusters the z-scored per-leaf mean of `Y` over `select_factors`, cuts the tree into
    `n_mega` clusters and numbers them MC1.. from left to right in the dendrogram.
    Returns {leaf label -> "MC<k>"}.
    """
    top_factors = select_factors(labels, Y)
    profiles = Y.groupby(labels).mean()
    pz = (profiles - profiles.mean()) / profiles.std().replace(0, 1)
    data = pz[top_factors].values
    Z = optimal_leaf_ordering(linkage(data, method="ward"), data)
    mega_raw = fcluster(Z, n_mega, criterion="maxclust")
    old_to_new: dict = {}
    for mc in (mega_raw[i] for i in leaves_list(Z)):
        old_to_new.setdefault(mc, len(old_to_new) + 1)
    return {
        profiles.index[i]: f"MC{old_to_new[mega_raw[i]]}" for i in range(len(profiles))
    }
