"""Engine 2: sample-level factor PCA.

PCA of the factor-score matrix (samples x factors), restricted to the 143
published-retained factors (`dense_0` carries no program assignment and is not
part of the manuscript's "filtered" factor space). PCs[f, k] is factor f's
loading on PC k, as unit-norm eigenvectors -- i.e. plain (mean-centered,
unscaled) PCA `components_.T`.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .data import load_factor_scores
from .programs import factor_annotation, program_set


def retained_factor_names() -> list[str]:
    ann = factor_annotation()
    return ann.loc[ann["program"].notna(), "factor"].tolist()


def fit(n_components: int = 8):
    scores = load_factor_scores()
    factors = [f for f in retained_factor_names() if f in scores.columns]
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(scores[factors].to_numpy())
    return pca, factors


def factor_loadings(n_components: int = 8) -> pd.DataFrame:
    """factor x PC unit-norm loadings."""
    pca, factors = fit(n_components)
    cols = [f"PC{i + 1}" for i in range(pca.n_components_)]
    return pd.DataFrame(pca.components_.T, index=factors, columns=cols)


def program_aggregate_loadings(
    n_pcs: int = 2, weight: str = "variance"
) -> pd.DataFrame:
    """One row per program: (variance-)weighted mean of member factors' PCA loadings.

    Column layout follows the published Fig2d source data, including its quirk
    of duplicating the program id into both `program` and `mega_program` while
    the biological family lives in `group` (row 0: program == mega_program ==
    "P10_Myeloid inflammation", group == "Inflamed secretory").
    """
    if weight not in ("variance", "uniform"):
        raise ValueError(f"weight must be 'variance' or 'uniform' (got {weight!r})")

    scores = load_factor_scores()
    loadings = factor_loadings(n_pcs)
    var_f = scores.var(axis=0)
    ann = factor_annotation().set_index("factor")
    pc_cols = list(loadings.columns)

    rows = []
    for prog in program_set():
        sub = ann[ann["program"] == prog]
        members = [f for f in sub.index if f in loadings.index]
        if not members:
            continue
        w = (
            var_f.reindex(members).to_numpy()
            if weight == "variance"
            else np.ones(len(members))
        )
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(len(members))
        agg = np.average(loadings.loc[members, pc_cols].to_numpy(), axis=0, weights=w)
        meta = sub.iloc[0]
        rows.append(
            {
                "program": prog,
                "mega_program": prog,
                "group": meta["mega_program"],
                "taxonomy": meta["taxonomy"],
                "annotation_label": meta["annotation_label"],
                "n_factors": len(members),
                **dict(zip(pc_cols, agg)),
            }
        )
    return pd.DataFrame(rows)
