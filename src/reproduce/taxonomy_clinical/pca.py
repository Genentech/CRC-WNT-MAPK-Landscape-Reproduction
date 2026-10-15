"""Sample-level factor PCA (shared machinery, serves Fig4d/SFig10g/SFig10h).

Built once and reused by every panel that needs it. `dense_0` is excluded
(constant/placeholder factor, no `program` membership; see tree_engine.py).
"""

from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def factor_pca(factor_scores: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    cols = [c for c in factor_scores.columns if c != "dense_0"]
    x = StandardScaler().fit_transform(factor_scores[cols].to_numpy())
    pcs = PCA(n_components=n_components, random_state=RANDOM_STATE).fit_transform(x)
    columns = [f"PC{i + 1}" for i in range(n_components)]
    return pd.DataFrame(pcs, index=factor_scores.index, columns=columns)
