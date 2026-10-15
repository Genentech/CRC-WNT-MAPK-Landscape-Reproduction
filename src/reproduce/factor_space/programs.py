"""Engine 1: factor -> program / mega_program aggregation.

Aggregates factor scores into program scores for the synthetic cohort, using its
factor reference table.

The manuscript reports 17 programs (P1-P17). Filtering the factor reference on
`program.notna()` recovers them, and `program_set()` checks the result against the shared
P1-P17 allowlist in `reproduce._program_guard`.
"""

import numpy as np
import pandas as pd

from .._program_guard import validate_program_set
from ..data import factor_annotation  # noqa: F401  (re-exported for pca.py)
from .data import load_factor_scores


def program_set() -> list[str]:
    """The 17 published programs, in the loadings sheet's row order."""
    ann = factor_annotation()
    programs = ann.loc[ann["program"].notna(), "program"].drop_duplicates().tolist()
    return validate_program_set(programs, source="factor_space loadings annotation")


def program_metadata() -> pd.DataFrame:
    """One row per program, indexed by program: mega_program, taxonomy, annotation_label, n_factors."""
    ann = factor_annotation()
    assigned = ann[ann["program"].notna()]
    n_factors = assigned.groupby("program").size().rename("n_factors")
    meta = assigned.drop_duplicates("program").set_index("program")
    meta = meta[["mega_program", "taxonomy", "annotation_label"]].join(n_factors)
    return meta.reindex(program_set())


def program_score_frame(weight: str = "variance") -> pd.DataFrame:
    """Per-sample program scores: (variance-)weighted mean of member factor scores.

    weight="variance" (the value used for the published Fig2b/Fig2d source data)
    weights each member factor by its sample variance; "uniform" is a plain mean.
    """
    if weight not in ("variance", "uniform"):
        raise ValueError(f"weight must be 'variance' or 'uniform' (got {weight!r})")

    scores = load_factor_scores()
    ann = factor_annotation()
    var_f = scores.var(axis=0)

    out = {}
    for prog, sub in ann[ann["program"].notna()].groupby("program"):
        members = [f for f in sub["factor"] if f in scores.columns]
        if not members:
            continue
        w = (
            var_f.reindex(members).to_numpy()
            if weight == "variance"
            else np.ones(len(members))
        )
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(len(members))
        out[prog] = np.average(scores[members].to_numpy(), axis=1, weights=w)

    return pd.DataFrame(out, index=scores.index).reindex(columns=program_set())
