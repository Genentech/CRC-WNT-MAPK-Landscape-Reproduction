"""Factor-score -> program aggregation (shared machinery, serves Fig4b/SFig8a/SFig8b).

Equivalent role to `reproduce.factor_space.programs`, but this subpackage keeps its
own aggregation logic rather than importing across sibling subpackages. Only the
P1-P17 allowlist check is shared, via `reproduce._program_guard`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._program_guard import validate_program_set


def program_metadata(loadings: pd.DataFrame) -> pd.DataFrame:
    """One row per program: program, program_concise, mega_program, taxonomy."""
    meta = loadings.dropna(subset=["program"])
    out = (
        meta.drop_duplicates("program")[
            ["program", "annotation_label", "mega_program", "taxonomy"]
        ]
        .rename(columns={"annotation_label": "program_concise"})
        .sort_values("program")
        .reset_index(drop=True)
    )
    validate_program_set(
        out["program"].tolist(), source="taxonomy_clinical loadings annotation"
    )
    return out


def _factor_to_program(loadings: pd.DataFrame, factor_columns) -> dict:
    meta = loadings.set_index("embedding_column")["program"]
    mapping = {}
    for col in factor_columns:
        prog = meta.get(col)
        if isinstance(prog, str):
            mapping[col] = prog
    return mapping


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std()
    if pd.isna(sd) or np.isclose(sd, 0):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / sd


def program_scores(
    factor_scores: pd.DataFrame, loadings: pd.DataFrame, *, z: bool = True
) -> pd.DataFrame:
    """Per-sample program score: mean of member-factor scores, optionally z-scored.

    `dense_0` has no `program` in the loadings reference and is excluded.
    """
    mapping = _factor_to_program(loadings, factor_scores.columns)
    programs = sorted(set(mapping.values()))
    validate_program_set(programs, source="taxonomy_clinical factor->program mapping")
    out = {}
    for program in programs:
        cols = [c for c, p in mapping.items() if p == program]
        out[program] = factor_scores[cols].mean(axis=1)
    df = pd.DataFrame(out, index=factor_scores.index)
    if z:
        df = df.apply(zscore, axis=0)
    return df
