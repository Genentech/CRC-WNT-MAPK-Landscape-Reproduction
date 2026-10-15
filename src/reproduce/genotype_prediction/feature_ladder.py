"""Engine 4: greedy feature ladder. Serves Fig3d / Fig3e / Fig3f / Fig3g.

Uses `forward_stepwise`, `run_ladders`, `space_matrix` and `FEATURE_SPACES` from
`reproduce.methods.feature_ladder`. `_build_blocks()` below supplies the four named blocks
(`mutation`, `cnv_focal`, `cnv_arm`, `he`) from the cohort tables.

Target: the RAW `wnt_ssgsea` score, NOT a program-level score and NOT the residualised
variant.

Cohort: treatment-naive primary samples, one per patient. The synthetic cohort has no
repeated patients, so the cohort is
`clinical.biopsy_site == "primary" & ~treatment.treated`. Every table covers all 5,000
samples except `he_features` (3,892), which narrows the `he` space accordingly.

Hyperparameters: `cv_folds=5, l1_ratio=0.5, random_state=42, max_features=25,
n_candidates=25`.

On the ElasticNet pre-filter: `forward_stepwise` shortlists candidates with
`coefs[coefs.abs() > 1e-12].sort_values().head(n_candidates)`. That is TWO filters --
a sparsity filter and a top-`n_candidates` cut -- and only the second is about
runtime. For the 12-feature `he` space the `head(25)` is a no-op but the sparsity
filter is not, so the ladder has fewer steps than features whenever ElasticNet zeroes
some H&E features. All four spaces run the same function unmodified.

The number of H&E ladder steps is a property of the synthetic cohort, not a quantity
to reproduce from the study. It records how many H&E coefficients ElasticNet leaves
non-zero and should be interpreted only as a workflow demonstration.
"""

from __future__ import annotations

import pandas as pd

from ..methods import feature_ladder as _fl
from .data import (
    load_arm_fractions,
    load_clinical,
    load_expression,
    load_focal_cnv,
    load_he_features,
    load_mutations,
    load_treatment,
)
from .signatures import wnt_ssgsea

LADDER_KW = dict(
    cv_folds=5,
    l1_ratio=0.5,
    random_state=42,
    max_features=25,
    n_candidates=25,
    verbose=False,
)


def _build_blocks() -> dict[str, pd.DataFrame]:
    # Focal-CNV columns share gene names with mutation columns (e.g. APC, KRAS); the
    # suffix keeps every predictor name unique once the blocks are concatenated.
    return {
        "mutation": load_mutations(),
        "cnv_focal": load_focal_cnv().add_suffix("_cnv"),
        "cnv_arm": load_arm_fractions(),
        "he": load_he_features(),
    }


def tx_naive_primary_cohort() -> pd.Index:
    clinical = load_clinical()
    treatment = load_treatment()
    mask = (clinical["biopsy_site"] == "primary") & (~treatment["treated"].astype(bool))
    return clinical.index[mask]


def wnt_ladders(
    spaces: tuple[str, ...] = ("combined", "arm", "mut_cnv", "he"),
) -> pd.DataFrame:
    """Run the WNT ssGSEA ladder over all requested `FEATURE_SPACES`; one tidy frame."""
    blocks = _build_blocks()
    cohort_idx = tx_naive_primary_cohort()
    wnt = wnt_ssgsea(load_expression())

    out = []
    for space in spaces:
        X = _fl.space_matrix(blocks, space)
        res = _fl.forward_stepwise(
            wnt.reindex(cohort_idx), X.reindex(cohort_idx).dropna(), **LADDER_KW
        )
        df = res["seq_df"].copy()
        df.insert(0, "space", space)
        df.insert(0, "target", "WNT")
        df["n_samples"] = res["n_samples"]
        df["n_candidates"] = res["n_candidates"]
        out.append(df)
    return pd.concat(out, ignore_index=True)
