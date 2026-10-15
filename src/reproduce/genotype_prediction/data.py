"""Cohort loaders used by the genotype-prediction panels (see `reproduce.data`).

Focal-CNV and arm calls are returned as floats here, as the regression code expects.
"""

import pandas as pd

from reproduce import data as _data
from reproduce.data import (  # noqa: F401
    factor_annotation,
    load_arm_fractions,
    load_clinical,
    load_expression,
    load_factor_scores,
    load_he_features,
    load_loadings,
    load_mutations,
    load_treatment,
    retained_factors,
)


def load_focal_cnv() -> pd.DataFrame:
    return _data.load_focal_cnv().astype(float)


def load_arm_calls() -> pd.DataFrame:
    return _data.load_arm_calls().astype(float)
