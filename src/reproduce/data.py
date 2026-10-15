"""Loaders for the synthetic cohort tables and the factor reference, shared by all panels."""

import pandas as pd

from .data_paths import factor_reference, synthetic_dir

#: Non-gene columns of the factor reference table.
LOADINGS_META_COLS = (
    "factor",
    "embedding_column",
    "in_sample_embeddings",
    "in_factors_matrix",
    "is_dense_factor",
    "program",
    "annotation_label",
    "mega_program",
    "taxonomy",
)


def load_table(name: str) -> pd.DataFrame:
    """One cohort table, e.g. `load_table("mutations")`."""
    return pd.read_parquet(synthetic_dir() / f"{name}.parquet")


def load_taxonomy() -> pd.DataFrame:
    return load_table("taxonomy")


def load_mutations() -> pd.DataFrame:
    return load_table("mutations")


def load_focal_cnv() -> pd.DataFrame:
    return load_table("focal_cnv")


def load_arm_fractions() -> pd.DataFrame:
    return load_table("arm_fractions")


def load_arm_calls() -> pd.DataFrame:
    return load_table("arm_calls")


def load_he_features() -> pd.DataFrame:
    """Rows with simulated H&E availability only."""
    return load_table("he_features")


def load_factor_scores() -> pd.DataFrame:
    return load_table("factor_scores")


def load_expression() -> pd.DataFrame:
    return load_table("expression")


def load_clinical() -> pd.DataFrame:
    return load_table("clinical")


def load_survival() -> pd.DataFrame:
    return load_table("survival")


def load_treatment() -> pd.DataFrame:
    return load_table("treatment")


def load_loadings() -> pd.DataFrame:
    return pd.read_parquet(factor_reference())


def loading_gene_columns(loadings: pd.DataFrame) -> list[str]:
    return [c for c in loadings.columns if c not in LOADINGS_META_COLS]


def factor_annotation() -> pd.DataFrame:
    """Return factor, program, mega-program, taxonomy, and annotation labels.

    `program` is null for factors outside the 17 published programs (e.g. `dense_0`).
    """
    return load_loadings()[
        [
            "factor",
            "program",
            "mega_program",
            "taxonomy",
            "annotation_label",
            "is_dense_factor",
        ]
    ].copy()


def retained_factors() -> list[str]:
    """The 143 retained factors (of 144 in `factor_scores`): all except `dense_0`.

    `dense_0` is the one factor_scores column with no `program` assignment in the
    factor reference.
    """
    ann = factor_annotation()
    return ann.loc[ann["program"].notna(), "factor"].tolist()
