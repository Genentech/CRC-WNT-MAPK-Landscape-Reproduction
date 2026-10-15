"""Call-time paths for the synthetic cohort."""

from __future__ import annotations

import os
from pathlib import Path


def synthetic_dir() -> Path:
    """The extracted `synthetic_cohort_data` folder named by `SYNTHCRC_SYNTHETIC_DIR`."""
    value = os.environ.get("SYNTHCRC_SYNTHETIC_DIR")
    if not value:
        raise RuntimeError(
            "SYNTHCRC_SYNTHETIC_DIR is not set; set it to the extracted "
            "synthetic_cohort_data folder (or use `python -m reproduce --data DIR`)"
        )
    return Path(value)


def factor_reference() -> Path:
    value = os.environ.get("SYNTHCRC_FACTOR_REFERENCE")
    return Path(value) if value else synthetic_dir() / "factor_reference.parquet"
