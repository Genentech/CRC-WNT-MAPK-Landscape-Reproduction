"""Panel emitters: regenerate each published figure's source-data CSV from the synthetic cohort.

`PANELS[panel_id]()` returns the panel's DataFrame; `emit_all(out_dir)` writes them as CSVs.
"""

import json
import warnings
from pathlib import Path

import pandas as pd

from .factor_space import PANELS as _FACTOR_SPACE
from .factor_space import PROVENANCE as _FACTOR_SPACE_PROV
from .genotype_prediction import PANELS as _GENOTYPE
from .genotype_prediction import PROVENANCE as _GENOTYPE_PROV

FAMILIES = {
    "factor_space": _FACTOR_SPACE,
    "genotype_prediction": _GENOTYPE,
}

# taxonomy_clinical needs `oncophenomap`, which is a separate package. Import it
# tolerantly so a reader without it still gets the other 14 panels rather than nothing.
UNAVAILABLE: dict[str, str] = {}
try:
    from .taxonomy_clinical import PANELS as _TAXONOMY
    from .taxonomy_clinical import PROVENANCE as _TAXONOMY_PROV

    FAMILIES["taxonomy_clinical"] = _TAXONOMY
except ImportError as exc:  # pragma: no cover - depends on the reader's environment
    _TAXONOMY, _TAXONOMY_PROV = {}, {}
    UNAVAILABLE["taxonomy_clinical"] = str(exc)
    warnings.warn(UNAVAILABLE["taxonomy_clinical"], stacklevel=2)

PANELS: dict[str, callable] = {**_FACTOR_SPACE, **_GENOTYPE, **_TAXONOMY}
PROVENANCE: dict[str, str] = {**_FACTOR_SPACE_PROV, **_GENOTYPE_PROV, **_TAXONOMY_PROV}

_collisions = sum(len(d) for d in FAMILIES.values()) - len(PANELS)
if _collisions:
    raise RuntimeError(f"{_collisions} panel id(s) declared by more than one family")

#: Appended to every emitted panel. These files carry figure-panel column names and
#: plausible-looking statistics, so a CSV separated from this repository could be
#: mistaken for real cohort results. The marker travels INSIDE the file, because a
#: sidecar or a directory README does not survive someone emailing one CSV.
SYNTHETIC_MARKER_COLUMN = "data_origin"
SYNTHETIC_MARKER_VALUE = "synthetic"

_PANELS_README = """\
Every CSV in this directory is SYNTHETIC.

The values are produced from a synthetic cohort and describe simulated tumours
with `syn_`-prefixed identifiers. They are NOT study-cohort results,
and no row describes a real patient. Column names and shapes support demonstration of
the analysis code; the numbers must not be read as findings or quoted as such.

Each file carries a `{col}` column set to `{val}` for the same reason.
""".format(col=SYNTHETIC_MARKER_COLUMN, val=SYNTHETIC_MARKER_VALUE)


def write_panel(df: pd.DataFrame, panel_id: str, out_dir: Path) -> Path:
    """Write one already-computed panel, stamping the synthetic marker.

    The single place any panel CSV is written. Callers that already hold a frame
    (e.g. tests) call this rather than writing it themselves -- otherwise such
    panels would silently lack the marker that `emit()` adds.
    """
    if SYNTHETIC_MARKER_COLUMN in df.columns:
        raise RuntimeError(
            f"panel {panel_id} already defines a {SYNTHETIC_MARKER_COLUMN!r} column; "
            f"rename it so the synthetic marker is unambiguous"
        )
    df = df.copy()
    df[SYNTHETIC_MARKER_COLUMN] = SYNTHETIC_MARKER_VALUE
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Written here rather than only in `emit_all`, so a directory filled
    # panel-by-panel still carries the notice.
    readme = out_dir / "READ_ME_SYNTHETIC.txt"
    if not readme.exists():
        readme.write_text(_PANELS_README)
    path = out_dir / f"{panel_id}.csv"
    df.to_csv(path, index=False)
    return path


def emit(panel_id: str, out_dir: Path) -> Path:
    return write_panel(PANELS[panel_id](), panel_id, out_dir)


def emit_all(out_dir: Path) -> dict[str, Path]:
    return {p: emit(p, out_dir) for p in PANELS}


#: Column headers of each panel, for checking a run. Fig4b's per-group `L2#<leaf>`
#: columns depend on the fitted tree, so only its fixed leading columns are listed.
EXPECTED_COLUMNS: dict[str, list[str]] = json.loads(
    (Path(__file__).with_name("expected_columns.json")).read_text()
)


def check_outputs(out_dir: Path) -> list[str]:
    """Return a list of problems with the panel CSVs in `out_dir` (empty when all is well)."""
    problems = []
    for panel_id, expected in EXPECTED_COLUMNS.items():
        path = Path(out_dir) / f"{panel_id}.csv"
        if not path.exists():
            problems.append(f"{panel_id}: missing")
            continue
        df = pd.read_csv(path, nrows=5)
        if (
            SYNTHETIC_MARKER_COLUMN not in df.columns
            or not (df[SYNTHETIC_MARKER_COLUMN] == SYNTHETIC_MARKER_VALUE).all()
        ):
            problems.append(f"{panel_id}: synthetic marker column missing or wrong")
        columns = [c for c in df.columns if c != SYNTHETIC_MARKER_COLUMN]
        if panel_id == "Fig4b":
            columns = [c for c in columns if not c.startswith("L2#")]
        if columns != expected:
            problems.append(f"{panel_id}: columns differ from the expected header")
    return problems


def family_of(panel_id: str) -> str:
    return next(name for name, d in FAMILIES.items() if panel_id in d)


def summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"panel_id": p, "family": family_of(p), "provenance": PROVENANCE.get(p, "")}
            for p in sorted(PANELS)
        ]
    )


__all__ = [
    "PANELS",
    "PROVENANCE",
    "FAMILIES",
    "emit",
    "emit_all",
    "write_panel",
    "check_outputs",
    "EXPECTED_COLUMNS",
    "family_of",
    "summary",
]
