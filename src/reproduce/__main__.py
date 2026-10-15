"""Run all panels on the synthetic cohort: `python -m reproduce --data DIR --out DIR`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m reproduce",
        description="Write the source-data CSV of every panel, computed on the synthetic cohort.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="extracted synthetic_cohort_data folder (default: $SYNTHCRC_SYNTHETIC_DIR)",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="folder for the panel CSVs"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="only check existing CSVs in --out; do not recompute",
    )
    args = parser.parse_args(argv)
    if args.data is not None:
        os.environ["SYNTHCRC_SYNTHETIC_DIR"] = str(args.data.resolve())

    import reproduce

    failed = {}
    if not args.check_only:
        data = os.environ.get("SYNTHCRC_SYNTHETIC_DIR")
        if not data or not (Path(data) / "taxonomy.parquet").exists():
            parser.error(
                f"no synthetic cohort found at {data or '(SYNTHCRC_SYNTHETIC_DIR unset)'}; "
                "pass --data"
            )
        for name, reason in reproduce.UNAVAILABLE.items():
            print(f"skipped {name}: {reason}", file=sys.stderr)
        # Remove earlier CSVs of every panel first, so a failed or skipped panel cannot
        # pass the check below with a file left over from a previous run.
        for panel_id in reproduce.EXPECTED_COLUMNS:
            (args.out / f"{panel_id}.csv").unlink(missing_ok=True)
        for panel_id in reproduce.PANELS:
            try:
                reproduce.emit(panel_id, args.out)
            except Exception as exc:  # report every failing panel, then exit non-zero
                failed[panel_id] = f"{type(exc).__name__}: {exc}"
                print(f"FAILED {panel_id}: {failed[panel_id]}", file=sys.stderr)
            else:
                print(f"wrote {panel_id}")
    problems = reproduce.check_outputs(args.out)
    n_ok = len(reproduce.EXPECTED_COLUMNS) - len({p.split(":")[0] for p in problems})
    print(
        f"{n_ok} of {len(reproduce.EXPECTED_COLUMNS)} panels present with the expected columns"
    )
    for problem in problems:
        print("  " + problem, file=sys.stderr)
    if failed:
        print(f"{len(failed)} panel(s) failed: {', '.join(failed)}", file=sys.stderr)
    print("All values are synthetic and must not be reported as study results.")
    return 1 if problems or failed else 0


if __name__ == "__main__":
    sys.exit(main())
