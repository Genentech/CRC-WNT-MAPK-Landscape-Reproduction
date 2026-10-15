"""Set up a workspace with the Zenodo synthetic cohort and the oncophenomap package.

    python setup_workspace.py --synthetic-zip ~/Downloads/synthetic_cohort_data.zip \
        --workspace crc_workspace

Creates

    <workspace>/data/synthetic_cohort_data/   extracted from the Zenodo ZIP
    <workspace>/oncophenomap/                 cloned from ONCOPHENOMAP_URL, or linked to --oncophenomap-path
    <workspace>/results/panels/               where the panel CSVs are written

and prints the commands that install the environment and run the analyses. Nothing is installed by
this script. Uses the Python standard library only.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

#: Public location of the oncophenomap repository.
ONCOPHENOMAP_URL = "https://github.com/Genentech/oncophenomap.git"

#: oncophenomap release tag to clone (empty clones the default branch).
ONCOPHENOMAP_REF = "v0.1.0"

#: Zenodo record holding `synthetic_cohort_data.zip` (all-versions DOI).
ZENODO_DOI = "10.5281/zenodo.23055294"

COHORT_DIR = "synthetic_cohort_data"
REQUIRED_TABLES = (
    "taxonomy",
    "mutations",
    "focal_cnv",
    "arm_fractions",
    "arm_calls",
    "he_features",
    "factor_scores",
    "expression",
    "clinical",
    "survival",
    "treatment",
    "factor_reference",
)


def extract_cohort(zip_path: Path, data_dir: Path) -> Path:
    """Extract the cohort folder from the Zenodo ZIP, refusing unsafe member paths."""
    target = data_dir / COHORT_DIR
    if target.exists():
        check_cohort(target)
        print(f"using existing {target}")
        return target
    with zipfile.ZipFile(zip_path) as archive:
        members = []
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit(f"refusing unsafe path in the ZIP: {info.filename}")
            if path.parts[0] == "__MACOSX" or path.name == ".DS_Store":
                continue
            if path.parts[0] != COHORT_DIR:
                raise SystemExit(
                    f"unexpected entry {info.filename}; is this synthetic_cohort_data.zip?"
                )
            members.append(info)
        data_dir.mkdir(parents=True, exist_ok=True)
        archive.extractall(data_dir, members=members)
    print(f"extracted {target}")
    return target


def check_cohort(cohort: Path) -> None:
    missing = [t for t in REQUIRED_TABLES if not (cohort / f"{t}.parquet").exists()]
    if missing:
        raise SystemExit(
            f"{cohort} is missing tables: {', '.join(missing)}. Delete the folder and run "
            "this script again."
        )


def get_oncophenomap(
    workspace: Path, url: str, local: Path | None, ref: str = ONCOPHENOMAP_REF
) -> Path:
    target = workspace / "oncophenomap"
    if target.is_symlink() and not target.exists():
        raise SystemExit(
            f"{target} is a broken link; remove it and run this script again"
        )
    if ref and (target.exists() or local is not None):
        print(
            f"note: --oncophenomap-ref {ref} is ignored for an existing or linked checkout"
        )
    if target.exists():
        print(f"using existing {target}")
    elif local is not None:
        if not (local / "oncophenomap").is_dir():
            raise SystemExit(f"{local} does not look like an oncophenomap checkout")
        target.symlink_to(local.resolve(), target_is_directory=True)
        print(f"linked {target} -> {local.resolve()}")
    else:
        if shutil.which("git") is None:
            raise SystemExit("git is required to clone oncophenomap")
        branch = ["--branch", ref] if ref else []
        subprocess.run(
            ["git", "clone", "--depth", "1", *branch, url, str(target)], check=True
        )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--synthetic-zip",
        type=Path,
        required=True,
        help=f"synthetic_cohort_data.zip downloaded from https://doi.org/{ZENODO_DOI}",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("crc_workspace"),
        help="folder to create (default: crc_workspace)",
    )
    onco = parser.add_mutually_exclusive_group()
    onco.add_argument(
        "--oncophenomap-url",
        default=ONCOPHENOMAP_URL,
        help="repository to clone oncophenomap from",
    )
    onco.add_argument(
        "--oncophenomap-path",
        type=Path,
        help="use an existing oncophenomap checkout instead of cloning",
    )
    parser.add_argument(
        "--oncophenomap-ref",
        default=ONCOPHENOMAP_REF,
        help="release tag to clone (default: the pinned release, if set)",
    )
    args = parser.parse_args(argv)

    if not args.synthetic_zip.is_file():
        raise SystemExit(f"{args.synthetic_zip} not found")

    workspace = args.workspace.resolve()
    cohort = extract_cohort(args.synthetic_zip, workspace / "data")
    check_cohort(cohort)
    onco = get_oncophenomap(
        workspace, args.oncophenomap_url, args.oncophenomap_path, args.oncophenomap_ref
    )
    results = workspace / "results" / "panels"
    results.mkdir(parents=True, exist_ok=True)

    repo = Path(__file__).resolve().parent
    q = shlex.quote
    print(
        "\nWorkspace ready. Next, from any directory:\n\n"
        f"  mamba env create -f {q(str(repo / 'environment.yml'))}\n"
        "  mamba activate crc-wnt-mapk-reproduction\n"
        f"  pip install -e {q(str(onco))} -e {q(str(repo))}\n"
        f"  python -m reproduce --data {q(str(cohort))} --out {q(str(results))}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
