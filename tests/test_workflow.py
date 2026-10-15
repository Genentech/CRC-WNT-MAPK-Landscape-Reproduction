"""Tests for the workspace setup, the run command, the output check and (with data) a full run.

The full run needs the extracted Zenodo synthetic cohort: set `SYNTHCRC_SYNTHETIC_DIR` to it and
install `oncophenomap`; otherwise that test is skipped. All other tests use no data.
"""

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import reproduce  # noqa: E402
from reproduce import __main__ as cli  # noqa: E402
from reproduce._program_guard import validate_program_set  # noqa: E402
from reproduce.data_paths import synthetic_dir  # noqa: E402
from reproduce.tree_settings import TREE_SETTINGS  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "setup_workspace", REPO / "setup_workspace.py"
)
setup_workspace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_workspace)

UNUSED_URL = "https://example.invalid/oncophenomap.git"  # never cloned in these tests
COHORT = setup_workspace.COHORT_DIR


def _zip(path, names):
    with zipfile.ZipFile(path, "w") as z:
        for name in names:
            z.writestr(name, b"x")


def _quiet(fn, *args):
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        return fn(*args)


def _write_expected_outputs(out: Path) -> None:
    """One CSV per panel with the expected header and the synthetic marker."""
    for panel_id, columns in reproduce.EXPECTED_COLUMNS.items():
        df = pd.DataFrame([[0] * len(columns)], columns=columns)
        df[reproduce.SYNTHETIC_MARKER_COLUMN] = reproduce.SYNTHETIC_MARKER_VALUE
        df.to_csv(out / f"{panel_id}.csv", index=False)


class SetupWorkspaceTest(unittest.TestCase):
    def test_extracts_cohort_and_skips_macos_clutter(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            names = [f"{COHORT}/{t}.parquet" for t in setup_workspace.REQUIRED_TABLES]
            _zip(tmp / "c.zip", names + ["__MACOSX/._x", f"{COHORT}/.DS_Store"])
            cohort = _quiet(setup_workspace.extract_cohort, tmp / "c.zip", tmp / "data")
            self.assertFalse((tmp / "data" / "__MACOSX").exists())
            self.assertFalse((cohort / ".DS_Store").exists())
            # A second run reuses the complete folder.
            self.assertEqual(
                _quiet(setup_workspace.extract_cohort, tmp / "c.zip", tmp / "data"),
                cohort,
            )

    def test_refuses_unsafe_or_foreign_members(self):
        for bad in ("../evil.parquet", "other_folder/taxonomy.parquet"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                _zip(tmp / "c.zip", [bad])
                with self.assertRaises(SystemExit):
                    setup_workspace.extract_cohort(tmp / "c.zip", tmp / "data")

    def test_refuses_partial_existing_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "data" / COHORT).mkdir(parents=True)
            (tmp / "data" / COHORT / "taxonomy.parquet").write_bytes(b"x")
            with self.assertRaises(SystemExit):
                setup_workspace.extract_cohort(tmp / "c.zip", tmp / "data")

    def test_local_checkout_is_linked_and_ref_note_printed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "checkout" / "oncophenomap").mkdir(parents=True)
            (tmp / "ws").mkdir()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                target = setup_workspace.get_oncophenomap(
                    tmp / "ws", UNUSED_URL, tmp / "checkout", ref="v1.0.0"
                )
            self.assertTrue(target.is_symlink())
            self.assertIn("is ignored", out.getvalue())

    def test_broken_link_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "oncophenomap").symlink_to(tmp / "missing", target_is_directory=True)
            with self.assertRaises(SystemExit):
                setup_workspace.get_oncophenomap(tmp, UNUSED_URL, tmp)

    def test_url_and_path_are_exclusive(self):
        with self.assertRaises(SystemExit):
            _quiet(
                setup_workspace.main,
                [
                    "--synthetic-zip",
                    "x.zip",
                    "--oncophenomap-url",
                    "u",
                    "--oncophenomap-path",
                    "p",
                ],
            )


class OutputCheckTest(unittest.TestCase):
    def test_complete_outputs_pass_and_problems_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_expected_outputs(out)
            self.assertEqual(reproduce.check_outputs(out), [])
            self.assertEqual(_quiet(cli.main, ["--out", str(out), "--check-only"]), 0)

            (out / "Fig2f.csv").unlink()
            df = pd.read_csv(out / "Fig2b.csv").iloc[:, 1:]
            df.to_csv(out / "Fig2b.csv", index=False)
            problems = reproduce.check_outputs(out)
            self.assertEqual(len(problems), 2)
            self.assertEqual(_quiet(cli.main, ["--out", str(out), "--check-only"]), 1)

    def test_stale_csvs_cannot_pass_a_new_run(self):
        """A panel that is not recomputed (e.g. oncophenomap missing) must not pass on an old file."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out, data = tmp / "out", tmp / "cohort"
            out.mkdir()
            data.mkdir()
            (data / "taxonomy.parquet").write_bytes(b"x")
            _write_expected_outputs(out)
            with mock.patch.object(reproduce, "PANELS", {}):
                rc = _quiet(cli.main, ["--data", str(data), "--out", str(out)])
            self.assertEqual(rc, 1)
            self.assertFalse(any(out.glob("*.csv")))

    def test_fig4b_tree_columns_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_expected_outputs(out)
            df = pd.read_csv(out / "Fig4b.csv")
            df["L2#12"] = 0.0
            df.to_csv(out / "Fig4b.csv", index=False)
            self.assertEqual(reproduce.check_outputs(out), [])

    def test_missing_cohort_is_an_error(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SYNTHCRC_SYNTHETIC_DIR", None)
            with self.assertRaises(RuntimeError):
                synthetic_dir()
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit):
                _quiet(cli.main, ["--out", tmp])


class PredictorNamesTest(unittest.TestCase):
    """Mutation and focal-CNV tables share gene names; combined predictors must stay unique."""

    def test_combined_predictors_have_unique_names(self):
        from reproduce.genotype_prediction import feature_ladder, panels
        from reproduce.methods.feature_ladder import FEATURE_SPACES, space_matrix

        idx = pd.Index([f"syn_{i:06d}" for i in range(1, 7)], name="sample_id")
        tables = {
            "mutations": pd.DataFrame(
                {"APC": [0, 1, 0, 1, 1, 0], "KRAS": [1, 0, 1, 0, 0, 1]}, index=idx
            ),
            "focal_cnv": pd.DataFrame(
                {"APC": [0, 2, 0, -2, 0, 1], "MYC": [1, 0, 2, 0, 0, 0]}, index=idx
            ),
            "arm_fractions": pd.DataFrame(
                {"arm_1p_gain_frac": [0.1, 0.9, 0.2, 0.8, 0.5, 0.3]}, index=idx
            ),
            "he_features": pd.DataFrame(
                {"heFibroblasts": [0.2, 0.4, 0.1, 0.3, 0.6, 0.5]}, index=idx
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, df in tables.items():
                df.to_parquet(Path(tmp) / f"{name}.parquet")
            with mock.patch.dict(os.environ, {"SYNTHCRC_SYNTHETIC_DIR": tmp}):
                X = panels._combined_predictor_matrix()
                self.assertFalse(X.columns.duplicated().any())
                self.assertIn("APC_cnv", X.columns)
                blocks = feature_ladder._build_blocks()
                for space in FEATURE_SPACES:
                    with self.subTest(space=space):
                        Xs = space_matrix(blocks, space)
                        self.assertFalse(Xs.columns.duplicated().any())

    def test_duplicate_names_are_rejected(self):
        from reproduce.methods.feature_ladder import space_matrix

        a = pd.DataFrame({"APC": [0, 1, 0, 1]})
        blocks = {
            "mutation": a,
            "cnv_focal": a.copy(),
            "cnv_arm": a.copy(),
            "he": a.copy(),
        }
        with self.assertRaises(ValueError):
            space_matrix(blocks, "mut_cnv")


class SettingsTest(unittest.TestCase):
    def test_tree_settings(self):
        self.assertEqual(
            (
                TREE_SETTINGS.l1_arm_leaf_cap,
                TREE_SETTINGS.l1_he_leaf_cap,
                TREE_SETTINGS.l1_min_samples_leaf,
                TREE_SETTINGS.l2_max_leaf_nodes,
                TREE_SETTINGS.l2_min_samples_leaf,
                TREE_SETTINGS.random_state,
            ),
            (8, 2, 500, 150, 50, 42),
        )

    def test_program_guard(self):
        programs = [f"P{i}_x" for i in range(1, 18)]
        self.assertEqual(validate_program_set(programs), programs)
        for bad in (
            programs + ["P0_x"],
            programs[:-1] + ["P99_x"],
            programs[:-1],
            programs + ["P3_y"],
            programs[:-1] + ["Q17_x"],
        ):
            with self.subTest(bad=bad[-1]), self.assertRaises(ValueError):
                validate_program_set(bad)


@unittest.skipUnless(
    os.environ.get("SYNTHCRC_SYNTHETIC_DIR")
    and importlib.util.find_spec("oncophenomap"),
    "set SYNTHCRC_SYNTHETIC_DIR to the extracted cohort and install oncophenomap",
)
class FullRunTest(unittest.TestCase):
    def test_all_panels(self):
        with tempfile.TemporaryDirectory() as out:
            reproduce.emit_all(Path(out))
            self.assertEqual(reproduce.check_outputs(Path(out)), [])
            self.assertEqual(len(reproduce.PANELS), 29)


if __name__ == "__main__":
    unittest.main()
