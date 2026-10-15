"""Greedy forward-selection feature ladder.

`FEATURE_SPACES` names the predictor spaces (mutations + focal CNV, chromosome arms, H&E,
combined), `space_matrix` builds the predictor matrix for one space, `forward_stepwise`
adds one feature at a time by cross-validated R^2, and `run_ladders` runs every space for
one or more targets.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping, Sequence

import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "FEATURE_SPACES",
    "space_matrix",
    "forward_stepwise",
    "run_ladders",
]

#: space key -> (display label, member block names)
FEATURE_SPACES: dict[str, tuple[str, tuple[str, ...]]] = {
    "mut_cnv": ("Mutations + focal CNV", ("mutation", "cnv_focal")),
    "arm": ("Chromosome arms", ("cnv_arm",)),
    "he": ("H&E composition", ("he",)),
    "combined": ("Combined", ("mutation", "cnv_focal", "cnv_arm", "he")),
}

LADDER_COLUMNS = [
    "target",
    "space",
    "step",
    "feature_added",
    "cv_r2_mean",
    "cv_r2_std",
    "delta_r2",
    "n_samples",
    "n_candidates",
]


def _safe_n_jobs(desired: int | None = None, reserve: int = 1) -> int:
    """joblib ``n_jobs`` sized from the affinity mask, never claiming every core."""
    try:
        avail = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        avail = os.cpu_count() or 1
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        avail = min(avail, int(slurm))
    cap = max(1, avail - reserve)
    return cap if desired is None else max(1, min(desired, cap))


def space_matrix(
    blocks: Mapping[str, pd.DataFrame], space: str, index: pd.Index | None = None
) -> pd.DataFrame:
    """Concatenate the blocks belonging to ``space`` on their shared index."""
    if space not in FEATURE_SPACES:
        raise ValueError(
            f"unknown feature space {space!r}; expected one of {list(FEATURE_SPACES)}"
        )
    members = [blocks[name] for name in FEATURE_SPACES[space][1]]
    idx = members[0].index
    for b in members[1:]:
        idx = idx.intersection(b.index)
    if index is not None:
        idx = idx.intersection(pd.Index(index))
    X = pd.concat([b.loc[idx] for b in members], axis=1)
    if X.columns.duplicated().any():
        raise ValueError(
            f"duplicate predictor names in space {space!r}: "
            f"{X.columns[X.columns.duplicated()].tolist()}"
        )
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    return X.loc[:, X.nunique(dropna=True) > 1]


def _empty_ladder() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["step", "feature_added", "cv_r2_mean", "cv_r2_std", "delta_r2"]
    )


def forward_stepwise(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    cv_folds: int = 5,
    l1_ratio: float = 0.5,
    random_state: int = 42,
    max_features: int = 25,
    n_candidates: int = 25,
    n_jobs: int | None = None,
    verbose: bool = True,
) -> dict:
    """Greedy forward selection by CV R2 (see module docstring).

    Args:
        y: Target, indexed by sample. NaNs dropped.
        X: Predictor matrix; rows intersected with ``y``.
        max_features: Ladder length cap (steps).
        n_candidates: Size of the ENet-selected shortlist the greedy search
            runs inside.

    Returns: ``{"seq_df", "full_r2", "n_features", "selected_features",
    "n_samples", "n_candidates"}``.
    """
    jobs = _safe_n_jobs(n_jobs)
    idx = y.dropna().index.intersection(X.index)
    empty = {
        "seq_df": _empty_ladder(),
        "full_r2": float("nan"),
        "n_features": 0,
        "selected_features": [],
        "n_samples": int(len(idx)),
        "n_candidates": 0,
    }
    if len(idx) < max(cv_folds, 100):
        return empty

    Xf = X.loc[idx]
    Xf = Xf.loc[:, Xf.nunique(dropna=True) > 1]
    if Xf.shape[1] == 0:
        return empty
    yv = y.loc[idx].astype(float).values

    # 1. ENet pre-filter -> shortlist
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "enet",
                ElasticNetCV(
                    l1_ratio=l1_ratio,
                    cv=cv_folds,
                    random_state=random_state,
                    n_jobs=jobs,
                    max_iter=5000,
                ),
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipe.fit(Xf, yv)
    coefs = pd.Series(pipe.named_steps["enet"].coef_, index=Xf.columns)
    candidates = list(
        coefs[coefs.abs() > 1e-12]
        .abs()
        .sort_values(ascending=False)
        .head(n_candidates)
        .index
    )
    if not candidates:
        return {**empty, "n_samples": int(len(idx))}

    # 2. Greedy forward selection inside the shortlist
    Xc = StandardScaler().fit_transform(Xf[candidates])
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    selected: list[int] = []
    remaining = set(range(len(candidates)))
    rows: list[dict] = []
    prev_r2 = 0.0

    def _eval(fidx: int) -> tuple[int, float, float]:
        cols = selected + [fidx]
        scores = cross_val_score(
            LinearRegression(), Xc[:, cols], yv, cv=kf, scoring="r2"
        )
        return fidx, float(scores.mean()), float(scores.std())

    for step in range(min(max_features, len(candidates))):
        results = Parallel(n_jobs=jobs)(delayed(_eval)(f) for f in remaining)
        best_idx, best_r2, best_std = max(results, key=lambda r: r[1])
        delta = best_r2 - prev_r2
        selected.append(best_idx)
        remaining.discard(best_idx)
        rows.append(
            {
                "step": step + 1,
                "feature_added": candidates[best_idx],
                "cv_r2_mean": best_r2,
                "cv_r2_std": best_std,
                "delta_r2": delta,
            }
        )
        if verbose:
            print(
                f"    step {step + 1:2d}: +{candidates[best_idx]:<24s} "
                f"R2={best_r2:.4f} d={delta:+.4f}",
                flush=True,
            )
        prev_r2 = best_r2

    return {
        "seq_df": pd.DataFrame(rows),
        "full_r2": prev_r2,
        "n_features": len(selected),
        "selected_features": [candidates[i] for i in selected],
        "n_samples": int(len(idx)),
        "n_candidates": len(candidates),
    }


def run_ladders(
    targets: Mapping[str, pd.Series],
    blocks: Mapping[str, pd.DataFrame],
    *,
    spaces: Sequence[str] = tuple(FEATURE_SPACES),
    index: pd.Index | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run one ladder per (target, space) pair; return one tidy long frame."""
    out: list[pd.DataFrame] = []
    for tname, y in targets.items():
        for space in spaces:
            X = space_matrix(blocks, space, index=index)
            print(f"[ladder] {tname} x {space}: X={X.shape}", flush=True)
            res = forward_stepwise(y, X, **kwargs)
            seq = res["seq_df"].copy()
            if not len(seq):
                print(
                    f"[ladder] {tname} x {space}: EMPTY (n={res['n_samples']})",
                    flush=True,
                )
                continue
            seq.insert(0, "space", space)
            seq.insert(0, "target", tname)
            seq["n_samples"] = res["n_samples"]
            seq["n_candidates"] = res["n_candidates"]
            out.append(seq)
    if not out:
        return pd.DataFrame(columns=LADDER_COLUMNS)
    return pd.concat(out, ignore_index=True)[LADDER_COLUMNS]
