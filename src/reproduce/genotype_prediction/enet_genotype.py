"""Engine 3: elastic-net per-factor genotype prediction. Serves Fig3a and Fig3b.

REIMPLEMENTED with the ElasticNetCV + KFold cross_val_score pattern (the same
Pipeline(StandardScaler, ElasticNetCV) recipe used by the ladder's ElasticNet pre-filter).
Fold-wise scoring is used because it recovers BOTH `r2_cv_mean` and `r2_cv_std` -- Fig3b's
schema needs the fold-wise std, which the single `cross_val_predict`-based `cv_score` of
`elastic_net_regression` does not expose.

Per-factor fits are independent (CPU-bound, joblib-parallel): `n_jobs` sized from
`len(os.sched_getaffinity(0)) - 1`, each worker pinned to a single BLAS thread via
`threadpool_limits(1)` to avoid `n_jobs x BLAS_threads` oversubscription.
"""

from __future__ import annotations

import os

import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

CV_FOLDS = 5
L1_RATIO = 0.5
RANDOM_STATE = 42


def safe_n_jobs(reserve: int = 1) -> int:
    try:
        avail = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        avail = os.cpu_count() or 1
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        avail = min(avail, int(slurm))
    return max(1, avail - reserve)


def _fit_one(X: pd.DataFrame, y: pd.Series) -> tuple[float, float, int]:
    # Defensive: drop rows with any NaN in X too (e.g. a caller passing an
    # outer-joined block), not just NaN targets -- ElasticNetCV cannot fit through NaN.
    complete_X = X.index[X.notna().all(axis=1)]
    idx = complete_X.intersection(y.dropna().index)
    Xf = X.loc[idx]
    yf = y.loc[idx].astype(float).to_numpy()
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "enet",
                ElasticNetCV(
                    l1_ratio=L1_RATIO,
                    cv=CV_FOLDS,
                    random_state=RANDOM_STATE,
                    max_iter=10000,
                ),
            ),
        ]
    )
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    with threadpool_limits(1):
        scores = cross_val_score(pipe, Xf, yf, cv=kf, scoring="r2", n_jobs=1)
    return float(scores.mean()), float(scores.std()), int(len(idx))


def factor_enet_r2(
    X: pd.DataFrame,
    targets: pd.DataFrame,
    factors: list[str],
    *,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """5-fold CV ElasticNet R2 per factor in `factors`; `targets` columns superset `factors`."""
    jobs = n_jobs or safe_n_jobs()
    results = Parallel(n_jobs=jobs)(delayed(_fit_one)(X, targets[f]) for f in factors)
    means, stds, ns = zip(*results) if results else ((), (), ())
    return pd.DataFrame(
        {
            "factor": list(factors),
            "r2_cv_mean": list(means),
            "r2_cv_std": list(stds),
            "n_samples": list(ns),
        }
    )
