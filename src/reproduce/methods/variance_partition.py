"""Variance partitioning across predictor modalities.

`run_variance_partition` fits a joint model and one model per left-out modality, and
`partial_r2` turns the two into each modality's unique contribution (floored at zero).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def partial_r2(r2_full: np.ndarray, r2_reduced: np.ndarray) -> np.ndarray:
    """Modality-unique variance (floored at zero)."""
    a = np.asarray(r2_full, dtype=float)
    b = np.asarray(r2_reduced, dtype=float)
    return np.clip((a - b) / np.maximum(1.0 - b, 1e-9), 0.0, None)


def run_variance_partition(
    modalities: dict[str, pd.DataFrame],
    Y: pd.DataFrame,
    *,
    cv: int = 5,
    random_state: int = 42,
    l1_ratio: float = 0.5,
    metric: str = "cv_score",
    oof_joint_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
    """Run full leave-one-out variance partition across modalities.

    Returns (partition_table, marginal_results).
    ``partition_table`` has columns: R2_<mod>, R2_joint, partial_R2_<mod>, residual.
    ``marginal_results`` maps modality name → (metrics_df, coef_df).

    When *oof_joint_only* is True, only the joint fit computes OOF R²;
    marginal and leave-one-out fits use in-sample R² (5-6× faster for
    single-target partitions).
    """
    from .elastic_net import elastic_net_regression_multiple_targets

    def _enet(
        Xin: pd.DataFrame, *, compute_oof: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        _r, m, c = elastic_net_regression_multiple_targets(
            X=Xin,
            Y=Y,
            task="regression",
            cv=cv,
            random_state=random_state,
            l1_ratio=l1_ratio,
            scale_features=True,
            show_progress=False,
            compute_oof=compute_oof,
        )
        m_idx = m.set_index("target")[
            [
                c
                for c in [
                    "cv_score",
                    "in_sample_score",
                    "r2_score",
                    "n_features_selected",
                ]
                if c in m.columns
            ]
        ]
        return m_idx, c

    def _r2(m: pd.DataFrame, *, prefer_oof: bool = True) -> np.ndarray:
        col = metric if prefer_oof else "in_sample_score"
        vals = (
            m[col].reindex(Y.columns).values
            if col in m.columns
            else m[metric].reindex(Y.columns).values
        )
        if prefer_oof and np.all(np.isnan(vals)) and "in_sample_score" in m.columns:
            vals = m["in_sample_score"].reindex(Y.columns).values
        return vals

    fast = not oof_joint_only  # if not oof_joint_only, everything gets OOF

    # Marginal fits (skip OOF when oof_joint_only)
    marginals = {
        name: _enet(block, compute_oof=fast) for name, block in modalities.items()
    }

    # Joint fit (always OOF)
    X_joint = pd.concat(list(modalities.values()), axis=1)
    if X_joint.columns.duplicated().any():
        raise ValueError(
            "duplicate predictor names across modalities: "
            f"{X_joint.columns[X_joint.columns.duplicated()].tolist()}"
        )
    m_joint, _ = _enet(X_joint, compute_oof=True)

    # Leave-one-out fits (skip OOF when oof_joint_only)
    leave_one_out = {}
    for name in modalities:
        other_blocks = [block for n, block in modalities.items() if n != name]
        leave_one_out[name], _ = _enet(
            pd.concat(other_blocks, axis=1), compute_oof=fast
        )

    # Build partition table
    part = pd.DataFrame({"target": list(Y.columns)})
    part["n_shared"] = len(Y)
    for name in modalities:
        part[f"R2_{name}"] = _r2(marginals[name][0], prefer_oof=fast)
    part["R2_joint"] = _r2(m_joint, prefer_oof=True)
    for name in modalities:
        part[f"partial_R2_{name}"] = partial_r2(
            part["R2_joint"].values, _r2(leave_one_out[name], prefer_oof=fast)
        )
    part["residual"] = (1.0 - part["R2_joint"]).clip(lower=0.0)

    return part, marginals


__all__ = ["partial_r2", "run_variance_partition"]
