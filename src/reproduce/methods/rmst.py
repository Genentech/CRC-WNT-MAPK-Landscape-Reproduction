"""Standard error of the restricted mean survival time (RMST) under left truncation.

`lifelines.utils.restricted_mean_survival_time(..., return_variance=True)` returns the
variance of the restricted-lifetime random variable, not the sampling variance of the RMST
estimator, and lifelines provides no corrected estimator SE under left truncation. The
Greenwood-type variance (Klein & Moeschberger 1997, eq. 4.5.5) is therefore computed from
the event table directly:

    Var[RMST(tau)] = Sum_i A_i^2 * d_i / (n_i * (n_i - d_i))

where A_i = integral_{t_i}^{tau} S(u) du is the area under the survival curve from the
i-th event time to tau, n_i is the number at risk just before t_i, and d_i is the number
of events at t_i. The integral is the right-continuous step-function area (KM is constant
between event times at its post-jump value).
"""

from __future__ import annotations

import numpy as np
from lifelines import KaplanMeierFitter


def rmst_se_greenwood(kmf: KaplanMeierFitter, tau: float) -> float:
    """Standard error of RMST(tau) for a KaplanMeierFitter under left truncation."""
    et = kmf.event_table
    times = et.index.to_numpy()
    d_all = et["observed"].to_numpy()
    n_all = et["at_risk"].to_numpy()
    mask = (times > 0) & (times <= tau) & (d_all > 0)
    if not mask.any():
        return 0.0
    t_e = times[mask]
    d_e = d_all[mask]
    n_e = n_all[mask]
    sf = kmf.survival_function_.iloc[:, 0]  # KM_estimate or BFH_estimate
    s_e = sf.reindex(t_e, method="ffill").to_numpy()
    all_t = np.concatenate([t_e, [float(tau)]])
    widths = np.diff(all_t)
    seg = s_e * widths
    # A_i = integral of S from t_i to tau via reverse cumulative sum
    A = np.cumsum(seg[::-1])[::-1]
    safe = n_e > d_e
    if not safe.any():
        return 0.0
    var = ((A[safe] ** 2) * d_e[safe] / (n_e[safe] * (n_e[safe] - d_e[safe]))).sum()
    return float(np.sqrt(var))
