"""Engine 3 -- propensity-score matching / treatment-selection pipeline.

Serves Fig4d and all eight SFig10 panels (one pipeline, built once and parameterised).
The PS fit, matching and diagnostics are REIMPLEMENTED from standard PSM practice
(logistic-regression PS model, 1:1 nearest-neighbor caliper matching on the logit(PS)
scale, SMD/C2ST/AUC diagnostics).

The synthetic `treatment` block carries a `regimen` column (`Chemo` /
`Chemo+anti-VEGF` / `Chemo+anti-EGFR` / `none`), so arm-stratified panels are matched
and reported per-arm rather than collapsed to a pooled row. The pooled ("Tier1_pooled")
scope is still computed separately where the published schema keeps it (SFig10a,
SFig10b's first row, SFig10d) -- matching ALL treated (any regimen) against untreated
-- while `match_arm` produces an independent 1:1 matched cohort per regimen against
the same untreated pool (arms are NOT mutually exclusive in which untreated samples
they may draw as controls; each arm is matched as its own two-group problem, a
standard choice).

`match_arm` REFITS its own propensity model `P(arm | arm or none)` on the
arm-vs-none subpopulation, as the study's analysis does. It does NOT reuse the pooled
`P(any treatment)` score merely subset to that subpopulation -- the pooled score
answers "is this patient treated with *anything*", which is a different conditional
distribution from "is this patient on *this specific arm*, given they're either on it
or untreated", and subsetting a pooled score to a subpopulation does not recover the
subpopulation-specific model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
CALIPER_SD_FRACTION = 0.2
ARMS = ("Chemo", "Chemo+anti-VEGF", "Chemo+anti-EGFR")


def build_ps_covariates(
    clinical: pd.DataFrame, mutations: pd.DataFrame, mega_cluster: pd.Series
) -> pd.DataFrame:
    """The PS covariates: age, purity, msi_h, sex_male, 2 sidedness dummies
    (drop-first: 3 categories -> 2 dummies), 5 biopsy-site dummies and 9 megacluster
    dummies (20 in total)."""
    df = pd.DataFrame(index=clinical.index)
    df["age"] = clinical["age"].astype(float)
    df["purity"] = clinical["purity"].astype(float)
    df["msi_h"] = mutations["MSI_High"].reindex(clinical.index).astype(int)
    df["sex_male"] = (clinical["sex"] == "male").astype(int)
    sidedness = pd.get_dummies(
        clinical["sidedness"], prefix="sidedness", drop_first=True, dtype=int
    )
    biopsy = pd.get_dummies(clinical["biopsy_site"], prefix="biopsy_site", dtype=int)
    mc = pd.get_dummies(
        mega_cluster.reindex(clinical.index), prefix="mega_cluster", dtype=int
    )
    return pd.concat([df, sidedness, biopsy, mc], axis=1)


def fit_propensity(covariates: pd.DataFrame, treated: pd.Series) -> pd.Series:
    x = StandardScaler().fit_transform(covariates.to_numpy())
    model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    model.fit(x, treated.astype(int).to_numpy())
    ps = model.predict_proba(x)[:, 1]
    ps = np.clip(ps, 1e-4, 1 - 1e-4)
    return pd.Series(ps, index=covariates.index, name="propensity_score")


def cross_validated_auc(covariates: pd.DataFrame, treated: pd.Series) -> float:
    x = StandardScaler().fit_transform(covariates.to_numpy())
    y = treated.astype(int).to_numpy()
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return float("nan")
    proba = cross_val_predict(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        x,
        y,
        cv=5,
        method="predict_proba",
    )[:, 1]
    return float(roc_auc_score(y, proba))


@dataclass
class MatchResult:
    treated_ids: list
    control_ids: list  # matched control, same order as treated_ids (1:1 pairs)
    caliper: float


def match_1to1(
    ps: pd.Series,
    treated: pd.Series,
    *,
    caliper_sd_fraction: float = CALIPER_SD_FRACTION,
) -> MatchResult:
    logit = np.log(ps / (1 - ps))
    caliper = caliper_sd_fraction * logit.std()

    treated_mask = treated.reindex(ps.index).astype(bool).to_numpy()
    ids = ps.index.to_numpy()
    treated_ids_all = ids[treated_mask]
    control_ids_all = ids[~treated_mask]
    control_logit = logit.reindex(control_ids_all).to_numpy()
    treated_logit = logit.reindex(treated_ids_all).to_numpy()

    rng = np.random.default_rng(RANDOM_STATE)
    order = rng.permutation(len(treated_ids_all))
    used = np.zeros(len(control_ids_all), dtype=bool)

    matched_treated, matched_control = [], []
    for t_pos in order:
        diffs = np.abs(control_logit - treated_logit[t_pos])
        diffs = np.where(used, np.inf, diffs)
        j = int(np.argmin(diffs))
        if diffs[j] <= caliper:
            matched_treated.append(treated_ids_all[t_pos])
            matched_control.append(control_ids_all[j])
            used[j] = True

    return MatchResult(
        treated_ids=matched_treated, control_ids=matched_control, caliper=float(caliper)
    )


def match_arm(
    covariates: pd.DataFrame,
    regimen: pd.Series,
    arm: str,
    *,
    caliper_sd_fraction: float = CALIPER_SD_FRACTION,
) -> MatchResult:
    """1:1 match samples on `arm` against `regimen == "none"`, restricted to that
    two-group subpopulation (arms are matched independently of each other).

    REFITS `P(arm | arm or none)` on the arm-vs-none subpopulation (as the study's
    analysis does) rather than reusing the pooled `P(any treatment)` score subset
    to this subpopulation -- see module docstring.
    """
    regimen = regimen.reindex(covariates.index)
    subset_mask = regimen.isin([arm, "none"])
    sub_covariates = covariates.loc[subset_mask]
    sub_treated = regimen.loc[subset_mask] == arm
    sub_ps = fit_propensity(sub_covariates, sub_treated)
    return match_1to1(sub_ps, sub_treated, caliper_sd_fraction=caliper_sd_fraction)


def smd(a: np.ndarray, b: np.ndarray) -> float:
    m1, m2 = np.nanmean(a), np.nanmean(b)
    s1, s2 = np.nanstd(a, ddof=1), np.nanstd(b, ddof=1)
    pooled = np.sqrt((s1**2 + s2**2) / 2)
    return float(abs(m1 - m2) / pooled) if pooled > 0 else 0.0


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = pd.Series(pvals, dtype=float)
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    if valid.empty:
        return out.to_numpy()
    m = len(valid)
    ranks = np.arange(1, m + 1)
    q = (valid.to_numpy() * m) / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    out.loc[valid.index] = np.clip(q, 0, 1)
    return out.to_numpy()
