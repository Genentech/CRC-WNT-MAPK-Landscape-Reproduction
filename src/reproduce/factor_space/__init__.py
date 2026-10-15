"""Factor-space panel reproduction: Fig2b, Fig2d, SFig3a, SFig3b, SFig3c.

Three shared engines (`programs.py`, `pca.py`, `reconstruction.py`) built once and
composed by `panels.py` into the five panel emitters below.
"""

from . import panels

PANELS = {
    "Fig2b": panels.fig2b,
    "Fig2d": panels.fig2d,
    "SFig3a": panels.sfig3a,
    "SFig3b": panels.sfig3b,
    "SFig3c": panels.sfig3c,
}

PROVENANCE = {
    "Fig2b": "reimplemented -- a variance-weighted aggregation of factor scores into "
    "program scores, written against the synthetic cohort as a short self-contained engine.",
    "Fig2d": "reimplemented -- the study's program-level PCA loadings come from a factor "
    "PCA fitted upstream, which is not part of the synthetic cohort; this is a plain PCA "
    "of factor_scores, aggregated to programs.",
    "SFig3a": "reimplemented -- per-factor reconstruction R2 from the linear factor-model "
    "identity Var(score)*||loading||^2.",
    "SFig3b": "reimplemented -- same reconstruction-R2 engine as SFig3a, shown as a "
    "cumulative-variance curve; partial (144 of the published 270 factors) because the "
    "synthetic cohort does not include the pre-filter factor space.",
    "SFig3c": "reimplemented -- a plain pandas .corr() on factor scores plus an "
    "annotation join.",
}
