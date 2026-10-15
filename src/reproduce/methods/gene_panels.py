"""WNT and MAPK gene panels and the covariates used to residualize their scores.

`WNT_GENES` is the 7-gene WNT panel, `MAPK_GENESETS` holds the MAPK panels (default
`DEFAULT_MAPK_GENESET`), and `RESID_COVARIATES` lists the covariates removed before
signatures are compared.
"""

from __future__ import annotations

#: Fixed 7-gene WNT panel.
WNT_GENES: tuple[str, ...] = (
    "AXIN2",
    "ASCL2",
    "LGR5",
    "WIF1",
    "NKD1",
    "SMOC2",
    "SLC12A2",
)

#: Default MAPK panel (``MAPK_GENESET`` = ``erk_activity_10``). The alternative
#: ``erk_feedback_7`` panel is also listed here.
MAPK_GENESETS: dict[str, tuple[str, ...]] = {
    "erk_activity_10": (
        "PHLDA1",
        "SPRY2",
        "SPRY4",
        "DUSP4",
        "DUSP6",
        "CCND1",
        "EPHA2",
        "EPHA4",
        "ETV4",
        "ETV5",
    ),
    "erk_feedback_7": (
        "SPRED1",
        "SPRED2",
        "SPRY2",
        "SPRY4",
        "ETV6",
        "DUSP4",
        "DUSP6",
    ),
}
DEFAULT_MAPK_GENESET = "erk_activity_10"

#: Covariates the ``*_resid`` scores are residualised against.
RESID_COVARIATES: tuple[str, ...] = ("Purity", "biopsy")

__all__ = [
    "WNT_GENES",
    "MAPK_GENESETS",
    "DEFAULT_MAPK_GENESET",
    "RESID_COVARIATES",
]
