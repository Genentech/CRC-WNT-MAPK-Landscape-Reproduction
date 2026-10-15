"""Shared validation for the 17-program set (P1-P17).

The manuscript reports 17 transcriptional programs, P1-P17. Both
`reproduce.factor_space.programs` and `reproduce.taxonomy_clinical.programs` aggregate
factors into programs independently; they share this allowlist check so that only
P1-P17 can enter a program-level analysis in either place.
"""

from __future__ import annotations

N_PUBLISHED_PROGRAMS = 17
ALLOWED_PREFIXES = tuple(f"P{i}" for i in range(1, N_PUBLISHED_PROGRAMS + 1))


def validate_program_set(
    programs: list[str], *, source: str = "annotation"
) -> list[str]:
    """Raise unless `programs` is exactly one label for each of P1..P17.

    Checks WHICH programs are present, not just how many: any label outside P1-P17 is
    rejected, and each of P1-P17 must appear exactly once.

    Args:
        programs: Program labels, e.g. `["P1_Myeloid infiltration", ...]`.
        source: Named in the error message, to identify the offending caller.

    Returns:
        `programs` unchanged, so this can wrap an expression inline.
    """
    prefixes = [p.split("_", 1)[0] for p in programs]
    unexpected = [
        p for p, pre in zip(programs, prefixes) if pre not in ALLOWED_PREFIXES
    ]
    if unexpected:
        raise ValueError(
            f"only programs P1-P{N_PUBLISHED_PROGRAMS} may be used; the {source} also "
            f"contains {unexpected}"
        )
    if sorted(prefixes) != sorted(ALLOWED_PREFIXES):
        raise ValueError(
            f"expected each of P1-P{N_PUBLISHED_PROGRAMS} exactly once in the {source}, "
            f"found {sorted(prefixes, key=lambda s: int(s[1:]))}"
        )
    return programs
