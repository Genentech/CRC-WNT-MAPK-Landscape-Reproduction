"""Cross-validated variance partition for Fig2f and Fig3c.

The genomic commonality calculation uses marginal mutation, chromosome-arm, and joint R² values:

    unique_mutation = clip(joint - arm, 0, None)
    unique_arm      = clip(joint - mutation, 0, None)
    shared          = clip(mutation + arm - joint, 0, None)

H&E contribution is its conditional partial R² in the full multimodal model. Mutation, focal-CNV,
and arm-CNV blocks are encoded as binary indicators and receive one joint 2% prevalence filter.
MSI is excluded from the mutation block because it enters the multimodal model separately. Age is
continuous because that is the representation supplied by the synthetic cohort.
"""

from __future__ import annotations

import pandas as pd

from ..methods.variance_partition import run_variance_partition

CV_KW = dict(cv=5, random_state=42, l1_ratio=0.5)

#: Minimum prevalence used by the analysis workflow.
MIN_PREVALENCE = 0.02

#: Event thresholds used by the CNV encodings.
FOCAL_AMP_THRESHOLD = 2
FOCAL_DEL_THRESHOLD = -2
ARM_GAIN_THRESHOLD = 1
ARM_LOSS_THRESHOLD = -1

#: Optional pre-encoded focal-region columns are passed through as binary events.
LOCUS_PREFIX = "CNV_LOCUS_"

#: Excluded from the genomic model; enters the ten-modality joint as its own block.
MSI_COLUMN = "MSI_High"


def genomic_blocks(
    mutations: pd.DataFrame,
    focal_cnv: pd.DataFrame,
    arm_calls: pd.DataFrame,
    *,
    min_prevalence: float = MIN_PREVALENCE,
) -> dict[str, pd.DataFrame]:
    """Encode the three genomic modality blocks as the study does.

    Returns `{"mut", "cnv_focal", "cnv_arm"}`, all binary int8, after a single joint
    prevalence filter over their concatenation (the study filters the concatenated matrix
    once and then splits by modality, so a column dropped for rarity is dropped from
    whichever block owns it -- filtering each block separately is not equivalent).

    `MSI_High` is removed from `mut`: it is not part of the genomic model.

    A modality can be empty when all of its indicators fall below the prevalence floor.
    """
    dna = mutations.drop(columns=[MSI_COLUMN], errors="ignore").astype("int8")

    locus_cols = [c for c in focal_cnv.columns if c.startswith(LOCUS_PREFIX)]
    gene_cols = [c for c in focal_cnv.columns if c not in set(locus_cols)]
    genes, loci = focal_cnv[gene_cols], focal_cnv[locus_cols].astype("int8")

    amp = (genes >= FOCAL_AMP_THRESHOLD).astype("int8").add_suffix("_amp")
    dele = (genes <= FOCAL_DEL_THRESHOLD).astype("int8").add_suffix("_del")
    gain = (arm_calls >= ARM_GAIN_THRESHOLD).astype("int8").add_suffix("_gain")
    loss = (arm_calls <= ARM_LOSS_THRESHOLD).astype("int8").add_suffix("_loss")

    members = {
        "mut": list(dna.columns),
        "cnv_focal": list(amp.columns) + list(dele.columns) + list(loci.columns),
        "cnv_arm": list(gain.columns) + list(loss.columns),
    }
    x_full = pd.concat([dna, amp, dele, loci, gain, loss], axis=1)
    keep = set(x_full.columns[x_full.mean(axis=0) >= min_prevalence])
    return {
        name: x_full[[c for c in cols if c in keep]] for name, cols in members.items()
    }


def _f64(block: pd.DataFrame) -> pd.DataFrame:
    """Cast a modality block to float64 before it reaches ElasticNetCV.

    sklearn's coordinate descent can accumulate enough float32 error to fail its internal Gram
    matrix validation for wide modality blocks.
    """
    return block.astype("float64")


def _onehot_drop_majority(series: pd.Series, name: str) -> pd.DataFrame:
    """One-hot a categorical with the most-prevalent class as reference (dropped).

    The majority class is the reference (not the first or alphabetical class), as in
    the study's analysis.
    """
    counts = series.value_counts()
    ref = counts.idxmax()
    dummies = pd.get_dummies(series, prefix=name, drop_first=False).astype("int8")
    return dummies.drop(columns=[f"{name}_{ref}"], errors="ignore")


def genomic_variance_partition(
    blocks: dict[str, pd.DataFrame],
    he: pd.DataFrame | None,
    target: pd.DataFrame,
    *,
    biopsy: pd.Series | None = None,
    msi: pd.Series | None = None,
    side: pd.Series | None = None,
    purity: pd.Series | None = None,
    age: pd.Series | None = None,
    sex: pd.Series | None = None,
) -> pd.DataFrame:
    """Per-target variance decomposition, indexed by target column name.

    Columns: `mutations_unique_partial_R2`, `mut_arm_shared_partial_R2`,
    `arm_cnv_unique_partial_R2`, `n_samples_genomic` (raw-scale genomic commonality,
    always computed), and -- only when `he` AND all seven additional modality blocks
    below are given -- `HE_partial_R2`, `n_samples_full` (conditional partial R2 of HE
    against the full ten-modality joint; see module docstring).

    `blocks` must come from `genomic_blocks()` and supply `mut`,
    `cnv_focal` and `cnv_arm`. The genomic commonality uses `mut` + `cnv_arm` only --
    focal CNV is not in the two-set genomic decomposition.
    `cnv_focal` enters only the ten-modality joint behind `HE_partial_R2`.

    `biopsy`, `msi`, `side`, `purity`, `age`, `sex` are optional/default-None so
    callers that never need `HE_partial_R2` (e.g. Fig3c, which passes `he=None`) are
    unaffected. A caller that passes `he` without the other six raises -- there is no
    reduced-modality fallback that reproduces the ten-modality conditional
    quantity, so silently computing something else on a partial set of blocks is avoided.
    """
    required = {"mut", "cnv_focal", "cnv_arm"}
    if not required <= set(blocks):
        raise ValueError(
            f"`blocks` must contain {sorted(required)} (use genomic_blocks); "
            f"got {sorted(blocks)}"
        )
    mut, arm = _f64(blocks["mut"]), _f64(blocks["cnv_arm"])
    part_genomic, _ = run_variance_partition({"mut": mut, "arm": arm}, target, **CV_KW)
    part_genomic = part_genomic.set_index("target")

    R2_mut = part_genomic["R2_mut"]
    R2_arm = part_genomic["R2_arm"]
    R2_genomic_joint = part_genomic["R2_joint"]

    out = pd.DataFrame(index=part_genomic.index)
    out["mutations_unique_partial_R2"] = (R2_genomic_joint - R2_arm).clip(lower=0.0)
    out["arm_cnv_unique_partial_R2"] = (R2_genomic_joint - R2_mut).clip(lower=0.0)
    out["mut_arm_shared_partial_R2"] = (R2_mut + R2_arm - R2_genomic_joint).clip(
        lower=0.0
    )
    out["n_samples_genomic"] = part_genomic["n_shared"]

    if he is not None:
        provided = {
            "biopsy": biopsy,
            "msi": msi,
            "side": side,
            "purity": purity,
            "age": age,
            "sex": sex,
        }
        missing = [name for name, val in provided.items() if val is None]
        if missing:
            raise ValueError(
                "genomic_variance_partition(he=...) requires the full ten-modality "
                f"joint used in the study's analysis; "
                f"missing blocks: {missing}. Pass cnv_focal, biopsy, msi, side, purity, "
                "age, sex (see module docstring)."
            )
        ten_modalities = {
            "mut": mut,
            "cnv_focal": blocks["cnv_focal"],
            "cnv_arm": arm,
            "HE": he,
            "biopsy": _onehot_drop_majority(biopsy, "biopsy"),
            "MSI": _onehot_drop_majority(msi, "MSI"),
            "Side": _onehot_drop_majority(side, "Side"),
            "purity": purity.to_frame("purity")
            if isinstance(purity, pd.Series)
            else purity,
            "age": age.to_frame("age") if isinstance(age, pd.Series) else age,
            "sex": _onehot_drop_majority(sex, "sex"),
        }
        common = target.index
        for block in ten_modalities.values():
            common = common.intersection(block.index)
        ten_modalities = {
            name: _f64(block.loc[common]) for name, block in ten_modalities.items()
        }
        target_full = target.loc[common]

        part_full, _ = run_variance_partition(ten_modalities, target_full, **CV_KW)
        part_full = part_full.set_index("target")
        out["HE_partial_R2"] = part_full["partial_R2_HE"].reindex(out.index)
        out["n_samples_full"] = part_full["n_shared"].reindex(out.index)

    return out
