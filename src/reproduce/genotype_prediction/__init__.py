"""Genotype-prediction panel reproduction: Fig2f, Fig2h, Fig3a-g.

Four shared engines built once and composed by `panels.py` into the nine panel
emitters below:

- `signatures.py`         WNT/MAPK ssGSEA scoring (+ purity/biopsy residualised variants)
- `variance_partition.py` cross-validated partial-R2 variance partition (Fig2f, Fig3c)
- `enet_genotype.py`      per-factor ElasticNet CV R2 (Fig3a, Fig3b)
- `feature_ladder.py`     greedy forward-selection R2 ladder (Fig3d-g)

The shared statistical methods live in `reproduce.methods`; `PROVENANCE` below
summarizes each panel.
"""

from . import panels

PANELS = {
    "Fig2f": panels.fig2f,
    "Fig2h": panels.fig2h,
    "Fig3a": panels.fig3a,
    "Fig3b": panels.fig3b,
    "Fig3c": panels.fig3c,
    "Fig3d": panels.fig3d,
    "Fig3e": panels.fig3e,
    "Fig3f": panels.fig3f,
    "Fig3g": panels.fig3g,
}

PROVENANCE = {
    "Fig2f": "Hybrid. The target (PC1/PC2 of the factor PCA) is reimplemented as a sklearn PCA of "
    "the standardized retained factor_scores; the variance-partition method is "
    "`reproduce.methods.variance_partition`. The mutation/arm commonality split uses raw-scale marginal R2s "
    "(`unique_x = joint - R2_other`, `shared = R2_a + R2_b - joint`, each clipped at 0), and "
    "`HE_partial_R2` is the conditional partial R2 against the full ten-modality joint (mut, "
    "cnv_focal, cnv_arm, HE, biopsy, MSI, Side, purity, age, sex). Categorical blocks are one-hot "
    "with the majority level dropped; age enters as a continuous variable because the synthetic "
    "clinical block has no age-bin column (the study bins it).",
    "Fig2h": "Hybrid. Plain per-factor Pearson correlation of factor_scores with the WNT and MAPK "
    "ssGSEA scores. ssGSEA scoring is a DataFrame-native gp.ssgsea call; MAPK uses the 10-gene "
    "erk_activity_10 panel.",
    "Fig3a": "Reimplemented. Per-factor ElasticNetCV(l1_ratio=0.5, cv=5) with 5-fold cross_val_score "
    "over the combined (mutation+focal_cnv+arm_fractions+HE) feature block, parallelized across "
    "factors with joblib.",
    "Fig3b": "Hybrid. Plain inner merge of Fig2h's correlation table with Fig3a's ElasticNet R2 table "
    "on `factor`; both input tables carry the same provenance as Fig2h/Fig3a.",
    "Fig3c": "Hybrid. Same variance-partition engine as Fig2f, with the raw wnt_ssgsea score as "
    "target (reimplemented ssGSEA scoring, see Fig2h), over the full 5000-sample synthetic cohort "
    "with no purity adjustment.",
    "Fig3d": "Hybrid. forward_stepwise, run_ladders and space_matrix come from "
    "`reproduce.methods.feature_ladder`; the feature blocks are built from the cohort tables. "
    "Target is raw wnt_ssgsea on a treatment-naive primary cohort proxy; cv_folds=5, "
    "l1_ratio=0.5, max_features=25, n_candidates=25, seed=42.",
    "Fig3e": "Hybrid. Same engine as Fig3d, space='arm' (arm_fractions only).",
    "Fig3f": "Hybrid. Same engine as Fig3d, space='mut_cnv' (mutations + focal_cnv only).",
    "Fig3g": "Hybrid. Same engine as Fig3d, space='he' (he_features only; H&E missingness "
    "(3,892 of 5,000 samples) narrows this space's cohort accordingly).",
}
