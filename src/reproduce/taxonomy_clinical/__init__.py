"""Taxonomy/tree + clinical-outcome panel reproduction: Fig4b, Fig4c, Fig4d, SFig6,
SFig7, SFig8a, SFig8b, SFig10a-h (15 panels).

Three engines (`tree_engine.py`, `cox_engine.py`, `psm_engine.py`) built once and
composed by `panels.py`. See each engine module's docstring for what is reused vs.
reimplemented.

On the synthetic cohort, every panel that compares ever-treated with untreated
samples is a treatment-selection/association demonstration only. Post-biopsy treatment cannot
cause biopsy phenotype, and a fixed ever-treated survival covariate is not a causal treatment
effect because initiation occurs after risk-set entry for many samples.
"""

from . import panels

PANELS = {
    "Fig4b": panels.fig4b,
    "Fig4c": panels.fig4c,
    "Fig4d": panels.fig4d,
    "SFig6": panels.sfig6,
    "SFig7": panels.sfig7,
    "SFig8a": panels.sfig8a,
    "SFig8b": panels.sfig8b,
    "SFig10a": panels.sfig10a,
    "SFig10b": panels.sfig10b,
    "SFig10c": panels.sfig10c,
    "SFig10d": panels.sfig10d,
    "SFig10e": panels.sfig10e,
    "SFig10f": panels.sfig10f,
    "SFig10g": panels.sfig10g,
    "SFig10h": panels.sfig10h,
}

PROVENANCE = {
    "SFig6": "the L1 arm/H&E rule rendering is adapted from the study's analysis "
    "code; the L1 context tree itself is fit with oncophenomap.residualize's "
    "`_binary_block`/`_median_block`/`_fit_context_tree`/`_routed_context` building "
    "blocks, using the study's settings (arm leaf cap 8, H&E leaf cap 2, minimum leaf "
    "size 500).",
    "SFig7": "reimplemented -- the pooled L2 tree (oncophenomap.GenotypePhenotypeTree.fit "
    "on the L1-context-residualized phenotype vs mutations+focal_cnv) is reused from the "
    "oncophenomap package; only the group_full pipe-string rendering is reimplemented. "
    "There is no CNV-locus collapse (no gene-coordinate reference ships with this "
    "package), so CNV features stay per-gene rather than collapsed into CNV_LOCUS_* "
    "strings.",
    "Fig4b": "reimplemented -- the z-scored program profile per L2 group (program "
    "re-z-scored across groups), computed from factor_scores and the L2 tree fit from "
    "SFig7. The non-metadata columns are named `L2#<leaf_id>`, one per fitted L2 group; "
    "the leaf ids are tree node indices specific to this tree fit, so they cannot "
    "match the leaf ids of the published panel.",
    "Fig4c": "reimplemented; ever-treated coefficients are association-only, not causal "
    "treatment effects. One joint left-truncated CoxPHFitter with `mutation_group` as "
    "`strata` (shared covariate effects, group-specific baseline hazard), directly "
    "standardized to the whole-cohort covariate distribution for rmst_36_adjusted. "
    "rmst_36_se and km_estimator come from an unadjusted per-group Kaplan-Meier fit "
    "with the Greenwood SE of the RMST estimator (Klein & Moeschberger eq. 4.5.5, "
    "`reproduce.methods.rmst`).",
    "SFig8a": "the per-program adjusted Cox scan is adapted from the study's analysis "
    "code: covariates age_z, purity_z, sex_male, msi_high, "
    "metastatic; penalizer 0.01; left truncation on time from entry; BH-FDR; forest "
    "selection (fdr/p/abs_log_hr sort, head(35), then hazard_ratio sort).",
    "SFig8b": "the joint multivariable Cox fit is adapted from the study's analysis "
    "code: penalizer 0.05, 17 programs + 5 clinical covariates fit "
    "simultaneously, hazard_ratio-sorted, with no significance filter on the rows.",
    "Fig4d": "reimplemented; treated/naive contrasts are selection associations, not "
    "treatment effects. Plain per-(mega_cluster, treated) PC1/PC2 centroids on the "
    "sample-level factor PCA, over the full cohort (not PSM-matched).",
    "SFig10a": "reimplemented treatment-selection/matching diagnostic, not a causal "
    "effect -- a logistic-regression propensity model, 1:1 nearest-neighbor caliper "
    "matching on logit(PS) and raw/matched SMD, implemented from standard PSM "
    "practice. Covariates: age, purity, msi_h, sex_male, 2 sidedness dummies "
    "[drop-first], 5 biopsy_site dummies and one dummy per assigned megacluster (19 in "
    "total on the deposited cohort); pooled scope "
    "only (no `arm` column).",
    "SFig10b": "reimplemented -- the Tier1_pooled row uses the same pooled PS/match as "
    "SFig10a (C2ST AUC via 5-fold cross-validated logistic regression); the 3 arm rows "
    "each use an independent 1:1 match per regimen (`psm_engine.match_arm`, each arm vs "
    "`none`). The independent-per-arm-match design is a documented choice.",
    "SFig10c": "reimplemented -- pre/post-match counts per regimen arm, one independent "
    "1:1 match per arm (see SFig10b). 3 rows, no pooled row (as in the published "
    "schema).",
    "SFig10d": "reimplemented -- 20-bin propensity-score histogram (treated of any "
    "regimen vs naive, pooled scope) from the same PS model as SFig10a.",
    "SFig10e": "partly producible -- real_pair_mean_delta needs same-patient pre/post "
    "longitudinal biopsy pairs, which the synthetic cohort does not have (one "
    "row per sample), so that column is emitted as NaN rather than "
    "simulated. psm_mean_delta (pooled matched treated-minus-control delta over the 17 "
    "programs plus PC1-PC3, 20 rows) is reimplemented.",
    "SFig10f": "reimplemented -- per-program matched paired-delta with SE/CI, one "
    "independent 1:1 match per regimen arm (see SFig10b), BH-FDR computed WITHIN each "
    "arm's 17 p-values (`fdr_per_arm`). 17 programs x 3 arms = 51 rows when every arm's "
    "match succeeds for every program.",
    "SFig10g": "reimplemented -- same per-arm paired-delta machinery as SFig10f, applied "
    "to PC1/PC2. 2 PCs x 3 arms = 6 rows.",
    "SFig10h": "reimplemented -- matched-cohort PC1/PC2 centroids per regimen arm (naive "
    "vs treated), reusing Fig4d's factor PCA and the per-arm matches from SFig10b/c. "
    "3 rows, one per arm.",
}
