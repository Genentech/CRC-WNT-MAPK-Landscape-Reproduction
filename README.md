# CRC WNT–MAPK analysis workflow on the synthetic cohort

This repository runs the analyses behind 29 figure panels of *Mutation architectures define a
WNT-MAPK transcriptional landscape in colorectal cancer* on the synthetic cohort deposited on Zenodo.
It lets you run, inspect and modify the analysis code without access to the licensed study data.

The synthetic cohort contains no real patients. Every number this workflow produces is a property of
synthetic data: do not report it as a finding about colorectal cancer or as corroboration of the
manuscript.

## What you need

- Python 3.10 or later, and conda/mamba (or another way to install the packages in `environment.yml`).
- `synthetic_cohort_data.zip` from the Zenodo record https://doi.org/10.5281/zenodo.23055294.
- The `oncophenomap` package (https://github.com/Genentech/oncophenomap), which fits the
  genotype–phenotype trees. `setup_workspace.py` clones it.

## Quick start

1. Download `synthetic_cohort_data.zip` from the Zenodo record.
2. Set up a workspace. This extracts the cohort, gets `oncophenomap`, and prints the next commands:

   ```bash
   git clone https://github.com/Genentech/CRC-WNT-MAPK-Landscape-Reproduction.git crc-wnt-mapk-reproduction
   python crc-wnt-mapk-reproduction/setup_workspace.py \
       --synthetic-zip ~/Downloads/synthetic_cohort_data.zip --workspace crc_workspace
   ```

   If you already have an `oncophenomap` checkout, pass `--oncophenomap-path /path/to/oncophenomap`.
   It clones oncophenomap release `v0.1.0`; `--oncophenomap-ref TAG` selects another release.

3. Create the environment and install both packages:

   ```bash
   mamba env create -f crc-wnt-mapk-reproduction/environment.yml
   mamba activate crc-wnt-mapk-reproduction
   pip install -e crc_workspace/oncophenomap -e crc-wnt-mapk-reproduction
   ```

4. Run all panels:

   ```bash
   python -m reproduce --data crc_workspace/data/synthetic_cohort_data \
       --out crc_workspace/results/panels
   ```

   This writes one CSV per panel, then checks that all 29 are present with the expected columns. It
   took 5–8 minutes on 7 CPU cores in our tests, with the package versions in `environment.yml`.

The workspace looks like this:

```
crc_workspace/
  data/synthetic_cohort_data/   the extracted Zenodo cohort (12 Parquet tables)
  oncophenomap/                 the oncophenomap repository
  results/panels/               one CSV per panel
```

You can also work from Python:

```python
import os
os.environ["SYNTHCRC_SYNTHETIC_DIR"] = "crc_workspace/data/synthetic_cohort_data"

import reproduce

fig2f = reproduce.PANELS["Fig2f"]()   # one panel as a DataFrame
reproduce.emit_all("crc_workspace/results/panels")
```

`SYNTHCRC_FACTOR_REFERENCE` can point to `factor_reference.parquet` if it is stored outside the cohort
folder.

## Panels

| Panel | Contents |
|---|---|
| Fig2b | Correlation matrix of the 17 transcriptional programs |
| Fig2d | Program coordinates in PC1–PC2 loading space |
| Fig2f | Cross-validated partial R² of genomic and H&E features for PC1 and PC2 |
| Fig2h | Correlation of each factor with the WNT and MAPK signature scores |
| Fig3a | Cross-validated elastic-net R² of each factor from genotype |
| Fig3b | Factor WNT correlation against genotype predictability |
| Fig3c | Variance decomposition of WNT activity by mutations and chromosome-arm alterations |
| Fig3d–Fig3g | Cumulative WNT variance explained as features are added (combined, arms, mutations and focal CNV, H&E) |
| Fig4b | Genotype–phenotype map: mean program profile of each mutation group |
| Fig4c | Mean WNT activity against adjusted 36-month restricted mean survival per group |
| Fig4d | Treated-minus-naive shift of each megacluster in PC1–PC2 space |
| SFig3a | Factors with the largest reconstruction R² |
| SFig3b | All factors ranked by reconstruction R² |
| SFig3c | Correlation matrix of the factor scores |
| SFig6 | First-stage tree on chromosome-arm and H&E context |
| SFig7 | Second-stage mutation tree defining the mutation groups |
| SFig8a | Per-program adjusted Cox hazard ratios for overall survival |
| SFig8b | Joint multivariable Cox model of programs and clinical covariates |
| SFig10a | Covariate balance before and after propensity-score matching |
| SFig10b | Post-match classifier AUC, pooled and per treatment class |
| SFig10c | Treated tumours before and after matching, per treatment class |
| SFig10d | Propensity-score distributions, treated and naive |
| SFig10e | Matched treatment-associated change per program and PC1–PC3 |
| SFig10f | Matched treatment-associated change per program, per treatment class |
| SFig10g | Matched treatment-associated change in PC1 and PC2, per treatment class |
| SFig10h | Naive and matched-treated centroids in PC1–PC2, per treatment class |

Fig4b, Fig4c, Fig4d, SFig6, SFig7, SFig8a, SFig8b and SFig10a–h need `oncophenomap`; without it the
other 14 panels still run, and the command reports which were skipped.

Notes on the outputs:

- Every CSV has a `data_origin` column set to `synthetic`, and the output folder contains
  `READ_ME_SYNTHETIC.txt`, so a file shared on its own is still marked as synthetic.
- In Fig3d–Fig3g, focal copy-number features end in `_cnv` (e.g. `APC_cnv`), so they are
  distinguishable from mutations of the same gene (`APC`).
- Fig4b has one `L2#<leaf>` column per fitted mutation group; the leaf numbers are specific to the
  tree fitted on this cohort.
- In SFig10e the comparison with longitudinal pre/post biopsy pairs is empty (NaN): the synthetic
  cohort has one sample per tumour.
- Comparisons of treated with untreated tumours (Fig4d, SFig10) are associations in synthetic data,
  not treatment effects.

## Tests

```bash
python -m unittest discover -s tests
```

The full-run test executes only when `SYNTHCRC_SYNTHETIC_DIR` points to the extracted cohort and
`oncophenomap` is installed.

## Citation

If you use this software, please cite the manuscript and the data record:

> Espinosa C, et al. *Mutation architectures define a WNT-MAPK transcriptional landscape in
> colorectal cancer.* Manuscript submitted.

> Espinosa C, et al. *Data for Mutation architectures define a WNT-MAPK transcriptional landscape
> in colorectal cancer.* Zenodo. https://doi.org/10.5281/zenodo.23055294

A citation for the software itself will be added here when its release is archived.

## License

MIT; see `LICENSE`. Copyright © 2026 Genentech, Inc.
