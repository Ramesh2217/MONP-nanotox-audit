# MONP-Benchmark-Audit

Reproducibility audit of the publicly available NanoTox metal-oxide nanoparticle
(MONP) cytotoxicity benchmark.

This repository contains analysis scripts, result files, computational
documentation, and supporting materials that enable inspection and independent
checking of the principal analyses reported in the manuscript.

**Manuscript:** Avuguddi, R. B.; Miditana, S. R.; Bobburi, N.; Korlam, S.;
Theertham, G.; Balireddi, D. A. “Validation strategy governs apparent
machine-learning performance in metal-oxide nanoparticle cytotoxicity
prediction.” Unpublished manuscript.

The analysis compares random stratified five-fold cross-validation with
leave-one-metal-oxide-out (LOMO) validation for Logistic Regression, Random
Forest, RBF Support Vector Machine, Multilayer Perceptron, and XGBoost. The
controlled duplicate-injection experiment uses Random Forest only. The
dose-threshold comparator is evaluated under LOMO, with its threshold selected
by MCC in the training partition and applied to the held-out oxide. The
validation-strategy gap is random five-fold MCC minus LOMO MCC.
## What this repository contains


The benchmark dataset is **not redistributed here** and must be obtained
directly from the original NanoTox repository (see
[`data/README.md`](data/README.md)).

```
MONP-Benchmark-Audit/
├── README.md
├── LICENSE
├── CITATION.cff
├── REPRODUCE.md
├── REVISION_ANALYSES.md
├── requirements.txt
├── MONP_supplementary_material.docx   # Supplementary tables (S1–S6) and figures
├── notebooks/            # Jupyter notebooks for analysis workflows
├── scripts/              # Standalone Python scripts (callable from CLI)
├── results/              # Generated CSV outputs and figures
├── manuscript/           # README describing the supplementary material
└── data/                 # Instructions for obtaining the dataset (no data here)
```

## Data

The analysis uses the NanoTox metal-oxide nanoparticle cytotoxicity benchmark
released by Subramanian and Palaniappan (2021), available from the NanoTox GitHub
repository: https://github.com/NanoTox/ToxicityModel (MIT License). No dataset is
shipped in this repository — `data/` contains only instructions.

Obtain the benchmark and place it in the `data/` folder before running any script
or notebook. The loader (`scripts/common.py`) auto-detects, in order,
`data/dataset.txt` (the tab-separated file as downloaded — recommended),
`data/NanoTox_dataset_clean.xlsx`, or `data/NanoTox.csv`; the first one present is
used. See [`data/README.md`](data/README.md) for the authoritative instructions.

## Reproducing the analysis

1. Obtain the dataset (see above) and place it in `data/` (e.g. as `data/dataset.txt`).
2. Create an environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run the repository's combined analysis script:
   ```bash
   cd scripts
   python run_all.py
   ```
   This runs all fourteen steps in order (nine core stages plus five revision
   analyses), writes 300-DPI grayscale-safe figures to
   `results/figures/`, writes all result tables to `results/`, and finishes with
   a self-check confirming the run reproduces the reported dataset composition
   (483 → 364 unique modeling observations, 68 cytotoxic).

   Alternatively, run the master notebook `notebooks/00_Run_Full_Pipeline.ipynb`,
   or the individual scripts/notebooks for a single analysis:
   ```bash
   python scripts/duplicate_audit.py
   python scripts/leakage_curve.py        # writes Figure 1
   python scripts/lomo_validation.py
   python scripts/baseline_models.py
   python scripts/bootstrap_ci.py
   python scripts/permutation_test.py
   python scripts/seed_stability.py        # writes Figure 2
   python scripts/feature_decomposition.py # writes Table S3
   python scripts/per_oxide_folds.py       # writes Figure S2
   ```

## Figures

Publication-grade figures (300 DPI, journal-neutral, grayscale-safe) are written
to `results/figures/`:

Main-text figures:

- `Figure1_leakage_curve.png` — Figure 1: validation-strategy gap vs controlled duplicate-injection fraction, with OLS trend.
- `Figure2_seed_stability.png` — Figure 2: per-seed LOMO MCC for each model (MLP
  instability vs the seed-stable behaviour of the other classifiers).
- `Figure_learning_curve.png` — Figure 3: leave-one-metal-oxide-out (LOMO) learning curves for
  XGBoost and the random forest.
- `Figure_workflow.png` — Scheme 1 (unnumbered): workflow overview of the audit
  and validation pipeline.

Supplementary figures:

- `FigureS1_permutation_null.png` — Figure S1a: permutation null of the XGBoost LOMO MCC.
- `FigureS1b_permutation_null_mlp.png` — Figure S1b: permutation null of the MLP LOMO MCC.
- `FigureS2_per_oxide_folds.png` — Figure S2: per-oxide held-out performance (n and
  cytotoxic counts).
- `Figure_oxide_diversity.png` — Figure S3: effect of training-set material diversity.

Repository-only (not numbered manuscript figures):

- `bootstrap_forest.png` — LOMO MCC with 95% bootstrap confidence intervals; a
  visualization of Table 1.
- `Graphical_abstract.png` — graphical abstract.

The copies shipped in this repository were generated by the authors' run and are
overwritten when you re-run the pipeline.

## Analysis overview

| Step | Script | Notebook | Purpose |
|------|--------|----------|---------|
| — | `run_all.py`          | `00_Run_Full_Pipeline.ipynb` | Run all stages end-to-end + self-check |
| 1 | `duplicate_audit.py`  | `01_Data_Audit.ipynb`        | Audit and remove model-space duplicate records (483 → 364) |
| 2 | `leakage_curve.py`    | `02_LOMO_Validation.ipynb`   | Validation-strategy gap vs controlled duplicate-injection fraction (**Figure 1**) |
| 3 | `lomo_validation.py`  | `02_LOMO_Validation.ipynb`   | Leave-one-metal-oxide-out (LOMO) evaluation of the classifiers |
| 4 | `baseline_models.py`  | `02_LOMO_Validation.ipynb`   | Single-variable dose-threshold baseline under LOMO |
| 5 | `bootstrap_ci.py`     | `03_Bootstrap_CI.ipynb`      | Bootstrap 95% confidence intervals for MCC |
| 6 | `permutation_test.py` | `04_Permutation_Test.ipynb`  | Permutation test of LOMO performance against the null |
| 7 | `seed_stability.py`   | `05_Seed_Stability.ipynb`    | Stability across random seeds (**Figure 2**, Table S2) |
| 8 | `feature_decomposition.py` | `00_Run_Full_Pipeline.ipynb` | Full / dose-only / no-dose descriptor sets under LOMO (Table S3) |
| 9 | `per_oxide_folds.py`  | `00_Run_Full_Pipeline.ipynb` | Per-oxide held-out performance with n / cytotoxic counts (**Figure S2**) |

Beyond these nine core stages, `run_all.py` also runs five additional analyses (documented in [`REVISION_ANALYSES.md`](REVISION_ANALYSES.md)),
for 14 steps in total.

`common.py` holds the shared data path, descriptor set, and deduplication rule;
`figstyle.py` holds the publication-grade, grayscale-safe figure style.

## Reproducibility settings

A fixed random seed (42) is used for the primary analysis; eight seeds
{0, 1, 2, 3, 7, 13, 42, 99} are used for the stability assessment. Core analysis package versions are pinned in `requirements.txt`; Jupyter is listed without a fixed version for notebook use.

## Citation

If you use this code, please cite the manuscript (see `CITATION.cff`) and the
original NanoTox benchmark (Subramanian and Palaniappan, 2021).

## License

Analysis code in this repository is released under the MIT License
(see `LICENSE`). The NanoTox benchmark dataset is governed by its own license
in the NanoTox repository and is not included here.

