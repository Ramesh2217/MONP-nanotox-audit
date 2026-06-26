# Reproducing the analysis

This document explains how to regenerate **every figure and table** (main text
and supplementary) from scratch.

## Recommended: run the notebook in VS Code

1. Open the repository folder in **VS Code** (with the Python and Jupyter
   extensions installed).
2. Download `dataset.txt` from the NanoTox repository and place it at
   `data/dataset.txt` (see `data/README.md`).
3. Create and select an environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Open `notebooks/00_Run_Full_Pipeline.ipynb`, select the venv kernel, and
   choose **Run All**.

The notebook runs all nine stages in order, saves every figure (300-DPI PNG) to
`results/figures/` and every table (CSV) to `results/`, displays each one inline,
and finishes with a self-check confirming 483 → 364 → 68.

## Alternative: command line

```bash
cd scripts
python run_all.py
```

## Or run any stage on its own

| Stage | Script | Outputs | Manuscript item |
|-------|--------|---------|-----------------|
| 1  | `duplicate_audit.py`       | `duplicate_audit_summary.csv`, `per_oxide_composition.csv` | Table 2 |
| 2  | `leakage_curve.py`         | `leakage_curve.csv`, `Figure1_leakage_curve.png` | Figure 1, Table S1 |
| 3  | `lomo_validation.py`       | `classifier_results.csv` | Table 1 |
| 4  | `baseline_models.py`       | `baseline_results.csv` | (Section 3.2) |
| 5  | `bootstrap_ci.py`          | `bootstrap_results.csv`, `Figure3_bootstrap_forest.png` | Figure 3 |
| 6  | `permutation_test.py`      | `permutation_results.csv`, `FigureS1_permutation_null.png` | Figure S1 |
| 7  | `seed_stability.py`        | `seed_stability.csv`, `Figure2_seed_stability.png` | Figure 2, Table S2 |
| 8  | `feature_decomposition.py` | `feature_decomposition.csv` | Table S4 |
| 9  | `per_oxide_folds.py`       | `per_oxide_folds.csv`, `FigureS2_per_oxide_folds.png` | Table S3, Figure S2 |

## Figures (all 300-DPI, grayscale-safe)

- **Figure 1** — validation gap (k-fold vs LOMO) vs duplicated fraction.
- **Figure 2** — per-seed LOMO MCC per model (MLP instability).
- **Figure 3** — forest plot of LOMO MCC with 95% bootstrap CIs.
- **Figure S1** — permutation null distribution of the XGBoost LOMO MCC.
- **Figure S2** — per-oxide held-out performance with n / cytotoxic counts.

Styling is centralized in `scripts/figstyle.py` — restyle everything in one place.


locked manuscript S5 values came from a specific signal-injection recipe, replace
numbers. All other stages reproduce the manuscript values directly.

## Reproducibility settings

Primary seed 42; stability seeds {0, 1, 2, 3, 7, 13, 42, 99}. Permutation and
bootstrap use 1,000 iterations. Software versions pinned in `requirements.txt`.
