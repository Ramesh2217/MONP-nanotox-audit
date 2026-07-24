# Revision analyses (reviewer-requested)

Two additional analyses were added during revision. Both read the same dataset
(`data/NanoTox_dataset_clean.xlsx`) and write to `results/`. Run them from the
repository root with the virtual environment active, exactly like the main
pipeline.

## 1. MLP permutation test  (reviewer comments 3 & 7)

Permutation test of leave-one-oxide-out performance for the multilayer
perceptron (the only model whose bootstrap CI lies above zero). Mirrors the
existing XGBoost permutation test.

```
python scripts/permutation_test_mlp.py
```

Outputs:
- `results/permutation_results_mlp.csv`  (observed MCC, null mean, null 95% interval, two-sided p, n_permutations)
- `results/figures/FigureS1b_permutation_null_mlp.png`

Full run uses 1,000 permutations and takes a few minutes.

## 2. Hyperparameter sensitivity  (reviewer comment 5)

LOMO MCC (with percentile bootstrap 95% CI) for a small reasonable grid of
settings for the random forest, SVM (RBF), and MLP. Descriptive robustness
check; not a model-selection procedure.

```
python scripts/hyperparameter_sensitivity.py
```

Output:
- `results/hyperparameter_sensitivity.csv`  (model, setting, LOMO_MCC, CI_low, CI_high, spans_zero)

Full run uses 1,000 bootstrap resamples per setting and takes a few minutes.

## What to send back

After running both, send the two CSVs (and the MLP figure) so the manuscript
numbers can be set to your machine's output:
- `results/permutation_results_mlp.csv`
- `results/hyperparameter_sensitivity.csv`

## 3. MLP per-oxide breakdown  (revision; verifies the interpretation)

Shows where the MLP's above-chance pooled score comes from, by reporting its
performance on each held-out oxide and recomputing the pooled MCC with the
copper-oxide fold removed.

```
python scripts/mlp_per_oxide_breakdown.py
```

Output:
- `results/mlp_per_oxide_breakdown.csv`

Expected (from the reference run): CuO fold 7/7 cytotoxic correct; pooled MCC
0.166 overall, 0.088 with the CuO fold removed — i.e. CuO is the largest single
contributor but not the sole source of the signal.

## 4. Learning curve  (reviewer priority 1)

Tests whether material-level performance improves with more training data, by
subsampling each LOMO fold's training set to a range of fractions.

```
python scripts/learning_curve.py
```

Outputs:
- `results/learning_curve.csv`
- `results/figures/Figure_learning_curve.png`  (manuscript Figure 3)

Expected (reference run): the LOMO MCC stays near chance and well below the dose
baseline (0.41) across all training-set sizes — performance plateaus, it does not
improve with more data of the same kind.

## 5. Oxide-diversity curve  (exploratory)

Complement to the learning curve: varies the NUMBER of distinct training oxide
classes (2, 3, 4) rather than the amount of data, to probe whether material
diversity helps cross-material transfer.

```
python scripts/oxide_diversity_curve.py
```

Outputs:
- `results/oxide_diversity_curve.csv`
- `results/figures/Figure_oxide_diversity.png`  (supplementary Figure S3)

IMPORTANT: this is exploratory only. With five oxides (two single-class), just
three oxides give defined test folds and combination counts are tiny (n=3 at the
top level), so SDs are large and intervals span zero. The mean trends gently
upward with diversity, but the benchmark cannot establish the effect. Report as a
motivation for larger/more diverse benchmarks, not as evidence.
