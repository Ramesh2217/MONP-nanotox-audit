# Revision analyses (reviewer-requested)

This file documents nine analyses and supporting scripts used during revision.
Most read the same dataset (`data/NanoTox_dataset_clean.xlsx`) and write to
`results/`; the two figure generators (sections 8–9) take no data input. Run any of
them from the repository root with the virtual environment active. Sections 1–5 are
also invoked by `run_all.py` as part of the main pipeline; sections 6–9 are run
individually.

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

After running the revision analyses in this file, send the generated files in
`results/` (CSVs) and `results/figures/` (PNGs) so the manuscript numbers and
figures can be set to your machine's output.

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

---

The remaining scripts are standalone supporting analyses and figure generators.
They are run individually and are **not** invoked by `run_all.py`; none produces a
number cited in the manuscript.

## 6. Dose-baseline vs MLP comparison  (exploratory)

Compares the dose-threshold baseline and the multilayer perceptron under the same
leave-one-oxide-out scheme. Reports each model's pooled LOMO MCC with a percentile
bootstrap 95% CI (dose 0.4103, MLP 0.166 — the point estimates reported in the
manuscript), a check of whether the two intervals overlap, and a McNemar paired
test on the two models' correctness over the identical held-out instances.

```
python scripts/dose_vs_mlp_comparison.py
```

Output:
- `results/dose_vs_mlp_comparison.csv`  (dose_mcc, dose_ci, mlp_mcc, mlp_ci, ci_overlap, mcnemar_b, mcnemar_c, mcnemar_p)

The McNemar result (p = 0.0007) is **exploratory and is not reported in the
manuscript.** McNemar tests whether the two models' per-instance accuracy differs,
not whether their MCC differs; under this benchmark's class imbalance (68 cytotoxic
of 364) accuracy and MCC diverge, so a significant McNemar result does not establish
that the dose baseline's MCC advantage is statistically significant. It was
considered and deliberately excluded on that basis — the dose-vs-MLP comparison in
the paper rests on the point estimates and their confidence intervals, not on this
test.

## 7. MLP confidence interval with CuO removed  (Reviewer 1, point 4)

Recomputes the MLP's pooled LOMO MCC with the copper-oxide fold removed and adds a
percentile bootstrap 95% CI, using the same bootstrap as `bootstrap_ci.py` (1,000
resamples) restricted to the non-CuO folds. The point estimate (0.166 with all
oxides, 0.088 with CuO removed) is in the main text; this reports whether the
CuO-excluded interval spans zero — i.e. whether the residual, non-CuO signal is
distinguishable from chance.

```
python scripts/mlp_no_cuo_ci.py
```

Output:
- `results/mlp_no_cuo_ci.csv`  (set = all_oxides / CuO_removed; MCC, CI_low, CI_high)

## 8. Graphical abstract  (figure generator; no data input)

Draws the graphical abstract as a standalone summary figure: a three-bar MCC panel
(random five-fold ≈ 0.61, LOMO ≈ −0.12, dose baseline ≈ 0.41) alongside the key
findings. It takes no data input and computes nothing — the values shown are the
headline results, hardcoded for display.

```
python scripts/make_graphical_abstract.py
```

Output:
- `results/figures/Graphical_abstract.png`

## 9. Workflow diagram  (figure generator; no data input)

Draws Scheme 1, the unnumbered workflow overview of the benchmark audit and
validation pipeline. Like the graphical abstract it takes no data input; the boxes
summarize the pipeline and its headline numbers for display, sourced from the
analyses above rather than recomputed.

```
python scripts/make_workflow_figure.py
```

Output:
- `results/figures/Figure_workflow.png`
