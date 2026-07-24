"""
dose_vs_mlp_comparison.py — dose-threshold baseline vs MLP under LOMO.

Adds, per dataset, the analyses requested in revision:
  (1) dose-threshold baseline LOMO MCC with a percentile bootstrap 95% CI;
  (2) MLP LOMO MCC with the same bootstrap CI;
  (3) a CI-overlap check between the two ("error overlap");
  (4) McNemar's paired test on their pooled out-of-fold (LOMO) predictions,
      which compares the two classifiers' error patterns on the SAME held-out
      instances (the statistically correct paired comparison).

NanoTox is loaded through common.py (NanoTox_dataset_clean.xlsx in data/).

All procedures reuse the paper's conventions: pooled out-of-fold predictions,
percentile bootstrap (1,000 resamples), seed = common.RANDOM_SEED, dose cut
chosen strictly within each training fold.

Usage:
    python dose_vs_mlp_comparison.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import matthews_corrcoef
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import common
from baseline_models import best_dose_threshold

N_BOOT = 1000


def make_mlp(seed):
    return make_pipeline(StandardScaler(), MLPClassifier(max_iter=2000, random_state=seed))


def bootstrap_ci(y_true, y_pred, seed, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(matthews_corrcoef(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(np.mean(scores)), float(lo), float(hi)


def mcnemar(y_true, pred_a, pred_b):
    """Paired McNemar test on two models' correctness over the same instances.
    Returns (b, c, statistic, p_value) where b = A-correct/B-wrong,
    c = A-wrong/B-correct. Uses exact binomial when discordants are few,
    else the continuity-corrected chi-square."""
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)
    b = int(np.sum(a_correct & ~b_correct))   # A right, B wrong
    c = int(np.sum(~a_correct & b_correct))   # A wrong, B right
    n = b + c
    if n == 0:
        return b, c, 0.0, 1.0
    if n < 25:
        # exact two-sided binomial test (p = 0.5)
        from scipy.stats import binomtest
        p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
        return b, c, float(min(b, c)), float(p)
    else:
        from scipy.stats import chi2
        stat = (abs(b - c) - 1) ** 2 / (b + c)  # continuity-corrected
        p = float(chi2.sf(stat, 1))
        return b, c, float(stat), p


def lomo_pooled(model, X, y, groups):
    pred = np.empty_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
    return pred


def dose_pooled(dose, y, groups):
    pred = np.empty_like(y)
    for tr, te in LeaveOneGroupOut().split(dose.reshape(-1, 1), y, groups):
        cut = best_dose_threshold(dose[tr], y[tr])
        pred[te] = (dose[te] >= cut).astype(int)
    return pred


def analyze(name, X, y, groups, dose, seed=None):
    seed = common.RANDOM_SEED if seed is None else seed
    print("\n" + "=" * 64)
    print(f"DATASET: {name}   (n={len(y)}, toxic={int(y.sum())}, oxides={len(set(groups))})")
    print("=" * 64)

    # pooled LOMO predictions
    mlp_pred = lomo_pooled(make_mlp(seed), X, y, groups)
    dz_pred = dose_pooled(dose, y, groups)

    mlp_mcc = matthews_corrcoef(y, mlp_pred)
    dz_mcc = matthews_corrcoef(y, dz_pred)
    mlp_m, mlp_lo, mlp_hi = bootstrap_ci(y, mlp_pred, seed)
    dz_m, dz_lo, dz_hi = bootstrap_ci(y, dz_pred, seed)

    print(f"Dose-threshold baseline LOMO MCC : {dz_mcc:+.4f}  95% CI [{dz_lo:+.4f}, {dz_hi:+.4f}]")
    print(f"MLP                  LOMO MCC : {mlp_mcc:+.4f}  95% CI [{mlp_lo:+.4f}, {mlp_hi:+.4f}]")

    # (3) CI overlap check
    overlap = not (dz_lo > mlp_hi or mlp_lo > dz_hi)
    print(f"\n(3) CI overlap (dose vs MLP): {'OVERLAP' if overlap else 'NO overlap'}")
    if not overlap:
        higher = "dose baseline" if dz_lo > mlp_hi else "MLP"
        print(f"    -> intervals are disjoint; {higher} is higher with non-overlapping 95% CIs.")
    else:
        print("    -> intervals overlap; difference not resolved by CI separation alone "
              "(see McNemar below for the paired test).")

    # (4) McNemar paired test on identical held-out instances
    b, c, stat, p = mcnemar(y, dz_pred, mlp_pred)
    print(f"\n(4) McNemar (paired, dose vs MLP on same LOMO predictions):")
    print(f"    discordant pairs: dose-right/MLP-wrong b={b}, dose-wrong/MLP-right c={c}")
    print(f"    statistic={stat:.4f}, two-sided p={p:.4f}  "
          f"({'significant' if p < 0.05 else 'not significant'} at 0.05)")
    return dict(dataset=name, dose_mcc=round(dz_mcc, 4), dose_ci=(round(dz_lo, 4), round(dz_hi, 4)),
                mlp_mcc=round(mlp_mcc, 4), mlp_ci=(round(mlp_lo, 4), round(mlp_hi, 4)),
                ci_overlap=overlap, mcnemar_b=b, mcnemar_c=c, mcnemar_p=round(p, 4))


def load_nanotox():
    df = common.load_analysis_frame()
    X = df[common.NINE_DESCRIPTORS].to_numpy(float)
    y = common.binary_labels(df).to_numpy()
    g = df[common.OXIDE_COL].to_numpy()
    dose = df[common.DOSE_COL].to_numpy(float)
    return X, y, g, dose


def main():
    results = []
    X, y, g, dose = load_nanotox()
    results.append(analyze("NanoTox (5 oxides)", X, y, g, dose))

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS_DIR / "dose_vs_mlp_comparison.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
