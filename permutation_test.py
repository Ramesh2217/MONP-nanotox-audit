"""
permutation_test.py — Step 6: permutation test of LOMO performance (Figure S1).

Assesses whether the observed leave-one-oxide-out performance differs from chance.
The cytotoxicity labels are randomly shuffled 1,000 times; the full LOMO procedure
is repeated for each shuffle to build a null distribution of MCC values, and a
two-sided p-value is computed as the proportion of permuted scores at least as
extreme as the observed score (manuscript Section 2.5, Figure S1).

The XGBoost model is used, matching the supplementary Figure S1 caption.

Writes results/permutation_results.csv and
results/figures/FigureS1_permutation_null.png (Figure S1).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import matthews_corrcoef
from sklearn.base import clone

try:
    from xgboost import XGBClassifier
    def make_model(seed):
        return XGBClassifier(eval_metric="logloss", random_state=seed, verbosity=0)
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    def make_model(seed):
        return GradientBoostingClassifier(random_state=seed)

import common
import figstyle

N_PERM = 1000


def lomo_mcc(model, X, y, groups) -> float:
    preds = np.empty_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = clone(model).fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return matthews_corrcoef(y, preds)


def main(seed: int = common.RANDOM_SEED, n_perm: int = N_PERM) -> None:
    rng = np.random.default_rng(seed)
    df = common.load_analysis_frame()
    X = df[common.NINE_DESCRIPTORS].to_numpy(dtype=float)
    y = common.binary_labels(df).to_numpy()
    groups = df[common.OXIDE_COL].to_numpy()

    model = make_model(seed)
    observed = lomo_mcc(model, X, y, groups)

    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y)
        null[i] = lomo_mcc(model, X, yp, groups)

    # two-sided p-value
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    lo, hi = np.percentile(null, [2.5, 97.5])
    print(f"Observed LOMO MCC (XGBoost): {observed:+.4f}")
    print(f"Permutation null mean: {null.mean():+.4f}")
    print(f"Null 95% interval: [{lo:+.3f}, {hi:+.3f}]")
    print(f"Two-sided p-value: {p:.4f}")

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS_DIR / "permutation_results.csv"
    pd.DataFrame([{"observed_MCC": round(observed, 4),
                   "null_mean": round(float(null.mean()), 4),
                   "null_ci_low": round(float(lo), 4),
                   "null_ci_high": round(float(hi), 4),
                   "p_value": round(p, 4),
                   "n_permutations": n_perm}]).to_csv(out, index=False)

    # Figure S1: permutation null distribution with observed value
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.hist(null, bins=30, color=figstyle.OKABE_ITO["skyblue"],
            edgecolor="white", linewidth=0.4)
    ax.axvline(lo, color=figstyle.OKABE_ITO["orange"], linestyle="--", linewidth=1.2)
    ax.axvline(hi, color=figstyle.OKABE_ITO["orange"], linestyle="--", linewidth=1.2,
               label="Null 95% interval")
    ax.axvline(observed, color=figstyle.OKABE_ITO["vermillion"], linestyle="-", linewidth=2.0,
               label=f"Observed = {observed:+.3f}")
    ax.set_xlabel("Leave-one-oxide-out MCC under label permutation")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left")
    figstyle.savefig(fig, common.FIGURES_DIR / "FigureS1_permutation_null.png")
    plt.close(fig)
    print(f"Wrote {out} and Figure S1.")


if __name__ == "__main__":
    main()
