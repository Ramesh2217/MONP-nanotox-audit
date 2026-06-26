"""
permutation_test_mlp.py — permutation test of LOMO performance for the
multilayer perceptron (revision; Figure S1b).

Added in revision to address the reviewer's request to permutation-test the
multilayer perceptron specifically, since it is the only model whose bootstrap
confidence interval lay entirely above zero. The procedure mirrors
permutation_test.py exactly: the cytotoxicity labels are randomly shuffled 1,000
times, the full leave-one-oxide-out procedure is repeated for each shuffle to
build a null distribution of MCC values, and a two-sided p-value is computed as
the proportion of permuted scores at least as extreme as the observed score.

The MLP is configured identically to lomo_validation.build_models:
StandardScaler -> MLPClassifier(max_iter=2000, random_state=seed).

Writes results/permutation_results_mlp.csv and
results/figures/FigureS1b_permutation_null_mlp.png.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import matthews_corrcoef
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import common
import figstyle

N_PERM = 1000


def make_mlp(seed):
    # identical to lomo_validation.build_models["MLP"]
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(max_iter=2000, random_state=seed),
    )


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

    model = make_mlp(seed)
    observed = lomo_mcc(model, X, y, groups)

    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y)
        null[i] = lomo_mcc(model, X, yp, groups)
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{n_perm} permutations")

    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    lo, hi = np.percentile(null, [2.5, 97.5])
    print(f"Observed LOMO MCC (MLP): {observed:+.4f}")
    print(f"Permutation null mean:   {null.mean():+.4f}")
    print(f"Null 95% interval:       [{lo:+.3f}, {hi:+.3f}]")
    print(f"Two-sided p-value:       {p:.4f}")

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS_DIR / "permutation_results_mlp.csv"
    pd.DataFrame([{"model": "MLP",
                   "observed_MCC": round(observed, 4),
                   "null_mean": round(float(null.mean()), 4),
                   "null_ci_low": round(float(lo), 4),
                   "null_ci_high": round(float(hi), 4),
                   "p_value": round(p, 4),
                   "n_permutations": n_perm}]).to_csv(out, index=False)

    figstyle.apply()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.hist(null, bins=30, color=figstyle.OKABE_ITO["skyblue"],
            edgecolor="white", linewidth=0.4)
    ax.axvline(lo, color=figstyle.OKABE_ITO["orange"], linestyle="--", linewidth=1.2)
    ax.axvline(hi, color=figstyle.OKABE_ITO["orange"], linestyle="--", linewidth=1.2,
               label="Null 95% interval")
    ax.axvline(observed, color=figstyle.OKABE_ITO["vermillion"], linestyle="-", linewidth=2.0,
               label=f"Observed = {observed:+.3f}")
    ax.set_xlabel("Leave-one-oxide-out MCC under label permutation (MLP)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left")
    figstyle.savefig(fig, common.FIGURES_DIR / "FigureS1b_permutation_null_mlp.png")
    plt.close(fig)
    print(f"Wrote {out} and Figure S1b.")


if __name__ == "__main__":
    main()
