"""
bootstrap_ci.py — Step 5: bootstrap confidence intervals for MCC.

Estimates 95% confidence intervals for the LOMO MCC of each model and the
dose-threshold baseline by bootstrap resampling (1,000 resamples, with
replacement) of the pooled out-of-fold predictions, taking the 2.5th and 97.5th
percentiles of the resampled MCC distribution (manuscript Section 2.5).

Intervals spanning zero are reported as indistinguishable from chance and are
not ranked against one another.

Writes results/bootstrap_results.csv.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import matthews_corrcoef
from sklearn.base import clone

import common
import figstyle
from lomo_validation import build_models
from baseline_models import best_dose_threshold

N_BOOT = 1000


def lomo_predictions(model, X, y, groups):
    preds = np.empty_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = clone(model).fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return preds


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


def main(seed: int = common.RANDOM_SEED) -> None:
    df = common.load_analysis_frame()
    X = df[common.NINE_DESCRIPTORS].to_numpy(dtype=float)
    y = common.binary_labels(df).to_numpy()
    groups = df[common.OXIDE_COL].to_numpy()
    dose = df[common.DOSE_COL].to_numpy(dtype=float)

    rows = []

    # Dose-threshold baseline
    base_pred = np.empty_like(y)
    for tr, te in LeaveOneGroupOut().split(dose, y, groups):
        cut = best_dose_threshold(dose[tr], y[tr])
        base_pred[te] = (dose[te] >= cut).astype(int)
    mean, lo, hi = bootstrap_ci(y, base_pred, seed)
    rows.append({"model": "dose_threshold_baseline", "MCC_mean": round(mean, 4),
                 "CI_low": round(lo, 4), "CI_high": round(hi, 4),
                 "spans_zero": lo <= 0 <= hi})
    print(f"dose_threshold_baseline  MCC={mean:+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]")

    for name, model in build_models(seed).items():
        pred = lomo_predictions(model, X, y, groups)
        mean, lo, hi = bootstrap_ci(y, pred, seed)
        rows.append({"model": name, "MCC_mean": round(mean, 4),
                     "CI_low": round(lo, 4), "CI_high": round(hi, 4),
                     "spans_zero": lo <= 0 <= hi})
        print(f"{name:24s} MCC={mean:+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]")

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS_DIR / "bootstrap_results.csv"
    res_df = pd.DataFrame(rows)
    res_df.to_csv(out, index=False)
    print(f"\nWrote {out}")

    # Forest plot of LOMO MCC with 95% bootstrap CIs (Okabe-Ito, grayscale-safe)
    figstyle.apply()
    order = res_df.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for i, r in order.iterrows():
        spans = r["spans_zero"]
        col = "#E69F00" if spans else "#0072B2"
        mk = "o" if spans else "s"
        ax.plot([r["CI_low"], r["CI_high"]], [i, i], color=col, linewidth=2.4)
        ax.plot(r["MCC_mean"], i, marker=mk, color=col,
                markerfacecolor="white", markeredgecolor=col, markeredgewidth=1.8,
                markersize=8, zorder=3)
    ax.axvline(0.0, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([figstyle.label(m) for m in order["model"]])
    ax.set_xlabel("Leave-one-oxide-out MCC (95% bootstrap CI)")
    ax.set_xlim(-0.25, 0.55)
    out_fig = common.FIGURES_DIR / "Figure3_bootstrap_forest.png"
    figstyle.savefig(fig, out_fig)
    plt.close(fig)
    print("Wrote Figure 3.")


if __name__ == "__main__":
    main()
