"""
leakage_curve.py — Step 2: the leakage gap as a function of duplication.

Reproduces the controlled-duplication analysis (manuscript Section 3.1). Starting
from the deduplicated dataset, records are duplicated at nine levels (duplicated
fractions 0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.55, 0.60). At each level the
MCC is computed under random five-fold and LOMO validation, and the leakage gap
(k-fold minus LOMO) is recorded.

The trend is summarized by an ordinary least-squares fit. NOTE: because the
duplication levels are generated from the same deduplicated dataset and are not
independent observations, this fit is a descriptive summary of the trend, not an
inferential test.

Writes results/leakage_curve.csv and results/figures/Figure1_leakage_curve.png.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import matthews_corrcoef
from sklearn.base import clone

import common
import figstyle

DUP_FRACTIONS = [0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.55, 0.60]


def pooled_mcc(model, X, y, splitter, groups=None) -> float:
    preds = np.empty_like(y)
    for tr, te in splitter.split(X, y, groups):
        m = clone(model).fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return matthews_corrcoef(y, preds)


def main(seed: int = common.RANDOM_SEED) -> None:
    rng = np.random.default_rng(seed)
    df = common.load_analysis_frame()
    X0 = df[common.NINE_DESCRIPTORS].to_numpy(dtype=float)
    y0 = common.binary_labels(df).to_numpy()
    g0 = df[common.OXIDE_COL].to_numpy()

    model = RandomForestClassifier(random_state=seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    logo = LeaveOneGroupOut()

    rows = []
    for frac in DUP_FRACTIONS:
        n_dup = int(round(frac * len(X0)))
        if n_dup > 0:
            idx = rng.integers(0, len(X0), size=n_dup)
            X = np.vstack([X0, X0[idx]])
            y = np.concatenate([y0, y0[idx]])
            g = np.concatenate([g0, g0[idx]])
        else:
            X, y, g = X0, y0, g0
        kf = pooled_mcc(model, X, y, skf)
        lomo = pooled_mcc(model, X, y, logo, g)
        rows.append({"dup_fraction": frac, "kfold_MCC": kf,
                     "LOMO_MCC": lomo, "leakage_gap": kf - lomo})
        print(f"frac={frac:.2f}  kfold={kf:+.3f}  LOMO={lomo:+.3f}  gap={kf-lomo:+.3f}")

    res = pd.DataFrame(rows)
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    common.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(common.RESULTS_DIR / "leakage_curve.csv", index=False)

    # Descriptive OLS fit (not inferential — see module docstring)
    slope, intercept = np.polyfit(res["dup_fraction"], res["leakage_gap"], 1)
    ss_res = np.sum((res["leakage_gap"] - (slope * res["dup_fraction"] + intercept)) ** 2)
    ss_tot = np.sum((res["leakage_gap"] - res["leakage_gap"].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"\nDescriptive OLS: slope={slope:.3f}, intercept={intercept:.3f}, R^2={r2:.3f}")

    figstyle.apply()
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    xs = res["dup_fraction"]
    ax.plot(xs, res["kfold_MCC"], marker="o", linestyle="-",
            color=figstyle.OKABE_ITO["blue"], label="Random $k$-fold")
    ax.plot(xs, res["LOMO_MCC"], marker="s", linestyle="--",
            color=figstyle.OKABE_ITO["vermillion"], label="Leave-one-oxide-out")
    ax.fill_between(xs, res["LOMO_MCC"], res["kfold_MCC"],
                    color=figstyle.OKABE_ITO["blue"], alpha=0.07)
    ax.axhline(0.0, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_xlim(-0.02, 0.62)
    ax.set_ylim(-0.25, 0.95)
    ax.set_xlabel("Fraction of duplicated records")
    ax.set_ylabel("Matthews correlation coefficient")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.55))
    figstyle.savefig(fig, common.FIGURES_DIR / "Figure1_leakage_curve.png")
    plt.close(fig)
    print(f"Wrote leakage_curve.csv and Figure 1.")


if __name__ == "__main__":
    main()
