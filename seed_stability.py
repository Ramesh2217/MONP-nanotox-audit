"""
seed_stability.py — Step 7: stability of LOMO results across random seeds.

Repeats the LOMO evaluation across eight random seeds {0, 1, 2, 3, 7, 13, 42, 99}
to assess how stable each model's performance is to the choice of seed
(manuscript Section 2.5). Models whose LOMO MCC varies widely across seeds
(notably the MLP) provide weaker evidence of genuine learned signal.

Writes results/seed_stability.csv.
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


def lomo_mcc(model, X, y, groups) -> float:
    preds = np.empty_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = clone(model).fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return matthews_corrcoef(y, preds)


def main() -> None:
    df = common.load_analysis_frame()
    X = df[common.NINE_DESCRIPTORS].to_numpy(dtype=float)
    y = common.binary_labels(df).to_numpy()
    groups = df[common.OXIDE_COL].to_numpy()

    rows = []
    per_seed = {}
    for name in build_models(common.RANDOM_SEED):
        scores = []
        for seed in common.STABILITY_SEEDS:
            model = build_models(seed)[name]
            scores.append(lomo_mcc(model, X, y, groups))
        scores = np.array(scores)
        per_seed[name] = scores
        rows.append({
            "model": name,
            "mean_MCC": round(float(scores.mean()), 4),
            "std_MCC": round(float(scores.std()), 4),
            "min_MCC": round(float(scores.min()), 4),
            "max_MCC": round(float(scores.max()), 4),
        })
        print(f"{name:24s} mean={scores.mean():+.3f}  "
              f"range=[{scores.min():+.3f}, {scores.max():+.3f}]")

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    common.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS_DIR / "seed_stability.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")

    # Publication-grade stability figure: per-seed LOMO MCC per model
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    names = list(per_seed.keys())
    for i, name in enumerate(names):
        xs = np.full(len(per_seed[name]), i, dtype=float)
        xs += np.linspace(-0.14, 0.14, len(xs))  # jitter
        ax.scatter(xs, per_seed[name],
                   facecolors=figstyle.color(name),
                   edgecolors="white", linewidths=0.6,
                   marker=figstyle.marker(name),
                   s=46, zorder=3)
        ax.plot([i - 0.24, i + 0.24],
                [per_seed[name].mean()] * 2,
                color="#333333", linewidth=1.6, zorder=2)
    ax.axhline(0.0, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([figstyle.label(n) for n in names], rotation=25, ha="right")
    ax.set_ylabel("Leave-one-oxide-out MCC across seeds")
    ax.set_ylim(-0.18, 0.32)
    figstyle.savefig(fig, common.FIGURES_DIR / "Figure2_seed_stability.png")
    plt.close(fig)
    print("Wrote Figure 2.")


if __name__ == "__main__":
    main()
