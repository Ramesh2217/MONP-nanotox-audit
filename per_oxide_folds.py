"""
per_oxide_folds.py — Table S3 + Figure S2: per-oxide leave-one-oxide-out folds.

For the full XGBoost model, holds out each oxide in turn and reports the per-fold
MCC together with the held-out oxide's sample size (n) and number of cytotoxic
instances (manuscript Section 3, S3/S2).

Folds whose held-out set contains no cytotoxic instances (Al2O3, Fe2O3) are
undefined and reported as NaN / hatched. CuO and TiO2 folds rest on very few
cytotoxic instances (7 and 1) and are correspondingly unreliable. Per-oxide
values are shown for transparency only and are not interpreted or ranked.

Writes results/per_oxide_folds.csv (Table S3) and
results/figures/FigureS2_per_oxide_folds.png (Figure S2).
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


def main(seed: int = common.RANDOM_SEED) -> None:
    df = common.load_analysis_frame()
    X = df[common.NINE_DESCRIPTORS].to_numpy(dtype=float)
    y = common.binary_labels(df).to_numpy()
    groups = df[common.OXIDE_COL].to_numpy()

    rows = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        oxide = groups[te][0]
        n = len(te)
        n_tox = int(y[te].sum())
        if len(np.unique(y[te])) < 2:
            mcc = np.nan  # single-class held-out fold: undefined
        else:
            m = clone(make_model(seed)).fit(X[tr], y[tr])
            mcc = matthews_corrcoef(y[te], m.predict(X[te]))
        rows.append({"oxide": oxide, "n": n, "n_cytotoxic": n_tox,
                     "fold_MCC": None if np.isnan(mcc) else round(mcc, 4)})
        label = "undefined" if np.isnan(mcc) else f"{mcc:+.3f}"
        print(f"{oxide:6s}  n={n:4d}  tox={n_tox:3d}  MCC={label}")

    res = pd.DataFrame(rows)
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    common.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(common.RESULTS_DIR / "per_oxide_folds.csv", index=False)

    # Figure S2: per-oxide fold MCC. Undefined folds (single-class held-out set)
    # are drawn as clearly-marked hatched bars spanning the axis so they are
    # visibly distinguished from a genuine zero score.
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    xs = np.arange(len(res))
    drew_undef = False
    for i, r in res.iterrows():
        val = r["fold_MCC"]
        is_undef = val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val)
        if is_undef:
            # visible placeholder spanning full axis, hatched, light fill
            ax.bar(i, 1.6, bottom=-0.6, color="#f2f2f2",
                   edgecolor=figstyle.OKABE_ITO["orange"], hatch="////",
                   linewidth=1.0, zorder=1,
                   label="Undefined (no cytotoxic instances)" if not drew_undef else None)
            ax.annotate("undefined", (i, 0.0), ha="center", va="center",
                        fontsize=7.5, rotation=90, color=figstyle.OKABE_ITO["vermillion"])
            drew_undef = True
        else:
            ax.bar(i, val, color=figstyle.OKABE_ITO["blue"],
                   edgecolor="white", linewidth=0.8, zorder=2)
        ax.annotate(f"n={r['n']}\ntox={r['n_cytotoxic']}",
                    (i, -0.5), ha="center", va="bottom", fontsize=7,
                    color="#333333")
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([o.replace("2O3", "$_2$O$_3$").replace("O2", "O$_2$") if o not in ("CuO","ZnO") else o for o in res["oxide"]])
    ax.set_ylabel("Per-oxide leave-one-oxide-out MCC")
    ax.set_ylim(-0.6, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=7.5, ncol=1)
    figstyle.savefig(fig, common.FIGURES_DIR / "FigureS2_per_oxide_folds.png")
    plt.close(fig)
    print("Wrote per_oxide_folds.csv (Table S3) and Figure S2.")


if __name__ == "__main__":
    main()
