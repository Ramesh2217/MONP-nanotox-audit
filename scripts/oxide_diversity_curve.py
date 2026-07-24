"""
oxide_diversity_curve.py — does cross-material performance improve with more
training-material diversity?

Added in revision to address the reviewer question: would adding more material
diversity help cross-material generalization? This is the complement to the
learning-curve analysis (which varies the amount of training data but not its
material diversity).

Design: for each held-out oxide (the leave-one-oxide-out test target), the model
is trained on subsets of the REMAINING oxides of increasing size (2, 3, then all
4 available training oxides). For each training-diversity level we enumerate (or
sample) the combinations of training oxides of that size, fit the model, predict
the held-out oxide, and record the MCC. Results are pooled/averaged across held-
out oxides and across training-oxide combinations at each diversity level.

Interpretation:
  - If MCC rises with the number of training oxides, more material diversity may
    help, and the manuscript can say so.
  - If MCC stays flat/near chance, added diversity (within the oxides available)
    does not resolve the lack of cross-material transfer for this benchmark.

Note: with only five oxide classes, two of which contain no cytotoxic instances,
the achievable diversity is limited; this analysis is descriptive and bounded by
the benchmark's composition.

Writes results/oxide_diversity_curve.csv and
results/figures/Figure_oxide_diversity.png.
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

import common
import figstyle

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def make_models(seed):
    models = {"RandomForest": RandomForestClassifier(random_state=seed)}
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=seed, verbosity=0)
    return models


def evaluate(model, X, y, groups):
    """For each held-out oxide and each training-oxide subset size, record MCC."""
    oxides = list(pd.unique(groups))
    rows = []
    for held in oxides:
        test_mask = groups == held
        if test_mask.sum() == 0:
            continue
        # the held-out test set must contain both classes to define an MCC;
        # if it is single-class, the fold MCC is undefined -> skip (documented)
        if len(np.unique(y[test_mask])) < 2:
            continue
        others = [o for o in oxides if o != held]
        for k in range(2, len(others) + 1):
            for combo in itertools.combinations(others, k):
                tr_mask = np.isin(groups, combo)
                if len(np.unique(y[tr_mask])) < 2:
                    continue  # need both classes in training
                m = clone(model).fit(X[tr_mask], y[tr_mask])
                pred = m.predict(X[test_mask])
                mcc = matthews_corrcoef(y[test_mask], pred) if len(np.unique(pred)) > 1 else 0.0
                rows.append({"held_out": held, "n_train_oxides": k, "MCC": mcc})
    return pd.DataFrame(rows)


def main(seed: int = common.RANDOM_SEED) -> None:
    df = common.load_analysis_frame()
    X = df[common.NINE_DESCRIPTORS].to_numpy(dtype=float)
    y = common.binary_labels(df).to_numpy()
    groups = df[common.OXIDE_COL].to_numpy()

    figstyle.apply()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    style = {"XGBoost": ("-o", figstyle.OKABE_ITO["purple"]),
             "RandomForest": ("-s", figstyle.OKABE_ITO["vermillion"])}
    label = {"XGBoost": "XGBoost", "RandomForest": "Random forest"}

    all_rows = []
    for name, model in make_models(seed).items():
        res = evaluate(model, X, y, groups)
        res["model"] = name
        all_rows.append(res)
        summary = res.groupby("n_train_oxides")["MCC"].agg(["mean", "std", "count"])
        print(f"\n{name}:")
        print(summary)
        ks = summary.index.to_numpy()
        mn = summary["mean"].to_numpy()
        sd = np.nan_to_num(summary["std"].to_numpy())
        ls, col = style.get(name, ("-o", figstyle.OKABE_ITO["blue"]))
        ax.plot(ks, mn, ls, color=col, label=f"{label.get(name,name)}")
        ax.fill_between(ks, mn - sd, mn + sd, color=col, alpha=0.12)

    ax.axhline(0.0, color="#888888", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Number of oxide classes in training set")
    ax.set_ylabel("LOMO MCC (held-out oxide)")
    ax.set_xticks([2, 3, 4])
    ax.legend(loc="best", fontsize=8)
    figstyle.savefig(fig, common.FIGURES_DIR / "Figure_oxide_diversity.png")
    plt.close(fig)

    out_df = pd.concat(all_rows, ignore_index=True)
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS_DIR / "oxide_diversity_curve.csv"
    out_df.to_csv(out, index=False)
    print(f"\nWrote {out} and Figure_oxide_diversity.png.")
    print("Note: held-out oxides with no cytotoxic instances (Al2O3, Fe2O3) are "
          "skipped as single-class folds; results reflect ZnO, TiO2, and CuO as "
          "achievable test targets.")


if __name__ == "__main__":
    main()
