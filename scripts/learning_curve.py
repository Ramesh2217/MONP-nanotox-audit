"""
learning_curve.py — does material-level performance improve with more data?

Added in revision to address the reviewer question: would performance improve if
the dataset were larger? For a range of training-set sizes, we subsample the
training data within each leave-one-oxide-out (LOMO) fold, evaluate on the full
held-out oxide, and record the pooled MCC. Subsampling is repeated several times
per size and averaged to reduce noise. Both a representative machine-learning
model (XGBoost) and the dose-threshold baseline are tracked.

Interpretation:
  - If the curve plateaus below the dose baseline, additional data of the same
    kind is unlikely to yield material-level transfer for this benchmark.
  - If the curve continues to rise, more data may help, and the manuscript can
    say so.

Writes results/learning_curve.csv and results/figures/Figure_learning_curve.png.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import matthews_corrcoef
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

import common
import figstyle
from baseline_models import best_dose_threshold  # reuse the paper's baseline rule

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# training-set fractions to evaluate
FRACTIONS = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_REPEATS = 10  # subsampling repeats per fraction


def make_models(seed):
    models = {"RandomForest": RandomForestClassifier(random_state=seed)}
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=seed, verbosity=0)
    return models


def lomo_mcc_subsampled(model, X, y, groups, frac, rng):
    """Pooled LOMO MCC when each fold's TRAINING set is subsampled to `frac`."""
    preds = np.full_like(y, fill_value=-1)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if frac < 1.0:
            n_keep = max(2, int(round(len(tr) * frac)))
            sub = rng.choice(tr, size=n_keep, replace=False)
            # need both classes present to fit; if not, skip subsample draw
            if len(np.unique(y[sub])) < 2:
                # fall back to a stratified-ish keep: ensure at least one of each class
                pos = tr[y[tr] == 1]; neg = tr[y[tr] == 0]
                if len(pos) == 0 or len(neg) == 0:
                    sub = tr  # cannot stratify; use full fold training set
                else:
                    k_pos = max(1, int(round(len(pos) * frac)))
                    k_neg = max(1, n_keep - k_pos)
                    sub = np.concatenate([rng.choice(pos, min(k_pos, len(pos)), replace=False),
                                          rng.choice(neg, min(k_neg, len(neg)), replace=False)])
        else:
            sub = tr
        m = clone(model).fit(X[sub], y[sub])
        preds[te] = m.predict(X[te])
    return matthews_corrcoef(y, preds)


def main(seed: int = common.RANDOM_SEED) -> None:
    rng = np.random.default_rng(seed)
    df = common.load_analysis_frame()
    X = df[common.NINE_DESCRIPTORS].to_numpy(dtype=float)
    y = common.binary_labels(df).to_numpy()
    groups = df[common.OXIDE_COL].to_numpy()

    # dose baseline reference (full training each fold) for the horizontal line
    dose = df[common.DOSE_COL].to_numpy(dtype=float)
    dose_pred = np.empty_like(y)
    for tr, te in LeaveOneGroupOut().split(dose, y, groups):
        cut = best_dose_threshold(dose[tr], y[tr])
        dose_pred[te] = (dose[te] >= cut).astype(int)
    dose_mcc = matthews_corrcoef(y, dose_pred)

    rows = []
    models = make_models(seed)
    curves = {}
    for name, model in models.items():
        means, sds = [], []
        for frac in FRACTIONS:
            vals = []
            for r in range(N_REPEATS):
                vals.append(lomo_mcc_subsampled(model, X, y, groups, frac,
                                                np.random.default_rng(seed + r)))
            vals = np.array(vals)
            means.append(float(vals.mean())); sds.append(float(vals.std()))
            rows.append({"model": name, "fraction": frac,
                         "approx_train_n": int(round(len(y) * 0.8 * frac)),
                         "mean_LOMO_MCC": round(float(vals.mean()), 4),
                         "sd": round(float(vals.std()), 4)})
            print(f"{name:13s} frac={frac:.2f}  mean LOMO MCC={vals.mean():+.4f}  (SD {vals.std():.4f})")
        curves[name] = (np.array(means), np.array(sds))

    print(f"\nDose baseline LOMO MCC (reference): {dose_mcc:+.4f}")

    out_df = pd.DataFrame(rows)
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS_DIR / "learning_curve.csv"
    out_df["dose_baseline_MCC"] = round(dose_mcc, 4)
    out_df.to_csv(out, index=False)

    # plot
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    fr = np.array(FRACTIONS)
    style = {"XGBoost": ("-o", figstyle.OKABE_ITO["purple"]),
             "RandomForest": ("-s", figstyle.OKABE_ITO["vermillion"])}
    label = {"XGBoost": "XGBoost", "RandomForest": "Random forest"}
    for name, (mn, sd) in curves.items():
        ls, col = style.get(name, ("-o", figstyle.OKABE_ITO["blue"]))
        ax.plot(fr, mn, ls, color=col, label=f"{label.get(name,name)} (LOMO)")
        ax.fill_between(fr, mn - sd, mn + sd, color=col, alpha=0.12)
    ax.axhline(dose_mcc, color=figstyle.OKABE_ITO["green"], linestyle="--",
               linewidth=1.4, label=f"Dose baseline = {dose_mcc:+.2f}")
    ax.axhline(0.0, color="#888888", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Training fraction within each LOMO fold")
    ax.set_ylabel("LOMO MCC")
    ax.legend(loc="best", fontsize=8)
    figstyle.savefig(fig, common.FIGURES_DIR / "Figure_learning_curve.png")
    plt.close(fig)
    print(f"Wrote {out} and Figure_learning_curve.png.")


if __name__ == "__main__":
    main()
