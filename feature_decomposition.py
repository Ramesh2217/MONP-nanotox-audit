"""
feature_decomposition.py — Table S4: feature-set decomposition under LOMO.

Evaluates the XGBoost model under leave-one-oxide-out validation using three
feature sets (manuscript Section 3, S4):

  * full        — all nine descriptors
  * dose only   — the dose descriptor alone
  * no dose     — the eight descriptors excluding dose

Each is reported as MCC with a 95% bootstrap CI. All subsets produce below- or
near-chance cross-material performance, indicating that no subset of the
available descriptors supports above-chance cross-material prediction.

Writes results/feature_decomposition.csv (Table S4).
"""

import numpy as np
import pandas as pd
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
    y = common.binary_labels(df).to_numpy()
    groups = df[common.OXIDE_COL].to_numpy()

    feature_sets = {
        "full": common.NINE_DESCRIPTORS,
        "dose_only": [common.DOSE_COL],
        "no_dose": [c for c in common.NINE_DESCRIPTORS if c != common.DOSE_COL],
    }

    rows = []
    for name, cols in feature_sets.items():
        X = df[cols].to_numpy(dtype=float)
        pred = lomo_predictions(make_model(seed), X, y, groups)
        mean, lo, hi = bootstrap_ci(y, pred, seed)
        rows.append({"feature_set": name, "n_features": len(cols),
                     "MCC_mean": round(mean, 4),
                     "CI_low": round(lo, 4), "CI_high": round(hi, 4),
                     "spans_zero": lo <= 0 <= hi})
        print(f"{name:10s} ({len(cols)} feat)  MCC={mean:+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]")

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS_DIR / "feature_decomposition.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}  (Table S4)")


if __name__ == "__main__":
    main()
