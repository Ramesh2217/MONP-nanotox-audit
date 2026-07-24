"""
hyperparameter_sensitivity.py — sensitivity of LOMO performance to
hyperparameter choice for the random forest, SVM (RBF), and multilayer
perceptron (revision; Table S6).

Added in revision to address the reviewer's request to show that the near-chance
leave-one-oxide-out (LOMO) performance is not an artifact of default model
settings. For each of the three flexible models, a small grid of reasonable
hyperparameter settings is evaluated under the same pooled LOMO procedure used
throughout the paper, and the LOMO MCC (with a percentile bootstrap 95% CI) is
reported for each setting. The point is descriptive: to show whether any
reasonable configuration moves LOMO performance materially above chance.

This is NOT used to select a "best" model (that would require nested,
material-level tuning to avoid leakage); it is a robustness check across a
plausible range.

Writes results/hyperparameter_sensitivity.csv (Table S6).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import matthews_corrcoef
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import common

N_BOOT = 1000


def lomo_predictions(model, X, y, groups):
    preds = np.empty_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = clone(model).fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return preds


def bootstrap_ci(y_true, y_pred, seed=common.RANDOM_SEED, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) > 1:
            scores.append(matthews_corrcoef(y_true[idx], y_pred[idx]))
    return (float(np.percentile(scores, 2.5)),
            float(np.percentile(scores, 97.5)))


def grids(seed):
    """A small, reasonable grid for each flexible model."""
    rf = [
        ("n_estimators=100, max_depth=None", RandomForestClassifier(n_estimators=100, random_state=seed)),
        ("n_estimators=300, max_depth=None", RandomForestClassifier(n_estimators=300, random_state=seed)),
        ("n_estimators=300, max_depth=5",    RandomForestClassifier(n_estimators=300, max_depth=5, random_state=seed)),
        ("n_estimators=500, max_depth=10",   RandomForestClassifier(n_estimators=500, max_depth=10, random_state=seed)),
        ("n_estimators=300, min_samples_leaf=5", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed)),
    ]
    svm = [
        ("C=0.1, gamma=scale",  make_pipeline(StandardScaler(), SVC(kernel="rbf", C=0.1, gamma="scale", random_state=seed))),
        ("C=1, gamma=scale",    make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed))),
        ("C=10, gamma=scale",   make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed))),
        ("C=1, gamma=0.1",      make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma=0.1, random_state=seed))),
        ("C=10, gamma=0.01",    make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma=0.01, random_state=seed))),
    ]
    mlp = [
        ("hidden=(50,), alpha=1e-4",      make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(50,), alpha=1e-4, max_iter=2000, random_state=seed))),
        ("hidden=(100,), alpha=1e-4",     make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100,), alpha=1e-4, max_iter=2000, random_state=seed))),
        ("hidden=(100,50), alpha=1e-4",   make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100, 50), alpha=1e-4, max_iter=2000, random_state=seed))),
        ("hidden=(100,), alpha=1e-2",     make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100,), alpha=1e-2, max_iter=2000, random_state=seed))),
        ("hidden=(100,), alpha=1.0",      make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100,), alpha=1.0, max_iter=2000, random_state=seed))),
    ]
    return {"RandomForest": rf, "SVM_RBF": svm, "MLP": mlp}


def main(seed: int = common.RANDOM_SEED) -> None:
    df = common.load_analysis_frame()
    X = df[common.NINE_DESCRIPTORS].to_numpy(dtype=float)
    y = common.binary_labels(df).to_numpy()
    groups = df[common.OXIDE_COL].to_numpy()

    rows = []
    for family, settings in grids(seed).items():
        for label, model in settings:
            preds = lomo_predictions(model, X, y, groups)
            mcc = matthews_corrcoef(y, preds)
            lo, hi = bootstrap_ci(y, preds, seed=seed)
            spans_zero = lo <= 0 <= hi
            rows.append({"model": family,
                         "setting": label,
                         "LOMO_MCC": round(mcc, 4),
                         "CI_low": round(lo, 4),
                         "CI_high": round(hi, 4),
                         "spans_zero": spans_zero})
            print(f"{family:13s} {label:32s} MCC={mcc:+.3f}  CI[{lo:+.3f},{hi:+.3f}]"
                  f"{'  (spans 0)' if spans_zero else ''}")

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS_DIR / "hyperparameter_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out} (Table S6).")
    # Summary line for convenience
    best = max(rows, key=lambda r: r["LOMO_MCC"])
    print(f"Best LOMO MCC across all settings: {best['LOMO_MCC']:+.3f} "
          f"({best['model']}, {best['setting']}); "
          f"CI {'spans zero' if best['spans_zero'] else 'excludes zero'}.")


if __name__ == "__main__":
    main()
