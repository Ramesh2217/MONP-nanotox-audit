"""
lomo_validation.py — Step 3: leave-one-oxide-out (LOMO) evaluation.

Evaluates five classifiers (logistic regression, random forest, SVM-RBF,
multilayer perceptron, gradient boosting / XGBoost) under both random
five-fold and LOMO cross-validation, reporting the Matthews correlation
coefficient (MCC) for each.

Continuous descriptors are standardized with parameters estimated on the
training partition only, to avoid information leakage between train and test.

Writes results/classifier_results.csv.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import matthews_corrcoef

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

import common


def build_models(seed: int) -> dict:
    models = {
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
        ),
        "RandomForest": RandomForestClassifier(random_state=seed),
        "SVM_RBF": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", random_state=seed)
        ),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(max_iter=2000, random_state=seed),
        ),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            eval_metric="logloss", random_state=seed, verbosity=0
        )
    return models


def pooled_oof_mcc(model, X, y, splitter, groups=None) -> float:
    """Matthews correlation on pooled out-of-fold predictions."""
    preds = np.empty_like(y)
    for tr, te in splitter.split(X, y, groups):
        m = clone_fit(model, X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return matthews_corrcoef(y, preds)


def clone_fit(model, X, y):
    from sklearn.base import clone
    m = clone(model)
    m.fit(X, y)
    return m


def main(seed: int = common.RANDOM_SEED) -> None:
    df = common.load_analysis_frame()
    X = df[common.NINE_DESCRIPTORS].to_numpy(dtype=float)
    y = common.binary_labels(df).to_numpy()
    groups = df[common.OXIDE_COL].to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    logo = LeaveOneGroupOut()

    rows = []
    for name, model in build_models(seed).items():
        kf_mcc = pooled_oof_mcc(model, X, y, skf)
        lomo_mcc = pooled_oof_mcc(model, X, y, logo, groups)
        rows.append(
            {
                "model": name,
                "kfold_MCC": round(kf_mcc, 4),
                "LOMO_MCC": round(lomo_mcc, 4),
                "leakage_gap": round(kf_mcc - lomo_mcc, 4),
            }
        )
        print(f"{name:20s}  kfold={kf_mcc:+.4f}  LOMO={lomo_mcc:+.4f}")

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS_DIR / "classifier_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
