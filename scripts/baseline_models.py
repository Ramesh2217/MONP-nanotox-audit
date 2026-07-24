"""
baseline_models.py — Step 4: single-variable dose-threshold baseline (LOMO).

Implements the trivial dose-threshold rule used as a reference under matched
leave-one-oxide-out evaluation (manuscript Section 3.2). For each LOMO fold the
dose cut that maximizes the training-partition MCC is selected and applied to
the held-out oxide. The pooled out-of-fold MCC is reported.

On the released benchmark this baseline achieves a LOMO MCC of approximately
0.41, performing comparably to or better than the five machine-learning models.

Writes results/baseline_results.csv.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import matthews_corrcoef

import common


def best_dose_threshold(dose: np.ndarray, y: np.ndarray):
    """Return the dose cut maximizing MCC on (dose, y); predict y=1 if dose>=cut."""
    best_cut, best_mcc = None, -np.inf
    for cut in np.unique(dose):
        pred = (dose >= cut).astype(int)
        mcc = matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0
        if mcc > best_mcc:
            best_mcc, best_cut = mcc, cut
    return best_cut


def main() -> None:
    df = common.load_analysis_frame()
    dose = df[common.DOSE_COL].to_numpy(dtype=float)
    y = common.binary_labels(df).to_numpy()
    groups = df[common.OXIDE_COL].to_numpy()

    logo = LeaveOneGroupOut()
    preds = np.empty_like(y)
    for tr, te in logo.split(dose, y, groups):
        cut = best_dose_threshold(dose[tr], y[tr])
        preds[te] = (dose[te] >= cut).astype(int)

    mcc = matthews_corrcoef(y, preds)
    print(f"Dose-threshold baseline LOMO MCC: {mcc:+.4f}")

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS_DIR / "baseline_results.csv"
    pd.DataFrame([{"model": "dose_threshold_baseline", "LOMO_MCC": round(mcc, 4)}]).to_csv(
        out, index=False
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
