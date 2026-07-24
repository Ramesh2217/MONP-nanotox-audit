"""
mlp_no_cuo_ci.py — bootstrap CI for the MLP's pooled LOMO MCC with the CuO
fold removed (added in revision, Reviewer 1 point 4).

Reviewer 1 asked that the confidence interval for the CuO-excluded MLP result
appear in the main text, since the point estimate alone (0.166 -> 0.088) does
not show whether the residual signal is distinguishable from chance.

This uses exactly the same percentile bootstrap as bootstrap_ci.py: 1,000
resamples with replacement of the pooled out-of-fold predictions, 2.5th/97.5th
percentiles. The only change is that the pooled vectors are restricted to the
non-CuO folds before resampling.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import matthews_corrcoef
from sklearn.base import clone

import common
from mlp_per_oxide_breakdown import make_mlp

N_BOOT = 1000


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
    return float(np.mean(scores)), float(lo), float(hi), len(scores)


def main(seed: int = common.RANDOM_SEED) -> None:
    df = common.load_analysis_frame()
    X = df[common.NINE_DESCRIPTORS].to_numpy(dtype=float)
    y = common.binary_labels(df).to_numpy()
    groups = df[common.OXIDE_COL].to_numpy()

    model = make_mlp(seed)
    pooled = np.empty_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = clone(model).fit(X[tr], y[tr])
        pooled[te] = m.predict(X[te])

    is_cuo = np.array([str(g).replace(" ", "").lower() == "cuo" for g in groups])
    keep = ~is_cuo

    mcc_all = matthews_corrcoef(y, pooled)
    mcc_no_cuo = matthews_corrcoef(y[keep], pooled[keep])

    mean_all, lo_all, hi_all, n_all = bootstrap_ci(y, pooled, seed)
    mean_nc, lo_nc, hi_nc, n_nc = bootstrap_ci(y[keep], pooled[keep], seed)

    print(f"Pooled LOMO MCC (all oxides):       {mcc_all:+.4f}  "
          f"95% CI [{lo_all:+.4f}, {hi_all:+.4f}]  (n_valid={n_all})")
    print(f"Pooled LOMO MCC (CuO removed):      {mcc_no_cuo:+.4f}  "
          f"95% CI [{lo_nc:+.4f}, {hi_nc:+.4f}]  (n_valid={n_nc})")
    spans_zero = lo_nc <= 0 <= hi_nc
    print(f"\nCuO-excluded interval spans zero: {spans_zero}")
    print("Interpretation: if it spans zero, the residual (non-CuO) signal is "
          "indistinguishable from chance, and the pooled above-chance result is "
          "attributable substantially to the CuO fold.")

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS_DIR / "mlp_no_cuo_ci.csv"
    pd.DataFrame([
        {"set": "all_oxides", "MCC": round(mcc_all, 4),
         "CI_low": round(lo_all, 4), "CI_high": round(hi_all, 4)},
        {"set": "CuO_removed", "MCC": round(mcc_no_cuo, 4),
         "CI_low": round(lo_nc, 4), "CI_high": round(hi_nc, 4)},
    ]).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
