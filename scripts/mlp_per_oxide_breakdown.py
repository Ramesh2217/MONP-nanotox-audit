"""
mlp_per_oxide_breakdown.py — where does the MLP's above-chance signal come from?

Added in revision to verify the interpretation that the multilayer perceptron's
above-chance pooled leave-one-oxide-out (LOMO) score is driven largely by a
single held-out oxide rather than by broad cross-material transfer.

For each held-out oxide, this prints the held-out sample size, the number of
true cytotoxic instances, the number the MLP predicted cytotoxic, how many were
correct, and the per-fold accuracy. It also reports the pooled MCC and the
pooled MCC recomputed with the copper-oxide fold removed, so the contribution of
that single fold is explicit.

The MLP is configured identically to lomo_validation.build_models:
StandardScaler -> MLPClassifier(max_iter=2000, random_state=seed).

Writes results/mlp_per_oxide_breakdown.csv.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import matthews_corrcoef
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import common


def make_mlp(seed):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(max_iter=2000, random_state=seed),
    )


def main(seed: int = common.RANDOM_SEED) -> None:
    df = common.load_analysis_frame()
    X = df[common.NINE_DESCRIPTORS].to_numpy(dtype=float)
    y = common.binary_labels(df).to_numpy()
    groups = df[common.OXIDE_COL].to_numpy()

    model = make_mlp(seed)

    # collect pooled out-of-fold predictions and per-oxide rows
    pooled_pred = np.empty_like(y)
    rows = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        ox = groups[te][0]
        m = clone(model).fit(X[tr], y[tr])
        p = m.predict(X[te])
        pooled_pred[te] = p
        yt = y[te]
        n = len(yt)
        true_pos = int(yt.sum())
        pred_pos = int(p.sum())
        correct = int((p == yt).sum())
        # cytotoxic instances correctly identified in this fold
        tp = int(((p == 1) & (yt == 1)).sum())
        rows.append({"held_out_oxide": ox,
                     "n": n,
                     "true_cytotoxic": true_pos,
                     "predicted_cytotoxic": pred_pos,
                     "cytotoxic_correct": tp,
                     "fold_accuracy": round(correct / n, 3)})
        print(f"{ox:7s} n={n:3d}  true_cytotoxic={true_pos:3d}  "
              f"predicted_cytotoxic={pred_pos:3d}  cytotoxic_correct={tp:3d}  "
              f"acc={correct / n:.2f}")

    pooled_mcc = matthews_corrcoef(y, pooled_pred)

    # pooled MCC with the CuO fold removed
    cuo_mask = np.array([str(g).replace(" ", "").lower() not in ("cuo",) for g in groups])
    if cuo_mask.sum() < len(groups):
        mcc_no_cuo = matthews_corrcoef(y[cuo_mask], pooled_pred[cuo_mask])
    else:
        mcc_no_cuo = float("nan")

    print(f"\nPooled LOMO MCC (all oxides):      {pooled_mcc:+.4f}")
    print(f"Pooled LOMO MCC (CuO fold removed): {mcc_no_cuo:+.4f}")
    print("If the second value drops toward or below zero, the above-chance "
          "pooled score is largely attributable to the CuO fold.")

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS_DIR / "mlp_per_oxide_breakdown.csv"
    df_out = pd.DataFrame(rows)
    df_out["pooled_MCC_all"] = round(pooled_mcc, 4)
    df_out["pooled_MCC_no_CuO"] = round(float(mcc_no_cuo), 4)
    df_out.to_csv(out, index=False)
    print(f"\nWrote {out}.")


if __name__ == "__main__":
    main()
