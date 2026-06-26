"""
run_all.py — run the complete MONP benchmark audit pipeline end to end.

Executes every stage in order, writes all results to results/ and all
publication-grade figures (300 DPI, grayscale-safe) to results/figures/, then
runs a self-check that compares key computed values against the values reported
in the manuscript. Intended as the single entry point for a clean rerun.

Usage:
    python run_all.py
"""

import sys

import common
import duplicate_audit
import leakage_curve
import lomo_validation
import baseline_models
import bootstrap_ci
import permutation_test
import permutation_test_mlp
import seed_stability
import feature_decomposition
import per_oxide_folds
import hyperparameter_sensitivity
import mlp_per_oxide_breakdown
import learning_curve
import oxide_diversity_curve


EXPECTED = {
    "raw_records": 483,
    "unique_records": 364,
    "cytotoxic": 68,
}


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def main() -> None:
    banner("STEP 1/14  Duplicate audit")
    duplicate_audit.main()

    banner("STEP 2/14  Leakage curve (Figure 1)")
    leakage_curve.main()

    banner("STEP 3/14  LOMO validation (Table 1)")
    lomo_validation.main()

    banner("STEP 4/14  Dose-threshold baseline")
    baseline_models.main()

    banner("STEP 5/14  Bootstrap confidence intervals")
    bootstrap_ci.main()

    banner("STEP 6/14  Permutation test (XGBoost, Figure S1)")
    permutation_test.main()

    banner("STEP 7/14  Permutation test (MLP, Figure S1b)")
    permutation_test_mlp.main()

    banner("STEP 8/14  Seed stability (Figure 2)")
    seed_stability.main()

    banner("STEP 9/14  Feature-set decomposition (Table S4)")
    feature_decomposition.main()

    banner("STEP 10/14  Per-oxide LOMO folds (Table S3, Figure S2)")
    per_oxide_folds.main()

    banner("STEP 11/14  Hyperparameter sensitivity (Table S6)")
    hyperparameter_sensitivity.main()

    banner("STEP 12/14  MLP per-oxide breakdown")
    mlp_per_oxide_breakdown.main()

    banner("STEP 13/14  Learning curve (Figure 4)")
    learning_curve.main()

    banner("STEP 14/14  Oxide-diversity curve (Figure S3)")
    oxide_diversity_curve.main()

    banner("SELF-CHECK  (computed vs manuscript)")
    raw = common.load_raw()
    dedup = common.deduplicate(raw)
    y = common.binary_labels(dedup)

    checks = [
        ("raw_records", len(raw), EXPECTED["raw_records"], 0),
        ("unique_records", len(dedup), EXPECTED["unique_records"], 0),
        ("cytotoxic", int(y.sum()), EXPECTED["cytotoxic"], 0),
    ]

    ok = True
    for name, got, exp, tol in checks:
        status = "OK" if abs(got - exp) <= tol else "MISMATCH"
        ok = ok and status == "OK"
        print(f"  [{status:8s}] {name:18s} computed={got}  expected={exp}")

    print("\nAll figures (300 DPI) in:", common.FIGURES_DIR)
    print("All result tables in:    ", common.RESULTS_DIR)
    if ok:
        print("\nSelf-check PASSED — outputs reproduce the reported dataset composition.")
        print("Note: the MLP permutation test (step 7) and hyperparameter sweep "
              "(step 11) are the slowest stages; a full run takes roughly 30-60 "
              "minutes depending on the machine.")
    else:
        print("\nSelf-check found a MISMATCH — inspect the dataset path and column names.")
        sys.exit(1)


if __name__ == "__main__":
    main()
