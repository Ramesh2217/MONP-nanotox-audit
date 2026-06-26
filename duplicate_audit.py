"""
duplicate_audit.py — Step 1: audit and remove exact duplicate records.

Reproduces the deduplication reported in the manuscript (Section 2.1):
the raw benchmark (483 records) is reduced to 364 unique instances
(68 cytotoxic, 18.7%) by removing 119 exact duplicates defined over the nine
descriptors plus oxide identity and class label, excluding viability.

Writes a short summary to results/duplicate_audit_summary.csv.
"""

import pandas as pd

import common


def main() -> None:
    raw = common.load_raw()
    dedup = common.deduplicate(raw)

    y_raw = common.binary_labels(raw)
    y = common.binary_labels(dedup)

    n_raw = len(raw)
    n_unique = len(dedup)
    n_removed = n_raw - n_unique
    n_pos = int(y.sum())

    print(f"Raw records:               {n_raw}")
    print(f"Exact duplicates removed:  {n_removed}")
    print(f"Unique records:            {n_unique}")
    print(f"Cytotoxic (Toxic):         {n_pos}  ({100 * y.mean():.1f}%)")
    print(f"Non-toxic:                 {n_unique - n_pos}")
    print()
    print("Per-oxide composition (unique records):")
    comp = (
        dedup.assign(_y=y)
        .groupby(common.OXIDE_COL)["_y"]
        .agg(n="size", positives="sum")
    )
    comp["pct_positive"] = (100 * comp["positives"] / comp["n"]).round(1)
    print(comp.to_string())

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        {
            "metric": [
                "raw_records",
                "duplicates_removed",
                "unique_records",
                "cytotoxic",
                "non_toxic",
                "percent_cytotoxic",
            ],
            "value": [
                n_raw,
                n_removed,
                n_unique,
                n_pos,
                n_unique - n_pos,
                round(100 * y.mean(), 2),
            ],
        }
    )
    out = common.RESULTS_DIR / "duplicate_audit_summary.csv"
    summary.to_csv(out, index=False)
    comp.to_csv(common.RESULTS_DIR / "per_oxide_composition.csv")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
