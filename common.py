"""
common.py — shared data loading and preprocessing for the MONP benchmark audit.

All scripts import from this module so that the dataset path, descriptor set,
and deduplication rule are defined in exactly one place.

The dataset (`dataset.txt`) is NOT distributed with this repository. Download it
from the NanoTox repository and place it at data/dataset.txt
(see data/README.md).
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
# The pipeline auto-detects file type. Place your dataset in the data/ folder.
# Default: the cleaned 24-column Excel file. (A tab-separated dataset.txt also works.)
def _resolve_data_path() -> Path:
    """Find the dataset in data/, accepting either the tab-separated dataset.txt
    or the cleaned Excel file. Users only need to drop the downloaded file in
    the data/ folder (see data/README.md)."""
    candidates = [
        REPO_ROOT / "data" / "dataset.txt",
        REPO_ROOT / "data" / "NanoTox_dataset_clean.xlsx",
        REPO_ROOT / "data" / "NanoTox.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # default for the error message


DATA_PATH = _resolve_data_path()
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# The nine descriptors used for modelling (manuscript Section 2.2).
NINE_DESCRIPTORS = [
    "coresize",    # core size
    "hydrosize",   # hydrodynamic size
    "surfcharge",  # surface charge
    "surfarea",    # surface area
    "Ec",          # conduction band energy
    "Expotime",    # exposure time
    "dosage",      # dose
    "e",           # electronegativity
    "NOxygen",     # number of oxygen atoms
]

OXIDE_COL = "NPs"        # oxide identity (LOMO grouping variable)
LABEL_COL = "class"      # binary label: 'Toxic' / 'nonToxic'
VIABILITY_COL = "viability"
DOSE_COL = "dosage"

RANDOM_SEED = 42
STABILITY_SEEDS = [0, 1, 2, 3, 7, 13, 42, 99]


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw NanoTox benchmark (auto-detects .xlsx or tab-separated .txt)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Place the dataset in the data/ folder. Either the cleaned Excel file "
            "(NanoTox_dataset_clean.xlsx) or the tab-separated dataset.txt from the "
            "NanoTox repository will work (see data/README.md)."
        )
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
        if df.shape[1] == 1:  # single-column tab-packed fallback
            header = str(df.columns[0]).split("\t")
            rows = [str(v).split("\t") for v in df.iloc[:, 0]]
            df = pd.DataFrame(rows, columns=header)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)  # comma-separated
    else:
        df = pd.read_csv(path, sep="\t")  # dataset.txt is tab-separated
    df.columns = [c.strip() for c in df.columns]
    # ensure numeric columns are numeric (xlsx usually already is)
    for c in df.columns:
        if c not in (OXIDE_COL, LABEL_COL, "Cellline", "Celltype"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate records.

    A duplicate is defined as a record identical to another across all nine
    descriptors together with the oxide identity and the cytotoxicity label;
    the continuous viability value is NOT used in this comparison
    (manuscript Section 2.1).

    On the released NanoTox benchmark this reduces 483 raw records to 364
    unique instances (68 cytotoxic).
    """
    key = [OXIDE_COL] + NINE_DESCRIPTORS + [LABEL_COL]
    return df.drop_duplicates(subset=key).reset_index(drop=True)


def binary_labels(df: pd.DataFrame) -> pd.Series:
    """Return the binary label as 1 (Toxic) / 0 (nonToxic)."""
    return (df[LABEL_COL].astype(str).str.strip() == "Toxic").astype(int)


def load_analysis_frame() -> pd.DataFrame:
    """Load, deduplicate, and return the analysis-ready frame."""
    return deduplicate(load_raw())


if __name__ == "__main__":
    raw = load_raw()
    dedup = deduplicate(raw)
    y = binary_labels(dedup)
    print(f"Raw records:        {len(raw)}")
    print(f"Unique records:     {len(dedup)}")
    print(f"Cytotoxic (Toxic):  {int(y.sum())}  ({100 * y.mean():.1f}%)")
