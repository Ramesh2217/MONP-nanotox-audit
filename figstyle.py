"""
figstyle.py — publication-grade, colorblind-safe figure styling.

Uses the Okabe-Ito palette (safe for all common types of color vision and in
grayscale print). Distinctions are carried by colour AND marker/line style.
Figures omit in-figure titles (the caption carries the title, per journal style),
use fixed axis ranges where appropriate, and render at 300 DPI.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
OKABE_ITO = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "skyblue":    "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
}
# Per-model colour assignment (stable across all figures)
MODEL_COLORS = {
    "LogisticRegression": OKABE_ITO["blue"],
    "RandomForest":       OKABE_ITO["vermillion"],
    "SVM_RBF":            OKABE_ITO["green"],
    "MLP":                OKABE_ITO["orange"],
    "XGBoost":            OKABE_ITO["purple"],
    "dose_threshold_baseline": OKABE_ITO["black"],
}
MODEL_MARKERS = {
    "LogisticRegression": "o",
    "RandomForest":       "s",
    "SVM_RBF":            "^",
    "MLP":                "D",
    "XGBoost":            "v",
    "dose_threshold_baseline": "P",
}
# Readable display labels (no code-style underscores)
MODEL_LABELS = {
    "LogisticRegression": "Logistic regression",
    "RandomForest":       "Random forest",
    "SVM_RBF":            "SVM (RBF)",
    "MLP":                "Multilayer perceptron",
    "XGBoost":            "XGBoost",
    "dose_threshold_baseline": "Dose-threshold baseline",
}

GREYS = ["#000000", "#4d4d4d", "#7f7f7f", "#a6a6a6", "#cccccc"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", ":"]
FIG_DPI = 300


def apply():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e8e8e8",
        "grid.linewidth": 0.6,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.5,
        "figure.constrained_layout.use": True,
    })


def label(model):
    return MODEL_LABELS.get(model, model)


def color(model):
    return MODEL_COLORS.get(model, GREYS[0])


def marker(model):
    return MODEL_MARKERS.get(model, "o")


def savefig(fig, path):
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    print("  figure saved:", path)
