"""Shared figure styling, so every plot in the project reads as one system.

Colours were validated with the six-check palette test (lightness band, chroma
floor, colour-vision-deficiency separation, normal-vision floor, contrast):

    ['#00A693', '#C2410C']  ->  all checks PASS
    worst adjacent pair dE 11.8 under protanopia, 25.7 under normal vision.

Every figure is saved on its own. Multi-panel figures are convenient on screen
but useless in a report, where each panel needs its own caption and its own
place in the argument.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- categorical: identity, fixed order, never cycled ----------------------
SERIES = ["#00A693", "#C2410C"]      # Persian green, burnt orange
ACCENT = "#00A693"
ALERT = "#C2410C"

# --- ink: text never wears the series colour ------------------------------
INK = "#1A2422"
MUTED = "#5C6B68"
FAINT = "#B6C2BF"
SURFACE = "#FFFFFF"

# --- sequential: one hue, light -> dark (for confusion-matrix magnitude) ---
GREEN_RAMP = LinearSegmentedColormap.from_list(
    "persian", ["#FFFFFF", "#CFEBE6", "#7FCFC3", "#1FA593", "#00695C", "#00332C"]
)

DPI = 220        # print-quality without being enormous


def apply_style() -> None:
    """Recessive axes and grid, legible type, tabular figures."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": FAINT,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.color": "#EDF2F1",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,          # 2px lines
        "lines.markersize": 4,
        "text.color": INK,
    })


def save(fig, path: Path, note: str | None = None) -> Path:
    """Write one figure to one file.

    `note` is accepted and deliberately ignored: titles and footnotes are not
    drawn into the image. A figure in a report carries its caption in the
    document, where it can be typeset, numbered and referenced. Burning that
    text into the PNG duplicates it and fixes its size and language.
    """
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  wrote {path}")
    return path
