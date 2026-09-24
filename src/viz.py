"""Shared chart style for notebooks and exported figures.

Colors come from a colorblind-validated categorical palette. Seniority uses
the first three slots, which stay distinguishable in every pairing.
Color follows the entity: Entry is always blue, Mid orange, Senior aqua.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]
SENIORITY_ORDER = ["Entry", "Mid", "Senior"]
SENIORITY_COLORS = dict(zip(SENIORITY_ORDER, SERIES))
# Work type gets its own (also validated) trio so it never reads as seniority.
WORK_TYPE_ORDER = ["on-site", "hybrid", "remote"]
WORK_TYPE_COLORS = {"on-site": "#4a3aa7", "hybrid": "#e87ba4", "remote": "#008300"}
# Label ink that stays readable on each fill.
WORK_TYPE_TEXT = {"on-site": "white", "hybrid": TEXT_PRIMARY, "remote": "white"}

# Sequential blue ramp (light -> dark) for heatmaps.
BLUE_RAMP = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]
SEQUENTIAL = mpl.colors.LinearSegmentedColormap.from_list(
    "seq_blue", [SURFACE] + BLUE_RAMP
)


def set_style() -> None:
    """Thin marks, recessive grid and axes, no chart junk."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "text.color": TEXT_PRIMARY,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
        }
    )


def barh(ax, labels, values, color=BLUE, fmt="{:.0f}", label_values=True):
    """Horizontal bars, largest on top, with value labels at the bar ends."""
    labels, values = list(labels)[::-1], list(values)[::-1]
    bars = ax.barh(
        labels, values, color=color, height=0.72, edgecolor=SURFACE, linewidth=1
    )
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    if label_values:
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                " " + fmt.format(value),
                va="center",
                fontsize=8,
                color=TEXT_SECONDARY,
            )
    return bars


def save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path
