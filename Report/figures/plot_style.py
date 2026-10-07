"""Common style for all the figures of the paper.

Figure widths are chosen so that, at the width used in main.tex (\\columnwidth for single panels,
0.95\\textwidth for figure*), the text in every figure comes out at about the same size (~7 pt).
"""
import matplotlib.pyplot as plt

SINGLE = 6.4          # inches, one-column figure (\columnwidth)
DOUBLE = 12.5         # two panels side by side (figure*, 0.95\textwidth)
TRIPLE = 13.5         # three panels side by side (figure*, 0.95\textwidth)

COLORS = {
    "pos": "tab:blue", "neg": "tab:red",              # B+ / B-
    "down": "tab:green", "up": "tab:orange",          # magnet polarity
    "data": "k", "model": "crimson", "signal": "steelblue", "background": "gray",
    "observed": "crimson", "replica": "steelblue",
}


def use():
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def panel_label(ax, text, loc="upper left"):
    """Short identification of a panel inside the axes (instead of a title)."""
    x, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    y, va = (0.95, "top") if "upper" in loc else (0.05, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))
