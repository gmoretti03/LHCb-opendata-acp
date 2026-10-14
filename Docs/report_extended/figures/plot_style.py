"""Common style for all the figures of the paper.

Figure widths are chosen so that, at the width used in main.tex (\\columnwidth for single panels,
0.95\\textwidth for figure*), the text in every figure comes out at about the same size (~7 pt).
"""
import os
import matplotlib.pyplot as plt


# ---- where the analysis code, its saved results and the data are: found by walking up from this folder (the LHCb
# folder is the first one that contains both Code_general and opendata-project), so the figures folder can be moved.
def _find_lhcb(start):
    d = os.path.abspath(start)
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "Code_general")) and os.path.isdir(os.path.join(d, "opendata-project")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("LHCb folder not found above " + start + "; set LHCB_DIR")


LHCB = os.environ.get("LHCB_DIR") or _find_lhcb(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(LHCB, "Code_general", "Analysis")                        # Bayesian/, Frequentist/, Machine_Learning/
DATA = os.environ.get("LHCB_DATA_DIR", os.path.join(LHCB, "opendata-project", "Data"))

SINGLE = 6.4          # inches, one-column figure (\columnwidth)
DOUBLE = 12.5         # two panels side by side (figure*, 0.95\textwidth)
TRIPLE = 13.5         # three panels side by side (figure*, 0.95\textwidth)

# One colour per role; a colour is not reused with another meaning across figures.
COLORS = {
    # charge and magnet polarity (mass spectra, splits)
    "pos": "tab:blue", "neg": "tab:red",              # B+ / B-
    "down": "tab:green", "up": "tab:orange",          # magnet polarity
    # fit components
    "data": "k", "model": "#1a2f5a", "signal": "steelblue", "background": "gray",
    "observed": "k", "replica": "steelblue",
    # inference methods (comparison figures)
    "bayes": "#a8412f",                               # Bayesian (brick red)
    "freq": "#00897b",                                # frequentist (teal)
    "count": "#555555",                               # counting (dark grey)
    "ml": "#b8860b",                                  # machine learning / score cut (dark goldenrod)
    # weak-label classes (ML figures), distinct from B+/B-
    "sig_like": "#5b7c99", "bkg_like": "#d9a05b",
    # mass-window shading: grey, hatched for the excluded region
    "shade": "0.85",
}
# models of the ML module, one colour each in every ML figure
MODELS = {"Logistic Regression": "0.45", "Logistic Regression (L1)": "0.45", "Decision Tree": "#9a9a1a",
          "Random Forest": "#1b9e77", "Gradient Boosting": "#b8860b", "FFNN": "#e7298a"}
ESTIMATORS = {"rf": "#1b9e77", "mc": "#e7298a", "de": "#8c564b"}   # uncertainty estimators


def sci(x, digits=1):
    """LaTeX scientific notation, e.g. 1.1\\times10^{-4}, for labels."""
    if x == 0:
        return "0"
    import math
    e = int(math.floor(math.log10(abs(x))))
    return f"{x / 10**e:.{digits}f}\\times10^{{{e}}}"


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


def letter(ax, text):
    """Panel letter, (a), (b), ... above the top-left corner of the axes; the caption refers to it."""
    ax.text(0.0, 1.02, f"({text})", transform=ax.transAxes, ha="left", va="bottom", fontsize=14, fontweight="bold")
