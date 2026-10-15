"""Common figure style of the three notebooks (colours and font sizes only)."""
import matplotlib.pyplot as plt

COLORS = {"pos": "tab:blue", "neg": "tab:red",            # B+ / B-
          "down": "tab:green", "up": "tab:orange",        # magnet polarity
          "data": "k", "model": "crimson", "signal": "steelblue", "background": "gray",
          "freq": "#00897b", "bayes": "#a8412f", "ml": "#b8860b"}   # one colour per method


def use():
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 10,
                         "figure.figsize": (7, 4.2), "figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.25})
