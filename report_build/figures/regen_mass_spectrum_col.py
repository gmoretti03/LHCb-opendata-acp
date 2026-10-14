"""Data-section figure, one column: m(KKK) by B charge (log scale) and counting asymmetry in bins of the mass for the
two magnet polarities. Self-contained (ROOT files + shared preselection), ~10 s.

usage (cern env):  python regen_mass_spectrum.py [OUTDIR]
Also prints the candidate counts quoted in the Data section.
"""
import os, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

LO, HI, BIN = 5100.0, 5600.0, 10.0          # nominal fit window [MeV] and bin width
M_K = 493.677
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else HERE
REPO = subprocess.check_output(["git", "-C", HERE, "rev-parse", "--show-toplevel"], text=True).strip()
DATA = os.path.join(REPO, "LHCb", "opendata-project", "Data")

br = [f"H{i}_{v}" for i in (1, 2, 3) for v in ("PX", "PY", "PZ", "ProbK", "ProbPi", "isMuon", "Charge")]
frames = []
for pol in ("Down", "Up"):
    with uproot.open(os.path.join(DATA, f"B2HHH_Magnet{pol}.root")) as f:
        d = f["DecayTree"].arrays(br, library="pd")
    d["magnet"] = pol.lower()
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
n_read = len(df)
sel = np.ones(len(df), bool)
for i in (1, 2, 3):
    sel &= (df[f"H{i}_ProbPi"] < 0.5) & (df[f"H{i}_ProbK"] > 0.5) & (~df[f"H{i}_isMuon"].astype(bool))
df = df[sel].reset_index(drop=True)
p = sum(df[[f"H{i}_PX", f"H{i}_PY", f"H{i}_PZ"]].to_numpy() for i in (1, 2, 3))
e = sum(np.sqrt((df[[f"H{i}_PX", f"H{i}_PY", f"H{i}_PZ"]].to_numpy() ** 2).sum(1) + M_K ** 2) for i in (1, 2, 3))
m = np.sqrt(np.clip(e ** 2 - (p ** 2).sum(1), 0, None))
q = (df.H1_Charge + df.H2_Charge + df.H3_Charge).to_numpy()
w = (m > LO) & (m < HI) & np.isin(q, [1, -1])
m, q, mag = m[w], q[w], df.magnet.to_numpy()[w]
print(f"read {n_read}, preselected {sel.sum()}, in [{LO:.0f},{HI:.0f}]: {len(m)}  "
      f"B+ {np.sum(q > 0)}  B- {np.sum(q < 0)}  down {np.sum(mag == 'down')}  up {np.sum(mag == 'up')}  "
      f"A_count {(np.sum(q < 0) - np.sum(q > 0)) / len(q):+.4f}")

sys.path.insert(0, HERE)
import plot_style as ps
ps.use(); C = ps.COLORS
A_ALL = (np.sum(q < 0) - np.sum(q > 0)) / len(q)
fig, (a, b) = plt.subplots(2, 1, figsize=(ps.SINGLE, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.16})
bins = np.arange(LO, HI + 1e-9, BIN)
for sel_, lab, col in ((q > 0, r"$B^+$", C["pos"]), (q < 0, r"$B^-$", C["neg"])):
    a.hist(m[sel_], bins=bins, histtype="step", lw=1.6, color=col, label=lab)
a.axvspan(5200, 5350, color=C["shade"], alpha=0.6, lw=0, zorder=0, label="Signal region")
a.set_yscale("log"); a.set_ylim(8, 5000); a.set_ylabel(f"Candidates / {BIN:.0f} MeV")
a.legend(loc="upper right", fontsize=10); ps.letter(a, "a")
bw = 25.0
edges = np.arange(LO, HI + 1e-9, bw); cen = 0.5 * (edges[:-1] + edges[1:])
for pol, lab, col, off in (("down", "Magnet down", C["down"], -2.0), ("up", "Magnet up", C["up"], +2.0)):
    s_ = mag == pol
    nm, _ = np.histogram(m[s_ & (q < 0)], bins=edges); npp, _ = np.histogram(m[s_ & (q > 0)], bins=edges)
    n = nm + npp; ok = n >= 20
    A = (nm - npp) / np.maximum(n, 1); err = np.sqrt(np.clip(1 - A ** 2, 1e-9, None) / np.maximum(n, 1))
    b.errorbar(cen[ok] + off, A[ok], yerr=err[ok], fmt="o", ms=4, lw=1.1, capsize=0, color=col, label=lab)
b.axhline(A_ALL, color=C["count"], ls="--", lw=1.2, label=rf"All candidates: ${A_ALL:+.3f}$")
b.axhline(0, color="k", lw=0.8)
b.axvspan(5200, 5350, color=C["shade"], alpha=0.6, lw=0, zorder=0)
b.set_ylim(-0.5, 0.62); b.set_xlim(LO, HI)
b.set_xlabel(r"$m(K^+K^+K^-)$ [MeV]"); b.set_ylabel(r"$A_{\mathrm{count}}$ per 25 MeV")
b.legend(loc="upper center", fontsize=9, ncol=3, columnspacing=0.8, handletextpad=0.3, borderpad=0.3); ps.letter(b, "b")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "mass_spectrum_col.pdf"))
print("saved mass_spectrum_col.pdf")
