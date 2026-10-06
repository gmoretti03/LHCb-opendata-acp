"""Data-section figure: reconstructed m(KKK) in the nominal fit window, split by B charge and by magnet
polarity. Self-contained (ROOT files + shared preselection), ~10 s.

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

LO, HI, BIN = 5100.0, 5600.0, 5.0          # nominal fit window [MeV] and bin width
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

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
bins = np.arange(LO, HI + 1e-9, BIN)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for sel_, lab, col in ((q > 0, r"$B^+$", "tab:blue"), (q < 0, r"$B^-$", "tab:red")):
    axs[0].hist(m[sel_], bins=bins, histtype="step", lw=1.5, color=col, label=lab)
for sel_, lab, col in ((mag == "down", "Magnet down", "tab:green"), (mag == "up", "Magnet up", "tab:orange")):
    axs[1].hist(m[sel_], bins=bins, histtype="step", lw=1.5, color=col, label=lab)
for a, t in zip(axs, ("Invariant mass, by charge", "Invariant mass, by magnet polarity")):
    a.set_title(t); a.set_xlabel(r"$m(KKK)$ [MeV]"); a.set_ylabel(f"Candidates / {BIN:.0f} MeV"); a.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUT, "mass_spectrum.pdf"))
print("saved mass_spectrum.pdf")
