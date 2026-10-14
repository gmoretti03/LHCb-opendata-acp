"""Extra validation figures: asymmetry in the phi(1020) band and in the significant Dalitz bins, Araw in terciles of the
momentum of the B, and the uncertainties of the twelve parameters by three methods.

Inputs: Code_general/Bayesian/results/nominal/dalitz_resonances.json (wt_bayes),
        Code_general/Frequentist/results/validation/{momentum,cov_summary}.json (wt_freq).
usage (cern env):  python regen_figures_extras.py [OUTDIR]
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else HERE
os.makedirs(OUT, exist_ok=True)
WORK = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
FREQ = os.environ.get("REGEN_FREQ_DIR", os.path.join(WORK, "wt_freq", "LHCb", "Code_general", "Frequentist", "results", "validation"))
BAYES = os.environ.get("REGEN_BAYES_DIR", os.path.join(WORK, "wt_bayes", "LHCb", "Code_general", "Bayesian", "results", "nominal"))
sys.path.insert(0, HERE)
import plot_style as ps
ps.use(); C = ps.COLORS


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("saved", name)


# ------------------------------------------------ (1) phi(1020) band and significant bins
r = json.load(open(os.path.join(BAYES, "dalitz_resonances.json")))
h = np.array(r["m_lo_hist_980_1100_5MeV"]); edges = np.arange(980, 1100.1, 5)
PHI = 1019.461
fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
a = axs[0]; ps.letter(a, "a")
a.stairs(h, edges, color="k", lw=1.2)
a.axvspan(PHI - 10, PHI + 10, color="0.88", zorder=0, label=r"$\phi(1020)$ band, $|m-m_\phi|<10\,$MeV")
a.axvline(PHI, color=C["observed"], ls="--", lw=1)
a.set_xlabel(r"$m_{\mathrm{lo}}(K^+K^-)$ [MeV]"); a.set_ylabel("Candidates / 5 MeV"); a.legend(fontsize=9, loc="upper right")
b = axs[1]; ps.letter(b, "b")
rows = [("all candidates", "global"), (r"$\phi$ band", "phi_band"),
        (r"$m^2_{\mathrm{lo}}<2\,\mathrm{GeV}^2$, outside the $\phi$ band", "phi_sideband_lo2_below_2"),
        ("significant bins", "significant_bins"), (r"significant bins, without $\phi$", "significant_bins_without_phi"),
        (r"significant bins, $\phi$ only", "significant_bins_phi_only")]
y = np.arange(len(rows))[::-1]
for yy, (lab, key) in zip(y, rows):
    v = r[key]
    b.errorbar(v["A"], yy, xerr=v["sigma"], fmt="o", color=C["count"], capsize=3)
    b.text(0.085, yy, "n=%d" % v["n"], va="center", fontsize=8)
b.axvline(r["global"]["A"], color=C["count"], ls=":", lw=1); b.axvline(0, color="k", lw=0.8)
b.set_yticks(y); b.set_yticklabels([l for l, _ in rows], fontsize=9)
b.set_xlim(-0.22, 0.14); b.set_xlabel(r"Counting asymmetry $A_{\mathrm{count}}$")
plt.tight_layout(); save(fig, "dalitz_phi_band.pdf")

# ------------------------------------------------ (2) Araw in terciles of the momentum of the B
mo = json.load(open(os.path.join(FREQ, "momentum.json")))
glob = json.load(open(os.path.join(FREQ, "nominal_values.json")))
Ag = glob.get("A", glob.get("A_CP", -0.0441)) if isinstance(glob, dict) else -0.0441
if not isinstance(Ag, float):
    Ag = -0.0441
fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 4.2), sharey=True)
for ax, key, lab, xl in ((axs[0], "B_PT", "a", r"$p_T$ of the $B$ candidate (terciles)"), (axs[1], "B_P", "b", r"$p$ of the $B$ candidate (terciles)")):
    ps.letter(ax, lab)
    rr = mo[key]
    x = np.arange(len(rr))
    ax.errorbar(x, [t["A"] for t in rr], yerr=[t["err"] for t in rr], fmt="o", color=C["freq"], capsize=3, ms=6)
    ax.axhspan(Ag - 0.0085, Ag + 0.0085, color="0.88", label=r"global $\pm1\sigma$")
    ax.axhline(Ag, color=C["count"], ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(["%.0f--%.0f" % (t["lo"] / 1e3, t["hi"] / 1e3) for t in rr])
    ax.set_xlabel(xl + r" [GeV]"); ax.set_xlim(-0.5, len(rr) - 0.5); ax.set_ylim(-0.08, -0.012)
axs[0].set_ylabel(r"$A_{\mathrm{raw}}$ (joint frequentist fit)"); axs[0].legend(fontsize=9, loc="lower left")
plt.tight_layout(); save(fig, "momentum_terciles.pdf")

# ------------------------------------------------ (3) uncertainties of the twelve parameters by three methods
cv = json.load(open(os.path.join(FREQ, "cov_summary.json")))["sigma"]
order = [("A", r"$A$"), ("A_b", r"$A_b$"), ("N_s", r"$N_s$"), ("N_b", r"$N_b$"), ("mu", r"$\mu$"), ("sigma", r"$\sigma$"),
         ("lam1_pos", r"$\lambda_1^+$"), ("lam2_pos", r"$\lambda_2^+$"), ("f_pos", r"$f_+$"),
         ("lam1_neg", r"$\lambda_1^-$"), ("lam2_neg", r"$\lambda_2^-$"), ("f_neg", r"$f_-$")]
x = np.arange(len(order))
fig, ax = plt.subplots(figsize=(ps.DOUBLE, 4.2))
ax.axhspan(0.9, 1.1, color="0.9", label=r"$\pm10\%$")
ax.axhline(1, color="k", lw=0.8)
ax.plot(x - 0.1, [cv[k]["bootstrap"] / cv[k]["hesse"] for k, _ in order], "s", color=C["freq"], label="bootstrap of the data (300)")
ax.plot(x + 0.1, [cv[k]["mcmc"] / cv[k]["hesse"] for k, _ in order], "o", color=C["bayes"], label="posterior (MCMC)")
ax.set_xticks(x); ax.set_xticklabels([l for _, l in order])
ax.set_ylabel(r"$\sigma_{\mathrm{method}}/\sigma_{\mathrm{Hesse}}$"); ax.set_ylim(0.4, 1.3)
ax.legend(fontsize=9, loc="lower left")
plt.tight_layout(); save(fig, "uncertainties_12.pdf")
