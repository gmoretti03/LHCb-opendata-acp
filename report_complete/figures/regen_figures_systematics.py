"""Systematic-study figures: shifts of all variants (Bayesian and frequentist), scan of the PID cuts,
and the charge-permutation test of the Dalitz homogeneity statistic.

Inputs (written by the analysis scripts):
  freq_systematics.json, freq_pid_scan.json   in  Code_general/Frequentist/results   (wt_freq)
  nominal_summary.json                        in  Code_general/Bayesian/results/nominal (wt_bayes)
  raw data + bayesian_nominal.load_candidates for the permutation test
usage (cern env):  python regen_figures_systematics.py [OUTDIR]
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else HERE
os.makedirs(OUT, exist_ok=True)
WORK = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
FREQ = os.environ.get("REGEN_FREQ_DIR", os.path.join(WORK, "wt_freq", "LHCb", "Code_general", "Frequentist", "results"))
BAYES = os.environ.get("REGEN_BAYES_DIR", os.path.join(WORK, "wt_bayes", "LHCb", "Code_general", "Bayesian"))
DATA = os.environ.get("LHCB_DATA_DIR", os.path.join(WORK, "fededegu_martiborsa", "LHCb", "opendata-project", "Data"))
sys.path.insert(0, HERE)
import plot_style as ps
ps.use(); C = ps.COLORS


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("saved", name)


# ---------------------------------------------------------------- (1) shifts of all variants
fs = json.load(open(os.path.join(FREQ, "freq_systematics.json")))["systematics"]["variations"]
bs = json.load(open(os.path.join(BAYES, "results", "nominal", "nominal_summary.json")))["systematics"]["variations"]
LAB = [("window_lo5050", "window $[5050,5600]$"), ("window_lo5150", "window $[5150,5600]$"),
       ("window_hi5450", "window $[5100,5450]$"), ("window_hi5700", "window $[5100,5700]$"),
       ("bkg_single_exp", "single exponential (rejected)"), ("bkg_shared_shape", "shape shared between charges"),
       ("bkg_symmetric", r"symmetric, $A_b=0$"), ("signal_crystal_ball", "Crystal Ball"),
       ("signal_per_charge", r"$\mu,\sigma$ per charge"), ("presel_probk_0.4", r"$\mathtt{ProbK}>0.4$"),
       ("presel_probk_0.6", r"$\mathtt{ProbK}>0.6$"), ("presel_probpi_0.4", r"$\mathtt{ProbPi}<0.4$"),
       ("presel_probpi_0.6", r"$\mathtt{ProbPi}<0.6$")]
fig, ax = plt.subplots(figsize=(ps.SINGLE, 6.2))
y = np.arange(len(LAB))[::-1]
for k, (key, lab) in enumerate(LAB):
    for d, col, off, mk, name in ((bs, C["bayes"], 0.17, "o", "Bayesian"), (fs, C["freq"], -0.17, "s", "frequentist")):
        v = d[key]; sb = v.get("sigma_barlow")
        a = ax.errorbar(v["shift"], y[k] + off, xerr=sb if sb else None, fmt=mk, color=col, ms=5, capsize=2.5,
                        mfc=col if not v.get("rejected_by_data") else "none", label=name if k == 0 else None)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([l for _, l in LAB])
for yy in (y[3] - 0.5, y[6] - 0.5, y[8] - 0.5):
    ax.axhline(yy, color="0.7", lw=0.6, ls=":")
ax.set_xlabel(r"Shift $\Delta=A_{\mathrm{raw}}^{\mathrm{var}}-A_{\mathrm{raw}}^{\mathrm{nom}}$")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout(); save(fig, "syst_variants.pdf")

# ---------------------------------------------------------------- (2) scan of the PID cuts
rows = json.load(open(os.path.join(FREQ, "freq_pid_scan.json")))["rows"]
nom = [r for r in rows if r["label"] == "nominal"][0]
fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 4.4), sharey=True)
for ax, key, par, lab, xl in ((axs[0], "probk_", "prob_k_min", "a", r"$\mathtt{ProbK}$ lower cut"),
                              (axs[1], "probpi_", "prob_pi_max", "b", r"$\mathtt{ProbPi}$ upper cut")):
    ps.letter(ax, lab)
    sc = sorted([r for r in rows if r["label"].startswith(key)] + [nom], key=lambda r: r[par])
    x = np.array([r[par] for r in sc]); s = np.array([r["shift"] for r in sc]); sb = np.array([r["sigma_barlow"] for r in sc])
    ax.fill_between(x, -sb, sb, color=C["shade"] if "shade" in C else "0.85", label=r"$\pm\sigma_B$")
    ax.fill_between(x, -2 * sb, 2 * sb, color="0.93", zorder=0, label=r"$\pm2\sigma_B$")
    ax.plot(x, s, "o-", color=C["freq"], label="shift")
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0.5, color=C["background"], ls="-.", lw=0.9)
    ax.set_xlabel(xl)
axs[0].set_ylabel(r"Shift of $A_{\mathrm{raw}}$ (frequentist fit)")
axs[0].legend(loc="lower left", fontsize=9)
plt.tight_layout(); save(fig, "syst_pid_scan.pdf")

# ---------------------------------------------------------------- (3) charge-permutation test, 8x8 Dalitz grid
sys.path.insert(0, os.path.join(BAYES))
import bayesian_nominal as bn
cands = bn.select(bn.load_candidates(DATA), bn.NOMINAL)
q = cands.q.to_numpy(); m12 = cands.m12.to_numpy() ** 2 / 1e6; m13 = cands.m13.to_numpy() ** 2 / 1e6   # GeV^2
lo, hi = np.minimum(m12, m13), np.maximum(m12, m13)
ix = np.clip((lo / 16 * 8).astype(int), 0, 7); iy = np.clip((hi / 30 * 8).astype(int), 0, 7)
cell = ix * 8 + iy


def chi2_homog(qv):
    A_glob = (qv == -1).mean() - (qv == 1).mean()
    n = np.bincount(cell, minlength=64).astype(float)
    nm = np.bincount(cell[qv == -1], minlength=64).astype(float)
    ok = n >= 20
    Ai = np.where(ok, (nm - (n - nm)) / np.maximum(n, 1), 0)
    sg = np.sqrt(np.clip(1 - Ai ** 2, 1e-9, None) / np.maximum(n, 1))
    return float((((Ai - A_glob) / sg) ** 2)[ok].sum()), int(ok.sum())


obs, nb = chi2_homog(q)
rng = np.random.default_rng(20261012)
perm = np.array([chi2_homog(rng.permutation(q))[0] for _ in range(2000)])
print("observed chi2", obs, "bins", nb, "permutations mean/sd", perm.mean(), perm.std(), "max", perm.max(),
      "frac>=obs", (perm >= obs).mean())
fig, ax = plt.subplots(figsize=(ps.SINGLE, 4.2))
ax.hist(perm, bins=40, density=True, color=C["replica"], alpha=0.6, label="2000 charge permutations")
xs = np.linspace(0, max(obs * 1.1, perm.max() * 1.1), 400)
ax.plot(xs, stats.chi2.pdf(xs, nb - 1), color="k", ls="--", lw=1.4, label=r"$\chi^2$, %d d.o.f." % (nb - 1))
ax.axvline(obs, color=C["observed"], lw=2, label=r"observed, $\chi^2=%.1f$" % obs)
ax.set_xlabel(r"Homogeneity statistic $\chi^2=\sum_iS_i^2$"); ax.set_ylabel("Density"); ax.legend(fontsize=9, loc="upper right")
plt.tight_layout(); save(fig, "dalitz_permutation.pdf")
