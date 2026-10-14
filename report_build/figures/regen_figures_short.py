"""Figures of the short version of the report, redrawn from the saved results (no fit is run):
coverage of the Wilks interval, Savage-Dickey ratios, scan of the PID cuts, bias/width/power of the estimator and
profile likelihood against posterior with the two polarities.

Inputs : Code_general/Frequentist/results/{validation/*,freq_pid_scan.json}, Code_general/Bayesian/results/nominal/*
Outputs: short_*.pdf in this folder (or OUTDIR).
usage (cern env):  python regen_figures_short.py [OUTDIR]
"""
import json, os, sys, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else HERE
CG = os.path.abspath(os.path.join(HERE, "..", "..", "Code_general"))
FREQ = os.path.join(CG, "Frequentist", "results")
VAL = os.path.join(FREQ, "validation")
BAYES = os.path.join(CG, "Bayesian", "results", "nominal")
sys.path.insert(0, HERE)
import plot_style as ps
ps.use(); C = ps.COLORS
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.0,
                     "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("saved", name)


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


# ---------------------------------------------------------------- coverage of the Wilks interval (1000 bootstrap replicas)
rep = [r for r in jl(os.path.join(VAL, "bootcov.jsonl")) if r.get("valid")]
if len(rep) > 500:
    t = np.array([r["t"] for r in rep]); Ah = np.array([r["A"] for r in rep]); er = np.array([r["err"] for r in rep])
    A_true = json.load(open(os.path.join(VAL, "nominal_values.json")))["A_CP"]
    cov = np.mean(t <= 1.0); dcov = np.sqrt(cov * (1 - cov) / len(t))
    pull = (Ah - A_true) / er
    fig, (a, b) = plt.subplots(2, 1, figsize=(ps.SINGLE, 6.2), gridspec_kw={"hspace": 0.42})
    xs = np.linspace(0, 9, 400)
    ts = np.sort(t); sf = 1 - np.arange(1, len(ts) + 1) / len(ts) + 1.0 / len(ts)
    a.step(ts, sf, where="post", color=C["freq"], lw=2.2, label=f"{len(t)} bootstrap replicas")
    a.plot(xs, stats.chi2.sf(xs, 1), color="k", lw=1.6, ls="--", label=r"$\chi^2_1$ (Wilks)")
    a.axvline(1.0, color=C["observed"], lw=1.2, ls=":")
    a.set_yscale("log"); a.set_ylim(2e-3, 1.3); a.set_xlim(0, 9)
    a.set_xlabel(r"$t=q(A_{\mathrm{true}})$"); a.set_ylabel(r"$P(q\geq t)$")
    a.text(1.25, 0.93, rf"$t\leq1$: ${100*cov:.1f}\pm{100*dcov:.1f}\%$", color=C["observed"], fontsize=12, va="top")
    a.legend(loc="lower left", fontsize=11); ps.letter(a, "a")
    bins = np.linspace(-3.5, 3.5, 29)
    b.hist(pull, bins=bins, density=True, color=C["freq"], alpha=0.45, edgecolor="white", lw=0.5, label="Replicas")
    xg = np.linspace(-3.5, 3.5, 300)
    b.plot(xg, stats.norm.pdf(xg), color="k", lw=1.8, label="Standard normal")
    b.text(0.03, 0.95, rf"mean $={pull.mean():+.2f}\pm{pull.std(ddof=1)/np.sqrt(len(pull)):.2f}$" + "\n" + rf"sd $={pull.std(ddof=1):.2f}$",
           transform=b.transAxes, va="top", fontsize=12)
    b.set_xlabel(r"$(\hat A-A_{\mathrm{true}})/\sigma(\hat A)$"); b.set_ylabel("Density"); b.legend(loc="upper right", fontsize=11)
    ps.letter(b, "b")
    save(fig, "short_coverage.pdf")
    print("coverage", cov, dcov, len(t))

# ---------------------------------------------------------------- Savage-Dickey ratios
summ = json.load(open(os.path.join(BAYES, "nominal_summary.json")))
names = summ["nominal"]["names"]
th = np.load(os.path.join(BAYES, "nominal_posterior_thinned.npy"))
NOM = summ["nominal"]["params"] if "params" in summ["nominal"] else summ["nominal"]
fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 4.4), gridspec_kw={"wspace": 0.22})
for k, (nm, lab, lim, ylim, prior_half) in enumerate((("A", r"$A$", (-0.1, 0.02), (1e-6, 6e2), 1.0),
                                                      ("A_b", r"$A_b$", (-0.14, 0.06), (3e-3, 6e2), 1.0))):
    a = axs[k]; x = th[:, names.index(nm)]
    st = NOM[nm]; mu, sd = st["mean"], st["sd"]                      # as in the analysis (nominal_summary.json)
    xs = np.linspace(*lim, 600)
    kde = stats.gaussian_kde(x, bw_method=0.18)
    a.fill_between(xs, 1e-12, kde(xs), color=C["bayes"], alpha=0.30, lw=0, label="Posterior (MCMC)")
    a.plot(xs, stats.norm.pdf(xs, mu, sd), color=C["bayes"], lw=2.0, label="Gaussian approximation")
    a.axhline(1 / (2 * prior_half), color=C["count"], lw=1.4, ls="--", label=r"Prior, uniform on $(-1,1)$")
    a.axvline(0, color="k", lw=0.9, ls=":")
    p0 = stats.norm.pdf(0, mu, sd)
    a.plot([0], [p0], "o", color=C["bayes"], ms=8, zorder=5, mec="white")
    a.plot([0], [1 / (2 * prior_half)], "o", color=C["count"], ms=8, zorder=5, mec="white")
    bf = p0 / (1 / (2 * prior_half))
    a.set_yscale("log"); a.set_xlim(*lim); a.set_ylim(*ylim)
    a.set_xlabel(lab); a.set_ylabel("Density"); ps.letter(a, "ab"[k])
    a.xaxis.set_major_locator(plt.MaxNLocator(6))
    a.text(0.97, 0.95, rf"$BF_{{01}}={ps.sci(bf, 1)}$" if bf < 0.01 else rf"$BF_{{01}}={bf:.1f}$", transform=a.transAxes,
           ha="right", va="top", fontsize=14, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))
    if k == 0:
        a.legend(loc="lower left", fontsize=10)
save(fig, "short_savage_dickey.pdf")

# ---------------------------------------------------------------- scan of the PID cuts, one cut at a time
rows = json.load(open(os.path.join(FREQ, "freq_pid_scan.json")))["rows"]
nom = [r for r in rows if r["label"] == "nominal"][0]
fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 4.4), sharey=True, gridspec_kw={"wspace": 0.06})
for ax, key, par, xl, k in ((axs[0], "probk_", "prob_k_min", r"$\mathtt{ProbK}$ lower cut", 0),
                            (axs[1], "probpi_", "prob_pi_max", r"$\mathtt{ProbPi}$ upper cut", 1)):
    sc = sorted([r for r in rows if r["label"].startswith(key)] + [nom], key=lambda r: r[par])
    x = np.array([r[par] for r in sc]); s = np.array([r["shift"] for r in sc]) * 1e3; sb = np.array([r["sigma_barlow"] for r in sc]) * 1e3
    ax.fill_between(x, -2 * sb, 2 * sb, color=C["freq"], alpha=0.10, lw=0, label=r"$\pm2\sigma_B$")
    ax.fill_between(x, -sb, sb, color=C["freq"], alpha=0.25, lw=0, label=r"$\pm\sigma_B$")
    ax.plot(x, s, "-", color=C["freq"], lw=1.6, zorder=3)
    out = np.abs(s) > sb
    ax.plot(x[~out], s[~out], "o", color=C["freq"], ms=7, zorder=4, mec="white", label=r"$|\Delta|<\sigma_B$")
    ax.plot(x[out], s[out], "o", color=C["observed"], ms=7, zorder=4, mec="white", label=r"$|\Delta|>\sigma_B$")
    ax.axhline(0, color="k", lw=0.9)
    ax.axvline(0.5, color=C["count"], ls="--", lw=1.0)
    ax.text(0.505, 0.97, "nominal", transform=ax.get_xaxis_transform(), color=C["count"], fontsize=10, va="top")
    ax.set_xlabel(xl); ps.letter(ax, "ab"[k])
    if k == 0:
        j = np.argmin(np.abs(x - 0.6))
        ax.annotate(rf"$\mathtt{{ProbK}}>0.6$: ${s[j]:+.1f}\times10^{{-3}}$" + "\n" + rf"($1.2\,\sigma_B$)", xy=(x[j], s[j]),
                    xytext=(0.62, 17), fontsize=10, arrowprops=dict(arrowstyle="->", color="k", lw=0.9))
        ax.set_ylabel(r"Shift of $A_{\mathrm{raw}}$ [$10^{-3}$]")
    if k == 1:
        ax.legend(loc="lower left", fontsize=10, ncol=2, columnspacing=0.9, bbox_to_anchor=(0.02, 0.0))
axs[0].set_ylim(-36, 36)
save(fig, "short_pid_scan.pdf")

# ---------------------------------------------------------------- toys: bias, width, power of the estimator and of the test
toys = [r for r in jl(os.path.join(VAL, "toys.jsonl")) if r.get("valid")]
byA = collections.defaultdict(list)
for r in toys:
    byA[r["a_true"]].append(r)
ks = sorted(byA)
fig, axs = plt.subplots(1, 3, figsize=(ps.TRIPLE, 4.4), gridspec_kw={"wspace": 0.32})
n_ = np.array([len(byA[k]) for k in ks])
bias = np.array([np.mean([r["A"] for r in byA[k]]) - k for k in ks]) * 1e4
ebias = np.array([np.std([r["A"] for r in byA[k]], ddof=1) / np.sqrt(len(byA[k])) for k in ks]) * 1e4
a = axs[0]; ps.letter(a, "a")
a.axhspan(-5, 5, color="0.9", lw=0)
a.errorbar(ks, bias, yerr=ebias, fmt="o", color=C["freq"], ms=8, capsize=4, lw=1.6, mec="white")
a.axhline(0, color="k", lw=0.9)
a.set_ylim(-16, 16); a.set_xlim(-0.12, 0.07)
a.set_xlabel(r"Injected $A_{\mathrm{true}}$"); a.set_ylabel(r"$\langle\hat A\rangle-A_{\mathrm{true}}$ [$10^{-4}$]")
b = axs[1]; ps.letter(b, "b")
sd = np.array([np.std([r["A"] for r in byA[k]], ddof=1) / np.mean([r["err"] for r in byA[k]]) for k in ks])
pw = np.array([np.std([(r["A"] - k) / r["err"] for r in byA[k]], ddof=1) for k in ks])
esd = sd / np.sqrt(2 * (n_ - 1))
b.axhspan(0.94, 1.06, color="0.9", lw=0)
b.errorbar(ks, sd, yerr=esd, fmt="s", color=C["count"], ms=8, capsize=4, lw=1.4, mec="white", label=r"$\mathrm{sd}(\hat A)/\langle\sigma_{\mathrm{Hesse}}\rangle$")
b.plot(ks, pw, "o", color=C["freq"], ms=5, label="sd of the pull")
b.axhline(1, color="k", lw=0.9)
b.set_ylim(0.85, 1.15); b.set_xlim(-0.12, 0.07)
b.set_xlabel(r"Injected $A_{\mathrm{true}}$"); b.set_ylabel("Width"); b.legend(loc="upper left", fontsize=10)
c = axs[2]; ps.letter(c, "c")
crit = stats.chi2.ppf(0.9973, 1)
p95 = np.array([np.mean([r["q0"] > 3.84 for r in byA[k]]) for k in ks]); p3 = np.array([np.mean([r["q0"] > crit for r in byA[k]]) for k in ks])
se = lambda p: np.sqrt(np.clip(p * (1 - p), 1e-4, None) / n_)
c.errorbar(ks, p95, yerr=se(p95), fmt="o-", color=C["freq"], ms=8, capsize=4, lw=1.8, mec="white", label=r"$q(0)>3.84$ (95%)")
c.errorbar(ks, p3, yerr=se(p3), fmt="s--", color=C["observed"], ms=7, capsize=4, lw=1.5, mec="white", label=r"$q(0)>9$ ($3\sigma$)")
c.axhline(0.05, color="k", ls=":", lw=1.0)
c.set_ylim(-0.03, 1.05); c.set_xlim(-0.12, 0.07)
c.set_xlabel(r"Injected $A_{\mathrm{true}}$"); c.set_ylabel(r"Probability to reject $A=0$"); c.legend(loc="center left", fontsize=10)
save(fig, "short_injection_power.pdf")

# ---------------------------------------------------------------- profile likelihood against posterior, magnet polarities
prof = json.load(open(os.path.join(VAL, "profiles.json")))
jf = json.load(open(os.path.join(VAL, "jointfit.json")))["profile"]
A_p, q_p = np.array(jf["A"]), np.array(jf["q"])
post = th[:, names.index("A")]
fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 4.4), gridspec_kw={"wspace": 0.22})
a = axs[0]; ps.letter(a, "a")
xs = np.linspace(-0.085, -0.005, 400)
like = np.exp(-0.5 * np.interp(xs, A_p, q_p)); like /= np.trapezoid(like, xs)
a.hist(post, bins=48, range=(xs[0], xs[-1]), density=True, color=C["bayes"], alpha=0.35, edgecolor="white", lw=0.4, label="Bayesian posterior")
a.plot(xs, like, color=C["freq"], lw=2.8, label=r"Frequentist $e^{-q(A)/2}$")
a.axvline(np.median(post), color=C["bayes"], lw=1.0, ls=":")
a.set_xlabel(r"$A_{\mathrm{raw}}$"); a.set_ylabel("Density"); a.legend(loc="upper left", fontsize=11)
b = axs[1]; ps.letter(b, "b")
def par(A, q):
    A, q = np.asarray(A), np.asarray(q); m = q < 9
    c2 = np.polyfit(A[m], q[m], 2); a0 = -c2[1] / (2 * c2[0]); return a0, 1 / np.sqrt(c2[0]), (lambda x: np.polyval(c2, x) - np.polyval(c2, a0))
xx = np.linspace(-0.1, 0.0, 400); qs = 0
for w, lab, col in (("down", "Magnet down", C["down"]), ("up", "Magnet up", C["up"])):
    p_ = prof["profile"][w]; a0, s0, f = par(p_["A"], p_["q"])
    b.plot(p_["A"], p_["q"], "o", color=col, ms=5, mec="white")
    b.plot(xx, f(xx), color=col, lw=2.0, label=rf"{lab}: ${a0:+.4f}\pm{s0:.4f}$"); qs = qs + f(xx)
j = np.argmin(qs)
b.plot(xx, qs - qs[j], color="k", lw=2.4, label="Sum of the two polarities")
b.plot(A_p, q_p, color=C["freq"], lw=2.4, ls=(0, (4, 2)), label="Joint fit")
b.axhline(1, color=C["count"], ls=":", lw=1.0)
b.set_ylim(0, 6); b.set_xlim(-0.1, 0.0)
b.set_xlabel(r"$A_{\mathrm{raw}}$"); b.set_ylabel(r"$q(A)=-2\Delta\ln L$")
b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10, columnspacing=1.2)
save(fig, "short_profile_polarity.pdf")

# ---------------------------------------------------------------- Dalitz plane: occupancy, local asymmetry, significance
import subprocess
import pandas as pd
import uproot
from matplotlib.patches import Rectangle
LO, HI, MK = 5100.0, 5600.0, 493.677
REPO = subprocess.check_output(["git", "-C", HERE, "rev-parse", "--show-toplevel"], text=True).strip()
DATA = os.path.join(REPO, "LHCb", "opendata-project", "Data")
br = [f"H{i}_{v}" for i in (1, 2, 3) for v in ("PX", "PY", "PZ", "ProbK", "ProbPi", "isMuon", "Charge")]
df = pd.concat([uproot.open(os.path.join(DATA, f"B2HHH_Magnet{p}.root"))["DecayTree"].arrays(br, library="pd") for p in ("Down", "Up")],
               ignore_index=True)
sel = np.ones(len(df), bool)
for i in (1, 2, 3):
    sel &= (df[f"H{i}_ProbPi"] < 0.5) & (df[f"H{i}_ProbK"] > 0.5) & (~df[f"H{i}_isMuon"].astype(bool))
df = df[sel].reset_index(drop=True)
P = {i: df[[f"H{i}_PX", f"H{i}_PY", f"H{i}_PZ"]].to_numpy() for i in (1, 2, 3)}
E_ = {i: np.sqrt((P[i] ** 2).sum(1) + MK ** 2) for i in (1, 2, 3)}
mB = np.sqrt(np.clip(sum(E_.values()) ** 2 - (sum(P.values()) ** 2).sum(1), 0, None))
def m2(i, j):
    return np.clip((E_[i] + E_[j]) ** 2 - ((P[i] + P[j]) ** 2).sum(1), 0, None) / 1e6      # GeV^2
q = (df.H1_Charge + df.H2_Charge + df.H3_Charge).to_numpy()
w = (mB > LO) & (mB < HI) & np.isin(q, [1, -1])
a12, a13, qq = m2(1, 2)[w], m2(1, 3)[w], q[w]
lo2, hi2 = np.minimum(a12, a13), np.maximum(a12, a13)
ag = (np.sum(qq < 0) - np.sum(qq > 0)) / len(qq)
el, eh = np.linspace(0, 16, 9), np.linspace(0, 30, 9)
npl, _, _ = np.histogram2d(lo2[qq > 0], hi2[qq > 0], [el, eh]); nmi, _, _ = np.histogram2d(lo2[qq < 0], hi2[qq < 0], [el, eh])
n = npl + nmi; ok = n >= 20
with np.errstate(divide="ignore", invalid="ignore"):
    ai = np.where(ok, (nmi - npl) / n, np.nan); sg = (ai - ag) / np.sqrt((1 - ai ** 2) / n)
print("Dalitz: n", len(qq), "A_glob", ag, "bins ok", ok.sum(), "|S|>3:", int((np.abs(np.nan_to_num(sg)) > 3).sum()))
fig, axs = plt.subplots(1, 3, figsize=(ps.TRIPLE, 4.6), gridspec_kw={"wspace": 0.75})
ext = [0, 16, 0, 30]
charm = {r"$D^0$": 1.86484, r"$J/\psi$": 3.0969, r"$\chi_{c0}$": 3.4147}
panels = ((np.where(n > 0, n, np.nan), "YlGnBu", None, None, "Candidates per bin"),
          (ai, "RdBu_r", -0.5, 0.5, r"$A_i$"), (sg, "RdBu_r", -4, 4, r"$S_i$"))
for k, (a, (arr, cmap, vmin, vmax, lab)) in enumerate(zip(axs, panels)):
    im = a.imshow(arr.T, origin="lower", aspect="auto", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    cb = fig.colorbar(im, ax=a, pad=0.03, fraction=0.05); cb.set_label(lab)
    for e in el: a.axvline(e, color="white", lw=0.5)
    for e in eh: a.axhline(e, color="white", lw=0.5)
    a.set_xlabel(r"$m^2_{\mathrm{lo}}(K^+K^-)$ [GeV$^2$]"); ps.letter(a, "abc"[k])
    if k == 0:
        a.set_ylabel(r"$m^2_{\mathrm{hi}}(K^+K^-)$ [GeV$^2$]")
    if k > 0:
        for lab_, mv in charm.items():
            a.axvline(mv ** 2, color="k", ls="--", lw=0.8, alpha=0.5); a.axhline(mv ** 2, color="k", ls="--", lw=0.8, alpha=0.5)
if True:
    for lab_, mv in charm.items():
        axs[1].text(mv ** 2 + 0.25, 29.3, lab_, fontsize=10, va="top", rotation=90)
for i_ in range(len(el) - 1):
    for j_ in range(len(eh) - 1):
        if ok[i_, j_] and abs(sg[i_, j_]) > 3:
            axs[2].add_patch(Rectangle((el[i_], eh[j_]), el[i_ + 1] - el[i_], eh[j_ + 1] - eh[j_], fill=False, ec="k", lw=2.2))
            axs[2].text(el[i_] + 1, eh[j_] + 1.9, f"{sg[i_, j_]:+.1f}", ha="center", va="center", fontsize=10, color="w", fontweight="bold")
save(fig, "short_dalitz_maps.pdf")
