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
sys.path.insert(0, HERE)
import plot_style as ps                                  # also gives the paths: ps.ANALYSIS, ps.DATA
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else HERE
CG = ps.ANALYSIS
FREQ = os.path.join(CG, "Frequentist", "results")
VAL = os.path.join(FREQ, "validation")
BAYES = os.path.join(CG, "Bayesian", "results", "nominal")
sys.path.insert(0, HERE)
import plot_style as ps
ps.use(); C = ps.COLORS
BC = BLUE = C["bayes"]  # one Bayesian colour in every figure
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
    a.fill_between(xs, 1e-12, kde(xs), color=BC, alpha=0.28, lw=0, label="Posterior (MCMC)")
    a.plot(xs, stats.norm.pdf(xs, mu, sd), color=BC, lw=2.2, ls=(0,(5,1.5)), label="Gaussian approximation")
    a.axhline(1 / (2 * prior_half), color=C["count"], lw=1.4, ls="--", label=r"Prior, uniform on $(-1,1)$")
    a.axvline(0, color="k", lw=0.9, ls=":")
    p0 = stats.norm.pdf(0, mu, sd)
    a.plot([0], [p0], "D", color=BC, ms=8, zorder=5, mec="white")
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
fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 3.5), gridspec_kw={"wspace": 0.22})
a = axs[0]; ps.letter(a, "a")
xs = np.linspace(-0.085, -0.005, 400)
like = np.exp(-0.5 * np.interp(xs, A_p, q_p)); like /= np.trapezoid(like, xs)
a.hist(post, bins=48, range=(xs[0], xs[-1]), density=True, color=BLUE, alpha=0.30, edgecolor=BLUE, lw=1.0, histtype="stepfilled", label="Bayesian posterior")
a.plot(xs, like, color=C["freq"], lw=2.4, ls="--", label=r"Frequentist $e^{-q(A)/2}$")
a.axvline(np.median(post), color=BLUE, lw=1.0, ls=":")
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
DATA = ps.DATA
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

# ---------------------------------------------------------------- A_raw with the three methods, before and after the ML selection
rows3 = (("Bayesian fit", C["bayes"], (-0.0443, 0.0085), (-0.0451, 0.0087), (-0.0008, 0.0018)),
         ("Frequentist fit", C["freq"], (-0.0441, 0.0085), (-0.0447, 0.0086), (-0.0006, 0.0012)),
         ("Counting", C["count"], (-0.0432, 0.0071), (-0.0446, 0.0080), (-0.0014, 0.0036)))
fig, ax = plt.subplots(figsize=(ps.SINGLE, 4.0))
ax.axvspan(-0.0443 - 0.0085, -0.0443 + 0.0085, color=C["bayes"], alpha=0.08, lw=0)
ax.axvline(-0.0443, color=C["bayes"], lw=0.9, ls=":")
for k, (lab, col, full, sel, sh) in enumerate(rows3):
    y = 2 - k
    ax.errorbar(full[0], y + 0.14, xerr=full[1], fmt="o", color=col, ms=8, capsize=0, lw=2.4, mec="white", zorder=3)
    ax.errorbar(sel[0], y - 0.14, xerr=sel[1], fmt="s", color=col, mfc="white", mec=col, mew=2.0, ms=8, capsize=0, lw=2.0, zorder=3)
    ax.text(0.985, (y + 0.0) , rf"${sh[0]:+.4f}$ ($\pm{sh[1]:.4f}$)", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=10.5)
ax.set_yticks([2, 1, 0]); ax.set_yticklabels([r[0] for r in rows3]); ax.set_ylim(-0.7, 2.7)
ax.set_xlim(-0.0655, -0.0215)
ax.set_xticks([-0.06, -0.05, -0.04, -0.03]); ax.set_xticklabels(["-0.06", "-0.05", "-0.04", "-0.03"])
ax.set_xlabel(r"$A_{\mathrm{raw}}$")
ax.text(0.985, 1.02, "shift (expected spread)", transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5, color="0.35")
ax.legend(handles=[Line2D([], [], marker="o", color="0.3", lw=2, ms=8, mec="white", label=r"Full window ($19{,}731$)"),
                   Line2D([], [], marker="s", color="0.3", lw=2, ms=8, mfc="white", mew=2, label=r"Score $>0.5$ ($15{,}709$)")],
          loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=10.5, handlelength=1.6, columnspacing=1.2)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
save(fig, "short_ml_three.pdf")

# ---------------------------------------------------------------- ML: weak labels on the mass spectrum, K-means partition of the Dalitz plane
if "m" not in globals() or True:
    import pandas as pd, uproot, subprocess
    LO_, HI_, MK_ = 5100.0, 5600.0, 493.677
    REPO_ = subprocess.check_output(["git", "-C", HERE, "rev-parse", "--show-toplevel"], text=True).strip()
    DATA_ = ps.DATA
    br_ = [f"H{i}_{v}" for i in (1, 2, 3) for v in ("PX", "PY", "PZ", "ProbK", "ProbPi", "isMuon", "Charge")]
    d_ = pd.concat([uproot.open(os.path.join(DATA_, f"B2HHH_Magnet{p}.root"))["DecayTree"].arrays(br_, library="pd") for p in ("Down", "Up")],
                   ignore_index=True)
    s_ = np.ones(len(d_), bool)
    for i in (1, 2, 3):
        s_ &= (d_[f"H{i}_ProbPi"] < 0.5) & (d_[f"H{i}_ProbK"] > 0.5) & (~d_[f"H{i}_isMuon"].astype(bool))
    d_ = d_[s_].reset_index(drop=True)
    P_ = {i: d_[[f"H{i}_PX", f"H{i}_PY", f"H{i}_PZ"]].to_numpy() for i in (1, 2, 3)}
    E__ = {i: np.sqrt((P_[i] ** 2).sum(1) + MK_ ** 2) for i in (1, 2, 3)}
    m_B = np.sqrt(np.clip(sum(E__.values()) ** 2 - (sum(P_.values()) ** 2).sum(1), 0, None))
    q_ = (d_.H1_Charge + d_.H2_Charge + d_.H3_Charge).to_numpy()
    w_ = (m_B > LO_) & (m_B < HI_) & np.isin(q_, [1, -1])
    mm = m_B[w_]
    def m2_(i, j):
        return np.clip((E__[i] + E__[j]) ** 2 - ((P_[i] + P_[j]) ** 2).sum(1), 0, None)[w_] / 1e6
    lo2_, hi2_ = np.minimum(m2_(1, 2), m2_(1, 3)), np.maximum(m2_(1, 2), m2_(1, 3))

    # weak labels
    mu_, sg_ = 5284.39, 18.00
    fig, ax = plt.subplots(figsize=(ps.SINGLE, 4.4))
    edges_ = np.arange(LO_, HI_ + 1e-9, 5.0)
    cnt, _ = np.histogram(mm, bins=edges_)
    cen_ = 0.5 * (edges_[:-1] + edges_[1:])
    sig = np.abs(cen_ - mu_) < 2 * sg_
    bkg = ((cen_ >= 5150) & (cen_ < mu_ - 4 * sg_)) | ((cen_ > mu_ + 4 * sg_) & (cen_ <= HI_))
    gap = ~sig & ~bkg & (cen_ >= 5150)
    exc = cen_ < 5150
    for mask, col, hatch in ((sig, C["sig_like"], None), (bkg, C["bkg_like"], None), (gap, "0.85", None), (exc, "white", "////")):
        ax.bar(cen_[mask], cnt[mask], width=5.0, color=col, edgecolor="0.55" if hatch else col, hatch=hatch, lw=0.0 if not hatch else 0.0)
    ax.step(edges_[:-1], cnt, where="post", color="k", lw=1.0)
    ax.set_yscale("log"); ax.set_ylim(8, 6000); ax.set_xlim(LO_, HI_)
    ax.set_xlabel(r"$m(K^+K^+K^-)$ [MeV]"); ax.set_ylabel("Candidates / 5 MeV")
    n_sig = int(np.sum(np.abs(mm - mu_) < 2 * sg_))
    n_bkg = int(np.sum(((mm >= 5150) & (mm <= mu_ - 4 * sg_)) | ((mm >= mu_ + 4 * sg_) & (mm <= HI_))))
    ax.text(mu_, 2800, "signal-like\n" + rf"${n_sig:,}$".replace(",", "{,}"), ha="center", va="top", fontsize=11, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc=C["sig_like"], ec="none"))
    ax.text(5480, 60, "background-like\n" + rf"${n_bkg:,}$".replace(",", "{,}"), ha="center", va="bottom", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.2", fc=C["bkg_like"], ec="none", alpha=0.9))
    ax.text(5125, 330, "not used\n" + r"($<5150$)", ha="center", va="bottom", fontsize=9.5, color="0.3")
    ax.text(0.5, 0.97, r"$m$ is not an input of the classifier", transform=ax.transAxes, ha="center", va="top", fontsize=10.5, color="0.35",
            style="italic")
    save(fig, "short_ml_labels.pdf")

    # K-means partition
    CGML = os.path.join(CG, "Machine_Learning", "results", "nominal")
    km8_ = np.load(os.path.join(CGML, "kmeans8_labels.npy"))
    kmj = json.load(open(os.path.join(CGML, "kmeans.json")))
    if len(km8_) == len(mm):
        groups_ = {g["group"]: g for g in kmj["8"]["all"]["groups"]}
        S_ = np.array([groups_[k]["S_vs_global"] for k in km8_])
        fig, ax = plt.subplots(figsize=(ps.SINGLE, 5.0))
        order_ = np.argsort(np.abs(S_))                       # significant clusters on top
        sct = ax.scatter(lo2_[order_], hi2_[order_], c=S_[order_], cmap="RdBu_r", vmin=-4, vmax=4, s=4, alpha=0.75, lw=0, rasterized=True)
        for k in range(int(km8_.max()) + 1):
            sel_ = km8_ == k
            ax.text(np.median(lo2_[sel_]), np.median(hi2_[sel_]), (f"{groups_[k]['S_vs_global']:+.1f}" if abs(groups_[k]['S_vs_global']) >= 0.05 else "0.0"), ha="center", va="center", fontsize=11,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.5", alpha=0.9, lw=0.6))
        ax.set_xlabel(r"$m^2_{\mathrm{lo}}(K^+K^-)$ [GeV$^2$]"); ax.set_ylabel(r"$m^2_{\mathrm{hi}}(K^+K^-)$ [GeV$^2$]")
        ax.set_xlim(0, 16); ax.set_ylim(0, 30)
        cb = fig.colorbar(sct, ax=ax, pad=0.02); cb.set_label(r"$S_k=(A_k-A_{\mathrm{glob}})/\sigma(A_k)$")
        save(fig, "short_ml_kmeans.pdf")
    else:
        print("K-means labels do not match the sample:", len(km8_), len(mm))

# ---------------------------------------------------------------- corner plot of the six physics parameters (Bayesian posterior)
PN = ["A", "A_b", "N_s", "N_b", "mu", "sigma"]
PL = [r"$A$", r"$A_b$", r"$N_s$", r"$N_b$", r"$\mu$ [MeV]", r"$\sigma$ [MeV]"]
X = np.column_stack([th[:, names.index(p)] for p in PN])
n_p = len(PN)
from matplotlib.ticker import FormatStrFormatter
fig, axs = plt.subplots(n_p, n_p, figsize=(ps.DOUBLE, 5.6), gridspec_kw={"wspace": 0.08, "hspace": 0.08})
lims = [(np.percentile(X[:, k], 0.1), np.percentile(X[:, k], 99.9)) for k in range(n_p)]
for i in range(n_p):
    for j in range(n_p):
        a = axs[i, j]
        if j > i:
            a.axis("off"); continue
        if i == j:
            a.hist(X[:, i], bins=50, range=lims[i], density=True, color=BC, alpha=0.30, edgecolor=BC, lw=0.8, histtype="stepfilled")
            lo, md, hi = np.percentile(X[:, i], [15.865, 50, 84.135])
            a.axvline(md, color=BC, lw=1.4, ls="--")
            a.axvspan(lo, hi, color=BC, alpha=0.10, lw=0)
            a.set_yticks([]); a.spines["left"].set_visible(False)
            fmt = "{:.4f}" if i < 2 else ("{:.0f}" if i in (2, 3) else "{:.2f}")
            a.set_title(fr"{fmt.format(md)} $\pm$ {fmt.format(0.5 * (hi - lo))}", fontsize=10, pad=3)
        else:
            a.hist2d(X[:, j], X[:, i], bins=40, range=[lims[j], lims[i]], cmap="Reds", cmin=1, rasterized=True)
            r = np.corrcoef(X[:, j], X[:, i])[0, 1]
            a.text(0.95, 0.95, rf"${r:+.2f}$", transform=a.transAxes, ha="right", va="top", fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        a.set_xlim(lims[j])
        if i < n_p - 1:
            a.set_xticklabels([])
        else:
            a.set_xlabel(PL[j]); a.xaxis.set_major_locator(plt.MaxNLocator(3)); a.tick_params(axis="x", labelsize=9)
            a.ticklabel_format(axis="x", useOffset=False, style="plain")
            if j == 4: a.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        if j > 0 or i == 0:
            a.set_yticklabels([]) if i != j else None
        if j == 0 and i > 0:
            a.set_ylabel(PL[i]); a.set_ylim(lims[i]); a.yaxis.set_major_locator(plt.MaxNLocator(3)); a.tick_params(axis="y", labelsize=9)
            a.ticklabel_format(axis="y", useOffset=False, style="plain")
            if i == 4: a.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        elif i != j:
            a.set_ylim(lims[i])
save(fig, "short_corner.pdf")

# ---------------------------------------------------------------- shifts of the 13 variants, Bayesian and frequentist
fs = json.load(open(os.path.join(FREQ, "freq_systematics.json")))["systematics"]
bs = summ["systematics"]
LABV = [("window", [("window_lo5050", r"$[5050,5600]$"), ("window_lo5150", r"$[5150,5600]$"), ("window_hi5450", r"$[5100,5450]$"),
                    ("window_hi5700", r"$[5100,5700]$")]),
        ("background", [("bkg_single_exp", "single exponential"), ("bkg_shared_shape", "shape shared"), ("bkg_symmetric", r"symmetric, $A_b=0$")]),
        ("signal", [("signal_crystal_ball", "Crystal Ball"), ("signal_per_charge", r"$\mu,\sigma$ per charge")]),
        ("preselection", [("presel_probk_0.4", r"$\mathtt{ProbK}>0.4$"), ("presel_probk_0.6", r"$\mathtt{ProbK}>0.6$"),
                          ("presel_probpi_0.4", r"$\mathtt{ProbPi}<0.4$"), ("presel_probpi_0.6", r"$\mathtt{ProbPi}<0.6$")])]
fig, ax = plt.subplots(figsize=(ps.SINGLE, 5.0))
y = 0.0; ticks = []; labs = []; seps = []; gl = []
for gname, items in LABV:
    y0 = y
    for key, lab in items:
        for src, col, off, mk in ((bs, C["bayes"], 0.17, "o"), (fs, C["freq"], -0.17, "s")):
            v = src["variations"][key]; sb = v.get("sigma_barlow")
            rej = v.get("rejected_by_data")
            if sb:
                ax.plot([v["shift"] - sb, v["shift"] + sb], [-(y + off)] * 2, color=col, lw=2.6, alpha=0.35, solid_capstyle="butt", zorder=1)
            ax.plot([v["shift"]], [-(y + off)], mk, color=col, ms=7, mfc="white" if rej else col, mec=col, mew=1.6, zorder=3)
        ticks.append(-y); labs.append(lab); y += 1.0
    seps.append(-(y - 0.5))
    gu = (bs["groups"][gname], fs["groups"][gname])
    gl.append((-(y0 + y - 1) / 2, gname, gu))
    y += 0.35
for sp in seps[:-1]:
    ax.axhline(sp - 0.17, color="0.8", lw=0.8, ls=":")
ax.axvline(0, color="k", lw=0.9)
ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=10.5)
ax.set_xlim(-0.0085, 0.0112)
ax.set_xlabel(r"Shift $\Delta=A_{\mathrm{raw}}^{\mathrm{var}}-A_{\mathrm{raw}}^{\mathrm{nom}}$")
for yy, g, gu in gl:
    ax.text(1.01, yy, f"{g}\n{gu[0]:.4f} / {gu[1]:.4f}", transform=ax.get_yaxis_transform(), fontsize=9.5, va="center", ha="left", color="0.25")
ax.text(1.01, 1.0, "group\nuncertainty", transform=ax.transAxes, fontsize=9.5, va="bottom", ha="left", color="0.25")
ax.legend(handles=[Line2D([], [], marker="o", color=C["bayes"], lw=0, ms=7, label="Bayesian"),
                   Line2D([], [], marker="s", color=C["freq"], lw=0, ms=7, label="Frequentist"),
                   Line2D([], [], color="0.5", lw=2.6, alpha=0.5, label=r"$\pm\sigma_B$"),
                   Line2D([], [], marker="o", color="0.4", mfc="white", lw=0, ms=7, label="rejected by the data")],
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9.5, frameon=False)
ax.set_ylim(min(ticks) - 0.8, 0.8)
ax.set_xticks([-0.005, 0.0, 0.005, 0.010]); ax.set_xticklabels(["-0.005", "0", "0.005", "0.010"])
save(fig, "short_syst_variants.pdf")


# ---------------------------------------------------------------- profile likelihood against posterior (one panel)
fig, a = plt.subplots(figsize=(ps.SINGLE, 3.6))
xs = np.linspace(-0.085, -0.005, 400)
like = np.exp(-0.5 * np.interp(xs, A_p, q_p)); like /= np.trapezoid(like, xs)
a.hist(post, bins=48, range=(xs[0], xs[-1]), density=True, color=BLUE, alpha=0.30, edgecolor=BLUE, lw=1.0, histtype="stepfilled", label="Bayesian posterior")
a.plot(xs, like, color=C["freq"], lw=2.4, ls="--", label=r"Frequentist $e^{-q(A)/2}$")
a.set_xlabel(r"$A_{\mathrm{raw}}$"); a.set_ylabel("Density"); a.legend(loc="upper left", fontsize=10.5)
a.set_ylim(0, a.get_ylim()[1] * 1.05)
save(fig, "short_profile_posterior.pdf")
