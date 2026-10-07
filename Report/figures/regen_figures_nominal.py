"""Figures of the nominal Bayesian model (Bayesian_Nominal.ipynb on the `bayes` branch).

No MCMC is rerun: the figures are built from the files the notebook saves in
`Code_general/Bayesian/results/nominal/` (thinned posterior sample, summary, toys) plus the ROOT data.
By default those files are taken from the `bayes` branch with `git show`; set
REGEN_NOMINAL_DIR=/path/to/results/dir to read a local directory instead (e.g. a test run).

usage (cern env):  python regen_figures_nominal.py [OUTDIR]        runtime ~1 min
Output names start with `bayes_nominal_`, so the figures of the simplified model are not overwritten.
"""
import json, os, subprocess, sys, tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else HERE
os.makedirs(OUT, exist_ok=True)
REPO = subprocess.check_output(["git", "-C", HERE, "rev-parse", "--show-toplevel"], text=True).strip()
DATA_DIR = os.path.join(REPO, "LHCb", "opendata-project", "Data")
BAYES_DIR = "LHCb/Code_general/Bayesian"
LOCAL = os.environ.get("REGEN_NOMINAL_DIR")

work = tempfile.mkdtemp(prefix="regen_nominal_")
def fetch(rel_in_bayes_dir, binary=False):
    """file from the bayes branch (or from REGEN_NOMINAL_DIR for results/nominal/*), copied to `work`."""
    dst = os.path.join(work, os.path.basename(rel_in_bayes_dir))
    if LOCAL:   # local test: results from LOCAL, module from the Bayesian directory two levels up
        src = os.path.join(LOCAL, os.path.basename(rel_in_bayes_dir)) if rel_in_bayes_dir.startswith("results/") \
            else os.path.join(LOCAL, "..", "..", rel_in_bayes_dir)
        with open(src, "rb") as f, open(dst, "wb") as g:
            g.write(f.read())
    else:
        blob = subprocess.check_output(["git", "-C", REPO, "show", f"bayes:{BAYES_DIR}/{rel_in_bayes_dir}"])
        with open(dst, "wb") as g:
            g.write(blob)
    return dst

sys.path.insert(0, work)
fetch("bayesian_nominal.py")
import bayesian_nominal as bn

summary = json.load(open(fetch("results/nominal/nominal_summary.json")))
thinned = np.load(fetch("results/nominal/nominal_posterior_thinned.npy"))
toys = [json.loads(l) for l in open(fetch("results/nominal/toys.jsonl")) if l.strip()]
mag = {p: np.load(fetch(f"results/nominal/magnet_{p}_A_thinned.npy")) for p in ("down", "up")}
nom = summary["nominal"]
names = nom["names"]
cfg = bn.Config(**{k: (tuple(v) if k == "veto" else v) for k, v in summary["model"].items()})
cands = bn.load_candidates(DATA_DIR)
data = bn.select(cands, cfg)

sys.path.insert(0, HERE)
import plot_style as ps
ps.use(); C = ps.COLORS
def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("saved", name)

# ---------------- corner plot of the six physics parameters (background shapes omitted for readability)
import corner
main = ["N_s", "N_b", "A", "A_b", "mu", "sigma"]
idx = [names.index(k) for k in main]
labels = [r"$N_s$", r"$N_b$", r"$A$", r"$A_b$", r"$\mu$ [MeV]", r"$\sigma$ [MeV]"]
fmts = [".0f", ".0f", ".4f", ".3f", ".2f", ".2f"]
th = thinned[:, idx]
fig = corner.corner(th, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=False, plot_datapoints=False,
                    label_kwargs={"fontsize": 14})
ax = np.array(fig.axes).reshape(len(main), len(main))
for k in range(len(main)):
    st = nom[main[k]]; q50 = st["median"]; q16, q84 = st["ci68"]; f = fmts[k]   # full chain, as in the text
    ax[k, k].set_title(f"{labels[k].split(' [')[0]} $= {q50:{f}}^{{+{q84-q50:{f}}}}_{{-{q50-q16:{f}}}}$", fontsize=13)
for a_ in fig.axes:   # plain tick labels (no "+5.284e3" offset on the mu axis)
    for axis in (a_.xaxis, a_.yaxis):
        fmt = axis.get_major_formatter()
        if hasattr(fmt, "set_useOffset"):
            fmt.set_useOffset(False)
save(fig, "bayes_nominal_corner.pdf")

# ---------------- mass fit, log scale, components and pulls
post = bn.Posterior(data.m.to_numpy(), data.q.to_numpy(), cfg)
theta = np.median(thinned, axis=0)
d = post.unpack(theta)
width = 10.0
edges = np.arange(cfg.lo, cfg.hi + 1e-9, width); cen = 0.5 * (edges[1:] + edges[:-1])
grid = np.linspace(cfg.lo, cfg.hi, 800)
fig, axs = plt.subplots(2, 2, figsize=(ps.DOUBLE, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
for col, (side, sg, lab) in enumerate((("pos", -1, r"$B^+$"), ("neg", +1, r"$B^-$"))):
    msk = post.plus if side == "pos" else ~post.plus
    obs, _ = np.histogram(post.m[msk], edges)
    s_c = d["N_s"] * (1 + sg * d["A"]) / 2 * bn._signal_pdf(grid, d, cfg) * width
    b_c = d["N_b"] * (1 + sg * d["A_b"]) / 2 * bn._bkg_pdf(grid, d, side, cfg) * width
    exp = (d["N_s"] * (1 + sg * d["A"]) / 2 * bn._signal_pdf(cen, d, cfg)
           + d["N_b"] * (1 + sg * d["A_b"]) / 2 * bn._bkg_pdf(cen, d, side, cfg)) * width
    pull = (obs - exp) / np.sqrt(exp)
    a = axs[0, col]
    a.errorbar(cen, obs, yerr=np.sqrt(obs), fmt="o", ms=3, color=C["data"], label="Data")
    a.plot(grid, s_c + b_c, color=C["model"], lw=1.5, label="Total model")
    a.plot(grid, s_c, color=C["signal"], ls="--", lw=1.2, label="Signal")
    a.fill_between(grid, 0.5, b_c, color=C["background"], alpha=0.35, label="Background")
    a.set_yscale("log"); a.set_ylim(5, obs.max() * 3)
    ps.panel_label(a, f"{lab}:  $\\chi^2 = {np.sum(pull**2):.1f}$ / {len(cen)} bins")
    p = axs[1, col]
    p.bar(cen, pull, width=width * 0.9, color=C[side], alpha=0.7)
    p.axhline(0, color="k", lw=0.8); [p.axhline(y, color="gray", ls=":") for y in (2, -2)]
    p.set_xlabel(r"$m(KKK)$ [MeV]")
axs[0, 0].set_ylabel(f"Candidates / {width:.0f} MeV"); axs[1, 0].set_ylabel("Pull")
axs[0, 1].legend(loc="upper right")
save(fig, "bayes_nominal_mass_fit.pdf")

# ---------------- magnet polarities
fig, a = plt.subplots(figsize=(ps.SINGLE, 4.4))
for pol in ("down", "up"):
    a.hist(mag[pol], bins=60, density=True, histtype="step", lw=1.5, color=C[pol],
           label=f"Magnet {pol} ($n={summary['checks'][f'magnet_{pol}']['n']}$)")
a.hist(thinned[:, names.index("A")], bins=60, density=True, histtype="step", color="k", lw=2,
       label=f"Combined ($n={nom['n']}$)")
a.axvline(0, color="gray", ls=":")
a.set_xlabel(r"$A_{\mathrm{raw}}$"); a.set_ylabel("Posterior density"); a.legend(loc="upper left")
save(fig, "bayes_nominal_magnet.pdf")

# ---------------- toys: pulls and coverage
from scipy.stats import norm
a_true = nom["A"]["median"]
pull = np.array([(t["A_median"] - a_true) / t["A_sd"] for t in toys])
fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 4.3), gridspec_kw={"width_ratios": [1.6, 1]})
a = axs[0]
a.hist(pull, bins=np.linspace(-3.5, 3.5, 29), density=True, histtype="stepfilled", color=C["replica"], alpha=0.5,
       label=f"{len(pull)} pseudo-datasets")
xs = np.linspace(-3.5, 3.5, 300)
a.plot(xs, norm.pdf(xs), color=C["model"], lw=1.5, label="Standard normal")
a.set_xlabel(r"Pull $(\hat A - A_{\mathrm{true}})/\sigma(\hat A)$"); a.set_ylabel("Density")
ps.panel_label(a, f"mean $= {pull.mean():+.2f}\\pm{pull.std(ddof=1)/np.sqrt(len(pull)):.2f}$\nsd $= {pull.std(ddof=1):.2f}$")
a.legend(loc="upper right")
a = axs[1]
cov = summary["toys"]
for x, nomcov, key in ((0, 0.6827, "coverage_68"), (1, 0.9545, "coverage_95")):
    a.errorbar([x], [cov[key]], yerr=[cov[key + "_err"]], fmt="o", color=C["data"], capsize=4,
               label="Observed" if x == 0 else None)
    a.hlines(nomcov, x - 0.3, x + 0.3, color=C["model"], lw=2, label="Nominal" if x == 0 else None)
a.set_xticks([0, 1]); a.set_xticklabels(["68% interval", "95% interval"]); a.set_xlim(-0.6, 1.6)
a.set_ylabel("Coverage"); a.set_ylim(0.55, 1.0); a.legend(loc="lower right")
plt.tight_layout()
save(fig, "bayes_nominal_toys.pdf")

# ---------------- Dalitz maps, nominal window, significance w.r.t. the global asymmetry
lo2, hi2 = (np.minimum(data.m12, data.m13) / 1e3) ** 2, (np.maximum(data.m12, data.m13) / 1e3) ** 2
q = data.q.to_numpy(); ag = (np.sum(q < 0) - np.sum(q > 0)) / len(q)
el, eh = np.linspace(0, 16, 9), np.linspace(0, 30, 9)
npl, _, _ = np.histogram2d(lo2[q > 0], hi2[q > 0], [el, eh]); nmi, _, _ = np.histogram2d(lo2[q < 0], hi2[q < 0], [el, eh])
n = npl + nmi; ok = n >= 20
with np.errstate(divide="ignore", invalid="ignore"):
    ai = np.where(ok, (nmi - npl) / n, np.nan); sg = (ai - ag) / np.sqrt((1 - ai ** 2) / n)
ext = [0, 16, 0, 30]
fig, axs = plt.subplots(1, 3, figsize=(ps.TRIPLE, 4.3))
im0 = axs[0].imshow(np.where(n > 0, n, np.nan).T, origin="lower", aspect="auto", extent=ext, cmap="Greys")
plt.colorbar(im0, ax=axs[0], label="Candidates per bin")
im1 = axs[1].imshow(ai.T, origin="lower", aspect="auto", extent=ext, cmap="coolwarm", vmin=-0.5, vmax=0.5)
plt.colorbar(im1, ax=axs[1], label=r"$A_i$")
im2 = axs[2].imshow(sg.T, origin="lower", aspect="auto", extent=ext, cmap="coolwarm", vmin=-4, vmax=4)
plt.colorbar(im2, ax=axs[2], label=r"$S_i$")
from matplotlib.patches import Rectangle
for i_ in range(len(el) - 1):
    for j_ in range(len(eh) - 1):
        if ok[i_, j_] and abs(sg[i_, j_]) > 3:
            axs[2].add_patch(Rectangle((el[i_], eh[j_]), el[i_ + 1] - el[i_], eh[j_ + 1] - eh[j_], fill=False, ec="k", lw=1.8))
charm = {r"$D^0$": 1.86484, r"$J/\psi$": 3.0969, r"$\chi_{c0}$": 3.4147}      # GeV, world averages
for a in axs[1:]:
    for lab_, mval in charm.items():
        a.axvline(mval ** 2, color="k", ls="--", lw=0.7, alpha=0.45); a.axhline(mval ** 2, color="k", ls="--", lw=0.7, alpha=0.45)
for lab_, mval in charm.items():
    axs[1].text(mval ** 2 + 0.2, 28.8, lab_, fontsize=10, va="top", rotation=90)
for a in axs:
    a.set_xlabel(r"$m^2_{\mathrm{lo}}(K^+K^-)$ [GeV$^2$]"); a.set_ylabel(r"$m^2_{\mathrm{hi}}(K^+K^-)$ [GeV$^2$]")
plt.tight_layout()
save(fig, "bayes_nominal_dalitz_maps.pdf")

# ---------------- Savage-Dickey: marginal posterior vs prior density at the null value, for A and A_b
fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 4.0))
for a, par, lab in ((axs[0], "A", r"$A$"), (axs[1], "A_b", r"$A_b$")):
    x = thinned[:, names.index(par)]; m_, s_ = nom[par]["mean"], nom[par]["sd"]
    lo_ = min(-0.01, m_ - 5 * s_); hi_ = max(0.01, m_ + 5 * s_)
    a.hist(x, bins=60, range=(lo_, hi_), density=True, histtype="stepfilled", color=C["replica"], alpha=0.45, label="Posterior (MCMC)")
    xs = np.linspace(lo_, hi_, 600)
    a.plot(xs, norm.pdf(xs, m_, s_), color=C["model"], lw=1.5, label="Gaussian approximation")
    a.axhline(0.5, color=C["background"], ls="--", lw=1.5, label=r"Prior, uniform on $(-1,1)$")
    p0 = norm.pdf(0.0, m_, s_)
    a.plot([0], [p0], "o", color=C["model"], ms=7); a.plot([0], [0.5], "o", color=C["background"], ms=7)
    a.axvline(0, color="k", ls=":", lw=0.8)
    a.set_yscale("log"); a.set_ylim(1e-5 if par == "A" else 0.05, 300)
    a.set_xlabel(lab); a.set_ylabel("Density")
    bf = p0 / 0.5
    ps.panel_label(a, f"$BF_{{01}}={bf:.1e}$" if bf < 0.01 else f"$BF_{{01}}={bf:.1f}$", loc="upper right" if par == "A" else "upper left")
axs[0].legend(loc="lower left", fontsize=10)
plt.tight_layout()
save(fig, "bayes_nominal_savage_dickey.pdf")

# ---------------- posterior predictive check and trace plot (Bayesian_Nominal_Checks.ipynb)
ppc = json.load(open(fetch("results/nominal/ppc.json")))
fig, axs = plt.subplots(1, 3, figsize=(ps.TRIPLE, 4.0))
for a, key, lab in ((axs[0], "chi2", r"$\chi^2$ (10 MeV bins, both charges)"), (axs[1], "A_window", r"Counting asymmetry, fit window"),
                    (axs[2], "A_sidebands", r"Counting asymmetry, $|m-\mu|>4\sigma$")):
    o = np.array([r["obs"][key] for r in ppc["rows"]]); r_ = np.array([r["rep"][key] for r in ppc["rows"]])
    a.hist(r_, bins=40, histtype="stepfilled", color=C["replica"], alpha=0.5, label="Replicated data")
    a.axvline(np.median(o), color=C["observed"], lw=2, label=f"Observed ($p_B={ppc['p_B'][key]:.3g}$)" if ppc['p_B'][key] >= 0.01 else "Observed ($p_B<0.001$)")
    a.set_xlabel(lab); a.legend(loc="upper right", fontsize=10)
axs[0].set_ylabel("Replicas")
plt.tight_layout()
save(fig, "bayes_nominal_ppc.pdf")

tr = np.load(fetch("results/nominal/trace_dispersed.npz"))
ch, tnames, thin = tr["chain"], list(tr["names"]), int(tr["thin"])
show = [("A", r"$A$"), ("N_s", r"$N_s$"), ("sigma", r"$\sigma$ [MeV]"), ("f_pos", r"$f_+$")]
steps = np.arange(ch.shape[0]) * thin
n_early = 3000 // thin
fig, axs = plt.subplots(len(show), 2, figsize=(ps.DOUBLE, 7.5), sharex="col", gridspec_kw={"width_ratios": [1, 2.2]})
for row, (k, lab) in enumerate(show):
    y = ch[:, :, tnames.index(k)]
    axs[row, 0].plot(steps[:n_early], y[:n_early], lw=0.4, alpha=0.6)
    axs[row, 1].plot(steps, y, lw=0.3, alpha=0.4)
    lo_q, hi_q = np.percentile(y[ch.shape[0] // 2:], [0.5, 99.5]); pad = 0.5 * (hi_q - lo_q)
    axs[row, 1].set_ylim(lo_q - pad, hi_q + pad)
    axs[row, 1].axvline(ch.shape[0] * thin // 2, color="k", ls=":", lw=1)
    axs[row, 0].set_ylabel(lab)
axs[-1, 0].set_xlabel("Step"); axs[-1, 1].set_xlabel("Step")
plt.tight_layout()
save(fig, "bayes_nominal_trace.pdf")
