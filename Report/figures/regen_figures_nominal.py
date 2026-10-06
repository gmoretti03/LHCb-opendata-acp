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

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
                     "legend.fontsize": 9, "savefig.dpi": 300, "savefig.bbox": "tight"})
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
                    label_kwargs={"fontsize": 12})
ax = np.array(fig.axes).reshape(len(main), len(main))
for k in range(len(main)):
    q16, q50, q84 = np.percentile(th[:, k], [16, 50, 84]); f = fmts[k]
    ax[k, k].set_title(f"{labels[k].split(' [')[0]} $= {q50:{f}}^{{+{q84-q50:{f}}}}_{{-{q50-q16:{f}}}}$", fontsize=11)
save(fig, "bayes_nominal_corner.pdf")

# ---------------- mass fit, log scale, components and pulls
post = bn.Posterior(data.m.to_numpy(), data.q.to_numpy(), cfg)
theta = np.median(thinned, axis=0)
d = post.unpack(theta)
width = 10.0
edges = np.arange(cfg.lo, cfg.hi + 1e-9, width); cen = 0.5 * (edges[1:] + edges[:-1])
grid = np.linspace(cfg.lo, cfg.hi, 800)
fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
for col, (side, sg, lab) in enumerate((("pos", -1, r"$B^+$"), ("neg", +1, r"$B^-$"))):
    msk = post.plus if side == "pos" else ~post.plus
    obs, _ = np.histogram(post.m[msk], edges)
    s_c = d["N_s"] * (1 + sg * d["A"]) / 2 * bn._signal_pdf(grid, d, cfg) * width
    b_c = d["N_b"] * (1 + sg * d["A_b"]) / 2 * bn._bkg_pdf(grid, d, side, cfg) * width
    exp = (d["N_s"] * (1 + sg * d["A"]) / 2 * bn._signal_pdf(cen, d, cfg)
           + d["N_b"] * (1 + sg * d["A_b"]) / 2 * bn._bkg_pdf(cen, d, side, cfg)) * width
    pull = (obs - exp) / np.sqrt(exp)
    a = axs[0, col]
    a.errorbar(cen, obs, yerr=np.sqrt(obs), fmt="o", ms=3, color="k", label="Data")
    a.plot(grid, s_c + b_c, color="crimson", lw=1.5, label="Total model")
    a.plot(grid, s_c, color="steelblue", ls="--", lw=1.2, label="Signal")
    a.fill_between(grid, 0.5, b_c, color="gray", alpha=0.35, label="Background")
    a.set_yscale("log"); a.set_ylim(5, obs.max() * 3)
    a.set_title(f"{lab}  ($\\chi^2 = {np.sum(pull**2):.1f}$ / {len(cen)} bins)")
    p = axs[1, col]
    p.bar(cen, pull, width=width * 0.9, color="steelblue" if side == "pos" else "indianred")
    p.axhline(0, color="k", lw=0.8); [p.axhline(y, color="gray", ls=":") for y in (2, -2)]
    p.set_xlabel(r"$m(KKK)$ [MeV]")
axs[0, 0].set_ylabel(f"Candidates / {width:.0f} MeV"); axs[1, 0].set_ylabel("Pull")
axs[0, 1].legend(loc="upper right")
save(fig, "bayes_nominal_mass_fit.pdf")

# ---------------- magnet polarities
fig, a = plt.subplots(figsize=(6, 4))
for pol in ("down", "up"):
    a.hist(mag[pol], bins=60, density=True, histtype="step", lw=1.5,
           label=f"Magnet {pol} ($n={summary['checks'][f'magnet_{pol}']['n']}$)")
a.hist(thinned[:, names.index("A")], bins=60, density=True, histtype="step", color="k", lw=2,
       label=f"Combined ($n={nom['n']}$)")
a.axvline(0, color="gray", ls=":")
a.set_xlabel(r"$A_{\mathrm{raw}}$"); a.set_ylabel("Posterior density"); a.legend(loc="upper left")
save(fig, "bayes_nominal_magnet.pdf")

# ---------------- toys
from scipy.stats import norm
est = np.array([t["A_median"] for t in toys]); a_true = nom["A"]["median"]; sd = nom["A"]["sd"]
fig, a = plt.subplots(figsize=(6, 4.2))
a.hist(est, bins=25, density=True, histtype="stepfilled", color="steelblue", alpha=0.5,
       label=f"Toy estimates ($n={len(est)}$): mean $= {est.mean():.5f}$, sd $= {est.std(ddof=1):.5f}$")
xs = np.linspace(est.min(), est.max(), 300)
a.plot(xs, norm.pdf(xs, a_true, sd), color="crimson", label=f"Gaussian: injected $A$, posterior sd $= {sd:.5f}$")
a.axvline(a_true, color="k", ls=":", label=f"Injected $A = {a_true:.4f}$")
a.set_xlabel(r"Posterior median of $A$ in each pseudo-dataset"); a.set_ylabel("Density")
a.legend(loc="upper left", fontsize=8.5); a.set_ylim(0, a.get_ylim()[1] * 1.45)
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
fig, axs = plt.subplots(1, 3, figsize=(17, 5))
im0 = axs[0].imshow(np.where(n > 0, n, np.nan).T, origin="lower", aspect="auto", extent=ext, cmap="Greys")
axs[0].set_title("Candidates per bin"); plt.colorbar(im0, ax=axs[0])
im1 = axs[1].imshow(ai.T, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
axs[1].set_title(r"Local asymmetry $A_i$ (bins with $\geq 20$ candidates)"); plt.colorbar(im1, ax=axs[1], label=r"$A_i$")
im2 = axs[2].imshow(sg.T, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r", vmin=-4, vmax=4)
axs[2].set_title(rf"$(A_i - A_{{\mathrm{{glob}}}})/\sigma(A_i)$,  $A_{{\mathrm{{glob}}}} = {ag:.3f}$")
plt.colorbar(im2, ax=axs[2], label="local significance")
for a in axs:
    a.set_xlabel(r"$m^2_{\mathrm{lo}}(K^+K^-)$ [GeV$^2$]"); a.set_ylabel(r"$m^2_{\mathrm{hi}}(K^+K^-)$ [GeV$^2$]")
plt.tight_layout()
save(fig, "bayes_nominal_dalitz_maps.pdf")
