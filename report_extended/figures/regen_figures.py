"""Regenerate the report figures (English labels) from the analysis notebooks.

Re-executes only the notebook cells the figures need -- same code, same seeds -- so the
expensive steps (toy-MC coverage, full-chain systematic study, hyperparameter and FFNN
architecture searches) are skipped; the systematic-study numbers are read from the
results JSON. The notebooks are taken from the `bayes` / `mach_lear` branches with
`git show`, so the script runs from `main` without checking anything out.

usage (from anywhere, cern env):  python regen_figures.py bayes|ml [OUTDIR]
OUTDIR defaults to this directory (Report/figures); figures are written as PDF. Runtime: ~10 min (bayes), ~3 min (ml).
The Bayesian chains reproduce the notebook bit-for-bit. The notebook's FFNN training is not
reproducible; here it is made deterministic, so the FFNN-based figures are reproducible but come
from a different training than the notebook's (test AUC spread between trainings ~0.004).
"""
import json, os, subprocess, sys, tempfile, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = sys.argv[1]
OUT = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else HERE
os.makedirs(OUT, exist_ok=True)
REPO = subprocess.check_output(["git", "-C", HERE, "rev-parse", "--show-toplevel"], text=True).strip()
DATA_DIR = os.path.join(REPO, "LHCb", "opendata-project", "Data")

BRANCH, NB_DIR, NB_NAME, EXTRA = {
    "bayes": ("bayes", "LHCb/Code_general/Bayesian", "Bayesian_Analysis.ipynb",
              ["bayesian_model.py", "results/posterior_summary.json",
               "results/toy_mc_coverage_final.json"]),
    "ml": ("mach_lear", "LHCb/Code_general/Machine_Learning", "ML_Analysis.ipynb", []),
}[MODE]
CELLS = {
    # bayes cells are located by content, so the script keeps working when cells are added
    "bayes": ["import bayesian_model as bm", "FILES = [", "presel = (", "M_K = 493.677",
              "bm.MASS_HI, 61)", "theta0 = np.array([bm.INITIAL_GUESS", "nwalkers = 32",
              "tau = sampler.get_autocorr_time()", "discard = nsteps // 6", "A_median = np.median(A_samples)",
              "theta_med = np.median(flat_samples", "def fit_subset", "def binned_gof(",
              "is_bachelor = charges_arr"],
    "ml": [1, 3, 4, 6, 8, 10, 13, 15, 17, 19, 20, 23, 25, 29, 31, 33, 35, 42, 44, 48, 49],
}[MODE]

# work in a throw-away directory: the notebooks' checkpoint JSONs land there, not in the repo
workdir = tempfile.mkdtemp(prefix=f"regen_{MODE}_")
for rel in [NB_NAME] + EXTRA:
    blob = subprocess.check_output(["git", "-C", REPO, "show", f"{BRANCH}:{NB_DIR}/{rel}"])
    with open(os.path.join(workdir, os.path.basename(rel)), "wb") as f:
        f.write(blob)
os.chdir(workdir)
sys.path.insert(0, workdir)
nb = json.load(open(NB_NAME))
plt.show = lambda *a, **k: plt.close("all")

ns = {"__name__": "__main__"}
if MODE == "ml":
    # the notebook only calls tf.random.set_seed, which does not make Keras training reproducible;
    # fixing all seeds and enabling op determinism makes these figures reproducible run to run
    # (they still differ from the notebook's own, non-reproducible, FFNN training)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
def _find(key):
    hits = [k for k, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and key in "".join(c["source"])]
    assert len(hits) == 1, f"cell marker {key!r} matches {len(hits)} cells"
    return hits[0]

CELLS = [_find(c) if isinstance(c, str) else c for c in CELLS]
for i in CELLS:
    src = "".join(nb["cells"][i]["source"])
    src = src.replace('"../../opendata-project/Data"', repr(DATA_DIR))
    src = src.replace("progress=True", "progress=False")
    t0 = time.time()
    print(f"\n>>> cell [{i}]", flush=True)
    exec(compile(src, f"cell{i}", "exec"), ns)
    print(f"<<< cell [{i}] done in {time.time()-t0:.0f}s", flush=True)
plt.close("all")
g = ns

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
                     "legend.fontsize": 9, "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("saved", name)


if MODE == "bayes":
    import corner
    bm = g["bm"]
    flat = g["flat_samples"].copy()
    flat[:, 5] *= 1e3  # lambda in units of 1e-3 / MeV for readable titles
    labels = [r"$N_s$", r"$N_b$", r"$A$", r"$\mu$ [MeV]", r"$\sigma$ [MeV]",
              r"$\lambda$ [$10^{-3}$ MeV$^{-1}$]"]
    fmts = [".0f", ".0f", ".4f", ".2f", ".2f", ".3f"]
    fig = corner.corner(flat, labels=labels, quantiles=[0.16, 0.5, 0.84],
                        show_titles=False, plot_datapoints=False, label_kwargs={"fontsize": 12})
    axes = np.array(fig.axes).reshape(6, 6)
    for k in range(6):
        q16, q50, q84 = np.percentile(flat[:, k], [16, 50, 84])
        f = fmts[k]
        axes[k, k].set_title(f"{labels[k].split(' [')[0]} $= {q50:{f}}"
                             f"^{{+{q84-q50:{f}}}}_{{-{q50-q16:{f}}}}$", fontsize=11)
    save(fig, "bayes_corner.pdf")

    # mass fit with signal/background components, and pulls underneath
    gof_bins, pull_p, pull_m = g["gof_bins"], g["pull_p"], g["pull_m"]
    N_s, N_b, A, mu, sigma, lam = g["theta_med"]
    m_obs, q_obs = g["m_obs"], g["q_obs"]
    centers = 0.5 * (gof_bins[:-1] + gof_bins[1:])
    bw = gof_bins[1] - gof_bins[0]
    grid = np.linspace(bm.MASS_LO, bm.MASS_HI, 600)
    fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    for col, (qs, pull, lab, c2) in enumerate(((1, pull_p, r"$B^+$", g["chi2_p"]), (-1, pull_m, r"$B^-$", g["chi2_m"]))):
        Nsq = N_s * (1 - qs * A) / 2
        obs, _ = np.histogram(m_obs[q_obs == qs], bins=gof_bins)
        ax = axs[0, col]
        ax.errorbar(centers, obs, yerr=np.sqrt(obs), fmt="o", ms=3, color="k", label="Data")
        sig_c = Nsq * bm.signal_pdf(grid, mu, sigma) * bw
        bkg_c = N_b / 2 * bm.background_pdf(grid, lam) * bw
        ax.plot(grid, sig_c + bkg_c, color="crimson", lw=1.5, label="Total model")
        ax.plot(grid, sig_c, color="steelblue", ls="--", lw=1.2, label="Signal")
        ax.fill_between(grid, 0, bkg_c, color="gray", alpha=0.35, label="Background")
        ax.set_yscale("log")
        ax.set_ylim(5, obs.max() * 3)
        ax.set_title(f"{lab}  ($\\chi^2 = {c2:.1f}$ / 30 bins)")
        axp = axs[1, col]
        axp.bar(centers, pull, width=bw * 0.9, color="steelblue" if qs > 0 else "indianred")
        axp.axhline(0, color="k", lw=0.8)
        for y in (2, -2):
            axp.axhline(y, color="gray", ls=":")
        axp.set_xlabel(r"$m(KKK)$ [MeV]")
    axs[0, 0].set_ylabel("Candidates / 10 MeV")
    axs[1, 0].set_ylabel("Pull")
    axs[0, 1].legend(loc="upper right")
    save(fig, "bayes_mass_fit.pdf")

    # magnet polarity posteriors
    rd, ru = g["result_down"], g["result_up"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rd["A_samples"], bins=60, density=True, histtype="step", lw=1.5,
            label=f"Magnet down ($n={rd['n']}$)")
    ax.hist(ru["A_samples"], bins=60, density=True, histtype="step", lw=1.5,
            label=f"Magnet up ($n={ru['n']}$)")
    ax.hist(g["A_samples"], bins=60, density=True, histtype="step", color="k", lw=2,
            label=f"Combined ($n={len(g['m_obs'])}$)")
    ax.axvline(0, color="gray", ls=":")
    ax.set_xlabel(r"$A_{\mathrm{raw}}$")
    ax.set_ylabel("Posterior density")
    ax.legend(loc="upper left")
    plt.tight_layout()
    save(fig, "bayes_magnet_posterior.pdf")

    # toy-MC: distribution of the posterior-median estimates of A over the 500 pseudo-datasets
    from scipy.stats import norm
    toys = np.array(json.load(open("toy_mc_coverage_final.json"))["toy_A_medians"])
    A_true = float(np.median(g["A_samples"]))
    sd_post = float(np.std(g["A_samples"]))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(toys, bins=30, density=True, histtype="stepfilled", color="steelblue", alpha=0.5,
            label=f"Toy estimates ($n={len(toys)}$): mean $= {toys.mean():.5f}$, sd $= {toys.std(ddof=1):.5f}$")
    xs = np.linspace(toys.min(), toys.max(), 300)
    ax.plot(xs, norm.pdf(xs, A_true, sd_post), color="crimson",
            label=f"Gaussian: injected $A$, posterior sd $= {sd_post:.5f}$")
    ax.axvline(A_true, color="k", ls=":", label=f"Injected $A = {A_true:.4f}$")
    ax.set_xlabel(r"Posterior median of $A$ in each pseudo-dataset")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    plt.tight_layout()
    save(fig, "bayes_toy_estimates.pdf")
    print(f"toys: bias = {toys.mean()-A_true:.5f} +- {toys.std(ddof=1)/np.sqrt(len(toys)):.5f}, "
          f"sd = {toys.std(ddof=1):.5f}, posterior sd = {sd_post:.5f}")

    # fixed-binning Dalitz maps; significance w.r.t. the GLOBAL asymmetry
    m2_lo, m2_hi, q = g["m2_lo_dz"], g["m2_hi_dz"], g["q_dz"]
    rng_lo, rng_hi = [0, 16], [0, 30]
    e_lo, e_hi = np.linspace(*rng_lo, 9), np.linspace(*rng_hi, 9)
    Np, _, _ = np.histogram2d(m2_lo[q > 0], m2_hi[q > 0], bins=[e_lo, e_hi])
    Nm, _, _ = np.histogram2d(m2_lo[q < 0], m2_hi[q < 0], bins=[e_lo, e_hi])
    N = Np + Nm
    A_glob = (np.sum(q < 0) - np.sum(q > 0)) / len(q)
    ok = N >= 20
    with np.errstate(divide="ignore", invalid="ignore"):
        Ai = np.where(ok, (Nm - Np) / N, np.nan)
        sA = np.sqrt((1 - Ai**2) / N)
        Sg = (Ai - A_glob) / sA
    chi2_h = np.nansum(Sg**2)
    print(f"A_global={A_glob:.4f}  bins={ok.sum()}  |S_glob|>3: {np.sum(np.abs(Sg[ok]) > 3)}  "
          f"chi2_hom={chi2_h:.1f}/{ok.sum()-1}")
    ext = [*rng_lo, *rng_hi]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    im0 = axs[0].imshow(np.where(N > 0, N, np.nan).T, origin="lower", aspect="auto", extent=ext, cmap="Greys")
    axs[0].set_title("Candidates per bin")
    plt.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(Ai.T, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axs[1].set_title(r"Local asymmetry $A_i$ (bins with $\geq 20$ candidates)")
    plt.colorbar(im1, ax=axs[1], label=r"$A_i$")
    im2 = axs[2].imshow(Sg.T, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r", vmin=-4, vmax=4)
    axs[2].set_title(rf"$(A_i - A_{{\mathrm{{glob}}}})/\sigma(A_i)$,  $A_{{\mathrm{{glob}}}} = {A_glob:.3f}$")
    plt.colorbar(im2, ax=axs[2], label="local significance")
    for ax in axs:
        ax.set_xlabel(r"$m^2_{\mathrm{lo}}(K^+K^-)$ [GeV$^2$]")
        ax.set_ylabel(r"$m^2_{\mathrm{hi}}(K^+K^-)$ [GeV$^2$]")
    plt.tight_layout()
    save(fig, "bayes_dalitz_maps.pdf")

else:
    # ROC curves, test set
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, sc_ in g["scores_test"].items():
        fpr, tpr, _ = roc_curve(g["y_test"], sc_)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax.set_xlabel("False positive rate (background efficiency)")
    ax.set_ylabel("True positive rate (signal efficiency)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    save(fig, "ml_roc.pdf")

    # Random Forest (Gini) feature importance
    imp = g["rf"].feature_importances_
    order = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(imp)), imp[order], color="steelblue")
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([g["FEATURE_COLS"][k] for k in order], rotation=90)
    ax.set_ylabel("Random Forest feature importance (Gini)")
    plt.tight_layout()
    save(fig, "ml_feature_importance.pdf")

    # isotonic recalibration of the reference RF
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(g["prob_pred"], g["prob_true"], "o-", label=f"Raw RF score (Brier = {g['brier']:.3f})")
    pt_c, pp_c = calibration_curve(g["y_test"], g["score_calibrated"], n_bins=10)
    ax.plot(pp_c, pt_c, "s-", label=f"Isotonic-recalibrated (Brier = {g['brier_calibrated']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction of signal-like events")
    ax.legend(loc="upper left")
    plt.tight_layout()
    save(fig, "ml_calibration_isotonic.pdf")

    # A_raw vs threshold; the degenerate 0.9 point is off-scale and annotated
    thr, A, s = g["thresholds"], np.array(g["A_vs_threshold"]), np.array(g["A_boot_std"])
    fin = np.isfinite(s)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhspan(-0.0497 - 0.0083, -0.0497 + 0.0083, color="crimson", alpha=0.1, label="Bayesian fit, 68% CI")
    ax.axhline(-0.0497, color="crimson", ls=":", label="Bayesian fit (full fit window)")
    ax.errorbar(thr[fin], A[fin], yerr=s[fin], fmt="o-", capsize=4, color="steelblue",
                label="Counting after RF-score cut (bootstrap error)")
    ylo, yhi = -0.12, 0.04
    for t, a in zip(thr[~fin], A[~fin]):
        ax.plot(t, ylo + 0.006, marker="v", ms=9, color="gray")
        ax.annotate(f"threshold {t:.1f}: $A_{{\\mathrm{{raw}}}} = {a:.0f}$\n({int(np.sum(g['scores_full'] > t))} events, off scale)",
                    xy=(t, ylo + 0.006), xytext=(t - 0.30, ylo + 0.03),
                    arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9, color="gray")
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(0.05, 0.95)
    ax.set_xticks(thr)
    ax.set_xlabel("RF-score threshold")
    ax.set_ylabel(r"$A_{\mathrm{raw}}$")
    ax.legend(loc="upper left")
    plt.tight_layout()
    save(fig, "ml_threshold_stability.pdf")

    # epistemic uncertainty: RF ensemble variance vs MC dropout
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    sc = axs[0].scatter(g["rf_mean"], g["rf_std"], c=g["y_test"], cmap="coolwarm", alpha=0.4, s=8, rasterized=True)
    axs[0].set_xlabel("Mean RF score (over trees)")
    axs[0].set_ylabel("Std. dev. of RF score across trees")
    axs[0].set_title("Ensemble-variance uncertainty (RF)")
    cb = plt.colorbar(sc, ax=axs[0], ticks=[0, 1])
    cb.ax.set_yticklabels(["bkg-like (0)", "signal-like (1)"])
    axs[1].hist(g["mc_std"], bins=30, histtype="step", density=True, lw=1.5, label="MC dropout (FFNN)")
    axs[1].hist(g["rf_std"], bins=30, histtype="step", density=True, lw=1.5, label="Ensemble variance (RF)")
    axs[1].set_xlabel("Estimated per-event uncertainty")
    axs[1].set_ylabel("Density")
    axs[1].set_title("Comparison of the two estimators")
    axs[1].legend()
    plt.tight_layout()
    save(fig, "ml_uncertainty_comparison.pdf")

    # K-means partition of the Dalitz plane
    K = g["K_CLUSTERS"]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    cmap = plt.get_cmap("tab10", K)
    sc = ax.scatter(g["m2_lo_ml"], g["m2_hi_ml"], c=g["cluster_id"], cmap=cmap, vmin=-0.5, vmax=K - 0.5,
                    s=3, alpha=0.5, rasterized=True)
    ax.set_xlabel(r"$m^2_{\mathrm{lo}}(K^+K^-)$ [GeV$^2$]")
    ax.set_ylabel(r"$m^2_{\mathrm{hi}}(K^+K^-)$ [GeV$^2$]")
    plt.colorbar(sc, ax=ax, label="Cluster", ticks=range(K))
    plt.tight_layout()
    save(fig, "ml_kmeans_dalitz.pdf")
