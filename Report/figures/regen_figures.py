"""Regenerate the report figures (English labels) from the analysis notebooks.

Re-executes only the notebook cells the figures need -- same code, same seeds -- so the
expensive steps (toy-MC coverage, full-chain systematic study, hyperparameter and FFNN
architecture searches) are skipped; the systematic-study numbers are read from the
results JSON. The notebooks are taken from the `bayes` / `mach_lear` branches with
`git show`, so the script runs from `main` without checking anything out.

usage (from anywhere, cern env):  python regen_figures.py bayes|ml [OUTDIR]
OUTDIR defaults to this directory (Report/figures); figures are written as PDF. Runtime: ~10 min (bayes), ~3 min (ml).
The Bayesian chains reproduce the notebook bit-for-bit; the FFNN (TensorFlow) is not
deterministic, so MC-dropout numbers can differ in the third significant digit.
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
              ["bayesian_model.py", "results/posterior_summary.json"]),
    "ml": ("mach_lear", "LHCb/Code_general/Machine_Learning", "ML_Analysis.ipynb", []),
}[MODE]
CELLS = {
    "bayes": [1, 3, 4, 6, 7, 10, 12, 14, 15, 18, 20, 26, 37, 46],
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

    # pulls
    gof_bins, pull_p, pull_m = g["gof_bins"], g["pull_p"], g["pull_m"]
    centers = 0.5 * (gof_bins[:-1] + gof_bins[1:])
    w = (gof_bins[1] - gof_bins[0]) * 0.9
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, pull, col, lab, c2 in ((axs[0], pull_p, "steelblue", r"$B^+$", g["chi2_p"]),
                                   (axs[1], pull_m, "indianred", r"$B^-$", g["chi2_m"])):
        ax.bar(centers, pull, width=w, color=col)
        ax.axhline(0, color="k", lw=0.8)
        for y in (2, -2):
            ax.axhline(y, color="gray", ls=":")
        ax.set_title(f"Pulls, {lab}  ($\\chi^2 = {c2:.1f}$ / 30 bins)")
        ax.set_xlabel(r"$m(KKK)$ [MeV]")
    axs[0].set_ylabel(r"(obs $-$ exp) / $\sqrt{\mathrm{exp}}$")
    plt.tight_layout()
    save(fig, "bayes_pulls.pdf")

    # magnet polarity posteriors
    rd, ru = g["result_down"], g["result_up"]
    fig, ax = plt.subplots(figsize=(8, 5))
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

    # systematic-variation forest plot (numbers from the full-length-chain study, JSON on `bayes`)
    syst = json.load(open("posterior_summary.json"))["systematic_variations"]
    names_en = [r"Nominal ([5150, 5450] MeV, exp. bkg)", r"Narrower window ([5200, 5450] MeV)",
                r"Reduced high-mass tail ([5150, 5400] MeV)", r"Linear background",
                r"Narrow prior, $A \in [-0.3, 0.3]$"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ys = list(range(len(syst)))[::-1]
    for y, r in zip(ys, syst):
        med, (lo, hi) = r["A_median"], r["A_ci68"]
        ax.errorbar(med, y, xerr=[[med - lo], [hi - med]], fmt="o", capsize=4,
                    color="crimson" if y == ys[0] else "steelblue")
    ax.axvline(syst[0]["A_median"], color="crimson", ls=":", alpha=0.6, label="Nominal median")
    ax.set_yticks(ys)
    ax.set_yticklabels(names_en)
    ax.set_xlabel(r"$A_{\mathrm{raw}}$ (median and 68% CI)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    save(fig, "bayes_systematics_forest.pdf")

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
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(g['y_test'], sc_):.3f})")
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
