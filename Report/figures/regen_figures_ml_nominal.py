"""Figures of the nominal ML analysis (ML_Nominal.ipynb on the `mach_lear` branch).

No model is retrained: the figures are built from the files the notebook saves in
`Code_general/Machine_Learning/results/nominal/` (test scores, uncertainties, JSON summaries) plus, for the
K-means map, the ROOT data. By default those files come from the `mach_lear` branch via `git show`; set
REGEN_ML_DIR=/path/to/results/dir to read a local directory instead (e.g. a test run).

usage (cern env):  python regen_figures_ml_nominal.py [OUTDIR]        runtime ~1 min
Output names start with `ml_nominal_`, so the figures of the simplified version are not overwritten.
"""
import io, json, os, subprocess, sys, tempfile
import matplotlib, matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else HERE
os.makedirs(OUT, exist_ok=True)
REPO = subprocess.check_output(["git", "-C", HERE, "rev-parse", "--show-toplevel"], text=True).strip()
DATA_DIR = os.path.join(REPO, "LHCb", "opendata-project", "Data")
ML_DIR = "LHCb/Code_general/Machine_Learning"
LOCAL = os.environ.get("REGEN_ML_DIR")


def raw(rel):
    """bytes of `rel` (relative to the ML directory) from REGEN_ML_DIR/the local tree or the mach_lear branch."""
    if LOCAL:
        path = os.path.join(LOCAL, os.path.basename(rel)) if rel.startswith("results/") else os.path.join(LOCAL, "..", "..", rel)
        with open(path, "rb") as f:
            return f.read()
    return subprocess.check_output(["git", "-C", REPO, "show", f"mach_lear:{ML_DIR}/{rel}"])


def js(name):
    return json.loads(raw(f"results/nominal/{name}"))


def npy(name):
    return np.load(io.BytesIO(raw(f"results/nominal/{name}")))


models = js("models.json"); feats = js("features.json"); bvc = js("bias_variance_calibration.json")
unc = js("uncertainty.json"); km = js("kmeans.json"); thr = js("threshold_stability.json")
three = js("three_methods_selected.json"); labels_info = js("data_labels.json")
scores = npy("test_scores.npy"); unc_arr = npy("uncertainty_test.npy"); km8 = npy("kmeans8_labels.npy")

sys.path.insert(0, HERE)
import plot_style as ps
ps.use(); C = ps.COLORS
def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("saved", name)

# ---------------- ROC curves (AUC values are in the table)
from sklearn.metrics import roc_curve
y = scores[:, 0]
fig, ax = plt.subplots(figsize=(ps.SINGLE, 5.4))
for k, name in enumerate(models["order"], 1):
    fpr, tpr, _ = roc_curve(y, scores[:, k])
    ax.plot(fpr, tpr, label=name, lw=1.5 if name == models["reference_model"] else 1.0)
ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
ax.set_xlabel("False positive rate (background efficiency)"); ax.set_ylabel("True positive rate (signal efficiency)")
ax.legend(loc="lower right")
save(fig, "ml_nominal_roc.pdf")

# ---------------- feature importance (RF, Gini) next to |Lasso coefficient|
names = list(feats["importance"].keys())
imp = np.array([feats["importance"][n] for n in names]); coef = np.array([feats["abs_lasso_coef"][n] for n in names])
order = np.argsort(imp)[::-1]
fig, ax1 = plt.subplots(figsize=(ps.DOUBLE, 4.8))
x = np.arange(len(names)); w = 0.4
ax1.bar(x - w / 2, imp[order], w, color="steelblue", label="RF importance (Gini)")
ax1.set_ylabel("RF importance (Gini)")
ax2 = ax1.twinx()
ax2.bar(x + w / 2, coef[order], w, color="darkorange", alpha=0.8, label=r"|Lasso coefficient|")
ax2.set_ylabel("|Lasso coefficient| (standardized features)")
ax1.set_xticks(x); ax1.set_xticklabels([names[i] for i in order], rotation=90)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper right")
save(fig, "ml_nominal_features.pdf")

# ---------------- bias-variance (GBM staged): accuracy and AUC
bv = bvc["bias_variance"]
fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 4.3))
n = np.arange(1, len(bv["accuracy_train"]) + 1)
axs[0].plot(n, bv["accuracy_train"], label="Train"); axs[0].plot(n, bv["accuracy_test"], label="Test")
axs[0].axhline(bv["accuracy_always_signal"], color="gray", ls=":", label="Always signal-like")
axs[0].set_ylabel("Accuracy")
axs[1].plot(n, bv["auc_train"], label="Train"); axs[1].plot(n, bv["auc_test"], label="Test"); axs[1].set_ylabel("AUC")
for a in axs:
    a.set_xlabel("Number of boosting stages"); a.legend()
plt.tight_layout()
save(fig, "ml_nominal_bias_variance.pdf")

# ---------------- calibration (reference RF raw vs isotonic)
cal = bvc["calibration"]
fig, ax = plt.subplots(figsize=(ps.SINGLE, 5.6))
ax.plot(cal["raw"]["pred"], cal["raw"]["obs"], "o-", label=f"Raw RF score (Brier = {cal['brier_raw_rf']:.3f})")
ax.plot(cal["isotonic"]["pred"], cal["isotonic"]["obs"], "s-", label=f"Isotonic-recalibrated (Brier = {cal['brier_isotonic_rf']:.3f})")
ax.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraction of signal-like events"); ax.legend(loc="lower right")
save(fig, "ml_nominal_calibration.pdf")

# ---------------- uncertainty: per-event RF spread vs score; distributions of the three estimators; training size
yt, rf_mean, rf_std, mc_std, de_std = unc_arr.T
fig, axs = plt.subplots(1, 3, figsize=(ps.TRIPLE, 4.3))
sc = axs[0].scatter(rf_mean, rf_std, c=yt, cmap="coolwarm", s=6, alpha=0.4, rasterized=True)
axs[0].set_xlabel("Mean RF score (over trees)"); axs[0].set_ylabel("Std. dev. across trees")
cb = plt.colorbar(sc, ax=axs[0], ticks=[0, 1]); cb.ax.set_yticklabels(["background-\nlike", "signal-\nlike"])
bins = np.linspace(0, max(rf_std.max(), mc_std.max(), de_std.max()), 50)
for arr, lab in ((rf_std, "RF trees"), (mc_std, "MC dropout (FFNN)"), (de_std, "Deep ensemble (FFNN, seeds)")):
    axs[1].hist(arr, bins=bins, histtype="step", density=True, lw=1.5, label=lab)
axs[1].set_xlabel("Estimated per-event uncertainty"); axs[1].set_ylabel("Density"); axs[1].set_ylim(0, axs[1].get_ylim()[1] * 1.35); axs[1].legend(loc="upper right")
ts = unc["training_size"]
axs[2].plot([t["n_train"] for t in ts], [t["rf_std_mean"] for t in ts], "o-", label="all")
axs[2].plot([t["n_train"] for t in ts], [t["rf_std_bkg"] for t in ts], "s--", label="background-like")
axs[2].plot([t["n_train"] for t in ts], [t["rf_std_sig"] for t in ts], "^--", label="signal-like")
axs[2].set_xlabel("Training-set size"); axs[2].set_ylabel("Mean RF spread across trees"); axs[2].legend()
plt.tight_layout()
save(fig, "ml_nominal_uncertainty.pdf")

# ---------------- K-means partition (K=8) of the Dalitz plane
sys.path.insert(0, tempfile.mkdtemp(prefix="mlmod_"))
with open(os.path.join(sys.path[0], "ml_nominal.py"), "wb") as f:
    f.write(raw("ml_nominal.py"))
import ml_nominal as mn
data = mn.load_data(DATA_DIR)
assert len(data) == len(km8), "K-means labels do not match the data sample"
lo2 = (np.minimum(data.m12, data.m13) / 1e3) ** 2; hi2 = (np.maximum(data.m12, data.m13) / 1e3) ** 2
fig, ax = plt.subplots(figsize=(ps.SINGLE, 5.4))
K = int(km8.max()) + 1
cmap = plt.get_cmap("tab10", K)
s = ax.scatter(lo2, hi2, c=km8, cmap=cmap, vmin=-0.5, vmax=K - 0.5, s=3, alpha=0.5, rasterized=True)
groups = {g["group"]: g for g in km["8"]["all"]["groups"]}
for k in range(K):
    if k in groups:
        sel = km8 == k
        ax.text(np.median(lo2[sel]), np.median(hi2[sel]), f"{groups[k]['S_vs_global']:+.1f}", ha="center", va="center",
                fontsize=9, weight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7))
ax.set_xlabel(r"$m^2_{\mathrm{lo}}(K^+K^-)$ [GeV$^2$]"); ax.set_ylabel(r"$m^2_{\mathrm{hi}}(K^+K^-)$ [GeV$^2$]")
plt.colorbar(s, ax=ax, label="Cluster", ticks=range(K))
save(fig, "ml_nominal_kmeans.pdf")

# ---------------- A_raw vs RF/GBM-score threshold, with the nested-subsample band
rows = thr["rows"]; a_full = thr["A_count_full"]
t_ = np.array([r["threshold"] for r in rows]); a_ = np.array([r["A"] for r in rows])
sb = np.array([r["sigma_shift_nested"] for r in rows]); sbin = np.array([r["sigma_binomial"] for r in rows])
fig, ax = plt.subplots(figsize=(ps.SINGLE, 4.4))
ax.fill_between(t_, a_full - sb, a_full + sb, color="gray", alpha=0.25, label="Expected spread (nested subsample)")
ax.axhline(a_full, color="k", ls=":", label="Counting, full fit window")
ax.errorbar(t_, a_, yerr=sbin, fmt="o-", capsize=3, color="steelblue", label="Counting after the score cut")
ax.set_xlabel(f"{models['reference_model']} score threshold"); ax.set_ylabel(r"$A_{\mathrm{raw}}$ (counting)")
ax.legend(loc="lower left")
save(fig, "ml_nominal_threshold.pdf")

# ---------------- three approaches, full vs selected sample
fig, ax = plt.subplots(figsize=(ps.SINGLE, 4.2))
lab = [("bayes", "Bayesian fit", "sd"), ("frequentist", "Frequentist fit", "A_err"), ("counting", "Counting", "sd")]
for i, (k, name, e) in enumerate(lab):
    f, s_ = three[k]["full"], three[k]["selected"]
    ax.errorbar([f["A"]], [i + 0.12], xerr=[f[e]], fmt="o", color="k", capsize=3, label="Full sample" if i == 0 else None)
    ax.errorbar([s_["A"]], [i - 0.12], xerr=[s_[e]], fmt="s", color=C["model"], capsize=3,
                label=f"Selected (score > {three['threshold']})" if i == 0 else None)
ax.set_yticks(range(len(lab))); ax.set_yticklabels([n for _, n, _ in lab])
ax.set_xlabel(r"$A_{\mathrm{raw}}$"); ax.invert_yaxis()
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
save(fig, "ml_nominal_three_methods.pdf")

# ---------------- mass spectrum before/after the score cut and cut efficiency vs mass (from saved counts)
orig = js("selection_mass_origin.json")
fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 4.3))
for key, lab, sty in (("all_20", "Selected, 20 features", "-"), ("no_momenta", "Selected, no momentum features", "--")):
    ev = orig[key]["efficiency_vs_mass"]
    lo_ = np.array([e["lo"] for e in ev]); hi_ = np.array([e["hi"] for e in ev])
    n_ = np.array([e["n"] for e in ev]); eff = np.array([e["eff"] for e in ev])
    if key == "all_20":
        axs[0].stairs(n_, np.append(lo_, hi_[-1]), color="k", lw=1.5, label="All candidates")
    axs[0].stairs(n_ * eff, np.append(lo_, hi_[-1]), lw=1.5, ls=sty, label=lab)
    axs[1].stairs(eff, np.append(lo_, hi_[-1]), lw=1.5, ls=sty, label=lab)
mu_, sg_ = labels_info["mu"], labels_info["sigma"]
for a in axs:
    a.axvspan(5100, 5150, color="tab:orange", alpha=0.12)
    a.axvspan(mu_ - 2 * sg_, mu_ + 2 * sg_, color="tab:green", alpha=0.10)
    a.set_xlabel(r"$m(KKK)$ [MeV]")
axs[0].set_yscale("log"); axs[0].set_ylabel("Candidates / 25 MeV"); axs[0].legend(loc="upper right")
axs[1].set_ylabel("Fraction passing the cut"); axs[1].set_ylim(0, 1); axs[1].legend(loc="upper right")
plt.tight_layout()
save(fig, "ml_nominal_mass_after_cut.pdf")
