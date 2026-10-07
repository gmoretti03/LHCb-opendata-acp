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

# ---------------- ROC curves, AUC in the legend, working point of the selection used later
from sklearn.metrics import roc_curve, roc_auc_score
y = scores[:, 0]
ref = models["reference_model"]
styles = {"Logistic Regression": ("tab:gray", "-"), "Logistic Regression (L1)": ("tab:gray", "--"),
          "Decision Tree": ("tab:olive", "-"), "Random Forest": ("tab:green", "-"),
          "Gradient Boosting": ("tab:purple", "-"), "FFNN": ("tab:brown", "-")}
fig, ax = plt.subplots(figsize=(ps.SINGLE, 5.4))
for k, name in enumerate(models["order"], 1):
    fpr, tpr, _ = roc_curve(y, scores[:, k]); col, ls_ = styles[name]
    ax.plot(fpr, tpr, color=col, ls=ls_, lw=2.2 if name == ref else 1.1,
            label=f"{name} ({roc_auc_score(y, scores[:, k]):.3f})")
k_ref = models["order"].index(ref) + 1
sel_ = scores[:, k_ref] > 0.5
wp_fpr, wp_tpr = np.mean(sel_[y == 0]), np.mean(sel_[y == 1])
ax.plot([wp_fpr], [wp_tpr], "o", ms=8, mfc="white", mec="k", mew=1.5, label=f"{ref}, score $>0.5$")
ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
ax.set_xlabel("Background-like efficiency (false positive rate)"); ax.set_ylabel("Signal-like efficiency (true positive rate)")
ax.legend(loc="lower right", fontsize=10, title="Test AUC", title_fontsize=10)
save(fig, "ml_nominal_roc.pdf")

# ---------------- feature importance (RF, Gini) and |Lasso coefficient|: two panels, same feature order
names = list(feats["importance"].keys())
imp = np.array([feats["importance"][n] for n in names]); coef = np.array([feats["abs_lasso_coef"][n] for n in names])
order = np.argsort(imp)                                  # ascending: most important at the top of a barh
fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 6.2), sharey=True)
yy = np.arange(len(names))
is_b = np.array([names[i].startswith("B_") for i in order])
for a, val, lab in ((axs[0], imp, "Random Forest importance (Gini)"), (axs[1], coef, "|Lasso coefficient| (standardized features)")):
    a.barh(yy, val[order], color=np.where(is_b, C["model"], C["signal"]), height=0.7)
    a.set_xlabel(lab); a.grid(axis="x", alpha=0.3)
axs[0].set_yticks(yy); axs[0].set_yticklabels([names[i].replace("_", r"\_") if False else names[i] for i in order], fontsize=11)
for t_, b_ in zip(axs[0].get_yticklabels(), is_b):
    if b_: t_.set_color(C["model"])
ps.panel_label(axs[1], f"Spearman $\\rho = {feats['spearman_importance_vs_lasso']:.2f}$", loc="lower right")
plt.tight_layout()
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

# ---------------- calibration (reference RF raw vs isotonic), with the score distribution below
cal = bvc["calibration"]
rf_s = scores[:, models["order"].index("Random Forest") + 1]
edges10 = np.linspace(0, 1, 11)
cnt, _ = np.histogram(rf_s, edges10); cnt = cnt[cnt > 0]                 # calibration_curve drops empty bins
obs_r = np.array(cal["raw"]["obs"]); err_r = np.sqrt(obs_r * (1 - obs_r) / cnt)
fig, (ax, axh) = plt.subplots(2, 1, figsize=(ps.SINGLE, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.06})
ax.errorbar(cal["raw"]["pred"], obs_r, yerr=err_r, fmt="o-", capsize=3, color=C["signal"],
            label=f"Raw RF score (Brier = {cal['brier_raw_rf']:.3f})")
ax.plot(cal["isotonic"]["pred"], cal["isotonic"]["obs"], "s-", color=C["up"], label=f"Isotonic-recalibrated (Brier = {cal['brier_isotonic_rf']:.3f})")
ax.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
ax.set_ylabel("Observed signal-like fraction"); ax.legend(loc="lower right", fontsize=10)
for lab_, msk, col in (("signal-like", y == 1, C["signal"]), ("background-like", y == 0, C["model"])):
    axh.hist(rf_s[msk], bins=edges10, histtype="step", lw=1.5, color=col, label=lab_)
axh.set_yscale("log"); axh.set_xlabel("Predicted probability (raw RF score)"); axh.set_ylabel("Test\ncandidates")
axh.legend(loc="upper left", fontsize=9, ncol=2)
save(fig, "ml_nominal_calibration.pdf")

# ---------------- uncertainty: per-event RF spread vs score; distributions of the three estimators; training size
yt, rf_mean, rf_std, mc_std, de_std = unc_arr.T
fig, axs = plt.subplots(1, 3, figsize=(ps.TRIPLE, 4.3))
eb = np.linspace(0, 1, 21); cb_ = 0.5 * (eb[1:] + eb[:-1])
for lab_, msk, col in (("signal-like", yt == 1, C["signal"]), ("background-like", yt == 0, C["model"])):
    med, lo_, hi_ = [], [], []
    for x0, x1 in zip(eb[:-1], eb[1:]):
        v = rf_std[msk & (rf_mean >= x0) & (rf_mean < x1)]
        med.append(np.median(v) if len(v) >= 10 else np.nan)
        lo_.append(np.percentile(v, 16) if len(v) >= 10 else np.nan); hi_.append(np.percentile(v, 84) if len(v) >= 10 else np.nan)
    axs[0].plot(cb_, med, color=col, lw=1.8, label=lab_)
    axs[0].fill_between(cb_, lo_, hi_, color=col, alpha=0.2)
axs[0].set_xlabel("Mean RF score (over trees)"); axs[0].set_ylabel("Spread across trees"); axs[0].legend(loc="lower center", fontsize=10)
bins = np.linspace(0, max(rf_std.max(), mc_std.max(), de_std.max()), 50)
for arr, lab, col in ((rf_std, "RF trees", "tab:green"), (mc_std, "MC dropout (FFNN)", "tab:brown"), (de_std, "Deep ensemble (FFNN)", "tab:purple")):
    axs[1].hist(arr, bins=bins, histtype="step", density=True, lw=1.5, label=lab, color=col)
axs[1].set_xlabel("Estimated per-event uncertainty"); axs[1].set_ylabel("Density"); axs[1].set_ylim(0, axs[1].get_ylim()[1] * 1.35); axs[1].legend(loc="upper right")
ts = unc["training_size"]
axs[2].plot([t["n_train"] for t in ts], [t["rf_std_mean"] for t in ts], "o-", color="k", label="all")
axs[2].plot([t["n_train"] for t in ts], [t["rf_std_bkg"] for t in ts], "s--", color=C["model"], label="background-like")
axs[2].plot([t["n_train"] for t in ts], [t["rf_std_sig"] for t in ts], "^--", color=C["signal"], label="signal-like")
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
groups = {g["group"]: g for g in km["8"]["all"]["groups"]}
S_ev = np.array([groups[k]["S_vs_global"] for k in km8])
sct = ax.scatter(lo2, hi2, c=S_ev, cmap="coolwarm", vmin=-4, vmax=4, s=3, alpha=0.6, rasterized=True)
for k in range(K):
    sel = km8 == k
    ax.text(np.median(lo2[sel]), np.median(hi2[sel]), f"{groups[k]['S_vs_global']:+.1f}", ha="center", va="center",
            fontsize=10, weight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75))
ax.set_xlabel(r"$m^2_{\mathrm{lo}}(K^+K^-)$ [GeV$^2$]"); ax.set_ylabel(r"$m^2_{\mathrm{hi}}(K^+K^-)$ [GeV$^2$]")
ax.set_xlim(0, 16); ax.set_ylim(0, 30)
plt.colorbar(sct, ax=ax, label=r"$S_k$")
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

# ---------------- ordered Dalitz plot with the resonances, and B+ vs B- mass in the region of LHCb-PAPER-2013-027
reg_mq = npy("paper_region_mass_charge.npy")
from matplotlib.patches import Rectangle
fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 5.0), gridspec_kw={"width_ratios": [1.1, 1]})
a = axs[0]
a.scatter(lo2, hi2, s=0.6, color="k", alpha=0.35, rasterized=True)
res_m = {r"$\phi(1020)$": 1.019461, r"$D^0$": 1.86484, r"$J/\psi$": 3.0969, r"$\chi_{c0}$": 3.4147}   # GeV
for lab_, mval in res_m.items():
    a.axvline(mval ** 2, color=C["model"], ls="--", lw=0.8, alpha=0.7)
    a.text(mval ** 2 + 0.15, 25.5, lab_, color=C["model"], fontsize=10, rotation=90, va="top")
a.add_patch(Rectangle((1.2, 0), 0.8, 15, fill=False, ec=C["signal"], lw=2))
a.text(2.15, 1.0, "LHCb 2013\nregion", color=C["signal"], fontsize=10)
a.set_xlim(0, 16); a.set_ylim(0, 26)
a.set_xlabel(r"$m^2_{\mathrm{lo}}(K^+K^-)$ [GeV$^2$]"); a.set_ylabel(r"$m^2_{\mathrm{hi}}(K^+K^-)$ [GeV$^2$]")
a = axs[1]
eb = np.arange(5100, 5600.1, 10)
for sgn, lab_, col in ((+1, r"$B^+$", C["pos"]), (-1, r"$B^-$", C["neg"])):
    mm = reg_mq[reg_mq[:, 1] == sgn, 0]
    a.hist(mm, bins=eb, histtype="step", lw=1.8, color=col, label=f"{lab_} ({len(mm)} candidates)")
a.set_xlabel(r"$m(KKK)$ [MeV]"); a.set_ylabel("Candidates / 10 MeV"); a.legend(loc="center right")
ps.panel_label(a, "LHCb 2013 region", loc="upper left")
plt.tight_layout()
save(fig, "ml_nominal_dalitz_region.pdf")

# ---------------- high-background sample (ProbK > 0.3): tighter PID cut vs cut on the GBM score
hb = js("high_background.json")
ns0 = hb["pid"][0]["N_sig"]
fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 4.2))
for kind, lab, col, mk in (("pid", r"Tighter PID cut (ProbK $>$ 0.3 ... 0.8)", C["signal"], "o"), ("ml", "Cut on the GBM score (0 ... 0.9)", C["model"], "s")):
    rows = [r for r in hb[kind] if r["N_sig"] / ns0 > 0.15]
    eff = np.array([r["N_sig"] / ns0 for r in rows])
    axs[0].plot(eff, [r["A_err"] for r in rows], mk + "-", color=col, label=lab)
    axs[1].errorbar(eff, [r["A"] for r in rows], yerr=[r["A_err"] for r in rows], fmt=mk + "-", color=col, capsize=3, label=lab)
nom = hb["pid"][2]
for a in axs:
    a.axvline(nom["N_sig"] / ns0, color=C["background"], ls=":", lw=1)
    a.set_xlabel("Signal efficiency (fitted $N_s$ relative to ProbK $>$ 0.3)")
axs[0].set_ylabel(r"$\sigma(A_{\mathrm{raw}})$"); axs[0].legend(loc="upper right", fontsize=10)
axs[1].set_ylabel(r"$A_{\mathrm{raw}}$ (frequentist fit)")
ps.panel_label(axs[0], "dotted: nominal selection", loc="lower left")
plt.tight_layout()
save(fig, "ml_nominal_high_background.pdf")
