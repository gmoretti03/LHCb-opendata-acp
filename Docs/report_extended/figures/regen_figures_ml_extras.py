"""Extra figures of the machine-learning appendix: weak labels on the mass spectrum, regularization scan of the logistic
regression, silhouette and inertia of K-means against K, and the FFNN architecture search.

Inputs: Code_general/Machine_Learning/results/nominal/{data_labels,models,kmeans,ffnn_search}.jsonl/json ,
        raw data via Code_general/Bayesian/bayesian_nominal.load_candidates .
usage (cern env):  python regen_figures_ml_extras.py [OUTDIR]
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_style as ps                                  # also gives the paths: ps.ANALYSIS, ps.DATA
CG = ps.ANALYSIS                                         # LHCb/Code_general/Analysis
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else HERE
os.makedirs(OUT, exist_ok=True)
ML = os.environ.get("REGEN_ML_DIR", os.path.join(CG, "Machine_Learning", "results", "nominal"))
BAYES = os.environ.get("REGEN_BAYES_CODE", os.path.join(CG, "Bayesian"))
DATA = ps.DATA
sys.path.insert(0, HERE)
import plot_style as ps
ps.use(); C = ps.COLORS


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("saved", name)


# ---------------------------------------------------------------- (1) weak labels on the mass spectrum
lab = json.load(open(os.path.join(ML, "data_labels.json")))
mu, sg = lab["mu"], lab["sigma"]
sys.path.insert(0, BAYES)
import bayes_model as bn
m = bn.select(bn.load_candidates(DATA), bn.NOMINAL).m.to_numpy()
sig = np.abs(m - mu) < 2 * sg
bkg = ((m >= 5150) & (m < mu - 4 * sg)) | ((m > mu + 4 * sg) & (m <= 5600))
print("signal-like", sig.sum(), "background-like", bkg.sum(), "(expected %d, %d)" % (lab["n_signal_like"], lab["n_background_like"]))
fig, ax = plt.subplots(figsize=(ps.SINGLE, 4.4))
bins = np.arange(5100, 5600.1, 5)
ax.hist(m, bins=bins, histtype="step", color="k", lw=1.0, label="all candidates")
ax.hist(m[sig], bins=bins, color=C["sig_like"], alpha=0.8, label="signal-like ($|m-\\mu|<2\\sigma$, %d)" % sig.sum())
ax.hist(m[bkg], bins=bins, color=C["bkg_like"], alpha=0.9, label="background-like (sidebands, %d)" % bkg.sum())
ax.axvspan(5100, 5150, facecolor="none", hatch="////", edgecolor="0.5", lw=0, label=r"excluded ($<5150$ MeV)")
ax.axvspan(mu - 4 * sg, mu - 2 * sg, color="0.93"); ax.axvspan(mu + 2 * sg, mu + 4 * sg, color="0.93", label=r"unlabeled gap ($2$--$4\sigma$)")
ax.set_yscale("log"); ax.set_ylim(1, 3e3)
ax.set_xlabel(r"$m(KKK)$ [MeV]"); ax.set_ylabel("Candidates / 5 MeV"); ax.legend(fontsize=8, loc="upper right")
plt.tight_layout(); save(fig, "ml_weak_labels.pdf")

# ---------------------------------------------------------------- (2) regularization scan of the logistic regression
rg = json.load(open(os.path.join(ML, "models.json")))["regularization"]
Cg = np.array(rg["C_grid"])
fig, ax = plt.subplots(figsize=(ps.SINGLE, 4.0))
ax.plot(Cg, rg["auc_l1"], "o-", ms=3.5, color=C["ml"], label=r"$L_1$ (Lasso)")
ax.plot(Cg, rg["auc_l2"], "s-", ms=3.5, color=C["freq"], label=r"$L_2$ (Ridge)")
ax.axvline(rg["C_best_l1"], color=C["ml"], ls=":", lw=1); ax.axvline(rg["C_best_l2"], color=C["freq"], ls=":", lw=1)
ax.set_xscale("log"); ax.set_xlabel(r"$C=1/\alpha$ (weaker regularization $\to$)"); ax.set_ylabel("Cross-validated AUC")
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout(); save(fig, "ml_regularization.pdf")

# ---------------------------------------------------------------- (3) silhouette and inertia against K
es = json.load(open(os.path.join(ML, "kmeans.json")))["elbow_silhouette"]
Ks = sorted(int(k) for k in es)
fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 4.0))
ps.letter(axs[0], "a"); ps.letter(axs[1], "b")
axs[0].plot(Ks, [es[str(k)]["inertia"] for k in Ks], "o-", color=C["count"]); axs[0].set_ylabel(r"Inertia $W=\sum_k\sum_{i\in C_k}\|x_i-\bar x_k\|^2$")
axs[1].plot(Ks, [es[str(k)]["silhouette"] for k in Ks], "o-", color=C["ml"]); axs[1].set_ylabel("Mean silhouette")
for ax in axs:
    ax.axvline(8, color=C["observed"], ls="--", lw=1, label=r"$K=8$ (fixed in advance)")
    ax.axvline(3, color=C["background"], ls="-.", lw=1, label=r"$K=3$ (silhouette maximum)")
    ax.set_xlabel("Number of clusters $K$"); ax.set_xticks(Ks)
axs[1].legend(fontsize=9, loc="upper right")
plt.tight_layout(); save(fig, "ml_kmeans_selection.pdf")

# ---------------------------------------------------------------- (4) FFNN architecture search
rows = [json.loads(l) for l in open(os.path.join(ML, "ffnn_search.jsonl")) if l.strip()]
archs = []
for r in rows:
    if r["arch"] not in archs: archs.append(r["arch"])
size = {a: sum(next(rr["layers"] for rr in rows if rr["arch"] == a)) for a in archs}
archs.sort(key=lambda a: size[a])
fig, ax = plt.subplots(figsize=(ps.SINGLE, 4.4))
rng = np.random.default_rng(1)
for i, a in enumerate(archs):
    v = np.array([r["cv_auc_mean"] for r in rows if r["arch"] == a])
    ax.plot(i + rng.uniform(-0.18, 0.18, len(v)), v, "o", ms=3, color=C["ml"], alpha=0.6)
    ax.plot([i - 0.3, i + 0.3], [v.max()] * 2, color="k", lw=1.5)
ax.set_xticks(range(len(archs))); ax.set_xticklabels([a.replace("_", "\n") for a in archs], fontsize=7)
ax.set_ylabel("Cross-validated AUC"); ax.set_xlabel("Architecture (increasing number of units)")
plt.tight_layout(); save(fig, "ml_ffnn_search.pdf")
