"""
bayes_validation.py -- further checks of the nominal Bayesian analysis, from the saved posterior sample (no new MCMC).

  1. Bayes factor of A=0 and A_b=0 as a function of the half-width W of the uniform prior (-W, W)
  2. Savage-Dickey ratios for the equality of the background shapes of B+ and B- (lam1, lam2, f separately)
  3. posterior predictive check with statistics that the fit does not adjust directly:
       T_slices: asymmetry in ten 50-MeV slices of the mass against the one predicted by the model
       D_chi2:   chi2(B+) - chi2(B-) of the binned fit (from results/nominal/ppc.json)
  4. sequential update: posterior of A from magnet down used as prior for magnet up

Reads results/nominal/{nominal_summary.json, nominal_posterior_thinned.npy, magnet_*_A_thinned.npy, ppc.json};
writes results/nominal/validation.json.
usage:  LHCB_DATA_DIR=... python bayes_validation.py
"""
import json, os, sys
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bayesian_nominal as bn

DATA_DIR = os.environ.get("LHCB_DATA_DIR", "../../opendata-project/Data")
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "nominal")
summary = json.load(open(f"{R}/nominal_summary.json"))
nom = summary["nominal"]; names = nom["names"]
th = np.load(f"{R}/nominal_posterior_thinned.npy")
cfg = bn.NOMINAL
res = {}

# 1. Bayes factor vs prior width -------------------------------------------------------------------------------
W = np.geomspace(0.1, 1.0, 19)
res["bf_vs_prior_width"] = {"W": W.tolist()}
for par in ("A", "A_b"):
    m_, s_ = nom[par]["mean"], nom[par]["sd"]
    p0 = norm.pdf(0.0, m_, s_)
    res["bf_vs_prior_width"][par] = dict(post_density_at_0=float(p0), bf01=(p0 * 2 * W).tolist())

# 2. Savage-Dickey for the equality of the background shapes ---------------------------------------------------
eps, lam_max = 1e-5, 0.1
def prior_diff_density_at_0(kind):
    if kind == "lam1":
        return 1.0 / (lam_max - eps)
    if kind == "f":
        return 1.0
    if kind == "lam2":
        # marginal prior of lam2 with lam2 ~ U(eps, lam1), lam1 ~ U(eps, lam_max)
        p = lambda x: np.log((lam_max - eps) / (x - eps)) / (lam_max - eps)
        val, _ = quad(lambda x: p(x) ** 2, eps * (1 + 1e-9), lam_max, limit=200)
        return val
sd_res = {}
for kind, a, b in (("lam1", "lam1_pos", "lam1_neg"), ("lam2", "lam2_pos", "lam2_neg"), ("f", "f_pos", "f_neg")):
    d = th[:, names.index(a)] - th[:, names.index(b)]
    m_, s_ = float(d.mean()), float(d.std())
    p0 = float(norm.pdf(0.0, m_, s_)); pr = float(prior_diff_density_at_0(kind))
    sd_res[kind] = dict(mean=m_, sd=s_, z=m_ / s_, post_density_at_0=p0, prior_density_at_0=pr, bf01=p0 / pr)
res["savage_dickey_background_shapes"] = sd_res

# 3. posterior predictive check with statistics not fitted directly ------------------------------------------
data = bn.select(bn.load_candidates(DATA_DIR), cfg)
m_obs, q_obs = data.m.to_numpy(), data.q.to_numpy()
post = bn.Posterior(m_obs, q_obs, cfg)
edges = np.linspace(cfg.lo, cfg.hi, 11)
grid = np.linspace(cfg.lo, cfg.hi, 5001)
idx_slice = np.digitize(grid, edges[1:-1])

def expected(theta_dict):
    d = post.unpack(np.array([theta_dict[n] for n in names]))
    sig = bn._signal_pdf(grid, d, cfg)
    ep = d["N_s"] * (1 - d["A"]) / 2 * sig + d["N_b"] * (1 - d["A_b"]) / 2 * bn._bkg_pdf(grid, d, "pos", cfg)
    en = d["N_s"] * (1 + d["A"]) / 2 * sig + d["N_b"] * (1 + d["A_b"]) / 2 * bn._bkg_pdf(grid, d, "neg", cfg)
    Ep = np.array([np.trapezoid(ep[idx_slice == k], grid[idx_slice == k]) for k in range(10)])
    En = np.array([np.trapezoid(en[idx_slice == k], grid[idx_slice == k]) for k in range(10)])
    return Ep, En

def T_slices(m, q, Ep, En):
    a_exp = (En - Ep) / (En + Ep)
    var = (1 - a_exp ** 2) / (En + Ep)
    npos, _ = np.histogram(m[q > 0], edges); nneg, _ = np.histogram(m[q < 0], edges)
    a_obs = (nneg - npos) / np.maximum(nneg + npos, 1)
    return float(np.sum((a_obs - a_exp) ** 2 / var))

rng = np.random.default_rng(2026)
n_rep = 1000
sel = rng.choice(len(th), n_rep, replace=False)
To, Tr = [], []
for i in sel:
    td = dict(zip(names, th[i]))
    Ep, En = expected(td)
    m_r, q_r = bn.generate_toy(td, cfg, rng)
    To.append(T_slices(m_obs, q_obs, Ep, En)); Tr.append(T_slices(m_r, q_r, Ep, En))
To, Tr = np.array(To), np.array(Tr)
ppc = json.load(open(f"{R}/ppc.json"))
Do = np.array([r["obs"]["chi2_pos"] - r["obs"]["chi2_neg"] for r in ppc["rows"]])
Dr = np.array([r["rep"]["chi2_pos"] - r["rep"]["chi2_neg"] for r in ppc["rows"]])
res["ppc_unfitted"] = dict(
    n_rep=n_rep, slices_edges=edges.tolist(),
    T_slices=dict(obs_median=float(np.median(To)), rep_mean=float(Tr.mean()), rep_sd=float(Tr.std()), p_B=float(np.mean(Tr >= To)),
                  obs=To.tolist(), rep=Tr.tolist(), ndof=10),
    D_chi2=dict(obs_median=float(np.median(Do)), rep_mean=float(Dr.mean()), rep_sd=float(Dr.std()),
                p_B_two_sided=float(2 * min(np.mean(Dr >= Do), np.mean(Dr <= Do))), obs=Do.tolist(), rep=Dr.tolist()))
# the same statistic for the single posterior draw closest to the median, for the slice-by-slice figure
td = dict(zip(names, np.median(th, axis=0)))
Ep, En = expected(td)
npos, _ = np.histogram(m_obs[q_obs > 0], edges); nneg, _ = np.histogram(m_obs[q_obs < 0], edges)
res["slices"] = dict(centers=(0.5 * (edges[1:] + edges[:-1])).tolist(), A_obs=((nneg - npos) / (nneg + npos)).tolist(),
                     A_err=np.sqrt((1 - ((nneg - npos) / (nneg + npos)) ** 2) / (nneg + npos)).tolist(),
                     A_model=((En - Ep) / (En + Ep)).tolist())

# 4. sequential update ----------------------------------------------------------------------------------------
from scipy.stats import gaussian_kde
d_ = np.load(f"{R}/magnet_down_A_thinned.npy"); u_ = np.load(f"{R}/magnet_up_A_thinned.npy")
xs = np.linspace(-0.14, 0.05, 4000)
pd_, pu_ = gaussian_kde(d_)(xs), gaussian_kde(u_)(xs)
post_seq = pd_ * pu_; post_seq /= np.trapezoid(post_seq, xs)
cdf = np.cumsum(post_seq) * (xs[1] - xs[0])
mean = float(np.trapezoid(xs * post_seq, xs)); sd = float(np.sqrt(np.trapezoid((xs - mean) ** 2 * post_seq, xs)))
res["sequential_update"] = dict(
    down=dict(median=float(np.median(d_)), sd=float(d_.std())), up=dict(median=float(np.median(u_)), sd=float(u_.std())),
    updated=dict(mean=mean, sd=sd, median=float(xs[np.searchsorted(cdf, 0.5)]),
                 ci68=[float(xs[np.searchsorted(cdf, 0.1587)]), float(xs[np.searchsorted(cdf, 0.8413)])]),
    joint_fit=dict(median=nom["A"]["median"], sd=nom["A"]["sd"], ci68=nom["A"]["ci68"]))
json.dump(res, open(f"{R}/validation.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k in ("savage_dickey_background_shapes", "sequential_update")}, indent=1))
print("ppc", json.dumps(res["ppc_unfitted"]["T_slices"]["p_B"]), json.dumps(res["ppc_unfitted"]["D_chi2"]))
