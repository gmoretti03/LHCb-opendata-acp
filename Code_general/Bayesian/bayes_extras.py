"""
bayes_extras.py -- further checks of the Bayesian analysis from saved samples (no long MCMC).

  1. conditional vs marginal posterior of A given A_b = 0 (Gaussian formula, chain, symmetric-background fit)
  2. prior sensitivity by importance reweighting of the chain: log-uniform yields, log-uniform and lifetime-uniform
     background slopes, prior flat in (N+, N-) instead of (N, A)  [reparametrization / Jacobian]
  3. equal-tailed and highest-posterior-density (HPD) 68% intervals of all parameters
  4. evidence by direct grid integration on A (course method), nuisance fixed at the MAP, and from the profile likelihood,
     against the Savage-Dickey ratio
  5. normality (skewness, kurtosis), consistency and scaling of the estimator with N, linear error propagation (low-stat toys)
  6. an own Metropolis-Hastings sampler on the nominal posterior vs emcee
Writes results/nominal/extras.json.   usage: LHCB_DATA_DIR=... python bayes_extras.py
"""
import json, os, sys, time
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bayesian_nominal as bn

DATA_DIR = os.environ.get("LHCB_DATA_DIR", "../../opendata-project/Data")
FREQ_PROFILES = os.environ.get("FREQ_PROFILES", os.path.abspath(os.path.join(
    HERE, "..", "..", "..", "..", "wt_freq", "LHCb", "Code_general", "Frequentist", "results", "validation", "profiles.json")))
R = os.path.join(HERE, "results", "nominal")
summary = json.load(open(f"{R}/nominal_summary.json")); nom = summary["nominal"]; names = nom["names"]
th = np.load(f"{R}/nominal_posterior_thinned.npy")
ia, ib = names.index("A"), names.index("A_b")
res = {}


def wq(x, w, q):
    o = np.argsort(x); c = np.cumsum(w[o]) / w.sum()
    return float(x[o][np.searchsorted(c, q)])


# 1. conditional vs marginal -----------------------------------------------------------------------------------
A, Ab = th[:, ia], th[:, ib]
cov = np.cov(A, Ab)
pred = float(A.mean() + cov[0, 1] / cov[1, 1] * (0.0 - Ab.mean()))
near = np.abs(Ab) < 0.004
res["conditional"] = dict(
    marginal_mean=float(A.mean()), corr=float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])), sd_A=float(A.std()), sd_Ab=float(Ab.std()),
    A_b_mean=float(Ab.mean()), predicted_A_given_Ab0=pred, predicted_sd=float(np.sqrt(cov[0, 0] - cov[0, 1] ** 2 / cov[1, 1])),
    chain_A_given_Ab_near0=dict(n=int(near.sum()), median=float(np.median(A[near])), sd=float(A[near].std())),
    symmetric_background_fit_bayes=summary["variations"]["bkg_symmetric"]["A"]["median"],
    symmetric_background_fit_sd=summary["variations"]["bkg_symmetric"]["A"]["sd"])

# 2. prior reweighting -----------------------------------------------------------------------------------------
col = lambda k: th[:, names.index(k)]
lam = np.vstack([col(k) for k in ("lam1_pos", "lam2_pos", "lam1_neg", "lam2_neg")])
variants = {
    "nominal": np.ones(len(th)),
    "log-uniform N_s, N_b": 1.0 / (col("N_s") * col("N_b")),
    "log-uniform slopes": 1.0 / lam.prod(axis=0),
    "uniform lifetimes 1/lambda": lam.prod(axis=0) ** -2.0,
    "flat in (N_s^+, N_s^-)": col("N_s"),
    "flat in (N_s^+, N_s^-) and (N_b^+, N_b^-)": col("N_s") * col("N_b"),
}
pr = {}
for k, w in variants.items():
    w = w / w.sum()
    pr[k] = dict(median=wq(A, w, 0.5), lo=wq(A, w, 0.1587), hi=wq(A, w, 0.8413), ess=float(1.0 / np.sum(w ** 2)))
res["prior_reweighting"] = pr

# 3. equal-tailed vs HPD ---------------------------------------------------------------------------------------
def hpd(x, p=0.6827):
    s = np.sort(x); n = len(s); k = int(np.ceil(p * n))
    w = s[k:] - s[:n - k]
    j = int(np.argmin(w)); return float(s[j]), float(s[j + k])
rows = {}
for k, n in enumerate(names):
    x = th[:, k]; q16, q84 = np.percentile(x, [15.865, 84.135]); h = hpd(x)
    rows[n] = dict(median=float(np.median(x)), equal_tailed=[float(q16), float(q84)], hpd=list(h),
                   skewness=float(stats.skew(x)))
res["hpd_vs_equal_tailed"] = rows

# 4. evidence: grid integration vs Savage-Dickey ------------------------------------------------------------------
data = bn.select(bn.load_candidates(DATA_DIR), bn.NOMINAL)
post = bn.Posterior(data.m.to_numpy(), data.q.to_numpy(), bn.NOMINAL)
mp = np.array([nom["map"][n] for n in names])
grid = np.linspace(-0.2, 0.2, 801)
ll = []
for a in grid:
    t = mp.copy(); t[ia] = a; ll.append(post(t))
ll = np.array(ll); L = np.exp(ll - ll.max())
Z1 = np.trapezoid(L * 0.5, grid)                  # prior density 1/2 on (-1,1); the likelihood vanishes outside the grid
L0 = np.exp(post(np.where(np.arange(len(mp)) == ia, 0.0, mp)) - ll.max())
ev = dict(K01_grid_nuisance_fixed_at_MAP=float(L0 / Z1))
if os.path.exists(FREQ_PROFILES):
    pf = json.load(open(FREQ_PROFILES))["profile"]["all"]
    a_, q_ = np.array(pf["A"]), np.array(pf["q"])
    c = np.polyfit(a_[q_ < 9], q_[q_ < 9], 2)
    fine = np.linspace(-0.2, 0.2, 4001)
    Lp = np.exp(-0.5 * (np.polyval(c, fine) - np.polyval(c, -c[1] / (2 * c[0]))))
    Zp = np.trapezoid(Lp * 0.5, fine)
    L0p = np.exp(-0.5 * (np.polyval(c, 0.0) - np.polyval(c, -c[1] / (2 * c[0]))))
    ev["K01_profile_likelihood"] = float(L0p / Zp)
ev["savage_dickey_marginal"] = summary["bayes_factor"]["A0_BF01"]
res["evidence"] = ev

# 5. low-statistics toys: normality and scaling ----------------------------------------------------------------
toys = [json.loads(l) for l in open(os.path.join(HERE, "results", "lowstat", "toys.jsonl")) if l.strip()]
truth = json.load(open(os.path.join(HERE, "results", "lowstat", "truth.json")))
f_sig = truth["map"]["N_s"] / (truth["map"]["N_s"] + truth["map"]["N_b"])
a_true = truth["map"]["A"]
rows = {}
for n in sorted({t["n_total"] for t in toys}):
    sel = [t for t in toys if t["n_total"] == n]
    a = np.array([t["freq"]["A"] for t in sel if t["freq"].get("valid")]); b = np.array([t["bayes"]["median"] for t in sel])
    ns = f_sig * n
    rows[str(n)] = dict(n_toys=len(a), mean_bias=float(a.mean() - a_true), sd=float(a.std(ddof=1)),
                        skewness=float(stats.skew(a)), excess_kurtosis=float(stats.kurtosis(a)),
                        shapiro_p=float(stats.shapiro(a).pvalue), sd_times_sqrtN=float(a.std(ddof=1) * np.sqrt(n)),
                        linear_propagation_signal_only=float(np.sqrt((1 - a_true ** 2) / ns)),
                        mean_hesse=float(np.mean([t["freq"]["hesse"] for t in sel if t["freq"].get("valid")])))
res["lowstat_scaling"] = rows

# 6. Metropolis-Hastings by hand -------------------------------------------------------------------------------
def metropolis(logp, x0, cov, n_steps, rng):
    d = len(x0); L = np.linalg.cholesky(cov * 2.38 ** 2 / d)
    x = x0.copy(); lp = logp(x); chain = np.empty((n_steps, d)); acc = 0
    for i in range(n_steps):
        y = x + L @ rng.standard_normal(d)
        lq = logp(y)
        if np.log(rng.random()) < lq - lp:
            x, lp = y, lq; acc += 1
        chain[i] = x
    return chain, acc / n_steps
rng = np.random.default_rng(11)
t0 = time.time()
n_steps = 120000
chain, acc = metropolis(post, mp.copy(), np.cov(th.T), n_steps, rng)
burn = n_steps // 6
ch = chain[burn:]
import emcee
try:
    tau = emcee.autocorr.integrated_time(ch[:, ia], quiet=True)[0]
except Exception:
    tau = float("nan")
res["metropolis_hastings"] = dict(n_steps=n_steps, burn=burn, acceptance=float(acc), seconds=time.time() - t0,
                                  A_median=float(np.median(ch[:, ia])), A_sd=float(ch[:, ia].std()),
                                  A_ci68=[float(np.percentile(ch[:, ia], 15.865)), float(np.percentile(ch[:, ia], 84.135))],
                                  tau_A=float(tau), n_eff=float(len(ch) / tau),
                                  emcee=dict(A_median=nom["A"]["median"], A_sd=nom["A"]["sd"], A_ci68=nom["A"]["ci68"]),
                                  note="Gaussian proposal with the covariance of the emcee posterior scaled by 2.38^2/d")
json.dump(res, open(os.path.join(R, "extras.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("hpd_vs_equal_tailed",)}, indent=1))
