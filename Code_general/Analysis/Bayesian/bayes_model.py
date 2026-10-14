"""
bayes_model.py -- Bayesian mass model and tools shared by the notebooks 01-03 of this directory.

Nominal model for B+- -> K+- K+ K- (common to the frequentist analysis).

    fit window   [5100, 5600] MeV
    signal       Gaussian truncated to the window, mu and sigma shared by the two charges
    background   double exponential, shape (lam1 > lam2, f) SEPARATE for B+ and B-,
                 yields N_b^pm = N_b (1 -+ A_b) / 2  (A_b free)
    signal yields N_s^pm = N_s (1 -+ A) / 2,  i.e.  A = (N_s^- - N_s^+) / (N_s^- + N_s^+)
    likelihood   extended, unbinned, product over the two charge samples
    priors       independent uniform (bounds in `_in_prior`)

Every variation of the systematic study is a `Config` (see VARIATIONS). Toys and variations are
independent, so they run in parallel (`run_parallel`), with per-task checkpoints so that an
interrupted run resumes instead of starting again.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace

import logging

import emcee
import numpy as np
import pandas as pd
import uproot
from scipy.optimize import minimize
from scipy.stats import crystalball, norm

logging.getLogger("emcee").setLevel(logging.ERROR)   # tau warnings are reported in the results instead

M_K = 493.677
CHARM_STATES = {"D0": 1864.84, "Jpsi": 3096.9, "chic0": 3414.7}   # MeV (PDG), decaying to K+K-
# veto half-widths, from the peak widths measured in m(K+K-) of the nominal window (2026-10-06):
# D0 sigma 6.5 MeV, J/psi ~8 MeV, chi_c0 ~18 MeV with its peak ~11 MeV below the PDG mass (natural width)
CHARM_HALFWIDTH = {"D0": 30.0, "Jpsi": 30.0, "chic0": 60.0}


# ---------------------------------------------------------------- configuration
@dataclass(frozen=True)
class Config:
    name: str = "nominal"
    lo: float = 5100.0
    hi: float = 5600.0
    nexp: int = 2                 # 1 or 2 exponentials in the background
    shape: str = "charge"         # "charge": separate shape for B+ and B-; "shared"
    asym: str = "free"            # "free": A_b free; "sym": A_b = 0
    signal: str = "gauss"         # "gauss" or "cb" (Crystal Ball, free beta and m)
    A_bound: float = 1.0          # prior A in (-A_bound, A_bound)
    veto: tuple = ()              # charm states vetoed, e.g. ("D0",); half-widths in CHARM_HALFWIDTH,
                                  # applied to both opposite-sign K+K- pairs
    signal_per_charge: bool = False  # mu and sigma separate for B+ and B-
    prob_k_min: float = 0.5       # preselection: ProbK > prob_k_min and ProbPi < prob_pi_max on all three tracks
    prob_pi_max: float = 0.5


NOMINAL = Config()
# common list of variations, identical in the frequentist analysis (freq: 06_frequentist_fit_stability_systematics.ipynb)
VARIATIONS = [
    replace(NOMINAL, name="window_lo5050", lo=5050.0),
    replace(NOMINAL, name="window_lo5150", lo=5150.0),
    replace(NOMINAL, name="window_hi5450", hi=5450.0),
    replace(NOMINAL, name="window_hi5700", hi=5700.0),
    replace(NOMINAL, name="bkg_single_exp", nexp=1),
    replace(NOMINAL, name="bkg_shared_shape", shape="shared"),
    replace(NOMINAL, name="bkg_symmetric", asym="sym"),
    replace(NOMINAL, name="signal_crystal_ball", signal="cb"),
    replace(NOMINAL, name="signal_per_charge", signal_per_charge=True),
    replace(NOMINAL, name="presel_probk_0.4", prob_k_min=0.4),
    replace(NOMINAL, name="presel_probk_0.6", prob_k_min=0.6),
    replace(NOMINAL, name="presel_probpi_0.4", prob_pi_max=0.4),
    replace(NOMINAL, name="presel_probpi_0.6", prob_pi_max=0.6),
]
SYST_GROUPS = {
    "window": ["window_lo5050", "window_lo5150", "window_hi5450", "window_hi5700"],
    "background": ["bkg_single_exp", "bkg_shared_shape", "bkg_symmetric"],
    "signal": ["signal_crystal_ball", "signal_per_charge"],
    "preselection": ["presel_probk_0.4", "presel_probk_0.6", "presel_probpi_0.4", "presel_probpi_0.6"],
}
# variations that change the sample: Barlow criterion applies
SAMPLE_CHANGING = SYST_GROUPS["window"] + SYST_GROUPS["preselection"]
# prior sensitivity: Bayesian-only check, not part of the systematic uncertainty
PRIOR_CHECKS = [replace(NOMINAL, name="prior_A_narrow", A_bound=0.3)]
# charm vetoes: informative robustness checks only (not in the systematic uncertainty), same
# definition as in the frequentist analysis; "all" is the reference one
CHARM_VETOES = [replace(NOMINAL, name="veto_D0", veto=("D0",)),
                replace(NOMINAL, name="veto_D0_Jpsi_chic0", veto=("D0", "Jpsi", "chic0"))]


# ---------------------------------------------------------------- data
def load_candidates(data_dir: str) -> pd.DataFrame:
    """Read both polarities, apply a preselection looser than every variation (ProbK > 0.4,
    ProbPi < 0.6, muon veto; `select` then applies the cuts of each Config), add B mass/charge,
    the two opposite-sign K+K- masses (m12, m13; H1 is the kaon with charge opposite to the B) and
    the per-candidate min ProbK / max ProbPi over the three tracks."""
    br = [f"H{i}_{v}" for i in (1, 2, 3) for v in ("PX", "PY", "PZ", "ProbK", "ProbPi", "isMuon", "Charge")]
    frames = []
    for pol in ("Down", "Up"):
        with uproot.open(os.path.join(data_dir, f"B2HHH_Magnet{pol}.root")) as f:
            d = f["DecayTree"].arrays(br, library="pd")
        d["magnet"] = pol.lower()
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    sel = np.ones(len(df), bool)
    for i in (1, 2, 3):
        sel &= (df[f"H{i}_ProbPi"] < 0.6) & (df[f"H{i}_ProbK"] > 0.4) & (~df[f"H{i}_isMuon"].astype(bool))
    df = df[sel].reset_index(drop=True)
    p = {i: df[[f"H{i}_PX", f"H{i}_PY", f"H{i}_PZ"]].to_numpy() for i in (1, 2, 3)}
    e = {i: np.sqrt((p[i] ** 2).sum(1) + M_K ** 2) for i in (1, 2, 3)}
    def mass(*idx):
        return np.sqrt(np.clip(sum(e[i] for i in idx) ** 2 - (sum(p[i] for i in idx) ** 2).sum(1), 0, None))
    out = pd.DataFrame({"m": mass(1, 2, 3), "q": (df.H1_Charge + df.H2_Charge + df.H3_Charge).to_numpy(),
                        "m12": mass(1, 2), "m13": mass(1, 3), "magnet": df["magnet"].to_numpy(),
                        "min_probk": df[["H1_ProbK", "H2_ProbK", "H3_ProbK"]].min(axis=1).to_numpy(),
                        "max_probpi": df[["H1_ProbPi", "H2_ProbPi", "H3_ProbPi"]].max(axis=1).to_numpy()})
    assert np.all(df.H1_Charge.to_numpy() == -out.q.to_numpy()), "H1 is expected to be the opposite-sign kaon"
    return out[np.isin(out.q, [1, -1])].reset_index(drop=True)


def select(cands: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    keep = (cands.m > cfg.lo) & (cands.m < cfg.hi) & (cands.min_probk > cfg.prob_k_min) & (cands.max_probpi < cfg.prob_pi_max)
    for st in cfg.veto:
        m0 = CHARM_STATES[st]
        hw = CHARM_HALFWIDTH[st]
        keep &= ~((np.abs(cands.m12 - m0) < hw) | (np.abs(cands.m13 - m0) < hw))
    return cands[keep].reset_index(drop=True)


# ---------------------------------------------------------------- model
def param_names(cfg: Config) -> list:
    names = ["N_s", "N_b", "A"] + (["A_b"] if cfg.asym == "free" else []) \
        + (["mu_pos", "mu_neg", "sigma_pos", "sigma_neg"] if cfg.signal_per_charge else ["mu", "sigma"])
    if cfg.signal == "cb":
        names += ["beta", "m_cb"]
    sides = ["pos", "neg"] if cfg.shape == "charge" else ["all"]
    for s in sides:
        names += [f"lam1_{s}"] + ([f"lam2_{s}", f"f_{s}"] if cfg.nexp == 2 else [])
    return names


def _signal_pdf(m, d, cfg, side=None):
    mu = d[f"mu_{side}"] if cfg.signal_per_charge else d["mu"]
    s = d[f"sigma_{side}"] if cfg.signal_per_charge else d["sigma"]
    if cfg.signal == "gauss":
        return norm.pdf(m, mu, s) / (norm.cdf(cfg.hi, mu, s) - norm.cdf(cfg.lo, mu, s))
    b, mm = d["beta"], d["m_cb"]
    z = crystalball.cdf((cfg.hi - mu) / s, b, mm) - crystalball.cdf((cfg.lo - mu) / s, b, mm)
    return crystalball.pdf((m - mu) / s, b, mm) / s / z


def _exp_pdf(m, lam, cfg):
    return lam * np.exp(-lam * (m - cfg.lo)) / (1.0 - np.exp(-lam * (cfg.hi - cfg.lo)))


def _bkg_pdf(m, d, side, cfg):
    k = side if cfg.shape == "charge" else "all"
    if cfg.nexp == 1:
        return _exp_pdf(m, d[f"lam1_{k}"], cfg)
    f = d[f"f_{k}"]
    return f * _exp_pdf(m, d[f"lam1_{k}"], cfg) + (1 - f) * _exp_pdf(m, d[f"lam2_{k}"], cfg)


def _in_prior(d, cfg, n):
    if not (0 < d["N_s"] < 5 * n and 0 < d["N_b"] < 5 * n and -cfg.A_bound < d["A"] < cfg.A_bound
            and -1 < d.get("A_b", 0.0) < 1):
        return False
    for k in (("mu_pos", "mu_neg", "sigma_pos", "sigma_neg") if cfg.signal_per_charge else ("mu", "sigma")):
        lo_, hi_ = (5260, 5310) if k.startswith("mu") else (3, 40)
        if not lo_ < d[k] < hi_:
            return False
    if cfg.signal == "cb" and not (0.3 < d["beta"] < 5 and 1.01 < d["m_cb"] < 50):
        return False
    for k in (["pos", "neg"] if cfg.shape == "charge" else ["all"]):
        if not 1e-5 < d[f"lam1_{k}"] < 0.1:
            return False
        if cfg.nexp == 2 and not (1e-5 < d[f"lam2_{k}"] < d[f"lam1_{k}"] and 0 < d[f"f_{k}"] < 1):
            return False      # lam1 > lam2 removes the label-switching symmetry of the two exponentials
    return True


def prior_volume_A(cfg: Config) -> float:
    return 2.0 * cfg.A_bound


class Posterior:
    """log posterior for one Config and one data sample; picklable (used by the worker processes)."""

    def __init__(self, m, q, cfg: Config):
        self.m, self.plus, self.cfg = np.asarray(m, float), np.asarray(q) > 0, cfg
        self.names, self.n = param_names(cfg), len(m)

    def unpack(self, th):
        d = dict(zip(self.names, th))
        d.setdefault("A_b", 0.0)
        return d

    def __call__(self, th):
        d = self.unpack(th)
        if not _in_prior(d, self.cfg, self.n):
            return -np.inf
        m, p, cfg = self.m, self.plus, self.cfg
        sp, sn = _signal_pdf(m[p], d, cfg, "pos"), _signal_pdf(m[~p], d, cfg, "neg")
        dp = d["N_s"] * (1 - d["A"]) / 2 * sp + d["N_b"] * (1 - d["A_b"]) / 2 * _bkg_pdf(m[p], d, "pos", cfg)
        dn = d["N_s"] * (1 + d["A"]) / 2 * sn + d["N_b"] * (1 + d["A_b"]) / 2 * _bkg_pdf(m[~p], d, "neg", cfg)
        if np.any(dp <= 0) or np.any(dn <= 0) or not (np.all(np.isfinite(dp)) and np.all(np.isfinite(dn))):
            return -np.inf
        return float(np.sum(np.log(dp)) + np.sum(np.log(dn)) - (d["N_s"] + d["N_b"]))


def _start(cfg: Config, n: int) -> dict:
    st = {"N_s": 0.78 * n, "N_b": 0.22 * n, "A": -0.04, "A_b": 0.0, "mu": 5284.4, "sigma": 18.0,
          "beta": 1.8, "m_cb": 10.0, "mu_pos": 5284.4, "mu_neg": 5284.4, "sigma_pos": 18.0, "sigma_neg": 18.0}
    for k in ("pos", "neg", "all"):
        st[f"lam1_{k}"] = 0.03 if cfg.nexp == 2 else 0.004
        st[f"lam2_{k}"], st[f"f_{k}"] = 0.002, 0.15
    return st


# ---------------------------------------------------------------- fit
def fit(m, q, cfg: Config, nsteps=25000, nwalkers=32, burn=None, seed=42, dispersed=False,
        keep_thinned=0, start=None) -> dict:
    """MAP by Nelder-Mead, then emcee. Walkers start in a tight ball around the MAP, or (dispersed=True)
    uniformly over the prior box. Returns medians, 16/84 and 2.5/97.5 percentiles, sd, MAP, per-parameter
    autocorrelation time, acceptance, and optionally a thinned posterior sample."""
    t0 = time.time()
    post = Posterior(m, q, cfg)
    names, nd = post.names, len(post.names)
    nwalkers = max(nwalkers, 2 * nd + 2)
    if burn is None:
        burn = nsteps // 2 if dispersed else nsteps // 6   # dispersed walkers start far from the bulk
    st = _start(cfg, post.n) if start is None else {**_start(cfg, post.n), **start}
    x0 = np.array([st[k] for k in names])
    best = x0
    for tol in (1e-6, 1e-8):
        res = minimize(lambda t: -post(t), best, method="Nelder-Mead",
                       options={"maxiter": 60000, "maxfev": 60000, "xatol": tol, "fatol": 1e-4})
        best = res.x
    rng = np.random.default_rng(seed)
    if dispersed:
        lo_, hi_ = _prior_box(cfg, post.n, names)
        pos = lo_ + (hi_ - lo_) * rng.uniform(0.02, 0.98, (nwalkers, nd))
    else:
        pos = best + np.abs(best) * 1e-3 * rng.standard_normal((nwalkers, nd))
        for k in ("A", "A_b"):
            if k in names:
                pos[:, names.index(k)] = best[names.index(k)] + 1e-3 * rng.standard_normal(nwalkers)
    for w in range(nwalkers):                 # every walker must start where the posterior is non-zero
        tries = 0
        while not np.isfinite(post(pos[w])):
            pos[w] = (best + np.abs(best) * 1e-4 * rng.standard_normal(nd)) if not dispersed else \
                lo_ + (hi_ - lo_) * rng.uniform(0.02, 0.98, nd)
            tries += 1
            if tries > 5000:
                raise RuntimeError(f"{cfg.name}: no valid starting point")
    np.random.seed(seed)
    sam = emcee.EnsembleSampler(nwalkers, nd, post)
    sam.run_mcmc(pos, nsteps, progress=False)
    flat = sam.get_chain(discard=burn, flat=True)
    tau = sam.get_autocorr_time(discard=burn, quiet=True)
    out = {"config": asdict(cfg), "n": int(post.n), "n_pos": int(post.plus.sum()), "n_neg": int((~post.plus).sum()),
           "names": names, "nwalkers": nwalkers, "nsteps": nsteps, "burn": burn, "dispersed": dispersed,
           "acceptance": float(np.mean(sam.acceptance_fraction)),
           "tau": dict(zip(names, map(float, tau))), "chain_over_tau_min": float((nsteps - burn) / np.max(tau)),
           "map": dict(zip(names, map(float, best))), "logp_map": float(post(best)),
           "seconds": round(time.time() - t0, 1)}
    for k, nm in enumerate(names):
        x = flat[:, k]
        out[nm] = {"median": float(np.median(x)), "mean": float(np.mean(x)), "sd": float(np.std(x)),
                   "ci68": [float(v) for v in np.percentile(x, [16, 84])],
                   "ci95": [float(v) for v in np.percentile(x, [2.5, 97.5])]}
    if dispersed:
        # converged when every walker's log-posterior is within 20 of the best value found by the MAP search
        # (the spread of log-posterior values in the bulk is ~ dim/2 = 6, so 20 is far outside the bulk)
        lp = sam.get_log_prob()
        ok = lp > out["logp_map"] - 20
        out["steps_to_bulk"] = int(np.argmax(ok.all(axis=1))) if ok.all(axis=1).any() else -1
        out["walkers_in_bulk_at_end"] = int(ok[-1].sum())
    out["gof"] = binned_gof(post, np.median(flat, axis=0))
    if keep_thinned:
        idx = np.random.default_rng(seed + 1).choice(len(flat), size=min(keep_thinned, len(flat)), replace=False)
        out["_thinned"] = flat[idx]
    return out


def _prior_box(cfg, n, names):
    box = {"N_s": (0, 5 * n), "N_b": (0, 5 * n), "A": (-cfg.A_bound, cfg.A_bound), "A_b": (-1, 1),
           "mu": (5260, 5310), "sigma": (3, 40), "beta": (0.3, 5), "m_cb": (1.01, 50)}
    lo_, hi_ = [], []
    for k in names:
        base = k.rsplit("_", 1)[0] if k.endswith(("_pos", "_neg")) and k.startswith(("mu", "sigma")) else k
        a, b = box.get(base, (1e-5, 0.1)) if not k.startswith("f_") else (0, 1)
        lo_.append(a); hi_.append(b)
    return np.array(lo_), np.array(hi_)


def binned_gof(post: Posterior, theta, width=10.0) -> dict:
    """Pearson chi2 of the data against the model at `theta`, `width`-MeV bins, both charges."""
    d, cfg = post.unpack(theta), post.cfg
    edges = np.arange(cfg.lo, cfg.hi + 1e-9, width)
    cen = 0.5 * (edges[1:] + edges[:-1])
    chi2, per = 0.0, {}
    for side, sg, msk in (("pos", -1, post.plus), ("neg", +1, ~post.plus)):
        obs, _ = np.histogram(post.m[msk], edges)
        exp = (d["N_s"] * (1 + sg * d["A"]) / 2 * _signal_pdf(cen, d, cfg, side)
               + d["N_b"] * (1 + sg * d["A_b"]) / 2 * _bkg_pdf(cen, d, side, cfg)) * width
        per[side] = float(np.sum((obs - exp) ** 2 / exp))
        chi2 += per[side]
    ndof = 2 * (len(edges) - 1) - len(post.names)
    from scipy.stats import chi2 as _chi2
    return {"chi2": chi2, "ndof": int(ndof), "p": float(_chi2.sf(chi2, ndof)), "per_charge": per, "bin_MeV": width}


# ---------------------------------------------------------------- toys
def generate_toy(theta: dict, cfg: Config, rng) -> tuple:
    """Pseudo-dataset from the fitted model (Poisson yields per charge and component)."""
    ms, qs = [], []
    for side, sg, charge in (("pos", -1, 1), ("neg", +1, -1)):
        ns = rng.poisson(theta["N_s"] * (1 + sg * theta["A"]) / 2)
        nb = rng.poisson(theta["N_b"] * (1 + sg * theta.get("A_b", 0.0)) / 2)
        a, b = (cfg.lo - theta["mu"]) / theta["sigma"], (cfg.hi - theta["mu"]) / theta["sigma"]
        u = rng.uniform(norm.cdf(a), norm.cdf(b), ns)
        sig = theta["mu"] + theta["sigma"] * norm.ppf(u)
        k = side if cfg.shape == "charge" else "all"
        if cfg.nexp == 2:
            first = rng.uniform(size=nb) < theta[f"f_{k}"]
            lam = np.where(first, theta[f"lam1_{k}"], theta[f"lam2_{k}"])
        else:
            lam = np.full(nb, theta[f"lam1_{k}"])
        w = cfg.hi - cfg.lo
        bkg = cfg.lo - np.log(1 - rng.uniform(size=nb) * (1 - np.exp(-lam * w))) / lam
        ms += [sig, bkg]
        qs += [np.full(ns + nb, charge)]
    return np.concatenate(ms), np.concatenate(qs)


def _toy_task(args):
    i, theta, cfg, nsteps, seed = args
    rng = np.random.default_rng(seed)
    m, q = generate_toy(theta, cfg, rng)
    start = {k: v for k, v in theta.items()}
    r = fit(m, q, cfg, nsteps=nsteps, seed=seed, start=start)
    a_true = theta["A"]
    return {"toy": i, "seed": seed, "n": r["n"], "A_median": r["A"]["median"], "A_sd": r["A"]["sd"],
            "A_ci68": r["A"]["ci68"], "A_ci95": r["A"]["ci95"],
            "in68": bool(r["A"]["ci68"][0] < a_true < r["A"]["ci68"][1]),
            "in95": bool(r["A"]["ci95"][0] < a_true < r["A"]["ci95"][1]),
            "tau_A": r["tau"]["A"], "chain_over_tau_min": r["chain_over_tau_min"], "seconds": r["seconds"]}


def _fit_task(args):
    """args = (cfg, m, q, nsteps, seed, dispersed[, path]); with `path`, the A column of a thinned
    posterior sample is saved there as .npy (used for figures)."""
    cfg, m, q, nsteps, seed, dispersed = args[:6]
    path = args[6] if len(args) > 6 else None
    r = fit(m, q, cfg, nsteps=nsteps, seed=seed, dispersed=dispersed, keep_thinned=20000 if path else 0)
    th = r.pop("_thinned", None)
    if path is not None:
        np.save(path, th[:, r["names"].index("A")])
    return r


def run_parallel(tasks, worker, checkpoint: str, key, n_workers=None, log_every=1):
    """Run `worker(task)` for every task in parallel. Results are appended to the JSON-lines file
    `checkpoint` as they complete; tasks whose key is already there are skipped (resume)."""
    done = {}
    if os.path.exists(checkpoint):
        with open(checkpoint) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[key(r)] = r
    todo = [t for t in tasks if _task_key(t, key) not in done]
    print(f"{os.path.relpath(checkpoint)}: {len(done)} already done, {len(todo)} to run", flush=True)
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex, open(checkpoint, "a") as fout:
        futs = [ex.submit(worker, t) for t in todo]
        for k, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            fout.write(json.dumps(r) + "\n"); fout.flush()
            done[key(r)] = r
            if k % log_every == 0 or k == len(futs):
                el = time.time() - t0
                print(f"  {k}/{len(futs)} done, {el/60:.1f} min elapsed, ~{el/k*(len(futs)-k)/60:.1f} min left", flush=True)
    return [done[_task_key(t, key)] for t in tasks]


def _task_key(task, key):
    # tasks are tuples; toy tasks start with the toy index, fit tasks with a Config
    head = task[0]
    return head.name if isinstance(head, Config) else head


def coverage_summary(toys: list, a_true: float, sd_data: float) -> dict:
    a = np.array([t["A_median"] for t in toys])
    n = len(toys)
    c68, c95 = np.mean([t["in68"] for t in toys]), np.mean([t["in95"] for t in toys])
    return {"n_toys": n, "coverage_68": float(c68), "coverage_95": float(c95),
            "coverage_68_err": float(np.sqrt(0.68 * 0.32 / n)), "coverage_95_err": float(np.sqrt(0.95 * 0.05 / n)),
            "bias": float(a.mean() - a_true), "bias_err": float(a.std(ddof=1) / np.sqrt(n)),
            "sd_estimates": float(a.std(ddof=1)), "posterior_sd_data": sd_data,
            "chain_over_tau_min_worst": float(min(t["chain_over_tau_min"] for t in toys))}


def systematics(nominal: dict, variations: list, dchi2_max: float = 25.0) -> dict:
    """Common rule (same as freq `lhcb_freq.combine_systematics`): for sample-changing variations the shift counts
    only if |shift| > sigma_B = sqrt(|sd_var^2 - sd_nom^2|) (Barlow); variations on the same sample whose binned
    chi2 is worse than the nominal by more than `dchi2_max` are rejected by the data; then the largest |shift| per
    group, groups added in quadrature."""
    a0, s0 = nominal["A"]["median"], nominal["A"]["sd"]
    by = {v["config"]["name"]: v for v in variations}
    per = {}
    for k, v in by.items():
        d = v["A"]["median"] - a0
        sb = float(np.sqrt(abs(v["A"]["sd"] ** 2 - s0 ** 2))) if k in SAMPLE_CHANGING else None
        rejected = (k not in SAMPLE_CHANGING) and (v["gof"]["chi2"] - nominal["gof"]["chi2"] > dchi2_max)
        per[k] = {"shift": float(d), "sigma_barlow": sb, "rejected_by_data": bool(rejected),
                  "counts": bool((not rejected) and (sb is None or abs(d) > sb))}
    groups = {g: float(max([abs(per[k]["shift"]) for k in ks if k in per and per[k]["counts"]] or [0.0]))
              for g, ks in SYST_GROUPS.items()}
    return {"A_nominal": a0, "sigma_stat": s0, "variations": per, "groups": groups,
            "total": float(np.sqrt(sum(v ** 2 for v in groups.values()))),
            "rule": f"Barlow for sample-changing variants; exclude dchi2>{dchi2_max:g}; max per group; groups in quadrature"}


def bayes_factor_gaussian(res: dict, par: str, prior_density: float) -> float:
    """Savage-Dickey BF_01 for par = 0, posterior density at 0 from the Gaussian approximation."""
    return float(norm.pdf(0.0, res[par]["mean"], res[par]["sd"]) / prior_density)


def to_jsonable(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}
