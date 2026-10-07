"""
Bayesian vs frequentist inference on the SAME likelihood at low statistics (Bayes_vs_Freq_LowStat.ipynb).

Model: the six-parameter model (window [5150, 5450] MeV, Gaussian signal, one exponential background with a shape
and a yield shared by the two charges), identifiable down to a few tens of events; it is the `bayesian_nominal`
Config(lo=5150, hi=5450, nexp=1, shape="shared", asym="sym").
For each pseudo-dataset:
    Bayesian     emcee posterior (flat priors), median, equal-tailed 68%/95% credible intervals
    frequentist  MLE with iminuit on -ln L of the same likelihood, Hesse (symmetric) and Minos (profile-likelihood,
                 Wilks) 68%/95% confidence intervals
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from iminuit import Minuit

import bayesian_nominal as bn

CFG = bn.Config(name="lowstat", lo=5150.0, hi=5450.0, nexp=1, shape="shared", asym="sym")


def frequentist_fit(m, q, cfg, start: dict) -> dict:
    post = bn.Posterior(m, q, cfg)
    names = post.names
    lo_, hi_ = bn._prior_box(cfg, post.n, names)

    def nll(*p):
        v = post(np.array(p))
        return 1e30 if not np.isfinite(v) else -v
    mi = Minuit(nll, *[start[k] for k in names], name=names)
    mi.errordef = Minuit.LIKELIHOOD
    for k, a, b in zip(names, lo_, hi_):
        mi.limits[k] = (a, b)
    mi.migrad()
    out = {"valid": bool(mi.valid), "A": float(mi.values["A"]), "hesse": float("nan"),
           "minos68": [float("nan")] * 2, "minos95": [float("nan")] * 2}
    if not mi.valid:
        return out
    mi.hesse()
    out["hesse"] = float(mi.errors["A"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cl, key in ((0.6827, "minos68"), (0.9545, "minos95")):
            try:
                mi.minos("A", cl=cl)
                me = mi.merrors["A"]
                out[key] = [float(mi.values["A"] + me.lower), float(mi.values["A"] + me.upper)]
                out[key + "_valid"] = bool(me.is_valid)
            except Exception:
                out[key + "_valid"] = False
    return out


def task(args):
    """args = (key, n_total, theta_true (dict, full-statistics fit), seed, nsteps)."""
    key, n_total, theta, seed, nsteps = args
    t0 = time.time()
    rng = np.random.default_rng(seed)
    scale = n_total / (theta["N_s"] + theta["N_b"])
    th = dict(theta, N_s=theta["N_s"] * scale, N_b=theta["N_b"] * scale)
    m, q = bn.generate_toy(th, CFG, rng)
    a_true = th["A"]
    r = bn.fit(m, q, CFG, nsteps=nsteps, seed=seed, start=th)
    f = frequentist_fit(m, q, CFG, {k: r["map"][k] for k in r["names"]})
    bay = r["A"]
    inside = lambda iv: bool(iv[0] < a_true < iv[1]) if np.all(np.isfinite(iv)) else None
    hes68 = [f["A"] - f["hesse"], f["A"] + f["hesse"]]; hes95 = [f["A"] - 2 * f["hesse"], f["A"] + 2 * f["hesse"]]
    return {"key": key, "n_total": n_total, "seed": seed, "n": int(len(m)), "A_true": a_true,
            "bayes": {"median": bay["median"], "mean": bay["mean"], "sd": bay["sd"], "ci68": bay["ci68"], "ci95": bay["ci95"],
                      "in68": inside(bay["ci68"]), "in95": inside(bay["ci95"]),
                      "skew_68": float((bay["ci68"][1] - bay["median"]) - (bay["median"] - bay["ci68"][0])),
                      "chain_over_tau_min": r["chain_over_tau_min"], "tau_A": r["tau"]["A"]},
            "freq": dict(f, in68_minos=inside(f["minos68"]), in95_minos=inside(f["minos95"]),
                         in68_hesse=inside(hes68) if f["valid"] else None, in95_hesse=inside(hes95) if f["valid"] else None),
            "seconds": round(time.time() - t0, 1)}
