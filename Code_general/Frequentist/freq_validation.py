"""
freq_validation.py -- additional validation of the frequentist analysis (uses lhcb_freq, does not modify it).

Modes (python freq_validation.py MODE [options]); results in results/validation/*.json(l):

  profiles   profile likelihood q(A) on the full sample and on each magnet polarity, fit per polarity and their
             combination by summing -2lnL; 2D (A, A_b) confidence regions and A_b with Minos
  toys       parametric toys from the nominal fit with A_true in a list: Ahat, Hesse error, t(A_true), q(0),
             binned chi2 (10 MeV bins); checkpointed in results/validation/toys.jsonl (resumable)
  belt       toys at a grid of A_true (free fit only) for the Neyman construction of the confidence belt
  multistart random starting points of the nominal fit
  momentum   A_raw in bins of the B momentum
  barlow     bootstrap of the shift between the full and a tighter-PID sample vs the sqrt(|s_sub^2 - s_full^2|) rule
  scans      stability scans of the nominal fit: window edges, signal region, grid of the two PID cuts (results/validation/scans.json)
  bootcov    the 1000 bootstrap replicas of the coverage study of the notebook (same seed, same replicas), in parallel:
             t = q(A_true), estimate and error of every replica (results/validation/bootcov.jsonl)
  jointfit   binned data, fitted curves (2 and 1 exponential) and pulls of the simultaneous fit (results/validation/jointfit.json)

Every long mode is parallel (multiprocessing) and appends one line per result, so it can be interrupted and resumed.
"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import truncnorm

warnings.filterwarnings("ignore")
import lhcb_freq as lf

DATA = Path(os.environ.get("LHCB_DATA_DIR", "../../opendata-project/Data"))
CFG = lf.Config(data_dir=DATA, sim_file=DATA / "PhaseSpaceSimulation.root")
OUT = Path(os.environ.get("VALIDATION_DIR", "results/validation")); OUT.mkdir(parents=True, exist_ok=True)
NPROC = int(os.environ.get("NPROC", 9))

_C = {}


def sample():
    """Nominal sample (preselection + window) with B_Mass, B_Charge, magnet and B momentum."""
    if "s" not in _C:
        d = lf.load_data(CFG)[CFG.columns]
        d = d[lf.kaon_preselection(d, CFG)].reset_index(drop=True)
        lf.add_B_variables(d, CFG)
        px = sum(d[f"H{i}_PX"] for i in (1, 2, 3)); py = sum(d[f"H{i}_PY"] for i in (1, 2, 3)); pz = sum(d[f"H{i}_PZ"] for i in (1, 2, 3))
        d["B_P"] = np.sqrt(px ** 2 + py ** 2 + pz ** 2); d["B_PT"] = np.sqrt(px ** 2 + py ** 2)
        _C["d"] = d
        _C["s"] = lf.select_window(d, CFG.fit_window).reset_index(drop=True)
    return _C["d"], _C["s"]


def split(df):
    return lf.split_by_charge(df, col="B_Mass")


def best_fit(df=None):
    s = sample()[1] if df is None else df
    return lf.fit_joint(split(s), CFG, minos=False)


def values_of(m):
    return {k: float(v) for k, v in zip(m.parameters, m.values)}


def append(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def done_keys(path, keys):
    if not Path(path).exists():
        return set()
    return {tuple(json.loads(l)[k] for k in keys) for l in open(path) if l.strip()}


# ---------------------------------------------------------------- toy generator
def gen(values, a_true, rng):
    """Pseudo-dataset from the fitted model with A = a_true: Poisson yields, truncated Gaussian, truncated exponentials."""
    lo, hi = CFG.fit_window
    out = {}
    for side, sd in lf.SIDES.items():
        n_sig = values["N_sig"] * (1 + sd.asym_sign * a_true) / 2
        ns, nb = rng.poisson(n_sig), rng.poisson(values[f"N_bkg_{side}"])
        mu, sg = values["mu"], values["sigma"]
        ms = truncnorm.rvs((lo - mu) / sg, (hi - mu) / sg, loc=mu, scale=sg, size=ns, random_state=rng)
        comp = rng.random(nb) < values[f"f_bkgs_{side}"]
        lam = np.where(comp, values[f"lam1_{side}"], values[f"lam2_{side}"])
        u = rng.random(nb)
        mb = lo - np.log1p(-u * (-np.expm1(-lam * (hi - lo)))) / lam
        out[side] = np.concatenate([ms, mb])
    return out


def _toy(args):
    i, a_true, seed, values, null_fits = args
    rng = np.random.default_rng([seed, i, int(round(1e4 * (a_true + 1)))])
    try:
        smp = gen(values, a_true, rng)
        free = lf.fit_joint(smp, CFG, start=dict(values), hesse=True)
        row = dict(i=i, a_true=a_true, valid=bool(free.valid), accurate=bool(free.accurate),
                   A=float(free.values["A_CP"]), err=float(free.errors["A_CP"]), fval=float(free.fval),
                   n=int(sum(len(v) for v in smp.values())))
        if null_fits:
            nt = lf.fit_joint(smp, CFG, start={**values, "A_CP": a_true}, fixed={"A_CP": a_true}, hesse=False)
            n0 = lf.fit_joint(smp, CFG, start={**values, "A_CP": 0.0}, fixed={"A_CP": 0.0}, hesse=False)
            g = lf.gof_joint(free, smp, CFG, bins=100)
            row.update(t_true=max(nt.fval - free.fval, 0.0), q0=max(n0.fval - free.fval, 0.0), chi2=g.chi2, ndof=g.ndof)
    except Exception as e:  # noqa
        row = dict(i=i, a_true=a_true, valid=False, error=f"{type(e).__name__}: {e}")
    return row


def run_toys(path, a_list, n_toys, seed, null_fits, values):
    have = done_keys(path, ["a_true", "i"])
    jobs = [(i, a, seed, values, null_fits) for a in a_list for i in range(n_toys) if (a, i) not in have]
    print(f"{len(jobs)} toys to run ({len(have)} done)", flush=True)
    t0 = time.time(); k = 0
    with Pool(NPROC) as pool:
        buf = []
        for r in pool.imap_unordered(_toy, jobs, chunksize=4):
            buf.append(r); k += 1
            if len(buf) >= 20:
                append(path, buf); buf = []
                print(f"  {k}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
        append(path, buf)


# ---------------------------------------------------------------- profiles, polarities, (A, A_b) region
def _profile_point(args):
    key, a, which = args
    d, s = sample()
    df = s if which == "all" else s[s.magnet == which]
    smp = split(df)
    if ("start", which) not in _C:
        _C[("start", which)] = values_of(lf.fit_joint(smp, CFG))
    start = _C[("start", which)]
    m = lf.fit_joint(smp, CFG, start={**start, "A_CP": a}, fixed={"A_CP": a}, hesse=False)
    return which, float(a), float(m.fval), bool(m.valid)


def mode_profiles():
    d, s = sample()
    res = {"nominal_n": int(len(s))}
    grid = np.round(np.linspace(-0.12, 0.04, 33), 4)
    fits = {}
    for which in ("all", "down", "up"):
        df = s if which == "all" else s[s.magnet == which]
        m = lf.fit_joint(split(df), CFG, minos=True)
        fits[which] = m
        mn = m.merrors["A_CP"]
        res[which] = dict(n=int(len(df)), A=float(m.values["A_CP"]), err=float(m.errors["A_CP"]),
                          minos=[float(m.values["A_CP"] + mn.lower), float(m.values["A_CP"] + mn.upper)],
                          fval=float(m.fval), valid=bool(m.valid))
        print(which, res[which], flush=True)
    jobs = [(None, a, w) for w in ("all", "down", "up") for a in grid]
    with Pool(NPROC) as pool:
        out = pool.map(_profile_point, jobs)
    prof = {w: {"A": [], "q": []} for w in ("all", "down", "up")}
    for w, a, fv, ok in out:
        prof[w]["A"].append(a); prof[w]["q"].append(fv - fits[w].fval)
    res["profile"] = prof
    # combination of the two polarities by summing the profile curves
    qd, qu = np.array(prof["down"]["q"]), np.array(prof["up"]["q"]); a_ = np.array(prof["down"]["A"])
    qs = qd + qu
    fine = np.linspace(a_.min(), a_.max(), 4001)
    qf = np.interp(fine, a_, qs)
    # parabolic refinement around the minimum
    j = int(np.argmin(qf)); a_hat = float(fine[j]); qmin = float(qf[j])
    inside = fine[qf <= qmin + 1.0]
    res["combined_sum"] = dict(A=a_hat, interval68=[float(inside.min()), float(inside.max())],
                               sigma=float(0.5 * (inside.max() - inside.min())), q_at_min=qmin,
                               note="sum of the two profile curves; each polarity has its own nuisance parameters")
    # (A, A_b) region: parametrization with N_b and A_b
    from iminuit import Minuit
    from iminuit.cost import ExtendedUnbinnedNLL
    import operator
    from functools import reduce
    smp = split(s)
    bv = values_of(fits["all"])
    costs = []
    for side, sd in lf.SIDES.items():
        base = lf.make_model(CFG, side, 2)
        sgn = sd.asym_sign

        def wrap(m, N_sig, A_CP, mu, sigma, N_b, A_b, l1, l2, fb, _base=base, _sgn=sgn):
            return _base(m, N_sig, A_CP, mu, sigma, N_b * (1 + _sgn * A_b) / 2, l1, l2, fb)
        lf._with_signature(wrap, ["N_sig", "A_CP", "mu", "sigma", "N_b", "A_b", f"lam1_{side}", f"lam2_{side}", f"f_bkgs_{side}"])
        costs.append(ExtendedUnbinnedNLL(smp[side], wrap))
    cost = reduce(operator.add, costs)
    npos, nneg = bv["N_bkg_pos"], bv["N_bkg_neg"]
    start = dict(N_sig=bv["N_sig"], A_CP=bv["A_CP"], mu=bv["mu"], sigma=bv["sigma"], N_b=npos + nneg,
                 A_b=(nneg - npos) / (nneg + npos))
    for side in ("pos", "neg"):
        for k in ("lam1", "lam2", "f_bkgs"):
            start[f"{k}_{side}"] = bv[f"{k}_{side}"]
    mm = lf.make_minuit(cost, start, CFG)
    mm.limits["A_b"] = (-1, 1); mm.limits["N_b"] = (0, None)
    mm.migrad(); mm.hesse(); mm.minos("A_CP", "A_b")
    ab = dict(A=float(mm.values["A_CP"]), A_err=float(mm.errors["A_CP"]), A_b=float(mm.values["A_b"]),
              A_b_err=float(mm.errors["A_b"]), A_b_minos=[float(mm.values["A_b"] + mm.merrors["A_b"].lower),
                                                          float(mm.values["A_b"] + mm.merrors["A_b"].upper)],
              corr=float(mm.covariance.correlation()[mm.parameters.index("A_CP"), mm.parameters.index("A_b")]),
              valid=bool(mm.valid))
    for cl, tag in ((0.6827, "c68"), (0.9545, "c95")):
        pts = mm.mncontour("A_CP", "A_b", cl=cl, size=60)
        ab[tag] = np.asarray(pts).tolist()
    ab["profile_A_b"] = None
    res["A_Ab"] = ab
    json.dump(res, open(OUT / "profiles.json", "w"), indent=1)
    print("saved profiles.json", flush=True)


# ---------------------------------------------------------------- other modes
def mode_toys(args):
    m = best_fit(); vals = values_of(m)
    json.dump(vals, open(OUT / "nominal_values.json", "w"), indent=1)
    a_nom = vals["A_CP"]
    run_toys(OUT / "toys.jsonl", [round(a_nom, 4)], args.n, args.seed, True, vals)            # nominal: --n toys
    run_toys(OUT / "toys.jsonl", [0.0, -0.02, -0.1, 0.05], args.n_other, args.seed, True, vals)  # injection / null


def mode_belt(args):
    m = best_fit(); vals = values_of(m)
    grid = [round(x, 3) for x in np.linspace(-0.10, 0.0, 11)]
    run_toys(OUT / "belt.jsonl", grid, args.n, args.seed + 1000, False, vals)


def _start_job(args):
    i, vals, seed = args
    rng = np.random.default_rng([seed, i])
    d, s = sample()
    start = dict(vals)
    for k in start:
        if k.startswith(("N_", "lam", "sigma")):
            start[k] = vals[k] * float(np.exp(rng.normal(0, 0.5)))
        elif k == "f_bkgs_pos" or k == "f_bkgs_neg":
            start[k] = float(rng.uniform(0.05, 0.95))
        elif k == "mu":
            start[k] = vals[k] + float(rng.normal(0, 8))
        elif k == "A_CP":
            start[k] = float(rng.uniform(-0.2, 0.2))
    t = time.time()
    try:
        m = lf.fit_joint(split(s), CFG, start=start, hesse=False)
        return dict(i=i, valid=bool(m.valid), A=float(m.values["A_CP"]), fval=float(m.fval),
                    lam1_gt_lam2=bool(m.values["lam1_pos"] > m.values["lam2_pos"] and m.values["lam1_neg"] > m.values["lam2_neg"]),
                    nfcn=int(m.nfcn), seconds=time.time() - t)
    except Exception as e:  # noqa
        return dict(i=i, valid=False, error=str(e))


def mode_multistart(args):
    m = best_fit(); vals = values_of(m)
    with Pool(NPROC) as pool:
        rows = pool.map(_start_job, [(i, vals, args.seed) for i in range(args.n)])
    json.dump(dict(nominal=dict(A=vals["A_CP"], fval=float(m.fval)), rows=rows), open(OUT / "multistart.json", "w"), indent=1)
    print("saved multistart.json")


def _mom_job(args):
    lab, lo, hi = args
    d, s = sample()
    sub = s[(s.B_PT >= lo) & (s.B_PT < hi)]
    m = lf.fit_joint(split(sub), CFG)
    return dict(label=lab, lo=float(lo), hi=float(hi), n=int(len(sub)), A=float(m.values["A_CP"]),
                err=float(m.errors["A_CP"]), valid=bool(m.valid), accurate=bool(m.accurate))


def mode_momentum(args):
    d, s = sample()
    out = {}
    for var in ("B_PT", "B_P"):
        q = np.quantile(s[var], [0, 1 / 3, 2 / 3, 1.0]); q[-1] += 1e-6
        jobs = []
        for k in range(3):
            lab = f"{var}_{k}"
            jobs.append((lab, q[k], q[k + 1]))
        _C["var"] = var
        res = []
        for lab, lo, hi in jobs:
            sub = s[(s[var] >= lo) & (s[var] < hi)]
            m = lf.fit_joint(split(sub), CFG)
            res.append(dict(label=lab, lo=float(lo), hi=float(hi), n=int(len(sub)), A=float(m.values["A_CP"]),
                            err=float(m.errors["A_CP"]), valid=bool(m.valid), accurate=bool(m.accurate)))
            print(res[-1], flush=True)
        out[var] = res
    json.dump(out, open(OUT / "momentum.json", "w"), indent=1)


def _barlow_job(args):
    i, seed, cuts = args
    d, s = sample()
    rng = np.random.default_rng([seed, i])
    boot = s.iloc[rng.integers(0, len(s), len(s))]
    full = lf.fit_joint(split(boot), CFG)
    rows = dict(i=i, A_full=float(full.values["A_CP"]), s_full=float(full.errors["A_CP"]), v_full=bool(full.valid))
    for name, (kmin, pimax) in cuts.items():
        keep = np.ones(len(boot), bool)
        for h in (1, 2, 3):
            keep &= (boot[f"H{h}_ProbK"] > kmin).to_numpy() & (boot[f"H{h}_ProbPi"] < pimax).to_numpy()
        sub = boot[keep]
        m = lf.fit_joint(split(sub), CFG)
        rows[name] = dict(n=int(len(sub)), A=float(m.values["A_CP"]), s=float(m.errors["A_CP"]), valid=bool(m.valid))
    return rows


def mode_barlow(args):
    cuts = {"probk_0.6": (0.6, 0.5), "probpi_0.4": (0.5, 0.4)}
    path = OUT / "barlow.jsonl"
    have = done_keys(path, ["i"])
    jobs = [(i, args.seed, cuts) for i in range(args.n) if (i,) not in have]
    with Pool(NPROC) as pool:
        buf = []
        for k, r in enumerate(pool.imap_unordered(_barlow_job, jobs, chunksize=2), 1):
            buf.append(r)
            if len(buf) >= 10:
                append(path, buf); buf = []; print(f"  {k}/{len(jobs)}", flush=True)
        append(path, buf)


def _cov_job(args):
    i, seed, vals = args
    d, s = sample()
    rng = np.random.default_rng([seed, i, 77])
    boot = s.iloc[rng.integers(0, len(s), len(s))]
    try:
        m = lf.fit_joint(split(boot), CFG, start=dict(vals), hesse=False)
        return dict(i=i, valid=bool(m.valid), values=values_of(m))
    except Exception as e:  # noqa
        return dict(i=i, valid=False, error=str(e))


def mode_cov(args):
    m = best_fit(); vals = values_of(m)
    names = list(m.parameters)
    cov = np.array(m.covariance)
    json.dump(dict(names=names, values=vals, hesse_cov=cov.tolist(), hesse_err={n: float(m.errors[n]) for n in names}),
              open(OUT / "cov_hesse.json", "w"), indent=1)
    path = OUT / "cov.jsonl"
    have = done_keys(path, ["i"])
    jobs = [(i, args.seed, vals) for i in range(args.n) if (i,) not in have]
    with Pool(NPROC) as pool:
        buf = []
        for k, r in enumerate(pool.imap_unordered(_cov_job, jobs, chunksize=2), 1):
            buf.append(r)
            if len(buf) >= 20:
                append(path, buf); buf = []; print(f"  {k}/{len(jobs)}", flush=True)
        append(path, buf)


def _pf_job(args):
    name, v, vals = args
    d, s = sample()
    m = lf.fit_joint(split(s), CFG, start={**vals, name: v}, fixed={name: v}, hesse=False)
    return name, float(v), float(m.fval), bool(m.valid)


def mode_profile_par(args):
    m = best_fit(); vals = values_of(m)
    out = {"best_fval": float(m.fval), "best": {k: vals[k] for k in ("f_bkgs_pos", "lam2_pos", "f_bkgs_neg", "lam2_neg")},
           "errors": {k: float(m.errors[k]) for k in ("f_bkgs_pos", "lam2_pos", "f_bkgs_neg", "lam2_neg")}}
    jobs = []
    for name in ("f_bkgs_pos", "lam2_pos"):
        b, e = vals[name], float(m.errors[name])
        for v in np.linspace(b - 4 * e, b + 4 * e, 17):
            if v > 0 and (name != "f_bkgs_pos" or v < 1):
                jobs.append((name, float(v), vals))
    with Pool(NPROC) as pool:
        rows = pool.map(_pf_job, jobs)
    out["profile"] = {n: {"x": [r[1] for r in rows if r[0] == n], "q": [r[2] - m.fval for r in rows if r[0] == n]}
                      for n in ("f_bkgs_pos", "lam2_pos")}
    json.dump(out, open(OUT / "profile_par.json", "w"), indent=1)


def _gof_job(args):
    i, seed, vals, a_true = args
    rng = np.random.default_rng([seed, i, int(round(1e4 * (a_true + 1)))])
    try:
        smp = gen(vals, a_true, rng)
        free = lf.fit_joint(smp, CFG, start=dict(vals), hesse=False)
        row = dict(i=i, valid=bool(free.valid))
        for b in (50, 100, 200):
            g = lf.gof_joint(free, smp, CFG, bins=b)
            row[f"chi2_{b}"], row[f"ndof_{b}"] = g.chi2, g.ndof
        return row
    except Exception as e:  # noqa
        return dict(i=i, valid=False, error=str(e))


def mode_gof(args):
    """Binned chi2 of toys generated from the fitted model (A = nominal) with 50, 100 and 200 bins per charge,
    and of the data with the same definitions (same seeds as `toys`, free fit only)."""
    d, s = sample()
    m = best_fit(); vals = values_of(m)
    smp = split(s)
    data_gof = {}
    for b in (50, 100, 200):
        g = lf.gof_joint(m, smp, CFG, bins=b)
        data_gof[str(b)] = dict(chi2=g.chi2, ndof=g.ndof, p_asymptotic=g.p_value)
    json.dump(data_gof, open(OUT / "gof_data.json", "w"), indent=1)
    path = OUT / "gof.jsonl"
    a_nom = round(vals["A_CP"], 4)
    have = done_keys(path, ["i"])
    jobs = [(i, args.seed, vals, a_nom) for i in range(args.n) if (i,) not in have]
    with Pool(NPROC) as pool:
        buf = []
        for k, r in enumerate(pool.imap_unordered(_gof_job, jobs, chunksize=4), 1):
            buf.append(r)
            if len(buf) >= 20:
                append(path, buf); buf = []; print(f"  {k}/{len(jobs)}", flush=True)
        append(path, buf)


def _bootcov_job(args):
    k, idx, start, truth = args
    d, s = sample()
    smp = split(s.iloc[idx])
    free = lf.fit_joint(smp, CFG, 2, start=start, fixed={}, hesse=False)
    if not free.valid:
        return dict(k=k, valid=False)
    null = lf.fit_joint(smp, CFG, 2, start={**start, "A_CP": truth}, fixed={"A_CP": truth}, hesse=False)
    if not null.valid:
        return dict(k=k, valid=False)
    return dict(k=k, valid=True, t=float(max(null.fval - free.fval, 0.0)), A=float(free.values["A_CP"]),
                err=float(free.errors["A_CP"]))


def mode_bootcov(args):
    """Coverage of the Wilks interval on A: the same bootstrap replicas as run_coverage_study(seed=1) of the notebook."""
    m = best_fit(); start = values_of(m); truth = start["A_CP"]
    n = len(sample()[1])
    rng = np.random.default_rng(1)
    idx = [rng.integers(0, n, n) for _ in range(args.n)]
    path = OUT / "bootcov.jsonl"
    have = done_keys(path, ["k"])
    jobs = [(k, idx[k], start, truth) for k in range(args.n) if (k,) not in have]
    with Pool(NPROC) as pool:
        buf = []
        for j, r in enumerate(pool.imap_unordered(_bootcov_job, jobs, chunksize=2), 1):
            buf.append(r)
            if len(buf) >= 25:
                append(path, buf); buf = []; print(f"  {j}/{len(jobs)}", flush=True)
        append(path, buf)


def mode_scans():
    """Fit-window, signal-region and PID-cut grids of the stability study (same grids and calls as notebook 06)."""
    import itertools
    cut = OUT / "scans.json"
    done = json.load(open(cut)) if cut.exists() else {}
    d = lf.load_data(CFG)[CFG.columns]
    loose = d[lf.kaon_preselection(d, replace(CFG, prob_k_min=0.4, prob_pi_max=0.6))].reset_index(drop=True)
    lf.add_B_variables(loose, CFG)
    nominal = d[lf.kaon_preselection(d, CFG)].reset_index(drop=True)
    lf.add_B_variables(nominal, CFG)
    del d
    keep = ["left", "right", "sig_lo", "sig_hi", "prob_k_min", "prob_pi_max", "valid", "accurate", "n_events", "A_CP", "A_CP_err"]
    grids = {
        "window": lambda: lf.scan_fit_window(nominal, CFG, [5000, 5050, 5100, 5150, 5200],
                                             [5350, 5400, 5450, 5500, 5550, 5600, 5650, 5700, 5750, 5800, 5850, 5900, 5950, 6000, 6050, 6100]),
        "signal_region": lambda: lf.scan_signal_region(nominal, CFG, [5150, 5170, 5185, 5200, 5215, 5230, 5250],
                                                       [5320, 5335, 5350, 5365, 5380, 5400, 5420]),
        "pid": lambda: lf.scan_preselection_cuts(loose, CFG, np.arange(0.45, 0.81, 0.025), np.arange(0.30, 0.56, 0.025)),
    }
    for name, run in grids.items():
        if name in done:
            continue
        t0 = time.time(); df = run()
        done[name] = json.loads(df[[c for c in keep if c in df.columns]].to_json(orient="records"))
        json.dump(done, open(cut, "w"), indent=1); print(f"scan {name}: {len(df)} fits in {time.time() - t0:.0f} s", flush=True)
    nom = lf.fit_joint(split(lf.select_window(nominal, CFG.fit_window)), CFG)
    done["nominal"] = dict(A_CP=float(nom.values["A_CP"]), A_CP_err=float(nom.errors["A_CP"]))
    json.dump(done, open(cut, "w"), indent=1)


def mode_jointfit():
    """Data histogram (10 MeV bins), fitted curves and pulls of the simultaneous fit with 2 and 1 exponential."""
    sm = split(sample()[1])
    bins = 50; edges = np.linspace(*CFG.fit_window, bins + 1); centres = 0.5 * (edges[:-1] + edges[1:]); w = edges[1] - edges[0]
    x = np.linspace(*CFG.fit_window, 600)
    out = dict(centres=centres.tolist(), width=float(w), x=x.tolist())
    for n in (2, 1):
        m = lf.fit_joint(sm, CFG, n_exp=n, minos=False)
        for side in ("pos", "neg"):
            curve = lf._model_curve(lf.make_model(CFG, side, n), m.values)
            counts, _ = np.histogram(sm[side], bins=edges)
            # expected counts per bin by integrating the curve over the bin (5 points per bin)
            xs = np.linspace(edges[0], edges[-1], bins * 20 + 1); ys = curve(0.5 * (xs[:-1] + xs[1:])) * (xs[1] - xs[0])
            exp = ys.reshape(bins, 20).sum(axis=1)
            out.setdefault(side, dict(counts=counts.tolist()))[f"curve_{n}"] = (w * curve(x)).tolist()
            out[side][f"pull_{n}"] = ((counts - exp) / np.sqrt(np.maximum(exp, 1))).tolist()
        out[f"A_{n}"] = float(m.values["A_CP"]); out[f"A_err_{n}"] = float(m.errors["A_CP"])
        if n == 2:
            v = m.values
            for side, sgn in (("pos", -1), ("neg", +1)):
                nsig = v["N_sig"] * (1 + sgn * v["A_CP"]) / 2
                sig = nsig * lf.gauss_pdf(x, v["mu"], v["sigma"], CFG.fit_window) * w
                bkg = v[f"N_bkg_{side}"] * lf.background_pdf(x, [v[f"lam1_{side}"], v[f"lam2_{side}"]], v[f"f_bkgs_{side}"],
                                                            CFG.window_intervals) * w
                out[side]["signal"] = sig.tolist(); out[side]["background"] = bkg.tolist(); out[side]["N_sig_side"] = float(nsig)
            # profile likelihood ratio on a fine grid (the other eleven parameters re-minimized at every A)
            a_grid, fval, _ = m.mnprofile("A_CP", size=86, bound=(-0.075, 0.010))
            fval = np.asarray(fval)
            out["profile"] = dict(A=np.asarray(a_grid).tolist(), q=((fval - fval.min()) / m.errordef).tolist(),
                                  A_hat=float(v["A_CP"]))
    json.dump(out, open(OUT / "jointfit.json", "w"))
    print("jointfit saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["profiles", "toys", "belt", "multistart", "momentum", "barlow", "cov", "profile_par", "gof", "scans", "jointfit", "bootcov"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--n-other", dest="n_other", type=int, default=300)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    if a.mode == "profiles":
        mode_profiles()
    elif a.mode == "toys":
        mode_toys(a)
    elif a.mode == "belt":
        mode_belt(a)
    elif a.mode == "multistart":
        mode_multistart(a)
    elif a.mode == "momentum":
        mode_momentum(a)
    elif a.mode == "barlow":
        mode_barlow(a)
    elif a.mode == "cov":
        mode_cov(a)
    elif a.mode == "profile_par":
        mode_profile_par(a)
    elif a.mode == "gof":
        mode_gof(a)
    elif a.mode == "bootcov":
        mode_bootcov(a)
    elif a.mode == "scans":
        mode_scans()
    elif a.mode == "jointfit":
        mode_jointfit()
