"""Figures of the validation studies (frequentist toys, profiles, belts; Bayesian posterior checks).

Inputs (written by the validation scripts, not by the notebooks):
  FREQ_DIR   = Code_general/Frequentist/results/validation  (freq_validation.py: profiles.json, toys.jsonl, belt.jsonl, ...)
  BAYES_DIR  = Code_general/Bayesian/results/nominal         (bayes_validation.py: validation.json; nominal_posterior_thinned.npy)
Both default to the sibling git worktrees ../wt_freq and ../wt_bayes; override with REGEN_FREQ_DIR / REGEN_BAYES_DIR.
usage (cern env):  python regen_figures_validation.py [OUTDIR]
Every figure is produced only if its inputs exist.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else HERE
os.makedirs(OUT, exist_ok=True)
WORK = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))        # .../Work
FREQ = os.environ.get("REGEN_FREQ_DIR", os.path.join(WORK, "wt_freq", "LHCb", "Code_general", "Frequentist", "results", "validation"))
BAYES = os.environ.get("REGEN_BAYES_DIR", os.path.join(WORK, "wt_bayes", "LHCb", "Code_general", "Bayesian", "results", "nominal"))
sys.path.insert(0, HERE)
import plot_style as ps
ps.use(); C = ps.COLORS


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("saved", name)


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def parabola(a, q, qmax=9.0):
    """Quadratic fit q(A) = q0 + (A - a0)^2 / s^2 on the points with q < qmax; returns a0, s and the function."""
    a, q = np.asarray(a), np.asarray(q)
    k = q < qmax
    c = np.polyfit(a[k], q[k], 2)
    a0 = -c[1] / (2 * c[0]); s = 1 / np.sqrt(c[0]); q0 = np.polyval(c, a0)
    return a0, s, (lambda x: np.polyval(c, x) - q0)


prof = json.load(open(os.path.join(FREQ, "profiles.json"))) if os.path.exists(os.path.join(FREQ, "profiles.json")) else None
bval = json.load(open(os.path.join(BAYES, "validation.json"))) if os.path.exists(os.path.join(BAYES, "validation.json")) else None
summ = json.load(open(os.path.join(BAYES, "nominal_summary.json")))
names = summ["nominal"]["names"]
th = np.load(os.path.join(BAYES, "nominal_posterior_thinned.npy"))
A_post, Ab_post = th[:, names.index("A")], th[:, names.index("A_b")]

# ------------------------------------------------ profile likelihood vs posterior, polarities
if prof:
    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 4.4))
    a = axs[0]; ps.letter(a, "a")
    pa = prof["profile"]["all"]
    a0, s0, f = parabola(pa["A"], pa["q"])
    xs = np.linspace(-0.085, 0.0, 400)
    like = np.exp(-0.5 * f(xs)); like /= np.trapezoid(like, xs)
    a.hist(A_post, bins=60, range=(xs[0], xs[-1]), density=True, color=C["bayes"], alpha=0.35,
           label="Bayesian: marginal posterior")
    a.plot(xs, like, color=C["freq"], lw=2, label=r"Frequentist: $\exp[-q(A)/2]$, normalized")
    a.axvline(0, color="k", ls=":", lw=0.8)
    a.set_xlabel(r"$A_{\mathrm{raw}}$"); a.set_ylabel("Density"); a.legend(loc="upper left", fontsize=10)
    a = axs[1]; ps.letter(a, "b")
    col = {"down": C["down"], "up": C["up"]}
    qs = 0
    for w in ("down", "up"):
        p_ = prof["profile"][w]
        a0_, s_, f_ = parabola(p_["A"], p_["q"])
        a.plot(p_["A"], p_["q"], "o", color=col[w], ms=4)
        a.plot(xs, f_(xs), color=col[w], lw=1.5, label=f"Magnet {w}: ${a0_:+.4f}\\pm{s_:.4f}$")
        qs = qs + f_(xs)
    j = np.argmin(qs); a_c = xs[j]
    inside = xs[qs <= qs[j] + 1]
    a.plot(xs, qs - qs[j], color="k", lw=2, label=f"Sum: ${a_c:+.4f}\\pm{0.5*(inside.max()-inside.min()):.4f}$")
    a.plot(xs, f(xs), color=C["freq"], lw=1.5, ls="--", label=f"Joint fit: ${a0:+.4f}\\pm{s0:.4f}$")
    a.axhline(1, color="gray", ls=":", lw=0.8)
    a.set_ylim(0, 6); a.set_xlim(-0.1, 0.0)
    a.set_xlabel(r"$A_{\mathrm{raw}}$"); a.set_ylabel(r"$q(A)=-2\Delta\ln L$"); a.legend(loc="upper left", fontsize=9)
    plt.tight_layout(); save(fig, "validation_profile_polarity.pdf")

    # (A, A_b) region: frequentist contours vs Bayesian HPD regions
    ab = prof["A_Ab"]
    fig, ax = plt.subplots(figsize=(ps.SINGLE, 5.0))
    from scipy.ndimage import gaussian_filter
    H, xe, ye = np.histogram2d(A_post, Ab_post, bins=50, density=True)
    H = gaussian_filter(H, 1.5)
    Hs = np.sort(H.ravel())[::-1]; cum = np.cumsum(Hs) * (xe[1] - xe[0]) * (ye[1] - ye[0])
    lev = [Hs[np.searchsorted(cum, p)] for p in (0.6827, 0.9545)]
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    ax.contourf(xc, yc, H.T, levels=[lev[1], lev[0], H.max() * 2], colors=[C["bayes"], C["bayes"]], alpha=0.0)
    cs = ax.contour(xc, yc, H.T, levels=sorted(lev), colors=[C["bayes"]], linewidths=[1.2, 2.0], linestyles=["--", "-"])
    for tag, ls_, lw in (("c95", "--", 1.2), ("c68", "-", 2.0)):
        pts = np.array(ab[tag]); pts = np.vstack([pts, pts[:1]])
        ax.plot(pts[:, 0], pts[:, 1], color=C["freq"], ls=ls_, lw=lw)
    ax.plot([ab["A"]], [ab["A_b"]], "o", color=C["freq"], ms=6)
    ax.plot([np.median(A_post)], [np.median(Ab_post)], "s", color=C["bayes"], ms=6)
    ax.axhline(0, color="k", ls=":", lw=0.8); ax.axvline(0, color="k", ls=":", lw=0.8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=C["freq"], lw=2, label="Frequentist (profile, $\\chi^2_2$)"),
                       Line2D([], [], color=C["bayes"], lw=2, label="Bayesian (HPD)"),
                       Line2D([], [], color="gray", lw=2, label="68% (solid), 95% (dashed)")], loc="lower right", fontsize=9)
    ax.set_xlabel(r"$A_{\mathrm{raw}}$"); ax.set_ylabel(r"$A_b$")
    plt.tight_layout(); save(fig, "validation_region_Ab.pdf")

# ------------------------------------------------ Bayes factor vs prior width, unfitted PPC statistics
if bval:
    fig, ax = plt.subplots(figsize=(ps.SINGLE, 4.4))
    W = np.array(bval["bf_vs_prior_width"]["W"])
    ax.plot(W, bval["bf_vs_prior_width"]["A"]["bf01"], color=C["bayes"], lw=2, label=r"$A$")
    ax.plot(W, bval["bf_vs_prior_width"]["A_b"]["bf01"], color=C["count"], lw=2, ls="--", label=r"$A_b$")
    ax.axhline(1, color="k", ls=":", lw=0.8); ax.axhline(1e-2, color="gray", ls=":", lw=0.8)
    ax.text(W[0] * 1.02, 1.15, "equal evidence", fontsize=9); ax.text(W[0] * 1.02, 1.15e-2, "decisive for $A\\neq0$", fontsize=9)
    ax.axvline(1.0, color=C["background"], lw=0.8, ls="-.")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Half-width $W$ of the uniform prior $(-W,W)$"); ax.set_ylabel(r"$BF_{01}$ (Savage-Dickey)")
    ax.legend(loc="center right")
    plt.tight_layout(); save(fig, "validation_bf_prior_width.pdf")

    pp = bval["ppc_unfitted"]; sl = bval["slices"]
    fig, axs = plt.subplots(1, 3, figsize=(ps.TRIPLE, 4.2), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    a = axs[0]; ps.letter(a, "a")
    x = np.array(sl["centers"])
    a.errorbar(x, sl["A_obs"], yerr=sl["A_err"], fmt="o", color=C["data"], label="Data")
    a.step(np.append(x - 25, x[-1] + 25), np.append(sl["A_model"], sl["A_model"][-1]), where="post", color=C["model"], lw=1.8,
           label="Model (posterior medians)")
    a.axhline(0, color="k", ls=":", lw=0.8)
    a.set_xlabel(r"$m(KKK)$ [MeV]"); a.set_ylabel("Counting asymmetry in 50 MeV slices"); a.legend(loc="upper right", fontsize=10)
    for k, (key, lab) in enumerate((("T_slices", r"$T=\sum_s(A_s-\bar A_s)^2/\sigma_s^2$"), ("D_chi2", r"$\chi^2(B^+)-\chi^2(B^-)$")), 1):
        a = axs[k]; ps.letter(a, "bc"[k - 1])
        o, r_ = np.array(pp[key]["obs"]), np.array(pp[key]["rep"])
        a.hist(r_, bins=40, color=C["replica"], alpha=0.5, label="Replicated data")
        a.axvline(np.median(o), color=C["observed"], lw=2,
                  label=f"Observed ($p_B={pp[key].get('p_B', pp[key].get('p_B_two_sided')):.2f}$)")
        a.set_xlabel(lab); a.set_ylim(0, a.get_ylim()[1] * 1.3); a.legend(loc="upper right", fontsize=10)
    axs[1].set_ylabel("Replicates")
    plt.tight_layout(); save(fig, "validation_ppc_unfitted.pdf")

# ------------------------------------------------ toys: injection, power, null distribution, coverage, chi2 p-value
toys = [r for r in jl(os.path.join(FREQ, "toys.jsonl")) if r.get("valid")]
if len(toys) > 200:
    import collections
    byA = collections.defaultdict(list)
    for r in toys:
        byA[r["a_true"]].append(r)
    keys = sorted(byA)
    nomA = [k for k in keys if len(byA[k]) == max(len(v) for v in byA.values())][0]
    # (1) injection and power
    fig, axs = plt.subplots(1, 3, figsize=(ps.TRIPLE, 4.2))
    a = axs[0]; ps.letter(a, "a")
    mA = [np.mean([r["A"] for r in byA[k]]) for k in keys]
    sA = [np.std([r["A"] for r in byA[k]], ddof=1) for k in keys]
    eA = [sA[i] / np.sqrt(len(byA[k])) for i, k in enumerate(keys)]
    a.errorbar(keys, np.array(mA) - np.array(keys), yerr=eA, fmt="o", color=C["freq"], capsize=3)
    a.axhline(0, color="k", ls=":", lw=0.8)
    a.set_xlabel(r"Injected $A_{\mathrm{true}}$"); a.set_ylabel(r"$\langle\hat A\rangle-A_{\mathrm{true}}$")
    a = axs[1]; ps.letter(a, "b")
    sd_pull = [np.std([(r["A"] - k) / r["err"] for r in byA[k]], ddof=1) for k in keys]
    a.plot(keys, sd_pull, "o", color=C["freq"]); a.axhline(1, color="k", ls=":", lw=0.8)
    eff = [sA[i] / np.mean([r["err"] for r in byA[k]]) for i, k in enumerate(keys)]
    a.plot(keys, eff, "s", color=C["count"], label=r"$\mathrm{sd}(\hat A)/\langle\sigma_{\mathrm{Hesse}}\rangle$")
    a.plot(keys, sd_pull, "o", color=C["freq"], label="sd of the pull")
    a.set_xlabel(r"Injected $A_{\mathrm{true}}$"); a.set_ylabel("Width"); a.set_ylim(0.8, 1.2); a.legend(fontsize=10)
    a = axs[2]; ps.letter(a, "c")
    crit = stats.chi2.ppf(0.9973, 1)    # 3 sigma
    pw = [np.mean([r["q0"] > 3.84 for r in byA[k]]) for k in keys]
    pw3 = [np.mean([r["q0"] > crit for r in byA[k]]) for k in keys]
    se = lambda p, n: np.sqrt(p * (1 - p) / n)
    a.errorbar(keys, pw, yerr=[se(p, len(byA[k])) for p, k in zip(pw, keys)], fmt="o-", color=C["freq"], label=r"$q(0)>3.84$ (95%)")
    a.errorbar(keys, pw3, yerr=[se(p, len(byA[k])) for p, k in zip(pw3, keys)], fmt="s--", color=C["count"], label=r"$q(0)>9$ ($3\sigma$)")
    a.set_xlabel(r"Injected $A_{\mathrm{true}}$"); a.set_ylabel("Probability to reject $A=0$"); a.legend(fontsize=10)
    plt.tight_layout(); save(fig, "validation_injection_power.pdf")

    # (2) null distribution and uniformity of the p-value, coverage statistic at the injected value
    fig, axs = plt.subplots(1, 3, figsize=(ps.TRIPLE, 4.2))
    q0 = np.array([r["q0"] for r in byA.get(0.0, [])]); a = axs[0]; ps.letter(a, "a")
    xs = np.logspace(-3, 1.2, 100)
    if len(q0):
        srt = np.sort(q0)[::-1]; sf = np.arange(1, len(srt) + 1) / len(srt)
        a.step(srt, sf, color=C["freq"], lw=1.8, label=f"Toys, $A_{{\\mathrm{{true}}}}=0$ ({len(q0)})")
    a.plot(xs, stats.chi2.sf(xs, 1), color=C["model"], lw=1.5, label=r"$\chi^2_1$ (Wilks)")
    a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(1e-3, 1.2); a.set_xlabel(r"$q(0)$"); a.set_ylabel(r"$P(q\geq q_{\mathrm{obs}})$")
    a.legend(fontsize=10, loc="lower left")
    a = axs[1]; ps.letter(a, "b")
    allp = []
    for k in keys:
        t_ = np.array([r["t_true"] for r in byA[k]]); allp.append(stats.chi2.sf(t_, 1))
        if k == nomA:
            pn = np.sort(stats.chi2.sf(t_, 1)); a.plot(np.arange(1, len(pn) + 1) / (len(pn) + 1), pn, color=C["freq"], lw=2,
                                                       label=f"Nominal ($n={len(pn)}$)")
    pa_ = np.sort(np.concatenate(allp)); a.plot(np.arange(1, len(pa_) + 1) / (len(pa_) + 1), pa_, color=C["count"], lw=1.2, ls="--",
                                               label="All injected values")
    a.plot([0, 1], [0, 1], "k:", lw=0.8)
    a.set_xlabel("Uniform quantile"); a.set_ylabel(r"$p$-value of $t(A_{\mathrm{true}})$"); a.legend(fontsize=10)
    a = axs[2]; ps.letter(a, "c")
    tn = np.array([r["t_true"] for r in byA[nomA]])
    a.hist(tn, bins=np.linspace(0, 10, 41), density=True, color=C["freq"], alpha=0.45, label="Toys, nominal")
    xs2 = np.linspace(0.02, 10, 300); a.plot(xs2, stats.chi2.pdf(xs2, 1), color=C["model"], lw=1.5, label=r"$\chi^2_1$")
    a.axvline(1.0, color="gray", ls=":", lw=0.8)
    a.set_xlabel(r"$t(A_{\mathrm{true}})=-2\ln[L(A_{\mathrm{true}})/L(\hat A)]$"); a.set_ylabel("Density"); a.set_ylim(0, 1.6); a.legend(fontsize=10)
    plt.tight_layout(); save(fig, "validation_null_wilks.pdf")

    # (3) goodness of fit with toys (50 bins of 10 MeV per charge, as in the Bayesian analysis)
    gd = json.load(open(os.path.join(FREQ, "gof_data.json"))) if os.path.exists(os.path.join(FREQ, "gof_data.json")) else None
    gt = [r for r in jl(os.path.join(FREQ, "gof.jsonl")) if r.get("valid")]
    if gd and len(gt) > 200:
        fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 4.2))
        for k, (b, lab) in enumerate((("50", "10 MeV bins"), ("200", "2.5 MeV bins"))):
            a = axs[k]; ps.letter(a, "ab"[k])
            ch = np.array([r[f"chi2_{b}"] for r in gt]); nd = np.median([r[f"ndof_{b}"] for r in gt])
            a.hist(ch, bins=40, density=True, color=C["replica"], alpha=0.5, label=f"Toys from the fitted model ($n={len(ch)}$)")
            xs3 = np.linspace(min(ch.min(), gd[b]["chi2"]) - 5, max(ch.max(), gd[b]["chi2"]) + 5, 300)
            a.plot(xs3, stats.chi2.pdf(xs3, nd), color=C["model"], lw=1.5, label=f"$\\chi^2$, {int(nd)} d.o.f.")
            a.axvline(gd[b]["chi2"], color=C["observed"], lw=2, label=f"Data: ${gd[b]['chi2']:.0f}$")
            a.set_xlabel(f"Binned $\\chi^2$ (2$\\times${b} bins, {lab})"); a.set_ylabel("Density")
            a.set_ylim(0, a.get_ylim()[1] * 1.25); a.legend(fontsize=10, loc="upper right")
        plt.tight_layout(); save(fig, "validation_gof_toys.pdf")
        gsum = {}
        for b in ("50", "100", "200"):
            ch = np.array([r[f"chi2_{b}"] for r in gt]); nd = float(np.median([r[f"ndof_{b}"] for r in gt]))
            gsum[b] = dict(n=len(ch), ndof=nd, toys_mean=float(ch.mean()), toys_sd=float(ch.std(ddof=1)), obs=gd[b]["chi2"],
                           p_toys=float((np.sum(ch >= gd[b]["chi2"]) + 1) / (len(ch) + 1)),
                           p_asymptotic=float(stats.chi2.sf(gd[b]["chi2"], nd)),
                           z_obs=float((gd[b]["chi2"] - ch.mean()) / ch.std(ddof=1)))
        json.dump(gsum, open(os.path.join(FREQ, "gof_summary.json"), "w"), indent=1); print(json.dumps(gsum, indent=1))
    obs_chi2 = None; ch = np.array([0.0]); ndof = 0.0
    # numbers for the text
    summ_t = {}
    for k in keys:
        rr = byA[k]; A_ = np.array([r["A"] for r in rr]); e_ = np.array([r["err"] for r in rr])
        t_ = np.array([r["t_true"] for r in rr]); q_ = np.array([r["q0"] for r in rr])
        summ_t[str(k)] = dict(n=len(rr), bias=float(A_.mean() - k), bias_err=float(A_.std(ddof=1) / np.sqrt(len(rr))),
                              sd=float(A_.std(ddof=1)), mean_hesse=float(e_.mean()),
                              pull_mean=float(np.mean((A_ - k) / e_)), pull_sd=float(np.std((A_ - k) / e_, ddof=1)),
                              cov68=float(np.mean(t_ <= 1.0)), cov95=float(np.mean(t_ <= 3.841)),
                              rej95=float(np.mean(q_ > 3.841)), rej3sigma=float(np.mean(q_ > 9.0)),
                              frac_invalid=float(1 - len(rr) / max(1, len(rr))))
    json.dump(summ_t, open(os.path.join(FREQ, "toys_summary.json"), "w"), indent=1)
    print(json.dumps(summ_t, indent=1))

# ------------------------------------------------ Neyman belt
belt = [r for r in jl(os.path.join(FREQ, "belt.jsonl")) if r.get("valid")]
if len(belt) > 500 and prof:
    import collections
    byA = collections.defaultdict(list)
    for r in belt:
        byA[r["a_true"]].append(r["A"])
    ks = sorted(byA)
    lo = np.array([np.percentile(byA[k], 15.865) for k in ks]); hi = np.array([np.percentile(byA[k], 84.135) for k in ks])
    lo95 = np.array([np.percentile(byA[k], 2.275) for k in ks]); hi95 = np.array([np.percentile(byA[k], 97.725) for k in ks])
    a_obs = prof["all"]["A"]
    ks_ = np.array(ks)
    fig, ax = plt.subplots(figsize=(ps.SINGLE, 5.0))
    ax.fill_betweenx(ks_, lo95, hi95, color=C["freq"], alpha=0.15, label="95% belt")
    ax.fill_betweenx(ks_, lo, hi, color=C["freq"], alpha=0.3, label="68% belt")
    ax.axvline(a_obs, color=C["observed"], lw=1.5, label=r"Observed $\hat A$")
    # interval: A_true such that a_obs lies inside the belt (interpolation of the belt edges)
    fine = np.linspace(ks_.min(), ks_.max(), 2000)
    lo_f, hi_f = np.interp(fine, ks_, lo), np.interp(fine, ks_, hi)
    inside = fine[(a_obs >= lo_f) & (a_obs <= hi_f)]
    ax.hlines([inside.min(), inside.max()], a_obs - 0.04, a_obs + 0.04, color=C["observed"], ls="--", lw=1)
    ax.set_xlabel(r"$\hat A$ in the toys"); ax.set_ylabel(r"$A_{\mathrm{true}}$"); ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout(); save(fig, "validation_neyman_belt.pdf")
    json.dump(dict(n_per_point=int(np.median([len(v) for v in byA.values()])), grid=ks, belt68=[lo.tolist(), hi.tolist()],
                   interval68=[float(inside.min()), float(inside.max())], observed=a_obs,
                   minos68=prof["all"]["minos"]), open(os.path.join(FREQ, "belt_summary.json"), "w"), indent=1)
    print("belt interval", inside.min(), inside.max(), "minos", prof["all"]["minos"])

# ------------------------------------------------ sigma_B rule
bl = jl(os.path.join(FREQ, "barlow.jsonl"))
if len(bl) > 50:
    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 4.2))
    out = {}
    for k, (cut, lab) in enumerate((("probk_0.6", r"$\mathtt{ProbK}>0.6$"), ("probpi_0.4", r"$\mathtt{ProbPi}<0.4$"))):
        rows = [r for r in bl if r["v_full"] and r[cut]["valid"]]
        sh = np.array([r[cut]["A"] - r["A_full"] for r in rows])
        sB = np.sqrt(np.abs(np.array([r[cut]["s"] ** 2 - r["s_full"] ** 2 for r in rows])))
        a = axs[k]; ps.letter(a, "ab"[k])
        a.hist(sh / np.median(sB), bins=30, density=True, color=C["freq"], alpha=0.45, label="Bootstrap replicas")
        xs = np.linspace(-4, 4, 200); a.plot(xs, stats.norm.pdf(xs), color=C["model"], lw=1.5, label="Standard normal")
        a.set_xlabel(r"Shift of $\hat A$ / median $\sigma_B$"); a.set_ylabel("Density"); a.set_xlim(-4, 4)
        ps.panel_label(a, f"{lab}: sd(shift) $= {sh.std(ddof=1):.4f}$\nmedian $\\sigma_B = {np.median(sB):.4f}$", loc="upper right")
        out[cut] = dict(n=len(rows), sd_shift=float(sh.std(ddof=1)), median_sigmaB=float(np.median(sB)),
                        mean_shift=float(sh.mean()), mean_shift_err=float(sh.std(ddof=1) / np.sqrt(len(rows))))
    axs[0].legend(loc="upper left", fontsize=10)
    plt.tight_layout(); save(fig, "validation_barlow.pdf")
    json.dump(out, open(os.path.join(FREQ, "barlow_summary.json"), "w"), indent=1); print(out)
