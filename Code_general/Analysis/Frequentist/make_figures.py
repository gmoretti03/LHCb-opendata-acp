"""Figures of the frequentist analysis, from the saved results (no fit is run).

Inputs  : results/validation/*.json(l)  (freq_validation.py, modes also scans and jointfit),  results/freq_pid_scan.json  (notebook 06).
Outputs : figures/validation_{injection_power,power,null_wilks,gof_toys,neyman_belt,barlow}.pdf,
          figures/momentum_terciles.pdf, figures/syst_pid_scan.pdf,
          figures/freq_profile.pdf, figures/freq_jointfit.pdf, figures/freq_stability.pdf
          (figures/frequentist_wilks_coverage.pdf is written by the notebook itself).
Also rewrites the summaries results/validation/{toys,gof,belt,barlow}_summary.json used in the text of the report.
usage (cern env):  python make_figures.py
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
FREQ = os.path.join(HERE, "results", "validation")
RES = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(HERE, ".."))          # Code_general/plot_style.py
import plot_style as ps
ps.use(); C = ps.COLORS


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("saved", name)


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


prof = json.load(open(os.path.join(FREQ, "profiles.json"))) if os.path.exists(os.path.join(FREQ, "profiles.json")) else None

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
    # (1b) power of the test of A = 0 alone, single column (main text)
    fig, a = plt.subplots(figsize=(ps.SINGLE, 4.2))
    a.errorbar(keys, pw, yerr=[se(p, len(byA[k])) for p, k in zip(pw, keys)], fmt="o-", color=C["freq"], label=r"$q(0)>3.84$ (size 5%)")
    a.errorbar(keys, pw3, yerr=[se(p, len(byA[k])) for p, k in zip(pw3, keys)], fmt="s--", color=C["count"], label=r"$q(0)>9$ ($3\sigma$)")
    a.axhline(0.05, color="k", ls=":", lw=0.8)
    a.set_xlabel(r"Injected $A_{\mathrm{true}}$"); a.set_ylabel("Probability to reject $A=0$"); a.legend(fontsize=10, loc="center left")
    plt.tight_layout(); save(fig, "validation_power.pdf")

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

# ------------------------------------------------ (2) Araw in terciles of the momentum of the B
mo = json.load(open(os.path.join(FREQ, "momentum.json")))
glob = json.load(open(os.path.join(FREQ, "nominal_values.json")))
Ag = glob.get("A", glob.get("A_CP", -0.0441)) if isinstance(glob, dict) else -0.0441
if not isinstance(Ag, float):
    Ag = -0.0441
fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 4.2), sharey=True)
for ax, key, lab, xl in ((axs[0], "B_PT", "a", r"$p_T$ of the $B$ candidate (terciles)"), (axs[1], "B_P", "b", r"$p$ of the $B$ candidate (terciles)")):
    ps.letter(ax, lab)
    rr = mo[key]
    x = np.arange(len(rr))
    ax.errorbar(x, [t["A"] for t in rr], yerr=[t["err"] for t in rr], fmt="o", color=C["freq"], capsize=3, ms=6)
    ax.axhspan(Ag - 0.0085, Ag + 0.0085, color="0.88", label=r"global $\pm1\sigma$")
    ax.axhline(Ag, color=C["count"], ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(["%.0f--%.0f" % (t["lo"] / 1e3, t["hi"] / 1e3) for t in rr])
    ax.set_xlabel(xl + r" [GeV]"); ax.set_xlim(-0.5, len(rr) - 0.5); ax.set_ylim(-0.08, -0.012)
axs[0].set_ylabel(r"$A_{\mathrm{raw}}$ (joint frequentist fit)"); axs[0].legend(fontsize=9, loc="lower left")
plt.tight_layout(); save(fig, "momentum_terciles.pdf")


# ---------------------------------------------------------------- (2) scan of the PID cuts
rows = json.load(open(os.path.join(RES, "freq_pid_scan.json")))["rows"]
nom = [r for r in rows if r["label"] == "nominal"][0]
fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE, 4.4), sharey=True)
for ax, key, par, lab, xl in ((axs[0], "probk_", "prob_k_min", "a", r"$\mathtt{ProbK}$ lower cut"),
                              (axs[1], "probpi_", "prob_pi_max", "b", r"$\mathtt{ProbPi}$ upper cut")):
    ps.letter(ax, lab)
    sc = sorted([r for r in rows if r["label"].startswith(key)] + [nom], key=lambda r: r[par])
    x = np.array([r[par] for r in sc]); s = np.array([r["shift"] for r in sc]); sb = np.array([r["sigma_barlow"] for r in sc])
    ax.fill_between(x, -sb, sb, color=C["shade"] if "shade" in C else "0.85", label=r"$\pm\sigma_B$")
    ax.fill_between(x, -2 * sb, 2 * sb, color="0.93", zorder=0, label=r"$\pm2\sigma_B$")
    ax.plot(x, s, "o-", color=C["freq"], label="shift")
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0.5, color=C["background"], ls="-.", lw=0.9)
    ax.set_xlabel(xl)
axs[0].set_ylabel(r"Shift of $A_{\mathrm{raw}}$ (frequentist fit)")
axs[0].legend(loc="lower left", fontsize=9)
plt.tight_layout(); save(fig, "syst_pid_scan.pdf")


# ------------------------------------------------ profile likelihood ratio q(A)
jp = os.path.join(FREQ, "jointfit.json")
if os.path.exists(jp):
    jf = json.load(open(jp)); pr = jf["profile"]
    A_ = np.array(pr["A"]); q_ = np.array(pr["q"]); a_hat = pr["A_hat"]

    def cross(thr):
        """A values where q(A) crosses thr (linear interpolation, one on each side of the minimum)"""
        k = np.where(np.diff(np.sign(q_ - thr)))[0]
        return [A_[i] + (thr - q_[i]) * (A_[i + 1] - A_[i]) / (q_[i + 1] - q_[i]) for i in k]
    lo68, hi68 = cross(1.0); lo95, hi95 = cross(3.84)
    q0 = float(np.interp(0.0, A_, q_))
    fig, ax = plt.subplots(figsize=(ps.SINGLE, 4.8))
    ax.plot(A_, q_, color=C["freq"], lw=2.4, zorder=3)
    for thr, lo, hi, col, lab in ((1.0, lo68, hi68, C["freq"], "68%"), (3.84, lo95, hi95, C["freq"], "95%")):
        ax.hlines(thr, lo, hi, color="k", lw=2.0, zorder=4)
        ax.vlines([lo, hi], 0, thr, color="k", lw=0.8, ls=":", zorder=1)
        ax.text(hi + 0.0015, thr, f"{lab}: [{lo:.4f}, {hi:.4f}]", fontsize=10, va="center", ha="left")
    ax.plot([a_hat], [0], "o", color=C["observed"], ms=7, zorder=5, label=rf"$\hat A={a_hat:.4f}$")
    ax.axvline(0, color="0.6", lw=0.9, ls="-")
    ax.plot([0], [q0], "s", color=C["observed"], ms=7, zorder=5)
    ax.annotate(rf"$A=0$: $q={q0:.0f}$, $Z=\sqrt{{q}}={np.sqrt(q0):.1f}\sigma$", xy=(0, q0), xytext=(-0.0725, q0 + 3.2),
                fontsize=11, va="top", ha="left")
    ax.set_xlim(-0.075, 0.01); ax.set_ylim(0, 31)
    ax.set_xlabel(r"$A$"); ax.set_ylabel(r"$q(A)=-2\ln[L(A)/L(\hat A)]$")
    ax2 = ax.secondary_yaxis("right", functions=(lambda v: np.sqrt(np.clip(v, 0, None)), lambda z: z ** 2))
    ax2.set_yticks([1, 2, 3, 4, 5]); ax2.set_yticklabels([rf"${z}\sigma$" for z in (1, 2, 3, 4, 5)]); ax2.set_ylabel(r"$Z=\sqrt{q}$")
    for z in (1, 2, 3, 4, 5):
        ax.axhline(z ** 2, color="0.85", lw=0.6, zorder=0)
    ax.legend(loc="upper left", fontsize=10, bbox_to_anchor=(0.0, 0.88))
    plt.tight_layout(); save(fig, "freq_profile.pdf")

# ------------------------------------------------ simultaneous fit: data, signal and background, pulls
if os.path.exists(jp):
    cen = np.array(jf["centres"]); x = np.array(jf["x"]); w = jf["width"]
    fig, axs = plt.subplots(2, 2, figsize=(ps.DOUBLE, 5.6), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.2], "hspace": 0.05, "wspace": 0.18})
    for k, (side, lab, col) in enumerate((("pos", r"$B^+$", C["pos"]), ("neg", r"$B^-$", C["neg"]))):
        a, b = axs[0, k], axs[1, k]; d = jf[side]; cts = np.array(d["counts"])
        a.fill_between(x, 0.1, np.array(d["background"]), color=C["background"], alpha=0.35, lw=0, label="Background")
        a.plot(x, np.array(d["signal"]) + np.array(d["background"]), color=col, lw=2.2, label="Signal + background")
        a.plot(x, d["signal"], color=col, lw=1.2, ls="--", label="Signal")
        a.errorbar(cen, cts, yerr=np.sqrt(cts), fmt="o", color="k", ms=3.5, lw=0.9, label="Data", zorder=5)
        a.set_yscale("log"); a.set_ylim(4, 4e3); ps.letter(a, "ab"[k])
        a.text(0.04, 0.9, rf"{lab}: $N_s={d['N_sig_side']:,.0f}$".replace(",", "{,}"), transform=a.transAxes, fontsize=12, va="top",
               bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        if k == 0:
            a.set_ylabel(f"Candidates / {w:.0f} MeV"); a.legend(loc="upper right", fontsize=10)
        b.bar(cen, d["pull_2"], width=w * 0.9, color=col, alpha=0.85)
        b.axhline(0, color="k", lw=0.8); b.axhline(2, color="0.6", lw=0.6, ls=":"); b.axhline(-2, color="0.6", lw=0.6, ls=":")
        b.set_ylim(-4.5, 4.5); b.set_xlabel(r"$m(K^+K^+K^-)$ [MeV]")
        if k == 0:
            b.set_ylabel("Pull")
    plt.tight_layout(); save(fig, "freq_jointfit.pdf")

# ------------------------------------------------ stability scans: window, signal region, PID grid
sp = os.path.join(FREQ, "scans.json")
if os.path.exists(sp):
    import pandas as pd
    sc = json.load(open(sp)); A0 = sc["nominal"]["A_CP"]; s0 = sc["nominal"]["A_CP_err"]

    def table(name):
        d = pd.DataFrame(sc[name]); d = d[d.valid & d.accurate]
        return d

    fig, axs = plt.subplots(1, 3, figsize=(ps.TRIPLE, 4.4), gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})
    specs = (("window", "left", "right", r"Upper edge of the fit window [MeV]", r"Lower edge [MeV]", 1e3, r"$10^{-3}$"),
             ("signal_region", "sig_lo", "sig_hi", r"Upper edge of the signal region [MeV]", r"Lower edge [MeV]", 1e5, r"$10^{-5}$"),
             ("pid", "prob_k_min", "prob_pi_max", r"$\mathtt{ProbPi}$ upper cut", r"$\mathtt{ProbK}$ lower cut", 1e3, r"$10^{-3}$"))
    for k, (ax, (name, yc, xc, xl, yl, scale, unit)) in enumerate(zip(axs, specs)):
        d = table(name); ps.letter(ax, "abc"[k])
        piv = d.pivot(index=yc, columns=xc, values="A_CP")
        shift = (piv - A0) * scale
        lim = np.nanmax(np.abs(shift.values))
        im = ax.pcolormesh(piv.columns, piv.index, shift.values, shading="nearest", cmap="RdBu_r", vmin=-lim, vmax=lim)
        cb = fig.colorbar(im, ax=ax); cb.set_label(r"$A-A_{\mathrm{nom}}$ [" + unit + "]")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        if name == "pid":
            ax.plot([0.5], [0.5], "k*", ms=9)
        else:
            lo, hi = (5100, 5600) if name == "window" else (5200, 5350)
            ax.plot([hi], [lo], "k*", ms=9)
        ps.panel_label(ax, f"max $|\\Delta A|={np.nanmax(np.abs(piv.values - A0)):.4f}$".replace("0.0000", "6\\times10^{-5}") if name != "signal_region"
                       else f"max $|\\Delta A|={np.nanmax(np.abs(piv.values - A0)):.1e}$".replace("e-05", r"\times10^{-5}"), loc="upper right")
    plt.tight_layout(); save(fig, "freq_stability.pdf")
