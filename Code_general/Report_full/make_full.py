"""Builds the production-scale notebooks of Report_full/ from those of ../Report/ (the numbers of replicas, toys and chains of
the report). `python make_full.py` writes ./ ; `python make_full.py smoke` writes ../Report_smoke/ with tiny counts, only
to test that every replaced block runs. Notebooks are written without outputs; run them with run_full.sh."""
import json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "Report"))
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
ONLY = [a for a in sys.argv[1:] if a != "smoke"]            # optional: build only these notebooks (without touching the others)
DST = os.path.abspath(os.path.join(HERE, "..", "Report_smoke")) if SMOKE else HERE
os.makedirs(DST, exist_ok=True)
for f in ("plot_style.py", "environment.yml"):
    shutil.copy(os.path.join(SRC, f), os.path.join(DST, f))
for d in ("data", "figures", "results"):
    os.makedirs(os.path.join(DST, d), exist_ok=True)
if not os.path.exists(os.path.join(DST, "data")) or not os.listdir(os.path.join(DST, "data")):
    for f in os.listdir(os.path.join(SRC, "data")):
        os.symlink(os.path.join(SRC, "data", f), os.path.join(DST, "data", f))   # same data files, not copied

# production values (report) and, for the smoke test, tiny ones
P = dict(multistart=30, boot=500, toy_ahat=1000, toy_other=300, toy_size=500, boot_syst=300,
         perm=2000, subs=500, nsteps=25000, burn=5000, ndisp=10000, mh=120000, mh_burn=20000,
         b_nsteps=25000, b_burn=5000, n_toys=200, toy_steps=12000, toy_burn=2000, ppc1=1000, ppc2=500,
         lowstat_min=10000, lowstat_burn=2000, low_toys1=450, low_toys2=900, low_steps=4000, low_burn=1000,
         rf_trees=[200, 400, 600, 800], gb_trees=[100, 200, 300, 400], cv=5, ffnn_cv=5, mc=300, de=50, bs=2000)
if SMOKE:
    P.update(multistart=3, boot=4, toy_ahat=3, toy_other=2, toy_size=3, boot_syst=3, perm=20, subs=5, nsteps=600, burn=100,
             ndisp=200, mh=2000, mh_burn=500, b_nsteps=500, b_burn=100, n_toys=2, toy_steps=400, toy_burn=100, ppc1=20, ppc2=10,
             lowstat_min=500, lowstat_burn=100, low_toys1=2, low_toys2=2, low_steps=300, low_burn=50,
             rf_trees=[50], gb_trees=[50], cv=2, ffnn_cv=2, mc=5, de=2, bs=20)

ffnn_old = """for layers in ((16,), (32, 16), (64, 32, 16)):
    for dropout in (0.1, 0.3):
        aucs = []
        for tr, te in cv3.split(Xs_train, y_train):
            net = train_ffnn(Xs_train[tr], y_train[tr], Xs_train[te], y_train[te], layers=layers, dropout=dropout)
            aucs.append(roc_auc_score(y_train[te], net.predict(Xs_train[te], verbose=0).ravel()))
        ffnn_res.append((layers, dropout, np.mean(aucs), np.std(aucs)))
ffnn_res = pd.DataFrame(ffnn_res, columns=["layers", "dropout", "cv_auc", "fold_sd"])"""
lrs = "(1e-3,)" if SMOKE else "(3e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2)"
archs = "((16,),)" if SMOKE else "((8,), (16,), (32, 16), (64, 32), (64, 32, 16), (128, 64, 32), (128, 64, 32, 16))"
drops = "(0.1,)" if SMOKE else "(0.1, 0.2, 0.3, 0.4)"
ffnn_new = f"""for layers in {archs}:
    for dropout in {drops}:
        for lr in {lrs}:                                  # 7 architectures x 4 dropouts x 6 learning rates = 168 configurations
            aucs = []
            for tr, te in cv3.split(Xs_train, y_train):
                net = train_ffnn(Xs_train[tr], y_train[tr], Xs_train[te], y_train[te], layers=layers, dropout=dropout, lr=lr)
                aucs.append(roc_auc_score(y_train[te], net.predict(Xs_train[te], verbose=0).ravel()))
            ffnn_res.append((layers, dropout, lr, np.mean(aucs), np.std(aucs)))
ffnn_res = pd.DataFrame(ffnn_res, columns=["layers", "dropout", "lr", "cv_auc", "fold_sd"])"""

R = {
 "01_Frequentist_fit": [("for k in range(10):\n    # random starting point", f"for k in range({P['multistart']}):\n    # random starting point")],
 "02_Frequentist_validation": [
    ("N_BOOT = 60", f"N_BOOT = {P['boot']}"),
    ("N_TOY = {a: (40 if a == A_hat else 20) for a in A_GRID}", f"N_TOY = {{a: ({P['toy_ahat']} if a == A_hat else {P['toy_other']}) for a in A_GRID}}"),
    ("for k in range(30):\n    mp_t, mn_t = make_toy(best_vals, A_hat)", f"for k in range({P['toy_size']}):\n    mp_t, mn_t = make_toy(best_vals, A_hat)")],
 "03_Frequentist_systematics": [("for r in range(20):\n    idx = np.random.choice(len(sample)", f"for r in range({P['boot_syst']}):\n    idx = np.random.choice(len(sample)")],
 "04_Dalitz_plane": [("for _ in range(500):\n    q_shuffled", f"for _ in range({P['perm']}):\n    q_shuffled"),
                     ("for _ in range(100):                                                 # subsamples", f"for _ in range({P['subs']}):                                          # subsamples")],
 "05_Bayesian_fit": [("NWALKERS, NSTEPS, BURN = 32, 8000, 1500", f"NWALKERS, NSTEPS, BURN = 32, {P['nsteps']}, {P['burn']}"),
                     ("N_DISP = 3000", f"N_DISP = {P['ndisp']}"),
                     ("np.cov(chain.T), 30000)\nmh = mh[5000:]", f"np.cov(chain.T), {P['mh']})\nmh = mh[{P['mh_burn']}:]")],
 "06_Bayesian_systematics_and_checks": [
    ("NWALKERS, NSTEPS, BURN = 32, 4000, 800", f"NWALKERS, NSTEPS, BURN = 32, {P['nsteps']}, {P['burn']}"),
    ("def bayes_fit(s, spec=NOM, nsteps=1500, burn=300,", f"def bayes_fit(s, spec=NOM, nsteps={P['b_nsteps']}, burn={P['b_burn']},"),
    ("N_TOYS = 20", f"N_TOYS = {P['n_toys']}"),
    ("bayes_fit(toy, nsteps=1000, burn=200, start_from=v_true)", f"bayes_fit(toy, nsteps={P['toy_steps']}, burn={P['toy_burn']}, start_from=v_true)"),
    ("bayes_fit(s, nsteps=2000, burn=400)", f"bayes_fit(s, nsteps={P['b_nsteps']}, burn={P['b_burn']})"),
    ("bayes_fit(region, nsteps=2000, burn=400)", f"bayes_fit(region, nsteps={P['b_nsteps']}, burn={P['b_burn']})"),
    ("bayes_fit(sample[keep.to_numpy()], nsteps=2000, burn=400)", f"bayes_fit(sample[keep.to_numpy()], nsteps={P['b_nsteps']}, burn={P['b_burn']})"),
    ("np.random.choice(len(chain), 200, replace=False)", f"np.random.choice(len(chain), {P['ppc1']}, replace=False)"),
    ("np.random.choice(len(chain), 100, replace=False)", f"np.random.choice(len(chain), {P['ppc2']}, replace=False)")],
 "07_Bayesian_low_statistics": [
    ("bayes_fit(s_min, MIN_SPEC, nsteps=2500, burn=500)", f"bayes_fit(s_min, MIN_SPEC, nsteps={P['lowstat_min']}, burn={P['lowstat_burn']})"),
    ("    for i in range(15):\n        mp_t, mn_t = toy_minimal(n_total)", f"    for i in range({P['low_toys1']}):\n        mp_t, mn_t = toy_minimal(n_total)"),
    ("nsteps=800, burn=200, nwalkers=16", f"nsteps={P['low_steps']}, burn={P['low_burn']}, nwalkers=16"),
    ("    for _ in range(60):\n        mp_t, mn_t = toy_minimal(n_total)", f"    for _ in range({P['low_toys2']}):\n        mp_t, mn_t = toy_minimal(n_total)")],
 "08_ML_classifiers": [
    ("{\"n_estimators\": [100, 300], \"max_depth\": [4, 8, None], \"min_samples_leaf\": [5, 20]},\n                     cv=3,",
     "{\"n_estimators\": %s, \"max_depth\": [6, 8, 10, None], \"min_samples_leaf\": [5, 10, 20]},\n                     cv=%d," % (P['rf_trees'], P['cv'])),
    ("{\"n_estimators\": [100, 200], \"max_depth\": [2, 3, 4], \"learning_rate\": [0.05, 0.1]},\n                     cv=3,",
     "{\"n_estimators\": %s, \"max_depth\": [1, 2, 3, 4], \"learning_rate\": [0.01, 0.05, 0.1]},\n                     cv=%d," % (P['gb_trees'], P['cv'])),
    ("cv3 = StratifiedKFold(n_splits=3,", f"cv3 = StratifiedKFold(n_splits={P['ffnn_cv']},"),
    (ffnn_old, ffnn_new),
    ("for _ in range(100)]", f"for _ in range({P['mc']})]"),
    ("for s in range(1, 5)]", f"for s in range(1, {P['de'] + 1})]"),
    ("def bootstrap_sd(y_, s_, n=300):", f"def bootstrap_sd(y_, s_, n={P['bs']}):")],
 "09_ML_measurement": [
    ("sam.run_mcmc(p0, 3000, progress=False)", f"sam.run_mcmc(p0, {P['nsteps']}, progress=False)"),
    ("A_bayes_sel = sam.get_chain(discard=600, flat=True)[:, 2]", f"A_bayes_sel = sam.get_chain(discard={P['burn']}, flat=True)[:, 2]"),
    ("sam_a.run_mcmc(p0a, 3000, progress=False)", f"sam_a.run_mcmc(p0a, {P['nsteps']}, progress=False)"),
    ("A_bayes_full = sam_a.get_chain(discard=600, flat=True)[:, 2]", f"A_bayes_full = sam_a.get_chain(discard={P['burn']}, flat=True)[:, 2]"),
    ("sam2.run_mcmc(p02, 3000, progress=False)", f"sam2.run_mcmc(p02, {P['nsteps']}, progress=False)"),
    ("A_bayes_sel2 = sam2.get_chain(discard=600, flat=True)[:, 2]", f"A_bayes_sel2 = sam2.get_chain(discard={P['burn']}, flat=True)[:, 2]")],
}
banner = ("> **Production run.** Same notebook with the numbers of replicas, toy experiments and chain lengths of the report. "
          "Run with `run_full.sh`; expect several hours.\n\n" if not SMOKE else "> **Smoke test.** Tiny counts: only checks that the code runs.\n\n")
bad = 0
for name, reps in R.items():
    if ONLY and name not in ONLY:
        continue
    nb = json.load(open(os.path.join(SRC, name + ".ipynb")))
    found = {old: 0 for old, _ in reps}
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            for old, new in reps:
                if old in src:
                    src = src.replace(old, new); found[old] += 1
            c["source"] = src.splitlines(keepends=True)
            c["outputs"] = []; c["execution_count"] = None
    missing = [o[:60] for o, n in found.items() if n == 0]
    for c in nb["cells"]:
        if c["cell_type"] == "markdown":
            c["source"] = (banner + "".join(c["source"])).splitlines(keepends=True); break
    json.dump(nb, open(os.path.join(DST, name + ".ipynb"), "w"), indent=1)
    print(f"{name}: replaced {sum(found.values())}/{len(reps)}", "MISSING " + str(missing) if missing else "")
    bad += len(missing)
sys.exit(1 if bad else 0)
