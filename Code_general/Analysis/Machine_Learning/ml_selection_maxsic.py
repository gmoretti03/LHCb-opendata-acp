"""Stability of A_raw for the cut at the maximum of the significance-improvement curve (extra row of the three-methods table).

Same code as the score>0.5 refits of 05_ml_selection_and_local_asymmetry.ipynb, with the threshold at the maximum of
eps_S/sqrt(eps_B) of the reference (gradient-boosting) score on the test set. Saves results/nominal/maxsic_selected.json.
usage (cern env, from this directory):  python ml_selection_maxsic.py
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, "."); sys.path.insert(0, "..")
import ml_tools as mn

DATA_DIR = "../../../opendata-project/Data"
P = lambda n: os.path.join(os.getcwd(), "results", "nominal", n)
NSTEPS = 25000                       # same chain length as the nominal Bayesian fit
t0 = time.time()
bayes, bn, lf = mn.load_bayes_summary(), mn.import_bayes_model(), mn.import_lhcb_freq()
MU, SIGMA = bayes["nominal"]["mu"]["median"], bayes["nominal"]["sigma"]["median"]
A_B, SD_B = bayes["nominal"]["A"]["median"], bayes["nominal"]["A"]["sd"]
data = mn.load_data(DATA_DIR); q = data.B_Charge.to_numpy()
labeled = mn.weak_labels(data, MU, SIGMA); sp = mn.make_split(labeled)
gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
gbm.fit(sp["X_train"], sp["y_train"], sample_weight=sp["sw_train"])
assert abs(roc_auc_score(sp["y_test"], gbm.predict_proba(sp["X_test"])[:, 1]) - json.load(open(P("models.json")))["test_auc"]["Gradient Boosting"]) < 1e-9
# working point: maximum of eps_S / sqrt(eps_B) on the test set (eps_B >= 1e-3), rounded to two decimals
f, t, th = roc_curve(sp["y_test"], gbm.predict_proba(sp["X_test"])[:, 1]); ok = f >= 1e-3
k = np.argmax(t[ok] / np.sqrt(f[ok])); THR = float(np.round(th[ok][k], 2))
print(f"max SIC {t[ok][k]/np.sqrt(f[ok][k]):.3f} at eps_S={t[ok][k]:.3f}, eps_B={f[ok][k]:.3f}; threshold {THR}", flush=True)
s_full = gbm.predict_proba(data[mn.FEATURES].to_numpy())[:, 1]; sel = s_full > THR
a_full, sa_full = mn.count_asym(q); a_sel, sa_sel = mn.count_asym(q[sel])
cfg_f = lf.Config(fit_window=(mn.LO, mn.HI))
def freq(m):
    d = pd.DataFrame({"B_Mass": data.B_M.to_numpy()[m], "B_Charge": q[m]})
    s = lf.split_by_charge(d, col="B_Mass"); fit = lf.fit_joint(s, cfg_f); tt, p, _ = lf.wilks_test(fit, {"A_CP": 0.0}); g = lf.gof_joint(fit, s, cfg_f)
    return {"n": int(m.sum()), "A": float(fit.values["A_CP"]), "A_err": float(fit.errors["A_CP"]), "valid": bool(fit.valid),
            "accurate": bool(fit.accurate), "chi2": float(g.chi2), "ndof": int(g.ndof), "wilks_t": float(tt), "wilks_p": float(p)}
ff, fs = freq(np.ones(len(data), bool)), freq(sel)
out = {"threshold": THR, "n_full": int(len(data)), "n_selected": int(sel.sum()),
       "eps_S": float(t[ok][k]), "eps_B": float(f[ok][k]), "sic": float(t[ok][k] / np.sqrt(f[ok][k])),
       "counting": {"full": {"A": a_full, "sd": sa_full}, "selected": {"A": a_sel, "sd": sa_sel}, "expected_shift_sd": mn.nested_shift_sigma(sa_sel, sa_full)},
       "frequentist": {"full": ff, "selected": fs, "expected_shift_sd": mn.nested_shift_sigma(fs["A_err"], ff["A_err"])}}
json.dump(out, open(P("maxsic_selected.json"), "w"), indent=1)
print(f"counting {a_sel:+.4f}+-{sa_sel:.4f}; freq {fs['A']:+.4f}+-{fs['A_err']:.4f} (n={sel.sum()}, {time.time()-t0:.0f} s); starting Bayesian refit", flush=True)
r = bn.fit(data.B_M.to_numpy()[sel], q[sel], bn.NOMINAL, nsteps=NSTEPS); r.pop("_thinned", None)
out["bayes"] = {"full": {"A": A_B, "sd": SD_B, "chi2": bayes["nominal"]["gof"]["chi2"], "ndof": bayes["nominal"]["gof"]["ndof"]},
                "selected": {"A": r["A"]["median"], "sd": r["A"]["sd"], "chi2": r["gof"]["chi2"], "ndof": r["gof"]["ndof"]},
                "expected_shift_sd": mn.nested_shift_sigma(r["A"]["sd"], SD_B)}
json.dump(out, open(P("maxsic_selected.json"), "w"), indent=1)
print(f"Bayes {r['A']['median']:+.4f}+-{r['A']['sd']:.4f}  total {time.time()-t0:.0f} s", flush=True)
