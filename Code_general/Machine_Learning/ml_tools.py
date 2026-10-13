"""
ml_tools.py -- helpers shared by the machine-learning notebooks 04 and 05 of this directory.

Aligned with the nominal mass model of the Bayesian and frequentist analyses (fit window [5100, 5600] MeV).

    weak labels    signal-like     |m - mu| < 2 sigma
                   background-like m in [SB_LO, mu - 4 sigma] U [mu + 4 sigma, HI]
                   (mu, sigma from the nominal Bayesian fit; SB_LO = 5150 MeV excludes the partially
                   reconstructed background below 5150 MeV, which is not representative of the
                   background under the peak)
    class weights  'balanced' for every model, w_c = N / (2 N_c)
    FFNN           seeds fixed and TensorFlow op determinism enabled (reproducible training)

Contents: loading of the other analyses (`load_bayes_summary`, `import_bayes_model`, `import_lhcb_freq`), data and
weak labels, statistical helpers (bootstrap AUC, counting asymmetry, nested-subsample uncertainty, homogeneity test of
the local asymmetry) and checkpoint helpers (`json_cached`, `jsonl_*`).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import uproot

M_K = 493.677
LO, HI = 5100.0, 5600.0          # nominal fit window
SB_LO = 5150.0                   # lower edge of the low sideband used for the weak labels
FEATURES = [f"H{i}_{v}" for i in (1, 2, 3) for v in ("PX", "PY", "PZ", "ProbK", "ProbPi", "IPChi2")] \
    + ["B_FlightDistance", "B_VertexChi2"]


# ---------------------------------------------------------------- inputs from the other analyses
HERE = os.path.dirname(os.path.abspath(__file__))
BAYES_DIR = os.path.join(HERE, "..", "Bayesian")
FREQ_DIR = os.path.join(HERE, "..", "Frequentist")


def load_bayes_summary(path: str | None = None) -> dict:
    """Nominal Bayesian summary (written by notebook 01 of the Bayesian analysis)."""
    path = path or os.path.join(BAYES_DIR, "results", "nominal", "nominal_summary.json")
    with open(path) as f:
        return json.load(f)


def import_bayes_model(directory: str | None = None):
    """Import bayes_model.py (Bayesian mass model and sampler) for the mass refits of the selected samples."""
    sys.path.insert(0, directory or BAYES_DIR)
    import bayes_model
    return bayes_model


def import_lhcb_freq(directory: str | None = None):
    """Import lhcb_freq.py (frequentist fits of the common mass model). Returns None when the frequentist analysis is not
    in the tree: the blocks that need it are then read from their saved results."""
    d = directory or FREQ_DIR
    if not os.path.exists(os.path.join(d, "lhcb_freq.py")):
        return None
    sys.path.insert(0, d)
    import lhcb_freq
    return lhcb_freq


# ---------------------------------------------------------------- data
def load_data(data_dir: str, prob_k_min: float = 0.5, prob_pi_max: float = 0.5) -> pd.DataFrame:
    """Shared preselection, nominal window, B mass/charge, the two opposite-sign K+K- masses, features.
    The PID cuts are arguments (defaults = nominal) for the high-background study."""
    br = [f"H{i}_{v}" for i in (1, 2, 3) for v in ("PX", "PY", "PZ", "ProbK", "ProbPi", "isMuon", "Charge", "IPChi2")]
    br += ["B_FlightDistance", "B_VertexChi2"]
    frames = []
    for pol in ("Down", "Up"):
        with uproot.open(os.path.join(data_dir, f"B2HHH_Magnet{pol}.root")) as f:
            d = f["DecayTree"].arrays(br, library="pd")
        d["magnet"] = pol.lower()
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    sel = np.ones(len(df), bool)
    for i in (1, 2, 3):
        sel &= (df[f"H{i}_ProbPi"] < prob_pi_max) & (df[f"H{i}_ProbK"] > prob_k_min) & (~df[f"H{i}_isMuon"].astype(bool))
    df = df[sel].reset_index(drop=True)
    p = {i: df[[f"H{i}_PX", f"H{i}_PY", f"H{i}_PZ"]].to_numpy() for i in (1, 2, 3)}
    e = {i: np.sqrt((p[i] ** 2).sum(1) + M_K ** 2) for i in (1, 2, 3)}
    def mass(*idx):
        return np.sqrt(np.clip(sum(e[i] for i in idx) ** 2 - (sum(p[i] for i in idx) ** 2).sum(1), 0, None))
    df["B_M"] = mass(1, 2, 3)
    df["B_Charge"] = df.H1_Charge + df.H2_Charge + df.H3_Charge
    df["m12"], df["m13"] = mass(1, 2), mass(1, 3)
    df = df[np.isin(df.B_Charge, [1, -1]) & (df.B_M > LO) & (df.B_M < HI)].reset_index(drop=True)
    assert np.all(df.H1_Charge == -df.B_Charge), "H1 is expected to be the opposite-sign kaon"
    return df


def weak_labels(df: pd.DataFrame, mu: float, sigma: float) -> pd.DataFrame:
    m = df.B_M
    sig = np.abs(m - mu) < 2 * sigma
    bkg = ((m > SB_LO) & (m < mu - 4 * sigma)) | ((m > mu + 4 * sigma) & (m < HI))
    lab = df[sig | bkg].copy()
    lab["y"] = sig[sig | bkg].astype(int).to_numpy()
    return lab


def make_split(labeled: pd.DataFrame, seed: int = 42):
    """Stratified 60/20/20 train/validation/test split of the weakly labelled candidates, standardisation fitted on the
    training set, and the 'balanced' class weights. Returns a dict (same split in notebooks 04 and 05)."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight
    X, y = labeled[FEATURES].to_numpy(), labeled.y.to_numpy()
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    scaler = StandardScaler().fit(X_train)
    return dict(X_all=X, y_all=y, X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test,
                scaler=scaler, X_train_s=scaler.transform(X_train), X_val_s=scaler.transform(X_val),
                X_test_s=scaler.transform(X_test), W=balanced_weights(y_train),
                sw_train=compute_sample_weight("balanced", y_train))


def balanced_weights(y) -> dict:
    """sklearn 'balanced': w_c = N / (2 N_c)."""
    y = np.asarray(y)
    n, n1 = len(y), int(y.sum())
    return {0: n / (2.0 * (n - n1)), 1: n / (2.0 * n1)}


# ---------------------------------------------------------------- statistics helpers
def bootstrap_auc_sd(y, s, n=2000, seed=1) -> float:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], s[i]))
    return float(np.std(vals))


def count_asym(q) -> tuple:
    q = np.asarray(q)
    n = len(q)
    if n == 0:
        return float("nan"), float("nan")
    a = (np.sum(q < 0) - np.sum(q > 0)) / n
    return float(a), float(np.sqrt(max(1 - a ** 2, 0) / n))


def nested_shift_sigma(sigma_sub: float, sigma_full: float) -> float:
    """Expected statistical spread of (estimate on a subsample) - (estimate on the full sample) when the
    subsample is contained in the full sample: sqrt(sigma_sub^2 - sigma_full^2)."""
    return float(np.sqrt(max(sigma_sub ** 2 - sigma_full ** 2, 0.0)))


def local_asym_vs_global(labels, q, min_events=20) -> dict:
    """Asymmetry of each group (cluster/bin) compared with the global one: S = (A_k - A_glob)/sigma_k,
    homogeneity chi2 = sum S^2 with (n_groups - 1) dof."""
    from scipy.stats import chi2, binom
    labels, q = np.asarray(labels), np.asarray(q)
    ag, _ = count_asym(q)
    rows, s_all = [], []
    for k in np.unique(labels):
        qk = q[labels == k]
        if len(qk) < min_events:
            continue
        a, sa = count_asym(qk)
        s = (a - ag) / sa
        rows.append({"group": int(k), "n": int(len(qk)), "A": a, "sigma_A": sa, "S_vs_global": float(s),
                     "S_vs_zero": float(a / sa)})
        s_all.append(s)
    s_all = np.array(s_all)
    nsig = int(np.sum(np.abs(s_all) > 3))
    c2, nd = float(np.sum(s_all ** 2)), len(s_all) - 1
    return {"A_glob": ag, "groups": rows, "n_groups": len(s_all), "n_3sigma": nsig, "chi2": c2, "ndof": nd,
            "p": float(chi2.sf(c2, nd)),
            "p_at_least_nsig_if_uniform": float(binom.sf(nsig - 1, len(s_all), 0.0027)) if nsig else 1.0}


# ---------------------------------------------------------------- checkpoints
def jsonl_load(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def jsonl_append(path: str, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def json_cached(path: str, compute, key: dict | None = None):
    """Load `path` if it exists and its stored `key` matches; otherwise compute(), save and return."""
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        if key is None or d.get("_key") == json.loads(json.dumps(key)):
            print("[resume]", os.path.relpath(path))
            return d
    d = compute()
    if key is not None:
        d["_key"] = key
    with open(path, "w") as f:
        json.dump(d, f, indent=1)
    print("[saved]", os.path.relpath(path))
    return d
