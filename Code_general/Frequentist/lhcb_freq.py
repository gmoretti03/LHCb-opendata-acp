"""
lhcb_freq.py -- helpers and settings for the frequentist B+- -> K+-K+K- analysis.

Sections
--------
1. Settings          Config, SIDES, feature lists, plot settings
2. Data & physics    loading, preselection, invariant mass, charge split
3. Plot helpers      feature plots, mass plots
4. Models            truncated Gaussian / exponentials, extended signal+background models
5. Fits              sideband fit, single fit, joint (simultaneous) fit, Wilks test
6. Diagnostics       chi2 goodness-of-fit, bootstrap coverage study

Typical use (in a notebook)::

    import lhcb_freq as lf
    cfg = lf.Config(data_dir=r"path/to/data")
    ...
    m = lf.fit_joint(samples, cfg, n_exp=2)

Parameter naming convention of the joint model
----------------------------------------------
shared   : N_sig, A_CP, mu, sigma
per side : N_bkg_<side>, lam1_<side>, [lam2_<side>, f_bkgs_<side>]   (side = "pos" | "neg")
with N_sig(B+) = N_sig (1 - A_CP) / 2 ,  N_sig(B-) = N_sig (1 + A_CP) / 2
i.e.  A_CP = (N_neg - N_pos) / (N_neg + N_pos).
"""
from __future__ import annotations

import copy
import inspect
import operator
import os
import re
import warnings
import itertools
from dataclasses import dataclass, field, replace
from functools import reduce
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
import vector
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL, UnbinnedNLL
from iminuit.util import describe
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm
from scipy.stats import crystalball   # [Giacomo, 2026-10-06] for the Crystal Ball signal variant
from tqdm.auto import tqdm

# =============================================================================
# 1. Settings
# =============================================================================
GLOBAL_FEATURES = ["B_FlightDistance", "B_VertexChi2"]
PRODUCT_FEATURES = ["PX", "PY", "PZ", "ProbK", "ProbPi", "Charge", "isMuon"]
EXTRA_PRODUCT_FEATURES = ["IPChi2"]  # present in the raw files only


@dataclass(frozen=True)
class Side:
    """One of the two charge samples (B+ or B-)."""
    charge: int
    label: str
    color: str
    asym_sign: int  # N_sig(side) = N_sig * (1 + asym_sign * A_CP) / 2


SIDES: Dict[str, Side] = {
    "pos": Side(+1, r"$B^+$", "navy",    -1),
    "neg": Side(-1, r"$B^-$", "darkred", +1),
}

DEFAULT_LIMITS = {
    "N_sig":  (0, None),
    "N_bkg":  (0, None),
    "A_CP":   (-1.0, 1.0),
    "sigma":  (1.0, 100.0),
    "lam":    (1e-6, 1.0),   # lam1, lam2, ...
    "f_bkgs": (0.0, 1.0),
    # [Giacomo, 2026-10-06] tail parameters of the Crystal Ball (signal variant)
    "beta":   (0.3, 5.0),
    "m_cb":   (1.01, 50.0),
    # "mu" defaults to cfg.sig_region (see param_limit)
}


@dataclass
class Config:
    """All the knobs of the analysis in one place."""
    # --- I/O --------------------------------------------------------------
    data_dir:    Path = Path(os.environ.get("LHCB_DATA_DIR", "./Data"))
    data_files:  Sequence[str] = ("B2HHH_MagnetDown.root", "B2HHH_MagnetUp.root")
    data_tree:   str = "DecayTree"
    sim_file:    Path = Path("./Data/PhaseSpaceSimulation.root")
    sim_tree:    str = "PhaseSpaceTree"
    # --- decay ------------------------------------------------------------
    n_products:  int = 3
    m_K:         float = 493.677      # kaon mass  [MeV]
    m_Pi:        float = 139.57039    # pi mass    [MeV]
    m_B:         float = 5279.41      # B mass     [MeV]
    # [Giacomo, 2026-10-06] chi_c0: half-width 60 MeV (not 30), as agreed and as in the Bayesian analysis.
    # In the data the chi_c0 peak in m(K+K-) is at ~3404 MeV (11 MeV below the PDG value) with sigma ~18 MeV
    # (natural width + resolution): +-30 MeV around the PDG value would leave a large part of it out.
    # D0 (sigma 6.5 MeV) and J/psi (sigma ~8 MeV) stay at +-30 MeV.
    res_windows = {
        'D':   (1864.84, 30.0),
        'J':   (3096.90, 30.0),
        'chi': (3414.70, 60.0)
    }
    # --- preselection (B -> KKK) -------------------------------------------
    prob_k_min:  float = 0.5
    prob_pi_max: float = 0.5
    # --- preselection (B -> πππ) -------------------------------------------
    prob_pi_min: float = 0.5
    prob_k_max:  float = 0.5
    # --- mass fit ---------------------------------------------------------
    fit_window:  tuple = (5100.0, 5600.0)
    sig_region:  tuple = (5200.0, 5350.0)
    n_exp:       int = 2              # number of exponentials in the background (1 or 2)
    # [Giacomo, 2026-10-06] options for the variants of the systematic study (defaults = nominal model):
    #   bkg_shape:  "charge" = background shape (lam1, lam2, f_bkgs) separate for B+ and B-;  "shared" = shared
    #   bkg_yield:  "free" = independent N_bkg_pos, N_bkg_neg;  "symmetric" = same background yield for the two charges
    #   signal:     "gauss" = truncated Gaussian;  "cb" = truncated Crystal Ball with free, shared tails (beta, m_cb)
    #   signal_per_charge: True = separate mu and sigma for B+ and B- (mu_pos, mu_neg, sigma_pos, sigma_neg)
    # [Giacomo, 2026-10-06] mass hypothesis of the tracks for B_Mass: "kaon" = kaon for all three (B->KKK analysis,
    # as stated in the report); "pid" = kaon or pion track by track from ProbK >= ProbPi (previous
    # behaviour). With the nominal cuts (ProbK > 0.5, ProbPi < 0.5) the two coincide; with looser cuts (preselection
    # variants) "pid" would reconstruct some tracks as pions and change the B mass.
    mass_hypothesis: str = "kaon"
    bkg_shape:   str = "charge"
    bkg_yield:   str = "free"
    signal:      str = "gauss"
    signal_per_charge: bool = False
    limits:      dict = field(default_factory=lambda: dict(DEFAULT_LIMITS))

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.sim_file = Path(self.sim_file)

    @property
    def particles(self) -> tuple:
        return tuple(range(1, self.n_products + 1))

    @property
    def columns(self) -> list:
        """Columns needed by the analysis (the rest is dropped)."""
        return [f"H{i}_{v}" for i in self.particles for v in PRODUCT_FEATURES] + ["magnet"]

    @property
    def window_intervals(self) -> list:
        return [tuple(self.fit_window)]

    @property
    def sideband_intervals(self) -> list:
        (lo, hi), (smin, smax) = self.fit_window, self.sig_region
        return [(lo, smin), (smax, hi)]


def _merge(base: dict, **overrides) -> dict:
    out = {k: dict(v) for k, v in base.items()}
    for k, v in overrides.items():
        out.setdefault(k, {}).update(v)
    return out


_BASE_PLOT = {
    "PX": dict(bins=1000), "PY": dict(bins=1000), "PZ": dict(bins=1000),
    "P": dict(bins=1000), "E": dict(bins=1000),
    "ProbK": dict(xlim=(-0.1, 1.1), bins=100), "ProbPi": dict(xlim=(-0.1, 1.1), bins=100),
    "Charge": dict(xlim=(-1.1, 1.1), bins=10), "isMuon": dict(xlim=(-0.1, 1.1), bins=10),
    "IPChi2": dict(bins=100),
}
PLOT_SETTINGS = {
    "sim": _merge(_BASE_PLOT, ProbK=dict(bins=10), ProbPi=dict(bins=10)),
    "raw": _merge(_BASE_PLOT,
                  PX=dict(xlim=(-80_000, 80_000), log_y=True),
                  PY=dict(xlim=(-30_000, 30_000), log_y=True),
                  PZ=dict(xlim=(0, 1e6), log_y=True),
                  IPChi2=dict(log_y=True)),
    "presel": _merge(_BASE_PLOT,
                     PX=dict(xlim=(-20_000, 20_000), log_y=True),
                     PY=dict(xlim=(-10_000, 10_000), log_y=True),
                     PZ=dict(xlim=(0, 0.13e6), log_y=True)),
}


# =============================================================================
# 2. Data & physics
# =============================================================================
def load_tree(path, tree: str, columns=None) -> pd.DataFrame:
    """Read one TTree into a DataFrame (all branches if `columns` is None)."""
    with uproot.open(path) as f:
        return f[tree].arrays(columns, library="pd")


def load_data(cfg: Config, columns=None) -> pd.DataFrame:
    """Read and concatenate the real-data files, adding a `magnet` column."""
    frames = []
    for name in cfg.data_files:
        df = load_tree(cfg.data_dir / name, cfg.data_tree, columns)
        df["magnet"] = "down" if "down" in name.lower() else "up"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def particle_mass(df: pd.DataFrame, cfg: Config, particles=None):
    particles = cfg.particles if particles is None else particles
    for i in particles:
        mass = np.ones(len(df[f'H{i}_ProbK']))
        K_mask  = (df[f'H{i}_ProbK'] >= df[f'H{i}_ProbPi'])
        Pi_mask = (df[f'H{i}_ProbK'] <  df[f'H{i}_ProbPi'])
        if getattr(cfg, "mass_hypothesis", "pid") == "kaon":   # [Giacomo, 2026-10-06] see Config.mass_hypothesis
            K_mask, Pi_mask = np.ones(len(mass), bool), np.zeros(len(mass), bool)
        mass[K_mask]  = cfg.m_K
        mass[Pi_mask] = cfg.m_Pi
        df[f'H{i}_Mass'] = mass
    return df


def total_four_momentum(df: pd.DataFrame, cfg: Config, particles=None):
    """Sum of the four-momenta of `particles` (default: all) with the mass hypothesis cfg.m_K or cfg.m_Pi."""
    particles = cfg.particles if particles is None else particles
    df   = particle_mass(df, cfg, particles=particles)
    vecs = [vector.array({"px": df[f"H{i}_PX"].to_numpy(), "py": df[f"H{i}_PY"].to_numpy(),
                          "pz": df[f"H{i}_PZ"].to_numpy(), "M": df[f"H{i}_Mass"].to_numpy()}) for i in particles]
    return reduce(operator.add, vecs)


def invariant_mass(df: pd.DataFrame, cfg: Config, particles=None) -> np.ndarray:
    """Invariant mass of `particles`, e.g. invariant_mass(df, cfg, (1, 3)) for a Dalitz Plot."""
    return np.asarray(total_four_momentum(df, cfg, particles).mass)


def add_kinematics(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add H{i}_P (|p|) and H{i}_E (energy) for every product."""
    for i in cfg.particles:
        v = total_four_momentum(df, cfg, (i,))
        df[f"H{i}_P"], df[f"H{i}_E"] = np.asarray(v.p), np.asarray(v.energy)
    return df


def add_B_variables(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add B_Mass (invariant mass of all products) and B_Charge (sum of the charges)."""
    df["B_Mass"] = invariant_mass(df, cfg)
    df["B_Charge"] = sum(df[f"H{i}_Charge"].to_numpy() for i in cfg.particles)
    return df


def check_charges(df: pd.DataFrame, cfg: Config) -> None:
    for i in cfg.particles:
        values = np.unique(df[f"H{i}_Charge"])
        print(f"H{i}_Charge:", values)
        assert set(values) <= {-1, 1}, f"unexpected charge values for H{i}"


def hadron_preselection(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Boolean mask: all products are hadrons and not muons."""
    mask = np.ones(len(df), dtype=bool)
    for i in cfg.particles:
        mask &= ((df[f"H{i}_isMuon"] == 0)).to_numpy()
    return mask


def kaon_preselection(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Boolean mask: all products are kaons (high ProbK, low ProbPi) and not muons."""
    mask = hadron_preselection(df, cfg)
    for i in cfg.particles:
        mask &= ((df[f"H{i}_ProbPi"] < cfg.prob_pi_max) & (df[f"H{i}_ProbK"] > cfg.prob_k_min))
    return mask


def pion_preselection(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Boolean mask: all products are pions (high ProbPi, low ProbH) and not muons."""
    mask = hadron_preselection(df, cfg)
    for i in cfg.particles:
        mask &= ((df[f"H{i}_ProbK"] < cfg.prob_k_max) & (df[f"H{i}_ProbPi"] > cfg.prob_pi_min))
    return mask


def select_window(df: pd.DataFrame, window, col: str = "B_Mass") -> pd.DataFrame:
    lo, hi = window
    return df[(df[col] > lo) & (df[col] < hi)].reset_index(drop=True)


def split_by_charge(df: pd.DataFrame, col: str = "B_Mass") -> Dict[str, np.ndarray]:
    """{'pos': masses of B+, 'neg': masses of B-}."""
    return {side: df.loc[df["B_Charge"] == s.charge, col] for side, s in SIDES.items()}


def in_region(m: np.ndarray, region) -> np.ndarray:
    return (m >= region[0]) & (m <= region[1])


def select_sideband(m: np.ndarray, cfg: Config) -> np.ndarray:
    return m[~in_region(m, cfg.sig_region)]


def raw_asymmetry(n_pos: float, n_neg: float):
    """A = (N- - N+)/(N- + N+) and its binomial error sqrt((1-A^2)/N)."""
    n = n_pos + n_neg
    a = (n_neg - n_pos) / n
    return a, np.sqrt((1 - a**2) / n)


def print_event_summary(df: pd.DataFrame, cfg: Config) -> None:
    n = {side: int((df["B_Charge"] == s.charge).sum()) for side, s in SIDES.items()}
    a, err = raw_asymmetry(n["pos"], n["neg"])
    print(f"events in fit window {list(cfg.fit_window)} MeV: {len(df)}")
    print(f"  B+ : {n['pos']}   B- : {n['neg']}  -->  raw A_CP = {a:.6f} ± {err:.6f}")
    if "magnet" in df:
        print("  " + "   ".join(f"magnet {k}: {v}" for k, v in df["magnet"].value_counts().items()))


# =============================================================================
# 3. Plot helpers
# =============================================================================
def _hist_panel(ax, values, label, bins=1000, xlim=None, log_y=False, color=None):
    rng = xlim if (xlim is not None and None not in xlim) else None
    ax.hist(values, bins=bins, color=color, range=rng)
    ax.set_ylabel("counts")
    ax.set_xlabel(label)
    if log_y:
        ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.text(0.05, 0.90, f"Entries: {len(values)}\n Min: {np.min(values):.3g}\n Max: {np.max(values):.3g}",
            transform=ax.transAxes, fontsize=12, verticalalignment="top")


def plot_all_features(df, global_features=None, product_features=None, settings=None, n_products=3):
    """Histogram of event-level features and of per-product features (H1_*, H2_*, ...)."""
    settings = settings or {}
    figs = {}
    if global_features:
        fig, axes = plt.subplots(1, len(global_features), figsize=(6 * len(global_features), 3.5), squeeze=False)
        for ax, name in zip(axes[0], global_features):
            _hist_panel(ax, df[name], name, log_y=True)
        fig.tight_layout()
        figs["global"] = fig
    if product_features:
        fig, axes = plt.subplots(len(product_features), n_products,
                                 figsize=(6 * n_products, 3.5 * len(product_features)), squeeze=False)
        for j in range(n_products):
            for i, name in enumerate(product_features):
                feat = f"H{j + 1}_{name}"
                _hist_panel(axes[i, j], df[feat], feat, color=f"C{j}", **settings.get(name, {}))
        fig.tight_layout()
        figs["products"] = fig
    return figs


def plot_pid_2d(df_all: pd.DataFrame, df_sel: pd.DataFrame, cfg: Config):
    """ProbK vs ProbPi for every product, before and after the preselection."""
    fig, axes = plt.subplots(cfg.n_products, 2, figsize=(14, 5 * cfg.n_products), squeeze=False)
    for j in cfg.particles:
        for k, (df, title) in enumerate([(df_all, "All Events"), (df_sel, "Preselected")]):
            ax = axes[j - 1, k]
            *_, im = ax.hist2d(df[f"H{j}_ProbK"], df[f"H{j}_ProbPi"], bins=100,
                               range=[[-0.05, 1.05], [-0.05, 1.05]], norm=LogNorm())
            ax.set(xlabel=f"H{j}_ProbK", ylabel=f"H{j}_ProbPi", title=f"H{j} {title}")
            fig.colorbar(im, ax=ax).set_label("Number of Events / bin")
    fig.tight_layout()
    return fig


def plot_mass_by_charge(df, cfg: Config, bins=300, bin_range=None, logy=False, xlim=None, title=None, ax=None):
    """Overlay of the B mass distribution for B+ and B-."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    for s in SIDES.values():
        ax.hist(df.loc[df["B_Charge"] == s.charge, "B_Mass"], bins=bins, range=bin_range,
                histtype="step", linewidth=1.5, label=s.label)
    ax.axvline(cfg.m_B, ls="--", alpha=0.7, color="k", label=f"M(B) = {cfg.m_B} MeV")
    if logy:
        ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set(title=title or r"$B^\pm$ invariant mass", xlabel="Mass(KKK) [MeV]", ylabel="Counts / bin")
    ax.legend(fontsize=14)
    return fig, ax


def easy_dalitz_plot(cfg, mass=None, data=None, bins=100):
    """Produce all the Dalitz Plot. If data are passed all the possible combinations are plotted"""
    if mass is None:
        if data is None:
            raise ValueError("Provide either `mass` (dict of pair masses) or `data`.")
        else:
            mass = {f'{i}{j}': invariant_mass(data, cfg, particles=(i,j)) for i,j in list(itertools.combinations(cfg.particles, 2))}
    
    fig, axes = plt.subplots(1, len(mass.keys()), figsize=(7 * len(mass.keys()), 7))
    for i, (couple1, couple2) in enumerate(list(itertools.combinations(mass.keys(), 2))):
        ax = axes[i]
        *_, im = ax.hist2d(mass[couple1]**2, mass[couple2]**2, bins=bins, norm=LogNorm())
        # ax.scatter(mass[couple1]**2, mass[couple2]**2, alpha=0.1)
        ax.set_xlabel(rf"M($K_{couple1[0]}K_{couple1[1]}$)$^2$ [MeV$^2$]", fontsize=12)
        ax.set_ylabel(rf"M($K_{couple2[0]}K_{couple2[1]}$)$^2$ [MeV$^2$]", fontsize=12)
        ax.set_title(rf"Dalitz Plot: $K_{couple1[0]}K_{couple1[1]}$ - $K_{couple2[0]}K_{couple2[1]}$", fontsize=16)
        ax.set_xlim(np.min(mass[couple1]**2)-0.03*np.max(mass[couple1]**2), 1.03*np.max(mass[couple1]**2))
        ax.set_ylim(np.min(mass[couple2]**2)-0.03*np.max(mass[couple2]**2), 1.03    *np.max(mass[couple2]**2))
        fig.colorbar(im, ax=ax).set_label("Number of Events / bin", fontsize=12)
    fig.tight_layout()
    return fig


def remove_charm_resonances(cfg, data, remove, couple_inv_mass=None):
    if couple_inv_mass is None:
            if data is None:
                raise ValueError("Provide either `mass` (dict of pair masses) or `data`.")
            else:
                # [Giacomo, 2026-10-06] only the two opposite-sign pairs (H1H2, H1H3: H1 is always the kaon with charge
                # opposite to the B). Previously all three pairs were used, including H2H3, which has equal charges and cannot
                # come from D0, J/psi or chi_c0: vetoing it rejected random events (~1600 in the full version).
                # Check in the window [5100, 5600]: 18744 (D0 only) and 16367 (D0 + J/psi + chi_c0) candidates remain.
                couple_inv_mass = {f'1{j}': invariant_mass(data, cfg, particles=(1, j)) for j in (2, 3)}
    return data[get_charm_veto_mask(cfg, couple_inv_mass, remove)]

def get_charm_veto_mask(cfg, couple_inv_mass, remove):
    """
    Boolean mask, True for the events to KEEP and False for the vetoed ones.

    Parameters:
    - couple_inv_mass: dictionary with the invariant masses (MeV) of the pairs, e.g. {'12': m12, '13': m13}
    - remove: 'a' or 'D' -> D0 veto only
            'b' or 'all' -> D0 + J/psi + chi_c0 veto
            '[...]' -> veto on the selected particles
    """
    
    if remove in ['a', 'D']:
        active_res = ['D']
    elif remove in ['b', 'all']:
        active_res = ['D', 'J', 'chi']
    elif isinstance(remove, (list, tuple, set)):
        active_res = remove
    else:
        raise ValueError(f"remove_resonances not valid.")

    first_key = list(couple_inv_mass.keys())[0]
    reject_mask = np.zeros(len(couple_inv_mass[first_key]), dtype=bool)

    # Reject the event if AT LEAST ONE of the two pairs falls in the window |m - m0| < half-width
    for pair_key, m_array in couple_inv_mass.items():
        for res in active_res:
            m0, half_width = cfg.res_windows[res]
            reject_mask |= (np.abs(m_array - m0) < half_width)

    return ~reject_mask


def dalitz_plot(cfg, mass=None, data=None, range=None, bins=100, add_title='', remove_resonances=None):
    """Produce all the Dalitz Plot with marginal 1D distributions."""
    if mass is None:
        if data is None:
            raise ValueError("Provide either `mass` (dict of pair masses) or `data`.")
        else:
            mass = {
                f'{i}{j}': invariant_mass(data, cfg, particles=(i, j))
                for i, j in itertools.combinations(cfg.particles, 2)
            }
    
    # resonances veto
    if remove_resonances:
        keep_mask = get_charm_veto_mask(cfg, mass, remove_resonances)
        mass = {k: v[keep_mask] for k, v in mass.items()}
    
    combinations = list(itertools.combinations(mass.keys(), 2))
    n_plots = len(combinations)
    
    fig = plt.figure(figsize=(8 * n_plots, 8))
    outer_grid = GridSpec(1, n_plots, figure=fig, wspace=0.3)

    for i, (couple1, couple2) in enumerate(combinations):
        gs = outer_grid[i].subgridspec(
            2, 2, 
            width_ratios=[4, 1], 
            height_ratios=[1, 4],
            wspace=0.05, 
            hspace=0.05
        )
        
        ax_main = fig.add_subplot(gs[1, 0])
        ax_left = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_top  = fig.add_subplot(gs[0, 0], sharex=ax_main)

        x_vals = mass[couple1]**2
        y_vals = mass[couple2]**2

        # Dalitz Plot
        h, x_edges, y_edges, im = ax_main.hist2d(x_vals, y_vals, range=range, bins=bins, norm=LogNorm())
        # ax_main.scatter(x_vals, y_vals, alpha=0.2, s=1, marker='.', color='steelblue')
        ax_main.set_xlabel(rf"$M^2(K_{{{couple1[0]}}}K_{{{couple1[1]}}})$ [MeV$^2$]", fontsize=13)
        ax_main.set_ylabel(rf"$M^2(K_{{{couple2[0]}}}K_{{{couple2[1]}}})$ [MeV$^2$]", fontsize=13)
        
        # Histogram (X axes)
        ax_top.hist(x_vals, bins=bins, color='steelblue', histtype='stepfilled', alpha=0.7)
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.set_ylabel("Events", fontsize=10)
        ax_top.set_title(rf"Dalitz Plot: $K_{{{couple1[0]}}}K_{{{couple1[1]}}}$ - $K_{{{couple2[0]}}}K_{{{couple2[1]}}}$" + add_title, fontsize=15, pad=12)

        # Histogram (Y axes)
        ax_left.hist(y_vals, bins=bins, color='steelblue', histtype='stepfilled', alpha=0.7, orientation='horizontal')
        ax_left.tick_params(axis="y", labelleft=False)
        ax_left.set_xlabel("Events", fontsize=10)

        x_min = range[0][0] if range is not None else np.min(x_vals) - 0.03 * np.max(x_vals)
        x_max = range[0][1] if range is not None else 1.03 * np.max(x_vals)
        y_min = range[1][0] if range is not None else np.min(y_vals) - 0.03 * np.max(y_vals)
        y_max = range[1][1] if range is not None else 1.03 * np.max(y_vals)
        ax_main.set_xlim(x_min, x_max)
        ax_main.set_ylim(y_min, y_max)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax_left, pad=0.1)
        cbar.set_label(rf"Number of Events / ({x_edges[1]-x_edges[0]:.1e}x{y_edges[1]-y_edges[0]:.1e} MeV$^2$)", fontsize=12)

    return fig, ax_main, ax_top, ax_left, h, x_edges, y_edges


# =============================================================================
# 4. Models
# =============================================================================
def _expo_integral(lam: float, intervals, ref: float):
    """Integral of exp(-lam (m - ref)) over a list of (a, b) intervals (numerically stable)."""
    return sum(np.exp(-lam * (a - ref)) * (-np.expm1(-lam * (b - a))) / lam for a, b in intervals)


def expo_pdf(m, lam, intervals):
    """Exponential normalised over the union of `intervals`. Shifted by the start of the first
    interval to avoid exp(-lam*5000) underflow."""
    ref = intervals[0][0]
    return np.exp(-lam * (np.asarray(m) - ref)) / _expo_integral(lam, intervals, ref)


def gauss_pdf(m, mu, sigma, window):
    """Gaussian truncated and normalised to `window`."""
    lo, hi = window
    return norm.pdf(m, mu, sigma) / (norm.cdf(hi, mu, sigma) - norm.cdf(lo, mu, sigma))


def cb_pdf(m, mu, sigma, beta, m_cb, window):
    """[Giacomo, 2026-10-06] Crystal Ball (low-mass tail) truncated and normalised on `window`."""
    lo, hi = window
    z = crystalball.cdf((hi - mu) / sigma, beta, m_cb) - crystalball.cdf((lo - mu) / sigma, beta, m_cb)
    return crystalball.pdf((np.asarray(m) - mu) / sigma, beta, m_cb) / sigma / z


def _bkg_weights(n_exp: int, f_bkgs):
    return [1.0] if n_exp == 1 else [f_bkgs, 1.0 - f_bkgs]


def background_pdf(m, lams, f_bkgs, intervals):
    """Mixture of 1 or 2 exponentials (weight f_bkgs on the first one)."""
    w = _bkg_weights(len(lams), f_bkgs)
    return sum(wk * expo_pdf(m, lam, intervals) for wk, lam in zip(w, lams))


def background_fraction(lams, f_bkgs, region, reference) -> float:
    """Fraction of the background pdf (normalised on `reference` intervals) that falls in `region`."""
    ref = reference[0][0]
    w = _bkg_weights(len(lams), f_bkgs)
    return sum(wk * _expo_integral(lam, region, ref) / _expo_integral(lam, reference, ref)
               for wk, lam in zip(w, lams))


def _with_signature(func: Callable, names: Sequence[str]) -> Callable:
    """Give `func(x, *args)` an explicit signature so that iminuit can read the parameter names."""
    params = [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in ["m", *names]]
    func.__signature__ = inspect.Signature(params)
    return func


def _bkg_keys(n_exp: int) -> list:
    return ["N_bkg"] + [f"lam{k}" for k in range(1, n_exp + 1)] + (["f_bkgs"] if n_exp == 2 else [])


def make_model(cfg: Config, side: str | None = None, n_exp: int | None = None) -> Callable:
    """
    Extended signal+background model ``model(m, *params) -> (N_expected, density)``.

    side=None  : single sample;  params = N_sig, mu, sigma, N_bkg, lam1[, lam2, f_bkgs]
    side='pos' : joint-fit B+;   params = N_sig, A_CP, mu, sigma, N_bkg_pos, lam1_pos[, lam2_pos, f_bkgs_pos]
    side='neg' : joint-fit B-;   same with the *_neg names (shared params keep the same name -> iminuit merges them)
    """
    n_exp = cfg.n_exp if n_exp is None else n_exp
    window, intervals = cfg.fit_window, cfg.window_intervals
    cb = cfg.signal == "cb"
    keys = ["N_sig"] + (["A_CP"] if side else []) + ["mu", "sigma"] + (["beta", "m_cb"] if cb else []) + _bkg_keys(n_exp)
    suffix = f"_{side}" if side else ""
    # [Giacomo, 2026-10-06] which parameters carry the _pos/_neg suffix (separate for each charge) depends on the
    # Config options: in the nominal model only the background (yield and shape). Parameters with the same name in
    # the two models are shared by iminuit in the joint fit.
    def _per_side(k):
        if k == "N_bkg":
            return cfg.bkg_yield != "symmetric"
        if k in _bkg_keys(n_exp):
            return cfg.bkg_shape != "shared"
        if k in ("mu", "sigma"):
            return cfg.signal_per_charge
        return False
    names = [k + suffix if (side and _per_side(k)) else k for k in keys]
    sign = SIDES[side].asym_sign if side else 0

    def model(m, *args):
        p = dict(zip(keys, args))
        n_sig = p["N_sig"] * (1 + sign * p["A_CP"]) / 2 if side else p["N_sig"]
        lams = [p[f"lam{k}"] for k in range(1, n_exp + 1)]
        bkg = background_pdf(m, lams, p.get("f_bkgs"), intervals)
        sig = (cb_pdf(m, p["mu"], p["sigma"], p["beta"], p["m_cb"], window) if cb
               else gauss_pdf(m, p["mu"], p["sigma"], window))
        dens = n_sig * sig + p["N_bkg"] * bkg
        return n_sig + p["N_bkg"], dens

    return _with_signature(model, names)


def make_sideband_pdf(cfg: Config, n_exp: int | None = None) -> Callable:
    """Normalised background pdf on the sideband: pdf(m, lam1[, lam2, f_bkgs])."""
    n_exp = cfg.n_exp if n_exp is None else n_exp
    intervals = cfg.sideband_intervals

    def pdf(m, *args):
        return background_pdf(m, args[:n_exp], args[n_exp] if n_exp == 2 else None, intervals)

    return _with_signature(pdf, _bkg_keys(n_exp)[1:])


def _model_curve(model: Callable, values: Mapping) -> Callable:
    """Bind the fitted `values` to a model: returns x -> extended density."""
    args = [values[n] for n in describe(model)[1:]]
    return lambda x: model(x, *args)[1]


def fitted_curve(m: Minuit, cfg: Config, side: str | None = None, n_exp: int | None = None) -> Callable:
    """x -> fitted extended density (counts/MeV) of a fit made with fit_single (side=None) or fit_joint (side='pos'/'neg')."""
    return _model_curve(make_model(cfg, side, n_exp), m.values)


# =============================================================================
# 5. Fits
# =============================================================================
def param_limit(name: str, cfg: Config):
    """Limits of a parameter from its name (strips _pos/_neg and trailing digits)."""
    base = re.sub(r"\d+$", "", re.sub(r"_(pos|neg)$", "", name))
    if base == "mu":
        return cfg.limits.get("mu", cfg.sig_region)
    return cfg.limits.get(base)


def make_minuit(cost, start: Mapping, cfg: Config, fixed: Mapping | None = None) -> Minuit:
    """Minuit with the standard limits; `fixed` = {param: value} are fixed at that value."""
    m = Minuit(cost, **{k: start[k] for k in describe(cost)})
    for name in m.parameters:
        lim = param_limit(name, cfg)
        if lim is not None:
            m.limits[name] = lim
    for name, val in (fixed or {}).items():
        m.values[name] = val
        m.fixed[name] = True
    return m


def run_fit(m: Minuit, hesse: bool = True, minos=False) -> Minuit:
    """migrad (+ hesse) (+ minos: True for all parameters, or a list of names)."""
    m.migrad()
    if hesse:
        m.hesse()
    if minos:
        if m.valid:
            m.minos(*([] if minos is True else minos))
        else:
            warnings.warn("migrad did not converge: MINOS skipped")
    return m


def print_params(m: Minuit) -> None:
    for name, val, err in zip(m.parameters, m.values, m.errors):
        print(f"{name:10s} -> {val:.6f} ± {err:.6f}")


# ---- sideband -----------------------------------------------------------------
def fit_sideband(m_sb: np.ndarray, cfg: Config, n_exp: int | None = None) -> Minuit:
    """Unbinned fit of the background shape on the sideband only."""
    n_exp = cfg.n_exp if n_exp is None else n_exp
    start = dict(lam1=0.02, lam2=0.001, f_bkgs=0.5)
    cost = UnbinnedNLL(m_sb, make_sideband_pdf(cfg, n_exp))
    return run_fit(make_minuit(cost, start, cfg), hesse=False)


def _bkg_args(values: Mapping, n_exp: int):
    return [values[f"lam{k}"] for k in range(1, n_exp + 1)], (values["f_bkgs"] if n_exp == 2 else None)


def expected_background_in_signal(sb_fit: Minuit, n_sb: int, cfg: Config, n_exp: int | None = None) -> float:
    """Extrapolate the sideband fit into the signal region: b = N_sb * (bkg fraction in SR / in SB)."""
    n_exp = cfg.n_exp if n_exp is None else n_exp
    lams, f = _bkg_args(sb_fit.values, n_exp)
    ratio = (background_fraction(lams, f, [cfg.sig_region], cfg.window_intervals)
             / background_fraction(lams, f, cfg.sideband_intervals, cfg.window_intervals))
    return n_sb * ratio


def background_start(m: np.ndarray, cfg: Config, n_exp: int, sb_fit: Minuit | None = None) -> dict:
    """Starting values {N_bkg, lam1, [lam2, f_bkgs]} for one sample, from its sideband fit."""
    m_sb = select_sideband(m, cfg)
    sb_fit = sb_fit or fit_sideband(m_sb, cfg, n_exp)
    lams, f = _bkg_args(sb_fit.values, n_exp)
    frac_sb = background_fraction(lams, f, cfg.sideband_intervals, cfg.window_intervals)
    start = {"N_bkg": len(m_sb) / frac_sb}
    start.update({f"lam{k}": lam for k, lam in enumerate(lams, 1)})
    if n_exp == 2:
        start["f_bkgs"] = f
    return start


# ---- single-sample fit ----------------------------------------------------------
def fit_single(m_data: np.ndarray, cfg: Config, n_exp: int | None = None, start: Mapping | None = None,
               minos=False) -> Minuit:
    """Extended unbinned fit of ONE sample (e.g. B+ only): N_sig, N_bkg, mu, sigma, lam1[, lam2, f_bkgs]."""
    n_exp = cfg.n_exp if n_exp is None else n_exp
    if start is None:
        start = background_start(m_data, cfg, n_exp)
        start.update(N_sig=len(m_data) - start["N_bkg"], mu=cfg.m_B, sigma=20.0)
    cost = ExtendedUnbinnedNLL(m_data, make_model(cfg, None, n_exp))
    return run_fit(make_minuit(cost, start, cfg), minos=minos)


# ---- joint fit ------------------------------------------------------------------
def build_joint_cost(samples: Mapping[str, np.ndarray], cfg: Config, n_exp: int | None = None):
    """Sum of the extended NLLs of all sides; parameters with the same name are shared."""
    return reduce(operator.add, (ExtendedUnbinnedNLL(samples[side], make_model(cfg, side, n_exp))
                                 for side in SIDES))


def initial_values(samples: Mapping[str, np.ndarray], cfg: Config, n_exp: int | None = None) -> dict:
    """Starting values for the joint fit, built from the sideband fits of each sample."""
    n_exp = cfg.n_exp if n_exp is None else n_exp
    start, n_sig = {"A_CP": 0.0, "mu": cfg.m_B, "sigma": 20.0}, 0.0
    per_side = []
    for side, m in samples.items():
        bs = background_start(m, cfg, n_exp)
        n_sig += len(m) - bs["N_bkg"]
        start.update({f"{k}_{side}": v for k, v in bs.items()})
        per_side.append(bs)
    start["N_sig"] = n_sig
    # [Giacomo, 2026-10-06] starting values also for the shared parameters of the variants (mean of the two charges),
    # for mu/sigma separate per charge and for the Crystal Ball tails; names not used by the model are ignored
    for k in per_side[0]:
        start.setdefault(k, float(np.mean([bs[k] for bs in per_side])))
    for side in samples:
        start.setdefault(f"mu_{side}", cfg.m_B); start.setdefault(f"sigma_{side}", 20.0)
    start.setdefault("beta", 1.8); start.setdefault("m_cb", 10.0)
    return start


def fit_joint(samples: Mapping[str, np.ndarray], cfg: Config, n_exp: int | None = None,
              start: Mapping | None = None, fixed: Mapping | None = None,
              hesse: bool = True, minos=False) -> Minuit:
    """Simultaneous fit of B+ and B- with shared N_sig, A_CP, mu, sigma."""
    n_exp = cfg.n_exp if n_exp is None else n_exp
    start = initial_values(samples, cfg, n_exp) if start is None else start
    m = make_minuit(build_joint_cost(samples, cfg, n_exp), start, cfg, fixed)
    return run_fit(m, hesse=hesse, minos=minos)


def signal_yields(m: Minuit) -> Dict[str, float]:
    """Signal yield of each side from the joint-fit parameters."""
    n, a = m.values["N_sig"], m.values["A_CP"]
    return {side: n * (1 + s.asym_sign * a) / 2 for side, s in SIDES.items()}


def print_asymmetry(m: Minuit) -> None:
    a, err = m.values["A_CP"], m.errors["A_CP"]
    print(f"A_CP = {a:.4f} ± {err:.4f}   (relative error {abs(err / a) * 100:.1f}%, "
          f"Hesse significance {abs(a / err):.2f} sigma)")


def asymmetry_from_yields(fits: Mapping[str, Minuit]):
    """A_CP and its error from two independent fits (B+ and B-), by error propagation."""
    p, n = (fits[s].values["N_sig"] for s in ("pos", "neg"))
    dp, dn = (fits[s].errors["N_sig"] for s in ("pos", "neg"))
    a = (n - p) / (n + p)
    err = 2 * np.sqrt((p * dn) ** 2 + (n * dp) ** 2) / (n + p) ** 2
    return a, err


def wilks_test(m: Minuit, null: Mapping[str, float]):
    """
    Likelihood-ratio test of H0: params in `null` fixed to the given values (e.g. {'A_CP': 0}).
    Returns (t, p_value, refitted_Minuit) with t = -2 ln(L_H0 / L_best) ~ chi2(len(null)).
    """
    m0 = copy.deepcopy(m)
    for name, val in null.items():
        m0.values[name] = val
        m0.fixed[name] = True
    run_fit(m0, hesse=False)
    t = max(m0.fval - m.fval, 0.0)  # iminuit costs already return -2 ln L
    return t, chi2_dist.sf(t, len(null)), m0


# ---- plots of fits ------------------------------------------------------------------
def plot_mass_fit(samples, curves, n_signal, cfg: Config, bins=200, xlim=(5100, 5600), title=None):
    """Data (points) + fitted extended density (line) for every side. `curves[side](x)` is counts/MeV."""
    fig, axes = plt.subplots(1, len(SIDES), figsize=(7 * len(SIDES), 7), squeeze=False)
    x = np.linspace(*cfg.fit_window, 10000)
    hists = {side: np.histogram(samples[side], bins=bins, range=cfg.fit_window) for side in SIDES}
    ymax = 1.2 * max(c.max() for c, _ in hists.values())
    for ax, (side, s) in zip(axes[0], SIDES.items()):
        counts, edges = hists[side]
        width, centres = edges[1] - edges[0], 0.5 * (edges[:-1] + edges[1:])
        ax.hist(samples[side], bins=bins, range=cfg.fit_window, histtype="step", linestyle="--", alpha=0.5, color="k")
        ax.errorbar(centres, counts, yerr=np.sqrt(counts), fmt=".", color="k", label=f"Data {s.label}")
        ax.plot(x, width * curves[side](x), color=s.color, label=f"Fit {s.label}")
        ax.text(0.05, 0.95, f"Signal {s.label} events: {n_signal[side]:.1f}", transform=ax.transAxes,
                verticalalignment="top", fontsize=12)
        ax.set(xlim=xlim, ylim=(0, ymax), title=f"{s.label} invariant mass",
               xlabel="Mass(KKK) [MeV]", ylabel=f"Counts / {width:.1f} MeV")
        ax.grid(alpha=0.3)
        ax.legend()
    if title:
        fig.suptitle(title, fontsize=18)
    return fig


def plot_joint_fit(m: Minuit, samples, cfg: Config, n_exp: int | None = None, **kw):
    n_exp = cfg.n_exp if n_exp is None else n_exp
    curves = {side: _model_curve(make_model(cfg, side, n_exp), m.values) for side in SIDES}
    kw.setdefault("title", rf"Joint fit of $B^+$ and $B^-$ ({n_exp} expon)")
    return plot_mass_fit(samples, curves, signal_yields(m), cfg, **kw)


def plot_separate_fits(fits: Mapping[str, Minuit], samples, cfg: Config, n_exp: int | None = None, **kw):
    n_exp = cfg.n_exp if n_exp is None else n_exp
    model = make_model(cfg, None, n_exp)
    curves = {side: _model_curve(model, fits[side].values) for side in SIDES}
    n_signal = {side: fits[side].values["N_sig"] for side in SIDES}
    kw.setdefault("title", rf"Separate fits of $B^+$ and $B^-$ ({n_exp} expon)")
    return plot_mass_fit(samples, curves, n_signal, cfg, **kw)


def plot_sideband(sb_samples, sb_fits, cfg: Config, n_exp: int | None = None, bins=100):
    """Sideband data (density histogram) with the fitted background pdf on top."""
    n_exp = cfg.n_exp if n_exp is None else n_exp
    pdf, x = make_sideband_pdf(cfg, n_exp), np.linspace(*cfg.fit_window, 1000)
    fig, ax = plt.subplots()
    for side, s in SIDES.items():
        ax.hist(sb_samples[side], bins=bins, range=cfg.fit_window, density=True, alpha=0.5,
                color=s.color, label=f"Data {s.label}")
        ax.plot(x, pdf(x, *sb_fits[side].values), color=s.color, label=f"Fit {s.label}")
    ax.set(xlabel="Mass(KKK) [MeV]", ylabel="Density")
    ax.legend()
    return fig


# ---- systematics scans ------------------------------------------------------------------
# def sideband_subtracted_asymmetry(samples, cfg, n_exp=None):
#     """A_CP from sideband-subtracted counting in the signal region (this one DOES depend on cfg.sig_region)."""
#     n_exp = cfg.n_exp if n_exp is None else n_exp
#     n_sig, n_obs = {}, {}
#     for side, m in samples.items():
#         sb = select_sideband(m, cfg)
#         b = expected_background_in_signal(fit_sideband(sb, cfg, n_exp), len(sb), cfg, n_exp)
#         n_obs[side] = int(in_region(m, cfg.sig_region).sum())
#         n_sig[side] = n_obs[side] - b
#     p, n = n_sig["pos"], n_sig["neg"]
#     a = (n - p) / (n + p)
#     err = 2 * np.sqrt(p**2 * n_obs["neg"] + n**2 * n_obs["pos"]) / (n + p) ** 2   # Poisson on N_obs only
#     return a, err

def scan_config(all_data, cfg, variations, n_exp=2, minos=False):
    """
    One joint fit per entry of `variations` = list of Config overrides, e.g.
    [{'fit_window': (5100, 6000)}, {'sig_region': (5230, 5330)}, ...].
    `data` must contain B_Mass and B_Charge over the FULL mass range (it is re-cut for every window).
    """
    rows = []
    for var in tqdm(variations, desc="scan"):
        # [Giacomo, 2026-10-06] a variant can have a label ("label") and change n_exp too
        var = dict(var); label = var.pop("label", None)
        n_exp_v = var.get("n_exp", n_exp)
        cfg_v = replace(cfg, **var)
        # [Giacomo, 2026-10-06] this used to be `('prob_pi_max' or 'prob_k_min') in var.keys()`, which only checks
        # 'prob_pi_max' (`or` between two strings returns the first): a variant with only 'prob_k_min'
        # would not have re-applied the preselection. Moreover `all_data` must be a sample NOT preselected
        # (or preselected with cuts looser than the whole grid), otherwise cuts looser than the
        # nominal ones cannot recover events: see the `data_loose` cell in the notebook.
        # [Giacomo, 2026-10-06] the preselection of the variant is always re-applied: with a sample already
        # preselected with the nominal cuts nothing changes; with `data_loose` (looser cuts) it gives the right
        # sample for every variant, so all variants can use the same input.
        data = all_data[kaon_preselection(all_data, cfg_v)]
        (left, right), (slo, shi) = cfg_v.fit_window, cfg_v.sig_region
        row = dict(label=label, left=left, right=right, sig_lo=slo, sig_hi=shi)
        try:
            if not (left <= slo < shi <= right):
                raise ValueError("signal region not inside the fit window")
            samples = split_by_charge(select_window(data, cfg_v.fit_window))
            if sum(len(select_sideband(m, cfg_v)) for m in samples.values()) == 0:
                raise ValueError("empty sideband")
            m = fit_joint(samples, cfg_v, n_exp=n_exp_v, minos=minos)
            # a_cnt, a_cnt_err = sideband_subtracted_asymmetry(samples, cfg_v, n_exp)
            row.update(valid=m.valid, accurate=m.accurate, n_events=sum(map(len, samples.values())),
                       A_CP=m.values["A_CP"], A_CP_err=m.errors["A_CP"], N_sig=m.values["N_sig"],
                       # [Giacomo, 2026-10-06] with signal_per_charge there are mu_pos/mu_neg: their mean is recorded
                       mu=(m.values["mu"] if "mu" in m.parameters else 0.5 * (m.values["mu_pos"] + m.values["mu_neg"])),
                       sigma=(m.values["sigma"] if "sigma" in m.parameters else 0.5 * (m.values["sigma_pos"] + m.values["sigma_neg"])),
                       gof=gof_joint(m, samples, cfg_v, n_exp_v), chi2=gof_joint(m, samples, cfg_v, n_exp_v).chi2,
                       fval=m.fval,
                       prob_k_min=cfg_v.prob_k_min, prob_pi_max=cfg_v.prob_pi_max,
                       # A_count=a_cnt, A_count_err=a_cnt_err
                       )
        except Exception as e:
            row.update(valid=False, error=f"{type(e).__name__}: {e}")
        rows.append(row)
    return pd.DataFrame(rows)

def scan_fit_window(data, cfg, lefts, rights, **kw):
    return scan_config(data, cfg, [dict(fit_window=(l, r))   for l,  r  in itertools.product(lefts, rights)], **kw)

def scan_signal_region(data, cfg, los, his, **kw):
    return scan_config(data, cfg, [dict(sig_region=(lo, hi)) for lo, hi in itertools.product(los, his)], **kw)

def scan_preselection_cuts(all_data, cfg, k_min, pi_max, **kw):
    return scan_config(all_data, cfg, [dict(prob_k_min=k, prob_pi_max=pi) for k, pi in itertools.product(k_min, pi_max)], **kw)


# =============================================================================
# 6. Diagnostics
# =============================================================================
@dataclass
class GoF:
    chi2: float
    ndof: int
    chi2_red: float
    p_value: float

    def __str__(self):
        return (f"Chi2:         {self.chi2:.2f}\nNDOF:         {self.ndof}\n"
                f"Chi2 / NDOF:  {self.chi2_red:.3f}\np-value:      {self.p_value:.4f}")


def _gof(chi2_val: float, n_bins: int, n_free: int) -> GoF:
    ndof = n_bins - n_free
    if ndof <= 0:
        return GoF(chi2_val, ndof, np.nan, np.nan)
    return GoF(chi2_val, ndof, chi2_val / ndof, chi2_dist.sf(chi2_val, ndof))


def _binned_chi2(data, curve, window, bins, oversample=5):
    """Pearson chi2 of a histogram vs. the extended density `curve` (bin average over a sub-grid)."""
    counts, edges = np.histogram(data, bins=bins, range=window)
    width = np.diff(edges)
    grid = edges[:-1, None] + (np.arange(oversample) + 0.5) / oversample * width[:, None]
    expected = curve(grid.ravel()).reshape(grid.shape).mean(axis=1) * width
    ok = expected > 0
    return float(np.sum((counts[ok] - expected[ok]) ** 2 / expected[ok])), int(ok.sum())


def chi2_gof(data, curve, window, bins, n_free) -> GoF:
    """Binned chi2 goodness of fit of one sample. `n_free` = number of fitted parameters."""
    return _gof(*_binned_chi2(data, curve, window, bins), n_free)


def gof_joint(m: Minuit, samples, cfg: Config, n_exp: int | None = None, bins: int = 200) -> GoF:
    """Combined chi2 of the two samples; ndof = (valid bins) - (free parameters of the joint fit)."""
    n_exp = cfg.n_exp if n_exp is None else n_exp
    parts = [_binned_chi2(samples[side], _model_curve(make_model(cfg, side, n_exp), m.values),
                          cfg.fit_window, bins) for side in SIDES]
    return _gof(sum(p[0] for p in parts), sum(p[1] for p in parts), m.nfit)


def scan_gof_bins(data, curve, window, n_free, bin_values):
    """chi2/ndof and p-value as a function of the number of bins."""
    res = [chi2_gof(data, curve, window, int(b), n_free) for b in bin_values]
    return np.array([r.chi2_red for r in res]), np.array([r.p_value for r in res])


@dataclass
class CoverageResult:
    t_values: np.ndarray
    estimates: Dict[str, np.ndarray]
    err_estimates: Dict[str, np.ndarray]
    coverage: float
    t_crit: float
    ndof: int
    n_failed: int


def run_coverage_study(data: pd.DataFrame, cfg: Config, best: Minuit, n_exp: int | None = None,
                       pois: Sequence[str] = ("A_CP", "mu"), n_toys: int = 30,
                       truth: Mapping[str, float] | None = None, seed: int | None = None,
                       fix_in_free_fit: Sequence[str] = ()) -> CoverageResult:
    """
    Bootstrap check of Wilks' theorem for the parameters of interest `pois`.

    For every resampled dataset:  t = -2 ln[L(pois = truth) / L(best)],  nuisance parameters profiled.
    `truth` defaults to the best-fit values of the original data. Coverage = fraction of toys with
    t <= chi2(len(pois)).ppf(0.6827).  `fix_in_free_fit` lists parameters to keep fixed in BOTH fits
    (e.g. ['sigma'] for a look-elsewhere-style study).
    """
    n_exp = cfg.n_exp if n_exp is None else n_exp
    rng = np.random.default_rng(seed)
    start = dict(zip(best.parameters, best.values))
    truth = dict(truth) if truth is not None else {p: start[p] for p in pois}
    keep_fixed = {p: start[p] for p in fix_in_free_fit}
    t_crit, ndof = chi2_dist(len(pois)).ppf(0.6827), len(pois)

    t_values, estimates, err_estimates, failed = [], {p: [] for p in pois}, {p: [] for p in pois}, 0
    for _ in tqdm(range(n_toys)):
        boot = data.iloc[rng.integers(0, len(data), len(data))]
        samples = split_by_charge(boot, col='B_Mass')
        free = fit_joint(samples, cfg, n_exp, start=start, fixed=keep_fixed, hesse=False)
        if not free.valid:
            failed += 1
            continue
        null = fit_joint(samples, cfg, n_exp, start={**start, **truth}, fixed={**keep_fixed, **truth}, hesse=False)
        if not null.valid:
            continue
        t_values.append(max(null.fval - free.fval, 0.0))
        for p in pois:
            estimates[p].append(free.values[p])
            err_estimates[p].append(free.errors[p])

    t_values = np.array(t_values)
    return CoverageResult(t_values, {p: np.array(v) for p, v in estimates.items()}, {p: np.array(v) for p, v in err_estimates.items()},
                          float(np.mean(t_values <= t_crit)), t_crit, ndof, failed)


def plot_coverage(res: CoverageResult, best: Minuit, bins: int = 40):
    """Toy distribution of t against the Wilks chi2 pdf, plus the toy estimates of the POIs."""
    n = 1 + len(res.estimates)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    ax = axes[0, 0]
    x = np.linspace(0, max(8, res.t_values.max()), 300)
    ax.hist(res.t_values, bins=bins*2, density=True, alpha=0.5, label="Toys")
    ax.plot(x, chi2_dist(res.ndof).pdf(x), "k--", lw=2, label=rf"Wilks: $\chi^2$(df={res.ndof})")
    ax.axvline(res.t_crit, color="red", ls=":", lw=2, label=f"68.3% limit ({res.t_crit:.2f})")
    ax.set(title=f"Coverage = {res.coverage * 100:.1f}%  ({len(res.t_values)} toys, {res.n_failed} failed)",
           xlabel=r"$t = -2\ln(L_0/L_{best})$", ylabel="Probability density")
    ax.legend()
    for ax, (name, vals), err_vals in zip(axes[0, 1:], res.estimates.items(), res.err_estimates.values()):
        pulls = (vals - best.values[name]) / err_vals
        ax.hist(pulls, bins=bins, density=True, alpha=0.5, label=f'{len(vals)} toys')
        x = np.linspace(np.min(pulls), np.max(pulls), 1000)
        ax.plot(x, norm.pdf(x), color='red', label='Standard normal')
        ax.set(xlabel=r"Pull $(\hat{A} - A_{true})/\sigma(\hat{A})$", ylabel="Density")
        ax.text(0.03, 0.95, rf"mean = {np.mean(pulls):.2f} $\pm$ {(np.std(pulls)/np.sqrt(len(pulls))):.2f}"+ f"\nsd = {np.std(pulls):.2f}",
            transform=ax.transAxes, fontsize=11, verticalalignment="top")
        ax.legend()
    fig.tight_layout()
    return fig


# ---- [Giacomo, 2026-10-06] combination of the variants of the systematic study ---------------------------------
# Rule common to the Bayesian analysis:
#  1) variants that change the sample (window, preselection): the shift counts only if it exceeds the expected
#     statistical fluctuation between the two samples (Barlow criterion), sigma_B = sqrt(|sigma_var^2 - sigma_nom^2|);
#  2) variants on the same sample rejected by the data (binned chi2 worse than the nominal by more than 25) are excluded;
#  3) for each group the largest absolute shift; groups added in quadrature.
def combine_systematics(nominal: Mapping, rows: Sequence[Mapping], groups: Mapping[str, Sequence[str]],
                        sample_changing: Sequence[str], dchi2_max: float = 25.0) -> dict:
    """`nominal` and every element of `rows` have A_CP, A_CP_err, chi2, label; `groups` = {group: [label, ...]}."""
    a0, s0, n0 = nominal["A_CP"], nominal["A_CP_err"], nominal.get("n_events")
    per_var = {}
    for r in rows:
        lab = r["label"]
        if not r.get("valid", True) or not np.isfinite(r.get("A_CP", np.nan)):
            per_var[lab] = dict(shift=None, sigma_barlow=None, rejected_by_data=False, counts=False, fit_failed=True)
            continue
        d = r["A_CP"] - a0
        # if the fit error is not reliable (accurate=False, e.g. a parameter at its limit) the statistical sigma of the
        # variant is estimated from the number of events, sigma_var ~ sigma_nom * sqrt(N_nom / N_var)
        err_from_n = (r.get("accurate") is False) and n0 and r.get("n_events")
        s_var = s0 * np.sqrt(n0 / r["n_events"]) if err_from_n else r["A_CP_err"]
        sb = float(np.sqrt(abs(s_var ** 2 - s0 ** 2))) if lab in sample_changing else None
        rejected = (lab not in sample_changing) and (r["chi2"] - nominal["chi2"] > dchi2_max)
        counts = (not rejected) and (sb is None or abs(d) > sb)
        per_var[lab] = dict(shift=float(d), sigma_barlow=sb, rejected_by_data=bool(rejected), counts=bool(counts),
                            sigma_from_n_events=bool(err_from_n))
    by_group = {}
    for g, labs in groups.items():
        vals = [abs(per_var[l]["shift"]) for l in labs if l in per_var and per_var[l]["counts"]]
        by_group[g] = float(max(vals)) if vals else 0.0
    return dict(A_nominal=float(a0), sigma_stat=float(s0), variations=per_var, groups=by_group,
                total=float(np.sqrt(sum(v ** 2 for v in by_group.values()))),
                rule="Barlow for sample-changing variants; exclude dchi2>%g; max per group; groups in quadrature" % dchi2_max)
