"""Shared model definitions for the Bayesian analysis of B+/- -> K+ K+ K-.

Imported by Bayesian_Analysis.ipynb, kept separate so the MCMC chain can be
re-run from the notebook without re-defining the model every time, and so
the model math is testable/reviewable on its own.

Parametrization (matches the frequentist model planned on branch `freq`, see
Docs/Proposta_progetto_LHCb_gruppo.tex, "Estensione bayesiana"):
    theta = (N_s, N_b, A, mu, sigma, lambda_bkg)
    f(m) = N_s * S(m; mu, sigma) + N_b * B(m; lambda_bkg)
    N_s^+ = N_s * (1 - A) / 2,  N_s^- = N_s * (1 + A) / 2
    (background assumed split evenly N_b/2, N_b/2 between charges -- no
    background asymmetry is modeled, only signal)

Mass window MASS_LO/MASS_HI was chosen (not the full simulation range) to
stay clear of a rise in background below ~5150 MeV visible in the real data,
consistent with a partially-reconstructed background component that a
single exponential can't describe -- see the "Systematic Uncertainties:
variazione finestre di massa" extension in the proposal for follow-up.
"""

import numpy as np
from scipy.stats import norm as _norm

# Fit range for the B invariant mass (MeV/c^2) -- must match the frequentist
# branch's fit window once agreed with Marco.
MASS_LO = 5150.0
MASS_HI = 5450.0

# Prior bounds, chosen from a first look at the real preselected data
# (n=17839 events in [MASS_LO, MASS_HI], peak at ~5283 MeV): see
# memory note lhcb-bayesian-coding-notes for how these were derived.
PRIOR_BOUNDS = {
    "N_s": (0.0, 5.0 * 17839),
    "N_b": (0.0, 5.0 * 17839),
    "A": (-1.0, 1.0),
    "mu": (5260.0, 5310.0),
    "sigma": (3.0, 40.0),
    "lambda_bkg": (1e-5, 0.05),
}

PARAM_NAMES = ["N_s", "N_b", "A", "mu", "sigma", "lambda_bkg"]
PARAM_LABELS = [r"$N_s$", r"$N_b$", r"$A$", r"$\mu$", r"$\sigma$", r"$\lambda$"]

# Data-driven initial guess (see the 2026-09-21 exploration of the real data).
INITIAL_GUESS = {
    "N_s": 14300.0,
    "N_b": 3550.0,
    "A": 0.0,
    "mu": 5283.0,
    "sigma": 15.0,
    "lambda_bkg": 0.0028,
}


def signal_pdf(m, mu, sigma):
    """Truncated-Gaussian signal shape S(m; mu, sigma) on [MASS_LO, MASS_HI]."""
    norm_const = _norm.cdf(MASS_HI, mu, sigma) - _norm.cdf(MASS_LO, mu, sigma)
    return _norm.pdf(m, mu, sigma) / norm_const


def background_pdf(m, lambda_bkg):
    """Truncated-exponential background shape B(m; lambda_bkg) on [MASS_LO, MASS_HI]."""
    num = lambda_bkg * np.exp(-lambda_bkg * (m - MASS_LO))
    den = 1.0 - np.exp(-lambda_bkg * (MASS_HI - MASS_LO))
    return num / den


def log_prior(theta):
    """Bounded-uniform prior (0.0 inside bounds, -np.inf outside), per the
    course's emcee idiom (see Statistics_exe/Week4_* -- confirmed uniform
    bounds are what's actually used in emcee log_prior functions there,
    not HalfNormal)."""
    N_s, N_b, A, mu, sigma, lambda_bkg = theta
    values = dict(zip(PARAM_NAMES, theta))
    for name, val in values.items():
        lo, hi = PRIOR_BOUNDS[name]
        if not (lo < val < hi):
            return -np.inf
    return 0.0


def log_likelihood(theta, m, q):
    """Extended unbinned log-likelihood, simultaneous fit over B+/B- charge
    subsamples (q = +-1), sharing theta with A tying N_s^+ and N_s^-."""
    N_s, N_b, A, mu, sigma, lambda_bkg = theta

    N_s_plus = N_s * (1.0 - A) / 2.0
    N_s_minus = N_s * (1.0 + A) / 2.0
    N_b_half = N_b / 2.0

    s = signal_pdf(m, mu, sigma)
    b = background_pdf(m, lambda_bkg)

    is_plus = q > 0
    dens_plus = N_s_plus * s[is_plus] + N_b_half * b[is_plus]
    dens_minus = N_s_minus * s[~is_plus] + N_b_half * b[~is_plus]

    if np.any(dens_plus <= 0) or np.any(dens_minus <= 0):
        return -np.inf

    ll = np.sum(np.log(dens_plus)) + np.sum(np.log(dens_minus))
    ll += -(N_s + N_b)  # extended-likelihood normalization term
    return ll


def log_posterior(theta, m, q):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, m, q)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll
