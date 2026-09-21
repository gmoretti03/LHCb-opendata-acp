"""Shared model definitions for the Bayesian analysis of B+/- -> h+ h+ h-.

Imported by Bayesian_Analysis.ipynb, kept separate so the MCMC chain can be
re-run from the notebook without re-defining the model every time, and so
the model math is testable/reviewable on its own.

Parametrization (matches the frequentist model on branch `freq`, see
Docs/Proposta_progetto_LHCb_gruppo.tex, "Estensione bayesiana"):
    theta = (N_s, N_b, A, mu, sigma, lambda_bkg)
    f(m) = N_s * S(m; mu, sigma) + N_b * B(m; lambda_bkg)
    N_s^+ = N_s * (1 - A) / 2,  N_s^- = N_s * (1 + A) / 2

Nothing here is implemented yet -- fill in once the shared preselection
and mass reconstruction are agreed with Marco (branch `freq`).
"""

import numpy as np

# Fit range for the B invariant mass (MeV/c^2) -- must match the frequentist
# branch's fit window once agreed.
MASS_LO = 5100.0
MASS_HI = 5450.0


def signal_pdf(m, mu, sigma):
    """Truncated-Gaussian signal shape S(m; mu, sigma) on [MASS_LO, MASS_HI]."""
    raise NotImplementedError


def background_pdf(m, lambda_bkg):
    """Truncated-exponential background shape B(m; lambda_bkg) on [MASS_LO, MASS_HI]."""
    raise NotImplementedError


def log_prior(theta):
    """Bounded-uniform prior, per the course's emcee idiom (0.0 inside bounds,
    -np.inf outside) -- see memory note lhcb-bayesian-coding-notes for the
    agreed bounds per parameter."""
    raise NotImplementedError


def log_likelihood(theta, m, q):
    """Extended unbinned log-likelihood, simultaneous fit over B+/B- charge
    subsamples (q = +-1), sharing theta with A tying N_s^+ and N_s^-."""
    raise NotImplementedError


def log_posterior(theta, m, q):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, m, q)
