# results/

Cached outputs from `Bayesian_Analysis.ipynb`, so the notebook doesn't need
to re-run the (slow) emcee sampling every time it's reopened.

Expected files (none yet):
- `mcmc_chain.npz` -- raw chain from `sampler.get_chain()`, plus burn-in/discard used
- `posterior_summary.json` -- A_raw median + 68%/95% credible intervals + Bayes factor,
  the numbers the final cross-branch comparison notebook will read
