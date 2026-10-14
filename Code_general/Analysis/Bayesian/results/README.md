# Bayesian results

Written by the notebooks (the notebook that writes each file is in the second column); they are read back by the
notebooks themselves and by the machine-learning analysis.

## `nominal/` — nominal 12-parameter model

| File | Notebook | Content |
|---|---|---|
| `nominal_fit.json` | 01 | nominal fit: statistics of every parameter (median, 68/95 % intervals), MAP, χ², autocorrelation times |
| `nominal_posterior_thinned.npy` | 01 | 20 000 posterior samples (12 parameters) used by all the figures |
| `fits.jsonl` | 01 | checkpoint of the ~20 fits: dispersed starts, magnet polarities, 13 variants, prior check, charm vetoes |
| `toys.jsonl` | 01 | checkpoint of the 200 toy-MC refits |
| `magnet_down_A_thinned.npy`, `magnet_up_A_thinned.npy` | 01 | posterior of A for each magnet polarity |
| `nominal_summary.json` | 01 | **everything quoted from notebook 01** (also read by notebooks 02–05 and the ML analysis) |
| `ppc.json` | 02 | 1000 posterior predictive replicas, discrepancies and Bayesian p-values |
| `trace_dispersed.npz` | 02 | 32 walkers started over the whole prior box (every 10th step) |
| `validation.json` | 02 | Bayes factor against the prior width, Savage–Dickey of the background shapes, unfitted PPC statistics, sequential update |
| `extras.json` | 02 | conditional posterior, prior reweighting, evidence, HPD intervals, Metropolis–Hastings |
| `dalitz_magnet.json` | 02 | Dalitz homogeneity by magnet polarity |
| `dalitz_resonances.json` | 02 | asymmetry in the φ(1020) band and in the significant bins |

## `lowstat/` — six-parameter model, low statistics

| File | Notebook | Content |
|---|---|---|
| `truth.json` | 03 | fit of the six-parameter model to the data (truth of the pseudo-datasets) |
| `toys.jsonl` | 03 | checkpoint of the 1200 pseudo-datasets (Bayesian and frequentist result of each) |
| `summary.json` | 03 | coverage, bias and interval shape per sample size |
| `scaling.json` | 03 | normality and scaling of the estimator with N |

## `external/` — copies of frequentist results (read only)

`freq_systematics.json`, `profiles.json`, `cov_summary.json`: written by the frequentist notebooks, copied here so that
the Bayesian notebooks run without the frequentist code. Copy them again if the frequentist analysis is rerun.
