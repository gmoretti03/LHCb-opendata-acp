# Frequentist results

Written by the notebook and by `freq_validation.py` (the mode that writes each file is in the second column); they are
read back by `make_figures.py`, by the Bayesian notebooks (`../Bayesian/results/external/` holds read-only copies) and
by the machine-learning notebook 05.

## Notebook 06

| File | Writer | Content |
|---|---|---|
| `freq_systematics.json` | 06 | nominal fit, the 13 variants (shift, $\sigma_B$, rejected by the data), group uncertainties, total, charm vetoes |
| `freq_pid_scan.json` | 06 | fine scan of the ProbK and ProbPi cuts, one cut at a time: shift, $\sigma_B$, pull |

The coverage figure `frequentist_wilks_coverage.pdf` is written by the notebook to `../figures/`.

## `validation/` — toy-MC validation

| File | Mode | Content |
|---|---|---|
| `nominal_values.json` | `profiles` | parameters of the nominal fit used to generate the toys |
| `profiles.json` | `profiles` | $q(A)$ on the full sample and on each magnet polarity, $(A, A_b)$ regions, Minos intervals |
| `multistart.json` | `multistart` | nominal fit from random starting points |
| `toys.jsonl`, `toys_summary.json` | `toys` | parametric toys at five injected asymmetries: $\hat A$, Hesse error, $t(A_{\rm true})$, $q(0)$; bias, coverage, size, power |
| `belt.jsonl`, `belt_summary.json` | `belt` | toys on a grid of $A_{\rm true}$, Neyman confidence belt and its 68 % interval |
| `barlow.jsonl`, `barlow_summary.json` | `barlow` | bootstrap of the shift between nested samples against $\sigma_B$ |
| `momentum.json` | `momentum` | $A_{\rm raw}$ in terciles of the momentum of the $B$ |
| `cov.jsonl`, `cov_summary.json`, `cov_hesse.json` | `cov` | bootstrap covariance of the twelve parameters against Hesse |
| `gof.jsonl`, `gof_data.json`, `gof_summary.json` | `gof` | binned $\chi^2$ of toys from the fitted model with 50, 100 and 200 bins |
| `profile_par.json` | `profile_par` | profile of the other parameters of the model |
| `run.log`, `*.log` | | logs of the runs |

Delete a `.jsonl` file to recompute its mode; every mode appends and skips what is already there.
The figures (`../figures/`) are rebuilt from these files by `make_figures.py` without rerunning any fit.
