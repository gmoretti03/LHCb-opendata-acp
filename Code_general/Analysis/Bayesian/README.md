# Bayesian analysis of the raw asymmetry in B± → K± K+ K−

Read the notebooks in order. Each one starts with a table of the sections and figures of the report that it produces,
and generates those figures itself (into `figures/`).

| # | Notebook | Question | Report |
|---|---|---|---|
| 01 | `01_bayes_mass_fit_and_asymmetry.ipynb` | What is the posterior of A_raw, how large is the systematic uncertainty, and is the procedure closed on toys? Bayes factor of A = 0, magnet polarities, local asymmetry in the Dalitz plane, charm vetoes. | Bayesian analysis, Systematics, Dalitz |
| 02 | `02_bayes_model_checks_and_priors.ipynb` | Can the posterior be trusted? Posterior predictive checks, convergence, own Metropolis–Hastings sampler, sensitivity to the priors, HPD intervals, sequential update, φ(1020) band, comparison with the frequentist profile likelihood. | Appendix: Bayesian analysis, Comparison |
| 03 | `03_bayes_vs_frequentist_low_statistics.ipynb` | Where do credible and confidence intervals differ? Same likelihood, 50–2000 candidates, coverage and bias. | Appendix: low statistics, asymptotic behavior |

Code: `bayes_model.py` (mass model, priors, sampler, toy generator, systematic study — used by all three notebooks),
`lowstat.py` (toy generator and frequentist fit of notebook 03).

## Running

* Environment: `environment.yml` at the root of the repository; data in `../../../opendata-project/Data`.
* `./run_all.sh` runs 01 → 02 → 03 in place (`./run_all.sh 02` runs one). Set `JUPYTER` if `jupyter` is not on the PATH.
* Every long block resumes from `results/`: with the saved results the three notebooks take about one minute;
  from scratch about 2 h 20 min (01), 25 min (02) and 2 h 30 min (03) on 10 cores. Delete a result file to recompute
  its block. All random numbers are generated with fixed seeds.
* 02 and the systematic figure of 01 read read-only copies of three frequentist result files in `results/external/`.

Files written: see `results/README.md`.
