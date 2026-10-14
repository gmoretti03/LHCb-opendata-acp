# Frequentist analysis of the raw asymmetry in B± → K± K+ K−

Read the notebook from the top. It starts with a table of the sections and figures of the report that it produces.

| # | Notebook | Question | Report |
|---|---|---|---|
| 06 | `06_frequentist_fit_stability_systematics.ipynb` | What is A_raw from the extended unbinned likelihood (separate and simultaneous fit), with which Minos intervals and which significance against A = 0? Does the Wilks interval cover (bootstrap)? Is the result stable against window, signal region and PID cuts? What is the systematic uncertainty (13 variants common to the Bayesian analysis)? Charm vetoes and local asymmetry in the Dalitz plane. | Frequentist analysis, Systematics, Dalitz |

Code: `lhcb_freq.py` (data, preselection, mass model, fits, profile likelihood, scans, systematic combination — used by
the notebook and by `freq_validation.py`), `freq_validation.py` (toy-MC validation of the estimator and of the
intervals: `profiles`, `toys`, `belt`, `multistart`, `momentum`, `barlow`, `cov`, `gof`, `profile_par`, `scans`, `jointfit`),
`make_figures.py` (figures of the validation, profile likelihood, simultaneous fit with pulls, stability grids, momentum
terciles and PID scan, from `results/`).

## Running

* Environment: `environment.yml` at the root of the repository; data in `../../../opendata-project/Data`
  (or the directory in `LHCB_DATA_DIR`).
* `./run_all.sh` runs the notebook in place and then the validation and `make_figures.py`
  (`./run_all.sh 06` only the notebook, `./run_all.sh validation` only the validation). Set `JUPYTER` if `jupyter` is
  not on the PATH.
* The notebook takes about 1 h. The validation modes resume from `results/validation/`: with the saved results it takes
  seconds; from scratch several hours on 9 cores (`NPROC`). All random numbers are generated with fixed seeds.
* The notebook needs no other result. The Bayesian notebooks 02 and 03 and the machine-learning notebook 05 read
  copies of some of its result files.

Files written: see `results/README.md`.

The original notebooks of the frequentist part (before the merge into notebook 06) are kept unchanged in
`../Frequentist_Origin/`.
