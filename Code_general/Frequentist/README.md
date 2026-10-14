# Frequentist analysis of the raw asymmetry in B± → K± K+ K−

One notebook, read top to bottom. It generates the numbers of the frequentist part of the report.

| Notebook | Question | Report |
|---|---|---|
| `Frequentist_Analysis.ipynb` | What is A_raw from the extended unbinned likelihood (separate and simultaneous fit), with which Minos intervals and which significance against A = 0? Does the Wilks interval cover (bootstrap)? Is the result stable against window, signal region and PID cuts? What is the systematic uncertainty (13 variants common to the Bayesian analysis)? Dalitz plots, charm vetoes, local asymmetry. | Frequentist analysis, Systematics, Dalitz |

Code: `lhcb_freq.py` (data, preselection, mass model, fits, profile likelihood, scans, systematic combination),
`freq_validation.py` (toy-MC validation of the estimator and of the intervals, resumable: `toys`, `belt`, `barlow`,
`multistart`, `momentum`, `gof`, `cov`).

## Running

* Environment: `environment.yml` at the root of the repository; data in `../../opendata-project/Data`
  (or the directory in `LHCB_DATA_DIR`).
* `./run_all.sh` runs the notebook in place (~1 h) and then the validation.
* The notebook writes `results/freq_systematics.json`, `results/freq_pid_scan.json` and
  `results/frequentist_wilks_coverage.pdf`; the validation writes `results/validation/`.
* Seeds are fixed (bootstrap `seed=1`).

The original notebooks (Marco's `Frequentist_Analysis_Ter`, the annotated `Frequentist_Analysis_Bis` and
`Frequentist_Systematics`) are kept unchanged in `../Frequentist_Origin/`.
