# Machine learning applied to B± → K± K+ K−

Read the notebooks in order. Each one starts with a table of the sections and figures of the report that it produces,
and generates those figures itself (into `figures/`).

| # | Notebook | Question | Report |
|---|---|---|---|
| 04 | `04_ml_signal_background_classifiers.ipynb` | How well can signal-like and background-like candidates be separated from variables other than the mass? Five classifiers, regularization, bias–variance, calibration, uncertainty, hyperparameters. | Appendix: machine-learning analysis |
| 05 | `05_ml_selection_and_local_asymmetry.ipynb` | Does the ML selection change the measurement of A_raw? Stability, refits with the three methods, origin of the mass change; K-means and null tests on the Dalitz plane; comparison with the published analysis; high-background sample. | Appendix: ML, Dalitz, Comparison |

The labels are *weak*: signal-like under the peak, background-like in the sidebands, from the mass of the nominal
Bayesian fit. The classifiers never see the mass. Helper functions (data, weak labels, split, statistics, checkpoints):
`ml_tools.py`.

## Running

* Needs the Bayesian notebook 01 (`../Bayesian/results/nominal/nominal_summary.json`) and the data in
  `../../opendata-project/Data`.
* `./run_all.sh` runs 04 → 05 in place (`./run_all.sh 05` runs one). Set `JUPYTER` if `jupyter` is not on the PATH.
* Every long block resumes from `results/nominal/`: with the saved results the two notebooks take about 5 minutes;
  from scratch about 3 h (04) and 1.5 h (05) on 10 cores. Seeds are fixed and TensorFlow op determinism is enabled.
* The frequentist fits of notebook 05 use `lhcb_freq.py` of the frequentist analysis (`../Frequentist/`). If it is
  not present the saved results are used; it is needed only to recompute those blocks.

Files written: see `results/README.md`.
