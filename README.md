# Frequentist, Bayesian and machine-learning analysis of the raw asymmetry in B± → K± K+ K−

Final project for the *Statistics and Data Analysis* course (Università di Milano-Bicocca, A.Y. 2025–2026),
Giacomo Moretti and Marco Belotti. The complete report is `Docs/report_extended/main.pdf`.

## Layout of the repository

```
Code_general/
  Report/      the notebooks that accompany the report (9 notebooks, self-contained, already executed)
  Analysis/    the complete analysis: modules, long runs, saved results (Frequentist, Bayesian, Machine_Learning)
Docs/          the report (sources and PDF), the papers used and the project proposals
Statistics_course/   material of the course
opendata-project/    CERN Open Data record 4900 (data, not tracked by git)
environment.yml      conda environment of all the code
```

## Data

CERN Open Data record 4900, *Matter Antimatter Differences – B meson decays to three hadrons*
(<https://opendata.cern.ch/record/4900>). The three files go in `opendata-project/Data/`:

```
opendata-project/Data/B2HHH_MagnetDown.root
opendata-project/Data/B2HHH_MagnetUp.root
opendata-project/Data/PhaseSpaceSimulation.root
```

The notebooks of `Code_general/Report/` read `Code_general/Report/data/` first (a copy of the three files, so that the
folder can be used on its own) and `opendata-project/Data/` otherwise. The notebooks of `Code_general/Analysis/` read
`../../../opendata-project/Data` (frequentist code: environment variable `LHCB_DATA_DIR`).

## Environment

```
conda env create -f environment.yml
conda activate lhcb-kkk
python -m ipykernel install --user --name cern
```

All notebooks use the kernel `cern`.

## Notebooks of the report: `Code_general/Report/`

A guided tour of the whole analysis **in miniature**: the same methods and the same order as the report, with fewer
replicas, toy experiments and shorter chains, so that each notebook runs in 1–15 minutes. The numbers can differ
slightly from the report; next to every result the notebook prints the value of the report (**report:**). Each notebook
saves its figures in `figures/` and its numbers in `results/` (`NN_summary.json` and tables), and tags what goes beyond
the techniques of the course as **[EXTRA]**. See `Code_general/Report/README.md`.

| Notebook | Content | Run time |
|---|---|---|
| `01_Frequentist_fit` | data and preselection, extended unbinned likelihood (`iminuit`), Minos, profile likelihood, Wilks test | 1 min |
| `02_Frequentist_validation` | bootstrap, toy experiments (bias, pulls, coverage, size, power), Neyman belt, momentum, goodness of fit and binning | 12 min |
| `03_Frequentist_systematics` | 13 variants, Barlow criterion, scans of window, signal region and PID cuts | 3 min |
| `04_Dalitz_plane` | local asymmetry, homogeneity test, null tests, φ(1020), published region, charm vetoes | < 1 min |
| `05_Bayesian_fit` | priors, MCMC (`emcee`), convergence, credible/HPD intervals, Bayes factors, prior dependence | 4 min |
| `06_Bayesian_systematics_and_checks` | Bayesian variants, coverage, posterior predictive check, polarities, comparison with the profile | 14 min |
| `07_Bayesian_low_statistics` | credible against confidence intervals at 50–2000 candidates, asymptotic behaviour | 2 min |
| `08_ML_classifiers` | weak labels, six classifiers, Ridge/Lasso, bias–variance, calibration, uncertainty, hyperparameters, ROC and significance improvement | 9 min |
| `09_ML_measurement` | effect of the ML selection on A_raw (score > 0.5 and at the maximum of the significance improvement, 0.73), sculpting test, K-means on the Dalitz plane, high-background sample | 5 min |

## Complete analysis: `Code_general/Analysis/`

The production code and results behind the numbers of the report (long runs, saved in `results/` and resumed on
rerun). Each directory has its own README.

| Notebook | What it does | Runtime | Output (`results/`) |
|---|---|---|---|
| `Frequentist/06_frequentist_fit_stability_systematics.ipynb` | data, preselection, frequentist fits (separate and simultaneous), profile likelihood, Wilks test, coverage, stability scans, common systematic study (13 variants), charm vetoes, Dalitz plots | ~1 h | `freq_systematics.json`, `freq_pid_scan.json` |
| `Bayesian/01_bayes_mass_fit_and_asymmetry.ipynb` | nominal Bayesian model: posterior of A_raw, Bayes factors, systematic variants, 200 toys, magnet polarities, Dalitz plane, charm vetoes | ~2.5 h (10 cores); minutes with saved results | `nominal/` |
| `Bayesian/02_bayes_model_checks_and_priors.ipynb` | posterior predictive checks, trace plot, Metropolis–Hastings sampler, sensitivity to the priors, HPD intervals, sequential update, φ(1020) band, comparison with the profile likelihood | ~25 min | `nominal/` |
| `Bayesian/03_bayes_vs_frequentist_low_statistics.ipynb` | six-parameter model; Bayesian against frequentist intervals at low statistics (toys, N = 50–2000) | ~2.5 h (10 cores) | `lowstat/` |
| `Machine_Learning/04_ml_signal_background_classifiers.ipynb` | weak labels, classifiers, regularization, bias–variance, calibration, uncertainty (RF, MC dropout, deep ensemble), hyperparameter search | ~3 h | `nominal/` |
| `Machine_Learning/05_ml_selection_and_local_asymmetry.ipynb` | ML selection and stability of A_raw, refits with three methods, origin of the mass change, K-means and fixed binning, null tests, comparison with the published analysis, high-background sample | ~1.5 h | `nominal/` |

Python modules: `Frequentist/lhcb_freq.py` (frequentist fits), `Frequentist/freq_validation.py` (toy-MC validation of
the estimator and of the intervals; `run_all.sh`), `Frequentist/make_figures.py`, `Bayesian/bayes_model.py` (mass
model, sampler, toys, systematic study), `Bayesian/lowstat.py`, `Machine_Learning/ml_tools.py` and
`plot_style.py` (common figure style). The machine-learning notebooks read the nominal Bayesian results
(`Bayesian/results/nominal/nominal_summary.json`), so `Bayesian/01_…` has to be run first; `Bayesian/results/external/`
holds read-only copies of three frequentist result files used for the comparison figures of `Bayesian/02_…`.
`Frequentist_Origin/` keeps the original frequentist notebooks, unchanged.

Every long block saves its results in `results/` as it goes and resumes from them when the notebook is run again;
the notebooks are delivered already executed. All random numbers are generated with fixed seeds.

## Report: `Docs/`

| Folder | Content |
|---|---|
| `Docs/report_extended/` | the complete report (`main.pdf`, LaTeX sources in `sections/`, figures in `figures/`) |
| `Docs/report_build/` | sources of the report and the scripts that build it (figures and their `regen_*.py` scripts) |
| `Docs/papers/`, `Docs/proposte/` | the LHCb papers used for the comparison and the project proposals |

Compile with `tectonic -X compile main.tex` (or any LaTeX distribution) from the folder of the report. The figures of
the frequentist, Bayesian and machine-learning parts are produced by the notebooks of `Code_general/Analysis/`
(`figures/` next to each notebook) from the saved results, without rerunning any fit.

## Where the numbers of the report come from

| Report | Source (in `Code_general/Analysis/`) |
|---|---|
| Frequentist results, Wilks coverage, stability scans, PID scan | `Frequentist/results/freq_systematics.json`, `freq_pid_scan.json`, `Frequentist/results/validation/` |
| Bayesian results, goodness of fit, Bayes factors, toys, Dalitz | `Bayesian/results/nominal/nominal_summary.json`, `toys.jsonl` |
| Posterior predictive check, Dalitz by polarity | `Bayesian/results/nominal/ppc.json`, `dalitz_magnet.json` |
| Systematic study table | `Bayesian/results/nominal/nominal_summary.json`, `Frequentist/results/freq_systematics.json` |
| Low-statistics comparison | `Bayesian/results/lowstat/summary.json` |
| ML tables and figures, three-method comparison, mass after the cut, Dalitz null tests | `Machine_Learning/results/nominal/*.json` |
