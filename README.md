# Frequentist, Bayesian and machine-learning analysis of the raw asymmetry in B± → K± K+ K−

Final project for the *Statistics and Data Analysis* course (Università di Milano-Bicocca, A.Y. 2025–2026),
Giacomo Moretti and Marco Belotti. The complete report is `Docs/report_extended/main.pdf`.

## Data

CERN Open Data record 4900, *Matter Antimatter Differences – B meson decays to three hadrons*
(<https://opendata.cern.ch/record/4900>). Put the three files in `opendata-project/Data/`:

```
opendata-project/Data/B2HHH_MagnetDown.root
opendata-project/Data/B2HHH_MagnetUp.root
opendata-project/Data/PhaseSpaceSimulation.root
```

The frequentist code reads the data directory from the environment variable `LHCB_DATA_DIR`
(e.g. `export LHCB_DATA_DIR=$PWD/opendata-project/Data`); the other notebooks use the relative path above.

## Environment

```
conda env create -f environment.yml
conda activate lhcb-kkk
python -m ipykernel install --user --name cern
```

All notebooks use the kernel `cern`. Every long block saves its results in `results/` as it goes and resumes
from them when the notebook is run again; the notebooks are delivered already executed.

## Code

| Notebook | What it does | Runtime | Output (`results/`) |
|---|---|---|---|
| `Code_general/Analysis/Frequentist/06_frequentist_fit_stability_systematics.ipynb` | data, preselection, frequentist fits (separate and simultaneous), profile likelihood, Wilks test, coverage, stability scans, common systematic study (13 variants), charm vetoes, Dalitz plots | ~1 h | `freq_systematics.json`, `freq_pid_scan.json`; `figures/frequentist_wilks_coverage.pdf` |
| `Code_general/Analysis/Bayesian/01_bayes_mass_fit_and_asymmetry.ipynb` | nominal Bayesian model: posterior of $A_{\rm raw}$, Bayes factors, systematic variants, 200 toys, magnet polarities, Dalitz plane, charm vetoes | ~2.5 h (10 cores); minutes with saved results | `nominal/` |
| `Code_general/Analysis/Bayesian/02_bayes_model_checks_and_priors.ipynb` | posterior predictive checks, trace plot, independent Metropolis–Hastings sampler, sensitivity to the priors, HPD intervals, sequential update, $\phi(1020)$ band, comparison with the frequentist profile likelihood | ~25 min; minutes with saved results | `nominal/` |
| `Code_general/Analysis/Bayesian/03_bayes_vs_frequentist_low_statistics.ipynb` | six-parameter model on the data; Bayesian vs frequentist intervals on the same likelihood at low statistics (toys, N = 50–2000) | ~2.5 h (10 cores); seconds with saved results | `lowstat/` |
| `Code_general/Analysis/Machine_Learning/04_ml_signal_background_classifiers.ipynb` | weak labels, classifiers, regularization, bias–variance, calibration, uncertainty (RF, MC dropout, deep ensemble), hyperparameter search | ~3 h; ~5 min with saved results | `nominal/` |
| `Code_general/Analysis/Machine_Learning/05_ml_selection_and_local_asymmetry.ipynb` | ML selection and stability of $A_{\rm raw}$, refits with the three methods, origin of the mass change, K-means and fixed binning on the Dalitz plane, null tests, comparison with the published analysis, high-background sample | ~1.5 h; ~5 min with saved results | `nominal/` |

The Bayesian and machine-learning notebooks are numbered in the order in which they have to be read and run
(`run_all.sh` in each directory runs them in order). Each notebook starts with a table of the sections and figures
of the report that it produces, and generates those figures itself (into `figures/`, next to the notebook).
Python modules: `lhcb_freq.py` (frequentist fits), `bayes_model.py` (Bayesian mass model, sampler, toys,
systematic study), `lowstat.py` (low-statistics comparison), `ml_tools.py` (machine-learning helpers),
`Code_general/Analysis/plot_style.py` (common figure style). The machine-learning notebooks read the nominal Bayesian results
(`Bayesian/results/nominal/nominal_summary.json`), so the Bayesian notebook 01 has to be run first; the frequentist
module is only needed to recompute the frequentist fits of notebook 05 (their saved results are in `results/`).
`Bayesian/results/external/` holds read-only copies of three result files of the frequentist analysis used for the
comparison figures of notebook 02.

## Report and figures

Three versions of the report are built from the same sources:

| Folder | Version | Content |
|---|---|---|
| `Docs/report_extended/` | **complete report** (`main.pdf`, 42 pages) | every analysis, check and figure in its natural place: data, model, frequentist and Bayesian analyses, systematic study, validation, Dalitz plane, machine learning, comparison, limitations |
| `Docs/report_cut/` | body of ~11 pages + appendices | the short body with a pointer to the appendix where each result is justified; the appendices summarize what was done and show the tables and figures |
| `Docs/report_short/` | body only (12 pages with the references) | the version without appendix |

`Docs/report_extended/main.tex` only sets the document up (packages and macros in `preamble.tex`); each section is a file in
`Docs/report_extended/sections/` (`introduction.tex`, `data.tex`, `model.tex`, `frequentist.tex`, `bayesian.tex`,
`systematics.tex`, `validation.tex`, `dalitz.tex`, `ml.tex`, `comparison.tex`, `conclusions.tex`, `code.tex`,
`contributions.tex`, `notation.tex`), included with `\input`; compile with `tectonic -X compile main.tex`. The
figures are in `Docs/report_extended/figures/` together with the scripts that redraw them from the saved results
(`regen_figures_*.py`, common style `plot_style.py`). `Docs/report_build/` holds the sources and the scripts that build the
three versions: `split_body_appendix/build_full.py` assembles the complete report; `make_cut.py` and then
`make_short.py` build the cut and the short version. The figures of the Frequentist, Bayesian and machine-learning
parts are produced by the notebooks listed above (`Code_general/Analysis/*/figures/`), from the saved results, without
rerunning any fit. `Docs/papers/` holds the two reference papers (LHCb-PAPER-2013-027, arXiv:2403.19921) and
`Docs/proposte/` the project proposals.

Where the numbers of the report come from:

| Report | Source |
|---|---|
| Bayesian results, goodness-of-fit and appendix tables, Bayes factors, toys, Dalitz | `Bayesian/results/nominal/nominal_summary.json`, `toys.jsonl` |
| Posterior predictive check, Dalitz by polarity | `Bayesian/results/nominal/ppc.json`, `dalitz_magnet.json` |
| Systematic study table | `Bayesian/results/nominal/nominal_summary.json`, `Frequentist/results/freq_systematics.json` |
| PID scan | `Frequentist/results/freq_pid_scan.json` |
| Low-statistics comparison | `Bayesian/results/lowstat/summary.json` |
| ML tables and figures, three-method comparison, mass after the cut, Dalitz null tests | `Machine_Learning/results/nominal/*.json` |
