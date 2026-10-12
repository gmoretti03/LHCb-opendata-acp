# Frequentist, Bayesian and machine-learning analysis of the raw asymmetry in B± → K± K+ K−

Final project for the *Statistics and Data Analysis* course (Università di Milano-Bicocca, A.Y. 2025–2026),
Giacomo Moretti and Marco Belotti. The report is `report_complete/main.pdf`.

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
| `Code_general/Frequentist/Frequentist_Analysis_Bis.ipynb` | data, preselection, frequentist fits (separate and simultaneous), profile likelihood, Wilks test, coverage, Dalitz plots | ~1 h | — |
| `Code_general/Frequentist/Frequentist_Systematics.ipynb` | common systematic study (13 variants), charm vetoes, fine PID scan | ~2 min | `freq_systematics.json`, `freq_pid_scan.json` |
| `Code_general/Bayesian/Bayesian_Nominal.ipynb` | nominal Bayesian model: posterior, Bayes factors, dispersed chains, magnet polarities, systematic variants, 200 toys, Dalitz, charm vetoes | ~2.5 h (10 cores) | `nominal/` |
| `Code_general/Bayesian/Bayesian_Nominal_Checks.ipynb` | posterior predictive check, trace plot, Dalitz structure by magnet polarity | ~30 min | `nominal/ppc.json`, `nominal/trace_dispersed.npz`, `nominal/dalitz_magnet.json` |
| `Code_general/Bayesian/Bayes_vs_Freq_LowStat.ipynb` | Bayesian vs frequentist intervals on the same likelihood at low statistics (toys, N = 50–2000) | ~1 h (10 cores) | `lowstat/` |
| `Code_general/Bayesian/Bayesian_Analysis.ipynb` | minimal six-parameter model used as a cross-check, 500 toys | ~16 h (1 core) | `*.json` |
| `Code_general/Machine_Learning/ML_Nominal.ipynb` | weak labels, classifiers, regularization, bias–variance, calibration, uncertainty (RF, MC dropout, deep ensemble), K-means on the Dalitz plane, ML selection and refits with the three methods, hyperparameter search | ~4.5 h | `nominal/` |

Python modules: `lhcb_freq.py` (frequentist fits), `bayesian_nominal.py` (nominal Bayesian model, toys, systematics),
`bayesian_model.py` (minimal model), `lowstat.py` (low-statistics comparison), `ml_nominal.py` (ML helpers).
`run_nominal.sh` and `run_ml_nominal.sh` run the two long notebooks from the command line.
`ML_Nominal.ipynb` reads the nominal Bayesian results (`Bayesian/results/nominal/nominal_summary.json`), so the
Bayesian notebook has to be run first.

## Report and figures

`report_complete/main.tex` (compile with `tectonic main.tex`) only sets the document up: packages and macros are in `report_complete/preamble.tex`, and each section is a separate file in `report_complete/sections/` (`introduction.tex`, `data.tex`, `frequentist.tex`, `bayesian.tex`, `systematics.tex`, `ml.tex`, `comparison.tex`, `limitations.tex`, `conclusions.tex`, `contributions.tex`, `code.tex`, `appendix.tex`, `abstract.tex`), included with `\input`. The figures are rebuilt from the saved results, without
rerunning any fit:

| Script | Figures |
|---|---|
| `report_complete/figures/regen_mass_spectrum.py` | `mass_spectrum.pdf` (Data section) |
| `report_complete/figures/regen_figures_nominal.py` | `bayes_nominal_*.pdf` (Bayesian section) |
| `report_complete/figures/regen_figures_ml_nominal.py` | `ml_nominal_*.pdf` (machine-learning section) |

`plot_style.py` holds the common style. By default the scripts read the results from the `bayes` and `mach_lear`
git branches; set `REGEN_NOMINAL_DIR` / `REGEN_ML_DIR` to a local `results/nominal` directory to read local files.

Where the numbers of the report come from:

| Report | Source |
|---|---|
| Bayesian results, goodness-of-fit and appendix tables, Bayes factors, toys, Dalitz | `Bayesian/results/nominal/nominal_summary.json`, `toys.jsonl` |
| Posterior predictive check, Dalitz by polarity | `Bayesian/results/nominal/ppc.json`, `dalitz_magnet.json` |
| Systematic study table | `Bayesian/results/nominal/nominal_summary.json`, `Frequentist/results/freq_systematics.json` |
| PID scan | `Frequentist/results/freq_pid_scan.json` |
| Low-statistics comparison | `Bayesian/results/lowstat/summary.json` |
| ML tables and figures, three-method comparison, mass after the cut, Dalitz null tests | `Machine_Learning/results/nominal/*.json` |
