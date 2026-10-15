# Notebooks of the project: raw asymmetry in B± → K± K+ K−

Nine notebooks that accompany the report, one for each part of the analysis. They reproduce the **whole analysis in
miniature**: the same methods and the same order as the report, with fewer replicas, toy experiments and shorter chains,
so that every notebook runs in a few minutes and the results can be navigated easily. Numbers can therefore differ
slightly from the report; next to every result the notebook prints the value of the report (**report:**).

Every notebook is **self-contained** (it starts with a *Common setup* that repeats what it needs), is delivered already
executed, saves its figures in `figures/` and its numbers in `results/` (`NN_summary.json` and tables in `.csv`), and tags
the parts that go beyond the techniques of the course as **[EXTRA]**.

| Notebook | Content | Report sections |
|---|---|---|
| `01_Frequentist_fit` | data, preselection, mass, extended unbinned likelihood (`iminuit`), Minos, profile likelihood and Wilks test, cross-checks | Data, Model, Frequentist analysis |
| `02_Frequentist_validation` | bootstrap coverage, toy experiments (bias, pulls, coverage, size and power), Neyman belt, momentum, goodness of fit and binning | Frequentist analysis, Validation |
| `03_Frequentist_systematics` | 13 variants, Barlow criterion, combination, scans of the window, signal region and PID cuts | Result and systematic uncertainties |
| `04_Dalitz_plane` | local asymmetry, homogeneity test, null tests, φ(1020), published region, charm vetoes | Local asymmetry in the Dalitz plane |
| `05_Bayesian_fit` | priors, MCMC (`emcee`), convergence, credible/HPD intervals, Bayes factors, prior dependence | Bayesian analysis |
| `06_Bayesian_systematics_and_checks` | Bayesian variants, coverage of the credible interval, posterior predictive check, polarities, comparison with the profile likelihood | Systematics, Validation, Discussion |
| `07_Bayesian_low_statistics` | credible against confidence intervals at 50–2000 candidates, asymptotic behaviour | Discussion |
| `08_ML_classifiers` | weak labels, six classifiers, Ridge/Lasso, bias–variance, calibration, uncertainty, hyperparameters, ROC and significance improvement | Machine learning |
| `09_ML_measurement` | effect of an ML selection on A_raw (three methods, score > 0.5 and > 0.73), sculpting test, K-means on the Dalitz plane, high-background sample | Machine learning |

Section numbers inside the frequentist notebooks (01–04) and the Bayesian ones (05–07) follow the whole analysis of each
method, so §5 of `02_…` comes after §4 of `01_…`.

## Data
CERN Open Data record 4900 (<https://opendata.cern.ch/record/4900>). Put the files in `data/`:

```
data/B2HHH_MagnetDown.root
data/B2HHH_MagnetUp.root
data/PhaseSpaceSimulation.root      (only for the null test of notebook 04)
```

(otherwise the notebooks look for `../../opendata-project/Data`).

## Environment
`conda env create -f environment.yml`, then `python -m ipykernel install --user --name cern`.
`plot_style.py` (colours and font sizes) is imported by the notebooks and must stay in this folder.
