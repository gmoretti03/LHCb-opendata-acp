# Machine-learning results (`nominal/`)

## Written by notebook 04

| File | Content |
|---|---|
| `data_labels.json` | μ, σ of the nominal fit, number of signal-like and background-like candidates, split sizes, class weights |
| `models.json` | CV and test AUC (with bootstrap), Brier score, reference model, regularization scan |
| `test_scores.npy` | test labels and scores of the six models (columns: y, then the models of `models.json["order"]`) |
| `features.json` | random-forest importance, Lasso coefficients, ranks, Spearman correlation |
| `bias_variance_calibration.json` | staged accuracy/AUC of the boosting model; reliability curves and Brier scores |
| `uncertainty.json`, `uncertainty_test.npy` | RF spread, MC dropout, deep ensemble; dependence on the training-set size |
| `deep_ensemble.json`, `training_size_study.json` | checkpoints of the two long uncertainty studies |
| `hyperparameter_search.json` | RF and GBM grids (repeated cross-validation) |
| `ffnn_search.jsonl`, `ffnn_search_summary.json` | checkpoint and summary of the 168-configuration FFNN search |
| `ml_classifiers_summary.json` | summary of notebook 04 |

## Written by notebook 05

| File | Content |
|---|---|
| `threshold_stability.json` | counting asymmetry against the score threshold, magnet polarities |
| `refit_selected.json` | nominal Bayesian fit of the selected sample |
| `three_methods_selected.json` | Bayesian, frequentist and counting results, full and selected sample |
| `selection_mass_origin.json` | tests of the change of the mass spectrum (no-momentum classifier, efficiency against mass, χ² by region) |
| `kmeans.json`, `kmeans8_labels.npy` | K-means clustering of the Dalitz plane, cluster significance, silhouette scan, labels of K = 8 |
| `dalitz_checks.json` | alternative binnings, phase-space null test, charge permutation, magnet polarity |
| `paper_region.json`, `paper_region_mass_charge.npy` | region of LHCb-PAPER-2013-027: three methods, with and without the D⁰ veto; mass and charge of its candidates |
| `high_background.json` | looser PID sample: tighter PID cut against ML cut |
| `ml_selection_summary.json` | summary of notebook 05 |

The `.json` results that come from long computations carry a `_key` with the settings that produced them; a notebook
recomputes the block if the key no longer matches.
