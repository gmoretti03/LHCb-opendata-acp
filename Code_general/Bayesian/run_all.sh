#!/bin/bash
# Runs the three Bayesian notebooks in order (01 -> 02 -> 03), in place, with the conda env `cern`.
# Every long block resumes from results/ (checkpoints *.jsonl, nominal_fit.json, ...): with the saved results the whole
# script takes a few minutes; from scratch about 2h20 (01) + 25 min (02) + 2h30 (03) on 10 cores.
# Usage:  ./run_all.sh            (all)     ./run_all.sh 01   (only notebook 01, etc.)
cd "$(dirname "$0")"
JUPYTER="${JUPYTER:-jupyter}"
for nb in 01_bayes_mass_fit_and_asymmetry 02_bayes_model_checks_and_priors 03_bayes_vs_frequentist_low_statistics; do
    if [ -n "$1" ] && [[ "$nb" != "$1"* ]]; then continue; fi
    echo "[$(date '+%F %T')] $nb"
    caffeinate -i "$JUPYTER" nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=cern "$nb.ipynb" || exit 1
done
echo "[$(date '+%F %T')] done"
