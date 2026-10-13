#!/bin/bash
# Runs the two machine-learning notebooks in order (04 -> 05), in place, with the conda env `cern`.
# Needs the Bayesian notebook 01 (../Bayesian/results/nominal/nominal_summary.json).
# Every long block resumes from results/nominal/: with the saved results the script takes about 10 minutes;
# from scratch about 3 h (04) + 1.5 h (05) on 10 cores.
# Usage:  ./run_all.sh            (both)     ./run_all.sh 04   (only notebook 04)
cd "$(dirname "$0")"
JUPYTER="${JUPYTER:-jupyter}"
for nb in 04_ml_signal_background_classifiers 05_ml_selection_and_local_asymmetry; do
    if [ -n "$1" ] && [[ "$nb" != "$1"* ]]; then continue; fi
    echo "[$(date '+%F %T')] $nb"
    caffeinate -i "$JUPYTER" nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=cern "$nb.ipynb" || exit 1
done
echo "[$(date '+%F %T')] done"
