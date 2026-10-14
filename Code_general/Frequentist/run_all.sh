#!/bin/bash
# Runs the frequentist analysis: notebook 06 (about 1 h) and then the toy-MC validation and the figures.
# The validation modes are resumable (every mode skips the items already in results/validation/*.jsonl); with the
# saved results the validation takes a few seconds, from scratch several hours.
# Usage:  ./run_all.sh            (notebook, validation, figures)
#         ./run_all.sh 06         (only the notebook)      ./run_all.sh validation   (only the validation + figures)
cd "$(dirname "$0")"
export LHCB_DATA_DIR="${LHCB_DATA_DIR:-../../opendata-project/Data}"
JUPYTER="${JUPYTER:-jupyter}"
PY="${PY:-python}"
if [ -z "$1" ] || [[ "06_frequentist_fit_stability_systematics" == "$1"* ]]; then
    echo "[$(date '+%F %T')] 06_frequentist_fit_stability_systematics"
    caffeinate -i "$JUPYTER" nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=cern 06_frequentist_fit_stability_systematics.ipynb || exit 1
fi
if [ -z "$1" ] || [ "$1" = "validation" ]; then
    mkdir -p results/validation
    {
    caffeinate -i $PY freq_validation.py profiles
    caffeinate -i $PY freq_validation.py multistart --n 30
    caffeinate -i $PY freq_validation.py momentum
    caffeinate -i $PY freq_validation.py toys --n 1000 --n-other 300
    caffeinate -i $PY freq_validation.py belt --n 300
    caffeinate -i $PY freq_validation.py barlow --n 300
    caffeinate -i $PY freq_validation.py gof
    caffeinate -i $PY freq_validation.py cov
    caffeinate -i $PY freq_validation.py profile_par
    caffeinate -i $PY freq_validation.py scans
    caffeinate -i $PY freq_validation.py jointfit
    echo "[$(date '+%F %T')] validation done"
    } 2>&1 | tee -a results/validation/run.log
    $PY make_figures.py
fi
echo "[$(date '+%F %T')] done"
