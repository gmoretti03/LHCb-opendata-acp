#!/bin/bash
# Runs the frequentist notebook top to bottom (~1 h) and then the toy-MC validation (long, resumable).
cd "$(dirname "$0")"
export LHCB_DATA_DIR="${LHCB_DATA_DIR:-../../opendata-project/Data}"
PY="conda run -n cern --no-capture-output python"
conda run -n cern --no-capture-output jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 Frequentist_Analysis.ipynb 2>&1 | tee results/run_notebook.log
mkdir -p results/validation
{
$PY freq_validation.py multistart --n 30
$PY freq_validation.py momentum
$PY freq_validation.py toys --n 1000 --n-other 300
$PY freq_validation.py belt --n 300
$PY freq_validation.py barlow --n 300
echo ALL_DONE
} 2>&1 | tee results/validation/run.log
