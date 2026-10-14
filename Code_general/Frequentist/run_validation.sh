#!/bin/bash
# Runs all the long validation modes in sequence (resumable: every mode appends/skips finished items).
cd "$(dirname "$0")"
export LHCB_DATA_DIR="${LHCB_DATA_DIR:-../../opendata-project/Data}"
PY="conda run -n cern --no-capture-output python"
{
$PY freq_validation.py multistart --n 30
$PY freq_validation.py momentum
$PY freq_validation.py toys --n 1000 --n-other 300
$PY freq_validation.py belt --n 300
$PY freq_validation.py barlow --n 300
echo ALL_DONE
} 2>&1 | tee results/validation/run.log
