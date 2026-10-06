#!/bin/bash
# Full run of Bayesian_Nominal.ipynb (~2h20 on 10 cores). Safe to relaunch: every expensive block
# resumes from results/nominal/*.jsonl and nominal_fit.json. Progress: wc -l results/nominal/*.jsonl
cd "$(dirname "$0")"
LOG=results/nominal_run.log
mkdir -p results/nominal
echo "start $(date '+%F %T')" >> "$LOG"
caffeinate -i /opt/homebrew/Caskroom/miniforge/base/envs/cern/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=cern Bayesian_Nominal.ipynb >> "$LOG" 2>&1
status=$?
echo "end $(date '+%F %T') exit=$status" >> "$LOG"
exit $status
