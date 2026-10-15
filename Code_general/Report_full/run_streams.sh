#!/bin/bash
# usage: run_streams.sh DIR   -- runs the notebooks of DIR in 4 parallel streams (in place), with caffeinate; logs in DIR/logs/
DIR="$(cd "$1" && pwd)"
J=/opt/homebrew/Caskroom/miniforge/base/envs/cern/bin/jupyter
mkdir -p "$DIR/logs"
stream () {   # name, notebooks...
  local name=$1; shift
  for nb in "$@"; do
    echo "[$(date '+%F %T')] start $nb" >> "$DIR/logs/$name.log"
    ( cd "$DIR" && caffeinate -i "$J" nbconvert --to notebook --execute --inplace --allow-errors \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=cern "$nb.ipynb" > "$DIR/logs/$nb.nbconvert.log" 2>&1 )
    echo "[$(date '+%F %T')] done $nb (exit $?)" >> "$DIR/logs/$name.log"
  done
  echo "STREAMDONE" >> "$DIR/logs/$name.log"
}
stream S1 06_Bayesian_systematics_and_checks &
stream S2 02_Frequentist_validation 03_Frequentist_systematics 01_Frequentist_fit 04_Dalitz_plane &
stream S3 07_Bayesian_low_statistics 05_Bayesian_fit 09_ML_measurement &
stream S4 08_ML_classifiers &
wait
echo "[$(date '+%F %T')] ALL STREAMS DONE" >> "$DIR/logs/run.log"
