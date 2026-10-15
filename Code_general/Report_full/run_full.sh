#!/bin/bash
# Production run (several hours). Checks, then launches the 4 streams detached from the terminal.
cd "$(dirname "$0")"
if pgrep -f "ipykernel_launcher|nbconvert" >/dev/null; then echo "kernels or nbconvert already running: stop them first"; pgrep -fl "ipykernel_launcher|nbconvert"; exit 1; fi
ls data/*.root >/dev/null 2>&1 || { echo "data/ is empty"; exit 1; }
mkdir -p logs; echo "[$(date '+%F %T')] launching" >> logs/run.log
nohup caffeinate -i ./run_streams.sh "$PWD" > logs/launcher.out 2>&1 &
echo "launched (pid $!). Follow with ./status.sh"
