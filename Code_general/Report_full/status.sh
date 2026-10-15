#!/bin/bash
# progress of a run: stream logs, running notebooks, elapsed time
DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"
echo "== $(date '+%F %T')  $DIR"
for f in "$DIR"/logs/S*.log; do echo "-- $(basename $f)"; tail -4 "$f"; done
echo "-- running"; ps -eo pid,etime,command | grep -E "nbconvert|ipykernel_launcher" | grep -v grep | cut -c1-150
echo "-- errors in executed notebooks"
python3 - "$DIR" <<'PY'
import json,glob,sys,os
for f in sorted(glob.glob(os.path.join(sys.argv[1],"0*.ipynb"))):
    nb=json.load(open(f)); ex=[c for c in nb["cells"] if c["cell_type"]=="code"]
    done=sum(1 for c in ex if c.get("execution_count")); err=sum(1 for c in ex for o in c.get("outputs",[]) if o.get("output_type")=="error")
    print(f"{os.path.basename(f):42s} cells run {done}/{len(ex)}  errors {err}")
PY
