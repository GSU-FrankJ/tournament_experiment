#!/usr/bin/env bash
# T3-R3 P1: DG1-DG8 and the retrospective table, at the code commit recorded in each record_*.json. Run from the repository root.
set -u
PY=/home/fjiang4/tournament_experiment/.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT=results/t3_r3/diag
for step in dg1 dg2 dg3 dg6 dg8 retro dg7 dg4 dg5; do
  echo "=== $step $(date -u +%FT%TZ)"
  $PY tools/t3/t3r3_diag.py $step --out-dir $OUT --workers 16
  echo "exit=$? $(date -u +%FT%TZ)"
done
echo done > $OUT/diag.done
