#!/usr/bin/env bash
# T3-R4 post-launch pipeline: checks -> evaluation -> analysis -> blind recomputation of the finished confirmation wave.
# It stops at the first nonzero exit (set -e). The results root must be absolute. Run it only after every run has finished.
#
# Usage, from the repository root, in tmux:
#   bash tools/t3/t3r4_post_launch.sh <results root, absolute>
# e.g. bash tools/t3/t3r4_post_launch.sh $PWD/results/t3_r4/confirmation
set -euo pipefail

ROOT=${1:?usage: t3r4_post_launch.sh <results root, absolute>}
case "$ROOT" in /*) ;; *) echo "the results root must be an absolute path, got '$ROOT'" >&2; exit 2;; esac

PY=${T3R4_PY:-/home/fjiang4/tournament_experiment/.venv/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
LAUNCH=$(cat "$ROOT/launch_commit.txt")
PLAN=reports/t3/r4/confirmation_plan.csv

step() { echo "=== $1 $(date -u +%FT%TZ)"; }

step checks
"$PY" tools/t3/t3r4_checks.py --root "$ROOT" --plan "$PLAN" --commit "$LAUNCH" \
    --out "$ROOT/launch_checks.json" --workers 40 > "$ROOT/launch_checks.out.txt"
step eval
"$PY" tools/t3/t3r4_eval.py --root "$ROOT" --out-dir "$ROOT/eval" --workers 40
step analysis
"$PY" tools/t3/t3r4_analysis.py --root "$ROOT" --eval "$ROOT/eval/per_run.csv" --checks "$ROOT/launch_checks.json" \
    --ledger "$ROOT/infra_events.json" --out-dir "$ROOT/analysis" --workers 16
step blind
"$PY" tools/t3/t3r4_blind.py --per-run "$ROOT/eval/per_run.csv" --ledger "$ROOT/infra_events.json" \
    --headline "$ROOT/analysis/headline.json" --out "$ROOT/analysis/blind.json"
echo "done $(date -u +%FT%TZ)"
