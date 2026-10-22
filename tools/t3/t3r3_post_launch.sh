#!/usr/bin/env bash
# T3-R3 post-launch pipeline: checks -> evaluation -> analysis -> blind recomputation of a finished development wave.
# It stops at the first nonzero exit (the first version, results/t3_r3/post_launch.sh, ran the evaluation, the analysis and the blind
# recomputation after the checks had exited 1; PI reply 28c). Both roots must be absolute.
#
# Usage, from the repository root, in tmux:
#   bash tools/t3/t3r3_post_launch.sh <results root, absolute> <T3-R2 results root, absolute>
# e.g. bash tools/t3/t3r3_post_launch.sh $PWD/results/t3_r3 /path/to/results/t3_r2
set -euo pipefail

ROOT=${1:?usage: t3r3_post_launch.sh <results root> <T3-R2 results root>}
R2=${2:?usage: t3r3_post_launch.sh <results root> <T3-R2 results root>}
case "$ROOT" in /*) ;; *) echo "the results root must be an absolute path, got '$ROOT'" >&2; exit 2;; esac
case "$R2" in /*) ;; *) echo "the T3-R2 root must be an absolute path, got '$R2'" >&2; exit 2;; esac

PY=${T3R3_PY:-/home/fjiang4/tournament_experiment/.venv/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
LAUNCH=$(cat "$ROOT/launch_commit.txt")
PLAN=reports/t3/r3/dev_plan.csv

step() { echo "=== $1 $(date -u +%FT%TZ)"; }

step checks
"$PY" tools/t3/t3r3_checks.py --root "$ROOT" --plan "$PLAN" --commit "$LAUNCH" --t3r2-root "$R2" \
    --out "$ROOT/dev/launch_checks.json" --workers 40 > "$ROOT/dev/launch_checks.out.txt"
step eval_dev
"$PY" tools/t3/t3r3_eval.py --root "$ROOT" --which dev --out-dir "$ROOT/dev/eval" --workers 40
step eval_diagnostic
"$PY" tools/t3/t3r3_eval.py --root "$ROOT" --which diagnostic --out-dir "$ROOT/diagnostic/eval" --workers 2
step analysis
"$PY" tools/t3/t3r3_analysis.py --root "$ROOT" --dev-eval "$ROOT/dev/eval/per_run.csv" --diag-eval "$ROOT/diagnostic/eval/per_run.csv" \
    --checks "$ROOT/dev/launch_checks.json" --ledger "$ROOT/infra_events.json" --out-dir "$ROOT/dev/analysis" --workers 16
step blind
"$PY" tools/t3/t3r3_blind.py --per-run "$ROOT/dev/eval/per_run.csv" --ledger "$ROOT/infra_events.json" \
    --headline "$ROOT/dev/analysis/headline.json" --out "$ROOT/dev/analysis/blind.json"
echo "done $(date -u +%FT%TZ)"
