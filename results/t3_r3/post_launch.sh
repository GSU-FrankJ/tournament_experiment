#!/usr/bin/env bash
# T3-R3 P2: post-launch checks, evaluation, analysis and blind recomputation of the development wave. Run from the repository root.
set -u
PY=/home/fjiang4/tournament_experiment/.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
R2=/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-audit-6f3a38/results/t3_r2
LAUNCH=$(cat results/t3_r3/launch_commit.txt)
step() { echo "=== $1 $(date -u +%FT%TZ)"; shift; "$@"; echo "exit=$? $(date -u +%FT%TZ)"; }
step checks $PY tools/t3/t3r3_checks.py --root results/t3_r3 --plan reports/t3/r3/dev_plan.csv --commit $LAUNCH --t3r2-root $R2 --out results/t3_r3/dev/launch_checks.json --workers 40 > results/t3_r3/dev/launch_checks.out.txt
step eval_dev $PY tools/t3/t3r3_eval.py --root results/t3_r3 --which dev --out-dir results/t3_r3/dev/eval --workers 40
step eval_diag $PY tools/t3/t3r3_eval.py --root results/t3_r3 --which diagnostic --out-dir results/t3_r3/diagnostic/eval --workers 2
step analysis $PY tools/t3/t3r3_analysis.py --root results/t3_r3 --dev-eval results/t3_r3/dev/eval/per_run.csv --diag-eval results/t3_r3/diagnostic/eval/per_run.csv --checks results/t3_r3/dev/launch_checks.json --ledger results/t3_r3/infra_events.json --out-dir results/t3_r3/dev/analysis --workers 16
step blind $PY tools/t3/t3r3_blind.py --per-run results/t3_r3/dev/eval/per_run.csv --ledger results/t3_r3/infra_events.json --headline results/t3_r3/dev/analysis/headline.json --out results/t3_r3/dev/analysis/blind.json
echo done > results/t3_r3/dev/post_launch.done
