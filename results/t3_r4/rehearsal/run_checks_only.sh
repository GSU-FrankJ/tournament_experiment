#!/usr/bin/env bash
# Rehearsal, checks only (after the draft lock was regenerated).
set -u
PY=/home/fjiang4/tournament_experiment/.venv/bin/python
R=/home/fjiang4/tournament_experiment/.claude/worktrees/t3-r4-lock-confirm/results/t3_r4/rehearsal/confirmation
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /home/fjiang4/tournament_experiment/.claude/worktrees/t3-r4-lock-confirm
"$PY" tools/t3/t3r4_checks.py --root "$R" --plan /tmp/claude-1331199693/-home-fjiang4-tournament-experiment--claude-worktrees-t3-bw-failures-diagnosis-25cb13/d068e02a-164b-4c01-950c-16d86f43269c/scratchpad/plan_test3.csv \
    --commit "$(cat "$R/launch_commit.txt")" --lock-dir reports/t3/r4/lock_draft --out "$R/launch_checks.json" --workers 16 > "$R/launch_checks.out.txt"; echo "exit=$?"
