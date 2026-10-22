#!/usr/bin/env bash
# Rehearsal of the T3-R4 post-launch tools on 80 symlinked production runs (T3-R3's bwA development runs mapped to seeds 80501-80540).
# The configs, the commit and the plan do not match, so the config / commit / plan verdicts fail by construction; the budgets, schedules,
# C-REPLAY, the evaluation, the analysis and the blind recomputation run on real production-size runs. Not a lock input; not tracked.
set -u
PY=/home/fjiang4/tournament_experiment/.venv/bin/python
R=/home/fjiang4/tournament_experiment/.claude/worktrees/t3-r4-lock-confirm/results/t3_r4/rehearsal/confirmation
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /home/fjiang4/tournament_experiment/.claude/worktrees/t3-r4-lock-confirm
step() { echo "=== $1 $(date -u +%FT%TZ)"; }
step eval
"$PY" tools/t3/t3r4_eval.py --root "$R" --out-dir "$R/eval" --workers 16; echo "exit=$?"
step checks
"$PY" tools/t3/t3r4_checks.py --root "$R" --plan /tmp/claude-1331199693/-home-fjiang4-tournament-experiment--claude-worktrees-t3-bw-failures-diagnosis-25cb13/d068e02a-164b-4c01-950c-16d86f43269c/scratchpad/plan_test3.csv \
    --commit "$(cat "$R/launch_commit.txt")" --lock-dir reports/t3/r4/lock_draft --out "$R/launch_checks.json" --workers 16 > "$R/launch_checks.out.txt"; echo "exit=$?"
step analysis
"$PY" tools/t3/t3r4_analysis.py --root "$R" --eval "$R/eval/per_run.csv" --ledger "$R/infra_events.json" --out-dir "$R/analysis" --workers 16; echo "exit=$?"
step blind
"$PY" tools/t3/t3r4_blind.py --per-run "$R/eval/per_run.csv" --ledger "$R/infra_events.json" --headline "$R/analysis/headline.json" --out "$R/analysis/blind.json" | tail -30; echo "exit=$?"
echo done
