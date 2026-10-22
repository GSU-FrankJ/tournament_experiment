#!/usr/bin/env bash
# Rehearsal of the report tables on the rehearsal outputs (written to results/t3_r4/rehearsal/tables, not to reports/).
set -u
cd /home/fjiang4/tournament_experiment/.claude/worktrees/t3-r4-lock-confirm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export T3R4_CONF=/home/fjiang4/tournament_experiment/.claude/worktrees/t3-r4-lock-confirm/results/t3_r4/rehearsal/confirmation
export T3R4_TABLES_OUT=/home/fjiang4/tournament_experiment/.claude/worktrees/t3-r4-lock-confirm/results/t3_r4/rehearsal/tables
/home/fjiang4/tournament_experiment/.venv/bin/python -B reports/t3/r4/report_scripts/confirmation_tables.py
