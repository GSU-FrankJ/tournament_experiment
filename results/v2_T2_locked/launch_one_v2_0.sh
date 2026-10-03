#!/bin/bash
# One run of the locked v2.0 entry point: launch_one_v2_0.sh <root relative to the repo> <q> <seed>
# Used for the rehearsal (results/v2_T2_locked/rehearsal_v2_0) and the confirmation (results/v2_T2_locked/confirmation_v2_0).
# The thread environment is set inline for every job (a tmux server that is already running does not pass it on).
# Appends "q<q> s<seed> rc=<exit code>" to <root>/launcher.out.
root=$1
q=$2
s=$3
repo=/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine
d=$repo/$root/q$q/seed$s
mkdir -p "$d"
cd "$repo" || exit 99
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python -B run/run_v2_T2_locked.py --q "$q" --seed "$s" --out-dir "$d" > "$d/run.log" 2>&1
echo "q$q s$s rc=$?" >> "$repo/$root/launcher.out"
