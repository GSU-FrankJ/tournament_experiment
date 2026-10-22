#!/usr/bin/env bash
# T3-R4 P2: the launch of the 80-run confirmation wave of protocol T3-1.1, from the LOCK-record commit, in tmux (session t3r4_conf).
# The launch commit, the machine state (nproc, load, free disk) and `git status --porcelain` were recorded beforehand in
# results/t3_r4/confirmation/launch_commit.txt and launch_machine.txt. The launcher checks every guard itself and refuses otherwise.
# Nothing is written outside results/ from here until every run has finished.
set -u
W=/home/fjiang4/tournament_experiment/.claude/worktrees/t3-r4-lock-confirm
R=$W/results/t3_r4/confirmation
cd "$W"
date -u +%FT%TZ > "$R/launch_start.txt"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/home/fjiang4/tournament_experiment/.venv/bin/python tools/t3/t3r4_launch.py --root "$R" --workers 40 --code-commit "$(cat "$R/launch_commit.txt")" > "$R/launch_stdout.txt" 2>&1
echo "exit=$?" >> "$R/launch_stdout.txt"
date -u +%FT%TZ > "$R/launch_end.txt"
