#!/usr/bin/env python3
"""Run `pytest tests` as 12 parallel shards (tmux sessions t3r4_sh0 ... t3r4_sh11) by distributing the test files by size (largest first to
the least loaded shard). Output: results/t3_r4/tests/shard{n}.txt and shard{n}.files. Run from the repository (worktree) root."""
import glob
import os
import subprocess

files = sorted(glob.glob("tests/test_*.py"), key=lambda f: -os.path.getsize(f))
N = 12
load = [0] * N
shards = [[] for _ in range(N)]
for f in files:
    i = load.index(min(load))
    shards[i].append(f)
    load[i] += os.path.getsize(f)
py = "/home/fjiang4/tournament_experiment/.venv/bin/python"
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
open("results/t3_r4/tests/pytest_commit.txt", "w").write(head + "\n")
for i, fs in enumerate(shards):
    open(f"results/t3_r4/tests/shard{i}.files", "w").write("\n".join(fs) + "\n")
    cmd = (f"cd {os.getcwd()} && export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && date -u +%FT%TZ > results/t3_r4/tests/shard{i}.start && "
           f"{py} -m pytest {' '.join(fs)} -q -p no:cacheprovider --durations=5 > results/t3_r4/tests/shard{i}.txt 2>&1; echo exit=$? >> results/t3_r4/tests/shard{i}.txt; "
           f"date -u +%FT%TZ > results/t3_r4/tests/shard{i}.end")
    subprocess.run(["tmux", "new-session", "-d", "-s", f"t3r4_sh{i}", cmd], check=True)
print("started", N, "shards at", head)
