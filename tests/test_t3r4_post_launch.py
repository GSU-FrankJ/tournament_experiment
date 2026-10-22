"""Tests of ``tools/t3/t3r4_post_launch.sh`` (absolute root, stop at the first nonzero exit) and of the job list of ``t3r4_eval.py``."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "ms", ROOT / "tools" / "t3", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import t3r4_eval as ev  # noqa: E402

SCRIPT = ROOT / "tools" / "t3" / "t3r4_post_launch.sh"


def test_the_script_refuses_a_relative_root():
    r = subprocess.run(["bash", str(SCRIPT), "results/t3_r4/confirmation"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2 and "absolute" in r.stderr and r.stdout == ""


def test_the_script_stops_at_the_first_failing_step(tmp_path):
    (tmp_path / "launch_commit.txt").write_text("abc\n")
    r = subprocess.run(["bash", str(SCRIPT), str(tmp_path)], capture_output=True, text=True, cwd=ROOT, env={**os.environ, "T3R4_PY": "false"})
    assert r.returncode != 0 and "=== checks" in r.stdout
    assert "=== eval" not in r.stdout and "=== analysis" not in r.stdout and "=== blind" not in r.stdout and "done" not in r.stdout
    assert not (tmp_path / "eval").exists() and not (tmp_path / "analysis").exists()


def test_the_script_runs_the_four_steps_in_order_and_needs_their_outputs(tmp_path):
    (tmp_path / "launch_commit.txt").write_text("abc\n")
    # every step exits 0 but writes nothing: the final check of the outputs fails, the script does not say "done"
    r = subprocess.run(["bash", str(SCRIPT), str(tmp_path)], capture_output=True, text=True, cwd=ROOT, env={**os.environ, "T3R4_PY": "true"})
    steps = [line.split()[1] for line in r.stdout.splitlines() if line.startswith("=== ")]
    assert steps == ["checks", "eval", "analysis", "blind"] and r.returncode == 1 and "done" not in r.stdout and "missing or empty output" in r.stderr
    # with every output present the script ends with "done"
    (tmp_path / "eval").mkdir()
    (tmp_path / "analysis").mkdir()
    for f in ("launch_checks.json", "eval/per_run.csv", "analysis/headline.json", "analysis/blind.json"):
        (tmp_path / f).write_text("x\n")
    r = subprocess.run(["bash", str(SCRIPT), str(tmp_path)], capture_output=True, text=True, cwd=ROOT, env={**os.environ, "T3R4_PY": "true"})
    assert r.returncode == 0 and "done" in r.stdout


def test_the_script_names_the_launch_commit_the_plan_and_the_tools():
    text = SCRIPT.read_text()
    for needle in ("set -euo pipefail", "launch_commit.txt", "reports/t3/r4/confirmation_plan.csv", "t3r4_checks.py", "t3r4_eval.py",
                   "t3r4_analysis.py", "t3r4_blind.py", "infra_events.json", "--headline", "--checks", "--ledger", "--commit"):
        assert needle in text, needle


def test_the_evaluation_has_80_jobs_with_an_arrays_directory(tmp_path):
    jobs = ev.plan_jobs(tmp_path, tmp_path / "arrays")
    assert len(jobs) == 80 and all(j["arrays_dir"] == str(tmp_path / "arrays") and j["kind"] == "bw" and j["T"] == 3 for j in jobs)
    assert jobs[0]["run_dir"] == str(tmp_path / "q50" / "seed80501" / "bwA")
