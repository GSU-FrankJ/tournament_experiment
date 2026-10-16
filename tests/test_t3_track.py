"""T3-R1: tests of ``tools/t3/t3_track.py`` (what the tracking rules allow) and ``tools/t3/compare_runs.py``
(bit-for-bit comparison of two runs of one config).

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_t3_track.py -p no:cacheprovider -q
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import t3_helpers as H
import t3_track as tt  # noqa: E402

BW_TRACKED = {"manifest.json", "run_config.json", "status.json", "gates.json", "rule_log.json", "drift_test.json",
              "ms_run_summary.json", "ms_checks_stage3.csv", "ms_checks_stage1.csv", "ms_binmaps_stage2.npz", "ms_updates.csv",
              "continuation_table_stage1.npz"}
BW_UNTRACKED = {"train_history.json", "state_end_stage1.pt", "freeze_stage1_final.npz", "freeze_stage3_development.npz",
                "weights/u00025.npz", "run.log"}
JOINT_TRACKED = {"run_config.json", "manifest.json", "status.json", "joint_run_summary.json", "joint_checks.csv",
                 "joint_updates.csv", "visitation.npz"}
JOINT_UNTRACKED = {"state_end.pt", "final_eval_final.npz", "final_eval_development.npz", "weights/u00025.npz", "run.log"}


def touch(p: Path, text: str = "x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def build_tree(root: Path) -> None:
    bw = root / "runs" / "q50" / "seed50501" / "bw"
    jt = root / "runs" / "q50" / "seed50501" / "jt"
    for name in BW_TRACKED | BW_UNTRACKED:
        touch(bw / name, json.dumps({"schema": "ms_run_config/1"}) if name == "run_config.json" else "x")
    for name in JOINT_TRACKED | JOINT_UNTRACKED:
        touch(jt / name, json.dumps({"schema": "joint_run_config/1"}) if name == "run_config.json" else "x")
    touch(jt / "extra_note.txt")                                   # not in any tracked class of a run directory
    touch(root / "launch_checks.json")
    touch(root / "analysis" / "criterion.csv")
    touch(root / "analysis" / "figures" / "paired.png")
    touch(root / "eval" / "per_run.csv")
    touch(root / "eval" / "arrays" / "run_a.npz")
    touch(root / "preflight" / "machine.txt")
    touch(root / "preflight" / "seed_hits_1.csv")
    touch(root / "preflight" / "debug.log")
    touch(root / "preflight" / "scratch.pt")


def test_only_the_classes_of_the_tracking_rules_are_listed(tmp_path):
    build_tree(tmp_path)
    got = {str(p.relative_to(tmp_path)) for p in tt.files_to_track(tmp_path)}
    bw = {f"runs/q50/seed50501/bw/{n}" for n in BW_TRACKED}
    jt = {f"runs/q50/seed50501/jt/{n}" for n in JOINT_TRACKED}
    other = {"launch_checks.json", "analysis/criterion.csv", "analysis/figures/paired.png", "eval/per_run.csv",
             "preflight/machine.txt", "preflight/seed_hits_1.csv"}
    assert got == bw | jt | other
    for n in BW_UNTRACKED:
        assert f"runs/q50/seed50501/bw/{n}" not in got
    for n in JOINT_UNTRACKED:
        assert f"runs/q50/seed50501/jt/{n}" not in got
    assert not any(g.endswith((".pt", ".log")) or "/weights/" in g or "arrays/" in g for g in got)


def test_the_real_smoke_directory_never_lists_weights_or_arrays():
    smoke = H.ROOT / "results" / "t3_r1" / "smoke"
    if not smoke.exists():
        pytest.skip("no smoke wave in this clone")
    for p in tt.files_to_track(smoke):
        s = str(p)
        assert not (s.endswith((".pt", ".log")) or "/weights/" in s or "final_eval_" in p.name or p.name.startswith("freeze_stage")
                    or p.name == "train_history.json"), s


def test_compare_runs_flags_a_changed_array_and_csv_but_not_wall_clock(tmp_path):
    runs = H.reduced_runs()
    src = Path(runs["t2_root"][0])
    a, b = tmp_path / "a", tmp_path / "b"
    shutil.copytree(src, a, ignore=shutil.ignore_patterns("ref_ms"))
    shutil.copytree(src, b, ignore=shutil.ignore_patterns("ref_ms"))
    script = str(H.ROOT / "tools" / "t3" / "compare_runs.py")

    def run() -> int:
        return subprocess.run([sys.executable, script, "joint", str(a), str(b)], capture_output=True, text=True).returncode
    assert run() == 0
    # the wall-clock columns may differ
    t = (b / "joint_updates.csv").read_text().splitlines()
    head = t[0].split(",")
    i = head.index("update_wall_sec")
    row = t[1].split(",")
    row[i] = "123.456"
    t[1] = ",".join(row)
    (b / "joint_updates.csv").write_text("\n".join(t) + "\n")
    assert run() == 0
    # a changed weight export is a difference
    w = dict(np.load(b / "weights" / "u00010.npz"))
    k = next(iter(w))
    w[k] = w[k] + np.float32(1e-7)
    np.savez(b / "weights" / "u00010.npz", **w)
    assert run() == 1
    # a file present in one run only is a difference, whichever run has it
    shutil.copy(a / "weights" / "u00010.npz", b / "weights" / "u00010.npz")
    assert run() == 0
    shutil.copy(a / "weights" / "u00010.npz", b / "weights" / "u00099.npz")
    assert run() == 1
    (b / "weights" / "u00099.npz").unlink()
    (b / "weights" / "u00020.npz").unlink()
    assert run() == 1
