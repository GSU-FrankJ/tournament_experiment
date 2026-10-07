"""Post-launch checks (tools/ms/launch_checks.py) on a tiny genuine run and on tampered copies of it."""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import launch_checks as LC  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402
from test_ms_runner import make_cfg, small_params  # noqa: E402

FAKE_GIT = {"commit": "abc1234", "short": "abc1234", "dirty": False}


@pytest.fixture(scope="module")
def genuine(tmp_path_factory):
    """A tiny T = 2 MS_s25a0 run under <root>/q50/seed10501/MS_s25a0 (git state patched to a clean commit)."""
    root = tmp_path_factory.mktemp("wave")
    d = root / "q50" / "seed10501" / "MS_s25a0"
    d.mkdir(parents=True)
    cfg = make_cfg(str(d), arm="MS_s25a0", params=small_params(2, K=10, block=20, cap=40, land=10), epu=64)
    mp = pytest.MonkeyPatch()
    mp.setattr(rms, "git_state", lambda: dict(FAKE_GIT))
    try:
        assert rms.run_pipeline(cfg, str(d), "pytest", band_step=2.0) == 0
    finally:
        mp.undo()
    return root


def _copy(genuine, tmp_path):
    dst = tmp_path / "w"
    shutil.copytree(genuine, dst)
    return dst, dst / "q50" / "seed10501" / "MS_s25a0"


def _run(root):
    return LC.check_root(root, ["MS_s25a0"], [50], [10501], "abc1234")


def test_a_genuine_run_passes_every_check(genuine):
    r = _run(genuine)
    run = r["runs"][0]
    assert r["all_ok"], json.dumps(run, indent=1)[:2000]
    assert run["status"]["ok"] and run["manifest"]["ok"] and run["files"]["ok"] and run["global_rng"]["ok"]
    assert run["tail_share"]["coverage_ok"] and run["tail_share"]["tail_mass_max_deviation_from_lambda_T"] <= 1e-12
    assert run["start_shares"]["n_tests"] >= 9      # blocks + landing, three strata each
    assert run["start_shares"]["max_abs_z"] < 5.0


def test_a_wrong_commit_or_a_dirty_manifest_fails(genuine, tmp_path):
    root, d = _copy(genuine, tmp_path)
    assert LC.check_root(root, ["MS_s25a0"], [50], [10501], "deadbee")["all_ok"] is False
    man = json.load(open(d / "manifest.json"))
    man["clean_tree"] = False
    json.dump(man, open(d / "manifest.json", "w"))
    r = _run(root)
    assert r["runs"][0]["manifest"]["ok"] is False and r["all_ok"] is False


def test_a_missing_weight_export_or_table_fails(genuine, tmp_path):
    root, d = _copy(genuine, tmp_path)
    os.remove(d / "weights" / "u00050.npz")
    r = _run(root)
    assert r["runs"][0]["files"]["ok"] is False and any("weights" in m for m in r["runs"][0]["files"]["missing"])
    root2, d2 = _copy(genuine, tmp_path / "b")
    os.remove(d2 / "ms_binmaps_stage2.npz")
    assert _run(root2)["all_ok"] is False


def test_a_global_rng_violation_and_a_failed_run_are_reported(genuine, tmp_path):
    root, d = _copy(genuine, tmp_path)
    g = json.load(open(d / "gates.json"))
    g["global_rng"] = {"status": "violation", "violations": [{"rng": "torch_global", "point": "end_of_stage2"}]}
    json.dump(g, open(d / "gates.json", "w"))
    assert _run(root)["runs"][0]["global_rng"]["ok"] is False
    root2, d2 = _copy(genuine, tmp_path / "c")
    st = json.load(open(d2 / "status.json"))
    st.update(state="failed", exit_code=1)
    json.dump(st, open(d2 / "status.json", "w"))
    r = _run(root2)
    assert r["runs"][0]["status"]["ok"] is False and r["all_ok"] is False


def test_a_missing_run_is_reported_not_skipped(genuine):
    r = LC.check_root(genuine, ["MS_s25a0", "MS_rule"], [50], [10501], "abc1234")
    assert r["n_runs"] == 2 and r["runs"][1]["status"]["error"] == "missing" and r["all_ok"] is False


def test_a_tampered_probability_vector_breaks_the_coverage_check(genuine, tmp_path):
    root, d = _copy(genuine, tmp_path)
    z = dict(np.load(d / "ms_binmaps_stage2.npz"))
    pt = z["probs_table"].copy()
    nb = pt.shape[1]
    pt[0, : nb // 4] += 0.01            # mass moved into the tail side, tail share != lambda_T
    pt[0] /= pt[0].sum()
    z["probs_table"] = pt
    np.savez(d / "ms_binmaps_stage2.npz", **z)
    r = _run(root)
    assert r["runs"][0]["tail_share"]["coverage_ok"] is False and r["all_ok"] is False


def test_counts_far_from_the_design_are_flagged(genuine, tmp_path):
    import csv
    root, d = _copy(genuine, tmp_path)
    rows = list(csv.DictReader(open(d / "ms_updates.csv")))
    for r in rows:
        if r["stage"] == "2":
            n = int(r["n_start_tail"]) + int(r["n_start_near"]) + int(r["n_start_mid"])
            r["n_start_tail"], r["n_start_near"], r["n_start_mid"] = str(n), "0", "0"
    with open(d / "ms_updates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    r = _run(root)
    assert r["runs"][0]["start_shares"]["n_flagged_abs_z_gt_3"] > 0
