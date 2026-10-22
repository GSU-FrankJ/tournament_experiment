"""Tests of ``tools/t3/t3r4_repro.py`` (C-REPRO): the comparison helpers and the job list."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "ms", ROOT / "tools" / "t3", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import t3r4_repro as rp  # noqa: E402

NAN = float("nan")


def test_the_two_reruns_are_the_fired_and_the_unfired_bwa_runs_of_t3r3():
    js = rp.jobs(Path("/r3"), Path("/out"))
    assert [(j["q"], j["seed"], j["fired"]) for j in js] == [(50, 70508, True), (60, 70501, False)]
    assert js[0]["orig"] == Path("/r3/dev/q50/seed70508/bwA") and js[1]["new"] == Path("/out/q60/seed70501/bwA")


def test_json_diff_reports_differences_and_skips_wall_clock_commit_and_tree_flag():
    a = {"x": 1, "commit": "a", "clean_tree": False, "wall_sec": 1.0, "n": {"cpu_sec": 2.0, "v": [1, 2, {"k": NAN}]}, "only_a": 1}
    b = {"x": 1, "commit": "b", "clean_tree": True, "wall_sec": 9.0, "n": {"cpu_sec": 3.0, "v": [1, 2, {"k": NAN}]}}
    assert rp.json_diff(a, b) == ["/only_a (missing)"]
    assert rp.json_diff({"x": 1}, {"x": 2}) == ["/x"]
    assert rp.json_diff({"v": [1, 2]}, {"v": [1, 3]}) == ["/v[1]"] and rp.json_diff({"v": [1, 2]}, {"v": [1, 2, 3]}) == ["/v"]
    assert rp.json_diff({"x": NAN}, {"x": 1.0}) == ["/x"]


def test_arrays_and_trees_are_compared_bitwise():
    a = np.array([1.0, NAN])
    assert rp.arrays_equal(a, a.copy()) and not rp.arrays_equal(a, np.array([1.0, 2.0])) and not rp.arrays_equal(a, a.astype(np.float32))
    t1 = {"w": torch.tensor([1.0, 2.0]), "n": [1, (2, 3)]}
    t2 = {"w": torch.tensor([1.0, 2.0]), "n": [1, (2, 3)]}
    assert rp.tree_equal(t1, t2)
    t2["w"][1] = torch.nextafter(t2["w"][1], torch.tensor(10.0))
    assert not rp.tree_equal(t1, t2) and not rp.tree_equal({"a": 1}, {"a": 1, "b": 2})


def test_file_sets_ignore_the_run_log(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.json").write_text("1")
    (tmp_path / "run.log").write_text("log")
    assert rp.files_of(tmp_path) == ["a/x.json"]


def test_the_tool_refuses_relative_paths(tmp_path):
    for argv in (["--t3r3-root", "results/t3_r3", "--out", str(tmp_path / "o")], ["--t3r3-root", str(tmp_path), "--out", "results/t3_r4/repro"]):
        with pytest.raises(SystemExit) as exc:
            rp.main(argv)
        assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:                                      # a missing original is a stop
        rp.main(["--t3r3-root", str(tmp_path), "--out", str(tmp_path / "o")])
    assert exc.value.code == 2 and not (tmp_path / "o").exists()
