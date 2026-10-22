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


# ---------------------------------------------------------------------------------- run_and_compare on reduced runs with perturbed re-runs
import json  # noqa: E402
import shutil  # noqa: E402

import t3_helpers as th  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """A reduced fired run and a reduced run whose trigger does not fire (budgets 40 / 40 / 20)."""
    out = {}
    for name, delta in (("fired", 1e-9), ("nofire", 1e9)):
        d = tmp_path_factory.mktemp(name) / "orig"
        d.mkdir()
        cfg = {**th.bw_cfg(str(d)), "stage3_retry": {"delta_gt": delta, "max_retries": 1}}
        cfg["run"] = cfg["record"]["run"] = f"t3r4repro_{name}"
        assert rms.run_pipeline(cfg, str(d), "pytest") == 0
        out[name] = d
    return out


def _first_float_tensor(obj):
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point() and obj.numel():
            yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _first_float_tensor(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _first_float_tensor(v)


def _ulp_npz(path):
    z = dict(np.load(path))
    key = next(k for k, v in z.items() if v.dtype.kind == "f" and v.size)
    z[key] = np.nextafter(z[key], np.inf * np.ones_like(z[key]))
    path.unlink()
    np.savez(path, **z)


def _ulp_pt(path):
    obj = torch.load(path, weights_only=False, map_location="cpu")
    t = next(_first_float_tensor(obj))
    t.view(-1)[0] = torch.nextafter(t.view(-1)[0], torch.tensor(float("inf"), dtype=t.dtype))
    torch.save(obj, path)


def _edit_json(path, fn):
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d))


def _fake_rerun(orig, perturb, monkeypatch):
    def run_job(cmd, out_dir, env):
        new = Path(out_dir)
        shutil.copytree(orig, new, dirs_exist_ok=True)
        (new / "run.log").write_text("log\n")
        if perturb is not None:
            perturb(new)
        return {"out": out_dir, "returncode": 0}
    monkeypatch.setattr(rp.L1, "run_job", run_job)


def _compare(runs, name, perturb, tmp_path, monkeypatch, expected_npz=None):
    orig = runs[name]
    _fake_rerun(orig, perturb, monkeypatch)
    job = {"q": 50, "seed": 50500, "fired": name == "fired", "orig": orig, "new": tmp_path / "q50" / "seed50500" / "bwA"}
    if expected_npz is not None:
        job["expected_npz"] = expected_npz
    return rp.run_and_compare(job)


def test_an_identical_rerun_is_ok_and_the_npz_count_is_pinned(runs, tmp_path, monkeypatch):
    rec = _compare(runs, "fired", None, tmp_path / "a", monkeypatch, expected_npz=1)               # the production count (162) is not met by a reduced run
    assert rec["ok"] is False and rec["compare_runs_returncode"] == 0 and rec["n_npz"] != 1
    n = rec["n_npz"]
    rec = _compare(runs, "fired", None, tmp_path / "b", monkeypatch, expected_npz=n)
    assert rec["ok"] is True and rec["attempt1"]["ok"] and rec["retry_json_diff"] == [] and all(rec["states_equal"].values())
    assert not any(rec["json_diffs"].values()) and rec["file_sets_equal"] and rec["fired_in_rerun"] is True
    assert (tmp_path / "b" / "compare_q50_s50500.json").is_file() and (tmp_path / "b" / "compare_q50_s50500.txt").is_file()
    rec = _compare(runs, "nofire", None, tmp_path / "c", monkeypatch, expected_npz=n)
    assert rec["ok"] is True and "attempt1" not in rec and rec["fired_in_rerun"] is False


PERTURBATIONS = {
    "an export of the main run": lambda d: _ulp_npz(d / "weights" / "u00025.npz"),
    "a freeze array": lambda d: _ulp_npz(d / "freeze_stage1_final.npz"),
    "an export of attempt 1": lambda d: _ulp_npz(d / "attempt1" / "weights" / "u00025.npz"),
    "the state of attempt 1": lambda d: _ulp_pt(d / "attempt1" / "state_attempt1_end.pt"),
    "the stage-3 end state": lambda d: _ulp_pt(d / "state_end_stage3.pt"),
    "the stage-2 end state": lambda d: _ulp_pt(d / "state_end_stage2.pt"),
    "retry.json (a trigger value)": lambda d: _edit_json(d / "retry.json", lambda j: j["attempt1"]["trigger_check"].__setitem__("Delta", 0.5)),
    "retry.json (a derived stream)": lambda d: _edit_json(d / "retry.json", lambda j: j["attempt2"]["entropy"]["init"].__setitem__(0, 1)),
    "the attempt-1 record": lambda d: _edit_json(d / "attempt1" / "attempt1_record.json", lambda j: j["counts"].__setitem__("updates", 1)),
    "gates.json": lambda d: _edit_json(d / "gates.json", lambda j: j.__setitem__("gate_values_changed", 1)),
    "the manifest (init digest)": lambda d: _edit_json(d / "manifest.json", lambda j: j.__setitem__("init_state_sha256", "0" * 64)),
    "the train history": lambda d: _edit_json(d / "train_history.json", lambda j: j["history"][0].__setitem__("policy_loss", 123.0)),
    "the attempt-1 train history": lambda d: _edit_json(d / "attempt1" / "train_history_stage3.json",
                                                        lambda j: j["history"][0].__setitem__("policy_loss", 123.0)),
    "an extra file": lambda d: (d / "extra.txt").write_text("x"),
    "a missing file": lambda d: (d / "drift_test.json").unlink(),
}

HARMLESS = {
    "wall-clock seconds in retry.json": lambda d: _edit_json(d / "retry.json", lambda j: j["attempt1"]["counts"].__setitem__("wall_sec", 1.0)),
    "the commit and the command line of the manifest": lambda d: _edit_json(d / "manifest.json",
                                                                          lambda j: (j.__setitem__("commit", "x"), j.__setitem__("cmd", "y"))),
}


def test_every_perturbation_of_a_rerun_is_caught_and_the_harmless_ones_are_not(runs, tmp_path, monkeypatch):
    n = _compare(runs, "fired", None, tmp_path / "base", monkeypatch)["n_npz"]
    for i, (name, perturb) in enumerate(PERTURBATIONS.items()):
        rec = _compare(runs, "fired", perturb, tmp_path / f"p{i}", monkeypatch, expected_npz=n)
        assert rec["ok"] is False, name
    for i, (name, perturb) in enumerate(HARMLESS.items()):
        assert _compare(runs, "fired", perturb, tmp_path / f"h{i}", monkeypatch, expected_npz=n)["ok"] is True, name


def test_a_rerun_that_fires_when_the_original_did_not_or_the_reverse_is_caught(runs, tmp_path, monkeypatch):
    n = _compare(runs, "nofire", None, tmp_path / "base", monkeypatch)["n_npz"]
    rec = _compare(runs, "nofire", lambda d: shutil.copy(runs["fired"] / "retry.json", d / "retry.json"), tmp_path / "x", monkeypatch, expected_npz=n)
    assert rec["ok"] is False and rec["retry_state_matches"] is False
    rec = _compare(runs, "fired", lambda d: (d / "retry.json").unlink(), tmp_path / "y", monkeypatch, expected_npz=n)
    assert rec["ok"] is False and rec["retry_state_matches"] is False


def test_a_rerun_that_exits_nonzero_is_not_ok(runs, tmp_path, monkeypatch):
    monkeypatch.setattr(rp.L1, "run_job", lambda cmd, out_dir, env: {"out": out_dir, "returncode": 1})
    rec = rp.run_and_compare({"q": 50, "seed": 50500, "fired": True, "orig": runs["fired"], "new": tmp_path / "q50" / "seed50500" / "bwA"})
    assert rec["ok"] is False and rec["returncode"] == 1


def test_bitwise_comparison_distinguishes_signed_zero_and_nan_payloads():
    assert not rp.arrays_equal(np.array([0.0]), np.array([-0.0]))
    a = np.array([np.nan])
    b = a.view(np.uint64) | np.uint64(1)
    assert rp.arrays_equal(a, a.copy()) and not rp.arrays_equal(a, b.view(np.float64))
    t0, t1 = torch.tensor([0.0]), torch.tensor([-0.0])
    assert not rp.tree_equal({"w": t0}, {"w": t1}) and rp.tree_equal({"w": t0}, {"w": t0.clone()})
    assert rp.tree_equal({"x": float("nan")}, {"x": float("nan")})
