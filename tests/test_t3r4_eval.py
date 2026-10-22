"""Tests of the evaluation-problem rule of ``tools/t3/t3r4_eval.py``: an evaluation error or an invalid evaluation is evaluated once more on the
same weights; the re-run replaces the first row only if it is usable; the re-run is recorded."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "ms", ROOT / "tools" / "t3", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import t3r4_eval as ev  # noqa: E402


def _good(job):
    return {"wave": job["wave"], "arm": job["arm"], "q": job["q"], "seed": job["seed"], "T": 3, "run": job["run"], "run_dir": job["run_dir"],
            "kind": "bw", "status": "done", "exit_code": 0, "global_rng_ok": True, "eval_error": "", "valid": True, "dev_valid": True,
            "eta_T_over_dw": 0.001}


def _read(path):
    with open(path, newline="") as f:
        return {r["run"]: r for r in csv.DictReader(f)}


def test_a_failed_evaluation_is_run_again_and_a_usable_rerun_replaces_the_row(tmp_path, monkeypatch):
    calls = {}

    def fake(job):
        n = calls[job["run"]] = calls.get(job["run"], 0) + 1
        row = _good(job)
        if job["run"] == "t3r4_q50_s80503_bwA" and n == 1:                        # raises the first time only
            row.update(eval_error="RuntimeError: boom", valid=False)
        if job["run"] == "t3r4_q50_s80504_bwA" and n == 1:                        # invalid on the development tier the first time only
            row.update(dev_valid=False)
        return row
    monkeypatch.setattr(ev.ce, "safe_evaluate_run", fake)
    res = ev.evaluate_all(tmp_path / "root", tmp_path / "out", workers=1)
    rows = _read(tmp_path / "out" / "per_run.csv")
    assert len(rows) == 80 and all(c == (2 if r in ("t3r4_q50_s80503_bwA", "t3r4_q50_s80504_bwA") else 1) for r, c in calls.items())
    assert rows["t3r4_q50_s80503_bwA"]["eval_error"] == "" and rows["t3r4_q50_s80503_bwA"]["valid"] == "True"
    assert rows["t3r4_q50_s80504_bwA"]["dev_valid"] == "True"
    rec = json.loads((tmp_path / "out" / "eval_reruns.json").read_text())
    assert [r["run"] for r in rec] == ["t3r4_q50_s80503_bwA", "t3r4_q50_s80504_bwA"] and all(r["rerun_succeeded"] for r in rec)
    assert rec[0]["first_eval_error"] == "RuntimeError: boom" and len(res["reruns"]) == 2


def test_a_rerun_that_fails_again_keeps_the_first_row_and_is_recorded(tmp_path, monkeypatch):
    calls = {}

    def fake(job):
        calls[job["run"]] = calls.get(job["run"], 0) + 1
        row = _good(job)
        if job["run"] == "t3r4_q60_s80510_bwA":
            row.update(eval_error=f"ValueError: attempt {calls[job['run']]}", valid=False)
        return row
    monkeypatch.setattr(ev.ce, "safe_evaluate_run", fake)
    ev.evaluate_all(tmp_path / "root", tmp_path / "out", workers=1)
    rows = _read(tmp_path / "out" / "per_run.csv")
    assert calls["t3r4_q60_s80510_bwA"] == 2                                          # run again once, not more
    assert rows["t3r4_q60_s80510_bwA"]["eval_error"] == "ValueError: attempt 1" and rows["t3r4_q60_s80510_bwA"]["valid"] == "False"
    rec = json.loads((tmp_path / "out" / "eval_reruns.json").read_text())
    assert len(rec) == 1 and rec[0]["rerun_succeeded"] is False and rec[0]["rerun_eval_error"] == "ValueError: attempt 2"


def test_a_run_that_did_not_finish_is_not_evaluated_again(tmp_path, monkeypatch):
    calls = {}

    def fake(job):
        calls[job["run"]] = calls.get(job["run"], 0) + 1
        row = _good(job)
        if job["run"] == "t3r4_q50_s80501_bwA":
            row.update(status="failed", exit_code=1, valid=False)
        return row
    monkeypatch.setattr(ev.ce, "safe_evaluate_run", fake)
    ev.evaluate_all(tmp_path / "root", tmp_path / "out", workers=1)
    assert calls["t3r4_q50_s80501_bwA"] == 1 and json.loads((tmp_path / "out" / "eval_reruns.json").read_text()) == []
