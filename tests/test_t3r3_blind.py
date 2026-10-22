"""Tests of ``tools/t3/t3r3_blind.py`` (and that it does not import the analysis tools)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "t3"):
    sys.path.insert(0, str(_p))

import t3r3_blind as bl  # noqa: E402

COLS = ["run", "status", "exit_code", "global_rng_ok", "eval_error", "valid", "dev_valid", "eta_T_over_dw", "rmse_pos_T_over_eT0",
        "tail_mean_T_over_eT0", "Gmax_full_over_dw", "dev_eta_T_over_dw", "dev_Gmax_full_over_dw"]


def _row(arm, q, s, **kw):
    r = {"run": bl.run_name(arm, q, s), "status": "done", "exit_code": "0", "global_rng_ok": "True", "eval_error": "", "valid": "True",
         "dev_valid": "True", "eta_T_over_dw": "0.002", "rmse_pos_T_over_eT0": "0.03", "tail_mean_T_over_eT0": "0.004",
         "Gmax_full_over_dw": "0.002", "dev_eta_T_over_dw": "0.0021", "dev_Gmax_full_over_dw": "0.0021"}
    r.update({k: str(v) for k, v in kw.items()})
    return r


def _write(tmp, rows, ledger=None):
    p, lg = tmp / "per_run.csv", tmp / "infra_events.json"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    lg.write_text(json.dumps(ledger or {"events": [], "stopped_without_verdict": []}))
    return p, lg


def _all(fail=None, **kw):
    fail = fail or {}
    rows = []
    for arm in bl.ARMS:
        for q in bl.QS:
            for s in bl.SEEDS:
                rows.append(_row(arm, q, s, **(fail.get((arm, q, s), {}))))
    return rows


def test_the_blind_tool_imports_no_analysis_tool():
    src = Path(bl.__file__).read_text()
    for name in ("t3r3_analysis", "t3c_analysis", "t3_analysis"):
        assert f"import {name}" not in src and f"from {name}" not in src


def test_pass_rule_is_inclusive_and_counts_a_missing_row_as_failed(tmp_path):
    g = json.loads(bl.CRITERIA.read_text())["gates"]
    assert bl.passes(_row("bw", 50, 70501, eta_T_over_dw=0.005, dev_eta_T_over_dw=0.006), g)                    # <= is inclusive
    assert not bl.passes(_row("bw", 50, 70501, eta_T_over_dw=0.0050000001), g)
    assert bl.passes(_row("bw", 50, 70501, rmse_pos_T_over_eT0=0.05, tail_mean_T_over_eT0=0.02, Gmax_full_over_dw=0.01, dev_Gmax_full_over_dw=0.011), g)
    assert not bl.passes(_row("bw", 50, 70501, Gmax_full_over_dw=0.0100001), g)
    assert not bl.passes(_row("bw", 50, 70501, dev_eta_T_over_dw=0.0031), g)                                      # |dev - final| > 0.001
    assert not bl.passes(_row("bw", 50, 70501, global_rng_ok="False"), g) and not bl.passes(_row("bw", 50, 70501, exit_code="1"), g)
    assert not bl.passes(_row("bw", 50, 70501, valid="False"), g) and not bl.passes(_row("bw", 50, 70501, eval_error="DomainError: x"), g)
    assert not bl.passes(_row("bw", 50, 70501, eta_T_over_dw="nan"), g) and not bl.passes(None, g)


def test_counts_target_and_selection(tmp_path):
    fail = {("bwA", 50, 70501): {"exit_code": 1}, ("bwA", 60, 70502): {"valid": "False"}, ("bwAB", 60, 70501): {"exit_code": 1},
            ("bwAB", 60, 70502): {"exit_code": 1}, ("bwAB", 60, 70503): {"exit_code": 1}, ("bw", 50, 70501): {"exit_code": 1}}
    p, lg = _write(tmp_path, _all(fail))
    res = bl.recompute(p, lg)
    a = res["arms"]
    assert (a["bwA"]["pass_q50"], a["bwA"]["pass_q60"], a["bwA"]["pass_total"]) == (19, 19, 38) and a["bwA"]["reaches_target"]
    assert a["bwAB"]["pass_total"] == 37 and not a["bwAB"]["reaches_target"] and a["bw"]["pass_total"] == 39 and a["bw"]["reaches_target"]
    assert res["proposed"] == "bwA" and not res["both_reach_target"]
    assert a["bwA"]["ci_lo_total"] < 38 / 40 < a["bwA"]["ci_hi_total"]


def test_per_q_condition_and_a_missing_run(tmp_path):
    fail = {("bwA", 60, 70500 + i): {"exit_code": 1} for i in (1, 2, 3)}                 # 17 of 20 at q = 60, 37 of 40
    rows = [r for r in _all(fail)]
    rows = [r for r in rows if r["run"] != bl.run_name("bwAB", 50, 70501)]               # a planned run without a row fails
    p, lg = _write(tmp_path, rows)
    a = bl.recompute(p, lg)["arms"]
    assert not a["bwA"]["reaches_target"] and a["bwAB"]["pass_q50"] == 19 and a["bwAB"]["reaches_target"]
    assert bl.recompute(p, lg)["proposed"] == "bwAB"


def test_ledger_stops_count_as_failures_and_a_missing_ledger_is_an_error(tmp_path):
    p, lg = _write(tmp_path, _all(), {"events": [{"run": bl.run_name("bw", 50, 70501), "attempt": 2}], "stopped_without_verdict": [bl.run_name("bwA", 60, 70520)]})
    res = bl.recompute(p, lg)
    assert res["arms"]["bw"]["pass_q50"] == 19 and res["arms"]["bwA"]["pass_q60"] == 19
    with pytest.raises(FileNotFoundError):
        bl.recompute(p, tmp_path / "missing.json")


def test_compare_with_a_headline(tmp_path):
    p, lg = _write(tmp_path, _all())
    res = bl.recompute(p, lg)
    head = {"counts": {a: dict(res["arms"][a]) for a in bl.ARMS}, "targets": {a: {"reaches_target": res["arms"][a]["reaches_target"]} for a in bl.ARMS},
            "selection": {"proposed": res["proposed"]}}
    ok, diffs = bl.compare(res, head)
    assert ok and diffs == []
    head["counts"]["bwA"]["pass_q50"] = 19
    head["selection"]["proposed"] = "bwAB"
    ok, diffs = bl.compare(res, head)
    assert not ok and len(diffs) == 2


def test_the_total_threshold_alone_decides_with_both_per_q_conditions_met(tmp_path):
    def world(n_fail_q50, n_fail_q60):
        fail = {}
        for i in range(n_fail_q50):
            fail[("bwA", 50, 70501 + i)] = {"exit_code": 1}
        for i in range(n_fail_q60):
            fail[("bwA", 60, 70501 + i)] = {"exit_code": 1}
        return fail
    p, lg = _write(tmp_path, _all(world(1, 2)))                    # 37 of 40: 19 and 18 per q
    a = bl.recompute(p, lg)["arms"]["bwA"]
    assert a["pass_total"] == 37 and a["pass_q50"] == 19 and a["pass_q60"] == 18 and not a["reaches_target"]
    (tmp_path / "b").mkdir()
    p, lg = _write(tmp_path / "b", _all(world(0, 2)))                                                          # 38 of 40
    a = bl.recompute(p, lg)["arms"]["bwA"]
    assert a["pass_total"] == 38 and a["reaches_target"]
