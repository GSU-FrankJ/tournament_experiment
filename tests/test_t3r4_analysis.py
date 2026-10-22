"""Tests of ``tools/t3/t3r4_analysis.py`` and ``tools/t3/t3r4_blind.py`` on synthetic tables with known verdicts: the rule (at most 6 failures
in all and at most 4 at each q), the failure counting (an evaluation error is a failed run; a second infrastructure failure is a stop
without a verdict), the agreement of the two tools, the classes and the descriptive comparisons."""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "ms", ROOT / "tools" / "t3", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import t3c_analysis as ca  # noqa: E402
import t3r4_analysis as ta  # noqa: E402
import t3r4_blind as bl  # noqa: E402

GATES = json.loads((ROOT / "reports" / "t3" / "r1" / "prereg_criteria.json").read_text())["gates"]
SEEDS = range(80501, 80541)


def eval_row(q, seed, **over):
    r = {"run": f"t3r4_q{q}_s{seed}_bwA", "q": q, "seed": seed, "status": "done", "exit_code": 0, "global_rng_ok": True, "eval_error": "",
         "valid": True, "dev_valid": True, "eta_T_over_dw": 0.002, "rmse_pos_T_over_eT0": 0.03, "tail_mean_T_over_eT0": 0.005,
         "Gmax_full_over_dw": 0.002, "dev_eta_T_over_dw": 0.0021, "dev_Gmax_full_over_dw": 0.0021}
    r.update(over)
    return r


FAIL = {"eta_T_over_dw": 0.02, "Gmax_full_over_dw": 0.02}                 # fails G-A and G-F


def table(fail50=(), fail60=(), eval_error50=(), eval_error60=()):
    """80 evaluation rows; the listed seed indices (0..39) fail by the gates or by an evaluation error."""
    rows = []
    for q, fails, errs in ((50, fail50, eval_error50), (60, fail60, eval_error60)):
        for i, s in enumerate(SEEDS):
            if i in errs:
                rows.append(eval_row(q, s, valid=False, eval_error="boom"))
            else:
                rows.append(eval_row(q, s, **(FAIL if i in fails else {})))
    return rows


def write(tmp_path, rows, ledger=None):
    p = tmp_path / "per_run.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lp = tmp_path / "infra_events.json"
    lp.write_text(json.dumps(ledger or {"schema": "x", "events": [], "stopped_without_verdict": []}))
    return p, lp


def merged(rows, stopped=()):
    """The analysis side: pass / fail by ``t3c_analysis.run_row`` joined to the plan (q, run)."""
    out = []
    for r in rows:
        v = ca.run_row(r, GATES)
        if r["run"] in stopped:
            v["passed"] = False
        out.append({"q": r["q"], "run": r["run"], "passed": v["passed"]})
    return out


# (fail at q=50, fail at q=60, verdict): the cases of the prompt (3+3 pass, 4+3 fail, 5+0 fail, 4+2 pass) and the boundaries
CASES = [((0, 1, 2), (0, 1, 2), "pass"), ((0, 1, 2, 3), (0, 1, 2), "fail"), ((0, 1, 2, 3, 4), (), "fail"), ((0, 1, 2, 3), (0, 1), "pass"),
         ((), (), "pass"), ((0, 1, 2, 3), (0, 1, 2, 3), "fail"), ((), (0, 1, 2, 3, 4), "fail"), ((0, 1, 2, 3), (0, 1), "pass"),
         ((0,), (1, 2, 3, 4, 5, 6), "fail"), ((0, 1, 2, 3, 4, 5), (), "fail")]


@pytest.mark.parametrize("f50,f60,want", CASES)
def test_the_rule_in_both_tools(tmp_path, f50, f60, want):
    rows = table(f50, f60)
    p, lp = write(tmp_path, rows)
    blind = bl.recompute(p, lp)
    assert blind["verdict"] == want and blind["fail_q50"] == len(f50) and blind["fail_q60"] == len(f60)
    cnt = ta.counts_of(merged(rows))
    v = ta.rule_verdict(cnt["fail_q50"], cnt["fail_q60"], [])
    assert v["verdict"] == want and cnt["fail_q50"] == len(f50) and cnt["fail_total"] == len(f50) + len(f60)
    head = {"counts": cnt, "verdict": v}
    ok, diffs = bl.compare(blind, head)
    assert ok, diffs


def test_the_conditions_name_what_failed():
    assert ta.rule_verdict(4, 3, [])["total_condition"] is False and ta.rule_verdict(4, 3, [])["q50_condition"] is True
    v = ta.rule_verdict(5, 0, [])
    assert (v["total_condition"], v["q50_condition"], v["q60_condition"]) == (True, False, True) and "q=50 failures <= 4" in v["reason"]
    v = ta.rule_verdict(0, 5, [])
    assert (v["total_condition"], v["q50_condition"], v["q60_condition"]) == (True, True, False)


def test_an_evaluation_error_is_a_failed_run(tmp_path):
    rows = table(fail50=(0, 1), fail60=(0, 1), eval_error50=(5,), eval_error60=(5,))              # 3 + 3 = 6 failures, two of them evaluation errors
    p, lp = write(tmp_path, rows)
    assert bl.recompute(p, lp)["verdict"] == "pass" and bl.recompute(p, lp)["fail_total"] == 6
    rows = table(fail50=(0, 1), fail60=(0, 1), eval_error50=(5,), eval_error60=(5, 6))             # a seventh failure
    p, lp = write(tmp_path, rows)
    b = bl.recompute(p, lp)
    assert b["verdict"] == "fail" and b["fail_total"] == 7
    assert ta.counts_of(merged(rows))["fail_total"] == 7
    rows = table(eval_error50=(0, 1, 2, 3, 4))                                                     # five evaluation errors at one q
    p, lp = write(tmp_path, rows)
    assert bl.recompute(p, lp)["verdict"] == "fail"


def test_a_missing_row_or_a_nonzero_exit_or_a_rng_violation_is_a_failed_run(tmp_path):
    rows = table()
    rows[0]["exit_code"] = 1
    rows[1]["global_rng_ok"] = False
    del rows[2]
    p, lp = write(tmp_path, rows)
    b = bl.recompute(p, lp)
    assert b["fail_q50"] == 3 and b["verdict"] == "pass"
    assert ca.run_row(rows[0], GATES)["passed"] is False and ca.run_row(rows[1], GATES)["passed"] is False and ca.run_row(None, GATES)["passed"] is False


def test_a_second_infrastructure_failure_is_a_stop_without_a_verdict(tmp_path):
    rows = table()
    ledger = {"schema": "x", "events": [{"run": rows[3]["run"], "attempt": 1}, {"run": rows[3]["run"], "attempt": 2}],
              "stopped_without_verdict": [rows[3]["run"]]}
    p, lp = write(tmp_path, rows, ledger)
    b = bl.recompute(p, lp)
    assert b["verdict"] == "no_verdict" and b["stopped_without_verdict"] == [rows[3]["run"]]
    cnt = ta.counts_of(merged(rows, stopped=[rows[3]["run"]]))
    v = ta.rule_verdict(cnt["fail_q50"], cnt["fail_q60"], [rows[3]["run"]])
    assert v["verdict"] == "no_verdict"
    # an operator who recorded only the second event is also a stop
    ledger2 = {"schema": "x", "events": [{"run": rows[3]["run"], "attempt": 2}], "stopped_without_verdict": []}
    p2, lp2 = write(tmp_path, rows, ledger2)
    assert bl.recompute(p2, lp2)["verdict"] == "no_verdict" and ca.stopped_runs(ledger2) == [rows[3]["run"]]
    # one infrastructure failure that the repeat cured is not a stop
    ledger3 = {"schema": "x", "events": [{"run": rows[3]["run"], "attempt": 1}], "stopped_without_verdict": []}
    p3, lp3 = write(tmp_path, rows, ledger3)
    assert bl.recompute(p3, lp3)["verdict"] == "pass"


def test_the_ledger_must_exist(tmp_path):
    p, _ = write(tmp_path, table())
    with pytest.raises(FileNotFoundError):
        bl.recompute(p, tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        ca.read_ledger(tmp_path / "missing.json")


def test_the_two_pass_rules_agree_on_random_rows_around_the_thresholds():
    rng = random.Random(20261022)
    disagreements = 0
    for _ in range(3000):
        r = eval_row(50, 80501,
                     eta_T_over_dw=rng.choice([0.0, 0.004999999, 0.005, 0.0050001, 0.02]),
                     rmse_pos_T_over_eT0=rng.choice([0.03, 0.05, 0.0500001, float("nan")]),
                     tail_mean_T_over_eT0=rng.choice([0.0, 0.02, 0.0200001]),
                     Gmax_full_over_dw=rng.choice([0.0, 0.01, 0.0100001, 0.05]),
                     dev_eta_T_over_dw=rng.choice([0.0, 0.001, 0.0011, 0.005]),
                     dev_Gmax_full_over_dw=rng.choice([0.0, 0.0011, 0.01]),
                     valid=rng.choice([True, True, False]), dev_valid=rng.choice([True, True, False]),
                     eval_error=rng.choice(["", "", "boom"]), global_rng_ok=rng.choice([True, True, False]),
                     exit_code=rng.choice([0, 0, 1]))
        disagreements += bl.passes(r, GATES) != ca.run_row(r, GATES)["passed"]
    assert disagreements == 0


def test_the_inclusive_limits_are_met_at_equality():
    assert bl.passes(eval_row(50, 80501, eta_T_over_dw=0.005, rmse_pos_T_over_eT0=0.05, tail_mean_T_over_eT0=0.02, Gmax_full_over_dw=0.01,
                              dev_eta_T_over_dw=0.005, dev_Gmax_full_over_dw=0.01), GATES)
    assert not bl.passes(eval_row(50, 80501, eta_T_over_dw=0.0050000001), GATES)
    assert not bl.passes(eval_row(50, 80501, Gmax_full_over_dw=0.0100000001), GATES)


# ---------------------------------------------------------------------------------- classes, failures, reported items
def _fr(**over):
    base = {"run": "t3r4_q50_s80501_bwA", "q": 50, "seed": 80501, "exit_ok": True, "eval_ok": True, "ga_eta": False, "ga_rmse": True, "ga_tail": True,
            "gate_G_F": True, "gate_G_N": True, "eta_T_over_dw": 0.006, "rmse_pos_T_over_eT0": 0.03, "tail_mean_T_over_eT0": 0.005,
            "Gmax_full_over_dw": 0.006, "Gmax_full_t": 3, "dev_eta_T_over_dw": 0.0061, "fail_reason": "gate_G_A", "S3_frozen": 0.01}
    base.update(over)
    return base


def test_the_failing_quantities_name_the_stage():
    q = ta.failing_quantities(_fr())
    assert [(x["quantity"], x["stage"]) for x in q] == [("eta_3", 3)]
    q = ta.failing_quantities(_fr(ga_eta=True, ga_rmse=False, ga_tail=False, gate_G_F=False, Gmax_full_t=2, Gmax_full_over_dw=0.05))
    assert [(x["quantity"], x["stage"]) for x in q] == [("rmse_pos_T_over_eT0", 3), ("tail_mean_T_over_eT0", 3), ("Gmax_full", 2)]
    assert ta.failing_quantities(_fr(exit_ok=False, fail_reason="exit: status 'failed'"))[0]["quantity"].startswith("exit")


@pytest.mark.parametrize("s3,f_u,rank,want", [(0.5, 99.0, 1, "K"), (0.05, 10.0, 1, "F"), (0.05, 99.0, 1, "V"), (0.05, 99.0, 5, "U"), (0.2, 99.0, 1, "V"),
                                              (0.2000001, 10.0, 1, "K")])
def test_the_suggested_class(s3, f_u, rank, want):
    r = _fr(S3_frozen=s3)
    l16 = [{"run": r["run"], "failing": True, "final_rank_from_top": rank}, {"run": r["run"], "failing": False, "final_rank_from_top": 1}]
    assert ta.suggested_class(r, f_u, 26.78, l16) == want


def test_a_rank_one_value_of_a_quantity_that_did_not_fail_does_not_make_a_run_class_v():
    r = _fr(S3_frozen=0.05)
    assert ta.suggested_class(r, 99.0, 26.78, [{"run": r["run"], "failing": False, "final_rank_from_top": 1}]) == "U"


def test_the_cost_and_value_tables():
    rows = [{"q": 50 if i < 2 else 60, "seed": 80501 + i, "run": f"r{i}", "passed": True, "fired": i == 0, "total_updates": 5400 if i == 0 else 3800,
             "total_transitions": 5400 * 512 if i == 0 else 3800 * 512, "total_minibatch_steps": 108000 if i == 0 else 76000,
             "total_dev_checks": 216 if i == 0 else 152, "wall_sec": 800.0 if i == 0 else 500.0, "dev_verifier_sec": 9.0,
             "Gmax_full_over_dw": 0.001 * (i + 1), "eta_T_over_dw": 0.001, "rmse_pos_T_over_eT0": 0.03, "tail_mean_T_over_eT0": 0.005,
             "peak_T_signed": -0.07, "EXP_root_over_dw": 0.0004} for i in range(4)]
    c = ta.cost_rows(rows)[-1]
    assert c["q"] == "both" and c["fired"] == 1 and c["fired_share"] == 0.25 and c["mean_updates"] == pytest.approx((5400 + 3 * 3800) / 4)
    assert c["ratio_updates_to_not_fired"] == pytest.approx((5400 + 3 * 3800) / 4 / 3800) and c["mean_wall_sec_fired"] == 800.0
    v = ta.values_table(rows)[-1]
    assert v["worst_Gmax_full_run"] == "r3" and v["worst_Gmax_full"] == pytest.approx(0.004) and v["median_Gmax_full_over_dw"] == pytest.approx(0.0025)
    assert ta.fired_rows([{**rows[0], "trig_Delta": 0.01, "trig_s": 1.0, "trig_C": 0.1, "trig_reasons": "delta_gt", "S3_first": 0.7,
                           "S3_frozen": 0.1, "a2_final_Delta": 0.001, "fail_reason": ""}])[0]["reasons"] == "delta_gt"


def test_the_descriptive_comparisons_on_the_tracked_tables():
    """The three blocks of T3-1.0's ``bw`` pass 36, 34 and 37 of 40 and ``bwA``'s development runs 39 of 40 (the figures of the prompt)."""
    mine = [{"q": q, "seed": s, "passed": True, "fired": False} for q in (50, 60) for s in SEEDS]
    mine[0]["passed"] = False
    mine[1]["fired"] = True
    comp = {(r["set"], r["q"]): r for r in ta.comparison_rows(mine, GATES)}
    assert comp[("t3r4_confirmation_bwA", "all")]["passes"] == 79 and comp[("t3r4_confirmation_bwA", "all")]["n"] == 80
    assert comp[("t3r4_first_attempt_bwA_equivalent", "all")]["passes"] == 78
    assert comp[("t3r3_development_bwA", "all")]["passes"] == 39 and comp[("t3r3_development_bwA", "all")]["n"] == 40
    assert [comp[(k, "all")]["passes"] for k in ("t3r1_development_bw", "t3r2_confirmation_bw", "t3r3_development_bw")] == [36, 34, 37]
    assert comp[("pooled_bw_120", "all")]["passes"] == 107 and comp[("pooled_bw_120", "all")]["n"] == 120
    assert comp[("pooled_bwA_120", "all")]["passes"] == 79 + 39 and comp[("pooled_bwA_120", "all")]["n"] == 120
    assert 0 < comp[("pooled_bwA_120", "all")]["ci_lo"] < comp[("pooled_bwA_120", "all")]["rate"] < comp[("pooled_bwA_120", "all")]["ci_hi"] < 1


def test_the_tools_refuse_a_relative_root(tmp_path):
    import t3r4_checks as ck
    import t3r4_eval as ev
    for fn, argv in ((ta.main, ["--root", "results/x", "--eval", "e.csv", "--ledger", "l.json", "--out-dir", str(tmp_path)]),
                     (ck.main, ["--root", "results/x", "--plan", "p.csv", "--commit", "c", "--out", str(tmp_path / "o.json")]),
                     (ev.main, ["--root", "results/x", "--out-dir", str(tmp_path)])):
        with pytest.raises(SystemExit) as exc:
            fn(argv)
        assert exc.value.code == 2
    assert list(tmp_path.iterdir()) == []
