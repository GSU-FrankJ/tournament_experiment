"""T3-R2: tests of the verdict tools ``tools/t3/t3c_analysis.py`` and ``tools/t3/t3c_blind.py`` on synthetic tables with
known verdicts, of the infrastructure ledger rule, and of ``tools/t3/t3c_checks.py``.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_t3r2_analysis.py -p no:cacheprovider -q
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "t3", ROOT / "tools" / "ms", ROOT / "tests"):
    sys.path.insert(0, str(_p))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import t3_helpers as H  # noqa: E402
import t3_eval as te  # noqa: E402
import t3c_analysis as an  # noqa: E402
import t3c_blind as bl  # noqa: E402
import t3c_checks as ck  # noqa: E402
import t3c_launch as cl  # noqa: E402

SEEDS = list(range(60501, 60521))
GOOD = dict(eta_T_over_dw=0.002, rmse_pos_T_over_eT0=0.03, tail_mean_T_over_eT0=0.005, Gmax_full_over_dw=0.004,
            dev_eta_T_over_dw=0.0021, dev_Gmax_full_over_dw=0.0042)


def row(q, s, **kw):
    r = {"wave": "t3c", "arm": "bw", "q": q, "seed": s, "T": 3, "run": f"t3r2_q{q}_s{s}_bw", "kind": "bw", "status": "done",
         "exit_code": 0, "global_rng_ok": True, "eval_error": "", "valid": True, "dev_valid": True, "EXP_root_over_dw": 0.001,
         "Gmax_full_t": 3, "Gmax_full_d": 0.0, **GOOD}
    r.update(kw)
    return r


def table(n50, n60, mods=None, drop=()):
    """40 rows with ``n50`` / ``n60`` passing runs per q (the first runs pass); ``mods`` edits single rows."""
    rows = []
    for q, n in ((50, n50), (60, n60)):
        for i, s in enumerate(SEEDS):
            if (q, s) in drop:
                continue
            r = row(q, s) if i < n else row(q, s, Gmax_full_over_dw=0.011, dev_Gmax_full_over_dw=0.0111)     # fails G-F
            rows.append(r)
    for (q, s), kw in (mods or {}).items():
        for r in rows:
            if r["q"] == q and r["seed"] == s:
                r.update(kw)
    return rows


def ledger(tmp_path, stopped=(), events=()):
    p = tmp_path / "infra_events.json"
    p.write_text(json.dumps({"schema": cl.LEDGER_SCHEMA, "events": list(events), "stopped_without_verdict": list(stopped)}))
    return p


def run_both(tmp_path, rows, led):
    pr = tmp_path / "per_run.csv"
    pd.DataFrame(rows).to_csv(pr, index=False)
    out = tmp_path / "an"
    rec = an.analyse(pr, led, out)
    blind = bl.recompute(pr, led)
    ok, diffs = bl.compare(blind, out / "verdict.json")
    assert ok, diffs
    return rec["verdict"], blind


@pytest.mark.parametrize("n50,n60,want", [(18, 17, "fail"),            # 35 of 40
                                          (16, 20, "fail"),            # 36 of 40, q = 50 below 17 (the per-q floor alone)
                                          (20, 16, "fail"),
                                          (18, 18, "pass"), (17, 19, "pass"), (19, 17, "pass"), (20, 20, "pass"),
                                          (17, 17, "fail"),            # 34 of 40: the total alone
                                          (17, 18, "fail"), (0, 0, "fail")])
def test_the_confirmation_rule_on_synthetic_tables(tmp_path, n50, n60, want):
    v, blind = run_both(tmp_path, table(n50, n60), ledger(tmp_path))
    assert v["verdict"] == want == blind["verdict"]
    assert (v["pass_q50"], v["pass_q60"], v["pass_total"]) == (n50, n60, n50 + n60)
    assert blind["pass_q50"] == n50 and blind["pass_q60"] == n60
    assert v["total_condition"] == (n50 + n60 >= 36) and v["q50_condition"] == (n50 >= 17) and v["q60_condition"] == (n60 >= 17)


def test_the_floor_per_q_is_a_condition_of_its_own(tmp_path):
    v, _ = run_both(tmp_path, table(16, 20), ledger(tmp_path))
    assert v["total_condition"] is True and v["q50_condition"] is False and v["verdict"] == "fail"
    assert "q=50" in v["reason"] and "total" not in v["reason"]


def test_failed_runs_of_every_kind_are_counted_as_failed(tmp_path):
    base = table(20, 20)
    cases = {"evaluation error": dict(eval_error="ValueError: x", valid=False, eta_T_over_dw=float("nan")),
             "invalid final tier": dict(valid=False), "invalid development tier": dict(dev_valid=False),
             "global-RNG violation": dict(global_rng_ok=False), "exit 5": dict(status="done", exit_code=5),
             "exit 1": dict(status="failed", exit_code=1), "missing run": dict(status="missing", exit_code=None),
             "G-A eta": dict(eta_T_over_dw=0.0050000001), "G-A rmse": dict(rmse_pos_T_over_eT0=0.0500001),
             "G-A tail": dict(tail_mean_T_over_eT0=0.0200001), "G-F": dict(Gmax_full_over_dw=0.0100001),
             "G-N eta": dict(dev_eta_T_over_dw=0.0031), "G-N gmax": dict(dev_Gmax_full_over_dw=0.0051)}
    for name, kw in cases.items():
        rows = table(20, 20, mods={(50, 60501): kw})
        d = tmp_path / name.replace(" ", "_")
        d.mkdir()
        v, blind = run_both(d, rows, ledger(d))
        assert v["pass_q50"] == 19 and blind["pass_q50"] == 19 and v["verdict"] == "pass", name
    # a planned run without any row is a failed run
    d = tmp_path / "norow"
    d.mkdir()
    v, blind = run_both(d, table(20, 20, drop={(60, 60520)}), ledger(d))
    assert v["pass_q60"] == 19 and blind["pass_q60"] == 19


def test_every_limit_is_inclusive_at_full_float64_precision(tmp_path):
    ok = dict(eta_T_over_dw=0.005, rmse_pos_T_over_eT0=0.05, tail_mean_T_over_eT0=0.02, Gmax_full_over_dw=0.01,
              dev_eta_T_over_dw=0.005 + 0.001, dev_Gmax_full_over_dw=0.01 + 0.001)
    # the G-N differences are computed as |dev - final|; whatever float64 gives, both tools must agree with Python's own comparison
    want_gn = abs(ok["dev_eta_T_over_dw"] - ok["eta_T_over_dw"]) <= 0.001 and abs(ok["dev_Gmax_full_over_dw"] - ok["Gmax_full_over_dw"]) <= 0.001
    d = tmp_path / "a"
    d.mkdir()
    v, blind = run_both(d, table(20, 20, mods={(50, 60501): ok}), ledger(d))
    assert v["pass_q50"] == (20 if want_gn else 19) == blind["pass_q50"]
    ok2 = dict(ok, dev_eta_T_over_dw=0.005, dev_Gmax_full_over_dw=0.01)           # equal tiers: G-N holds; the others sit on the limit
    d = tmp_path / "b"
    d.mkdir()
    v, blind = run_both(d, table(20, 20, mods={(50, 60501): ok2}), ledger(d))
    assert v["pass_q50"] == 20 == blind["pass_q50"]


def test_a_second_infrastructure_failure_stops_without_a_verdict_and_the_run_is_counted_neither_way(tmp_path):
    led = ledger(tmp_path, stopped=["t3r2_q60_s60501_bw"],
                 events=[{"run": "t3r2_q60_s60501_bw", "attempt": 1, "reason": "negative_returncode"},
                         {"run": "t3r2_q60_s60501_bw", "attempt": 2, "reason": "enospc"}])
    rows = table(20, 20, drop={(60, 60501)})
    v, blind = run_both(tmp_path, rows, led)
    assert v["verdict"] == "no_verdict" == blind["verdict"]
    assert blind["n_counted_q60"] == 19 and blind["pass_q60"] == 19 and blind["n_counted_total"] == 39
    assert v["stopped_without_verdict"] == ["t3r2_q60_s60501_bw"]
    # an infrastructure event with a successful repeat does not stop anything
    d = tmp_path / "ok"
    d.mkdir()
    led2 = ledger(d, events=[{"run": "t3r2_q60_s60501_bw", "attempt": 1, "reason": "negative_returncode"}])
    v2, blind2 = run_both(d, table(20, 20), led2)
    assert v2["verdict"] == "pass" == blind2["verdict"]


def test_both_tools_refuse_a_missing_ledger(tmp_path):
    pr = tmp_path / "per_run.csv"
    pd.DataFrame(table(20, 20)).to_csv(pr, index=False)
    with pytest.raises(FileNotFoundError):
        an.analyse(pr, tmp_path / "missing.json", tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        bl.recompute(pr, tmp_path / "missing.json")


def test_the_clopper_pearson_intervals_of_the_two_tools_agree_and_match_known_values():
    for n in (20, 39, 40):
        for k in range(0, n + 1):
            a, b = an.clopper_pearson(k, n), bl.cp_interval(k, n)
            assert abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9, (k, n)
    lo, hi = an.clopper_pearson(18, 20)
    assert (lo, hi) == pytest.approx((0.6830, 0.9877), abs=2e-4)           # the exact 95 % interval of 18 / 20


def test_the_analysis_reports_pooled_and_gate_value_tables_and_the_other_outputs(tmp_path):
    rows = table(18, 17, mods={(50, 60501): dict(rmse_pos_T_over_eT0=0.06)})
    pr = tmp_path / "per_run.csv"
    pd.DataFrame(rows).to_csv(pr, index=False)
    out = tmp_path / "an"
    rec = an.analyse(pr, ledger(tmp_path), out)
    pooled = pd.read_csv(out / "pooled_descriptive.csv")
    assert set(pooled["set"]) == {"t3r2_confirmation", "t3r1_bw_development", "pooled_80_descriptive"}
    pool = pooled[(pooled["set"] == "pooled_80_descriptive") & (pooled["q"] == "all")].iloc[0]
    old = pooled[(pooled["set"] == "t3r1_bw_development") & (pooled["q"] == "all")].iloc[0]
    assert pool["n"] == 80 and old["n"] == 40 and old["passes"] == 36                       # T3-R1: 18 + 18
    assert pool["passes"] == old["passes"] + rec["verdict"]["pass_total"]
    gv = pd.read_csv(out / "gate_values_vs_t3r1.csv")
    assert set(gv["set"]) == {"t3r2_confirmation", "t3r1_bw"} and len(gv) == 2 * 2 * 6
    assert (out / "per_run_table.csv").is_file() and (out / "verdict.json").is_file() and (out / "pass_counts.csv").is_file()
    t = pd.read_csv(out / "per_run_table.csv")
    assert len(t) == 40 and bool(t[(t.q == 50) & (t.seed == 60501)]["gate_G_A"].iloc[0]) is False
    assert t[(t.q == 50) & (t.seed == 60501)]["fail_reason"].iloc[0] == "gate_G_A"


# ====================================================================== diagnosis on a reduced run
def test_the_diagnosis_and_the_stretch_table_run_on_a_reduced_run(tmp_path):
    d, cfg = H.reduced_bw_run()
    root = tmp_path / "conf"
    shutil.copytree(d, cl.run_dir(root, 50, 60501))
    job = {"wave": "t3c", "arm": "bw", "q": 50, "seed": 60501, "T": 3, "run": "t3r2_q50_s60501_bw",
           "run_dir": str(cl.run_dir(root, 50, 60501)), "kind": "bw", "arrays_dir": None}
    r = te.evaluate_run(job)
    rows = [dict(r)]
    crit_row = an.planned_table(pd.DataFrame(rows), an.tm.load_criteria()[0]["gates"])
    t = [x for x in crit_row if x["seed"] == 60501 and x["q"] == 50]
    assert len(t) == 1
    t[0]["passed"] = False
    t[0]["ga_eta"] = False                     # force a failure to be diagnosed
    res = an.diagnose(root, t, tmp_path / "diag", 1)
    assert any(x.get("failing") == "G-A eta_3" and x["stage"] == 3 for x in res["failures"])
    assert len(res["stretches"]) == 1 and "nodes" in res["stretches"][0]
    assert (tmp_path / "diag" / "failures.csv").is_file() and (tmp_path / "diag" / "stretches.csv").is_file()


# ====================================================================== the post-launch checks
def test_the_checks_report_a_missing_run_and_require_the_ledger(tmp_path):
    job = {"wave": "t3c", "arm": "bw", "q": 50, "seed": 60501, "run": "t3r2_q50_s60501_bw", "run_dir": str(tmp_path / "nope")}
    res = ck.check_run((job, None, None, {}))
    assert res["exit_ok"] is False and res["files_complete"] is False
    agg = {"n_runs": 40, "verdicts": {v: {"pass": 40, "fail": 0, "not_evaluated": 0} for v in ck.VERDICTS},
           "infra_events_present": True}
    assert ck.checks_ok(agg)
    assert not ck.checks_ok({**agg, "infra_events_present": False})
    bad = json.loads(json.dumps(agg))
    bad["verdicts"]["replay_ok"]["fail"] = 1
    assert not ck.checks_ok(bad)
    bad = json.loads(json.dumps(agg))
    bad["verdicts"]["init_equals_plan"]["not_evaluated"] = 1
    assert not ck.checks_ok(bad)
    p = tmp_path / "plan.csv"
    p.write_text("wave,arm,q,seed,run,out_dir,run_config_sha256,run_config_sha256_relpath,init_state_sha256\n"
                 "t3c,bw,50,60501,t3r2_q50_s60501_bw,/x,a,b,c\n")
    assert ck.read_plan(p)["t3r2_q50_s60501_bw"]["init_state_sha256"] == "c"
