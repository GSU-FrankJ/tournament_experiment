"""Tests of the R2c selection rule and its inputs (tools/v2/refine_r2c_analysis.py).

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_v2_r2c_analysis.py -p no:cacheprovider -q

Sections: the pure function ``select_arm`` on synthetic per-arm tables (eligibility, the ordered tie-break,
invariance to the order of the arms), the inclusive 0.05 count, the arm metadata read from the launcher tables, and
the path from a synthetic per-run table through the R1 pairing / criterion code to the selection inputs. No result
file is read and nothing is trained.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import refine_r2c_analysis as A  # noqa: E402

QS = (50, 60)
SEEDS = tuple(range(10501, 10511))


def arm(name: str, **kw: Any) -> Dict[str, Any]:
    """An eligible synthetic arm; keyword arguments override the fields."""
    t: Dict[str, Any] = {"arm": name, "complete": True, "a_met": True, "b_status": "holds",
                         "n_abs_peak_le_0.05_total": 10, "mean_abs_peak_error": 0.04, "local_first": 1, "share": 0.5}
    t.update(kw)
    return t


# ====================================================================== 1. eligibility
def test_none_meets_a_and_b_selects_nothing() -> None:
    tab = [arm("A", a_met=False), arm("B", b_status="violated", b_violations=["q60/10503"]),
           arm("C", a_met=False, b_status="violated")]
    sel = A.select_arm(tab)
    assert sel["selected"] is None and sel["outcome"] == "none_eligible"
    assert sel["ranking"] == [] and set(sel["ineligible"]) == {"A", "B", "C"}
    assert "No arm meets both parts of the criterion" in sel["reason"]
    assert "q60/10503" in sel["reason"]            # says why


def test_empty_table_selects_nothing() -> None:
    sel = A.select_arm([])
    assert sel["selected"] is None and sel["ranking"] == []


def test_exactly_one_eligible_is_selected() -> None:
    sel = A.select_arm([arm("A", a_met=False), arm("B"), arm("C", b_status="violated")])
    assert sel["selected"] == "B" and sel["outcome"] == "selected"
    assert [r["arm"] for r in sel["ranking"]] == ["B"] and sel["decided_by"] == "the only eligible arm"
    assert set(sel["ineligible"]) == {"A", "C"}


def test_a_at_one_q_only_is_not_eligible() -> None:
    a = {"q50": {"met": True}, "q60": {"met": False}}
    sel = A.select_arm([arm("A", a_met=False, a=a, **{"n_abs_peak_le_0.05_total": 20, "mean_abs_peak_error": 0.0}),
                        arm("B", **{"n_abs_peak_le_0.05_total": 1, "mean_abs_peak_error": 0.09})])
    assert sel["selected"] == "B"
    assert sel["ineligible"]["A"] == ["(a) not met at q60"]


def test_b_violation_is_not_eligible_even_with_the_best_counts() -> None:
    best = {"n_abs_peak_le_0.05_total": 20, "mean_abs_peak_error": 0.001}
    sel = A.select_arm([arm("A", b_status="violated", b_violations=["q60/10510"], **best),
                        arm("B", **{"n_abs_peak_le_0.05_total": 3, "mean_abs_peak_error": 0.08})])
    assert sel["selected"] == "B"
    assert "(b) violated (q60/10510)" in sel["ineligible"]["A"][0]


def test_b_pending_is_not_eligible() -> None:
    sel = A.select_arm([arm("A", b_status="incomplete")])
    assert sel["selected"] is None and sel["ineligible"]["A"] == ["(b) incomplete"]


def test_incomplete_arm_is_not_eligible() -> None:
    best = {"n_abs_peak_le_0.05_total": 20, "mean_abs_peak_error": 0.001}
    sel = A.select_arm([arm("A", complete=False, not_done_runs=["q50/10503 (missing)"], **best), arm("B")])
    assert sel["selected"] == "B"
    assert sel["ineligible"]["A"] == ["incomplete: q50/10503 (missing)"]
    only = A.select_arm([arm("A", complete=False)])
    assert only["selected"] is None and "incomplete" in only["reason"]


# ====================================================================== 2. the ordered tie-break
def test_larger_count_wins_even_with_a_worse_mean() -> None:
    sel = A.select_arm([arm("A", **{"n_abs_peak_le_0.05_total": 12, "mean_abs_peak_error": 0.040}),
                        arm("B", **{"n_abs_peak_le_0.05_total": 14, "mean_abs_peak_error": 0.060, "local_first": 1})])
    assert sel["selected"] == "B" and "number of runs" in sel["decided_by"]


def test_equal_count_smaller_mean_wins_even_with_a_larger_change_from_the_sampler() -> None:
    sel = A.select_arm([arm("A", mean_abs_peak_error=0.041, local_first=1201, share=0.1),
                        arm("B", mean_abs_peak_error=0.039, local_first=1, share=0.9)])
    assert sel["selected"] == "B" and "mean" in sel["decided_by"]


def test_equal_count_and_mean_later_local_first_wins() -> None:
    sel = A.select_arm([arm("A", local_first=1, share=0.35), arm("B", local_first=801, share=0.5),
                        arm("C", local_first=1201, share=0.9)])
    assert sel["selected"] == "C" and "local_first" in sel["decided_by"]
    assert [r["arm"] for r in sel["ranking"]] == ["C", "B", "A"]


def test_equal_local_first_smaller_share_wins() -> None:
    sel = A.select_arm([arm("A", share=0.50), arm("B", share=0.35), arm("C", share=0.40)])
    assert sel["selected"] == "B" and "share" in sel["decided_by"]
    assert [r["arm"] for r in sel["ranking"]] == ["B", "C", "A"]


def test_complete_tie_selects_nothing_and_says_so() -> None:
    sel = A.select_arm([arm("A"), arm("B")])
    assert sel["selected"] is None and sel["outcome"] == "unresolved_tie"
    assert [r["arm"] for r in sel["ranking"]] == ["A", "B"] and "agree in all four" in sel["reason"]


def test_ranking_records_the_sort_key_values_in_order() -> None:
    tab = [arm("A", **{"n_abs_peak_le_0.05_total": 11, "mean_abs_peak_error": 0.05, "local_first": 1, "share": 0.35}),
           arm("B", **{"n_abs_peak_le_0.05_total": 14, "mean_abs_peak_error": 0.06, "local_first": 801, "share": 0.5}),
           arm("C", **{"n_abs_peak_le_0.05_total": 14, "mean_abs_peak_error": 0.04, "local_first": 1201, "share": 0.5}),
           arm("D", **{"n_abs_peak_le_0.05_total": 14, "mean_abs_peak_error": 0.04, "local_first": 1, "share": 0.4})]
    sel = A.select_arm(tab)
    assert [r["arm"] for r in sel["ranking"]] == ["C", "D", "B", "A"]
    assert [r["rank"] for r in sel["ranking"]] == [1, 2, 3, 4]
    assert sel["ranking"][0]["sort_key"] == [-14.0, 0.04, -1201.0, 0.5]
    assert sel["ranking"][3]["sort_key"] == [-11.0, 0.05, -1.0, 0.35]
    assert sel["selected"] == "C"


def test_selection_is_invariant_to_the_order_of_the_arms() -> None:
    tab = [arm("A", a_met=False), arm("B", **{"n_abs_peak_le_0.05_total": 14, "mean_abs_peak_error": 0.05}),
           arm("C", **{"n_abs_peak_le_0.05_total": 14, "mean_abs_peak_error": 0.04, "local_first": 801}),
           arm("D", **{"n_abs_peak_le_0.05_total": 14, "mean_abs_peak_error": 0.04, "local_first": 1201}),
           arm("E", complete=False)]
    ref = A.select_arm(tab)
    assert ref["selected"] == "D"
    for perm in itertools.permutations(tab):
        assert A.select_arm(list(perm)) == ref
    tied = [arm("X"), arm("Y"), arm("Z", a_met=False)]
    ref2 = A.select_arm(tied)
    for perm in itertools.permutations(tied):
        assert A.select_arm(list(perm)) == ref2


def test_select_arm_does_not_modify_its_input_and_rejects_duplicates() -> None:
    tab = [arm("A"), arm("B", share=0.35)]
    before = json.dumps(tab, sort_keys=True)
    A.select_arm(tab)
    assert json.dumps(tab, sort_keys=True) == before
    with pytest.raises(ValueError):
        A.select_arm([arm("A"), arm("A")])


# ====================================================================== 3. the 0.05 count is inclusive
def test_count_le_counts_a_run_at_exactly_0_05() -> None:
    assert A.count_le([0.05]) == 1
    assert A.count_le([0.05, 0.0500000001, 0.049999, 0.2]) == 2
    assert A.count_le(pd.Series([0.05, np.nan, 0.01])) == 2          # NaN never counts
    assert A.TAIL_PEAK == 0.05


# ====================================================================== 4. arm metadata
def test_arm_meta_reads_local_first_and_share_from_the_launcher_tables() -> None:
    m = {a: A.arm_meta(a) for a in A.L.R2C_WAVE_S_ARMS}
    assert [(m[a]["local_first"], m[a]["share"]) for a in A.L.R2C_WAVE_S_ARMS] == [
        (1, 0.35), (1, 0.40), (1201, 0.50), (801, 0.50)]
    assert A.arm_meta("A_peak50")["local_first"] == 1 and A.arm_meta("A_peak50")["share"] == 0.5   # R2b arm: default 1
    assert A.arm_meta("A_peak25")["start_weights"]["peak_share"] == 0.25
    assert A.arm_meta("A_censored")["start_weights"] is None
    with pytest.raises(SystemExit):
        A.arm_meta("A_nonexistent")


# ====================================================================== 5. synthetic per-run table -> inputs -> selection
def synth_df(arms: Dict[str, Dict[int, List[float]]], base: Dict[int, List[float]], gate_fail: Dict[str, List[Any]] = None,
             missing: Dict[str, List[Any]] = None) -> pd.DataFrame:
    """Per-run table with the columns the pairing, the criterion, the tail table and the selection inputs read."""
    gate_fail, missing = gate_fail or {}, missing or {}
    rows = []
    for name, per_q in [("A_base", base)] + list(arms.items()):
        for q in QS:
            for s, v in zip(SEEDS, per_q[q]):
                if (q, s) in missing.get(name, []):
                    continue
                ok = (q, s) not in gate_fail.get(name, [])
                rows.append({"arm": name, "q": q, "seed": s, "status": "done", "complete": True,
                             A.PRIMARY: v, "gate_pass": ok, "eta_T_over_dw": 0.001,
                             "stage2_tail_mean_over_g2_0": 0.01})
    return pd.DataFrame(rows)


def run_pipeline(df: pd.DataFrame, names: List[str]):
    """The pairing, criterion and selection-input steps of ``compute`` on a synthetic table."""
    A.R.BOOT_SEED = 20261005
    comps = [(a, A.BASE, A.CRIT_LABEL) for a in names]
    paired, _ = A.R.paired_tables(df, "stage2", comps, QS, SEEDS)
    crit = A.R.criterion_table(df, "stage2", paired, comps, QS, SEEDS)
    meta = {a: A.arm_meta(a) for a in names}
    inputs = A.selection_inputs(df, crit, names, QS, SEEDS, meta)
    return paired, crit, inputs, A.select_arm(inputs)


BASE_VALS = {50: [0.10, 0.09, 0.11, 0.08, 0.12, 0.10, 0.09, 0.11, 0.10, 0.08],
             60: [0.08, 0.09, 0.07, 0.10, 0.08, 0.09, 0.07, 0.08, 0.09, 0.10]}


def improved(base: Dict[int, List[float]], delta: float) -> Dict[int, List[float]]:
    """Every seed improves by ``delta`` (so the CI of the mean difference is exactly -delta: it excludes 0)."""
    return {q: [v - delta for v in vs] for q, vs in base.items()}


def test_pipeline_selects_the_arm_with_more_runs_below_0_05_and_counts_inclusively() -> None:
    x = improved(BASE_VALS, 0.06)                         # 0.04, 0.03, ... ; the first q50 run is exactly 0.10 - 0.06
    x[50][0] = 0.05                                       # a run at exactly 0.05 counts
    x[60][1] = 0.0500001                                  # just above does not
    y = improved(BASE_VALS, 0.04)
    df = synth_df({"A_peak35": x, "A_peak40": y}, BASE_VALS)
    paired, crit, inputs, sel = run_pipeline(df, ["A_peak35", "A_peak40"])
    ix = {t["arm"]: t for t in inputs}
    assert all(t["complete"] and t["a_met"] and t["b_status"] == "holds" for t in inputs)
    exp_x = [sum(v <= 0.05 for v in x[q]) for q in QS]
    assert [ix["A_peak35"]["n_abs_peak_le_0.05"][f"q{q}"] for q in QS] == exp_x
    assert ix["A_peak35"]["n_abs_peak_le_0.05_total"] == sum(exp_x)
    assert x[50][0] <= 0.05 < x[60][1]
    assert ix["A_peak35"]["mean_abs_peak_error"] == pytest.approx(np.mean(x[50] + x[60]))
    assert ix["A_peak40"]["n_abs_peak_le_0.05_total"] == sum(v <= 0.05 for q in QS for v in y[q])
    assert sel["selected"] == "A_peak35" and [r["arm"] for r in sel["ranking"]] == ["A_peak35", "A_peak40"]
    # the tail table of the R2b tool counts the same runs
    tail = A.B.tail_table(df, ["A_peak35", "A_peak40"], QS)
    t35 = tail[tail.arm == "A_peak35"].set_index("q")["n_abs_peak_le_0.05"]
    assert [int(t35[q]) for q in QS] == exp_x


def test_pipeline_a_at_one_q_only_and_b_violation_make_an_arm_ineligible() -> None:
    one_q = {50: [v - 0.06 for v in BASE_VALS[50]], 60: list(BASE_VALS[60])}      # no change at q60: CI contains 0
    viol = improved(BASE_VALS, 0.07)                                              # best counts, but a gate failure
    ok = improved(BASE_VALS, 0.03)
    df = synth_df({"A_peak35": one_q, "A_peak40": viol, "A_peak50_late800": ok}, BASE_VALS,
                  gate_fail={"A_peak40": [(60, 10503)]})
    _, crit, inputs, sel = run_pipeline(df, ["A_peak35", "A_peak40", "A_peak50_late800"])
    ix = {t["arm"]: t for t in inputs}
    assert ix["A_peak35"]["a"]["q50"]["met"] and not ix["A_peak35"]["a"]["q60"]["met"] and not ix["A_peak35"]["a_met"]
    assert ix["A_peak40"]["a_met"] and ix["A_peak40"]["b_status"] == "violated"
    assert ix["A_peak40"]["b_violations"] == ["q60/10503"]
    assert ix["A_peak40"]["n_abs_peak_le_0.05_total"] > ix["A_peak50_late800"]["n_abs_peak_le_0.05_total"]
    assert sel["selected"] == "A_peak50_late800"
    assert set(sel["ineligible"]) == {"A_peak35", "A_peak40"}


def test_pipeline_an_arm_with_a_missing_run_is_incomplete_and_never_selected() -> None:
    best = improved(BASE_VALS, 0.07)
    df = synth_df({"A_peak35": best, "A_peak40": improved(BASE_VALS, 0.03)}, BASE_VALS, missing={"A_peak35": [(50, 10507)]})
    _, _, inputs, sel = run_pipeline(df, ["A_peak35", "A_peak40"])
    ix = {t["arm"]: t for t in inputs}
    assert not ix["A_peak35"]["complete"] and "q50/10507 (missing)" in ix["A_peak35"]["not_done_runs"]
    assert not ix["A_peak35"]["eligible"] and ix["A_peak40"]["eligible"]
    assert sel["selected"] == "A_peak40"


def test_pipeline_with_no_runs_of_the_arms_reports_everything_incomplete() -> None:
    df = synth_df({}, BASE_VALS)
    _, _, inputs, sel = run_pipeline(df, ["A_peak35", "A_peak40"])
    assert all(not t["complete"] and len(t["not_done_runs"]) == 20 for t in inputs)
    assert sel["selected"] is None and sel["outcome"] == "none_eligible" and "incomplete" in sel["reason"]


def test_selection_record_is_valid_json_with_the_required_fields() -> None:
    df = synth_df({"A_peak35": improved(BASE_VALS, 0.06)}, BASE_VALS)
    _, _, inputs, sel = run_pipeline(df, ["A_peak35"])
    args = SimpleNamespace(boot_seed=20261005, root="/r", wave_dir="/r/waveS", r1_root="/r1", ref_root="/ref",
                           r2b_root="", out="/o", reports="/rep")
    rec = A.selection_record(inputs, sel, args, {"A_peak35": A.arm_meta("A_peak35")}, [])
    txt = json.dumps(rec, allow_nan=False)                # no NaN / inf survives
    back = json.loads(txt)
    for k in ("rule", "boot_seed", "roots", "arms", "ranking", "selected", "reason", "outcome"):
        assert k in back
    assert back["boot_seed"] == 20261005 and back["selected"] == "A_peak35"
    assert back["selected_start_weights"]["peak_share"] == 0.35
    a0 = back["arms"][0]
    for k in ("a", "a_met", "b_status", "b_violations", "n_abs_peak_le_0.05", "n_abs_peak_le_0.05_total",
              "mean_abs_peak_error", "complete", "local_first", "share", "eligible"):
        assert k in a0
    assert set(a0["a"]["q50"]) == {"n_pairs", "mean", "ci_lo", "ci_hi", "met"}
    assert back["roots"]["wave_dir"] == "/r/waveS" and back["roots"]["r2b_root"] is None


def test_the_bootstrap_uses_the_seed_that_was_set_not_the_r2b_import_value() -> None:
    """The R2b module sets 20261004 when it is imported; ``compute`` sets the CLI seed (default 20261005) before any CI."""
    rng = np.random.default_rng(7)
    x = {q: [v - float(rng.uniform(0.0, 0.08)) for v in BASE_VALS[q]] for q in QS}
    df = synth_df({"A_peak35": x}, BASE_VALS)
    comps = [("A_peak35", A.BASE, A.CRIT_LABEL)]
    d50 = np.array(x[50]) - np.array(BASE_VALS[50])
    lo = {}
    for seed in (20261004, 20261005):
        A.R.BOOT_SEED = seed
        paired, _ = A.R.paired_tables(df, "stage2", comps, QS, SEEDS)
        crit = A.R.criterion_table(df, "stage2", paired, comps, QS, SEEDS)
        idx = np.random.default_rng(seed).integers(0, 10, size=(10000, 10))
        want = np.percentile(d50[idx].mean(axis=1), [2.5, 97.5])
        assert crit["ci_mean_lo_q50"].iloc[0] == pytest.approx(want[0], abs=1e-15)
        assert crit["ci_mean_hi_q50"].iloc[0] == pytest.approx(want[1], abs=1e-15)
        lo[seed] = crit["ci_mean_lo_q50"].iloc[0]
    assert lo[20261004] != lo[20261005]
    assert A.build_parser().parse_args([]).boot_seed == 20261005 == A.DEFAULT_BOOT_SEED


def test_outcome_sentence_follows_the_outcome_not_only_the_selected_arm() -> None:
    """A tie, an incomplete wave and a plain 'no arm meets (a) and (b)' each get their own sentence."""
    base = {"selected": None, "outcome": "none_eligible", "incomplete_arms": []}
    assert A.outcome_sentence({**base, "selected": "A_peak40", "outcome": "selected"}) == "Selected arm A_peak40."
    assert A.outcome_sentence(base) == "No arm meets both parts of the criterion; the round stops after the pilot report."
    tie = A.outcome_sentence({**base, "outcome": "unresolved_tie"})
    assert "no answer" in tie and "No arm meets" not in tie
    inc = A.outcome_sentence({**base, "incomplete_arms": ["A_peak35"]})
    assert "incomplete" in inc and "not final" in inc and "No arm meets" not in inc
