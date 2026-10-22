"""Tests of the pure parts of ``tools/t3/t3r3_analysis.py`` on synthetic run tables."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "ms", ROOT / "tools" / "t3"):
    sys.path.insert(0, str(_p))

import t3r3_analysis as an  # noqa: E402
import t3r3_launch as lt  # noqa: E402


def _row(arm, q, seed, passed=True, fired=False, would_fire=False, eta=0.001, gmax=0.001, s3=0.0, peak=-0.06, **kw):
    r = dict(an.RECORD_DEFAULTS)
    r.update(run=f"t3r3_q{q}_s{seed}_{arm}", arm=arm, q=q, seed=seed, run_kind="dev", passed=passed, fired=fired, would_fire=would_fire,
             eta_T_over_dw=eta, Gmax_full_over_dw=gmax, S3_first=s3, S3_frozen=s3, peak_T_signed=peak, rmse_pos_T_over_eT0=0.02,
             tail_mean_T_over_eT0=0.003, EXP_root_over_dw=0.001, total_updates=3800, total_transitions=3800 * 512,
             total_minibatch_steps=76000, total_dev_checks=152, wall_sec=600.0, dev_verifier_sec=7.0, fail_reason="")
    r.update(kw)
    return r


def _world(fail_by_arm):
    """All 120 runs; ``fail_by_arm[arm]`` is a set of (q, seed) that fail."""
    rows = []
    for arm in an.ARMS:
        for q in an.QS:
            for s in lt.SEEDS:
                rows.append(_row(arm, q, s, passed=(q, s) not in fail_by_arm.get(arm, set())))
    return rows


def test_counts_and_the_target_rule():
    fails = {"bwA": {(50, 70501), (60, 70502)}, "bwAB": {(50, 70501), (60, 70502), (60, 70503)}, "bw": {(50, 70501 + i) for i in range(4)}}
    rows = _world(fails)
    cnt = {a: an.arm_counts(rows, a) for a in an.ARMS}
    assert (cnt["bwA"]["pass_q50"], cnt["bwA"]["pass_q60"], cnt["bwA"]["pass_total"]) == (19, 19, 38)
    assert cnt["bwAB"]["pass_total"] == 37 and cnt["bw"]["pass_q50"] == 16
    t = {a: an.target_outcome(cnt[a]) for a in an.ARMS}
    assert t["bwA"]["reaches_target"] and not t["bwAB"]["reaches_target"] and not t["bw"]["reaches_target"]
    assert cnt["bwA"]["ci_lo_total"] == pytest.approx(0.8308, abs=1e-3) and cnt["bwA"]["ci_hi_total"] == pytest.approx(0.9937, abs=1e-3)
    # the per-q condition: 38 of 40 but 17 at q = 60
    c = dict(cnt["bwA"], pass_q50=21 - 1, pass_q60=17, pass_total=37 + 1)
    assert not an.target_outcome({**c, "pass_q50": 20, "pass_q60": 17, "pass_total": 37})["reaches_target"]
    assert not an.target_outcome({**c, "pass_q50": 20, "pass_q60": 18, "pass_total": 37, "n_q50": 20, "n_q60": 20})["reaches_target"]
    assert an.target_outcome({**c, "pass_q50": 20, "pass_q60": 18, "pass_total": 38, "n_q50": 20, "n_q60": 20})["reaches_target"]
    assert not an.target_outcome({**c, "pass_q50": 20, "pass_q60": 18, "pass_total": 38, "n_q50": 20, "n_q60": 19})["reaches_target"]   # a planned run is missing


def test_selection_order_and_the_control_is_not_eligible():
    def tgt(**ok):
        return {a: {"reaches_target": ok.get(a, False)} for a in an.ARMS}
    assert an.selection(tgt(bwA=True, bwAB=True))["proposed"] == "bwA" and an.selection(tgt(bwA=True, bwAB=True))["both_reach_target"]
    assert an.selection(tgt(bwAB=True))["proposed"] == "bwAB"
    assert an.selection(tgt(bwA=True))["proposed"] == "bwA"
    assert an.selection(tgt(bw=True))["proposed"] is None


def test_mcnemar_exact():
    assert an.mcnemar_exact([True] * 5, [True] * 5)["p_exact"] == 1.0
    m = an.mcnemar_exact([True] * 8 + [False] * 2, [False] * 8 + [False] * 2)
    assert (m["n_x_only"], m["n_y_only"], m["n_discordant"]) == (8, 0, 8) and m["p_exact"] == pytest.approx(2 * 0.5 ** 8)
    m = an.mcnemar_exact([True, False, True, False], [False, True, True, False])
    assert m["p_exact"] == 1.0 and m["n_discordant"] == 2


def test_first_attempt_convention_and_the_bwA_invariant():
    rows = _world({})
    # one seed: bw fails and would fire, bwA fired and then passed: the first attempts agree (both failures)
    for r in rows:
        if (r["q"], r["seed"]) == (50, 70501):
            if r["arm"] == "bw":
                r.update(passed=False, would_fire=True)
            if r["arm"] == "bwA":
                r.update(passed=True, fired=True)
            if r["arm"] == "bwAB":
                r.update(passed=True, fired=True)
    fa = an.first_attempt_table(rows)
    assert not fa["stop"] and fa["arms"]["bw"]["first_attempt_pass"] == 39 and fa["arms"]["bwA"]["first_attempt_pass"] == 39
    assert fa["arms"]["bwA"]["pass_final"] == 40 and fa["arms"]["bw"]["pass_final"] == 39 and fa["arms"]["bwA"]["fired"] == 1
    # a bw run that would fire but passes the gates (a "fired-but-passing" bw run) is counted as a first-attempt failure in both arms
    rows2 = _world({})
    for r in rows2:
        if (r["q"], r["seed"]) == (60, 70505):
            r.update(would_fire=(r["arm"] == "bw"), fired=(r["arm"] == "bwA"))
    fa2 = an.first_attempt_table(rows2)
    assert not fa2["stop"] and fa2["bw_fired_runs_that_pass_in_bw"] == 1 and fa2["arms"]["bw"]["first_attempt_pass"] == 39
    # a mismatch (bwA did not fire where bw would) is a stop
    rows3 = _world({})
    for r in rows3:
        if (r["q"], r["seed"]) == (60, 70505) and r["arm"] == "bw":
            r.update(would_fire=True)
    assert an.first_attempt_table(rows3)["stop"]


def test_paired_bootstrap_uses_the_registered_seed_and_resamples():
    rows = _world({})
    rng = np.random.default_rng(5)
    delta = {(q, s): float(rng.normal(-0.0002, 0.0003)) for q in an.QS for s in lt.SEEDS}
    for r in rows:
        if r["arm"] == "bwA":
            r["Gmax_full_over_dw"] = 0.001 + delta[(r["q"], r["seed"])]
    p = an.paired_vs(rows, "bwA", "bw", "Gmax_full_over_dw", "both")
    d = np.array([delta[(q, s)] for q in an.QS for s in lt.SEEDS])
    assert p["n_pairs"] == 40 and p["mean_diff"] == pytest.approx(d.mean())
    idx = np.random.default_rng(20261021).integers(0, 40, size=(10000, 40))
    m = d[idx].mean(axis=1)
    assert p["ci_lo"] == pytest.approx(np.percentile(m, 2.5)) and p["ci_hi"] == pytest.approx(np.percentile(m, 97.5))
    other = d[np.random.default_rng(20261022).integers(0, 40, size=(10000, 40))].mean(axis=1)
    assert abs(np.percentile(other, 2.5) - p["ci_lo"]) > 1e-9                         # a different seed would give another interval
    x = an.crosstab(rows, "bwA")
    assert x["both_pass"] == 40 and x["ref_only"] == x["arm_only"] == x["both_fail"] == 0


def test_causal_b_uses_first_attempt_regime_and_pairs_the_seeds():
    rows = _world({})
    for r in rows:
        if r["arm"] in ("bw", "bwA") and (r["q"], r["seed"]) in {(50, 70501), (50, 70502), (50, 70503), (60, 70501)}:
            r["S3_first"] = 0.7
        if r["arm"] == "bwAB" and (r["q"], r["seed"]) == (50, 70501):
            r["S3_first"] = 0.7
        if r["arm"] != "bw" and (r["q"], r["seed"]) == (50, 70502):
            r["fired"] = True
    per_arm, pairs = an.causal_b(rows)
    a = {(x["arm"], x["q"]): x for x in per_arm}
    assert a[("bw", "both")]["S3_first_gt_0.2"] == 4 and a[("bwAB", "both")]["S3_first_gt_0.2"] == 1 and a[("bwAB", 50)]["fired"] == 1
    p = {(x["arm_a"], x["arm_b"], x["q"]): x for x in pairs}
    x = p[("bwAB", "bw", "both")]
    assert (x["n_x_only"], x["n_y_only"]) == (0, 3) and x["p_exact"] == pytest.approx(0.25)
    assert p[("bwA", "bw", "both")]["n_discordant"] == 0
    assert x["n_neither_fired"] == 39                                         # the seed (50, 70502) fired in bwAB


def test_cost_ratios_and_fired_table():
    rows = _world({})
    for r in rows:
        if r["arm"] in ("bwA", "bwAB") and r["seed"] == 70501 and r["q"] == 50:
            r.update(fired=True, total_updates=5400, total_transitions=5400 * 512, total_minibatch_steps=108000, total_dev_checks=216,
                     wall_sec=835.0, trig_Delta=0.006, trig_s=60.0, trig_C=0.03, trig_reasons="delta_gt", a2_final_Delta=0.001, passed=True)
    ct = {x["arm"]: x for x in an.cost_table(rows)}
    assert ct["bw"]["ratio_mean_updates"] == 1.0 and ct["bwA"]["mean_updates"] == pytest.approx((3800 * 39 + 5400) / 40)
    assert ct["bwA"]["ratio_mean_updates"] == pytest.approx(ct["bwA"]["mean_updates"] / 3800)
    assert ct["bwA"]["ratio_total_wall_sec"] == pytest.approx((600 * 39 + 835) / (600 * 40))
    f = an.fired_table(rows)
    assert len(f) == 2 and {x["arm"] for x in f} == {"bwA", "bwAB"} and f[0]["trigger_Delta"] == 0.006


def test_counts_table_worst_gmax_and_medians():
    rows = _world({})
    for r in rows:
        if r["run"] == "t3r3_q60_s70510_bwA":
            r.update(Gmax_full_over_dw=0.05, passed=False)
    t = {(x["arm"], x["q"]): x for x in an.counts_table(rows)}
    assert t[("bwA", 60)]["worst_Gmax_full"] == 0.05 and t[("bwA", 60)]["worst_Gmax_full_run"] == "t3r3_q60_s70510_bwA"
    assert t[("bwA", "both")]["pass"] == 39 and t[("bw", "both")]["pass"] == 40 and t[("bw", 50)]["median_eta_T_over_dw"] == 0.001
