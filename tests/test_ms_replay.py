"""MS-R1 offline calibration tool (tools/ms/replay_dev_rule.py): pure helpers, synthetic rule grid, one stored run.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_replay.py -p no:cacheprovider -q

Sections: 1. statistics and formatting helpers; 2. rule logic on synthetic checks (fire export, EMA, classification);
3. tables on synthetic runs; 4. CSV / refusal / preflight behaviour; 5. the replay on stored v2.0 exports
(skipped when the reference root is absent): bit-for-bit agreement with the run's own verifier logs.
"""

from __future__ import annotations

import copy
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import replay_dev_rule as RP  # noqa: E402
from envs.curriculum_env import GameSpec  # noqa: E402
from utils.ms_residual import StageDiag  # noqa: E402
from utils.ms_rule import StageRule  # noqa: E402

ROOT_V2 = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked")
REF_RUN = ROOT_V2 / "rehearsal_v2_0" / "q50" / "seed10501"
needs_ref = pytest.mark.skipif(not REF_RUN.is_dir(), reason="reference root absent")


# ====================================================================== 1. statistics and formatting
def test_rankdata_average_ties() -> None:
    assert RP.rankdata([10, 20, 20, 30]).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert RP.rankdata([3, 1, 2]).tolist() == [3.0, 1.0, 2.0]


def test_spearman_monotone_reversed_constant_nan() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert RP.spearman(x, [v ** 3 for v in x]) == pytest.approx(1.0, abs=1e-15)
    assert RP.spearman(x, [-v for v in x]) == pytest.approx(-1.0, abs=1e-15)
    assert math.isnan(RP.spearman(x, [1.0] * 5))                       # constant input
    assert math.isnan(RP.spearman([1.0, 2.0], [2.0, 1.0]))             # fewer than 3 pairs
    y = [1.0, 2.0, float("nan"), 4.0, 5.0]
    assert RP.spearman(x, y) == pytest.approx(1.0, abs=1e-15)          # non-finite pairs dropped


def test_spearman_matches_scipy_with_ties() -> None:
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(3)
    a = np.round(rng.normal(size=200), 1)
    b = np.round(a + rng.normal(size=200), 1)
    assert RP.spearman(a, b) == pytest.approx(stats.spearmanr(a, b).correlation, abs=1e-12)


def test_pearson_and_med_rng_and_formatting() -> None:
    assert RP.pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert RP.med_rng([3.0, float("nan"), 1.0, 2.0]) == (2.0, 1.0, 3.0)
    assert all(math.isnan(v) for v in RP.med_rng([float("nan")]))
    assert RP.fnum(float("nan")) == "-" and RP.fnum(0.12345, 3) == "0.123" and RP.fnum(3) == "3"
    assert RP.fnum(1.2e-7, 4) == "1.20e-07"
    assert RP.mr([1.0, 2.0, 3.0], 1) == "2.0 [1.0, 3.0]"
    md = RP.md_table(["a", "b"], [[1, 2]])
    assert md.splitlines()[0] == "| a | b |" and md.splitlines()[2] == "| 1 | 2 |"


def test_lr_of_v20_phase_b_is_the_linear_decay() -> None:
    assert RP.lr_v20_phase_b(1) == pytest.approx(3e-4)
    assert RP.lr_v20_phase_b(600) == pytest.approx(3e-5)


# ====================================================================== 2. rule logic on synthetic checks
def _diag(R: float = 0.01, delta: float = 0.001, tail: float = 0.01, C: float = 0.03, valid: bool = True,
          stage: int = 2, tail_term: bool = True) -> StageDiag:
    return StageDiag(stage=stage, valid=valid, delta_over_dw=delta, s=60.0, R=R, R_tail=tail, C=C,
                     tail_term=tail_term, rho_bins=np.full(20, 0.01), argmax_d=0.0)


def _seq(flags: List[bool], start: int = 25, step: int = 25) -> List:
    """(update, diag) with eligible / ineligible (R above rho) checks at start, start + step, ..."""
    return [(start + i * step, _diag(R=0.01 if ok else 0.5)) for i, ok in enumerate(flags)]


def test_fire_export_needs_m_consecutive_and_resets() -> None:
    rule = RP.new_rule(0.03)                      # M = 3, K = 25
    assert RP.fire_export(_seq([True, True, True]), rule) == 75
    assert RP.fire_export(_seq([True, True, False, True, True]), rule) is None
    assert RP.fire_export(_seq([True, True, False, True, True, True]), rule) == 150
    assert RP.fire_export(_seq([False, True, True, True, True]), rule) == 100     # first M-th, not later
    assert RP.fire_export([], rule) is None


def test_fire_export_cadence_and_offset() -> None:
    k100 = RP.original_rule(0.02, K=100)
    seq = _seq([True] * 16)                       # exports 25..400
    assert RP.fire_export(seq, k100) == 300       # checks only at 100, 200, 300
    rule = RP.new_rule(0.03, stage=1)
    b = [(1600 + 25 * (i + 1), _diag(stage=1, tail_term=False)) for i in range(5)]
    assert RP.fire_export(b, rule, offset=1600) == 1675
    k50 = RP.new_rule(0.03, K=50)
    assert RP.fire_export(b, k50, offset=1600) is None      # only 1650 and 1700 are checks (2 < M)


def test_eligibility_boundaries_are_inclusive_and_each_component_binds() -> None:
    rule = StageRule(stage=2, eps=0.005, rho=0.03, tau=0.02, M=1, K=25)
    ok = _diag(R=0.03, delta=0.005, tail=0.02, C=0.04)
    assert RP.eligible(rule, ok)
    for kw in ({"R": 0.0300001}, {"delta": 0.0050001}, {"tail": 0.0200001}, {"C": 0.0400001},
               {"valid": False}, {"R": float("nan")}):
        assert not RP.eligible(rule, _diag(**{**{"R": 0.03, "delta": 0.005, "tail": 0.02, "C": 0.04}, **kw}))


def test_eligibility_uses_the_pipeline_function(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[int] = []
    real = RP.is_eligible

    def spy(rule: StageRule, diag: StageDiag) -> bool:
        calls.append(1)
        return real(rule, diag)
    monkeypatch.setattr(RP, "is_eligible", spy)
    RP.fire_export(_seq([True, True, True]), RP.new_rule(0.03))
    assert len(calls) == 3


def test_original_rule_ignores_the_first_order_part_but_not_delta_validity_or_concentration() -> None:
    rule = RP.original_rule(0.005)
    huge = _diag(R=5.0, tail=5.0)                  # first-order part hopeless, second-order part fine
    assert RP.eligible(rule, huge)
    assert RP.fire_export([(25 * (i + 1), huge) for i in range(3)], rule) == 75
    assert not RP.eligible(rule, _diag(delta=0.0051))
    assert not RP.eligible(rule, _diag(C=0.0401))
    assert not RP.eligible(rule, _diag(valid=False))
    assert RP.eligible(RP.original_rule(0.02), _diag(delta=0.015))


def test_tail_term_is_void_when_the_stage_has_no_tail() -> None:
    d = _diag(stage=1, tail_term=False)
    d = StageDiag(**{**d.__dict__, "R_tail": float("nan")})
    assert RP.eligible(RP.new_rule(0.03, stage=1), d)
    d_tail = StageDiag(**{**d.__dict__, "tail_term": True})
    assert not RP.eligible(RP.new_rule(0.03, stage=2), d_tail)          # a NaN tail fails when the term applies


def _synth_rows(q: int, n_bins: int, seed: int, rho_fn, R_fn, delta_fn, peak_fn=None, label: str = "rehearsal_v2_0"
                ) -> List[Dict[str, object]]:
    """Phase-A (u25..u1600, dev) rows, the u1600 final row, Phase-B rows (u1625..u2200) of a synthetic run."""
    rows: List[Dict[str, object]] = []
    for u in RP.EXPORT_US:
        a = u <= RP.PHASE_A_END
        for tier in (["dev", "final"] if u == RP.PHASE_A_END else ["dev"]):
            rho = np.array([rho_fn(u, i) for i in range(n_bins)]) if a else np.full(n_bins, np.nan)
            row: Dict[str, object] = {
                "root": "/x/" + label, "q": q, "seed": seed, "update": u, "phase": "A" if a else "B",
                "stage": 2 if a else 1, "tier": tier, "valid": True, "res_valid": True,
                "Delta": delta_fn(u), "s": 60.0, "R": R_fn(u), "R_tail": 0.01, "C": 0.03,
                "R_argmax_d": 0.0, "tail_term": a}
            for i in range(n_bins):
                row[f"rho_bin_{i}"] = float(rho[i])
            pk = 0.06 if peak_fn is None else peak_fn(u)
            row.update(stage2_peak_rel_err_signed=-pk, stage2_peak_rel_err_abs=pk,
                       stage2_rmse_pos_over_g2_0=0.02, stage2_tail_mean_over_g2_0=0.01,
                       stage2_tail_max_over_g2_0=0.02, stage1_rel_err_signed=0.0, e1_at_0=47.0, e2_at_0=60.0,
                       eta_T_over_dw=delta_fn(u), Gmax_full_over_dw=0.001, verifier_sec=0.01, diag_sec=0.0005,
                       error="")
            rows.append(row)
    return rows


def _geo(q: int) -> RP.Geometry:
    proto = RP.load_protocol()
    return RP.geometry(GameSpec(**proto["records"][str(q)]["game"]))


def test_geometry_matches_the_brief_strata_counts() -> None:
    g50, g60 = _geo(50), _geo(60)
    assert (g50.n_bins, g50.n_nontail, g50.cap) == (40, 20, 5)
    assert (g60.n_bins, g60.n_nontail, g60.cap) == (44, 24, 6)
    for g in (g50, g60):
        assert int((g.labels == 0).sum()) == 20 and int((g.labels == 1).sum()) == 4
        assert abs(g.centers[g.labels == 1]).max() == 15.0                    # bins intersecting (-20, 20)
        assert g.outer == (int(np.flatnonzero(g.labels != 0)[0]), int(np.flatnonzero(g.labels != 0)[-1]))
    assert int((g50.labels == 2).sum()) == 16 and int((g60.labels == 2).sum()) == 20


def test_rho_bar_series_is_the_pipeline_ema() -> None:
    rows = _synth_rows(50, 40, 1, rho_fn=lambda u, i: 0.1 if u <= 50 else 0.0, R_fn=lambda u: 0.1,
                       delta_fn=lambda u: 0.001)
    ser = RP.rho_bar_series([r for r in rows if r["phase"] == "A" and r["tier"] == "dev"], 0.5)
    nontail = ~np.isnan(ser[25])
    assert ser[25][nontail].tolist() == [0.1] * int(nontail.sum())                    # initialised at u25
    assert ser[50][nontail][0] == pytest.approx(0.1)                                  # 0.5 * 0.1 + 0.5 * 0.1
    assert ser[75][nontail][0] == pytest.approx(0.05)                                 # 0.5 * 0.1 + 0.5 * 0
    assert ser[100][nontail][0] == pytest.approx(0.025)


def test_rho_bar_series_skips_invalid_checks() -> None:
    rows = _synth_rows(50, 40, 1, rho_fn=lambda u, i: 0.1, R_fn=lambda u: 0.1, delta_fn=lambda u: 0.001)
    a = [copy.deepcopy(r) for r in rows if r["phase"] == "A" and r["tier"] == "dev"]
    a[1]["valid"] = False
    for i in range(40):
        a[1][f"rho_bin_{i}"] = 9.0                                                    # must not enter the EMA
    ser = RP.rho_bar_series(a, 0.5)
    assert ser[50][0] == pytest.approx(0.1)


def test_classify_export_localized_broad_and_location() -> None:
    g = _geo(50)
    nt = np.flatnonzero(g.labels != 0)
    rb = np.full(g.n_bins, np.nan)
    rb[nt] = 0.01
    rb[[18, 19]] = 0.08                                    # two near-tie bins above rho
    c = RP.classify_export(rb, 0.08, g, 0.03)
    assert c["class"] == "localized" and c["n_S"] == 2 and c["n_S_near"] == 2 and c["n_S_mid"] == 0
    assert c["n_S_neg"] == 2 and c["n_S_pos"] == 0 and c["S_bins"] == "18;19"
    assert RP.classify_export(rb, 0.02, g, 0.03)["class"] == "broad"                  # R <= rho clause
    rb2 = np.full(g.n_bins, np.nan)
    rb2[nt] = 0.01
    rb2[nt[:6]] = 0.08                                     # 6 of 20 > cap 5
    c2 = RP.classify_export(rb2, 0.08, g, 0.03)
    assert c2["class"] == "broad" and c2["n_S"] == 6 and not c2["S_within_cap_ignoring_R"]
    rb3 = np.full(g.n_bins, np.nan)
    rb3[nt] = 0.01
    c3 = RP.classify_export(rb3, 0.08, g, 0.03)
    assert c3["class"] == "broad" and c3["n_S"] == 0
    c4 = RP.classify_export(None, 0.08, g, 0.03)
    assert c4["class"] == "broad" and c4["n_S"] == 0
    assert RP.argmax_region(-10.0, g) == "near-tie" and RP.argmax_region(60.0, g) == "middle"
    assert RP.argmax_region(-96.0, g) == "boundary" and RP.argmax_region(90.0, g) == "boundary"


def test_rule_decisions_do_not_read_the_closed_form_columns() -> None:
    rows = _synth_rows(50, 40, 1, rho_fn=lambda u, i: 0.01, R_fn=lambda u: 0.01 if u >= 300 else 0.5,
                       delta_fn=lambda u: 0.001)
    a = [r for r in rows if r["phase"] == "A" and r["tier"] == "dev"]
    stub = copy.deepcopy(a)
    for r in stub:
        for c in RP.CLOSED_FORM_COLS:
            r[c] = float("nan")
    for rho in RP.RHO_GRID:
        rule = RP.new_rule(rho)
        assert RP.fire_export(RP.run_diags(a), rule) == RP.fire_export(RP.run_diags(stub), rule) == 350
    ser = RP.rho_bar_series(a)
    ser_stub = RP.rho_bar_series(stub)
    assert all(np.array_equal(ser[u], ser_stub[u], equal_nan=True) for u in ser)


# ====================================================================== 3. tables on synthetic runs
def _synth_run(q: int, seed: int, fire_at: int, peak_after: float = 0.04) -> RP.RunData:
    n_bins = 40 if q == 50 else 44
    rows = _synth_rows(q, n_bins, seed, rho_fn=lambda u, i: 0.2 if u < fire_at else 0.01,
                       R_fn=lambda u: 0.2 if u < fire_at else 0.01, delta_fn=lambda u: 0.001,
                       peak_fn=lambda u: 0.1 if u < fire_at else peak_after)
    return RP.RunData("rehearsal_v2_0", q, seed, rows)


def test_rule_outcome_and_summary_on_synthetic_runs() -> None:
    runs = [_synth_run(50, 1, fire_at=300), _synth_run(50, 2, fire_at=600), _synth_run(60, 3, fire_at=5000)]
    rule = RP.new_rule(0.03)
    outs = [RP.rule_outcome(r, rule, "x") for r in runs]
    assert [o["fire_update"] for o in outs[:2]] == [350.0, 650.0]          # first eligible 300 + 2 more checks
    assert outs[0]["peak_fire"] == 0.04 and outs[0]["peak_u1600"] == 0.04
    assert outs[2]["fired"] is False and math.isnan(outs[2]["fire_update"])
    s = RP.summarize_outcomes(outs)
    assert (s["n_runs"], s["n_fire"], s["n_never"]) == (3, 2, 1)
    assert s["fire_update_med"] == 500.0 and (s["fire_update_min"], s["fire_update_max"]) == (350.0, 650.0)
    assert s["n_ok_ga_parts"] == 2 and s["frac_ok_of_all"] == pytest.approx(2 / 3)
    assert s["n_fire_after_1200"] == 0


def test_table2_and_table1_on_synthetic_runs() -> None:
    runs = [_synth_run(50, 1, fire_at=300), _synth_run(60, 2, fire_at=300, peak_after=0.045)]
    t2, t2r = RP.table2(runs)
    names = [n for n, _ in RP.rule_grid()]
    assert len(t2r.csv_rows) == len(names) * 2
    row = [r for r in t2.csv_rows if r["rule"] == "new rho=0.03" and r["group"] == "pooled"][0]
    assert row["n_fire"] == 2 and row["fire_update_med"] == 350.0
    orig = [r for r in t2.csv_rows if r["rule"] == "orig eps=0.02" and r["group"] == "pooled"][0]
    assert orig["fire_update_med"] == 75.0                                  # delta 0.001 <= 0.02 from the start
    geos = {50: _geo(50), 60: _geo(60)}
    runs9 = [_synth_run(50, 1, fire_at=900), _synth_run(60, 2, fire_at=900)]    # a step inside u >= 400
    t1 = RP.table1(runs9, geos)
    pr = [r for r in t1.csv_rows if r["primary"] and r["group"] == "pooled"]
    assert {r["y"] for r in pr} == {"R", "Delta"} and all(r["n_pairs"] == 98 for r in pr)
    # |peak| and R are both the same step function of u in the synthetic runs -> Spearman 1 (R vs peak);
    # Delta is constant there -> undefined
    assert [r for r in pr if r["y"] == "R"][0]["spearman_pooled"] == pytest.approx(1.0)
    assert math.isnan([r for r in pr if r["y"] == "Delta"][0]["spearman_pooled"])
    sup = [r for r in t1.csv_rows if r["y"] == "R_near" and r["group"] == "pooled" and r["subset"] == "u>=400"][0]
    assert sup["supplementary_metric"] is True and sup["spearman_pooled"] == pytest.approx(1.0)


def test_table2b_component_rates_and_interior_residual() -> None:
    geos = {50: _geo(50), 60: _geo(60)}
    rows = _synth_rows(50, 40, 1, rho_fn=lambda u, i: 0.01, R_fn=lambda u: 0.01, delta_fn=lambda u: 0.001)
    a = [r for r in rows if r["phase"] == "A" and r["tier"] == "dev"][0]
    assert RP._r_interior(a, geos[50]) == pytest.approx(0.01)
    rows = _synth_rows(50, 40, 1, rho_fn=lambda u, i: 0.5 if i in (geos[50].outer[0], geos[50].outer[1]) else 0.01,
                       R_fn=lambda u: 0.5, delta_fn=lambda u: 0.001)
    a = [r for r in rows if r["phase"] == "A" and r["tier"] == "dev"][0]
    assert RP._r_interior(a, geos[50]) == pytest.approx(0.01)       # the two boundary-adjacent bins are excluded
    t = RP.table2b([RP.RunData("rehearsal_v2_0", 50, 1, rows)], geos)         # q60 group is empty -> skipped
    assert {x["group"] for x in t.csv_rows} == {"pooled", "q50"}
    r = [x for x in t.csv_rows if x["scope"] == "u>=400" and x["group"] == "pooled"][0]
    assert r["R<=0.05"] == 0.0 and r["R_int<=0.03"] == 1.0 and r["Delta<=0.005"] == 1.0


def test_table3_dev_vs_final_on_synthetic_rows() -> None:
    geos = {50: _geo(50), 60: _geo(60)}
    run = _synth_run(50, 1, fire_at=300)
    fin = run.a_final
    assert fin is not None
    fin["R"] = 0.012
    fin["rho_bin_10"] = float(run.a_by_u[1600]["rho_bin_10"]) + 0.02
    t, tp = RP.table3([run], geos)
    p = tp.csv_rows[0]
    assert p["R_dev"] == 0.01 and p["R_final"] == 0.012 and p["dR"] == pytest.approx(0.002)
    assert p["bin_maxabs_diff"] == pytest.approx(0.02) and p["bin_maxabs_diff_bin"] == 10
    assert p["flip_R<=0.02"] is False


def test_table5_fire_budget_arithmetic() -> None:
    q = 50
    rows = _synth_rows(q, 40, 1, rho_fn=lambda u, i: 0.01, R_fn=lambda u: 0.01, delta_fn=lambda u: 0.001)
    for r in rows:
        if r["phase"] == "B":
            r["R"] = 0.01 if r["update"] >= 1700 else 0.5          # local 100 onward eligible
            r["stage1_rel_err_signed"] = 0.01
            r["s"] = 47.0
    run = RP.RunData("rehearsal_v2_0", q, 1, rows)
    _, _, _, tf = RP.table5([run])
    r = [x for x in tf.csv_rows if x["rho"] == 0.03 and x["group"] == "pooled"][0]
    assert r["n_fire"] == 1 and r["fire_local_median"] == 150.0           # checks at local 100, 125, 150
    assert r["saved_training_median"] == 450.0 and r["total_with_land400_median"] == 550.0


# ====================================================================== 4. csv, refusals, preflight
def test_csv_round_trip_and_overwrite_refusal(tmp_path: Path) -> None:
    rows = _synth_rows(50, 40, 7, rho_fn=lambda u, i: 0.01 * (i + 1), R_fn=lambda u: 1 / 3,
                       delta_fn=lambda u: 1e-3 / 7)
    p = str(tmp_path / "sub" / "run.csv")
    RP.write_csv(p, rows, RP.csv_fields(40))
    back = RP.read_csv_rows(p)
    assert len(back) == len(rows) == 89
    assert back[0]["R"] == 1 / 3 and back[0]["Delta"] == 1e-3 / 7                    # repr round trip, exact
    assert back[0]["valid"] is True and back[0]["tier"] == "dev" and back[0]["q"] == 50
    assert math.isnan(back[-1]["rho_bin_0"]) and back[-1]["phase"] == "B"
    with pytest.raises(FileExistsError):
        RP.write_csv(p, rows, RP.csv_fields(40))
    RP.write_csv(p, rows[:3], RP.csv_fields(40), overwrite=True)
    assert len(RP.read_csv_rows(p)) == 3


def test_missing_reference_file_stops_the_replay_and_writes_nothing(tmp_path: Path, capsys) -> None:
    root = tmp_path / "v2"
    for lab, seeds in RP.EXPECTED_SEEDS.items():
        for q in RP.QS:
            for s in seeds:
                (root / lab / f"q{q}" / f"seed{s}").mkdir(parents=True)
    ref = RP.RunRef("rehearsal_v2_0", str(root / "rehearsal_v2_0"), 50, 10501)
    assert len(RP.missing_files(ref)) == len(RP.REQUIRED_FILES) + len(RP.EXPORT_US)
    assert RP.layout_problems(RP.discover_runs(str(root))) == []
    out = tmp_path / "out"
    rc = RP.main(["replay", "--root-v2", str(root), "--out", str(out)])
    assert rc == 2 and not out.exists()
    assert "missing" in capsys.readouterr().out


def test_layout_problems_reports_wrong_seed_sets(tmp_path: Path) -> None:
    root = tmp_path / "v2"
    for s in range(10501, 10510):                                    # one seed short
        (root / "rehearsal_v2_0" / "q50" / f"seed{s}").mkdir(parents=True)
    probs = RP.layout_problems(RP.discover_runs(str(root), ("rehearsal_v2_0",)), ("rehearsal_v2_0",))
    assert any("rehearsal_v2_0/q50" in p for p in probs)


def test_replay_refuses_an_output_dir_inside_the_reference_root(tmp_path: Path, capsys) -> None:
    pytest.importorskip("torch")
    if not REF_RUN.is_dir():
        pytest.skip("reference root absent")
    rc = RP.main(["replay", "--root-v2", str(ROOT_V2), "--out", str(ROOT_V2 / "ms_out"),
                  "--only", "rehearsal_v2_0/q50/seed10501"])
    assert rc == 2 and not (ROOT_V2 / "ms_out").exists()
    assert "inside the (read-only) reference root" in capsys.readouterr().out


# ====================================================================== 5. replay on stored exports
@pytest.fixture(scope="module")
def replayed_ref_run():
    """One full replay (89 verifier calls, ~3 s) of rehearsal_v2_0 q50 seed10501."""
    if not REF_RUN.is_dir():
        pytest.skip("reference root absent")
    proto = RP.load_protocol()
    rec = proto["records"]["50"]
    spec = GameSpec(**rec["game"])
    ref = RP.RunRef("rehearsal_v2_0", str(ROOT_V2 / "rehearsal_v2_0"), 50, 10501)
    return ref, spec, RP.replay_exports(ref, spec, rec["ppo"])


@needs_ref
def test_replay_reproduces_the_runs_own_verifier_logs_bit_for_bit(replayed_ref_run) -> None:
    ref, _, out = replayed_ref_run
    summ, mism = RP.validate_run(ref, out)
    assert summ["gate_ok"] is True and summ["gate_eta_absdiff"] == 0.0
    assert summ["n_compared"] == summ["n_bitwise_equal"] == 444 and mism == []     # 37 calls x 12 fields


@needs_ref
def test_replay_row_structure_and_d3_invariants(replayed_ref_run) -> None:
    ref, spec, out = replayed_ref_run
    rows = out.rows
    assert len(rows) == 89 and sum(r["tier"] == "final" for r in rows) == 1
    assert [r["update"] for r in rows if r["tier"] == "dev"] == list(RP.EXPORT_US)
    assert set(r["phase"] for r in rows if r["update"] <= 1600) == {"A"} and all(
        r["stage"] == 1 for r in rows if r["update"] > 1600)
    assert [r for r in rows if r["tier"] == "final"][0]["update"] == 1600
    fields = RP.csv_fields(40)
    assert all(set(r.keys()) == set(fields) for r in rows)
    for r in rows:
        assert r["valid"] and r["error"] == "" and r["root"] == ref.root_dir
        if r["stage"] == 2:
            rho = np.array([r[f"rho_bin_{i}"] for i in range(40)])
            assert np.isnan(rho[:10]).all() and np.isnan(rho[30:]).all()               # tail bins
            assert np.isfinite(rho[10:30]).all() and r["R"] == np.nanmax(rho)           # R = max of the bin map
            assert r["tail_term"] is True and np.isfinite(r["R_tail"])
        else:
            assert np.isnan([r[f"rho_bin_{i}"] for i in range(40)]).all() and r["tail_term"] is False
            assert r["R"] == pytest.approx(abs(r["e1_at_0"] - r["s"]) / r["s"], abs=1e-12)


@needs_ref
def test_replay_stage1_uses_the_frozen_u1600_network_at_stage2(replayed_ref_run) -> None:
    _, _, out = replayed_ref_run
    ra = [r for r in out.rows if r["update"] == 1600 and r["tier"] == "dev"][0]
    for r in out.rows:
        if r["phase"] == "B":                      # the frozen terminal stage never changes in Phase B
            assert r["stage2_peak_rel_err_signed"] == ra["stage2_peak_rel_err_signed"]
            assert r["eta_T_over_dw"] == ra["eta_T_over_dw"] and r["e2_at_0"] == ra["e2_at_0"]


@needs_ref
def test_replay_final_tier_row_differs_only_by_grid(replayed_ref_run) -> None:
    _, _, out = replayed_ref_run
    d = [r for r in out.rows if r["update"] == 1600 and r["tier"] == "dev"][0]
    f = [r for r in out.rows if r["tier"] == "final"][0]
    assert f["stage2_peak_rel_err_signed"] == d["stage2_peak_rel_err_signed"]      # closed form on the same 0.5 grid
    assert abs(f["R"] - d["R"]) < 0.05 and abs(f["Delta"] - d["Delta"]) < 1e-3
    assert f["verifier_sec"] > 0 and d["verifier_sec"] > 0


@needs_ref
def test_actor_rebuild_is_bit_identical_to_the_export_forward_pass() -> None:
    """The BetaActor forward pass on the exported float32 arrays equals ``mean_effort_numpy`` to float32 rounding."""
    from agents.ppo_curriculum import mean_effort_numpy
    arrays = RP.load_export(str(REF_RUN / "weights" / "u01600.npz"))
    net = RP.build_actor(arrays)
    spec = GameSpec(**RP.load_protocol()["records"]["50"]["game"])
    mean_fn, _ = RP.candidate_fns(spec, {2: net})
    d = np.linspace(-200.0, 200.0, 101)
    ref, _, _ = mean_effort_numpy(arrays, spec.encode_obs(2, d))
    got = mean_fn(2, d)
    assert np.max(np.abs(got - ref)) < 1e-4                   # the numpy re-implementation is not bitwise
    assert np.array_equal(got, mean_fn(2, d))                 # deterministic


@needs_ref
def test_fact3_numbers_of_the_preamble() -> None:
    f = RP.fact3(str(ROOT_V2), RP.load_protocol())
    ind = f["independent_F_xi"]
    assert f["gates_json_e2_at_0"] == pytest.approx(66.45, abs=0.01)
    assert ind["best_response"] == pytest.approx(68.54, abs=0.01)
    assert ind["best_response_minus_e_hat"] == pytest.approx(2.09, abs=0.005)
    assert ind["gain_at_0_over_dw"] == pytest.approx(5.3e-4, abs=0.05e-4)
    assert f["verifier_final"]["a_dev_2_0"] == pytest.approx(ind["best_response"], abs=1e-6)
