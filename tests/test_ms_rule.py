"""Tests of the MS-R1 development stop rule (spec 2.2 "Rule (D4)"): ``utils/ms_rule.py``.

Run:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_rule.py -q

Pure state-machine tests on synthetic ``StageDiag`` sequences (no training, no verifier). The
controller protocol under test: before local update ``j`` the loop reads ``ctrl.lr(j)`` and
``ctrl.sampler_setting()``; after it, ``ctrl.after_update(j, diag_fn)``; ``diag_fn`` is called only
when a check is due. The landing LR is ``run.run_final_dp_br_round3_dense.lr_at`` in its linear
form,
called with the dict that ``run.run_v2_stagewise.Run.lr_for`` builds.

Terminology of the synthetic maps: stage 2 of T = 2 at q = 50 has 40 ES bins; bins 0-9 and 30-39 are
tail bins (NaN in ``rho_bins``), bins 10-29 are the 20 non-tail bins, so ``n_nontail_bins = 20`` and
the localized cap is ceil(0.25 * 20) = 5.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ.setdefault(f"{_k}_NUM_THREADS", "1")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run.run_final_dp_br_round3_dense import lr_at  # noqa: E402
from utils.ms_residual import StageDiag  # noqa: E402
from utils.ms_rule import (SamplerSetting, StageController, StageRule, Step,  # noqa: E402
                           classify_block_end, is_eligible)

LR_BASE, LR_END = 3e-4, 3e-5
NT = 20                              # non-tail bins at q = 50, T = 2, stage 2
NBINS = 40


def lr_linear(start: float, end: float, first: int, last: int, j: int) -> float:
    """The landing LR as ``Run.lr_for`` builds it: lr_at's linear branch over [first, last]."""
    sched = {"ab_lr": LR_BASE, "kind": "linear", "c_start_lr": float(start), "c_end_lr": float(end),
             "c_local_first": int(first), "linear_denominator": int(last) - int(first)}
    return lr_at(sched, "C", j)


def lr_must_not_be_called(*args: object) -> float:
    raise AssertionError("the linear landing LR must not be used by this controller")


def rb(vals: Optional[Dict[int, float]] = None, default: float = 0.01) -> np.ndarray:
    """Per-bin residual map: NaN on the 20 tail bins, ``default`` on bins 10-29, with overrides."""
    out = np.full(NBINS, np.nan)
    out[10:30] = default
    for b, v in (vals or {}).items():
        out[b] = v
    return out


def mk(delta: float = 0.0, R: float = 0.0, R_tail: float = 0.0, C: float = 0.0, valid: bool = True,
       tail_term: bool = True, rho_bins: Optional[np.ndarray] = None, stage: int = 2) -> StageDiag:
    """A StageDiag; the defaults are an eligible check (all gates met with margin)."""
    return StageDiag(stage=stage, valid=valid, delta_over_dw=delta, s=50.0, R=R, R_tail=R_tail, C=C,
                     tail_term=tail_term, rho_bins=rb() if rho_bins is None else rho_bins)


GOOD = mk()
BAD_BROAD = mk(R=0.10, rho_bins=rb({10 + i: 0.10 for i in range(8)}))   # ineligible, 8 bins > rho
BAD_LOC = mk(R=0.10, rho_bins=rb({12: 0.10, 13: 0.08}))                 # ineligible, 2 bins > rho
BAD_R = BAD_BROAD


def rule_of(**kw: object) -> StageRule:
    """A stage-2 rule with D4 defaults unless overridden (n_nontail_bins = 20)."""
    kw.setdefault("stage", 2)
    kw.setdefault("n_nontail_bins", NT)
    return StageRule(**kw)           # type: ignore[arg-type]


def ctrl_of(**kw: object) -> StageController:
    return StageController(rule_of(**kw), lr_linear)


class Row:
    """What the training loop sees at one local update."""

    def __init__(self, j: int, lr: float, setting: SamplerSetting, label, step: Step):
        self.j, self.lr, self.setting, self.label, self.step = j, lr, setting, label, step


def drive(ctrl: StageController, diag_for: Callable[[int], StageDiag], max_j: int = 5000
          ) -> List[Row]:
    """Run the controller until the freeze; ``diag_for(j)`` is the check result at update j."""
    rows: List[Row] = []
    j = 0
    while not ctrl.finished and j < max_j:
        j += 1
        lr, st, label = ctrl.lr(j), ctrl.sampler_setting(), ctrl.block_label()
        step = ctrl.after_update(j, lambda jj=j: diag_for(jj))
        rows.append(Row(j, lr, st, label, step))
    return rows


def seq(diags: Sequence[StageDiag]) -> Callable[[int], StageDiag]:
    """``diag_for`` that returns the diags in order of the checks, repeating the last one."""
    state = {"i": 0}

    def f(j: int) -> StageDiag:
        d = diags[min(state["i"], len(diags) - 1)]
        state["i"] += 1
        return d
    return f


def check_js(rows: List[Row]) -> List[int]:
    return [r.j for r in rows if r.step.check is not None]


# ----------------------------------------------------------------------------------------------
# eligibility
# ----------------------------------------------------------------------------------------------

def test_eligibility_comparisons_are_inclusive_and_float64():
    rule = rule_of()
    up = lambda x: float(np.nextafter(x, np.inf))             # noqa: E731
    assert is_eligible(rule, mk(delta=rule.eps)) and not is_eligible(rule, mk(delta=up(rule.eps)))
    assert is_eligible(rule, mk(R=rule.rho)) and not is_eligible(rule, mk(R=up(rule.rho)))
    assert is_eligible(rule, mk(R_tail=rule.tau)) and not is_eligible(rule, mk(R_tail=up(rule.tau)))
    assert is_eligible(rule, mk(C=rule.conc_limit))
    assert not is_eligible(rule, mk(C=up(rule.conc_limit)))
    assert is_eligible(rule, mk(delta=0.005, R=0.03, R_tail=0.02, C=0.04))      # all at the limits
    assert (rule.eps, rule.rho, rule.tau, rule.conc_limit) == (0.005, 0.03, 0.02, 0.04)


def test_tail_term_void_flag():
    rule = rule_of()
    assert not is_eligible(rule, mk(R_tail=0.5, tail_term=True))
    assert not is_eligible(rule, mk(R_tail=float("nan"), tail_term=True))
    # void: large is fine
    assert is_eligible(rule, mk(R_tail=0.5, tail_term=False))
    assert is_eligible(rule, mk(R_tail=float("nan"), tail_term=False))          # void: NaN is fine
    assert is_eligible(rule, mk(R_tail=float("nan"), tail_term=False, stage=1,
                                rho_bins=np.zeros(0)))


@pytest.mark.parametrize("field", ["delta", "R", "C"])
def test_nan_and_invalid_diag_is_never_eligible(field):
    rule = rule_of()
    assert not is_eligible(rule, mk(**{field: float("nan")}))
    assert not is_eligible(rule, mk(valid=False))
    assert not is_eligible(rule, mk(valid=False, delta=0.0, R=0.0, R_tail=0.0, C=0.0))
    # the bare placeholder
    assert not is_eligible(rule, StageDiag(stage=2, valid=False))


# ----------------------------------------------------------------------------------------------
# the M-consecutive logic
# ----------------------------------------------------------------------------------------------

def run_elig_pattern(pattern: str, K: int = 5, M: int = 3, **kw: object):
    """Checks follow ``pattern`` (E eligible, N not, I invalid); the last symbol repeats."""
    table = {"E": GOOD, "N": BAD_R, "I": mk(valid=False)}
    ctrl = ctrl_of(K=K, M=M, n_block=1000, u_cap=2000, n_land=4, **kw)
    rows = drive(ctrl, seq([table[c] for c in pattern]))
    return ctrl, rows


def test_m_consecutive_streak_fires_at_the_third_eligible_check():
    ctrl, rows = run_elig_pattern("EE")                         # E, E, E, ... -> fires at 3rd check
    assert ctrl.fire_local == 15 and ctrl.would_fire == 15
    cons = [r.step.check["consecutive"] for r in rows if r.step.check][:3]
    assert cons == [1, 2, 3]


def test_a_single_ineligible_check_breaks_the_streak():
    ctrl, rows = run_elig_pattern("EENEEE")                     # streak 1, 2, 0, 1, 2, 3
    chk = [r.step.check for r in rows if r.step.check]
    assert [c["consecutive"] for c in chk[:6]] == [1, 2, 0, 1, 2, 3]
    assert [c["eligible"] for c in chk[:6]] == [True, True, False, True, True, True]
    assert ctrl.fire_local == 30


def test_an_invalid_check_breaks_the_streak_and_nan_is_ineligible():
    ctrl, rows = run_elig_pattern("EEIEEE")
    assert ctrl.fire_local == 30
    nan = StageDiag(stage=2, valid=True, delta_over_dw=float("nan"), s=1.0, R=0.0, R_tail=0.0,
                    C=0.0, tail_term=True, rho_bins=rb())
    ctrl2 = ctrl_of(K=5, M=3, n_block=1000, u_cap=40, n_land=4)
    drive(ctrl2, seq([GOOD, GOOD, nan]))                        # never 3 in a row -> forced
    assert ctrl2.fire_local is None and ctrl2.budget_forced


@pytest.mark.parametrize("M", [1, 2, 4])
def test_other_values_of_M(M):
    ctrl, _ = run_elig_pattern("E", M=M)
    assert ctrl.fire_local == 5 * M


def test_the_streak_is_not_reset_by_a_block_boundary():
    """K = 5, n_block = 12: checks at 5, 10, 12 (block end), 15. Eligible at 10, 12, 15 only: the
    streak 1, 2 crosses the end of block 1 (no stop at j = 12 since 2 < M) and fires at 15."""
    elig = {10, 12, 15, 20, 25}
    ctrl = ctrl_of(K=5, M=3, n_block=12, u_cap=100, n_land=4)
    rows = drive(ctrl, lambda j: GOOD if j in elig else BAD_R)
    assert ctrl.fire_local == 15
    b = ctrl.blocks
    assert [(x["first_local"], x["last_local"], x["exit_reason"]) for x in b] == [
        (1, 12, "block_end"), (13, 15, "development_stop")]
    chk = {r.j: r.step.check for r in rows if r.step.check}
    assert [chk[j]["consecutive"] for j in (5, 10, 12, 15)] == [0, 1, 2, 3]


def test_an_ineligible_block_end_check_resets_the_streak():
    # 12 (block end) is ineligible: 5 -> 1, 10 -> 2, 12 -> 0
    elig = {5, 10, 15}
    ctrl = ctrl_of(K=5, M=3, n_block=12, u_cap=100, n_land=4)
    drive(ctrl, lambda j: GOOD if j in elig else BAD_R)
    assert ctrl.fire_local is None and ctrl.budget_forced


# ----------------------------------------------------------------------------------------------
# stop inside the first block; landing
# ----------------------------------------------------------------------------------------------

def test_stop_inside_the_first_block():
    ctrl = ctrl_of(K=5, M=3, n_block=40, u_cap=100, n_land=10)
    rows = drive(ctrl, seq([GOOD]))
    assert [r.j for r in rows][-1] == 25 and ctrl.finished
    rec = ctrl.record()
    b0 = rec["blocks"][0]
    assert len(rec["blocks"]) == 1
    assert (b0["type"], b0["first_local"], b0["last_local"]) == ("global", 1, 15)
    assert b0["exit_reason"] == "development_stop" and b0["fire_local"] == 15
    assert b0["classification"] is None
    land = rec["landing"]
    assert (land["first_local"], land["last_local"], land["n_land"]) == (16, 25, 10)
    assert land["budget_forced"] is False and land["followed_block_type"] == "global"
    assert land["followed_block_id"] == 1 and land["done"] is True
    assert rec["fire_local"] == 15 and rec["would_fire_local"] == 15
    assert rec["budget_forced"] is False
    assert rec["training_updates"] == 15 and rec["total_updates"] == 25 and rec["n_checks"] == 5
    events = [e for r in rows for e in r.step.events]
    assert events == ["development_stop", "freeze"]
    assert rows[14].step.events == ["development_stop"] and rows[-1].step.finished


def test_stop_at_the_block_end_check_wins_over_the_block_end_classification():
    ctrl = ctrl_of(K=5, M=3, n_block=15, u_cap=100, n_land=4)
    drive(ctrl, seq([GOOD]))
    b0 = ctrl.blocks[0]
    assert b0["exit_reason"] == "development_stop" and b0["classification"] is None
    assert len(ctrl.blocks) == 1 and not ctrl.budget_forced


def test_no_update_after_the_freeze():
    ctrl = ctrl_of(K=5, M=3, n_block=40, u_cap=100, n_land=10)
    drive(ctrl, seq([GOOD]))
    assert ctrl.finished
    with pytest.raises(RuntimeError, match="after the freeze"):
        ctrl.after_update(26, lambda: GOOD)
    legacy = StageController(rule_of(enabled=False, fixed_budget=10, K=5), lr_must_not_be_called,
                             lambda j: LR_BASE)
    drive(legacy, seq([GOOD]))
    with pytest.raises(RuntimeError, match="after the freeze"):
        legacy.after_update(11, lambda: GOOD)


def test_landing_lr_follows_lr_at_linear_form_and_has_n_land_updates():
    n_land = 400
    ctrl = ctrl_of(K=25, M=3, n_block=400, u_cap=2000, n_land=n_land)
    rows = drive(ctrl, seq([GOOD]))
    fire = ctrl.fire_local
    assert fire == 75
    land = ctrl.landing
    assert land["first_local"] == 76 and land["last_local"] == 75 + n_land
    train_lrs = [r.lr for r in rows if r.j <= fire]
    land_rows = [r for r in rows if r.j > fire]
    assert len(land_rows) == n_land and rows[-1].j == fire + n_land
    assert all(x == LR_BASE for x in train_lrs)                         # constant 3e-4 in training
    lrs = np.array([r.lr for r in land_rows])
    # 3e-4 at the first landing update
    assert lrs[0] == LR_BASE
    assert lrs[-1] == pytest.approx(LR_END, rel=1e-12, abs=0)           # 3e-5 at the last
    first, last = land["first_local"], land["last_local"]
    want = LR_BASE + (LR_END - LR_BASE) * (np.arange(first, last + 1) - first) / (n_land - 1)
    np.testing.assert_allclose(lrs, want, rtol=1e-12, atol=0)
    assert np.all(np.diff(lrs) < 0) and np.allclose(np.diff(lrs), np.diff(lrs)[0], rtol=1e-9)
    # the same numbers from lr_at with the dict Run.lr_for builds (linear window [first, last])
    sched = {"ab_lr": LR_BASE, "kind": "linear", "c_start_lr": LR_BASE, "c_end_lr": LR_END,
             "c_local_first": first, "linear_denominator": last - first}
    assert [r.lr for r in land_rows] == [lr_at(sched, "C", r.j) for r in land_rows]


def test_landing_sampler_setting_equals_the_followed_block_and_is_frozen():
    """Followed block = global (alpha_global = 0.5, focus rho_bar): the landing setting is that
    setting, frozen at the stop; later checks (different residual maps) do not change it."""
    stop_map = rb({12: 0.08, 13: 0.04})
    later_map = rb({25: 0.9, 26: 0.7})                                  # very different
    ctrl = ctrl_of(K=5, M=3, n_block=40, u_cap=100, n_land=20, alpha_global=0.5)
    rows = drive(ctrl, lambda j: mk(R=0.02, rho_bins=stop_map if j <= 15 else later_map))
    assert ctrl.fire_local == 15
    # every global-block update before the stop uses (global, alpha_global, focus rho_bar or None)
    pre = [r.setting for r in rows if r.j <= 15]
    assert all(s.block_type == "global" and s.alpha == 0.5 for s in pre)
    assert all(s.focus is None for s in pre[:5])                         # before the first check
    expect_focus = np.nan_to_num(stop_map, nan=0.0)
    np.testing.assert_array_equal(pre[-1].focus, expect_focus)           # EMA of a constant map
    land_rows = [r for r in rows if r.j > 15]
    assert len(land_rows) == 20
    for r in land_rows:
        s = r.setting
        assert s.block_type == "landing" and s.followed_type == "global" and s.alpha == 0.5
        np.testing.assert_array_equal(s.focus, expect_focus)             # snapshot frozen
        assert r.label == (2, "landing")
    assert ctrl.landing["followed_block_type"] == "global" and ctrl.landing["alpha"] == 0.5
    assert ctrl.rho_bar is not None and ctrl.rho_bar[25] > 0.0           # checks did go on ...
    assert any(r.step.check is not None for r in land_rows)              # ... during landing
    assert ctrl.fire_local == 15 and ctrl.would_fire == 15              # ... and decided nothing


def test_stop_inside_a_polishing_block_lands_with_the_polishing_setting():
    S_map = rb({13: 0.06, 14: 0.08})
    ctrl = ctrl_of(K=5, M=3, n_block=20, u_cap=100, n_land=6, alpha_polish=0.4)
    rows = drive(ctrl, lambda j: mk(R=0.08, rho_bins=S_map) if j <= 20 else mk(rho_bins=S_map))
    assert [b["type"] for b in ctrl.blocks] == ["global", "polish"]
    assert ctrl.blocks[1]["exit_reason"] == "development_stop" and ctrl.fire_local == 35
    assert ctrl.blocks[1]["S"] == [13, 14] and not ctrl.budget_forced     # S kept on a stop
    land = [r for r in rows if r.j > 35]
    assert len(land) == 6
    for r in land:
        assert r.setting.followed_type == "polish" and r.setting.alpha == 0.4
        assert set(np.flatnonzero(r.setting.focus)) == {13, 14}
    assert ctrl.landing["followed_block_id"] == 2
    assert ctrl.landing["followed_block_type"] == "polish"


# ----------------------------------------------------------------------------------------------
# block end without a stop: localized -> polish, broad -> global
# ----------------------------------------------------------------------------------------------

def test_block_end_localized_goes_to_a_polishing_block_with_S_frozen():
    S_map = rb({13: 0.06, 14: 0.08, 22: 0.05})                            # 3 bins above rho = 0.03
    other = rb({20: 0.9, 21: 0.9, 28: 0.9, 29: 0.9, 11: 0.9, 10: 0.9})   # appears only in block 2
    ctrl = ctrl_of(K=5, M=3, n_block=20, u_cap=100, n_land=6, alpha_polish=0.5, alpha_global=0.25)
    rows = drive(ctrl, lambda j: mk(R=0.08, rho_bins=S_map) if j <= 20 else
                 mk(R=0.06, rho_bins=other), max_j=35)                      # block 2 still open
    b0, b1 = ctrl.blocks[0], ctrl.blocks[1]
    assert (b0["exit_reason"], b0["classification"], b0["last_local"]) == (
        "block_end", "localized", 20)
    assert b0["S"] == [13, 14, 22] and b0["n_S"] == 3 and "last_check" in b0
    assert (b1["type"], b1["first_local"], b1["S"]) == ("polish", 21, [13, 14, 22])
    rows2 = [r for r in rows if 21 <= r.j <= 35]
    assert len(rows2) == 15 and all(r.label == (2, "polish") for r in rows2)
    for r in rows2:
        s = r.setting
        # alpha_polish, not alpha_global
        assert s.block_type == "polish" and s.alpha == 0.5
        assert set(np.flatnonzero(s.focus)) <= {13, 14, 22}              # supported on S only
    # the focus values at the first polishing update are rho_bar restricted to S
    expect = np.zeros(NBINS)
    for b in (13, 14, 22):
        expect[b] = S_map[b]
    np.testing.assert_array_equal(rows2[0].setting.focus, expect)
    # later checks moved rho_bar elsewhere (bins 10, 11, 20, ...) but S and the support did not move
    assert ctrl.rho_bar[20] > 0.4 and b1["S"] == [13, 14, 22]
    assert set(np.flatnonzero(rows2[-1].setting.focus)) == {13, 14, 22}
    # at the end of block 2 the record's "S" becomes that block's own classification S (the
    # polishing set it ran on stays in the previous decision, blocks[0]["S"])
    assert ctrl.after_update(36, lambda: mk(R=0.06, rho_bins=other)).check is None
    for j in range(37, 41):
        ctrl.after_update(j, lambda: mk(R=0.06, rho_bins=other))
    assert ctrl.blocks[1]["exit_reason"] == "block_end" and ctrl.blocks[1]["last_local"] == 40
    assert ctrl.blocks[1]["S"] == [10, 11, 20, 21, 28, 29] and ctrl.blocks[0]["S"] == [13, 14, 22]
    assert ctrl.blocks[1]["classification"] == "broad" and ctrl.blocks[2]["type"] == "global"


def test_block_end_broad_goes_to_a_global_block_with_alpha_global_and_focus_rho_bar():
    broad = rb({10 + i: 0.06 for i in range(6)})                          # 6 > ceil(0.25 * 20) = 5
    ctrl = ctrl_of(K=5, M=3, n_block=20, u_cap=100, n_land=6, alpha_polish=0.5, alpha_global=0.5)
    rows = drive(ctrl, lambda j: mk(R=0.06, rho_bins=broad), max_j=35)
    b0, b1 = ctrl.blocks[0], ctrl.blocks[1]
    assert b0["classification"] == "broad" and b0["n_S"] == 6 and b0["S"] == list(range(10, 16))
    assert b1["type"] == "global" and b1["S"] == [] and b1["first_local"] == 21
    s = [r.setting for r in rows if r.j == 21][0]
    assert s.block_type == "global" and s.alpha == 0.5
    np.testing.assert_array_equal(s.focus, np.nan_to_num(broad, nan=0.0))   # all bins, tail zero


@pytest.mark.parametrize("name,diag,expect", [
    ("S qualifies but R <= rho (only delta fails)", mk(delta=0.5, R=0.02, rho_bins=rb({13: 0.06})),
     "broad"),
    ("S empty, R > rho", mk(R=0.06, rho_bins=rb()), "broad"),
    ("invalid check at the block end (R NaN)", mk(valid=False, R=float("nan"),
                                                     rho_bins=np.full(NBINS, np.nan)), "broad"),
    ("S of 5 bins and R > rho", mk(R=0.06, rho_bins=rb({10 + i: 0.06 for i in range(5)})),
     "localized"),
    ("S of 6 bins", mk(R=0.06, rho_bins=rb({10 + i: 0.06 for i in range(6)})), "broad"),
    ("one bin", mk(R=0.06, rho_bins=rb({29: 0.06})), "localized"),
])
def test_block_end_classification_cases(name, diag, expect):
    ctrl = ctrl_of(K=5, M=3, n_block=10, u_cap=100, n_land=4)
    drive(ctrl, lambda j: diag, max_j=12)
    assert ctrl.blocks[0]["classification"] == expect, name
    assert ctrl.blocks[1]["type"] == ("polish" if expect == "localized" else "global")


def test_classification_uses_the_ema_map_and_the_last_checks_R():
    """j = 5: bins 10-17 at 0.07; j = 10 (block end): all 0.02. EMA = 0.045 on 8 bins > rho."""
    maps = {5: rb({10 + i: 0.07 for i in range(8)}), 10: rb(default=0.02)}
    ctrl = ctrl_of(K=5, M=3, n_block=10, u_cap=100, n_land=4, ema_beta=0.5)
    drive(ctrl, lambda j: mk(R=0.05, rho_bins=maps[j]), max_j=10)
    assert ctrl.blocks[0]["S"] == list(range(10, 18))               # EMA, not the last raw map
    assert ctrl.blocks[0]["classification"] == "broad"
    np.testing.assert_allclose(ctrl.rho_bar[10:18], 0.045)
    # ema_beta enters: beta = 0 keeps only the latest map -> S empty (all 0.02), still broad
    ctrl0 = ctrl_of(K=5, M=3, n_block=10, u_cap=100, n_land=4, ema_beta=0.0)
    drive(ctrl0, lambda j: mk(R=0.05, rho_bins=maps[j]), max_j=10)
    assert ctrl0.blocks[0]["S"] == [] and ctrl0.blocks[0]["classification"] == "broad"


def test_an_invalid_check_does_not_update_the_residual_ema():
    ctrl = ctrl_of(K=5, M=3, n_block=1000, u_cap=2000, n_land=4)
    drive(ctrl, lambda j: mk(R=0.06, rho_bins=rb({13: 0.06})) if j == 5 else
          mk(valid=False, rho_bins=np.full(NBINS, np.nan)), max_j=15)
    np.testing.assert_array_equal(ctrl.rho_bar, rb({13: 0.06}))


# ----------------------------------------------------------------------------------------------
# classify_block_end boundary cases
# ----------------------------------------------------------------------------------------------

def _rho_bar(n_over: int, n: int = 20, over: float = 0.06, under: float = 0.01) -> np.ndarray:
    out = np.full(n, under)
    out[:n_over] = over
    return out


@pytest.mark.parametrize("n_nontail,cap", [(20, 5), (24, 6), (22, 6), (21, 6), (4, 1), (40, 10)])
def test_classify_cap_is_ceil_of_a_quarter(n_nontail, cap):
    assert cap == math.ceil(0.25 * n_nontail)
    cls, S = classify_block_end(0.1, _rho_bar(cap, n_nontail), 0.03, n_nontail, 0.25)
    assert cls == "localized" and S.size == cap                       # |S| = cap accepted
    cls, S = classify_block_end(0.1, _rho_bar(cap + 1, n_nontail), 0.03, n_nontail, 0.25)
    assert cls == "broad" and S.size == cap + 1                       # +1 refused


def test_classify_other_boundaries():
    rho_bar = _rho_bar(3)
    assert classify_block_end(0.1, rho_bar, 0.03, 20, 0.25)[0] == "localized"
    # R == rho refused
    assert classify_block_end(0.03, rho_bar, 0.03, 20, 0.25)[0] == "broad"
    just_above = float(np.nextafter(0.03, 1))
    assert classify_block_end(just_above, rho_bar, 0.03, 20, 0.25)[0] == "localized"
    # R <= rho refused
    assert classify_block_end(0.02, rho_bar, 0.03, 20, 0.25)[0] == "broad"
    cls, S = classify_block_end(0.1, np.full(20, 0.01), 0.03, 20, 0.25)
    assert cls == "broad" and S.size == 0                                          # empty S
    cls, S = classify_block_end(0.1, None, 0.03, 20, 0.25)
    assert cls == "broad" and S.size == 0
    # S uses a strict '>' and ignores NaN; entries equal to rho are not in S
    rho_bar = np.array([0.03, 0.0300001, np.nan, 0.5, 0.02] + [0.0] * 15)
    cls, S = classify_block_end(0.1, rho_bar, 0.03, 20, 0.25)
    assert list(S) == [1, 3] and cls == "localized"
    # R = NaN fails 'R > rho'
    assert classify_block_end(float("nan"), _rho_bar(2), 0.03, 20, 0.25)[0] == "broad"
    # a different localized fraction
    assert classify_block_end(0.1, _rho_bar(10), 0.03, 20, 0.5)[0] == "localized"
    assert classify_block_end(0.1, _rho_bar(11), 0.03, 20, 0.5)[0] == "broad"


# ----------------------------------------------------------------------------------------------
# cap path
# ----------------------------------------------------------------------------------------------

def test_cap_forces_the_landing_and_the_last_block_can_be_shorter():
    ctrl = ctrl_of(K=5, M=3, n_block=40, u_cap=100, n_land=10)
    rows = drive(ctrl, seq([BAD_BROAD]))
    rec = ctrl.record()
    spans = [(b["first_local"], b["last_local"]) for b in rec["blocks"]]
    assert spans == [(1, 40), (41, 80), (81, 100)]                       # 40, 40, 20
    assert [b["exit_reason"] for b in rec["blocks"]] == ["block_end", "block_end", "cap"]
    assert rec["budget_forced"] is True and rec["fire_local"] is None
    land = rec["landing"]
    assert (land["first_local"], land["last_local"]) == (101, 110) and land["budget_forced"] is True
    assert rec["training_updates"] == 100 and rec["total_updates"] == 110
    assert rows[-1].j == 110 and rows[99].step.events[-2:] == ["block_end:broad", "cap"]
    # the third block's classification is recorded too
    assert rec["blocks"][2]["classification"] == "broad"
    assert land["followed_block_id"] == 3 and land["followed_block_type"] == "global"


def test_cap_that_is_a_multiple_of_the_block_length_and_cap_shorter_than_a_block():
    ctrl = ctrl_of(K=5, M=3, n_block=40, u_cap=80, n_land=4)
    drive(ctrl, seq([BAD_BROAD]))
    assert [(b["first_local"], b["last_local"]) for b in ctrl.blocks] == [(1, 40), (41, 80)]
    assert ctrl.blocks[-1]["exit_reason"] == "cap" and ctrl.budget_forced
    ctrl = ctrl_of(K=5, M=3, n_block=400, u_cap=30, n_land=4)
    drive(ctrl, seq([BAD_BROAD]))
    assert [(b["first_local"], b["last_local"]) for b in ctrl.blocks] == [(1, 30)]
    assert ctrl.budget_forced and ctrl.landing["first_local"] == 31


def test_cap_after_polishing_blocks_lands_with_the_last_blocks_setting():
    ctrl = ctrl_of(K=5, M=3, n_block=40, u_cap=100, n_land=10)
    rows = drive(ctrl, seq([BAD_LOC]))
    assert [b["type"] for b in ctrl.blocks] == ["global", "polish", "polish"]
    assert [b["classification"] for b in ctrl.blocks] == ["localized"] * 3
    assert ctrl.landing["followed_block_type"] == "polish" and ctrl.budget_forced
    assert all(r.setting.followed_type == "polish" for r in rows if r.j > 100)


def test_stop_at_the_cap_check_is_a_development_stop_not_forced():
    ctrl = ctrl_of(K=5, M=3, n_block=40, u_cap=100, n_land=4)
    drive(ctrl, lambda j: GOOD if j in {90, 95, 100} else BAD_R)
    assert ctrl.fire_local == 100 and not ctrl.budget_forced
    assert ctrl.blocks[-1]["exit_reason"] == "development_stop"
    assert ctrl.blocks[-1]["last_local"] == 100
    assert ctrl.landing["budget_forced"] is False and ctrl.landing["first_local"] == 101


def test_forced_landing_follows_a_polishing_block_with_its_setting():
    maps = rb({13: 0.06, 14: 0.08})
    ctrl = ctrl_of(K=5, M=3, n_block=10, u_cap=20, n_land=6, alpha_polish=0.4)
    rows = drive(ctrl, lambda j: mk(R=0.08, rho_bins=maps))
    assert [b["type"] for b in ctrl.blocks] == ["global", "polish"]
    assert ctrl.blocks[1]["exit_reason"] == "cap" and ctrl.budget_forced
    land = [r for r in rows if r.j > 20]
    assert len(land) == 6
    for r in land:
        assert r.setting.followed_type == "polish" and r.setting.alpha == 0.4
        assert set(np.flatnonzero(r.setting.focus)) <= {13, 14}
    assert ctrl.landing["followed_block_type"] == "polish" and ctrl.landing["alpha"] == 0.4


# ----------------------------------------------------------------------------------------------
# checks: cadence, landing reports, t = 1
# ----------------------------------------------------------------------------------------------

def test_checks_every_K_updates_and_at_every_block_end_reported_in_landing():
    ctrl = ctrl_of(K=5, M=3, n_block=12, u_cap=36, n_land=10)
    calls: List[int] = []

    def diag_for(j: int) -> StageDiag:
        calls.append(j)
        # landing checks: eligible
        return BAD_BROAD if j <= 36 else GOOD
    rows = drive(ctrl, diag_for)
    assert calls == [5, 10, 12, 15, 20, 24, 25, 30, 35, 36, 40, 45]       # K cadence + block ends
    assert check_js(rows) == calls and ctrl.n_checks == 12
    modes = {r.j: r.step.check["mode"] for r in rows if r.step.check}
    assert {j: m for j, m in modes.items() if j > 36} == {40: "land", 45: "land"}
    assert all(m == "train" for j, m in modes.items() if j <= 36)
    labels = {r.j: (r.step.check["block_id"], r.step.check["block_type"])
              for r in rows if r.step.check}
    assert labels[5] == (1, "global") and labels[12] == (1, "global")
    assert labels[15] == (2, "global")
    assert labels[36] == (3, "global") and labels[40] == (4, "landing")
    # the eligible landing checks decided nothing
    assert ctrl.fire_local is None and ctrl.would_fire is None and ctrl.budget_forced
    assert ctrl.landing["last_local"] == 46 and rows[-1].j == 46


def test_t1_degenerate_rule_has_no_polishing_and_a_forced_landing():
    rule = StageRule(stage=1, n_nontail_bins=0, n_block=200, u_cap=600, n_land=400, K=25)
    ctrl = StageController(rule, lr_linear)
    d1 = mk(R=0.1, tail_term=False, R_tail=float("nan"), rho_bins=np.zeros(0), stage=1)
    rows = drive(ctrl, lambda j: d1)
    rec = ctrl.record()
    assert [(b["first_local"], b["last_local"]) for b in rec["blocks"]] == [(1, 200), (201, 400),
                                                                           (401, 600)]
    assert [b["classification"] for b in rec["blocks"]] == ["degenerate"] * 3
    assert [b["type"] for b in rec["blocks"]] == ["global"] * 3       # no polishing block ever
    assert [b["exit_reason"] for b in rec["blocks"]] == ["block_end", "block_end", "cap"]
    assert all(b["S"] == [] and b["n_S"] == 0 for b in rec["blocks"])
    assert rec["budget_forced"] is True and rec["landing"]["first_local"] == 601
    assert rec["landing"]["last_local"] == 1000 and rec["total_updates"] == 1000
    assert rec["n_checks"] == 24 + 16 and rows[-1].j == 1000
    assert all(r.setting.focus is None for r in rows)                  # no residual map at t = 1
    assert all(r.setting.block_type in ("global", "landing") for r in rows)


def test_t1_degenerate_rule_can_stop_on_M_eligible_checks():
    rule = StageRule(stage=1, n_nontail_bins=0, n_block=200, u_cap=600, n_land=400, K=25)
    ctrl = StageController(rule, lr_linear)
    good1 = mk(tail_term=False, R_tail=float("nan"), rho_bins=np.zeros(0), stage=1)
    drive(ctrl, lambda j: good1)
    assert ctrl.fire_local == 75 and not ctrl.budget_forced
    assert ctrl.landing["first_local"] == 76 and ctrl.landing["last_local"] == 475


# ----------------------------------------------------------------------------------------------
# legacy controller
# ----------------------------------------------------------------------------------------------

def v20_lr(j: int) -> float:
    """v2.0's phase-A LR: 3e-4 to local 1200, linear 3e-4 -> 3e-5 over 1201-1600 (Run.lr_for)."""
    if 1201 <= j <= 1600:
        sched = {"ab_lr": LR_BASE, "kind": "linear", "c_start_lr": LR_BASE, "c_end_lr": LR_END,
                 "c_local_first": 1201, "linear_denominator": 1600 - 1201}
        return lr_at(sched, "C", j)
    return LR_BASE


@pytest.mark.parametrize("eligible_from", [500, 10 ** 9])
def test_legacy_controller_runs_the_fixed_budget_and_only_records_would_fire(eligible_from):
    rule = rule_of(enabled=False, fixed_budget=1600, K=25, M=3)
    ctrl = StageController(rule, lr_must_not_be_called, v20_lr)
    rows = drive(ctrl, lambda j: GOOD if j >= eligible_from else BAD_R)
    assert rows[-1].j == 1600 and ctrl.finished
    # LR from the supplied function
    assert [r.lr for r in rows] == [v20_lr(j) for j in range(1, 1601)]
    assert all(r.setting.block_type == "legacy" and r.setting.alpha == 0.0
               and r.setting.focus is None for r in rows)
    assert all(r.label == (1, "legacy") for r in rows)
    assert check_js(rows) == list(range(25, 1601, 25)) and ctrl.n_checks == 64
    rec = ctrl.record()
    assert rec["enabled"] is False and rec["landing"] is None and rec["fire_local"] is None
    assert rec["budget_forced"] is False and rec["total_updates"] == 1600
    assert rec["training_updates"] is None
    assert [(b["type"], b["first_local"], b["last_local"], b["exit_reason"])
            for b in rec["blocks"]] == [
        ("legacy", 1, 1600, "fixed_budget")]
    assert rec["would_fire_local"] == (550 if eligible_from == 500 else None)
    assert rows[-1].step.events[-1] == "fixed_budget_end" and rows[-1].step.finished
    # nothing happens at would-fire
    assert rows[549].step.events == []
    assert rec["params"]["fixed_budget"] == 1600 and rec["params"]["enabled"] is False


def test_legacy_decisions_do_not_depend_on_the_checks():
    rule = rule_of(enabled=False, fixed_budget=300, K=25, M=3)
    a = drive(StageController(rule, lr_must_not_be_called, v20_lr), lambda j: GOOD)
    b = drive(StageController(rule, lr_must_not_be_called, v20_lr), lambda j: BAD_R)
    assert [(r.j, r.lr, r.label) for r in a] == [(r.j, r.lr, r.label) for r in b]
    assert [r.step.events for r in a] == [r.step.events for r in b]


# ----------------------------------------------------------------------------------------------
# constructor refusals and record()
# ----------------------------------------------------------------------------------------------

def test_constructor_refusals():
    with pytest.raises(ValueError, match="fixed budget"):
        StageController(rule_of(fixed_budget=1600), lr_linear)                  # enabled + budget
    with pytest.raises(ValueError, match="legacy"):
        StageController(rule_of(enabled=False), lr_linear, lambda j: LR_BASE)    # legacy, no budget
    with pytest.raises(ValueError, match="legacy"):
        StageController(rule_of(enabled=False, fixed_budget=0), lr_linear, lambda j: LR_BASE)
    with pytest.raises(ValueError, match="legacy"):
        # legacy, no LR function
        StageController(rule_of(enabled=False, fixed_budget=100), lr_linear)
    for n_land in (1, 0, -3):
        with pytest.raises(ValueError, match="n_land"):
            ctrl_of(n_land=n_land)
    for kw in ({"n_block": 0}, {"u_cap": 0}, {"K": 0}, {"M": 0}):
        with pytest.raises(ValueError):
            ctrl_of(**kw)
    ctrl_of(n_land=2)                                                              # smallest legal
    StageController(rule_of(enabled=False, fixed_budget=1, n_land=1), lr_linear, lambda j: LR_BASE)


def test_record_content_and_json():
    ctrl = ctrl_of(K=5, M=3, n_block=10, u_cap=20, n_land=4)
    maps = rb({13: 0.06})
    drive(ctrl, lambda j: mk(R=0.08, rho_bins=maps))
    rec = ctrl.record()
    assert set(rec) == {"stage", "enabled", "n_checks", "blocks", "landing", "fire_local",
                        "would_fire_local", "budget_forced", "training_updates", "total_updates",
                        "params"}
    assert rec["stage"] == 2 and rec["enabled"] is True
    assert set(rec["landing"]) == {"first_local", "last_local", "n_land", "followed_block_id",
                                   "followed_block_type", "alpha", "budget_forced", "done"}
    assert set(rec["blocks"][0]) == {"block_id", "type", "first_local", "last_local", "exit_reason",
                                     "fire_local", "S", "n_S", "classification", "last_check"}
    assert set(rec["params"]) == set(StageRule.__dataclass_fields__)
    assert rec["params"]["rho"] == 0.03 and rec["params"]["n_land"] == 4
    assert [b["block_id"] for b in rec["blocks"]] == [1, 2]
    assert rec["blocks"][0]["last_check"]["R"] == 0.08
    json.dumps(rec)                                                                # serialisable
    # StageRule defaults are the D4 values
    r = StageRule(stage=2)
    assert (r.K, r.M, r.eps, r.rho, r.tau, r.conc_limit) == (25, 3, 0.005, 0.03, 0.02, 0.04)
    assert (r.n_block, r.u_cap, r.n_land, r.loc_frac, r.alpha_polish, r.ema_beta) == (
        400, 2000, 400, 0.25, 0.5, 0.5)
    assert (r.lr_base, r.lr_end, r.alpha_global) == (3e-4, 3e-5, 0.0)
