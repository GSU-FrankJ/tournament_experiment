"""Tests of the R1 pre-registered analysis tool (tools/v2/refine_analysis.py).

Statistics: the percentile bootstrap against a hand-coded one (fixed seed 20261003, one fresh
generator per call), coverage sanity, the paired-difference / median / n_better logic, the SD
ratio with paired resampling, the per-arm criterion on synthetic tables (both parts, strictness,
failures under the arm, missing runs). Physics helpers: the conc_scale-aware actor loader, the
offline A_detmean objective on a tiny synthetic actor, the smoothed-game prediction against a
direct computation and against ``run.run_v2_T2_locked.smoothed_share``. End to end: tiny synthetic
run directories (including a missing run, a failed run, a run that is still running and a gate
violation) through the three sub-commands, reports and figures included. Nothing here trains.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "refine_analysis", ROOT / "tools" / "v2" / "refine_analysis.py")
R = importlib.util.module_from_spec(_spec)
sys.modules["refine_analysis"] = R
_spec.loader.exec_module(R)

from scipy.stats import beta as beta_dist  # noqa: E402

from agents.ppo_curriculum import PPOConfig, mean_effort_numpy  # noqa: E402
from agents.ppo_curriculum_v2 import CurriculumPPOv2  # noqa: E402
from run.run_final_dp_br import make_policy_fns  # noqa: E402
from run.run_v2_T2_locked import smoothed_share  # noqa: E402
from utils.theory_multistage import F_xi, f_xi, g1_two_stage, g2_two_stage  # noqa: E402

Q = 50
SEEDS = (10501, 10502, 10503)


# --------------------------------------------------------------------------- helpers
def make_agent(seed: int = 0, conc_scale: float = 1.0, out_scale: float = 0.8) -> CurriculumPPOv2:
    """Agent whose actor has a non-trivial output layer (the initial head is all zero)."""
    gen = torch.Generator().manual_seed(seed)
    agent = CurriculumPPOv2(PPOConfig(), gen, np.random.default_rng(seed))
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        agent.actor.out.weight.copy_(
            out_scale * torch.randn(agent.actor.out.weight.shape, generator=g))
        agent.actor.out.bias.copy_(0.3 * torch.randn(2, generator=g))
    agent.actor.conc_scale = conc_scale
    agent.refresh_snapshot()
    return agent


def ctx_for(tmp: Path) -> Any:
    return R.Ctx(root=str(tmp / "root"), ref_root=str(tmp / "ref"))


# --------------------------------------------------------------------------- bootstrap
def test_boot_ci_equals_hand_coded_percentile_bootstrap() -> None:
    x = np.array([0.3, -1.2, 0.7, 2.5, -0.4, 0.9, 1.1, -0.8, 0.05, 1.7])
    rng = np.random.default_rng(20261003)
    idx = rng.integers(0, x.size, size=(10000, x.size))
    exp_mean = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    exp_med = np.percentile(np.median(x[idx], axis=1), [2.5, 97.5])
    assert R.boot_ci(x, "mean") == (float(exp_mean[0]), float(exp_mean[1]))
    assert R.boot_ci(x, "median") == (float(exp_med[0]), float(exp_med[1]))


def test_boot_ci_reproducible_and_generator_is_fresh_per_call() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0, 3.0])
    a, b = R.boot_ci(x, "mean"), R.boot_ci(x, "mean")
    assert a == b
    # a call in between does not change the next one (no shared generator state)
    R.boot_ci(np.arange(7.0), "median")
    assert R.boot_ci(x, "mean") == a
    assert R.boot_ci(np.array([]), "mean")[0] != R.boot_ci(np.array([]), "mean")[0]  # nan


def test_boot_ci_coverage_sanity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "N_BOOT", 1000)
    mu, hits, reps = 0.3, 0, 300
    for i in range(reps):
        x = np.random.default_rng(1000 + i).normal(mu, 1.0, size=10)
        lo, hi = R.boot_ci(x, "mean")
        hits += int(lo <= mu <= hi)
    assert 0.85 < hits / reps < 0.99


def test_paired_summary_counts_and_direction() -> None:
    d = np.array([-1.0, -2.0, 0.0, 3.0, -0.5])
    s = R.paired_summary(d, True)
    assert s["n_pairs"] == 5 and s["n_better"] == 3
    assert (s["n_pos"], s["n_neg"], s["n_zero"]) == (1, 3, 1)
    assert s["mean"] == pytest.approx(-0.1) and s["median"] == -0.5
    assert s["ci_mean_lo"] <= s["mean"] <= s["ci_mean_hi"]
    nd = R.paired_summary(d, None)
    assert nd["n_better"] is None and nd["excl0_improving"] is None
    assert R.paired_summary(-np.ones(6), True)["excl0_improving"] is True
    assert R.paired_summary(np.array([]), True)["n_pairs"] == 0


def test_paired_tables_pair_by_seed_and_skip_incomplete() -> None:
    rows = []
    for arm, vals in (("B_base", [0.10, 0.20, 0.30, np.nan]), ("X", [0.08, 0.25, 0.10, 0.5])):
        for s, v in zip((1, 2, 3, 4), vals):
            rows.append({"arm": arm, "q": 50, "seed": s, "complete": s != 3 or arm == "B_base",
                         "stage1_rel_err_abs": v})
    df = pd.DataFrame(rows)
    p, long = R.paired_tables(df, "stage1", [("X", "B_base", "vs baseline")], [50], [1, 2, 3, 4])
    r = p[p.metric == "stage1_rel_err_abs"].iloc[0]
    # seed 3: X incomplete; seed 4: baseline value NaN -> only seeds 1 and 2 are paired
    assert r.n_pairs == 2 and r["mean"] == pytest.approx((-0.02 + 0.05) / 2)
    assert list(long["seed"]) == [1, 2] and r.n_better == 1
    assert set(p.metric) == {"stage1_rel_err_abs"}      # other metrics have no column


# --------------------------------------------------------------------------- dispersion
def test_sd_ratio_paired_resampling_and_degenerate_cases() -> None:
    arm = np.array([0.1, 0.5, -0.2, 0.9, 0.3, -0.6])
    base = np.array([0.2, 0.3, 0.1, 0.4, 0.0, 0.25])
    out = R.sd_ratio(arm, base)
    rng = np.random.default_rng(20261003)
    idx = rng.integers(0, 6, size=(10000, 6))
    ratio = arm[idx].std(axis=1, ddof=1) / base[idx].std(axis=1, ddof=1)
    lo, hi = np.percentile(ratio, [2.5, 97.5])
    assert out["ratio"] == pytest.approx(arm.std(ddof=1) / base.std(ddof=1))
    assert (out["ci_lo"], out["ci_hi"]) == (float(lo), float(hi))
    same = R.sd_ratio(base, base)
    assert same["ratio"] == pytest.approx(1.0) and same["ci_lo"] == pytest.approx(1.0)
    assert same["ci_hi"] == pytest.approx(1.0)
    const = R.sd_ratio(arm, np.full(6, 0.3))
    assert np.isnan(const["ratio"]) and const["n_nonfinite_resamples"] == 10000
    assert np.isnan(R.sd_ratio(np.array([1.0]), np.array([2.0]))["ratio"])


# --------------------------------------------------------------------------- criterion
def _prim(rows: Dict[int, Dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"q": q, **v} for q, v in rows.items()])


def _gates(entries: List[tuple]) -> pd.DataFrame:
    return pd.DataFrame([{"q": q, "seed": s, "base_pass": bp, "arm_status": st, "arm_pass": ap}
                         for q, s, bp, st, ap in entries])


GOOD = {50: {"n_pairs": 3, "mean": -0.02, "ci_mean_lo": -0.03, "ci_mean_hi": -0.01},
        60: {"n_pairs": 3, "mean": -0.03, "ci_mean_lo": -0.05, "ci_mean_hi": -0.001}}
ALL_OK = [(q, s, True, "done", True) for q in (50, 60) for s in (1, 2, 3)]


def test_criterion_both_parts_met() -> None:
    c = R.criterion(_prim(GOOD), _gates(ALL_OK), (50, 60), 3)
    assert c["a_met"] and c["a_complete"] and c["b_status"] == "holds" and c["overall"] == "met"
    assert c["b_violations"] == "" and c["n_base_pass"] == 6


def test_criterion_strict_inequality_and_both_q_required() -> None:
    edge = {**GOOD, 60: {**GOOD[60], "ci_mean_hi": 0.0}}          # CI touches 0: not excluded
    c = R.criterion(_prim(edge), _gates(ALL_OK), (50, 60), 3)
    assert c["a_q50"] and not c["a_q60"] and not c["a_met"] and c["overall"] == "not met"
    one = {**GOOD, 50: {**GOOD[50], "ci_mean_hi": 0.004}}
    assert not R.criterion(_prim(one), _gates(ALL_OK), (50, 60), 3)["a_met"]
    worse = {**GOOD, 50: {**GOOD[50], "mean": 0.02, "ci_mean_lo": 0.01, "ci_mean_hi": 0.03}}
    assert not R.criterion(_prim(worse), _gates(ALL_OK), (50, 60), 3)["a_q50"]


def test_criterion_violation_failure_and_missing_runs() -> None:
    viol = list(ALL_OK)
    viol[1] = (50, 2, True, "done", False)                          # passed baseline, fails arm
    c = R.criterion(_prim(GOOD), _gates(viol), (50, 60), 3)
    assert c["a_met"] and c["b_status"] == "violated" and c["b_violations"] == "q50/2"
    assert c["overall"] == "not met"                                # (a) alone does not suffice
    ok_fail = list(ALL_OK)
    ok_fail[1] = (50, 2, False, "done", False)                      # failed baseline too: no issue
    assert R.criterion(_prim(GOOD), _gates(ok_fail), (50, 60), 3)["b_status"] == "holds"
    crashed = list(ALL_OK)
    crashed[2] = (50, 3, True, "failed", np.nan)                    # arm run crashed
    cc = R.criterion(_prim(GOOD), _gates(crashed), (50, 60), 3)
    assert cc["b_status"] == "violated" and cc["b_violations"] == "q50/3"
    nan_gate = list(ALL_OK)
    nan_gate[0] = (50, 1, True, "done", np.nan)                     # gate not computable
    assert R.criterion(_prim(GOOD), _gates(nan_gate), (50, 60), 3)["b_status"] == "violated"
    pending = list(ALL_OK)
    pending[3] = (60, 1, True, "missing", np.nan)
    part = {**GOOD, 60: {**GOOD[60], "n_pairs": 2}}
    cp = R.criterion(_prim(part), _gates(pending), (50, 60), 3)
    assert cp["b_status"] == "incomplete" and cp["b_pending"] == "q60/1"
    assert not cp["a_complete"] and cp["overall"] == "incomplete"
    none = R.criterion(_prim({50: {"n_pairs": 0, "mean": np.nan, "ci_mean_lo": np.nan,
                                   "ci_mean_hi": np.nan}}), _gates(ALL_OK[:3]), (50,), 3)
    assert not none["a_met"] and none["overall"] != "met"


# --------------------------------------------------------------------------- actor loader
def test_load_actor_applies_conc_scale_of_export_and_state(tmp_path: Path) -> None:
    spec = R.GameSpec(w_h=6, w_l=2, k=1 / 3500, q=50, T=2)
    agent = make_agent(3, conc_scale=2.0)
    npz, pt = tmp_path / "u01600.npz", tmp_path / "state_end_A.pt"
    agent.export_weights_npz(str(npz))
    torch.save({"agent": agent.full_state()}, pt)
    d = np.array([-120.0, -10.0, 0.0, 33.0, 180.0])
    _, own_beta = make_policy_fns(agent, spec)
    a0, b0 = own_beta(2, d)                                        # the run's own beta_fn
    for src in (npz, pt):
        actor = R.load_actor(src)
        assert actor.conc_scale == 2.0
        a, b = R.actor_alpha_beta(actor, spec, 2, d)
        np.testing.assert_allclose(a, a0, rtol=1e-6)
        np.testing.assert_allclose(b, b0, rtol=1e-6)
    # a loader that ignores the factor (as pilot2_analysis._actor_fns) is off by the factor
    ign = R.load_actor(npz, conc_scale=1.0)
    ai, bi = R.actor_alpha_beta(ign, spec, 2, d)
    np.testing.assert_allclose((a0 + b0) / (ai + bi), 2.0, rtol=1e-6)
    # the framework-free reload reads the exported factor as well
    with np.load(npz) as z:
        w = {k: z[k] for k in z.files}
    obs = spec.encode_obs(2, d)
    _, a_np, b_np = mean_effort_numpy(w, obs)
    np.testing.assert_allclose(a_np, a0, rtol=1e-5)
    # an unscaled export carries no factor and reloads to 1.0
    un = make_agent(3)
    un.export_weights_npz(str(tmp_path / "u.npz"))
    assert R.load_actor(tmp_path / "u.npz").conc_scale == 1.0


# --------------------------------------------------------------------------- offline objective
def test_offline_objective_matches_independent_numpy(tmp_path: Path) -> None:
    spec = R.GameSpec(w_h=6, w_l=2, k=1 / 3500, q=50, T=2)
    agent = make_agent(5)
    cen = R.bin_centres(spec, 10.0)
    assert cen.size == 40 and cen[0] == -195.0 and cen[-1] == 195.0
    out = R.offline_objective(agent.actor, spec, cen)
    npz = tmp_path / "w.npz"
    agent.export_weights_npz(str(npz))
    with np.load(npz) as z:
        w = {k: z[k] for k in z.files}
    e, _, _ = mean_effort_numpy(w, spec.encode_obs(2, cen))
    eo, _, _ = mean_effort_numpy(w, spec.encode_obs(2, -cen))
    r = spec.w_l + spec.dw * F_xi(cen + e - eo, spec.q) - spec.k * e * e
    foc = np.abs(spec.dw * f_xi(cen + e - eo, spec.q) - 2.0 * spec.k * e)
    assert out["J"] == pytest.approx(float(r.mean()), abs=1e-6)
    assert out["foc_mean"] == pytest.approx(float(foc.mean()), abs=1e-7)
    assert out["foc_max"] == pytest.approx(float(foc.max()), abs=1e-7)


# --------------------------------------------------------------------------- smoothed game
def test_smoothed_prediction_against_direct_computation_and_locked_method() -> None:
    spec = R.GameSpec(w_h=6, w_l=2, k=1 / 3500, q=50, T=2)
    a, b = 140.0, 160.0
    n = 25
    x = np.array([spec.e_range * (beta_dist.ppf((i + 0.5) / n, a, b) - a / (a + b))
                  for i in range(n)])
    direct = sum(float(f_xi(xi - xj, spec.q)) for xi in x for xj in x) / n ** 2
    direct *= spec.dw / (2 * spec.k)
    assert R.smoothed_pred_d0(a, b, spec, n_nodes=n) == pytest.approx(direct, rel=1e-12)
    # the locked method (run_v2_T2_locked.smoothed_share) on an agent without a Run object
    agent = make_agent(7, conc_scale=1.5)
    run = SimpleNamespace(spec=spec, agent=agent)
    locked = smoothed_share(run, None)
    al, be = R.actor_alpha_beta(agent.actor, spec, 2, np.zeros(1))
    mine = R.smoothed_record(float(al[0]), float(be[0]), spec, locked["smoothed_e_learned_0"])
    assert mine["smoothed_e_pred_0"] == pytest.approx(locked["smoothed_e_pred_0"], rel=1e-12)
    assert mine["smoothed_share_peak_gap_d0"] == pytest.approx(
        locked["smoothed_share_peak_gap_d0"], rel=1e-9)
    g20 = float(g2_two_stage(np.zeros(1), spec.q, spec.w_h, spec.w_l, spec.k, spec.e_max)[0])
    assert mine["smoothed_pred_gap_d0"] == pytest.approx(g20 - locked["smoothed_e_pred_0"])


# --------------------------------------------------------------------------- synthetic runs
def _write(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def build_reference(tmp: Path, ctx: Any) -> None:
    """Rehearsal parents: gates.json, induced_band.json, state_end_A.pt per (q=50, seed)."""
    spec = R.spec_for_q(ctx, Q)
    g1 = float(g1_two_stage(Q, spec.w_h, spec.w_l, spec.k))
    g20 = R.g2_peak(spec)
    for i, seed in enumerate(SEEDS):
        d = tmp / "ref" / f"q{Q}" / f"seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"agent": make_agent(seed).full_state()}, d / "state_end_A.pt")
        fin = {"stage2_peak_rel_err_signed": -0.1 + 0.01 * i,
               "stage2_peak_rel_err_abs": 0.1 - 0.01 * i,
               "stage2_rmse_pos_over_g2_0": 0.04, "stage2_tail_mean_over_g2_0": 0.007,
               "stage2_tail_max_over_g2_0": 0.06, "stage2_tail_mean": 0.5, "stage2_tail_max": 4.0,
               "stage2_sym_err_max": 3.0, "eta_T_over_dw": 0.003, "sigma_effort_at_0_t2": 3.4,
               "g2_at_0": g20, "e2_at_0": g20 * 0.9, "stage2_peak_locfree_rel_err": -0.09,
               "stage2_peak_locfree_argmax_d": -0.5}
        _write(d / "gates.json", {
            "metric_values": {"eta_final": 0.003, "eta_dev": 0.003, "rmse": 0.04, "tail": 0.007,
                              "gmax_final": 0.004, "gmax_dev": 0.004, "s1": 0.01},
            "outcome": "pass", "G-A": {"pass": True},
            "G-N": {"criteria": [{"pass": True}, {"pass": True}]},
            "reported": {"end_of_A": {"final": fin, "development": {**fin},
                                      "smoothed_game": {"smoothed_e_pred_0": g20 * 0.95,
                                                        "smoothed_share_peak_gap_d0": 0.3}}}})
        _write(d / "induced_band.json", {"e_tilde": g1 + 0.3, "band_lo": g1 + 0.3,
                                         "band_hi": g1 + 0.5, "sweep_lo": 18.0, "sweep_hi": 80.0,
                                         "g1": g1})


def _history(phase: str, n: int, first: int) -> Dict[str, Any]:
    return {"history": [{"update": first + i, "phase": phase, "grad_norm_actor_mean": 1.0 + i,
                         "grad_norm_actor_max": 2.0 + i, "n_minibatch_steps": 40,
                         "n_actor_steps": 40} for i in range(n)],
            "curriculum": [{"phase": phase, "episodes": 512 * n, "transitions": 1024 * n,
                            "local_updates": n}]}


def _updates(first: int, n: int, diverge_at: Optional[int], kl_arm: bool = False) -> pd.DataFrame:
    rows = []
    for i in range(n):
        u = first + i
        late = diverge_at is not None and u >= diverge_at
        rows.append({"update": u, "adv_s1_std": 1.0, "adv_all_std": 0.9, "kl_final_epoch": 0.004,
                     "clip_frac": 0.05, "n_epochs_run": (3 if i % 2 else 10) if kl_arm else 10,
                     "rngpos_env": f"e:0:{u}", "rngpos_learn": f"l:0:{u}{'x' if late else ''}",
                     "rngpos_opp": f"o:0:{u}{'y' if late else ''}", "rngpos_start": f"s:0:{u}",
                     "rngpos_minibatch": f"m:0:{u}"})
    return pd.DataFrame(rows)


def write_common(d: Path, phase: str, parent: str, wall: float, status: str = "done",
                 scale: float = 1.0) -> None:
    d.mkdir(parents=True, exist_ok=True)
    if status == "failed":
        _write(d / "status.json", {"state": "failed", "exit_code": 1,
                                   "traceback": "Traceback...\nRuntimeError: boom"})
        return
    if status == "running":
        _write(d / "status.json", {"state": "running"})
        return
    _write(d / "status.json", {"state": "done", "exit_code": 0, "total_wall_sec": wall + 1.0})
    _write(d / "manifest.json", {"parent_checkpoint": parent,
                                 "git": {"short": "abc1234", "dirty": False}})
    _write(d / "v2_run_summary.json", {"phase_timing": {phase: {"wall_sec": wall}},
                                       "costs": {"train_update_sec": wall * 0.9},
                                       "conc_scale_final": {"actor": scale, "opponent": scale}})


STAGE1_ERR = {"B_base": (0.02, -0.03, 0.04), "B_polish1": (0.01, -0.015, 0.02),
              "B_kl005": (0.018, -0.027, 0.036), "B_expcont": (0.006, -0.009, 0.012)}


def build_stage1(tmp: Path, ctx: Any) -> None:
    # an original moved aside by a re-run: counted in the completeness table, never analysed
    aside = tmp / "root" / "stage1" / "dirty_rerun" / "q50" / "seed10501" / "B_base"
    write_common(aside, "B", "x", 999.0)
    _write(aside / "final_v2.json", {"final": {"stage1_rel_err_abs": 9.0, "valid": True}})
    _write(tmp / "root" / "stage1" / "dirty_rerun" / "moved.json", ["q50/seed10501/B_base"])
    _write(tmp / "root" / "stage1_dirty_rerun_comparison.json",
           {"q50/seed10501/B_base": {"all_identical": True}})
    _write(tmp / "root" / "stage1" / "launch_20260101_000000.json",
           {"state": "done", "n_planned": 1, "runs": [{"returncode": 0}, {"returncode": 1}],
            "workers": 2, "head": "abcdef0123", "code_commit": "0123456789", "nproc": 64,
            "loadavg_at_start": [1.0, 2.0, 3.0]})
    spec = R.spec_for_q(ctx, Q)
    g1 = float(g1_two_stage(Q, spec.w_h, spec.w_l, spec.k))
    for i, seed in enumerate(SEEDS):
        parent_path = tmp / "ref" / f"q{Q}" / f"seed{seed}" / "state_end_A.pt"
        par_sd = torch.load(parent_path, weights_only=False)["agent"]["actor"]
        for arm, errs in STAGE1_ERR.items():
            d = ctx.run_dir("stage1", Q, seed, arm)
            if arm == "B_polish1" and seed == 10502:
                write_common(d, "B", str(parent_path), 50.0, status="failed")
                continue
            if arm == "B_kl005" and seed == 10503:
                continue                                         # never launched: no directory
            wall = {"B_base": 100.0, "B_polish1": 110.0, "B_kl005": 90.0, "B_expcont": 120.0}[arm]
            write_common(d, "B", str(parent_path), wall)
            err = errs[i]
            gmax = 0.02 if (arm == "B_kl005" and seed == 10501) else 0.004   # a G-F violation
            sc = {"e1_at_0": g1 * (1 + err), "g1": g1, "stage1_rel_err_signed": err,
                  "stage1_rel_err_abs": abs(err), "sigma_effort_at_0_t1": 3.0,
                  "Gmax_full_over_dw": gmax, "Gmax_full_t": 2, "Gmax_full_d": -4.0,
                  "EXP_root_over_dw": 0.001, "dReach_over_dw": 0.004,
                  "Deltamax_all_over_dw": gmax, "dFull_over_dw": 0.004, "eta_T_over_dw": 0.003,
                  "valid": True}
            _write(d / "final_v2.json", {"stage1_status": "trained", "final": sc,
                                         "development": {**sc, "Gmax_full_over_dw": gmax + 1e-4}})
            _write(d / "drift_test.json", {"pass": True})
            frozen = {k: v.clone() for k, v in par_sd.items()}
            if arm == "B_kl005" and seed == 10502:                 # shared stage 2 broken
                frozen["l1.bias"] = frozen["l1.bias"] + 1e-3
            torch.save({"agent": {"frozen": frozen}}, d / "state_end_B.pt")
            (d / "weights").mkdir(exist_ok=True)
            for u in R.LAST5:
                make_agent(1000 * seed + u).export_weights_npz(
                    str(d / "weights" / f"u{u:05d}.npz"))
            _write(d / "train_history.json", _history("B", 6, 1601))
            kl = arm == "B_kl005"
            _updates(1601, 6, None if arm == "B_base" else (1601 if arm == "B_expcont" else 1603),
                     kl_arm=kl).to_csv(d / "v2_updates.csv", index=False)


S2_ARMS = ("A_base", "A_anneal2", "A_ctrl200", "A_detmean")


def build_stage2(tmp: Path, ctx: Any) -> None:
    spec = R.spec_for_q(ctx, Q)
    g20 = R.g2_peak(spec)
    grid = np.linspace(-200.0, 200.0, 201)
    rgrid = np.linspace(-200.0, 200.0, 801)
    for i, seed in enumerate(SEEDS):
        parent_path = tmp / "ref" / f"q{Q}" / f"seed{seed}" / "state_end_A.pt"
        for j, arm in enumerate(S2_ARMS):
            d = ctx.run_dir("stage2", Q, seed, arm)
            phase = "P" if arm == "A_detmean" else "A"
            first = 1601 if arm in ("A_ctrl200", "A_detmean") else 1201
            if arm == "A_anneal2" and seed == 10503:
                write_common(d, "A", str(parent_path), 40.0, status="running")
                continue
            scale = 2.0 if arm == "A_anneal2" else 1.0
            wall = {"A_base": 50.0, "A_anneal2": 45.0, "A_ctrl200": 25.0, "A_detmean": 1.0}[arm]
            write_common(d, phase, str(parent_path), wall, scale=scale)
            agent = make_agent(100 * seed + j, conc_scale=scale, out_scale=0.6 + 0.1 * j)
            torch.save({"agent": agent.full_state()},
                       d / ("state_end_P.pt" if phase == "P" else "state_end_A.pt"))
            mean_fn, beta_fn = make_policy_fns(agent, spec)
            al, be = beta_fn(2, grid)
            e2r = mean_fn(2, rgrid)
            g2r = g2_two_stage(rgrid, spec.q, spec.w_h, spec.w_l, spec.k, spec.e_max)
            np.savez(d / "final_final.npz", v_t2_d_grid=grid, v_t2_alpha=al, v_t2_beta=be,
                     recovery_d_grid=rgrid, recovery_e2=e2r, recovery_g2=g2r)
            a0, b0 = beta_fn(2, np.zeros(1))
            s = a0[0] + b0[0]
            sigma = spec.e_range * float(np.sqrt(a0[0] * b0[0] / (s * s * (s + 1.0))))
            e20 = float(mean_fn(2, np.zeros(1))[0])
            sc = {"stage2_peak_rel_err_signed": (e20 - g20) / g20,
                  "stage2_peak_rel_err_abs": abs(e20 - g20) / g20,
                  "stage2_rmse_pos_over_g2_0": 0.04, "stage2_tail_mean_over_g2_0": 0.007,
                  "stage2_tail_max_over_g2_0": 0.06, "stage2_tail_mean": 0.5,
                  "stage2_tail_max": 4.0, "stage2_sym_err_max": 3.0,
                  "eta_T_over_dw": 0.003, "sigma_effort_at_0_t2": sigma, "g2_at_0": g20,
                  "e2_at_0": e20, "valid": True}
            _write(d / "final_v2.json", {"stage1_status": "stage1_untrained", "final": sc,
                                         "development": dict(sc)})
            (d / "weights").mkdir(exist_ok=True)
            for u in (first + 99, first + 199):
                agent.export_weights_npz(str(d / "weights" / f"u{u:05d}.npz"))
            if phase == "P":
                rows = [{"update": first + k, "local": k + 1, "actor_lr": 3e-5,
                         "loss": -3.7 + 0.05 * np.sin(k), "grad_norm_pre_clip": 0.1,
                         "foc_abs_mean": 7e-4, "foc_abs_max": 4e-3, "e0": 61.0,
                         "rngpos_env": "e:0:0", "rngpos_learn": "l:0:0", "rngpos_opp": "o:0:0",
                         "rngpos_start": f"s:0:{k}", "rngpos_minibatch": "m:0:0"}
                        for k in range(25)]
                pd.DataFrame(rows).to_csv(d / "v2_updates.csv", index=False)
                pd.DataFrame([{"update": first + 24, "local": 25, "reason": "timeout"}]).to_csv(
                    d / "v2_checkpoints_P.csv", index=False)
                _write(d / "phaseP_checks.json", {"concentration_head_bit_identical": True,
                                                  "critic_bit_identical": True,
                                                  "critic_adam_state_bit_identical": True})
            else:
                diverge = None if arm in ("A_base", "A_ctrl200") else first + 2
                _updates(first, 25, diverge).to_csv(d / "v2_updates.csv", index=False)
            _write(d / "train_history.json", _history(phase, 25, first))


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """Run stage1, stage2 and decision on the synthetic runs once; share the outputs."""
    tmp = tmp_path_factory.mktemp("refine")
    ctx = ctx_for(tmp)
    build_reference(tmp, ctx)
    build_stage1(tmp, ctx)
    build_stage2(tmp, ctx)
    out, rep = tmp / "out", tmp / "reports"
    common = ["--root", str(tmp / "root"), "--ref-root", str(tmp / "ref"), "--qs", str(Q),
              "--seeds", *map(str, SEEDS), "--workers", "1", "--out", str(out),
              "--reports", str(rep)]
    # the D1 / D2 / check-(ii) inputs of the decision report (tiny files with the real columns)
    (tmp / "root" / "d1_clamp").mkdir(parents=True)
    d1 = [{"group": g, "q": q, "phase": ph, "n_runs": 20, "M1_median_over_runs": m1,
           "M1_threshold": 0.001, "M1_outcome": "exceeded" if m1 > 0.001 else "not exceeded",
           "M2_runs_share_above_threshold": 3, "M2_runs_needed_more_than": 2,
           "M2_max_share": 0.02, "M2_outcome": "exceeded"}
          for g, q, ph, m1 in (("v11_reproduction", "50", "A", 0.002),
                               ("v11_reproduction", "all", "A", 0.002),
                               ("v11_reproduction", "all", "B", 0.0),
                               ("stage1/B_expcont", "all", "B", 0.0),
                               ("stage2/A_anneal2", "all", "A", 0.0007))]
    pd.DataFrame(d1).to_csv(tmp / "root" / "d1_clamp" / "d1_flags.csv", index=False)
    (tmp / "root" / "d2_verifier_sensitivity").mkdir(parents=True)
    pd.DataFrame([{"family": "a", "q": Q, "tier": "final", "criterion": "G-F",
                   "perturbation_name": "delta_stage1", "side": "any",
                   "limit_str": "not reached on the grid", "max_abs_pert_on_grid": 0.15,
                   "max_value_on_grid": 0.007}]).to_csv(
        tmp / "root" / "d2_verifier_sensitivity" / "detection_limits.csv", index=False)
    _write(tmp / "root" / "continuation_check.json", {
        "table_vs_verifier": {"final": {"q50": {"max_abs_diff_over_dw": 3e-5,
                                                "spec_tolerance_over_dw": 1e-6,
                                                "meets_spec_tolerance": False}},
                              "development": {"q50": {"max_abs_diff_over_dw": 1e-4}}},
        "diagnosis": {"verifier_grid_sweep_max_abs_diff_over_dw": {
            "q50": {"state_step_0.25": {"gl_half_64": 5e-7}}}}})
    R.main(["stage1", "--arms", *STAGE1_ERR, *common, "--figures", str(tmp / "figs")])
    R.main(["stage2", "--arms", *S2_ARMS, *common, "--figures", str(tmp / "figs")])
    R.main(["decision", *common, "--figures", str(tmp / "figs")])
    return {"tmp": tmp, "out": out, "rep": rep, "ctx": ctx, "common": common}


def test_end_to_end_completeness_lists_failed_missing_and_running(
        synthetic: Dict[str, Any]) -> None:
    c = pd.read_csv(synthetic["out"] / "stage1_completeness.csv",
                    dtype={"failed_seeds": str, "not_done_seeds": str}).set_index("arm")
    assert (c.loc["B_polish1", "done"], c.loc["B_polish1", "failed"]) == (2, 1)
    assert c.loc["B_polish1", "failed_seeds"] == "10502"
    assert (c.loc["B_kl005", "done"], c.loc["B_kl005", "missing"]) == (2, 1)
    assert c.loc["B_base", "done"] == 3 and c.loc["B_expcont", "done"] == 3
    assert c.loc["B_base", "set_aside_originals"] == 1          # counted per arm and q
    assert c["set_aside_originals"].sum() == 1 and "set_aside_originals_in_wave" not in c
    pr0 = pd.read_csv(synthetic["out"] / "stage1_per_run.csv")
    assert not (pr0.stage1_rel_err_abs == 9.0).any()
    lr = pd.read_csv(synthetic["out"] / "stage1_launch_record.csv")
    assert lr.iloc[0].n_returncode_0 == 1 and lr.iloc[0].n_returncode_nonzero == 1
    c2 = pd.read_csv(synthetic["out"] / "stage2_completeness.csv").set_index("arm")
    assert (c2.loc["A_anneal2", "done"], c2.loc["A_anneal2", "incomplete"]) == (2, 1)
    pr = pd.read_csv(synthetic["out"] / "stage1_per_run.csv")
    assert len(pr) == len(STAGE1_ERR) * len(SEEDS)              # every planned run has a row
    row = pr[(pr.arm == "B_kl005") & (pr.seed == 10503)].iloc[0]
    assert row.status == "missing" and not row.complete
    fl = pr[(pr.arm == "B_polish1") & (pr.seed == 10502)].iloc[0]
    assert fl.status == "failed" and "boom" in fl.status_info


def test_end_to_end_stage1_values_verdicts_and_identity(synthetic: Dict[str, Any]) -> None:
    pr = pd.read_csv(synthetic["out"] / "stage1_per_run.csv")
    b = pr[(pr.arm == "B_base") & (pr.seed == 10501)].iloc[0]
    assert b.stage1_rel_err_abs == pytest.approx(0.02) and bool(b.run_pass) and b.outcome == "pass"
    assert bool(b.G_F) and bool(b.G_A_parent) and bool(b.S1_pass) and bool(b.target0929_pass)
    assert b.learning_rel == pytest.approx(0.02 - 0.3 / b.g1)
    v = pr[(pr.arm == "B_kl005") & (pr.seed == 10501)].iloc[0]
    assert not bool(v.G_F) and not bool(v.run_pass) and v.outcome == "fail_G-F"
    # the shared stage 2 is verified per run; a perturbed frozen snapshot is flagged
    broken = pr[(pr.arm == "B_kl005") & (pr.seed == 10502)].iloc[0]
    assert not bool(broken.frozen_bit_identical_to_parent)
    assert "e~1 not shared" in broken.anomalies
    ok = pr[(pr.arm == "B_expcont") & pr.complete]
    assert ok.frozen_bit_identical_to_parent.astype(bool).all()
    assert ok.within_run_sd_e1_last5.notna().all() and (ok.e1_last5_n == 5).all()
    # rng divergence against the baseline run: B_expcont diverges at the first update
    e = pr[(pr.arm == "B_expcont") & (pr.seed == 10501)].iloc[0]
    assert e.rng_div_learn == 1601 and e.rng_div_env == "never"
    k = pr[(pr.arm == "B_kl005") & (pr.seed == 10501)].iloc[0]
    assert k.rng_div_learn == 1603 and k.n_epochs_cnt_3 == 3 and k.n_epochs_cnt_10 == 3


def test_end_to_end_paired_tables_and_criterion(synthetic: Dict[str, Any]) -> None:
    p = pd.read_csv(synthetic["out"] / "stage1_paired.csv")
    r = p[(p.arm == "B_expcont") & (p.metric == "stage1_rel_err_abs")].iloc[0]
    diffs = np.array([0.006 - 0.02, 0.009 - 0.03, 0.012 - 0.04])
    assert r.n_pairs == 3 and r["mean"] == pytest.approx(diffs.mean())
    assert r["median"] == pytest.approx(np.median(diffs)) and r.n_better == 3
    lo, hi = R.boot_ci(diffs, "mean")
    assert (r.ci_mean_lo, r.ci_mean_hi) == pytest.approx((lo, hi))
    # learning_rel differences equal signed-error differences (shared e~1)
    s = p[(p.arm == "B_expcont") & (p.metric == "stage1_rel_err_signed")].iloc[0]
    l = p[(p.arm == "B_expcont") & (p.metric == "learning_rel")].iloc[0]
    assert s["mean"] == pytest.approx(l["mean"])
    c = pd.read_csv(synthetic["out"] / "stage1_criterion.csv").set_index("arm")
    assert c.loc["B_expcont", "overall"] == "met" and bool(c.loc["B_expcont", "a_met"])
    assert c.loc["B_expcont", "b_status"] == "holds"
    assert c.loc["B_kl005", "b_status"] == "violated" and "q50/10501" in c.loc["B_kl005",
                                                                              "b_violations"]
    assert c.loc["B_polish1", "b_status"] == "violated"        # the failed run passed in the base
    assert "q50/10502" in c.loc["B_polish1", "b_violations"]
    d = pd.read_csv(synthetic["out"] / "stage1_dispersion.csv")
    assert set(d.metric) == {"stage1_rel_err_signed", "e1_at_0"}
    adv = pd.read_csv(synthetic["out"] / "stage1_adv_ratio.csv")
    assert (adv.mean_ratio == 1.0).all()


def test_end_to_end_stage2_annealing_reload_and_method5(synthetic: Dict[str, Any]) -> None:
    pr = pd.read_csv(synthetic["out"] / "stage2_per_run.csv")
    an = pr[(pr.arm == "A_anneal2") & pr.complete]
    assert (an.conc_scale_actor == 2.0).all()
    assert (an.sigma2_0_reload_rel_diff < 1e-6).all() and (an.alpha_reload_rel_diff < 1e-6).all()
    assert an.anomalies.fillna("").eq("").all()
    ref = pr[pr.arm == "parent_u1600"]
    assert len(ref) == 3 and ref.complete.all()
    dm = pr[(pr.arm == "A_detmean") & pr.complete]
    assert dm.J_offline_end.notna().all() and dm.foc_offline_end_max.notna().all()
    assert (dm.verifier_last_reason == "timeout").all() and dm.loss_last20_mean.notna().all()
    assert dm.n_epochs_run_mean.isna().all()                    # no such column by construction
    assert dm.anomalies.fillna("").eq("").all()
    tr = pd.read_csv(synthetic["out"] / "stage2_detmean_trajectory_per_run.csv")
    assert set(tr.arm) == {"A_detmean", "A_ctrl200"}
    assert (tr[tr["local"] == 0]["update"] == 1600).all()
    p = pd.read_csv(synthetic["out"] / "stage2_paired.csv")
    assert set(p.comparison) >= {"vs baseline", "ablation vs matched control",
                                 "vs parent u1600 candidate"}
    ab = p[(p.comparison == "ablation vs matched control")
           & (p.metric == "stage2_peak_rel_err_abs")]
    assert len(ab) == 1 and ab.iloc[0].arm == "A_detmean" and ab.iloc[0].baseline == "A_ctrl200"
    c = pd.read_csv(synthetic["out"] / "stage2_criterion.csv").set_index("arm")
    assert "A_anneal2" in c.index and c.loc["A_anneal2", "n_pairs_q50"] == 2
    ann = pd.read_csv(synthetic["out"] / "stage2_annealing.csv")
    assert set(ann.quantity) >= {"smoothed_pred_gap_d0", "observed_gap_d0", "e2_at_0"}
    assert (ann.n_pairs == 2).all()                              # seed 10503 still running
    assert "corr_dpred_dobs_over_seeds" not in ann.columns
    cc = pd.read_csv(synthetic["out"] / "stage2_annealing_corr.csv")
    assert len(cc) == 1 and cc.iloc[0].arm == "A_anneal2" and cc.iloc[0].n_pairs == 2
    # (h) the ablation has its own criterion and dispersion rows against the matched control
    crit = pd.read_csv(synthetic["out"] / "stage2_criterion.csv")
    ctl = crit[(crit.arm == "A_detmean") & (crit.baseline == "A_ctrl200")]
    assert len(ctl) == 1 and ctl.iloc[0].comparison == "ablation vs matched control"
    assert ctl.iloc[0].n_pairs_q50 == 3 and ctl.iloc[0].b_status in ("holds", "violated")
    assert len(crit[(crit.arm == "A_detmean") & (crit.comparison == "vs baseline")]) == 1
    disp = pd.read_csv(synthetic["out"] / "stage2_dispersion.csv")
    assert set(disp[disp.arm == "A_detmean"].baseline) == {"A_base", "A_ctrl200"}
    # (f) arms without n_epochs_run have no stop-epoch rows
    se = pd.read_csv(synthetic["out"] / "stage2_stop_epoch.csv")
    assert "A_detmean" not in set(se.arm) and (se.n_updates > 0).all()
    # (g) cost: per-update ratios and the matched-control ratio of the method-5 pair
    cost = pd.read_csv(synthetic["out"] / "stage2_cost.csv").set_index("arm")
    assert cost.loc["A_base", "wall_per_update_ratio_vs_base"] == pytest.approx(1.0)
    assert cost.loc["A_ctrl200", "wall_ratio_vs_matched_control"] == pytest.approx(1.0)
    assert cost.loc["A_detmean", "wall_ratio_vs_matched_control"] == pytest.approx(1.0 / 25.0)
    assert np.isnan(cost.loc["A_base", "wall_ratio_vs_matched_control"])
    assert cost.loc["A_ctrl200", "mean_wall_sec_per_update"] == pytest.approx(25.0 / 25)
    mc = pd.read_csv(synthetic["out"] / "stage2_manifest_commits.csv")
    assert list(mc.manifest_commit) == ["abc1234"] and not mc.manifest_dirty.iloc[0]


def test_end_to_end_reports_figures_and_placeholder(synthetic: Dict[str, Any]) -> None:
    rep, out = synthetic["rep"], synthetic["out"]
    t1 = (rep / "04_pilot_stage1.md").read_text()
    assert t1.index("## 1. Checks") < t1.index("## 2. Completeness") < t1.index("## 5. Arms")
    for arm in STAGE1_ERR:
        assert f"### `{arm}`" in t1
    assert "10502" in t1 and "failed" in t1                      # the failed run is listed
    assert "Source: `" in t1 and "stage1_paired.csv" in t1
    assert "python tools/v2/refine_analysis.py stage1" in t1
    # MAJOR 1: per-(arm, q) set-aside counts and the one-sentence explanation
    assert "1 originals (all q = 50) were set aside because their manifests recorded dirty = true" \
        in t1
    assert "compare bit-identical to the re-runs" in t1 and "prereg Addendum 1 item 3" in t1
    assert "stage1_dirty_rerun_comparison.json" in t1 and "set_aside_originals_in_wave" not in t1
    # MAJOR 2: the check-(ii) outcome (section 1, section 4 and the B_expcont section)
    assert t1.count("is NOT met") >= 2 and "3.000e-05 DW at q = 50" in t1
    assert "<= 1e-6 DW) is NOT met" in t1 and "(development tier 1.000e-04 DW)" in t1
    assert "`B_expcont` is the only arm that meets the pre-registered criterion" in t1
    assert "open as stated" in t1 and "continuation_check.json" in t1
    assert "the refined verifier configuration" in t1 and "5.000e-07 DW" in t1
    # MINOR (b), (c), (d): within-run definition, commit note, n_zero column
    assert "sample SD (ddof = 1)" in t1 and "2100, 2125, 2150, 2175, 2200" in t1
    assert "HEAD advanced during the wave" in t1 and "prereg Addendum 1 item 1" in t1
    assert "`abc1234`" in t1 and "n(diff>0)/n(diff<0)/n(diff=0)" in t1
    assert (out / "checks_per_run.csv").exists()                  # MINOR (a): always written
    t2 = (rep / "05_pilot_stage2.md").read_text()
    for arm in S2_ARMS:
        assert f"### `{arm}`" in t2
    assert "Smoothing explanation test" in t2 and "Offline fixed-grid objective" in t2
    # MINOR (e), (g), (h), (i)
    assert "must be exactly 0" not in t2 and "about 1e-6 instead of 0" in t2
    assert "A_ctrl200` and `A_detmean` run 200 updates against 400" in t2
    assert "exploring-start rows" in t2 and "wall_per_update_ratio_vs_base" in t2
    assert "ablation vs matched control" in t2 and "matched control `A_ctrl200`" in t2
    assert t2.count("Pearson correlation over seeds") == 1        # once per annealing arm (one arm)
    assert "corr(d predicted gap" not in t2
    t6 = (rep / "06_decision_inputs.md").read_text()
    assert t6.rstrip().endswith("## Observations (what the data show; not recommendations)\n\n"
                                "<!-- OBSERVATIONS-PLACEHOLDER -->")
    for m in R.METHOD_ARMS:
        assert f"| {m} |" in t6
    assert "not reached on the grid" in t6 and "5e-07" in t6
    assert "d1_clamp/d1_flags.csv" in t6 and "d1_clamp_crR1" not in t6
    assert "`B_expcont` (phase B): M1 not exceeded" in t6      # per-arm D1 cell of method 6
    assert "locked pipeline (v11_reproduction, q = all): phase A: M1 exceeded" in t6
    d1a = pd.read_csv(out / "decision_d1_flags_arms.csv")
    assert set(d1a.group) == {"stage1/B_expcont", "stage2/A_anneal2"}
    di = pd.read_csv(out / "decision_inputs.csv")
    assert set(di.method) == set(R.METHOD_ARMS)
    figs = synthetic["tmp"] / "figs"
    for stem in ("stage1_B_expcont_paired_primary", "stage1_overview", "stage1_B_kl005_stop_epoch",
                 "stage2_A_anneal2_pred_vs_obs", "stage2_A_detmean_objective",
                 "stage2_A_detmean_profile", "stage2_overview"):
        assert (figs / f"{stem}.png").exists() and (figs / f"{stem}.pdf").exists(), stem
    import matplotlib
    assert matplotlib.rcParams["pdf.fonttype"] == 42


def test_end_to_end_deterministic(synthetic: Dict[str, Any], tmp_path: Path) -> None:
    common = list(synthetic["common"])
    common[common.index("--out") + 1] = str(tmp_path / "out2")
    common[common.index("--reports") + 1] = str(tmp_path / "rep2")
    R.main(["stage1", "--arms", *STAGE1_ERR, *common, "--figures", str(tmp_path / "f2")])
    for name in ("stage1_paired.csv", "stage1_criterion.csv", "stage1_dispersion.csv",
                 "stage1_per_run.csv"):
        a = (synthetic["out"] / name).read_bytes()
        b = (tmp_path / "out2" / name).read_bytes()
        assert a == b, name


def test_classify_run_states(tmp_path: Path) -> None:
    assert R.classify_run(tmp_path / "none")["status"] == "missing"
    d = tmp_path / "r"
    d.mkdir()
    (d / "status.json").write_text(json.dumps({"state": "running"}))
    assert R.classify_run(d)["status"] == "incomplete"
    (d / "status.json").write_text(json.dumps({"state": "done", "exit_code": 0}))
    assert R.classify_run(d)["status"] == "incomplete"          # no final_v2.json yet
    (d / "final_v2.json").write_text(json.dumps({"final": {"valid": True}}))
    assert R.classify_run(d)["status"] == "done"
    (d / "final_v2.json").write_text(json.dumps({"final": {"error": "DomainError"}}))
    assert R.classify_run(d)["status"] == "failed"
    (d / "status.json").write_text(json.dumps({"state": "done", "exit_code": 3}))
    assert R.classify_run(d)["status"] == "failed"
