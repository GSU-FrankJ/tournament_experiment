"""MS-R3 section 2.3 (a) tool (tools/ms/r3_supervised_screen.py): LR schedule, constants, the initial actor (bit-identical
across variants and equal to the runner's), the d stream and the start shares, the fit and its verifier-side metrics at
reduced budgets for every variant and both starts, the reload against ``mean_effort_numpy``, no-overwrite, ``summarise`` and
the premise check (PASS, DROP, STOP) on synthetic cells, and a CLI smoke with a process pool.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_r3_screen.py -p no:cacheprovider -q

Sections: 1. schedule and constants; 2. the initial actor; 3. the d stream and the start shares; 4. the fit and the
metrics (reduced budgets); 5. no overwrite, run / resume; 6. summarise and the premise check on synthetic cells; 7. CLI smoke.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import r3_supervised_screen as SS  # noqa: E402
from agents.ppo_curriculum import ACTOR_VARIANTS, BetaActor, PPOConfig, mean_effort_numpy  # noqa: E402
from agents.ppo_curriculum_v2 import CurriculumPPOv2  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402
from run import run_ms_stagewise as RUN  # noqa: E402
from run.run_final_dp_br import make_policy_fns  # noqa: E402
from run.run_final_dp_br_round3_dense import strict_dataclass  # noqa: E402
from utils.theory_multistage import g2_two_stage  # noqa: E402

PROTOCOL = ROOT / "protocols" / "v2_T2_locked_v2_0.json"
with open(PROTOCOL) as _f:
    PROTO = json.load(_f)
TINY = dict(steps=400, decay_steps=100, checkpoints=(100, 200, 400))
E0 = {50: 70.0, 60: 4.0 * 3500.0 / (4.0 * 60.0)}       # DW / (4 k q), k = 1 / 3500, DW = 4


def _ctx(q: int) -> Tuple[Dict[str, Any], GameSpec, PPOConfig]:
    rec = PROTO["records"][str(q)]
    return rec, strict_dataclass(GameSpec, rec["game"], "game"), strict_dataclass(PPOConfig, rec["ppo"], "ppo")


# ====================================================================== 1. schedule and constants
def test_lr_schedule_values_for_the_56000_step_run() -> None:
    cell = SS.CellSpec("t1", "bb", 50, 10501, SS.STEPS_MAIN, SS.DECAY_STEPS, SS.CHECKPOINTS_MAIN)
    assert cell.const_steps == 48000
    f = lambda s: SS.lr_at_step(s, 3e-4, 3e-5, cell.const_steps, cell.decay_steps)       # noqa: E731
    assert f(1) == 3e-4 and f(48000) == 3e-4
    assert f(48001) == pytest.approx(3e-4 + (3e-5 - 3e-4) * 1 / 8000, rel=1e-12)
    assert f(48001) == pytest.approx(3e-4 - 2.7e-4 / 8000, rel=1e-12) and f(48001) < 3e-4
    assert f(52000) == pytest.approx(1.65e-4, rel=1e-12)
    assert f(56000) == pytest.approx(3e-5, rel=1e-12)
    decay = [f(s) for s in range(48000, 56001)]
    assert all(a > b for a, b in zip(decay, decay[1:]))
    ext = SS.CellSpec("t1", "bb", 50, 10501, SS.STEPS_EXTENDED, SS.DECAY_STEPS, SS.CHECKPOINTS_EXTENDED, True)
    assert ext.const_steps == 216000 and ext.name == "t1_bb_q50_seed10501_ext"
    assert SS.lr_at_step(216000, 3e-4, 3e-5, ext.const_steps, ext.decay_steps) == 3e-4
    assert SS.lr_at_step(224000, 3e-4, 3e-5, ext.const_steps, ext.decay_steps) == pytest.approx(3e-5, rel=1e-12)


def test_the_run_grid_matches_the_prompt() -> None:
    cells = SS.plan_cells(ACTOR_VARIANTS, SS.STARTS, SS.QS, SS.SEEDS)
    main = [c for c in cells if not c.extended]
    ext = [c for c in cells if c.extended]
    assert len(main) == 3 * 2 * 2 * 10 and len(ext) == 2 * 10 and cells[: len(ext)] == ext
    assert {c.steps for c in main} == {56000} and {c.steps for c in ext} == {224000}
    assert {c.checkpoints for c in main} == {(16000, 32000, 48000, 56000)}
    assert {c.checkpoints for c in ext} == {(16000, 32000, 48000, 56000, 112000, 168000, 224000)}
    assert {(c.actor, c.starts) for c in ext} == {("t1", "bb")} and {c.seed for c in main} == set(range(10501, 10511))
    assert len({c.name for c in cells}) == len(cells) == 140
    assert SS.plan_cells(["relu"], ["bb"], [50], [1])[0].extended is False       # no extended cell without t1 / bb
    assert not any(c.extended for c in SS.plan_cells(ACTOR_VARIANTS, SS.STARTS, [50], [1], with_extended=False))
    assert SS.parse_seeds(["10501-10503", "7"]) == [10501, 10502, 10503, 7]


def test_screen_constants_equal_their_repository_sources() -> None:
    import ms_configs as MC                       # local import: only this test depends on the config builder
    rec, _, cfg = _ctx(50)
    win = MC.LEGACY_PIPELINE_NL["lr_windows"]["2"][0]
    assert SS.LR_END == win["end"] == PROTO["pipeline"]["lr_decay"][0]["end_lr"] and cfg.lr == win["start"] == 3e-4
    assert SS.STRAT_LAMBDA_P == MC.R2_ARM_TABLE["NL_st_s1"]["lambda_P"] == MC.R2_ARM_TABLE["NL_st_s16"]["lambda_P"]
    assert SS.STRAT_NEAR_HALF_WIDTH == MC.DEFAULT_PARAMS["near_tie_half_width"]
    assert SS.STRAT_ALPHA == 0.0
    # 56,000 = the 2800 terminal updates of MS-R2 x 20 minibatch steps (10 epochs x 512 learner rows / 256)
    per_update = rec["ppo"]["epochs"] * (rec["protocol"]["episodes_per_update"] // rec["ppo"]["minibatch"])
    assert per_update == 20 and SS.STEPS_MAIN == MC.LEGACY_PIPELINE_NL["budgets"]["2"] * per_update
    assert SS.STEPS_EXTENDED == 4 * SS.STEPS_MAIN and SS.DECAY_STEPS == 400 * per_update
    assert SS.SCREEN_STREAM_ID not in rec["protocol"]["rng_namespaces"].values()


@pytest.mark.parametrize("q", [50, 60])
def test_target_is_the_closed_form_tent_of_the_repository_function(q: int) -> None:
    _, spec, _ = _ctx(q)
    d = np.linspace(-spec.B, spec.B, 4001)
    e0 = spec.dw / (4.0 * spec.k * q)
    tent = e0 * np.maximum(0.0, 1.0 - np.abs(d) / (2.0 * q))
    g = g2_two_stage(d, q, spec.w_h, spec.w_l, spec.k, spec.e_max)
    assert np.allclose(g, tent, rtol=1e-12, atol=1e-12) and e0 == pytest.approx(E0[q], rel=1e-12)


# ====================================================================== 2. the initial actor
@pytest.mark.parametrize("q,seed", [(50, 10501), (60, 10507)])
def test_initial_actor_is_the_runners_and_bit_identical_across_variants(q: int, seed: int) -> None:
    rec, _, cfg = _ctx(q)
    ns = rec["protocol"]["rng_namespaces"]
    init_state = np.random.SeedSequence([seed, q, ns["init"]]).generate_state(1)[0]
    agent = CurriculumPPOv2(cfg, torch.Generator().manual_seed(int(init_state)), np.random.default_rng(0),
                            device=rec["device"])                    # what MSRun.__init__ builds
    ref = {k: v.detach().clone() for k, v in agent.actor.state_dict().items()}
    digests = set()
    for v in ACTOR_VARIANTS:
        actor, info = SS.build_initial_actor(cfg, seed, q, ns, v)
        assert actor.variant == v and actor.mu_clamp == cfg.mu_clamp and actor.c_min == cfg.c_min
        sd = actor.state_dict()
        assert set(sd) == set(ref)
        for k in ref:
            assert torch.equal(sd[k], ref[k]), (v, k)
        assert info["seed_material"] == [seed, q, ns["init"]] and info["torch_seed"] == int(init_state)
        digests.add(info["actor_sha256"])
    assert len(digests) == 1
    # the digest recipe is MSRun._init_digest's (actor then critic) with an empty critic
    stub = type("Stub", (), {})()
    stub.actor, stub.critic = actor, torch.nn.Identity()
    assert RUN.MSRun._init_digest(stub) == SS.actor_digest(actor) == digests.pop()
    # the output layer starts at zero: every variant's Beta mean is 0.5 (effort 50) everywhere
    m = SS.evaluate_fit(actor, _ctx(q)[1], 0.5)
    assert m["e_hat_0"] == pytest.approx(50.0, abs=1e-5)


def test_distinct_seeds_and_qs_give_distinct_initial_weights() -> None:
    rec, _, cfg = _ctx(50)
    ns = rec["protocol"]["rng_namespaces"]
    a = SS.build_initial_actor(cfg, 10501, 50, ns, "t1")[1]["actor_sha256"]
    b = SS.build_initial_actor(cfg, 10502, 50, ns, "t1")[1]["actor_sha256"]
    c = SS.build_initial_actor(cfg, 10501, 60, ns, "t1")[1]["actor_sha256"]
    assert len({a, b, c}) == 3


# ====================================================================== 3. the d stream and the start shares
@pytest.mark.parametrize("q,lam_t", [(50, 0.5), (60, 20.0 / 44.0)])
def test_stratified_bin_probabilities_sum_to_one_with_the_tail_and_near_shares(q: int, lam_t: float) -> None:
    _, spec, _ = _ctx(q)
    sampler = StartSampler(spec, 10.0)
    p = sampler.stratified_bin_probs(2, SS.STRAT_LAMBDA_P, SS.STRAT_NEAR_HALF_WIDTH, SS.STRAT_ALPHA)
    st = sampler.strata(2, SS.STRAT_NEAR_HALF_WIDTH)
    assert p.sum() == pytest.approx(1.0, abs=1e-12) and (p > 0).all() and p.size == (40 if q == 50 else 44)
    assert p[st["tail"]].sum() == pytest.approx(lam_t, abs=1e-12)
    assert p[st["near"]].sum() == pytest.approx(0.35, abs=1e-12)
    assert p[st["mid"]].sum() == pytest.approx(1.0 - 0.35 - lam_t, abs=1e-12)
    for k in st:                                                       # uniform within a stratum
        assert np.ptp(p[st[k]]) < 1e-15
    _, info = SS.make_start_draw("st", spec, 10.0, 10501, q)
    assert info["n_bins"] == p.size and sum(info["bin_probs"]) == pytest.approx(1.0, abs=1e-12)
    assert info["stratum_shares"]["tail"] == pytest.approx(lam_t, abs=1e-12)
    assert info["stratum_shares"]["near"] == pytest.approx(0.35, abs=1e-12)
    _, info_bb = SS.make_start_draw("bb", spec, 10.0, 10501, q)
    assert info_bb["stratum_shares"]["tail"] == pytest.approx(lam_t, abs=1e-12)       # bin-balanced tail share
    assert info_bb["stratum_shares"]["near"] == pytest.approx(4.0 / p.size, abs=1e-12)


@pytest.mark.parametrize("q", [50, 60])
def test_d_stream_is_the_documented_one_and_draws_the_stated_shares(q: int) -> None:
    rec, spec, _ = _ctx(q)
    seed = 10503
    draw_bb, info_bb = SS.make_start_draw("bb", spec, 10.0, seed, q)
    assert info_bb["seed_material"] == [seed, q, SS.SCREEN_STREAM_ID] and info_bb["stream_id"] == SS.SCREEN_STREAM_ID
    ref_rng = np.random.default_rng(np.random.SeedSequence([seed, q, SS.SCREEN_STREAM_ID]))
    sampler = StartSampler(spec, 10.0)
    first = draw_bb(256)
    assert first.shape == (256,) and np.array_equal(first, sampler.balanced(2, 256, ref_rng))
    again, _ = SS.make_start_draw("bb", spec, 10.0, seed, q)
    assert np.array_equal(again(256), first)                          # a stream is reproducible from its seed material
    n = 400_000
    lam_t = 0.5 if q == 50 else 20.0 / 44.0
    for starts, tail, near in (("bb", lam_t, 4.0 / (40 if q == 50 else 44)), ("st", lam_t, 0.35)):
        draw, _ = SS.make_start_draw(starts, spec, 10.0, seed, q)
        d = draw(n)
        assert np.abs(d).max() <= spec.B
        for share, got in ((tail, np.mean(np.abs(d) >= 2 * q)), (near, np.mean(np.abs(d) < 20.0))):
            assert abs(got - share) < 5.0 * math.sqrt(share * (1 - share) / n), (starts, share, got)
        assert abs(np.mean(d < 0) - 0.5) < 5.0 * math.sqrt(0.25 / n)


def test_the_stream_is_shared_by_the_variants_of_a_cell() -> None:
    cells = [SS.fit_cell(SS.CellSpec(v, "st", 50, 10501, steps=40, decay_steps=10, checkpoints=(40,)))
             for v in ACTOR_VARIANTS]
    assert cells[0]["d_stream"] == cells[1]["d_stream"] == cells[2]["d_stream"]
    assert cells[0]["init"] == cells[1]["init"] == cells[2]["init"]


# ====================================================================== 4. the fit and the metrics (reduced budgets)
@pytest.fixture(scope="module")
def tiny() -> Dict[Tuple[str, str], Tuple[BetaActor, Dict[str, Any]]]:
    """Every variant x both starts, q = 50, seed 10501, 400 steps (LR decay over the last 100)."""
    return {(v, st): SS.fit_cell_actor(SS.CellSpec(v, st, 50, 10501, **TINY))
            for v in ACTOR_VARIANTS for st in SS.STARTS}


def test_reduced_budget_cells_for_all_variants_and_both_starts(tiny) -> None:
    assert len(tiny) == 6
    for (v, st), (actor, rec) in tiny.items():
        assert rec["schema"] == SS.CELL_SCHEMA and rec["cell"]["name"] == f"{v}_{st}_q50_seed10501"
        assert rec["cell"]["actor"] == v and actor.variant == v and rec["g2_at_0"] == pytest.approx(70.0)
        assert [c["step"] for c in rec["checkpoints"]] == [100, 200, 400]
        assert [c["lr"] for c in rec["checkpoints"]][:2] == [3e-4, 3e-4]
        assert rec["checkpoints"][-1]["lr"] == pytest.approx(3e-5, rel=1e-12)
        assert rec["budget"]["steps"] == 400 and rec["budget"]["const_steps"] == 300
        assert rec["budget"]["minibatch"] == 256 and rec["budget"]["max_grad_norm"] == 0.5
        assert rec["d_stream"]["seed_material"] == [10501, 50, SS.SCREEN_STREAM_ID]
        assert rec["init"]["seed_material"] == [10501, 50, 0] and len(rec["init"]["actor_sha256"]) == 64
        assert rec["protocol"]["sha256"] == SS.sha256_file(PROTOCOL) and rec["versions_match_record"] is True
        assert rec["wall_sec"] > 0 and rec["process_cpu_sec"] > 0 and set(rec["versions"]) == {"python", "torch", "numpy"}
        for c in rec["checkpoints"]:
            assert all(math.isfinite(c[m]) for m in SS.METRICS)
            assert c["tip_deficit"] == pytest.approx(70.0 - c["e_hat_0"], abs=1e-12)
            assert c["tip_deficit_over_g2_0"] == pytest.approx(c["tip_deficit"] / 70.0, abs=1e-12)
            assert c["rmse_pos_over_g2_0"] == pytest.approx(c["rmse_pos"] / 70.0, rel=1e-12)
            assert c["tail_mean_over_g2_0"] == pytest.approx(c["tail_mean"] / 70.0, rel=1e-12)
            assert c["w_eff"] == pytest.approx(c["tip_deficit"] / (70.0 / 100.0), rel=1e-12)       # deficit / (e2*(0) / 2q)
        # the fit reduces the training loss (mean over the window since the previous checkpoint); the t1 actor
        # sits on its flat plateau at this budget, the kink-capable variants leave it
        first, last = rec["checkpoints"][0]["train_mse"], rec["checkpoints"][-1]["train_mse"]
        assert last < first and (v == "t1" or last < 0.5 * first)
    # the same (q, seed) -> bit-identical initial weights for every variant and both starts
    assert len({rec["init"]["actor_sha256"] for _, rec in tiny.values()}) == 1
    # the variants differ in what they learn
    assert len({rec["checkpoints"][-1]["e_hat_0"] for _, rec in tiny.values()}) > 3


def _numpy_reference_fit(actor: BetaActor, variant: str, spec: GameSpec, batches: Sequence[np.ndarray],
                         lrs: Sequence[float]) -> Tuple[List[np.ndarray], List[float]]:
    """Independent float64 implementation of the fit (the formulas of the PI sandbox, written out by hand): manual
    back-propagation of the MSE in effort units through the mean head, global-norm clip 0.5, Adam (0.9, 0.999, 1e-8)."""
    q, B = float(spec.q), spec.B
    e0 = spec.dw / (4.0 * spec.k * q)
    f64 = lambda t: t.detach().numpy().astype(np.float64).copy()                      # noqa: E731
    W1, b1, W2, b2, W3, b3 = (f64(actor.l1.weight), f64(actor.l1.bias), f64(actor.l2.weight), f64(actor.l2.bias),
                              f64(actor.out.weight)[:1], f64(actor.out.bias)[:1])
    params = [W1, b1, W2, b2, W3, b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    if variant == "relu":
        act, dact, scale = (lambda x: np.maximum(x, 0.0)), (lambda x: (x > 0).astype(float)), 1.0
    else:
        act, dact, scale = np.tanh, (lambda x: 1.0 - np.tanh(x) ** 2), (10.0 if variant == "t10" else 1.0)
    losses = []
    for step, (d, lr) in enumerate(zip(batches, lrs), 1):
        x = np.stack([np.ones_like(d), d / B * scale], 1)                              # (1, d / B), d column scaled
        a1 = x @ W1.T + b1
        h1 = act(a1)
        a2 = h1 @ W2.T + b2
        h2 = act(a2)
        mu = 1.0 / (1.0 + np.exp(-(h2 @ W3.T + b3)[:, 0]))
        e = 100.0 * mu
        tgt = e0 * np.maximum(0.0, 1.0 - np.abs(d) / (2.0 * q))
        losses.append(float(np.mean((e - tgt) ** 2)))
        g_z = 2.0 * (e - tgt) / d.size * 100.0 * mu * (1.0 - mu)
        g_a2 = g_z[:, None] * W3 * dact(a2)
        g_a1 = (g_a2 @ W2) * dact(a1)
        grads = [g_a1.T @ x, g_a1.sum(0), g_a2.T @ h1, g_a2.sum(0), g_z[None, :] @ h2, np.array([g_z.sum()])]
        gn = math.sqrt(sum(float((g * g).sum()) for g in grads))
        if gn > 0.5:
            grads = [g * (0.5 / gn) for g in grads]
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** step)) / (np.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8)
    return params, losses


@pytest.mark.parametrize("variant", ACTOR_VARIANTS)
def test_the_torch_fit_equals_an_independent_numpy_implementation(variant: str) -> None:
    """Same initial weights, same minibatches, same LR sequence: the real-class fit (float32, torch Adam and
    clip_grad_norm_) and a hand-written float64 implementation end in the same weights; the first-step loss is the
    MSE in effort units of the flat start (e = 50 everywhere)."""
    rec, spec, cfg = _ctx(50)
    n, const, decay = 25, 20, 5
    rng = np.random.default_rng(5)
    batches = [rng.uniform(-spec.B, spec.B, cfg.minibatch) for _ in range(n)]
    actor, _ = SS.build_initial_actor(cfg, 10501, 50, rec["protocol"]["rng_namespaces"], variant)
    lrs = [SS.lr_at_step(s, 3e-4, 3e-5, const, decay) for s in range(1, n + 1)]
    ref_params, ref_losses = _numpy_reference_fit(actor, variant, spec, batches, lrs)
    it = iter(batches)
    opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr, betas=cfg.adam_betas, eps=cfg.adam_eps,
                           weight_decay=cfg.weight_decay)
    cell = SS.CellSpec(variant, "bb", 50, 10501, n, decay, (1, n))
    cps = SS.fit_actor(actor, opt, spec, cfg, lambda k: next(it), cell, 3e-4, 3e-5, 0.5)
    mine = [actor.l1.weight, actor.l1.bias, actor.l2.weight, actor.l2.bias, actor.out.weight[:1], actor.out.bias[:1]]
    for t, p in zip(mine, ref_params):
        assert np.max(np.abs(t.detach().numpy().astype(np.float64) - p)) < 1e-5     # observed ~1.6e-7 (float32)
    tgt0 = 70.0 * np.maximum(0.0, 1.0 - np.abs(batches[0]) / 100.0)
    assert cps[0]["train_mse"] == pytest.approx(float(np.mean((50.0 - tgt0) ** 2)), rel=1e-5)
    assert cps[0]["train_mse"] == pytest.approx(ref_losses[0], rel=1e-5)
    assert cps[1]["train_mse"] == pytest.approx(float(np.mean(ref_losses[1:])), rel=1e-5)
    assert any(abs(lr - 3e-4) > 1e-9 for lr in lrs) and lrs[-1] == pytest.approx(3e-5, rel=1e-12)


MS_R2_PILOT = ROOT / "results" / "ms_r2" / "pilot"


@pytest.mark.skipif(not (MS_R2_PILOT / "q50" / "seed10501" / "NL_bb_s1" / "weights" / "u02800.npz").is_file(),
                    reason="the MS-R2 pilot runs (read-only reference) are absent")
@pytest.mark.parametrize("q,arm", [(50, "NL_bb_s1"), (60, "NL_st_s16")])
def test_the_metrics_reproduce_the_check_row_of_an_ms_r2_run(q: int, arm: str) -> None:
    """``evaluate_fit`` on the terminal-stage export u02800 of an RL run (a trained t1 actor, once with a concentration
    scale) gives the numbers in the run's own ``ms_checks_stage2.csv`` row of update 2800 (the verifier's metrics)."""
    run = MS_R2_PILOT / f"q{q}" / "seed10501" / arm
    _, spec, cfg = _ctx(q)
    z = np.load(run / "weights" / "u02800.npz")
    net = BetaActor(cfg.hidden, cfg.c_min, cfg.mu_clamp, torch.Generator().manual_seed(0))
    net.load_state_dict({k[6:]: torch.as_tensor(z[k]) for k in z.files if k.startswith("actor.")})
    if "conc_scale" in z.files:
        net.conc_scale = float(z["conc_scale"])
    row = [r for r in csv.DictReader(open(run / "ms_checks_stage2.csv")) if r["update"] == "2800"]
    assert len(row) == 1
    row = row[0]
    m = SS.evaluate_fit(net, spec, 0.5)
    assert m["e_hat_0"] == pytest.approx(float(row["e2_at_0"]), abs=1e-9)
    assert m["tip_deficit"] == pytest.approx(float(row["gap"]), abs=1e-9)
    assert m["tip_deficit_over_g2_0"] == pytest.approx(-float(row["stage2_peak_rel_err_signed"]), abs=1e-12)
    assert m["rmse_pos_over_g2_0"] == pytest.approx(float(row["stage2_rmse_pos_over_g2_0"]), abs=1e-12)
    assert m["tail_mean_over_g2_0"] == pytest.approx(float(row["stage2_tail_mean_over_g2_0"]), abs=1e-12)
    assert m["g2_at_0"] == pytest.approx(float(row["g2_0"]), abs=1e-12)


def test_cells_are_deterministic_and_the_hook_is_gone() -> None:
    cell = SS.CellSpec("relu", "st", 60, 10502, steps=150, decay_steps=50, checkpoints=(50, 150))
    a1, r1 = SS.fit_cell_actor(cell)
    a2, r2 = SS.fit_cell_actor(cell)
    assert r1["checkpoints"] == r2["checkpoints"]                                  # equal floats, no tolerance
    assert SS.actor_digest(a1) == SS.actor_digest(a2) != r1["init"]["actor_sha256"]
    assert len(a1.out._forward_hooks) == 0                                         # the fit removed its hook


def test_the_concentration_head_gets_no_gradient(tiny) -> None:
    for (v, st), (actor, rec) in tiny.items():
        assert torch.count_nonzero(actor.out.weight[1]) == 0 and actor.out.bias[1].item() == 0.0, (v, st)
        assert torch.count_nonzero(actor.out.weight[0]) > 0                        # the mean head was trained


def test_the_extended_cell_repeats_the_main_cell_until_the_lr_decay_starts() -> None:
    main = SS.fit_cell(SS.CellSpec("t1", "bb", 50, 10501, 300, 100, (100, 200, 300)))
    ext = SS.fit_cell(SS.CellSpec("t1", "bb", 50, 10501, 600, 100, (100, 200, 600), True))
    assert main["checkpoints"][:2] == ext["checkpoints"][:2]                       # constant LR on both sides until 200
    assert [c["lr"] for c in main["checkpoints"]] == [3e-4, 3e-4, pytest.approx(3e-5, rel=1e-12)]
    assert ext["checkpoints"][2]["lr"] == pytest.approx(3e-5, rel=1e-12) and ext["budget"]["const_steps"] == 500
    assert ext["cell"]["extended"] is True and main["cell"]["extended"] is False


@pytest.mark.parametrize("variant", ACTOR_VARIANTS)
def test_e_hat_0_equals_mean_effort_numpy_of_an_exported_copy(tiny, tmp_path: Path, variant: str) -> None:
    actor, rec = tiny[(variant, "bb")]
    _, spec, cfg = _ctx(50)
    agent = CurriculumPPOv2(cfg, torch.Generator().manual_seed(0), np.random.default_rng(0))
    agent.actor.load_state_dict(actor.state_dict())
    agent.set_actor_variant(variant)
    path = tmp_path / "w.npz"
    agent.export_weights_npz(str(path))                                           # the repo's own export
    z = np.load(path)
    w = {k: z[k] for k in z.files}
    assert ("actor_variant" in w) == (variant != "t1")
    e_np, _, _ = mean_effort_numpy(w, spec.encode_obs(2, np.array([0.0])), cfg.c_min, cfg.mu_clamp,
                                   spec.e_min, spec.e_max)
    assert float(e_np[0]) == pytest.approx(rec["checkpoints"][-1]["e_hat_0"], rel=1e-5, abs=1e-4)
    grid = np.linspace(-spec.B, spec.B, 801)
    mean_fn, _ = make_policy_fns(SS._ActorShim(actor), spec)
    e_grid, _, _ = mean_effort_numpy(w, spec.encode_obs(2, grid), cfg.c_min, cfg.mu_clamp, spec.e_min, spec.e_max)
    assert np.max(np.abs(e_grid - mean_fn(2, grid))) < 1e-3                       # the reload is the screen's function


def test_t10_reports_ten_times_the_stored_first_layer_weight(tiny) -> None:
    for v, factor in (("t1", 1.0), ("relu", 1.0), ("t10", 10.0)):
        actor, rec = tiny[(v, "bb")]
        stored = float(actor.l1.weight[:, 1].abs().max())
        assert SS.max_abs_d_weight(actor) == pytest.approx(factor * stored, rel=1e-12)
        assert rec["checkpoints"][-1]["max_abs_w_d"] == pytest.approx(factor * stored, rel=1e-6)
    # only the d column (column 1) enters; the stage feature (column 0) does not
    actor = tiny[("t10", "bb")][0]
    with torch.no_grad():
        actor.l1.weight[:, 0] = 5.0
        actor.l1.weight[:, 1] = torch.linspace(-0.25, 0.125, actor.l1.weight.shape[0])
    assert SS.max_abs_d_weight(actor) == pytest.approx(2.5, rel=1e-6)           # 10 x max |linspace| = 10 x 0.25


@pytest.mark.parametrize("q", [50, 60])
def test_evaluate_fit_matches_an_independent_computation_for_a_constant_actor(q: int) -> None:
    rec, spec, cfg = _ctx(q)
    actor, _ = SS.build_initial_actor(cfg, 10501, q, rec["protocol"]["rng_namespaces"], "t1")
    b0 = math.log(0.3 / 0.7)
    with torch.no_grad():
        actor.out.weight.zero_()
        actor.out.bias.copy_(torch.tensor([b0, 0.0]))
    m = SS.evaluate_fit(actor, spec, 0.5)
    c = 100.0 * 0.3
    D = np.linspace(-spec.B, spec.B, int(round(4 * spec.B)) + 1)                 # step 0.5, 0 an exact node
    tent = E0[q] * np.maximum(0.0, 1.0 - np.abs(D) / (2.0 * q))
    pos = np.abs(D) < 2.0 * q
    assert m["e_hat_0"] == pytest.approx(c, rel=1e-6) and m["g2_at_0"] == pytest.approx(E0[q], rel=1e-12)
    assert m["tip_deficit"] == pytest.approx(E0[q] - c, rel=1e-6)
    assert m["rmse_pos"] == pytest.approx(math.sqrt(np.mean((c - tent[pos]) ** 2)), rel=1e-6)
    assert m["tail_mean"] == pytest.approx(c, rel=1e-6)
    assert m["w_eff"] == pytest.approx((E0[q] - c) * 2.0 * q / E0[q], rel=1e-6)


# ====================================================================== 5. no overwrite, run / resume
def test_write_new_json_never_overwrites(tmp_path: Path) -> None:
    p = tmp_path / "a" / "x.json"
    SS.write_new_json(p, {"a": 1})
    with pytest.raises(FileExistsError):
        SS.write_new_json(p, {"a": 2})
    assert json.load(open(p)) == {"a": 1}
    assert [f.name for f in p.parent.iterdir()] == ["x.json"]                        # no temporary file is left behind


def test_run_refuses_existing_cells_and_resume_fills_in_only_the_missing_ones(tmp_path: Path, capsys) -> None:
    out = tmp_path / "scr"
    base = ["run", "--out", str(out), "--workers", "1", "--steps", "60", "--decay-steps", "20", "--checkpoints", "30",
            "--qs", "50", "--actors", "t1", "--starts", "bb", "--no-extended"]
    assert SS.main(base + ["--seeds", "10501"]) == 0
    f = out / "cells" / "t1_bb_q50_seed10501.json"
    before = f.read_bytes()
    assert (out / "run_manifest.json").is_file()
    capsys.readouterr()
    assert SS.main(base + ["--seeds", "10501"]) == 2                                 # refuses: nothing is overwritten
    msg = capsys.readouterr().out
    assert "already exist" in msg and "--resume" in msg and f.read_bytes() == before
    assert not (out / "run_manifest_2.json").exists()
    assert SS.main(base + ["--seeds", "10501", "--resume"]) == 0
    assert "nothing to run" in capsys.readouterr().out and f.read_bytes() == before
    assert SS.main(base + ["--seeds", "10501", "10502", "--resume"]) == 0
    assert f.read_bytes() == before and (out / "cells" / "t1_bb_q50_seed10502.json").is_file()
    man2 = json.load(open(out / "run_manifest_2.json"))
    assert man2["plan"]["cells"] == ["t1_bb_q50_seed10502"] and man2["plan"]["n_skipped_existing"] == 1
    assert man2["constants"]["SCREEN_STREAM_ID"] == SS.SCREEN_STREAM_ID and len(man2["tool_sha256"]) == 64


def test_a_failing_cell_is_reported_with_exit_3_and_leaves_no_cell_file(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "proto.json"
    bad.write_text(json.dumps({"q_values": [50], "records": {}}))                      # no record for q = 50: the cell fails
    out = tmp_path / "scr"
    argv = ["run", "--out", str(out), "--workers", "1", "--protocol", str(bad), "--qs", "50", "--seeds", "1",
            "--actors", "t1", "--starts", "bb", "--no-extended", "--steps", "40", "--decay-steps", "10"]
    assert SS.main(argv) == 3
    text = capsys.readouterr().out
    assert "FAILED" in text and "1 ok" not in text and "0 ok, 1 failed" in text
    assert not list((out / "cells").glob("*.json")) and (out / "run_manifest.json").is_file()


def test_run_stops_on_bad_input_without_writing(tmp_path: Path, capsys) -> None:
    out = tmp_path / "scr"
    assert SS.main(["run", "--out", str(out), "--qs", "55", "--seeds", "1", "--steps", "40", "--decay-steps", "10"]) == 2
    assert "q_values" in capsys.readouterr().out and not out.exists()
    assert SS.main(["run", "--out", str(out), "--qs", "50", "--seeds", "1", "--steps", "40", "--decay-steps", "40"]) == 2
    assert not out.exists()
    assert SS.main(["run", "--out", str(out), "--qs", "50", "--seeds", "1", "--protocol", str(tmp_path / "no.json")]) == 2
    assert not out.exists()


# ====================================================================== 6. summarise and the premise check (synthetic cells)
def _synthetic_cell(actor: str, starts: str, q: int, seed: int, deficit: float, step: int = 56000,
                    extended: bool = False, sha: str = "a" * 64, extra_steps: Sequence[int] = ()) -> Dict[str, Any]:
    def cp(s: int, dfc: float) -> Dict[str, Any]:
        return {"step": s, "lr": 3e-4, "e_hat_0": E0[q] - dfc, "g2_at_0": E0[q], "tip_deficit": dfc,
                "tip_deficit_over_g2_0": dfc / E0[q], "rmse_pos": 0.5 + dfc, "rmse_pos_over_g2_0": (0.5 + dfc) / E0[q],
                "tail_mean": 0.1, "tail_mean_over_g2_0": 0.1 / E0[q], "w_eff": dfc * 2 * q / E0[q], "max_abs_w_d": 1.7,
                "train_mse": 0.3}
    return {"schema": SS.CELL_SCHEMA, "cell": {"name": "x", "actor": actor, "starts": starts, "q": q, "seed": seed,
                                                 "extended": extended},
            "budget": {"steps": step}, "init": {"actor_sha256": sha},
            "checkpoints": [cp(s, deficit * 2) for s in extra_steps] + [cp(step, deficit)]}


def _write_scenario(root: Path, table: Dict[Tuple[str, int], Sequence[float]], starts: str = "bb",
                    shas: Optional[Dict[Tuple[str, int, int], str]] = None, **kw: Any) -> List[Path]:
    """One synthetic cell per (actor, q, seed index) with the given deficits (the seeds are 10501...);
    ``shas`` overrides the initial-weight hash of a cell by (actor, q, seed index)."""
    paths = []
    for (actor, q), vals in table.items():
        for i, dfc in enumerate(vals):
            c = _synthetic_cell(actor, starts, q, 10501 + i, dfc, sha=(shas or {}).get((actor, q, i), "a" * 64), **kw)
            p = root / "cells" / f"{actor}_{starts}_q{q}_seed{10501 + i}.json"
            SS.write_new_json(p, c)
            paths.append(p)
    return paths


PASS_TABLE = {("t1", 50): [1.8, 1.9, 2.0, 2.1], ("t1", 60): [1.2, 1.4, 1.6, 6.4],
              ("relu", 50): [0.4, 0.5, 0.5, 0.6], ("relu", 60): [0.0, 0.04, 0.05, 0.1],
              ("t10", 50): [0.5, 0.54, 0.54, 0.6], ("t10", 60): [0.4, 0.45, 0.45, 0.5]}


def _premise(tmp_path: Path, table: Dict[Tuple[str, int], Sequence[float]], capsys, **kw: Any) -> Dict[str, Any]:
    root = tmp_path / "s"
    _write_scenario(root, table)
    assert SS.main(["summarise", "--out", str(root), "--expected-seeds", "4"]) == 0
    capsys.readouterr()
    return json.load(open(root / "premise_check.json"))


def _with(table: Dict[Tuple[str, int], Sequence[float]], **upd: Sequence[float]) -> Dict[Tuple[str, int], Sequence[float]]:
    out = dict(table)
    for k, v in upd.items():
        actor, q = k.rsplit("_q", 1)
        out[(actor, int(q))] = v
    return out


def test_premise_pass(tmp_path: Path, capsys) -> None:
    pc = _premise(tmp_path, PASS_TABLE, capsys)
    assert pc["outcome"] == "PASS" and pc["complete"] is True and pc["variants_failing_ii"] == []
    assert pc["median_tip_deficit"]["t1"] == {"50": pytest.approx(1.95), "60": pytest.approx(1.5)}   # even n: mean of the middle two
    assert pc["median_tip_deficit"]["relu"]["60"] == pytest.approx(0.045)
    assert pc["condition_i"]["pass"] is True and pc["condition_ii"]["relu"]["pass"] and pc["condition_ii"]["t10"]["pass"]
    r = pc["condition_ii"]["relu"]["by_q"]["50"]
    assert r["median_variant"] == pytest.approx(0.5) and r["median_t1"] == pytest.approx(1.95)
    assert r["limit"] == pytest.approx(0.975) and r["ratio"] == pytest.approx(0.5 / 1.95)
    assert pc["per_seed_tip_deficit"]["t1"]["60"]["10504"] == 6.4 and pc["n_seeds"]["t10"] == {"50": 4, "60": 4}
    assert pc["step"] == 56000 and pc["starts"] == "bb" and pc["init_identical_across_actors"] is True
    assert pc["expected_seeds"] == 4 and pc["rule"]["i"] and pc["rule"]["ii"]


def test_premise_drop_relu(tmp_path: Path, capsys) -> None:
    pc = _premise(tmp_path, _with(PASS_TABLE, relu_q60=[0.8, 0.8, 0.9, 1.0]), capsys)       # median 0.85 > 0.75 at q = 60 only
    assert pc["outcome"] == "DROP relu" and pc["variants_failing_ii"] == ["relu"]
    assert pc["condition_ii"]["relu"]["by_q"]["50"]["pass"] and not pc["condition_ii"]["relu"]["by_q"]["60"]["pass"]
    assert pc["condition_ii"]["t10"]["pass"] and pc["condition_i"]["pass"]


def test_premise_drop_t10(tmp_path: Path, capsys) -> None:
    pc = _premise(tmp_path, _with(PASS_TABLE, t10_q50=[1.0, 1.0, 1.1, 1.1]), capsys)         # median 1.05 > 0.975
    assert pc["outcome"] == "DROP t10" and pc["variants_failing_ii"] == ["t10"]


def test_premise_stop_when_i_fails(tmp_path: Path, capsys) -> None:
    pc = _premise(tmp_path, _with(PASS_TABLE, t1_q60=[0.8, 0.9, 0.9, 1.0]), capsys)          # median 0.9 < 1.0 at q = 60
    assert pc["outcome"] == "STOP" and pc["condition_i"]["pass"] is False
    assert pc["condition_i"]["by_q"]["60"]["pass"] is False and pc["condition_i"]["by_q"]["50"]["pass"] is True
    assert "(i)" in pc["outcome_reason"]


def test_premise_stop_when_both_variants_fail_ii(tmp_path: Path, capsys) -> None:
    table = _with(PASS_TABLE, relu_q50=[1.0, 1.0, 1.1, 1.1], t10_q60=[0.9, 0.9, 1.0, 1.0])
    pc = _premise(tmp_path, table, capsys)
    assert pc["outcome"] == "STOP" and pc["condition_i"]["pass"] is True
    assert pc["variants_failing_ii"] == ["relu", "t10"] and "both" in pc["outcome_reason"]


def test_premise_boundaries_are_inclusive(tmp_path: Path, capsys) -> None:
    table = {("t1", 50): [1.0] * 4, ("t1", 60): [2.0] * 4, ("relu", 50): [0.5] * 4, ("relu", 60): [1.0] * 4,
             ("t10", 50): [0.0] * 4, ("t10", 60): [-1.0] * 4}          # (i) at exactly 1.0, (ii) at exactly 0.5 x t1
    pc = _premise(tmp_path, table, capsys)
    assert pc["outcome"] == "PASS" and pc["condition_ii"]["relu"]["by_q"]["60"]["ratio"] == 0.5


def test_premise_uses_only_the_main_bin_balanced_cells_at_the_premise_step(tmp_path: Path, capsys) -> None:
    root = tmp_path / "s"
    _write_scenario(root, PASS_TABLE)
    bad = {k: [99.0] * 4 for k in PASS_TABLE}
    _write_scenario(root, bad, starts="st")                                                  # stratified cells: not gated
    for (actor, q), vals in bad.items():                                                     # extended cells and other steps
        for i, dfc in enumerate(vals):
            if actor == "t1":
                SS.write_new_json(root / "cells" / f"{actor}_bb_q{q}_seed{10501 + i}_ext.json",
                                  _synthetic_cell(actor, "bb", q, 10501 + i, dfc, step=224000, extended=True,
                                                  extra_steps=(56000,)))
    assert SS.main(["summarise", "--out", str(root), "--expected-seeds", "4"]) == 0
    pc = json.load(open(root / "premise_check.json"))
    assert pc["outcome"] == "PASS" and pc["median_tip_deficit"]["t1"]["50"] == pytest.approx(1.95)
    rows = list(csv.DictReader(open(root / "summary_extended.csv")))
    assert {r["steps"] for r in rows} == {"56000", "224000"} and {r["n"] for r in rows} == {"4"}


def test_premise_flags_incomplete_groups_and_differing_initial_weights(tmp_path: Path, capsys) -> None:
    root = tmp_path / "s"
    _write_scenario(root, _with(PASS_TABLE, relu_q50=[0.4, 0.5, 0.5]),                       # relu q50: one seed short
                    shas={("t10", 50, 0): "b" * 64})                                         # t10 q50 seed 10501: other init
    assert SS.main(["summarise", "--out", str(root), "--expected-seeds", "4"]) == 0
    out = capsys.readouterr().out
    pc = json.load(open(root / "premise_check.json"))
    assert pc["complete"] is False and pc["n_seeds"]["relu"]["50"] == 3 and "WARNING" in out
    assert pc["init_identical_across_actors"] is False
    assert pc["outcome"] == "PASS"                      # computed from the available data; the flags say what it rests on


def test_summarise_writes_the_tables_and_never_overwrites(tmp_path: Path, capsys) -> None:
    root = tmp_path / "s"
    _write_scenario(root, PASS_TABLE, extra_steps=(16000,))
    _write_scenario(root, {k: [0.1, 0.2, 0.3, 0.4] for k in PASS_TABLE}, starts="st")
    assert SS.main(["summarise", "--out", str(root), "--expected-seeds", "4"]) == 0
    out = capsys.readouterr().out
    assert "premise check at step 56000" in out and "outcome PASS" in out
    names = sorted(p.name for p in root.iterdir() if p.is_file())
    assert names == ["premise_check.json", "summary_by_cell.csv", "summary_median.csv"]       # no extended cell: no extended file
    by_cell = list(csv.DictReader(open(root / "summary_by_cell.csv")))
    assert len(by_cell) == 24 * 2 + 24 and list(by_cell[0]) == SS.ROW_COLS
    r = [x for x in by_cell if (x["actor"], x["starts"], x["q"], x["seed"], x["steps"]) == ("t1", "bb", "60", "10504", "56000")]
    assert len(r) == 1 and float(r[0]["tip_deficit"]) == 6.4 and r[0]["extended"] == "0" and r[0]["cell_steps"] == "56000"
    med = list(csv.DictReader(open(root / "summary_median.csv")))
    g = [x for x in med if (x["actor"], x["starts"], x["q"], x["steps"]) == ("t1", "bb", "60", "56000")]
    assert len(g) == 1 and g[0]["n"] == "4"
    assert float(g[0]["tip_deficit_median"]) == pytest.approx(1.5) and float(g[0]["tip_deficit_min"]) == 1.2
    assert float(g[0]["tip_deficit_max"]) == 6.4
    g16 = [x for x in med if (x["actor"], x["starts"], x["q"], x["steps"]) == ("t1", "bb", "60", "16000")]
    assert float(g16[0]["tip_deficit_median"]) == pytest.approx(3.0)
    assert [x["actor"] for x in med][:2] == ["t1", "t1"] and set(med[0]) == set(SS.MEDIAN_COLS)
    before = {n: (root / n).read_bytes() for n in names}
    assert SS.main(["summarise", "--out", str(root), "--expected-seeds", "4"]) == 2           # refuses to overwrite
    assert "already exist" in capsys.readouterr().out
    assert {n: (root / n).read_bytes() for n in names} == before
    assert SS.main(["summarise", "--out", str(root), "--expected-seeds", "4", "--tag", "v2"]) == 0
    assert (root / "premise_check_v2.json").is_file() and (root / "summary_median_v2.csv").is_file()


def test_summarise_without_a_complete_premise_group_writes_the_tables_and_exits_3(tmp_path: Path, capsys) -> None:
    root = tmp_path / "s"
    _write_scenario(root, {k: v for k, v in PASS_TABLE.items() if k != ("t10", 60)})
    assert SS.main(["summarise", "--out", str(root), "--expected-seeds", "4"]) == 3
    out = capsys.readouterr().out
    assert "NOT computed" in out and "t10 q60" in out
    assert (root / "summary_median.csv").is_file() and not (root / "premise_check.json").exists()
    assert SS.main(["summarise", "--out", str(tmp_path / "none")]) == 2


# ====================================================================== 7. CLI smoke (process pool)
def test_cli_smoke_with_a_process_pool_then_summarise(tmp_path: Path, capsys) -> None:
    out = tmp_path / "scr"
    argv = ["run", "--out", str(out), "--workers", "2", "--steps", "100", "--decay-steps", "20", "--checkpoints", "50",
            "--extended-steps", "200", "--extended-checkpoints", "100", "--seeds", "10501", "--qs", "50", "60",
            "--starts", "bb"]
    assert SS.main(argv) == 0
    log = capsys.readouterr().out
    names = sorted(p.name for p in (out / "cells").glob("*.json"))
    assert names == sorted([f"{a}_bb_q{q}_seed10501.json" for a in ACTOR_VARIANTS for q in (50, 60)]
                           + [f"t1_bb_q{q}_seed10501_ext.json" for q in (50, 60)])
    assert "8 cell(s), 2 worker(s)" in log and "8 ok, 0 failed" in log
    c = json.load(open(out / "cells" / "relu_bb_q60_seed10501.json"))
    assert [x["step"] for x in c["checkpoints"]] == [50, 100] and c["init"]["seed_material"] == [10501, 60, 0]
    pids = [json.load(open(out / "cells" / n))["pid"] for n in names]
    assert len(set(pids)) == len(names) and os.getpid() not in pids                  # one fresh process per cell
    e = json.load(open(out / "cells" / "t1_bb_q60_seed10501_ext.json"))
    assert [x["step"] for x in e["checkpoints"]] == [100, 200] and e["budget"]["const_steps"] == 180
    # the same cell run in this process gives the same numbers as in the spawned worker
    ref = SS.fit_cell(SS.CellSpec("relu", "bb", 60, 10501, 100, 20, (50, 100)))
    assert ref["checkpoints"] == c["checkpoints"] and ref["init"] == c["init"]
    man = json.load(open(out / "run_manifest.json"))
    assert man["workers"] == 2 and man["plan"]["n_cells"] == 8 and man["omp_num_threads"] == "1"
    assert SS.main(["summarise", "--out", str(out), "--premise-step", "100", "--expected-seeds", "1"]) == 0
    pc = json.load(open(out / "premise_check.json"))
    assert pc["step"] == 100 and pc["complete"] is True and pc["init_identical_across_actors"] is True
    assert pc["outcome"] in ("PASS", "STOP", "DROP relu", "DROP t10")
    ext = list(csv.DictReader(open(out / "summary_extended.csv")))
    assert {(x["q"], x["steps"]) for x in ext} == {("50", "100"), ("50", "200"), ("60", "100"), ("60", "200")}
