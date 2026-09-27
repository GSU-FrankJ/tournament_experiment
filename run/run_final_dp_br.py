#!/usr/bin/env python3
"""Backward-curriculum TEL-PPO runner for the final DP-BR verification protocol (T=2).

Protocol source: ``MultiStage/Discussion/090726 final DP-BR verification(1).md``
(merge notes + revised BR-reachable set) and the task notes in
``docs/tasks/final-dp-br-t2/``.

Pipeline
    phase A (<=400 updates): 512 stage-2 state-balanced exploring-start episodes per update
    phase B (<=600):         512 root episodes per update
    phase C (<=1000):        256 root + 256 stage-2 ES episodes per update (shuffled)
    per phase: 100-update warm-up (forced verifier call at local 100), stability
    checkpoints every 20 local updates (drift <= .01 and KL <= .01, 2 consecutive ->
    verifier call), timeout 100 updates since the last call, one call per update,
    a phase-end call if none happened at the cap. Eligible pass = phase BR criterion
    (A: max_{D2dev} Delta_2/DW <= .02, B: EXP_root/DW <= .02, C: dReach/DW <= .015)
    AND concentration <= .04 on the same theta. A/B advance after 3 consecutive
    eligible passes, C stops after 5; unused budget is not transferred.
    Main checkpoint = the C stopping point, else the C budget-exhaustion point.
    No best-checkpoint restore. Final evaluation re-runs the development and the
    final verifier tiers on that same checkpoint.

Launch in tmux (repo rule), single-threaded CPU:
    OMP_NUM_THREADS=1 python run/run_final_dp_br.py --q 50 --seed 47
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.ppo_curriculum import CurriculumPPO, PPOConfig  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler, gap_bin_index, stage_reward, step_gap  # noqa: E402
from utils.dp_br_verifier import (  # noqa: E402
    DEV_CONFIG,
    FINAL_CONFIG,
    DomainError,
    VerifierResult,
    beta_std_norm,
    concentration_stats,
    stage_result_arrays,
    verify,
)
from utils.theory_multistage import f_xi, g1_two_stage, g2_two_stage, validate_two_stage_params  # noqa: E402

PROTOCOL: Dict[str, object] = {
    "w_h": 6.0, "w_l": 2.0, "k": 1.0 / 3500.0, "e_min": 0.0, "e_max": 100.0,
    "episodes_per_update": 512,
    "phase_caps": {"A": 400, "B": 600, "C": 1000},
    "warmup": 100, "stability_every": 20, "drift_thr": 0.01, "kl_thr": 0.01,
    "stability_consecutive": 2, "verifier_timeout": 100,
    "k_phase": 3, "k_stop": 5,
    "phase_thr_over_dw": {"A": 0.02, "B": 0.02, "C": 0.015}, "conc_thr": 0.04,
    "snapshot_every": 20, "es_bin_width": 10.0, "phase_c_root": 256, "phase_c_es": 256,
    "final_thr_over_dw": 0.01, "refine_thr_over_dw": 0.002, "sensitivity_thr_over_dw": 0.03,
    "dense_conc_step": 0.05, "recovery_step": 0.5,
    "direct_rollout_episodes": 200000, "direct_rollout_reps": 3, "direct_rollout_seed_base": 9005000,
    "rng_namespaces": {"init": 0, "env_noise": 1, "learner_action": 2, "opponent_action": 3,
                       "starts_roles": 4, "minibatch": 5},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        root = str(Path(__file__).resolve().parent.parent)
        return subprocess.check_output(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def write_json(path: str, obj) -> None:
    """Write JSON with numpy coercion."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)


class Costs:
    """Wall-clock accumulators by category."""

    def __init__(self) -> None:
        self.t: Dict[str, float] = {}
        self.n: Dict[str, int] = {}

    def add(self, key: str, sec: float) -> None:
        self.t[key] = self.t.get(key, 0.0) + sec
        self.n[key] = self.n.get(key, 0) + 1

    def as_dict(self) -> Dict[str, object]:
        return {"seconds": dict(self.t), "count": dict(self.n)}


def make_policy_fns(agent: CurriculumPPO, spec: GameSpec, net=None
                    ) -> Tuple[Callable, Callable]:
    """Verifier-facing float64 mean-effort and (alpha, beta) functions.

    The observation is built exactly as in training (float64 gap -> float32 input),
    the network runs in float32, and the mean e_min + range * alpha/(alpha+beta)
    is formed in float64 from the float32 (alpha, beta).
    """
    def beta_fn(t: int, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        obs = spec.encode_obs(t, np.asarray(d, dtype=float))
        a, b = agent.beta_params(obs, net)
        return a.astype(float), b.astype(float)

    def mean_fn(t: int, d: np.ndarray) -> np.ndarray:
        a, b = beta_fn(t, d)
        return spec.e_min + spec.e_range * a / (a + b)

    return mean_fn, beta_fn


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def collect_batch(spec: GameSpec, agent: CurriculumPPO, t0: np.ndarray, d0: np.ndarray,
                  roles: np.ndarray, rng_env: np.random.Generator,
                  rng_learn: np.random.Generator, rng_opp: np.random.Generator,
                  gamma: float, lam: float, bin_width: float) -> Dict[str, object]:
    """Roll complete learner episodes against the frozen snapshot (vectorized).

    Args:
        spec: Game parameters.
        agent: Learner (actor/critic) + frozen opponent.
        t0: Start stage per episode.
        d0: Player-0 gap per episode at the start stage.
        roles: Learner physical role per episode (0 or 1).
        rng_env: Environment noise stream.
        rng_learn: Learner action stream.
        rng_opp: Opponent action stream.
        gamma: Discount.
        lam: GAE lambda.
        bin_width: Visitation bin width on D_t.

    Returns:
        Dict with flattened learner transitions and rollout statistics.
    """
    n = t0.size
    T = spec.T
    d_learner = np.where(roles == 0, d0, -d0).astype(float)
    rewards = np.zeros((n, T + 1))
    values = np.zeros((n, T + 1))
    per_stage: Dict[int, Dict[str, np.ndarray]] = {}
    effort_by_stage: Dict[int, float] = {}
    visitation: Dict[str, np.ndarray] = {}
    for t in range(1, T + 1):
        idx = np.nonzero(t0 <= t)[0]
        if idx.size == 0:
            continue
        dL = d_learner[idx]
        obs_L = spec.encode_obs(t, dL)
        obs_O = spec.encode_obs(t, -dL)
        a_L, b_L = agent.beta_params(obs_L)
        act_L = agent.sample_actions(a_L, b_L, rng_learn)
        a_O, b_O = agent.beta_params(obs_O, agent.opponent)
        act_O = agent.sample_actions(a_O, b_O, rng_opp)
        eps0 = rng_env.uniform(-spec.q, spec.q, size=idx.size)
        eps1 = rng_env.uniform(-spec.q, spec.q, size=idx.size)
        r = roles[idx]
        eps_L = np.where(r == 0, eps0, eps1)
        eps_O = np.where(r == 0, eps1, eps0)
        e_L = spec.effort_from_action(act_L)
        e_O = spec.effort_from_action(act_O)
        d_next = step_gap(spec, dL, e_L, e_O, eps_L, eps_O)
        rew = stage_reward(spec, t, e_L, d_next)
        logp = agent.log_prob(a_L, b_L, act_L)
        val = agent.value(obs_L)
        rewards[idx, t] = rew
        values[idx, t] = val
        per_stage[t] = {"idx": idx, "obs": obs_L, "act": act_L, "logp": logp, "val": val}
        effort_by_stage[t] = float(e_L.mean())
        if t >= 2:
            for src_name, src_mask in (("root_path", t0[idx] == 1), ("direct_es", t0[idx] == t)):
                if src_mask.any():
                    b, nb = gap_bin_index(spec, t, dL[src_mask], bin_width)
                    key = f"stage{t}_{src_name}"
                    visitation[key] = np.bincount(b, minlength=nb)
        d_learner[idx] = d_next
    # GAE with zero terminal bootstrap (V_{T+1} = 0)
    adv = np.zeros((n, T + 1))
    ret = np.zeros((n, T + 1))
    last = np.zeros(n)
    next_v = np.zeros(n)
    for t in range(T, 0, -1):
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
        ret[:, t] = adv[:, t] + values[:, t]
        next_v = values[:, t]
    states = np.concatenate([per_stage[t]["obs"] for t in sorted(per_stage)])
    actions = np.concatenate([per_stage[t]["act"] for t in sorted(per_stage)])
    logps = np.concatenate([per_stage[t]["logp"] for t in sorted(per_stage)])
    returns = np.concatenate([ret[per_stage[t]["idx"], t] for t in sorted(per_stage)])
    advs = np.concatenate([adv[per_stage[t]["idx"], t] for t in sorted(per_stage)])
    ep_return = np.array([ret[i, t0[i]] for i in range(n)])
    return {
        "states": states.astype(np.float32), "actions": actions.astype(np.float32),
        "logp": logps.astype(np.float32), "returns": returns.astype(np.float32),
        "advantages": advs.astype(np.float32),
        "n_episodes": int(n), "n_transitions": int(states.shape[0]),
        "mean_episode_return": float(ep_return.mean()),
        "mean_effort_by_stage": effort_by_stage,
        "visitation": visitation,
    }


# ---------------------------------------------------------------------------
# Evaluation pieces
# ---------------------------------------------------------------------------

def run_verifier(mean_fn, beta_fn, spec: GameSpec, cfg) -> Tuple[Optional[VerifierResult], Optional[str]]:
    """Run the verifier, returning (result, error_string)."""
    try:
        return verify(mean_fn, w_h=spec.w_h, w_l=spec.w_l, k=spec.k, q=spec.q, T=spec.T,
                      e_min=spec.e_min, e_max=spec.e_max, cfg=cfg, beta_fn=beta_fn), None
    except (DomainError, FloatingPointError, ValueError) as exc:  # numerical invalid call
        return None, f"{type(exc).__name__}: {exc}"


def dense_grid(half: float, step: float) -> np.ndarray:
    """Symmetric grid on [-half, half] with spacing <= step containing 0 and endpoints."""
    n_half = int(np.ceil(half / step - 1e-9))
    return np.linspace(-half, half, 2 * n_half + 1)


def gl_onpath(q: float, n_half: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n_half)
    nodes = np.concatenate([-q + q * x, q + q * x])
    weights = np.concatenate([q * w, q * w]) * f_xi(nodes, q)
    return nodes, weights


def recovery_metrics(mean_fn, spec: GameSpec, step: float) -> Dict[str, object]:
    """Closed-form recovery on the Delta d = step grid over D_2 (both roles)."""
    q, k = spec.q, spec.k
    g1 = g1_two_stage(q, spec.w_h, spec.w_l, k)
    e1 = float(mean_fn(1, np.array([0.0]))[0])
    D = dense_grid(spec.domain_half(2), step)
    e2_r1 = np.asarray(mean_fn(2, D), dtype=float)          # player 1 at physical gap D
    e2_r2 = np.asarray(mean_fn(2, -D), dtype=float)         # player 2 at the same physical gap
    g2 = g2_two_stage(D, q, spec.w_h, spec.w_l, k, spec.e_max)
    pos = np.abs(D) < 2.0 * q
    tail = ~pos
    zero = int(np.argmin(np.abs(D)))

    def _region(e: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        err = e[mask] - g2[mask]
        return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
                "max_abs": float(np.max(np.abs(err))), "n": int(mask.sum())}

    pooled_pos = np.concatenate([e2_r1[pos] - g2[pos], e2_r2[pos] - g2[pos]])
    tail_pooled = np.concatenate([e2_r1[tail], e2_r2[tail]])
    nodes, weights = gl_onpath(q)
    e2_path = np.asarray(mean_fn(2, nodes), dtype=float)
    g2_path = g2_two_stage(nodes, q, spec.w_h, spec.w_l, k, spec.e_max)
    e2_onpath = float(np.sum(weights * e2_path))
    onpath_mae = float(np.sum(weights * np.abs(e2_path - g2_path)))
    onpath_rmse = float(np.sqrt(np.sum(weights * (e2_path - g2_path) ** 2)))
    return {
        "grid_step": step, "n_grid": int(D.size), "g1": g1, "g2_at_0": float(g2[zero]),
        "stage1": {
            "role1_e1": e1, "role2_e1": e1, "signed_error": e1 - g1, "abs_error": abs(e1 - g1),
            "relative_pct": 100.0 * abs(e1 - g1) / g1,
            "max_role_abs_error": abs(e1 - g1), "max_role_relative_pct": 100.0 * abs(e1 - g1) / g1,
        },
        "stage2_positive_region": {
            "role1": _region(e2_r1, pos), "role2": _region(e2_r2, pos),
            "pooled": {"mae": float(np.mean(np.abs(pooled_pos))),
                       "rmse": float(np.sqrt(np.mean(pooled_pos ** 2))),
                       "max_abs": float(np.max(np.abs(pooled_pos))), "n": int(pooled_pos.size)},
        },
        "peak_d0": {
            "role1_e2": float(e2_r1[zero]), "role2_e2": float(e2_r2[zero]),
            "signed_error_role1": float(e2_r1[zero] - g2[zero]),
            "signed_error_role2": float(e2_r2[zero] - g2[zero]),
            "max_role_abs_error": float(max(abs(e2_r1[zero] - g2[zero]), abs(e2_r2[zero] - g2[zero]))),
        },
        "tail_region": {
            "n_nodes_pooled": int(tail_pooled.size),
            "role1_mean": float(e2_r1[tail].mean()), "role1_max": float(e2_r1[tail].max()),
            "role2_mean": float(e2_r2[tail].mean()), "role2_max": float(e2_r2[tail].max()),
            "pooled_mean": float(tail_pooled.mean()), "pooled_max": float(tail_pooled.max()),
        },
        "symmetry": {
            "max_abs_e2_d_minus_e2_negd": float(np.max(np.abs(e2_r1 - e2_r1[::-1]))),
            "mean_abs_e2_d_minus_e2_negd": float(np.mean(np.abs(e2_r1 - e2_r1[::-1]))),
        },
        "onpath": {"expected_e2": e2_onpath, "target_g1": g1, "abs_error": abs(e2_onpath - g1),
                   "mae": onpath_mae, "rmse": onpath_rmse},
        "arrays": {"d_grid": D, "e2_role1": e2_r1, "e2_role2": e2_r2, "g2": g2},
    }


def direct_rollout(mean_fn, spec: GameSpec, seed: int, n_ep: int, reps: int, base: int
                   ) -> Dict[str, object]:
    """Sampled self-play payoff of the frozen mean policy from the root (diagnostic)."""
    out = []
    for rep in range(reps):
        streams = {(p, t): np.random.default_rng(
            np.random.SeedSequence([base, int(spec.q), seed, rep, p, t]))
            for p in (0, 1) for t in range(1, spec.T + 1)}
        d = np.zeros(n_ep)
        pay0 = np.zeros(n_ep)
        pay1 = np.zeros(n_ep)
        for t in range(1, spec.T + 1):
            e0 = np.asarray(mean_fn(t, d), dtype=float)
            e1 = np.asarray(mean_fn(t, -d), dtype=float)
            eps0 = streams[(0, t)].uniform(-spec.q, spec.q, size=n_ep)
            eps1 = streams[(1, t)].uniform(-spec.q, spec.q, size=n_ep)
            d = step_gap(spec, d, e0, e1, eps0, eps1)
            pay0 -= spec.k * e0 ** 2
            pay1 -= spec.k * e1 ** 2
        pay0 += spec.terminal_reward(d)
        pay1 += spec.terminal_reward(-d)
        out.append({"rep": rep, "n_episodes": n_ep,
                    "payoff0_mean": float(pay0.mean()), "payoff0_se": float(pay0.std() / np.sqrt(n_ep)),
                    "payoff1_mean": float(pay1.mean()), "payoff1_se": float(pay1.std() / np.sqrt(n_ep)),
                    "win_rate0": float((d > 0).mean())})
    return {"reps": out, "seed_namespace": [base, int(spec.q), seed, "rep", "player", "stage"]}


def final_evaluation(agent: CurriculumPPO, spec: GameSpec, seed: int, costs: Costs
                     ) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """Two-tier verifier + concentration + recovery + direct rollout on the frozen checkpoint."""
    P = PROTOCOL
    dw = spec.dw
    mean_fn, beta_fn = make_policy_fns(agent, spec)
    arrays: Dict[str, np.ndarray] = {}
    out: Dict[str, object] = {}

    tiers = {}
    for cfg in (DEV_CONFIG, FINAL_CONFIG):
        t0 = time.perf_counter()
        res, err = run_verifier(mean_fn, beta_fn, spec, cfg)
        dt = time.perf_counter() - t0
        costs.add(f"final_eval_verifier_{cfg.name}", dt)
        if res is None:
            tiers[cfg.name] = {"valid": False, "error": err, "time_sec": dt}
        else:
            s = res.summary()
            s["time_sec"] = dt
            s["error"] = None
            tiers[cfg.name] = s
            arrays.update(stage_result_arrays(res, cfg.name))
            tiers[cfg.name + "_result"] = res
    out["tiers"] = {k: v for k, v in tiers.items() if not k.endswith("_result")}
    res_dev: Optional[VerifierResult] = tiers.get("development_result")
    res_fin: Optional[VerifierResult] = tiers.get("final_result")

    # --- pass flags ------------------------------------------------------------
    flags: Dict[str, object] = {}
    if res_fin is not None and res_dev is not None:
        d_dr = abs(res_fin.dreach - res_dev.dreach) / dw
        d_exp = abs(res_fin.exp_root - res_dev.exp_root) / dw
        d_dmax = abs(res_fin.delta_max_all - res_dev.delta_max_all) / dw
        d_dfull = abs(res_fin.dfull - res_dev.dfull) / dw
        flags = {
            "valid_final": bool(res_fin.valid), "valid_dev": bool(res_dev.valid),
            "dreach_final_over_dw": res_fin.dreach / dw,
            "main_threshold_over_dw": P["final_thr_over_dw"],
            "main_pass": bool(res_fin.valid and res_fin.dreach / dw <= P["final_thr_over_dw"]),
            "refine_dreach_diff_over_dw": d_dr, "refine_exp_diff_over_dw": d_exp,
            "refine_threshold_over_dw": P["refine_thr_over_dw"],
            "refine_dreach_pass": bool(d_dr <= P["refine_thr_over_dw"]),
            "refine_exp_pass": bool(d_exp <= P["refine_thr_over_dw"]),
            "delta_max_all_diff_over_dw": d_dmax, "dfull_diff_over_dw": d_dfull,
            "sensitivity_thr_over_dw": P["sensitivity_thr_over_dw"],
            "sensitivity_pass_0p03": bool(res_fin.valid and res_fin.dreach / dw <= P["sensitivity_thr_over_dw"]),
            "exp_le_dreach_final": bool(res_fin.exp_root <= res_fin.dreach + 1e-12),
        }
        flags["overall_pass"] = bool(flags["valid_final"] and flags["valid_dev"] and flags["main_pass"]
                                     and flags["refine_dreach_pass"] and flags["refine_exp_pass"])
        flags["numerically_unstable"] = bool(not (flags["refine_dreach_pass"] and flags["refine_exp_pass"]))
    else:
        flags = {"valid_final": res_fin is not None and res_fin.valid,
                 "valid_dev": res_dev is not None and res_dev.valid,
                 "main_pass": False, "overall_pass": False,
                 "note": "a verifier tier failed; see tiers[*].error"}
    out["pass_flags"] = flags

    # --- concentration diagnostics --------------------------------------------
    t0 = time.perf_counter()
    dev_grid2 = res_dev.stages[2].d_grid if res_dev is not None else dense_grid(spec.domain_half(2), DEV_CONFIG.state_step)
    dense2 = dense_grid(spec.domain_half(2), P["dense_conc_step"])
    conc = {
        "threshold_reference": P["conc_thr"],
        "dev_grid_root_plus_stage2": concentration_stats(beta_fn, {1: np.zeros(1), 2: dev_grid2}, spec.e_range),
        "all_dense_step": P["dense_conc_step"],
        "C_all": concentration_stats(beta_fn, {1: np.zeros(1), 2: dense2}, spec.e_range),
    }
    a2, b2 = beta_fn(2, dense2)
    arrays["dense_conc_d_grid"] = dense2
    arrays["dense_conc_std_norm"] = beta_std_norm(a2, b2)
    arrays["dense_conc_alpha"] = a2
    arrays["dense_conc_beta"] = b2
    if res_fin is not None:
        pts_mask = {t: s.d_grid[s.reach] for t, s in res_fin.stages.items()}
        pts_pmf = {t: s.d_grid[s.pmf > 0] for t, s in res_fin.stages.items()}
        conc["C_BR"] = concentration_stats(beta_fn, pts_mask, spec.e_range)
        conc["C_BR_pmf_support"] = concentration_stats(beta_fn, pts_pmf, spec.e_range)
    conc["C_all_le_reference"] = bool(conc["C_all"].get("valid") and conc["C_all"]["max_std_norm"] <= P["conc_thr"])
    if "C_BR" in conc:
        conc["C_BR_le_reference"] = bool(conc["C_BR"].get("valid") and conc["C_BR"]["max_std_norm"] <= P["conc_thr"])
    out["concentration"] = conc
    costs.add("final_eval_concentration", time.perf_counter() - t0)

    # --- recovery vs closed form ----------------------------------------------
    t0 = time.perf_counter()
    rec = recovery_metrics(mean_fn, spec, P["recovery_step"])
    for k_, v_ in rec.pop("arrays").items():
        arrays[f"recovery_{k_}"] = v_
    if res_fin is not None:
        from utils.theory_multistage import eq_utility_two_stage
        rec["payoff_loss_vs_closed_form"] = eq_utility_two_stage(spec.q, spec.w_h, spec.w_l, spec.k) - res_fin.v_mean_root
    out["recovery"] = rec
    costs.add("final_eval_recovery", time.perf_counter() - t0)

    # --- direct sampled rollout (diagnostic) -----------------------------------
    t0 = time.perf_counter()
    dr = direct_rollout(mean_fn, spec, seed, P["direct_rollout_episodes"], P["direct_rollout_reps"],
                        P["direct_rollout_seed_base"])
    if res_fin is not None:
        dr["v_mean_root_final_verifier"] = res_fin.v_mean_root
    out["direct_rollout"] = dr
    costs.add("final_eval_direct_rollout", time.perf_counter() - t0)
    return out, arrays


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Final DP-BR protocol runner (T=2 backward curriculum)")
    p.add_argument("--q", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--T", type=int, default=2)
    p.add_argument("--out-root", type=str, default="results/final_dp_br_T2")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--threads", type=int, default=1)
    # smoke-only overrides (never used for formal runs; recorded in config.json)
    p.add_argument("--phase-caps", type=str, default=None, help="smoke only: A,B,C caps")
    p.add_argument("--episodes", type=int, default=None, help="smoke only: episodes per update")
    p.add_argument("--warmup", type=int, default=None, help="smoke only")
    p.add_argument("--stability-every", type=int, default=None, help="smoke only")
    p.add_argument("--timeout", type=int, default=None, help="smoke only")
    p.add_argument("--direct-rollout-episodes", type=int, default=None, help="smoke only")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    P = dict(PROTOCOL)
    overrides = {}
    if args.phase_caps:
        caps = [int(x) for x in args.phase_caps.split(",")]
        P["phase_caps"] = {"A": caps[0], "B": caps[1], "C": caps[2]}
        overrides["phase_caps"] = P["phase_caps"]
    for name, key in (("episodes", "episodes_per_update"), ("warmup", "warmup"),
                      ("stability_every", "stability_every"), ("timeout", "verifier_timeout"),
                      ("direct_rollout_episodes", "direct_rollout_episodes")):
        v = getattr(args, name)
        if v is not None:
            P[key] = v
            overrides[key] = v
    if args.T != 2:
        raise SystemExit("this runner implements the T=2 curriculum (A/B/C) only")
    if P["episodes_per_update"] != P["phase_c_root"] + P["phase_c_es"]:
        P["phase_c_root"] = P["episodes_per_update"] // 2
        P["phase_c_es"] = P["episodes_per_update"] - P["phase_c_root"]
    PROTOCOL.update(P)

    spec = GameSpec(w_h=P["w_h"], w_l=P["w_l"], k=P["k"], q=float(args.q), T=args.T,
                    e_min=P["e_min"], e_max=P["e_max"])
    dw = spec.dw
    # closed-form validity screen (existing analytic check; not a new gate)
    val = validate_two_stage_params(q=spec.q, w_h=spec.w_h, w_l=spec.w_l, k=spec.k, e_bar=spec.e_max)
    if not val.ok:
        raise SystemExit(f"q={spec.q} fails closed-form validity: {val.messages}")

    tag = f"_{args.tag}" if args.tag else ""
    run_name = f"tel_q{spec.q:g}_s{args.seed}{tag}"
    out_dir = os.path.join(args.out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    status_path = os.path.join(out_dir, "status.json")
    t_wall0 = time.perf_counter()
    t_cpu0 = time.process_time()
    status = {"run": run_name, "state": "running", "pid": os.getpid(), "host": socket.gethostname(),
              "cmd": " ".join(sys.argv), "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "git_commit": _git_commit()}
    write_json(status_path, status)

    # RNG streams (seed, q, namespace)
    ns = P["rng_namespaces"]
    def _rng(name: str) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([args.seed, int(spec.q), ns[name]]))
    init_state = np.random.SeedSequence([args.seed, int(spec.q), ns["init"]]).generate_state(1)[0]
    torch_gen = torch.Generator().manual_seed(int(init_state))
    rng_env, rng_learn, rng_opp = _rng("env_noise"), _rng("learner_action"), _rng("opponent_action")
    rng_start, rng_mb = _rng("starts_roles"), _rng("minibatch")

    ppo_cfg = PPOConfig()
    agent = CurriculumPPO(ppo_cfg, torch_gen, rng_mb, device=args.device)
    sampler = StartSampler(spec, P["es_bin_width"])

    config = {
        "run": run_name, "q": spec.q, "seed": args.seed, "T": spec.T,
        "game": asdict(spec), "dw": dw, "B": spec.B, "domain_half_stage2": spec.domain_half(2),
        "ppo": asdict(ppo_cfg), "protocol": P, "smoke_overrides": overrides,
        "obs_encoding": "tau=(t-1)/(T-1); dtilde_1=0; dtilde_t=d/((t-1)B); float64 -> float32",
        "action_mapping": "e = e_min + a (e_max - e_min); a ~ Beta(mu c, (1-mu) c)",
        "action_sampling": "a drawn by numpy Generator.beta (float64), clipped to "
                           "[action_clamp, 1 - action_clamp] = [1e-6, 1-1e-6], stored as float32; "
                           "log-prob evaluated on the stored clipped value",
        "mean_extraction": "e_hat = e_min + (e_max - e_min) alpha/(alpha+beta) in float64 from float32 alpha,beta",
        "es_bins_stage2": sampler.n_bins(2),
        "verifier": {"development": asdict(DEV_CONFIG), "final": asdict(FINAL_CONFIG)},
        "dtypes": {"network": "float32", "env_gap_reward": "float64", "verifier": "float64"},
        "device": args.device, "torch_threads": args.threads,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "git_commit": status["git_commit"], "cmd": status["cmd"],
    }
    write_json(os.path.join(out_dir, "config.json"), config)
    print(f"[run] {run_name} q={spec.q} seed={args.seed} out={out_dir} device={args.device}", flush=True)
    print(f"[cfg] caps={P['phase_caps']} episodes/update={P['episodes_per_update']} "
          f"ES bins={sampler.n_bins(2)} dev grid stage2 spacing={DEV_CONFIG.state_step}", flush=True)

    costs = Costs()
    history: List[Dict] = []
    stability_log: List[Dict] = []
    verifier_log: List[Dict] = []
    curriculum_log: List[Dict] = []
    snapshot_log: List[Dict] = [{"update": 0, "reason": "init"}]
    visitation_by_phase: Dict[str, Dict[str, np.ndarray]] = {}
    mean_fn, beta_fn = make_policy_fns(agent, spec)
    dev_grid2 = dense_grid(spec.domain_half(2), DEV_CONFIG.state_step)

    total_episodes = 0
    total_transitions = 0
    global_u = 0
    stop_record: Optional[Dict] = None
    phases = [("A", [spec.T]), ("B", list(range(1, spec.T + 1))), ("C", list(range(1, spec.T + 1)))]

    def stability_points(phase: str) -> Dict[int, np.ndarray]:
        return {2: dev_grid2} if phase == "A" else {1: np.zeros(1), 2: dev_grid2}

    def phase_criterion(phase: str, res: VerifierResult) -> Tuple[str, float]:
        if phase == "A":
            return "max_D2dev_Delta2_over_dw", res.full_delta_max[spec.T] / dw
        if phase == "B":
            return "exp_root_over_dw", res.exp_root / dw
        return "dreach_over_dw", res.dreach / dw

    try:
        for phase, active in phases:
            cap = P["phase_caps"][phase]
            entry = global_u
            local = 0
            eligible = 0
            stab_consec = 0
            prev_stab: Optional[Dict[int, np.ndarray]] = None
            last_call: Optional[int] = None
            agent.refresh_snapshot()
            snapshot_log.append({"update": global_u, "reason": f"phase_{phase}_entry"})
            vis_phase: Dict[str, np.ndarray] = {}
            exit_reason = "budget_exhausted"
            print(f"[phase {phase}] entry at global update {global_u}, cap {cap}, active stages {active}", flush=True)
            while local < cap:
                local += 1
                global_u += 1
                n_ep = P["episodes_per_update"]
                # ---- starts and roles
                if phase == "A":
                    t0 = np.full(n_ep, spec.T)
                    d0 = sampler.balanced(spec.T, n_ep, rng_start)
                elif phase == "B":
                    t0 = np.ones(n_ep, dtype=int)
                    d0 = np.zeros(n_ep)
                else:
                    nr, ne = P["phase_c_root"], P["phase_c_es"]
                    t0 = np.concatenate([np.ones(nr, dtype=int), np.full(ne, spec.T)])
                    d0 = np.concatenate([np.zeros(nr), sampler.balanced(spec.T, ne, rng_start)])
                    perm = rng_start.permutation(n_ep)
                    t0, d0 = t0[perm], d0[perm]
                roles = rng_start.integers(0, 2, size=n_ep)
                # ---- rollout + update
                t_r = time.perf_counter()
                batch = collect_batch(spec, agent, t0, d0, roles, rng_env, rng_learn, rng_opp,
                                      ppo_cfg.gamma, ppo_cfg.gae_lambda, P["es_bin_width"])
                costs.add("train_rollout", time.perf_counter() - t_r)
                t_u = time.perf_counter()
                diag = agent.update(batch["states"], batch["actions"], batch["logp"],
                                    batch["returns"], batch["advantages"])
                costs.add("train_update", time.perf_counter() - t_u)
                total_episodes += batch["n_episodes"]
                total_transitions += batch["n_transitions"]
                for k_, v_ in batch["visitation"].items():
                    vis_phase[k_] = vis_phase.get(k_, 0) + v_
                snap = False
                if global_u % P["snapshot_every"] == 0:
                    agent.refresh_snapshot()
                    snapshot_log.append({"update": global_u, "reason": "every_20"})
                    snap = True
                rec = {"update": global_u, "phase": phase, "local": local,
                       "n_episodes": batch["n_episodes"], "n_transitions": batch["n_transitions"],
                       "mean_episode_return": batch["mean_episode_return"],
                       "mean_effort_by_stage": {str(t): v for t, v in batch["mean_effort_by_stage"].items()},
                       "snapshot_refreshed": snap, **{k_: v_ for k_, v_ in diag.items() if k_ != "kl_epochs"},
                       "kl_epochs": diag["kl_epochs"]}
                history.append(rec)
                # ---- stability checkpoint
                stab_now = False
                if local % P["stability_every"] == 0:
                    t_s = time.perf_counter()
                    pts = stability_points(phase)
                    cur = {t: np.asarray(mean_fn(t, d), dtype=float) for t, d in pts.items()}
                    drift = None
                    if prev_stab is not None:
                        drift = max(float(np.max(np.abs(cur[t] - prev_stab[t]))) / spec.e_range for t in cur)
                    kl = diag["kl_final_epoch"]
                    stable = drift is not None and drift <= P["drift_thr"] and kl <= P["kl_thr"]
                    stab_consec = stab_consec + 1 if stable else 0
                    conc_s = concentration_stats(beta_fn, pts, spec.e_range)
                    stability_log.append({"update": global_u, "phase": phase, "local": local,
                                          "drift": drift, "kl": kl, "stable": bool(stable),
                                          "consecutive": stab_consec,
                                          "max_std_norm": conc_s.get("max_std_norm"),
                                          "e_hat": {str(t): cur[t] for t in cur}})
                    prev_stab = cur
                    stab_now = True
                    costs.add("stability_check", time.perf_counter() - t_s)
                # ---- verifier trigger
                reason = None
                if local == P["warmup"]:
                    reason = "warmup_forced"
                elif local > P["warmup"]:
                    if stab_now and stab_consec >= P["stability_consecutive"]:
                        reason = "stability"
                    elif last_call is None or local - last_call >= P["verifier_timeout"]:
                        reason = "timeout"
                if reason is None and local == cap and last_call != local:
                    reason = "phase_end"
                if reason is not None:
                    t_v = time.perf_counter()
                    res, err = run_verifier(mean_fn, beta_fn, spec, DEV_CONFIG)
                    dt = time.perf_counter() - t_v
                    costs.add("dev_verifier", dt)
                    pts = stability_points(phase)
                    conc_s = concentration_stats(beta_fn, pts, spec.e_range)
                    conc_pass = bool(conc_s.get("valid") and conc_s["max_std_norm"] <= P["conc_thr"])
                    if res is None:
                        valid, br_pass, crit_name, crit_val, summ = False, False, None, None, None
                    else:
                        valid = bool(res.valid)
                        crit_name, crit_val = phase_criterion(phase, res)
                        br_pass = bool(valid and crit_val <= P["phase_thr_over_dw"][phase])
                        summ = res.summary()
                    elig = bool(valid and br_pass and conc_pass)
                    eligible = eligible + 1 if elig else 0
                    stab_consec = 0
                    last_call = local
                    entry_log = {"update": global_u, "phase": phase, "local": local, "reason": reason,
                                 "active_stages": active, "valid": valid, "error": err,
                                 "criterion": crit_name, "criterion_value_over_dw": crit_val,
                                 "criterion_threshold_over_dw": P["phase_thr_over_dw"][phase],
                                 "br_pass": br_pass, "concentration": conc_s, "conc_pass": conc_pass,
                                 "eligible": elig, "consecutive_eligible": eligible, "time_sec": dt,
                                 "summary": summ,
                                 "e_hat_dev": {str(t): np.asarray(mean_fn(t, d), dtype=float) for t, d in pts.items()},
                                 "std_norm_dev": {str(t): beta_std_norm(*beta_fn(t, d)) for t, d in pts.items()},
                                 "visitation_cumulative_phase": {k_: np.asarray(v_) for k_, v_ in vis_phase.items()}}
                    verifier_log.append(entry_log)
                    print(f"[u{global_u:>5} {phase}{local:>4}] verifier({reason}) valid={valid} "
                          f"{crit_name}={crit_val if crit_val is None else round(crit_val, 5)} "
                          f"EXP/DW={summ['exp_root_over_dw']:.5f} dReach/DW={summ['dreach_over_dw']:.5f} "
                          f"dFull/DW={summ['dfull_over_dw']:.4f} C={conc_s.get('max_std_norm', float('nan')):.4f} "
                          f"elig={elig} consec={eligible} | kl={diag['kl_final_epoch']:.4f} "
                          f"e1(0)={float(mean_fn(1, np.zeros(1))[0]):.2f} e2(0)={float(mean_fn(2, np.zeros(1))[0]):.2f}"
                          if summ is not None else
                          f"[u{global_u:>5} {phase}{local:>4}] verifier({reason}) INVALID: {err}", flush=True)
                    if phase != "C" and eligible >= P["k_phase"]:
                        exit_reason = "verifier_passed"
                        break
                    if phase == "C" and eligible >= P["k_stop"]:
                        exit_reason = "k_stop_passes"
                        break
                elif local % 50 == 0:
                    print(f"[u{global_u:>5} {phase}{local:>4}] ret={batch['mean_episode_return']:.4f} "
                          f"kl={diag['kl_final_epoch']:.4f} pl={diag['policy_loss']:.4f} vl={diag['value_loss']:.4f} "
                          f"ent={diag['entropy_post_update']:.3f} "
                          f"e1(0)={float(mean_fn(1, np.zeros(1))[0]):.2f} e2(0)={float(mean_fn(2, np.zeros(1))[0]):.2f}",
                          flush=True)
            visitation_by_phase[phase] = vis_phase
            curriculum_log.append({"phase": phase, "entry_update": entry, "exit_update": global_u,
                                   "local_updates": local, "cap": cap, "exit_reason": exit_reason,
                                   "active_stages": active, "consecutive_eligible_at_exit": eligible,
                                   "n_verifier_calls": sum(1 for v in verifier_log if v["phase"] == phase)})
            print(f"[phase {phase}] exit at global update {global_u} after {local} local updates: {exit_reason}", flush=True)
            if phase == "C":
                stop_record = {"stop_update": global_u, "reason": exit_reason,
                               "phase_C_local_at_stop": local, "total_episodes": total_episodes,
                               "total_transitions": total_transitions,
                               "n_verifier_calls": len(verifier_log),
                               "development_stopping_criterion_satisfied": exit_reason == "k_stop_passes"}
        train_wall = time.perf_counter() - t_wall0

        # ---- freeze + persist the main checkpoint BEFORE evaluation
        ckpt_path = os.path.join(out_dir, "checkpoint.pt")
        agent.save(ckpt_path)
        agent.export_weights_npz(os.path.join(out_dir, "checkpoint_weights.npz"))

        # ---- final evaluation on the frozen checkpoint
        t_f = time.perf_counter()
        final, arrays = final_evaluation(agent, spec, args.seed, costs)
        costs.add("final_eval_total", time.perf_counter() - t_f)
        for ph, vis in visitation_by_phase.items():
            for k_, v_ in vis.items():
                arrays[f"visitation_{ph}_{k_}"] = np.asarray(v_)
        np.savez(os.path.join(out_dir, "arrays.npz"), **arrays)

        wall = time.perf_counter() - t_wall0
        cpu = time.process_time() - t_cpu0
        cost_summary = costs.as_dict()
        cost_summary["training_wall_sec"] = train_wall
        cost_summary["training_sec"] = costs.t.get("train_rollout", 0.0) + costs.t.get("train_update", 0.0)
        cost_summary["stability_sec"] = costs.t.get("stability_check", 0.0)
        cost_summary["dev_verifier_sec"] = costs.t.get("dev_verifier", 0.0)
        cost_summary["dev_verifier_calls"] = costs.n.get("dev_verifier", 0)
        cost_summary["final_eval_sec"] = costs.t.get("final_eval_total", 0.0)
        cost_summary["total_wall_sec"] = wall
        cost_summary["total_process_cpu_sec"] = cpu
        cost_summary["total_episodes"] = total_episodes
        cost_summary["total_transitions"] = total_transitions
        cost_summary["total_updates"] = global_u
        cost_summary["minibatch_steps"] = int(sum(h["n_minibatch_steps"] for h in history))

        final["stopping_record"] = stop_record
        final["curriculum"] = curriculum_log
        final["costs"] = cost_summary
        final["checkpoint"] = {"path": ckpt_path, "weights_npz": os.path.join(out_dir, "checkpoint_weights.npz"),
                               "selection_rule": "C-phase stopping point (5 consecutive eligible passes) "
                                                 "or C budget exhaustion; no best-checkpoint restore",
                               "stop_update": stop_record["stop_update"] if stop_record else None}
        final["config_ref"] = os.path.join(out_dir, "config.json")
        final["run"] = run_name
        write_json(os.path.join(out_dir, "final_eval.json"), final)
        write_json(os.path.join(out_dir, "train_history.json"),
                   {"run": run_name, "history": history, "stability": stability_log,
                    "verifier_calls": verifier_log, "curriculum": curriculum_log,
                    "snapshots": snapshot_log, "stopping_record": stop_record})
        status.update({"state": "done", "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "stop_update": stop_record["stop_update"] if stop_record else None,
                       "stop_reason": stop_record["reason"] if stop_record else None,
                       "overall_pass": final["pass_flags"].get("overall_pass"),
                       "dreach_final_over_dw": final["pass_flags"].get("dreach_final_over_dw"),
                       "total_wall_sec": wall})
        write_json(status_path, status)
        fl = final["pass_flags"]
        tf = final["tiers"]["final"]
        print(f"[done] stop@u{stop_record['stop_update']} ({stop_record['reason']}) | final: valid={tf['valid']} "
              f"EXP/DW={tf.get('exp_root_over_dw', float('nan')):.5f} dReach/DW={tf.get('dreach_over_dw', float('nan')):.5f} "
              f"dFull/DW={tf.get('dfull_over_dw', float('nan')):.4f} | refine diffs dReach={fl.get('refine_dreach_diff_over_dw', float('nan')):.5f} "
              f"EXP={fl.get('refine_exp_diff_over_dw', float('nan')):.5f} | main_pass={fl.get('main_pass')} overall={fl.get('overall_pass')} "
              f"| C_all={final['concentration']['C_all'].get('max_std_norm', float('nan')):.4f} "
              f"| e1(0)={final['recovery']['stage1']['role1_e1']:.3f} (g1={final['recovery']['g1']:.3f}) "
              f"e2(0)={final['recovery']['peak_d0']['role1_e2']:.3f} (g2={final['recovery']['g2_at_0']:.3f}) "
              f"| wall={wall:.0f}s train={train_wall:.0f}s dev={cost_summary['dev_verifier_sec']:.0f}s "
              f"final={cost_summary['final_eval_sec']:.0f}s", flush=True)
        return 0
    except Exception:
        tb = traceback.format_exc()
        status.update({"state": "failed", "end_time": time.strftime("%Y-%m-%d %H:%M:%S"), "traceback": tb,
                       "updates_completed": global_u})
        write_json(status_path, status)
        try:
            write_json(os.path.join(out_dir, "train_history_partial.json"),
                       {"run": run_name, "history": history, "stability": stability_log,
                        "verifier_calls": verifier_log, "curriculum": curriculum_log})
        except Exception:
            pass
        print(tb, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
