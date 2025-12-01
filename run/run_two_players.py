#!/usr/bin/env python3
"""
One-Stage Two-Player Experiment (spec-compliant)

Writes standardized CSV and figure overlays. For one-stage, we map the single
stage to the CSV's stage-2 fields (stage-1 fields set to 0).
"""

import sys
import os
import argparse
import math
import datetime
import csv
from contextlib import contextmanager
from typing import Dict, List, Optional
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.one_stage_two_players import config as base_config
from utils.theory import e_star_two_players, clip_stage2
from utils.eval import build_csv_row
from utils.plot import plot_effort_curve
from utils.logger import save_standardized_result
from envs.two_players_env import TwoPlayersEnv
from agents.ppo_two_players_clean import PPOTwoPlayersBandit, PPOConfig
from agents.mc_fd_crn_solver import MCFDConfig, gradient_ascent_dynamics


MCFD_FIELDNAMES = [
    "sigma",
    "delta",
    "eta",
    "num_samples",
    "final_effort",
    "mcfd_iterations",
    "mcfd_tol",
    "mcfd_effort_min",
    "mcfd_effort_max",
    "seed",
]


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextmanager
def _tee_console_to_file(log_path: str):
    """Mirror stdout/stderr to a log file while preserving console output."""
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = _TeeStream(old_stdout, log_file)
        sys.stderr = _TeeStream(old_stderr, log_file)
        try:
            yield log_path
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def _build_log_path(args: argparse.Namespace) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    method_tag = getattr(args, "method", "run")
    
    # For MC-FD, use sigma parameters instead of q
    if method_tag == "mcfd":
        sigma1 = getattr(args, "mcfd_sigma1", 20.0)
        sigma2 = getattr(args, "mcfd_sigma2", 20.0)
        q_tag = f"σ{sigma1:.0f}_{sigma2:.0f}"
    else:
        # For gradient/PPO methods, use q parameter
        q_val = getattr(args, "q", None)
        if q_val is None:
            q_tag = "q_all"
        else:
            q_clean = f"{q_val:g}".replace("-", "neg").replace(".", "p")
            q_tag = f"q{q_clean}"
    
    episodes_val = getattr(args, "episodes", None)
    episodes_tag = ""
    if episodes_val is not None:
        episodes_tag = f"_ep{int(episodes_val)}"
    seed_val = getattr(args, "seed", None)
    seed_tag = f"_seed{int(seed_val)}" if seed_val is not None else ""
    filename = f"one_stage_two_players_{method_tag}_{q_tag}{episodes_tag}{seed_tag}_{timestamp}.log"
    return os.path.join("results", "logs", filename)


def _save_mcfd_result(row: Dict[str, float], csv_path: str) -> None:
    """Persist MC-FD rows using the requested minimal layout."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=MCFD_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _clip_effort(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float(np.clip(value, lo, hi))


def _batch_payoffs_uniform(env: TwoPlayersEnv, e1: float, e2: float, eps1: np.ndarray, eps2: np.ndarray, tie_breaks: np.ndarray) -> tuple[float, float]:
    """Vectorized payoff batch using provided Uniform(-q, q) noises and tie-breaks."""
    y1 = e1 + eps1
    y2 = e2 + eps2
    winners = np.where(y1 > y2, 0, np.where(y2 > y1, 1, tie_breaks))
    payoff1 = np.where(winners == 0, env.w_h, env.w_l)
    payoff2 = np.where(winners == 0, env.w_l, env.w_h)
    u1 = payoff1 - env.k * (e1 ** 2)
    u2 = payoff2 - env.k * (e2 ** 2)
    return float(u1.mean()), float(u2.mean())


def _stochastic_fd_gradients(
    env: TwoPlayersEnv,
    e1: float,
    e2: float,
    delta: float,
    num_samples: int,
) -> tuple[float, float]:
    """Central-difference gradients for each player using uniform noise samples."""
    eps1, eps2, tie_breaks = env.draw_noise_batch(num_samples)

    e1_plus = _clip_effort(e1 + delta, (env.effort_low, env.effort_high))
    e1_minus = _clip_effort(e1 - delta, (env.effort_low, env.effort_high))
    e2_plus = _clip_effort(e2 + delta, (env.effort_low, env.effort_high))
    e2_minus = _clip_effort(e2 - delta, (env.effort_low, env.effort_high))

    u1_plus, _ = _batch_payoffs_uniform(env, e1_plus, e2, eps1, eps2, tie_breaks)
    u1_minus, _ = _batch_payoffs_uniform(env, e1_minus, e2, eps1, eps2, tie_breaks)
    _, u2_plus = _batch_payoffs_uniform(env, e1, e2_plus, eps1, eps2, tie_breaks)
    _, u2_minus = _batch_payoffs_uniform(env, e1, e2_minus, eps1, eps2, tie_breaks)

    g1 = (u1_plus - u1_minus) / (2.0 * delta)
    g2 = (u2_plus - u2_minus) / (2.0 * delta)
    return float(g1), float(g2)


def gradient_descent_two_players(
    cfg: Dict,
    *,
    lr: float = 0.1,
    steps: int = 2000,
    eps: float = 0.1,
    tol: float = 1e-4,
    num_samples: int = 64,
    init_perturb: float = 1.0,
    log: bool = True,
) -> tuple[tuple[float, float], Dict[str, float]]:
    """Two-player gradient ascent with uniform noise and distinct starts."""
    effort_bounds = tuple(cfg["effort_bounds_stage2"])
    env = TwoPlayersEnv(
        w_h=cfg["w_h"],
        w_l=cfg["w_l"],
        k=cfg["k"],
        q=cfg["q"],
        effort_bounds=effort_bounds,
        seed=cfg.get("seed", 42),
    )
    if eps <= 0:
        raise ValueError("grad_eps must be positive for finite differences")
    num_samples = max(1, int(num_samples))
    lo, hi = effort_bounds
    e_theory = float(e_star_two_players(cfg["q"], cfg["w_h"], cfg["w_l"], cfg["k"]))
    # Start near theory but enforce e1 != e2 to avoid trivial symmetry.
    half_perturb = max(init_perturb * 0.5, 1e-6)
    e1 = _clip_effort(e_theory - half_perturb, effort_bounds)
    e2 = _clip_effort(e_theory + half_perturb, effort_bounds)
    if abs(e1 - e2) < 1e-8:
        jitter = max(half_perturb, 0.01 * (hi - lo))
        e2 = _clip_effort(e1 + jitter, effort_bounds)
        if abs(e1 - e2) < 1e-8:
            e1 = _clip_effort(e1 - jitter, effort_bounds)

    history = {
        "init_e1": e1,
        "init_e2": e2,
        "final_grad": 0.0,
        "iterations": 0.0,
    }

    for step in range(1, steps + 1):
        g1, g2 = _stochastic_fd_gradients(env, e1, e2, delta=eps, num_samples=num_samples)
        e1_new = _clip_effort(e1 + lr * g1, effort_bounds)
        e2_new = _clip_effort(e2 + lr * g2, effort_bounds)

        delta_e1 = abs(e1_new - e1)
        delta_e2 = abs(e2_new - e2)
        grad_norm = max(abs(g1), abs(g2))

        e1, e2 = e1_new, e2_new
        history["iterations"] = float(step)
        history["final_grad"] = float(grad_norm)
        history["final_grad_pair"] = (float(g1), float(g2))
        if log and (step == 1 or step % 250 == 0 or step == steps):
            print(
                f"[gradient-2p] step={step:05d} e1={e1:.6f} e2={e2:.6f} "
                f"grad=({g1:.6f},{g2:.6f}) delta=({delta_e1:.3e},{delta_e2:.3e})"
            )
        if grad_norm < tol or max(delta_e1, delta_e2) < tol:
            if log:
                print(f"[gradient-2p] converged at step={step} with grad_norm={grad_norm:.3e}")
            break

    history["final_e1"] = e1
    history["final_e2"] = e2
    return (e1, e2), history


def run_gradient(
    cfg: Dict,
    *,
    lr: float = 0.1,
    steps: int = 2000,
    grad_eps: float = 0.1,
    tol: float = 1e-4,
    num_samples: int = 64,
    init_perturb: float = 1.0,
    log: bool = True,
) -> Dict:
    w_h, w_l, k, q = cfg["w_h"], cfg["w_l"], cfg["k"], cfg["q"]
    theoretical_e = clip_stage2(e_star_two_players(q, w_h, w_l, k), tuple(cfg["effort_bounds_stage2"]))
    (e1, e2), meta = gradient_descent_two_players(
        cfg,
        lr=lr,
        steps=steps,
        eps=grad_eps,
        tol=tol,
        num_samples=num_samples,
        init_perturb=init_perturb,
        log=log,
    )
    final_e = 0.5 * (e1 + e2)
    if log:
        gap_sym = abs(e1 - e2)
        grad_pair = meta.get("final_grad_pair", (0.0, 0.0))
        print(
            f"[gradient-2p] final e1={e1:.6f} e2={e2:.6f} avg={final_e:.6f} "
            f"grad=({grad_pair[0]:.6f},{grad_pair[1]:.6f}) gap_sym={gap_sym:.3e}"
        )
        print(f"[gradient-2p] meta: iterations={meta['iterations']:.0f} final_grad={meta['final_grad']:.6f}")

    row = build_csv_row(
        stage1_weight=cfg["stage1_weight"],
        stage2_weight=cfg["stage2_weight"],
        k1=cfg["k1"],
        k2=cfg["k2"],
        information_revelation=cfg.get("information_revelation", "none"),
        theoretical_stage1_effort=0.0,
        theoretical_stage2_effort=theoretical_e,
        model_training="gradient",
        final_stage1_effort=0.0,
        final_stage2_effort=final_e,
        episodes=0,
    )
    row["stage2_gap_unweighted"] = abs(float(row["final_stage2_effort"]) - float(row["theoretical_stage2_effort"]))
    row["gradient_iterations"] = meta["iterations"]
    row["gradient_final_grad"] = meta["final_grad"]
    row["gradient_mode"] = "stochastic_uniform"
    row["final_e1"] = e1
    row["final_e2"] = e2
    row["symmetry_gap"] = abs(e1 - e2)
    return row


def run_mcfd(cfg: Dict, args: argparse.Namespace) -> Dict:
    """Execute Monte Carlo finite-difference solver with Gaussian noise (σ₁, σ₂).
    
    Uses simulation-based gradient estimation with Common Random Numbers.
    No q parameter is used - noise is controlled by sigma1 and sigma2.
    """

    bounds = tuple(cfg.get("effort_bounds_stage2", (0.0, 200.0)))
    e_min = float(args.mcfd_effort_min) if args.mcfd_effort_min is not None else bounds[0]
    e_max = float(args.mcfd_effort_max) if args.mcfd_effort_max is not None else bounds[1]
    if e_min >= e_max:
        raise ValueError(f"mcfd effort bounds invalid: [{e_min}, {e_max}]")

    mcfd_cfg = MCFDConfig(
        w_h=float(cfg["w_h"]),
        w_l=float(cfg["w_l"]),
        k=float(cfg["k"]),
        sigma1=float(args.mcfd_sigma1),
        sigma2=float(args.mcfd_sigma2),
        delta=float(args.mcfd_delta),
        eta=float(args.mcfd_eta),
        num_samples=int(args.mcfd_num_samples),
        e_min=e_min,
        e_max=e_max,
        max_iters=int(args.mcfd_max_iters),
        tol=float(args.mcfd_tol),
        seed=int(args.mcfd_seed) if args.mcfd_seed is not None else cfg.get("seed"),
    )

    sim_results = gradient_ascent_dynamics(mcfd_cfg)
    e1_history = sim_results["effort_player1"]
    e2_history = sim_results["effort_player2"]
    e1_final = float(e1_history[-1])
    e2_final = float(e2_history[-1])
    avg_final_effort = 0.5 * (e1_final + e2_final)
    iterations = max(0, len(e1_history) - 1)

    mcfd_row = {
        "sigma": mcfd_cfg.sigma1,
        "delta": mcfd_cfg.delta,
        "eta": mcfd_cfg.eta,
        "num_samples": mcfd_cfg.num_samples,
        "final_effort": avg_final_effort,
        "mcfd_iterations": iterations,
        "mcfd_tol": mcfd_cfg.tol,
        "mcfd_effort_min": e_min,
        "mcfd_effort_max": e_max,
        "seed": mcfd_cfg.seed if mcfd_cfg.seed is not None else "",
    }

    return mcfd_row


def run_ppo(
    cfg: Dict,
    episodes: Optional[int] = None,
    train_qs: Optional[List[float]] = None,
    eval_qs: Optional[List[float]] = None,
    *,
    eval_symmetric: bool = True,
    eval_vs_opponent: bool = False,
    eval_vs_history: bool = False,
) -> List[Dict]:
    """Train PPO via self-play with conditioning on (q, k, w_gap).

    - Trains over ``train_qs`` (defaults to cfg["q_list" ]).
    - Returns a list of CSV rows, one per q in ``eval_qs`` (defaults to train_qs).
    """
    if episodes is None:
        episodes = int(cfg.get("episodes", 1_800_000))
    else:
        episodes = int(episodes)

    w_h, w_l, k = cfg["w_h"], cfg["w_l"], cfg["k"]
    effort_bounds = tuple(cfg["effort_bounds_stage2"])  # (0, 200)
    # Respect CLI-provided training set; default to config q_list
    train_qs = list(train_qs if train_qs is not None else cfg["q_list"])
    eval_qs = list(eval_qs if eval_qs is not None else train_qs)

    # PPO agent with 3-dim state: [q, k, w_gap]
    ppo_cfg = PPOConfig(
        steps_per_update=int(cfg.get("steps_per_update", 4096)),
        epochs=int(cfg.get("update_epochs", 6)),
        minibatch_size=int(cfg.get("minibatch_size", 1024)),
        state_dim=3,
        hidden=128,
        opponent_mode=cfg.get("opponent_mode", "periodic"),
        opponent_sync_interval=int(cfg.get("opponent_sync_interval", 2)),
        opponent_ema_tau=float(cfg.get("opponent_ema_tau", 0.20)),
        opponent_snapshot_keep=int(cfg.get("opponent_snapshot_keep", 10)),
        opponent_history_sample_p=float(cfg.get("opponent_history_sample_p", 0.3)),
        entropy_coef=float(cfg.get("entropy_coef_start", 0.02)),
        lr=float(cfg.get("lr_start", 3e-4)),
        clip_eps=float(cfg.get("clip_range_start", 0.30)),
    )
    agent = PPOTwoPlayersBandit(effort_bounds=effort_bounds, cfg=ppo_cfg)
    agent.cfg.entropy_coef = float(cfg.get("entropy_coef_start", agent.cfg.entropy_coef))
    agent.cfg.clip_eps = float(cfg.get("clip_range_start", agent.cfg.clip_eps))
    for g in agent.opt.param_groups:
        g["lr"] = float(cfg.get("lr_start", ppo_cfg.lr))

    history: List[float] = []
    total_steps_target = int(episodes)
    max_updates_cfg = int(cfg.get("max_updates", 0) or 0)
    if max_updates_cfg > 0:
        capped_steps = max_updates_cfg * ppo_cfg.steps_per_update
        if total_steps_target > capped_steps:
            total_steps_target = capped_steps
            print(f"[config] max_updates={max_updates_cfg} -> total_steps capped at {total_steps_target}", flush=True)
    steps_done = 0
    rng = np.random.default_rng(cfg.get("seed", 42))
    # Entropy / LR schedules: hold high values until ~2/3 progress, then anneal
    entropy_start = float(cfg.get("entropy_coef_start", agent.cfg.entropy_coef))
    entropy_hold = float(cfg.get("entropy_coef_hold", entropy_start))
    entropy_final = float(cfg.get("entropy_coef_end", 0.005))
    update_idx = 0
    # Late-phase settings
    total_updates = (total_steps_target + ppo_cfg.steps_per_update - 1) // ppo_cfg.steps_per_update
    hold_fraction = float(cfg.get("entropy_hold_fraction", 2.0 / 3.0))
    hold_fraction = max(0.0, min(1.0, hold_fraction))
    hold_updates = max(1, int(math.ceil(total_updates * hold_fraction)))
    tail_updates = max(1, total_updates - hold_updates)
    # Learning rate schedule: hold at starting value, then anneal to final value
    lr_hold = float(cfg.get("lr_start", agent.cfg.lr))
    lr_final = float(cfg.get("lr_end", 2e-4))
    # Clip schedule parameters
    clip_max = float(cfg.get("clip_range_start", agent.cfg.clip_eps))
    clip_min = float(cfg.get("clip_range_end", 0.15))
    # Self-play lag schedule: short warmup then fade
    lag_warmup_updates = max(0, int(cfg.get("lag_warmup_updates", 10)))
    lag_fade_cfg = cfg.get("lag_fade_updates")
    lag_fade_updates = max(0, int(lag_fade_cfg)) if lag_fade_cfg is not None else max(1, total_updates // 3)

    history_prob_start = float(cfg.get("opponent_history_sample_p", agent.opponent_history_sample_p))
    history_prob_end = float(cfg.get("opponent_history_sample_p_end", history_prob_start))
    agent.opponent_history_sample_p = history_prob_start

    clip_factor = 1.0
    lr_factor = 1.0

    clip_floor = float(cfg.get("clip_floor", 0.10))
    clip_ceiling = float(cfg.get("clip_ceiling", 0.60))

    min_lr = float(cfg.get("min_lr", 5e-5))
    max_lr = float(cfg.get("max_lr", 8e-4))

    target_kl = float(cfg.get("target_kl", 0.01))
    kl_low = float(cfg.get("kl_low", 0.5 * target_kl))
    kl_high = float(cfg.get("kl_high", 3.0 * target_kl))

    kl_clip_factor_up = float(cfg.get("kl_clip_factor_up", 1.5))
    kl_clip_factor_down = float(cfg.get("kl_clip_factor_down", 0.7))
    kl_lr_factor_up = float(cfg.get("kl_lr_factor_up", 1.5))
    kl_lr_factor_down = float(cfg.get("kl_lr_factor_down", 0.7))

    warm_decay_ratio = float(cfg.get("warm_decay_ratio", 0.7))
    force_kl_gate = bool(cfg.get("force_kl_gate", True))
    kl_reached_low = False

    last_update_metrics: Optional[Dict[str, float]] = None
    eval_every = int(cfg.get("eval_every_updates", 20) or 0)
    es_abs = float(cfg.get("early_stop_abs_err", 1.0))
    es_pat = int(cfg.get("early_stop_patience", 5) or 0)
    es_counter = 0
    early_stop_triggered = False

    while steps_done < total_steps_target:
        if total_updates > 1:
            hist_progress = float(update_idx) / float(total_updates - 1)
            hist_progress = max(0.0, min(1.0, hist_progress))
        else:
            hist_progress = 1.0
        agent.opponent_history_sample_p = history_prob_start + (history_prob_end - history_prob_start) * hist_progress

        # Entropy: hold high for first ~2/3 updates, then ramp down
        if update_idx < hold_updates:
            if hold_updates > 1:
                hold_progress = float(update_idx) / float(hold_updates - 1)
            else:
                hold_progress = 1.0
            hold_progress = max(0.0, min(1.0, hold_progress))
            agent.cfg.entropy_coef = entropy_start + (entropy_hold - entropy_start) * hold_progress
        else:
            tail_progress = float(update_idx - hold_updates) / float(max(1, tail_updates - 1))
            tail_progress = max(0.0, min(1.0, tail_progress))
            agent.cfg.entropy_coef = entropy_hold + (entropy_final - entropy_hold) * tail_progress
        progress = float(update_idx) / float(max(1, total_updates - 1))
        progress = max(0.0, min(1.0, progress))
        use_decay = progress >= warm_decay_ratio
        if force_kl_gate and not kl_reached_low:
            use_decay = False

        # Clip schedule with adaptive scaling
        if not use_decay:
            clip_base = clip_max
        else:
            if update_idx < hold_updates:
                clip_base = clip_max
            else:
                tail_progress = float(update_idx - hold_updates) / float(max(1, tail_updates - 1))
                tail_progress = max(0.0, min(1.0, tail_progress))
                clip_base = clip_max + (clip_min - clip_max) * tail_progress
        clip_val = max(clip_floor, min(clip_ceiling, clip_base * clip_factor))
        agent.cfg.clip_eps = clip_val
        clip_base_current = clip_base

        # Learning rate schedule with adaptive scaling
        if not use_decay:
            lr_base = lr_hold
        else:
            if update_idx < hold_updates:
                lr_base = lr_hold
            else:
                lr_tail_progress = float(update_idx - hold_updates) / float(max(1, tail_updates - 1))
                lr_tail_progress = max(0.0, min(1.0, lr_tail_progress))
                lr_base = lr_hold + (lr_final - lr_hold) * lr_tail_progress
        lr_val = max(min_lr, min(max_lr, lr_base * lr_factor))
        for g in agent.opt.param_groups:
            g["lr"] = lr_val
        lr_base_current = lr_base
        # Determine probability of sampling lagged-opponent paths for this update
        if update_idx < lag_warmup_updates:
            lag_prob = 1.0
        elif update_idx < lag_warmup_updates + lag_fade_updates:
            denom = max(1, lag_fade_updates - 1)
            lag_phase = update_idx - lag_warmup_updates
            lag_prob = max(0.0, 1.0 - (lag_phase / denom))
        else:
            lag_prob = 0.0
        steps_this = min(ppo_cfg.steps_per_update, total_steps_target - steps_done)
        for _ in range(steps_this):
            q = float(rng.choice(train_qs))
            env = TwoPlayersEnv(w_h=w_h, w_l=w_l, k=k, q=q, effort_bounds=effort_bounds, seed=cfg.get("seed", 42))

            # Sampling strategy: early-phase mixes in lagged opponent samples (no stochastic env noise);
            # late-phase switches to fully on-policy symmetric sampling and stores both trajectories.
            s1 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
            s2 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
            a1_norm, e1, logp1, v1 = agent.act(s1)
            # Early phase: with prob=lag_prob, draw opponent action from lagged/historical policy.
            # Late phase: fully on-policy symmetric sampling.
            use_opponent = (lag_prob > 0.0) and (rng.random() < lag_prob)
            if use_opponent:
                a2_norm, e2, logp2, _ = agent.act_opponent(s2)
                v2 = agent.value_only(s2)
            else:
                a2_norm, e2, logp2, v2 = agent.act(s2)

            _, rewards, _, done, _ = env.step((torch.tensor([float(e1.item())]), torch.tensor([float(e2.item())])))

            agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
            agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
            history.append(float((e1.item() + e2.item()) / 2.0))
        last_update_metrics = agent.update()
        kl_val = float(last_update_metrics.get("approx_kl", 0.0) if last_update_metrics else 0.0)
        if not math.isfinite(kl_val):
            kl_val = 0.0

        if kl_val >= kl_low and not kl_reached_low:
            kl_reached_low = True

        if kl_val < kl_low:
            clip_factor = min(clip_factor * kl_clip_factor_up, 2.0)
            lr_factor = min(lr_factor * kl_lr_factor_up, 2.0)
        elif kl_val > kl_high:
            clip_factor = max(clip_factor * kl_clip_factor_down, 0.3)
            lr_factor = max(lr_factor * kl_lr_factor_down, 0.3)

        clip_val = max(clip_floor, min(clip_ceiling, clip_base_current * clip_factor))
        agent.cfg.clip_eps = clip_val
        lr_val = max(min_lr, min(max_lr, lr_base_current * lr_factor))
        for g in agent.opt.param_groups:
            g["lr"] = lr_val
        # After each PPO update, evaluate and log gaps for quick monitoring
        upd_i = update_idx + 1
        try:
            for q_eval in eval_qs:
                e2_star_val = clip_stage2(e_star_two_players(q_eval, w_h, w_l, k), effort_bounds)
                s_eval = agent.state_from_params(q=float(q_eval), k=k, w_h=w_h, w_l=w_l)
                with torch.no_grad():
                    dist, _ = agent.net.dist(s_eval)
                    a_eval = dist.mean.squeeze()
                    a_eval = a_eval.clamp(0.0, 1.0)
                    final_e2_eval = float(effort_bounds[0] + a_eval.detach().cpu().item() * (effort_bounds[1] - effort_bounds[0]))
                    alpha_mean = float(dist.concentration1.mean().item())
                    beta_mean = float(dist.concentration0.mean().item())
                gap = abs(final_e2_eval - e2_star_val)
                kl_val = last_update_metrics.get("approx_kl", float("nan")) if last_update_metrics else float("nan")
                adv_mean = last_update_metrics.get("adv_mean", float("nan")) if last_update_metrics else float("nan")
                hist_size = last_update_metrics.get("opponent_history_size", float(len(agent._opponent_history))) if last_update_metrics else float(len(agent._opponent_history))
                last_sync = last_update_metrics.get("opponent_last_sync", float(agent._last_sync_step)) if last_update_metrics else float(agent._last_sync_step)
                print(
                    f"[Update {upd_i}] q={q_eval}: e*={e2_star_val:.2f}, policy={final_e2_eval:.2f}, gap={gap:.2f}, "
                    f"entropy={agent.cfg.entropy_coef:.3f}, lag_prob={lag_prob:.2f}, adv_mean={adv_mean:.4f}, "
                    f"approx_kl={kl_val:.4f}, alpha_mean={alpha_mean:.2f}, beta_mean={beta_mean:.2f}, "
                    f"opp_mode={agent.opponent_mode}, last_sync={last_sync:.0f}, opp_hist_size={hist_size:.0f}"
                )
        except Exception as _e:
            # Keep training robust to any eval hiccup
            pass
        update_idx += 1
        steps_done += steps_this

        if eval_every > 0 and es_pat > 0 and (update_idx % eval_every == 0):
            abs_errs = []
            for q_eval in eval_qs:
                e2_star_val = clip_stage2(e_star_two_players(q_eval, w_h, w_l, k), effort_bounds)
                state_eval = agent.state_from_params(q=float(q_eval), k=k, w_h=w_h, w_l=w_l)
                e_eval = agent.mean_effort(state_eval)
                abs_errs.append(abs(e_eval - e2_star_val))
            mean_abs_err = float(np.mean(abs_errs)) if abs_errs else float("inf")
            if mean_abs_err < es_abs:
                es_counter += 1
            else:
                es_counter = 0
            print(f"[EarlyStopProbe] updates={update_idx} mean_abs_err={mean_abs_err:.3f} ({es_counter}/{es_pat})")
            if es_counter >= es_pat:
                print("[EarlyStop] satisfied mean_abs_err threshold and patience. Stopping training.")
                early_stop_triggered = True
                break

    # Build rows for each evaluation q
    rows: List[Dict] = []
    for q in eval_qs:
        e2_star_val = clip_stage2(e_star_two_players(q, w_h, w_l, k), effort_bounds)

        # Helper utilities for evaluation
        env_eval = TwoPlayersEnv(
            w_h=w_h,
            w_l=w_l,
            k=k,
            q=q,
            effort_bounds=effort_bounds,
            seed=cfg.get("seed", 42),
        )

        def _compute_effort(policy_net: torch.nn.Module) -> float:
            state = agent.state_from_params(q=float(q), k=k, w_h=w_h, w_l=w_l)
            with torch.no_grad():
                dist, _ = policy_net.dist(state)
                a_mean = dist.mean.squeeze().clamp(0.0, 1.0)
                return float(effort_bounds[0] + a_mean.detach().cpu().item() * (effort_bounds[1] - effort_bounds[0]))

        def _evaluate_pair(policy_net: torch.nn.Module, opponent_net: torch.nn.Module) -> Dict[str, float]:
            effort_self = _compute_effort(policy_net)
            effort_opp = _compute_effort(opponent_net)
            _, rewards, _, _, _ = env_eval.step(
                (
                    torch.tensor([effort_self], dtype=torch.float32),
                    torch.tensor([effort_opp], dtype=torch.float32),
                )
            )
            reward_self = float(rewards[0].item())
            reward_opp = float(rewards[1].item())
            return {
                "effort_self": effort_self,
                "effort_opp": effort_opp,
                "reward_self": reward_self,
                "reward_opp": reward_opp,
            }

        s_agent = agent.state_from_params(q=float(q), k=k, w_h=w_h, w_l=w_l)
        with torch.no_grad():
            dist_agent, _ = agent.net.dist(s_agent)
            a_agent = dist_agent.mean.squeeze().clamp(0.0, 1.0)
            final_e2 = float(effort_bounds[0] + a_agent.detach().cpu().item() * (effort_bounds[1] - effort_bounds[0]))
            alpha_eval = float(dist_agent.concentration1.mean().item())
            beta_eval = float(dist_agent.concentration0.mean().item())
        stage2_gap = abs(final_e2 - e2_star_val)

        row = build_csv_row(
            stage1_weight=cfg["stage1_weight"],
            stage2_weight=cfg["stage2_weight"],
            k1=cfg["k1"],
            k2=cfg["k2"],
            information_revelation=cfg.get("information_revelation", "none"),
            theoretical_stage1_effort=0.0,
            theoretical_stage2_effort=e2_star_val,
            model_training="ppo",
            final_stage1_effort=0.0,
            final_stage2_effort=final_e2,
            episodes=episodes,
        )
        row["stage2_gap_unweighted"] = stage2_gap
        row["abs_err"] = stage2_gap
        row["opp_mode"] = agent.opponent_mode
        row["opp_sync_interval"] = agent.opponent_sync_interval
        row["opp_ema_tau"] = agent.opponent_ema_tau
        row["opp_hist_size"] = len(agent._opponent_history)
        row["last_sync_step"] = agent._last_sync_step
        row["approx_kl"] = last_update_metrics.get("approx_kl", float("nan")) if last_update_metrics else float("nan")
        row["batch_entropy"] = last_update_metrics.get("batch_entropy", float("nan")) if last_update_metrics else float("nan")
        row["alpha_mean"] = alpha_eval
        row["beta_mean"] = beta_eval

        if eval_symmetric:
            sym_eval = _evaluate_pair(agent.net, agent.net)
            row["eval_symmetric_effort"] = sym_eval["effort_self"]
            row["eval_symmetric_reward"] = sym_eval["reward_self"]
            row["eval_symmetric_abs_err"] = abs(sym_eval["effort_self"] - e2_star_val)

        if eval_vs_opponent:
            opp_eval = _evaluate_pair(agent.net, agent.opponent_policy)
            row["eval_vs_opponent_effort"] = opp_eval["effort_self"]
            row["eval_vs_opponent_reward"] = opp_eval["reward_self"]
            row["eval_vs_opponent_opp_effort"] = opp_eval["effort_opp"]
            row["eval_vs_opponent_abs_err"] = abs(opp_eval["effort_self"] - e2_star_val)

        if eval_vs_history:
            history_nets = list(agent._opponent_history)
            if history_nets:
                hist_results = [_evaluate_pair(agent.net, hist_net) for hist_net in history_nets]
                efforts = np.array([res["effort_self"] for res in hist_results], dtype=np.float32)
                rewards = np.array([res["reward_self"] for res in hist_results], dtype=np.float32)
                row["eval_vs_history_effort_mean"] = float(efforts.mean())
                row["eval_vs_history_effort_std"] = float(efforts.std(ddof=0)) if efforts.size > 1 else 0.0
                row["eval_vs_history_reward_mean"] = float(rewards.mean())
                row["eval_vs_history_reward_std"] = float(rewards.std(ddof=0)) if rewards.size > 1 else 0.0
                row["eval_vs_history_abs_err_mean"] = float(np.mean(np.abs(efforts - e2_star_val)))
            else:
                row["eval_vs_history_effort_mean"] = float("nan")
                row["eval_vs_history_effort_std"] = float("nan")
                row["eval_vs_history_reward_mean"] = float("nan")
                row["eval_vs_history_reward_std"] = float("nan")
                row["eval_vs_history_abs_err_mean"] = float("nan")

        rows.append(row)

    # Plot overlays for each q (training history)
    plot_effort_curve(
        efforts=history,
        qs=eval_qs,
        e_star_fn=e_star_two_players,
        w_h=w_h,
        w_l=w_l,
        k=k,
        title="One-Stage Two-Player learned effort vs episodes",
        output_png=os.path.join("results", "one_stage_two_players.png"),
        effort_bounds=effort_bounds,
    )

    return rows


def _run_cli(args: argparse.Namespace) -> str:
    cfg = dict(base_config)
    if args.k is not None:
        cfg["k"] = float(args.k)
        cfg["k1"] = float(args.k)
        cfg["k2"] = float(args.k)
    if args.w_h is not None:
        cfg["w_h"] = float(args.w_h)
    if args.w_l is not None:
        cfg["w_l"] = float(args.w_l)
    if args.effort_range is not None:
        lo, hi = args.effort_range
        bounds = [float(lo), float(hi)]
        cfg["effort_bounds_stage2"] = bounds
        cfg["effort_range"] = bounds
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    cfg["episodes"] = int(args.episodes)

    csv_path = os.path.join("results", "one_stage_two_players.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if args.method == "gradient":
        q_values = [args.q] if args.q is not None else list(cfg["q_list"])
        for q in q_values:
            cfg["q"] = float(q)
            if args.grad_epsilon < 1e-4:
                print(f"[gradient-2p] warning: grad-epsilon={args.grad_epsilon:.2e} may be too small for stable finite differences.", flush=True)
            row = run_gradient(
                cfg,
                lr=args.grad_lr,
                steps=args.grad_steps,
                grad_eps=args.grad_epsilon,
                tol=args.grad_tol,
                num_samples=args.grad_samples,
                init_perturb=args.grad_init_perturb,
                log=True,
            )
            save_standardized_result(row, csv_path)
    elif args.method == "mcfd":
        # MC-FD uses Gaussian noise (σ parameters) and custom CSV layout.
        row = run_mcfd(cfg, args)
        _save_mcfd_result(row, csv_path)
    else:
        # Train once; evaluate for all q (or the specified q)
        train_qs = [args.q] if args.q is not None else list(cfg["q_list"])
        eval_qs = train_qs
        rows = run_ppo(
            cfg,
            episodes=args.episodes,
            train_qs=train_qs,
            eval_qs=eval_qs,
            eval_symmetric=args.eval_symmetric,
            eval_vs_opponent=args.eval_vs_opponent,
            eval_vs_history=args.eval_vs_history,
        )
        for row in rows:
            save_standardized_result(row, csv_path)

    print(f"Saved results to {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="One-Stage Two-Player Experiment (spec)")
    parser.add_argument("--method", choices=["gradient", "ppo", "mcfd"], default="gradient")
    parser.add_argument("--q", type=float, help="Override q (otherwise run all in config q_list)")
    parser.add_argument(
        "--episodes",
        type=int,
        default=base_config.get("episodes", 1_800_000),
        help="Episodes for PPO (default config value, e.g. 2.4e6 ≈ 585 updates at 4096 steps/update)",
    )
    parser.add_argument("--grad-lr", type=float, default=base_config.get("gradient_lr", 0.1), help="Learning rate for gradient descent solver.")
    parser.add_argument("--grad-steps", type=int, default=base_config.get("gradient_steps", 2000), help="Maximum gradient descent iterations.")
    parser.add_argument("--grad-epsilon", type=float, default=base_config.get("gradient_delta", 0.1), help="Finite-difference epsilon for gradients.")
    parser.add_argument("--grad-tol", type=float, default=base_config.get("gradient_tol", 1e-4), help="Terminate when |grad| < tol.")
    parser.add_argument("--grad-samples", type=int, default=base_config.get("gradient_num_samples", 64), help="Monte Carlo samples for uniform-noise gradients.")
    parser.add_argument("--grad-init-perturb", type=float, default=base_config.get("gradient_init_perturb", 1.0), help="Initial asymmetry to avoid symmetric starts.")
    parser.add_argument("--mcfd-sigma1", type=float, default=20.0, help="Player 1 noise std (suggested values: 15, 20, 25).")
    parser.add_argument("--mcfd-sigma2", type=float, default=20.0, help="Player 2 noise std (suggested values: 15, 20, 25).")
    parser.add_argument("--mcfd-delta", type=float, default=1.0, help="Finite-difference perturbation size.")
    parser.add_argument("--mcfd-eta", type=float, default=0.1, help="Gradient-ascent learning rate for MC-FD solver.")
    parser.add_argument("--mcfd-num-samples", type=int, default=64, help="Monte Carlo samples per gradient estimate.")
    parser.add_argument("--mcfd-max-iters", type=int, default=500, help="Maximum MC-FD iterations.")
    parser.add_argument("--mcfd-tol", type=float, default=1e-3, help="Convergence tolerance for MC-FD updates.")
    parser.add_argument("--mcfd-seed", type=int, help="RNG seed for MC-FD solver (defaults to config seed).")
    parser.add_argument("--mcfd-effort-min", type=float, help="Override MC-FD effort lower bound (defaults to config stage2 min).")
    parser.add_argument("--mcfd-effort-max", type=float, help="Override MC-FD effort upper bound (defaults to config stage2 max).")
    parser.add_argument("--eval-vs-opponent", action="store_true", help="Evaluate trained policy against lagged opponent policy.")
    parser.add_argument("--eval-vs-history", action="store_true", help="Evaluate policy against each opponent snapshot and report averages.")
    parser.add_argument("--eval-symmetric", dest="eval_symmetric", action="store_true", help="Evaluate policy against itself (default enabled).")
    parser.add_argument("--no-eval-symmetric", dest="eval_symmetric", action="store_false", help="Disable symmetric self-play evaluation.")
    parser.set_defaults(eval_symmetric=True)
    parser.add_argument("--k", type=float, help="Override symmetric cost k.")
    parser.add_argument("--w_h", type=float, help="Override high prize w_h.")
    parser.add_argument("--w_l", type=float, help="Override low prize w_l.")
    parser.add_argument("--effort-range", type=float, nargs=2, metavar=("LO", "HI"), help="Override symmetric effort bounds.")
    parser.add_argument("--seed", type=int, help="Override RNG seed.")
    args = parser.parse_args()

    log_path = _build_log_path(args)
    with _tee_console_to_file(log_path):
        print(f"[log] Mirroring console output to {log_path}")
        _run_cli(args)
        print(f"[log] Run complete. Full console trace saved to {log_path}")


if __name__ == "__main__":
    main()

 
