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
from utils.rollout_stats import (
    RolloutStatsAccumulator,
    compute_policy_mean_effort,
    verify_policy_mean,
)
from envs.two_players_env import TwoPlayersEnv
from agents.ppo_two_players_clean import PPOTwoPlayersBandit, PPOConfig
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
    # Use q parameter to tag runs for gradient/PPO methods
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


def run_ppo(
    cfg: Dict,
    episodes: Optional[int] = None,
    train_qs: Optional[List[float]] = None,
    eval_qs: Optional[List[float]] = None,
    *,
    rollout_mode: str = "vs_opponent",
    eval_symmetric: bool = True,
    eval_vs_opponent: bool = False,
    eval_vs_history: bool = False,
    run_id: Optional[str] = None,
    variant_name: str = "baseline",
) -> List[Dict]:
    """Train PPO via self-play with conditioning on (q, k, w_gap).

    - Trains over ``train_qs`` (defaults to cfg["q_list"]).
    - Returns a list of CSV rows, one per q in ``eval_qs`` (defaults to train_qs).
    - rollout_mode controls action generation and storage:
        * "selfplay": Both players always use learner policy; store both transitions
        * "vs_opponent": Player1 uses learner; Player2 may use opponent (with lag schedule);
                         store only learner-generated transitions
    - run_id: Unique identifier for this run (timestamp string for log/CSV correlation)
    - variant_name: Name of the sweep variant (e.g., "baseline", "entropy_end_0.025")
    """
    # Validate rollout_mode
    if rollout_mode not in ("selfplay", "vs_opponent"):
        raise ValueError(f"rollout_mode must be 'selfplay' or 'vs_opponent', got '{rollout_mode}'")
    if episodes is None:
        episodes = int(cfg.get("episodes", 1_800_000))
    else:
        episodes = int(episodes)

    w_h, w_l, k = cfg["w_h"], cfg["w_l"], cfg["k"]
    effort_bounds = tuple(cfg["effort_bounds_stage2"])  # (0, 200)
    # Respect CLI-provided training set; default to config q_list
    train_qs = list(train_qs if train_qs is not None else cfg["q_list"])
    eval_qs = list(eval_qs if eval_qs is not None else train_qs)

    # === Print final resolved hyperparameters BEFORE agent creation ===
    # These are the values that will actually be used in training schedules.
    # Include run_id and variant_name for sweep correlation.
    _entropy_end_final = float(cfg.get("entropy_coef_end", 0.005))
    _lr_end_final = float(cfg.get("lr_end", 2e-4))
    _clip_end_final = float(cfg.get("clip_range_end", 0.15))
    _target_kl_final = float(cfg.get("target_kl", 0.01))
    print(
        f"[config] final: entropy_coef_end={_entropy_end_final} lr_end={_lr_end_final} "
        f"clip_range_end={_clip_end_final} target_kl={_target_kl_final} "
        f"run_id={run_id or 'none'} variant_name={variant_name}",
        flush=True,
    )

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

    # Print rollout mode for clarity
    print(f"[PPO] Rollout mode: {rollout_mode.upper()}")
    if rollout_mode == "selfplay":
        print("[PPO]   - Both players use learner policy")
        print("[PPO]   - Store both transitions every step")
    else:  # vs_opponent
        print("[PPO]   - Player1 uses learner; Player2 may use opponent (lag schedule)")
        print("[PPO]   - Store only learner-generated transitions")
    print(flush=True)

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

    # Reuse a single env so its RNG advances across steps; recreating it each step
    # would re-seed and make noise effectively deterministic for a fixed q.
    q_init = float(train_qs[0]) if train_qs else float(cfg.get("q", 0.0))
    env = TwoPlayersEnv(
        w_h=w_h,
        w_l=w_l,
        k=k,
        q=q_init,
        effort_bounds=effort_bounds,
        seed=cfg.get("seed", 42),
    )

    # Storage counters for debugging/verification
    stored_p1_total = 0
    stored_p2_total = 0
    skipped_p2_due_to_opponent_total = 0
    
    # Rollout stats accumulator for tracking sampled efforts (instrumentation)
    rollout_stats = RolloutStatsAccumulator()
    
    # Last rollout stats snapshot for logging (persists across updates)
    last_rollout_stats: Optional[Dict[str, float]] = None

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
        # Per-update storage counters
        stored_p1_this_update = 0
        stored_p2_this_update = 0
        skipped_p2_this_update = 0
        
        # Reset rollout stats for this update period
        rollout_stats.reset()

        for _ in range(steps_this):
            q = float(rng.choice(train_qs))
            env.q = q

            # Generate states for both players
            s1 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
            s2 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)

            # Player 1: ALWAYS uses learner policy (both modes)
            a1_norm, e1, logp1, v1 = agent.act(s1)

            # Player 2: Mode-dependent action generation
            if rollout_mode == "selfplay":
                # SELFPLAY MODE: Player2 always uses learner policy
                # Opponent lag mechanism is disabled for action generation
                a2_norm, e2, logp2, v2 = agent.act(s2)
                use_opponent = False  # Not used for action selection in selfplay

            else:  # rollout_mode == "vs_opponent"
                # VS_OPPONENT MODE: Player2 may use opponent based on lag schedule
                # Early phase: with prob=lag_prob, draw opponent action from lagged/historical policy.
                # Late phase: fully on-policy learner sampling.
                use_opponent = (lag_prob > 0.0) and (rng.random() < lag_prob)
                if use_opponent:
                    # Player2 uses opponent policy (lagged/historical)
                    a2_norm, e2, logp2, _ = agent.act_opponent(s2)
                    v2 = agent.value_only(s2)
                else:
                    # Player2 uses learner policy
                    a2_norm, e2, logp2, v2 = agent.act(s2)

            # Execute environment step with both actions
            _, rewards, _, done, _ = env.step((torch.tensor([float(e1.item())]), torch.tensor([float(e2.item())])))

            # Storage: Mode-dependent logic
            # Player 1: ALWAYS store (learner-generated in both modes)
            agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
            stored_p1_this_update += 1
            # Track P1's sampled effort (always learner-generated)
            rollout_stats.update_effort(float(e1.item()))

            # Player 2: Mode-dependent storage
            if rollout_mode == "selfplay":
                # SELFPLAY: Always store player2 (learner-generated)
                agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
                stored_p2_this_update += 1
                # Track P2's sampled effort (learner-generated in selfplay)
                rollout_stats.update_effort(float(e2.item()))

            else:  # rollout_mode == "vs_opponent"
                # VS_OPPONENT: Only store player2 when it used learner policy
                if not use_opponent:
                    # Player2 used learner -> store for PPO update
                    agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
                    stored_p2_this_update += 1
                    # Track P2's sampled effort (learner-generated)
                    rollout_stats.update_effort(float(e2.item()))
                else:
                    # Player2 used opponent -> treat as environment dynamics, don't store
                    skipped_p2_this_update += 1
                    # NOTE: Do NOT track P2 effort when using opponent policy

            history.append(float((e1.item() + e2.item()) / 2.0))
        last_update_metrics = agent.update()
        
        # Capture rollout stats snapshot for this update before reset
        last_rollout_stats = rollout_stats.get_summary()
        
        # Accumulate storage counters
        stored_p1_total += stored_p1_this_update
        stored_p2_total += stored_p2_this_update
        skipped_p2_due_to_opponent_total += skipped_p2_this_update
        
        # Periodic storage statistics logging (every 20 updates)
        if (update_idx + 1) % 20 == 0:
            total_stored = stored_p1_total + stored_p2_total
            effective_batch_size_this_update = stored_p1_this_update + stored_p2_this_update
            if rollout_mode == "vs_opponent" and stored_p1_total > 0:
                skip_pct = 100.0 * skipped_p2_due_to_opponent_total / float(stored_p1_total)
            else:
                skip_pct = 0.0
            print(f"[Storage Stats] Update {update_idx + 1}: "
                  f"stored_p1={stored_p1_this_update}, stored_p2={stored_p2_this_update}, "
                  f"skipped_p2={skipped_p2_this_update}, "
                  f"effective_batch={effective_batch_size_this_update} | "
                  f"Total: p1={stored_p1_total}, p2={stored_p2_total}, "
                  f"skipped={skipped_p2_due_to_opponent_total} ({skip_pct:.1f}%)", flush=True)
        
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
                e2_star_val = clip_stage2(e_star_two_players(q_eval, w_h, w_l, k), effort_bounds)  # theoretical optimal
                s_eval = agent.state_from_params(q=float(q_eval), k=k, w_h=w_h, w_l=w_l)  # build normalized state
                with torch.no_grad():
                    dist, _ = agent.net.dist(s_eval)  # get Beta distribution
                    a_eval = dist.mean.squeeze()  # get mean (Beta mean = alpha/(alpha+beta))
                    a_eval = a_eval.clamp(0.0, 1.0)
                    # policy_mean_effort: Beta mean mapped to effort range
                    # This is the "policy" value printed below
                    final_e2_eval = float(effort_bounds[0] + a_eval.detach().cpu().item() * (effort_bounds[1] - effort_bounds[0]))
                    alpha_mean = float(dist.concentration1.mean().item())
                    beta_mean = float(dist.concentration0.mean().item())
                
                # Compute policy_mean_effort from alpha/beta for verification
                policy_mean_effort_check = compute_policy_mean_effort(
                    alpha_mean, beta_mean, effort_bounds[0], effort_bounds[1]
                )
                policy_mean_check_err = abs(final_e2_eval - policy_mean_effort_check)
                
                # Sample average effort from rollout (learner transitions only)
                sample_avg_effort = last_rollout_stats.get("sample_avg_effort", 0.0) if last_rollout_stats else 0.0
                effort_sample_count = last_rollout_stats.get("effort_sample_count", 0) if last_rollout_stats else 0
                
                # mean_vs_sample_gap: policy_mean_effort - sample_avg_effort
                # Positive means policy predicts higher effort than sampled average
                mean_vs_sample_gap = final_e2_eval - sample_avg_effort
                
                gap = abs(final_e2_eval - e2_star_val)
                kl_val = last_update_metrics.get("approx_kl", float("nan")) if last_update_metrics else float("nan")
                adv_mean = last_update_metrics.get("adv_mean", float("nan")) if last_update_metrics else float("nan")
                adv_std = last_update_metrics.get("adv_std", float("nan")) if last_update_metrics else float("nan")
                adv_norm_std = last_update_metrics.get("adv_norm_std", float("nan")) if last_update_metrics else float("nan")
                state_mean = last_update_metrics.get("state_mean", float("nan")) if last_update_metrics else float("nan")
                state_std = last_update_metrics.get("state_std", float("nan")) if last_update_metrics else float("nan")
                reward_mean = last_update_metrics.get("reward_mean", float("nan")) if last_update_metrics else float("nan")
                reward_std = last_update_metrics.get("reward_std", float("nan")) if last_update_metrics else float("nan")
                hist_size = last_update_metrics.get("opponent_history_size", float(len(agent._opponent_history))) if last_update_metrics else float(len(agent._opponent_history))
                last_sync = last_update_metrics.get("opponent_last_sync", float(agent._last_sync_step)) if last_update_metrics else float(agent._last_sync_step)
                
                # Main update line: policy and gap
                print(
                    f"[Update {upd_i}] q={q_eval}: e*={e2_star_val:.2f}, policy={final_e2_eval:.2f}, gap={gap:.2f}, "
                    f"entropy={agent.cfg.entropy_coef:.3f}, lag_prob={lag_prob:.2f}, "
                    f"approx_kl={kl_val:.4f}, alpha_mean={alpha_mean:.2f}, beta_mean={beta_mean:.2f}"
                )
                # Rollout sample metrics line
                print(
                    f"  [Rollout] sample_avg_effort={sample_avg_effort:.2f}, mean_vs_sample_gap={mean_vs_sample_gap:.2f}, "
                    f"effort_samples={effort_sample_count}"
                )
                # Scale stats line
                print(
                    f"  [Scale] state_mean={state_mean:.4f}, state_std={state_std:.4f}, "
                    f"reward_mean={reward_mean:.4f}, reward_std={reward_std:.4f}, "
                    f"adv_mean={adv_mean:.4f}, adv_std={adv_std:.4f}, adv_norm_std={adv_norm_std:.4f}"
                )
                # Policy verification (once at first update to confirm definition)
                if upd_i == 1 or upd_i % 100 == 0:
                    print(
                        f"  [PolicyCheck] policy_mean_check_err={policy_mean_check_err:.6f} "
                        f"(expected <0.01; confirms policy=alpha/(alpha+beta) scaled to effort_range)"
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
        # === Run identification columns (for sweep correlation) ===
        row["run_id"] = run_id if run_id else ""
        row["variant_name"] = variant_name
        row["stage2_gap_unweighted"] = stage2_gap
        row["abs_err"] = stage2_gap
        row["rollout_mode"] = rollout_mode
        row["opp_mode"] = agent.opponent_mode
        row["opp_sync_interval"] = agent.opponent_sync_interval
        row["opp_ema_tau"] = agent.opponent_ema_tau
        row["opp_hist_size"] = len(agent._opponent_history)
        row["last_sync_step"] = agent._last_sync_step
        row["approx_kl"] = last_update_metrics.get("approx_kl", float("nan")) if last_update_metrics else float("nan")
        row["batch_entropy"] = last_update_metrics.get("batch_entropy", float("nan")) if last_update_metrics else float("nan")
        row["alpha_mean"] = alpha_eval
        row["beta_mean"] = beta_eval
        row["stored_p1_total"] = stored_p1_total
        row["stored_p2_total"] = stored_p2_total
        row["skipped_p2_total"] = skipped_p2_due_to_opponent_total
        row["effective_batch_size_total"] = stored_p1_total + stored_p2_total
        
        # === NEW INSTRUMENTATION COLUMNS ===
        # Policy mean effort (confirmed: Beta mean α/(α+β) scaled to effort range)
        policy_mean_effort = compute_policy_mean_effort(alpha_eval, beta_eval, effort_bounds[0], effort_bounds[1])
        row["policy_mean_effort"] = policy_mean_effort
        
        # Rollout sample metrics (from last update's rollout)
        sample_avg_effort_final = last_rollout_stats.get("sample_avg_effort", float("nan")) if last_rollout_stats else float("nan")
        row["sample_avg_effort"] = sample_avg_effort_final
        row["mean_vs_sample_gap"] = policy_mean_effort - sample_avg_effort_final if math.isfinite(sample_avg_effort_final) else float("nan")
        row["effort_sample_count"] = last_rollout_stats.get("effort_sample_count", 0) if last_rollout_stats else 0
        
        # Scale statistics (from agent.update())
        row["state_mean"] = last_update_metrics.get("state_mean", float("nan")) if last_update_metrics else float("nan")
        row["state_std"] = last_update_metrics.get("state_std", float("nan")) if last_update_metrics else float("nan")
        row["reward_mean"] = last_update_metrics.get("reward_mean", float("nan")) if last_update_metrics else float("nan")
        row["reward_std"] = last_update_metrics.get("reward_std", float("nan")) if last_update_metrics else float("nan")
        row["adv_mean"] = last_update_metrics.get("adv_mean", float("nan")) if last_update_metrics else float("nan")
        row["adv_std"] = last_update_metrics.get("adv_std", float("nan")) if last_update_metrics else float("nan")
        row["adv_norm_std"] = last_update_metrics.get("adv_norm_std", float("nan")) if last_update_metrics else float("nan")
        row["value_mean"] = last_update_metrics.get("value_mean", float("nan")) if last_update_metrics else float("nan")
        row["value_std"] = last_update_metrics.get("value_std", float("nan")) if last_update_metrics else float("nan")

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
    
    # === Hyperparameter overrides for automated sweeps ===
    # These allow one-at-a-time changes to specific PPO hyperparameters.
    # MUTUAL EXCLUSION: Only ONE override flag may be provided at a time.
    override_flags = [
        ("--override-entropy-end", args.override_entropy_end),
        ("--override-lr-end", args.override_lr_end),
        ("--override-clip-end", args.override_clip_end),
        ("--override-target-kl", args.override_target_kl),
    ]
    active_overrides = [(name, val) for name, val in override_flags if val is not None]
    if len(active_overrides) > 1:
        flag_names = [name for name, _ in active_overrides]
        raise ValueError(
            f"[config] ERROR: Only ONE override flag may be provided at a time. "
            f"Got {len(active_overrides)}: {', '.join(flag_names)}. "
            f"Use separate runs for each hyperparameter change."
        )
    
    # Apply the single override (if any) and determine variant_name
    overrides_applied = []
    variant_name = "baseline"
    if args.override_entropy_end is not None:
        cfg["entropy_coef_end"] = float(args.override_entropy_end)
        overrides_applied.append(f"entropy_coef_end={args.override_entropy_end}")
        variant_name = f"entropy_end_{args.override_entropy_end}"
    if args.override_lr_end is not None:
        cfg["lr_end"] = float(args.override_lr_end)
        overrides_applied.append(f"lr_end={args.override_lr_end}")
        variant_name = f"lr_end_{args.override_lr_end}"
    if args.override_clip_end is not None:
        cfg["clip_range_end"] = float(args.override_clip_end)
        overrides_applied.append(f"clip_range_end={args.override_clip_end}")
        variant_name = f"clip_end_{args.override_clip_end}"
    if args.override_target_kl is not None:
        cfg["target_kl"] = float(args.override_target_kl)
        overrides_applied.append(f"target_kl={args.override_target_kl}")
        variant_name = f"target_kl_{args.override_target_kl}"
    if overrides_applied:
        print(f"[config] Hyperparameter overrides: {', '.join(overrides_applied)}", flush=True)
    
    # Allow explicit override of variant_name via CLI (sweep script may pass this)
    if args.variant_name is not None:
        variant_name = args.variant_name
    
    # Generate run_id: use CLI-provided value or generate from current timestamp
    if args.run_id is not None:
        run_id = args.run_id
    else:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[config] run_id={run_id} variant_name={variant_name}", flush=True)

    # Use new v2 CSV path with run_id and variant_name columns
    csv_path = os.path.join("results", "one_stage_two_players_v2.csv")
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
    else:
        # Train once; evaluate for all q (or the specified q)
        train_qs = [args.q] if args.q is not None else list(cfg["q_list"])
        eval_qs = train_qs
        rows = run_ppo(
            cfg,
            episodes=args.episodes,
            train_qs=train_qs,
            eval_qs=eval_qs,
            rollout_mode=args.rollout_mode,
            eval_symmetric=args.eval_symmetric,
            eval_vs_opponent=args.eval_vs_opponent,
            eval_vs_history=args.eval_vs_history,
            run_id=run_id,
            variant_name=variant_name,
        )
        for row in rows:
            save_standardized_result(row, csv_path)

    print(f"Saved results to {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="One-Stage Two-Player Experiment (spec)")
    parser.add_argument("--method", choices=["gradient", "ppo"], default="gradient")
    parser.add_argument(
        "--rollout-mode",
        choices=["selfplay", "vs_opponent"],
        default="vs_opponent",
        help="Rollout mode for PPO: 'selfplay' (both use learner, store both) or 'vs_opponent' (p2 may use opponent, store only learner samples)",
    )
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
    # Hyperparameter override flags for automated sweeps (one-at-a-time changes)
    parser.add_argument(
        "--override-entropy-end",
        type=float,
        default=None,
        help="Override entropy_coef_end config value (e.g., 0.025 instead of baseline 0.015).",
    )
    parser.add_argument(
        "--override-lr-end",
        type=float,
        default=None,
        help="Override lr_end config value (e.g., 3e-4 instead of baseline 2e-4).",
    )
    parser.add_argument(
        "--override-clip-end",
        type=float,
        default=None,
        help="Override clip_range_end config value (e.g., 0.45 instead of baseline 0.35).",
    )
    parser.add_argument(
        "--override-target-kl",
        type=float,
        default=None,
        help="Override target_kl config value (e.g., 0.12 instead of baseline 0.08).",
    )
    # Run identification flags for sweep correlation
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Unique run identifier (timestamp string). Auto-generated if not provided.",
    )
    parser.add_argument(
        "--variant-name",
        type=str,
        default=None,
        help="Sweep variant name (e.g., 'baseline', 'entropy_end_0.025'). Auto-derived from overrides if not provided.",
    )
    args = parser.parse_args()

    log_path = _build_log_path(args)
    with _tee_console_to_file(log_path):
        print(f"[log] Mirroring console output to {log_path}")
        _run_cli(args)
        print(f"[log] Run complete. Full console trace saved to {log_path}")


if __name__ == "__main__":
    main()

 
