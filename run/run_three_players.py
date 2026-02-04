#!/usr/bin/env python3
"""
One-Stage Three-Player Experiment (self-play only)

Implements a simplified symmetric three-player tournament:
- One winner receives w_h, two losers receive w_l
- Pure self-play only (no opponent/lag modes)
- Theoretical equilibrium: e* = (w_h - w_l) / (4 * k * q)
- Direct JSON output for convergence history

Usage:
    # Gradient baseline
    python run/run_three_players.py --method gradient --q 40
    
    # PPO training (self-play is the only mode)
    python run/run_three_players.py --method ppo --q 40 --episodes 2048000 --seed 42
    
    # PPO with convergence evaluation
    python run/run_three_players.py --method ppo --q 40 --episodes 2048000 --seed 42 \
      --enable-convergence-eval --cheap-gate-profile relaxed
"""

from __future__ import annotations

import sys
import os
import argparse
import math
import json
import datetime
from collections import deque
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.one_stage_three_players import config as base_config
from utils.theory import e_star_three_players, clip_stage2
from utils.prob import win_prob_three_players_grad
from envs.three_players_env import ThreePlayersEnv
from agents.ppo_three_players import PPOThreePlayersBandit, PPOConfig


# === Console logging utilities ===

class _TeeStream:
    """Stream that writes to multiple outputs."""
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
    """Build log file path based on arguments."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    method_tag = getattr(args, "method", "run")
    q_val = getattr(args, "q", None)
    if q_val is None:
        q_tag = "q_all"
    else:
        q_clean = f"{q_val:g}".replace("-", "neg").replace(".", "p")
        q_tag = f"q{q_clean}"

    episodes_val = getattr(args, "episodes", None)
    episodes_tag = f"_ep{int(episodes_val)}" if episodes_val is not None else ""
    seed_val = getattr(args, "seed", None)
    seed_tag = f"_seed{int(seed_val)}" if seed_val is not None else ""
    filename = f"three_players_{method_tag}_{q_tag}{episodes_tag}{seed_tag}_{timestamp}.log"
    return os.path.join("results", "logs", filename)


def _clip_effort(value: float, bounds: tuple[float, float]) -> float:
    """Clip effort to bounds."""
    lo, hi = bounds
    return float(np.clip(value, lo, hi))


# === Gradient-based solver ===

def _stochastic_fd_gradients_3p(
    env: ThreePlayersEnv,
    e1: float,
    e2: float,
    e3: float,
    delta: float,
    num_samples: int,
) -> tuple[float, float, float]:
    """Central-difference gradients for three-player tournament using Monte Carlo.
    
    For symmetric equilibrium, we can use the analytic gradient from the environment.
    This function provides a stochastic alternative for validation.
    """
    # Use analytic gradient from environment (more accurate)
    if env.use_analytic:
        return env.expected_utility_gradient(e1, e2, e3)
    
    # Fallback to finite differences if analytic not available
    lo, hi = env.effort_range
    
    def _utility_estimate(e_i: float, e_j: float, e_k: float, player: int) -> float:
        """Estimate utility for given player via Monte Carlo."""
        rng = np.random.default_rng(42)
        utils = []
        for _ in range(num_samples):
            eps = rng.uniform(-env.q, env.q, size=3)
            scores = [e_i + eps[0], e_j + eps[1], e_k + eps[2]]
            winner = int(np.argmax(scores))
            efforts = [e_i, e_j, e_k]
            for p in range(3):
                payoff = env.w_h if p == winner else env.w_l
                cost = env.k * efforts[p] ** 2
                if p == player:
                    utils.append(payoff - cost)
        return float(np.mean(utils))
    
    # Compute finite difference gradients
    e1_plus = _clip_effort(e1 + delta, (lo, hi))
    e1_minus = _clip_effort(e1 - delta, (lo, hi))
    e2_plus = _clip_effort(e2 + delta, (lo, hi))
    e2_minus = _clip_effort(e2 - delta, (lo, hi))
    e3_plus = _clip_effort(e3 + delta, (lo, hi))
    e3_minus = _clip_effort(e3 - delta, (lo, hi))
    
    g1 = (_utility_estimate(e1_plus, e2, e3, 0) - _utility_estimate(e1_minus, e2, e3, 0)) / (2.0 * delta)
    g2 = (_utility_estimate(e1, e2_plus, e3, 1) - _utility_estimate(e1, e2_minus, e3, 1)) / (2.0 * delta)
    g3 = (_utility_estimate(e1, e2, e3_plus, 2) - _utility_estimate(e1, e2, e3_minus, 2)) / (2.0 * delta)
    
    return float(g1), float(g2), float(g3)


def gradient_descent_three_players(
    cfg: Dict,
    *,
    lr: float = 0.1,
    steps: int = 2000,
    eps: float = 0.1,
    tol: float = 1e-5,
    num_samples: int = 256,
    init_perturb: float = 1.0,
    lr_decay: float = 0.9995,
    symmetry_enforce_every: int = 50,
    symmetry_tol: float = 0.1,
    log: bool = True,
) -> tuple[tuple[float, float, float], Dict[str, Any]]:
    """Three-player gradient ascent with uniform noise and distinct starts."""
    effort_bounds = tuple(cfg["effort_range"])
    env = ThreePlayersEnv(
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
    e_theory = float(e_star_three_players(cfg["q"], cfg["w_h"], cfg["w_l"], cfg["k"]))
    
    # Start near theory but enforce distinct starting points
    perturb = max(init_perturb * 0.5, 1e-6)
    e1 = _clip_effort(e_theory - perturb, effort_bounds)
    e2 = _clip_effort(e_theory, effort_bounds)
    e3 = _clip_effort(e_theory + perturb, effort_bounds)

    history = {
        "init_e1": e1,
        "init_e2": e2,
        "init_e3": e3,
        "final_grad": 0.0,
        "iterations": 0.0,
        "e1_history": [float(e1)],
        "e2_history": [float(e2)],
        "e3_history": [float(e3)],
        "step_history": [0],
    }

    for step in range(1, steps + 1):
        # Adaptive learning rate with exponential decay
        lr_current = lr * (lr_decay ** step)
        
        # Use analytic gradient from environment
        g1, g2, g3 = env.expected_utility_gradient(e1, e2, e3)
        
        e1_new = _clip_effort(e1 + lr_current * g1, effort_bounds)
        e2_new = _clip_effort(e2 + lr_current * g2, effort_bounds)
        e3_new = _clip_effort(e3 + lr_current * g3, effort_bounds)

        delta_e1 = abs(e1_new - e1)
        delta_e2 = abs(e2_new - e2)
        delta_e3 = abs(e3_new - e3)
        grad_norm = max(abs(g1), abs(g2), abs(g3))

        e1, e2, e3 = e1_new, e2_new, e3_new
        
        # Periodic symmetry enforcement
        if symmetry_enforce_every > 0 and step % symmetry_enforce_every == 0:
            e_avg = (e1 + e2 + e3) / 3.0
            e1 = e2 = e3 = e_avg
            if log and step <= 100:
                print(f"  [symmetry enforce] step={step} e_avg={e_avg:.6f}")
        
        history["iterations"] = float(step)
        history["final_grad"] = float(grad_norm)
        history["final_grad_tuple"] = (float(g1), float(g2), float(g3))
        
        # Record convergence history
        history["e1_history"].append(float(e1))
        history["e2_history"].append(float(e2))
        history["e3_history"].append(float(e3))
        history["step_history"].append(step)
        
        # Compute symmetry gap (max difference among efforts)
        symmetry_gap = max(abs(e1 - e2), abs(e1 - e3), abs(e2 - e3))
        max_delta = max(delta_e1, delta_e2, delta_e3)
        
        if log and (step == 1 or step % 250 == 0 or step == steps):
            print(
                f"[gradient-3p] step={step:05d} e1={e1:.6f} e2={e2:.6f} e3={e3:.6f} "
                f"grad=({g1:.6f},{g2:.6f},{g3:.6f}) delta=({delta_e1:.3e},{delta_e2:.3e},{delta_e3:.3e}) "
                f"lr={lr_current:.6f} sym_gap={symmetry_gap:.4f}"
            )
        
        # Convergence criteria
        if (grad_norm < tol and 
            symmetry_gap < symmetry_tol and
            max_delta < tol):
            if log:
                print(
                    f"[gradient-3p] converged at step={step} "
                    f"grad_norm={grad_norm:.3e} symmetry_gap={symmetry_gap:.3e}"
                )
            break

    history["final_e1"] = e1
    history["final_e2"] = e2
    history["final_e3"] = e3
    return (e1, e2, e3), history


def run_gradient(
    cfg: Dict,
    *,
    lr: float = 0.1,
    steps: int = 2000,
    grad_eps: float = 0.1,
    tol: float = 1e-5,
    num_samples: int = 256,
    init_perturb: float = 1.0,
    lr_decay: float = 0.9995,
    symmetry_enforce_every: int = 50,
    symmetry_tol: float = 0.1,
    log: bool = True,
) -> Dict:
    """Run gradient-based solver for three-player tournament."""
    w_h, w_l, k, q = cfg["w_h"], cfg["w_l"], cfg["k"], cfg["q"]
    effort_bounds = tuple(cfg["effort_range"])
    theoretical_e = clip_stage2(e_star_three_players(q, w_h, w_l, k), effort_bounds)
    
    (e1, e2, e3), meta = gradient_descent_three_players(
        cfg,
        lr=lr,
        steps=steps,
        eps=grad_eps,
        tol=tol,
        num_samples=num_samples,
        init_perturb=init_perturb,
        lr_decay=lr_decay,
        symmetry_enforce_every=symmetry_enforce_every,
        symmetry_tol=symmetry_tol,
        log=log,
    )
    final_e = (e1 + e2 + e3) / 3.0
    
    if log:
        gap_sym = max(abs(e1 - e2), abs(e1 - e3), abs(e2 - e3))
        grad_tuple = meta.get("final_grad_tuple", (0.0, 0.0, 0.0))
        print(
            f"[gradient-3p] final e1={e1:.6f} e2={e2:.6f} e3={e3:.6f} avg={final_e:.6f} "
            f"grad=({grad_tuple[0]:.6f},{grad_tuple[1]:.6f},{grad_tuple[2]:.6f}) gap_sym={gap_sym:.3e}"
        )
        print(f"[gradient-3p] theoretical_e={theoretical_e:.6f} abs_err={abs(final_e - theoretical_e):.4f}")

    # Save convergence history
    if log:
        convergence_data = {
            "algorithm": "gradient",
            "num_players": 3,
            "q": float(q),
            "theoretical_effort": float(theoretical_e),
            "steps": meta["step_history"],
            "agent1_effort": meta["e1_history"],
            "agent2_effort": meta["e2_history"],
            "agent3_effort": meta["e3_history"],
            "parameters": {
                "lr": float(lr),
                "grad_eps": float(grad_eps),
                "tol": float(tol),
                "num_samples": int(num_samples),
                "init_perturb": float(init_perturb),
            },
            "final_results": {
                "final_e1": float(e1),
                "final_e2": float(e2),
                "final_e3": float(e3),
                "final_avg": float(final_e),
                "theoretical_e": float(theoretical_e),
                "abs_error": float(abs(final_e - theoretical_e)),
            }
        }
        
        convergence_dir = os.path.join("results", "convergence_history")
        os.makedirs(convergence_dir, exist_ok=True)
        convergence_file = os.path.join(
            convergence_dir, 
            f"gradient_3p_q{q:.1f}_convergence.json"
        )
        with open(convergence_file, 'w') as f:
            json.dump(convergence_data, f, indent=2)
        print(f"[gradient-3p] Saved convergence history to {convergence_file}")
    
    return {
        "method": "gradient",
        "num_players": 3,
        "q": float(q),
        "theoretical_effort": float(theoretical_e),
        "final_effort": float(final_e),
        "abs_error": float(abs(final_e - theoretical_e)),
        "iterations": int(meta["iterations"]),
        "symmetry_gap": float(max(abs(e1 - e2), abs(e1 - e3), abs(e2 - e3))),
    }


# === PPO training ===

class CheapGateTracker:
    """Rolling-window tracker for cheap stability metrics (KL + policy drift)."""

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.kl_hist: deque[float] = deque(maxlen=window_size)
        self.policy_hist: deque[float] = deque(maxlen=window_size)

    def update(self, approx_kl: float | None, policy_mean_effort: float | None) -> None:
        if approx_kl is not None and math.isfinite(approx_kl):
            self.kl_hist.append(float(approx_kl))
        if policy_mean_effort is not None and math.isfinite(policy_mean_effort):
            self.policy_hist.append(float(policy_mean_effort))

    def compute(self) -> dict:
        if len(self.kl_hist) < self.window_size or len(self.policy_hist) < self.window_size:
            return {
                "mean_kl_window": None,
                "std_kl_window": None,
                "drift_effort": None,
            }
        kl_vals = list(self.kl_hist)
        policy_vals = list(self.policy_hist)
        mean_kl = float(np.mean(kl_vals))
        std_kl = float(np.std(kl_vals))
        drift = abs(policy_vals[-1] - policy_vals[0])
        return {
            "mean_kl_window": mean_kl,
            "std_kl_window": std_kl,
            "drift_effort": float(drift),
        }


def run_ppo(
    cfg: Dict,
    episodes: Optional[int] = None,
    train_qs: Optional[List[float]] = None,
    *,
    ablation_name: str = "baseline",
) -> List[Dict]:
    """Train PPO via pure self-play for three-player tournament.
    
    All three players share the same policy (symmetric equilibrium).
    Each environment step produces 3 transitions (one per player).
    
    Args:
        cfg: Configuration dictionary
        episodes: Total training steps (default from config)
        train_qs: List of q values to train on (default from config)
        ablation_name: Tag for this variant
        
    Returns:
        List of result dictionaries, one per q value
    """
    if episodes is None:
        episodes = int(cfg.get("episodes", 2_048_000))
    else:
        episodes = int(episodes)

    w_h, w_l, k = cfg["w_h"], cfg["w_l"], cfg["k"]
    effort_bounds = tuple(cfg["effort_range"])
    
    # Training configuration
    train_qs = list(train_qs if train_qs is not None else cfg["q_list"])
    
    # Theory-align settings
    theory_align_v2_enabled = bool(cfg.get("theory_align_v2", False))
    
    # PPO agent configuration (simplified for self-play)
    ppo_cfg = PPOConfig(
        steps_per_update=int(cfg.get("steps_per_update", 4096)),
        epochs=int(cfg.get("update_epochs", 6)),
        minibatch_size=int(cfg.get("minibatch_size", 1024)),
        state_dim=3,  # [q_norm, k_norm, wgap_norm]
        hidden=128,
        entropy_coef=float(cfg.get("entropy_coef_start", 0.03)),
        lr=float(cfg.get("lr_start", 3e-4)),
        clip_eps=float(cfg.get("clip_range_start", 0.50)),
        kl_early_stop=bool(cfg.get("kl_early_stop", False)),
        kl_stop_patience=int(cfg.get("kl_stop_patience", 1)),
        kl_stop_threshold=cfg.get("kl_stop_threshold"),
        ratio_stop_threshold=cfg.get("ratio_stop_threshold"),
        target_kl=float(cfg.get("target_kl", 0.08)),
        theory_align_v2=theory_align_v2_enabled,
    )
    
    agent = PPOThreePlayersBandit(effort_bounds=effort_bounds, cfg=ppo_cfg)
    
    # Initialize schedules
    agent.cfg.entropy_coef = float(cfg.get("entropy_coef_start", agent.cfg.entropy_coef))
    agent.cfg.clip_eps = float(cfg.get("clip_range_start", agent.cfg.clip_eps))
    for g in agent.opt.param_groups:
        g["lr"] = float(cfg.get("lr_start", ppo_cfg.lr))

    print(f"[PPO-3p] Pure self-play mode: all 3 players share same policy")
    print(f"[PPO-3p] Training on q values: {train_qs}")
    print(flush=True)

    # Training state
    total_steps_target = int(episodes)
    max_updates_cfg = int(cfg.get("max_updates", 0) or 0)
    if max_updates_cfg > 0:
        capped_steps = max_updates_cfg * ppo_cfg.steps_per_update
        if total_steps_target > capped_steps:
            total_steps_target = capped_steps
            print(f"[config] max_updates={max_updates_cfg} -> total_steps capped at {total_steps_target}", flush=True)
    
    steps_done = 0
    rng = np.random.default_rng(cfg.get("seed", 42))
    
    # Schedule parameters
    entropy_start = float(cfg.get("entropy_coef_start", agent.cfg.entropy_coef))
    entropy_hold = float(cfg.get("entropy_coef_hold", entropy_start))
    entropy_final = float(cfg.get("entropy_coef_end", 0.015))
    lr_hold = float(cfg.get("lr_start", agent.cfg.lr))
    lr_final = float(cfg.get("lr_end", 2e-4))
    clip_max = float(cfg.get("clip_range_start", agent.cfg.clip_eps))
    clip_min = float(cfg.get("clip_range_end", 0.35))
    
    update_idx = 0
    total_updates = (total_steps_target + ppo_cfg.steps_per_update - 1) // ppo_cfg.steps_per_update
    hold_fraction = float(cfg.get("entropy_hold_fraction", 2.0 / 3.0))
    hold_updates = max(1, int(math.ceil(total_updates * hold_fraction)))
    tail_updates = max(1, total_updates - hold_updates)
    
    # Convergence tracking
    convergence_cfg = cfg.get("convergence", {}) or {}
    convergence_enabled = bool(convergence_cfg.get("enabled", False))
    cheap_cfg = convergence_cfg.get("cheap_gate", {}) if convergence_enabled else {}
    cheap_tracker = CheapGateTracker(int(cheap_cfg.get("window_size", 20))) if convergence_enabled else None
    
    # Create environment (reuse across steps for RNG continuity)
    q_init = float(train_qs[0]) if train_qs else float(cfg.get("q", 40.0))
    env = ThreePlayersEnv(
        w_h=w_h,
        w_l=w_l,
        k=k,
        q=q_init,
        effort_bounds=effort_bounds,
        seed=cfg.get("seed", 42),
    )
    
    # Convergence history for plotting
    convergence_history: Dict[str, Any] = {
        "steps": [],
        "agent1_effort": [],
        "agent2_effort": [],
        "agent3_effort": [],
        "policy_mean_effort": [],
        "approx_kl": [],
        "batch_entropy": [],
    }
    
    last_update_metrics: Optional[Dict[str, float]] = None
    
    while steps_done < total_steps_target:
        # Update entropy schedule
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
        
        # Update clip schedule
        if update_idx < hold_updates:
            clip_base = clip_max
        else:
            tail_progress = float(update_idx - hold_updates) / float(max(1, tail_updates - 1))
            tail_progress = max(0.0, min(1.0, tail_progress))
            clip_base = clip_max + (clip_min - clip_max) * tail_progress
        agent.cfg.clip_eps = clip_base
        
        # Update learning rate schedule
        if update_idx < hold_updates:
            lr_base = lr_hold
        else:
            lr_tail_progress = float(update_idx - hold_updates) / float(max(1, tail_updates - 1))
            lr_tail_progress = max(0.0, min(1.0, lr_tail_progress))
            lr_base = lr_hold + (lr_final - lr_hold) * lr_tail_progress
        for g in agent.opt.param_groups:
            g["lr"] = lr_base
        
        # Collect rollout
        steps_this = min(ppo_cfg.steps_per_update, total_steps_target - steps_done)
        
        for _ in range(steps_this):
            q = float(rng.choice(train_qs))
            env.q = q
            
            # Generate state for all players (same state in symmetric case)
            state = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
            
            # Sample actions for all 3 players from same policy
            a1_norm, e1, logp1, v1 = agent.act(state)
            a2_norm, e2, logp2, v2 = agent.act(state)
            a3_norm, e3, logp3, v3 = agent.act(state)
            
            # Execute environment step
            efforts = (
                torch.tensor([float(e1.item())]),
                torch.tensor([float(e2.item())]),
                torch.tensor([float(e3.item())]),
            )
            _, rewards, _, done, _ = env.step(efforts)
            
            # Store all 3 transitions (symmetric self-play)
            agent.store(state, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
            agent.store(state, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
            agent.store(state, a3_norm, logp3, float(rewards[2].item()), v3, bool(done))
            
            steps_done += 1
        
        # PPO update
        metrics = agent.update()
        last_update_metrics = metrics
        
        # Compute policy mean effort
        test_state = agent.state_from_params(q=float(train_qs[0]), k=k, w_h=w_h, w_l=w_l)
        policy_mean_effort = agent.mean_effort(test_state)
        theoretical_e = e_star_three_players(float(train_qs[0]), w_h, w_l, k)
        
        # Update convergence tracker
        if cheap_tracker is not None:
            cheap_tracker.update(metrics.get("approx_kl"), policy_mean_effort)
        
        # Record convergence history
        convergence_history["steps"].append(steps_done)
        convergence_history["agent1_effort"].append(policy_mean_effort)
        convergence_history["agent2_effort"].append(policy_mean_effort)
        convergence_history["agent3_effort"].append(policy_mean_effort)
        convergence_history["policy_mean_effort"].append(policy_mean_effort)
        convergence_history["approx_kl"].append(metrics.get("approx_kl", 0.0))
        convergence_history["batch_entropy"].append(metrics.get("batch_entropy", 0.0))
        
        # Logging
        if update_idx % 20 == 0 or update_idx == total_updates - 1:
            abs_err = abs(policy_mean_effort - theoretical_e)
            print(
                f"[PPO-3p] update={update_idx:04d} steps={steps_done:08d} "
                f"policy_mean={policy_mean_effort:.4f} theory={theoretical_e:.4f} "
                f"abs_err={abs_err:.4f} entropy={agent.cfg.entropy_coef:.4f} "
                f"kl={metrics.get('approx_kl', 0.0):.6f}",
                flush=True,
            )
        
        update_idx += 1
    
    # Final evaluation
    results = []
    for q in train_qs:
        test_state = agent.state_from_params(q=float(q), k=k, w_h=w_h, w_l=w_l)
        policy_mean_effort = agent.mean_effort(test_state)
        theoretical_e = e_star_three_players(float(q), w_h, w_l, k)
        abs_err = abs(policy_mean_effort - theoretical_e)
        
        print(
            f"[PPO-3p] Final q={q:.1f}: policy_mean={policy_mean_effort:.4f} "
            f"theory={theoretical_e:.4f} abs_err={abs_err:.4f}"
        )
        
        result = {
            "method": "ppo",
            "num_players": 3,
            "q": float(q),
            "theoretical_effort": float(theoretical_e),
            "final_effort": float(policy_mean_effort),
            "abs_error": float(abs_err),
            "episodes": int(episodes),
            "updates": int(update_idx),
            "ablation_name": ablation_name,
        }
        results.append(result)
        
        # Save convergence history
        convergence_data = {
            "algorithm": "ppo",
            "num_players": 3,
            "q": float(q),
            "theoretical_effort": float(theoretical_e),
            "ablation_name": ablation_name,
            **convergence_history,
            "final_results": {
                "final_effort": float(policy_mean_effort),
                "theoretical_e": float(theoretical_e),
                "abs_error": float(abs_err),
            }
        }
        
        convergence_dir = os.path.join("results", "convergence_history")
        os.makedirs(convergence_dir, exist_ok=True)
        seed_val = cfg.get("seed", 42)
        convergence_file = os.path.join(
            convergence_dir, 
            f"ppo_3p_q{q:.1f}_seed{seed_val}_{ablation_name}_convergence.json"
        )
        with open(convergence_file, 'w') as f:
            json.dump(convergence_data, f, indent=2)
        print(f"[PPO-3p] Saved convergence history to {convergence_file}")
    
    return results


# === CLI ===

def _run_cli(args: argparse.Namespace) -> str:
    """Execute CLI command."""
    cfg = dict(base_config)
    
    # Apply CLI overrides
    if args.q is not None:
        cfg["q"] = float(args.q)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.k is not None:
        cfg["k"] = float(args.k)
    if args.w_h is not None:
        cfg["w_h"] = float(args.w_h)
    if args.w_l is not None:
        cfg["w_l"] = float(args.w_l)
    if args.effort_range is not None:
        cfg["effort_range"] = list(args.effort_range)
    
    # Theory-align-v2 settings
    if args.theory_align_v2:
        cfg["entropy_coef_start"] = 0.0
        cfg["entropy_coef_hold"] = 0.0
        cfg["entropy_coef_end"] = 0.0
        cfg["theory_align_v2"] = True
        print("[TheoryAlignV2] enabled: entropy=0", flush=True)
    
    # Convergence eval settings
    if "convergence" not in cfg:
        cfg["convergence"] = {}
    if args.enable_convergence_eval:
        cfg["convergence"]["enabled"] = True
    if args.cheap_gate_profile is not None:
        cfg["convergence"]["cheap_gate_profile"] = args.cheap_gate_profile
    
    if args.method == "gradient":
        q_values = [args.q] if args.q is not None else list(cfg["q_list"])
        for q in q_values:
            cfg["q"] = float(q)
            run_gradient(
                cfg,
                lr=args.grad_lr,
                steps=args.grad_steps,
                grad_eps=args.grad_epsilon,
                tol=args.grad_tol,
                num_samples=args.grad_samples,
                init_perturb=args.grad_init_perturb,
                lr_decay=args.grad_lr_decay,
                symmetry_enforce_every=args.grad_symmetry_enforce,
                symmetry_tol=args.grad_symmetry_tol,
                log=True,
            )
    else:
        train_qs = [args.q] if args.q is not None else list(cfg["q_list"])
        run_ppo(
            cfg,
            episodes=args.episodes,
            train_qs=train_qs,
            ablation_name=args.ablation_name,
        )
    
    return "OK"


def main():
    parser = argparse.ArgumentParser(description="One-Stage Three-Player Experiment (self-play)")
    parser.add_argument("--method", choices=["gradient", "ppo"], default="gradient")
    parser.add_argument("--q", type=float, help="Override q (otherwise run all in config q_list)")
    parser.add_argument(
        "--episodes",
        type=int,
        default=base_config.get("episodes", 2_048_000),
        help="Episodes for PPO training",
    )
    parser.add_argument("--seed", type=int, help="Override RNG seed")
    
    # Gradient method settings
    parser.add_argument("--grad-lr", type=float, default=base_config.get("gradient_lr", 0.08))
    parser.add_argument("--grad-steps", type=int, default=base_config.get("gradient_steps", 1500))
    parser.add_argument("--grad-epsilon", type=float, default=base_config.get("gradient_delta", 0.5))
    parser.add_argument("--grad-tol", type=float, default=base_config.get("gradient_tol", 1e-4))
    parser.add_argument("--grad-samples", type=int, default=base_config.get("gradient_num_samples", 64))
    parser.add_argument("--grad-init-perturb", type=float, default=base_config.get("gradient_init_perturb", 1.0))
    parser.add_argument("--grad-lr-decay", type=float, default=0.9995)
    parser.add_argument("--grad-symmetry-enforce", type=int, default=50)
    parser.add_argument("--grad-symmetry-tol", type=float, default=0.1)
    
    # Game parameter overrides
    parser.add_argument("--k", type=float, help="Override cost parameter k")
    parser.add_argument("--w_h", type=float, help="Override high prize w_h")
    parser.add_argument("--w_l", type=float, help="Override low prize w_l")
    parser.add_argument("--effort-range", type=float, nargs=2, metavar=("LO", "HI"))
    
    # Theory-align settings
    parser.add_argument(
        "--theory-align-v2",
        action="store_true",
        dest="theory_align_v2",
        help="Enable theory-align-v2 (zero entropy, mean+conc head)",
    )
    
    # Convergence evaluation
    parser.add_argument(
        "--enable-convergence-eval",
        action="store_true",
        dest="enable_convergence_eval",
        help="Enable convergence evaluation",
    )
    parser.add_argument(
        "--no-convergence-eval",
        action="store_false",
        dest="enable_convergence_eval",
    )
    parser.set_defaults(enable_convergence_eval=False)
    
    cheap_gate_profiles = base_config.get("convergence", {}).get("cheap_gate_profiles", {}) or {}
    cheap_gate_profile_choices = sorted(cheap_gate_profiles.keys()) if cheap_gate_profiles else ["default", "conservative", "aggressive", "relaxed"]
    parser.add_argument(
        "--cheap-gate-profile",
        choices=cheap_gate_profile_choices,
        default=None,
    )
    
    # Ablation name
    parser.add_argument(
        "--ablation-name",
        type=str,
        default="baseline",
        help="Ablation variant name for output files",
    )
    
    args = parser.parse_args()
    
    log_path = _build_log_path(args)
    with _tee_console_to_file(log_path):
        print(f"[log] Mirroring console output to {log_path}")
        _run_cli(args)
        print(f"[log] Run complete. Full console trace saved to {log_path}")


if __name__ == "__main__":
    main()
