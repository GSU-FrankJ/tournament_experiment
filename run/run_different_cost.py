#!/usr/bin/env python3
"""
One-Stage Two-Player Different Cost Experiment (k1 < k2, l1 = l2)

Implements experiment type "III.2.b Two Players with Different Cost Functions" where:
- Cost functions: C_i(e) = k_i * e^2 with k1 < k2
- Ability parameters: l1 = l2 (equal, so ability doesn't affect win probability)
- Theoretical equilibrium efforts differ per player

Usage:
    # Gradient baseline
    python run/run_different_cost.py --method gradient --q 40
    
    # PPO training
    python run/run_different_cost.py --method ppo --q 40 --episodes 2048000 --seed 42
    
    # PPO with convergence evaluation
    python run/run_different_cost.py --method ppo --q 40 --episodes 2048000 --seed 42 \
        --enable-convergence-eval --cheap-gate-profile relaxed
    
    # Custom k1, k2 values (CLI override)
    python run/run_different_cost.py --method ppo --q 40 --k1 0.0003 --k2 0.0006
    
    # Sweep all q values
    python run/run_different_cost.py --method gradient
"""

from __future__ import annotations

import sys
import os
import argparse
import math
import json
import csv
import datetime
from collections import deque
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.one_stage_different_cost import config as base_config
from utils.theory import e_star_two_players_asymmetric_cost, clip_stage2
from utils.prob import p_from_efforts
from utils.exploit_asymmetric import eval_exploitability_asymmetric
from envs.different_cost_env import DifferentCostEnv
from agents.ppo_two_players_clean import PPOTwoPlayersBandit, PPOConfig


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
    filename = f"different_cost_{method_tag}_{q_tag}{episodes_tag}{seed_tag}_{timestamp}.log"
    return os.path.join("results", "different_cost", "logs", filename)


def _clip_effort(value: float, bounds: Tuple[float, float]) -> float:
    """Clip effort to bounds."""
    lo, hi = bounds
    return float(np.clip(value, lo, hi))


# === Gradient-based solver ===

def _compute_gradients_different_cost(
    env: DifferentCostEnv,
    e1: float,
    e2: float,
    delta: float,
    num_samples: int,
) -> Tuple[float, float]:
    """
    Central-difference gradients for two-player tournament with different costs.
    
    Uses Monte Carlo estimation to compute expected utility gradients.
    No symmetry assumption - players have different cost parameters k1, k2.
    
    Args:
        env: DifferentCostEnv instance
        e1: Current effort for player 1
        e2: Current effort for player 2
        delta: Finite difference step size
        num_samples: Number of Monte Carlo samples
        
    Returns:
        (g1, g2): Gradients for player 1 and player 2
    """
    lo, hi = env.low, env.high
    
    def _utility_estimate(e_self: float, e_opp: float, k_self: float) -> float:
        """Estimate expected utility via Monte Carlo."""
        # Use the exact expected utility from environment (closed-form)
        return env.expected_utility(e_self=e_self, e_opp=e_opp, k_self=k_self)
    
    # Compute finite difference gradients for each player
    # Player 1 gradient: d/de1 EU1(e1, e2)
    e1_plus = _clip_effort(e1 + delta, (lo, hi))
    e1_minus = _clip_effort(e1 - delta, (lo, hi))
    g1 = (_utility_estimate(e1_plus, e2, env.k1) - _utility_estimate(e1_minus, e2, env.k1)) / (2.0 * delta)
    
    # Player 2 gradient: d/de2 EU2(e2, e1)
    e2_plus = _clip_effort(e2 + delta, (lo, hi))
    e2_minus = _clip_effort(e2 - delta, (lo, hi))
    g2 = (_utility_estimate(e2_plus, e1, env.k2) - _utility_estimate(e2_minus, e1, env.k2)) / (2.0 * delta)
    
    return float(g1), float(g2)


def gradient_descent_different_cost(
    cfg: Dict,
    *,
    lr: float = 0.1,
    steps: int = 2000,
    eps: float = 0.1,
    tol: float = 1e-5,
    num_samples: int = 256,
    init_perturb: float = 1.0,
    lr_decay: float = 0.9995,
    log: bool = True,
) -> Tuple[Tuple[float, float], Dict[str, Any]]:
    """
    Two-player gradient ascent for k1 != k2 (no symmetry enforcement).
    
    Each player independently maximizes their expected utility by gradient ascent.
    Unlike symmetric case, we do NOT average or enforce equal efforts.
    
    Args:
        cfg: Configuration dictionary with k1, k2, q, w_h, w_l, effort_range
        lr: Initial learning rate
        steps: Maximum gradient steps
        eps: Finite difference delta
        tol: Convergence tolerance
        num_samples: MC samples (unused - using closed-form utility)
        init_perturb: Initial perturbation from theory
        lr_decay: Learning rate decay per step
        log: Whether to print progress
        
    Returns:
        ((e1, e2), history): Final efforts and convergence history
    """
    effort_bounds = tuple(cfg["effort_range"])
    
    # Create environment
    env = DifferentCostEnv(
        w_h=cfg["w_h"],
        w_l=cfg["w_l"],
        k1=cfg["k1"],
        k2=cfg["k2"],
        q=cfg["q"],
        effort_bounds=effort_bounds,
        seed=cfg.get("seed", 42),
    )
    
    if eps <= 0:
        raise ValueError("grad_eps must be positive for finite differences")
    
    lo, hi = effort_bounds
    
    # Get theoretical equilibrium efforts (for eval logging only)
    e1_star, e2_star = e_star_two_players_asymmetric_cost(
        cfg["q"], cfg["w_h"], cfg["w_l"], cfg["k1"], cfg["k2"]
    )
    e1_star = _clip_effort(e1_star, effort_bounds)
    e2_star = _clip_effort(e2_star, effort_bounds)

    # Start at fixed fractions of effort range (no e* dependency)
    e1 = _clip_effort(lo + (hi - lo) * 0.3, effort_bounds)
    e2 = _clip_effort(lo + (hi - lo) * 0.7, effort_bounds)
    
    history = {
        "init_e1": e1,
        "init_e2": e2,
        "e1_star": e1_star,
        "e2_star": e2_star,
        "final_grad": 0.0,
        "iterations": 0,
        "e1_history": [float(e1)],
        "e2_history": [float(e2)],
        "gap1_history": [abs(e1 - e1_star)],
        "gap2_history": [abs(e2 - e2_star)],
        "step_history": [0],
    }
    
    for step in range(1, steps + 1):
        # Adaptive learning rate with exponential decay
        lr_current = lr * (lr_decay ** step)
        
        # Compute gradients (no symmetry - each player independent)
        g1, g2 = _compute_gradients_different_cost(env, e1, e2, eps, num_samples)
        
        # Gradient ascent update
        e1_new = _clip_effort(e1 + lr_current * g1, effort_bounds)
        e2_new = _clip_effort(e2 + lr_current * g2, effort_bounds)
        
        delta_e1 = abs(e1_new - e1)
        delta_e2 = abs(e2_new - e2)
        grad_norm = max(abs(g1), abs(g2))
        
        e1, e2 = e1_new, e2_new
        
        # Compute gaps to respective theoretical values
        gap1 = abs(e1 - e1_star)
        gap2 = abs(e2 - e2_star)
        max_gap = max(gap1, gap2)
        
        history["iterations"] = step
        history["final_grad"] = float(grad_norm)
        history["final_grad_tuple"] = (float(g1), float(g2))
        
        # Record convergence history
        history["e1_history"].append(float(e1))
        history["e2_history"].append(float(e2))
        history["gap1_history"].append(float(gap1))
        history["gap2_history"].append(float(gap2))
        history["step_history"].append(step)
        
        if log and (step == 1 or step % 250 == 0 or step == steps):
            print(
                f"[gradient-diff-cost] step={step:05d} e1={e1:.6f} e2={e2:.6f} "
                f"grad=({g1:.6f},{g2:.6f}) gap1={gap1:.4f} gap2={gap2:.4f} "
                f"lr={lr_current:.6f}"
            )
        
        # Convergence criteria: both players near their respective theoretical values
        max_delta = max(delta_e1, delta_e2)
        if grad_norm < tol and max_gap < tol and max_delta < tol:
            if log:
                print(
                    f"[gradient-diff-cost] converged at step={step} "
                    f"grad_norm={grad_norm:.3e} max_gap={max_gap:.3e}"
                )
            break
    
    history["final_e1"] = e1
    history["final_e2"] = e2
    history["final_gap1"] = abs(e1 - e1_star)
    history["final_gap2"] = abs(e2 - e2_star)
    
    return (e1, e2), history


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
    log: bool = True,
) -> Dict:
    """Run gradient-based solver for different cost tournament."""
    k1, k2 = cfg["k1"], cfg["k2"]
    w_h, w_l, q = cfg["w_h"], cfg["w_l"], cfg["q"]
    effort_bounds = tuple(cfg["effort_range"])
    
    # Get theoretical efforts
    e1_star, e2_star = e_star_two_players_asymmetric_cost(q, w_h, w_l, k1, k2)
    e1_star = clip_stage2(e1_star, effort_bounds)
    e2_star = clip_stage2(e2_star, effort_bounds)
    
    (e1, e2), meta = gradient_descent_different_cost(
        cfg,
        lr=lr,
        steps=steps,
        eps=grad_eps,
        tol=tol,
        num_samples=num_samples,
        init_perturb=init_perturb,
        lr_decay=lr_decay,
        log=log,
    )
    
    gap1 = abs(e1 - e1_star)
    gap2 = abs(e2 - e2_star)
    max_gap = max(gap1, gap2)
    
    if log:
        grad_tuple = meta.get("final_grad_tuple", (0.0, 0.0))
        print(
            f"[gradient-diff-cost] final e1={e1:.6f} e2={e2:.6f} "
            f"grad=({grad_tuple[0]:.6f},{grad_tuple[1]:.6f})"
        )
        print(
            f"[gradient-diff-cost] theoretical e1*={e1_star:.6f} e2*={e2_star:.6f} "
            f"gap1={gap1:.4f} gap2={gap2:.4f} max_gap={max_gap:.4f}"
        )
    
    # Save convergence history
    if log:
        convergence_data = {
            "algorithm": "gradient",
            "scenario": "different_cost",
            "num_players": 2,
            "q": float(q),
            "k1": float(k1),
            "k2": float(k2),
            "w_h": float(w_h),
            "w_l": float(w_l),
            "theoretical": {
                "effort1": float(e1_star),
                "effort2": float(e2_star),
            },
            "steps": meta["step_history"],
            "agent1_effort": meta["e1_history"],
            "agent2_effort": meta["e2_history"],
            "gap_agent1": meta["gap1_history"],
            "gap_agent2": meta["gap2_history"],
            "parameters": {
                "lr": float(lr),
                "grad_eps": float(grad_eps),
                "tol": float(tol),
                "num_samples": int(num_samples),
                "init_perturb": float(init_perturb),
            },
            "final": {
                "effort1": float(e1),
                "effort2": float(e2),
                "gap1": float(gap1),
                "gap2": float(gap2),
                "max_gap": float(max_gap),
            },
        }
        
        convergence_dir = os.path.join("results", "different_cost", "convergence")
        os.makedirs(convergence_dir, exist_ok=True)
        convergence_file = os.path.join(
            convergence_dir,
            f"different_cost_gradient_q{q:.1f}_convergence.json"
        )
        with open(convergence_file, 'w') as f:
            json.dump(convergence_data, f, indent=2)
        print(f"[gradient-diff-cost] Saved convergence history to {convergence_file}")
    
    return {
        "method": "gradient",
        "scenario": "different_cost",
        "num_players": 2,
        "q": float(q),
        "k1": float(k1),
        "k2": float(k2),
        "theoretical_effort1": float(e1_star),
        "theoretical_effort2": float(e2_star),
        "final_effort1": float(e1),
        "final_effort2": float(e2),
        "gap1": float(gap1),
        "gap2": float(gap2),
        "max_gap": float(max_gap),
        "iterations": int(meta["iterations"]),
    }


# === PPO training ===

class CheapGateTracker:
    """Rolling-window tracker for cheap stability metrics (KL + policy drift)."""

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.kl_hist: deque[float] = deque(maxlen=window_size)
        self.policy1_hist: deque[float] = deque(maxlen=window_size)
        self.policy2_hist: deque[float] = deque(maxlen=window_size)

    def update(
        self,
        approx_kl: Optional[float],
        policy_mean_effort1: Optional[float],
        policy_mean_effort2: Optional[float],
    ) -> None:
        if approx_kl is not None and math.isfinite(approx_kl):
            self.kl_hist.append(float(approx_kl))
        if policy_mean_effort1 is not None and math.isfinite(policy_mean_effort1):
            self.policy1_hist.append(float(policy_mean_effort1))
        if policy_mean_effort2 is not None and math.isfinite(policy_mean_effort2):
            self.policy2_hist.append(float(policy_mean_effort2))

    def compute(self) -> dict:
        if len(self.kl_hist) < self.window_size:
            return {
                "mean_kl_window": None,
                "std_kl_window": None,
                "drift_effort1": None,
                "drift_effort2": None,
            }
        kl_vals = list(self.kl_hist)
        p1_vals = list(self.policy1_hist)
        p2_vals = list(self.policy2_hist)
        mean_kl = float(np.mean(kl_vals))
        std_kl = float(np.std(kl_vals))
        drift1 = abs(p1_vals[-1] - p1_vals[0]) if len(p1_vals) >= self.window_size else None
        drift2 = abs(p2_vals[-1] - p2_vals[0]) if len(p2_vals) >= self.window_size else None
        return {
            "mean_kl_window": mean_kl,
            "std_kl_window": std_kl,
            "drift_effort1": drift1,
            "drift_effort2": drift2,
        }


def run_ppo_different_cost(
    cfg: Dict,
    episodes: Optional[int] = None,
    train_qs: Optional[List[float]] = None,
    *,
    ablation_name: str = "baseline",
    exploit_eps: Optional[float] = None,
    patience_exploit: int = 5,
    exploit_every_updates: int = 10,
    exploit_M: Optional[int] = None,
    disable_cheap_gate: bool = False,
    disable_exploitability: bool = False,
) -> List[Dict]:
    """
    Train PPO for asymmetric cost scenario with two separate agents.
    
    Each player has their own PPO agent with player-specific state encoding [q, k_i, w_gap].
    Both agents train simultaneously via self-play.
    
    Args:
        cfg: Configuration dictionary
        episodes: Total training steps (default from config)
        train_qs: List of q values to train on (default from config)
        ablation_name: Tag for this variant
        exploit_eps: Exploitability threshold for convergence (default: from config or 0.03)
        patience_exploit: Consecutive passes required for stopping (default 5)
        exploit_every_updates: Max interval between exploitability evaluations (default 10)
        exploit_M: Monte Carlo samples for exploitability (default: from config or 8192)
        disable_cheap_gate: If True, always evaluate exploitability (no gate)
        disable_exploitability: If True, skip exploitability evaluation entirely
        
    Returns:
        List of result dictionaries, one per q value
    """
    if episodes is None:
        episodes = int(cfg.get("episodes", 2_048_000))
    else:
        episodes = int(episodes)

    # Resolve exploit params from config when not provided via CLI
    exploit_cfg = cfg.get("convergence", {}).get("exploit", {})
    if exploit_eps is None:
        exploit_eps = float(exploit_cfg.get("exploit_eps", 0.03))
    if exploit_M is None:
        exploit_M = int(exploit_cfg.get("M", 8192))

    k1, k2 = cfg["k1"], cfg["k2"]
    w_h, w_l = cfg["w_h"], cfg["w_l"]
    effort_bounds = tuple(cfg["effort_range"])
    
    # Training configuration
    train_qs = list(train_qs if train_qs is not None else cfg["q_list"])
    
    # Theory-align settings
    theory_align_v2_enabled = bool(cfg.get("theory_align_v2", False))
    
    # PPO configuration (shared by both agents)
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
    
    # Create two separate agents (one per player, different k values)
    agent1 = PPOTwoPlayersBandit(effort_bounds=effort_bounds, cfg=ppo_cfg)
    agent2 = PPOTwoPlayersBandit(effort_bounds=effort_bounds, cfg=ppo_cfg)
    
    # Initialize schedules for both agents
    for agent in [agent1, agent2]:
        agent.cfg.entropy_coef = float(cfg.get("entropy_coef_start", agent.cfg.entropy_coef))
        agent.cfg.clip_eps = float(cfg.get("clip_range_start", agent.cfg.clip_eps))
        for g in agent.opt.param_groups:
            g["lr"] = float(cfg.get("lr_start", ppo_cfg.lr))
    
    print(f"[PPO-diff-cost] Two-agent self-play mode: agent1 (k1={k1}), agent2 (k2={k2})")
    print(f"[PPO-diff-cost] Training on q values: {train_qs}")
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
    entropy_start = float(cfg.get("entropy_coef_start", agent1.cfg.entropy_coef))
    entropy_hold = float(cfg.get("entropy_coef_hold", entropy_start))
    entropy_final = float(cfg.get("entropy_coef_end", 0.015))
    lr_hold = float(cfg.get("lr_start", agent1.cfg.lr))
    lr_final = float(cfg.get("lr_end", 2e-4))
    clip_max = float(cfg.get("clip_range_start", agent1.cfg.clip_eps))
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
    
    # Exploitability tracking
    joint_exploit_ok_streak = 0
    stop_reason = "max_updates"  # default if we exhaust episodes
    converged_flag = 0
    last_exploit_1 = None
    last_exploit_2 = None
    last_br_effort_1 = None
    last_br_effort_2 = None
    updates_since_exploit_eval = 0
    
    # Grid config for exploitability evaluation
    exploit_grid_cfg = {
        "stage_a_step": 5.0,
        "stage_b_radius": 15.0,
        "stage_b_step": 1.0,
        "stage_c_radius": 3.0,
        "stage_c_step": 0.25,
    }
    
    # Create environment (reuse across steps)
    q_init = float(train_qs[0]) if train_qs else float(cfg.get("q", 40.0))
    env = DifferentCostEnv(
        w_h=w_h,
        w_l=w_l,
        k1=k1,
        k2=k2,
        q=q_init,
        effort_bounds=effort_bounds,
        seed=cfg.get("seed", 42),
    )
    
    # Convergence history for plotting
    convergence_history: Dict[str, Any] = {
        "steps": [],
        "agent1_effort": [],
        "agent2_effort": [],
        "gap_agent1": [],
        "gap_agent2": [],
        "approx_kl_agent1": [],
        "approx_kl_agent2": [],
        "batch_entropy_agent1": [],
        "batch_entropy_agent2": [],
    }
    
    # Exploitability history (sparse, only when evaluated)
    exploit_history: List[Dict[str, Any]] = []
    
    # Print exploitability settings
    if not disable_exploitability:
        print(
            f"[PPO-diff-cost] Exploitability: eps={exploit_eps}, patience={patience_exploit}, "
            f"every={exploit_every_updates}, M={exploit_M}, disable_gate={disable_cheap_gate}",
            flush=True,
        )
    
    while steps_done < total_steps_target:
        # Update entropy schedule for both agents
        for agent in [agent1, agent2]:
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
            
            # Generate player-specific states (different k values)
            state1 = agent1.state_from_params(q=q, k=k1, w_h=w_h, w_l=w_l)
            state2 = agent2.state_from_params(q=q, k=k2, w_h=w_h, w_l=w_l)
            
            # Sample actions from respective policies
            a1_norm, e1, logp1, v1 = agent1.act(state1)
            a2_norm, e2, logp2, v2 = agent2.act(state2)
            
            # Execute environment step
            efforts = (
                torch.tensor([float(e1.item())]),
                torch.tensor([float(e2.item())]),
            )
            _, rewards, _, done, _ = env.step(efforts)
            
            # Store transitions for each agent
            agent1.store(state1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
            agent2.store(state2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
            
            steps_done += 1
        
        # PPO updates for both agents
        metrics1 = agent1.update()
        metrics2 = agent2.update()
        
        # Compute policy mean efforts at test q
        test_q = float(train_qs[0])
        test_state1 = agent1.state_from_params(q=test_q, k=k1, w_h=w_h, w_l=w_l)
        test_state2 = agent2.state_from_params(q=test_q, k=k2, w_h=w_h, w_l=w_l)
        policy_mean1 = agent1.mean_effort(test_state1)
        policy_mean2 = agent2.mean_effort(test_state2)
        
        # Get theoretical efforts for comparison
        e1_star, e2_star = e_star_two_players_asymmetric_cost(test_q, w_h, w_l, k1, k2)
        gap1 = abs(policy_mean1 - e1_star)
        gap2 = abs(policy_mean2 - e2_star)
        
        # Update convergence tracker
        if cheap_tracker is not None:
            cheap_tracker.update(
                (metrics1.get("approx_kl", 0) + metrics2.get("approx_kl", 0)) / 2,
                policy_mean1,
                policy_mean2,
            )
        
        # Record convergence history
        convergence_history["steps"].append(steps_done)
        convergence_history["agent1_effort"].append(policy_mean1)
        convergence_history["agent2_effort"].append(policy_mean2)
        convergence_history["gap_agent1"].append(gap1)
        convergence_history["gap_agent2"].append(gap2)
        convergence_history["approx_kl_agent1"].append(metrics1.get("approx_kl", 0.0))
        convergence_history["approx_kl_agent2"].append(metrics2.get("approx_kl", 0.0))
        convergence_history["batch_entropy_agent1"].append(metrics1.get("batch_entropy", 0.0))
        convergence_history["batch_entropy_agent2"].append(metrics2.get("batch_entropy", 0.0))
        
        # Logging
        if update_idx % 20 == 0 or update_idx == total_updates - 1:
            print(
                f"[PPO-diff-cost] update={update_idx:04d} steps={steps_done:08d} "
                f"e1={policy_mean1:.4f} (theory={e1_star:.4f}, gap={gap1:.4f}) "
                f"e2={policy_mean2:.4f} (theory={e2_star:.4f}, gap={gap2:.4f}) "
                f"entropy={agent1.cfg.entropy_coef:.4f}",
                flush=True,
            )
        
        # === Exploitability evaluation and early stopping ===
        updates_since_exploit_eval += 1
        
        if not disable_exploitability:
            # Decide whether to evaluate exploitability:
            # - If disable_cheap_gate: evaluate every exploit_every_updates
            # - Otherwise: evaluate when cheap gate passes OR max interval reached
            should_eval_exploit = False
            
            if disable_cheap_gate:
                # No gate - evaluate on schedule
                should_eval_exploit = (updates_since_exploit_eval >= exploit_every_updates)
            else:
                # Check cheap gate condition
                if cheap_tracker is not None:
                    cheap_stats = cheap_tracker.compute()
                    mean_kl = cheap_stats.get("mean_kl_window")
                    std_kl = cheap_stats.get("std_kl_window")
                    drift1 = cheap_stats.get("drift_effort1")
                    drift2 = cheap_stats.get("drift_effort2")
                    
                    # Cheap gate thresholds (from config or defaults)
                    mean_thresh = float(cheap_cfg.get("mean_kl_thresh", 0.012))
                    std_thresh = float(cheap_cfg.get("std_kl_thresh", 0.0035))
                    drift_thresh = float(cheap_cfg.get("drift_effort_thresh", 2.0))
                    
                    mean_ok = mean_kl is not None and mean_kl <= mean_thresh
                    std_ok = std_kl is not None and std_kl <= std_thresh
                    drift_ok = (drift1 is not None and drift2 is not None and 
                               max(drift1, drift2) <= drift_thresh)
                    
                    gate_pass = mean_ok and std_ok and drift_ok
                    should_eval_exploit = gate_pass or (updates_since_exploit_eval >= exploit_every_updates)
                else:
                    # No tracker - evaluate on schedule
                    should_eval_exploit = (updates_since_exploit_eval >= exploit_every_updates)
            
            if should_eval_exploit:
                updates_since_exploit_eval = 0
                
                # Evaluate exploitability for both players
                exploit_result = eval_exploitability_asymmetric(
                    agent1=agent1,
                    agent2=agent2,
                    q=test_q,
                    effort_bounds=effort_bounds,
                    M=exploit_M,
                    grid_cfg=exploit_grid_cfg,
                    seed=cfg.get("seed", 42) + update_idx,  # Vary seed per eval
                    w_h=w_h,
                    w_l=w_l,
                    k1=k1,
                    k2=k2,
                    l1=0.0,  # No ability difference in different-cost game
                    l2=0.0,
                    game_type="different_cost",
                )
                
                exploit_1 = exploit_result["exploit_1"]
                exploit_2 = exploit_result["exploit_2"]
                exploit_max = exploit_result["exploit_max"]
                br_effort_1 = exploit_result["br_effort_1"]
                br_effort_2 = exploit_result["br_effort_2"]
                
                # Store for final output
                last_exploit_1 = exploit_1
                last_exploit_2 = exploit_2
                last_br_effort_1 = br_effort_1
                last_br_effort_2 = br_effort_2
                
                # Log exploitability
                print(
                    f"[Exploit] upd={update_idx} exploit_1={exploit_1:.4f} exploit_2={exploit_2:.4f} "
                    f"exploit_max={exploit_max:.4f} br_effort_1={br_effort_1:.2f} "
                    f"br_effort_2={br_effort_2:.2f} streak={joint_exploit_ok_streak}/{patience_exploit}",
                    flush=True,
                )
                
                # Record in history
                exploit_history.append({
                    "update": update_idx,
                    "exploit_1": exploit_1,
                    "exploit_2": exploit_2,
                    "exploit_max": exploit_max,
                    "br_effort_1": br_effort_1,
                    "br_effort_2": br_effort_2,
                    "streak": joint_exploit_ok_streak,
                })
                
                # Joint pass: both players must have exploitability < eps
                if exploit_1 < exploit_eps and exploit_2 < exploit_eps:
                    joint_exploit_ok_streak += 1
                else:
                    joint_exploit_ok_streak = 0
                
                # Check stopping condition
                if joint_exploit_ok_streak >= patience_exploit:
                    stop_reason = "exploitability"
                    converged_flag = 1
                    print(
                        f"[Convergence] Exploitability satisfied for {joint_exploit_ok_streak} evals; stopping training.",
                        flush=True,
                    )
                    break
        
        update_idx += 1
    
    # Final evaluation
    results = []
    for q in train_qs:
        test_state1 = agent1.state_from_params(q=float(q), k=k1, w_h=w_h, w_l=w_l)
        test_state2 = agent2.state_from_params(q=float(q), k=k2, w_h=w_h, w_l=w_l)
        policy_mean1 = agent1.mean_effort(test_state1)
        policy_mean2 = agent2.mean_effort(test_state2)
        e1_star, e2_star = e_star_two_players_asymmetric_cost(float(q), w_h, w_l, k1, k2)
        gap1 = abs(policy_mean1 - e1_star)
        gap2 = abs(policy_mean2 - e2_star)
        max_gap = max(gap1, gap2)
        
        print(
            f"[PPO-diff-cost] Final q={q:.1f}: "
            f"e1={policy_mean1:.4f} (theory={e1_star:.4f}, gap={gap1:.4f}) "
            f"e2={policy_mean2:.4f} (theory={e2_star:.4f}, gap={gap2:.4f})"
        )
        
        result = {
            "method": "ppo",
            "scenario": "different_cost",
            "num_players": 2,
            "q": float(q),
            "k1": float(k1),
            "k2": float(k2),
            "w_h": float(w_h),
            "w_l": float(w_l),
            "theoretical_effort1": float(e1_star),
            "theoretical_effort2": float(e2_star),
            "final_effort1": float(policy_mean1),
            "final_effort2": float(policy_mean2),
            "gap1": float(gap1),
            "gap2": float(gap2),
            "max_gap": float(max_gap),
            "episodes": int(episodes),
            "updates": int(update_idx),
            "ablation_name": ablation_name,
            # Exploitability fields
            "stop_reason": stop_reason,
            "converged_flag": converged_flag,
            "joint_exploit_ok_streak": joint_exploit_ok_streak,
            "final_exploit_1": last_exploit_1,
            "final_exploit_2": last_exploit_2,
            "final_exploit_max": max(last_exploit_1 or 0.0, last_exploit_2 or 0.0) if last_exploit_1 is not None else None,
            "final_br_effort_1": last_br_effort_1,
            "final_br_effort_2": last_br_effort_2,
        }
        results.append(result)
        
        # Save convergence history
        seed_val = cfg.get("seed", 42)
        convergence_data = {
            "algorithm": "ppo",
            "scenario": "different_cost",
            "num_players": 2,
            "q": float(q),
            "k1": float(k1),
            "k2": float(k2),
            "w_h": float(w_h),
            "w_l": float(w_l),
            "seed": int(seed_val),
            "theoretical": {
                "effort1": float(e1_star),
                "effort2": float(e2_star),
            },
            "ablation_name": ablation_name,
            "history": convergence_history,
            "exploit_history": exploit_history,
            "final": {
                "effort1": float(policy_mean1),
                "effort2": float(policy_mean2),
                "gap1": float(gap1),
                "gap2": float(gap2),
                "max_gap": float(max_gap),
            },
            # Stopping info
            "stop_reason": stop_reason,
            "stopped_at_update": update_idx,
            "joint_exploit_ok_streak": joint_exploit_ok_streak,
            "final_exploit_1": last_exploit_1,
            "final_exploit_2": last_exploit_2,
            "final_exploit_max": max(last_exploit_1 or 0.0, last_exploit_2 or 0.0) if last_exploit_1 is not None else None,
            "final_br_effort_1": last_br_effort_1,
            "final_br_effort_2": last_br_effort_2,
            # Exploit config
            "exploit_config": {
                "exploit_eps": exploit_eps,
                "patience_exploit": patience_exploit,
                "exploit_every_updates": exploit_every_updates,
                "exploit_M": exploit_M,
                "disable_cheap_gate": disable_cheap_gate,
                "disable_exploitability": disable_exploitability,
            },
        }
        
        convergence_dir = os.path.join("results", "different_cost", "convergence")
        os.makedirs(convergence_dir, exist_ok=True)
        convergence_file = os.path.join(
            convergence_dir,
            f"different_cost_ppo_q{q:.1f}_seed{seed_val}_{ablation_name}_convergence.json"
        )
        with open(convergence_file, 'w') as f:
            json.dump(convergence_data, f, indent=2)
        print(f"[PPO-diff-cost] Saved convergence history to {convergence_file}")
    
    # Save results to CSV
    _save_results_csv(results, cfg)
    
    return results


def _save_results_csv(results: List[Dict], cfg: Dict) -> None:
    """Save results to CSV file with per-player metrics."""
    csv_path = os.path.join("results", "different_cost", "summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    fieldnames = [
        "q", "k1", "k2", "w_h", "w_l", "seed",
        "theoretical_effort1", "theoretical_effort2",
        "final_effort1", "final_effort2",
        "gap1", "gap2", "max_gap",
        "method", "episodes", "converged",
    ]
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        for r in results:
            # Determine if converged (max_gap < threshold)
            converged = r.get("max_gap", float("inf")) < cfg.get("convergence_rel_err_threshold", 0.10) * max(
                r.get("theoretical_effort1", 1), r.get("theoretical_effort2", 1)
            )
            
            row = {
                "q": r.get("q"),
                "k1": r.get("k1"),
                "k2": r.get("k2"),
                "w_h": r.get("w_h"),
                "w_l": r.get("w_l"),
                "seed": cfg.get("seed", 42),
                "theoretical_effort1": r.get("theoretical_effort1"),
                "theoretical_effort2": r.get("theoretical_effort2"),
                "final_effort1": r.get("final_effort1"),
                "final_effort2": r.get("final_effort2"),
                "gap1": r.get("gap1"),
                "gap2": r.get("gap2"),
                "max_gap": r.get("max_gap"),
                "method": r.get("method"),
                "episodes": r.get("episodes", ""),
                "converged": converged,
            }
            writer.writerow(row)
    
    print(f"[save] Results appended to {csv_path}")


# === CLI ===

def _run_cli(args: argparse.Namespace) -> str:
    """Execute CLI command."""
    cfg = dict(base_config)
    
    # Apply CLI overrides
    if args.q is not None:
        cfg["q"] = float(args.q)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.k1 is not None:
        cfg["k1"] = float(args.k1)
    if args.k2 is not None:
        cfg["k2"] = float(args.k2)
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

    # Re-apply explicit CLI entropy override (takes priority over mode defaults)
    if hasattr(args, 'override_entropy_end') and args.override_entropy_end is not None:
        cfg["entropy_coef_end"] = float(args.override_entropy_end)
        if float(args.override_entropy_end) > 0:
            cfg["entropy_coef_start"] = max(float(cfg.get("entropy_coef_start", 0)),
                                             float(args.override_entropy_end) * 2)
            cfg["entropy_coef_hold"] = max(float(cfg.get("entropy_coef_hold", 0)),
                                            float(args.override_entropy_end) * 2)
        print(f"[config] CLI entropy override re-applied after mode defaults: "
              f"entropy_coef_end={args.override_entropy_end}", flush=True)

    # --disable-entropy: zero entropy regularization (mechanism ablation)
    if args.disable_entropy:
        cfg["entropy_coef_start"] = 0.0
        cfg["entropy_coef_hold"] = 0.0
        cfg["entropy_coef_end"] = 0.0
        print("[ablation] --disable-entropy: entropy_coef=0 throughout training", flush=True)

    # Convergence eval settings
    if "convergence" not in cfg:
        cfg["convergence"] = {}
    if args.enable_convergence_eval:
        cfg["convergence"]["enabled"] = True
    if args.cheap_gate_profile is not None:
        cfg["convergence"]["cheap_gate_profile"] = args.cheap_gate_profile
    
    # Print configuration
    print(f"[config] Different Cost Experiment Configuration:")
    print(f"  k1={cfg['k1']}, k2={cfg['k2']} (k1 < k2: {cfg['k1'] < cfg['k2']})")
    print(f"  w_h={cfg['w_h']}, w_l={cfg['w_l']}")
    print(f"  effort_range={cfg['effort_range']}")
    print(f"  seed={cfg.get('seed', 42)}")
    
    if args.method == "gradient":
        q_values = [args.q] if args.q is not None else list(cfg["q_list"])
        for q in q_values:
            cfg["q"] = float(q)
            
            # Print theoretical efforts
            e1_star, e2_star = e_star_two_players_asymmetric_cost(
                q, cfg["w_h"], cfg["w_l"], cfg["k1"], cfg["k2"]
            )
            print(f"\n[q={q}] Theoretical: e1*={e1_star:.4f}, e2*={e2_star:.4f}")
            
            run_gradient(
                cfg,
                lr=args.grad_lr,
                steps=args.grad_steps,
                grad_eps=args.grad_epsilon,
                tol=args.grad_tol,
                num_samples=args.grad_samples,
                init_perturb=args.grad_init_perturb,
                lr_decay=args.grad_lr_decay,
                log=True,
            )
    else:
        train_qs = [args.q] if args.q is not None else list(cfg["q_list"])
        
        # Print theoretical efforts for all q values
        print("\n[theoretical] Expected equilibrium efforts:")
        for q in train_qs:
            e1_star, e2_star = e_star_two_players_asymmetric_cost(
                q, cfg["w_h"], cfg["w_l"], cfg["k1"], cfg["k2"]
            )
            print(f"  q={q}: e1*={e1_star:.4f}, e2*={e2_star:.4f}")
        print()
        
        run_ppo_different_cost(
            cfg,
            episodes=args.episodes,
            train_qs=train_qs,
            ablation_name=args.ablation_name,
            exploit_eps=args.exploit_eps,
            patience_exploit=args.exploit_patience,
            exploit_every_updates=args.exploit_every_updates,
            exploit_M=args.exploit_M,
            disable_cheap_gate=args.disable_cheap_gate,
            disable_exploitability=args.disable_exploitability,
        )
    
    return "OK"


def main():
    parser = argparse.ArgumentParser(
        description="One-Stage Two-Player Different Cost Experiment (k1 < k2, l1 = l2)"
    )
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
    
    # Cost parameter overrides
    parser.add_argument("--k1", type=float, help="Override player 1 cost parameter k1")
    parser.add_argument("--k2", type=float, help="Override player 2 cost parameter k2")
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
    parser.add_argument(
        "--override-entropy-end",
        type=float,
        default=None,
        help="Override entropy_coef_end (takes priority over --theory-align-v2 defaults).",
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
    
    # Exploitability evaluation settings
    parser.add_argument(
        "--exploit-eps",
        type=float,
        default=None,
        help="Override exploit_eps threshold (default: from config, 0.05)",
    )
    parser.add_argument(
        "--exploit-patience",
        type=int,
        default=5,
        help="Consecutive passes required for stopping (default: 5)",
    )
    parser.add_argument(
        "--exploit-every-updates",
        type=int,
        default=10,
        help="Max interval between exploitability evaluations (default: 10)",
    )
    parser.add_argument(
        "--exploit-M",
        type=int,
        default=None,
        help="Override exploit MC samples (default: from config, 16384)",
    )
    parser.add_argument(
        "--disable-cheap-gate",
        action="store_true",
        help="Always evaluate exploitability (no gate)",
    )
    parser.add_argument(
        "--disable-exploitability",
        action="store_true",
        help="Skip exploitability evaluation entirely",
    )
    parser.add_argument(
        "--disable-entropy",
        action="store_true",
        help="Set entropy_coef to 0 throughout training (mechanism ablation).",
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
