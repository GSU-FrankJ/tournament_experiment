#!/usr/bin/env python3
"""
One-Stage Two-Player Different Ability Experiment (l1 > l2, k1 = k2)

Implements experiment type "III.2.c Players with Different Abilities" where:
- Output model: y_i = e_i + l_i + ε_i (additive ability)
- Ability parameters: l1 > l2 (player 1 has advantage)
- Cost parameters: k1 = k2 = k (equal costs)
- Symmetric equilibrium: both players exert same effort e*
- Player 1 wins more often due to ability advantage

Theoretical equilibrium:
    e* = ((2q - (l1 - l2)) * (w_H - w_L)) / (8kq²)

Usage:
    # Gradient baseline
    python run/run_different_ability.py --method gradient --q 40
    
    # PPO training
    python run/run_different_ability.py --method ppo --q 40 --episodes 2048000 --seed 42
    
    # PPO with convergence evaluation
    python run/run_different_ability.py --method ppo --q 40 --episodes 2048000 --seed 42 \
        --enable-convergence-eval --cheap-gate-profile relaxed
    
    # Custom ability values (CLI override)
    python run/run_different_ability.py --method ppo --q 40 --l1 15 --l2 5
    
    # Sweep all q values
    python run/run_different_ability.py --method gradient
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

from config.one_stage_different_ability import config as base_config
from utils.theory import e_star_two_players_different_ability, p_win_different_ability, clip_stage2
from utils.exploit_asymmetric import eval_exploitability_asymmetric
from envs.different_ability_env import DifferentAbilityEnv
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
    filename = f"different_ability_{method_tag}_{q_tag}{episodes_tag}{seed_tag}_{timestamp}.log"
    return os.path.join("results", "different_ability", "logs", filename)


def _clip_effort(value: float, bounds: Tuple[float, float]) -> float:
    """Clip effort to bounds."""
    lo, hi = bounds
    return float(np.clip(value, lo, hi))


# === Gradient-based solver ===

def _batch_payoffs_uniform_da(
    env: DifferentAbilityEnv,
    e1: float,
    e2: float,
    eps1: np.ndarray,
    eps2: np.ndarray,
    tie_breaks: np.ndarray,
) -> Tuple[float, float]:
    """Vectorized sampled mean payoffs using provided Uniform(-q, q) noise (CRN).

    y_i = e_i + l_i + eps_i; the realized winner takes w_H, the loser w_L;
    each player's payoff subtracts k_i * e_i^2.
    """
    y1 = e1 + env.l1 + eps1
    y2 = e2 + env.l2 + eps2
    winners = np.where(y1 > y2, 0, np.where(y2 > y1, 1, tie_breaks))
    payoff1 = np.where(winners == 0, env.w_h, env.w_l)
    payoff2 = np.where(winners == 0, env.w_l, env.w_h)
    u1 = payoff1 - env.k1 * (e1 ** 2)
    u2 = payoff2 - env.k2 * (e2 ** 2)
    return float(u1.mean()), float(u2.mean())


def _compute_gradients_different_ability(
    env: DifferentAbilityEnv,
    e1: float,
    e2: float,
    delta: float,
    num_samples: int,
) -> Tuple[float, float]:
    """MC-FD gradients (Appendix A): central differences on SAMPLED payoffs.

    One shared batch of uniform noise draws + tie-breaks (common random
    numbers) is reused for all four perturbed evaluations, so each central
    difference is taken under identical randomness. Mirrors
    _stochastic_fd_gradients in run_two_players.py with the ability-shifted
    outputs y_i = e_i + l_i + eps_i.

    Args:
        env: DifferentAbilityEnv instance
        e1: Current effort for player 1
        e2: Current effort for player 2
        delta: Finite difference step size
        num_samples: Number of Monte Carlo samples

    Returns:
        (g1, g2): Gradients for player 1 and player 2
    """
    lo, hi = env.effort_range
    eps1, eps2, tie_breaks = env.draw_noise_batch(num_samples)

    e1_plus = _clip_effort(e1 + delta, (lo, hi))
    e1_minus = _clip_effort(e1 - delta, (lo, hi))
    e2_plus = _clip_effort(e2 + delta, (lo, hi))
    e2_minus = _clip_effort(e2 - delta, (lo, hi))

    u1_plus, _ = _batch_payoffs_uniform_da(env, e1_plus, e2, eps1, eps2, tie_breaks)
    u1_minus, _ = _batch_payoffs_uniform_da(env, e1_minus, e2, eps1, eps2, tie_breaks)
    _, u2_plus = _batch_payoffs_uniform_da(env, e1, e2_plus, eps1, eps2, tie_breaks)
    _, u2_minus = _batch_payoffs_uniform_da(env, e1, e2_minus, eps1, eps2, tie_breaks)

    g1 = (u1_plus - u1_minus) / (2.0 * delta)
    g2 = (u2_plus - u2_minus) / (2.0 * delta)
    return float(g1), float(g2)


def gradient_descent_different_ability(
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
    MC-FD gradient play (Appendix A) for the different-ability game: sampled
    payoffs with common random numbers, central finite differences (step
    ``eps``), projected gradient ascent, simultaneous updates, tolerance
    ``tol``. No closed-form win probability and no e*-dependent stopping —
    gaps to theory are logged for evaluation only. The equilibrium is
    symmetric, but symmetry is never enforced.

    Args:
        cfg: Configuration dictionary with l1, l2, k, q, w_h, w_l, effort_range
        lr: Initial learning rate
        steps: Maximum gradient steps
        eps: Finite difference delta
        tol: Convergence tolerance
        num_samples: MC samples per CRN batch
        init_perturb: Initial perturbation from theory
        lr_decay: Learning rate decay per step
        log: Whether to print progress

    Returns:
        ((e1, e2), history): Final efforts and convergence history
    """
    effort_bounds = tuple(cfg["effort_range"])
    
    # Create environment
    env_config = {
        "l1": cfg["l1"],
        "l2": cfg["l2"],
        "k": cfg["k"],
        "k1": cfg["k"],
        "k2": cfg["k"],
        "q": cfg["q"],
        "w_h": cfg["w_h"],
        "w_l": cfg["w_l"],
        "effort_range": effort_bounds,
        "seed": cfg.get("seed", 42),
    }
    env = DifferentAbilityEnv(env_config)
    
    if eps <= 0:
        raise ValueError("grad_eps must be positive for finite differences")
    
    lo, hi = effort_bounds
    
    # Get theoretical equilibrium effort (for eval logging only)
    e_star = e_star_two_players_different_ability(
        cfg["q"], cfg["w_h"], cfg["w_l"], cfg["k"], cfg["l1"], cfg["l2"]
    )
    e_star = _clip_effort(e_star, effort_bounds)

    # Start at fixed fractions of effort range (no e* dependency)
    e1 = _clip_effort(lo + (hi - lo) * 0.3, effort_bounds)
    e2 = _clip_effort(lo + (hi - lo) * 0.7, effort_bounds)
    
    history = {
        "init_e1": e1,
        "init_e2": e2,
        "e_star": e_star,
        "final_grad": 0.0,
        "iterations": 0,
        "e1_history": [float(e1)],
        "e2_history": [float(e2)],
        "gap1_history": [abs(e1 - e_star)],
        "gap2_history": [abs(e2 - e_star)],
        "step_history": [0],
    }
    
    for step in range(1, steps + 1):
        # Adaptive learning rate with exponential decay
        lr_current = lr * (lr_decay ** step)
        
        # Sampled MC-FD gradient with common random numbers (Appendix A)
        g1, g2 = _compute_gradients_different_ability(env, e1, e2, eps, num_samples)
        
        # Gradient ascent update
        e1_new = _clip_effort(e1 + lr_current * g1, effort_bounds)
        e2_new = _clip_effort(e2 + lr_current * g2, effort_bounds)
        
        delta_e1 = abs(e1_new - e1)
        delta_e2 = abs(e2_new - e2)
        grad_norm = max(abs(g1), abs(g2))
        
        e1, e2 = e1_new, e2_new
        
        # Compute gaps to theoretical value
        gap1 = abs(e1 - e_star)
        gap2 = abs(e2 - e_star)
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
                f"[gradient-diff-ability] step={step:05d} e1={e1:.6f} e2={e2:.6f} "
                f"grad=({g1:.6f},{g2:.6f}) gap1={gap1:.4f} gap2={gap2:.4f} "
                f"lr={lr_current:.6f}"
            )
        
        # Tolerance tau: gradient estimate and update step both below tol.
        # (Gap to theoretical e* is logged above but is NOT a stop criterion —
        # the baseline must not condition its termination on the answer.)
        max_delta = max(delta_e1, delta_e2)
        if grad_norm < tol and max_delta < tol:
            if log:
                print(
                    f"[gradient-diff-ability] converged at step={step} "
                    f"grad_norm={grad_norm:.3e} max_gap={max_gap:.3e}"
                )
            break
    
    history["final_e1"] = e1
    history["final_e2"] = e2
    history["final_gap1"] = abs(e1 - e_star)
    history["final_gap2"] = abs(e2 - e_star)
    
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
    """Run gradient-based solver for different ability tournament."""
    l1, l2 = cfg["l1"], cfg["l2"]
    k = cfg["k"]
    w_h, w_l, q = cfg["w_h"], cfg["w_l"], cfg["q"]
    effort_bounds = tuple(cfg["effort_range"])
    
    # Get theoretical effort (same for both players)
    e_star = e_star_two_players_different_ability(q, w_h, w_l, k, l1, l2)
    e_star = clip_stage2(e_star, effort_bounds)
    
    (e1, e2), meta = gradient_descent_different_ability(
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
    
    # For different ability, theoretical equilibrium is symmetric
    gap1 = abs(e1 - e_star)
    gap2 = abs(e2 - e_star)
    max_gap = max(gap1, gap2)
    
    # Win probability at converged efforts
    p1_win = p_win_different_ability(e1, e2, l1, l2, q)
    
    if log:
        grad_tuple = meta.get("final_grad_tuple", (0.0, 0.0))
        print(
            f"[gradient-diff-ability] final e1={e1:.6f} e2={e2:.6f} "
            f"grad=({grad_tuple[0]:.6f},{grad_tuple[1]:.6f})"
        )
        print(
            f"[gradient-diff-ability] theoretical e*={e_star:.6f} "
            f"gap1={gap1:.4f} gap2={gap2:.4f} max_gap={max_gap:.4f}"
        )
        print(f"[gradient-diff-ability] P(p1 wins)={p1_win:.4f}")
    
    # Save convergence history
    if log:
        convergence_data = {
            "algorithm": "gradient",
            "scenario": "different_ability",
            "num_players": 2,
            "q": float(q),
            "k": float(k),
            "l1": float(l1),
            "l2": float(l2),
            "w_h": float(w_h),
            "w_l": float(w_l),
            "theoretical": {
                "effort": float(e_star),  # Symmetric equilibrium
                "p1_win": float(p_win_different_ability(e_star, e_star, l1, l2, q)),
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
                "init_perturb": float(init_perturb),
            },
            "final": {
                "effort1": float(e1),
                "effort2": float(e2),
                "gap1": float(gap1),
                "gap2": float(gap2),
                "max_gap": float(max_gap),
                "p1_win": float(p1_win),
            },
        }
        
        convergence_dir = os.path.join("results", "different_ability", "convergence")
        os.makedirs(convergence_dir, exist_ok=True)
        convergence_file = os.path.join(
            convergence_dir,
            f"different_ability_gradient_q{q:.1f}_convergence.json"
        )
        with open(convergence_file, 'w') as f:
            json.dump(convergence_data, f, indent=2)
        print(f"[gradient-diff-ability] Saved convergence history to {convergence_file}")
    
    return {
        "method": "gradient",
        "scenario": "different_ability",
        "num_players": 2,
        "q": float(q),
        "k": float(k),
        "l1": float(l1),
        "l2": float(l2),
        "theoretical_effort": float(e_star),
        "final_effort1": float(e1),
        "final_effort2": float(e2),
        "gap1": float(gap1),
        "gap2": float(gap2),
        "max_gap": float(max_gap),
        "p1_win": float(p1_win),
        "iterations": int(meta["iterations"]),
    }


# === PPO training ===

class CheapGateTracker:
    """Rolling-window tracker for cheap stability metrics (KL + policy drift)."""

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.kl_hist: deque[float] = deque(maxlen=window_size)
        self.policy_hist: deque[float] = deque(maxlen=window_size)

    def update(
        self,
        approx_kl: Optional[float],
        policy_mean_effort: Optional[float],
    ) -> None:
        if approx_kl is not None and math.isfinite(approx_kl):
            self.kl_hist.append(float(approx_kl))
        if policy_mean_effort is not None and math.isfinite(policy_mean_effort):
            self.policy_hist.append(float(policy_mean_effort))

    def compute(self) -> dict:
        if len(self.kl_hist) < self.window_size:
            return {
                "mean_kl_window": None,
                "std_kl_window": None,
                "drift_effort": None,
            }
        kl_vals = list(self.kl_hist)
        p_vals = list(self.policy_hist)
        mean_kl = float(np.mean(kl_vals))
        std_kl = float(np.std(kl_vals))
        drift = abs(p_vals[-1] - p_vals[0]) if len(p_vals) >= self.window_size else None
        return {
            "mean_kl_window": mean_kl,
            "std_kl_window": std_kl,
            "drift_effort": drift,
        }


def run_ppo_different_ability(
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
    Train PPO for different ability scenario using a single shared agent.
    
    Since the equilibrium is symmetric (both players exert same effort),
    we use a single PPO agent that learns the optimal policy for both.
    The agent state encodes [q, k, w_gap]; ability affects environment dynamics.
    
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

    l1, l2 = cfg["l1"], cfg["l2"]
    k = cfg["k"]
    w_h, w_l = cfg["w_h"], cfg["w_l"]
    effort_bounds = tuple(cfg["effort_range"])
    
    # Training configuration
    train_qs = list(train_qs if train_qs is not None else cfg["q_list"])
    
    # Theory-align settings
    theory_align_v2_enabled = bool(cfg.get("theory_align_v2", False))
    theory_align_v2_conc_min = float(cfg.get("theory_align_v2_conc_min", 1.0))
    theory_align_v2_conc_scale = float(cfg.get("theory_align_v2_conc_scale", 1.0))
    theory_align_v2_conc_max = cfg.get("theory_align_v2_conc_max", None)
    theory_align_v2_conc_min_start = float(cfg.get("theory_align_v2_conc_min_start", theory_align_v2_conc_min))
    theory_align_v2_conc_scale_start = float(cfg.get("theory_align_v2_conc_scale_start", theory_align_v2_conc_scale))
    theory_align_v2_var_coef = float(cfg.get("theory_align_v2_var_coef", 0.0))
    theory_align_v2_var_coef_start = float(cfg.get("theory_align_v2_var_coef_start", theory_align_v2_var_coef))
    theory_align_v2_ramp_warmup = int(cfg.get("theory_align_v2_ramp_warmup", 0))
    theory_align_v2_ramp_steps = int(cfg.get("theory_align_v2_ramp_steps", 0))

    # PPO configuration (single shared agent)
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
        theory_align_v2_conc_min=theory_align_v2_conc_min,
        theory_align_v2_conc_scale=theory_align_v2_conc_scale,
        theory_align_v2_conc_max=theory_align_v2_conc_max,
    )
    
    # Create single shared agent (symmetric equilibrium)
    agent = PPOTwoPlayersBandit(effort_bounds=effort_bounds, cfg=ppo_cfg)
    
    # Initialize schedules
    agent.cfg.entropy_coef = float(cfg.get("entropy_coef_start", agent.cfg.entropy_coef))
    agent.cfg.clip_eps = float(cfg.get("clip_range_start", agent.cfg.clip_eps))
    for g in agent.opt.param_groups:
        g["lr"] = float(cfg.get("lr_start", ppo_cfg.lr))
    
    print(f"[PPO-diff-ability] Single-agent self-play mode: l1={l1}, l2={l2}, k={k}")
    print(f"[PPO-diff-ability] Training on q values: {train_qs}")
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
    
    # Exploitability tracking
    joint_exploit_ok_streak = 0
    stop_reason = "max_updates"  # default if we exhaust episodes
    min_updates = int(cfg.get("min_updates", 0))
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
    env_config = {
        "l1": l1,
        "l2": l2,
        "k": k,
        "k1": k,
        "k2": k,
        "q": q_init,
        "w_h": w_h,
        "w_l": w_l,
        "effort_range": effort_bounds,
        "seed": cfg.get("seed", 42),
    }
    env = DifferentAbilityEnv(env_config)
    
    # Convergence history for plotting
    convergence_history: Dict[str, Any] = {
        "steps": [],
        "effort": [],
        "gap": [],
        "approx_kl": [],
        "batch_entropy": [],
    }
    
    # Exploitability history (sparse, only when evaluated)
    exploit_history: List[Dict[str, Any]] = []
    
    # Print exploitability settings
    if not disable_exploitability:
        print(
            f"[PPO-diff-ability] Exploitability: eps={exploit_eps}, patience={patience_exploit}, "
            f"every={exploit_every_updates}, M={exploit_M}, disable_gate={disable_cheap_gate}",
            flush=True,
        )
    
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

        # Concentration ramp schedule (ported from run_two_players.py:889-910)
        if theory_align_v2_enabled:
            warmup = max(0, int(theory_align_v2_ramp_warmup))
            ramp_steps = max(0, int(theory_align_v2_ramp_steps))
            if update_idx < warmup:
                ramp_t = 0.0
            elif ramp_steps <= 0:
                ramp_t = 1.0
            else:
                ramp_t = float(update_idx - warmup + 1) / float(ramp_steps)
                ramp_t = max(0.0, min(1.0, ramp_t))
            conc_min = theory_align_v2_conc_min_start + (theory_align_v2_conc_min - theory_align_v2_conc_min_start) * ramp_t
            conc_scale = theory_align_v2_conc_scale_start + (theory_align_v2_conc_scale - theory_align_v2_conc_scale_start) * ramp_t
            var_coef = theory_align_v2_var_coef_start + (theory_align_v2_var_coef - theory_align_v2_var_coef_start) * ramp_t
            if hasattr(agent.net, "conc_min"):
                agent.net.conc_min = float(conc_min)
            if hasattr(agent.net, "conc_scale"):
                agent.net.conc_scale = float(conc_scale)
            if hasattr(agent.opponent_policy, "conc_min"):
                agent.opponent_policy.conc_min = float(conc_min)
            if hasattr(agent.opponent_policy, "conc_scale"):
                agent.opponent_policy.conc_scale = float(conc_scale)
            agent.cfg.theory_align_v2_var_coef = float(var_coef)

        # Collect rollout
        steps_this = min(ppo_cfg.steps_per_update, total_steps_target - steps_done)
        
        for _ in range(steps_this):
            q = float(rng.choice(train_qs))
            env.q = q
            
            # Generate state (same for both players since symmetric policy)
            state = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
            
            # Sample actions from shared policy (both players use same agent)
            a1_norm, e1, logp1, v1 = agent.act(state)
            a2_norm, e2, logp2, v2 = agent.act(state)
            
            # Execute environment step
            actions = [
                torch.tensor([float(e1.item())]),
                torch.tensor([float(e2.item())]),
            ]
            _, rewards, _, done, _ = env.step(actions)
            
            # Store transitions (train on both players' experiences)
            agent.store(state, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
            agent.store(state, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
            
            steps_done += 1
        
        # PPO update
        metrics = agent.update()
        
        # Compute policy mean effort at test q
        test_q = float(train_qs[0])
        test_state = agent.state_from_params(q=test_q, k=k, w_h=w_h, w_l=w_l)
        policy_mean = agent.mean_effort(test_state)
        
        # Get theoretical effort
        e_star = e_star_two_players_different_ability(test_q, w_h, w_l, k, l1, l2)
        gap = abs(policy_mean - e_star)
        
        # Update convergence tracker
        if cheap_tracker is not None:
            cheap_tracker.update(metrics.get("approx_kl", 0), policy_mean)
        
        # Record convergence history
        convergence_history["steps"].append(steps_done)
        convergence_history["effort"].append(policy_mean)
        convergence_history["gap"].append(gap)
        convergence_history["approx_kl"].append(metrics.get("approx_kl", 0.0))
        convergence_history["batch_entropy"].append(metrics.get("batch_entropy", 0.0))
        
        # Logging
        if update_idx % 20 == 0 or update_idx == total_updates - 1:
            print(
                f"[PPO-diff-ability] update={update_idx:04d} steps={steps_done:08d} "
                f"e={policy_mean:.4f} (theory={e_star:.4f}, gap={gap:.4f}) "
                f"entropy={agent.cfg.entropy_coef:.4f}",
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
                    drift = cheap_stats.get("drift_effort")
                    
                    # Cheap gate thresholds (from config or defaults)
                    mean_thresh = float(cheap_cfg.get("mean_kl_thresh", 0.012))
                    std_thresh = float(cheap_cfg.get("std_kl_thresh", 0.0035))
                    drift_thresh = float(cheap_cfg.get("drift_effort_thresh", 2.0))
                    
                    mean_ok = mean_kl is not None and mean_kl <= mean_thresh
                    std_ok = std_kl is not None and std_kl <= std_thresh
                    drift_ok = drift is not None and drift <= drift_thresh
                    
                    gate_pass = mean_ok and std_ok and drift_ok
                    should_eval_exploit = gate_pass or (updates_since_exploit_eval >= exploit_every_updates)
                else:
                    # No tracker - evaluate on schedule
                    should_eval_exploit = (updates_since_exploit_eval >= exploit_every_updates)
            
            if should_eval_exploit:
                updates_since_exploit_eval = 0
                
                # Evaluate exploitability for both players
                # Note: Different-ability uses same agent for both, but different ability params
                exploit_result = eval_exploitability_asymmetric(
                    agent1=agent,  # Same agent for both (symmetric policy)
                    agent2=agent,
                    q=test_q,
                    effort_bounds=effort_bounds,
                    M=exploit_M,
                    grid_cfg=exploit_grid_cfg,
                    seed=cfg.get("seed", 42) + update_idx,  # Vary seed per eval
                    w_h=w_h,
                    w_l=w_l,
                    k1=k,  # Same cost for both
                    k2=k,
                    l1=l1,  # Different abilities
                    l2=l2,
                    game_type="different_ability",
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
                    if update_idx < min_updates:
                        print(
                            f"[Convergence] Exploitability streak satisfied at upd={update_idx}, "
                            f"but min_updates={min_updates} not reached; continuing.",
                            flush=True,
                        )
                    else:
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
        test_state = agent.state_from_params(q=float(q), k=k, w_h=w_h, w_l=w_l)
        policy_mean = agent.mean_effort(test_state)
        e_star = e_star_two_players_different_ability(float(q), w_h, w_l, k, l1, l2)
        gap = abs(policy_mean - e_star)
        
        # Win probability at policy mean (symmetric)
        p1_win = p_win_different_ability(policy_mean, policy_mean, l1, l2, q)
        
        print(
            f"[PPO-diff-ability] Final q={q:.1f}: "
            f"e={policy_mean:.4f} (theory={e_star:.4f}, gap={gap:.4f}) "
            f"P(p1 wins)={p1_win:.4f}"
        )
        
        result = {
            "method": "ppo",
            "scenario": "different_ability",
            "num_players": 2,
            "q": float(q),
            "k": float(k),
            "l1": float(l1),
            "l2": float(l2),
            "w_h": float(w_h),
            "w_l": float(w_l),
            "theoretical_effort": float(e_star),
            "final_effort": float(policy_mean),
            "gap": float(gap),
            "p1_win": float(p1_win),
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
            "scenario": "different_ability",
            "num_players": 2,
            "q": float(q),
            "k": float(k),
            "l1": float(l1),
            "l2": float(l2),
            "w_h": float(w_h),
            "w_l": float(w_l),
            "seed": int(seed_val),
            "theoretical": {
                "effort": float(e_star),
                "p1_win": float(p_win_different_ability(e_star, e_star, l1, l2, q)),
            },
            "ablation_name": ablation_name,
            "history": convergence_history,
            "exploit_history": exploit_history,
            "final": {
                "effort": float(policy_mean),
                "gap": float(gap),
                "p1_win": float(p1_win),
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
        
        convergence_dir = os.path.join("results", "different_ability", "convergence")
        os.makedirs(convergence_dir, exist_ok=True)
        convergence_file = os.path.join(
            convergence_dir,
            f"different_ability_ppo_q{q:.1f}_seed{seed_val}_{ablation_name}_convergence.json"
        )
        with open(convergence_file, 'w') as f:
            json.dump(convergence_data, f, indent=2)
        print(f"[PPO-diff-ability] Saved convergence history to {convergence_file}")
    
    # Save results to CSV
    _save_results_csv(results, cfg)
    
    return results


def _save_results_csv(results: List[Dict], cfg: Dict) -> None:
    """Save results to CSV file."""
    csv_path = os.path.join("results", "different_ability", "summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    fieldnames = [
        "q", "k", "l1", "l2", "w_h", "w_l", "seed",
        "theoretical_effort", "final_effort", "gap", "p1_win",
        "method", "episodes", "converged",
    ]
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        for r in results:
            # Determine if converged (gap < threshold)
            converged = r.get("gap", float("inf")) < cfg.get("convergence_rel_err_threshold", 0.10) * r.get("theoretical_effort", 1)
            
            row = {
                "q": r.get("q"),
                "k": r.get("k"),
                "l1": r.get("l1"),
                "l2": r.get("l2"),
                "w_h": r.get("w_h"),
                "w_l": r.get("w_l"),
                "seed": cfg.get("seed", 42),
                "theoretical_effort": r.get("theoretical_effort"),
                "final_effort": r.get("final_effort") or r.get("final_effort1"),
                "gap": r.get("gap") or r.get("max_gap"),
                "p1_win": r.get("p1_win"),
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
    if args.l1 is not None:
        cfg["l1"] = float(args.l1)
    if args.l2 is not None:
        cfg["l2"] = float(args.l2)
    if args.k is not None:
        cfg["k"] = float(args.k)
        cfg["k1"] = float(args.k)
        cfg["k2"] = float(args.k)
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
        cfg["theory_align_v2_conc_min"] = 1000.0
        cfg["theory_align_v2_conc_scale"] = 10000.0
        cfg["theory_align_v2_conc_max"] = 100000.0
        cfg["theory_align_v2_var_coef"] = 5e-2
        cfg["theory_align_v2_conc_min_start"] = 100.0
        cfg["theory_align_v2_conc_scale_start"] = 100.0
        cfg["theory_align_v2_var_coef_start"] = 0.0
        cfg["theory_align_v2_ramp_warmup"] = 20
        cfg["theory_align_v2_ramp_steps"] = 50
        cfg["lr_start"] = 5e-5
        cfg["lr_end"] = 2e-5
        cfg["update_epochs"] = 1
        cfg["clip_range_start"] = 0.2
        cfg["clip_range_end"] = 0.15
        cfg["target_kl"] = 0.06
        cfg["ratio_stop_threshold"] = 2.2
        cfg["max_grad_norm"] = 0.25
        cfg["value_coef"] = 1.0
        print(
            "[TheoryAlignV2] enabled: entropy=0, mean+conc head, var_coef=5e-2, "
            "conc_min=1000, conc_scale=10000, conc_max=100000, ramp_warmup=20, ramp_steps=50, "
            "lr/clip/epochs softened",
            flush=True,
        )

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

    # --override-conc-ramp-warmup: override concentration ramp warmup
    if hasattr(args, 'override_conc_ramp_warmup') and args.override_conc_ramp_warmup is not None:
        cfg["theory_align_v2_ramp_warmup"] = int(args.override_conc_ramp_warmup)
        print(f"[config] conc_ramp_warmup override: {args.override_conc_ramp_warmup}", flush=True)

    # --min-updates: minimum updates before early stop
    if hasattr(args, 'min_updates') and args.min_updates > 0:
        cfg["min_updates"] = int(args.min_updates)
        print(f"[config] min_updates: {args.min_updates}", flush=True)

    # Convergence eval settings
    if "convergence" not in cfg:
        cfg["convergence"] = {}
    if args.enable_convergence_eval:
        cfg["convergence"]["enabled"] = True
    if args.cheap_gate_profile is not None:
        cfg["convergence"]["cheap_gate_profile"] = args.cheap_gate_profile
    
    # Print configuration
    print(f"[config] Different Ability Experiment Configuration:")
    print(f"  l1={cfg['l1']}, l2={cfg['l2']} (Δl={cfg['l1'] - cfg['l2']})")
    print(f"  k={cfg['k']}")
    print(f"  w_h={cfg['w_h']}, w_l={cfg['w_l']}")
    print(f"  effort_range={cfg['effort_range']}")
    print(f"  seed={cfg.get('seed', 42)}")
    
    if args.method == "gradient":
        q_values = [args.q] if args.q is not None else list(cfg["q_list"])
        for q in q_values:
            cfg["q"] = float(q)
            
            # Print theoretical effort
            e_star = e_star_two_players_different_ability(
                q, cfg["w_h"], cfg["w_l"], cfg["k"], cfg["l1"], cfg["l2"]
            )
            p1_win = p_win_different_ability(e_star, e_star, cfg["l1"], cfg["l2"], q)
            print(f"\n[q={q}] Theoretical: e*={e_star:.4f}, P(p1 wins)={p1_win:.4f}")
            
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
            e_star = e_star_two_players_different_ability(
                q, cfg["w_h"], cfg["w_l"], cfg["k"], cfg["l1"], cfg["l2"]
            )
            p1_win = p_win_different_ability(e_star, e_star, cfg["l1"], cfg["l2"], q)
            print(f"  q={q}: e*={e_star:.4f}, P(p1 wins)={p1_win:.4f}")
        print()
        
        run_ppo_different_ability(
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
        description="One-Stage Two-Player Different Ability Experiment (l1 > l2, k1 = k2)"
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
    parser.add_argument("--grad-samples", type=int, default=base_config.get("gradient_num_samples", 64),
                        help="Monte Carlo samples per CRN batch for uniform-noise gradients.")
    parser.add_argument("--grad-init-perturb", type=float, default=base_config.get("gradient_init_perturb", 1.0))
    parser.add_argument("--grad-lr-decay", type=float, default=0.9995)
    
    # Ability/cost parameter overrides
    parser.add_argument("--l1", type=float, help="Override player 1 ability parameter l1")
    parser.add_argument("--l2", type=float, help="Override player 2 ability parameter l2")
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
    parser.add_argument(
        "--override-entropy-end",
        type=float,
        default=None,
        help="Override entropy_coef_end (takes priority over --theory-align-v2 defaults).",
    )
    parser.add_argument(
        "--override-conc-ramp-warmup",
        type=int,
        default=None,
        help="Override theory_align_v2_ramp_warmup (updates before concentration ramp begins).",
    )
    parser.add_argument(
        "--min-updates",
        type=int,
        default=0,
        help="Minimum updates before exploitability-based early stop is allowed.",
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
        help="Override exploit_eps threshold (default: from config, 0.03)",
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
