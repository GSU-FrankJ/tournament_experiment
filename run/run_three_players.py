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
    return os.path.join("results", "three_players", "logs", filename)


def _clip_effort(value: float, bounds: tuple[float, float]) -> float:
    """Clip effort to bounds."""
    lo, hi = bounds
    return float(np.clip(value, lo, hi))


# === Three-player exploitability evaluation ===

def _sample_policy_efforts_3p(
    agent,
    q: float,
    effort_bounds: tuple[float, float],
    M: int,
    seed: int,
    *,
    k: float,
    w_h: float,
    w_l: float,
) -> torch.Tensor:
    """Sample M efforts from the agent's current policy for given game params.
    
    Uses deterministic seeding per call for reproducibility (CRN).
    """
    device = next(agent.net.parameters()).device
    gen_key = (int(seed), device.type, getattr(device, "index", None))
    gen_map = getattr(agent, "_policy_effort_generators", None)
    if gen_map is None:
        gen_map = {}
        setattr(agent, "_policy_effort_generators", gen_map)
    generator = gen_map.get(gen_key)
    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        gen_map[gen_key] = generator
    state = agent.state_from_params(q=float(q), k=k, w_h=w_h, w_l=w_l)
    with torch.no_grad():
        dist, _ = agent.dist(state)
        try:
            samples = dist.sample((M,), generator=generator)
        except TypeError:
            # Fallback: derive per-call seed from generator for older torch
            seed_gen_map = getattr(agent, "_policy_effort_seed_generators", None)
            if seed_gen_map is None:
                seed_gen_map = {}
                setattr(agent, "_policy_effort_seed_generators", seed_gen_map)
            seed_gen = seed_gen_map.get(gen_key)
            if seed_gen is None:
                seed_gen = torch.Generator(device="cpu")
                seed_gen.manual_seed(int(seed))
                seed_gen_map[gen_key] = seed_gen
            seed_value = int(torch.randint(0, 2**31 - 1, (1,), generator=seed_gen).item())
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed_value)
                samples = dist.sample((M,))
        samples = samples.reshape(M).clamp(0.0, 1.0)
    low, high = effort_bounds
    return (low + samples * (high - low)).to(device)


def _payoff_player1_3p(
    e1: torch.Tensor,
    e2: torch.Tensor,
    e3: torch.Tensor,
    eps1: torch.Tensor,
    eps2: torch.Tensor,
    eps3: torch.Tensor,
    *,
    w_h: float,
    w_l: float,
    k: float,
) -> torch.Tensor:
    """Vectorized payoff for player 1 in a three-player tournament.
    
    Winner (highest score y_i = e_i + eps_i) gets w_h, two losers get w_l.
    Cost for player 1 is k * e1^2.
    Ties broken randomly via stacking order + argmax (favors lower index).
    """
    y1 = e1 + eps1
    y2 = e2 + eps2
    y3 = e3 + eps3
    # Stack scores: shape (3, M). argmax returns index of winner.
    scores = torch.stack([y1, y2, y3], dim=0)  # (3, M)
    winners = torch.argmax(scores, dim=0)  # (M,)
    payoff1 = torch.where(winners == 0, w_h, w_l) - k * (e1 ** 2)
    return payoff1


def eval_exploitability_3p(
    agent,
    q: float,
    effort_bounds: tuple[float, float],
    *,
    M: int,
    grid_cfg: dict,
    seed: int,
    w_h: float,
    w_l: float,
    k: float,
) -> dict:
    """Monte Carlo exploitability evaluation for three-player tournament.
    
    Measures how much utility player 1 can gain by unilaterally deviating
    from the shared policy while players 2 and 3 follow it.
    
    Uses Common Random Numbers (CRN) and coarse-to-fine grid search.
    
    Returns:
        {
            "exploitability": float,  # max(u_dev - u_self, 0)
            "best_dev_effort": float,
            "u_dev": float,
            "u_selfplay": float,
            "num_candidates": int,
        }
    """
    device = next(agent.net.parameters()).device
    low, high = effort_bounds
    with torch.no_grad():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            # Common random numbers (CRN) for all candidates
            eps1 = (torch.rand(M, device=device) * 2.0 - 1.0) * q
            eps2 = (torch.rand(M, device=device) * 2.0 - 1.0) * q
            eps3 = (torch.rand(M, device=device) * 2.0 - 1.0) * q

            # Sample policy efforts for self-play (3 independent draws)
            efforts_p1 = _sample_policy_efforts_3p(agent, q, effort_bounds, M, seed=seed, k=k, w_h=w_h, w_l=w_l)
            efforts_p2 = _sample_policy_efforts_3p(agent, q, effort_bounds, M, seed=seed + 17, k=k, w_h=w_h, w_l=w_l)
            efforts_p3 = _sample_policy_efforts_3p(agent, q, effort_bounds, M, seed=seed + 31, k=k, w_h=w_h, w_l=w_l)

        # U_self = E[u_1(pi, pi, pi)] — player 1's payoff under self-play
        u_self = _payoff_player1_3p(
            efforts_p1, efforts_p2, efforts_p3, eps1, eps2, eps3,
            w_h=w_h, w_l=w_l, k=k,
        ).mean().item()

        # Coarse-to-fine grid search for best deviation effort
        stage_a = torch.arange(low, high + 1e-6, grid_cfg.get("stage_a_step", 5.0), device=device)

        def _around(center: float, radius: float, step: float):
            lo_bound = max(low, center - radius)
            hi_bound = min(high, center + radius)
            return torch.arange(lo_bound, hi_bound + 1e-6, step, device=device)

        best_delta = -float("inf")
        best_e = float(low)
        num_candidates = 0

        def eval_candidates(candidates: torch.Tensor):
            nonlocal best_delta, best_e, num_candidates
            for e_dev in candidates:
                e_dev_val = e_dev.item()
                efforts_dev = torch.full((M,), e_dev_val, device=device)
                # Deviator plays e_dev against two policy opponents
                payoff_dev = _payoff_player1_3p(
                    efforts_dev, efforts_p2, efforts_p3, eps1, eps2, eps3,
                    w_h=w_h, w_l=w_l, k=k,
                )
                delta = payoff_dev.mean().item() - u_self
                num_candidates += 1
                if delta > best_delta:
                    best_delta = delta
                    best_e = e_dev_val

        eval_candidates(stage_a)
        stage_b = _around(best_e, grid_cfg.get("stage_b_radius", 15.0), grid_cfg.get("stage_b_step", 1.0))
        eval_candidates(stage_b)
        stage_c = _around(best_e, grid_cfg.get("stage_c_radius", 3.0), grid_cfg.get("stage_c_step", 0.25))
        eval_candidates(stage_c)

    return {
        "exploitability": float(max(best_delta, 0.0)),
        "best_dev_effort": float(best_e),
        "u_dev": float(u_self + best_delta),
        "u_selfplay": float(u_self),
        "num_candidates": int(num_candidates),
    }


# === Gradient-based solver ===

def _batch_payoffs_uniform_3p(
    env: ThreePlayersEnv,
    e1: float,
    e2: float,
    e3: float,
    eps: np.ndarray,
    tie_breaks: np.ndarray,
) -> tuple[float, float, float]:
    """Vectorized sampled mean payoffs for all three players under provided noise (CRN).

    y_i = e_i + eps_i; the realized winner takes w_H, both losers w_L; each
    player's payoff subtracts k*e_i^2. Exact output ties (probability zero
    under continuous noise) are resolved with the provided tie-break draws.
    """
    efforts = np.array([e1, e2, e3], dtype=float)
    y = efforts[None, :] + eps  # (batch, 3)
    winners = np.argmax(y, axis=1)
    is_max = y == y.max(axis=1, keepdims=True)
    tie_rows = np.flatnonzero(is_max.sum(axis=1) > 1)
    for r in tie_rows:
        idxs = np.flatnonzero(is_max[r])
        winners[r] = idxs[int(tie_breaks[r]) % len(idxs)]
    costs = env.k * efforts ** 2
    u1 = float(np.where(winners == 0, env.w_h, env.w_l).mean() - costs[0])
    u2 = float(np.where(winners == 1, env.w_h, env.w_l).mean() - costs[1])
    u3 = float(np.where(winners == 2, env.w_h, env.w_l).mean() - costs[2])
    return u1, u2, u3


def _stochastic_fd_gradients_3p(
    env: ThreePlayersEnv,
    e1: float,
    e2: float,
    e3: float,
    delta: float,
    num_samples: int,
) -> tuple[float, float, float]:
    """MC-FD gradients (Appendix A): central differences on SAMPLED payoffs.

    One shared batch of uniform noise draws + tie-breaks (common random
    numbers) is reused for all six perturbed evaluations, so each central
    difference is taken under identical randomness. Mirrors
    _stochastic_fd_gradients in run_two_players.py for the 3-player game.
    """
    eps, tie_breaks = env.draw_noise_batch(num_samples)
    lo, hi = env.effort_range

    e1_plus = _clip_effort(e1 + delta, (lo, hi))
    e1_minus = _clip_effort(e1 - delta, (lo, hi))
    e2_plus = _clip_effort(e2 + delta, (lo, hi))
    e2_minus = _clip_effort(e2 - delta, (lo, hi))
    e3_plus = _clip_effort(e3 + delta, (lo, hi))
    e3_minus = _clip_effort(e3 - delta, (lo, hi))

    u1_plus, _, _ = _batch_payoffs_uniform_3p(env, e1_plus, e2, e3, eps, tie_breaks)
    u1_minus, _, _ = _batch_payoffs_uniform_3p(env, e1_minus, e2, e3, eps, tie_breaks)
    _, u2_plus, _ = _batch_payoffs_uniform_3p(env, e1, e2_plus, e3, eps, tie_breaks)
    _, u2_minus, _ = _batch_payoffs_uniform_3p(env, e1, e2_minus, e3, eps, tie_breaks)
    _, _, u3_plus = _batch_payoffs_uniform_3p(env, e1, e2, e3_plus, eps, tie_breaks)
    _, _, u3_minus = _batch_payoffs_uniform_3p(env, e1, e2, e3_minus, eps, tie_breaks)

    g1 = (u1_plus - u1_minus) / (2.0 * delta)
    g2 = (u2_plus - u2_minus) / (2.0 * delta)
    g3 = (u3_plus - u3_minus) / (2.0 * delta)
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
    log: bool = True,
) -> tuple[tuple[float, float, float], Dict[str, Any]]:
    """MC-FD gradient play (Appendix A), three-player variant: sampled payoffs
    with common random numbers, central finite differences (step ``eps``),
    projected gradient ascent, simultaneous updates, tolerance ``tol``. No
    closed-form win probability and no symmetry projection — symmetry is only
    reported."""
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
    # Start at fixed fractions of effort range (no e* dependency)
    e1 = _clip_effort(lo + (hi - lo) * 0.25, effort_bounds)
    e2 = _clip_effort(lo + (hi - lo) * 0.50, effort_bounds)
    e3 = _clip_effort(lo + (hi - lo) * 0.75, effort_bounds)

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

        # Sampled MC-FD gradient with common random numbers (Appendix A)
        g1, g2, g3 = _stochastic_fd_gradients_3p(env, e1, e2, e3, delta=eps, num_samples=num_samples)

        e1_new = _clip_effort(e1 + lr_current * g1, effort_bounds)
        e2_new = _clip_effort(e2 + lr_current * g2, effort_bounds)
        e3_new = _clip_effort(e3 + lr_current * g3, effort_bounds)

        delta_e1 = abs(e1_new - e1)
        delta_e2 = abs(e2_new - e2)
        delta_e3 = abs(e3_new - e3)
        grad_norm = max(abs(g1), abs(g2), abs(g3))

        # Simultaneous (projected) update
        e1, e2, e3 = e1_new, e2_new, e3_new

        history["iterations"] = float(step)
        history["final_grad"] = float(grad_norm)
        history["final_grad_tuple"] = (float(g1), float(g2), float(g3))
        
        # Record convergence history
        history["e1_history"].append(float(e1))
        history["e2_history"].append(float(e2))
        history["e3_history"].append(float(e3))
        history["step_history"].append(step)
        
        # Symmetry gap is REPORTED only (never enforced, never a stop criterion)
        symmetry_gap = max(abs(e1 - e2), abs(e1 - e3), abs(e2 - e3))
        max_delta = max(delta_e1, delta_e2, delta_e3)

        if log and (step == 1 or step % 250 == 0 or step == steps):
            print(
                f"[gradient-3p] step={step:05d} e1={e1:.6f} e2={e2:.6f} e3={e3:.6f} "
                f"grad=({g1:.6f},{g2:.6f},{g3:.6f}) delta=({delta_e1:.3e},{delta_e2:.3e},{delta_e3:.3e}) "
                f"lr={lr_current:.6f} sym_gap={symmetry_gap:.4f}"
            )

        # Tolerance tau: gradient estimate and update step both below tol
        if grad_norm < tol and max_delta < tol:
            if log:
                print(
                    f"[gradient-3p] converged at step={step} "
                    f"grad_norm={grad_norm:.3e} sym_gap={symmetry_gap:.3e}"
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
    log: bool = True,
    ablation_name: str = "baseline",
    force: bool = False,
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
        
        convergence_dir = os.path.join("results", "three_players", "convergence")
        os.makedirs(convergence_dir, exist_ok=True)
        # Seed- and tag-qualified filename (mirrors the PPO scheme) so distinct
        # gradient runs never share a file
        seed_val = int(cfg.get("seed", 42))
        convergence_data["seed"] = seed_val
        parts = [f"gradient_3p_q{q:.1f}", f"seed{seed_val}"]
        if ablation_name not in ("baseline", "", None):
            parts.append(ablation_name)
        convergence_file = os.path.join(
            convergence_dir, "_".join(parts) + "_convergence.json"
        )
        if os.path.exists(convergence_file) and not force:
            raise FileExistsError(
                f"Refusing to overwrite existing gradient result: {convergence_file}. "
                "Pass --force to overwrite, or distinguish the run via --seed/--ablation-name."
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
    output_tag: Optional[str] = None,
) -> List[Dict]:
    """Train PPO for three-player tournament (pure self-play).

    Args:
        cfg: Configuration dictionary
        episodes: Total training steps (default from config)
        train_qs: List of q values to train on (default from config)
        ablation_name: Tag for this variant
        output_tag: Optional tag inserted into output filename (e.g., 'round3')

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
    theory_align_v2_conc_min = float(cfg.get("theory_align_v2_conc_min", 1.0))
    theory_align_v2_conc_scale = float(cfg.get("theory_align_v2_conc_scale", 1.0))
    theory_align_v2_conc_max = cfg.get("theory_align_v2_conc_max", None)
    theory_align_v2_conc_min_start = float(cfg.get("theory_align_v2_conc_min_start", theory_align_v2_conc_min))
    theory_align_v2_conc_scale_start = float(cfg.get("theory_align_v2_conc_scale_start", theory_align_v2_conc_scale))
    theory_align_v2_var_coef = float(cfg.get("theory_align_v2_var_coef", 0.0))
    theory_align_v2_var_coef_start = float(cfg.get("theory_align_v2_var_coef_start", theory_align_v2_var_coef))
    theory_align_v2_ramp_warmup = int(cfg.get("theory_align_v2_ramp_warmup", 0))
    theory_align_v2_ramp_steps = int(cfg.get("theory_align_v2_ramp_steps", 0))

    # Component-2 mode-conc ramp settings (exploitability-triggered κ ramp)
    mode_conc_ramp_enabled = bool(cfg.get("mode_conc_ramp", False))
    kappa_schedule = [float(x) for x in cfg.get("kappa_schedule", [20.0, 50.0, 100.0, 200.0])]
    ramp_trigger_exp = float(cfg.get("ramp_trigger_exp", 0.05))
    ramp_trigger_patience = int(cfg.get("ramp_trigger_patience", 3))
    kappa_stage_hold = int(cfg.get("kappa_stage_hold", 20))
    kappa_explore_min = float(cfg.get("mode_conc_kappa_min", 1.0))
    kappa_explore_max = float(cfg.get("mode_conc_kappa_max", 20.0))

    # Claim-A κ-continuation settings (kinematic-convergence-gated ladder with
    # adaptive batch; distinct from the Component-2 gain-triggered ramp above)
    continuation_enabled = bool(cfg.get("kappa_continuation", False))
    cont_ladder = [float(x) for x in cfg.get("continuation_ladder",
                                             [20.0, 35.0, 60.0, 100.0, 200.0, 400.0])]
    cont_batches = [int(x) for x in cfg.get("continuation_batch",
                                            [16384, 16384, 16384, 65536, 65536, 65536])]
    if continuation_enabled and len(cont_batches) != len(cont_ladder):
        raise ValueError(f"--continuation-batch needs {len(cont_ladder)} entries "
                         f"(one per ladder stage), got {len(cont_batches)}")
    cont_conv_tau = float(cfg.get("cont_conv_tau", 0.3))
    cont_conv_window = int(cfg.get("cont_conv_window", 30))
    cont_min_hold = int(cfg.get("cont_min_hold", 60))
    cont_max_hold = int(cfg.get("cont_max_hold", 250))
    cont_explore_min = int(cfg.get("cont_explore_min", 200))
    cont_lr = float(cfg.get("cont_lr", 3e-4))
    cont_entropy_coef = float(cfg.get("cont_entropy_coef", 0.02))
    cont_done_exploit_every = int(cfg.get("cont_done_exploit_every", 3))
    if mode_conc_ramp_enabled and continuation_enabled:
        raise ValueError("--mode-conc-ramp and --kappa-continuation are mutually exclusive")

    # PPO agent configuration (simplified for self-play)
    ppo_cfg = PPOConfig(
        steps_per_update=int(cfg.get("steps_per_update", 4096)),
        epochs=int(cfg.get("update_epochs", 6)),
        minibatch_size=int(cfg.get("minibatch_size", 1024)),
        state_dim=4,  # [q_norm, k_norm, wgap_norm, lgap_norm]
        hidden=int(cfg.get("hidden_size", 128)),
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
        mode_conc_ramp=mode_conc_ramp_enabled,
        kappa_continuation=continuation_enabled,
        mode_conc_kappa_min=kappa_explore_min,
        mode_conc_kappa_scale=float(cfg.get("mode_conc_kappa_scale", 1.0)),
        mode_conc_kappa_max=kappa_explore_max,
        # Pass through stability parameters from config
        gamma=float(cfg.get("gamma", 0.99)),
        gae_lambda=float(cfg.get("gae_lambda", 0.95)),
        value_coef=float(cfg.get("value_coef", 0.5)),
        max_grad_norm=float(cfg.get("max_grad_norm", 0.5)),
        # Opponent lag parameters
        opponent_mode=cfg.get("opponent_mode", "periodic"),
        opponent_sync_interval=int(cfg.get("opponent_sync_interval", 0)),
        opponent_ema_tau=float(cfg.get("opponent_ema_tau", 0.05)),
        opponent_snapshot_keep=int(cfg.get("opponent_snapshot_keep", 10)),
        opponent_history_sample_p=float(cfg.get("opponent_history_sample_p", 0.0)),
        normalize_advantages=bool(cfg.get("normalize_advantages", True)),
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

    # Convergence tracking (exploitability-based, theory-free)
    convergence_cfg = cfg.get("convergence", {}) or {}
    convergence_enabled = bool(convergence_cfg.get("enabled", False))
    # Resolve cheap_gate config: use named profile if specified, else inline "cheap_gate" block
    cheap_profiles = convergence_cfg.get("cheap_gate_profiles", {}) if convergence_enabled else {}
    cheap_profile_name = convergence_cfg.get("cheap_gate_profile", "default") if convergence_enabled else "default"
    if cheap_profiles and cheap_profile_name in cheap_profiles:
        cheap_cfg = cheap_profiles[cheap_profile_name]
    else:
        cheap_cfg = convergence_cfg.get("cheap_gate", {}) if convergence_enabled else {}
    exploit_cfg = convergence_cfg.get("exploit", {}) if convergence_enabled else {}
    cheap_tracker = CheapGateTracker(int(cheap_cfg.get("window_size", 20))) if convergence_enabled else None
    
    # Convergence state tracking
    drift_ok_streak = 0
    exploit_ok_streak = 0
    last_exploitability = None
    last_best_dev_effort = None
    converged_flag = 0
    stop_reason = "max_updates"  # default if we exhaust episodes
    early_stop_triggered = False
    min_updates = int(cfg.get("min_updates", 0))
    last_exploit_eval_step = -999999
    exploit_every_updates = int(cfg.get("exploit_every_updates", 10))

    # Component-2 mode-conc ramp state machine. Phases:
    #   "explore"  — κ pinned [kappa_explore_min, kappa_explore_max], entropy-driven;
    #                policy mean free to climb. Normal exploit-stop SUPPRESSED here.
    #   "ramping"  — after EXP_raw < ramp_trigger_exp for ramp_trigger_patience
    #                consecutive in-loop evals, walk κ through kappa_schedule,
    #                kappa_stage_hold updates per stage (κ pinned min=max=stage).
    #   "done"     — final stage (κ=κ_max) reached; hand off to the normal
    #                exploitability stop (eps_eq=0.03, patience 5).
    # Only active when mode_conc_ramp_enabled; otherwise a no-op ("disabled").
    ramp_phase = "explore" if mode_conc_ramp_enabled else "disabled"
    ramp_trigger_streak = 0
    ramp_stage_idx = -1
    ramp_stage_entered_update = None
    if mode_conc_ramp_enabled:
        print(
            f"[Component-2] mode-conc ramp enabled: explore κ∈[{kappa_explore_min:g},"
            f"{kappa_explore_max:g}], trigger EXP_raw<{ramp_trigger_exp:g}×{ramp_trigger_patience}, "
            f"schedule={kappa_schedule}, stage_hold={kappa_stage_hold}",
            flush=True,
        )

    # Claim-A κ-continuation state machine. Phases:
    #   "explore" — κ free in [kappa_explore_min, kappa_explore_max], base batch,
    #               normal schedules. Advance via the kinematic gate after
    #               cont_explore_min updates.
    #   "ladder"  — κ pinned min=max=cont_ladder[stage]; batch/minibatch scaled to
    #               cont_batches[stage]; lr/entropy pinned (cont_lr/cont_entropy).
    #               Advance stage when |mean(mode last W) − mean(mode prior W)| <
    #               cont_conv_tau (W=cont_conv_window) after cont_min_hold updates;
    #               forced advance at cont_max_hold (logged, counts against pilot).
    #   "done"    — top κ reached + converged; normal exploit-stop takes over
    #               (in-loop eval every cont_done_exploit_every updates).
    # Stop is SUPPRESSED until "done". No payoff-gain trigger, no clock, no
    # analytic e* anywhere in the gate (mode history is a policy statistic).
    cont_phase = "explore" if continuation_enabled else "disabled"
    cont_stage_idx = -1
    cont_stage_entered_update = 0
    cont_forced_advances = 0
    base_steps_per_update = int(ppo_cfg.steps_per_update)
    base_minibatch_size = int(ppo_cfg.minibatch_size)
    if continuation_enabled:
        print(
            f"[Claim-A cont] κ-continuation enabled: explore κ∈[{kappa_explore_min:g},"
            f"{kappa_explore_max:g}] (min {cont_explore_min} upd), ladder={cont_ladder}, "
            f"batches={cont_batches}, gate |Δmode|<{cont_conv_tau:g} over "
            f"W={cont_conv_window} (hold {cont_min_hold}-{cont_max_hold}), "
            f"ladder lr={cont_lr:g} entropy={cont_entropy_coef:g}",
            flush=True,
        )
    # Ablation toggles (Fig. 7): each verification component is independently
    # disable-able. disable_cheap_gate => stability screen always passes;
    # disable_exploitability => exploitability is never evaluated (run goes to
    # budget, stop_reason stays "max_updates").
    disable_cheap_gate = bool(cfg.get("disable_cheap_gate", False))
    disable_exploitability = bool(cfg.get("disable_exploitability", False))
    if disable_cheap_gate:
        print("[ablation] disable_cheap_gate: stability screen always passes", flush=True)
    if disable_exploitability:
        print("[ablation] disable_exploitability: exploitability never evaluated", flush=True)
    exploit_eval_steps: List[int] = []
    
    if convergence_enabled:
        print(f"[Convergence] enabled=True cheap_gate_profile={cheap_profile_name}", flush=True)
    
    # Create environment (reuse across steps for RNG continuity).
    # Rewards are SAMPLED rank-order outcomes (winner w_H, losers w_L) — the
    # env exposes no closed-form reward mode anymore (spec: train/eval split).
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
        "alpha_mean": [],
        "beta_mean": [],
        # Component-2 diagnostics (mode is diagnostic-only; headline effort = mean)
        "mode_effort": [],
        "kappa": [],
        "ramp_phase": [],
        # Claim-A κ-continuation diagnostics
        "cont_phase": [],
        "batch_size": [],
        "mean_kl_window": [],
        "drift_effort": [],
        # Exploitability: sparse (NaN when not evaluated this update)
        "exploitability": [],
        "exploitability_is_valid": [],
    }
    
    last_update_metrics: Optional[Dict[str, float]] = None
    
    # Asymmetric warmup: bias players to different efforts to break symmetry
    asymmetric_warmup_updates = int(cfg.get("asymmetric_warmup_updates", 50))
    asymmetric_bias_magnitude = float(cfg.get("asymmetric_bias_magnitude", 0.3))  # 30% of e*
    if asymmetric_warmup_updates > 0:
        print(f"[PPO-3p] Asymmetric warmup for {asymmetric_warmup_updates} updates:")
        print(f"[PPO-3p]   Player 1: +{asymmetric_bias_magnitude*100:.0f}% bias (explore higher)")
        print(f"[PPO-3p]   Player 2: no bias (follow policy)")
        print(f"[PPO-3p]   Player 3: -{asymmetric_bias_magnitude*100:.0f}% bias (explore lower)")
        print(flush=True)
    
    while steps_done < total_steps_target:
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

        # Component-2: advance the κ ramp stage and pin the net's κ range. The
        # trigger (explore -> ramping) happens after the exploitability eval
        # below; here we only handle stage timing once "ramping"/"done".
        if ramp_phase == "ramping":
            if ramp_stage_entered_update is None:
                ramp_stage_entered_update = update_idx
            if update_idx - ramp_stage_entered_update >= kappa_stage_hold:
                if ramp_stage_idx + 1 >= len(kappa_schedule):
                    ramp_phase = "done"  # final stage reached; normal stop takes over
                    print(f"[Component-2] κ ramp complete at κ={kappa_schedule[-1]:g} "
                          f"(update {update_idx}); normal exploit-stop now active.", flush=True)
                else:
                    ramp_stage_idx += 1
                    ramp_stage_entered_update = update_idx
                    print(f"[Component-2] κ ramp -> stage {ramp_stage_idx} "
                          f"(κ={kappa_schedule[ramp_stage_idx]:g}) at update {update_idx}.", flush=True)
        if ramp_phase in ("ramping", "done"):
            stage_kappa = float(kappa_schedule[max(0, ramp_stage_idx)])
            agent.net.kappa_min = stage_kappa
            agent.net.kappa_max = stage_kappa
        elif ramp_phase == "explore":
            agent.net.kappa_min = kappa_explore_min
            agent.net.kappa_max = kappa_explore_max

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

        # Claim-A κ-continuation: kinematic gate + stage transitions + pinning.
        # Runs AFTER the schedule assignments above so ladder pins override them.
        if cont_phase in ("explore", "ladder"):
            mode_hist = convergence_history["mode_effort"]
            stage_len = update_idx - cont_stage_entered_update
            w = cont_conv_window
            min_hold = cont_explore_min if cont_phase == "explore" else cont_min_hold
            gate_ready = stage_len >= max(min_hold, 2 * w) and len(mode_hist) >= 2 * w
            converged_here = False
            if gate_ready:
                recent = float(np.mean(mode_hist[-w:]))
                prior = float(np.mean(mode_hist[-2 * w:-w]))
                converged_here = abs(recent - prior) < cont_conv_tau
            forced = cont_phase == "ladder" and stage_len >= cont_max_hold
            if converged_here or forced:
                if forced and not converged_here:
                    cont_forced_advances += 1
                    print(f"[Claim-A cont] FORCED advance at update {update_idx} "
                          f"(stage κ={cont_ladder[cont_stage_idx]:g} not converged "
                          f"within {cont_max_hold} updates).", flush=True)
                if cont_stage_idx + 1 >= len(cont_ladder):
                    cont_phase = "done"
                    print(f"[Claim-A cont] ladder complete at κ="
                          f"{cont_ladder[-1]:g} (update {update_idx}); normal "
                          f"exploit-stop now active (eval every "
                          f"{cont_done_exploit_every} updates).", flush=True)
                else:
                    cont_phase = "ladder"
                    cont_stage_idx += 1
                    cont_stage_entered_update = update_idx
                    print(f"[Claim-A cont] advance -> stage {cont_stage_idx} "
                          f"(κ={cont_ladder[cont_stage_idx]:g}, batch="
                          f"{cont_batches[cont_stage_idx]}) at update {update_idx} "
                          f"({'forced' if forced and not converged_here else 'gate'}).",
                          flush=True)
        if cont_phase in ("ladder", "done"):
            stage_kappa = float(cont_ladder[max(0, cont_stage_idx)])
            stage_batch = int(cont_batches[max(0, cont_stage_idx)])
            agent.net.kappa_min = stage_kappa
            agent.net.kappa_max = stage_kappa
            agent.cfg.steps_per_update = stage_batch
            agent.cfg.minibatch_size = max(
                base_minibatch_size,
                base_minibatch_size * (stage_batch // max(1, base_steps_per_update)))
            agent.cfg.entropy_coef = cont_entropy_coef
            for g in agent.opt.param_groups:
                g["lr"] = cont_lr
        elif cont_phase == "explore":
            agent.net.kappa_min = kappa_explore_min
            agent.net.kappa_max = kappa_explore_max

        # Collect rollout
        steps_this = min(ppo_cfg.steps_per_update, total_steps_target - steps_done)

        # Compute asymmetric bias for this update (decays over warmup period)
        apply_asymmetric_bias = update_idx < asymmetric_warmup_updates
        if apply_asymmetric_bias:
            # Bias strength decays linearly from 1.0 to 0.0 over warmup
            bias_strength = 1.0 - (float(update_idx) / float(max(1, asymmetric_warmup_updates)))
            # Use fixed fraction of effort range (no e* dependency)
            bias_amount = (effort_bounds[1] - effort_bounds[0]) * asymmetric_bias_magnitude * bias_strength
        else:
            bias_amount = 0.0
        
        for _ in range(steps_this):
            q = float(rng.choice(train_qs))
            env.q = q
            
            # Generate state for all players (same state in symmetric case)
            state = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
            
            # Player 1: ALWAYS uses learner policy
            a1_norm, e1, logp1, v1 = agent.act(state)

            # Player 2: uses learner policy (self-play)
            a2_norm, e2, logp2, v2 = agent.act(state)

            # Player 3: uses learner policy (self-play)
            a3_norm, e3, logp3, v3 = agent.act(state)
            
            # Apply asymmetric bias during warmup to break symmetry
            if apply_asymmetric_bias and bias_amount > 0:
                lo, hi = effort_bounds
                # Player 1: bias upward (explore higher efforts)
                e1_biased = torch.clamp(e1 + bias_amount, lo, hi)
                # Player 2: no bias (follow policy mean)
                e2_biased = e2
                # Player 3: bias downward (explore lower efforts)
                e3_biased = torch.clamp(e3 - bias_amount, lo, hi)
            else:
                e1_biased, e2_biased, e3_biased = e1, e2, e3
            
            # Execute environment step with (potentially biased) efforts
            efforts = (
                torch.tensor([float(e1_biased.item())]),
                torch.tensor([float(e2_biased.item())]),
                torch.tensor([float(e3_biased.item())]),
            )
            _, rewards, _, done, _ = env.step(efforts)

            r1 = float(rewards[0].item())
            r2 = float(rewards[1].item())
            r3 = float(rewards[2].item())

            # Store transitions for all 3 players (all learner-generated in self-play)
            agent.store(state, a1_norm, logp1, r1, v1, bool(done))
            agent.store(state, a2_norm, logp2, r2, v2, bool(done))
            agent.store(state, a3_norm, logp3, r3, v3, bool(done))
            
            steps_done += 1
        
        # PPO update
        metrics = agent.update()
        last_update_metrics = metrics

        # Reset Adam state if requested (clears crash-polluted second moment)
        reset_at = cfg.get("reset_adam_at")
        if reset_at is not None and update_idx == int(reset_at):
            agent.opt.state.clear()
            print(f"[optimizer] Adam state reset at update {update_idx}", flush=True)

        # Compute policy mean effort and concentration
        test_state = agent.state_from_params(q=float(train_qs[0]), k=k, w_h=w_h, w_l=w_l)
        policy_mean_effort = agent.mean_effort(test_state)
        theoretical_e = e_star_three_players(float(train_qs[0]), w_h, w_l, k)
        with torch.no_grad():
            _dist, _ = agent.dist(test_state)
            _alpha = _dist.concentration1
            _beta = _dist.concentration0
            _alpha_mean = float(_alpha.mean().item())
            _beta_mean = float(_beta.mean().item())
            # Diagnostic only (Component-2): Beta MODE mapped to effort bounds.
            # Headline effort stays the MEAN (policy_mean_effort); at high κ the
            # two coincide. mode = (α-1)/(α+β-2) for α,β>1, else boundary.
            _denom = _alpha + _beta - 2.0
            _mode01 = torch.where(_denom > 1e-8, (_alpha - 1.0) / _denom,
                                  torch.full_like(_denom, 0.5))
            _mode01 = torch.clamp(_mode01, 0.0, 1.0)
            _lo, _hi = effort_bounds
            _mode_effort = float((_lo + _mode01 * (_hi - _lo)).mean().item())

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
        convergence_history["alpha_mean"].append(_alpha_mean)
        convergence_history["beta_mean"].append(_beta_mean)
        convergence_history["mode_effort"].append(_mode_effort)
        convergence_history["kappa"].append(
            float(getattr(agent.net, "kappa_max", float("nan")))
            if (mode_conc_ramp_enabled or continuation_enabled) else float("nan"))
        convergence_history["ramp_phase"].append(ramp_phase)
        convergence_history["cont_phase"].append(cont_phase)
        convergence_history["batch_size"].append(int(agent.cfg.steps_per_update))
        convergence_history["mean_kl_window"].append(float("nan"))
        convergence_history["drift_effort"].append(float("nan"))
        convergence_history["exploitability"].append(float("nan"))
        convergence_history["exploitability_is_valid"].append(False)
        
        # Logging (theory shown for reference only, NOT used for stopping)
        if update_idx % 20 == 0 or update_idx == total_updates - 1:
            abs_err = abs(policy_mean_effort - theoretical_e)
            exploit_str = f" exploit={last_exploitability:.4f}" if last_exploitability is not None else ""
            print(
                f"[PPO-3p] update={update_idx:04d} steps={steps_done:08d} "
                f"policy_mean={policy_mean_effort:.4f} theory={theoretical_e:.4f} "
                f"abs_err={abs_err:.4f} entropy={agent.cfg.entropy_coef:.4f} "
                f"kl={metrics.get('approx_kl', 0.0):.6f}{exploit_str}",
                flush=True,
            )
        
        # === Convergence evaluation (exploitability-based, theory-free) ===
        if convergence_enabled and cheap_tracker is not None:
            cheap_stats = cheap_tracker.compute()
            mean_kl_window = cheap_stats.get("mean_kl_window")
            std_kl_window = cheap_stats.get("std_kl_window")
            drift_effort = cheap_stats.get("drift_effort")
            
            # Update convergence history with cheap gate metrics
            if mean_kl_window is not None:
                convergence_history["mean_kl_window"][-1] = float(mean_kl_window)
            if drift_effort is not None:
                convergence_history["drift_effort"][-1] = float(drift_effort)
            
            # Cheap gate thresholds
            mean_thresh = float(cheap_cfg.get("mean_kl_thresh", 0.0045))
            std_thresh = float(cheap_cfg.get("std_kl_thresh", 0.0035))
            drift_thresh = float(cheap_cfg.get("drift_effort_thresh", 2.0))
            patience_drift = int(cheap_cfg.get("patience_drift", 2))
            exploit_eps = float(exploit_cfg.get("exploit_eps", 0.03))
            
            mean_ok = mean_kl_window is not None and mean_kl_window <= mean_thresh
            std_ok = std_kl_window is not None and std_kl_window <= std_thresh
            drift_ok = drift_effort is not None and drift_effort <= drift_thresh
            # Ablation: --disable-cheap-gate forces the stability screen to pass
            drift_pass = True if disable_cheap_gate else (mean_ok and std_ok and drift_ok)

            if drift_pass:
                drift_ok_streak += 1
            else:
                drift_ok_streak = 0

            exploitability_val = None
            best_dev_effort = None

            # Determine whether to run exploitability evaluation
            steps_since_last_exploit = update_idx - last_exploit_eval_step
            # Claim-A continuation "done": each update is ~16x cost, so evaluate
            # more often to confirm the stop streak without burning budget.
            effective_exploit_every = (cont_done_exploit_every
                                       if cont_phase == "done" else exploit_every_updates)
            periodic_due = steps_since_last_exploit >= effective_exploit_every
            gate_triggered = drift_pass and drift_ok_streak >= patience_drift
            # Ablation: --disable-exploitability never evaluates (run goes to budget)
            if disable_exploitability:
                run_exploit = False
            else:
                run_exploit = periodic_due or (gate_triggered and steps_since_last_exploit >= 1)
            
            if run_exploit:
                last_exploit_eval_step = update_idx
                exploit_eval_steps.append(steps_done)
                # Run exploitability (coarse-to-fine grid) using CRN
                eval_seed = int(cfg.get("seed", 42)) + int(update_idx + 1)
                grid_cfg = exploit_cfg.get("grid", {}) if exploit_cfg else {}
                q_for_exploit = float(train_qs[0]) if train_qs else float(cfg.get("q", q_init))
                exploit_res = eval_exploitability_3p(
                    agent,
                    q=q_for_exploit,
                    effort_bounds=effort_bounds,
                    M=int(exploit_cfg.get("M", 8192)),
                    grid_cfg={
                        "stage_a_step": grid_cfg.get("stage_a_step", 5.0),
                        "stage_b_radius": grid_cfg.get("stage_b_radius", 15.0),
                        "stage_b_step": grid_cfg.get("stage_b_step", 1.0),
                        "stage_c_radius": grid_cfg.get("stage_c_radius", 3.0),
                        "stage_c_step": grid_cfg.get("stage_c_step", 0.25),
                    },
                    seed=eval_seed,
                    w_h=w_h,
                    w_l=w_l,
                    k=k,
                )
                exploitability_val = exploit_res["exploitability"]
                best_dev_effort = exploit_res["best_dev_effort"]
                last_exploitability = exploitability_val
                last_best_dev_effort = best_dev_effort
                
                # Update convergence_history with exploitability
                convergence_history["exploitability"][-1] = float(exploitability_val)
                convergence_history["exploitability_is_valid"][-1] = True
                
                trigger_reason = "periodic" if periodic_due else "cheap_gate"
                print(
                    f"[ConvergenceDebug] trigger={trigger_reason} eval_seed={eval_seed} "
                    f"candidates={exploit_res.get('num_candidates', 'NA')}",
                    flush=True,
                )

                # Component-2: fire the κ ramp once raw exploitability is low for
                # ramp_trigger_patience consecutive in-loop evals (EXP_raw is this
                # exploitability_val — the same sampled deviation gain, at explore κ).
                if ramp_phase == "explore":
                    if exploitability_val < ramp_trigger_exp:
                        ramp_trigger_streak += 1
                    else:
                        ramp_trigger_streak = 0
                    if ramp_trigger_streak >= ramp_trigger_patience:
                        ramp_phase = "ramping"
                        ramp_stage_idx = 0
                        ramp_stage_entered_update = update_idx
                        print(
                            f"[Component-2] ramp TRIGGERED at update {update_idx} "
                            f"(EXP_raw<{ramp_trigger_exp:g} ×{ramp_trigger_streak}); "
                            f"entering stage 0 (κ={kappa_schedule[0]:g}).",
                            flush=True,
                        )

                if exploitability_val < exploit_eps:
                    exploit_ok_streak += 1
                else:
                    exploit_ok_streak = 0
                    drift_ok_streak = 0  # reset cheap gate streak on failure
                # Component-2: suppress the normal exploit-stop until the κ ramp
                # has reached its final stage — stopping mid-explore would freeze
                # the wide, undershooting low-κ policy this retrain aims to sharpen.
                # Claim-A continuation: same suppression until the ladder is done.
                ramp_allows_stop = (ramp_phase in ("disabled", "done")
                                    and cont_phase in ("disabled", "done"))
                if ramp_allows_stop and exploit_ok_streak >= int(exploit_cfg.get("patience_exploit", 5)):
                    if update_idx < min_updates:
                        print(
                            f"[Convergence] Exploitability streak satisfied at upd={update_idx}, "
                            f"but min_updates={min_updates} not reached; continuing.",
                            flush=True,
                        )
                    else:
                        converged_flag = 1
                        stop_reason = "exploitability"
                        print(
                            f"[Convergence] Exploitability satisfied for {exploit_ok_streak} evals; stopping training.",
                            flush=True,
                        )
                        early_stop_triggered = True
            else:
                # Non-eval update: do NOT reset exploit_ok_streak or
                # last_best_dev_effort.  Only an actual eval failure (line 1109)
                # should reset the streak.  Resetting here made it impossible
                # to accumulate a streak of 5 with periodic-only (gap=10) evals.
                # See docs/3p_stop_drilldown.md for full analysis.
                last_exploitability = None
            
            # Convergence status logging (every 20 updates or when exploit evaluated)
            if update_idx % 20 == 0 or run_exploit:
                def _fmt(val, digits=4):
                    return "NA" if val is None else f"{val:.{digits}f}"
                decision = "NOT_EVAL"
                if run_exploit:
                    decision = "EXPLOIT_OK++" if exploitability_val is not None and exploitability_val < exploit_eps else "RUN_EXPLOIT"
                print(
                    f"[Convergence] upd={update_idx + 1} mean_kl={_fmt(mean_kl_window)} "
                    f"std_kl={_fmt(std_kl_window)} drift={_fmt(drift_effort, 2)} "
                    f"exploit={_fmt(exploitability_val)} best_dev={_fmt(best_dev_effort)} "
                    f"streaks: drift={drift_ok_streak}/{patience_drift} "
                    f"exploit={exploit_ok_streak}/{exploit_cfg.get('patience_exploit', 5)} "
                    f"decision={decision}",
                    flush=True,
                )
            
            if early_stop_triggered:
                break
        
        update_idx += 1
    
    # Forced final exploitability evaluation (ensures BR fields are populated
    # regardless of stop_reason — fixes null final_br_effort on max_updates stops)
    if convergence_enabled:
        q_for_exploit = float(train_qs[0]) if train_qs else float(cfg.get("q", q_init))
        eval_seed = int(cfg.get("seed", 42)) + int(update_idx + 9999)
        grid_cfg = exploit_cfg.get("grid", {}) if exploit_cfg else {}
        try:
            final_exploit_res = eval_exploitability_3p(
                agent,
                q=q_for_exploit,
                effort_bounds=effort_bounds,
                M=int(exploit_cfg.get("M", 8192)),
                grid_cfg={
                    "stage_a_step": grid_cfg.get("stage_a_step", 5.0),
                    "stage_b_radius": grid_cfg.get("stage_b_radius", 15.0),
                    "stage_b_step": grid_cfg.get("stage_b_step", 1.0),
                    "stage_c_radius": grid_cfg.get("stage_c_radius", 3.0),
                    "stage_c_step": grid_cfg.get("stage_c_step", 0.25),
                },
                seed=eval_seed,
                w_h=w_h,
                w_l=w_l,
                k=k,
            )
            last_exploitability = final_exploit_res["exploitability"]
            last_best_dev_effort = final_exploit_res["best_dev_effort"]
            print(
                f"[Convergence] Forced final exploit eval: "
                f"exploit={last_exploitability:.4f}, BR={last_best_dev_effort:.2f}",
                flush=True,
            )
        except Exception as e:
            print(f"[Convergence] Forced final exploit eval failed: {e}", flush=True)

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
            "converged_flag": converged_flag,
            "last_exploitability": float(last_exploitability) if last_exploitability is not None else None,
            "last_best_dev_effort": float(last_best_dev_effort) if last_best_dev_effort is not None else None,
            "exploit_ok_streak": exploit_ok_streak,
        }
        results.append(result)
        
        # Save convergence history
        convergence_data = {
            "algorithm": "ppo",
            "num_players": 3,
            "q": float(q),
            "theoretical_effort": float(theoretical_e),
            "ablation_name": ablation_name,
            # Structured dicts expected by sweep parser
            "theoretical": {
                "effort": float(theoretical_e),
            },
            "final": {
                "effort": float(policy_mean_effort),
                "gap": float(abs_err),
            },
            "final_results": {
                "final_effort": float(policy_mean_effort),
                "theoretical_e": float(theoretical_e),
                "abs_error": float(abs_err),
            },
            # Stopping info (required by sweep_exploit_ablation.py)
            "stop_reason": stop_reason,
            "stopped_at_update": update_idx,
            "joint_exploit_ok_streak": exploit_ok_streak,
            "final_exploit_1": last_exploitability,
            "final_exploit_2": last_exploitability,  # symmetric 3-player game
            "final_exploit_max": last_exploitability,
            "final_br_effort_1": last_best_dev_effort,
            "final_br_effort_2": last_best_dev_effort,  # symmetric 3-player game
            # Exploit config
            "exploit_config": {
                "exploit_eps": float(exploit_cfg.get("exploit_eps", 0.03)),
                "patience_exploit": int(exploit_cfg.get("patience_exploit", 5)),
                "exploit_every_updates": exploit_every_updates,
                "exploit_M": int(exploit_cfg.get("M", 8192)),
                "disable_cheap_gate": disable_cheap_gate,
                "disable_exploitability": disable_exploitability,
            },
            # Claim-A κ-continuation config + outcome (None when disabled)
            "continuation_config": ({
                "ladder": cont_ladder,
                "batches": cont_batches,
                "conv_tau": cont_conv_tau,
                "conv_window": cont_conv_window,
                "min_hold": cont_min_hold,
                "max_hold": cont_max_hold,
                "explore_min": cont_explore_min,
                "cont_lr": cont_lr,
                "cont_entropy_coef": cont_entropy_coef,
                "done_exploit_every": cont_done_exploit_every,
                "forced_advances": cont_forced_advances,
                "final_phase": cont_phase,
                "final_stage_idx": cont_stage_idx,
            } if continuation_enabled else None),
            **convergence_history,
        }
        
        convergence_dir = os.path.join("results", "three_players", "convergence")
        os.makedirs(convergence_dir, exist_ok=True)
        seed_val = cfg.get("seed", 42)
        tag_part = f"_{output_tag}" if output_tag else ""
        convergence_file = os.path.join(
            convergence_dir,
            f"ppo_3p_q{q:.1f}_seed{seed_val}{tag_part}_{ablation_name}_convergence.json"
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
    
    # Theory-align-v2 settings (copied from two-player for consistency)
    if args.theory_align_v2:
        cfg["entropy_coef_start"] = 0.0
        cfg["entropy_coef_hold"] = 0.0
        cfg["entropy_coef_end"] = 0.0
        cfg["theory_align_v2"] = True
        # Concentration parameters for sharp (deterministic) policy
        cfg["theory_align_v2_conc_min"] = 1000.0
        cfg["theory_align_v2_conc_scale"] = 10000.0
        cfg["theory_align_v2_conc_max"] = 100000.0
        # Regularization coefficients
        cfg["theory_align_v2_var_coef"] = 5e-2
        # Ramping schedule for concentration
        cfg["theory_align_v2_conc_min_start"] = 100.0
        cfg["theory_align_v2_conc_scale_start"] = 100.0
        cfg["theory_align_v2_var_coef_start"] = 0.0
        cfg["theory_align_v2_ramp_warmup"] = 20
        cfg["theory_align_v2_ramp_steps"] = 50
        cfg["theory_align_v2_early_stop_window"] = 20
        # Stability tweaks (v2 only): reduce update aggressiveness
        cfg["lr_start"] = 5e-5
        cfg["lr_end"] = 2e-5
        cfg["update_epochs"] = 1
        cfg["clip_range_start"] = 0.2
        cfg["clip_range_end"] = 0.15
        cfg["target_kl"] = 0.06
        cfg["ratio_stop_threshold"] = 2.2
        cfg["max_grad_norm"] = 0.25
        cfg["value_coef"] = 1.0
        cfg["gae_lambda"] = 1.0
        cfg["gamma"] = 1.0

    # --mean-conc-param: switch to mean+conc network architecture only, keep all other hyperparams
    if hasattr(args, 'mean_conc_param') and args.mean_conc_param and not args.theory_align_v2:
        cfg["theory_align_v2"] = True
        cfg["theory_align_v2_conc_min"] = 1.0
        cfg["theory_align_v2_conc_scale"] = 1.0
        cfg["theory_align_v2_conc_max"] = None
        cfg["theory_align_v2_var_coef"] = 0.0
        print("[config] --mean-conc-param: using ActorCriticMeanConc with zero regularization, "
              "all other hyperparams unchanged", flush=True)

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

    # --override-entropy-start: override start/hold while keeping end unchanged
    if hasattr(args, 'override_entropy_start') and args.override_entropy_start is not None:
        cfg["entropy_coef_start"] = float(args.override_entropy_start)
        cfg["entropy_coef_hold"] = float(args.override_entropy_start)
        print(f"[config] entropy_coef_start/hold overridden to {args.override_entropy_start}, "
              f"entropy_coef_end={cfg.get('entropy_coef_end', 0.005)}", flush=True)

    # --override-conc-ramp-warmup: override concentration ramp warmup
    if hasattr(args, 'override_conc_ramp_warmup') and args.override_conc_ramp_warmup is not None:
        cfg["theory_align_v2_ramp_warmup"] = int(args.override_conc_ramp_warmup)
        print(f"[config] conc_ramp_warmup override: {args.override_conc_ramp_warmup}", flush=True)

    # --min-updates: minimum updates before early stop
    if hasattr(args, 'min_updates') and args.min_updates > 0:
        cfg["min_updates"] = int(args.min_updates)
        print(f"[config] min_updates: {args.min_updates}", flush=True)

    # Component-2 mode-conc ramp (exploitability-triggered κ ramp)
    if getattr(args, 'mode_conc_ramp', False):
        cfg["mode_conc_ramp"] = True
        cfg["kappa_schedule"] = [float(x) for x in str(args.kappa_schedule).split(",")]
        cfg["ramp_trigger_exp"] = float(args.ramp_trigger_exp)
        cfg["ramp_trigger_patience"] = int(args.ramp_trigger_patience)
        cfg["kappa_stage_hold"] = int(args.kappa_stage_hold)
        # Explore-phase κ range = [1, first stage]; stages pin κ_min=κ_max=stage.
        cfg["mode_conc_kappa_min"] = 1.0
        cfg["mode_conc_kappa_max"] = float(cfg["kappa_schedule"][0])
        print(f"[config] --mode-conc-ramp: schedule={cfg['kappa_schedule']} "
              f"trigger EXP_raw<{cfg['ramp_trigger_exp']}×{cfg['ramp_trigger_patience']} "
              f"stage_hold={cfg['kappa_stage_hold']}", flush=True)

    # Claim-A κ-continuation (kinematic-gated ladder + adaptive batch)
    if getattr(args, 'kappa_continuation', False):
        cfg["kappa_continuation"] = True
        cfg["continuation_ladder"] = [float(x) for x in str(args.continuation_ladder).split(",")]
        cfg["continuation_batch"] = [int(x) for x in str(args.continuation_batch).split(",")]
        cfg["cont_conv_tau"] = float(args.cont_conv_tau)
        cfg["cont_conv_window"] = int(args.cont_conv_window)
        cfg["cont_min_hold"] = int(args.cont_min_hold)
        cfg["cont_max_hold"] = int(args.cont_max_hold)
        cfg["cont_explore_min"] = int(args.cont_explore_min)
        cfg["cont_lr"] = float(args.cont_lr)
        cfg["cont_entropy_coef"] = float(args.cont_entropy_coef)
        cfg["cont_done_exploit_every"] = int(args.cont_done_exploit_every)
        # Explore-phase κ range = [1, first ladder stage]
        cfg["mode_conc_kappa_min"] = 1.0
        cfg["mode_conc_kappa_max"] = float(cfg["continuation_ladder"][0])
        print(f"[config] --kappa-continuation: ladder={cfg['continuation_ladder']} "
              f"batches={cfg['continuation_batch']} gate |Δmode|<{cfg['cont_conv_tau']} "
              f"W={cfg['cont_conv_window']} hold {cfg['cont_min_hold']}-{cfg['cont_max_hold']} "
              f"explore_min={cfg['cont_explore_min']}", flush=True)

    # --disable-adv-norm: skip advantage normalization
    if hasattr(args, 'disable_adv_norm') and args.disable_adv_norm:
        cfg["normalize_advantages"] = False
        print("[ablation] --disable-adv-norm: advantage normalization disabled", flush=True)


    # --hidden-size: override network hidden layer size
    if hasattr(args, 'hidden_size') and args.hidden_size is not None:
        cfg["hidden_size"] = int(args.hidden_size)

    # --reset-adam-at: reset Adam optimizer state at specified update
    if hasattr(args, 'reset_adam_at') and args.reset_adam_at is not None:
        cfg["reset_adam_at"] = int(args.reset_adam_at)
        print(f"[config] Adam optimizer will be reset at update {args.reset_adam_at}", flush=True)

    # --max-updates: override training budget
    if hasattr(args, 'max_updates') and args.max_updates is not None:
        cfg["max_updates"] = int(args.max_updates)
        cfg["episodes"] = int(args.max_updates) * int(cfg.get("steps_per_update", 4096))
        print(f"[config] max_updates overridden to {args.max_updates}", flush=True)

    # Convergence eval settings
    if "convergence" not in cfg:
        cfg["convergence"] = {}
    if args.enable_convergence_eval:
        cfg["convergence"]["enabled"] = True
    if args.cheap_gate_profile is not None:
        cfg["convergence"]["cheap_gate_profile"] = args.cheap_gate_profile

    # Exploitability overrides
    if hasattr(args, 'exploit_every_updates') and args.exploit_every_updates is not None:
        cfg["exploit_every_updates"] = int(args.exploit_every_updates)
    # Ablation toggles (Fig. 7): independently disable each verification component
    if getattr(args, 'disable_cheap_gate', False):
        cfg["disable_cheap_gate"] = True
    if getattr(args, 'disable_exploitability', False):
        cfg["disable_exploitability"] = True
    if args.exploit_eps is not None:
        if "convergence" not in cfg:
            cfg["convergence"] = {}
        if "exploit" not in cfg["convergence"]:
            cfg["convergence"]["exploit"] = {}
        cfg["convergence"]["exploit"]["exploit_eps"] = float(args.exploit_eps)
    if args.exploit_patience is not None:
        if "convergence" not in cfg:
            cfg["convergence"] = {}
        if "exploit" not in cfg["convergence"]:
            cfg["convergence"]["exploit"] = {}
        cfg["convergence"]["exploit"]["patience_exploit"] = int(args.exploit_patience)
    if args.exploit_M is not None:
        if "convergence" not in cfg:
            cfg["convergence"] = {}
        if "exploit" not in cfg["convergence"]:
            cfg["convergence"]["exploit"] = {}
        cfg["convergence"]["exploit"]["M"] = int(args.exploit_M)
    
    # PPO tuning overrides
    if args.steps_per_update is not None:
        cfg["steps_per_update"] = int(args.steps_per_update)
        print(f"[config] CLI override: steps_per_update={cfg['steps_per_update']}", flush=True)
    if args.minibatch_size is not None:
        cfg["minibatch_size"] = int(args.minibatch_size)
        print(f"[config] CLI override: minibatch_size={cfg['minibatch_size']}", flush=True)
    if args.update_epochs is not None:
        cfg["update_epochs"] = int(args.update_epochs)
        print(f"[config] CLI override: update_epochs={cfg['update_epochs']}", flush=True)

    # Asymmetric warmup settings
    if hasattr(args, 'no_asymmetric_warmup') and args.no_asymmetric_warmup:
        cfg["asymmetric_warmup_updates"] = 0
    elif hasattr(args, 'asymmetric_warmup_updates'):
        cfg["asymmetric_warmup_updates"] = int(args.asymmetric_warmup_updates)
    if hasattr(args, 'asymmetric_bias_magnitude'):
        cfg["asymmetric_bias_magnitude"] = float(args.asymmetric_bias_magnitude)
    
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
                log=True,
                ablation_name=args.ablation_name,
                force=args.force,
            )
    else:
        train_qs = [args.q] if args.q is not None else list(cfg["q_list"])
        run_ppo(
            cfg,
            episodes=args.episodes,
            train_qs=train_qs,
            ablation_name=args.ablation_name,
            output_tag=args.output_tag,
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
    parser.add_argument(
        "--mean-conc-param",
        action="store_true",
        help="Use mean+conc network parameterization (decoupled from concentration) "
             "without changing any other hyperparameters. No regularization, no entropy override.",
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
        help="Override theory_align_v2_ramp_warmup (updates before concentration ramp begins). "
             "Use 200 for the concentration fix validated in Round 2 two-player experiments.",
    )

    parser.add_argument(
        "--min-updates",
        type=int,
        default=0,
        help="Minimum updates before exploitability-based early stop is allowed. "
             "0 = no minimum (default).",
    )

    # Component-2: mode-conc head + exploitability-triggered concentration ramp
    parser.add_argument(
        "--mode-conc-ramp",
        action="store_true",
        help="Enable Component-2: mode-concentration policy head (α,β≥1) + an "
             "exploitability-triggered κ ramp. Distinct code path from "
             "--theory-align-v2. Tests whether PPO's own output reaches e* (Claim A).",
    )
    parser.add_argument(
        "--kappa-schedule",
        type=str,
        default="20,50,100,200",
        help="Comma-separated κ stages walked after the ramp triggers "
             "(default: 20,50,100,200).",
    )
    parser.add_argument(
        "--ramp-trigger-exp",
        type=float,
        default=0.05,
        help="Raw exploitability threshold that (for --ramp-trigger-patience "
             "consecutive in-loop evals) fires the κ ramp (default: 0.05).",
    )
    parser.add_argument(
        "--ramp-trigger-patience",
        type=int,
        default=3,
        help="Consecutive in-loop evals with EXP_raw below --ramp-trigger-exp "
             "required to fire the ramp (default: 3, guards against transient dips).",
    )
    parser.add_argument(
        "--kappa-stage-hold",
        type=int,
        default=20,
        help="Updates held at each κ stage before advancing (default: 20).",
    )

    # Claim-A κ-continuation: kinematic-convergence-gated κ ladder + adaptive batch
    parser.add_argument(
        "--kappa-continuation",
        action="store_true",
        help="Enable Claim-A κ-continuation: ModeConc head + a κ ladder advanced by a "
             "kinematic convergence gate (|Δmode| over a window), with per-stage "
             "enlarged batch to shrink the plateau diffusion band. Mutually exclusive "
             "with --mode-conc-ramp.",
    )
    parser.add_argument(
        "--continuation-ladder",
        type=str,
        default="20,35,60,100,200,400",
        help="Comma-separated κ ladder stages (default: 20,35,60,100,200,400).",
    )
    parser.add_argument(
        "--continuation-batch",
        type=str,
        default="16384,16384,16384,65536,65536,65536",
        help="Comma-separated steps_per_update per ladder stage (parallel to "
             "--continuation-ladder). Minibatch is scaled proportionally so the "
             "grad-step count per update stays constant.",
    )
    parser.add_argument("--cont-conv-tau", type=float, default=0.3,
                        help="Kinematic gate: |mean(mode last W) − mean(mode prior W)| "
                             "threshold in effort units (default: 0.3).")
    parser.add_argument("--cont-conv-window", type=int, default=30,
                        help="Kinematic gate window W in updates (default: 30).")
    parser.add_argument("--cont-min-hold", type=int, default=60,
                        help="Minimum updates per ladder stage (default: 60; the gate "
                             "also needs 2W samples).")
    parser.add_argument("--cont-max-hold", type=int, default=250,
                        help="Forced stage advance after this many updates (default: "
                             "250; logged, counts against the pilot gate).")
    parser.add_argument("--cont-explore-min", type=int, default=200,
                        help="Minimum explore updates before the ladder may start "
                             "(default: 200).")
    parser.add_argument("--cont-lr", type=float, default=3e-4,
                        help="lr pinned during ladder+done (default: 3e-4, the value "
                             "at which c2's healthy ramp segments were measured).")
    parser.add_argument("--cont-entropy-coef", type=float, default=0.02,
                        help="entropy_coef pinned during ladder+done (default: 0.02).")
    parser.add_argument("--cont-done-exploit-every", type=int, default=3,
                        help="In-loop exploitability eval interval during done "
                             "(default: 3; each done update is ~16x cost).")

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
    
    # Exploitability evaluation controls
    parser.add_argument(
        "--exploit-every-updates",
        type=int,
        default=10,
        help="Max interval between exploitability evaluations (default: 10)",
    )
    parser.add_argument(
        "--exploit-eps",
        type=float,
        default=None,
        help="Override exploit_eps threshold (default: from config, 0.03)",
    )
    parser.add_argument(
        "--exploit-patience",
        type=int,
        default=None,
        help="Override patience_exploit (default: from config, 5)",
    )
    parser.add_argument(
        "--exploit-M",
        type=int,
        default=None,
        help="Monte Carlo samples for exploitability (default: from config, 8192)",
    )
    
    # Ablation flags
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
    parser.add_argument(
        "--override-entropy-start",
        type=float,
        default=None,
        help="Override entropy_coef_start and entropy_coef_hold (keeps entropy_coef_end unchanged).",
    )
    parser.add_argument(
        "--disable-adv-norm",
        action="store_true",
        help="Disable advantage normalization in PPO updates (weak-signal experiment).",
    )
    parser.add_argument(
        "--ablation-name",
        type=str,
        default="baseline",
        help="Ablation variant name for output files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing gradient result file (default: refuse).",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Optional tag inserted into convergence JSON filename (e.g., 'round3').",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Override max_updates (default: from config, 1500).",
    )
    parser.add_argument(
        "--reset-adam-at",
        type=int,
        default=None,
        help="Reset Adam optimizer state (m and v) at this update number.",
    )

    # Asymmetric warmup to break symmetry
    parser.add_argument(
        "--asymmetric-warmup-updates",
        type=int,
        default=50,
        help="Number of updates with asymmetric bias (0 to disable). Default: 50",
    )
    parser.add_argument(
        "--asymmetric-bias-magnitude",
        type=float,
        default=0.3,
        help="Bias magnitude as fraction of theoretical e* (0.3 = 30%%). Default: 0.3",
    )
    parser.add_argument(
        "--no-asymmetric-warmup",
        action="store_true",
        help="Disable asymmetric warmup (equivalent to --asymmetric-warmup-updates 0)",
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Hidden layer size for actor-critic network (default: 128, 2P uses 64).",
    )

    # PPO tuning overrides
    parser.add_argument("--steps-per-update", type=int, default=None,
                        help="Override steps_per_update (default: from config, 4096)")
    parser.add_argument("--minibatch-size", type=int, default=None,
                        help="Override minibatch_size (default: from config, 1024)")
    parser.add_argument("--update-epochs", type=int, default=None,
                        help="Override update_epochs (default: from config, 6)")

    args = parser.parse_args()
    
    log_path = _build_log_path(args)
    with _tee_console_to_file(log_path):
        print(f"[log] Mirroring console output to {log_path}")
        _run_cli(args)
        print(f"[log] Run complete. Full console trace saved to {log_path}")


if __name__ == "__main__":
    main()
