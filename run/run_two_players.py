#!/usr/bin/env python3
"""
One-Stage Two-Player Experiment (spec-compliant)

Writes standardized CSV and figure overlays. For one-stage, we map the single
stage to the CSV's stage-2 fields (stage-1 fields set to 0).
"""

# Smoke tests (theory-align experiment):
# 1) Baseline: python run/run_two_players.py --method ppo --rollout-mode selfplay --q 40 --episodes 2048000 --seed 42
# 2) Theory-align:
#    python run/run_two_players.py --method ppo --rollout-mode selfplay --q 40 --episodes 2048000 --seed 42 \
#      --enable-convergence-eval --cheap-gate-profile aggressive --theory-align
# Expect: [TheoryAlign] lines, ent=0, conc_mean >= 80, mean_vs_sample_gap ~0, policy -> e*=54.69.

from __future__ import annotations  # Python 3.8 compatibility for type hints

import sys
import os
import argparse
import math
import json
import subprocess
from collections import deque
import datetime
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
import numpy as np
import torch


def _get_git_sha() -> str:
    """Get current git commit SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]  # Short SHA
    except Exception:
        pass
    return "unknown"

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
    return os.path.join("results", "two_players", "logs", filename)


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


# === Convergence / Exploitability helpers =========================================================


class CheapGateTracker:
    """
    Rolling-window tracker for cheap stability metrics (KL + policy drift).
    """

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


def _sample_policy_efforts(agent, q: float, effort_bounds: tuple[float, float], M: int, *, seed: int, k: float, w_h: float, w_l: float):
    """
    Sample efforts from current policy (Beta distribution) using torch vectorized sampling.
    """
    device = next(agent.net.parameters()).device
    gen_key = (int(seed), device.type, device.index)
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
            # Fallback for torch versions where Distribution.sample lacks a generator arg.
            # Derive a per-call seed from the provided generator so randomness still advances deterministically.
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
            if device.type == "cuda":
                device_index = device.index if device.index is not None else torch.cuda.current_device()
                with torch.random.fork_rng(devices=[device_index]):
                    torch.manual_seed(seed_value)
                    torch.cuda.manual_seed_all(seed_value)
                    samples = dist.sample((M,))
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed_value)
                    samples = dist.sample((M,))
        samples = samples.squeeze(-1).clamp(0.0, 1.0)
    low, high = effort_bounds
    return (low + samples * (high - low)).to(device)


def _payoff_player1(
    e1: torch.Tensor,
    e2: torch.Tensor,
    eps1: torch.Tensor,
    eps2: torch.Tensor,
    tie_breaks: torch.Tensor,
    *,
    w_h: float,
    w_l: float,
    k: float,
) -> torch.Tensor:
    """
    Vectorized payoff for player1 given efforts e1 (deviator) and e2 (policy), with CRN noise.
    """
    y1 = e1 + eps1
    y2 = e2 + eps2
    winners = torch.where(y1 > y2, 0, torch.where(y2 > y1, 1, tie_breaks))
    payoff1 = torch.where(winners == 0, w_h, w_l) - k * (e1 ** 2)
    return payoff1


def eval_exploitability(
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
    """
    Monte Carlo exploitability evaluation (approx ε-Nash) with CRN and coarse-to-fine grid.

    Returns:
        {
            "exploitability": float,
            "best_dev_effort": float,
            "u_dev": float,
            "u_selfplay": float,
            "num_candidates": int,
        }
    """
    device = next(agent.net.parameters()).device
    low, high = effort_bounds
    with torch.no_grad():
        if device.type == "cuda":
            device_index = device.index if device.index is not None else torch.cuda.current_device()
            with torch.random.fork_rng(devices=[device_index]):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                # Common random numbers (CRN) for all candidates
                eps1 = (torch.rand(M, device=device) * 2.0 - 1.0) * q
                eps2 = (torch.rand(M, device=device) * 2.0 - 1.0) * q
                tie_breaks = torch.randint(0, 2, (M,), device=device)

                # Sample policy efforts for self-play and opponent
                efforts_policy_1 = _sample_policy_efforts(agent, q, effort_bounds, M, seed=seed, k=k, w_h=w_h, w_l=w_l)
                efforts_policy_2 = _sample_policy_efforts(agent, q, effort_bounds, M, seed=seed + 17, k=k, w_h=w_h, w_l=w_l)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                # Common random numbers (CRN) for all candidates
                eps1 = (torch.rand(M, device=device) * 2.0 - 1.0) * q
                eps2 = (torch.rand(M, device=device) * 2.0 - 1.0) * q
                tie_breaks = torch.randint(0, 2, (M,), device=device)

                # Sample policy efforts for self-play and opponent
                efforts_policy_1 = _sample_policy_efforts(agent, q, effort_bounds, M, seed=seed, k=k, w_h=w_h, w_l=w_l)
                efforts_policy_2 = _sample_policy_efforts(agent, q, effort_bounds, M, seed=seed + 17, k=k, w_h=w_h, w_l=w_l)

        # U_self = E[u(π, π)]
        u_self = _payoff_player1(
            efforts_policy_1, efforts_policy_2, eps1, eps2, tie_breaks, w_h=w_h, w_l=w_l, k=k
        ).mean().item()

        # Candidate grids (coarse-to-fine)
        stage_a = torch.arange(low, high + 1e-6, grid_cfg.get("stage_a_step", 5.0), device=device)

        def _around(center: float, radius: float, step: float):
            lo = max(low, center - radius)
            hi = min(high, center + radius)
            return torch.arange(lo, hi + 1e-6, step, device=device)

        best_delta = -float("inf")
        best_e = float(low)
        num_candidates = 0

        def eval_candidates(candidates: torch.Tensor, eps1: torch.Tensor, eps2: torch.Tensor, tie_breaks: torch.Tensor):
            nonlocal best_delta, best_e, num_candidates
            for e_dev in candidates:
                e_dev = e_dev.item()
                efforts_dev = torch.full((M,), e_dev, device=device)
                payoff_dev = _payoff_player1(
                    efforts_dev, efforts_policy_2, eps1, eps2, tie_breaks, w_h=w_h, w_l=w_l, k=k
                )
                delta = payoff_dev.mean().item() - u_self
                num_candidates += 1
                if delta > best_delta:
                    best_delta = delta
                    best_e = e_dev

        eval_candidates(stage_a, eps1, eps2, tie_breaks)
        stage_b = _around(best_e, grid_cfg.get("stage_b_radius", 15.0), grid_cfg.get("stage_b_step", 1.0))
        eval_candidates(stage_b, eps1, eps2, tie_breaks)
        stage_c = _around(best_e, grid_cfg.get("stage_c_radius", 3.0), grid_cfg.get("stage_c_step", 0.25))
        eval_candidates(stage_c, eps1, eps2, tie_breaks)

    return {
        "exploitability": float(best_delta),
        "best_dev_effort": float(best_e),
        "u_dev": float(u_self + best_delta),
        "u_selfplay": float(u_self),
        "num_candidates": int(num_candidates),
    }


def gradient_descent_two_players(
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
        # Convergence history for plotting
        "e1_history": [float(e1)],  # Start with initial value
        "e2_history": [float(e2)],  # Start with initial value
        "step_history": [0],         # Step 0 is the initial state
    }

    for step in range(1, steps + 1):
        # Adaptive learning rate with exponential decay
        lr_current = lr * (lr_decay ** step)
        
        g1, g2 = _stochastic_fd_gradients(env, e1, e2, delta=eps, num_samples=num_samples)
        e1_new = _clip_effort(e1 + lr_current * g1, effort_bounds)
        e2_new = _clip_effort(e2 + lr_current * g2, effort_bounds)

        delta_e1 = abs(e1_new - e1)
        delta_e2 = abs(e2_new - e2)
        grad_norm = max(abs(g1), abs(g2))

        e1, e2 = e1_new, e2_new
        
        # Periodic symmetry enforcement (prevent drift to asymmetric points)
        if symmetry_enforce_every > 0 and step % symmetry_enforce_every == 0:
            e_avg = 0.5 * (e1 + e2)
            e1 = e2 = e_avg
            if log and step <= 100:  # Debug info in early steps
                print(f"  [symmetry enforce] step={step} e_avg={e_avg:.6f}")
        
        history["iterations"] = float(step)
        history["final_grad"] = float(grad_norm)
        history["final_grad_pair"] = (float(g1), float(g2))
        
        # Record convergence history at every step
        history["e1_history"].append(float(e1))
        history["e2_history"].append(float(e2))
        history["step_history"].append(step)
        
        # Enhanced logging with symmetry gap
        symmetry_gap = abs(e1 - e2)
        if log and (step == 1 or step % 250 == 0 or step == steps):
            print(
                f"[gradient-2p] step={step:05d} e1={e1:.6f} e2={e2:.6f} "
                f"grad=({g1:.6f},{g2:.6f}) delta=({delta_e1:.3e},{delta_e2:.3e}) "
                f"lr={lr_current:.6f} sym_gap={symmetry_gap:.4f}"
            )
        
        # Stricter convergence criteria: gradient small AND symmetric AND stable
        if (grad_norm < tol and 
            symmetry_gap < symmetry_tol and
            max(delta_e1, delta_e2) < tol):
            if log:
                print(
                    f"[gradient-2p] converged at step={step} "
                    f"grad_norm={grad_norm:.3e} symmetry_gap={symmetry_gap:.3e}"
                )
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
    tol: float = 1e-5,
    num_samples: int = 256,
    init_perturb: float = 1.0,
    lr_decay: float = 0.9995,
    symmetry_enforce_every: int = 50,
    symmetry_tol: float = 0.1,
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
        lr_decay=lr_decay,
        symmetry_enforce_every=symmetry_enforce_every,
        symmetry_tol=symmetry_tol,
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
    
    # Save convergence history for plotting
    if log:
        convergence_data = {
            "algorithm": "gradient",
            "q": float(q),
            "theoretical_effort": float(theoretical_e),
            "steps": meta["step_history"],
            "agent1_effort": meta["e1_history"],
            "agent2_effort": meta["e2_history"],
            "parameters": {
                "lr": float(lr),
                "grad_eps": float(grad_eps),
                "tol": float(tol),
                "num_samples": int(num_samples),
                "init_perturb": float(init_perturb),
            }
        }
        
        # Create convergence_history directory if it doesn't exist
        convergence_dir = os.path.join("results", "two_players", "convergence")
        os.makedirs(convergence_dir, exist_ok=True)
        
        # Save to JSON file
        convergence_file = os.path.join(
            convergence_dir, 
            f"gradient_q{q:.1f}_convergence.json"
        )
        with open(convergence_file, 'w') as f:
            json.dump(convergence_data, f, indent=2)
        print(f"[gradient-2p] Saved convergence history to {convergence_file}")
    
    return row


def get_effort_for_side(
    side: str,
    agent: PPOTwoPlayersBandit,
    obs: torch.Tensor,
    q: float,
    effort_bounds: tuple[float, float],
    fixed_effort: Optional[float],
    is_train_side: bool,
    seed: int,
    fallback_rollout_stats: Optional[Dict[str, float]],
) -> Dict[str, object]:
    """Return effort/action payload for a side; freeze non-train side when fixed effort is set."""
    del side, q, seed, fallback_rollout_stats
    if fixed_effort is not None and not is_train_side:
        fixed_value = _clip_effort(float(fixed_effort), effort_bounds)
        return {
            "effort": torch.tensor([fixed_value], dtype=torch.float32),
            "action_norm": None,
            "logp": None,
            "value": None,
            "source": "fixed",
        }
    action_norm, effort, logp, value = agent.act(obs)
    return {
        "effort": effort,
        "action_norm": action_norm,
        "logp": logp,
        "value": value,
        "source": "policy_sample",
    }


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
    fixed_opponent_effort: Optional[float] = None,
    train_side: str = "p1",
    # Paper artifact ablation flags
    exploit_every_updates: int = 10,
    disable_cheap_gate: bool = False,
    disable_exploitability: bool = False,
    ablation_name: str = "baseline",
    # Exploit ablation sweep parameters (override config values)
    exploit_eps: Optional[float] = None,
    patience_exploit: Optional[int] = None,
    exploit_M: Optional[int] = None,
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
    
    Paper artifact ablation flags:
    - exploit_every_updates: Max interval between exploitability evals (default 10)
    - disable_cheap_gate: Gate always ON (eval eligible every update)
    - disable_exploitability: Never evaluate exploitability (converge on effort only)
    - ablation_name: Tag for this variant (appears in all output files)
    
    Exploit ablation sweep parameters (override config if provided):
    - exploit_eps: Exploitability threshold for convergence (default: from config or 0.03)
    - patience_exploit: Consecutive passes required for stopping (default: from config or 5)
    - exploit_M: Monte Carlo samples for exploitability (default: from config or 8192)
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
    theory_align_enabled = bool(cfg.get("theory_align", False))
    theory_align_conc_min = float(cfg.get("theory_align_conc_min", 0.0))
    theory_align_conc_weight = float(cfg.get("theory_align_conc_weight", 0.0))
    theory_align_v2_enabled = bool(cfg.get("theory_align_v2", False))
    theory_align_v2_conc_min = float(cfg.get("theory_align_v2_conc_min", 1.0))
    theory_align_v2_conc_scale = float(cfg.get("theory_align_v2_conc_scale", 1.0))
    theory_align_v2_conc_min_start = float(cfg.get("theory_align_v2_conc_min_start", theory_align_v2_conc_min))
    theory_align_v2_conc_scale_start = float(cfg.get("theory_align_v2_conc_scale_start", theory_align_v2_conc_scale))
    theory_align_v2_conc_max = cfg.get("theory_align_v2_conc_max", None)
    if theory_align_v2_conc_max is not None:
        theory_align_v2_conc_max = float(theory_align_v2_conc_max)
    theory_align_v2_var_coef = float(cfg.get("theory_align_v2_var_coef", 0.0))
    theory_align_v2_br_coef = float(cfg.get("theory_align_v2_br_coef", 0.0))
    theory_align_v2_var_coef_start = float(cfg.get("theory_align_v2_var_coef_start", theory_align_v2_var_coef))
    theory_align_v2_ramp_warmup = int(cfg.get("theory_align_v2_ramp_warmup", 0))
    theory_align_v2_ramp_steps = int(cfg.get("theory_align_v2_ramp_steps", 0))
    theory_align_v2_es_window = int(cfg.get("theory_align_v2_early_stop_window", 20))
    train_side = str(train_side).lower()
    if train_side not in ("p1", "p2"):
        raise ValueError(f"train_side must be 'p1' or 'p2', got '{train_side}'")
    fixed_opponent_enabled = fixed_opponent_effort is not None
    fixed_effort_clamped = None
    if fixed_opponent_enabled:
        fixed_effort_clamped = _clip_effort(float(fixed_opponent_effort), effort_bounds)
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
        kl_early_stop=bool(cfg.get("kl_early_stop", False)),
        kl_stop_patience=int(cfg.get("kl_stop_patience", 1)),
        kl_stop_threshold=cfg.get("kl_stop_threshold"),
        ratio_stop_threshold=cfg.get("ratio_stop_threshold"),
        target_kl=float(cfg.get("target_kl", 0.01)),
        theory_align=theory_align_enabled,
        theory_align_conc_min=theory_align_conc_min,
        theory_align_conc_weight=theory_align_conc_weight,
        theory_align_v2=theory_align_v2_enabled,
        theory_align_v2_conc_min=theory_align_v2_conc_min,
        theory_align_v2_conc_scale=theory_align_v2_conc_scale,
        theory_align_v2_conc_max=theory_align_v2_conc_max,
        theory_align_v2_var_coef=theory_align_v2_var_coef,
        theory_align_v2_br_coef=theory_align_v2_br_coef,
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
    if fixed_opponent_enabled:
        print(
            f"[Experiment] fixed_opponent_effort={fixed_effort_clamped:.2f} train_side={train_side}",
            flush=True,
        )

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
    early_stop_triggered = False
    convergence_cfg = cfg.get("convergence", {}) or {}
    convergence_enabled = bool(convergence_cfg.get("enabled", False))
    conv_eval_every = int(convergence_cfg.get("eval_every_updates", eval_every or 20))
    cheap_cfg = convergence_cfg.get("cheap_gate", {}) if convergence_enabled else {}
    cheap_profiles = convergence_cfg.get("cheap_gate_profiles", {}) if convergence_enabled else {}
    cheap_profile_name = convergence_cfg.get("cheap_gate_profile", "default") if convergence_enabled else "default"
    if convergence_enabled and cheap_profiles:
        selected_profile = cheap_profiles.get(cheap_profile_name)
        if selected_profile is None:
            fallback_name = "default" if "default" in cheap_profiles else next(iter(cheap_profiles))
            print(
                f"[Convergence] cheap_gate_profile='{cheap_profile_name}' not found; using '{fallback_name}' instead.",
                flush=True,
            )
            cheap_profile_name = fallback_name
            selected_profile = cheap_profiles.get(cheap_profile_name, {})
        cheap_cfg = dict(selected_profile)
    if convergence_enabled:
        print(f"[Convergence] cheap_gate_profile={cheap_profile_name}", flush=True)
    exploit_cfg = convergence_cfg.get("exploit", {}) if convergence_enabled else {}
    symmetry_cfg = convergence_cfg.get("symmetry", {}) if convergence_enabled else {}
    cheap_tracker = CheapGateTracker(int(cheap_cfg.get("window_size", 20))) if convergence_enabled else None
    drift_ok_streak = 0
    exploit_ok_streak = 0
    symmetry_fail_streak = 0
    last_mean_kl_window = None
    last_std_kl_window = None
    last_drift_effort = None
    last_exploitability = None
    last_best_dev_effort = None
    last_symmetry_gap = None
    converged_flag = 0
    stop_reason = "max_updates"  # default if we exhaust episodes

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
    # Early-stop rate tracker for TheoryAlignV2 diagnostics
    early_stop_hist = deque(maxlen=theory_align_v2_es_window) if theory_align_v2_enabled else None
    
    # Convergence history tracking for plotting (stores per-update efforts)
    # Extended for paper artifacts with time-series metrics
    convergence_history: Dict[str, Any] = {
        "steps": [],
        "agent1_effort": [],
        "agent2_effort": [],
        "policy_mean_effort": [],
        # Time-series metrics (logged every update)
        "approx_kl": [],
        "batch_entropy": [],
        "alpha_mean": [],
        "beta_mean": [],
        "mean_kl_window": [],
        "drift_effort": [],
        # Exploitability: sparse (NaN when not evaluated)
        "exploitability": [],           # NaN if not evaluated this step
        "exploitability_is_valid": [],  # bool: True if evaluated this step
    }
    # Track steps where exploitability was actually evaluated (for paper)
    exploit_eval_steps: List[int] = []
    last_exploit_eval_step: int = -999999  # Track for periodic evaluation

    while steps_done < total_steps_target:
        if total_updates > 1:
            hist_progress = float(update_idx) / float(total_updates - 1)
            hist_progress = max(0.0, min(1.0, hist_progress))
        else:
            hist_progress = 1.0
        agent.opponent_history_sample_p = history_prob_start + (history_prob_end - history_prob_start) * hist_progress

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

        policy_mean_effort_current: Optional[float] = None
        symmetry_gap_current: Optional[float] = None
        
        # Warmup phase: apply effort bias to create asymmetric start
        # Agent1 starts above theoretical, Agent2 starts below
        warmup_updates = int(cfg.get("asymmetric_warmup_updates", 20))  # 前20个updates应用偏移
        apply_warmup_bias = update_idx < warmup_updates
        
        if apply_warmup_bias:
            # 计算当前q的理论值
            q_for_theory = float(train_qs[0]) if train_qs else float(cfg.get("q", 40.0))
            e_theory = clip_stage2(e_star_two_players(q_for_theory, w_h, w_l, k), effort_bounds)
            # 偏移量随训练逐渐减小
            bias_strength = 1.0 - (float(update_idx) / float(max(1, warmup_updates)))
            # Agent1向上偏移30%，Agent2向下偏移30%
            bias_magnitude = e_theory * 0.3 * bias_strength
            if update_idx == 0:
                print(f"[AsymmetricInit] Warmup for {warmup_updates} updates: Agent1 bias +{bias_magnitude:.2f}, Agent2 bias -{bias_magnitude:.2f}")

        for _ in range(steps_this):
            q = float(rng.choice(train_qs))
            env.q = q

            # Generate states for both players
            s1 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
            s2 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)

            if fixed_opponent_enabled:
                res_p1 = get_effort_for_side(
                    "p1",
                    agent,
                    s1,
                    q,
                    effort_bounds,
                    fixed_effort_clamped,
                    train_side == "p1",
                    cfg.get("seed", 42),
                    last_rollout_stats,
                )
                res_p2 = get_effort_for_side(
                    "p2",
                    agent,
                    s2,
                    q,
                    effort_bounds,
                    fixed_effort_clamped,
                    train_side == "p2",
                    cfg.get("seed", 42),
                    last_rollout_stats,
                )
                a1_norm, e1, logp1, v1 = (
                    res_p1["action_norm"],
                    res_p1["effort"],
                    res_p1["logp"],
                    res_p1["value"],
                )
                a2_norm, e2, logp2, v2 = (
                    res_p2["action_norm"],
                    res_p2["effort"],
                    res_p2["logp"],
                    res_p2["value"],
                )
                use_opponent = False
            else:
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

            # Apply warmup bias to create asymmetric starting points
            if apply_warmup_bias:
                # Agent1: bias upward (higher effort)
                e1_biased = e1 + bias_magnitude
                e1_biased = torch.clamp(e1_biased, effort_bounds[0], effort_bounds[1])
                
                # Agent2: bias downward (lower effort)  
                e2_biased = e2 - bias_magnitude
                e2_biased = torch.clamp(e2_biased, effort_bounds[0], effort_bounds[1])
                
                # Use biased efforts for environment step
                e1_env, e2_env = e1_biased, e2_biased
            else:
                # No bias after warmup
                e1_env, e2_env = e1, e2

            # Execute environment step with both actions
            _, rewards, _, done, _ = env.step((torch.tensor([float(e1_env.item())]), torch.tensor([float(e2_env.item())])))

            # Storage: Mode-dependent logic
            if fixed_opponent_enabled:
                # Fixed-opponent mode: only store train-side transitions to keep PPO on-policy.
                if train_side == "p1":
                    agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
                    stored_p1_this_update += 1
                    rollout_stats.update_effort(float(e1_env.item()), player="p1")
                else:
                    agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
                    stored_p2_this_update += 1
                    rollout_stats.update_effort(float(e2_env.item()), player="p2")
            else:
                # Player 1: ALWAYS store (learner-generated in both modes)
                agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
                stored_p1_this_update += 1
                # Track P1's sampled effort (always learner-generated)
                rollout_stats.update_effort(float(e1_env.item()), player="p1")

                # Player 2: Mode-dependent storage
                if rollout_mode == "selfplay":
                    # SELFPLAY: Always store player2 (learner-generated)
                    agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
                    stored_p2_this_update += 1
                    # Track P2's sampled effort (learner-generated in selfplay)
                    rollout_stats.update_effort(float(e2_env.item()), player="p2")

                else:  # rollout_mode == "vs_opponent"
                    # VS_OPPONENT: Only store player2 when it used learner policy
                    if not use_opponent:
                        # Player2 used learner -> store for PPO update
                        agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
                        stored_p2_this_update += 1
                        # Track P2's sampled effort (learner-generated)
                        rollout_stats.update_effort(float(e2_env.item()), player="p2")
                    else:
                        # Player2 used opponent -> treat as environment dynamics, don't store
                        skipped_p2_this_update += 1
                        # NOTE: Do NOT track P2 effort when using opponent policy

            history.append(float((e1_env.item() + e2_env.item()) / 2.0))
        last_update_metrics = agent.update()
        if theory_align_v2_enabled and early_stop_hist is not None and last_update_metrics:
            early_stop_hist.append(1.0 if last_update_metrics.get("early_stop_triggered") else 0.0)
        
        # Capture rollout stats snapshot for this update before reset
        last_rollout_stats = rollout_stats.get_summary()
        if last_rollout_stats:
            p1_mean = last_rollout_stats.get("sample_avg_effort_p1")
            p2_mean = last_rollout_stats.get("sample_avg_effort_p2")
            if p1_mean is not None and p2_mean is not None:
                symmetry_gap_current = abs(float(p1_mean) - float(p2_mean))
        
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
        conc_values: List[float] = []
        conc_values_v2: List[float] = []
        var_effort_values_v2: List[float] = []
        ta_policy_mean_effort: Optional[float] = None
        ta_sample_avg_effort: Optional[float] = None
        ta_mean_vs_sample_gap: Optional[float] = None
        ta_v2_policy_mean_effort: Optional[float] = None
        ta_v2_sample_avg_effort: Optional[float] = None
        ta_v2_mean_vs_sample_gap: Optional[float] = None
        try:
            for q_eval in eval_qs:
                e2_star_val = clip_stage2(e_star_two_players(q_eval, w_h, w_l, k), effort_bounds)  # theoretical optimal
                s_eval = agent.state_from_params(q=float(q_eval), k=k, w_h=w_h, w_l=w_l)  # build normalized state
                with torch.no_grad():
                    dist, _ = agent.dist(s_eval)  # get Beta distribution
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
                
                # Per-player sampled efforts (for convergence tracking)
                sample_avg_effort_p1 = last_rollout_stats.get("sample_avg_effort_p1") if last_rollout_stats else None
                sample_avg_effort_p2 = last_rollout_stats.get("sample_avg_effort_p2") if last_rollout_stats else None
                
                # Record convergence history if both p1 and p2 efforts are available
                if sample_avg_effort_p1 is not None and sample_avg_effort_p2 is not None:
                    convergence_history["steps"].append(steps_done)
                    convergence_history["agent1_effort"].append(float(sample_avg_effort_p1))
                    convergence_history["agent2_effort"].append(float(sample_avg_effort_p2))
                    convergence_history["policy_mean_effort"].append(float(final_e2_eval))
                    # Extended time-series metrics for paper artifacts
                    _kl_for_hist = last_update_metrics.get("approx_kl", float("nan")) if last_update_metrics else float("nan")
                    _entropy_for_hist = last_update_metrics.get("entropy_mean", float("nan")) if last_update_metrics else float("nan")
                    convergence_history["approx_kl"].append(float(_kl_for_hist) if math.isfinite(_kl_for_hist) else float("nan"))
                    convergence_history["batch_entropy"].append(float(_entropy_for_hist) if math.isfinite(_entropy_for_hist) else float("nan"))
                    convergence_history["alpha_mean"].append(float(alpha_mean))
                    convergence_history["beta_mean"].append(float(beta_mean))
                    # mean_kl_window and drift_effort will be filled after cheap_tracker.compute() or as NaN
                    # These are placeholders; they will be updated in the convergence eval section
                    convergence_history["mean_kl_window"].append(float("nan"))
                    convergence_history["drift_effort"].append(float("nan"))
                    # Exploitability: NaN placeholder (updated in convergence eval section if evaluated)
                    convergence_history["exploitability"].append(float("nan"))
                    convergence_history["exploitability_is_valid"].append(False)
                
                # mean_vs_sample_gap: policy_mean_effort - sample_avg_effort
                # Positive means policy predicts higher effort than sampled average
                mean_vs_sample_gap = final_e2_eval - sample_avg_effort
                
                gap = abs(final_e2_eval - e2_star_val)
                kl_val = last_update_metrics.get("approx_kl", float("nan")) if last_update_metrics else float("nan")
                kl_proxy_max = last_update_metrics.get("kl_proxy_max", float("nan")) if last_update_metrics else float("nan")
                ratio_max = last_update_metrics.get("ratio_max", float("nan")) if last_update_metrics else float("nan")
                clip_frac_max = last_update_metrics.get("clip_frac_max", float("nan")) if last_update_metrics else float("nan")
                approx_kl_max_abs = last_update_metrics.get("approx_kl_max_abs", float("nan")) if last_update_metrics else float("nan")
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
                if policy_mean_effort_current is None:
                    policy_mean_effort_current = final_e2_eval
                update_line = (
                    f"[Update {upd_i}] q={q_eval}: e*={e2_star_val:.2f}, policy={final_e2_eval:.2f}, gap={gap:.2f}, "
                    f"entropy={agent.cfg.entropy_coef:.3f}, lag_prob={lag_prob:.2f}, "
                    f"approx_kl={kl_val:.4f}, kl_proxy_max={kl_proxy_max:.4f}, ratio_max={ratio_max:.4f}, "
                    f"clip_frac_max={clip_frac_max:.4f}, approx_kl_max_abs={approx_kl_max_abs:.4f}, "
                    f"alpha_mean={alpha_mean:.2f}, beta_mean={beta_mean:.2f}"
                )
                if fixed_opponent_enabled:
                    update_line += f", opponent_effort_used={fixed_effort_clamped:.2f}"
                print(update_line)
                # Rollout sample metrics line
                rollout_line = f"  [Rollout] sample_avg_effort={sample_avg_effort:.2f}, mean_vs_sample_gap={mean_vs_sample_gap:.2f}, effort_samples={effort_sample_count}"
                # Add per-player efforts if available (for convergence tracking)
                if sample_avg_effort_p1 is not None and sample_avg_effort_p2 is not None:
                    rollout_line += f", p1_effort={sample_avg_effort_p1:.2f}, p2_effort={sample_avg_effort_p2:.2f}"
                print(rollout_line)
                if theory_align_enabled:
                    conc_vals = (dist.concentration1 + dist.concentration0).detach().cpu().view(-1).tolist()
                    conc_values.extend([float(v) for v in conc_vals])
                    if ta_policy_mean_effort is None:
                        ta_policy_mean_effort = final_e2_eval
                        ta_sample_avg_effort = sample_avg_effort
                        ta_mean_vs_sample_gap = mean_vs_sample_gap
                if theory_align_v2_enabled:
                    conc_vals_v2 = (dist.concentration1 + dist.concentration0).detach().cpu().view(-1).tolist()
                    conc_values_v2.extend([float(v) for v in conc_vals_v2])
                    alpha = dist.concentration1
                    beta = dist.concentration0
                    conc_val = alpha + beta
                    denom = (conc_val * conc_val) * (conc_val + 1.0) + 1e-8
                    var_action = (alpha * beta) / denom
                    var_effort = var_action * ((effort_bounds[1] - effort_bounds[0]) ** 2)
                    var_effort_values_v2.extend(var_effort.detach().cpu().view(-1).tolist())
                    if ta_v2_policy_mean_effort is None:
                        ta_v2_policy_mean_effort = final_e2_eval
                        ta_v2_sample_avg_effort = sample_avg_effort
                        ta_v2_mean_vs_sample_gap = mean_vs_sample_gap
                if last_update_metrics and last_update_metrics.get("early_stop_triggered"):
                    print(
                        f"  [EarlyStop] triggered={last_update_metrics.get('early_stop_triggered')} "
                        f"reason={last_update_metrics.get('early_stop_reason','')} "
                        f"epochs_completed={last_update_metrics.get('epochs_completed')} "
                        f"mb_completed={last_update_metrics.get('minibatches_completed')} "
                        f"clip_eps_used={last_update_metrics.get('clip_eps_used')} "
                        f"ratio_thr_used={last_update_metrics.get('ratio_stop_threshold_used')} "
                        f"kl_thr_used={last_update_metrics.get('kl_stop_threshold_used')}",
                        flush=True,
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
            if theory_align_enabled and conc_values:
                conc_arr = np.asarray(conc_values, dtype=float)
                conc_mean = float(np.mean(conc_arr))
                conc_p10 = float(np.percentile(conc_arr, 10))
                conc_p50 = float(np.percentile(conc_arr, 50))
                conc_p90 = float(np.percentile(conc_arr, 90))
                if ta_policy_mean_effort is None:
                    ta_policy_mean_effort = policy_mean_effort_current if policy_mean_effort_current is not None else float("nan")
                if ta_sample_avg_effort is None:
                    ta_sample_avg_effort = last_rollout_stats.get("sample_avg_effort", float("nan")) if last_rollout_stats else float("nan")
                if ta_mean_vs_sample_gap is None:
                    if math.isfinite(ta_policy_mean_effort) and math.isfinite(ta_sample_avg_effort):
                        ta_mean_vs_sample_gap = ta_policy_mean_effort - ta_sample_avg_effort
                    else:
                        ta_mean_vs_sample_gap = float("nan")
                print(
                    f"[TheoryAlign] ent={agent.cfg.entropy_coef:.3f} conc_mean={conc_mean:.2f} "
                    f"conc_p10={conc_p10:.2f} conc_p50={conc_p50:.2f} conc_p90={conc_p90:.2f} "
                    f"policy_mean_effort={ta_policy_mean_effort:.2f} "
                    f"sample_avg_effort={ta_sample_avg_effort:.2f} "
                    f"mean_vs_sample_gap={ta_mean_vs_sample_gap:.2f}",
                    flush=True,
                )
            if theory_align_v2_enabled and conc_values_v2:
                conc_arr_v2 = np.asarray(conc_values_v2, dtype=float)
                conc_mean_v2 = float(np.mean(conc_arr_v2))
                conc_p10_v2 = float(np.percentile(conc_arr_v2, 10))
                conc_p50_v2 = float(np.percentile(conc_arr_v2, 50))
                conc_p90_v2 = float(np.percentile(conc_arr_v2, 90))
                var_mean_v2 = float(np.mean(var_effort_values_v2)) if var_effort_values_v2 else float("nan")
                var_p90_v2 = float(np.percentile(var_effort_values_v2, 90)) if var_effort_values_v2 else float("nan")
                if ta_v2_policy_mean_effort is None:
                    ta_v2_policy_mean_effort = policy_mean_effort_current if policy_mean_effort_current is not None else float("nan")
                if ta_v2_sample_avg_effort is None:
                    ta_v2_sample_avg_effort = last_rollout_stats.get("sample_avg_effort", float("nan")) if last_rollout_stats else float("nan")
                if ta_v2_mean_vs_sample_gap is None:
                    if math.isfinite(ta_v2_policy_mean_effort) and math.isfinite(ta_v2_sample_avg_effort):
                        ta_v2_mean_vs_sample_gap = ta_v2_policy_mean_effort - ta_v2_sample_avg_effort
                    else:
                        ta_v2_mean_vs_sample_gap = float("nan")
                early_stop_rate = float(np.mean(early_stop_hist)) if early_stop_hist else float("nan")
                early_stop_window = len(early_stop_hist) if early_stop_hist is not None else 0
                print(
                    f"[TheoryAlignV2] ent={agent.cfg.entropy_coef:.3f} conc_mean={conc_mean_v2:.2f} "
                    f"conc_p10={conc_p10_v2:.2f} conc_p50={conc_p50_v2:.2f} conc_p90={conc_p90_v2:.2f} "
                    f"var_effort_mean={var_mean_v2:.3f} var_effort_p90={var_p90_v2:.3f} "
                    f"policy_mean_effort={ta_v2_policy_mean_effort:.2f} "
                    f"sample_avg_effort={ta_v2_sample_avg_effort:.2f} "
                    f"mean_vs_sample_gap={ta_v2_mean_vs_sample_gap:.2f} "
                    f"early_stop_rate={early_stop_rate:.2f}({early_stop_window})",
                    flush=True,
                )
        except Exception as _e:
            # Keep training robust to any eval hiccup
            pass
        update_idx += 1
        steps_done += steps_this

        # === Convergence evaluation (optional, default OFF) ===
        # Ablation flags: disable_cheap_gate, disable_exploitability, exploit_every_updates
        if convergence_enabled and cheap_tracker is not None and rollout_mode == "selfplay":
            policy_effort_source = "policy_eval"
            if policy_mean_effort_current is None and last_rollout_stats:
                policy_mean_effort_current = last_rollout_stats.get("sample_avg_effort", None)
                policy_effort_source = "rollout_sample"
            if policy_mean_effort_current is None:
                policy_effort_source = "none"
            cheap_tracker.update(kl_val, policy_mean_effort_current)
            do_eval = (update_idx + 1) % conv_eval_every == 0
            if do_eval:
                len_kl = len(cheap_tracker.kl_hist)
                len_policy = len(cheap_tracker.policy_hist)
                gate_stats = cheap_tracker.compute()
                mean_kl_window = gate_stats["mean_kl_window"]
                std_kl_window = gate_stats["std_kl_window"]
                drift_effort = gate_stats["drift_effort"]
                last_mean_kl_window = mean_kl_window
                last_std_kl_window = std_kl_window
                last_drift_effort = drift_effort
                
                # Update convergence_history with latest mean_kl_window and drift_effort
                if convergence_history["steps"] and len(convergence_history["mean_kl_window"]) > 0:
                    # Update the last entry with computed values
                    if mean_kl_window is not None:
                        convergence_history["mean_kl_window"][-1] = float(mean_kl_window)
                    if drift_effort is not None:
                        convergence_history["drift_effort"][-1] = float(drift_effort)

                mean_thresh = float(cheap_cfg.get("mean_kl_thresh", 0.0045))
                std_thresh = float(cheap_cfg.get("std_kl_thresh", 0.0035))
                drift_thresh = float(cheap_cfg.get("drift_effort_thresh", 2.0))
                patience_drift = int(cheap_cfg.get("patience_drift", 2))
                # CLI override for exploit params, else fall back to config
                exploit_eps_val = exploit_eps if exploit_eps is not None else float(exploit_cfg.get("exploit_eps", 0.03))
                patience_exploit_val = patience_exploit if patience_exploit is not None else int(exploit_cfg.get("patience_exploit", 5))
                exploit_M_val = exploit_M if exploit_M is not None else int(exploit_cfg.get("M", 8192))
                mean_ok = mean_kl_window is not None and mean_kl_window <= mean_thresh
                std_ok = std_kl_window is not None and std_kl_window <= std_thresh
                drift_ok = drift_effort is not None and drift_effort <= drift_thresh
                
                # Gate logic: if --disable-cheap-gate, gate always passes
                if disable_cheap_gate:
                    drift_pass = True  # Gate always ON
                else:
                    drift_pass = mean_ok and std_ok and drift_ok
                if drift_pass:
                    drift_ok_streak += 1
                else:
                    drift_ok_streak = 0
                exploitability_val = None
                best_dev_effort = None

                # Symmetry check (only when we have p1/p2 means)
                symmetry_gap_val = symmetry_gap_current
                last_symmetry_gap = symmetry_gap_val
                if symmetry_gap_val is not None:
                    if symmetry_gap_val > float(symmetry_cfg.get("symmetry_gap_thresh", 0.5)):
                        symmetry_fail_streak += 1
                    else:
                        symmetry_fail_streak = 0
                    if symmetry_fail_streak >= int(symmetry_cfg.get("symmetry_fail_patience", 3)):
                        raise RuntimeError(
                            f"[convergence] Symmetry gap persisted: gap={symmetry_gap_val:.3f} over {symmetry_fail_streak} evals"
                        )
                else:
                    symmetry_fail_streak = 0

                # Exploitability evaluation logic with ablation flags
                # --disable-exploitability: never evaluate
                # --exploit-every-updates N: caps max interval between evals
                # --disable-cheap-gate: gate always ON, combined with above evals every N updates
                steps_since_last_exploit = update_idx - last_exploit_eval_step
                periodic_due = steps_since_last_exploit >= exploit_every_updates
                gate_triggered = drift_pass and drift_ok_streak >= patience_drift
                
                # Determine whether to run exploitability
                if disable_exploitability:
                    run_exploit = False
                else:
                    # Eval if periodic OR (gate passed AND not too frequent)
                    run_exploit = periodic_due or (gate_triggered and steps_since_last_exploit >= 1)
                
                if run_exploit:
                    last_exploit_eval_step = update_idx
                    exploit_eval_steps.append(steps_done)
                    # Run exploitability (coarse-to-fine grid) using CRN
                    eval_seed = int(cfg.get("seed", 42)) + int(update_idx + 1)
                    grid_cfg = exploit_cfg.get("grid", {}) if exploit_cfg else {}
                    q_for_exploit = eval_qs[0] if eval_qs else float(cfg.get("q", q_init))
                    exploit_res = eval_exploitability(
                        agent,
                        q=q_for_exploit,
                        effort_bounds=effort_bounds,
                        M=exploit_M_val,
                        grid_cfg={
                            "stage_a_step": float(grid_cfg.get("stage_a_step", 5.0)),
                            "stage_b_radius": float(grid_cfg.get("stage_b_radius", 15.0)),
                            "stage_b_step": float(grid_cfg.get("stage_b_step", 1.0)),
                            "stage_c_radius": float(grid_cfg.get("stage_c_radius", 3.0)),
                            "stage_c_step": float(grid_cfg.get("stage_c_step", 0.25)),
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
                    if convergence_history["steps"] and len(convergence_history["exploitability"]) > 0:
                        convergence_history["exploitability"][-1] = float(exploitability_val)
                        convergence_history["exploitability_is_valid"][-1] = True
                    
                    trigger_reason = "periodic" if periodic_due else "cheap_gate"
                    print(
                        f"[ConvergenceDebug] trigger={trigger_reason} eval_seed={eval_seed} "
                        f"candidates={exploit_res.get('num_candidates', 'NA')}",
                        flush=True,
                    )
                    if exploitability_val < exploit_eps_val:
                        exploit_ok_streak += 1
                    else:
                        exploit_ok_streak = 0
                        if not disable_cheap_gate:
                            drift_ok_streak = 0  # reset cheap gate streak on failure
                    if exploit_ok_streak >= patience_exploit_val:
                        converged_flag = 1
                        stop_reason = "exploitability"
                        print(
                            f"[Convergence] Exploitability satisfied for {exploit_ok_streak} evals; stopping training.",
                            flush=True,
                        )
                        early_stop_triggered = True
                        break
                else:
                    if not disable_exploitability:
                        exploit_ok_streak = 0
                    last_exploitability = None
                    last_best_dev_effort = None

                def _fmt(val: float | None, digits: int = 4) -> str:
                    return "NA" if val is None else f"{val:.{digits}f}"

                fail_reasons = []
                if not disable_cheap_gate:
                    if mean_kl_window is None:
                        fail_reasons.append("mean_kl:window")
                    elif not mean_ok:
                        fail_reasons.append("mean_kl")
                    if std_kl_window is None:
                        fail_reasons.append("std_kl:window")
                    elif not std_ok:
                        fail_reasons.append("std_kl")
                    if drift_effort is None:
                        fail_reasons.append("drift:window")
                    elif not drift_ok:
                        fail_reasons.append("drift")
                    if drift_pass and drift_ok_streak < patience_drift:
                        fail_reasons.append("patience")
                if not run_exploit:
                    if disable_exploitability:
                        decision = "EXPLOIT_DISABLED"
                    elif not fail_reasons:
                        fail_reasons.append("waiting_periodic" if not periodic_due else "unknown")
                        decision = f"GATE_FAIL({','.join(fail_reasons)})"
                    else:
                        decision = f"GATE_FAIL({','.join(fail_reasons)})"
                else:
                    decision = "EXPLOIT_OK_STREAK++" if exploitability_val is not None and exploitability_val < exploit_eps_val else "RUN_EXPLOIT"

                mean_status = f"{_fmt(mean_kl_window)}({'OK' if mean_ok else 'FAIL'}<={mean_thresh:.4f})"
                std_status = f"{_fmt(std_kl_window)}({'OK' if std_ok else 'FAIL'}<={std_thresh:.4f})"
                drift_status = f"{_fmt(drift_effort, 2)}({'OK' if drift_ok else 'FAIL'}<={drift_thresh:.2f})"
                gate_mode = "DISABLED" if disable_cheap_gate else "enabled"
                print(
                    f"[ConvergenceDebug] upd={update_idx + 1} len_kl={len_kl} len_pol={len_policy} "
                    f"mean_kl={mean_status} std_kl={std_status} drift={drift_status} "
                    f"drift_streak={drift_ok_streak}/{patience_drift} source={policy_effort_source} gate={gate_mode} => {decision}",
                    flush=True,
                )

                # Logging line
                print(
                    f"[Convergence] upd={update_idx + 1} mean_kl={mean_kl_window} std_kl={std_kl_window} "
                    f"drift_effort={drift_effort} exploitability={exploitability_val if exploitability_val is not None else 'NA'} "
                    f"best_dev={best_dev_effort if best_dev_effort is not None else 'NA'} "
                    f"sym_gap={symmetry_gap_val if symmetry_gap_val is not None else 'NA'} "
                    f"streaks: drift={drift_ok_streak}/{patience_drift} "
                    f"exploit={exploit_ok_streak}/{patience_exploit_val}",
                    flush=True,
                )

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
                if theory_align_enabled:
                    dist, _ = policy_net.dist(
                        state,
                        theory_align=True,
                        theory_align_conc_min=theory_align_conc_min,
                    )
                else:
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
            dist_agent, _ = agent.dist(s_agent)
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
        row["kl_proxy_max"] = last_update_metrics.get("kl_proxy_max", float("nan")) if last_update_metrics else float("nan")
        row["kl_proxy_mean"] = last_update_metrics.get("kl_proxy_mean", float("nan")) if last_update_metrics else float("nan")
        row["ratio_max"] = last_update_metrics.get("ratio_max", float("nan")) if last_update_metrics else float("nan")
        row["ratio_mean"] = last_update_metrics.get("ratio_mean", float("nan")) if last_update_metrics else float("nan")
        row["clip_frac_max"] = last_update_metrics.get("clip_frac_max", float("nan")) if last_update_metrics else float("nan")
        row["clip_frac_mean"] = last_update_metrics.get("clip_frac_mean", float("nan")) if last_update_metrics else float("nan")
        row["approx_kl_max_abs"] = last_update_metrics.get("approx_kl_max_abs", float("nan")) if last_update_metrics else float("nan")
        
        # === NEW INSTRUMENTATION COLUMNS ===
        # Policy mean effort (confirmed: Beta mean α/(α+β) scaled to effort range)
        policy_mean_effort = compute_policy_mean_effort(alpha_eval, beta_eval, effort_bounds[0], effort_bounds[1])
        row["policy_mean_effort"] = policy_mean_effort
        row["mean_kl_window"] = last_mean_kl_window if convergence_enabled else float("nan")
        row["std_kl_window"] = last_std_kl_window if convergence_enabled else float("nan")
        row["drift_effort"] = last_drift_effort if convergence_enabled else float("nan")
        row["exploitability"] = last_exploitability if convergence_enabled else float("nan")
        row["best_dev_effort"] = last_best_dev_effort if convergence_enabled else float("nan")
        row["symmetry_gap"] = last_symmetry_gap if convergence_enabled else float("nan")
        row["drift_ok_streak"] = drift_ok_streak if convergence_enabled else 0
        row["exploit_ok_streak"] = exploit_ok_streak if convergence_enabled else 0
        row["converged_flag"] = converged_flag if convergence_enabled else 0
        
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
    
    # Save convergence history for each trained q value (extended format for paper artifacts)
    if convergence_history["steps"]:  # Only save if we have data
        convergence_dir = os.path.join("results", "two_players", "convergence")
        os.makedirs(convergence_dir, exist_ok=True)
        
        # Get seed from config (for filename)
        seed_val = int(cfg.get("seed", 42))
        
        for q_val in train_qs:
            # Compute final effort for this q value
            _theo_effort = float(clip_stage2(
                e_star_two_players(q_val, w_h, w_l, k), tuple(effort_bounds)
            ))
            _s = agent.state_from_params(q=float(q_val), k=k, w_h=w_h, w_l=w_l)
            with torch.no_grad():
                _dist, _ = agent.dist(_s)
                _alpha = float(_dist.concentration1.mean().item())
                _beta = float(_dist.concentration0.mean().item())
            _final_effort = compute_policy_mean_effort(_alpha, _beta, effort_bounds[0], effort_bounds[1])
            _gap = abs(_final_effort - _theo_effort)

            convergence_data = {
                "algorithm": "PPO",
                "q": float(q_val),
                "seed": seed_val,
                "ablation_name": ablation_name,
                "theoretical_effort": _theo_effort,
                # Structured dicts expected by sweep parser
                "theoretical": {
                    "effort": _theo_effort,
                },
                "final": {
                    "effort": _final_effort,
                    "gap": _gap,
                },
                # Stopping info (required by sweep_exploit_ablation.py)
                "stop_reason": stop_reason,
                "stopped_at_update": update_idx,
                "joint_exploit_ok_streak": exploit_ok_streak,
                "final_exploit_1": last_exploitability,
                "final_exploit_2": last_exploitability,  # symmetric game
                "final_exploit_max": last_exploitability,
                "final_br_effort_1": last_best_dev_effort,
                "final_br_effort_2": last_best_dev_effort,  # symmetric game
                # Core effort series
                "steps": convergence_history["steps"],
                "agent1_effort": convergence_history["agent1_effort"],
                "agent2_effort": convergence_history["agent2_effort"],
                "policy_mean_effort": convergence_history["policy_mean_effort"],
                # Extended time-series metrics (paper artifacts)
                "approx_kl": convergence_history["approx_kl"],
                "batch_entropy": convergence_history["batch_entropy"],
                "alpha_mean": convergence_history["alpha_mean"],
                "beta_mean": convergence_history["beta_mean"],
                "mean_kl_window": convergence_history["mean_kl_window"],
                "drift_effort": convergence_history["drift_effort"],
                # Exploitability (sparse: NaN when not evaluated)
                "exploitability": convergence_history["exploitability"],
                "exploitability_is_valid": convergence_history["exploitability_is_valid"],
                "exploit_eval_steps": exploit_eval_steps,
                # Metadata
                "rollout_mode": rollout_mode,
                "total_episodes": episodes,
                "disable_cheap_gate": disable_cheap_gate,
                "disable_exploitability": disable_exploitability,
                "exploit_every_updates": exploit_every_updates,
                "exploit_config": {
                    "exploit_eps": exploit_eps if exploit_eps is not None else float(exploit_cfg.get("exploit_eps", 0.03)),
                    "patience_exploit": patience_exploit if patience_exploit is not None else int(exploit_cfg.get("patience_exploit", 5)),
                    "exploit_every_updates": exploit_every_updates,
                    "exploit_M": exploit_M if exploit_M is not None else int(exploit_cfg.get("M", 8192)),
                    "disable_cheap_gate": disable_cheap_gate,
                    "disable_exploitability": disable_exploitability,
                },
            }
            
            # Build filename with seed and ablation (new format)
            # Format: ppo_q{q}_seed{seed}_{ablation}_convergence.json
            if ablation_name == "baseline":
                convergence_file = os.path.join(
                    convergence_dir,
                    f"ppo_q{q_val:.1f}_seed{seed_val}_convergence.json"
                )
            else:
                convergence_file = os.path.join(
                    convergence_dir,
                    f"ppo_q{q_val:.1f}_seed{seed_val}_{ablation_name}_convergence.json"
                )
            
            with open(convergence_file, 'w') as f:
                json.dump(convergence_data, f, indent=2)
            print(f"[PPO] Saved convergence history to {convergence_file}")
            
            # Write metadata.json for this run (for reproducibility)
            metadata_file = convergence_file.replace("_convergence.json", "_metadata.json")
            metadata = {
                "run_id": run_id if run_id else datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                "method": "ppo",
                "q": float(q_val),
                "seed": seed_val,
                "ablation_name": ablation_name,
                "git_sha": _get_git_sha(),
                "cmdline": " ".join(sys.argv),
                "timestamp": datetime.datetime.now().isoformat(),
                "convergence_file": os.path.basename(convergence_file),
                "variant_name": variant_name,
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"[PPO] Saved metadata to {metadata_file}")

    return rows


def _run_cli(args: argparse.Namespace) -> str:
    # === Apply method-specific defaults ===
    # For PPO: enable modern defaults (selfplay, theory-align-v2, convergence-eval, relaxed profile)
    # For gradient: keep traditional defaults
    if args.method == "ppo":
        if args.rollout_mode is None:
            args.rollout_mode = "selfplay"
            print("[config] PPO default: rollout_mode='selfplay'")
        
        if args.theory_align_v2 is None:
            args.theory_align_v2 = True
            print("[config] PPO default: theory_align_v2=True")
        
        if args.enable_convergence_eval is None:
            args.enable_convergence_eval = True
            print("[config] PPO default: enable_convergence_eval=True")
        
        if args.cheap_gate_profile is None:
            args.cheap_gate_profile = "relaxed"
            print("[config] PPO default: cheap_gate_profile='relaxed'")
    else:  # gradient
        if args.rollout_mode is None:
            args.rollout_mode = "vs_opponent"
        if args.theory_align_v2 is None:
            args.theory_align_v2 = False
        if args.enable_convergence_eval is None:
            args.enable_convergence_eval = False
    
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

    if args.theory_align and args.theory_align_v2:
        raise ValueError("[config] ERROR: --theory-align and --theory-align-v2 are mutually exclusive.")
    if args.theory_align:
        cfg["entropy_coef_start"] = 0.0
        cfg["entropy_coef_hold"] = 0.0
        cfg["entropy_coef_end"] = 0.0
        cfg["theory_align"] = True
        cfg["theory_align_conc_min"] = 100.0
        cfg["theory_align_conc_weight"] = 5e-4
        print(
            "[TheoryAlign] enabled: entropy_coef_start/hold/end=0, "
            "conc_min=100.0, conc_weight=5e-4",
            flush=True,
        )
    if args.theory_align_v2:
        cfg["entropy_coef_start"] = 0.0
        cfg["entropy_coef_hold"] = 0.0
        cfg["entropy_coef_end"] = 0.0
        cfg["theory_align_v2"] = True
        cfg["theory_align_v2_conc_min"] = 1000.0
        cfg["theory_align_v2_conc_scale"] = 10000.0
        cfg["theory_align_v2_conc_max"] = 100000.0
        cfg["theory_align_v2_var_coef"] = 5e-2
        cfg["theory_align_v2_br_coef"] = 3e-3
        cfg["theory_align_v2_conc_min_start"] = 100.0
        cfg["theory_align_v2_conc_scale_start"] = 100.0
        cfg["theory_align_v2_var_coef_start"] = 0.0
        cfg["theory_align_v2_ramp_warmup"] = 20
        cfg["theory_align_v2_ramp_steps"] = 50
        cfg["theory_align_v2_early_stop_window"] = 20
        # Stability tweaks (v2 only): reduce update aggressiveness.
        cfg["lr_start"] = min(float(cfg.get("lr_start", 3e-4)), 1e-4)
        cfg["lr_end"] = min(float(cfg.get("lr_end", 2e-4)), 5e-5)
        cfg["update_epochs"] = min(int(cfg.get("update_epochs", 6)), 2)
        cfg["clip_range_start"] = min(float(cfg.get("clip_range_start", 0.5)), 0.3)
        cfg["clip_range_end"] = min(float(cfg.get("clip_range_end", 0.35)), 0.2)
        cfg["target_kl"] = min(float(cfg.get("target_kl", 0.08)), 0.06)
        cfg["force_kl_gate"] = False
        cfg["kl_clip_factor_up"] = 1.0
        cfg["kl_lr_factor_up"] = 1.0
        cfg["clip_ceiling"] = 0.35
        cfg["ratio_stop_threshold"] = 3.0
        cfg["update_epochs"] = 1
        cfg["lr_start"] = min(float(cfg.get("lr_start", 1e-4)), 5e-5)
        cfg["lr_end"] = min(float(cfg.get("lr_end", 5e-5)), 2e-5)
        cfg["clip_range_start"] = min(float(cfg.get("clip_range_start", 0.3)), 0.2)
        cfg["clip_range_end"] = min(float(cfg.get("clip_range_end", 0.2)), 0.15)
        cfg["ratio_stop_threshold"] = 2.2
        cfg["max_grad_norm"] = 0.25
        cfg["value_coef"] = 1.0
        cfg["gae_lambda"] = 1.0
        cfg["gamma"] = 1.0
        print(
            "[TheoryAlignV2] enabled: entropy=0, mean+conc head, var_coef=5e-2, "
            "conc_min=1000, conc_scale=10000, conc_max=100000, ramp_warmup=20, ramp_steps=50, "
            "lr/clip/epochs softened",
            flush=True,
        )
    
    # --disable-entropy: zero entropy regularization (mechanism ablation)
    if args.disable_entropy:
        cfg["entropy_coef_start"] = 0.0
        cfg["entropy_coef_hold"] = 0.0
        cfg["entropy_coef_end"] = 0.0
        print("[ablation] --disable-entropy: entropy_coef=0 throughout training", flush=True)

    # === PPO training dynamics override (separate from schedule endpoints) ===
    # This is NOT subject to mutual exclusion - can be combined with other overrides.
    if args.override_update_epochs is not None:
        if args.override_update_epochs < 1:
            raise ValueError(
                f"[config] ERROR: --override-update-epochs must be >= 1, got {args.override_update_epochs}"
            )
        cfg["update_epochs"] = int(args.override_update_epochs)
        print(f"[config] Override: update_epochs -> {args.override_update_epochs}", flush=True)
    
    # Allow explicit override of variant_name via CLI (sweep script may pass this)
    if args.variant_name is not None:
        variant_name = args.variant_name
    if args.fixed_opponent_effort is not None:
        fixed_suffix = f"fixedopp{args.fixed_opponent_effort:g}_{args.train_side}"
        variant_name = f"{variant_name}_{fixed_suffix}" if variant_name else fixed_suffix
    
    # Generate run_id: use CLI-provided value or generate from current timestamp
    if args.run_id is not None:
        run_id = args.run_id
    else:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[config] run_id={run_id} variant_name={variant_name}", flush=True)

    # Use new v2 CSV path with run_id and variant_name columns
    csv_path = os.path.join("results", "two_players", "summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Convergence eval enable flag (preserves default off)
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
                lr_decay=args.grad_lr_decay,
                symmetry_enforce_every=args.grad_symmetry_enforce,
                symmetry_tol=args.grad_symmetry_tol,
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
            fixed_opponent_effort=args.fixed_opponent_effort,
            train_side=args.train_side,
            # Paper artifact ablation flags
            exploit_every_updates=args.exploit_every_updates,
            disable_cheap_gate=args.disable_cheap_gate,
            disable_exploitability=args.disable_exploitability,
            ablation_name=args.ablation_name,
            # Exploit ablation sweep parameters (override config)
            exploit_eps=args.exploit_eps,
            patience_exploit=args.exploit_patience,
            exploit_M=args.exploit_M,
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
        default=None,  # Will be set based on method
        help="Rollout mode for PPO: 'selfplay' (both use learner, store both) or 'vs_opponent' (p2 may use opponent, store only learner samples). Default: 'selfplay' for PPO, 'vs_opponent' for gradient.",
    )
    # Experiment 1 (fixed opponent best-response):
    # python run/run_two_players.py --method ppo --rollout-mode selfplay --q 40 --episodes 2048000 --seed 42 \
    #   --fixed-opponent-effort 54.69 --train-side p1
    parser.add_argument(
        "--fixed-opponent-effort",
        type=float,
        default=None,
        help="Enable fixed-opponent mode with constant effort (clamped to effort bounds).",
    )
    parser.add_argument(
        "--train-side",
        choices=["p1", "p2"],
        default="p1",
        help="Which side to train when fixed-opponent mode is enabled.",
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
    parser.add_argument("--grad-tol", type=float, default=base_config.get("gradient_tol", 1e-5), help="Terminate when |grad| < tol.")
    parser.add_argument("--grad-samples", type=int, default=base_config.get("gradient_num_samples", 256), help="Monte Carlo samples for uniform-noise gradients.")
    parser.add_argument("--grad-init-perturb", type=float, default=base_config.get("gradient_init_perturb", 1.0), help="Initial asymmetry to avoid symmetric starts.")
    parser.add_argument("--grad-lr-decay", type=float, default=base_config.get("gradient_lr_decay", 0.9995), help="Exponential decay factor for learning rate.")
    parser.add_argument("--grad-symmetry-enforce", type=int, default=base_config.get("gradient_symmetry_enforce_every", 50), help="Force symmetry every N steps (0 to disable).")
    parser.add_argument("--grad-symmetry-tol", type=float, default=base_config.get("gradient_symmetry_tol", 0.1), help="Symmetry tolerance for convergence criterion.")
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
    parser.add_argument(
        "--override-update-epochs",
        type=int,
        default=None,
        help="Override PPO update_epochs (number of optimization epochs per update). Must be >= 1.",
    )
    parser.add_argument(
        "--theory-align",
        action="store_true",
        help="EXPERIMENT: keep PPO stochastic but collapse policy variance to align with pure-strategy theory.",
    )
    parser.add_argument(
        "--theory-align-v2",
        action="store_true",
        dest="theory_align_v2",
        help="Enable theory-align-v2. Default: enabled for PPO, disabled for gradient.",
    )
    parser.add_argument(
        "--no-theory-align-v2",
        action="store_false",
        dest="theory_align_v2",
        help="Disable theory-align-v2 (override PPO default).",
    )
    parser.set_defaults(theory_align_v2=None)  # Will be set based on method
    
    parser.add_argument(
        "--enable-convergence-eval",
        action="store_true",
        dest="enable_convergence_eval",
        help="Enable convergence evaluation/early-stop. Default: enabled for PPO, disabled for gradient.",
    )
    parser.add_argument(
        "--no-convergence-eval",
        action="store_false",
        dest="enable_convergence_eval",
        help="Disable convergence evaluation (override PPO default).",
    )
    parser.set_defaults(enable_convergence_eval=None)  # Will be set based on method
    cheap_gate_profiles = base_config.get("convergence", {}).get("cheap_gate_profiles", {}) or {}
    cheap_gate_profile_choices = sorted(cheap_gate_profiles.keys()) if cheap_gate_profiles else ["default", "conservative", "aggressive"]
    parser.add_argument(
        "--cheap-gate-profile",
        choices=cheap_gate_profile_choices,
        default=None,  # Will be set based on method
        help="Select cheap gate profile (default/conservative/aggressive/relaxed). Default: 'relaxed' for PPO with theory-align-v2, 'default' otherwise.",
    )
    # Smoke test: `python run/run_two_players.py --method ppo --rollout-mode selfplay --episodes 40960 --enable-convergence-eval`
    # Expect: after a few eval points, ConvergenceDebug shows RUN_EXPLOIT and CSV exploitability is not NA if gate passes.
    # Verification: FAIL reasons are explicit when mean/std/drift exceed thresholds; no RUN_EXPLOIT in that case.
    # Trigger: when mean_kl approx 0.002-0.004, std_kl approx 0.002-0.004, drift approx 1-2 for consecutive windows,
    # expect RUN_EXPLOIT within <=3 eval points. Tuning: relax only failing threshold by 10-20%.
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
    # === Paper artifact flags (ablation + time-series logging) ===
    parser.add_argument(
        "--exploit-every-updates",
        type=int,
        default=10,
        help="Maximum interval between exploitability evaluations (caps worst-case cost). Default: 10.",
    )
    parser.add_argument(
        "--disable-entropy",
        action="store_true",
        help="Set entropy_coef to 0 throughout training (mechanism ablation).",
    )
    parser.add_argument(
        "--disable-cheap-gate",
        action="store_true",
        help="Gate always ON: exploitability evaluation is eligible every update (combined with --exploit-every-updates N, evals every N updates).",
    )
    parser.add_argument(
        "--disable-exploitability",
        action="store_true",
        help="Never evaluate exploitability; convergence is based on effort gap only. All exploitability values = NaN.",
    )
    parser.add_argument(
        "--ablation-name",
        type=str,
        default="baseline",
        help="Ablation variant name for paper artifacts. Included in JSON, CSV, and metadata. Default: 'baseline'.",
    )
    # Exploit ablation sweep parameters (override config values)
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
    args = parser.parse_args()

    log_path = _build_log_path(args)
    with _tee_console_to_file(log_path):
        print(f"[log] Mirroring console output to {log_path}")
        _run_cli(args)
        print(f"[log] Run complete. Full console trace saved to {log_path}")


if __name__ == "__main__":
    main()

 
