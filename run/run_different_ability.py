#!/usr/bin/env python3
"""
Different-Ability Two-Player Experiment (True PPO + Gradient)
=============================================================

This script runs the two-player different-ability scenario (l1 > l2, k1=k2)
with two options:
- gradient: uses agents.different_ability_solver.different_ability_gradient_descent_solver
- ppo:      uses a true PPO implementation (Beta policy + GAE + clipping)
            from agents.ppo_two_players_clean.PPOTwoPlayersBandit, extended
            with ability features in the state for self-play.

Results are saved to results/tables/different_ability_two_players.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

# Ensure project root is on sys.path when executing as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.different_ability_two_players import (
    DIFFERENT_ABILITY_CONFIG,
    PARAM_SETS,
    Q_VALUES,
    EFFORT_RANGES,
    build_different_ability_config,
    build_param_grid_configs,
    calculate_theoretical_efforts_different_ability,
)
from envs.different_ability_env import DifferentAbilityEnv
from agents.different_ability_solver import different_ability_gradient_descent_solver
from agents.ppo_two_players_clean import PPOTwoPlayersBandit, PPOConfig


TRACES_DIR = os.path.join("results", "traces", "different_ability")
PLOTS_DIR = os.path.join("results", "plots", "different_ability")


def _ensure_outdir_and_logging(args: argparse.Namespace) -> str:
    """Ensure outdir exists, configure logging, and tee stdout to a log file."""
    if not getattr(args, "outdir", None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.outdir = os.path.join("results", "runs", f"different_ability_{timestamp}")
    os.makedirs(args.outdir, exist_ok=True)

    log_path = os.path.join(args.outdir, "run.log")

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    import builtins as _builtins

    def tee_print(*args: Any, **kwargs: Any) -> None:
        sep = kwargs.pop("sep", " ")
        end = kwargs.pop("end", "\n")
        kwargs.pop("file", None)
        kwargs.pop("flush", None)
        message = sep.join(str(x) for x in args)
        suffix = "" if end == "\n" else end
        logging.info("%s%s", message, suffix)

    _builtins.print = tee_print

    logging.info("Logging initialized. Writing to %s", log_path)
    return log_path


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_theory_fields(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Inject theoretical values into the config for env/analysis."""
    required = ["q", "k1", "k2", "l1", "l2", "w_h", "w_l"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    enriched = dict(cfg)
    e1, e2, c1, c2, eu1, eu2 = calculate_theoretical_efforts_different_ability(
        float(enriched["q"]),
        float(enriched["k1"]),
        float(enriched["k2"]),
        float(enriched["l1"]),
        float(enriched["l2"]),
        float(enriched["w_h"]),
        float(enriched["w_l"]),
    )
    enriched.update(
        {
            "theoretical_effort1": e1,
            "theoretical_effort2": e2,
            "theoretical_cost1": c1,
            "theoretical_cost2": c2,
            "theoretical_eu1": eu1,
            "theoretical_eu2": eu2,
            "theoretical_efforts": [e1, e2],
            "theoretical_costs": [c1, c2],
        }
    )
    if "effort_range" in enriched:
        low, high = enriched["effort_range"]
        enriched["effort_range"] = [float(low), float(high)]
    return enriched


def float_slug(value: float, precision: str = ".4f") -> str:
    return format(float(value), precision).replace(".", "p").replace("-", "m")


def effort_tag(bounds: List[float]) -> str:
    low, high = bounds
    return f"{int(low)}-{int(high)}"


def experiment_tag(cfg: Dict[str, Any]) -> str:
    return (
        f"q{int(round(float(cfg['q'])))}_"
        f"k{float_slug(cfg['k1'])}_"
        f"wh{float_slug(cfg['w_h'], '.1f')}_"
        f"wl{float_slug(cfg['w_l'], '.1f')}_"
        f"l{int(cfg['l1'])}-{int(cfg['l2'])}_"
        f"range{effort_tag(cfg['effort_range'])}"
    )


def save_gap_records(records: List[Dict[str, Any]], method: str, tag: str) -> str:
    """Persist per-update/step gap traces for later plotting."""
    if not records:
        return ""
    ensure_dir(TRACES_DIR)
    df = pd.DataFrame(records)
    path = os.path.join(TRACES_DIR, f"{method}_{tag}_gap.csv")
    df.to_csv(path, index=False)
    return path


def plot_gap_curve(records: List[Dict[str, Any]], method: str, tag: str) -> str:
    """Plot gap vs update curve for a given set of records."""
    if not records:
        return ""
    ensure_dir(PLOTS_DIR)
    df = pd.DataFrame(records)
    plt.figure(figsize=(8, 5))
    plt.plot(df["update"], df["gap"], marker="o", linewidth=1.5, label=f"{method} gap")
    plt.xlabel("update")
    plt.ylabel("|effort - e*| (max gap)")
    plt.title(f"Gap convergence ({method}, {tag})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"{method}_{tag}_gap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_parameter_sensitivity(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Aggregate final gaps across the parameter grid into a sensitivity plot."""
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if df.empty:
        return None

    # Compute max gap for safety if not present
    if "max_gap" not in df.columns:
        df["max_gap"] = (
            (df["final_e1"] - df["theoretical_e1"]).abs().clip(lower=0).combine_first(
                (df["final_e2"] - df["theoretical_e2"]).abs()
            )
        )

    df["effort_high"] = df["effort_range_high"]
    df["scenario"] = df.apply(
        lambda r: f"k={r['k1']:.4f}, w_h={r['w_h']:.1f}, range={int(r['effort_high'])}",
        axis=1,
    )

    ensure_dir(PLOTS_DIR)
    out_path = os.path.join(PLOTS_DIR, "parameter_sensitivity.png")

    plt.figure(figsize=(9, 5))
    for method, group in df.groupby("method"):
        pivot = (
            group.groupby(["q", "scenario"])["max_gap"]
            .mean()
            .reset_index()
            .sort_values("q")
        )
        scenarios = pivot["scenario"].unique()
        for scenario in scenarios:
            subset = pivot[pivot["scenario"] == scenario]
            plt.plot(
                subset["q"],
                subset["max_gap"],
                marker="o",
                label=f"{method} | {scenario}",
            )
    plt.xlabel("q")
    plt.ylabel("mean max gap")
    plt.title("Different-Ability sensitivity: final gap vs q")
    plt.grid(alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def norm01(x: float, denom: float, clip: float = 1e-9) -> float:
    d = max(clip, float(denom))
    return float(x) / d


def run_gradient(cfg: Dict[str, Any], store_history: bool = True) -> Dict[str, Any]:
    cfg = ensure_theory_fields(cfg)
    env = DifferentAbilityEnv(cfg)
    e1_th = cfg["theoretical_effort1"]
    e2_th = cfg["theoretical_effort2"]

    efforts, utils, costs, info = different_ability_gradient_descent_solver(
        env, lr=0.1, steps=100000, eps=1e-3, adaptive_lr=True, convergence_threshold=1e-4, patience=1000, verbose=False
    )
    gap = max(abs(efforts[0] - e1_th), abs(efforts[1] - e2_th))
    quality = (
        "Excellent" if gap < 0.5 else
        "Good" if gap < 1.0 else
        "Fair" if gap < 5.0 else
        "Poor"
    )
    records: List[Dict[str, Any]] = []
    trace_path = ""
    plot_path = ""
    if store_history:
        for item in info.get("convergence_history", []):
            if item.get("gap") is None:
                continue
            records.append(
                {
                    "update": int(item["step"]),
                    "step": int(item["step"]),
                    "gap": float(item["gap"]),
                    "e1": float(item["efforts"][0]),
                    "e2": float(item["efforts"][1]),
                }
            )
        if records:
            tag = experiment_tag(cfg)
            trace_path = save_gap_records(records, "gradient", tag)
            plot_path = plot_gap_curve(records, "gradient", tag)

    return {
        "method": "gradient",
        "q": cfg["q"],
        "l1": cfg["l1"],
        "l2": cfg["l2"],
        "k1": cfg["k1"],
        "k2": cfg["k2"],
        "w_h": cfg["w_h"],
        "w_l": cfg["w_l"],
        "theoretical_e1": e1_th,
        "theoretical_e2": e2_th,
        "final_e1": efforts[0],
        "final_e2": efforts[1],
        "episodes": info.get("final_step", 0),
        "updates": info.get("final_step", 0),
        "max_gap": gap,
        "quality": quality,
        "effort_range_low": cfg["effort_range"][0],
        "effort_range_high": cfg["effort_range"][1],
        "gap_trace_path": trace_path,
        "gap_plot_path": plot_path,
    }


def run_ppo(
    cfg: Dict[str, Any],
    *,
    episodes: int,
    updates: Optional[int] = None,
    steps_per_update: Optional[int] = None,
    epochs: Optional[int] = None,
    minibatch_size: Optional[int] = None,
    log_interval: int = 1,
    store_history: bool = True,
    metrics_outdir: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = ensure_theory_fields(cfg)
    env = DifferentAbilityEnv(cfg)
    e1_th = cfg["theoretical_effort1"]
    e2_th = cfg["theoretical_effort2"]

    bounds = tuple(cfg["effort_range"])
    low, high = bounds

    # Configure PPO with ability-aware state features
    ppo_cfg = PPOConfig(
        steps_per_update=steps_per_update or 16384,
        epochs=epochs or 20,
        minibatch_size=minibatch_size or 1024,
        state_dim=5,  # [q_norm, k_norm, wgap_norm, l_self_norm, l_other_norm]
        hidden=64,
        opponent_sync_interval=1,
        opponent_ema_tau=0.0,
        entropy_coef=0.02,
        lr=3e-4,
        clip_eps=0.25,
    )
    agent = PPOTwoPlayersBandit(effort_bounds=bounds, cfg=ppo_cfg)
    agent.cfg.entropy_coef = float(cfg.get("entropy_coef_start", agent.cfg.entropy_coef))
    agent.cfg.clip_eps = float(cfg.get("clip_range_start", agent.cfg.clip_eps))
    for group in agent.opt.param_groups:
        group["lr"] = float(cfg.get("lr_start", ppo_cfg.lr))

    def state_for(l_self: float, l_other: float) -> torch.Tensor:
        q_norm = norm01(cfg["q"], 60.0)
        k_norm = norm01(cfg["k1"], 1e-3)
        wgap_norm = norm01(cfg["w_h"] - cfg["w_l"], 10.0)
        denom = max(cfg["l1"], cfg["l2"], 1.0)
        l_self_n = norm01(l_self, denom)
        l_other_n = norm01(l_other, denom)
        s = torch.tensor([q_norm, k_norm, wgap_norm, l_self_n, l_other_n], dtype=torch.float32)
        return s.unsqueeze(0).to(agent.device)

    total_steps_target = int(episodes)
    if updates is not None:
        total_steps_target = int(updates) * ppo_cfg.steps_per_update

    steps_done = 0
    update_idx = 0
    total_updates = (total_steps_target + ppo_cfg.steps_per_update - 1) // ppo_cfg.steps_per_update

    capture_history = store_history or metrics_outdir is not None
    entropy_start = float(cfg.get("entropy_coef_start", agent.cfg.entropy_coef))
    entropy_hold = float(cfg.get("entropy_coef_hold", entropy_start))
    entropy_final = float(cfg.get("entropy_coef_end", 0.002))
    hold_fraction = float(cfg.get("entropy_hold_fraction", 2.0 / 3.0))
    hold_fraction = max(0.0, min(1.0, hold_fraction))
    hold_updates = max(1, int(math.ceil(total_updates * hold_fraction)))
    tail_updates = max(1, total_updates - hold_updates)
    lr_start_val = float(cfg.get("lr_start", ppo_cfg.lr))
    lr_final_val = float(cfg.get("lr_final", lr_start_val))
    lr_min = float(cfg.get("lr_min", 5e-5))
    lr_max = float(cfg.get("lr_max", 5e-4))
    clip_start = float(cfg.get("clip_range_start", agent.cfg.clip_eps))
    clip_end = float(cfg.get("clip_range_end", 0.15))
    clip_floor = float(cfg.get("clip_range_floor", 0.05))
    clip_ceiling = float(cfg.get("clip_range_ceiling", 0.5))
    target_kl = float(cfg.get("target_kl", 0.01))
    kl_low = 0.5 * target_kl
    kl_high = 3.0 * target_kl
    clip_factor = 1.0
    lr_factor = 1.0
    rng = np.random.default_rng(cfg.get("seed", 42))
    records: List[Dict[str, Any]] = []

    last_update_metrics: Optional[Dict[str, Any]] = None

    while steps_done < total_steps_target:
        if update_idx < hold_updates:
            if hold_updates > 1:
                hold_progress = float(update_idx) / float(hold_updates - 1)
            else:
                hold_progress = 1.0
            hold_progress = max(0.0, min(1.0, hold_progress))
            entropy_val = entropy_start + (entropy_hold - entropy_start) * hold_progress
            clip_base = clip_start
            lr_base = lr_start_val
        else:
            if tail_updates > 1:
                tail_progress = float(update_idx - hold_updates) / float(tail_updates - 1)
            else:
                tail_progress = 1.0
            tail_progress = max(0.0, min(1.0, tail_progress))
            entropy_val = entropy_hold + (entropy_final - entropy_hold) * tail_progress
            clip_base = clip_start + (clip_end - clip_start) * tail_progress
            lr_base = lr_start_val + (lr_final_val - lr_start_val) * tail_progress

        agent.cfg.entropy_coef = entropy_val
        clip_val = clip_base * clip_factor
        clip_val = max(clip_floor, min(clip_ceiling, clip_val))
        agent.cfg.clip_eps = clip_val

        current_lr = lr_base * lr_factor
        current_lr = max(lr_min, min(lr_max, current_lr))
        for group in agent.opt.param_groups:
            group["lr"] = current_lr

        steps_this = min(ppo_cfg.steps_per_update, total_steps_target - steps_done)
        for _ in range(steps_this):
            s1 = state_for(cfg["l1"], cfg["l2"])
            s2 = state_for(cfg["l2"], cfg["l1"])

            a1_norm, e1, logp1, v1 = agent.act(s1)
            a2_norm, e2, logp2, v2 = agent.act_opponent(s2)
            if v2 is None:
                v2 = agent.value_only(s2)

            _, rewards, _, done, _ = env.step(
                [
                    torch.tensor([float(e1.item())]),
                    torch.tensor([float(e2.item())]),
                ]
            )

            # Store both players from the start for stable self-play
            agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
            agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))

        last_update_metrics = agent.update()
        approx_kl = float(last_update_metrics.get("approx_kl", 0.0)) if isinstance(last_update_metrics, dict) else 0.0
        if approx_kl < kl_low:
            clip_factor = min(clip_factor * 1.2, 1.5)
            lr_factor = min(lr_factor * 1.25, 1.5)
        elif approx_kl > kl_high:
            clip_factor = max(clip_factor * 0.8, 0.5)
            lr_factor = max(lr_factor * 0.8, 0.2)

        update_idx += 1
        steps_done += steps_this

        with torch.no_grad():
            se1 = state_for(cfg["l1"], cfg["l2"])
            se2 = state_for(cfg["l2"], cfg["l1"])
            dist1, _ = agent.net.dist(se1)
            dist2, _ = agent.net.dist(se2)
            a1 = dist1.mean.squeeze().clamp(0.0, 1.0)
            a2 = dist2.mean.squeeze().clamp(0.0, 1.0)
            e1_eval = float(low + a1.item() * (high - low))
            e2_eval = float(low + a2.item() * (high - low))
            gap1 = abs(e1_eval - e1_th)
            gap2 = abs(e2_eval - e2_th)
            gap = max(gap1, gap2)

        if log_interval and (update_idx % log_interval == 0 or update_idx == 1):
            print(
                f"[Update {update_idx}/{total_updates}] q={cfg['q']}: "
                f"e*={(e1_th, e2_th)} policy=({e1_eval:.2f}, {e2_eval:.2f}) "
                f"gap={gap:.3f} entropy={agent.cfg.entropy_coef:.3f} "
                f"clip={agent.cfg.clip_eps:.3f} lr={current_lr:.2e} "
                f"kl={approx_kl:.4f}"
            )

        if capture_history:
            records.append(
                {
                    "update": update_idx,
                    "steps": steps_done,
                    "gap": gap,
                    "gap1": gap1,
                    "gap2": gap2,
                    "e1": e1_eval,
                    "e2": e2_eval,
                    "entropy": agent.cfg.entropy_coef,
                    "clip_eps": agent.cfg.clip_eps,
                    "lr": current_lr,
                    "approx_kl": approx_kl,
                }
            )

    with torch.no_grad():
        se1 = state_for(cfg["l1"], cfg["l2"])
        se2 = state_for(cfg["l2"], cfg["l1"])
        d1, _ = agent.net.dist(se1)
        d2, _ = agent.net.dist(se2)
        a1 = d1.mean.squeeze().clamp(0.0, 1.0)
        a2 = d2.mean.squeeze().clamp(0.0, 1.0)
        e1_final = float(low + a1.item() * (high - low))
        e2_final = float(low + a2.item() * (high - low))

    gap = max(abs(e1_final - e1_th), abs(e2_final - e2_th))
    quality = (
        "Excellent" if gap < 0.5 else
        "Good" if gap < 1.0 else
        "Fair" if gap < 5.0 else
        "Poor"
    )

    trace_path = ""
    plot_path = ""
    if store_history and records:
        tag = experiment_tag(cfg)
        trace_path = save_gap_records(records, "ppo", tag)
        plot_path = plot_gap_curve(records, "ppo", tag)
    if metrics_outdir and records:
        os.makedirs(metrics_outdir, exist_ok=True)
        df_metrics = pd.DataFrame(records)
        if "update" not in df_metrics.columns:
            df_metrics.insert(0, "update", np.arange(1, len(df_metrics) + 1))
        metrics_path = os.path.join(metrics_outdir, "metrics.csv")
        df_metrics.to_csv(metrics_path, index=False)

    return {
        "method": "ppo",
        "q": cfg["q"],
        "l1": cfg["l1"],
        "l2": cfg["l2"],
        "k1": cfg["k1"],
        "k2": cfg["k2"],
        "w_h": cfg["w_h"],
        "w_l": cfg["w_l"],
        "theoretical_e1": e1_th,
        "theoretical_e2": e2_th,
        "final_e1": e1_final,
        "final_e2": e2_final,
        "episodes": total_steps_target,
        "updates": update_idx,
        "max_gap": gap,
        "quality": quality,
        "effort_range_low": cfg["effort_range"][0],
        "effort_range_high": cfg["effort_range"][1],
        "gap_trace_path": trace_path,
        "gap_plot_path": plot_path,
        "steps_per_update": ppo_cfg.steps_per_update,
        "ppo_epochs": ppo_cfg.epochs,
        "minibatch_size": ppo_cfg.minibatch_size,
    }


def save_results(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("No rows to save")
    os.makedirs("results/tables", exist_ok=True)
    path = "results/tables/different_ability_two_players.csv"
    new_df = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        merged = pd.concat([old, new_df], ignore_index=True, sort=False)
    else:
        merged = new_df
    merged.to_csv(path, index=False)
    print(f"Saved results to {path}")
    return path


def parse_effort_range(values: Optional[List[float]]) -> Optional[List[float]]:
    if not values:
        return None
    if len(values) != 2:
        raise ValueError("Effort range requires two values: low high")
    low, high = float(values[0]), float(values[1])
    if low >= high:
        raise ValueError("Effort range low must be < high")
    return [low, high]


def main():
    parser = argparse.ArgumentParser(description="Different-Ability Two-Player Experiment")
    parser.add_argument("--method", choices=["gradient", "ppo", "both"], default="both")
    parser.add_argument("--grid", action="store_true", help="Run full parameter grid (k, w_h, q, effort ranges)")
    parser.add_argument("--q", type=float, help="Override q for single run (ignored with --grid)")
    parser.add_argument("--k", type=float, help="Override k (k1=k2) for single run")
    parser.add_argument("--w-h", dest="w_h", type=float, help="Override w_h for single run")
    parser.add_argument("--w-l", dest="w_l", type=float, help="Override w_l for single run")
    parser.add_argument("--effort-range", nargs=2, type=float, metavar=("LOW", "HIGH"), help="Effort bounds for single run")
    parser.add_argument("--episodes", type=int, default=100000, help="Total PPO environment steps (overridden by --updates)")
    parser.add_argument("--updates", type=int, help="Number of PPO updates (episodes = updates * steps_per_update)")
    parser.add_argument("--steps-per-update", type=int, help="Override PPO steps_per_update")
    parser.add_argument("--epochs", dest="ppo_epochs", type=int, help="Override PPO epochs per update")
    parser.add_argument("--minibatch-size", type=int, help="Override PPO minibatch size")
    parser.add_argument("--log-interval", type=int, default=1, help="Log PPO progress every N updates")
    parser.add_argument("--seed", type=int, help="Random seed override")
    parser.add_argument("--skip-history", action="store_true", help="Skip saving per-update gap traces")
    parser.add_argument("--make-plots", action="store_true", help="Force regeneration of summary plots from current run")
    parser.add_argument("--lr-start", type=float, default=0.0003, help="Initial PPO learning rate")
    parser.add_argument("--lr-final", type=float, default=0.0001, help="Final PPO learning rate after annealing")
    parser.add_argument("--target-kl", type=float, default=0.015, help="Target KL divergence for adaptive scaling")
    parser.add_argument("--entropy-hold-fraction", type=float, default=0.85, help="Fraction of updates to hold entropy before annealing")
    parser.add_argument("--clip-range-end", type=float, default=0.2, help="Final PPO clipping range")
    parser.add_argument("--outdir", type=str, help="Directory to store per-update metrics (metrics.csv)")
    args = parser.parse_args()

    _ensure_outdir_and_logging(args)

    params_path = os.path.join(args.outdir, "params.json")
    with open(params_path, "w", encoding="utf-8") as params_file:
        json.dump(vars(args), params_file, indent=2, sort_keys=True)
    print("Saved params.json to", params_path)

    command_path = os.path.join(args.outdir, "command.txt")
    with open(command_path, "w", encoding="utf-8") as command_file:
        command_file.write(" ".join(sys.argv) + "\n")
    print("Saved command.txt to", command_path)

    if args.grid:
        configs = build_param_grid_configs()
    else:
        base = dict(DIFFERENT_ABILITY_CONFIG)
        if args.q is not None:
            base["q"] = float(args.q)
        if args.k is not None:
            k_val = float(args.k)
            base["k"] = k_val
            base["k1"] = k_val
            base["k2"] = k_val
        if args.w_h is not None:
            base["w_h"] = float(args.w_h)
        if args.w_l is not None:
            base["w_l"] = float(args.w_l)
        if args.effort_range:
            base["effort_range"] = parse_effort_range(list(args.effort_range))
        if "effort_range" not in base:
            base["effort_range"] = list(DIFFERENT_ABILITY_CONFIG.get("effort_range", [0, 100]))
        if args.lr_start is not None:
            base["lr_start"] = float(args.lr_start)
        if args.lr_final is not None:
            base["lr_final"] = float(args.lr_final)
        if args.target_kl is not None:
            base["target_kl"] = float(args.target_kl)
        if args.entropy_hold_fraction is not None:
            base["entropy_hold_fraction"] = float(args.entropy_hold_fraction)
        if args.clip_range_end is not None:
            base["clip_range_end"] = float(args.clip_range_end)
        configs = [base]

    if args.lr_start is not None:
        for cfg in configs:
            cfg["lr_start"] = float(args.lr_start)
    if args.lr_final is not None:
        for cfg in configs:
            cfg["lr_final"] = float(args.lr_final)
    if args.target_kl is not None:
        for cfg in configs:
            cfg["target_kl"] = float(args.target_kl)
    if args.entropy_hold_fraction is not None:
        for cfg in configs:
            cfg["entropy_hold_fraction"] = float(args.entropy_hold_fraction)
    if args.clip_range_end is not None:
        for cfg in configs:
            cfg["clip_range_end"] = float(args.clip_range_end)

    # Apply optional seed override
    if args.seed is not None:
        for cfg in configs:
            cfg["seed"] = int(args.seed)

    rows: List[Dict[str, Any]] = []
    for cfg_raw in configs:
        cfg_for_logging = ensure_theory_fields(cfg_raw)
        tag = experiment_tag(cfg_for_logging)
        print("=" * 80)
        print(f"Running config: {tag}")
        print(
            f"Parameters -> k={cfg_for_logging['k1']}, w_h={cfg_for_logging['w_h']}, "
            f"w_l={cfg_for_logging['w_l']}, q={cfg_for_logging['q']}, "
            f"range={tuple(cfg_for_logging['effort_range'])}"
        )

        if args.method in ("gradient", "both"):
            grad_row = run_gradient(dict(cfg_for_logging), store_history=not args.skip_history)
            rows.append(grad_row)
            print(f"Gradient solver max_gap={grad_row['max_gap']:.4f}, quality={grad_row['quality']}")

        if args.method in ("ppo", "both"):
            metrics_dir = None
            if args.outdir:
                metrics_dir = os.path.join(args.outdir, tag) if len(configs) > 1 else args.outdir
            ppo_row = run_ppo(
                dict(cfg_for_logging),
                episodes=args.episodes,
                updates=args.updates,
                steps_per_update=args.steps_per_update,
                epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                log_interval=max(1, args.log_interval),
                store_history=not args.skip_history,
                metrics_outdir=metrics_dir,
            )
            rows.append(ppo_row)
            print(f"PPO max_gap={ppo_row['max_gap']:.4f}, quality={ppo_row['quality']}")

    if not rows:
        print("No experiments were executed (check CLI arguments).")
        return

    csv_path = save_results(rows)
    if args.make_plots or args.grid:
        sensitivity_path = plot_parameter_sensitivity(rows)
        if sensitivity_path:
            print(f"Saved parameter sensitivity plot to {sensitivity_path}")
    print(f"Run complete. Results appended to {csv_path}")


if __name__ == "__main__":
    main()
