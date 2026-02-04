#!/usr/bin/env python3
"""
Quick experiment launcher with common presets.

Usage:
    # Fast debug run
    python quick_experiment.py --preset debug --q 40

    # Production run
    python quick_experiment.py --preset production --q 40 --seed 42

    # Custom parameters
    python quick_experiment.py --k 0.0005 --w-h 8.0 --q 25 --episodes 1000000
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from config.one_stage_two_players import config as base_config
from run.run_two_players import run_ppo, run_gradient


# Preset configurations
PRESETS = {
    "debug": {
        "episodes": 204_800,
        "steps_per_update": 2048,
        "eval_every_updates": 5,
        "convergence": {"enabled": False},
    },
    "fast": {
        "episodes": 512_000,
        "steps_per_update": 4096,
        "eval_every_updates": 10,
        "convergence": {"enabled": True, "cheap_gate_profile": "aggressive"},
    },
    "production": {
        "episodes": 2_048_000,
        "steps_per_update": 4096,
        "eval_every_updates": 20,
        "convergence": {"enabled": True, "cheap_gate_profile": "relaxed"},
    },
    "extended": {
        "episodes": 4_096_000,
        "steps_per_update": 4096,
        "eval_every_updates": 20,
        "convergence": {"enabled": True, "cheap_gate_profile": "conservative"},
    },
}


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Quick experiment launcher with presets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Preset selection
    parser.add_argument(
        "--preset", choices=list(PRESETS.keys()), default="production",
        help="Configuration preset"
    )
    
    # Method selection
    parser.add_argument(
        "--method", choices=["ppo", "gradient"], default="ppo",
        help="Training method"
    )
    
    # Core parameters (override preset)
    parser.add_argument("--q", type=float, help="Noise parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, help="Total steps (overrides preset)")
    
    # Game parameters
    parser.add_argument("--k", type=float, help="Cost coefficient")
    parser.add_argument("--w-h", type=float, help="High prize")
    parser.add_argument("--w-l", type=float, help="Low prize")
    
    # Experiment naming
    parser.add_argument("--ablation-name", type=str, help="Ablation identifier")
    
    # Rollout mode
    parser.add_argument(
        "--rollout-mode", choices=["selfplay", "vs_opponent"], default="selfplay",
        help="PPO rollout mode"
    )
    
    return parser


def compute_theoretical_effort(k: float, w_h: float, w_l: float, q: float) -> float:
    """Compute theoretical equilibrium effort."""
    return (w_h - w_l) / (4.0 * k * q)


def run_experiment(args: argparse.Namespace) -> list:
    """Run experiment with given configuration."""
    # Start with base config
    cfg = base_config.copy()
    
    # Apply preset
    preset = PRESETS[args.preset]
    for key, value in preset.items():
        if isinstance(value, dict) and key in cfg and isinstance(cfg[key], dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    
    # Apply CLI overrides
    if args.k is not None:
        cfg["k"] = args.k
        cfg["k1"] = args.k
        cfg["k2"] = args.k
    if args.w_h is not None:
        cfg["w_h"] = args.w_h
        cfg["stage2_weight"] = args.w_h
    if args.w_l is not None:
        cfg["w_l"] = args.w_l
        cfg["stage1_weight"] = args.w_l
    if args.episodes is not None:
        cfg["episodes"] = args.episodes
    
    cfg["seed"] = args.seed
    
    # Determine q values
    if args.q is not None:
        q_values = [args.q]
        cfg["q"] = args.q
        cfg["q_list"] = [args.q]
    else:
        q_values = cfg.get("q_list", [40.0])
    
    # Print experiment summary
    print("=" * 60)
    print(f"Experiment: {args.method.upper()} with {args.preset} preset")
    print("=" * 60)
    print(f"k={cfg['k']}, w_h={cfg['w_h']}, w_l={cfg['w_l']}")
    print(f"q values: {q_values}")
    print(f"episodes: {cfg['episodes']:,}")
    print(f"seed: {args.seed}")
    if args.ablation_name:
        print(f"ablation: {args.ablation_name}")
    print()
    
    # Print theoretical efforts
    print("Theoretical equilibrium efforts:")
    for q in q_values:
        e_star = compute_theoretical_effort(cfg["k"], cfg["w_h"], cfg["w_l"], q)
        print(f"  q={q}: e* = {e_star:.2f}")
    print("=" * 60)
    
    # Run experiment
    if args.method == "ppo":
        # Enable theory align v2 by default
        cfg["theory_align_v2"] = True
        
        results = run_ppo(
            cfg=cfg,
            episodes=cfg["episodes"],
            train_qs=q_values,
            eval_qs=q_values,
            rollout_mode=args.rollout_mode,
            eval_symmetric=True,
            ablation_name=args.ablation_name,
        )
    else:
        results = run_gradient(
            cfg=cfg,
            q_values=q_values,
        )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    for result in results:
        q = result.get("q", "?")
        theoretical = result.get("theoretical_effort", 0)
        final = result.get("final_effort", 0)
        gap = result.get("Gap_from_theoretical", abs(theoretical - final))
        print(f"q={q}: e*={theoretical:.2f}, final={final:.2f}, gap={gap:.2f}")
    
    return results


def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
