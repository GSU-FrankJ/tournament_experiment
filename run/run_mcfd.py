#!/usr/bin/env python3
"""
Standalone MC-FD runner that uses only `agents.mc_fd_crn_solver` defaults.

This script intentionally bypasses the TaskMaster config layer and exposes
command-line flags for every solver hyperparameter so experiments can be
launched directly without touching `config/` files.
"""

from __future__ import annotations

import sys
import os
import argparse
import math
from typing import Dict

# Add project root to Python path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.mc_fd_crn_solver import MCFDConfig, gradient_ascent_dynamics


def build_parser() -> argparse.ArgumentParser:
    """
    Build a CLI parser that mirrors the dataclass defaults.

    Keeping the defaults synchronized with `MCFDConfig` guarantees that
    invoking the script with zero flags behaves exactly like importing the
    solver module directly.
    """

    parser = argparse.ArgumentParser(description="Standalone MC-FD solver runner")
    parser.add_argument("--w-h", type=float, default=MCFDConfig.w_h, help="High prize w_h")
    parser.add_argument("--w-l", type=float, default=MCFDConfig.w_l, help="Low prize w_l")
    parser.add_argument("--k", type=float, default=MCFDConfig.k, help="Quadratic cost coefficient")
    parser.add_argument("--sigma1", type=float, default=MCFDConfig.sigma1, help="Noise std for player 1")
    parser.add_argument("--sigma2", type=float, default=MCFDConfig.sigma2, help="Noise std for player 2")
    parser.add_argument("--delta", type=float, default=MCFDConfig.delta, help="Finite-difference step")
    parser.add_argument("--eta", type=float, default=MCFDConfig.eta, help="Gradient-ascent learning rate")
    parser.add_argument("--num-samples", type=int, default=MCFDConfig.num_samples, help="Monte Carlo batch size")
    parser.add_argument("--e-min", type=float, default=MCFDConfig.e_min, help="Effort lower bound")
    parser.add_argument("--e-max", type=float, default=MCFDConfig.e_max, help="Effort upper bound")
    parser.add_argument("--max-iters", type=int, default=MCFDConfig.max_iters, help="Max gradient-ascent iterations")
    parser.add_argument("--tol", type=float, default=MCFDConfig.tol, help="Convergence tolerance on effort deltas")
    parser.add_argument("--seed", type=int, default=MCFDConfig.seed, help="Seed for RNG (None disables seeding)")
    return parser


def theoretical_effort(params: Dict[str, float]) -> float:
    """
    Compute the analytic benchmark e* = (w_H - w_L) / (4 k sqrt(pi) sigma).

    Even though the solver never plugs this formula into the dynamics, we
    report it so users can gauge how far the Monte Carlo optimum deviates.
    """

    numerator = params["w_h"] - params["w_l"]
    denominator = 4.0 * params["k"] * math.sqrt(math.pi) * params["sigma1"]
    if denominator == 0.0:
        raise ValueError("Denominator for theoretical effort is zero; check k and sigma1.")
    return numerator / denominator


def run_mcfd(args: argparse.Namespace) -> None:
    """
    Construct the dataclass, run gradient ascent, and print diagnostics.

    All console outputs are kept minimal and numeric so that downstream shell
    scripts can parse them without extra processing.
    """

    cfg = MCFDConfig(
        w_h=args.w_h,
        w_l=args.w_l,
        k=args.k,
        sigma1=args.sigma1,
        sigma2=args.sigma2,
        delta=args.delta,
        eta=args.eta,
        num_samples=args.num_samples,
        e_min=args.e_min,
        e_max=args.e_max,
        max_iters=args.max_iters,
        tol=args.tol,
        seed=args.seed,
    )

    theory = theoretical_effort(
        {"w_h": cfg.w_h, "w_l": cfg.w_l, "k": cfg.k, "sigma1": cfg.sigma1}
    )

    results = gradient_ascent_dynamics(cfg)
    e1 = results["effort_player1"][-1]
    e2 = results["effort_player2"][-1]
    avg_effort = 0.5 * (e1 + e2)
    iterations = len(results["effort_player1"]) - 1

    print("=== MC-FD Standalone Run ===")
    print(f"w_h={cfg.w_h:.4f}, w_l={cfg.w_l:.4f}, k={cfg.k:.6f}")
    print(f"sigma1={cfg.sigma1:.4f}, sigma2={cfg.sigma2:.4f}")
    print(f"delta={cfg.delta:.4f}, eta={cfg.eta:.4f}, num_samples={cfg.num_samples}")
    print(f"e_bounds=[{cfg.e_min:.2f}, {cfg.e_max:.2f}], max_iters={cfg.max_iters}, tol={cfg.tol}")
    print(f"theoretical_e*={theory:.6f}")
    print(f"final_e1={e1:.6f}, final_e2={e2:.6f}, avg={avg_effort:.6f}")
    print(f"iterations={iterations}")


def main() -> None:
    """Parse CLI flags and run the solver."""

    parser = build_parser()
    args = parser.parse_args()
    run_mcfd(args)


if __name__ == "__main__":
    main()







