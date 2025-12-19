#!/usr/bin/env python3
"""
Multi-Q Gradient Parameter Sweep

Runs parameter sweep across multiple q values to find robust parameter sets
that work well across different noise levels.

Usage:
    python run/run_gradient_sweep_multi_q.py --strategy random --n-trials 100
"""

import sys
import os
import argparse
import json
from datetime import datetime
from typing import Dict, List
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run.run_gradient_sweep import (
    get_parameter_bounds,
    get_parameter_space,
    random_search,
    grid_search,
    evaluate_parameters,
    save_sweep_results,
)
from config.one_stage_two_players import config as base_config


def evaluate_multi_q(cfg: Dict, params: Dict, q_values: List[float], 
                    verbose: bool = False) -> Dict:
    """
    Evaluate parameters across multiple q values and aggregate results.
    
    Args:
        cfg: Base configuration
        params: Parameter combination
        q_values: List of q values to test
        verbose: Verbose output
    
    Returns:
        Aggregated results dictionary
    """
    q_results = []
    total_gap = 0.0
    total_iterations = 0.0
    all_qualities = []
    worst_gap = 0.0
    worst_q = None
    
    for q in q_values:
        result = evaluate_parameters(cfg, params, q, verbose=False)
        if result["status"] == "success":
            q_results.append(result)
            total_gap += result["gap"]
            total_iterations += result["iterations"]
            all_qualities.append(result["quality"])
            if result["gap"] > worst_gap:
                worst_gap = result["gap"]
                worst_q = q
        else:
            # Failed on this q value
            return {
                **params,
                "status": "error",
                "error": f"Failed on q={q}",
                "robustness_score": float('inf'),
            }
    
    # Calculate robustness metrics
    # Note: result["gap"] is now max_gap (max deviation from theoretical)
    mean_gap = total_gap / len(q_values)
    mean_iterations = total_iterations / len(q_values)
    
    # Calculate mean symmetry gap across all q values
    total_symmetry_gap = sum(r.get("symmetry_gap", 0) for r in q_results)
    mean_symmetry_gap = total_symmetry_gap / len(q_values)
    
    # Robustness score: penalize worst-case performance
    # worst_gap is now the worst max_gap across all q values
    robustness_score = worst_gap + 0.5 * mean_gap + 0.1 * mean_symmetry_gap + 0.01 * (mean_iterations / params["steps"])
    
    # Quality distribution
    quality_counts = {}
    for q in all_qualities:
        quality_counts[q] = quality_counts.get(q, 0) + 1
    
    # Collect detailed per-q results
    per_q_details = {}
    for q, r in zip(q_values, q_results):
        per_q_details[f"q_{q}"] = {
            "e1": r.get("e1", 0),
            "e2": r.get("e2", 0),
            "theoretical": r.get("theoretical_effort", 0),
            "max_gap": r.get("gap", 0),
            "gap_e1": r.get("gap_e1", 0),
            "gap_e2": r.get("gap_e2", 0),
            "symmetry_gap": r.get("symmetry_gap", 0),
        }
    
    return {
        **params,
        "status": "success",
        "q_values": q_values,
        "mean_gap": mean_gap,
        "worst_gap": worst_gap,
        "worst_q": worst_q,
        "mean_symmetry_gap": mean_symmetry_gap,
        "mean_iterations": mean_iterations,
        "robustness_score": robustness_score,
        "quality_distribution": quality_counts,
        "all_gaps": [r["gap"] for r in q_results],  # These are now max_gaps
        "per_q_details": per_q_details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Q Gradient Parameter Sweep for Robustness"
    )
    parser.add_argument(
        "--strategy",
        choices=["grid", "random", "bayesian"],
        default="random",
        help="Search strategy (default: random)"
    )
    parser.add_argument(
        "--q-values",
        type=float,
        nargs="+",
        default=[25.0, 40.0, 55.0],
        help="Q values to test (default: 25.0 40.0 55.0)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials for random/bayesian search (default: 100)"
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        help="Maximum combinations for grid search (default: all, no limit)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sweeps",
        help="Output directory (default: results/sweeps)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    cfg = dict(base_config)
    
    print("="*60)
    print(f"🔍 Multi-Q Gradient Parameter Sweep")
    print(f"   Strategy: {args.strategy}")
    print(f"   Q values: {args.q_values}")
    if args.strategy == "grid":
        print(f"   Combinations: all (no limit)" if args.max_combinations is None else f"   Max combinations: {args.max_combinations}")
    else:
        print(f"   Trials: {args.n_trials}")
    print("="*60)
    
    # Generate parameter combinations
    if args.strategy == "grid":
        space = get_parameter_space()
        param_combinations = grid_search(space, max_combinations=args.max_combinations)
        total_combinations = len(space["lr"]) * len(space["steps"]) * len(space["grad_eps"]) * \
                           len(space["tol"]) * len(space["num_samples"]) * len(space["init_perturb"])
        if args.max_combinations is None:
            print(f"📊 Grid search: {len(param_combinations)} combinations (all)")
            total_evals = len(param_combinations) * len(args.q_values)
            print(f"⚠️  Total evaluations: {len(param_combinations)} × {len(args.q_values)} = {total_evals}")
            print(f"⚠️  This may take a very long time. Consider using --max-combinations to limit.")
        else:
            print(f"📊 Grid search: {len(param_combinations)} combinations (limited from {total_combinations})")
        
        # Evaluate across all q values
        print(f"\n🚀 Running {len(param_combinations)} evaluations across {len(args.q_values)} q values...")
        results = []
        
        for i, params in enumerate(param_combinations, 1):
            if i % 100 == 0 or i == len(param_combinations):
                print(f"[{i}/{len(param_combinations)}] Testing parameters across q={args.q_values}...")
            result = evaluate_multi_q(cfg, params, args.q_values, verbose=False)
            results.append(result)
            
            if args.verbose and result["status"] == "success":
                print(f"    ✅ Mean gap: {result['mean_gap']:.6f}, Worst gap: {result['worst_gap']:.6f}")
    
    elif args.strategy == "random":
        bounds = get_parameter_bounds()
        param_combinations = random_search(bounds, args.n_trials, seed=args.seed)
        
        # Evaluate across all q values
        print(f"\n🚀 Running {len(param_combinations)} evaluations across {len(args.q_values)} q values...")
        results = []
        
        for i, params in enumerate(param_combinations, 1):
            print(f"[{i}/{len(param_combinations)}] Testing parameters across q={args.q_values}...")
            result = evaluate_multi_q(cfg, params, args.q_values, args.verbose)
            results.append(result)
            
            if args.verbose and result["status"] == "success":
                print(f"    ✅ Mean gap: {result['mean_gap']:.6f}, Worst gap: {result['worst_gap']:.6f}")
    
    elif args.strategy == "bayesian":
        # Bayesian optimization for multi-q
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "Optuna is required for Bayesian optimization. "
                "Install with: pip install optuna"
            )
        
        def objective(trial):
            # Suggest parameters
            params = {
                "lr": trial.suggest_float("lr", bounds["lr"][0], bounds["lr"][1], log=False),
                "steps": trial.suggest_int("steps", int(bounds["steps"][0]), int(bounds["steps"][1])),
                "grad_eps": trial.suggest_float("grad_eps", bounds["grad_eps"][0], bounds["grad_eps"][1]),
                "tol": trial.suggest_float("tol", bounds["tol"][0], bounds["tol"][1], log=True),
                "num_samples": trial.suggest_int("num_samples", int(bounds["num_samples"][0]), 
                                                int(bounds["num_samples"][1]), log=True),
                "init_perturb": trial.suggest_float("init_perturb", bounds["init_perturb"][0], 
                                                    bounds["init_perturb"][1]),
            }
            
            # Evaluate across all q values
            result = evaluate_multi_q(cfg, params, args.q_values, verbose=False)
            
            if result["status"] == "error":
                # Return a large penalty for failed evaluations
                return float('inf')
            
            # Minimize robustness_score
            return result["robustness_score"]
        
        print(f"\n🧠 Running Bayesian optimization with {args.n_trials} trials...")
        study = optuna.create_study(
            direction="minimize", 
            sampler=optuna.samplers.TPESampler(seed=args.seed)
        )
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
        
        # Get top N trials and evaluate them fully
        top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))[:min(20, args.n_trials)]
        results = []
        
        print(f"\n📊 Evaluating top {len(top_trials)} parameter combinations...")
        for i, trial in enumerate(top_trials, 1):
            if trial.value is None or trial.value == float('inf'):
                continue
            params = trial.params
            print(f"[{i}/{len(top_trials)}] Evaluating parameters...")
            result = evaluate_multi_q(cfg, params, args.q_values, args.verbose)
            results.append(result)
            
            if args.verbose and result["status"] == "success":
                print(f"    ✅ Robustness score: {result['robustness_score']:.6f}")
    
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    import csv
    csv_path = os.path.join(args.output_dir, f"sweep_multiq_{args.strategy}_{timestamp}.csv")
    successful_results = [r for r in results if r.get("status") == "success"]
    
    if successful_results:
        # Flatten results for CSV
        csv_rows = []
        for r in successful_results:
            row = {k: v for k, v in r.items() if k not in ["q_values", "quality_distribution", "all_gaps"]}
            row["q_values"] = ",".join(map(str, r["q_values"]))
            row["quality_dist"] = str(r["quality_distribution"])
            csv_rows.append(row)
        
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"✅ Saved results to {csv_path}")
        
        # Find best robust parameters
        best_result = min(successful_results, key=lambda x: x["robustness_score"])
        
        print("\n" + "="*60)
        print("🏆 BEST ROBUST PARAMETER COMBINATION:")
        print("="*60)
        for key in ["lr", "steps", "grad_eps", "tol", "num_samples", "init_perturb"]:
            print(f"  {key:15s}: {best_result[key]}")
        print(f"\n  Mean Max Gap: {best_result['mean_gap']:.6f}")
        print(f"  Worst Max Gap: {best_result['worst_gap']:.6f} (at q={best_result['worst_q']})")
        print(f"  Mean Symmetry Gap: {best_result.get('mean_symmetry_gap', 0):.6f}")
        print(f"  Robustness Score: {best_result['robustness_score']:.6f}")
        print(f"  Quality Distribution: {best_result['quality_distribution']}")
        print("\n  Per-Q Details:")
        per_q = best_result.get("per_q_details", {})
        for q_key, details in per_q.items():
            q_val = q_key.replace("q_", "")
            print(f"    q={q_val}:")
            print(f"      e1={details['e1']:.6f} (gap: {details['gap_e1']:.6f})")
            print(f"      e2={details['e2']:.6f} (gap: {details['gap_e2']:.6f})")
            print(f"      theoretical={details['theoretical']:.6f}")
            print(f"      max_gap={details['max_gap']:.6f}, symmetry={details['symmetry_gap']:.6f}")
        print("="*60)
        
        # Save summary
        summary = {
            "strategy": args.strategy,
            "q_values": args.q_values,
            "total_trials": len(results),
            "successful_trials": len(successful_results),
            "best_parameters": {k: best_result[k] for k in ["lr", "steps", "grad_eps", "tol", 
                                                           "num_samples", "init_perturb"]},
            "best_metrics": {
                "mean_gap": best_result["mean_gap"],
                "worst_gap": best_result["worst_gap"],
                "worst_q": best_result["worst_q"],
                "robustness_score": best_result["robustness_score"],
                "quality_distribution": best_result["quality_distribution"],
            },
            "top_5": sorted(successful_results, key=lambda x: x["robustness_score"])[:5],
        }
        
        json_path = os.path.join(args.output_dir, f"sweep_multiq_{args.strategy}_{timestamp}_summary.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Saved summary to {json_path}")


if __name__ == "__main__":
    main()

