#!/usr/bin/env python3
"""
Gradient Parameter Sweep for One-Stage Two-Player Experiments

Supports multiple search strategies:
- Grid Search: Exhaustive search over parameter grid
- Random Search: Random sampling from parameter space
- Bayesian Optimization: Adaptive search (requires optuna)

Usage:
    # Grid search
    python run/run_gradient_sweep.py --strategy grid --q 40.0
    
    # Random search (100 trials)
    python run/run_gradient_sweep.py --strategy random --n-trials 100 --q 40.0
    
    # Bayesian optimization (requires optuna)
    python run/run_gradient_sweep.py --strategy bayesian --n-trials 50 --q 40.0
"""

import sys
import os
import argparse
import json
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from itertools import product
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.one_stage_two_players import config as base_config
from run.run_two_players import run_gradient
from utils.theory import e_star_two_players, clip_stage2


# ============================================================================
# Parameter Space Definition
# ============================================================================

def get_parameter_space() -> Dict[str, List]:
    """
    Define the search space for gradient parameters.
    
    Returns:
        Dictionary mapping parameter names to lists of candidate values
    """
    return {
        "lr": [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20],  # Learning rate
        "steps": [500, 1000, 1500, 2000, 3000],  # Max iterations
        "grad_eps": [0.1, 0.2, 0.3, 0.5, 0.8, 1.0],  # Finite difference epsilon
        "tol": [1e-5, 1e-4, 1e-3],  # Convergence tolerance
        "num_samples": [16, 32, 64, 128, 256],  # Monte Carlo samples
        "init_perturb": [0.5, 1.0, 2.0, 3.0],  # Initial perturbation
    }


def get_parameter_bounds() -> Dict[str, Tuple[float, float]]:
    """
    Define continuous bounds for random/bayesian search.
    
    Returns:
        Dictionary mapping parameter names to (min, max) tuples
    """
    return {
        "lr": (0.01, 0.20),
        "steps": (500, 3000),
        "grad_eps": (0.1, 1.0),
        "tol": (1e-5, 1e-3),
        "num_samples": (16, 256),
        "init_perturb": (0.5, 3.0),
    }


# ============================================================================
# Search Strategies
# ============================================================================

def grid_search(space: Dict[str, List], max_combinations: Optional[int] = None) -> List[Dict]:
    """
    Generate all parameter combinations for grid search.
    
    Args:
        space: Parameter space dictionary
        max_combinations: Maximum number of combinations (None = all)
    
    Returns:
        List of parameter dictionaries
    """
    keys = list(space.keys())
    values = list(space.values())
    combinations = list(product(*values))
    
    if max_combinations and len(combinations) > max_combinations:
        print(f"⚠️  Grid search has {len(combinations)} combinations, limiting to {max_combinations}")
        # Sample uniformly
        indices = np.linspace(0, len(combinations) - 1, max_combinations, dtype=int)
        combinations = [combinations[i] for i in indices]
    
    return [dict(zip(keys, combo)) for combo in combinations]


def random_search(bounds: Dict[str, Tuple[float, float]], n_trials: int, 
                  seed: Optional[int] = None) -> List[Dict]:
    """
    Generate random parameter combinations.
    
    Args:
        bounds: Parameter bounds dictionary
        n_trials: Number of random trials
        seed: Random seed for reproducibility
    
    Returns:
        List of parameter dictionaries
    """
    if seed is not None:
        np.random.seed(seed)
    
    trials = []
    for _ in range(n_trials):
        params = {}
        for key, (min_val, max_val) in bounds.items():
            if key in ["steps", "num_samples"]:
                # Integer parameters
                params[key] = int(np.random.uniform(min_val, max_val + 1))
            elif key == "tol":
                # Log-uniform for tolerance
                params[key] = 10 ** np.random.uniform(np.log10(min_val), np.log10(max_val))
            else:
                # Continuous parameters
                params[key] = np.random.uniform(min_val, max_val)
        trials.append(params)
    
    return trials


def evaluate_parameters(cfg: Dict, params: Dict, q_value: float, 
                       verbose: bool = False) -> Dict:
    """
    Evaluate a parameter combination and return metrics.
    
    Args:
        cfg: Base configuration dictionary
        params: Parameter combination to test
        q_value: Noise parameter q
        verbose: Whether to print progress
    
    Returns:
        Dictionary with parameters and evaluation metrics
    """
    # Update config with q value
    test_cfg = dict(cfg)
    test_cfg["q"] = q_value
    
    # Calculate theoretical effort for reference
    theoretical_e = clip_stage2(
        e_star_two_players(q_value, cfg["w_h"], cfg["w_l"], cfg["k"]),
        tuple(cfg["effort_bounds_stage2"])
    )
    
    try:
        # Run gradient descent with these parameters
        result = run_gradient(
            test_cfg,
            lr=params["lr"],
            steps=params["steps"],
            grad_eps=params["grad_eps"],
            tol=params["tol"],
            num_samples=int(params["num_samples"]),
            init_perturb=params["init_perturb"],
            log=verbose,
        )
        
        # Extract key metrics
        e1 = result.get("final_e1", 0.0)
        e2 = result.get("final_e2", 0.0)
        final_effort = result["final_stage2_effort"]  # Keep for backward compatibility
        symmetry_gap = result.get("symmetry_gap", abs(e1 - e2))
        iterations = result.get("gradient_iterations", params["steps"])
        final_grad = result.get("gradient_final_grad", float('inf'))
        
        # NEW EVALUATION CRITERIA: Both e1 and e2 should be close to theoretical
        # Calculate individual gaps
        gap_e1 = abs(e1 - theoretical_e)
        gap_e2 = abs(e2 - theoretical_e)
        
        # Primary metric: maximum deviation from theoretical (worst of the two)
        max_gap = max(gap_e1, gap_e2)
        
        # Secondary metric: average gap (for reference)
        avg_gap = 0.5 * (gap_e1 + gap_e2)
        
        # Quality assessment based on max_gap
        if max_gap < 0.5:
            quality = "Excellent"
        elif max_gap < 1.0:
            quality = "Good"
        elif max_gap < 5.0:
            quality = "Fair"
        else:
            quality = "Poor"
        
        # Efficiency score (lower is better): 
        # Combines max deviation, symmetry, and convergence efficiency
        # Weight: max_gap (primary) + symmetry penalty + convergence penalty
        symmetry_penalty = 0.1 * symmetry_gap  # Penalize asymmetry
        convergence_penalty = 0.01 * (iterations / params["steps"])  # Penalize incomplete convergence
        efficiency_score = max_gap + symmetry_penalty + convergence_penalty
        
        return {
            **params,
            "q": q_value,
            "theoretical_effort": theoretical_e,
            "final_effort": final_effort,  # Keep for backward compatibility
            "e1": e1,
            "e2": e2,
            "gap": max_gap,  # Now represents max deviation
            "gap_e1": gap_e1,
            "gap_e2": gap_e2,
            "avg_gap": avg_gap,  # Average gap for reference
            "symmetry_gap": symmetry_gap,
            "iterations": iterations,
            "final_grad": final_grad,
            "quality": quality,
            "efficiency_score": efficiency_score,
            "converged": final_grad < params["tol"] * 10,  # Reasonable convergence check
            "status": "success",
        }
    except Exception as e:
        if verbose:
            print(f"❌ Error with params {params}: {e}")
        return {
            **params,
            "q": q_value,
            "status": "error",
            "error": str(e),
            "gap": float('inf'),
            "max_gap": float('inf'),
            "efficiency_score": float('inf'),
        }


# ============================================================================
# Bayesian Optimization (Optional)
# ============================================================================

def bayesian_search(bounds: Dict[str, Tuple[float, float]], n_trials: int,
                    cfg: Dict, q_value: float, seed: Optional[int] = None) -> List[Dict]:
    """
    Bayesian optimization using Optuna (if available).
    
    Args:
        bounds: Parameter bounds
        n_trials: Number of optimization trials
        cfg: Base configuration
        q_value: Noise parameter q
        seed: Random seed
    
    Returns:
        List of best parameter combinations
    """
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
        
        result = evaluate_parameters(cfg, params, q_value, verbose=False)
        return result["efficiency_score"]  # Minimize this
    
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Return top N trials
    top_trials = sorted(study.trials, key=lambda t: t.value)[:min(10, n_trials)]
    results = []
    for trial in top_trials:
        params = trial.params
        result = evaluate_parameters(cfg, params, q_value, verbose=False)
        results.append(result)
    
    return results


# ============================================================================
# Results Management
# ============================================================================

def save_sweep_results(results: List[Dict], output_dir: str, strategy: str, q_value: float):
    """
    Save sweep results to CSV and JSON files.
    
    Args:
        results: List of evaluation results
        output_dir: Output directory path
        strategy: Search strategy name
        q_value: Noise parameter q
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed CSV
    csv_path = os.path.join(output_dir, f"sweep_{strategy}_q{q_value}_{timestamp}.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"✅ Saved detailed results to {csv_path}")
    
    # Save summary JSON
    successful_results = [r for r in results if r.get("status") == "success"]
    if successful_results:
        # Sort by efficiency_score (which now uses max_gap as primary metric)
        best_result = min(successful_results, key=lambda x: x["efficiency_score"])
        summary = {
            "strategy": strategy,
            "q_value": q_value,
            "total_trials": len(results),
            "successful_trials": len(successful_results),
            "best_parameters": {k: best_result[k] for k in ["lr", "steps", "grad_eps", "tol", 
                                                           "num_samples", "init_perturb"]},
            "best_metrics": {
                "max_gap": best_result["gap"],  # gap now represents max deviation
                "gap_e1": best_result.get("gap_e1", 0),
                "gap_e2": best_result.get("gap_e2", 0),
                "avg_gap": best_result.get("avg_gap", 0),
                "e1": best_result.get("e1", 0),
                "e2": best_result.get("e2", 0),
                "theoretical_effort": best_result.get("theoretical_effort", 0),
                "symmetry_gap": best_result.get("symmetry_gap", 0),
                "quality": best_result["quality"],
                "iterations": best_result["iterations"],
                "efficiency_score": best_result["efficiency_score"],
            },
            "top_5": sorted(successful_results, key=lambda x: x["efficiency_score"])[:5],
        }
        
        json_path = os.path.join(output_dir, f"sweep_{strategy}_q{q_value}_{timestamp}_summary.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Saved summary to {json_path}")
        
        # Print best result
        print("\n" + "="*60)
        print("🏆 BEST PARAMETER COMBINATION:")
        print("="*60)
        for key, value in summary["best_parameters"].items():
            print(f"  {key:15s}: {value}")
        metrics = summary["best_metrics"]
        print(f"\n  Theoretical Effort: {metrics.get('theoretical_effort', 0):.6f}")
        print(f"  e1: {metrics.get('e1', 0):.6f} (gap: {metrics.get('gap_e1', 0):.6f})")
        print(f"  e2: {metrics.get('e2', 0):.6f} (gap: {metrics.get('gap_e2', 0):.6f})")
        print(f"  Max Gap: {metrics['max_gap']:.6f} (worst deviation)")
        print(f"  Avg Gap: {metrics.get('avg_gap', 0):.6f}")
        print(f"  Symmetry Gap: {metrics.get('symmetry_gap', 0):.6f}")
        print(f"  Quality: {metrics['quality']}")
        print(f"  Iterations: {metrics['iterations']:.0f}")
        print(f"  Efficiency Score: {metrics['efficiency_score']:.6f}")
        print("="*60)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gradient Parameter Sweep for One-Stage Two-Player Experiments"
    )
    parser.add_argument(
        "--strategy",
        choices=["grid", "random", "bayesian"],
        default="random",
        help="Search strategy (default: random)"
    )
    parser.add_argument(
        "--q",
        type=float,
        default=40.0,
        help="Noise parameter q (default: 40.0)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials for random/bayesian search (default: 50)"
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        help="Maximum combinations for grid search (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sweeps",
        help="Output directory for results (default: results/sweeps)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress for each trial"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)"
    )
    
    args = parser.parse_args()
    
    # Prepare configuration
    cfg = dict(base_config)
    
    print("="*60)
    print(f"🔍 Gradient Parameter Sweep")
    print(f"   Strategy: {args.strategy}")
    print(f"   Q value: {args.q}")
    print(f"   Trials: {args.n_trials if args.strategy != 'grid' else 'all'}")
    print("="*60)
    
    # Generate parameter combinations
    if args.strategy == "grid":
        space = get_parameter_space()
        param_combinations = grid_search(space, max_combinations=args.max_combinations)
        print(f"📊 Grid search: {len(param_combinations)} combinations")
    elif args.strategy == "random":
        bounds = get_parameter_bounds()
        param_combinations = random_search(bounds, args.n_trials, seed=args.seed)
        print(f"🎲 Random search: {args.n_trials} trials")
    elif args.strategy == "bayesian":
        bounds = get_parameter_bounds()
        print(f"🧠 Bayesian optimization: {args.n_trials} trials")
        results = bayesian_search(bounds, args.n_trials, cfg, args.q, seed=args.seed)
        save_sweep_results(results, args.output_dir, args.strategy, args.q)
        return
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    
    # Evaluate all combinations
    print(f"\n🚀 Running {len(param_combinations)} evaluations...")
    results = []
    
    if args.parallel > 1:
        # Parallel execution (requires joblib or multiprocessing)
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=args.parallel, verbose=10)(
                delayed(evaluate_parameters)(cfg, params, args.q, args.verbose)
                for params in param_combinations
            )
        except ImportError:
            print("⚠️  joblib not available, falling back to sequential execution")
            for i, params in enumerate(param_combinations, 1):
                print(f"[{i}/{len(param_combinations)}] Testing parameters...")
                result = evaluate_parameters(cfg, params, args.q, args.verbose)
                results.append(result)
    else:
        # Sequential execution
        for i, params in enumerate(param_combinations, 1):
            print(f"[{i}/{len(param_combinations)}] Testing parameters...")
            result = evaluate_parameters(cfg, params, args.q, args.verbose)
            results.append(result)
            if args.verbose:
                if result["status"] == "success":
                    print(f"    ✅ Gap: {result['gap']:.6f}, Quality: {result['quality']}")
                else:
                    print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Save results
    save_sweep_results(results, args.output_dir, args.strategy, args.q)
    
    print(f"\n✅ Sweep complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

