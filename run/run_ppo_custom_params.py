#!/usr/bin/env python3
"""
Run PPO experiments with custom parameters for convergence plot generation.

IMPORTANT: Each q value is trained SEPARATELY with its own network instance.
This ensures each network can fully specialize on its specific q value,
rather than learning an average across all q values.

Parameters:
- k = 0.0005 (cost parameter)
- w_h = 8 (high prize)
- w_l = 3 (low prize)
- q values: 25, 40, 55 (trained separately in sequence)
- seed: 42

Theoretical equilibrium efforts (e* = (w_h - w_l) / (4 * q * k)):
- q=25: e* = 100.0
- q=40: e* = 62.5
- q=55: e* ≈ 45.45

Output: Convergence JSON files with ablation_name "k5e4_wh8_wl3"
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.one_stage_two_players import config as base_config
from run.run_two_players import run_ppo


def main():
    """Run PPO experiments with custom parameters - each q value trained separately."""
    
    # Custom parameters
    CUSTOM_K = 0.0005
    CUSTOM_W_H = 8.0
    CUSTOM_W_L = 3.0
    CUSTOM_SEED = 42
    Q_VALUES = [25.0, 40.0, 55.0]
    ABLATION_NAME = "k5e4_wh8_wl3"
    
    # Calculate theoretical efforts for reference
    print("=" * 60)
    print("PPO Experiments with Custom Parameters")
    print("(Each q value trained SEPARATELY)")
    print("=" * 60)
    print(f"k = {CUSTOM_K}")
    print(f"w_h = {CUSTOM_W_H}")
    print(f"w_l = {CUSTOM_W_L}")
    print(f"seed = {CUSTOM_SEED}")
    print(f"q values = {Q_VALUES}")
    print(f"ablation_name = {ABLATION_NAME}")
    print()
    print("Theoretical equilibrium efforts:")
    for q in Q_VALUES:
        e_star = (CUSTOM_W_H - CUSTOM_W_L) / (4.0 * q * CUSTOM_K)
        print(f"  q={q}: e* = {e_star:.2f}")
    print("=" * 60)
    
    all_results = []
    
    # Train each q value SEPARATELY with its own network
    for i, q_val in enumerate(Q_VALUES):
        print(f"\n{'#' * 60}")
        print(f"# Training q={q_val} ({i+1}/{len(Q_VALUES)})")
        print(f"# Theoretical e* = {(CUSTOM_W_H - CUSTOM_W_L) / (4.0 * q_val * CUSTOM_K):.2f}")
        print(f"{'#' * 60}\n")
        
        # Create fresh config for this q value
        cfg = base_config.copy()
        cfg["k"] = CUSTOM_K
        cfg["k1"] = CUSTOM_K
        cfg["k2"] = CUSTOM_K
        cfg["w_h"] = CUSTOM_W_H
        cfg["w_l"] = CUSTOM_W_L
        cfg["seed"] = CUSTOM_SEED
        cfg["q"] = q_val  # Set the specific q value
        cfg["q_list"] = [q_val]  # Only this q value
        
        # Update derived values for this specific q
        cfg["stage1_weight"] = CUSTOM_W_L
        cfg["stage2_weight"] = CUSTOM_W_H
        cfg["effort"] = (CUSTOM_W_H - CUSTOM_W_L) / (4 * CUSTOM_K * q_val)
        cfg["cost"] = CUSTOM_K * cfg["effort"] ** 2
        cfg["eu"] = round(((CUSTOM_W_H + CUSTOM_W_L) / 2 - CUSTOM_K * cfg["effort"] ** 2), 2)
        
        # Enable PPO modern defaults
        cfg["theory_align_v2"] = True
        
        # Enable convergence evaluation with relaxed profile
        if "convergence" not in cfg:
            cfg["convergence"] = {}
        cfg["convergence"]["enabled"] = True
        cfg["convergence"]["cheap_gate_profile"] = "relaxed"
        
        # Run PPO for this single q value (fresh network each time)
        results = run_ppo(
            cfg=cfg,
            episodes=cfg.get("episodes", 2_048_000),
            train_qs=[q_val],  # Train only on this q
            eval_qs=[q_val],   # Eval only on this q
            rollout_mode="selfplay",
            eval_symmetric=True,
            ablation_name=ABLATION_NAME,
        )
        
        all_results.extend(results)
        
        # Print intermediate result
        if results:
            result = results[0]
            theoretical = result.get("theoretical_effort", 0)
            final_effort = result.get("final_effort", 0)
            gap = result.get("Gap_from_theoretical", 0)
            quality = result.get("Convergence_Quality", "?")
            print(f"\n>>> q={q_val} completed: theoretical={theoretical:.2f}, "
                  f"final={final_effort:.2f}, gap={gap:.2f}, quality={quality}")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY (All q values)")
    print("=" * 60)
    
    for result in all_results:
        q = result.get("q", "?")
        theoretical = result.get("theoretical_effort", 0)
        final_effort = result.get("final_effort", 0)
        gap = result.get("Gap_from_theoretical", 0)
        quality = result.get("Convergence_Quality", "?")
        
        print(f"q={q}: theoretical={theoretical:.2f}, final={final_effort:.2f}, "
              f"gap={gap:.2f}, quality={quality}")
    
    print("=" * 60)
    print("\nConvergence files saved to: results/convergence_history/")
    print(f"Files: ppo_q{{q}}_seed{CUSTOM_SEED}_{ABLATION_NAME}_convergence.json")
    print("\nDone!")
    
    return all_results


if __name__ == "__main__":
    main()
