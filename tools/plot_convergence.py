#!/usr/bin/env python3
"""
Plot convergence curves for PPO and Gradient algorithms.

Shows how agent efforts converge from initial values to theoretical optimal values
across different noise levels (q values).

Supports:
- Symmetric two-player tournaments (single theoretical line)
- Asymmetric two-player tournaments (two theoretical lines for different_cost)
- Three-player tournaments (single theoretical line for symmetric)
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


def load_convergence_data(
    convergence_dir: str = "results/two_players/convergence",
    scenario_filter: Optional[str] = None,
) -> Dict:
    """
    Load all convergence history JSON files from the directory.
    
    Args:
        convergence_dir: Directory containing convergence JSON files
        scenario_filter: Optional filter for scenario type (e.g., "different_cost")
    
    Returns:
        Dictionary organized by algorithm and q value:
        {
            "ppo": {
                25.0: {...data...},
                40.0: {...data...},
                55.0: {...data...}
            },
            "gradient": {
                25.0: {...data...},
                40.0: {...data...},
                55.0: {...data...}
            }
        }
    """
    data = {"ppo": {}, "gradient": {}}
    
    if not os.path.exists(convergence_dir):
        print(f"❌ Convergence history directory not found: {convergence_dir}")
        return data
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(convergence_dir, "*_convergence.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                file_data = json.load(f)
            
            algorithm = file_data.get("algorithm", "unknown").lower()
            q_value = file_data.get("q", 0.0)
            scenario = file_data.get("scenario", "symmetric")
            
            # Apply scenario filter if specified
            if scenario_filter is not None and scenario != scenario_filter:
                continue
            
            if algorithm in data:
                data[algorithm][q_value] = file_data
                scenario_tag = f" [{scenario}]" if scenario != "symmetric" else ""
                print(f"✅ Loaded {algorithm} q={q_value}{scenario_tag} from {os.path.basename(json_file)}")
        except Exception as e:
            print(f"❌ Error loading {json_file}: {e}")
    
    return data


def _is_different_cost_scenario(file_data: Dict) -> bool:
    """Check if data is from a different_cost (asymmetric) scenario."""
    return file_data.get("scenario") == "different_cost"


def _get_theoretical_efforts(file_data: Dict) -> tuple:
    """
    Extract theoretical efforts from convergence data.
    
    Returns:
        (e1_star, e2_star) for asymmetric scenarios, or (e_star, e_star) for symmetric
    """
    # Check for asymmetric (different_cost) scenario
    if _is_different_cost_scenario(file_data):
        theoretical = file_data.get("theoretical", {})
        if theoretical:
            return theoretical.get("effort1"), theoretical.get("effort2")
    
    # Symmetric scenario - single theoretical value
    e_star = file_data.get("theoretical_effort")
    return e_star, e_star


def _get_effort_history(file_data: Dict) -> tuple:
    """
    Extract effort history arrays from convergence data.
    
    Handles both direct arrays and nested "history" structure.
    
    Returns:
        (steps, e1_history, e2_history)
    """
    # Check for nested "history" structure (used by different_cost PPO)
    history = file_data.get("history", {})
    if history:
        steps = history.get("steps", file_data.get("steps", []))
        e1 = history.get("agent1_effort", file_data.get("agent1_effort", []))
        e2 = history.get("agent2_effort", file_data.get("agent2_effort", []))
    else:
        steps = file_data.get("steps", [])
        e1 = file_data.get("agent1_effort", [])
        e2 = file_data.get("agent2_effort", [])
    
    return steps, e1, e2


def plot_convergence_figure(
    data: Dict,
    output_file: str = "results/convergence_comparison.png",
    title_prefix: str = "",
):
    """
    Create convergence figure with panels for different q values.
    
    Supports both symmetric and asymmetric (different_cost) scenarios:
    - Symmetric: single theoretical line
    - Asymmetric: two theoretical lines (e1*, e2*) with different colors
    
    Figure structure:
    - x-axis: training steps
    - y-axis: effort (two agents)
    - lines: theory (horizontal), gradient agent1/agent2, PPO agent1/agent2
    - panels: different noise levels (q values)
    """
    # Get unique q values across all algorithms
    all_q_values = set()
    for alg_data in data.values():
        all_q_values.update(alg_data.keys())
    q_values_sorted = sorted(list(all_q_values))
    
    if not q_values_sorted:
        print("❌ No data to plot!")
        return
    
    # Detect if this is an asymmetric scenario
    is_asymmetric = False
    for alg_data in data.values():
        for file_data in alg_data.values():
            if _is_different_cost_scenario(file_data):
                is_asymmetric = True
                break
        if is_asymmetric:
            break
    
    # Create figure with subplots (one per q value)
    n_panels = len(q_values_sorted)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    
    # Handle single panel case
    if n_panels == 1:
        axes = [axes]
    
    # Color scheme
    colors = {
        "theory1": "#000000",       # Black for e1* (player 1 theory)
        "theory2": "#666666",       # Gray for e2* (player 2 theory)
        "gradient_p1": "#1f77b4",   # Blue
        "gradient_p2": "#ff7f0e",   # Orange
        "ppo_p1": "#2ca02c",        # Green
        "ppo_p2": "#d62728"         # Red
    }
    
    for idx, q in enumerate(q_values_sorted):
        ax = axes[idx]
        
        # Set panel title
        title = f"q = {q:.1f}"
        if title_prefix:
            title = f"{title_prefix} | {title}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Training Steps", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Effort", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Track theoretical values
        e1_star, e2_star = None, None
        
        # Plot gradient algorithm
        if q in data.get("gradient", {}):
            grad_data = data["gradient"][q]
            e1_star, e2_star = _get_theoretical_efforts(grad_data)
            
            steps, e1, e2 = _get_effort_history(grad_data)
            
            ax.plot(steps, e1, label="Gradient Agent1", 
                   color=colors["gradient_p1"], linewidth=2, linestyle='-')
            ax.plot(steps, e2, label="Gradient Agent2",
                   color=colors["gradient_p2"], linewidth=2, linestyle='-')
        
        # Plot PPO algorithm
        if q in data.get("ppo", {}):
            ppo_data = data["ppo"][q]
            if e1_star is None:
                e1_star, e2_star = _get_theoretical_efforts(ppo_data)
            
            steps, e1, e2 = _get_effort_history(ppo_data)
            
            # Subsample PPO data for better visualization if too many points
            if len(steps) > 1000:
                subsample = len(steps) // 1000
                steps = steps[::subsample]
                e1 = e1[::subsample]
                e2 = e2[::subsample]
            
            ax.plot(steps, e1, label="PPO Agent1",
                   color=colors["ppo_p1"], linewidth=2, linestyle='--', alpha=0.8)
            ax.plot(steps, e2, label="PPO Agent2",
                   color=colors["ppo_p2"], linewidth=2, linestyle='--', alpha=0.8)
        
        # Plot theoretical values
        if e1_star is not None:
            if is_asymmetric and e2_star is not None and abs(e1_star - e2_star) > 0.01:
                # Asymmetric case: two distinct theoretical lines
                ax.axhline(y=e1_star, color=colors["theory1"],
                          linewidth=2.5, linestyle=':', 
                          label=f"Theory e1*={e1_star:.2f}")
                ax.axhline(y=e2_star, color=colors["theory2"],
                          linewidth=2.5, linestyle='--',
                          label=f"Theory e2*={e2_star:.2f}")
            else:
                # Symmetric case: single theoretical line
                ax.axhline(y=e1_star, color=colors["theory1"],
                          linewidth=2.5, linestyle=':', 
                          label=f"Theory (e*={e1_star:.2f})")
        
        # Add legend to first panel
        if idx == 0:
            ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Convergence figure saved to {output_file}")
    plt.close()


def plot_convergence_separate_agents(
    data: Dict,
    output_file: str = "results/convergence_separate_agents.png",
    title_prefix: str = "",
):
    """
    Create convergence figure showing agent1 and agent2 in separate panels.
    
    Supports both symmetric and asymmetric scenarios:
    - Symmetric: same theoretical line for both agents
    - Asymmetric: different theoretical lines (e1*, e2*) for each agent
    
    Figure structure:
    - Top row: Agent 1 convergence for all q values
    - Bottom row: Agent 2 convergence for all q values
    """
    # Get unique q values
    all_q_values = set()
    for alg_data in data.values():
        all_q_values.update(alg_data.keys())
    q_values_sorted = sorted(list(all_q_values))
    
    if not q_values_sorted:
        print("❌ No data to plot!")
        return
    
    # Detect if this is an asymmetric scenario
    is_asymmetric = False
    for alg_data in data.values():
        for file_data in alg_data.values():
            if _is_different_cost_scenario(file_data):
                is_asymmetric = True
                break
        if is_asymmetric:
            break
    
    n_panels = len(q_values_sorted)
    fig, axes = plt.subplots(2, n_panels, figsize=(6 * n_panels, 10))
    
    # Handle single panel case
    if n_panels == 1:
        axes = axes.reshape(2, 1)
    
    colors = {
        "theory": "black",
        "gradient": "#1f77b4",
        "ppo": "#2ca02c"
    }
    
    for idx, q in enumerate(q_values_sorted):
        # Agent 1 panel (top row)
        ax1 = axes[0, idx]
        title1 = f"Agent 1, q={q:.1f}"
        if title_prefix:
            title1 = f"{title_prefix} | {title1}"
        ax1.set_title(title1, fontsize=14, fontweight='bold')
        ax1.set_xlabel("Training Steps", fontsize=12)
        if idx == 0:
            ax1.set_ylabel("Agent 1 Effort", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Agent 2 panel (bottom row)
        ax2 = axes[1, idx]
        title2 = f"Agent 2, q={q:.1f}"
        if title_prefix:
            title2 = f"{title_prefix} | {title2}"
        ax2.set_title(title2, fontsize=14, fontweight='bold')
        ax2.set_xlabel("Training Steps", fontsize=12)
        if idx == 0:
            ax2.set_ylabel("Agent 2 Effort", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        e1_star, e2_star = None, None
        
        # Plot gradient
        if q in data.get("gradient", {}):
            grad_data = data["gradient"][q]
            e1_star, e2_star = _get_theoretical_efforts(grad_data)
            
            steps, e1, e2 = _get_effort_history(grad_data)
            
            ax1.plot(steps, e1, label="Gradient", color=colors["gradient"], linewidth=2)
            ax2.plot(steps, e2, label="Gradient", color=colors["gradient"], linewidth=2)
        
        # Plot PPO
        if q in data.get("ppo", {}):
            ppo_data = data["ppo"][q]
            if e1_star is None:
                e1_star, e2_star = _get_theoretical_efforts(ppo_data)
            
            steps, e1, e2 = _get_effort_history(ppo_data)
            
            # Subsample if needed
            if len(steps) > 1000:
                subsample = len(steps) // 1000
                steps = steps[::subsample]
                e1 = e1[::subsample]
                e2 = e2[::subsample]
            
            ax1.plot(steps, e1, label="PPO", color=colors["ppo"], linewidth=2, linestyle='--')
            ax2.plot(steps, e2, label="PPO", color=colors["ppo"], linewidth=2, linestyle='--')
        
        # Add theoretical lines (different for asymmetric case)
        if e1_star is not None:
            ax1.axhline(y=e1_star, color=colors["theory"],
                       linewidth=2.5, linestyle=':', label=f"Theory e1*={e1_star:.2f}")
        if e2_star is not None:
            ax2.axhline(y=e2_star, color=colors["theory"],
                       linewidth=2.5, linestyle=':', label=f"Theory e2*={e2_star:.2f}")
        
        if idx == 0:
            ax1.legend(loc='best', fontsize=10)
            ax2.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Separate agents figure saved to {output_file}")
    plt.close()


def plot_different_cost_convergence(
    convergence_dir: str = "results/two_players/convergence",
    output_dir: str = "results",
):
    """
    Plot convergence specifically for different_cost (asymmetric) experiments.
    
    Creates dedicated plots with:
    - Two theoretical reference lines (e1*, e2*) with clear labeling
    - k1, k2 values shown in title
    """
    data = load_convergence_data(convergence_dir, scenario_filter="different_cost")
    
    if not any(data.values()):
        print("❌ No different_cost data found to plot!")
        return
    
    # Get k1, k2 values from any loaded data
    k1, k2 = None, None
    for alg_data in data.values():
        for file_data in alg_data.values():
            k1 = file_data.get("k1", k1)
            k2 = file_data.get("k2", k2)
            break
        if k1 is not None:
            break
    
    title_prefix = f"Different Cost (k1={k1}, k2={k2})" if k1 and k2 else "Different Cost"
    
    # Generate plots
    plot_convergence_figure(
        data,
        output_file=os.path.join(output_dir, "different_cost_convergence.png"),
        title_prefix=title_prefix,
    )
    plot_convergence_separate_agents(
        data,
        output_file=os.path.join(output_dir, "different_cost_separate_agents.png"),
        title_prefix=title_prefix,
    )
    
    print(f"✅ Different cost plots generated in {output_dir}/")


def main():
    """Main execution function."""
    print("="*60)
    print("📊 Convergence Plotting Tool")
    print("="*60)
    
    # Load all data
    data = load_convergence_data()
    
    # Print summary
    print("\n📋 Data Summary:")
    for algorithm, alg_data in data.items():
        if alg_data:
            q_vals = sorted(alg_data.keys())
            # Check for scenarios
            scenarios = set()
            for file_data in alg_data.values():
                scenarios.add(file_data.get("scenario", "symmetric"))
            scenario_str = f" ({', '.join(sorted(scenarios))})" if scenarios else ""
            print(f"  {algorithm}: {len(q_vals)} q values - {q_vals}{scenario_str}")
        else:
            print(f"  {algorithm}: No data found")
    
    # Check for different_cost data and generate dedicated plots
    diff_cost_data = load_convergence_data(scenario_filter="different_cost")
    if any(diff_cost_data.values()):
        print("\n🎨 Generating different_cost specific plots...")
        plot_different_cost_convergence()
        print("\n✅ Different cost plots generated!")
    
    print("\n✅ All plotting complete!")


if __name__ == "__main__":
    main()
