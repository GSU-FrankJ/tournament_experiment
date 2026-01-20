#!/usr/bin/env python3
"""
Plot convergence curves for PPO and Gradient algorithms.

Shows how agent efforts converge from initial values to theoretical optimal values
across different noise levels (q values).
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def load_convergence_data(convergence_dir: str = "results/convergence_history") -> Dict:
    """
    Load all convergence history JSON files from the directory.
    
    Returns:
        Dictionary organized by algorithm and q value:
        {
            "PPO": {
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
    data = {"PPO": {}, "gradient": {}}
    
    if not os.path.exists(convergence_dir):
        print(f"❌ Convergence history directory not found: {convergence_dir}")
        return data
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(convergence_dir, "*_convergence.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                file_data = json.load(f)
            
            algorithm = file_data.get("algorithm", "unknown")
            q_value = file_data.get("q", 0.0)
            
            if algorithm in data:
                data[algorithm][q_value] = file_data
                print(f"✅ Loaded {algorithm} q={q_value} from {os.path.basename(json_file)}")
        except Exception as e:
            print(f"❌ Error loading {json_file}: {e}")
    
    return data


def plot_convergence_figure(data: Dict, output_file: str = "results/convergence_comparison.png"):
    """
    Create convergence figure with panels for different q values.
    
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
    
    # Create figure with subplots (one per q value)
    n_panels = len(q_values_sorted)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    
    # Handle single panel case
    if n_panels == 1:
        axes = [axes]
    
    # Color scheme
    colors = {
        "theory": "black",
        "gradient_p1": "#1f77b4",
        "gradient_p2": "#ff7f0e", 
        "ppo_p1": "#2ca02c",
        "ppo_p2": "#d62728"
    }
    
    for idx, q in enumerate(q_values_sorted):
        ax = axes[idx]
        
        # Set panel title
        ax.set_title(f"q = {q:.1f}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Training Steps", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Effort", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Plot theoretical value (horizontal line)
        theoretical_effort = None
        
        # Plot gradient algorithm
        if q in data.get("gradient", {}):
            grad_data = data["gradient"][q]
            theoretical_effort = grad_data.get("theoretical_effort")
            
            steps = grad_data.get("steps", [])
            e1 = grad_data.get("agent1_effort", [])
            e2 = grad_data.get("agent2_effort", [])
            
            ax.plot(steps, e1, label="Gradient Agent1", 
                   color=colors["gradient_p1"], linewidth=2, linestyle='-')
            ax.plot(steps, e2, label="Gradient Agent2",
                   color=colors["gradient_p2"], linewidth=2, linestyle='-')
        
        # Plot PPO algorithm
        if q in data.get("PPO", {}):
            ppo_data = data["PPO"][q]
            if theoretical_effort is None:
                theoretical_effort = ppo_data.get("theoretical_effort")
            
            steps = ppo_data.get("steps", [])
            e1 = ppo_data.get("agent1_effort", [])
            e2 = ppo_data.get("agent2_effort", [])
            
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
        
        # Plot theoretical value
        if theoretical_effort is not None:
            ax.axhline(y=theoretical_effort, color=colors["theory"],
                      linewidth=2.5, linestyle=':', label=f"Theory (e*={theoretical_effort:.2f})")
        
        # Add legend to first panel
        if idx == 0:
            ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Convergence figure saved to {output_file}")
    plt.close()


def plot_convergence_separate_agents(data: Dict, output_file: str = "results/convergence_separate_agents.png"):
    """
    Create convergence figure showing agent1 and agent2 in separate panels.
    
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
        ax1.set_title(f"Agent 1, q={q:.1f}", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Training Steps", fontsize=12)
        if idx == 0:
            ax1.set_ylabel("Agent 1 Effort", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Agent 2 panel (bottom row)
        ax2 = axes[1, idx]
        ax2.set_title(f"Agent 2, q={q:.1f}", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Training Steps", fontsize=12)
        if idx == 0:
            ax2.set_ylabel("Agent 2 Effort", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        theoretical_effort = None
        
        # Plot gradient
        if q in data.get("gradient", {}):
            grad_data = data["gradient"][q]
            theoretical_effort = grad_data.get("theoretical_effort")
            
            steps = grad_data.get("steps", [])
            e1 = grad_data.get("agent1_effort", [])
            e2 = grad_data.get("agent2_effort", [])
            
            ax1.plot(steps, e1, label="Gradient", color=colors["gradient"], linewidth=2)
            ax2.plot(steps, e2, label="Gradient", color=colors["gradient"], linewidth=2)
        
        # Plot PPO
        if q in data.get("PPO", {}):
            ppo_data = data["PPO"][q]
            if theoretical_effort is None:
                theoretical_effort = ppo_data.get("theoretical_effort")
            
            steps = ppo_data.get("steps", [])
            e1 = ppo_data.get("agent1_effort", [])
            e2 = ppo_data.get("agent2_effort", [])
            
            # Subsample if needed
            if len(steps) > 1000:
                subsample = len(steps) // 1000
                steps = steps[::subsample]
                e1 = e1[::subsample]
                e2 = e2[::subsample]
            
            ax1.plot(steps, e1, label="PPO", color=colors["ppo"], linewidth=2, linestyle='--')
            ax2.plot(steps, e2, label="PPO", color=colors["ppo"], linewidth=2, linestyle='--')
        
        # Add theoretical line
        if theoretical_effort is not None:
            ax1.axhline(y=theoretical_effort, color=colors["theory"],
                       linewidth=2.5, linestyle=':', label=f"Theory ({theoretical_effort:.2f})")
            ax2.axhline(y=theoretical_effort, color=colors["theory"],
                       linewidth=2.5, linestyle=':', label=f"Theory ({theoretical_effort:.2f})")
        
        if idx == 0:
            ax1.legend(loc='best', fontsize=10)
            ax2.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Separate agents figure saved to {output_file}")
    plt.close()


def main():
    """Main execution function."""
    print("="*60)
    print("📊 Convergence Plotting Tool")
    print("="*60)
    
    # Load data
    data = load_convergence_data()
    
    # Print summary
    print("\n📋 Data Summary:")
    for algorithm, alg_data in data.items():
        if alg_data:
            q_vals = sorted(alg_data.keys())
            print(f"  {algorithm}: {len(q_vals)} q values - {q_vals}")
        else:
            print(f"  {algorithm}: No data found")
    
    # Create plots
    if any(data.values()):
        print("\n🎨 Generating plots...")
        plot_convergence_figure(data)
        plot_convergence_separate_agents(data)
        print("\n✅ All plots generated successfully!")
    else:
        print("\n❌ No data available to plot.")
        print("💡 Run your experiments first to generate convergence data:")
        print("   - PPO: python run/run_two_players.py --method ppo --q 25 --episodes 2048000")
        print("   - Gradient: python run/run_two_players.py --method gradient --q 25")


if __name__ == "__main__":
    main()
