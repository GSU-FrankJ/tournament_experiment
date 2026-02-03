#!/usr/bin/env python3
"""
Generate detailed convergence plots for individual algorithms.

Creates combined and separated views showing how two agents learn and converge
to the theoretical optimal effort value.
"""

import json
import os
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


def load_convergence_file(filepath: str) -> Optional[Dict]:
    """Load a single convergence history JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded {os.path.basename(filepath)}")
        return data
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None


def plot_combined(data: Dict, output_dir: str = "results/convergence_plots", suffix: str = ""):
    """
    Create a combined plot showing both agents on the same axes.
    
    Figure shows:
    - Agent1 effort (blue solid line)
    - Agent2 effort (orange solid line)
    - Theoretical effort (black dashed line)
    - Average effort (green dotted line, optional)
    
    Args:
        data: Convergence data dictionary
        output_dir: Base output directory for plots
        suffix: Optional suffix to add to output filename (e.g., "k0.0005_wh8_wl3")
    """
    algorithm = data.get("algorithm", "unknown")
    q = data.get("q", 0.0)
    theoretical_effort = data.get("theoretical_effort", 0.0)
    steps = data.get("steps", [])
    agent1_effort = data.get("agent1_effort", [])
    agent2_effort = data.get("agent2_effort", [])
    
    if not steps or not agent1_effort or not agent2_effort:
        print(f"❌ Incomplete data for {algorithm} q={q}")
        return
    
    # Create output directory
    algo_dir = os.path.join(output_dir, algorithm.lower())
    os.makedirs(algo_dir, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot agent efforts
    ax.plot(steps, agent1_effort, 
            color='#1f77b4', linewidth=2.5, label='Agent 1', alpha=0.9)
    ax.plot(steps, agent2_effort,
            color='#ff7f0e', linewidth=2.5, label='Agent 2', alpha=0.9)
    
    # Plot theoretical value
    ax.axhline(y=theoretical_effort, 
               color='black', linewidth=2, linestyle='--', 
               label=f'Theoretical (e*={theoretical_effort:.2f})', alpha=0.7)
    
    # Calculate and plot average effort (optional)
    if len(agent1_effort) == len(agent2_effort):
        avg_effort = [(e1 + e2) / 2.0 for e1, e2 in zip(agent1_effort, agent2_effort)]
        ax.plot(steps, avg_effort,
                color='#2ca02c', linewidth=2, linestyle=':', 
                label='Average Effort', alpha=0.6)
    
    # Styling
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Effort', fontsize=14, fontweight='bold')
    ax.set_title(f'{algorithm.upper()} Convergence: Both Agents (q={q:.1f})', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=12, framealpha=0.95)
    
    # Force scientific notation on x-axis for consistency across all plots
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
    
    # Add info box
    final_e1 = agent1_effort[-1]
    final_e2 = agent2_effort[-1]
    gap1 = abs(final_e1 - theoretical_effort)
    gap2 = abs(final_e2 - theoretical_effort)
    
    info_text = (
        f'Final State:\n'
        f'Agent 1: {final_e1:.2f} (gap: {gap1:.2f})\n'
        f'Agent 2: {final_e2:.2f} (gap: {gap2:.2f})'
    )
    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure with optional suffix
    if suffix:
        output_file = os.path.join(algo_dir, f'q{q:.1f}_{suffix}_combined.png')
    else:
        output_file = os.path.join(algo_dir, f'q{q:.1f}_combined.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved combined plot: {output_file}")
    plt.close()


def plot_separated(data: Dict, output_dir: str = "results/convergence_plots", suffix: str = ""):
    """
    Create separated plots showing each agent in its own subplot.
    
    Figure has two vertically stacked subplots:
    - Top: Agent 1 convergence
    - Bottom: Agent 2 convergence
    Both show theoretical value as reference.
    
    Args:
        data: Convergence data dictionary
        output_dir: Base output directory for plots
        suffix: Optional suffix to add to output filename (e.g., "k0.0005_wh8_wl3")
    """
    algorithm = data.get("algorithm", "unknown")
    q = data.get("q", 0.0)
    theoretical_effort = data.get("theoretical_effort", 0.0)
    steps = data.get("steps", [])
    agent1_effort = data.get("agent1_effort", [])
    agent2_effort = data.get("agent2_effort", [])
    
    if not steps or not agent1_effort or not agent2_effort:
        print(f"❌ Incomplete data for {algorithm} q={q}")
        return
    
    # Create output directory
    algo_dir = os.path.join(output_dir, algorithm.lower())
    os.makedirs(algo_dir, exist_ok=True)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- Agent 1 subplot ---
    ax1.plot(steps, agent1_effort,
             color='#1f77b4', linewidth=2.5, label='Agent 1 Effort')
    ax1.axhline(y=theoretical_effort,
                color='black', linewidth=2, linestyle='--',
                label=f'Theoretical (e*={theoretical_effort:.2f})', alpha=0.7)
    
    ax1.set_ylabel('Effort', fontsize=13, fontweight='bold')
    ax1.set_title(f'Agent 1 Convergence', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=11, framealpha=0.95)
    
    # Add final value annotation
    final_e1 = agent1_effort[-1]
    gap1 = abs(final_e1 - theoretical_effort)
    ax1.text(0.98, 0.05, f'Final: {final_e1:.2f}\nGap: {gap1:.2f}',
             transform=ax1.transAxes,
             fontsize=10,
             horizontalalignment='right',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # --- Agent 2 subplot ---
    ax2.plot(steps, agent2_effort,
             color='#ff7f0e', linewidth=2.5, label='Agent 2 Effort')
    ax2.axhline(y=theoretical_effort,
                color='black', linewidth=2, linestyle='--',
                label=f'Theoretical (e*={theoretical_effort:.2f})', alpha=0.7)
    
    ax2.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Effort', fontsize=13, fontweight='bold')
    ax2.set_title(f'Agent 2 Convergence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=11, framealpha=0.95)
    
    # Force scientific notation on x-axis for consistency across all plots
    ax2.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
    
    # Add final value annotation
    final_e2 = agent2_effort[-1]
    gap2 = abs(final_e2 - theoretical_effort)
    ax2.text(0.98, 0.05, f'Final: {final_e2:.2f}\nGap: {gap2:.2f}',
             transform=ax2.transAxes,
             fontsize=10,
             horizontalalignment='right',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.8))
    
    # Overall title
    fig.suptitle(f'{algorithm.upper()} Agent Convergence (q={q:.1f})',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save figure with optional suffix
    if suffix:
        output_file = os.path.join(algo_dir, f'q{q:.1f}_{suffix}_separated.png')
    else:
        output_file = os.path.join(algo_dir, f'q{q:.1f}_separated.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved separated plot: {output_file}")
    plt.close()


def plot_convergence_for_file(filepath: str, output_dir: str = "results/convergence_plots", suffix: str = ""):
    """Load a convergence file and generate both combined and separated plots.
    
    Args:
        filepath: Path to convergence JSON file
        output_dir: Base output directory for plots
        suffix: Optional suffix to add to output filenames
    """
    data = load_convergence_file(filepath)
    if data is None:
        return
    
    print(f"\n📊 Generating plots for {data.get('algorithm', 'unknown')} q={data.get('q', 0.0)}")
    plot_combined(data, output_dir, suffix=suffix)
    plot_separated(data, output_dir, suffix=suffix)


def process_all_files(convergence_dir: str = "results/convergence_history",
                      output_dir: str = "results/convergence_plots",
                      algorithm: Optional[str] = None,
                      q_value: Optional[float] = None,
                      suffix: str = "",
                      file_pattern: Optional[str] = None):
    """
    Process all convergence history files and generate plots.
    
    Args:
        convergence_dir: Directory containing convergence JSON files
        output_dir: Directory to save generated plots
        algorithm: Optional filter for specific algorithm (e.g., "PPO", "gradient")
        q_value: Optional filter for specific q value
        suffix: Optional suffix to add to output filenames (e.g., "k0.0005_wh8_wl3")
        file_pattern: Optional substring to filter convergence files (e.g., "k5e4_wh8_wl3")
    """
    if not os.path.exists(convergence_dir):
        print(f"❌ Convergence history directory not found: {convergence_dir}")
        return
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(convergence_dir, "*_convergence.json"))
    
    if not json_files:
        print(f"❌ No convergence files found in {convergence_dir}")
        return
    
    print(f"Found {len(json_files)} convergence files")
    print("="*60)
    
    processed = 0
    for filepath in sorted(json_files):
        # Apply file pattern filter if specified
        if file_pattern and file_pattern not in os.path.basename(filepath):
            continue
        
        # Load data to check filters
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Apply filters
        if algorithm and data.get("algorithm", "").lower() != algorithm.lower():
            continue
        if q_value is not None and abs(data.get("q", 0.0) - q_value) > 0.01:
            continue
        
        plot_convergence_for_file(filepath, output_dir, suffix=suffix)
        processed += 1
    
    print("="*60)
    print(f"✅ Processed {processed} files")
    print(f"📁 Plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate detailed convergence plots from convergence history JSON files"
    )
    parser.add_argument(
        "--convergence-dir",
        type=str,
        default="results/convergence_history",
        help="Directory containing convergence JSON files (default: results/convergence_history)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/convergence_plots",
        help="Directory to save generated plots (default: results/convergence_plots)"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["PPO", "ppo", "gradient", "Gradient"],
        help="Filter by algorithm (PPO or gradient)"
    )
    parser.add_argument(
        "--q",
        type=float,
        help="Filter by q value (e.g., 25.0, 40.0, 55.0)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Process a single specific convergence file"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to add to output filenames (e.g., 'k0.0005_wh8_wl3')"
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        help="Filter convergence files by substring in filename (e.g., 'k5e4_wh8_wl3')"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("📊 Detailed Convergence Plot Generator")
    print("="*60)
    
    if args.file:
        # Process single file
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            return
        plot_convergence_for_file(args.file, args.output_dir, suffix=args.suffix)
    else:
        # Process all files with optional filters
        process_all_files(
            convergence_dir=args.convergence_dir,
            output_dir=args.output_dir,
            algorithm=args.algorithm,
            q_value=args.q,
            suffix=args.suffix,
            file_pattern=args.file_pattern
        )
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
