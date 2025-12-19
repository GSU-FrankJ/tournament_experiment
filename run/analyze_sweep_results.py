#!/usr/bin/env python3
"""
Analyze and visualize gradient parameter sweep results.

Usage:
    # Single file
    python run/analyze_sweep_results.py results/sweeps/sweep_random_q40.0_20251202_231142.csv
    
    # Multiple files (shell expands wildcards)
    python run/analyze_sweep_results.py results/sweeps/sweep_random_q40.0_*.csv
    
    # Multiple files explicitly
    python run/analyze_sweep_results.py file1.csv file2.csv file3.csv
"""

import sys
import os
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_sweep_results(csv_paths: list) -> pd.DataFrame:
    """Load sweep results from one or more CSV files."""
    dfs = []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"⚠️  Warning: File not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        # Filter successful results
        df = df[df["status"] == "success"].copy()
        if len(df) > 0:
            dfs.append(df)
            print(f"  ✅ Loaded {len(df)} results from {os.path.basename(csv_path)}")
    
    if not dfs:
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def plot_parameter_sensitivity(df: pd.DataFrame, output_dir: str):
    """Plot sensitivity of gap to each parameter."""
    os.makedirs(output_dir, exist_ok=True)
    
    params = ["lr", "steps", "grad_eps", "tol", "num_samples", "init_perturb"]
    n_params = len(params)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, param in enumerate(params):
        ax = axes[i]
        
        # Scatter plot: parameter vs gap (now max_gap)
        ax.scatter(df[param], df["gap"], alpha=0.5, s=20)
        ax.set_xlabel(param)
        ax.set_ylabel("Max Gap (max(|e1-theory|, |e2-theory|))")
        ax.set_title(f"Max Gap vs {param}")
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(df) > 1:
            z = np.polyfit(df[param], df["gap"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df[param].min(), df[param].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, label="Trend")
            ax.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "parameter_sensitivity.png")
    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved parameter sensitivity plot to {output_path}")
    plt.close()


def plot_quality_distribution(df: pd.DataFrame, output_dir: str):
    """Plot distribution of quality ratings."""
    os.makedirs(output_dir, exist_ok=True)
    
    quality_counts = df["quality"].value_counts()
    quality_order = ["Excellent", "Good", "Fair", "Poor"]
    quality_counts = quality_counts.reindex([q for q in quality_order if q in quality_counts.index])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    quality_counts.plot(kind="bar", ax=ax, color=["green", "blue", "orange", "red"])
    ax.set_xlabel("Quality")
    ax.set_ylabel("Count")
    ax.set_title("Quality Distribution")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "quality_distribution.png")
    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved quality distribution to {output_path}")
    plt.close()


def plot_top_parameters(df: pd.DataFrame, output_dir: str, top_n: int = 10):
    """Visualize top N parameter combinations."""
    os.makedirs(output_dir, exist_ok=True)
    
    top_df = df.nsmallest(top_n, "gap")
    
    params = ["lr", "steps", "grad_eps", "tol", "num_samples", "init_perturb"]
    
    fig, axes = plt.subplots(len(params), 1, figsize=(10, 2 * len(params)))
    
    for i, param in enumerate(params):
        ax = axes[i]
        ax.barh(range(len(top_df)), top_df[param])
        ax.set_yticks(range(len(top_df)))
        ax.set_yticklabels([f"#{j+1}" for j in range(len(top_df))])
        ax.set_xlabel(param)
        ax.set_title(f"Top {top_n} Parameter Combinations: {param}")
        ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "top_parameters.png")
    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved top parameters visualization to {output_path}")
    plt.close()


def generate_summary_statistics(df: pd.DataFrame) -> dict:
    """Generate summary statistics."""
    best_row = df.nsmallest(1, "gap").iloc[0]
    stats = {
        "total_trials": len(df),
        "gap_stats": {
            "mean": df["gap"].mean(),
            "median": df["gap"].median(),
            "std": df["gap"].std(),
            "min": df["gap"].min(),
            "max": df["gap"].max(),
        },
        "quality_distribution": df["quality"].value_counts().to_dict(),
        "best_parameters": best_row[["lr", "steps", "grad_eps", "tol", 
                                     "num_samples", "init_perturb"]].to_dict(),
        "best_gap": df["gap"].min(),
        "best_e1": best_row.get("e1", 0),
        "best_e2": best_row.get("e2", 0),
        "best_theoretical": best_row.get("theoretical_effort", 0),
        "best_symmetry_gap": best_row.get("symmetry_gap", 0),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze gradient parameter sweep results")
    parser.add_argument(
        "csv_files",
        type=str,
        nargs="+",
        help="Path(s) to sweep results CSV file(s). Can specify multiple files or use wildcards."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sweeps/analysis",
        help="Output directory for plots (default: results/sweeps/analysis)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top results to visualize (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Handle wildcard expansion (if shell didn't expand it)
    csv_paths = []
    for pattern in args.csv_files:
        if '*' in pattern or '?' in pattern:
            # Expand glob pattern
            expanded = glob.glob(pattern)
            csv_paths.extend(expanded)
        else:
            csv_paths.append(pattern)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in csv_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    
    if not unique_paths:
        print("❌ No CSV files found")
        return
    
    print(f"📊 Loading results from {len(unique_paths)} file(s)...")
    df = load_sweep_results(unique_paths)
    
    if len(df) == 0:
        print("❌ No successful results found in CSV file")
        return
    
    print(f"✅ Loaded {len(df)} successful results")
    
    # Generate statistics
    stats = generate_summary_statistics(df)
    print("\n" + "="*60)
    print("📈 SUMMARY STATISTICS")
    print("="*60)
    print(f"Total trials: {stats['total_trials']}")
    print(f"\nMax Gap Statistics (max(|e1-theory|, |e2-theory|)):")
    print(f"  Mean: {stats['gap_stats']['mean']:.6f}")
    print(f"  Median: {stats['gap_stats']['median']:.6f}")
    print(f"  Std: {stats['gap_stats']['std']:.6f}")
    print(f"  Min: {stats['gap_stats']['min']:.6f}")
    print(f"  Max: {stats['gap_stats']['max']:.6f}")
    print(f"\nQuality Distribution:")
    for quality, count in stats["quality_distribution"].items():
        print(f"  {quality}: {count}")
    print(f"\nBest Parameters:")
    for param, value in stats["best_parameters"].items():
        print(f"  {param}: {value}")
    print(f"\nBest Result Details:")
    print(f"  Theoretical Effort: {stats['best_theoretical']:.6f}")
    print(f"  e1: {stats['best_e1']:.6f}")
    print(f"  e2: {stats['best_e2']:.6f}")
    print(f"  Max Gap: {stats['best_gap']:.6f}")
    print(f"  Symmetry Gap: {stats['best_symmetry_gap']:.6f}")
    print("="*60)
    
    # Generate plots
    print("\n📊 Generating visualizations...")
    plot_parameter_sensitivity(df, args.output_dir)
    plot_quality_distribution(df, args.output_dir)
    plot_top_parameters(df, args.output_dir, top_n=args.top_n)
    
    print(f"\n✅ Analysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

