"""
Figure Generation for Paper Artifacts.

Implements:
- Main convergence figure (1x3 faceted by q)
- KL/exploitability dynamics
- Beta distribution evolution
- Ablation comparison
"""

import os
from typing import List, Optional, Tuple, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import beta as beta_dist

from .config import (
    FIGURES_DIR,
    DATA_DIR,
    OUTPUT_DPI,
    METHOD_COLORS,
    METHOD_LINESTYLES,
    ABLATION_COLORS,
    FIGURE_SIZES,
    FONT_SIZES,
    Q_VALUES,
    e_star,
    THEORY_PARAMS,
)
from .extract import (
    load_all_convergence_data,
    forward_fill_exploitability,
    aggregate_seeds,
)


def setup_matplotlib_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.size': FONT_SIZES['tick_label'],
        'axes.titlesize': FONT_SIZES['title'],
        'axes.labelsize': FONT_SIZES['axis_label'],
        'legend.fontsize': FONT_SIZES['legend'],
        'figure.dpi': 100,
        'savefig.dpi': OUTPUT_DPI,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def _get_color(method: str) -> str:
    """Get color for a method."""
    method_upper = method.upper()
    if method_upper == "PPO":
        method_upper = "TEL-PPO"
    return METHOD_COLORS.get(method, METHOD_COLORS.get(method_upper, "gray"))


def _get_linestyle(method: str) -> str:
    """Get line style for a method."""
    method_upper = method.upper()
    if method_upper == "PPO":
        method_upper = "TEL-PPO"
    return METHOD_LINESTYLES.get(method, METHOD_LINESTYLES.get(method_upper, "-"))


def plot_convergence_main(
    df: pd.DataFrame = None,
    q_values: List[float] = None,
    output_path: str = None,
    save_data: bool = True,
) -> Tuple[plt.Figure, str]:
    """
    Generate main convergence figure: 1x3 grid faceted by q.
    
    Each panel shows:
    - x-axis: Training steps
    - y-axis: Effort
    - Lines: Theory (dashed), Gradient (blue), TEL-PPO (orange)
    - Multi-seed: mean + 95% CI band
    
    Args:
        df: DataFrame with convergence data (if None, loads automatically)
        q_values: Q values to plot (default: [25, 40, 55])
        output_path: Path to save figure (default: paper_out/figures/convergence_main.png)
        save_data: Whether to save underlying data to CSV
    
    Returns:
        (figure, output_path)
    """
    setup_matplotlib_style()
    ensure_output_dirs()
    
    if df is None:
        df = load_all_convergence_data()
    if q_values is None:
        q_values = Q_VALUES
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "convergence_main.png")
    
    # Filter to requested q values
    df = df[df["q"].isin(q_values)]
    
    # Check if we have multiple seeds
    has_multi_seed = df.groupby(["method", "q", "ablation"])["seed"].nunique().max() > 1
    
    # Create figure
    fig, axes = plt.subplots(1, len(q_values), figsize=FIGURE_SIZES["convergence_main"])
    if len(q_values) == 1:
        axes = [axes]
    
    for ax, q in zip(axes, q_values):
        q_df = df[df["q"] == q]
        e_theory = e_star(q, **THEORY_PARAMS)
        
        # Plot theory line (horizontal dashed)
        ax.axhline(y=e_theory, color=_get_color("Theory"), linestyle=_get_linestyle("Theory"),
                   linewidth=2, label="Theory", zorder=1)
        
        # Plot each method
        for method in ["Gradient", "TEL-PPO"]:
            method_df = q_df[q_df["method"] == method]
            if method_df.empty:
                # Try alternate names
                if method == "TEL-PPO":
                    method_df = q_df[q_df["method"].str.upper() == "PPO"]
                if method_df.empty:
                    continue
            
            color = _get_color(method)
            linestyle = _get_linestyle(method)
            
            if has_multi_seed:
                # Aggregate across seeds
                agg_df = aggregate_seeds(method_df)
                if "effort_mean_mean" in agg_df.columns:
                    steps = agg_df["step"].values
                    effort_mean = agg_df["effort_mean_mean"].values
                    effort_ci = agg_df["effort_mean_ci95"].values
                    
                    ax.plot(steps, effort_mean, color=color, linestyle=linestyle,
                           linewidth=2, label=method, zorder=2)
                    ax.fill_between(steps, effort_mean - effort_ci, effort_mean + effort_ci,
                                   color=color, alpha=0.2, zorder=1)
            else:
                # Single seed: plot directly
                for (seed, ablation), seed_df in method_df.groupby(["seed", "ablation"]):
                    seed_df = seed_df.sort_values("step")
                    ax.plot(seed_df["step"], seed_df["effort_mean"], color=color,
                           linestyle=linestyle, linewidth=2, label=method if seed == method_df["seed"].iloc[0] else None,
                           zorder=2)
        
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Effort")
        ax.set_title(f"q = {q}")
        ax.legend(loc="best")
        
        # Format x-axis with scientific notation for large numbers
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k" if x >= 1e3 else f"{x:.0f}"))
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
    
    # Also save as PDF
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    # Save underlying data
    if save_data:
        data_path = os.path.join(DATA_DIR, "convergence_main.csv")
        df_out = df[["step", "method", "q", "seed", "ablation", "effort_mean", 
                     "agent1_effort", "agent2_effort", "theoretical_effort"]].copy()
        df_out.to_csv(data_path, index=False)
        print(f"[plots] Saved data to {data_path}")
    
    print(f"[plots] Saved figure to {output_path}")
    print(f"[plots] Saved figure to {pdf_path}")
    
    return fig, output_path


def plot_kl_dynamics(
    df: pd.DataFrame = None,
    q_values: List[float] = None,
    output_path: str = None,
    save_data: bool = True,
) -> Tuple[plt.Figure, str]:
    """
    Plot KL divergence dynamics over training.
    
    Shows approx_kl and mean_kl_window over steps.
    """
    setup_matplotlib_style()
    ensure_output_dirs()
    
    if df is None:
        df = load_all_convergence_data()
    if q_values is None:
        q_values = Q_VALUES
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "kl_dynamics.png")
    
    # Filter to requested q values and PPO only
    df = df[(df["q"].isin(q_values)) & (df["method"].isin(["TEL-PPO", "PPO"]))]
    
    if df.empty or "approx_kl" not in df.columns:
        print("[plots] Warning: No KL data available for kl_dynamics plot")
        return None, None
    
    fig, axes = plt.subplots(1, len(q_values), figsize=FIGURE_SIZES["kl_dynamics"])
    if len(q_values) == 1:
        axes = [axes]
    
    for ax, q in zip(axes, q_values):
        q_df = df[df["q"] == q].sort_values("step")
        
        if q_df.empty:
            ax.set_title(f"q = {q} (no data)")
            continue
        
        # Plot approx_kl
        valid_kl = q_df[~q_df["approx_kl"].isna()]
        if not valid_kl.empty:
            ax.plot(valid_kl["step"], valid_kl["approx_kl"], 
                   color="#1f77b4", alpha=0.6, linewidth=1, label="approx_kl")
        
        # Plot mean_kl_window if available
        if "mean_kl_window" in q_df.columns:
            valid_window = q_df[~q_df["mean_kl_window"].isna()]
            if not valid_window.empty:
                ax.plot(valid_window["step"], valid_window["mean_kl_window"],
                       color="#ff7f0e", linewidth=2, label="mean_kl_window")
        
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("KL Divergence")
        ax.set_title(f"q = {q}")
        ax.legend(loc="best")
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    if save_data:
        data_path = os.path.join(DATA_DIR, "kl_dynamics.csv")
        cols = ["step", "method", "q", "seed", "ablation", "approx_kl"]
        if "mean_kl_window" in df.columns:
            cols.append("mean_kl_window")
        df[cols].to_csv(data_path, index=False)
    
    print(f"[plots] Saved figure to {output_path}")
    return fig, output_path


def plot_exploitability_dynamics(
    df: pd.DataFrame = None,
    q_values: List[float] = None,
    output_path: str = None,
    save_data: bool = True,
) -> Tuple[plt.Figure, str]:
    """
    Plot exploitability over training.
    
    Uses forward-fill for sparse evaluations.
    """
    setup_matplotlib_style()
    ensure_output_dirs()
    
    if df is None:
        df = load_all_convergence_data()
    if q_values is None:
        q_values = Q_VALUES
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "exploitability_dynamics.png")
    
    df = df[(df["q"].isin(q_values)) & (df["method"].isin(["TEL-PPO", "PPO"]))]
    
    if df.empty or "exploitability" not in df.columns:
        print("[plots] Warning: No exploitability data available")
        return None, None
    
    # Forward-fill exploitability
    df = forward_fill_exploitability(df)
    
    fig, axes = plt.subplots(1, len(q_values), figsize=FIGURE_SIZES["exploitability_dynamics"])
    if len(q_values) == 1:
        axes = [axes]
    
    for ax, q in zip(axes, q_values):
        q_df = df[df["q"] == q].sort_values("step")
        
        if q_df.empty:
            ax.set_title(f"q = {q} (no data)")
            continue
        
        # Plot forward-filled exploitability
        valid = q_df[~q_df["exploitability_ffill"].isna()]
        if not valid.empty:
            ax.plot(valid["step"], valid["exploitability_ffill"],
                   color="#1f77b4", linewidth=2, label="Exploitability (ffill)")
        
        # Mark actual evaluation points
        if "exploitability_is_valid" in q_df.columns:
            eval_points = q_df[q_df["exploitability_is_valid"] == True]
            if not eval_points.empty:
                ax.scatter(eval_points["step"], eval_points["exploitability"],
                          color="#ff7f0e", s=30, zorder=3, label="Evaluated")
        
        # Threshold line
        ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="Threshold (0.05)")
        
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Exploitability")
        ax.set_title(f"q = {q}")
        ax.legend(loc="best")
    
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    if save_data:
        data_path = os.path.join(DATA_DIR, "exploitability_dynamics.csv")
        cols = ["step", "method", "q", "seed", "ablation", "exploitability", "exploitability_is_valid"]
        if "exploitability_ffill" in df.columns:
            cols.append("exploitability_ffill")
        df[cols].to_csv(data_path, index=False)
    
    print(f"[plots] Saved figure to {output_path}")
    return fig, output_path


def plot_beta_evolution(
    df: pd.DataFrame = None,
    q_values: List[float] = None,
    output_path: str = None,
    save_data: bool = True,
) -> Tuple[plt.Figure, str]:
    """
    Plot alpha/beta parameter evolution over training.
    """
    setup_matplotlib_style()
    ensure_output_dirs()
    
    if df is None:
        df = load_all_convergence_data()
    if q_values is None:
        q_values = Q_VALUES
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "beta_evolution.png")
    
    df = df[(df["q"].isin(q_values)) & (df["method"].isin(["TEL-PPO", "PPO"]))]
    
    if df.empty or "alpha_mean" not in df.columns:
        print("[plots] Warning: No alpha/beta data available")
        return None, None
    
    fig, axes = plt.subplots(2, len(q_values), figsize=(12, 6))
    if len(q_values) == 1:
        axes = axes.reshape(2, 1)
    
    for col_idx, q in enumerate(q_values):
        q_df = df[df["q"] == q].sort_values("step")
        
        if q_df.empty:
            axes[0, col_idx].set_title(f"q = {q} (no data)")
            continue
        
        valid = q_df[~q_df["alpha_mean"].isna()]
        
        # Top: alpha
        axes[0, col_idx].plot(valid["step"], valid["alpha_mean"], 
                             color="#1f77b4", linewidth=2)
        axes[0, col_idx].set_ylabel("Alpha")
        axes[0, col_idx].set_title(f"q = {q}")
        
        # Bottom: beta
        axes[1, col_idx].plot(valid["step"], valid["beta_mean"],
                             color="#ff7f0e", linewidth=2)
        axes[1, col_idx].set_xlabel("Training Steps")
        axes[1, col_idx].set_ylabel("Beta")
    
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    if save_data:
        data_path = os.path.join(DATA_DIR, "beta_evolution.csv")
        df[["step", "method", "q", "seed", "ablation", "alpha_mean", "beta_mean"]].to_csv(data_path, index=False)
    
    print(f"[plots] Saved figure to {output_path}")
    return fig, output_path


def plot_beta_snapshots(
    df: pd.DataFrame = None,
    q: float = 40.0,
    snapshot_fractions: List[float] = None,
    output_path: str = None,
) -> Tuple[plt.Figure, str]:
    """
    Plot Beta distribution snapshots at different training stages.
    
    Args:
        df: DataFrame with convergence data
        q: Q value to use
        snapshot_fractions: Fractions of training to snapshot (default: [0.1, 0.5, 1.0])
        output_path: Output path
    """
    setup_matplotlib_style()
    ensure_output_dirs()
    
    if df is None:
        df = load_all_convergence_data()
    if snapshot_fractions is None:
        snapshot_fractions = [0.1, 0.5, 1.0]
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "beta_snapshots.png")
    
    df = df[(df["q"] == q) & (df["method"].isin(["TEL-PPO", "PPO"]))]
    
    if df.empty or "alpha_mean" not in df.columns:
        print("[plots] Warning: No alpha/beta data available for snapshots")
        return None, None
    
    df = df.sort_values("step")
    max_step = df["step"].max()
    
    fig, axes = plt.subplots(1, len(snapshot_fractions), figsize=FIGURE_SIZES["beta_snapshots"])
    
    x = np.linspace(0.001, 0.999, 200)
    
    for ax, frac in zip(axes, snapshot_fractions):
        target_step = int(max_step * frac)
        # Find closest step
        closest_idx = (df["step"] - target_step).abs().idxmin()
        row = df.loc[closest_idx]
        
        alpha = row["alpha_mean"]
        beta_val = row["beta_mean"]
        
        if np.isnan(alpha) or np.isnan(beta_val):
            ax.set_title(f"Step {target_step} ({frac*100:.0f}%) - No data")
            continue
        
        # Plot Beta PDF
        y = beta_dist.pdf(x, alpha, beta_val)
        ax.plot(x, y, color="#1f77b4", linewidth=2)
        ax.fill_between(x, y, alpha=0.3, color="#1f77b4")
        
        # Mark mean
        mean = alpha / (alpha + beta_val)
        ax.axvline(x=mean, color="#ff7f0e", linestyle="--", linewidth=2, label=f"Mean={mean:.3f}")
        
        ax.set_xlabel("Action (normalized)")
        ax.set_ylabel("Density")
        ax.set_title(f"Step {int(row['step'])} ({frac*100:.0f}%)\nα={alpha:.1f}, β={beta_val:.1f}")
        ax.legend(loc="best")
    
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    print(f"[plots] Saved figure to {output_path}")
    return fig, output_path


def plot_ablation_comparison(
    df: pd.DataFrame = None,
    q_values: List[float] = None,
    output_path: str = None,
    save_data: bool = True,
) -> Tuple[plt.Figure, str]:
    """
    Plot comparison of ablation conditions.
    
    Shows convergence curves for baseline, no_cheap_gate, no_exploitability.
    """
    setup_matplotlib_style()
    ensure_output_dirs()
    
    if df is None:
        df = load_all_convergence_data()
    if q_values is None:
        q_values = Q_VALUES
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "ablation_comparison.png")
    
    df = df[(df["q"].isin(q_values)) & (df["method"].isin(["TEL-PPO", "PPO"]))]
    
    if df.empty:
        print("[plots] Warning: No data for ablation comparison")
        return None, None
    
    # Get unique ablations
    ablations = df["ablation"].unique()
    
    fig, axes = plt.subplots(len(q_values), 1, figsize=FIGURE_SIZES["ablation_comparison"])
    if len(q_values) == 1:
        axes = [axes]
    
    for ax, q in zip(axes, q_values):
        q_df = df[df["q"] == q]
        e_theory = e_star(q, **THEORY_PARAMS)
        
        # Theory line
        ax.axhline(y=e_theory, color="black", linestyle="--", linewidth=1.5, label="Theory")
        
        # Plot each ablation
        for ablation in sorted(ablations):
            abl_df = q_df[q_df["ablation"] == ablation].sort_values("step")
            if abl_df.empty:
                continue
            
            color = ABLATION_COLORS.get(ablation, "gray")
            ax.plot(abl_df["step"], abl_df["effort_mean"], 
                   color=color, linewidth=2, label=ablation)
        
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Effort")
        ax.set_title(f"q = {q}")
        ax.legend(loc="best")
    
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    if save_data:
        data_path = os.path.join(DATA_DIR, "ablation_comparison.csv")
        df[["step", "method", "q", "seed", "ablation", "effort_mean", "theoretical_effort"]].to_csv(data_path, index=False)
    
    print(f"[plots] Saved figure to {output_path}")
    return fig, output_path


def generate_all_figures(
    df: pd.DataFrame = None,
    q_values: List[float] = None,
    output_dir: str = None,
) -> Dict[str, str]:
    """
    Generate all figures for the paper.
    
    Returns dict mapping figure name to output path.
    """
    if df is None:
        df = load_all_convergence_data()
    if q_values is None:
        q_values = Q_VALUES
    if output_dir is not None:
        global FIGURES_DIR, DATA_DIR
        FIGURES_DIR = os.path.join(output_dir, "figures")
        DATA_DIR = os.path.join(output_dir, "data")
    
    results = {}
    
    # Main convergence
    fig, path = plot_convergence_main(df, q_values)
    if path:
        results["convergence_main"] = path
    
    # KL dynamics
    fig, path = plot_kl_dynamics(df, q_values)
    if path:
        results["kl_dynamics"] = path
    
    # Exploitability dynamics
    fig, path = plot_exploitability_dynamics(df, q_values)
    if path:
        results["exploitability_dynamics"] = path
    
    # Beta evolution
    fig, path = plot_beta_evolution(df, q_values)
    if path:
        results["beta_evolution"] = path
    
    # Beta snapshots (for q=40)
    fig, path = plot_beta_snapshots(df, q=40.0)
    if path:
        results["beta_snapshots"] = path
    
    # Ablation comparison
    fig, path = plot_ablation_comparison(df, q_values)
    if path:
        results["ablation_comparison"] = path
    
    plt.close('all')  # Clean up
    
    return results


if __name__ == "__main__":
    print("Generating all figures...")
    results = generate_all_figures()
    print("\nGenerated figures:")
    for name, path in results.items():
        print(f"  {name}: {path}")
