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
from matplotlib.lines import Line2D
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
    AGENT_COLORS,
    AGENT_MARKERS,
    WEIGHT_VARIANT_LABELS,
    THEORY_LINE_COLOR,
    THEORY_LINE_WIDTH,
    CHEAP_GATE_CONFIG,
    CONVERGENCE_CONFIG,
)
from .extract import (
    load_all_convergence_data,
    forward_fill_exploitability,
    aggregate_seeds,
    compute_effort_error,
    get_final_values,
    get_convergence_step,
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


def _unify_ylim(axes, margin: float = 0.05):
    """Set a shared y-axis range across all given axes with optional margin."""
    y_min = float("inf")
    y_max = float("-inf")
    for ax in np.asarray(axes).flat:
        lo, hi = ax.get_ylim()
        y_min = min(y_min, lo)
        y_max = max(y_max, hi)
    span = y_max - y_min
    for ax in np.asarray(axes).flat:
        ax.set_ylim(y_min - margin * span, y_max + margin * span)


def plot_convergence_main(
    df: pd.DataFrame = None,
    q_values: List[float] = None,
    weight_variants: List[str] = None,
    output_path: str = None,
    save_data: bool = True,
) -> Tuple[plt.Figure, str]:
    """
    Generate main convergence figure: 2x3 grid faceted by q and weight variant.

    Each panel shows per-agent efforts:
    - Agent 1 (solid) and Agent 2 (dashed)
    - Multi-seed: light individual traces (alpha=0.25) + bold aggregate mean with CI band
    - Theory line from data's theoretical_effort column

    Rows correspond to weight variants (top=baseline, bottom=wh8_wl4).
    Columns correspond to q values.

    Args:
        df: DataFrame with convergence data (if None, loads automatically)
        q_values: Q values to plot (default: [25, 40, 55])
        weight_variants: Ablation variants for rows (default: ["baseline", "wh8_wl4"])
        output_path: Path to save figure
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
    if weight_variants is None:
        weight_variants = ["baseline", "wh8_wl4"]
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "convergence_main.png")

    # Filter to two_players and selected variants
    df = df[df["q"].isin(q_values)]
    if "experiment" in df.columns:
        df = df[df["experiment"] == "two_players"]
    df = df[df["ablation"].isin(weight_variants)]

    n_rows = len(weight_variants)
    n_cols = len(q_values)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=FIGURE_SIZES["convergence_main"],
        squeeze=False,
    )

    # Precompute convergence steps for vertical lines
    conv_steps_df = get_convergence_step(df)

    for row_idx, variant in enumerate(weight_variants):
        var_df = df[df["ablation"] == variant]

        for col_idx, q in enumerate(q_values):
            ax = axes[row_idx, col_idx]
            q_df = var_df[var_df["q"] == q]

            if q_df.empty:
                ax.set_title(f"Noise Level q = {q} (no data)")
                continue

            # Theory line from data (use first non-NaN theoretical_effort) → red bold
            theory_vals = q_df["theoretical_effort"].dropna()
            e_theory = np.nan
            if not theory_vals.empty:
                e_theory = theory_vals.iloc[0]
                ax.axhline(
                    y=e_theory, color=THEORY_LINE_COLOR, linestyle="--",
                    linewidth=THEORY_LINE_WIDTH, label="Theory $e^*$", zorder=1,
                )

            # Filter to PPO method
            method_df = q_df[q_df["method"].isin(["TEL-PPO", "PPO"])]
            if method_df.empty:
                continue

            seeds = sorted(method_df["seed"].unique())
            has_multi_seed = len(seeds) > 1

            # Plot per-agent efforts
            for agent_key, col_name, ls, label_base in [
                ("agent1", "agent1_effort", "-", "Agent 1"),
                ("agent2", "agent2_effort", "--", "Agent 2"),
            ]:
                color = AGENT_COLORS[agent_key]

                if col_name not in method_df.columns:
                    continue

                # Individual seed traces (light) — lowered alpha
                if has_multi_seed:
                    for seed in seeds:
                        seed_df = method_df[method_df["seed"] == seed].sort_values("step")
                        ax.plot(
                            seed_df["step"], seed_df[col_name],
                            color=color, linestyle=ls, alpha=0.15,
                            linewidth=0.8, zorder=1,
                        )

                # Aggregate mean + CI
                agg_col_mean = f"{col_name}_mean"
                agg_col_ci = f"{col_name}_ci95"

                if has_multi_seed:
                    agg_df = aggregate_seeds(method_df)
                    if agg_col_mean in agg_df.columns:
                        steps = agg_df["step"].values
                        effort_mean = agg_df[agg_col_mean].values
                        effort_ci = agg_df[agg_col_ci].values if agg_col_ci in agg_df.columns else np.zeros_like(effort_mean)

                        ax.plot(
                            steps, effort_mean, color=color, linestyle=ls,
                            linewidth=2, label=label_base, zorder=3,
                        )
                        ax.fill_between(
                            steps, effort_mean - effort_ci, effort_mean + effort_ci,
                            color=color, alpha=0.10, zorder=2,
                        )
                else:
                    # Single seed: plot directly
                    single = method_df.sort_values("step")
                    ax.plot(
                        single["step"], single[col_name],
                        color=color, linestyle=ls, linewidth=2,
                        label=label_base, zorder=3,
                    )

            # --- Final summary annotation ---
            if not np.isnan(e_theory):
                final_effort = method_df.groupby("seed")["effort_mean"].last().mean()
                final_error = abs(final_effort - e_theory)
                # Final exploitability (last valid per seed, then mean)
                exploit_vals = []
                sym_gaps = []
                for seed in seeds:
                    sdf = method_df[method_df["seed"] == seed].sort_values("step")
                    if "exploitability" in sdf.columns:
                        valid_ex = sdf[sdf["exploitability"].notna()]["exploitability"]
                        if not valid_ex.empty:
                            exploit_vals.append(valid_ex.iloc[-1])
                    if "agent1_effort" in sdf.columns and "agent2_effort" in sdf.columns:
                        sym_gaps.append(abs(sdf["agent1_effort"].iloc[-1] - sdf["agent2_effort"].iloc[-1]))

                ann_lines = [f"|ē−e*|={final_error:.2f}"]
                if exploit_vals:
                    ann_lines.append(f"ε={np.mean(exploit_vals):.3f}")
                if sym_gaps:
                    ann_lines.append(f"Δsym={np.mean(sym_gaps):.2f}")

                ax.text(
                    0.97, 0.97, "\n".join(ann_lines),
                    transform=ax.transAxes, fontsize=7,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
                )

            # Axis formatting
            if row_idx == n_rows - 1:
                ax.set_xlabel("Training Steps")
            ax.set_ylabel("Effort")

            # Title: "Noise Level q = {q}" on top row only
            if row_idx == 0:
                ax.set_title(f"Noise Level q = {int(q)}")

            # Row label on the leftmost column
            if col_idx == 0:
                variant_label = WEIGHT_VARIANT_LABELS.get(variant, variant)
                ax.annotate(
                    variant_label, xy=(0, 0.5),
                    xytext=(-50, 0), textcoords="offset points",
                    xycoords="axes fraction", ha="right", va="center",
                    fontsize=FONT_SIZES["axis_label"], rotation=90,
                )

            # Legend only on top-left panel
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper left", fontsize=FONT_SIZES["legend"])

            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(
                    lambda x, p: f"{x/1e6:.1f}M" if x >= 1e6
                    else f"{x/1e3:.0f}k" if x >= 1e3
                    else f"{x:.0f}"
                )
            )

    # Unify y-axis across all panels
    _unify_ylim(axes)

    plt.tight_layout()
    plt.subplots_adjust(left=0.12)

    # Save figure
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')

    # Save underlying data
    if save_data:
        data_path = os.path.join(DATA_DIR, "convergence_main.csv")
        out_cols = ["step", "method", "q", "seed", "ablation", "effort_mean",
                    "agent1_effort", "agent2_effort", "theoretical_effort"]
        df_out = df[[c for c in out_cols if c in df.columns]].copy()
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
    
    # Filter to two_players baseline PPO only
    df = df[(df["q"].isin(q_values)) & (df["method"].isin(["TEL-PPO", "PPO"])) & (df["ablation"] == "baseline")]
    if "experiment" in df.columns:
        df = df[df["experiment"] == "two_players"]

    if df.empty or "approx_kl" not in df.columns:
        print("[plots] Warning: No KL data available for kl_dynamics plot")
        return None, None
    
    # Colorblind-safe palette
    CB_BLUE = "#0072B2"
    CB_CYAN = "#56B4E9"
    CB_RED = "#D55E00"

    x_formatter = ticker.FuncFormatter(
        lambda x, p: f"{x/1e6:.1f}M" if x >= 1e6
        else f"{x/1e3:.0f}k" if x >= 1e3
        else f"{x:.0f}"
    )

    fig, axes = plt.subplots(1, len(q_values), figsize=FIGURE_SIZES["kl_dynamics"])
    if len(q_values) == 1:
        axes = [axes]

    mean_kl_thresh = CHEAP_GATE_CONFIG["mean_kl_thresh"]

    for idx, (ax, q) in enumerate(zip(axes, q_values)):
        q_df = df[df["q"] == q].copy()

        # Filter to positive KL only (negative values are estimation artifacts)
        q_df = q_df[q_df["approx_kl"].notna() & (q_df["approx_kl"] > 0)]

        if q_df.empty:
            ax.set_title(f"q = {int(q)} (no data)")
            continue

        # Bin the step axis into ~150 equal-width bins
        step_min, step_max = q_df["step"].min(), q_df["step"].max()
        bin_width = max(1, int((step_max - step_min) / 150))
        q_df["step_bin"] = (q_df["step"] // bin_width) * bin_width

        # Aggregate across all seeds per bin
        binned = q_df.groupby("step_bin")["approx_kl"].agg(
            median="median",
            p10=lambda x: np.percentile(x, 10),
            p90=lambda x: np.percentile(x, 90),
        ).reset_index()

        # Light rolling smooth (window=5, centered)
        for col in ["median", "p10", "p90"]:
            binned[col] = binned[col].rolling(5, center=True, min_periods=1).mean()

        steps = binned["step_bin"].values

        # Percentile envelope
        ax.fill_between(
            steps, binned["p10"].values, binned["p90"].values,
            color=CB_CYAN, alpha=0.25, label="10\u201390th pctl",
        )

        # Bold median line
        ax.plot(steps, binned["median"].values,
                color=CB_BLUE, linewidth=2, label="Median KL")

        # Threshold line
        ax.axhline(
            y=mean_kl_thresh, color=CB_RED, linestyle="--",
            linewidth=1.5, label=f"Threshold ({mean_kl_thresh})",
        )

        ax.set_title(f"q = {int(q)}")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 2e-1)
        ax.xaxis.set_major_formatter(x_formatter)
        if idx == 0:
            ax.set_ylabel("KL Divergence")
            ax.legend(loc="lower right", fontsize=8)
        ax.set_xlabel("Training Steps")

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
    
    df = df[(df["q"].isin(q_values)) & (df["method"].isin(["TEL-PPO", "PPO"])) & (df["ablation"] == "baseline")]
    if "experiment" in df.columns:
        df = df[df["experiment"] == "two_players"]

    if df.empty or "exploitability" not in df.columns:
        print("[plots] Warning: No exploitability data available")
        return None, None
    
    # Forward-fill exploitability
    df = forward_fill_exploitability(df)

    # Precompute convergence steps
    conv_steps_df = get_convergence_step(df)

    fig, axes = plt.subplots(1, len(q_values), figsize=FIGURE_SIZES["exploitability_dynamics"])
    if len(q_values) == 1:
        axes = [axes]

    for ax, q in zip(axes, q_values):
        q_df = df[df["q"] == q].sort_values("step")

        if q_df.empty:
            ax.set_title(f"q = {int(q)} (no data)")
            continue

        seeds = sorted(q_df["seed"].unique())

        # Per-seed thin lines with step drawstyle
        for seed in seeds:
            sdf = q_df[q_df["seed"] == seed].sort_values("step")
            valid = sdf[~sdf["exploitability_ffill"].isna()]
            if not valid.empty:
                ax.plot(
                    valid["step"], valid["exploitability_ffill"],
                    color="#1f77b4", alpha=0.3, linewidth=0.8,
                    drawstyle="steps-post",
                )

        # Bold mean across seeds with step drawstyle
        if len(seeds) > 1:
            agg = aggregate_seeds(q_df)
            if "exploitability_ffill" not in agg.columns:
                # exploitability_ffill wasn't aggregated; compute from exploitability_mean
                col = "exploitability_mean" if "exploitability_mean" in agg.columns else None
            else:
                col = "exploitability_ffill"
            if col is None and "exploitability_mean" in agg.columns:
                col = "exploitability_mean"
            if col and col in agg.columns:
                valid = agg[~agg[col].isna()]
                ax.plot(
                    valid["step"], valid[col],
                    color="#1f77b4", linewidth=2,
                    drawstyle="steps-post", label="Exploitability",
                )
        else:
            valid = q_df[~q_df["exploitability_ffill"].isna()]
            if not valid.empty:
                ax.plot(
                    valid["step"], valid["exploitability_ffill"],
                    color="#1f77b4", linewidth=2,
                    drawstyle="steps-post", label="Exploitability",
                )

        # Threshold line → bolder
        exploit_thresh = CONVERGENCE_CONFIG["exploit_threshold"]
        ax.axhline(
            y=exploit_thresh, color="red", linestyle="--",
            linewidth=2.5, alpha=0.8, label=f"Threshold ({exploit_thresh})",
        )

        # Convergence vertical lines per seed
        conv_label_added = False
        for seed in seeds:
            match = conv_steps_df[
                (conv_steps_df["q"] == q)
                & (conv_steps_df["seed"] == seed)
                & (conv_steps_df["ablation"] == "baseline")
            ]
            if "experiment" in conv_steps_df.columns:
                match = match[match["experiment"] == "two_players"]
            if not match.empty:
                cs = match["convergence_step"].iloc[0]
                if not np.isnan(cs):
                    label = "Convergence step" if not conv_label_added else None
                    ax.axvline(
                        x=cs, color="green", linestyle=":",
                        linewidth=1.2, alpha=0.6, label=label,
                    )
                    conv_label_added = True

        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Exploitability")
        ax.set_title(f"q = {int(q)}")
        ax.set_yscale("log")
        ax.set_ylim(0.01, 1)
        ax.legend(loc="best", fontsize=8)

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
    
    df = df[(df["q"].isin(q_values)) & (df["method"].isin(["TEL-PPO", "PPO"])) & (df["ablation"] == "baseline")]
    if "experiment" in df.columns:
        df = df[df["experiment"] == "two_players"]

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
        snapshot_fractions = [0.1, 0.5, 0.9]
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "beta_snapshots.png")
    
    df = df[(df["q"] == q) & (df["method"].isin(["TEL-PPO", "PPO"])) & (df["ablation"] == "baseline")]
    if "experiment" in df.columns:
        df = df[df["experiment"] == "two_players"]

    if df.empty or "alpha_mean" not in df.columns:
        print("[plots] Warning: No alpha/beta data available for snapshots")
        return None, None
    
    df = df.sort_values("step")
    max_step = df["step"].max()

    # Compute e* for this q and the effort scale factor
    e_theory = e_star(q, **THEORY_PARAMS)
    e_max = 250.0  # effort upper bound from environment config

    fig, axes = plt.subplots(1, len(snapshot_fractions), figsize=FIGURE_SIZES["beta_snapshots"])

    x_norm = np.linspace(0.001, 0.999, 200)
    x_effort = x_norm * e_max  # Convert normalized to effort scale

    y_max_global = 0.0  # for unifying y-axis

    pdf_data = []  # store (ax, frac, y_vals, ...) for second pass

    for ax, frac in zip(axes, snapshot_fractions):
        target_step = int(max_step * frac)
        # Find closest step
        closest_idx = (df["step"] - target_step).abs().idxmin()
        row = df.loc[closest_idx]

        alpha = row["alpha_mean"]
        beta_val = row["beta_mean"]

        if np.isnan(alpha) or np.isnan(beta_val):
            ax.set_title(f"Step {target_step} ({frac*100:.0f}%) - No data")
            pdf_data.append(None)
            continue

        kappa = alpha + beta_val

        # Plot Beta PDF on effort scale (divide density by e_max for proper scaling)
        y = beta_dist.pdf(x_norm, alpha, beta_val) / e_max
        ax.plot(x_effort, y, color=AGENT_COLORS["agent1"], linewidth=2,
                label="Both agents (symmetric)")
        ax.fill_between(x_effort, y, alpha=0.3, color=AGENT_COLORS["agent1"])

        # Mark policy mean in effort space
        mean_norm = alpha / kappa
        mean_effort = mean_norm * e_max
        ax.axvline(x=mean_effort, color=AGENT_COLORS["agent2"], linestyle="--",
                    linewidth=2, label=f"Mean={mean_effort:.1f}")

        # e* vertical line (green)
        ax.axvline(x=e_theory, color="green", linestyle="-.",
                    linewidth=2, label=f"$e^*$={e_theory:.1f}")

        # κ annotation
        ax.text(
            0.03, 0.95, f"κ={kappa:.1f}",
            transform=ax.transAxes, fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.7),
        )

        ax.set_xlabel("Effort")
        ax.set_ylabel("Density")
        ax.set_title(f"Step {int(row['step'])} ({frac*100:.0f}%)\nα={alpha:.1f}, β={beta_val:.1f}")
        ax.legend(loc="best", fontsize=7)

        y_max_global = max(y_max_global, np.max(y))

    # Unify y-axis across all panels
    for ax in axes:
        ax.set_ylim(0, y_max_global * 1.1)

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
    if "experiment" in df.columns:
        df = df[df["experiment"] == "two_players"]

    if df.empty:
        print("[plots] Warning: No data for ablation comparison")
        return None, None
    
    # Keep only the 3 key ablations for the main paper figure
    df = df[df["ablation"].isin(["baseline", "no_cheap_gate", "no_exploitability"])]

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


def plot_hyperparam_sensitivity(
    df: pd.DataFrame = None,
    q_values: List[float] = None,
    output_path: str = None,
    save_data: bool = True,
) -> Tuple[plt.Figure, str]:
    """
    Plot hyperparameter sensitivity sweeps (appendix figure).

    Top row: epsilon sweep (eps_001..eps_020 + baseline).
    Bottom row: patience sweep (pat_01..pat_10 + baseline).
    Columns: q values.
    """
    setup_matplotlib_style()
    ensure_output_dirs()

    if df is None:
        df = load_all_convergence_data()
    if q_values is None:
        q_values = Q_VALUES
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "hyperparam_sensitivity.png")

    # Filter to two_players PPO
    df = df[
        (df["q"].isin(q_values))
        & (df["method"].isin(["TEL-PPO", "PPO"]))
    ]
    if "experiment" in df.columns:
        df = df[df["experiment"] == "two_players"]

    # Exclude non-hyperparam ablations
    exclude = {"k5e4_wh8_wl3", "q25_seed68", "wh8_wl4", "baseline_v2",
               "no_cheap_gate", "no_exploitability", "no_entropy"}
    df = df[~df["ablation"].isin(exclude)]

    if df.empty:
        print("[plots] Warning: No data for hyperparam_sensitivity")
        return None, None

    # Row definitions
    eps_ablations = ["eps_001", "eps_003", "eps_010", "eps_020"]
    eps_labels = {
        "eps_001": "ε=0.01", "eps_003": "ε=0.03",
        "eps_010": "ε=0.10", "eps_020": "ε=0.20",
        "baseline": "baseline (ε=0.05)",
    }
    pat_ablations = ["pat_01", "pat_03", "pat_10"]
    pat_labels = {
        "pat_01": "p=1", "pat_03": "p=3", "pat_10": "p=10",
        "baseline": "baseline (p=5)",
    }

    rows = [
        ("Clipping ε", eps_ablations, eps_labels, plt.cm.Blues),
        ("Patience p", pat_ablations, pat_labels, plt.cm.Oranges),
    ]

    n_cols = len(q_values)
    fig, axes = plt.subplots(2, n_cols, figsize=FIGURE_SIZES["hyperparam_sensitivity"],
                             squeeze=False)

    x_formatter = ticker.FuncFormatter(
        lambda x, p: f"{x/1e6:.1f}M" if x >= 1e6
        else f"{x/1e3:.0f}k" if x >= 1e3
        else f"{x:.0f}"
    )

    for row_idx, (row_label, sweep_ablations, label_map, cmap) in enumerate(rows):
        # Colors: sequential for sweep, black for baseline
        n_sweep = len(sweep_ablations)
        cmap_positions = np.linspace(0.3, 0.85, n_sweep)
        sweep_colors = {abl: cmap(pos) for abl, pos in zip(sweep_ablations, cmap_positions)}

        plot_order = sweep_ablations + ["baseline"]

        for col_idx, q in enumerate(q_values):
            ax = axes[row_idx, col_idx]
            e_theory = e_star(q, **THEORY_PARAMS)

            # Theory line
            ax.axhline(y=e_theory, color="black", linestyle="--",
                       linewidth=1.5, label="Theory $e^*$", zorder=1)

            for ablation in plot_order:
                abl_df = df[(df["q"] == q) & (df["ablation"] == ablation)].sort_values("step")
                if abl_df.empty:
                    continue

                color = "black" if ablation == "baseline" else sweep_colors.get(ablation, "gray")
                lw = 1.5 if ablation == "baseline" else 2.0
                label = label_map.get(ablation, ablation)

                seeds = sorted(abl_df["seed"].unique())
                has_multi = len(seeds) > 1

                if has_multi:
                    # Per-seed thin traces
                    for seed in seeds:
                        sdf = abl_df[abl_df["seed"] == seed].sort_values("step")
                        ax.plot(sdf["step"], sdf["effort_mean"],
                                color=color, alpha=0.2, linewidth=0.7, zorder=1)

                    # Aggregated mean + CI
                    agg = aggregate_seeds(abl_df)
                    if "effort_mean_mean" in agg.columns:
                        steps = agg["step"].values
                        mean_vals = agg["effort_mean_mean"].values
                        ci_vals = agg.get("effort_mean_ci95", pd.Series(np.zeros(len(agg)))).values
                        ax.plot(steps, mean_vals, color=color, linewidth=lw,
                                label=label, zorder=3)
                        ax.fill_between(steps, mean_vals - ci_vals, mean_vals + ci_vals,
                                        color=color, alpha=0.12, zorder=2)
                    else:
                        ax.plot(abl_df["step"], abl_df["effort_mean"],
                                color=color, linewidth=lw, label=label, zorder=3)
                else:
                    ax.plot(abl_df["step"], abl_df["effort_mean"],
                            color=color, linewidth=lw, label=label, zorder=3)

            # Axis formatting
            ax.xaxis.set_major_formatter(x_formatter)
            if col_idx == 0:
                ax.set_ylabel("Effort")
            if row_idx == 0:
                ax.set_title(f"q = {int(q)}")
            if row_idx == 1:
                ax.set_xlabel("Training Steps")

            # Row label on leftmost panel
            if col_idx == 0:
                ax.annotate(
                    row_label, xy=(0, 0.5),
                    xytext=(-50, 0), textcoords="offset points",
                    xycoords="axes fraction", ha="right", va="center",
                    fontsize=FONT_SIZES["axis_label"], rotation=90,
                )

            # Legend on rightmost panel of this row
            if col_idx == n_cols - 1:
                ax.legend(loc="best", fontsize=FONT_SIZES["legend"])

    plt.tight_layout()
    plt.subplots_adjust(left=0.12)

    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')

    if save_data:
        data_path = os.path.join(DATA_DIR, "hyperparam_sensitivity.csv")
        out_cols = ["step", "method", "q", "seed", "ablation", "effort_mean", "theoretical_effort"]
        df_out = df[[c for c in out_cols if c in df.columns]].copy()
        df_out.to_csv(data_path, index=False)
        print(f"[plots] Saved data to {data_path}")

    print(f"[plots] Saved figure to {output_path}")
    return fig, output_path


def plot_distance_to_equilibrium(
    df: pd.DataFrame = None,
    q_values: List[float] = None,
    output_path: str = None,
    save_data: bool = True,
) -> Tuple[plt.Figure, str]:
    """
    Plot distance to equilibrium |ē − e*| over training (Figure 2b).

    Shows convergence speed as effort error shrinks toward zero.
    Multi-seed aggregation with 95% CI bands.
    """
    setup_matplotlib_style()
    ensure_output_dirs()

    if df is None:
        df = load_all_convergence_data()
    if q_values is None:
        q_values = Q_VALUES
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "distance_to_equilibrium.png")

    # Filter to two_players TEL-PPO baseline
    df = df[
        (df["q"].isin(q_values))
        & (df["method"].isin(["TEL-PPO", "PPO"]))
        & (df["ablation"] == "baseline")
    ]
    if "experiment" in df.columns:
        df = df[df["experiment"] == "two_players"]

    if df.empty:
        print("[plots] Warning: No data for distance_to_equilibrium")
        return None, None

    # Compute effort error
    df = compute_effort_error(df)

    # Precompute convergence steps
    conv_steps_df = get_convergence_step(df)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    q_colors = {25.0: "#1f77b4", 40.0: "#ff7f0e", 55.0: "#2ca02c"}

    for q in q_values:
        q_df = df[df["q"] == q]
        if q_df.empty:
            continue

        color = q_colors.get(q, "gray")

        # Aggregate across seeds
        has_multi = q_df["seed"].nunique() > 1
        if has_multi:
            agg = aggregate_seeds(q_df)
            if "effort_error_mean" not in agg.columns:
                agg["effort_error_mean"] = np.abs(
                    agg["effort_mean_mean"] - agg["theoretical_effort"]
                )
                agg["effort_error_ci95"] = agg.get("effort_mean_ci95", 0)

            steps = agg["step"].values
            err_mean = agg["effort_error_mean"].values
            err_ci = agg.get("effort_error_ci95", pd.Series([0] * len(agg))).values

            ax.plot(steps, err_mean, color=color, linewidth=2, label=f"q={int(q)}")
            ax.fill_between(
                steps,
                np.maximum(err_mean - err_ci, 0),
                err_mean + err_ci,
                color=color,
                alpha=0.12,
            )
        else:
            q_df = q_df.sort_values("step")
            ax.plot(
                q_df["step"],
                q_df["effort_error"],
                color=color,
                linewidth=2,
                label=f"q={int(q)}",
            )

        # Convergence vertical line (mean across seeds for this q)
        q_conv = conv_steps_df[conv_steps_df["q"] == q]
        if "experiment" in conv_steps_df.columns:
            q_conv = q_conv[q_conv["experiment"] == "two_players"]
        q_conv = q_conv[q_conv["ablation"] == "baseline"]
        mean_conv = q_conv["convergence_step"].dropna().mean()
        if not np.isnan(mean_conv):
            ax.axvline(
                x=mean_conv, color=color, linestyle=":",
                linewidth=1.2, alpha=0.7,
                label=f"Conv. step q={int(q)}",
            )

    # ε threshold horizontal line
    effort_delta = CONVERGENCE_CONFIG["effort_delta"]
    ax.axhline(
        y=effort_delta, color="gray", linestyle="--",
        linewidth=1.5, alpha=0.7, label=f"ε={effort_delta}",
    )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("|ē − e*|")
    ax.set_title("Distance to the Nash Equilibrium Across Noise Levels")
    ax.legend(loc="best")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.1)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, p: f"{x/1e6:.1f}M"
            if x >= 1e6
            else f"{x/1e3:.0f}k"
            if x >= 1e3
            else f"{x:.0f}"
        )
    )

    plt.tight_layout()

    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

    if save_data:
        data_path = os.path.join(DATA_DIR, "distance_to_equilibrium.csv")
        cols = ["step", "method", "q", "seed", "ablation", "effort_mean", "theoretical_effort"]
        if "effort_error" in df.columns:
            cols.append("effort_error")
        df[cols].to_csv(data_path, index=False)

    print(f"[plots] Saved figure to {output_path}")
    return fig, output_path


def plot_effort_drift(
    df: pd.DataFrame = None,
    q_values: List[float] = None,
    output_path: str = None,
    save_data: bool = True,
) -> Tuple[plt.Figure, str]:
    """
    Plot effort drift over training (Figure 3b).

    Shows drift_effort metric which measures episode-to-episode policy instability.
    """
    setup_matplotlib_style()
    ensure_output_dirs()

    if df is None:
        df = load_all_convergence_data()
    if q_values is None:
        q_values = Q_VALUES
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "effort_drift.png")

    # Filter to TEL-PPO with drift data
    df = df[
        (df["q"].isin(q_values))
        & (df["method"].isin(["TEL-PPO", "PPO"]))
        & (df["ablation"] == "baseline")
    ]
    if "experiment" in df.columns:
        df = df[df["experiment"] == "two_players"]

    if df.empty or "drift_effort" not in df.columns or df["drift_effort"].isna().all():
        print("[plots] Warning: No drift data available for effort_drift plot")
        return None, None

    fig, axes = plt.subplots(1, len(q_values), figsize=(12, 4))
    if len(q_values) == 1:
        axes = [axes]

    for ax, q in zip(axes, q_values):
        q_df = df[df["q"] == q].sort_values("step")

        if q_df.empty:
            ax.set_title(f"q = {q} (no data)")
            continue

        valid = q_df[~q_df["drift_effort"].isna()]
        if valid.empty:
            ax.set_title(f"q = {q} (no drift data)")
            continue

        # Multi-seed: plot individual traces with transparency
        for seed, seed_df in valid.groupby("seed"):
            seed_df = seed_df.sort_values("step")
            ax.plot(
                seed_df["step"],
                seed_df["drift_effort"],
                alpha=0.4,
                linewidth=1,
                color="#1f77b4",
            )

        # Aggregate mean
        has_multi = valid["seed"].nunique() > 1
        if has_multi:
            agg = aggregate_seeds(valid)
            if "drift_effort_mean" in agg.columns:
                ax.plot(
                    agg["step"],
                    agg["drift_effort_mean"],
                    color="#ff7f0e",
                    linewidth=2,
                    label="Mean",
                )

        # Threshold line → bold red with value annotation
        drift_thresh = CHEAP_GATE_CONFIG["drift_effort_thresh"]
        ax.axhline(
            y=drift_thresh, color="red", linestyle="--",
            linewidth=2.5, alpha=0.8, label=f"Threshold ({drift_thresh})",
        )
        # Annotate the threshold value
        ax.text(
            ax.get_xlim()[1], drift_thresh, f" {drift_thresh}",
            color="red", fontsize=8, va="bottom",
        )

        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Effort Drift")
        ax.set_title(f"q = {int(q)}")
        ax.legend(loc="best")

    # Unify y-axis across all panels
    _unify_ylim(axes)

    plt.tight_layout()

    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

    if save_data:
        data_path = os.path.join(DATA_DIR, "effort_drift.csv")
        cols = ["step", "method", "q", "seed", "ablation", "drift_effort"]
        df[[c for c in cols if c in df.columns]].to_csv(data_path, index=False)

    print(f"[plots] Saved figure to {output_path}")
    return fig, output_path


def plot_equilibrium_recovery_dotplot(
    df: pd.DataFrame = None,
    output_path: str = None,
    save_data: bool = True,
) -> Tuple[plt.Figure, str]:
    """
    Plot equilibrium recovery dot plot across scenarios (Figure 6).

    Shows learned equilibrium effort vs theoretical for each experiment type.
    x-axis: scenario, y-axis: effort, dots for learned, lines for theoretical.
    """
    setup_matplotlib_style()
    ensure_output_dirs()

    if df is None:
        df = load_all_convergence_data()
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "equilibrium_recovery_dotplot.png")

    # Only use baseline TEL-PPO runs
    ppo_df = df[
        (df["method"].isin(["TEL-PPO", "PPO"]))
        & (df["ablation"] == "baseline")
    ]

    if ppo_df.empty:
        print("[plots] Warning: No data for equilibrium recovery dotplot")
        return None, None

    # Get final values per run
    final = get_final_values(ppo_df)
    if "experiment" not in final.columns:
        # Merge experiment from original df
        exp_map = ppo_df.groupby(["method", "q", "seed", "ablation"])["experiment"].first()
        final = final.merge(
            exp_map.reset_index(), on=["method", "q", "seed", "ablation"], how="left"
        )

    experiments = ["two_players", "three_players", "different_cost", "different_ability"]
    exp_labels = {
        "two_players": "Two-Player\nSymmetric",
        "three_players": "Three-Player",
        "different_cost": "Heterogeneous\nCost",
        "different_ability": "Heterogeneous\nAbility",
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x_pos = 0
    x_ticks = []
    x_labels = []
    # Track whether per-agent legend entries have been added
    _legend_added = {"agent1": False, "agent2": False, "single": False, "mean": False}

    for exp in experiments:
        exp_final = final[final["experiment"] == exp] if "experiment" in final.columns else pd.DataFrame()
        if exp_final.empty:
            continue

        q_vals = sorted(exp_final["q"].unique())
        is_heterogeneous_cost = (exp == "different_cost")

        for q in q_vals:
            q_final = exp_final[exp_final["q"] == q]
            if q_final.empty:
                continue

            rng = np.random.RandomState(42 + int(x_pos * 100))

            if is_heterogeneous_cost:
                # --- Per-agent markers with separate theory lines ---
                for agent_key, effort_col, theory_col, agent_label in [
                    ("agent1", "agent1_effort", "theoretical_effort1", "Agent 1 (low-cost)"),
                    ("agent2", "agent2_effort", "theoretical_effort2", "Agent 2 (high-cost)"),
                ]:
                    color = AGENT_COLORS[agent_key]
                    marker = AGENT_MARKERS[agent_key]

                    # Per-agent theory line (if column exists)
                    if theory_col in q_final.columns:
                        e_theory_agent = q_final[theory_col].dropna()
                        if not e_theory_agent.empty:
                            ax.hlines(
                                e_theory_agent.iloc[0],
                                x_pos - 0.3, x_pos + 0.3,
                                colors=color, linestyles="--", linewidth=2, zorder=2,
                            )

                    # Per-seed scatter
                    if effort_col in q_final.columns:
                        efforts = q_final[effort_col].values
                        jitter = rng.uniform(-0.12, 0.12, len(efforts))
                        ax.scatter(
                            x_pos + jitter, efforts,
                            color=color, marker=marker, s=60, zorder=3,
                            alpha=0.8, edgecolors="white", linewidth=0.5,
                            label=agent_label if not _legend_added[agent_key] else None,
                        )
                        _legend_added[agent_key] = True
            else:
                # --- Single-marker behavior (original) ---
                e_theory = q_final["theoretical_effort"].iloc[0]
                ax.hlines(
                    e_theory,
                    x_pos - 0.3, x_pos + 0.3,
                    colors=THEORY_LINE_COLOR, linestyles="--", linewidth=3, zorder=2,
                )

                efforts = q_final["effort_mean"].values
                jitter = rng.uniform(-0.15, 0.15, len(efforts))
                ax.scatter(
                    x_pos + jitter, efforts,
                    color="#ff7f0e", s=60, zorder=3, alpha=0.8,
                    edgecolors="white", linewidth=0.5,
                    label="Per-seed" if not _legend_added["single"] else None,
                )
                _legend_added["single"] = True

                # Mean marker
                mean_effort = efforts.mean()
                ax.scatter(
                    x_pos, mean_effort,
                    color="#d62728", marker="D", s=100, zorder=4,
                    edgecolors="black", linewidth=1,
                    label="Seed mean" if not _legend_added["mean"] else None,
                )
                _legend_added["mean"] = True

            x_ticks.append(x_pos)
            x_labels.append(f"q={int(q)}")
            x_pos += 1

        # Add separator between experiments
        if x_pos > 0:
            x_pos += 0.5

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=FONT_SIZES["tick_label"])
    ax.set_ylabel("Equilibrium Effort")
    ax.set_title("Equilibrium Recovery Across Scenarios")

    # Add experiment group labels
    group_starts = []
    x_pos_track = 0
    for exp in experiments:
        exp_final = final[final["experiment"] == exp] if "experiment" in final.columns else pd.DataFrame()
        if exp_final.empty:
            continue
        n_q = len(exp_final["q"].unique())
        group_center = x_pos_track + (n_q - 1) / 2.0
        group_starts.append((group_center, exp_labels.get(exp, exp)))
        x_pos_track += n_q + 0.5

    for center, label in group_starts:
        ax.annotate(
            label,
            xy=(center, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -35),
            textcoords="offset points",
            ha="center", va="top",
            fontsize=FONT_SIZES["annotation"],
            fontweight="bold",
        )

    # Legend
    legend_elements = [
        Line2D([0], [0], color=THEORY_LINE_COLOR, linestyle="--", linewidth=3, label="Theory e*"),
    ]
    if _legend_added["single"]:
        legend_elements.append(Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#ff7f0e",
            markersize=8, label="Per-seed",
        ))
    if _legend_added["mean"]:
        legend_elements.append(Line2D(
            [0], [0], marker="D", color="w", markerfacecolor="#d62728",
            markeredgecolor="black", markersize=8, label="Seed mean",
        ))
    if _legend_added["agent1"]:
        legend_elements.append(Line2D(
            [0], [0], marker=AGENT_MARKERS["agent1"], color="w",
            markerfacecolor=AGENT_COLORS["agent1"], markersize=8,
            label="Agent 1 (low-cost)",
        ))
    if _legend_added["agent2"]:
        legend_elements.append(Line2D(
            [0], [0], marker=AGENT_MARKERS["agent2"], color="w",
            markerfacecolor=AGENT_COLORS["agent2"], markersize=8,
            label="Agent 2 (high-cost)",
        ))
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

    if save_data:
        # Compute per-seed relative error |e_learned - e*| / e*
        save_df = final.copy()
        save_df["relative_error"] = np.where(
            save_df["theoretical_effort"] != 0,
            np.abs(save_df["effort_mean"] - save_df["theoretical_effort"]) / np.abs(save_df["theoretical_effort"]),
            np.nan,
        )
        data_path = os.path.join(DATA_DIR, "equilibrium_recovery_dotplot.csv")
        cols_to_save = ["method", "q", "seed", "ablation", "effort_mean",
                        "theoretical_effort", "agent1_effort", "agent2_effort",
                        "relative_error"]
        for extra_col in ["theoretical_effort1", "theoretical_effort2"]:
            if extra_col in save_df.columns:
                cols_to_save.append(extra_col)
        if "experiment" in save_df.columns:
            cols_to_save.insert(0, "experiment")
        save_df[[c for c in cols_to_save if c in save_df.columns]].to_csv(data_path, index=False)
        # Print mean relative error for caption reference
        mean_rel_err = save_df["relative_error"].mean()
        print(f"[plots] Mean relative error for dotplot: {mean_rel_err:.4f}")

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

    # Hyperparameter sensitivity (appendix)
    fig, path = plot_hyperparam_sensitivity(df, q_values)
    if path:
        results["hyperparam_sensitivity"] = path

    # Distance to equilibrium (Fig 2b)
    fig, path = plot_distance_to_equilibrium(df, q_values)
    if path:
        results["distance_to_equilibrium"] = path

    # Effort drift (Fig 3b)
    fig, path = plot_effort_drift(df, q_values)
    if path:
        results["effort_drift"] = path

    # Equilibrium recovery dotplot (Fig 6)
    fig, path = plot_equilibrium_recovery_dotplot(df)
    if path:
        results["equilibrium_recovery_dotplot"] = path

    plt.close('all')  # Clean up

    return results


if __name__ == "__main__":
    print("Generating all figures...")
    results = generate_all_figures()
    print("\nGenerated figures:")
    for name, path in results.items():
        print(f"  {name}: {path}")
