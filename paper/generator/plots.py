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
    ABLATION_LABELS,
    ABLATION_LINEWIDTHS,
    FIGURE_SIZES,
    FONT_SIZES,
    Q_VALUES,
    e_star,
    format_q,
    THEORY_PARAMS,
    AGENT_COLORS,
    AGENT_MARKERS,
    WEIGHT_VARIANT_LABELS,
    THEORY_LINE_COLOR,
    THEORY_LINE_WIDTH,
    SHADE_ALPHA,
    CONV_VLINE_COLOR,
    CONV_VLINE_LINESTYLE,
    CONV_VLINE_LINEWIDTH,
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


def _baseline_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to baseline weight variant (exclude wh8_wl4 etc.)."""
    if "weight_variant" in df.columns:
        return df[df["weight_variant"] == "baseline"]
    return df


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
    if "weight_variant" in df.columns:
        df = df[df["weight_variant"].isin(weight_variants)]
    else:
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
        wv_col = "weight_variant" if "weight_variant" in df.columns else "ablation"
        var_df = df[df[wv_col] == variant]

        for col_idx, q in enumerate(q_values):
            ax = axes[row_idx, col_idx]
            q_df = var_df[var_df["q"] == q]

            if q_df.empty:
                ax.set_title(f"Noise Level {format_q(q)} (no data)")
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

                # Individual seed traces (light)
                if has_multi_seed:
                    for seed in seeds:
                        seed_df = method_df[method_df["seed"] == seed].sort_values("step")
                        ax.plot(
                            seed_df["step"], seed_df[col_name],
                            color=color, linestyle=ls, alpha=0.10,
                            linewidth=0.6, zorder=1,
                        )

                # Aggregate mean + CI
                agg_col_mean = f"{col_name}_mean"
                agg_col_ci = f"{col_name}_ci95"

                if has_multi_seed:
                    agg_df = aggregate_seeds(method_df)
                    if agg_col_mean in agg_df.columns:
                        steps = agg_df["step"].values
                        sample_effort_mean = agg_df[agg_col_mean].values
                        effort_ci = agg_df[agg_col_ci].values if agg_col_ci in agg_df.columns else np.zeros_like(sample_effort_mean)

                        ax.plot(
                            steps, sample_effort_mean, color=color, linestyle=ls,
                            linewidth=2, label=label_base, zorder=3,
                        )
                        ax.fill_between(
                            steps, sample_effort_mean - effort_ci, sample_effort_mean + effort_ci,
                            color=color, alpha=SHADE_ALPHA, zorder=2,
                        )
                else:
                    # Single seed: plot directly
                    single = method_df.sort_values("step")
                    ax.plot(
                        single["step"], single[col_name],
                        color=color, linestyle=ls, linewidth=2,
                        label=label_base, zorder=3,
                    )

            # Convergence vertical line (median across seeds)
            mask = (
                (conv_steps_df["q"] == q)
                & (conv_steps_df["ablation"] == variant)
                & (conv_steps_df["method"].isin(["TEL-PPO", "PPO"]))
            )
            if "experiment" in conv_steps_df.columns:
                mask = mask & (conv_steps_df["experiment"] == "two_players")
            conv_vals = conv_steps_df.loc[mask, "convergence_step"].dropna()
            if not conv_vals.empty:
                median_conv = conv_vals.median()
                ax.axvline(
                    x=median_conv, color=CONV_VLINE_COLOR,
                    linestyle=CONV_VLINE_LINESTYLE,
                    linewidth=CONV_VLINE_LINEWIDTH,
                    label="Convergence step", zorder=4,
                )

            # Axis formatting
            if row_idx == n_rows - 1:
                ax.set_xlabel("Training Steps")
            ax.set_ylabel("Effort")

            # Title: "Noise Level q = {q}" on top row only
            if row_idx == 0:
                ax.set_title(f"Noise Level {format_q(q)}")

            # Row label on the leftmost column (bold, rotated)
            if col_idx == 0:
                variant_label = WEIGHT_VARIANT_LABELS.get(variant, variant)
                ax.annotate(
                    variant_label, xy=(0, 0.5),
                    xytext=(-50, 0), textcoords="offset points",
                    xycoords="axes fraction", ha="right", va="center",
                    fontsize=FONT_SIZES["axis_label"],
                    fontweight="bold", rotation=90,
                )

            # Legend only on top-left panel (deduplicate labels)
            if row_idx == 0 and col_idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                seen = set()
                deduped_h, deduped_l = [], []
                for h, l in zip(handles, labels):
                    if l not in seen:
                        seen.add(l)
                        deduped_h.append(h)
                        deduped_l.append(l)
                ax.legend(deduped_h, deduped_l, loc="upper left",
                          fontsize=FONT_SIZES["legend"])

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
        out_cols = ["step", "method", "q", "seed", "ablation", "policy_mean_effort",
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
    df = _baseline_only(df)
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
            ax.set_title(f"{format_q(q)} (no data)")
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
            color=CB_CYAN, alpha=SHADE_ALPHA, label="10\u201390% interval",
        )

        # Bold median line
        ax.plot(steps, binned["median"].values,
                color=CB_BLUE, linewidth=2, label="Median KL")

        # Threshold line
        ax.axhline(
            y=mean_kl_thresh, color=CB_RED, linestyle="--",
            linewidth=2.5, label=f"Reference threshold ({mean_kl_thresh})",
        )

        ax.set_title(format_q(q))
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
    df = _baseline_only(df)
    if "experiment" in df.columns:
        df = df[df["experiment"] == "two_players"]

    if df.empty or "exploitability" not in df.columns:
        print("[plots] Warning: No exploitability data available")
        return None, None
    
    # Forward-fill exploitability
    df = forward_fill_exploitability(df)

    # Precompute cheap gate steps and Nash convergence steps
    from .extract import get_cheap_gate_step, get_nash_convergence_step
    gate_steps_df = get_cheap_gate_step(df)
    nash_steps_df = get_nash_convergence_step(df)

    fig, axes = plt.subplots(1, len(q_values), figsize=FIGURE_SIZES["exploitability_dynamics"])
    if len(q_values) == 1:
        axes = [axes]

    for ax, q in zip(axes, q_values):
        q_df = df[df["q"] == q].sort_values("step")

        if q_df.empty:
            ax.set_title(f"{format_q(q)} (no data)")
            continue

        seeds = sorted(q_df["seed"].unique())

        # Smoothing window for rolling mean
        smooth_window = 15

        # Per-seed thin lines (smoothed)
        for seed in seeds:
            sdf = q_df[q_df["seed"] == seed].sort_values("step")
            valid = sdf[~sdf["exploitability_ffill"].isna()]
            if not valid.empty:
                smoothed = valid["exploitability_ffill"].rolling(
                    window=smooth_window, min_periods=1, center=True,
                ).mean()
                ax.plot(
                    valid["step"].values, smoothed.values,
                    color="#1f77b4", alpha=0.3, linewidth=0.8,
                )

        # Bold mean across seeds (smoothed)
        if len(seeds) > 1:
            agg = aggregate_seeds(q_df)
            if "exploitability_ffill" not in agg.columns:
                col = "exploitability_mean" if "exploitability_mean" in agg.columns else None
            else:
                col = "exploitability_ffill"
            if col is None and "exploitability_mean" in agg.columns:
                col = "exploitability_mean"
            if col and col in agg.columns:
                valid = agg[~agg[col].isna()]
                smoothed = valid[col].rolling(
                    window=smooth_window, min_periods=1, center=True,
                ).mean()
                ax.plot(
                    valid["step"].values, smoothed.values,
                    color="#1f77b4", linewidth=2, label="Exploitability",
                )
        else:
            valid = q_df[~q_df["exploitability_ffill"].isna()]
            if not valid.empty:
                smoothed = valid["exploitability_ffill"].rolling(
                    window=smooth_window, min_periods=1, center=True,
                ).mean()
                ax.plot(
                    valid["step"].values, smoothed.values,
                    color="#1f77b4", linewidth=2, label="Exploitability",
                )

        # Threshold line → bolder
        exploit_thresh = CONVERGENCE_CONFIG["exploit_threshold"]
        ax.axhline(
            y=exploit_thresh, color="red", linestyle="--",
            linewidth=2.5, alpha=0.8, label=f"Tolerance threshold ({exploit_thresh})",
        )

        # Cheap gate: median across seeds → single orange line
        gate_match = gate_steps_df[
            (gate_steps_df["q"] == q) & (gate_steps_df["ablation"] == "baseline")
        ]
        if "experiment" in gate_steps_df.columns:
            gate_match = gate_match[gate_match["experiment"] == "two_players"]
        gate_vals = gate_match["cheap_gate_step"].dropna()
        if not gate_vals.empty:
            ax.axvline(
                x=gate_vals.median(), color="orange", linestyle=":",
                linewidth=1.5, alpha=0.8, label="Stability screening passed",
            )

        # Nash convergence: max across seeds → single green line (conservative: all seeds converged)
        nash_match = nash_steps_df[
            (nash_steps_df["q"] == q) & (nash_steps_df["ablation"] == "baseline")
        ]
        if "experiment" in nash_steps_df.columns:
            nash_match = nash_match[nash_match["experiment"] == "two_players"]
        nash_vals = nash_match["nash_step"].dropna()
        if not nash_vals.empty:
            ax.axvline(
                x=nash_vals.max(), color="green", linestyle="-.",
                linewidth=1.5, alpha=0.8, label="Approx. Nash verified",
            )

        ax.set_xlabel("Training Steps")
        if ax == axes[0]:
            ax.set_ylabel("Exploitability")
        ax.set_title(format_q(q))
        ax.set_yscale("log")
        ax.set_ylim(0.005, 2)

    # Single shared legend at the top — collect from all panels
    all_handles, all_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)
    # Deduplicate while preserving order
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(all_handles, all_labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    fig.legend(
        unique_handles, unique_labels,
        loc="upper center", ncol=len(unique_labels),
        fontsize=8, frameon=True,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
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


def plot_exploitability_q25(
    df: pd.DataFrame = None,
    output_path: str = None,
) -> Tuple[plt.Figure, str]:
    """
    Plot exploitability for the excluded low-noise case (q=25) — Figure 6b.

    Single-panel figure matching the style of Figure 6a.
    """
    setup_matplotlib_style()
    ensure_output_dirs()

    if df is None:
        df = load_all_convergence_data()
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "exploitability_q25.png")

    q = 25.0
    df = df[
        (df["q"] == q)
        & (df["method"].isin(["TEL-PPO", "PPO"]))
        & (df["ablation"] == "baseline")
    ]
    df = _baseline_only(df)
    if "experiment" in df.columns:
        df = df[df["experiment"] == "two_players"]

    if df.empty or "exploitability" not in df.columns:
        print("[plots] Warning: No exploitability data for q=25")
        return None, None

    df = forward_fill_exploitability(df)

    from .extract import get_cheap_gate_step, get_nash_convergence_step
    gate_steps_df = get_cheap_gate_step(df)
    nash_steps_df = get_nash_convergence_step(df)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    seeds = sorted(df["seed"].unique())
    smooth_window = 15

    for seed in seeds:
        sdf = df[df["seed"] == seed].sort_values("step")
        valid = sdf[~sdf["exploitability_ffill"].isna()]
        if not valid.empty:
            smoothed = valid["exploitability_ffill"].rolling(
                window=smooth_window, min_periods=1, center=True,
            ).mean()
            ax.plot(valid["step"].values, smoothed.values,
                    color="#1f77b4", alpha=0.3, linewidth=0.8)

    if len(seeds) > 1:
        agg = aggregate_seeds(df)
        col = "exploitability_ffill" if "exploitability_ffill" in agg.columns else "exploitability_mean"
        if col in agg.columns:
            valid = agg[~agg[col].isna()]
            smoothed = valid[col].rolling(
                window=smooth_window, min_periods=1, center=True,
            ).mean()
            ax.plot(valid["step"].values, smoothed.values,
                    color="#1f77b4", linewidth=2, label="Exploitability")
    else:
        valid = df[~df["exploitability_ffill"].isna()].sort_values("step")
        if not valid.empty:
            smoothed = valid["exploitability_ffill"].rolling(
                window=smooth_window, min_periods=1, center=True,
            ).mean()
            ax.plot(valid["step"].values, smoothed.values,
                    color="#1f77b4", linewidth=2, label="Exploitability")

    exploit_thresh = CONVERGENCE_CONFIG["exploit_threshold"]
    ax.axhline(y=exploit_thresh, color="red", linestyle="--",
               linewidth=2.5, alpha=0.8, label=f"Tolerance threshold ({exploit_thresh})")

    gate_match = gate_steps_df[gate_steps_df["ablation"] == "baseline"]
    if "experiment" in gate_steps_df.columns:
        gate_match = gate_match[gate_match["experiment"] == "two_players"]
    gate_vals = gate_match["cheap_gate_step"].dropna()
    if not gate_vals.empty:
        ax.axvline(x=gate_vals.median(), color="orange", linestyle=":",
                   linewidth=1.5, alpha=0.8, label="Stability screening passed")

    nash_match = nash_steps_df[nash_steps_df["ablation"] == "baseline"]
    if "experiment" in nash_steps_df.columns:
        nash_match = nash_match[nash_match["experiment"] == "two_players"]
    nash_vals = nash_match["nash_step"].dropna()
    if not nash_vals.empty:
        ax.axvline(x=nash_vals.max(), color="green", linestyle="-.",
                   linewidth=1.5, alpha=0.8, label="Approx. Nash verified")

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Exploitability")
    ax.set_title(f"{format_q(q)} (excluded low-noise case)")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 2)
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()

    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

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
    df = _baseline_only(df)
    if "experiment" in df.columns:
        df = df[df["experiment"] == "two_players"]

    if df.empty or "alpha_mean" not in df.columns:
        print("[plots] Warning: No alpha/beta data available")
        return None, None

    x_formatter = ticker.FuncFormatter(
        lambda x, p: f"{x/1e6:.1f}M" if x >= 1e6
        else f"{x/1e3:.0f}k" if x >= 1e3
        else f"{x:.0f}"
    )

    param_info = [
        ("alpha_mean", r"$\alpha$", "#1f77b4"),
        ("beta_mean", r"$\beta$", "#ff7f0e"),
    ]

    fig, axes = plt.subplots(2, len(q_values), figsize=(12, 6))
    if len(q_values) == 1:
        axes = axes.reshape(2, 1)

    # Collect y-ranges per row for unified axes
    y_ranges = [[float("inf"), float("-inf")] for _ in range(2)]

    for col_idx, q in enumerate(q_values):
        q_df = df[df["q"] == q]

        if q_df.empty:
            axes[0, col_idx].set_title(f"{format_q(q)} (no data)")
            continue

        seeds = sorted(q_df["seed"].unique())
        has_multi = len(seeds) > 1

        for row_idx, (col_name, label, color) in enumerate(param_info):
            ax = axes[row_idx, col_idx]

            if has_multi:
                # Per-seed thin traces
                for seed in seeds:
                    seed_df = q_df[q_df["seed"] == seed].sort_values("step")
                    valid = seed_df[~seed_df[col_name].isna()]
                    ax.plot(valid["step"], valid[col_name],
                            color=color, alpha=0.2, linewidth=0.8, zorder=1)

                # Aggregate mean + 95% CI band
                agg = aggregate_seeds(q_df)
                mean_col = f"{col_name}_mean"
                ci_col = f"{col_name}_ci95"
                if mean_col in agg.columns:
                    steps = agg["step"].values
                    mean = agg[mean_col].values
                    ci = agg[ci_col].values if ci_col in agg.columns else np.zeros_like(mean)
                    ax.plot(steps, mean, color=color, linewidth=2, zorder=3)
                    ax.fill_between(steps, mean - ci, mean + ci,
                                    color=color, alpha=SHADE_ALPHA, zorder=2)
                    y_ranges[row_idx][0] = min(y_ranges[row_idx][0], np.nanmin(mean - ci))
                    y_ranges[row_idx][1] = max(y_ranges[row_idx][1], np.nanmax(mean + ci))
            else:
                single = q_df.sort_values("step")
                valid = single[~single[col_name].isna()]
                ax.plot(valid["step"], valid[col_name],
                        color=color, linewidth=2, zorder=3)
                y_ranges[row_idx][0] = min(y_ranges[row_idx][0], valid[col_name].min())
                y_ranges[row_idx][1] = max(y_ranges[row_idx][1], valid[col_name].max())

            ax.xaxis.set_major_formatter(x_formatter)

            # Title only on top row
            if row_idx == 0:
                ax.set_title(format_q(q))
            # Y-label only on leftmost column
            if col_idx == 0:
                ax.set_ylabel(label)
            # X-label only on bottom row
            if row_idx == len(param_info) - 1:
                ax.set_xlabel("Training Steps")

    # Unify y-axis per row
    for row_idx in range(2):
        ymin, ymax = y_ranges[row_idx]
        if ymin < float("inf"):
            margin = (ymax - ymin) * 0.05
            for col_idx in range(len(q_values)):
                axes[row_idx, col_idx].set_ylim(ymin - margin, ymax + margin)

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
    q: float = 45.0,
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
    df = _baseline_only(df)
    if "experiment" in df.columns:
        df = df[df["experiment"] == "two_players"]

    if df.empty or "alpha_mean" not in df.columns:
        print("[plots] Warning: No alpha/beta data available for snapshots")
        return None, None

    # Compute e* for this q and the effort scale factor
    e_theory = e_star(q, **THEORY_PARAMS)

    # Select best seed (smallest final effort error)
    seeds = df["seed"].unique()
    if len(seeds) > 1:
        best_seed, best_err = None, float("inf")
        for s in seeds:
            s_df = df[df["seed"] == s].sort_values("step")
            final_effort = s_df["policy_mean_effort"].iloc[-1]
            err = abs(final_effort - e_theory)
            if err < best_err:
                best_seed, best_err = s, err
        df = df[df["seed"] == best_seed]
        print(f"[plots] Beta snapshots: using seed={best_seed} (|e-e*|={best_err:.2f})")

    df = df.sort_values("step")
    max_step = df["step"].max()
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
                label="Both agents")
        ax.fill_between(x_effort, y, alpha=SHADE_ALPHA, color=AGENT_COLORS["agent1"])

        # Mark policy mean in effort space
        mean_norm = alpha / kappa
        mean_effort = mean_norm * e_max
        ax.axvline(x=mean_effort, color=AGENT_COLORS["agent2"], linestyle="--",
                    linewidth=2, label=f"Mean={mean_effort:.1f}")

        # e* vertical line (green)
        ax.axvline(x=e_theory, color="green", linestyle="-.",
                    linewidth=2, label=f"$e^*$={e_theory:.1f}")

        ax.set_xlabel("Effort")
        ax.set_ylabel("Density")
        ax.set_title(f"Step {int(row['step'])} ({frac*100:.0f}%)\nκ={kappa:.1f}")
        ax.legend(loc="upper right", fontsize=7)

        y_max_global = max(y_max_global, np.max(y))

    # Unify axes across all panels
    for ax in axes:
        ax.set_xlim(0, 150)
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
    df = _baseline_only(df)

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
    
    x_formatter = ticker.FuncFormatter(
        lambda x, p: f"{x/1e6:.1f}M" if x >= 1e6
        else f"{x/1e3:.0f}k" if x >= 1e3
        else f"{x:.0f}"
    )

    # First pass: collect y-range across all panels for unified axis
    y_min_global, y_max_global = float("inf"), float("-inf")

    for ax, q in zip(axes, q_values):
        q_df = df[df["q"] == q]
        e_theory = e_star(q, **THEORY_PARAMS)

        # Theory line — prominent
        ax.axhline(
            y=e_theory, color=THEORY_LINE_COLOR, linestyle="--",
            linewidth=THEORY_LINE_WIDTH, label="Theory", zorder=4,
        )

        # Plot each ablation with per-seed traces + aggregate mean + CI
        for ablation in sorted(ablations):
            abl_df = q_df[q_df["ablation"] == ablation]
            if abl_df.empty:
                continue

            color = ABLATION_COLORS.get(ablation, "gray")
            label = ABLATION_LABELS.get(ablation, ablation)
            lw = ABLATION_LINEWIDTHS.get(ablation, 1.5)
            seeds = sorted(abl_df["seed"].unique())
            has_multi = len(seeds) > 1

            if has_multi:
                # Per-seed thin traces
                for seed in seeds:
                    seed_df = abl_df[abl_df["seed"] == seed].sort_values("step")
                    ax.plot(seed_df["step"], seed_df["policy_mean_effort"],
                            color=color, alpha=0.2, linewidth=0.8, zorder=1)

                # Aggregate mean + 95% CI band
                agg = aggregate_seeds(abl_df)
                if "policy_mean_effort_mean" in agg.columns:
                    steps = agg["step"].values
                    mean = agg["policy_mean_effort_mean"].values
                    ci = agg["policy_mean_effort_ci95"].values if "policy_mean_effort_ci95" in agg.columns else np.zeros_like(mean)
                    ax.plot(steps, mean, color=color, linewidth=lw,
                            label=label, zorder=3)
                    ax.fill_between(steps, mean - ci, mean + ci,
                                    color=color, alpha=SHADE_ALPHA, zorder=2)
                    y_min_global = min(y_min_global, np.nanmin(mean - ci))
                    y_max_global = max(y_max_global, np.nanmax(mean + ci))
            else:
                single = abl_df.sort_values("step")
                ax.plot(single["step"], single["policy_mean_effort"],
                        color=color, linewidth=lw, label=label, zorder=3)
                y_min_global = min(y_min_global, single["policy_mean_effort"].min())
                y_max_global = max(y_max_global, single["policy_mean_effort"].max())

        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Effort")
        ax.set_title(format_q(q))
        ax.xaxis.set_major_formatter(x_formatter)

    # Unify y-axis across all panels
    if y_min_global < float("inf"):
        margin = (y_max_global - y_min_global) * 0.05
        for ax in axes:
            ax.set_ylim(y_min_global - margin, y_max_global + margin)

    plt.tight_layout()

    # Single legend at top of figure, outside all axes
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        ncol=len(labels), fontsize=FONT_SIZES["legend"], frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    if save_data:
        data_path = os.path.join(DATA_DIR, "ablation_comparison.csv")
        df[["step", "method", "q", "seed", "ablation", "policy_mean_effort", "theoretical_effort"]].to_csv(data_path, index=False)
    
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

    # Filter to two_players PPO, baseline weight variant only
    df = df[
        (df["q"].isin(q_values))
        & (df["method"].isin(["TEL-PPO", "PPO"]))
    ]
    if "experiment" in df.columns:
        df = df[df["experiment"] == "two_players"]
    df = _baseline_only(df)

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
                        ax.plot(sdf["step"], sdf["policy_mean_effort"],
                                color=color, alpha=0.2, linewidth=0.7, zorder=1)

                    # Aggregated mean + CI
                    agg = aggregate_seeds(abl_df)
                    if "policy_mean_effort_mean" in agg.columns:
                        steps = agg["step"].values
                        mean_vals = agg["policy_mean_effort_mean"].values
                        ci_vals = agg.get("policy_mean_effort_ci95", pd.Series(np.zeros(len(agg)))).values
                        ax.plot(steps, mean_vals, color=color, linewidth=lw,
                                label=label, zorder=3)
                        ax.fill_between(steps, mean_vals - ci_vals, mean_vals + ci_vals,
                                        color=color, alpha=0.12, zorder=2)
                    else:
                        ax.plot(abl_df["step"], abl_df["policy_mean_effort"],
                                color=color, linewidth=lw, label=label, zorder=3)
                else:
                    ax.plot(abl_df["step"], abl_df["policy_mean_effort"],
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
        out_cols = ["step", "method", "q", "seed", "ablation", "policy_mean_effort", "theoretical_effort"]
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
    df = _baseline_only(df)
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

    q_colors = {25.0: "#1f77b4", 35.0: "#9467bd", 40.0: "#ff7f0e", 55.0: "#2ca02c"}

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
                    agg["policy_mean_effort_mean"] - agg["theoretical_effort"]
                )
                agg["effort_error_ci95"] = agg.get("policy_mean_effort_ci95", 0)

            steps = agg["step"].values
            err_mean = agg["effort_error_mean"].values
            err_ci = agg.get("effort_error_ci95", pd.Series([0] * len(agg))).values

            ax.plot(steps, err_mean, color=color, linewidth=2, label=format_q(q))
            ax.fill_between(
                steps,
                np.maximum(err_mean - err_ci, 0),
                err_mean + err_ci,
                color=color,
                alpha=SHADE_ALPHA,
            )
        else:
            q_df = q_df.sort_values("step")
            ax.plot(
                q_df["step"],
                q_df["effort_error"],
                color=color,
                linewidth=2,
                label=format_q(q),
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
                linewidth=1.0, alpha=0.7,
                label=f"Detected convergence step ({format_q(q)})",
            )

    # ε threshold horizontal line
    effort_delta = CONVERGENCE_CONFIG["effort_delta"]
    ax.axhline(
        y=effort_delta, color="gray", linestyle="--",
        linewidth=2.0, alpha=0.7, label=f"Target error threshold (ε = {effort_delta})",
    )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Equilibrium error |ē − e*|")
    ax.set_title("Convergence Error to the Analytical Equilibrium")
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
        cols = ["step", "method", "q", "seed", "ablation", "policy_mean_effort", "theoretical_effort"]
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
    df = _baseline_only(df)
    if "experiment" in df.columns:
        df = df[df["experiment"] == "two_players"]

    if df.empty or "drift_effort" not in df.columns or df["drift_effort"].isna().all():
        print("[plots] Warning: No drift data available for effort_drift plot")
        return None, None

    fig, axes = plt.subplots(1, len(q_values), figsize=(12, 4))
    if len(q_values) == 1:
        axes = [axes]

    conv_steps_df = get_convergence_step(df)

    for i, (ax, q) in enumerate(zip(axes, q_values)):
        q_df = df[df["q"] == q].sort_values("step")

        if q_df.empty:
            ax.set_title(f"{format_q(q)} (no data)")
            continue

        valid = q_df[~q_df["drift_effort"].isna()]
        if valid.empty:
            ax.set_title(f"{format_q(q)} (no drift data)")
            continue

        # Aggregate by step across seeds (uniform grid, no binning needed)
        binned = valid.groupby("step")["drift_effort"].agg(
            median="median",
            p10=lambda x: np.percentile(x, 10),
            p90=lambda x: np.percentile(x, 90),
        ).reset_index()
        binned = binned.sort_values("step")

        # Light rolling smooth (window=3) to reduce jaggedness
        for col in ["median", "p10", "p90"]:
            binned[col] = binned[col].rolling(3, center=True, min_periods=1).mean()

        # Shaded 10–90th percentile band
        ax.fill_between(
            binned["step"], binned["p10"], binned["p90"],
            color="#56B4E9", alpha=SHADE_ALPHA, label="10\u201390% interval",
        )

        # Bold median line
        ax.plot(
            binned["step"], binned["median"],
            color="#0072B2", linewidth=2, label="Median drift",
        )

        # Threshold line
        drift_thresh = CHEAP_GATE_CONFIG["drift_effort_thresh"]
        ax.axhline(
            y=drift_thresh, color="#D55E00", linestyle="--",
            linewidth=2.5, label=f"Drift threshold ({drift_thresh})",
        )

        # Convergence step vertical line
        q_conv = conv_steps_df[conv_steps_df["q"] == q]
        if "experiment" in conv_steps_df.columns:
            q_conv = q_conv[q_conv["experiment"] == "two_players"]
        q_conv = q_conv[q_conv["ablation"] == "baseline"]
        mean_conv = q_conv["convergence_step"].dropna().mean()
        if not np.isnan(mean_conv):
            ax.axvline(
                x=mean_conv, color="#009E73", linestyle=":",
                linewidth=1.5, alpha=0.8, label="Detected convergence step",
            )

        ax.set_xlabel("Training Steps")
        if i == 0:
            ax.set_ylabel("Effort Drift")
        ax.set_title(format_q(q))
        ax.set_ylim(0, 2.8)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda x, p: f"{x/1e6:.1f}M"
                if x >= 1e6
                else f"{x/1e3:.0f}k"
                if x >= 1e3
                else f"{x:.0f}"
            )
        )
        if i == 0:
            ax.legend(loc="best", fontsize=8)


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
    ppo_df = _baseline_only(ppo_df)

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

    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZES["equilibrium_recovery_dotplot"])

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
                            color=color, marker=marker, s=40, zorder=3,
                            alpha=0.5, edgecolors="white", linewidth=0.5,
                            label=agent_label if not _legend_added[agent_key] else None,
                        )
                        _legend_added[agent_key] = True

                        # Per-agent mean marker
                        valid_efforts = efforts[~np.isnan(efforts)]
                        if len(valid_efforts) > 0:
                            ax.scatter(
                                x_pos, valid_efforts.mean(),
                                color=color, marker="D", s=100, zorder=4,
                                edgecolors="black", linewidth=1,
                            )

            else:
                # --- Single-marker behavior (original) ---
                e_theory = q_final["theoretical_effort"].iloc[0]
                ax.hlines(
                    e_theory,
                    x_pos - 0.3, x_pos + 0.3,
                    colors="#333333", linestyles="--", linewidth=3, zorder=2,
                )

                efforts = q_final["policy_mean_effort"].values
                jitter = rng.uniform(-0.15, 0.15, len(efforts))
                ax.scatter(
                    x_pos + jitter, efforts,
                    color="#ff7f0e", s=40, zorder=3, alpha=0.5,
                    edgecolors="white", linewidth=0.5,
                    label="Per-seed estimate" if not _legend_added["single"] else None,
                )
                _legend_added["single"] = True

                # Mean marker
                mean_effort = efforts.mean()
                ax.scatter(
                    x_pos, mean_effort,
                    color="#d62728", marker="D", s=100, zorder=4,
                    edgecolors="black", linewidth=1,
                    label="Across-seed mean" if not _legend_added["mean"] else None,
                )
                _legend_added["mean"] = True

            x_ticks.append(x_pos)
            x_labels.append(f"q={int(q)}")
            x_pos += 1

        # Add separator between experiments
        if x_pos > 0:
            ax.axvline(x=x_pos - 0.25, color="gray", linewidth=0.5, alpha=0.3, zorder=0)
            x_pos += 0.5

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=FONT_SIZES["tick_label"])
    ax.set_ylabel("Equilibrium Effort")
    ax.set_title("Equilibrium Recovery Across Scenarios and Noise Levels", pad=30)

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

    # Add alternating background shading for scenario groups
    x_pos_bg = 0
    for i, exp in enumerate(experiments):
        exp_final = final[final["experiment"] == exp] if "experiment" in final.columns else pd.DataFrame()
        if exp_final.empty:
            continue
        n_q = len(exp_final["q"].unique())
        if i % 2 == 1:
            ax.axvspan(x_pos_bg - 0.4, x_pos_bg + n_q - 0.6,
                       color="#f0f0f0", alpha=0.5, zorder=0)
        x_pos_bg += n_q + 0.5

    trans = ax.get_xaxis_transform()
    for center, label in group_starts:
        ax.text(
            center, -0.08, label,
            transform=trans,
            ha="center", va="top",
            fontsize=FONT_SIZES["tick_label"],
            fontweight="bold",
        )

    # Legend
    legend_elements = [
        Line2D([0], [0], color="#333333", linestyle="--", linewidth=3, label="Theory e*"),
    ]
    if _legend_added["single"]:
        legend_elements.append(Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#ff7f0e",
            markersize=7, alpha=0.5, label="Per-seed estimate",
        ))
    if _legend_added["mean"]:
        legend_elements.append(Line2D(
            [0], [0], marker="D", color="w", markerfacecolor="#d62728",
            markeredgecolor="black", markersize=8, label="Across-seed mean",
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
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_elements),
        frameon=False,
        fontsize=FONT_SIZES["legend"],
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, top=0.88)

    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

    if save_data:
        # Compute per-seed relative error |e_learned - e*| / e*
        save_df = final.copy()
        save_df["relative_error"] = np.where(
            save_df["theoretical_effort"] != 0,
            np.abs(save_df["policy_mean_effort"] - save_df["theoretical_effort"]) / np.abs(save_df["theoretical_effort"]),
            np.nan,
        )
        data_path = os.path.join(DATA_DIR, "equilibrium_recovery_dotplot.csv")
        cols_to_save = ["method", "q", "seed", "ablation", "policy_mean_effort",
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

    # Exploitability q=25 (excluded low-noise case, Figure 6b)
    fig, path = plot_exploitability_q25(df)
    if path:
        results["exploitability_q25"] = path

    # Beta evolution
    fig, path = plot_beta_evolution(df, q_values)
    if path:
        results["beta_evolution"] = path
    
    # Beta snapshots (for q=45, mid-range 2P Set 1)
    fig, path = plot_beta_snapshots(df, q=45.0)
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
