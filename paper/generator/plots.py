"""
Figure Generation for Paper Artifacts.

Implements:
- Main convergence figure (1x3 faceted by q)
- KL/exploitability dynamics
- Beta distribution evolution
- Ablation comparison
"""

import json
import os
from typing import List, Optional, Tuple, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.stats import beta as beta_dist

from .config import (
    FIGURES_DIR,
    DATA_DIR,
    RESULTS_DIR,
    CONVERGENCE_DIR,
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
    get_verified_convergence_step,
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


def _holdforward_seeds(
    method_df: pd.DataFrame, value_cols: List[str]
) -> pd.DataFrame:
    """Extend each seed to the panel's common step grid by holding its last
    observed value forward past its verified early-stop.

    TEL-PPO runs early-stop at different updates when their exploitability
    verification fires, so a plain step-wise average over the *union* of steps is
    dominated in the tail by the few longest-running seeds (survivorship): the
    cross-seed mean drifts toward whichever seed ran longest (the q=35 rebound)
    and the CI band jumps whenever a seed drops out (the q=55 tail step). A
    converged-and-stopped seed sits at its final effort thereafter, so holding
    that value forward keeps every seed in each step's average — removing the
    rebound and drop-out jumps while preserving the genuine cross-seed spread.

    Args:
        method_df: one (experiment, method, q, ablation) panel, multiple seeds.
        value_cols: numeric columns to forward-fill (e.g. per-agent efforts).

    Returns:
        DataFrame with every seed present on the full union step grid; value_cols
        forward-filled after each seed's last logged step.
    """
    all_steps = np.sort(method_df["step"].unique())
    id_cols = [
        c for c in ("experiment", "method", "q", "ablation", "weight_variant",
                    "theoretical_effort")
        if c in method_df.columns
    ]
    frames = []
    for seed, g in method_df.groupby("seed"):
        g = g.sort_values("step").drop_duplicates("step").set_index("step")
        g = g.reindex(all_steps)
        for c in value_cols:
            if c in g.columns:
                g[c] = g[c].ffill()
        for c in id_cols:
            if c in g.columns:
                g[c] = g[c].ffill().bfill()
        g["seed"] = seed
        frames.append(g.reset_index())
    return pd.concat(frames, ignore_index=True)


def _first_pass_updates(df: pd.DataFrame, steps_per_update: int = 4096) -> pd.DataFrame:
    """Per-run FIRST-PASS verification update: the first in-training
    exploitability check the raw self-play profile passes
    (exploitability < eps_eq), before the patience streak completes.

    Reported in the same 1-based update convention as ``stopped_at_update``
    (a verified run's 5th consecutive passing check sits exactly at
    ``stopped_at_update``; series row i corresponds to update i+1).

    Returns one row per run with columns:
        experiment, method, q, seed, ablation [, weight_variant],
        first_pass_update (NaN if no check ever passed)
    """
    eps = CONVERGENCE_CONFIG["exploit_threshold"]
    group_cols = ["method", "q", "seed", "ablation"]
    if "weight_variant" in df.columns:
        group_cols.append("weight_variant")
    if "experiment" in df.columns:
        group_cols = ["experiment"] + group_cols

    records = []
    for key, grp in df.groupby(group_cols):
        grp = grp.sort_values("step")
        fp = np.nan
        if "exploitability" in grp.columns and "exploitability_is_valid" in grp.columns:
            passed = grp[
                (grp["exploitability_is_valid"] == True)  # noqa: E712
                & grp["exploitability"].notna()
                & (grp["exploitability"] < eps)
            ]
            if not passed.empty:
                fp = float(passed["step"].iloc[0]) / steps_per_update + 1.0
        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        rec["first_pass_update"] = fp
        records.append(rec)

    return pd.DataFrame(records)


def _polished_two_player_means() -> Dict[Tuple[str, float], float]:
    """Cross-seed mean MC-BR polished effort per (weight_variant, q) for the
    two-player cells. Set 1 ("baseline") comes from the per-seed polish
    artifact (``results/one_stage_ablation/polish_per_seed_all.json``); Set 2
    ("wh8_wl4") from the sibling artifact ``polish_per_seed_set2.json``.
    A missing artifact contributes no keys (its panels then get no star)."""
    out: Dict[Tuple[str, float], float] = {}
    dfp = load_polished_dotplot_final()
    if dfp is not None and "experiment" in dfp.columns:
        sub = dfp[dfp["experiment"] == "two_players"]
        for q, g in sub.groupby("q"):
            out[("baseline", float(q))] = float(g["policy_mean_effort"].mean())
    set2_path = os.path.join(
        RESULTS_DIR, "one_stage_ablation", "polish_per_seed_set2.json")
    if os.path.exists(set2_path):
        try:
            rows = json.load(open(set2_path)).get("rows", [])
        except Exception:
            rows = []
        by_q: Dict[float, List[float]] = {}
        for r in rows:
            if r.get("experiment") == "two_players_set2":
                by_q.setdefault(float(r["q"]), []).append(float(r["single_value"]))
        for q, vals in by_q.items():
            out[("wh8_wl4", q)] = float(np.mean(vals))
    return out


def plot_convergence_main(
    df: pd.DataFrame = None,
    q_values: List[float] = None,
    weight_variants: List[str] = None,
    output_path: str = None,
    save_data: bool = True,
    figsize: Tuple[float, float] = None,
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
        n_rows, n_cols, figsize=figsize or FIGURE_SIZES["convergence_main"],
        squeeze=False,
    )

    # PPO rollout batch size (env steps per policy update). Trajectories are
    # logged once per update, so update index = step / STEPS_PER_UPDATE. The
    # x-axis is displayed in *updates* to match the Claim-B "Conv. Update" column.
    STEPS_PER_UPDATE = 4096

    # Gray vertical line = VERIFIED stop (mean ``stopped_at_update`` across
    # seeds: the 5th consecutive passing exploitability check, where training
    # stops and the raw estimate is read out — the Claim-B "Conv. Update
    # (verified)" column). First-pass updates are computed alongside and
    # printed for reference only (see the per-panel comment below).
    verified_conv_df = get_verified_convergence_step(df)
    first_pass_df = _first_pass_updates(df, STEPS_PER_UPDATE)

    # MC-BR polished endpoint (star marker) exists only for the Set-1
    # two-player cell; Set 2 has no polish artifact and gets no star.
    polished_means = _polished_two_player_means()

    # Global x-extent of the baseline runs (used to offset the star marker).
    _ppo_mask = df["method"].isin(["TEL-PPO", "PPO"])
    if "ablation" in df.columns:
        _ppo_mask = _ppo_mask & (df["ablation"] == "baseline")
    _gx_max = float(df.loc[_ppo_mask, "step"].max()) if _ppo_mask.any() else 0.0

    for row_idx, variant in enumerate(weight_variants):
        wv_col = "weight_variant" if "weight_variant" in df.columns else "ablation"
        var_df = df[df[wv_col] == variant]
        # Fix #1: when selecting a weight-variant row, restrict to the baseline
        # ablation so non-baseline arms (r5_fig7_*, eps_*/pat_* sweeps) do not
        # leak into the baseline convergence curve / CI band. (No-op when keyed
        # on ablation, i.e. when there is no weight_variant column.)
        if wv_col == "weight_variant":
            var_df = var_df[var_df["ablation"] == "baseline"]

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
                # The numeric e* label is placed after the trajectories are
                # drawn, so its slot can be chosen against the curve extent.

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

            # Endpoint markers at the end of the mean trajectory:
            #   open circle = raw final estimate (cross-seed mean of each
            #                 seed's final policy-mean effort, no polish)
            #   filled star = MC-BR polished estimate (cross-seed mean of the
            #                 per-seed polished efforts; Set-1 cell only)
            # Star is offset slightly in x so the two markers never overlap.
            if not method_df.empty and "policy_mean_effort" in method_df.columns:
                seed_final = (
                    method_df.sort_values("step")
                    .groupby("seed")["policy_mean_effort"].last()
                )
                raw_final = float(seed_final.mean())
                x_end = float(method_df["step"].max())
                ax.plot(
                    [x_end], [raw_final], marker="o", ms=8, mfc="white",
                    mec="#333333", mew=1.8, linestyle="none", zorder=6,
                    clip_on=False,
                )
                pol_val = polished_means.get((variant, float(q)))
                if pol_val is not None:
                    x_star = x_end + 0.015 * (_gx_max if _gx_max > 0 else x_end)
                    ax.plot(
                        [x_star], [pol_val], marker="*",
                        ms=15, color="#d62728", mec="k", mew=0.5,
                        linestyle="none", zorder=7, clip_on=False,
                    )

            # Numeric e* label. Default slot is the right edge just above the
            # theory line. Every panel shares the longest run's x-limit, so that
            # slot is empty only for panels whose run ends well short of the
            # right margin. The longest run (q=55) is still descending toward
            # e* when it reaches that margin, and its label would sit on the
            # trajectory; drop it below the line at mid-panel instead. The band
            # under e* is free everywhere except the polished star at the far
            # right, because the curves approach the equilibrium from above.
            if np.isfinite(e_theory):
                x_end_panel = float(method_df["step"].max())
                crowded = _gx_max > 0 and x_end_panel >= 0.85 * _gx_max
                ax.annotate(
                    f"$e^*={e_theory:.2f}$",
                    xy=(0.5 if crowded else 1.0, e_theory),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, -6) if crowded else (-4, 4),
                    textcoords="offset points",
                    ha="center" if crowded else "right",
                    va="top" if crowded else "bottom",
                    color=THEORY_LINE_COLOR,
                    fontsize=FONT_SIZES["legend"], fontweight="bold", zorder=6,
                )

            # Gray vertical line: FIRST-PASS verification update (mean across
            # seeds of the first passing exploitability check). Positioned in
            # step space (update * STEPS_PER_UPDATE) since curves are plotted
            # against cumulative steps. Streak completion (stopped_at_update)
            # is printed alongside for reference.
            def _panel_mask(frame):
                m = (
                    (frame["q"] == q)
                    & (frame["method"].isin(["TEL-PPO", "PPO"]))
                    & (frame["ablation"] == "baseline")
                )
                if "weight_variant" in frame.columns:
                    m = m & (frame["weight_variant"] == variant)
                if "experiment" in frame.columns:
                    m = m & (frame["experiment"] == "two_players")
                return m

            # Line position = VERIFIED stop: the update at which the 5-check
            # exploitability streak completes, training stops, and the raw
            # estimate (final.effort) is read out — mean stopped_at_update
            # across seeds (the Claim-B "Conv. Update (verified)" column).
            # The FIRST-PASS update (first single passing check) is computed
            # and printed for reference only: the asymmetric exploration
            # warm-up keeps the plotted rollout curves far apart while the
            # near-e* initialized policy can already pass one loose
            # (eps_eq=0.03 absolute) check as early as update ~9, so marking
            # first-pass misreads as premature verification.
            fp_vals = first_pass_df.loc[
                _panel_mask(first_pass_df), "first_pass_update"
            ].dropna()
            conv_vals = verified_conv_df.loc[
                _panel_mask(verified_conv_df), "convergence_update"
            ].dropna()
            if not conv_vals.empty:
                mean_conv = conv_vals.mean()
                ax.axvline(
                    x=mean_conv * STEPS_PER_UPDATE, color=CONV_VLINE_COLOR,
                    linestyle=CONV_VLINE_LINESTYLE,
                    linewidth=CONV_VLINE_LINEWIDTH,
                    label="First verification update", zorder=4,
                )
                print(
                    f"[plots] convergence_main [{variant} q={q}]: line at "
                    f"verified stop, mean update {mean_conv:.1f} (per-seed "
                    f"{sorted(conv_vals.astype(int).tolist())}); first-pass "
                    f"reference {fp_vals.mean():.1f} (per-seed "
                    f"{sorted(fp_vals.astype(int).tolist())})"
                )

            # Axis formatting
            if row_idx == n_rows - 1:
                ax.set_xlabel("Training Updates")
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

            # Display the x-axis in training updates (step / STEPS_PER_UPDATE).
            # A single figure-level legend is added after the loop.
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(
                    lambda x, p: f"{x / STEPS_PER_UPDATE:.0f}"
                )
            )

    # Unify y-axis across all panels
    _unify_ylim(axes)

    # Unify x-axis across all panels: extend every panel to the longest
    # (high-noise) run so the extra updates that high noise requires are visible.
    # Restrict to the baseline ablation that is actually plotted, so long
    # non-baseline sweep arms (eps_*/pat_*, r5_fig7_*) do not stretch the axis.
    ppo_mask = df["method"].isin(["TEL-PPO", "PPO"])
    if "ablation" in df.columns:
        ppo_mask = ppo_mask & (df["ablation"] == "baseline")
    ppo_steps = df.loc[ppo_mask, "step"]
    if not ppo_steps.empty:
        x_max = float(ppo_steps.max())
        for ax in np.asarray(axes).flat:
            # Extra right headroom so the x-offset polished star at the longest
            # (q=55) run's endpoint is fully inside the frame, not clipped.
            ax.set_xlim(0, x_max * 1.05)

    plt.tight_layout()
    plt.subplots_adjust(left=0.12)

    # Single figure-level legend, placed outside the grid at the top-right.
    # Order matters: matplotlib fills columns top-to-bottom, so with ncol=3 and
    # two rows this lays out
    #   col 1: Theory / First verification update
    #   col 2: Agent 1 / Agent 2
    #   col 3: Raw estimate / MC-BR polished estimate
    legend_handles = [
        Line2D([0], [0], color=THEORY_LINE_COLOR, linestyle="--",
               linewidth=THEORY_LINE_WIDTH, label="Theory $e^*$"),
        Line2D([0], [0], color=CONV_VLINE_COLOR, linestyle=CONV_VLINE_LINESTYLE,
               linewidth=CONV_VLINE_LINEWIDTH, label="First verification update"),
        Line2D([0], [0], color=AGENT_COLORS["agent1"], linestyle="-",
               linewidth=2, label="Agent 1"),
        Line2D([0], [0], color=AGENT_COLORS["agent2"], linestyle="--",
               linewidth=2, label="Agent 2"),
        Line2D([0], [0], marker="o", linestyle="none", mfc="white",
               mec="#333333", mew=1.8, markersize=8, label="Raw estimate"),
        Line2D([0], [0], marker="*", linestyle="none", color="#d62728",
               mec="k", mew=0.5, markersize=15,
               label="MC-BR polished estimate"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower right",
        bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=True,
        fontsize=FONT_SIZES["legend"],
    )

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


def plot_convergence_main_1x3(
    df: pd.DataFrame = None,
    q_values: List[float] = None,
    output_path: str = None,
) -> Tuple[plt.Figure, str]:
    """1x3 companion of the convergence-main figure: only the Set-1
    (w_H, w_L) = (6.5, 3.0) row across q = 35/45/55, same styling and legend,
    sized like the exploitability-dynamics 1x3 layout. Written to a NEW file
    (``convergence_main_1x3``); never overwrites the 3-row figure and does not
    rewrite the shared convergence_main.csv."""
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "convergence_main_1x3.png")
    return plot_convergence_main(
        df=df,
        q_values=q_values,
        weight_variants=["baseline"],
        output_path=output_path,
        save_data=False,
        figsize=FIGURE_SIZES["exploitability_dynamics"],
    )


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

    # Unified x-axis: share one training-step range across all q panels
    _pos_kl = df[df["approx_kl"].notna() & (df["approx_kl"] > 0)]
    global_step_max = _pos_kl["step"].max() if not _pos_kl.empty else None

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
        if global_step_max is not None:
            ax.set_xlim(0, global_step_max)
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

    # Forward-fill exploitability (within each seed, across its own logged steps)
    df = forward_fill_exploitability(df)

    # PPO rollout batch size (env steps per policy update). The x-axis is shown in
    # *updates* to match the Claim-B "Conv. Update (verified)" column and the
    # convergence-main figure.
    STEPS_PER_UPDATE = 4096

    # Precompute the stability-screen (cheap-gate) step and the Claim-B verified
    # ε-equilibrium convergence update. The green marker uses the method's OWN
    # verified stop (stability + exploitability streak) — i.e. the Claim-B
    # "Conv. Update (verified)" value (two-player baseline: q35=55, q55=87) — so
    # the figure and the reported table agree.
    from .extract import get_cheap_gate_step
    gate_steps_df = get_cheap_gate_step(df)
    verified_conv_df = get_verified_convergence_step(df)

    fig, axes = plt.subplots(1, len(q_values), figsize=FIGURE_SIZES["exploitability_dynamics"])
    if len(q_values) == 1:
        axes = [axes]

    # Smoothing window for rolling mean (dense per-update grid, so window << grid).
    smooth_window = 15

    for ax, q in zip(axes, q_values):
        q_df = df[df["q"] == q].sort_values("step")

        if q_df.empty:
            ax.set_title(f"{format_q(q)} (no data)")
            continue

        seeds = sorted(q_df["seed"].unique())

        # Hold each seed's exploitability forward past its verified early-stop onto
        # the panel's common step grid, so every seed contributes at every step.
        # A plain step-wise mean over the union of steps is dominated in the tail
        # by the few longest-running seeds (survivorship); more importantly, the
        # cross-seed mean has only ~7-11 support points (one per sparse evaluation)
        # and a window-15 rolling mean over so few points collapses it to a flat
        # line. Holding forward yields a dense per-update grid over which the mean
        # is a genuine, smoothly declining cross-seed average.
        panel_df = _holdforward_seeds(q_df, ["exploitability_ffill"])

        # Per-seed thin lines (smoothed)
        for seed in seeds:
            sdf = panel_df[panel_df["seed"] == seed].sort_values("step")
            valid = sdf[~sdf["exploitability_ffill"].isna()]
            if not valid.empty:
                smoothed = valid["exploitability_ffill"].rolling(
                    window=smooth_window, min_periods=1, center=True,
                ).mean()
                ax.plot(
                    valid["step"].values, smoothed.values,
                    color="#1f77b4", alpha=0.3, linewidth=0.8,
                )

        # Bold mean across seeds (smoothed) — cross-seed average of the held-forward
        # exploitability at each step.
        mean_series = (
            panel_df.groupby("step")["exploitability_ffill"].mean().dropna()
        )
        if not mean_series.empty:
            smoothed = mean_series.rolling(
                window=smooth_window, min_periods=1, center=True,
            ).mean()
            ax.plot(
                mean_series.index.values, smoothed.values,
                color="#1f77b4", linewidth=2, label="Mean Exploit.",
            )

        # Threshold line → bolder
        exploit_thresh = CONVERGENCE_CONFIG["exploit_threshold"]
        ax.axhline(
            y=exploit_thresh, color="red", linestyle="--",
            linewidth=2.5, alpha=0.8,
            label=f"Tolerance $\\varepsilon_{{eq}} = {exploit_thresh}$",
        )

        # Stability screen: first step the cheap gate passed (median across seeds)
        # → single orange line.
        gate_match = gate_steps_df[
            (gate_steps_df["q"] == q) & (gate_steps_df["ablation"] == "baseline")
        ]
        if "experiment" in gate_steps_df.columns:
            gate_match = gate_match[gate_match["experiment"] == "two_players"]
        gate_vals = gate_match["cheap_gate_step"].dropna()
        if not gate_vals.empty:
            ax.axvline(
                x=gate_vals.median(), color="orange", linestyle=":",
                linewidth=1.5, alpha=0.8, label="Stability passed",
            )

        # Verified ε-equilibrium: mean verified convergence update across seeds
        # (the method's own stability + exploitability stop = Claim-B "Conv. Update
        # (verified)") → single green line, positioned in step space.
        conv_match = verified_conv_df[
            (verified_conv_df["q"] == q) & (verified_conv_df["ablation"] == "baseline")
        ]
        if "experiment" in verified_conv_df.columns:
            conv_match = conv_match[conv_match["experiment"] == "two_players"]
        conv_vals = conv_match["convergence_update"].dropna()
        if not conv_vals.empty:
            ax.axvline(
                x=conv_vals.mean() * STEPS_PER_UPDATE, color="green", linestyle="-.",
                linewidth=1.5, alpha=0.8, label="Verified",
            )

        ax.set_xlabel("Training Updates")
        if ax == axes[0]:
            # The plotted series is the exploitability of the RAW self-play
            # profile evaluated at the periodic in-training verification
            # checks (run_two_players.eval_exploitability on the current
            # policy) — the quantity that drives the streak stop criterion,
            # NOT the post-polish exploitability.
            ax.set_ylabel("Raw profile exploitability")
        ax.set_title(format_q(q))
        ax.set_yscale("log")
        ax.set_ylim(0.005, 2)
        # Display the x-axis in training updates (step / STEPS_PER_UPDATE).
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, p: f"{x / STEPS_PER_UPDATE:.0f}")
        )

    # Unify the x-axis across panels: extend every panel to the longest (high-noise)
    # baseline run so the three noise levels are compared on a common scale.
    ppo_steps = df["step"]
    if not ppo_steps.empty:
        x_max = float(ppo_steps.max())
        for ax in axes:
            ax.set_xlim(0, x_max * 1.02)

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
    # Compact single-row legend fully above the axes region: the four short
    # entries no longer squeeze the panels (tight_layout uses the full canvas
    # and bbox_inches='tight' grows the saved figure around the legend).
    fig.legend(
        unique_handles, unique_labels,
        loc="lower center", ncol=len(unique_labels),
        fontsize=8, frameon=True,
        bbox_to_anchor=(0.5, 1.0),
    )

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


# --- ablation figure constants -----------------------------------------------
# Component ablation of TEL-PPO (two-player). On-disk ablation tags are mapped
# to the canonical keys used by ABLATION_COLORS/LABELS/LINEWIDTHS.
_ABL_DISK_TO_CANON: Dict[str, str] = {
    "baseline": "baseline",
    "r5_sampled": "baseline",
    "r5_fig7_no_stability": "no_cheap_gate",
    "r5_fig7_no_exploitability": "no_exploitability",
    "r5_fig7_no_exploit": "no_exploitability",
}
_ABL_CANON_TO_DISK: Dict[str, str] = {
    "baseline": "r5_sampled",
    "no_cheap_gate": "r5_fig7_no_stability",
    "no_exploitability": "r5_fig7_no_exploit",
}
_ABL_ORDER = ["baseline", "no_cheap_gate", "no_exploitability"]
_ABL_THEORY_LW = 3.0          # item b: theory line more prominent than default
_ABL_NOPOLISH_COLOR = "#9467bd"
_ABL_X_BREAK = 0.55e6         # split: convergence detail | non-terminating tail
_ABL_X_MAX = 6.25e6
_ABL_STEP_PER_UPDATE = 4096
_ABL_MIN_STEP = 8192          # drop first 1-2 evals (random-init smear)


def _ablation_polished_landings() -> Dict[float, float]:
    """Claim-B MC-BR polished landing (per q) from the one-stage ablation JSON."""
    path = os.path.join(RESULTS_DIR, "one_stage_ablation", "ablation_results.json")
    out: Dict[float, float] = {}
    if not os.path.exists(path):
        return out
    try:
        cells = json.load(open(path)).get("cells", {})
        for k, cell in cells.items():
            if "polish_mean" in cell:
                out[float(k)] = float(cell["polish_mean"])
    except Exception:
        pass
    return out


def _ablation_arm_metrics(q: float, canon: str, seeds: List[int]) -> Dict:
    """Per-seed summary metrics for one (q, arm): terminal effort, exploitability,
    non-convergence rate, and time-to-verification (mean over verified seeds)."""
    tag = _ABL_CANON_TO_DISK[canon]
    terms, expl, ncs, upds = [], [], [], []
    for s in seeds:
        f = os.path.join(CONVERGENCE_DIR, f"ppo_q{q}_seed{s}_{tag}_convergence.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        terms.append(d["policy_mean_effort"][-1])
        e = d.get("final_exploit_max")
        if e is not None and not (isinstance(e, float) and np.isnan(e)):
            expl.append(e)
        nc = d.get("stop_reason") == "max_updates"
        ncs.append(nc)
        if not nc and d.get("stopped_at_update") is not None:
            upds.append(d["stopped_at_update"])
    return dict(
        term_raw=(float(np.mean(terms)) if terms else None),
        exploit=(float(np.mean(expl)) if expl else None),
        nc=(float(np.mean(ncs)) if ncs else None),
        verify_M=(float(np.mean(upds)) * _ABL_STEP_PER_UPDATE / 1e6 if upds else None),
    )


def _ablation_aggregate(sub: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean + 95% CI over seeds, keyed by step."""
    g = (sub.groupby("step")["policy_mean_effort"]
         .agg(["mean", "std", "count"]).reset_index().sort_values("step"))
    ci = 1.96 * g["std"].fillna(0.0) / np.sqrt(g["count"].clip(lower=1))
    return g["step"].values, g["mean"].values, ci.values


def plot_ablation_comparison(
    df: pd.DataFrame = None,
    q_values: List[float] = None,
    output_path: str = None,
    save_data: bool = True,
) -> Tuple[plt.Figure, str]:
    """Component ablation of TEL-PPO (Figure 7).

    Per-q panels (broken x-axis: convergence detail | non-terminating tail) show
    the raw-PPO training traces of three arms — TEL-PPO (baseline), No stability
    screening, No exploitability verification — with per-seed traces + 95% CI
    bands and a prominent Theory e* line. The TEL-PPO endpoint carries a
    no-polish (raw) vs MC-BR-polished (Claim-B) fork. A summary table reports
    terminal |e-e*|, final exploitability, non-convergence rate, and
    time-to-verification per arm.
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

    # Map on-disk ablation tags to canonical arm keys, keep the 3 main arms.
    df = df.copy()
    df["ablation"] = df["ablation"].map(lambda a: _ABL_DISK_TO_CANON.get(a, a))
    df = df[df["ablation"].isin(_ABL_ORDER)]
    df = df[df["step"] >= _ABL_MIN_STEP]

    if df.empty:
        print("[plots] Warning: No data for ablation comparison")
        return None, None

    polished = _ablation_polished_landings()

    # ---- pre-compute aggregates + unified y-range ----
    agg_cache: Dict[Tuple[float, str], Tuple] = {}
    y_min, y_max = float("inf"), float("-inf")
    for q in q_values:
        for arm in _ABL_ORDER:
            sub = df[(df["q"] == q) & (df["ablation"] == arm)]
            if sub.empty:
                continue
            st, mn, ci = _ablation_aggregate(sub)
            agg_cache[(q, arm)] = (st, mn, ci)
            y_min = min(y_min, np.nanmin(mn - ci))
            y_max = max(y_max, np.nanmax(mn + ci))
    y_min = min(y_min, min(e_star(q, **THEORY_PARAMS) for q in q_values))
    pad = (y_max - y_min) * 0.08
    ylim = (y_min - pad, y_max + pad)

    # ---- layout: q panels (broken x) + spacer (x label) + summary table ----
    n = len(q_values)
    fig = plt.figure(figsize=(11, 3.0 * n + 0.7))
    gs = gridspec.GridSpec(
        n + 2, 2, width_ratios=[3.0, 1.7],
        height_ratios=[1] * n + [0.16, 0.60], hspace=0.42, wspace=0.04,
    )
    tail_fmt = ticker.FuncFormatter(lambda x, p: f"{x/1e6:.0f}")
    detail_fmt = ticker.FuncFormatter(lambda x, p: f"{x/1e6:.1f}")

    axL_list = []
    for i, q in enumerate(q_values):
        axL = fig.add_subplot(gs[i, 0])
        axR = fig.add_subplot(gs[i, 1], sharey=axL)
        axL_list.append(axL)
        e_theory = e_star(q, **THEORY_PARAMS)

        for ax in (axL, axR):
            ax.axhline(e_theory, color=THEORY_LINE_COLOR, linestyle="--",
                       linewidth=_ABL_THEORY_LW, zorder=5,
                       label="Theory $e^*$" if ax is axL else None)
            for arm in _ABL_ORDER:
                sub = df[(df["q"] == q) & (df["ablation"] == arm)]
                if sub.empty:
                    continue
                color = ABLATION_COLORS.get(arm, "gray")
                for _, sd in sub.groupby("seed"):
                    sd = sd.sort_values("step")
                    ax.plot(sd["step"], sd["policy_mean_effort"], color=color,
                            alpha=0.13, linewidth=0.7, zorder=1)
                st, mn, ci = agg_cache[(q, arm)]
                ax.plot(st, mn, color=color, linewidth=ABLATION_LINEWIDTHS.get(arm, 1.5),
                        zorder=3, label=ABLATION_LABELS.get(arm, arm) if ax is axL else None)
                ax.fill_between(st, mn - ci, mn + ci, color=color,
                                alpha=SHADE_ALPHA, zorder=2)

        # no-polish (raw) vs MC-BR-polished (Claim-B) fork at TEL-PPO endpoint
        if (q, "baseline") in agg_cache:
            st_b, mn_b, _ = agg_cache[(q, "baseline")]
            x_end, raw = st_b[-1], mn_b[-1]
            axL.plot([x_end], [raw], marker="o", ms=8, mfc="white",
                     mec=_ABL_NOPOLISH_COLOR, mew=1.8, zorder=6,
                     label="No polish (raw PPO)" if i == 0 else None)
            if q in polished:
                axL.annotate("", xy=(x_end, polished[q]), xytext=(x_end, raw),
                             arrowprops=dict(arrowstyle="->", color="#444", lw=1.3),
                             zorder=6)
                axL.plot([x_end], [polished[q]], marker="*", ms=15,
                         color=ABLATION_COLORS["baseline"], mec="k", mew=0.5, zorder=7,
                         label="TEL-PPO + MC-BR polish" if i == 0 else None)

        # broken-axis cosmetics
        axL.set_xlim(0, _ABL_X_BREAK)
        axR.set_xlim(_ABL_X_BREAK, _ABL_X_MAX)
        axL.set_ylim(*ylim)
        axL.spines["right"].set_visible(False)
        axR.spines["left"].set_visible(False)
        axR.tick_params(labelleft=False, left=False)
        d = 0.012
        kw = dict(transform=axL.transAxes, color="k", clip_on=False, lw=1)
        axL.plot((1 - d, 1 + d), (-d, d), **kw)
        axL.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
        dr = d * 3.0 / 1.7
        kw2 = dict(transform=axR.transAxes, color="k", clip_on=False, lw=1)
        axR.plot((-dr, dr), (-d, d), **kw2)
        axR.plot((-dr, dr), (1 - d, 1 + d), **kw2)
        axL.xaxis.set_major_formatter(detail_fmt)
        axR.xaxis.set_major_formatter(tail_fmt)
        axR.set_xticks([2e6, 4e6, 6e6])
        axL.set_ylabel("Effort")
        axL.set_title(format_q(q), loc="left", x=0.02, fontsize=FONT_SIZES["title"])
        axL.grid(alpha=0.25)
        axR.grid(alpha=0.25)
        axR.annotate("no verification →\nnever terminates", xy=(0.5, 0.06),
                     xycoords="axes fraction", ha="center",
                     fontsize=FONT_SIZES["annotation"],
                     color=ABLATION_COLORS["no_exploitability"])

    # shared x label in the spacer row
    p_spacer = gs[n, :].get_position(fig)
    fig.text(0.5, p_spacer.y0 + p_spacer.height * 0.35,
             r"Training Steps ($\times 10^6$)", ha="center", va="center",
             fontsize=FONT_SIZES["axis_label"])

    # shared legend at top
    handles, labels = axL_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=FONT_SIZES["legend"], bbox_to_anchor=(0.5, 1.005))

    # ---- summary table ----
    seeds_by_arm = {arm: sorted(df[df["ablation"] == arm]["seed"].unique().tolist())
                    for arm in _ABL_ORDER}
    met = {arm: {q: _ablation_arm_metrics(q, arm, seeds_by_arm[arm])
                 for q in q_values} for arm in _ABL_ORDER}

    def _err(arm, q, use_polish=False):
        if use_polish and q in polished:
            return abs(polished[q] - e_star(q, **THEORY_PARAMS))
        t = met[arm][q]["term_raw"]
        return None if t is None else abs(t - e_star(q, **THEORY_PARAMS))

    def _join(vals, fmt):
        return "/".join("—" if v is None else fmt(v) for v in vals)

    qs = q_values
    e2 = lambda v: f"{v:.2f}"
    e3 = lambda v: f"{v:.3f}"
    qlab = "/".join(f"q{int(q)}" for q in qs)
    rows = [
        ["TEL-PPO (polished)",
         _join([_err("baseline", q, True) for q in qs], e2),
         _join([met["baseline"][q]["exploit"] for q in qs], e3),
         "0%",
         _join([met["baseline"][q]["verify_M"] for q in qs], e2)],
        ["  · No polish (raw)",
         _join([_err("baseline", q) for q in qs], e2), "—", "—", "—"],
        ["No stability screening",
         _join([_err("no_cheap_gate", q) for q in qs], e2),
         _join([met["no_cheap_gate"][q]["exploit"] for q in qs], e3),
         "0%",
         _join([met["no_cheap_gate"][q]["verify_M"] for q in qs], e2)],
        ["No exploitability verif.",
         _join([_err("no_exploitability", q) for q in qs], e2),
         "—", "100%", "never"],
    ]
    cols = ["Arm", f"Terminal |ē−e*|\n({qlab})", "Final\nexploit.",
            "NC\nrate", "Time-to-verify\n(×10⁶ steps)"]

    axT = fig.add_subplot(gs[n + 1, :])
    axT.axis("off")
    tbl = axT.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FONT_SIZES["tick_label"])
    tbl.scale(1, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
        elif c == 0:
            cell.get_text().set_ha("left")

    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

    if save_data:
        data_path = os.path.join(DATA_DIR, "ablation_comparison.csv")
        keep = ["step", "method", "q", "seed", "ablation", "policy_mean_effort"]
        if "theoretical_effort" in df.columns:
            keep.append("theoretical_effort")
        df[keep].to_csv(data_path, index=False)

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
        # canonical gate is the unified eps_eq=0.03 (eps_003 therefore
        # duplicates the baseline gate and serves as a consistency arm)
        "baseline": "baseline (ε=0.03)",
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

    # Precompute the verified convergence update per run (Claim-B criterion: the
    # PPO update at which the exploitability verification fired). The diagnostic
    # effort-band detector is structurally unsatisfiable for early-stopped runs
    # (all-NaN => no line drawn), which is why the old figure showed no marker.
    conv_steps_df = get_verified_convergence_step(df)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    q_colors = {25.0: "#1f77b4", 35.0: "#9467bd", 40.0: "#ff7f0e", 55.0: "#2ca02c"}

    for q in q_values:
        q_df = df[df["q"] == q]
        if q_df.empty:
            continue

        color = q_colors.get(q, "gray")

        # Aggregate across seeds. Plot the per-seed MAE  <|e_i - e*|>  — the same
        # quantity Claim-B reports as |ē - e*| (settling at 1.87 for q=35, 0.79
        # for q=55) — NOT |<e_i> - e*|. The latter collapses toward zero whenever
        # the *mean* effort transits e* on its way into the basin, producing the
        # spurious log-scale plunges the old figure showed (q=35 near 100k steps).
        # Seeds are also held forward past their verified early-stop so the tail
        # is not dominated by the few longest-running seeds (the q=55 dip).
        has_multi = q_df["seed"].nunique() > 1
        if has_multi:
            padded = _holdforward_seeds(q_df, ["effort_error", "policy_mean_effort"])
            stat = (
                padded.groupby("step")["effort_error"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("step")
            )
            steps = stat["step"].values
            err_mean = stat["mean"].values
            # Band = +/-1 standard error of the mean. On this log axis a linear
            # 95% CI (1.96*SEM) dips below zero while the mean is still ~1-2 and
            # clips to the y-floor, drawing spurious downward spikes; +/-1 SEM
            # stays positive and is the tighter band the "too wide" note asks for.
            err_sem = stat["std"].fillna(0.0).values / np.sqrt(stat["count"].values)

            ax.plot(steps, err_mean, color=color, linewidth=2, label=format_q(q))
            ax.fill_between(
                steps,
                np.clip(err_mean - err_sem, 1e-3, None),
                err_mean + err_sem,
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

        # Detected convergence step: mean verified convergence update across seeds
        # (Claim-B "Conv. Update (verified)": q=35 -> 55, q=55 -> 87) mapped onto
        # the training-step axis via the run's own steps-per-update stride.
        q_conv = conv_steps_df[conv_steps_df["q"] == q]
        if "experiment" in conv_steps_df.columns:
            q_conv = q_conv[q_conv["experiment"] == "two_players"]
        q_conv = q_conv[q_conv["ablation"] == "baseline"]
        mean_conv_update = q_conv["convergence_update"].dropna().mean()
        steps_per_update = q_df["step"].drop_duplicates().sort_values().diff().median()
        if not np.isnan(mean_conv_update) and not np.isnan(steps_per_update):
            # No per-q label: a single proxy entry is added to the legend below so
            # the three colored lines do not each claim a legend row.
            ax.axvline(
                x=mean_conv_update * steps_per_update, color=color, linestyle=":",
                linewidth=1.0, alpha=0.7,
            )

    # ε threshold horizontal line
    effort_delta = CONVERGENCE_CONFIG["effort_delta"]
    ax.axhline(
        y=effort_delta, color="gray", linestyle="--",
        linewidth=2.0, alpha=0.7, label=f"Target error threshold (ε = {effort_delta})",
    )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel(r"Equilibrium error  $\langle\,|e_i - e^*|\,\rangle$")
    ax.set_title("Convergence Error to the Analytical Equilibrium")
    # One proxy entry for the (per-q coloured) convergence-step verticals so they
    # occupy a single legend row instead of three, keeping the compact layout.
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="0.5", linestyle=":", linewidth=1.2))
    labels.append("Detected convergence step")
    ax.legend(handles, labels, loc="best", fontsize=FONT_SIZES["legend"])
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

    # Use the method's OWN verified stop (Claim B: mean stopped_at_update from the
    # exploitability streak) rather than the diagnostic effort-band detector, which
    # is structurally unsatisfiable for early-stopped runs (all-NaN => no line).
    conv_steps_df = get_verified_convergence_step(df)

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

        # Verified convergence step vertical line: mean stopped_at_update (in PPO
        # updates) mapped onto the training-step axis via steps-per-update, derived
        # from the run's own step grid (no hard-coded constant).
        q_conv = conv_steps_df[conv_steps_df["q"] == q]
        if "experiment" in conv_steps_df.columns:
            q_conv = q_conv[q_conv["experiment"] == "two_players"]
        q_conv = q_conv[q_conv["ablation"] == "baseline"]
        mean_conv_update = q_conv["convergence_update"].dropna().mean()
        steps_per_update = q_df["step"].drop_duplicates().sort_values().diff().median()
        if not np.isnan(mean_conv_update) and not np.isnan(steps_per_update):
            mean_conv = mean_conv_update * steps_per_update
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


# Claim-B MC-BR polished per-seed efforts (produced by
# tools/one_stage_polish_per_seed.py). When present, this is the canonical source
# for the equilibrium-recovery dot plot — it shows the *polished* (verified,
# ~e*) efforts rather than the raw PPO basin landings.
POLISH_PER_SEED_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "one_stage_ablation", "polish_per_seed_all.json",
)


def load_polished_dotplot_final(json_path: str = None) -> Optional[pd.DataFrame]:
    """Load Claim-B polished per-seed efforts as a final-values DataFrame.

    Maps ``polish_per_seed_all.json`` into exactly the columns
    ``plot_equilibrium_recovery_dotplot`` reads via ``final_override``. Returns
    ``None`` when the polished artifact is absent, so callers can fall back to the
    raw convergence extraction.

    Args:
        json_path: Path to the polished per-seed JSON (defaults to
            ``POLISH_PER_SEED_JSON``).

    Returns:
        One row per experiment/q/seed with polished efforts, or ``None`` if the
        artifact does not exist.
    """
    if json_path is None:
        json_path = POLISH_PER_SEED_JSON
    if not os.path.exists(json_path):
        return None

    with open(json_path) as fh:
        rows = json.load(fh)["rows"]

    records = []
    for r in rows:
        exp = r["experiment"]
        e_pol = r["e_polished_per_player"]
        e_star = r["e_star_per_player"]
        rec = {
            "experiment": exp,
            "method": "TEL-PPO",
            "q": float(r["q"]),
            "seed": int(r["seed"]),
            "ablation": "baseline",
        }
        if exp == "different_cost":
            # Per-agent markers + per-agent theory lines.
            rec["agent1_effort"] = e_pol[0]
            rec["agent2_effort"] = e_pol[1]
            rec["theoretical_effort1"] = e_star[0]
            rec["theoretical_effort2"] = e_star[1]
            rec["theoretical_effort"] = sum(e_star) / len(e_star)
            rec["policy_mean_effort"] = sum(e_pol) / len(e_pol)
        else:
            # Single-marker (symmetric) cells.
            val = r["single_value"]
            rec["policy_mean_effort"] = val
            rec["agent1_effort"] = val
            rec["agent2_effort"] = val
            rec["theoretical_effort"] = e_star[0]
            rec["theoretical_effort1"] = e_star[0]
            rec["theoretical_effort2"] = e_star[0]
        records.append(rec)

    return pd.DataFrame(records)


def plot_equilibrium_recovery_dotplot(
    df: pd.DataFrame = None,
    output_path: str = None,
    save_data: bool = True,
    final_override: pd.DataFrame = None,
) -> Tuple[plt.Figure, str]:
    """
    Plot equilibrium recovery dot plot across scenarios (Figure 6).

    Shows learned equilibrium effort vs theoretical for each experiment type.
    x-axis: scenario, y-axis: effort, dots for learned, lines for theoretical.

    Args:
        df: Convergence DataFrame (loaded automatically if None). Ignored when
            ``final_override`` is provided.
        output_path: Destination PNG (a sibling PDF is also written).
        save_data: Whether to dump the backing per-seed CSV.
        final_override: Pre-computed per-run final values (one row per
            experiment/method/q/seed/ablation) to plot INSTEAD of extracting them
            from the convergence JSONs. Used to render Claim-B MC-BR *polished*
            efforts while keeping the drawing style unchanged. Must carry the same
            columns the drawing reads: ``experiment, method, q, seed, ablation,
            policy_mean_effort, agent1_effort, agent2_effort, theoretical_effort``
            (plus ``theoretical_effort1/2`` for heterogeneous-cost).
    """
    setup_matplotlib_style()
    ensure_output_dirs()

    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "equilibrium_recovery_dotplot.png")

    if final_override is not None:
        # Injected per-run final values (e.g. Claim-B MC-BR polished efforts).
        # Bypasses convergence-JSON extraction; the drawing logic below is unchanged.
        final = final_override.copy()
    else:
        if df is None:
            df = load_all_convergence_data()

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
                # --- Per-agent markers with a single shared theory style ---
                # Agent mapping verified against the run config + theory:
                # k1 = 0.0004 < k2 = 0.00055, so agent 1 is the LOW-cost agent
                # and carries the higher equilibrium effort (e1* > e2*).
                # Both per-agent theory lines use the SAME "Theory e*" style
                # (dashed, #333333) as every other panel, so the legend keeps a
                # single "Theory e*" entry (no per-agent line-style split). The
                # two agents remain distinguishable by their colored markers.
                for agent_key, effort_col, theory_col, agent_label in [
                    ("agent1", "agent1_effort", "theoretical_effort1",
                     "Low-cost agent"),
                    ("agent2", "agent2_effort", "theoretical_effort2",
                     "High-cost agent"),
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
                                colors="#333333", linestyles="--",
                                linewidth=3, zorder=2,
                            )

                    # Per-seed scatter: semi-transparent, BEHIND the mean
                    # diamonds (zorder 3 < 4) so the means carry the panel.
                    if effort_col in q_final.columns:
                        efforts = q_final[effort_col].values
                        jitter = rng.uniform(-0.12, 0.12, len(efforts))
                        ax.scatter(
                            x_pos + jitter, efforts,
                            facecolors=to_rgba(color, 0.35), marker=marker,
                            s=55, zorder=3,
                            edgecolors=to_rgba("black", 0.45), linewidth=0.8,
                            label=agent_label if not _legend_added[agent_key] else None,
                        )
                        _legend_added[agent_key] = True

                        # Per-agent mean marker
                        valid_efforts = efforts[~np.isnan(efforts)]
                        if len(valid_efforts) > 0:
                            ax.scatter(
                                x_pos, valid_efforts.mean(),
                                color=color, marker="D", s=130, zorder=4,
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
                    facecolors=to_rgba("#ff7f0e", 0.35), s=55, zorder=3,
                    edgecolors=to_rgba("black", 0.45), linewidth=0.8,
                    label="Seed-level estimates" if not _legend_added["single"] else None,
                )
                _legend_added["single"] = True

                # Mean marker
                mean_effort = efforts.mean()
                ax.scatter(
                    x_pos, mean_effort,
                    color="#d62728", marker="D", s=130, zorder=4,
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
    ax.set_ylabel("Effort Level")
    # pad clears the two-row legend that sits between the axes and the title.
    ax.set_title(
        "Equilibrium Effort Recovery Across Tournament Variants and Noise Levels",
        pad=52,
    )

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

    # Legend — a single "Theory e*" entry covers every panel (all theory
    # lines share one dashed #333333 style, including heterogeneous cost).
    legend_elements = [
        Line2D([0], [0], color="#333333", linestyle="--", linewidth=3, label="Theory e*"),
    ]
    if _legend_added["single"]:
        legend_elements.append(Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=to_rgba("#ff7f0e", 0.35),
            markeredgecolor=to_rgba("black", 0.45), markersize=7,
            label="Seed-level estimates",
        ))
    if _legend_added["mean"]:
        legend_elements.append(Line2D(
            [0], [0], marker="D", color="w", markerfacecolor="#d62728",
            markeredgecolor="black", markersize=8, label="Across-seed mean",
        ))
    if _legend_added["agent1"]:
        legend_elements.append(Line2D(
            [0], [0], marker=AGENT_MARKERS["agent1"], color="w",
            markerfacecolor=to_rgba(AGENT_COLORS["agent1"], 0.35),
            markeredgecolor=to_rgba("black", 0.45), markersize=7,
            label="Low-cost agent",
        ))
    if _legend_added["agent2"]:
        legend_elements.append(Line2D(
            [0], [0], marker=AGENT_MARKERS["agent2"], color="w",
            markerfacecolor=to_rgba(AGENT_COLORS["agent2"], 0.35),
            markeredgecolor=to_rgba("black", 0.45), markersize=7,
            label="High-cost agent",
        ))
    # Display order. matplotlib fills legend columns top-to-bottom, so with
    # ncol=3 this lays out
    #   col 1: Theory e*        / High-cost agent
    #   col 2: Across-seed mean / Low-cost agent
    #   col 3: Seed-level estimates
    # keeping the low-/high-cost pair together in one column instead of
    # splitting it across the grid.
    _legend_order = [
        "Theory e*", "High-cost agent", "Across-seed mean",
        "Low-cost agent", "Seed-level estimates",
    ]
    legend_elements.sort(
        key=lambda h: _legend_order.index(h.get_label())
        if h.get_label() in _legend_order else len(_legend_order)
    )

    # Two-row legend (3 columns for the 6 entries) — a single row runs the
    # full figure width and crowds the title.
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(1, (len(legend_elements) + 1) // 2),
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

    # NOTE: plot_exploitability_q25 and plot_beta_snapshots retired 2026-06-12 —
    # both were dormant under canonical data (q=25 dropped in the parameter
    # overhaul; no alpha/beta snapshot series in the canonical runs).

    # Beta evolution
    fig, path = plot_beta_evolution(df, q_values)
    if path:
        results["beta_evolution"] = path

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

    # Equilibrium recovery dotplot (Fig 6) — prefer Claim-B polished efforts.
    polished_final = load_polished_dotplot_final()
    if polished_final is not None:
        print("[plots] equilibrium_recovery_dotplot: using Claim-B polished efforts "
              f"({POLISH_PER_SEED_JSON})")
    else:
        print("[plots] equilibrium_recovery_dotplot: polished artifact missing, "
              "falling back to raw PPO landings")
    fig, path = plot_equilibrium_recovery_dotplot(df, final_override=polished_final)
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
