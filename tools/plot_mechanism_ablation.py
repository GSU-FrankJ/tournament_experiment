#!/usr/bin/env python3
"""
Plot Mechanism Ablation Results

Generates comparison plots for the mechanism ablation experiment:
- Convergence speed comparison (bar chart per experiment type)
- Final exploitability comparison
- Learning curves overlay (baseline vs each ablation)

Usage:
    python tools/plot_mechanism_ablation.py --input results/ablation/mechanism/

    python tools/plot_mechanism_ablation.py --input results/ablation/mechanism/ \
        --output results/ablation/mechanism/figures/

    python tools/plot_mechanism_ablation.py --input results/ablation/mechanism/ \
        --experiments two_players,different_cost
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

SETTING_COLORS = {
    "baseline": "#1f77b4",      # Blue
    "no_cheap_gate": "#ff7f0e", # Orange
    "no_exploit": "#2ca02c",    # Green
    "no_entropy": "#d62728",    # Red
}

SETTING_LABELS = {
    "baseline": "Baseline",
    "no_cheap_gate": "No Cheap Gate",
    "no_exploit": "No Exploitability",
    "no_entropy": "No Entropy",
}

EXPERIMENT_LABELS = {
    "two_players": "Two Players",
    "three_players": "Three Players",
    "different_cost": "Different Cost",
    "different_ability": "Different Ability",
}

Q_VALUES = [25.0, 40.0, 55.0]
SEEDS = [42, 123, 456]


# =============================================================================
# Data Loading
# =============================================================================

def load_results(input_dir: str) -> List[Dict[str, Any]]:
    """Load all run results from the runs/ subdirectory."""
    runs_dir = os.path.join(input_dir, "runs")
    results = []

    if not os.path.exists(runs_dir):
        # Try loading from summary.json
        summary_path = os.path.join(input_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                data = json.load(f)
            return data.get("results", [])
        return results

    for filename in sorted(os.listdir(runs_dir)):
        if not filename.endswith(".json"):
            continue
        try:
            with open(os.path.join(runs_dir, filename)) as f:
                data = json.load(f)
            if data.get("success"):
                results.append(data)
        except Exception:
            pass

    return results


def load_convergence_data(experiment: str, q: float, seed: int, setting_id: str) -> Optional[Dict]:
    """Load convergence JSON for a specific run to get time-series data."""
    convergence_dir = os.path.join("results", experiment, "convergence")

    patterns = []
    if experiment == "different_cost":
        patterns.append(f"different_cost_ppo_q{q:.1f}_seed{seed}_{setting_id}_convergence.json")
    elif experiment == "different_ability":
        patterns.append(f"different_ability_ppo_q{q:.1f}_seed{seed}_{setting_id}_convergence.json")
    elif experiment == "two_players":
        patterns.append(f"ppo_q{q:.1f}_seed{seed}_{setting_id}_convergence.json")
    elif experiment == "three_players":
        patterns.append(f"ppo_3p_q{q:.1f}_seed{seed}_{setting_id}_convergence.json")

    for pattern in patterns:
        path = os.path.join(convergence_dir, pattern)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass

    return None


# =============================================================================
# Plot: Convergence Speed Comparison
# =============================================================================

def plot_convergence_speed(
    results: List[Dict],
    experiments: List[str],
    output_path: str,
) -> None:
    """Bar chart: median stopped_at_update per setting, grouped by experiment."""
    fig, axes = plt.subplots(1, len(experiments), figsize=(4 * len(experiments), 5), squeeze=False)

    settings = list(SETTING_COLORS.keys())

    for col, experiment in enumerate(experiments):
        ax = axes[0, col]
        exp_results = [r for r in results if r["experiment"] == experiment]

        medians = []
        colors = []
        labels = []

        for setting in settings:
            setting_results = [
                r for r in exp_results
                if r["setting_id"] == setting and r.get("stopped_at_update") is not None
            ]
            if setting_results:
                updates = [r["stopped_at_update"] for r in setting_results]
                medians.append(np.median(updates))
            else:
                medians.append(0)
            colors.append(SETTING_COLORS[setting])
            labels.append(SETTING_LABELS[setting])

        x = np.arange(len(settings))
        bars = ax.bar(x, medians, color=colors, width=0.6, edgecolor="black", linewidth=0.5)

        ax.set_title(EXPERIMENT_LABELS.get(experiment, experiment), fontsize=12)
        ax.set_ylabel("Median Updates to Converge" if col == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

        # Add value labels on bars
        for bar, val in zip(bars, medians):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"{int(val)}",
                    ha="center", va="bottom", fontsize=8,
                )

    fig.suptitle("Convergence Speed by Mechanism Setting", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved convergence speed: {output_path}")


# =============================================================================
# Plot: Final Error Comparison
# =============================================================================

def plot_final_error(
    results: List[Dict],
    experiments: List[str],
    output_path: str,
) -> None:
    """Box plot: final_abs_err_max per setting, grouped by experiment."""
    fig, axes = plt.subplots(1, len(experiments), figsize=(4 * len(experiments), 5), squeeze=False)

    settings = list(SETTING_COLORS.keys())

    for col, experiment in enumerate(experiments):
        ax = axes[0, col]
        exp_results = [r for r in results if r["experiment"] == experiment]

        data_per_setting = []
        colors_list = []
        labels_list = []

        for setting in settings:
            setting_results = [
                r for r in exp_results
                if r["setting_id"] == setting and r.get("final_abs_err_max") is not None
            ]
            errors = [r["final_abs_err_max"] for r in setting_results]
            data_per_setting.append(errors if errors else [0])
            colors_list.append(SETTING_COLORS[setting])
            labels_list.append(SETTING_LABELS[setting])

        bp = ax.boxplot(
            data_per_setting,
            patch_artist=True,
            labels=labels_list,
            widths=0.5,
        )

        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(EXPERIMENT_LABELS.get(experiment, experiment), fontsize=12)
        ax.set_ylabel("Final |e - e*| max" if col == 0 else "")
        ax.tick_params(axis="x", rotation=45)

        for label in ax.get_xticklabels():
            label.set_fontsize(8)
            label.set_ha("right")

    fig.suptitle("Final Effort Error by Mechanism Setting", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved final error: {output_path}")


# =============================================================================
# Plot: Learning Curves Overlay
# =============================================================================

def plot_learning_curves(
    results: List[Dict],
    experiment: str,
    q: float,
    output_path: str,
) -> None:
    """Overlay learning curves for all settings on one plot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    settings = list(SETTING_COLORS.keys())

    for setting in settings:
        # Load all seeds for this setting
        efforts_by_step = defaultdict(list)

        for seed in SEEDS:
            conv_data = load_convergence_data(experiment, q, seed, setting)
            if conv_data is None:
                continue

            steps = conv_data.get("steps", [])
            effort = conv_data.get("policy_mean_effort", conv_data.get("agent1_effort", []))

            if not steps or not effort:
                continue

            for s, e in zip(steps, effort):
                if e is not None and not (isinstance(e, float) and (e != e)):  # skip NaN
                    efforts_by_step[s].append(e)

        if not efforts_by_step:
            continue

        sorted_steps = sorted(efforts_by_step.keys())
        means = [np.mean(efforts_by_step[s]) for s in sorted_steps]

        ax.plot(
            sorted_steps, means,
            color=SETTING_COLORS[setting],
            label=SETTING_LABELS[setting],
            linewidth=1.5,
            alpha=0.8,
        )

    # Add theoretical effort line
    setting_results = [
        r for r in results
        if r["experiment"] == experiment and r.get("theoretical_effort_1") is not None
    ]
    if setting_results:
        theo = setting_results[0]["theoretical_effort_1"]
        ax.axhline(y=theo, color="black", linestyle="--", linewidth=1, label=f"e* = {theo:.1f}")

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Effort")
    ax.set_title(f"{EXPERIMENT_LABELS.get(experiment, experiment)} (q={q:.0f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved learning curves: {output_path}")


# =============================================================================
# Plot: Exploitability Comparison
# =============================================================================

def plot_exploitability(
    results: List[Dict],
    experiments: List[str],
    output_path: str,
) -> None:
    """Bar chart: median final exploitability per setting."""
    fig, axes = plt.subplots(1, len(experiments), figsize=(4 * len(experiments), 5), squeeze=False)

    settings = list(SETTING_COLORS.keys())

    for col, experiment in enumerate(experiments):
        ax = axes[0, col]
        exp_results = [r for r in results if r["experiment"] == experiment]

        medians = []
        colors = []
        labels = []

        for setting in settings:
            setting_results = [
                r for r in exp_results
                if r["setting_id"] == setting and r.get("final_exploit_max") is not None
            ]
            if setting_results:
                exploits = [r["final_exploit_max"] for r in setting_results]
                medians.append(np.median(exploits))
            else:
                medians.append(0)
            colors.append(SETTING_COLORS[setting])
            labels.append(SETTING_LABELS[setting])

        x = np.arange(len(settings))
        bars = ax.bar(x, medians, color=colors, width=0.6, edgecolor="black", linewidth=0.5)

        ax.set_title(EXPERIMENT_LABELS.get(experiment, experiment), fontsize=12)
        ax.set_ylabel("Median Final Exploitability" if col == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

        for bar, val in zip(bars, medians):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7,
                )

    fig.suptitle("Final Exploitability by Mechanism Setting", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved exploitability: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot Mechanism Ablation Results")

    parser.add_argument(
        "--input",
        type=str,
        default="results/ablation/mechanism/",
        help="Input directory with sweep results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for figures (default: {input}/figures/)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated experiment types to plot (default: all with data)",
    )

    args = parser.parse_args()

    output_dir = args.output or os.path.join(args.input, "figures")
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    results = load_results(args.input)

    if not results:
        print(f"No results found in {args.input}")
        print("Run the sweep first: python tools/sweep_mechanism_ablation.py")
        return

    print(f"Loaded {len(results)} successful results")

    # Determine experiments to plot
    available_experiments = sorted(set(r["experiment"] for r in results))
    if args.experiments:
        experiments = [e.strip() for e in args.experiments.split(",")]
        experiments = [e for e in experiments if e in available_experiments]
    else:
        experiments = available_experiments

    if not experiments:
        print("No experiments found to plot")
        return

    print(f"Plotting experiments: {', '.join(experiments)}")

    # Generate plots
    plot_convergence_speed(
        results, experiments,
        os.path.join(output_dir, "convergence_speed.png"),
    )

    plot_final_error(
        results, experiments,
        os.path.join(output_dir, "final_error.png"),
    )

    plot_exploitability(
        results, experiments,
        os.path.join(output_dir, "exploitability.png"),
    )

    # Learning curves for each experiment and q value
    for experiment in experiments:
        for q in Q_VALUES:
            exp_q_results = [
                r for r in results
                if r["experiment"] == experiment and r["q"] == q
            ]
            if exp_q_results:
                plot_learning_curves(
                    results, experiment, q,
                    os.path.join(output_dir, f"learning_curves_{experiment}_q{q:.0f}.png"),
                )

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
