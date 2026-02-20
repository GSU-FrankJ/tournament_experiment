"""
Table Generation for Paper Artifacts.

Implements:
- Summary metrics table (CSV + LaTeX)
- Ablation results table
- Final paper table
"""

import os
from typing import List, Optional, Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    TABLES_DIR,
    Q_VALUES,
    e_star,
    THEORY_PARAMS,
    CONVERGENCE_CONFIG,
    classify_quality,
)
from .extract import load_all_convergence_data, get_final_values
from .metrics import compute_summary_metrics, summary_metrics_to_dataframe


def ensure_tables_dir():
    """Create tables output directory if it doesn't exist."""
    os.makedirs(TABLES_DIR, exist_ok=True)


def _format_float(val: float, decimals: int = 2) -> str:
    """Format float for table display, handling NaN."""
    if val is None or np.isnan(val):
        return "-"
    return f"{val:.{decimals}f}"


def _to_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    float_format: str = "%.2f",
) -> str:
    """Convert DataFrame to LaTeX table string."""
    # Clean up column names for LaTeX
    latex_cols = []
    for col in df.columns:
        col_clean = col.replace("_", " ").title()
        latex_cols.append(col_clean)
    
    df_copy = df.copy()
    df_copy.columns = latex_cols
    
    latex_str = df_copy.to_latex(
        index=False,
        float_format=float_format,
        na_rep="-",
        escape=False,
    )
    
    # Wrap in table environment with caption
    full_latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{latex_str}
\\end{{table}}
"""
    return full_latex


def generate_summary_metrics_table(
    df: pd.DataFrame = None,
    output_dir: str = None,
) -> Tuple[str, str]:
    """
    Generate summary metrics table.
    
    Columns: q, Method, Steps to Conv., Gate-on Ratio, First Gate Step, Final Exploit.
    
    Returns:
        (csv_path, tex_path)
    """
    ensure_tables_dir()
    
    if df is None:
        df = load_all_convergence_data()
    if output_dir is None:
        output_dir = TABLES_DIR
    
    # Compute summary metrics
    metrics = compute_summary_metrics(df)
    metrics_df = summary_metrics_to_dataframe(metrics)
    
    # Select and rename columns for paper
    table_df = metrics_df[[
        "method", "q", "seed", "ablation",
        "convergence_step", "gate_on_ratio", "first_gate_step",
        "final_exploitability", "abs_error", "quality"
    ]].copy()
    
    # Rename for readability
    table_df = table_df.rename(columns={
        "convergence_step": "Conv. Step",
        "gate_on_ratio": "Gate-on Ratio",
        "first_gate_step": "First Gate",
        "final_exploitability": "Final Exploit.",
        "abs_error": "|e - e*|",
    })
    
    # Sort
    table_df = table_df.sort_values(["method", "q", "seed", "ablation"])
    
    # Save CSV
    csv_path = os.path.join(output_dir, "summary_metrics.csv")
    table_df.to_csv(csv_path, index=False)
    
    # Save LaTeX
    tex_path = os.path.join(output_dir, "summary_metrics.tex")
    latex_str = _to_latex_table(
        table_df,
        caption="Summary metrics for all experiment runs.",
        label="tab:summary_metrics",
    )
    with open(tex_path, 'w') as f:
        f.write(latex_str)
    
    print(f"[tables] Saved summary metrics to {csv_path}")
    print(f"[tables] Saved summary metrics to {tex_path}")
    
    return csv_path, tex_path


def generate_ablation_table(
    df: pd.DataFrame = None,
    output_dir: str = None,
) -> Tuple[str, str]:
    """
    Generate ablation results table.
    
    Columns: Ablation, q, Conv. Steps, Final Gap, Exploitability
    
    Returns:
        (csv_path, tex_path)
    """
    ensure_tables_dir()
    
    if df is None:
        df = load_all_convergence_data()
    if output_dir is None:
        output_dir = TABLES_DIR
    
    # Compute summary metrics
    metrics = compute_summary_metrics(df)
    metrics_df = summary_metrics_to_dataframe(metrics)
    
    # Filter to TEL-PPO only (ablations are for PPO)
    ppo_df = metrics_df[metrics_df["method"].isin(["TEL-PPO", "PPO"])].copy()
    
    if ppo_df.empty:
        print("[tables] Warning: No PPO data for ablation table")
        return None, None
    
    # Group by ablation and q, aggregate across seeds
    grouped = ppo_df.groupby(["ablation", "q"]).agg({
        "convergence_step": ["mean", "std"],
        "abs_error": ["mean", "std"],
        "final_exploitability": ["mean", "std"],
        "seed": "count",
    })
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    # Format for table
    table_df = pd.DataFrame({
        "Ablation": grouped["ablation"],
        "q": grouped["q"],
        "Conv. Steps": grouped["convergence_step_mean"].apply(lambda x: _format_float(x, 0)),
        "Final Gap": grouped["abs_error_mean"].apply(lambda x: _format_float(x, 3)),
        "Exploitability": grouped["final_exploitability_mean"].apply(lambda x: _format_float(x, 4)),
        "N": grouped["seed_count"],
    })
    
    # Sort by Ablation and q columns (the renamed columns)
    table_df = table_df.sort_values(["Ablation", "q"])
    
    # Save CSV
    csv_path = os.path.join(output_dir, "ablation_results.csv")
    table_df.to_csv(csv_path, index=False)
    
    # Save LaTeX
    tex_path = os.path.join(output_dir, "ablation_results.tex")
    latex_str = _to_latex_table(
        table_df,
        caption="Ablation study results comparing baseline, no-cheap-gate, and no-exploitability variants.",
        label="tab:ablation_results",
    )
    with open(tex_path, 'w') as f:
        f.write(latex_str)
    
    print(f"[tables] Saved ablation results to {csv_path}")
    print(f"[tables] Saved ablation results to {tex_path}")
    
    return csv_path, tex_path


def generate_final_paper_table(
    df: pd.DataFrame = None,
    output_dir: str = None,
) -> Tuple[str, str]:
    """
    Generate final summary table for paper (Table 2).

    Covers all experiments. For each scenario/q, shows Theory, Gradient, TEL-PPO.

    Returns:
        (csv_path, tex_path)
    """
    ensure_tables_dir()

    if df is None:
        df = load_all_convergence_data()
    if output_dir is None:
        output_dir = TABLES_DIR

    rows = []

    experiments = ["two_players", "three_players", "different_cost", "different_ability"]
    exp_labels = {
        "two_players": "Two-Player",
        "three_players": "Three-Player",
        "different_cost": "Het. Cost",
        "different_ability": "Het. Ability",
    }

    for experiment in experiments:
        if "experiment" in df.columns:
            exp_df = df[df["experiment"] == experiment]
        else:
            exp_df = df if experiment == "two_players" else pd.DataFrame()

        if exp_df.empty:
            continue

        q_values = sorted(exp_df["q"].unique())

        for q in q_values:
            q_df = exp_df[exp_df["q"] == q]

            # Get theoretical effort from data (handles heterogeneous)
            ppo_baseline = q_df[(q_df["method"].isin(["TEL-PPO", "PPO"])) & (q_df["ablation"] == "baseline")]
            if not ppo_baseline.empty:
                e_theory = ppo_baseline["theoretical_effort"].iloc[0]
            else:
                e_theory = e_star(q, **THEORY_PARAMS)

            # Theory row
            rows.append({
                "Scenario": exp_labels.get(experiment, experiment),
                "Method": "Theory",
                "q": q,
                "Effort": e_theory,
                "|e - e*|": 0.0,
                "Exploitability": 0.0,
                "Symmetric Gap": 0.0,
            })

            # Gradient
            grad_df = q_df[q_df["method"] == "Gradient"]
            if not grad_df.empty:
                final = get_final_values(grad_df)
                if not final.empty:
                    rows.append({
                        "Scenario": exp_labels.get(experiment, experiment),
                        "Method": "Gradient",
                        "q": q,
                        "Effort": final["effort_mean"].iloc[0],
                        "|e - e*|": final["effort_error"].iloc[0],
                        "Exploitability": final.get("exploitability_final", pd.Series([np.nan])).iloc[0],
                        "Symmetric Gap": final["symmetry_gap"].iloc[0],
                    })

            # TEL-PPO (baseline ablation)
            if not ppo_baseline.empty:
                final = get_final_values(ppo_baseline)
                if not final.empty:
                    rows.append({
                        "Scenario": exp_labels.get(experiment, experiment),
                        "Method": "TEL-PPO",
                        "q": q,
                        "Effort": final["effort_mean"].mean(),
                        "|e - e*|": final["effort_error"].mean(),
                        "Exploitability": final.get("exploitability_final", pd.Series([np.nan])).mean(),
                        "Symmetric Gap": final["symmetry_gap"].mean(),
                    })

    table_df = pd.DataFrame(rows)

    # Format numbers
    table_df["Effort"] = table_df["Effort"].apply(lambda x: _format_float(x, 2))
    table_df["|e - e*|"] = table_df["|e - e*|"].apply(lambda x: _format_float(x, 3))
    table_df["Exploitability"] = table_df["Exploitability"].apply(lambda x: _format_float(x, 4))
    table_df["Symmetric Gap"] = table_df["Symmetric Gap"].apply(lambda x: _format_float(x, 3))

    # Save CSV
    csv_path = os.path.join(output_dir, "final_summary.csv")
    table_df.to_csv(csv_path, index=False)

    # Save LaTeX
    tex_path = os.path.join(output_dir, "final_summary.tex")
    latex_str = _to_latex_table(
        table_df,
        caption="Quantitative summary across all scenarios: learned effort, gap from equilibrium, exploitability, and symmetry gap.",
        label="tab:final_summary",
    )
    with open(tex_path, "w") as f:
        f.write(latex_str)

    print(f"[tables] Saved final summary to {csv_path}")
    print(f"[tables] Saved final summary to {tex_path}")

    return csv_path, tex_path


def generate_convergence_comparison_table(
    df: pd.DataFrame = None,
    output_dir: str = None,
) -> Tuple[str, str]:
    """
    Generate convergence comparison table across methods.
    
    Shows convergence steps and quality for each (method, q) combination.
    """
    ensure_tables_dir()
    
    if df is None:
        df = load_all_convergence_data()
    if output_dir is None:
        output_dir = TABLES_DIR
    
    metrics = compute_summary_metrics(df)
    metrics_df = summary_metrics_to_dataframe(metrics)
    
    # Pivot table: rows = q, columns = method
    # Filter to baseline only
    baseline_df = metrics_df[metrics_df["ablation"] == "baseline"].copy()
    
    # Group by method and q, average across seeds
    grouped = baseline_df.groupby(["method", "q"]).agg({
        "convergence_step": "mean",
        "abs_error": "mean",
        "quality": lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown",
    }).reset_index()
    
    # Pivot
    pivot_steps = grouped.pivot(index="q", columns="method", values="convergence_step")
    pivot_error = grouped.pivot(index="q", columns="method", values="abs_error")
    pivot_quality = grouped.pivot(index="q", columns="method", values="quality")
    
    # Combine into single table
    table_rows = []
    for q in sorted(grouped["q"].unique()):
        row = {"q": q}
        for method in ["Gradient", "TEL-PPO"]:
            if method in pivot_steps.columns:
                steps = pivot_steps.loc[q, method] if q in pivot_steps.index else np.nan
                error = pivot_error.loc[q, method] if q in pivot_error.index else np.nan
                quality = pivot_quality.loc[q, method] if q in pivot_quality.index else "N/A"
                row[f"{method} Conv."] = _format_float(steps, 0)
                row[f"{method} Gap"] = _format_float(error, 3)
                row[f"{method} Quality"] = quality
        table_rows.append(row)
    
    table_df = pd.DataFrame(table_rows)
    
    # Save CSV
    csv_path = os.path.join(output_dir, "convergence_comparison.csv")
    table_df.to_csv(csv_path, index=False)
    
    # Save LaTeX
    tex_path = os.path.join(output_dir, "convergence_comparison.tex")
    latex_str = _to_latex_table(
        table_df,
        caption="Convergence comparison: steps to convergence and final gap for each method.",
        label="tab:convergence_comparison",
    )
    with open(tex_path, 'w') as f:
        f.write(latex_str)
    
    print(f"[tables] Saved convergence comparison to {csv_path}")
    print(f"[tables] Saved convergence comparison to {tex_path}")
    
    return csv_path, tex_path


def generate_environment_config_table(
    output_dir: str = None,
) -> Tuple[str, str]:
    """
    Generate environment configuration table (Table 1).

    Static table listing game parameters, training hyperparameters,
    and convergence criteria.

    Returns:
        (csv_path, tex_path)
    """
    ensure_tables_dir()

    if output_dir is None:
        output_dir = TABLES_DIR

    rows = [
        {"Category": "Game", "Parameter": "$w_H$ (high prize)", "Value": str(THEORY_PARAMS["w_h"])},
        {"Category": "Game", "Parameter": "$w_L$ (low prize)", "Value": str(THEORY_PARAMS["w_l"])},
        {"Category": "Game", "Parameter": "$k$ (cost coeff.)", "Value": str(THEORY_PARAMS["k"])},
        {"Category": "Game", "Parameter": "$q$ (noise)", "Value": "\\{25, 40, 55\\}"},
        {"Category": "Game", "Parameter": "Effort bounds", "Value": "[0, 250]"},
        {"Category": "Game", "Parameter": "Noise dist.", "Value": "$\\text{Uniform}[-q, q]$"},
        {"Category": "Training", "Parameter": "Algorithm", "Value": "PPO (clip)"},
        {"Category": "Training", "Parameter": "Episodes per update", "Value": "4096"},
        {"Category": "Training", "Parameter": "Max updates", "Value": "500"},
        {"Category": "Training", "Parameter": "PPO clip $\\epsilon$", "Value": "0.2"},
        {"Category": "Training", "Parameter": "Learning rate", "Value": "$3 \\times 10^{-4}$"},
        {"Category": "Training", "Parameter": "Policy", "Value": "Beta distribution"},
        {"Category": "Convergence", "Parameter": "Effort $\\delta$", "Value": str(CONVERGENCE_CONFIG["effort_delta"])},
        {"Category": "Convergence", "Parameter": "Effort window", "Value": str(CONVERGENCE_CONFIG["effort_window"])},
        {"Category": "Convergence", "Parameter": "Exploit. threshold $\\varepsilon$", "Value": str(CONVERGENCE_CONFIG["exploit_threshold"])},
        {"Category": "Convergence", "Parameter": "Exploit. patience", "Value": str(CONVERGENCE_CONFIG["exploit_patience"])},
    ]

    # Compute theoretical equilibrium for each q
    e_star_str = ", ".join(
        f"$e^*(q={int(q)})={e_star(q, **THEORY_PARAMS):.2f}$" for q in Q_VALUES
    )
    rows.append({"Category": "Theory", "Parameter": "$e^* = (w_H - w_L) / (4qk)$", "Value": e_star_str})
    table_df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(output_dir, "environment_config.csv")
    table_df.to_csv(csv_path, index=False)

    # Save LaTeX
    tex_path = os.path.join(output_dir, "environment_config.tex")
    latex_str = _to_latex_table(
        table_df,
        caption="Environment and training configuration.",
        label="tab:environment_config",
    )
    with open(tex_path, "w") as f:
        f.write(latex_str)

    print(f"[tables] Saved environment config to {csv_path}")
    print(f"[tables] Saved environment config to {tex_path}")

    return csv_path, tex_path


def generate_all_tables(
    df: pd.DataFrame = None,
    output_dir: str = None,
) -> Dict[str, Tuple[str, str]]:
    """
    Generate all tables for the paper.
    
    Returns dict mapping table name to (csv_path, tex_path).
    """
    if df is None:
        df = load_all_convergence_data()
    
    results = {}

    # Environment config (static, no data needed)
    paths = generate_environment_config_table(output_dir)
    if paths[0]:
        results["environment_config"] = paths

    # Summary metrics
    paths = generate_summary_metrics_table(df, output_dir)
    if paths[0]:
        results["summary_metrics"] = paths

    # Ablation results
    paths = generate_ablation_table(df, output_dir)
    if paths and paths[0]:
        results["ablation_results"] = paths

    # Final summary
    paths = generate_final_paper_table(df, output_dir)
    if paths[0]:
        results["final_summary"] = paths

    # Convergence comparison
    paths = generate_convergence_comparison_table(df, output_dir)
    if paths[0]:
        results["convergence_comparison"] = paths

    return results


if __name__ == "__main__":
    print("Generating all tables...")
    results = generate_all_tables()
    print("\nGenerated tables:")
    for name, (csv_path, tex_path) in results.items():
        print(f"  {name}:")
        print(f"    CSV: {csv_path}")
        print(f"    LaTeX: {tex_path}")
