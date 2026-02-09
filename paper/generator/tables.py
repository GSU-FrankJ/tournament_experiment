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
    Generate final summary table for paper.
    
    Columns: Method, q, Effort, |e - e*|, Exploitability, Symmetric Gap
    
    Includes Theory, Gradient, and TEL-PPO.
    
    Returns:
        (csv_path, tex_path)
    """
    ensure_tables_dir()
    
    if df is None:
        df = load_all_convergence_data()
    if output_dir is None:
        output_dir = TABLES_DIR
    
    rows = []
    
    # Get q values from data
    q_values = sorted(df["q"].unique())
    
    for q in q_values:
        e_theory = e_star(q, **THEORY_PARAMS)
        
        # Theory row
        rows.append({
            "Method": "Theory",
            "q": q,
            "Effort": e_theory,
            "|e - e*|": 0.0,
            "Exploitability": 0.0,
            "Symmetric Gap": 0.0,
        })
        
        # Gradient
        grad_df = df[(df["q"] == q) & (df["method"] == "Gradient")]
        if not grad_df.empty:
            final = get_final_values(grad_df)
            if not final.empty:
                rows.append({
                    "Method": "Gradient",
                    "q": q,
                    "Effort": final["effort_mean"].iloc[0],
                    "|e - e*|": final["effort_error"].iloc[0],
                    "Exploitability": final.get("exploitability_final", pd.Series([np.nan])).iloc[0],
                    "Symmetric Gap": final["symmetry_gap"].iloc[0],
                })
        
        # TEL-PPO (baseline ablation)
        ppo_df = df[(df["q"] == q) & (df["method"].isin(["TEL-PPO", "PPO"])) & (df["ablation"] == "baseline")]
        if not ppo_df.empty:
            final = get_final_values(ppo_df)
            if not final.empty:
                # Average across seeds
                rows.append({
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
        caption="Final comparison of theoretical equilibrium, gradient descent, and TEL-PPO methods.",
        label="tab:final_summary",
    )
    with open(tex_path, 'w') as f:
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
