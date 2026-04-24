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
    e_star_for_experiment,
    THEORY_PARAMS,
    get_theory_params,
    get_q_values,
    EXPERIMENT_Q_VALUES,
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


def _classify_failure_mode(row: pd.Series) -> str:
    """Classify failure mode for ablation table."""
    if row.get("converged_ratio", 0) >= 0.8:
        return "-"
    abs_err = row.get("abs_error_mean", np.nan)
    exploit = row.get("final_exploitability_mean", np.nan)
    conv_step = row.get("convergence_step_mean", np.nan)

    if np.isnan(conv_step) or conv_step == 0:
        # Never converged
        if not np.isnan(abs_err) and abs_err > 10:
            return "diverge"
        if not np.isnan(exploit) and exploit > 0.5:
            return "cycle"
        return "never terminate"
    if not np.isnan(abs_err) and abs_err > 2:
        return "biased mean"
    return "-"


def generate_ablation_table(
    df: pd.DataFrame = None,
    output_dir: str = None,
) -> Tuple[str, str]:
    """
    Generate ablation results table (Table 3).

    Rows: TEL-PPO (baseline) | No cheap gate | No exploitability gate
    Columns: Ablation | Final Error | Exploitability | Steps to Conv. | Failure Mode

    Aggregated across all q values and seeds.

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

    # Filter to TEL-PPO, two_players experiment only
    ppo_df = metrics_df[metrics_df["method"].isin(["TEL-PPO", "PPO"])].copy()
    if "experiment" in ppo_df.columns:
        ppo_df = ppo_df[ppo_df["experiment"] == "two_players"]

    if ppo_df.empty:
        print("[tables] Warning: No PPO data for ablation table")
        return None, None

    # Map ablation names to display labels
    ablation_labels = {
        "baseline": "TEL-PPO (baseline)",
        "no_cheap_gate": "No stability gate",
        "no_exploitability": "No exploitability gate",
        "no_entropy": "+Entropy disabled",
    }

    # Group by ablation, aggregate across q and seeds
    grouped = ppo_df.groupby("ablation").agg(
        abs_error_mean=("abs_error", "mean"),
        abs_error_std=("abs_error", "std"),
        final_exploitability_mean=("final_exploitability", "mean"),
        final_exploitability_std=("final_exploitability", "std"),
        convergence_step_mean=("convergence_step", "mean"),
        convergence_step_std=("convergence_step", "std"),
        converged_ratio=("converged", "mean"),
        n_runs=("seed", "count"),
    ).reset_index()

    # Build table rows
    rows = []
    # Ensure baseline comes first
    ablation_order = ["baseline", "no_cheap_gate", "no_exploitability"]
    for abl in ablation_order:
        match = grouped[grouped["ablation"] == abl]
        if match.empty:
            continue
        r = match.iloc[0]
        failure = _classify_failure_mode(r)
        rows.append({
            "Ablation": ablation_labels.get(abl, abl),
            "Final Error": f"{_format_float(r['abs_error_mean'], 3)}±{_format_float(r['abs_error_std'], 3)}",
            "Exploitability": f"{_format_float(r['final_exploitability_mean'], 4)}±{_format_float(r['final_exploitability_std'], 4)}",
            "Steps to Conv.": _format_float(r["convergence_step_mean"], 0),
            "Failure Mode": failure,
        })

    # Also include any other ablations not in the standard order
    for _, r in grouped.iterrows():
        if r["ablation"] not in ablation_order:
            failure = _classify_failure_mode(r)
            rows.append({
                "Ablation": ablation_labels.get(r["ablation"], r["ablation"]),
                "Final Error": f"{_format_float(r['abs_error_mean'], 3)}±{_format_float(r['abs_error_std'], 3)}",
                "Exploitability": f"{_format_float(r['final_exploitability_mean'], 4)}±{_format_float(r['final_exploitability_std'], 4)}",
                "Steps to Conv.": _format_float(r["convergence_step_mean"], 0),
                "Failure Mode": failure,
            })

    table_df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(output_dir, "ablation_results.csv")
    table_df.to_csv(csv_path, index=False)

    # Save LaTeX
    tex_path = os.path.join(output_dir, "ablation_results.tex")
    latex_str = _to_latex_table(
        table_df,
        caption="Ablation study: effect of removing TEL-PPO components on convergence quality.",
        label="tab:ablation_results",
    )
    with open(tex_path, "w") as f:
        f.write(latex_str)

    print(f"[tables] Saved ablation results to {csv_path}")
    print(f"[tables] Saved ablation results to {tex_path}")

    return csv_path, tex_path


def _compute_error_metrics(final_df: pd.DataFrame, e_theory_avg: float) -> Tuple[float, float]:
    """Compute (abs_err, rel_err%) aggregated across seeds.

    For heterogeneous scenarios with distinct per-agent equilibria
    (theoretical_effort1 != theoretical_effort2, e.g. `different_cost`), uses the
    per-seed max-across-agents gap — i.e. `max(|a1-e1*|, |a2-e2*|)` for abs_err
    and `max(|a1-e1*|/e1*, |a2-e2*|/e2*)` for rel_err, then mean across seeds.
    This matches the report's definition (see
    `docs/round3_round4_report.md` §3.3) and correctly reports the worst
    per-agent distance to NE rather than the symmetric-average which can
    mask agent-level misfit.

    For symmetric scenarios (agents share e*), falls back to the canonical
    avg-based `|ē−e*|/e*` computation.
    """
    has_per_agent_theory = (
        "theoretical_effort1" in final_df.columns
        and "theoretical_effort2" in final_df.columns
        and not final_df["theoretical_effort1"].isna().all()
        and not final_df["theoretical_effort2"].isna().all()
    )
    if has_per_agent_theory:
        gap1 = np.abs(final_df["agent1_effort"] - final_df["theoretical_effort1"])
        gap2 = np.abs(final_df["agent2_effort"] - final_df["theoretical_effort2"])
        rel1 = gap1 / final_df["theoretical_effort1"]
        rel2 = gap2 / final_df["theoretical_effort2"]
        abs_err = float(np.maximum(gap1, gap2).mean())
        rel_err = float(np.maximum(rel1, rel2).mean() * 100)
    else:
        abs_err = float(final_df["effort_error"].mean())
        rel_err = (abs_err / e_theory_avg * 100) if e_theory_avg > 0 else float("nan")
    return abs_err, rel_err


def generate_final_paper_table(
    df: pd.DataFrame = None,
    output_dir: str = None,
) -> Tuple[str, str]:
    """
    Generate final summary table for paper (Table 2).

    Columns: Scenario | q | Method | Mean±std | |ē−e*| | Exploitability | Symmetry Gap | Steps to Convergence

    Covers all experiments. For each scenario/q, shows Theory, Gradient, TEL-PPO.

    Returns:
        (csv_path, tex_path)
    """
    ensure_tables_dir()

    if df is None:
        df = load_all_convergence_data()
    if output_dir is None:
        output_dir = TABLES_DIR

    # Compute convergence steps
    from .extract import get_convergence_step
    conv_df = get_convergence_step(df)

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

        # Filter to baseline weight variant (exclude Set 2 / wh8_wl4)
        if "weight_variant" in exp_df.columns:
            exp_df = exp_df[exp_df["weight_variant"] == "baseline"]

        if exp_df.empty:
            continue

        # Use primary q values only (per-experiment q sets)
        exp_q_values = get_q_values(experiment)
        q_values_exp = sorted(q for q in exp_df["q"].unique() if q in exp_q_values)

        for q in q_values_exp:
            q_df = exp_df[exp_df["q"] == q]

            # Get theoretical effort from data (handles heterogeneous)
            ppo_baseline = q_df[(q_df["method"].isin(["TEL-PPO", "PPO"])) & (q_df["ablation"] == "baseline")]
            if not ppo_baseline.empty:
                e_theory = ppo_baseline["theoretical_effort"].iloc[0]
            else:
                e_theory = e_star_for_experiment(q, experiment)

            # Theory row
            rows.append({
                "Scenario": exp_labels.get(experiment, experiment),
                "q": int(q),
                "Method": "Theory",
                "Mean±std": _format_float(e_theory, 2),
                "|ē−e*|": "0.00",
                "RelErr": "0.00%",
                "Exploitability": "0.000",
                "Symmetry Gap": "0.00",
                "Steps to Conv.": "-",
            })

            # Gradient (baseline ablation only, exclude weight variants)
            grad_df = q_df[(q_df["method"] == "Gradient") & (q_df["ablation"] == "baseline")]
            if not grad_df.empty:
                final = get_final_values(grad_df)
                if not final.empty:
                    effort_mean = final["policy_mean_effort"].mean()
                    effort_std = final["policy_mean_effort"].std() if len(final) > 1 else 0
                    abs_err, rel_err = _compute_error_metrics(final, e_theory)
                    rows.append({
                        "Scenario": exp_labels.get(experiment, experiment),
                        "q": int(q),
                        "Method": "Gradient",
                        "Mean±std": f"{effort_mean:.2f}±{effort_std:.2f}",
                        "|ē−e*|": _format_float(abs_err, 2),
                        "RelErr": f"{rel_err:.2f}%",
                        "Exploitability": _format_float(final.get("exploitability_final", pd.Series([np.nan])).mean(), 3),
                        "Symmetry Gap": _format_float(final["symmetry_gap"].mean(), 2),
                        "Steps to Conv.": "-",
                    })

            # TEL-PPO (baseline ablation)
            if not ppo_baseline.empty:
                final = get_final_values(ppo_baseline)
                if not final.empty:
                    effort_mean = final["policy_mean_effort"].mean()
                    effort_std = final["policy_mean_effort"].std() if len(final) > 1 else 0
                    # Get convergence steps
                    conv_match = conv_df[
                        (conv_df["q"] == q)
                        & (conv_df["ablation"] == "baseline")
                        & (conv_df["method"].isin(["TEL-PPO", "PPO"]))
                    ]
                    if "experiment" in conv_df.columns:
                        conv_match = conv_match[conv_match["experiment"] == experiment]
                    mean_conv = conv_match["convergence_step"].dropna().mean()
                    conv_str = _format_float(mean_conv, 0) if not np.isnan(mean_conv) else "NC"

                    abs_err, rel_err = _compute_error_metrics(final, e_theory)
                    rows.append({
                        "Scenario": exp_labels.get(experiment, experiment),
                        "q": int(q),
                        "Method": "TEL-PPO",
                        "Mean±std": f"{effort_mean:.2f}±{effort_std:.2f}",
                        "|ē−e*|": _format_float(abs_err, 2),
                        "RelErr": f"{rel_err:.2f}%",
                        "Exploitability": _format_float(final.get("exploitability_final", pd.Series([np.nan])).mean(), 3),
                        "Symmetry Gap": _format_float(final["symmetry_gap"].mean(), 2),
                        "Steps to Conv.": conv_str,
                    })

    table_df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(output_dir, "final_summary.csv")
    table_df.to_csv(csv_path, index=False)

    # Save LaTeX
    tex_path = os.path.join(output_dir, "final_summary.tex")
    latex_str = _to_latex_table(
        table_df,
        caption="Quantitative summary of equilibrium recovery across all scenarios.",
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
    # Filter to baseline and standard q values only
    # Use all known q values across experiments
    all_q = set()
    for qv in EXPERIMENT_Q_VALUES.values():
        all_q.update(qv)
    baseline_df = metrics_df[
        (metrics_df["ablation"] == "baseline") & (metrics_df["q"].isin(all_q))
    ].copy()
    
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
        # Game parameters
        {"Category": "Game", "Parameter": "$w_H$ (high prize)", "Value": "6.5 / 8"},
        {"Category": "Game", "Parameter": "$w_L$ (low prize)", "Value": "3.0 / 4 / 5.5"},
        {"Category": "Game", "Parameter": "$k$ (cost coeff.)", "Value": "0.00055 / 0.001 / 0.0005"},
        {"Category": "Game", "Parameter": "$q$ (noise)", "Value": "\\{35, 45, 55\\} / \\{35, 55\\}"},
        {"Category": "Game", "Parameter": "Effort bounds", "Value": "[0, 100]"},
        {"Category": "Game", "Parameter": "Noise distribution", "Value": "$\\varepsilon_i \\sim \\text{Uniform}[-q, q]$"},
        {"Category": "Game", "Parameter": "Number of players", "Value": "2 (baseline) / 3"},
        # Training hyperparameters
        {"Category": "Training", "Parameter": "Algorithm", "Value": "PPO (clip)"},
        {"Category": "Training", "Parameter": "Learning rate", "Value": "$3 \\times 10^{-4}$"},
        {"Category": "Training", "Parameter": "PPO clip $\\epsilon$", "Value": "0.2"},
        {"Category": "Training", "Parameter": "Batch size (episodes/update)", "Value": "4096"},
        {"Category": "Training", "Parameter": "Max updates", "Value": "500"},
        {"Category": "Training", "Parameter": "Policy parameterization", "Value": "Beta($\\alpha$, $\\beta$)"},
        {"Category": "Training", "Parameter": "Seeds per configuration", "Value": "5"},
        # Convergence criteria
        {"Category": "Convergence", "Parameter": "Effort $\\delta$", "Value": str(CONVERGENCE_CONFIG["effort_delta"])},
        {"Category": "Convergence", "Parameter": "Effort window", "Value": str(int(CONVERGENCE_CONFIG["effort_window"]))},
        {"Category": "Convergence", "Parameter": "Exploit. threshold $\\varepsilon$", "Value": str(CONVERGENCE_CONFIG["exploit_threshold"])},
        {"Category": "Convergence", "Parameter": "Exploit. patience", "Value": str(int(CONVERGENCE_CONFIG["exploit_patience"]))},
        {"Category": "Convergence", "Parameter": "Min steps", "Value": str(int(CONVERGENCE_CONFIG["min_steps"]))},
    ]

    # Compute theoretical equilibrium for each q (two_players Set 1)
    e_star_str = ", ".join(
        f"$e^*(q={int(q)})={e_star(q, **THEORY_PARAMS):.2f}$" for q in Q_VALUES
    )
    rows.append({"Category": "Theory", "Parameter": "$e^* = (w_H - w_L) / (4qk)$ (Set 1)", "Value": e_star_str})
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
