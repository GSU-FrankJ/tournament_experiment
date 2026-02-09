"""
Data Extraction: Load convergence JSON and CSV into tidy DataFrames.

Handles:
- Legacy format (effort only)
- New format (effort + time-series metrics)
- Sparse exploitability (NaN when not evaluated)
- Multiple runs aggregation
"""

import json
import warnings
from typing import List, Dict, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd

from .run_registry import Run, discover_runs
from .config import CONVERGENCE_DIR, CSV_PATH, e_star, THEORY_PARAMS


# Column schema for convergence DataFrame
CONVERGENCE_COLUMNS = [
    # Identifiers
    "step",
    "method", 
    "q",
    "seed",
    "ablation",
    # Effort (always present)
    "effort_mean",      # Average of agent1 and agent2
    "agent1_effort",
    "agent2_effort",
    "policy_mean_effort",
    # Theory reference
    "theoretical_effort",
    # Training metrics (present in new format, NaN in legacy)
    "approx_kl",
    "batch_entropy",
    "alpha_mean",
    "beta_mean",
    "mean_kl_window",
    "drift_effort",
    # Exploitability (sparse: NaN when not evaluated)
    "exploitability",
    "exploitability_is_valid",
]


def load_convergence_json(run: Run) -> pd.DataFrame:
    """
    Load convergence history from a single run's JSON file.
    
    Args:
        run: Run object with path to convergence JSON
    
    Returns:
        DataFrame with step-by-step convergence data
    """
    if not run.path or not Path(run.path).exists():
        warnings.warn(f"No convergence file for run: {run}")
        return pd.DataFrame(columns=CONVERGENCE_COLUMNS)
    
    with open(run.path, 'r') as f:
        data = json.load(f)
    
    # Get number of steps
    steps = data.get("steps", [])
    n_steps = len(steps)
    
    if n_steps == 0:
        return pd.DataFrame(columns=CONVERGENCE_COLUMNS)
    
    # Build DataFrame
    df = pd.DataFrame({
        "step": steps,
        "method": run.method,
        "q": run.q,
        "seed": run.seed,
        "ablation": run.ablation,
        # Effort series
        "agent1_effort": data.get("agent1_effort", [np.nan] * n_steps),
        "agent2_effort": data.get("agent2_effort", [np.nan] * n_steps),
        "policy_mean_effort": data.get("policy_mean_effort", [np.nan] * n_steps),
    })
    
    # Compute effort_mean as average of agent1 and agent2
    df["effort_mean"] = (df["agent1_effort"] + df["agent2_effort"]) / 2.0
    
    # Add theoretical effort
    df["theoretical_effort"] = e_star(run.q, **THEORY_PARAMS)
    
    # Time-series metrics (new format only)
    for col in ["approx_kl", "batch_entropy", "alpha_mean", "beta_mean", 
                "mean_kl_window", "drift_effort"]:
        if col in data and isinstance(data[col], list) and len(data[col]) == n_steps:
            df[col] = data[col]
        else:
            df[col] = np.nan
    
    # Exploitability (sparse)
    df["exploitability"] = _load_exploitability_series(data, n_steps)
    df["exploitability_is_valid"] = _load_exploitability_valid_series(data, n_steps)
    
    return df


def _load_exploitability_series(data: Dict, n_steps: int) -> List[float]:
    """
    Load exploitability with proper NaN handling.
    
    Legacy format: single final value -> expand to series with NaN
    New format: list with NaN where not evaluated
    """
    exploit = data.get("exploitability", [])
    
    # Legacy format: single final value
    if isinstance(exploit, (int, float)):
        series = [np.nan] * n_steps
        if n_steps > 0:
            series[-1] = float(exploit) if not np.isnan(exploit) else np.nan
        return series
    
    # New format: list
    if isinstance(exploit, list) and len(exploit) == n_steps:
        return [float(e) if e is not None and not (isinstance(e, float) and np.isnan(e)) 
                else np.nan for e in exploit]
    
    # Fallback: all NaN
    return [np.nan] * n_steps


def _load_exploitability_valid_series(data: Dict, n_steps: int) -> List[bool]:
    """Load exploitability validity flags."""
    is_valid = data.get("exploitability_is_valid", [])
    
    # If not present, infer from exploitability values
    if not isinstance(is_valid, list) or len(is_valid) != n_steps:
        exploit = data.get("exploitability", [])
        
        if isinstance(exploit, (int, float)):
            # Legacy: only final value is valid
            series = [False] * n_steps
            if n_steps > 0 and not np.isnan(float(exploit)):
                series[-1] = True
            return series
        
        if isinstance(exploit, list) and len(exploit) == n_steps:
            # Infer validity from NaN
            return [not (e is None or (isinstance(e, float) and np.isnan(e))) 
                    for e in exploit]
        
        return [False] * n_steps
    
    return [bool(v) for v in is_valid]


def load_multiple_runs(runs: List[Run]) -> pd.DataFrame:
    """
    Load convergence data from multiple runs into a single DataFrame.
    
    Args:
        runs: List of Run objects
    
    Returns:
        Combined DataFrame with all runs
    """
    dfs = []
    
    for run in runs:
        if run.method == "Theory":
            continue  # Skip theory placeholder runs
        
        df = load_convergence_json(run)
        if not df.empty:
            dfs.append(df)
    
    if not dfs:
        return pd.DataFrame(columns=CONVERGENCE_COLUMNS)
    
    return pd.concat(dfs, ignore_index=True)


def load_all_convergence_data(
    convergence_dir: str = None,
    csv_path: str = None,
    q_values: Optional[List[float]] = None,
    methods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load all convergence data from discovered runs.
    
    Args:
        convergence_dir: Path to convergence_history directory
        csv_path: Path to results CSV for fallback info
        q_values: Filter to specific q values
        methods: Filter to specific methods
    
    Returns:
        Combined DataFrame with all matching runs
    """
    runs = discover_runs(convergence_dir, csv_path)
    
    # Filter
    if q_values is not None:
        runs = [r for r in runs if r.q in q_values]
    if methods is not None:
        methods_upper = [m.upper() for m in methods]
        runs = [r for r in runs if r.method.upper() in methods_upper or
                (r.method == "TEL-PPO" and "PPO" in methods_upper)]
    
    return load_multiple_runs(runs)


def add_theory_reference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add theoretical equilibrium effort as a reference line.
    
    For plotting, we need theory values at the same steps as data.
    """
    if df.empty:
        return df
    
    # Ensure theoretical_effort column exists and is correct
    df = df.copy()
    df["theoretical_effort"] = df["q"].apply(lambda q: e_star(q, **THEORY_PARAMS))
    
    return df


def forward_fill_exploitability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill exploitability values for plotting.
    
    Since exploitability is only evaluated at certain steps,
    this fills gaps with the last known value for smoother plots.
    """
    df = df.copy()
    df["exploitability_ffill"] = (
        df.groupby(["method", "q", "seed", "ablation"])["exploitability"]
        .ffill()
    )
    return df


def compute_symmetry_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Add column for |agent1_effort - agent2_effort|."""
    df = df.copy()
    df["symmetry_gap"] = np.abs(df["agent1_effort"] - df["agent2_effort"])
    return df


def compute_effort_error(df: pd.DataFrame) -> pd.DataFrame:
    """Add column for |effort_mean - theoretical_effort|."""
    df = df.copy()
    df["effort_error"] = np.abs(df["effort_mean"] - df["theoretical_effort"])
    return df


def get_final_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract final values for each run (method, q, seed, ablation).
    
    Returns one row per run with final effort, exploitability, etc.
    """
    if df.empty:
        return df
    
    # Group by run identifier and get last row
    grouped = df.groupby(["method", "q", "seed", "ablation"])
    
    final = grouped.agg({
        "step": "max",
        "effort_mean": "last",
        "agent1_effort": "last",
        "agent2_effort": "last",
        "policy_mean_effort": "last",
        "theoretical_effort": "last",
        "approx_kl": "last",
        "alpha_mean": "last",
        "beta_mean": "last",
    }).reset_index()
    
    # For exploitability, get last valid value
    def last_valid_exploit(group):
        valid = group[group["exploitability_is_valid"] == True]["exploitability"]
        return valid.iloc[-1] if len(valid) > 0 else np.nan
    
    exploit_df = grouped.apply(last_valid_exploit).reset_index(name="exploitability_final")
    final = final.merge(exploit_df, on=["method", "q", "seed", "ablation"])
    
    # Compute derived metrics
    final["symmetry_gap"] = np.abs(final["agent1_effort"] - final["agent2_effort"])
    final["effort_error"] = np.abs(final["effort_mean"] - final["theoretical_effort"])
    
    return final


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across seeds for multi-seed experiments.
    
    Returns mean and std for each (method, q, ablation, step) combination.
    """
    if df.empty:
        return df
    
    # Numeric columns to aggregate
    numeric_cols = [
        "effort_mean", "agent1_effort", "agent2_effort", "policy_mean_effort",
        "approx_kl", "batch_entropy", "alpha_mean", "beta_mean",
        "mean_kl_window", "drift_effort", "exploitability",
    ]
    
    # Group by (method, q, ablation, step) across seeds
    grouped = df.groupby(["method", "q", "ablation", "step"])
    
    # Compute mean and std for each numeric column
    agg_dict = {}
    for col in numeric_cols:
        if col in df.columns:
            agg_dict[f"{col}_mean"] = (col, "mean")
            agg_dict[f"{col}_std"] = (col, "std")
    
    agg_dict["n_seeds"] = ("seed", "nunique")
    agg_dict["theoretical_effort"] = ("theoretical_effort", "first")
    
    result = grouped.agg(**agg_dict).reset_index()
    
    # Compute 95% CI half-width: 1.96 * std / sqrt(n)
    for col in numeric_cols:
        if f"{col}_std" in result.columns:
            result[f"{col}_ci95"] = 1.96 * result[f"{col}_std"] / np.sqrt(result["n_seeds"])
    
    return result


def load_results_csv(csv_path: str = None) -> pd.DataFrame:
    """Load the main results CSV file."""
    if csv_path is None:
        csv_path = CSV_PATH
    
    if not Path(csv_path).exists():
        warnings.warn(f"Results CSV not found: {csv_path}")
        return pd.DataFrame()
    
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    # Quick test: load all convergence data
    print("Loading convergence data...")
    df = load_all_convergence_data()
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"\nUnique runs:")
    print(df.groupby(["method", "q", "seed", "ablation"]).size())
    
    # Test final values extraction
    print("\nFinal values:")
    final = get_final_values(df)
    print(final[["method", "q", "seed", "ablation", "effort_mean", "theoretical_effort", "effort_error"]])
