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
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd

from .run_registry import Run, discover_runs
from .config import CONVERGENCE_DIR, CSV_PATH, e_star, THEORY_PARAMS, CONVERGENCE_DIRS, CONVERGENCE_CONFIG


# Column schema for convergence DataFrame
CONVERGENCE_COLUMNS = [
    # Identifiers
    "step",
    "method",
    "q",
    "seed",
    "ablation",
    "experiment",
    # Effort (always present)
    "effort_mean",      # Average of agent1 and agent2
    "agent1_effort",
    "agent2_effort",
    "policy_mean_effort",
    # Theory reference
    "theoretical_effort",
    "theoretical_effort1",   # Per-agent (for heterogeneous scenarios)
    "theoretical_effort2",
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

    Handles flat format (two_players, three_players) and nested format
    (different_cost, different_ability with history dict).

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

    # Dispatch to appropriate loader
    if run.is_nested_format:
        return _load_nested_format(data, run)
    else:
        return _load_flat_format(data, run)


def _load_flat_format(data: Dict, run: Run) -> pd.DataFrame:
    """Load flat-format convergence JSON (two_players, three_players)."""
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
        "experiment": run.experiment,
        # Effort series
        "agent1_effort": data.get("agent1_effort", [np.nan] * n_steps),
        "agent2_effort": data.get("agent2_effort", [np.nan] * n_steps),
        "policy_mean_effort": data.get("policy_mean_effort", [np.nan] * n_steps),
    })

    # Compute effort_mean as average of agent1 and agent2
    df["effort_mean"] = (df["agent1_effort"] + df["agent2_effort"]) / 2.0

    # Add theoretical effort — prefer values from the JSON itself over formula
    theoretical = data.get("theoretical", {})
    if "effort1" in theoretical and "effort2" in theoretical:
        theo_e1 = theoretical["effort1"]
        theo_e2 = theoretical["effort2"]
        df["theoretical_effort"] = (theo_e1 + theo_e2) / 2.0
        df["theoretical_effort1"] = theo_e1
        df["theoretical_effort2"] = theo_e2
    elif "effort" in theoretical:
        df["theoretical_effort"] = theoretical["effort"]
        df["theoretical_effort1"] = np.nan
        df["theoretical_effort2"] = np.nan
    elif "theoretical_effort" in data:
        # Top-level theoretical_effort (gradient format)
        df["theoretical_effort"] = float(data["theoretical_effort"])
        df["theoretical_effort1"] = np.nan
        df["theoretical_effort2"] = np.nan
    else:
        df["theoretical_effort"] = e_star(run.q, **THEORY_PARAMS)
        df["theoretical_effort1"] = np.nan
        df["theoretical_effort2"] = np.nan

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


def _load_nested_format(data: Dict, run: Run) -> pd.DataFrame:
    """Load nested-format convergence JSON (different_cost, different_ability)."""
    history = data.get("history", {})
    steps = history.get("steps", [])
    n_steps = len(steps)

    if n_steps == 0:
        return pd.DataFrame(columns=CONVERGENCE_COLUMNS)

    scenario = data.get("scenario", run.experiment)

    # Extract efforts based on scenario
    if scenario == "different_cost":
        agent1_effort = history.get("agent1_effort", [np.nan] * n_steps)
        agent2_effort = history.get("agent2_effort", [np.nan] * n_steps)
        # Per-agent theoretical values
        theoretical = data.get("theoretical", {})
        theo_e1 = theoretical.get("effort1", np.nan)
        theo_e2 = theoretical.get("effort2", np.nan)
        # Aggregate approx_kl from per-agent
        kl1 = history.get("approx_kl_agent1", [np.nan] * n_steps)
        kl2 = history.get("approx_kl_agent2", [np.nan] * n_steps)
        approx_kl = [
            np.nanmean([float(k1) if k1 is not None else np.nan,
                        float(k2) if k2 is not None else np.nan])
            for k1, k2 in zip(kl1, kl2)
        ]
        ent1 = history.get("batch_entropy_agent1", [np.nan] * n_steps)
        ent2 = history.get("batch_entropy_agent2", [np.nan] * n_steps)
        batch_entropy = [
            np.nanmean([float(e1) if e1 is not None else np.nan,
                        float(e2) if e2 is not None else np.nan])
            for e1, e2 in zip(ent1, ent2)
        ]
    elif scenario == "different_ability":
        # Single "effort" field (shared policy), both agents same effort
        effort = history.get("effort", [np.nan] * n_steps)
        agent1_effort = effort
        agent2_effort = effort
        theoretical = data.get("theoretical", {})
        # different_ability: theoretical.effort is the shared equilibrium
        theo_e1 = theoretical.get("effort", np.nan)
        theo_e2 = theo_e1
        approx_kl = history.get("approx_kl", [np.nan] * n_steps)
        batch_entropy = history.get("batch_entropy", [np.nan] * n_steps)
    else:
        # Fallback
        agent1_effort = history.get("agent1_effort", [np.nan] * n_steps)
        agent2_effort = history.get("agent2_effort", [np.nan] * n_steps)
        theo_e1 = np.nan
        theo_e2 = np.nan
        approx_kl = history.get("approx_kl", [np.nan] * n_steps)
        batch_entropy = history.get("batch_entropy", [np.nan] * n_steps)

    agent1_arr = np.array(agent1_effort, dtype=float)
    agent2_arr = np.array(agent2_effort, dtype=float)
    effort_mean = (agent1_arr + agent2_arr) / 2.0
    theoretical_effort = (theo_e1 + theo_e2) / 2.0 if not (np.isnan(theo_e1) or np.isnan(theo_e2)) else np.nan

    df = pd.DataFrame({
        "step": steps,
        "method": run.method,
        "q": run.q,
        "seed": run.seed,
        "ablation": run.ablation,
        "experiment": run.experiment,
        "agent1_effort": agent1_effort,
        "agent2_effort": agent2_effort,
        "effort_mean": effort_mean,
        "policy_mean_effort": effort_mean,
        "theoretical_effort": theoretical_effort,
        "theoretical_effort1": theo_e1,
        "theoretical_effort2": theo_e2,
        "approx_kl": approx_kl,
        "batch_entropy": batch_entropy,
        "alpha_mean": np.nan,
        "beta_mean": np.nan,
        "mean_kl_window": np.nan,
        "drift_effort": np.nan,
    })

    # Exploitability from exploit_history or top-level fields
    exploit_series, exploit_valid = _load_exploitability_nested(data, n_steps)
    df["exploitability"] = exploit_series
    df["exploitability_is_valid"] = exploit_valid

    return df


def _load_exploitability_nested(data: Dict, n_steps: int) -> Tuple[List[float], List[bool]]:
    """Load exploitability and validity for nested-format files.

    Returns:
        (exploit_series, is_valid_series) tuple
    """
    # Check for exploit_history list
    exploit_history = data.get("exploit_history", [])
    if isinstance(exploit_history, list) and len(exploit_history) > 0:
        # exploit_history entries have "update" index (0-based update number)
        series = [np.nan] * n_steps

        for entry in exploit_history:
            if isinstance(entry, dict):
                update_idx = entry.get("update")
                exploit_max = entry.get("exploit_max")
                if update_idx is not None and exploit_max is not None:
                    if 0 <= update_idx < n_steps:
                        series[update_idx] = float(exploit_max)

        # If no entries matched, put final value at end
        if all(np.isnan(s) for s in series):
            final_exploit = data.get("final_exploit_max")
            if final_exploit is not None and n_steps > 0:
                series[-1] = float(final_exploit)
    else:
        # Fallback: use final_exploit_max at last step
        series = [np.nan] * n_steps
        final_exploit = data.get("final_exploit_max")
        if final_exploit is not None and n_steps > 0:
            series[-1] = float(final_exploit)

    valid = [not (e is None or (isinstance(e, float) and np.isnan(e))) for e in series]
    return series, valid


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

    For standard two_players experiments, uses e_star formula.
    For nested experiments, theoretical values are already loaded from JSON.
    """
    if df.empty:
        return df

    df = df.copy()
    # Only overwrite if not already set (nested format sets per-file values)
    mask = df["theoretical_effort"].isna()
    if mask.any():
        df.loc[mask, "theoretical_effort"] = df.loc[mask, "q"].apply(
            lambda q: e_star(q, **THEORY_PARAMS)
        )

    return df


def forward_fill_exploitability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill exploitability values for plotting.

    Since exploitability is only evaluated at certain steps,
    this fills gaps with the last known value for smoother plots.
    """
    df = df.copy()
    group_cols = ["method", "q", "seed", "ablation"]
    if "experiment" in df.columns:
        group_cols = ["experiment"] + group_cols
    df["exploitability_ffill"] = (
        df.groupby(group_cols)["exploitability"]
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
    Extract final values for each run (experiment, method, q, seed, ablation).

    Returns one row per run with final effort, exploitability, etc.
    """
    if df.empty:
        return df

    # Include experiment in groupby to avoid merging across experiments
    group_cols = ["method", "q", "seed", "ablation"]
    if "experiment" in df.columns:
        group_cols = ["experiment"] + group_cols

    grouped = df.groupby(group_cols)

    agg_dict = {
        "step": "max",
        "effort_mean": "last",
        "agent1_effort": "last",
        "agent2_effort": "last",
        "policy_mean_effort": "last",
        "theoretical_effort": "last",
        "approx_kl": "last",
        "alpha_mean": "last",
        "beta_mean": "last",
    }
    # Include per-agent theoretical if present
    for col in ["theoretical_effort1", "theoretical_effort2"]:
        if col in df.columns:
            agg_dict[col] = "last"

    final = grouped.agg(agg_dict).reset_index()

    # For exploitability, get last valid value
    def last_valid_exploit(group):
        valid = group[group["exploitability_is_valid"] == True]["exploitability"]
        return valid.iloc[-1] if len(valid) > 0 else np.nan

    exploit_df = grouped.apply(last_valid_exploit).reset_index(name="exploitability_final")
    final = final.merge(exploit_df, on=group_cols)

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
    
    # Group by (experiment, method, q, ablation, step) across seeds
    group_cols = ["method", "q", "ablation", "step"]
    if "experiment" in df.columns:
        group_cols = ["experiment"] + group_cols
    grouped = df.groupby(group_cols)
    
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


def get_convergence_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the convergence step for each run (experiment, method, q, seed, ablation).

    Uses the effort_delta and effort_window from CONVERGENCE_CONFIG to detect
    the first step where |effort_mean - theoretical_effort| < delta for
    `window` consecutive steps.

    Returns a DataFrame with one row per run and columns:
        experiment, method, q, seed, ablation, convergence_step (NaN if not converged)
    """
    delta = CONVERGENCE_CONFIG["effort_delta"]
    window = int(CONVERGENCE_CONFIG["effort_window"])
    min_steps = int(CONVERGENCE_CONFIG["min_steps"])

    group_cols = ["method", "q", "seed", "ablation"]
    if "experiment" in df.columns:
        group_cols = ["experiment"] + group_cols

    records = []
    for key, grp in df.groupby(group_cols):
        grp = grp.sort_values("step")
        effort = grp["effort_mean"].values
        theo = grp["theoretical_effort"].dropna()
        if theo.empty:
            e_star_val = e_star(grp["q"].iloc[0], **THEORY_PARAMS)
        else:
            e_star_val = theo.iloc[0]

        errors = np.abs(effort - e_star_val)
        within = errors < delta

        conv_step = np.nan
        steps_arr = grp["step"].values
        for i in range(max(0, min_steps - window), len(effort) - window + 1):
            if np.all(within[i : i + window]):
                conv_step = float(steps_arr[i + window - 1])
                break

        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        rec["convergence_step"] = conv_step
        records.append(rec)

    return pd.DataFrame(records)


def get_final_effort_error_from_json(path: str, theoretical_effort: float) -> float:
    """Load a convergence JSON and return the mean per-agent absolute error.

    Uses theoretical values stored in the JSON when available (handles
    asymmetric experiments like different_cost where agents have different
    equilibria). Falls back to ``theoretical_effort`` for symmetric experiments.

    For nested format with per-agent efforts:
        error = (|e1_final - theo_e1| + |e2_final - theo_e2|) / 2
    For nested format with shared effort:
        error = |effort_final - theo_effort|
    For flat format:
        error = |mean(e1, e2)_final - theoretical_effort|

    Returns float('inf') if the file cannot be loaded or has no data.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception:
        return float('inf')

    theoretical = data.get("theoretical", {})

    # Nested format (different_cost, different_ability)
    history = data.get("history", {})
    if isinstance(history, dict) and "steps" in history:
        if "effort" in history and len(history["effort"]) > 0:
            final_effort = float(history["effort"][-1])
            theo_val = float(theoretical["effort"]) if "effort" in theoretical else theoretical_effort
            return abs(final_effort - theo_val)
        elif "agent1_effort" in history and "agent2_effort" in history:
            e1 = history["agent1_effort"]
            e2 = history["agent2_effort"]
            if len(e1) > 0 and len(e2) > 0:
                final_e1 = float(e1[-1])
                final_e2 = float(e2[-1])
                theo_e1 = float(theoretical.get("effort1", theoretical_effort))
                theo_e2 = float(theoretical.get("effort2", theoretical_effort))
                return (abs(final_e1 - theo_e1) + abs(final_e2 - theo_e2)) / 2.0
            else:
                return float('inf')
        else:
            return float('inf')
    else:
        # Flat format (two_players, three_players)
        e1 = data.get("agent1_effort", [])
        e2 = data.get("agent2_effort", [])
        if len(e1) > 0 and len(e2) > 0:
            final_effort = (float(e1[-1]) + float(e2[-1])) / 2.0
        else:
            return float('inf')

    return abs(final_effort - theoretical_effort)


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
