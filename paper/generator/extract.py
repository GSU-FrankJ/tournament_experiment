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
from .config import (
    CONVERGENCE_DIR, CSV_PATH, e_star, e_star_for_experiment, THEORY_PARAMS,
    CONVERGENCE_DIRS, CONVERGENCE_CONFIG, BASELINE_OVERRIDES, get_theory_params,
)


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
    "sample_effort_mean",      # Average of agent1 and agent2
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
    # Run-level verification outcome (constant per run; from the runner's own
    # stability + exploitability stopping rule)
    "stop_reason",
    "stopped_at_update",
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
    agent1 = data.get("agent1_effort", [np.nan] * n_steps)
    agent2 = data.get("agent2_effort", [np.nan] * n_steps)

    if "policy_mean_effort" in data:
        # PPO/TEL-PPO flat format: agent1_effort/agent2_effort are sample means
        # from rollouts; policy_mean_effort is the deterministic Beta-distribution
        # mean (Metric B).
        policy_mean = data["policy_mean_effort"]
    elif run.method in ("TEL-PPO", "PPO"):
        raise ValueError(
            f"PPO flat-format JSON is missing 'policy_mean_effort'. "
            f"All PPO runners (run_two_players.py, run_three_players.py) "
            f"write this field; missing = corrupted file or ancient runner. "
            f"Run path: {run.path}"
        )
    else:
        # Gradient / other methods: efforts are deterministic, no separate
        # policy distribution.  agent1/agent2 efforts ARE the computed efforts.
        policy_mean = [(a1 + a2) / 2.0 for a1, a2 in zip(agent1, agent2)]

    df = pd.DataFrame({
        "step": steps,
        "method": run.method,
        "q": run.q,
        "seed": run.seed,
        "ablation": run.ablation,
        "weight_variant": run.weight_variant,
        "experiment": run.experiment,
        "agent1_effort": agent1,
        "agent2_effort": agent2,
        "policy_mean_effort": policy_mean,
    })

    # sample_effort_mean: average of per-agent sample means from rollouts.
    # For PPO: distinct from policy_mean_effort.
    # For Gradient: identical to policy_mean_effort (both are deterministic).
    df["sample_effort_mean"] = (df["agent1_effort"] + df["agent2_effort"]) / 2.0

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
        params = get_theory_params(run.experiment)
        df["theoretical_effort"] = e_star(run.q, **params)
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

    # Run-level verification outcome (constant per run): the runner's OWN
    # convergence verdict. stop_reason == "exploitability" means the
    # stability-screen + exploitability-streak verification fired;
    # "max_updates" means the run exhausted its budget (NC).
    df["stop_reason"] = data.get("stop_reason")
    stopped = data.get("stopped_at_update")
    df["stopped_at_update"] = float(stopped) if stopped is not None else np.nan

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
    theoretical_effort = (theo_e1 + theo_e2) / 2.0 if not (np.isnan(theo_e1) or np.isnan(theo_e2)) else np.nan

    # policy_mean_effort: nested-format runners (run_different_cost.py:716-717,
    # run_different_ability.py) store policy means directly in agent1_effort /
    # agent2_effort (via agent.mean_effort()), NOT sample means.  If the JSON
    # explicitly provides a policy_mean_effort field, use it; otherwise average
    # the two per-agent policy means.
    if "policy_mean_effort" in history:
        policy_mean_arr = np.array(history["policy_mean_effort"], dtype=float)
    else:
        policy_mean_arr = (agent1_arr + agent2_arr) / 2.0

    # Nested format does not record sample-action averages.  Set to NaN so
    # downstream code that accidentally reads this column gets an explicit
    # signal rather than silently using policy means as if they were samples.
    sample_effort_mean_arr = np.full(n_steps, np.nan)

    df = pd.DataFrame({
        "step": steps,
        "method": run.method,
        "q": run.q,
        "seed": run.seed,
        "ablation": run.ablation,
        "weight_variant": run.weight_variant,
        "experiment": run.experiment,
        "agent1_effort": agent1_effort,
        "agent2_effort": agent2_effort,
        "sample_effort_mean": sample_effort_mean_arr,
        "policy_mean_effort": policy_mean_arr,
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

    # Run-level verification outcome (constant per run; see flat loader)
    df["stop_reason"] = data.get("stop_reason")
    stopped = data.get("stopped_at_update")
    df["stopped_at_update"] = float(stopped) if stopped is not None else np.nan

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


def promote_preferred_ablations(df: pd.DataFrame) -> pd.DataFrame:
    """Replace baseline with preferred ablation for overridden (experiment, q) combos.

    For each entry in BASELINE_OVERRIDES, drops old "baseline" rows and relabels
    the preferred ablation as "baseline" so all downstream code works transparently.
    """
    if df.empty or not BASELINE_OVERRIDES:
        return df

    df = df.copy()
    for (experiment, q), preferred in BASELINE_OVERRIDES.items():
        mask_old = (
            (df["experiment"] == experiment)
            & (df["q"] == q)
            & (df["ablation"] == "baseline")
            & (df["method"].isin(["TEL-PPO", "PPO"]))
        )
        mask_new = (
            (df["experiment"] == experiment)
            & (df["q"] == q)
            & (df["ablation"] == preferred)
        )
        if mask_new.any():
            df = df[~mask_old]
            df.loc[mask_new, "ablation"] = "baseline"
    return df


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
    
    df = load_multiple_runs(runs)
    return promote_preferred_ablations(df)


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
        if "experiment" in df.columns:
            for exp, exp_mask in df.loc[mask].groupby("experiment").groups.items():
                df.loc[exp_mask, "theoretical_effort"] = df.loc[exp_mask, "q"].apply(
                    lambda q, _exp=exp: e_star_for_experiment(q, _exp)
                )
        else:
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
    """Add column for |policy_mean_effort - theoretical_effort|."""
    df = df.copy()
    df["effort_error"] = np.abs(df["policy_mean_effort"] - df["theoretical_effort"])
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
        "sample_effort_mean": "last",
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
    final["effort_error"] = np.abs(final["policy_mean_effort"] - final["theoretical_effort"])

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
        "sample_effort_mean", "agent1_effort", "agent2_effort", "policy_mean_effort",
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
    DIAGNOSTIC-ONLY effort-band detector — NOT the paper convergence criterion.

    Detects the first step where |policy_mean_effort - theoretical_effort| <
    effort_delta for effort_window consecutive logged steps (CONVERGENCE_CONFIG).
    This is structurally unsatisfiable for runs that early-stop via the method's
    own exploitability verification (they terminate before min_steps logged
    updates), which is exactly what produced the all-"NC" tables. Paper tables
    must use ``get_verified_convergence_step`` instead; this helper is kept only
    for trajectory diagnostics and for the gradient baseline, which has no
    verification module.

    Returns a DataFrame with one row per run and columns:
        experiment, method, q, seed, ablation, convergence_step (NaN if not converged)
    """
    delta = CONVERGENCE_CONFIG["effort_delta"]
    window = int(CONVERGENCE_CONFIG["effort_window"])
    min_steps = int(CONVERGENCE_CONFIG["min_steps"])

    group_cols = ["method", "q", "seed", "ablation"]
    if "weight_variant" in df.columns:
        group_cols.append("weight_variant")  # keep Set 1 / Set 2 runs distinct
    if "experiment" in df.columns:
        group_cols = ["experiment"] + group_cols

    records = []
    for key, grp in df.groupby(group_cols):
        grp = grp.sort_values("step")
        effort = grp["policy_mean_effort"].values
        theo = grp["theoretical_effort"].dropna()
        if theo.empty:
            exp = grp["experiment"].iloc[0] if "experiment" in grp.columns else None
            e_star_val = e_star_for_experiment(grp["q"].iloc[0], exp) if exp else e_star(grp["q"].iloc[0])
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


def get_verified_convergence_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-run convergence from the method's OWN verification (the paper criterion).

    A TEL-PPO run converges when its in-training verification fires: the
    stability screen + exploitability streak stop the run with
    ``stop_reason == "exploitability"``. The reported value is
    ``stopped_at_update`` — the PPO update index at which verification fired.
    Runs that hit the budget (``stop_reason == "max_updates"``) are NC (NaN).

    Returns a DataFrame with one row per run and columns:
        experiment, method, q, seed, ablation, stop_reason, verified (bool),
        convergence_update (NaN if not verified)
    """
    group_cols = ["method", "q", "seed", "ablation"]
    if "weight_variant" in df.columns:
        group_cols.append("weight_variant")  # keep Set 1 / Set 2 runs distinct
    if "experiment" in df.columns:
        group_cols = ["experiment"] + group_cols

    records = []
    for key, grp in df.groupby(group_cols):
        stop_reason = None
        if "stop_reason" in grp.columns:
            non_null = grp["stop_reason"].dropna()
            if len(non_null) > 0:
                stop_reason = str(non_null.iloc[0])
        stopped_at = np.nan
        if "stopped_at_update" in grp.columns:
            vals = grp["stopped_at_update"].dropna()
            if len(vals) > 0:
                stopped_at = float(vals.iloc[0])

        verified = stop_reason == "exploitability"
        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        rec["stop_reason"] = stop_reason
        rec["verified"] = verified
        rec["convergence_update"] = stopped_at if verified and not np.isnan(stopped_at) else np.nan
        records.append(rec)

    return pd.DataFrame(records)


def get_cheap_gate_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the first step where the cheap gate passed (exploitability was evaluated).

    Returns a DataFrame with one row per run and columns:
        experiment, method, q, seed, ablation, cheap_gate_step (NaN if never passed)
    """
    group_cols = ["method", "q", "seed", "ablation"]
    if "experiment" in df.columns:
        group_cols = ["experiment"] + group_cols

    records = []
    for key, grp in df.groupby(group_cols):
        grp = grp.sort_values("step")
        gate_step = np.nan

        if "exploitability_is_valid" in grp.columns:
            valid_rows = grp[grp["exploitability_is_valid"] == True]
            if not valid_rows.empty:
                gate_step = float(valid_rows["step"].iloc[0])

        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        rec["cheap_gate_step"] = gate_step
        records.append(rec)

    return pd.DataFrame(records)


def get_nash_convergence_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the step where Nash equilibrium is declared: exploitability < threshold
    for exploit_patience consecutive valid evaluations.

    Returns a DataFrame with one row per run and columns:
        experiment, method, q, seed, ablation, nash_step (NaN if not converged)
    """
    threshold = CONVERGENCE_CONFIG["exploit_threshold"]
    patience = int(CONVERGENCE_CONFIG["exploit_patience"])

    group_cols = ["method", "q", "seed", "ablation"]
    if "experiment" in df.columns:
        group_cols = ["experiment"] + group_cols

    records = []
    for key, grp in df.groupby(group_cols):
        grp = grp.sort_values("step")
        nash_step = np.nan

        if "exploitability" in grp.columns:
            valid = grp[grp["exploitability"].notna() & (grp.get("exploitability_is_valid", True) == True)]
            if len(valid) >= patience:
                exploit_vals = valid["exploitability"].values
                step_vals = valid["step"].values
                # Sliding window: patience consecutive evals below threshold
                below = exploit_vals < threshold
                streak = 0
                for j in range(len(below)):
                    if below[j]:
                        streak += 1
                        if streak >= patience:
                            nash_step = float(step_vals[j])
                            break
                    else:
                        streak = 0

        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        rec["nash_step"] = nash_step
        records.append(rec)

    return pd.DataFrame(records)


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
    print(final[["method", "q", "seed", "ablation", "sample_effort_mean", "theoretical_effort", "effort_error"]])
