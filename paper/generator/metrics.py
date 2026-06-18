"""
Metrics Computation for Paper Artifacts.

Implements:
- Convergence step detection
- Summary metrics (abs_error, exploitability, symmetric_gap)
- Cheap-gate statistics (gate_on_ratio, first_gate_activation)
- Quality classification
"""

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    CONVERGENCE_CONFIG,
    CHEAP_GATE_CONFIG,
    classify_quality,
    e_star,
    e_star_for_experiment,
    THEORY_PARAMS,
    get_theory_params,
)


@dataclass
class ConvergenceResult:
    """Result of convergence step detection."""
    converged: bool
    convergence_step: Optional[int]  # Step index where converged
    convergence_steps: Optional[int]  # Number of steps (for comparison)
    final_effort: float
    theoretical_effort: float
    effort_error: float
    quality: str


def convergence_step(
    effort_series: np.ndarray,
    e_star_val: float,
    delta: float = None,
    window: int = None,
    min_steps: int = None,
) -> ConvergenceResult:
    """
    Find the first step where effort converges to theoretical value.
    
    Convergence criterion:
    - Effort is within delta of e* for window consecutive steps
    - Must have at least min_steps total
    
    Args:
        effort_series: Array of effort values over training
        e_star_val: Theoretical equilibrium effort
        delta: Threshold for |e - e*| (default from config)
        window: Required consecutive steps within delta (default from config)
        min_steps: Minimum steps before declaring convergence (default from config)
    
    Returns:
        ConvergenceResult with convergence info
    """
    if delta is None:
        delta = CONVERGENCE_CONFIG["effort_delta"]
    if window is None:
        window = CONVERGENCE_CONFIG["effort_window"]
    if min_steps is None:
        min_steps = CONVERGENCE_CONFIG["min_steps"]
    
    effort_series = np.asarray(effort_series, dtype=float)
    n = len(effort_series)
    
    if n == 0:
        return ConvergenceResult(
            converged=False,
            convergence_step=None,
            convergence_steps=None,
            final_effort=np.nan,
            theoretical_effort=e_star_val,
            effort_error=np.nan,
            quality="Poor",
        )
    
    # Compute error at each step
    errors = np.abs(effort_series - e_star_val)
    within_delta = errors < delta
    
    # Find first index where error < delta for window consecutive steps
    convergence_idx = None
    for i in range(min_steps - window, n - window + 1):
        if i < 0:
            continue
        if np.all(within_delta[i:i + window]):
            convergence_idx = i + window - 1  # Last step of the window
            break
    
    final_effort = float(effort_series[-1])
    effort_error = float(errors[-1])
    quality = classify_quality(effort_error)
    
    if convergence_idx is not None:
        return ConvergenceResult(
            converged=True,
            convergence_step=convergence_idx,
            convergence_steps=int(convergence_idx),
            final_effort=final_effort,
            theoretical_effort=e_star_val,
            effort_error=effort_error,
            quality=quality,
        )
    else:
        return ConvergenceResult(
            converged=False,
            convergence_step=None,
            convergence_steps=None,
            final_effort=final_effort,
            theoretical_effort=e_star_val,
            effort_error=effort_error,
            quality=quality,
        )


def convergence_step_with_exploitability(
    effort_series: np.ndarray,
    exploitability_series: np.ndarray,
    e_star_val: float,
    effort_delta: float = None,
    effort_window: int = None,
    exploit_threshold: float = None,
    exploit_patience: int = None,
    min_steps: int = None,
) -> ConvergenceResult:
    """
    Find convergence step using BOTH effort and exploitability criteria.
    
    Full convergence criterion:
    1. Effort within delta of e* for window consecutive steps, AND
    2. Exploitability < threshold for patience consecutive evaluations
    
    Args:
        effort_series: Array of effort values
        exploitability_series: Array of exploitability values (may contain NaN)
        e_star_val: Theoretical equilibrium
        effort_delta, effort_window: Effort convergence params
        exploit_threshold, exploit_patience: Exploitability convergence params
        min_steps: Minimum steps
    
    Returns:
        ConvergenceResult
    """
    # Use defaults from config
    if effort_delta is None:
        effort_delta = CONVERGENCE_CONFIG["effort_delta"]
    if effort_window is None:
        effort_window = CONVERGENCE_CONFIG["effort_window"]
    if exploit_threshold is None:
        exploit_threshold = CONVERGENCE_CONFIG["exploit_threshold"]
    if exploit_patience is None:
        exploit_patience = CONVERGENCE_CONFIG["exploit_patience"]
    if min_steps is None:
        min_steps = CONVERGENCE_CONFIG["min_steps"]
    
    effort_series = np.asarray(effort_series, dtype=float)
    exploitability_series = np.asarray(exploitability_series, dtype=float)
    n = len(effort_series)
    
    if n == 0:
        return ConvergenceResult(
            converged=False,
            convergence_step=None,
            convergence_steps=None,
            final_effort=np.nan,
            theoretical_effort=e_star_val,
            effort_error=np.nan,
            quality="Poor",
        )
    
    # Check effort convergence
    effort_errors = np.abs(effort_series - e_star_val)
    effort_within_delta = effort_errors < effort_delta
    
    # Track exploit evaluations
    valid_exploit_mask = ~np.isnan(exploitability_series)
    valid_exploit_indices = np.where(valid_exploit_mask)[0]
    
    # Find convergence point where both criteria are met
    convergence_idx = None
    
    # For each potential effort convergence point
    for i in range(min_steps - effort_window, n - effort_window + 1):
        if i < 0:
            continue
        
        # Check effort criterion
        if not np.all(effort_within_delta[i:i + effort_window]):
            continue
        
        effort_conv_step = i + effort_window - 1
        
        # Check exploitability criterion up to this step
        # Need exploit_patience consecutive valid exploitability < threshold
        valid_before = valid_exploit_indices[valid_exploit_indices <= effort_conv_step]
        
        if len(valid_before) >= exploit_patience:
            # Check last exploit_patience evaluations
            last_patience = valid_before[-exploit_patience:]
            if np.all(exploitability_series[last_patience] < exploit_threshold):
                convergence_idx = effort_conv_step
                break
    
    final_effort = float(effort_series[-1])
    effort_error = float(effort_errors[-1])
    quality = classify_quality(effort_error)
    
    if convergence_idx is not None:
        return ConvergenceResult(
            converged=True,
            convergence_step=convergence_idx,
            convergence_steps=int(convergence_idx),
            final_effort=final_effort,
            theoretical_effort=e_star_val,
            effort_error=effort_error,
            quality=quality,
        )
    else:
        return ConvergenceResult(
            converged=False,
            convergence_step=None,
            convergence_steps=None,
            final_effort=final_effort,
            theoretical_effort=e_star_val,
            effort_error=effort_error,
            quality=quality,
        )


@dataclass
class CheapGateStats:
    """Statistics about cheap-gate behavior."""
    total_evaluations: int            # Total number of convergence evaluations
    gate_passed_count: int            # Number of times gate passed
    gate_on_ratio: float              # Fraction of evals where gate passed
    first_gate_activation: Optional[int]  # First step where gate triggered exploit eval
    exploit_eval_count: int           # Number of exploitability evaluations


def compute_cheap_gate_stats(
    mean_kl_series: np.ndarray,
    std_kl_series: Optional[np.ndarray] = None,
    drift_series: Optional[np.ndarray] = None,
    exploit_is_valid_series: Optional[np.ndarray] = None,
    steps: Optional[np.ndarray] = None,
    mean_kl_thresh: float = None,
    std_kl_thresh: float = None,
    drift_thresh: float = None,
) -> CheapGateStats:
    """
    Compute statistics about cheap-gate behavior.
    
    Args:
        mean_kl_series: Mean KL window values (NaN where not computed)
        std_kl_series: Std KL window values (optional)
        drift_series: Drift effort values (optional)
        exploit_is_valid_series: Boolean array of when exploitability was evaluated
        steps: Step numbers (for first_gate_activation)
        mean_kl_thresh, std_kl_thresh, drift_thresh: Thresholds (default from config)
    
    Returns:
        CheapGateStats
    """
    if mean_kl_thresh is None:
        mean_kl_thresh = CHEAP_GATE_CONFIG["mean_kl_thresh"]
    if std_kl_thresh is None:
        std_kl_thresh = CHEAP_GATE_CONFIG["std_kl_thresh"]
    if drift_thresh is None:
        drift_thresh = CHEAP_GATE_CONFIG["drift_effort_thresh"]
    
    mean_kl_series = np.asarray(mean_kl_series, dtype=float)
    n = len(mean_kl_series)
    
    # Count valid evaluations (where mean_kl is not NaN)
    valid_mask = ~np.isnan(mean_kl_series)
    total_evaluations = int(np.sum(valid_mask))
    
    if total_evaluations == 0:
        return CheapGateStats(
            total_evaluations=0,
            gate_passed_count=0,
            gate_on_ratio=0.0,
            first_gate_activation=None,
            exploit_eval_count=0,
        )
    
    # Determine gate pass at each evaluation
    gate_passed = valid_mask & (mean_kl_series <= mean_kl_thresh)
    
    if std_kl_series is not None:
        std_kl_series = np.asarray(std_kl_series, dtype=float)
        gate_passed = gate_passed & (
            np.isnan(std_kl_series) | (std_kl_series <= std_kl_thresh)
        )
    
    if drift_series is not None:
        drift_series = np.asarray(drift_series, dtype=float)
        gate_passed = gate_passed & (
            np.isnan(drift_series) | (drift_series <= drift_thresh)
        )
    
    gate_passed_count = int(np.sum(gate_passed))
    gate_on_ratio = gate_passed_count / total_evaluations if total_evaluations > 0 else 0.0
    
    # First gate activation
    first_gate_activation = None
    gate_indices = np.where(gate_passed)[0]
    if len(gate_indices) > 0:
        first_idx = gate_indices[0]
        if steps is not None:
            first_gate_activation = int(steps[first_idx])
        else:
            first_gate_activation = int(first_idx)
    
    # Count exploitability evaluations
    exploit_eval_count = 0
    if exploit_is_valid_series is not None:
        exploit_is_valid_series = np.asarray(exploit_is_valid_series, dtype=bool)
        exploit_eval_count = int(np.sum(exploit_is_valid_series))
    
    return CheapGateStats(
        total_evaluations=total_evaluations,
        gate_passed_count=gate_passed_count,
        gate_on_ratio=gate_on_ratio,
        first_gate_activation=first_gate_activation,
        exploit_eval_count=exploit_eval_count,
    )


@dataclass
class SummaryMetrics:
    """Summary metrics for a single run."""
    experiment: str
    method: str
    q: float
    seed: int
    ablation: str
    weight_variant: str
    # Effort metrics
    final_effort: float
    theoretical_effort: float
    abs_error: float          # |e - e*|
    quality: str
    # Exploitability
    final_exploitability: Optional[float]
    # Symmetry
    symmetry_gap: float       # |e1 - e2|
    # Convergence
    converged: bool
    convergence_step: Optional[int]
    # Cheap-gate
    gate_on_ratio: Optional[float]
    first_gate_step: Optional[int]


def compute_summary_metrics(df: pd.DataFrame) -> List[SummaryMetrics]:
    """
    Compute summary metrics for each run in the DataFrame.
    
    Args:
        df: DataFrame with convergence data (from extract.load_convergence_data)
    
    Returns:
        List of SummaryMetrics, one per run
    """
    results = []

    # Group by run (include experiment / weight_variant when present so that
    # Set 1 / Set 2 prize variants never merge into one run)
    group_cols = ["method", "q", "seed", "ablation"]
    if "weight_variant" in df.columns:
        group_cols.append("weight_variant")
    if "experiment" in df.columns:
        group_cols = ["experiment"] + group_cols

    grouped = df.groupby(group_cols)

    for group_key, group in grouped:
        key_map = dict(zip(group_cols, group_key if isinstance(group_key, tuple) else (group_key,)))
        experiment = key_map.get("experiment", "two_players")
        method = key_map["method"]
        q = key_map["q"]
        seed = key_map["seed"]
        ablation = key_map["ablation"]
        weight_variant = key_map.get("weight_variant", "baseline")
        group = group.sort_values("step")

        # Get effort series
        effort_series = group["policy_mean_effort"].values
        # Use per-file theoretical effort if available, else compute from formula
        if "theoretical_effort" in group.columns and not group["theoretical_effort"].isna().all():
            e_star_val = group["theoretical_effort"].iloc[0]
        else:
            e_star_val = e_star_for_experiment(q, experiment)
        
        # Get exploitability series
        exploit_series = group["exploitability"].values
        exploit_valid = group["exploitability_is_valid"].values if "exploitability_is_valid" in group.columns else None

        # Convergence: prefer the method's OWN verification verdict when the
        # run recorded one (PPO runners write stop_reason/stopped_at_update —
        # stop_reason == "exploitability" means the stability screen +
        # exploitability streak fired; "max_updates" means NC). The effort-band
        # detectors below remain only as a diagnostic fallback for runs without
        # a recorded verdict (e.g., the gradient baseline).
        stop_reason = None
        if "stop_reason" in group.columns:
            non_null = group["stop_reason"].dropna()
            if len(non_null) > 0:
                stop_reason = str(non_null.iloc[0])
        stopped_at_update = np.nan
        if "stopped_at_update" in group.columns:
            vals = group["stopped_at_update"].dropna()
            if len(vals) > 0:
                stopped_at_update = float(vals.iloc[0])

        if method in ("TEL-PPO", "PPO") and stop_reason is not None:
            converged_flag = stop_reason == "exploitability"
            convergence_step_val = (
                int(round(stopped_at_update))
                if converged_flag and not np.isnan(stopped_at_update)
                else None
            )
        else:
            if exploit_valid is not None and np.any(exploit_valid):
                conv_result = convergence_step_with_exploitability(
                    effort_series, exploit_series, e_star_val
                )
            else:
                conv_result = convergence_step(effort_series, e_star_val)
            converged_flag = conv_result.converged
            convergence_step_val = conv_result.convergence_step

        # Final values
        final_effort = float(effort_series[-1]) if len(effort_series) > 0 else np.nan
        abs_error = abs(final_effort - e_star_val) if not np.isnan(final_effort) else np.nan
        
        # Final exploitability (last valid value)
        final_exploitability = None
        if exploit_valid is not None:
            valid_exploit = exploit_series[exploit_valid]
            if len(valid_exploit) > 0:
                final_exploitability = float(valid_exploit[-1])
        
        # Symmetry gap
        if "agent1_effort" in group.columns and "agent2_effort" in group.columns:
            e1 = group["agent1_effort"].values[-1] if len(group) > 0 else np.nan
            e2 = group["agent2_effort"].values[-1] if len(group) > 0 else np.nan
            symmetry_gap = abs(e1 - e2) if not (np.isnan(e1) or np.isnan(e2)) else np.nan
        else:
            symmetry_gap = np.nan
        
        # Cheap-gate stats
        gate_on_ratio = None
        first_gate_step = None
        if "mean_kl_window" in group.columns:
            gate_stats = compute_cheap_gate_stats(
                mean_kl_series=group["mean_kl_window"].values,
                drift_series=group["drift_effort"].values if "drift_effort" in group.columns else None,
                exploit_is_valid_series=exploit_valid,
                steps=group["step"].values,
            )
            gate_on_ratio = gate_stats.gate_on_ratio
            first_gate_step = gate_stats.first_gate_activation
        
        results.append(SummaryMetrics(
            experiment=experiment,
            method=method,
            q=q,
            seed=seed,
            ablation=ablation,
            weight_variant=weight_variant,
            final_effort=final_effort,
            theoretical_effort=e_star_val,
            abs_error=abs_error,
            quality=classify_quality(abs_error) if not np.isnan(abs_error) else "Unknown",
            final_exploitability=final_exploitability,
            symmetry_gap=symmetry_gap,
            converged=converged_flag,
            convergence_step=convergence_step_val,
            gate_on_ratio=gate_on_ratio,
            first_gate_step=first_gate_step,
        ))
    
    return results


def summary_metrics_to_dataframe(metrics: List[SummaryMetrics]) -> pd.DataFrame:
    """Convert list of SummaryMetrics to DataFrame."""
    records = []
    for m in metrics:
        records.append({
            "experiment": m.experiment,
            "method": m.method,
            "q": m.q,
            "seed": m.seed,
            "ablation": m.ablation,
            "weight_variant": m.weight_variant,
            "final_effort": m.final_effort,
            "theoretical_effort": m.theoretical_effort,
            "abs_error": m.abs_error,
            "quality": m.quality,
            "final_exploitability": m.final_exploitability,
            "symmetry_gap": m.symmetry_gap,
            "converged": m.converged,
            "convergence_step": m.convergence_step,
            "gate_on_ratio": m.gate_on_ratio,
            "first_gate_step": m.first_gate_step,
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    # Quick test
    from .extract import load_all_convergence_data
    
    print("Loading convergence data...")
    df = load_all_convergence_data()
    
    print("\nComputing summary metrics...")
    metrics = compute_summary_metrics(df)
    
    print("\nSummary metrics:")
    metrics_df = summary_metrics_to_dataframe(metrics)
    print(metrics_df.to_string())
