#!/usr/bin/env python3
"""
Automated Hyperparameter Sweep for One-Stage Two-Player PPO (vs_opponent mode)

This script runs sequential hyperparameter sweep experiments, changing exactly ONE
config value per run while keeping all other parameters at baseline.

SINGLE CLI OVERRIDE PATHWAY:
----------------------------
This sweep uses ONLY CLI override flags to change hyperparameters. No config file
editing or overlay mechanism is used. Each variant passes exactly ONE override flag:

    Variant 0: --override-entropy-end 0.025   (entropy_coef_end: 0.015 -> 0.025)
    Variant 1: --override-lr-end 3e-4         (lr_end: 2e-4 -> 3e-4)
    Variant 2: --override-clip-end 0.45       (clip_range_end: 0.35 -> 0.45)
    Variant 3: --override-target-kl 0.12      (target_kl: 0.08 -> 0.12)

The run_two_players.py script enforces mutual exclusion - only ONE override flag
may be provided per run, ensuring clean single-variable experiments.

Quick Start:
------------
    # Full sweep (4 variants, ~8+ hours total)
    python tools/sweep_one_stage_vs_opponent.py

    # Dry run - preview commands without executing
    python tools/sweep_one_stage_vs_opponent.py --dry-run

    # Smoke test - quick validation with reduced episodes
    python tools/sweep_one_stage_vs_opponent.py --smoke-test

    # Custom baseline parameters
    python tools/sweep_one_stage_vs_opponent.py --q 55.0 --seed 123 --episodes 1024000

Output Structure:
-----------------
    results/sweeps/<timestamp>_q<q>_seed<seed>/
    ├── variant_0_entropy_coef_end/
    │   ├── console.log
    │   ├── variant.json       (includes run_id, variant_name, cli_flag, command)
    │   └── training.log       (copy of the actual training log)
    ├── variant_1_lr_end/
    │   └── ...
    ├── summary.json           (full machine-readable results)
    ├── summary.csv            (tabular summary with run_id, variant_name)
    └── README.md              (human-readable report)

CSV Output:
-----------
Results are written to results/one_stage_two_players_v2.csv (new schema with run_id
and variant_name columns). The original one_stage_two_players.csv is left untouched.

Author: Automated Sweep Tool
"""

import argparse
import csv
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Configuration: Baseline & Sweep Variants
# =============================================================================

# Baseline command components (fixed across all runs)
BASELINE_SCRIPT = "run/run_two_players.py"
BASELINE_METHOD = "ppo"
BASELINE_ROLLOUT_MODE = "vs_opponent"
BASELINE_EPISODES = 2_048_000
BASELINE_Q = 40.0
BASELINE_SEED = 42

# Baseline config values (from config/one_stage_two_players.py)
BASELINE_CONFIG = {
    "entropy_coef_end": 0.015,
    "lr_end": 2e-4,
    "clip_range_end": 0.35,
    "target_kl": 0.08,
}

# Sweep variants: each tuple is (param_name, cli_flag, new_value, description)
SWEEP_VARIANTS: List[Tuple[str, str, float, str]] = [
    ("entropy_coef_end", "--override-entropy-end", 0.025, "Higher final entropy for more exploration late"),
    ("lr_end", "--override-lr-end", 3e-4, "Higher final learning rate for faster late adaptation"),
    ("clip_range_end", "--override-clip-end", 0.45, "Wider final clip range for larger policy updates"),
    ("target_kl", "--override-target-kl", 0.12, "Higher KL target for more aggressive updates"),
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VariantResult:
    """Stores results from a single sweep variant run."""
    variant_idx: int
    param_name: str
    baseline_value: float
    new_value: float
    description: str
    command: str
    run_dir: str
    
    # Run identification (for CSV/log correlation)
    run_id: str = ""
    variant_name: str = ""
    
    log_path: Optional[str] = None
    
    # Parsed metrics (nullable if parsing failed)
    final_update: Optional[int] = None
    final_gap: Optional[float] = None
    final_mean_abs_err: Optional[float] = None
    final_policy: Optional[float] = None
    policy_mean_effort: Optional[float] = None
    sample_avg_effort: Optional[float] = None
    avg_approx_kl_last20: Optional[float] = None
    avg_entropy_last20: Optional[float] = None
    early_stop_triggered: bool = False
    early_stop_updates: Optional[int] = None
    early_stop_mean_abs_err: Optional[float] = None
    
    # Run metadata
    success: bool = False
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)


@dataclass  
class SweepSummary:
    """Aggregated sweep results."""
    sweep_id: str
    timestamp: str
    baseline_command: str
    baseline_config: Dict[str, float]
    q: float
    seed: int
    episodes: int
    variants: List[VariantResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    successful_runs: int = 0
    failed_runs: int = 0


# =============================================================================
# Log Parsing Functions
# =============================================================================

def parse_log_file(log_path: str) -> Dict[str, Any]:
    """
    Parse a training log file and extract key metrics.
    
    Returns a dict with:
        - final_update: last update number
        - final_gap: gap from last [Update N] line
        - final_policy: policy value from last update
        - policy_mean_effort: policy_mean_effort if logged
        - sample_avg_effort: sample_avg_effort if logged
        - avg_approx_kl_last20: average approx_kl over last 20 updates
        - avg_entropy_last20: average entropy over last 20 updates
        - early_stop_triggered: whether early stop was triggered
        - early_stop_updates: updates count at early stop
        - early_stop_mean_abs_err: mean_abs_err at early stop
    """
    metrics: Dict[str, Any] = {
        "final_update": None,
        "final_gap": None,
        "final_policy": None,
        "policy_mean_effort": None,
        "sample_avg_effort": None,
        "avg_approx_kl_last20": None,
        "avg_entropy_last20": None,
        "early_stop_triggered": False,
        "early_stop_updates": None,
        "early_stop_mean_abs_err": None,
    }
    
    if not os.path.exists(log_path):
        return metrics
    
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return metrics
    
    # Patterns for parsing
    # [Update N] q=40.0: e*=54.69, policy=49.41, gap=5.28, entropy=0.015, ..., approx_kl=-0.0003, ...
    update_pattern = re.compile(
        r"\[Update (\d+)\] q=[\d.]+: e\*=[\d.]+, policy=([\d.]+), gap=([\d.]+), entropy=([\d.]+).*?approx_kl=([-\d.]+)"
    )
    
    # [EarlyStopProbe] updates=480 mean_abs_err=5.796 (0/6)
    early_stop_probe_pattern = re.compile(
        r"\[EarlyStopProbe\] updates=(\d+) mean_abs_err=([\d.]+)"
    )
    
    # [EarlyStop] satisfied mean_abs_err threshold and patience. Stopping training.
    early_stop_trigger_pattern = re.compile(r"\[EarlyStop\].*Stopping training")
    
    # [Rollout] sample_avg_effort=47.78, mean_vs_sample_gap=...
    rollout_pattern = re.compile(r"\[Rollout\] sample_avg_effort=([\d.]+)")
    
    # Collect update data for averaging
    update_data: List[Dict[str, float]] = []
    last_early_stop_probe: Optional[Tuple[int, float]] = None
    
    for line in lines:
        # Parse update lines
        match = update_pattern.search(line)
        if match:
            update_num = int(match.group(1))
            policy = float(match.group(2))
            gap = float(match.group(3))
            entropy = float(match.group(4))
            approx_kl = float(match.group(5))
            
            update_data.append({
                "update": update_num,
                "policy": policy,
                "gap": gap,
                "entropy": entropy,
                "approx_kl": approx_kl,
            })
        
        # Parse rollout lines for sample_avg_effort
        rollout_match = rollout_pattern.search(line)
        if rollout_match:
            metrics["sample_avg_effort"] = float(rollout_match.group(1))
        
        # Parse early stop probes
        probe_match = early_stop_probe_pattern.search(line)
        if probe_match:
            last_early_stop_probe = (int(probe_match.group(1)), float(probe_match.group(2)))
        
        # Check for early stop trigger
        if early_stop_trigger_pattern.search(line):
            metrics["early_stop_triggered"] = True
    
    # Extract final metrics from collected data
    if update_data:
        last_update = update_data[-1]
        metrics["final_update"] = last_update["update"]
        metrics["final_gap"] = last_update["gap"]
        metrics["final_policy"] = last_update["policy"]
        metrics["policy_mean_effort"] = last_update["policy"]  # policy IS the mean effort
        
        # Calculate averages over last 20 updates
        last_20 = update_data[-20:] if len(update_data) >= 20 else update_data
        if last_20:
            metrics["avg_approx_kl_last20"] = sum(d["approx_kl"] for d in last_20) / len(last_20)
            metrics["avg_entropy_last20"] = sum(d["entropy"] for d in last_20) / len(last_20)
    
    # Set early stop probe results
    if last_early_stop_probe:
        metrics["early_stop_updates"] = last_early_stop_probe[0]
        metrics["early_stop_mean_abs_err"] = last_early_stop_probe[1]
        metrics["final_mean_abs_err"] = last_early_stop_probe[1]
    elif update_data:
        metrics["final_mean_abs_err"] = update_data[-1]["gap"]
    
    return metrics


def find_log_for_run(log_dir: str, q: float, seed: int, start_time: datetime.datetime) -> Optional[str]:
    """
    Find the log file generated by a run, matching by q, seed, and timestamp.
    
    The log naming convention is:
        one_stage_two_players_ppo_q<q>_ep<episodes>_seed<seed>_<timestamp>.log
    """
    if not os.path.exists(log_dir):
        return None
    
    # Format q value like the log file does (q40 or q40p5)
    q_clean = f"{q:g}".replace("-", "neg").replace(".", "p")
    pattern = f"one_stage_two_players_ppo_q{q_clean}_"
    
    candidates = []
    for fname in os.listdir(log_dir):
        if fname.startswith(pattern) and fname.endswith(".log"):
            # Extract timestamp from filename
            # Format: one_stage_two_players_ppo_q40_ep2048000_seed42_20251218_124353.log
            parts = fname.rsplit("_", 2)
            if len(parts) >= 3:
                try:
                    date_part = parts[-2]
                    time_part = parts[-1].replace(".log", "")
                    file_timestamp = datetime.datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
                    # Log must be created after run started
                    if file_timestamp >= start_time - datetime.timedelta(seconds=10):
                        candidates.append((fname, file_timestamp))
                except ValueError:
                    continue
    
    if not candidates:
        return None
    
    # Return the most recent matching log
    candidates.sort(key=lambda x: x[1], reverse=True)
    return os.path.join(log_dir, candidates[0][0])


# =============================================================================
# Command Building & Execution
# =============================================================================

def build_baseline_command(
    q: float = BASELINE_Q,
    seed: int = BASELINE_SEED,
    episodes: int = BASELINE_EPISODES,
    method: str = BASELINE_METHOD,
    rollout_mode: str = BASELINE_ROLLOUT_MODE,
) -> List[str]:
    """Build the baseline command as a list of arguments."""
    return [
        sys.executable,
        BASELINE_SCRIPT,
        "--method", method,
        "--rollout-mode", rollout_mode,
        "--episodes", str(episodes),
        "--q", str(q),
        "--seed", str(seed),
        "--eval-vs-opponent",
    ]


def build_variant_command(
    base_cmd: List[str],
    cli_flag: str,
    new_value: float,
    run_id: str,
    variant_name: str,
) -> List[str]:
    """Build a variant command by adding an override flag plus run_id and variant_name."""
    return base_cmd + [
        cli_flag, str(new_value),
        "--run-id", run_id,
        "--variant-name", variant_name,
    ]


def run_variant(
    variant_idx: int,
    param_name: str,
    baseline_value: float,
    new_value: float,
    description: str,
    cli_flag: str,
    base_cmd: List[str],
    run_dir: str,
    log_dir: str,
    q: float,
    seed: int,
    dry_run: bool = False,
) -> VariantResult:
    """
    Execute a single sweep variant and collect results.
    
    Args:
        variant_idx: Index of this variant in the sweep
        param_name: Name of the hyperparameter being changed
        baseline_value: Original config value
        new_value: Override value for this run
        description: Human-readable description of the change
        cli_flag: CLI flag to pass for the override
        base_cmd: Base command list (without overrides)
        run_dir: Directory to store variant outputs
        log_dir: Directory where run logs are written
        q: Q parameter value
        seed: Random seed
        dry_run: If True, don't actually execute
        
    Returns:
        VariantResult with parsed metrics and run metadata
    """
    # Generate run_id (timestamp) and variant_name for this run
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # variant_name format: <param>_<value> (e.g., entropy_end_0.025, lr_end_0.0003)
    variant_name = f"{param_name.replace('_coef', '').replace('_range', '')}_{new_value}"
    
    # Build the full command with run_id and variant_name
    cmd = build_variant_command(base_cmd, cli_flag, new_value, run_id, variant_name)
    cmd_str = " ".join(cmd)
    
    # Create result object with run_id and variant_name
    result = VariantResult(
        variant_idx=variant_idx,
        param_name=param_name,
        baseline_value=baseline_value,
        new_value=new_value,
        description=description,
        command=cmd_str,
        run_dir=run_dir,
        run_id=run_id,
        variant_name=variant_name,
    )
    
    # Create run directory
    os.makedirs(run_dir, exist_ok=True)
    
    # Write variant.json before running (includes run_id and variant_name)
    variant_info = {
        "variant_idx": variant_idx,
        "param_name": param_name,
        "baseline_value": baseline_value,
        "new_value": new_value,
        "description": description,
        "command": cmd_str,
        "cli_flag": cli_flag,
        "run_id": run_id,
        "variant_name": variant_name,
    }
    with open(os.path.join(run_dir, "variant.json"), "w") as f:
        json.dump(variant_info, f, indent=2)
    
    if dry_run:
        print(f"  [DRY RUN] Would execute: {cmd_str}")
        result.success = True
        return result
    
    # Execute the run
    start_time = datetime.datetime.now()
    console_log_path = os.path.join(run_dir, "console.log")
    
    try:
        with open(console_log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=None,  # No timeout (runs can be very long)
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Project root
            )
        
        end_time = datetime.datetime.now()
        result.duration_seconds = (end_time - start_time).total_seconds()
        
        if proc.returncode == 0:
            result.success = True
        else:
            result.success = False
            result.error_message = f"Process exited with code {proc.returncode}"
            
    except subprocess.TimeoutExpired as e:
        result.success = False
        result.error_message = f"Process timed out: {e}"
    except Exception as e:
        result.success = False
        result.error_message = f"Exception during run: {e}\n{traceback.format_exc()}"
    
    # Find and parse the run log
    actual_log = find_log_for_run(log_dir, q, seed, start_time)
    if actual_log:
        result.log_path = actual_log
        # Copy log to run directory for convenience
        try:
            shutil.copy2(actual_log, os.path.join(run_dir, "training.log"))
        except Exception:
            pass
        
        # Parse metrics
        metrics = parse_log_file(actual_log)
        result.final_update = metrics.get("final_update")
        result.final_gap = metrics.get("final_gap")
        result.final_mean_abs_err = metrics.get("final_mean_abs_err")
        result.final_policy = metrics.get("final_policy")
        result.policy_mean_effort = metrics.get("policy_mean_effort")
        result.sample_avg_effort = metrics.get("sample_avg_effort")
        result.avg_approx_kl_last20 = metrics.get("avg_approx_kl_last20")
        result.avg_entropy_last20 = metrics.get("avg_entropy_last20")
        result.early_stop_triggered = metrics.get("early_stop_triggered", False)
        result.early_stop_updates = metrics.get("early_stop_updates")
        result.early_stop_mean_abs_err = metrics.get("early_stop_mean_abs_err")
    else:
        result.log_path = None
        if result.success:
            result.error_message = "Could not find training log file"
    
    return result


# =============================================================================
# Summary Generation
# =============================================================================

def generate_summary_json(summary: SweepSummary, output_path: str) -> None:
    """Write sweep summary as JSON."""
    data = {
        "sweep_id": summary.sweep_id,
        "timestamp": summary.timestamp,
        "baseline_command": summary.baseline_command,
        "baseline_config": summary.baseline_config,
        "q": summary.q,
        "seed": summary.seed,
        "episodes": summary.episodes,
        "total_duration_seconds": summary.total_duration_seconds,
        "successful_runs": summary.successful_runs,
        "failed_runs": summary.failed_runs,
        "variants": [v.to_dict() for v in summary.variants],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def generate_summary_csv(summary: SweepSummary, output_path: str) -> None:
    """Write sweep summary as CSV."""
    fieldnames = [
        "variant_idx",
        "run_id",
        "variant_name",
        "param_name",
        "baseline_value",
        "new_value",
        "success",
        "final_update",
        "final_gap",
        "final_mean_abs_err",
        "final_policy",
        "avg_approx_kl_last20",
        "avg_entropy_last20",
        "early_stop_triggered",
        "duration_seconds",
        "error_message",
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for v in summary.variants:
            writer.writerow(v.to_dict())


def generate_summary_readme(summary: SweepSummary, output_path: str) -> None:
    """Write human-readable sweep README."""
    lines = [
        f"# Hyperparameter Sweep: {summary.sweep_id}",
        "",
        f"**Timestamp:** {summary.timestamp}",
        f"**q:** {summary.q}",
        f"**seed:** {summary.seed}",
        f"**episodes:** {summary.episodes:,}",
        f"**Total Duration:** {summary.total_duration_seconds / 3600:.2f} hours",
        f"**Successful Runs:** {summary.successful_runs}/{len(summary.variants)}",
        "",
        "## Baseline Configuration",
        "",
        "```",
        f"Command: {summary.baseline_command}",
        "",
        "Config overrides baseline values:",
        *[f"  {k}: {v}" for k, v in summary.baseline_config.items()],
        "```",
        "",
        "## Sweep Variants",
        "",
        "Each variant changes exactly ONE hyperparameter from baseline:",
        "",
        "| # | Parameter | Baseline | New Value | Final Gap | Final Update | KL (last20) | Entropy (last20) | Status |",
        "|---|-----------|----------|-----------|-----------|--------------|-------------|------------------|--------|",
    ]
    
    for v in summary.variants:
        gap_str = f"{v.final_gap:.3f}" if v.final_gap is not None else "N/A"
        update_str = str(v.final_update) if v.final_update is not None else "N/A"
        kl_str = f"{v.avg_approx_kl_last20:.5f}" if v.avg_approx_kl_last20 is not None else "N/A"
        ent_str = f"{v.avg_entropy_last20:.4f}" if v.avg_entropy_last20 is not None else "N/A"
        status = "✅" if v.success else "❌"
        
        lines.append(
            f"| {v.variant_idx} | {v.param_name} | {v.baseline_value} | {v.new_value} | "
            f"{gap_str} | {update_str} | {kl_str} | {ent_str} | {status} |"
        )
    
    lines.extend([
        "",
        "## Variant Details",
        "",
    ])
    
    for v in summary.variants:
        dur_str = f"{v.duration_seconds / 60:.1f} min" if v.duration_seconds else "N/A"
        lines.extend([
            f"### Variant {v.variant_idx}: {v.param_name}",
            "",
            f"**Description:** {v.description}",
            f"**Change:** {v.baseline_value} → {v.new_value}",
            f"**run_id:** `{v.run_id}`",
            f"**variant_name:** `{v.variant_name}`",
            "",
            f"**Command:**",
            "```bash",
            v.command,
            "```",
            "",
            f"**Results:**",
            f"- Success: {v.success}",
            f"- Final Update: {v.final_update}",
            f"- Final Gap: {v.final_gap}",
            f"- Final Policy (mean effort): {v.final_policy}",
            f"- Avg Approx KL (last 20): {v.avg_approx_kl_last20}",
            f"- Avg Entropy (last 20): {v.avg_entropy_last20}",
            f"- Early Stop Triggered: {v.early_stop_triggered}",
            f"- Duration: {dur_str}",
            "",
            f"**Log Path:** `{v.log_path or 'Not found'}`",
            "",
        ])
        
        if v.error_message:
            lines.extend([
                f"**Error:** {v.error_message}",
                "",
            ])
    
    lines.extend([
        "---",
        f"*Generated by sweep_one_stage_vs_opponent.py*",
    ])
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# =============================================================================
# Main Sweep Runner
# =============================================================================

def print_banner(msg: str, char: str = "=") -> None:
    """Print a formatted banner."""
    width = max(60, len(msg) + 4)
    print(char * width)
    print(f"  {msg}")
    print(char * width)


def run_sweep(
    q: float = BASELINE_Q,
    seed: int = BASELINE_SEED,
    episodes: int = BASELINE_EPISODES,
    dry_run: bool = False,
) -> SweepSummary:
    """
    Run the full hyperparameter sweep.
    
    Args:
        q: Q parameter for all runs
        seed: Random seed for all runs
        episodes: Number of episodes per run
        dry_run: If True, print commands without executing
        
    Returns:
        SweepSummary with all results
    """
    # Create sweep output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_id = f"{timestamp}_q{q}_seed{seed}"
    sweep_dir = os.path.join("results", "sweeps", sweep_id)
    os.makedirs(sweep_dir, exist_ok=True)
    
    # Build baseline command
    base_cmd = build_baseline_command(q=q, seed=seed, episodes=episodes)
    baseline_cmd_str = " ".join(base_cmd)
    
    # Initialize summary
    summary = SweepSummary(
        sweep_id=sweep_id,
        timestamp=timestamp,
        baseline_command=baseline_cmd_str,
        baseline_config=BASELINE_CONFIG.copy(),
        q=q,
        seed=seed,
        episodes=episodes,
    )
    
    print_banner(f"HYPERPARAMETER SWEEP: {sweep_id}")
    print(f"\nBaseline command:\n  {baseline_cmd_str}")
    print(f"\nOutput directory: {sweep_dir}")
    print(f"\nVariants to run: {len(SWEEP_VARIANTS)}")
    for i, (param, flag, val, desc) in enumerate(SWEEP_VARIANTS):
        baseline_val = BASELINE_CONFIG[param]
        print(f"  [{i}] {param}: {baseline_val} -> {val} ({desc})")
    print()
    
    if dry_run:
        print("[DRY RUN MODE - Commands will be printed but not executed]\n")
    
    # Run each variant
    log_dir = os.path.join("results", "logs")
    sweep_start = datetime.datetime.now()
    
    for i, (param_name, cli_flag, new_value, description) in enumerate(SWEEP_VARIANTS):
        variant_dir_name = f"variant_{i}_{param_name}"
        variant_dir = os.path.join(sweep_dir, variant_dir_name)
        baseline_value = BASELINE_CONFIG[param_name]
        
        print_banner(f"VARIANT {i}/{len(SWEEP_VARIANTS)-1}: {param_name}", char="-")
        print(f"  Change: {baseline_value} -> {new_value}")
        print(f"  Description: {description}")
        print(f"  Output: {variant_dir}")
        print()
        
        result = run_variant(
            variant_idx=i,
            param_name=param_name,
            baseline_value=baseline_value,
            new_value=new_value,
            description=description,
            cli_flag=cli_flag,
            base_cmd=base_cmd,
            run_dir=variant_dir,
            log_dir=log_dir,
            q=q,
            seed=seed,
            dry_run=dry_run,
        )
        
        summary.variants.append(result)
        
        if result.success:
            summary.successful_runs += 1
            status_icon = "✅"
        else:
            summary.failed_runs += 1
            status_icon = "❌"
        
        print(f"\n  {status_icon} Result: {'SUCCESS' if result.success else 'FAILED'}")
        if result.final_gap is not None:
            print(f"     Final Gap: {result.final_gap:.3f}")
            print(f"     Final Policy: {result.final_policy:.2f}")
        if result.avg_approx_kl_last20 is not None:
            print(f"     Avg KL (last20): {result.avg_approx_kl_last20:.5f}")
        if result.duration_seconds:
            print(f"     Duration: {result.duration_seconds / 60:.1f} min")
        if result.error_message:
            print(f"     Error: {result.error_message}")
        print()
    
    sweep_end = datetime.datetime.now()
    summary.total_duration_seconds = (sweep_end - sweep_start).total_seconds()
    
    # === UNIQUENESS ASSERTION: All run_ids must be distinct ===
    # This prevents silent overwrites of logs/summaries due to timestamp collisions.
    if not dry_run:
        from collections import Counter
        run_ids = [v.run_id for v in summary.variants if v.run_id]
        cnt = Counter(run_ids)
        dups = [rid for rid, c in cnt.items() if c > 1]
        if dups:
            raise RuntimeError(
                f"[FATAL] run_id collision detected! Duplicates: {dups}. "
                f"All {len(summary.variants)} variants must have unique run_ids to prevent "
                f"silent log/summary overwrites. This should never happen if runs are sequential."
            )
        print(f"[OK] All {len(run_ids)} run_ids are unique: {run_ids}")
    
    # Generate summary files
    print_banner("GENERATING SUMMARY FILES")
    
    json_path = os.path.join(sweep_dir, "summary.json")
    csv_path = os.path.join(sweep_dir, "summary.csv")
    readme_path = os.path.join(sweep_dir, "README.md")
    
    generate_summary_json(summary, json_path)
    print(f"  ✓ {json_path}")
    
    generate_summary_csv(summary, csv_path)
    print(f"  ✓ {csv_path}")
    
    generate_summary_readme(summary, readme_path)
    print(f"  ✓ {readme_path}")
    
    # Print final summary table
    print_banner("SWEEP COMPLETE")
    print(f"\n{'Variant':<10} {'Parameter':<20} {'Baseline':<12} {'New':<12} {'Gap':<10} {'Status':<8}")
    print("-" * 80)
    for v in summary.variants:
        gap_str = f"{v.final_gap:.3f}" if v.final_gap is not None else "N/A"
        status = "✅ OK" if v.success else "❌ FAIL"
        print(f"{v.variant_idx:<10} {v.param_name:<20} {v.baseline_value:<12} {v.new_value:<12} {gap_str:<10} {status:<8}")
    
    print(f"\nTotal Duration: {summary.total_duration_seconds / 3600:.2f} hours")
    print(f"Successful: {summary.successful_runs}/{len(summary.variants)}")
    print(f"Output: {sweep_dir}")
    
    return summary


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Automated hyperparameter sweep for One-Stage Two-Player PPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full sweep with defaults
  python tools/sweep_one_stage_vs_opponent.py

  # Dry run (preview commands)
  python tools/sweep_one_stage_vs_opponent.py --dry-run

  # Quick smoke test
  python tools/sweep_one_stage_vs_opponent.py --smoke-test

  # Custom parameters
  python tools/sweep_one_stage_vs_opponent.py --q 55.0 --seed 123
""",
    )
    
    parser.add_argument(
        "--q",
        type=float,
        default=BASELINE_Q,
        help=f"Q parameter for all runs (default: {BASELINE_Q})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BASELINE_SEED,
        help=f"Random seed for all runs (default: {BASELINE_SEED})",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=BASELINE_EPISODES,
        help=f"Episodes per run (default: {BASELINE_EPISODES:,})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing (preview mode)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run with reduced episodes (20,000) for quick validation",
    )
    
    args = parser.parse_args()
    
    # Smoke test overrides episodes
    episodes = args.episodes
    if args.smoke_test:
        episodes = 20_000
        print("[SMOKE TEST MODE] Running with reduced episodes for quick validation\n")
    
    run_sweep(
        q=args.q,
        seed=args.seed,
        episodes=episodes,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
