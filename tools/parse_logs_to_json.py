#!/usr/bin/env python3
"""
Parse PPO training logs and extract convergence data to JSON format.

This script parses log files from PPO training runs and extracts all training
metrics including alpha/beta parameters, KL divergence, and per-agent efforts.

Usage:
    python tools/parse_logs_to_json.py <log_file> [--output <output_file>]
    python tools/parse_logs_to_json.py --all  # Parse all mapped log files

Example:
    python tools/parse_logs_to_json.py results/logs/one_stage_two_players_ppo_q25_ep2048000_seed50_20260119_155813.log
"""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


# Batch size used in training (steps per update)
BATCH_SIZE = 4096

# Mapping of log files to their metadata
LOG_FILE_MAPPING = {
    "one_stage_two_players_ppo_q25_ep2048000_seed50_20260119_155813.log": {
        "q": 25.0,
        "seed": 50,
        "theoretical_effort": 87.5,
    },
    "one_stage_two_players_ppo_q40_ep2048000_seed42_20260115_191132.log": {
        "q": 40.0,
        "seed": 42,
        "theoretical_effort": 54.6875,
    },
    "one_stage_two_players_ppo_q55_ep2048000_seed50_20260116_190512.log": {
        "q": 55.0,
        "seed": 50,
        "theoretical_effort": 39.77272727272727,
    },
}


@dataclass
class UpdateData:
    """Data extracted from a single training update."""
    update_num: int
    step: int
    q: float
    theoretical_effort: float
    policy_mean_effort: float
    gap: float
    entropy: float
    approx_kl: float
    alpha_mean: float
    beta_mean: float
    # From rollout line
    agent1_effort: float
    agent2_effort: float
    sample_avg_effort: float


def parse_update_line(line: str) -> Optional[Dict]:
    """
    Parse an [Update N] line to extract training metrics.
    
    Example line:
    [Update 1] q=25.0: e*=87.50, policy=104.10, gap=16.60, entropy=0.000, 
    lag_prob=1.00, approx_kl=0.0159, kl_proxy_max=0.0492, ratio_max=2.6062, 
    clip_frac_max=0.5381, approx_kl_max_abs=0.0445, alpha_mean=85.93, beta_mean=79.16
    """
    # Match the update number
    update_match = re.match(r'\[Update (\d+)\]', line)
    if not update_match:
        return None
    
    update_num = int(update_match.group(1))
    
    # Extract key-value pairs
    result = {"update_num": update_num}
    
    # Parse q value
    q_match = re.search(r'q=([\d.]+)', line)
    if q_match:
        result["q"] = float(q_match.group(1))
    
    # Parse e* (theoretical effort)
    estar_match = re.search(r'e\*=([\d.]+)', line)
    if estar_match:
        result["theoretical_effort"] = float(estar_match.group(1))
    
    # Parse policy mean effort
    policy_match = re.search(r'policy=([\d.]+)', line)
    if policy_match:
        result["policy_mean_effort"] = float(policy_match.group(1))
    
    # Parse gap
    gap_match = re.search(r'gap=([\d.]+)', line)
    if gap_match:
        result["gap"] = float(gap_match.group(1))
    
    # Parse entropy
    entropy_match = re.search(r'entropy=([\d.]+)', line)
    if entropy_match:
        result["entropy"] = float(entropy_match.group(1))
    
    # Parse approx_kl
    kl_match = re.search(r'approx_kl=([\d.]+)', line)
    if kl_match:
        result["approx_kl"] = float(kl_match.group(1))
    
    # Parse alpha_mean
    alpha_match = re.search(r'alpha_mean=([\d.]+)', line)
    if alpha_match:
        result["alpha_mean"] = float(alpha_match.group(1))
    
    # Parse beta_mean
    beta_match = re.search(r'beta_mean=([\d.]+)', line)
    if beta_match:
        result["beta_mean"] = float(beta_match.group(1))
    
    return result


def parse_rollout_line(line: str) -> Optional[Dict]:
    """
    Parse a [Rollout] line to extract per-agent effort metrics.
    
    Example line:
    [Rollout] sample_avg_effort=106.96, mean_vs_sample_gap=-2.86, 
    effort_samples=8192, p1_effort=133.35, p2_effort=80.58
    """
    if '[Rollout]' not in line:
        return None
    
    result = {}
    
    # Parse sample_avg_effort
    sample_match = re.search(r'sample_avg_effort=([\d.]+)', line)
    if sample_match:
        result["sample_avg_effort"] = float(sample_match.group(1))
    
    # Parse p1_effort (agent1)
    p1_match = re.search(r'p1_effort=([\d.]+)', line)
    if p1_match:
        result["agent1_effort"] = float(p1_match.group(1))
    
    # Parse p2_effort (agent2)
    p2_match = re.search(r'p2_effort=([\d.]+)', line)
    if p2_match:
        result["agent2_effort"] = float(p2_match.group(1))
    
    return result


def parse_log_file(log_path: str) -> List[UpdateData]:
    """
    Parse a complete log file and extract all update data.
    
    Args:
        log_path: Path to the log file
        
    Returns:
        List of UpdateData objects, one per training update
    """
    updates = []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for [Update N] lines
        if line.startswith('[Update '):
            update_data = parse_update_line(line)
            
            if update_data:
                # Look for the following [Rollout] line (should be next non-empty line or within 2 lines)
                rollout_data = None
                for j in range(i + 1, min(i + 5, len(lines))):
                    if '[Rollout]' in lines[j]:
                        rollout_data = parse_rollout_line(lines[j])
                        break
                
                if rollout_data:
                    update_data.update(rollout_data)
                
                # Calculate step from update number
                update_num = update_data["update_num"]
                step = (update_num - 1) * BATCH_SIZE
                
                # Create UpdateData object
                try:
                    data = UpdateData(
                        update_num=update_num,
                        step=step,
                        q=update_data.get("q", 0.0),
                        theoretical_effort=update_data.get("theoretical_effort", 0.0),
                        policy_mean_effort=update_data.get("policy_mean_effort", 0.0),
                        gap=update_data.get("gap", 0.0),
                        entropy=update_data.get("entropy", 0.0),
                        approx_kl=update_data.get("approx_kl", 0.0),
                        alpha_mean=update_data.get("alpha_mean", 0.0),
                        beta_mean=update_data.get("beta_mean", 0.0),
                        agent1_effort=update_data.get("agent1_effort", 0.0),
                        agent2_effort=update_data.get("agent2_effort", 0.0),
                        sample_avg_effort=update_data.get("sample_avg_effort", 0.0),
                    )
                    updates.append(data)
                except (KeyError, TypeError) as e:
                    print(f"Warning: Skipping update {update_num} due to missing data: {e}")
        
        i += 1
    
    return updates


def create_convergence_json(
    updates: List[UpdateData],
    q: float,
    seed: int,
    theoretical_effort: float,
    ablation_name: str = "baseline",
) -> Dict:
    """
    Create a convergence JSON structure from parsed update data.
    
    Args:
        updates: List of UpdateData objects from parsing
        q: Noise parameter value
        seed: Random seed used
        theoretical_effort: Expected equilibrium effort
        ablation_name: Name of the ablation variant
        
    Returns:
        Dictionary in the expected JSON format
    """
    # Sort updates by step to ensure correct order
    updates = sorted(updates, key=lambda u: u.step)
    
    # Extract arrays
    steps = [u.step for u in updates]
    agent1_effort = [u.agent1_effort for u in updates]
    agent2_effort = [u.agent2_effort for u in updates]
    policy_mean_effort = [u.policy_mean_effort for u in updates]
    alpha_mean = [u.alpha_mean for u in updates]
    beta_mean = [u.beta_mean for u in updates]
    approx_kl = [u.approx_kl for u in updates]
    batch_entropy = [u.entropy for u in updates]
    
    # Calculate total episodes
    total_episodes = max(steps) + BATCH_SIZE if steps else 0
    
    return {
        "algorithm": "PPO",
        "q": q,
        "seed": seed,
        "ablation_name": ablation_name,
        "theoretical_effort": theoretical_effort,
        "steps": steps,
        "agent1_effort": agent1_effort,
        "agent2_effort": agent2_effort,
        "policy_mean_effort": policy_mean_effort,
        "alpha_mean": alpha_mean,
        "beta_mean": beta_mean,
        "approx_kl": approx_kl,
        "batch_entropy": batch_entropy,
        "rollout_mode": "selfplay",
        "total_episodes": total_episodes,
    }


def infer_metadata_from_filename(filename: str) -> Dict:
    """
    Infer q value and seed from log filename.
    
    Filename format: one_stage_two_players_ppo_q{q}_ep{episodes}_seed{seed}_{timestamp}.log
    """
    # Match q value
    q_match = re.search(r'_q(\d+)_', filename)
    q = float(q_match.group(1)) if q_match else None
    
    # Match seed
    seed_match = re.search(r'_seed(\d+)_', filename)
    seed = int(seed_match.group(1)) if seed_match else 42
    
    # Calculate theoretical effort: e* = (k1 * V) / (k1 + k2 + 2*q)
    # With k1=k2=50 and V=100: e* = 2500 / (100 + 2*q)
    theoretical_effort = 2500 / (100 + 2 * q) if q else None
    
    return {
        "q": q,
        "seed": seed,
        "theoretical_effort": theoretical_effort,
    }


def parse_log_to_json(
    log_path: str,
    output_path: Optional[str] = None,
    q: Optional[float] = None,
    seed: Optional[int] = None,
    theoretical_effort: Optional[float] = None,
) -> str:
    """
    Parse a log file and save as JSON.
    
    Args:
        log_path: Path to input log file
        output_path: Path to output JSON file (auto-generated if None)
        q: Override q value (inferred from filename if None)
        seed: Override seed value (inferred from filename if None)
        theoretical_effort: Override theoretical effort (calculated if None)
        
    Returns:
        Path to the generated JSON file
    """
    log_path = Path(log_path)
    filename = log_path.name
    
    # Try to get metadata from mapping first
    if filename in LOG_FILE_MAPPING:
        metadata = LOG_FILE_MAPPING[filename]
    else:
        # Infer from filename
        metadata = infer_metadata_from_filename(filename)
    
    # Allow overrides
    q = q if q is not None else metadata.get("q")
    seed = seed if seed is not None else metadata.get("seed", 42)
    theoretical_effort = theoretical_effort if theoretical_effort is not None else metadata.get("theoretical_effort")
    
    if q is None:
        raise ValueError(f"Could not determine q value from filename: {filename}")
    
    if theoretical_effort is None:
        # Calculate: e* = 2500 / (100 + 2*q)
        theoretical_effort = 2500 / (100 + 2 * q)
    
    print(f"Parsing: {log_path}")
    print(f"  q={q}, seed={seed}, e*={theoretical_effort:.4f}")
    
    # Parse the log file
    updates = parse_log_file(str(log_path))
    print(f"  Extracted {len(updates)} updates")
    
    if not updates:
        raise ValueError(f"No update data found in log file: {log_path}")
    
    # Create JSON structure
    json_data = create_convergence_json(
        updates=updates,
        q=q,
        seed=seed,
        theoretical_effort=theoretical_effort,
        ablation_name="baseline",
    )
    
    # Determine output path
    if output_path is None:
        # Use standard naming: ppo_q{q}_convergence.json
        output_dir = log_path.parent.parent / "convergence_history"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"ppo_q{q}_convergence.json"
    else:
        output_path = Path(output_path)
    
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"  Saved to: {output_path}")
    print(f"  Steps: {len(json_data['steps'])}, Final policy effort: {json_data['policy_mean_effort'][-1]:.2f}")
    
    # Verify alpha/beta data is present
    if json_data["alpha_mean"] and json_data["beta_mean"]:
        print(f"  alpha_mean: [{json_data['alpha_mean'][0]:.2f}, ..., {json_data['alpha_mean'][-1]:.2f}]")
        print(f"  beta_mean: [{json_data['beta_mean'][0]:.2f}, ..., {json_data['beta_mean'][-1]:.2f}]")
    else:
        print("  WARNING: alpha_mean/beta_mean data is empty!")
    
    return str(output_path)


def parse_all_mapped_logs(logs_dir: str = "results/two_players/logs", output_dir: str = "results/two_players/convergence"):
    """
    Parse all log files in the mapping and generate JSON files.
    
    Args:
        logs_dir: Directory containing log files
        output_dir: Directory to save JSON files
    """
    logs_dir = Path(logs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Parsing all mapped log files from: {logs_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    results = []
    
    for filename, metadata in LOG_FILE_MAPPING.items():
        log_path = logs_dir / filename
        
        if not log_path.exists():
            print(f"\n❌ Log file not found: {log_path}")
            continue
        
        q = metadata["q"]
        output_path = output_dir / f"ppo_q{q}_convergence.json"
        
        try:
            result_path = parse_log_to_json(
                log_path=str(log_path),
                output_path=str(output_path),
                q=metadata["q"],
                seed=metadata["seed"],
                theoretical_effort=metadata["theoretical_effort"],
            )
            results.append((q, result_path))
            print()
        except Exception as e:
            print(f"\n❌ Error parsing {filename}: {e}")
    
    print("=" * 60)
    print(f"Successfully generated {len(results)} JSON files:")
    for q, path in sorted(results):
        print(f"  q={q}: {path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Parse PPO training logs and extract convergence data to JSON"
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        help="Path to the log file to parse"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path (auto-generated if not specified)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Parse all mapped log files"
    )
    parser.add_argument(
        "--logs-dir",
        default="results/two_players/logs",
        help="Directory containing log files (default: results/two_players/logs)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/two_players/convergence",
        help="Directory to save JSON files (default: results/two_players/convergence)"
    )
    parser.add_argument(
        "--q",
        type=float,
        help="Override q value"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override seed value"
    )
    
    args = parser.parse_args()
    
    if args.all:
        parse_all_mapped_logs(args.logs_dir, args.output_dir)
    elif args.log_file:
        parse_log_to_json(
            log_path=args.log_file,
            output_path=args.output,
            q=args.q,
            seed=args.seed,
        )
    else:
        parser.print_help()
        print("\nError: Either provide a log file path or use --all flag")
        exit(1)


if __name__ == "__main__":
    main()
