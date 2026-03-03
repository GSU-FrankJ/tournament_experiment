#!/usr/bin/env python3
"""
Compare v1 (baseline) vs v2 (baseline_v2) convergence results side-by-side.

Loads convergence JSONs for both ablation tags, computes per-(experiment, q)
metrics (mean |e-e*|, exploitability, convergence rate), prints a comparison
table, and saves to results/comparison_v1_v2.csv.

Usage:
    python run/compare_v1_v2.py
    python run/compare_v1_v2.py --v1-tag baseline --v2-tag baseline_v2
"""
from __future__ import annotations

import sys
import os
import json
import csv
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper.generator.config import CONVERGENCE_DIRS

# Theory functions per experiment
from utils.theory import e_star as e_star_symmetric
from utils.theory import e_star_two_players_asymmetric_cost
from config.one_stage_different_ability import calculate_theoretical_effort_different_ability
from config.one_stage_different_cost import config as diff_cost_cfg
from config.one_stage_different_ability import config as diff_ability_cfg


def get_theoretical_effort(experiment: str, q: float) -> float:
    """Return scalar theoretical effort for a given experiment and q."""
    if experiment in ("two_players", "three_players"):
        # Symmetric: e* = (w_H - w_L) / (4kq)
        return (6.5 - 3.0) / (4 * 0.0004 * q)
    elif experiment == "different_cost":
        e1, e2 = e_star_two_players_asymmetric_cost(
            q, diff_cost_cfg["w_h"], diff_cost_cfg["w_l"],
            diff_cost_cfg["k1"], diff_cost_cfg["k2"],
        )
        return (e1 + e2) / 2.0  # average effort
    elif experiment == "different_ability":
        return calculate_theoretical_effort_different_ability(
            q, diff_ability_cfg["k"], diff_ability_cfg["l1"],
            diff_ability_cfg["l2"], diff_ability_cfg["w_h"],
            diff_ability_cfg["w_l"],
        )
    raise ValueError(f"Unknown experiment: {experiment}")


def load_runs(tag: str) -> Dict[Tuple[str, float], List[dict]]:
    """
    Scan all convergence dirs for runs matching the given ablation tag.

    Returns: dict mapping (experiment, q) -> list of run dicts.
    Each run dict has keys: seed, final_effort, theoretical_e, abs_error,
    converged_step, path.
    """
    runs = defaultdict(list)

    for experiment, cdir in CONVERGENCE_DIRS.items():
        if not os.path.isdir(cdir):
            continue
        for fname in sorted(os.listdir(cdir)):
            if not fname.endswith("_convergence.json"):
                continue

            fpath = os.path.join(cdir, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            # Filter by ablation tag using JSON field (not filename).
            # Legacy baseline files without the field default to "baseline".
            file_ablation = data.get("ablation_name", "baseline")
            if file_ablation != tag:
                continue

            q = data.get("q")
            if q is None:
                continue

            # Filter out runs from a different config (e.g., old w_h=8/w_l=4
            # vs current w_h=6.5/w_l=3.0). Compare the file's stored
            # theoretical_effort against what the current config produces.
            expected_theo = get_theoretical_effort(experiment, q)
            file_theo = (
                data.get("theoretical_effort")
                or data.get("theoretical", {}).get("effort")
                or (  # different_cost: average of both players
                    (data.get("theoretical", {}).get("effort1", 0)
                     + data.get("theoretical", {}).get("effort2", 0)) / 2.0
                    if "effort1" in data.get("theoretical", {})
                    else None
                )
            )
            if file_theo is not None and abs(file_theo - expected_theo) > 1.0:
                continue  # Skip runs trained under a different config

            # Extract final effort and theoretical value from multiple JSON formats:
            # 1. final_results.final_effort (three_players)
            # 2. final.effort (two_players, different_ability)
            # 3. final.effort1/effort2 avg (different_cost)
            # 4. time-series fallback: agent1_effort[-1]
            fr = data.get("final_results", {})
            fin = data.get("final", {})
            theo = data.get("theoretical", {})

            final_effort = (
                fr.get("final_effort")
                or fin.get("effort")
                or (  # different_cost: average of both players
                    (fin["effort1"] + fin["effort2"]) / 2.0
                    if "effort1" in fin and "effort2" in fin
                    else None
                )
            )
            # Fallback to last time-series value
            if final_effort is None:
                for ts_key in ["policy_mean_effort", "agent1_effort"]:
                    ts = data.get(ts_key, [])
                    if ts:
                        final_effort = ts[-1]
                        break

            theoretical_e = (
                fr.get("theoretical_e")
                or data.get("theoretical_effort")
                or theo.get("effort")
                or (  # different_cost: average of both players
                    (theo["effort1"] + theo["effort2"]) / 2.0
                    if "effort1" in theo and "effort2" in theo
                    else None
                )
            )

            abs_error = (
                fr.get("abs_error")
                or fin.get("gap")
                or (  # different_cost: max gap
                    fin.get("max_gap")
                )
            )

            # Compute abs_error if still missing
            if abs_error is None and final_effort is not None and theoretical_e is not None:
                abs_error = abs(final_effort - theoretical_e)

            # Detect seed from data or filename
            seed = data.get("seed")
            if seed is None:
                # Try to extract from filename
                for part in fname.split("_"):
                    if part.startswith("seed"):
                        try:
                            seed = int(part[4:])
                        except ValueError:
                            pass

            # Detect converged_step
            converged_step = data.get("converged_step") or fr.get("converged_step")

            runs[(experiment, q)].append({
                "seed": seed,
                "final_effort": final_effort,
                "theoretical_e": theoretical_e or get_theoretical_effort(experiment, q),
                "abs_error": abs_error,
                "converged_step": converged_step,
                "path": fpath,
            })

    return dict(runs)


def compute_metrics(run_list: List[dict]) -> dict:
    """Compute aggregate metrics for a list of runs."""
    efforts = [r["final_effort"] for r in run_list if r["final_effort"] is not None]
    errors = [r["abs_error"] for r in run_list if r["abs_error"] is not None]
    converged = [r for r in run_list if r["converged_step"] is not None]
    theoretical_e = run_list[0]["theoretical_e"] if run_list else 0.0

    n = len(run_list)
    n_converged = len(converged)

    if errors:
        mean_abs_err = sum(errors) / len(errors)
        rel_err = mean_abs_err / theoretical_e * 100 if theoretical_e > 0 else 0.0
    else:
        mean_abs_err = float("nan")
        rel_err = float("nan")

    if converged:
        mean_conv_step = sum(r["converged_step"] for r in converged) / len(converged)
    else:
        mean_conv_step = None

    return {
        "n_runs": n,
        "n_converged": n_converged,
        "mean_abs_err": mean_abs_err,
        "rel_err_pct": rel_err,
        "mean_conv_step": mean_conv_step,
        "theoretical_e": theoretical_e,
    }


EXPERIMENT_LABELS = {
    "two_players": "Two-Player",
    "three_players": "Three-Player",
    "different_cost": "Het. Cost",
    "different_ability": "Het. Ability",
}


def main():
    parser = argparse.ArgumentParser(description="Compare v1 vs v2 convergence results")
    parser.add_argument("--v1-tag", default="baseline", help="v1 ablation tag (default: baseline)")
    parser.add_argument("--v2-tag", default="baseline_v2", help="v2 ablation tag (default: baseline_v2)")
    parser.add_argument("--output", default="results/comparison_v1_v2.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"Loading v1 runs (tag={args.v1_tag})...")
    v1_runs = load_runs(args.v1_tag)
    print(f"  Found {sum(len(v) for v in v1_runs.values())} runs across {len(v1_runs)} (experiment, q) groups")

    print(f"Loading v2 runs (tag={args.v2_tag})...")
    v2_runs = load_runs(args.v2_tag)
    print(f"  Found {sum(len(v) for v in v2_runs.values())} runs across {len(v2_runs)} (experiment, q) groups")

    # Collect all keys
    all_keys = sorted(set(v1_runs.keys()) | set(v2_runs.keys()))

    if not all_keys:
        print("\nNo runs found for either tag. Nothing to compare.")
        return

    # Print comparison table
    header = f"{'Experiment':<15} {'q':>5}  {'v1 |e-e*|':>10} {'v1 Rel%':>8} {'v1 Conv':>8}  {'v2 |e-e*|':>10} {'v2 Rel%':>8} {'v2 Conv':>8}  {'Delta':>8}"
    print("\n" + "=" * len(header))
    print("Comparison: v1 (baseline) vs v2 (baseline_v2)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    csv_rows = []
    total_v1_rel = []
    total_v2_rel = []

    for exp, q in all_keys:
        label = EXPERIMENT_LABELS.get(exp, exp)

        v1_m = compute_metrics(v1_runs.get((exp, q), []))
        v2_m = compute_metrics(v2_runs.get((exp, q), []))

        def fmt_err(m):
            if m["n_runs"] == 0:
                return ("  -  ", "  -  ")
            return (f"{m['mean_abs_err']:10.2f}", f"{m['rel_err_pct']:7.1f}%")

        def fmt_conv(m):
            if m["n_runs"] == 0:
                return "  -  "
            if m["n_converged"] == 0:
                return "   NC"
            return f"{m['n_converged']}/{m['n_runs']}"

        v1_err, v1_rel = fmt_err(v1_m)
        v2_err, v2_rel = fmt_err(v2_m)
        v1_conv = fmt_conv(v1_m)
        v2_conv = fmt_conv(v2_m)

        # Delta in relative error
        if v1_m["n_runs"] > 0 and v2_m["n_runs"] > 0:
            delta = v2_m["rel_err_pct"] - v1_m["rel_err_pct"]
            delta_str = f"{delta:+7.1f}%"
            total_v1_rel.append(v1_m["rel_err_pct"])
            total_v2_rel.append(v2_m["rel_err_pct"])
        elif v1_m["n_runs"] > 0:
            delta_str = "  (v2 N/A)"
            total_v1_rel.append(v1_m["rel_err_pct"])
        else:
            delta_str = "  (v1 N/A)"

        print(f"{label:<15} {q:5.0f}  {v1_err} {v1_rel} {v1_conv:>8}  {v2_err} {v2_rel} {v2_conv:>8}  {delta_str}")

        csv_rows.append({
            "experiment": exp,
            "q": q,
            "v1_n_runs": v1_m["n_runs"],
            "v1_abs_err": v1_m["mean_abs_err"],
            "v1_rel_err_pct": v1_m["rel_err_pct"],
            "v1_n_converged": v1_m["n_converged"],
            "v1_mean_conv_step": v1_m["mean_conv_step"],
            "v2_n_runs": v2_m["n_runs"],
            "v2_abs_err": v2_m["mean_abs_err"],
            "v2_rel_err_pct": v2_m["rel_err_pct"],
            "v2_n_converged": v2_m["n_converged"],
            "v2_mean_conv_step": v2_m["mean_conv_step"],
        })

    print("-" * len(header))

    # Summary
    if total_v1_rel and total_v2_rel:
        mean_v1 = sum(total_v1_rel) / len(total_v1_rel)
        mean_v2 = sum(total_v2_rel) / len(total_v2_rel)
        print(f"{'Mean Rel Err':<15} {'':>5}  {'':>10} {mean_v1:7.1f}% {'':>8}  {'':>10} {mean_v2:7.1f}% {'':>8}  {mean_v2 - mean_v1:+7.1f}%")

    # Save CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
