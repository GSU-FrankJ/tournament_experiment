#!/usr/bin/env python3
"""
Directory Structure Migration Script

Reorganizes the project from a flat results/ + paper_artifacts/ + paper_out/
structure into experiment-type-organized directories.

New structure:
    results/
        two_players/convergence/, logs/, summary.csv
        three_players/convergence/, logs/, summary.csv
        different_cost/convergence/, logs/, summary.csv
        different_ability/convergence/, logs/, summary.csv
        ablation/exploit_params/runs/
        ablation/mechanism/runs/
        plots/gradient/, ppo/, k5e4_wh8_wl3/
    paper/
        generator/  (was paper_artifacts/)
        figures/    (was paper_out/figures/)
        tables/     (was paper_out/tables/)
        data/       (was paper_out/data/)

Usage:
    # Preview what would be moved (no changes)
    python tools/migrate_structure.py --dry-run

    # Run migration
    python tools/migrate_structure.py

    # Run migration and remove empty source directories
    python tools/migrate_structure.py --cleanup
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Project root
ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def classify_convergence_file(filename: str) -> str:
    """Classify a convergence JSON file into an experiment type.

    Returns: 'two_players', 'three_players', 'different_cost', 'different_ability'
    """
    if filename.startswith("different_cost_"):
        return "different_cost"
    elif filename.startswith("different_ability_"):
        return "different_ability"
    elif filename.startswith("ppo_3p_") or filename.startswith("gradient_3p_"):
        return "three_players"
    else:
        # ppo_q*, gradient_q* -> two_players
        return "two_players"


def classify_log_file(filename: str) -> str:
    """Classify a log file into an experiment type."""
    lower = filename.lower()
    if "different_cost" in lower:
        return "different_cost"
    elif "different_ability" in lower:
        return "different_ability"
    elif "three_player" in lower:
        return "three_players"
    else:
        return "two_players"


def plan_convergence_moves(src_dir: Path, results_dir: Path) -> List[Tuple[Path, Path]]:
    """Plan moves for convergence JSON and metadata files."""
    moves = []
    if not src_dir.exists():
        return moves

    for f in sorted(src_dir.iterdir()):
        if not f.is_file():
            continue
        if f.suffix != ".json":
            continue

        experiment = classify_convergence_file(f.name)
        dest_dir = results_dir / experiment / "convergence"
        moves.append((f, dest_dir / f.name))

    return moves


def plan_log_moves(src_dir: Path, results_dir: Path) -> List[Tuple[Path, Path]]:
    """Plan moves for training log files."""
    moves = []
    if not src_dir.exists():
        return moves

    for f in sorted(src_dir.iterdir()):
        if not f.is_file():
            continue

        experiment = classify_log_file(f.name)
        dest_dir = results_dir / experiment / "logs"
        moves.append((f, dest_dir / f.name))

    return moves


def plan_csv_moves(results_dir: Path) -> List[Tuple[Path, Path]]:
    """Plan CSV renames."""
    moves = []

    csv_mappings = {
        "one_stage_two_players_v2.csv": "two_players/summary.csv",
        "one_stage_two_players.csv": "two_players/summary_legacy.csv",
        "different_cost_two_players.csv": "different_cost/summary.csv",
        "different_ability_two_players.csv": "different_ability/summary.csv",
    }

    for old_name, new_path in csv_mappings.items():
        src = results_dir / old_name
        if src.exists():
            moves.append((src, results_dir / new_path))

    return moves


def plan_plot_moves(results_dir: Path) -> List[Tuple[Path, Path]]:
    """Plan convergence_plots -> plots moves."""
    moves = []
    src_dir = results_dir / "convergence_plots"
    if not src_dir.exists():
        return moves

    for item in sorted(src_dir.rglob("*")):
        if item.is_file():
            rel = item.relative_to(src_dir)
            dest = results_dir / "plots" / rel
            moves.append((item, dest))

    return moves


def plan_exploit_ablation_moves(results_dir: Path) -> List[Tuple[Path, Path]]:
    """Plan exploit_ablation -> ablation/exploit_params moves."""
    moves = []
    src_dir = results_dir / "exploit_ablation"
    if not src_dir.exists():
        return moves

    for item in sorted(src_dir.rglob("*")):
        if item.is_file():
            rel = item.relative_to(src_dir)
            dest = results_dir / "ablation" / "exploit_params" / rel
            moves.append((item, dest))

    return moves


def plan_paper_moves(root: Path) -> List[Tuple[Path, Path]]:
    """Plan paper_artifacts -> paper/generator and paper_out -> paper/ moves."""
    moves = []

    # paper_artifacts/ -> paper/generator/
    src_dir = root / "paper_artifacts"
    if src_dir.exists():
        for f in sorted(src_dir.iterdir()):
            if f.is_file():
                moves.append((f, root / "paper" / "generator" / f.name))

    # paper_out/ -> paper/
    src_dir = root / "paper_out"
    if src_dir.exists():
        for item in sorted(src_dir.rglob("*")):
            if item.is_file():
                rel = item.relative_to(src_dir)
                moves.append((item, root / "paper" / rel))

    return moves


def plan_misc_moves(results_dir: Path) -> List[Tuple[Path, Path]]:
    """Plan moves for miscellaneous results files (PNG, etc.)."""
    moves = []
    misc_files = [
        "one_stage_two_players.png",
        "convergence_comparison.png",
        "convergence_separate_agents.png",
        "exploit_ablation_sweep.log",
    ]

    for fname in misc_files:
        src = results_dir / fname
        if src.exists():
            moves.append((src, results_dir / "plots" / fname))

    return moves


def execute_moves(
    moves: List[Tuple[Path, Path]],
    dry_run: bool = True,
    label: str = "",
) -> int:
    """Execute file moves. Returns count of moves executed."""
    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")

    if not moves:
        print("  (no files to move)")
        return 0

    count = 0
    for src, dest in moves:
        rel_src = src.relative_to(ROOT)
        rel_dest = dest.relative_to(ROOT)

        if dry_run:
            print(f"  [DRY RUN] {rel_src} -> {rel_dest}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                print(f"  [SKIP] {rel_dest} already exists")
                continue
            shutil.copy2(str(src), str(dest))
            print(f"  [MOVED] {rel_src} -> {rel_dest}")
        count += 1

    return count


def create_new_directories(results_dir: Path, root: Path, dry_run: bool = True):
    """Create the new directory structure."""
    dirs = [
        results_dir / "two_players" / "convergence",
        results_dir / "two_players" / "logs",
        results_dir / "three_players" / "convergence",
        results_dir / "three_players" / "logs",
        results_dir / "different_cost" / "convergence",
        results_dir / "different_cost" / "logs",
        results_dir / "different_ability" / "convergence",
        results_dir / "different_ability" / "logs",
        results_dir / "ablation" / "exploit_params" / "runs",
        results_dir / "ablation" / "mechanism" / "runs",
        results_dir / "plots" / "gradient",
        results_dir / "plots" / "ppo",
        root / "paper" / "generator",
        root / "paper" / "figures",
        root / "paper" / "tables",
        root / "paper" / "data",
    ]

    print(f"\n{'=' * 60}")
    print("  Creating directories")
    print(f"{'=' * 60}")

    for d in dirs:
        rel = d.relative_to(ROOT)
        if dry_run:
            if not d.exists():
                print(f"  [DRY RUN] mkdir {rel}")
        else:
            d.mkdir(parents=True, exist_ok=True)
            if not d.exists():
                print(f"  [CREATED] {rel}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate directory structure to experiment-type organization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview moves without executing (default)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the migration (copy files)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove empty source directories after migration",
    )

    args = parser.parse_args()

    # Default to dry run unless --execute is specified
    dry_run = not args.execute

    if dry_run:
        print("=" * 60)
        print("  DRY RUN MODE (use --execute to apply)")
        print("=" * 60)
    else:
        print("=" * 60)
        print("  EXECUTING MIGRATION")
        print("=" * 60)

    results_dir = ROOT / "results"

    # 1. Create directory structure
    create_new_directories(results_dir, ROOT, dry_run)

    total = 0

    # 2. Convergence JSON files
    moves = plan_convergence_moves(results_dir / "convergence_history", results_dir)
    total += execute_moves(moves, dry_run, "Convergence JSON files")

    # 3. Log files
    moves = plan_log_moves(results_dir / "logs", results_dir)
    total += execute_moves(moves, dry_run, "Training log files")

    # 4. CSV renames
    moves = plan_csv_moves(results_dir)
    total += execute_moves(moves, dry_run, "CSV file renames")

    # 5. Convergence plots
    moves = plan_plot_moves(results_dir)
    total += execute_moves(moves, dry_run, "Convergence plots")

    # 6. Exploit ablation
    moves = plan_exploit_ablation_moves(results_dir)
    total += execute_moves(moves, dry_run, "Exploit ablation results")

    # 7. Misc results files
    moves = plan_misc_moves(results_dir)
    total += execute_moves(moves, dry_run, "Miscellaneous results files")

    # 8. Paper artifacts/output
    moves = plan_paper_moves(ROOT)
    total += execute_moves(moves, dry_run, "Paper artifacts & outputs")

    # Summary
    print(f"\n{'=' * 60}")
    action = "would be copied" if dry_run else "copied"
    print(f"  Total: {total} files {action}")
    print(f"{'=' * 60}")

    if dry_run:
        print("\nRun with --execute to apply these changes.")
        print("Note: Original files are preserved (copy, not move).")
        print("After verifying, manually remove old directories.")


if __name__ == "__main__":
    main()
