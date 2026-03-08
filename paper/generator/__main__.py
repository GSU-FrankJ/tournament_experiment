"""
CLI for Paper Artifacts Generation.

Usage:
    # Full pipeline (dry-run first to see what will be generated)
    python -m paper.generator --dry-run

    # Generate all artifacts
    python -m paper.generator make_all

    # Generate specific figure
    python -m paper.generator plot convergence_main

    # Generate specific table
    python -m paper.generator table summary_metrics

    # Custom paths (scans all results/*/convergence/ dirs by default)
    python -m paper.generator make_all \
        --runs-dir results/two_players/convergence \
        --csv results/two_players/summary.csv \
        --out-dir paper \
        --q 25,40,55
"""

import os
import sys
import argparse
from typing import List, Optional

from .config import (
    CONVERGENCE_DIR,
    CSV_PATH,
    OUTPUT_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    DATA_DIR,
    Q_VALUES,
)
from .run_registry import discover_runs, print_discovery_report, select_best_runs
from .extract import load_all_convergence_data
from .plots import (
    generate_all_figures,
    plot_convergence_main,
    plot_kl_dynamics,
    plot_exploitability_dynamics,
    plot_beta_evolution,
    plot_beta_snapshots,
    plot_ablation_comparison,
    plot_distance_to_equilibrium,
    plot_effort_drift,
    plot_equilibrium_recovery_dotplot,
)
from .tables import (
    generate_all_tables,
    generate_summary_metrics_table,
    generate_ablation_table,
    generate_final_paper_table,
    generate_convergence_comparison_table,
    generate_environment_config_table,
)


# Available plot types
PLOT_TYPES = [
    "convergence_main",
    "kl_dynamics",
    "exploitability_dynamics",
    "beta_evolution",
    "beta_snapshots",
    "ablation_comparison",
    "distance_to_equilibrium",
    "effort_drift",
    "equilibrium_recovery_dotplot",
]

# Available table types
TABLE_TYPES = [
    "environment_config",
    "summary_metrics",
    "ablation_results",
    "final_summary",
    "convergence_comparison",
]


def parse_q_values(q_str: str) -> List[float]:
    """Parse comma-separated q values."""
    return [float(q.strip()) for q in q_str.split(",")]


def cmd_dry_run(args: argparse.Namespace) -> int:
    """Dry-run: list discovered runs without generating anything."""
    print("=" * 60)
    print("DRY RUN: Discovering runs...")
    print("=" * 60)
    
    runs = discover_runs(
        convergence_dir=args.runs_dir,
        csv_path=args.csv,
    )
    
    print_discovery_report(runs)
    
    print("\n" + "=" * 60)
    print("WOULD GENERATE:")
    print("=" * 60)
    
    print("\nFigures:")
    for plot_type in PLOT_TYPES:
        print(f"  - {plot_type}.png")
        print(f"  - {plot_type}.pdf")
    
    print("\nTables:")
    for table_type in TABLE_TYPES:
        print(f"  - {table_type}.csv")
        print(f"  - {table_type}.tex")
    
    print("\nData files:")
    for plot_type in PLOT_TYPES:
        print(f"  - {plot_type}.csv")
    
    print(f"\nOutput directory: {args.out_dir}")
    
    return 0


def cmd_make_all(args: argparse.Namespace) -> int:
    """Generate all figures and tables."""
    print("=" * 60)
    print("GENERATING ALL PAPER ARTIFACTS")
    print("=" * 60)
    
    # Setup output directories
    os.makedirs(args.out_dir, exist_ok=True)
    figures_dir = os.path.join(args.out_dir, "figures")
    tables_dir = os.path.join(args.out_dir, "tables")
    data_dir = os.path.join(args.out_dir, "data")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Parse q values
    q_values = parse_q_values(args.q) if args.q else Q_VALUES
    print(f"Q values: {q_values}")
    print(f"Output dir: {args.out_dir}")
    
    # Load data
    print("\nLoading convergence data...")
    if args.best_only:
        from .extract import load_multiple_runs
        all_runs = discover_runs(
            convergence_dir=args.runs_dir,
            csv_path=args.csv,
        )
        # Filter to requested q values
        if q_values:
            all_runs = [r for r in all_runs if r.q in q_values]
        print(f"\nSelecting best run per (experiment, q)...")
        best = select_best_runs(all_runs)
        print_discovery_report(best)
        df = load_multiple_runs(best)
    else:
        df = load_all_convergence_data(
            convergence_dir=args.runs_dir,
            csv_path=args.csv,
            q_values=q_values,
        )
    print(f"Loaded {len(df)} rows from {df['method'].nunique()} methods")
    
    if df.empty:
        print("\nERROR: No data found. Check your --runs-dir and --csv paths.")
        return 1
    
    # Generate figures
    print("\n" + "-" * 40)
    print("GENERATING FIGURES")
    print("-" * 40)
    figure_results = generate_all_figures(df, q_values, args.out_dir)
    
    # Generate tables
    print("\n" + "-" * 40)
    print("GENERATING TABLES")
    print("-" * 40)
    table_results = generate_all_tables(df, tables_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nGenerated {len(figure_results)} figures:")
    for name, path in figure_results.items():
        print(f"  {name}: {path}")
    
    print(f"\nGenerated {len(table_results)} tables:")
    for name, (csv_path, tex_path) in table_results.items():
        print(f"  {name}: {csv_path}")
    
    print(f"\nAll artifacts saved to: {args.out_dir}")
    
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    """Generate a specific figure."""
    plot_type = args.plot_type
    
    if plot_type not in PLOT_TYPES:
        print(f"ERROR: Unknown plot type '{plot_type}'")
        print(f"Available: {', '.join(PLOT_TYPES)}")
        return 1
    
    # Load data
    q_values = parse_q_values(args.q) if args.q else Q_VALUES
    if args.best_only:
        from .extract import load_multiple_runs
        all_runs = discover_runs(
            convergence_dir=args.runs_dir,
            csv_path=args.csv,
        )
        if q_values:
            all_runs = [r for r in all_runs if r.q in q_values]
        best = select_best_runs(all_runs)
        df = load_multiple_runs(best)
    else:
        df = load_all_convergence_data(
            convergence_dir=args.runs_dir,
            csv_path=args.csv,
            q_values=q_values,
        )

    if df.empty:
        print("ERROR: No data found")
        return 1

    # Generate specific plot
    output_dir = args.out_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    plot_funcs = {
        "convergence_main": lambda: plot_convergence_main(df, q_values, os.path.join(output_dir, "convergence_main.png")),
        "kl_dynamics": lambda: plot_kl_dynamics(df, q_values, os.path.join(output_dir, "kl_dynamics.png")),
        "exploitability_dynamics": lambda: plot_exploitability_dynamics(df, q_values, os.path.join(output_dir, "exploitability_dynamics.png")),
        "beta_evolution": lambda: plot_beta_evolution(df, q_values, os.path.join(output_dir, "beta_evolution.png")),
        "beta_snapshots": lambda: plot_beta_snapshots(df, q=40.0, output_path=os.path.join(output_dir, "beta_snapshots.png")),
        "ablation_comparison": lambda: plot_ablation_comparison(df, q_values, os.path.join(output_dir, "ablation_comparison.png")),
        "distance_to_equilibrium": lambda: plot_distance_to_equilibrium(df, q_values, os.path.join(output_dir, "distance_to_equilibrium.png")),
        "effort_drift": lambda: plot_effort_drift(df, q_values, os.path.join(output_dir, "effort_drift.png")),
        "equilibrium_recovery_dotplot": lambda: plot_equilibrium_recovery_dotplot(df, os.path.join(output_dir, "equilibrium_recovery_dotplot.png")),
    }
    
    fig, path = plot_funcs[plot_type]()
    if path:
        print(f"Generated: {path}")
        return 0
    else:
        print(f"Failed to generate {plot_type}")
        return 1


def cmd_table(args: argparse.Namespace) -> int:
    """Generate a specific table."""
    table_type = args.table_type
    
    if table_type not in TABLE_TYPES:
        print(f"ERROR: Unknown table type '{table_type}'")
        print(f"Available: {', '.join(TABLE_TYPES)}")
        return 1
    
    # Load data
    q_values = parse_q_values(args.q) if args.q else Q_VALUES
    if args.best_only:
        from .extract import load_multiple_runs
        all_runs = discover_runs(
            convergence_dir=args.runs_dir,
            csv_path=args.csv,
        )
        if q_values:
            all_runs = [r for r in all_runs if r.q in q_values]
        best = select_best_runs(all_runs)
        df = load_multiple_runs(best)
    else:
        df = load_all_convergence_data(
            convergence_dir=args.runs_dir,
            csv_path=args.csv,
            q_values=q_values,
        )

    if df.empty:
        print("ERROR: No data found")
        return 1

    # Generate specific table
    output_dir = args.out_dir or TABLES_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    table_funcs = {
        "environment_config": lambda: generate_environment_config_table(output_dir),
        "summary_metrics": lambda: generate_summary_metrics_table(df, output_dir),
        "ablation_results": lambda: generate_ablation_table(df, output_dir),
        "final_summary": lambda: generate_final_paper_table(df, output_dir),
        "convergence_comparison": lambda: generate_convergence_comparison_table(df, output_dir),
    }
    
    paths = table_funcs[table_type]()
    if paths and paths[0]:
        print(f"Generated CSV: {paths[0]}")
        print(f"Generated LaTeX: {paths[1]}")
        return 0
    else:
        print(f"Failed to generate {table_type}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="TEL-PPO Paper Artifacts Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Global arguments
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Path to a specific convergence directory (default: scan all results/*/convergence/)",
    )
    parser.add_argument(
        "--csv",
        default=CSV_PATH,
        help=f"Path to results CSV (default: {CSV_PATH})",
    )
    parser.add_argument(
        "--out-dir",
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--q",
        default=None,
        help="Comma-separated q values (default: 25,40,55)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered runs without generating anything",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Select only the best seed per (experiment, q) based on lowest final effort error",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # make_all
    make_all_parser = subparsers.add_parser("make_all", help="Generate all figures and tables")
    
    # plot
    plot_parser = subparsers.add_parser("plot", help="Generate a specific figure")
    plot_parser.add_argument(
        "plot_type",
        choices=PLOT_TYPES,
        help="Type of plot to generate",
    )
    
    # table
    table_parser = subparsers.add_parser("table", help="Generate a specific table")
    table_parser.add_argument(
        "table_type",
        choices=TABLE_TYPES,
        help="Type of table to generate",
    )
    
    # discover
    discover_parser = subparsers.add_parser("discover", help="Discover and report runs")
    
    args = parser.parse_args()
    
    # Handle commands
    if args.dry_run or args.command == "discover":
        return cmd_dry_run(args)
    elif args.command == "make_all":
        return cmd_make_all(args)
    elif args.command == "plot":
        return cmd_plot(args)
    elif args.command == "table":
        return cmd_table(args)
    else:
        # Default: make_all
        return cmd_make_all(args)


if __name__ == "__main__":
    sys.exit(main())
