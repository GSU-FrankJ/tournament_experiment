"""
TEL-PPO Paper Artifacts Pipeline

This module generates figures, tables, and underlying data for the TEL-PPO research paper.

Usage:
    python -m paper_artifacts.make_all --dry-run       # List discovered runs
    python -m paper_artifacts.make_all                  # Generate all artifacts
    python -m paper_artifacts.plots convergence_main   # Single figure

Structure:
    - run_registry.py: Discover runs and map to (method, q, seed, ablation)
    - extract.py: Load convergence JSON -> tidy DataFrames
    - metrics.py: Convergence step, exploitability summary, gaps
    - plots.py: matplotlib figures (PNG/PDF)
    - tables.py: CSV + LaTeX table generation
    - config.py: Constants, paths, style settings
"""

__version__ = "0.1.0"

from .config import (
    RESULTS_DIR,
    CONVERGENCE_DIR,
    OUTPUT_DIR,
    Q_VALUES,
    THEORY_PARAMS,
)

__all__ = [
    "RESULTS_DIR",
    "CONVERGENCE_DIR",
    "OUTPUT_DIR",
    "Q_VALUES",
    "THEORY_PARAMS",
]
