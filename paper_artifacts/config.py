"""
Configuration for paper artifacts generation.

Contains:
- Directory paths
- Theory parameters (e*, convergence thresholds)
- Plotting styles
- Ablation flag semantics
"""

import os
from typing import Dict, Tuple

# ==============================================================================
# Directory Paths
# ==============================================================================

# Workspace root (detect dynamically)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(_SCRIPT_DIR)

# Input directories
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "results")
CONVERGENCE_DIR = os.path.join(RESULTS_DIR, "convergence_history")
CSV_PATH = os.path.join(RESULTS_DIR, "one_stage_two_players_v2.csv")

# Output directory (paper_out/)
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "paper_out")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")


# ==============================================================================
# Theory Parameters
# ==============================================================================

# Game parameters (tournament model)
THEORY_PARAMS: Dict[str, float] = {
    "w_h": 6.5,      # High prize
    "w_l": 3.0,      # Low prize
    "k": 0.0004,     # Cost coefficient
}

# Theoretical equilibrium effort formula: e* = (w_H - w_L) / (4 * q * k)
def e_star(q: float, w_h: float = 6.5, w_l: float = 3.0, k: float = 0.0004) -> float:
    """Compute theoretical equilibrium effort for noise parameter q."""
    return (w_h - w_l) / (4.0 * q * k)


# Standard q values for experiments
Q_VALUES = [25.0, 40.0, 55.0]


# ==============================================================================
# Convergence Criteria
# ==============================================================================

CONVERGENCE_CONFIG: Dict[str, float] = {
    # Effort convergence: within delta of e* for window consecutive updates
    "effort_delta": 0.5,      # |e - e*| < delta
    "effort_window": 20,      # Consecutive updates
    
    # Exploitability convergence: < threshold for patience consecutive evals
    "exploit_threshold": 0.05,
    "exploit_patience": 5,
    
    # Minimum steps before declaring convergence
    "min_steps": 100,
}


# ==============================================================================
# Cheap-Gate Thresholds
# ==============================================================================

CHEAP_GATE_CONFIG: Dict[str, float] = {
    "mean_kl_thresh": 0.0045,
    "std_kl_thresh": 0.0035,
    "drift_effort_thresh": 2.0,
    "patience_drift": 2,
}


# ==============================================================================
# Quality Classification
# ==============================================================================

def classify_quality(gap: float) -> str:
    """Classify convergence quality based on gap from theoretical effort."""
    if gap < 0.5:
        return "Excellent"
    elif gap < 1.0:
        return "Good"
    elif gap < 5.0:
        return "Fair"
    else:
        return "Poor"


# ==============================================================================
# Plotting Styles
# ==============================================================================

# Method colors (consistent across all figures)
METHOD_COLORS: Dict[str, str] = {
    "Theory": "black",
    "Gradient": "#1f77b4",  # Blue
    "TEL-PPO": "#ff7f0e",   # Orange
    "PPO": "#ff7f0e",       # Alias for TEL-PPO
}

# Method line styles
METHOD_LINESTYLES: Dict[str, str] = {
    "Theory": "--",        # Dashed
    "Gradient": "-",       # Solid
    "TEL-PPO": "-",        # Solid
    "PPO": "-",            # Alias
}

# Ablation colors
ABLATION_COLORS: Dict[str, str] = {
    "baseline": "#1f77b4",
    "no_cheap_gate": "#ff7f0e",
    "no_exploitability": "#2ca02c",
}

# Figure sizes (inches)
FIGURE_SIZES: Dict[str, Tuple[float, float]] = {
    "convergence_main": (12, 4),       # 1x3 grid
    "kl_dynamics": (10, 4),
    "exploitability_dynamics": (10, 4),
    "beta_evolution": (10, 4),
    "beta_snapshots": (12, 4),
    "ablation_comparison": (10, 6),
}

# Font sizes
FONT_SIZES: Dict[str, int] = {
    "title": 14,
    "axis_label": 12,
    "tick_label": 10,
    "legend": 10,
    "annotation": 9,
}

# DPI for raster output
OUTPUT_DPI = 300


# ==============================================================================
# Ablation Flag Semantics (Reference)
# ==============================================================================

ABLATION_FLAGS_DOC = """
Ablation Flag Semantics:

--exploit-every-updates N (default: 10)
    Maximum interval between exploitability evaluations.
    Caps worst-case cost when cheap-gate is disabled.
    Gate can still trigger eval earlier if it passes.

--disable-cheap-gate
    Gate is always ON → exploitability eligible every update.
    Combined with --exploit-every-updates N, evals every N updates guaranteed.
    Without the interval cap, would eval EVERY update (too expensive).
    Use case: Ablation to measure cheap-gate benefit.

--disable-exploitability
    Completely skip exploitability computation.
    No exploitability eval, convergence based on effort gap only.
    All exploitability values = NaN, exploitability_is_valid = False.
    Overrides --exploit-every-updates (no evals regardless).
    Use case: Ablation to measure exploitability term benefit.

--ablation-name <name>
    Required in every record (JSON and CSV).
    Default: "baseline"
    Must appear in: convergence JSON, CSV row, metadata.json

Flag Interaction Matrix:
| disable-cheap-gate | disable-exploitability | exploit-every-updates | Behavior |
|-------------------|------------------------|----------------------|----------|
| False             | False                  | 10                   | Normal: gate controls, but at least every 10 updates |
| True              | False                  | 10                   | Eval every 10 updates (gate always passes) |
| True              | False                  | 1                    | Eval every update (expensive, for debugging) |
| False             | True                   | any                  | Never eval, converge on effort only |
| True              | True                   | any                  | Never eval (disable-exploitability wins) |
"""
