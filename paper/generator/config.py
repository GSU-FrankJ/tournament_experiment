"""
Configuration for paper artifacts generation.

Contains:
- Directory paths
- Theory parameters (e*, convergence thresholds)
- Plotting styles
- Ablation flag semantics
"""

import os
from typing import Dict, List, Tuple

# ==============================================================================
# Directory Paths
# ==============================================================================

# Workspace root (detect dynamically: paper/generator/ is 2 levels from root)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

# Results root
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "results")

# Per-experiment convergence directories
CONVERGENCE_DIRS = {
    "two_players": os.path.join(RESULTS_DIR, "two_players", "convergence"),
    "three_players": os.path.join(RESULTS_DIR, "three_players", "convergence"),
    "different_cost": os.path.join(RESULTS_DIR, "different_cost", "convergence"),
    "different_ability": os.path.join(RESULTS_DIR, "different_ability", "convergence"),
}

# Legacy convergence dir (backward compat for single-dir discovery)
CONVERGENCE_DIR = os.path.join(RESULTS_DIR, "two_players", "convergence")

# Per-experiment CSV paths
CSV_PATHS = {
    "two_players": os.path.join(RESULTS_DIR, "two_players", "summary.csv"),
    "different_cost": os.path.join(RESULTS_DIR, "different_cost", "summary.csv"),
    "different_ability": os.path.join(RESULTS_DIR, "different_ability", "summary.csv"),
}
CSV_PATH = CSV_PATHS["two_players"]

# Output directory (paper/)
_PAPER_DIR = os.path.dirname(_SCRIPT_DIR)
OUTPUT_DIR = _PAPER_DIR
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
Q_VALUES: List[float] = [35.0, 40.0, 55.0]


# ==============================================================================
# Baseline Overrides
# ==============================================================================
# Map (experiment, q) → ablation name that should be treated as "baseline".
# Used when the default "baseline" runs are known to be broken and a different
# ablation variant is the correct primary result.
#
# Rationale: theory_align_v2 (the default PPO mode) fails at q=55 because
# concentration grows unchecked and kills the learning signal.  Standard mode
# with entropy_end=0.002 ("no_tv2_ent002") fixes this.  See docs/tasks/
# q55-convergence/STATE.md for details.
BASELINE_OVERRIDES: Dict[Tuple[str, float], str] = {
    ("two_players", 55.0): "no_tv2_ent002",
}


def format_q(q: float) -> str:
    """Format q value without .0 suffix (e.g., 35.0 -> 'q = 35')."""
    return f"q = {int(q)}" if q == int(q) else f"q = {q}"


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

# Global shading alpha for fill_between bands (lighter than previous defaults)
SHADE_ALPHA = 0.15

# Theory line style (used across all figures)
THEORY_LINE_COLOR = "red"
THEORY_LINE_WIDTH = 2.5

# Convergence vertical line style (marks detected convergence step)
CONV_VLINE_COLOR = "#888888"
CONV_VLINE_LINESTYLE = "--"
CONV_VLINE_LINEWIDTH = 1.0

# Method colors (consistent across all figures)
METHOD_COLORS: Dict[str, str] = {
    "Theory": THEORY_LINE_COLOR,
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

# Ablation colors (keyed by internal ablation_name from convergence JSON)
ABLATION_COLORS: Dict[str, str] = {
    "baseline": "#1f77b4",
    "no_cheap_gate": "#ff7f0e",
    "no_exploitability": "#2ca02c",
}

# Ablation display labels (internal name -> paper label)
ABLATION_LABELS: Dict[str, str] = {
    "baseline": "TEL-PPO",
    "no_cheap_gate": "No stability screening",
    "no_exploitability": "No exploitability verification",
}

# Ablation line widths (TEL-PPO thicker, ablation variants thinner)
ABLATION_LINEWIDTHS: Dict[str, float] = {
    "baseline": 2.5,
    "no_cheap_gate": 1.5,
    "no_exploitability": 1.5,
}

# Figure sizes (inches)
FIGURE_SIZES: Dict[str, Tuple[float, float]] = {
    "convergence_main": (12, 8),       # 2x3 grid
    "kl_dynamics": (10, 4),
    "exploitability_dynamics": (10, 4),
    "beta_evolution": (10, 4),
    "beta_snapshots": (12, 4),
    "ablation_comparison": (10, 6),
    "hyperparam_sensitivity": (14, 8),
    "equilibrium_recovery_dotplot": (10, 6),
}

# Font sizes
FONT_SIZES: Dict[str, int] = {
    "title": 14,
    "axis_label": 12,
    "tick_label": 10,
    "legend": 10,
    "annotation": 9,
}

# Per-agent colors and markers (for heterogeneous experiments)
AGENT_COLORS: Dict[str, str] = {
    "agent1": "#1f77b4",
    "agent2": "#ff7f0e",
}

AGENT_MARKERS: Dict[str, str] = {
    "agent1": "o",
    "agent2": "^",
}

# Weight-variant labels for convergence figure rows (include k value)
WEIGHT_VARIANT_LABELS: Dict[str, str] = {
    "baseline": r"$w_H=6.5,\; w_L=3.0,\; k=0.0004$",
    "wh8_wl4": r"$w_H=8,\; w_L=4,\; k=0.0004$",
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
