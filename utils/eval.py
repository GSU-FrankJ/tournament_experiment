"""
Evaluation helpers for convergence metrics and gap calculations.
"""

from __future__ import annotations

from typing import Dict


def gap_from_theoretical(actual: float, theoretical: float) -> float:
    """Absolute gap from theoretical value."""
    return abs(float(actual) - float(theoretical))


def relative_error(actual: float, theoretical: float) -> float:
    """Relative error |a - t| / max(t, 1e-9)."""
    denom = max(abs(float(theoretical)), 1e-9)
    return abs(float(actual) - float(theoretical)) / denom


def convergence_quality_from_gap(gap: float) -> str:
    """Map absolute gap to quality buckets per README policy."""
    if gap < 0.5:
        return "Excellent"
    if gap < 1.0:
        return "Good"
    if gap < 5.0:
        return "Fair"
    return "Poor"


def build_csv_row(
    stage1_weight: float,
    stage2_weight: float,
    k1: float,
    k2: float,
    information_revelation: str,
    theoretical_stage1_effort: float,
    theoretical_stage2_effort: float,
    model_training: str,
    final_stage1_effort: float,
    final_stage2_effort: float,
    episodes: int,
) -> Dict[str, object]:
    """Build a standardized CSV row matching the exact required header."""
    final_weighted_effort = stage1_weight * float(final_stage1_effort) + stage2_weight * float(final_stage2_effort)
    theoretical_weighted = stage1_weight * float(theoretical_stage1_effort) + stage2_weight * float(theoretical_stage2_effort)
    gap = gap_from_theoretical(final_weighted_effort, theoretical_weighted)
    return {
        "stage1_weight": stage1_weight,
        "stage2_weight": stage2_weight,
        "k1": k1,
        "k2": k2,
        "information_revelation": information_revelation,
        "theoretical_stage1_effort": theoretical_stage1_effort,
        "theoretical_stage2_effort": theoretical_stage2_effort,
        "Model_training": model_training,
        "final_stage1_effort": final_stage1_effort,
        "final_stage2_effort": final_stage2_effort,
        "final_weighted_effort": final_weighted_effort,
        "Convergence_Quality": convergence_quality_from_gap(gap),
        "episodes": episodes,
        "Gap_from_theoretical": gap,
    }








