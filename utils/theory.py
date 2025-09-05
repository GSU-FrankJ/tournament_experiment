"""
Theoretical benchmarks and helper utilities.

Implements the closed-form symmetric benchmark for the two-player tournament:

    e*(q) = (w_H - w_L) / (6 q k)

Stage-specific clipping is handled by convenience helpers to ensure efforts
respect the stage effort bounds.
"""

from __future__ import annotations

from typing import Tuple


def e_star(q: float, w_h: float, w_l: float, k: float) -> float:
    """Closed-form symmetric benchmark effort.

    Args:
        q: Noise parameter (> 0)
        w_h: High prize
        w_l: Low prize
        k: Cost parameter

    Returns:
        Theoretical symmetric effort e* (unclipped).
    """
    return (w_h - w_l) / (6.0 * q * k)


def clip_stage1(e: float, bounds: Tuple[float, float] = (0.0, 100.0)) -> float:
    """Clip effort to Stage-1 bounds [0, 100]."""
    lo, hi = bounds
    return max(lo, min(hi, float(e)))


def clip_stage2(e: float, bounds: Tuple[float, float] = (0.0, 200.0)) -> float:
    """Clip effort to Stage-2 bounds [0, 200]."""
    lo, hi = bounds
    return max(lo, min(hi, float(e)))






