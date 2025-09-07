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
    """Closed-form symmetric benchmark (legacy, denominator 6).

    Kept for backward compatibility with two-stage code paths that still
    reference the spec using denominator 6. For two-player single-stage
    experiments, use ``e_star_two_players`` instead (denominator 4).
    """
    return (w_h - w_l) / (6.0 * q * k)


def e_star_two_players(q: float, w_h: float, w_l: float, k: float) -> float:
    """Two-player single-stage benchmark effort with denominator 4.

    e*(q) = (w_H - w_L) / (4 q k)
    """
    return (w_h - w_l) / (4.0 * q * k)


def clip_stage1(e: float, bounds: Tuple[float, float] = (0.0, 100.0)) -> float:
    """Clip effort to Stage-1 bounds [0, 100]."""
    lo, hi = bounds
    return max(lo, min(hi, float(e)))


def clip_stage2(e: float, bounds: Tuple[float, float] = (0.0, 200.0)) -> float:
    """Clip effort to Stage-2 bounds [0, 200]."""
    lo, hi = bounds
    return max(lo, min(hi, float(e)))









