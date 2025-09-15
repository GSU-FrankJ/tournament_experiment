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


def e_star_three_players(q: float, w_h: float, w_l: float, k: float) -> float:
    """Three-player single-stage benchmark effort per user specification.

    The user’s requirement states the same closed-form as the two-player
    single-stage formula:

        e*(q) = (w_H - w_L) / (4 q k)

    We expose it as a dedicated helper for clarity in run scripts.
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


# ---- Asymmetric-cost two-player formulas ----
def e_star_two_players_asymmetric_cost(q: float, w_h: float, w_l: float, k1: float, k2: float) -> tuple[float, float]:
    """Closed-form equilibrium efforts for k1 != k2, l1 = l2.

    Based on the user's provided expressions (reduces to symmetric case when k1=k2=k):

        e1* = 2 k2 q (w_H - w_L) / (8 k1 k2 q^2 - (k1 - k2) (w_H - w_L))
        e2* = 2 k1 q (w_H - w_L) / (8 k1 k2 q^2 - (k1 - k2) (w_H - w_L))

    Returns (e1*, e2*).
    """
    w_gap = float(w_h) - float(w_l)
    k1 = float(k1)
    k2 = float(k2)
    q = float(q)
    denom = (8.0 * k1 * k2 * q * q) - ((k1 - k2) * w_gap)
    if abs(denom) < 1e-12:
        # Fallback to symmetric-like approximation to avoid division by zero
        e_sym = e_star_two_players(q, w_h, w_l, (k1 + k2) / 2.0)
        return e_sym, e_sym
    e1 = (2.0 * k2 * q * w_gap) / denom
    e2 = (2.0 * k1 * q * w_gap) / denom
    return e1, e2


def eu_two_players_asymmetric_cost(q: float, w_h: float, w_l: float, k1: float, k2: float) -> tuple[float, float]:
    """Expected utilities at the asymmetric-cost equilibrium.

    Uses the exact expressions provided by the user:

        Eu1 = ((wH-wL)(32 k1^2 k2^2 q^4 - k1 k2 (16 k1 q^2 - 12 k2 q^2)(wH-wL) + (k1-k2)^2 (wH-wL)^2))
               / (8 k1 k2 q^2 - k1 (wH-wL) + k2 (wH-wL))^2

        Eu2 = ((4 k1)^2 k2 q^2 (8 k2 q^2 - (wH-wL))(wH-wL))
               / (8 k1 k2 q^2 - k1 (wH-wL) + k2 (wH-wL))^2

    Returns (Eu1, Eu2).
    """
    w_gap = float(w_h) - float(w_l)
    k1 = float(k1)
    k2 = float(k2)
    q = float(q)
    denom = (8.0 * k1 * k2 * q * q - k1 * w_gap + k2 * w_gap) ** 2
    if denom <= 0:
        return 0.0, 0.0
    eu1_num = (w_gap * (
        32.0 * (k1**2) * (k2**2) * (q**4)
        - k1 * k2 * ((16.0 * k1 * (q**2) - 12.0 * k2 * (q**2)) * w_gap)
        + ((k1 - k2) ** 2) * (w_gap ** 2)
    ))
    eu2_num = (((4.0 * k1) ** 2) * k2 * (q**2) * (8.0 * k2 * (q**2) - w_gap) * w_gap)
    return eu1_num / denom, eu2_num / denom













