"""
Probability utilities for tournament experiments.

Implements the closed-form win probability under independent Uniform(-q, q) noise:

Let d = e_i - e_j. Then

    p(d, q) =
        0,                                      if d <= -2q
        1/2 + d/(2q) - d*|d|/(8 q^2),           if |d| < 2q
        1,                                      if d >= 2q

All functions are vectorized-friendly where applicable.
"""

from __future__ import annotations

from typing import Union
import numpy as np

Number = Union[int, float, np.ndarray]


def p_from_diff(d: Number, q: float) -> Number:
    """Compute win probability given d = e_i - e_j and noise half-width q.

    This implements the exact piecewise function specified in the project brief.

    Args:
        d: Effort difference e_i - e_j (scalar or numpy array)
        q: Noise parameter (> 0)

    Returns:
        Probability p that i wins against j (same shape as d).
    """
    d = np.asarray(d)
    out = np.zeros_like(d, dtype=float)

    # Regions
    mask_low = d <= -2.0 * q
    mask_high = d >= 2.0 * q
    mask_mid = (~mask_low) & (~mask_high)

    # Middle formula: 1/2 + d/(2q) - d*|d|/(8 q^2)
    if np.any(mask_mid):
        d_mid = d[mask_mid]
        out[mask_mid] = 0.5 + (d_mid / (2.0 * q)) - (d_mid * np.abs(d_mid)) / (8.0 * q * q)

    # Saturation regions
    out[mask_low] = 0.0
    out[mask_high] = 1.0

    return out if out.shape != () else float(out)


def p_from_efforts(e_i: Number, e_j: Number, q: float) -> Number:
    """Compute win probability from efforts e_i and e_j under Uniform(-q, q) noise.

    Args:
        e_i: Effort of player i (scalar or numpy array)
        e_j: Effort of player j (scalar or numpy array)
        q: Noise parameter (> 0)

    Returns:
        Probability p that i wins against j (broadcasted shape of e_i and e_j).
    """
    e_i = np.asarray(e_i)
    e_j = np.asarray(e_j)
    d = e_i - e_j
    return p_from_diff(d, q)


def _clip_scalar(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _primitive_g(eps: float, c: float, q: float) -> float:
    """Primitive of g(eps + c) with respect to eps."""
    x = eps + c
    if x <= -q:
        return 0.0
    if x >= q:
        # Continuation of linear tail; slope is 1.
        return eps + c
    return ((x + q) ** 2) / (4.0 * q)


def _poly_integral(eps: float, a: float, b: float, q: float) -> float:
    """Primitive where both terms are in the linear regime."""
    denom = 4.0 * q * q
    return (
        (eps ** 3) / 3.0
        + 0.5 * (a + b + 2.0 * q) * (eps ** 2)
        + (a + q) * (b + q) * eps
    ) / denom


def _mode(value: float, q: float, tol: float = 1e-12) -> str:
    if value <= -q + tol:
        return "zero"
    if value >= q - tol:
        return "one"
    return "linear"


def _integrate_product(a: float, b: float, q: float, lo: float, hi: float) -> float:
    """Integrate g(eps+a) * g(eps+b) over [lo, hi]."""
    if hi <= lo:
        return 0.0

    points = [lo, hi]
    for bound in (-q - a, q - a, -q - b, q - b):
        if lo < bound < hi:
            points.append(bound)
    points = sorted(points)

    total = 0.0
    for start, end in zip(points[:-1], points[1:]):
        seg_lo = start
        seg_hi = end
        if seg_hi <= seg_lo:
            continue
        mid = 0.5 * (seg_lo + seg_hi)
        mode_a = _mode(mid + a, q)
        mode_b = _mode(mid + b, q)

        if mode_a == "zero" or mode_b == "zero":
            continue
        if mode_a == "one" and mode_b == "one":
            total += seg_hi - seg_lo
            continue
        if mode_a == "one" and mode_b == "linear":
            total += _primitive_g(seg_hi, b, q) - _primitive_g(seg_lo, b, q)
            continue
        if mode_a == "linear" and mode_b == "one":
            total += _primitive_g(seg_hi, a, q) - _primitive_g(seg_lo, a, q)
            continue
        if mode_a == "linear" and mode_b == "linear":
            total += _poly_integral(seg_hi, a, b, q) - _poly_integral(seg_lo, a, b, q)
            continue
        # Remaining combinations (one with zero) contribute nothing.

    return total


def win_prob_three_players(e_i: float, e_j: float, e_k: float, q: float) -> float:
    """Closed-form win probability for player i against j and k."""
    a = float(e_i - e_j)
    b = float(e_i - e_k)
    integral = _integrate_product(a, b, q, -q, q)
    p = integral / (2.0 * q)
    return float(_clip_scalar(p, 0.0, 1.0))


def win_prob_three_players_grad(e_i: float, e_j: float, e_k: float, q: float) -> tuple[float, float, float]:
    """Closed-form gradient of win probability w.r.t efforts (e_i, e_j, e_k)."""
    a = float(e_i - e_j)
    b = float(e_i - e_k)

    denom = (2.0 * q) ** 2

    la = max(-q, -q - a)
    ua = min(q, q - a)
    if ua <= la:
        dp_da = 0.0
    else:
        dp_da = (_primitive_g(ua, b, q) - _primitive_g(la, b, q)) / denom

    lb = max(-q, -q - b)
    ub = min(q, q - b)
    if ub <= lb:
        dp_db = 0.0
    else:
        dp_db = (_primitive_g(ub, a, q) - _primitive_g(lb, a, q)) / denom

    dp_de_i = dp_da + dp_db
    dp_de_j = -dp_da
    dp_de_k = -dp_db

    return float(dp_de_i), float(dp_de_j), float(dp_de_k)






















