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










