"""
Two-player single-stage gradient/static agent.

Implements the closed-form symmetric benchmark e*(q) and exposes a simple
solve() API used by run scripts.
"""

from __future__ import annotations

from typing import Tuple
from utils.theory import e_star


class TwoPlayersGradientAgent:
    """Closed-form two-player gradient/static solver."""

    def __init__(self, w_h: float, w_l: float, k: float, q: float, bounds: Tuple[float, float] = (0.0, 200.0)):
        self.w_h = float(w_h)
        self.w_l = float(w_l)
        self.k = float(k)
        self.q = float(q)
        self.bounds = bounds

    def solve(self) -> float:
        """Return closed-form symmetric effort clipped to bounds."""
        raw = e_star(self.q, self.w_h, self.w_l, self.k)
        lo, hi = self.bounds
        return max(lo, min(hi, float(raw)))


