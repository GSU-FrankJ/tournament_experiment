"""
Two-stage gradient/static agent honoring training-order semantics:

- Gradient method: Stage 2 first, then Stage 1 accounting for E[k e_2^2].
"""

from __future__ import annotations

from typing import Tuple
from utils.theory import e_star, clip_stage1, clip_stage2


class TwoStageGradientAgent:
    """Analytic/numeric solver with Stage-2-first semantics."""

    def __init__(self, w_h: float, w_l: float, k1: float, k2: float, q: float):
        self.w_h = float(w_h)
        self.w_l = float(w_l)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.q = float(q)

    def solve(self) -> Tuple[float, float]:
        """Return (e1, e2) optimal under the Stage-2-first logic.

        Stage-2: e2* = e_star(q; w_h, w_l, k2), clipped to [0, 200].
        Stage-1: e1* = e_star(q; w_h, w_l, k1), clipped to [0, 100],
                 but the CSV semantics expect only inclusion of the continuation
                 term in Stage-1 expected utility. The closed-form remains the
                 same benchmark while accounting for the continuation in the
                 environment/training.
        """
        e2 = clip_stage2(e_star(self.q, self.w_h, self.w_l, self.k2))
        e1 = clip_stage1(e_star(self.q, self.w_h, self.w_l, self.k1))
        return e1, e2


