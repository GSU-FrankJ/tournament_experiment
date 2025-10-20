"""
Shared payoff helpers used across environments.
"""

from __future__ import annotations


def expected_utility_two_player(w_h: float, w_l: float, p_win: float, k: float, e: float) -> float:
    """E[u] = w_L + p_win (w_H - w_L) - k e^2."""
    return float(w_l) + float(p_win) * (float(w_h) - float(w_l)) - float(k) * float(e) * float(e)























