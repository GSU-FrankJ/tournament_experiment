"""Policy-noise quantities of the terminal stage at the tie (MS-R2, D4 and D6; reporting only).

The tie first-order condition of the game both players play when their actions carry the policy's own
noise is ``e_sigma(0) = DW / (2k) * E[f_xi(n_L - n_O)]`` with ``n`` the centred learned Beta noise at
``d = 0`` (``run.run_v2_T2_locked.smoothed_share``). With ``e*(0) = DW / (4 k q)`` and the triangular
density ``f_xi``, whose kink at 0 lowers the mean marginal benefit by ``E|n_L - n_O| / (4 q^2)``,

    smoothing part = e*(0) - e_sigma(0) = e*(0) * E|n_L - n_O| / (2 q)
                   = e*(0) * sigma / (sqrt(pi) * q)         for Gaussian noise (E|n_L - n_O| = 2 sigma / sqrt(pi)),
    remainder      = e_sigma(0) - e_hat(0),
    gap            = e*(0) - e_hat(0) = smoothing part + remainder.

Nothing here enters training, the sampler, the schedule or any rule: the closed form ``e*(0)`` is used
for the decomposition only.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from envs.curriculum_env import GameSpec
from utils.theory_multistage import f_xi

SMOOTH_NODES = 400          # run/run_v2_T2_locked.py:SMOOTH_NODES


def smoothed_tie_prediction(alpha: float, beta: float, dw: float, k: float, q: float, e_range: float,
                            nodes: int = SMOOTH_NODES) -> Tuple[float, float]:
    """Tie best response of the game played with the learned noise, and the node standard deviation.

    The computation of ``run.run_v2_T2_locked.smoothed_share``: ``nodes`` Beta quantile midpoints of the
    centred effort noise, ``e_sigma = DW / (2k) * mean f_xi(n_L - n_O)`` over all pairs of nodes.

    Args:
        alpha: Beta shape alpha of the policy at d = 0.
        beta: Beta shape beta of the policy at d = 0.
        dw: Prize spread w_H - w_L.
        k: Cost coefficient.
        q: Noise half-width.
        e_range: Effort range e_max - e_min (the Beta variable is scaled by it).
        nodes: Number of quantile midpoints per player.

    Returns:
        ``(e_sigma, sigma_nodes)``; ``sigma_nodes`` is the standard deviation of the node noise.
    """
    from scipy.stats import beta as beta_dist
    u = (np.arange(nodes) + 0.5) / nodes
    x = e_range * (beta_dist.ppf(u, alpha, beta) - alpha / (alpha + beta))
    e_sig = dw / (2.0 * k) * float(f_xi(x[:, None] - x[None, :], q).mean())
    return e_sig, float(x.std())


def beta_std_effort(alpha: float, beta: float, e_range: float) -> float:
    """Analytic standard deviation of the effort noise, ``e_range * sqrt(ab / ((a + b)^2 (a + b + 1)))``."""
    s = alpha + beta
    return float(e_range * math.sqrt(alpha * beta / (s * s * (s + 1.0))))


def beta_noise_moments(alpha: float, beta: float) -> Tuple[float, float]:
    """Skewness and excess kurtosis of Beta(alpha, beta) (a Gaussian has 0 and 0)."""
    s = alpha + beta
    skew = 2.0 * (beta - alpha) * math.sqrt(s + 1.0) / ((s + 2.0) * math.sqrt(alpha * beta))
    exk = 6.0 * ((alpha - beta) ** 2 * (s + 1.0) - alpha * beta * (s + 2.0)) / (alpha * beta * (s + 2.0) * (s + 3.0))
    return skew, exk


def smoothing_formula(g2_0: float, sigma: float, q: float) -> float:
    """Gaussian-noise smoothing part ``e*(0) * sigma / (sqrt(pi) * q)``."""
    return g2_0 * sigma / (math.sqrt(math.pi) * q)


def decompose_gap(g2_0: float, e_sigma: float, e_hat_0: float, sigma: float, q: float) -> Dict[str, float]:
    """Gap, smoothing part, remainder, the Gaussian formula and the ratio of the smoothing part to it."""
    formula = smoothing_formula(g2_0, sigma, q)
    smooth = g2_0 - e_sigma
    return {"gap": g2_0 - e_hat_0, "smoothing": smooth, "remainder": e_sigma - e_hat_0, "formula": formula,
            "ratio": smooth / formula if formula else float("nan")}


def tie_noise_report(mean_fn: Callable, beta_fn: Callable, spec: GameSpec, stage: int,
                     nodes: int = SMOOTH_NODES, g2_0: Optional[float] = None,
                     e_hat_0: Optional[float] = None) -> Dict[str, float]:
    """Reporting columns of one check: e_hat(0), sigma(0), e_sigma(0) of the candidate at ``d = 0``.

    Args:
        mean_fn: ``(stage, d) -> mean effort`` of the candidate.
        beta_fn: ``(stage, d) -> (alpha, beta)`` of the candidate (includes the concentration scale).
        spec: Game specification (T = 2, ``stage`` = the terminal stage).
        stage: The stage whose tie is reported.
        nodes: Quantile midpoints per player.
        g2_0: Closed-form tie effort e*(0) (default ``DW / (4 k q)``; the runner passes the verifier's value).
        e_hat_0: Learned tie effort (default: ``mean_fn`` on the single point d = 0; the runner passes the verifier's
            ``e2_at_0`` so that the gap equals the closed-form pipeline's, whose batch evaluation differs from the
            single-point one by float32 rounding, about 2e-6 effort units).

    Returns:
        ``e_hat_0``, ``sigma_0`` (analytic Beta standard deviation in effort units), ``e_sigma_0``,
        ``smoothing`` and ``remainder`` (both in effort units; ``g2_0`` is the closed form at d = 0).
    """
    a, b = beta_fn(stage, np.zeros(1))
    a0, b0 = float(a[0]), float(b[0])
    e_hat = float(mean_fn(stage, np.zeros(1))[0]) if e_hat_0 is None else float(e_hat_0)
    e_sig, _ = smoothed_tie_prediction(a0, b0, spec.dw, spec.k, spec.q, spec.e_range, nodes)
    g2 = spec.dw / (4.0 * spec.k * spec.q) if g2_0 is None else float(g2_0)
    out = decompose_gap(g2, e_sig, e_hat, beta_std_effort(a0, b0, spec.e_range), spec.q)
    return {"e_hat_0": e_hat, "sigma_0": beta_std_effort(a0, b0, spec.e_range), "e_sigma_0": e_sig,
            "g2_0": g2, "smoothing": out["smoothing"], "remainder": out["remainder"], "gap": out["gap"]}


def tie_residual_R0(res: Any, stage: int) -> float:
    """``R0 = r_2(0) / s_2`` of a verifier result: ``|e_hat - a_dev| / a_dev`` at the node d = 0 of ``stage``."""
    sr = res.stages[stage]
    d = np.asarray(sr.d_grid, dtype=float)
    j0 = int(np.argmin(np.abs(d)))
    a = float(np.asarray(sr.a_dev, dtype=float)[j0])
    return float(abs(float(np.asarray(sr.e_hat, dtype=float)[j0]) - a) / a) if a > 0.0 else float("nan")
