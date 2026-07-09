"""Closed-form two-stage benchmark and validity region for the multi-stage tournament.

Implements the T=2 dynamic Lazear-Rosen benchmark from
``docs/Experiments Plan_Multi-stage.md`` (section 2.5), converted to the REPO
cost convention

    c(e) = k * e**2          (c' = 2ke, c'' = 2k)

The plan document derives the benchmark under c(e) = (k/2) e**2; all formulas
here have the documented "replace k by 2k" conversion already applied.
Cross-checked numerically in ``tools/verify_two_stage_benchmark.py``.

Model (two players, T=2, terminal reward only):
    y_it = e_it + eps_it,  eps ~ U(-q, q) i.i.d.
    d_{t+1} = d_t + e_i - e_j + xi_t,  xi = eps_i - eps_j ~ Triangular(-2q, 2q)
    Terminal: winner (final gap > 0) gets w_h, loser w_l; per-stage cost k e^2.

Benchmark (interior, valid only for q > q_crit):
    g2(d) = DW * f_xi(d) / (2k)        (stage-2 effort function; even in d)
    g1    = DW / (6 k q)               (stage-1 effort at d1 = 0)
    E[g2(d2)] on-path equals g1 exactly.

Validity region q_crit = max(q_soc, q_bound2, q_bound1, q_pc):
    q_soc    = sqrt(DW / (8k))   -- global concavity of the stage objectives.
               NOTE: contrary to the plan doc, the stage-1 SOC is NOT
               unconditional. V2*(d) has a convex kink at d=0 (from the
               -k*g2(d)^2 term), so E[V2*''(xi)] = +DW^2/(32 k q^4) > 0 and
               stage-1 curvature is -2k + DW^2/(32 k q^4), which is negative
               exactly when q > q_soc. Numerically the constraint is tight:
               at q just below q_soc the symmetric candidate becomes a local
               minimum and global deviations (give-up) are profitable.
    q_bound2 = DW / (4 k e_bar)  -- stage-2 peak effort g2(0) <= e_bar.
    q_bound1 = DW / (6 k e_bar)  -- stage-1 effort g1 <= e_bar.
    q_pc     = participation vs outside option u_bar:
               U_eq = (w_h + w_l)/2 - 17 DW^2 / (288 k q^2) >= u_bar.

All of the above are analytic necessary/sufficient screens; use
``validate_two_stage_params`` for the full check including a numerical
global-deviation scan (covers deviate-to-zero-effort explicitly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Shock-difference distribution: xi = eps_i - eps_j, Triangular on [-2q, 2q]
# ---------------------------------------------------------------------------

def f_xi(x: np.ndarray, q: float) -> np.ndarray:
    """Density of the stage shock difference xi ~ Triangular(-2q, 2q).

    Args:
        x: Evaluation points (array or scalar).
        q: Half-width of the per-player uniform noise U(-q, q).

    Returns:
        Density values, zero outside [-2q, 2q].
    """
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) <= 2.0 * q, (2.0 * q - np.abs(x)) / (4.0 * q * q), 0.0)


def F_xi(x: np.ndarray, q: float) -> np.ndarray:
    """CDF of xi ~ Triangular(-2q, 2q); use for closed-form terminal integration.

    The terminal winning probability from gap d (both players following the
    even benchmark, whose stage-2 efforts cancel in the gap) is exactly
    ``F_xi(d, q)``. Verifiers must use this closed form at the terminal step
    instead of interpolating the step-function reward R(d).

    Args:
        x: Evaluation points (array or scalar).
        q: Half-width of the per-player uniform noise U(-q, q).

    Returns:
        CDF values in [0, 1].
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x)
    lo = x <= -2.0 * q
    hi = x >= 2.0 * q
    mid_neg = (~lo) & (x < 0.0)
    mid_pos = (~hi) & (x >= 0.0)
    out[lo] = 0.0
    out[hi] = 1.0
    out[mid_neg] = (x[mid_neg] + 2.0 * q) ** 2 / (8.0 * q * q)
    out[mid_pos] = 1.0 - (2.0 * q - x[mid_pos]) ** 2 / (8.0 * q * q)
    return out


# ---------------------------------------------------------------------------
# Closed-form two-stage benchmark (repo convention c(e) = k e^2)
# ---------------------------------------------------------------------------

def g2_two_stage(
    d: np.ndarray, q: float, w_h: float, w_l: float, k: float, e_bar: float = 100.0
) -> np.ndarray:
    """Stage-2 (final-stage) benchmark effort function, clipped to [0, e_bar].

    g2(d) = DW * f_xi(d) / (2k): triangular hump, peak DW/(4kq) at d=0,
    linearly decreasing to zero at |d| = 2q. Even in d: at the final stage
    leader and follower exert IDENTICAL effort (their FOCs coincide because
    f_xi is even) - do not interpret this as a training failure.

    Args:
        d: Score gap(s) at the start of stage 2.
        q: Noise half-width.
        w_h: Winner prize.
        w_l: Loser prize.
        k: Cost coefficient in c(e) = k e^2.
        e_bar: Effort upper bound.

    Returns:
        Benchmark efforts (same shape as ``d``).
    """
    dw = float(w_h) - float(w_l)
    return np.clip(dw * f_xi(d, q) / (2.0 * k), 0.0, e_bar)


def g1_two_stage(q: float, w_h: float, w_l: float, k: float) -> float:
    """Stage-1 benchmark effort at initial gap d1 = 0: g1 = DW / (6 k q).

    Also equals the on-path expected stage-2 effort E[g2(xi_1)] exactly.

    Args:
        q: Noise half-width.
        w_h: Winner prize.
        w_l: Loser prize.
        k: Cost coefficient in c(e) = k e^2.

    Returns:
        Benchmark stage-1 effort.
    """
    return (float(w_h) - float(w_l)) / (6.0 * k * q)


def v2_star(
    d: np.ndarray, q: float, w_h: float, w_l: float, k: float, e_bar: float = 100.0
) -> np.ndarray:
    """Stage-2 equilibrium continuation value V2*(d) = w_l + DW F_xi(d) - k g2(d)^2.

    Args:
        d: Score gap(s) at the start of stage 2.
        q: Noise half-width.
        w_h: Winner prize.
        w_l: Loser prize.
        k: Cost coefficient in c(e) = k e^2.
        e_bar: Effort upper bound (for the clipped g2).

    Returns:
        Continuation values (same shape as ``d``).
    """
    dw = float(w_h) - float(w_l)
    return float(w_l) + dw * F_xi(d, q) - k * g2_two_stage(d, q, w_h, w_l, k, e_bar) ** 2


def eq_utility_two_stage(q: float, w_h: float, w_l: float, k: float) -> float:
    """Ex-ante equilibrium utility: (w_h + w_l)/2 - 17 DW^2 / (288 k q^2).

    Args:
        q: Noise half-width.
        w_h: Winner prize.
        w_l: Loser prize.
        k: Cost coefficient in c(e) = k e^2.

    Returns:
        Expected equilibrium payoff per player.
    """
    dw = float(w_h) - float(w_l)
    return (float(w_h) + float(w_l)) / 2.0 - 17.0 * dw * dw / (288.0 * k * q * q)


def stage1_curvature(q: float, w_h: float, w_l: float, k: float) -> float:
    """Stage-1 objective curvature at the symmetric candidate: -2k + DW^2/(32 k q^4).

    CORRECTED versus the plan document: the kink of V2* at d=0 contributes
    +DW^2/(16 k q^4) to E[V2*''(xi)] (a Dirac term the doc's smooth
    differentiation misses), flipping the sign of the correction term.
    Negative (SOC holds) exactly when q > q_soc.

    Args:
        q: Noise half-width.
        w_h: Winner prize.
        w_l: Loser prize.
        k: Cost coefficient in c(e) = k e^2.

    Returns:
        Second derivative of the stage-1 objective at e = g1.
    """
    dw = float(w_h) - float(w_l)
    return -2.0 * k + dw * dw / (32.0 * k * q ** 4)


# ---------------------------------------------------------------------------
# Validity region
# ---------------------------------------------------------------------------

def q_soc(w_h: float, w_l: float, k: float) -> float:
    """SOC / global-concavity threshold: q_soc = sqrt(DW / (8k)).

    Sufficient for stage-2 global concavity and (empirically tight) for the
    stage-1 SOC via ``stage1_curvature``.

    Args:
        w_h: Winner prize.
        w_l: Loser prize.
        k: Cost coefficient in c(e) = k e^2.

    Returns:
        Threshold noise level.
    """
    return float(np.sqrt((float(w_h) - float(w_l)) / (8.0 * k)))


def q_bound_stage2(w_h: float, w_l: float, k: float, e_bar: float = 100.0) -> float:
    """Effort-bound threshold for stage 2: g2(0) = DW/(4kq) <= e_bar.

    Args:
        w_h: Winner prize.
        w_l: Loser prize.
        k: Cost coefficient in c(e) = k e^2.
        e_bar: Effort upper bound.

    Returns:
        Threshold noise level DW / (4 k e_bar).
    """
    return (float(w_h) - float(w_l)) / (4.0 * k * e_bar)


def q_bound_stage1(w_h: float, w_l: float, k: float, e_bar: float = 100.0) -> float:
    """Effort-bound threshold for stage 1: g1 = DW/(6kq) <= e_bar.

    Args:
        w_h: Winner prize.
        w_l: Loser prize.
        k: Cost coefficient in c(e) = k e^2.
        e_bar: Effort upper bound.

    Returns:
        Threshold noise level DW / (6 k e_bar).
    """
    return (float(w_h) - float(w_l)) / (6.0 * k * e_bar)


def q_pc(w_h: float, w_l: float, k: float, u_bar: float = 0.0) -> float:
    """Participation threshold vs outside option u_bar.

    U_eq >= u_bar  <=>  q >= DW * sqrt(17 / (288 k ((w_h + w_l)/2 - u_bar))).

    Args:
        w_h: Winner prize.
        w_l: Loser prize.
        k: Cost coefficient in c(e) = k e^2.
        u_bar: Outside option (0 when players cannot exit).

    Returns:
        Threshold noise level (inf if the prize mean does not cover u_bar).
    """
    dw = float(w_h) - float(w_l)
    slack = (float(w_h) + float(w_l)) / 2.0 - float(u_bar)
    if slack <= 0.0:
        return float("inf")
    return dw * float(np.sqrt(17.0 / (288.0 * k * slack)))


def q_crit(w_h: float, w_l: float, k: float, e_bar: float = 100.0, u_bar: float = 0.0) -> float:
    """Effective validity threshold: max of SOC, both effort bounds, and participation.

    The unprojected closed-form benchmark is valid only for q > q_crit.

    Args:
        w_h: Winner prize.
        w_l: Loser prize.
        k: Cost coefficient in c(e) = k e^2.
        e_bar: Effort upper bound.
        u_bar: Outside option.

    Returns:
        q_crit threshold.
    """
    return max(
        q_soc(w_h, w_l, k),
        q_bound_stage2(w_h, w_l, k, e_bar),
        q_bound_stage1(w_h, w_l, k, e_bar),
        q_pc(w_h, w_l, k, u_bar),
    )


# ---------------------------------------------------------------------------
# Full validation (analytic screens + numerical global-deviation scan)
# ---------------------------------------------------------------------------

@dataclass
class TwoStageValidation:
    """Result of ``validate_two_stage_params``."""

    ok: bool
    q: float
    q_crit: float
    thresholds: Dict[str, float]
    g1: float
    g2_at_0: float
    eq_utility: float
    stage1_curvature: float
    max_stage1_deviation_gain: float
    max_stage2_deviation_gain: float
    zero_effort_deviation_gain: float
    messages: List[str] = field(default_factory=list)


def validate_two_stage_params(
    q: float,
    w_h: float,
    w_l: float,
    k: float,
    e_bar: float = 100.0,
    u_bar: float = 0.0,
    dev_tol: float = 1e-4,
    n_e_grid: int = 4001,
    n_xi_grid: int = 8001,
    n_d_grid: int = 81,
) -> TwoStageValidation:
    """Validate a two-stage parameter set before any training run.

    Runs the analytic q_crit screens AND a numerical global-deviation scan:
    for stage 2, checks that g2(d) is the global best response on a d-grid;
    for stage 1, scans the full effort grid (this covers the deviate-to-zero
    global deviation explicitly, which local SOC alone does not rule out).

    Args:
        q: Noise half-width to validate.
        w_h: Winner prize.
        w_l: Loser prize.
        k: Cost coefficient in c(e) = k e^2.
        e_bar: Effort upper bound.
        u_bar: Outside option for the participation constraint.
        dev_tol: Max tolerated profitable-deviation gain (payoff units).
        n_e_grid: Effort-grid resolution for the deviation scans.
        n_xi_grid: Quadrature-grid resolution over the triangular shock.
        n_d_grid: Score-gap grid resolution for the stage-2 scan.

    Returns:
        A ``TwoStageValidation`` report; ``ok`` is True when all analytic
        screens pass and no deviation gain exceeds ``dev_tol``.
    """
    msgs: List[str] = []
    thresholds = {
        "q_soc": q_soc(w_h, w_l, k),
        "q_bound_stage2": q_bound_stage2(w_h, w_l, k, e_bar),
        "q_bound_stage1": q_bound_stage1(w_h, w_l, k, e_bar),
        "q_pc": q_pc(w_h, w_l, k, u_bar),
    }
    qc = max(thresholds.values())
    for name, thr in thresholds.items():
        if q <= thr:
            msgs.append(f"FAIL {name}: q={q:g} <= {thr:.3f}")

    g1 = g1_two_stage(q, w_h, w_l, k)
    curv = stage1_curvature(q, w_h, w_l, k)
    if curv >= 0.0:
        msgs.append(f"FAIL stage-1 SOC: curvature {curv:+.3e} >= 0 (kink-corrected)")

    # Quadrature over xi ~ Triangular(-2q, 2q)
    xg = np.linspace(-2.0 * q, 2.0 * q, n_xi_grid)
    w = f_xi(xg, q)
    w = w / w.sum()
    e_grid = np.linspace(0.0, e_bar, n_e_grid)

    # Stage-2 global-deviation scan (opponent at g2(d); g2 even in d)
    dev2 = 0.0
    for d in np.linspace(-2.0 * q, 2.0 * q, n_d_grid):
        e_opp = float(g2_two_stage(np.asarray(d), q, w_h, w_l, k, e_bar))
        u_dev = (w_h - w_l) * F_xi(d + e_grid - e_opp, q) - k * e_grid ** 2
        u_eq2 = float((w_h - w_l) * F_xi(np.asarray([d]), q)[0]) - k * e_opp ** 2
        dev2 = max(dev2, float(u_dev.max()) - u_eq2)
    if dev2 > dev_tol:
        msgs.append(f"FAIL stage-2 deviation scan: max gain {dev2:.2e} > {dev_tol:g}")

    # Stage-1 global-deviation scan (opponent at g1, equilibrium continuation)
    u1 = np.array(
        [
            -k * e ** 2 + float((v2_star(e - g1 + xg, q, w_h, w_l, k, e_bar) * w).sum())
            for e in e_grid
        ]
    )
    u_eq1 = -k * g1 ** 2 + float((v2_star(xg, q, w_h, w_l, k, e_bar) * w).sum())
    dev1 = float(u1.max()) - u_eq1
    zero_dev = float(u1[0]) - u_eq1
    if dev1 > dev_tol:
        msgs.append(f"FAIL stage-1 deviation scan: max gain {dev1:.2e} > {dev_tol:g}")

    return TwoStageValidation(
        ok=not msgs,
        q=q,
        q_crit=qc,
        thresholds=thresholds,
        g1=g1,
        g2_at_0=float(g2_two_stage(np.asarray(0.0), q, w_h, w_l, k, e_bar)),
        eq_utility=eq_utility_two_stage(q, w_h, w_l, k),
        stage1_curvature=curv,
        max_stage1_deviation_gain=dev1,
        max_stage2_deviation_gain=dev2,
        zero_effort_deviation_gain=zero_dev,
        messages=msgs,
    )
