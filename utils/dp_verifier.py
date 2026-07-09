"""Independent dynamic-programming best-response verifier for the multi-stage game.

Certifies whether a candidate symmetric effort function e_hat_t(d) is an
approximate Markov perfect equilibrium, INDEPENDENTLY of how it was produced.
This module imports only the closed-form shock distribution from
``utils.theory_multistage``; it never imports the training env or the PPO
agent, so its verdict is a genuine external check (blind-spot risks H6/M7).

Implements ``docs/Experiments Plan_Multi-stage.md`` section 4 with the
owner-mandated numerics (2026-07-09):

  - Backward-induction best response with the opponent fixed at e_hat_t(-d).
  - CLOSED-FORM terminal integration: E_xi[R(y + xi)] = w_l + DW * F_xi(y),
    never interpolation of the step reward R near y = 0.
  - Deterministic quadrature over the triangular xi for the t < T
    continuations, evaluated through a 1-D "smoothed value"
    W_{t+1}(y) = E_xi[V_{t+1}(y + xi)] (avoids an O(M*K*quad) tensor).
  - Δ_t(d) one-step deviation gaps as the PRIMARY certificate: by the
    performance-difference lemma, EXP <= sum_t max_d Δ_t(d), a true upper
    bound on exploitability. Root-state EXP and the on-path average
    E_{d~e_hat}[Δ_t(d)] are reported alongside.

Cost convention: repo standard c(e) = k e^2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from utils.theory_multistage import F_xi, f_xi

# A policy is a vectorized callable (stage t, gaps d[array]) -> efforts[array].
Policy = Callable[[int, np.ndarray], np.ndarray]


@dataclass
class VerifierResult:
    """Output of :func:`verify`.

    Three deviation-gap aggregates are reported (all Δ_t(d) >= 0):

      - ``delta_sum_reachable`` (PRIMARY certificate): Σ_t max over the
        BR-reachable support of Δ_t(d). By the performance-difference lemma
        this upper-bounds root exploitability, and — unlike the full-grid
        version — it excludes states that cannot be reached from d_1 = 0
        (e.g. all d != 0 at stage 1). Robust to the DP best response being
        an approximation, so it cannot silently over-certify.
      - ``delta_sum_full``: Σ_t max over the WHOLE grid of Δ_t(d). The
        plan's conservative worst-case; meaningful for a policy defined at
        every state, but intentionally over-states for an on-path-only
        benchmark (its stage-1 term maxes over unreachable gaps).
      - ``delta_onpath_sum``: Σ_t E_{d~mu^e}[Δ_t(d)]. Local on-path quality.

    Attributes:
        exp: Root-state exploitability V_1^BR(0) - V_1^e(0).
        exp_over_dw: ``exp`` normalized by the prize spread ΔW.
        delta_sum_reachable: BR-reachable-support Δ-sum (primary certificate).
        delta_sum_full: Full-grid worst-case Δ-sum (robustness diagnostic).
        delta_onpath_sum: On-path Δ-sum.
        reachable_delta_by_stage: max over BR-reachable support of Δ_t.
        worst_delta_by_stage: max over the full grid of Δ_t.
        onpath_delta_by_stage: E_{d~mu^e}[Δ_t].
        v_br_root: V_1^BR(0).
        v_e_root: V_1^e(0).
        d_grid: Score-gap grid used.
        delta_by_stage: Δ_t(d) arrays on the grid, keyed by stage.
        v_e_by_stage: V_t^e(d) arrays on the grid, keyed by stage.
        onpath_dist_by_stage: on-path state pmf mu^e_t(d), keyed by stage.
        br_reach_dist_by_stage: BR-reachable state pmf mu^BR_t(d), keyed by stage.
        certified: Whether delta_sum_reachable / ΔW <= ``epsilon_over_dw``.
    """

    exp: float
    exp_over_dw: float
    delta_sum_reachable: float
    delta_sum_full: float
    delta_onpath_sum: float
    reachable_delta_by_stage: Dict[int, float]
    worst_delta_by_stage: Dict[int, float]
    onpath_delta_by_stage: Dict[int, float]
    v_br_root: float
    v_e_root: float
    d_grid: np.ndarray
    delta_by_stage: Dict[int, np.ndarray] = field(default_factory=dict)
    v_e_by_stage: Dict[int, np.ndarray] = field(default_factory=dict)
    onpath_dist_by_stage: Dict[int, np.ndarray] = field(default_factory=dict)
    br_reach_dist_by_stage: Dict[int, np.ndarray] = field(default_factory=dict)
    certified: Optional[bool] = None


def _quadrature_nodes(q: float, n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic quadrature nodes/weights for xi ~ Triangular(-2q, 2q).

    Trapezoidal nodes on [-2q, 2q] weighted by the triangular density and
    renormalized to sum to one.

    Args:
        q: Noise half-width.
        n_quad: Number of nodes (odd recommended so 0 is a node).

    Returns:
        ``(nodes, weights)`` with ``weights`` summing to 1.
    """
    nodes = np.linspace(-2.0 * q, 2.0 * q, n_quad)
    dens = f_xi(nodes, q)
    trap = np.ones(n_quad)
    trap[0] = trap[-1] = 0.5
    w = dens * trap
    w = w / w.sum()
    return nodes, w


def _smoothed_continuation(
    landing: np.ndarray,
    d_grid: np.ndarray,
    v_next: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """E_xi[V_next(landing + xi)] via quadrature + linear interp (const tails).

    Args:
        landing: Pre-shock landing points (any shape).
        d_grid: Score-gap grid on which ``v_next`` is defined.
        v_next: Continuation values on ``d_grid``.
        nodes: Quadrature nodes over xi.
        weights: Quadrature weights (sum to 1).

    Returns:
        Expected continuation values, same shape as ``landing``.
    """
    acc = np.zeros_like(landing, dtype=float)
    for s, w in zip(nodes, weights):
        # np.interp clamps out-of-range to edge values -> constant tail extrapolation
        acc += w * np.interp(landing + s, d_grid, v_next)
    return acc


def _parabolic_polish(
    e_grid: np.ndarray, q_vals: np.ndarray, argmax_idx: np.ndarray
) -> np.ndarray:
    """Refine a grid argmax with a 3-point parabolic vertex, clamped to bounds.

    Args:
        e_grid: Effort grid (1-D, uniform).
        q_vals: Objective values, shape ``(M, K)`` over (state, effort).
        argmax_idx: Per-state index of the grid maximum, shape ``(M,)``.

    Returns:
        Polished maximum value per state, shape ``(M,)``.
    """
    k = e_grid.size
    idx = np.clip(argmax_idx, 1, k - 2)
    rows = np.arange(q_vals.shape[0])
    y0 = q_vals[rows, idx - 1]
    y1 = q_vals[rows, idx]
    y2 = q_vals[rows, idx + 1]
    denom = y0 - 2.0 * y1 + y2
    # vertex offset in units of grid step; 0 where denom ~ 0 (flat)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = 0.5 * (y0 - y2) / denom
    delta = np.where(np.abs(denom) < 1e-12, 0.0, delta)
    delta = np.clip(delta, -1.0, 1.0)
    v_polished = y1 - 0.25 * (y0 - y2) * delta
    # never below the raw grid max (guards against non-concave triples)
    grid_max = q_vals[rows, argmax_idx]
    return np.maximum(v_polished, grid_max)


def _forward_distribution(
    d_grid: np.ndarray,
    drift_by_stage: Dict[int, np.ndarray],
    T: int,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Forward-propagate a state pmf p_t(d) from a unit mass at d = 0.

    Pushes mass through d' = d + drift_t(d) + xi using the quadrature kernel,
    splitting each landing linearly between adjacent grid cells.

    Args:
        d_grid: Score-gap grid.
        drift_by_stage: Per-stage deterministic drift array on ``d_grid``
            (player-i effort minus opponent effort), for t = 1..T-1.
        T: Horizon.
        nodes: Quadrature nodes over xi.
        weights: Quadrature weights.

    Returns:
        Dict stage -> pmf array on ``d_grid`` (sums to 1) for t = 1..T.
    """
    m = d_grid.size
    p = np.zeros(m)
    p[np.argmin(np.abs(d_grid))] = 1.0  # unit mass at d=0
    dist = {1: p.copy()}
    for t in range(1, T):
        mean = d_grid + drift_by_stage[t]  # pre-shock landing per source
        nxt = np.zeros(m)
        for s, w in zip(nodes, weights):
            land = mean + s
            idx = np.clip(np.searchsorted(d_grid, land), 1, m - 1)
            left = d_grid[idx - 1]
            right = d_grid[idx]
            frac = np.clip((land - left) / (right - left), 0.0, 1.0)
            np.add.at(nxt, idx - 1, w * p * (1.0 - frac))
            np.add.at(nxt, idx, w * p * frac)
        nxt = nxt / nxt.sum()
        dist[t + 1] = nxt
        p = nxt
    return dist


def verify(
    policy: Policy,
    *,
    w_h: float,
    w_l: float,
    k: float,
    q: float,
    T: int,
    e_bar: float = 100.0,
    d_grid_size: int = 201,
    e_grid_size: int = 401,
    n_quad: int = 129,
    d_max: Optional[float] = None,
    d_max_margin: float = 50.0,
    epsilon_over_dw: Optional[float] = None,
    polish: bool = True,
) -> VerifierResult:
    """Certify a candidate effort function by backward-induction best response.

    Args:
        policy: Vectorized symmetric effort function ``e_hat(t, d_array)``.
        w_h: Winner prize.
        w_l: Loser prize.
        k: Cost coefficient in c(e) = k e^2.
        q: Noise half-width.
        T: Horizon (stages).
        e_bar: Effort upper bound.
        d_grid_size: Score-gap grid points (forced odd so 0 is a node).
        e_grid_size: Effort grid points for the BR max.
        n_quad: Quadrature nodes over the triangular shock.
        d_max: Score-gap grid half-width; default T*(e_bar + 2q) + margin.
        d_max_margin: Margin added to the default ``d_max``.
        epsilon_over_dw: If given, certify when delta_sum_reachable/ΔW <= this.
        polish: Parabolic refinement of the BR effort argmax.

    Returns:
        A :class:`VerifierResult`.
    """
    dw = float(w_h) - float(w_l)
    if d_grid_size % 2 == 0:
        d_grid_size += 1  # ensure 0 is a node
    if d_max is None:
        d_max = T * (e_bar + 2.0 * q) + d_max_margin
    d_grid = np.linspace(-d_max, d_max, d_grid_size)
    e_grid = np.linspace(0.0, e_bar, e_grid_size)
    nodes, weights = _quadrature_nodes(q, n_quad)
    cost_e = k * e_grid ** 2  # (K,)

    def terminal_W(y: np.ndarray) -> np.ndarray:
        """Closed-form E_xi[R(y+xi)] = w_l + DW F_xi(y). Exact, no interp."""
        return w_l + dw * F_xi(y, q)

    v_br: Optional[np.ndarray] = None
    v_e: Optional[np.ndarray] = None
    delta_by_stage: Dict[int, np.ndarray] = {}
    v_e_by_stage: Dict[int, np.ndarray] = {}
    br_effort_by_stage: Dict[int, np.ndarray] = {}
    rows = np.arange(d_grid.size)

    for t in range(T, 0, -1):
        own_e = policy(t, d_grid)      # e_hat_t(d)   (M,)
        opp_e = policy(t, -d_grid)     # e_hat_t(-d)  (M,)

        # W_br(y), W_e(y): expected continuation given the two recursions.
        if t == T:
            def W_br(y):
                return terminal_W(y)
            W_e = W_br
        else:
            v_br_next, v_e_next = v_br, v_e

            def W_br(y, _v=v_br_next):
                return _smoothed_continuation(y, d_grid, _v, nodes, weights)

            def W_e(y, _v=v_e_next):
                return _smoothed_continuation(y, d_grid, _v, nodes, weights)

        # Best response: max over effort of -c(e) + W_br(d + e - opp_e).
        landing_br = d_grid[:, None] + e_grid[None, :] - opp_e[:, None]  # (M,K)
        q_br = -cost_e[None, :] + W_br(landing_br)                       # (M,K)
        am = np.argmax(q_br, axis=1)
        br_effort_by_stage[t] = e_grid[am]                              # (M,) grid argmax effort
        v_br = _parabolic_polish(e_grid, q_br, am) if polish else q_br[rows, am]

        # Learned value: both play e_hat.
        landing_e = d_grid + own_e - opp_e
        v_e = -k * own_e ** 2 + W_e(landing_e)
        v_e_by_stage[t] = v_e.copy()

        # One-step deviation gap against the LEARNED continuation W_e.
        q_dev = -cost_e[None, :] + W_e(landing_br)                       # (M,K)
        am_dev = np.argmax(q_dev, axis=1)
        max_q_dev = (
            _parabolic_polish(e_grid, q_dev, am_dev)
            if polish
            else q_dev[rows, am_dev]
        )
        delta_by_stage[t] = np.maximum(max_q_dev - v_e, 0.0)

    root = int(np.argmin(np.abs(d_grid)))
    exp = float(v_br[root] - v_e[root])

    # On-path distribution mu^e (both players e_hat) and BR-reachable mu^BR
    # (player i best-responds, opponent plays e_hat). Support of mu^BR is the
    # set of states the deviator can reach from d_1 = 0 with positive prob.
    onpath_drift = {t: policy(t, d_grid) - policy(t, -d_grid) for t in range(1, T)}
    br_drift = {t: br_effort_by_stage[t] - policy(t, -d_grid) for t in range(1, T)}
    onpath = _forward_distribution(d_grid, onpath_drift, T, nodes, weights)
    br_reach = _forward_distribution(d_grid, br_drift, T, nodes, weights)

    tol = 1e-9
    worst = {t: float(delta_by_stage[t].max()) for t in delta_by_stage}
    reach = {
        t: float(delta_by_stage[t][br_reach[t] > tol].max())
        if np.any(br_reach[t] > tol)
        else 0.0
        for t in delta_by_stage
    }
    onp = {t: float((onpath[t] * delta_by_stage[t]).sum()) for t in delta_by_stage}

    delta_sum_reachable = float(sum(reach.values()))
    delta_sum_full = float(sum(worst.values()))
    certified = None
    if epsilon_over_dw is not None:
        certified = bool(delta_sum_reachable / dw <= epsilon_over_dw)

    return VerifierResult(
        exp=exp,
        exp_over_dw=exp / dw,
        delta_sum_reachable=delta_sum_reachable,
        delta_sum_full=delta_sum_full,
        delta_onpath_sum=float(sum(onp.values())),
        reachable_delta_by_stage=reach,
        worst_delta_by_stage=worst,
        onpath_delta_by_stage=onp,
        v_br_root=float(v_br[root]),
        v_e_root=float(v_e[root]),
        d_grid=d_grid,
        delta_by_stage=delta_by_stage,
        v_e_by_stage=v_e_by_stage,
        onpath_dist_by_stage=onpath,
        br_reach_dist_by_stage=br_reach,
        certified=certified,
    )


def verify_grid_refinement(
    policy: Policy,
    *,
    w_h: float,
    w_l: float,
    k: float,
    q: float,
    T: int,
    e_bar: float = 100.0,
    d_grid_sizes: List[int] = (51, 101, 201),
    **kwargs,
) -> Dict[str, object]:
    """Run ``verify`` across score-gap grids and Richardson-extrapolate EXP.

    Args:
        policy: Vectorized effort function.
        w_h: Winner prize.
        w_l: Loser prize.
        k: Cost coefficient.
        q: Noise half-width.
        T: Horizon.
        e_bar: Effort upper bound.
        d_grid_sizes: Ascending grid sizes to run.
        **kwargs: Forwarded to ``verify``.

    Returns:
        Dict with per-grid ``exp``/``delta_sum_reachable`` lists and an h^2
        Richardson extrapolate of EXP from the two finest grids.
    """
    exps: List[float] = []
    dsu: List[float] = []
    results: List[VerifierResult] = []
    for m in d_grid_sizes:
        r = verify(policy, w_h=w_h, w_l=w_l, k=k, q=q, T=T, e_bar=e_bar,
                   d_grid_size=m, **kwargs)
        results.append(r)
        exps.append(r.exp)
        dsu.append(r.delta_sum_reachable)

    richardson = exps[-1]
    if len(exps) >= 2:
        # linear-interp error ~ O(h^2); h halves as grid ~ doubles => ratio ~4
        ratio = ((d_grid_sizes[-1] - 1) / (d_grid_sizes[-2] - 1)) ** 2
        richardson = exps[-1] + (exps[-1] - exps[-2]) / (ratio - 1.0)

    return {
        "d_grid_sizes": list(d_grid_sizes),
        "exp": exps,
        "delta_sum_reachable": dsu,
        "exp_richardson": float(richardson),
        "results": results,
    }
