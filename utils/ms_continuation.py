"""Nested expected-continuation tables for a frozen suffix of any horizon T (MS-R1, D2).

Generalises ``utils.v2_continuation`` (which integrates the FINAL stage only) to a suffix of frozen
stages t+1, ..., T. The table used by the phase of stage ``t`` is

    V~_{t+1}(y) = E_z[ g_{t+1}(y + z) ],   z = eps_L - eps_O,  density (2q - |z|) / (4 q^2) on [-2q, 2q]

with, for the frozen Beta means e_hat_s (the ``mean`` float path of ``v2_continuation``),

    g_T(d)       = w_l + DW F_xi(d + e_hat_T(d) - e_hat_T(-d)) - k e_hat_T(d)^2          (s = T)
    g_s(d)       = -k e_hat_s(d)^2 + V~_{s+1}(d + e_hat_s(d) - e_hat_s(-d))              (s < T)

where ``V~_{s+1}`` in the second line is the table of stage ``s`` (linear lookup), built before. The
argument ``y`` of ``V~_{t+1}`` is the gap shift ``d_t + e_t - e_t^opp`` of stage ``t``; its grid
covers every reachable value, ``[-(domain_half(t) + e_range), domain_half(t) + e_range]``, and the
quadrature nodes ``y + z`` must lie in ``D_{t+1}`` (checked at every chunk).

Integration rule = the locked v2.0 rule (composite Gauss-Legendre, 6 nodes per panel, panels of width
1 aligned at z = 0 and +-2q, y-grid step 0.05, float64, linear lookup); the frozen actor is evaluated
directly at the nodes, never interpolated. The build is pure: no RNG is drawn, nothing is mutated.
At T = 2 the table of stage 1 equals ``utils.v2_continuation.build_continuation_table`` bit for bit
(same operations, same chunking).

The closed-form equilibrium does not enter.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional

import numpy as np

from envs.curriculum_env import GameSpec
from utils import v2_continuation as vc
from utils.v2_continuation import (
    ActorFn, ContinuationTable, DEFAULT_NODES_PER_PANEL, DEFAULT_PANEL_WIDTH, DEFAULT_STEP,
    frozen_mean_effort, shock_quadrature, terminal_value)

SCHEMA = "ms_continuation_table/1"


def y_grid_for_stage(spec: GameSpec, t: int, step: float = DEFAULT_STEP) -> np.ndarray:
    """Symmetric y-grid of the table used by the phase of stage ``t``, containing 0, spacing <= step.

    The half-width is ``domain_half(t) + e_range``; for t = 1 it is ``e_range``, the grid of
    ``utils.v2_continuation.y_grid_for``.
    """
    half = spec.domain_half(t) + spec.e_range
    n_half = int(np.ceil(half / float(step) - 1e-9))
    return np.linspace(-half, half, 2 * n_half + 1)


def g_values(actor: ActorFn, spec: GameSpec, s: int, d: np.ndarray,
             next_table: Optional[ContinuationTable]) -> np.ndarray:
    """Continuation value ``g_s(d)`` of the frozen stage-``s`` policy (both players execute the mean).

    Args:
        actor: Frozen actor of stage ``s``.
        spec: Game specification.
        s: Stage of the actor, ``2 <= s <= T``.
        d: Signed gaps (float64) inside D_s.
        next_table: Table of stage ``s`` (``V~_{s+1}`` as a function of ``d + e_s - e_s^opp``); required
            iff ``s < T``.

    Returns:
        Float64 values, 1-D.
    """
    d = np.asarray(d, dtype=float).reshape(-1)
    both = frozen_mean_effort(actor, spec, s, np.concatenate([d, -d]))
    e_own, e_opp = both[:d.size], both[d.size:]
    if s == spec.T:
        return terminal_value(spec, d, e_own, e_opp)
    if next_table is None:
        raise ValueError(f"stage {s} < T={spec.T} needs the continuation table of stage {s}")
    return -spec.k * e_own ** 2 + next_table.lookup(d + e_own - e_opp)


def build_table(actor: ActorFn, spec: GameSpec, t: int, next_table: Optional[ContinuationTable] = None,
                step: float = DEFAULT_STEP, panel_width: float = DEFAULT_PANEL_WIDTH,
                nodes_per_panel: int = DEFAULT_NODES_PER_PANEL) -> ContinuationTable:
    """Table ``V~_{t+1}`` used by the phase of stage ``t`` from the frozen stage-``t+1`` actor.

    Args:
        actor: Frozen actor of stage ``t + 1``.
        spec: Game specification.
        t: Stage whose phase uses the table, ``1 <= t <= T - 1``.
        next_table: The table of stage ``t + 1`` (``V~_{t+2}``); required iff ``t + 1 < T``.
        step: Maximal y-grid spacing.
        panel_width: Maximal quadrature panel width.
        nodes_per_panel: Gauss-Legendre nodes per panel.

    Returns:
        A :class:`ContinuationTable` on :func:`y_grid_for_stage` ``(spec, t)``.

    Raises:
        ValueError: ``t`` out of range, a missing ``next_table``, or a quadrature node outside D_{t+1}.
    """
    T = int(spec.T)
    if not 1 <= int(t) <= T - 1:
        raise ValueError(f"t must lie in [1, {T - 1}] for T={T}; got {t}")
    s = int(t) + 1
    if s < T and next_table is None:
        raise ValueError(f"the table of stage {s} is required to build the table of stage {t}")
    t0 = time.perf_counter()
    y = y_grid_for_stage(spec, t, step)
    nodes, weights, n_panels = shock_quadrature(spec.q, panel_width, nodes_per_panel)
    half = spec.domain_half(s)
    rows = max(1, vc._MAX_POINTS_PER_CHUNK // (2 * nodes.size))
    values = np.empty(y.size)
    for i in range(0, y.size, rows):
        d = y[i:i + rows, None] + nodes[None, :]
        if np.abs(d).max() > half * (1.0 + vc._EDGE_TOL):
            raise ValueError(f"landing gap {np.abs(d).max():.6g} outside D_{s} (half {half:g})")
        g = g_values(actor, spec, s, d.reshape(-1), next_table).reshape(d.shape)
        values[i:i + rows] = g @ weights
    meta: Dict[str, Any] = {
        "schema": SCHEMA,
        "rule": "composite Gauss-Legendre in z, panels aligned at z=0 and z=+-2q; frozen actor "
                "evaluated directly at d=y+z (no interpolation of g); nested through the linear "
                "lookup of the next table",
        "stage": int(s), "phase_stage": int(t), "T": T, "nested": bool(s < T),
        "q": float(spec.q), "w_h": float(spec.w_h), "w_l": float(spec.w_l), "k": float(spec.k),
        "dw": float(spec.dw), "e_min": float(spec.e_min), "e_max": float(spec.e_max),
        "step_requested": float(step), "step_actual": float(y[1] - y[0]), "n_y": int(y.size),
        "y_half": float(y[-1]), "panel_width": float(panel_width),
        "nodes_per_panel": int(nodes_per_panel), "n_panels": int(n_panels),
        "n_nodes": int(nodes.size), "weight_sum": float(weights.sum()),
        "conc_scale": float(getattr(actor, "conc_scale", 1.0)),
        "build_seconds": float(time.perf_counter() - t0),
    }
    return ContinuationTable(y_grid=y, values=values, meta=meta)


def build_nested_tables(actors: Mapping[int, ActorFn], spec: GameSpec, t_min: int = 1,
                        step: float = DEFAULT_STEP, panel_width: float = DEFAULT_PANEL_WIDTH,
                        nodes_per_panel: int = DEFAULT_NODES_PER_PANEL) -> Dict[int, ContinuationTable]:
    """All tables ``t = T-1, ..., t_min`` of a frozen suffix ``actors = {s: actor_s}`` for s = t_min+1..T.

    Returns:
        ``{t: V~_{t+1} as a function of y at stage t}``.
    """
    T = int(spec.T)
    tables: Dict[int, ContinuationTable] = {}
    for t in range(T - 1, int(t_min) - 1, -1):
        tables[t] = build_table(actors[t + 1], spec, t, tables.get(t + 1), step, panel_width,
                                nodes_per_panel)
    return tables
