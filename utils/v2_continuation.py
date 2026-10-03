"""Expected-continuation table for the stage-1 return (R1 refinement, method 6).

Definition (PI spec D4). For the final stage ``stage`` of a two-player tournament with the
FROZEN stage policy, both players execute the Beta mean ``e_hat(d)`` and the conditional
expected terminal reward from a stage-``stage`` gap ``d`` is

    g(d)  = w_l + DW * F_xi(d + e_hat(d) - e_hat(-d)) - k * e_hat(d)^2

(exactly ``run.v2_rollout.expected_terminal_reward`` with both players executing the mean).
The table stores the shock-integrated value as a function of the stage-1 gap shift
``y = e_1 - e_1^opp`` (both efforts as executed)

    V(y) = E_z[ g(y + z) ],   z = eps_L - eps_O,   density f(z) = (2q - |z|) / (4 q^2) on
    [-2q, 2q]

on a symmetric y-grid of step <= 0.05 covering [-e_range, e_range], float64 throughout, with
linear interpolation at lookup.

``e_hat`` is formed exactly as in ``continuation_action_mode="mean"``: observation
``spec.encode_obs(stage, d)`` (float32), ``(alpha, beta) = actor(obs)`` in float32 without
gradient, ``mean = alpha / (alpha + beta)`` in float64 from the float32 values,
``e_hat = e_min + e_range * mean``. If the actor carries a concentration scale, its own
``forward`` applies it (the mean does not depend on it).

Integration rule (the only numerical approximation). The frozen actor is evaluated DIRECTLY at
the quadrature nodes ``d = y + z``; g is never interpolated. The z-integral uses a composite
Gauss-Legendre rule with ``n`` nodes per panel on equal panels of width <= ``panel_width``;
the panels are aligned with the kink points of the density: ``[-2q, 0]`` and ``[0, 2q]`` are each
split into ``ceil(2q / panel_width)`` equal panels, so z = 0 and z = +-2q are panel edges and f
is exactly linear inside every panel. The weights are ``(panel half-width) * w_GL * f(node)``
and sum to 1 to rounding. The remaining non-smooth points of the integrand are the kinks of
``F_xi`` in g (where ``d + e_hat(d) - e_hat(-d)`` crosses 0 and +-2q); they depend on y and
sit inside panels, which is why panels are narrow. The default rule and its convergence evidence
(halving the panel width and doubling the nodes: max abs change of the table in units of DW) are
recorded in ``results/v2_refine/continuation_check.json`` (key ``convergence``).

This module deliberately imports neither ``utils.dp_br_verifier`` nor ``utils.v2_metrics``, so
that agreement with the verifier's stage-1 Q (checked in
``tests/test_v2_refine_continuation.py``) is an independent check. Measured outcome of that
check (same JSON, keys ``spec_requirement`` and ``table_vs_verifier``): on the production final
verifier tier the table does NOT meet the 1e-6 DW of PROMPT.md 2.3 (ii); the discrepancy is
verifier-side (its stage-2 interpolation and Gauss-Legendre resolution) and the criterion is
open for a PI decision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch

from envs.curriculum_env import GameSpec
from utils.theory_multistage import F_xi

DEFAULT_STEP = 0.05
DEFAULT_PANEL_WIDTH = 1.0
DEFAULT_NODES_PER_PANEL = 6
_MAX_POINTS_PER_CHUNK = 1 << 19      # actor evaluations per forward call (d and -d together)
_EDGE_TOL = 1e-9                     # float noise tolerated on the y range / the stage domain

ActorFn = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


# ---------------------------------------------------------------------------
# Frozen-policy pieces
# ---------------------------------------------------------------------------

def _device_of(actor: Any) -> torch.device:
    """Device of an ``nn.Module`` actor (CPU for a plain callable)."""
    if isinstance(actor, torch.nn.Module):
        for p in actor.parameters():
            return p.device
    return torch.device("cpu")


def _check_stage(spec: GameSpec, stage: int) -> None:
    """Validate the stage index (the integrated stage must be the final one)."""
    if int(stage) != int(spec.T) or int(stage) < 2:
        raise ValueError(
            f"stage must be the final stage T={spec.T} (>= 2) of the game, got {stage}")


@torch.no_grad()
def frozen_mean_effort(actor: ActorFn, spec: GameSpec, stage: int, d: np.ndarray) -> np.ndarray:
    """Frozen Beta-mean effort ``e_hat(d)`` on the float path of ``continuation_action_mode=mean``.

    Args:
        actor: Frozen actor, ``actor(obs_float32_tensor) -> (alpha, beta)`` (float32 tensors).
        spec: Game specification.
        stage: Stage whose policy is evaluated (enters only through ``spec.encode_obs``).
        d: Signed gaps (float64), any shape (flattened internally).

    Returns:
        Float64 efforts, 1-D.
    """
    obs = spec.encode_obs(stage, np.asarray(d, dtype=float).reshape(-1))
    x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=_device_of(actor))
    a, b = actor(x)
    a = a.cpu().numpy().astype(float)
    b = b.cpu().numpy().astype(float)
    return spec.e_min + spec.e_range * (a / (a + b))


def terminal_value(spec: GameSpec, d: np.ndarray, e_own: np.ndarray, e_opp: np.ndarray
                   ) -> np.ndarray:
    """Conditional expected terminal reward ``w_l + DW F_xi(d + e_own - e_opp) - k e_own^2``."""
    d = np.asarray(d, dtype=float)
    e_own = np.asarray(e_own, dtype=float)
    y = d + e_own - np.asarray(e_opp, dtype=float)
    return spec.w_l + spec.dw * F_xi(y, spec.q) - spec.k * e_own ** 2


def g_frozen(actor: ActorFn, spec: GameSpec, stage: int, d: np.ndarray) -> np.ndarray:
    """Stage-``stage`` continuation value g(d) with both players executing the frozen mean.

    Args:
        actor: Frozen actor.
        spec: Game specification.
        stage: Final stage index.
        d: Signed gaps (float64).

    Returns:
        ``terminal_value(d, e_hat(d), e_hat(-d))`` as float64, flattened to 1-D.
    """
    d = np.asarray(d, dtype=float).reshape(-1)
    both = frozen_mean_effort(actor, spec, stage, np.concatenate([d, -d]))
    return terminal_value(spec, d, both[:d.size], both[d.size:])


# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------

def shock_quadrature(q: float, panel_width: float, nodes_per_panel: int
                     ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Composite Gauss-Legendre rule for ``E_z[.]`` with the triangular density on [-2q, 2q].

    ``[-2q, 0]`` and ``[0, 2q]`` are each split into ``ceil(2q / panel_width)`` equal panels
    (so z = 0 and z = +-2q are panel edges); weights include the density at the node.

    Args:
        q: Per-player noise half-width.
        panel_width: Maximal panel width.
        nodes_per_panel: Gauss-Legendre nodes per panel.

    Returns:
        ``(nodes, weights, n_panels)``; nodes are strictly interior to (-2q, 2q).
    """
    q = float(q)
    m = max(1, int(np.ceil(2.0 * q / float(panel_width) - 1e-9)))
    x, w = np.polynomial.legendre.leggauss(int(nodes_per_panel))
    edges = np.linspace(0.0, 2.0 * q, m + 1)
    lo, hi = edges[:-1][:, None], edges[1:][:, None]
    half = 0.5 * (hi - lo)
    pos = (0.5 * (lo + hi) + half * x[None, :]).reshape(-1)
    w_pos = (half * w[None, :]).reshape(-1)
    nodes = np.concatenate([-pos[::-1], pos])
    base = np.concatenate([w_pos[::-1], w_pos])
    weights = base * (2.0 * q - np.abs(nodes)) / (4.0 * q * q)
    return nodes, weights, 2 * m


def y_grid_for(spec: GameSpec, step: float = DEFAULT_STEP) -> np.ndarray:
    """Symmetric grid on ``[-e_range, e_range]`` containing 0, spacing <= ``step``."""
    n_half = int(np.ceil(spec.e_range / float(step) - 1e-9))
    return np.linspace(-spec.e_range, spec.e_range, 2 * n_half + 1)


def expected_continuation(actor: ActorFn, spec: GameSpec, stage: int, y: np.ndarray,
                          panel_width: float = DEFAULT_PANEL_WIDTH,
                          nodes_per_panel: int = DEFAULT_NODES_PER_PANEL) -> np.ndarray:
    """``V(y) = E_z[g(y + z)]`` at the given y values by the composite rule of the module doc.

    Args:
        actor: Frozen actor.
        spec: Game specification (``spec.T`` must equal ``stage``).
        stage: Final stage index.
        y: Gap shifts (float64), 1-D, each within ``[-e_range, e_range]``.
        panel_width: Maximal quadrature panel width.
        nodes_per_panel: Gauss-Legendre nodes per panel.

    Returns:
        Float64 values, same length as ``y``.

    Raises:
        ValueError: If ``stage`` is not the final stage or a landing gap leaves D_stage.
    """
    _check_stage(spec, stage)
    y = np.asarray(y, dtype=float).reshape(-1)
    nodes, weights, _ = shock_quadrature(spec.q, panel_width, nodes_per_panel)
    half = spec.domain_half(stage)
    rows = max(1, _MAX_POINTS_PER_CHUNK // (2 * nodes.size))
    out = np.empty(y.size)
    for i in range(0, y.size, rows):
        d = y[i:i + rows, None] + nodes[None, :]
        if np.abs(d).max() > half * (1.0 + _EDGE_TOL):
            raise ValueError(f"landing gap {np.abs(d).max():.6g} outside D_{stage} (half {half:g})")
        g = g_frozen(actor, spec, stage, d.reshape(-1)).reshape(d.shape)
        out[i:i + rows] = g @ weights
    return out


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

@dataclass
class ContinuationTable:
    """Tabulated ``V(y)`` with linear interpolation.

    Attributes:
        y_grid: Increasing float64 grid of gap shifts (symmetric, contains 0).
        values: Float64 ``V`` at the grid nodes.
        meta: Build record (rule, panels, nodes, weight sum, timing, game).
    """

    y_grid: np.ndarray
    values: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def lookup(self, y: np.ndarray) -> np.ndarray:
        """Linear interpolation of the table at ``y`` (float64).

        Args:
            y: Gap shifts, any shape.

        Returns:
            Interpolated values with the shape of ``y``.

        Raises:
            ValueError: If any ``y`` lies outside the grid by more than float noise.
        """
        y = np.asarray(y, dtype=float)
        lo, hi = float(self.y_grid[0]), float(self.y_grid[-1])
        tol = _EDGE_TOL * max(1.0, hi - lo)
        if y.size and (not np.all(np.isfinite(y)) or y.min() < lo - tol or y.max() > hi + tol):
            raise ValueError(
                f"lookup y in [{y.min():.9g}, {y.max():.9g}] outside the table grid "
                f"[{lo:g}, {hi:g}]")
        return np.interp(np.clip(y, lo, hi), self.y_grid, self.values)


def build_continuation_table(actor: ActorFn, spec: GameSpec, stage: int,
                             step: float = DEFAULT_STEP,
                             panel_width: float = DEFAULT_PANEL_WIDTH,
                             nodes_per_panel: int = DEFAULT_NODES_PER_PANEL) -> ContinuationTable:
    """Build the expected-continuation table of the frozen stage-``stage`` policy.

    Args:
        actor: Frozen actor (not modified; evaluated under ``torch.no_grad``).
        spec: Game specification (``spec.T`` must equal ``stage``).
        stage: Index of the final stage whose frozen Beta mean is integrated.
        step: Maximal y-grid spacing (<= 0.05 for the R1 protocol).
        panel_width: Maximal quadrature panel width.
        nodes_per_panel: Gauss-Legendre nodes per panel.

    Returns:
        A :class:`ContinuationTable` on the y-grid of :func:`y_grid_for`.
    """
    _check_stage(spec, stage)
    t0 = time.perf_counter()
    y = y_grid_for(spec, step)
    values = expected_continuation(actor, spec, stage, y, panel_width, nodes_per_panel)
    nodes, weights, n_panels = shock_quadrature(spec.q, panel_width, nodes_per_panel)
    meta: Dict[str, Any] = {
        "schema": "v2_continuation_table/1",
        "rule": "composite Gauss-Legendre in z, panels aligned at z=0 and z=+-2q; frozen actor "
                "evaluated directly at d=y+z (no interpolation of g)",
        "stage": int(stage), "q": float(spec.q), "w_h": float(spec.w_h), "w_l": float(spec.w_l),
        "k": float(spec.k), "dw": float(spec.dw), "e_min": float(spec.e_min),
        "e_max": float(spec.e_max), "step_requested": float(step),
        "step_actual": float(y[1] - y[0]), "n_y": int(y.size),
        "panel_width": float(panel_width), "nodes_per_panel": int(nodes_per_panel),
        "n_panels": int(n_panels), "n_nodes": int(nodes.size),
        "weight_sum": float(weights.sum()),
        "conc_scale": float(getattr(actor, "conc_scale", 1.0)),
        "build_seconds": float(time.perf_counter() - t0),
    }
    return ContinuationTable(y_grid=y, values=values, meta=meta)
