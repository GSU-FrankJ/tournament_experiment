"""Independent best-response calculator of round T3-R2 (audit A5).

For a policy given as mean-effort functions ``policy(t, d) -> effort`` (float64), the calculator computes, for the
two-player T-stage tournament of the round, the quantities of the prompt's table (D3, A5):

    Q_T(x, e)    = w_l + DW F(x + e - e_T(-x)) - k e^2                        (terminal stage)
    W_{t+1}(y)   = E_xi[ V_{t+1}(y + xi) ]                                    (mean and best-response chain)
    Q_t^m(d, e)  = -k e^2 + W_{t+1}^mean(d + e - e_t(-d))
    V_t^mean(d)  = Q_t^m(d, e_t(d));   V_t^BR(d) = max_e { -k e^2 + W_{t+1}^BR(d + e - e_t(-d)) }
    Delta_t(d)   = max_e Q_t^m(d, e) - V_t^mean(d);   G_t(d) = V_t^BR(d) - V_t^mean(d);   EXP_root = G_1(0)

Numerics (all float64, none of them taken from the verifier or the metrics module): the value functions are
tabulated on a symmetric grid of D_t (step <= 0.05) and interpolated by a cubic spline; the expectation is a
composite Gauss-Legendre rule on panels of width <= 1 aligned at z = 0 and z = +-2q (at least 4 nodes per panel, the
triangular density in the weights); every maximisation over the effort is a grid search (step 0.01 on [e_min, e_max])
followed by a bounded refinement around the best grid point and at both bounds (golden section, vectorised, for the
tabulations; ``scipy`` bounded Brent for the individually reported states). The learned effort is always one of the
candidates, so ``Delta >= 0`` and ``G >= 0`` by construction.

The module imports no verifier and no metrics numerics (an AST test checks the names); only the triangular density
and its CDF of ``utils.theory_multistage`` are used (their own test integrates the density numerically).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from utils.theory_multistage import F_xi, f_xi

Policy = Callable[[int, np.ndarray], np.ndarray]
GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True)
class Game:
    """Parameters of the tournament (the ``GameSpec`` fields the calculator needs)."""

    w_h: float
    w_l: float
    k: float
    q: float
    T: int
    e_min: float = 0.0
    e_max: float = 100.0

    @property
    def dw(self) -> float:
        """Prize spread."""
        return float(self.w_h - self.w_l)

    @property
    def e_range(self) -> float:
        """Effort range."""
        return float(self.e_max - self.e_min)

    @property
    def B(self) -> float:
        """Largest one-stage change of the gap."""
        return self.e_range + 2.0 * float(self.q)

    def half(self, t: int) -> float:
        """Half-width (t - 1) B of the gap domain D_t."""
        return (int(t) - 1) * self.B


@dataclass(frozen=True)
class CalcConfig:
    """Resolution of the calculator (defaults: the A5 settings)."""

    table_step: float = 0.05
    effort_step: float = 0.01
    panel_width: float = 1.0
    nodes_per_panel: int = 4
    golden_iters: int = 60
    chunk_cells: int = 4_000_000

    def refined(self) -> "CalcConfig":
        """Every step halved, the nodes per panel doubled (the convergence test)."""
        return CalcConfig(self.table_step / 2.0, self.effort_step / 2.0, self.panel_width / 2.0,
                          self.nodes_per_panel * 2, self.golden_iters, self.chunk_cells)


# ---------------------------------------------------------------------------------- grids and quadrature
def sym_grid(half: float, step: float) -> np.ndarray:
    """Symmetric grid on ``[-half, half]`` with spacing <= ``step``; the middle node is exactly 0."""
    n = max(1, int(np.ceil(half / float(step) - 1e-9))) if half > 0.0 else 0
    if n == 0:
        return np.zeros(1)
    return np.arange(-n, n + 1) * (half / n)


def shock_rule(q: float, panel_width: float, nodes_per_panel: int) -> Tuple[np.ndarray, np.ndarray]:
    """Composite Gauss-Legendre rule for ``E_xi[.]``, xi triangular on [-2q, 2q].

    ``[0, 2q]`` is split into ``ceil(2q / panel_width)`` equal panels (so z = 0 and z = +-2q are panel edges and the
    density is exactly linear inside every panel); the rule is mirrored to ``[-2q, 0]``; the weights carry the
    density at the node and sum to 1 up to rounding.
    """
    q = float(q)
    m = max(1, int(math.ceil(2.0 * q / float(panel_width) - 1e-9)))
    x, w = np.polynomial.legendre.leggauss(int(nodes_per_panel))
    edges = np.linspace(0.0, 2.0 * q, m + 1)
    lo, hi = edges[:-1], edges[1:]
    mid, hw = 0.5 * (lo + hi), 0.5 * (hi - lo)
    pos = (mid[:, None] + hw[:, None] * x[None, :]).reshape(-1)
    wpos = (hw[:, None] * w[None, :]).reshape(-1)
    nodes = np.concatenate([-pos[::-1], pos])
    base = np.concatenate([wpos[::-1], wpos])
    return nodes, base * f_xi(nodes, q)


# ---------------------------------------------------------------------------------- the calculator
class BRCalculator:
    """Tabulating best-response calculator for one policy (see the module docstring).

    Args:
        game: Game parameters (``T`` >= 2).
        policy: ``policy(t, d) -> mean effort`` for ``t = 1..T`` (float64 array in, float64 array out).
        cfg: Resolution.
    """

    def __init__(self, game: Game, policy: Policy, cfg: CalcConfig = CalcConfig()):
        if int(game.T) < 2:
            raise ValueError("the calculator needs T >= 2")
        self.game = game
        self.policy = policy
        self.cfg = cfg
        self.nodes, self.weights = shock_rule(game.q, cfg.panel_width, cfg.nodes_per_panel)
        n_e = int(round(game.e_range / cfg.effort_step))
        self.E = np.linspace(game.e_min, game.e_max, n_e + 1)
        self._V: Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._W: Dict[Tuple[int, str], CubicSpline] = {}

    # ------------------------------------------------------------------ policy helpers
    def effort(self, t: int, d: np.ndarray) -> np.ndarray:
        """Learned mean effort ``e_t(d)`` (float64, 1-D)."""
        out = np.asarray(self.policy(int(t), np.asarray(d, dtype=float)), dtype=float).reshape(-1)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError(f"policy returned a non-finite effort at stage {t}")
        return out

    # ------------------------------------------------------------------ continuation functions
    def cont(self, t: int, kind: str) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
        """``(value, slope)`` of the continuation of stage ``t`` as functions of ``y = d + e - e_opp``.

        At the terminal stage the value is ``w_l + DW F(y)`` (closed form); below it is ``W_{t+1}^kind`` (cubic spline).
        """
        g = self.game
        if int(t) == int(g.T):
            return (lambda y: g.w_l + g.dw * F_xi(y, g.q)), (lambda y: g.dw * f_xi(y, g.q))
        spl = self.W(int(t) + 1, kind)
        lim = g.half(t) + g.e_range
        return ((lambda y: spl(np.clip(y, -lim, lim))),
                (lambda y: spl(np.clip(y, -lim, lim), 1)))

    def V(self, t: int, kind: str) -> Tuple[np.ndarray, np.ndarray]:
        """Table ``V_t^kind`` on the grid of D_t (t >= 2; ``kind`` is ``mean`` or ``br``)."""
        key = (int(t), kind)
        if key in self._V:
            return self._V[key]
        g = self.game
        if int(t) < 2:
            raise ValueError("V is tabulated for t >= 2 (D_1 is the single state 0)")
        grid = sym_grid(g.half(t), self.cfg.table_step)
        e_hat = self.effort(t, grid)
        base = grid - self.effort(t, -grid)
        cont, _ = self.cont(t, kind)
        if kind == "mean":
            vals = -g.k * e_hat ** 2 + cont(base + e_hat)
        elif kind == "br":
            vals, _ = self.maximize(cont, base, e_hat)
        else:
            raise ValueError(f"unknown kind {kind!r}")
        self._V[key] = (grid, vals)
        return self._V[key]

    def W(self, s: int, kind: str) -> CubicSpline:
        """Spline of ``W_s^kind(y) = E_xi[V_s^kind(y + xi)]`` on the y-grid of stage ``s - 1``."""
        key = (int(s), kind)
        if key in self._W:
            return self._W[key]
        g = self.game
        grid, vals = self.V(s, kind)
        vspl = CubicSpline(grid, vals)
        half_y = g.half(s - 1) + g.e_range
        y = sym_grid(half_y, self.cfg.table_step)
        lim = g.half(s)
        rows = max(1, self.cfg.chunk_cells // self.nodes.size)
        w = np.empty(y.size)
        for i in range(0, y.size, rows):
            pts = np.clip(y[i:i + rows, None] + self.nodes[None, :], -lim, lim)
            w[i:i + rows] = vspl(pts) @ self.weights
        self._W[key] = CubicSpline(y, w)
        return self._W[key]

    # ------------------------------------------------------------------ maximisation over the effort
    def _golden(self, f: Callable[[np.ndarray], np.ndarray], lo: np.ndarray, hi: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised golden-section maximisation of ``f`` on ``[lo, hi]`` (one bracket per row)."""
        a, c = lo.astype(float).copy(), hi.astype(float).copy()
        x1, x2 = c - GOLDEN * (c - a), a + GOLDEN * (c - a)
        f1, f2 = f(x1), f(x2)
        for _ in range(self.cfg.golden_iters):
            left = f1 > f2
            c = np.where(left, x2, c)
            a = np.where(left, a, x1)
            xn = np.where(left, c - GOLDEN * (c - a), a + GOLDEN * (c - a))
            fn = f(xn)
            x2n = np.where(left, x1, xn)
            f2n = np.where(left, f1, fn)
            x1n = np.where(left, xn, x2)
            f1n = np.where(left, fn, f2)
            x1, x2, f1, f2 = x1n, x2n, f1n, f2n
        take1 = f1 >= f2
        return np.where(take1, x1, x2), np.where(take1, f1, f2)

    def maximize(self, cont: Callable[[np.ndarray], np.ndarray], base: np.ndarray,
                 e_extra: Optional[np.ndarray] = None, brent: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """``max_e { -k e^2 + cont(base + e) }`` for every ``base`` (the effort on [e_min, e_max]).

        Grid search on :attr:`E`, bounded refinement around the best grid point and at both bounds, and the optional
        extra candidate ``e_extra`` (the learned effort). Ties go to the smaller effort.

        Args:
            cont: Continuation value as a function of ``y = base + e``.
            base: ``d - e_opp`` per state.
            e_extra: Optional extra candidate effort per state.
            brent: Use ``scipy`` bounded Brent per state (slow; for reported states) instead of the vectorised golden
                section.

        Returns:
            ``(value, argmax effort)``.
        """
        g, E = self.game, self.E
        k = g.k
        base = np.asarray(base, dtype=float).reshape(-1)
        M = base.size
        h = float(E[1] - E[0])
        n_cols = E.size
        rows = max(1, self.cfg.chunk_cells // n_cols)
        vbest, abest = np.empty(M), np.empty(M)
        for i in range(0, M, rows):
            b = base[i:i + rows]
            m = b.size
            vals = -k * E[None, :] ** 2 + cont(b[:, None] + E[None, :])
            j = np.argmax(vals, axis=1)
            cand_v = [vals[np.arange(m), j]]
            cand_a = [E[j]]
            brackets = [(np.maximum(E[j] - h, g.e_min), np.minimum(E[j] + h, g.e_max)),
                        (np.full(m, g.e_min), np.full(m, min(g.e_min + h, g.e_max))),
                        (np.full(m, max(g.e_max - h, g.e_min)), np.full(m, g.e_max))]
            for lo, hi in brackets:
                if brent:
                    a_ = np.empty(m)
                    for r in range(m):
                        res = minimize_scalar(lambda e, br=float(b[r]): -float(-k * e * e + cont(np.array([br + e]))[0]),
                                              bounds=(float(lo[r]), float(hi[r])), method="bounded",
                                              options={"xatol": 1e-12})
                        a_[r] = res.x
                else:
                    a_, _ = self._golden(lambda x, b=b: -k * x ** 2 + cont(b + x), lo, hi)
                cand_a.append(a_)
                cand_v.append(-k * a_ ** 2 + cont(b + a_))
            if e_extra is not None:
                ex = np.asarray(e_extra, dtype=float).reshape(-1)[i:i + rows]
                cand_a.append(ex)
                cand_v.append(-k * ex ** 2 + cont(b + ex))
            V = np.stack(cand_v)                     # (C, m)
            A = np.stack(cand_a)
            top = V.max(axis=0)
            tie = V >= top[None, :]
            pick = np.where(tie, A, np.inf).argmin(axis=0)       # smaller effort among the maximal candidates
            vbest[i:i + rows] = top
            abest[i:i + rows] = A[pick, np.arange(m)]
        return vbest, abest

    # ------------------------------------------------------------------ per-state report
    def stage_values(self, t: int, d: Sequence[float], brent: bool = True) -> Dict[str, np.ndarray]:
        """All quantities of the A5 table at the states ``d`` of stage ``t``.

        Returns:
            Dict of 1-D arrays: ``d``, ``e_hat`` (learned effort), ``e_opp``, ``V_mean``, ``V_br``, ``Delta`` (one-step
            gain), ``G`` (dynamic gain), ``a_dev`` (one-step best-response effort), ``a_br`` (dynamic best-response
            effort), ``slope`` (dQ^m/de at the learned effort).
        """
        g, t = self.game, int(t)
        d = np.asarray(d, dtype=float).reshape(-1)
        e_hat = self.effort(t, d)
        e_opp = self.effort(t, -d)
        base = d - e_opp
        cont_m, slope_m = self.cont(t, "mean")
        v_mean = -g.k * e_hat ** 2 + cont_m(base + e_hat)
        q_dev, a_dev = self.maximize(cont_m, base, e_hat, brent=brent)
        if t == int(g.T):
            q_br, a_br = q_dev, a_dev
        else:
            cont_b, _ = self.cont(t, "br")
            q_br, a_br = self.maximize(cont_b, base, e_hat, brent=brent)
        return {"d": d, "e_hat": e_hat, "e_opp": e_opp, "V_mean": v_mean, "V_br": q_br,
                "Delta": q_dev - v_mean, "G": q_br - v_mean, "a_dev": a_dev, "a_br": a_br,
                "slope": -2.0 * g.k * e_hat + slope_m(base + e_hat)}

    def mean_value(self, t: int, d: Sequence[float]) -> np.ndarray:
        """``V_t^mean(d)`` at the states ``d`` (needs the mean chain only: no maximisation)."""
        g = self.game
        d = np.asarray(d, dtype=float).reshape(-1)
        e_hat = self.effort(t, d)
        base = d - self.effort(t, -d)
        cont_m, _ = self.cont(int(t), "mean")
        return -g.k * e_hat ** 2 + cont_m(base + e_hat)

    def with_policy(self, policy: Policy, shared_from_stage: int) -> "BRCalculator":
        """A calculator for another policy that shares the tables of the stages ``>= shared_from_stage``.

        Valid only if ``policy`` agrees with this calculator's policy at those stages (the tables ``V_s`` and ``W_s`` for
        ``s >= shared_from_stage`` depend on the policy at the stages ``>= s`` only).
        """
        other = BRCalculator(self.game, policy, self.cfg)
        other._V = {k: v for k, v in self._V.items() if k[0] >= shared_from_stage}
        other._W = {k: v for k, v in self._W.items() if k[0] >= shared_from_stage}
        return other

    def root(self) -> Dict[str, float]:
        """Root values: ``V_mean`` and ``V_br`` at (1, 0) and ``EXP_root = G_1(0)``."""
        r = self.stage_values(1, [0.0])
        return {"V1_mean": float(r["V_mean"][0]), "V1_br": float(r["V_br"][0]), "EXP_root": float(r["G"][0]),
                "Delta_1": float(r["Delta"][0]), "e1": float(r["e_hat"][0]), "a_br_1": float(r["a_br"][0])}

    def q_curve(self, t: int, d: float, e: np.ndarray, kind: str = "mean") -> np.ndarray:
        """``Q_t(d, e)`` at the efforts ``e`` for one state (continuation ``mean`` or ``br``)."""
        g = self.game
        e = np.asarray(e, dtype=float)
        base = float(d) - float(self.effort(t, np.array([-float(d)]))[0])
        cont, _ = self.cont(t, kind)
        return -g.k * e ** 2 + cont(base + e)

    def w_values(self, s: int, y: np.ndarray, kind: str = "mean") -> np.ndarray:
        """``W_s^kind(y)`` (the table of stage ``s - 1``'s phase) at the points ``y``."""
        return self.W(s, kind)(np.asarray(y, dtype=float))


# ---------------------------------------------------------------------------------- convergence test
def convergence(game: Game, policy: Policy, points: Sequence[Tuple[int, float]], cfg: CalcConfig = CalcConfig(),
                brent: bool = True) -> Dict[str, Any]:
    """Change of the reported quantities when every step is halved and the nodes per panel doubled.

    Args:
        game: Game.
        policy: Mean-effort functions.
        points: ``(t, d)`` states (stage 1 uses d = 0).
        cfg: The base resolution.
        brent: Bounded Brent for the individual states.

    Returns:
        ``{"rows": [...], "max_value_change_over_dw", "max_effort_change", "ok_values", "ok_efforts"}``: per point the
        absolute changes of ``Delta``, ``G``, ``V_mean``, ``V_br`` (in units of DW) and of the learned, one-step and
        dynamic best-response efforts.
    """
    a = BRCalculator(game, policy, cfg)
    b = BRCalculator(game, policy, cfg.refined())
    rows = []
    for t, d in points:
        ra, rb = a.stage_values(int(t), [float(d)], brent), b.stage_values(int(t), [float(d)], brent)
        row: Dict[str, Any] = {"t": int(t), "d": float(d)}
        for key in ("Delta", "G", "V_mean", "V_br"):
            row[f"{key}_change_over_dw"] = abs(float(ra[key][0]) - float(rb[key][0])) / game.dw
        for key in ("e_hat", "a_dev", "a_br"):
            row[f"{key}_change"] = abs(float(ra[key][0]) - float(rb[key][0]))
        rows.append(row)
    vmax = max(max(r[f"{k}_change_over_dw"] for k in ("Delta", "G", "V_mean", "V_br")) for r in rows)
    emax = max(max(r[f"{k}_change"] for k in ("e_hat", "a_dev", "a_br")) for r in rows)
    return {"rows": rows, "max_value_change_over_dw": vmax, "max_effort_change": emax,
            "ok_values": bool(vmax <= 1e-6), "ok_efforts": bool(emax <= 1e-3)}


def closed_form_policy(game: Game) -> Policy:
    """The T = 2 closed-form policy as functions (tent at stage 2, g1 at the root): used by the tests and the report
    only, never by anything that trains."""
    from utils.theory_multistage import g1_two_stage, g2_two_stage

    g1 = g1_two_stage(game.q, game.w_h, game.w_l, game.k)

    def pol(t: int, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        if int(t) == 1:
            return np.full(d.shape, g1)
        return g2_two_stage(d, game.q, game.w_h, game.w_l, game.k, game.e_max)
    return pol
