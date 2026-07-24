#!/usr/bin/env python3
"""Deterministic exploitability referee for the one-stage two-player tournament.

NEW analysis module (Phase 0 of the one-stage raw-vs-polish ablation). It does
NOT import or modify any existing agent/env/runner/polish module: it is a
self-contained closed-form evaluator used as the *measurement* instrument, with
the legacy Monte-Carlo estimator reported only as a comparison exhibit.

Model (canonical one-stage Set 1):
    y_i = e_i + eps_i,  eps ~ U(-q, q) i.i.d.  =>  xi = eps_i - eps_j ~ Tri(-2q, 2q)
    P(i wins | e, e_opp) = P(xi > e_opp - e) = 1 - F_xi(e_opp - e)
    U(e; e_opp) = w_L + DW * (1 - F_xi(e_opp - e)) - k e^2
    e* = DW / (4 q k)
    EXP(x) = max_e U(e; x) - U(x; x)   (exploitability of the symmetric profile)

Two INDEPENDENT best-response paths are provided and cross-checked:
  (i)  ``br_analytic``  - piecewise-linear FOC roots + corner/kink/region checks
  (ii) ``br_grid``      - fine grid argmax + parabolic sub-grid refinement

No Monte Carlo anywhere in this module.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# Canonical one-stage Set 1 parameters (config/one_stage_two_players.py:8-17).
W_H: float = 6.5
W_L: float = 3.0
DW: float = W_H - W_L          # 3.5
K: float = 0.00055
BOUNDS: Tuple[float, float] = (0.0, 100.0)


# ------------------------------- shock law ---------------------------------
def f_xi(x: np.ndarray, q: float) -> np.ndarray:
    """Triangular density of xi = eps_i - eps_j on [-2q, 2q].

    Args:
        x: Evaluation point(s).
        q: Noise half-width of the per-player U(-q, q).

    Returns:
        Density values (0 outside [-2q, 2q]).
    """
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) <= 2.0 * q, (2.0 * q - np.abs(x)) / (4.0 * q * q), 0.0)


def F_xi(x: np.ndarray, q: float) -> np.ndarray:
    """Triangular CDF of xi on [-2q, 2q].

    F(x) = (2q+x)^2/(8q^2) on [-2q, 0]; 1 - (2q-x)^2/(8q^2) on [0, 2q].

    Args:
        x: Evaluation point(s).
        q: Noise half-width.

    Returns:
        CDF values in [0, 1] (same shape as ``x``).
    """
    x = np.asarray(x, dtype=float)
    out = np.empty(np.shape(x), dtype=float)
    lo = x <= -2.0 * q
    hi = x >= 2.0 * q
    mid_neg = (~lo) & (x < 0.0)
    mid_pos = (~hi) & (x >= 0.0)
    out[lo] = 0.0
    out[hi] = 1.0
    out[mid_neg] = (x[mid_neg] + 2.0 * q) ** 2 / (8.0 * q * q)
    out[mid_pos] = 1.0 - (2.0 * q - x[mid_pos]) ** 2 / (8.0 * q * q)
    return out


# ------------------------------- payoff ------------------------------------
def U(e: np.ndarray, e_opp: float, q: float, *, w_l: float = W_L,
      dw: float = DW, k: float = K) -> np.ndarray:
    """Expected payoff of effort ``e`` against a deterministic opponent ``e_opp``.

    Args:
        e: Own effort(s).
        e_opp: Opponent's deterministic effort.
        q: Noise half-width.
        w_l: Loser prize.
        dw: Prize gap w_H - w_L.
        k: Cost coefficient in c(e) = k e^2.

    Returns:
        Expected payoff(s), same shape as ``e``.
    """
    e = np.asarray(e, dtype=float)
    return w_l + dw * (1.0 - F_xi(e_opp - e, q)) - k * e ** 2


def e_star(q: float, *, dw: float = DW, k: float = K) -> float:
    """Closed-form symmetric equilibrium effort e* = DW / (4 q k)."""
    return dw / (4.0 * q * k)


def br_slopes(q: float, *, dw: float = DW, k: float = K) -> Dict[str, float]:
    """Local BR-map slopes and the expansive-regime threshold.

    With a = DW/(4q^2): slope approaching e* from below is +a/(2k+a);
    from above it is -a/(2k-a). |slope above| > 1 iff q < sqrt(DW/(4k)).

    Args:
        q: Noise half-width.
        dw: Prize gap.
        k: Cost coefficient.

    Returns:
        Dict with ``a``, ``slope_below``, ``slope_above``, ``q_expansive``.
    """
    a = dw / (4.0 * q * q)
    return {
        "a": a,
        "slope_below": a / (2.0 * k + a),
        "slope_above": -a / (2.0 * k - a) if (2.0 * k - a) != 0 else float("-inf"),
        "q_expansive": float(np.sqrt(dw / (4.0 * k))),
    }


# --------------------------- best response (i) ------------------------------
def br_analytic(x: float, q: float, *, bounds: Tuple[float, float] = BOUNDS,
                w_l: float = W_L, dw: float = DW, k: float = K) -> float:
    """Analytic best response to a deterministic opponent effort ``x``.

    Builds the candidate set from the two piecewise-linear FOC roots, the kink
    at e = x, the region boundaries e = x +/- 2q (where f_xi hits zero), and the
    effort-bound corners; returns the payoff-maximizing candidate.

    Args:
        x: Opponent's deterministic effort.
        q: Noise half-width.
        bounds: Effort bounds (low, high).
        w_l: Loser prize.
        dw: Prize gap.
        k: Cost coefficient.

    Returns:
        Best-response effort in ``bounds``.
    """
    lo, hi = bounds
    a = dw / (4.0 * q * q)
    cands: List[float] = [lo, hi, float(np.clip(x, lo, hi))]

    # Branch e > x : f_xi(x-e) = (2q - (e-x))/(4q^2)  =>  e = a(2q+x)/(2k+a)
    e_above = a * (2.0 * q + x) / (2.0 * k + a)
    if e_above >= x and (e_above - x) <= 2.0 * q:
        cands.append(float(np.clip(e_above, lo, hi)))

    # Branch e < x : f_xi(x-e) = (2q - (x-e))/(4q^2)  =>  e = a(2q-x)/(2k-a)
    denom = 2.0 * k - a
    if denom > 0.0:
        e_below = a * (2.0 * q - x) / denom
        if e_below <= x and (x - e_below) <= 2.0 * q:
            cands.append(float(np.clip(e_below, lo, hi)))

    # Region boundaries: beyond |x-e| = 2q the win prob saturates (f_xi = 0).
    for c in (x - 2.0 * q, x + 2.0 * q):
        if lo <= c <= hi:
            cands.append(float(c))

    cand = np.unique(np.array(cands, dtype=float))
    vals = U(cand, x, q, w_l=w_l, dw=dw, k=k)
    return float(cand[int(np.argmax(vals))])


# --------------------------- best response (ii) -----------------------------
def _parabolic_vertex(xg: np.ndarray, y: np.ndarray, i: int) -> float:
    """Parabolic sub-grid refinement of an interior grid argmax at index ``i``."""
    if i <= 0 or i >= xg.size - 1:
        return float(xg[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom >= 0.0 or denom == 0.0:      # not concave -> keep grid argmax
        return float(xg[i])
    h = float(xg[i + 1] - xg[i])
    shift = 0.5 * h * (y0 - y2) / denom
    return float(np.clip(xg[i] + shift, xg[i - 1], xg[i + 1]))


def br_grid(x: float, q: float, *, n: int = 20001,
            bounds: Tuple[float, float] = BOUNDS, w_l: float = W_L,
            dw: float = DW, k: float = K, refine: bool = True) -> float:
    """Grid best response with parabolic sub-grid refinement (independent path).

    Args:
        x: Opponent's deterministic effort.
        q: Noise half-width.
        n: Effort-grid points over ``bounds``.
        bounds: Effort bounds.
        w_l: Loser prize.
        dw: Prize gap.
        k: Cost coefficient.
        refine: Apply parabolic sub-grid refinement at the argmax.

    Returns:
        Best-response effort.
    """
    lo, hi = bounds
    xg = np.linspace(lo, hi, n)
    y = U(xg, x, q, w_l=w_l, dw=dw, k=k)
    i = int(np.argmax(y))
    if not refine:
        return float(xg[i])
    e_ref = _parabolic_vertex(xg, y, i)
    # Keep the refinement only if it does not lose payoff (guards the kink).
    return e_ref if U(e_ref, x, q, w_l=w_l, dw=dw, k=k) >= y[i] else float(xg[i])


# ------------------------------ exploitability ------------------------------
def exp_det(x: float, q: float, *, n: int = 20001,
            bounds: Tuple[float, float] = BOUNDS) -> Dict[str, float]:
    """Deterministic exploitability of the symmetric profile (x, x).

    EXP(x) = max_e U(e; x) - U(x; x), computed on BOTH BR paths.

    Args:
        x: Symmetric profile effort.
        q: Noise half-width.
        n: Fine effort-grid size for the grid path.
        bounds: Effort bounds.

    Returns:
        Dict with ``exp`` (max of the two paths), per-path EXP, the two BR
        efforts, and their disagreement.
    """
    u_self = float(U(np.array([x]), x, q)[0])
    b_a = br_analytic(x, q, bounds=bounds)
    b_g = br_grid(x, q, n=n, bounds=bounds)
    e_a = float(U(np.array([b_a]), x, q)[0]) - u_self
    e_g = float(U(np.array([b_g]), x, q)[0]) - u_self
    return {
        "exp": max(e_a, e_g, 0.0),
        "exp_analytic": e_a,
        "exp_grid": e_g,
        "br_analytic": b_a,
        "br_grid": b_g,
        "br_disagreement": abs(b_a - b_g),
        "u_self": u_self,
    }


def exp_ucb(x: float, q: float, *, n_fine: int = 20001, n_coarse: int = 5001,
            bounds: Tuple[float, float] = BOUNDS) -> Dict[str, float]:
    """EXP with a deterministic discretization margin (two-stage convention).

    EXP_UCB = EXP_fine + |EXP_fine - EXP_coarse| on the grid path; the analytic
    path is exact and reported alongside as the reference.

    Args:
        x: Symmetric profile effort.
        q: Noise half-width.
        n_fine: Fine effort grid.
        n_coarse: Coarse effort grid.
        bounds: Effort bounds.

    Returns:
        Dict with fine/coarse/UCB EXP and the analytic EXP.
    """
    u_self = float(U(np.array([x]), x, q)[0])
    bf = br_grid(x, q, n=n_fine, bounds=bounds)
    bc = br_grid(x, q, n=n_coarse, bounds=bounds)
    ef = float(U(np.array([bf]), x, q)[0]) - u_self
    ec = float(U(np.array([bc]), x, q)[0]) - u_self
    ba = br_analytic(x, q, bounds=bounds)
    ea = float(U(np.array([ba]), x, q)[0]) - u_self
    return {
        "exp_fine": ef,
        "exp_coarse": ec,
        "exp_ucb": ef + abs(ef - ec),
        "exp_analytic": ea,
        "br_fine": bf,
        "br_analytic": ba,
    }


# --------------------------------- tests ------------------------------------
def self_test(verbose: bool = True) -> int:
    """Referee unit tests. Returns the number of failures (0 = all pass)."""
    fails = 0

    def chk(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        if not ok:
            fails += 1
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    for q in (35.0, 55.0):
        # F_xi against numeric integration of f_xi
        grid = np.linspace(-2 * q, 2 * q, 200001)
        cdf_num = np.cumsum(f_xi(grid, q)) * (grid[1] - grid[0])
        cdf_num -= cdf_num[0]
        err_cdf = float(np.max(np.abs(cdf_num - F_xi(grid, q))))
        chk(f"q{q:g} F_xi == integral f_xi", err_cdf < 1e-4, f"max|dF|={err_cdf:.2e}")
        chk(f"q{q:g} F_xi(0)=0.5", abs(float(F_xi(np.array(0.0), q)) - 0.5) < 1e-12)
        chk(f"q{q:g} F_xi(+/-2q) endpoints",
            abs(float(F_xi(np.array(-2 * q), q))) < 1e-12
            and abs(float(F_xi(np.array(2 * q), q)) - 1.0) < 1e-12)
        chk(f"q{q:g} f_xi integrates to 1",
            abs(float(np.trapezoid(f_xi(grid, q), grid)) - 1.0) < 1e-8)

        es = e_star(q)
        # BR(e*) == e*
        ba, bg = br_analytic(es, q), br_grid(es, q)
        chk(f"q{q:g} BR_analytic(e*)==e*", abs(ba - es) < 1e-6, f"|d|={abs(ba-es):.2e}")
        chk(f"q{q:g} BR_grid(e*)==e*", abs(bg - es) < 5e-3, f"|d|={abs(bg-es):.2e}")
        # EXP(e*) ~ 0 (numerical floor)
        r = exp_det(es, q)
        chk(f"q{q:g} EXP(e*)~0 floor", r["exp"] < 1e-9, f"floor={r['exp']:.3e}")
        # EXP >= 0 and both paths agree on probes
        worst = 0.0
        for x in (0.0, 5.0, 20.0, 30.0, 40.0, 45.0, 50.0, 70.0, 100.0):
            rr = exp_det(x, q)
            worst = max(worst, rr["br_disagreement"])
            if rr["exp"] < -1e-12:
                chk(f"q{q:g} EXP>=0 at x={x}", False, f"{rr['exp']:.3e}")
        chk(f"q{q:g} BR paths agree on probes", worst < 1e-2, f"max|BR_a-BR_g|={worst:.2e}")
        # Slope structure
        s = br_slopes(q)
        h = 1e-3
        num_below = (br_analytic(es - h, q) - br_analytic(es - 2 * h, q)) / h
        num_above = (br_analytic(es + 2 * h, q) - br_analytic(es + h, q)) / h
        chk(f"q{q:g} BR slope below == +a/(2k+a)", abs(num_below - s["slope_below"]) < 1e-3,
            f"num={num_below:.4f} ana={s['slope_below']:.4f}")
        chk(f"q{q:g} BR slope above == -a/(2k-a)", abs(num_above - s["slope_above"]) < 1e-3,
            f"num={num_above:.4f} ana={s['slope_above']:.4f}")
    s35 = br_slopes(35.0)
    chk("q_expansive == sqrt(DW/4k) == 39.886", abs(s35["q_expansive"] - 39.8862) < 1e-3,
        f"{s35['q_expansive']:.4f}")
    return fails


if __name__ == "__main__":
    print("=" * 78)
    print("Deterministic one-stage referee — unit tests")
    print(f"  w_H={W_H} w_L={W_L} DW={DW} k={K} bounds={BOUNDS}")
    print("=" * 78)
    n = self_test()
    print("=" * 78)
    print(f"{'ALL TESTS PASS' if n == 0 else f'{n} TEST(S) FAILED'}")
    raise SystemExit(1 if n else 0)
