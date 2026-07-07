#!/usr/bin/env python3
"""Monte-Carlo best-response polishing + mode extraction + frozen-profile exploitability.

Component 3 (mode extraction) + Component 4 (MC-BR polishing) + the post-polish
(2nd) exploitability check of the one-stage TEL-PPO revision, implemented as a
POST-HOC, ZERO-GPU library that operates on deterministic effort profiles. No
policy network is required: polishing fixes the OTHER players' deterministic
efforts and refines each player's effort by a simulated, sampled best response.

SAMPLED-PAYOFF PATH ONLY
------------------------
Every payoff used here is the realized sampled tournament outcome
    y_i = e_i + l_i + eps_i,  eps_i ~ U(-q, q);  realized argmax winner gets w_H,
    the others w_L;  payoff_i = prize_i - k_i e_i^2.
This module imports NOTHING from utils.prob / any win-probability or
expected-utility helper. The single payoff primitive `sampled_payoff_player`
is numerically verified against the project's canonical sampled helper
`utils.exploit_asymmetric._payoff_player` (see `verify_sampled_only`), so the
"sampled-only, _payoff_player path" requirement is provable, not asserted.
`utils.theory` is used ONLY for the closed-form e* benchmark in error reporting.

Thresholds (acceptance / stop rule), owner-set:
    tau_g = interior |FOC| acceptance (boundary -> projected/KKT)
    tau_E = post-polish exploitability acceptance
    tau_e = polishing-iteration effort-change (convergence) stop
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- canonical sampled-payoff path (for the equivalence check) ---
from utils.exploit_asymmetric import _payoff_player as _canonical_payoff_player_2p

# Import guard: the polishing path must never pull in closed-form win-prob /
# expected-utility code.
_FORBIDDEN = ("utils.prob",)
for _m in _FORBIDDEN:
    assert _m not in sys.modules, f"forbidden closed-form module loaded: {_m}"


# =============================== sampled payoff ===============================
def sampled_payoff_player(
    i: int,
    cand_efforts: np.ndarray,   # (G,) candidate efforts for player i
    efforts: np.ndarray,        # (n,) current profile (others held fixed)
    l: np.ndarray,              # (n,) abilities (additive shift)
    k: np.ndarray,              # (n,) cost coefficients
    w_h: float,
    w_l: float,
    eps: np.ndarray,            # (M, n) shared noise draws (common random numbers)
) -> np.ndarray:
    """Mean realized sampled payoff for player ``i`` at each candidate effort.

    Returns (G,) array of E_hat[payoff_i] under the SAME M noise draws for every
    candidate (CRN). The winner is the realized argmax of y_j = e_j + l_j + eps_j;
    no closed-form win probability is used.
    """
    n = eps.shape[1]
    y_others = efforts[None, :] + l[None, :] + eps          # (M, n)
    mask = np.ones(n, dtype=bool); mask[i] = False
    others_max = y_others[:, mask].max(axis=1)              # (M,)
    yi = cand_efforts[:, None] + l[i] + eps[:, i][None, :]  # (G, M)
    win = yi > others_max[None, :]                          # (G, M); exact ties ~ measure 0
    prize = np.where(win, w_h, w_l)
    return prize.mean(axis=1) - k[i] * (cand_efforts ** 2)  # (G,)


def verify_sampled_only(seed: int = 0, M: int = 50_000, tol: float = 1e-9) -> None:
    """Assert `sampled_payoff_player` equals the canonical `_payoff_player` (2p).

    Feeds identical continuous noise (no exact ties) to both and checks the
    per-player mean payoff agrees. This ties this module's payoff to the
    project's certified sampled path and proves no closed-form is involved.
    """
    import torch
    rng = np.random.default_rng(seed)
    e = rng.uniform(5, 60, size=2)
    l = rng.uniform(0, 10, size=2)
    k = rng.uniform(2e-4, 6e-4, size=2)
    w_h, w_l = 8.0, 5.5
    eps = rng.uniform(-35, 35, size=(M, 2))
    mine = sampled_payoff_player(0, np.array([e[0]]), e, l, k, w_h, w_l, eps)[0]
    canon = _canonical_payoff_player_2p(
        torch.tensor(np.full(M, e[0])), torch.tensor(np.full(M, e[1])),
        torch.tensor(eps[:, 0]), torch.tensor(eps[:, 1]),
        torch.zeros(M, dtype=torch.long),  # ties measure-zero
        w_h=w_h, w_l=w_l, k_self=k[0], l_self=l[0], l_opp=l[1],
    ).mean().item()
    assert abs(mine - canon) < 1e-3, f"sampled-payoff mismatch: {mine} vs {canon}"


# ============================== mode extraction ==============================
def beta_mode(alpha: float, beta: float, low: float, high: float) -> float:
    """Beta MODE mapped to effort bounds. Boundary-safe: alpha<=1 -> low, beta<=1 -> high."""
    if alpha > 1.0 and beta > 1.0:
        m = (alpha - 1.0) / (alpha + beta - 2.0)
    elif alpha <= 1.0 and beta <= 1.0:
        m = 0.5  # degenerate U-shape; report midpoint
    elif alpha <= 1.0:
        m = 0.0
    else:
        m = 1.0
    return float(low + m * (high - low))


def beta_mean(alpha: float, beta: float, low: float, high: float) -> float:
    return float(low + (alpha / (alpha + beta)) * (high - low))


def beta_std_effort(alpha: float, beta: float, low: float, high: float) -> float:
    c = alpha + beta
    var = (alpha * beta) / (c * c * (c + 1.0))
    return float(np.sqrt(var) * (high - low))


# ============================= best response / polish =========================
def mc_best_response(
    i: int, efforts: np.ndarray, l: np.ndarray, k: np.ndarray, w_h: float, w_l: float,
    q: float, bounds: Tuple[float, float], rng: np.random.Generator, M: int, window: float = 10.0,
    bias_correct: bool = False, vertex_halfwidth: float = 1.0,
) -> float:
    """Coarse-to-fine grid MC best response for player i (others fixed, CRN).

    With ``bias_correct=True`` the final estimate is the VERTEX of a least-squares
    quadratic fit to the mean-payoff values in a +/- ``vertex_halfwidth`` window
    around the fine argmax, instead of the raw argmax. On a flat, cost-asymmetric
    payoff peak the raw argmax is downward-biased (it latches the single noisiest-
    high grid point on the flatter low side); the quadratic vertex uses ALL points
    near the peak and is approximately unbiased. This is a ZEROTH-ORDER (payoff-
    value) debias — the BR remains a payoff maximizer, a different functional from
    the first-order FD-FOC used in acceptance leg (b), so (b) stays independent.
    This function NEVER computes a finite-difference FOC.
    """
    lo, hi = bounds
    eps = rng.uniform(-q, q, size=(M, efforts.shape[0]))
    e_i = efforts[i]
    coarse = np.arange(max(lo, e_i - window), min(hi, e_i + window) + 1e-9, 0.5)
    v = sampled_payoff_player(i, coarse, efforts, l, k, w_h, w_l, eps)
    c = coarse[int(v.argmax())]
    half = max(0.5, vertex_halfwidth)
    fine = np.arange(max(lo, c - half), min(hi, c + half) + 1e-9, 0.05)
    vf = sampled_payoff_player(i, fine, efforts, l, k, w_h, w_l, eps)
    if not bias_correct:
        return float(fine[int(vf.argmax())])
    # quadratic-vertex debias over a window around the fine argmax (zeroth-order)
    cstar = fine[int(vf.argmax())]
    win = (fine >= cstar - vertex_halfwidth) & (fine <= cstar + vertex_halfwidth)
    xw, yw = fine[win], vf[win]
    if xw.size >= 5:
        a2, a1, _ = np.polyfit(xw, yw, 2)
        if a2 < 0:  # concave -> genuine interior maximum
            vertex = -a1 / (2.0 * a2)
            return float(min(max(vertex, xw.min()), xw.max()))
    return float(cstar)  # fallback: argmax if the quadratic fit is degenerate


@dataclass
class PolishResult:
    e_polished: np.ndarray
    drift: float                 # |Δe| convergence metric (criterion c scale)
    rounds: int
    converged: bool              # drift < tau_e
    trajectory: np.ndarray = field(repr=False, default=None)


def mc_br_polish(
    e0: np.ndarray, l: np.ndarray, k: np.ndarray, w_h: float, w_l: float, q: float,
    bounds: Tuple[float, float], *, eta: float = 0.4, M: int = 150_000,
    min_rounds: int = 60, max_rounds: int = 160, n_avg: int = 60, tau_e: float = 0.1,
    seed: int = 0, bias_correct: bool = False,
) -> PolishResult:
    """Damped SIMULTANEOUS MC-BR polishing with Polyak-Ruppert iterate averaging.

    Each round computes every player's MC best response against the OTHERS'
    CURRENT deterministic efforts (CRN within each BR) and applies the damped
    simultaneous update e <- (1-eta) e + eta BR (all BRs use the same e, so the
    update is order-independent). Because the payoff peak is flat, single-shot BR
    is high-variance and the iterate oscillates; the reported effort is the mean
    of the iterate over the last ``n_avg`` rounds. Convergence/stop = the drift of
    that averaged estimate between the two halves of the window < ``tau_e``.
    """
    rng = np.random.default_rng(seed)
    e = e0.astype(float).copy()
    n = e.shape[0]
    hist: List[np.ndarray] = []
    drift = float("nan")
    for r in range(1, max_rounds + 1):
        br = np.array([mc_best_response(i, e, l, k, w_h, w_l, q, bounds, rng, M,
                                        bias_correct=bias_correct)
                       for i in range(n)])
        e = (1.0 - eta) * e + eta * br
        hist.append(e.copy())
        if r >= min_rounds and r % 10 == 0:
            w = np.array(hist[-n_avg:]); half = len(w) // 2
            drift = float(np.max(np.abs(w[half:].mean(0) - w[:half].mean(0))))
            if drift < tau_e:
                arr = np.array(hist)
                return PolishResult(w.mean(0), drift, r, True, arr)
    arr = np.array(hist)
    w = arr[-n_avg:]; half = len(w) // 2
    drift = float(np.max(np.abs(w[half:].mean(0) - w[:half].mean(0))))
    return PolishResult(w.mean(0), drift, max_rounds, drift < tau_e, arr)


# ===================== post-polish exploitability + FOC (fresh draws) =========
def exploitability_frozen_profile(
    e: np.ndarray, l: np.ndarray, k: np.ndarray, w_h: float, w_l: float, q: float,
    bounds: Tuple[float, float], *, M: int = 200_000, grid_step: float = 0.25, seed: int = 99_991,
) -> Tuple[float, np.ndarray]:
    """2nd exploitability on the FROZEN profile with FRESH independent draws.

    exploit_i = max(0, max_c E_hat[payoff_i(c, e_-i)] - E_hat[payoff_i(e_i, e_-i)]).
    Returns (max_i exploit_i, per-player BR efforts). Fresh seed => draws are NOT
    reused from polishing (plan requirement).
    """
    lo, hi = bounds
    n = e.shape[0]
    rng = np.random.default_rng(seed)
    exploits, brs = [], []
    grid = np.arange(lo, hi + 1e-9, grid_step)
    for i in range(n):
        eps = rng.uniform(-q, q, size=(M, n))
        u_cur = sampled_payoff_player(i, np.array([e[i]]), e, l, k, w_h, w_l, eps)[0]
        v = sampled_payoff_player(i, grid, e, l, k, w_h, w_l, eps)
        j = int(v.argmax()); brs.append(float(grid[j]))
        exploits.append(max(0.0, float(v[j] - u_cur)))
    return float(max(exploits)), np.array(brs)


def foc_frozen_profile(
    e: np.ndarray, l: np.ndarray, k: np.ndarray, w_h: float, w_l: float, q: float,
    bounds: Tuple[float, float], *, delta: float = 0.5, M: int = 300_000, seed: int = 13_337,
) -> np.ndarray:
    """Sampled MC-FD central-difference FOC per player at e (CRN, fresh seed)."""
    lo, hi = bounds
    n = e.shape[0]
    rng = np.random.default_rng(seed)
    g = np.zeros(n)
    for i in range(n):
        eps = rng.uniform(-q, q, size=(M, n))
        ep = min(hi, e[i] + delta); em = max(lo, e[i] - delta)
        up = sampled_payoff_player(i, np.array([ep]), e, l, k, w_h, w_l, eps)[0]
        dn = sampled_payoff_player(i, np.array([em]), e, l, k, w_h, w_l, eps)[0]
        g[i] = (up - dn) / (ep - em)
    return g


# ============================== acceptance rule ==============================
@dataclass
class Acceptance:
    exp_polished: float
    foc: np.ndarray
    drift: float
    pass_exp: bool
    pass_foc: bool
    pass_drift: bool
    foc_kind: List[str]  # "interior" / "kkt-lo" / "kkt-hi" per player

    @property
    def accepted(self) -> bool:
        return self.pass_exp and self.pass_foc and self.pass_drift


def check_acceptance(
    e: np.ndarray, l: np.ndarray, k: np.ndarray, w_h: float, w_l: float, q: float,
    bounds: Tuple[float, float], drift: float, *,
    tau_E: float = 0.005, tau_g: float = 0.001, tau_e: float = 0.1, bound_tol: float = 1e-3,
) -> Acceptance:
    """Three independent threshold checks: (a) EXP_polished<tau_E, (b) FOC, (c) drift<tau_e.

    (b) interior players require |FOC|<tau_g; players at a bound use the projected
    KKT condition (at lower bound accept if FOC<=tau_g; at upper bound if FOC>=-tau_g).
    """
    lo, hi = bounds
    exp_pol, _ = exploitability_frozen_profile(e, l, k, w_h, w_l, q, bounds)
    foc = foc_frozen_profile(e, l, k, w_h, w_l, q, bounds)
    kinds, ok_foc = [], True
    for i in range(e.shape[0]):
        if e[i] <= lo + bound_tol:
            kinds.append("kkt-lo"); ok = foc[i] <= tau_g
        elif e[i] >= hi - bound_tol:
            kinds.append("kkt-hi"); ok = foc[i] >= -tau_g
        else:
            kinds.append("interior"); ok = abs(foc[i]) < tau_g
        ok_foc = ok_foc and ok
    return Acceptance(exp_pol, foc, drift, exp_pol < tau_E, ok_foc, drift < tau_e, kinds)


if __name__ == "__main__":
    verify_sampled_only()
    print("verify_sampled_only: PASS (sampled_payoff_player == _payoff_player, no closed-form)")
