"""v2 verifier metrics: full-domain dynamic deviation, invariants, recovery, on/off-path split.

This module wraps :func:`utils.dp_br_verifier.verify` without changing it. Every quantity
below is an aggregation of the arrays that ``verify`` already computes on the full
per-stage grids D_t, plus a forward pmf of the candidate's OWN chain (``verify`` only
records the pmf of the deviator's BR chain).

Arrays used (per stage t, on the grid of D_t; see ``utils/dp_br_verifier.py``):
    v_br   V_t^BR(d): the deviator re-optimizes at t and at every later stage
           (backward induction, lines 382-402), the opponent plays e_hat_t(-d) and the
           continuation is the deviator's own V_{t+1}^BR.
    v_mean V_t^e_hat(d) = Q_t^mean(d, e_hat_t(d)) (line 406).
    delta  one-step gap Delta_t(d) against the e_hat continuation (lines 405-418).
    G_t(d) = v_br - v_mean is the full dynamic deviation gain.

The closed-form benchmark (``utils.theory_multistage``) is used ONLY by
:func:`recovery_metrics` and never by anything that touches training.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from envs.curriculum_env import GameSpec
from utils.dp_br_verifier import (
    BetaFn,
    DomainError,
    MeanPolicy,
    VerifierConfig,
    VerifierResult,
    stage_result_arrays,
    verify,
)
from utils.theory_multistage import F_xi, g1_two_stage, g2_two_stage


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

def symmetric_grid(half: float, step: float) -> np.ndarray:
    """Symmetric grid on [-half, half] with spacing <= step; 0 must be an exact node.

    Same construction as ``run.run_final_dp_br.dense_grid`` and ``stage_grid``.

    Args:
        half: Half-width of the interval.
        step: Target spacing.

    Returns:
        1-D float64 grid with an odd number of points and grid[mid] == 0.0 exactly.

    Raises:
        ValueError: If the midpoint is not exactly 0.0.
    """
    n_half = int(np.ceil(half / step - 1e-9))
    g = np.linspace(-half, half, 2 * n_half + 1)
    if g[n_half] != 0.0:
        raise ValueError(f"grid midpoint {g[n_half]!r} is not exactly 0")
    return g


def zero_index(grid: np.ndarray) -> int:
    """Index of the exact 0.0 node of a grid (raises if absent)."""
    hits = np.nonzero(grid == 0.0)[0]
    if hits.size != 1:
        raise ValueError("grid has no exact 0.0 node")
    return int(hits[0])


# ---------------------------------------------------------------------------
# Candidate forward pmf and on/off-path split
# ---------------------------------------------------------------------------

def candidate_pmf(res: VerifierResult, q: float) -> Dict[int, np.ndarray]:
    """Forward state pmf of the candidate's own chain from the root.

    Both players follow e_hat (the opponent at -d). The kernel is exactly the one used
    by ``verify`` for its BR-chain pmf (GL nodes of the shock difference, linear mass
    split onto the two neighbouring grid nodes, lines 470-484), with the drift
    e_hat_t(d) - e_hat_t(-d) instead of a_BR_t(d) - e_hat_t(-d).

    Args:
        res: Verifier result for the candidate.
        q: Noise half-width (only for the domain message).

    Returns:
        Stage -> pmf on that stage's grid (stage 1 = [1.0]).

    Raises:
        DomainError: If a landing point leaves D_{t+1}.
    """
    tol = res.config.grid_tol
    pmf: Dict[int, np.ndarray] = {1: np.ones(res.stages[1].d_grid.size)}
    for t in range(1, res.T):
        s = res.stages[t]
        Gn = res.stages[t + 1].d_grid
        lo_dom, hi_dom = float(Gn[0]), float(Gn[-1])
        eps = tol * max(1.0, hi_dom)
        drift = s.e_hat - s.e_opp
        p = pmf[t]
        nxt = np.zeros(Gn.size)
        for x, w in zip(res.gl_nodes, res.gl_weights):
            land = s.d_grid + drift + x
            if land.min() < lo_dom - eps or land.max() > hi_dom + eps:
                raise DomainError(f"candidate pmf stage {t}->{t + 1}: landing outside D_{t + 1} (q={q})")
            idx = np.clip(np.searchsorted(Gn, land, side="right"), 1, Gn.size - 1)
            left = Gn[idx - 1]
            right = Gn[idx]
            frac = np.clip((land - left) / (right - left), 0.0, 1.0)
            np.add.at(nxt, idx - 1, w * p * (1.0 - frac))
            np.add.at(nxt, idx, w * p * frac)
        pmf[t + 1] = nxt
    return pmf


def onpath_mask(d_grid: np.ndarray, drift: float, q: float) -> np.ndarray:
    """On-path set {d in D_t : |d - drift| < 2q} (open interval; nodes at exactly 2q are off-path).

    For T=2 under the candidate's own chain from the root, d_2 = drift + xi_1 with
    drift = e_hat_1(0) - e_hat_1(-0), so this is the continuous support of the stage-2 law.
    """
    return np.abs(np.asarray(d_grid, dtype=float) - float(drift)) < 2.0 * float(q)


def cell_masses(d_grid: np.ndarray, drift: float, q: float) -> Tuple[np.ndarray, float]:
    """Exact probability of each grid cell under d = drift + xi, xi ~ Triangular(-2q, 2q).

    Cell boundaries are the midpoints between adjacent nodes; the outermost boundaries are the
    domain edges d_grid[0], d_grid[-1]. mass_i = F_xi(b_{i+1} - drift) - F_xi(b_i - drift).

    Returns:
        ``(normalized_masses, captured_total)``; the total is the raw sum before normalization.
    """
    g = np.asarray(d_grid, dtype=float)
    b = np.concatenate([[g[0]], 0.5 * (g[1:] + g[:-1]), [g[-1]]])
    m = np.diff(F_xi(b - float(drift), q))
    tot = float(m.sum())
    return m / tot, tot


def onoff_split(values: np.ndarray, on: np.ndarray, weights: np.ndarray,
                d_grid: np.ndarray) -> Dict[str, float]:
    """Split a per-state quantity into on-path and off-path parts.

    Args:
        values: Quantity per grid node (e.g. Delta_2 or |stage-2 drift|).
        on: Boolean on-path mask (see :func:`onpath_mask`).
        weights: Normalized cell masses (see :func:`cell_masses`).
        d_grid: The grid.

    Returns:
        Dict with on-path max (+argmax d), unweighted mean and cell-mass-weighted mean
        (weights renormalized over the on-path cells), off-path max (+argmax d) and unweighted
        mean, node counts and the cell mass on each side. Empty parts give NaN.
    """
    v = np.asarray(values, dtype=float)
    on = np.asarray(on, dtype=bool)
    off = ~on
    out: Dict[str, float] = {"n_on": int(on.sum()), "n_off": int(off.sum()),
                             "on_mass": float(weights[on].sum()), "off_mass": float(weights[off].sum())}
    for name, m in (("on", on), ("off", off)):
        if m.any():
            j = int(np.argmax(np.where(m, v, -np.inf)))
            out[f"{name}_max"] = float(v[j])
            out[f"{name}_argmax_d"] = float(d_grid[j])
            out[f"{name}_mean_unweighted"] = float(v[m].mean())
        else:
            out[f"{name}_max"] = out[f"{name}_argmax_d"] = out[f"{name}_mean_unweighted"] = float("nan")
    out["on_mean_cellmass_weighted"] = (float(np.sum(weights[on] * v[on]) / weights[on].sum())
                                        if on.any() and weights[on].sum() > 0 else float("nan"))
    return out


# ---------------------------------------------------------------------------
# Recovery against the closed form (evaluation only)
# ---------------------------------------------------------------------------

def recovery_metrics(policy: MeanPolicy, spec: GameSpec, step: float) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Signed / normalized recovery metrics of a T=2 candidate against the closed form.

    Args:
        policy: Mean-effort function ``policy(t, d)``.
        spec: Game parameters (T must be 2).
        step: State-grid spacing on D_2 (0 is an exact node).

    Returns:
        ``(scalars, arrays)``.
    """
    if spec.T != 2:
        raise ValueError("closed-form recovery is defined for T=2 only")
    q, k = spec.q, spec.k
    g1 = g1_two_stage(q, spec.w_h, spec.w_l, k)
    D = symmetric_grid(spec.domain_half(2), step)
    z = zero_index(D)
    e1 = float(np.asarray(policy(1, np.zeros(1)), dtype=float)[0])
    e2 = np.asarray(policy(2, D), dtype=float)
    g2 = g2_two_stage(D, q, spec.w_h, spec.w_l, k, spec.e_max)
    g20 = float(g2[z])
    pos = np.abs(D) < 2.0 * q          # support of f_xi (open); g2 > 0 exactly here
    tail = ~pos                        # theoretical zero-effort region within D_2
    err_pos = e2[pos] - g2[pos]
    rmse = float(np.sqrt(np.mean(err_pos ** 2)))
    sym = np.abs(e2 - e2[::-1])
    js = int(np.argmax(sym))
    jt = int(np.argmax(np.where(tail, e2, -np.inf)))
    sc = {
        "g1": g1, "g2_at_0": g20, "e1_at_0": e1, "e2_at_0": float(e2[z]),
        "stage1_rel_err_signed": (e1 - g1) / g1,
        "stage2_peak_rel_err_signed": (float(e2[z]) - g20) / g20,
        "stage2_peak_rel_err_abs": abs(float(e2[z]) - g20) / g20,
        "stage2_rmse_pos": rmse, "stage2_rmse_pos_over_g2_0": rmse / g20,
        "stage2_tail_mean": float(e2[tail].mean()), "stage2_tail_max": float(e2[jt]),
        "stage2_tail_argmax_d": float(D[jt]),
        "stage2_tail_mean_over_g2_0": float(e2[tail].mean()) / g20,
        "stage2_tail_max_over_g2_0": float(e2[jt]) / g20,
        "stage2_sym_err_max": float(sym[js]), "stage2_sym_err_max_over_g2_0": float(sym[js]) / g20,
        "stage2_sym_err_argmax_d": float(abs(D[js])),
        "recovery_grid_step": float(step), "recovery_n_pos": int(pos.sum()), "recovery_n_tail": int(tail.sum()),
    }
    return sc, {"recovery_d_grid": D, "recovery_e2": e2, "recovery_g2": g2}


# ---------------------------------------------------------------------------
# Induced stage-1 target (analysis only; never enters training)
# ---------------------------------------------------------------------------

def stage1_Q(res: VerifierResult, k: float):
    """Q_1(0, e' | opponent e) with the candidate's V_2^e_hat continuation (verifier formula)."""
    s2 = res.stages[2]
    G, V = s2.d_grid, s2.v_mean
    nodes, weights = res.gl_nodes, res.gl_weights

    def Q(ep, e):
        ep = np.atleast_1d(np.asarray(ep, dtype=float))
        y = ep - e
        acc = np.zeros_like(y)
        for x, w in zip(nodes, weights):
            acc += w * np.interp(y + x, G, V)
        return -k * ep ** 2 + acc
    return Q


def stage1_br(Q, E: np.ndarray, e: float, e_min: float, e_max: float) -> Tuple[float, float]:
    """(argmax, max) of Q(., e): dense search on E, then bounded refinement on the adjacent cells."""
    from scipy.optimize import minimize_scalar

    h = float(E[1] - E[0])
    vals = Q(E, e)
    j = int(np.argmax(vals))
    lo, hi = max(e_min, E[j] - h), min(e_max, E[j] + h)
    r = minimize_scalar(lambda a: -float(Q(a, e)[0]), bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-12})
    if -float(r.fun) > vals[j]:
        return float(r.x), -float(r.fun)
    return float(E[j]), float(vals[j])


def induced_stage1_target(res: VerifierResult, k: float, e_min: float = 0.0, e_max: float = 100.0,
                          xtol: float = 1e-10) -> Dict[str, float]:
    """Symmetric fixed point e~1 of the root stage game given the candidate's stage-2 mapping.

    Q_1(0, e' | opponent e) = -k e'^2 + sum_x w_x interp(e' - e + x; D_2 grid, V_2^e_hat), i.e. the
    verifier's stage-1 Q^mean formula (``utils/dp_br_verifier.py:364-375, 405``) with the GL
    nodes/weights and V_2^e_hat (``v_mean`` at t=2, which does not depend on the stage-1 policy)
    of ``res``. BR(e) = argmax_{e' in [e_min, e_max]} Q_1: dense search on the verifier's effort
    grid, then bounded scalar refinement on the neighbouring cells. The fixed point solves
    h(e) = BR(e) - e = 0: h is scanned on the effort grid (bracketing) and the bracket is refined
    with Brent's method. The opponent's own action is NOT a BR candidate (it would create a band
    of spurious grid fixed points).

    Returns:
        e_tilde, number of sign changes of h on the grid, Brent iterations, final bracket width,
        residual |BR(e~) - e~|, and the number of BR evaluations.
    """
    from scipy.optimize import brentq

    E = res.e_grid
    n_eval = [0]
    Q = stage1_Q(res, k)

    def br(e):
        n_eval[0] += 1
        return stage1_br(Q, E, e, e_min, e_max)[0]

    hv = np.array([br(e) - e for e in E])
    sc = np.nonzero(np.sign(hv[:-1]) * np.sign(hv[1:]) <= 0)[0]
    out: Dict[str, float] = {"induced_n_sign_changes": int(sc.size)}
    if sc.size == 0:
        out.update(induced_e1=float("nan"), induced_brent_iters=0, induced_final_bracket=float("nan"),
                   induced_residual=float("nan"), induced_br_evals=n_eval[0])
        return out
    j = int(sc[0])
    a, b = float(E[j]), float(E[j + 1])
    if hv[j] == 0.0:
        root, it = a, 0
    else:
        root, info = brentq(lambda e: br(e) - e, a, b, xtol=xtol, full_output=True)
        it = info.iterations
    out.update(induced_e1=float(root), induced_brent_iters=int(it), induced_final_bracket=float(xtol),
               induced_bracket_grid=f"[{a:g},{b:g}]", induced_residual=abs(br(root) - root),
               induced_br_evals=n_eval[0])
    return out


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

@dataclass
class V2Eval:
    """Output of :func:`evaluate`."""

    res: VerifierResult
    G: Dict[int, np.ndarray]
    pmf_cand: Dict[int, np.ndarray]
    scalars: Dict[str, object]
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)


def _invariants(res: VerifierResult, G: Dict[int, np.ndarray]) -> Dict[str, object]:
    """Invariant residuals (raw, never clamped) in units of DeltaW with locations.

    For an inequality lhs <= rhs, ``*_excess_over_dw`` = (lhs - rhs)/DeltaW; a positive
    value is a violation. For an equality, ``*_absdiff_over_dw`` = max |lhs - rhs|/DeltaW.
    """
    dw = res.dw
    T = res.T
    out: Dict[str, object] = {}
    # 1. G_T == Delta_T on the whole terminal grid
    s = res.stages[T]
    diff = np.abs(G[T] - s.delta)
    j = int(np.argmax(diff))
    out["inv_GT_eq_DeltaT_absdiff_over_dw"] = float(diff[j] / dw)
    out["inv_GT_eq_DeltaT_at_d"] = float(s.d_grid[j])
    # 2a. pointwise Delta_t(d) <= G_t(d)
    best = (-np.inf, None, None)
    for t, st in res.stages.items():
        ex = st.delta - G[t]
        i = int(np.argmax(ex))
        if ex[i] > best[0]:
            best = (float(ex[i]), t, float(st.d_grid[i]))
    out["inv_Delta_le_G_pointwise_excess_over_dw"] = best[0] / dw
    out["inv_Delta_le_G_pointwise_at_t"] = best[1]
    out["inv_Delta_le_G_pointwise_at_d"] = best[2]
    # 2b. aggregate Delta_max_all <= Gmax_full (the requested relation)
    gmax = max(float(G[t].max()) for t in G)
    out["inv_Deltamax_le_Gmax_excess_over_dw"] = (res.delta_max_all - gmax) / dw
    # 3. Gmax_full <= dFull
    out["inv_Gmax_le_dFull_excess_over_dw"] = (gmax - res.dfull) / dw
    # 4. EXP_root == G_1(0)
    out["inv_EXP_eq_G1_absdiff_over_dw"] = abs(res.exp_root - float(G[1][0])) / dw
    # 5. EXP_root <= dReach <= dFull
    out["inv_EXP_le_dReach_excess_over_dw"] = (res.exp_root - res.dreach) / dw
    out["inv_dReach_le_dFull_excess_over_dw"] = (res.dreach - res.dfull) / dw
    # companion: EXP_root <= dReach over the BR-pmf support (PDL bound)
    out["inv_EXP_le_dReachPMF_excess_over_dw"] = (res.exp_root - res.dreach_pmf_support) / dw
    return out


def evaluate(policy: MeanPolicy, spec: GameSpec, cfg: VerifierConfig,
             beta_fn: Optional[BetaFn] = None, recovery_step: float = 0.5) -> V2Eval:
    """Run the existing verifier and compute every v2 metric for a frozen candidate.

    Pure function: consumes no RNG and mutates nothing (the existing ``verify`` is pure).

    Args:
        policy: Deterministic candidate ``policy(t, d)`` (the Beta mean for a network).
        spec: Game parameters.
        cfg: Verifier tier.
        beta_fn: Optional ``(t, d) -> (alpha, beta)`` for distribution logging.
        recovery_step: State spacing of the closed-form recovery grid (T=2 only).

    Returns:
        A :class:`V2Eval`.
    """
    res = verify(policy, w_h=spec.w_h, w_l=spec.w_l, k=spec.k, q=spec.q, T=spec.T,
                 e_min=spec.e_min, e_max=spec.e_max, cfg=cfg, beta_fn=beta_fn)
    dw = res.dw
    for t, s in res.stages.items():
        zero_index(s.d_grid)
    G = {t: s.v_br - s.v_mean for t, s in res.stages.items()}
    best_t, best_j, gmax = None, None, -np.inf
    sc: Dict[str, object] = {"verifier_tier": cfg.name, "state_step": cfg.state_step,
                             "effort_step": cfg.effort_step, "gl_half": cfg.gl_half,
                             "valid": bool(res.valid), "invalid_reasons": ";".join(res.invalid_reasons)}
    for t in sorted(G):
        j = int(np.argmax(G[t]))
        sc[f"G_max_t{t}_over_dw"] = float(G[t][j] / dw)
        sc[f"G_argmax_d_t{t}"] = float(res.stages[t].d_grid[j])
        sc[f"Delta_max_t{t}_over_dw"] = res.full_delta_max[t] / dw
        sc[f"Delta_argmax_d_t{t}"] = res.full_argmax_d[t]
        if G[t][j] > gmax:
            gmax, best_t, best_j = float(G[t][j]), t, j
    sc.update({
        "Gmax_full_over_dw": gmax / dw, "Gmax_full_t": best_t,
        "Gmax_full_d": float(res.stages[best_t].d_grid[best_j]),
        "EXP_root_over_dw": res.exp_root / dw, "dReach_over_dw": res.dreach / dw,
        "Deltamax_all_over_dw": res.delta_max_all / dw, "dFull_over_dw": res.dfull / dw,
        "eta_T_over_dw": res.full_delta_max[spec.T] / dw,
        "pdl_residual_over_dw": res.pdl_residual / dw,
    })
    sc.update(_invariants(res, G))

    pmf_c = candidate_pmf(res, spec.q)
    T = spec.T
    sT = res.stages[T]
    # on-path rule (2026-10-01 decision): open support |d - drift| < 2q, exact cell masses
    s1 = res.stages[1]
    drift = float(s1.e_hat[0] - s1.e_opp[0])
    on_T = onpath_mask(sT.d_grid, drift, spec.q)
    w_T, w_tot = cell_masses(sT.d_grid, drift, spec.q)
    split = onoff_split(sT.delta / dw, on_T, w_T, sT.d_grid)
    sc.update({f"DeltaT_over_dw_{k_}": v for k_, v in split.items()})
    sc["stage1_drift"] = drift
    sc["stage1_drift_is_zero"] = bool(drift == 0.0)
    sc["cellmass_captured_total"] = w_tot
    sc["node_pmf_n_positive_T"] = int((pmf_c[T] > 0).sum())

    arrays = stage_result_arrays(res, "v")
    for t in G:
        s = res.stages[t]
        arrays[f"v_t{t}_G"] = G[t]
        arrays[f"v_t{t}_cand_pmf"] = pmf_c[t]   # node-based GL pmf (kept for reference only)
        if s.alpha is not None:
            arrays[f"v_t{t}_sigma_effort"] = s.std_norm * spec.e_range
            sc[f"sigma_effort_at_0_t{t}"] = float(arrays[f"v_t{t}_sigma_effort"][zero_index(s.d_grid)])
    arrays[f"v_t{T}_onpath"] = on_T
    arrays[f"v_t{T}_cell_mass"] = w_T
    if res.stages[T].alpha is not None and T == 2:
        pos = np.abs(sT.d_grid) < 2.0 * spec.q
        sc["sigma2_effort_mean_pos"] = float(arrays["v_t2_sigma_effort"][pos].mean())
    if T == 2:
        rsc, rarr = recovery_metrics(policy, spec, recovery_step)
        sc.update(rsc)
        arrays.update(rarr)
        sc["stage1_rel_err_abs"] = abs(sc["stage1_rel_err_signed"])
        ind = induced_stage1_target(res, spec.k, spec.e_min, spec.e_max)
        sc.update(ind)
        g1 = sc["g1"]
        sc["stage1_learning_err"] = sc["e1_at_0"] - ind["induced_e1"]          # e_hat_1 - e~1[e_hat_2]
        sc["stage1_inherited_err"] = ind["induced_e1"] - g1                    # e~1[e_hat_2] - e1*
        sc["stage1_learning_err_rel"] = sc["stage1_learning_err"] / g1
        sc["stage1_inherited_err_rel"] = sc["stage1_inherited_err"] / g1
    return V2Eval(res=res, G=G, pmf_cand=pmf_c, scalars=sc, arrays=arrays)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def save_npz(ev: V2Eval, path: str, **extra: np.ndarray) -> None:
    """Write every array of one checkpoint evaluation to ``path`` (.npz)."""
    np.savez(path, **ev.arrays, **extra)


def append_csv(path: str, row: Dict[str, object]) -> None:
    """Append one scalar row; the header is written once and must not change.

    Raises:
        ValueError: If an existing file has a different header.
    """
    keys = list(row.keys())
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    if exists:
        with open(path, newline="") as f:
            header = next(csv.reader(f))
        if header != keys:
            raise ValueError(f"CSV header mismatch in {path}")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if not exists:
            w.writeheader()
        w.writerow(row)


def plot_stage2(ev: V2Eval, spec: GameSpec, path: str, title: str = "") -> None:
    """Final-checkpoint plot: e_hat_2(d) vs e2*(d) and sigma_2(d) (effort units)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = ev.arrays["recovery_d_grid"]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.4]})
    axes[0].plot(D, ev.arrays["recovery_g2"], color="0.3", lw=1.5, ls="--", label=r"$e_2^*(d)$")
    axes[0].plot(D, ev.arrays["recovery_e2"], color="C0", lw=1.5, label=r"$\hat e_2(d)$ (Beta mean)")
    axes[0].set_ylabel("effort")
    axes[0].legend(frameon=False)
    if "v_t2_sigma_effort" in ev.arrays:
        axes[1].plot(ev.arrays["v_t2_d_grid"], ev.arrays["v_t2_sigma_effort"], color="C1", lw=1.2)
    axes[1].set_ylabel(r"$\sigma_2(d)$")
    axes[1].set_xlabel("d (stage-2 gap)")
    if title:
        axes[0].set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
