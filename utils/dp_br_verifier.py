"""DP best-response verifier for the final DP-BR verification protocol (2026-09-07).

Implements the numerical specification in
``MultiStage/Discussion/090726 final DP-BR verification(1).md`` (merge notes and
the revised BR-reachable set):

  - Per-stage state domains D_1 = {0}, D_t = [-(t-1)B, (t-1)B], B = e_max - e_min + 2q,
    each represented by a symmetric grid containing 0 and both endpoints.
  - Gauss-Legendre quadrature on [-2q, 0] and [0, 2q] weighted by the triangular
    shock density; used ONLY for continuation expectations.
  - Linear interpolation of non-terminal continuation values; a landing point
    outside D_{t+1} is a domain error (no constant-tail extrapolation).
  - Terminal expectation in closed form: w_l + DW * F_xi(y).
  - Best-response search over the effort grid, the candidate's own mean action
    (evaluated directly, never rounded) and valid concave parabola vertices whose
    Q is actually recomputed; ties resolved toward the smaller effort.
  - One-step deviation gap Delta_t(d) against the MEAN continuation, whose
    candidate set additionally contains the selected dynamic BR action, so that
    Delta_t(d) >= Q^mean_t(d, a^BR_t(d)) - V^mean_t(d) holds by construction.
  - BR-reachable region propagated as continuous intervals
    [d + a^BR - e_hat(-d) - 2q, d + a^BR - e_hat(-d) + 2q] intersected with D_{t+1};
    every grid point covered by an interval is reachable. Separately, the
    GL + linear-mass forward pmf (support p > 0, no truncation) is recorded for
    the performance-difference-lemma residual and as a diagnostic.
  - dReach = sum_t max_{d in R_t} Delta_t(d); EXP_root = V_1^BR(0) - V_1^mean(0);
    dFull = sum_t max_{d in G_t} Delta_t(d) and Delta_max_all are exploratory
    diagnostics only.

All arithmetic is float64. The policy callable is responsible for reproducing
the network's float32 behaviour; this module never imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from utils.theory_multistage import F_xi, f_xi

MeanPolicy = Callable[[int, np.ndarray], np.ndarray]
BetaFn = Callable[[int, np.ndarray], Tuple[np.ndarray, np.ndarray]]

SRC_GRID, SRC_VERTEX, SRC_MEAN, SRC_BR = 0, 1, 2, 3
SOURCE_NAMES = {SRC_GRID: "grid", SRC_VERTEX: "vertex", SRC_MEAN: "mean_action", SRC_BR: "br_action"}


class DomainError(RuntimeError):
    """A continuation landing point fell outside the next-stage feasible domain."""


@dataclass(frozen=True)
class VerifierConfig:
    """Numerical resolution of one verifier tier."""

    name: str
    state_step: float
    effort_step: float
    gl_half: int
    grid_tol: float = 1e-9
    mass_tol: float = 1e-10
    pdl_tol_over_dw: float = 1e-10


DEV_CONFIG = VerifierConfig("development", state_step=4.0, effort_step=1.0, gl_half=16)
FINAL_CONFIG = VerifierConfig("final", state_step=2.0, effort_step=0.5, gl_half=32)


@dataclass
class StageResult:
    """Per-stage arrays on the stage grid (all float64 / bool)."""

    d_grid: np.ndarray
    e_hat: np.ndarray          # candidate mean action e_hat_t(d)
    e_opp: np.ndarray          # opponent action e_hat_t(-d)
    v_br: np.ndarray           # V_t^BR(d)
    a_br: np.ndarray           # selected dynamic BR action
    a_br_source: np.ndarray    # candidate type of a_br (SRC_*)
    v_mean: np.ndarray         # V_t^mean(d) = Q_t^mean(d, e_hat_t(d))
    delta: np.ndarray          # one-step deviation gap Delta_t(d)
    a_dev: np.ndarray          # argmax action of the one-step deviation search
    a_dev_source: np.ndarray
    q_mean_at_abr: np.ndarray  # Q_t^mean(d, a_br)
    reach: np.ndarray          # BR-reachable mask (interval definition)
    pmf: np.ndarray            # GL + linear-mass forward pmf under the BR chain
    q_br_grid: np.ndarray      # Q_t^BR(d, e) on (state, effort) grid
    q_mean_grid: np.ndarray    # Q_t^mean(d, e) on (state, effort) grid
    std_norm: Optional[np.ndarray] = None   # std(e)/(e_max-e_min) if beta_fn given
    alpha: Optional[np.ndarray] = None
    beta: Optional[np.ndarray] = None


@dataclass
class VerifierResult:
    """Aggregate verifier output."""

    config: VerifierConfig
    T: int
    dw: float
    e_grid: np.ndarray
    gl_nodes: np.ndarray
    gl_weights: np.ndarray
    gl_weight_sum: float
    stages: Dict[int, StageResult]
    exp_root: float
    v_br_root: float
    v_mean_root: float
    dreach: float
    reach_delta_max: Dict[int, float]
    reach_argmax_d: Dict[int, float]
    reach_count: Dict[int, int]
    dreach_pmf_support: float
    pmf_support_delta_max: Dict[int, float]
    pmf_support_count: Dict[int, int]
    dfull: float
    full_delta_max: Dict[int, float]
    full_argmax_d: Dict[int, float]
    delta_max_all: float
    pdl_sum: float
    pdl_residual: float
    pmf_mass: Dict[int, float]
    pmf_mass_err_max: float
    reach_interval_diag: Dict[int, Dict[str, float]]
    valid: bool
    invalid_reasons: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, object]:
        """JSON-serializable scalar summary."""
        dw = self.dw
        return {
            "config": self.config.name,
            "state_step": self.config.state_step,
            "effort_step": self.config.effort_step,
            "gl_half": self.config.gl_half,
            "gl_weight_sum": self.gl_weight_sum,
            "valid": bool(self.valid),
            "invalid_reasons": list(self.invalid_reasons),
            "exp_root": self.exp_root,
            "exp_root_over_dw": self.exp_root / dw,
            "v_br_root": self.v_br_root,
            "v_mean_root": self.v_mean_root,
            "dreach": self.dreach,
            "dreach_over_dw": self.dreach / dw,
            "reach_delta_max": {str(t): v for t, v in self.reach_delta_max.items()},
            "reach_argmax_d": {str(t): v for t, v in self.reach_argmax_d.items()},
            "reach_count": {str(t): v for t, v in self.reach_count.items()},
            "dreach_pmf_support": self.dreach_pmf_support,
            "dreach_pmf_support_over_dw": self.dreach_pmf_support / dw,
            "pmf_support_delta_max": {str(t): v for t, v in self.pmf_support_delta_max.items()},
            "pmf_support_count": {str(t): v for t, v in self.pmf_support_count.items()},
            "dfull": self.dfull,
            "dfull_over_dw": self.dfull / dw,
            "full_delta_max": {str(t): v for t, v in self.full_delta_max.items()},
            "full_argmax_d": {str(t): v for t, v in self.full_argmax_d.items()},
            "delta_max_all": self.delta_max_all,
            "delta_max_all_over_dw": self.delta_max_all / dw,
            "pdl_sum": self.pdl_sum,
            "pdl_residual": self.pdl_residual,
            "pdl_residual_over_dw": self.pdl_residual / dw,
            "pmf_mass": {str(t): v for t, v in self.pmf_mass.items()},
            "pmf_mass_err_max": self.pmf_mass_err_max,
            "reach_interval_diag": {str(t): v for t, v in self.reach_interval_diag.items()},
            "grid_points": {str(t): int(s.d_grid.size) for t, s in self.stages.items()},
            "effort_grid_points": int(self.e_grid.size),
            "a_br_source_counts": {
                str(t): {SOURCE_NAMES[c]: int((s.a_br_source == c).sum()) for c in SOURCE_NAMES}
                for t, s in self.stages.items()
            },
            "a_dev_source_counts": {
                str(t): {SOURCE_NAMES[c]: int((s.a_dev_source == c).sum()) for c in SOURCE_NAMES}
                for t, s in self.stages.items()
            },
        }


# ---------------------------------------------------------------------------
# Grids and quadrature
# ---------------------------------------------------------------------------

def stage_domain_half(t: int, B: float) -> float:
    """Half-width (t-1)B of the feasible score-gap domain D_t."""
    return (t - 1) * B


def stage_grid(t: int, B: float, step: float) -> np.ndarray:
    """Symmetric state grid on D_t containing 0 and both endpoints.

    When ``step`` does not divide (t-1)B the number of half-intervals is rounded
    up so the realized spacing never exceeds ``step``.

    Args:
        t: Stage (1-indexed).
        B: Per-stage maximal gap change, e_max - e_min + 2q.
        step: Target spacing.

    Returns:
        1-D float64 grid (``[0.0]`` for t = 1).
    """
    if t <= 1:
        return np.zeros(1)
    half = stage_domain_half(t, B)
    n_half = int(np.ceil(half / step - 1e-9))
    return np.linspace(-half, half, 2 * n_half + 1)


def effort_grid(e_min: float, e_max: float, step: float) -> np.ndarray:
    """Uniform effort grid from e_min to e_max (both included), spacing <= step."""
    n = int(np.ceil((e_max - e_min) / step - 1e-9))
    return np.linspace(e_min, e_max, n + 1)


def gl_shock_rule(q: float, n_half: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights on [-2q,0] U [0,2q] times the triangular density.

    The density is linear on each half, so the weights sum to exactly 1 (up to
    rounding) for any n_half >= 1. Nodes are strictly interior to (-2q, 2q).

    Args:
        q: Noise half-width.
        n_half: Nodes per half interval.

    Returns:
        ``(nodes, weights)`` float64 arrays of length ``2 * n_half``.
    """
    x, w = np.polynomial.legendre.leggauss(n_half)
    nodes = np.concatenate([-q + q * x, q + q * x])
    base = np.concatenate([q * w, q * w])
    weights = base * f_xi(nodes, q)
    return nodes, weights


# ---------------------------------------------------------------------------
# Candidate selection helpers
# ---------------------------------------------------------------------------

def _select(efforts: np.ndarray, values: np.ndarray, sources: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise max with ties broken toward the smaller effort.

    Args:
        efforts: Candidate efforts, shape (M, C).
        values: Candidate values, shape (M, C); non-finite entries are ignored.
        sources: Candidate type codes, shape (M, C) or (C,).

    Returns:
        ``(value_max, effort_choice, source_choice)`` each of shape (M,).
    """
    vals = np.where(np.isfinite(values), values, -np.inf)
    vmax = vals.max(axis=1)
    tie = vals == vmax[:, None]
    eff_choice = np.where(tie, efforts, np.inf).min(axis=1)
    pick = tie & (efforts == eff_choice[:, None])
    idx = np.argmax(pick, axis=1)
    src = np.broadcast_to(sources, efforts.shape)
    return vmax, eff_choice, src[np.arange(efforts.shape[0]), idx]


def _vertex_candidates(e_grid: np.ndarray, q_vals: np.ndarray, e_min: float, e_max: float
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Parabola vertices at interior grid local maxima with a strictly concave triple.

    Args:
        e_grid: Effort grid (K,), uniform.
        q_vals: Objective on the grid, shape (M, K).
        e_min: Lower effort bound.
        e_max: Upper effort bound.

    Returns:
        ``(vertex_efforts, valid_mask)`` of shape (M, K-2). Invalid entries hold
        the interior grid effort (so they may be evaluated harmlessly) and are
        masked out by ``valid_mask``.
    """
    h = float(e_grid[1] - e_grid[0])
    y0 = q_vals[:, :-2]
    y1 = q_vals[:, 1:-1]
    y2 = q_vals[:, 2:]
    denom = y0 - 2.0 * y1 + y2
    valid = (y1 >= y0) & (y1 >= y2) & (denom < 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(valid, 0.5 * (y0 - y2) / denom, 0.0)
    off = np.clip(off, -1.0, 1.0)
    e_v = e_grid[None, 1:-1] + off * h
    e_v = np.clip(e_v, e_min, e_max)
    return e_v, valid


# ---------------------------------------------------------------------------
# Main verifier
# ---------------------------------------------------------------------------

def verify(
    policy: MeanPolicy,
    *,
    w_h: float,
    w_l: float,
    k: float,
    q: float,
    T: int,
    e_min: float = 0.0,
    e_max: float = 100.0,
    cfg: VerifierConfig = FINAL_CONFIG,
    beta_fn: Optional[BetaFn] = None,
    opponent_policy: Optional[MeanPolicy] = None,
) -> VerifierResult:
    """Run the DP-BR verifier for one deviating player against a fixed opponent.

    With ``opponent_policy=None`` both players use ``policy`` (shared symmetric
    policy; the opponent is queried at -d). With an explicit ``opponent_policy``
    (asymmetric role policies, e.g. MC-BR) the deviating player's own candidate
    is ``policy`` and the opponent plays ``opponent_policy(t, -d)``, i.e. it reads
    its own signed state. Nothing is averaged across roles.

    Args:
        policy: ``policy(t, d_array) -> effort_array`` (float64 output expected).
        w_h: Winner prize.
        w_l: Loser prize.
        k: Cost coefficient, c(e) = k e^2.
        q: Per-player noise half-width.
        T: Horizon.
        e_min: Lower effort bound.
        e_max: Upper effort bound.
        cfg: Verifier tier configuration.
        beta_fn: Optional ``(t, d_array) -> (alpha, beta)`` for std diagnostics.
        opponent_policy: Optional opponent effort function (own-signed-state input).

    Returns:
        A :class:`VerifierResult`.

    Raises:
        DomainError: If a continuation landing point leaves D_{t+1}.
    """
    w_h, w_l, k, q = float(w_h), float(w_l), float(k), float(q)
    dw = w_h - w_l
    B = (e_max - e_min) + 2.0 * q
    grids = {t: stage_grid(t, B, cfg.state_step) for t in range(1, T + 1)}
    E = effort_grid(e_min, e_max, cfg.effort_step)
    nodes, weights = gl_shock_rule(q, cfg.gl_half)
    gl_sum = float(weights.sum())
    cost_grid = k * E ** 2
    tol = cfg.grid_tol

    opp_policy = policy if opponent_policy is None else opponent_policy

    def _pol(t: int, d: np.ndarray, fn: MeanPolicy = policy) -> np.ndarray:
        out = np.asarray(fn(t, np.asarray(d, dtype=float)), dtype=float).reshape(-1)
        if out.shape != (d.size,):
            raise ValueError(f"policy returned shape {out.shape} for {d.size} states")
        if not np.all(np.isfinite(out)):
            raise FloatingPointError(f"policy returned non-finite effort at stage {t}")
        if out.min() < e_min - 1e-9 or out.max() > e_max + 1e-9:
            raise ValueError(f"policy effort outside [{e_min}, {e_max}] at stage {t}")
        return out

    def _continuation(t: int, v_next: Optional[np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
        if t == T:
            def W(y: np.ndarray) -> np.ndarray:
                return w_l + dw * F_xi(y, q)
            return W
        g = grids[t + 1]
        lo, hi = float(g[0]), float(g[-1])
        eps = tol * max(1.0, hi)

        def W(y: np.ndarray) -> np.ndarray:
            y = np.asarray(y, dtype=float)
            acc = np.zeros_like(y)
            for x, w in zip(nodes, weights):
                land = y + x
                mn, mx = float(land.min()), float(land.max())
                if mn < lo - eps or mx > hi + eps:
                    raise DomainError(
                        f"stage {t}->{t + 1}: landing [{mn:.6g}, {mx:.6g}] outside "
                        f"D_{t + 1}=[{lo:g}, {hi:g}]")
                acc += w * np.interp(land, g, v_next)
            return acc
        return W

    stages: Dict[int, StageResult] = {}
    v_br_next: Optional[np.ndarray] = None
    v_mean_next: Optional[np.ndarray] = None

    for t in range(T, 0, -1):
        G = grids[t]
        M = G.size
        e_hat = _pol(t, G)
        e_opp = _pol(t, -G, opp_policy)
        W_br = _continuation(t, v_br_next)
        W_mean = _continuation(t, v_mean_next)
        base = G - e_opp                                             # (M,)
        landing_grid = base[:, None] + E[None, :]                    # (M, K)

        # --- dynamic best response --------------------------------------
        q_br_grid = -cost_grid[None, :] + W_br(landing_grid)
        e_v, v_valid = _vertex_candidates(E, q_br_grid, e_min, e_max)
        q_br_v = -k * e_v ** 2 + W_br(base[:, None] + e_v)
        q_br_v = np.where(v_valid, q_br_v, -np.inf)
        q_br_mean = -k * e_hat ** 2 + W_br(base + e_hat)
        eff_c = np.concatenate([np.broadcast_to(E, (M, E.size)), e_v, e_hat[:, None]], axis=1)
        val_c = np.concatenate([q_br_grid, q_br_v, q_br_mean[:, None]], axis=1)
        src_c = np.concatenate([np.full(E.size, SRC_GRID), np.full(E.size - 2, SRC_VERTEX),
                                np.array([SRC_MEAN])])
        v_br, a_br, a_br_src = _select(eff_c, val_c, src_c)

        # --- mean chain and one-step deviation gap -----------------------
        q_mean_grid = -cost_grid[None, :] + W_mean(landing_grid)
        v_mean = -k * e_hat ** 2 + W_mean(base + e_hat)
        e_vm, vm_valid = _vertex_candidates(E, q_mean_grid, e_min, e_max)
        q_mean_v = -k * e_vm ** 2 + W_mean(base[:, None] + e_vm)
        q_mean_v = np.where(vm_valid, q_mean_v, -np.inf)
        q_mean_at_abr = -k * a_br ** 2 + W_mean(base + a_br)
        eff_d = np.concatenate([np.broadcast_to(E, (M, E.size)), e_vm, e_hat[:, None],
                                a_br[:, None]], axis=1)
        val_d = np.concatenate([q_mean_grid, q_mean_v, v_mean[:, None], q_mean_at_abr[:, None]],
                               axis=1)
        src_d = np.concatenate([np.full(E.size, SRC_GRID), np.full(E.size - 2, SRC_VERTEX),
                                np.array([SRC_MEAN, SRC_BR])])
        vdev, a_dev, a_dev_src = _select(eff_d, val_d, src_d)
        delta = vdev - v_mean
        if delta.min() < 0.0:
            raise FloatingPointError("negative one-step deviation gap despite mean candidate")

        std_norm = alpha = beta = None
        if beta_fn is not None:
            a_, b_ = beta_fn(t, G)
            alpha = np.asarray(a_, dtype=float).reshape(-1)
            beta = np.asarray(b_, dtype=float).reshape(-1)
            std_norm = beta_std_norm(alpha, beta)

        stages[t] = StageResult(
            d_grid=G, e_hat=e_hat, e_opp=e_opp, v_br=v_br, a_br=a_br, a_br_source=a_br_src,
            v_mean=v_mean, delta=delta, a_dev=a_dev, a_dev_source=a_dev_src,
            q_mean_at_abr=q_mean_at_abr, reach=np.zeros(M, dtype=bool), pmf=np.zeros(M),
            q_br_grid=q_br_grid, q_mean_grid=q_mean_grid,
            std_norm=std_norm, alpha=alpha, beta=beta,
        )
        v_br_next, v_mean_next = v_br, v_mean

    # --- forward: interval reachability and GL + linear-mass pmf -------------
    stages[1].reach[:] = True
    stages[1].pmf[:] = 1.0
    interval_diag: Dict[int, Dict[str, float]] = {}
    for t in range(1, T):
        s = stages[t]
        Gn = grids[t + 1]
        lo_dom, hi_dom = float(Gn[0]), float(Gn[-1])
        drift = s.a_br - s.e_opp
        src = np.nonzero(s.reach)[0]
        lo = np.maximum(s.d_grid[src] + drift[src] - 2.0 * q, lo_dom)
        hi = np.minimum(s.d_grid[src] + drift[src] + 2.0 * q, hi_dom)
        i0 = np.searchsorted(Gn, lo - tol, side="left")
        i1 = np.searchsorted(Gn, hi + tol, side="right")
        cover = np.zeros(Gn.size + 1, dtype=int)
        np.add.at(cover, i0, 1)
        np.add.at(cover, i1, -1)
        mask = np.cumsum(cover)[:-1] > 0
        stages[t + 1].reach[:] = mask
        # endpoint diagnostics: how far interval endpoints sit from covered grid points
        on_grid_lo = np.abs(Gn[np.clip(i0, 0, Gn.size - 1)] - lo) <= tol
        on_grid_hi = np.abs(Gn[np.clip(i1 - 1, 0, Gn.size - 1)] - hi) <= tol
        gap_lo = Gn[np.clip(i0, 0, Gn.size - 1)] - lo
        gap_hi = hi - Gn[np.clip(i1 - 1, 0, Gn.size - 1)]
        interval_diag[t + 1] = {
            "n_intervals": int(src.size),
            "n_endpoints_on_grid": int(on_grid_lo.sum() + on_grid_hi.sum()),
            "max_uncovered_margin": float(max(gap_lo.max(initial=0.0), gap_hi.max(initial=0.0))),
            "union_lo": float(lo.min()),
            "union_hi": float(hi.max()),
            "n_grid_covered": int(mask.sum()),
        }
        # pmf with the same GL kernel and the linear mass split used by interpolation
        p = s.pmf
        nxt = np.zeros(Gn.size)
        eps = tol * max(1.0, hi_dom)
        for x, w in zip(nodes, weights):
            land = s.d_grid + drift + x
            if land.min() < lo_dom - eps or land.max() > hi_dom + eps:
                raise DomainError(f"forward pmf stage {t}->{t + 1}: landing outside D_{t + 1}")
            idx = np.clip(np.searchsorted(Gn, land, side="right"), 1, Gn.size - 1)
            left = Gn[idx - 1]
            right = Gn[idx]
            frac = np.clip((land - left) / (right - left), 0.0, 1.0)
            np.add.at(nxt, idx - 1, w * p * (1.0 - frac))
            np.add.at(nxt, idx, w * p * frac)
        stages[t + 1].pmf[:] = nxt

    # --- aggregates -----------------------------------------------------------
    reach_dm, reach_arg, reach_cnt = {}, {}, {}
    sup_dm, sup_cnt = {}, {}
    full_dm, full_arg = {}, {}
    pmf_mass = {}
    pdl_sum = 0.0
    for t, s in stages.items():
        r = s.reach
        reach_cnt[t] = int(r.sum())
        if reach_cnt[t] == 0:
            reach_dm[t], reach_arg[t] = float("nan"), float("nan")
        else:
            j = int(np.argmax(np.where(r, s.delta, -np.inf)))
            reach_dm[t], reach_arg[t] = float(s.delta[j]), float(s.d_grid[j])
        sp = s.pmf > 0.0
        sup_cnt[t] = int(sp.sum())
        sup_dm[t] = float(s.delta[sp].max()) if sup_cnt[t] else float("nan")
        j = int(np.argmax(s.delta))
        full_dm[t], full_arg[t] = float(s.delta[j]), float(s.d_grid[j])
        pmf_mass[t] = float(s.pmf.sum())
        pdl_sum += float(np.sum(s.pmf * (s.q_mean_at_abr - s.v_mean)))

    exp_root = float(stages[1].v_br[0] - stages[1].v_mean[0])
    dreach = float(sum(reach_dm.values()))
    dreach_sup = float(sum(sup_dm.values()))
    dfull = float(sum(full_dm.values()))
    delta_max_all = float(max(full_dm.values()))
    pdl_residual = exp_root - pdl_sum
    mass_err = float(max(abs(m - 1.0) for m in pmf_mass.values()))

    reasons: List[str] = []
    if not np.isfinite([exp_root, dreach, dfull, pdl_residual]).all():
        reasons.append("non-finite aggregate")
    if any(not np.isfinite(v) for v in reach_dm.values()):
        reasons.append("empty reachable set at some stage")
    if mass_err > cfg.mass_tol:
        reasons.append(f"pmf mass error {mass_err:.3e} > {cfg.mass_tol:g}")
    if abs(pdl_residual) / dw > cfg.pdl_tol_over_dw:
        reasons.append(f"PDL residual/DW {pdl_residual / dw:.3e} > {cfg.pdl_tol_over_dw:g}")
    if abs(gl_sum - 1.0) > 1e-12:
        reasons.append(f"GL weight sum {gl_sum!r} != 1")

    return VerifierResult(
        config=cfg, T=T, dw=dw, e_grid=E, gl_nodes=nodes, gl_weights=weights,
        gl_weight_sum=gl_sum, stages=stages, exp_root=exp_root,
        v_br_root=float(stages[1].v_br[0]), v_mean_root=float(stages[1].v_mean[0]),
        dreach=dreach, reach_delta_max=reach_dm, reach_argmax_d=reach_arg, reach_count=reach_cnt,
        dreach_pmf_support=dreach_sup, pmf_support_delta_max=sup_dm, pmf_support_count=sup_cnt,
        dfull=dfull, full_delta_max=full_dm, full_argmax_d=full_arg, delta_max_all=delta_max_all,
        pdl_sum=pdl_sum, pdl_residual=pdl_residual, pmf_mass=pmf_mass, pmf_mass_err_max=mass_err,
        reach_interval_diag=interval_diag, valid=not reasons, invalid_reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Concentration diagnostics
# ---------------------------------------------------------------------------

def beta_std_norm(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Normalized Beta standard deviation sqrt(ab / ((a+b)^2 (a+b+1)))."""
    a = np.asarray(alpha, dtype=float)
    b = np.asarray(beta, dtype=float)
    s = a + b
    return np.sqrt(a * b / (s * s * (s + 1.0)))


def concentration_stats(beta_fn: BetaFn, points: Dict[int, np.ndarray], e_range: float = 100.0
                        ) -> Dict[str, object]:
    """Max normalized conditional std over the given (stage -> d array) points.

    Args:
        beta_fn: ``(t, d_array) -> (alpha, beta)``.
        points: Stage -> evaluation gaps.
        e_range: Effort range (std normalization uses the Beta on [0,1], so the
            normalized std is std(e)/e_range).

    Returns:
        Dict with the max value, its location and the Beta parameters there;
        ``valid`` is False when any non-finite value occurs.
    """
    best = None
    finite = True
    n = 0
    for t, d in points.items():
        d = np.asarray(d, dtype=float).reshape(-1)
        if d.size == 0:
            continue
        a, b = beta_fn(t, d)
        a = np.asarray(a, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)
        sd = beta_std_norm(a, b)
        n += d.size
        if not (np.isfinite(sd).all() and np.isfinite(a).all() and np.isfinite(b).all()):
            finite = False
        j = int(np.nanargmax(sd)) if np.isfinite(sd).any() else 0
        cand = (float(sd[j]), t, float(d[j]), float(a[j]), float(b[j]))
        if best is None or cand[0] > best[0]:
            best = cand
    if best is None:
        return {"valid": False, "n_points": 0}
    sd, t, d, a, b = best
    return {
        "valid": bool(finite),
        "n_points": int(n),
        "max_std_norm": sd,
        "max_std_effort": sd * e_range,
        "stage": int(t),
        "d": d,
        "alpha": a,
        "beta": b,
        "conc": a + b,
        "mean_norm": a / (a + b),
        "mean_effort": e_range * a / (a + b),
    }


def stage_result_arrays(res: VerifierResult, prefix: str) -> Dict[str, np.ndarray]:
    """Flatten per-stage arrays for ``np.savez`` (keys ``{prefix}_t{t}_{field}``)."""
    out: Dict[str, np.ndarray] = {
        f"{prefix}_e_grid": res.e_grid,
        f"{prefix}_gl_nodes": res.gl_nodes,
        f"{prefix}_gl_weights": res.gl_weights,
    }
    for t, s in res.stages.items():
        for name in ("d_grid", "e_hat", "e_opp", "v_br", "a_br", "a_br_source", "v_mean", "delta",
                     "a_dev", "a_dev_source", "q_mean_at_abr", "reach", "pmf", "q_br_grid",
                     "q_mean_grid", "std_norm", "alpha", "beta"):
            arr = getattr(s, name)
            if arr is not None:
                out[f"{prefix}_t{t}_{name}"] = np.asarray(arr)
    return out
