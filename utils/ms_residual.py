"""First- and second-order residual metrics of a stage candidate (MS-R1, D3).

Everything here is computed from the arrays of one ``utils.dp_br_verifier.verify`` call (development
tier in the pipeline). The closed-form equilibrium does not enter: the rule and the sampler read a
:class:`StageDiag` and nothing else.

For the stage ``t`` of a game with horizon ``T`` the verifier gives, per node ``d`` of the stage grid,
the one-step deviation gain ``delta_t(d)``, the policy mean ``e_hat_t(d)`` and the one-step
best-response effort ``a_dev_t(d)`` (argmax of the one-step deviation search against the policy's own
continuation). With ``dw = w_h - w_l``, ``thr = 2 q (T - t + 1)`` and ``s_t = a_dev_t(0)``:

    Delta_t   = full_delta_max[t] / dw                                (second order)
    r_t(d)    = |e_hat_t(d) - a_dev_t(d)|                              (first order)
    R_t       = max { r_t(d) / s_t : |d| < thr }                       (non-tail region)
    R_t^tail  = mean { e_hat_t(d) : |d| >= thr } / s_t                 (void when there is no tail bin)
    rho_b     = max { r_t(d) / s_t : d in non-tail bin b }             (per-bin map, nodes by gap_bin_index)

The per-bin maps are averaged across checks by :func:`ema_update`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from envs.curriculum_env import GameSpec, StartSampler, gap_bin_index
from utils.dp_br_verifier import VerifierResult

NODE_TOL = 1e-9


@dataclass
class StageDiag:
    """D3 quantities of one check at stage ``stage`` (NaN where undefined)."""

    stage: int
    valid: bool
    delta_over_dw: float = float("nan")
    s: float = float("nan")
    R: float = float("nan")
    R_tail: float = float("nan")
    C: float = float("nan")
    tail_term: bool = False
    rho_bins: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_nodes_nontail: int = 0
    n_nodes_tail: int = 0
    argmax_d: float = float("nan")

    def scalars(self) -> Dict[str, object]:
        """Flat scalar form for the CSV rows."""
        return {"valid": bool(self.valid), "Delta": self.delta_over_dw, "s": self.s, "R": self.R,
                "R_tail": self.R_tail, "C": self.C, "tail_term": bool(self.tail_term),
                "R_argmax_d": self.argmax_d}


def invalid_diag(stage: int, n_bins: int = 0) -> StageDiag:
    """Placeholder for a check whose verifier call failed (never eligible)."""
    return StageDiag(stage=stage, valid=False, rho_bins=np.full(n_bins, np.nan))


def stage_diag(res: VerifierResult, spec: GameSpec, stage: int, conc: Optional[Dict[str, object]],
               bin_width: float = 10.0) -> StageDiag:
    """D3 quantities of the candidate behind ``res`` at ``stage``.

    Args:
        res: Verifier result of the composite candidate (live stage ``stage``, frozen stages above).
        spec: Game specification.
        stage: Stage whose phase this check belongs to.
        conc: ``concentration_stats`` of the stage-``stage`` development grid (``valid`` and
            ``max_std_norm``), or ``None``.
        bin_width: Exploring-start bin width of D_stage.

    Returns:
        The :class:`StageDiag`; ``valid`` is False when the verifier result is invalid, the
        concentration is not finite or the best-response scale ``s_t`` is not positive.
    """
    sr = res.stages[stage]
    d = np.asarray(sr.d_grid, dtype=float)
    e_hat = np.asarray(sr.e_hat, dtype=float)
    a_dev = np.asarray(sr.a_dev, dtype=float)
    j0 = int(np.argmin(np.abs(d)))
    if abs(d[j0]) > NODE_TOL:
        raise ValueError(f"stage {stage} grid has no zero node")
    s = float(a_dev[j0])
    thr = 2.0 * float(spec.q) * (spec.T - stage + 1)
    nontail = np.abs(d) < thr - NODE_TOL
    tail_nodes = ~nontail
    r = np.abs(e_hat - a_dev)
    n_bins = 0
    tail_bins = np.zeros(0, dtype=bool)
    if stage >= 2:
        sampler = StartSampler(spec, bin_width)
        n_bins = sampler.n_bins(stage)
        tail_bins = sampler.tail_mask(stage)
    has_tail = bool(stage >= 2 and tail_bins.any() and tail_nodes.any())
    conc_ok = bool(conc is not None and conc.get("valid"))
    c = float(conc["max_std_norm"]) if conc_ok else float("nan")
    ok = bool(res.valid and conc_ok and np.isfinite(s) and s > 0.0 and np.isfinite(c))
    delta = float(res.full_delta_max[stage]) / float(res.dw)
    if not (np.isfinite(s) and s > 0.0):
        return StageDiag(stage=stage, valid=False, delta_over_dw=delta, s=s, C=c,
                         rho_bins=np.full(n_bins, np.nan), tail_term=has_tail)
    ratio = r / s
    jm = int(np.argmax(np.where(nontail, ratio, -np.inf)))
    big_r = float(ratio[jm])
    r_tail = float(e_hat[tail_nodes].mean() / s) if has_tail else float("nan")
    rho = np.full(n_bins, np.nan)
    if stage >= 2:
        idx, nb = gap_bin_index(spec, stage, d, bin_width)
        acc = np.full(nb, -np.inf)
        use = nontail & ~tail_bins[idx]
        np.maximum.at(acc, idx[use], ratio[use])
        rho = np.where(np.isfinite(acc), acc, np.nan)
    return StageDiag(stage=stage, valid=ok, delta_over_dw=delta, s=s, R=big_r, R_tail=r_tail, C=c,
                     tail_term=has_tail, rho_bins=rho, n_nodes_nontail=int(nontail.sum()),
                     n_nodes_tail=int(tail_nodes.sum()), argmax_d=float(d[jm]))


def ema_update(prev: Optional[np.ndarray], new: np.ndarray, beta: float) -> np.ndarray:
    """``rho_bar <- beta * rho_bar + (1 - beta) * rho``, initialised at the first check (NaN stays NaN).

    Args:
        prev: Previous EMA (``None`` before the first check).
        new: Per-bin map of the current check.
        beta: EMA weight of the past, in [0, 1).
    """
    new = np.asarray(new, dtype=float)
    if prev is None:
        return new.copy()
    prev = np.asarray(prev, dtype=float)
    return np.where(np.isnan(prev), new, np.where(np.isnan(new), prev, beta * prev + (1.0 - beta) * new))
