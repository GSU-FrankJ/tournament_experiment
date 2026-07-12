"""Two-stage recovery metrics and the pre-registered T=2 acceptance gate.

Implements the plan's Experiment-1 recovery metrics (RE_1, MAE_2, RMSE_2,
RPE_2, PL_2) against the closed-form benchmark, plus the pre-registered
acceptance gate. Under the owner's Claim-B framing the PRIMARY gate is the
independent exploitability CERTIFICATE (reproducible across seeds); the
recovery metrics are reported diagnostics whose target bands may legitimately
be missed by the exploration-smoothed candidate (see
``docs/tasks/multistage-tel-ppo/preregistration_T2.md``).

Cost convention: repo standard c(e) = k e^2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from utils.theory_multistage import (
    eq_utility_two_stage,
    f_xi,
    g1_two_stage,
    g2_two_stage,
)

# Effort function: (stage t, gaps d[array]) -> efforts[array].
EffortFn = Callable[[int, np.ndarray], np.ndarray]


def onpath_expected_stage2_effort(effort_fn: EffortFn, *, q: float, n_half: int = 48) -> float:
    """On-path expected stage-2 effort E[e_hat_2(d_2)] by deterministic quadrature.

    Under symmetric stage-1 play both players exert e_hat_1(0), so the stage-1
    drift cancels and the stage-2 gap is exactly d_2 = xi_1 ~ Triangular(-2q, 2q).
    Hence the on-path recovery target is

        E[e_hat_2(d_2)] = int_{-2q}^{2q} e_hat_2(delta) f_xi(delta) d(delta),

    which equals g1 = DW/(6qk) for the closed-form benchmark. Computed by
    Gauss-Legendre quadrature split at 0 (f_xi is piecewise linear), so it is
    exact on affine effort segments and needs no simulation. This is the
    quantity the T=2 report previously omitted (defect D1): it must be compared
    to 46.6667, NOT the probe value e_hat_2(0) which targets DW/(4qk)=70.

    Args:
        effort_fn: Vectorized learned effort function ``e_hat(t, d_array)``.
        q: Noise half-width.
        n_half: Gauss-Legendre nodes per half of ``[-2q, 2q]`` (>= 32).

    Returns:
        The on-path expected stage-2 effort (float).
    """
    x, w = np.polynomial.legendre.leggauss(n_half)
    nodes_list, weights_list = [], []
    for a, b in ((-2.0 * q, 0.0), (0.0, 2.0 * q)):
        nodes_list.append(0.5 * (b - a) * x + 0.5 * (a + b))
        weights_list.append(0.5 * (b - a) * w)
    nodes = np.concatenate(nodes_list)
    weights = np.concatenate(weights_list)
    e2 = np.asarray(effort_fn(2, nodes), dtype=float).reshape(-1)
    return float(np.sum(weights * e2 * f_xi(nodes, q)))


@dataclass
class RecoveryMetrics:
    """Two-stage recovery metrics vs the closed form (plan Experiment 1)."""

    ae_1: float          # |e_hat_1(0) - g1|
    re_1: float          # AE_1 / (1 + g1)
    mae_2: float         # mean |e_hat_2(d) - g2(d)| over |d| <= 2q
    rmse_2: float        # RMS over |d| <= 2q
    rpe_2: float         # RMSE_2 / (1 + mean g2) over |d| <= 2q
    rpe_2_core: float    # RPE_2 restricted to |d| <= q (interior core)
    pl_2: float          # U(e*_CF) - U(e_hat)  (payoff loss at root)
    pl_2_over_dw: float  # PL_2 / (w_h - w_l)
    e_hat_1_at_0: float
    g1: float
    e2_onpath_expected: float = float("nan")  # E[e_hat_2(d_2)] (target g1=46.6667)
    d_grid: List[float] = field(default_factory=list)
    e_hat_2: List[float] = field(default_factory=list)
    g2: List[float] = field(default_factory=list)


def recovery_metrics(
    effort_fn: EffortFn,
    *,
    q: float,
    w_h: float,
    w_l: float,
    k: float,
    e_bar: float = 100.0,
    v_e_root: Optional[float] = None,
    n_grid: int = 81,
) -> RecoveryMetrics:
    """Compute two-stage recovery metrics for a learned effort function.

    Args:
        effort_fn: Vectorized learned effort function ``e_hat(t, d_array)``.
        q: Noise half-width.
        w_h: Winner prize.
        w_l: Loser prize.
        k: Cost coefficient in c(e) = k e^2.
        e_bar: Effort upper bound.
        v_e_root: Learned on-path root value V_1^e_hat(0) (from the verifier);
            used for the payoff loss. If ``None``, PL_2 is set to NaN.
        n_grid: Stage-2 comparison grid points over ``[-2q, 2q]``.

    Returns:
        A :class:`RecoveryMetrics`.
    """
    g1 = g1_two_stage(q, w_h, w_l, k)
    e1 = float(np.asarray(effort_fn(1, np.array([0.0]))).reshape(-1)[0])
    ae1 = abs(e1 - g1)
    re1 = ae1 / (1.0 + g1)

    d = np.linspace(-2.0 * q, 2.0 * q, n_grid)
    e2 = np.asarray(effort_fn(2, d), dtype=float).reshape(-1)
    g2 = g2_two_stage(d, q, w_h, w_l, k, e_bar)
    err = e2 - g2
    mae2 = float(np.mean(np.abs(err)))
    rmse2 = float(np.sqrt(np.mean(err ** 2)))
    rpe2 = rmse2 / (1.0 + float(np.mean(g2)))

    core = np.abs(d) <= q
    err_c = err[core]
    g2_c = g2[core]
    rmse2_c = float(np.sqrt(np.mean(err_c ** 2)))
    rpe2_core = rmse2_c / (1.0 + float(np.mean(g2_c)))

    dw = float(w_h) - float(w_l)
    if v_e_root is None:
        pl2 = float("nan")
    else:
        pl2 = eq_utility_two_stage(q, w_h, w_l, k) - float(v_e_root)

    e2_onpath = onpath_expected_stage2_effort(effort_fn, q=q)

    return RecoveryMetrics(
        ae_1=ae1, re_1=re1, mae_2=mae2, rmse_2=rmse2, rpe_2=rpe2,
        rpe_2_core=rpe2_core, pl_2=pl2, pl_2_over_dw=pl2 / dw,
        e_hat_1_at_0=e1, g1=g1, e2_onpath_expected=e2_onpath,
        d_grid=d.tolist(), e_hat_2=e2.tolist(), g2=g2.tolist(),
    )


# ---------------------------------------------------------------------------
# Pre-registered T=2 acceptance gate
# ---------------------------------------------------------------------------

# PRIMARY (hard) gate — the Claim-B equilibrium test. Frozen 2026-07-09
# before any gated run; justification in preregistration_T2.md.
GATE_DREACH_OVER_DW = 0.03        # per-seed conservative certificate threshold
GATE_MIN_CERT_SEEDS_FRAC = 0.8    # >= 4/5 seeds must certify (reproducibility)

# SECONDARY (reported) recovery target bands — a miss is explained by
# exploration smoothing, not a pipeline failure.
TARGET_RE_1 = 0.10
TARGET_RPE_2_CORE = 0.15


@dataclass
class SeedGateInput:
    """Per-seed inputs to the gate aggregation."""

    seed: int
    dreach_over_dw: float
    exp_over_dw: float
    certified: bool
    re_1: float
    rpe_2_core: float


@dataclass
class GateVerdict:
    """Aggregated gate verdict across seeds."""

    passed: bool
    n_seeds: int
    n_certified: int
    cert_fraction: float
    dreach_mean: float
    dreach_std: float
    dreach_max: float
    exp_mean: float
    exp_std: float
    re1_mean: float
    rpe2_core_mean: float
    reasons: List[str] = field(default_factory=list)


def evaluate_gate(seeds: List[SeedGateInput]) -> GateVerdict:
    """Apply the pre-registered T=2 gate across seeds.

    PRIMARY (decides "proceed to T=3"):
      - certification reproducibility: >= GATE_MIN_CERT_SEEDS_FRAC of seeds
        have dreach_over_dw <= GATE_DREACH_OVER_DW.

    Args:
        seeds: Per-seed gate inputs.

    Returns:
        A :class:`GateVerdict`.
    """
    n = len(seeds)
    dreach = np.array([s.dreach_over_dw for s in seeds])
    exp = np.array([s.exp_over_dw for s in seeds])
    certs = np.array([s.dreach_over_dw <= GATE_DREACH_OVER_DW for s in seeds])
    n_cert = int(certs.sum())
    frac = n_cert / n if n else 0.0

    reasons: List[str] = []
    passed = frac >= GATE_MIN_CERT_SEEDS_FRAC
    if passed:
        reasons.append(
            f"PASS: {n_cert}/{n} seeds certify (dReach/DW <= {GATE_DREACH_OVER_DW}), "
            f">= {GATE_MIN_CERT_SEEDS_FRAC:.0%} required"
        )
    else:
        reasons.append(
            f"FAIL: only {n_cert}/{n} seeds certify (need >= {GATE_MIN_CERT_SEEDS_FRAC:.0%})"
        )

    re1 = np.array([s.re_1 for s in seeds])
    rpe2c = np.array([s.rpe_2_core for s in seeds])
    # Secondary diagnostics (reported, not gating)
    if float(re1.mean()) > TARGET_RE_1:
        reasons.append(f"note: mean RE_1={re1.mean():.3f} exceeds target {TARGET_RE_1} "
                       "(smoothing-explained; not gating)")
    if float(rpe2c.mean()) > TARGET_RPE_2_CORE:
        reasons.append(f"note: mean RPE_2_core={rpe2c.mean():.3f} exceeds target "
                       f"{TARGET_RPE_2_CORE} (smoothing-explained; not gating)")

    return GateVerdict(
        passed=bool(passed),
        n_seeds=n,
        n_certified=n_cert,
        cert_fraction=frac,
        dreach_mean=float(dreach.mean()),
        dreach_std=float(dreach.std(ddof=0)),
        dreach_max=float(dreach.max()),
        exp_mean=float(exp.mean()),
        exp_std=float(exp.std(ddof=0)),
        re1_mean=float(re1.mean()),
        rpe2_core_mean=float(rpe2c.mean()),
        reasons=reasons,
    )
