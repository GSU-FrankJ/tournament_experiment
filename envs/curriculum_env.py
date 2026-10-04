"""Two-player multi-stage tournament dynamics for the backward-curriculum runner.

Model (PlanB section 1; final DP-BR protocol 2026-09-07):
    y_{i,t} = e_{i,t} + eps_{i,t}, eps ~ U(-q, q) i.i.d. per player and stage
    d_{t+1} = d_t + e_i - e_j + eps_i - eps_j          (player i's signed gap)
    terminal reward R(d_{T+1}) = w_h (d > 0), w_l (d < 0), (w_h + w_l)/2 (d == 0)
    stage reward r_t = -k e_t^2 (+ R at t = T)

Observation encoding (PlanB section 3.1): s_t = (tau_t, d~_t) with
tau_t = (t-1)/(T-1), d~_1 = 0, d~_t = d_t / ((t-1) B), B = (e_max - e_min) + 2q.
The gap and rewards are kept in float64; only the network input is float32.

REPO INVARIANT: rewards are SAMPLED outcomes; no closed-form probability enters
the training reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class GameSpec:
    """Economic parameters of the tournament."""

    w_h: float
    w_l: float
    k: float
    q: float
    T: int
    e_min: float = 0.0
    e_max: float = 100.0

    @property
    def dw(self) -> float:
        """Prize spread w_h - w_l."""
        return float(self.w_h - self.w_l)

    @property
    def e_range(self) -> float:
        """Effort range e_max - e_min."""
        return float(self.e_max - self.e_min)

    @property
    def B(self) -> float:
        """Maximal one-stage gap change (e_max - e_min) + 2q."""
        return self.e_range + 2.0 * float(self.q)

    def domain_half(self, t: int) -> float:
        """Half-width (t-1)B of the feasible gap domain D_t."""
        return (t - 1) * self.B

    def terminal_reward(self, d_final: np.ndarray) -> np.ndarray:
        """Realized prize from the final signed gap (ties split the prizes)."""
        d_final = np.asarray(d_final, dtype=float)
        return np.where(d_final > 0.0, self.w_h,
                        np.where(d_final < 0.0, self.w_l, 0.5 * (self.w_h + self.w_l)))

    def encode_obs(self, t: int, d: np.ndarray) -> np.ndarray:
        """Float32 network input [(t-1)/(T-1), d/((t-1)B)] for a batch of gaps.

        Args:
            t: Stage (1-indexed).
            d: Signed gaps (float64 array).

        Returns:
            Array of shape (N, 2), dtype float32.
        """
        d = np.asarray(d, dtype=float).reshape(-1)
        tau = 0.0 if self.T <= 1 else (t - 1) / (self.T - 1)
        if t <= 1:
            dn = np.zeros_like(d)
        else:
            dn = d / ((t - 1) * self.B)
        out = np.empty((d.size, 2), dtype=np.float32)
        out[:, 0] = np.float32(tau)
        out[:, 1] = dn.astype(np.float32)
        return out

    def effort_from_action(self, a: np.ndarray) -> np.ndarray:
        """Map normalized actions in [0,1] to efforts (float64)."""
        return self.e_min + np.asarray(a, dtype=float) * self.e_range


class StartSampler:
    """State-balanced exploring starts on D_t (equal-width bins, bin then uniform)."""

    def __init__(self, spec: GameSpec, bin_width: float = 10.0):
        self.spec = spec
        self.bin_width = float(bin_width)

    def n_bins(self, t: int) -> int:
        """Number of equal-width bins covering D_t."""
        half = self.spec.domain_half(t)
        return int(np.ceil(2.0 * half / self.bin_width - 1e-9))

    def bin_edges(self, t: int) -> np.ndarray:
        """Bin edges over D_t (length n_bins + 1)."""
        half = self.spec.domain_half(t)
        return np.linspace(-half, half, self.n_bins(t) + 1)

    def balanced(self, t: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n gaps: pick a bin uniformly, then uniform within the bin."""
        edges = self.bin_edges(t)
        b = rng.integers(0, edges.size - 1, size=n)
        u = rng.random(n)
        return edges[b] + u * (edges[b + 1] - edges[b])

    def peak_set(self, t: int, half_width: float) -> np.ndarray:
        """Boolean mask over the bins of D_t: bins whose interval intersects (-half_width, half_width)."""
        edges = self.bin_edges(t)
        return (edges[:-1] < half_width) & (edges[1:] > -half_width)

    def peak_bin_probs(self, t: int, half_width: float, share: float) -> np.ndarray:
        """Bin probabilities of the peak-focused scheme: ``share`` spread uniformly over the peak set,
        ``1 - share`` uniformly over the other bins (the bin-balanced scheme gives every bin 1/n_bins)."""
        if not 0.0 < share < 1.0:
            raise ValueError(f"peak share must lie in (0, 1); got {share}")
        m = self.peak_set(t, half_width)
        n_peak, n = int(m.sum()), int(m.size)
        if n_peak == 0 or n_peak == n:
            raise ValueError(f"peak half-width {half_width} gives {n_peak} peak bins of {n} on D_{t}")
        return np.where(m, share / n_peak, (1.0 - share) / (n - n_peak))

    def peak_focused(self, t: int, n: int, rng: np.random.Generator, half_width: float,
                     share: float) -> np.ndarray:
        """Draw n gaps with a share of the episodes in the peak set (R2b mechanism 1).

        One ``rng.random(n)`` call is mapped through the inverse CDF of the bin probabilities to the
        bin; the start inside the bin is uniform from a second ``rng.random(n)`` call, as in
        :meth:`balanced`. (``balanced`` draws its bins with ``rng.integers``, so the two schemes use
        the ``start`` stream differently: the streams desynchronise at the first update.)
        """
        edges = self.bin_edges(t)
        cdf = np.cumsum(self.peak_bin_probs(t, half_width, share))
        cdf[-1] = 1.0
        b = np.minimum(np.searchsorted(cdf, rng.random(n), side="right"), edges.size - 2)
        u = rng.random(n)
        return edges[b] + u * (edges[b + 1] - edges[b])

    @staticmethod
    def root(n: int) -> np.ndarray:
        """n root gaps (all zero)."""
        return np.zeros(n)


def step_gap(spec: GameSpec, d: np.ndarray, e_own: np.ndarray, e_opp: np.ndarray,
             eps_own: np.ndarray, eps_opp: np.ndarray) -> np.ndarray:
    """Vectorized one-stage transition of the signed gap (float64)."""
    return (np.asarray(d, dtype=float) + np.asarray(e_own, dtype=float)
            - np.asarray(e_opp, dtype=float) + np.asarray(eps_own, dtype=float)
            - np.asarray(eps_opp, dtype=float))


def stage_reward(spec: GameSpec, t: int, e_own: np.ndarray, d_next: np.ndarray) -> np.ndarray:
    """Sampled stage reward -k e^2 (+ realized prize at the final stage)."""
    r = -spec.k * np.asarray(e_own, dtype=float) ** 2
    if t >= spec.T:
        r = r + spec.terminal_reward(d_next)
    return r


def gap_bin_index(spec: GameSpec, t: int, d: np.ndarray, bin_width: float = 10.0
                  ) -> Tuple[np.ndarray, int]:
    """Bin index of gaps on the equal-width partition of D_t (for visitation counts)."""
    half = spec.domain_half(t)
    nb = int(np.ceil(2.0 * half / bin_width - 1e-9))
    idx = np.floor((np.asarray(d, dtype=float) + half) / (2.0 * half) * nb).astype(int)
    return np.clip(idx, 0, nb - 1), nb
