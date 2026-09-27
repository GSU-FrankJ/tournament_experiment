"""Coverage, dense policy profiles, self-play economics and diagnosis fields (T=3 pilot).

Populations are kept apart by construction:
  - training coverage: learner-signed pre-action gaps of the mixed-start rollouts;
  - mean / stochastic self-play: root episodes of the frozen checkpoint, separate
    numpy Generators (never the training streams);
  - BR chain: the verifier's pmf on its own state grid (not handled here).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402,F401  (sets sys.path for W)

from envs.curriculum_env import GameSpec, StartSampler, step_gap  # noqa: E402
from utils.dp_br_verifier import beta_std_norm  # noqa: E402

BetaFn = Callable[[int, np.ndarray], Tuple[np.ndarray, np.ndarray]]
MeanFn = Callable[[int, np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Strict binning of gaps on D_t (no silent clipping of real out-of-domain values)
# ---------------------------------------------------------------------------

class CoverageDomainError(ValueError):
    """A gap fell outside D_t beyond numerical tolerance."""


def stage_bin_edges(spec: GameSpec, t: int, bin_width: float) -> np.ndarray:
    """Width-``bin_width`` edges on D_t (stage 1: the single root point [0, 0])."""
    if t == 1:
        return np.zeros(2)
    return StartSampler(spec, bin_width).bin_edges(t)


def strict_gap_bins(spec: GameSpec, t: int, d: np.ndarray, bin_width: float, tol: float = 1e-9,
                    origin: str = "") -> np.ndarray:
    """Bin indices of gaps on D_t after checking the domain.

    Only endpoint roundoff within ``tol * max(1, half)`` is mapped to the end
    bins; anything further out raises.

    Args:
        spec: Game specification.
        t: Stage.
        d: Gaps (float64).
        bin_width: Bin width.
        tol: Relative tolerance (the verifier's ``grid_tol``).
        origin: Label for error messages.

    Returns:
        Integer bin indices.

    Raises:
        CoverageDomainError: If any gap is non-finite or outside D_t +- tolerance.
    """
    d = np.asarray(d, dtype=float).reshape(-1)
    half = spec.domain_half(t)
    eps = tol * max(1.0, half)
    bad = ~np.isfinite(d) | (d < -half - eps) | (d > half + eps)
    if bad.any():
        j = int(np.nonzero(bad)[0][0])
        raise CoverageDomainError(
            f"stage {t} origin={origin!r}: gap {d[j]!r} outside D_{t}=[{-half:g}, {half:g}] "
            f"(tolerance {eps:g}; {int(bad.sum())} offending values)")
    if t == 1:
        return np.zeros(d.size, dtype=np.int64)
    edges = stage_bin_edges(spec, t, bin_width)
    nb = edges.size - 1
    idx = np.searchsorted(edges, d, side="right") - 1
    return np.clip(idx, 0, nb - 1).astype(np.int64)


def n_stage_bins(spec: GameSpec, t: int, bin_width: float) -> int:
    """Number of bins on D_t (1 for the root stage)."""
    return stage_bin_edges(spec, t, bin_width).size - 1


def center_bin_mask(spec: GameSpec, t: int, bin_width: float, half_width: float, tol: float = 1e-9
                    ) -> np.ndarray:
    """Mask of the D_t bins lying inside [-half_width, half_width].

    Raises:
        ValueError: If the interval does not coincide with bin edges.
    """
    edges = stage_bin_edges(spec, t, bin_width)
    eps = tol * max(1.0, half_width)
    left, right = edges[:-1], edges[1:]
    m = (left >= -half_width - eps) & (right <= half_width + eps)
    if not m.any() or abs(left[m][0] + half_width) > eps or abs(right[m][-1] - half_width) > eps:
        raise ValueError(f"[-{half_width}, {half_width}] does not align with the stage-{t} bin edges")
    return m


def center_balanced(spec: GameSpec, t: int, n: int, rng: np.random.Generator, bin_width: float,
                    half_width: float) -> np.ndarray:
    """ES gaps uniform on [-half_width, half_width]: pick a central bin uniformly, then uniform inside.

    Same draw pattern as StartSampler.balanced (integers, then random) on the
    subset of D_t bins inside the interval.
    """
    edges = stage_bin_edges(spec, t, bin_width)
    m = center_bin_mask(spec, t, bin_width, half_width)
    left, right = edges[:-1][m], edges[1:][m]
    b = rng.integers(0, left.size, size=n)
    u = rng.random(n)
    return left[b] + u * (right[b] - left[b])


def stage_region_metrics(d_grid: np.ndarray, gain: np.ndarray, std_norm: Optional[np.ndarray],
                         half_center: float, dw: float, tol: float = 1e-9) -> Dict[str, Any]:
    """Max/mean of a per-state gain on the whole grid, |d| <= half_center, and |d| > half_center.

    Args:
        d_grid: Stage grid.
        gain: Per-state gain (e.g. V_t^BR - V_t^mean), raw units.
        std_norm: Optional per-state normalized Beta std on the same grid.
        half_center: Center half-width (B for stage 3).
        dw: Prize spread.
        tol: Relative boundary tolerance (|d| = half_center counts as center).

    Returns:
        ``{region}_max_over_dw``, ``{region}_argmax_d``, ``{region}_mean_over_dw`` (grid average),
        ``{region}_max_std_norm`` for region in all/center/outer, plus point counts.
    """
    d = np.asarray(d_grid, dtype=float)
    g = np.asarray(gain, dtype=float)
    center = np.abs(d) <= half_center + tol * max(1.0, half_center)
    out: Dict[str, Any] = {"center_half_width": float(half_center), "n_all": int(d.size),
                           "n_center": int(center.sum()), "n_outer": int((~center).sum())}
    for name, m in (("all", np.ones(d.size, dtype=bool)), ("center", center), ("outer", ~center)):
        if not m.any():
            out.update({f"{name}_max_over_dw": None, f"{name}_argmax_d": None, f"{name}_mean_over_dw": None,
                        f"{name}_reason": "empty region"})
            continue
        j = int(np.argmax(np.where(m, g, -np.inf)))
        out[f"{name}_max_over_dw"] = float(g[j]) / dw
        out[f"{name}_argmax_d"] = float(d[j])
        out[f"{name}_mean_over_dw"] = float(g[m].mean()) / dw
        if std_norm is not None:
            sn = np.asarray(std_norm, dtype=float)
            k = int(np.argmax(np.where(m, sn, -np.inf)))
            out[f"{name}_max_std_norm"] = float(sn[k])
            out[f"{name}_max_std_norm_d"] = float(d[k])
    return out


# ---------------------------------------------------------------------------
# Training coverage accumulator
# ---------------------------------------------------------------------------

def coverage_layout(spec: GameSpec, bin_width: float) -> Tuple[List[Dict[str, Any]], int]:
    """Flat layout of all coverage count segments.

    Segments: ``visits`` (start_stage s, current_stage t >= s; learner-signed
    pre-action gap), ``start_raw`` (player-0 ES/root start gap before the role
    flip) and ``start_learner`` (learner-signed start gap).
    """
    T = spec.T
    segs: List[Dict[str, Any]] = []
    off = 0
    for s in range(1, T + 1):
        for t in range(s, T + 1):
            nb = n_stage_bins(spec, t, bin_width)
            segs.append({"kind": "visits", "start_stage": s, "current_stage": t, "nbins": nb,
                         "offset": off})
            off += nb
    for kind in ("start_raw", "start_learner"):
        for s in range(1, T + 1):
            nb = n_stage_bins(spec, s, bin_width)
            segs.append({"kind": kind, "start_stage": s, "current_stage": s, "nbins": nb,
                         "offset": off})
            off += nb
    return segs, off


def seg_key(seg: Dict[str, Any]) -> str:
    """Batch dict key of a layout segment."""
    if seg["kind"] == "visits":
        return f"s{seg['start_stage']}_t{seg['current_stage']}"
    return f"s{seg['start_stage']}"


class CoverageAccumulator:
    """Per-phase integer counts on the flat layout plus snapshots at verifier calls."""

    def __init__(self, spec: GameSpec, bin_width: float, phases: List[str]):
        self.spec = spec
        self.bin_width = bin_width
        self.layout, self.length = coverage_layout(spec, bin_width)
        self.phases = list(phases)
        self.totals = {p: np.zeros(self.length, dtype=np.int64) for p in self.phases}
        self.phase_updates = {p: 0 for p in self.phases}
        self.snap_vectors: List[np.ndarray] = []
        self.snap_ids: List[Tuple[int, int, int, int]] = []   # phase_idx, local, global, call_index

    def vector_from_batch(self, batch: Dict[str, Any]) -> np.ndarray:
        """Flat count vector of one collected batch."""
        v = np.zeros(self.length, dtype=np.int64)
        src = {"visits": batch["visitation_by_start_stage"], "start_raw": batch["start_bins_raw"],
               "start_learner": batch["start_bins_learner"]}
        for seg in self.layout:
            arr = src[seg["kind"]].get(seg_key(seg))
            if arr is not None:
                v[seg["offset"]:seg["offset"] + seg["nbins"]] = arr
        return v

    def add(self, phase: str, batch: Dict[str, Any]) -> None:
        """Add one update's counts."""
        self.totals[phase] += self.vector_from_batch(batch)
        self.phase_updates[phase] += 1

    def snapshot(self, phase: str, local: int, global_update: int, call_index: int) -> int:
        """Store the phase-cumulative counts at a verifier call; return its index."""
        self.snap_vectors.append(self.totals[phase].copy())
        self.snap_ids.append((self.phases.index(phase), local, global_update, call_index))
        return len(self.snap_vectors) - 1

    def arrays(self) -> Dict[str, np.ndarray]:
        """Arrays for ``coverage.npz``."""
        import json

        out = {
            "layout_json": np.array(json.dumps(self.layout)),
            "phases": np.array(self.phases),
            "phase_totals": np.stack([self.totals[p] for p in self.phases]),
            "phase_updates": np.array([self.phase_updates[p] for p in self.phases], dtype=np.int64),
            "snapshot_counts": (np.stack(self.snap_vectors) if self.snap_vectors
                                else np.zeros((0, self.length), dtype=np.int64)),
            "snapshot_ids": (np.array(self.snap_ids, dtype=np.int64) if self.snap_ids
                             else np.zeros((0, 4), dtype=np.int64)),
            "snapshot_ids_columns": np.array(["phase_index", "local_update", "global_update",
                                              "call_index"]),
            "bin_width": np.array(self.bin_width),
        }
        for t in range(1, self.spec.T + 1):
            out[f"edges_t{t}"] = stage_bin_edges(self.spec, t, self.bin_width)
        return out


def count_stats(counts: np.ndarray) -> Dict[str, Any]:
    """min, p05, median, max, mean, CV (population SD / mean), zero bins."""
    c = np.asarray(counts, dtype=float)
    if c.size == 0:
        return {"n_bins": 0}
    mean = float(c.mean())
    return {"n_bins": int(c.size), "total": float(c.sum()), "min": float(c.min()),
            "p05": float(np.percentile(c, 5)), "median": float(np.median(c)), "max": float(c.max()),
            "mean": mean, "cv": (float(c.std()) / mean) if mean > 0 else None,
            "zero_bin_count": int((c == 0).sum())}


def tail_masks(edges: np.ndarray, half: float, frac: float) -> Tuple[np.ndarray, np.ndarray]:
    """Positive/negative tail bins: |midpoint| >= frac * half."""
    mid = 0.5 * (edges[:-1] + edges[1:])
    if half <= 0:
        z = np.zeros(mid.size, dtype=bool)
        return z, z
    return (mid >= frac * half), (mid <= -frac * half)


# ---------------------------------------------------------------------------
# Dense policy profiles
# ---------------------------------------------------------------------------

def dense_points(half: float, step: float) -> np.ndarray:
    """Symmetric grid on [-half, half] with spacing <= step containing 0 and endpoints."""
    if half <= 0:
        return np.zeros(1)
    n_half = int(np.ceil(half / step - 1e-9))
    return np.linspace(-half, half, 2 * n_half + 1)


def policy_profiles(mean_fn: MeanFn, beta_fn: BetaFn, spec: GameSpec, step: float
                    ) -> Dict[str, Any]:
    """Full-domain policy profiles and the dense concentration from the same alpha/beta.

    Args:
        mean_fn: Verifier-facing mean effort function.
        beta_fn: ``(t, d) -> (alpha, beta)`` (float64 of the float32 network output).
        spec: Game specification.
        step: Dense spacing (0.05).

    Returns:
        Dict with ``arrays`` (per-stage profile arrays + asymmetry curves),
        ``concentration`` (max std_norm over all stages/points) and
        ``mean_fn_consistency_max_abs`` (mean_fn vs alpha/beta mean).
    """
    arrays: Dict[str, np.ndarray] = {}
    best = None
    finite = True
    n_pts = 0
    consistency = 0.0
    for t in range(1, spec.T + 1):
        d = dense_points(spec.domain_half(t), step)
        a, b = beta_fn(t, d)
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        s = a + b
        mean = spec.e_min + spec.e_range * a / s
        mean_direct = np.asarray(mean_fn(t, d), dtype=float)
        consistency = max(consistency, float(np.max(np.abs(mean_direct - mean))))
        var = spec.e_range ** 2 * a * b / (s * s * (s + 1.0))
        std_norm = beta_std_norm(a, b)
        opp = np.asarray(mean_fn(t, -d), dtype=float)
        dn = np.zeros_like(d) if t == 1 else d / ((t - 1) * spec.B)
        arrays.update({f"t{t}_d": d, f"t{t}_d_normalized": dn, f"t{t}_alpha": a, f"t{t}_beta": b,
                       f"t{t}_mean_effort": mean, f"t{t}_effort_variance": var,
                       f"t{t}_std_effort": np.sqrt(var), f"t{t}_std_norm": std_norm,
                       f"t{t}_opponent_mean_effort": opp})
        if t >= 2:
            pos = d > 0
            arrays[f"t{t}_asym_x"] = d[pos]
            arrays[f"t{t}_leader_mean_effort"] = mean[pos]
            arrays[f"t{t}_follower_mean_effort"] = opp[pos]
            arrays[f"t{t}_lead_follow_policy_difference"] = mean[pos] - opp[pos]
        n_pts += d.size
        if not (np.isfinite(std_norm).all() and np.isfinite(a).all() and np.isfinite(b).all()):
            finite = False
        if np.isfinite(std_norm).any():
            j = int(np.nanargmax(std_norm))
            cand = (float(std_norm[j]), t, float(d[j]), float(a[j]), float(b[j]))
            if best is None or cand[0] > best[0]:
                best = cand
    conc: Dict[str, Any] = {"valid": bool(finite and best is not None), "n_points": n_pts, "step": step}
    if best is not None:
        conc.update({"max_std_norm": best[0], "max_std_effort": best[0] * spec.e_range,
                     "stage": best[1], "d": best[2], "alpha": best[3], "beta": best[4]})
    return {"arrays": arrays, "concentration": conc, "mean_fn_consistency_max_abs": consistency}


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

def summarize_moments(n: float, total: float, total_sq: float) -> Dict[str, Any]:
    """Mean, sample variance and MCSE from (n, sum, sum of squares).

    Args:
        n: Sample count.
        total: Sum.
        total_sq: Sum of squares.

    Returns:
        Dict with n/sum/sum_sq, mean, variance, mcse and reasons for nulls.

    Raises:
        ArithmeticError: On a substantively negative variance (implementation error).
    """
    n = int(round(float(n)))
    out: Dict[str, Any] = {"n": n, "sum": float(total), "sum_sq": float(total_sq),
                           "mean": None, "variance": None, "mcse": None}
    if n <= 0:
        out["mean_reason"] = "n=0"
        out["variance_reason"] = "n<2"
        return out
    out["mean"] = float(total) / n
    if n < 2:
        out["variance_reason"] = "n<2"
        return out
    var = (float(total_sq) - float(total) ** 2 / n) / (n - 1)
    if var < 0.0:
        scale = max(1.0, abs(float(total_sq)) / n)
        if -var <= 1e-10 * scale:
            var = 0.0
        else:
            raise ArithmeticError(f"negative variance {var!r} (n={n}, sum={total!r}, sum_sq={total_sq!r})")
    out["variance"] = var
    out["mcse"] = math.sqrt(var / n)
    return out


def hist_quantile(edges: np.ndarray, counts: np.ndarray, p: float) -> Optional[float]:
    """Approximate quantile ``histogram_linear_width10``.

    The first non-empty bin whose cumulative mass reaches ``p``; linear
    interpolation inside it by ``(p - F_left) / mass_bin``.
    """
    c = np.asarray(counts, dtype=float)
    tot = c.sum()
    if tot <= 0:
        return None
    if edges[0] == edges[-1]:
        return float(edges[0])
    mass = c / tot
    cum = np.cumsum(mass)
    for j in range(c.size):
        if c[j] > 0 and cum[j] >= p - 1e-15:
            f_left = cum[j] - mass[j]
            frac = min(max((p - f_left) / mass[j], 0.0), 1.0)
            return float(edges[j] + frac * (edges[j + 1] - edges[j]))
    return float(edges[-1])


# ---------------------------------------------------------------------------
# Self-play economics
# ---------------------------------------------------------------------------

def simulate_chunk(spec: GameSpec, beta_fn: BetaFn, mode: int, n: int,
                   noise_fn: Callable[[int, int], Tuple[np.ndarray, np.ndarray]],
                   action_fn: Optional[Callable[[int, int, np.ndarray, np.ndarray], np.ndarray]]
                   ) -> Dict[str, np.ndarray]:
    """Roll ``n`` root episodes of symmetric self-play with the frozen actor.

    Player 0 observes its physical gap d, player 1 observes -d.

    Args:
        spec: Game specification.
        beta_fn: ``(t, d) -> (alpha, beta)``.
        mode: 0 = both players use the Beta mean; 1 = both sample Beta actions.
        n: Episodes.
        noise_fn: ``(t, n) -> (eps0, eps1)``.
        action_fn: ``(player, t, alpha, beta) -> normalized actions`` (mode 1).

    Returns:
        Arrays: ``d_pre (T, n)``, ``e0``, ``e1``, conditional Beta moments
        ``m0, m1, m2_0, m2_1``, ``d_final``, prizes, costs and payoffs.
    """
    T = spec.T
    out = {key: np.zeros((T, n)) for key in ("d_pre", "e0", "e1", "m0", "m1", "m2_0", "m2_1")}
    d = np.zeros(n)
    for t in range(1, T + 1):
        out["d_pre"][t - 1] = d
        a0, b0 = (np.asarray(x, dtype=float) for x in beta_fn(t, d))
        a1, b1 = (np.asarray(x, dtype=float) for x in beta_fn(t, -d))
        s0, s1 = a0 + b0, a1 + b1
        m0 = spec.e_min + spec.e_range * a0 / s0
        m1 = spec.e_min + spec.e_range * a1 / s1
        v0 = spec.e_range ** 2 * a0 * b0 / (s0 * s0 * (s0 + 1.0))
        v1 = spec.e_range ** 2 * a1 * b1 / (s1 * s1 * (s1 + 1.0))
        if mode == 0:
            e0, e1 = m0, m1
        else:
            e0 = spec.effort_from_action(action_fn(0, t, a0, b0))
            e1 = spec.effort_from_action(action_fn(1, t, a1, b1))
        eps0, eps1 = noise_fn(t, n)
        out["e0"][t - 1], out["e1"][t - 1] = e0, e1
        out["m0"][t - 1], out["m1"][t - 1] = m0, m1
        out["m2_0"][t - 1], out["m2_1"][t - 1] = v0 + m0 ** 2, v1 + m1 ** 2
        d = step_gap(spec, d, e0, e1, eps0, eps1)
    out["d_final"] = d
    out["prize0"] = spec.terminal_reward(d)
    out["prize1"] = spec.terminal_reward(-d)
    out["cost0"] = spec.k * out["e0"] ** 2
    out["cost1"] = spec.k * out["e1"] ** 2
    out["payoff0"] = out["prize0"] - out["cost0"].sum(axis=0)
    out["payoff1"] = out["prize1"] - out["cost1"].sum(axis=0)
    return out


STAGE_VARS = ("eff_p0", "eff_p1", "cost_p0", "cost_p1", "X", "Y", "leader", "follower",
              "lf_diff", "phys_diff", "gap_p0", "theo_mean_p0", "theo_mean_p1",
              "theo_m2_p0", "theo_m2_p1")
EPISODE_VARS = ("payoff0", "payoff1", "U", "prize0", "prize1")


class SelfPlayAccumulator:
    """Sufficient statistics (n, sum, sum_sq), histograms and per-bin sums."""

    def __init__(self, spec: GameSpec, bin_width: float):
        self.spec = spec
        self.bw = bin_width
        T = spec.T
        self.stage = {v: np.zeros((T, 3)) for v in STAGE_VARS}
        self.episode = {v: np.zeros(3) for v in EPISODE_VARS}
        self.gap_sign = np.zeros((T, 3), dtype=np.int64)      # positive, negative, tie
        self.lf_ties = np.zeros(T, dtype=np.int64)
        self.outcome = np.zeros(3, dtype=np.int64)            # player0 wins, player1 wins, ties
        self.edges = {t: stage_bin_edges(spec, t, bin_width) for t in range(1, T + 1)}
        self.bins: Dict[str, np.ndarray] = {}
        for t in range(1, T + 1):
            nb = self.edges[t].size - 1
            for p in (0, 1):
                for name in ("count", "eff_sum", "eff_sq_sum", "cost_sum"):
                    self.bins[f"p{p}_{name}_t{t}"] = np.zeros(nb)

    @staticmethod
    def _add(arr: np.ndarray, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        arr += (x.size, float(x.sum()), float(np.dot(x, x)))

    def add(self, ch: Dict[str, np.ndarray]) -> None:
        """Accumulate one simulated chunk."""
        k = self.spec.k
        for t in range(1, self.spec.T + 1):
            i = t - 1
            d = ch["d_pre"][i]
            e0, e1 = ch["e0"][i], ch["e1"][i]
            S = self.stage
            self._add(S["eff_p0"][i], e0)
            self._add(S["eff_p1"][i], e1)
            self._add(S["cost_p0"][i], k * e0 ** 2)
            self._add(S["cost_p1"][i], k * e1 ** 2)
            self._add(S["X"][i], 0.5 * (e0 + e1))
            self._add(S["Y"][i], 0.5 * (e0 ** 2 + e1 ** 2))
            pos, neg = d > 0, d < 0
            lead = pos | neg
            leader = np.where(pos, e0, e1)[lead]
            follower = np.where(pos, e1, e0)[lead]
            self._add(S["leader"][i], leader)
            self._add(S["follower"][i], follower)
            self._add(S["lf_diff"][i], leader - follower)
            self.lf_ties[i] += int((~lead).sum())
            self._add(S["phys_diff"][i], e0 - e1)
            self._add(S["gap_p0"][i], d)
            self.gap_sign[i] += (int(pos.sum()), int(neg.sum()), int((~lead).sum()))
            self._add(S["theo_mean_p0"][i], ch["m0"][i])
            self._add(S["theo_mean_p1"][i], ch["m1"][i])
            self._add(S["theo_m2_p0"][i], ch["m2_0"][i])
            self._add(S["theo_m2_p1"][i], ch["m2_1"][i])
            nb = self.edges[t].size - 1
            for p, (g, e) in enumerate(((d, e0), (-d, e1))):
                b = strict_gap_bins(self.spec, t, g, self.bw, origin=f"selfplay_p{p}")
                self.bins[f"p{p}_count_t{t}"] += np.bincount(b, minlength=nb)
                self.bins[f"p{p}_eff_sum_t{t}"] += np.bincount(b, weights=e, minlength=nb)
                self.bins[f"p{p}_eff_sq_sum_t{t}"] += np.bincount(b, weights=e * e, minlength=nb)
                self.bins[f"p{p}_cost_sum_t{t}"] += np.bincount(b, weights=k * e * e, minlength=nb)
        for v in ("payoff0", "payoff1", "prize0", "prize1"):
            self._add(self.episode[v], ch[v])
        self._add(self.episode["U"], 0.5 * (ch["payoff0"] + ch["payoff1"]))
        df = ch["d_final"]
        self.outcome += (int((df > 0).sum()), int((df < 0).sum()), int((df == 0).sum()))

    def arrays(self, prefix: str) -> Dict[str, np.ndarray]:
        """All accumulator arrays under ``prefix``."""
        out = {f"{prefix}_stage_{v}": a for v, a in self.stage.items()}
        out.update({f"{prefix}_episode_{v}": a for v, a in self.episode.items()})
        out[f"{prefix}_gap_sign_counts"] = self.gap_sign
        out[f"{prefix}_lf_ties"] = self.lf_ties
        out[f"{prefix}_outcome_counts"] = self.outcome
        out.update({f"{prefix}_bins_{k}": v for k, v in self.bins.items()})
        return out


def selfplay_summary(stage: Dict[str, np.ndarray], episode: Dict[str, np.ndarray],
                     gap_sign: np.ndarray, lf_ties: np.ndarray, outcome: np.ndarray,
                     bins: Dict[str, np.ndarray], edges: Dict[int, np.ndarray], k: float
                     ) -> Dict[str, Any]:
    """Readable summary recomputed from saved sufficient statistics."""
    T = len(edges)
    per_stage = {}
    for t in range(1, T + 1):
        i = t - 1
        row: Dict[str, Any] = {}
        for v in STAGE_VARS:
            row[v] = summarize_moments(*stage[v][i])
        for p in (0, 1):
            m = row[f"eff_p{p}"]
            row[f"E_e2_p{p}"] = (m["sum_sq"] / m["n"]) if m["n"] > 0 else None
            row[f"E_cost_p{p}"] = row[f"cost_p{p}"]["mean"]
        row["representative_E_e2"] = row["Y"]["mean"]
        row["representative_expected_cost"] = (k * row["Y"]["mean"]) if row["Y"]["mean"] is not None else None
        row["leader_follower_ties"] = int(lf_ties[i])
        if row["leader"]["n"] == 0:
            row["leader_follower_note"] = "no leader/follower samples (all pre-action gaps tie)"
        n_gap = int(gap_sign[i].sum())
        row["gap_fraction_positive"] = gap_sign[i][0] / n_gap if n_gap else None
        row["gap_fraction_negative"] = gap_sign[i][1] / n_gap if n_gap else None
        row["gap_fraction_tie"] = gap_sign[i][2] / n_gap if n_gap else None
        c0 = bins[f"p0_count_t{t}"]
        crep = 0.5 * (c0 + bins[f"p1_count_t{t}"])
        row["gap_quantiles_player0"] = {f"p{int(p * 100):02d}": hist_quantile(edges[t], c0, p)
                                        for p in (0.05, 0.5, 0.95)}
        row["gap_quantiles_representative"] = {f"p{int(p * 100):02d}": hist_quantile(edges[t], crep, p)
                                               for p in (0.05, 0.5, 0.95)}
        row["gap_quantile_method"] = "histogram_linear_width10 (approximate)"
        row["hist_mass_player0"] = float(c0.sum() / max(1.0, row["eff_p0"]["n"]))
        per_stage[str(t)] = row
    ep = {v: summarize_moments(*episode[v]) for v in EPISODE_VARS}
    n_ep = int(episode["payoff0"][0])
    cost0 = sum(stage["cost_p0"][i][1] for i in range(T))
    cost1 = sum(stage["cost_p1"][i][1] for i in range(T))
    acct0 = episode["prize0"][1] - cost0 - episode["payoff0"][1]
    acct1 = episode["prize1"][1] - cost1 - episode["payoff1"][1]
    return {
        "stages": per_stage,
        "episode": ep,
        "win_rate_player0": outcome[0] / n_ep if n_ep else None,
        "win_rate_player1": outcome[1] / n_ep if n_ep else None,
        "tie_rate": outcome[2] / n_ep if n_ep else None,
        "accounting_residual_player0": float(acct0),
        "accounting_residual_player1": float(acct1),
        "accounting_ok": bool(abs(acct0) <= 1e-9 * max(1.0, n_ep) and abs(acct1) <= 1e-9 * max(1.0, n_ep)),
    }


def econ_streams(eval_cfg: Dict[str, Any], seed: int, q: float, mode_id: int, rep_id: int
                 ) -> Tuple[Dict[int, np.random.Generator], List[int]]:
    """Independent evaluation generators (environment, player0, player1)."""
    base = [int(eval_cfg["seed_base"]), int(seed), int(q), int(eval_cfg["seed_tag"]), int(mode_id),
            int(rep_id)]
    return {sid: np.random.default_rng(np.random.SeedSequence(base + [sid])) for sid in (0, 1, 2)}, base


def evaluate_economics(agent: Any, spec: GameSpec, seed: int, eval_cfg: Dict[str, Any],
                       bin_width: float, v_mean_root_final: Optional[float] = None,
                       tracker: Any = None) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Mean-policy and stochastic-policy root self-play of a frozen checkpoint.

    Args:
        agent: CurriculumPPO holding the frozen actor (never updated here).
        spec: Game specification.
        seed: Training seed (enters the evaluation SeedSequence).
        eval_cfg: ``resolved.economics``.
        bin_width: Histogram bin width.
        v_mean_root_final: Final-tier V1_mean(0) for the mean-mode comparison.
        tracker: Optional ResourceTracker.

    Returns:
        ``(json_data, array_data)``.
    """
    import torch

    from run.run_final_dp_br import make_policy_fns

    was_training = agent.actor.training
    agent.actor.eval()
    _, beta_fn = make_policy_fns(agent, spec)
    out: Dict[str, Any] = {"modes": {}, "config": eval_cfg, "bin_width": bin_width,
                           "note_mean_vs_stochastic": "separate occupancies; never pooled",
                           "stochastic_moment_labels": {
                               "theo_mean_p*/theo_m2_p*": "conditional Beta moments E[e|d], E[e^2|d] "
                                                          "averaged over the stochastic occupancy",
                               "eff_p*": "realized clamped sampled actions"}}
    arrays: Dict[str, np.ndarray] = {f"edges_t{t}": stage_bin_edges(spec, t, bin_width)
                                     for t in range(1, spec.T + 1)}
    try:
        with torch.no_grad():
            for mode_id, mode_name in ((0, "mean"), (1, "stochastic")):
                reps = []
                pooled = SelfPlayAccumulator(spec, bin_width)
                for rep in range(int(eval_cfg["replicates"])):
                    gens, base = econ_streams(eval_cfg, seed, spec.q, mode_id, rep)
                    acc = SelfPlayAccumulator(spec, bin_width)

                    def noise_fn(t: int, n: int, g: np.random.Generator = gens[0]
                                 ) -> Tuple[np.ndarray, np.ndarray]:
                        return g.uniform(-spec.q, spec.q, size=n), g.uniform(-spec.q, spec.q, size=n)

                    def action_fn(p: int, t: int, a: np.ndarray, b: np.ndarray,
                                  g: Dict[int, np.random.Generator] = gens) -> np.ndarray:
                        return agent.sample_actions(a, b, g[1 + p])

                    seg_ctx = tracker.segment(f"economics_{mode_name}_rep{rep}") if tracker else None
                    seg = seg_ctx.__enter__() if seg_ctx else {}
                    try:
                        total = int(eval_cfg["episodes_per_replicate"])
                        chunk = int(eval_cfg["chunk_size"])
                        for start in range(0, total, chunk):
                            n = min(chunk, total - start)
                            ch = simulate_chunk(spec, beta_fn, mode_id, n, noise_fn,
                                                action_fn if mode_id == 1 else None)
                            acc.add(ch)
                    finally:
                        if seg_ctx:
                            seg_ctx.__exit__(None, None, None)
                    for key, arr in acc.stage.items():
                        pooled.stage[key] += arr
                    for key, arr in acc.episode.items():
                        pooled.episode[key] += arr
                    pooled.gap_sign += acc.gap_sign
                    pooled.lf_ties += acc.lf_ties
                    pooled.outcome += acc.outcome
                    for key, arr in acc.bins.items():
                        pooled.bins[key] += arr
                    arrays.update(acc.arrays(f"m{mode_id}_r{rep}"))
                    summ = selfplay_summary(acc.stage, acc.episode, acc.gap_sign, acc.lf_ties,
                                            acc.outcome, acc.bins, acc.edges, spec.k)
                    reps.append({"rep_id": rep, "seed_sequence_base": base,
                                 "stream_ids": {"0": "environment", "1": "player0_actions",
                                                "2": "player1_actions"},
                                 "episodes": int(eval_cfg["episodes_per_replicate"]),
                                 "resources": {k_: v_ for k_, v_ in seg.items()},
                                 "summary": summ})
                arrays.update(pooled.arrays(f"m{mode_id}_pooled"))
                pooled_summary = selfplay_summary(pooled.stage, pooled.episode, pooled.gap_sign,
                                                  pooled.lf_ties, pooled.outcome, pooled.bins,
                                                  pooled.edges, spec.k)
                rep_u = [r["summary"]["episode"]["U"]["mean"] for r in reps]
                rep_x = {str(t): [r["summary"]["stages"][str(t)]["X"]["mean"] for r in reps]
                         for t in range(1, spec.T + 1)}
                out["modes"][mode_name] = {
                    "mode_id": mode_id, "replicates": reps, "pooled": pooled_summary,
                    "replicate_means_U": rep_u,
                    "replicate_sd_U": float(np.std(rep_u, ddof=1)) if len(rep_u) > 1 else None,
                    "replicate_means_X_by_stage": rep_x,
                    "replicate_sd_X_by_stage": {t: (float(np.std(v, ddof=1)) if len(v) > 1 else None)
                                                for t, v in rep_x.items()},
                }
    finally:
        agent.actor.train(was_training)
    mean_u = out["modes"]["mean"]["pooled"]["episode"]["U"]
    cmp: Dict[str, Any] = {"v1_mean_root_final": v_mean_root_final,
                           "mc_mean_U": mean_u["mean"], "mc_mcse_U": mean_u["mcse"]}
    if v_mean_root_final is None or mean_u["mean"] is None:
        cmp.update(difference=None, z=None, reason="final V1_mean(0) unavailable")
    else:
        cmp["difference"] = mean_u["mean"] - v_mean_root_final
        if mean_u["mcse"]:
            cmp["z"] = cmp["difference"] / mean_u["mcse"]
        else:
            cmp.update(z=None, reason="MCSE is 0 or unavailable")
    cmp["note"] = "difference contains MC noise and DP discretization; not a selection criterion"
    out["mean_payoff_vs_dp"] = cmp
    su = out["modes"]["stochastic"]["pooled"]["episode"]["U"]
    out["stochastic_minus_mean_U"] = {
        "difference": (su["mean"] - mean_u["mean"]) if su["mean"] is not None and mean_u["mean"] is not None else None,
        "se_independent": (math.sqrt(su["mcse"] ** 2 + mean_u["mcse"] ** 2)
                           if su["mcse"] is not None and mean_u["mcse"] is not None else None),
        "note": "descriptive only; not an exploitability measure"}
    return out, arrays


# ---------------------------------------------------------------------------
# Verifier-derived fields
# ---------------------------------------------------------------------------

def stage_continuation(res: Any) -> Dict[str, Dict[str, Any]]:
    """Per-stage max continuation gain V_t^BR - V_t^mean with argmax (ties: smallest d)."""
    out = {}
    for t, s in sorted(res.stages.items()):
        g = s.v_br - s.v_mean
        j = int(np.argmax(g))
        out[str(t)] = {"max_raw": float(g[j]), "max_over_dw": float(g[j]) / res.dw,
                       "argmax_d": float(s.d_grid[j]), "min_raw": float(g.min())}
    return out


def certification_flags(dev: Optional[Dict[str, Any]], fin: Optional[Dict[str, Any]],
                        dense_conc: Optional[Dict[str, Any]], has_candidate: bool,
                        cert_cfg: Dict[str, Any], dw: float) -> Dict[str, Any]:
    """The six certification conditions with every component kept.

    Args:
        dev: Recomputed development summary (None if the tier raised).
        fin: Final summary (None if the tier raised).
        dense_conc: Dense concentration dict.
        has_candidate: Whether the endpoint is a C first-eligible candidate.
        cert_cfg: ``resolved.final_certification``.
        dw: Prize spread.

    Returns:
        Flags dict including ``numeric_thresholds_pass``, ``final_joint_pass``
        and ``certification``.
    """
    def fin_num(x: Any) -> Optional[float]:
        return float(x) if x is not None and math.isfinite(float(x)) else None

    valid_dev = bool(dev is not None and dev.get("valid"))
    valid_final = bool(fin is not None and fin.get("valid"))
    f_dr = fin_num(fin.get("dreach")) if fin else None
    d_dr = fin_num(dev.get("dreach")) if dev else None
    f_ex = fin_num(fin.get("exp_root")) if fin else None
    d_ex = fin_num(dev.get("exp_root")) if dev else None
    f_dm = fin_num(fin.get("delta_max_all")) if fin else None
    d_dm = fin_num(dev.get("delta_max_all")) if dev else None
    f_df = fin_num(fin.get("dfull")) if fin else None
    d_df = fin_num(dev.get("dfull")) if dev else None

    def diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return abs(a - b) / dw if a is not None and b is not None else None

    dreach_final_over_dw = f_dr / dw if f_dr is not None else None
    r_dr, r_ex = diff(f_dr, d_dr), diff(f_ex, d_ex)
    flags: Dict[str, Any] = {
        "has_candidate": bool(has_candidate),
        "valid_dev": valid_dev, "valid_final": valid_final,
        "dreach_final_over_dw": dreach_final_over_dw,
        "main_threshold_over_dw": cert_cfg["main_dreach_final_thr_over_dw"],
        "main_pass": bool(valid_final and dreach_final_over_dw is not None
                          and dreach_final_over_dw <= cert_cfg["main_dreach_final_thr_over_dw"]),
        "refine_dreach_diff_over_dw": r_dr,
        "refine_dreach_pass": bool(r_dr is not None and r_dr <= cert_cfg["refine_dreach_thr_over_dw"]),
        "refine_exp_diff_over_dw": r_ex,
        "refine_exp_pass": bool(r_ex is not None and r_ex <= cert_cfg["refine_exp_root_thr_over_dw"]),
        "refine_delta_max_all_diff_over_dw": diff(f_dm, d_dm),
        "refine_dfull_diff_over_dw": diff(f_df, d_df),
        "refine_threshold_over_dw": cert_cfg["refine_dreach_thr_over_dw"],
    }
    for key in ("reach_delta_max", "full_delta_max"):
        per = {}
        for t in ("1", "2", "3"):
            a = fin_num((fin or {}).get(key, {}).get(t)) if fin else None
            b = fin_num((dev or {}).get(key, {}).get(t)) if dev else None
            per[t] = diff(a, b)
        flags[f"refine_{key}_diff_over_dw_by_stage"] = per
    dc = dense_conc or {}
    dc_val = fin_num(dc.get("max_std_norm")) if dc else None
    flags["dense_conc_max_std_norm"] = dc_val
    flags["dense_conc_threshold"] = cert_cfg["dense_conc_thr"]
    flags["dense_conc_valid"] = bool(dc.get("valid"))
    flags["dense_conc_pass"] = bool(dc.get("valid") and dc_val is not None and dc_val <= cert_cfg["dense_conc_thr"])
    flags["numeric_thresholds_pass"] = bool(valid_dev and valid_final and flags["main_pass"]
                                            and flags["refine_dreach_pass"] and flags["refine_exp_pass"]
                                            and flags["dense_conc_pass"])
    flags["final_joint_pass"] = bool(has_candidate and flags["numeric_thresholds_pass"])
    if not has_candidate:
        flags["certification"] = "not_applicable_no_candidate"
    else:
        flags["certification"] = "certified" if flags["final_joint_pass"] else "not_certified"
    return flags


def _max_state(delta_max: Dict[str, Any], argmax: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    best = None
    for t in sorted(delta_max, key=int):
        v = delta_max[t]
        if v is None or not math.isfinite(float(v)):
            continue
        if best is None or float(v) > best["delta"]:
            best = {"stage": int(t), "d": argmax.get(t), "delta": float(v)}
    return best


def diagnose_minimum_C(verifier_records: List[Dict[str, Any]],
                       min_checkpoint_metadata: Optional[Dict[str, Any]],
                       c_reached: bool = True, dw: float = 4.0) -> Dict[str, Any]:
    """Select the minimum valid C development call and its same-call diagnostics.

    Args:
        verifier_records: All development-call records (any phase).
        min_checkpoint_metadata: ``min_dev_record.json`` content (or None).
        c_reached: Whether phase C was entered.
        dw: Prize spread.

    Returns:
        Diagnosis dict; null fields carry ``reason``.
    """
    if not c_reached:
        return {"min_valid_C_dreach_over_dw": None, "reason": "C_not_reached"}
    valid = []
    for r in verifier_records:
        if r.get("phase") != "C" or not r.get("valid"):
            continue
        s = r.get("summary") or {}
        v = s.get("dreach_over_dw")
        if v is None or not math.isfinite(float(v)):
            continue
        valid.append(r)
    if not valid:
        return {"min_valid_C_dreach_over_dw": None, "reason": "no_valid_C_verifier"}
    best = min(valid, key=lambda r: (float(r["summary"]["dreach_over_dw"]), int(r["global_update"])))
    s = best["summary"]
    reach = {t: s["reach_delta_max"].get(t) for t in ("1", "2", "3")}
    full = {t: s["full_delta_max"].get(t) for t in ("1", "2", "3")}
    contrib_sum = sum(float(v) for v in reach.values() if v is not None)
    out = {
        "min_valid_C_dreach_over_dw": float(s["dreach_over_dw"]),
        "min_valid_C_dreach": float(s["dreach"]),
        "min_C_global_update": int(best["global_update"]),
        "min_C_local_update": int(best["local_update"]),
        "min_C_call_index": int(best["call_index"]),
        "min_C_conc": (best.get("concentration") or {}).get("max_std_norm"),
        "min_C_conc_stage": (best.get("concentration") or {}).get("stage"),
        "min_C_conc_d": (best.get("concentration") or {}).get("d"),
        "min_C_exp_root_over_dw": s.get("exp_root_over_dw"),
        "min_C_delta_max_all_over_dw": s.get("delta_max_all_over_dw"),
        "reach_contribution_raw": reach,
        "reach_contribution_over_dw": {t: (None if v is None else float(v) / dw) for t, v in reach.items()},
        "reach_argmax_d": {t: s["reach_argmax_d"].get(t) for t in ("1", "2", "3")},
        "full_contribution_raw": full,
        "full_contribution_over_dw": {t: (None if v is None else float(v) / dw) for t, v in full.items()},
        "full_argmax_d": {t: s["full_argmax_d"].get(t) for t in ("1", "2", "3")},
        "max_reach_state": _max_state(reach, s["reach_argmax_d"]),
        "max_all_state": _max_state(full, s["full_argmax_d"]),
        "contribution_sum_matches_dreach": bool(abs(contrib_sum - float(s["dreach"]))
                                                <= 1e-12 * max(1.0, abs(float(s["dreach"])))),
        "coverage_snapshot_index": best.get("coverage_snapshot_index"),
        "n_valid_C_calls": len(valid),
    }
    meta = min_checkpoint_metadata or {}
    ident = meta.get("identity", {})
    out["min_checkpoint_ref"] = meta.get("files")
    out["min_record_matches_selected_call"] = bool(
        ident.get("global_update") == out["min_C_global_update"]
        and ident.get("call_index") == out["min_C_call_index"])
    return out


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------

WILSON_Z = 1.959963984540054


def wilson(k: int, n: int, z: float = WILSON_Z) -> Dict[str, Any]:
    """95% Wilson score interval; n=0 returns NA."""
    if n <= 0:
        return {"k": k, "n": n, "p": None, "lo": None, "hi": None, "reason": "n=0 (NA)"}
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = 0.0 if k == 0 else max(0.0, center - half)        # k=0 / k=n bounds are exact
    hi = 1.0 if k == n else min(1.0, center + half)
    return {"k": k, "n": n, "p": p, "lo": lo, "hi": hi}
