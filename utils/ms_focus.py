"""Within-stratum focus and the bin probabilities of the E2 arms of MS-R4 (prompt D4).

The existing sampler (``envs.curriculum_env.StartSampler.stratified_bin_probs``) mixes the focus over ALL non-tail bins,
so at alpha = 0.5 the near-tie and middle shares move with the focus. The within-stratum sampler keeps the three stratum
masses fixed at lambda_T, lambda_P, lambda_M and lets the focus move samples only inside a stratum. For a stage t >= 2
with the strata tail T, near-tie P and middle M of ``StartSampler.strata`` (bin counts n_T, n_P, n_M; ``n_bins`` bins),

    lambda_T = n_T / n_bins,   lambda_P = the configured near-tie share,   lambda_M = 1 - lambda_P - lambda_T,

    p(b) = lambda_T 1[b in T] / n_T + sum_{S in {P, M}} lambda_S [ (1 - alpha) 1[b in S] / n_S + alpha f_S(b) ],

with f_S a distribution on S (``f_S(b) = 0`` outside S). A stage without a tail bin (n_T = 0, e.g. stage 2 of T = 3) has no
tail term and lambda_T = 0. Before the first valid check there is no map: the probabilities are then exactly
``StartSampler.stratified_bin_probs(t, lambda_P, half_width, alpha=0.0)`` (called, so that the floats are the same).

The focus ``f_S`` is built from the per-bin maps of the development verifier (``utils.ms_residual``): the EMA
``rho_bar`` of the action residual map and, for the ``residual_gain`` focus, the EMA ``gain_bar`` of the per-bin deviation
gain map:

    residual focus:        f_S = rho_bar / sum_S rho_bar,
    residual + gain focus: f_S = 1/2 rho_bar / sum_S rho_bar + 1/2 gain_bar / sum_S gain_bar,

NaN read as 0. Fallbacks (per stratum): a map whose sum over S is 0 or not finite is dropped and the other one is used
alone; if both are dropped, ``f_S`` is uniform on S.

Nothing here reads the closed-form equilibrium: the only inputs are the strata masks and the two verifier maps.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

from envs.curriculum_env import StartSampler

FOCUS_SCOPES = ("within_strata",)
FOCUS_MAPS = ("residual_gain",)
STRATUM_KEYS = ("near", "mid")      # the strata on which a focus acts (the tail stays uniform)


def _clean(w: Optional[np.ndarray], n: int, what: str) -> Optional[np.ndarray]:
    """A per-bin weight vector with NaN read as 0; ``None`` stays ``None``.

    Raises:
        ValueError: Wrong length, or a negative entry.
    """
    if w is None:
        return None
    v = np.asarray(w, dtype=float)
    if v.shape != (n,):
        raise ValueError(f"{what} must have one entry per bin ({n}); got shape {v.shape}")
    v = np.where(np.isnan(v), 0.0, v)
    if np.any(v < 0.0):
        raise ValueError(f"{what} must be non-negative")
    return v


def _normalised_on(v: Optional[np.ndarray], mask: np.ndarray) -> Optional[np.ndarray]:
    """``v / sum_mask(v)`` on ``mask`` (0 elsewhere); ``None`` if ``v`` is missing or its sum is 0 or not finite."""
    if v is None:
        return None
    x = np.where(mask, v, 0.0)
    s = float(x.sum())
    if not (np.isfinite(s) and s > 0.0):
        return None
    return x / s


def within_stratum_focus(strata: Mapping[str, np.ndarray], rho_bar: np.ndarray,
                         gain_bar: Optional[np.ndarray] = None) -> np.ndarray:
    """The focus ``f`` with ``f_S`` on the near-tie and middle strata (0 on the tail bins; each ``f_S`` sums to 1 over S).

    Args:
        strata: ``StartSampler.strata`` masks (``tail``, ``near``, ``mid``).
        rho_bar: EMA of the per-bin action residual map (NaN read as 0).
        gain_bar: EMA of the per-bin deviation gain map, or ``None`` for the residual focus.

    Returns:
        A float64 vector over the bins.

    Raises:
        ValueError: A map of the wrong length or with a negative entry, or an empty stratum.
    """
    n = int(np.asarray(strata["near"]).size)
    rho = _clean(rho_bar, n, "rho_bar")
    gain = _clean(gain_bar, n, "gain_bar")
    f = np.zeros(n)
    for key in STRATUM_KEYS:
        m = np.asarray(strata[key], dtype=bool)
        cnt = int(m.sum())
        if cnt == 0:
            raise ValueError(f"stratum {key!r} is empty")
        a = _normalised_on(rho, m)
        b = _normalised_on(gain, m)
        if a is not None and b is not None:
            part = 0.5 * a + 0.5 * b
        elif a is not None:
            part = a
        elif b is not None:
            part = b
        else:
            part = m / float(cnt)
        f = f + part
    return f


def within_strata_probs(sampler: StartSampler, t: int, lambda_p: float, near_tie_half_width: float, alpha: float,
                        focus_within: Optional[np.ndarray]) -> np.ndarray:
    """Bin probabilities of the within-stratum sampler (see the module docstring).

    Args:
        sampler: Start sampler of the game.
        t: Stage, ``t >= 2``.
        lambda_p: Near-tie share lambda_P, in (0, 1).
        near_tie_half_width: Half-width of the near-tie stratum.
        alpha: Weight of the focus, in [0, 1].
        focus_within: ``f`` from :func:`within_stratum_focus`, or ``None`` before the first valid check (then the
            probabilities are ``stratified_bin_probs(..., alpha=0.0)``).

    Returns:
        Probabilities over the bins of D_t, summing to 1; the mass of every stratum is lambda_T, lambda_P, lambda_M.

    Raises:
        ValueError: lambda_P or alpha out of range, lambda_M <= 0, a focus of the wrong length, or the strata errors of
            ``StartSampler.strata``.
    """
    if focus_within is None:
        return sampler.stratified_bin_probs(t, lambda_p, near_tie_half_width, alpha=0.0)
    if not 0.0 < lambda_p < 1.0:
        raise ValueError(f"lambda_P must lie in (0, 1); got {lambda_p}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1]; got {alpha}")
    st = sampler.strata(t, near_tie_half_width)
    n = sampler.n_bins(t)
    f = np.asarray(focus_within, dtype=float)
    if f.shape != (n,) or not np.all(np.isfinite(f)) or np.any(f < 0.0):
        raise ValueError(f"focus_within must be a finite non-negative vector of length {n}")
    n_tail = int(st["tail"].sum())
    lam_t = n_tail / n
    lam_m = 1.0 - lambda_p - lam_t
    if lam_m <= 0.0:
        raise ValueError(f"lambda_M = 1 - {lambda_p} - {lam_t:.6g} = {lam_m:.6g} must be positive")
    p = np.zeros(n)
    if n_tail:
        p[st["tail"]] = lam_t / n_tail
    for key, lam in (("near", lambda_p), ("mid", lam_m)):
        m = st[key]
        p = p + lam * ((1.0 - alpha) * (m / float(m.sum())) + alpha * np.where(m, f, 0.0))
    return p


def stratum_masses(probs: np.ndarray, strata: Mapping[str, np.ndarray]) -> Dict[str, float]:
    """Probability mass of the tail, near-tie and middle strata."""
    p = np.asarray(probs, dtype=float)
    return {k: float(p[np.asarray(strata[k], dtype=bool)].sum()) for k in ("tail", "near", "mid")}
