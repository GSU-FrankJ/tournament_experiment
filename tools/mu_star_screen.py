#!/usr/bin/env python3
"""Zero-GPU mu*(kappa) screen across all experiment conditions. NO GPU.

Generalizes the 3P-q35 smoothed-equilibrium analysis
(tools/claim_a_continuation_design.py) to the other conditions, to test whether
their already-measured raw-PPO landing points are consistent with converging to
the exploration-smoothed equilibrium mu*(kappa) (Claim B) rather than to the
deterministic e* (Claim A).

For each condition it computes:
  - mu*(kappa) for a few kappa via a damped best-response fixed point, where each
    opponent plays the ModeConc Beta family (mean=mu_j, alpha+beta=kappa+2);
  - the deterministic (kappa->inf) fixed point via the certified frozen-profile
    BR (utils.mc_br_polish.exploitability_frozen_profile) -- a self-check that
    should land at e*;
then compares the raw-PPO landing to the [mu*(low kappa) .. e*] smoothed band and
prints a per-cell verdict.

Sampled-only: realized-argmax winner, w_H/w_L prizes, quadratic per-player cost
k_i, additive ability l_i. No closed-form win probability anywhere; analytic e*
is a REPORTING BENCHMARK only and never enters a BR or a fixed-point iterate
(the iteration is seeded from the empirical raw-PPO landing, not from e*).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mc_br_polish import exploitability_frozen_profile
from paper.generator.config import e_star_for_experiment  # denominator-4 per experiment

BOUNDS = (0.0, 100.0)
M_BR = 120_000
DAMP = 0.5
MAX_ITER = 30
TOL = 0.03
KAPPAS = [20.0, 60.0, 200.0]
COARSE_HI = 55.0     # efforts across all conditions live well below this


# --------------------------- smoothed best response --------------------------
def _beta_from_mean_kappa(mu_eff: float, kappa: float,
                          bounds: Tuple[float, float]) -> Tuple[float, float]:
    lo, hi = bounds
    m = (mu_eff - lo) / (hi - lo)
    a = m * (kappa + 2.0)
    b = (kappa + 2.0) - a
    return a, b


def br_i_smoothed(i: int, mu_vec: np.ndarray, kappa: float, l: np.ndarray,
                  k: np.ndarray, w_h: float, w_l: float, q: float,
                  bounds: Tuple[float, float], *, seed: int,
                  M: int = M_BR) -> float:
    """Sampled BR of a deterministic player i vs opponents ~ Beta(mean=mu_j, kappa)."""
    lo, hi = bounds
    n = len(mu_vec)
    rng = np.random.default_rng(seed)
    opp = np.zeros((M, n))
    for j in range(n):
        if j == i:
            continue
        a, b = _beta_from_mean_kappa(float(mu_vec[j]), kappa, bounds)
        opp[:, j] = lo + (hi - lo) * rng.beta(max(a, 1e-3), max(b, 1e-3), size=M)
    eps = rng.uniform(-q, q, size=(M, n))
    y_others = opp + l[None, :] + eps
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    others_max = y_others[:, mask].max(axis=1)

    def _payoff(cands: np.ndarray) -> np.ndarray:
        yi = cands[:, None] + l[i] + eps[:, i][None, :]
        win = yi > others_max[None, :]
        prize = np.where(win, w_h, w_l)
        return prize.mean(axis=1) - k[i] * (cands ** 2)

    coarse = np.arange(lo, COARSE_HI + 1e-9, 0.5)
    c = coarse[int(_payoff(coarse).argmax())]
    fine = np.arange(max(lo, c - 1.5), min(hi, c + 1.5) + 1e-9, 0.1)
    return float(fine[int(_payoff(fine).argmax())])


def _fixed_point(step_fn, e_init: np.ndarray, symmetric: bool) -> np.ndarray:
    """Damped best-response iteration; step_fn(i, mu) -> BR of player i."""
    mu = np.array(e_init, dtype=float)
    n = len(mu)
    for it in range(MAX_ITER):
        if symmetric:
            b0 = step_fn(0, mu, it)
            br = np.full(n, b0)
        else:
            br = np.array([step_fn(i, mu, it) for i in range(n)])
        mu_new = DAMP * br + (1.0 - DAMP) * mu
        if np.max(np.abs(mu_new - mu)) < TOL:
            return mu_new
        mu = mu_new
    return mu


def mu_star_smoothed(kappa: float, e_init: np.ndarray, l: np.ndarray, k: np.ndarray,
                     w_h: float, w_l: float, q: float, bounds: Tuple[float, float],
                     symmetric: bool, *, seed0: int = 770_000) -> np.ndarray:
    def step(i, mu, it):
        return br_i_smoothed(i, mu, kappa, l, k, w_h, w_l, q, bounds,
                             seed=seed0 + int(kappa) * 1000 + it * 10 + i)
    return _fixed_point(step, e_init, symmetric)


def mu_star_deterministic(e_init: np.ndarray, l: np.ndarray, k: np.ndarray,
                          w_h: float, w_l: float, q: float,
                          bounds: Tuple[float, float], symmetric: bool,
                          *, seed0: int = 880_000) -> np.ndarray:
    def step(i, mu, it):
        _, br = exploitability_frozen_profile(
            np.array(mu, float), l, k, w_h, w_l, q, bounds,
            M=200_000, grid_step=0.25, seed=seed0 + it * 7 + i)
        return float(br[i])
    return _fixed_point(step, e_init, symmetric)


# -------------------------------- conditions --------------------------------
# (experiment, q, n, symmetric, k_vec, l_vec, w_h, w_l, e_star_avg, raw_ppo_avg)
def build_conditions() -> List[dict]:
    raw = {  # raw-PPO (TEL-PPO) cross-player mean landing, from paper/tables/final_summary.csv
        ("two_players", 35.0): 43.58, ("two_players", 45.0): 35.46, ("two_players", 55.0): 29.65,
        ("three_players", 35.0): 22.99, ("three_players", 55.0): 15.31,
        ("different_cost", 35.0): 32.47, ("different_cost", 55.0): 22.71,
        ("different_ability", 35.0): 43.99, ("different_ability", 55.0): 29.70,
    }
    specs = {
        "two_players":       dict(n=2, sym=True,  w_h=6.5, w_l=3.0, k=[0.00055, 0.00055], l=[0.0, 0.0], qs=[35.0, 45.0, 55.0]),
        "three_players":     dict(n=3, sym=True,  w_h=6.5, w_l=3.0, k=[0.001] * 3,        l=[0.0] * 3,  qs=[35.0, 55.0]),
        "different_cost":    dict(n=2, sym=False, w_h=8.0, w_l=5.5, k=[0.0004, 0.00055],  l=[0.0, 0.0], qs=[35.0, 55.0]),
        "different_ability": dict(n=2, sym=False, w_h=6.5, w_l=3.0, k=[0.0005, 0.0005],   l=[10.0, 5.0], qs=[35.0, 55.0]),
    }
    conds = []
    for exp, s in specs.items():
        for q in s["qs"]:
            conds.append(dict(
                exp=exp, q=q, n=s["n"], sym=s["sym"],
                k=np.array(s["k"]), l=np.array(s["l"]),
                w_h=s["w_h"], w_l=s["w_l"],
                e_star_avg=float(e_star_for_experiment(q, exp)),
                raw_ppo=raw[(exp, q)],
            ))
    return conds


def _verdict(raw: float, mu_lo: float, e_star: float) -> str:
    gap = e_star - mu_lo               # smoothing gap at low kappa
    # is the raw landing inside the smoothed band [min(mu_lo,e*), max(...)] (+/- tol)?
    lo_b, hi_b = min(mu_lo, e_star) - 0.4, max(mu_lo, e_star) + 0.4
    in_band = lo_b <= raw <= hi_b
    if abs(gap) < 0.4:
        return f"CLEAN (mu*~e*, gap {gap:+.2f}; raw{'OK' if in_band else ' OUT'})"
    tag = "CONSISTENT" if in_band else "MISMATCH"
    return f"{tag} (smoothing gap {gap:+.2f}; raw {'in' if in_band else 'OUT of'} band)"


def main() -> None:
    print("=" * 100)
    print("mu*(kappa) SCREEN across conditions — is raw PPO landing the smoothed equilibrium? ZERO GPU")
    print("=" * 100)
    conds = build_conditions()
    dump = []
    hdr = (f"{'condition':<22}{'e*':>7}{'raw':>7}{'mu*(20)':>9}{'mu*(60)':>9}"
           f"{'mu*(200)':>10}{'mu*(det)':>10}   verdict")
    print(hdr)
    print("-" * 100)
    for c in conds:
        einit = np.full(c["n"], c["raw_ppo"])   # seed iteration from EMPIRICAL landing
        mus = {}
        for kap in KAPPAS:
            v = mu_star_smoothed(kap, einit, c["l"], c["k"], c["w_h"], c["w_l"],
                                 c["q"], BOUNDS, c["sym"])
            mus[kap] = float(np.mean(v))
        det = mu_star_deterministic(einit, c["l"], c["k"], c["w_h"], c["w_l"],
                                    c["q"], BOUNDS, c["sym"])
        mus["det"] = float(np.mean(det))
        name = f"{c['exp']} q{c['q']:g}"
        verdict = _verdict(c["raw_ppo"], mus[20.0], c["e_star_avg"])
        print(f"{name:<22}{c['e_star_avg']:>7.2f}{c['raw_ppo']:>7.2f}"
              f"{mus[20.0]:>9.2f}{mus[60.0]:>9.2f}{mus[200.0]:>10.2f}"
              f"{mus['det']:>10.2f}   {verdict}")
        dump.append(dict(condition=name, e_star=round(c["e_star_avg"], 2),
                         raw_ppo=c["raw_ppo"], mu_star={str(k): round(v, 2) for k, v in mus.items()},
                         verdict=verdict))
    print("-" * 100)
    print("Reading: mu*(det) should ~ e* (self-check). If raw lands in the "
          "[mu*(20)..e*] band => Claim B fits (raw = smoothed equilibrium).")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "docs", "mu_star_screen.json")
    with open(os.path.abspath(out), "w") as f:
        json.dump(dump, f, indent=2)
    print(f"[dump] {os.path.abspath(out)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
