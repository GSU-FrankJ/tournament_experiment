#!/usr/bin/env python3
"""Opening check 1a: independent reference for the terminal one-step gap Delta_2(d).

Delta_2^ref(d) = max_{e in [0,100]} [DW F_xi(d + e - e2(-d)) - k e^2]
                 - [DW F_xi(d + e2(d) - e2(-d)) - k e2(d)^2]          (w_l cancels)

The reference imports ONLY ``F_xi`` (validated against the environment by Monte Carlo in
Phase 1) and ``g2_two_stage`` (to build the test policies). It reuses no verifier search,
quadrature or interpolation code. Search: dense effort grid of ``--n-dense`` points on [0, 100]
(default 100001, spacing 0.001), then bounded scalar maximization (scipy ``minimize_scalar``,
method 'bounded', xatol 1e-12) on [e_j - h, e_j + h] around the best dense point e_j; the
larger of the dense and refined values is kept.

Usage: python tools/v2/reference_delta2.py --out results/v2_pilots/phase2_opening/delta2_reference.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.optimize import minimize_scalar

from common import DEV_2X, K, W_H, W_L, spec_for  # noqa: E402
from utils.dp_br_verifier import DEV_CONFIG  # noqa: E402
from utils.theory_multistage import F_xi, g1_two_stage, g2_two_stage  # noqa: E402
from utils.v2_metrics import append_csv, evaluate  # noqa: E402


def delta2_reference(e2, d_grid: np.ndarray, q: float, dw: float, k: float, n_dense: int):
    """Return (delta_ref, best_e, refine_gain) arrays over ``d_grid`` (see module docstring)."""
    E = np.linspace(0.0, 100.0, n_dense)
    h = E[1] - E[0]
    out, best_e, gain = np.empty(d_grid.size), np.empty(d_grid.size), np.empty(d_grid.size)
    for i, d in enumerate(d_grid):
        opp = float(e2(np.array([-d]))[0])
        own = float(e2(np.array([d]))[0])

        def obj(e):
            return dw * F_xi(np.atleast_1d(d + e - opp), q) - k * np.atleast_1d(e) ** 2
        vals = obj(E)
        j = int(np.argmax(vals))
        lo, hi = max(0.0, E[j] - h), min(100.0, E[j] + h)
        r = minimize_scalar(lambda e: -float(obj(e)[0]), bounds=(lo, hi), method="bounded",
                            options={"xatol": 1e-12})
        v_ref = max(float(vals[j]), -float(r.fun))
        best_e[i] = float(r.x) if -float(r.fun) > vals[j] else float(E[j])
        gain[i] = v_ref - float(vals[j])
        out[i] = v_ref - float(obj(own)[0])
    return out, best_e, gain


def main() -> int:
    """Compare the verifier's Delta_2 with the reference on both tiers, q in {50, 60}."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n-dense", type=int, default=100001)
    a = p.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for q in (50.0, 60.0):
        spec = spec_for(q)
        g1 = g1_two_stage(q, W_H, W_L, K)
        pols = {
            "zero": lambda d: np.zeros(np.asarray(d).shape),
            "eq_shift_q_over_2": lambda d, q=q: g2_two_stage(np.asarray(d, float) - q / 2, q, W_H, W_L, K, 100.0),
        }
        for name, e2 in pols.items():
            def policy(t, d, e2=e2):
                d = np.asarray(d, float)
                return np.full(d.shape, g1) if t == 1 else e2(d)
            for cfg in (DEV_CONFIG, DEV_2X):
                ev = evaluate(policy, spec, cfg)
                s2 = ev.res.stages[2]
                ref, best_e, gain = delta2_reference(e2, s2.d_grid, q, spec.dw, spec.k, a.n_dense)
                diff = (s2.delta - ref) / spec.dw
                for i, d in enumerate(s2.d_grid):
                    append_csv(a.out, {"q": q, "policy": name, "tier": cfg.name,
                                       "effort_step": cfg.effort_step, "state_step": cfg.state_step,
                                       "d": d, "delta_verifier_over_dw": s2.delta[i] / spec.dw,
                                       "delta_ref_over_dw": ref[i] / spec.dw, "diff_over_dw": diff[i],
                                       "a_dev_verifier": s2.a_dev[i], "a_ref": best_e[i],
                                       "refine_gain_over_dw": gain[i] / spec.dw})
                j = int(np.argmax(np.abs(diff)))
                print(f"q={q:g} {name:18s} {cfg.name:12s} max|diff|/DW={abs(diff[j]):.3e} at d={s2.d_grid[j]:g} "
                      f"max(+diff)={diff.max():.3e} min(diff)={diff.min():.3e} n(+>1e-15)={(diff > 1e-15).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
