#!/usr/bin/env python3
"""Monte Carlo check that the closed-form terminal CDF matches the environment's noise model.

For each (d, e_i, e_j) on a grid, N terminal transitions are simulated with the environment's
own code path (``step_gap`` + ``GameSpec.terminal_reward``, shocks U(-q, q) per player as in
``collect_batch``), for both players, and compared with F_xi(d + e_i - e_j) (player i) and
F_xi(-(d + e_i - e_j)) (player j). The mean sampled terminal stage reward is also compared with
w_l + DW F_xi(.) - k e^2 (z-score; for a constant reward column
the raw difference r_diff is the result and r_z is NaN). Separate numpy seed (9100000 + q); no training stream is used.

Usage: python tools/v2/benchmark_consistency.py --out results/v2_pilots/phase1/benchmark_consistency.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from common import spec_for  # noqa: E402
from envs.curriculum_env import stage_reward, step_gap  # noqa: E402
from utils.theory_multistage import F_xi  # noqa: E402
from utils.v2_metrics import append_csv  # noqa: E402


def main() -> int:
    """Write one CSV row per (q, d, e_i, e_j) and print the per-q maxima."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=200_000)
    a = p.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    efforts = np.arange(0.0, 100.0 + 1e-9, 20.0)
    for q in (50.0, 60.0):
        spec = spec_for(q)
        rng = np.random.default_rng(9100000 + int(q))
        D = np.linspace(-spec.B, spec.B, 21)
        zmax = (0.0, None)
        for d in D:
            for ei in efforts:
                for ej in efforts:
                    n = a.n
                    eps_i = rng.uniform(-q, q, size=n)
                    eps_j = rng.uniform(-q, q, size=n)
                    dn_i = step_gap(spec, np.full(n, d), np.full(n, ei), np.full(n, ej), eps_i, eps_j)
                    dn_j = step_gap(spec, np.full(n, -d), np.full(n, ej), np.full(n, ei), eps_j, eps_i)
                    win_i = (spec.terminal_reward(dn_i) - spec.w_l) / spec.dw
                    win_j = (spec.terminal_reward(dn_j) - spec.w_l) / spec.dw
                    r_i = stage_reward(spec, 2, np.full(n, ei), dn_i)
                    r_j = stage_reward(spec, 2, np.full(n, ej), dn_j)
                    y = d + ei - ej
                    row = {"q": q, "d": d, "e_i": ei, "e_j": ej, "n": n}
                    for who, w, r, yy, e in (("i", win_i, r_i, y, ei), ("j", win_j, r_j, -y, ej)):
                        F = float(F_xi(np.array([yy]), q)[0])
                        se = np.sqrt(F * (1 - F) / n)
                        diff = float(w.mean()) - F
                        z = diff / se if se > 0 else (0.0 if diff == 0 else np.inf)
                        rbar = spec.w_l + spec.dw * F - spec.k * e ** 2
                        rdiff = float(r.mean()) - rbar
                        # a constant reward column (F in {0, 1}) has no MC error: compare exactly
                        r_se = 0.0 if np.ptp(r) == 0 else float(r.std(ddof=1) / np.sqrt(n))
                        row.update({f"F_{who}": F, f"p_mc_{who}": float(w.mean()), f"diff_{who}": diff,
                                    f"se_{who}": se, f"z_{who}": z, f"rbar_{who}": rbar,
                                    f"r_mc_{who}": float(r.mean()), f"r_diff_{who}": rdiff,
                                    f"r_z_{who}": (rdiff / r_se) if r_se > 0 else np.nan,
                                    f"r_const_{who}": bool(r_se == 0)})
                        if abs(z) > abs(zmax[0]):
                            zmax = (z, (d, ei, ej, who))
                    append_csv(a.out, row)
        print(f"q={q}: max |z| = {abs(zmax[0]):.3f} at (d, e_i, e_j, player) = {zmax[1]}")
        # recheck of the max-|z| cell with 40 fresh streams (seeds 9200000 + 100 q + s)
        d, ei, ej, who = zmax[1]
        if who == "j":
            d, ei, ej = -d, ej, ei
        F = float(F_xi(np.array([d + ei - ej]), q)[0])
        zs = []
        for s in range(40):
            r2 = np.random.default_rng(9200000 + 100 * int(q) + s)
            e_a = r2.uniform(-q, q, size=a.n)
            e_b = r2.uniform(-q, q, size=a.n)
            dn = step_gap(spec, np.full(a.n, d), np.full(a.n, ei), np.full(a.n, ej), e_a, e_b)
            p_ = float(((spec.terminal_reward(dn) - spec.w_l) / spec.dw).mean())
            zs.append((p_ - F) / np.sqrt(F * (1 - F) / a.n))
        zs = np.array(zs)
        append_csv(os.path.join(os.path.dirname(a.out), "benchmark_recheck.csv"),
                   {"q": q, "cell_d_own": d, "cell_e_own": ei, "cell_e_opp": ej, "F": F, "n_per_rep": a.n,
                    "reps": 40, "z_mean": float(zs.mean()), "z_sd": float(zs.std(ddof=1)),
                    "z_mean_over_se": float(zs.mean() / (zs.std(ddof=1) / np.sqrt(40))),
                    "max_abs_z": float(np.abs(zs).max())})
        print(f"   recheck 40 fresh reps: z mean {zs.mean():.3f}, sd {zs.std(ddof=1):.3f}, max|z| {np.abs(zs).max():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
