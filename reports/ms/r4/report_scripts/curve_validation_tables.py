#!/usr/bin/env python3
"""Tables of reports/ms/r4/01b_curve_tool.md from results/ms_r4/curves_validation/ (the offline curve tool on MS-R3's t1_st_s16).

Usage: python reports/ms/r4/report_scripts/curve_validation_tables.py [--root results/ms_r4/curves_validation]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ETA, DEV = "final_eta_T_over_dw", "dev_eta_T_over_dw"


def trap(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/ms_r4/curves_validation")
    a = ap.parse_args()
    root = Path(a.root)
    c = pd.read_csv(root / "c_off.csv")
    cur = pd.read_csv(root / "curves_all.csv")
    info = json.load(open(root / "curves_info.json"))
    print("## C-OFF (development tier offline against the online checks; last point against the freeze record)")
    print(f"runs {len(c)}, ok {int(c['ok'].sum())}, exports {int(c['n_exports'].sum())}, compared values {int(c['n_compared'].sum())}, "
          f"mismatches {int(c['n_mismatch'].sum())}, max abs diff {c['max_abs_diff'].max()}, last final eta == freeze {int(c['last_final_eta_equals_freeze'].sum())}, "
          f"last dev eta == freeze {int(c['last_dev_eta_equals_freeze'].sum())}, last final |peak| == freeze {int(c['last_final_peak_equals_freeze'].sum())}")
    print(f"evaluation seconds: final tier {info['eval_sec_final_total']:.1f} (sum over runs), development tier {info['eval_sec_dev_total']:.1f}, wall {info['wall_sec']:.1f} with {info['workers']} workers")
    print()
    rows = []
    for (q, s), g in cur.groupby(["q", "seed"]):
        g = g.sort_values("update")
        g64 = g[g["update"] <= 1600]
        u, yf, yd = g64["update"].to_numpy(float), g64[ETA].to_numpy(float), g64[DEV].to_numpy(float)
        m200 = u <= 200
        rows.append({"q": q, "seed": s, "A_final": trap(u, yf) / 1575.0, "A_dev": trap(u, yd) / 1575.0,
                     "share_first200_final": trap(u[m200], yf[m200]) / trap(u, yf), "share_first200_dev": trap(u[m200], yd[m200]) / trap(u, yd),
                     "eta_final_u1600": yf[-1], "eta_dev_u1600": yd[-1], "ratio_dev_over_final_median": float(np.median(yd / yf)),
                     "ratio_dev_over_final_min": float(np.min(yd / yf)), "ratio_dev_over_final_max": float(np.max(yd / yf)),
                     "first_final_le_0.005": float(u[np.flatnonzero(yf <= 0.005)[0]]) if (yf <= 0.005).any() else float("nan")})
    t = pd.DataFrame(rows)
    print("## Final-tier curves of the 20 runs on the 64 checkpoints u25 ... u1600 (the E2 grid), per q")
    print("| q | runs | mean A (final) | mean A (dev) | share of the final-tier area over u <= 200: mean [min, max] | dev share: mean | eta final at u1600: median | dev / final: median ratio per run (min over runs - max over runs) | first export with eta <= 0.005: median |")
    print("|---|---|---|---|---|---|---|---|---|")
    for q, g in t.groupby("q"):
        print(f"| {q} | {len(g)} | {g['A_final'].mean():.5f} | {g['A_dev'].mean():.5f} | {g['share_first200_final'].mean():.3f} [{g['share_first200_final'].min():.3f}, {g['share_first200_final'].max():.3f}] | "
              f"{g['share_first200_dev'].mean():.3f} | {g['eta_final_u1600'].median():.2e} | {g['ratio_dev_over_final_median'].median():.3f} ({g['ratio_dev_over_final_min'].min():.3f} - {g['ratio_dev_over_final_max'].max():.3f}) | {g['first_final_le_0.005'].median():.0f} |")
    allg = t
    print(f"| both | {len(allg)} | {allg['A_final'].mean():.5f} | {allg['A_dev'].mean():.5f} | {allg['share_first200_final'].mean():.3f} [{allg['share_first200_final'].min():.3f}, {allg['share_first200_final'].max():.3f}] | {allg['share_first200_dev'].mean():.3f} | {allg['eta_final_u1600'].median():.2e} | | {allg['first_final_le_0.005'].median():.0f} |")
    print()
    print("## Median final-tier eta_2/DW at selected updates (seed median)")
    for q, g in cur.groupby("q"):
        med = g.groupby("update")[[ETA, DEV]].median()
        sel = [25, 50, 100, 200, 400, 800, 1200, 1600, 2000, 2800]
        print(f"q={q}: " + "; ".join(f"u{u}: final {med.loc[u, ETA]:.2e} / dev {med.loc[u, DEV]:.2e}" for u in sel))
    g = cur[cur["update"] <= 1600].copy()
    g["ratio"] = g[DEV] / g[ETA]
    low = g[g["ratio"] < 0.9]
    print()
    print("## Development tier against final tier on the 64-point grid")
    print(f"points {len(g)}; ratio < 0.9 at {len(low)} points, all with final-tier eta <= {low[ETA].max():.2e}; smallest ratio {g['ratio'].min():.3f}; "
          f"quantiles 1% {g['ratio'].quantile(0.01):.3f}, 5% {g['ratio'].quantile(0.05):.3f}, 50% {g['ratio'].median():.4f}; largest ratio {g['ratio'].max():.13f}")
    t.to_csv(root / "curve_validation_per_run.csv", index=False)


if __name__ == "__main__":
    main()
