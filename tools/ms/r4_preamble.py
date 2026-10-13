#!/usr/bin/env python3
"""Reproduction of the PI-side preamble numbers of the MS-R4 prompt (section 2.3 (a)); offline, no training.

  * **Strata shares** (adjustment 1): arm means of ``t2_share_near_all`` / ``t2_share_mid_all`` of MS-R1 ``MS_s35a0`` and
    ``MS_s35a5`` (``results/ms_r1/analysis/per_run.csv``) and MS-R3 ``t1_st_s16`` (``results/ms_r3/analysis/per_run.csv``), both q.
  * **MS-R1 areas** (adjustment 5): per run of the arms ``MS_s25a0``, ``MS_s25a5``, ``MS_s35a0``, ``MS_s35a5`` (10 seeds per q)
    the rows of ``ms_checks_stage2.csv`` with a finite ``Delta`` and ``local`` <= 1600, the trapezoidal area of ``Delta`` over
    ``local``, the share of that area over local <= 200, and per q and lambda_P in {0.25, 0.35}
    ``1 - mean(A_a5) / mean(A_a0)`` with the paired percentile bootstrap interval over the full window and over local >= 200.
    The PI-side intervals were drawn from ONE ``numpy.random.default_rng(12345)`` in this order: q in (50, 60), lambda_P 0.25
    then 0.35, the full window then the window >= 200, each interval ``rng.integers(0, 10, size=(10000, 10))`` over the seeds
    sorted ascending, 2.5 / 97.5 percentiles. That draw is reproduced exactly, and next to it the intervals under the
    convention of D6 (a fresh generator per statistic; seed 20261014, the E2 seed).
  * **Check cost** (adjustment 7): the median ``verifier_sec`` over the stage-2 checks and the median ``update_wall_sec`` over the
    stage-2 updates of the MS-R1 pilot (all arms), and the share of one check per 25 updates.
  * **Zero-failure power** (adjustment 6): the exact Clopper-Pearson 95 % upper limit of 0 failures in 40 runs, 0.9^20, and the
    1-of-10 interval.

A number that differs from the prompt is reported as such (a difference is not a stop condition). Outputs: ``--out-json`` and
``--out-md``.

Usage:
    python tools/ms/r4_preamble.py --ms-r1-root results/ms_r1 --ms-r3-root results/ms_r3 \
        --out-json results/ms_r4/preamble/preamble.json --out-md reports/ms/r4/01_preamble_numbers.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))

import r4_common as C  # noqa: E402

QS = (50, 60)
SEEDS = tuple(range(10501, 10511))
AREA_ARMS = {0.25: ("MS_s25a0", "MS_s25a5"), 0.35: ("MS_s35a0", "MS_s35a5")}
PILOT_ARMS_ALL = ("MS_base2400", "MS_rule", "MS_s25a0", "MS_s25a5", "MS_s35a0", "MS_s35a5")
PI_RNG_SEED = 12345
D6_SEED = 20261014
N_BOOT = 10000
LOCAL_MAX = 1600
#: the numbers as the prompt states them (PI-side)
PI_SHARES = {("MS_s35a0", 50): (0.348, 0.152), ("MS_s35a0", 60): (0.310, 0.236), ("MS_s35a5", 50): (0.234, 0.266),
             ("MS_s35a5", 60): (0.259, 0.286), ("t1_st_s16", 50): (0.232, 0.268), ("t1_st_s16", 60): (0.236, 0.309)}
PI_REDUCTION = {("full", 50, 0.25): (-0.067, -0.110, -0.029), ("full", 50, 0.35): (0.067, -0.000, 0.131),
                ("full", 60, 0.25): (-0.192, -0.245, -0.150), ("full", 60, 0.35): (-0.146, -0.171, -0.117),
                ("late", 50, 0.25): (0.088, -0.000, 0.155), ("late", 50, 0.35): (0.311, 0.219, 0.390),
                ("late", 60, 0.25): (-0.044, -0.183, 0.076), ("late", 60, 0.35): (0.078, 0.043, 0.106)}
PI_SHARE_200 = {50: (0.61, 0.66, 0.49, 0.62), 60: (0.72, 0.75, 0.67, 0.73)}      # MS_s25a0, MS_s25a5, MS_s35a0, MS_s35a5
PI_COST = {"verifier_sec_median": 0.0165, "n_checks": 11520, "update_wall_sec_median": 0.161, "n_updates": 288000,
           "check_share": 0.004}


def run_area(csv_path: Path, first: int = 0) -> Tuple[float, float]:
    """``(area over local in [first, 1600], area over local <= 200)`` of ``Delta`` from a stage-2 check table (finite rows)."""
    ck = pd.read_csv(csv_path, usecols=["local", "Delta"])
    ck = ck[np.isfinite(ck["Delta"].astype(float)) & (ck["local"] <= LOCAL_MAX)].sort_values("local")
    full = C.trapezoid(ck["local"], ck["Delta"]) if len(ck) > 1 else float("nan")
    win = ck[ck["local"] >= first]
    a_win = C.trapezoid(win["local"], win["Delta"]) if len(win) > 1 else float("nan")
    early = ck[ck["local"] <= 200]
    a_early = C.trapezoid(early["local"], early["Delta"]) if len(early) > 1 else float("nan")
    return a_win, a_early / full if full else float("nan")


def areas_table(ms_r1_root: Path) -> pd.DataFrame:
    """Per run: the full area, the area over local >= 200 and the share of the area over local <= 200."""
    rows = []
    for arm in sum((list(v) for v in AREA_ARMS.values()), []):
        for q in QS:
            for s in SEEDS:
                p = ms_r1_root / "pilot" / f"q{q}" / f"seed{s}" / arm / "ms_checks_stage2.csv"
                a_full, share200 = run_area(p, 0)
                a_late, _ = run_area(p, 200)
                rows.append({"arm": arm, "q": q, "seed": s, "area_full": a_full, "area_late": a_late, "share_first200": share200})
    return pd.DataFrame(rows)


def reductions(areas: pd.DataFrame) -> pd.DataFrame:
    """The reduction table: PI-side draw (one generator, PI order), the D6 convention (fresh generator), the PI numbers."""
    rng = np.random.default_rng(PI_RNG_SEED)
    rows = []
    for q in QS:
        for lam, (a0, a5) in AREA_ARMS.items():
            for win, col in (("full", "area_full"), ("late", "area_late")):
                x0 = areas[(areas["arm"] == a0) & (areas["q"] == q)].sort_values("seed")[col].to_numpy(float)
                x5 = areas[(areas["arm"] == a5) & (areas["q"] == q)].sort_values("seed")[col].to_numpy(float)
                idx = rng.integers(0, 10, size=(N_BOOT, 10))              # the PI-side draw: one shared generator
                rho_b = 1.0 - x5[idx].mean(axis=1) / x0[idx].mean(axis=1)
                pi_lo, pi_hi = float(np.percentile(rho_b, 2.5)), float(np.percentile(rho_b, 97.5))
                d6 = C.boot_reduction(x5, x0, N_BOOT, D6_SEED)
                ref = PI_REDUCTION[(win, q, lam)]
                rows.append({"window": win, "q": q, "lambda_P": lam, "rho": d6["rho"], "pi_draw_lo": pi_lo, "pi_draw_hi": pi_hi,
                             "d6_lo": d6["intervals"][0][0], "d6_hi": d6["intervals"][0][1], "prompt_rho": ref[0],
                             "prompt_lo": ref[1], "prompt_hi": ref[2],
                             "matches_prompt_to_3dp": bool(all(abs(round(v, 3) - r) < 1e-9 for v, r in
                                                               zip((d6["rho"], pi_lo, pi_hi), ref)))})
    return pd.DataFrame(rows)


def shares_table(ms_r1_root: Path, ms_r3_root: Path) -> pd.DataFrame:
    """Arm means of the realised near-tie and middle shares (``t2_share_near_all`` / ``t2_share_mid_all``)."""
    r1 = pd.read_csv(ms_r1_root / "analysis" / "per_run.csv")
    r3 = pd.read_csv(ms_r3_root / "analysis" / "per_run.csv")
    rows = []
    for (arm, q), ref in PI_SHARES.items():
        d = (r3 if arm == "t1_st_s16" else r1)
        g = d[(d["arm"] == arm) & (d["q"] == q) & d["status"].astype(str).eq("done")]
        rows.append({"arm": arm, "q": q, "n_runs": len(g), "near": float(g["t2_share_near_all"].mean()),
                     "mid": float(g["t2_share_mid_all"].mean()), "prompt_near": ref[0], "prompt_mid": ref[1],
                     "matches_prompt_to_3dp": bool(abs(round(float(g["t2_share_near_all"].mean()), 3) - ref[0]) < 1e-9
                                                   and abs(round(float(g["t2_share_mid_all"].mean()), 3) - ref[1]) < 1e-9)})
    return pd.DataFrame(rows)


def cost_numbers(ms_r1_root: Path) -> Dict[str, Any]:
    """Median check and update times of the MS-R1 pilot (all arms), stage 2 only."""
    v: List[np.ndarray] = []
    u: List[np.ndarray] = []
    for arm in PILOT_ARMS_ALL:
        for q in QS:
            for s in SEEDS:
                d = ms_r1_root / "pilot" / f"q{q}" / f"seed{s}" / arm
                ck = pd.read_csv(d / "ms_checks_stage2.csv", usecols=["verifier_sec"])
                v.append(ck["verifier_sec"].to_numpy(float))
                up = pd.read_csv(d / "ms_updates.csv", usecols=["stage", "update_wall_sec"])
                u.append(up[up["stage"] == 2]["update_wall_sec"].to_numpy(float))
    va, ua = np.concatenate(v), np.concatenate(u)
    out = {"verifier_sec_median": float(np.median(va)), "n_checks": int(va.size), "update_wall_sec_median": float(np.median(ua)),
           "n_updates": int(ua.size)}
    out["check_share"] = out["verifier_sec_median"] / (25.0 * out["update_wall_sec_median"])
    return out


def power_numbers() -> Dict[str, Any]:
    """Adjustment 6: the zero-failure bound and the pass probability."""
    return {"cp95_upper_0_of_40": C.clopper_pearson(0, 40)[1], "pass_all_20_at_rate_0.1": 0.9 ** 20,
            "cp95_1_of_10": list(C.clopper_pearson(1, 10))}


def _fmt(x: float, nd: int = 3) -> str:
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{nd}f}"


def markdown(shares: pd.DataFrame, areas: pd.DataFrame, red: pd.DataFrame, cost: Dict[str, Any], power: Dict[str, Any],
             roots: Dict[str, str]) -> str:
    """The report text (every number is printed from the tables above)."""
    L = ["# MS-R4 preamble numbers: reproduction of the PI-side values", "",
         "Generated by `tools/ms/r4_preamble.py` (offline, no training). A number that differs from the prompt is reported as "
         "such; a difference is not a stop condition.", "", f"Roots: `{roots['ms_r1']}`, `{roots['ms_r3']}`.", "",
         "## 1. Strata shares (adjustment 1)", "",
         "Arm means of the realised shares over the terminal stage (`t2_share_near_all`, `t2_share_mid_all`); source "
         "`results/ms_r1/analysis/per_run.csv` and `results/ms_r3/analysis/per_run.csv`.", "",
         "| arm | q | runs | near | mid | prompt near | prompt mid | match (3 dp) |", "|---|---|---|---|---|---|---|---|"]
    for r in shares.itertuples(index=False):
        L.append(f"| {r.arm} | {r.q} | {r.n_runs} | {r.near:.3f} | {r.mid:.3f} | {r.prompt_near:.3f} | {r.prompt_mid:.3f} | "
                 f"{'yes' if r.matches_prompt_to_3dp else 'NO'} |")
    L += ["", "## 2. MS-R1 areas of the development-tier eta_2/DW (adjustment 5)", "",
          "Per run, the trapezoidal area of `Delta` over `local` on the rows of `ms_checks_stage2.csv` with a finite `Delta` and "
          "`local` <= 1600 (`results/ms_r1/pilot/q*/seed*/<arm>/ms_checks_stage2.csv`, 10 seeds per q).", "",
          "Share of the area over local <= 200 (arm means of the per-run share; prompt: 0.61, 0.66, 0.49, 0.62 at q = 50 and "
          "0.72, 0.75, 0.67, 0.73 at q = 60):", "", "| q | MS_s25a0 | MS_s25a5 | MS_s35a0 | MS_s35a5 | range over runs |", "|---|---|---|---|---|---|"]
    for q in QS:
        means = [float(areas[(areas["arm"] == a) & (areas["q"] == q)]["share_first200"].mean())
                 for a in ("MS_s25a0", "MS_s25a5", "MS_s35a0", "MS_s35a5")]
        sub = areas[areas["q"] == q]["share_first200"]
        L.append(f"| {q} | " + " | ".join(f"{m:.2f}" for m in means) + f" | {sub.min():.2f}-{sub.max():.2f} |")
    L += ["", "Reduction `1 - mean(A_alpha0.5) / mean(A_alpha0)` (negative = the area grew), with the paired percentile bootstrap "
          "interval. `PI draw` = one `default_rng(12345)` shared in the PI-side order; `D6` = a fresh `default_rng(20261014)` per "
          "statistic. The prompt's numbers are in the last column.", "",
          "| window | q | lambda_P | rho | PI draw 95 % | D6 95 % | prompt | match (3 dp) |", "|---|---|---|---|---|---|---|---|"]
    for r in red.itertuples(index=False):
        L.append(f"| {'full' if r.window == 'full' else 'local 200-1600'} | {r.q} | {r.lambda_P} | {_fmt(r.rho)} | "
                 f"[{_fmt(r.pi_draw_lo)}, {_fmt(r.pi_draw_hi)}] | [{_fmt(r.d6_lo)}, {_fmt(r.d6_hi)}] | "
                 f"{_fmt(r.prompt_rho)} [{_fmt(r.prompt_lo)}, {_fmt(r.prompt_hi)}] | {'yes' if r.matches_prompt_to_3dp else 'NO'} |")
    L += ["", "## 3. Cost of the checks (adjustment 7)", "",
          f"MS-R1 pilot, all six arms, stage 2 only (`results/ms_r1/pilot/q*/seed*/*/ms_checks_stage2.csv`, `ms_updates.csv`): "
          f"median `verifier_sec` {cost['verifier_sec_median']:.4f} s (n = {cost['n_checks']:,}; prompt 0.0165 s, n = 11,520); median "
          f"`update_wall_sec` {cost['update_wall_sec_median']:.4f} s (n = {cost['n_updates']:,}; prompt 0.161 s, n = 288,000); one check "
          f"per 25 updates = {100 * cost['check_share']:.2f} % of the stage-2 wall time (prompt about 0.4 %).", "",
          "## 4. Power of the zero-failure target (adjustment 6)", "",
          f"Zero failures in 40 runs: exact Clopper-Pearson 95 % upper limit {power['cp95_upper_0_of_40']:.4f} (prompt 8.8 %). A candidate "
          f"with a per-run failure rate of 0.1 passes all 20 q = 50 runs with probability 0.9^20 = {power['pass_all_20_at_rate_0.1']:.4f} "
          f"(prompt 0.12). One failure in 10 runs: CP 95 % interval [{power['cp95_1_of_10'][0]:.4f}, {power['cp95_1_of_10'][1]:.4f}]."]
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    p.add_argument("--ms-r1-root", required=True, help="results/ms_r1 (pilot/, analysis/per_run.csv)")
    p.add_argument("--ms-r3-root", required=True, help="results/ms_r3 (analysis/per_run.csv)")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    a = p.parse_args(argv)
    r1, r3 = Path(a.ms_r1_root), Path(a.ms_r3_root)
    shares = shares_table(r1, r3)
    areas = areas_table(r1)
    red = reductions(areas)
    cost = cost_numbers(r1)
    power = power_numbers()
    res = {"tool": "tools/ms/r4_preamble.py", "roots": {"ms_r1": str(r1), "ms_r3": str(r3)},
           "shares": shares.to_dict("records"), "reductions": red.to_dict("records"), "cost": cost, "power": power}
    res["share_first200_by_arm_q"] = {f"{x}|{y}": float(areas[(areas["arm"] == x) & (areas["q"] == y)]["share_first200"].mean())
                                      for x in ("MS_s25a0", "MS_s25a5", "MS_s35a0", "MS_s35a5") for y in QS}
    res["share_first200_range"] = [float(areas["share_first200"].min()), float(areas["share_first200"].max())]
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(res, f, indent=1)
    Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_md).write_text(markdown(shares, areas, red, cost, power, res["roots"]))
    n_mis = int((~shares["matches_prompt_to_3dp"]).sum()) + int((~red["matches_prompt_to_3dp"]).sum())
    print(json.dumps({"n_numbers_that_differ_from_the_prompt": n_mis, "cost": cost, "power": power}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
