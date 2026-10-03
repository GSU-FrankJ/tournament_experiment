#!/usr/bin/env python3
"""Calibration of the verifier at the lock (no training): analytic equilibrium and zero effort.

For q in {50, 60} and the final and development tiers, ``utils.v2_metrics.evaluate`` on
  - the closed-form equilibrium (e1*, e2*) (tools/v2/common.py:analytic_policy), and
  - the zero-effort policy e_hat == 0,
reporting Gmax_full/DW with (t*, d*), eta_2, EXP_root, dReach, Delta_max_all and dFull, the stage-wise
G maxima, and the differences from the Phase 1 calibration (results/v2_pilots/phase1/calibration/
calibration.csv, base commit 657f54a). PI-side references for the zero policy (independent, no repo
code): Gmax_full/DW 0.2593 (q=50, stage 2, d ~ -26) and 0.1955 (q=60, stage 2, d ~ -23.5); root
dynamic gain 0.243 and 0.191.

Usage: python tools/v2/locked_calibration.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))

from common import analytic_policy, spec_for, zero_policy  # noqa: E402
from utils.dp_br_verifier import DEV_CONFIG, FINAL_CONFIG  # noqa: E402
from utils.v2_metrics import evaluate  # noqa: E402

OUT = ROOT / "results" / "v2_T2_locked" / "calibration"
P1 = ROOT / "results" / "v2_pilots" / "phase1" / "calibration" / "calibration.csv"
KEYS = ("Gmax_full_over_dw", "Gmax_full_t", "Gmax_full_d", "G_max_t1_over_dw", "G_argmax_d_t1", "G_max_t2_over_dw",
        "G_argmax_d_t2", "eta_T_over_dw", "EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw", "dFull_over_dw",
        "valid")
PI_REF = {50: {"Gmax_full_over_dw": 0.2593, "t": 2, "d": -26.0, "root_gain": 0.243},
          60: {"Gmax_full_over_dw": 0.1955, "t": 2, "d": -23.5, "root_gain": 0.191}}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).decode().strip()
    p1 = pd.read_csv(P1)
    rows = []
    for q in (50, 60):
        spec = spec_for(q)
        for name, pol in (("analytic_eq", analytic_policy(spec)), ("zero", zero_policy)):
            for tier in (FINAL_CONFIG, DEV_CONFIG):
                s = evaluate(pol, spec, tier).scalars
                r = {"commit": commit, "q": q, "policy": name, "tier": tier.name, **{k: s[k] for k in KEYS}}
                old = p1[(p1.q == q) & (p1.policy == name) & (p1.verifier_tier == tier.name)].iloc[0]
                for k in KEYS:
                    if k != "valid":
                        r[f"phase1_{k}"] = old[k]
                        r[f"diff_vs_phase1_{k}"] = float(s[k]) - float(old[k])
                if name == "zero":
                    ref = PI_REF[q]
                    r.update(pi_ref_Gmax=ref["Gmax_full_over_dw"], pi_ref_t=ref["t"], pi_ref_d=ref["d"],
                             pi_ref_root_gain=ref["root_gain"], diff_vs_pi_Gmax=float(s["Gmax_full_over_dw"]) - ref["Gmax_full_over_dw"],
                             diff_vs_pi_root_gain=float(s["EXP_root_over_dw"]) - ref["root_gain"])
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "calibration_locked.csv", index=False)
    pd.set_option("display.width", 250)
    print(df[["q", "policy", "tier", "Gmax_full_over_dw", "Gmax_full_t", "Gmax_full_d", "eta_T_over_dw", "EXP_root_over_dw",
              "dReach_over_dw", "Deltamax_all_over_dw", "dFull_over_dw", "diff_vs_phase1_Gmax_full_over_dw",
              "diff_vs_phase1_EXP_root_over_dw", "diff_vs_phase1_dReach_over_dw"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
