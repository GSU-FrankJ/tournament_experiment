#!/usr/bin/env python3
"""Report-only check of the dReach reach mask R_2 against the continuous root-BR support.

Reference support: {d in D_2 : |d - drift_BR| < 2q}, drift_BR = a_BR_1(0) - e_hat_1(0) (the
deviator's stage-1 BR effort minus the opponent's e_hat_1(-0) = e_hat_1(0)). The official
dReach (utils/dp_br_verifier.py, interval mask) is NOT changed; an alternative value with the
reference support as the stage-2 mask is computed next to it for comparison only.

Usage: python tools/v2/dreach_mask_check.py --raw /home/fjiang4/tournament_experiment/experiments \
    --out results/v2_pilots/dreach_mask_check/dreach_mask_check.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from common import analytic_policy, checkpoint_policy, spec_for, zero_policy  # noqa: E402
from utils.dp_br_verifier import DEV_CONFIG, FINAL_CONFIG, verify  # noqa: E402
from utils.v2_metrics import append_csv  # noqa: E402

COHORTS = ("two_stage_confirmation_T2_20260922", "two_stage_E1_q50_p_20260923",
           "two_stage_q50_restarts_20260924", "two_stage_q50_precision_20260924")


def main() -> int:
    """One CSV row per (candidate, tier)."""
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cands = []
    for q in (50.0, 60.0):
        spec = spec_for(q)
        cands += [("analytic", q, analytic_policy(spec)), ("zero", q, zero_policy)]
    for coh in COHORTS:
        for ck in sorted(glob.glob(os.path.join(a.raw, coh, "runs", "*", "*", "checkpoint.pt"))):
            cfg = json.load(open(os.path.join(os.path.dirname(ck), "config.json")))
            cands.append((f"{coh}/{cfg['run']}", float(cfg["q"]), checkpoint_policy(ck, spec_for(cfg["q"]))[0]))
    for name, q, pol in cands:
        spec = spec_for(q)
        for tier in (DEV_CONFIG, FINAL_CONFIG):
            res = verify(pol, w_h=spec.w_h, w_l=spec.w_l, k=spec.k, q=q, T=2, e_min=0.0, e_max=100.0, cfg=tier)
            s1, s2 = res.stages[1], res.stages[2]
            g = s2.d_grid
            drift_br = float(s1.a_br[0] - s1.e_opp[0])
            ref = np.abs(g - drift_br) < 2.0 * q
            R = s2.reach
            holes, extra = g[ref & ~R], g[R & ~ref]
            alt = float(s1.delta[0] + (s2.delta[ref].max() if ref.any() else np.nan))
            append_csv(a.out, {
                "candidate": name, "q": q, "tier": tier.name, "valid": res.valid, "n_grid": g.size,
                "a_br_1": float(s1.a_br[0]), "e_hat_1": float(s1.e_hat[0]), "drift_BR": drift_br,
                "n_R2": int(R.sum()), "n_ref": int(ref.sum()), "n_holes": int(holes.size), "n_extra": int(extra.size),
                "holes_d": " ".join(f"{x:g}" for x in holes), "extra_d": " ".join(f"{x:g}" for x in extra),
                "extra_dist_to_support_edge": " ".join(f"{abs(abs(x - drift_br) - 2 * q):.6g}" for x in extra),
                "dreach_official_over_dw": res.dreach / res.dw, "dreach_refmask_over_dw": alt / res.dw,
                "dreach_official_minus_refmask_over_dw": (res.dreach - alt) / res.dw,
                "max_delta2_R2_over_dw": res.reach_delta_max[2] / res.dw,
                "max_delta2_ref_over_dw": float(s2.delta[ref].max()) / res.dw if ref.any() else float("nan"),
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
