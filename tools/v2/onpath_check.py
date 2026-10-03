#!/usr/bin/env python3
"""Opening checks 1b and 1c.

1b: is e1*(0) an exact node of the effort grid of each tier?
1c: is the exact-positivity on-path set of the candidate pmf equal to {d in D_2 : |d| < 2q}?
    Candidates: closed form, e_hat == 0, and the 80 existing final checkpoints (all of them).

Usage: python tools/v2/onpath_check.py --raw /home/fjiang4/tournament_experiment/experiments \
    --out results/v2_pilots/phase2_opening
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from common import TIERS, analytic_policy, checkpoint_policy, spec_for, zero_policy  # noqa: E402
from utils.dp_br_verifier import effort_grid  # noqa: E402
from utils.theory_multistage import g1_two_stage  # noqa: E402
from utils.v2_metrics import append_csv, evaluate  # noqa: E402

COHORTS = ("two_stage_confirmation_T2_20260922", "two_stage_E1_q50_p_20260923",
           "two_stage_q50_restarts_20260924", "two_stage_q50_precision_20260924")


def main() -> int:
    """Write effort_grid_membership.csv and onpath_sets.csv."""
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    f1, f2 = os.path.join(a.out, "effort_grid_membership.csv"), os.path.join(a.out, "onpath_sets.csv")
    for f in (f1, f2):
        if os.path.exists(f):
            raise SystemExit(f"{f} exists")
    for q in (50.0, 60.0):
        spec = spec_for(q)
        g1 = g1_two_stage(q, spec.w_h, spec.w_l, spec.k)
        for cfg in TIERS:
            E = effort_grid(0.0, 100.0, cfg.effort_step)
            j = int(np.argmin(np.abs(E - g1)))
            append_csv(f1, {"q": q, "tier": cfg.name, "effort_step": cfg.effort_step, "e1_star": g1,
                            "nearest_node": E[j], "dist_to_nearest": abs(E[j] - g1),
                            "exact_node": bool(np.any(E == g1))})
    cands = []
    for q in (50.0, 60.0):
        spec = spec_for(q)
        cands += [("analytic", q, analytic_policy(spec)), ("zero", q, zero_policy)]
    for coh in COHORTS:
        for ck in sorted(glob.glob(os.path.join(a.raw, coh, "runs", "*", "*", "checkpoint.pt"))):
            cfg = json.load(open(os.path.join(os.path.dirname(ck), "config.json")))
            cands.append((f"{coh}/{cfg['run']}", float(cfg["q"]),
                          checkpoint_policy(ck, spec_for(cfg["q"]))[0]))
    for name, q, pol in cands:
        spec = spec_for(q)
        for cfg in TIERS:
            ev = evaluate(pol, spec, cfg)
            g = ev.res.stages[2].d_grid
            on = ev.pmf_cand[2] > 0.0
            ref = np.abs(g) < 2.0 * q
            e1 = float(np.asarray(pol(1, np.zeros(1)))[0])
            e1m = float(np.asarray(pol(1, -np.zeros(1)))[0])
            only_on = g[on & ~ref]
            only_ref = g[ref & ~on]
            append_csv(f2, {"candidate": name, "q": q, "tier": cfg.name, "n_grid": g.size,
                            "stage1_drift": e1 - e1m, "n_onpath": int(on.sum()), "n_abs_lt_2q": int(ref.sum()),
                            "equal": bool(np.array_equal(on, ref)),
                            "n_onpath_not_in_ref": int(only_on.size), "n_ref_not_onpath": int(only_ref.size),
                            "onpath_not_in_ref_d": " ".join(f"{x:g}" for x in only_on),
                            "ref_not_onpath_d": " ".join(f"{x:g}" for x in only_ref)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
