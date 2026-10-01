#!/usr/bin/env python3
"""Run the v2 metrics + invariant residuals on existing final T=2 checkpoints (read-only).

Usage:
    python tools/v2/invariants_on_checkpoints.py --raw /home/fjiang4/tournament_experiment/experiments \
        --out results/v2_pilots/phase1/invariants_checkpoints.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from common import TIERS, spec_for, checkpoint_policy  # noqa: E402
from utils.v2_metrics import evaluate, append_csv  # noqa: E402

COHORTS = ("two_stage_confirmation_T2_20260922", "two_stage_E1_q50_p_20260923",
           "two_stage_q50_restarts_20260924", "two_stage_q50_precision_20260924")


def main() -> int:
    """Evaluate every final checkpoint of the four final-protocol cohorts on three tiers."""
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; refusing to append")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for coh in COHORTS:
        for ck in sorted(glob.glob(os.path.join(a.raw, coh, "runs", "*", "*", "checkpoint.pt"))):
            cfg = json.load(open(os.path.join(os.path.dirname(ck), "config.json")))
            spec = spec_for(cfg["q"])
            mean_fn, beta_fn = checkpoint_policy(ck, spec)
            for tier in TIERS:
                ev = evaluate(mean_fn, spec, tier, beta_fn=beta_fn)
                row = {"cohort": coh, "run": cfg["run"], "q": cfg["q"], "seed": cfg["seed"],
                       "checkpoint": ck}
                row.update(ev.scalars)
                append_csv(a.out, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
