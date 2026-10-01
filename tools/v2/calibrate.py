#!/usr/bin/env python3
"""Verifier calibration: closed-form equilibrium and e_hat == 0, q in {50, 60}, three tiers.

Tiers: development (state 4, effort 1, GL 16/half), dev_2x (state 2, effort 0.5, GL 16/half:
2x finer in state and action only) and final (state 2, effort 0.5, GL 32/half). Also writes one
example of the per-checkpoint storage (NPZ + CSV row + stage-2 plot) for an existing trained
final checkpoint, so the storage format can be inspected.

Usage: python tools/v2/calibrate.py --out results/v2_pilots/phase1/calibration \
    --example-ckpt /home/fjiang4/tournament_experiment/experiments/two_stage_q50_precision_20260924/runs/FINAL_A400_B25_C25/tel_q50_s10231/checkpoint.pt
"""

from __future__ import annotations

import argparse
import os

from common import BASE_COMMIT, TIERS, analytic_policy, checkpoint_policy, spec_for, zero_policy  # noqa: E402
from utils.dp_br_verifier import DEV_CONFIG  # noqa: E402
from utils.v2_metrics import append_csv, evaluate, plot_stage2, save_npz  # noqa: E402


def main() -> int:
    """Run the calibration grid and the storage example."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--example-ckpt", required=True)
    p.add_argument("--example-q", type=float, default=50.0)
    a = p.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    os.makedirs(os.path.join(a.out, "npz"))
    csv_path = os.path.join(a.out, "calibration.csv")
    for q in (50.0, 60.0):
        spec = spec_for(q)
        for name, pol in (("analytic_eq", analytic_policy(spec)), ("zero", zero_policy)):
            for cfg in TIERS:
                ev = evaluate(pol, spec, cfg)
                row = {"base_commit": BASE_COMMIT, "q": q, "policy": name}
                row.update(ev.scalars)
                append_csv(csv_path, row)
                save_npz(ev, os.path.join(a.out, "npz", f"q{q:g}_{name}_{cfg.name}.npz"))
    # storage example on a trained checkpoint (development tier, as at a training checkpoint)
    ex = os.path.join(a.out, "example_checkpoint")
    os.makedirs(ex)
    spec = spec_for(a.example_q)
    mean_fn, beta_fn = checkpoint_policy(a.example_ckpt, spec)
    ev = evaluate(mean_fn, spec, DEV_CONFIG, beta_fn=beta_fn)
    row = {"base_commit": BASE_COMMIT, "source_checkpoint": a.example_ckpt, "update": "final"}
    row.update(ev.scalars)
    append_csv(os.path.join(ex, "metrics.csv"), row)
    save_npz(ev, os.path.join(ex, "ckpt_final.npz"))
    plot_stage2(ev, spec, os.path.join(ex, "stage2_final.png"),
                title=f"example: {os.path.basename(os.path.dirname(a.example_ckpt))} (q={a.example_q:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
