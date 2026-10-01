#!/usr/bin/env python3
"""Invariant probe on synthetic candidates: closed-form stage 2 plus sinusoidal perturbations.

Each candidate is e1 = U(0,100) at the root and e2(d) = clip(e2*(d) + A sin(f d + phi), 0, 100),
with A = 0 for every third candidate (stage 2 exactly closed-form, the case where Q^BR_1 and
Q^mean_1 coincide most closely). Fixed numpy seed 123 (not a training stream).

Usage: python tools/v2/probe_invariants.py --out results/v2_pilots/phase1/invariants_probe.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from common import TIERS, analytic_policy, spec_for  # noqa: E402
from utils.v2_metrics import append_csv, evaluate  # noqa: E402


def main() -> int:
    """Evaluate 150 synthetic candidates per q on the three tiers."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=150)
    a = p.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rng = np.random.default_rng(123)
    for q in (50.0, 60.0):
        spec = spec_for(q)
        eq = analytic_policy(spec)
        for i in range(a.n):
            a1 = rng.uniform(0, 100)
            amp = rng.uniform(0, 10) * (i % 3 != 0)
            ph = rng.uniform(0, 6.3)
            fr = rng.uniform(0.005, 0.05)

            def pol(t, d, a1=a1, amp=amp, ph=ph, fr=fr):
                d = np.asarray(d, float)
                if t == 1:
                    return np.full(d.shape, a1)
                return np.clip(eq(2, d) + amp * np.sin(fr * d + ph), 0, 100)
            for cfg in TIERS:
                row = {"q": q, "candidate": i, "e1": a1, "amp": amp, "freq": fr, "phase": ph}
                row.update(evaluate(pol, spec, cfg).scalars)
                append_csv(a.out, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
