#!/usr/bin/env python3
"""Supplement (post hoc, descriptive) for ``04_pilot.md`` section 3.2: how many hidden units of the ``relu`` actors are active on D_2.

For every ``relu`` run the final terminal-stage weight export ``weights/u02800.npz`` (the freeze) is evaluated on the stage-2 input
(stage feature 1, d / B) over a grid of D_2 = [-B, B] with step 0.5, B = 100 + 2q: a first-layer (second-layer) unit is "alive" when its
ReLU output is positive at one grid point at least; the output of the network is the Beta mean effort ``100 * sigmoid(z0)``. The
weights are not tracked in git; ``results/ms_r3/analysis/relu_units.csv`` is written by this script (never overwritten) and tracked.

Usage (repository root):
    python reports/ms/r3/report_scripts/relu_units.py --write           # writes results/ms_r3/analysis/relu_units.csv
    python reports/ms/r3/report_scripts/relu_units.py --block summary|failed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

PILOT = Path("results/ms_r3/pilot")
CSV = Path("results/ms_r3/analysis/relu_units.csv")
ARMS = ["relu_bb_s1", "relu_bb_s16", "relu_st_s1", "relu_st_s16"]
SEEDS = list(range(10501, 10511))
QS = (50, 60)
FAILED = {("relu_bb_s1", 50, 10504), ("relu_bb_s1", 50, 10506), ("relu_bb_s16", 50, 10504), ("relu_st_s1", 50, 10506),
          ("relu_st_s16", 50, 10506)}      # the runs of block `failures` of pilot_tables.py (G-A with G-N(eta) fails)


def compute() -> pd.DataFrame:
    """One row per relu run: alive units of both hidden layers and the output range over the grid."""
    rows = []
    for arm in ARMS:
        for q in QS:
            b = 100.0 + 2.0 * q
            d = np.arange(-b, b + 0.25, 0.5)
            x = np.stack([np.ones_like(d), d / b], 1).astype(np.float32)
            for seed in SEEDS:
                z = np.load(PILOT / f"q{q}" / f"seed{seed}" / arm / "weights" / "u02800.npz")
                if str(np.asarray(z["actor_variant"])) != "relu":
                    raise SystemExit(f"{arm} q{q} seed{seed}: the export is not a relu export")
                h1 = np.maximum(x @ z["actor.l1.weight"].T + z["actor.l1.bias"], 0.0)
                h2 = np.maximum(h1 @ z["actor.l2.weight"].T + z["actor.l2.bias"], 0.0)
                zz = (h2 @ z["actor.out.weight"].T + z["actor.out.bias"])[:, 0].astype(np.float64)
                e = 100.0 / (1.0 + np.exp(-np.clip(zz, -60.0, 60.0)))
                rows.append({"arm": arm, "q": q, "seed": seed, "alive_layer1": int((h1 > 0).any(0).sum()), "alive_layer2": int((h2 > 0).any(0).sum()),
                             "failed_gate": (arm, q, seed) in FAILED, "effort_min": float(e.min()), "effort_max": float(e.max()),
                             "effort_at_0": float(e[int(np.argmin(np.abs(d)))])})
    return pd.DataFrame(rows)


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--block", choices=["summary", "failed"])
    a = ap.parse_args(argv)
    if a.write:
        if CSV.exists():
            raise SystemExit(f"{CSV} exists; not overwriting")
        compute().to_csv(CSV, index=False)
        print(f"wrote {CSV}")
        return 0
    df = pd.read_csv(CSV)
    if a.block == "summary":
        rows = []
        for (arm, q), g in df.groupby(["arm", "q"], sort=False):
            rows.append({"arm": arm, "q": int(q), "layer-1 units alive (of 64): median [min, max]": f"{g.alive_layer1.median():.1f} [{g.alive_layer1.min()}, {g.alive_layer1.max()}]",
                         "layer-2 units alive (of 64): median [min, max]": f"{g.alive_layer2.median():.1f} [{g.alive_layer2.min()}, {g.alive_layer2.max()}]"})
        print(md(pd.DataFrame(rows)))
    elif a.block == "failed":
        f = df[df.failed_gate].sort_values(["arm", "seed"])
        print(md(pd.DataFrame([{"arm": r.arm, "q": r.q, "seed": r.seed, "layer-1 alive": r.alive_layer1, "layer-2 alive": r.alive_layer2,
                                "effort min / max / at d = 0 over D_2": f"{r.effort_min:.2f} / {r.effort_max:.2f} / {r.effort_at_0:.2f}"} for r in f.itertuples()])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
