#!/usr/bin/env python3
"""Figure for ``04_pilot.md``: the learned stage-2 policy near the tie, run by run and as the seed MEDIAN, for each actor.

The analysis figure ``tie_profile_near_tie.png`` shows seed means with min-max bands; a single collapsed run pulls a mean and a
band over the whole panel. This figure (a report-side supplement, not part of the analysis tool) draws every run as a thin line and
the seed median as a thick one, from the final-tier freeze arrays ``freeze_stage2_final.npz`` (``recovery_d_grid``,
``recovery_e2``, ``recovery_g2``) of ``results/ms_r3/pilot`` (the arrays are not tracked in git; the PNG is).

Usage (repository root):
    python reports/ms/r3/report_scripts/tie_profile_runs.py [--pilot results/ms_r3/pilot] [--out results/ms_r3/analysis/figures/tie_profile_runs.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ACTORS = ["t1", "relu", "t10"]
COLORS = {"t1": "#1f77b4", "relu": "#d62728", "t10": "#2ca02c"}
SEEDS = list(range(10501, 10511))
HALF = 30.0


def load(pilot: Path, arm: str, q: int) -> Dict[str, np.ndarray]:
    """Profiles of the ten seeds of one arm and q on |d| <= HALF: ``d``, ``e`` (seeds x nodes), ``g`` (closed form)."""
    es: List[np.ndarray] = []
    d = g = None
    for s in SEEDS:
        z = np.load(pilot / f"q{q}" / f"seed{s}" / arm / "freeze_stage2_final.npz")
        dd = z["recovery_d_grid"]
        m = np.abs(dd) <= HALF
        d, g = dd[m], z["recovery_g2"][m]
        es.append(z["recovery_e2"][m])
    return {"d": d, "e": np.asarray(es), "g": g}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pilot", default="results/ms_r3/pilot")
    ap.add_argument("--out", default="results/ms_r3/analysis/figures/tie_profile_runs.png")
    a = ap.parse_args(argv)
    pilot, out = Path(a.pilot), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    fig, axes = plt.subplots(4, 2, figsize=(10, 13), dpi=100)
    for r, (starts, s) in enumerate([("bb", 1), ("bb", 16), ("st", 1), ("st", 16)]):
        for c, q in enumerate((50, 60)):
            ax = axes[r, c]
            for actor in ACTORS:
                p = load(pilot, f"{actor}_{starts}_s{s}", q)
                for e in p["e"]:
                    ax.plot(p["d"], e, color=COLORS[actor], lw=0.5, alpha=0.35)
                ax.plot(p["d"], np.median(p["e"], axis=0), color=COLORS[actor], lw=2.0, label=f"{actor} (median of 10 runs)")
            ax.plot(p["d"], p["g"], "k--", lw=1.0, label="closed form e2*(d)")
            ax.set_title(f"{'bin-balanced' if starts == 'bb' else 'stratified'} starts, s = {s}, q = {q}", fontsize=9)
            ax.set_ylim(40.0 if q == 50 else 38.0, 72.0 if q == 50 else 61.0)
            ax.set_xlabel("d", fontsize=8)
            ax.set_ylabel("effort", fontsize=8)
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="lower center")
    fig.suptitle("Descriptive: the learned e_hat_2(d) at the final-tier terminal freeze, every run (thin) and the seed median (thick);\n"
                 "a run below the plotted range is clipped (the collapsed run has e_hat_2 = 0)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
