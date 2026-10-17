#!/usr/bin/env python3
"""T3-R1 P3: the per-phase CPU time of the recorded ``v20`` runs (``analysis/v20_cpu_phase.csv``).

The recorded v2.0 confirmation runs carry the process CPU time of each phase in ``v2_run_summary.json``
(``phase_timing.A.process_cpu_sec`` = the stage-2 phase, ``phase_timing.B.process_cpu_sec`` = the stage-1 phase); they carry no
whole-process CPU time and no CPU time by category. ``analysis/cost.csv`` (written by ``tools/t3/t3_analysis.py``) leaves the
``v20`` CPU cells empty and is NOT changed; ``results_tables.py`` reads this file for the ``v20`` CPU cells of the cost tables.

The tracked copies of the run summaries under ``results/v2_T2_locked/confirmation_v2_0/q{q}/seed{s}/`` are read (the D9 reference
root holds the same files).

Usage:
    python reports/t3/r1/report_scripts/v20_cpu.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "results" / "v2_T2_locked" / "confirmation_v2_0"
OUT = ROOT / "results" / "t3_r1" / "analysis" / "v20_cpu_phase.csv"


def main() -> int:
    """Write one row per v20 run: the wall and CPU time of the stage-2 phase (A) and the stage-1 phase (B)."""
    rows = []
    for q in (50, 60):
        for seed in range(30501, 30521):
            p = SRC / f"q{q}" / f"seed{seed}" / "v2_run_summary.json"
            ph = json.loads(p.read_text())["phase_timing"]
            a, b = ph["A"], ph["B"]
            rows.append({"q": q, "seed": seed, "wall_stage2_sec": a["wall_sec"], "cpu_stage2_sec": a["process_cpu_sec"],
                         "wall_stage1_sec": b["wall_sec"], "cpu_stage1_sec": b["process_cpu_sec"],
                         "cpu_sum_sec": a["process_cpu_sec"] + b["process_cpu_sec"]})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
