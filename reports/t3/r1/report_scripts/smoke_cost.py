#!/usr/bin/env python3
"""Cost and correctness record of the smoke wave (seed 50500, both q, four arms at full budget): wall and CPU time per run, per
phase and per category, peak RSS, and the estimated wall time of the production waves ``t3`` and ``t2j`` at 40 workers
(longest-processing-time list scheduling of 40 runs per arm with the smoke wall time of the arm and q).

The smoke outputs serve cost and correctness only; they are never analysed or reported as results.

Usage:
    python reports/t3/r1/report_scripts/smoke_cost.py --root results/t3_r1 --out results/t3_r1/smoke_cost.csv
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import t3_analysis as ta  # noqa: E402
import t3_configs as tc  # noqa: E402
import t3_eval as te  # noqa: E402


def lpt_makespan(durations, workers):
    """Makespan of longest-processing-time-first list scheduling on ``workers`` identical workers."""
    heap = [0.0] * workers
    heapq.heapify(heap)
    for d in sorted(durations, reverse=True):
        heapq.heappush(heap, heapq.heappop(heap) + d)
    return max(heap)


def main() -> int:
    """Write the cost CSV and print the estimate."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(tc.DEFAULT_ROOT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=40)
    a = ap.parse_args()
    jobs = te.plan_jobs(Path(a.root), ["smoke"], None, None)
    rows = []
    for j in jobs:
        st = te.status_of(Path(j["run_dir"]))
        rows.append({**j, "exit_ok": st.get("state") == "done" and st.get("exit_code") == 0, "status": st.get("state"),
                     "exit_code": st.get("exit_code")})
    df = pd.DataFrame(rows)
    cost = ta.cost_table(df)
    cost.to_csv(a.out, index=False)
    print(cost[["arm", "q", "updates", "policy_rows_total", "minibatch_steps", "wall_sec", "cpu_sec"]].to_string(index=False))
    wall = {(r.arm, r.q): r.wall_sec for r in cost.itertuples()}
    plan = [(arm, q) for arm in ("bw", "jt", "jt_es") for q in tc.QS for _ in range(20)] + \
           [("jt2", q) for q in tc.QS for _ in range(20)]
    durations = [wall[k] for k in plan if k in wall]
    if len(durations) == len(plan):
        mk = lpt_makespan(durations, a.workers)
        print(f"production waves t3 + t2j (160 runs) on {a.workers} workers: sum of run wall times "
              f"{sum(durations) / 3600:.2f} h, makespan {mk / 60:.1f} min (smoke wall times, longest first)")
        t3 = [wall[(arm, q)] for arm in ("bw", "jt", "jt_es") for q in tc.QS for _ in range(20)]
        print(f"wave t3 alone ({len(t3)} runs, {a.workers} workers): {lpt_makespan(t3, a.workers) / 60:.1f} min; "
              f"wave t2j alone (40 runs): {lpt_makespan([wall[('jt2', q)] for q in tc.QS for _ in range(20)], a.workers) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
