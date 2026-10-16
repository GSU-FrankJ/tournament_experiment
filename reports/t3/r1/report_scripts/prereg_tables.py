#!/usr/bin/env python3
"""Prints the markdown tables of ``02_preregistration.md`` from the tool outputs: the D5 budget table (pinned values), the
smoke wave cost table, the smoke check verdicts and the pre-launch checks.

Usage:
    python reports/t3/r1/report_scripts/prereg_tables.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import t3_configs as tc  # noqa: E402

RES = ROOT / "results" / "t3_r1"


def fmt(n):
    return f"{int(n):,}"


def budget_table():
    print("| arm | updates | policy rows per update (t = 1 / 2 / 3) | policy rows (t = 1 / 2 / 3; total) | minibatch steps |")
    print("|---|---|---|---|---|")
    for arm in ("bw", "jt", "jt_es", "v20", "jt2"):
        b = tc.expected_budget(arm)
        rpu = "512 in its stage" if arm in ("bw", "v20") else " / ".join(str(b["rows_per_update"][t]) for t in sorted(b["rows_per_update"]))
        rows = " / ".join(fmt(b["rows_by_stage"][t]) for t in sorted(b["rows_by_stage"]))
        print(f"| `{arm}` | {fmt(b['updates'])} | {rpu} | {rows}; {fmt(b['rows_total'])} | {fmt(b['minibatch_steps'])} |")


def smoke_table():
    rows = list(csv.DictReader(open(RES / "smoke_cost.csv")))
    print("| arm | q | updates | policy rows | minibatch steps | wall (s) | CPU (s) | train_rollout (s) | train_update (s) | dev_verifier (s) | final_eval (s) | peak RSS (MB) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        def g(k):
            v = r.get(k, "")
            return float(v) if v not in ("", None) else float("nan")
        rss = g("peak_rss_kb") / 1024.0
        print(f"| `{r['arm']}` | {r['q']} | {fmt(float(r['updates']))} | {fmt(float(r['policy_rows_total']))} | {fmt(float(r['minibatch_steps']))} | "
              f"{g('wall_sec'):.0f} | {g('cpu_sec'):.0f} | {g('train_rollout_sec'):.1f} | {g('train_update_sec'):.1f} | {g('dev_verifier_sec'):.1f} | "
              f"{g('final_eval_sec'):.1f} | {rss:.0f} |")
    for r in rows:
        if r["arm"] == "bw":
            print(f"\n`bw` q = {r['q']} per phase (wall s): stage 3 {float(r['wall_stage3_sec']):.0f}, stage 2 {float(r['wall_stage2_sec']):.0f}, "
                  f"stage 1 {float(r['wall_stage1_sec']):.0f}; CPU s: {float(r['cpu_stage3_sec']):.0f} / {float(r['cpu_stage2_sec']):.0f} / {float(r['cpu_stage1_sec']):.0f}")


def check_table():
    d = json.load(open(RES / "smoke_checks.json"))
    print("| check | pass | fail | not evaluated |")
    print("|---|---|---|---|")
    for k, v in d["verdicts"].items():
        print(f"| {k} | {v['pass']} | {v['fail']} | {v['not_evaluated']} |")
    print(f"\nC-INIT within (q, seed): {d['c_init_within_q_seed']}; equals the plan: {d['c_init_equals_plan']}; start-share flags: {d['start_share_flags']}")


if __name__ == "__main__":
    print("## D5 budget table\n")
    budget_table()
    print("\n## Smoke cost\n")
    smoke_table()
    print("\n## Smoke checks\n")
    check_table()
