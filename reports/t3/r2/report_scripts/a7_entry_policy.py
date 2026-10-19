#!/usr/bin/env python3
"""A7 addition (descriptive, not in D3): the stage-2 policy at the PHASE ENTRY, i.e. the stage-3 network ``u01600`` evaluated at the stage-2
inputs (tau = 0.5, d / B), before any stage-2 update, at the stretch nodes of the seven stretch runs and, for comparison, at the same nodes
of the 33 other runs of the q. Reads the stored exports only (pure, no random number).

Writes ``reports/t3/r2/tables/a07f_phase_entry.md`` and ``results/t3_r2/audit/a7_phase_entry.csv``.

Usage:
    python reports/t3/r2/report_scripts/a7_entry_policy.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd
import torch

from common import RES, md, num, write  # noqa: E402

import t3_eval as te  # noqa: E402
from replay_dev_rule import candidate_fns  # noqa: E402

torch.set_num_threads(1)
UNTRACKED = Path(os.environ.get("T3R1_UNTRACKED_ROOT", "/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-vs-jt-51e472/results/t3_r1"))


def main() -> int:
    """Evaluate the entry policy at the stretch nodes and write the table."""
    runs = json.loads((RES / "audit" / "a7_runs.json").read_text())
    rows, csv_rows = [], []
    for x in (y for y in runs if y["group"] == "stretch"):
        q, s = x["q"], x["seed"]
        d = UNTRACKED / "runs" / f"q{q}" / f"seed{s}" / "bw"
        cfg = te.read_json(d / "run_config.json")
        spec = te.spec_of(cfg)
        entry = candidate_fns(spec, {2: te.net_from_export(d / "weights" / "u01600.npz", cfg)})[0]
        final = candidate_fns(spec, {2: te.net_from_export(d / "weights" / "u03200.npz", cfg)})[0]
        nodes = np.array(x["nodes"], dtype=float)
        e0, e1 = entry(2, nodes), final(2, nodes)
        # the other runs of the q at the same nodes (entry policy)
        others = [y for y in runs if y["group"] == "comparison" and y["q"] == q]
        frac_other = []
        for y in others:
            dd = UNTRACKED / "runs" / f"q{q}" / f"seed{y['seed']}" / "bw"
            cf = te.read_json(dd / "run_config.json")
            en = candidate_fns(te.spec_of(cf), {2: te.net_from_export(dd / "weights" / "u01600.npz", cf)})[0]
            frac_other.append(float((en(2, nodes) < 1.0).mean()))
        rows.append((q, s, len(nodes), int((e0 < 1.0).sum()), num(float(np.median(e0)), 3), num(float(e0.max()), 3), int((e1 < 1.0).sum()),
                     f"{min(frac_other):.2f} - {max(frac_other):.2f}", f"{sum(1 for f in frac_other if f >= 0.5)} of {len(frac_other)}"))
        for dv, a, b in zip(nodes, e0, e1):
            csv_rows.append({"q": q, "seed": s, "d": dv, "e_entry_u01600": float(a), "e_final_u03200": float(b)})
    write("a07f_phase_entry", md(["q", "seed", "stretch nodes", "nodes with effort < 1 at the phase entry (u01600 at the stage-2 input)",
                                  "median entry effort", "max entry effort", "nodes with effort < 1 at the end (u03200)",
                                  "the other runs of the q: share of these nodes with entry effort < 1 (range)",
                                  "other runs with at least half of these nodes below 1 at the entry"], rows))
    pd.DataFrame(csv_rows).to_csv(RES / "audit" / "a7_phase_entry.csv", index=False)
    print(md(["q", "seed", "nodes", "entry<1", "median", "max", "final<1", "others", "others>=half"], rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
