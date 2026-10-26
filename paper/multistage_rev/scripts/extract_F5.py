"""F5 extraction: G_max,full/DeltaW of the T3-R1 arms bw, jt, jt_es (paired by seed), final tier.

Reads the tracked record ``results/t3_r1/eval/per_run.csv`` (no evaluation, no run directory is touched),
checks the row structure and the G-F pass counts against ``results/t3_r1/analysis/gate_counts.csv``, and writes
``data/F5.csv`` (long format: q, seed, arm, value, source). The two gate-limit rows (arm ``gate_limit``, no seed)
carry the plotted G-F limit from ``reports/t3/r1/prereg_criteria.json``.

Run:  /home/fjiang4/tournament_experiment/.venv/bin/python paper/multistage_rev/scripts/extract_F5.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

ARMS = ("bw", "jt", "jt_es")
SEEDS = tuple(range(50501, 50521))
VALUE_COL = "Gmax_full_over_dw"
# G-F pass counts (value <= gate limit) stated by the task: q=50, q=60
EXPECTED_GF = {("bw", 50): 20, ("bw", 60): 18, ("jt", 50): 0, ("jt", 60): 0, ("jt_es", 50): 4, ("jt_es", 60): 4}

SRC_VALUE = "results/t3_r1/eval/per_run.csv:Gmax_full_over_dw"
SRC_GATE = "reports/t3/r1/prereg_criteria.json:gates.G-F.Gmax_full_over_dw"
SRC_COUNTS = "results/t3_r1/analysis/gate_counts.csv:G-F_pass"
OUT_CSV = C.OUT / "data" / "F5.csv"


def fail(msg: str) -> None:
    """Stop-and-report: print the mismatch and exit non-zero without writing any output."""
    print("STOP:", msg)
    sys.exit(1)


def main() -> None:
    gate = C.read_json(C.ROOT / "reports" / "t3" / "r1" / "prereg_criteria.json")["gates"]["G-F"]["Gmax_full_over_dw"]
    if gate != 0.01:
        fail(f"G-F limit in prereg_criteria.json is {gate!r}, expected 0.01")

    df = C.load_per_run(C.PER_RUN_R1)
    sub = df[df["arm"].isin(ARMS)].copy()

    # row structure: 3 arms x 2 q x 20 seeds = 120 rows, seeds 50501..50520, each (q, seed) has all three arms
    if len(sub) != 120:
        fail(f"{len(sub)} rows for arms {ARMS}, expected 120")
    for arm in ARMS:
        for q in C.QS:
            s = sorted(sub[(sub["arm"] == arm) & (sub["q"] == q)]["seed"].tolist())
            if s != list(SEEDS):
                fail(f"arm {arm} q {q}: seeds {s[:3]}...{s[-3:]} (n={len(s)}), expected 50501..50520")
    if sub.duplicated(["arm", "q", "seed"]).any():
        fail("duplicate (arm, q, seed) rows")
    for q in C.QS:
        for seed in SEEDS:
            got = set(sub[(sub["q"] == q) & (sub["seed"] == seed)]["arm"])
            if got != set(ARMS):
                fail(f"q {q} seed {seed}: arms {sorted(got)}")
    if not bool(sub["valid"].all()) or not (sub["status"] == "done").all() or not (sub["exit_code"] == 0).all():
        fail("a row is not valid / done / exit_code 0")
    vals = sub[VALUE_COL].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        fail("non-finite G_max,full value")
    vmin = float(vals.min())
    n_floor = int((vals < 1e-8).sum())
    print(f"rows: {len(sub)}; min {VALUE_COL} = {vmin!r}; max = {float(vals.max())!r}; values < 1e-8: {n_floor}")
    if n_floor:
        fail(f"{n_floor} values below 1e-8: a floor is needed (not expected for F5)")

    # cross-check (not drawn): G-F pass counts vs gate_counts.csv and vs the task's stated counts
    gc = C.load_per_run(C.ROOT / "results" / "t3_r1" / "analysis" / "gate_counts.csv")
    for arm in ARMS:
        for q in C.QS:
            n = int((sub[(sub["arm"] == arm) & (sub["q"] == q)][VALUE_COL] <= gate).sum())
            rec = gc[(gc["arm"] == arm) & (gc["q"] == q)]
            if len(rec) != 1:
                fail(f"gate_counts.csv: {len(rec)} rows for arm {arm} q {q}")
            stored = int(rec["G-F_pass"].iloc[0])
            print(f"G-F pass (<= {gate}) arm {arm} q {q}: recount {n}/20, gate_counts.csv {stored}, task {EXPECTED_GF[(arm, q)]}")
            if n != stored or n != EXPECTED_GF[(arm, q)]:
                fail(f"G-F pass count arm {arm} q {q}: recount {n}, gate_counts.csv {stored}, task {EXPECTED_GF[(arm, q)]}")

    rows = []
    for q in C.QS:
        for arm in ARMS:
            for seed in SEEDS:
                v = sub[(sub["arm"] == arm) & (sub["q"] == q) & (sub["seed"] == seed)][VALUE_COL].iloc[0]
                rows.append({"q": q, "seed": seed, "arm": arm, "value": C.fmt(v), "source": SRC_VALUE})
    for q in C.QS:
        rows.append({"q": q, "seed": "", "arm": "gate_limit", "value": C.fmt(gate), "source": SRC_GATE})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["q", "seed", "arm", "value", "source"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} rows: {len(rows) - 2} data rows + 2 gate_limit rows)")


if __name__ == "__main__":
    main()
