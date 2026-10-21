#!/usr/bin/env python3
"""Reference-root check of round T3-R3 (P0.4, D8): the 80 recorded ``bw`` runs and the 40 v2.0 confirmation runs.

For each of the 40 T3-R1 ``bw`` runs (``<t3r1>/runs/q{q}/seed{s}/bw``) and 40 T3-R2 ``bw`` runs
(``<t3r2>/confirmation/q{q}/seed{s}/bw``) it records the export count (152 expected), the untracked
files (``state_end_stage{1,2,3}.pt``, the six ``freeze_stage*_*.npz``, ``train_history.json``), the tracked
files (``ms_updates.csv``, ``ms_checks_stage{1,2,3}.csv``, ``ms_binmaps_stage{2,3}.npz``, ``run_config.json``,
``gates.json``) and whether each tracked file is in the git index of the branch. For each of the 40 v2.0 runs
it records the export count (88 expected) and ``v2_updates.csv``. Read-only. Exit code 1 on any departure.

Usage:
    python tools/t3/t3r3_check_roots.py --t3r1-root <results/t3_r1> --t3r2-root <results/t3_r2> \
        --v20-root <confirmation_v2_0> --out <csv>
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
QS = (50, 60)
SEEDS3_R1 = range(50501, 50521)
SEEDS3_R2 = range(60501, 60521)
SEEDS2 = range(30501, 30521)
UNTRACKED_BW = (
    ["state_end_stage1.pt", "state_end_stage2.pt", "state_end_stage3.pt", "train_history.json"]
    + [f"freeze_stage{t}_{k}.npz" for t in (1, 2, 3) for k in ("final", "development")]
)
TRACKED_BW = (
    ["ms_updates.csv", "run_config.json", "gates.json"]
    + [f"ms_checks_stage{t}.csv" for t in (1, 2, 3)]
    + [f"ms_binmaps_stage{t}.npz" for t in (2, 3)]
)
TRACKED_ROOTS = (
    "results/t3_r1",
    "results/t3_r2",
    "results/v2_T2_locked/confirmation_v2_0",
    "reports/t3/r1",
    "reports/t3/r2",
)


def git_files(prefix: str) -> set:
    """The set of tracked paths under a prefix of this branch."""
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", prefix], capture_output=True, text=True).stdout
    return set(out.split("\n")) - {""}


def bw_dirs(t3r1: Path, t3r2: Path) -> List[Dict[str, Any]]:
    """The 80 ``bw`` run directories with their round and the tracked-copy prefix."""
    out = []
    for q in QS:
        for s in SEEDS3_R1:
            out.append({"round": "t3_r1", "q": q, "seed": s, "dir": t3r1 / "runs" / f"q{q}" / f"seed{s}" / "bw",
                        "tracked": f"results/t3_r1/runs/q{q}/seed{s}/bw"})
    for q in QS:
        for s in SEEDS3_R2:
            out.append({"round": "t3_r2", "q": q, "seed": s, "dir": t3r2 / "confirmation" / f"q{q}" / f"seed{s}" / "bw",
                        "tracked": f"results/t3_r2/confirmation/q{q}/seed{s}/bw"})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R3 reference-root check", allow_abbrev=False)
    ap.add_argument("--t3r1-root", required=True)
    ap.add_argument("--t3r2-root", required=True)
    ap.add_argument("--v20-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t3r1, t3r2, v20 = Path(a.t3r1_root), Path(a.t3r2_root), Path(a.v20_root)
    tracked = set()
    for p in ("results/t3_r1", "results/t3_r2", "results/v2_T2_locked/confirmation_v2_0"):
        tracked |= git_files(p)
    rows: List[Dict[str, Any]] = []
    for r in bw_dirs(t3r1, t3r2):
        d = r["dir"]
        n = len(list((d / "weights").glob("u*.npz"))) if (d / "weights").is_dir() else 0
        miss_u = [f for f in UNTRACKED_BW if not (d / f).is_file()]
        miss_t = [f for f in TRACKED_BW if not (d / f).is_file()]
        notin_git = [f for f in TRACKED_BW if f"{r['tracked']}/{f}" not in tracked]
        rows.append({"arm": "bw", "round": r["round"], "q": r["q"], "seed": r["seed"], "dir_exists": d.is_dir(),
                     "n_exports": n, "exports_expected": 152,
                     "train_history_present": (d / "train_history.json").is_file(),
                     "untracked_missing": ";".join(miss_u), "tracked_missing_in_root": ";".join(miss_t),
                     "tracked_not_in_git": ";".join(notin_git), "dir": str(d),
                     "ok": bool(d.is_dir() and n == 152 and not miss_u and not miss_t and not notin_git)})
    for q in QS:
        for s in SEEDS2:
            d = v20 / f"q{q}" / f"seed{s}"
            n = len(list((d / "weights").glob("u*.npz"))) if (d / "weights").is_dir() else 0
            has_u = (d / "v2_updates.csv").is_file()
            in_git = f"results/v2_T2_locked/confirmation_v2_0/q{q}/seed{s}/v2_updates.csv" in tracked
            rows.append({"arm": "v20", "round": "v2_0", "q": q, "seed": s, "dir_exists": d.is_dir(),
                         "n_exports": n, "exports_expected": 88, "train_history_present": "",
                         "untracked_missing": "", "tracked_missing_in_root": "" if has_u else "v2_updates.csv",
                         "tracked_not_in_git": "" if in_git else "v2_updates.csv", "dir": str(d),
                         "ok": bool(d.is_dir() and n == 88 and has_u and in_git)})
    tracked_roots = {p: bool(git_files(p)) for p in TRACKED_ROOTS}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(1 for r in rows if r["ok"])
    th = sum(1 for r in rows if r["arm"] == "bw" and r["train_history_present"])
    print(f"runs ok: {n_ok} / {len(rows)}; bw with train_history.json: {th} / 80")
    for r in rows:
        if not r["ok"]:
            print("NOT OK:", r["arm"], r["round"], r["q"], r["seed"], r["n_exports"], r["untracked_missing"],
                  r["tracked_missing_in_root"], r["tracked_not_in_git"])
    print("tracked roots present in the branch:", tracked_roots)
    return 0 if (n_ok == len(rows) and all(tracked_roots.values())) else 1


if __name__ == "__main__":
    sys.exit(main())
