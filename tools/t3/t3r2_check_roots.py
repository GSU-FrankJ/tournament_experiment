#!/usr/bin/env python3
"""Reference-root check of round T3-R2 (P0.4, D6): the run directories of T3-R1 (untracked root) and the v2.0 confirmation runs.

For each of the 40 ``bw`` (152 exports, ``state_end_stage{1,2,3}.pt``), ``jt`` (50 exports, ``state_end.pt``), ``jt_es`` (70) and
``jt2`` (44) runs of T3-R1 and each of the 40 ``v20`` runs (88 exports, ``state_end_A.pt`` / ``state_end_B.pt``) it records
whether the directory exists and the export count, plus ``eval/arrays/`` (200 files). Read-only. Exit code 1 on any departure.

Usage:
    python tools/t3/t3r2_check_roots.py --t3r1-root <results/t3_r1> --v20-root <confirmation_v2_0> --out <csv>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

QS = (50, 60)
SEEDS3 = range(50501, 50521)
SEEDS2 = range(30501, 30521)
EXPORTS = {"bw": 152, "jt": 50, "jt_es": 70, "jt2": 44, "v20": 88}
STATE = {"bw": ["state_end_stage1.pt", "state_end_stage2.pt", "state_end_stage3.pt"], "jt": ["state_end.pt"],
         "jt_es": ["state_end.pt"], "jt2": ["state_end.pt"], "v20": ["state_end_A.pt", "state_end_B.pt"]}


def run_dirs(t3r1: Path, v20: Path) -> List[Dict[str, Any]]:
    """The 200 run directories: ``{arm, q, seed, dir}``."""
    out = []
    for arm in ("bw", "jt", "jt_es"):
        for q in QS:
            for s in SEEDS3:
                out.append({"arm": arm, "q": q, "seed": s, "dir": t3r1 / "runs" / f"q{q}" / f"seed{s}" / arm})
    for q in QS:
        for s in SEEDS2:
            out.append({"arm": "jt2", "q": q, "seed": s, "dir": t3r1 / "runs_t2" / f"q{q}" / f"seed{s}" / "jt2"})
    for q in QS:
        for s in SEEDS2:
            out.append({"arm": "v20", "q": q, "seed": s, "dir": v20 / f"q{q}" / f"seed{s}"})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R2 reference-root check", allow_abbrev=False)
    ap.add_argument("--t3r1-root", required=True)
    ap.add_argument("--v20-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t3r1, v20 = Path(a.t3r1_root), Path(a.v20_root)
    rows = []
    for r in run_dirs(t3r1, v20):
        d = r["dir"]
        n = len(list((d / "weights").glob("u*.npz"))) if (d / "weights").is_dir() else 0
        missing = [f for f in STATE[r["arm"]] + ["run_config.json", "status.json"] if not (d / f).is_file()]
        rows.append({"arm": r["arm"], "q": r["q"], "seed": r["seed"], "dir_exists": d.is_dir(), "n_exports": n,
                     "exports_expected": EXPORTS[r["arm"]], "files_missing": ";".join(missing), "dir": str(d),
                     "ok": bool(d.is_dir() and n == EXPORTS[r["arm"]] and not missing)})
    arrays = list((t3r1 / "eval" / "arrays").glob("*.npz"))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    per_arm = {arm: (sum(1 for r in rows if r["arm"] == arm and r["ok"]), sum(1 for r in rows if r["arm"] == arm))
               for arm in EXPORTS}
    print("ok / total per arm:", per_arm, "; eval/arrays files:", len(arrays))
    bad = [r for r in rows if not r["ok"]]
    if bad or len(arrays) != 200:
        print("problems:", [(r["arm"], r["q"], r["seed"]) for r in bad][:10], "arrays", len(arrays))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
