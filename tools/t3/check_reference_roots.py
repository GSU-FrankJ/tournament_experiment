#!/usr/bin/env python3
"""Verify the reference roots of D9 (T3-R1, P0.4): the 40 v2.0 confirmation runs (export counts, status, files) and the
byte-identity of each tracked ``gates.json`` of the branch with its reference copy. Read-only.

Usage:
    python tools/t3/check_reference_roots.py --v20-root <D9 root> --out <csv>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
QS = (50, 60)
SEEDS = range(30501, 30521)
NEED = ["state_end_A.pt", "state_end_B.pt", "gates.json", "v2_run_summary.json", "v2_updates.csv", "run_config.json"]
TRACKED = "results/v2_T2_locked/confirmation_v2_0"


def sha256(path: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    """Write one row per run; exit 1 if anything is missing or differs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--v20-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.v20_root)
    tracked = set(subprocess.run(["git", "-C", str(ROOT), "ls-files", TRACKED], capture_output=True, text=True).stdout.split())
    rows: List[Dict[str, Any]] = []
    for q in QS:
        for s in SEEDS:
            d = root / f"q{q}" / f"seed{s}"
            row: Dict[str, Any] = {"q": q, "seed": s, "dir_exists": d.is_dir()}
            if d.is_dir():
                st = json.load(open(d / "status.json"))
                g = json.load(open(d / "gates.json"))
                ref = d / "gates.json"
                trk = ROOT / TRACKED / f"q{q}" / f"seed{s}" / "gates.json"
                rel = f"{TRACKED}/q{q}/seed{s}/gates.json"
                row.update({
                    "n_exports": len(list((d / "weights").glob("u*.npz"))),
                    "files_missing": ";".join(f for f in NEED if not (d / f).exists()),
                    "status": st.get("state"), "exit_code": st.get("exit_code"), "outcome": g.get("outcome"),
                    "run_pass": g.get("run_pass"), "G_A": g["G-A"]["pass"], "G_F": g["G-F"]["pass"],
                    "G_N": g["G-N"]["pass"], "G_S": g["G-S"]["pass"], "global_rng": g["global_rng"]["status"],
                    "gates_sha256": sha256(ref), "tracked_in_git": rel in tracked,
                    "tracked_copy_identical": trk.exists() and sha256(trk) == sha256(ref)})
            rows.append(row)
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if not (r["dir_exists"] and r["n_exports"] == 88 and not r["files_missing"] and r["status"] == "done"
                                   and r["exit_code"] == 0 and r["tracked_in_git"] and r["tracked_copy_identical"])]
    fails = [(r["q"], r["seed"], r["outcome"]) for r in rows if r.get("run_pass") is not True]
    print(f"{len(rows)} runs; {sum(1 for r in rows if r['dir_exists'])} directories; "
          f"{sum(1 for r in rows if r.get('n_exports') == 88)} with 88 exports; "
          f"{sum(1 for r in rows if r.get('status') == 'done' and r.get('exit_code') == 0)} done with exit code 0; "
          f"{sum(1 for r in rows if r.get('tracked_copy_identical'))} tracked gates.json byte-identical")
    print("runs that did not pass (q, seed, outcome):", fails)
    print("problems:", [(r["q"], r["seed"]) for r in bad])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
