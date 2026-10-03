#!/usr/bin/env python3
"""Inventory of every seed value recorded on disk, and a disjointness check for a proposed block.

Scans (read-only) every *.json file for keys "seed" / "seeds" / "*_seed" / "seed_*" (integer or list
of integers, any nesting depth, via a regex on the text), every *.csv file with a column whose name
contains "seed" (all integer values of that column), and the Phase 0 audit report
(reports/v2/phase0_audit.md: every integer on a line that mentions "seed"). Roots: the whole
tournament_experiment tree (all worktrees included; .git and .venv skipped) plus the extra roots
given on the command line.

Usage: python tools/v2/seed_inventory.py --block 20501 20520 --out <csv> [--extra-root DIR ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TREE = Path("/home/fjiang4/tournament_experiment")
SKIP = {".git", ".venv", "node_modules", "__pycache__"}
KEY_RE = re.compile(r'"([A-Za-z_]*seed[A-Za-z_]*)"\s*:\s*(\[[^\]]*\]|-?\d+)')
INT_RE = re.compile(r"-?\d+")


def scan(roots):
    found = defaultdict(set)          # seed -> set of (file, key)
    n_json = n_csv = 0
    for r in roots:
        for d, dirs, fs in os.walk(r):
            dirs[:] = [x for x in dirs if x not in SKIP]
            for f in fs:
                p = os.path.join(d, f)
                try:
                    if f.endswith(".json"):
                        n_json += 1
                        if os.path.getsize(p) > 200_000_000:
                            continue
                        txt = open(p, errors="ignore").read()
                        for key, val in KEY_RE.findall(txt):
                            for s in INT_RE.findall(val):
                                found[int(s)].add((p, key))
                    elif f.endswith(".csv"):
                        n_csv += 1
                        with open(p, newline="", errors="ignore") as fh:
                            rd = csv.reader(fh)
                            head = next(rd, [])
                            cols = [i for i, h in enumerate(head) if "seed" in h.lower()]
                            if not cols:
                                continue
                            for row in rd:
                                for i in cols:
                                    if i < len(row) and re.fullmatch(r"-?\d+(\.0)?", row[i].strip()):
                                        found[int(float(row[i]))].add((p, head[i]))
                except (OSError, UnicodeDecodeError, csv.Error):
                    continue
    return found, n_json, n_csv


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--block", type=int, nargs=2, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--extra-root", nargs="*", default=[])
    a = p.parse_args()
    found, nj, nc = scan([str(TREE)] + a.extra_root)
    audit = ROOT / "reports" / "v2" / "phase0_audit.md"
    for line in open(audit):
        if "seed" in line.lower():
            for s in INT_RE.findall(line):
                found[int(s)].add((str(audit), "phase0_audit_line"))
    block = list(range(a.block[0], a.block[1] + 1))
    hits = {s: sorted(found[s])[:5] for s in block if s in found}
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "n_sources", "example_source", "example_key"])
        for s in sorted(found):
            ex = sorted(found[s])[0]
            w.writerow([s, len(found[s]), ex[0], ex[1]])
    near = sorted(s for s in found if a.block[0] - 1000 <= s <= a.block[1] + 1000)
    print(f"scanned {nj} json, {nc} csv files; {len(found)} distinct seed values")
    print(f"block {a.block[0]}-{a.block[1]}: {len(hits)} collisions")
    for s, src in hits.items():
        print(" ", s, src)
    print("recorded seeds within +-1000 of the block:", near[:50])
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
