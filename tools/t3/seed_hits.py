#!/usr/bin/env python3
"""List every file that records a seed of a block (T3-R1, P0.3).

Uses the matching rules of ``tools/v2/seed_inventory.py`` (its ``scan``: one read-only pass over the
whole tournament_experiment tree, all worktrees) and writes one row per (seed, source file, key) for
the seeds of each requested block. The ``class`` column comes from path rules; every row of class
``other`` is read by hand.

Usage:
    python tools/t3/seed_hits.py --out-dir results/t3_r1/preflight \
        --block 50500 50520 --block 30501 30520
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.v2.seed_inventory import TREE, scan  # noqa: E402

RUN_FILES = {"run_config.json", "manifest.json", "status.json", "gates.json"}
RUN_ROOTS = ("/results/", "/experiments/", "/MultiStage/")

Rule = Tuple[str, Callable[[str], bool]]

RULES: List[Rule] = [
    ("seed_inventory_output", lambda p: os.path.basename(p).startswith("seed_inventory")),
    # files this round wrote itself (smoke runs, preflight tables, the config-hash table); the first scan precedes them
    ("t3_r1_own_outputs", lambda p: "/results/t3_r1/" in p or "/reports/t3/r1/" in p),
    ("protocol", lambda p: "/protocols/" in p),
    (
        # the 40 v2.0 confirmation runs, their copies and the analyses / re-evaluations of them: the confirmation directories
        # themselves and their evidence copies; the MS-R1 offline calibration (bit-identical replays of the recorded exports,
        # results/ms_r1/calibration/) and the R2b diagnostic tables of seed 30510 (computed from the recorded files,
        # results/v2_refine_r2b/diag_30510/); none of them contains a training run
        "v20_confirmation_or_copy",
        lambda p: "confirmation_v2_0" in p or "/t2_refine_100526/evidence/" in p
        or "/results/ms_r1/calibration/" in p or "/results/v2_refine_r2b/diag_30510/" in p,
    ),
    (
        "run_file_under_results_experiments_multistage",
        lambda p: os.path.basename(p) in RUN_FILES and any(r in p for r in RUN_ROOTS),
    ),
]


def classify(path: str) -> str:
    """Return the class of a hit file, ``other`` if no rule matches.

    Args:
        path: Absolute path of the file that records the seed.

    Returns:
        The class name of the first matching rule.
    """
    for name, pred in RULES:
        if pred(path):
            return name
    return "other"


def main() -> int:
    """Scan once, write one CSV per block, print the class counts."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--block", type=int, nargs=2, action="append", required=True)
    ap.add_argument("--extra-root", nargs="*", default=[])
    a = ap.parse_args()
    found, n_json, n_csv = scan([str(TREE)] + a.extra_root)
    print(f"scanned {n_json} json, {n_csv} csv files; {len(found)} distinct seed values")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for lo, hi in a.block:
        rows: List[Dict[str, object]] = []
        for s in range(lo, hi + 1):
            for path, key in sorted(found.get(s, ())):
                rows.append({"seed": s, "class": classify(path), "source_file": path, "key": key})
        out = out_dir / f"seed_hits_{lo}_{hi}.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["seed", "class", "source_file", "key"])
            w.writeheader()
            w.writerows(rows)
        by_class = Counter(r["class"] for r in rows)
        print(f"block {lo}-{hi}: {len(rows)} hit rows, {len({r['seed'] for r in rows})} seeds -> {out}")
        for c, n in sorted(by_class.items()):
            files = {r["source_file"] for r in rows if r["class"] == c}
            names = Counter(os.path.basename(f) for f in files)
            print(f"  {c}: {n} rows in {len(files)} files; most common file names: {names.most_common(6)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
