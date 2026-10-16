#!/usr/bin/env python3
"""T3-R1 P0.3 (supplement): file and directory NAMES that contain a seed of 50500-50520 or 30501-30520.

The seed inventory and ``tools/t3/seed_hits.py`` read the contents of JSON and CSV files; this lists names, so a run
directory such as ``seed30501`` is seen as well. Walks the whole tournament_experiment tree (all worktrees), skips
``.git``, ``__pycache__`` and ``.venv``, and prints the match counts and, per block, the matches grouped by their first
three path components below ``results/``, ``reports/``, ``tools/`` and so on.

Usage:
    python reports/t3/r1/report_scripts/seed_names.py > results/t3_r1/preflight/seed_names.out.txt
"""

from __future__ import annotations

import os
import re
from typing import Dict, List

ROOT = "/home/fjiang4/tournament_experiment"
BLOCKS = {
    "50500-50520": re.compile(r"(?<!\d)(505(0[0-9]|1[0-9]|20))(?!\d)"),
    "30501-30520": re.compile(r"(?<!\d)(305(0[1-9]|1[0-9]|20))(?!\d)"),
}
TOP = r"^.*?/(results|experiments|MultiStage|reports|docs|protocols|tools|tests|run)/"


def main() -> None:
    """Walk the tree once and print the grouped matches."""
    hits: Dict[str, List[str]] = {k: [] for k in BLOCKS}
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in (".git", "node_modules", "__pycache__", ".venv")]
        for name in dns + fns:
            for k, p in BLOCKS.items():
                if p.search(name):
                    hits[k].append(os.path.join(dp, name))
    for k, v in hits.items():
        print(f"{k}: {len(v)} names")
        groups: Dict[str, int] = {}
        for h in v:
            key = "/".join(re.sub(TOP, r"\1/", h).split("/")[:3])
            groups[key] = groups.get(key, 0) + 1
        for g, n in sorted(groups.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {n:5d}  {g}")


if __name__ == "__main__":
    main()
