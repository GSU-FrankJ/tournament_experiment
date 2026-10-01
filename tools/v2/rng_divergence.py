#!/usr/bin/env python3
"""First update at which two paired v2 runs' RNG streams diverge (diagnostic).

Reads the per-update ``rngpos_<stream>`` columns of ``v2_updates.csv`` (stream position after
each update) of two run directories and reports, per stream, the first update whose positions
differ, or "never".

Usage: python tools/v2/rng_divergence.py <run_dir_a> <run_dir_b> [--json]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict

STREAMS = ("env", "learn", "opp", "start", "minibatch")


def first_divergence(dir_a: str, dir_b: str) -> Dict[str, object]:
    """Per stream: first differing update, or 'never'; plus the number of compared updates."""
    ra = list(csv.DictReader(open(os.path.join(dir_a, "v2_updates.csv"))))
    rb = list(csv.DictReader(open(os.path.join(dir_b, "v2_updates.csv"))))
    if [r["update"] for r in ra] != [r["update"] for r in rb]:
        raise ValueError("the two runs have different update sequences")
    out: Dict[str, object] = {"n_updates_compared": len(ra)}
    for s in STREAMS:
        out[s] = next((int(x["update"]) for x, y in zip(ra, rb) if x[f"rngpos_{s}"] != y[f"rngpos_{s}"]), "never")
    return out


def main() -> int:
    """Print the divergence record."""
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    a = p.parse_args()
    print(json.dumps(first_divergence(a.a, a.b)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
