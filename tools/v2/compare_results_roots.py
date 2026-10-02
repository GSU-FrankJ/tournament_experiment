#!/usr/bin/env python3
"""SHA-256 comparison of two copies of results/v2_pilots/ (read-only; deletes nothing).

Lists every file of the ORIGINAL root and compares it byte-for-byte (SHA-256) with the same
relative path under the CANONICAL root. Pilot 4 parents (Phase A extension state_u01200.pt and
state_u01600.pt) are flagged. Files only in the canonical root (Pilot 4 outputs) are counted.

Usage: python tools/v2/compare_results_roots.py ORIGINAL_ROOT CANONICAL_ROOT OUT_CSV [--workers N]
"""

from __future__ import annotations

import argparse
import hashlib
import os
from multiprocessing import Pool

import pandas as pd


def sha(path: str) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _job(args):
    rel, a, b = args
    ha = sha(a)
    hb = sha(b) if os.path.exists(b) else None
    return {"path": rel, "sha256_original": ha, "sha256_canonical": hb, "in_canonical": hb is not None,
            "identical": ha == hb,
            "pilot4_parent": rel.startswith("phaseA_ext/") and os.path.basename(rel) in ("state_u01200.pt", "state_u01600.pt")}


def walk(root: str):
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            yield os.path.relpath(p, root)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("original")
    p.add_argument("canonical")
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    jobs = [(r, os.path.join(a.original, r), os.path.join(a.canonical, r)) for r in sorted(walk(a.original))]
    with Pool(a.workers) as pool:
        df = pd.DataFrame(pool.map(_job, jobs, chunksize=16))
    df.to_csv(a.out, index=False)
    only_canon = sum(1 for r in walk(a.canonical) if not os.path.exists(os.path.join(a.original, r)))
    par = df[df.pilot4_parent]
    print(f"original files: {len(df)}; identical in canonical: {int(df.identical.sum())}; missing in canonical: "
          f"{int((~df.in_canonical).sum())}; differing: {int((df.in_canonical & ~df.identical).sum())}")
    print(f"pilot4 parents: {len(par)}; identical: {int(par.identical.sum())}")
    print(f"files only in canonical: {only_canon}")
    if (df.in_canonical & ~df.identical).any():
        print(df[df.in_canonical & ~df.identical].path.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
