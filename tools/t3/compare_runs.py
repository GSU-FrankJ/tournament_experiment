#!/usr/bin/env python3
"""Bit-for-bit comparison of two runs of the same config (T3-R1 determinism check): weight exports, evaluation / freeze arrays,
continuation tables, bin maps and visitation counts, the CSVs without their wall-clock columns, and the final network and RNG
states. The two runs may differ in their output directory, commit and wall-clock columns only.

Usage:
    python tools/t3/compare_runs.py <kind: bw|joint> <run_dir_a> <run_dir_b>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

NPZ = {"joint": ["weights/*.npz", "final_eval_final.npz", "final_eval_development.npz", "visitation.npz"],
       "bw": ["weights/*.npz", "freeze_stage*_final.npz", "freeze_stage*_development.npz", "continuation_table_stage*.npz",
              "ms_binmaps_stage*.npz"]}
CSV = {"joint": {"joint_updates.csv": {"update_wall_sec"}, "joint_checks.csv": {"verifier_sec"}},
       "bw": {"ms_updates.csv": {"update_wall_sec"}, "ms_checks_stage3.csv": {"verifier_sec"},
              "ms_checks_stage2.csv": {"verifier_sec"}, "ms_checks_stage1.csv": {"verifier_sec"}}}
PT = {"joint": "state_end.pt", "bw": "state_end_stage1.pt"}


def rows(path: Path, wall: set) -> List[dict]:
    """CSV rows without the wall-clock columns."""
    with open(path, newline="") as f:
        return [{k: v for k, v in r.items() if k not in wall} for r in csv.DictReader(f)]


def same_array(x: np.ndarray, y: np.ndarray) -> bool:
    """Exact equality (NaN equal to NaN for floating arrays; strings compared as they are)."""
    return x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y, equal_nan=x.dtype.kind in "fc")


def main() -> int:
    """Compare and print one line per class; exit 1 on any difference."""
    kind, a, b = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    ok = True
    n = 0
    for pat in NPZ[kind]:
        for fa in sorted(a.glob(pat)):
            fb = b / fa.relative_to(a)
            za, zb = np.load(fa), np.load(fb)
            same = za.files == zb.files and all(same_array(za[k], zb[k]) for k in za.files)
            n += 1
            if not same:
                ok = False
                print("DIFFERENT", fa.relative_to(a))
    print(f"{n} npz files compared, all identical: {ok}")
    for name, wall in CSV[kind].items():
        same = rows(a / name, wall) == rows(b / name, wall)
        ok = ok and same
        print(f"{name} identical (wall-clock columns excluded): {same}")
    sa = torch.load(a / PT[kind], weights_only=False, map_location="cpu")
    sb = torch.load(b / PT[kind], weights_only=False, map_location="cpu")
    nets = [("actor", sa["agent"]["actor"], sb["agent"]["actor"]), ("critic", sa["agent"]["critic"], sb["agent"]["critic"])]
    same = all(torch.equal(x[k], y[k]) for _, x, y in nets for k in x) and sa["rng"] == sb["rng"]
    if kind == "bw":
        same = same and all(torch.equal(sa["frozen_stages"][s][k], sb["frozen_stages"][s][k])
                            for s in sa["frozen_stages"] for k in sa["frozen_stages"][s])
    ok = ok and same
    print(f"final actor / critic{' / frozen stages' if kind == 'bw' else ''} and stream states identical: {same}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
