#!/usr/bin/env python3
"""Compare two runner output directories field by field (training-side identity check).

Timing fields (keys ending in '_sec', 'seconds', or containing 'wall' / 'cpu') and run metadata
(pid, host, cmd, output paths, start/end time, git commit) are excluded and listed.

Usage: python tools/v2/compare_runs.py DIR_A DIR_B
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

SKIP_SUBSTR = ("wall", "cpu")
SKIP_KEYS = {"pid", "host", "cmd", "output_dir", "path", "weights_npz", "config_ref", "git_commit",
             "start_time", "end_time", "smoke_root"}


def _walk(a, b, path, diffs, skipped):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in SKIP_KEYS or k == "seconds" or k.endswith("_sec") or any(s in k for s in SKIP_SUBSTR):
                skipped.add(k)
                continue
            if k not in a or k not in b:
                diffs.append(f"{path}/{k}: missing on one side")
                continue
            _walk(a[k], b[k], f"{path}/{k}", diffs, skipped)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, f"{path}[{i}]", diffs, skipped)
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b)):
        diffs.append(f"{path}: {a!r} != {b!r}")


def main() -> int:
    """Print the comparison and return 0 iff everything compared is identical."""
    da, db = sys.argv[1], sys.argv[2]
    bad = 0
    for f in ("train_history.json", "final_eval.json"):
        diffs, skipped = [], set()
        _walk(json.load(open(f"{da}/{f}")), json.load(open(f"{db}/{f}")), f, diffs, skipped)
        print(f"{f}: {len(diffs)} differing leaves; excluded keys: {sorted(skipped)}")
        for d in diffs[:20]:
            print("   ", d)
        bad += len(diffs)
    for f in ("arrays.npz", "checkpoint_weights.npz", "phase_A_exit_arrays.npz", "phase_B_exit_arrays.npz"):
        A, B = np.load(f"{da}/{f}"), np.load(f"{db}/{f}")
        nd = [k for k in sorted(set(A.files) | set(B.files))
              if k not in A.files or k not in B.files or not np.array_equal(A[k], B[k], equal_nan=True)]
        print(f"{f}: {len(A.files)} arrays, {len(nd)} differ {nd[:10]}")
        bad += len(nd)
    ca = torch.load(f"{da}/checkpoint.pt", weights_only=False)
    cb = torch.load(f"{db}/checkpoint.pt", weights_only=False)
    nd = 0
    for part in ("actor", "critic", "opponent"):
        nd += sum(not torch.equal(ca[part][k], cb[part][k]) for k in ca[part])
    for part in ("opt_actor", "opt_critic"):
        for pid, st in ca[part]["state"].items():
            for k, v in st.items():
                nd += int(not torch.equal(torch.as_tensor(v), torch.as_tensor(cb[part]["state"][pid][k])))
    print(f"checkpoint.pt: {nd} differing tensors (actor/critic/opponent/optimizer states)")
    bad += nd
    print("IDENTICAL" if bad == 0 else f"DIFFERENT ({bad})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
