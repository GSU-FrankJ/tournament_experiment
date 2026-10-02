#!/usr/bin/env python3
"""Bit-identity check of two v2 run directories: history, weights, verifier calls, end state.

Compares
  - train_history.json: history, stability, verifier_calls (excluding timing keys);
  - every weight export weights/u*.npz and checkpoint_weights.npz (every array);
  - v2_updates.csv RNG positions (all streams) at every update;
  - an end-state full checkpoint (default state_end_B.pt): agent actor / critic / opponent /
    frozen / both Adam states / minibatch RNG, the numpy streams and the torch generator.

Usage: python tools/v2/compare_full_runs.py DIR_A DIR_B [--state state_end_B.pt] [--out JSON]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

TIMING = {"time_sec", "update_wall_sec"}


def teq(a, b) -> bool:
    """Recursive exact equality for tensors / arrays / containers."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(teq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(teq(x, y) for x, y in zip(a, b))
    return a == b


def strip(xs):
    return [{k: v for k, v in x.items() if k not in TIMING} for x in xs]


def compare(a: str, b: str, state: str) -> dict:
    ha, hb = (json.load(open(os.path.join(x, "train_history.json"))) for x in (a, b))
    out = {"history_identical": strip(ha["history"]) == strip(hb["history"]),
           "stability_identical": ha["stability"] == hb["stability"],
           "verifier_calls_identical": json.dumps(strip(ha["verifier_calls"]), sort_keys=True, default=str)
           == json.dumps(strip(hb["verifier_calls"]), sort_keys=True, default=str),
           "n_updates": len(ha["history"])}
    ea = sorted(glob.glob(os.path.join(a, "weights", "u*.npz")))
    eb = sorted(glob.glob(os.path.join(b, "weights", "u*.npz")))
    out["n_weight_exports"] = len(ea)
    out["weight_exports_identical"] = [os.path.basename(x) for x in ea] == [os.path.basename(x) for x in eb] and all(
        set(np.load(x).files) == set(np.load(y).files) and all(np.array_equal(np.load(x)[k], np.load(y)[k]) for k in np.load(x).files)
        for x, y in zip(ea, eb))
    wa, wb = (np.load(os.path.join(x, "checkpoint_weights.npz")) for x in (a, b))
    out["final_weights_identical"] = set(wa.files) == set(wb.files) and all(np.array_equal(wa[k], wb[k]) for k in wa.files)
    ua, ub = (pd.read_csv(os.path.join(x, "v2_updates.csv")) for x in (a, b))
    rc = [c for c in ua.columns if c.startswith("rngpos_")]
    out["rng_positions_identical"] = ua[["update"] + rc].equals(ub[["update"] + rc])
    sa, sb = (torch.load(os.path.join(x, state), weights_only=False) for x in (a, b))
    for k in ("actor", "critic", "opponent", "frozen", "opt_actor", "opt_critic", "rng_minibatch"):
        out[f"end_state_{k}_identical"] = teq(sa["agent"].get(k), sb["agent"].get(k))
    out["end_state_rng_streams_identical"] = teq(sa["rng"], sb["rng"])
    out["end_state_torch_generator_identical"] = teq(sa["torch_generator_state"], sb["torch_generator_state"])
    out["ALL_IDENTICAL"] = all(v for k, v in out.items() if k.endswith("_identical"))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--state", default="state_end_B.pt")
    p.add_argument("--out")
    x = p.parse_args()
    r = {"dir_a": x.a, "dir_b": x.b, "state_file": x.state, **compare(x.a, x.b, x.state)}
    print(json.dumps(r, indent=1))
    if x.out:
        json.dump(r, open(x.out, "w"), indent=1)
    return 0 if r["ALL_IDENTICAL"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
