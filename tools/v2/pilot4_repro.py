#!/usr/bin/env python3
"""Pilot 4 section 2a reproducibility: 'constant' arm (u1200 -> u1600) vs the Phase A extension.

For each (q, seed) compares the re-entered 'constant' run with
results/v2_pilots/phaseA_ext/q<q>/seed<seed>/expected_ext over updates 1201..1600:
  - weight exports u1225..u1600 and checkpoint_weights.npz (every array, bit-exact);
  - state_u01600.pt agent tensors (actor, critic, opponent, both Adam states) and RNG states;
  - the per-update training history (every key except timing) and the per-update RNG positions
    (v2_updates.csv rngpos_* for env, learn, opp, start, minibatch);
  - the verifier-call updates of both runs (they differ, because the phase-local cadence
    restarts at u1200); identical RNG positions at every update despite different call updates
    confirm that the verifier consumes no RNG.

Usage: python tools/v2/pilot4_repro.py            (comparison)
       python tools/v2/pilot4_repro.py --verifier-rng   (direct no-RNG check of the verifier)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
PX = ROOT / "results" / "v2_pilots" / "phaseA_ext"
P4A = ROOT / "results" / "v2_pilots" / "pilot4_A"
OUT = ROOT / "results" / "v2_pilots" / "pilot4" / "analysis"
TIMING = ("time_sec", "update_wall_sec")


def _tensors_equal(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_tensors_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_tensors_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def main() -> int:
    rows = []
    for q in (50, 60):
        for seed in range(10501, 10511):
            x = PX / f"q{q}" / f"seed{seed}" / "expected_ext"
            c = P4A / f"q{q}" / f"seed{seed}" / "constant"
            ex_c = sorted(c.glob("weights/u*.npz"))
            pairs = [(x / "weights" / f.name, f) for f in ex_c]
            w_eq = all(all(np.array_equal(np.load(a)[k], np.load(b)[k]) for k in np.load(a).files)
                       and set(np.load(a).files) == set(np.load(b).files) for a, b in pairs)
            fin_eq = all(np.array_equal(np.load(x / "weights" / "u01600.npz")[k], np.load(c / "checkpoint_weights.npz")[k])
                         for k in np.load(c / "checkpoint_weights.npz").files)
            sx = torch.load(x / "state_u01600.pt", weights_only=False)
            sc = torch.load(c / "state_u01600.pt", weights_only=False)
            agent_eq = all(_tensors_equal(sx["agent"][k], sc["agent"][k])
                           for k in ("actor", "critic", "opponent", "opt_actor", "opt_critic", "rng_minibatch"))
            rng_eq = _tensors_equal(sx["rng"], sc["rng"]) and _tensors_equal(sx["torch_generator_state"], sc["torch_generator_state"])
            hx = [h for h in json.load(open(x / "train_history.json"))["history"] if 1200 < h["update"] <= 1600]
            hc = json.load(open(c / "train_history.json"))["history"]
            strip = lambda hs: [{k: v for k, v in h.items() if k not in TIMING and k != "local"} for h in hs]  # noqa: E731
            hist_eq = strip(hx) == strip(hc)
            ux = pd.read_csv(x / "v2_updates.csv").set_index("update").loc[1201:1600]
            uc = pd.read_csv(c / "v2_updates.csv").set_index("update")
            rcols = [k for k in uc.columns if k.startswith("rngpos_")]
            rngpos_eq = ux[rcols].equals(uc[rcols])
            vx = [v["update"] for v in json.load(open(x / "train_history.json"))["verifier_calls"] if 1200 < v["update"] <= 1600]
            vc = [v["update"] for v in json.load(open(c / "train_history.json"))["verifier_calls"]]
            rows.append({"q": q, "seed": seed, "n_exports_compared": len(pairs), "weight_exports_identical": w_eq,
                         "final_weights_identical_to_ext_u1600": fin_eq, "state_u1600_agent_identical": agent_eq,
                         "state_u1600_rng_identical": rng_eq, "history_1201_1600_identical": hist_eq,
                         "rng_positions_every_update_identical": rngpos_eq, "n_updates": len(hc),
                         "verifier_updates_ext": " ".join(map(str, vx)), "verifier_updates_constant": " ".join(map(str, vc)),
                         "verifier_cadence_differs": vx != vc})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "repro_2a_constant_vs_ext.csv", index=False)
    flags = [k for k in df.columns if k.endswith("identical")]
    print(df[["q", "seed"] + flags + ["verifier_cadence_differs"]].to_string())
    print("ALL IDENTICAL:", bool(df[flags].all().all()))
    return 0 if df[flags].all().all() else 2


if __name__ == "__main__" and "--verifier-rng" not in __import__("sys").argv:
    raise SystemExit(main())


def verifier_rng_check() -> pd.DataFrame:
    """Direct check: verifier calls (dev + final tier) leave every RNG state unchanged.

    Restores each 2a parent (state_u01200.pt) into a Run, records all RNG states (numpy env/learn/
    opp/start, minibatch, torch generator, torch/numpy/python global), calls verify_candidate on
    both tiers three times, and compares.
    """
    import copy
    import random
    import sys
    sys.path.insert(0, str(ROOT))
    from run.run_v2_stagewise import Run

    def snap(r):
        return (copy.deepcopy({k: g.bit_generator.state for k, g in r.rngs.items()}),
                copy.deepcopy(r.agent.rng_mb.bit_generator.state), r.torch_gen.get_state().clone(),
                torch.get_rng_state().clone(), copy.deepcopy(np.random.get_state()), random.getstate())
    rows = []
    for q in (50, 60):
        for seed in range(10501, 10511):
            d = P4A / f"q{q}" / f"seed{seed}" / "constant"
            cfg = json.load(open(d / "run_config.json"))
            r = Run(cfg, str(OUT / "_tmp_verifier_rng"))
            r.restore(cfg["parent_checkpoint"])
            before = snap(r)
            for _ in range(3):
                for tier in (r.dev_cfg, r.fin_cfg):
                    ev, err = r.verify_candidate(tier)
                    assert err is None
            after = snap(r)
            same = all(_tensors_equal(a, b) for a, b in zip(before, after))
            rows.append({"q": q, "seed": seed, "rng_states_unchanged_after_6_verifier_calls": bool(same)})
    return pd.DataFrame(rows)


if __name__ == "__main__" and "--verifier-rng" in __import__("sys").argv:
    v = verifier_rng_check()
    v.to_csv(OUT / "verifier_consumes_no_rng.csv", index=False)
    print(v.to_string(), "\nALL UNCHANGED:", bool(v.iloc[:, 2].all()))
