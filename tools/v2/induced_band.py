#!/usr/bin/env python3
"""Induced stage-1 target by residual minimization (decision D2), final verifier tier.

For a stage-2 mapping e_hat_2, Delta_1(e; e_hat_2) (the verifier's one-step root residual with
both players' root action = e and continuation e_hat_2) is swept over a grid anchored at e1*(0)
with spacing STEP; e~1 = argmin, band = {e : Delta_1 <= min + floor}, floor =
max(Delta_1(e1*; e2*) on the final tier for that q, 1e-12 DW).

Sub-commands
  calib    floors and calibration gate (e_hat_2 = e2*) for q in {50, 60}
  parents  one sweep per Pilot-1 'expected' parent (the frozen stage 2 of Pilot 2 B1/B2 and of
           Pilot 3): range [0.4 e1*, 1.7 e1*]
  joint    one sweep per Pilot-2 A_joint weight export (live stage 2): range
           [min(e1, e1*) - MARGIN, max(e1, e1*) + MARGIN]
Outputs under results/v2_pilots/induced_band/ (CSV + NPZ of every sweep).

Usage: python tools/v2/induced_band.py {calib|parents|joint} [--workers N]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))

from common import analytic_policy, spec_for  # noqa: E402
from agents.ppo_curriculum import CurriculumPPO, PPOConfig  # noqa: E402
from run.run_final_dp_br import make_policy_fns  # noqa: E402
from utils.dp_br_verifier import FINAL_CONFIG  # noqa: E402
from utils.v2_metrics import induced_band, stage1_residual_sweep, sweep_grid  # noqa: E402

OUT = ROOT / "results" / "v2_pilots" / "induced_band"
STEP = 0.01
MARGIN = 2.0
PARENT_RANGE = (0.4, 1.7)


def actor_policy(path: str, spec):
    """(mean_fn) of an actor stored in a weight-export NPZ or a v2 full-state checkpoint."""
    agent = CurriculumPPO(PPOConfig(), torch.Generator().manual_seed(0), np.random.default_rng(0))
    if path.endswith(".npz"):
        w = np.load(path)
        sd = {k[len("actor."):]: torch.as_tensor(w[k]) for k in w.files if k.startswith("actor.")}
    else:
        sd = torch.load(path, weights_only=False)["agent"]["actor"]
    agent.actor.load_state_dict(sd)
    return make_policy_fns(agent, spec)[0]


def floors() -> dict:
    """Delta_1(e1*; e2*) on the final tier per q."""
    out = {}
    for q in (50.0, 60.0):
        spec = spec_for(q)
        eq = analytic_policy(spec)
        g1 = float(eq(1, np.zeros(1))[0])
        cal = float(stage1_residual_sweep(eq, spec, FINAL_CONFIG, np.array([g1]))[0])
        out[int(q)] = {"g1": g1, "calibration_delta1": cal, "floor": max(cal, 1e-12 * spec.dw)}
    return out


def _sweep(job):
    tag, q, policy_path, lo, hi, floor, npz_out = job
    spec = spec_for(q)
    g1 = float(analytic_policy(spec)(1, np.zeros(1))[0])
    pol = actor_policy(policy_path, spec)
    E = sweep_grid(g1, lo, hi, STEP)
    D = stage1_residual_sweep(pol, spec, FINAL_CONFIG, E)
    np.savez(npz_out, e_sweep=E, delta1=D)
    b = induced_band(E, D, floor)
    return {**tag, "q": int(q), "g1": g1, "policy": os.path.relpath(policy_path, ROOT), "step": STEP,
            "sweep_npz": os.path.relpath(npz_out, ROOT), "delta1_min_over_dw": b["delta1_min"] / spec.dw, **b}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("calib", "parents", "joint"))
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    os.makedirs(OUT / "sweeps", exist_ok=True)
    fl = floors()
    json.dump(fl, open(OUT / "floors.json", "w"), indent=1)
    if a.cmd == "calib":
        rows = []
        for q in (50.0, 60.0):
            spec = spec_for(q)
            eq = analytic_policy(spec)
            g1 = fl[int(q)]["g1"]
            E = sweep_grid(g1, g1 - 5.0, g1 + 5.0, STEP)
            D = stage1_residual_sweep(eq, spec, FINAL_CONFIG, E)
            np.savez(OUT / "sweeps" / f"calib_q{int(q)}.npz", e_sweep=E, delta1=D)
            b = induced_band(E, D, fl[int(q)]["floor"])
            rows.append({"q": int(q), "g1": g1, **b, "contains_e1star": bool(b["band_lo"] <= g1 <= b["band_hi"]),
                         "band_width_over_e1star": (b["band_hi"] - b["band_lo"]) / g1,
                         "e1star_is_sweep_node": bool(np.any(E == g1))})
        pd.DataFrame(rows).to_csv(OUT / "calibration.csv", index=False)
        print(pd.DataFrame(rows).to_string())
        if not all(r["contains_e1star"] for r in rows):
            print("CALIBRATION GATE FAILED")
            return 2
        return 0
    jobs = []
    if a.cmd == "parents":
        for ck in sorted(glob.glob(str(ROOT / "results/v2_pilots/pilot1/q*/seed*/expected/state_end_A.pt"))):
            man = json.load(open(os.path.join(os.path.dirname(ck), "manifest.json")))
            q = float(man["q"])
            g1 = fl[int(q)]["g1"]
            jobs.append(({"seed": man["seed"], "source": "parent"}, q, ck, PARENT_RANGE[0] * g1, PARENT_RANGE[1] * g1,
                         fl[int(q)]["floor"], str(OUT / "sweeps" / f"parent_q{int(q)}_s{man['seed']}.npz")))
        name = "parent_bands.csv"
    else:
        for d in sorted(glob.glob(str(ROOT / "results/v2_pilots/pilot2/q*/seed*/A_joint"))):
            man = json.load(open(os.path.join(d, "manifest.json")))
            q = float(man["q"])
            spec = spec_for(q)
            g1 = fl[int(q)]["g1"]
            for f in sorted(glob.glob(os.path.join(d, "weights", "u*.npz"))):
                u = int(os.path.basename(f)[1:6])
                if u <= 400:
                    continue
                e1 = float(actor_policy(f, spec)(1, np.zeros(1))[0])
                jobs.append(({"seed": man["seed"], "arm": "A_joint", "update": u, "e1_at_0": e1, "source": "weights"},
                             q, f, min(e1, g1) - MARGIN, max(e1, g1) + MARGIN, fl[int(q)]["floor"],
                             str(OUT / "sweeps" / f"joint_q{int(q)}_s{man['seed']}_u{u:05d}.npz")))
        name = "joint_export_bands.csv"
    with Pool(a.workers) as pool:
        rows = pool.map(_sweep, jobs, chunksize=1)
    pd.DataFrame(rows).to_csv(OUT / name, index=False)
    print(f"{len(rows)} sweeps -> {OUT / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
