#!/usr/bin/env python3
"""Which steps between seeding and the first Phase A update draw from a process-global RNG.

Seeds torch / numpy legacy / Python random with the run seed (as run_v2_T2_locked.run_pipeline does),
then performs the construction steps of the locked pipeline one at a time and records, after each,
whether each global RNG state changed. Steps: torch.nn.Linear (PyTorch default initializer), the
BetaActor, the Critic, CurriculumPPOv2 (actor + critic + opponent deep copy + Adam), StartSampler,
and the complete Run(cfg) built from the v1.1 protocol. Read-only diagnostic.

Usage: python tools/v2/global_rng_trace.py [--q 50 --seed 10501]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.ppo_curriculum import BetaActor, Critic, PPOConfig  # noqa: E402
from agents.ppo_curriculum_v2 import CurriculumPPOv2  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402
from run import run_v2_T2_locked as L  # noqa: E402
from run.run_v2_stagewise import Run  # noqa: E402

OUT = ROOT / "results" / "v2_T2_locked" / "v1_1" / "global_rng_trace.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--q", type=int, default=50)
    p.add_argument("--seed", type=int, default=10501)
    a = p.parse_args()
    proto = json.load(open(L.PROTOCOL_PATH))
    cfg = L.build_config(proto, a.q, a.seed, tempfile.mkdtemp())
    spec = GameSpec(**proto["records"][str(a.q)]["game"])
    pc = PPOConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in proto["records"][str(a.q)]["ppo"].items()})
    steps = [
        ("torch.nn.Linear(2, 64) (PyTorch default init)", lambda: torch.nn.Linear(2, 64)),
        ("agents.ppo_curriculum.BetaActor(...) with explicit generator", lambda: BetaActor(64, 100.0, 1e-6, torch.Generator().manual_seed(0))),
        ("agents.ppo_curriculum.Critic(...) with explicit generator", lambda: Critic(64, torch.Generator().manual_seed(0))),
        ("agents.ppo_curriculum_v2.CurriculumPPOv2(...)", lambda: CurriculumPPOv2(pc, torch.Generator().manual_seed(0), np.random.default_rng(0))),
        ("envs.curriculum_env.StartSampler(...)", lambda: StartSampler(spec, 10)),
        ("run.run_v2_stagewise.Run(cfg) (complete)", lambda: Run(cfg, cfg["record"]["output_dir"])),
    ]
    rows = []
    for name, fn in steps:
        L.seed_globals(a.seed)
        before = L.digests(L.global_states())
        fn()
        after = L.digests(L.global_states())
        rows.append({"step": name, **{f"{k}_changed": before[k] != after[k] for k in L.GLOBAL_RNGS}})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"q": a.q, "seed": a.seed, "steps": rows}, open(OUT, "w"), indent=1)
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
