#!/usr/bin/env python3
"""RNG-stream alignment between paired arms (rule A6), after one episode and after N updates.

Streams: env (shocks), learn (learner Beta draws), opp (opponent Beta draws), start (starts and
roles) and minibatch (held by the agent). Two arms are aligned on a stream when the numpy
bit-generator states are identical.

Usage: python tools/v2/rng_alignment.py --parent <state_end_A.pt> --out <json> [--updates 20]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run.run_v2_stagewise import Run  # noqa: E402
from run.v2_rollout import collect_batch_v2  # noqa: E402

STREAMS = ("env", "learn", "opp", "start", "minibatch")
B_ARMS = {
    "A_joint": {"stage2_update_mode": "joint", "adv_norm_scope": "all_rows", "continuation_action_mode": "stochastic"},
    "B1_frozen_allnorm": {"stage2_update_mode": "frozen", "adv_norm_scope": "all_rows", "continuation_action_mode": "stochastic"},
    "B2_frozen_s1norm": {"stage2_update_mode": "frozen", "adv_norm_scope": "stage1_rows", "continuation_action_mode": "stochastic"},
    "B1_frozen_allnorm_mean": {"stage2_update_mode": "frozen", "adv_norm_scope": "all_rows", "continuation_action_mode": "mean"},
}


def base_config(seed: int = 10501, q: int = 50) -> Dict:
    """In-process config template (q=50 record of the precision cohort, seed replaced)."""
    man = json.load(open(ROOT / "experiments/two_stage_q50_precision_20260924/manifest.json"))
    rec = copy.deepcopy([r for r in man["runs"] if r["seed"] == 10231][0])
    rec.update(seed=seed, run=f"test_q{q}_s{seed}", output_dir="unused")
    return {"schema": "v2_run_config/1", "base_commit": "657f54a", "pilot": "test", "arm": "x",
            "run": rec["run"], "q": q, "seed": seed, "mode": "phase_A", "fixed_budget": True,
            "flags": {"reward_mode": "sampled", "stage2_update_mode": "joint", "adv_norm_scope": "all_rows",
                      "continuation_action_mode": "stochastic"},
            "parent_checkpoint": None, "parent_sha256": None, "record": rec,
            "threads_per_process": man["threads_per_process"],
            "budget_overrides": {"phase_caps": {"A": 6, "B": 20, "C": 1}, "warmup": 5,
                                 "stability_every": 5, "verifier_timeout": 5}}


def make_run(mode: str, flags: Dict, out: str, parent: str = None, caps: Dict = None) -> Run:
    """Construct a Run (restored from ``parent`` for phase_B)."""
    cfg = base_config()
    cfg["mode"] = mode
    cfg["flags"].update(flags)
    if caps:
        cfg["budget_overrides"]["phase_caps"].update(caps)
    if mode == "phase_B":
        cfg["parent_checkpoint"], cfg["parent_sha256"] = parent, "not-checked-in-process"
    os.makedirs(out, exist_ok=True)
    r = Run(cfg, out)
    if mode == "phase_B":
        r.restore(parent)
        if flags.get("stage2_update_mode") == "frozen":
            r.agent.freeze_stage2_snapshot()
    return r


def states(r: Run) -> Dict[str, object]:
    """Bit-generator states of every stream."""
    out = {k: copy.deepcopy(g.bit_generator.state) for k, g in r.rngs.items()}
    out["minibatch"] = copy.deepcopy(r.agent.rng_mb.bit_generator.state)
    return out


def one_episode_batch(r: Run, phase: str) -> None:
    """One rollout batch with the run's flags (the batch draws of one update)."""
    spec, P = r.spec, r.P
    n = int(P["episodes_per_update"])
    if phase == "A":
        t0 = np.full(n, spec.T)
        d0 = r.sampler.balanced(spec.T, n, r.rngs["start"])
    else:
        t0, d0 = np.ones(n, dtype=int), np.zeros(n)
    roles = r.rngs["start"].integers(0, 2, size=n)
    frozen = r.agent.frozen if r.flags["stage2_update_mode"] == "frozen" else None
    collect_batch_v2(spec, r.agent, t0, d0, roles, r.rngs["env"], r.rngs["learn"], r.rngs["opp"],
                     1.0, 1.0, P["es_bin_width"], reward_mode=r.flags["reward_mode"], frozen=frozen,
                     continuation_action_mode=r.flags["continuation_action_mode"])


def compare(sa: Dict, sb: Dict) -> Dict[str, bool]:
    """Per-stream equality."""
    return {k: sa[k] == sb[k] for k in STREAMS}


def alignment_table(parent: str, updates: int, workdir: str) -> Dict[str, object]:
    """All requested pairs, after one batch and after ``updates`` updates."""
    out: Dict[str, object] = {}
    pairs = {
        "sampled_vs_expected (phase A from init)": ("phase_A", {"reward_mode": "sampled"}, {"reward_mode": "expected"}, "A"),
        "stochastic_vs_mean (frozen B1, phase B)": ("phase_B", B_ARMS["B1_frozen_allnorm"], B_ARMS["B1_frozen_allnorm_mean"], "B"),
        "A_joint_vs_B1 (phase B)": ("phase_B", B_ARMS["A_joint"], B_ARMS["B1_frozen_allnorm"], "B"),
        "A_joint_vs_B2 (phase B)": ("phase_B", B_ARMS["A_joint"], B_ARMS["B2_frozen_s1norm"], "B"),
        "B1_vs_B2 (phase B)": ("phase_B", B_ARMS["B1_frozen_allnorm"], B_ARMS["B2_frozen_s1norm"], "B"),
    }
    for name, (mode, fa, fb, phase) in pairs.items():
        res = {}
        ra = make_run(mode, fa, os.path.join(workdir, name, "a1"), parent)
        rb = make_run(mode, fb, os.path.join(workdir, name, "b1"), parent)
        one_episode_batch(ra, phase)
        one_episode_batch(rb, phase)
        res["after_one_batch"] = compare(states(ra), states(rb))
        caps = {phase: updates}
        ra = make_run(mode, fa, os.path.join(workdir, name, "aN"), parent, caps)
        rb = make_run(mode, fb, os.path.join(workdir, name, "bN"), parent, caps)
        ra.run_phase(phase)
        rb.run_phase(phase)
        res[f"after_{updates}_updates"] = compare(states(ra), states(rb))
        out[name] = res
    return out


def main() -> int:
    """Write the alignment table."""
    p = argparse.ArgumentParser()
    p.add_argument("--parent", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--updates", type=int, default=20)
    a = p.parse_args()
    with tempfile.TemporaryDirectory() as td:
        tab = alignment_table(a.parent, a.updates, td)
    json.dump(tab, open(a.out, "w"), indent=1)
    print(json.dumps(tab, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
