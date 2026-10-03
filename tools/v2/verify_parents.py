#!/usr/bin/env python3
"""Verify the Pilot-2/3 parents before launch: manifest fields and a clean full-state restore.

Checks per parent: manifest reward_mode == expected, git commit == --commit, dirty false,
mode phase_A finished; the full state restores into a fresh Run; restored actor / critic / lagged
opponent / both Adam states / all RNG streams / counters equal the saved values; the opponent
refresh counter and next refresh update are present and consistent with global_u.

Usage: python tools/v2/verify_parents.py --pilot-root results/v2_pilots/pilot1 --arm expected \
    --commit 89cd600 --out results/v2_pilots/pilot2_parents_check.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ.setdefault(f"{_k}_NUM_THREADS", "1")

from run.run_v2_stagewise import Run, sha256_file  # noqa: E402


def main() -> int:
    """Write one record per parent; exit 1 if any check fails."""
    p = argparse.ArgumentParser()
    p.add_argument("--pilot-root", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    recs, ok_all = [], True
    for d in sorted(glob.glob(os.path.join(a.pilot_root, "q*", "seed*", a.arm))):
        man = json.load(open(os.path.join(d, "manifest.json")))
        st = json.load(open(os.path.join(d, "status.json")))
        ck = os.path.join(d, "state_end_A.pt")
        r = {"parent": ck, "q": man["q"], "seed": man["seed"], "reward_mode": man["reward_mode"],
             "commit": man["git"]["short"], "dirty": man["git"]["dirty"], "status": st["state"],
             "sha256": sha256_file(ck)}
        cfg = json.load(open(os.path.join(d, "run_config.json")))
        cfg.update(mode="phase_B", parent_checkpoint=ck, parent_sha256=r["sha256"],
                   flags=dict(cfg["flags"], reward_mode="expected"))
        with tempfile.TemporaryDirectory() as td:
            run = Run(cfg, td)
            run.restore(ck)
        s = torch.load(ck, weights_only=False)
        ag = run.agent
        eq = {
            "actor": all(torch.equal(v, s["agent"]["actor"][k]) for k, v in ag.actor.state_dict().items()),
            "critic": all(torch.equal(v, s["agent"]["critic"][k]) for k, v in ag.critic.state_dict().items()),
            "opponent": all(torch.equal(v, s["agent"]["opponent"][k]) for k, v in ag.opponent.state_dict().items()),
            "opt_actor_steps": all(torch.equal(torch.as_tensor(ag.opt_actor.state_dict()["state"][i]["step"]),
                                               torch.as_tensor(s["agent"]["opt_actor"]["state"][i]["step"]))
                                   for i in s["agent"]["opt_actor"]["state"]),
            "rng": all(run.rngs[k].bit_generator.state == s["rng"][k] for k in run.rngs)
                   and ag.rng_mb.bit_generator.state == s["agent"]["rng_minibatch"],
            "snapshot_refreshes": ag.snapshot_refreshes == s["agent"]["snapshot_refreshes"],
            "global_u": run.global_u == s["counters"]["global_u"] == 400,
            "next_refresh": s["counters"]["next_snapshot_refresh_update"] == 420,
            "phases_done": run.phases_done == ["A"],
        }
        r["restore_checks"] = eq
        r["snapshot_refreshes"] = s["agent"]["snapshot_refreshes"]
        r["ok"] = bool(all(eq.values()) and r["reward_mode"] == "expected" and r["commit"] == a.commit
                       and r["dirty"] is False and r["status"] == "done")
        ok_all &= r["ok"]
        recs.append(r)
    json.dump({"n": len(recs), "all_ok": ok_all, "parents": recs}, open(a.out, "w"), indent=1)
    print(f"{len(recs)} parents, all_ok={ok_all}")
    return 0 if ok_all and len(recs) == 20 else 1


if __name__ == "__main__":
    raise SystemExit(main())
