#!/usr/bin/env python3
"""v2 pilot launcher: Phase A runs, or Phase B branches from named parent checkpoints.

Layout: <root>/<pilot>/q<q>/seed<seed>/<arm>/{run_config.json, run.log, <runner outputs>}
Batch:  <root>/<pilot>/launch_<timestamp>.json (planned + finished runs)

The embedded record (game, PPO, protocol, verifier tiers, LR schedule, versions) is the archived
as-run record of the final T=2 protocol: q=50 from experiments/two_stage_q50_precision_20260924
(tel_q50_s10231), q=60 from experiments/two_stage_confirmation_T2_20260922 (first q=60 record);
only run / seed / output_dir are replaced. Every pilot run uses fixed_budget=true.

Examples (run inside tmux; see reports/v2/phase2_infra.md):
  python tools/v2/launch_pilot.py --pilot pilot1 --phase A --qs 50 60 --seeds 10501 10502 10503 \
      --arms sampled expected --workers 12
  python tools/v2/launch_pilot.py --pilot pilot2 --phase B --parent-pilot pilot1 --parent-arm expected \
      --reward-mode expected --qs 50 60 --seeds 10501 10502 10503 \
      --arms A_joint B1_frozen_allnorm B2_frozen_s1norm --workers 18
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
BASE_COMMIT = "657f54a"
SOURCES = {50: ("experiments/two_stage_q50_precision_20260924/manifest.json", 10231),
           60: ("experiments/two_stage_confirmation_T2_20260922/manifest.json", None)}
ARMS_A = {
    "sampled": {"reward_mode": "sampled"},
    "expected": {"reward_mode": "expected"},
}
ARMS_B = {
    "A_joint": {"stage2_update_mode": "joint", "adv_norm_scope": "all_rows", "continuation_action_mode": "stochastic"},
    "B1_frozen_allnorm": {"stage2_update_mode": "frozen", "adv_norm_scope": "all_rows", "continuation_action_mode": "stochastic"},
    "B2_frozen_s1norm": {"stage2_update_mode": "frozen", "adv_norm_scope": "stage1_rows", "continuation_action_mode": "stochastic"},
    "B1_frozen_allnorm_mean": {"stage2_update_mode": "frozen", "adv_norm_scope": "all_rows", "continuation_action_mode": "mean"},
    "B2_frozen_s1norm_mean": {"stage2_update_mode": "frozen", "adv_norm_scope": "stage1_rows", "continuation_action_mode": "mean"},
}


def sha256_file(path: str) -> str:
    """sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_record(q: int):
    """(record, threads_per_process) of the archived as-run source for this q."""
    path, seed = SOURCES[q]
    man = json.load(open(ROOT / path))
    recs = [r for r in man["runs"] if int(r["q"]) == q and (seed is None or int(r["seed"]) == seed)]
    return copy.deepcopy(recs[0]), man["threads_per_process"], path


def build_config(pilot, phase, q, seed, arm, reward_mode, parent, overrides, root):
    """Run config for one (q, seed, arm)."""
    rec, threads, src = source_record(q)
    out = os.path.join(root, pilot, f"q{q}", f"seed{seed}", arm)
    rec.update(run=f"{pilot}_q{q}_s{seed}_{arm}", seed=seed, output_dir=out)
    flags = {"reward_mode": "sampled", "stage2_update_mode": "joint", "adv_norm_scope": "all_rows",
             "continuation_action_mode": "stochastic"}
    if phase == "A":
        flags.update(ARMS_A[arm])
    else:
        flags.update(ARMS_B[arm])
        flags["reward_mode"] = reward_mode
    cfg = {"schema": "v2_run_config/1", "base_commit": BASE_COMMIT, "pilot": pilot, "arm": arm,
           "run": rec["run"], "q": q, "seed": seed, "mode": f"phase_{phase}", "fixed_budget": True,
           "flags": flags, "parent_checkpoint": None, "parent_sha256": None, "record": rec,
           "threads_per_process": threads, "budget_overrides": overrides,
           "record_source_manifest": src}
    if phase == "B":
        cfg["parent_checkpoint"] = parent
        cfg["parent_sha256"] = sha256_file(parent)
    return cfg, out


def main() -> int:
    """Write configs and run them with a bounded process pool."""
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", required=True)
    p.add_argument("--phase", choices=("A", "B"), required=True)
    p.add_argument("--qs", type=int, nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--root", default=str(ROOT / "results" / "v2_pilots"))
    p.add_argument("--parent-pilot")
    p.add_argument("--parent-arm")
    p.add_argument("--reward-mode", choices=("sampled", "expected"))
    p.add_argument("--budget-overrides", default="{}", help="JSON; smoke runs only (recorded)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    arms_ok = ARMS_A if a.phase == "A" else ARMS_B
    for arm in a.arms:
        if arm not in arms_ok:
            p.error(f"arm {arm} not valid for phase {a.phase}: {sorted(arms_ok)}")
    if a.phase == "B" and not (a.parent_pilot and a.parent_arm and a.reward_mode):
        p.error("phase B needs --parent-pilot, --parent-arm and --reward-mode")
    overrides = json.loads(a.budget_overrides)
    jobs = []
    for q in a.qs:
        for seed in a.seeds:
            parent = None
            if a.phase == "B":
                parent = os.path.join(a.root, a.parent_pilot, f"q{q}", f"seed{seed}", a.parent_arm, "state_end_A.pt")
                if not os.path.exists(parent):
                    p.error(f"missing parent {parent}")
            for arm in a.arms:
                cfg, out = build_config(a.pilot, a.phase, q, seed, arm, a.reward_mode, parent, overrides, a.root)
                if os.path.exists(os.path.join(out, "status.json")):
                    p.error(f"{out} already has a run; choose a new --pilot")
                jobs.append((cfg, out))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    batch_path = os.path.join(a.root, a.pilot, f"launch_{stamp}.json")
    os.makedirs(os.path.dirname(batch_path), exist_ok=True)
    state = {"pilot": a.pilot, "phase": a.phase, "argv": sys.argv, "workers": a.workers,
             "nproc": os.cpu_count(), "loadavg_at_start": os.getloadavg(), "planned": [o for _, o in jobs],
             "runs": [], "state": "running", "started": stamp}
    for cfg, out in jobs:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "run_config.json"), "w") as f:
            json.dump(cfg, f, indent=1)
    json.dump(state, open(batch_path, "w"), indent=1)
    print(f"{len(jobs)} run(s); batch record {batch_path}", flush=True)
    if a.dry_run:
        return 0
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")

    def one(job):
        cfg, out = job
        t0 = time.monotonic()
        with open(os.path.join(out, "run.log"), "x") as log:
            r = subprocess.run([PY, "-B", "-u", str(ROOT / "run" / "run_v2_stagewise.py"),
                                "--config", os.path.join(out, "run_config.json"), "--out-dir", out],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        return {"out": out, "returncode": r.returncode, "wall_sec": time.monotonic() - t0}

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for fut in as_completed([pool.submit(one, j) for j in jobs]):
            res = fut.result()
            state["runs"].append(res)
            json.dump(state, open(batch_path, "w"), indent=1)
            print(json.dumps(res), flush=True)
    state["state"] = "done" if all(r["returncode"] == 0 for r in state["runs"]) else "failed"
    state["loadavg_at_end"] = os.getloadavg()
    json.dump(state, open(batch_path, "w"), indent=1)
    return 0 if state["state"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
