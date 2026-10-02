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
  Pilot 4 (LR decay; parents are mid-run full states of the Phase A extension):
  python tools/v2/launch_pilot.py --pilot pilot4_A --phase Acont --parent-pilot phaseA_ext \
      --parent-arm expected_ext --parent-file state_u01200.pt --reward-mode expected \
      --budget-overrides '{"phase_caps": {"A": 400, "B": 600, "C": 1000}}' \
      --qs 50 60 --seeds ... --arms constant decay --workers 40
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
ARMS_ACONT = {
    "expected_ext": {"reward_mode": "expected"},
    "constant": {"reward_mode": "expected"},
    "decay": {"reward_mode": "expected"},
}
ACONT_CAPS = {"A": 1200, "B": 600, "C": 1000}     # continue phase A from u400 to u1600
ACONT_FULL_STATE_AT = [800, 1200, 1600]
ARMS_B = {
    "A_joint": {"stage2_update_mode": "joint", "adv_norm_scope": "all_rows", "continuation_action_mode": "stochastic"},
    "B1_frozen_allnorm": {"stage2_update_mode": "frozen", "adv_norm_scope": "all_rows", "continuation_action_mode": "stochastic"},
    "B2_frozen_s1norm": {"stage2_update_mode": "frozen", "adv_norm_scope": "stage1_rows", "continuation_action_mode": "stochastic"},
    "B1_frozen_allnorm_mean": {"stage2_update_mode": "frozen", "adv_norm_scope": "all_rows", "continuation_action_mode": "mean"},
    "B2_frozen_s1norm_mean": {"stage2_update_mode": "frozen", "adv_norm_scope": "stage1_rows", "continuation_action_mode": "mean"},
    "B2_mean_constant": {"stage2_update_mode": "frozen", "adv_norm_scope": "stage1_rows", "continuation_action_mode": "mean"},
    "B2_mean_decay": {"stage2_update_mode": "frozen", "adv_norm_scope": "stage1_rows", "continuation_action_mode": "mean"},
}
# Arms whose whole phase follows the existing linear schedule form (lr_at, kind "linear") from
# ab_lr to the existing linear end value 3e-5 (groups G3/G4, MultiStage/Discussion/
# ROUND2_T2_FOUR_GROUP_SETTINGS_20260909.md lines 24 and 38: 3e-4 linearly to 3e-5).
DECAY_ARMS = ("decay", "B2_mean_decay")
LINEAR_END_LR = 3e-5


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
    elif phase == "Acont":
        flags.update(ARMS_ACONT[arm])
    else:
        flags.update(ARMS_B[arm])
        flags["reward_mode"] = reward_mode
    cfg = {"schema": "v2_run_config/1", "base_commit": BASE_COMMIT, "pilot": pilot, "arm": arm,
           "run": rec["run"], "q": q, "seed": seed,
           "mode": "phase_A_continue" if phase == "Acont" else f"phase_{phase}", "fixed_budget": True,
           "flags": flags, "parent_checkpoint": None, "parent_sha256": None, "record": rec,
           "threads_per_process": threads, "budget_overrides": overrides,
           "record_source_manifest": src}
    if phase == "Acont":
        cfg["budget_overrides"] = dict(overrides) or {"phase_caps": dict(ACONT_CAPS)}
        cfg["full_state_at"] = list(ACONT_FULL_STATE_AT)
    if phase in ("B", "Acont"):
        cfg["parent_checkpoint"] = parent
        cfg["parent_sha256"] = sha256_file(parent)
    if arm in DECAY_ARMS:
        ph = "A" if phase == "Acont" else "B"
        caps = cfg["budget_overrides"].get("phase_caps", rec["protocol"]["phase_caps"])
        cfg["lr_decay"] = {"phase": ph, "start_lr": float(rec["lr_schedule"]["ab_lr"]),
                           "end_lr": LINEAR_END_LR, "local_first": 1, "local_last": int(caps[ph])}
    return cfg, out


def main() -> int:
    """Write configs and run them with a bounded process pool."""
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", required=True)
    p.add_argument("--phase", choices=("A", "B", "Acont"), required=True)
    p.add_argument("--qs", type=int, nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--root", default=str(ROOT / "results" / "v2_pilots"))
    p.add_argument("--parent-pilot")
    p.add_argument("--parent-arm")
    p.add_argument("--parent-file", default="state_end_A.pt", help="full-state file inside the parent run dir")
    p.add_argument("--reward-mode", choices=("sampled", "expected"))
    p.add_argument("--budget-overrides", default="{}", help="JSON; smoke runs only (recorded)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    arms_ok = {"A": ARMS_A, "B": ARMS_B, "Acont": ARMS_ACONT}[a.phase]
    for arm in a.arms:
        if arm not in arms_ok:
            p.error(f"arm {arm} not valid for phase {a.phase}: {sorted(arms_ok)}")
    if a.phase in ("B", "Acont") and not (a.parent_pilot and a.parent_arm and a.reward_mode):
        p.error("phase B / Acont need --parent-pilot, --parent-arm and --reward-mode")
    overrides = json.loads(a.budget_overrides)
    jobs = []
    for q in a.qs:
        for seed in a.seeds:
            parent = None
            if a.phase in ("B", "Acont"):
                parent = os.path.join(a.root, a.parent_pilot, f"q{q}", f"seed{seed}", a.parent_arm, a.parent_file)
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
