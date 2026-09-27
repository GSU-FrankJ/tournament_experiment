#!/usr/bin/env python3
"""Run the published T2 settings in a separate output directory."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
COHORTS = {
    "confirmation": "two_stage_confirmation_T2_20260922",
    "e1": "two_stage_E1_q50_p_20260923",
    "restarts": "two_stage_q50_restarts_20260924",
    "precision": "two_stage_q50_precision_20260924",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--q", type=int, choices=(50, 60), help="Run only this q.")
    parser.add_argument("--seed", type=int, help="Run only this archived seed.")
    parser.add_argument("--workers", type=int, help="Concurrent processes; default is the archived value.")
    parser.add_argument("--out-root", type=Path, help="New batch directory; existing batches are never overwritten.")
    parser.add_argument("--smoke", action="store_true",
                        help="Run only the first selected record with 2/2/3 updates; flow check, not research data.")
    args = parser.parse_args()
    cohort = COHORTS[args.cohort]
    source = PROJECT / "experiments" / cohort / "manifest.json"
    manifest = copy.deepcopy(json.loads(source.read_text()))
    jobs = [r for r in manifest["runs"]
            if (args.q is None or r["q"] == args.q)
            and (args.seed is None or r["seed"] == args.seed)]
    if not jobs:
        parser.error("No archived record matches the requested q and seed.")
    if args.smoke:
        jobs = jobs[:1]
    workers = args.workers if args.workers is not None else manifest["max_concurrent_processes"]
    if workers < 1:
        parser.error("--workers must be positive.")
    out = (args.out_root or PROJECT / "scratch" / "two_stage" /
           (args.cohort + ("_smoke" if args.smoke else ""))).resolve()
    archived_root = (PROJECT / "experiments").resolve()
    if out == archived_root or archived_root in out.parents:
        parser.error("--out-root must be outside experiments/, which contains archived results.")
    status_path = out / "launch_status.json"
    manifest_path = out / "manifest.json"
    if status_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Existing batch at {out}; choose a new --out-root.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    for rec in jobs:
        rec["output_dir"] = str(out / "runs" / rec["group"] / rec["run"])
    manifest.update(runs=jobs, result_root=str(out), worktree=str(PROJECT),
                    runner=str(PROJECT / "run" / "run_final_dp_br_round3_dense.py"),
                    python=sys.executable, max_concurrent_processes=workers,
                    publication_source_manifest=str(source), publication_smoke=args.smoke,
                    publication_selected_run_count=len(jobs))
    # Existing exact software-version, schema, verifier and overwrite checks remain in the runner.
    # No training/verifier setting is rewritten here.
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    begin = time.monotonic()
    state = {"state": "running", "manifest": str(manifest_path),
             "planned_runs": len(jobs), "smoke": args.smoke, "runs": []}

    def save():
        status_path.write_text(json.dumps(state, indent=2) + "\n")

    def run_one(rec):
        log_path = out / "logs" / f"{rec['run']}.log"
        command = [sys.executable, "-B", "-u", manifest["runner"],
                   "--manifest", str(manifest_path), "--group", rec["group"],
                   "--q", str(rec["q"]), "--seed", str(rec["seed"])]
        if args.smoke:
            command += ["--smoke", "--smoke-phase-caps", "2,2,3",
                        "--smoke-warmup", "1", "--smoke-stability-every", "1",
                        "--smoke-timeout", "1", "--smoke-direct-rollout-episodes", "1000",
                        "--smoke-direct-rollout-reps", "1", "--smoke-root", str(out / "smoke")]
        started = time.monotonic()
        with log_path.open("x") as log:
            result = subprocess.run(command, cwd=PROJECT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        return {"run": rec["run"], "q": rec["q"], "seed": rec["seed"],
                "returncode": result.returncode, "wall_sec": time.monotonic() - started,
                "log": str(log_path), "output_dir": str(out / "smoke" / rec["group"] / rec["run"])
                if args.smoke else rec["output_dir"]}

    save()
    print(f"Running {len(jobs)} record(s); outputs: {out}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, rec): rec for rec in jobs}
        for future in as_completed(futures):
            rec = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"run": rec["run"], "returncode": None,
                          "error": f"{type(exc).__name__}: {exc}"}
            state["runs"].append(result)
            save()
            print(json.dumps(result), flush=True)
    ok = all(r.get("returncode") == 0 for r in state["runs"])
    state.update(state="done" if ok else "failed", wall_sec=time.monotonic() - begin)
    save()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
