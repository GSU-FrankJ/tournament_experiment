#!/usr/bin/env python3
"""Start every run of one or more fixed manifests with one shared worker limit.

Each manifest seed is started at most once; an existing run directory is never
overwritten (recorded as ``skipped_existing_dir``). Children run single-threaded.
Exit codes are kept, including signal terminations (negative return codes).

    .venv/bin/python -B experiments/three_stage_implementation_pilot_20260924/launch.py \
        --manifest experiments/three_stage_implementation_pilot_20260924/manifests/pilot_q60.json --max-workers 3
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import E, PYTHON, now_iso, write_json_atomic  # noqa: E402


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", action="append", required=True)
    p.add_argument("--max-workers", type=int, required=True)
    args = p.parse_args()
    manifests = [Path(m).resolve() for m in args.manifest]
    jobs: List[Dict[str, Any]] = []
    for mp in manifests:
        m = json.loads(mp.read_text())
        for r in m["runs"]:
            jobs.append({**r, "manifest": str(mp)})
    ids = [j["run_id"] for j in jobs]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate run_id across manifests")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (E / "launch_logs").mkdir(exist_ok=True)
    (E / "logs").mkdir(exist_ok=True)
    rec_path = E / "launch_logs" / f"launch_{stamp}_{'+'.join(Path(m).stem for m in manifests)}.json"
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    state: Dict[str, Any] = {"state": "running", "host": socket.gethostname(), "pid": os.getpid(),
                             "start_time": now_iso(), "manifests": [str(m) for m in manifests],
                             "max_workers": args.max_workers, "planned_runs": len(jobs), "runs": []}
    lock = threading.Lock()

    def save() -> None:
        with lock:
            write_json_atomic(rec_path, state)

    save()

    def run_one(job: Dict[str, Any]) -> Dict[str, Any]:
        out_dir = E / job["output_dir"]
        base = {"run_id": job["run_id"], "q": job["q"], "seed": job["seed"], "manifest": job["manifest"],
                "output_dir": str(out_dir)}
        if out_dir.exists():
            return {**base, "result": "skipped_existing_dir", "returncode": None}
        log_path = E / "logs" / f"{job['run_id']}.log"
        cmd = [str(PYTHON), "-B", "-u", str(E / "run_experiment.py"), "--manifest", job["manifest"],
               "--seed", str(job["seed"])]
        begin = time.time()
        with log_path.open("x") as log:
            log.write(f"# host={socket.gethostname()} start={now_iso()} cmd={' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=str(E.parent.parent), env=env, stdout=log,
                                  stderr=subprocess.STDOUT)
        rc = proc.returncode
        res = {**base, "result": "exited", "returncode": rc, "wall_sec": time.time() - begin,
               "log": str(log_path), "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(begin)),
               "end_time": now_iso()}
        if rc is not None and rc < 0:
            res["signal"] = signal.Signals(-rc).name
        return res

    print(f"[launch] {len(jobs)} runs, max_workers={args.max_workers}, record={rec_path}", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run_one, j): j for j in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # launcher-side error; the run itself may not have started
                res = {"run_id": job["run_id"], "result": "launcher_error", "returncode": None,
                       "error": f"{type(exc).__name__}: {exc}"}
            with lock:
                state["runs"].append(res)
            save()
            print(json.dumps(res), flush=True)
    ok = all(r.get("returncode") == 0 for r in state["runs"])
    state.update(state="done" if ok else "done_with_errors", end_time=now_iso())
    save()
    print(f"[launch] finished: {state['state']}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
