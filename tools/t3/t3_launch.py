#!/usr/bin/env python3
"""Wave launcher of round T3-R1 (modelled on ``tools/ms/r4_launch.py``; reuses the job, record and host
functions of ``tools/ms/launch_ms_r1.py`` by import, unchanged).

Waves (``--wave``):
  smoke   seed 50500 x q in {50, 60} x the four arms (``bw``, ``jt``, ``jt_es``, ``jt2``) at full budget = 8 runs into
          ``<root>/smoke/q{q}/{arm}`` (cost and correctness only; never analysed).
  t3      seeds 50501-50520 x q x ``bw``, ``jt``, ``jt_es`` = 120 runs into ``<root>/runs/q{q}/seed{s}/{arm}``.
  t2j     seeds 30501-30520 x q x ``jt2`` = 40 runs into ``<root>/runs_t2/q{q}/seed{s}/jt2`` (the v2.0 confirmation
          seeds, reused on purpose: the new arm pairs with the locked solver's recorded runs; it confirms nothing new).

Several waves in one call (``--wave t3 t2j``) are one launch: one pool, one record. Seed guards: ``t3`` accepts exactly 50501-50520, ``t2j`` exactly 30501-30520 and ``smoke`` exactly 50500; no wave
accepts 40501-40520 or 10501-10530. Run names are ``t3r1_q{q}_s{seed}_{arm}``.

``bw`` runs ``run/run_ms_stagewise.py --config ... --out-dir ...`` (config ``ms_run_config/1``, parameters from
``reports/t3/r1/params_bw_t3.json``); the joint arms run ``run/run_joint.py --config ... --out-dir ...``. Every config
is validated by its runner's ``validate_config`` and checked against the production values of D5
(``t3_configs.production_problems``). ``--plan-csv`` writes the run name, the output directory and the SHA-256 of
every planned ``run_config.json`` and exits without creating anything. ``--dry-run`` writes the configs and
``dryrun_<stamp>.json`` under ``--root`` (do not dry-run into a real results root). A real launch writes
``launch_<stamp>.json`` (planned and finished runs with return codes and wall time, nproc, load average, free disk, HEAD,
``git diff --stat <code-commit> HEAD`` (whole tree and the run code paths), ``git status --porcelain``, the parameter
file hash) and runs the jobs single-threaded through a bounded pool of at most 40 workers.

Examples (inside tmux):
  python tools/t3/t3_launch.py --wave smoke --workers 8 --code-commit <sha>
  python tools/t3/t3_launch.py --wave t3 t2j --workers 40 --code-commit <sha>
  python tools/t3/t3_launch.py --wave t3 --plan-csv plan_t3.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import launch_ms_r1 as L1  # noqa: E402
import t3_configs as tc  # noqa: E402
from run import run_joint as rj  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402

PY = sys.executable
MAX_WORKERS = L1.MAX_WORKERS
PLAN_FIELDS = ["wave", "arm", "q", "seed", "run", "out_dir", "run_config_sha256", "run_config_sha256_relpath"]


def check_params_file(path: Path) -> None:
    """Refuse a parameter file whose SHA-256 differs from the pre-registered one.

    Raises:
        ValueError: On a hash mismatch.
    """
    sha = tc.sha256_file(path)
    if sha != tc.PARAMS_SHA256:
        raise ValueError(f"{path} has SHA-256 {sha}, expected {tc.PARAMS_SHA256}")


def build_jobs(wave: str, qs: Sequence[int], seeds: Sequence[int], arms: Optional[Sequence[str]], root: Path,
               params: Dict[str, Any]) -> List[L1.Job]:
    """All jobs of a wave (configs are built, not written).

    Raises:
        ValueError: A seed that the wave refuses, an arm that is not in the wave, a q outside {50, 60}.
    """
    tc.check_seeds(wave, seeds)
    bad_q = [q for q in qs if q not in tc.QS]
    if bad_q:
        raise ValueError(f"q must be in {tc.QS}; got {bad_q}")
    allowed = tc.WAVE_ARMS[wave]
    chosen = tuple(arms) if arms else allowed
    bad = [a for a in chosen if a not in allowed]
    if bad:
        raise ValueError(f"arms {bad} are not in wave {wave} (arms: {list(allowed)})")
    proto = tc.load_protocol()
    jobs: List[L1.Job] = []
    for q in qs:
        for s in seeds:
            for arm in chosen:
                out = str(tc.run_dir(root, wave, int(q), int(s), arm))
                jobs.append(L1.Job(tc.build_config(proto, int(q), int(s), arm, out, params), out))
    return jobs


def job_command(cfg: Dict[str, Any], out_dir: str) -> List[str]:
    """Subprocess command of a job (never ``git``; the interpreter is the launcher's own)."""
    runner = "run_ms_stagewise.py" if cfg["arm"] == "bw" else "run_joint.py"
    return [PY, "-B", "-u", str(ROOT / "run" / runner), "--config", os.path.join(out_dir, "run_config.json"),
            "--out-dir", out_dir]


def validate_jobs(jobs: Sequence[L1.Job]) -> List[Dict[str, Any]]:
    """The runner's ``validate_config`` and the production check on every config; one row per job."""
    rows = []
    for cfg, out in jobs:
        row: Dict[str, Any] = {"out": out, "arm": cfg["arm"], "q": cfg["q"], "seed": cfg["seed"],
                               "ok": True, "error": None}
        try:
            (rms if cfg["arm"] == "bw" else rj).validate_config(cfg)
            problems = tc.production_problems(cfg)
            if problems:
                raise ValueError("; ".join(problems))
        except Exception as exc:  # noqa: BLE001 - reported
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def relpath_sha256(cfg: Dict[str, Any], root: Path) -> str:
    """SHA-256 of the config text with ``record.output_dir`` made relative to the results root's parent (so the
    hash does not depend on the worktree that launches)."""
    c = json.loads(json.dumps(cfg))
    out = Path(c["record"]["output_dir"])
    try:
        c["record"]["output_dir"] = str(out.relative_to(Path(root).parent))
    except ValueError:
        pass
    return hashlib.sha256(tc.config_text(c).encode()).hexdigest()


def plan_rows(wave: str, jobs: Sequence[L1.Job], root: Path) -> List[Dict[str, Any]]:
    """One row per planned run: wave, arm, q, seed, run name, output directory and the SHA-256 of its
    ``run_config.json`` (as written, and with the output directory made relative)."""
    return [{"wave": wave, "arm": j.cfg["arm"], "q": j.cfg["q"], "seed": j.cfg["seed"], "run": j.cfg["run"],
             "out_dir": j.out_dir, "run_config_sha256": tc.config_sha256(j.cfg),
             "run_config_sha256_relpath": relpath_sha256(j.cfg, root)} for j in jobs]


def write_plan_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """The plan as a CSV (refuses to replace an existing file)."""
    with open(path, "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PLAN_FIELDS)
        w.writeheader()
        w.writerows(rows)


def run_pool(jobs: Sequence[L1.Job], workers: int, record: Dict[str, Any], record_path: Path) -> None:
    """Run all jobs through a bounded pool of subprocesses; the record is updated per finish."""
    env = L1.child_env()

    def one(job: L1.Job) -> Dict[str, Any]:
        try:
            return L1.run_job(job_command(*job), job.out_dir, env)
        except Exception as exc:  # noqa: BLE001 - recorded as a failed run
            return {"out": job.out_dir, "returncode": None, "error": f"{type(exc).__name__}: {exc}"}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(one, j) for j in jobs]):
            res = fut.result()
            record["runs"].append(res)
            L1.write_record(record_path, record)
            print(json.dumps(res), flush=True)
    record["state"] = "done" if all(r.get("returncode") == 0 for r in record["runs"]) else "failed"
    record["loadavg_at_end"] = list(os.getloadavg())
    L1.write_record(record_path, record)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    p = argparse.ArgumentParser(description="T3-R1 wave launcher", allow_abbrev=False)
    p.add_argument("--wave", choices=tc.WAVES, nargs="+", required=True,
                   help="one or more waves; several waves are one launch (one pool, one record)")
    p.add_argument("--qs", type=int, nargs="+", default=list(tc.QS))
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="default: the whole block of each wave (explicit seeds need a single wave)")
    p.add_argument("--arms", nargs="+", default=None, help="default: all arms of the wave (single wave only)")
    p.add_argument("--params", default=str(tc.PARAMS_PATH), help="the pre-registered bw parameters (JSON)")
    p.add_argument("--workers", type=int, default=20, help=f"at most {MAX_WORKERS}")
    p.add_argument("--root", default=None, help="results root (default results/t3_r1)")
    p.add_argument("--code-commit", default=None, help="SHA of the code commit (recorded)")
    p.add_argument("--dry-run", action="store_true", help="write configs + dryrun record under --root, validate")
    p.add_argument("--plan-csv", default=None, help="write the run plan with the config hashes and exit")
    a = p.parse_args(argv)
    if not 1 <= a.workers <= MAX_WORKERS:
        p.error(f"--workers must be in 1..{MAX_WORKERS}; got {a.workers}")
    waves = list(dict.fromkeys(a.wave))
    if (a.seeds is not None or a.arms) and len(waves) > 1:
        p.error("explicit --seeds and --arms need a single wave")
    root = Path(a.root).resolve() if a.root else tc.DEFAULT_ROOT.resolve()
    jobs: List[L1.Job] = []
    plan: List[Dict[str, Any]] = []
    try:
        check_params_file(Path(a.params))
        params = tc.load_bw_params(Path(a.params))
        for wave in waves:
            seeds = list(a.seeds) if a.seeds is not None else list(tc.WAVE_SEEDS[wave])
            wj = build_jobs(wave, a.qs, seeds, a.arms, root, params)
            jobs += wj
            plan += plan_rows(wave, wj, root)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    wave_name = "+".join(waves)
    if a.plan_csv:
        write_plan_csv(Path(a.plan_csv), plan)
        print(f"{len(jobs)} planned run(s) of wave {wave_name}; plan {a.plan_csv}", flush=True)
        return 0
    busy = L1.check_no_previous_run(jobs)
    if busy:
        p.error(f"{len(busy)} out dir(s) already hold a run (status.json or run.log), e.g. {busy[0]}")
    rows = validate_jobs(jobs)
    validation = {"per_arm": L1.summarize_validation(rows), "n_invalid": sum(1 for r in rows if not r["ok"])}
    if not a.dry_run and validation["n_invalid"]:
        print(json.dumps(validation, indent=1))
        p.error("validation fails for the configs above; nothing was launched")
    wave_dir = root / tc.WAVE_DIR[waves[0]] if len(waves) == 1 else root
    wave_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "" if len(waves) == 1 else "_" + "_".join(waves)
    record_path = wave_dir / (f"dryrun{tag}_{stamp}.json" if a.dry_run else f"launch{tag}_{stamp}.json")
    record: Dict[str, Any] = {
        "tool": "tools/t3/t3_launch.py", "wave": wave_name, "dry_run": bool(a.dry_run),
        "argv": list(sys.argv if argv is None else argv), "started": stamp, "workers": a.workers,
        **L1.host_record(wave_dir), **L1.git_record(a.code_commit), "params_file": a.params,
        "params_sha256": tc.sha256_file(Path(a.params)), "params": params,
        "n_planned": len(jobs), "planned": [j.out_dir for j in jobs], "validation": validation, "runs": [],
        "state": "dry_run" if a.dry_run else "running"}
    for cfg, out in jobs:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "run_config.json"), "w") as f:
            json.dump(cfg, f, indent=1)
    L1.write_record(record_path, record, create=True)
    print(f"{len(jobs)} run(s) of wave {wave_name}; record {record_path}", flush=True)
    print(json.dumps(validation["per_arm"], indent=1), flush=True)
    if a.dry_run:
        return 0
    run_pool(jobs, a.workers, record, record_path)
    return 0 if record["state"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
