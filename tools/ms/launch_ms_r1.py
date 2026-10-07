#!/usr/bin/env python3
"""Wave launcher of round MS-R1 (prompt section 2.1; modelled on ``tools/v2/launch_refine.py``).

Waves (``--wave``):
  base       the 20 ``MS_base`` runs (q x seeds 10501-10510) into ``<root>/base/q{q}/seed{s}/MS_base``; they are
             check C-MS1 (terminal-stage end state = ``parents_A``) and the ``MS_base`` arm of the pilot.
  v20_repro  C-R4: the UNCHANGED ``run/run_v2_T2_locked.py --q Q --seed S`` into
             ``<root>/v20_reproduction/q{Q}/seed{S}`` (no config is written).
  pilot      the five rule arms (``MS_rule``, ``MS_s25a0``, ``MS_s25a5``, ``MS_s35a0``, ``MS_s35a5``) x 20 runs into
             ``<root>/pilot/q{q}/seed{s}/<arm>``.

Every config is the full ``ms_run_config/1`` (every key written out); the rule / sampler parameters come from
``--params`` (the pre-registered JSON file; default: the D4 / D5 defaults, recorded as such) and the file's SHA-256
is recorded. ``--dry-run`` writes the configs and ``dryrun_<stamp>.json`` and validates every config (the
configs are written into the run directories, so do not dry-run a wave whose root is a real results root until the
wave is meant to be written); a real launch writes ``launch_<stamp>.json`` (planned and finished runs with return
codes and wall time, nproc, load average, free disk, the HEAD hash, ``git diff --stat <code-commit> HEAD``,
``git status --porcelain``, the parameter file hash) and runs the jobs single-threaded through a bounded pool (at
most 40 workers).

Examples (inside tmux):
  python tools/ms/launch_ms_r1.py --wave base --workers 20 --code-commit <sha>
  python tools/ms/launch_ms_r1.py --wave pilot --params reports/ms/r1/prereg_parameters.json --workers 40 \
      --code-commit <sha> --dry-run --root /tmp/somewhere
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))

import ms_configs as mc  # noqa: E402

PY = sys.executable
DEFAULT_ROOT = ROOT / "results" / "ms_r1"
MAX_WORKERS = 40
WAVES = ("base", "v20_repro", "pilot")
WAVE_DIR = {"base": "base", "v20_repro": "v20_reproduction", "pilot": "pilot"}
WAVE_ARMS = {"base": ("MS_base",), "v20_repro": (), "pilot": mc.PILOT_ARMS}


class Job(NamedTuple):
    """One run: its config (or the locked-entry stub) and its output directory."""

    cfg: Dict[str, Any]
    out_dir: str


def load_params(path: Optional[str]) -> Dict[str, Any]:
    """The rule / sampler parameters (default: the D4 / D5 defaults)."""
    if path is None:
        return copy.deepcopy(mc.DEFAULT_PARAMS)
    with open(path) as f:
        prm = json.load(f)
    if set(prm) != set(mc.DEFAULT_PARAMS):
        raise ValueError(f"parameter file keys {sorted(prm)} != {sorted(mc.DEFAULT_PARAMS)}")
    return prm


def build_jobs(wave: str, qs: Sequence[int], seeds: Sequence[int], arms: Optional[Sequence[str]], root: Path,
               params: Dict[str, Any]) -> List[Job]:
    """All jobs of a wave (configs are built, not written)."""
    proto = mc.load_protocol()
    allowed = WAVE_ARMS[wave]
    chosen = tuple(arms) if arms else allowed
    bad = [a for a in chosen if a not in allowed]
    if bad:
        raise ValueError(f"arms {bad} are not in wave {wave} (arms: {list(allowed)})")
    jobs: List[Job] = []
    for q in qs:
        for s in seeds:
            if wave == "v20_repro":
                out = str(root / WAVE_DIR[wave] / f"q{q}" / f"seed{s}")
                jobs.append(Job({"locked_entry_point": True, "arm": "v20_repro", "q": int(q), "seed": int(s),
                                 "run": f"v2T2locked_q{q}_s{s}"}, out))
                continue
            for arm in chosen:
                out = str(root / WAVE_DIR[wave] / f"q{q}" / f"seed{s}" / arm)
                jobs.append(Job(mc.build_config(proto, int(q), int(s), arm, out, params), out))
    return jobs


def is_locked_entry(cfg: Dict[str, Any]) -> bool:
    """True for the C-R4 stub (run through ``run/run_v2_T2_locked.py``)."""
    return bool(cfg.get("locked_entry_point"))


def job_command(cfg: Dict[str, Any], out_dir: str) -> List[str]:
    """Subprocess command of a job (never ``git``; the interpreter is the launcher's own)."""
    if is_locked_entry(cfg):
        return [PY, "-B", "-u", str(ROOT / "run" / "run_v2_T2_locked.py"), "--q", str(cfg["q"]),
                "--seed", str(cfg["seed"]), "--out-dir", out_dir]
    return [PY, "-B", "-u", str(ROOT / "run" / "run_ms_stagewise.py"), "--config",
            os.path.join(out_dir, "run_config.json"), "--out-dir", out_dir]


def validate_jobs(jobs: Sequence[Job]) -> List[Dict[str, Any]]:
    """``run.run_ms_stagewise.validate_config`` on every config; one row per job."""
    from run import run_ms_stagewise as rms

    rows = []
    for cfg, out in jobs:
        if is_locked_entry(cfg):
            continue
        row: Dict[str, Any] = {"out": out, "arm": cfg["arm"], "q": cfg["q"], "seed": cfg["seed"],
                               "ok": True, "error": None}
        try:
            rms.validate_config(cfg)
        except Exception as exc:  # noqa: BLE001 - reported
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def summarize_validation(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per arm: number of configs and how many validate."""
    arms: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        a = arms.setdefault(r["arm"], {"n": 0, "valid": 0, "errors": set()})
        a["n"] += 1
        a["valid"] += int(r["ok"])
        if r["error"]:
            a["errors"].add(r["error"])
    return {k: {**v, "errors": sorted(v["errors"])} for k, v in arms.items()}


def git_record(code_commit: Optional[str]) -> Dict[str, Any]:
    """HEAD hash, ``git diff --stat <code_commit> HEAD`` and ``git status --porcelain``."""
    def git(*args: str) -> str:
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        if r.returncode != 0:
            return f"[git {' '.join(args)} failed] {r.stderr.strip()}"
        return r.stdout.strip()
    rec: Dict[str, Any] = {"head": git("rev-parse", "HEAD"), "code_commit": code_commit,
                           "status_porcelain": git("status", "--porcelain").splitlines()}
    if code_commit:
        rec["code_commit_resolved"] = git("rev-parse", "--verify", code_commit + "^{commit}")
        rec["diff_stat_code_commit_to_head"] = git("diff", "--stat", code_commit, "HEAD")
    return rec


def host_record(path: Path) -> Dict[str, Any]:
    """nproc, load average and free disk, recorded before a launch."""
    du = shutil.disk_usage(path)
    return {"nproc": os.cpu_count(), "loadavg_at_start": list(os.getloadavg()),
            "disk_free_bytes": du.free, "disk_total_bytes": du.total}


def child_env() -> Dict[str, str]:
    """Single-threaded environment of every run."""
    return dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                PYTHONDONTWRITEBYTECODE="1")


def run_job(cmd: Sequence[str], out_dir: str, env: Dict[str, str]) -> Dict[str, Any]:
    """Run one command, stdout/stderr to ``<out_dir>/run.log`` (opened ``x``: no overwrite)."""
    t0 = time.monotonic()
    with open(os.path.join(out_dir, "run.log"), "x") as log:
        r = subprocess.run(list(cmd), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return {"out": out_dir, "returncode": r.returncode, "wall_sec": time.monotonic() - t0}


def write_record(path: Path, record: Dict[str, Any], create: bool = False) -> None:
    """Write the launch record; ``create=True`` refuses to replace an existing file."""
    with open(path, "x" if create else "w") as f:
        json.dump(record, f, indent=1)


def run_pool(jobs: Sequence[Job], workers: int, record: Dict[str, Any], record_path: Path) -> None:
    """Run all jobs through a bounded pool of subprocesses; the record is updated per finish."""
    env = child_env()

    def one(job: Job) -> Dict[str, Any]:
        try:
            return run_job(job_command(*job), job.out_dir, env)
        except Exception as exc:  # noqa: BLE001 - recorded as a failed run
            return {"out": job.out_dir, "returncode": None, "error": f"{type(exc).__name__}: {exc}"}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(one, j) for j in jobs]):
            res = fut.result()
            record["runs"].append(res)
            write_record(record_path, record)
            print(json.dumps(res), flush=True)
    record["state"] = "done" if all(r.get("returncode") == 0 for r in record["runs"]) else "failed"
    record["loadavg_at_end"] = list(os.getloadavg())
    write_record(record_path, record)


def check_no_previous_run(jobs: Sequence[Job]) -> List[str]:
    """Out dirs that already hold a run (status.json) or a log (run.log)."""
    return [j.out_dir for j in jobs if os.path.exists(os.path.join(j.out_dir, "status.json"))
            or os.path.exists(os.path.join(j.out_dir, "run.log"))]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    p = argparse.ArgumentParser(description="MS-R1 wave launcher", allow_abbrev=False)
    p.add_argument("--wave", choices=WAVES, required=True)
    p.add_argument("--qs", type=int, nargs="+", default=list(mc.DEFAULT_QS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(mc.DEFAULT_SEEDS))
    p.add_argument("--arms", nargs="+", default=None, help="default: all arms of the wave")
    p.add_argument("--params", default=None, help="pre-registered rule / sampler parameters (JSON); default D4/D5")
    p.add_argument("--workers", type=int, default=20, help=f"at most {MAX_WORKERS}")
    p.add_argument("--root", default=str(DEFAULT_ROOT), help="results root (default results/ms_r1)")
    p.add_argument("--code-commit", default=None, help="SHA of the code commit (recorded)")
    p.add_argument("--dry-run", action="store_true", help="write configs + dryrun record, validate")
    a = p.parse_args(argv)
    if not 1 <= a.workers <= MAX_WORKERS:
        p.error(f"--workers must be in 1..{MAX_WORKERS}; got {a.workers}")
    if a.wave == "v20_repro" and a.arms:
        p.error("wave v20_repro has no arms")
    if any(sd not in mc.DEFAULT_SEEDS for sd in a.seeds):   # D1: nothing outside 10501-10510
        p.error(f"seeds must be development seeds {mc.DEFAULT_SEEDS[0]}-{mc.DEFAULT_SEEDS[-1]}; got {a.seeds}")
    root = Path(a.root).resolve()
    try:
        params = load_params(a.params)
        jobs = build_jobs(a.wave, a.qs, a.seeds, a.arms, root, params)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    busy = check_no_previous_run(jobs)
    if busy:
        p.error(f"{len(busy)} out dir(s) already hold a run (status.json or run.log), e.g. {busy[0]}")
    validation: Dict[str, Any] = {}
    if a.wave != "v20_repro":
        rows = validate_jobs(jobs)
        validation = {"per_arm": summarize_validation(rows), "n_invalid": sum(1 for r in rows if not r["ok"])}
        if not a.dry_run and validation["n_invalid"]:
            print(json.dumps(validation, indent=1))
            p.error("validate_config fails for the configs above; nothing was launched")
    wave_dir = root / WAVE_DIR[a.wave]
    wave_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    record_path = wave_dir / (f"dryrun_{stamp}.json" if a.dry_run else f"launch_{stamp}.json")
    record: Dict[str, Any] = {
        "wave": a.wave, "dry_run": bool(a.dry_run), "argv": list(sys.argv if argv is None else argv),
        "started": stamp, "workers": a.workers, **host_record(wave_dir), **git_record(a.code_commit),
        "params_file": a.params,
        "params_sha256": mc.sha256_file(Path(a.params)) if a.params else "defaults (D4/D5)",
        "params": params, "n_planned": len(jobs), "planned": [j.out_dir for j in jobs],
        "validation": validation, "runs": [], "state": "dry_run" if a.dry_run else "running"}
    for cfg, out in jobs:
        os.makedirs(out, exist_ok=True)
        if not is_locked_entry(cfg):
            with open(os.path.join(out, "run_config.json"), "w") as f:
                json.dump(cfg, f, indent=1)
    write_record(record_path, record, create=True)
    print(f"{len(jobs)} run(s) of wave {a.wave}; record {record_path}", flush=True)
    if validation:
        print(json.dumps(validation["per_arm"], indent=1), flush=True)
    if a.dry_run:
        return 0
    run_pool(jobs, a.workers, record, record_path)
    return 0 if record["state"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
