#!/usr/bin/env python3
"""Wave launcher of round MS-R4 (prompt section 2.1; reuses the job, pool and record functions of ``launch_ms_r1.py``).

Waves (``--wave``):
  r4        the seven arms (E1: ``t1_st_s16``, ``relu_st_s16``, ``lrsm_st_s16``; E2: ``fb_a0``, ``fb_r``, ``fb_rg``,
            ``fb_rx``) x q in {50, 60} x the new development seeds 10511-10530 = 280 runs into
            ``<root>/pilot/q{q}/seed{s}/<arm>``.
  r4_cms6   check C-MS6: the two controls ``t1_st_s16`` and ``relu_st_s16`` x q x seeds 10501-10510 = 40 runs into
            ``<root>/cms6/q{q}/seed{s}/<arm>``, to be compared with MS-R3's runs of the same names (their configs differ
            from MS-R3's only in the round tag, the run name and the output directory).
C-R7 (the unchanged ``run/run_v2_T2_locked.py`` against ``rehearsal_v2_0``) is the existing wave ``v20_repro`` of
``tools/ms/launch_ms_r1.py`` with ``--root results/ms_r4``.

Seed guards: wave ``r4`` accepts only seeds of 10511-10530 and wave ``r4_cms6`` only seeds of 10501-10510; no wave accepts
any seed of 30501-30520 (the v2.0 confirmation block) or 40501-40520 (reserved for the confirmation after a lock).

Every config is the full ``ms_run_config/1``; the rule / sampler parameters come from ``--params`` (the pre-registered
``reports/ms/r1/prereg_parameters.json``, SHA-256 ``0fbfc018...``; a real launch checks the hash). ``--plan-csv`` writes the
SHA-256 of every planned ``run_config.json`` and exits without creating any run directory. ``--dry-run`` writes the configs
and ``dryrun_<stamp>.json`` under ``--root`` and validates them (do not dry-run into a real results root). A real launch
writes ``launch_<stamp>.json`` (planned and finished runs with return codes and wall time, nproc, load average, free disk, the
HEAD hash, ``git diff --stat <code-commit> HEAD`` (whole tree and the run code paths), ``git status --porcelain``, the
parameter file hash) and runs the jobs single-threaded through a bounded pool of at most 40 workers.

Examples (inside tmux):
  python tools/ms/r4_launch.py --wave r4_cms6 --params reports/ms/r1/prereg_parameters.json --workers 40 --code-commit <sha>
  python tools/ms/r4_launch.py --wave r4 --params reports/ms/r1/prereg_parameters.json --workers 40 --code-commit <sha>
  python tools/ms/r4_launch.py --wave r4 --params reports/ms/r1/prereg_parameters.json --plan-csv plan.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))

import launch_ms_r1 as L1  # noqa: E402
import ms_configs as mc  # noqa: E402
import r4_configs as rc  # noqa: E402

DEFAULT_ROOT = ROOT / "results" / "ms_r4"
MAX_WORKERS = L1.MAX_WORKERS
WAVES = ("r4", "r4_cms6")
WAVE_DIR = {"r4": "pilot", "r4_cms6": "cms6"}
WAVE_ARMS = {"r4": rc.ARMS, "r4_cms6": rc.CMS6_ARMS}
WAVE_SEEDS = {"r4": rc.SEEDS, "r4_cms6": rc.SEEDS_CMS6}
PARAMS_SHA256 = "0fbfc01857c5abe339ab1926361cd57d6e988da20e754fd6dea3ffa4e076b406"


def check_seeds(wave: str, seeds: Sequence[int]) -> None:
    """Refuse a seed outside the wave's block, and every seed of the confirmation / reserved blocks.

    Raises:
        ValueError: A seed of 30501-30520 or 40501-40520, or outside the block of the wave.
    """
    bad_forbidden = [s for s in seeds if s in rc.FORBIDDEN_SEEDS]
    if bad_forbidden:
        raise ValueError(f"seeds {bad_forbidden} belong to the v2.0 confirmation block (30501-30520) or to the block reserved "
                         f"for the confirmation after a lock (40501-40520): no wave of MS-R4 accepts them")
    allowed = WAVE_SEEDS[wave]
    bad = [s for s in seeds if s not in allowed]
    if bad:
        raise ValueError(f"wave {wave} accepts the seeds {allowed[0]}-{allowed[-1]} only; got {bad}")


def build_jobs(wave: str, qs: Sequence[int], seeds: Sequence[int], arms: Optional[Sequence[str]], root: Path,
               params: Dict[str, Any]) -> List[L1.Job]:
    """All jobs of a wave (configs are built, not written).

    Raises:
        ValueError: A seed that the wave refuses, an arm that is not in the wave, a q outside {50, 60}.
    """
    check_seeds(wave, seeds)
    bad_q = [q for q in qs if q not in rc.QS]
    if bad_q:
        raise ValueError(f"q must be in {rc.QS}; got {bad_q}")
    allowed = WAVE_ARMS[wave]
    chosen = tuple(arms) if arms else allowed
    bad = [a for a in chosen if a not in allowed]
    if bad:
        raise ValueError(f"arms {bad} are not in wave {wave} (arms: {list(allowed)})")
    proto = mc.load_protocol()
    jobs: List[L1.Job] = []
    for q in qs:
        for s in seeds:
            for arm in chosen:
                out = str(root / WAVE_DIR[wave] / f"q{q}" / f"seed{s}" / arm)
                jobs.append(L1.Job(rc.build_config_r4(proto, int(q), int(s), arm, out, params), out))
    return jobs


def plan_rows(wave: str, jobs: Sequence[L1.Job]) -> List[Dict[str, Any]]:
    """One row per planned run: wave, arm, q, seed, run name, output directory and the SHA-256 of its ``run_config.json``."""
    return [{"wave": wave, "arm": j.cfg["arm"], "q": j.cfg["q"], "seed": j.cfg["seed"], "run": j.cfg["run"],
             "out_dir": j.out_dir, "run_config_sha256": rc.config_sha256(j.cfg)} for j in jobs]


def write_plan_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """The plan as a CSV (refuses to replace an existing file)."""
    with open(path, "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["wave", "arm", "q", "seed", "run", "out_dir", "run_config_sha256"])
        w.writeheader()
        w.writerows(rows)


def load_params(path: Optional[str], real_launch: bool) -> Dict[str, Any]:
    """The pre-registered parameters; a real launch demands the file and its SHA-256.

    Raises:
        ValueError: No file for a real launch, or a hash that differs from :data:`PARAMS_SHA256`.
    """
    if path is None:
        if real_launch:
            raise ValueError("a real launch needs --params reports/ms/r1/prereg_parameters.json (passed explicitly)")
        return L1.load_params(None)
    sha = mc.sha256_file(Path(path))
    if real_launch and sha != PARAMS_SHA256:
        raise ValueError(f"{path} has SHA-256 {sha}, expected {PARAMS_SHA256}")
    return L1.load_params(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    p = argparse.ArgumentParser(description="MS-R4 wave launcher", allow_abbrev=False)
    p.add_argument("--wave", choices=WAVES, required=True)
    p.add_argument("--qs", type=int, nargs="+", default=list(rc.QS))
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="default: the whole block of the wave")
    p.add_argument("--arms", nargs="+", default=None, help="default: all arms of the wave")
    p.add_argument("--params", default=None, help="the pre-registered rule / sampler parameters (JSON)")
    p.add_argument("--workers", type=int, default=20, help=f"at most {MAX_WORKERS}")
    p.add_argument("--root", default=None, help="results root (default results/ms_r4)")
    p.add_argument("--code-commit", default=None, help="SHA of the code commit (recorded)")
    p.add_argument("--dry-run", action="store_true", help="write configs + dryrun record under --root, validate")
    p.add_argument("--plan-csv", default=None, help="write the run plan with the config hashes and exit (nothing is created)")
    a = p.parse_args(argv)
    if not 1 <= a.workers <= MAX_WORKERS:
        p.error(f"--workers must be in 1..{MAX_WORKERS}; got {a.workers}")
    seeds = list(a.seeds) if a.seeds is not None else list(WAVE_SEEDS[a.wave])
    root = Path(a.root).resolve() if a.root else DEFAULT_ROOT.resolve()
    real = not a.dry_run and a.plan_csv is None
    try:
        params = load_params(a.params, real)
        jobs = build_jobs(a.wave, a.qs, seeds, a.arms, root, params)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    if a.plan_csv:
        write_plan_csv(Path(a.plan_csv), plan_rows(a.wave, jobs))
        print(f"{len(jobs)} planned run(s) of wave {a.wave}; plan {a.plan_csv}", flush=True)
        return 0
    busy = L1.check_no_previous_run(jobs)
    if busy:
        p.error(f"{len(busy)} out dir(s) already hold a run (status.json or run.log), e.g. {busy[0]}")
    rows = L1.validate_jobs(jobs)
    validation = {"per_arm": L1.summarize_validation(rows), "n_invalid": sum(1 for r in rows if not r["ok"])}
    if not a.dry_run and validation["n_invalid"]:
        print(json.dumps(validation, indent=1))
        p.error("validate_config fails for the configs above; nothing was launched")
    wave_dir = root / WAVE_DIR[a.wave]
    wave_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    record_path = wave_dir / (f"dryrun_{stamp}.json" if a.dry_run else f"launch_{stamp}.json")
    record: Dict[str, Any] = {
        "tool": "tools/ms/r4_launch.py", "wave": a.wave, "dry_run": bool(a.dry_run),
        "argv": list(sys.argv if argv is None else argv), "started": stamp, "workers": a.workers,
        **L1.host_record(wave_dir), **L1.git_record(a.code_commit), "params_file": a.params,
        "params_sha256": mc.sha256_file(Path(a.params)) if a.params else "defaults (D4/D5)", "params": params,
        "n_planned": len(jobs), "planned": [j.out_dir for j in jobs], "validation": validation, "runs": [],
        "state": "dry_run" if a.dry_run else "running"}
    for cfg, out in jobs:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "run_config.json"), "w") as f:
            json.dump(cfg, f, indent=1)
    L1.write_record(record_path, record, create=True)
    print(f"{len(jobs)} run(s) of wave {a.wave}; record {record_path}", flush=True)
    print(json.dumps(validation["per_arm"], indent=1), flush=True)
    if a.dry_run:
        return 0
    L1.run_pool(jobs, a.workers, record, record_path)
    return 0 if record["state"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
