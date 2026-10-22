#!/usr/bin/env python3
"""Launcher of the T3-R3 development wave (prompt D5, G1 reply ``reports/t3/r3/pi_record/28b_g1_reply.md``).

Arms (all with ``cfg["arm"] == "bw"``, so every existing reader of a ``bw`` run applies; the arm is the directory name and the keys):

  ``bw``    the locked T3-1.0 settings (control; not eligible);
  ``bwA``   ``bw`` plus ``stage3_retry = {"delta_gt": 0.005, "max_retries": 1}`` (candidate A);
  ``bwAB``  ``bwA`` plus ``terminal_clamp_likelihood = "censored"`` (candidate B).

Runs: seeds 70501-70520 at q in {50, 60}, all three arms (120 runs) into ``<root>/dev/q{q}/seed{s}/{arm}``, plus the diagnostic runs
(q = 60, seed 60518) of ``bwA`` and ``bwAB`` into ``<root>/diagnostic/q60/seed60518/{arm}`` (2 runs, never counted). Run names are
``t3r3_q{q}_s{seed}_{arm}``. Any other seed is refused.

A config is ``tools/t3/t3_configs.build_bw_config`` with ``seed``, ``run``, ``record.run``, ``record.seed``, ``record.output_dir``
and ``pilot`` = "t3_r3" set (the D4 fields of the lock's launcher) and, for a candidate arm, the scheme's own keys. It is checked
against the lock's ``reference_config_sha256`` with ``tools/t3/t3c_launch.normalise`` after the scheme's keys are removed. The lock
(``protocols/LOCK`` record, the protocol files and the tool files of the lock) is verified with ``t3c_launch.check_lock``.

``--plan-csv`` writes, for every run, the run name, the output directory, the config SHA-256 (as written and with the output
directory made relative to the root), the planned init digest of attempt 1 and, for the candidate arms, of attempt 2, then exits.

A real launch needs a clean working tree outside ``results/``, writes the configs and an empty infrastructure ledger, runs the pool
(at most 40 single-threaded workers), and handles infrastructure failures as D6 of the prompt says: a negative return code, an
out-of-memory kill or ENOSPC is moved to ``<root>/crashed/`` and the run repeated once; a second one is recorded as stopped.
Any other nonzero exit is a failed run, never repeated. A re-initialisation inside a run is part of the run.

Examples (inside tmux):
  python tools/t3/t3r3_launch.py --plan-csv reports/t3/r3/dev_plan.csv
  python tools/t3/t3r3_launch.py --workers 40 --code-commit <sha>
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")           # the plan constructs the runs in this process; a run refuses another thread environment

import argparse
import copy
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import launch_ms_r1 as L1  # noqa: E402
import t3_configs as tc  # noqa: E402
import t3c_launch as cl  # noqa: E402
from run import ms_retry  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402
from t3_launch import check_params_file, job_command, relpath_sha256, run_pool  # noqa: E402

WAVE = "t3r3"
ROUND = "t3_r3"
ARMS = ("bw", "bwA", "bwAB")
CANDIDATES = ("bwA", "bwAB")
QS = (50, 60)
SEEDS = tuple(range(70501, 70520 + 1))
DIAG_Q, DIAG_SEED = 60, 60518
RETRY = {"delta_gt": 0.005, "max_retries": 1}
SCHEME_KEYS = ("stage3_retry", "terminal_clamp_likelihood")
DEFAULT_ROOT = ROOT / "results" / "t3_r3"
MAX_WORKERS = L1.MAX_WORKERS
PLAN_FIELDS = ["wave", "kind", "arm", "q", "seed", "run", "out_dir", "run_config_sha256", "run_config_sha256_relpath",
               "init_state_sha256", "attempt2_init_state_sha256"]
LEDGER_SCHEMA = "t3r3_infra_events/1"


# ---------------------------------------------------------------------------------- names and configs
def run_name(arm: str, q: int, seed: int) -> str:
    """``t3r3_q{q}_s{seed}_{arm}``."""
    return f"t3r3_q{q}_s{seed}_{arm}"


def kind_of(q: int, seed: int) -> str:
    """``dev`` (the development block) or ``diagnostic`` ((60, 60518))."""
    if q in QS and seed in SEEDS:
        return "dev"
    if (q, seed) == (DIAG_Q, DIAG_SEED):
        return "diagnostic"
    raise ValueError(f"(q, seed) = ({q}, {seed}) is outside the development block {SEEDS[0]}-{SEEDS[-1]} at q in {QS} "
                     f"and is not the diagnostic run ({DIAG_Q}, {DIAG_SEED})")


def run_dir(root: Path, arm: str, q: int, seed: int) -> Path:
    """``<root>/dev/q{q}/seed{s}/{arm}`` or ``<root>/diagnostic/q60/seed60518/{arm}``."""
    return Path(root) / kind_of(q, seed) / f"q{q}" / f"seed{seed}" / arm


def scheme_keys(arm: str) -> Dict[str, Any]:
    """The scheme's own config keys of an arm."""
    if arm == "bw":
        return {}
    keys: Dict[str, Any] = {"stage3_retry": dict(RETRY)}
    if arm == "bwAB":
        keys["terminal_clamp_likelihood"] = "censored"
    elif arm != "bwA":
        raise ValueError(f"unknown arm {arm!r}")
    return keys


def build_arm_config(proto: Dict[str, Any], params: Dict[str, Any], arm: str, q: int, seed: int, out_dir: str,
                     protocol_path: Path = tc.PROTOCOL_PATH) -> Dict[str, Any]:
    """The control config of (q, seed) with the D5 fields set (``pilot`` = "t3_r3", run name ``t3r3_q{q}_s{seed}_{arm}``), plus the
    arm's scheme keys.

    Raises:
        ValueError: A seed outside the development block that is not the diagnostic run.
    """
    kind_of(int(q), int(seed))
    cfg = tc.build_bw_config(proto, int(q), int(seed), out_dir, params, protocol_path)
    name = run_name(arm, int(q), int(seed))
    cfg["pilot"] = ROUND
    cfg["run"] = name
    cfg["record"]["run"] = name
    cfg.update(scheme_keys(arm))
    return cfg


def strip_scheme(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The config without the scheme's own keys (the control form)."""
    c = copy.deepcopy(cfg)
    for k in SCHEME_KEYS:
        c.pop(k, None)
    return c


def config_problems(cfg: Dict[str, Any], lock: Dict[str, Any], arm: str) -> List[str]:
    """Departures of a planned config from the lock and from the arm's definition (empty list: none)."""
    out: List[str] = []
    q, seed = cfg.get("q"), cfg.get("seed")
    try:
        kind_of(q, seed)
    except ValueError as exc:
        return [str(exc)]
    if cfg.get("arm") != "bw":
        out.append(f"arm field {cfg.get('arm')!r} != 'bw'")
    if cfg.get("pilot") != ROUND:
        out.append(f"pilot {cfg.get('pilot')!r} != {ROUND!r}")
    if cfg.get("run") != run_name(arm, q, seed) or cfg["record"].get("run") != cfg.get("run") or cfg["record"].get("seed") != seed:
        out.append(f"run name / record fields differ from {run_name(arm, q, seed)!r}")
    have = {k: cfg[k] for k in SCHEME_KEYS if k in cfg}
    if have != scheme_keys(arm):
        out.append(f"scheme keys {have} != {scheme_keys(arm)} of arm {arm}")
    norm = cl.normalise(strip_scheme(cfg))
    got = tc.config_sha256(norm)
    want = lock["reference_config_sha256"][str(q)]
    if got != want:
        out.append(f"the config differs from the lock outside the D5 fields and the scheme's keys "
                   f"(normalised SHA-256 {got} != {want})")
    out += tc.production_problems(norm)
    return out


# ---------------------------------------------------------------------------------- jobs and plan
def job_list() -> List[Dict[str, Any]]:
    """The 122 planned runs as ``{arm, q, seed}``: the diagnostic runs first, then ``bwAB``, ``bwA``, ``bw`` (the arms that may
    retrain start first), q-major, seed order."""
    rows = [{"arm": a, "q": DIAG_Q, "seed": DIAG_SEED} for a in CANDIDATES]
    for arm in ("bwAB", "bwA", "bw"):
        for q in QS:
            for s in SEEDS:
                rows.append({"arm": arm, "q": q, "seed": s})
    return rows


def build_jobs(root: Path, params: Dict[str, Any]) -> List[L1.Job]:
    """The 122 jobs (configs are built, not written)."""
    proto = tc.load_protocol()
    jobs = []
    for r in job_list():
        out = str(run_dir(root, r["arm"], r["q"], r["seed"]))
        jobs.append(L1.Job(build_arm_config(proto, params, r["arm"], r["q"], r["seed"], out), out))
    return jobs


def arm_of_dir(out_dir: str) -> str:
    """The arm of a run directory (its last component)."""
    return Path(out_dir).name


def init_digests(cfg: Dict[str, Any], arm: str) -> Dict[str, str]:
    """The planned init digests: attempt 1 (the run is constructed, no training, no file) and, for the candidates, attempt 2."""
    run = rms.MSRun(copy.deepcopy(cfg), str(Path(cfg["record"]["output_dir"]) / "_not_written"))
    d1 = run.init_state_sha256
    d2 = ms_retry.attempt2_init_digest(run) if arm in CANDIDATES else ""
    return {"init_state_sha256": d1, "attempt2_init_state_sha256": d2}


def plan_rows(jobs: Sequence[L1.Job], root: Path, with_digests: bool = True) -> List[Dict[str, Any]]:
    """One row per planned run (see the module docstring)."""
    rows = []
    for cfg, out in jobs:
        arm = arm_of_dir(out)
        row = {"wave": WAVE, "kind": kind_of(cfg["q"], cfg["seed"]), "arm": arm, "q": cfg["q"], "seed": cfg["seed"], "run": cfg["run"],
               "out_dir": out, "run_config_sha256": tc.config_sha256(cfg), "run_config_sha256_relpath": relpath_sha256(cfg, root)}
        row.update(init_digests(cfg, arm) if with_digests else {"init_state_sha256": "", "attempt2_init_state_sha256": ""})
        rows.append(row)
    return rows


def write_plan(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """The plan CSV (refuses to replace an existing file)."""
    with open(path, "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PLAN_FIELDS)
        w.writeheader()
        w.writerows(rows)


def validate_jobs(jobs: Sequence[L1.Job], lock: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The runner's ``validate_config`` and :func:`config_problems` on every config; one row per job."""
    rows = []
    for cfg, out in jobs:
        row: Dict[str, Any] = {"out": out, "arm": arm_of_dir(out), "q": cfg["q"], "seed": cfg["seed"], "ok": True, "error": None}
        try:
            rms.validate_config(cfg)
            problems = config_problems(cfg, lock, arm_of_dir(out))
            if problems:
                raise ValueError("; ".join(problems))
        except Exception as exc:  # noqa: BLE001 - reported
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def tree_problems(repo: Path = ROOT) -> List[str]:
    """A working tree that is not clean outside ``results/`` (the runner's own definition of ``clean_tree``)."""
    dirty = cl._git("status", "--porcelain", "--", ".", ":(exclude)results", repo=repo).splitlines()
    return [f"working tree is not clean ({len(dirty)} entries outside results/, e.g. {dirty[0]!r})"] if dirty else []


# ---------------------------------------------------------------------------------- infrastructure failures
def handle_infra_failures(jobs: Sequence[L1.Job], results: Sequence[Dict[str, Any]], root: Path,
                          runner: Callable[[L1.Job], Dict[str, Any]], ledger: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Move every infrastructure-failed attempt to ``<root>/crashed/`` and repeat the run once into the original path (the
    classification is ``t3c_launch.classify_attempt``). A second infrastructure failure of a run is recorded in
    ``stopped_without_verdict``. Returns the results of the repeats."""
    by_out = {j.out_dir: j for j in jobs}
    repeats: List[Dict[str, Any]] = []
    for res in results:
        job = by_out.get(res["out"])
        if job is None:
            continue
        cls = cl.classify_attempt(res, Path(job.out_dir))
        if cls is None:
            continue
        arm, q, s = arm_of_dir(job.out_dir), int(job.cfg["q"]), int(job.cfg["seed"])
        dest = Path(root) / "crashed" / kind_of(q, s) / f"q{q}" / f"seed{s}" / f"{arm}_attempt1"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(job.out_dir, str(dest))
        ledger["events"].append({"run": job.cfg["run"], "q": q, "seed": s, "arm": arm, "attempt": 1, "reason": cls[0],
                                 "evidence": cls[1], "returncode": res.get("returncode"), "moved_to": str(dest)})
        os.makedirs(job.out_dir, exist_ok=True)
        with open(os.path.join(job.out_dir, "run_config.json"), "w") as f:
            json.dump(job.cfg, f, indent=1)
        rep = runner(job)
        rep["repeat_of_infrastructure_failure"] = True
        repeats.append(rep)
        cls2 = cl.classify_attempt(rep, Path(job.out_dir))
        if cls2 is not None:
            dest2 = Path(root) / "crashed" / kind_of(q, s) / f"q{q}" / f"seed{s}" / f"{arm}_attempt2"
            shutil.move(job.out_dir, str(dest2))
            ledger["events"].append({"run": job.cfg["run"], "q": q, "seed": s, "arm": arm, "attempt": 2, "reason": cls2[0],
                                     "evidence": cls2[1], "returncode": rep.get("returncode"), "moved_to": str(dest2)})
            ledger["stopped_without_verdict"].append(job.cfg["run"])
    return repeats


# ---------------------------------------------------------------------------------- CLI
def require_absolute(parser: argparse.ArgumentParser, flag: str, value: str) -> Path:
    """The path given for ``flag``, which must be absolute (a relative ``--root`` made ``config_equals_plan`` fail for all 122 runs of the
    development wave: the plan hashes the output directory relative to the root's parent, which a relative root cannot do)."""
    path = Path(value)
    if not path.is_absolute():
        parser.error(f"{flag} must be an absolute path, got {value!r}")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    p = argparse.ArgumentParser(description="T3-R3 development-wave launcher", allow_abbrev=False)
    p.add_argument("--params", default=str(tc.PARAMS_PATH), help="the pre-registered bw parameters (JSON)")
    p.add_argument("--workers", type=int, default=40, help=f"at most {MAX_WORKERS}")
    p.add_argument("--root", default=str(DEFAULT_ROOT), help="results root, an absolute path (default <repo>/results/t3_r3)")
    p.add_argument("--code-commit", default=None, help="SHA of the code commit (recorded)")
    p.add_argument("--plan-csv", default=None, help="write the run plan with the config hashes and init digests and exit")
    p.add_argument("--dry-run", action="store_true", help="validate and write the configs + a dry-run record; train nothing")
    a = p.parse_args(argv)
    if not 1 <= a.workers <= MAX_WORKERS:
        p.error(f"--workers must be in 1..{MAX_WORKERS}")
    root = require_absolute(p, "--root", a.root)
    try:
        check_params_file(Path(a.params))
        params = tc.load_bw_params(Path(a.params))
        lock = cl.check_lock(ROOT / "protocols", use_record=True, lock_file=cl.LOCK_FILE)
        jobs = build_jobs(root, params)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    rows = validate_jobs(jobs, lock)
    bad = [r for r in rows if not r["ok"]]
    if bad:
        print(json.dumps(bad[:3], indent=1))
        p.error(f"{len(bad)} config(s) are invalid or differ from the lock; nothing was launched")
    if a.plan_csv:
        write_plan(Path(a.plan_csv), plan_rows(jobs, root))
        print(f"{len(jobs)} planned run(s); plan {a.plan_csv}", flush=True)
        return 0
    if not a.dry_run:
        problems = tree_problems()
        if problems:
            p.error("; ".join(problems))
    busy = L1.check_no_previous_run(jobs)
    if busy:
        p.error(f"{len(busy)} out dir(s) already hold a run (status.json or run.log), e.g. {busy[0]}")
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    record_path = root / (f"dryrun_{stamp}.json" if a.dry_run else f"launch_{stamp}.json")
    record: Dict[str, Any] = {
        "tool": "tools/t3/t3r3_launch.py", "wave": WAVE, "dry_run": bool(a.dry_run), "argv": list(sys.argv if argv is None else argv),
        "started": stamp, "workers": a.workers, **L1.host_record(root), **L1.git_record(a.code_commit),
        "params_file": a.params, "params_sha256": tc.sha256_file(Path(a.params)), "n_planned": len(jobs),
        "planned": [j.out_dir for j in jobs], "runs": [], "state": "dry_run" if a.dry_run else "running"}
    for cfg, out in jobs:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "run_config.json"), "w") as f:
            json.dump(cfg, f, indent=1)
    ledger_path = root / "infra_events.json"
    ledger = {"schema": LEDGER_SCHEMA, "events": [], "stopped_without_verdict": []}
    cl.write_ledger(ledger_path, ledger)
    L1.write_record(record_path, record, create=True)
    print(f"{len(jobs)} run(s) of wave {WAVE}; record {record_path}", flush=True)
    if a.dry_run:
        return 0
    run_pool(jobs, a.workers, record, record_path)
    env = L1.child_env()
    repeats = handle_infra_failures(jobs, record["runs"], root, lambda j: L1.run_job(job_command(*j), j.out_dir, env), ledger)
    cl.write_ledger(ledger_path, ledger)
    if repeats:
        record["infra_repeats"] = repeats
    ok_codes = {r["out"]: r.get("returncode") for r in record["runs"]}
    for r in repeats:
        ok_codes[r["out"]] = r.get("returncode")
    record["state"] = "done" if all(c == 0 for c in ok_codes.values()) else "failed"
    L1.write_record(record_path, record)
    return 0 if record["state"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
