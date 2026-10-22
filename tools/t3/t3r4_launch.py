#!/usr/bin/env python3
"""Confirmation launcher of round T3-R4 (wave ``t3r4``): the arm ``bwA`` (protocol T3-1.1) on the fresh seeds 80501-80540 at q in {50, 60}
= 80 runs into ``<root>/q{q}/seed{s}/bwA`` (default ``results/t3_r4/confirmation``).

``bwA`` is the locked T3-1.0 ``bw`` plus ``stage3_retry = {"delta_gt": 0.005, "max_retries": 1}`` (``run/ms_retry.py``). The launcher reuses
by import, unchanged, ``t3_configs``, ``t3_launch`` (``job_command``, ``run_pool``, ``relpath_sha256``, ``check_params_file``),
``t3c_launch`` (``normalise``, the lock-file readers, the infrastructure classifier and ledger) and ``t3r3_launch`` (the scheme keys, the
planned init digests, ``require_absolute``, the tree guard).

  * seeds exactly 80501-80540 at both q, the arm ``bwA`` only (anything else is refused, also by the config builder);
  * a config is ``t3_configs.build_bw_config`` with ``seed``, ``run``, ``record.run``, ``record.seed``, ``record.output_dir`` and ``pilot`` =
    "t3_r4" set and ``stage3_retry`` added; its normalised form (``t3c_launch.normalise``) must hash to ``reference_config_sha256`` of the
    lock (T3-1.1), and the config without the scheme key must pass ``t3_configs.production_problems``;
  * at a launch: the T3-1.1 record is the LAST record of ``protocols/LOCK``; the lock JSON and MD match its SHA-256; every file listed in
    ``tool_sha256`` of the lock matches; HEAD is the commit that last changed ``protocols/LOCK`` (the LOCK-record commit); the working
    tree is clean outside ``results/`` (the runner's definition of ``clean_tree``); ``git diff --stat b565f0e8 HEAD`` over ``agents envs
    utils run tools/ms tools/v2`` is empty; ``--root`` is absolute. With ``--plan-csv`` / ``--dry-run`` the drafts in
    ``reports/t3/r4/lock_draft/`` are used instead of the record and the git guards are skipped.

``--plan-csv`` writes, for every run, the run name, the output directory, the config SHA-256 (as written and with the output directory made
relative), the planned init digest of attempt 1 and of attempt 2 (the run is constructed without training), then exits.

A real launch writes the configs and an empty infrastructure ledger ``infra_events.json`` next to the runs, runs the pool (at most 40
single-threaded workers) and classifies every finished attempt: a negative return code, an out-of-memory kill or ENOSPC is an
infrastructure failure, moved to ``<root>/crashed/`` and repeated once; a second one is recorded in ``stopped_without_verdict``. Any other
nonzero exit is a failed run, never repeated. A re-initialisation inside a run is part of the run.

Examples (inside tmux):
  python tools/t3/t3r4_launch.py --root $PWD/results/t3_r4/confirmation --plan-csv reports/t3/r4/confirmation_plan.csv
  python tools/t3/t3r4_launch.py --root $PWD/results/t3_r4/confirmation --workers 40 --code-commit <sha>
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
import t3r3_launch as r3  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402
from t3_launch import check_params_file, job_command, relpath_sha256, run_pool  # noqa: E402

WAVE = "t3r4"
ROUND = "t3_r4"
ARM = "bwA"
PROTOCOL_VERSION = "T3-1.1"
QS = (50, 60)
SEEDS = tuple(range(80501, 80541))
REFERENCE_SEED = 50501
RETRY = dict(r3.RETRY)                                  # {"delta_gt": 0.005, "max_retries": 1}
SCHEME_KEYS = r3.SCHEME_KEYS                            # ("stage3_retry", "terminal_clamp_likelihood")
BASE_COMMIT = "b565f0e8dd644a39043ebd3c24875c210b2cfe63"
LOCK_REL = "protocols/v2_T3_locked_v1_1.json"
LOCK_MD_REL = "protocols/v2_T3_locked_v1_1.md"
LOCK_FILE = ROOT / "protocols" / "LOCK"
DRAFT_DIR = ROOT / "reports" / "t3" / "r4" / "lock_draft"
DEFAULT_ROOT = ROOT / "results" / "t3_r4" / "confirmation"
ALLOWED_DIFF = ("seed", "run", "record.run", "record.seed", "record.output_dir", "pilot")
CODE_DIFF_PATHS = ("agents", "envs", "utils", "run", "tools/ms", "tools/v2")
PLAN_FIELDS = ["wave", "arm", "q", "seed", "run", "out_dir", "run_config_sha256", "run_config_sha256_relpath",
               "init_state_sha256", "attempt2_init_state_sha256"]
LEDGER_SCHEMA = "t3r4_infra_events/1"
MAX_WORKERS = L1.MAX_WORKERS


# ---------------------------------------------------------------------------------- names, seeds, configs
def run_name(q: int, seed: int) -> str:
    """``t3r4_q{q}_s{seed}_bwA``."""
    return f"t3r4_q{q}_s{seed}_{ARM}"


def run_dir(root: Path, q: int, seed: int) -> Path:
    """``<root>/q{q}/seed{seed}/bwA``."""
    return Path(root) / f"q{q}" / f"seed{seed}" / ARM


def in_block(q: Any, seed: Any) -> bool:
    """Whether (q, seed) is a planned run of the confirmation."""
    return q in QS and seed in SEEDS


def check_seeds(qs: Sequence[int], seeds: Sequence[int]) -> None:
    """Refuse anything but the seeds 80501-80540 (all of them) at q in {50, 60}.

    Raises:
        ValueError: A q or a seed outside the block, or a partial block.
    """
    if sorted(qs) != list(QS):
        raise ValueError(f"q must be exactly {list(QS)}; got {list(qs)}")
    if sorted(seeds) != list(SEEDS):
        raise ValueError(f"seeds must be exactly {SEEDS[0]}-{SEEDS[-1]}; got {sorted(seeds)}")


def scheme_keys() -> Dict[str, Any]:
    """The scheme's config keys of ``bwA`` (a fresh copy)."""
    return {"stage3_retry": dict(RETRY)}


def build_confirmation_config(proto: Dict[str, Any], params: Dict[str, Any], q: int, seed: int, out_dir: str,
                              protocol_path: Path = tc.PROTOCOL_PATH) -> Dict[str, Any]:
    """The ``bwA`` config of (q, seed): ``t3_configs.build_bw_config`` with the run fields set (``pilot`` = "t3_r4", run name
    ``t3r4_q{q}_s{seed}_bwA``) and ``stage3_retry`` added.

    Raises:
        ValueError: A (q, seed) outside the confirmation block.
    """
    if not in_block(q, seed):
        raise ValueError(f"(q, seed) = ({q}, {seed}) is outside the confirmation block {SEEDS[0]}-{SEEDS[-1]} at q in {QS}")
    cfg = tc.build_bw_config(proto, int(q), int(seed), out_dir, params, protocol_path)
    name = run_name(q, seed)
    cfg["pilot"] = ROUND
    cfg["run"] = name
    cfg["record"]["run"] = name
    cfg.update(scheme_keys())
    return cfg


def reference_config(proto: Dict[str, Any], params: Dict[str, Any], q: int, protocol_path: Path = tc.PROTOCOL_PATH) -> Dict[str, Any]:
    """The T3-1.1 reference config of q: T3-R1's ``bw`` run (seed 50501, ``record.output_dir`` = REFERENCE) with ``stage3_retry``."""
    cfg = tc.build_bw_config(proto, int(q), REFERENCE_SEED, "REFERENCE", params, protocol_path)
    cfg.update(scheme_keys())
    return cfg


def reference_config_sha256(proto: Dict[str, Any], params: Dict[str, Any], q: int, protocol_path: Path = tc.PROTOCOL_PATH) -> str:
    """SHA-256 of the config text of :func:`reference_config`."""
    return tc.config_sha256(reference_config(proto, params, q, protocol_path))


def config_problems(cfg: Dict[str, Any], lock: Dict[str, Any]) -> List[str]:
    """Departures of a planned config from the lock and from the definition of ``bwA`` (empty list: none)."""
    q, seed = cfg.get("q"), cfg.get("seed")
    if not in_block(q, seed):
        return [f"(q, seed) = ({q}, {seed}) is outside the confirmation block"]
    out: List[str] = []
    if cfg.get("arm") != "bw":
        return [f"arm field {cfg.get('arm')!r} != 'bw' (the confirmation runs the arm bwA only)"]
    if cfg.get("pilot") != ROUND:
        out.append(f"pilot {cfg.get('pilot')!r} != {ROUND!r}")
    if cfg.get("run") != run_name(q, seed) or cfg["record"].get("run") != cfg.get("run") or cfg["record"].get("seed") != seed:
        out.append(f"run name / record fields differ from {run_name(q, seed)!r}")
    have = {k: cfg[k] for k in SCHEME_KEYS if k in cfg}
    if have != scheme_keys():
        out.append(f"scheme keys {have} != {scheme_keys()}")
    norm = cl.normalise(cfg)
    got = tc.config_sha256(norm)
    want = lock["reference_config_sha256"][str(q)]
    if got != want:
        out.append(f"the config differs from the lock outside the six run fields (normalised SHA-256 {got} != {want})")
    out += tc.production_problems(r3.strip_scheme(norm))
    return out


# ---------------------------------------------------------------------------------- lock
def read_lock_records(path: Path = LOCK_FILE) -> List[Dict[str, Any]]:
    """All records of ``protocols/LOCK`` (concatenated JSON objects)."""
    return cl.read_lock_records(path)


def t31_record(path: Path = LOCK_FILE) -> Optional[Dict[str, Any]]:
    """The T3-1.1 record of ``protocols/LOCK`` (``None`` if absent)."""
    for r in read_lock_records(path):
        if r.get("protocol") == LOCK_REL:
            return r
    return None


def check_lock(lock_dir: Path, use_record: bool, lock_file: Path = LOCK_FILE, repo: Path = ROOT) -> Dict[str, Any]:
    """Verify the lock files and return the lock JSON.

    Args:
        lock_dir: Directory holding ``v2_T3_locked_v1_1.json`` / ``.md`` (``protocols/`` or the drafts).
        use_record: Require the T3-1.1 record to be the LAST record of ``protocols/LOCK`` and match both files against its SHA-256 (a launch).
        lock_file: The LOCK file.
        repo: Repository root the paths of the lock are resolved against.

    Raises:
        ValueError: A missing or non-final record or file, a hash mismatch, or a listed file whose SHA-256 differs from the lock's.
    """
    js, md = Path(lock_dir) / Path(LOCK_REL).name, Path(lock_dir) / Path(LOCK_MD_REL).name
    for p in (js, md):
        if not p.is_file():
            raise ValueError(f"lock file {p} is missing")
    if use_record:
        records = read_lock_records(lock_file)
        named = [r for r in records if r.get("protocol") == LOCK_REL]
        if not named:
            raise ValueError(f"{lock_file} holds no record for {LOCK_REL}: the confirmation is not locked")
        if len(named) > 1:
            raise ValueError(f"{lock_file} holds {len(named)} records for {LOCK_REL}; there must be exactly one")
        rec = named[0]
        if records[-1] is not rec:
            raise ValueError(f"the {PROTOCOL_VERSION} record is not the last record of {lock_file}")
        if cl.sha256_file(js) != rec["protocol_sha256"] or cl.sha256_file(md) != rec["protocol_md_sha256"]:
            raise ValueError("the lock JSON / MD do not match the SHA-256 of the LOCK record")
    lock = json.loads(js.read_text())
    if lock.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"protocol_version {lock.get('protocol_version')!r} != {PROTOCOL_VERSION!r}")
    bad = [f for f, sha in lock["tool_sha256"].items() if not (Path(repo) / f).is_file() or cl.sha256_file(Path(repo) / f) != sha]
    if bad:
        raise ValueError(f"file(s) differ from the lock: {bad}")
    for rel, key in ((lock["protocol_file"], "protocol_sha256"), (lock["params_file"], "params_sha256"),
                     (lock["criteria_file"], "criteria_sha256")):
        if cl.sha256_file(Path(repo) / rel) != lock[key]:
            raise ValueError(f"{rel} differs from the lock ({key})")
    return lock


# ---------------------------------------------------------------------------------- git guards
def tree_problems(repo: Path = ROOT) -> List[str]:
    """A working tree that is not clean outside ``results/`` (the runner's own definition of ``clean_tree``)."""
    return r3.tree_problems(repo)


def code_diff_problems(repo: Path = ROOT, base: str = BASE_COMMIT) -> List[str]:
    """A non-empty ``git diff`` of the frozen code paths between ``base`` and HEAD."""
    out = cl._git("diff", "--name-status", base, "HEAD", "--", *CODE_DIFF_PATHS, repo=repo).splitlines()
    return [f"code changed since {base[:8]}: {line}" for line in out]


def head_problems(repo: Path = ROOT, lock_file_rel: str = "protocols/LOCK") -> List[str]:
    """HEAD must be the commit that last changed ``protocols/LOCK`` (the LOCK-record commit)."""
    head = cl._git("rev-parse", "HEAD", repo=repo).strip()
    last = cl._git("log", "-1", "--format=%H", "--", lock_file_rel, repo=repo).strip()
    if not last:
        return [f"{lock_file_rel} has no commit"]
    return [] if head == last else [f"HEAD {head[:8]} is not the LOCK-record commit {last[:8]}"]


def launch_problems(repo: Path = ROOT, base: str = BASE_COMMIT) -> List[str]:
    """All git guards of a real launch (see the module docstring)."""
    return tree_problems(repo) + code_diff_problems(repo, base) + head_problems(repo)


# ---------------------------------------------------------------------------------- jobs and plan
def build_jobs(qs: Sequence[int], seeds: Sequence[int], root: Path, params: Dict[str, Any]) -> List[L1.Job]:
    """The 80 jobs (configs are built, not written); q-major, then seed.

    Raises:
        ValueError: A q or seed outside the confirmation block.
    """
    check_seeds(qs, seeds)
    proto = tc.load_protocol()
    jobs: List[L1.Job] = []
    for q in sorted(qs):
        for s in sorted(seeds):
            out = str(run_dir(root, int(q), int(s)))
            jobs.append(L1.Job(build_confirmation_config(proto, params, int(q), int(s), out), out))
    return jobs


def plan_rows(jobs: Sequence[L1.Job], root: Path, with_digests: bool = True) -> List[Dict[str, Any]]:
    """One row per planned run (see the module docstring)."""
    rows = []
    for cfg, out in jobs:
        row = {"wave": WAVE, "arm": ARM, "q": cfg["q"], "seed": cfg["seed"], "run": cfg["run"], "out_dir": out,
               "run_config_sha256": tc.config_sha256(cfg), "run_config_sha256_relpath": relpath_sha256(cfg, root)}
        row.update(r3.init_digests(cfg, ARM) if with_digests else {"init_state_sha256": "", "attempt2_init_state_sha256": ""})
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
        row: Dict[str, Any] = {"out": out, "q": cfg["q"], "seed": cfg["seed"], "ok": True, "error": None}
        try:
            rms.validate_config(cfg)
            problems = config_problems(cfg, lock)
            if problems:
                raise ValueError("; ".join(problems))
        except Exception as exc:  # noqa: BLE001 - reported
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------- infrastructure failures
def new_ledger() -> Dict[str, Any]:
    """An empty infrastructure ledger."""
    return {"schema": LEDGER_SCHEMA, "events": [], "stopped_without_verdict": []}


def handle_infra_failures(jobs: Sequence[L1.Job], results: Sequence[Dict[str, Any]], root: Path,
                          runner: Callable[[L1.Job], Dict[str, Any]], ledger: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Move every infrastructure-failed attempt to ``<root>/crashed/`` and repeat the run once into the original path (the classification
    is ``t3c_launch.classify_attempt``: a negative return code, an out-of-memory kill, ENOSPC). A second infrastructure failure of a run is
    recorded in ``stopped_without_verdict``. Returns the results of the repeats."""
    by_out = {j.out_dir: j for j in jobs}
    repeats: List[Dict[str, Any]] = []
    ledger_path = Path(root) / "infra_events.json"
    for res in results:
        job = by_out.get(res["out"])
        if job is None or res.get("returncode") == 0:             # a run that exited 0 is not an infrastructure failure, whatever its log says
            continue
        cls = cl.classify_attempt(res, Path(job.out_dir))
        if cls is None:
            continue
        q, s = int(job.cfg["q"]), int(job.cfg["seed"])
        dest = Path(root) / "crashed" / f"q{q}" / f"seed{s}" / f"{ARM}_attempt1"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(job.out_dir, str(dest))
        ledger["events"].append({"run": job.cfg["run"], "q": q, "seed": s, "arm": ARM, "attempt": 1, "reason": cls[0],
                                 "evidence": cls[1], "returncode": res.get("returncode"), "moved_to": str(dest)})
        cl.write_ledger(ledger_path, ledger)                       # the ledger is saved after every event, before the repeat starts
        os.makedirs(job.out_dir, exist_ok=True)
        with open(os.path.join(job.out_dir, "run_config.json"), "w") as f:
            json.dump(job.cfg, f, indent=1)
        rep = runner(job)
        rep["repeat_of_infrastructure_failure"] = True
        repeats.append(rep)
        cls2 = cl.classify_attempt(rep, Path(job.out_dir)) if rep.get("returncode") != 0 else None
        if cls2 is not None:
            dest2 = Path(root) / "crashed" / f"q{q}" / f"seed{s}" / f"{ARM}_attempt2"
            shutil.move(job.out_dir, str(dest2))
            ledger["events"].append({"run": job.cfg["run"], "q": q, "seed": s, "arm": ARM, "attempt": 2, "reason": cls2[0],
                                     "evidence": cls2[1], "returncode": rep.get("returncode"), "moved_to": str(dest2)})
            ledger["stopped_without_verdict"].append(job.cfg["run"])
            cl.write_ledger(ledger_path, ledger)
    return repeats


# ---------------------------------------------------------------------------------- CLI
def require_absolute(parser: argparse.ArgumentParser, flag: str, value: str) -> Path:
    """The path given for ``flag``, which must be absolute (see ``t3r3_launch.require_absolute``)."""
    return r3.require_absolute(parser, flag, value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    p = argparse.ArgumentParser(description="T3-R4 confirmation launcher", allow_abbrev=False)
    p.add_argument("--qs", type=int, nargs="+", default=list(QS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--params", default=str(tc.PARAMS_PATH), help="the pre-registered bw parameters (JSON)")
    p.add_argument("--workers", type=int, default=40, help=f"at most {MAX_WORKERS}")
    p.add_argument("--root", default=str(DEFAULT_ROOT), help="results root, an absolute path (default <repo>/results/t3_r4/confirmation)")
    p.add_argument("--lock-dir", default=None, help="default: protocols/ (a launch), the drafts with --plan-csv / --dry-run")
    p.add_argument("--code-commit", default=None, help="SHA of the code commit (recorded)")
    p.add_argument("--plan-csv", default=None, help="write the run plan with the config hashes and init digests and exit")
    p.add_argument("--dry-run", action="store_true", help="validate and write the configs + a dry-run record; train nothing")
    a = p.parse_args(argv)
    if not 1 <= a.workers <= MAX_WORKERS:
        p.error(f"--workers must be in 1..{MAX_WORKERS}")
    root = require_absolute(p, "--root", a.root)
    planning = a.plan_csv is not None
    try:
        check_params_file(Path(a.params))
        params = tc.load_bw_params(Path(a.params))
        lock_dir = Path(a.lock_dir) if a.lock_dir else (DRAFT_DIR if (planning or a.dry_run) else ROOT / "protocols")
        lock = check_lock(lock_dir, use_record=not planning and not a.dry_run, lock_file=LOCK_FILE)
        jobs = build_jobs(a.qs, a.seeds, root, params)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    rows = validate_jobs(jobs, lock)
    bad = [r for r in rows if not r["ok"]]
    if bad:
        print(json.dumps(bad[:3], indent=1))
        p.error(f"{len(bad)} config(s) are invalid or differ from the lock; nothing was launched")
    if planning:
        write_plan(Path(a.plan_csv), plan_rows(jobs, root))
        print(f"{len(jobs)} planned run(s) of wave {WAVE}; plan {a.plan_csv}", flush=True)
        return 0
    if not a.dry_run:
        problems = launch_problems()
        if problems:
            p.error("; ".join(problems))
    busy = L1.check_no_previous_run(jobs)
    if busy:
        p.error(f"{len(busy)} out dir(s) already hold a run (status.json or run.log), e.g. {busy[0]}")
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    record_path = root / (f"dryrun_{stamp}.json" if a.dry_run else f"launch_{stamp}.json")
    record: Dict[str, Any] = {
        "tool": "tools/t3/t3r4_launch.py", "wave": WAVE, "dry_run": bool(a.dry_run), "argv": list(sys.argv if argv is None else argv),
        "started": stamp, "workers": a.workers, **L1.host_record(root), **L1.git_record(a.code_commit),
        "base_commit": BASE_COMMIT, "lock_dir": str(lock_dir), "lock_protocol_sha256": cl.sha256_file(lock_dir / Path(LOCK_REL).name),
        "params_file": a.params, "params_sha256": tc.sha256_file(Path(a.params)), "n_planned": len(jobs),
        "planned": [j.out_dir for j in jobs], "runs": [], "state": "dry_run" if a.dry_run else "running"}
    for cfg, out in jobs:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "run_config.json"), "w") as f:
            json.dump(cfg, f, indent=1)
    ledger_path = root / "infra_events.json"
    ledger = new_ledger()
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
