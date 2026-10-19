#!/usr/bin/env python3
"""Confirmation launcher of round T3-R2 (wave ``t3c``): the arm ``bw`` of round T3-R1, unchanged, on the fresh seeds
60501-60520 at q in {50, 60} = 40 runs into ``<root>/q{q}/seed{s}/bw`` (default ``results/t3_r2/confirmation``).

Reuses by import, unchanged, ``job_command``, ``run_pool``, ``plan_rows``, ``relpath_sha256`` and ``check_params_file`` of
``tools/t3/t3_launch.py``. ``t3_configs.check_seeds`` / ``WAVES`` / ``production_problems`` reject this wave by design, so the
seed guard and the production check are this module's own:

  * seeds exactly 60501-60520 at both q, the arm ``bw`` only;
  * the config of a run is ``t3_configs.build_bw_config`` with the D4 fields set (``seed``, ``run``, ``record.run``,
    ``record.seed``, ``record.output_dir``, ``pilot`` = "t3_r2"); a config that differs from the lock outside those fields
    is refused (its normalised form must hash to ``reference_config_sha256`` of the lock);
  * at launch the T3 record is read from ``protocols/LOCK`` (absent: refused), the lock JSON and MD must match its SHA-256,
    and every tool file listed in the lock JSON must match its recorded SHA-256; with ``--plan-csv`` the drafts in
    ``reports/t3/r2/lock_draft/`` are checked instead (no record exists yet);
  * the working tree must be clean outside ``results/`` (the runner's definition of ``clean_tree``) and ``git diff --name-status be2b7e9f HEAD`` over ``agents envs utils run tools/ms
    tools/v2 tools/t3`` may show only new ``tools/t3/t3c_*`` and ``tools/t3/t3r2_*`` files.

``--plan-csv`` writes, for every run, the run name, the output directory, the config SHA-256 (as written and with the output
directory made relative) and the planned init digest (the run is constructed without training), then exits.

A real launch writes the configs and an empty infrastructure ledger ``infra_events.json`` next to the runs, runs the pool
(at most 40 single-threaded workers), classifies every finished attempt (D4, "Failure counting"): a negative return code,
an out-of-memory kill in the run's log or an ENOSPC ``OSError`` is an infrastructure failure; such an attempt is moved to
``results/t3_r2/crashed/`` and the run is repeated once into the original path; a second infrastructure failure stops the
confirmation without a verdict (recorded in the ledger). Any other nonzero exit is a failed run, never repeated.

Examples (inside tmux):
  python tools/t3/t3c_launch.py --plan-csv reports/t3/r2/confirmation_plan.csv
  python tools/t3/t3c_launch.py --workers 40 --code-commit <sha>
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import launch_ms_r1 as L1  # noqa: E402
import t3_configs as tc  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402
from t3_launch import (check_params_file, job_command, plan_rows, relpath_sha256,  # noqa: E402,F401
                       run_pool)

WAVE = "t3c"
ROUND = "t3_r2"
ARM = "bw"
QS = (50, 60)
SEEDS = tuple(range(60501, 60521))
REFERENCE_SEED = 50501
BASE_COMMIT = "be2b7e9fe3e0aad34f7481a1d8306af7cc2963e7"
LOCK_REL = "protocols/v2_T3_locked_v1_0.json"
LOCK_MD_REL = "protocols/v2_T3_locked_v1_0.md"
LOCK_FILE = ROOT / "protocols" / "LOCK"
DRAFT_DIR = ROOT / "reports" / "t3" / "r2" / "lock_draft"
DEFAULT_ROOT = ROOT / "results" / "t3_r2" / "confirmation"
CRASHED_REL = Path("results") / "t3_r2" / "crashed"
ALLOWED_DIFF = ("seed", "run", "record.run", "record.seed", "record.output_dir", "pilot")
CODE_PATHS = ("agents", "envs", "utils", "run", "tools/ms", "tools/v2", "tools/t3")
LEDGER_SCHEMA = "t3c_infra_events/1"
PLAN_FIELDS = ["wave", "arm", "q", "seed", "run", "out_dir", "run_config_sha256", "run_config_sha256_relpath",
               "init_state_sha256"]
MAX_WORKERS = L1.MAX_WORKERS


# ---------------------------------------------------------------------------------- names, configs
def run_name(q: int, seed: int) -> str:
    """``t3r2_q{q}_s{seed}_bw``."""
    return f"t3r2_q{q}_s{seed}_bw"


def run_dir(root: Path, q: int, seed: int) -> Path:
    """``<root>/q{q}/seed{seed}/bw``."""
    return Path(root) / f"q{q}" / f"seed{seed}" / ARM


def check_seeds(qs: Sequence[int], seeds: Sequence[int]) -> None:
    """Refuse anything but the seeds 60501-60520 (all of them) at q in {50, 60}.

    Raises:
        ValueError: A q or a seed outside the block, or a partial block.
    """
    if sorted(qs) != list(QS):
        raise ValueError(f"q must be exactly {list(QS)}; got {list(qs)}")
    if sorted(seeds) != list(SEEDS):
        raise ValueError(f"seeds must be exactly {SEEDS[0]}-{SEEDS[-1]}; got {sorted(seeds)}")


def build_confirmation_config(proto: Dict[str, Any], q: int, seed: int, out_dir: str, params: Dict[str, Any],
                              protocol_path: Path = tc.PROTOCOL_PATH) -> Dict[str, Any]:
    """T3-R1's ``bw`` config of (q, seed) with the D4 fields set: ``pilot`` = "t3_r2", run name ``t3r2_q{q}_s{seed}_bw``."""
    cfg = tc.build_bw_config(proto, q, seed, out_dir, params, protocol_path)
    name = run_name(q, seed)
    cfg["pilot"] = ROUND
    cfg["run"] = name
    cfg["record"]["run"] = name
    return cfg


def normalise(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The config with the D4 fields reset to the reference run of T3-R1 (seed 50501, ``record.output_dir`` = REFERENCE)."""
    c = copy.deepcopy(cfg)
    q = int(c["q"])
    name = tc.run_name(q, REFERENCE_SEED, ARM)
    c["seed"] = REFERENCE_SEED
    c["record"]["seed"] = REFERENCE_SEED
    c["run"] = name
    c["record"]["run"] = name
    c["record"]["output_dir"] = "REFERENCE"
    c["pilot"] = tc.ROUND
    return c


def reference_config_sha256(proto: Dict[str, Any], q: int, params: Dict[str, Any],
                            protocol_path: Path = tc.PROTOCOL_PATH) -> str:
    """SHA-256 of the config text of T3-R1's ``bw`` run (q, seed 50501) with ``record.output_dir`` = REFERENCE."""
    return tc.config_sha256(tc.build_bw_config(proto, q, REFERENCE_SEED, "REFERENCE", params, protocol_path))


def config_problems(cfg: Dict[str, Any], lock: Dict[str, Any]) -> List[str]:
    """Departures of a planned config from the lock (empty list: none)."""
    out: List[str] = []
    q, seed = cfg.get("q"), cfg.get("seed")
    if cfg.get("arm") != ARM:
        return [f"arm {cfg.get('arm')!r} != {ARM!r}"]
    if q not in QS or seed not in SEEDS:
        out.append(f"(q, seed) = ({q}, {seed}) is outside the confirmation block")
        return out
    if cfg.get("pilot") != ROUND:
        out.append(f"pilot {cfg.get('pilot')!r} != {ROUND!r}")
    if cfg.get("run") != run_name(q, seed) or cfg["record"].get("run") != cfg.get("run") or cfg["record"].get("seed") != seed:
        out.append(f"run name / record fields differ from {run_name(q, seed)!r}")
    norm = normalise(cfg)
    got = tc.config_sha256(norm)
    want = lock["reference_config_sha256"][str(q)]
    if got != want:
        out.append(f"the config differs from the lock outside the D4 fields (normalised SHA-256 {got} != {want})")
    out += tc.production_problems(norm)
    return out


# ---------------------------------------------------------------------------------- lock
def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_lock_records(path: Path = LOCK_FILE) -> List[Dict[str, Any]]:
    """All records of ``protocols/LOCK`` (concatenated JSON objects)."""
    text = Path(path).read_text()
    dec, i, out = json.JSONDecoder(), 0, []
    while True:
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            return out
        obj, i = dec.raw_decode(text, i)
        out.append(obj)


def t3_record(path: Path = LOCK_FILE) -> Optional[Dict[str, Any]]:
    """The T3 record of ``protocols/LOCK`` (``None`` if absent)."""
    for r in read_lock_records(path):
        if r.get("protocol") == LOCK_REL:
            return r
    return None


def check_lock(lock_dir: Path, use_record: bool, lock_file: Path = LOCK_FILE, repo: Path = ROOT) -> Dict[str, Any]:
    """Verify the lock files and return the lock JSON.

    Args:
        lock_dir: Directory holding ``v2_T3_locked_v1_0.json`` / ``.md`` (``protocols/`` or the drafts).
        use_record: Require the T3 record of ``protocols/LOCK`` and match both files against its SHA-256 (a launch).
        lock_file: The LOCK file.
        repo: Repository root the tool paths of the lock are resolved against.

    Raises:
        ValueError: A missing record or file, a hash mismatch, or a tool file whose SHA-256 differs from the lock's.
    """
    js, md = Path(lock_dir) / Path(LOCK_REL).name, Path(lock_dir) / Path(LOCK_MD_REL).name
    for p in (js, md):
        if not p.is_file():
            raise ValueError(f"lock file {p} is missing")
    if use_record:
        rec = t3_record(lock_file)
        if rec is None:
            raise ValueError(f"{lock_file} holds no record for {LOCK_REL}: the confirmation is not locked")
        if sha256_file(js) != rec["protocol_sha256"] or sha256_file(md) != rec["protocol_md_sha256"]:
            raise ValueError("the lock JSON / MD do not match the SHA-256 of the LOCK record")
    lock = json.loads(js.read_text())
    bad = [f for f, sha in lock["tool_sha256"].items() if not (Path(repo) / f).is_file() or sha256_file(Path(repo) / f) != sha]
    if bad:
        raise ValueError(f"tool file(s) differ from the lock: {bad}")
    for rel, key in ((lock["protocol_file"], "protocol_sha256"), (lock["params_file"], "params_sha256"),
                     (lock["criteria_file"], "criteria_sha256")):
        if sha256_file(Path(repo) / rel) != lock[key]:
            raise ValueError(f"{rel} differs from the lock ({key})")
    return lock


# ---------------------------------------------------------------------------------- git guards
def _git(*args: str, repo: Path = ROOT) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def tree_problems(repo: Path = ROOT, base: str = BASE_COMMIT) -> List[str]:
    """A dirty working tree, or a code path that changed since ``base`` other than by a new ``t3c_*`` / ``t3r2_*`` file."""
    out: List[str] = []
    # the runner's own definition of "dirty" (``run.run_v2_stagewise.git_state``, the ``clean_tree`` flag of every manifest):
    # tracked changes or untracked files anywhere except results/ (run outputs, weights and arrays live there, untracked)
    dirty = _git("status", "--porcelain", "--", ".", ":(exclude)results", repo=repo).splitlines()
    if dirty:
        out.append(f"working tree is not clean ({len(dirty)} entries outside results/, e.g. {dirty[0]!r})")
    for line in _git("diff", "--name-status", base, "HEAD", "--", *CODE_PATHS, repo=repo).splitlines():
        status, path = line.split("\t", 1)
        ok = status == "A" and re.fullmatch(r"tools/t3/(t3c_|t3r2_)[^/]*", path) is not None
        if not ok:
            out.append(f"code changed since {base[:8]}: {line}")
    return out


# ---------------------------------------------------------------------------------- jobs and plan
def build_jobs(qs: Sequence[int], seeds: Sequence[int], root: Path, params: Dict[str, Any]) -> List[L1.Job]:
    """The 40 jobs (configs are built, not written); q-major, then seed.

    Raises:
        ValueError: A q or seed outside the confirmation block.
    """
    check_seeds(qs, seeds)
    proto = tc.load_protocol()
    jobs: List[L1.Job] = []
    for q in sorted(qs):
        for s in sorted(seeds):
            out = str(run_dir(root, int(q), int(s)))
            jobs.append(L1.Job(build_confirmation_config(proto, int(q), int(s), out, params), out))
    return jobs


def init_digest(cfg: Dict[str, Any]) -> str:
    """The init digest of a planned run: the run is constructed (no training, no file written) and its initial actor and
    critic weights are hashed."""
    return rms.MSRun(copy.deepcopy(cfg), str(Path(cfg["record"]["output_dir"]) / "_not_written")).init_state_sha256


def plan_with_digests(jobs: Sequence[L1.Job], root: Path) -> List[Dict[str, Any]]:
    """``plan_rows`` of t3_launch plus the planned init digest of every run."""
    rows = plan_rows(WAVE, jobs, root)
    for r, j in zip(rows, jobs):
        r["init_state_sha256"] = init_digest(j.cfg)
    return rows


def write_plan(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """The plan CSV (refuses to replace an existing file). ``t3_launch.write_plan_csv`` has fixed fieldnames and cannot carry
    the init digest column, so this module writes the same columns plus ``init_state_sha256``."""
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


def write_ledger(path: Path, ledger: Dict[str, Any]) -> None:
    """Write ``infra_events.json``."""
    with open(path, "w") as f:
        json.dump(ledger, f, indent=1)


def classify_attempt(result: Dict[str, Any], out_dir: Path) -> Optional[Tuple[str, str]]:
    """``(reason, evidence)`` if a finished attempt is an infrastructure failure, else ``None``.

    The three reasons of D4: a negative return code (a signal), an out-of-memory kill recorded in the run's log, an
    ``OSError`` ENOSPC in the run's ``status.json`` or traceback (or in the log / the launcher's error text).
    """
    rc = result.get("returncode")
    if isinstance(rc, int) and rc < 0:
        return "negative_returncode", f"returncode {rc}"
    texts = [str(result.get("error", ""))]
    st = Path(out_dir) / "status.json"
    if st.is_file():
        try:
            texts.append(str(json.loads(st.read_text()).get("traceback", "")))
        except json.JSONDecodeError:
            pass
    log = Path(out_dir) / "run.log"
    log_text = log.read_text(errors="replace")[-20000:] if log.is_file() else ""
    texts.append(log_text)
    for t in texts:
        if "OSError" in t and ("[Errno 28]" in t or "No space left on device" in t):
            return "enospc", "OSError [Errno 28] No space left on device"
    if re.search(r"MemoryError|Out of memory|oom-kill|Cannot allocate memory", log_text):
        return "out_of_memory", "out-of-memory message in run.log"
    return None


def handle_infra_failures(jobs: Sequence[L1.Job], results: Sequence[Dict[str, Any]], root: Path,
                          runner: Callable[[L1.Job], Dict[str, Any]], ledger: Dict[str, Any], repo: Path = ROOT
                          ) -> List[Dict[str, Any]]:
    """Move every infrastructure-failed attempt to ``results/t3_r2/crashed/`` and repeat the run once into the original path.

    A second infrastructure failure of a run is recorded in ``stopped_without_verdict``.

    Args:
        jobs: The planned jobs.
        results: The pool's result per finished attempt (``out``, ``returncode``, ...).
        root: The confirmation root.
        runner: ``runner(job) -> result`` (the repeat; ``L1.run_job`` on the job's command in a real launch).
        ledger: The ledger (updated in place).
        repo: The repository root (the crashed directory is under it).

    Returns:
        The results of the repeats.
    """
    by_out = {j.out_dir: j for j in jobs}
    repeats: List[Dict[str, Any]] = []
    for res in results:
        job = by_out.get(res["out"])
        if job is None:
            continue
        cls = classify_attempt(res, Path(job.out_dir))
        if cls is None:
            continue
        q, s = int(job.cfg["q"]), int(job.cfg["seed"])
        dest = Path(repo) / CRASHED_REL / f"q{q}" / f"seed{s}" / f"{ARM}_attempt1"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(job.out_dir, str(dest))
        ledger["events"].append({"run": job.cfg["run"], "q": q, "seed": s, "attempt": 1, "reason": cls[0],
                                 "evidence": cls[1], "returncode": res.get("returncode"), "moved_to": str(dest)})
        os.makedirs(job.out_dir, exist_ok=True)
        with open(os.path.join(job.out_dir, "run_config.json"), "w") as f:
            json.dump(job.cfg, f, indent=1)
        rep = runner(job)
        rep["repeat_of_infrastructure_failure"] = True
        repeats.append(rep)
        cls2 = classify_attempt(rep, Path(job.out_dir))
        if cls2 is not None:
            dest2 = Path(repo) / CRASHED_REL / f"q{q}" / f"seed{s}" / f"{ARM}_attempt2"
            shutil.move(job.out_dir, str(dest2))
            ledger["events"].append({"run": job.cfg["run"], "q": q, "seed": s, "attempt": 2, "reason": cls2[0],
                                     "evidence": cls2[1], "returncode": rep.get("returncode"), "moved_to": str(dest2)})
            ledger["stopped_without_verdict"].append(job.cfg["run"])
    return repeats


# ---------------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    p = argparse.ArgumentParser(description="T3-R2 confirmation launcher", allow_abbrev=False)
    p.add_argument("--qs", type=int, nargs="+", default=list(QS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--params", default=str(tc.PARAMS_PATH), help="the pre-registered bw parameters (JSON)")
    p.add_argument("--workers", type=int, default=40, help=f"at most {MAX_WORKERS}")
    p.add_argument("--root", default=str(DEFAULT_ROOT), help="results root (default results/t3_r2/confirmation)")
    p.add_argument("--lock-dir", default=None, help="default: protocols/ (a launch), the drafts with --plan-csv")
    p.add_argument("--code-commit", default=None, help="SHA of the code commit (recorded)")
    p.add_argument("--plan-csv", default=None, help="write the run plan with the config hashes and init digests and exit")
    p.add_argument("--dry-run", action="store_true", help="validate and write the configs + a dry-run record; train nothing")
    a = p.parse_args(argv)
    if not 1 <= a.workers <= MAX_WORKERS:
        p.error(f"--workers must be in 1..{MAX_WORKERS}")
    root = Path(a.root).resolve()
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
        write_plan(Path(a.plan_csv), plan_with_digests(jobs, root))
        print(f"{len(jobs)} planned run(s) of wave {WAVE}; plan {a.plan_csv}", flush=True)
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
        "tool": "tools/t3/t3c_launch.py", "wave": WAVE, "dry_run": bool(a.dry_run), "argv": list(sys.argv if argv is None else argv),
        "started": stamp, "workers": a.workers, **L1.host_record(root), **L1.git_record(a.code_commit or BASE_COMMIT),
        "base_commit": BASE_COMMIT, "lock_dir": str(lock_dir), "lock_protocol_sha256": sha256_file(lock_dir / Path(LOCK_REL).name),
        "params_file": a.params, "params_sha256": tc.sha256_file(Path(a.params)), "n_planned": len(jobs),
        "planned": [j.out_dir for j in jobs], "runs": [], "state": "dry_run" if a.dry_run else "running"}
    for cfg, out in jobs:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "run_config.json"), "w") as f:
            json.dump(cfg, f, indent=1)
    ledger_path = root / "infra_events.json"
    ledger = new_ledger()
    write_ledger(ledger_path, ledger)
    L1.write_record(record_path, record, create=True)
    print(f"{len(jobs)} run(s) of wave {WAVE}; record {record_path}", flush=True)
    if a.dry_run:
        return 0
    run_pool(jobs, a.workers, record, record_path)
    env = L1.child_env()
    repeats = handle_infra_failures(jobs, record["runs"], root,
                                    lambda j: L1.run_job(job_command(*j), j.out_dir, env), ledger,
                                    repo=ROOT)
    write_ledger(ledger_path, ledger)
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
