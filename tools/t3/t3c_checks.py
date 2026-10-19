#!/usr/bin/env python3
"""Post-launch checks of the T3-R2 confirmation (prompt section 3.2), on the 40 runs of wave ``t3c``.

  basics     exit code 0 / status done, complete files, manifest at the launch commit with a clean tree, the process-global RNG
             assertion ok, the config equal to the plan (``confirmation_plan.csv``, SHA-256 with the output directory made
             relative), the production values (D4), ``infra_events.json`` present;
  budgets    updates 1600 / 1600 / 600 per phase, policy rows 819,200 / 819,200 / 307,200, minibatch steps 76,000;
  schedules  the learning rate at every update, snapshot refreshes at the phase entries and at multiples of 20, 152 exports
             (``u00025`` ... ``u03800``), 64 / 64 / 24 checks per stage;
  C-INIT     every recorded init digest equals the planned one of ``confirmation_plan.csv``;
  C-REPLAY   ``t3_eval`` (through ``t3c_eval``) reproduces each run's ``freeze_stage1_{final,development}.npz`` bit for bit.

The per-run budget, schedule and start checks are those of ``tools/t3/t3_launch_checks.py`` (``_check_bw``, imported unchanged).
Wall-clock columns are not compared. A deterministic failure is a stop-and-report (exit code 1).

Usage:
    python tools/t3/t3c_checks.py --root results/t3_r2/confirmation --plan reports/t3/r2/confirmation_plan.csv \
        --commit <launch HEAD> --out results/t3_r2/confirmation/launch_checks.json --workers 40
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import t3_configs as tc  # noqa: E402
import t3_eval as te  # noqa: E402
import t3_launch_checks as lc  # noqa: E402
import t3c_eval as ce  # noqa: E402
import t3c_launch as cl  # noqa: E402

VERDICTS = ["exit_ok", "files_complete", "manifest_commit_ok", "clean_tree", "global_rng_ok", "config_equals_plan",
            "config_production_ok", "budget_ok", "lr_ok", "refresh_ok", "exports_ok", "checks_ok", "init_equals_plan",
            "replay_ok"]


def read_plan(path: Path) -> Dict[str, Dict[str, str]]:
    """``{run name: plan row}`` of ``confirmation_plan.csv``."""
    return {r["run"]: r for r in lc.read_csv(Path(path))}


def check_run(args: Any) -> Dict[str, Any]:
    """All post-launch checks of one run.

    Args:
        args: ``(job, plan_row, commit, lock)`` with a job of :func:`t3c_eval.plan_jobs`.

    Returns:
        A flat dict of verdicts (booleans) and details.
    """
    job, plan, commit, lock = args
    torch.set_num_threads(1)
    d = Path(job["run_dir"])
    out: Dict[str, Any] = {k: job[k] for k in ("wave", "arm", "q", "seed", "run")}
    st = te.status_of(d)
    out["exit_ok"] = bool(st.get("state") == "done" and st.get("exit_code") == 0)
    missing = [f for f in lc.REQUIRED_FILES["bw"] if not (d / f).exists()]
    out["files_complete"] = not missing
    out["files_missing"] = ";".join(missing)
    if not out["exit_ok"] or missing:
        return out
    cfg = te.read_json(d / "run_config.json")
    man = te.read_json(d / "manifest.json")
    out["manifest_commit_ok"] = bool(commit is None or man["commit"] == commit)
    out["clean_tree"] = bool(man["clean_tree"] is True)
    out["init_state_sha256"] = man["init_state_sha256"]
    out["global_rng_ok"] = bool(te.global_rng_ok(d, "bw"))
    out["config_equals_plan"] = bool(
        plan is not None and cl.relpath_sha256(cfg, Path(d).parents[2]) == plan["run_config_sha256_relpath"])
    out["config_production_ok"] = not cl.config_problems(cfg, lock)
    sched = cfg["record"]["lr_schedule"]
    lc._check_bw(d, cfg, sched, tc.expected_budget("bw"), out)      # budgets, schedules, exports, checks
    out["init_equals_plan"] = bool(plan is not None and man["init_state_sha256"] == plan["init_state_sha256"])
    out["replay_ok"] = not te.replay_mismatches(d, "bw")
    return out


def run_checks(root: Path, plan_path: Path, commit: Optional[str], workers: int, lock: Mapping[str, Any]) -> Dict[str, Any]:
    """Run every check on the 40 runs; the aggregate (counts per verdict, the ledger, per-run results)."""
    plan = read_plan(plan_path)
    jobs = ce.plan_jobs(root, None)
    args = [(j, plan.get(j["run"]), commit, lock) for j in jobs]
    if workers <= 1:
        res = [check_run(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(args)), mp_context=mp.get_context("spawn")) as pool:
            res = list(pool.map(check_run, args))
    agg: Dict[str, Any] = {"n_runs": len(res), "verdicts": {}}
    for v in VERDICTS:
        vals = [r.get(v) for r in res]
        agg["verdicts"][v] = {"pass": sum(1 for x in vals if x is True), "fail": sum(1 for x in vals if x is False),
                              "not_evaluated": sum(1 for x in vals if x is None)}
    ledger_path = Path(root) / "infra_events.json"
    agg["infra_events_present"] = ledger_path.is_file()
    agg["infra_events"] = json.loads(ledger_path.read_text()) if ledger_path.is_file() else None
    agg["runs"] = res
    return agg


def checks_ok(res: Mapping[str, Any]) -> bool:
    """True iff every deterministic check passed on every run and the infrastructure ledger exists."""
    bad = [v for v, c in res["verdicts"].items() if c["fail"] or c["not_evaluated"]]
    return bool(not bad and res["infra_events_present"] and res["n_runs"] == 40)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R2 post-launch checks", allow_abbrev=False)
    ap.add_argument("--root", default=str(cl.DEFAULT_ROOT))
    ap.add_argument("--plan", required=True, help="confirmation_plan.csv (from t3c_launch.py --plan-csv)")
    ap.add_argument("--commit", required=True, help="the launch commit (HEAD at the launch) the manifests must carry")
    ap.add_argument("--lock-dir", default=str(cl.ROOT / "protocols"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    lock = cl.check_lock(Path(a.lock_dir), use_record=Path(a.lock_dir).resolve() == (cl.ROOT / "protocols").resolve())
    res = run_checks(Path(a.root), Path(a.plan), a.commit, a.workers, lock)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1, default=str))
    return 0 if checks_ok(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
