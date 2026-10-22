#!/usr/bin/env python3
"""Post-launch checks of the T3-R4 confirmation (prompt 3.2), on the 80 ``bwA`` runs.

Per run (all verdicts are booleans; a deterministic failure is a stop-and-report, exit code 1):

  basics       ``exit_ok`` (status done, exit code 0), ``files_complete`` (the files of a ``bw`` run; for a run that fired also
               ``retry.json`` and a complete ``attempt1/``; for a run that did not fire neither), ``manifest_commit_ok`` (the manifest
               carries the launch commit) and ``clean_tree``, ``global_rng_ok``, ``config_equals_plan`` (SHA-256 with the output directory
               made relative to the root), ``config_production_ok`` (the T3-1.1 lock: ``t3r4_launch.config_problems``);
  trigger      ``trigger_consistent``: the trigger recomputed with ``run.ms_retry.trigger`` from the check file it read (the last row of
               ``ms_checks_stage3.csv``, or of ``attempt1/ms_checks_stage3.csv`` for a run that fired) agrees with the presence and the
               reasons of ``retry.json``;
  budgets      the frozen path has the ``bw`` budget (updates 1600 / 1600 / 600, policy rows 819,200 / 819,200 / 307,200, 76,000 minibatch
               steps); a run that fired has in addition 5,400 updates, 2,764,800 policy rows, 108,000 steps and 216 checks in total
               (``retry.json``) and 64 exports, 1600 rows and 64 checks in ``attempt1/``;
  schedules    the learning rate at every update (attempt 1 as well), snapshot refreshes, 152 exports, 64 / 64 / 24 checks;
  C-INIT       the recorded init digest equals the planned one (attempt 1) and, for a run that fired, the attempt-2 digest of
               ``retry.json`` equals the planned (derived) one;
  C-REPLAY     ``t3_eval`` reproduces ``freeze_stage1_{final,development}.npz`` bit for bit.

The infrastructure ledger ``infra_events.json`` must exist next to the runs, and exactly 80 runs are checked. The exit code is 0 iff every
verdict of every run passes.

Usage:
    python tools/t3/t3r4_checks.py --root $PWD/results/t3_r4/confirmation --plan reports/t3/r4/confirmation_plan.csv \
        --commit <launch HEAD> --out results/t3_r4/confirmation/launch_checks.json --workers 40
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import t3_configs as tc  # noqa: E402
import t3_eval as te  # noqa: E402
import t3_launch_checks as lc  # noqa: E402
import t3r3_checks as r3ck  # noqa: E402
import t3r4_launch as lt  # noqa: E402
from run import ms_retry  # noqa: E402
from t3_launch import relpath_sha256  # noqa: E402

RETRY_FILES = r3ck.RETRY_FILES
FIRED_TOTALS = r3ck.FIRED_TOTALS
VERDICTS = ["exit_ok", "files_complete", "manifest_commit_ok", "clean_tree", "global_rng_ok", "config_equals_plan",
            "config_production_ok", "trigger_consistent", "budget_ok", "lr_ok", "refresh_ok", "exports_ok", "checks_ok",
            "retry_budget_ok", "init_equals_plan", "replay_ok"]
N_RUNS = len(lt.QS) * len(lt.SEEDS)


def read_plan(path: Path) -> Dict[str, Dict[str, str]]:
    """``{run name: plan row}`` of the plan CSV."""
    return {r["run"]: r for r in lc.read_csv(Path(path))}


def job_of(root: Path, q: int, seed: int) -> Dict[str, Any]:
    """A ``t3_eval`` job of one run."""
    return {"wave": lt.WAVE, "arm": lt.ARM, "q": q, "seed": seed, "T": 3, "run": lt.run_name(q, seed),
            "run_dir": str(lt.run_dir(root, q, seed)), "kind": "bw", "run_kind": "confirmation"}


def all_jobs(root: Path) -> List[Dict[str, Any]]:
    """The 80 jobs, q-major then seed (the order of the plan)."""
    return [job_of(root, q, s) for q in lt.QS for s in lt.SEEDS]


def check_run(args: Any) -> Dict[str, Any]:
    """All post-launch checks of one run; an exception inside the checks of a run is recorded as ``error`` and leaves the verdicts it
    did not reach unevaluated (which fails :func:`checks_ok`), so it cannot abort the checks of the other runs.

    Args:
        args: ``(job, plan_row, commit, lock, root)``.
    """
    job = args[0]
    try:
        return _check_run(args)
    except Exception as exc:  # noqa: BLE001 - recorded
        return {**{k: job[k] for k in ("arm", "q", "seed", "run", "run_kind")}, "error": f"{type(exc).__name__}: {exc}"}


def _check_run(args: Any) -> Dict[str, Any]:
    job, plan, commit, lock, root = args
    import torch

    torch.set_num_threads(1)
    d = Path(job["run_dir"])
    out: Dict[str, Any] = {k: job[k] for k in ("arm", "q", "seed", "run", "run_kind")}
    st = te.status_of(d)
    out["exit_ok"] = bool(st.get("state") == "done" and st.get("exit_code") == 0)
    fired = (d / "retry.json").is_file()
    out["fired"] = fired
    need = list(lc.REQUIRED_FILES["bw"]) + (RETRY_FILES if fired else [])
    missing = [f for f in need if not (d / f).exists()]
    extra = [] if fired else (["attempt1"] if (d / "attempt1").exists() else [])      # attempt1/ without retry.json
    out["files_complete"] = not missing and not extra
    out["files_missing"] = ";".join(missing + extra)
    if not out["exit_ok"] or missing:
        return out
    cfg = te.read_json(d / "run_config.json")
    man = te.read_json(d / "manifest.json")
    out["manifest_commit_ok"] = bool(commit is None or man["commit"] == commit)
    out["clean_tree"] = bool(man["clean_tree"] is True)
    out["global_rng_ok"] = bool(te.global_rng_ok(d, "bw"))
    out["config_equals_plan"] = bool(plan is not None and relpath_sha256(cfg, Path(root)) == plan["run_config_sha256_relpath"])
    out["config_production_ok"] = not lt.config_problems(cfg, lock)
    sched = cfg["record"]["lr_schedule"]
    lc._check_bw(d, cfg, sched, tc.expected_budget("bw"), out)             # budgets, schedules, exports, checks of the frozen path
    # ---- the trigger, recomputed from the check file it read
    check_file = (d / "attempt1" / "ms_checks_stage3.csv") if fired else (d / "ms_checks_stage3.csv")
    last = lc.read_csv(check_file)[-1]
    chk = {"stage": 3, "local": int(float(last["local"])), "Delta": float(last["Delta"]), "s": float(last["s"]), "C": float(last["C"])}
    b3 = int(cfg["pipeline"]["budgets"]["3"])
    fires, reasons = ms_retry.trigger(chk, cfg["stage3_retry"]["delta_gt"], b3)
    out["would_fire"] = bool(fires)
    consistent = bool(fires == fired)
    if fired:                                                       # retry.json agrees with the check files of both attempts
        rj = te.read_json(d / "retry.json")
        t1, f2 = rj["attempt1"]["trigger_check"], rj["attempt2"]["final_check"]
        last2 = lc.read_csv(d / "ms_checks_stage3.csv")[-1]
        consistent = bool(consistent and rj["attempt1"]["reasons"] == reasons
                          and all(float(t1[k]) == chk[k] for k in ("Delta", "s", "C"))
                          and all(float(f2[k]) == float(last2[k]) for k in ("Delta", "s", "C")))
    out["trigger_consistent"] = consistent
    out["trigger_Delta"], out["trigger_s"], out["trigger_C"] = chk["Delta"], chk["s"], chk["C"]
    # ---- budgets of a run that fired (retry.json totals and attempt1/)
    if fired:
        rj = te.read_json(d / "retry.json")
        a_rows = lc.read_csv(d / "attempt1" / "ms_updates_stage3.csv")
        wins = {int(t): w for t, w in cfg["pipeline"]["lr_windows"].items()}
        lr_ok = all(float(r["lr"]) == lc.lr_expected(sched, wins[3], int(r["local"])) for r in a_rows)
        names = sorted(p.name for p in (d / "attempt1" / "weights").glob("u*.npz"))
        n_check_rows = len(lc.read_csv(d / "attempt1" / "ms_checks_stage3.csv"))
        out["retry_budget_ok"] = bool(
            all(rj["totals"][k] == v for k, v in FIRED_TOTALS.items()) and rj["frozen_attempt"] == 2
            and len(a_rows) == b3 and lr_ok and names == [f"u{u:05d}.npz" for u in range(25, b3 + 1, 25)]
            and n_check_rows == 64 and rj["attempt1"]["counts"]["updates"] == b3)
        out["total_updates"], out["total_transitions"] = rj["totals"]["updates"], rj["totals"]["transitions"]
    else:
        out["retry_budget_ok"] = True
        out["total_updates"], out["total_transitions"] = 3800, 3800 * 512
    # ---- C-INIT
    ok_init = bool(plan is not None and man["init_state_sha256"] == plan["init_state_sha256"])
    if fired:
        ok_init = bool(ok_init and plan["attempt2_init_state_sha256"] != "" and
                       te.read_json(d / "retry.json")["attempt2"]["init_state_sha256"] == plan["attempt2_init_state_sha256"])
    out["init_equals_plan"] = ok_init
    out["init_state_sha256"] = man["init_state_sha256"]
    # ---- C-REPLAY
    out["replay_ok"] = not te.replay_mismatches(d, "bw")
    return out


def checks_ok(res: Mapping[str, Any]) -> bool:
    """True iff every deterministic check passed on every run, the infrastructure ledger exists and exactly 80 runs were checked."""
    bad = [v for v, c in res["verdicts"].items() if c["fail"] or c["not_evaluated"] or c["pass"] != res["n_runs"]]
    return bool(not bad and res["infra_events_present"] and res["n_runs"] == N_RUNS)


def run_checks(root: Path, plan_path: Path, commit: Optional[str], workers: int, lock: Mapping[str, Any]) -> Dict[str, Any]:
    """Run every check on the 80 runs; the aggregate (counts per verdict, the ledger, per-run results)."""
    plan = read_plan(plan_path)
    jobs = all_jobs(root)
    args = [(j, plan.get(j["run"]), commit, lock, str(root)) for j in jobs]
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
    agg["fired"] = sum(1 for r in res if r.get("fired"))
    agg["would_fire"] = sum(1 for r in res if r.get("would_fire"))
    agg["errors"] = [{"run": r["run"], "error": r["error"]} for r in res if "error" in r]
    ledger_path = Path(root) / "infra_events.json"
    agg["infra_events_present"] = ledger_path.is_file()
    agg["infra_events"] = json.loads(ledger_path.read_text()) if ledger_path.is_file() else None
    agg["runs"] = res
    return agg


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R4 post-launch checks", allow_abbrev=False)
    ap.add_argument("--root", default=str(lt.DEFAULT_ROOT), help="results root, an absolute path")
    ap.add_argument("--plan", required=True, help="the plan CSV (from t3r4_launch.py --plan-csv)")
    ap.add_argument("--commit", required=True, help="the launch commit (HEAD at the launch) the manifests must carry")
    ap.add_argument("--lock-dir", default=str(ROOT / "protocols"), help="directory of the T3-1.1 lock (default protocols/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if not 1 <= a.workers <= 40:
        ap.error("--workers must be in 1..40")
    root = lt.require_absolute(ap, "--root", a.root)
    lock = lt.check_lock(Path(a.lock_dir), use_record=Path(a.lock_dir).resolve() == (ROOT / "protocols").resolve())
    res = run_checks(root, Path(a.plan), a.commit, a.workers, lock)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1, default=str))
    return 0 if checks_ok(res) else 1


if __name__ == "__main__":
    sys.exit(main())
