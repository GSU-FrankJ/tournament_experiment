#!/usr/bin/env python3
"""Post-launch checks of the T3-R3 development wave (prompt D5), on the 120 development runs and the 2 diagnostic runs.

Per run (all verdicts are booleans; a deterministic failure is a stop-and-report, exit code 1):

  basics       ``exit_ok`` (status done, exit code 0), ``files_complete`` (the files of a ``bw`` run; for a run that fired also
               ``retry.json`` and ``attempt1/``; for a run that did not fire neither), ``manifest_commit_ok`` / ``clean_tree``,
               ``global_rng_ok``, ``config_equals_plan`` (SHA-256 with the output directory made relative to the root),
               ``config_production_ok`` (the lock and the arm's keys, ``t3r3_launch.config_problems``);
  trigger      ``trigger_consistent``: the trigger recomputed with ``run.ms_retry.trigger`` from the check file it read (the last row of
               ``ms_checks_stage3.csv``, or of ``attempt1/ms_checks_stage3.csv`` for a run that fired) agrees with ``retry.json``;
               ``bw`` has neither ``retry.json`` nor ``attempt1/``;
  budgets      the frozen path has the ``bw`` budget (updates 1600 / 1600 / 600, policy rows 819,200 / 819,200 / 307,200, 76,000
               minibatch steps); a run that fired has in addition 5,400 updates, 2,764,800 policy rows, 108,000 steps and 216 checks in
               total (``retry.json``) and 64 exports, 1600 rows and 64 checks in ``attempt1/``;
  schedules    the learning rate at every update (attempt 1 as well), snapshot refreshes, 152 exports, 64 / 64 / 24 checks;
  C-INIT       the recorded init digest equals the planned one (attempt 1) and, for a run that fired, the attempt-2 digest of
               ``retry.json`` equals the planned one; the attempt-1 digest is equal across the arms of one (q, seed), and the
               attempt-2 digests of ``bwA`` and ``bwAB`` are equal;
  C-REPLAY     ``t3_eval`` reproduces ``freeze_stage1_{final,development}.npz`` bit for bit;
  C-PAIR       ``bwA`` against ``bw`` of the same seed (``t3r3_pair.compare_pair``), the diagnostic runs against T3-R2's original run of
               seed 60518 (binding for both arms); ``bwAB`` against ``bw`` is computed and reported, deciding nothing.

Usage:
    python tools/t3/t3r3_checks.py --root results/t3_r3 --plan reports/t3/r3/dev_plan.csv --commit <launch HEAD> \
        --t3r2-root <results/t3_r2> --out results/t3_r3/dev/launch_checks.json --workers 40
"""

from __future__ import annotations

import argparse
import csv
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
import t3c_launch as cl  # noqa: E402
import t3r3_launch as lt  # noqa: E402
import t3r3_pair as tp  # noqa: E402
from run import ms_retry  # noqa: E402
from t3_launch import relpath_sha256  # noqa: E402

DEFAULT_T3R2 = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-audit-6f3a38/results/t3_r2")
RETRY_FILES = ["retry.json", "attempt1/attempt1_record.json", "attempt1/ms_updates_stage3.csv", "attempt1/train_history_stage3.json",
               "attempt1/state_attempt1_end.pt", "attempt1/ms_checks_stage3.csv", "attempt1/ms_binmaps_stage3.npz"]
VERDICTS = ["exit_ok", "files_complete", "manifest_commit_ok", "clean_tree", "global_rng_ok", "config_equals_plan",
            "config_production_ok", "trigger_consistent", "budget_ok", "lr_ok", "refresh_ok", "exports_ok", "checks_ok",
            "retry_budget_ok", "init_equals_plan", "replay_ok", "pair_ok"]
# fired runs: totals of retry.json (stage 3 trained twice)
FIRED_TOTALS = {"updates": 5400, "transitions": 5400 * 512, "minibatch_steps": 108000, "dev_checks": 216}


def read_plan(path: Path) -> Dict[str, Dict[str, str]]:
    """``{run name: plan row}`` of the plan CSV."""
    return {r["run"]: r for r in lc.read_csv(Path(path))}


def job_of(root: Path, arm: str, q: int, seed: int) -> Dict[str, Any]:
    """A ``t3_eval`` job of one run."""
    kind = lt.kind_of(q, seed)
    return {"wave": f"{lt.WAVE}_{kind}", "arm": arm, "q": q, "seed": seed, "T": 3, "run": lt.run_name(arm, q, seed),
            "run_dir": str(lt.run_dir(root, arm, q, seed)), "kind": "bw", "run_kind": kind}


def all_jobs(root: Path) -> List[Dict[str, Any]]:
    """The 122 jobs in the order of ``t3r3_launch.job_list``."""
    return [job_of(root, r["arm"], r["q"], r["seed"]) for r in lt.job_list()]


def reference_dir(job: Mapping[str, Any], root: Path, t3r2_root: Path) -> Path:
    """The run a candidate is paired with: the ``bw`` run of the same seed (development), T3-R2's original run (diagnostic)."""
    if job["run_kind"] == "diagnostic":
        return Path(t3r2_root) / "confirmation" / f"q{job['q']}" / f"seed{job['seed']}" / "bw"
    return Path(root) / "dev" / f"q{job['q']}" / f"seed{job['seed']}" / "bw"


def check_run(args: Any) -> Dict[str, Any]:
    """All post-launch checks of one run.

    Args:
        args: ``(job, plan_row, commit, lock, root, t3r2_root)``.
    """
    job, plan, commit, lock, root, t3r2_root = args
    torch.set_num_threads(1)
    d = Path(job["run_dir"])
    arm = job["arm"]
    out: Dict[str, Any] = {k: job[k] for k in ("wave", "arm", "q", "seed", "run", "run_kind")}
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
    root_of_run = Path(d).parents[2]
    out["config_equals_plan"] = bool(plan is not None and relpath_sha256(cfg, root_of_run) == plan["run_config_sha256_relpath"])
    out["config_production_ok"] = not lt.config_problems(cfg, lock, arm)
    sched = cfg["record"]["lr_schedule"]
    lc._check_bw(d, cfg, sched, tc.expected_budget("bw"), out)             # the frozen path: budgets, schedules, exports, checks
    # ---- the trigger, recomputed from the check file it read
    check_file = (d / "attempt1" / "ms_checks_stage3.csv") if fired else (d / "ms_checks_stage3.csv")
    rows = lc.read_csv(check_file)
    last = rows[-1]
    chk = {"stage": 3, "local": int(float(last["local"])), "Delta": float(last["Delta"]), "s": float(last["s"]), "C": float(last["C"])}
    b3 = int(cfg["pipeline"]["budgets"]["3"])
    if arm == "bw":
        out["trigger_consistent"] = bool(not fired and not (d / "attempt1").exists())
        out["would_fire"] = bool(ms_retry.trigger(chk, lt.RETRY["delta_gt"], b3)[0])
    else:
        fires, reasons = ms_retry.trigger(chk, cfg["stage3_retry"]["delta_gt"], b3)
        out["would_fire"] = bool(fires)
        out["trigger_consistent"] = bool(fires == fired and (not fired or te.read_json(d / "retry.json")["attempt1"]["reasons"] == reasons))
    out["trigger_Delta"], out["trigger_s"], out["trigger_C"] = chk["Delta"], chk["s"], chk["C"]
    # ---- budgets of a run that fired (retry.json totals and attempt1/)
    if fired:
        rj = te.read_json(d / "retry.json")
        a_rows = lc.read_csv(d / "attempt1" / "ms_updates_stage3.csv")
        wins = {int(t): w for t, w in cfg["pipeline"]["lr_windows"].items()}
        lr_ok = all(float(r["lr"]) == lc.lr_expected(sched, wins[3], int(r["local"])) for r in a_rows)
        names = sorted(p.name for p in (d / "attempt1" / "weights").glob("u*.npz"))
        out["retry_budget_ok"] = bool(
            all(rj["totals"][k] == v for k, v in FIRED_TOTALS.items()) and rj["frozen_attempt"] == 2
            and len(a_rows) == b3 and lr_ok and names == [f"u{u:05d}.npz" for u in range(25, b3 + 1, 25)]
            and len(rows) == 64 and rj["attempt1"]["counts"]["updates"] == b3)
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
    # ---- C-PAIR
    if arm == "bw":
        out["pair_ok"] = True
        out["pair_binding"] = False
    else:
        ref = reference_dir(job, Path(root), Path(t3r2_root))
        pair = tp.compare_pair(ref, d)
        out["pair_ok"] = bool(pair["ok"])
        out["pair_fired"] = bool(pair["fired"])
        out["pair_binding"] = bool(arm == "bwA" or job["run_kind"] == "diagnostic")
        out["pair_detail"] = {k: v for k, v in pair.items() if k not in ("npz",)}
    return out


def cross_arm(res: Sequence[Mapping[str, Any]], root: Path) -> Dict[str, Any]:
    """C-INIT across arms: the attempt-1 digest is equal over the arms of one (q, seed), the attempt-2 digests of ``bwA`` and ``bwAB``
    are equal for the runs that fired in both."""
    by: Dict[Any, Dict[str, Mapping[str, Any]]] = {}
    for r in res:
        by.setdefault((r["run_kind"], r["q"], r["seed"]), {})[r["arm"]] = r
    bad1: List[str] = []
    bad2: List[str] = []
    for key, arms in by.items():
        d1 = {a: r.get("init_state_sha256") for a, r in arms.items()}
        if len(set(d1.values())) > 1:
            bad1.append(str(key))
        if "bwA" in arms and "bwAB" in arms and arms["bwA"].get("fired") and arms["bwAB"].get("fired"):
            a = te.read_json(Path(lt.run_dir(root, "bwA", key[1], key[2])) / "retry.json")["attempt2"]["init_state_sha256"]
            b = te.read_json(Path(lt.run_dir(root, "bwAB", key[1], key[2])) / "retry.json")["attempt2"]["init_state_sha256"]
            if a != b:
                bad2.append(str(key))
    return {"attempt1_digest_differs_across_arms": bad1, "attempt2_digest_differs_bwA_bwAB": bad2, "ok": not bad1 and not bad2}


def run_checks(root: Path, plan_path: Path, commit: Optional[str], workers: int, lock: Mapping[str, Any], t3r2_root: Path) -> Dict[str, Any]:
    """Run every check on the 122 runs; the aggregate (counts per verdict and arm, the ledger, per-run results)."""
    plan = read_plan(plan_path)
    jobs = all_jobs(root)
    args = [(j, plan.get(j["run"]), commit, lock, str(root), str(t3r2_root)) for j in jobs]
    if workers <= 1:
        res = [check_run(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(args)), mp_context=mp.get_context("spawn")) as pool:
            res = list(pool.map(check_run, args))
    agg: Dict[str, Any] = {"n_runs": len(res), "verdicts": {}, "by_arm": {}}
    for v in VERDICTS:
        # ``pair_ok`` of a non-binding comparison (bwAB against bw) is reported, never a failure
        vals = [(r.get(v) if not (v == "pair_ok" and not r.get("pair_binding", True)) else True) for r in res]
        agg["verdicts"][v] = {"pass": sum(1 for x in vals if x is True), "fail": sum(1 for x in vals if x is False),
                              "not_evaluated": sum(1 for x in vals if x is None)}
    for arm in lt.ARMS:
        sub = [r for r in res if r["arm"] == arm]
        agg["by_arm"][arm] = {"runs": len(sub), "fired": sum(1 for r in sub if r.get("fired")),
                              "would_fire": sum(1 for r in sub if r.get("would_fire")),
                              "verdict_failures": {v: sum(1 for r in sub if r.get(v) is False) for v in VERDICTS
                                                   if sum(1 for r in sub if r.get(v) is False)}}
    agg["cross_arm_init"] = cross_arm(res, Path(root))
    ledger_path = Path(root) / "infra_events.json"
    agg["infra_events_present"] = ledger_path.is_file()
    agg["infra_events"] = json.loads(ledger_path.read_text()) if ledger_path.is_file() else None
    agg["runs"] = res
    return agg


def checks_ok(res: Mapping[str, Any]) -> bool:
    """True iff every deterministic check passed on every run, the cross-arm digests agree and the infrastructure ledger exists."""
    bad = [v for v, c in res["verdicts"].items() if c["fail"] or c["not_evaluated"]]
    return bool(not bad and res["infra_events_present"] and res["n_runs"] == 122 and res["cross_arm_init"]["ok"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R3 post-launch checks", allow_abbrev=False)
    ap.add_argument("--root", default=str(lt.DEFAULT_ROOT))
    ap.add_argument("--plan", required=True, help="the plan CSV (from t3r3_launch.py --plan-csv)")
    ap.add_argument("--commit", required=True, help="the launch commit (HEAD at the launch) the manifests must carry")
    ap.add_argument("--t3r2-root", default=str(DEFAULT_T3R2))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if not 1 <= a.workers <= 40:
        ap.error("--workers must be in 1..40")
    lock = cl.check_lock(ROOT / "protocols", use_record=True)
    res = run_checks(Path(a.root), Path(a.plan), a.commit, a.workers, lock, Path(a.t3r2_root))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1, default=str))
    return 0 if checks_ok(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
