#!/usr/bin/env python3
"""Checks of round T3-R1: the pre-launch checks C-INIT (plan) and C-T2R (prompt section 2.3) and the post-launch checks
of section 3.2 (also run on the smoke wave).

  c-init-plan  Construct every planned run without training and record its init digest: within each (q, seed) of the
               waves ``smoke`` and ``t3`` the digests of ``bw`` (``MSRun``), ``jt`` and ``jt_es`` must be equal; for
               each (q, seed) of 30501-30520 the digest of ``jt2`` must equal that of ``MSRun`` at T = 2 (``MS_base`` with
               ``reports/ms/r1/prereg_parameters.json`` and ``init_digest`` true).
  c-t2r        ``tools/t3/t3_eval.py`` on the 40 ``v20`` runs must reproduce each run's ``gates.json`` bit for bit: the
               values ``eta_final``, ``eta_dev``, ``rmse``, ``tail``, ``gmax_final``, ``gmax_dev``, ``s1`` and the
               verdicts G-A, G-F, G-N, G-S.
  post-launch  Basics (exit codes, manifests at the launch commit with a clean tree, complete files, global-RNG
               assertions, the config equal to the plan), budgets, schedules (learning rate, snapshot refreshes,
               exports, checks), starts (``jt_es`` 256 / 85 / 171 direct starts per update; the descriptive z flags),
               C-INIT (recorded digests equal within a (q, seed) and to the planned ones) and C-REPLAY (the evaluation
               of ``t3_eval`` reproduces each run's own final-evaluation arrays bit for bit).

Wall-clock columns are excluded from every comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import t3_configs as tc  # noqa: E402
import t3_eval as te  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402
from run import run_joint as rj  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402
from run.run_final_dp_br_round3_dense import lr_at  # noqa: E402

Z_LIMIT = 3.0
NULL_RATE = 0.0027                 # two-sided P(|z| > 3) under the normal approximation
REQUIRED_FILES = {
    "bw": ["run_config.json", "manifest.json", "status.json", "gates.json", "rule_log.json", "ms_updates.csv",
           "ms_run_summary.json", "drift_test.json", "ms_checks_stage3.csv", "ms_checks_stage2.csv",
           "ms_checks_stage1.csv", "state_end_stage3.pt", "state_end_stage2.pt", "state_end_stage1.pt",
           "continuation_table_stage2.npz", "continuation_table_stage1.npz", "freeze_stage3_final.npz",
           "freeze_stage3_development.npz", "freeze_stage2_final.npz", "freeze_stage2_development.npz",
           "freeze_stage1_final.npz", "freeze_stage1_development.npz"],
    "joint": ["run_config.json", "manifest.json", "status.json", "joint_run_summary.json", "joint_checks.csv",
              "joint_updates.csv", "visitation.npz", "state_end.pt", "final_eval_final.npz",
              "final_eval_development.npz"]}


# ---------------------------------------------------------------------------------- small readers
def read_csv(path: Path) -> List[Dict[str, str]]:
    """Rows of a CSV as dicts of strings."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def kind_of(arm: str) -> str:
    """Run format of an arm: ``bw`` or ``joint``."""
    return "bw" if arm == "bw" else "joint"


def lr_expected(sched: Mapping[str, Any], windows: Sequence[Mapping[str, Any]], j: int) -> float:
    """Learning rate before (local) update ``j``: linear inside a window, ``ab_lr`` outside (the form of
    ``MSRun.lr_linear`` / ``_legacy_lr``)."""
    for w in windows:
        if int(w["first"]) <= j <= int(w["last"]):
            lin = dict(sched, kind="linear", c_start_lr=float(w["start"]), c_end_lr=float(w["end"]),
                       c_local_first=int(w["first"]), linear_denominator=int(w["last"]) - int(w["first"]))
            return lr_at(lin, "C", j)
    return lr_at(sched, "A", j)


def z_flags(counts: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    """Cell-wise z = (n - N p) / sqrt(N p (1 - p)) of a multinomial total against probabilities ``p`` (cells with
    p = 0 or 1 are skipped); the number of cells and of |z| > 3."""
    N = float(np.sum(counts))
    ok = (p > 0.0) & (p < 1.0)
    z = (np.asarray(counts, float)[ok] - N * p[ok]) / np.sqrt(N * p[ok] * (1.0 - p[ok]))
    return {"cells": int(ok.sum()), "flags": int((np.abs(z) > Z_LIMIT).sum()), "max_abs_z": float(np.abs(z).max()) if z.size else float("nan")}


# ---------------------------------------------------------------------------------- one run, post launch
def check_run(job: Mapping[str, Any], expected: Mapping[str, Any], commit: Optional[str]) -> Dict[str, Any]:
    """All post-launch checks of one run (a job of ``t3_eval.plan_jobs``).

    Args:
        job: ``wave``, ``arm``, ``q``, ``seed``, ``run``, ``run_dir``.
        expected: ``t3_configs.expected_budget(arm)`` (updates, rows by stage, minibatch steps, total rows).
        commit: The launch commit the manifests must carry (``None``: not checked).

    Returns:
        A flat dict of verdicts (booleans) and details.
    """
    torch.set_num_threads(1)
    d = Path(job["run_dir"])
    arm, kind = job["arm"], kind_of(job["arm"])
    out: Dict[str, Any] = {k: job[k] for k in ("wave", "arm", "q", "seed", "run")}
    st = te.status_of(d)
    out["exit_ok"] = bool(st.get("state") == "done" and st.get("exit_code") == 0)
    missing = [f for f in REQUIRED_FILES[kind] if not (d / f).exists()]
    out["files_complete"] = not missing
    out["files_missing"] = ";".join(missing)
    if not out["exit_ok"] or missing:
        return out
    cfg = te.read_json(d / "run_config.json")
    man = te.read_json(d / "manifest.json")
    out["manifest_commit_ok"] = bool(commit is None or man["commit"] == commit)
    out["clean_tree"] = bool(man["clean_tree"] is True)
    out["init_state_sha256"] = man["init_state_sha256"]
    out["global_rng_ok"] = bool(te.global_rng_ok(d, kind))
    plan_sha = tc.config_sha256(tc.build_config(tc.load_protocol(), int(job["q"]), int(job["seed"]), arm,
                                                cfg["record"]["output_dir"], tc.load_bw_params()))
    with open(d / "run_config.json", "rb") as f:
        out["config_equals_plan"] = bool(hashlib.sha256(f.read()).hexdigest() == plan_sha)
    out["config_production_ok"] = not tc.production_problems(cfg)
    T = int(cfg["record"]["game"]["T"])
    sched = cfg["record"]["lr_schedule"]
    if kind == "bw":
        _check_bw(d, cfg, sched, expected, out)
    else:
        _check_joint(d, cfg, sched, expected, T, out)
    out["replay_ok"] = not te.replay_mismatches(d, kind)
    return out


def _check_bw(d: Path, cfg: Mapping[str, Any], sched: Mapping[str, Any], expected: Mapping[str, Any],
              out: Dict[str, Any]) -> None:
    rows = read_csv(d / "ms_updates.csv")
    summ = te.read_json(d / "ms_run_summary.json")
    gates = te.read_json(d / "gates.json")
    n_ep = int(cfg["record"]["protocol"]["episodes_per_update"])
    by_stage = {t: sum(1 for r in rows if int(r["stage"]) == t) for t in (1, 2, 3)}
    rows_by_stage = {t: n * n_ep for t, n in by_stage.items()}
    out["budget_ok"] = bool(
        summ["final_global_update"] == expected["updates"] == len(rows)
        and rows_by_stage == dict(expected["rows_by_stage"])
        and summ["total_episodes"] == summ["total_transitions"] == expected["rows_total"]
        and gates["budget_totals"]["minibatch_steps"] == expected["minibatch_steps"])
    wins = {int(t): w for t, w in cfg["pipeline"]["lr_windows"].items()}
    ok = True
    for r in rows:
        if float(r["lr"]) != lr_expected(sched, wins[int(r["stage"])], int(r["local"])):
            ok = False
            break
    out["lr_ok"] = ok
    st1 = torch.load(d / "state_end_stage1.pt", map_location="cpu", weights_only=False)
    log = st1["snapshot_log"]
    every = [e["update"] for e in log if e["reason"] == "every_20"]
    entries = {e["reason"]: e["update"] for e in log if e["reason"].startswith("phase_stage")}
    b3, b2 = int(cfg["pipeline"]["budgets"]["3"]), int(cfg["pipeline"]["budgets"]["2"])
    out["refresh_ok"] = bool(every == list(range(20, expected["updates"] + 1, 20))
                             and entries == {"phase_stage3_entry": 0, "phase_stage2_entry": b3, "phase_stage1_entry": b3 + b2})
    names = sorted(p.name for p in (d / "weights").glob("u*.npz"))
    out["exports_ok"] = bool(names == [f"u{u:05d}.npz" for u in range(25, expected["updates"] + 1, 25)])
    out["n_exports"] = len(names)
    counts = {t: len(read_csv(d / f"ms_checks_stage{t}.csv")) for t in (3, 2, 1)}
    out["checks_ok"] = bool(counts == {3: 64, 2: 64, 1: 24})
    out["n_checks"] = "/".join(str(counts[t]) for t in (3, 2, 1))
    spec = GameSpec(**cfg["record"]["game"])
    smp = StartSampler(spec, float(cfg["record"]["protocol"]["es_bin_width"]))
    flags = cells = 0
    for t in (3, 2):
        n_bins = smp.n_bins(t)
        sh = smp.strata(t, rms.REPORT_NEAR_TIE_HALF_WIDTH)
        p = np.array([sh[k].sum() / n_bins for k in ("tail", "near", "mid")])
        tot = np.array([sum(int(r[f"n_start_{k}"]) for r in rows if int(r["stage"]) == t) for k in ("tail", "near", "mid")])
        zf = z_flags(tot, p)
        flags, cells = flags + zf["flags"], cells + zf["cells"]
    out["starts_ok"] = True      # bw: the starts are bin-balanced by the pinned config; only the descriptive z flags are reported
    out["z_flags"], out["z_cells"] = flags, cells


def _check_joint(d: Path, cfg: Mapping[str, Any], sched: Mapping[str, Any], expected: Mapping[str, Any], T: int,
                 out: Dict[str, Any]) -> None:
    summ = te.read_json(d / "joint_run_summary.json")
    rows = read_csv(d / "joint_updates.csv")
    j = cfg["joint"]
    U = int(j["updates"])
    n_ep = int(cfg["record"]["protocol"]["episodes_per_update"])
    per_update = {int(t): v for t, v in expected["rows_per_update"].items()}
    rows_ok = all([int(r[f"rows_t{t}"]) for t in range(1, T + 1)] == [per_update[t] for t in range(1, T + 1)] for r in rows)
    out["budget_ok"] = bool(
        len(rows) == U == expected["updates"] and rows_ok and summ["updates"] == U
        and summ["episodes"] == U * n_ep
        and {int(t): v for t, v in summ["policy_rows_by_stage"].items()} == dict(expected["rows_by_stage"])
        and summ["policy_rows_total"] == expected["rows_total"] and summ["minibatch_steps"] == expected["minibatch_steps"])
    out["lr_ok"] = all(float(r["lr"]) == lr_expected(sched, j["lr_windows"], int(r["update"])) for r in rows)
    st = torch.load(d / "state_end.pt", map_location="cpu", weights_only=False)
    log = st["snapshot_log"]
    every = [e["update"] for e in log if e["reason"] == "every_20"]
    csv_ok = all((r["snapshot_refreshed"] == "True") == (int(r["update"]) % 20 == 0) for r in rows)
    out["refresh_ok"] = bool(log[0] == {"update": 0, "reason": "init"} and log[1] == {"update": 0, "reason": "entry"}
                             and every == list(range(20, U + 1, 20)) and csv_ok)
    names = sorted(p.name for p in (d / "weights").glob("u*.npz"))
    out["exports_ok"] = bool(names == [f"u{u:05d}.npz" for u in range(25, U + 1, 25)])
    out["n_exports"] = len(names)
    checks = read_csv(d / "joint_checks.csv")
    out["checks_ok"] = bool([int(c["update"]) for c in checks] == list(range(25, U + 1, 25)))
    out["n_checks"] = str(len(checks))
    st_cfg = j["starts"]
    starts_ok = True
    if cfg["arm"] == "jt_es":
        starts_ok = bool(st_cfg == {"root": 256, "stage2": 85, "stage3": 171} and all(
            int(r["rows_t2"]) - int(r["rows_t1"]) == 85 and int(r["rows_t3"]) - int(r["rows_t2"]) == 171
            and int(r["rows_t1"]) == 256 for r in rows))
    out["starts_ok"] = starts_ok
    flags = cells = 0
    if cfg["arm"] == "jt_es":
        v = np.load(d / "visitation.npz")
        for t in (2, 3):
            cnt = v[f"stage{t}_direct_es"]
            zf = z_flags(cnt, np.full(cnt.size, 1.0 / cnt.size))
            flags, cells = flags + zf["flags"], cells + zf["cells"]
            starts_ok = starts_ok and int(cnt.sum()) == {2: 85, 3: 171}[t] * U
        out["starts_ok"] = bool(starts_ok)
    out["z_flags"], out["z_cells"] = flags, cells


# ---------------------------------------------------------------------------------- aggregation
VERDICTS = ["exit_ok", "files_complete", "manifest_commit_ok", "clean_tree", "global_rng_ok", "config_equals_plan",
            "config_production_ok", "budget_ok", "lr_ok", "refresh_ok", "exports_ok", "checks_ok", "starts_ok",
            "replay_ok"]


def _worker(args: Any) -> Dict[str, Any]:
    return check_run(*args)


def post_launch(jobs: Sequence[Mapping[str, Any]], commit: Optional[str], workers: int,
                plan_digests: Optional[Mapping[Any, str]] = None) -> Dict[str, Any]:
    """Run every post-launch check on the planned runs; the aggregate (counts per verdict, C-INIT, z flags)."""
    args = [(j, tc.expected_budget(j["arm"]), commit) for j in jobs]
    if workers <= 1 or len(args) <= 1:
        res = [_worker(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(args)), mp_context=mp.get_context("spawn")) as pool:
            res = list(pool.map(_worker, args))
    agg: Dict[str, Any] = {"n_runs": len(res), "verdicts": {}}
    for v in VERDICTS:
        vals = [r.get(v) for r in res]
        agg["verdicts"][v] = {"pass": sum(1 for x in vals if x is True), "fail": sum(1 for x in vals if x is False),
                              "not_evaluated": sum(1 for x in vals if x is None)}
    # C-INIT: recorded digests equal within (q, seed) across the arms, and equal to the planned digests
    by_key: Dict[Any, set] = {}
    for r in res:
        if r.get("init_state_sha256") and r["wave"] != "t2j":
            by_key.setdefault((r["q"], r["seed"]), set()).add(r["init_state_sha256"])
    agg["c_init_within_q_seed"] = {"groups": len(by_key), "equal": sum(1 for s in by_key.values() if len(s) == 1)}
    if plan_digests is not None:
        eq = sum(1 for r in res if r.get("init_state_sha256") == plan_digests.get((r["arm"], r["q"], r["seed"])))
        agg["c_init_equals_plan"] = {"runs": len(res), "equal": eq}
    cells = sum(int(r.get("z_cells", 0)) for r in res)
    flags = sum(int(r.get("z_flags", 0)) for r in res)
    agg["start_share_flags"] = {"cells": cells, "flags": flags, "expected_under_null": NULL_RATE * cells}
    agg["runs"] = res
    return agg


def post_launch_ok(res: Mapping[str, Any]) -> bool:
    """True iff every deterministic check passed: no failed or unevaluated verdict, C-INIT equal within every
    (q, seed) group and (when the plan was given) equal to the plan in every run. The z flags are descriptive."""
    bad = [v for v, c in res["verdicts"].items() if c["fail"] or c["not_evaluated"]]
    ci = res["c_init_within_q_seed"]
    ok = not bad and ci["equal"] == ci["groups"]
    if "c_init_equals_plan" in res:
        ok = ok and res["c_init_equals_plan"]["equal"] == res["c_init_equals_plan"]["runs"]
    return bool(ok)


# ---------------------------------------------------------------------------------- C-INIT (plan)
def c_init_plan(waves: Sequence[str], out_csv: Path, ms_params: Dict[str, Any]) -> Dict[str, Any]:
    """Construct every planned run without training; record the digests; compare as the prompt demands."""
    proto = tc.load_protocol()
    params = tc.load_bw_params()
    rows: List[Dict[str, Any]] = []
    scratch = str(out_csv.parent / "_scratch_not_written")
    for wave in waves:
        for q in tc.QS:
            for seed in tc.WAVE_SEEDS[wave]:
                for arm in tc.WAVE_ARMS[wave]:
                    cfg = tc.build_config(proto, q, seed, arm, scratch, params)
                    run = (rms.MSRun if arm == "bw" else rj.JointRun)(cfg, scratch)
                    rows.append({"wave": wave, "arm": arm, "q": q, "seed": seed,
                                 "init_state_sha256": run.init_state_sha256})
                if "jt2" in tc.WAVE_ARMS[wave]:
                    cfg = rms_t2_reference(proto, q, seed, scratch, ms_params)
                    rows.append({"wave": wave, "arm": "ms_t2_reference", "q": q, "seed": seed,
                                 "init_state_sha256": rms.MSRun(cfg, scratch).init_state_sha256})
    with open(out_csv, "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["wave", "arm", "q", "seed", "init_state_sha256"])
        w.writeheader()
        w.writerows(rows)
    by: Dict[Any, Dict[str, str]] = {}
    for r in rows:
        by.setdefault((r["q"], r["seed"]), {})[r["arm"]] = r["init_state_sha256"]
    t3_ok = [len({d[a] for a in ("bw", "jt", "jt_es")}) == 1 for k, d in by.items() if "bw" in d]
    t2_ok = [d["jt2"] == d["ms_t2_reference"] for k, d in by.items() if "jt2" in d]
    return {"rows": len(rows), "t3_groups": len(t3_ok), "t3_equal": sum(t3_ok), "t2_groups": len(t2_ok),
            "t2_equal": sum(t2_ok)}


def rms_t2_reference(proto: Dict[str, Any], q: int, seed: int, out_dir: str, ms_params: Dict[str, Any]) -> Dict[str, Any]:
    """``MS_base`` at T = 2 with the MS-R1 parameters and ``init_digest`` true (the reference of section 2.2)."""
    cfg = tc.mc.build_config(proto, q, seed, "MS_base", out_dir, ms_params)
    cfg["init_digest"] = True
    return cfg


# ---------------------------------------------------------------------------------- C-T2R
GATE_KEYS = {"eta_final": "eta_T_over_dw", "eta_dev": "dev_eta_T_over_dw", "rmse": "rmse_pos_T_over_eT0",
             "tail": "tail_mean_T_over_eT0", "gmax_final": "Gmax_full_over_dw", "gmax_dev": "dev_Gmax_full_over_dw",
             "s1": "stage1_rel_err_abs"}
VERDICT_KEYS = {"G-A": "gate_G_A", "G-F": "gate_G_F", "G-N": "gate_G_N", "G-S": "gate_G_S"}


def compare_to_gates(row: Mapping[str, Any], gates: Mapping[str, Any]) -> Dict[str, bool]:
    """Equality (exact) of the seven gate values and the four verdicts of one re-evaluated v20 run with its gates.json."""
    out = {k: row.get(col) is not None and float(row[col]) == float(gates["metric_values"][k])
           for k, col in GATE_KEYS.items()}
    for g, col in VERDICT_KEYS.items():
        out[g] = row.get(col) is not None and bool(row[col]) == bool(gates[g]["pass"])
    return out


def c_t2r(v20_root: Path, out_dir: Path, workers: int) -> Dict[str, Any]:
    """Re-evaluate the 40 v20 runs and compare with their gates.json."""
    jobs = te.plan_jobs(Path("."), [], v20_root, None)
    rows = te.run_parallel(te.evaluate_run, jobs, workers)
    te.write_rows(out_dir / "c_t2r_per_run.csv", rows)
    cmp_rows = []
    for r in rows:
        g = te.read_json(Path(r["run_dir"]) / "gates.json")
        c = compare_to_gates(r, g)
        cmp_rows.append({"q": r["q"], "seed": r["seed"], **c, "all_identical": all(c.values())})
    with open(out_dir / "c_t2r.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cmp_rows[0]))
        w.writeheader()
        w.writerows(cmp_rows)
    return {"runs": len(cmp_rows), "identical": sum(1 for c in cmp_rows if c["all_identical"]),
            "per_field_identical": {k: sum(1 for c in cmp_rows if c[k]) for k in list(GATE_KEYS) + list(VERDICT_KEYS)}}


# ---------------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R1 checks", allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("c-init-plan")
    a1.add_argument("--out", required=True)
    a1.add_argument("--waves", nargs="+", default=["smoke", "t3", "t2j"])
    a2 = sub.add_parser("c-t2r")
    a2.add_argument("--v20-root", required=True)
    a2.add_argument("--out-dir", required=True)
    a2.add_argument("--workers", type=int, default=8)
    a3 = sub.add_parser("post-launch")
    a3.add_argument("--root", default=str(tc.DEFAULT_ROOT))
    a3.add_argument("--waves", nargs="+", default=["t3", "t2j"])
    a3.add_argument("--commit", required=True, help="the launch commit (HEAD at the launch) the manifests must carry")
    a3.add_argument("--plan-digests", default=None, help="c_init_plan.csv of the pre-launch check")
    a3.add_argument("--out", required=True)
    a3.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd == "c-init-plan":
        res = c_init_plan(a.waves, Path(a.out), json.load(open(ROOT / tc.MS_PARAMS_REL)))
        print(json.dumps(res, indent=1))
        return 0 if res["t3_equal"] == res["t3_groups"] and res["t2_equal"] == res["t2_groups"] else 1
    if a.cmd == "c-t2r":
        out_dir = Path(a.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        res = c_t2r(Path(a.v20_root), out_dir, a.workers)
        print(json.dumps(res, indent=1))
        return 0 if res["identical"] == res["runs"] else 1
    pd = None
    if a.plan_digests:
        pd = {(r["arm"], int(r["q"]), int(r["seed"])): r["init_state_sha256"] for r in read_csv(Path(a.plan_digests))}
    jobs = te.plan_jobs(Path(a.root), a.waves, None, None)
    res = post_launch(jobs, a.commit, a.workers, pd)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1))
    return 0 if post_launch_ok(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
