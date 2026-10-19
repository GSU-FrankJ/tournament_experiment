#!/usr/bin/env python3
"""Audit tool of round T3-R2 (prompt D3): A1-A3 and A5-A8. Pure: it consumes no random number and changes no file
outside ``--out-dir`` (A1 additionally trains the four re-runs into ``results/t3_r2/reruns/``).

Subcommands (every one reads the T3-R1 untracked root ``--t3r1-root`` and, for the v2.0 runs, ``--v20-root``; both are
recorded in the outputs):

  a1-rerun      re-run the four failing ``bw`` runs from their own ``run_config.json`` (unchanged), at most 4 workers
  a1-compare    each re-run against its original: ``tools/t3/compare_runs.py`` and ``t3_eval.evaluate_run`` against the row
                of ``per_run.csv`` (text equality of every metric column)
  a2            snapshot identity for the 40 ``bw`` runs (``state_end_stage1.pt`` against the exports)
  a3            continuation tables of 8 ``bw`` runs: rebuilt bit for bit, and compared with the calculator
  a5            the independent calculator applied to the four failure points, the 200 runs (G, Delta at the stored
                states, Delta_T, EXP_root), the five verifier-agreement runs and the convergence test
  a6            the refined verifier tier (state step 1, effort step 0.25, 64 nodes per half) for the 200 runs
  a7            the near-zero stretches of the seven runs of ``zero_effort.csv`` and the 33 comparison runs
  a8            final-tier eta_3 and the shape parts of G-A at the last 16 stage-3 exports of the 40 ``bw`` runs

The pure analysis cores (:func:`analyze_nodes`, :func:`stretch_nodes`, :func:`first_below`) take callables, so the tests
run them on synthetic policies and synthetic exports.
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import t3_configs as tc  # noqa: E402
import t3_eval as te  # noqa: E402
import t3_metrics as tm  # noqa: E402
import t3r2_br_calc as bc  # noqa: E402
from replay_dev_rule import candidate_fns  # noqa: E402
from utils.dp_br_verifier import FINAL_CONFIG, VerifierConfig  # noqa: E402
from utils.ms_continuation import build_nested_tables  # noqa: E402

PY = sys.executable
DEFAULT_T3R1 = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-vs-jt-51e472/results/t3_r1")
DEFAULT_V20 = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/"
                   "confirmation_v2_0")
DEFAULT_PER_RUN = ROOT / "results" / "t3_r1" / "eval" / "per_run.csv"
DEFAULT_OUT = ROOT / "results" / "t3_r2" / "audit"
REFINED = VerifierConfig("refined", state_step=1.0, effort_step=0.25, gl_half=64)
FAILURES = ((50, 50504), (50, 50517), (60, 50514), (60, 50517))
A3_RUNS = FAILURES + ((50, 50501), (50, 50502), (60, 50501), (60, 50502))
AGREE_RUNS = (("bw", 50, 50501), ("bw", 60, 50501), ("jt", 50, 50501), ("jt_es", 60, 50501), ("v20", 50, 30501))
STAGE2_EXPORTS = tuple(range(1625, 3201, 25))
LAST16 = tuple(range(1225, 1601, 25))
STRETCH_EFFORT = 1.0          # a node is "zero effort" when the learned mean effort is below 1 (the effort range is 100)
BAND_TOL = 1e-9
FLAG_TOL_OVER_DW = 1e-5
GN_TOL = 1e-3


# ---------------------------------------------------------------------------------- small helpers
def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Rows of a CSV as dicts of strings."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Dict rows to a CSV (union of the keys, first-seen order)."""
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    """JSON file with indent 1."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def game_of(spec: Any) -> bc.Game:
    """Calculator ``Game`` of a ``GameSpec``."""
    return bc.Game(float(spec.w_h), float(spec.w_l), float(spec.k), float(spec.q), int(spec.T), float(spec.e_min),
                   float(spec.e_max))


def audit_jobs(t3r1_root: Path, v20_root: Path) -> List[Dict[str, Any]]:
    """The 200 runs of ``per_run.csv`` (120 T = 3 runs, 40 ``jt2``, 40 ``v20``) as ``t3_eval`` jobs."""
    return te.plan_jobs(Path(t3r1_root), ["t3", "t2j"], Path(v20_root), None)


def job_of(jobs: Sequence[Mapping[str, Any]], arm: str, q: int, seed: int) -> Dict[str, Any]:
    """The job of one run."""
    for j in jobs:
        if j["arm"] == arm and int(j["q"]) == int(q) and int(j["seed"]) == int(seed):
            return dict(j)
    raise KeyError((arm, q, seed))


def load_run(job: Mapping[str, Any]) -> Tuple[Dict[str, Any], te.Candidate, bc.Game, Callable[[int, np.ndarray], np.ndarray]]:
    """``(config, candidate, game, mean_fn)`` of a run's last iterate."""
    d = Path(job["run_dir"])
    cfg = te.read_json(d / "run_config.json")
    cand = te.load_candidate(job["kind"], d, cfg)
    mean_fn, _ = candidate_fns(cand.spec, cand.nets)
    return cfg, cand, game_of(cand.spec), mean_fn


def per_run_table(path: Path) -> Dict[Tuple[str, int, int], Dict[str, str]]:
    """``{(arm, q, seed): row}`` of ``per_run.csv``."""
    return {(r["arm"], int(float(r["q"])), int(float(r["seed"]))): r for r in read_csv_rows(path)}


def num(x: Any) -> float:
    """CSV cell as float (NaN if empty)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def parallel(fn: Callable[[Any], Any], items: Sequence[Any], workers: int) -> List[Any]:
    """``fn`` over ``items`` in spawned single-threaded processes (input order kept)."""
    if workers <= 1 or len(items) <= 1:
        return [fn(i) for i in items]
    with ProcessPoolExecutor(max_workers=min(workers, len(items)), mp_context=mp.get_context("spawn")) as pool:
        return list(pool.map(fn, items))


def log_cost(out_dir: Path, step: str, seconds: float, workers: int, n: int) -> None:
    """Append one line to ``costs.csv``."""
    p = out_dir / "costs.csv"
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["step", "wall_sec", "workers", "items", "when_utc"])
        w.writerow([step, f"{seconds:.1f}", workers, n, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())])


# ---------------------------------------------------------------------------------- A1
def a1_dirs(rerun_root: Path, q: int, seed: int) -> Path:
    """``<rerun_root>/q{q}/seed{seed}/bw``."""
    return Path(rerun_root) / f"q{q}" / f"seed{seed}" / "bw"


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def a1_rerun(t3r1_root: Path, rerun_root: Path, workers: int) -> List[Dict[str, Any]]:
    """Run the four failing ``bw`` runs again from their own, unchanged ``run_config.json``."""
    import launch_ms_r1 as L1

    env = L1.child_env()
    items = []
    for q, s in FAILURES:
        src = Path(t3r1_root) / "runs" / f"q{q}" / f"seed{s}" / "bw" / "run_config.json"
        if not src.is_file():
            raise FileNotFoundError(src)
        out = a1_dirs(rerun_root, q, s)
        os.makedirs(out, exist_ok=False)
        cmd = [PY, "-B", "-u", str(ROOT / "run" / "run_ms_stagewise.py"), "--config", str(src), "--out-dir", str(out)]
        items.append((q, s, src, out, cmd))

    def one(it: Tuple[Any, ...]) -> Dict[str, Any]:
        q, s, src, out, cmd = it
        res = L1.run_job(cmd, str(out), env)
        res.update(q=q, seed=s, config=str(src), cmd=" ".join(cmd))
        return res
    with ThreadPoolExecutor(max_workers=min(workers, 4)) as pool:
        return list(pool.map(one, items))


def compare_text(row: Mapping[str, Any], stored: Mapping[str, str], cols: Sequence[str]) -> List[str]:
    """Columns in which the text of a freshly evaluated row differs from the stored CSV cell."""
    bad = []
    for c in cols:
        v = row.get(c)
        txt = "" if v is None else str(v)
        if txt != stored.get(c, ""):
            bad.append(c)
    return bad


def a1_compare(t3r1_root: Path, rerun_root: Path, per_run: Path, out_dir: Path) -> List[Dict[str, Any]]:
    """Compare every re-run with its original, two ways (see the module docstring)."""
    table = per_run_table(per_run)
    rows = []
    for q, s in FAILURES:
        orig = Path(t3r1_root) / "runs" / f"q{q}" / f"seed{s}" / "bw"
        new = a1_dirs(rerun_root, q, s)
        r = subprocess.run([PY, str(ROOT / "tools" / "t3" / "compare_runs.py"), "bw", str(orig), str(new)],
                           capture_output=True, text=True, cwd=ROOT)
        (out_dir / f"a1_compare_runs_q{q}_s{s}.txt").write_text(r.stdout + r.stderr)
        job = {"wave": "t3", "arm": "bw", "q": q, "seed": s, "T": 3, "run": tc.run_name(q, s, "bw"),
               "run_dir": str(new), "kind": "bw", "arrays_dir": None}
        row = te.evaluate_run(job)
        stored = table[("bw", q, s)]
        skip = set(te.ID_COLS)
        cols = [c for c in stored if c not in skip]
        bad = compare_text(row, stored, cols)
        cfg_same = sha256_file(orig / "run_config.json") == sha256_file(new / "run_config.json")
        rows.append({"q": q, "seed": s, "compare_runs_exit": r.returncode,
                     "compare_runs_summary": " | ".join(l for l in r.stdout.splitlines() if "identical" in l),
                     "metric_columns_compared": len(cols), "metric_columns_differing": len(bad),
                     "differing": ";".join(bad), "run_config_sha256_equal": cfg_same,
                     "bit_identical": bool(r.returncode == 0 and not bad and cfg_same)})
    return rows


# ---------------------------------------------------------------------------------- A2
def a2_one(job: Mapping[str, Any]) -> Dict[str, Any]:
    """Snapshot identity of one ``bw`` run (``state_end_stage1.pt`` against the exports)."""
    import torch

    torch.set_num_threads(1)
    d = Path(job["run_dir"])
    st = torch.load(d / "state_end_stage1.pt", map_location="cpu", weights_only=False)
    out: Dict[str, Any] = {"q": job["q"], "seed": job["seed"]}
    e3, e2, e1 = job.get("exports", ("u01600", "u03200", "u03800"))      # the exports at the end of each phase
    pairs = (("frozen3_eq_u01600", st["frozen_stages"]["3"], e3), ("frozen2_eq_u03200", st["frozen_stages"]["2"], e2),
             ("live_eq_u03800", st["agent"]["actor"], e1))
    for name, sd, export in pairs:
        with np.load(d / "weights" / f"{export}.npz") as z:
            ex = {k[len("actor."):]: z[k] for k in z.files if k.startswith("actor.")}
        arr = {k: v.detach().cpu().numpy() for k, v in sd.items()}
        same = sorted(ex) == sorted(arr) and all(ex[k].dtype == arr[k].dtype and ex[k].shape == arr[k].shape
                                                 and np.array_equal(ex[k], arr[k]) for k in arr)
        out[name] = bool(same)
    out["all_identical"] = bool(out["frozen3_eq_u01600"] and out["frozen2_eq_u03200"] and out["live_eq_u03800"])
    return out


# ---------------------------------------------------------------------------------- A3
def same_array(x: np.ndarray, y: np.ndarray) -> bool:
    """Exact equality (dtype, shape, values; NaN equal to NaN)."""
    return x.dtype == y.dtype and x.shape == y.shape and bool(np.array_equal(x, y, equal_nan=x.dtype.kind in "fc"))


def a3_one(job: Mapping[str, Any]) -> Dict[str, Any]:
    """Rebuild the two continuation tables of one ``bw`` run from its frozen snapshots; compare with the calculator."""
    import torch

    torch.set_num_threads(1)
    d = Path(job["run_dir"])
    cfg = te.read_json(d / "run_config.json")
    spec = te.spec_of(cfg)
    st = torch.load(d / "state_end_stage1.pt", map_location="cpu", weights_only=False)
    nets = {int(s): te.net_from_state_dict(sd, cfg) for s, sd in st["frozen_stages"].items()}
    rebuilt = build_nested_tables({3: nets[3], 2: nets[2]}, spec, t_min=1, step=0.05)
    out: Dict[str, Any] = {"q": job["q"], "seed": job["seed"]}
    cand = te.load_bw(d, cfg)
    mean_fn, _ = candidate_fns(spec, cand.nets)
    calc = bc.BRCalculator(game_of(spec), mean_fn)
    dw = float(spec.dw)
    for stage, s_calc in ((2, 3), (1, 2)):
        with np.load(d / f"continuation_table_stage{stage}.npz") as z:
            y_grid, values = z["y_grid"], z["values"]
        t = rebuilt[stage]
        out[f"table{stage}_bit_identical"] = bool(same_array(t.y_grid, y_grid) and same_array(t.values, values))
        y = np.linspace(float(y_grid[0]), float(y_grid[-1]), 201)
        stored = np.interp(y, y_grid, values)
        ref = calc.w_values(s_calc, y)
        diff = np.abs(stored - ref) / dw
        out[f"table{stage}_max_diff_over_dw"] = float(diff.max())
        out[f"table{stage}_argmax_y"] = float(y[int(np.argmax(diff))])
        out[f"table{stage}_flag"] = bool(diff.max() > FLAG_TOL_OVER_DW)
    return out


# ---------------------------------------------------------------------------------- A5
def a5_run(args: Tuple[Dict[str, Any], Dict[str, str], str]) -> Dict[str, Any]:
    """The calculator at the stored states of one run (placement b) and its EXP_root (placement c)."""
    job, stored, arrays_dir = args
    import torch

    torch.set_num_threads(1)
    t0 = time.perf_counter()
    cfg, cand, game, mean_fn = load_run(job)
    T = int(job["T"])
    calc = bc.BRCalculator(game, mean_fn)
    dw = game.dw
    out: Dict[str, Any] = {k: job[k] for k in ("arm", "q", "seed", "T", "run")}
    arr_path = Path(arrays_dir) / f"{job['run']}.npz"
    arrays = dict(np.load(arr_path)) if arr_path.is_file() else {}
    t_g, d_g = int(float(stored["Gmax_full_t"])), float(stored["Gmax_full_d"])
    sv = calc.stage_values(t_g, [d_g])
    out.update(Gmax_t=t_g, Gmax_d=d_g, G_stored_over_dw=num(stored["Gmax_full_over_dw"]),
               G_calc_over_dw=float(sv["G"][0]) / dw, Delta_calc_over_dw_at_Gmax=float(sv["Delta"][0]) / dw,
               e_hat_at_Gmax=float(sv["e_hat"][0]), a_br_at_Gmax=float(sv["a_br"][0]), a_dev_at_Gmax=float(sv["a_dev"][0]),
               slope_at_Gmax=float(sv["slope"][0]))
    if f"fin_t{t_g}_d_grid" in arrays:
        i = int(np.argmin(np.abs(arrays[f"fin_t{t_g}_d_grid"] - d_g)))
        out["Delta_stored_over_dw_at_Gmax"] = float(arrays[f"fin_t{t_g}_delta"][i]) / dw
        out["G_stored_array_over_dw"] = float(arrays[f"fin_t{t_g}_G"][i]) / dw
        out["e_hat_stored_at_Gmax"] = float(arrays[f"fin_t{t_g}_e_hat"][i])
    d_t = float(stored[f"Delta_argmax_d_t{T}"])
    svT = calc.stage_values(T, [d_t])
    out.update(DeltaT_d=d_t, DeltaT_stored_over_dw=num(stored[f"Delta_max_t{T}_over_dw"]),
               DeltaT_calc_over_dw=float(svT["Delta"][0]) / dw, eta_T_stored_over_dw=num(stored["eta_T_over_dw"]),
               e_hat_at_DeltaT=float(svT["e_hat"][0]), a_dev_at_DeltaT=float(svT["a_dev"][0]),
               a_br_at_DeltaT=float(svT["a_br"][0]), G_at_DeltaT_over_dw=float(svT["G"][0]) / dw,
               slope_at_DeltaT=float(svT["slope"][0]))
    sv1 = calc.stage_values(1, [0.0])
    r = calc.root()
    out.update(EXP_stored_over_dw=num(stored["EXP_root_over_dw"]), EXP_calc_over_dw=r["EXP_root"] / dw,
               V1_mean_calc=r["V1_mean"], e1_calc=r["e1"], a_br_1=r["a_br_1"], a_dev_1=float(sv1["a_dev"][0]),
               Delta_1_over_dw=float(sv1["Delta"][0]) / dw, slope_1=float(sv1["slope"][0]))
    out["G_diff_over_dw"] = abs(out["G_calc_over_dw"] - out["G_stored_over_dw"])
    out["DeltaT_diff_over_dw"] = abs(out["DeltaT_calc_over_dw"] - out["DeltaT_stored_over_dw"])
    out["EXP_diff_over_dw"] = abs(out["EXP_calc_over_dw"] - out["EXP_stored_over_dw"])
    if "Delta_stored_over_dw_at_Gmax" in out:
        out["Delta_diff_over_dw"] = abs(out["Delta_calc_over_dw_at_Gmax"] - out["Delta_stored_over_dw_at_Gmax"])
    out["sec"] = time.perf_counter() - t0
    return out


def a5_failure_points(args: Tuple[Dict[str, Any], Dict[str, str]]) -> Dict[str, Any]:
    """Placement (a): the calculator at the stored failure point of one failing run (a row of ``points.csv``)."""
    job, pt = args
    import torch

    torch.set_num_threads(1)
    cfg, cand, game, mean_fn = load_run(job)
    calc = bc.BRCalculator(game, mean_fn)
    t, d = int(float(pt["stage"])), float(pt["d"])
    sv = calc.stage_values(t, [d])
    dw = game.dw
    return {"q": job["q"], "seed": job["seed"], "stage": t, "d": d, "quantity": pt["quantity"],
            "stored_value": num(pt["value"]), "stored_Delta": num(pt["Delta"]), "stored_G": num(pt["G"]),
            "stored_e": num(pt["e"]), "stored_br": num(pt["br"]), "stored_dev": num(pt["dev"]),
            "calc_Delta_over_dw": float(sv["Delta"][0]) / dw, "calc_G_over_dw": float(sv["G"][0]) / dw,
            "calc_e_hat": float(sv["e_hat"][0]), "calc_a_br": float(sv["a_br"][0]), "calc_a_dev": float(sv["a_dev"][0]),
            "calc_slope": float(sv["slope"][0]), "calc_V_mean": float(sv["V_mean"][0]), "calc_V_br": float(sv["V_br"][0]),
            "Delta_diff_over_dw": abs(float(sv["Delta"][0]) / dw - num(pt["Delta"])),
            "G_diff_over_dw": abs(float(sv["G"][0]) / dw - num(pt["G"])),
            "br_effort_diff": abs(float(sv["a_br"][0]) - num(pt["br"]))}


def agree_one(args: Tuple[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculator ``V^mean`` at the root and at 20 states per stage against the final-tier verifier arrays of one run."""
    (job,) = args
    import torch

    torch.set_num_threads(1)
    cfg, cand, game, mean_fn = load_run(job)
    d = Path(job["run_dir"])
    if job["kind"] == "bw":
        with np.load(d / "freeze_stage1_final.npz") as z:
            arr = {k: z[k] for k in z.files}
        source = "freeze_stage1_final.npz"
    elif job["kind"] == "joint":
        with np.load(d / "final_eval_final.npz") as z:
            arr = {k: z[k] for k in z.files}
        source = "final_eval_final.npz"
    else:
        arr = te.evaluate_tier(cand, FINAL_CONFIG).arrays
        source = "t3_eval.evaluate_tier(load_v20, FINAL_CONFIG)"
    calc = bc.BRCalculator(game, mean_fn)
    out: Dict[str, Any] = {k: job[k] for k in ("arm", "q", "seed")}
    out["verifier_arrays"] = source
    worst = 0.0
    for t in range(1, game.T + 1):
        grid, vm = np.asarray(arr[f"v_t{t}_d_grid"]), np.asarray(arr[f"v_t{t}_v_mean"])
        idx = np.unique(np.round(np.linspace(0, grid.size - 1, 20)).astype(int)) if t > 1 else np.array([0])
        vc = calc.mean_value(t, grid[idx])
        diff = np.abs(vc - vm[idx]) / game.dw
        out[f"max_diff_over_dw_t{t}"] = float(diff.max())
        out[f"n_states_t{t}"] = int(idx.size)
        worst = max(worst, float(diff.max()))
    out["V1_mean_stored"] = float(arr["v_t1_v_mean"][0])
    out["V1_mean_calc"] = float(calc.mean_value(1, [0.0])[0])
    out["max_diff_over_dw"] = worst
    out["exceeds_1e-4"] = bool(worst > 1e-4)
    return out


def conv_one(args: Tuple[Dict[str, Any], List[Tuple[int, float]], str]) -> Dict[str, Any]:
    """Convergence test of the calculator for one run at the given (stage, d) points."""
    job, points, label = args
    import torch

    torch.set_num_threads(1)
    cfg, cand, game, mean_fn = load_run(job)
    res = bc.convergence(game, mean_fn, points)
    res["label"] = label
    res["run"] = job["run"]
    return res


# ---------------------------------------------------------------------------------- A6
def a6_one(args: Tuple[Dict[str, Any], Dict[str, str]]) -> Dict[str, Any]:
    """The refined verifier tier on one run's last iterate against the stored final tier."""
    job, stored = args
    import torch

    torch.set_num_threads(1)
    t0 = time.perf_counter()
    cfg = te.read_json(Path(job["run_dir"]) / "run_config.json")
    cand = te.load_candidate(job["kind"], Path(job["run_dir"]), cfg)
    mean_fn, _ = candidate_fns(cand.spec, cand.nets)
    crit, _ = tm.load_criteria()
    ev = te.evaluate_tier(cand, REFINED)
    m = tm.tier_metrics(ev, mean_fn, cand.spec, cand.recovery_step)
    T = int(job["T"])
    out: Dict[str, Any] = {k: job[k] for k in ("arm", "q", "seed", "T", "run")}
    out["refined_valid"] = bool(m["valid"])
    keys = ["Gmax_full_over_dw", "eta_T_over_dw", "EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw"]
    keys += [f"G_max_t{t}_over_dw" for t in range(1, T + 1)] + [f"Delta_max_t{t}_over_dw" for t in range(1, T + 1)]
    for k in keys:
        out[f"refined_{k}"] = float(m[k])
        out[f"final_{k}"] = num(stored[k])
        out[f"absdiff_{k}"] = abs(float(m[k]) - num(stored[k]))
    out["refined_Gmax_full_t"], out["refined_Gmax_full_d"] = m["Gmax_full_t"], m["Gmax_full_d"]
    g = crit["gates"]
    ga = (float(m["eta_T_over_dw"]) <= g["G-A"]["eta_T_over_dw"] and float(m["rmse_pos_T_over_eT0"]) <= g["G-A"]["rmse_pos_T_over_eT0"]
          and float(m["tail_mean_T_over_eT0"]) <= g["G-A"]["tail_mean_T_over_eT0"])
    gf = float(m["Gmax_full_over_dw"]) <= g["G-F"]["Gmax_full_over_dw"]
    ga_fin = stored["gate_G_A"] == "True"
    gf_fin = stored["gate_G_F"] == "True"
    out.update(refined_G_A=bool(ga), refined_G_F=bool(gf), final_G_A=ga_fin, final_G_F=gf_fin,
               verdict_differs=bool(ga != ga_fin or gf != gf_fin),
               distance_to_limit_eta=abs(num(stored["eta_T_over_dw"]) - g["G-A"]["eta_T_over_dw"]),
               distance_to_limit_gmax=abs(num(stored["Gmax_full_over_dw"]) - g["G-F"]["Gmax_full_over_dw"]))
    out["sec"] = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------------------------- A7: pure cores
def beta_sd_norm(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Normalised standard deviation sqrt(ab / ((a+b)^2 (a+b+1))) of a Beta(a, b) on [0, 1]."""
    a, b = np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float)
    s = a + b
    return np.sqrt(a * b / (s * s * (s + 1.0)))


def stretch_nodes(grid: np.ndarray, e_hat: np.ndarray, q: float, effort: float = STRETCH_EFFORT) -> np.ndarray:
    """Indices of the in-band (``|d| < 2q``) stage-2 nodes with learned mean effort below ``effort``."""
    return np.where((np.asarray(e_hat) < effort) & (np.abs(np.asarray(grid)) < 2.0 * q - BAND_TOL))[0]


def runs_of_indices(idx: np.ndarray) -> List[Tuple[int, int]]:
    """Maximal runs of consecutive indices as ``(first, last)``."""
    out: List[Tuple[int, int]] = []
    for i in idx:
        if out and i == out[-1][1] + 1:
            out[-1] = (out[-1][0], int(i))
        else:
            out.append((int(i), int(i)))
    return out


def first_below(e_by_export: np.ndarray, effort: float = STRETCH_EFFORT) -> Tuple[np.ndarray, np.ndarray]:
    """Per node (column): the index of the first export with mean effort below ``effort`` (-1 if none) and the index
    from which the node stays below ``effort`` at every later export (-1 if it is not below at the last export).

    Args:
        e_by_export: Array ``(n_exports, n_nodes)``.
    """
    below = np.asarray(e_by_export) < effort
    n, m = below.shape
    first = np.where(below.any(axis=0), below.argmax(axis=0), -1)
    stay = np.full(m, -1)
    for j in range(m):
        if below[-1, j]:
            k = n - 1
            while k > 0 and below[k - 1, j]:
                k -= 1
            stay[j] = k
    return first, stay


def local_maxima(e: np.ndarray, q_vals: np.ndarray) -> List[Tuple[float, float]]:
    """Discrete local maxima ``(effort, Q)`` of a curve on a grid, the end points included (``>=`` to the neighbour)."""
    n = q_vals.size
    out = []
    for j in range(n):
        left = q_vals[j] >= q_vals[j - 1] if j > 0 else True
        right = q_vals[j] >= q_vals[j + 1] if j < n - 1 else True
        if left and right and not (j > 0 and j < n - 1 and q_vals[j] == q_vals[j - 1] == q_vals[j + 1]):
            out.append((float(e[j]), float(q_vals[j])))
    return out


def objective_at_node(calc: bc.BRCalculator, d: float, e_grid: np.ndarray) -> Dict[str, Any]:
    """Shape of ``Q_2^m(d, .)`` at the learned effort: slope, local maxima near zero, the global maximum.

    Args:
        calc: Calculator of the export's policy (stage 3 frozen).
        d: Stage-2 gap.
        e_grid: Effort grid on [0, 100] (step 0.01).
    """
    dw = calc.game.dw
    sv = calc.stage_values(2, [d], brent=False)
    curve = calc.q_curve(2, d, e_grid, "mean")
    e_hat = float(sv["e_hat"][0])
    q_hat = float(sv["V_mean"][0])
    jstar = int(np.argmax(curve))
    lm = local_maxima(e_grid, curve)
    near_zero = [m for m in lm if m[0] <= STRETCH_EFFORT + 1e-12]
    return {"d": float(d), "e_hat": e_hat, "slope_at_e_hat": float(sv["slope"][0]),
            "e_opp": float(sv["e_opp"][0]), "Delta_over_dw": float(sv["Delta"][0]) / dw, "G_over_dw": float(sv["G"][0]) / dw,
            "a_dev": float(sv["a_dev"][0]), "a_br": float(sv["a_br"][0]),
            "grid_argmax_effort": float(e_grid[jstar]), "gain_to_grid_max_over_dw": (float(curve[jstar]) - q_hat) / dw,
            "distance_to_global_max": float(sv["a_dev"][0]) - e_hat,
            "n_local_maxima": len(lm), "local_maxima_efforts": ";".join(f"{m[0]:.2f}" for m in lm),
            "local_max_near_zero": bool(len(near_zero) > 0),
            "q_at_zero_minus_q_hat_over_dw": (float(curve[0]) - q_hat) / dw,
            "class": "equilibrium_like" if float(sv["a_dev"][0]) < STRETCH_EFFORT else "policy_error"}


def analyze_nodes(game: bc.Game, stage3_policy: Callable[[np.ndarray], np.ndarray], exports: Sequence[Tuple[int, Callable[[np.ndarray], np.ndarray],
                  Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]]], grid: np.ndarray, nodes: Optional[Sequence[float]] = None,
                  cfg: bc.CalcConfig = bc.CalcConfig(), e_step: float = 0.01) -> Dict[str, Any]:
    """The formation and the objective analysis (A7 steps 1, 2, 3, 5) for a sequence of stage-2 exports.

    Args:
        game: T = 3 game.
        stage3_policy: Frozen stage-3 mean effort ``d -> e``.
        exports: ``[(update, mean2, beta2)]`` in training order: the stage-2 mean effort ``d -> e`` and Beta parameters
            ``d -> (alpha, beta)`` of each export. The last export is the evaluated policy.
        grid: The final-tier stage-2 grid.
        nodes: Gaps to analyse; ``None`` takes the stretch nodes of the last export (in band, mean effort below 1).
        cfg: Calculator resolution.
        e_step: Effort grid of the objective curves.

    Returns:
        ``{"nodes", "formation": [...per node], "series": [...long rows], "objective": [...], "summary": {...}}``.
    """
    grid = np.asarray(grid, dtype=float)
    e_mat = np.stack([np.asarray(m(grid), dtype=float) for _, m, _ in exports])        # (n_exports, n_grid)
    updates = np.array([u for u, _, _ in exports])
    if nodes is None:
        idx = stretch_nodes(grid, e_mat[-1], game.q)
    else:
        idx = np.array([int(np.argmin(np.abs(grid - float(d)))) for d in nodes], dtype=int)
    node_d = grid[idx]
    first, stay = first_below(e_mat[:, idx])
    formation = [{"d": float(node_d[j]), "first_below1_update": int(updates[first[j]]) if first[j] >= 0 else None,
                  "stays_below1_from_update": int(updates[stay[j]]) if stay[j] >= 0 else None,
                  "e_hat_final": float(e_mat[-1, idx[j]]), "n_exports_below1": int((e_mat[:, idx[j]] < STRETCH_EFFORT).sum())}
                 for j in range(idx.size)]
    nb = (e_mat[:, idx] < STRETCH_EFFORT).sum(axis=1) if idx.size else np.zeros(len(exports), dtype=int)
    series = []
    for (u, _, bfun), e_row in zip(exports, e_mat):
        a, b = bfun(node_d) if idx.size else (np.zeros(0), np.zeros(0))
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        sd = beta_sd_norm(a, b) if idx.size else np.zeros(0)
        for j in range(idx.size):
            series.append({"update": int(u), "d": float(node_d[j]), "e_hat": float(e_row[idx[j]]), "alpha": float(a[j]),
                           "beta": float(b[j]), "sd_norm": float(sd[j])})
    e_grid = np.linspace(game.e_min, game.e_max, int(round(game.e_range / e_step)) + 1)

    def policy_of(mean2: Callable[[np.ndarray], np.ndarray]) -> Callable[[int, np.ndarray], np.ndarray]:
        def pol(t: int, d: np.ndarray) -> np.ndarray:
            return stage3_policy(d) if int(t) == 3 else mean2(d)
        return pol

    base = bc.BRCalculator(game, policy_of(exports[-1][1]), cfg)
    base.W(3, "mean")                    # warm the stage-3 tables that the per-export clones share
    base.W(3, "br")
    objective: List[Dict[str, Any]] = []
    for j in range(idx.size):
        picks = [("final", len(exports) - 1)]
        if first[j] >= 0 and first[j] != len(exports) - 1:
            picks.insert(0, ("first_below1", int(first[j])))
        elif first[j] >= 0:
            picks = [("first_below1", int(first[j])), ("final", len(exports) - 1)]
        for label, ei in picks:
            calc = base if ei == len(exports) - 1 else base.with_policy(policy_of(exports[ei][1]), 3)
            row = objective_at_node(calc, float(node_d[j]), e_grid)
            row.update(export=label, update=int(updates[ei]))
            objective.append(row)
    final_rows = [r for r in objective if r["export"] == "final"]
    summary = {"n_nodes": int(idx.size), "n_equilibrium_like": sum(1 for r in final_rows if r["class"] == "equilibrium_like"),
               "n_policy_error": sum(1 for r in final_rows if r["class"] == "policy_error"),
               "n_local_max_near_zero_final": sum(1 for r in final_rows if r["local_max_near_zero"]),
               "n_positive_slope_final": sum(1 for r in final_rows if r["slope_at_e_hat"] > 0.0),
               "stretches": [(float(grid[a]), float(grid[b])) for a, b in runs_of_indices(idx)],
               "n_below1_by_export": [int(x) for x in nb], "exports": [int(u) for u in updates]}
    return {"nodes": node_d, "node_index": idx, "formation": formation, "series": series, "objective": objective,
            "summary": summary, "e_mat": e_mat, "updates": updates}


# ---------------------------------------------------------------------------------- A7: run level
def ms_windows(rows: Sequence[Mapping[str, str]], first: int, last: int) -> Dict[str, Any]:
    """Training-signal summary of the stage-2 phase over the local updates ``first..last`` of ``ms_updates.csv``."""
    sel = [r for r in rows if int(r["stage"]) == 2 and first <= int(r["local"]) <= last]
    if not sel:
        return {}
    f = lambda c: np.array([float(r[c]) for r in sel])  # noqa: E731
    n, lo = f("d1_L_s2_n").sum(), f("d1_L_s2_lo").sum()
    return {"local_first": first, "local_last": last, "updates": len(sel),
            "kl_final_epoch_mean": float(f("kl_final_epoch").mean()), "clip_frac_mean": float(f("clip_frac").mean()),
            "adv_raw_mean_mean": float(f("adv_raw_mean").mean()), "adv_raw_std_mean": float(f("adv_raw_std").mean()),
            "starts_tail": int(f("n_start_tail").sum()), "starts_near": int(f("n_start_near").sum()),
            "starts_mid": int(f("n_start_mid").sum()), "learner_draws": int(n), "learner_draws_below_clamp": int(lo),
            "share_below_clamp": float(lo / n) if n else float("nan")}


def a7_run(args: Tuple[Dict[str, Any], Optional[List[float]], str]) -> Dict[str, Any]:
    """The A7 analysis of one ``bw`` run at its own stretch nodes (``nodes`` None) or at given nodes."""
    job, nodes, group = args
    import torch

    torch.set_num_threads(1)
    t0 = time.perf_counter()
    d = Path(job["run_dir"])
    cfg = te.read_json(d / "run_config.json")
    spec = te.spec_of(cfg)
    game = game_of(spec)
    net3 = te.net_from_export(d / "weights" / "u01600.npz", cfg)
    mean3, _ = candidate_fns(spec, {3: net3})
    exports = []
    for u in STAGE2_EXPORTS:
        net = te.net_from_export(d / "weights" / f"u{u:05d}.npz", cfg)
        m2, b2 = candidate_fns(spec, {2: net})
        exports.append((u, (lambda x, m=m2: m(2, x)), (lambda x, b=b2: b(2, x))))
    arrays_path = Path(job["arrays_dir"]) / f"{job['run']}.npz" if job.get("arrays_dir") else None
    grid = np.load(arrays_path)["fin_t2_d_grid"] if arrays_path and arrays_path.is_file() else bc.sym_grid(game.B, 2.0)
    res = analyze_nodes(game, lambda x: mean3(3, x), exports, grid, nodes)
    rows = read_csv_rows(d / "ms_updates.csv")
    out: Dict[str, Any] = {"q": job["q"], "seed": job["seed"], "group": group, "analysis": res}
    form = [f["first_below1_update"] for f in res["formation"] if f["first_below1_update"] is not None]
    f_local = (min(form) - 1600) if form else None
    out["formation_local"] = f_local
    out["windows"] = {"formation": ms_windows(rows, max(1, f_local - 99), f_local) if f_local else {},
                      "decay": ms_windows(rows, 1201, 1600)}
    for i, (a, b) in enumerate(((1, 400), (401, 800), (801, 1200), (1201, 1600)), start=1):
        out["windows"][f"quarter{i}"] = ms_windows(rows, a, b)
    out["sec"] = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------------------------- A8
def a8_one(args: Tuple[Dict[str, Any], int]) -> Dict[str, Any]:
    """Final-tier eta_3 and the shape parts of G-A at one stage-3 export (its network used at every stage)."""
    job, u = args
    import torch

    torch.set_num_threads(1)
    d = Path(job["run_dir"])
    cfg = te.read_json(d / "run_config.json")
    spec = te.spec_of(cfg)
    net = te.net_from_export(d / "weights" / f"u{u:05d}.npz", cfg)
    cand = te.Candidate(spec, {t: net for t in range(1, spec.T + 1)}, float(cfg["record"]["protocol"]["recovery_step"]))
    ev = te.evaluate_tier(cand, FINAL_CONFIG)
    mean_fn, _ = candidate_fns(spec, cand.nets)
    m = tm.tier_metrics(ev, mean_fn, spec, cand.recovery_step)
    return {"q": job["q"], "seed": job["seed"], "update": u, "eta_3_over_dw": float(m["eta_T_over_dw"]),
            "rmse_pos_T_over_eT0": float(m["rmse_pos_T_over_eT0"]), "tail_mean_T_over_eT0": float(m["tail_mean_T_over_eT0"]),
            "valid": bool(m["valid"])}


# ---------------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R2 audit tool", allow_abbrev=False)
    ap.add_argument("step", choices=["a1-rerun", "a1-compare", "a2", "a3", "a5", "a5-conv", "a6", "a7", "a8"])
    ap.add_argument("--t3r1-root", default=str(DEFAULT_T3R1))
    ap.add_argument("--v20-root", default=str(DEFAULT_V20))
    ap.add_argument("--per-run", default=str(DEFAULT_PER_RUN))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--rerun-root", default=str(ROOT / "results" / "t3_r2" / "reruns"))
    ap.add_argument("--workers", type=int, default=8, help="at most 40")
    a = ap.parse_args(argv)
    if not 1 <= a.workers <= 40:
        ap.error("--workers must be in 1..40")
    t3r1, v20, out = Path(a.t3r1_root), Path(a.v20_root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = {"t3r1_root": str(t3r1), "v20_root": str(v20), "per_run": a.per_run, "step": a.step,
              "head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
              "status_porcelain": subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True,
                                                 text=True).stdout.splitlines()}
    t_start = time.perf_counter()
    jobs = audit_jobs(t3r1, v20)
    for j in jobs:
        j["arrays_dir"] = str(t3r1 / "eval" / "arrays")
    table = per_run_table(Path(a.per_run))
    bw_jobs = [j for j in jobs if j["arm"] == "bw"]
    n_items = 0
    if a.step == "a1-rerun":
        res = a1_rerun(t3r1, Path(a.rerun_root), a.workers)
        write_csv_rows(out / "a1_rerun_exit.csv", res)
        n_items = len(res)
        ok = all(r["returncode"] == 0 for r in res)
    elif a.step == "a1-compare":
        res = a1_compare(t3r1, Path(a.rerun_root), Path(a.per_run), out)
        write_csv_rows(out / "a1_compare.csv", res)
        n_items = len(res)
        ok = all(r["bit_identical"] for r in res)
    elif a.step == "a2":
        res = parallel(a2_one, bw_jobs, a.workers)
        write_csv_rows(out / "a2_snapshot_identity.csv", res)
        n_items = len(res)
        ok = all(r["all_identical"] for r in res)
    elif a.step == "a3":
        sel = [job_of(jobs, "bw", q, s) for q, s in A3_RUNS]
        res = parallel(a3_one, sel, a.workers)
        write_csv_rows(out / "a3_tables.csv", res)
        n_items = len(res)
        ok = all(r["table2_bit_identical"] and r["table1_bit_identical"] for r in res)
    elif a.step == "a5":
        pts = read_csv_rows(ROOT / "results" / "t3_r1" / "analysis" / "bw_failures" / "points.csv")
        fa = [(job_of(jobs, "bw", int(p["q"]), int(p["seed"])), p) for p in pts]
        write_csv_rows(out / "a5_failure_points.csv", parallel(a5_failure_points, fa, min(a.workers, 4)))
        items = [(j, table[(j["arm"], int(j["q"]), int(j["seed"]))], str(t3r1 / "eval" / "arrays")) for j in jobs]
        res = parallel(a5_run, items, a.workers)
        write_csv_rows(out / "a5_runs.csv", res)
        ag = parallel(agree_one, [(job_of(jobs, arm, q, s),) for arm, q, s in AGREE_RUNS], min(a.workers, 5))
        write_csv_rows(out / "a5_verifier_agreement.csv", ag)
        n_items = len(res)
        ok = True
    elif a.step == "a5-conv":
        pts = read_csv_rows(ROOT / "results" / "t3_r1" / "analysis" / "bw_failures" / "points.csv")
        items = []
        for p in pts:
            items.append((job_of(jobs, "bw", int(p["q"]), int(p["seed"])), [(int(float(p["stage"])), float(p["d"]))], "failure_point"))
        for arm, q, s in AGREE_RUNS:
            items.append((job_of(jobs, arm, q, s), [(1, 0.0)], "root"))
        stretch = read_csv_rows(ROOT / "results" / "t3_r1" / "analysis" / "bw_failures" / "zero_effort.csv")
        for r in stretch:
            if int(r["q"]) == 60:
                lo, hi = (int(x) for x in r["stretches"].strip("[]").replace(" ", "").split(",")[:2]) \
                    if ";" not in r["stretches"] else (None, None)
                nodes = list(range(lo, hi + 1, 2)) if lo is not None else []
                n_take = 6 if int(r["seed"]) == 50519 else 7
                pick = [nodes[int(round(i))] for i in np.linspace(0, len(nodes) - 1, min(n_take, len(nodes)))] if nodes else []
                if pick:
                    items.append((job_of(jobs, "bw", 60, int(r["seed"])), [(2, float(x)) for x in pick], "stretch_nodes"))
        for r in stretch:                      # extra (beyond the 20 nodes of the prompt): the four q = 50 stretch runs, 5 nodes each
            if int(r["q"]) == 50:
                lo, hi = (int(x) for x in r["stretches"].strip("[]").replace(" ", "").split(",")[:2])
                nodes = list(range(lo, hi + 1, 2))
                pick = [nodes[int(round(i))] for i in np.linspace(0, len(nodes) - 1, min(5, len(nodes)))]
                items.append((job_of(jobs, "bw", 50, int(r["seed"])), [(2, float(x)) for x in pick], "stretch_nodes_q50_extra"))
        res = parallel(conv_one, items, a.workers)
        write_json(out / "a5_convergence.json", res)
        n_items = len(res)
        ok = True
    elif a.step == "a6":
        items = [(j, table[(j["arm"], int(j["q"]), int(j["seed"]))]) for j in jobs]
        res = parallel(a6_one, items, a.workers)
        write_csv_rows(out / "a6_refined.csv", res)
        n_items = len(res)
        ok = True
    elif a.step == "a7":
        ze = read_csv_rows(ROOT / "results" / "t3_r1" / "analysis" / "bw_failures" / "zero_effort.csv")
        stretch_runs = {(int(r["q"]), int(r["seed"])) for r in ze}
        items = [(j, None, "stretch") for j in bw_jobs if (int(j["q"]), int(j["seed"])) in stretch_runs]
        res = parallel(a7_run, items, a.workers)
        nodes_by_q: Dict[int, List[float]] = {}
        for r in res:
            nodes_by_q.setdefault(int(r["q"]), [])
            nodes_by_q[int(r["q"])] = sorted(set(nodes_by_q[int(r["q"])]) | {float(x) for x in r["analysis"]["nodes"]})
        comp = [(j, nodes_by_q[int(j["q"])], "comparison") for j in bw_jobs if (int(j["q"]), int(j["seed"])) not in stretch_runs]
        res += parallel(a7_run, comp, a.workers)
        write_json(out / "a7_runs.json", [{k: v for k, v in r.items() if k != "analysis"} | {
            "summary": r["analysis"]["summary"], "formation": r["analysis"]["formation"],
            "objective": r["analysis"]["objective"], "nodes": r["analysis"]["nodes"]} for r in res])
        write_csv_rows(out / "a7_series.csv", [dict(q=r["q"], seed=r["seed"], group=r["group"], **s)
                                              for r in res for s in r["analysis"]["series"]])
        np.savez(out / "a7_emat.npz", **{f"q{r['q']}_s{r['seed']}": r["analysis"]["e_mat"] for r in res},
                 updates=np.array(STAGE2_EXPORTS))
        n_items = len(res)
        ok = True
    else:
        items = [(j, u) for j in bw_jobs for u in LAST16]
        res = parallel(a8_one, items, a.workers)
        write_csv_rows(out / "a8_last16.csv", res)
        n_items = len(res)
        ok = True
    sec = time.perf_counter() - t_start
    log_cost(out, a.step, sec, a.workers, n_items)
    record.update(wall_sec=sec, ok=ok, items=n_items)
    write_json(out / f"record_{a.step}.json", record)
    print(f"{a.step}: {n_items} items, {sec:.1f} s, ok={ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
