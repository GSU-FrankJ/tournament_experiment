#!/usr/bin/env python3
"""Offline learning curves from the stage-2 weight exports (MS-R4 prompt D5 and adjustment 3; evaluation only).

For every stage-2 export ``weights/u<N>.npz`` (N = 25, 50, ..., the terminal-stage budget: 112 points for the 2800-update
arms, 64 for the 1600-update arms) the tool builds the composite candidate with the export at every stage, exactly as the
live actor is used during the stage-2 phase (the same ``BetaActor`` forward in float32, ``conc_scale`` and ``actor_variant``
read from the export; a variant the loader does not know is refused, never read as the tanh actor), and runs
``utils.v2_metrics.evaluate`` on

  * the FINAL tier (the primary curve of E2 and the E1 learning curves): ``eta_2/DW``, ``|peak error|``, ``e_hat_2(0)``, the
    signed peak error, ``g2_at_0``, RMSE_pos/e2*(0), tail mean/e2*(0), the symmetry error, ``Gmax_full/DW``;
  * the DEVELOPMENT tier (check C-OFF): the same scalars, compared with the online check rows of
    ``ms_checks_stage2.csv`` at the same update (the columns of ``CHECK_REPORT`` and ``Delta``; exact equality of the parsed
    floats, which the CSV round-trips) and, at the last export, with the freeze record of ``gates.json`` (final-tier eta, the
    development-tier eta and the final-tier peak).

The closed form enters the evaluation (``g2_at_0``, the peak and RMSE errors), never a run. The tool is pure: it moves no RNG
stream, writes nothing under the run directories, and runs one process per run (at most 40 workers, single-threaded).

Outputs under ``--out`` (an existing non-empty directory is refused): ``q<q>/seed<seed>/<arm>.csv`` (one row per export),
``curves_all.csv`` (all rows), ``c_off.csv`` (one row per run), ``curves_info.json``. Exit code 0 when every C-OFF check passed,
3 when a run is missing / failed or a C-OFF comparison differs, 2 on an argument error.

Usage:
    python tools/ms/r4_curves.py --pilot-root results/ms_r4/pilot --out results/ms_r4/curves --workers 40
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))

MAX_WORKERS = 40
EXPORT_EVERY = 25
#: scalars stored per tier (``final_*`` / ``dev_*`` columns)
TIER_KEYS = ("eta_T_over_dw", "stage2_peak_rel_err_abs", "stage2_peak_rel_err_signed", "e2_at_0", "g2_at_0",
             "stage2_rmse_pos_over_g2_0", "stage2_tail_mean_over_g2_0", "stage2_sym_err_max", "Gmax_full_over_dw")
#: online check column -> offline development-tier scalar (C-OFF); ``Delta`` is full_delta_max[2] / DW = eta_T_over_dw at T = 2
C_OFF_PAIRS = (("Delta", "eta_T_over_dw"), ("eta_T_over_dw", "eta_T_over_dw"),
               ("stage2_peak_rel_err_signed", "stage2_peak_rel_err_signed"),
               ("stage2_peak_rel_err_abs", "stage2_peak_rel_err_abs"),
               ("stage2_rmse_pos_over_g2_0", "stage2_rmse_pos_over_g2_0"),
               ("stage2_tail_mean_over_g2_0", "stage2_tail_mean_over_g2_0"),
               ("stage2_tail_max_over_g2_0", "stage2_tail_max_over_g2_0"),
               ("e1_at_0", "e1_at_0"), ("e2_at_0", "e2_at_0"),
               ("stage1_rel_err_signed", "stage1_rel_err_signed"), ("Gmax_full_over_dw", "Gmax_full_over_dw"))
CURVE_COLS = (["arm", "q", "seed", "update", "local"] + [f"final_{k}" for k in TIER_KEYS] + [f"dev_{k}" for k in TIER_KEYS]
              + ["final_valid", "dev_valid", "final_eval_sec", "dev_eval_sec", "conc_scale", "actor_variant"])
C_OFF_COLS = ["arm", "q", "seed", "status", "n_exports", "n_compared", "n_mismatch", "max_abs_diff", "first_mismatch",
              "last_export_update", "last_final_eta_equals_freeze", "last_dev_eta_equals_freeze",
              "last_final_peak_equals_freeze", "ok", "eval_sec_final", "eval_sec_dev"]


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def export_paths(run_dir: Path, budget: int, every: int = EXPORT_EVERY, strict: bool = True) -> List[Tuple[int, Path]]:
    """``(update, path)`` of the stage-2 exports u<every> ... u<budget>.

    Args:
        run_dir: The run directory.
        budget: Last terminal-stage update.
        every: Export interval.
        strict: ``True`` (default): a missing export is a ``FileNotFoundError``; ``False``: the exports that exist (a failed
            run stops early), the caller reports the gap.
    """
    out: List[Tuple[int, Path]] = []
    for u in range(every, budget + 1, every):
        p = run_dir / "weights" / f"u{u:05d}.npz"
        if not p.exists():
            if strict:
                raise FileNotFoundError(str(p))
            continue
        out.append((u, p))
    return out


def _spec_ppo(cfg: Mapping[str, Any]) -> Tuple[Any, Mapping[str, Any]]:
    from envs.curriculum_env import GameSpec
    g = cfg["record"]["game"]
    spec = GameSpec(**{k: g[k] for k in ("w_h", "w_l", "k", "q", "T", "e_min", "e_max")})
    return spec, cfg["record"]["ppo"]


def evaluate_export(arrays: Mapping[str, np.ndarray], spec: Any, ppo: Mapping[str, Any], recovery_step: float,
                    tiers: Sequence[str] = ("final", "development")) -> Dict[str, Any]:
    """Final- and development-tier ``evaluate`` of the composite candidate built from one export (pure).

    The same network (the export) serves every stage, as the live actor does during the stage-2 phase.

    Raises:
        ValueError: An ``actor_variant`` the loader does not know.
    """
    import replay_dev_rule as rdr
    from utils.dp_br_verifier import DEV_CONFIG, FINAL_CONFIG
    from utils.v2_metrics import evaluate

    import torch
    # nn.Linear's default initialisation (overwritten by the exported weights) draws from the process-global torch RNG:
    # fork it so that the tool moves no RNG
    with torch.random.fork_rng(devices=[]):
        net = rdr.build_actor(arrays, int(ppo["hidden"]), float(ppo["c_min"]), float(ppo["mu_clamp"]))
    mean_fn, beta_fn = rdr.candidate_fns(spec, {t: net for t in range(1, int(spec.T) + 1)})
    cfgs = {"final": FINAL_CONFIG, "development": DEV_CONFIG}
    out: Dict[str, Any] = {"actor_variant": net.variant, "conc_scale": float(net.conc_scale)}
    for tier in tiers:
        t0 = time.perf_counter()
        ev = evaluate(mean_fn, spec, cfgs[tier], beta_fn=beta_fn, recovery_step=recovery_step)
        out[tier] = {"scalars": ev.scalars, "sec": time.perf_counter() - t0}
    return out


def curve_run(run_dir: Path, arm: str, q: int, seed: int, c_off: bool = True, n_max: Optional[int] = None,
              strict_online: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The curve rows and the C-OFF record of one run.

    Args:
        run_dir: The run directory (``run_config.json``, ``weights/``, ``ms_checks_stage2.csv``, ``gates.json``).
        arm: Arm name (a label of the rows).
        q: Noise half-width.
        seed: Run seed.
        c_off: Also evaluate the development tier and compare it with the online checks and the freeze record.
        n_max: Evaluate only the first ``n_max`` exports (tests).
        strict_online: An export without an online check row at the same update is a mismatch (the real runs export and
            check every 25 updates). ``False`` (tests with more exports than checks) counts such exports in
            ``n_no_online_row`` instead.

    Returns:
        ``(rows, c_off_record)``; a run without a readable directory is a record with ``status`` ``missing``.
    """
    rec: Dict[str, Any] = {"arm": arm, "q": q, "seed": seed, "status": "ok", "n_exports": 0, "n_compared": 0, "n_mismatch": 0,
                           "max_abs_diff": 0.0, "first_mismatch": "", "ok": False}
    rows: List[Dict[str, Any]] = []
    try:
        cfg = json.load(open(run_dir / "run_config.json"))
        spec, ppo = _spec_ppo(cfg)
        recovery_step = float(cfg["record"]["protocol"]["recovery_step"])
        budget = int(cfg["pipeline"]["budgets"]["2"])
        every = int(cfg["record"]["protocol"]["weights_every"])
        exports = export_paths(run_dir, budget, every, strict=False)       # a failed run stops early: evaluate what exists
        if not exports:
            raise FileNotFoundError(f"no stage-2 export under {run_dir / 'weights'}")
        n_missing = budget // every - len(exports)
        if n_max is not None:
            exports = exports[:n_max]
        online = {int(r["local"]): r for r in csv.DictReader(open(run_dir / "ms_checks_stage2.csv"))} if c_off else {}
    except Exception as exc:  # noqa: BLE001 - reported as a missing / unreadable run
        rec.update(status="missing", first_mismatch=f"{type(exc).__name__}: {exc}", ok=False)
        return rows, rec
    tiers = ("final", "development") if c_off else ("final",)
    sec_f = sec_d = 0.0
    for u, p in exports:
        with np.load(p) as z:
            arrays = {k: z[k] for k in z.files}
        ev = evaluate_export(arrays, spec, ppo, recovery_step, tiers)
        fin = ev["final"]["scalars"]
        row: Dict[str, Any] = {"arm": arm, "q": q, "seed": seed, "update": u, "local": u,
                               "conc_scale": ev["conc_scale"], "actor_variant": ev["actor_variant"],
                               "final_valid": bool(fin["valid"]), "final_eval_sec": ev["final"]["sec"]}
        sec_f += ev["final"]["sec"]
        row.update({f"final_{k}": _f(fin.get(k)) for k in TIER_KEYS})
        if c_off:
            dev = ev["development"]["scalars"]
            sec_d += ev["development"]["sec"]
            row.update({f"dev_{k}": _f(dev.get(k)) for k in TIER_KEYS})
            row.update(dev_valid=bool(dev["valid"]), dev_eval_sec=ev["development"]["sec"])
            on = online.get(u)
            if on is None:
                if strict_online:
                    rec["n_mismatch"] += 1
                    rec["first_mismatch"] = rec["first_mismatch"] or f"no online check row at local {u}"
                else:
                    rec["n_no_online_row"] = rec.get("n_no_online_row", 0) + 1
            else:
                for col, key in C_OFF_PAIRS:
                    a, b = _f(on.get(col)), _f(dev.get(key))
                    rec["n_compared"] += 1
                    same = (a == b) or (math.isnan(a) and math.isnan(b))
                    if not same:
                        rec["n_mismatch"] += 1
                        d = abs(a - b) if (math.isfinite(a) and math.isfinite(b)) else float("inf")
                        rec["max_abs_diff"] = max(rec["max_abs_diff"], d)
                        rec["first_mismatch"] = rec["first_mismatch"] or f"local {u} column {col}: online {a!r} offline {b!r}"
        rows.append(row)
    rec.update(n_exports=len(rows), eval_sec_final=sec_f, eval_sec_dev=sec_d)
    if n_missing and n_max is None:
        rec["status"] = "incomplete"
        rec["first_mismatch"] = rec["first_mismatch"] or f"{n_missing} of {budget // every} stage-2 exports are missing"
    if n_missing and n_max is None:
        rec["ok"] = False
    elif c_off and rows and n_max is None:
        try:
            g = json.load(open(run_dir / "gates.json"))
            last = rows[-1]
            mv = g["metric_values"]
            rep = g["reported"]["end_of_stage2"]["final"]
            rec["last_export_update"] = int(last["update"])
            rec["last_final_eta_equals_freeze"] = bool(last["final_eta_T_over_dw"] == float(mv["eta_final"]))
            rec["last_dev_eta_equals_freeze"] = bool(last["dev_eta_T_over_dw"] == float(mv["eta_dev"]))
            rec["last_final_peak_equals_freeze"] = bool(
                last["final_stage2_peak_rel_err_abs"] == float(rep["stage2_peak_rel_err_abs"]))
        except Exception as exc:  # noqa: BLE001
            rec["first_mismatch"] = rec["first_mismatch"] or f"freeze record unreadable: {type(exc).__name__}: {exc}"
            rec["last_final_eta_equals_freeze"] = rec["last_dev_eta_equals_freeze"] = rec["last_final_peak_equals_freeze"] = False
        rec["ok"] = bool(rec["n_mismatch"] == 0 and rec["n_compared"] > 0 and rec["last_final_eta_equals_freeze"]
                         and rec["last_dev_eta_equals_freeze"] and rec["last_final_peak_equals_freeze"])
    elif c_off:
        rec["ok"] = bool(rec["n_mismatch"] == 0 and rec["n_compared"] > 0)
    else:
        rec["ok"] = bool(rows)
    return rows, rec


def _worker(args: Tuple[str, str, int, int, bool]) -> Tuple[Tuple[str, int, int], List[Dict[str, Any]], Dict[str, Any]]:
    run_dir, arm, q, seed, c_off = args
    import torch
    torch.set_num_threads(1)
    rows, rec = curve_run(Path(run_dir), arm, q, seed, c_off)
    return (arm, q, seed), rows, rec


def write_csv(path: Path, cols: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Write ``rows`` (a missing key is an empty cell); an existing file is refused."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols), restval="")
        w.writeheader()
        w.writerows(rows)


def run_all(pilot_root: Path, out: Path, arms: Sequence[str], qs: Sequence[int], seeds: Sequence[int], workers: int,
            c_off: bool = True) -> Dict[str, Any]:
    """Curves of every (arm, q, seed) under ``pilot_root``; returns the info dict (also written to ``curves_info.json``)."""
    jobs = [(str(Path(pilot_root) / f"q{q}" / f"seed{s}" / arm), arm, q, s, c_off) for arm in arms for q in qs for s in seeds]
    t0 = time.perf_counter()
    results: Dict[Tuple[str, int, int], Tuple[List[Dict[str, Any]], Dict[str, Any]]] = {}
    if workers <= 1:
        for j in jobs:
            k, rows, rec = _worker(j)
            results[k] = (rows, rec)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_worker, j) for j in jobs]
            for fut in as_completed(futs):
                k, rows, rec = fut.result()
                results[k] = (rows, rec)
    all_rows: List[Dict[str, Any]] = []
    recs: List[Dict[str, Any]] = []
    for arm in arms:
        for q in qs:
            for s in seeds:
                rows, rec = results[(arm, q, s)]
                if rows:
                    write_csv(out / f"q{q}" / f"seed{s}" / f"{arm}.csv", CURVE_COLS, rows)
                all_rows.extend(rows)
                recs.append(rec)
    write_csv(out / "curves_all.csv", CURVE_COLS, all_rows)
    write_csv(out / "c_off.csv", C_OFF_COLS, recs)
    info = {"tool": "tools/ms/r4_curves.py", "pilot_root": str(pilot_root), "arms": list(arms), "qs": list(qs),
            "seeds": list(seeds), "workers": workers, "c_off": bool(c_off), "n_runs": len(recs),
            "n_ok": sum(1 for r in recs if r["ok"]), "n_points": len(all_rows),
            "wall_sec": time.perf_counter() - t0,
            "eval_sec_final_total": sum(_f(r.get("eval_sec_final")) for r in recs if r.get("status") == "ok"),
            "eval_sec_dev_total": sum(_f(r.get("eval_sec_dev")) for r in recs if r.get("status") == "ok"),
            "note": "evaluation only: the closed form enters g2_at_0 and the peak / RMSE errors, no run decision"}
    with open(out / "curves_info.json", "w") as f:
        json.dump(info, f, indent=1)
    return info


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    p.add_argument("--pilot-root", required=True, help="the root holding q*/seed*/<arm>")
    p.add_argument("--out", required=True, help="output directory (absent or empty)")
    p.add_argument("--arms", nargs="+", default=None, help="default: the seven arms of MS-R4")
    p.add_argument("--qs", type=int, nargs="+", default=[50, 60])
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="default: 10511-10530")
    p.add_argument("--workers", type=int, default=20, help=f"at most {MAX_WORKERS}")
    p.add_argument("--no-c-off", action="store_true", help="final tier only (no development-tier comparison)")
    a = p.parse_args(argv)
    if not 1 <= a.workers <= MAX_WORKERS:
        p.error(f"--workers must be in 1..{MAX_WORKERS}; got {a.workers}")
    import r4_configs as rc
    arms = list(a.arms) if a.arms else list(rc.ARMS)
    seeds = list(a.seeds) if a.seeds else list(rc.SEEDS)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        p.error(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    info = run_all(Path(a.pilot_root), out, arms, a.qs, seeds, a.workers, not a.no_c_off)
    print(json.dumps({k: info[k] for k in ("n_runs", "n_ok", "n_points", "wall_sec")}, indent=1))
    return 0 if info["n_ok"] == info["n_runs"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
