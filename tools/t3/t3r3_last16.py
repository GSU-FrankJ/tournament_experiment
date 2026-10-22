#!/usr/bin/env python3
"""Last-16-exports evidence for the class (V) of D2 (round T3-R3): for every failing development run, ``eta_3``, ``rmse_pos_T_over_eT0`` and
``tail_mean_T_over_eT0`` of its stage-3 exports ``u01225`` ... ``u01600`` (final tier, ``tools/t3/t3r2_audit.a8_one``: the export's network at
every stage), and where the run's own failing value sits among them.

The frozen export ``u01600`` must reproduce the run's evaluated value (``analysis/per_run.csv``) to 1e-9; otherwise the script exits 1. Read-only
on the runs; writes ``<out>/last16.csv`` (one row per run and export) and ``<out>/last16_summary.csv`` (one row per failing run and failing
quantity). The closed form (the tent) enters the reported columns only.

Usage:
    python tools/t3/t3r3_last16.py --root $PWD/results/t3_r3 --per-run results/t3_r3/dev/analysis/per_run.csv \
        --out results/t3_r3/dev/analysis --workers 24
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import csv
import multiprocessing as mp
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import t3r2_audit as au  # noqa: E402
import t3r3_launch as lt  # noqa: E402

QUANTITIES = (("ga_eta", "eta_3_over_dw", "eta_T_over_dw", 0.005), ("ga_rmse", "rmse_pos_T_over_eT0", "rmse_pos_T_over_eT0", 0.05),
              ("ga_tail", "tail_mean_T_over_eT0", "tail_mean_T_over_eT0", 0.02))


def _one(args: Any) -> Dict[str, Any]:
    return au.a8_one(args)


def failing_runs(per_run: Path) -> List[Dict[str, str]]:
    """The failing development runs of the evaluation table."""
    with open(per_run, newline="") as f:
        return [r for r in csv.DictReader(f) if r["run_kind"] == "dev" and r["passed"] != "True"]


def rank_from_top(values: Sequence[float], value: float) -> int:
    """1 + the number of exports with a strictly larger value."""
    return 1 + sum(1 for v in values if v > value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R3 last-16-exports evidence", allow_abbrev=False)
    ap.add_argument("--root", default=str(lt.DEFAULT_ROOT))
    ap.add_argument("--per-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8, help="at most 40")
    a = ap.parse_args(argv)
    if not 1 <= a.workers <= 40:
        ap.error("--workers must be in 1..40")
    root = lt.require_absolute(ap, "--root", a.root)
    fails = failing_runs(Path(a.per_run))
    jobs = [{"run_dir": str(root / "dev" / f"q{r['q']}" / f"seed{r['seed']}" / r["arm"]), "q": int(r["q"]), "seed": int(r["seed"]), "run": r["run"]} for r in fails]
    items = [(j, u) for j in jobs for u in au.LAST16]
    with ProcessPoolExecutor(max_workers=min(a.workers, len(items)), mp_context=mp.get_context("spawn")) as pool:
        res = list(pool.map(_one, items))
    rows = []
    for (j, u), r in zip(items, res):
        rows.append({"run": j["run"], "q": j["q"], "seed": j["seed"], "update": u, "eta_3_over_dw": r["eta_3_over_dw"],
                     "rmse_pos_T_over_eT0": r["rmse_pos_T_over_eT0"], "tail_mean_T_over_eT0": r["tail_mean_T_over_eT0"], "valid": r["valid"]})
    out = Path(a.out)
    with open(out / "last16.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary, bad = [], []
    for r in fails:
        mine = [x for x in rows if x["run"] == r["run"]]
        final = next(x for x in mine if x["update"] == 1600)
        for flag, col, pcol, limit in QUANTITIES:
            if r[flag] == "True":
                continue                                    # the gate on this quantity passed
            vals = [x[col] for x in mine]
            ref = float(r[pcol])
            if abs(final[col] - ref) > 1e-9:
                bad.append((r["run"], col, final[col], ref))
            summary.append({"run": r["run"], "quantity": col, "limit": limit, "final_value": ref, "u01600_value": final[col], "last16_min": min(vals),
                            "last16_median": statistics.median(vals),
                            "last16_max": max(vals), "n_above_limit": sum(1 for v in vals if v > limit), "final_rank_from_top": rank_from_top(vals, final[col])})
    with open(out / "last16_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"{len(fails)} failing run(s), {len(items)} exports -> {out / 'last16.csv'}; mismatches with per_run.csv: {len(bad)}", flush=True)
    for b in bad:
        print("MISMATCH", b, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
