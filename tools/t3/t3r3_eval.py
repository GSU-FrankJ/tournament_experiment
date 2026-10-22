#!/usr/bin/env python3
"""Evaluation of the T3-R3 runs (development wave: 120 runs; diagnostic: 2 runs).

The evaluated policy is the one the scheme freezes: ``t3_eval.load_bw`` reads ``state_end_stage1.pt``, whose ``frozen_stages`` hold the
stage-3 network of the frozen attempt (attempt 2 for a run that fired). ``tools/t3/t3_eval.py``'s ``evaluate_run`` (final and
development tier), ``write_rows`` and ``run_parallel`` are used unchanged, as are the evaluation-problem rules of
``t3c_eval`` (an evaluation error or an invalid evaluation is evaluated once more with the same evaluator on the same weights;
the re-run replaces the first row only if it has no error and both tiers valid; a run is never re-trained).

Outputs in --out-dir: ``per_run.csv``, ``eval_reruns.json`` and ``arrays/{run}.npz`` (final-tier arrays, untracked).

Usage:
    python tools/t3/t3r3_eval.py --root $PWD/results/t3_r3 --which dev --out-dir results/t3_r3/dev/eval --workers 40
    python tools/t3/t3r3_eval.py --root $PWD/results/t3_r3 --which diagnostic --out-dir results/t3_r3/diagnostic/eval
(--root must be absolute.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import t3_eval as te  # noqa: E402
import t3c_eval as ce  # noqa: E402
import t3r3_checks as ck  # noqa: E402
import t3r3_launch as lt  # noqa: E402


def plan_jobs(root: Path, which: str, arrays_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """The jobs of ``which`` (``dev`` = 120 runs, ``diagnostic`` = 2 runs) in the order of ``t3r3_launch.job_list``."""
    if which not in ("dev", "diagnostic"):
        raise ValueError(f"which must be 'dev' or 'diagnostic'; got {which!r}")
    jobs = [j for j in ck.all_jobs(root) if j["run_kind"] == which]
    for j in jobs:
        j["arrays_dir"] = str(arrays_dir) if arrays_dir else None
    return jobs


def evaluate_all(root: Path, which: str, out_dir: Path, workers: int) -> Dict[str, Any]:
    """Evaluate the runs, re-run the evaluation of every row that needs it, write the CSV and the re-run record."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = plan_jobs(root, which, out_dir / "arrays")
    rows = te.run_parallel(ce.safe_evaluate_run, jobs, workers)
    reruns: List[Dict[str, Any]] = []
    for i, (job, row) in enumerate(zip(jobs, rows)):
        if not ce.needs_rerun(row):
            continue
        again = ce.safe_evaluate_run(job)
        ok = ce.usable(again)
        reruns.append({"run": job["run"], "first_eval_error": row.get("eval_error", ""), "first_valid": row.get("valid"),
                       "first_dev_valid": row.get("dev_valid"), "rerun_eval_error": again.get("eval_error", ""),
                       "rerun_valid": again.get("valid"), "rerun_dev_valid": again.get("dev_valid"), "rerun_succeeded": ok})
        if ok:
            rows[i] = again
    te.write_rows(out_dir / "per_run.csv", rows)
    with open(out_dir / "eval_reruns.json", "w") as f:
        json.dump(reruns, f, indent=1)
    return {"rows": rows, "reruns": reruns}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R3 evaluation", allow_abbrev=False)
    ap.add_argument("--root", default=str(lt.DEFAULT_ROOT))
    ap.add_argument("--which", choices=["dev", "diagnostic"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8, help="at most 40")
    a = ap.parse_args(argv)
    if not 1 <= a.workers <= 40:
        ap.error("--workers must be in 1..40")
    root = lt.require_absolute(ap, "--root", a.root)
    res = evaluate_all(root, a.which, Path(a.out_dir), a.workers)
    print(f"{len(res['rows'])} runs -> {Path(a.out_dir) / 'per_run.csv'}; {len(res['reruns'])} evaluation re-run(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
