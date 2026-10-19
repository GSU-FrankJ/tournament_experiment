#!/usr/bin/env python3
"""Evaluation of the 40 confirmation runs of round T3-R2 (wave ``t3c``).

Builds the 40 jobs (``<root>/q{q}/seed{s}/bw``, run names ``t3r2_q{q}_s{s}_bw``) and calls ``tools/t3/t3_eval.py``'s
``evaluate_run`` (last iterate, final and development tier, ``bw`` format), ``write_rows`` and ``replay_mismatches`` unchanged:
``t3_eval.py``'s own CLI accepts only T3-R1's waves.

Evaluation problems (D4, "Failure counting"): a row with an evaluation error, or with ``valid`` = False on either tier, is
evaluated once more with the same locked evaluator on the same weights; the re-run is reported in ``eval_reruns.json`` and
its row replaces the first one if it has no error and both tiers valid (a run is never re-trained). The evaluation consumes no
random number and changes no run file.

Outputs in --out-dir: ``per_run.csv``, ``eval_reruns.json`` and ``arrays/{run}.npz`` (final-tier arrays).

Usage:
    python tools/t3/t3c_eval.py --root results/t3_r2/confirmation --out-dir results/t3_r2/confirmation/eval --workers 40
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
import t3c_launch as cl  # noqa: E402

WAVE = cl.WAVE


def plan_jobs(root: Path, arrays_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """The 40 ``t3c`` jobs in q-major, seed order."""
    jobs = []
    for q in cl.QS:
        for seed in cl.SEEDS:
            jobs.append({"wave": WAVE, "arm": cl.ARM, "q": q, "seed": seed, "T": 3, "run": cl.run_name(q, seed),
                         "run_dir": str(cl.run_dir(Path(root), q, seed)), "kind": "bw",
                         "arrays_dir": str(arrays_dir) if arrays_dir else None})
    return jobs


def safe_evaluate_run(job: Dict[str, Any]) -> Dict[str, Any]:
    """``t3_eval.evaluate_run`` with every other exception of the evaluation recorded as an evaluation error of that run (D4: an
    evaluation error counts as a failed run; it must not abort the evaluation of the other runs)."""
    try:
        return te.evaluate_run(job)
    except Exception as exc:  # noqa: BLE001 - recorded as the run's evaluation error
        st = te.status_of(Path(job["run_dir"]))
        return {"wave": job["wave"], "arm": job["arm"], "q": int(job["q"]), "seed": int(job["seed"]), "T": int(job["T"]),
                "run": job["run"], "run_dir": str(job["run_dir"]), "kind": job["kind"], "status": st.get("state", "missing"),
                "exit_code": st.get("exit_code"), "global_rng_ok": te.global_rng_ok(Path(job["run_dir"]), job["kind"]),
                "eval_error": f"{type(exc).__name__}: {exc}"}


def needs_rerun(row: Dict[str, Any]) -> bool:
    """A finished run whose evaluation raised an error or is invalid on either tier."""
    if row.get("status") != "done" or row.get("exit_code") != 0:
        return False
    return bool(row.get("eval_error")) or row.get("valid") is not True or row.get("dev_valid") is not True


def usable(row: Dict[str, Any]) -> bool:
    """No evaluation error, both tiers valid."""
    return not row.get("eval_error") and row.get("valid") is True and row.get("dev_valid") is True


def evaluate_all(root: Path, out_dir: Path, workers: int) -> Dict[str, Any]:
    """Evaluate the 40 runs; re-run the evaluation of every row that needs it; write the CSV and the re-run record."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = plan_jobs(root, out_dir / "arrays")
    rows = te.run_parallel(safe_evaluate_run, jobs, workers)
    reruns: List[Dict[str, Any]] = []
    for i, (job, row) in enumerate(zip(jobs, rows)):
        if not needs_rerun(row):
            continue
        again = safe_evaluate_run(job)
        ok = usable(again)
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
    ap = argparse.ArgumentParser(description="T3-R2 confirmation evaluation", allow_abbrev=False)
    ap.add_argument("--root", default=str(cl.DEFAULT_ROOT))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8, help="at most 40")
    a = ap.parse_args(argv)
    if not 1 <= a.workers <= 40:
        ap.error("--workers must be in 1..40")
    res = evaluate_all(Path(a.root), Path(a.out_dir), a.workers)
    print(f"{len(res['rows'])} runs -> {Path(a.out_dir) / 'per_run.csv'}; {len(res['reruns'])} evaluation re-run(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
