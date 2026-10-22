#!/usr/bin/env python3
"""C-LEGACY of round T3-R3 (prompt D5): the unchanged ``bw`` path of ``run/run_ms_stagewise.py`` at the final code commit.

Re-runs T3-R2's runs (q = 50, seed 60502) and (q = 60, seed 60518) from their own ``run_config.json`` (no scheme key) into
``<out>/q{q}/seed{s}/bw`` (default ``results/t3_r3/legacy``), single-threaded, and compares each with its original in the T3-R2
untracked root with ``tools/t3/compare_runs.py bw`` (weight exports, freeze arrays, continuation tables, bin maps, the CSVs without
their wall-clock columns, the final networks and stream states). Both must be bit-identical; a mismatch is a stop (exit code 1).
Writes ``<out>/compare_q{q}_s{seed}.json`` and ``.txt``. Run inside tmux:

    python tools/t3/t3r3_legacy.py --t3r2-root <results/t3_r2> --out results/t3_r3/legacy
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))

import launch_ms_r1 as L1  # noqa: E402

RUNS = ((50, 60502), (60, 60518))
DEFAULT_T3R2 = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-audit-6f3a38/results/t3_r2")
DEFAULT_OUT = ROOT / "results" / "t3_r3" / "legacy"


def jobs(t3r2_root: Path, out: Path) -> List[Dict[str, Any]]:
    """The two re-runs: original directory, new directory."""
    return [{"q": q, "seed": s, "orig": Path(t3r2_root) / "confirmation" / f"q{q}" / f"seed{s}" / "bw",
             "new": Path(out) / f"q{q}" / f"seed{s}" / "bw"} for q, s in RUNS]


def run_and_compare(job: Dict[str, Any]) -> Dict[str, Any]:
    """Re-run one run from the original config and compare."""
    t0 = time.perf_counter()
    new, orig = Path(job["new"]), Path(job["orig"])
    new.mkdir(parents=True, exist_ok=False)
    cfg = new / "run_config.json"
    cfg.write_text((orig / "run_config.json").read_text())
    cmd = [sys.executable, "-B", "-u", str(ROOT / "run" / "run_ms_stagewise.py"), "--config", str(cfg), "--out-dir", str(new)]
    res = L1.run_job(cmd, str(new), L1.child_env())
    cmp_ = subprocess.run([sys.executable, str(ROOT / "tools" / "t3" / "compare_runs.py"), "bw", str(orig), str(new)],
                          capture_output=True, text=True)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rec = {"run": json.loads(cfg.read_text())["run"], "q": job["q"], "seed": job["seed"], "orig": str(orig), "new": str(new), "commit": head,
           "returncode": res.get("returncode"), "compare_returncode": cmp_.returncode,
           "ok": bool(res.get("returncode") == 0 and cmp_.returncode == 0), "wall_sec": time.perf_counter() - t0,
           "n_npz": int(next((l.split()[0] for l in cmp_.stdout.splitlines() if "npz files compared" in l), 0))}
    out = new.parents[2]
    (out / f"compare_q{job['q']}_s{job['seed']}.txt").write_text(cmp_.stdout + cmp_.stderr)
    (out / f"compare_q{job['q']}_s{job['seed']}.json").write_text(json.dumps(rec, indent=1))
    return rec


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R3 C-LEGACY", allow_abbrev=False)
    ap.add_argument("--t3r2-root", default=str(DEFAULT_T3R2))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)
    js = jobs(Path(a.t3r2_root), Path(a.out))
    for j in js:
        if not (j["orig"] / "run_config.json").is_file():
            ap.error(f"{j['orig']}: no run_config.json (a missing original is a stop-and-report)")
    with ThreadPoolExecutor(max_workers=len(js)) as pool:
        recs = list(pool.map(run_and_compare, js))
    print(json.dumps(recs, indent=1))
    return 0 if all(r["ok"] for r in recs) else 1


if __name__ == "__main__":
    sys.exit(main())
