#!/usr/bin/env python3
"""C-REPRO of round T3-R4 (prompt 2.3): the unchanged ``bwA`` path of ``run/run_ms_stagewise.py`` at the final code commit.

Re-runs two of T3-R3's ``bwA`` development runs, (q = 50, seed 70508), whose trigger fired, and (q = 60, seed 70501), whose trigger did not,
from their own ``run_config.json`` into ``<out>/q{q}/seed{s}/bwA`` (default ``results/t3_r4/repro``), single-threaded, and compares each
with its original in the T3-R3 untracked root:

  * ``tools/t3/compare_runs.py bw`` (weight exports, freeze arrays, continuation tables, bin maps, the CSVs without their wall-clock
    columns, the final networks, frozen stages and stream states);
  * the set of files of the two runs (``run.log`` aside);
  * for the run that fired: ``attempt1/`` (the 64 exports, the stage-3 rows and checks without wall-clock columns, the bin maps, the training
    state) and ``retry.json`` (every field except wall-clock and cpu seconds).

Both must be bit-identical; a mismatch is a stop (exit code 1). The JSON records of the runs (gates, rule log, drift test, summary) are
compared too and reported, but the fields that legitimately differ (the commit, the clean-tree flag) are left out. Writes
``<out>/compare_q{q}_s{seed}.json`` and ``.txt``. Run inside tmux:

    python tools/t3/t3r4_repro.py --t3r3-root <results/t3_r3> --out results/t3_r4/repro
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import launch_ms_r1 as L1  # noqa: E402

RUNS = ((50, 70508, True), (60, 70501, False))                    # (q, seed, the trigger fired in the original)
DEFAULT_T3R3 = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-diagnosis-25cb13/results/t3_r3")
DEFAULT_OUT = ROOT / "results" / "t3_r4" / "repro"
WALL_KEYS = ("wall", "cpu", "seconds", "_sec", "sec_")
SKIP_KEYS = ("commit", "short", "dirty", "clean_tree")           # the original and the re-run differ in the commit and the tree flag only
WALL_UPDATES, WALL_CHECKS = {"update_wall_sec"}, {"verifier_sec"}


def jobs(t3r3_root: Path, out: Path) -> List[Dict[str, Any]]:
    """The two re-runs: original directory, new directory."""
    return [{"q": q, "seed": s, "fired": fired, "orig": Path(t3r3_root) / "dev" / f"q{q}" / f"seed{s}" / "bwA",
             "new": Path(out) / f"q{q}" / f"seed{s}" / "bwA"} for q, s, fired in RUNS]


def json_diff(a: Any, b: Any, path: str = "", skip: Sequence[str] = SKIP_KEYS) -> List[str]:
    """Paths at which two JSON values differ; wall-clock / cpu-second fields and the ``skip`` keys are left out, NaN equals NaN."""
    out: List[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in skip or any(w in k for w in WALL_KEYS):
                continue
            p = f"{path}/{k}"
            if k not in a or k not in b:
                out.append(p + " (missing)")
            else:
                out += json_diff(a[k], b[k], p, skip)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            out += json_diff(x, y, f"{path}[{i}]", skip)
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and a != a and b != b):
        out.append(path)
    return out


def arrays_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """Exact equality (NaN equals NaN for floating arrays; dtype and shape must agree)."""
    return x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y, equal_nan=x.dtype.kind in "fc")


def npz_equal(a: Path, b: Path) -> bool:
    """Whether two ``.npz`` files hold the same arrays under the same names."""
    za, zb = np.load(a), np.load(b)
    return za.files == zb.files and all(arrays_equal(za[k], zb[k]) for k in za.files)


def tree_equal(x: Any, y: Any) -> bool:
    """Exact (bitwise for tensors and arrays) equality of nested dict / list / tuple / tensor structures."""
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
    if isinstance(x, np.ndarray):
        return isinstance(y, np.ndarray) and arrays_equal(x, y)
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(tree_equal(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(tree_equal(p, q) for p, q in zip(x, y))
    return bool(x == y)


def csv_rows(path: Path, drop: set) -> List[Dict[str, str]]:
    """CSV rows without the dropped (wall-clock) columns."""
    with open(path, newline="") as f:
        return [{k: v for k, v in r.items() if k not in drop} for r in csv.DictReader(f)]


def files_of(d: Path) -> List[str]:
    """Relative paths of all files under ``d`` except ``run.log``."""
    return sorted(p.relative_to(d).as_posix() for p in Path(d).rglob("*") if p.is_file() and p.name != "run.log")


def compare_attempt1(orig: Path, new: Path) -> Dict[str, Any]:
    """``attempt1/`` of a re-run against the original: exports, stage-3 rows and checks, bin maps, state, record."""
    ao, an = Path(orig) / "attempt1", Path(new) / "attempt1"
    names_o = sorted(p.name for p in (ao / "weights").glob("u*.npz"))
    names_n = sorted(p.name for p in (an / "weights").glob("u*.npz"))
    res: Dict[str, Any] = {"n_exports": len(names_o), "exports_same_names": names_o == names_n,
                           "exports_identical": names_o == names_n and all(npz_equal(ao / "weights" / n, an / "weights" / n) for n in names_o)}
    res["binmaps"] = npz_equal(ao / "ms_binmaps_stage3.npz", an / "ms_binmaps_stage3.npz")
    res["updates_csv"] = csv_rows(ao / "ms_updates_stage3.csv", WALL_UPDATES) == csv_rows(an / "ms_updates_stage3.csv", WALL_UPDATES)
    res["checks_csv"] = csv_rows(ao / "ms_checks_stage3.csv", WALL_CHECKS) == csv_rows(an / "ms_checks_stage3.csv", WALL_CHECKS)
    so = torch.load(ao / "state_attempt1_end.pt", weights_only=False, map_location="cpu")
    sn = torch.load(an / "state_attempt1_end.pt", weights_only=False, map_location="cpu")
    res["state"] = bool(tree_equal(so, sn))
    res["record_json_diff"] = json_diff(json.loads((ao / "attempt1_record.json").read_text()), json.loads((an / "attempt1_record.json").read_text()))
    res["ok"] = bool(res["exports_identical"] and res["binmaps"] and res["updates_csv"] and res["checks_csv"] and res["state"]
                     and not res["record_json_diff"])
    return res


def run_and_compare(job: Dict[str, Any]) -> Dict[str, Any]:
    """Re-run one run from the original config and compare."""
    t0 = time.perf_counter()
    new, orig = Path(job["new"]), Path(job["orig"])
    new.mkdir(parents=True, exist_ok=False)
    cfg = new / "run_config.json"
    cfg.write_text((orig / "run_config.json").read_text())
    cmd = [sys.executable, "-B", "-u", str(ROOT / "run" / "run_ms_stagewise.py"), "--config", str(cfg), "--out-dir", str(new)]
    res = L1.run_job(cmd, str(new), L1.child_env())
    out = new.parents[2]
    rec: Dict[str, Any] = {"run": json.loads(cfg.read_text())["run"], "q": job["q"], "seed": job["seed"], "fired_in_original": job["fired"],
                           "orig": str(orig), "new": str(new), "returncode": res.get("returncode")}
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", ".", ":(exclude)results"], capture_output=True, text=True).stdout
    rec.update(commit=head, tree_clean_outside_results=not dirty.strip())
    if res.get("returncode") != 0:
        rec.update(ok=False, wall_sec=time.perf_counter() - t0)
    else:
        cmp_ = subprocess.run([sys.executable, str(ROOT / "tools" / "t3" / "compare_runs.py"), "bw", str(orig), str(new)], capture_output=True, text=True)
        (out / f"compare_q{job['q']}_s{job['seed']}.txt").write_text(cmp_.stdout + cmp_.stderr)
        rec["compare_runs_returncode"] = cmp_.returncode
        rec["n_npz"] = int(next((line.split()[0] for line in cmp_.stdout.splitlines() if "npz files compared" in line), 0))
        fo, fn = files_of(orig), files_of(new)
        rec["file_sets_equal"], rec["file_set_difference"] = fo == fn, sorted(set(fo) ^ set(fn))
        rec["fired_in_rerun"] = (new / "retry.json").is_file()
        rec["retry_state_matches"] = rec["fired_in_rerun"] == job["fired"]
        rec["json_diffs"] = {n: json_diff(json.loads((orig / n).read_text()), json.loads((new / n).read_text()))
                             for n in ("gates.json", "rule_log.json", "drift_test.json", "ms_run_summary.json") if (orig / n).is_file()}
        ok = cmp_.returncode == 0 and rec["file_sets_equal"] and rec["retry_state_matches"]
        if job["fired"]:
            rec["attempt1"] = compare_attempt1(orig, new)
            rec["retry_json_diff"] = json_diff(json.loads((orig / "retry.json").read_text()), json.loads((new / "retry.json").read_text()))
            ok = ok and rec["attempt1"]["ok"] and not rec["retry_json_diff"]
        rec.update(ok=bool(ok), wall_sec=time.perf_counter() - t0)
    (out / f"compare_q{job['q']}_s{job['seed']}.json").write_text(json.dumps(rec, indent=1))
    return rec


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R4 C-REPRO", allow_abbrev=False)
    ap.add_argument("--t3r3-root", default=str(DEFAULT_T3R3), help="the T3-R3 results root, an absolute path")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="the output directory, an absolute path")
    a = ap.parse_args(argv)
    for flag, value in (("--t3r3-root", a.t3r3_root), ("--out", a.out)):
        if not Path(value).is_absolute():
            ap.error(f"{flag} must be an absolute path, got {value!r}")
    js = jobs(Path(a.t3r3_root), Path(a.out))
    for j in js:
        if not (j["orig"] / "run_config.json").is_file():
            ap.error(f"{j['orig']}: no run_config.json (a missing original is a stop-and-report)")
    with ThreadPoolExecutor(max_workers=len(js)) as pool:
        recs = list(pool.map(run_and_compare, js))
    print(json.dumps(recs, indent=1))
    return 0 if all(r["ok"] for r in recs) else 1


if __name__ == "__main__":
    sys.exit(main())
