#!/usr/bin/env python3
"""T3-R3 dirty-tree verification, part 3: clean-tree re-runs of selected dirty-flagged runs (verification only, never counted) and a bit-for-bit
comparison with each original. Run from the repository root inside tmux.

For each selected run: copy its run_config.json to verify_clean_tree/q{q}/seed{s}/{arm}/, run run/run_ms_stagewise.py on it (single-threaded),
compare with the original by ``tools/t3/compare_runs.py bw``, by the file sets, by attempt1/ (weight exports, CSVs without wall-clock columns, bin maps,
state) and retry.json (all fields except wall-clock / cpu seconds) where present, and by the small JSON records; the re-run manifest must record
the same commit and a clean tree."""
import csv, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools" / "ms"))
import launch_ms_r1 as L1  # noqa: E402

BASE = ROOT / "results/t3_r3/dev"
OUT = BASE / "verify_clean_tree"
EXPECT_HEAD = "b565f0e8"


def arrays_equal(x, y):
    return x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y, equal_nan=x.dtype.kind in "fc")


def npz_equal(a, b):
    za, zb = np.load(a), np.load(b)
    return za.files == zb.files and all(arrays_equal(za[k], zb[k]) for k in za.files)


def tree_equal(x, y):
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
    if isinstance(x, np.ndarray):
        return isinstance(y, np.ndarray) and arrays_equal(x, y)
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(tree_equal(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(tree_equal(p, q) for p, q in zip(x, y))
    return x == y


def csv_rows(p, drop):
    with open(p, newline="") as f:
        return [{k: v for k, v in r.items() if k not in drop} for r in csv.DictReader(f)]


WALL = ("wall", "cpu", "seconds", "_sec", "sec_")


def json_diff(a, b, path=""):
    """Paths at which two JSON values differ, wall-clock / cpu-second fields and the run-location fields skipped."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f"{path}/{k}"
            if any(w in k for w in WALL):
                continue
            if k not in a or k not in b:
                out.append(p + " (missing)")
            else:
                out += json_diff(a[k], b[k], p)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            out += json_diff(x, y, f"{path}[{i}]")
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and a != a and b != b):          # NaN equals NaN
        out.append(path)
    return out


def files_of(d):
    return sorted(p.relative_to(d).as_posix() for p in Path(d).rglob("*") if p.is_file() and p.name != "run.log")


def compare_attempt1(orig, new):
    res = {}
    ao, an = orig / "attempt1", new / "attempt1"
    res["weights"] = all(npz_equal(f, an / "weights" / f.name) for f in sorted((ao / "weights").glob("u*.npz"))) and \
        sorted(p.name for p in (ao / "weights").glob("u*.npz")) == sorted(p.name for p in (an / "weights").glob("u*.npz"))
    res["n_weights"] = len(list((ao / "weights").glob("u*.npz")))
    res["binmaps"] = npz_equal(ao / "ms_binmaps_stage3.npz", an / "ms_binmaps_stage3.npz")
    res["updates_csv"] = csv_rows(ao / "ms_updates_stage3.csv", {"update_wall_sec"}) == csv_rows(an / "ms_updates_stage3.csv", {"update_wall_sec"})
    res["checks_csv"] = csv_rows(ao / "ms_checks_stage3.csv", {"verifier_sec"}) == csv_rows(an / "ms_checks_stage3.csv", {"verifier_sec"})
    so = torch.load(ao / "state_attempt1_end.pt", weights_only=False, map_location="cpu")
    sn = torch.load(an / "state_attempt1_end.pt", weights_only=False, map_location="cpu")
    res["state"] = bool(tree_equal(so, sn))
    res["record_json_diff"] = json_diff(json.load(open(ao / "attempt1_record.json")), json.load(open(an / "attempt1_record.json")))
    th = json_diff(json.load(open(ao / "train_history_stage3.json")), json.load(open(an / "train_history_stage3.json")))
    res["train_history_diff_n"] = len(th)
    res["ok"] = bool(res["weights"] and res["binmaps"] and res["updates_csv"] and res["checks_csv"] and res["state"] and not res["record_json_diff"])
    return res


def run_and_compare(job):
    t0 = time.perf_counter()
    orig, new = Path(job["orig"]), Path(job["new"])
    cfg = new / "run_config.json"
    if not job.get("compare_only"):
        new.mkdir(parents=True, exist_ok=False)
        cfg.write_text((orig / "run_config.json").read_text())
    cmd = [sys.executable, "-B", "-u", str(ROOT / "run" / "run_ms_stagewise.py"), "--config", str(cfg), "--out-dir", str(new)]
    res = L1.run_job(cmd, str(new), L1.child_env()) if not job.get("compare_only") else {"returncode": 0}
    wall = time.perf_counter() - t0
    rec = {"run": json.load(open(cfg))["run"], "q": job["q"], "seed": job["seed"], "arm": job["arm"], "why": job["why"], "orig": str(orig), "new": str(new),
           "returncode": res.get("returncode"), "wall_sec": wall}
    if res.get("returncode") != 0:
        rec["ok"] = False
        return rec
    cmp_ = subprocess.run([sys.executable, str(ROOT / "tools/t3/compare_runs.py"), "bw", str(orig), str(new)], capture_output=True, text=True)
    rec["compare_runs_returncode"] = cmp_.returncode
    rec["compare_runs_stdout"] = cmp_.stdout.strip().splitlines()
    (OUT / f"compare_{orig.parts[-4]}_{orig.parts[-3]}_{job['arm']}.txt").write_text(cmp_.stdout + cmp_.stderr)
    fo, fn = files_of(orig), files_of(new)
    rec["file_sets_equal"] = fo == fn
    rec["file_set_difference"] = sorted(set(fo) ^ set(fn))
    mo, mn = json.load(open(orig / "manifest.json")), json.load(open(new / "manifest.json"))
    rec["orig_manifest_clean_tree"], rec["new_manifest_clean_tree"] = mo["clean_tree"], mn["clean_tree"]
    rec["orig_commit8"], rec["new_commit8"] = mo["commit"][:8], mn["commit"][:8]
    rec["init_state_sha256_equal"] = mo["init_state_sha256"] == mn["init_state_sha256"]
    rec["json_diffs"] = {n: [p for p in json_diff(json.load(open(orig / n)), json.load(open(new / n))) if p != "/clean_tree"]
                         for n in ("gates.json", "rule_log.json", "drift_test.json", "ms_run_summary.json") if (orig / n).is_file()}
    rec["fired"] = (orig / "retry.json").is_file()
    if rec["fired"]:
        rec["attempt1"] = compare_attempt1(orig, new)
        rec["retry_json_diff"] = json_diff(json.load(open(orig / "retry.json")), json.load(open(new / "retry.json")))
    ok = cmp_.returncode == 0 and rec["file_sets_equal"] and rec["new_manifest_clean_tree"] is True and rec["new_commit8"] == EXPECT_HEAD and rec["init_state_sha256_equal"]
    ok = ok and not any(rec["json_diffs"].values())
    if rec["fired"]:
        ok = ok and rec["attempt1"]["ok"] and not rec["retry_json_diff"]
    rec["ok"] = bool(ok)
    return rec


def main():
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", ".", ":(exclude)results"], capture_output=True, text=True).stdout.strip()
    if not head.startswith(EXPECT_HEAD) or dirty:
        print("PRECONDITION FAILED", head, repr(dirty)); return 2
    sel = json.load(open(OUT / "boundary.json"))["selection"]
    wanted = []
    for r in sel["fired_dirty_bwA"]: wanted.append((r, "dirty-flagged bwA run whose trigger fired"))
    for r in sel["fired_dirty_bwAB"]: wanted.append((r, "dirty-flagged bwAB run whose trigger fired"))
    for r in sel["fired_dirty_diagnostic"]: wanted.append((r, "dirty-flagged diagnostic run whose trigger fired"))
    for q, r in sel["nonfired_dirty_bwAB_lowest_per_q"].items():
        if r: wanted.append((r, f"lowest-numbered dirty-flagged bwAB run at q={q} whose trigger did not fire"))
    for q, r in sel["nonfired_dirty_bwA_lowest_per_q"].items():
        if r: wanted.append((r, f"ADDITION: lowest-numbered dirty-flagged bwA run at q={q} whose trigger did not fire"))
    jobs = []
    for run, why in wanted:
        _, qs, ss, arm = run.split("_"); q, s = int(qs[1:]), int(ss[1:])
        kind = "diagnostic" if s == 60518 else "dev"
        jobs.append({"q": q, "seed": s, "arm": arm, "why": why, "orig": BASE.parent / kind / f"q{q}" / f"seed{s}" / arm,
                     "new": OUT / f"q{q}" / f"seed{s}" / arm})
    if "--compare-only" in sys.argv:
        for j in jobs: j["compare_only"] = True
    print(f"{len(jobs)} re-runs at HEAD {head[:8]}, tree clean outside results/", flush=True)
    for j in jobs: print(" ", j["why"], j["orig"].relative_to(ROOT), flush=True)
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        recs = list(pool.map(run_and_compare, jobs))
    json.dump(recs, open(OUT / ("rerun_results_final.json" if "--compare-only" in sys.argv else "rerun_results.json"), "w"), indent=1, default=str)
    for r in recs:
        print(r["run"], "returncode", r["returncode"], "ok", r["ok"], "wall", round(r["wall_sec"]), flush=True)
    return 0 if all(r["ok"] for r in recs) else 1


if __name__ == "__main__":
    sys.exit(main())
