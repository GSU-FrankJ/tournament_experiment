#!/usr/bin/env python3
"""T3-R3 dirty-tree verification, parts 1 and 2 (reads only): the clean/dirty boundary and the bwA-bw pairs across it.
Writes results/t3_r3/dev/verify_clean_tree/{boundary.csv,boundary.json,pairs_across_boundary.csv,selection.json}."""
import csv, json, datetime as dt, subprocess, sys
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "results/t3_r3/dev/verify_clean_tree"
CUT = dt.datetime(2026, 10, 22, 2, 21, 15, 679567, tzinfo=dt.timezone.utc)      # birth time of results_tables.py (stat)
checks = json.load(open(ROOT / "results/t3_r3/dev/launch_checks_absroot.json"))
birth = subprocess.run(["stat", "-c", "%w", "reports/t3/r3/report_scripts/results_tables.py"], capture_output=True, text=True).stdout.strip()

def ts(s):
    if isinstance(s, str):
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)      # status.json start_time is UTC without a zone mark
    return dt.datetime.fromtimestamp(float(s), dt.timezone.utc)

rows = []
for r in checks["runs"]:
    d = ROOT / "results/t3_r3" / ("diagnostic" if r["run_kind"] == "diagnostic" else "dev") / f"q{r['q']}" / f"seed{r['seed']}" / r["arm"]
    man = json.load(open(d / "manifest.json")); st = json.load(open(d / "status.json"))
    t0 = ts(st["start_time"])
    rows.append({"run": r["run"], "kind": r["run_kind"], "arm": r["arm"], "q": r["q"], "seed": r["seed"], "start_utc": t0.isoformat(timespec="milliseconds"),
                 "manifest_dirty": man["git"]["dirty"], "manifest_commit8": man["git"]["commit"][:8],
                 "started_after_file": t0 > CUT, "fired": r["fired"], "would_fire": r["would_fire"]})
rows.sort(key=lambda x: x["start_utc"])
with open(OUT / "boundary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
clean = [x for x in rows if not x["manifest_dirty"]]; dirty = [x for x in rows if x["manifest_dirty"]]
disagree = [x["run"] for x in rows if x["manifest_dirty"] != x["started_after_file"]]
summary = {"results_tables_py_birth_utc": birth, "cut_utc": CUT.isoformat(), "n_runs": len(rows), "n_clean": len(clean), "n_dirty": len(dirty),
           "last_clean_start": clean[-1]["start_utc"], "first_dirty_start": dirty[0]["start_utc"],
           "runs_whose_flag_disagrees_with_the_boundary": disagree, "commits": sorted({x["manifest_commit8"] for x in rows}),
           "dirty_by_arm": {a: sum(1 for x in dirty if x["arm"] == a) for a in ("bw", "bwA", "bwAB")},
           "clean_by_arm": {a: sum(1 for x in clean if x["arm"] == a) for a in ("bw", "bwA", "bwAB")},
           "dirty_by_kind": {k: sum(1 for x in dirty if x["kind"] == k) for k in ("dev", "diagnostic")}}
# pairs bwA-bw across the boundary
by = {(x["kind"], x["q"], x["seed"], x["arm"]): x for x in rows}
res = {(r["run_kind"], r["q"], r["seed"], r["arm"]): r for r in checks["runs"]}
prs = []
for (kind, q, s, arm), x in by.items():
    if arm != "bwA" or kind != "dev":
        continue
    b = by[(kind, q, s, "bw")]; c = res[(kind, q, s, "bwA")]
    prs.append({"q": q, "seed": s, "bw_dirty": b["manifest_dirty"], "bwA_dirty": x["manifest_dirty"], "bwA_fired": c["fired"], "pair_ok": c["pair_ok"],
                "bit_identical_full": bool(c["pair_ok"] and not c["fired"]), "attempt1_identical_to_bw_stage3": bool(c["pair_ok"] and c["fired"]),
                "flags_differ": b["manifest_dirty"] != x["manifest_dirty"]})
with open(OUT / "pairs_across_boundary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(prs[0])); w.writeheader(); w.writerows(prs)
summary["bwA_bw_pairs"] = len(prs)
summary["pairs_flags_differ"] = sum(p["flags_differ"] for p in prs)
summary["pairs_bit_identical_full_and_flags_differ"] = sum(p["bit_identical_full"] and p["flags_differ"] for p in prs)
summary["pairs_attempt1_identical_and_flags_differ"] = sum(p["attempt1_identical_to_bw_stage3"] and p["flags_differ"] for p in prs)
summary["pairs_bit_identical_full_total"] = sum(p["bit_identical_full"] for p in prs)
summary["pairs_attempt1_identical_total"] = sum(p["attempt1_identical_to_bw_stage3"] for p in prs)
summary["pairs_not_ok"] = [(p["q"], p["seed"]) for p in prs if not p["pair_ok"]]
# selection of re-runs
dirty_by = lambda a, fired=None: [x for x in dirty if x["arm"] == a and x["kind"] == "dev" and (fired is None or x["fired"] == fired)]
sel = {"fired_dirty_bwA": [x["run"] for x in dirty_by("bwA", True)], "fired_dirty_bwAB": [x["run"] for x in dirty_by("bwAB", True)],
       "fired_dirty_diagnostic": [x["run"] for x in dirty if x["kind"] == "diagnostic" and x["fired"]],
       "nonfired_dirty_bwAB_lowest_per_q": {}, "nonfired_dirty_bwA_lowest_per_q": {}}
for q in (50, 60):
    c = sorted((x for x in dirty_by("bwAB", False) if x["q"] == q), key=lambda x: x["seed"])
    sel["nonfired_dirty_bwAB_lowest_per_q"][str(q)] = c[0]["run"] if c else None
    c = sorted((x for x in dirty_by("bwA", False) if x["q"] == q), key=lambda x: x["seed"])
    sel["nonfired_dirty_bwA_lowest_per_q"][str(q)] = c[0]["run"] if c else None
summary["selection"] = sel
json.dump(summary, open(OUT / "boundary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
