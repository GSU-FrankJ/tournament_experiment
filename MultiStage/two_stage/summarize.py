#!/usr/bin/env python3
"""Recount archived all-seed tables, or summarize a newly reproduced T2 batch."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

from run_experiment import COHORTS, PROJECT

SOURCE = PROJECT / "experiments" / COHORTS["restarts"] / "summarize_restarts.py"
spec = importlib.util.spec_from_file_location("t2_restart_metrics", SOURCE)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def summarize_rows(rows):
    by_q = {}
    for q in sorted({r["q"] for r in rows}):
        selected = [r for r in rows if r["q"] == q]
        n = len(selected)
        candidates = sum(r["candidate"] for r in selected)
        successes = sum(r["joint_certified"] for r in selected)
        by_q[str(q)] = {
            "candidate_discovery": metrics.rate(candidates, n),
            "conditional_joint_certification": metrics.rate(successes, candidates),
            "end_to_end": metrics.rate(successes, n),
            "operation_errors": sum(r["operation"] != "done" for r in selected),
        }
    return by_q


def archived_rows(cohort):
    root = PROJECT / "experiments" / COHORTS[cohort]
    path = root / ("formal_results.json" if cohort in ("confirmation", "e1") else "summary.json")
    rows = json.loads(path.read_text())["rows"]
    if cohort in ("confirmation", "e1"):
        rows = [dict(row, candidate=row["search_success"],
                     joint_certified=row["final_joint_pass"]) for row in rows]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", choices=(*COHORTS, "all"),
                        help="Read compact published outcomes; print recalculated rates without writing files.")
    source.add_argument("--results-root", type=Path, help="Batch directory created by run_experiment.py.")
    args = parser.parse_args()
    if args.archive:
        choices = COHORTS if args.archive == "all" else [args.archive]
        output = {name: summarize_rows(archived_rows(name)) for name in choices}
        if args.archive in ("all", "restarts"):
            manifest = json.loads((PROJECT / "experiments" / COHORTS["restarts"] / "manifest.json").read_text())
            restart = metrics.summarize(manifest, archived_rows("restarts"))
            output["restart_groups"] = restart["by_k"]
        if args.archive == "all":
            output["historical_confirmation_E1_q50"] = summarize_rows(
                [r for name in ("confirmation", "e1") for r in archived_rows(name) if r["q"] == 50])
            output["descriptive_restart_plus_precision"] = summarize_rows(
                archived_rows("restarts") + archived_rows("precision"))
        print(json.dumps(output, indent=2, allow_nan=False))
        return
    root = args.results_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("publication_smoke"):
        parser.error("Smoke outputs are flow checks and have no formal success rate.")
    launch = json.loads((root / "launch_status.json").read_text())
    launches = {row["run"]: row for row in launch["runs"]}
    rows = [metrics.read_run(row, launches.get(row["run"])) for row in manifest["runs"]]
    result = {"by_q": summarize_rows(rows), "rows": rows,
              "batch_wall_seconds": launch.get("wall_sec"),
              "source_manifest": manifest.get("publication_source_manifest")}
    assignments = manifest.get("restart_evaluation", {}).get("groups")
    if assignments and [s for g in assignments for s in g["seeds"]] == [r["seed"] for r in manifest["runs"]]:
        result["restart_evaluation"] = metrics.summarize(manifest, rows)
    (root / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (root / "runs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result["by_q"], indent=2))


if __name__ == "__main__":
    main()
