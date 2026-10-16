#!/usr/bin/env python3
"""Writes ``reports/t3/r1/run_config_hashes.csv`` (the SHA-256 of every planned ``run_config.json``: 168 rows) and
``reports/t3/r1/arm_config_differences.csv`` (per arm, the keys in which its config differs from its comparator's).

The configs are built by ``tools/t3/t3_launch.py`` (``--plan-csv``): nothing is created or launched. The hash is that of the
``run_config.json`` text, which embeds the absolute output directory; ``run_config_sha256_relpath`` is the hash with the output
directory made relative to the results root's parent (it does not depend on the worktree that launches).

Usage:
    python reports/t3/r1/report_scripts/config_differences.py --v20-root <D9 root>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import t3_configs as tc  # noqa: E402
import t3_launch as tl  # noqa: E402

OUT = ROOT / "reports" / "t3" / "r1"
LABELS = ["arm", "run", "record.run", "record.output_dir"]


def main() -> int:
    """Write the two CSVs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--v20-root", required=True)
    a = ap.parse_args()
    proto = tc.load_protocol()
    params = tc.load_bw_params()
    root = ROOT / "results" / "t3_r1"
    rows = []
    for wave in tc.WAVES:
        jobs = tl.build_jobs(wave, tc.QS, list(tc.WAVE_SEEDS[wave]), None, root, params)
        rows += tl.plan_rows(wave, jobs, root)
    with open(OUT / "run_config_hashes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tl.PLAN_FIELDS)
        w.writeheader()
        w.writerows(rows)
    diffs = []
    for q in tc.QS:
        cfg = {arm: tc.build_config(proto, q, 50501, arm, f"<out>/{arm}", params) for arm in ("bw", "jt", "jt_es")}
        cfg2 = tc.build_config(proto, q, 30501, "jt2", "<out>/jt2", params)
        d = [k for k in tc.config_diff(cfg["jt_es"], cfg["jt"]) if k not in LABELS]
        diffs.append({"arm": "jt_es", "comparator": "jt", "q": q, "purpose": "H3 pair: the same runner; only the budget keys differ",
                      "keys_that_differ": ";".join(d), "labels_that_differ": ";".join(LABELS)})
        for arm in ("jt", "jt_es"):
            ra, rb = dict(cfg["bw"]["record"]), dict(cfg[arm]["record"])
            for r in (ra, rb):
                r.pop("run")
                r.pop("output_dir")
            top = sorted(k for k in set(cfg["bw"]) | set(cfg[arm]) if cfg["bw"].get(k, "<absent>") != cfg[arm].get(k, "<absent>"))
            diffs.append({"arm": arm, "comparator": "bw", "q": q,
                          "purpose": f"{'H1' if arm == 'jt' else 'H2'} pair: different runners (ms_run_config/1 against joint_run_config/1); "
                                     f"embedded records equal apart from run and output_dir: {ra == rb}; "
                                     f"base_commit, protocol_ref, record_sha256, threads_per_process, clamp_likelihood, init_digest equal: "
                                     f"{all(cfg['bw'][k] == cfg[arm][k] for k in ('base_commit', 'protocol_ref', 'record_sha256', 'threads_per_process', 'clamp_likelihood', 'init_digest'))}",
                          "keys_that_differ": ";".join(top), "labels_that_differ": "top-level keys only"})
        v20 = json.load(open(Path(a.v20_root) / f"q{q}" / "seed30501" / "run_config.json"))
        ra, rb = dict(v20["record"]), dict(cfg2["record"])
        keys = sorted(k for k in set(ra) | set(rb) if ra.get(k, "<absent>") != rb.get(k, "<absent>"))
        diffs.append({"arm": "jt2", "comparator": "v20 (recorded run_config.json of seed 30501)", "q": q,
                      "purpose": "H1-T2 pair: the recorded v2.0 config is v2_run_config/1; the embedded v2.0 records are compared",
                      "keys_that_differ": ";".join(f"record.{k}" for k in keys), "labels_that_differ": "run and output_dir only"})
    with open(OUT / "arm_config_differences.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(diffs[0]))
        w.writeheader()
        w.writerows(diffs)
    print(f"{len(rows)} config hashes -> {OUT / 'run_config_hashes.csv'}; {len(diffs)} difference rows -> {OUT / 'arm_config_differences.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
