#!/usr/bin/env python3
"""Checklist over v2 run directories (artifacts present, logs populated, drift test, timing).

Usage: python tools/v2/smoke_checklist.py <run_dir> [<run_dir> ...] --out <csv>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np


def check(d: str) -> dict:
    """One row of checks for a run directory."""
    man = json.load(open(os.path.join(d, "manifest.json")))
    summ = json.load(open(os.path.join(d, "v2_run_summary.json")))
    status = json.load(open(os.path.join(d, "status.json")))
    upd = list(csv.DictReader(open(os.path.join(d, "v2_updates.csv"))))
    ck = list(csv.DictReader(open(os.path.join(d, "v2_checkpoints.csv"))))
    npz = sorted(glob.glob(os.path.join(d, "checkpoints", "u*.npz")))
    ph = man["mode"].split("_")[1]
    z = np.load(npz[-1]) if npz else None
    row = {
        "run_dir": d, "arm": man["arm"], "mode": man["mode"], "status": status["state"],
        "manifest_fields": all(k in man for k in ("git", "base_commit", "flags", "resolved_protocol", "grids",
                                                   "verifier_cadence", "parent_checkpoint", "parent_sha256",
                                                   "seed_namespaces")),
        "git_commit": man["git"]["short"], "git_dirty": man["git"]["dirty"],
        "flags": json.dumps(man["flags"], sort_keys=True), "fixed_budget": man["fixed_budget"],
        "n_updates": len(upd), "n_checkpoints_csv": len(ck), "n_checkpoint_npz": len(npz),
        "adv_stats_logged": all(r["adv_all_std"] != "" for r in upd) and all(r["adv_s1_std"] != "" for r in upd),
        "npz_has_G_pmf_sigma": bool(z is not None and all(k in z.files for k in ("v_t2_G", "v_t2_cand_pmf", "v_t2_sigma_effort", "v_t2_alpha", "v_t2_beta"))),
        "plot": os.path.exists(os.path.join(d, "stage2_final.png")),
        "full_state_ckpt": os.path.exists(os.path.join(d, f"state_end_{ph}.pt")),
        "phase_wall_sec": round(summ["phase_timing"][ph]["wall_sec"], 3),
        "phase_cpu_sec": round(summ["phase_timing"][ph]["process_cpu_sec"], 3),
        "phase_train_update_sec": round(summ["phase_timing"][ph]["by_category_sec"].get("train_update", 0.0), 3),
        "would_have_fired": json.dumps(summ["would_have_fired"][ph]),
        "stage1_status_last": ck[-1]["stage1_status"] if ck else "",
        "Gmax_full_over_dw_last": ck[-1]["Gmax_full_over_dw"] if ck else "",
    }
    if ph == "B":
        t = json.load(open(os.path.join(d, "drift_test.json")))
        row["drift_test"] = json.dumps(t["max_abs_diff_vs_freeze_time"])
        row["drift_test_pass"] = t.get("pass", "n/a (joint)")
        row["parent_sha256_prefix"] = man["parent_sha256"][:12]
        row["adv_norm_rows_used"] = "s1" if all(float(r["adv_used_std"]) == float(r["adv_s1_std"]) for r in upd) else \
            ("all" if all(float(r["adv_used_std"]) == float(r["adv_all_std"]) for r in upd) else "mixed")
    return row


def main() -> int:
    """Write the checklist CSV."""
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = [check(d) for d in a.dirs]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
