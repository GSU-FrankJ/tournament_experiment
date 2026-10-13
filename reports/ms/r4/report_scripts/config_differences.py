#!/usr/bin/env python3
"""Pre-registration table: the keys in which the run configs of an MS-R4 arm differ from those of its comparator, and the
SHA-256 of every planned ``run_config.json`` (wave r4: 280 runs, wave r4_cms6: 40 runs).

Writes ``reports/ms/r4/arm_config_differences.csv`` and ``reports/ms/r4/run_config_hashes.csv`` (the latter through
``tools/ms/r4_launch.py --plan-csv`` for each wave, with the real results root of the worktree, which is part of every hash).

Usage: python reports/ms/r4/report_scripts/config_differences.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))

import ms_configs as mc  # noqa: E402
import r4_configs as rc  # noqa: E402

PAIRS = [("relu_st_s16", "t1_st_s16", "E1 control relu against control t1"),
         ("lrsm_st_s16", "t1_st_s16", "E1 candidate against control t1 (the criterion's comparison)"),
         ("lrsm_st_s16", "relu_st_s16", "E1 candidate against control relu (descriptive)"),
         ("fb_r", "fb_a0", "E2 arm (2) against arm (1)"), ("fb_rg", "fb_a0", "E2 arm (3) against arm (1)"),
         ("fb_rx", "fb_a0", "E2 descriptive arm against arm (1)"), ("fb_rg", "fb_r", "E2 arm (3) against arm (2)"),
         ("fb_rx", "t1_st_s16", "check C-X2: fb_rx against t1_st_s16 (same sampler, different budget / landing)")]


def main() -> None:
    proto = mc.load_protocol()
    params = json.load(open(ROOT / "reports" / "ms" / "r1" / "prereg_parameters.json"))
    rows = []
    for arm, base, why in PAIRS:
        for q in rc.QS:
            a = rc.build_config_r4(proto, q, 10511, arm, f"<out>/{arm}", params)
            b = rc.build_config_r4(proto, q, 10511, base, f"<out>/{base}", params)
            d = [k for k in rc.config_diff(a, b) if k not in ("arm", "run", "record.run", "record.output_dir")]
            rows.append({"arm": arm, "comparator": base, "q": q, "purpose": why, "keys_that_differ": ";".join(d),
                         "labels_that_differ": "arm;run;record.run;record.output_dir"})
    for arm in rc.CMS6_ARMS:
        for q in rc.QS:
            a = rc.build_config_r4(proto, q, 10503, arm, "<out>/x", params)
            b = mc.build_config(proto, q, 10503, arm, "<out>/x", params)
            d = rc.config_diff(a, b)
            rows.append({"arm": f"{arm} (MS-R4, wave r4_cms6)", "comparator": f"{arm} (MS-R3, same seed)", "q": q,
                         "purpose": "check C-MS6: the control re-run on an MS-R3 seed against MS-R3's own config",
                         "keys_that_differ": ";".join(d), "labels_that_differ": "(all differences listed: round tag, run name)"})
    out = ROOT / "reports" / "ms" / "r4" / "arm_config_differences.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hashes = ROOT / "reports" / "ms" / "r4" / "run_config_hashes.csv"
    tmp = ROOT / "reports" / "ms" / "r4" / "_plan_tmp.csv"
    allrows = []
    for wave in ("r4", "r4_cms6"):
        tmp.unlink(missing_ok=True)
        subprocess.run([sys.executable, str(ROOT / "tools" / "ms" / "r4_launch.py"), "--wave", wave, "--params",
                        str(ROOT / "reports" / "ms" / "r1" / "prereg_parameters.json"), "--plan-csv", str(tmp)], check=True,
                       capture_output=True)
        allrows += list(csv.DictReader(open(tmp)))
    tmp.unlink()
    with open(hashes, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0]))
        w.writeheader()
        w.writerows(allrows)
    print(f"{len(rows)} difference rows -> {out}; {len(allrows)} config hashes -> {hashes}")


if __name__ == "__main__":
    main()
