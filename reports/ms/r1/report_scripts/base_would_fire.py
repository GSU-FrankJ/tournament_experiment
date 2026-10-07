#!/usr/bin/env python3
"""Would-fire of the MS_base runs at the pre-registered thresholds, recomputed from their logged checks.

The MS_base wave was launched without ``--params`` and therefore carries the D4 default rho_2 = 0.03 in its config;
the config's thresholds decide nothing in the legacy arm (the training state is identical, check C-MS1) and only
define the ``would_fire_local`` entry of ``rule_log.json``. This script recomputes the entry with the thresholds of
``reports/ms/r1/prereg_parameters.json`` from the check rows of ``ms_checks_stage{t}.csv`` (Delta, R, R_tail, C,
valid: the controller's eligibility, M consecutive checks at the K = 25 cadence).

Usage: python reports/ms/r1/report_scripts/base_would_fire.py results/ms_r1/base reports/ms/r1/prereg_parameters.json \
           results/ms_r1/base_would_fire.csv
"""
import json
import sys

import numpy as np
import pandas as pd

root, params, out = sys.argv[1:4]
prm = json.load(open(params))
rows = []
for q in (50, 60):
    for seed in range(10501, 10511):
        d = f"{root}/q{q}/seed{seed}/MS_base"
        row = dict(q=q, seed=seed)
        for t in ("2", "1"):
            st = prm["stages"][t]
            c = pd.read_csv(f"{d}/ms_checks_stage{t}.csv")
            c = c[c["mode"] == "train"]
            tail_ok = (~c["tail_term"].astype(bool)) | (c["R_tail"] <= st["tau"])
            el = (c["valid"].astype(bool) & (c["Delta"] <= st["eps"]) & (c["R"] <= st["rho"]) & tail_ok
                  & (c["C"] <= prm["conc_limit"]))
            streak, fire = 0, None
            for loc, e in zip(c["local"], el):
                streak = streak + 1 if e else 0
                if streak >= prm["M"]:
                    fire = int(loc)
                    break
            logged = json.load(open(f"{d}/gates.json"))["rule"][t]["would_fire_local"]
            row.update({f"would_fire_stage{t}_prereg": fire, f"would_fire_stage{t}_logged": logged,
                        f"min_R_stage{t}": float(c["R"].min()), f"R_at_end_stage{t}": float(c["R"].iloc[-1])})
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(out, index=False)
for q in (50, 60):
    x = df[df.q == q]
    f2, f1 = x["would_fire_stage2_prereg"].dropna(), x["would_fire_stage1_prereg"].dropna()
    print(f"q={q}: stage 2 would fire in {len(f2)}/10 runs {sorted(f2.astype(int).tolist())} "
          f"(logged at rho_2=0.03: {int(x['would_fire_stage2_logged'].notna().sum())}/10); min R over the checks "
          f"median {x['min_R_stage2'].median():.4f}; stage 1 would fire in {len(f1)}/10 runs, "
          f"median local update {f1.median() if len(f1) else float('nan')}")
