#!/usr/bin/env python3
"""Descriptive comparison of the MS_base stage-1 end state with rehearsal_v2_0's end of Phase B (not a check).

The stage-1 phase of MS_base is not bit-identical to v2.0's Phase B (the batch holds stage-1 rows only), so this is
descriptive: the closed-form stage-1 error, G-S, G-F, Gmax and the v2.0 combination per run, and the first-order
residual of stage 1 at the freeze (final and development tier).

Usage: python reports/ms/r1/report_scripts/base_vs_rehearsal_stage1.py <rehearsal_v2_0 root> results/ms_r1/base \
           results/ms_r1/base_vs_rehearsal_stage1.csv
"""
import json
import sys

import pandas as pd

ref, base, out = sys.argv[1:4]
rows = []
for q in (50, 60):
    for s in range(10501, 10511):
        g = json.load(open(f"{base}/q{q}/seed{s}/MS_base/gates.json"))
        r = json.load(open(f"{ref}/q{q}/seed{s}/gates.json"))
        ms, rh = g["reported"]["end_of_stage1"]["final"], r["reported"]["end_of_B"]["final"]
        rows.append(dict(
            q=q, seed=s, ms_e1_err_signed=ms["stage1_rel_err_signed"], rh_e1_err_signed=rh["stage1_rel_err_signed"],
            ms_G_S=g["G-S"]["pass"], rh_G_S=r["G-S"]["pass"], ms_G_F=g["G-F"]["pass"], rh_G_F=r["G-F"]["pass"],
            ms_Gmax=ms["Gmax_full_over_dw"], rh_Gmax=rh["Gmax_full_over_dw"],
            ms_v20_combination=g["v20_combination_pass"], rh_v20_combination=r["run_pass"],
            ms_R1_final=g["d3_at_freeze"]["1"]["final"]["R"], ms_R1_dev=g["d3_at_freeze"]["1"]["development"]["R"]))
d = pd.DataFrame(rows)
d.to_csv(out, index=False)
for q in (50, 60):
    x = d[d.q == q]
    print(f"q={q}: |err1| MS_base median {x.ms_e1_err_signed.abs().median():.4f} (max {x.ms_e1_err_signed.abs().max():.4f}); "
          f"rehearsal {x.rh_e1_err_signed.abs().median():.4f} (max {x.rh_e1_err_signed.abs().max():.4f}); mean signed "
          f"MS {x.ms_e1_err_signed.mean():+.4f} vs rh {x.rh_e1_err_signed.mean():+.4f}; G-S {x.ms_G_S.sum()}/10 vs "
          f"{x.rh_G_S.sum()}/10; G-F {x.ms_G_F.sum()}/10 vs {x.rh_G_F.sum()}/10; v2.0 combination "
          f"{x.ms_v20_combination.sum()}/10 vs {x.rh_v20_combination.sum()}/10")
print("stage-1 R_1 at the freeze, MS_base: final tier median %.4f, development tier median %.4f"
      % (d.ms_R1_final.median(), d.ms_R1_dev.median()))
