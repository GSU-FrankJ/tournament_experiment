#!/usr/bin/env python3
"""Render the Markdown tables of reports/v2/protocol_lock_and_rehearsal.md from the result CSVs.

Output: results/v2_T2_locked/report_tables.md (one '### title' section per table, with its source).

Usage: python tools/v2/locked_tables.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LK = ROOT / "results" / "v2_T2_locked"
RA = LK / "rehearsal_analysis"


def fmt(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "—"
        if float(v).is_integer() and abs(v) < 1e7:
            return str(int(v))
        return "0" if v == 0 else f"{v:.4g}"
    return str(v)


def md(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> int:
    s = []

    def sec(title, src, df):
        s.append(f"### {title}\n\nSource: `{src}`\n\n{md(df)}\n")

    rel = lambda p: str(p.relative_to(ROOT))  # noqa: E731
    cal = pd.read_csv(LK / "calibration" / "calibration_locked.csv")
    sec("calibration", rel(LK / "calibration" / "calibration_locked.csv"),
        cal[["q", "policy", "tier", "Gmax_full_over_dw", "Gmax_full_t", "Gmax_full_d", "G_max_t1_over_dw", "eta_T_over_dw",
             "EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw", "dFull_over_dw"]])
    dcols = [c for c in cal.columns if c.startswith("diff_vs_phase1_") and not c.endswith(("_t", "_d", "_t1", "_t2"))]
    t = cal[["q", "policy", "tier"] + dcols].copy()
    t["max_abs_diff_vs_phase1"] = t[dcols].abs().max(axis=1)
    sec("calibration vs Phase 1", rel(LK / "calibration" / "calibration_locked.csv"),
        t[["q", "policy", "tier", "max_abs_diff_vs_phase1"] + [c for c in dcols if "Gmax" in c or "EXP" in c or "dReach" in c]])
    loc = cal[["q", "policy", "tier", "Gmax_full_t", "phase1_Gmax_full_t", "Gmax_full_d", "phase1_Gmax_full_d"]]
    sec("calibration argmax vs Phase 1", rel(LK / "calibration" / "calibration_locked.csv"), loc)
    z = cal[cal.policy == "zero"][["q", "tier", "Gmax_full_over_dw", "pi_ref_Gmax", "diff_vs_pi_Gmax", "Gmax_full_t", "pi_ref_t",
                                   "Gmax_full_d", "pi_ref_d", "EXP_root_over_dw", "pi_ref_root_gain", "diff_vs_pi_root_gain"]]
    sec("zero policy vs PI reference", rel(LK / "calibration" / "calibration_locked.csv"), z)
    c1 = pd.read_csv(RA / "check1_phaseA_vs_stitched.csv")
    cols = [c for c in c1.columns if c.endswith("_identical")]
    sec("check 1 counts", rel(RA / "check1_phaseA_vs_stitched.csv"),
        pd.DataFrame({"field": cols, "identical (of 20)": [int(c1[c].sum()) for c in cols]}))
    c2 = pd.read_csv(RA / "check2_phaseB_vs_launcher.csv")
    cols = [c for c in c2.columns if c.endswith("_identical") or c in ("q", "seed", "n_exports_B")]
    sec("check 2", rel(RA / "check2_phaseB_vs_launcher.csv"), c2[cols].T.reset_index().rename(columns={"index": "field", 0: "q50 s10503", 1: "q60 s10503"}))
    g = pd.read_csv(RA / "gates_per_run.csv")
    sec("per-run gates (final tier; dev tier in parentheses columns)", rel(RA / "gates_per_run.csv"),
        g[["q", "seed", "eta_T_over_dw_final", "eta_T_over_dw_dev", "stage2_rmse_pos_over_g2_0_final", "stage2_tail_mean_over_g2_0_final",
           "G-A", "Gmax_full_over_dw_final", "Gmax_full_over_dw_dev", "stage1_rel_err_abs_final", "G-F", "run_pass", "outcome",
           "G-A_dev", "G-F_dev"]])
    sec("pass counts", rel(RA / "pass_counts.csv"), pd.read_csv(RA / "pass_counts.csv"))
    rep_a = g[["q", "seed", "A_stage2_peak_rel_err_signed", "A_stage2_peak_locfree_rel_err", "A_stage2_peak_locfree_argmax_d",
               "A_stage2_sym_err_max", "A_stage2_tail_max", "A_DeltaT_over_dw_on_max", "A_DeltaT_over_dw_off_max",
               "A_sigma_effort_at_0_t2", "A_smoothed_share_peak_gap_d0"]]
    sec("reported stage-2 metrics at the end of A (final tier)", rel(RA / "gates_per_run.csv"), rep_a)
    rep_b = g[["q", "seed", "B_e1_at_0", "B_stage1_rel_err_signed", "dec_learning_rel", "dec_learning_rel_lo", "dec_learning_rel_hi",
               "dec_inherited_rel", "dec_inherited_rel_lo", "dec_inherited_rel_hi", "B_EXP_root_over_dw", "B_dReach_over_dw",
               "B_Deltamax_all_over_dw", "B_dFull_over_dw", "B_Gmax_full_t", "B_Gmax_full_d", "B_sigma_effort_at_0_t1",
               "dec_band_contiguous", "dec_e1_inside_sweep", "drift_test_pass", "wall_sec"]]
    sec("reported stage-1 / full-policy metrics at the end of B (final tier)", rel(RA / "gates_per_run.csv"), rep_b)
    dist = pd.read_csv(RA / "gate_distributions.csv")
    gm = ("eta_T_over_dw", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean_over_g2_0", "Gmax_full_over_dw", "stage1_rel_err_abs")
    sec("gate-metric distributions (final tier)", rel(RA / "gate_distributions.csv"),
        dist[dist.metric.isin([f"{m}_final" for m in gm])].sort_values(["metric", "q"]))
    sec("gate-metric distributions (dev tier)", rel(RA / "gate_distributions.csv"),
        dist[dist.metric.isin([f"{m}_dev" for m in gm])].sort_values(["metric", "q"]))
    sec("dev - final differences of the gate metrics", rel(RA / "gate_distributions.csv"),
        dist[dist.metric.isin([f"{m}_dev_minus_final" for m in gm])].sort_values(["metric", "q"]))
    dmf = [c for c in g.columns if c.startswith(("A_dmf_", "B_dmf_"))]
    rows = []
    for q, gg in g.groupby("q"):
        for c in dmf:
            x = gg[c].astype(float)
            rows.append({"q": q, "metric": c.replace("A_dmf_", "end-of-A ").replace("B_dmf_", "end-of-B "),
                         "min": x.min(), "median": x.median(), "max": x.max(), "max_abs": x.abs().max()})
    sec("dev - final differences of every reported metric", rel(RA / "gates_per_run.csv"), pd.DataFrame(rows))
    cd = LK / "cusp_diagnostic"
    sec("cusp thresholds", rel(cd / "thresholds_summary.csv"), pd.read_csv(cd / "thresholds_summary.csv"))
    sec("cusp thresholds per init", rel(cd / "thresholds_per_init.csv"), pd.read_csv(cd / "thresholds_per_init.csv"))
    sec("cusp identity vs Pilot 4", rel(cd / "identity_vs_pilot4.csv"), pd.read_csv(cd / "identity_vs_pilot4.csv"))
    (LK / "report_tables.md").write_text("\n".join(s))
    print("wrote", LK / "report_tables.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
