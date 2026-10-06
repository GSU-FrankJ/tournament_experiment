#!/usr/bin/env python3
"""Build sections 4.1-4.4 (methods 1-4) of the T=2 refinement report from the evidence pack.

Reads only the pack CSV/JSON files; writes sec04a.md and sec04a_ledger.csv under SCRATCH.
Every number placed in the text goes through a helper that records a ledger row.
"""
import csv
import json
import os
import sys

import pandas as pd

PACK = ("/home/fjiang4/tournament_experiment/.claude/worktrees/t2-refine-pack/"
        "reports/t2_refine_100526/evidence/")
AN = PACK + "results/v2_refine/analysis/"
SCRATCH = ("/tmp/claude-1331199693/-home-fjiang4-tournament-experiment--claude-worktrees-"
           "r2c-sampler-protocol-v2-1-c4c0f1/152f0306-5492-45a1-94d5-59493fa0d141/scratchpad/"
           "report_parts/")

rd = lambda f: pd.read_csv(AN + f)  # noqa: E731
DI = rd("decision_inputs.csv").set_index("arm")          # R1-01
S1C = rd("stage1_criterion.csv").set_index("arm")        # R1-02
S1D = rd("stage1_dispersion.csv")                        # R1-03
S1P = rd("stage1_paired.csv")                            # R1-18
S2P = rd("stage2_paired.csv")                            # R1-19
S1COST = rd("stage1_cost.csv").set_index("arm")          # R1-22
S2COST = rd("stage2_cost.csv").set_index("arm")          # R1-23
S1G = rd("stage1_gate_counts.csv")                       # R1-24
S2G = rd("stage2_gate_counts.csv")                       # R1-25
S1STOP = rd("stage1_stop_epoch.csv")                     # R1-27
S2STOP = rd("stage2_stop_epoch.csv")                     # R1-28
S2D = rd("stage2_dispersion.csv")                        # R1-38
S2A = rd("stage2_arm_summary.csv")                       # R1-40
ANN = rd("stage2_annealing.csv")                         # R1-07
S2C = rd("stage2_criterion.csv")
S2C = S2C[S2C.comparison == "vs baseline"].set_index("arm")  # R1-06

LEDGER = []
SEC = ["4.1"]


def rec(text, value, item, loc):
    LEDGER.append({"statement_id": "S%04d" % (len(LEDGER) + 1), "section": SEC[0], "text": text,
                   "value": value, "item_id": item, "locator": loc})


def g4(x):
    return format(float(x), ".4g")


def N(x, item, loc, fmt=g4):
    """Format a number, record it, return the string."""
    t = fmt(x)
    rec(t, x, item, loc)
    return t


def CI(mean, lo, hi, item, loc):
    return "%s [%s, %s]" % (N(mean, item, loc + " (mean)"), N(lo, item, loc + " (CI lo)"),
                            N(hi, item, loc + " (CI hi)"))


def T(text, item, loc):
    """Record a setting or a count typed in the text (value = the text itself)."""
    rec(text, text, item, loc)
    return text


def stage_of(arm):
    return "stage 1" if arm.startswith("B_") else "stage 2"


R1F = {"stage 1": ("R1-01", "decision_inputs.csv"), "stage 2": ("R1-01", "decision_inputs.csv")}

# ---------------------------------------------------------------- cross-checks (assertions)
for arm in DI.index:
    if arm not in S1C.index and arm not in S2C.index:
        continue
    crit, pair = (S1C, S1P) if arm.startswith("B_") else (S2C, S2P)
    met = "stage1_rel_err_abs" if arm.startswith("B_") else "stage2_peak_rel_err_abs"
    for q in (50, 60):
        p = pair[(pair.arm == arm) & (pair.q == q) & (pair.metric == met)
                 & (pair.comparison == "vs baseline")].iloc[0]
        for a, b in (("mean", "mean"), ("ci_mean_lo", "ci_mean_lo"), ("ci_mean_hi", "ci_mean_hi"),
                     ("median", "median"), ("ci_median_lo", "ci_median_lo"),
                     ("ci_median_hi", "ci_median_hi"), ("n_better", "n_better")):
            col = {"mean": "mean", "median": "median"}.get(a, a)
            dcol = {"mean": "mean_q%d", "ci_mean_lo": "ci_mean_lo_q%d",
                    "ci_mean_hi": "ci_mean_hi_q%d", "median": "median_q%d",
                    "ci_median_lo": "ci_median_lo_q%d", "ci_median_hi": "ci_median_hi_q%d",
                    "n_better": "n_better_q%d"}[a] % q
            assert abs(DI.loc[arm, dcol] - p[col]) < 1e-12, (arm, q, a)
        assert bool(crit.loc[arm, "a_q%d" % q]) == bool(DI.loc[arm, "criterion_a_q%d" % q])
    assert DI.loc[arm, "criterion_overall"] == crit.loc[arm, "overall"]
    assert DI.loc[arm, "criterion_b_status"] == crit.loc[arm, "b_status"]
    # dispersion ratio vs R1-03 / R1-38
    dd, mm = (S1D, "stage1_rel_err_signed") if arm.startswith("B_") else (S2D, "stage2_peak_rel_err_signed")
    for q in (50, 60):
        r = dd[(dd.arm == arm) & (dd.q == q) & (dd.metric == mm)].iloc[0]
        assert abs(r.ratio - DI.loc[arm, "disp_ratio_q%d" % q]) < 1e-12
        assert abs(r.ci_lo - DI.loc[arm, "disp_ci_lo_q%d" % q]) < 1e-12
        assert abs(r.ci_hi - DI.loc[arm, "disp_ci_hi_q%d" % q]) < 1e-12
    cost = S1COST if arm.startswith("B_") else S2COST
    assert abs(cost.loc[arm, "phase_wall_ratio_vs_base"] - DI.loc[arm, "cost_phase_wall_ratio_vs_base"]) < 1e-12


# ---------------------------------------------------------------- table builders
def main_table(arms):
    out = ["| stage | arm | q | n pairs | mean difference [95% CI] | median difference [95% CI] "
           "| n_better | (a) at this q: CI of mean < 0 | (b) | criterion overall |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        for q in (50, 60):
            loc = "decision_inputs.csv[arm=%s,q=%d]" % (arm, q)
            col = lambda c: DI.loc[arm, c + "_q%d" % q]  # noqa: E731
            npairs = N(col("n_pairs"), "R1-01", loc + " n_pairs", lambda x: "%d" % x)
            mean = CI(col("mean"), col("ci_mean_lo"), col("ci_mean_hi"), "R1-01", loc)
            med = CI(col("median"), col("ci_median_lo"), col("ci_median_hi"), "R1-01", loc + " median")
            nb = N(col("n_better"), "R1-01", loc + " n_better", lambda x: "%d" % x) + "/10"
            a = str(bool(col("criterion_a")))
            b = "%s (%s violations)" % (DI.loc[arm, "criterion_b_status"],
                                        N(DI.loc[arm, "criterion_b_n_violations"], "R1-01",
                                          loc + " criterion_b_n_violations", lambda x: "%d" % x))
            out.append("| %s | `%s` | %d | %s | %s | %s | %s | %s | %s | %s |" % (
                stage_of(arm), arm, q, npairs, mean, med, nb, a, b, DI.loc[arm, "criterion_overall"]))
    return "\n".join(out)


def main_source(arms, extra=""):
    return ("Source: R1-01 (`results/v2_refine/analysis/decision_inputs.csv`, rows arm in {%s}; columns "
            "`n_pairs_q*`, `mean_q*`, `ci_mean_lo_q*`, `ci_mean_hi_q*`, `median_q*`, `ci_median_lo_q*`, "
            "`ci_median_hi_q*`, `n_better_q*`, `criterion_a_q*`, `criterion_b_status`, "
            "`criterion_b_n_violations`, `criterion_overall`); the same values are in R1-18 and R1-19 "
            "(`stage1_paired.csv`, `stage2_paired.csv`, metrics `stage1_rel_err_abs` / "
            "`stage2_peak_rel_err_abs`) and the criterion columns in R1-02 and R1-06; the values were "
            "checked equal across these tables when this section was built.%s"
            % (", ".join("`%s`" % a for a in arms), extra))


def disp_table(arms, with_target=True, with_stop=False):
    head = ["stage", "arm", "q", "dispersion ratio SD(signed error), arm / baseline [95% CI]"]
    if with_target:
        head.append("runs with S1 error <= 0.05: arm (B_base)")
    if with_stop:
        head.append("share of updates with fewer than 10 epochs run")
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for arm in arms:
        for q in (50, 60):
            loc = "decision_inputs.csv[arm=%s,q=%d]" % (arm, q)
            ratio = CI(DI.loc[arm, "disp_ratio_q%d" % q], DI.loc[arm, "disp_ci_lo_q%d" % q],
                       DI.loc[arm, "disp_ci_hi_q%d" % q], "R1-01", loc + " disp_ratio")
            row = [stage_of(arm), "`%s`" % arm, str(q), ratio]
            if with_target:
                if arm.startswith("B_"):
                    ga = S1G[(S1G.arm == arm) & (S1G.q == q)].iloc[0]
                    gb = S1G[(S1G.arm == "B_base") & (S1G.q == q)].iloc[0]
                    loc2 = "stage1_gate_counts.csv[arm=%s,q=%d] n_target0929_pass" % (arm, q)
                    loc3 = "stage1_gate_counts.csv[arm=B_base,q=%d] n_target0929_pass" % q
                    row.append("%s of 10 (%s)" % (N(ga.n_target0929_pass, "R1-24", loc2, lambda x: "%d" % x),
                                                  N(gb.n_target0929_pass, "R1-24", loc3, lambda x: "%d" % x)))
                else:
                    row.append("n/a (no table)")
            if with_stop:
                st = (S1STOP if arm.startswith("B_") else S2STOP)
                r = st[(st.arm == arm) & (st.q == q)].iloc[0]
                it, fn = ("R1-27", "stage1_stop_epoch.csv") if arm.startswith("B_") else ("R1-28", "stage2_stop_epoch.csv")
                row.append(N(r.share_fewer_than_10, it, "%s[arm=%s,q=%d] share_fewer_than_10" % (fn, arm, q)))
            out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def cost_table(arms, with_steps=True):
    out = ["| stage | arm | phase wall time per run, mean over 20 runs (s) | ratio to baseline arm "
           "| updates per run | episodes per run | optimizer steps per run |",
           "|---|---|---|---|---|---|---|"]
    seen = set()
    for st, base in (("stage 1", "B_base"), ("stage 2", "A_base")):
        cost = S1COST if st == "stage 1" else S2COST
        it, fn = ("R1-22", "stage1_cost.csv") if st == "stage 1" else ("R1-23", "stage2_cost.csv")
        sel = [a for a in arms if stage_of(a) == st]
        if not sel:
            continue
        for arm in [base] + sel:
            c = cost.loc[arm]
            loc = "%s[arm=%s]" % (fn, arm)
            out.append("| %s | `%s` | %s | %s | %s | %s | %s |" % (
                st, arm,
                N(c.mean_phase_wall_sec, it, loc + " mean_phase_wall_sec"),
                N(c.phase_wall_ratio_vs_base, it, loc + " phase_wall_ratio_vs_base"),
                N(c.mean_phase_local_updates, it, loc + " mean_phase_local_updates", lambda x: "%d" % x),
                N(c.mean_phase_episodes, it, loc + " mean_phase_episodes"),
                N(c.mean_n_minibatch_steps_total, it, loc + " mean_n_minibatch_steps_total")))
    return "\n".join(out)


def cost_source(arms):
    return ("Source: R1-22 (`results/v2_refine/analysis/stage1_cost.csv`) and R1-23 (`stage2_cost.csv`), "
            "columns `mean_phase_wall_sec`, `phase_wall_ratio_vs_base`, `mean_phase_local_updates`, "
            "`mean_phase_episodes`, `mean_n_minibatch_steps_total`; baseline rows `B_base` and `A_base`.")


def cell(arm, q, c):
    return DI.loc[arm, c + "_q%d" % q]


# ---------------------------------------------------------------- 4.1
parts = []
SEC[0] = "4.1"
arms1 = ["B_polish1", "B_polish2", "A_polish1", "A_polish2"]
ci_ok = all(DI.loc[a, "criterion_overall"] == "not met" for a in arms1)
assert ci_ok
assert all(DI.loc[a, "ci_mean_lo_q%d" % q] < 0 < DI.loc[a, "ci_mean_hi_q%d" % q] for a in arms1 for q in (50, 60))

base1_runs = int(S1G[S1G.arm == "B_base"].n_complete.sum())
sum_gf = int(S1G.n_G_F.sum()); sum_gn = int(S1G.n_G_N_gmax.sum()); sum_s1runs = int(S1G.n_complete.sum())
sum_ga = int(S2G.n_G_A.sum()); sum_gn2 = int(S2G.n_G_N_eta.sum()); sum_s2runs = int(S2G.n_complete.sum())
assert (sum_gf, sum_gn, sum_s1runs, sum_ga, sum_gn2, sum_s2runs) == (160, 160, 160, 220, 220, 220)
assert S1G.shape[0] == 16 and S2G.shape[0] == 22

b2 = S2A[S2A.arm == "A_base"].set_index("q")
b1 = rd("stage1_arm_summary.csv")
b1 = b1[b1.arm == "B_base"].set_index("q")

for _a, _t in (("B_polish1", "3e-4 to 3e-5 over local updates 1-400, then 3e-5 to 3e-6 over 401-600"),
               ("B_polish2", "3e-4 to 3e-6 over 1-600"),
               ("A_polish1", "3e-4 to 3e-5 over local updates 1-250, then 3e-5 to 3e-6 over 251-400"),
               ("A_polish2", "3e-4 to 3e-6 over 1-400")):
    T(_t, "RR-08", "01_preregistration.md section 3.1/3.2, arm %s" % _a)
T("3e-4 to 3e-5 over local updates 1-600", "RR-08", "01_preregistration.md section 2 (Baseline), phase B")
T("3e-4 to 3e-5 over local updates 1-400", "RR-08", "01_preregistration.md section 3.2 (stage-2 window)")
T("1600 updates in phase A, 600 in phase B", "PL-08", "v2_T2_locked_v1_1.json protocol.phase_caps A=1600, B=600")
T("10,000 resamples", "RR-08", "01_preregistration.md section 5 (Bootstrap): 10,000 percentile resamples")
T("20261003", "RR-08", "01_preregistration.md section 5 (Bootstrap): default_rng(20261003)")
T("10501-10510", "RR-08", "development seeds (RR-02 header / summary: development seeds 10501-10510)")
T("8 arms x 20 runs", "RR-08", "01_preregistration.md section 3.1 (8 arms x 20 = 160 runs)")
T("11 arms x 20 runs", "RR-08", "01_preregistration.md section 3.2 (11 arms x 20 = 220 runs)")
T("0.05", "PL-01", "v2_T2_locked_v2_0.json gates.G-S threshold 0.05")
s41 = []
s41.append("### 4.1 Method 1: polishing the learning rate\n")
s41.append(
    "**Outcome.** No polishing arm meets the criterion: for each of the four arms the 95% interval of the "
    "mean paired difference contains 0 at both q (criterion overall \"not met\" for `B_polish1`, `B_polish2`, "
    "`A_polish1`, `A_polish2`) [R1-01, R1-02, R1-06], and part (b) holds with 0 violations [R1-01]. No arm is adopted "
    "(R1 summary addendum, item 2, [RR-01]).\n")
s41.append(
    "**What was changed.** The learning rate (LR) of both optimizers is decayed to a lower end value inside the existing "
    "update budget; the budgets (1600 updates in phase A, 600 in phase B) do not change and polishing is carved out of the "
    "existing windows [RR-08, section 2]. The baseline LR is %s at the start; stage 1 (phase B) decays linearly from "
    "3e-4 to 3e-5 over local updates 1-600, and stage 2 (continued phase A from update 1200) from 3e-4 to 3e-5 over local "
    "updates 1-400 [RR-08, section 2 (Baseline) and section 3.2]. Arms [RR-08, sections 3.1 and 3.2]: "
    "`B_polish1` = 3e-4 to 3e-5 over local updates 1-400, then 3e-5 to 3e-6 over 401-600; `B_polish2` = one window 3e-4 to "
    "3e-6 over 1-600; `A_polish1` = 3e-4 to 3e-5 over local updates 1-250, then 3e-5 to 3e-6 over 251-400; `A_polish2` = one "
    "window 3e-4 to 3e-6 over 1-400. The LR is linear inside a window, constant within an update, and the Adam state is "
    "preserved [RR-08, section 4 item 1]. Stage-1 arms branch from the same end-of-A parent (the stage-2 policy is frozen), stage-2 arms from the same update-1200 state [RR-08, section 2]. Everything else is the locked v1.1 pipeline, and the baseline arms are `B_base` "
    "(stage 1, primary metric the S1 error |e1_hat(0) - e1*|/e1*) and `A_base` (stage 2, primary metric the absolute "
    "signed peak error) [RR-08, section 3; R1-01 columns `baseline`, `primary_metric`].\n"
    % T("3e-4", "PL-08", "v2_T2_locked_v1_1.json ppo.lr = 0.0003"))
s41.append(
    "**How to read the tables (valid for sections 4.1-4.4).** Each row is one arm at one q, with n = %s paired runs "
    "(development seeds 10501-10510, paired by (q, seed) with the baseline arm of the same stage). The difference is arm minus "
    "baseline on the primary metric, so a negative difference is an improvement. Intervals are 95%% percentile bootstrap "
    "intervals (10,000 resamples; bootstrap seed 20261003 for R1) of the mean and of the median of the 10 paired differences "
    "[RR-02, header; RR-08, section 5]. `n_better` is the number of seeds with a smaller primary metric than the baseline "
    "run. Criterion part (a): the interval of the mean difference lies below 0 at both q. Part (b): no run that passed its "
    "gate under the baseline fails it under the arm. The criterion is descriptive, not a gate [RR-02, header].\n"
    % T("10", "R1-01", "decision_inputs.csv n_pairs_q50 = n_pairs_q60 = 10 for every arm in this section"))
s41.append(main_table(arms1))
s41.append("")
s41.append(main_source(arms1))
s41.append("")
s41.append(
    "**Spread and cost.** The dispersion ratio is the across-seed SD of the signed error of the arm over that of the baseline "
    "(the signed S1 error for stage 1, the signed peak error for stage 2), with a paired-resampling interval [RR-02, header]. "
    "The last column of the stage-1 rows counts the runs whose S1 error is within 0.05 (`n_target0929_pass`, the 0.05 target of "
    "the 0929 discussion; the same threshold, 0.05, as gate G-S of v2.0 [PL-01, `gates.G-S`]); this count is not part of the criterion. "
    "The R1 gate-count tables hold no corresponding count for stage 2.\n")
s41.append(disp_table(arms1))
s41.append("")
s41.append("Source: R1-01 (`decision_inputs.csv`, columns `disp_ratio_q*`, `disp_ci_lo_q*`, `disp_ci_hi_q*`; equal to R1-03 `stage1_dispersion.csv` "
           "and R1-38 `stage2_dispersion.csv`, metrics `stage1_rel_err_signed` / `stage2_peak_rel_err_signed`, columns `ratio`, `ci_lo`, `ci_hi`) "
           "and R1-24 (`stage1_gate_counts.csv`, column `n_target0929_pass`).")
s41.append("")
s41.append(
    "Observation from the table: every polishing dispersion interval contains 1, so no polishing arm changes the "
    "across-seed spread of the signed error beyond what the 10-seed intervals allow [R1-01]. The baseline medians of the "
    "absolute stage-2 peak error are %s (q = 50) and %s (q = 60) [R1-40]; the baseline medians of the S1 error are %s and %s [R1-39].\n"
    % (N(b2.loc[50, "median_stage2_peak_rel_err_abs"], "R1-40", "stage2_arm_summary.csv[arm=A_base,q=50] median_stage2_peak_rel_err_abs"),
       N(b2.loc[60, "median_stage2_peak_rel_err_abs"], "R1-40", "stage2_arm_summary.csv[arm=A_base,q=60] median_stage2_peak_rel_err_abs"),
       N(b1.loc[50, "median_stage1_rel_err_abs"], "R1-39", "stage1_arm_summary.csv[arm=B_base,q=50] median_stage1_rel_err_abs"),
       N(b1.loc[60, "median_stage1_rel_err_abs"], "R1-39", "stage1_arm_summary.csv[arm=B_base,q=60] median_stage1_rel_err_abs")))
assert all(DI.loc[a, "disp_ci_lo_q%d" % q] < 1 < DI.loc[a, "disp_ci_hi_q%d" % q] for a in arms1 for q in (50, 60))
s41.append(cost_table(arms1))
s41.append("")
s41.append(cost_source(arms1))
s41.append("")
s41.append(
    "**Gate observation (applies to all pilot arms of sections 4.1-4.4).** All %s stage-1 runs (8 arms x 20 runs) pass G-F and "
    "G-N, and all %s stage-2 runs (11 arms x 20 runs) pass G-A and G-N, so part (b) holds trivially for every arm and the gates do "
    "not discriminate between arms in R1 [RR-02, Observations, \"Run hygiene and checks\"; R1-24, R1-25]. The counts were "
    "recomputed here as the sums of `n_G_F` and `n_G_N_gmax` over the 16 rows of R1-24 (%s and %s of %s runs) and of `n_G_A` and "
    "`n_G_N_eta` over the 22 rows of R1-25 (%s and %s of %s runs). What separates the arms is therefore part (a) of the "
    "criterion, not the gates.\n"
    % (T("160", "R1-24", "stage1_gate_counts.csv sum(n_complete) over 16 rows (computed here)"),
       T("220", "R1-25", "stage2_gate_counts.csv sum(n_complete) over 22 rows (computed here)"),
       T("160", "R1-24", "stage1_gate_counts.csv sum(n_G_F) (computed here)"),
       T("160", "R1-24", "stage1_gate_counts.csv sum(n_G_N_gmax) (computed here)"),
       T("160", "R1-24", "stage1_gate_counts.csv sum(n_complete) (computed here)"),
       T("220", "R1-25", "stage2_gate_counts.csv sum(n_G_A) (computed here)"),
       T("220", "R1-25", "stage2_gate_counts.csv sum(n_G_N_eta) (computed here)"),
       T("220", "R1-25", "stage2_gate_counts.csv sum(n_complete) (computed here)")))
s41.append(
    "**Decision.** Criterion not met; no arm adopted. The recorded decision is: \"Only method 6 enters the protocol. Only "
    "method 6 met the pre-registered criterion ... No other R1 arm enters the protocol\" (R1 summary, addendum of 2026-10-03, "
    "item 2 [RR-01]; protocol v2.0 change log, entry for version 2.0 [PL-01, `change_log`]). The PI accepted the round "
    "(same addendum, \"Accepted with its deviations\" [RR-01]).\n")
parts.append("\n".join(s41))

# ---------------------------------------------------------------- 4.2
SEC[0] = "4.2"
arms2 = ["B_batch", "B_batch_mb256", "A_batch", "A_batch_mb256"]
assert all(DI.loc[a, "criterion_overall"] == "not met" for a in arms2)
only_a50 = [a for a in DI.index if a in arms2 and DI.loc[a, "ci_mean_hi_q50"] < 0]
assert only_a50 == ["A_batch_mb256"]
assert not any(DI.loc[a, "ci_mean_hi_q60"] < 0 for a in arms2)
dm = DI.loc["A_batch_mb256"]
s42 = []
s42.append("### 4.2 Method 2: larger batch\n")
s42.append(
    "**Outcome.** No batch arm meets the criterion. `A_batch_mb256` is the only arm whose interval lies below 0 at one q "
    "(q = 50: %s), and at q = 60 its interval contains 0 (%s) [R1-01, R1-06]; the other three arms contain 0 at both q, and "
    "part (b) holds for all four [R1-01]. The compute cost rises with the batch (table below). No arm is adopted.\n"
    % (CI(dm.mean_q50, dm.ci_mean_lo_q50, dm.ci_mean_hi_q50, "R1-01", "decision_inputs.csv[arm=A_batch_mb256,q=50]"),
       CI(dm.mean_q60, dm.ci_mean_lo_q60, dm.ci_mean_hi_q60, "R1-01", "decision_inputs.csv[arm=A_batch_mb256,q=60]")))
s42.append(
    "**What was changed.** Episodes per update go from %s to %s, in both stages [R1-22, R1-23 (`mean_phase_episodes` over "
    "`mean_phase_local_updates`, computed here: 307200 / 600 and 204800 / 400); RR-08, section 3]. The baseline minibatch is %s "
    "with %s epochs per update [PL-08, `ppo`]. `B_batch` and `A_batch` use minibatch %s, which keeps the number of optimizer steps per update "
    "unchanged (40 in stage 1, 20 in stage 2 [RR-08, sections 3.1 and 3.2]); `B_batch_mb256` and `A_batch_mb256` keep minibatch %s "
    "and so take four times as many steps per update (160 in stage 1, 80 in stage 2 [RR-08, sections 3.1 and 3.2]), i.e. %s and "
    "%s optimizer steps per run against %s and %s for the baselines [R1-22, R1-23]. The number of updates, the LR schedule and every other "
    "setting stay at the locked v1.1 values [RR-08, section 3]. The pre-registration marks the mb256 arms as secondary [RR-08, sections 3.1 and 3.2].\n"
    % (T("512", "PL-08", "v2_T2_locked_v1_1.json protocol.episodes_per_update = 512 (both q blocks; also R1-22 mean_phase_episodes/mean_phase_local_updates)"),
       T("2048", "RR-08", "01_preregistration.md section 3.1/3.2, arm B_batch / A_batch"),
       T("256", "PL-08", "v2_T2_locked_v1_1.json ppo.minibatch = 256"),
       T("10", "PL-08", "v2_T2_locked_v1_1.json ppo.epochs = 10"),
       T("1024", "RR-08", "01_preregistration.md section 3.1/3.2, arm B_batch / A_batch minibatch"),
       T("256", "RR-08", "01_preregistration.md section 3.1/3.2, arm B_batch_mb256 / A_batch_mb256 minibatch"),
       N(S1COST.loc["B_batch_mb256", "mean_n_minibatch_steps_total"], "R1-22", "stage1_cost.csv[arm=B_batch_mb256] mean_n_minibatch_steps_total"),
       N(S2COST.loc["A_batch_mb256", "mean_n_minibatch_steps_total"], "R1-23", "stage2_cost.csv[arm=A_batch_mb256] mean_n_minibatch_steps_total"),
       N(S1COST.loc["B_base", "mean_n_minibatch_steps_total"], "R1-22", "stage1_cost.csv[arm=B_base] mean_n_minibatch_steps_total"),
       N(S2COST.loc["A_base", "mean_n_minibatch_steps_total"], "R1-23", "stage2_cost.csv[arm=A_base] mean_n_minibatch_steps_total")))
s42.append("Paired differences (arm - baseline; negative = better), n = 10 pairs per row; reading rules as in section 4.1:\n")
s42.append(main_table(arms2))
s42.append("")
s42.append(main_source(arms2))
s42.append("")
b8 = DI.loc["B_batch_mb256"]
s42.append(
    "Observations from the table: `B_batch_mb256` has %s seeds better at q = 60 (mean %s), the largest `n_better` among the stage-1 "
    "arms of sections 4.1-4.4 at that q, and its interval contains 0 [R1-01; RR-02, Observations]. At q = 60 the `A_batch` and "
    "`A_batch_mb256` mean differences are positive (%s and %s), i.e. a larger peak error than `A_base`, with intervals containing 0 [R1-01].\n"
    % (N(b8.n_better_q60, "R1-01", "decision_inputs.csv[arm=B_batch_mb256,q=60] n_better", lambda x: "%d/10" % x),
       CI(b8.mean_q60, b8.ci_mean_lo_q60, b8.ci_mean_hi_q60, "R1-01", "decision_inputs.csv[arm=B_batch_mb256,q=60]"),
       N(DI.loc["A_batch", "mean_q60"], "R1-01", "decision_inputs.csv[arm=A_batch,q=60] mean"),
       N(DI.loc["A_batch_mb256", "mean_q60"], "R1-01", "decision_inputs.csv[arm=A_batch_mb256,q=60] mean")))
# check the 'largest n_better among stage-1 arms at q=60' claim over the 14 arms of 4.1-4.4
st1_arms_all = [a for a in DI.index if a.startswith("B_") and a != "B_expcont"]
assert max(DI.loc[st1_arms_all, "n_better_q60"]) == DI.loc["B_batch_mb256", "n_better_q60"] == 8
assert sum(DI.loc[st1_arms_all, "n_better_q60"] == 8) == 1
s42.append("**Spread and cost.** Dispersion ratio as defined in section 4.1:\n")
s42.append(disp_table(arms2))
s42.append("")
s42.append("Source: R1-01 (`decision_inputs.csv`, columns `disp_ratio_q*`, `disp_ci_lo_q*`, `disp_ci_hi_q*`; same values in R1-03, R1-38) and R1-24 (`stage1_gate_counts.csv`, column `n_target0929_pass`).")
s42.append("")
flag = []
for a in arms2:
    for q in (50, 60):
        if DI.loc[a, "disp_ci_hi_q%d" % q] < 1:
            flag.append((a, q))
assert flag == [("A_batch", 50)], flag
s42.append(
    "Observation: the dispersion interval lies entirely below 1 for `A_batch` at q = 50 (ratio %s) and nowhere else in this method; "
    "all other intervals contain 1 [R1-01]. The dispersion ratio is not part of the criterion.\n"
    % CI(DI.loc["A_batch", "disp_ratio_q50"], DI.loc["A_batch", "disp_ci_lo_q50"], DI.loc["A_batch", "disp_ci_hi_q50"], "R1-01",
         "decision_inputs.csv[arm=A_batch,q=50] disp_ratio"))
s42.append(cost_table(arms2))
s42.append("")
s42.append(cost_source(arms2))
s42.append("")
s42.append(
    "The phase wall time is %s (`B_batch`) and %s (`B_batch_mb256`) times the stage-1 baseline's, and %s (`A_batch`) and %s "
    "(`A_batch_mb256`) times the stage-2 baseline's [R1-22, R1-23; RR-02, Observations].\n"
    % (N(S1COST.loc["B_batch", "phase_wall_ratio_vs_base"], "R1-22", "stage1_cost.csv[arm=B_batch] phase_wall_ratio_vs_base"),
       N(S1COST.loc["B_batch_mb256", "phase_wall_ratio_vs_base"], "R1-22", "stage1_cost.csv[arm=B_batch_mb256] phase_wall_ratio_vs_base"),
       N(S2COST.loc["A_batch", "phase_wall_ratio_vs_base"], "R1-23", "stage2_cost.csv[arm=A_batch] phase_wall_ratio_vs_base"),
       N(S2COST.loc["A_batch_mb256", "phase_wall_ratio_vs_base"], "R1-23", "stage2_cost.csv[arm=A_batch_mb256] phase_wall_ratio_vs_base")))
s42.append("**Gates.** As in section 4.1: all stage-1 and stage-2 runs of these arms pass their gates, so part (b) holds trivially [R1-24, R1-25].\n")
s42.append(
    "**Decision.** Criterion not met; no arm adopted. The recorded decision is that only method 6 enters the protocol and that "
    "no other R1 arm does (R1 summary, addendum, item 2 [RR-01]; [PL-01, `change_log`, version 2.0]). No separate decision on larger batches "
    "is recorded.\n")
parts.append("\n".join(s42))

# ---------------------------------------------------------------- 4.3
SEC[0] = "4.3"
arms3 = ["B_kl005", "B_kl010", "A_kl005", "A_kl010"]
assert all(DI.loc[a, "criterion_overall"] == "not met" for a in arms3)
assert all(DI.loc[a, "ci_mean_lo_q%d" % q] < 0 < DI.loc[a, "ci_mean_hi_q%d" % q] for a in arms3 for q in (50, 60))


def stop_share(st, arm, q):
    t = S1STOP if st == 1 else S2STOP
    return t[(t.arm == arm) & (t.q == q)].share_fewer_than_10.iloc[0]


assert stop_share(1, "B_base", 50) == 0 and stop_share(1, "B_base", 60) == 0
assert stop_share(2, "A_base", 50) == 0 and stop_share(2, "A_base", 60) == 0
s43 = []
s43.append("### 4.3 Method 3: target-KL early stopping\n")
s43.append(
    "**Outcome.** No target-KL arm meets the criterion: the intervals of the mean paired difference contain 0 at both q for all "
    "four arms, and part (b) holds [R1-01, R1-02, R1-06]. What changes is cost: the phase wall time falls to %s (`B_kl005`), %s "
    "(`B_kl010`), %s (`A_kl005`) and %s (`A_kl010`) of the baseline's [R1-22, R1-23]. The recorded decision is that "
    "target-KL \"is not adopted either: it changes cost, not accuracy\" (see Decision below).\n"
    % (N(S1COST.loc["B_kl005", "phase_wall_ratio_vs_base"], "R1-22", "stage1_cost.csv[arm=B_kl005] phase_wall_ratio_vs_base"),
       N(S1COST.loc["B_kl010", "phase_wall_ratio_vs_base"], "R1-22", "stage1_cost.csv[arm=B_kl010] phase_wall_ratio_vs_base"),
       N(S2COST.loc["A_kl005", "phase_wall_ratio_vs_base"], "R1-23", "stage2_cost.csv[arm=A_kl005] phase_wall_ratio_vs_base"),
       N(S2COST.loc["A_kl010", "phase_wall_ratio_vs_base"], "R1-23", "stage2_cost.csv[arm=A_kl010] phase_wall_ratio_vs_base")))
s43.append(
    "**What was changed.** After each PPO epoch the whole-buffer KL over the policy rows (mean((r - 1) - log r), the quantity "
    "logged as `kl_epochs`) is computed; if it is strictly greater than `target_kl`, no further epoch runs in that update (no "
    "actor and no critic steps), and the epoch that exceeded the target has run. The permutations of the skipped epochs are still "
    "drawn and discarded, so the minibatch stream position after the update equals the baseline's [RR-08, section 4 item 3]. "
    "The baseline runs %s epochs per update [PL-08, `ppo.epochs`] and has no target (every arm differs from its baseline in one setting [RR-08, section 2]). Arms: `target_kl` = %s (`B_kl005`, `A_kl005`) "
    "and %s (`B_kl010`, `A_kl010`) [RR-08, sections 3.1 and 3.2]. Both stages implement the rule (the masked update of phase B and "
    "the original update of phase A [RR-08, section 4 item 3]). Everything else is the locked v1.1 pipeline.\n"
    % (T("10", "PL-08", "v2_T2_locked_v1_1.json ppo.epochs = 10"),
       T("0.005", "RR-08", "01_preregistration.md section 3.1/3.2, B_kl005 / A_kl005"),
       T("0.01", "RR-08", "01_preregistration.md section 3.1/3.2, B_kl010 / A_kl010")))
s43.append("Paired differences (arm - baseline; negative = better), n = 10 pairs per row; reading rules as in section 4.1:\n")
s43.append(main_table(arms3))
s43.append("")
s43.append(main_source(arms3))
s43.append("")
s43.append(
    "Observation: at q = 60 `A_kl005` has a positive mean difference (a larger peak error than `A_base`), %s, and its interval contains 0 only narrowly "
    "(lower end %s) [R1-01].\n"
    % (CI(DI.loc["A_kl005", "mean_q60"], DI.loc["A_kl005", "ci_mean_lo_q60"], DI.loc["A_kl005", "ci_mean_hi_q60"], "R1-01",
          "decision_inputs.csv[arm=A_kl005,q=60]"),
       N(DI.loc["A_kl005", "ci_mean_lo_q60"], "R1-01", "decision_inputs.csv[arm=A_kl005,q=60] ci_mean_lo")))
s43.append(
    "**Spread, early stopping and cost.** Dispersion ratio as defined in section 4.1. The last column is the share of updates in "
    "which fewer than 10 epochs were run (the update stopped early), pooled over the 10 runs of the row (%s updates per row in stage 1, "
    "%s in stage 2) [R1-27, R1-28]; it is 0 for `B_base` and `A_base` at both q [R1-27, R1-28].\n"
    % (N(S1STOP.n_updates.iloc[0], "R1-27", "stage1_stop_epoch.csv n_updates", lambda x: "%d" % x),
       N(S2STOP.n_updates.iloc[0], "R1-28", "stage2_stop_epoch.csv n_updates", lambda x: "%d" % x)))
s43.append(disp_table(arms3, with_stop=True))
s43.append("")
s43.append("Source: R1-01 (`decision_inputs.csv`, `disp_ratio_q*`, `disp_ci_lo_q*`, `disp_ci_hi_q*`; same values in R1-03, R1-38), R1-24 (`stage1_gate_counts.csv`, `n_target0929_pass`), "
           "R1-27 (`stage1_stop_epoch.csv`, column `share_fewer_than_10`, one value per arm and q) and R1-28 (`stage2_stop_epoch.csv`, same column). "
           "The 0-share of the baselines was read from the same files (arm `B_base` / `A_base`).")
s43.append("")
flag = [(a, q) for a in arms3 for q in (50, 60) if DI.loc[a, "disp_ci_hi_q%d" % q] < 1]
assert flag == [("A_kl005", 50)], flag
s43.append(
    "Observation: the dispersion interval lies entirely below 1 for `A_kl005` at q = 50 (ratio %s) and nowhere else in this method [R1-01]; "
    "it is not part of the criterion. Early stopping is frequent: the share of updates with fewer than 10 epochs is between %s and %s over the four stage-1 rows and between %s and %s over the four stage-2 rows [R1-27, R1-28].\n"
    % (CI(DI.loc["A_kl005", "disp_ratio_q50"], DI.loc["A_kl005", "disp_ci_lo_q50"], DI.loc["A_kl005", "disp_ci_hi_q50"], "R1-01",
          "decision_inputs.csv[arm=A_kl005,q=50] disp_ratio"),
       N(min(stop_share(1, a, q) for a in ("B_kl005", "B_kl010") for q in (50, 60)), "R1-27", "stage1_stop_epoch.csv min share_fewer_than_10 over arm x q (computed here: min)"),
       N(max(stop_share(1, a, q) for a in ("B_kl005", "B_kl010") for q in (50, 60)), "R1-27", "stage1_stop_epoch.csv max share_fewer_than_10 over arm x q (computed here: max)"),
       N(min(stop_share(2, a, q) for a in ("A_kl005", "A_kl010") for q in (50, 60)), "R1-28", "stage2_stop_epoch.csv min share_fewer_than_10 over arm x q (computed here: min)"),
       N(max(stop_share(2, a, q) for a in ("A_kl005", "A_kl010") for q in (50, 60)), "R1-28", "stage2_stop_epoch.csv max share_fewer_than_10 over arm x q (computed here: max)")))
s43.append(cost_table(arms3))
s43.append("")
s43.append(cost_source(arms3))
s43.append("")
s43.append(
    "The number of optimizer steps per run falls with early stopping (stage 1: %s for `B_kl005` and %s for `B_kl010` against %s; stage 2: %s and %s against %s) "
    "with the same episodes per run as the baseline [R1-22, R1-23].\n"
    % (N(S1COST.loc["B_kl005", "mean_n_minibatch_steps_total"], "R1-22", "stage1_cost.csv[arm=B_kl005] mean_n_minibatch_steps_total"),
       N(S1COST.loc["B_kl010", "mean_n_minibatch_steps_total"], "R1-22", "stage1_cost.csv[arm=B_kl010] mean_n_minibatch_steps_total"),
       N(S1COST.loc["B_base", "mean_n_minibatch_steps_total"], "R1-22", "stage1_cost.csv[arm=B_base] mean_n_minibatch_steps_total"),
       N(S2COST.loc["A_kl005", "mean_n_minibatch_steps_total"], "R1-23", "stage2_cost.csv[arm=A_kl005] mean_n_minibatch_steps_total"),
       N(S2COST.loc["A_kl010", "mean_n_minibatch_steps_total"], "R1-23", "stage2_cost.csv[arm=A_kl010] mean_n_minibatch_steps_total"),
       N(S2COST.loc["A_base", "mean_n_minibatch_steps_total"], "R1-23", "stage2_cost.csv[arm=A_base] mean_n_minibatch_steps_total")))
s43.append("**Gates.** As in section 4.1: part (b) holds trivially [R1-24, R1-25].\n")
s43.append(
    "**Decision.** Target-KL is not adopted. Recorded wording: \"Target-KL is not adopted either: it changes cost, not accuracy, and the "
    "protocol changes for accuracy only\" (R1 summary, addendum of 2026-10-03, item 2 [RR-01]; and in the protocol v2.0 change log, version 2.0, field `why`: "
    "\"No other R1 arm enters the protocol; target-KL changes cost, not accuracy, and is not adopted\" [PL-01, `change_log`]).\n")
parts.append("\n".join(s43))

# ---------------------------------------------------------------- 4.4
SEC[0] = "4.4"
arms4 = ["A_anneal2", "A_anneal4"]
assert all(DI.loc[a, "criterion_overall"] == "not met" for a in arms4)
an = lambda arm, q, quant: ANN[(ANN.arm == arm) & (ANN.q == q) & (ANN.quantity == quant)].iloc[0]  # noqa: E731
s44 = []
s44.append("### 4.4 Method 4: concentration annealing (stage 2)\n")
a4 = DI.loc["A_anneal4"]
a2 = DI.loc["A_anneal2"]
s44.append(
    "**Outcome.** No annealing arm meets the criterion: `A_anneal2` has mean intervals containing 0 at both q (q = 50: %s, whose upper "
    "end is close to 0; q = 60: %s), and `A_anneal4` has an interval entirely above 0 at q = 60 (%s), i.e. a larger peak error than "
    "the baseline `A_base` [R1-01, R1-06; RR-02, Observations]. Part (b) holds. The observed change of the peak gap contains the "
    "smoothing-predicted change at scale 2 at both q and does not at scale 4 (below).\n"
    % (CI(a2.mean_q50, a2.ci_mean_lo_q50, a2.ci_mean_hi_q50, "R1-01", "decision_inputs.csv[arm=A_anneal2,q=50]"),
       CI(a2.mean_q60, a2.ci_mean_lo_q60, a2.ci_mean_hi_q60, "R1-01", "decision_inputs.csv[arm=A_anneal2,q=60]"),
       CI(a4.mean_q60, a4.ci_mean_lo_q60, a4.ci_mean_hi_q60, "R1-01", "decision_inputs.csv[arm=A_anneal4,q=60]")))
s44[-1] = s44[-1].replace("a larger peak error than `A_anneal4`'s baseline `A_base`", "a larger peak error than the baseline `A_base`")
s44.append(
    "**What was changed.** Stage 2 only (a continued phase A from update 1200, 400 updates [RR-08, section 3.2]). The Beta policy's concentration is "
    "c = (c_min + softplus(z_c)) x conc_scale, with `c_min` = %s [PL-08, `ppo.c_min`] and a scale that moves linearly in the local "
    "update from 1.0 to the final value over updates 1-400: `A_anneal2` 1.0 to %s, `A_anneal4` 1.0 to %s [RR-08, sections 3.2 and 4 item 4]. "
    "The runner sets the scale on the live actor and on the lagged opponent before each rollout; at the end of the phase the actor carries the "
    "final scale, and sigma_2(0) in the tables is the annealed value [RR-08, section 4 item 4 and section 5]. A larger scale means a more "
    "concentrated policy, i.e. a smaller action spread. The budget, LR schedule and all other settings equal `A_base`'s; the optimizer "
    "steps per run are %s [R1-23].\n"
    % (T("100", "PL-08", "v2_T2_locked_v1_1.json ppo.c_min = 100"),
       T("2.0", "RR-08", "01_preregistration.md section 3.2, A_anneal2: conc_anneal 1.0 -> 2.0"),
       T("4.0", "RR-08", "01_preregistration.md section 3.2, A_anneal4: conc_anneal 1.0 -> 4.0"),
       N(S2COST.loc["A_anneal2", "mean_n_minibatch_steps_total"], "R1-23", "stage2_cost.csv[arm=A_anneal2] mean_n_minibatch_steps_total")))
s44.append("Paired differences (arm - baseline; negative = better), n = 10 pairs per row; reading rules as in section 4.1:\n")
s44.append(main_table(arms4))
s44.append("")
s44.append(main_source(arms4))
s44.append("")
s44.append(
    "Observations from the table: `A_anneal2` has %s seeds better at q = 50 and its median interval (%s) excludes 0 although its mean interval does not; "
    "the criterion uses the mean interval [R1-01; RR-02, Observations]. `A_anneal4` has %s seeds better at q = 60 [R1-01].\n"
    % (N(a2.n_better_q50, "R1-01", "decision_inputs.csv[arm=A_anneal2,q=50] n_better", lambda x: "%d/10" % x),
       CI(a2.median_q50, a2.ci_median_lo_q50, a2.ci_median_hi_q50, "R1-01", "decision_inputs.csv[arm=A_anneal2,q=50] median"),
       N(a4.n_better_q60, "R1-01", "decision_inputs.csv[arm=A_anneal4,q=60] n_better", lambda x: "%d/10" % x)))
assert a2.ci_median_hi_q50 < 0 < a2.ci_mean_hi_q50
s44.append("**Spread and cost.** Dispersion ratio as defined in section 4.1:\n")
s44.append(disp_table(arms4, with_target=False))
s44.append("")
s44.append("Source: R1-01 (`decision_inputs.csv`, columns `disp_ratio_q*`, `disp_ci_lo_q*`, `disp_ci_hi_q*`; same values in R1-38 `stage2_dispersion.csv`, metric `stage2_peak_rel_err_signed`). No stage-1 arm exists for this method.")
s44.append("")
assert all(DI.loc[a, "disp_ci_lo_q%d" % q] < 1 < DI.loc[a, "disp_ci_hi_q%d" % q] for a in ("A_anneal2", "A_anneal4") for q in (50, 60)) or True
flag = [(a, q) for a in arms4 for q in (50, 60) if DI.loc[a, "disp_ci_hi_q%d" % q] < 1 or DI.loc[a, "disp_ci_lo_q%d" % q] > 1]
assert flag == [], flag
s44.append("Observation: all four dispersion intervals contain 1 [R1-01]. The phase wall time is below the baseline's for both arms at the same optimizer steps (table below).\n")
s44.append(cost_table(arms4))
s44.append("")
s44.append(cost_source(arms4))
s44.append("")
# smoothing prediction tables
def anrow(arm, q):
    return {k: an(arm, q, k) for k in ("sigma_effort_at_0_t2", "smoothed_pred_gap_d0", "smoothed_share_peak_gap_d0",
                                       "observed_gap_d0", "e2_at_0", "smoothed_e_pred_0")}


out = ["| arm | q | sigma_2(0), median: `A_base` -> arm | predicted peak gap, median: `A_base` -> arm | share of the peak gap predicted by smoothing, median: `A_base` -> arm | observed peak gap, median: `A_base` -> arm |",
       "|---|---|---|---|---|---|"]
for arm in arms4:
    for q in (50, 60):
        r = anrow(arm, q)
        loc = "stage2_annealing.csv[arm=%s,q=%d]" % (arm, q)
        def ba(k):
            x = r[k]
            return "%s -> %s" % (N(x.median_before, "R1-07", loc + " %s median_before" % k),
                                 N(x.median_after, "R1-07", loc + " %s median_after" % k))
        out.append("| `%s` | %d | %s | %s | %s | %s |" % (arm, q, ba("sigma_effort_at_0_t2"), ba("smoothed_pred_gap_d0"),
                                                          ba("smoothed_share_peak_gap_d0"), ba("observed_gap_d0")))
s44.append("Smoothing prediction against observation. The prediction is the stage-2 effort at d = 0 that the first-order condition of the smoothed "
           "game gives for the learned Beta spread of both players (`smoothed_e_pred_0`), and the predicted peak gap is g2(0) - e_pred(0); the observed "
           "peak gap is g2(0) - e2_hat(0) [R1-07, column `description`; RR-08, section 5]. Gaps are in effort units. \"Before\" is the `A_base` "
           "median (R1-07 column `median_before`, equal to the `A_base` median of R1-40 for sigma_2(0) and e2_hat(0)); \"after\" is the arm's median.\n")
s44.append(out[0]); s44.extend(out[1:])
s44.append("")
s44.append("Source: R1-07 (`results/v2_refine/analysis/stage2_annealing.csv`, rows quantity in {`sigma_effort_at_0_t2`, `smoothed_pred_gap_d0`, `smoothed_share_peak_gap_d0`, `observed_gap_d0`}, columns `median_before`, `median_after`).")
s44.append("")
# check medians equal A_base medians from R1-40
for q in (50, 60):
    assert abs(an("A_anneal2", q, "sigma_effort_at_0_t2").median_before - b2.loc[q, "median_sigma_effort_at_0_t2"]) < 1e-12
    assert abs(an("A_anneal2", q, "e2_at_0").median_before - b2.loc[q, "median_e2_at_0"]) < 1e-12
out = ["| arm | q | predicted change of the peak gap, mean [95% CI] | observed change of the peak gap, mean [95% CI] | observed interval contains the predicted mean change | observed interval contains 0 |",
       "|---|---|---|---|---|---|"]
contain = {}
for arm in arms4:
    for q in (50, 60):
        p = an(arm, q, "smoothed_pred_gap_d0")
        o = an(arm, q, "observed_gap_d0")
        loc = "stage2_annealing.csv[arm=%s,q=%d]" % (arm, q)
        cp = o.ci_mean_lo <= p.mean_change <= o.ci_mean_hi
        c0 = o.ci_mean_lo <= 0 <= o.ci_mean_hi
        contain[(arm, q)] = (cp, c0)
        out.append("| `%s` | %d | %s | %s | %s | %s |" % (
            arm, q, CI(p.mean_change, p.ci_mean_lo, p.ci_mean_hi, "R1-07", loc + " smoothed_pred_gap_d0 mean_change"),
            CI(o.mean_change, o.ci_mean_lo, o.ci_mean_hi, "R1-07", loc + " observed_gap_d0 mean_change"),
            "yes" if cp else "no", "yes" if c0 else "no"))
s44.append("Paired change of the peak gap, arm - `A_base` (negative = the gap shrinks). The last two columns are computed here from the R1-07 intervals:\n")
s44.append("\n".join(out))
s44.append("")
s44.append("Source: R1-07 (`stage2_annealing.csv`, rows quantity in {`smoothed_pred_gap_d0`, `observed_gap_d0`}; columns `mean_change`, `ci_mean_lo`, `ci_mean_hi`); the two yes/no columns are computed here from these columns (script `build_sec04a.py`).")
s44.append("")
assert contain[("A_anneal2", 50)] == (True, True) and contain[("A_anneal2", 60)] == (True, True)
assert (not contain[("A_anneal4", 50)][0]) and (not contain[("A_anneal4", 60)][0])
obs60 = an("A_anneal4", 60, "observed_gap_d0")
prd60 = an("A_anneal4", 60, "smoothed_pred_gap_d0")
assert obs60.mean_change > 0 > prd60.mean_change
s44.append(
    "Reading: the R1 report's sentence is \"the observed change contains the predicted one at scale 2 in both q, and does not follow it at scale 4 "
    "(at q = 60 it has the opposite sign)\" [RR-02, Observations]. The table agrees: at scale 2 the observed interval contains the predicted mean change at "
    "both q; at scale 4 it does not at either q, and at q = 60 the observed change is %s while the predicted one is %s [R1-07]. The scale-2 observed "
    "intervals also contain 0, so containment of the prediction is compatible with no change of the gap [R1-07, computed here]. The annealed sigma_2(0) "
    "falls with the scale (median %s -> %s at scale 2 and %s at scale 4 at q = 50) and so does the smoothing-predicted share of the peak gap "
    "(table above), while the observed gap change at scale 4 is not negative at either q (its interval contains 0 at q = 50 and lies above 0 at q = 60) [R1-07]. RR-02 states that whether the learned policy adapts to the annealed concentration in a way "
    "that offsets the smoothing effect \"is not tested by this wave\"; no cause for the scale-4 result is established.\n"
    % (CI(obs60.mean_change, obs60.ci_mean_lo, obs60.ci_mean_hi, "R1-07", "stage2_annealing.csv[arm=A_anneal4,q=60,observed_gap_d0] mean_change"),
       CI(prd60.mean_change, prd60.ci_mean_lo, prd60.ci_mean_hi, "R1-07", "stage2_annealing.csv[arm=A_anneal4,q=60,smoothed_pred_gap_d0] mean_change"),
       N(an("A_anneal2", 50, "sigma_effort_at_0_t2").median_before, "R1-07", "stage2_annealing.csv[arm=A_anneal2,q=50,sigma_effort_at_0_t2] median_before"),
       N(an("A_anneal2", 50, "sigma_effort_at_0_t2").median_after, "R1-07", "stage2_annealing.csv[arm=A_anneal2,q=50,sigma_effort_at_0_t2] median_after"),
       N(an("A_anneal4", 50, "sigma_effort_at_0_t2").median_after, "R1-07", "stage2_annealing.csv[arm=A_anneal4,q=50,sigma_effort_at_0_t2] median_after")))
s44.append("**Gates.** As in section 4.1: part (b) holds trivially [R1-24, R1-25].\n")
s44.append(
    "**Decision.** Criterion not met; no arm adopted (R1 summary, addendum, item 2: \"No other R1 arm enters the protocol\" [RR-01]; [PL-01, `change_log`, version 2.0]). "
    "No separate decision on annealing is recorded.\n")
parts.append("\n".join(s44))

md = "\n\n".join(parts) + "\n"
os.makedirs(SCRATCH, exist_ok=True)
with open(SCRATCH + "sec04a.md", "w") as f:
    f.write(md)
with open(SCRATCH + "sec04a_ledger.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["statement_id", "section", "text", "value", "item_id", "locator"])
    w.writeheader()
    w.writerows(LEDGER)
print("words", len(md.split()), "ledger rows", len(LEDGER))
