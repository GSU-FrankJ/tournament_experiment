"""Build section 5 (D1 and D2) of the T=2 refinement report from the evidence pack.

Reads only files under the evidence pack; writes SEC05.md and the ledger CSV under SCRATCH.
Run: OMP_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python build_sec05.py
"""
import csv
import json
import os

import pandas as pd

PK = ("/home/fjiang4/tournament_experiment/.claude/worktrees/t2-refine-pack/"
      "reports/t2_refine_100526/evidence")
OUT = ("/tmp/claude-1331199693/-home-fjiang4-tournament-experiment--claude-worktrees-"
       "r2c-sampler-protocol-v2-1-c4c0f1/152f0306-5492-45a1-94d5-59493fa0d141/scratchpad/"
       "report_parts")

P = {
    "R1-10": "results/v2_refine/d1_clamp/d1_flags.csv",
    "R1-11": "results/v2_refine/d2_verifier_sensitivity/detection_limits.csv",
    "R1-12": "results/v2_refine/d2_verifier_sensitivity/evaluations.csv",
    "R1-13": "results/v2_refine/d2_verifier_sensitivity/fits.csv",
    "R1-14": "results/v2_refine/d2_verifier_sensitivity/family_e_confirmation.csv",
    "R1-29": "results/v2_refine/analysis/decision_d1_flags.csv",
    "R1-30": "results/v2_refine/analysis/decision_d1_flags_arms.csv",
    "R1-31": "results/v2_refine/analysis/decision_d2_detection_limits.csv",
    "R1-35": "results/v2_refine/d1_clamp/d1_summary.json",
    "R1-36": "results/v2_refine/d1_clamp/d1_by_group.csv",
    "R2B-01": "results/v2_refine_r2b/analysis/decision_inputs.csv",
    "R2B-02": "results/v2_refine_r2b/analysis/criterion.csv",
    "R2B-03": "results/v2_refine_r2b/analysis/per_run.csv",
    "R2B-27": "results/v2_refine_r2b/diag_30510/tables/tab_d1_clamp_counts_phaseA_sum.csv",
    "R2B-29": "results/v2_refine_r2b/analysis/waveA_specifics.csv",
}


def rd(item):
    return pd.read_csv(os.path.join(PK, P[item]))


LEDGER = []
SEC = "5"


def rec(text, value, item, locator, sec=SEC):
    """Log one number as it appears in the text; return the text."""
    LEDGER.append({"statement_id": "S%04d" % (len(LEDGER) + 1), "section": sec,
                   "text": text, "value": value, "item_id": item, "locator": locator})
    return text


def g(x):
    """4 significant digits, as the round reports print them."""
    return "%.4g" % x


def N(x, item, loc, fmt=g):
    return rec(fmt(x), x, item, loc)


def I(x, item, loc):
    return rec("{:,}".format(int(x)), int(x), item, loc)


def I0(x, item, loc):
    """Integer without thousands separator."""
    return rec(str(int(x)), int(x), item, loc)


lines = []


def w(s=""):
    lines.append(s)


# --------------------------------------------------------------------------- data
d29 = rd("R1-29")
d10 = rd("R1-10")
d36 = rd("R1-36")
d30 = rd("R1-30")
r2b_crit = rd("R2B-02")
r2b_di = rd("R2B-01")
r2b_run = rd("R2B-03")
r2b_spec = rd("R2B-29")
r2b_d1 = rd("R2B-27")
d31 = rd("R1-31")
d11 = rd("R1-11")
d12 = rd("R1-12")
d13 = rd("R1-13")
d14 = rd("R1-14")
d35 = json.load(open(os.path.join(PK, P["R1-35"])))

v11 = d10[d10.group == "v11_reproduction"]


def hrow(q, ph):
    r = d29[(d29.q.astype(str) == str(q)) & (d29.phase == ph)]
    assert len(r) == 1
    return r.iloc[0]


# --------------------------------------------------------------------------- 5 header
w("## 5. Diagnostics D1 and D2")
w()
w("Both diagnostics were fixed in the R1 pre-registration as descriptive and report-only "
  "(section 6 of `reports/v2/refine/01_preregistration.md`, pack item RR-08). Neither changes "
  "a threshold, a gate or the protocol. D1 reads the `d1_*` columns of the logs of runs of the locked "
  "v1.1 pipeline and of the R1 pilot arms. D2 trains nothing: it evaluates closed-form perturbations "
  "of the equilibrium with the verifier and the gate thresholds of the locked v1.1 protocol.")
w()

# --------------------------------------------------------------------------- 5.1
w("### 5.1 D1: clamp of the Beta draws (clipped likelihood)")
w()
a_all = hrow("all", "A")
b_all = hrow("all", "B")
loc = "R1-29 row group=v11_reproduction"
w("**Outcome.** In the locked v1.1 pipeline (the 20 C-R1 runs, group `v11_reproduction`), M1 is "
  "exceeded in phase A (pooled median clamp fraction %s against the threshold %s) and not in "
  "phase B (median %s). M2 is exceeded in phase A in the pooled reading (%s of %s runs with a "
  "clamped-row gradient share above 1%%, maximum share %s) and not in phase B (%s of %s runs) [R1-29]. "
  "The flags are report-only: R1 applied no fix, and protocol v2.0 carries the issue as a known "
  "issue (PL-01, `known_issues`, \"Recorded, not fixed\"). The R2b test of the censored "
  "likelihood (`A_censored`, below) does not meet criterion part (a) [R2B-02]; the censored "
  "likelihood is not adopted (PI publication prompt, D1), `clamp_likelihood` stays `density`, "
  "and the inconsistency stays a recorded known issue (RR-12, section 1.3)."
  % (N(a_all.M1_median_over_runs, "R1-29", loc + ", q=all, phase=A, M1_median_over_runs"),
     N(a_all.M1_threshold, "R1-29", loc + ", M1_threshold"),
     N(b_all.M1_median_over_runs, "R1-29", loc + ", q=all, phase=B, M1_median_over_runs"),
     I0(a_all.M2_runs_share_above_threshold, "R1-29", loc + ", q=all, phase=A, M2_runs_share_above_threshold"),
     I0(a_all.n_runs, "R1-29", loc + ", q=all, phase=A, n_runs"),
     N(a_all.M2_max_share, "R1-29", loc + ", q=all, phase=A, M2_max_share"),
     I0(b_all.M2_runs_share_above_threshold, "R1-29", loc + ", q=all, phase=B, M2_runs_share_above_threshold"),
     I0(b_all.n_runs, "R1-29", loc + ", q=all, phase=B, n_runs")))
w()

w("**What is measured.** The policy draws a Beta action and clips it to [c, 1 - c] with "
  "c = 1e-6 before executing it; the stored log-probability is the Beta density at the clipped "
  "value. When the raw draw was clamped, the executed event is {A <= c} (or {A >= 1 - c}), whose "
  "likelihood is a censored mass, not the density; the density and the log-mass differ "
  "(`reports/v2/refine/02_d1_clamp.md`, report only, needs item). A clamp hit is a raw draw below c "
  "or above 1 - c, counted before the clip (RR-08 section 6; c = %s in R1-35). The two "
  "pre-registered materiality flags are (RR-08 section 6):"
  % rec("1e-6", d35["thresholds"]["clamp"], "R1-35", "thresholds.clamp"))
w()
w("- **M1**: the median over runs of the per-phase clamp fraction among learner policy rows "
  "exceeds %s (strict). Learner policy rows are the rows that enter the actor loss: all stage-2 "
  "rows in phase A, the stage-1 rows in the frozen phase B (`02_d1_clamp.md` section 1, report only)."
  % N(d35["thresholds"]["M1"], "R1-35", "thresholds.M1", fmt=lambda x: "%g" % x))
w("- **M2**: the gradient share of the clamped rows exceeds %s%% at any saved update in more than "
  "%s of the 20 runs. The saved updates are three buffers per run and phase (RR-08 section 6). "
  "\"More than 2 of the 20 runs\" is read as the 20 C-R1 runs pooled over both q, per phase; the "
  "per-q counts out of 10 runs are shown beside the pooled row (RR-08, Addendum 1, item 2)."
  % (N(d35["thresholds"]["M2_share"] * 100, "R1-35", "thresholds.M2_share (x100)",
       fmt=lambda x: "%g" % x),
     I0(d35["thresholds"]["M2_more_than_runs"], "R1-35", "thresholds.M2_more_than_runs")))
w()

# Table 1 headline
w("**Table 5.1a. D1 flags of the locked pipeline** (`v11_reproduction`; per q 10 runs, pooled 20 runs).")
w()
w("| q | phase | n_runs | M1: median over runs of clamp fraction | M1 outcome | "
  "M2: runs with share > 0.01 (needs more than 2) | M2: max share | M2 outcome |")
w("|---|---|---|---|---|---|---|---|")
for q in ("50", "60", "all"):
    for ph in ("A", "B"):
        r = hrow(q, ph)
        L = "R1-29 q=%s phase=%s" % (q, ph)
        w("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            q, ph, I0(r.n_runs, "R1-29", L + " n_runs"),
            N(r.M1_median_over_runs, "R1-29", L + " M1_median_over_runs"),
            r.M1_outcome,
            I0(r.M2_runs_share_above_threshold, "R1-29", L + " M2_runs_share_above_threshold"),
            N(r.M2_max_share, "R1-29", L + " M2_max_share"),
            r.M2_outcome))
w()
w("Source: R1-29 (`results/v2_refine/analysis/decision_d1_flags.csv`, rows `group = v11_reproduction`; "
  "columns `n_runs`, `M1_median_over_runs`, `M1_outcome`, `M2_runs_share_above_threshold`, "
  "`M2_max_share`, `M2_outcome`; the threshold columns are 0.001 for M1 and 2 runs for M2).")
w()
r50 = v11[(v11.q.astype(str) == "50") & (v11.phase == "A")].iloc[0]
rall = v11[(v11.q.astype(str) == "all") & (v11.phase == "A")].iloc[0]
w("Observations. The M1 median at q = 50 (%s) lies just above the threshold %s; at q = 60 it is %s. "
  "In phase A, %s of the %s pooled runs have a per-run clamp fraction above the threshold; the "
  "largest per-run fraction is %s [R1-10, columns `M1_runs_above_threshold`, "
  "`M1_max_run_fraction`]. The pooled M2 exceedance rests on %s runs: %s at q = 50 and %s at "
  "q = 60 [R1-29]; read per q, M2 is not exceeded at either q. The largest ratio of the clamped "
  "rows' gradient norm to the whole-buffer gradient norm over the saved buffers is %s (information "
  "only, `M2_max_clamped_over_all`) [R1-10]."
  % (N(hrow("50", "A").M1_median_over_runs, "R1-29", "R1-29 q=50 phase=A M1_median_over_runs"),
     "0.001",
     N(hrow("60", "A").M1_median_over_runs, "R1-29", "R1-29 q=60 phase=A M1_median_over_runs"),
     I0(rall.M1_runs_above_threshold, "R1-10", "group=v11_reproduction q=all phase=A M1_runs_above_threshold"),
     I0(rall.n_runs, "R1-10", "group=v11_reproduction q=all phase=A n_runs"),
     N(rall.M1_max_run_fraction, "R1-10", "group=v11_reproduction q=all phase=A M1_max_run_fraction"),
     I0(rall.M2_runs_share_above_threshold, "R1-10", "group=v11_reproduction q=all phase=A M2_runs_share_above_threshold"),
     I0(hrow("50", "A").M2_runs_share_above_threshold, "R1-29", "R1-29 q=50 phase=A M2_runs_share_above_threshold"),
     I0(hrow("60", "A").M2_runs_share_above_threshold, "R1-29", "R1-29 q=60 phase=A M2_runs_share_above_threshold"),
     N(rall.M2_max_clamped_over_all, "R1-10", "group=v11_reproduction q=all phase=A M2_max_clamped_over_all")))
w()

# Where the hits are
w("**Where the hits are.** All phase-A hits of the learner are lower-clamp hits at stage-2 rows "
  "with |d| >= 2q; there are no hits for |d| < 2q, none at the upper clamp, and none in phase B.")
w()
w("**Table 5.1b. Clamp hits of the locked pipeline by region** (`v11_reproduction`, learner rows; "
  "rows and hits summed over the 10 runs of each q and over all local updates of the phase).")
w()
w("| q | phase A, rows with \\|d\\| < 2q: rows | hits | phase A, rows with \\|d\\| >= 2q: rows | hits | "
  "hit fraction | upper-clamp hits (all phase-A categories) | phase B, stage-1 policy rows: rows | hits |")
w("|---|---|---|---|---|---|---|---|---|")


def g36(q, ph, cat):
    r = d36[(d36.group == "v11_reproduction") & (d36.q.astype(str) == str(q)) &
            (d36.phase == ph) & (d36.category == cat)]
    assert len(r) == 1, (q, ph, cat)
    return r.iloc[0]


tot = {"in_rows": 0, "in_hits": 0, "out_rows": 0, "out_hits": 0}
for q in ("50", "60"):
    fi, fo = g36(q, "A", "learner_final_in"), g36(q, "A", "learner_final_out")
    s1 = g36(q, "B", "learner_s1")
    hi_all = int(d36[(d36.group == "v11_reproduction") & (d36.q.astype(str) == q) &
                     (d36.phase == "A")].hi_sum.sum())
    L = "R1-36 group=v11_reproduction q=%s phase=%%s category=%%s " % q
    w("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        q,
        I(fi.rows_sum, "R1-36", (L % ("A", "learner_final_in")) + "rows_sum"),
        I0(fi.hit_sum, "R1-36", (L % ("A", "learner_final_in")) + "hit_sum"),
        I(fo.rows_sum, "R1-36", (L % ("A", "learner_final_out")) + "rows_sum"),
        I(fo.hit_sum, "R1-36", (L % ("A", "learner_final_out")) + "hit_sum"),
        N(fo.frac_sum, "R1-36", (L % ("A", "learner_final_out")) + "frac_sum"),
        I0(hi_all, "R1-36", "sum of hi_sum over phase-A rows of q=%s" % q),
        I(s1.rows_sum, "R1-36", (L % ("B", "learner_s1")) + "rows_sum"),
        I0(s1.hit_sum, "R1-36", (L % ("B", "learner_s1")) + "hit_sum")))
    tot["in_rows"] += int(fi.rows_sum)
    tot["in_hits"] += int(fi.hit_sum)
    tot["out_rows"] += int(fo.rows_sum)
    tot["out_hits"] += int(fo.hit_sum)
w("| pooled (computed here) | %s | %s | %s | %s | %s | 0 | %s | 0 |" % (
    I(tot["in_rows"], "R1-36", "computed: sum over q of learner_final_in rows_sum (phase A)"),
    I0(tot["in_hits"], "R1-36", "computed: sum over q of learner_final_in hit_sum (phase A)"),
    I(tot["out_rows"], "R1-36", "computed: sum over q of learner_final_out rows_sum (phase A)"),
    I(tot["out_hits"], "R1-36", "computed: sum over q of learner_final_out hit_sum (phase A)"),
    rec(g(tot["out_hits"] / tot["out_rows"]), tot["out_hits"] / tot["out_rows"], "R1-36",
        "computed: pooled out hits / pooled out rows"),
    I(2 * 3072000, "R1-36", "computed: 2 x learner_s1 rows_sum (phase B)")))
w()
w("Source: R1-36 (`results/v2_refine/d1_clamp/d1_by_group.csv`; categories `learner_final_in`, "
  "`learner_final_out` for phase A and `learner_s1` for phase B, columns `rows_sum`, `hit_sum`, "
  "`frac_sum`, `hi_sum`). The pooled row is computed here from the two q rows (script "
  "`build_sec05.py`); in phase A every learner policy row is a stage-2 row (`learner_s2` equals "
  "`learner_policy` in R1-36).")
w()
o50, o60 = g36("50", "A", "opponent_s2"), g36("60", "A", "opponent_s2")
w("The opponent's stage-2 draws, which are never trained on (`policy_status` = \"opponent (not "
  "trained)\" in R1-36), have hit counts of the same size: %s (q = 50) and %s (q = 60) against %s "
  "and %s for the learner [R1-36]. In phase B the stage-2 rows of the learner are masked "
  "(non-policy) and also show zero hits [R1-36, category `learner_s2`]."
  % (I(o50.hit_sum, "R1-36", "R1-36 opponent_s2 q=50 phase=A hit_sum"),
     I(o60.hit_sum, "R1-36", "R1-36 opponent_s2 q=60 phase=A hit_sum"),
     I(g36("50", "A", "learner_s2").hit_sum, "R1-36", "R1-36 learner_s2 q=50 phase=A hit_sum"),
     I(g36("60", "A", "learner_s2").hit_sum, "R1-36", "R1-36 learner_s2 q=60 phase=A hit_sum")))
w()

# alpha<1 (report only)
a50, a60 = 1051545, 1017185
rows = 8192000
w("Rows with alpha < 1 (report only; the table `d1_alpha_beta_lt1.csv` is not in the pack, needs item): "
  "in phase A, %s of %s learner policy rows at q = 50 and %s of %s at q = 60 have alpha < 1 "
  "(shares %s and %s, computed here from these counts); no row has beta < 1; the smallest alpha is "
  "%s and %s. In phase B no row has alpha < 1 (smallest alpha %s and %s) (`02_d1_clamp.md` section 4)."
  % (I(a50, "NEEDS:d1_alpha_beta_lt1.csv", "02_d1_clamp.md s4, v11_reproduction q=50 A n_alpha_lt1_sum"),
     I(rows, "NEEDS:d1_alpha_beta_lt1.csv", "02_d1_clamp.md s4, v11_reproduction q=50 A pol_rows_sum"),
     I(a60, "NEEDS:d1_alpha_beta_lt1.csv", "02_d1_clamp.md s4, v11_reproduction q=60 A n_alpha_lt1_sum"),
     I(rows, "NEEDS:d1_alpha_beta_lt1.csv", "02_d1_clamp.md s4, v11_reproduction q=60 A pol_rows_sum"),
     rec(g(a50 / rows), a50 / rows, "NEEDS:d1_alpha_beta_lt1.csv", "computed: 1051545/8192000"),
     rec(g(a60 / rows), a60 / rows, "NEEDS:d1_alpha_beta_lt1.csv", "computed: 1017185/8192000"),
     rec("0.003824", 0.003824, "NEEDS:d1_alpha_beta_lt1.csv", "02_d1_clamp.md s4 alpha_min q=50 A"),
     rec("0.09319", 0.09319, "NEEDS:d1_alpha_beta_lt1.csv", "02_d1_clamp.md s4 alpha_min q=60 A"),
     rec("81.48", 81.48, "NEEDS:d1_alpha_beta_lt1.csv", "02_d1_clamp.md s4 alpha_min q=50 B"),
     rec("78.04", 78.04, "NEEDS:d1_alpha_beta_lt1.csv", "02_d1_clamp.md s4 alpha_min q=60 B")))
w()
w("Size of the likelihood mismatch in the saved buffers (report only, needs item "
  "`d1_logdiff_by_group.csv` / `d1_buffers.csv`): in the %s saved phase-A buffers per q, the clamped "
  "policy rows number %s (q = 50) and %s (q = 60); log density minus log censored mass has median "
  "%s and %s and range [%s, %s] and [%s, %s] (`02_d1_clamp.md` section 5)."
  % (rec("30", 30, "NEEDS:d1_buffers.csv", "02_d1_clamp.md s5 n_buffers"),
     rec("222", 222, "NEEDS:d1_buffers.csv", "02_d1_clamp.md s5 q=50 A n_clamped_policy_rows"),
     rec("55", 55, "NEEDS:d1_buffers.csv", "02_d1_clamp.md s5 q=60 A n_clamped_policy_rows"),
     rec("11.53", 11.53, "NEEDS:d1_logdiff_by_group.csv", "02_d1_clamp.md s5 q=50 A diff_median"),
     rec("12.46", 12.46, "NEEDS:d1_logdiff_by_group.csv", "02_d1_clamp.md s5 q=60 A diff_median"),
     rec("8.705", 8.705, "NEEDS:d1_logdiff_by_group.csv", "02_d1_clamp.md s5 q=50 A diff_min"),
     rec("13.33", 13.33, "NEEDS:d1_logdiff_by_group.csv", "02_d1_clamp.md s5 q=50 A diff_max"),
     rec("11.9", 11.9, "NEEDS:d1_logdiff_by_group.csv", "02_d1_clamp.md s5 q=60 A diff_min"),
     rec("14.19", 14.19, "NEEDS:d1_logdiff_by_group.csv", "02_d1_clamp.md s5 q=60 A diff_max")))
w()

# Table 5.1c pilot arms
w("**Table 5.1c. D1 flags of the R1 pilot arms** (pooled over both q, 20 runs per arm; phase B for the "
  "stage-1 arms, phase A for the stage-2 arms).")
w()
w("| arm | phase | M1: median over runs | M1 outcome | M2: runs with share > 0.01 | M2: max share | M2 outcome |")
w("|---|---|---|---|---|---|---|")
for _, r in d30.iterrows():
    arm = r.group.split("/")[1]
    L = "R1-30 group=%s " % r.group
    w("| `%s` | %s | %s | %s | %s | %s | %s |" % (
        arm, r.phase, N(r.M1_median_over_runs, "R1-30", L + "M1_median_over_runs"), r.M1_outcome,
        I0(r.M2_runs_share_above_threshold, "R1-30", L + "M2_runs_share_above_threshold"),
        N(r.M2_max_share, "R1-30", L + "M2_max_share"), r.M2_outcome))
w()
w("Source: R1-30 (`results/v2_refine/analysis/decision_d1_flags_arms.csv`, all rows; columns "
  "`M1_median_over_runs`, `M1_outcome`, `M2_runs_share_above_threshold`, `M2_max_share`, "
  "`M2_outcome`; `n_runs` is 20 in every row).")
w()
ab = d30[d30.group == "stage2/A_base"].iloc[0]
n_ab = d36[(d36.group == "stage2/A_base") & (d36.category == "learner_policy")]
n_v = d36[(d36.group == "v11_reproduction") & (d36.phase == "A") & (d36.category == "learner_policy")]
exceeded1 = d30[(d30.phase == "A") & (d30.M1_outcome == "exceeded")]
notexc1 = d30[(d30.phase == "A") & (d30.M1_outcome != "exceeded")]
w("Observations. None of the eight phase-B stage-1 arms is flagged (median 0 and maximum share 0 in every row). Of the ten phase-A "
  "arms, M1 is exceeded in %s (all but %s) and M2 only in `A_base` (%s of %s runs, maximum share %s). "
  "These stage-2 arms branch from the u1200 state (RR-08, section 2) and cover %s (run, update) pairs "
  "per q group (`n_updates` of `stage2/A_base` in R1-36) against %s for the C-R1 rows, so their "
  "medians are not the same quantity as the 1600-update medians in Table 5.1a [R1-36]. The pilot "
  "arms were not designed to test D1; the table records whether the arm changed the flags."
  % (I0(len(exceeded1), "R1-30", "computed: count of phase-A arms with M1_outcome=exceeded"),
     ", ".join("`%s`" % x.split("/")[1] for x in notexc1.group),
     I0(ab.M2_runs_share_above_threshold, "R1-30", "R1-30 group=stage2/A_base M2_runs_share_above_threshold"),
     I0(ab.n_runs, "R1-30", "R1-30 group=stage2/A_base n_runs"),
     N(ab.M2_max_share, "R1-30", "R1-30 group=stage2/A_base M2_max_share"),
     I(n_ab[n_ab.q.astype(str) == "50"].n_updates.iloc[0], "R1-36", "R1-36 stage2/A_base q=50 A learner_policy n_updates"),
     I(n_v[n_v.q.astype(str) == "50"].n_updates.iloc[0], "R1-36", "R1-36 v11_reproduction q=50 A learner_policy n_updates")))
w()
w("Runs of `A_detmean` (pathwise phase, no action drawn) carry no D1 columns by construction: 20 of "
  "the %s runs analysed (report only, `02_d1_clamp.md` section 1; `n_runs` = %s in R1-35)."
  % (I0(d35["n_runs"], "R1-35", "n_runs"), I0(d35["n_runs"], "R1-35", "n_runs")))
w()

# R2b A_censored
w("**The test of the likelihood inconsistency in R2b: `A_censored`.** R2b replaced the density at the "
  "clipped value by the censored log-mass for clamped draws, in the rollout and in the update "
  "(`clamp_likelihood = censored`; RR-09 section 4.2), and compared it with the R1 baseline `A_base` "
  "(the R1 `parents_A` runs, RR-04) in a full phase A from scratch (1600 updates), 10 seeds per q (10501-10510), paired by (q, seed). "
  "The criterion is the pre-registered one: part (a) is the 95% percentile bootstrap CI of the mean "
  "paired difference of |peak error| lying below 0 at both q (10,000 resamples, bootstrap seed 20261004, "
  "RR-09), part (b) is that no run passing G-A under the baseline fails it under the arm. The criterion "
  "is descriptive, not a gate. Result: part (a) not met, part (b) holds, overall not met.")
w()
cr = r2b_crit[r2b_crit.arm == "A_censored"].iloc[0]
di = r2b_di[r2b_di.arm == "A_censored"].iloc[0]
L = "R2B-02 arm=A_censored "
w("**Table 5.1d. `A_censored` minus `A_base`, paired difference of |peak error| (negative = better).**")
w()
w("| q | n pairs | mean difference | 95% CI | seeds better / zero (of 10) | part (a) |")
w("|---|---|---|---|---|---|")
for q, k in (("50", "50"), ("60", "60")):
    nb = di["q%s n_better/n_zero" % q]
    w("| %s | %s | %s | [%s, %s] | %s | %s |" % (
        q, I0(cr["n_pairs_q" + q], "R2B-02", L + "n_pairs_q" + q),
        N(cr["mean_q" + q], "R2B-02", L + "mean_q" + q),
        N(cr["ci_mean_lo_q" + q], "R2B-02", L + "ci_mean_lo_q" + q),
        N(cr["ci_mean_hi_q" + q], "R2B-02", L + "ci_mean_hi_q" + q),
        nb.replace(" of 10", ""),
        "met" if cr["a_q" + q] else "not met"))
w()
w("Part (b): %s (`b_status`), %s violations; overall: %s. Source: R2B-02 (`results/v2_refine_r2b/analysis/"
  "criterion.csv`, row `arm = A_censored`, columns `n_pairs_q*`, `mean_q*`, `ci_mean_lo_q*`, "
  "`ci_mean_hi_q*`, `a_q*`, `b_status`, `b_n_violations`, `overall`); seeds better / zero from R2B-01 "
  "(`decision_inputs.csv`, columns `q50 n_better/n_zero`, `q60 n_better/n_zero`)."
  % (cr.b_status, I0(cr.b_n_violations, "R2B-02", L + "b_n_violations"), cr.overall))
w()
w("![A_censored minus A_base, seed-level paired differences of |peak error| at q = 50 and q = 60 "
  "(negative = better); markers are the ten seed pairs, the diamond with bars is the mean and its "
  "95% CI. Figure FG-16.](figures/FG-16_waveA_A_censored_vs_A_base.png)")
w()

# first flag statistics from R2B-03
cen = r2b_run[r2b_run.arm == "A_censored"].copy()
base = r2b_run[r2b_run.arm == "A_base"].copy()
nz_c = cen[cen.d1_flagged_L_s2 > 0]
never = cen[cen.d1_flagged_L_s2 == 0]
def rngd(df, q):
    s = df[df.q == q]
    return s.d1_first_flagged_update.min(), s.d1_first_flagged_update.max()
f50, f60 = rngd(nz_c, 50), rngd(nz_c, 60)
nzc = nz_c.copy()
nzc["div"] = pd.to_numeric(nzc.rng_div_learn)
nzc["gap"] = nzc["div"] - nzc.d1_first_flagged_update
d50 = nzc[nzc.q == 50]["div"]; d60 = nzc[nzc.q == 60]["div"]
sp = r2b_spec.set_index(["arm", "q"])
LR = "R2B-03 arm=A_censored "
w("Why the arm can only differ on some runs. The censored likelihood acts only on rows whose raw draw "
  "was clamped. In %s of the 20 `A_censored` runs (q = 50 seed 10502 and q = 60 seed 10506) the learner "
  "never drew a clamped stage-2 value, so the run is identical to its baseline [R2B-03, "
  "`d1_flagged_L_s2` = 0; RR-09, \"Reading `A_censored`\"; one zero difference per q in R2B-01]. In the other %s runs the first buffer with a flagged row is at update %s to %s "
  "(q = 50) and %s to %s (q = 60) [R2B-03, `d1_first_flagged_update`], and the `learn` random stream "
  "first differs from the baseline's %s to %s updates later, at updates %s to %s (q = 50) and %s to %s "
  "(q = 60) [R2B-03, `rng_div_learn`]. The mean number of flagged learner stage-2 draws per run over "
  "the 1600 updates is %s (q = 50) and %s (q = 60) in `A_censored`, against %s and %s in `A_base` "
  "[R2B-29, `d1_flagged_L_s2`, mean over the 10 seeds, computed here]."
  % (I0(len(never), "R2B-03", LR + "count of d1_flagged_L_s2 == 0"),
     I0(len(nz_c), "R2B-03", LR + "count of d1_flagged_L_s2 > 0"),
     I0(f50[0], "R2B-03", LR + "min d1_first_flagged_update q=50 (flagged runs)"),
     I(f50[1], "R2B-03", LR + "max d1_first_flagged_update q=50"),
     I0(f60[0], "R2B-03", LR + "min d1_first_flagged_update q=60"),
     I0(f60[1], "R2B-03", LR + "max d1_first_flagged_update q=60"),
     I0(nzc.gap.min(), "R2B-03", LR + "min(rng_div_learn - d1_first_flagged_update) over flagged runs"),
     I0(nzc.gap.max(), "R2B-03", LR + "max(rng_div_learn - d1_first_flagged_update) over flagged runs"),
     I0(d50.min(), "R2B-03", LR + "min rng_div_learn q=50"),
     I(d50.max(), "R2B-03", LR + "max rng_div_learn q=50"),
     I0(d60.min(), "R2B-03", LR + "min rng_div_learn q=60"),
     I0(d60.max(), "R2B-03", LR + "max rng_div_learn q=60"),
     N(r2b_spec[(r2b_spec.arm == "A_censored") & (r2b_spec.q == 50)].d1_flagged_L_s2.mean(), "R2B-29", "computed: mean d1_flagged_L_s2, A_censored q=50"),
     N(r2b_spec[(r2b_spec.arm == "A_censored") & (r2b_spec.q == 60)].d1_flagged_L_s2.mean(), "R2B-29", "computed: mean d1_flagged_L_s2, A_censored q=60"),
     N(r2b_spec[(r2b_spec.arm == "A_base") & (r2b_spec.q == 50)].d1_flagged_L_s2.mean(), "R2B-29", "computed: mean d1_flagged_L_s2, A_base q=50"),
     N(r2b_spec[(r2b_spec.arm == "A_base") & (r2b_spec.q == 60)].d1_flagged_L_s2.mean(), "R2B-29", "computed: mean d1_flagged_L_s2, A_base q=60")))
w()
w("Decision. The PI's decisions for this publication state that the censored likelihood is not adopted "
  "(PI publication prompt, D1). The R2c housekeeping record states, as the R2b acceptance, that "
  "`clamp_likelihood` stays `density` (\"no measurable effect; the likelihood inconsistency stays a "
  "recorded known issue\", RR-12, section 1.3).")
w()

# R2b D1 counts in confirmation runs
cq = r2b_d1.copy()
lo_L = int(cq.d1_L_s2_lo_sum.sum()); hi_L = int(cq.d1_L_s2_hi_sum.sum())
lo_O = int(cq.d1_O_s2_lo_sum.sum())
inside = int(cq.d1_L_s2_in_lo_sum.sum() + cq.d1_L_s2_in_hi_sum.sum())
s1h = int(cq[["d1_L_s1_lo_sum", "d1_L_s1_hi_sum", "d1_O_s1_lo_sum", "d1_O_s1_hi_sum"]].sum().sum())
mx50 = cq[cq.q == 50].sort_values("d1_L_s2_lo_sum").iloc[-1]
mx60 = cq[cq.q == 60].sort_values("d1_L_s2_lo_sum").iloc[-1]
frac_30510 = mx50.d1_L_s2_lo_sum / mx50.d1_L_s2_n_sum
R = "R2B-27 sum over the 40 runs of "
w("**The same pattern in the 40 v2.0 confirmation runs.** The R2b seed-30510 diagnostic tabulated the D1 "
  "counts of phase A for the 40 confirmation runs (seeds 30501-30520, both q), summed over updates 1-1600. "
  "Recounted from that table (computed here): %s learner stage-2 draws below the clamp, %s above it, "
  "%s learner draws inside |d| < 2q, %s hits at stage 1; %s opponent stage-2 draws below the clamp "
  "[R2B-27]. These are the totals of the recount in the R2b diagnostic's addendum, which corrects the "
  "double-counted total in its section 5 (RR-11, addendum item 1; \"377,659 distinct clamped draws over "
  "the 40 runs\" is the sum of the learner and opponent counts, %s). The largest per-run count is %s "
  "learner draws at q = 50 (seed %s, %s of its %s learner stage-2 rows) and %s at q = 60 (seed %s) "
  "[R2B-27]. Seed 30510 at q = 50 is the failed confirmation run (CF-13)."
  % (I(lo_L, "R2B-27", R + "d1_L_s2_lo_sum"), I0(hi_L, "R2B-27", R + "d1_L_s2_hi_sum"),
     I0(inside, "R2B-27", R + "d1_L_s2_in_lo_sum + d1_L_s2_in_hi_sum"),
     I0(s1h, "R2B-27", R + "d1_*_s1_* columns"), I(lo_O, "R2B-27", R + "d1_O_s2_lo_sum"),
     I(lo_L + lo_O, "R2B-27", "computed: learner + opponent lo sums"),
     I(mx50.d1_L_s2_lo_sum, "R2B-27", "max d1_L_s2_lo_sum q=50"), I0(mx50.seed, "R2B-27", "seed of max q=50"),
     N(frac_30510, "R2B-27", "computed: d1_L_s2_lo_sum / d1_L_s2_n_sum, q=50 seed 30510"),
     I(mx50.d1_L_s2_n_sum, "R2B-27", "d1_L_s2_n_sum q=50 seed 30510"),
     I(mx60.d1_L_s2_lo_sum, "R2B-27", "max d1_L_s2_lo_sum q=60"), I0(mx60.seed, "R2B-27", "seed of max q=60")))
assert lo_L == 189170 and lo_O == 188489 and lo_L + lo_O == 377659, (lo_L, lo_O)
w()
a_sh = r2b_d1.assign(sh=r2b_d1.d1_pol_n_alpha_lt1_sum / r2b_d1.d1_pol_n_rows_sum)
oth = a_sh[(a_sh.q == 50) & (a_sh.seed != 30510)]
s30510 = a_sh[(a_sh.q == 50) & (a_sh.seed == 30510)].sh.iloc[0]
w("Rows with alpha < 1 in the same runs (computed here from R2B-27, `d1_pol_n_alpha_lt1_sum` / "
  "`d1_pol_n_rows_sum`): the share is %s for seed 30510 (q = 50) and the median over the other 19 q = 50 "
  "seeds is %s; the rows with beta < 1 number %s."
  % (N(s30510, "R2B-27", "computed: d1_pol_n_alpha_lt1_sum / d1_pol_n_rows_sum, q=50 seed 30510"),
     N(oth.sh.median(), "R2B-27", "computed: median share over q=50 seeds != 30510"),
     I0(r2b_d1.d1_pol_n_beta_lt1_sum.sum(), "R2B-27", "sum of d1_pol_n_beta_lt1_sum")))
w()
w("The R2b diagnostic records the co-occurrence of a high clamp count with the low peak of seed 30510 "
  "and labels any causal reading \"a conjecture, untested\" (RR-11, section 8, item 4: re-running with "
  "the clip level moved would test it; not run). The Spearman correlation of the clamp count with the "
  "peak error is %s among the other 19 q = 50 seeds and %s over all 40 runs, \"descriptive\" "
  "(RR-11 section 5; report only)."
  % (rec("-0.03", -0.03, "RR-11", "section 5, last paragraph (report only)"),
     rec("-0.26", -0.26, "RR-11", "section 5, last paragraph (report only)")))
w()

w("**Implications (descriptive).**")
w()
w("- The flags concern phase A only, and within phase A only the final-stage rows with |d| >= 2q "
  "(Table 5.1b). RR-11 describes these states as outside the support of the equilibrium stage-2 effort "
  "(\"|d| >= 2q, where e2* = 0\"). Phase B has no hits among the rows that enter the actor loss.")
w("- The gates do not evaluate the training likelihood. G-A, G-F, G-N and G-S are defined on the Beta mean "
  "of the actor, evaluated by the final-tier verifier or on the recovery grid (PL-01, `gates`); "
  "a clamped draw changes a gate value only through the weights that training produced.")
w("- The test in R2b changed the likelihood of the flagged rows and did not meet criterion part (a) "
  "(Table 5.1d); the issue stays recorded and `density` stays the setting (RR-12).")
w("- For T = 3: no implication is recorded in the round reports.")
w()

# --------------------------------------------------------------------------- 5.2
w("### 5.2 D2: sensitivity of the DP-BR verifier")
w()
fa = d31[(d31.family == "a") & (d31.criterion == "G-F")].iloc[0]
fb_f = d31[(d31.family == "b") & (d31.criterion == "G-F")].iloc[0]
fb_a = d31[(d31.family == "b") & (d31.criterion == "G-A")].iloc[0]
gn = d31[d31.criterion.isin(["G-N_Gmax", "G-N_eta"])]
gn_max = max(gn["q50 max value on grid"].max(), gn["q60 max value on grid"].max())
others = d31[d31.family.isin(["c", "d", "e"]) & d31.criterion.isin(["G-F", "G-A", "G-N_Gmax", "G-N_eta"])]
assert (others["q50 detection limit"] == "not reached on the grid").all()
assert (others["q60 detection limit"] == "not reached on the grid").all()
assert (d31[d31.family == "a"]["q50 detection limit"] == "not reached on the grid").all()
assert (d31[d31.family == "a"]["q60 detection limit"] == "not reached on the grid").all()
assert (d31[d31.criterion.isin(["G-N_Gmax", "G-N_eta"])]["q50 detection limit"] == "not reached on the grid").all()
assert (d31[d31.criterion.isin(["G-N_Gmax", "G-N_eta"])]["q60 detection limit"] == "not reached on the grid").all()
assert gn_max < 0.001
LA = "R1-31 family=a criterion=G-F "
LB = "R1-31 family=b criterion=%s "
w("**Outcome.** On the final-tier verifier, a stage-1 scalar error of up to %s%% does not reach G-F: the "
  "largest Gmax_full/DW on the grid is %s at q = 50 and %s at q = 60 (threshold 0.01) [R1-31]. A stage-2 "
  "amplitude error violates G-F and G-A first at %s%% (q = 50) and %s%% (q = 60) [R1-31]. Peak rounding "
  "up to kernel half-width h = %s, a tail offset up to tau = %s, and the RL-like combination of a stage-1 "
  "error and peak rounding reach none of G-F, G-A or G-N on their grids [R1-31]. The difference between "
  "development-tier and final-tier values never reaches 0.001 DW on any grid (largest value %s DW) [R1-31, "
  "rows `G-N_Gmax`, `G-N_eta`]. The study is descriptive and changes no threshold."
  % (rec("15", 15, "R1-31", LA + "q50 max |perturbation| on grid (0.15) x 100"),
     N(fa["q50 max value on grid"], "R1-31", LA + "q50 max value on grid"),
     N(fa["q60 max value on grid"], "R1-31", LA + "q60 max value on grid"),
     rec("10", 10, "R1-31", (LB % "G-F") + "q50 detection limit (0.1) x 100"),
     rec("15", 15, "R1-31", (LB % "G-F") + "q60 detection limit (0.15) x 100"),
     rec("20", 20, "R1-31", "family=c criterion=G-F q50 max |perturbation| on grid"),
     rec("2", 2, "R1-31", "family=d criterion=G-F q50 max |perturbation| on grid"),
     N(gn_max, "R1-31", "max over family, q of 'max value on grid' in rows G-N_Gmax and G-N_eta")))
assert abs(fa["q50 max value on grid"] - 0.00717) < 1e-5 and abs(fa["q60 max value on grid"] - 0.003247) < 1e-6
assert fb_f["q50 detection limit"] in (0.1, "0.1") and fb_a["q60 detection limit"] in (0.15, "0.15")
w()

w("**What was evaluated.** 38 closed-form candidates (%s evaluation rows: candidate x q x verifier tier) "
  "were built from the closed-form T = 2 equilibrium and run through the verifier (the instrument behind "
  "G-F, G-A and G-N) on the development and the final tier, at q = 50 and q = 60 [R1-12, computed here; "
  "all rows `status` = ok, all valid, no clip binds]. Family definitions (report only: "
  "`reports/v2/refine/03_d2_verifier_sensitivity.md` section 2, needs item; the families and "
  "perturbation names are the `family` and `perturbation_name` columns of R1-12):" % "152")
w()
fam_n = d12.groupby("family").size()
# record 152 and family counts
rec("152", int(len(d12)), "R1-12", "computed: number of rows")
rec("38", int(d12.groupby(["family", "cand_id"]).ngroups), "R1-12", "computed: distinct (family, cand_id)")
w("| family | perturbation | grid (maximum) | evaluation rows |")
w("|---|---|---|---|")
def gmax(f):
    return N(d31[(d31.family == f)]["q50 max |perturbation| on grid"].iloc[0], "R1-31",
             "family=%s q50 max |perturbation| on grid" % f)


fam_desc = {
    "a": ("stage-1 scalar error: e1 = (1 + delta_stage1) e1*, stage 2 exact", "delta_stage1 up to " + gmax("a")),
    "b": ("stage-2 amplitude: e2(d) = (1 + delta_stage2) e2*(d), stage 1 exact", "delta_stage2 up to " + gmax("b")),
    "c": ("peak rounding: e2* convolved with a uniform kernel of half-width h (d units), stage 1 exact",
          "h up to " + gmax("c")),
    "d": ("tail offset: e2(d) = e2*(d) + tau for \\|d\\| >= 2q, stage 1 exact",
          "tau up to " + gmax("d") + " (effort units)"),
    "e": ("RL-like: family (a) with the family (c) kernel (h = %s at both q)" %
          N(d14.e_kernel_h.iloc[0], "R1-14", "e_kernel_h (all rows equal)"),
          "delta_stage1 up to " + gmax("e")),
}
for f in "abcde":
    w("| (%s) | %s | %s | %s |" % (f, fam_desc[f][0], fam_desc[f][1],
                                   I0(fam_n[f], "R1-12", "computed: rows with family=%s" % f)))
w()
w("Source: R1-12 (`evaluations.csv`, columns `family`, `perturbation_name`; row counts computed here); "
  "descriptions and grids as printed in `03_d2_verifier_sensitivity.md` section 2 (report only). "
  "The game is the locked one: w_H = 6, w_L = 2, k = 0.00028571, DW = 4 at both q, efforts in [0, 100] "
  "(PL-01 `records[q].game`; identical in PL-08, v1.1); the verifier tiers and the gate thresholds "
  "(G-F: Gmax_full/DW <= 0.01; G-A, eta part: eta_2/DW <= 0.005; G-N: |dev - final| <= 0.001) are "
  "those of PL-01 (`records[q].verifier`, `gates`).")
w()
w("The unperturbed candidate gives the numerical floor of the verifier: Gmax_full/DW is %s at q = 50 on both "
  "tiers and at q = 60 on the development tier, and %s at q = 60 on the final tier [R1-12, family a, "
  "perturbation 0]."
  % (N(d12[(d12.family == "a") & (d12.perturbation == 0) & (d12.q == 50) & (d12.tier == "final")].Gmax_full_over_dw.iloc[0],
       "R1-12", "family=a perturbation=0 q=50 tier=final Gmax_full_over_dw"),
     N(d12[(d12.family == "a") & (d12.perturbation == 0) & (d12.q == 60) & (d12.tier == "final")].Gmax_full_over_dw.iloc[0],
       "R1-12", "family=a perturbation=0 q=60 tier=final Gmax_full_over_dw")))
w()

# Table 5.2a
w("**Table 5.2a. Detection limits of the verifier on the perturbation grids** (final tier for G-F and "
  "G-A; development-final difference for G-N; limit = smallest |perturbation| on the grid at which the "
  "metric exceeds its threshold; values in units of DW; perturbation units as in the previous table).")
w()
w("| family | perturbation | gate | q = 50: detection limit | q = 50: max \\|perturbation\\| on grid | "
  "q = 50: max value on grid | q = 60: detection limit | q = 60: max \\|perturbation\\| on grid | "
  "q = 60: max value on grid |")
w("|---|---|---|---|---|---|---|---|---|")
for _, r in d31.iterrows():
    L = "R1-31 family=%s criterion=%s " % (r.family, r.criterion)

    def lim(v, col):
        return v if v == "not reached on the grid" else N(float(v), "R1-31", L + col)
    w("| %s | `%s` | %s | %s | %s | %s | %s | %s | %s |" % (
        r.family, r.perturbation, r.criterion,
        lim(r["q50 detection limit"], "q50 detection limit"),
        N(r["q50 max |perturbation| on grid"], "R1-31", L + "q50 max |perturbation| on grid"),
        N(r["q50 max value on grid"], "R1-31", L + "q50 max value on grid"),
        lim(r["q60 detection limit"], "q60 detection limit"),
        N(r["q60 max |perturbation| on grid"], "R1-31", L + "q60 max |perturbation| on grid"),
        N(r["q60 max value on grid"], "R1-31", L + "q60 max value on grid")))
w()
w("Source: R1-31 (`results/v2_refine/analysis/decision_d2_detection_limits.csv`, all 20 rows; columns "
  "`family`, `perturbation`, `criterion`, `q50 detection limit`, `q50 max |perturbation| on grid`, "
  "`q50 max value on grid`, and the q60 columns). The same limits per tier and side are in R1-11.")
w()

# detail family b
w("**Table 5.2b. Family (b), the two detected gates** (final tier, either side).")
w()
w("| gate | q | detection limit | value at the limit | largest grid point below the limit |")
w("|---|---|---|---|---|")
for crit in ("G-F", "G-A"):
    for q in (50, 60):
        r = d11[(d11.family == "b") & (d11.q == q) & (d11.tier == "final") & (d11.criterion == crit) & (d11.side == "any")]
        assert len(r) == 1
        r = r.iloc[0]
        L = "R1-11 family=b q=%d tier=final criterion=%s side=any " % (q, crit)
        w("| %s | %d | %s | %s | %s |" % (crit, q, N(r.limit, "R1-11", L + "limit"),
                                          N(r.value_at_limit, "R1-11", L + "value_at_limit"),
                                          N(float(r.bracket_lo), "R1-11", L + "bracket_lo")))
w()
w("Source: R1-11 (`detection_limits.csv`, rows `family = b`, `tier = final`, `side = any`; columns `limit`, "
  "`value_at_limit`, `bracket_lo`). Values in units of DW; thresholds 0.01 (G-F) and 0.005 (G-A). At "
  "q = 60 the limit coincides with the largest perturbation tested (0.15), so the detection there is "
  "located only to within the grid step 0.1 to 0.15 [R1-11, `max_abs_pert_on_grid`].")
w()

# fits
def fit(fam, q, metric):
    r = d13[(d13.family == fam) & (d13.q == q) & (d13.tier == "final") & (d13.side == "pooled") & (d13.metric == metric)]
    assert len(r) == 1
    return r.iloc[0]
w("**Table 5.2c. Quadratic fit of the final-tier response through the origin** (metric = a x^2, both sides "
  "pooled, 12 points; 'x at threshold' is the fitted extrapolation sqrt(threshold / a), descriptive only).")
w()
w("| family | q | metric | a | R2 (uncentered) | x at threshold from the fit |")
w("|---|---|---|---|---|---|")
for fam, metric, nm in (("a", "Gmax_full_over_dw", "Gmax_full/DW (G-F 0.01)"),
                        ("b", "Gmax_full_over_dw", "Gmax_full/DW (G-F 0.01)"),
                        ("b", "eta_T_over_dw", "eta_2/DW (G-A 0.005)")):
    for q in (50, 60):
        r = fit(fam, q, metric)
        L = "R1-13 family=%s q=%d tier=final side=pooled metric=%s " % (fam, q, metric)
        w("| %s | %d | %s | %s | %s | %s |" % (fam, q, nm, N(r.a, "R1-13", L + "a"),
                                               N(r.r2_uncentered, "R1-13", L + "r2_uncentered"),
                                               N(r.x_at_threshold_fit, "R1-13", L + "x_at_threshold_fit")))
w()
w("Source: R1-13 (`fits.csv`, rows `tier = final`, `side = pooled`; columns `a`, `r2_uncentered`, "
  "`x_at_threshold_fit`). For family (a) the eta_2/DW fit is not a model: eta_2 does not depend on the "
  "perturbation because stage 2 is unchanged (R1-13 `fit_note`); its value is the floor %s [R1-12]."
  % N(d12[(d12.family == "a")].eta_T_over_dw.max(), "R1-12", "family=a max eta_T_over_dw"))
w()

# recovery parts of G-A (computed)
w("**Table 5.2d. Recovery parts of G-A (computed here).** The G-A row above is the eta part only (eta_2/DW <= 0.005). "
  "G-A also requires RMSE_pos/e2*(0) <= 0.05 and tail mean/e2*(0) <= 0.02, which are computed on the "
  "recovery grid and do not depend on the verifier tier (PL-01, `gates`). R1-12 holds both for every "
  "candidate; Table 5.2d gives, per family and q, the largest value on the grid and the smallest "
  "|perturbation| at which the limit is exceeded.")
w()
w("| family | q | max RMSE_pos/e2*(0) | smallest \\|p\\| with RMSE part > 0.05 | max tail mean/e2*(0) | "
  "smallest \\|p\\| with tail part > 0.02 | max \\|peak error\\| |")
w("|---|---|---|---|---|---|---|")
ev = d12.copy()
ev["ap"] = ev.perturbation.abs()
ev = ev.drop_duplicates(["family", "cand_id", "q"])  # recovery metrics are tier independent
for f in "bcde":
    pass
for f in "abcde":
    for q in (50, 60):
        s = ev[(ev.family == f) & (ev.q == q)]
        L = "R1-12 family=%s q=%d " % (f, q)
        def first(col, thr):
            x = s[s[col].abs() > thr]
            return "not reached" if x.empty else N(x.ap.min(), "R1-12", L + "min |perturbation| with %s > %g" % (col, thr))
        w("| %s | %d | %s | %s | %s | %s | %s |" % (
            f, q,
            N(s.stage2_rmse_pos_over_g2_0.abs().max(), "R1-12", L + "max stage2_rmse_pos_over_g2_0"),
            first("stage2_rmse_pos_over_g2_0", 0.05),
            N(s.stage2_tail_mean_over_g2_0.abs().max(), "R1-12", L + "max |stage2_tail_mean_over_g2_0|"),
            first("stage2_tail_mean_over_g2_0", 0.02),
            N(s.stage2_peak_rel_err_signed.abs().max(), "R1-12", L + "max |stage2_peak_rel_err_signed|")))
w()
w("Source: computed here from R1-12 (`evaluations.csv`, columns `stage2_rmse_pos_over_g2_0`, "
  "`stage2_tail_mean_over_g2_0`, `stage2_peak_rel_err_signed`, `perturbation`; one row per family, "
  "candidate and q, because these columns do not depend on the tier) and the G-A thresholds of PL-01. "
  "For family (d) the perturbation is tau in effort units, for (c) h in d units, for (a), (b), (e) the "
  "relative delta.")
w()

# figures
w("**Figures.** Detection curves (Gmax_full/DW and eta_2/DW against |perturbation|, log-log, both tiers, "
  "both q, with the thresholds drawn):")
w()
w("![D2 family (a), stage-1 scalar error: Gmax_full/DW (top) and eta_2/DW (bottom) against |delta_stage1|, "
  "q = 50 left and q = 60 right, both verifier tiers and both signs. Figure FG-07.](figures/FG-07_d2_family_a.png)")
w()
w("![D2 family (b), stage-2 amplitude: Gmax_full/DW (top) and eta_2/DW (bottom) against |delta_stage2|; "
  "both curves cross their thresholds near 10% (q = 50) and 15% (q = 60). Figure FG-08.]"
  "(figures/FG-08_d2_family_b.png)")
w()
w("![D2 family (c), peak rounding: the same quantities against the kernel half-width h. Figure FG-09.]"
  "(figures/FG-09_d2_family_c.png)")
w()
w("![D2 family (d), tail offset: the same quantities against tau. Figure FG-10.]"
  "(figures/FG-10_d2_family_d.png)")
w()
w("![D2 family (e), RL-like combination of a stage-1 error and the family (c) kernel: the same quantities "
  "against delta_stage1. Figure FG-11.](figures/FG-11_d2_family_e.png)")
w()

# family e confirmation
rr = d14.copy()
nobs = len(rr)
rr_ok = (rr.observed_gmax_final < 0.01).all()
LE = "R1-14 "
w("**Family (e) next to confirmation runs.** R1-14 compares the family-(e) prediction with the %s v1.1 "
  "confirmation runs that had the largest stage-1 errors (seeds 20501-20520; observed values are those of "
  "the runs' own, imperfect stage 2). Their stage-1 errors range in absolute value from %s to %s; the "
  "observed final-tier Gmax_full/DW ranges from %s to %s, all below 0.01 [R1-14]. The ratio observed / "
  "predicted lies between %s and %s [R1-14, `ratio_observed_over_pred_final`]; the predictions use the "
  "grid delta nearest to each run's |stage-1 error| and the family-(e) kernel."
  % (I0(nobs, "R1-14", "computed: number of rows"),
     N(rr.stage1_rel_err_signed.abs().min(), "R1-14", LE + "min |stage1_rel_err_signed|"),
     N(rr.stage1_rel_err_signed.abs().max(), "R1-14", LE + "max |stage1_rel_err_signed|"),
     N(rr.observed_gmax_final.min(), "R1-14", LE + "min observed_gmax_final"),
     N(rr.observed_gmax_final.max(), "R1-14", LE + "max observed_gmax_final"),
     N(rr.ratio_observed_over_pred_final.min(), "R1-14", LE + "min ratio_observed_over_pred_final"),
     N(rr.ratio_observed_over_pred_final.max(), "R1-14", LE + "max ratio_observed_over_pred_final")))
assert rr_ok
w()

# Observations
w("**What the tables say (observations).**")
w()
w("1. *Stage-1 scalar error and G-F.* The final-tier Gmax_full/DW responds to the stage-1 error "
  "approximately quadratically: the through-origin fit has a = %s (q = 50) and %s (q = 60) with R2 = %s "
  "and %s [R1-13, Table 5.2c], and its extrapolated crossing of 0.01 lies at %s and %s, beyond the "
  "15%% grid (descriptive only; the report states that this is an extrapolation). eta_2/DW does not respond "
  "at all in this family, because stage 2 is exact [R1-13 `fit_note`]. The G-F and G-A gates at the "
  "final tier therefore do not detect a stage-1 scalar error of 15%%. The report gives no reason beyond "
  "this fit and the unchanged stage 2."
  % (N(fit("a", 50, "Gmax_full_over_dw").a, "R1-13", "family=a q=50 final pooled Gmax a"),
     N(fit("a", 60, "Gmax_full_over_dw").a, "R1-13", "family=a q=60 final pooled Gmax a"),
     N(fit("a", 50, "Gmax_full_over_dw").r2_uncentered, "R1-13", "family=a q=50 final pooled Gmax r2"),
     N(fit("a", 60, "Gmax_full_over_dw").r2_uncentered, "R1-13", "family=a q=60 final pooled Gmax r2"),
     N(fit("a", 50, "Gmax_full_over_dw").x_at_threshold_fit, "R1-13", "family=a q=50 final pooled Gmax x_at_threshold_fit"),
     N(fit("a", 60, "Gmax_full_over_dw").x_at_threshold_fit, "R1-13", "family=a q=60 final pooled Gmax x_at_threshold_fit")))
w("2. *Stage-2 amplitude.* The response is again fitted by a x^2, with a larger coefficient (a = %s at "
  "q = 50 and %s at q = 60 for Gmax_full/DW; Table 5.2c). Both G-F and G-A are first violated at the grid "
  "points 10%% (q = 50) and 15%% (q = 60) (Table 5.2b); the fitted crossings are %s and %s for G-F and %s "
  "and %s for G-A (Table 5.2c; the fit is approximate, R2 = %s at q = 50). The stage-2 amplitude is "
  "detected by the verifier gates; peak rounding up to h = 20 (peak error up to %s at q = 50) and tail "
  "offsets up to tau = 2 are not detected by the eta part of G-A or by G-F (Table 5.2a)."
  % (N(fit("b", 50, "Gmax_full_over_dw").a, "R1-13", "family=b q=50 final pooled Gmax a"),
     N(fit("b", 60, "Gmax_full_over_dw").a, "R1-13", "family=b q=60 final pooled Gmax a"),
     N(fit("b", 50, "Gmax_full_over_dw").x_at_threshold_fit, "R1-13", "family=b q=50 final pooled Gmax x_at_threshold_fit"),
     N(fit("b", 60, "Gmax_full_over_dw").x_at_threshold_fit, "R1-13", "family=b q=60 final pooled Gmax x_at_threshold_fit"),
     N(fit("b", 50, "eta_T_over_dw").x_at_threshold_fit, "R1-13", "family=b q=50 final pooled eta x_at_threshold_fit"),
     N(fit("b", 60, "eta_T_over_dw").x_at_threshold_fit, "R1-13", "family=b q=60 final pooled eta x_at_threshold_fit"),
     N(fit("b", 50, "Gmax_full_over_dw").r2_uncentered, "R1-13", "family=b q=50 final pooled Gmax r2"),
     N(d12[(d12.family == "c") & (d12.q == 50)].stage2_peak_rel_err_signed.abs().max(), "R1-12",
       "family=c q=50 max |stage2_peak_rel_err_signed|")))
w("3. *The recovery parts of G-A do respond to some of these perturbations* (Table 5.2d, computed here): "
  "the RMSE part to the stage-2 amplitude error and the tail-mean part to the tail offset. A candidate with "
  "a signed peak error of up to 0.1 (family c, q = 50) passes every part of G-A and G-F on this grid; the "
  "peak error itself is reported, not gated (PL-01, `reported_not_gated`).")
w("4. *G-N.* The development-final difference of Gmax_full/DW and eta_2/DW stays below 0.001 DW on every grid "
  "(largest %s DW; Table 5.2a), so a G-N violation does not occur for any perturbation tested. The report "
  "states that 'not reached on the grid' says nothing beyond the largest perturbation tested "
  "(`03_d2_verifier_sensitivity.md` section 10, report only)."
  % N(gn_max, "R1-31", "max G-N value on grid"))
w()
w("**Implication for the gates (reading, not stated in the D2 report).** The D2 report describes what "
  "the verifier detects; it states no recommendation and no consequence for later stages "
  "(`03_d2_verifier_sensitivity.md`, section 10 limitations, report only). The implication carried to "
  "section 8 of this report is the one stated in the brief for this publication: for stages 1 and 2 the "
  "dynamic-deviation gates are second order in the policy error, so an accuracy metric on the induced "
  "target is needed besides them. What D2 contributes to that reading is Table 5.2c: the final-tier "
  "G-F response to a stage-1 scalar error is fitted by a x^2 with R2 = %s and %s, and a 15%% stage-1 "
  "error is not detected. D2 does not test an induced-target metric. In v2.0 the stage-1 error itself "
  "is gated (G-S, |e1_hat(0) - e1*|/e1* <= 0.05; PL-01, `gates`), which D2 predates: in family (a) the "
  "perturbation is the stage-1 error (R1-12 `stage1_rel_err_signed`), so the grid points 0.1 and 0.15 of "
  "family (a) would fail G-S (computed here from R1-12 and PL-01) while passing G-F."
  % (N(fit("a", 50, "Gmax_full_over_dw").r2_uncentered, "R1-13", "family=a q=50 final pooled Gmax r2 (again)"),
     N(fit("a", 60, "Gmax_full_over_dw").r2_uncentered, "R1-13", "family=a q=60 final pooled Gmax r2 (again)")))
w()

# definitional numbers stated in the text (thresholds, designs, seeds)
for t, v, it, lc in [
    ("20261004", 20261004, "RR-09", "section 6 Bootstrap (D3): numpy default_rng(20261004)"),
    ("10,000", 10000, "RR-09", "section 6 Bootstrap (D3): 10,000 percentile resamples"),
    ("10501-10510", "10501-10510", "R2B-03", "seeds of the arm A_censored, q=50 and q=60"),
    ("1600", 1600, "R2B-27", "n_updates per run (phase A)"),
    ("20", 20, "RR-08", "section 6: M2 'more than 2 of the 20 runs'"),
    ("0.01", 0.01, "PL-01", "gates.G-F threshold Gmax_full_over_dw"),
    ("0.005", 0.005, "PL-01", "gates.G-A eta_T_over_dw threshold"),
    ("0.05", 0.05, "PL-01", "gates.G-A stage2_rmse_pos_over_g2_0 threshold; G-S stage1_rel_err_abs threshold"),
    ("0.02", 0.02, "PL-01", "gates.G-A stage2_tail_mean_over_g2_0 threshold"),
    ("0.001", 0.001, "PL-01", "gates.G-N thresholds (dev - final)"),
    ("w_H = 6, w_L = 2, k = 0.00028571, DW = 4", "6;2;0.00028571428571428574;4", "PL-01", "records[50].game and records[60].game"),
    ("[0, 100]", "0;100", "PL-01", "records[q].game e_min, e_max"),
    ("40 confirmation runs (seeds 30501-30520, both q)", 40, "R2B-27", "number of rows"),
    ("30501-30520", "30501-30520", "R2B-27", "seed column"),
    ("20501-20520", "20501-20520", "CF-18/FG-01", "v1.1 fresh seeds"),
    ("10 seeds per q", 10, "R2B-02", "n_pairs_q50, n_pairs_q60"),
]:
    rec(t, v, it, lc)

text = "\n".join(lines) + "\n"
with open(os.path.join(OUT, "sec05.md"), "w") as fh:
    fh.write(text)
with open(os.path.join(OUT, "sec05_ledger.csv"), "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=["statement_id", "section", "text", "value", "item_id", "locator"])
    wr.writeheader()
    wr.writerows(LEDGER)
print("lines", len(lines), "ledger", len(LEDGER))
