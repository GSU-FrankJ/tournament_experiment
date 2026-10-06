"""Build SCRATCH/sec02.md and SCRATCH/sec02_ledger.csv from the evidence pack (read-only)."""
import sys

sys.path.insert(0, "/tmp/claude-1331199693/-home-fjiang4-tournament-experiment--claude-worktrees-"
                   "r2c-sampler-protocol-v2-1-c4c0f1/152f0306-5492-45a1-94d5-59493fa0d141/"
                   "scratchpad/report_parts/W-F")
from common import *  # noqa: F401,F403  (local helper module; names: rd, rj, g4, ci, Ledger, ...)

L = Ledger("02")

# ---------------------------------------------------------------- data
s1c = rd("R1-02")      # stage1_criterion.csv
s2c = rd("R1-06")      # stage2_criterion.csv
dec = rd("R1-01")      # decision_inputs.csv
r2b = rd("R2B-02")     # criterion.csv (R2b)
r2c = rd("R2C-02")     # criterion.csv (R2c)
pc = rd("CF-02")       # pass_counts.csv
pr = rd("CF-03")       # per_run.csv (v2.0 confirmation)
d1 = rd("R1-29")       # decision_d1_flags.csv
d2 = rd("R1-31")       # decision_d2_detection_limits.csv
ev = rd("R1-12")       # evaluations.csv (152 candidates)
cf4 = rd("CF-04")      # s1_summary.csv
chk = rj("CF-08")      # rehearsal_v2_0_checks.json

P = {
    "R1-02": PATH["R1-02"], "R1-06": PATH["R1-06"], "R1-01": PATH["R1-01"],
    "R2B-02": PATH["R2B-02"], "R2C-02": PATH["R2C-02"], "CF-02": PATH["CF-02"],
    "CF-03": PATH["CF-03"], "R1-29": PATH["R1-29"], "R1-31": PATH["R1-31"],
    "R1-12": PATH["R1-12"], "CF-04": PATH["CF-04"], "CF-08": PATH["CF-08"],
}


def crow(df, arm, baseline=None):
    """One row of a criterion table by arm (and baseline when an arm appears twice)."""
    m = df[df.arm == arm]
    if baseline is not None:
        m = m[m.baseline == baseline]
    assert len(m) == 1, (arm, baseline, len(m))
    return m.iloc[0]


def cstr(item, df, arm, q, baseline=None):
    """Registered 'mean [lo, hi]' string of a criterion row at q (50 or 60)."""
    r = crow(df, arm, baseline)
    t = ci(r["mean_q%d" % q], r["ci_mean_lo_q%d" % q], r["ci_mean_hi_q%d" % q])
    loc = "%s; arm=%s%s; columns mean_q%d, ci_mean_lo_q%d, ci_mean_hi_q%d" % (
        P[item], arm, "" if baseline is None else "; baseline=%s" % baseline, q, q, q)
    return L.add(t, "%s|%s|%s" % (r["mean_q%d" % q], r["ci_mean_lo_q%d" % q], r["ci_mean_hi_q%d" % q]),
                 item, loc)


def contains0(r, q):
    return r["ci_mean_lo_q%d" % q] <= 0 <= r["ci_mean_hi_q%d" % q]


def method_counts(df, arms):
    """(n arms, n with (a) met, n with CI containing 0 at both q, n with (b) holds)."""
    sub = df[df.arm.isin(arms) & (df.baseline.isin(["B_base", "A_base"]))]
    assert len(sub) == len(arms), (arms, len(sub))
    n_a = int(sub.a_met.sum())
    n_0 = int(sum(contains0(r, 50) and contains0(r, 60) for _, r in sub.iterrows()))
    n_b = int((sub.b_status == "holds").sum())
    return len(sub), n_a, n_0, n_b


def nreg(text, value, item, locator):
    return L.add(text, value, item, locator)


# ---------------------------------------------------------------- method 1..4 counts
m1 = (["B_polish1", "B_polish2"], ["A_polish1", "A_polish2"])
m2 = (["B_batch", "B_batch_mb256"], ["A_batch", "A_batch_mb256"])
m3 = (["B_kl005", "B_kl010"], ["A_kl005", "A_kl010"])
m4 = ([], ["A_anneal2", "A_anneal4"])


def counts(m):
    a = method_counts(s1c, m[0]) if m[0] else (0, 0, 0, 0)
    b = method_counts(s2c, m[1])
    return tuple(x + y for x, y in zip(a, b))


def cnt_cell(m, label, item_loc):
    n, na, n0, nb = counts(m)
    t_na = nreg("%d of %d" % (na, n), na, "R1-02/R1-06", item_loc + "; count of rows with a_met=True")
    t_n0 = nreg("%d of %d" % (n0, n), n0, "R1-02/R1-06",
                item_loc + "; count of arms with ci_mean_lo<=0<=ci_mean_hi at both q")
    t_nb = nreg("%d of %d" % (nb, n), nb, "R1-02/R1-06", item_loc + "; count of rows with b_status=holds")
    return t_na, t_n0, t_nb


LOC1 = "%s (stage-1 arms) and %s (stage-2 arms)" % (P["R1-02"], P["R1-06"])

# method 1
a1, z1, b1 = cnt_cell(m1, "polish", LOC1)
out1 = "(a) met by %s arms; the CI contains 0 at both q for %s arms; (b) holds for %s. [R1-02, R1-06]" % (a1, z1, b1)

# method 2
a2, z2, b2 = cnt_cell(m2, "batch", LOC1)
bm = "A_batch_mb256"
out2 = ("(a) met by %s arms; the CI contains 0 at both q for %s arms; (b) holds for %s.<br>"
        "`%s`: q = 50 %s, q = 60 %s. [R1-06]" % (
            a2, z2, b2, bm, cstr("R1-06", s2c, bm, 50, "A_base"), cstr("R1-06", s2c, bm, 60, "A_base")))

# method 3
a3, z3, b3 = cnt_cell(m3, "kl", LOC1)


def wall(arm):
    r = dec[(dec.arm == arm)].iloc[0]
    return nreg(g4(r["cost_phase_wall_ratio_vs_base"]), r["cost_phase_wall_ratio_vs_base"], "R1-01",
                "%s; arm=%s; column cost_phase_wall_ratio_vs_base" % (P["R1-01"], arm))


out3 = ("(a) met by %s arms; the CI contains 0 at both q for %s arms; (b) holds for %s.<br>"
        "Phase wall time relative to the baseline: %s (`B_kl005`), %s (`B_kl010`), %s (`A_kl005`), "
        "%s (`A_kl010`). [R1-02, R1-06, R1-01]" % (
            a3, z3, b3, wall("B_kl005"), wall("B_kl010"), wall("A_kl005"), wall("A_kl010")))

# method 4
a4, z4, b4 = cnt_cell(m4, "anneal", LOC1)
out4 = ("(a) met by %s arms; the CI contains 0 at both q for %s arms; (b) holds for %s.<br>"
        "`A_anneal4` at q = 60 is above 0 (larger peak error than the baseline): %s. [R1-06]" % (
            a4, z4, b4, cstr("R1-06", s2c, "A_anneal4", 60, "A_base")))

# method 5
r5 = crow(s2c, "A_detmean", "A_ctrl200")
assert not r5.a_met and r5.b_status == "holds"
for arm, base in [("P20_lr3e-5", "A_ctrl200"), ("P20_lr3e-4", "A_ctrl200_lr3e-4")]:
    rr = crow(r2b, arm, base)
    assert (not rr.a_met) and rr.b_status == "holds"
rr4 = crow(r2b, "P20_lr3e-4", "A_ctrl200_lr3e-4")
assert rr4["ci_mean_lo_q50"] > 0
out5 = ("(a) not met and (b) holds in all three comparisons with the matched PPO control "
        "(difference arm - control):<br>"
        "R1 `A_detmean`: q = 50 %s, q = 60 %s [R1-06]<br>"
        "R2b `P20_lr3e-5`: q = 50 %s, q = 60 %s [R2B-02]<br>"
        "R2b `P20_lr3e-4`: q = 50 %s (above 0, worse), q = 60 %s [R2B-02]" % (
            cstr("R1-06", s2c, "A_detmean", 50, "A_ctrl200"), cstr("R1-06", s2c, "A_detmean", 60, "A_ctrl200"),
            cstr("R2B-02", r2b, "P20_lr3e-5", 50, "A_ctrl200"), cstr("R2B-02", r2b, "P20_lr3e-5", 60, "A_ctrl200"),
            cstr("R2B-02", r2b, "P20_lr3e-4", 50, "A_ctrl200_lr3e-4"),
            cstr("R2B-02", r2b, "P20_lr3e-4", 60, "A_ctrl200_lr3e-4")))

# method 6
r6 = crow(s1c, "B_expcont")
assert r6.a_met and r6.b_status == "holds"
d6 = dec[dec.arm == "B_expcont"].iloc[0]
rat50 = nreg(g4(d6["disp_ratio_q50"]), d6["disp_ratio_q50"], "R1-01",
             "%s; arm=B_expcont; column disp_ratio_q50" % P["R1-01"])
rat60 = nreg(g4(d6["disp_ratio_q60"]), d6["disp_ratio_q60"], "R1-01",
             "%s; arm=B_expcont; column disp_ratio_q60" % P["R1-01"])
pc50 = pc[pc.q == 50].iloc[0]
pc60 = pc[pc.q == 60].iloc[0]
assert pc50.n_pass == 19 and pc50.n_expected == 20 and pc60.n_pass == 20 and pc60.n_expected == 20
c50 = nreg("%d / %d" % (pc50.n_pass, pc50.n_expected), "19/20", "CF-02",
           "%s; q=50; columns n_pass, n_expected" % P["CF-02"])
c60 = nreg("%d / %d" % (pc60.n_pass, pc60.n_expected), "20/20", "CF-02",
           "%s; q=60; columns n_pass, n_expected" % P["CF-02"])
f = pr[(pr.q == 50) & (pr.seed == 30510)].iloc[0]
assert f.outcome == "stage2_failure" and (not f["G-A_eta_pass"])
eta = nreg(g4(f.eta_final), f.eta_final, "CF-03", "%s; q=50 seed=30510; column eta_final" % P["CF-03"])
out6 = ("R1: (a) met and (b) holds. S1 error, difference to `B_base`: q = 50 %s, q = 60 %s [R1-02]; "
        "across-seed SD of the signed error, ratio to the baseline: %s (q = 50), %s (q = 60) [R1-01].<br>"
        "v2.0 confirmation (fresh seeds): %s runs pass at q = 50 and %s at q = 60, rule >= 18 of 20 met "
        "at both q [CF-01, CF-02]; the one failed run is q = 50 seed 30510, outcome `stage2_failure`, "
        "eta_2/DW %s against the G-A limit 0.005 [CF-03]." % (
            cstr("R1-02", s1c, "B_expcont", 50), cstr("R1-02", s1c, "B_expcont", 60),
            rat50, rat60, c50, c60, eta))
assert rj("CF-01")["overall"] == "PASS"

# D1
def d1row(q, ph):
    m = d1[(d1.group == "v11_reproduction") & (d1.q.astype(str) == str(q)) & (d1.phase == ph)]
    assert len(m) == 1
    return m.iloc[0]


dA = d1row("all", "A")
dB = d1row("all", "B")
dA50 = d1row(50, "A")
dA60 = d1row(60, "A")
assert dA.M1_outcome == "exceeded" and dA.M2_outcome == "exceeded"
assert dB.M1_outcome == "not exceeded" and dB.M2_outcome == "not exceeded"
m1v = nreg(g4(dA.M1_median_over_runs), dA.M1_median_over_runs, "R1-29",
           "%s; group=v11_reproduction q=all phase=A; column M1_median_over_runs" % P["R1-29"])
m1th = nreg(g4(dA.M1_threshold), dA.M1_threshold, "R1-29", "%s; same row; column M1_threshold" % P["R1-29"])
m2n = nreg("%d of %d" % (dA.M2_runs_share_above_threshold, dA.n_runs), "3 of 20", "R1-29",
           "%s; same row; columns M2_runs_share_above_threshold, n_runs" % P["R1-29"])
m2mx = nreg(g4(dA.M2_max_share), dA.M2_max_share, "R1-29", "%s; same row; column M2_max_share" % P["R1-29"])
m2q50 = nreg("%d of %d" % (dA50.M2_runs_share_above_threshold, dA50.n_runs), "2 of 10", "R1-29",
             "%s; q=50 phase=A; columns M2_runs_share_above_threshold, n_runs" % P["R1-29"])
m2q60 = nreg("%d of %d" % (dA60.M2_runs_share_above_threshold, dA60.n_runs), "1 of 10", "R1-29",
             "%s; q=60 phase=A; columns M2_runs_share_above_threshold, n_runs" % P["R1-29"])
out_d1 = ("Locked pipeline (20 C-R1 runs), phase A, both q pooled: M1 exceeded (median clamp fraction %s "
          "against the threshold %s); M2 exceeded (%s runs with a clamped-row gradient share above 1%%, "
          "maximum %s; per q: %s and %s, not exceeded). Phase B: neither flag exceeded. [R1-29]<br>"
          "R2b `A_censored`: q = 50 %s, q = 60 %s; (a) not met, (b) holds. [R2B-02]" % (
              m1v, m1th, m2n, m2mx, m2q50, m2q60,
              cstr("R2B-02", r2b, "A_censored", 50, "A_base"), cstr("R2B-02", r2b, "A_censored", 60, "A_base")))
rc = crow(r2b, "A_censored", "A_base")
assert (not rc.a_met) and rc.b_status == "holds"

# D2
def d2cell(fam, crit, q, col):
    m = d2[(d2.family == fam) & (d2.criterion == crit)]
    assert len(m) == 1
    return m.iloc[0]["q%d %s" % (q, col)]


n152 = nreg("%d" % len(ev), len(ev), "R1-12", "%s; number of rows" % P["R1-12"])
assert len(ev) == 152
a_gf50 = d2cell("a", "G-F", 50, "max value on grid")
a_gf60 = d2cell("a", "G-F", 60, "max value on grid")
assert d2cell("a", "G-F", 50, "detection limit") == "not reached on the grid"
assert d2cell("b", "G-F", 50, "detection limit") == 0.1 or str(d2cell("b", "G-F", 50, "detection limit")) == "0.1"
b50 = nreg(g4(float(d2cell("b", "G-F", 50, "detection limit"))), 0.1, "R1-31",
           "%s; family=b criterion=G-F; column q50 detection limit" % P["R1-31"])
b60 = nreg(g4(float(d2cell("b", "G-F", 60, "detection limit"))), 0.15, "R1-31",
           "%s; family=b criterion=G-F; column q60 detection limit" % P["R1-31"])
assert str(d2cell("b", "G-A", 50, "detection limit")) == str(d2cell("b", "G-F", 50, "detection limit"))
assert str(d2cell("b", "G-A", 60, "detection limit")) == str(d2cell("b", "G-F", 60, "detection limit"))
a_max = d2[(d2.family == "a") & (d2.criterion == "G-F")].iloc[0]["q50 max |perturbation| on grid"]
a_pert = nreg(g4(a_max), a_max, "R1-31", "%s; family=a criterion=G-F; column q50 max |perturbation| on grid" % P["R1-31"])
a_v50 = nreg(g4(a_gf50), a_gf50, "R1-31", "%s; family=a criterion=G-F; column q50 max value on grid" % P["R1-31"])
a_v60 = nreg(g4(a_gf60), a_gf60, "R1-31", "%s; family=a criterion=G-F; column q60 max value on grid" % P["R1-31"])
for fam in "cde":
    sub = d2[(d2.family == fam) & (d2.criterion.isin(["G-F", "G-A", "G-N_Gmax", "G-N_eta"]))]
    assert (sub["q50 detection limit"] == "not reached on the grid").all()
    assert (sub["q60 detection limit"] == "not reached on the grid").all()
c_max = d2[(d2.family == "c") & (d2.criterion == "G-F")].iloc[0]["q50 max |perturbation| on grid"]
d_max = d2[(d2.family == "d") & (d2.criterion == "G-F")].iloc[0]["q50 max |perturbation| on grid"]
c_h = nreg(g4(c_max), c_max, "R1-31", "%s; family=c; column q50 max |perturbation| on grid" % P["R1-31"])
d_t = nreg(g4(d_max), d_max, "R1-31", "%s; family=d; column q50 max |perturbation| on grid" % P["R1-31"])
out_d2 = ("%s perturbed closed-form candidates [R1-12]. Family a (stage-1 scalar error, up to %s): G-F not "
          "reached on the grid, largest Gmax_full/DW %s (q = 50) and %s (q = 60). Family b (stage-2 amplitude): "
          "G-F and G-A first violated at %s (q = 50) and %s (q = 60). Families c (peak rounding, kernel h up to "
          "%s), d (tail offset, tau up to %s) and e (RL-like combination): no gate reached on the grid. [R1-31]" % (
              n152, a_pert, a_v50, a_v60, b50, b60, c_h, d_t))

# ---------------------------------------------------------------- Table 1
HEAD1 = "| # | item | mechanism as implemented | arms and rounds | pre-registered criterion | outcome | decision |"
SEP1 = "|---|---|---|---|---|---|---|"

CRIT = "(a) and (b), n = 10 seeds per q, criterion and primary metric as defined below the tables"
rows1 = [
    ("1", "polish",
     "Extra learning-rate decay windows carved out of the existing update budgets (final LR 3e-6 instead of 3e-5); "
     "`polish1` has two windows, `polish2` one [RR-08 section 3]",
     "R1: `B_polish1`, `B_polish2` (stage 1, against `B_base`); `A_polish1`, `A_polish2` (stage 2, against `A_base`); 20 runs per arm",
     CRIT, out1,
     "Not adopted: no R1 arm other than method 6 enters the protocol [RR-01 Addendum, item 2]"),
    ("2", "batch",
     "Larger rollout batch: 2048 episodes per update; minibatch 1024 (same number of optimiser steps) or 256 (more steps) "
     "[RR-08 section 3]",
     "R1: `B_batch`, `B_batch_mb256` (stage 1); `A_batch`, `A_batch_mb256` (stage 2); 20 runs per arm",
     CRIT, out2,
     "Not adopted: no R1 arm other than method 6 enters the protocol [RR-01 Addendum, item 2]"),
    ("3", "target_kl",
     "KL early stopping: no further PPO epoch in an update once the whole-buffer KL exceeds the target (0.005 or 0.01) "
     "[RR-08 section 4, item 3]",
     "R1: `B_kl005`, `B_kl010` (stage 1); `A_kl005`, `A_kl010` (stage 2); 20 runs per arm",
     CRIT, out3,
     "Not adopted: \"it changes cost, not accuracy, and the protocol changes for accuracy only\" [RR-01 Addendum, item 2]"),
    ("4", "annealing",
     "Stage-2 concentration annealing: the Beta concentration of the actor is multiplied by a scale that rises linearly "
     "from 1.0 to 2.0 or 4.0 over the 400 updates [RR-08 section 4, item 4]",
     "R1: `A_anneal2`, `A_anneal4` (stage 2, against `A_base`); 20 runs per arm",
     CRIT, out4,
     "Not adopted: no R1 arm other than method 6 enters the protocol [RR-01 Addendum, item 2]"),
    ("5", "deterministic mean (pathwise fine-tuning)",
     "Terminal fine-tuning by exact-gradient ascent of the conditional expected terminal payoff at the learner's Beta mean "
     "(model-based, ablation only); R1 one step per update for 200 updates, R2b 20 steps per update [RR-08 section 4, item 5; RR-09 section 4.3]",
     "R1: `A_detmean` against its PPO control `A_ctrl200`. R2b (wave P): `P20_lr3e-5` against `A_ctrl200`, "
     "`P20_lr3e-4` against `A_ctrl200_lr3e-4`; 20 runs per arm",
     "(a) and (b) of the pathwise arm against the matched PPO control",
     out5,
     "Closed as negative at matched budgets (PI publication prompt, D1). R2b D2: model-based, ablation and diagnostic only [RR-05]"),
    ("6", "expected continuation",
     "In the stage-1 return, the sampled continuation is replaced by a shock-integrated table value of the frozen stage-2 "
     "policy, built once at Phase-B entry; stage-2 rows unchanged [RR-08 section 4, item 6]",
     "R1: `B_expcont` (stage 1, against `B_base`), 20 runs. v2.0: protocol v2.0 (method 6 plus gate G-S), "
     "re-rehearsal on the development seeds and confirmation on fresh seeds 30501-30520, 40 runs",
     "R1: (a) and (b). v2.0: at least 18 of 20 runs pass (G-A, G-F, G-N, G-S) at each q [RR-03 section 4.1]",
     out6,
     "Enters protocol v2.0 [RR-01 Addendum, item 2; PL-01 change_log]; the locked T=2 solver is v2.0 (PI publication prompt, D1)"),
    ("D1", "clamp (clipped-Beta likelihood)",
     "Counts of raw Beta draws below `action_clamp` or above 1 - `action_clamp` before the clip, with flags M1 (median "
     "clamp fraction above 1e-3) and M2 (clamped-row gradient share above 1% in more than 2 of 20 runs); R2b arm "
     "`A_censored` replaces the density of a flagged row by the censored log-mass [RR-08 section 6; RR-09 section 4.2]",
     "R1: diagnostic on the 20 locked-pipeline runs and the pilot arms. R2b (wave A): `A_censored` against `A_base`, 20 runs",
     "M1 and M2 are report-only flags; `A_censored`: (a) and (b)",
     out_d1,
     "Report-only; no fix applied [RR-01, headline D1; PL-01 known_issues]. `clamp_likelihood` stays `density`; the censored likelihood is not adopted "
     "(PI publication prompt, D1; RR-12 section 1.3)"),
    ("D2", "verifier sensitivity",
     "Closed-form candidates perturbed in five families (a-e) and evaluated on both verifier tiers; detection limit = smallest "
     "perturbation at which G-F, G-A or G-N is violated [RR-08 section 6]",
     "R1 only: pure evaluation, no training runs",
     "Descriptive; detection limit or \"not reached on the grid\"; changes no threshold [RR-08 section 6]",
     out_d2,
     "Report-only (RR-08 section 6). The five v2.0 changes (RR-03 section 1.3) do not refer to this diagnostic; no decision on D2 is recorded"),
]

t1 = [HEAD1, SEP1] + ["| " + " | ".join(r) + " |" for r in rows1]

# ---------------------------------------------------------------- Table 2 (rounds)
n1 = nreg("%d" % len(rd("R1-05")), len(rd("R1-05")), "R1-05", "%s; number of rows" % PATH["R1-05"])
s2pr = rd("R1-37")
n2 = nreg("%d" % int((s2pr.status == "done").sum()), int((s2pr.status == "done").sum()), "R1-37",
          "%s; rows with status=done (the 20 parent_u1600 rows are references)" % PATH["R1-37"])
r2bn = rj("R2B-07")
n_r2b = nreg("%d" % r2bn["n_ok"], r2bn["n_ok"], "R2B-07", "%s; key n_ok (of n=%d)" % (PATH["R2B-07"], r2bn["n"]))
r2cn = rj("R2C-06")["summary"]
n_r2c = nreg("%d" % r2cn["n_runs_ok"], r2cn["n_runs_ok"], "R2C-06",
             "%s; summary.n_runs_ok (of n_runs=%d)" % (PATH["R2C-06"], r2cn["n_runs"]))
assert rj("R2C-03")["outcome"] == "none_eligible" and rj("R2C-03")["selected"] is None
p50 = crow(r2b, "A_peak50", "A_base")
assert p50.a_met and p50.b_status == "violated" and p50.b_violations == "q60/10503 q60/10504 q60/10510"
assert all((not r.a_met) and r.b_status == "holds" for _, r in r2c.iterrows())
assert all(contains0(r, 60) for _, r in r2c.iterrows())  # every R2c q=60 CI contains 0
assert chk["R7"]["pass"] is False and all(chk[k]["pass"] for k in ["R1", "R2", "R3", "R4", "R5", "R6"])

HEAD2 = "| round | question | branch | outcome |"
SEP2 = "|---|---|---|---|"
rows2 = [
    ("R1", "Does one of the six candidate changes to the locked v1.1 pipeline improve accuracy, tested one factor at a time? "
           "Diagnostics D1 and D2.",
     "`v2-t2-refine`, code commit `32a8c21` [RR-01]",
     "%s stage-1 runs [R1-05] and %s stage-2 runs [R1-37]; only method 6 meets (a) and (b); accepted by the PI "
     "[RR-01 Addendum, item 1]" % (n1, n2)),
    ("v2.0", "Protocol v2.0 (method 6 plus gate G-S): does it pass the confirmation rule on fresh seeds?",
     "`v2-t2-refine`; lock commit `1d6d4d0`, confirmation launch commit `d2e377d` [RR-03 section 1.5]",
     "Re-rehearsal checks R1-R6 pass, R7 false as the checks tool computed it and accepted as met under D-R7 [CF-08]; "
     "confirmation PASS, %s (q = 50) and %s (q = 60) [CF-02]" % (c50, c60)),
    ("R2b", "Does a stage-2 peak mechanism (peak-focused starts, censored likelihood, pathwise fine-tuning at 20 steps per update) "
            "reduce the peak error? Read-only diagnostic of q = 50 seed 30510.",
     "`v2-t2-r2b`, code commit `1ff99bd`, base `b55d389` [RR-04, RR-09]",
     "%s pilot runs [R2B-07]; no arm meets both parts; `A_peak50` meets (a) at both q and violates (b) at q = 60 "
     "(seeds 10503, 10504, 10510, G-A tail-mean limit) [R2B-02]" % n_r2b),
    ("R2c", "Does another share or timing of peak-focused starts meet the criterion under a pre-registered selection rule "
            "(the path to a v2.1)?",
     "`v2-t2-r2c`, code commit `58c26716`, base `62ecc436` [RR-06, RR-10]",
     "%s runs [R2C-06]; no arm selected: (a) not met at q = 60 by any arm, (b) holds for all four; no v2.1 [R2C-02, R2C-03]" % n_r2c),
]
t2 = [HEAD2, SEP2] + ["| " + " | ".join(r) + " |" for r in rows2]

# ---------------------------------------------------------------- mapping line
boot = cf4.bootstrap_seed.unique()
assert list(boot) == [20261002] and list(cf4.bootstrap_resamples.unique()) == [10000]
assert list(r2c.boot_seed.unique()) == [20261005]
mseed1 = nreg("20261003", 20261003, "RR-02", "reports/v2/refine/06_decision_inputs.md, header: default_rng(20261003) (report only)")
mseed2 = nreg("20261004", 20261004, "RR-05", "reports/v2/refine_r2b/05_decision_inputs.md, header: default_rng(20261004) (report only)")
mseed3 = nreg("20261005", 20261005, "R2C-02", "%s; column boot_seed" % PATH["R2C-02"])
mseed0 = nreg("20261002", 20261002, "CF-04", "%s; column bootstrap_seed" % PATH["CF-04"])
nboot = nreg("10,000", 10000, "R2C-02", "%s; column n_boot" % PATH["R2C-02"])

mapping = (
    "Criteria and statistics by round. R1 (bootstrap seed %s): (b) is about G-F with its G-N part for stage-1 arms and G-A "
    "with its G-N part for stage-2 arms; comparators `B_base`, `A_base`, and `A_ctrl200` for the method-5 ablation [RR-02, RR-08 section 5]. "
    "R2b (seed %s): primary metric is the absolute signed peak error, (b) is about G-A with its G-N part; comparators "
    "`A_base` (= R1 `parents_A`) for wave A, the PPO control of the same learning rate for wave P [RR-05, RR-09 section 6]. "
    "R2c (seed %s): comparator `parents_A`, with the selection rule applied mechanically [R2C-02, R2C-03, RR-07]. "
    "All three use %s percentile resamples of the 10 paired seeds, one fresh generator per (q, statistic). "
    "Confirmation statistics (seed %s): exact Clopper-Pearson intervals for the pass counts and a percentile bootstrap "
    "of the mean stage-1 error [CF-02, CF-04, PL-02 section 3]." % (mseed1, mseed2, mseed3, nboot, mseed0))

crit_text = (
    "**The criterion.** For every arm the difference arm - baseline is formed per (q, seed) on the development seeds "
    "10501-10510 (n = 10 seeds per q). The primary metric is the S1 error |e1_hat(0) - e1*| / e1* (`stage1_rel_err_abs`) for stage-1 "
    "arms and the absolute signed peak error |e2_hat(0) - e2*(0)| / e2*(0) (`stage2_peak_rel_err_abs`) for stage-2 arms; "
    "an improvement is a negative difference. Part (a): the 95% percentile bootstrap CI of the mean difference lies below 0 "
    "at both q = 50 and q = 60. Part (b): no run that passed its gate under the baseline fails it under the arm. "
    "The criterion is descriptive: it is not a gate and it does not select by itself; a CI that contains 0 with 10 seeds "
    "does not show that an arm has no effect [RR-02 header; RR-05 header; RR-09 section 6].")

lead = (
    "Each of the note's six candidate methods and two diagnostics was run as a one-factor pilot on development seeds with a "
    "pre-registered, descriptive criterion. Only method 6 met it, and it is the only candidate that entered the locked protocol, "
    "v2.0 [RR-01 Addendum, item 2]. Numbers follow the numbering of the R1 report (methods 1-6, diagnostics D1 and D2) [RR-02 section 1]. "
    "Decisions are quoted from the round reports or from the PI publication prompt, D1 (the PI's binding decisions for this report).")

# ---------------------------------------------------------------- peak-focused start evidence (PI decision D1)
tail50 = rd("R2B-04")
tail_c = rd("R2C-04")


def tailmax(df, arm, q):
    m = df[(df.arm == arm) & (df.q == q)]
    assert len(m) == 1
    return m.iloc[0]["max_tail_mean_over_g2_0"]


def met(df, arm, q):
    r = crow(df, arm)
    return "met" if r["a_q%d" % q] else "not met"


def tailstr(item, df, arm, q, tailname):
    v = tailmax(df, arm, q)
    return nreg(g4(v), v, item, "%s; arm=%s q=%d; column max_tail_mean_over_g2_0" % (PATH[item], tailname, q))


def bcell(df, arm):
    r = crow(df, arm)
    if r.b_status == "holds":
        return "holds"
    return "violated (q = 60, seeds 10503, 10504, 10510)"


peak_rows = []
spec = [
    ("`A_peak25` (R2b)", "0.25 from update 1", r2b, "A_peak25", "R2B-04", tail50, "A_peak25"),
    ("`A_peak50` (R2b)", "0.50 from update 1", r2b, "A_peak50", "R2B-04", tail50, "A_peak50"),
    ("`A_peak35` (R2c)", "0.35 from update 1", r2c, "A_peak35", "R2C-04", tail_c, "A_peak35"),
    ("`A_peak40` (R2c)", "0.40 from update 1", r2c, "A_peak40", "R2C-04", tail_c, "A_peak40"),
    ("`A_peak50_late400` (R2c)", "0.50 from update 1201", r2c, "A_peak50_late400", "R2C-04", tail_c, "A_peak50_late400"),
    ("`A_peak50_late800` (R2c)", "0.50 from update 801", r2c, "A_peak50_late800", "R2C-04", tail_c, "A_peak50_late800"),
]
for label, sh, cdf, arm, titem, tdf, tarm in spec:
    cit = "R2B-02" if cdf is r2b else "R2C-02"
    peak_rows.append("| %s | %s | %s | %s | %s | %s / %s |" % (
        label, sh, met(cdf, arm, 50) + " [" + cit + "]", met(cdf, arm, 60) + " [" + cit + "]",
        bcell(cdf, arm), tailstr(titem, tdf, tarm, 50, arm), tailstr(titem, tdf, tarm, 60, arm)))
base_t50 = tailstr("R2B-04", tail50, "A_base", 50, "A_base")
base_t60 = tailstr("R2B-04", tail50, "A_base", 60, "A_base")
peak_rows.append("| baseline `parents_A` | | | | | %s / %s |" % (base_t50, base_t60))
t_peak = ["| arm | share, from update | (a) at q = 50 | (a) at q = 60 | (b) | largest tail mean / e2*(0), q = 50 / q = 60 |",
          "|---|---|---|---|---|---|"] + peak_rows
# consistency with the claims in the text below
assert met(r2b, "A_peak50", 50) == "met" and met(r2b, "A_peak50", 60) == "met"
assert [met(r2c, a, 50) for a in ["A_peak35", "A_peak40", "A_peak50_late400", "A_peak50_late800"]] == \
    ["met", "met", "not met", "met"]
assert all(met(r2c, a, 60) == "not met" for a in ["A_peak35", "A_peak40", "A_peak50_late400", "A_peak50_late800"])
assert float(tailmax(tail50, "A_peak50", 60)) > 0.02 and float(tailmax(tail_c, "A_peak35", 60)) < 0.02

limit = nreg("0.02", 0.02, "PL-01", "protocols/v2_T2_locked_v2_0.json gates.G-A.all_must_hold[stage2_tail_mean_over_g2_0].threshold")
peak_text = (
    "**Evidence behind the decision on the peak-focused start distribution.** The PI decided (PI publication prompt, D1) that the "
    "peak-focused start distribution is not adopted at T=2 and is carried as a design input for the T=3 terminal stage. "
    "The pilots show the following, by arm (comparator `parents_A`; bootstrap seed 20261004 for the R2b rows and 20261005 for the R2c rows). "
    "R2b `A_peak50` meets part (a) at both q and violates part (b) at q = 60, where three runs exceed the G-A tail-mean limit %s. "
    "In R2c three arms (`A_peak35`, `A_peak40`, `A_peak50_late800`) meet part (a) at q = 50 and no R2c arm meets it at q = 60; part (b) "
    "holds for all four R2c arms. No arm of the two rounds meets both parts at both q. The largest tail mean of an R2c arm stays below "
    "the limit (table); R2b `A_peak50` exceeds it at q = 60." % limit)

# ---------------------------------------------------------------- settings quoted in the text (report only)
SET = [
    ("3e-6", 3e-6, "RR-08", "reports/v2/refine/01_preregistration.md section 3.1/3.2: final LR of the polish arms"),
    ("3e-5", 3e-5, "RR-08", "reports/v2/refine/01_preregistration.md section 3.1/3.2: LR at the end of the baseline window"),
    ("2048", 2048, "RR-08", "reports/v2/refine/01_preregistration.md section 3.1: episodes_per_update"),
    ("1024", 1024, "RR-08", "reports/v2/refine/01_preregistration.md section 3.1: minibatch of B_batch"),
    ("256", 256, "RR-08", "reports/v2/refine/01_preregistration.md section 3.1: minibatch of B_batch_mb256"),
    ("0.005", 0.005, "RR-08", "reports/v2/refine/01_preregistration.md section 3.1: target_kl of B_kl005 (also the G-A eta_2/DW limit, PL-01 gates)"),
    ("0.01", 0.01, "RR-08", "reports/v2/refine/01_preregistration.md section 3.1: target_kl of B_kl010"),
    ("1.0 to 2.0 or 4.0 over the 400 updates", "1.0->2.0|4.0, 400", "RR-08", "reports/v2/refine/01_preregistration.md section 3.2: A_anneal2, A_anneal4 conc_anneal"),
    ("200 updates", 200, "RR-08", "reports/v2/refine/01_preregistration.md section 3.2: cap of A_ctrl200 and A_detmean"),
    ("20 steps per update", 20, "RR-09", "reports/v2/refine_r2b/01_preregistration.md section 4.3: pathwise_epochs 10 x 512/256 minibatches"),
    ("1e-3", 1e-3, "RR-08", "reports/v2/refine/01_preregistration.md section 6: M1 threshold (also R1-29 column M1_threshold)"),
    ("1%", 0.01, "RR-08", "reports/v2/refine/01_preregistration.md section 6: M2 gradient-share threshold"),
    ("more than 2 of 20 runs", 2, "RR-08", "reports/v2/refine/01_preregistration.md section 6 and Addendum 1 item 2"),
    ("at least 18 of 20", 18, "PL-01", "protocols/v2_T2_locked_v2_0.json confirmation rule; RR-03 section 4.1 (rule >= 18 of 20)"),
    ("30501-30520", "30501-30520", "PL-01", "protocols/v2_T2_locked_v2_0.json confirmation.seed_block"),
    ("10501-10510", "10501-10510", "PL-01", "protocols/v2_T2_locked_v2_0.json development_seeds"),
    ("n = 10 seeds per q", 10, "R1-02", "results/v2_refine/analysis/stage1_criterion.csv column n_pairs_q50 (=10 in all rows)"),
    ("20 runs per arm", 20, "R1-05", "results/v2_refine/analysis/stage1_per_run.csv: 20 rows per arm (10 seeds x 2 q)"),
    ("40 runs", 40, "CF-03", "results/v2_T2_locked/confirmation_v2_0_analysis/per_run.csv: 40 rows"),
    ("10,000 percentile resamples", 10000, "R2C-02", "results/v2_refine_r2c/analysis/criterion.csv column n_boot; RR-02/RR-05 headers for R1/R2b"),
    ("0.25 from update 1", "0.25,1", "RR-09", "reports/v2/refine_r2b/01_preregistration.md section 3.1: A_peak25 peak_share 0.25"),
    ("0.35 from update 1", "0.35,1", "RR-10", "reports/v2/refine_r2c/01_preregistration.md section 2: A_peak35"),
    ("0.40 from update 1", "0.40,1", "RR-10", "reports/v2/refine_r2c/01_preregistration.md section 2: A_peak40"),
    ("0.50 from update 1201", "0.50,1201", "RR-10", "reports/v2/refine_r2c/01_preregistration.md section 2: A_peak50_late400"),
    ("0.50 from update 801", "0.50,801", "RR-10", "reports/v2/refine_r2c/01_preregistration.md section 2: A_peak50_late800"),
    ("0.50 from update 1", "0.50,1", "RR-09", "reports/v2/refine_r2b/01_preregistration.md section 3.1: A_peak50 peak_share 0.50"),
    ("seeds 10503, 10504, 10510", "q60/10503 q60/10504 q60/10510", "R2B-02", "results/v2_refine_r2b/analysis/criterion.csv row A_peak50 column b_violations"),
    ("q = 50 seed 30510", "30510", "CF-03", "results/v2_T2_locked/confirmation_v2_0_analysis/per_run.csv row q=50 seed=30510 (outcome stage2_failure)"),
    ("32a8c21", "32a8c21", "RR-01", "reports/v2/refine/summary.md header: code commit 32a8c21"),
    ("1d6d4d0", "1d6d4d0", "RR-03", "reports/v2/protocol_v2_0_confirmation.md section 1.5: v2.0 lock commit"),
    ("d2e377d", "d2e377d", "RR-03", "reports/v2/protocol_v2_0_confirmation.md section 1.5: confirmation launch commit"),
    ("1ff99bd", "1ff99bd", "RR-04", "reports/v2/refine_r2b/summary.md: code commit 1ff99bd"),
    ("b55d389", "b55d389", "RR-09", "reports/v2/refine_r2b/01_preregistration.md section 1: branched from b55d389"),
    ("58c26716", "58c26716", "RR-10", "reports/v2/refine_r2c/01_preregistration.md Addendum 1 item 1: code commit 58c26716"),
    ("62ecc436", "62ecc436", "RR-10", "reports/v2/refine_r2c/01_preregistration.md section 1: branched from 62ecc436"),
]


def must(item, sub):
    with open(EV + PATH[item], errors="replace") as fh:
        assert sub in fh.read(), (item, sub)


must("RR-08", "target_kl")
must("RR-08", "3e-6")
must("RR-09", "20 steps")
assert len(rd("CF-03")) == 40
assert (s1c.n_pairs_q50 == 10).all() and (s2c.n_pairs_q50 == 10).all()
for t, v, it, loc in SET:
    L.add(t, v, it, loc + " (setting quoted from the report/protocol, not a result)")

out = []
out.append("## 2. The note's items and how each was tested\n")
out.append(lead + "\n")
out.append("\n".join(t1) + "\n")
out.append("Source: R1-02 and R1-06 (`results/v2_refine/analysis/stage1_criterion.csv`, `stage2_criterion.csv`: columns `mean_q50`, "
           "`ci_mean_lo_q50`, `ci_mean_hi_q50` and the q = 60 columns, `a_met`, `b_status`; R1 bootstrap seed 20261003); R1-01 "
           "(`decision_inputs.csv`: `cost_phase_wall_ratio_vs_base`, `disp_ratio_q50`, `disp_ratio_q60`); R2B-02 (`results/v2_refine_r2b/analysis/criterion.csv`: "
           "same columns; R2b bootstrap seed 20261004); CF-01, CF-02, CF-03 (v2.0 confirmation `verdict.json`, `pass_counts.csv`, `per_run.csv`); "
           "R1-29 (`decision_d1_flags.csv`, rows `group = v11_reproduction`); R1-31 (`decision_d2_detection_limits.csv`); R1-12 (`evaluations.csv`, row count). "
           "Counts such as \"0 of 4\" are computed here from the criterion tables (script `W-F/gen_sec02.py`). Per-arm tables are in sections 4.1-4.6; "
           "the diagnostics are in section 5.\n")
out.append("**The four rounds.**\n")
out.append("\n".join(t2) + "\n")
out.append("Source: R1-05 and R1-37 (row counts: stage-1 rows; stage-2 rows with `status = done`); R2B-07 (`launch_checks.json`, key `n_ok`); R2C-06 "
           "(`launch_checks.json`, `summary.n_runs_ok`); R2C-02 and R2C-03 (`criterion.csv`, `selection.json`, key `outcome`); R2B-02 (row `A_peak50`); "
           "CF-02 (`pass_counts.csv`); CF-08 (`rehearsal_v2_0_checks.json`, keys R1-R7). Branch names and commit hashes are from the round reports "
           "named in the cells (report only).\n")
out.append(crit_text + "\n")
out.append(mapping + "\n")
out.append(peak_text + "\n")
out.append("\n".join(t_peak) + "\n")
out.append("Source: R2B-02 and R2C-02 (`criterion.csv`: `a_q50`, `a_q60`, `b_status`, `b_violations`); R2B-04 and R2C-04 (`tail.csv`: `max_tail_mean_over_g2_0`, "
           "largest over the 10 seeds of the arm); the G-A tail-mean limit 0.02 is from PL-01 (`gates`, G-A). Shares and start updates are the arm "
           "definitions of RR-09 section 3.1 and RR-10 section 2 (report only).\n")
text = "\n".join(out)

missing = [r for r in L.rows if r[2] not in text]
print("LEDGER TEXTS NOT FOUND IN TEXT:", [(r[0], r[2]) for r in missing])
with open(os.path.join(SCRATCH, "sec02.md"), "w") as fh:
    fh.write(text)
L.save("sec02_ledger.csv")
print(text)
print("ledger rows:", len(L.rows))
