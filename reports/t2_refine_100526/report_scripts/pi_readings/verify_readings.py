"""Verify the PI-side readings against the evidence pack (read-only).

Writes, next to the report parts:
  pi_readings.csv             one row per reading (R-xx) and per commit hash (H-xx)
  pi_readings_components.csv  one row per number compared (feeds the ledger)
  pi_readings.md              the Markdown table
Run: OMP_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python verify_readings.py
"""
import csv
import json
import math
import os
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

PACK = ("/home/fjiang4/tournament_experiment/.claude/worktrees/t2-refine-pack/"
        "reports/t2_refine_100526/evidence/")
OUTDIR = ("/tmp/claude-1331199693/-home-fjiang4-tournament-experiment--claude-worktrees-"
          "r2c-sampler-protocol-v2-1-c4c0f1/152f0306-5492-45a1-94d5-59493fa0d141/scratchpad/"
          "report_parts/")
HERE = OUTDIR + "pi_readings/"


def P(rel):
    return PACK + rel


def rd(rel):
    return pd.read_csv(P(rel))


def rj(rel):
    return json.load(open(P(rel)))


# ---------------------------------------------------------------- comparison helpers
def round_like(pi, x):
    """Round x to the number of digits written in the string pi; return (string, float)."""
    s = pi.strip().lstrip("+").rstrip("x")
    if "e" in s.lower():
        mant = s.lower().split("e")[0]
        d = len(mant.split(".")[1]) if "." in mant else 0
        r = float(f"{x:.{d}e}")
        return f"{x:.{d}e}", r
    d = len(s.split(".")[1]) if "." in s else 0
    q = Decimal(1).scaleb(-d)
    r = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    return str(r), float(r)


def same(pi, x):
    s = pi.strip().lstrip("+").rstrip("x")
    rs, rf = round_like(pi, x)
    return math.isclose(rf, float(s), rel_tol=0, abs_tol=1e-300) or rf == float(s), rs


COMP = []  # component rows


def num(rid, label, pi, x, item, loc):
    ok, rs = same(pi, x)
    COMP.append(dict(rid=rid, label=label, pi=pi, evidence=repr(float(x)), rounded=rs,
                     status="AGREE" if ok else "DISAGREE", item=item, locator=loc))
    return ok


def cnt(rid, label, pi, x, item, loc):
    ok = int(x) == int(pi)
    COMP.append(dict(rid=rid, label=label, pi=str(pi), evidence=str(int(x)), rounded=str(int(x)),
                     status="AGREE" if ok else "DISAGREE", item=item, locator=loc))
    return ok


def txt(rid, label, pi, ev, ok, item, loc):
    COMP.append(dict(rid=rid, label=label, pi=pi, evidence=ev, rounded=ev,
                     status="AGREE" if ok else "DISAGREE", item=item, locator=loc))
    return ok


# ---------------------------------------------------------------- R1
di1 = rd("results/v2_refine/analysis/decision_inputs.csv")
exp = di1[di1.arm == "B_expcont"].iloc[0]
L = "results/v2_refine/analysis/decision_inputs.csv row arm=B_expcont"
num("R-01", "q50 mean", "-0.02386", exp.mean_q50, "R1-01", L + " col mean_q50")
num("R-01", "q50 ci lo", "-0.04061", exp.ci_mean_lo_q50, "R1-01", L + " col ci_mean_lo_q50")
num("R-01", "q50 ci hi", "-0.009831", exp.ci_mean_hi_q50, "R1-01", L + " col ci_mean_hi_q50")
num("R-01", "q60 mean", "-0.03783", exp.mean_q60, "R1-01", L + " col mean_q60")
num("R-01", "q60 ci lo", "-0.06321", exp.ci_mean_lo_q60, "R1-01", L + " col ci_mean_lo_q60")
num("R-01", "q60 ci hi", "-0.01472", exp.ci_mean_hi_q60, "R1-01", L + " col ci_mean_hi_q60")

disp = rd("results/v2_refine/analysis/stage1_dispersion.csv")
dsel = disp[(disp.arm == "B_expcont") & (disp.metric == "stage1_rel_err_signed")].set_index("q")
Ld = "results/v2_refine/analysis/stage1_dispersion.csv row arm=B_expcont, metric=stage1_rel_err_signed"
num("R-02", "q50 ratio", "0.3464", dsel.loc[50, "ratio"], "R1-03", Ld + ", q=50, col ratio")
num("R-02", "q60 ratio", "0.2687", dsel.loc[60, "ratio"], "R1-03", Ld + ", q=60, col ratio")
# cross-check with decision_inputs
assert abs(dsel.loc[50, "ratio"] - exp.disp_ratio_q50) < 1e-12

adv = rd("results/v2_refine/analysis/stage1_adv_ratio.csv")
asel = adv[adv.arm == "B_expcont"].set_index("q")
La = "results/v2_refine/analysis/stage1_adv_ratio.csv row arm=B_expcont"
num("R-03", "q50 mean_ratio", "0.0656", asel.loc[50, "mean_ratio"], "R1-04", La + ", q=50, col mean_ratio")
num("R-03", "q60 mean_ratio", "0.0531", asel.loc[60, "mean_ratio"], "R1-04", La + ", q=60, col mean_ratio")

cost = rd("results/v2_refine/analysis/stage1_cost.csv")
cexp = cost[cost.arm == "B_expcont"].iloc[0]
num("R-04", "wall ratio", "1.084", cexp.phase_wall_ratio_vs_base, "R1-22",
    "results/v2_refine/analysis/stage1_cost.csv row arm=B_expcont col phase_wall_ratio_vs_base")
assert abs(cexp.phase_wall_ratio_vs_base - exp.cost_phase_wall_ratio_vs_base) < 1e-12

# ---------------------------------------------------------------- v2.0 confirmation
pc = rd("results/v2_T2_locked/confirmation_v2_0_analysis/pass_counts.csv").set_index("q")
Lp = "results/v2_T2_locked/confirmation_v2_0_analysis/pass_counts.csv"
cnt("R-05", "q50 n_pass", 19, pc.loc[50, "n_pass"], "CF-02", Lp + " q=50 col n_pass")
num("R-05", "q50 CP lo", "0.7513", pc.loc[50, "cp95_lo"], "CF-02", Lp + " q=50 col cp95_lo")
num("R-05", "q50 CP hi", "0.9987", pc.loc[50, "cp95_hi"], "CF-02", Lp + " q=50 col cp95_hi")
cnt("R-05", "q60 n_pass", 20, pc.loc[60, "n_pass"], "CF-02", Lp + " q=60 col n_pass")
num("R-05", "q60 CP lo", "0.8316", pc.loc[60, "cp95_lo"], "CF-02", Lp + " q=60 col cp95_lo")
num("R-05", "q60 CP hi", "1", pc.loc[60, "cp95_hi"], "CF-02", Lp + " q=60 col cp95_hi")
cnt("R-06", "q50 G-S", 20, pc.loc[50, "n_G-S_pass"], "CF-02", Lp + " q=50 col n_G-S_pass")
cnt("R-06", "q60 G-S", 20, pc.loc[60, "n_G-S_pass"], "CF-02", Lp + " q=60 col n_G-S_pass")

pr = rd("results/v2_T2_locked/confirmation_v2_0_analysis/per_run.csv")
pr["abs_signed"] = pr["stage1_rel_err_signed"].abs()
assert (pr["abs_signed"] - pr["s1"]).abs().max() < 1e-12
s1sum = rd("results/v2_T2_locked/confirmation_v2_0_analysis/s1_summary.csv")
s1sum = s1sum[s1sum.criterion == "G-S"].set_index("q")
Lc = "results/v2_T2_locked/confirmation_v2_0_analysis/per_run.csv col stage1_rel_err_signed (abs), "
for q, pi_med, pi_max, pi_sd in [(50, "0.0131", "0.0464", "0.0211"), (60, "0.0117", "0.0324", "0.0171")]:
    d = pr[pr.q == q]
    assert len(d) == 20
    num("R-07", f"q{q} median |err|", pi_med, d.abs_signed.median(), "CF-03", Lc + f"q={q}, median over 20 (computed)")
    num("R-08", f"q{q} max |err|", pi_max, d.abs_signed.max(), "CF-03", Lc + f"q={q}, max over 20 (computed)")
    num("R-09", f"q{q} SD signed", pi_sd, s1sum.loc[q, "sd_signed"], "CF-04",
        f"results/v2_T2_locked/confirmation_v2_0_analysis/s1_summary.csv q={q} row criterion=G-S col sd_signed")
    # cross-check ddof=1 recomputation
    assert abs(d.stage1_rel_err_signed.std(ddof=1) - s1sum.loc[q, "sd_signed"]) < 1e-12

p11 = rd("results/v2_T2_locked/confirmation_analysis/per_run.csv")
p11["abs_signed"] = p11["stage1_rel_err_signed"].abs()
seeds11 = (int(p11.seed.min()), int(p11.seed.max()), len(p11))
Lc = "results/v2_T2_locked/confirmation_analysis/per_run.csv col stage1_rel_err_signed (abs), "
for q, pi_med, pi_max in [(50, "0.0378", "0.0862"), (60, "0.0455", "0.1530")]:
    d = p11[p11.q == q]
    assert len(d) == 20
    num("R-10", f"q{q} median |err|", pi_med, d.abs_signed.median(), "CF-10", Lc + f"q={q}, median over 20 (computed)")
    num("R-10", f"q{q} max |err|", pi_max, d.abs_signed.max(), "CF-10", Lc + f"q={q}, max over 20 (computed)")

# ---------------------------------------------------------------- failed run
g = rj("results/v2_T2_locked/confirmation_v2_0/q50/seed30510/gates.json")
nb = rj("results/v2_refine_r2b/diag_30510/numbers.json")["values"]
num("R-11", "eta_2", "0.00580334", g["metric_values"]["eta_final"], "CF-13", "gates.json metric_values.eta_final")
num("R-12", "peak error", "-0.1458", nb["owner.peak_err_signed"], "R2B-15", "numbers.json values owner.peak_err_signed")
num("R-13", "on-path eta_2", "0.0058", nb["owner.on_max"], "R2B-15", "numbers.json values owner.on_max")
num("R-13", "off-path eta_2", "0.00082", nb["owner.off_max"], "R2B-15", "numbers.json values owner.off_max")
num("R-13", "R2b report on-path", "0.00580", nb["owner.on_max"], "R2B-15", "numbers.json values owner.on_max (report quote)")
num("R-13", "R2b report off-path", "0.000819", nb["owner.off_max"], "R2B-15", "numbers.json values owner.off_max (report quote)")
num("R-14", "smoothed share", "0.279", nb["owner.smoothed_share"], "R2B-15", "numbers.json values owner.smoothed_share")

# ---------------------------------------------------------------- check (ii)
ch = rd("results/v2_refine/analysis/decision_method6_check_ii.csv")
fin = ch[ch.tier == "final"].set_index("q")
ref_col = [c for c in ch.columns if c.startswith("refined_verifier")][0]
Lk = "results/v2_refine/analysis/decision_method6_check_ii.csv tier=final "
num("R-15", "q50 standard final tier", "3.229e-05", fin.loc[50, "max_abs_diff_over_dw"], "R1-32", Lk + "q=50 col max_abs_diff_over_dw")
num("R-15", "q60 standard final tier", "1.960e-05", fin.loc[60, "max_abs_diff_over_dw"], "R1-32", Lk + "q=60 col max_abs_diff_over_dw")
num("R-16", "q50 refined verifier", "4.982e-07", fin.loc[50, ref_col], "R1-32", Lk + "q=50 col refined_verifier...")
num("R-16", "q60 refined verifier", "4.100e-07", fin.loc[60, ref_col], "R1-32", Lk + "q=60 col refined_verifier...")
c2 = rj("results/v2_T2_locked/v2_0/continuation_check_v2_0.json")
num("R-17", "q50 (ii-a)", "4.76e-09", c2["ii_a"]["q50"]["max_abs_change_over_dw"], "PL-05",
    "continuation_check_v2_0.json ii_a.q50.max_abs_change_over_dw")
num("R-17", "q60 (ii-a)", "3.22e-09", c2["ii_a"]["q60"]["max_abs_change_over_dw"], "PL-05",
    "continuation_check_v2_0.json ii_a.q60.max_abs_change_over_dw")

# ---------------------------------------------------------------- D1 / D2
d1 = rd("results/v2_refine/analysis/decision_d1_flags.csv")
a = d1[(d1.group == "v11_reproduction") & (d1.q.astype(str) == "all") & (d1.phase == "A")].iloc[0]
Ld1 = "results/v2_refine/analysis/decision_d1_flags.csv row group=v11_reproduction,q=all,phase=A "
num("R-18", "M1 median pooled", "0.001385", a.M1_median_over_runs, "R1-29", Ld1 + "col M1_median_over_runs")
cnt("R-19", "runs above M2 share threshold", 3, a.M2_runs_share_above_threshold, "R1-29", Ld1 + "col M2_runs_share_above_threshold")
cnt("R-19", "n_runs", 20, a.n_runs, "R1-29", Ld1 + "col n_runs")
num("R-19", "max share", "0.02353", a.M2_max_share, "R1-29", Ld1 + "col M2_max_share")

bg = rd("results/v2_refine/d1_clamp/d1_by_group.csv")
hin = int(bg[bg.category == "learner_final_in"].hit_sum.sum())
hout = int(bg[bg.category == "learner_final_out"].hit_sum.sum())
hs1 = int(bg[bg.category == "learner_s1"].hit_sum.sum())
cnt("R-20", "clamp hits, final-stage rows |d|<2q (all 19 groups, both q, both phases)", 0, hin, "R1-36",
    "d1_by_group.csv category=learner_final_in, sum of hit_sum")
COMP.append(dict(rid="R-20", label="context: hits in |d|>=2q rows / stage-1 rows", pi="n/a",
                 evidence=f"final_out={hout}; learner_s1={hs1}", rounded="", status="AGREE",
                 item="R1-36", locator="d1_by_group.csv category=learner_final_out / learner_s1, sum of hit_sum"))

d2 = rd("results/v2_refine/analysis/decision_d2_detection_limits.csv")
fa = d2[(d2.family == "a") & (d2.criterion == "G-F")].iloc[0]
Lq = "results/v2_refine/analysis/decision_d2_detection_limits.csv row family=a,criterion=G-F "
txt("R-21", "q50 detection limit", "not reached", str(fa["q50 detection limit"]), "not reached" in str(fa["q50 detection limit"]), "R1-31", Lq + "col q50 detection limit")
txt("R-21", "q60 detection limit", "not reached", str(fa["q60 detection limit"]), "not reached" in str(fa["q60 detection limit"]), "R1-31", Lq + "col q60 detection limit")
num("R-21", "q50 grid max |pert|", "0.15", fa["q50 max |perturbation| on grid"], "R1-31", Lq + "col q50 max |perturbation| on grid")
num("R-21", "q50 max value", "0.00717", fa["q50 max value on grid"], "R1-31", Lq + "col q50 max value on grid")
num("R-21", "q60 max value", "0.003247", fa["q60 max value on grid"], "R1-31", Lq + "col q60 max value on grid")
fb = d2[(d2.family == "b") & (d2.criterion == "G-F")].iloc[0]
Lq = "results/v2_refine/analysis/decision_d2_detection_limits.csv row family=b,criterion=G-F "
num("R-22", "q50 stage-2 detection limit (G-F, G-A)", "0.1", float(fb["q50 detection limit"]), "R1-31", Lq + "col q50 detection limit")
num("R-22", "q60 stage-2 detection limit (G-F, G-A)", "0.15", float(fb["q60 detection limit"]), "R1-31", Lq + "col q60 detection limit")
fbA = d2[(d2.family == "b") & (d2.criterion == "G-A")].iloc[0]
assert fbA["q50 detection limit"] == fb["q50 detection limit"] and fbA["q60 detection limit"] == fb["q60 detection limit"]

# ---------------------------------------------------------------- R2b
di2 = rd("results/v2_refine_r2b/analysis/decision_inputs.csv")  # strings
cr2 = rd("results/v2_refine_r2b/analysis/criterion.csv").set_index("arm")
Lr = "results/v2_refine_r2b/analysis/criterion.csv row arm=A_peak50 "
r = cr2.loc["A_peak50"]
num("R-23", "q50 mean", "-0.0235", r.mean_q50, "R2B-02", Lr + "col mean_q50")
num("R-23", "q50 ci lo", "-0.0410", r.ci_mean_lo_q50, "R2B-02", Lr + "col ci_mean_lo_q50")
num("R-23", "q50 ci hi", "-0.0040", r.ci_mean_hi_q50, "R2B-02", Lr + "col ci_mean_hi_q50")
num("R-23", "q60 mean", "-0.0118", r.mean_q60, "R2B-02", Lr + "col mean_q60")
num("R-23", "q60 ci lo", "-0.0230", r.ci_mean_lo_q60, "R2B-02", Lr + "col ci_mean_lo_q60")
num("R-23", "q60 ci hi", "-0.0010", r.ci_mean_hi_q60, "R2B-02", Lr + "col ci_mean_hi_q60")

pr2 = rd("results/v2_refine_r2b/analysis/per_run.csv")
tt = pr2[(pr2.arm == "A_peak50") & (pr2.q == 60) & (pr2.seed.isin([10503, 10504, 10510]))].set_index("seed")
Lt = "results/v2_refine_r2b/analysis/per_run.csv rows arm=A_peak50,q=60 col stage2_tail_mean_over_g2_0 "
for sd, pv in [(10503, "0.0204"), (10504, "0.0212"), (10510, "0.0221")]:
    num("R-24", f"seed {sd} tail mean", pv, tt.loc[sd, "stage2_tail_mean_over_g2_0"], "R2B-03", Lt + f"seed={sd}")
viol = pr2[(pr2.arm == "A_peak50") & (pr2.stage2_tail_mean_over_g2_0 > 0.02)][["q", "seed"]].values.tolist()
txt("R-24", "set of q/seeds with tail mean > 0.02 for A_peak50", "q60: 10503, 10504, 10510",
    str(viol), sorted(viol) == [[60, 10503], [60, 10504], [60, 10510]], "R2B-03",
    "per_run.csv arm=A_peak50, stage2_tail_mean_over_g2_0 > 0.02 (computed); also R2B-04 tail.csv n_tail_mean_over_0.02")

tl = rd("results/v2_refine_r2b/analysis/tail.csv").set_index(["arm", "q"])
Ltl = "results/v2_refine_r2b/analysis/tail.csv col n_abs_peak_le_0.05 "
cnt("R-25", "A_base q50", 2, tl.loc[("A_base", 50), "n_abs_peak_le_0.05"], "R2B-04", Ltl + "row A_base,50")
cnt("R-25", "A_peak50 q50", 8, tl.loc[("A_peak50", 50), "n_abs_peak_le_0.05"], "R2B-04", Ltl + "row A_peak50,50")
cnt("R-25", "A_base q60", 5, tl.loc[("A_base", 60), "n_abs_peak_le_0.05"], "R2B-04", Ltl + "row A_base,60")
cnt("R-25", "A_peak50 q60", 8, tl.loc[("A_peak50", 60), "n_abs_peak_le_0.05"], "R2B-04", Ltl + "row A_peak50,60")

for arm, pq50, pq60 in [("P20_lr3e-5", "-0.0026", "0.0006"), ("P20_lr3e-4", "0.0127", "0.0029")]:
    Lm = f"results/v2_refine_r2b/analysis/criterion.csv row arm={arm} (comparator in col baseline) "
    num("R-26", f"{arm} q50", pq50, cr2.loc[arm].mean_q50, "R2B-02", Lm + "col mean_q50")
    num("R-26", f"{arm} q60", pq60, cr2.loc[arm].mean_q60, "R2B-02", Lm + "col mean_q60")
txt("R-26", "matched controls", "P20_lr3e-5 vs A_ctrl200; P20_lr3e-4 vs A_ctrl200_lr3e-4",
    f"{cr2.loc['P20_lr3e-5'].baseline}; {cr2.loc['P20_lr3e-4'].baseline}",
    cr2.loc["P20_lr3e-5"].baseline == "A_ctrl200" and cr2.loc["P20_lr3e-4"].baseline == "A_ctrl200_lr3e-4",
    "R2B-02", "criterion.csv col baseline")

# R-27: search
traj = rd("results/v2_refine_r2b/analysis/trajectory_per_run.csv")
foc0 = traj[traj["local"] == 0].groupby("q").foc_mean.median()
foc200 = traj[traj["local"] == 200].groupby(["arm", "q"]).foc_mean.median()
rm = pr2.groupby(["arm", "q"]).stage2_rmse_pos_over_g2_0.median()
Lf = "results/v2_refine_r2b/analysis/trajectory_per_run.csv col foc_mean, local=0 / 200, median over 10 seeds (computed) "
num("R-27", "FOC start q50 (parent u1600, median over seeds)", "5.05e-04", foc0.loc[50], "R2B-05", Lf + "q=50, local=0")
num("R-27", "FOC end q50 P20_lr3e-5 (median)", "3.6e-04", foc200.loc[("P20_lr3e-5", 50)], "R2B-05", Lf + "arm=P20_lr3e-5,q=50,local=200")
Lrm = "results/v2_refine_r2b/analysis/per_run.csv col stage2_rmse_pos_over_g2_0, median over 10 seeds (computed) "
num("R-27", "RMSE q50 parent -> ", "0.0201", rm.loc[("parent_u1600", 50)], "R2B-03", Lrm + "arm=parent_u1600,q=50")
num("R-27", "RMSE q50 P20_lr3e-5", "0.0149", rm.loc[("P20_lr3e-5", 50)], "R2B-03", Lrm + "arm=P20_lr3e-5,q=50")
num("R-27", "RMSE q60 parent", "0.0198", rm.loc[("parent_u1600", 60)], "R2B-03", Lrm + "arm=parent_u1600,q=60")
num("R-27", "RMSE q60 P20_lr3e-4", "0.0134", rm.loc[("P20_lr3e-4", 60)], "R2B-03", Lrm + "arm=P20_lr3e-4,q=60")

di2n = cr2.loc["A_ctrl200_lr3e-4"]
pd28 = rd("results/v2_refine_r2b/analysis/paired.csv")
row28 = pd28[(pd28.arm == "A_ctrl200_lr3e-4") & (pd28.baseline == "parent_u1600") & (pd28.q == 50)
             & (pd28.metric == "stage2_peak_rel_err_abs")].iloc[0]
num("R-28", "q50 mean diff", "-0.019", row28["mean"], "R2B-08",
    "paired.csv row arm=A_ctrl200_lr3e-4,baseline=parent_u1600,q=50,metric=stage2_peak_rel_err_abs col mean")
assert abs(row28["mean"] - cr2.loc["A_ctrl200_lr3e-4"].mean_q50) < 1e-12
di2n = cr2.loc["A_ctrl200_lr3e-4"]
cnt("R-28", "q50 n_better of 10", 8, row28["n_better"], "R2B-08",
    "paired.csv same row, col n_better (n_pairs=10)")
txt("R-28", "comparator", "parent u1600", str(di2n.baseline), di2n.baseline == "parent_u1600", "R2B-02", "criterion.csv col baseline")

# ---------------------------------------------------------------- R2c
cr3 = rd("results/v2_refine_r2c/analysis/criterion.csv").set_index("arm")
pis = {
    "R-29": ("A_peak35", ["-0.0308", "-0.0486", "-0.0136"], ["-0.0054", "-0.0164", "0.0071"]),
    "R-30": ("A_peak40", ["-0.0235", "-0.0397", "-0.0077"], ["-0.0065", "-0.0187", "0.0065"]),
    "R-31": ("A_peak50_late400", ["-0.0143", "-0.0303", "0.0005"], ["-0.0059", "-0.0179", "0.0058"]),
    "R-32": ("A_peak50_late800", ["-0.0178", "-0.0335", "-0.0013"], ["0.0002", "-0.0156", "0.0151"]),
}
for rid, (arm, p50, p60) in pis.items():
    r = cr3.loc[arm]
    Lr = f"results/v2_refine_r2c/analysis/criterion.csv row arm={arm} "
    num(rid, "q50 mean", p50[0], r.mean_q50, "R2C-02", Lr + "col mean_q50")
    num(rid, "q50 ci lo", p50[1], r.ci_mean_lo_q50, "R2C-02", Lr + "col ci_mean_lo_q50")
    num(rid, "q50 ci hi", p50[2], r.ci_mean_hi_q50, "R2C-02", Lr + "col ci_mean_hi_q50")
    num(rid, "q60 mean", p60[0], r.mean_q60, "R2C-02", Lr + "col mean_q60")
    num(rid, "q60 ci lo", p60[1], r.ci_mean_lo_q60, "R2C-02", Lr + "col ci_mean_lo_q60")
    num(rid, "q60 ci hi", p60[2], r.ci_mean_hi_q60, "R2C-02", Lr + "col ci_mean_hi_q60")

t3 = rd("results/v2_refine_r2c/analysis/tail.csv")
wave = t3[t3.arm.isin(["A_peak35", "A_peak40", "A_peak50_late400", "A_peak50_late800"])]
mx = wave.loc[wave.max_tail_mean_over_g2_0.idxmax()]
num("R-33", "largest tail mean over the four wave-S arms", "0.0194", mx.max_tail_mean_over_g2_0, "R2C-04",
    f"tail.csv col max_tail_mean_over_g2_0, max over the 4 wave-S arms x 2 q = arm {mx.arm}, q={int(mx.q)} (computed)")
sel = rj("results/v2_refine_r2c/analysis/selection.json")
txt("R-33", "no arm selected", "none", str(sel.get("selected")) + " / " + str(sel.get("outcome")),
    sel.get("selected") in (None, "None") and sel.get("outcome") == "none_eligible", "R2C-03",
    "selection.json keys selected / outcome")
ref = t3[t3.arm == "R2b_A_peak50"].set_index("q")

# R-34 facts
r2b_a50 = cr2.loc["A_peak50"]
txt("R-34a", "R2b A_peak50 part (a): CI below 0 at q50 / q60", "meets the criterion at q=50 only",
    f"(a) q50={bool(r2b_a50.a_q50)}, q60={bool(r2b_a50.a_q60)}; a_met={bool(r2b_a50.a_met)}",
    False, "R2B-02", "criterion.csv row A_peak50 cols a_q50, a_q60, a_met")
txt("R-34b", "R2b A_peak50 part (b) and overall", "tail constraint binds at q=60",
    f"b_status={r2b_a50.b_status}; violations={r2b_a50.b_violations}; overall={r2b_a50.overall}",
    str(r2b_a50.b_status) == "violated" and "q60/10503" in str(r2b_a50.b_violations), "R2B-02",
    "criterion.csv row A_peak50 cols b_status, b_violations, overall")
for arm in ["A_peak35", "A_peak40", "A_peak50_late400", "A_peak50_late800"]:
    r = cr3.loc[arm]
    txt("R-34c", f"R2c {arm}: (a) q50 / q60 / both; (b); overall", "(a) at q=50 only; (b) holds",
        f"a_q50={bool(r.a_q50)}, a_q60={bool(r.a_q60)}, a_met={bool(r.a_met)}; b_status={r.b_status}; "
        f"overall={r.overall}", True, "R2C-02", f"criterion.csv row {arm}")
txt("R-34d", "R2c largest tail mean (limit 0.02), per arm max over q", "tail constraint binds at q=60",
    "; ".join(f"{a}: q50 {wave[(wave.arm == a) & (wave.q == 50)].max_tail_mean_over_g2_0.iloc[0]:.6g}, "
              f"q60 {wave[(wave.arm == a) & (wave.q == 60)].max_tail_mean_over_g2_0.iloc[0]:.6g}"
              for a in ["A_peak35", "A_peak40", "A_peak50_late400", "A_peak50_late800"]),
    True, "R2C-04", "tail.csv col max_tail_mean_over_g2_0; n_tail_mean_over_0.02 = 0 for all four arms")
txt("R-34e", "R2b A_peak50 tail: runs with tail mean > 0.02; max tail mean at q=60", "tail constraint binds at q=60",
    f"q60 n_tail_mean_over_0.02={int(tl.loc[('A_peak50', 60), 'n_tail_mean_over_0.02'])}; "
    f"max={tl.loc[('A_peak50', 60), 'max_tail_mean_over_g2_0']:.6g}; q50 n=0, max="
    f"{tl.loc[('A_peak50', 50), 'max_tail_mean_over_g2_0']:.6g}",
    int(tl.loc[("A_peak50", 60), "n_tail_mean_over_0.02"]) == 3, "R2B-04", "tail.csv rows A_peak50")

json.dump(dict(seeds_v11_confirmation=seeds11), open(HERE + "extra_checks.json", "w"))

# ---------------------------------------------------------------- git hashes
G = json.load(open(HERE + "git_facts.json"))
G2 = json.load(open(HERE + "git_facts2.json"))
H = []  # id, as given, short, verdict, note


def hrow(hid, given, short, status, note):
    c = G["commits"].get(short) or {}
    H.append(dict(id=hid, given=given, short=short, full=c.get("full", ""), subject=c.get("subject", ""),
                  date=c.get("date", ""), status=status, note=note))


hrow("H-01", "R1 code 32a8c21", "32a8c21", "AGREE",
     "feat commit adding the R1 flags, phase P, continuation table and tools; no code file under agents/run/utils/envs/tools/tests "
     "differs between it and 655b14c (the manifest_commit of most R1 runs); 6e99e01 (other R1 manifest commit) changes only the "
     "D1 tool and its test")
hrow("H-02", "R1 pre-registration 6c902db", "6c902db", "AGREE", "adds reports/v2/refine/00_preflight.md and 01_preregistration.md")
hrow("H-03", "R1 final record 155cdec", "155cdec", "AGREE",
     "last commit of R1 (git log 32a8c21^..155cdec): adds results/v2_refine/code_pytest_final.txt, the final test-suite record")
hrow("H-04", "v2.0 lock 1d6d4d0", "1d6d4d0", "AGREE", "feat: lock; tag t2-v2-lock-v2.0 points here")
hrow("H-05", "v2.0 LOCK record f2d616c", "f2d616c", "AGREE", "chore: record v2.0 lock in protocols/LOCK; child of 1d6d4d0")
hrow("H-06", "v2.0 rehearsal/launch d2e377d", "d2e377d", "AGREE",
     "subject is the re-rehearsal records commit; it is the code state the confirmation ran at (CF-01 verdict.json commits = [d2e377d...]; "
     "tag t2-v2-confirmation-v2.0 points here), hence 'launch'; the subject line itself says re-rehearsal records, not launch")
hrow("H-07", "v2.0 confirmation records 85c294e", "85c294e", "AGREE",
     "adds results/v2_T2_locked/confirmation_v2_0/ run records and the pre-registered analysis output; child of d2e377d")
hrow("H-08", "v2.0 report 6e216e2", "6e216e2", "AGREE",
     "adds reports/v2/protocol_v2_0_confirmation.md and updates summary, README, STATE")
hrow("H-09", "v2.0 publication head b55d389", "b55d389", "AGREE",
     "head of v2-t2-refine (local and origin) and target of tag t2-v2-main-v2.0; its own subject concerns publication status and R2b P0 housekeeping")
hrow("H-10", "R2b code 1ff99bd", "1ff99bd", "AGREE", "feat: R2b mechanisms (peak starts, censored likelihood, pathwise epochs)")
hrow("H-11", "R2b launch d581b3c", "d581b3c", "AGREE",
     "subject is addendum 1 to the R2b pre-registration; it is the manifest_commit of the R2b wave-A/P runs "
     "(R2B-03 per_run.csv col manifest_commit = d581b3c for A_peak25, A_peak50, A_censored, P20_*, A_ctrl200_lr3e-4); "
     "the commit that records the runs is c6c8da6e")
hrow("H-12", "R2b final 62ecc43", "62ecc43", "AGREE",
     "head of v2-t2-r2b = 62ecc436b80d15e4740f310e2e6c72c708c1624c (local and origin); docs: correct the R2b reports after the verification pass")
hrow("H-13", "R2c code 58c2671", "58c2671", "AGREE", "feat: scheduled peak-focused start sampler and R2c wave-S tools")
hrow("H-14", "R2c results 3ad1b07", "3ad1b07", "DISAGREE",
     "3ad1b07 is 'record R2c C-R3 reproduction, tool validation and review': it is the manifest_commit (code state at launch) of the "
     "wave-S runs (R2C-01 per_run.csv manifest_commit = 3ad1b07 for A_peak35/40/late400/late800), not the results record. "
     "The commit that records the wave-S runs, launch checks and analysis tables is 3492cac4ffd49ade581d3b0651d53e296991eed4 "
     "('chore: record R2c wave-S runs, launch checks and analysis tables', 2026-10-05 19:10:24); the reports and selection outcome are in 6a8f4492")
H.append(dict(id="H-15", given="R2c final head as pushed", short="6a8f4492", full=G2["6a8f449"]["full"],
              subject=G2["6a8f449"]["subject"], date=G2["6a8f449"]["date"], status="AGREE",
              note="git rev-parse v2-t2-r2c = origin/v2-t2-r2c = " + G["branches"]["v2-t2-r2c"]))
H.append(dict(id="H-16", given="head of v2-t2-r2b (task text)", short="62ecc43", full=G["branches"]["v2-t2-r2b"],
              subject=G["commits"]["62ecc43"]["subject"], date=G["commits"]["62ecc43"]["date"], status="AGREE",
              note="rev-parse v2-t2-r2b = origin/v2-t2-r2b"))
H.append(dict(id="H-17", given="head of v2-t2-refine (task text asks for it)", short="b55d389", full=G["branches"]["v2-t2-refine"],
              subject=G["commits"]["b55d389"]["subject"], date=G["commits"]["b55d389"]["date"], status="AGREE",
              note="rev-parse v2-t2-refine = origin/v2-t2-refine; branch t2-refine-pack (this worktree) head = "
                   + G["branches"]["t2-refine-pack"] + "; origin/main = " + G2["origin/main"]))

# ---------------------------------------------------------------- assemble reading rows
comp = pd.DataFrame(COMP)
comp.to_csv(HERE + "pi_readings_components.csv", index=False)

READINGS = {
    "R-01": "R1 `B_expcont` S1 paired difference -0.02386 [-0.04061, -0.009831] (q=50) and -0.03783 [-0.06321, -0.01472] (q=60)",
    "R-02": "R1 `B_expcont` dispersion ratios 0.3464 / 0.2687 (q=50/60)",
    "R-03": "R1 `B_expcont` advantage-SD ratios 0.0656 / 0.0531",
    "R-04": "R1 `B_expcont` wall ratio 1.084x",
    "R-05": "v2.0 confirmation q=50: 19/20 with exact CI [0.7513, 0.9987]; q=60: 20/20 with exact CI [0.8316, 1]",
    "R-06": "v2.0 confirmation: G-S 20/20 at both q",
    "R-07": "v2.0 stage-1 |error| median 0.0131 / 0.0117 (q=50/60)",
    "R-08": "v2.0 stage-1 |error| max 0.0464 / 0.0324",
    "R-09": "v2.0 stage-1 SD 0.0211 / 0.0171 (SD of the signed error)",
    "R-10": "v1.1 stage-1 |error| median 0.0378 / 0.0455 and max 0.0862 / 0.1530 (fresh seeds 20501-20520)",
    "R-11": "failed run q=50 seed 30510: eta_2 0.00580334",
    "R-12": "failed run: peak error -0.1458",
    "R-13": "failed run: on-path eta_2 0.0058 against off-path 0.00082 (R2b report: 0.00580 against 0.000819)",
    "R-14": "failed run: smoothed-game share 0.279",
    "R-15": "Check (ii): 3.229e-05 / 1.960e-05 DW on the standard final tier (q=50/60)",
    "R-16": "Check (ii): 4.982e-07 / 4.100e-07 DW on the refined verifier",
    "R-17": "Table convergence <= 4.76e-09 / 3.22e-09 (v2.0 check (ii-a))",
    "R-18": "D1 phase-A M1 median 0.001385 pooled",
    "R-19": "D1 phase-A M2: 3 of 20 runs, max share 0.02353",
    "R-20": "D1: no clamp hits at |d| < 2q",
    "R-21": "D2: G-F not reached by a 15% stage-1 error (max 0.00717 at q=50, 0.003247 at q=60)",
    "R-22": "D2: stage-2 amplitude detected at 10% (q=50) and 15% (q=60)",
    "R-23": "R2b `A_peak50` paired difference -0.0235 [-0.0410, -0.0040] (q=50) and -0.0118 [-0.0230, -0.0010] (q=60)",
    "R-24": "R2b `A_peak50` tail violations at q=60 seeds 10503/10504/10510 with tail means 0.0204/0.0212/0.0221",
    "R-25": "R2b `A_peak50` runs within 0.05: 2 -> 8 (q=50) and 5 -> 8 (q=60)",
    "R-26": "R2b pathwise vs matched controls: `P20_lr3e-5` -0.0026 (q=50) / +0.0006 (q=60); `P20_lr3e-4` +0.0127 (q=50) / +0.0029 (q=60)",
    "R-27": "R2b FOC residual falling 5.05e-04 -> 3.6e-04 while RMSE falls 0.0201 -> 0.0149 and 0.0198 -> 0.0134",
    "R-28": "R2b `A_ctrl200_lr3e-4` vs parent u1600: -0.019 with 8/10 seeds better at q=50",
    "R-29": "R2c `A_peak35`: -0.0308 [-0.0486, -0.0136] (q=50) / -0.0054 [-0.0164, 0.0071] (q=60)",
    "R-30": "R2c `A_peak40`: -0.0235 [-0.0397, -0.0077] / -0.0065 [-0.0187, 0.0065]",
    "R-31": "R2c `A_peak50_late400`: -0.0143 [-0.0303, 0.0005] / -0.0059 [-0.0179, 0.0058]",
    "R-32": "R2c `A_peak50_late800`: -0.0178 [-0.0335, -0.0013] / +0.0002 [-0.0156, 0.0151]",
    "R-33": "R2c largest tail mean 0.0194; no arm selected",
    "R-34a": "D1: peak-focused starts 'meets the criterion at q=50 only' - R2b A_peak50",
    "R-34b": "D1: 'the tail constraint binds at q=60' - R2b A_peak50",
    "R-34c": "D1: 'meets the criterion at q=50 only' - R2c arms (a)/(b)",
    "R-34d": "D1: 'the tail constraint binds at q=60' - R2c arms",
    "R-34e": "D1: 'the tail constraint binds at q=60' - tail counts R2b A_peak50",
}

EVID_ITEM = {}
rows = []
for rid, given in READINGS.items():
    c = comp[comp.rid == rid]
    ok = (c.status == "AGREE").all()
    status = "AGREE" if ok else "DISAGREE"
    items = sorted(set(c.item))
    ev = "; ".join(f"{r.label}={r.evidence}" for r in c.itertuples() if r.pi != "n/a" or True)
    rnd = "; ".join(f"{r.label}={r.rounded}" for r in c.itertuples())
    notes = []
    for r in c[c.status == "DISAGREE"].itertuples():
        if r.rid.startswith("R-34"):
            continue
        notes.append(f"{r.label}: PI {r.pi}, evidence {r.evidence} -> rounded {r.rounded}")
    rows.append(dict(id=rid, reading_as_given=given, evidence_item=", ".join(items),
                     evidence_locator=" | ".join(sorted(set(c.locator))),
                     value_in_evidence_full_precision=ev, value_rounded_as_given=rnd, status=status,
                     difference_note="; ".join(notes)))

df = pd.DataFrame(rows).set_index("id")

# per-reading manual notes / status overrides
def note(rid, s):
    df.loc[rid, "difference_note"] = (df.loc[rid, "difference_note"] + " " + s).strip()


note("R-05", "Bounds are the exact Clopper-Pearson 95% CI (CF-02, cp95_lo / cp95_hi); upper bound at q=60 is exactly 1.0.")
note("R-07", "Not in a table: median of |stage1_rel_err_signed| over the 20 runs per q, computed here from CF-03 (col s1 equals |signed| to 1e-12).")
note("R-08", "Not in a table: max of |signed| over 20 runs, computed from CF-03.")
note("R-09", "CF-04 sd_signed is the sample SD (ddof=1); recomputed from CF-03 and equal to 1e-12.")
note("R-10", f"Not in a table: median and max of |signed| computed from CF-10; seeds in CF-10 are {seeds11[0]}-{seeds11[1]} (n={seeds11[2]}).")
note("R-13", "R2b report values 0.00580 / 0.000819 and the PI's 0.0058 / 0.00082 are all roundings of 0.0058033 / 0.00081898 (final-tier on/off-path max; dev-tier off-path max is 0.00081836, which would round to 0.000818).")
note("R-17", "Field: ii_a.q{50,60}.max_abs_change_over_dw (panel width 1.0 -> 0.5, nodes 6 -> 12), threshold_over_dw 1e-08, pass true. These are maxima, not bounds; '<=' is a statement about the maxima.")
note("R-18", "Pooled over both q, 20 runs, phase A, group v11_reproduction (locked pipeline); M1 outcome 'exceeded' (threshold 0.001).")
note("R-19", "M2 outcome 'exceeded' in the pooled reading (threshold share 0.01, 'more than 2 runs'); per q: 2 of 10 at q=50, 1 of 10 at q=60 (not exceeded per q).")
note("R-20", "Reading identified as the category learner_final_in (final-stage learner rows with |d| < 2q): hit_sum 0 in all 19 groups; all hits are in learner_final_out (|d| >= 2q).")
note("R-21", "Family a (stage-1 perturbation), tier final, G-F; detection limit text 'not reached on the grid'. Family e gives 0.006813 / 0.003196 (not 0.00717 / 0.003247).")
note("R-22", "Family b (stage-2 perturbation), G-F and G-A (identical): detection limit 0.1 at q=50, 0.15 at q=60 (= top of the grid, so 'detected at 15%' means the largest grid point).")
note("R-23", "R2b bootstrap seed 20261004. Evidence: q=50 [-0.04145, -0.003759]; q=60 [-0.02273, -0.001032]. The R2b report prints the same values at 4 s.f. (R2B-01, RR-05).")
note("R-24", "Seed set also confirmed by R2B-04 tail.csv (n_tail_mean_over_0.02 = 3 at q=60, 0 at q=50).")
note("R-26", "Matched controls: P20_lr3e-5 vs A_ctrl200; P20_lr3e-4 vs A_ctrl200_lr3e-4. Mean paired difference of |peak error|.")
note("R-27", "IDENTIFIED (no pack table or report holds these exact numbers): all are MEDIANS over the 10 seeds, computed here. FOC = mean |dR/de| on the bin centres (R2B-05 trajectory_per_run.csv): parent (local 0) 5.05e-04 at q=50 -> 3.61e-04 at q=50 for P20_lr3e-5 at the end (P20_lr3e-4: 3.68e-04). RMSE = stage2_rmse_pos_over_g2_0 (R2B-03): q=50 parent 0.0201 -> P20_lr3e-5 0.0149; q=60 parent 0.0198 -> P20_lr3e-4 0.0134 (P20_lr3e-5 at q=60 is 0.01453). So the two RMSE pairs belong to different arms (lr 3e-5 at q=50, lr 3e-4 at q=60) and the FOC pair is q=50 only; the R2b report quotes FOC as MEANS over seeds (R2B-01/RR-05: q=50 0.0005152 -> 0.0003927 for P20_lr3e-5; q=60 0.0004246 -> 0.000319 for P20_lr3e-4). Also the R1 A_detmean median FOC 5.045e-4 -> 4.234e-4 (q=50) is a different arm.")
note("R-28", "Paired difference of |peak error|, A_ctrl200_lr3e-4 minus parent_u1600 (bootstrap seed 20261004); -0.01901 [-0.03334, -0.004326] at q=50; criterion (b) violated at q50/10510.")
note("R-29", "R2c bootstrap seed 20261005. Only the q=50 upper bound differs: evidence -0.013549 rounds to -0.0135.")
note("R-30", "R2c bootstrap seed 20261005.")
note("R-31", "R2c bootstrap seed 20261005. Only the q=50 lower bound differs: evidence -0.030246 rounds to -0.0302.")
note("R-32", "R2c bootstrap seed 20261005.")
note("R-33", "0.0194 = 0.019374 = A_peak35 at q=60 (largest over the four wave-S arms; limit 0.02). The R2b reference A_peak50 has 0.022142 at q=60 in the same table (not a wave-S arm). 'No arm selected': selection.json selected=None, outcome none_eligible.")

# R-34 notes: exact vs shorthand
df.loc["R-34a", "status"] = "DISAGREE"
df.loc["R-34a", "difference_note"] = ("Literal reading fails for R2b A_peak50: part (a) holds at BOTH q (q50 -0.02346 [-0.04145, -0.003759], q60 -0.01176 [-0.02273, -0.001032]); the criterion is 'not met' because part (b) is violated at q=60 (3 runs). "
                                      "'q=50 only' is exact only for part (a) of the R2c arms (R-34c).")
df.loc["R-34b", "status"] = "AGREE"
df.loc["R-34b", "difference_note"] = "Exact: (b) violated by q60/10503, q60/10504, q60/10510 (tail mean > 0.02); overall 'not met'."
df.loc["R-34c", "status"] = "AGREE"
df.loc["R-34c", "difference_note"] = ("Shorthand, exact only for part (a): A_peak35, A_peak40, A_peak50_late800 meet (a) at q=50 only; A_peak50_late400 meets (a) at neither q; part (b) holds for all four; criterion overall 'not met' for all four (a_met False). "
                                      "No R2c arm 'meets the criterion' at q=50 as a criterion (which needs both q).")
df.loc["R-34d", "status"] = "AGREE"
df.loc["R-34d", "difference_note"] = ("'Binds' is shorthand for R2c: no R2c arm violates the tail limit (n_tail_mean_over_0.02 = 0 for all; largest 0.019374 A_peak35 q60, 0.019143 A_peak50_late800 q50); the limit is approached, not crossed. "
                                      "It is a violation only for R2b A_peak50 at q=60.")
df.loc["R-34e", "status"] = "AGREE"
df.loc["R-34e", "difference_note"] = "R2b A_peak50 tail: 3 of 10 runs above 0.02 at q=60 (max 0.022142), none at q=50 (max 0.016452)."

df = df.reset_index()

# H rows into same frame
hdf = pd.DataFrame([dict(id=h["id"], reading_as_given=h["given"], evidence_item="git (read-only)",
                         evidence_locator=f"git cat-file -t / show -s {h['short']}",
                         value_in_evidence_full_precision=f"{h['full']} | {h['date']} | {h['subject']}",
                         value_rounded_as_given=h["short"], status=h["status"], difference_note=h["note"])
                    for h in H])
out = pd.concat([df, hdf], ignore_index=True)
out.to_csv(HERE + "pi_readings.csv", index=False)

# also at SCRATCH root as the spec asks
out.to_csv(OUTDIR + "pi_readings.csv", index=False)

# markdown
def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


lines = ["# PI-side readings: verification against the evidence pack", "",
         "Statuses: AGREE / DISAGREE / CANNOT_VERIFY. Rounding rule: the evidence value is rounded to the digits the PI wrote.", "",
         "| id | reading as given | evidence item | value in evidence (full precision) | status | note |",
         "|---|---|---|---|---|---|"]
for r in out.itertuples():
    lines.append(f"| {r.id} | {esc(r.reading_as_given)} | {esc(r.evidence_item)} | {esc(r.value_in_evidence_full_precision)} | {r.status} | {esc(r.difference_note)} |")
lines += ["", "## Component-level detail", "",
          "| id | component | PI | evidence (full) | evidence rounded like PI | status | item | locator |",
          "|---|---|---|---|---|---|---|---|"]
for r in comp.itertuples():
    lines.append(f"| {r.rid} | {esc(r.label)} | {esc(r.pi)} | {esc(r.evidence)} | {esc(r.rounded)} | {r.status} | {r.item} | {esc(r.locator)} |")
open(OUTDIR + "pi_readings.md", "w").write("\n".join(lines) + "\n")

# ledger: one row per component number
led = []
for i, r in enumerate(comp.itertuples(), 1):
    led.append(dict(statement_id=f"PIR-{i:03d}", section="pi_readings", text=f"{r.rid} {r.label}: PI {r.pi}",
                    value=r.evidence, item_id=r.item, locator=r.locator))
pd.DataFrame(led).to_csv(OUTDIR + "pi_readings_ledger.csv", index=False)

print(out[["id", "status"]].to_string())
print(out.status.value_counts())
print(comp[comp.status == "DISAGREE"][["rid", "label", "pi", "evidence", "rounded"]].to_string())
