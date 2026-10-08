#!/usr/bin/env python3
"""Every table of report.md, derived from the evidence copies (this pack and the 100526 pack).

    python tables.py --list                 # block names and table ids
    python tables.py --block accuracy       # print one table
    python tables.py --inject ../report.md  # rewrite the marked blocks of report.md in place
    python tables.py --verify ../report.md  # exit 1 if a marked block differs from the script output

In report.md a table sits between ``<!-- TBL:name -->`` and ``<!-- /TBL:name -->``.
Nothing here is run on weights or on a verifier: only tracked CSV/JSON records are read.
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
HERE = Path(__file__).resolve().parent
PACK = HERE.parent
REPO = Path(subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"],
                                    text=True).strip())
T2R_EV = REPO / "reports/t2_refine_100526/evidence"
E_STAR0 = {50: 70.0, 60: 3500.0 / 60.0}

# ------------------------------------------------------------------ loading helpers


def ev(pack: Path, rel: str) -> pd.DataFrame:
    return pd.read_csv(Path(pack) / "evidence" / rel)


def evj(pack: Path, rel: str) -> dict:
    return json.loads((Path(pack) / "evidence" / rel).read_text())


def t2r(rel: str) -> pd.DataFrame:
    return pd.read_csv(T2R_EV / "results" / rel)


def fmt(x, nd: int = 4, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    s = ("%+." if sign else "%.") + str(nd) + "f"
    return s % x


def ci(m, lo, hi, nd: int = 4) -> str:
    return "%s [%s, %s]%s" % (fmt(m, nd, True), fmt(lo, nd, True), fmt(hi, nd, True),
                              " *" if (hi < 0 or lo > 0) else "")


def md(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def runs(pack: Path) -> pd.DataFrame:
    """MS-R1..R3 per-run tables stacked, with derived decomposition columns."""
    frames = []
    for r, tag in ((1, "MS-R1"), (2, "MS-R2"), (3, "MS-R3")):
        d = ev(pack, "results/ms_r%d/analysis/per_run.csv" % r)
        d["round"] = tag
        frames.append(d)
    d = pd.concat(frames, ignore_index=True, sort=False)
    d["abs_peak"] = d["stage2_peak_rel_err_abs"]
    d["gap_c"] = d["g2_at_0"] - d["e2_at_0"]
    d["smooth_c"] = d["g2_at_0"] - d["smoothed_e_pred_0"]
    d["rem_c"] = d["smoothed_e_pred_0"] - d["e2_at_0"]
    d["fail2"] = ~(d["G_A_pass"].astype(bool) & d["G_N_eta_pass"].astype(bool))
    s1 = d[["G_F_pass", "G_N_gmax_pass", "G_S_pass"]]
    d["has_s1"] = s1.notna().all(axis=1)
    d["fail1"] = d["has_s1"] & ~(s1.fillna(True).astype(bool).all(axis=1))
    return d


def sel(d: pd.DataFrame, rnd: str, arm: str, q: int, role: str = None) -> pd.DataFrame:
    m = d[(d["round"] == rnd) & (d["arm"] == arm) & (d["q"] == q)]
    if role:
        m = m[m["role"] == role]
    return m


def confirmation() -> pd.DataFrame:
    a = t2r("v2_refine_r2b/diag_30510/tables/tab_decomposition_all_runs.csv")
    c = t2r("v2_T2_locked/confirmation_v2_0_analysis/per_run.csv")
    m = a.merge(c[["q", "seed", "G-A_pass", "G-N_eta_pass", "G-F_pass", "G-N_gmax_pass", "G-S_pass",
                   "run_pass", "stage1_rel_err_signed", "wall_sec"]], on=["q", "seed"], suffixes=("", "_cf"))
    m["abs_peak"] = m["peak_rel_err"].abs()
    m["fail2"] = ~(m["G-A_pass"].astype(bool) & m["G-N_eta_pass"].astype(bool))
    m["fail1"] = ~(m["G-F_pass"].astype(bool) & m["G-N_gmax_pass"].astype(bool) & m["G-S_pass"].astype(bool))
    return m


# ------------------------------------------------------------------ section 7.2: accuracy

ROWS_72 = [
    ("v2.0, fresh seeds (confirmation)", "CF", None),
    ("v2.0, development seeds (`parents_A`)", "MS-R3", "parents_A"),
    ("v2.0 re-rehearsal (`rehearsal_v2_0`, full pipeline)", "MS-R3", "rehearsal_v2_0"),
    ("`MS_base2400` (MS-R1 budget control, 2400 updates)", "MS-R1", "MS_base2400"),
    ("`MS_s35a5` (MS-R1 best sampler arm)", "MS-R1", "MS_s35a5"),
    ("`t1_bb_s1` (MS-R3; = MS-R2 `NL_bb_s1`)", "MS-R3", "t1_bb_s1"),
    ("`t1_st_s16` (MS-R3; = MS-R2 `NL_st_s16`)", "MS-R3", "t1_st_s16"),
    ("`t10_st_s16`", "MS-R3", "t10_st_s16"),
    ("`relu_st_s16`", "MS-R3", "relu_st_s16"),
    ("`relu_bb_s16`", "MS-R3", "relu_bb_s16"),
]


def _cells(pack: Path) -> List[dict]:
    d = runs(pack)
    cf = confirmation()
    cells = []
    for label, rnd, arm in ROWS_72:
        for q in (50, 60):
            if rnd == "CF":
                x = cf[cf.q == q]
                cells.append(dict(
                    label=label, q=q, seeds="fresh 30501-30520", n=len(x), status="confirmed (solver v2.0; 19/20, 20/20)",
                    abs=x.abs_peak, signed=x.peak_rel_err, gap=x.rl_gap, smooth=x.smooth_gap, rem=x.remainder_gap,
                    rmse=x.rmse_over_g20, tail=x.tail_mean_over_g20, eta=x.eta2_over_dw,
                    f2=int(x.fail2.sum()), f1=int(x.fail1.sum()),
                    s1=x.stage1_rel_err_signed.abs()))
            else:
                role = "comparator" if arm in ("parents_A", "rehearsal_v2_0") else "ms_arm"
                x = sel(d, rnd, arm, q, role)
                if arm in ("parents_A", "rehearsal_v2_0"):
                    status = "development only (solver v2.0; not a confirmation)"
                else:
                    status = "development only; not confirmed"
                cells.append(dict(
                    label=label, q=q, seeds="dev 10501-10510", n=len(x), status=status,
                    abs=x.abs_peak, signed=x.stage2_peak_rel_err_signed, gap=x.gap_c, smooth=x.smooth_c,
                    rem=x.rem_c, rmse=x.stage2_rmse_pos_over_g2_0, tail=x.stage2_tail_mean_over_g2_0,
                    eta=x.eta_T_over_dw, f2=int(x.fail2.sum()),
                    f1=(int(x.fail1.sum()) if x.has_s1.all() else None),
                    s1=(x.stage1_rel_err_abs if x.has_s1.all() else pd.Series(dtype=float))))
    return cells


def accuracy_a(pack: Path) -> str:
    rows = []
    for c in _cells(pack):
        rows.append([c["label"], c["q"], "%s, n = %d" % (c["seeds"], c["n"]), fmt(c["abs"].mean()),
                     fmt(c["abs"].median()), fmt(c["signed"].median(), 4, True),
                     "%d of %d" % ((c["abs"] <= 0.05).sum(), c["n"]), c["status"]])
    return md(["row", "q", "seed set, n", "mean \\|peak\\|", "median \\|peak\\|", "median signed peak",
               "runs with \\|peak\\| <= 0.05", "status"], rows)


def accuracy_b(pack: Path) -> str:
    rows = []
    for c in _cells(pack):
        s1 = "n/a (Phase A only)" if (c["f1"] is None) else "%s (med), %s (max)" % (fmt(c["s1"].median()), fmt(c["s1"].max()))
        rows.append([c["label"], c["q"], fmt(c["gap"].median(), 2), fmt(c["smooth"].mean(), 2),
                     fmt(c["rem"].mean(), 2), fmt(c["rmse"].mean()), fmt(c["tail"].mean()),
                     fmt(c["eta"].max(), 5), str(c["f2"]) if c["f2"] is not None else "n/a",
                     "n/a" if c["f1"] is None else str(c["f1"]), s1])
    return md(["row", "q", "median gap", "mean smoothing part", "mean remainder", "mean RMSE_pos/e2*(0)",
               "mean tail mean/e2*(0)", "max eta_2/DW", "stage-2 gate failures (G-A, G-N(eta))",
               "stage-1 gate failures (G-F, G-N(Gmax), G-S)", "stage-1 \\|error\\| (S1)"], rows)


# ------------------------------------------------------------------ section 2: parameters


def params(pack: Path) -> str:
    p = json.loads((T2R_EV / "protocols/v2_T2_locked_v2_0.json").read_text())
    rows = []
    for q in (50, 60):
        g = p["records"][str(q)]["game"]
        dw = p["records"][str(q)]["dw"]
        k = g["k"]
        rows.append([q, 100 + 2 * q, "%.6g (= 1/3500)" % k, g["w_h"], g["w_l"], dw,
                     "%d-%d" % (g["e_min"], g["e_max"]), fmt(dw / (4 * q * k), 2), p["pipeline"]["flags"]["reward_mode"],
                     "%d + %d" % (p["pipeline"]["phase_A"]["updates"], p["pipeline"]["phase_B"]["updates"])])
    return md(["q", "B = 100 + 2q", "k", "w_H", "w_L", "DW = w_H - w_L", "effort range",
               "e2*(0) = DW/(4qk)", "reward_mode", "updates (terminal stage + stage 1)"], rows)


# ------------------------------------------------------------------ section 7.3: sign, formula, quadrature


def signs(pack: Path) -> str:
    rows = []
    cf = confirmation()
    rows.append(["v2.0 confirmation, fresh 30501-30520 [T2R:R2B-18]", len(cf), int((cf.peak_rel_err >= 0).sum()),
                 fmt(cf.peak_rel_err.min(), 4, True)])
    for tag, rel, col, armcol, filt in (
            ("R1 stage-2 runs, development [T2R:R1-37]", "v2_refine/analysis/stage2_per_run.csv", "stage2_peak_rel_err_signed", "arm", None),
            ("R2b runs, development [T2R:R2B-03]", "v2_refine_r2b/analysis/per_run.csv", "stage2_peak_rel_err_signed", "arm", None),
            ("R2c runs, development [T2R:R2C-01]", "v2_refine_r2c/analysis/per_run.csv", "stage2_peak_rel_err_signed", "arm", None)):
        x = t2r(rel)
        rows.append([tag, len(x), int((x[col] >= 0).sum()), fmt(x[col].min(), 4, True)])
    d = runs(pack)
    for rnd in ("MS-R1", "MS-R2", "MS-R3"):
        x = d[(d["round"] == rnd) & (d.role == "ms_arm")]
        rows.append(["%s arms, development [M%s-08]" % (rnd, rnd[-1]), len(x), int((x.stage2_peak_rel_err_signed >= 0).sum()),
                     fmt(x.stage2_peak_rel_err_signed.min(), 4, True)])
    x = d[(d["round"] == "MS-R3") & (d.role == "ms_arm") & (d.arm.str.startswith(("t1_", "t10_")))]
    rows.append(["MS-R3 `t1` and `t10` arms only [M3-08]", len(x), int((x.stage2_peak_rel_err_signed >= 0).sum()),
                 fmt(x.stage2_peak_rel_err_signed.min(), 4, True)])
    x = d[(d["round"] == "MS-R3") & (d.role == "ms_arm") & (d.arm.str.startswith("relu_"))]
    rows.append(["MS-R3 `relu` arms only [M3-08]", len(x), int((x.stage2_peak_rel_err_signed >= 0).sum()),
                 fmt(x.stage2_peak_rel_err_signed.min(), 4, True)])
    return md(["runs", "n runs", "runs with signed peak error >= 0", "most negative signed peak error"], rows)


def nonneg_runs(pack: Path) -> str:
    rows = []
    for tag, rel, tid in (("R1", "v2_refine/analysis/stage2_per_run.csv", "T2R:R1-37"),
                          ("R2b", "v2_refine_r2b/analysis/per_run.csv", "T2R:R2B-03"),
                          ("R2c", "v2_refine_r2c/analysis/per_run.csv", "T2R:R2C-01")):
        x = t2r(rel)
        for r in x[x.stage2_peak_rel_err_signed >= 0].itertuples():
            rows.append([tag, r.arm, int(r.q), int(r.seed), fmt(r.stage2_peak_rel_err_signed, 6, True), tid])
    d = runs(pack)
    x = d[(d.role == "ms_arm") & (d.stage2_peak_rel_err_signed >= 0)]
    for r in x.itertuples():
        rows.append([r.round, r.arm, int(r.q), int(r.seed), fmt(r.stage2_peak_rel_err_signed, 6, True),
                     "M%s-08" % r.round[-1]])
    return md(["round", "arm", "q", "seed", "signed peak error", "source"], rows)


def formula(pack: Path) -> str:
    """Smoothing part = e2*(0) sigma_2(0) / (sqrt(pi) q): ratio of the recorded smoothing part to the formula."""
    d = runs(pack)
    d["form"] = d.g2_at_0 * d.sigma_effort_at_0_t2 / (math.sqrt(math.pi) * d.q)
    d["ratio"] = d.smooth_c / d.form
    rows = []
    for rnd in ("MS-R1", "MS-R2", "MS-R3"):
        for role, lab in (("ms_arm", "MS arms"), ("comparator", "comparator rows (`parents_A`, `rehearsal_v2_0`, reference arms)")):
            x = d[(d["round"] == rnd) & (d.role == role)]
            # the collapsed `relu` run (e_hat_2(0) ~ 1e-4) is outside the formula's regime: shown separately
            out = x[(x.arm.str.startswith("relu_bb")) & (x.q == 50) & (x.seed == 10504)]
            rest = x.drop(out.index)
            rows.append([rnd, lab, len(x), fmt(rest.ratio.min(), 5), fmt(rest.ratio.max(), 5),
                         fmt(((rest.ratio - 1).abs()).max() * 100, 3) + " %",
                         ("%s (%d runs)" % (", ".join(fmt(v, 4) for v in out.ratio), len(out))) if len(out) else "-"])
    return md(["round", "rows", "runs", "min ratio", "max ratio", "largest deviation from 1",
               "ratio in the collapsed `relu` run(s) (excluded from the previous columns)"], rows)


def share(pack: Path) -> str:
    d = runs(pack)
    rows = []
    specs = [("rehearsal_v2_0", "MS-R3", "comparator"), ("t1_bb_s1", "MS-R3", "ms_arm"), ("t1_st_s1", "MS-R3", "ms_arm"),
             ("t1_bb_s16", "MS-R3", "ms_arm"), ("t1_st_s16", "MS-R3", "ms_arm")]
    for arm, rnd, role in specs:
        for q in (50, 60):
            x = sel(d, rnd, arm, q, role)
            rows.append([arm, q, len(x), fmt(x.gap_c.mean(), 3), fmt(x.smooth_c.mean(), 3), fmt(x.rem_c.mean(), 3),
                         "%.0f %%" % (100 * x.smooth_c.mean() / x.gap_c.mean())])
    return md(["arm", "q", "n", "mean gap", "mean smoothing part", "mean remainder", "smoothing part / gap (ratio of means)"], rows)


def quad(pack: Path) -> str:
    qd = ev(pack, "results/ms_r3/analysis/quadrature_check.csv")
    rows = []
    for r in qd.itertuples():
        rows.append([r.actor, r.starts, r.q, fmt(r.gap_s1, 3), fmt(r.smoothing_s16, 3), fmt(r.additive_pred, 3),
                     fmt(r.quadrature_pred, 3), fmt(r.gap_s16, 3), r.closer])
    n = int((qd.closer == "quadrature").sum())
    return md(["actor", "starts", "q", "gap(s=1)", "smoothing(s=16)", "additive prediction of gap(s=16)",
               "quadrature prediction", "observed gap(s=16)", "closer"], rows) + \
        "\n\nQuadrature closer in %d of %d cells (computed from `quadrature_check.csv`)." % (n, len(qd))


# ------------------------------------------------------------------ section 7.4: interventions

CRIT_SOURCES = [
    ("R1", "t2r", "v2_refine/analysis/stage2_criterion.csv", "v2_refine/analysis/stage2_per_run.csv", "T2R:R1-06"),
    ("R2b", "t2r", "v2_refine_r2b/analysis/criterion.csv", "v2_refine_r2b/analysis/per_run.csv", "T2R:R2B-02"),
    ("R2c", "t2r", "v2_refine_r2c/analysis/criterion.csv", "v2_refine_r2c/analysis/per_run.csv", "T2R:R2C-02"),
    ("MS-R1", "ev", "results/ms_r1/analysis/criterion.csv", None, "M1-09"),
    ("MS-R1 (vs `MS_base2400`)", "ev", "results/ms_r1/analysis/criterion_vs_MS_base2400.csv", None, "M1-10"),
    ("MS-R2", "ev", "results/ms_r2/analysis/criterion.csv", None, "M2-09"),
    ("MS-R3", "ev", "results/ms_r3/analysis/criterion.csv", None, "M3-09"),
]


def _guard(pack: Path, rnd: str, arm: str, per: pd.DataFrame, msd: pd.DataFrame) -> Tuple[str, str]:
    if per is not None:
        x = per[per.arm == arm]
        tail = x.stage2_tail_mean_over_g2_0.max()
        fails = int((~(x.G_A.astype(bool) & x.G_N_eta.astype(bool))).sum())
    else:
        base = "MS-R1" if rnd.startswith("MS-R1") else rnd
        x = msd[(msd["round"] == base) & (msd.arm == arm) & (msd.role == "ms_arm")]
        tail = x.stage2_tail_mean_over_g2_0.max()
        fails = int(x.fail2.sum())
    return fmt(tail, 4), "%d of %d" % (fails, len(x))


def interventions(pack: Path) -> str:
    msd = runs(pack)
    rows = []
    for rnd, kind, rel, perrel, tid in CRIT_SOURCES:
        c = ev(pack, rel) if kind == "ev" else t2r(rel)
        per = t2r(perrel) if perrel else None
        for r in c.itertuples():
            if r.arm in ("MS_base",):
                continue
            tail, fails = _guard(pack, rnd, r.arm, per, msd)
            a50, a60 = bool(r.a_q50), bool(r.a_q60)
            b = str(r.b_status)
            if a50 and a60 and b.startswith("holds"):
                verdict = "(a) and (b) met"
            else:
                parts = []
                parts.append("(a) met at %s" % ("both q" if a50 and a60 else "q=50 only" if a50 else "q=60 only" if a60 else "neither q"))
                parts.append("(b) %s" % ("holds" if b.startswith("holds") else "violated"))
                verdict = "not met: " + "; ".join(parts)
            rows.append([rnd, "`%s`" % r.arm, "`%s`" % r.baseline, ci(r.mean_q50, r.ci_mean_lo_q50, r.ci_mean_hi_q50),
                         ci(r.mean_q60, r.ci_mean_lo_q60, r.ci_mean_hi_q60), tail, fails,
                         b if not b.startswith("holds") else "holds", verdict, tid])
    return md(["round", "arm", "comparator", "paired change of \\|peak\\|, q=50 [95% CI]", "q=60 [95% CI]",
               "max tail mean/e2*(0) of the arm (limit 0.02)", "arm runs failing G-A or G-N(eta)", "(b) status",
               "verdict (as the round pre-registered it)", "source"], rows)


# ------------------------------------------------------------------ section 7.5: MS-R3


def premise(pack: Path) -> str:
    pc = evj(pack, "results/ms_r3/supervised_screen/premise_check.json")
    d = runs(pack)
    rows = []
    for act in ("t1", "relu", "t10"):
        med = pc["median_tip_deficit"][act]
        per = pc["per_seed_tip_deficit"][act]["60"]
        plateau = sum(1 for v in per.values() if v >= 6.0)
        rows.append([act, fmt(med["50"], 2), fmt(med["60"], 2),
                     "%d of 10 seeds >= 6.0 (range %s-%s)" % (plateau, fmt(min(v for v in per.values() if v >= 6.0), 2), fmt(max(v for v in per.values() if v >= 6.0), 2)) if plateau else "0 of 10 seeds >= 6.0"])
    return md(["actor", "median tip deficit q=50 (effort units)", "q=60", "q=60 plateau"], rows)


def screen_vs_rl(pack: Path) -> str:
    d = runs(pack)
    sm = ev(pack, "results/ms_r3/supervised_screen/summary_median.csv")
    sm = sm[sm.steps == 56000]
    rows = []
    n_larger, cells, ratios = 0, 0, []
    for act in ("t1", "relu", "t10"):
        for st in ("bb", "st"):
            for q in (50, 60):
                scr = float(sm[(sm.actor == act) & (sm.starts == st) & (sm.q == q)].tip_deficit_median.iloc[0])
                x1 = sel(d, "MS-R3", "%s_%s_s1" % (act, st), q, "ms_arm").gap_c.median()
                x16 = sel(d, "MS-R3", "%s_%s_s16" % (act, st), q, "ms_arm").gap_c.median()
                cells += 1
                n_larger += int(x1 > scr)
                ratios.append(x1 / scr)
                rows.append([act, st, q, fmt(scr, 3), fmt(x1, 3), fmt(x16, 3), fmt(x1 / scr, 2)])
    return md(["actor", "starts", "q", "supervised screen: median tip deficit (56,000 steps)", "RL median gap, s=1",
               "RL median gap, s=16", "RL s=1 median gap / screen"], rows) + \
        "\n\nThe RL median gap at s=1 exceeds the screen's median deficit in %d of %d cells (ratios %s to %s)." % (
            n_larger, cells, fmt(min(r for r in ratios if r > 1), 1), fmt(max(ratios), 1))


def relu_fail(pack: Path) -> str:
    d = runs(pack)
    x = d[(d["round"] == "MS-R3") & (d.arm.str.startswith("relu_")) & (d.role == "ms_arm")]
    f = x[x.fail2]
    rows = []
    for r in f.sort_values(["seed", "arm"]).itertuples():
        rows.append([r.arm, int(r.q), int(r.seed), fmt(r.e2_at_0, 4), fmt(r.eta_T_over_dw, 5),
                     fmt(r.stage2_rmse_pos_over_g2_0, 4), fmt(r.stage2_tail_mean_over_g2_0, 4)])
    tab = md(["arm", "q", "seed", "e_hat_2(0)", "eta_2/DW (limit 0.005)", "RMSE_pos/e2*(0)", "tail mean/e2*(0)"], rows)
    n = len(f)
    cases = f.groupby(["q", "seed"]).ngroups
    n60 = int(x[x.q == 60].fail2.sum())
    return tab + "\n\n%d of %d `relu` runs fail G-A or G-N(eta) (%d at q=50, %d at q=60), from %d (q, seed) cases; `t1` and `t10`: %d failures in %d runs." % (
        n, len(x), int(x[x.q == 50].fail2.sum()), n60, cases,
        int(d[(d["round"] == "MS-R3") & (d.arm.str.match(r"^(t1|t10)_")) & (d.role == "ms_arm")].fail2.sum()),
        len(d[(d["round"] == "MS-R3") & (d.arm.str.match(r"^(t1|t10)_")) & (d.role == "ms_arm")]))


def relu_typical(pack: Path) -> str:
    d = runs(pack)
    rows = []
    for st in ("bb", "st"):
        for s in (1, 16):
            for q in (50, 60):
                r = sel(d, "MS-R3", "relu_%s_s%d" % (st, s), q, "ms_arm")
                t = sel(d, "MS-R3", "t1_%s_s%d" % (st, s), q, "ms_arm")
                rows.append(["relu_%s_s%d" % (st, s), q, fmt(r.gap_c.median(), 2), fmt(t.gap_c.median(), 2),
                             "%d / %d" % ((r.gap_c <= 1).sum(), (t.gap_c <= 1).sum()),
                             fmt(r.smooth_c.mean(), 3) + " / " + fmt(t.smooth_c.mean(), 3),
                             fmt(r.stage2_tail_mean_over_g2_0.mean(), 4) + " / " + fmt(t.stage2_tail_mean_over_g2_0.mean(), 4),
                             int(r.fail2.sum())])
    return md(["relu arm", "q", "median gap (relu)", "median gap (t1 same starts, s)", "runs with gap <= 1: relu / t1",
               "mean smoothing part: relu / t1", "mean tail mean/e2*(0): relu / t1", "relu runs failing G-A/G-N(eta)"], rows)


def t10_vs_t1(pack: Path) -> str:
    d = runs(pack)
    rows = []
    for st in ("bb", "st"):
        for s in (1, 16):
            for q in (50, 60):
                a = sel(d, "MS-R3", "t10_%s_s%d" % (st, s), q, "ms_arm")
                b = sel(d, "MS-R3", "t1_%s_s%d" % (st, s), q, "ms_arm")
                rows.append(["t10_%s_s%d" % (st, s), q, fmt(a.abs_peak.mean()), fmt(b.abs_peak.mean()),
                             fmt(a.abs_peak.mean() - b.abs_peak.mean(), 4, True),
                             fmt(a.w1d_bend_d_min.median(), 1) if "w1d_bend_d_min" in a else "n/a",
                             fmt(b.w1d_bend_d_min.median(), 1), fmt(a.w_eff.mean(), 2), fmt(b.w_eff.mean(), 2)])
    return md(["arm", "q", "mean \\|peak\\| t10", "t1", "difference", "t10 bend width of sharpest unit (units of d, median)",
               "t1", "w_eff t10 (units of d, mean)", "t1"], rows)


def transmission(pack: Path) -> str:
    out = []
    for tag, rel in (("MS-R2", "results/ms_r2/analysis/transmission.csv"), ("MS-R3", "results/ms_r3/analysis/transmission.csv")):
        t = ev(pack, rel)
        for r in t.itertuples():
            actor = getattr(r, "actor", "t1")
            starts = getattr(r, "starts", None)
            out.append([tag, "`%s`" % r.arm, int(r.q), fmt(r.mean_d_gap, 3, True), fmt(r.mean_d_smoothing, 3, True),
                        "%s [%s, %s]%s" % (fmt(r.ratio, 3, True), fmt(r.ci_lo, 3, True), fmt(r.ci_hi, 3, True),
                                           " *" if (r.ci_lo > 0 or r.ci_hi < 0) else "")])
    return md(["round", "arm (s=16 vs s=1 of the same starts)", "q", "mean change of gap", "mean change of smoothing part",
               "transmission ratio [95% CI]"], out)


# ------------------------------------------------------------------ F_d (B8)


def fd(pack: Path) -> str:
    d = runs(pack)
    x = d[(d["round"] == "MS-R3") & (d.role == "ms_arm")].copy()
    x["sm_d"] = 2 * x.sigma_effort_at_0_t2 / math.sqrt(math.pi) / (x.g2_at_0 / (2 * x.q))
    # smoothing part expressed in units of d: smoothing / (e2*(0) / 2q) = 2 q smoothing / e2*(0)
    x["sm_d"] = x.smooth_c / (x.g2_at_0 / (2 * x.q))
    x["sm_w"] = 2 * x.sigma_effort_at_0_t2 / math.sqrt(math.pi)
    rows = []
    for arm, g in x.groupby("arm", sort=False):
        for q, h in g.groupby("q"):
            wm = h.w_eff.mean()
            sm = 2 * h.sigma_effort_at_0_t2.mean() / math.sqrt(math.pi)
            fm = math.sqrt(wm ** 2 - sm ** 2) if wm ** 2 > sm ** 2 else float("nan")
            pr = np.sqrt(np.clip(h.w_eff ** 2 - (2 * h.sigma_effort_at_0_t2 / math.sqrt(math.pi)) ** 2, 0, None))
            collapsed = ((h.arm.str.startswith("relu_bb")) & (q == 50)).any()
            rows.append([arm, int(q), fmt(h.w_eff.mean(), 2), fmt(sm, 2), fmt(fm, 2), fmt(float(np.median(pr)), 2),
                         "contains the collapsed run (seed 10504)" if collapsed else ""])
    return md(["arm", "q", "mean w_eff (units of d)", "2 sigma_2(0)/sqrt(pi) from mean sigma_2(0)", "F_d from arm means",
               "median over runs of the per-run F_d", "note"], rows)


# ------------------------------------------------------------------ MS-R1: stop rule, R0


def stoprule(pack: Path) -> str:
    r = ev(pack, "results/ms_r1/analysis/rule.csv")
    r = r[r.arm.isin(["MS_rule", "MS_s25a0", "MS_s25a5", "MS_s35a0", "MS_s35a5"])]
    rows = []
    for arm, g in r.groupby("arm", sort=False):
        rows.append([arm, int(g.t2_n_fire.sum()), int(g.n_runs.sum()), int(g.t2_n_budget_forced.sum()),
                     "%d / %d" % (int(g[g.q == 50].t2_n_runs_with_polish.sum()), int(g[g.q == 50].n_runs.sum())),
                     "%d / %d" % (int(g[g.q == 60].t2_n_runs_with_polish.sum()), int(g[g.q == 60].n_runs.sum())),
                     int(g.t2_n_would_fire.sum())])
    return md(["rule arm", "terminal-stage runs that stopped before the cap", "runs", "runs ended by the cap (budget_forced)",
               "runs with a polishing block, q=50", "q=60", "runs where the rule would have fired (record)"], rows)


def r0(pack: Path) -> str:
    t = ev(pack, "results/ms_r3/analysis/r0_spearman.csv")
    t = t[t.scope == "freeze"]
    rows = [[r.actors, r.q, r.n, fmt(r.spearman_R0_vs_abs_peak, 3)] for r in t.itertuples()]
    return md(["actors", "q", "n runs (terminal freeze)", "Spearman(R0, \\|peak error\\|)"], rows)


# ------------------------------------------------------------------ all-arms overview (7.7)


def arms_all(pack: Path) -> str:
    d = runs(pack)
    rows = []
    order = [("MS-R1", a) for a in ("MS_base", "MS_base2400", "MS_rule", "MS_s25a0", "MS_s25a5", "MS_s35a0", "MS_s35a5")] + \
            [("MS-R2", a) for a in ("NL_bb_s4", "NL_st_s4")] + \
            [("MS-R3", a) for a in ("t1_bb_s1", "t1_bb_s16", "t1_st_s1", "t1_st_s16", "relu_bb_s1", "relu_bb_s16",
                                    "relu_st_s1", "relu_st_s16", "t10_bb_s1", "t10_bb_s16", "t10_st_s1", "t10_st_s16")]
    for rnd, arm in order:
        for q in (50, 60):
            x = sel(d, rnd, arm, q, "ms_arm")
            if not len(x):
                continue
            rows.append([rnd, arm, q, len(x), fmt(x.abs_peak.mean()), "%d" % (x.abs_peak <= 0.05).sum(),
                         fmt(x.stage2_rmse_pos_over_g2_0.mean()), fmt(x.stage2_tail_mean_over_g2_0.mean()),
                         fmt(x.eta_T_over_dw.max(), 5), int(x.fail2.sum()), int(x.fail1.sum()),
                         fmt(x.stage1_rel_err_abs.median()), fmt(x.stage1_rel_err_abs.max())])
    return md(["round", "arm", "q", "n", "mean \\|peak\\|", "runs <= 0.05", "mean RMSE_pos/e2*(0)", "mean tail mean/e2*(0)",
               "max eta_2/DW", "stage-2 gate failures", "stage-1 gate failures", "median stage-1 \\|error\\|",
               "max stage-1 \\|error\\|"], rows)


def stage1_ref(pack: Path) -> str:
    c = confirmation()
    d = runs(pack)
    rows = []
    for q in (50, 60):
        x = c[c.q == q].stage1_rel_err_signed.abs()
        rows.append(["v2.0 confirmation, fresh 30501-30520 [T2R:CF-03]", q, len(x), fmt(x.median()), fmt(x.max()),
                     int((x > 0.05).sum())])
        y = sel(d, "MS-R3", "rehearsal_v2_0", q, "comparator").stage1_rel_err_abs
        rows.append(["`rehearsal_v2_0`, development [M3-08]", q, len(y), fmt(y.median()), fmt(y.max()), int((y > 0.05).sum())])
    return md(["row", "q", "n", "median S1 \\|error\\|", "max", "runs above the G-S limit 0.05"], rows)


# ------------------------------------------------------------------ trajectory (MS-R3)


def trajectory(pack: Path) -> str:
    t = ev(pack, "results/ms_r3/analysis/trajectory_by_arm.csv")
    rows = []
    for (arm, q), g in t.groupby(["arm", "q"], sort=False):
        g = g.sort_values("local")
        g = g[(g.local >= 1800) & (g.local <= 2800)]
        m = g.set_index("local").e2_at_0_mean
        d_ = m.diff().abs().dropna()
        rows.append([arm, int(q), fmt(m.loc[1800], 2), fmt(m.loc[2400], 2), fmt(m.loc[2800], 2),
                     fmt(m.loc[2800] - m.loc[1800], 2, True), fmt(d_.max(), 2), fmt(m.max() - m.min(), 2)])
    return md(["arm", "q", "mean e_hat_2(0) at local 1800", "at 2400", "at 2800", "change 1800 to 2800",
               "largest move between consecutive checks (25 updates)", "range over 1800-2800"], rows)


# ------------------------------------------------------------------ floor and targets (10)


def floor(pack: Path) -> str:
    d = runs(pack)
    cf = confirmation()
    out = []
    for q in (50, 60):
        for lab, rnd, arm, role in (("v2.0, fresh seeds (n=20) [T2R:R2B-18]", "CF", None, None),
                                    ("v2.0, development (`parents_A`)", "MS-R3", "parents_A", "comparator"),
                                    ("`t1_st_s1`", "MS-R3", "t1_st_s1", "ms_arm"), ("`t1_st_s16`", "MS-R3", "t1_st_s16", "ms_arm"),
                                    ("`t10_st_s16`", "MS-R3", "t10_st_s16", "ms_arm"), ("`relu_st_s16`", "MS-R3", "relu_st_s16", "ms_arm"),
                                    ("`relu_bb_s16`", "MS-R3", "relu_bb_s16", "ms_arm")):
            if rnd == "CF":
                x = cf[cf.q == q]
                sig, ab = x.sigma0, x.abs_peak
            else:
                x = sel(d, rnd, arm, q, role)
                sig, ab = x.sigma_effort_at_0_t2, x.abs_peak
            # smoothing floor relative to e2*(0): sigma_2(0) / (sqrt(pi) q)
            fl = float((sig / (math.sqrt(math.pi) * q)).mean())
            out.append([lab, q, len(x), fmt(sig.mean(), 3), fmt(100 * fl, 2) + " %", fmt(ab.mean()), fmt(ab.mean() / fl, 1)])
    return md(["row", "q", "n", "mean sigma_2(0) (effort units)", "smoothing floor sigma_2(0)/(sqrt(pi) q), % of e2*(0)",
               "mean \\|peak\\|", "mean \\|peak\\| / floor"], out)


def floor_s(pack: Path) -> str:
    d = runs(pack)
    rows = []
    for q in (50, 60):
        for s, arms in ((1, ["t1_bb_s1", "t1_st_s1", "t10_bb_s1", "t10_st_s1"]), (16, ["t1_bb_s16", "t1_st_s16", "t10_bb_s16", "t10_st_s16"])):
            v = []
            for a in arms:
                x = sel(d, "MS-R3", a, q, "ms_arm")
                v.append(float((x.sigma_effort_at_0_t2 / (math.sqrt(math.pi) * q)).mean()))
            rows.append([q, s, fmt(100 * min(v), 2) + " - " + fmt(100 * max(v), 2) + " %"])
    return md(["q", "s", "smoothing floor sigma_2(0)/(sqrt(pi) q), % of e2*(0) (range over the tanh arms)"], rows)


# ------------------------------------------------------------------ workload (B12)


def workload(pack: Path) -> str:
    rows = []
    for tag, rel, bud, rnd in (("MS-R1 pilot", "results/ms_r1/pilot/launch_20261007_055045.json", "M1-18", None),
                               ("MS-R1 base wave", "results/ms_r1/base/launch_20261007_030049.json", "M1-21", None),
                               ("MS-R2 pilot", "results/ms_r2/pilot/launch_20261007_094801.json", "M2-17", None),
                               ("MS-R3 pilot", "results/ms_r3/pilot/launch_20261008_002423.json", "M3-20", None)):
        j = evj(pack, rel)
        w = np.array([r["wall_sec"] for r in j["runs"]])
        rows.append([tag, "`%s`" % bud, len(w), j["workers"], j["started"], fmt(w.min(), 0), fmt(np.median(w), 0), fmt(w.max(), 0),
                     fmt(w.sum() / 3600.0, 1), fmt(w.sum() / 3600.0 / j["workers"], 2), j["state"]])
    c = confirmation()
    w = c.wall_sec.values
    rows.append(["v2.0 confirmation (40 runs) [T2R:CF-03]", "`T2R:CF-03`", len(w), "not in table", "-", fmt(w.min(), 0), fmt(np.median(w), 0),
                 fmt(w.max(), 0), fmt(w.sum() / 3600.0, 1), "n/a", "-"])
    return md(["wave", "record", "runs", "workers", "launched", "per-run wall min (s)", "median", "max",
               "sum of per-run wall (hours)", "sum / workers (hours; derived lower bound of the elapsed time)", "state"], rows)


def gate_counts(pack: Path) -> str:
    d = runs(pack)
    rows = []
    for rnd in ("MS-R1", "MS-R2", "MS-R3"):
        x = d[(d["round"] == rnd) & (d.role == "ms_arm")]
        rows.append([rnd, len(x), int(x.fail2.sum()), int(x.fail1.sum()),
                     fmt(x.eta_T_over_dw.max(), 5), fmt(x.stage2_tail_mean_over_g2_0.max(), 4)])
    return md(["round", "terminal-stage runs", "stage-2 failures (G-A or G-N(eta))", "stage-1 failures (G-F, G-N(Gmax), G-S)",
               "max eta_2/DW", "max tail mean/e2*(0)"], rows)


def eta_t1t10(pack: Path) -> str:
    d = runs(pack)
    x = d[(d["round"] == "MS-R3") & (d.role == "ms_arm") & (d.arm.str.match(r"^(t1|t10)_"))]
    return md(["runs", "min eta_2/DW", "max eta_2/DW", "gate limit", "runs passing every gate"],
              [["MS-R3 `t1` and `t10`, %d runs [M3-08]" % len(x), fmt(x.eta_T_over_dw.min(), 5), fmt(x.eta_T_over_dw.max(), 5),
                "0.005", "%d of %d" % (int((~x.fail2 & ~x.fail1).sum()), len(x))]])


def budget_control(pack: Path) -> str:
    c = ev(pack, "results/ms_r1/analysis/criterion.csv")
    r = c[c.arm == "MS_base2400"].iloc[0]
    return md(["row", "q=50 [95% CI]", "q=60 [95% CI]"],
              [["`MS_base2400` minus `parents_A` (1600 to 2400 updates)", ci(r.mean_q50, r.ci_mean_lo_q50, r.ci_mean_hi_q50),
                ci(r.mean_q60, r.ci_mean_lo_q60, r.ci_mean_hi_q60)]])


# ------------------------------------------------------------------ rounds table (6)

ROUNDS_META = [
    # round, dates, branch/head, prompt, question, design, report
    ("R1", "2026-10-03", "`v2-t2-refine` (R1 final record `155cdece`)", "prompt 12 (not available) [T2R:RR-01]",
     "six single-factor changes and two diagnostics on the locked v1.1 pipeline",
     "development seeds 10501-10510 x q in {50, 60}; methods 1-6, one at a time, paired by (q, seed)", "`reports/v2/refine/summary.md` [T2R:RR-01]"),
    ("v2.0", "2026-10-03/04", "`v2-t2-refine` (head `b55d3890`; lock `1d6d4d00`)", "prompt 13 (not available) [T2R:RR-03]",
     "lock protocol v2.0 (method 6 + gate G-S) and confirm it on fresh seeds",
     "re-rehearsal on 10501-10510; confirmation on fresh seeds 30501-30520, 40 runs, pass rule >= 18 of 20 per q", "`reports/v2/protocol_v2_0_confirmation.md` [T2R:RR-03]"),
    ("R2b", "2026-10-04", "`v2-t2-r2b` (head `62ecc436`)", "prompt 14 (not available) [T2R:RR-04]",
     "three mechanisms for the stage-2 peak and a diagnostic of the failed run",
     "peak-focused starts (shares 0.25/0.50), censored likelihood, pathwise epochs; development seeds", "`reports/v2/refine_r2b/summary.md` [T2R:RR-04]"),
    ("R2c", "2026-10-05", "`v2-t2-r2c` (head `6a8f4492`)", "prompt 15 (transcription) [T2R:RR-06]",
     "four start-distribution arms with a pre-registered selection rule",
     "shares 0.35/0.40/0.50 (two timings); development seeds, 80 runs", "`reports/v2/refine_r2c/summary.md` [T2R:RR-06]"),
    ("MS-R1", "2026-10-07", "`ms-r1` (head `71c58904`)", "prompt 17 [PI-01]; G1 reply [PI-02]",
     "restore the development stop rule, add coverage-constrained verifier-guided starts and targeted polishing",
     "5 rule/sampler arms + budget control `MS_base2400`; development seeds; 120 pilot runs + 20 base runs; terminal stage 2000-2400 updates",
     "`reports/ms/r1/summary.md` [M1-01]"),
    ("MS-R2", "2026-10-07", "`ms-r2` (head `e8eb9a08`)", "prompt 19 [PI-03]",
     "is the tip deficit the policy-noise floor? noise landing (scale s in {1, 4, 16}) x sampler",
     "6 arms; development seeds; 120 pilot runs; terminal stage 2800 updates", "`reports/ms/r2/summary.md` [M2-01]"),
    ("MS-R3", "2026-10-08", "`ms-r3` (head `be4fd202`)", "prompt 20 [PI-04]",
     "is the tip deficit set by the actor's resolution at the kink? actor (t1/relu/t10) x sampler x s in {1, 16}",
     "12 arms; development seeds; 240 pilot runs; terminal stage 2800 updates", "`reports/ms/r3/summary.md` [M3-01]"),
]


def _outcome(pack: Path, rnd: str) -> str:
    def cnt(c: pd.DataFrame, skip=("MS_base",)) -> Tuple[int, int, int]:
        c = c[~c.arm.isin(skip)]
        both = sum(1 for r in c.itertuples() if r.a_q50 and r.a_q60)
        full = sum(1 for r in c.itertuples() if r.a_q50 and r.a_q60 and str(r.b_status).startswith("holds"))
        return len(c), both, full
    if rnd == "R1":
        c = t2r("v2_refine/analysis/stage2_criterion.csv")
        c = c[c.baseline == "A_base"] if False else c
        n, both, full = cnt(c)
        return "stage-2 criterion: %d of %d rows meet (a) at both q; %d meet (a) and (b) [T2R:R1-06]. Method 6 met its (stage-1) criterion and entered v2.0 [T2R:RR-01]" % (both, n, full)
    if rnd == "v2.0":
        pc = t2r("v2_T2_locked/confirmation_v2_0_analysis/pass_counts.csv")
        return "pre-registered rule (>= 18 of 20 per q) passed: %s [T2R:CF-02]" % "; ".join(
            "q=%d: %d/%d (exact 95%% CI [%.4f, %.4f])" % (int(r.q), int(r.n_pass), int(r.n_expected), r.cp95_lo, r.cp95_hi)
            for r in pc.itertuples())
    if rnd == "R2b":
        c = t2r("v2_refine_r2b/analysis/criterion.csv")
        n, both, full = cnt(c)
        return "%d of %d rows meet (a) at both q; %d meet (a) and (b) [T2R:R2B-02]" % (both, n, full)
    if rnd == "R2c":
        sel_ = json.loads((T2R_EV / "results/v2_refine_r2c/analysis/selection.json").read_text())
        c = t2r("v2_refine_r2c/analysis/criterion.csv")
        n, both, full = cnt(c)
        return "%d of %d arms meet (a) at both q; %d meet (a) and (b); selection record: %s [T2R:R2C-02, T2R:R2C-03]" % (
            both, n, full, "no arm selected" if not sel_.get("selected") else "selected " + str(sel_.get("selected")))
    if rnd == "MS-R1":
        c = ev(pack, "results/ms_r1/analysis/criterion.csv")
        n, both, full = cnt(c)
        return "%d of %d rows meet (a) at both q; %d meet (a) and (b) [M1-09]" % (both, n, full)
    if rnd == "MS-R2":
        c = ev(pack, "results/ms_r2/analysis/criterion.csv")
        n, both, full = cnt(c)
        return "%d of %d rows meet (a) at both q; %d meet (a) and (b) [M2-09]" % (both, n, full)
    c = ev(pack, "results/ms_r3/analysis/criterion.csv")
    n, both, full = cnt(c)
    return "%d of %d rows meet (a) at both q; %d meet (a) and (b) [M3-09]" % (both, n, full)


FINDINGS = {
    "R1": "only expected continuation (method 6, stage 1) meets its criterion; no stage-2 mechanism is admissible [T2R:RR-01]",
    "v2.0": "v2.0 passes its confirmation; one run (q=50, seed 30510) fails G-A through eta_2 [T2R:CF-02, T2R:CF-13]",
    "R2b": "peak-focused starts at share 0.50 meet (a) at both q but violate (b) at q=60; pathwise and censored arms not separated from controls [T2R:RR-04]",
    "R2c": "no arm selected: all four hold (b); none meets (a) at q=60 [T2R:R2C-03]",
    "MS-R1": "samplers meet (a) at q=50 only; the budget control reproduces part of the improvement; the stop rule never fires at rho_2 = 0.05 [M1-01]",
    "MS-R2": "the noise landing lowers the smoothing part as predicted; the remainder rises and the gap does not follow [M2-01]",
    "MS-R3": "`relu` lowers the typical tie deficit but 5 runs fail G-A; `t10` shows no detectable change of the mean deficit [M3-01]",
}


def rounds(pack: Path) -> str:
    rows = []
    for rnd, date, br, prompt, q, des, rep in ROUNDS_META:
        rows.append([rnd, date, br, prompt, q, des, _outcome(pack, rnd), FINDINGS[rnd], rep])
    return md(["round", "dates", "branch (head)", "prompt", "question", "design (arms, seeds, q, budget)",
               "pre-registered criterion: outcome", "main finding", "report"], rows)


# ------------------------------------------------------------------ comparison table (10)


def compare(pack: Path) -> str:
    cf = confirmation()
    d = runs(pack)
    a50, a60 = cf[cf.q == 50], cf[cf.q == 60]
    rows = [
        ["What the path consists of", "No new runs; v2.0 stays the T=2 solver; MS-R1..R3 produce no protocol change [T2R:PL-01]",
         "New round(s) on development seeds; adoption of anything needs a lock, a re-rehearsal and a fresh-seed confirmation (precedent: the v2.0 round [T2R:RR-03])"],
        ["Accuracy on the fresh seeds it rests on (v2.0, n = 20 per q)",
         "mean \\|peak\\| %s (q=50) / %s (q=60); runs <= 0.05: %d / %d of 20 [T2R:R2B-18]" % (
             fmt(a50.abs_peak.mean()), fmt(a60.abs_peak.mean()), (a50.abs_peak <= 0.05).sum(), (a60.abs_peak <= 0.05).sum()),
         "The same numbers are the starting point; no MS configuration has a fresh-seed number"],
        ["Best development-seed numbers (not confirmed)", "-",
         "mean \\|peak\\|: `relu_st_s16` %s / %s; `t10_st_s16` %s / %s; `t1_st_s16` %s / %s (q=50 / q=60) [TBL-accuracy]" % (
             fmt(sel(d, 'MS-R3', 'relu_st_s16', 50, 'ms_arm').abs_peak.mean(), 4), fmt(sel(d, 'MS-R3', 'relu_st_s16', 60, 'ms_arm').abs_peak.mean(), 4),
             fmt(sel(d, 'MS-R3', 't10_st_s16', 50, 'ms_arm').abs_peak.mean(), 4), fmt(sel(d, 'MS-R3', 't10_st_s16', 60, 'ms_arm').abs_peak.mean(), 4),
             fmt(sel(d, 'MS-R3', 't1_st_s16', 50, 'ms_arm').abs_peak.mean(), 4), fmt(sel(d, 'MS-R3', 't1_st_s16', 60, 'ms_arm').abs_peak.mean(), 4))],
        ["Main risk", "The tip deficit stays at its measured size (negative signed error in 40 of 40 confirmation runs [T2R:R2B-18]); a reader who needs a tighter peak is not served",
         "`relu` failures at q=50 (5 of 40 runs, 2 (q, seed) cases) and any guard-rail regression; the failure rate cannot be estimated from the current runs [M3-09]"],
        ["Cost in recorded units", "none beyond the write-up",
         "see the workload table (runs and per-run wall times of comparable rounds) [TBL-workload]; person-time cannot be estimated"],
    ]
    return md(["", "Path A: close now", "Path B: continue improving accuracy"], rows)


# ------------------------------------------------------------------ registry

BLOCKS: Dict[str, Tuple[str, str, Callable[[Path], str], str]] = {
    "params": ("TBL-params", "Game and protocol parameters", params, "T2R:PL-01"),
    "rounds": ("TBL-rounds", "Completed work, one row per round", rounds, "T2R:R1-06, T2R:R2B-02, T2R:R2C-02, M1-09, M2-09, M3-09"),
    "accuracy_a": ("TBL-accuracy-a", "Current accuracy: peak error", accuracy_a, "T2R:R2B-18, T2R:CF-03, M1-08, M3-08"),
    "accuracy_b": ("TBL-accuracy-b", "Current accuracy: decomposition and guard rails", accuracy_b, "T2R:R2B-18, T2R:CF-03, M1-08, M3-08"),
    "signs": ("TBL-signs", "Sign of the tip deficit by round", signs, "T2R:R1-37, T2R:R2B-03, T2R:R2C-01, T2R:R2B-18, M1-08, M2-08, M3-08"),
    "nonneg": ("TBL-nonneg", "Runs with a non-negative signed peak error", nonneg_runs, "M1-08, M2-08, M3-08"),
    "formula": ("TBL-formula", "Smoothing-part formula check", formula, "M1-08, M2-08, M3-08"),
    "share": ("TBL-share", "Smoothing part and remainder at s = 1 and s = 16", share, "M3-08"),
    "quad": ("TBL-quad", "Additive against quadrature", quad, "M3-12"),
    "interventions": ("TBL-interventions", "What moved the tip and what did not", interventions,
                      "T2R:R1-06, T2R:R2B-02, T2R:R2C-02, M1-09, M1-10, M2-09, M3-09 and the per-run tables"),
    "premise": ("TBL-premise", "Premise check", premise, "M3-23"),
    "rl_vs_screen": ("TBL-rlscreen", "RL medians", screen_vs_rl, "M3-08"),
    "relu_fail": ("TBL-relufail", "The failed relu runs", relu_fail, "M3-08"),
    "relu_typical": ("TBL-relutyp", "relu against t1, typical run", relu_typical, "M3-08"),
    "t10": ("TBL-t10", "t10 against t1", t10_vs_t1, "M3-08"),
    "transmission": ("TBL-transmission", "Noise-landing transmission", transmission, "M2-11, M3-11"),
    "fd": ("TBL-fd", "Non-smoothing rounding width F_d (post hoc)", fd, "M3-08"),
    "stoprule": ("TBL-stoprule", "MS-R1 stop rule and polishing", stoprule, "M1-13"),
    "r0": ("TBL-r0", "R0 against the peak error", r0, "M3-13"),
    "arms_all": ("TBL-arms", "Global accuracy and stage 1 across arms", arms_all, "M1-08, M2-08, M3-08"),
    "stage1_ref": ("TBL-stage1", "Stage-1 error of v2.0", stage1_ref, "T2R:CF-03, M3-08"),
    "trajectory": ("TBL-trajectory", "Tie effort along local updates 1800-2800", trajectory, "M3-15"),
    "floor": ("TBL-floor", "Smoothing floor against the observed peak error", floor, "T2R:R2B-18, M3-08"),
    "floor_s": ("TBL-floors", "Smoothing floor at s = 1 and s = 16", floor_s, "M3-08"),
    "workload": ("TBL-workload", "Recorded workload of comparable waves", workload, "M1-18, M1-21, M2-17, M3-20, T2R:CF-03"),
    "gates": ("TBL-gates", "Gate failures in the MS rounds", gate_counts, "M1-08, M2-08, M3-08"),
    "eta": ("TBL-eta", "eta_2/DW in the t1 and t10 runs", eta_t1t10, "M3-08"),
    "budget": ("TBL-budget", "MS-R1 budget control", budget_control, "M1-09"),
    "compare": ("TBL-compare", "Path A against Path B", compare, "T2R:R2B-18, M3-08"),
}


def render_all(pack: Path) -> List[Tuple[str, str, str, str]]:
    out = []
    for name, (tid, title, fn, inputs) in BLOCKS.items():
        out.append((tid, title, fn(Path(pack)), inputs))
    return out


def inject(path: Path, pack: Path, verify: bool) -> int:
    text = path.read_text()
    bad = 0
    for name, (tid, title, fn, inputs) in BLOCKS.items():
        pat = re.compile(r"(<!-- TBL:%s -->\n)(.*?)(\n<!-- /TBL:%s -->)" % (re.escape(name), re.escape(name)), re.S)
        m = pat.search(text)
        if not m:
            continue
        new = fn(pack)
        if m.group(2) != new:
            bad += 1
            if verify:
                print("DIFF block %s" % name)
        text = text[:m.start(2)] + new + text[m.end(2):]
    if not verify:
        path.write_text(text)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--block")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--inject")
    ap.add_argument("--verify")
    ap.add_argument("--pack", default=str(PACK))
    a = ap.parse_args()
    pack = Path(a.pack)
    if a.list:
        for n, (tid, title, _, inputs) in BLOCKS.items():
            print("%-14s %-20s %s | inputs: %s" % (n, tid, title, inputs))
        return 0
    if a.block:
        print(BLOCKS[a.block][2](pack))
        return 0
    if a.all:
        for tid, title, text, inputs in render_all(pack):
            print("### %s %s\n%s\n" % (tid, title, text))
        return 0
    if a.inject:
        inject(Path(a.inject), pack, verify=False)
        return 0
    if a.verify:
        bad = inject(Path(a.verify), pack, verify=True)
        print("tables --verify: %s" % ("PASS" if not bad else "FAIL (%d blocks differ)" % bad))
        return 1 if bad else 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
