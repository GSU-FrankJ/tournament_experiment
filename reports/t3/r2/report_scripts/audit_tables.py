#!/usr/bin/env python3
"""Tables, figures and inline numbers of ``reports/t3/r2/01_audit.md``, generated from ``results/t3_r2/audit/``.

Reads the CSV / JSON files of ``tools/t3/t3r2_audit.py`` (A1-A3, A5-A8), the A1 re-run comparison, the stored T3-R1 tables
(``results/t3_r1/eval/per_run.csv``, ``results/t3_r1/analysis/bw_failures/*.csv``) and the pytest records of A4. Writes one Markdown
fragment per table into ``reports/t3/r2/tables/``, the figures into ``reports/t3/r2/figures/`` and ``tables/audit_numbers.json`` (every
number the prose quotes). ``assemble_audit.py`` substitutes them into the template. Nothing here is a statistic of its own: it selects,
orders, formats and computes maxima, medians and counts of the audit columns.

Usage:
    python reports/t3/r2/report_scripts/audit_tables.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import REP, RES, ROOT, md, num, write  # noqa: E402

import t3_blind as tb  # noqa: E402

AUD = RES / "audit"
R1 = ROOT / "results" / "t3_r1"
UNTRACKED = Path(os.environ.get("T3R1_UNTRACKED_ROOT", "/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-vs-jt-51e472/results/t3_r1"))
FIG = REP / "figures"
DW = 4.0
N: Dict[str, Any] = {}


def sci(x: float) -> str:
    """Scientific format with two digits."""
    return f"{float(x):.2e}"


# ---------------------------------------------------------------------------------- A1, A2, A3
def a1() -> None:
    c = pd.read_csv(AUD / "a1_compare.csv")
    e = pd.read_csv(AUD / "a1_rerun_exit.csv")
    rows = []
    for _, r in c.iterrows():
        ex = e[(e.q == r.q) & (e.seed == r.seed)].iloc[0]
        rows.append((int(r.q), int(r.seed), int(ex.returncode), f"{ex.wall_sec:.0f}", r.compare_runs_summary.replace(" | ", "<br>"),
                     int(r.metric_columns_compared), int(r.metric_columns_differing), "yes" if r.run_config_sha256_equal else "NO",
                     "**yes**" if r.bit_identical else "**NO**"))
    write("a01_reproduction", md(["q", "seed", "exit code", "wall s", "`compare_runs.py bw` (original against re-run)",
                                  "metric columns compared (`evaluate_run` against `per_run.csv`)", "columns differing",
                                  "`run_config.json` byte-equal", "bit-identical"], rows))
    N["a1_all_identical"] = bool(c.bit_identical.all())
    N["a1_n"] = int(len(c))
    N["a1_cols"] = int(c.metric_columns_compared.iloc[0])
    N["a1_wall_max"] = float(e.wall_sec.max())


def a2() -> None:
    d = pd.read_csv(AUD / "a2_snapshot_identity.csv")
    rows = []
    for q in (50, 60):
        s = d[d.q == q]
        rows.append((q, len(s), int(s.frozen3_eq_u01600.sum()), int(s.frozen2_eq_u03200.sum()), int(s.live_eq_u03800.sum()),
                     int(s.all_identical.sum())))
    rows.append(("all", len(d), int(d.frozen3_eq_u01600.sum()), int(d.frozen2_eq_u03200.sum()), int(d.live_eq_u03800.sum()),
                 int(d.all_identical.sum())))
    write("a02_snapshot_identity", md(["q", "runs", "`frozen_stages[\"3\"]` = `u01600`", "`frozen_stages[\"2\"]` = `u03200`",
                                       "live actor = `u03800`", "all three identical"], rows))
    N["a2_identical"], N["a2_n"] = int(d.all_identical.sum()), int(len(d))


def a3() -> None:
    d = pd.read_csv(AUD / "a3_tables.csv")
    rows = [(int(r.q), int(r.seed), "failing" if (r.q, r.seed) in {(50, 50504), (50, 50517), (60, 50514), (60, 50517)} else "passing",
             "yes" if r.table2_bit_identical else "NO", "yes" if r.table1_bit_identical else "NO", sci(r.table2_max_diff_over_dw),
             sci(r.table1_max_diff_over_dw), "no" if not (r.table2_flag or r.table1_flag) else "FLAG") for r in d.itertuples()]
    write("a03_tables", md(["q", "seed", "run", "stage-2 table rebuilt = stored (bit)", "stage-1 table rebuilt = stored (bit)",
                            "max abs diff to the calculator, stage-2 table (/dW)", "same, stage-1 table (/dW)", "flag (> 1e-5 dW)"], rows))
    N["a3_bit"] = int((d.table2_bit_identical & d.table1_bit_identical).sum())
    N["a3_n"] = int(len(d))
    N["a3_maxdiff"] = float(max(d.table2_max_diff_over_dw.max(), d.table1_max_diff_over_dw.max()))


# ---------------------------------------------------------------------------------- A5
def a5() -> None:
    f = pd.read_csv(AUD / "a5_failure_points.csv")
    rows = []
    for r in f.itertuples():
        rows.append((int(r.q), int(r.seed), int(r.stage), f"{r.d:+.0f}", r.quantity, num(r.stored_Delta, 6), num(r.calc_Delta_over_dw, 6),
                     num(r.stored_G, 6), num(r.calc_G_over_dw, 6), sci(r.Delta_diff_over_dw), sci(r.G_diff_over_dw),
                     num(r.stored_e, 2), num(r.calc_e_hat, 2), num(r.stored_br, 2), num(r.calc_a_br, 2), num(r.stored_dev, 2),
                     num(r.calc_a_dev, 2), num(r.calc_slope, 5)))
    write("a05a_failure_points", md(["q", "seed", "stage", "gap d", "failing quantity", "Delta stored (/dW)", "Delta calc", "G stored", "G calc",
                                     "abs diff Delta", "abs diff G", "learned effort stored", "calc", "best-response effort stored",
                                     "calc (dynamic)", "one-step argmax stored", "calc", "slope dQ/de at the learned effort"], rows))
    N["a5_fail_maxdiff"] = float(max(f.Delta_diff_over_dw.max(), f.G_diff_over_dw.max()))
    N["a5_fail_br_maxdiff"] = float(f.br_effort_diff.max())

    r = pd.read_csv(AUD / "a5_runs.csv")
    rows = []
    for arm in ("bw", "jt", "jt_es", "jt2", "v20"):
        s = r[r.arm == arm]
        rows.append((f"`{arm}`", len(s), sci(s.G_diff_over_dw.max()), sci(s.G_diff_over_dw.median()), sci(s.DeltaT_diff_over_dw.max()),
                     sci(s.EXP_diff_over_dw.max()), sci(s.EXP_diff_over_dw.median()),
                     int((s[["G_diff_over_dw", "DeltaT_diff_over_dw", "EXP_diff_over_dw"]].max(axis=1) > 1e-4).sum()),
                     int((s[["G_diff_over_dw", "DeltaT_diff_over_dw", "EXP_diff_over_dw"]].max(axis=1) > 1e-3).sum())))
    s = r
    rows.append(("all", len(s), sci(s.G_diff_over_dw.max()), sci(s.G_diff_over_dw.median()), sci(s.DeltaT_diff_over_dw.max()),
                 sci(s.EXP_diff_over_dw.max()), sci(s.EXP_diff_over_dw.median()),
                 int((s[["G_diff_over_dw", "DeltaT_diff_over_dw", "EXP_diff_over_dw"]].max(axis=1) > 1e-4).sum()),
                 int((s[["G_diff_over_dw", "DeltaT_diff_over_dw", "EXP_diff_over_dw"]].max(axis=1) > 1e-3).sum())))
    write("a05b_runs", md(["arm", "runs", "max abs diff G at the stored state (/dW)", "median", "max abs diff Delta_T at the stored state",
                           "max abs diff EXP_root", "median", "runs with a difference > 1e-4 dW", "> 1e-3 dW"], rows))
    for k, c in (("G", "G_diff_over_dw"), ("DeltaT", "DeltaT_diff_over_dw"), ("EXP", "EXP_diff_over_dw")):
        N[f"a5_{k}_max"], N[f"a5_{k}_median"] = float(r[c].max()), float(r[c].median())
        i = r[c].idxmax()
        N[f"a5_{k}_argmax"] = f"{r.loc[i, 'arm']} q={int(r.loc[i, 'q'])} seed {int(r.loc[i, 'seed'])}"
    N["a5_n"] = int(len(r))
    N["a5_over_1e3"] = int((r[["G_diff_over_dw", "DeltaT_diff_over_dw", "EXP_diff_over_dw"]].max(axis=1) > 1e-3).sum())
    N["a5_over_1e4"] = int((r[["G_diff_over_dw", "DeltaT_diff_over_dw", "EXP_diff_over_dw"]].max(axis=1) > 1e-4).sum())
    N["a5_sec_max"] = float(r.sec.max())
    N["a5_sec_median"] = float(r.sec.median())

    g = pd.read_csv(AUD / "a5_verifier_agreement.csv")
    rows = [(f"`{x.arm}`", int(x.q), int(x.seed), x.verifier_arrays, num(x.V1_mean_stored, 6), num(x.V1_mean_calc, 6), sci(x.max_diff_over_dw_t1),
             sci(x.max_diff_over_dw_t2), "-" if pd.isna(x.max_diff_over_dw_t3) else sci(x.max_diff_over_dw_t3), sci(x.max_diff_over_dw),
             "no" if not x.exceeds_1e_4 else "YES") for x in g.rename(columns={"exceeds_1e-4": "exceeds_1e_4"}).itertuples()]
    write("a05c_verifier_agreement", md(["arm", "q", "seed", "verifier arrays", "V_1^mean(0) verifier", "calculator", "max abs diff stage 1 (/dW)",
                                         "stage 2 (20 states)", "stage 3 (20 states)", "max over stages", "> 1e-4 dW"], rows))
    N["a5_agree_max"] = float(g.max_diff_over_dw.max())

    cv = json.loads((AUD / "a5_convergence.json").read_text())
    rows = [(x["label"].replace("_", " "), f"`{x['run']}`", len(x["rows"]), sci(x["max_value_change_over_dw"]),
             "yes" if x["ok_values"] else "NO", f"{x['max_effort_change']:.4f}", "yes" if x["ok_efforts"] else "NO") for x in cv]
    write("a05d_convergence", md(["test set", "run", "points", "max change of Delta, G, V (/dW) when every step is halved", "<= 1e-6",
                                  "max change of the efforts", "<= 1e-3"], rows))
    N["a5_conv_val_max"] = float(max(x["max_value_change_over_dw"] for x in cv))
    N["a5_conv_eff_max"] = float(max(x["max_effort_change"] for x in cv))
    N["a5_conv_ok"] = bool(all(x["ok_values"] and x["ok_efforts"] for x in cv))
    N["a5_conv_n"] = int(len(cv))


def flips() -> None:
    """Verdict flips under the calculator values and under the refined tier (gates of every run, then H1, H2, H3, H1-T2)."""
    r = pd.read_csv(AUD / "a5_runs.csv")
    a6 = pd.read_csv(AUD / "a6_refined.csv")
    per = pd.read_csv(R1 / "eval" / "per_run.csv")
    crit = json.loads((ROOT / "reports" / "t3" / "r1" / "prereg_criteria.json").read_text())
    lim_eta, lim_g = crit["gates"]["G-A"]["eta_T_over_dw"], crit["gates"]["G-F"]["Gmax_full_over_dw"]
    flips_rows = []
    for x in r.itertuples():
        s_eta, c_eta = x.eta_T_stored_over_dw <= lim_eta, x.DeltaT_calc_over_dw <= lim_eta
        s_g, c_g = x.G_stored_over_dw <= lim_g, x.G_calc_over_dw <= lim_g
        for name, s, c, val, lim in (("G-A eta", s_eta, c_eta, x.DeltaT_calc_over_dw, lim_eta), ("G-F", s_g, c_g, x.G_calc_over_dw, lim_g)):
            if s != c:
                flips_rows.append((f"`{x.arm}`", int(x.q), int(x.seed), name, "calculator at the stored state", num(val, 7), lim,
                                   "(N)" if abs(val - lim) <= 1e-4 else "candidate (E)"))
    for x in a6.itertuples():
        if x.verdict_differs:
            flips_rows.append((f"`{x.arm}`", int(x.q), int(x.seed), "G-A or G-F", "refined tier", "-", "-", "see A6"))
    write("a05e_gate_flips", md(["arm", "q", "seed", "gate", "source", "value", "limit", "class"], flips_rows) if flips_rows else "none\n")
    N["gate_flips"] = len(flips_rows)
    # hypotheses: stored, calculator-at-the-stored-state, refined tier
    rows = []
    variants = {}
    with tempfile.TemporaryDirectory() as td:
        for name, col, src in (("stored (T3-R1)", None, None), ("calculator at the stored state", "G_calc_over_dw", r),
                               ("refined verifier tier", "refined_Gmax_full_over_dw", a6)):
            d = per.copy()
            if col is not None:
                m = {(a, int(q), int(s)): v for a, q, s, v in zip(src.arm, src.q, src.seed, src[col])}
                d["Gmax_full_over_dw"] = [m.get((a, int(q), int(s)), g0) for a, q, s, g0 in zip(d.arm, d.q, d.seed, d.Gmax_full_over_dw)]
            p = Path(td) / f"{abs(hash(name))}.csv"
            d.to_csv(p, index=False)
            variants[name] = tb.criterion_rows(tb.read_table(p), crit)
    base = {(x["hypothesis"], x["q"]): x for x in variants["stored (T3-R1)"]}
    nflip = 0
    for name, rr in variants.items():
        for x in rr:
            b = base[(x["hypothesis"], x["q"])]
            flip = bool(x["ci_hi_below_0"]) != bool(b["ci_hi_below_0"]) or bool(x["hypothesis_supported"]) != bool(b["hypothesis_supported"])
            nflip += int(flip and name != "stored (T3-R1)")
            rows.append((name, x["hypothesis"], x["q"], x["n_pairs"], num(x["mean_diff"], 6), num(x["ci95_hi"], 6),
                         "yes" if x["ci_hi_below_0"] else "no", "yes" if x["hypothesis_supported"] else "no", "FLIP" if flip and name != "stored (T3-R1)" else ""))
    write("a05f_hypotheses", md(["Gmax_full source", "hypothesis", "q", "pairs", "mean paired difference", "upper end of the 95 % interval",
                                 "upper end < 0", "hypothesis supported", "flip"], rows))
    N["hyp_flips"] = nflip
    N["hyp_supported_stored"] = sorted({h for (h, q), x in base.items() if x["hypothesis_supported"]})


# ---------------------------------------------------------------------------------- A6
def a6() -> None:
    d = pd.read_csv(AUD / "a6_refined.csv")
    cols = ["Gmax_full_over_dw", "eta_T_over_dw", "EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw", "G_max_t1_over_dw",
            "G_max_t2_over_dw", "G_max_t3_over_dw", "Delta_max_t1_over_dw", "Delta_max_t2_over_dw", "Delta_max_t3_over_dw"]
    rows = []
    for arm in ("bw", "jt", "jt_es", "jt2", "v20"):
        s = d[d.arm == arm]
        rows.append((f"`{arm}`", len(s), *[sci(s[f"absdiff_{c}"].max()) if f"absdiff_{c}" in s and s[f"absdiff_{c}"].notna().any() else "-" for c in cols[:5]],
                     int(s.verdict_differs.sum())))
    rows.append(("all", len(d), *[sci(d[f"absdiff_{c}"].max()) for c in cols[:5]], int(d.verdict_differs.sum())))
    write("a06a_refined", md(["arm", "runs", "max abs diff Gmax_full (/dW)", "eta_T", "EXP_root", "dReach", "Deltamax_all", "G-A or G-F verdict differs"], rows))
    st = []
    for c in cols[5:]:
        if f"absdiff_{c}" in d and d[f"absdiff_{c}"].notna().any():
            st.append((c.replace("_over_dw", ""), int(d[f"absdiff_{c}"].notna().sum()), sci(d[f"absdiff_{c}"].max())))
    write("a06b_refined_stages", md(["per-stage maximum", "runs", "max abs diff refined - final (/dW)"], st))
    top = d.sort_values("absdiff_Gmax_full_over_dw", ascending=False).head(6)
    write("a06c_refined_top", md(["arm", "q", "seed", "Gmax_full final", "Gmax_full refined", "abs diff", "G-A final / refined", "G-F final / refined"],
                                 [(f"`{x.arm}`", int(x.q), int(x.seed), num(x.final_Gmax_full_over_dw, 6), num(x.refined_Gmax_full_over_dw, 6),
                                   sci(x.absdiff_Gmax_full_over_dw), f"{x.final_G_A} / {x.refined_G_A}", f"{x.final_G_F} / {x.refined_G_F}") for x in top.itertuples()]))
    N["a6_flips"] = int(d.verdict_differs.sum())
    N["a6_valid"] = int(d.refined_valid.sum())
    N["a6_G_max"], N["a6_eta_max"], N["a6_exp_max"] = (float(d[f"absdiff_{c}"].max()) for c in cols[:3])
    N["a6_G_argmax"] = f"{top.iloc[0].arm} q={int(top.iloc[0].q)} seed {int(top.iloc[0].seed)}"
    N["a6_n"] = int(len(d))
    N["a6_sec_max"] = float(d.sec.max())
    N["a6_over_1e3"] = int((d[[f"absdiff_{c}" for c in cols[:5]]].max(axis=1) > 1e-3).sum())


# ---------------------------------------------------------------------------------- A7
def a7() -> None:
    runs = json.loads((AUD / "a7_runs.json").read_text())
    ser = pd.read_csv(AUD / "a7_series.csv")
    ze = pd.read_csv(R1 / "analysis" / "bw_failures" / "zero_effort.csv")
    rows, rows2 = [], []
    for x in (y for y in runs if y["group"] == "stretch"):
        s, f = x["summary"], x["formation"]
        fb = [a["first_below1_update"] for a in f]
        stay = [a["stays_below1_from_update"] for a in f]
        fin = [o for o in x["objective"] if o["export"] == "final"]
        last = ser[(ser.q == x["q"]) & (ser.seed == x["seed"]) & (ser["update"] == 3200)]
        w = x["windows"]
        passed = bool(ze[(ze.q == x["q"]) & (ze.seed == x["seed"])].passed.iloc[0])
        rows.append((int(x["q"]), int(x["seed"]), "pass" if passed else "**fail**", s["n_nodes"], ", ".join(f"[{a:+.0f}, {b:+.0f}]" for a, b in s["stretches"]),
                     f"{min(fb)} - {max(fb)}", f"{min(stay)} - {max(stay)}", s["n_policy_error"], s["n_equilibrium_like"],
                     f"{s['n_positive_slope_final']} of {s['n_nodes']}", f"{s['n_local_max_near_zero_final']} of {s['n_nodes']}",
                     f"{int((last.alpha < 1).sum())} of {len(last)}"))
        rows2.append((int(x["q"]), int(x["seed"]), num(min(o["slope_at_e_hat"] for o in fin), 5), num(min(o["a_dev"] for o in fin), 2), num(max(o["a_dev"] for o in fin), 2),
                      num(max(o["Delta_over_dw"] for o in fin), 5), num(max(o["G_over_dw"] for o in fin), 5),
                      f"{w['decay']['share_below_clamp']:.3f}", f"{w['formation']['share_below_clamp']:.3f}" if w["formation"] else "-",
                      f"{w['decay']['kl_final_epoch_mean']:.4f}", f"{w['decay']['clip_frac_mean']:.3f}", f"{w['decay']['adv_raw_std_mean']:.4f}"))
    write("a07a_stretches", md(["q", "seed", "run", "stretch nodes", "stretches (d)", "first export below 1 (update, range over nodes)",
                                "export from which the node stays below 1", "policy error (one-step argmax >= 1)", "equilibrium-like (< 1)",
                                "Q increasing at the learned effort", "local maximum of Q at or near e = 0", "alpha < 1 at the final export"], rows))
    write("a07b_stretch_objective", md(["q", "seed", "smallest slope dQ/de at the learned effort", "smallest one-step argmax", "largest",
                                        "largest Delta (/dW) at a stretch node", "largest G", "share of stage-2 learner draws below the clamp, local 1201-1600",
                                        "same, formation window", "kl_final_epoch (mean, decay window)", "clip_frac", "adv_raw_std"], rows2))
    # the other runs
    comp = [y for y in runs if y["group"] == "comparison"]
    rows3 = []
    for q in (50, 60):
        c = [y for y in comp if y["q"] == q]
        sh = [y["windows"]["decay"]["share_below_clamp"] for y in c]
        st = [y for y in runs if y["group"] == "stretch" and y["q"] == q]
        sh_s = [y["windows"]["decay"]["share_below_clamp"] for y in st]
        n_nodes = len(c[0]["nodes"])
        pe = [y["summary"]["n_policy_error"] for y in c]
        eq = [y["summary"]["n_equilibrium_like"] for y in c]
        ever = []
        for y in c:
            e_mat = np.load(AUD / "a7_emat.npz")[f"q{y['q']}_s{y['seed']}"]
            grid = bc_grid(y["q"])
            idx = [int(np.argmin(np.abs(grid - d))) for d in y["nodes"]]
            ever.append(int(((e_mat[:, idx] < 1.0).any(axis=0)).sum()))
        rows3.append((q, len(c), n_nodes, f"{min(sh):.4f} - {max(sh):.4f}", f"{np.median(sh):.4f}", f"{min(sh_s):.3f} - {max(sh_s):.3f}",
                      f"{min(pe)} - {max(pe)}", f"{min(eq)} - {max(eq)}", f"{min(ever)} - {max(ever)}"))
    write("a07c_comparison", md(["q", "runs without a stretch", "nodes (union of the stretch nodes of the q)", "share of learner draws below the clamp, decay window (range)",
                                 "median", "the stretch runs of the q (range)", "of those nodes: policy error (range)", "equilibrium-like (range)",
                                 "nodes that were below 1 at some export (range)"], rows3))
    # training signals of the decay window: stretch runs against the others
    rows5 = []
    for q in (50, 60):
        for grp, label in (("stretch", "runs with a stretch"), ("comparison", "runs without a stretch")):
            ws = [y["windows"]["decay"] for y in runs if y["group"] == grp and y["q"] == q]
            rows5.append((q, label, len(ws), *[f"{min(w[k] for w in ws):.4f} - {max(w[k] for w in ws):.4f}" for k in
                                                ("kl_final_epoch_mean", "clip_frac_mean", "adv_raw_std_mean", "share_below_clamp")]))
    write("a07g_training_signal", md(["q", "group", "runs", "kl_final_epoch (mean over local 1201-1600)", "clip_frac", "adv_raw_std",
                                      "share of learner draws below the clamp"], rows5))
    # placement (d) of A5: the calculator at every stretch node against the stored final-tier arrays of the evaluated iterate
    rows4 = []
    for x in (y for y in runs if y["group"] == "stretch"):
        arr = np.load(UNTRACKED / "eval" / "arrays" / f"t3r1_q{x['q']}_s{x['seed']}_bw.npz")
        g, dl, gg = arr["fin_t2_d_grid"], arr["fin_t2_delta"], arr["fin_t2_G"]
        fin = [o for o in x["objective"] if o["export"] == "final"]
        dD = [abs(o["Delta_over_dw"] - float(dl[int(np.argmin(np.abs(g - o["d"])))]) / DW) for o in fin]
        dG = [abs(o["G_over_dw"] - float(gg[int(np.argmin(np.abs(g - o["d"])))]) / DW) for o in fin]
        rows4.append((int(x["q"]), int(x["seed"]), len(fin), sci(max(dD)), sci(max(dG))))
    write("a07e_stretch_vs_stored", md(["q", "seed", "stretch nodes", "max abs diff Delta (calculator - stored final tier, /dW)", "max abs diff G"], rows4))
    N["a7_stored_maxdiff"] = float(max(max(float(r[3]), float(r[4])) for r in rows4))
    N["a7_nodes_total"] = int(sum(y["summary"]["n_nodes"] for y in runs if y["group"] == "stretch"))
    N["a7_policy_error"] = int(sum(y["summary"]["n_policy_error"] for y in runs if y["group"] == "stretch"))
    N["a7_eq_like"] = int(sum(y["summary"]["n_equilibrium_like"] for y in runs if y["group"] == "stretch"))
    N["a7_locmax0"] = int(sum(y["summary"]["n_local_max_near_zero_final"] for y in runs if y["group"] == "stretch"))
    N["a7_posslope"] = int(sum(y["summary"]["n_positive_slope_final"] for y in runs if y["group"] == "stretch"))
    N["a7_first_export"] = int(sum(1 for y in runs if y["group"] == "stretch" and max(a["first_below1_update"] for a in y["formation"]) == 1625))
    late = [y for y in runs if y["group"] == "stretch" and max(a["first_below1_update"] for a in y["formation"]) > 1625]
    N["a7_late_n"] = len(late)
    N["a7_late_lo"] = int(min(a["first_below1_update"] for y in late for a in y["formation"]))
    N["a7_late_hi"] = int(max(a["first_below1_update"] for y in late for a in y["formation"]))
    # the two q = 60 failures: the failure node
    keyrows = []
    for (q, s, d0) in ((60, 50514, -14.0), (60, 50517, -66.0)):
        x = [y for y in runs if y["q"] == q and y["seed"] == s][0]
        for o in x["objective"]:
            if o["d"] == d0:
                keyrows.append((q, s, o["export"], o["update"], f"{o['d']:+.0f}", num(o["e_hat"], 3), num(o["slope_at_e_hat"], 5), num(o["a_dev"], 2),
                                num(o["a_br"], 2), num(o["Delta_over_dw"], 5), num(o["G_over_dw"], 5), num(o["gain_to_grid_max_over_dw"], 5),
                                "yes" if o["local_max_near_zero"] else "no", o["local_maxima_efforts"], o["class"].replace("_", " ")))
    write("a07d_failure_nodes", md(["q", "seed", "export", "update", "gap d", "learned mean effort", "slope dQ/de there", "one-step argmax", "dynamic best response",
                                    "Delta (/dW)", "G (/dW)", "Q at the grid maximum minus Q at the learned effort (/dW)", "local max near e = 0", "efforts of the local maxima", "class"], keyrows))
    # figure
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    em = np.load(AUD / "a7_emat.npz")
    upd = em["updates"]
    for ax, x in zip(axes.ravel(), (y for y in runs if y["group"] == "stretch")):
        grid = bc_grid(x["q"])
        e = em[f"q{x['q']}_s{x['seed']}"]
        band = np.abs(grid) < 2 * x["q"] - 1e-9
        ax.pcolormesh(upd, grid[band], np.minimum(e[:, band].T, 20.0), shading="auto", cmap="viridis", vmin=0, vmax=20)
        ax.contour(upd, grid[band], (e[:, band].T < 1.0).astype(float), levels=[0.5], colors="red", linewidths=0.8)
        ax.set_title(f"q={x['q']} seed {x['seed']} ({len(x['nodes'])} nodes)", fontsize=9)
        ax.set_xlabel("update")
        ax.set_ylabel("stage-2 gap d")
    axes.ravel()[-1].axis("off")
    fig.suptitle("learned stage-2 mean effort (colour, capped at 20) in the band |d| < 2q against the update; red: effort below 1", fontsize=10)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / "f01_a7_formation.png", dpi=110)
    plt.close(fig)


def bc_grid(q: float) -> np.ndarray:
    """The final-tier D_2 grid (step 2, B = 100 + 2q): ``arange(-n, n + 1) * (B / n)``."""
    B = 100.0 + 2.0 * q
    n = int(np.ceil(B / 2.0 - 1e-9))
    return np.arange(-n, n + 1) * (B / n)


# ---------------------------------------------------------------------------------- A8
def a8() -> None:
    a = pd.read_csv(AUD / "a8_last16.csv")
    per = pd.read_csv(R1 / "eval" / "per_run.csv")
    per = per[per.arm == "bw"].set_index(["q", "seed"])
    s = a.groupby(["q", "seed"]).agg(eta_min=("eta_3_over_dw", "min"), eta_med=("eta_3_over_dw", "median"), eta_max=("eta_3_over_dw", "max"),
                                     rmse_max=("rmse_pos_T_over_eT0", "max"), tail_max=("tail_mean_T_over_eT0", "max"))
    s["n_above"] = a.assign(x=a.eta_3_over_dw > 0.005).groupby(["q", "seed"]).x.sum()
    s["eta_final"] = per.eta_T_over_dw
    rows = []
    for (q, sd), r in s.iterrows():
        mark = " (**fails G-A**)" if (q, sd) in {(50, 50504), (50, 50517)} else ""
        rows.append((q, f"{sd}{mark}", num(r.eta_min, 5), num(r.eta_med, 5), num(r.eta_max, 5), num(r.eta_final, 5), int(r.n_above), num(r.rmse_max, 4), num(r.tail_max, 4)))
    write("a08a_eta3_per_run", md(["q", "seed", "min eta_3/dW over the last 16 exports", "median", "max", "eta_3/dW of the evaluated iterate (u01600)",
                                   "exports above 0.005", "max RMSE_pos/e_3(0)*", "max tail mean/e_3(0)*"], rows))
    rows = []
    for q in (50, 60):
        t = s.loc[q]
        rows.append((q, len(t), int((t.eta_max > 0.005).sum()), int((t.eta_min > 0.005).sum()), int((t.eta_final > 0.005).sum()),
                     f"{t.eta_med.median():.5f}", int((t.rmse_max > 0.05).sum()), int((t.tail_max > 0.02).sum())))
    write("a08b_eta3_counts", md(["q", "runs", "runs with eta_3/dW > 0.005 at one or more of the 16 exports", "at all 16 exports", "at the evaluated iterate (u01600)",
                                  "median over runs of the per-run median", "runs with RMSE_pos > 0.05 at some export", "runs with tail mean > 0.02 at some export"], rows))
    for q in (50, 60):
        t = s.loc[q]
        N[f"a8_q{q}_any"], N[f"a8_q{q}_all"], N[f"a8_q{q}_final"] = int((t.eta_max > 0.005).sum()), int((t.eta_min > 0.005).sum()), int((t.eta_final > 0.005).sum())
    N["a8_consistent"] = float((a[a["update"] == 1600].set_index(["q", "seed"]).eta_3_over_dw - per.eta_T_over_dw).abs().max())
    for sd in (50504, 50517):
        t = s.loc[(50, sd)]
        N[f"a8_{sd}_min"], N[f"a8_{sd}_med"], N[f"a8_{sd}_max"], N[f"a8_{sd}_n"] = float(t.eta_min), float(t.eta_med), float(t.eta_max), int(t.n_above)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, q in zip(axes, (50, 60)):
        for sd in sorted(a[a.q == q].seed.unique()):
            t = a[(a.q == q) & (a.seed == sd)].sort_values("update")
            hot = (q, sd) in {(50, 50504), (50, 50517)}
            ax.plot(t["update"], t.eta_3_over_dw, color="C3" if hot else "0.7", lw=1.6 if hot else 0.8, zorder=3 if hot else 1)
        ax.axhline(0.005, color="k", ls="--", lw=0.8)
        ax.set_title(f"q = {q}")
        ax.set_xlabel("stage-3 export (update)")
    axes[0].set_ylabel("eta_3 / dW (final tier)")
    axes[0].set_ylim(0, 0.03)
    fig.tight_layout()
    fig.savefig(FIG / "f02_a8_eta3.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------------- A4 (pytest record)
def a4() -> None:
    p = RES / "preflight" / "tests" / "semantics.txt"
    if p.is_file():
        N["a4_summary"] = p.read_text().strip().splitlines()[-1]


def costs() -> None:
    d = pd.read_csv(AUD / "costs.csv")
    rows = [(f"`{r.step}`", int(r.items), int(r.workers), f"{r.wall_sec:.1f}") for r in d.itertuples()]
    write("a00_costs", md(["step", "items", "workers", "wall seconds"], rows))
    N["cost_total_sec"] = float(d.wall_sec.sum())


def main() -> int:
    """Write every fragment, the figures and the numbers file."""
    for fn in (costs, a1, a2, a3, a5, a6, flips, a7, a8, a4):
        fn()
    (REP / "tables").mkdir(parents=True, exist_ok=True)
    (REP / "tables" / "audit_numbers.json").write_text(json.dumps(N, indent=1, default=str))
    print(json.dumps(N, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
