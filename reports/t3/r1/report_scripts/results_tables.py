#!/usr/bin/env python3
"""T3-R1 P3: every table of the results reports, generated from the CSVs of ``results/t3_r1/``.

Reads ``analysis/per_run_table.csv`` (the evaluation table with the gate verdicts), ``analysis/*.csv``, ``launch_checks.json`` and ``launch_*.json`` and writes one Markdown fragment per
table into ``reports/t3/r1/tables/``. ``assemble_reports.py`` substitutes the fragments into the report templates. Nothing here is
a statistic of its own: the paired statistics, the gate counts and their intervals come from ``tools/t3/t3_analysis.py``; this
script selects, orders and formats, and computes only descriptive medians, maxima and counts of columns of the per-run table.

Usage:
    python reports/t3/r1/report_scripts/results_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
RES = ROOT / "results" / "t3_r1"
OUT = ROOT / "reports" / "t3" / "r1" / "tables"
sys.path.insert(0, str(ROOT / "tools" / "t3"))
sys.path.insert(0, str(ROOT))

import t3_configs as tc  # noqa: E402

ARMS3 = ("bw", "jt", "jt_es")
ARMS2 = ("v20", "jt2")
QS = (50, 60)
DASH = "-"


def num(x: object, nd: int = 5, sci: bool = False) -> str:
    """Format a number (NaN and None become a dash)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return DASH
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if sci:
        return f"{float(x):.2e}"
    return f"{float(x):.{nd}f}"


def signed(x: object, nd: int = 5) -> str:
    """Format a signed number."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return DASH
    return f"{float(x):+.{nd}f}"


def md(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Render a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines) + "\n"


def write(name: str, text: str) -> None:
    """Write one fragment."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.md").write_text(text)


def load() -> Dict[str, object]:
    """Load the CSVs and JSONs the tables are made from."""
    per = pd.read_csv(RES / "analysis" / "per_run_table.csv").copy()
    per["gates_run_pass"] = per["passed"].astype(bool)
    d: Dict[str, object] = {
        "per": per,
        "crit": pd.read_csv(RES / "analysis" / "criterion.csv"),
        "sec": pd.read_csv(RES / "analysis" / "secondary.csv"),
        "gate": pd.read_csv(RES / "analysis" / "gate_counts.csv"),
        "bwg": pd.read_csv(RES / "analysis" / "bw_t3_gates.csv"),
        "best": pd.read_csv(RES / "analysis" / "best_of_run.csv"),
        "cov": pd.read_csv(RES / "analysis" / "coverage.csv"),
        "cost": pd.read_csv(RES / "analysis" / "cost.csv"),
        "bud": pd.read_csv(RES / "analysis" / "budget_check.csv"),
        "chk": json.loads((RES / "launch_checks.json").read_text()),
        "head": json.loads((RES / "analysis" / "headline.json").read_text()),
    }
    rec = sorted(RES.glob("launch_t3_t2j_*.json"))
    d["launch"] = json.loads(rec[-1].read_text())
    return d


def sel(per: pd.DataFrame, arm: str, q: int) -> pd.DataFrame:
    """The rows of one arm and q, ordered by seed."""
    return per[(per["arm"] == arm) & (per["q"] == q)].sort_values("seed")


# ------------------------------------------------------------------------------------------------ checks and arms
def t01_checks(d: Dict[str, object]) -> None:
    """The post-launch verdicts and the launch facts."""
    chk, ln = d["chk"], d["launch"]
    rows = [(k, v["pass"], v["fail"], v["not_evaluated"]) for k, v in chk["verdicts"].items()]
    t = md(["check", "pass", "fail", "not evaluated"], rows)
    t += "\n"
    t += md(["quantity", "value"], [
        ("runs", chk["n_runs"]),
        ("C-INIT within (q, seed) of wave t3: groups / equal", f"{chk['c_init_within_q_seed']['groups']} / {chk['c_init_within_q_seed']['equal']}"),
        ("C-INIT against the planned digests: runs / equal", f"{chk['c_init_equals_plan']['runs']} / {chk['c_init_equals_plan']['equal']}"),
        ("start-share cells in all (descriptive)", chk["start_share_flags"]["cells"]),
        ("flags (abs z > 3)", chk["start_share_flags"]["flags"]),
        ("flags expected under the null (0.0027 per cell)", f"{chk['start_share_flags']['expected_under_null']:.2f}"),
    ])
    write("t01_checks", t)
    per = pd.DataFrame(chk["runs"])
    rows = []
    for arm in ("bw", "jt", "jt_es", "jt2"):
        for q in QS:
            m = per[(per["arm"] == arm) & (per["q"] == q)]
            cells, flags = int(m["z_cells"].sum()), int(m["z_flags"].sum())
            rows.append((f"`{arm}`", q, len(m), cells, flags, f"{cells * 0.0027:.2f}"))
    rows.append(("all", "", len(per), int(per["z_cells"].sum()), int(per["z_flags"].sum()), f"{per['z_cells'].sum() * 0.0027:.2f}"))
    write("t01d_start_flags", md(["arm", "q", "runs", "z cells", "flags (abs z > 3)", "expected under the null (0.0027 per cell)"], rows))
    st = str(ln["started"])
    started = f"{st[:4]}-{st[4:6]}-{st[6:8]} {st[9:11]}:{st[11:13]}:{st[13:15]}" if "_" in st and len(st) == 15 else st
    bud = d["bud"]
    write("t01b_budget", md(["arm", "runs", "updates (expected)", "updates equal", "rows total (expected)", "rows total equal",
                              "rows by stage equal", "minibatch steps (expected)", "steps equal"],
                             [(r.arm, r.n_runs, num(int(r.updates_expected)), r.updates_equal, num(int(r.rows_total_expected)), r.rows_total_equal,
                               r.rows_by_stage_equal, num(int(r.minibatch_steps_expected)), r.minibatch_steps_equal) for r in bud.itertuples()]))
    write("t01c_launch", md(["quantity", "value"], [
        ("launch record", "`results/t3_r1/launch_t3_t2j_20261017_010337.json`"),
        ("started (UTC)", started), ("HEAD at the launch", f"`{ln['head']}`"), ("code commit", f"`{ln['code_commit']}`"),
        ("workers / nproc", f"{ln['workers']} / {ln['nproc']}"),
        ("load average at start", ", ".join(f"{x:.1f}" for x in ln["loadavg_at_start"])),
        ("load average at end", ", ".join(f"{x:.1f}" for x in ln["loadavg_at_end"])),
        ("free disk at start", f"{ln['disk_free_bytes'] / 1e9:.0f} GB of {ln['disk_total_bytes'] / 1e9:.0f} GB"),
        ("runs launched / exit code 0", f"{len(ln['runs'])} / {sum(1 for r in ln['runs'] if r['returncode'] == 0)}"),
        ("criteria file SHA-256", "`b586a5e39b1fad503282b1bb903696ad7d47c52e75fb9ff6e47f06781ad88332`"),
    ]))


def t02_arms(d: Dict[str, object]) -> None:
    """The arms with their configuration."""
    rows = []
    desc = {
        "bw": ("3", "backward induction (existing `run/run_ms_stagewise.py`, T = 3)", "stage 3 alone, then stage 2 (stage 3 frozen), then stage 1 (stages 2, 3 frozen)",
               "three phases: 1,600 / 1,600 / 600 updates, 512 episodes each"),
        "jt": ("3", "joint whole-game training (`run/run_joint.py`)", "one network for all stages; starts at stage 1 only (root play)", "512 episodes per update"),
        "jt_es": ("3", "joint whole-game training with exploring starts (`run/run_joint.py`)", "one network for all stages; starts 256 root + 85 stage-2 + 171 stage-3 per update", "512 episodes per update"),
        "v20": ("2", "recorded v2.0 confirmation runs, re-evaluated", "two-phase backward induction of protocol v2.0", "Phase A 1,600 + Phase B 600 updates"),
        "jt2": ("2", "joint whole-game training at T = 2 (`run/run_joint.py`)", "one network for both stages; root play only", "512 episodes per update"),
    }
    bud = d["bud"].set_index("arm")
    for arm in ("bw", "jt", "jt_es", "v20", "jt2"):
        b = tc.expected_budget(arm)
        win = ""
        if arm in tc.JOINT_BUDGETS:
            first, last = tc.decay_window(int(b["updates"]))
            win = f"3e-4 up to {first - 1}, linear to 3e-5 over {first}-{last}"
        elif arm == "bw":
            win = "3e-4, linear decay to 3e-5 over local updates 1201-1600 (stages 3 and 2) and 1-600 (stage 1)"
        else:
            win = "as recorded by v2.0"
        rows.append((f"`{arm}`", desc[arm][0], desc[arm][1], desc[arm][2], num(int(bud.loc[arm, "updates_expected"])),
                     num(int(bud.loc[arm, "rows_total_expected"])), num(int(bud.loc[arm, "minibatch_steps_expected"])), win))
    write("t02_arms", md(["arm", "T", "scheme", "policy / starts", "updates", "policy rows", "minibatch steps", "learning rate (actor, critic)"], rows))


# ------------------------------------------------------------------------------------------------ per-seed tables
def t03_per_seed(d: Dict[str, object]) -> None:
    """Per-seed Gmax_full, EXP_root and gate verdicts, T = 3 and T = 2."""
    per = d["per"]
    for q in QS:
        rows = []
        bw, jt, es = (sel(per, a, q).set_index("seed") for a in ARMS3)
        for s in bw.index:
            rows.append((s, num(bw.loc[s, "Gmax_full_over_dw"]), num(bw.loc[s, "EXP_root_over_dw"]), "pass" if bw.loc[s, "gates_run_pass"] else "FAIL",
                         num(jt.loc[s, "Gmax_full_over_dw"]), num(jt.loc[s, "EXP_root_over_dw"]),
                         num(es.loc[s, "Gmax_full_over_dw"]), num(es.loc[s, "EXP_root_over_dw"])))
        write(f"t03_per_seed_t3_q{q}", md(["seed", "`bw` Gmax_full/dW", "`bw` EXP_root/dW", "`bw` run gates", "`jt` Gmax_full/dW", "`jt` EXP_root/dW",
                                           "`jt_es` Gmax_full/dW", "`jt_es` EXP_root/dW"], rows))
        rows = []
        v, j = (sel(per, a, q).set_index("seed") for a in ARMS2)
        for s in v.index:
            rows.append((s, num(v.loc[s, "Gmax_full_over_dw"]), num(v.loc[s, "EXP_root_over_dw"]), "pass" if v.loc[s, "gates_run_pass"] else "FAIL",
                         num(j.loc[s, "Gmax_full_over_dw"]), num(j.loc[s, "EXP_root_over_dw"]),
                         "pass" if j.loc[s, "gate_G_A"] else "fail", "pass" if j.loc[s, "gate_G_F"] else "fail", "pass" if j.loc[s, "gate_G_S"] else "fail"))
        write(f"t03_per_seed_t2_q{q}", md(["seed", "`v20` Gmax_full/dW", "`v20` EXP_root/dW", "`v20` run gates", "`jt2` Gmax_full/dW", "`jt2` EXP_root/dW",
                                           "`jt2` G-A", "`jt2` G-F", "`jt2` G-S"], rows))


# ------------------------------------------------------------------------------------------------ bw at T = 3
def gate_rows(g: pd.DataFrame, arms: Sequence[str], gates: Sequence[str]) -> List[Sequence[object]]:
    """Gate-count rows with the Clopper-Pearson intervals."""
    rows = []
    for arm in arms:
        for q in QS:
            r = g[(g["arm"] == arm) & (g["q"] == q)].iloc[0]
            cells: List[object] = [f"`{arm}`", q, int(r["n_exit_ok"])]
            for gt in gates:
                col = "run_pass" if gt == "run_pass" else f"{gt}_pass"
                if pd.isna(r.get(col, np.nan)):
                    cells.append(DASH)
                else:
                    pre = "run_pass" if gt == "run_pass" else gt
                    cells.append(f"{int(r[col])} [{r[f'{pre}_lo']:.3f}, {r[f'{pre}_hi']:.3f}]")
            rows.append(cells)
    return rows


def t04_bw(d: Dict[str, object]) -> None:
    """BW at T = 3: gates, per-stage values, failures."""
    g, bwg, per = d["gate"], d["bwg"], d["per"]
    write("t04a_bw_gates", md(["arm", "q", "runs with exit 0", "G-A pass [95 % CP]", "G-F pass [95 % CP]", "G-N pass [95 % CP]", "all three [95 % CP]"],
                              gate_rows(g, ("bw",), ("G-A", "G-F", "G-N", "run_pass"))))
    write("t04a2_gates_all_t3", md(["arm", "q", "runs with exit 0", "G-A pass [95 % CP]", "G-F pass [95 % CP]", "G-N pass [95 % CP]", "all three [95 % CP]"],
                                   gate_rows(g, ARMS3, ("G-A", "G-F", "G-N", "run_pass"))))
    rows = []
    for q in QS:
        b = bwg[bwg["q"] == q]
        for col, label in (("eta_T_over_dw", "eta_3 / dW (maximum stage-3 deviation gain)"), ("rmse_pos_T_over_eT0", "RMSE_pos / e_3(0)* (abs d < 2q)"),
                           ("tail_mean_T_over_eT0", "tail mean / e_3(0)* (abs d >= 2q)"), ("Gmax_full_over_dw", "Gmax_full / dW"),
                           ("G_max_t1_over_dw", "G_max,1 / dW"), ("G_max_t2_over_dw", "G_max,2 / dW"), ("G_max_t3_over_dw", "G_max,3 / dW"),
                           ("Delta_max_t1_over_dw", "Delta_max,1 (eta_1) / dW"), ("Delta_max_t2_over_dw", "Delta_max,2 (eta_2) / dW")):
            rows.append((q, label, num(b[col].median()), num(b[col].max()), num(b[col].min())))
    write("t04b_bw_values", md(["q", "quantity (20 seeds)", "median", "max", "min"], rows))
    f = bwg[~bwg["passed"].astype(bool)].sort_values(["q", "seed"])
    rows = []
    for r in f.itertuples():
        rows.append((r.q, r.seed, "fail" if not r.gate_G_A else "pass", "fail" if not r.gate_G_F else "pass", "fail" if not r.gate_G_N else "pass",
                     num(r.eta_T_over_dw), num(r.rmse_pos_T_over_eT0, 4), num(r.tail_mean_T_over_eT0, 4), num(r.Gmax_full_over_dw), int(r.Gmax_full_t),
                     f"{r.Gmax_full_d:+.1f}", num(r.dev_eta_T_over_dw), num(r.dev_Gmax_full_over_dw), r.diagnosis))
    write("t04c_bw_failures", md(["q", "seed", "G-A", "G-F", "G-N", "eta_3/dW", "RMSE_pos", "tail mean", "Gmax_full/dW", "at stage", "at d",
                                  "development tier eta_3/dW", "development tier Gmax_full/dW", "diagnosis (tool)"], rows))
    # closeness of all 20 runs per q (the four failures included) to the thresholds
    rows = []
    for q in QS:
        b = bwg[bwg["q"] == q]
        rows.append((q, int((b.eta_T_over_dw > 0.004).sum()), int((b.rmse_pos_T_over_eT0 > 0.04).sum()), int((b.tail_mean_T_over_eT0 > 0.016).sum()),
                     int((b.Gmax_full_over_dw > 0.008).sum())))
    write("t04d_bw_margins", md(["q (all 20 runs, the four failures included)", "runs with eta_3/dW > 0.004 (limit 0.005)", "RMSE_pos > 0.04 (limit 0.05)", "tail mean > 0.016 (limit 0.02)",
                                 "Gmax_full/dW > 0.008 (limit 0.01)"], rows))


# ------------------------------------------------------------------------------------------------ the comparison
def ci(lo: float, hi: float, nd: int = 5) -> str:
    """Format a confidence interval."""
    return f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"


def sec_cells(r: pd.Series, means: bool = True) -> List[object]:
    """Table cells of one paired statistic: means, difference, 95 % and 98.75 % intervals, median difference with its interval, signs."""
    cells: List[object] = []
    if means:
        cells += [num(r.mean_a, 5), num(r.mean_b, 5)]
    cells += [signed(r.mean_diff), ci(r.ci95_lo, r.ci95_hi), ci(r.ci98_75_lo, r.ci98_75_hi), f"{signed(r.median_diff)} {ci(r.median_ci95_lo, r.median_ci95_hi)}",
              f"{int(r.n_neg)} / {int(r.n_pos)}"]
    return cells


def t05_comparison(d: Dict[str, object]) -> None:
    """The criterion table and the secondary tables."""
    c, s = d["crit"], d["sec"]
    rows = []
    for h in ("H1", "H2", "H3", "H1-T2"):
        for q in QS:
            r = c[(c["hypothesis"] == h) & (c["q"] == q)].iloc[0]
            rows.append((h, q, f"`{r.a}` - `{r.b}`", int(r.n_pairs), int(r.dropped_pairs), num(r.mean_a), num(r.mean_b), signed(r.mean_diff), ci(r.ci95_lo, r.ci95_hi),
                         "yes" if r.ci_hi_below_0 else "no", ci(r.ci98_75_lo, r.ci98_75_hi), signed(r.median_diff), ci(r.median_ci95_lo, r.median_ci95_hi),
                         f"{int(r.n_neg)} / {int(r.n_pos)}", int(r.bootstrap_seed)))
    write("t05a_criterion", md(["hypothesis", "q", "pair (a - b)", "n pairs", "dropped pairs", "mean a", "mean b", "mean difference", "95 % interval", "upper end < 0",
                                "98.75 % interval", "median difference", "median 95 % interval", "seeds a<b / a>b", "bootstrap S"], rows))
    verdict = []
    for h in ("H1", "H2", "H3", "H1-T2"):
        sub = c[c["hypothesis"] == h]
        verdict.append((h, "supported" if bool(sub["hypothesis_supported"].all()) else "not supported", ", ".join(str(int(q)) for q in sub[~sub["ci_hi_below_0"].astype(bool)]["q"]) or "none"))
    write("t05a2_verdicts", md(["hypothesis", "verdict (upper end of the 95 % interval < 0 at both q)", "q at which it fails"], verdict))
    hdr = ["statistic", "hypothesis", "q", "mean a", "mean b", "mean difference", "95 % interval", "98.75 % interval", "median difference [95 %]", "seeds a<b / a>b"]
    stats = [("EXP_root_over_dw", "EXP_root / dW"), ("dReach_over_dw", "dReach / dW"), ("rmse_pos_T_over_eT0", "RMSE_pos / e_T(0)*"),
             ("tail_mean_T_over_eT0", "tail mean / e_T(0)*"), ("peak_T_abs", "abs relative error of e_T(0) (peak_T_abs)"), ("e1_at_0", "e_1(0)")]
    for tag, hs in (("t3", ("H1", "H2", "H3")), ("t2", ("H1-T2",))):
        rows = []
        for stat, label in stats:
            for h in hs:
                for q in QS:
                    r = s[(s["hypothesis"] == h) & (s["q"] == q) & (s["statistic"] == stat)].iloc[0]
                    rows.append([label, h, q] + sec_cells(r))
        if tag == "t2":
            for q in QS:
                r = s[(s["hypothesis"] == "H1-T2") & (s["q"] == q) & (s["statistic"] == "stage1_rel_err_abs")].iloc[0]
                rows.append(["stage-1 relative error (abs)", "H1-T2", q] + sec_cells(r))
        write(f"t05b_secondary_{tag}", md(hdr, rows))
    hdr2 = ["statistic", "hypothesis", "q", "mean difference", "95 % interval", "98.75 % interval", "median difference [95 %]", "seeds a<b / a>b"]
    for tag, T, hs in (("t3", 3, ("H1", "H2", "H3")), ("t2", 2, ("H1-T2",))):
        rows = []
        for t in range(1, T + 1):
            names = [f"Delta_max_t{t}_over_dw", f"G_max_t{t}_over_dw"] + ([] if t == 1 else [f"G_in_t{t}", f"G_out_t{t}", f"Delta_in_t{t}", f"Delta_out_t{t}"])
            for stat in names:
                for h in hs:
                    for q in QS:
                        m = s[(s["hypothesis"] == h) & (s["q"] == q) & (s["statistic"] == stat)]
                        if len(m):
                            rows.append([stat, h, q] + sec_cells(m.iloc[0], means=False))
        write(f"t05c_stage_band_{tag}", md(hdr2, rows))


def t06_where(d: Dict[str, object]) -> None:
    """Where the gains are: by stage and band, the location of the maximum, and EXP_root against Gmax_full."""
    per = d["per"].copy()
    rows, rows_d = [], []
    for arm in ARMS3 + ARMS2:
        T = 3 if arm in ARMS3 else 2
        for q in QS:
            r = sel(per, arm, q)
            cells = [f"`{arm}`", q]
            cd = [f"`{arm}`", q]
            for t in (1, 2, 3):
                cells.append(num(r[f"G_max_t{t}_over_dw"].median(), 5) if t <= T else DASH)
                cd.append(num(r[f"Delta_max_t{t}_over_dw"].median(), 5) if t <= T else DASH)
            for t in (2, 3):
                cells.append(f"{num(r[f'G_in_t{t}'].median(), 5)} / {num(r[f'G_out_t{t}'].median(), 5)}" if t <= T else DASH)
                cd.append(f"{num(r[f'Delta_in_t{t}'].median(), 5)} / {num(r[f'Delta_out_t{t}'].median(), 5)}" if t <= T else DASH)
            rows.append(cells)
            rows_d.append(cd)
    write("t06a_stage_band_medians", md(["arm", "q", "G_max,1 / dW", "G_max,2 / dW", "G_max,3 / dW", "G_in,2 / G_out,2", "G_in,3 / G_out,3"], rows))
    write("t06c_delta_medians", md(["arm", "q", "Delta_max,1 (eta_1) / dW", "Delta_max,2 (eta_2) / dW", "Delta_max,3 (eta_3) / dW", "Delta_in,2 / Delta_out,2",
                                    "Delta_in,3 / Delta_out,3"], rows_d))
    rows = []
    for arm in ARMS3 + ARMS2:
        for q in QS:
            r = sel(per, arm, q)
            t = r["Gmax_full_t"].astype(int)
            edge = 2 * q * (t - 1)
            a = r["Gmax_full_d"].abs()
            out = int(((t >= 2) & (a >= edge - 1e-9)).sum())
            on_edge = int(((t >= 2) & ((a - edge).abs() <= 1e-9)).sum())
            rows.append((f"`{arm}`", q, int((t == 1).sum()), int((t == 2).sum()), int((t == 3).sum()), out, on_edge, 20 - out - int((t == 1).sum())))
    write("t06b_argmax_location", md(["arm", "q", "Gmax_full at stage 1", "at stage 2", "at stage 3", "of which out of band (stages >= 2)",
                                      "of the out-of-band ones, exactly on the edge node (abs d = 2q(t-1))", "in band (stages >= 2)"], rows))
    rows = []
    for arm in ARMS3 + ARMS2:
        for q in QS:
            r = sel(per, arm, q)
            rows.append((f"`{arm}`", q, num(r["Gmax_full_over_dw"].mean()), num(r["Gmax_full_over_dw"].median()), num(r["Gmax_full_over_dw"].max()),
                         num(r["EXP_root_over_dw"].mean()), num(r["EXP_root_over_dw"].median()), num(r["EXP_root_over_dw"].max()),
                         num(r["Gmax_full_over_dw"].mean() / r["EXP_root_over_dw"].mean(), 1)))
    write("t07_exp_vs_gmax", md(["arm", "q", "Gmax_full mean", "median", "max", "EXP_root mean", "median", "max", "ratio of the means Gmax_full / EXP_root"], rows))


def t16_diag(d: Dict[str, object]) -> None:
    """D6 items 2, 3 and 7: argmax gaps, dReach / Deltamax_all / dFull, and the maximum normalised Beta standard deviation."""
    per = d["per"]
    rows = []
    for arm in ARMS3 + ARMS2:
        T = 3 if arm in ARMS3 else 2
        for q in QS:
            r = sel(per, arm, q)
            cells = [f"`{arm}`", q]
            for t in (1, 2, 3):
                if t <= T:
                    cells.append(f"{num(r[f'conc_max_std_norm_t{t}'].median(), 4)} ({num(r[f'conc_max_std_norm_t{t}'].max(), 4)})")
                else:
                    cells.append(DASH)
            for t in (1, 2, 3):
                if t <= T:
                    cells.append(f"{num(r[f'dev_conc_max_std_norm_t{t}'].median(), 4)} ({num(r[f'dev_conc_max_std_norm_t{t}'].max(), 4)})")
                else:
                    cells.append(DASH)
            rows.append(cells)
    write("t16a_conc", md(["arm", "q", "final tier t = 1: median (max)", "t = 2", "t = 3", "development tier t = 1: median (max)", "t = 2", "t = 3"], rows))
    rows = []
    for arm in ARMS3 + ARMS2:
        for q in QS:
            r = sel(per, arm, q)
            rows.append((f"`{arm}`", q) + tuple(f"{num(r[c].median())} ({num(r[c].max())})" for c in
                                                ("EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw", "dFull_over_dw")))
    write("t16b_exp_all", md(["arm", "q", "EXP_root / dW: median (max)", "dReach / dW", "Deltamax_all / dW", "dFull / dW"], rows))
    rows = []
    for arm in ARMS3 + ARMS2:
        T = 3 if arm in ARMS3 else 2
        for q in QS:
            r = sel(per, arm, q)
            for t in range(2, T + 1):
                cells = [f"`{arm}`", q, t]
                for col in (f"G_argmax_d_t{t}", f"Delta_argmax_d_t{t}", f"G_in_argmax_d_t{t}", f"G_out_argmax_d_t{t}", f"Delta_in_argmax_d_t{t}", f"Delta_out_argmax_d_t{t}"):
                    v = r[col].astype(float)
                    cells.append(f"{num(v.abs().median(), 1)} [{num(v.min(), 0)}, {num(v.max(), 0)}]")
                rows.append(cells)
    write("t16c_argmax_gaps", md(["arm", "q", "stage", "G_max,t: median abs d [min, max over seeds]", "Delta_max,t", "G_in", "G_out", "Delta_in", "Delta_out"], rows))


def t08_best(d: Dict[str, object]) -> None:
    """Best of run."""
    b = d["best"]
    run = pd.read_csv(RES / "eval" / "best_of_run.csv")
    rows, rows_g = [], []
    for r in b.itertuples():
        U = tc.JOINT_BUDGETS[r.arm]["updates"]
        m = run[(run["arm"] == r.arm) & (run["q"] == r.q)]
        n_last = int((m["best_update"] == U).sum())
        rows.append((f"`{r.arm}`", r.q, r.n, num(r.Gmax_full_over_dw_last_mean), num(r.Gmax_full_over_dw_best_mean), num(r.Gmax_full_over_dw_last_median),
                     num(r.Gmax_full_over_dw_best_median), num(r.EXP_root_over_dw_last_mean), num(r.EXP_root_over_dw_best_mean), num(r.best_update_mean, 1),
                     n_last, int(r.n_gates_pass_best)))
        gs = lambda c: int(m[c].astype(bool).sum()) if c in m and m[c].notna().any() else None
        rows_g.append((f"`{r.arm}`", q_ := r.q, gs("gate_G_A"), gs("gate_G_F"), gs("gate_G_N"), gs("gate_G_S") if r.arm == "jt2" else DASH,
                       int(r.n_gates_pass_best), num(m["eta_T_over_dw"].median()), num(m["rmse_pos_T_over_eT0"].median(), 4), num(m["tail_mean_T_over_eT0"].median(), 4)))
    write("t08_best_of_run", md(["arm", "q", "n", "Gmax_full mean, last iterate", "mean, best of run", "median, last", "median, best", "EXP_root mean, last",
                                 "mean, best", "mean best update", "runs whose best update is the last update", "best-of-run gates all pass"], rows))
    write("t08b_best_gates", md(["arm", "q", "G-A pass (of 20)", "G-F pass", "G-N pass", "G-S pass (T = 2)", "all gates", "eta_T / dW median", "RMSE_pos median", "tail mean median"], rows_g))


def t09_coverage(d: Dict[str, object]) -> None:
    """Coverage by start stage and bin (all columns: medians over seeds of the per-run values)."""
    c = d["cov"]
    rows = []
    for arm in ("bw", "jt", "jt_es", "jt2"):
        for q in QS:
            for stage in (2, 3):
                m = c[(c["arm"] == arm) & (c["q"] == q) & (c["stage"] == stage)]
                if len(m) == 0:
                    continue
                if arm == "bw":
                    split = f"direct design {num(m['rows_direct_design'].median(), 0)}"
                else:
                    split = (f"root path {num(m['rows_root_path'].median(), 0)}; direct {num(m['rows_direct_es'].median(), 0)}"
                             + (f"; from stage-2 starts (approximate) {num(m['rows_from_stage2_start_approx'].median(), 0)}" if stage == 3 and arm == "jt_es" else ""))
                rows.append((f"`{arm}`", q, stage, num(m["rows"].median(), 0), split, f"{num(m['bins_visited'].median(), 0)} of {num(m['n_bins'].median(), 0)}",
                             num(m["share_in_band"].median(), 4), num(m["mean_rows_per_bin_in_band"].median(), 1), num(m["mean_rows_per_bin_out_of_band"].median(), 1)))
    write("t09_coverage", md(["arm", "q", "stage", "policy rows", "rows by start (root path / direct / from stage-2 starts)", "bins visited", "share of rows in band",
                              "mean rows per in-band bin", "mean rows per out-of-band bin"], rows))
    rows = []
    for q in QS:
        for stage in (2, 3):
            m = c[(c["arm"] == "bw") & (c["q"] == q) & (c["stage"] == stage)]
            rows.append((q, stage, num(m["n_start_tail"].median(), 0), num(m["n_start_near"].median(), 0), num(m["n_start_mid"].median(), 0)))
    write("t09b_bw_design", md(["q", "stage", "direct starts in the tail stratum (median)", "near stratum", "mid stratum"], rows))


def t10_cost(d: Dict[str, object]) -> None:
    """Cost: per run, per category and per phase."""
    k = d["cost"]
    rows = []
    for arm in ("bw", "jt", "jt_es", "jt2", "v20"):
        for q in QS:
            m = k[(k["arm"] == arm) & (k["q"] == q)]
            rows.append((f"`{arm}`", q, num(m["updates"].median(), 0), num(m["policy_rows_total"].median(), 0), num(m["minibatch_steps"].median(), 0),
                         num(m["wall_sec"].median(), 0), num(m["wall_sec"].max(), 0), num(m["cpu_sec"].median(), 0) if m["cpu_sec"].notna().any() else DASH,
                         num(m["train_rollout_sec"].median(), 1), num(m["train_update_sec"].median(), 1), num(m["dev_verifier_sec"].median(), 1),
                         num(m["final_eval_sec"].median(), 2) if m["final_eval_sec"].notna().any() else DASH,
                         num(m["peak_rss_kb"].median() / 1024, 0) if m["peak_rss_kb"].notna().any() else DASH))
    write("t10_cost", md(["arm", "q", "updates", "policy rows", "minibatch steps", "wall s (median)", "wall s (max)", "CPU s (median; v20: not recorded)", "rollout s",
                          "update s", "dev verifier s", "final evaluation s", "peak RSS MB (ru_maxrss / 1024)"], rows))
    rows = []
    for arm, stages in (("bw", (3, 2, 1)), ("v20", (2, 1))):
        for q in QS:
            m = k[(k["arm"] == arm) & (k["q"] == q)]
            cells = [f"`{arm}`", q]
            for t in (3, 2, 1):
                if t in stages and f"wall_stage{t}_sec" in m:
                    w = m[f"wall_stage{t}_sec"].median()
                    c = m[f"cpu_stage{t}_sec"].median() if f"cpu_stage{t}_sec" in m else np.nan
                    cells.append(f"{num(w, 1)} / {num(c, 1) if not np.isnan(c) else DASH}")
                else:
                    cells.append(DASH)
            rows.append(cells)
    write("t10c_cost_phases", md(["arm", "q", "stage-3 phase: wall s / CPU s", "stage-2 phase", "stage-1 phase"], rows))
    tot = k.groupby("arm")["wall_sec"].sum() / 3600
    write("t10b_cost_total", md(["arm", "runs", "total wall hours"], [(f"`{a}`", int((k['arm'] == a).sum()), f"{tot[a]:.2f}") for a in tot.index]
                                + [("all", len(k), f"{tot.sum():.2f}")]))


def t11_t2(d: Dict[str, object]) -> None:
    """The T = 2 comparison: gates and closed-form recovery."""
    g, per = d["gate"], d["per"]
    write("t11a_gates_t2", md(["arm", "q", "runs with exit 0", "G-A pass [95 % CP]", "G-F pass [95 % CP]", "G-N pass [95 % CP]", "G-S pass [95 % CP]",
                               "all four [95 % CP]"], gate_rows(g, ARMS2, ("G-A", "G-F", "G-N", "G-S", "run_pass"))))
    rows = []
    for arm in ARMS2:
        for q in QS:
            r = sel(per, arm, q)
            rows.append((f"`{arm}`", q, num(r["eT_at_0"].median(), 2), num(r["eT_star_at_0"].median(), 2), num(r["peak_T_signed"].median(), 4), num(r["peak_T_abs"].median(), 4),
                         num(r["rmse_pos_T_over_eT0"].median(), 4), num(r["tail_mean_T_over_eT0"].median(), 4), num(r["tail_max_T_over_eT0"].median(), 4),
                         num(r["stage1_rel_err_signed"].median(), 4), num(r["stage1_rel_err_abs"].median(), 4), num(r["eta_T_over_dw"].median(), 5),
                         num(r["Gmax_full_over_dw"].median(), 5), num(r["e1_at_0"].median(), 3)))
    write("t11b_recovery_t2", md(["arm", "q", "e_2(0)", "e_2(0)* (tent)", "signed relative error of e_2(0) (peak)", "abs value (peak_T_abs)", "RMSE_pos / e_2(0)*", "tail mean / e_2(0)*",
                                  "tail max / e_2(0)*", "stage-1 relative error, signed", "abs", "eta_2 / dW", "Gmax_full / dW", "e_1(0)"], rows))




def t12_recovery_t3(d: Dict[str, object]) -> None:
    """The terminal-stage recovery of the three T = 3 arms."""
    per = d["per"]
    rows = []
    for arm in ARMS3:
        for q in QS:
            r = sel(per, arm, q)
            rows.append((f"`{arm}`", q, num(r["eT_at_0"].median(), 2), num(r["eT_star_at_0"].median(), 2), num(r["peak_T_signed"].median(), 4), num(r["peak_T_abs"].median(), 4),
                         num(r["rmse_pos_T_over_eT0"].median(), 4), num(r["tail_mean_T_over_eT0"].median(), 4), num(r["tail_max_T_over_eT0"].median(), 4),
                         num(r["eta_T_over_dw"].median(), 5), num(r["e1_at_0"].median(), 3)))
    write("t12_recovery_t3", md(["arm", "q", "e_3(0)", "e_3(0)* (tent)", "signed relative error of e_3(0) (peak)", "abs value (peak_T_abs)", "RMSE_pos / e_3(0)*",
                                 "tail mean / e_3(0)*", "tail max / e_3(0)*", "eta_3 / dW", "e_1(0)"], rows))


def t13_curves(d: Dict[str, object]) -> None:
    """The development checks of the joint arms at fractions of their budget (descriptive; the figures show the whole curves)."""
    rows = []
    for arm in ("jt", "jt_es", "jt2"):
        U = tc.JOINT_BUDGETS[arm]["updates"]
        for q in QS:
            seeds = sorted(sel(d["per"], arm, q)["seed"])
            cells: Dict[float, List[List[float]]] = {f: [] for f in (0.25, 0.5, 0.75, 1.0)}
            for sd in seeds:
                base = RES / ("runs_t2" if arm == "jt2" else "runs") / f"q{q}" / f"seed{sd}" / arm
                c = pd.read_csv(base / "joint_checks.csv")
                c = c[c["valid"].astype(bool)]
                for f in cells:
                    i = (c["update"] - f * U).abs().idxmin()
                    cells[f].append([c.loc[i, "update"], c.loc[i, "Gmax_full_over_dw"], c.loc[i, "EXP_root_over_dw"]])
            for f, v in cells.items():
                a = np.array(v)
                rows.append((f"`{arm}`", q, f"{int(f * 100)} %", num(np.median(a[:, 0]), 0), num(np.median(a[:, 1])), num(np.median(a[:, 2]))))
    write("t13_curves", md(["arm", "q", "fraction of the updates", "check update (median over seeds)", "development Gmax_full / dW (median)",
                            "development EXP_root / dW (median)"], rows))


def t14_reading(d: Dict[str, object]) -> None:
    """The reading rule of D7 and what 'at least 18 of 20 at both q' means as a function of a run pass probability (arithmetic only)."""
    from math import comb
    g = d["gate"]
    rows = []
    for q in QS:
        r = g[(g["arm"] == "bw") & (g["q"] == q)].iloc[0]
        rows.append((f"`bw` T = 3", q, int(r["run_pass"]), f"[{r['run_pass_lo']:.3f}, {r['run_pass_hi']:.3f}]", "yes" if r["run_pass"] >= 18 else "no"))
    for q in QS:
        r = g[(g["arm"] == "v20") & (g["q"] == q)].iloc[0]
        rows.append((f"`v20` T = 2 (recorded)", q, int(r["run_pass"]), f"[{r['run_pass_lo']:.3f}, {r['run_pass_hi']:.3f}]", "yes" if r["run_pass"] >= 18 else "no"))
    write("t14a_reading", md(["arm", "q", "runs passing all gates (of 20)", "95 % Clopper-Pearson", "at least 18"], rows))
    rows = []
    for p in (0.70, 0.80, 0.85, 0.90, 0.95, 0.98):
        one = sum(comb(20, k) * p ** k * (1 - p) ** (20 - k) for k in range(18, 21))
        rows.append((f"{p:.2f}", f"{one:.3f}", f"{one * one:.3f}"))
    write("t14b_rule_probability", md(["true per-run pass probability p (same at both q)", "P(at least 18 of 20) at one q", "P(at least 18 of 20) at both q"], rows))


def t15_ga_parts(d: Dict[str, object]) -> None:
    """The three parts of G-A and the two parts of G-N, counted per arm and q, and the T = 2 criterion rows."""
    per, c = d["per"], d["crit"]
    rows = []
    for arm in ARMS3 + ARMS2:
        for q in QS:
            r = sel(per, arm, q)
            rows.append((f"`{arm}`", q, int(r["ga_eta"].astype(bool).sum()), int(r["ga_rmse"].astype(bool).sum()), int(r["ga_tail"].astype(bool).sum()),
                         int(r["gate_G_A"].astype(bool).sum()), int(r["gate_G_F"].astype(bool).sum()), int(r["gn_eta"].astype(bool).sum()),
                         int(r["gn_gmax"].astype(bool).sum())))
    write("t15_gate_parts", md(["arm", "q", "G-A eta part (<= 0.005)", "G-A RMSE_pos part (<= 0.05)", "G-A tail part (<= 0.02)", "G-A all", "G-F",
                                "G-N eta part", "G-N Gmax part"], rows))
    rows = []
    for q in QS:
        r = c[(c["hypothesis"] == "H1-T2") & (c["q"] == q)].iloc[0]
        rows.append((q, f"`{r.a}` - `{r.b}`", int(r.n_pairs), num(r.mean_a), num(r.mean_b), signed(r.mean_diff), ci(r.ci95_lo, r.ci95_hi),
                     "yes" if r.ci_hi_below_0 else "no", ci(r.ci98_75_lo, r.ci98_75_hi), signed(r.median_diff), ci(r.median_ci95_lo, r.median_ci95_hi),
                     f"{int(r.n_neg)} / {int(r.n_pos)}"))
    write("t05a_criterion_t2", md(["q", "pair (a - b)", "n pairs", "mean a", "mean b", "mean difference", "95 % interval", "upper end < 0", "98.75 % interval",
                                   "median difference", "median 95 % interval", "seeds a<b / a>b"], rows))


def main() -> int:
    """Generate every table."""
    d = load()
    for fn in (t01_checks, t02_arms, t03_per_seed, t04_bw, t05_comparison, t06_where, t08_best, t09_coverage, t10_cost, t11_t2, t12_recovery_t3, t13_curves, t14_reading, t15_ga_parts, t16_diag):
        fn(d)
    print(f"{len(list(OUT.glob('*.md')))} tables in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
