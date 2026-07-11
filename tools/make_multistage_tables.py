"""Generate the multi-stage paper tables (plan section 6, Tables 1-4) as LaTeX.

Reads the committed convergence JSONs and writes booktabs-style .tex files to
``paper/multistage/tables/``:

  Table 1  Two-stage recovery metrics and exploitability      [T=2]
  Table 2  Three-stage exploitability certificate             [T=3]
  Table 3  Grid refinement, seed robustness, falsification
  Table 4  T=2,3,4,5 benchmark comparison

Run:
    python tools/make_multistage_tables.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.theory_multistage import (  # noqa: E402
    eq_utility_two_stage, g1_two_stage, g2_two_stage,
)

OUT_DIR = os.path.join("paper", "multistage", "tables")
CONV = os.path.join("results", "multi_stage", "convergence")
WH, WL, K, EBAR, Q = 6.0, 2.0, 1.0 / 3500.0, 100.0, 50.0
DW = WH - WL


def load_T(T: int) -> List[dict]:
    return [json.load(open(f)) for f in
            sorted(glob.glob(os.path.join(CONV, f"ms_T{T}_q50_seed*_gateT{T}_convergence.json")))]


def _write(name: str, tex: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        f.write(tex)
    print(f"  wrote {name}")


def total_effort_cost(d: dict, T: int) -> tuple[float, float]:
    """(total on-path effort, total cost) for a run, from onpath_summary or curves."""
    if d.get("onpath_summary"):
        return d["onpath_summary"]["total_effort"], d["onpath_summary"]["total_cost"]
    c = d.get("effort_curves")
    if c:
        te = tc = 0.0
        for _, s in c["stages"].items():
            p = np.array(s["onpath_dist"]); p = p / p.sum()
            e = np.array(s["learned"]); te += float((p * e).sum()); tc += float(K * (p * e * e).sum())
        return te, tc
    return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Table 1: two-stage recovery
# ---------------------------------------------------------------------------

def table1() -> None:
    runs = load_T(2)
    if not runs:
        print("  [skip T1]"); return
    rows = []
    for r in runs:
        rm = r["recovery_metrics"]; fe = r["final_eval"]; gr = r["grid_refinement"]
        rows.append((r["params"]["seed"], rm["re_1"], rm["rpe_2"], rm["rpe_2_core"],
                     rm["pl_2_over_dw"], fe["exp_over_dw"], gr["exp_ucb_over_dw"],
                     fe["delta_sum_reachable"] / DW))
    arr = np.array([row[1:] for row in rows])
    lines = [
        r"\begin{tabular}{lccccccc}", r"\toprule",
        r"seed & $RE_1$ & $RPE_2$ & $RPE_2^{\text{core}}$ & $PL_2/\Delta W$ & "
        r"$EXP/\Delta W$ & $EXP^{UCB}/\Delta W$ & $d_{\text{reach}}/\Delta W$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(f"{row[0]} & " + " & ".join(f"{v:.4f}" for v in row[1:]) + r" \\")
    lines.append(r"\midrule")
    lines.append("mean & " + " & ".join(f"{v:.4f}" for v in arr.mean(0)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("table1_two_stage_recovery.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table 2: three-stage certificate
# ---------------------------------------------------------------------------

def table2() -> None:
    runs = load_T(3)
    if not runs:
        print("  [skip T2]"); return
    rows = []
    for r in runs:
        fe = r["final_eval"]; gr = r["grid_refinement"]
        rows.append((r["params"]["seed"], fe["exp"], fe["exp_over_dw"],
                     gr["exp_ucb_over_dw"], fe["delta_sum_reachable"] / DW,
                     fe["delta_sum_full"] / DW,
                     "yes" if fe["delta_sum_reachable"] / DW <= 0.03 else "no"))
    lines = [
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"seed & $EXP$ & $EXP/\Delta W$ & $EXP^{UCB}/\Delta W$ & "
        r"$d_{\text{reach}}/\Delta W$ & $d_{\text{full}}/\Delta W$ & certified \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(f"{row[0]} & {row[1]:.4f} & {row[2]:.4f} & {row[3]:.4f} & "
                     f"{row[4]:.4f} & {row[5]:.4f} & {row[6]} \\\\")
    num = np.array([row[1:6] for row in rows], float)
    lines.append(r"\midrule")
    lines.append("mean & " + " & ".join(f"{v:.4f}" for v in num.mean(0)) +
                 f" & {sum(1 for r in rows if r[6]=='yes')}/{len(rows)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("table2_three_stage_certificate.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table 3: robustness (grid refinement, seed robustness, falsification)
# ---------------------------------------------------------------------------

def table3() -> None:
    from utils.dp_verifier import verify
    # (a) grid refinement on the T=3 seed-42 policy proxy: use its saved sequence
    r3 = load_T(3)
    # (b) falsification at T=2
    g1 = g1_two_stage(Q, WH, WL, K); e_one = DW / (4 * Q * K)
    def cf(t, d):
        d = np.asarray(d, float)
        return g2_two_stage(d, Q, WH, WL, K, EBAR) if t >= 2 else np.full_like(d, g1)
    def clo(t, d): return np.full_like(np.asarray(d, float), 5.0)
    def chi(t, d): return np.full_like(np.asarray(d, float), EBAR)
    def rep(t, d): return np.full_like(np.asarray(d, float), e_one)
    def ngap(t, d):
        d = np.asarray(d, float)
        return (np.full_like(d, float(g2_two_stage(np.asarray(0.0), Q, WH, WL, K, EBAR)))
                if t >= 2 else np.full_like(d, g1))
    fals = [("closed form $e^*_{CF}$", cf), ("constant low", clo),
            ("constant high", chi), ("one-stage repeated", rep), ("no-gap stage 2", ngap)]

    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"\multicolumn{3}{l}{\emph{(a) Grid refinement of $EXP$ (T=3, seed 42)}} \\",
             r"\midrule", r"grid $M$ & $EXP$ & $d_{\text{reach}}/\Delta W$ \\", r"\midrule"]
    if r3:
        gr = r3[0]["grid_refinement"]
        for M, e, dr in zip(gr["d_grid_sizes"], gr["exp"], gr["delta_sum_reachable"]):
            lines.append(f"{M} & {e:.5f} & {dr/DW:.5f} \\\\")
        lines.append(f"Richardson & {gr['exp_richardson']:.5f} & -- \\\\")
    lines += [r"\midrule",
              r"\multicolumn{3}{l}{\emph{(b) Seed robustness: cross-seed std}} \\",
              r"\midrule", r"$T$ & std $EXP/\Delta W$ & std $d_{\text{reach}}/\Delta W$ \\", r"\midrule"]
    for T in (2, 3, 4, 5):
        runs = load_T(T)
        if not runs: continue
        ex = np.array([r["final_eval"]["exp_over_dw"] for r in runs])
        dr = np.array([r["final_eval"]["delta_sum_reachable"] / DW for r in runs])
        lines.append(f"{T} & {ex.std(ddof=0):.4f} & {dr.std(ddof=0):.4f} \\\\")
    lines += [r"\midrule",
              r"\multicolumn{3}{l}{\emph{(c) Falsification (T=2): $EXP/\Delta W$}} \\",
              r"\midrule", r"policy & \multicolumn{2}{c}{$EXP/\Delta W$} \\", r"\midrule"]
    for nm, pol in fals:
        rr = verify(pol, w_h=WH, w_l=WL, k=K, q=Q, T=2, e_bar=EBAR)
        lines.append(f"{nm} & \\multicolumn{{2}}{{c}}{{{rr.exp_over_dw:.4f}}} \\\\")
    if load_T(2):
        tp = np.mean([r["final_eval"]["exp_over_dw"] for r in load_T(2)])
        lines.append(f"TEL-PPO (learned) & \\multicolumn{{2}}{{c}}{{{tp:.4f}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("table3_robustness.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table 4: multi-stage benchmark comparison
# ---------------------------------------------------------------------------

def table4() -> None:
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"$T$ & total effort & exp. cost & $EXP/\Delta W$ & "
             r"$d_{\text{reach}}/\Delta W$ & certified \\", r"\midrule"]
    for T in (2, 3, 4, 5):
        runs = load_T(T)
        if not runs: continue
        ex = np.mean([r["final_eval"]["exp_over_dw"] for r in runs])
        dr = np.mean([r["final_eval"]["delta_sum_reachable"] / DW for r in runs])
        nc = sum(1 for r in runs if r["final_eval"]["delta_sum_reachable"] / DW <= 0.03)
        if T == 2:  # analytic recovered values (run predates onpath fields)
            te = 2 * g1_two_stage(Q, WH, WL, K)
            tc = (WH + WL) / 2 - eq_utility_two_stage(Q, WH, WL, K)
            te_s = f"{te:.1f}$^\\ast$"; tc_s = f"{tc:.3f}$^\\ast$"
        else:
            tes, tcs = zip(*[total_effort_cost(r, T) for r in runs])
            te_s = f"{np.mean(tes):.1f}"; tc_s = f"{np.mean(tcs):.3f}"
        lines.append(f"{T} & {te_s} & {tc_s} & {ex:.4f} & {dr:.4f} & {nc}/{len(runs)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    lines.append(r"% $^\ast$ T=2 total effort/cost are analytic recovered values")
    _write("table4_multistage_summary.tex", "\n".join(lines) + "\n")


def main() -> int:
    print(f"generating multi-stage tables -> {OUT_DIR}")
    table1(); table2(); table3(); table4()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
