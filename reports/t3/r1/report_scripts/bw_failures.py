#!/usr/bin/env python3
"""T3-R1 P3: the four failing ``bw`` runs at T = 3 in detail (``04_t3_results.md`` section 4.1).

Descriptive only: no new run, nothing of the pre-registered numbers, criteria or verdicts changes. For every failing
``bw`` run (a run of ``analysis/per_run_table.csv`` that does not pass G-A and G-F and G-N) this script reports

  * the failing gate and part, its value and limit, and the result of the other gates;
  * where the maximum is (stage, gap d, in band or not) and, at that d, Delta_t(d) and G_t(d), the learned mean effort
    e_t(d), the verifier's best-response effort, the normalised Beta standard deviation;
  * e_t(d) against the best-response effort over d +- 20 (table and figure);
  * the development checks of the failing stage's phase (``ms_checks_stage{t}.csv``, ``Delta`` against the local update);
  * the rank of the value among the 20 runs of its q;
  * the same quantities, at the same stage and d, for the other 18 runs of the q.

The best-response effort, the one-step argmax and the Beta spread are not in ``analysis/*.csv``; they are re-read from the
stored last-iterate weights of the 40 ``bw`` runs (``state_end_stage1.pt``) with the evaluator of the round
(``tools/t3/t3_eval.py``, final tier; pure, no random number). Those weights and ``eval/arrays/*.npz`` are untracked and
live in the round's own worktree: set ``T3R1_UNTRACKED_ROOT`` to its ``results/t3_r1`` (default: this worktree's).
``analysis/bw_failures/replay_check.json`` records that the evaluation reproduces the stored arrays (mean effort, one-step
gain and dynamic gain at every node of every stage) and the per-run scalars of ``eval/per_run.csv`` bit for bit; the script stops otherwise.

Writes the CSVs to ``results/t3_r1/analysis/bw_failures/``, two figures to ``results/t3_r1/analysis/figures/`` and the
table fragments ``tables/t17*.md`` to ``reports/t3/r1/tables/``.

Usage:
    T3R1_UNTRACKED_ROOT=<round worktree>/results/t3_r1 python reports/t3/r1/report_scripts/bw_failures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RES = ROOT / "results" / "t3_r1"
UNTRACKED = Path(os.environ.get("T3R1_UNTRACKED_ROOT", str(RES)))
OUT_D = RES / "analysis" / "bw_failures"
FIG = RES / "analysis" / "figures"
for p in (ROOT, ROOT / "tools" / "ms", ROOT / "tools" / "t3", HERE):
    sys.path.insert(0, str(p))

import t3_eval as te  # noqa: E402
import t3_metrics as tm  # noqa: E402
import results_tables as rt  # noqa: E402
from utils.dp_br_verifier import FINAL_CONFIG  # noqa: E402

QS = (50, 60)
STAGE_ARRAYS = ("d", "e", "br", "dev", "delta", "G", "sd")
Evals = Dict[Tuple[int, int], Dict[int, Dict[str, np.ndarray]]]
WINDOW = 20.0
HIST_LOCALS = (25, 100, 200, 400, 800, 1200, 1400, 1600)
ZERO_EFFORT = 1.0  # a stage-2 node counts as "zero effort" when the learned mean effort is below 1 (the effort range is 100)
TOL = 1e-9


def evaluate_bw(per: pd.DataFrame) -> Tuple[Evals, Dict[str, Any]]:
    """Final-tier evaluation of the 40 ``bw`` runs from their stored last-iterate weights.

    Args:
        per: The ``bw`` rows of the per-run table.

    Returns:
        ``({(q, seed): {stage: {name: array}}}, replay_check)``. The arrays are on the stage grid: ``d``, mean effort ``e``,
        dynamic best-response effort ``br``, one-step argmax ``dev``, ``delta`` and ``G`` divided by dW, and the normalised
        Beta standard deviation ``sd``. ``replay_check`` counts the exact matches with the stored arrays and scalars.
    """
    import torch

    torch.set_num_threads(1)
    # the default pandas float parser is not exact; the round-trip parser returns the stored doubles
    exact = pd.read_csv(RES / "eval" / "per_run.csv", float_precision="round_trip").set_index("run")
    out: Evals = {}
    arrays_equal, scalars_equal, n = 0, 0, 0
    mism: List[str] = []
    for r in per.sort_values(["q", "seed"]).itertuples():
        rd = UNTRACKED / "runs" / f"q{r.q}" / f"seed{r.seed}" / "bw"
        cfg = te.read_json(rd / "run_config.json")
        ev = te.evaluate_tier(te.load_candidate("bw", rd, cfg), FINAL_CONFIG)
        dw = float(ev.res.dw)
        stored = np.load(UNTRACKED / "eval" / "arrays" / f"{r.run}.npz")
        ok = True
        stages: Dict[int, Dict[str, np.ndarray]] = {}
        for t, s in ev.res.stages.items():
            g = np.asarray(ev.G[t])
            stages[t] = {"d": np.asarray(s.d_grid), "e": np.asarray(s.e_hat), "br": np.asarray(s.a_br),
                         "dev": np.asarray(s.a_dev), "delta": np.asarray(s.delta) / dw, "G": g / dw,
                         "sd": np.asarray(s.std_norm)}
            for key, mine in (("e_hat", s.e_hat), ("delta", s.delta), ("G", g), ("d_grid", s.d_grid)):
                ok = ok and np.array_equal(stored[f"fin_t{t}_{key}"], np.asarray(mine))
        sc = ev.scalars
        sc_ok = all(float(sc[k]) == float(exact.loc[r.run, k]) for k in ("Gmax_full_over_dw", "EXP_root_over_dw", "eta_T_over_dw"))
        arrays_equal += int(ok)
        scalars_equal += int(sc_ok)
        n += 1
        if not (ok and sc_ok):
            mism.append(str(r.run))
        out[(int(r.q), int(r.seed))] = stages
    check = {"runs": n, "stored_arrays_bit_identical": arrays_equal, "per_run_scalars_bit_identical": scalars_equal,
             "mismatching_runs": mism, "tier": "final", "arrays_compared": "d_grid, e_hat, delta, G at every stage",
             "scalars_compared": "Gmax_full_over_dw, EXP_root_over_dw, eta_T_over_dw of eval/per_run.csv"}
    return out, check


def failing_points(per: pd.DataFrame, crit: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One record per failing run: the failing gate and part, the quantity, its limit, and the stage and gap of its maximum."""
    ga, gf = crit["gates"]["G-A"], crit["gates"]["G-F"]
    pts = []
    for r in per[~per["passed"]].sort_values(["q", "seed"]).itertuples():
        T = int(r.T)
        if not r.gate_G_A:
            part = [k for k, v in (("eta_T_over_dw", r.ga_eta), ("rmse_pos_T_over_eT0", r.ga_rmse), ("tail_mean_T_over_eT0", r.ga_tail)) if not v]
            assert part == ["eta_T_over_dw"], part
            pts.append({"q": int(r.q), "seed": int(r.seed), "gate": "G-A (first part, eta_3)", "col": "eta_T_over_dw", "quantity": "eta_3/dW",
                        "value": float(r.eta_T_over_dw), "limit": ga["eta_T_over_dw"], "stage": T, "d": float(r.Delta_argmax_d_t3)})
        elif not r.gate_G_F:
            pts.append({"q": int(r.q), "seed": int(r.seed), "gate": "G-F", "col": "Gmax_full_over_dw", "quantity": "Gmax_full/dW",
                        "value": float(r.Gmax_full_over_dw), "limit": gf["Gmax_full_over_dw"], "stage": int(r.Gmax_full_t), "d": float(r.Gmax_full_d)})
        else:
            raise ValueError(f"run {r.run} fails G-N only: not a case of this round")
    return pts


def at_node(arrs: Dict[str, np.ndarray], d: float) -> Dict[str, float]:
    """The stage arrays at the node ``d`` (an exact node of the grid)."""
    j = int(np.argmin(np.abs(arrs["d"] - d)))
    assert abs(arrs["d"][j] - d) < TOL, (d, arrs["d"][j])
    return {k: float(arrs[k][j]) for k in STAGE_ARRAYS}


def in_band(q: int, t: int, d: float) -> bool:
    """D6: the band is |d| < 2q(t-1); a node exactly on the edge is out of band."""
    return bool(abs(d) < 2.0 * q * (t - 1) - TOL)


def history(q: int, seed: int, stage: int) -> pd.DataFrame:
    """The valid development checks of the stage's phase: local update and ``Delta`` (= eta_stage / dW of the check)."""
    c = pd.read_csv(RES / "runs" / f"q{q}" / f"seed{seed}" / "bw" / f"ms_checks_stage{stage}.csv")
    c = c[c["valid"].astype(bool)]
    return c[["local", "Delta"]].reset_index(drop=True)


def f5(x: float) -> str:
    """Five decimals."""
    return f"{x:.5f}"


def build(per: pd.DataFrame, ev: Evals, pts: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """All derived rows (CSV form) of the failure analysis."""
    seeds = {q: sorted(per[per.q == q].seed) for q in QS}
    pr = per.set_index(["q", "seed"])
    point_rows, window_rows, same_rows, hist_rows, rank_rows = [], [], [], [], []
    for p in pts:
        q, s, t, d0 = p["q"], p["seed"], p["stage"], p["d"]
        arr = ev[(q, s)][t]
        v = at_node(arr, d0)
        sd_all = arr["sd"]
        point_rows.append({**p, "edge": 2.0 * q * (t - 1), "in_band": in_band(q, t, d0), "Delta": v["delta"], "G": v["G"], "e": v["e"], "br": v["br"],
                           "dev": v["dev"], "e_minus_br": v["e"] - v["br"], "sd": v["sd"], "sd_grid_median": float(np.median(sd_all)),
                           "sd_grid_max": float(sd_all.max()), "sd_grid_max_d": float(arr["d"][int(np.argmax(sd_all))])})
        for j in np.where(np.abs(arr["d"] - d0) <= WINDOW + TOL)[0]:
            window_rows.append({"q": q, "seed": s, "stage": t, "d": float(arr["d"][j]), **{k: float(arr[k][j]) for k in STAGE_ARRAYS if k != "d"}})
        for o in seeds[q]:
            same_rows.append({"point_q": q, "point_seed": s, "stage": t, "d": d0, "seed": o, "is_failing_run": o == s,
                              "is_other_failure": bool(o != s and not pr.loc[(q, o), "passed"]), **at_node(ev[(q, o)][t], d0)})
        for r in history(q, s, t).itertuples():
            hist_rows.append({"q": q, "seed": s, "stage": t, "local": int(r.local), "Delta": float(r.Delta)})
        # ranks among the 20 runs of the q (1 = largest)
        quantities = [(p["quantity"], p["col"], True)] + [(nm, c, False) for nm, c in (
            ("eta_3/dW", "eta_T_over_dw"), ("Delta_max,2 (eta_2)/dW", "Delta_max_t2_over_dw"), ("G_max,2/dW", "G_max_t2_over_dw"),
            ("G_max,3/dW", "G_max_t3_over_dw"), ("Gmax_full/dW", "Gmax_full_over_dw")) if c != p["col"]]
        for name, c, failing in quantities:
            vals = per[per.q == q].set_index("seed")[c].astype(float)
            rank_rows.append({"q": q, "seed": s, "quantity": name, "is_failing_quantity": failing, "value": float(vals.loc[s]),
                              "rank": int((vals > vals.loc[s]).sum()) + 1, "largest_other": float(vals.drop(s).max()), "median_of_20": float(vals.median())})
        hs = {int(o): history(q, o, t) for o in seeds[q]}
        for name, fn in ((f"Delta of the last check of the stage-{t} phase (local {int(hs[s]['local'].iloc[-1])}; development tier)", lambda h: float(h["Delta"].iloc[-1])),
                         (f"mean Delta of the last 10 checks of the stage-{t} phase (development tier)", lambda h: float(h["Delta"].tail(10).mean()))):
            vals_ = {k: fn(h) for k, h in hs.items()}
            rank_rows.append({"q": q, "seed": s, "quantity": name, "is_failing_quantity": False, "value": vals_[s],
                              "rank": 1 + sum(v_ > vals_[s] for v_ in vals_.values()), "largest_other": max(v_ for k, v_ in vals_.items() if k != s),
                              "median_of_20": float(np.median(list(vals_.values())))})
    zero_rows = []
    for (q, o), st in sorted(ev.items()):
        a = st[2]
        m = (a["e"] < ZERO_EFFORT) & (np.abs(a["d"]) < 2.0 * q - TOL)
        if not m.any():
            continue
        idx = np.where(m)[0]
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        zero_rows.append({"q": q, "seed": o, "nodes": int(m.sum()), "nodes_in_band": int((np.abs(a["d"]) < 2.0 * q - TOL).sum()),
                          "stretches": "; ".join(f"[{a['d'][g[0]]:+.0f}, {a['d'][g[-1]]:+.0f}]" for g in groups),
                          "max_delta": float(a["delta"][m].max()), "max_G": float(a["G"][m].max()),
                          "G_max_t2": float(a["G"].max()), "passed": bool(pr.loc[(q, o), "passed"])})
    return {"points": pd.DataFrame(point_rows), "windows": pd.DataFrame(window_rows), "same_d": pd.DataFrame(same_rows),
            "history": pd.DataFrame(hist_rows), "ranks": pd.DataFrame(rank_rows), "zero_effort": pd.DataFrame(zero_rows)}


def tables(per: pd.DataFrame, pts: List[Dict[str, Any]], d: Dict[str, pd.DataFrame], crit: Dict[str, Any]) -> None:
    """The Markdown fragments ``t17*``."""
    ga, gf, gn = crit["gates"]["G-A"], crit["gates"]["G-F"], crit["gates"]["G-N"]
    pr = per.set_index(["q", "seed"])

    def cell(val: float, lim: float, ok: bool, nd: int = 5) -> str:
        return f"{val:.{nd}f} ({'pass' if ok else '**FAIL**'}, limit {lim:g})"

    rows = []
    for p in pts:
        r = pr.loc[(p["q"], p["seed"])]
        rows.append((p["q"], p["seed"], p["gate"],
                     cell(r.eta_T_over_dw, ga["eta_T_over_dw"], bool(r.ga_eta)), cell(r.rmse_pos_T_over_eT0, ga["rmse_pos_T_over_eT0"], bool(r.ga_rmse), 4),
                     cell(r.tail_mean_T_over_eT0, ga["tail_mean_T_over_eT0"], bool(r.ga_tail), 4),
                     cell(r.Gmax_full_over_dw, gf["Gmax_full_over_dw"], bool(r.gate_G_F)),
                     cell(abs(r.dev_eta_T_over_dw - r.eta_T_over_dw), gn["eta_T_over_dw_dev_minus_final_abs"], bool(r.gn_eta)),
                     cell(abs(r.dev_Gmax_full_over_dw - r.Gmax_full_over_dw), gn["Gmax_full_over_dw_dev_minus_final_abs"], bool(r.gn_gmax)),
                     "pass" if bool(r.passed) else "**FAIL**"))
    rt.write("t17a_failure_gates", rt.md(["q", "seed", "failing gate", "G-A eta_3/dW", "G-A RMSE_pos/e_3(0)*", "G-A tail mean/e_3(0)*", "G-F Gmax_full/dW",
                                         "G-N abs(eta_3 dev - final)/dW", "G-N abs(Gmax_full dev - final)/dW", "run"], rows))
    rows = []
    for r in d["points"].itertuples():
        band = f"in band (abs d < {r.edge:g})" if r.in_band else f"out of band (abs d >= {r.edge:g})"
        rows.append((r.q, r.seed, r.quantity, r.stage, f"{r.d:+.0f}", band, f5(r.Delta), f5(r.G), f"{r.e:.2f}", f"{r.br:.2f}", f"{r.dev:.2f}", f"{r.e_minus_br:+.2f}",
                     f"{r.sd:.4f}", f"{r.sd_grid_median:.4f}", f"{r.sd_grid_max:.4f} (d = {r.sd_grid_max_d:+.0f})"))
    rt.write("t17b_failure_point", rt.md(["q", "seed", "failing quantity", "stage t", "gap d of the maximum", "band", "Delta_t(d)/dW (one step)", "G_t(d)/dW (dynamic)",
                                        "learned mean effort e_t(d)", "best-response effort", "one-step argmax effort", "e_t(d) - best response", "normalised Beta sd at d",
                                        "median of the sd over the stage grid", "maximum of the sd over the stage grid"], rows))
    w = d["windows"]
    rows = []
    for p in pts:
        for r in w[(w.q == p["q"]) & (w.seed == p["seed"])].itertuples():
            off = r.d - p["d"]
            if abs(off) % 4 > TOL:
                continue
            mark = " **(the maximum)**" if abs(off) < TOL else ""
            rows.append((p["q"], p["seed"], r.stage, f"{r.d:+.0f}{mark}", f"{off:+.0f}", f"{r.e:.2f}", f"{r.br:.2f}", f"{r.e - r.br:+.2f}", f5(r.delta), f5(r.G), f"{r.sd:.4f}"))
    rt.write("t17c_failure_windows", rt.md(["q", "seed", "stage t", "gap d", "d - d(max)", "learned mean effort e_t(d)", "best-response effort", "e_t(d) - best response",
                                          "Delta_t(d)/dW", "G_t(d)/dW", "normalised Beta sd"], rows))
    h = d["history"]
    rows, rows2 = [], []
    for p in pts:
        sub = h[(h.q == p["q"]) & (h.seed == p["seed"])].set_index("local")["Delta"]
        last10 = sub.tail(10)
        rows.append([p["q"], p["seed"], p["stage"]] + [f5(sub.loc[x]) for x in HIST_LOCALS])
        rows2.append((p["q"], p["seed"], p["stage"], f"{sub.min():.5f} (local {int(sub.idxmin())})", f"{last10.min():.5f} to {last10.max():.5f}",
                      f"{int((sub <= 0.005).sum())} of {len(sub)}; {int((last10 <= 0.005).sum())} of 10", f"{int((sub <= 0.01).sum())} of {len(sub)}; {int((last10 <= 0.01).sum())} of 10"))
    rt.write("t17d_failure_history", rt.md(["q", "seed", "phase (stage)"] + [f"Delta at local {x}" for x in HIST_LOCALS], rows))
    rt.write("t17d2_failure_history_summary", rt.md(["q", "seed", "phase (stage)", "smallest Delta of the 64 checks (local update)",
                                                    "range of Delta over the last 10 checks (local 1375 to 1600)",
                                                    "checks with Delta <= 0.005: all 64; last 10", "checks with Delta <= 0.01: all 64; last 10"], rows2))
    rk = d["ranks"]
    rows = []
    for p in pts:
        for r in rk[(rk.q == p["q"]) & (rk.seed == p["seed"])].itertuples():
            tag = " (**failing quantity**)" if r.is_failing_quantity else ""
            rows.append((p["q"], p["seed"], r.quantity + tag, f5(r.value), f"{r.rank} of 20", f5(r.largest_other), f5(r.median_of_20)))
    rt.write("t17e_failure_rank", rt.md(["q", "seed", "quantity (final tier unless stated)", "value", "rank among the 20 runs of the q (1 = largest)",
                                        "largest value among the other 19", "median of the 20"], rows))
    z = d["zero_effort"]
    rows = []
    for r in z.itertuples():
        rows.append((r.q, r.seed, f"{r.nodes} of {r.nodes_in_band}", r.stretches, f5(r.max_delta), f5(r.max_G), f5(r.G_max_t2),
                     "pass" if r.passed else "**FAIL**"))
    rt.write("t17h_zero_effort_stage2", rt.md(["q", "seed", f"in-band stage-2 nodes with learned mean effort below {ZERO_EFFORT:g}", "stretches of d (consecutive nodes)",
                                              "largest Delta_2(d)/dW over these nodes", "largest G_2(d)/dW over these nodes", "G_max,2/dW of the run", "run gates"], rows))
    rows = []
    for q in QS:
        for p in [x for x in pts if x["q"] == q][:1]:
            vals = per[per.q == q].set_index("seed")[p["col"]].astype(float).sort_values(ascending=False)
            top = ", ".join(f"{int(sd_)}: {v_:.5f}" for sd_, v_ in vals.head(5).items())
            rows.append((q, p["quantity"], p["limit"], int((vals > p["limit"]).sum()), top, f5(float(vals.median()))))
    rt.write("t17i_failing_quantity_top5", rt.md(["q", "quantity (the failing one of the q)", "limit", "runs above the limit (of 20)", "the five largest runs (seed: value)", "median of the 20"], rows))
    s_ = d["same_d"]
    rows = []
    for p in pts:
        sub = s_[(s_.point_q == p["q"]) & (s_.point_seed == p["seed"])]
        me = sub[sub.is_failing_run].iloc[0]
        oth = sub[~sub.is_failing_run & ~sub.is_other_failure]
        lim = p["limit"]
        key = "G" if p["gate"] == "G-F" else "delta"
        n_above = int((oth[key] > lim).sum())
        note = f"{n_above} of the 18 have {'G_t(d)' if key == 'G' else 'Delta_t(d)'}/dW above {lim:g}"
        for f2 in sub[sub.is_other_failure].itertuples():
            note += f"; the other failing run of the q (seed {int(f2.seed)}) has e_t(d) {f2.e:.2f}, best response {f2.br:.2f}, G_t(d)/dW {f5(f2.G)} here"
        label = f"q = {p['q']}, stage {p['stage']}, d = {p['d']:+.0f}"
        rows.append((label, f"seed {p['seed']} (failing)", f5(me.delta), f5(me.G), f"{me.e:.2f}", f"{me.br:.2f}", f"{me.sd:.4f}", note))
        for nm, fn in (("other 18 runs: minimum", np.min), ("other 18 runs: median", np.median), ("other 18 runs: maximum", np.max)):
            rows.append(("", nm, f5(fn(oth.delta)), f5(fn(oth.G)), f"{fn(oth.e):.2f}", f"{fn(oth.br):.2f}", f"{fn(oth.sd):.4f}", ""))
    rt.write("t17f_same_d_summary", rt.md(["failing point", "runs", "Delta_t(d)/dW", "G_t(d)/dW", "learned mean effort e_t(d)", "best-response effort",
                                          "normalised Beta sd at d", "note"], rows))
    for p in pts:
        if p["q"] != 60:
            continue
        rows = []
        for r in s_[(s_.point_q == 60) & (s_.point_seed == p["seed"])].itertuples():
            tag = " (**failing run**)" if r.is_failing_run else (" (other failing run)" if r.is_other_failure else "")
            rows.append((f"{r.seed}{tag}", f"{r.e:.2f}", f"{r.br:.2f}", f"{r.dev:.2f}", f"{r.e - r.br:+.2f}", f5(r.delta), f5(r.G), f"{r.sd:.4f}"))
        rt.write(f"t17g_same_d_q60_seed{p['seed']}", rt.md(["seed", "learned mean effort e_2(d)", "best-response effort", "one-step argmax effort", "e_2(d) - best response",
                                                          "Delta_2(d)/dW", "G_2(d)/dW", "normalised Beta sd at d"], rows))


def figures(per: pd.DataFrame, ev: Evals, pts: List[Dict[str, Any]]) -> None:
    """The window figure (e_t(d) against the best response, d +- 20) and the history figure (Delta against the local update)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = {q: sorted(per[per.q == q].seed) for q in QS}
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), squeeze=False)
    for ax, p in zip(axes.ravel(), pts):
        q, s, t, d0 = p["q"], p["seed"], p["stage"], p["d"]
        first = True
        for o in seeds[q]:
            if o == s:
                continue
            a = ev[(q, o)][t]
            m = np.abs(a["d"] - d0) <= WINDOW + TOL
            ax.plot(a["d"][m], a["e"][m], color="0.75", lw=0.8, label="learned mean effort, the other 19 runs of the q" if first else None)
            first = False
        a = ev[(q, s)][t]
        m = np.abs(a["d"] - d0) <= WINDOW + TOL
        ax.plot(a["d"][m], a["e"][m], color="#d95f02", lw=2, marker="o", ms=3, label=f"learned mean effort, seed {s}")
        ax.plot(a["d"][m], a["br"][m], color="k", lw=1.5, ls="--", marker="s", ms=3, label=f"best-response effort, seed {s}")
        ax.axvline(d0, color="0.4", lw=0.8, ls=":")
        ax.set_title(f"q = {q}, seed {s}: {p['gate']}, stage {t}, d = {d0:+.0f}", fontsize=9)
        ax.set_xlabel("gap d")
        ax.set_ylabel("effort")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "bw_failures_windows.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), squeeze=False)
    for ax, p in zip(axes.ravel(), pts):
        q, s, t = p["q"], p["seed"], p["stage"]
        first = True
        for o in seeds[q]:
            if o == s:
                continue
            h = history(q, o, t)
            ax.plot(h["local"], np.maximum(h["Delta"], 1e-5), color="0.75", lw=0.8, label="the other 19 runs of the q" if first else None)
            first = False
        h = history(q, s, t)
        ax.plot(h["local"], np.maximum(h["Delta"], 1e-5), color="#d95f02", lw=1.8, label=f"seed {s}")
        if t == 3:
            ax.axhline(0.005, color="k", ls="--", lw=0.9, label="G-A limit of eta_3 (0.005)")
        ax.axvline(1200.5, color="0.5", ls=":", lw=0.8)
        ax.set_yscale("log")
        ax.set_title(f"q = {q}, seed {s}: phase of stage {t}, Delta = eta_{t}/dW", fontsize=9)
        ax.set_xlabel("local update of the phase (dotted: start of the decay)")
        ax.set_ylabel("Delta of the development check")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "bw_failures_history.png", dpi=130)
    plt.close(fig)


def main() -> int:
    """Evaluate, write the CSVs, the tables and the figures."""
    crit, _ = tm.load_criteria()
    per = pd.read_csv(RES / "analysis" / "per_run_table.csv")
    per = per[per["arm"] == "bw"].copy()
    per["passed"] = per["passed"].astype(bool)
    pts = failing_points(per, crit)
    print(f"{len(per)} bw runs, {len(pts)} failing: {[(p['q'], p['seed'], p['gate'], p['stage'], p['d']) for p in pts]}")
    ev, check = evaluate_bw(per)
    print("replay check:", check)
    if check["stored_arrays_bit_identical"] != check["runs"] or check["per_run_scalars_bit_identical"] != check["runs"]:
        print("STOP: the re-evaluation does not reproduce the stored arrays bit for bit")
        return 1
    d = build(per, ev, pts)
    OUT_D.mkdir(parents=True, exist_ok=True)
    for name, df in d.items():
        df.to_csv(OUT_D / f"{name}.csv", index=False)
    (OUT_D / "replay_check.json").write_text(json.dumps(check, indent=1) + "\n")
    tables(per, pts, d, crit)
    figures(per, ev, pts)
    print(f"wrote {OUT_D}, {FIG / 'bw_failures_windows.png'}, {FIG / 'bw_failures_history.png'} and the t17* fragments")
    return 0


if __name__ == "__main__":
    sys.exit(main())
