#!/usr/bin/env python3
"""Phase 1 of the one-stage ablation: raw PPO vs MC-BR polish, deterministic referee.

Consumes:
  results/two_players/convergence/ppo_q{35,55}.0_seed{42..46}_r5_sampled_convergence.json  [JSON]
  results/one_stage_ablation/a2_polish_per_seed.json      (arm A2, per-seed polish)
  results/one_stage_ablation/phase0_polish_probe.json     (arm A3, controls c=0 / c=50)

Produces:
  results/one_stage_ablation/ablation_results.json
  results/one_stage_ablation/figures/{F1_landing_map,F2_exp_bars,F3_slopegraph}.png

Measurement = the deterministic referee (tools/one_stage_referee.py).
Comparison exhibit = the SHIPPED legacy MC estimator
(run/run_two_players.eval_exploitability) driven through a degenerate point-policy
adapter, R=5 reps. Nothing in agents/ envs/ run/ utils/ is modified.

IMPORT ORDER: utils.mc_br_polish must be imported before anything that pulls in
utils.prob (its module-level import guard asserts utils.prob is not loaded).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.mc_br_polish as _mp  # noqa: E402,F401  (import guard: must precede utils.prob)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.one_stage_mc_adapter import PointPolicyAgent  # noqa: E402
from tools.one_stage_referee import (  # noqa: E402
    BOUNDS, DW, K, W_H, W_L, br_analytic, br_slopes, e_star, exp_det, exp_ucb,
)
from run.run_two_players import eval_exploitability  # noqa: E402

QS = (35.0, 55.0)
SEEDS = (42, 43, 44, 45, 46)
R_REPS = 5
MC_M = 8192
GRID_CFG = {"stage_a_step": 5.0, "stage_b_radius": 15.0, "stage_b_step": 1.0,
            "stage_c_radius": 3.0, "stage_c_step": 0.25}
ETA, N_ROUNDS, N_AVG = 0.4, 320, 200
EPS_HIST_ABS = 0.03            # historical one-stage stop threshold (ABSOLUTE payoff units)
OUT_DIR = "results/one_stage_ablation"


# ------------------------------- inputs -------------------------------------
def raw_per_seed(q: float) -> list:
    """Per-seed raw effort reconstructed from the final Beta (alpha, beta)."""
    lo, hi = BOUNDS
    rows = []
    for s in SEEDS:
        p = f"results/two_players/convergence/ppo_q{q:g}.0_seed{s}_r5_sampled_convergence.json"
        d = json.load(open(p))
        a, b = d["alpha_mean"][-1], d["beta_mean"][-1]
        rows.append({"seed": s, "alpha": a, "beta": b, "e": lo + (hi - lo) * a / (a + b),
                     "json_final_effort": d["final"]["effort"],
                     "json_exploit_max": d["final_exploit_max"]})
    return rows


def load_arms(q: float) -> dict:
    a2 = json.load(open(f"{OUT_DIR}/a2_polish_per_seed.json"))
    a3 = json.load(open(f"{OUT_DIR}/phase0_polish_probe.json"))
    pol = [r for r in a2["rows"] if r["q"] == q]
    ctl = [r for r in a3["rows"] if r["q"] == q and r["start"] in (0.0, 50.0)]
    return {"polish": pol, "controls": ctl}


# ------------------------------ legacy MC -----------------------------------
def legacy_mc(effort: float, q: float, reps: int = R_REPS) -> dict:
    """SHIPPED eval_exploitability on a deterministic effort via the point adapter."""
    vals = []
    for r in range(reps):
        ag = PointPolicyAgent(effort, BOUNDS)
        out = eval_exploitability(ag, q=q, effort_bounds=BOUNDS, M=MC_M,
                                  grid_cfg=GRID_CFG, seed=100 + 7 * r,
                                  w_h=W_H, w_l=W_L, k=K)
        vals.append(float(out["exploitability"]))
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
            "mean_over_dw": float(np.mean(vals)) / DW,
            "sd_over_dw": float(np.std(vals, ddof=1)) / DW, "vals": vals}


# ------------------------- A5 / P11 attribution probe -----------------------
def analytic_damped_iteration(start: float, q: float) -> dict:
    """Same damped iteration as the polish, but with the ANALYTIC BR (no MC).

    Isolates whether the fixed-point offset comes from damping/averaging (it does
    not) or from the MC-BR estimator itself.
    """
    e = float(start)
    hist = []
    for _ in range(N_ROUNDS):
        e = (1.0 - ETA) * e + ETA * br_analytic(e, q)
        hist.append(e)
    land = float(np.mean(hist[-N_AVG:]))
    return {"start": start, "landing": land, "e_star": e_star(q),
            "abs_err": abs(land - e_star(q))}


def main() -> int:
    os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)
    res = {"params": {"w_h": W_H, "w_l": W_L, "dw": DW, "k": K, "bounds": list(BOUNDS)},
           "conventions": {
               "rel_err_pct": "(e - e*)/e* * 100  (paper table)",
               "RE_T3": "AE/(1+e*)  (cross-paper consistency, stored here)",
               "eps_historical": "0.03 ABSOLUTE payoff units = 0.0085714 /DW "
                                 "(run/run_two_players.py:1425); two-stage's 0.03 is /DW",
               "legacy_mc": "shipped run_two_players.eval_exploitability via point-policy "
                            "adapter; reading is a LEVEL (upward-biased max-statistic), "
                            "mean +/- sd over R=5 reps",
               "e_fp": "mean of ALL polish landings in the cell (A2 n=5 + A3 n=2 = 7)",
           },
           "cells": {}}

    for q in QS:
        es = e_star(q)
        raw = raw_per_seed(q)
        arms = load_arms(q)
        pol = arms["polish"]
        ctl = arms["controls"]

        # ---- e_fp: mean of ALL landings in the cell (A2 5 + A3 2 = 7) ----
        all_land = [r["landing"] for r in pol] + [r["landing"] for r in ctl]
        e_fp = float(np.mean(all_land))
        e_fp_sd = float(np.std(all_land, ddof=1))
        spread = float(max(all_land) - min(all_land))

        # ---- referee EXP per seed, per arm ----
        for r in raw:
            d = exp_det(r["e"], q)
            r["exp_det"] = d["exp"]
            r["exp_det_over_dw"] = d["exp"] / DW
            r["exp_ucb_over_dw"] = exp_ucb(r["e"], q)["exp_ucb"] / DW
            r["signed_err"] = r["e"] - es
            r["abs_err"] = abs(r["e"] - es)
            r["rel_pct"] = (r["e"] - es) / es * 100.0
            r["RE_T3"] = r["abs_err"] / (1.0 + es)
        for r in pol + ctl:
            d = exp_det(r["landing"], q)
            r["exp_det"] = d["exp"]
            r["exp_det_over_dw"] = d["exp"] / DW
            r["exp_ucb_over_dw"] = exp_ucb(r["landing"], q)["exp_ucb"] / DW
            r["signed_err"] = r["landing"] - es
            r["abs_err"] = abs(r["landing"] - es)
            r["rel_pct"] = (r["landing"] - es) / es * 100.0
            r["RE_T3"] = r["abs_err"] / (1.0 + es)

        raw_mean = float(np.mean([r["e"] for r in raw]))
        raw_sd = float(np.std([r["e"] for r in raw], ddof=1))
        pol_mean = float(np.mean([r["landing"] for r in pol]))
        pol_sd = float(np.std([r["landing"] for r in pol], ddof=1))

        # ---- legacy MC (shipped) at representative profiles ----
        mc = {
            "e_star": legacy_mc(es, q),          # P10
            "raw_mean": legacy_mc(raw_mean, q),
            "e_fp": legacy_mc(e_fp, q),
        }
        for r in ctl:
            mc[f"control_c{int(r['start'])}"] = legacy_mc(r["landing"], q)

        # ---- gates ----
        pol_by_seed = {r["seed"]: r for r in pol}
        c1_wins = sum(1 for r in raw if pol_by_seed[r["seed"]]["abs_err"] < r["abs_err"])
        c2_wins = sum(1 for r in raw if pol_by_seed[r["seed"]]["exp_det"] < r["exp_det"])
        ratios = [r["exp_det"] / pol_by_seed[r["seed"]]["exp_det"] for r in raw
                  if pol_by_seed[r["seed"]]["exp_det"] > 0]
        c2_median_ratio = float(np.median(ratios))
        ref_ratio = exp_det(raw_mean, q)["exp"] / exp_det(e_fp, q)["exp"]
        d_mc = abs(mc["raw_mean"]["mean"] - mc["e_fp"]["mean"])
        rep_sd = max(mc["raw_mean"]["sd"], mc["e_fp"]["sd"])
        c4_referee_sep = ref_ratio >= 5.0
        c4_mc_cannot_sep = d_mc < 2.0 * rep_sd
        gates = {
            "C1": {"wins": c1_wins, "n": 5, "passed": c1_wins >= 4},
            "C2": {"wins": c2_wins, "n": 5, "median_ratio": c2_median_ratio,
                   "passed": c2_wins >= 4 and c2_median_ratio >= 5.0},
            "C4": {"referee_ratio": ref_ratio, "referee_separates": c4_referee_sep,
                   "delta_mc": d_mc, "two_rep_sd": 2.0 * rep_sd,
                   "mc_cannot_separate": c4_mc_cannot_sep,
                   "passed": bool(c4_referee_sep and c4_mc_cannot_sep)},
        }

        # ---- seeds better than the solver floor ----
        floor = abs(e_fp - es)
        better = [r["seed"] for r in raw if r["abs_err"] < floor]

        # ---- A5 / P11 ----
        a5 = [analytic_damped_iteration(s, q) for s in (raw_mean, 0.0, 50.0)]
        sl = br_slopes(q)
        damped = {"below": 1 - ETA + ETA * sl["slope_below"],
                  "above": 1 - ETA + ETA * sl["slope_above"]}

        res["cells"][str(q)] = {
            "e_star": es, "raw": raw, "raw_mean": raw_mean, "raw_sd": raw_sd,
            "polish": pol, "polish_mean": pol_mean, "polish_sd": pol_sd,
            "controls": ctl, "e_fp": e_fp, "e_fp_sd": e_fp_sd,
            "e_fp_offset": e_fp - es, "landing_spread_all": spread,
            "exp_det_at_raw_mean_over_dw": exp_det(raw_mean, q)["exp"] / DW,
            "exp_det_at_e_fp_over_dw": exp_det(e_fp, q)["exp"] / DW,
            "legacy_mc": mc, "gates": gates,
            "seeds_better_than_floor": better, "solver_floor_abs": floor,
            "A4_br_1step_at_raw_mean": br_analytic(raw_mean, q),
            "A5_analytic_damped": a5,
            "br_slopes": sl, "damped_slopes": damped,
        }

    with open(f"{OUT_DIR}/ablation_results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"[saved] {OUT_DIR}/ablation_results.json")
    make_figures(res)
    report(res)
    return 0


# -------------------------------- figures -----------------------------------
def make_figures(res: dict) -> None:
    # F1 — landing map
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, q in zip(axes, QS):
        c = res["cells"][str(q)]
        es, e_fp = c["e_star"], c["e_fp"]
        cg = np.linspace(0, 100, 801)
        br = np.array([br_analytic(x, q) for x in cg])
        ax.plot(cg, cg, color="grey", ls="--", lw=0.9, alpha=0.55, label="y = x")
        ax.plot(cg, br, lw=2.2, color="tab:blue", label="analytic BR(c)  [A4]")
        ax.axvline(es, color="green", ls=":", lw=1.6, label=f"e* = {es:.2f}")
        starts = [r["start_raw"] for r in c["polish"]] + [r["start"] for r in c["controls"]]
        # Curved arrows off the diagonal so they never read as a second y=x line.
        for s0 in starts:
            ax.annotate("", xy=(e_fp, e_fp), xytext=(s0, s0),
                        arrowprops=dict(arrowstyle="-|>", color="crimson", alpha=0.55,
                                        lw=1.3, shrinkA=3, shrinkB=6,
                                        connectionstyle="arc3,rad=0.32"))
        ax.scatter(starts, starts, s=34, facecolor="white", edgecolor="dimgrey",
                   zorder=5, label="polish starts (on the diagonal)")
        ax.scatter([e_fp], [e_fp], s=190, color="crimson", zorder=7, marker="*",
                   label=f"$e_{{fp}}$ = {e_fp:.2f}  (all 7 landings)")
        if abs(c["br_slopes"]["slope_above"]) > 1:
            ax.axvspan(es, 100, color="orange", alpha=0.10)
            ax.text(0.62, 0.06, "expansive side\n|BR′| = "
                    f"{abs(c['br_slopes']['slope_above']):.2f} > 1",
                    transform=ax.transAxes, fontsize=7.5, color="darkorange")
        d = c["damped_slopes"]
        ax.set_title(f"q={q:g}  damped-iteration slopes: below {d['below']:+.3f}, "
                     f"above {d['above']:+.3f}\n(both |·|<1 ⇒ contractive; "
                     f"raw BR slope above = {c['br_slopes']['slope_above']:+.3f})", fontsize=9)
        ax.set_xlabel("start / opponent effort c")
        ax.set_ylabel("BR(c)  /  landing")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25)
    fig.suptitle("F1 — one-step BR map vs. the polish's init-independent fixed point", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figures/F1_landing_map.png", dpi=150)
    plt.close(fig)

    # F2 — EXP bars (log)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    for ax, q in zip(axes, QS):
        c = res["cells"][str(q)]
        labels, vals = ["e*", "raw PPO", "PPO+polish"], [
            max(exp_det(c["e_star"], q)["exp"], 1e-12) / DW,
            c["exp_det_at_raw_mean_over_dw"], c["exp_det_at_e_fp_over_dw"]]
        for r in c["controls"]:
            labels.append(f"polish c={int(r['start'])}")
            vals.append(r["exp_det_over_dw"])
        vals = [max(v, 1e-12) for v in vals]
        ax.bar(labels, vals, color=["green", "tab:blue", "crimson", "grey", "grey"][:len(vals)])
        ax.set_yscale("log")
        mcm = c["legacy_mc"]["e_star"]["mean_over_dw"]
        mcs = c["legacy_mc"]["e_star"]["sd_over_dw"]
        ax.axhspan(max(abs(mcm) - 2 * mcs, 1e-12), abs(mcm) + 2 * mcs, color="orange",
                   alpha=0.22, label="legacy MC reading at e* (|mean| ± 2sd)")
        ax.axhline(EPS_HIST_ABS / DW, color="red", ls="--", lw=1.4,
                   label=f"historical ε = 0.03 abs = {EPS_HIST_ABS/DW:.2e} /ΔW")
        ax.set_ylabel("EXP / ΔW  (log)")
        ax.set_title(f"q={q:g}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.suptitle("F2 — deterministic-referee exploitability by arm (log scale)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figures/F2_exp_bars.png", dpi=150)
    plt.close(fig)

    # F3 — per-seed slopegraph
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for j, q in enumerate(QS):
        c = res["cells"][str(q)]
        pol_by_seed = {r["seed"]: r for r in c["polish"]}
        for i, (key, lab) in enumerate((("abs_err", "|error|  (effort units)"),
                                        ("exp_det_over_dw", "EXP / ΔW"))):
            ax = axes[i][j]
            for r in c["raw"]:
                y0, y1 = r[key], pol_by_seed[r["seed"]][key]
                ax.plot([0, 1], [y0, y1], "-o", ms=4, alpha=0.8,
                        color="crimson" if y1 > y0 else "tab:blue",
                        label=f"seed {r['seed']}")
                ax.annotate(f"{r['seed']}", (0, y0), textcoords="offset points",
                            xytext=(-16, -3), fontsize=7)
            if key == "exp_det_over_dw":
                ax.set_yscale("log")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["raw PPO", "PPO+polish"])
            ax.set_ylabel(lab)
            ax.set_title(f"q={q:g}  (red = polish made it worse)", fontsize=9)
            ax.grid(alpha=0.25, axis="y")
    fig.suptitle("F3 — per-seed raw → polished (polished column collapses: init-independence)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figures/F3_slopegraph.png", dpi=150)
    plt.close(fig)
    print(f"[saved] {OUT_DIR}/figures/F1_landing_map.png, F2_exp_bars.png, F3_slopegraph.png")


# -------------------------------- report ------------------------------------
def report(res: dict) -> None:
    for q in QS:
        c = res["cells"][str(q)]
        print("\n" + "=" * 100)
        print(f"q={q:g}   e*={c['e_star']:.4f}")
        print("=" * 100)
        print(f"  raw   mean={c['raw_mean']:.4f} sd={c['raw_sd']:.4f}   "
              f"EXP_det/DW@mean={c['exp_det_at_raw_mean_over_dw']:.3e}")
        print(f"  polish mean={c['polish_mean']:.4f} sd={c['polish_sd']:.4f}  "
              f"e_fp={c['e_fp']:.4f} sd={c['e_fp_sd']:.4f} offset={c['e_fp_offset']:+.4f} "
              f"spread(all 7)={c['landing_spread_all']:.4f}")
        print(f"  EXP_det/DW@e_fp={c['exp_det_at_e_fp_over_dw']:.3e}")
        print(f"  legacy MC (shipped, R=5) /DW:  e*={c['legacy_mc']['e_star']['mean_over_dw']:+.3e}"
              f"±{c['legacy_mc']['e_star']['sd_over_dw']:.1e}  "
              f"raw={c['legacy_mc']['raw_mean']['mean_over_dw']:+.3e}"
              f"±{c['legacy_mc']['raw_mean']['sd_over_dw']:.1e}  "
              f"e_fp={c['legacy_mc']['e_fp']['mean_over_dw']:+.3e}"
              f"±{c['legacy_mc']['e_fp']['sd_over_dw']:.1e}")
        g = c["gates"]
        print(f"  GATES: C1 {g['C1']['wins']}/5 -> {'PASS' if g['C1']['passed'] else 'FAIL'} | "
              f"C2 {g['C2']['wins']}/5 median_ratio={g['C2']['median_ratio']:.1f}x -> "
              f"{'PASS' if g['C2']['passed'] else 'FAIL'} | "
              f"C4 ref_ratio={g['C4']['referee_ratio']:.1f}x sep={g['C4']['referee_separates']}, "
              f"|dMC|={g['C4']['delta_mc']:.2e} vs 2sd={g['C4']['two_rep_sd']:.2e} "
              f"mc_cannot_sep={g['C4']['mc_cannot_separate']} -> "
              f"{'PASS' if g['C4']['passed'] else 'FAIL'}")
        print(f"  seeds better than solver floor ({c['solver_floor_abs']:.3f}): "
              f"{c['seeds_better_than_floor']} ({len(c['seeds_better_than_floor'])}/5)")
        print(f"  A5/P11 analytic-BR damped iteration -> "
              + ", ".join(f"start {a['start']:.2f}: land {a['landing']:.6f} "
                          f"(|land-e*|={a['abs_err']:.2e})" for a in c["A5_analytic_damped"]))


if __name__ == "__main__":
    sys.exit(main())
