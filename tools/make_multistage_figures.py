"""Generate the multi-stage paper figures (plan section 6, Figures 1-5).

Reads the committed convergence JSONs and produces publication-style PDF+PNG:

  Fig 1  Two-stage closed-form vs TEL-PPO effort (recovery)          [T=2]
  Fig 2  Verifier calibration: EXP of closed-form / TEL-PPO / bad     [T=2]
  Fig 3  Three-stage learned effort functions e_hat_t(d)             [T=3]
  Fig 4  Three-stage best-response vs learned effort                 [T=3]
  Fig 5  Three-stage state-wise deviation gaps Delta_t(d)            [T=3]

Style matches the one-stage paper (TrueType-embedded PDF, no top/right
spines, TEL-PPO orange, theory/reference red dashed). Output:
``paper/multistage/figures/``.

Run:
    python tools/make_multistage_figures.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils.theory_multistage import g1_two_stage, g2_two_stage  # noqa: E402

OUT_DIR = os.path.join("paper", "multistage", "figures")
CONV = os.path.join("results", "multi_stage", "convergence")
WH, WL, K, EBAR, Q = 6.0, 2.0, 1.0 / 3500.0, 100.0, 50.0
DW = WH - WL

# Colors (Wong-ish, matching the one-stage paper conventions)
C_LEARNED = "#E69F00"   # orange  (TEL-PPO learned)
C_REF = "#D55E00"       # vermillion (closed form / best response reference)
C_STAGE = ["#56B4E9", "#0072B2", "#000000", "#009E73", "#CC79A7"]  # per stage


def _style() -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.fontsize": 9, "figure.dpi": 100, "savefig.dpi": 300,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _save(fig, name: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def load_T(T: int) -> List[dict]:
    tag = f"gateT{T}"
    return [json.load(open(f)) for f in
            sorted(glob.glob(os.path.join(CONV, f"ms_T{T}_q50_seed*_{tag}_convergence.json")))]


def stack_curves(runs: List[dict], stage: int, key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (d_grid, mean, min, max) of a per-stage curve across seeds."""
    d = np.array(runs[0]["effort_curves"]["d_grid"])
    arr = np.array([r["effort_curves"]["stages"][str(stage)][key] for r in runs])
    return d, arr.mean(0), arr.min(0), arr.max(0)


# ---------------------------------------------------------------------------
# Figure 1: two-stage recovery
# ---------------------------------------------------------------------------

def fig1_recovery() -> None:
    runs = load_T(2)
    if not runs:
        print("  [skip Fig1] no T=2 runs")
        return
    d_probe = np.array(runs[0]["final_effort"]["stage2_probe_d"])
    e2 = np.array([r["final_effort"]["stage2_learned"] for r in runs])   # (seeds, 5)
    e1 = np.array([r["final_effort"]["stage1_at_0"] for r in runs])
    g1 = g1_two_stage(Q, WH, WL, K)
    dd = np.linspace(-2 * Q, 2 * Q, 400)
    g2 = g2_two_stage(dd, Q, WH, WL, K, EBAR)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(dd, g2, "--", color=C_REF, lw=2, label=r"closed form $e^*_{2,CF}(d)$")
    ax.errorbar(d_probe, e2.mean(0), yerr=e2.std(0), fmt="o", color=C_LEARNED,
                ms=7, capsize=3, lw=1.5, label=r"TEL-PPO $\hat{e}_2(d)$ (5 seeds)")
    ax.axhline(g1, color="gray", ls=":", lw=1)
    ax.annotate(rf"$g_1={g1:.1f}$; $\hat{{e}}_1(0)={e1.mean():.1f}\pm{e1.std():.1f}$",
                xy=(0.02, 0.06), xycoords="axes fraction", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    ax.set_xlabel("score gap $d$")
    ax.set_ylabel("stage-2 effort")
    ax.set_title("Two-stage equilibrium recovery ($T=2$, $q=50$)")
    ax.legend(loc="upper right")
    _save(fig, "F1_two_stage_recovery")


# ---------------------------------------------------------------------------
# Figure 2: verifier calibration (closed form / TEL-PPO / bad policies)
# ---------------------------------------------------------------------------

def fig2_calibration() -> None:
    from utils.dp_verifier import verify
    g1 = g1_two_stage(Q, WH, WL, K)
    e_one = DW / (4.0 * Q * K)

    def cf(t, d):
        d = np.asarray(d, float)
        return g2_two_stage(d, Q, WH, WL, K, EBAR) if t >= 2 else np.full_like(d, g1)

    def const_lo(t, d): return np.full_like(np.asarray(d, float), 5.0)
    def const_hi(t, d): return np.full_like(np.asarray(d, float), EBAR)
    def one_stage(t, d): return np.full_like(np.asarray(d, float), e_one)
    def no_gap(t, d):
        d = np.asarray(d, float)
        return (np.full_like(d, float(g2_two_stage(np.asarray(0.0), Q, WH, WL, K, EBAR)))
                if t >= 2 else np.full_like(d, g1))

    policies = [("closed\nform", cf), ("TEL-PPO", None), ("const\nlow", const_lo),
                ("const\nhigh", const_hi), ("one-stage\nrepeat", one_stage),
                ("no-gap\nstage2", no_gap)]
    runs = load_T(2)
    telppo_exp = float(np.mean([r["final_eval"]["exp_over_dw"] for r in runs])) if runs else 0.0

    names, vals = [], []
    for nm, pol in policies:
        if pol is None:
            vals.append(telppo_exp)
        else:
            r = verify(pol, w_h=WH, w_l=WL, k=K, q=Q, T=2, e_bar=EBAR)
            vals.append(r.exp_over_dw)
        names.append(nm)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = [C_REF, C_LEARNED] + ["#999999"] * 4
    bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="black", lw=0.6)
    ax.axhline(0.03, color="red", ls="--", lw=1, label=r"$\epsilon/\Delta W=0.03$ threshold")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(r"exploitability $EXP/\Delta W$")
    ax.set_title("Verifier calibration & falsification ($T=2$)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                va="bottom", fontsize=7)
    ax.legend(loc="upper left")
    _save(fig, "F2_verifier_calibration")


# ---------------------------------------------------------------------------
# Figure 3: three-stage learned effort functions
# ---------------------------------------------------------------------------

def fig3_effort_functions() -> None:
    runs = load_T(3)
    if not runs:
        print("  [skip Fig3] no T=3 runs")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    lim = 2.5 * Q
    for t in (1, 2, 3):
        d, m, lo, hi = stack_curves(runs, t, "learned")
        msk = np.abs(d) <= lim
        ax.plot(d[msk], m[msk], color=C_STAGE[t - 1], lw=2, label=rf"$\hat{{e}}_{t}(d)$")
        ax.fill_between(d[msk], lo[msk], hi[msk], color=C_STAGE[t - 1], alpha=0.15)
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("score gap $d$ (player $i$'s lead)")
    ax.set_ylabel("effort")
    ax.set_title("Three-stage learned effort functions ($T=3$, 5 seeds)")
    ax.legend(loc="upper right", title="stage")
    _save(fig, "F3_three_stage_effort")


# ---------------------------------------------------------------------------
# Figure 4: three-stage BR vs learned
# ---------------------------------------------------------------------------

def fig4_br_vs_learned() -> None:
    runs = load_T(3)
    if not runs:
        print("  [skip Fig4] no T=3 runs")
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    lim = 2.5 * Q
    for t, ax in zip((1, 2, 3), axes):
        d, ml, _, _ = stack_curves(runs, t, "learned")
        _, mb, _, _ = stack_curves(runs, t, "br")
        msk = np.abs(d) <= lim
        ax.plot(d[msk], ml[msk], color=C_LEARNED, lw=2, label=r"learned $\hat{e}_t$")
        ax.plot(d[msk], mb[msk], "--", color=C_REF, lw=1.8, label=r"best response $BR_t$")
        ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set_title(f"stage {t}")
        ax.set_xlabel("score gap $d$")
    axes[0].set_ylabel("effort")
    axes[0].legend(loc="upper right")
    fig.suptitle("Three-stage best response vs learned effort ($T=3$)", y=1.02)
    _save(fig, "F4_three_stage_br_vs_learned")


# ---------------------------------------------------------------------------
# Figure 5: three-stage deviation gaps
# ---------------------------------------------------------------------------

def fig5_deviation_gaps() -> None:
    runs = load_T(3)
    if not runs:
        print("  [skip Fig5] no T=3 runs")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    lim = 2.5 * Q
    # on-path support band (stage-2 on-path std) as a shaded region
    d0, p_mean, _, _ = stack_curves(runs, 2, "onpath_dist")
    for t in (1, 2, 3):
        d, m, lo, hi = stack_curves(runs, t, "delta")
        msk = np.abs(d) <= lim
        ax.plot(d[msk], m[msk], color=C_STAGE[t - 1], lw=2, label=rf"$\Delta_{t}(d)$")
        ax.fill_between(d[msk], lo[msk], hi[msk], color=C_STAGE[t - 1], alpha=0.12)
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("score gap $d$")
    ax.set_ylabel(r"one-step deviation gap $\Delta_t(d)$")
    ax.set_title("Three-stage state-wise deviation gaps ($T=3$, 5 seeds)")
    ax.legend(loc="upper right", title="stage")
    _save(fig, "F5_three_stage_deviation_gaps")


def main() -> int:
    _style()
    print(f"generating multi-stage figures -> {OUT_DIR}")
    fig1_recovery()
    fig2_calibration()
    fig3_effort_functions()
    fig4_br_vs_learned()
    fig5_deviation_gaps()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
