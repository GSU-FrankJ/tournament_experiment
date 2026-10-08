#!/usr/bin/env python3
"""The two new figures of the status pack, drawn only from evidence copies.

F1 (FIG-12): |peak error| per run, every arm of MS-R1..MS-R3 and the v2.0 confirmation.
F2 (FIG-13): the d = 0 gap split into smoothing part and remainder, per arm.

Inputs: this pack's evidence/ (MS per-run tables) and the 100526 pack's evidence/ (confirmation
decomposition, T2R:R2B-18). Nothing is run on weights.
"""
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
REPO = Path(subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"],
                                    text=True).strip())
T2R_EV = REPO / "reports/t2_refine_100526/evidence"
C_FRESH, C_R1, C_R2, C_R3 = "#444444", "#0072B2", "#E69F00", "#009E73"
SMOOTH, REMAIN = "#56B4E9", "#D55E00"


def _runs(pack: Path) -> pd.DataFrame:
    fr = []
    for r, tag in ((1, "MS-R1"), (2, "MS-R2"), (3, "MS-R3")):
        d = pd.read_csv(pack / "evidence" / ("results/ms_r%d/analysis/per_run.csv" % r))
        d["round"] = tag
        fr.append(d)
    d = pd.concat(fr, ignore_index=True, sort=False)
    d["gap_c"] = d["g2_at_0"] - d["e2_at_0"]
    d["smooth_c"] = d["g2_at_0"] - d["smoothed_e_pred_0"]
    d["rem_c"] = d["smoothed_e_pred_0"] - d["e2_at_0"]
    return d


def _arm_list() -> List[Tuple[str, str, str]]:
    """(round, arm, role) in drawing order. MS-R3 `t1` arms are bit-identical to MS-R2 `NL_*` (C-MS5): not drawn twice."""
    out = [("MS-R1", "parents_A", "comparator"), ("MS-R1", "MS_base2400", "ms_arm"), ("MS-R1", "MS_rule", "ms_arm"),
           ("MS-R1", "MS_s25a0", "ms_arm"), ("MS-R1", "MS_s25a5", "ms_arm"), ("MS-R1", "MS_s35a0", "ms_arm"),
           ("MS-R1", "MS_s35a5", "ms_arm")]
    out += [("MS-R2", a, "ms_arm") for a in ("NL_bb_s1", "NL_bb_s4", "NL_bb_s16", "NL_st_s1", "NL_st_s4", "NL_st_s16")]
    out += [("MS-R3", a, "ms_arm") for a in ("relu_bb_s1", "relu_bb_s16", "relu_st_s1", "relu_st_s16",
                                             "t10_bb_s1", "t10_bb_s16", "t10_st_s1", "t10_st_s16")]
    return out


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=130, metadata={"Software": None}, bbox_inches="tight")
    plt.close(fig)


def f1(pack: Path, figdir: Path) -> None:
    d = _runs(pack)
    cf = pd.read_csv(T2R_EV / "results/v2_refine_r2b/diag_30510/tables/tab_decomposition_all_runs.csv")
    arms = _arm_list()
    colors = {"MS-R1": C_R1, "MS-R2": C_R2, "MS-R3": C_R3}
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.2), sharex=True)
    ymax = 0.20
    for ax, q in zip(axes, (50, 60)):
        x0 = 0
        # fresh-seed block
        v = cf[cf.q == q].peak_rel_err.abs().values
        jit = (np.arange(len(v)) - (len(v) - 1) / 2) / len(v) * 0.6
        ax.scatter(x0 + jit, v, s=16, color=C_FRESH, zorder=3)
        ax.plot([x0 - 0.35, x0 + 0.35], [v.mean(), v.mean()], color="k", lw=2, zorder=4)
        ax.axvspan(-0.6, 0.6, color="#dddddd", alpha=0.6, zorder=0)
        for i, (rnd, arm, role) in enumerate(arms, start=1):
            x = d[(d["round"] == rnd) & (d.arm == arm) & (d.q == q) & (d.role == role)].sort_values("seed")
            v = x.stage2_peak_rel_err_abs.values
            jit = (np.arange(len(v)) - (len(v) - 1) / 2) / max(len(v), 1) * 0.6
            vc = np.minimum(v, ymax)
            ax.scatter(i + jit[v <= ymax], v[v <= ymax], s=16, color=colors[rnd], zorder=3)
            if (v > ymax).any():
                ax.scatter(i + jit[v > ymax], vc[v > ymax], s=40, marker="^", color=colors[rnd], edgecolor="k", zorder=4)
            ax.plot([i - 0.35, i + 0.35], [v.mean(), v.mean()], color="k", lw=2, zorder=4)
        ax.axhline(0.05, color="#CC0000", ls="--", lw=1)
        ax.text(len(arms) + 1.6, 0.052, "0.05", color="#CC0000", fontsize=8, ha="right")
        ax.set_ylim(0, ymax * 1.05)
        ax.set_ylabel("|peak error| at d = 0 (q = %d)" % q)
        ax.grid(axis="y", alpha=0.3)
    labels = ["v2.0 fresh\n(30501-30520)\nn=20"] + [("%s" % a).replace("_", "\n", 1) for (_, a, _) in arms]
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=90, fontsize=7.5)
    axes[0].set_title("F1. |peak error| per run. Grey: v2.0 confirmation on FRESH seeds (n = 20 per q). Coloured: DEVELOPMENT seeds "
                      "10501-10510 (n = 10 per arm and q).\nBlue MS-R1, orange MS-R2, green MS-R3; black bar = mean; triangles = clipped at 0.20 "
                      "(collapsed `relu` run); `parents_A` = v2.0 on the development seeds", fontsize=9)
    _save(fig, figdir / "FIG-12_F1_abs_peak_per_run.png")


def f2(pack: Path, figdir: Path) -> None:
    d = _runs(pack)
    arms = [("MS-R3", "rehearsal_v2_0", "comparator")] + _arm_list()[1:]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.2), sharex=True)
    cap = 6.0
    for ax, q in zip(axes, (50, 60)):
        for i, (rnd, arm, role) in enumerate(arms):
            x = d[(d["round"] == rnd) & (d.arm == arm) & (d.q == q) & (d.role == role)]
            s, r = x.smooth_c.mean(), x.rem_c.mean()
            ax.bar(i, s, color=SMOOTH, label="smoothing part" if i == 0 else None, zorder=3)
            top = s + r
            if top > cap:
                ax.bar(i, cap - s, bottom=s, color=REMAIN, zorder=3)
                ax.text(i, cap + 0.05, "%.1f" % top, ha="center", fontsize=8)
            elif r >= 0:
                ax.bar(i, r, bottom=s, color=REMAIN, label="remainder" if i == 0 else None, zorder=3)
            else:
                ax.bar(i, r, bottom=s, color=REMAIN, zorder=3)
            ax.plot([i - 0.4, i + 0.4], [x.gap_c.median()] * 2, color="k", lw=1.6, zorder=4,
                    label="median gap" if i == 0 else None)
        ax.set_ylim(0, cap + 0.7)
        ax.set_ylabel("effort units at d = 0 (q = %d)" % q)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    names = [("%s" % a).replace("_", "\n", 1) for (_, a, _) in arms]
    axes[1].set_xticks(range(len(arms)))
    axes[1].set_xticklabels(names, rotation=90, fontsize=7.5)
    axes[0].set_title("F2. The d = 0 gap e2*(0) - e_hat_2(0): mean smoothing part (blue) + mean remainder (orange), development seeds "
                      "10501-10510 (n = 10 per arm and q).\n`rehearsal_v2_0` = v2.0; MS-R3 `t1` arms equal MS-R2 `NL_*` arms (C-MS5) and "
                      "are not drawn twice; bars above 6 are clipped and labelled with the total", fontsize=9)
    _save(fig, figdir / "FIG-13_F2_gap_decomposition.png")


def generate(pack: Path, figdir: Path) -> None:
    pack, figdir = Path(pack), Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    f1(pack, figdir)
    f2(pack, figdir)


if __name__ == "__main__":
    generate(Path(sys.argv[1]) if len(sys.argv) > 1 else PACK, Path(sys.argv[2]) if len(sys.argv) > 2 else PACK / "figures")
