#!/usr/bin/env python3
"""F3: T = 3 effort policies of the T3-1.1 confirmation (arm bwA, 40 runs per q), q = 50 | q = 60.

Reads ONLY ``data/F3.csv`` (written by ``extract_F3.py``) and writes ``figures/F3.pdf`` and ``figures/F3.png``.
No data transformation: the region edges 2q come from the q column, axis limits from the data.

Rows (top to bottom): stage 1 mean effort e_1(0) (one point per run), stage 2 mean effort, stage 3 mean effort
(with the closed-form equilibrium), state-visitation distribution at stage 2, at stage 3 (median and 10-90 % band
over the 40 runs). Seed 80502 at q = 60 (the only failed run) is drawn in the mark colour.

Usage:  .venv/bin/python paper/multistage_rev/scripts/F3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

import _style as S  # noqa: E402

QS = (50, 60)
FAILED_SEED = 80502
RUN_KW = dict(color=S.C_RUN, lw=0.5, alpha=0.4, zorder=2)
MED_KW = dict(color=S.C_MEDIAN, lw=1.8, zorder=5)
FAIL_KW = dict(color=S.C_MARK, lw=1.1, zorder=4)
TIE_KW = dict(color="black", lw=0.7, ls=":", zorder=1)


def runs_of(sub: pd.DataFrame, series: str):
    """Yield ``(seed, d, value)`` per run of the given series, sorted by seed and d."""
    s = sub[sub.series == series]
    for seed, g in s.groupby("seed"):
        g = g.sort_values("d")
        yield int(seed), g.d.to_numpy(), g.value.to_numpy()


def curve(sub: pd.DataFrame, series: str):
    """``(d, value)`` of a one-curve series (median, p10, p90, closed form), sorted by d."""
    g = sub[sub.series == series].sort_values("d")
    return g.d.to_numpy(), g.value.to_numpy()


def shade_stage3(ax, q: float, half: float) -> None:
    """Positive-effort region (|d| < 2q) and theoretical zero-effort tails (|d| >= 2q) of stage 3."""
    ax.axvspan(-2 * q, 2 * q, color=S.C_POS, alpha=0.25, lw=0, zorder=0)
    ax.axvspan(-half, -2 * q, color=S.C_TAILS, alpha=0.22, lw=0, zorder=0)
    ax.axvspan(2 * q, half, color=S.C_TAILS, alpha=0.22, lw=0, zorder=0)


def main() -> None:
    """Draw the figure from data/F3.csv."""
    df = pd.read_csv(S.DATA_DIR / "F3.csv", float_precision="round_trip")
    fig = S.new_figure(10.1)
    gs = fig.add_gridspec(5, 2, height_ratios=[0.75, 1.35, 1.5, 1.1, 1.1])
    axes = {}
    for j in range(2):
        for i in range(5):
            axes[(i, j)] = fig.add_subplot(gs[i, j])
    # shared x within a column and stage (whole D_2 resp. D_3); shared y across the q columns
    for j in range(2):
        axes[(0, j)].sharex(axes[(1, j)])
        axes[(3, j)].sharex(axes[(1, j)])
        axes[(4, j)].sharex(axes[(2, j)])
    for i in range(1, 5):
        axes[(i, 1)].sharey(axes[(i, 0)])

    for j, q in enumerate(QS):
        d_q = df[df.q == q]
        s1 = d_q[d_q.panel == "stage1_effort"]
        s2 = d_q[d_q.panel == "stage2_effort"]
        s3 = d_q[d_q.panel == "stage3_effort"]
        cf = d_q[d_q.panel == "closed_form_equilibrium"]
        v2 = d_q[d_q.panel == "stage2_visitation"]
        v3 = d_q[d_q.panel == "stage3_visitation"]
        half2 = float(s2.d.abs().max())
        half3 = float(s3.d.abs().max())

        # ---- row 0: stage 1 (the single state d = 0)
        ax = axes[(0, j)]
        ax.axvline(0.0, **TIE_KW)
        for series in ("run", "failed_run"):
            sub = s1[s1.series == series]
            if series == "run":
                ax.scatter(np.zeros(len(sub)), sub.value, s=16, facecolors="none", edgecolors=S.C_RUN,
                           linewidths=0.6, zorder=3)
            elif len(sub):
                ax.plot([0.0], sub.value.to_numpy(), marker="D", ms=5, ls="none", mfc=S.C_MARK, mec=S.C_MARK, zorder=6)
                ax.annotate(f"seed {FAILED_SEED}", xy=(0.0, float(sub.value.iloc[0])), xytext=(11, 0),
                            textcoords="offset points", va="center", ha="left", color=S.C_MARK)
        med = float(s1[s1.series == "median"].value.iloc[0])
        ax.plot([0.0], [med], marker="_", ms=26, mew=2.0, ls="none", color=S.C_MEDIAN, zorder=5)
        ax.margins(y=0.16)
        ax.set_ylabel("stage 1\nmean effort\n$e_1(0)$")
        ax.text(0.02, 0.93, rf"$\mathbf{{q = {q}}}$", transform=ax.transAxes, va="top", ha="left")

        # ---- rows 1 and 2: stage 2 and stage 3 mean effort
        for i, (sub, half) in ((1, (s2, half2)), (2, (s3, half3))):
            ax = axes[(i, j)]
            ax.axvline(0.0, **TIE_KW)
            if i == 2:
                shade_stage3(ax, q, half)
            for seed, d, v in runs_of(sub, "run"):
                ax.plot(d, v, **RUN_KW)
            for seed, d, v in runs_of(sub, "failed_run"):
                ax.plot(d, v, **FAIL_KW)
            ax.plot(*curve(sub, "median"), **MED_KW)
            if i == 2:
                ax.plot(*curve(cf, "closed_form_equilibrium"), color="black", lw=1.0, ls=(0, (3, 2)), zorder=6)
            ax.set_xlim(-half, half)
            ax.text(0.02, 0.93, rf"$\mathbf{{q = {q}}}$", transform=ax.transAxes, va="top", ha="left")
        axes[(1, j)].set_ylabel("stage 2\nmean effort $e_2(d)$")
        axes[(2, j)].set_ylabel("stage 3\nmean effort $e_3(d)$")

        # ---- rows 3 and 4: state-visitation distribution (median and 10-90 % band)
        for i, (sub, half) in ((3, (v2, half2)), (4, (v3, half3))):
            ax = axes[(i, j)]
            ax.axvline(0.0, **TIE_KW)
            if i == 4:
                shade_stage3(ax, q, half)
            d10, p10 = curve(sub, "p10")
            d90, p90 = curve(sub, "p90")
            ax.fill_between(d10, p10, p90, color=S.C_RUN, alpha=0.35, lw=0, zorder=2)
            ax.plot(*curve(sub, "median"), **{**MED_KW, "lw": 0.8})
            ax.set_xlim(-half, half)
            ax.set_xlabel("score gap $d$")
            ax.text(0.02, 0.93, rf"$\mathbf{{q = {q}}}$", transform=ax.transAxes, va="top", ha="left")
        axes[(3, j)].set_ylabel("stage 2 state-visitation\ndistribution\n(mass per node)")
        axes[(4, j)].set_ylabel("stage 3 state-visitation\ndistribution\n(mass per node)")

    # right column: no repeated y tick labels / y labels for the shared-y rows
    for i in range(1, 5):
        axes[(i, 1)].tick_params(labelleft=False)
        axes[(i, 1)].set_ylabel("")
    # every effort panel carries its own x label: mean effort is shown against the score gap d (the stage-1
    # panel is the single value e_1(0) at d = 0 on the same axis)
    for i in range(0, 3):
        for j in range(2):
            axes[(i, j)].set_xlabel("score gap $d$")
            axes[(i, j)].tick_params(labelbottom=True)
    for i in range(5):
        for j in range(2):
            axes[(i, j)].xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10], integer=True))
    for j in range(2):
        axes[(1, j)].set_ylim(bottom=0.0)
        axes[(2, j)].set_ylim(bottom=0.0)
        axes[(3, j)].set_ylim(bottom=0.0)
        axes[(4, j)].set_ylim(bottom=0.0)

    handles = [
        Line2D([], [], color="black", lw=1.0, ls=(0, (3, 2)), label="closed-form equilibrium (stage 3)"),
        Line2D([], [], **{**RUN_KW, "alpha": 0.7}, label="learned mean effort (one line per run)"),
        Line2D([], [], **{**MED_KW, "lw": 1.8}, label="median over the 40 runs"),
        Line2D([], [], **FAIL_KW, label=f"seed {FAILED_SEED} (failed run)"),
        Patch(facecolor=S.C_RUN, alpha=0.35, lw=0, label="10-90 % band over the 40 runs"),
        Line2D([], [], **{**TIE_KW}, label="tie point ($d=0$)"),
        # stage-1 marks (row 0): hollow circle, bold bar, vermilion diamond
        Line2D([], [], marker="o", ms=4, mfc="none", mec=S.C_RUN, mew=0.6, ls="none",
               label="stage 1: a run's $e_1(0)$"),
        Line2D([], [], marker="_", ms=14, mew=2.0, color=S.C_MEDIAN, ls="none",
               label="stage 1: median over the 40 runs"),
        Line2D([], [], marker="D", ms=5, mfc=S.C_MARK, mec=S.C_MARK, ls="none",
               label=f"stage 1: seed {FAILED_SEED} (failed run)"),
        Patch(facecolor=S.C_POS, alpha=0.25, lw=0,
              label="positive-effort region ($|d| < 2q$ at the terminal stage)"),
        Patch(facecolor=S.C_TAILS, alpha=0.22, lw=0,
              label="theoretical zero-effort tails ($|d| \\geq 2q$ at the terminal stage)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, handlelength=2.2, columnspacing=1.5)
    S.save(fig, "F3")


if __name__ == "__main__":
    main()
