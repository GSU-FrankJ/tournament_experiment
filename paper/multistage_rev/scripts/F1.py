"""F1 plot (T = 2 analytical recovery): reads ``data/F1.csv`` only and writes ``figures/F1.pdf`` / ``F1.png``.

Columns q = 50 | q = 60. Top row: learned mean effort of stage 2 against the score gap d over the whole D_2 (one thin
line per run, the failed run in a distinct style, the closed-form equilibrium dashed), with the positive-effort region
(|d| < 2q) and the theoretical zero-effort tails (|d| >= 2q) shaded and the tie point (d = 0) marked. Bottom row:
learned mean effort e_1(0) of stage 1, one point per run in seed order, against the closed-form equilibrium.
The region edges 2q are derived from the q column only.

Usage (any cwd):  .venv/bin/python paper/multistage_rev/scripts/F1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402

import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

QS = (50, 60)
FAILED_LABEL = "seed 30510 at $q$ = 50 (failed run)"
Y2_LOWER = -2.0   # stage-2 y lower limit (the closed form is exactly 0 in the theoretical zero-effort tails)


def main() -> None:
    """Draw and save F1."""
    df = pd.read_csv(S.DATA_DIR / "F1.csv", float_precision="round_trip")
    fig = S.new_figure(5.6)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.35])
    ymax2 = float(df[df.panel == "stage2"].value.max()) * 1.06

    for j, q in enumerate(QS):
        s2 = df[(df.panel == "stage2") & (df.q == q)]
        s1 = df[(df.panel == "stage1") & (df.q == q)]

        # ---------------- stage 2: mean effort against the score gap d on D_2
        ax = fig.add_subplot(gs[0, j])
        xmin, xmax = float(s2.d.min()), float(s2.d.max())
        ax.axvspan(xmin, -2 * q, color=S.C_TAILS, alpha=0.22, lw=0, zorder=0)
        ax.axvspan(2 * q, xmax, color=S.C_TAILS, alpha=0.22, lw=0, zorder=0)
        ax.axvspan(-2 * q, 2 * q, color=S.C_POS, alpha=0.28, lw=0, zorder=0)
        ax.axvline(0.0, color=S.BLACK, lw=0.8, ls=":", zorder=1)
        for seed, g in s2[s2.series == "run"].groupby("seed"):
            ax.plot(g.d, g.value, color=S.C_RUN, lw=0.6, alpha=0.65, zorder=2)
        for seed, g in s2[s2.series == "failed_run"].groupby("seed"):
            ax.plot(g.d, g.value, color=S.C_MARK, lw=1.4, ls="-.", zorder=4)
        th = s2[s2.series == "closed_form_equilibrium"]
        ax.plot(th.d, th.value, color=S.C_THEORY, lw=1.2, ls="--", zorder=5)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(Y2_LOWER, ymax2)   # lower limit below 0 so the dashed zero line of the tails is visible
        ax.set_yticks([t for t in range(0, int(ymax2) + 1, 10)])   # ticks start at 0
        ax.set_xticks([xmin, -2 * q, 0, 2 * q, xmax])
        ax.set_xticklabels([f"{x:g}".replace("-", "\u2212") for x in (xmin, -2 * q, 0, 2 * q, xmax)])
        ax.set_xlabel("score gap $d$")
        if j == 0:
            ax.set_ylabel("mean effort")
        else:
            ax.tick_params(labelleft=False)
        ax.text(0.02, 0.96, f"$q$ = {q}, stage 2", transform=ax.transAxes, va="top", ha="left")

        # ---------------- stage 1: e_1(0) of each run (points in seed order) against the closed form
        ax1 = fig.add_subplot(gs[1, j])
        runs = s1[s1.series == "run"]
        bad = s1[s1.series == "failed_run"]
        g1 = float(s1[s1.series == "closed_form_equilibrium"].value.iloc[0])
        ax1.axhline(g1, color=S.C_THEORY, lw=1.2, ls="--", zorder=2)
        ax1.scatter(runs.seed, runs.value, s=16, color=S.C_RUN, marker="o", zorder=3, linewidths=0)
        ax1.scatter(bad.seed, bad.value, s=46, color=S.C_MARK, marker="X", zorder=4, linewidths=0)
        lo, hi = float(min(s1.value.min(), g1)), float(max(s1.value.max(), g1))
        pad = 0.12 * (hi - lo)
        ax1.set_ylim(lo - pad, hi + 0.85 * (hi - lo) + pad)
        ax1.set_xlim(float(s1.seed.min()) - 1, float(s1.seed.max()) + 1)
        ax1.set_xticks([30501, 30505, 30510, 30515, 30520])
        ax1.set_xlabel("seed (points in seed order)")
        ax1.set_ylabel("mean effort")
        ax1.text(0.02, 0.96, f"$q$ = {q}, stage 1 ($d$ = 0)", transform=ax1.transAxes, va="top", ha="left")

    handles = [
        Line2D([], [], color=S.C_RUN, lw=0.8, marker="o", ms=3.5, label="learned mean effort (one line or point per run)"),
        Line2D([], [], color=S.C_THEORY, lw=1.2, ls="--", label="closed-form equilibrium"),
        Line2D([], [], color=S.C_MARK, lw=1.4, ls="-.", marker="X", ms=6, label=FAILED_LABEL),
        Line2D([], [], color=S.BLACK, lw=0.8, ls=":", label="tie point ($d$ = 0)"),
        Patch(facecolor=S.C_POS, alpha=0.28, label="positive-effort region ($|d| < 2q$ at the terminal stage)"),
        Patch(facecolor=S.C_TAILS, alpha=0.22, label="theoretical zero-effort tails ($|d| \\geq 2q$ at the terminal stage)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, columnspacing=1.5, handlelength=2.2)
    for p in S.save(fig, "F1"):
        print("wrote", p)


if __name__ == "__main__":
    main()
