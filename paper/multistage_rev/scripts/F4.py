"""F4: T = 3 best-response diagnostics of three T3-1.1 confirmation runs (final verifier tier).

Reads ONLY ``data/F4.csv`` (written by ``extract_F4.py``) and the style module ``_style``; writes
``figures/F4.pdf`` and ``figures/F4.png``. Runnable from any directory.

Columns: the passing run with the median G_max,full at q = 50, the passing run with the median G_max,full at
q = 60, and the failed run (q = 60, seed 80502). Rows: stage 2 effort, stage 2 gains, stage 3 effort, stage 3
gains, stage 1 (points at the tie point d = 0: effort | gains). The BR-reachable region is shaded; the gains are
on a log axis with a floor at 1e-8 (values below it are drawn at the floor); the y-ranges are common to all
columns. The G_max,full marker is drawn for the failed run (seed 80502, stage 2) only: ``data/F4.csv`` holds
exactly one ``gmax_full_marker`` row.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import _style as S  # noqa: E402

DATA = S.DATA_DIR / "F4.csv"
RUN_ORDER = ["q50_median_passing", "q60_median_passing", "q60_failed"]
FLOOR = 1e-8
G_YLIM = (4e-9, 0.3)   # upper limit just above the largest plotted value (0.0586)
E_YLIM = (0.0, 75.0)
G_TICKS = [1e-8, 1e-6, 1e-4, 1e-2]
G_TICKLABELS = [r"$\leq 10^{-8}$", r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$"]
REACH_ALPHA = 0.30
GMAX_TEX = r"$G_{\max,\mathrm{full}}$"
GMAX_LEGEND = GMAX_TEX + " (seed 80502, stage 2)"
HEIGHT_IN = 9.6

# Page geometry in inches (x from the left edge, y from the top edge). The three run columns are equally wide and
# equally spaced, and the q brackets, the run headers and every panel of a column share the x-extent of the column.
LEFT_IN, RIGHT_IN, COL_GAP_IN = 0.80, 0.14, 0.34
COL_W_IN = (S.WIDTH_IN - LEFT_IN - RIGHT_IN - 2 * COL_GAP_IN) / 3
STAGE1_W_IN = 0.50                     # width of each of the two small stage-1 panels of a column
ROWS_IN = {                            # (top, height)
    "q": (0.05, 0.30), "run": (0.40, 0.42),
    "e2": (0.88, 1.38), "g2": (2.34, 1.38), "e3": (4.30, 1.38), "g3": (5.76, 1.38), "s1": (7.78, 0.78),
}


def series(df: pd.DataFrame, label: str, stage: int, quantity: str):
    """``(d, value)`` of one quantity of one run and stage, sorted by d."""
    s = df[(df.run_label == label) & (df.stage == stage) & (df.quantity == quantity)].sort_values("d")
    return s.d.to_numpy(dtype=float), s.value.to_numpy(dtype=float)


def floored(v: np.ndarray) -> np.ndarray:
    """Values below the floor are drawn at the floor."""
    return np.maximum(v, FLOOR)


def style_axes(ax) -> None:
    """Light horizontal grid for reading the common y-ranges across columns."""
    ax.grid(axis="y", color="0.88", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def effort_panel(ax, df, label, stage) -> None:
    """Learned mean effort against the dynamic best-response effort, with the BR-reachable region shaded."""
    d, e = series(df, label, stage, "e_hat")
    _, a = series(df, label, stage, "a_br")
    _, r = series(df, label, stage, "reach")
    ax.fill_between(d, 0, 1, where=r > 0.5, transform=ax.get_xaxis_transform(), step="mid", color=S.C_REACH,
                    alpha=REACH_ALPHA, lw=0, zorder=1)
    ax.plot(d, a, color=S.C_BR, lw=2.2, solid_capstyle="butt", zorder=3)
    ax.plot(d, e, color=S.C_RUN, lw=0.9, zorder=4)
    ax.set_ylim(*E_YLIM)
    ax.set_xlim(d.min(), d.max())
    style_axes(ax)


def gain_panel(ax, df, label, stage, marker) -> None:
    """State-wise dynamic gain G_t(d)/dW and one-step deviation gain Delta_t(d)/dW on a log axis with a floor."""
    d, g = series(df, label, stage, "G_over_dw")
    _, de = series(df, label, stage, "delta_over_dw")
    _, r = series(df, label, stage, "reach")
    ax.fill_between(d, 0, 1, where=r > 0.5, transform=ax.get_xaxis_transform(), step="mid", color=S.C_REACH,
                    alpha=REACH_ALPHA, lw=0, zorder=1)
    ax.plot(d, floored(g), color=S.C_G, lw=1.6, zorder=3)
    ax.plot(d, floored(de), color=S.C_DELTA, lw=1.0, ls=(0, (3, 2)), zorder=4)
    ax.axhline(FLOOR, color="0.45", lw=0.6, ls=":", zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(*G_YLIM)
    ax.set_yticks(G_TICKS)
    ax.set_yticklabels(G_TICKLABELS)
    ax.minorticks_off()
    ax.set_xlim(d.min(), d.max())
    style_axes(ax)
    if marker is not None and int(marker.stage) == stage:
        md, mv = float(marker.d), float(marker.value)
        ax.plot([md], [mv], marker="o", ms=7, mfc="none", mec=S.C_MARK, mew=1.5, ls="none", zorder=6)
        ax.annotate(GMAX_TEX, xy=(md, mv), xytext=(8, 0), textcoords="offset points", ha="left", va="center",
                    color=S.C_MARK, zorder=7)


def stage1_panels(ax_e, ax_g, df, label) -> None:
    """Stage 1 as points at the tie point d = 0: efforts (left) and gains (right)."""
    _, e = series(df, label, 1, "e_hat")
    _, a = series(df, label, 1, "a_br")
    _, g = series(df, label, 1, "G_over_dw")
    _, de = series(df, label, 1, "delta_over_dw")
    ax_e.plot([0], a, marker="D", ms=9, mfc="none", mec=S.C_BR, mew=1.6, ls="none", zorder=3)
    ax_e.plot([0], e, marker="o", ms=4.5, color=S.C_RUN, ls="none", zorder=4)
    ax_e.set_ylim(*E_YLIM)
    ax_g.plot([0], floored(de), marker="s", ms=8, mfc="none", mec=S.C_DELTA, mew=1.4, ls="none", zorder=3)
    ax_g.plot([0], floored(g), marker="o", ms=4.5, color=S.C_G, ls="none", zorder=4)
    ax_g.axhline(FLOOR, color="0.45", lw=0.6, ls=":", zorder=2)
    ax_g.set_yscale("log")
    ax_g.set_ylim(*G_YLIM)
    ax_g.set_yticks(G_TICKS)
    ax_g.set_yticklabels(G_TICKLABELS)
    ax_g.minorticks_off()
    for ax in (ax_e, ax_g):
        ax.set_xlim(-1, 1)
        ax.set_xticks([0])
        ax.set_xticklabels([r"$d = 0$"])
        style_axes(ax)


def main() -> None:
    """Draw F4 from ``data/F4.csv``."""
    df = pd.read_csv(DATA, float_precision="round_trip")
    labels = [lab for lab in RUN_ORDER if lab in set(df.run_label)]
    assert labels == RUN_ORDER, f"expected the runs {RUN_ORDER}, found {sorted(set(df.run_label))}"
    info = {lab: df[df.run_label == lab].iloc[0] for lab in labels}
    mk = df[df.quantity == "gmax_full_marker"]
    assert len(mk) == 1 and mk.iloc[0].run_label == "q60_failed" and int(mk.iloc[0].stage) == 2, \
        "F4.csv must hold exactly one gmax_full_marker row: seed 80502, stage 2"
    markers = {lab: (mk.iloc[0] if lab == "q60_failed" else None) for lab in labels}

    fig = S.new_figure(HEIGHT_IN)
    fig.set_layout_engine("none")      # explicit panel positions (see the geometry constants above)

    def add_ax(c, row, x_off=0.0, width=None, ncols=1, **kw):
        """Axes at the x-extent of run column ``c`` (or of ``ncols`` columns from ``c``), in the row ``row``."""
        top, height = ROWS_IN[row]
        left = LEFT_IN + c * (COL_W_IN + COL_GAP_IN) + x_off
        width = ncols * COL_W_IN + (ncols - 1) * COL_GAP_IN if width is None else width
        return fig.add_axes([left / S.WIDTH_IN, 1 - (top + height) / HEIGHT_IN, width / S.WIDTH_IN,
                             height / HEIGHT_IN], **kw)

    # headers: q groups and the run of each column
    q_of = [int(info[lab].q) for lab in labels]
    groups = {}
    for c, q in enumerate(q_of):
        groups.setdefault(q, []).append(c)
    for q, cols in groups.items():
        ax = add_ax(cols[0], "q", ncols=len(cols))
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, f"q = {q}", ha="center", va="center", fontweight="bold")
        ax.plot([0, 1], [0.02, 0.02], color="0.4", lw=0.8, clip_on=False)
    for c, lab in enumerate(labels):
        ax = add_ax(c, "run")
        ax.axis("off")
        seed = int(info[lab].seed)
        txt = (f"failed run,\nseed {seed}" if lab.endswith("failed")
               else f"passing run with median\n{GMAX_TEX}, seed {seed}")
        ax.text(0.5, 0.5, txt, ha="center", va="center", color=S.C_MARK if lab.endswith("failed") else "black")

    # stages 2 and 3
    for stage in (2, 3):
        for c, lab in enumerate(labels):
            ax_e = add_ax(c, f"e{stage}")
            ax_g = add_ax(c, f"g{stage}", sharex=ax_e)
            effort_panel(ax_e, df, lab, stage)
            gain_panel(ax_g, df, lab, stage, markers[lab])
            ax_e.tick_params(labelbottom=False)
            ax_g.set_xlabel("score gap d")
            half = float(series(df, lab, stage, "e_hat")[0].max())
            ax_g.set_xticks([x for x in (-400, -200, 0, 200, 400) if abs(x) <= half])
            if c == 0:
                ax_e.set_ylabel(f"Stage {stage}: effort")
                ax_g.set_ylabel(f"Stage {stage}: gain / $\\Delta W$")
            else:
                ax_e.tick_params(labelleft=False)
                ax_g.tick_params(labelleft=False)
            if stage == 3:
                ax_g.text(0.5, 0.97, r"$G_3(d) = \Delta_3(d)$ at every $d$", transform=ax_g.transAxes,
                          ha="center", va="top")

    # stage 1: the effort panel at the left and the gain panel at the right of the column
    for c, lab in enumerate(labels):
        ax_e = add_ax(c, "s1", width=STAGE1_W_IN)
        ax_g = add_ax(c, "s1", x_off=COL_W_IN - STAGE1_W_IN, width=STAGE1_W_IN)
        stage1_panels(ax_e, ax_g, df, lab)
        if c == 0:
            ax_e.set_ylabel("Stage 1: effort")
        else:
            ax_e.tick_params(labelleft=False)
        ax_g.set_ylabel("gain / $\\Delta W$")

    handles = [
        Line2D([], [], color=S.C_RUN, lw=0.9, marker="o", ms=4.5, label="learned mean effort"),
        Line2D([], [], color=S.C_BR, lw=2.2, marker="D", ms=7, mfc="none", mew=1.4,
               label="dynamic best-response effort"),
        Line2D([], [], color=S.C_G, lw=1.6, marker="o", ms=4.5,
               label=r"state-wise dynamic gain $G_t(d)/\Delta W$" "\n" r"(stage 1: root exploitability $G_1(0)/\Delta W$)"),
        Line2D([], [], color=S.C_DELTA, lw=1.0, ls=(0, (3, 2)), marker="s", ms=6, mfc="none", mew=1.2,
               label=r"one-step deviation gain $\Delta_t(d)/\Delta W$"),
        Patch(facecolor=S.C_REACH, alpha=REACH_ALPHA, edgecolor="none", label="BR-reachable region"),
        Line2D([], [], color=S.C_MARK, lw=0, marker="o", ms=7, mfc="none", mew=1.5, label=GMAX_LEGEND),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, columnspacing=1.5,
               handlelength=2.4)
    S.save(fig, "F4")


if __name__ == "__main__":
    main()
