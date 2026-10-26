"""F2: T = 2 verifier calibration (final tier).  Reads ``data/F2.csv`` only; writes figures/F2.pdf and F2.png.

Per q (two panels side by side, log axis), for each policy type: eta_2/DeltaW (square), G_max,full/DeltaW
(circle) and root exploitability (diamond), offset within each category; the 20 frozen v2.0 policies are
jittered by seed rank.  Values below the floor are drawn at the floor with a hollow down-pointing marker.

Usage: .venv/bin/python paper/multistage_rev/scripts/F2.py   (any cwd)
"""

from __future__ import annotations

import _style as S

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# metric -> (x offset within a category, colour, marker, legend label)
METRIC_STYLE = {
    "eta_2_over_dw": (-0.27, S.C_DELTA, "s", r"$\eta_2/\Delta W$"),
    "Gmax_full_over_dw": (0.0, S.C_G, "o", r"$G_{\max,\mathrm{full}}/\Delta W$"),
    "EXP_root_over_dw": (0.27, S.SKY, "D", "root exploitability"),
}
GATE_STYLE = {"eta_2_over_dw": (S.C_DELTA, (0, (4, 2))), "Gmax_full_over_dw": (S.C_G, (0, (1, 1.5)))}
TICK_LABEL = {
    "closed-form equilibrium": "closed-form\nequilibrium",
    "zero effort": "zero effort",
    "frozen v2.0 policies": "frozen v2.0\npolicies",
    "M1: 0.8 x closed-form equilibrium": "M1: $0.8\\times$ closed-form\nequilibrium",
    "M2: 1.2 x closed-form equilibrium": "M2: $1.2\\times$ closed-form\nequilibrium",
    "M3: stage 2 constant at g2(0)": "M3: stage 2 constant\nat $g_2(0)$",
}
YLIM = (2e-9, 3.0)


def _ytick_label(y: float, _pos: int, floor: float) -> str:
    """Decade tick label; the floor tick reads <= 10^-8."""
    e = int(round(np.log10(y)))
    return r"$\leq 10^{%d}$" % e if abs(y - floor) < 1e-3 * floor else r"$10^{%d}$" % e


def draw_panel(ax, df: pd.DataFrame, q: int, floor: float, categories: list) -> None:
    """One q panel."""
    d = df[df["q"] == q]
    ax.set_xlim(-0.6, len(categories) - 0.4)
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    # shading of the new diagnostic evaluations
    new_x = sorted(d[d["kind"] == "new diagnostic evaluation"]["x_order"].unique())
    lo, hi = min(new_x) - 0.5, max(new_x) + 0.5
    ax.axvspan(lo, hi, color=S.GREY, alpha=0.10, lw=0, zorder=0)
    ax.annotate("", xy=(lo + 0.08, 0.20), xytext=(hi - 0.08, 0.20), xycoords=ax.get_xaxis_transform(),
                textcoords=ax.get_xaxis_transform(),
                arrowprops=dict(arrowstyle="|-|,widthA=0.25,widthB=0.25", color="0.25", lw=0.7, shrinkA=0, shrinkB=0))
    ax.text((lo + hi) / 2, 0.22, "new diagnostic\nevaluation", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=S.MIN_FONT, color="0.15", linespacing=1.1)
    # floor and gate limits
    ax.axhline(floor, color="0.45", lw=0.6, ls=(0, (2, 2)), zorder=1)
    for metric, (col, ls) in GATE_STYLE.items():
        g = d[(d["kind"] == "gate limit") & (d["metric"] == metric)]["value"].iloc[0]
        ax.axhline(g, color=col, lw=0.9, ls=ls, zorder=1.5)
    # points
    pts = d[d["kind"].isin(["stored record", "new diagnostic evaluation"])]
    for metric, (off, col, mk, _lab) in METRIC_STYLE.items():
        m = pts[pts["metric"] == metric]
        for _, r in m.iterrows():
            x = r["x_order"] + off
            if r["policy_type"] == "frozen v2.0 policies":
                x += (r["seed"] - 30501 - 9.5) / 19.0 * 0.22
                size = 2.6
            else:
                size = 5.5
            if r["at_floor"]:
                ax.plot([x], [r["plotted_value"]], ls="none", marker="v", ms=6.5, mfc="none", mec=col, mew=1.1,
                        zorder=4, clip_on=False)
            else:
                ax.plot([x], [r["plotted_value"]], ls="none", marker=mk, ms=size, mfc=col, mec="white",
                        mew=0.3, alpha=0.95, zorder=3)
            if r["marked"]:
                ax.plot([x], [r["plotted_value"]], ls="none", marker="o", ms=8, mfc="none", mec=S.C_MARK,
                        mew=1.3, zorder=5)
    # failed run label
    mk_rows = pts[pts["marked"] & (pts["metric"] == "eta_2_over_dw")]
    if len(mk_rows):
        r = mk_rows.iloc[0]
        ax.annotate("seed 30510", xy=(r["x_order"] + METRIC_STYLE["eta_2_over_dw"][0], r["plotted_value"]),
                    xytext=(r["x_order"] - 0.55, 0.05), textcoords="data", ha="center", va="bottom",
                    fontsize=S.MIN_FONT, color=S.C_MARK,
                    arrowprops=dict(arrowstyle="-", color=S.C_MARK, lw=0.8, shrinkA=1, shrinkB=5))
    ax.text(0.02, 0.97, f"$q = {q}$", transform=ax.transAxes, ha="left", va="top", fontsize=S.MIN_FONT)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([TICK_LABEL.get(c, c) for c in categories], rotation=60, ha="right",
                       rotation_mode="anchor")
    ax.yaxis.set_major_locator(FixedLocator([10.0 ** e for e in range(-8, 2)]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: _ytick_label(y, p, floor)))
    ax.tick_params(axis="x", length=2.5)


def main() -> None:
    """Read data/F2.csv and write figures/F2.{pdf,png}."""
    df = pd.read_csv(S.DATA_DIR / "F2.csv", float_precision="round_trip")
    floor = float(df[df["kind"] == "plot floor"]["value"].iloc[0])
    cats = (df[df["x_order"].notna()].drop_duplicates("policy_type").sort_values("x_order")["policy_type"].tolist())
    fig = S.new_figure(5.4)
    axes = fig.subplots(1, 2, sharey=True)
    for ax, q in zip(axes, (50, 60)):
        draw_panel(ax, df, q, floor, cats)
    axes[0].set_ylabel("$\\eta_2/\\Delta W$ and $G_{\\max,\\mathrm{full}}/\\Delta W$,\n"
                       "root exploitability (units of $\\Delta W$)")
    handles = [Line2D([], [], ls="none", marker=mk, ms=5.5, mfc=col, mec="white", mew=0.3, label=lab)
               for (_m, (_o, col, mk, lab)) in METRIC_STYLE.items()]
    handles += [
        Line2D([], [], ls="none", marker="v", ms=6.5, mfc="none", mec="0.3", mew=1.1,
               label=r"$\leq 10^{-8}$ (drawn at the floor)"),
        Line2D([], [], ls="none", marker="o", ms=8, mfc="none", mec=S.C_MARK, mew=1.3,
               label="failed run (seed 30510, $q = 50$)"),
        Line2D([], [], color=S.C_DELTA, lw=0.9, ls=GATE_STYLE["eta_2_over_dw"][1],
               label=r"gate limit $\eta_2/\Delta W \leq 0.005$"),
        Line2D([], [], color=S.C_G, lw=0.9, ls=GATE_STYLE["Gmax_full_over_dw"][1],
               label=r"gate limit $G_{\max,\mathrm{full}}/\Delta W \leq 0.01$"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, handletextpad=0.4, columnspacing=1.4)
    S.save(fig, "F2")


if __name__ == "__main__":
    main()
