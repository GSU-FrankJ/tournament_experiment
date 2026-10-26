"""F5: method comparison (T3-R1, paired). G_max,full/DeltaW of bw, jt, jt_es per seed, q = 50 | q = 60.

Reads ``data/F5.csv`` only (plus ``_style``) and writes ``figures/F5.pdf`` and ``figures/F5.png``.
Run from any directory:  /home/fjiang4/tournament_experiment/.venv/bin/python paper/multistage_rev/scripts/F5.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import _style as S  # noqa: E402

ARMS = ["bw", "jt", "jt_es"]


def main() -> None:
    """Draw the two panels (q = 50, q = 60) from data/F5.csv."""
    df = pd.read_csv(S.DATA_DIR / "F5.csv", float_precision="round_trip")
    gate = df[df["arm"] == "gate_limit"].set_index("q")["value"]
    runs = df[df["arm"].isin(ARMS)]
    qs = sorted(runs["q"].unique())

    fig = S.new_figure(3.1)
    axes = fig.subplots(1, len(qs), sharey=True)
    xs = {a: i for i, a in enumerate(ARMS)}
    for ax, q in zip(axes, qs):
        d = runs[runs["q"] == q]
        for seed, g in d.groupby("seed"):
            g = g.set_index("arm").loc[ARMS]
            ax.plot([xs[a] for a in ARMS], g["value"].to_numpy(), color=S.C_RUN, alpha=0.35, lw=0.6, zorder=2)
        for a in ARMS:
            ax.scatter([xs[a]] * int((d["arm"] == a).sum()), d[d["arm"] == a]["value"], s=9, color=S.C_RUN,
                       edgecolor="white", linewidth=0.3, zorder=3)
        ax.axhline(gate[q], color=S.BLACK, ls="--", lw=0.9, zorder=1)
        ax.set_yscale("log")
        ax.set_ylim(2e-4, 1.0)
        ax.set_xlim(-0.45, 2.45)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels(ARMS)
        ax.set_xlabel("arm")
        ax.text(0.03, 0.97, f"q = {q}", transform=ax.transAxes, ha="left", va="top")
        ax.legend(handles=[Line2D([], [], color=S.BLACK, ls="--", lw=0.9,
                                  label=r"gate limit $G_{\mathrm{max,full}}/\Delta W \leq 0.01$")],
                  loc="lower right", handlelength=1.6, borderaxespad=0.2)
    axes[0].set_ylabel("full-domain dynamic best-response gain\n" r"$G_{\mathrm{max,full}}/\Delta W$")
    for p in S.save(fig, "F5"):
        print("wrote", p)


if __name__ == "__main__":
    main()
