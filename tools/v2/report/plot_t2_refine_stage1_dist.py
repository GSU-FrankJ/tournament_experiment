#!/usr/bin/env python3
"""FG-01: the stage-1 error |e1_hat(0) - e1*| / e1* of the fresh-seed confirmations, v1.1 against v2.0.

Reads the pack copies of the two per-run tables (v1.1: seeds 20501-20520, v2.0: seeds 30501-30520; column ``s1``
= the absolute relative stage-1 error of the final-tier end-of-B last iterate) and draws, per q, every run as a
point (deterministic horizontal offsets, no random numbers) with the median as a bar and the G-S threshold 0.05
and the S1 threshold 0.10 as lines. Nothing is estimated: every point is a value of ``s1``. The PNG carries no
timestamp or software tag, so a rebuild is byte-identical.

Usage: python tools/v2/report/plot_t2_refine_stage1_dist.py <v1.1 per_run.csv> <v2.0 per_run.csv> <out.png>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"v1.1": "#c0504d", "v2.0": "#1f77b4"}


def make(out_png: Path, v11_csv: Path, v20_csv: Path) -> None:
    """Draw the figure from the two per-run tables.

    Args:
        out_png: output PNG path.
        v11_csv: ``per_run.csv`` of the v1.1 confirmation (column ``s1``).
        v20_csv: ``per_run.csv`` of the v2.0 confirmation (column ``s1``).
    """
    tabs = {"v1.1": pd.read_csv(v11_csv), "v2.0": pd.read_csv(v20_csv)}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), sharey=True)
    for ax, q in zip(axes, (50, 60)):
        for k, (name, df) in enumerate(tabs.items()):
            d = df[(df["q"] == q) & (df["status"] == "completed")].sort_values("s1")
            y = d["s1"].to_numpy(dtype=float)
            n = len(y)
            x = k + (np.arange(n) - (n - 1) / 2.0) * 0.025          # deterministic spread by rank
            ax.scatter(x, y, s=22, color=COLORS[name], alpha=0.85, edgecolor="none", zorder=3)
            med = float(np.median(y))
            ax.hlines(med, k - 0.3, k + 0.3, color="black", lw=2.0, zorder=4)
            ax.text(k + 0.32, med, f"median {med:.4f}", va="center", fontsize=8)
            ax.text(k, -0.012, f"n = {n}, max {y.max():.4f}", ha="center", fontsize=8)
        ax.axhline(0.05, color="grey", ls="--", lw=1.0, zorder=1)
        ax.axhline(0.10, color="grey", ls=":", lw=1.0, zorder=1)
        ax.text(1.62, 0.052, "G-S 0.05", fontsize=8, color="grey", ha="right")
        ax.text(1.62, 0.102, "S1 0.10", fontsize=8, color="grey", ha="right")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["v1.1\n(seeds 20501-20520)", "v2.0\n(seeds 30501-30520)"])
        ax.set_xlim(-0.5, 1.7)
        ax.set_title(f"q = {q}")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("|stage-1 error| = |e1_hat(0) - e1*| / e1*")
    axes[0].set_ylim(-0.02, 0.17)
    fig.suptitle("Stage-1 error on fresh confirmation seeds: protocol v1.1 vs v2.0", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, metadata={"Software": None})
    plt.close(fig)


def main(argv: List[str]) -> int:
    """CLI entry."""
    if len(argv) != 3:
        print(__doc__)
        return 2
    make(Path(argv[2]), Path(argv[0]), Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
