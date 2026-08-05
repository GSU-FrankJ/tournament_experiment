#!/usr/bin/env python3
"""Regenerate paper/figures/equilibrium_recovery_dotplot.png from Claim-B polished values.

Thin convenience wrapper: the mapping from ``polish_per_seed_all.json`` to the plot's
``final_override`` columns lives in ``paper.generator.plots.load_polished_dotplot_final``,
which the paper generator (``python -m paper.generator make_all``) now uses by default.
This script just calls that same path for a quick one-off regen, overwriting
``paper/figures/equilibrium_recovery_dotplot.{png,pdf}`` and the backing
``paper/data/equilibrium_recovery_dotplot.csv``.

Run:
    .venv/bin/python tools/regen_equilibrium_recovery_dotplot.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper.generator.config import FIGURES_DIR  # noqa: E402
from paper.generator.plots import (  # noqa: E402
    POLISH_PER_SEED_JSON,
    load_polished_dotplot_final,
    plot_equilibrium_recovery_dotplot,
)


def main() -> int:
    final = load_polished_dotplot_final()
    if final is None:
        print(f"ERROR: {POLISH_PER_SEED_JSON} not found — "
              "run tools/one_stage_polish_per_seed.py first", file=sys.stderr)
        return 1

    n_cells = final.groupby(["experiment", "q"]).ngroups
    print(f"[regen] loaded {len(final)} per-seed polished rows across {n_cells} cells")
    print(final.groupby(["experiment", "q"])["policy_mean_effort"]
          .agg(["mean", "std", "count"]).round(3).to_string())

    out_png = os.path.join(FIGURES_DIR, "equilibrium_recovery_dotplot.png")
    fig, path = plot_equilibrium_recovery_dotplot(final_override=final, output_path=out_png)
    print(f"[regen] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
