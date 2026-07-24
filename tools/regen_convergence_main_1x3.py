#!/usr/bin/env python3
"""Generate paper/figures/convergence_main_1x3.{png,pdf} — the 1x3 companion
of the convergence-main figure.

Keeps only the Set-1 (w_H, w_L) = (6.5, 3.0) row across q = 35/45/55, with the
same styling and legend as the full figure (first-pass verification line, raw
circle + MC-BR polished star endpoint markers), sized like the
exploitability-dynamics 1x3 layout. Writes a NEW file; never overwrites the
full convergence_main figure or its backing CSV.

Run:
    .venv/bin/python tools/regen_convergence_main_1x3.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper.generator.extract import load_all_convergence_data  # noqa: E402
from paper.generator.plots import plot_convergence_main_1x3  # noqa: E402


def main() -> int:
    df = load_all_convergence_data()
    if df.empty:
        print("ERROR: no convergence data found", file=sys.stderr)
        return 1
    fig, path = plot_convergence_main_1x3(df=df)
    print(f"[regen] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
