"""Shared helpers of the T3-R2 report scripts (Markdown tables, number formats, paths)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RES = ROOT / "results" / "t3_r2"
REP = ROOT / "reports" / "t3" / "r2"
TABLES = REP / "tables"
for _p in (ROOT, ROOT / "tools" / "t3", ROOT / "tools" / "ms"):
    sys.path.insert(0, str(_p))
DASH = "-"


def num(x: object, nd: int = 5, sci: bool = False) -> str:
    """Format a number (NaN and None become a dash)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return DASH
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return f"{int(x):,}"
    if sci:
        return f"{float(x):.2e}"
    return f"{float(x):.{nd}f}"


def md(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Render a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines) + "\n"


def write(name: str, text: str) -> Path:
    """Write a table fragment into ``reports/t3/r2/tables/`` and return its path."""
    TABLES.mkdir(parents=True, exist_ok=True)
    p = TABLES / f"{name}.md"
    p.write_text(text)
    return p
