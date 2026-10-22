"""Shared helpers of the report scripts of round T3-R4: markdown tables and the paths. Imported by ``lock_tables.py`` and
``confirmation_tables.py``; writes nothing itself."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "reports" / "t3" / "r4" / "tables"


def fmt(v: Any, nd: int = 4) -> str:
    """Cell text: NaN -> '-', booleans yes / no, floats with 4 significant digits (scientific below 1e-3 / above 1e5)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if v == 0.0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e5:
            return f"{v:.3g}"
        return f"{v:.{nd}g}"
    return str(v).replace("|", "\\|")


def md(rows: Sequence[Dict[str, Any]], cols: Optional[Sequence[str]] = None, heads: Optional[Sequence[str]] = None, nd: int = 4) -> str:
    """Markdown table of dict rows."""
    cols = list(cols or rows[0].keys())
    heads = list(heads or cols)
    out = ["| " + " | ".join(heads) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c), nd) for c in cols) + " |")
    return "\n".join(out)
