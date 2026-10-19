#!/usr/bin/env python3
"""Operating characteristic of the T3-R2 confirmation rule (D4).

With per-run pass probability p, the same at both q and independent between the q blocks, the confirmation passes iff
X50 >= 17 and X60 >= 17 and X50 + X60 >= 36 with X_q ~ Binomial(20, p). Prints the table and writes
``reports/t3/r2/tables/t01_operating_characteristic.md`` and ``results/t3_r2/preflight/oc_table.csv``.

Usage:
    python reports/t3/r2/report_scripts/oc_table.py
"""

from __future__ import annotations

import csv

import numpy as np
from scipy.stats import binom

from common import RES, md, write

P_VALUES = (0.80, 0.85, 0.90, 0.95)
STATED = {0.80: 0.071, 0.85: 0.249, 0.90: 0.607, 0.95: 0.942}      # the PI's numbers in D4


def pass_probability(p: float, n: int = 20, per_q: int = 17, total: int = 36) -> float:
    """P(X50 >= per_q and X60 >= per_q and X50 + X60 >= total), X_q ~ Binomial(n, p) independent."""
    pm = binom.pmf(np.arange(n + 1), n, p)
    return float(sum(pm[a] * pm[b] for a in range(per_q, n + 1) for b in range(per_q, n + 1) if a + b >= total))


def main() -> int:
    """Compute, compare with the numbers stated in D4, write the table."""
    rows = [(p, pass_probability(p), STATED[p]) for p in P_VALUES]
    ok = all(round(v, 3) == s for _, v, s in rows)
    text = md(["per-run pass probability p", "P(confirmation passes), recomputed", "stated in D4", "agrees at 3 decimals"],
              [(f"{p:.2f}", f"{v:.4f}", f"{s:.3f}", "yes" if round(v, 3) == s else "NO") for p, v, s in rows])
    write("t01_operating_characteristic", text)
    RES.joinpath("preflight").mkdir(parents=True, exist_ok=True)
    with open(RES / "preflight" / "oc_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "p_pass", "stated_in_D4"])
        w.writerows(rows)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
