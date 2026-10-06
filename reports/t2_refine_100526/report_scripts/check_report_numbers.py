#!/usr/bin/env python3
"""Backstop of the fact-check: does every decimal / scientific number of a Markdown file occur in the evidence pack?

For each token of the form 0.0123, -1.5e-05, 4.982e-07 in the given Markdown file(s) (code fences and the SHA-256 /
hash-like tokens excluded), the script asks whether some number in the pack's data files (csv, json, out, txt; everything
under ``evidence/`` except the round reports ``*.md``) equals it after rounding to the token's number of significant
digits (a table value rounded the way the report prints it), or occurs in the pack's report copies (``*.md``) only.

  FOUND_TABLE        a table / record value rounds to the token
  FOUND_REPORT_ONLY  no table value does, but a round report of the pack prints it (a "report only" number)
  NOT_FOUND          neither: the number must be a value the report computed (a median, a difference, a ratio, design
                     arithmetic) or a constant (a threshold, a setting); the fact-check re-derives each one

A match is a necessary condition for a number to be right, not a sufficient one: it does not say the number is attached to the
right statement. The fact-check reviewers do that. Usage:

    python report_scripts/check_report_numbers.py <markdown file> [...] [--pack reports/t2_refine_100526/evidence] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

NUM = re.compile(r"(?<![\w.])[-+−]?\d+\.\d+(?:[eE][-+]?\d+)?(?![\w])|(?<![\w.])[-+−]?\d+[eE][-+]?\d+(?![\w])")
DATA_SUFFIX = {".csv", ".json", ".out", ".txt"}


def sig_digits(tok: str) -> int:
    """Number of significant digits of a printed number (mantissa digits without leading zeros)."""
    m = tok.lstrip("+-−").lower().split("e")[0].replace(".", "")
    m = m.lstrip("0")
    return max(1, len(m))


def parse(tok: str) -> float:
    """Float value of a printed number (accepts the Unicode minus)."""
    return float(tok.replace("−", "-"))


def collect(pack: Path) -> Tuple[List[float], List[float]]:
    """(sorted |values| of the data files, sorted |values| of the report copies) of the pack."""
    tab, rep = [], []
    for p in sorted(pack.rglob("*")):
        if not p.is_file() or p.name in ("SHA256SUMS",):
            continue
        suf = p.suffix.lower()
        dst = tab if suf in DATA_SUFFIX or p.name == "LOCK" else rep if suf == ".md" else None
        if dst is None:
            continue
        txt = p.read_text(errors="replace")
        for m in re.finditer(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+", txt):
            try:
                dst.append(abs(float(m.group(0))))
            except ValueError:
                pass
    return sorted(tab), sorted(rep)


def has_rounding(vals: Sequence[float], t: float, s: int) -> bool:
    """True if some value in the sorted ``vals`` rounds to ``t`` at ``s`` significant digits (half-ulp interval)."""
    a = abs(t)
    if a == 0.0:
        return bisect_left(vals, 0.0) < len(vals) and vals[bisect_left(vals, 0.0)] == 0.0
    ulp = 10.0 ** (math.floor(math.log10(a)) - s + 1)
    lo, hi = a - ulp / 2 - 1e-15 * a, a + ulp / 2 + 1e-15 * a
    return bisect_left(vals, lo) < bisect_right(vals, hi)


def tokens(md: str) -> List[Tuple[int, str]]:
    """(line number, token) of every decimal / scientific number outside code fences and hash-like strings."""
    out, fence = [], False
    for i, line in enumerate(md.splitlines(), 1):
        if line.strip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        line = re.sub(r"\b[0-9a-f]{7,64}\b", " ", line)             # commit hashes, SHA-256 digests
        for m in NUM.finditer(line):
            out.append((i, m.group(0)))
    return out


def main(argv: Sequence[str]) -> int:
    """CLI entry."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--pack", default="reports/t2_refine_100526/evidence")
    ap.add_argument("--csv", default=None, help="write every token with its category")
    a = ap.parse_args(argv)
    tab, rep = collect(Path(a.pack))
    rows: List[Dict[str, str]] = []
    counts: Dict[str, int] = {}
    for f in a.files:
        for line_no, tok in tokens(Path(f).read_text()):
            s, v = sig_digits(tok), parse(tok)
            cat = "FOUND_TABLE" if has_rounding(tab, v, s) else "FOUND_REPORT_ONLY" if has_rounding(rep, v, s) else "NOT_FOUND"
            counts[cat] = counts.get(cat, 0) + 1
            rows.append({"file": f, "line": str(line_no), "token": tok, "category": cat})
    print(f"{len(rows)} numeric tokens: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "line", "token", "category"], lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
