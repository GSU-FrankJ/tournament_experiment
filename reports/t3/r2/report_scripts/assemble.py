#!/usr/bin/env python3
"""Substitutes table fragments and inline numbers into a report template (the P3 reports of T3-R2).

Tokens: ``{{T:name}}`` = ``tables/name.md``; ``{{n:key}}`` / ``{{n:key:fmt}}`` = the number ``key`` of the numbers files (default format
``.3g``); ``{{s:key}}`` = the value as text. The numbers files are ``tables/confirmation_numbers.json`` and ``tables/audit_numbers.json``
(a key of the first wins). An unknown token stops the script.

Usage:
    python reports/t3/r2/report_scripts/assemble.py templates/04_confirmation.tpl.md ../04_confirmation.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REP = HERE.parent
TOKEN = re.compile(r"\{\{(T|n|s):([^}:]+)(?::([^}]+))?\}\}")


def main(argv=None) -> int:
    """Assemble one report."""
    argv = list(sys.argv[1:] if argv is None else argv)
    tpl, out = HERE / argv[0], (HERE / argv[1]).resolve()
    nums = {}
    for name in ("audit_numbers.json", "confirmation_numbers.json"):
        p = REP / "tables" / name
        if p.is_file():
            nums.update(json.loads(p.read_text()))
    text = tpl.read_text()

    def sub(m: re.Match) -> str:
        kind, key, fmt = m.group(1), m.group(2), m.group(3)
        if kind == "T":
            return (REP / "tables" / f"{key}.md").read_text().rstrip()
        v = nums[key]
        return str(v) if kind == "s" else format(v, fmt or ".3g")
    res = TOKEN.sub(sub, text)
    left = re.findall(r"\{\{[^}]*\}\}", res)
    if left:
        print("unresolved tokens:", left)
        return 1
    out.write_text(res)
    print(f"wrote {out} ({len(res.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
