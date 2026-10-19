#!/usr/bin/env python3
"""Substitutes the table fragments and the inline numbers into ``templates/01_audit.tpl.md`` and writes ``reports/t3/r2/01_audit.md``.

Tokens: ``{{T:name}}`` = the fragment ``tables/name.md``; ``{{n:key}}`` / ``{{n:key:fmt}}`` = the number ``key`` of
``tables/audit_numbers.json`` formatted with ``fmt`` (default ``.3g``); ``{{s:key}}`` = the value as text. An unknown token stops the script.

Usage:
    python reports/t3/r2/report_scripts/assemble_audit.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REP = HERE.parent
TOKEN = re.compile(r"\{\{(T|n|s):([^}:]+)(?::([^}]+))?\}\}")


def main() -> int:
    """Assemble the audit report."""
    nums = json.loads((REP / "tables" / "audit_numbers.json").read_text())
    text = (HERE / "templates" / "01_audit.tpl.md").read_text()

    def sub(m: re.Match) -> str:
        kind, key, fmt = m.group(1), m.group(2), m.group(3)
        if kind == "T":
            return (REP / "tables" / f"{key}.md").read_text().rstrip()
        v = nums[key]
        if kind == "s":
            return str(v)
        return format(v, fmt or ".3g")
    out = TOKEN.sub(sub, text)
    left = re.findall(r"\{\{[^}]*\}\}", out)
    if left:
        print("unresolved tokens:", left)
        return 1
    (REP / "01_audit.md").write_text(out)
    print(f"wrote {REP / '01_audit.md'} ({len(out.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
