#!/usr/bin/env python3
"""T3-R1 P3: assemble the results reports from their templates and the generated tables.

Substitutes every ``{{name}}`` of ``report_scripts/templates/*.md.in`` by the contents of ``tables/name.md`` and writes the report
to ``reports/t3/r1/<template name without .in>``. Stops if a marker has no table. Run ``results_tables.py`` first.

Usage:
    python reports/t3/r1/report_scripts/results_tables.py
    python reports/t3/r1/report_scripts/assemble_reports.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R1 = HERE.parent
TABLES = R1 / "tables"
MARK = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}")


def main() -> int:
    """Assemble every template; return 1 if a marker has no table."""
    missing = []
    for tpl in sorted((HERE / "templates").glob("*.md.in")):
        text = tpl.read_text()

        def sub(m: "re.Match[str]") -> str:
            f = TABLES / f"{m.group(1)}.md"
            if not f.exists():
                missing.append((tpl.name, m.group(1)))
                return m.group(0)
            return f.read_text().rstrip("\n")

        out = MARK.sub(sub, text)
        (R1 / tpl.name[:-3]).write_text(out)
        print(f"{tpl.name} -> {tpl.name[:-3]} ({len(MARK.findall(text))} tables)")
    if missing:
        print("MISSING tables:", missing)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
