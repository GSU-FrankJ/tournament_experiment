#!/usr/bin/env python3
"""Render a report template: every ``{{table:name}}`` is replaced by ``reports/t3/r3/tables/<name>.md`` (written by
``diagnosis_tables.py`` / ``proposal_tables.py`` / ``prereg_tables.py`` / ``results_tables.py``) and every ``{{name}}`` by the value given
as ``name=value`` on the command line. Usage: ``python build_report.py <template.md> <output.md> [name=value ...]``."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Mapping

TABLES = Path(__file__).resolve().parents[1] / "tables"


def render(template: str, variables: Mapping[str, str] = {}) -> str:
    """Substitute the table placeholders (a missing table is an error) and the variables (an unknown variable is an error)."""
    def table(m: "re.Match[str]") -> str:
        return (TABLES / f"{m.group(1)}.md").read_text().rstrip("\n")

    def var(m: "re.Match[str]") -> str:
        if m.group(1) not in variables:
            raise KeyError(f"no value given for {{{{{m.group(1)}}}}}")
        return str(variables[m.group(1)])
    text = re.sub(r"\{\{table:([A-Za-z0-9_]+)\}\}", table, template)
    return re.sub(r"\{\{([A-Za-z0-9_]+)\}\}", var, text)


def main() -> int:
    """CLI entry."""
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    variables: Dict[str, str] = dict(a.split("=", 1) for a in sys.argv[3:])
    dst.write_text(render(src.read_text(), variables))
    print(f"{src} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
