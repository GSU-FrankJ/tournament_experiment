#!/usr/bin/env python3
"""Render a report template: every ``{{table:name}}`` is replaced by ``reports/t3/r3/tables/<name>.md`` (written by
``diagnosis_tables.py`` / ``proposal_tables.py``). Usage: ``python build_report.py <template.md> <output.md>``."""

from __future__ import annotations

import re
import sys
from pathlib import Path

TABLES = Path(__file__).resolve().parents[1] / "tables"


def render(template: str) -> str:
    """Substitute the table placeholders (a missing table is an error)."""
    def sub(m: "re.Match[str]") -> str:
        return (TABLES / f"{m.group(1)}.md").read_text().rstrip("\n")
    return re.sub(r"\{\{table:([A-Za-z0-9_]+)\}\}", sub, template)


def main() -> int:
    """CLI entry."""
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.write_text(render(src.read_text()))
    print(f"{src} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
