#!/usr/bin/env python3
"""T3-R1 P3: build ``reports/t3/r1/handoff.md`` (Chinese only, for the coworker) from ``handoff_draft_zh.md``.

Replaces the ``COMMIT`` placeholder of the links by the full hash of the commit that holds the reports (the hash must be 40
hex digits and an existing commit of this repository). Stops if more than five links remain, if a placeholder is left, or if the
text contains a mail address.

Usage:
    python reports/t3/r1/report_scripts/build_handoff.py <full commit hash>
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DRAFT = HERE / "handoff_draft_zh.md"
OUT = ROOT / "reports" / "t3" / "r1" / "handoff.md"
LINK = re.compile(r"https://github\.com/[^\s)]+")


def main(argv: list) -> int:
    """Write handoff.md for the given commit; return 1 on any violated condition."""
    if len(argv) != 2 or not re.fullmatch(r"[0-9a-f]{40}", argv[1]):
        print("usage: build_handoff.py <full 40-digit commit hash>")
        return 1
    commit = argv[1]
    r = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-t", commit], capture_output=True, text=True)
    if r.stdout.strip() != "commit":
        print(f"{commit} is not a commit of this repository")
        return 1
    text = DRAFT.read_text().replace("COMMIT", commit)
    links = LINK.findall(text)
    if len(links) > 5 or "COMMIT" in text or re.search(r"[\w.]+@[\w.]+\.\w+", text):
        print(f"{len(links)} links; placeholder or mail address left: refuse")
        return 1
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(links)} links, commit {commit})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
