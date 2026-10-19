#!/usr/bin/env python3
"""Summarises the sharded full-suite run of ``pytest tests`` (``results/t3_r2/preflight/tests/shard_NN.txt``).

Writes ``results/t3_r2/preflight/tests/full_suite_shards.txt`` (one line per shard, then the totals and the failed tests) and prints the
totals. The shards were run concurrently, single-threaded, each over a round-robin share of the ``tests/test_*.py`` files.

Usage:
    python reports/t3/r2/report_scripts/suite_summary.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from common import RES

TESTS = RES / "preflight" / "tests"
COUNT = re.compile(r"(\d+) (passed|failed|skipped|xfailed|xpassed|error|errors|warnings?|deselected)")


def main() -> int:
    """Parse the shard files and write the summary."""
    files = sorted(TESTS.glob("shard_*.txt"))
    if not files:
        print("no shard files")
        return 1
    tot = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "error": 0}
    failed, lines = [], ["shard | test files | result | exit"]
    for f in files:
        text = f.read_text().splitlines()
        head = text[0].split(": ", 1)[1].split() if text and text[0].startswith("shard") else []
        summ = next((l for l in reversed(text) if re.search(r"\d+ (passed|failed|error)", l) and " in " in l), "")
        ex = next((l.split("=")[1] for l in reversed(text) if l.startswith("exit=")), "?")
        for n, kind in COUNT.findall(summ):
            kind = "error" if kind.startswith("error") else kind
            if kind in tot:
                tot[kind] += int(n)
        failed += [l.split(" - ")[0].replace("FAILED ", "") for l in text if l.startswith("FAILED ")]
        lines.append(f"{f.stem.split('_')[1]} | {len(head)} | {summ.strip('= ').strip()} | {ex}")
    lines.append("")
    lines.append("total: " + ", ".join(f"{v} {k}" for k, v in tot.items()))
    lines.append("failed tests: " + (", ".join(failed) if failed else "none"))
    (TESTS / "full_suite_shards.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
