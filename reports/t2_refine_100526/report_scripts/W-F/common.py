"""Shared helpers of writer W-F: read the evidence pack, format numbers as the round reports do,
and keep a ledger of every number written into the report text."""
import csv
import json
import os

import pandas as pd

PACK = "/home/fjiang4/tournament_experiment/.claude/worktrees/t2-refine-pack/reports/t2_refine_100526"
EV = PACK + "/evidence/"
SCRATCH = ("/tmp/claude-1331199693/-home-fjiang4-tournament-experiment--claude-worktrees-"
           "r2c-sampler-protocol-v2-1-c4c0f1/152f0306-5492-45a1-94d5-59493fa0d141/scratchpad/"
           "report_parts")

# item id -> source path inside the pack (read from manifest.csv)
_M = pd.read_csv(EV + "manifest.csv")
PATH = dict(zip(_M.item_id, _M.source_path))


def rd(item: str) -> pd.DataFrame:
    """Read a pack CSV by item id."""
    return pd.read_csv(EV + PATH[item])


def rj(item: str):
    """Read a pack JSON by item id."""
    with open(EV + PATH[item]) as fh:
        return json.load(fh)


def g4(x: float) -> str:
    """Format as the round reports do: 4 significant digits (general format)."""
    return format(float(x), ".4g")


def ci(m: float, lo: float, hi: float) -> str:
    """mean [lo, hi] with 4 significant digits."""
    return "%s [%s, %s]" % (g4(m), g4(lo), g4(hi))


class Ledger:
    """Collects (statement_id, section, text, value, item_id, locator) rows."""

    def __init__(self, section: str):
        self.section = section
        self.rows = []

    def add(self, text: str, value, item: str, locator: str) -> str:
        """Register a number string as it appears in the text; return the text unchanged."""
        sid = "S%s-%03d" % (self.section, len(self.rows) + 1)
        self.rows.append((sid, self.section, text, value, item, locator))
        return text

    def save(self, name: str) -> None:
        with open(os.path.join(SCRATCH, name), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["statement_id", "section", "text", "value", "item_id", "locator"])
            w.writerows(self.rows)
