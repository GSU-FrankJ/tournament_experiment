"""Check that every ledger text occurs in sec05.md and list numeric tokens not covered by the ledger."""
import csv
import re

base = ("/tmp/claude-1331199693/-home-fjiang4-tournament-experiment--claude-worktrees-r2c-sampler-"
        "protocol-v2-1-c4c0f1/152f0306-5492-45a1-94d5-59493fa0d141/scratchpad/report_parts/")
md = open(base + "sec05.md").read()
led = list(csv.DictReader(open(base + "sec05_ledger.csv")))
missing = [r["text"] for r in led if r["text"] not in md]
print("ledger texts not in md:", missing)
toks = set(r["text"] for r in led)
txt = re.sub(r"`[^`]*`", "", md)
txt = re.sub(r"\]\(figures/[^)]*\)", "", txt)
nums = re.findall(r"(?<![\w.])-?\d[\d,]*\.?\d*(?:e[+-]?\d+)?", txt)
unc = sorted(set(n for n in nums if n not in toks and n.rstrip(".,") not in toks))
print(unc)
