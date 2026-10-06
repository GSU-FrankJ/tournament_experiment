"""Heuristic audit: every numeric token in sec06.md (outside code spans, bracketed item references,
image links and headings) should appear as the text of a ledger row. Prints the tokens that do not."""
import csv
import re
import collections

BASE = ('/tmp/claude-1331199693/-home-fjiang4-tournament-experiment--claude-worktrees-'
        'r2c-sampler-protocol-v2-1-c4c0f1/152f0306-5492-45a1-94d5-59493fa0d141/scratchpad/report_parts/')
text = open(BASE + 'sec06.md').read()
led = list(csv.DictReader(open(BASE + 'sec06_ledger.csv')))
texts = collections.Counter(r['text'] for r in led)
alltext = ' || '.join(texts)

t = re.sub(r'`[^`]*`', ' ', text)                # code spans
t = re.sub(r'\[[A-Z0-9, \-;"a-z.\(\)=_:]*\]', ' ', t)  # bracketed refs (also removes some prose; fine)
t = re.sub(r'\!\[[^\]]*\]\([^)]*\)', ' ', t)     # images
t = re.sub(r'^#+ .*$', ' ', t, flags=re.M)       # headings
t = re.sub(r'^Source:.*$', ' ', t, flags=re.M)   # source lines: file names, limits
t = re.sub(r'\bR[12][a-z]?\b|\bR2[bc]\b|\bRR-\d+|\bR\d-\d+|\bR2[BC]-\d+|\bPL-\d+|\bFG-\d+', ' ', t)
t = re.sub(r'\bsections? \d(\.\d)?(-\d(\.\d)?)?|\btables? \d\.\d[a-d]?|\bTable \d\.\d[a-d]?', ' ', t)
tokens = re.findall(r'(?<![A-Za-z_\d.])-?\d+(?:\.\d+)?(?:e[-+]?\d+)?', t)
missing = collections.Counter()
for tok in tokens:
    if tok in texts or ('| ' + tok) in alltext:
        continue
    # accept if token is a substring of a ledger text (e.g. inside "mean [lo, hi]" cells)
    if any(tok == x or tok in re.findall(r'-?\d+(?:\.\d+)?(?:e[-+]?\d+)?', x) for x in texts):
        continue
    missing[tok] += 1
print('numeric tokens:', len(tokens), ' not in ledger:', sum(missing.values()))
for k, v in sorted(missing.items(), key=lambda kv: -kv[1]):
    print(repr(k), v)
