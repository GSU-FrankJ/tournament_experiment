"""Check that every ledger text occurs in sec04b.md and that every number token in the md occurs in some ledger text."""
import csv
import os
import re

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md = open(os.path.join(base, 'sec04b.md')).read()
rows = list(csv.DictReader(open(os.path.join(base, 'sec04b_ledger.csv'))))
miss = [r for r in rows if r['text'] not in md]
print('ledger rows', len(rows), 'texts not found in md:', len(miss))
for r in miss:
    print('  ', r['statement_id'], r['text'], r['item_id'])
texts = ' '.join(r['text'] for r in rows)
# number tokens in the md (skip item ids like R2B-05, FG-17, section numbers handled by listing)
body = re.sub(r'\[[^\]]*\]\s*(?=[;.,\s])', lambda m: m.group(0) if re.search(r'\d\.\d|e-', m.group(0)) else '', md)
toks = set(re.findall(r'(?<![A-Za-z0-9_.-])-?\d+(?:\.\d+)?(?:e-?\d+)?', md))
unl = sorted(t for t in toks if t not in texts)
print('number tokens not in any ledger text:', len(unl))
print(unl)
