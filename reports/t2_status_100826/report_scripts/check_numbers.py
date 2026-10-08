#!/usr/bin/env python3
"""check_numbers: the provenance rules of report.md.

1. Every table of report.md sits in a ``<!-- TBL:name -->`` block and equals the output of tables.py
   (a markdown table outside a block is an error: tables are not typed by hand).
2. Every citation ``[ID]`` names an item of evidence/manifest.csv (M1-.., M2-.., M3-.., PI-.., BG-.., FIG-..,
   TBL-.., T2R:..).
3. Every numeric sentence outside the tables, code blocks and headings cites at least
   one evidence item. Numbers that are not data (section numbers, ids, dates, seed names, q values) are exempt.

Exit status 0 iff all three hold.
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
sys.path.insert(0, str(HERE))

CITE = re.compile(r"\[((?:M[123]-|PI-|FIG-|TBL-|BG-|T2R:)[^\]]*)\]")
IDS = re.compile(r"(M[123]-\d\d|PI-\d\d|FIG-\d\d|BG-\d\d|TBL-[\w-]+|T2R:[A-Za-z0-9\-]+)")
# tokens that look numeric but are not data
EXEMPT = [
    r"(?i)\bsections?\s+\d+(?:\.\d+)?(?:\s*(?:-|and|to|,)\s*\d+(?:\.\d+)?)*", r"\bn\s*=\s*\d+\b", r"SHA-?256",
    r"\b[A-Z]-?\d+(?:-\d+)?\b",                       # U1, D1, B4, H1, C-MS1, MS-R1, R1, T=2 (the letter forms)
    r"\bT\s*=\s*\d\b", r"\bq\s*=\s*\d+\b", r"\bq in \{[^}]*\}", r"\bs\s*(?:=|in)\s*\{?[\d, ]+\}?", r"\bs\s*=\s*\d+\b",
    r"\b(?:prompts?|item|items|U|H|B|A|D)\s*\d+(?:\s*(?:-|and|to)\s*\d+)?\b",
    r"\b20\d\d-\d\d-\d\d(?:/\d\d)?\b", r"\b2026\d{4}\b", r"\b1050\d\b|\b10510\b|\b3050\d\b|\b30520\b|\b4050\d\b|\b40520\b",
    r"\bseeds?\s+\d+", r"\b[\w/.-]*\d[\w/.-]*\.(?:md|py|csv|json|png|txt)\b", r"\bt2_status_100826\b|\b100526\w*\b",
    r"\b[0-9a-f]{8,40}\b", r"\bv\d(?:\.\d)?\b", r"\bw_[HL]\b", r"\bk = 1/3500\b", r"\b3500\b",
    r"\bsha256\w*\b", r"\bsection\b", r"\bFIG-\d\d\b", r"\b\d+(?:st|nd|rd|th)\b",
    r"\b(?:one|two|three|four|five|six|seven|eight|ten|twelve|twenty)\b",
    r"\bL\d+\b", r"\bstage[- ][12]\b", r"\be[12]\b", r"\be_?hat_?[12]\b", r"\bê[12]\b", r"\bsigma_2\b", r"\bσ_2\b", r"\bF_d\b",
    r"\bR0\b", r"\bR_?[12]\b", r"\bDelta_2\b", r"\beta_2\b", r"\beta_2/DW\b", r"\be2\*", r"\bt1\b|\bt10\b", r"\bg_2\b",
]
EXEMPT_RE = [re.compile(p) for p in EXEMPT]
NUM = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d+|\d+/\d+|\d+%|\d{2,}|\d(?:,\d{3})+)(?![\w])")


def manifest_ids() -> set:
    with open(PACK / "evidence" / "manifest.csv") as f:
        return {r["item_id"] for r in csv.DictReader(f)}


def strip_citations(t: str) -> str:
    return CITE.sub(" ", t)


def data_numbers(t: str):
    t = strip_citations(t)
    t = re.sub(r"`[^`]*`", " ", t)            # inline code
    t = re.sub(r"\([^)]*\.(?:md|py|csv|json|png)[^)]*\)", " ", t)   # link targets
    for rx in EXEMPT_RE:
        t = rx.sub(" ", t)
    return NUM.findall(t)


def units(lines):
    """Yield (first line number, text) for paragraphs and list items outside tables, code and headings."""
    buf, start, in_code, in_tbl = [], 0, False, False
    for i, ln in enumerate(lines, 1):
        if ln.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if ln.startswith("<!-- TBL:"):
            in_tbl = True
            continue
        if ln.startswith("<!-- /TBL:"):
            in_tbl = False
            continue
        if in_tbl:
            continue
        if not ln.strip() or ln.startswith("#") or ln.startswith("![") or ln.startswith(">"):
            if buf:
                yield start, " ".join(buf)
            buf, start = [], 0
            continue
        is_item = re.match(r"^\s*(?:[-*]|\d+\.)\s", ln) is not None
        if is_item and buf:
            yield start, " ".join(buf)
            buf, start = [], 0
        if not buf:
            start = i
        buf.append(ln.strip())
    if buf:
        yield start, " ".join(buf)


def main() -> int:
    import tables as T
    rep = PACK / "report.md"
    lines = rep.read_text().splitlines()
    text = "\n".join(lines)
    bad = 0
    # (1) blocks equal the script output; no table outside a block
    bad += T.inject(rep, PACK, verify=True)
    for u_start, u in units(lines):
        pass
    in_tbl, in_code = False, False
    for i, ln in enumerate(lines, 1):
        if ln.strip().startswith("```"):
            in_code = not in_code
        if ln.startswith("<!-- TBL:"):
            in_tbl = True
        elif ln.startswith("<!-- /TBL:"):
            in_tbl = False
        elif not in_tbl and not in_code and re.match(r"^\|.*\|\s*$", ln):
            print("TABLE OUTSIDE A BLOCK: line %d: %s" % (i, ln[:80]))
            bad += 1
    names = set(re.findall(r"<!-- TBL:(\w+) -->", text))
    for n in sorted(names):
        if n not in T.BLOCKS:
            print("UNKNOWN BLOCK %s" % n)
            bad += 1
    # (2) citations exist
    ids = manifest_ids()
    for blk in CITE.findall(text):
        for iid in IDS.findall(blk):
            if iid not in ids:
                print("UNKNOWN ID %s in [%s]" % (iid, blk[:60]))
                bad += 1
    # (3) numeric sentences cite (a sentence is a run of text ended by '. ', or by a Chinese full stop)
    n_units = n_flag = 0
    for start, u in units(lines):
        u = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", u)
        for sent in re.split(r"(?<=[.。])\s+(?=[A-Z(\[`*\-])|(?<=。)", u):
            nums = data_numbers(sent)
            if not nums:
                continue
            n_units += 1
            if not CITE.search(sent):
                n_flag += 1
                bad += 1
                print("NO CITATION: line %d: %s ... numbers %s" % (start, sent[:110], nums[:6]))
    print("check_numbers: %s (%d numeric sentences checked, %d without a citation)" % ("PASS" if not bad else "FAIL", n_units, n_flag))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
