#!/usr/bin/env python3
"""check_links: every relative link and image of report.md, README.md and handoff.md resolves.

    python check_links.py                      # against the working tree
    python check_links.py --ref origin/t2-status-pack   # against `git ls-tree -r <ref>` (run after the push)

With --ref the script also checks that every file of the folder (working tree) appears in the tree of the ref.
External links (http, https, mailto) and pure anchors are not followed. A link may carry an anchor or a query;
they are stripped before the lookup.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Set
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
REPO = Path(subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"], text=True).strip())
PACK_REL = PACK.relative_to(REPO).as_posix()
FILES = ["report.md", "README.md", "handoff.md"]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def tree_files(ref: str) -> Set[str]:
    out = subprocess.check_output(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", ref], text=True)
    return set(out.splitlines())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", help="resolve against git ls-tree -r of this ref instead of the working tree")
    a = ap.parse_args()
    files: Set[str] = tree_files(a.ref) if a.ref else set()
    bad = checked = 0
    for name in FILES:
        p = PACK / name
        if not p.exists():
            print("MISSING FILE %s" % name)
            bad += 1
            continue
        text = p.read_text()
        in_code = False
        for ln_no, ln in enumerate(text.splitlines(), 1):
            if ln.strip().startswith("```"):
                in_code = not in_code
            if in_code:
                continue
            for target in LINK.findall(ln):
                if re.match(r"^(https?:|mailto:|#)", target):
                    continue
                t = unquote(target.split("#")[0].split("?")[0])
                if not t:
                    continue
                rel = os.path.normpath(os.path.join(PACK_REL, os.path.dirname(name), t)).replace("\\", "/")
                checked += 1
                ok = (rel in files) if a.ref else (REPO / rel).exists()
                if not ok:
                    bad += 1
                    print("BROKEN %s:%d -> %s (%s)" % (name, ln_no, target, rel))
    if a.ref:
        local = [p.relative_to(REPO).as_posix() for p in PACK.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
        missing = [f for f in local if f not in files]
        for f in missing:
            print("NOT IN %s: %s" % (a.ref, f))
        bad += len(missing)
        print("folder files in working tree: %d; missing from %s: %d" % (len(local), a.ref, len(missing)))
    print("check_links: %s (%d relative links checked)" % ("PASS" if not bad else "FAIL", checked))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
