"""Final mechanical checks of reports/t2_refine_100526: links, item tags, table shapes, checksum files, forbidden files and sizes."""
import csv
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/t2-refine-pack/reports/t2_refine_100526")
MDS = [ROOT / "100526report.md", ROOT / "README.md", ROOT / "pi_record" / "README.md", ROOT / "pi_record" / "plans" / "README.md",
       ROOT / "pi_record" / "01_factcheck.md", ROOT / "pi_record" / "00_publication_log.md", ROOT / "report_scripts" / "README.md"]


def sha(p: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main() -> int:
    """Run all checks; return the number of problems."""
    bad = 0
    with open(ROOT / "evidence" / "manifest.csv", newline="") as f:
        ids = {r["item_id"] for r in csv.DictReader(f)}
    print(f"manifest rows: {len(ids)}")
    for md in MDS:
        if not md.exists():
            print("MISSING FILE", md)
            bad += 1
            continue
        txt = md.read_text()
        # links
        nl = 0
        for m in re.finditer(r"\]\(([^)#\s]+)(#[^)]*)?\)", txt):
            tgt = m.group(1)
            if tgt.startswith(("http://", "https://", "mailto:")):
                continue
            nl += 1
            if not (md.parent / tgt).exists():
                print(f"BROKEN LINK in {md.name}: {tgt}")
                bad += 1
        # tags
        toks = set(re.findall(r"\b((?:PL|CF|R1|R2B|R2C|RR|UT|FG)-\d{2})\b", txt))
        miss = sorted(t for t in toks if t not in ids)
        if miss:
            print(f"UNRESOLVED TAGS in {md.name}: {miss}")
            bad += 1
        # tables
        lines = txt.splitlines()
        i, ntab, nrow = 0, 0, 0
        fence = False
        while i < len(lines):
            if lines[i].strip().startswith("```"):
                fence = not fence
            if not fence and lines[i].lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
                def cells(s: str) -> int:
                    s = s.strip()
                    return len(re.split(r"(?<!\\)\|", s)) - 2
                n = cells(lines[i])
                j = i + 2
                ntab += 1
                while j < len(lines) and lines[j].lstrip().startswith("|"):
                    nrow += 1
                    if cells(lines[j]) != n:
                        print(f"TABLE ROW with {cells(lines[j])} cells (header {n}) in {md.name} line {j + 1}")
                        bad += 1
                    j += 1
                i = j
                continue
            i += 1
        print(f"{md.name}: {nl} relative links, {len(toks)} distinct tags, {ntab} tables, {nrow} data rows")
    # checksum files
    for sub in ("evidence", "figures", "report_scripts", "pi_record"):
        sf = ROOT / sub / "SHA256SUMS"
        if not sf.exists():
            print(f"NO SHA256SUMS in {sub}")
            if sub != "pi_record" or "--final" in sys.argv:
                bad += 1
            continue
        listed = {}
        for line in sf.read_text().splitlines():
            h, _, name = line.partition("  ")
            listed[name] = h
        ok = 0
        for name, h in listed.items():
            p = ROOT / sub / name
            if not p.is_file() or sha(p) != h:
                print(f"SUMS MISMATCH {sub}/{name}")
                bad += 1
            else:
                ok += 1
        on_disk = {p.relative_to(ROOT / sub).as_posix() for p in (ROOT / sub).rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
        for name in sorted(on_disk - set(listed)):
            print(f"NOT LISTED in {sub}/SHA256SUMS: {name}")
            bad += 1
        print(f"{sub}/SHA256SUMS: {ok} of {len(listed)} OK; {len(on_disk)} files on disk")
    # forbidden files and sizes
    big, forb, n = [], [], 0
    for p in ROOT.rglob("*"):
        if p.is_file():
            n += 1
            if p.stat().st_size >= 1 << 20:
                big.append(p)
            if p.suffix in (".pt", ".pth", ".pyc") or p.name.startswith("train_history") or "__pycache__" in p.parts:
                forb.append(p)
    print(f"files: {n}; >= 1 MiB: {len(big)}; forbidden: {len(forb)}")
    bad += len(big) + len(forb)
    print("PROBLEMS:", bad)
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
