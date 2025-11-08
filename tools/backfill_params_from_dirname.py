#!/usr/bin/env python3
import os, re, json
from glob import glob

DIR_PAT = re.compile(
    r'lr(?P<lr_start>\d+p\d+em\d+)_lrf(?P<lr_final>\d+p\d+em\d+)'
    r'_kl(?P<target_kl>\d+p\d+em\d+)'
    r'_ent(?P<entropy_hold_fraction>\d+p\d+em\d+)'
    r'_clip(?P<clip_range_end>\d+p\d+em\d+)',
    re.IGNORECASE
)

def decode_token(tok: str) -> float:
    """
    '3p000em04' -> 3.000e-04
    '1p5em02'   -> 1.5e-02
    """
    m = re.fullmatch(r'(\d+)p(\d+)em(\d+)', tok, re.IGNORECASE)
    if not m:
        raise ValueError(f"bad token: {tok}")
    A, B, C = m.groups()
    base = float(f"{A}.{B}")
    return base * (10 ** (-int(C)))

def try_parse_from_dirname(d: str):
    name = os.path.basename(d)
    m = DIR_PAT.search(name)
    if not m:
        return None
    vals = {k: decode_token(v) for k, v in m.groupdict().items()}
    return vals

def main():
    runs = sorted(set(os.path.dirname(p) for p in glob("results/**/metrics.csv", recursive=True)))
    if not runs:
        print("No metrics.csv under results/**")
        return 1

    wrote = 0
    for d in runs:
        pj = os.path.join(d, "params.json")
        if os.path.exists(pj):
            print(f"[skip] {d} (params.json exists)")
            continue
        vals = try_parse_from_dirname(d)
        if not vals:
            print(f"[warn] {d}: name does not match pattern")
            continue
        os.makedirs(d, exist_ok=True)
        with open(pj, "w", encoding="utf-8") as f:
            json.dump(vals, f, indent=2)
        print(f"[ok] wrote {pj}: " + ", ".join(f"{k}={vals[k]}" for k in [
            "lr_start","lr_final","target_kl","entropy_hold_fraction","clip_range_end"]))
        wrote += 1
    print(f"\nSummary: wrote={wrote}, total_dirs={len(runs)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

