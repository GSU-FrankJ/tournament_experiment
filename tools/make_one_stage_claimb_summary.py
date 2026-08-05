"""Emit the one-stage Claim-B summary (raw + MC-BR polished) for the four experiments.

Writes ``results/one_stage_claimb_summary.csv`` in the column format of
``paper/tables/final_summary.csv`` EXTENDED with three MC-BR *polished* columns,
restricted to the four one-stage experiments at q in {35, 55} — the cells backed
by the 40 r5_sampled convergence JSONs (4 experiments x 2 q x 5 seeds) that the
Phase-0 Claim-B verification consumed.

Columns:
  Scenario, q, Method,
  Mean±std, |ē−e*|, RelErr,            # RAW TEL-PPO (basin) — from final_summary.csv
  Polished, Polished |ē−e*|, Polished RelErr,   # MC-BR polished — from the phase0 log
  Exploitability, Symmetry Gap, Conv. Update (verified)

Sources (both canonical, already committed):
  paper/tables/final_summary.csv                      -> raw rows (generator output for the 40 JSONs)
  results/phase0_verify_20260701_1941.log             -> MC-BR polished cross-seed means

CAVEAT on the error metric (documented, not a bug):
  * RAW ``|ē−e*|`` is the mean over the 5 seeds of |seed_effort − e*| (a MAE that
    includes cross-seed scatter), as the paper generator defines it.
  * ``Polished |ē−e*|`` is |cross-seed-mean_polished − e*| (the bias reported in the
    phase0 log). Per-seed polished values were never persisted, so a matching
    MAE-across-seeds is not available for the polished column.
  For asymmetric cells (Het. Cost) both raw and polished are player-averaged, as in
  final_summary.csv. The polished column is the Claim-B "bridged to e*" result; the
  raw column is the "PPO reaches basin" half.

Run:
    python tools/make_one_stage_claimb_summary.py
"""

from __future__ import annotations

import csv
import os
import re
import sys

SRC = os.path.join("paper", "tables", "final_summary.csv")
LOG = os.path.join("results", "phase0_verify_20260701_1941.log")
OUT = os.path.join("results", "one_stage_claimb_summary.csv")

SCENARIOS = {"Two-Player", "Three-Player", "Het. Cost", "Het. Ability"}
QS = {"35", "55"}
METHODS = {"Theory", "TEL-PPO"}
# phase0-log cell tag -> final_summary scenario label
TAG2SCEN = {"3P": "Three-Player", "dc": "Het. Cost", "da": "Het. Ability"}

POL_COLS = ["Polished", "Polished |ē−e*|", "Polished RelErr"]


def parse_polished(path: str):
    """Parse MC-BR polished cross-seed means from the phase0 verify log.

    Returns dict[(scenario, q)] -> (polished_mean, abs_err, rel_pct), player-averaged.
    """
    text = open(path).read()
    lines = text.splitlines()
    out = {}

    # Main cells (3P/dc/da): a '### <tag> q<qq>' header then one 'polished X (+-e, r%)' per player.
    cur = None
    acc = {}  # (scen,q) -> [(polished, |err|, rel), ...]
    hdr = re.compile(r"^### (3P|dc|da) q(\d+)\b")
    pol = re.compile(r"polished\s+([\d.]+)\s+\(([+-]?[\d.]+),\s*([\d.]+)%\)")
    for ln in lines:
        m = hdr.search(ln)
        if m:
            cur = (TAG2SCEN[m.group(1)], m.group(2))
            acc.setdefault(cur, [])
            continue
        if cur:
            mp = pol.search(ln)
            if mp:
                acc[cur].append((float(mp.group(1)), abs(float(mp.group(2))), float(mp.group(3))))
    for key, players in acc.items():
        n = len(players)
        out[key] = (sum(p[0] for p in players) / n,
                    sum(p[1] for p in players) / n,
                    sum(p[2] for p in players) / n)

    # 2P do-no-harm lines: '2P q35 (e*=45.45, ...): ... polished=44.95(|e|=0.508)'
    p2 = re.compile(r"2P q(\d+) \(e\*=([\d.]+),.*polished=([\d.]+)\(\|e\|=([\d.]+)\)")
    for ln in lines:
        m = p2.search(ln)
        if m:
            q, estar, polv, aerr = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
            out[("Two-Player", q)] = (polv, aerr, 100.0 * aerr / estar)
    return out


def main() -> int:
    for p in (SRC, LOG):
        if not os.path.exists(p):
            print(f"ERROR: {p} not found", file=sys.stderr)
            return 1

    polished = parse_polished(LOG)

    with open(SRC, newline="") as fh:
        reader = csv.DictReader(fh)
        base_cols = reader.fieldnames
        rows = [r for r in reader
                if r["Scenario"] in SCENARIOS and r["q"] in QS and r["Method"] in METHODS]

    # insert polished columns right after RelErr
    ri = base_cols.index("RelErr") + 1
    header = base_cols[:ri] + POL_COLS + base_cols[ri:]

    for r in rows:
        if r["Method"] == "TEL-PPO":
            pv, ae, rel = polished[(r["Scenario"], r["q"])]
            r["Polished"] = f"{pv:.2f}"
            r["Polished |ē−e*|"] = f"{ae:.2f}"
            r["Polished RelErr"] = f"{rel:.2f}%"
        else:  # Theory reference row: no polish
            for c in POL_COLS:
                r[c] = "-"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[wrote] {OUT}  ({len(rows)} rows)")
    print(",".join(header))
    for r in rows:
        print(",".join(r[c] for c in header))
    return 0


if __name__ == "__main__":
    sys.exit(main())
