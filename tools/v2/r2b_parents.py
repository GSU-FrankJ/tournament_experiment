#!/usr/bin/env python3
"""R2b section 2.3: record the parents and comparator runs of every (q, seed) in ``parents.csv``.

For every (q, seed) of the development block (q in {50, 60} x 10501-10510):

  * the wave-P parent ``rehearsal_v1_1/q*/seed*/state_end_A.pt`` of the canonical worktree: path, SHA-256;
  * the R1 ``parents_A`` run directory (the wave-A baseline), its manifest commit / dirty flag, its
    ``parents_A_checks.json`` verdict (end-of-A state bit-identical to ``rehearsal_v1_1``), and whether the
    end-of-A evaluation files exist;
  * the R1 ``A_ctrl200`` and ``A_detmean`` run directories (wave-P comparators), their manifest commit and
    dirty flag, their final state and whether the evaluation files exist.

A missing reference file is an error (stop and report), never an empty field. Read-only.

Usage: python tools/v2/r2b_parents.py --out results/v2_refine_r2b/parents.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

CAN = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2")
R1 = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_refine")
QS, SEEDS = (50, 60), tuple(range(10501, 10511))
EVAL_FILES = ("final_final.npz", "final_development.npz", "final_v2.json", "manifest.json", "status.json")
FIELDS = ["q", "seed", "waveP_parent_path", "waveP_parent_sha256", "parentsA_dir", "parentsA_manifest_commit",
          "parentsA_manifest_dirty", "parentsA_checks_ALL", "parentsA_checks_file", "parentsA_eval_files_present",
          "A_ctrl200_dir", "A_ctrl200_manifest_commit", "A_ctrl200_manifest_dirty", "A_ctrl200_status",
          "A_ctrl200_eval_files_present", "A_detmean_dir", "A_detmean_manifest_commit", "A_detmean_manifest_dirty",
          "A_detmean_status", "A_detmean_eval_files_present", "A_ctrl200_parent_sha256", "A_detmean_parent_sha256"]


def sha256_file(p: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def need(p: Path) -> Path:
    """The path, or FileNotFoundError."""
    if not p.exists():
        raise FileNotFoundError(f"required reference file is missing: {p}")
    return p


def run_record(d: Path, prefix: str) -> Dict[str, Any]:
    """Manifest commit / dirty, status and the evaluation files of one run directory."""
    man = json.load(open(need(d / "manifest.json")))
    st = json.load(open(need(d / "status.json")))
    present = [f for f in EVAL_FILES if (d / f).exists()]
    if len(present) != len(EVAL_FILES):
        raise FileNotFoundError(f"{d}: missing {sorted(set(EVAL_FILES) - set(present))}")
    return {f"{prefix}_dir": str(d), f"{prefix}_manifest_commit": man["git"]["commit"],
            f"{prefix}_manifest_dirty": man["git"]["dirty"], f"{prefix}_status": f"{st['state']}:{st.get('exit_code')}",
            f"{prefix}_eval_files_present": len(present), f"{prefix}_parent_sha256": man["parent_sha256"]}


def build_rows(can: Path = CAN, r1: Path = R1) -> List[Dict[str, Any]]:
    """One row per (q, seed); ``can`` = canonical worktree root, ``r1`` = the R1 results root."""
    rehearsal = can / "results" / "v2_T2_locked" / "rehearsal_v1_1"
    checks_path = need(r1 / "parents_A_checks.json")
    checks = json.load(open(checks_path))
    by_run = {(r["q"], r["seed"]): r for r in checks["runs"]}
    rows = []
    for q in QS:
        for s in SEEDS:
            parent = need(rehearsal / f"q{q}" / f"seed{s}" / "state_end_A.pt")
            pa = r1 / "parents_A" / f"q{q}" / f"seed{s}"
            man = json.load(open(need(pa / "manifest.json")))
            present = [f for f in EVAL_FILES if (pa / f).exists()]
            if len(present) != len(EVAL_FILES):
                raise FileNotFoundError(f"{pa}: missing {sorted(set(EVAL_FILES) - set(present))}")
            row: Dict[str, Any] = {
                "q": q, "seed": s, "waveP_parent_path": str(parent), "waveP_parent_sha256": sha256_file(parent),
                "parentsA_dir": str(pa), "parentsA_manifest_commit": man["git"]["commit"],
                "parentsA_manifest_dirty": man["git"]["dirty"], "parentsA_checks_ALL": by_run[(q, s)]["ALL"],
                "parentsA_checks_file": str(checks_path), "parentsA_eval_files_present": len(present)}
            row.update(run_record(r1 / "stage2" / f"q{q}" / f"seed{s}" / "A_ctrl200", "A_ctrl200"))
            row.update(run_record(r1 / "stage2" / f"q{q}" / f"seed{s}" / "A_detmean", "A_detmean"))
            for arm in ("A_ctrl200", "A_detmean"):   # the comparators started from this very parent state
                if row[f"{arm}_parent_sha256"] != row["waveP_parent_sha256"]:
                    raise ValueError(f"{arm} q{q} seed{s}: its manifest parent_sha256 differs from the parent file")
            rows.append(row)
    return rows


def main() -> int:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--can-root", default=str(CAN), help="canonical worktree (rehearsal_v1_1 parents)")
    p.add_argument("--r1-root", default=str(R1), help="R1 results root (parents_A, stage2 comparators)")
    a = p.parse_args()
    rows = build_rows(Path(a.can_root), Path(a.r1_root))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["parentsA_checks_ALL"] is not True]
    print(f"wrote {out}: {len(rows)} rows; parents_A checks ALL true in {len(rows) - len(bad)}/{len(rows)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
