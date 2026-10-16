#!/usr/bin/env python3
"""Lists (and with ``--add`` stages) the files of ``results/t3_r1/`` that the tracking rules allow.

Rules (prompt "Results", ``reports/ms/README.md``): for ``bw`` runs what the MS rules track for MS runs (``manifest.json``,
``run_config.json``, ``status.json``, ``gates.json``, ``rule_log.json``, ``drift_test.json``, ``ms_run_summary.json``, the check CSVs
``ms_checks_stage*.csv``, the bin maps ``ms_binmaps_stage*.npz``, ``ms_updates.csv`` and the continuation tables
``continuation_table_stage*.npz``); for joint runs ``run_config.json``, ``manifest.json``, ``status.json``,
``joint_run_summary.json``, ``joint_checks.csv``, ``joint_updates.csv`` and ``visitation.npz``. Never tracked: ``.pt`` files,
``train_history.json``, weight exports, ``freeze_stage*_*.npz``, ``final_eval_*.npz``, the per-run NPZ arrays of
``tools/t3/t3_eval.py`` (``eval/arrays/``) and ``run.log``. Outside the run directories everything else under the root is
listed (launch records, check JSONs, tables), minus the never-tracked classes.

Usage:
    python tools/t3/t3_track.py --root results/t3_r1 [--subdir smoke] [--add]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]

BW_FILES = ["manifest.json", "run_config.json", "status.json", "gates.json", "rule_log.json", "drift_test.json",
            "ms_run_summary.json", "ms_checks_stage*.csv", "ms_binmaps_stage*.npz", "ms_updates.csv",
            "continuation_table_stage*.npz"]
JOINT_FILES = ["run_config.json", "manifest.json", "status.json", "joint_run_summary.json", "joint_checks.csv",
               "joint_updates.csv", "visitation.npz"]
NEVER = ["*.pt", "*.pth", "*.log", "train_history.json", "freeze_stage*_*.npz", "final_eval_*.npz", "weights/*",
         "arrays/*", "*/arrays/*", "__pycache__/*"]


def run_dirs(base: Path) -> List[Path]:
    """Directories that hold a run (``run_config.json`` and ``status.json``)."""
    return sorted({p.parent for p in base.rglob("run_config.json") if (p.parent / "status.json").exists()})


def is_never(rel: str) -> bool:
    """True if a path (relative to the root) belongs to a never-tracked class."""
    return any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(Path(rel).name, pat) for pat in NEVER) or "/weights/" in rel \
        or "/arrays/" in rel or rel.startswith("arrays/")


def files_to_track(base: Path) -> List[Path]:
    """All files under ``base`` that may be tracked."""
    rd = run_dirs(base)
    out: List[Path] = []
    in_run = set()
    for d in rd:
        cfg = json.load(open(d / "run_config.json"))
        pats = JOINT_FILES if cfg.get("schema") == "joint_run_config/1" else BW_FILES
        for pat in pats:
            out += sorted(d.glob(pat))
        in_run.add(d)
    for p in sorted(base.rglob("*")):
        if not p.is_file() or any(d in p.parents for d in in_run):
            continue
        if not is_never(str(p.relative_to(base))):
            out.append(p)
    return out


def main() -> int:
    """List or stage."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results" / "t3_r1"))
    ap.add_argument("--subdir", default=None)
    ap.add_argument("--add", action="store_true")
    a = ap.parse_args()
    base = Path(a.root) / a.subdir if a.subdir else Path(a.root)
    files = files_to_track(base)
    print(f"{len(files)} trackable files under {base}")
    if a.add:
        for i in range(0, len(files), 500):
            subprocess.run(["git", "-C", str(ROOT), "add", "--"] + [str(f) for f in files[i:i + 500]], check=True)
        print("staged")
    else:
        for f in files[:20]:
            print(" ", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
