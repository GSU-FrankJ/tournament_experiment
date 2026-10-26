"""Shared helpers of the P-FIG round (paper/multistage_rev): paths, single-thread guard, run roots, loaders.

Evaluation only: nothing here trains, and no stored record or run file is written. The extraction scripts
(``extract_F*.py``) import this module; the plot scripts (``F*.py``) do not (they read ``data/F*.csv`` only).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent                      # paper/multistage_rev
ROOT = OUT.parents[1]                  # repository (worktree) root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

WT = Path("/home/fjiang4/tournament_experiment/.claude/worktrees")
R1_ROOT = WT / "t3-bw-vs-jt-51e472" / "results" / "t3_r1"                       # untracked, read-only
R4_ROOT = WT / "t3-r4-lock-confirm" / "results" / "t3_r4" / "confirmation"       # untracked, read-only
V20_ROOT = WT / "v2-t2-refine" / "results" / "v2_T2_locked" / "confirmation_v2_0"  # untracked, read-only

PER_RUN_R1 = ROOT / "results" / "t3_r1" / "eval" / "per_run.csv"
PER_RUN_R4 = ROOT / "results" / "t3_r4" / "confirmation" / "eval" / "per_run.csv"
CALIB = ROOT / "results" / "v2_T2_locked" / "calibration" / "calibration_locked.csv"

QS = (50, 60)
K, W_H, W_L = 1.0 / 3500.0, 6.0, 2.0


def single_thread() -> None:
    """Pin torch to one thread (call once per process, before any evaluation)."""
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def git_commit() -> str:
    """Full SHA of the commit the worktree is at (``git rev-parse HEAD``)."""
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip()


def read_json(path: Path) -> Dict[str, Any]:
    """A JSON file as a dict."""
    with open(path) as f:
        return json.load(f)


def v20_run_dir(q: int, seed: int) -> Path:
    """Run directory of a v2.0 confirmation run."""
    return V20_ROOT / f"q{q}" / f"seed{seed}"


def r4_run_dir(q: int, seed: int) -> Path:
    """Run directory of a T3-R4 confirmation run (arm ``bwA``)."""
    return R4_ROOT / f"q{q}" / f"seed{seed}" / "bwA"


def r1_run_dir(q: int, seed: int, arm: str) -> Path:
    """Run directory of a T3-R1 run (arms ``bw``, ``jt``, ``jt_es``)."""
    return R1_ROOT / "runs" / f"q{q}" / f"seed{seed}" / arm


def load_per_run(path: Path) -> pd.DataFrame:
    """A per-run CSV with round-trip float parsing (so stored values compare bit for bit)."""
    return pd.read_csv(path, float_precision="round_trip")


def fmt(x: float) -> str:
    """Round-trip text of a float for the data CSVs."""
    return repr(float(x))
