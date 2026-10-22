"""Tests of ``tools/t3/t3r3_last16.py``: which runs it evaluates, the rank of the failing value, the refusal of a relative root."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "ms", ROOT / "tools" / "t3"):
    sys.path.insert(0, str(_p))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import t3r3_last16 as l16  # noqa: E402


def test_rank_from_top_counts_strictly_larger_values():
    assert l16.rank_from_top([1.0, 2.0, 3.0], 3.0) == 1
    assert l16.rank_from_top([1.0, 2.0, 3.0], 2.0) == 2
    assert l16.rank_from_top([1.0, 3.0, 3.0], 3.0) == 1            # a tie is not above
    assert l16.rank_from_top([1.0, 2.0, 3.0], 0.5) == 4


def test_only_failing_development_runs_are_evaluated(tmp_path):
    rows = [{"run": "a", "run_kind": "dev", "passed": "True"}, {"run": "b", "run_kind": "dev", "passed": "False"},
            {"run": "c", "run_kind": "diagnostic", "passed": "False"}]
    p = tmp_path / "per_run.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert [r["run"] for r in l16.failing_runs(p)] == ["b"]


def test_a_relative_root_is_refused(tmp_path):
    with pytest.raises(SystemExit) as exc:
        l16.main(["--root", "results/t3_r3", "--per-run", str(tmp_path / "x.csv"), "--out", str(tmp_path)])
    assert exc.value.code == 2
