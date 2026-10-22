#!/usr/bin/env python3
"""C-PAIR comparator of round T3-R3: a ``bwA`` run against the ``bw`` run of the same (q, seed).

* **The trigger did not fire** (no ``retry.json``): the two runs are bit-identical: the weight exports, the freeze arrays, the
  continuation tables, the bin maps, the CSVs without their wall-clock columns, the final networks and stream states
  (the checks of ``tools/t3/compare_runs.py bw``, returned as data), the set of files is the same, and the candidate run has no
  ``attempt1/``.
* **The trigger fired**: attempt 1 (``attempt1/``) equals the ``bw`` run's stage 3 up to the check: the stage-3 exports
  ``u00025`` ... ``u01600`` (as many as the stage-3 budget), the stage-3 rows of ``ms_updates.csv`` (``attempt1/ms_updates_stage3.csv``)
  and ``ms_checks_stage3.csv`` without wall-clock columns, the stage-3 bin maps, and the actor / critic / opponent / optimiser /
  stream states of ``attempt1/state_attempt1_end.pt`` against ``state_end_stage3.pt`` of the ``bw`` run.

Usage:
    python tools/t3/t3r3_pair.py <bw_run_dir> <candidate_run_dir>
Exit code 0 iff the pair is consistent with the rule above.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

# ``tools/t3/compare_runs.py`` by path (``tools/v2/compare_runs.py`` has the same module name)
_spec = importlib.util.spec_from_file_location("t3_compare_runs", Path(__file__).with_name("compare_runs.py"))
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)  # type: ignore[union-attr]

WALL_UPDATES = {"update_wall_sec"}
WALL_CHECKS = {"verifier_sec", "run"}        # the check files carry the run name, which differs between arms and between rounds
# the CSVs of ``compare_runs.CSV["bw"]`` with the run name dropped as well
CSV_BW = {"ms_updates.csv": WALL_UPDATES, "ms_checks_stage3.csv": WALL_CHECKS, "ms_checks_stage2.csv": WALL_CHECKS,
          "ms_checks_stage1.csv": WALL_CHECKS}


def read_json(path: Path) -> Dict[str, Any]:
    """A JSON file as a dict."""
    with open(path) as f:
        return json.load(f)


def file_set(d: Path) -> List[str]:
    """Relative paths of all files under ``d`` (``attempt1/`` and ``retry.json`` excluded: the scheme's own records)."""
    out = []
    for p in sorted(Path(d).rglob("*")):
        if p.is_file():
            rel = p.relative_to(d).as_posix()
            if rel.startswith("attempt1/") or rel == "retry.json":
                continue
            out.append(rel)
    return out


def _rows(path: Path, drop: set, stage: Optional[str] = None) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        rows = [{k: v for k, v in r.items() if k not in drop} for r in csv.DictReader(f)]
    return [r for r in rows if stage is None or r.get("stage") == stage]


def compare_full(a: Path, b: Path) -> Dict[str, Any]:
    """The checks of ``compare_runs.py bw`` on two complete runs, as data (``ok`` = all identical)."""
    a, b = Path(a), Path(b)
    res: Dict[str, Any] = {"npz": {}, "csv": {}}
    ok = True
    for pat in cr.NPZ["bw"]:
        names_a = sorted(str(f.relative_to(a)) for f in a.glob(pat))
        names_b = sorted(str(f.relative_to(b)) for f in b.glob(pat))
        res["npz"][pat] = {"files": len(names_a), "same_file_set": names_a == names_b, "different": []}
        if names_a != names_b:
            ok = False
        for fa in sorted(a.glob(pat)):
            fb = b / fa.relative_to(a)
            if not fb.exists():
                continue
            za, zb = np.load(fa), np.load(fb)
            if not (za.files == zb.files and all(cr.same_array(za[k], zb[k]) for k in za.files)):
                res["npz"][pat]["different"].append(str(fa.relative_to(a)))
                ok = False
    for name, wall in CSV_BW.items():
        same = cr.rows(a / name, wall) == cr.rows(b / name, wall)
        res["csv"][name] = same
        ok = ok and same
    sa = torch.load(a / cr.PT["bw"], weights_only=False, map_location="cpu")
    sb = torch.load(b / cr.PT["bw"], weights_only=False, map_location="cpu")
    nets = [(sa["agent"]["actor"], sb["agent"]["actor"]), (sa["agent"]["critic"], sb["agent"]["critic"])]
    same = all(torch.equal(x[k], y[k]) for x, y in nets for k in x) and sa["rng"] == sb["rng"]
    same = same and all(torch.equal(sa["frozen_stages"][s][k], sb["frozen_stages"][s][k])
                        for s in sa["frozen_stages"] for k in sa["frozen_stages"][s])
    res["final_state_identical"] = bool(same)
    res["same_file_set"] = file_set(a) == file_set(b)
    res["ok"] = bool(ok and same and res["same_file_set"])
    return res


def _tree_equal(x: Any, y: Any) -> bool:
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
    if isinstance(x, np.ndarray):
        return isinstance(y, np.ndarray) and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=True)
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(_tree_equal(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(_tree_equal(p, q) for p, q in zip(x, y))
    return x == y


def compare_first_attempt(orig: Path, cand: Path) -> Dict[str, Any]:
    """Attempt 1 of the candidate run ``cand`` (``attempt1/``) against the stage-3 phase of the run ``orig`` (a ``bw`` run of the same
    config and seed)."""
    orig, cand = Path(orig), Path(cand)
    adir = cand / "attempt1"
    cfg = read_json(cand / "run_config.json")
    every = int(cfg["record"]["protocol"]["weights_every"])
    n3 = int(cfg["pipeline"]["budgets"]["3"])
    ups = list(range(every, n3 + 1, every))
    bad_w: List[str] = []
    for u in ups:
        fa, fb = adir / "weights" / f"u{u:05d}.npz", orig / "weights" / f"u{u:05d}.npz"
        if not (fa.is_file() and fb.is_file()):
            bad_w.append(f"u{u:05d}: missing")
            continue
        za, zb = np.load(fa), np.load(fb)
        if not (za.files == zb.files and all(cr.same_array(za[k], zb[k]) for k in za.files)):
            bad_w.append(f"u{u:05d}")
    extra = sorted(p.name for p in (adir / "weights").glob("u*.npz") if p.name not in {f"u{u:05d}.npz" for u in ups})
    res: Dict[str, Any] = {"exports_compared": len(ups), "exports_bad": bad_w, "exports_extra_in_attempt1": extra}
    res["updates_rows"] = _rows(adir / "ms_updates_stage3.csv", WALL_UPDATES) == _rows(orig / "ms_updates.csv", WALL_UPDATES, "3")
    res["checks"] = _rows(adir / "ms_checks_stage3.csv", WALL_CHECKS) == _rows(orig / "ms_checks_stage3.csv", WALL_CHECKS)
    za, zb = np.load(adir / "ms_binmaps_stage3.npz"), np.load(orig / "ms_binmaps_stage3.npz")
    res["binmaps"] = bool(za.files == zb.files and all(cr.same_array(za[k], zb[k]) for k in za.files))
    sa = torch.load(adir / "state_attempt1_end.pt", weights_only=False, map_location="cpu")
    sb = torch.load(orig / "state_end_stage3.pt", weights_only=False, map_location="cpu")
    keys = ("actor", "critic", "opponent", "opt_actor", "opt_critic", "rng_minibatch", "snapshot_refreshes")
    res["state"] = bool(_tree_equal({k: sa["agent"][k] for k in keys}, {k: sb["agent"][k] for k in keys}) and sa["rng"] == sb["rng"])
    res["ok"] = bool(not bad_w and not extra and res["updates_rows"] and res["checks"] and res["binmaps"] and res["state"])
    return res


def compare_pair(bw_dir: Path, cand_dir: Path) -> Dict[str, Any]:
    """The C-PAIR verdict of one pair: ``fired`` (a ``retry.json`` exists) decides which comparison applies."""
    cand = Path(cand_dir)
    fired = (cand / "retry.json").is_file()
    if fired:
        res = compare_first_attempt(Path(bw_dir), cand)
        res.update(fired=True, attempt1_present=(cand / "attempt1").is_dir())
        res["ok"] = bool(res["ok"] and res["attempt1_present"])
        return res
    res = compare_full(Path(bw_dir), cand)
    res.update(fired=False, attempt1_present=(cand / "attempt1").exists())
    res["ok"] = bool(res["ok"] and not res["attempt1_present"])
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print(__doc__)
        return 2
    res = compare_pair(Path(argv[0]), Path(argv[1]))
    print(json.dumps(res, indent=1, default=str))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
