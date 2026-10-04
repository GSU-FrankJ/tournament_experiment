#!/usr/bin/env python3
"""C-R2: the unchanged v2.0 entry point, run from the R2b branch, reproduces ``rehearsal_v2_0``.

Compares a new run root with the reference root (the v2.0 re-rehearsal, an explicit path: its large
files are untracked and live in the worktree that produced them) on the v2.0 training-relevant state:

  * everything ``tools/v2/cr1_compare.py --mode cr1`` compares (state_end_A.pt / state_end_B.pt with
    actor, critic, opponent, frozen snapshot, both Adam states, the minibatch stream, the four numpy
    streams and the torch generator; all weight exports and ``checkpoint_weights.npz``; histories,
    stability, verifier calls, curriculum, snapshots; ``v2_updates.csv`` and ``v2_checkpoints_{A,B}.csv``
    on the common columns; the gate metric values; the evaluation NPZs; ``induced_band.json``,
    ``drift_test.json``);
  * the continuation table: ``continuation_table.npz`` (``y_grid`` and ``values``) bit for bit and the
    file's SHA-256, and the rule record of ``gates.json`` without its wall-clock fields;
  * the v2.0 verdict fields of ``gates.json``: G-S, S1, the v1.1 and v1.0 outcomes, run pass, outcome.

Excluded, as in v2.0: manifests, wall-clock fields and the three process-global RNG states. A missing
reference or new file is a difference (reported with the first differing field), never skipped.

Usage:
    python tools/v2/cr2_compare.py --ref <rehearsal_v2_0 root> --new results/v2_refine_r2b/v20_reproduction \
        --out results/v2_refine_r2b/v20_reproduction_checks.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))

import cr1_compare as C  # noqa: E402

REF_DEFAULT = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0")
QS, SEEDS = (50, 60), tuple(range(10501, 10511))
VERDICT_KEYS = ("G-S", "S1", "v1_1_outcome", "v1_0_outcome", "run_pass", "outcome")
WALL_SUBSTR = ("sec", "wall", "time")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _drop_wall(x: Any) -> Any:
    """The object without keys that name wall-clock quantities."""
    if isinstance(x, dict):
        return {k: _drop_wall(v) for k, v in x.items() if not any(s in str(k).lower() for s in WALL_SUBSTR)}
    if isinstance(x, list):
        return [_drop_wall(v) for v in x]
    return x


def compare_run(ref_dir: Path, new_dir: Path) -> Dict[str, Any]:
    """C-R2 comparison of one (q, seed): the cr1 fields plus the v2.0 additions."""
    base = C.compare_run("cr1", ref_dir, new_dir)
    res = C.Result()
    for name, ok in base["fields"].items():
        res.fields[name] = bool(ok)
        res._in_all[name] = True
        if name in base["differences"]:
            res.diffs[name] = C.Diff(**base["differences"][name])
    res.info.update(base["info"])

    def table() -> Optional[C.Diff]:
        a, b = np.load(ref_dir / "continuation_table.npz"), np.load(new_dir / "continuation_table.npz")
        if set(a.files) != set(b.files):
            return C.Diff("continuation_table.npz{files}", str(sorted(a.files)), str(sorted(b.files)))
        for k in sorted(a.files):
            d = C._array_diff(a[k], b[k], f"continuation_table.{k}")
            if d is not None:
                return d
        sa, sb = _sha(ref_dir / "continuation_table.npz"), _sha(new_dir / "continuation_table.npz")
        return None if sa == sb else C.Diff("continuation_table.npz.sha256", sa, sb)

    def gates_v20() -> Optional[C.Diff]:
        ga, gb = C.load_json(ref_dir / "gates.json"), C.load_json(new_dir / "gates.json")
        for k in VERDICT_KEYS:
            d = C.first_diff(ga.get(k), gb.get(k), f"gates.{k}")
            if d is not None:
                return d
        return C.first_diff(_drop_wall(ga.get("continuation_table")), _drop_wall(gb.get("continuation_table")),
                            "gates.continuation_table")

    def status() -> Optional[C.Diff]:
        sa, sb = C.load_json(ref_dir / "status.json"), C.load_json(new_dir / "status.json")
        return C.first_diff([sa.get("state"), sa.get("exit_code")], [sb.get("state"), sb.get("exit_code")], "status")

    res.check("continuation_table", table)
    res.check("gates_v20_verdict_fields", gates_v20)
    res.check("status_done_exit_0", status)
    return res.as_dict()


def compare_roots(ref: Path, new: Path, qs: Sequence[int], seeds: Sequence[int]) -> Dict[str, Any]:
    """Compare every (q, seed); the summary written by the CLI."""
    runs = []
    for q in qs:
        for s in seeds:
            r = compare_run(C.run_dir(ref, q, s, None), C.run_dir(new, q, s, None))
            runs.append({"q": int(q), "seed": int(s), **r})
    n_ok = sum(1 for r in runs if r["ALL"])
    return {"tool": "tools/v2/cr2_compare.py", "ref": str(ref), "new": str(new), "n": len(runs),
            "n_identical": n_ok, "ALL": bool(runs) and n_ok == len(runs),
            "failing_fields": sorted({k for r in runs for k, v in r["fields"].items() if not v}),
            "first_differences": [{"q": r["q"], "seed": r["seed"], **r["first_difference"]}
                                  for r in runs if r["first_difference"]],
            "runs": runs}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI; exit code 0 iff every run is identical."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ref", default=str(REF_DEFAULT))
    p.add_argument("--new", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--qs", type=int, nargs="+", default=list(QS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = p.parse_args(argv)
    summary = compare_roots(Path(a.ref), Path(a.new), a.qs, a.seeds)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=1)
    print(f"C-R2 identical {summary['n_identical']}/{summary['n']} ALL={summary['ALL']}")
    for fd in summary["first_differences"]:
        print(f"  q={fd['q']} seed={fd['seed']} first differing field {fd['field']}: path={fd['path']} "
              f"ref={fd['ref']} new={fd['new']} {fd.get('note', '')}")
    return 0 if summary["ALL"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
