#!/usr/bin/env python3
"""C-MS1: the terminal-stage phase of the MS runner under legacy settings reproduces v2.0's Phase A.

Compares, per (q, seed), the new run ``<new>/q<q>/seed<seed>/<arm>`` (arm ``MS_base``) with a reference
Phase-A run ``<ref>/q<q>/seed<seed>`` (``parents_A``: a ``phase_A`` run, or ``rehearsal_v2_0``: a locked run;
``--ref-kind``), on the v2.0 training-relevant state at the end of Phase A / of the terminal-stage phase:

  * ``state_end_stage2.pt`` (new) against ``state_end_A.pt`` (ref): actor, critic, lagged opponent, frozen
    snapshot (None on both sides), both Adam states, the minibatch stream, the env / learner / opponent / start
    streams, the torch generator, ``snapshot_refreshes``, the counters (``global_u``, episodes, transitions);
  * every weight export ``weights/u0025 ... u1600`` bit for bit (same set of files);
  * the per-update training series (updates 1..1600) of ``train_history.json`` on the keys both writers
    share (losses, KL, clip fraction, advantage statistics, gradient norms, return, LR, snapshot flag, the
    learner's mean effort) and the stream positions of ``v2_updates.csv`` / ``ms_updates.csv``;
  * the end-of-phase evaluation arrays on both verifier tiers (``freeze_stage2_{tier}.npz`` against
    ``final_{tier}.npz`` / ``gateA_{tier}.npz``), bit for bit.

The three process-global RNG states are excluded (as in v2.0: ``parents_A`` did not seed them). A missing file on
either side is a difference (reported), never skipped.

Usage:
    python tools/ms/cms1_compare.py --ref <.../parents_A> --ref-kind parentsA \
        --new results/ms_r1/base --out results/ms_r1/base_checks.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))

import cr1_compare as C  # noqa: E402

QS, SEEDS = (50, 60), tuple(range(10501, 10511))
AGENT_KEYS = ("actor", "critic", "opponent", "frozen", "opt_actor", "opt_critic", "rng_minibatch")
LAST_UPDATE = 1600
NOT_SHARED = {"phase", "stage", "mean_effort_by_stage", "mean_effort", "n_transitions"}
RNGPOS = ("rngpos_env", "rngpos_learn", "rngpos_opp", "rngpos_start", "rngpos_minibatch")
UPDATE_COLS = RNGPOS + ("kl_final_epoch", "clip_frac", "policy_loss", "value_loss")
EVAL_PAIRS = {"parentsA": (("final_final.npz", "freeze_stage2_final.npz"),
                           ("final_development.npz", "freeze_stage2_development.npz")),
              "rehearsal": (("gateA_final.npz", "freeze_stage2_final.npz"),
                            ("gateA_development.npz", "freeze_stage2_development.npz"))}


def _history(path: Path) -> List[Dict[str, Any]]:
    return C.load_json(path)["history"]


def _series_diff(ref_dir: Path, new_dir: Path, terminal_stage: int, last: int = LAST_UPDATE) -> Optional[C.Diff]:
    """Per-update training series of the terminal-stage phase (updates 1..1600), shared keys, exact."""
    ra = {int(h["update"]): h for h in _history(ref_dir / "train_history.json") if int(h["update"]) <= last}
    rb = {int(h["update"]): h for h in _history(new_dir / "train_history.json") if int(h["update"]) <= last}
    if sorted(ra) != sorted(rb) or len(ra) != last:
        return C.Diff("train_history.updates", f"{len(ra)} updates", f"{len(rb)} updates")
    for u in sorted(ra):
        a, b = ra[u], rb[u]
        keys = sorted((set(a) & set(b)) - NOT_SHARED)
        d = C.first_diff({k: a[k] for k in keys}, {k: b[k] for k in keys}, f"history[u{u}]")
        if d is not None:
            return d
        d = C.first_diff(a["mean_effort_by_stage"][str(terminal_stage)], b["mean_effort"], f"history[u{u}].mean_effort")
        if d is not None:
            return d
    return None


def _updates_csv_diff(ref_dir: Path, new_dir: Path, last: int = LAST_UPDATE) -> Optional[C.Diff]:
    ra = C.read_csv_text(ref_dir / "v2_updates.csv")
    rb = C.read_csv_text(new_dir / "ms_updates.csv")
    ra = ra[ra["update"].astype(int) <= last].reset_index(drop=True)
    rb = rb[rb["update"].astype(int) <= last].reset_index(drop=True)
    if ra["update"].tolist() != rb["update"].tolist():
        return C.Diff("updates.csv.update", f"{len(ra)} rows", f"{len(rb)} rows")
    for col in UPDATE_COLS:
        if col not in ra.columns or col not in rb.columns:
            return C.Diff(f"updates.csv.{col}", "present" if col in ra.columns else "absent",
                          "present" if col in rb.columns else "absent")
        bad = (ra[col] != rb[col]).to_numpy()
        if bad.any():
            i = int(bad.argmax())
            return C.Diff(f"updates.csv[update={ra['update'][i]}].{col}", ra[col][i], rb[col][i],
                          f"{int(bad.sum())} cells differ")
    return None


def compare_run(ref_dir: Path, new_dir: Path, ref_kind: str, terminal_stage: int = 2,
                last_update: int = LAST_UPDATE) -> Dict[str, Any]:
    """C-MS1 comparison of one (q, seed) (``last_update`` = the length of the terminal-stage phase, 1600)."""
    res = C.Result()
    ref_dir, new_dir = Path(ref_dir), Path(new_dir)
    if not new_dir.is_dir():
        res.check("run_dir_present", lambda: C.Diff(str(new_dir), "present", "missing"))
        return res.as_dict()
    cache: Dict[str, Any] = {}

    def both() -> Any:
        if not cache:
            cache["r"] = C.load_state(ref_dir / "state_end_A.pt")
            cache["n"] = C.load_state(new_dir / f"state_end_stage{terminal_stage}.pt")
        return cache["r"], cache["n"]

    for k in AGENT_KEYS:
        res.check(f"state:{k}", lambda k=k: C.first_diff(both()[0]["agent"].get(k), both()[1]["agent"].get(k), k))
    res.check("state:snapshot_refreshes", lambda: C.first_diff(
        both()[0]["agent"].get("snapshot_refreshes"), both()[1]["agent"].get("snapshot_refreshes"),
        "snapshot_refreshes"))
    res.check("state:conc_scale", lambda: C.first_diff(C.norm_conc(both()[0]["agent"].get("conc_scale")),
                                                       C.norm_conc(both()[1]["agent"].get("conc_scale")), "conc_scale"))
    res.check("state:rng_streams", lambda: C.first_diff(both()[0]["rng"], both()[1]["rng"], "rng"))
    res.check("state:torch_generator", lambda: C.first_diff(both()[0]["torch_generator_state"],
                                                            both()[1]["torch_generator_state"], "torch_generator_state"))
    res.check("state:counters", lambda: C.first_diff(
        {k: both()[0]["counters"][k] for k in ("global_u", "total_episodes", "total_transitions")},
        {k: both()[1]["counters"][k] for k in ("global_u", "total_episodes", "total_transitions")}, "counters"))
    spec = C.ModeSpec("cms1", 1, last_update, (), "A", (), snap="all")
    res.check("weight_exports", lambda: C._weights_diff(ref_dir, new_dir, spec, res.info))
    res.check("train_history:series", lambda: _series_diff(ref_dir, new_dir, terminal_stage, last_update))
    res.check("updates_csv:stream_positions_and_losses", lambda: _updates_csv_diff(ref_dir, new_dir, last_update))
    for ref_name, new_name in EVAL_PAIRS[ref_kind]:
        res.check(f"eval:{new_name}", lambda r=ref_name, n=new_name: C.first_diff(
            C.load_npz(ref_dir / r), C.load_npz(new_dir / n), n))
    return res.as_dict()


def compare_roots(ref: Path, new: Path, ref_kind: str, arm: str, qs: Sequence[int],
                  seeds: Sequence[int]) -> Dict[str, Any]:
    """Compare every (q, seed); the summary written by the CLI."""
    runs = []
    for q in qs:
        for s in seeds:
            r = compare_run(C.run_dir(ref, q, s, None), C.run_dir(new, q, s, arm), ref_kind)
            runs.append({"q": int(q), "seed": int(s), **r})
    n_ok = sum(1 for r in runs if r["ALL"])
    return {"tool": "tools/ms/cms1_compare.py", "ref": str(ref), "ref_kind": ref_kind, "new": str(new),
            "arm": arm, "n": len(runs), "n_identical": n_ok, "ALL": bool(runs) and n_ok == len(runs),
            "failing_fields": sorted({k for r in runs for k, v in r["fields"].items() if not v}),
            "first_differences": [{"q": r["q"], "seed": r["seed"], **r["first_difference"]}
                                  for r in runs if r["first_difference"]],
            "runs": runs}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI; exit code 0 iff every run is identical."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ref", required=True, help="dir containing q*/seed* of the reference Phase-A runs")
    p.add_argument("--ref-kind", choices=sorted(EVAL_PAIRS), default="parentsA")
    p.add_argument("--new", required=True, help="dir containing q*/seed*/<arm>")
    p.add_argument("--arm", default="MS_base")
    p.add_argument("--out", default=None)
    p.add_argument("--qs", type=int, nargs="+", default=list(QS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = p.parse_args(argv)
    summary = compare_roots(Path(a.ref), Path(a.new), a.ref_kind, a.arm, a.qs, a.seeds)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=1)
    print(f"C-MS1 ({a.ref_kind}) identical {summary['n_identical']}/{summary['n']} ALL={summary['ALL']}")
    for fd in summary["first_differences"]:
        print(f"  q={fd['q']} seed={fd['seed']} first differing field {fd['field']}: path={fd['path']} "
              f"ref={fd['ref']} new={fd['new']} {fd.get('note', '')}")
    return 0 if summary["ALL"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
