#!/usr/bin/env python3
"""Post-launch checks of an MS-R1 wave (prompt section 3.1): ``<root>/launch_checks.json``.

For every planned run (arms x q x seeds) of a wave root (``results/ms_r1/pilot`` or ``.../base``):

  * ``status``       status.json is done with exit code 0 (a non-zero exit is a failed run, reported);
  * ``manifest``     manifest.json is at the code commit (``--code-commit``) with ``clean_tree`` true;
  * ``files``        the run has its rule record for every stage, its check tables, its bin maps (stages >= 2),
                     ``gates.json`` and every weight export u25, u50, ... up to the final update;
  * ``global_rng``   the global-RNG assertions of gates.json all passed (no violation);
  * ``start_shares`` the measured start counts per stratum (tail / near-tie / middle) of every block, and of every
                     landing window, equal the design shares within 3 binomial standard errors. The design
                     probabilities are those in force at every update (``ms_binmaps_stage{t}.npz``,
                     ``probs_first_update`` / ``probs_table``; uniform bins for the bin-balanced scheme), so the
                     polishing blocks are compared with their own focus. With many tests a few |z| > 3 are
                     expected by chance: the summary reports the number of tests, the number of exceedances and the
                     number expected under the null; a flag is not by itself an error;
  * ``tail_share``   the tail mass of every probability vector in force equals lambda_T (the coverage
                     constraint) to 1e-12, and the pooled measured tail share of every block is within 3 binomial
                     standard errors of it (same remark).

Check C-MS2 (addendum A1 of 2026-10-07; ``--cms2-ref``): the budget-matched control ``MS_base2400`` equals
``parents_A`` bit for bit through update 1201 -- the weight exports u0025 ... u1200, the per-update series (incl. the
learning rate: update 1201, the first of ``parents_A``'s LR window, still runs at 3e-4) and the five stream positions
after every update 1..1201 -- and its first export after that, u1225, differs from ``parents_A``'s. A failure is a
stop-and-report.

Usage:
    python tools/ms/launch_checks.py --root results/ms_r1/pilot --code-commit <sha> \
        --out results/ms_r1/pilot/launch_checks.json \
        [--cms2-ref <.../parents_A> [--cms2-arm MS_base2400]]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))

import cms1_compare as M1  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402

C = M1.C                                    # tools/v2/cr1_compare: Result, Diff, ModeSpec, first_diff, load_npz
WEIGHTS_EVERY = 25
CMS2_THROUGH = 1201                         # the first update of parents_A's LR window (v2.0 pipeline.lr_decay A)
Z_LIMIT = 3.0
STRATA = ("tail", "near", "mid")
REPORT_NEAR_TIE_HALF_WIDTH = 20.0


def _load(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _spec(cfg: Dict[str, Any]) -> GameSpec:
    g = cfg["record"]["game"]
    return GameSpec(**{k: g[k] for k in ("w_h", "w_l", "k", "q", "T", "e_min", "e_max")})


def per_update_probs(d: Path, t: int, n_bins: int, updates: np.ndarray) -> np.ndarray:
    """Bin probabilities in force at each global update of stage ``t`` (uniform if not stratified)."""
    f = d / f"ms_binmaps_stage{t}.npz"
    if f.exists():
        z = np.load(f)
        if "probs_table" in z.files:
            idx = np.searchsorted(z["probs_first_update"], updates, side="right") - 1
            if idx.min() < 0:
                raise ValueError(f"stage {t}: update {int(updates.min())} precedes the first recorded probabilities")
            return z["probs_table"][idx]
    return np.full((updates.size, n_bins), 1.0 / n_bins)


def start_share_tests(d: Path, cfg: Dict[str, Any], rule_log: Dict[str, Any]) -> Dict[str, Any]:
    """z-scores of the measured start counts per (stage, block, stratum) against the probabilities in force."""
    spec = _spec(cfg)
    bin_w = float(cfg["record"]["protocol"]["es_bin_width"])
    sw = cfg["start_weights"]
    hw = float(sw["near_tie_half_width"]) if sw["scheme"] == "stratified_priority" else REPORT_NEAR_TIE_HALF_WIDTH
    sampler = StartSampler(spec, bin_w)
    rows = list(csv.DictReader(open(d / "ms_updates.csv")))
    out: List[Dict[str, Any]] = []
    tail_mass_max_dev = 0.0
    for t in range(2, spec.T + 1):
        labels = sampler.stratum_labels(t, hw)
        n_bins = sampler.n_bins(t)
        lam_t = float(sampler.coverage_lambda_t(t))
        sel = [r for r in rows if int(r["stage"]) == t]
        upd = np.array([int(r["update"]) for r in sel])
        probs = per_update_probs(d, t, n_bins, upd)
        shares = np.stack([probs[:, labels == k].sum(axis=1) for k in (0, 1, 2)], axis=1)
        if sw["scheme"] == "stratified_priority":
            tail_mass_max_dev = max(tail_mass_max_dev, float(np.abs(shares[:, 0] - lam_t).max()))
        counts = np.array([[int(r["n_start_tail"]), int(r["n_start_near"]), int(r["n_start_mid"])] for r in sel])
        n = counts.sum(axis=1)
        groups: Dict[Tuple[int, str], List[int]] = {}
        for i, r in enumerate(sel):
            groups.setdefault((int(r["block_id"]), r["block_type"]), []).append(i)
        for (bid, btype), idx in sorted(groups.items()):
            ii = np.array(idx)
            for k, name in enumerate(STRATA):
                p = shares[ii, k]
                mean, var = float((n[ii] * p).sum()), float((n[ii] * p * (1.0 - p)).sum())
                obs = float(counts[ii, k].sum())
                z = (obs - mean) / math.sqrt(var) if var > 0 else (0.0 if abs(obs - mean) < 1e-9 else float("inf"))
                out.append({"stage": t, "block_id": bid, "block_type": btype, "stratum": name, "n_updates": len(idx),
                            "observed": obs, "expected": mean, "z": z, "design_tail_share": lam_t})
    n_tests = len(out)
    flagged = [o for o in out if abs(o["z"]) > Z_LIMIT]
    p_two = math.erfc(Z_LIMIT / math.sqrt(2.0))
    return {"n_tests": n_tests, "n_flagged_abs_z_gt_3": len(flagged), "expected_flagged_under_null": n_tests * p_two,
            "flagged": flagged, "max_abs_z": max((abs(o["z"]) for o in out), default=0.0),
            "tail_mass_max_deviation_from_lambda_T": tail_mass_max_dev,
            "tail_share_tests": [o for o in out if o["stratum"] == "tail"]}


def check_run(d: Path, code_commit: Optional[str]) -> Dict[str, Any]:
    """All checks of one run directory (a missing file is a failed check, never a skip)."""
    res: Dict[str, Any] = {"dir": str(d)}
    try:
        st = _load(d / "status.json")
        res["status"] = {"state": st.get("state"), "exit_code": st.get("exit_code"),
                         "ok": st.get("state") == "done" and st.get("exit_code") == 0}
    except Exception as exc:  # noqa: BLE001
        res["status"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return res
    try:
        man = _load(d / "manifest.json")
        res["manifest"] = {"commit": man.get("commit"), "clean_tree": man.get("clean_tree"),
                           "ok": bool(man.get("clean_tree") is True
                                      and (code_commit is None or man.get("commit") == code_commit))}
    except Exception as exc:  # noqa: BLE001
        res["manifest"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        cfg = _load(d / "run_config.json")
        spec = _spec(cfg)
        T = int(spec.T)
        rule = _load(d / "rule_log.json")
        gates = _load(d / "gates.json")
        missing: List[str] = []
        for t in range(1, T + 1):
            for f in (f"ms_checks_stage{t}.csv", f"state_end_stage{t}.pt"):
                if not (d / f).exists():
                    missing.append(f)
            if t >= 2 and not (d / f"ms_binmaps_stage{t}.npz").exists():
                missing.append(f"ms_binmaps_stage{t}.npz")
            e = rule["stages"].get(str(t))
            if not e or "blocks" not in e or "freeze" not in e:
                missing.append(f"rule_log.stages.{t}")
        final_u = int(st.get("final_global_update", 0))
        want = [f"u{u:05d}.npz" for u in range(WEIGHTS_EVERY, final_u + 1, WEIGHTS_EVERY)]
        have = sorted(p.name for p in (d / "weights").glob("u*.npz"))
        if have != want:
            missing.append(f"weights ({len(have)} found, {len(want)} expected)")
        res["files"] = {"ok": not missing, "missing": missing}
        rng = gates.get("global_rng", {})
        res["global_rng"] = {"ok": rng.get("status") == "ok" and not rng.get("violations"),
                             "status": rng.get("status"), "violations": rng.get("violations")}
        ss = start_share_tests(d, cfg, rule)
        res["start_shares"] = {k: ss[k] for k in ("n_tests", "n_flagged_abs_z_gt_3", "expected_flagged_under_null",
                                                  "max_abs_z", "flagged")}
        res["tail_share"] = {"tail_mass_max_deviation_from_lambda_T": ss["tail_mass_max_deviation_from_lambda_T"],
                             "coverage_ok": ss["tail_mass_max_deviation_from_lambda_T"] <= 1e-12,
                             "n_tests": len(ss["tail_share_tests"]),
                             "n_flagged_abs_z_gt_3": sum(1 for o in ss["tail_share_tests"] if abs(o["z"]) > Z_LIMIT)}
    except Exception as exc:  # noqa: BLE001 - reported
        res["files"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return res


def check_root(root: Path, arms: Sequence[str], qs: Sequence[int], seeds: Sequence[int],
               code_commit: Optional[str]) -> Dict[str, Any]:
    """Checks of every planned run under ``root`` and the summary."""
    runs = []
    for q in qs:
        for s in seeds:
            for arm in arms:
                d = root / f"q{q}" / f"seed{s}" / arm
                r = check_run(d, code_commit) if d.is_dir() else {"dir": str(d), "status": {"ok": False, "error": "missing"}}
                r.update(q=q, seed=s, arm=arm)
                runs.append(r)
    keys = ("status", "manifest", "files", "global_rng")
    summ = {k: sum(1 for r in runs if r.get(k, {}).get("ok")) for k in keys}
    summ["tail_share_coverage"] = sum(1 for r in runs if r.get("tail_share", {}).get("coverage_ok"))
    tests = sum(r.get("start_shares", {}).get("n_tests", 0) for r in runs)
    flagged = sum(r.get("start_shares", {}).get("n_flagged_abs_z_gt_3", 0) for r in runs)
    exp = sum(r.get("start_shares", {}).get("expected_flagged_under_null", 0.0) for r in runs)
    return {"tool": "tools/ms/launch_checks.py", "root": str(root), "code_commit": code_commit, "n_runs": len(runs),
            "n_ok_per_check": summ,
            "all_ok": bool(all(r.get(k, {}).get("ok") for r in runs for k in keys)
                           and all(r.get("tail_share", {}).get("coverage_ok") for r in runs)),
            "start_share_tests": {"n_tests": tests, "n_flagged": flagged, "expected_flagged_under_null": exp,
                                  "note": "a few flags are expected by chance at 3 standard errors; they are not "
                                          "errors by themselves"},
            "runs": runs}


def _first_export_after_differs(ref_dir: Path, new_dir: Path, through: int) -> Optional[Any]:
    """C-MS2 part 3: the first export after ``through`` exists on both sides and differs in at least one array
    (a missing file is a failure, not a difference)."""
    u = (through // WEIGHTS_EVERY + 1) * WEIGHTS_EVERY
    a, b = C.load_npz(ref_dir / "weights" / f"u{u:05d}.npz"), C.load_npz(new_dir / "weights" / f"u{u:05d}.npz")
    if C.first_diff(a, b, f"u{u:05d}.npz") is None:
        return C.Diff(f"weights/u{u:05d}.npz", "identical to the reference", "identical to the reference",
                      "expected to differ: the reference's learning rate has decayed from update 1202")
    return None


def cms2_run(ref_dir: Path, new_dir: Path, through: int = CMS2_THROUGH) -> Dict[str, Any]:
    """C-MS2 of one (q, seed): ``new_dir`` (``MS_base2400``) against ``ref_dir`` (``parents_A``).

    Args:
        ref_dir: ``parents_A/q<q>/seed<seed>`` (a v2.0 ``phase_A`` run).
        new_dir: the ``MS_base2400`` run directory.
        through: Last update compared (1201: the first update of ``parents_A``'s LR window, still at 3e-4).

    Returns:
        The ``cr1_compare.Result`` dict (``fields``, ``ALL``, ``first_difference``, ``differences``, ``info``); a
        missing file is a difference.
    """
    res = C.Result()
    ref_dir, new_dir = Path(ref_dir), Path(new_dir)
    if not new_dir.is_dir():
        res.check("run_dir_present", lambda: C.Diff(str(new_dir), "present", "missing"))
        return res.as_dict()
    spec = C.ModeSpec("cms2", 1, through, (), "A", (), snap="all")
    res.check("weight_exports_through", lambda: C._weights_diff(ref_dir, new_dir, spec, res.info))
    res.check("train_history:series", lambda: M1._series_diff(ref_dir, new_dir, 2, through))
    res.check("updates_csv:stream_positions_and_losses", lambda: M1._updates_csv_diff(ref_dir, new_dir, through))
    res.check("first_export_after_differs", lambda: _first_export_after_differs(ref_dir, new_dir, through))
    return res.as_dict()


def cms2_root(ref: Path, new: Path, arm: str, qs: Sequence[int], seeds: Sequence[int],
              through: int = CMS2_THROUGH) -> Dict[str, Any]:
    """C-MS2 over every (q, seed): ``ref/q<q>/seed<seed>`` against ``new/q<q>/seed<seed>/<arm>``."""
    runs = []
    for q in qs:
        for s in seeds:
            r = cms2_run(C.run_dir(ref, q, s, None), C.run_dir(new, q, s, arm), through)
            runs.append({"q": int(q), "seed": int(s), **r})
    n_ok = sum(1 for r in runs if r["ALL"])
    return {"check": "C-MS2", "ref": str(ref), "new": str(new), "arm": arm, "through_update": through,
            "n": len(runs), "n_identical": n_ok, "ALL": bool(runs) and n_ok == len(runs),
            "failing_fields": sorted({k for r in runs for k, v in r["fields"].items() if not v}),
            "first_differences": [{"q": r["q"], "seed": r["seed"], **r["first_difference"]}
                                  for r in runs if r["first_difference"]],
            "runs": runs}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI; exit code 0 iff every check of every run passed (and C-MS2, if requested)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True, help="wave root: q*/seed*/<arm>")
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--code-commit", default=None)
    p.add_argument("--qs", type=int, nargs="+", default=[50, 60])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10501, 10511)))
    p.add_argument("--out", default=None)
    p.add_argument("--cms2-ref", default=None, help="parents_A root (q*/seed*): also run C-MS2 on --cms2-arm")
    p.add_argument("--cms2-arm", default="MS_base2400")
    p.add_argument("--cms2-through", type=int, default=CMS2_THROUGH, help="last update compared (1201)")
    a = p.parse_args(argv)
    summary = check_root(Path(a.root), a.arms, a.qs, a.seeds, a.code_commit)
    cms2_ok = True
    if a.cms2_ref:
        summary["c_ms2"] = cms2_root(Path(a.cms2_ref), Path(a.root), a.cms2_arm, a.qs, a.seeds,
                                    a.cms2_through)
        cms2_ok = bool(summary["c_ms2"]["ALL"])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("n_runs", "n_ok_per_check", "all_ok", "start_share_tests")}, indent=1))
    if a.cms2_ref:
        c = summary["c_ms2"]
        print(f"C-MS2 ({c['arm']} against parents_A through update {c['through_update']}) identical "
              f"{c['n_identical']}/{c['n']} ALL={c['ALL']}")
        for fd in c["first_differences"]:
            print(f"  q={fd['q']} seed={fd['seed']} first differing field {fd['field']}: path={fd['path']} "
                  f"ref={fd['ref']} new={fd['new']} {fd.get('note', '')}")
    return 0 if summary["all_ok"] and cms2_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
