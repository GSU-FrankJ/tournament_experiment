#!/usr/bin/env python3
"""Post-launch checks of the MS-R3 pilot (prompt section 3.2): ``<root>/launch_checks.json``.

For the arms ``{t1,relu,t10}_{bb,st}_s{1,16}`` x q x seeds of a wave root. ``--actors`` (a comma list, default all three)
names the actors that were launched: the arms of the others are not planned and not checked. A planned run that is missing
is a failure, never a skip, and so is a missing reference file. The expected number of comparisons of every check is derived
from the planned arms and recorded next to the counts found (``expected``); ``all_ok`` needs them to agree.

  * the usual checks of ``tools/ms/launch_checks.py`` (status exit 0, manifest at the launch commit with a clean tree, files
    complete, process-global RNG assertions, the tail share lambda_T of every probability vector in force and the measured
    start shares per stratum within 3 binomial standard errors), exactly as ``tools/ms/r2_launch_checks.py`` runs them;
  * **scale**: the applied concentration scale equals the D3 schedule (1 up to local update 2001, linear to s at 2200, s
    afterwards; no scale is ever set in an s = 1 arm) at EVERY terminal-stage update and is 1.0 throughout stage 1; the
    recorded schedule and every exported ``conc_scale`` agree (``r2_launch_checks.scale_check``, the same function: the D3
    schedule is the MS-R2 D2 schedule);
  * **C-INIT**: for every (q, seed) the manifests of ALL arms run carry ``init_state_sha256`` and it is one and the same
    value, i.e. the initial actor and critic weights are identical across actor variants, starts and s;
  * **C-NL**: within each (actor, starts, q, seed) the s = 16 run equals the s = 1 run bit for bit through update 2001 and
    its u2025 export differs. "Equal" is the MS-R2 definition (weight exports, the per-update history of
    ``train_history.json``, ``ms_updates.csv`` with its five stream positions, the columns common to both runs of the check
    rows of ``ms_checks_stage2.csv``; wall-clock columns and run / arm labels excluded), with the array comparison made
    safe for the string entry ``actor_variant`` of a ``relu`` / ``t10`` export;
  * **C-MS5**: each ``t1`` arm equals the MS-R2 arm of the same starts and s (``t1_bb_s1`` = ``NL_bb_s1``, ...) in
    ``--ms-r2-pilot-root`` bit for bit OVER THE WHOLE RUN: every array (key set, dtype, shape, values) of every weight
    export, of ``freeze_stage{1,2}_{final,development}.npz``, ``ms_binmaps_stage2.npz`` and ``continuation_table_stage1.npz``;
    the per-update history of ``train_history.json`` (and its snapshot / weight-export logs); every column of
    ``ms_updates.csv`` (the stream positions after every update are columns of it); the columns common to both runs of
    ``ms_checks_stage1.csv`` and ``ms_checks_stage2.csv``; the values of ``gates.json``.

What the comparisons do NOT look at (recorded in the output under ``ignored``): the wall-clock columns
``update_wall_sec, verifier_sec, diag_sec`` and the label columns ``run, arm`` of the CSV files; in ``gates.json`` the
labels ``arm, run``, the git fields ``commit, clean_tree`` and the timings ``budget.*.wall_sec``,
``budget.*.process_cpu_sec``, ``continuation_tables.*.build_seconds`` and ``continuation_tables.*.meta.build_seconds``;
in ``train_history.json`` the labels ``run, arm`` and ``phase_timing``; the files ``run_config.json, manifest.json,
status.json, run.log, rule_log.json, ms_run_summary.json, drift_test.json, induced_band.json, band_sweep.npz,
state_end_stage{1,2}.pt`` (labels, git, time, host, the new MS-R3 keys; not part of the C-MS5 list).

Usage:
    python tools/ms/r3_launch_checks.py --root results/ms_r3/pilot --ms-r2-pilot-root <MS-R2 pilot root> \
        --code-commit <launch HEAD> --out results/ms_r3/pilot/launch_checks.json [--actors t1,relu,t10]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))

import launch_checks as LC1  # noqa: E402
import ms_configs as mc  # noqa: E402
import r2_launch_checks as RC2  # noqa: E402

RAMP_FIRST, RAMP_LAST, EXPORT_EVERY = RC2.RAMP_FIRST, RC2.RAMP_LAST, RC2.EXPORT_EVERY
WALL_COLS, LABEL_COLS = RC2.WALL_COLS, RC2.LABEL_COLS
KINDS = ("bb", "st")
S_LOW, S_HIGH = mc.R3_SCALES              # the two values of the concentration-scale end value s: 1 and 16
ALL_UPDATES = 10 ** 9
#: arrays of a run compared over the whole run by C-MS5, besides the weight exports (key set, dtype, shape, values)
FREEZE_FILES = ("freeze_stage1_final.npz", "freeze_stage1_development.npz",
                "freeze_stage2_final.npz", "freeze_stage2_development.npz")
BINMAPS_FILE = "ms_binmaps_stage2.npz"
TABLE_FILE = "continuation_table_stage1.npz"
#: paths of ``gates.json`` (dotted, ``*`` matches anything) that C-MS5 does not compare: labels, git state, timings
GATES_IGNORED = ("arm", "run", "commit", "clean_tree", "budget.*.wall_sec", "budget.*.process_cpu_sec",
                 "continuation_tables.*.build_seconds", "continuation_tables.*.meta.build_seconds")
HISTORY_KEYS = ("history", "snapshots", "weight_checkpoints")     # train_history.json keys compared (not run, arm, phase_timing)
NOT_COMPARED_FILES = ("run_config.json", "manifest.json", "status.json", "run.log", "rule_log.json", "ms_run_summary.json",
                      "drift_test.json", "induced_band.json", "band_sweep.npz", "state_end_stage1.pt", "state_end_stage2.pt")


# ---------------------------------------------------------------------------------------------- exact comparisons
def _arrays_diff(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Optional[str]:
    """None if two array dicts are identical (same keys, dtype, shape and values; NaN equals NaN in float arrays).

    The values of non-float arrays (including the string entry ``actor_variant``) are compared without NaN handling
    (``np.array_equal(..., equal_nan=True)`` raises on a string array).
    """
    if a.keys() != b.keys():
        return f"keys differ ({sorted(set(a) ^ set(b))[:4]})"
    for k in a:
        x, y = a[k], b[k]
        if x.dtype != y.dtype or x.shape != y.shape:
            return f"{k}: {x.dtype}{list(x.shape)} against {y.dtype}{list(y.shape)}"
        if not np.array_equal(x, y, equal_nan=x.dtype.kind in "fc"):
            return f"{k}: values differ"
    return None


def _same_json(a: Any, b: Any) -> bool:
    """Exact equality of parsed JSON values: same types, NaN equals NaN."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same_json(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(_same_json(x, y) for x, y in zip(a, b))
    if type(a) is not type(b):
        return False
    if isinstance(a, float) and math.isnan(a):
        return math.isnan(b)
    return bool(a == b)


def _flatten(o: Any, path: str = "") -> Dict[str, Any]:
    """Leaves of a parsed JSON value by dotted path (list items ``[i]``; an empty container is a leaf)."""
    if isinstance(o, dict) and o:
        out: Dict[str, Any] = {}
        for k, v in o.items():
            out.update(_flatten(v, f"{path}.{k}" if path else str(k)))
        return out
    if isinstance(o, list) and o:
        out = {}
        for i, v in enumerate(o):
            out.update(_flatten(v, f"{path}[{i}]"))
        return out
    return {path: o}


def _gates_diff(a: Path, b: Path, n: Dict[str, int]) -> Optional[str]:
    """None if the values of two ``gates.json`` agree outside :data:`GATES_IGNORED`."""
    def keep(flat: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in flat.items() if not any(fnmatch.fnmatchcase(k, pat) for pat in GATES_IGNORED)}
    fa, fb = keep(_flatten(RC2._json(a))), keep(_flatten(RC2._json(b)))
    if fa.keys() != fb.keys():
        return f"gate paths differ ({sorted(set(fa) ^ set(fb))[:3]})"
    for k in fa:
        if not _same_json(fa[k], fb[k]):
            return f"{k}: {fa[k]!r} against {fb[k]!r}"
    n["gate_values"] += len(fa)
    return None


def _csv_diff(a: Path, b: Path, same_columns: bool, n: Dict[str, int], count: str) -> Optional[str]:
    """None if two CSV files agree as text on every row.

    ``same_columns`` demands the same header (``ms_updates.csv``: every column); otherwise the columns common to both are
    compared (the check tables). Wall-clock and label columns are never compared.
    """
    ra, rb = RC2._csv(a), RC2._csv(b)
    if not ra or len(ra) != len(rb):
        return f"{len(ra)} rows against {len(rb)}"
    if same_columns and list(ra[0]) != list(rb[0]):
        return f"columns differ ({sorted(set(ra[0]) ^ set(rb[0]))[:4] or 'order'})"
    cols = [c for c in ra[0] if c in rb[0] and c not in WALL_COLS and c not in LABEL_COLS]
    for i, (x, y) in enumerate(zip(ra, rb)):
        bad = [c for c in cols if x[c] != y[c]]
        if bad:
            return f"row {i + 1} (update {x.get('update')}): column {bad[0]} ({x[bad[0]]} against {y[bad[0]]})"
    n[count] += len(ra)
    return None


def _field_runner(res: Dict[str, Any], msgs: List[str]) -> Callable[[str, Callable[[], Optional[str]]], None]:
    """``field(name, fn)``: ``fn()`` returns None if the part agrees, else a message; an exception (a missing file) fails."""
    def field(name: str, fn: Callable[[], Optional[str]]) -> None:
        try:
            m = fn()
        except Exception as exc:  # noqa: BLE001 - a missing file is a failure
            m = f"{type(exc).__name__}: {exc}"
        res[name] = m is None
        if m is not None:
            msgs.append(f"{name}: {m}")
    return field


# ---------------------------------------------------------------------------------------------- C-NL
def prefix_identity(ref: Path, new: Path, upto: int, export_every: int = EXPORT_EVERY) -> Dict[str, Any]:
    """Bit-for-bit equality of two runs through local update ``upto`` of the terminal stage (the C-NL definition of MS-R2).

    Returns:
        ``{field: bool, ..., "ALL": bool, "first_difference": str | None, "first_export_after": int,
        "first_export_after_differs": bool}``; the export that follows ``upto`` is the u2025 export of the real pilot.
    """
    res: Dict[str, Any] = {}
    msgs: List[str] = []
    field = _field_runner(res, msgs)

    def exports() -> Optional[str]:
        ea, eb = RC2._exports(ref, upto), RC2._exports(new, upto)
        if set(ea) != set(eb) or not ea:
            return f"export sets differ ({len(ea)} against {len(eb)})"
        for u in sorted(ea):
            m = _arrays_diff(RC2._npz(ea[u]), RC2._npz(eb[u]))
            if m:
                return f"u{u:05d}.npz differs ({m})"
        return None

    def history() -> Optional[str]:
        ha = [h for h in RC2._json(ref / "train_history.json")["history"] if h["update"] <= upto]
        hb = [h for h in RC2._json(new / "train_history.json")["history"] if h["update"] <= upto]
        if len(ha) != upto or len(hb) != upto:
            return f"{len(ha)} / {len(hb)} entries, {upto} expected"
        keys = sorted((set(ha[0]) & set(hb[0])) - {"stage", "local"})
        for x, y in zip(ha, hb):
            if not all(_same_json(x[k], y[k]) for k in keys):
                return f"update {x['update']}"
        return None

    field("weight_exports", exports)
    field("train_history", history)
    field("updates_csv_and_stream_positions",
          lambda: RC2._rows_equal(RC2._csv(ref / "ms_updates.csv"), RC2._csv(new / "ms_updates.csv"), upto, ()))
    field("check_rows",
          lambda: RC2._rows_equal(RC2._csv(ref / "ms_checks_stage2.csv"), RC2._csv(new / "ms_checks_stage2.csv"), upto, ()))
    res["ALL"] = bool(all(res.values()))
    res["first_difference"] = msgs[0] if msgs else None
    u = (upto // export_every + 1) * export_every
    res["first_export_after"] = u
    try:
        res["first_export_after_differs"] = _arrays_diff(RC2._npz(ref / "weights" / f"u{u:05d}.npz"),
                                                         RC2._npz(new / "weights" / f"u{u:05d}.npz")) is not None
    except Exception:  # noqa: BLE001 - a missing export is not a difference
        res["first_export_after_differs"] = False
    return res


# ---------------------------------------------------------------------------------------------- C-MS5
def whole_run_identity(ref: Path, new: Path) -> Dict[str, Any]:
    """Bit-for-bit equality of two complete runs: C-MS5 (see the module docstring for what is and is not compared).

    Returns:
        ``{part: bool, ..., "ALL": bool, "first_difference": str | None, "compared": {counts}}`` with the parts
        ``weight_exports, train_history, updates_csv, checks_stage1, checks_stage2, freeze_arrays, binmaps_stage2,
        continuation_table_stage1, gates``. A missing file fails its part.
    """
    res: Dict[str, Any] = {}
    msgs: List[str] = []
    n = {"exports": 0, "arrays": 0, "update_rows": 0, "check_rows": 0, "gate_values": 0}
    field = _field_runner(res, msgs)

    def exports() -> Optional[str]:
        ea, eb = RC2._exports(ref, ALL_UPDATES), RC2._exports(new, ALL_UPDATES)
        if set(ea) != set(eb) or not ea:
            return f"export sets differ ({len(ea)} against {len(eb)})"
        for u in sorted(ea):
            za, zb = RC2._npz(ea[u]), RC2._npz(eb[u])
            m = _arrays_diff(za, zb)
            if m:
                return f"u{u:05d}.npz differs ({m})"
            n["exports"] += 1
            n["arrays"] += len(za)
        return None

    def history() -> Optional[str]:
        ta, tb = RC2._json(ref / "train_history.json"), RC2._json(new / "train_history.json")
        for key in HISTORY_KEYS:
            la, lb = ta[key], tb[key]
            if len(la) != len(lb) or not la:
                return f"{key}: {len(la)} entries against {len(lb)}"
            for i, (x, y) in enumerate(zip(la, lb)):
                if not _same_json(x, y):
                    return f"{key}[{i}]" + (f" (update {x['update']})" if isinstance(x, dict) and "update" in x else "")
        return None

    def npz_files(names: Sequence[str]) -> Callable[[], Optional[str]]:
        def fn() -> Optional[str]:
            for f in names:
                za, zb = RC2._npz(ref / f), RC2._npz(new / f)
                m = _arrays_diff(za, zb)
                if m:
                    return f"{f}: {m}"
                n["arrays"] += len(za)
            return None
        return fn

    field("weight_exports", exports)
    field("train_history", history)
    field("updates_csv", lambda: _csv_diff(ref / "ms_updates.csv", new / "ms_updates.csv", True, n, "update_rows"))
    field("checks_stage1", lambda: _csv_diff(ref / "ms_checks_stage1.csv", new / "ms_checks_stage1.csv", False, n, "check_rows"))
    field("checks_stage2", lambda: _csv_diff(ref / "ms_checks_stage2.csv", new / "ms_checks_stage2.csv", False, n, "check_rows"))
    field("freeze_arrays", npz_files(FREEZE_FILES))
    field("binmaps_stage2", npz_files((BINMAPS_FILE,)))
    field("continuation_table_stage1", npz_files((TABLE_FILE,)))
    field("gates", lambda: _gates_diff(ref / "gates.json", new / "gates.json", n))
    res["ALL"] = bool(all(res.values()))
    res["first_difference"] = msgs[0] if msgs else None
    res["compared"] = n
    return res


# ---------------------------------------------------------------------------------------------- C-INIT
def init_check(root: Path, arms: Sequence[str], q: int, seed: int) -> Dict[str, Any]:
    """C-INIT of one (q, seed): every planned arm records ``init_state_sha256`` and all the values are one value."""
    digests: Dict[str, str] = {}
    errors: List[str] = []
    for arm in arms:
        try:
            dg = RC2._json(root / f"q{q}" / f"seed{seed}" / arm / "manifest.json").get("init_state_sha256")
            if isinstance(dg, str) and dg:
                digests[arm] = dg
            else:
                errors.append(f"{arm}: no init_state_sha256 in manifest.json")
        except Exception as exc:  # noqa: BLE001 - a missing manifest is a failure
            errors.append(f"{arm}: {type(exc).__name__}: {exc}")
    distinct = sorted(set(digests.values()))
    ok = bool(arms) and not errors and len(digests) == len(arms) and len(distinct) == 1
    return {"q": q, "seed": seed, "n_arms": len(arms), "n_with_digest": len(digests), "n_distinct": len(distinct),
            "digest": distinct[0] if len(distinct) == 1 else None, "digests": digests, "errors": errors, "pass": ok}


# ---------------------------------------------------------------------------------------------- the checks
def run_dir(root: Path, q: int, seed: int, arm: str) -> Path:
    """``<root>/q<q>/seed<seed>/<arm>``."""
    return root / f"q{q}" / f"seed{seed}" / arm


def check_root(root: Path, ms_r2_root: Path, qs: Sequence[int], seeds: Sequence[int], code_commit: Optional[str],
               actors: Optional[Sequence[str]] = None, ramp: Tuple[int, int] = (RAMP_FIRST, RAMP_LAST),
               export_every: int = EXPORT_EVERY) -> Dict[str, Any]:
    """All checks of a wave root for the arms of ``actors`` (default: all three actors)."""
    arms = mc.r3_arms_of(actors)
    acts = tuple(a for a in mc.R3_ACTORS if a in (mc.R3_ACTORS if actors is None else actors))
    base = LC1.check_root(root, list(arms), qs, seeds, code_commit)
    upto = ramp[0]
    scales, c_init, c_nl, c_ms5 = [], [], [], []
    for q in qs:
        for s in seeds:
            for arm in arms:
                d = run_dir(root, q, s, arm)
                r = RC2.scale_check(d, mc.r2_equivalent(arm), ramp) if d.is_dir() else {"ok": False, "errors": ["missing run"]}
                scales.append({"arm": arm, "q": q, "seed": s, **r})
            c_init.append(init_check(root, arms, q, s))
            for actor in acts:
                for kind in KINDS:
                    r = prefix_identity(run_dir(root, q, s, f"{actor}_{kind}_s{S_LOW}"),
                                        run_dir(root, q, s, f"{actor}_{kind}_s{S_HIGH}"), upto, export_every)
                    c_nl.append({"actor": actor, "sampler": kind, "s": S_HIGH, "q": q, "seed": s, **r,
                                 "pass": bool(r["ALL"] and r["first_export_after_differs"])})
            if "t1" in acts:
                for arm, ref_arm in mc.R3_MS_R2_REFERENCE.items():
                    r = whole_run_identity(run_dir(ms_r2_root, q, s, ref_arm), run_dir(root, q, s, arm))
                    c_ms5.append({"arm": arm, "reference": ref_arm, "q": q, "seed": s, **r, "pass": bool(r["ALL"])})
    npass = lambda xs: sum(1 for x in xs if x.get("pass") or x.get("ok"))  # noqa: E731
    summary = {"scale_ok": npass(scales), "scale_n": len(scales), "C_INIT_pass": npass(c_init), "C_INIT_n": len(c_init),
               "C_NL_pass": npass(c_nl), "C_NL_n": len(c_nl), "C_MS5_pass": npass(c_ms5), "C_MS5_n": len(c_ms5)}
    nrun = len(qs) * len(seeds)
    expected = {"scale_n": len(arms) * nrun, "C_INIT_n": nrun, "C_NL_n": len(acts) * len(KINDS) * nrun,
                "C_MS5_n": (len(mc.R3_MS_R2_REFERENCE) if "t1" in acts else 0) * nrun}
    all_ok = bool(base["all_ok"] and summary["scale_ok"] == summary["scale_n"] == expected["scale_n"]
                  and summary["C_INIT_pass"] == summary["C_INIT_n"] == expected["C_INIT_n"]
                  and summary["C_NL_pass"] == summary["C_NL_n"] == expected["C_NL_n"]
                  and summary["C_MS5_pass"] == summary["C_MS5_n"] == expected["C_MS5_n"])
    ignored = {"csv_wall_clock_columns": sorted(WALL_COLS), "csv_label_columns": sorted(LABEL_COLS),
               "gates_json_paths": list(GATES_IGNORED), "train_history_keys_compared_by_C_MS5": list(HISTORY_KEYS),
               "files_not_compared_by_C_MS5": list(NOT_COMPARED_FILES)}
    return {"tool": "tools/ms/r3_launch_checks.py", "root": str(root), "ms_r2_pilot_root": str(ms_r2_root),
            "code_commit": code_commit, "actors": list(acts), "arms": list(arms), "ramp": list(ramp),
            "export_every": export_every, "base": base, "summary": summary, "expected": expected, "all_ok": all_ok,
            "ignored": ignored, "scale": scales, "C_INIT": c_init, "C_NL": c_nl, "C_MS5": c_ms5}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI; exit code 0 iff every check passed."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True, help="the MS-R3 pilot root: q*/seed*/<arm>")
    p.add_argument("--ms-r2-pilot-root", required=True, help="the MS-R2 pilot root (the NL_* arms: reference of C-MS5)")
    p.add_argument("--code-commit", required=True, help="the launch commit every manifest must carry")
    p.add_argument("--out", required=True, help="the launch_checks.json to write")
    p.add_argument("--actors", default=None, help="comma list, a subset of t1,relu,t10 (default all three): the actors launched")
    p.add_argument("--qs", type=int, nargs="+", default=[50, 60])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10501, 10511)))
    p.add_argument("--ramp-first", type=int, default=RAMP_FIRST, help="tests only")
    p.add_argument("--ramp-last", type=int, default=RAMP_LAST, help="tests only")
    p.add_argument("--export-every", type=int, default=EXPORT_EVERY, help="tests only")
    a = p.parse_args(argv)
    try:
        actors = mc.parse_actors(a.actors) if a.actors is not None else None
    except ValueError as exc:
        p.error(str(exc))
    res = check_root(Path(a.root), Path(a.ms_r2_pilot_root), a.qs, a.seeds, a.code_commit, actors,
                     (a.ramp_first, a.ramp_last), a.export_every)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"actors": res["actors"], "base_n_ok_per_check": res["base"]["n_ok_per_check"],
                      "base_start_share_tests": res["base"]["start_share_tests"], "summary": res["summary"],
                      "expected": res["expected"], "all_ok": res["all_ok"]}, indent=1))
    for check in ("scale", "C_INIT", "C_NL", "C_MS5"):             # the first failures of each check (all of them are in --out)
        for x in [x for x in res[check] if not (x.get("pass") or x.get("ok"))][:3]:
            who = x.get("arm") or "_".join(str(x[k]) for k in ("actor", "sampler") if k in x)
            why = x.get("first_difference") or x.get("errors") or ("export after the ramp start does not differ"
                                                                    if x.get("ALL") else "differs")
            print(f"FAILED {check} {who} q={x['q']} seed={x['seed']}: {why}")
    return 0 if res["all_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
