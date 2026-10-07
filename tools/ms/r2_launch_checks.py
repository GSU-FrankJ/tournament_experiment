#!/usr/bin/env python3
"""Post-launch checks of the MS-R2 pilot (prompt section 3.2): ``<root>/launch_checks.json``.

For the six arms ``NL_{bb,st}_s{1,4,16}`` x q x seeds of a wave root:

  * the usual checks of ``tools/ms/launch_checks.py`` (status exit 0, manifest at the launch commit with a clean
    tree, files complete, process-global RNG assertions, the tail share lambda_T of every probability vector in force
    and the measured start shares per stratum within 3 binomial standard errors);
  * **scale**: the applied concentration scale (``conc_scale`` of ``ms_updates.csv``) equals the D2 schedule
    (1 up to local update 2001, linear to s at 2200, s afterwards; 1.0 in every run of s = 1) at EVERY terminal-stage
    update and is 1.0 throughout stage 1; the schedule in ``run_config.json`` / ``manifest.json`` equals D2; every
    exported ``conc_scale`` equals the schedule at that update;
  * **C-NL**: within each sampler and (q, seed) the s = 4 and s = 16 runs equal the s = 1 run bit for bit through
    update 2001 and their u2025 export differs (2 samplers x 2 values of s x 20 = 80 comparisons);
  * **C-MS3**: ``NL_bb_s1`` equals MS-R1's ``MS_base2400`` through update 2001 and its u2025 export differs (20);
  * **C-MS4**: ``NL_st_s1`` equals MS-R1's ``MS_s35a5`` through update 2001 where that run's terminal stage had no
    polishing block, otherwise through the last update before its first polishing block (20); the first export after
    that update is reported to differ (informational). When the reference polishes, its check row at the block end
    before the first polishing block carries the polishing sampler's digest in ``p_digest_next``; that one column is
    not compared (``p_digest`` of every update is).

"Equal" = the weight exports, the per-update series of ``ms_updates.csv`` (which holds the five stream positions after
every update), the per-update history of ``train_history.json`` and the columns common to both runs of the check rows
of ``ms_checks_stage2.csv`` -- wall-clock columns and the run / arm / block labels excluded. A missing file is a
failure, never a skip.

Usage:
    python tools/ms/r2_launch_checks.py --root results/ms_r2/pilot --ms-r1-pilot-root <MS-R1 pilot root> \
        --code-commit <launch HEAD> --out results/ms_r2/pilot/launch_checks.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))

import launch_checks as LC1  # noqa: E402

ARMS = tuple(f"NL_{k}_s{s}" for k in ("bb", "st") for s in (1, 4, 16))
RAMP_FIRST, RAMP_LAST, EXPORT_EVERY = 2001, 2200, 25
WALL_COLS = {"update_wall_sec", "verifier_sec", "diag_sec"}
LABEL_COLS = {"run", "arm"}
SCHEDULE_KEYS = {"stage": 2, "local_first": RAMP_FIRST, "local_last": RAMP_LAST, "scale_first": 1.0}


def arm_scale(arm: str) -> float:
    """The end value s of an ``NL_{kind}_s{s}`` arm."""
    m = re.fullmatch(r"NL_(bb|st)_s(\d+)", arm)
    if not m:
        raise ValueError(f"not an MS-R2 arm: {arm!r}")
    return float(m.group(2))


def schedule_value(local: int, s: float, first: int = RAMP_FIRST, last: int = RAMP_LAST) -> float:
    """The D2 scale before local update ``local``: 1 up to ``first``, linear to ``s`` at ``last``, ``s`` afterwards."""
    if s == 1.0 or local <= first:
        return 1.0
    if local >= last:
        return float(s)
    return 1.0 + (float(s) - 1.0) * (local - first) / (last - first)


def _csv(path: Path) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _same_arrays(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)


def _exports(d: Path, upto: int) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for p in sorted((d / "weights").glob("u*.npz")):
        m = re.fullmatch(r"u(\d+)\.npz", p.name)
        if m and int(m.group(1)) <= upto:
            out[int(m.group(1))] = p
    return out


def _rows_equal(ra: List[Dict[str, str]], rb: List[Dict[str, str]], upto: int, skip: Sequence[str]) -> Optional[str]:
    """None if the rows with ``update <= upto`` agree on the common columns (minus wall clock, labels, ``skip``)."""
    ra = [r for r in ra if int(r["update"]) <= upto]
    rb = [r for r in rb if int(r["update"]) <= upto]
    if not ra or len(ra) != len(rb):
        return f"{len(ra)} rows against {len(rb)}"
    cols = (set(ra[0]) & set(rb[0])) - WALL_COLS - LABEL_COLS - set(skip)
    for x, y in zip(ra, rb):
        bad = [c for c in sorted(cols) if x[c] != y[c]]
        if bad:
            return f"update {x['update']}: column {bad[0]} ({x[bad[0]]} against {y[bad[0]]})"
    return None


def prefix_identity(ref: Path, new: Path, upto: int, skip: Sequence[str] = (), export_every: int = EXPORT_EVERY) -> Dict[str, Any]:
    """Bit-for-bit equality of two runs through local update ``upto`` of the terminal stage (see the module docstring).

    Returns:
        ``{field: bool, ..., "ALL": bool, "first_difference": str | None, "first_export_after_differs": bool}``.
    """
    res: Dict[str, Any] = {}
    msgs: List[str] = []

    def field(name: str, fn: Any) -> None:
        try:
            m = fn()
        except Exception as exc:  # noqa: BLE001 - a missing file is a failure
            m = f"{type(exc).__name__}: {exc}"
        res[name] = m is None
        if m is not None:
            msgs.append(f"{name}: {m}")

    def exports() -> Optional[str]:
        ea, eb = _exports(ref, upto), _exports(new, upto)
        if set(ea) != set(eb) or not ea:
            return f"export sets differ ({len(ea)} against {len(eb)})"
        for u in sorted(ea):
            if not _same_arrays(_npz(ea[u]), _npz(eb[u])):
                return f"u{u:05d}.npz differs"
        return None

    def history() -> Optional[str]:
        ha = [h for h in _json(ref / "train_history.json")["history"] if h["update"] <= upto]
        hb = [h for h in _json(new / "train_history.json")["history"] if h["update"] <= upto]
        if len(ha) != upto or len(hb) != upto:
            return f"{len(ha)} / {len(hb)} entries, {upto} expected"
        keys = sorted((set(ha[0]) & set(hb[0])) - {"stage", "local"})
        for x, y in zip(ha, hb):
            if any(x[k] != y[k] for k in keys):
                return f"update {x['update']}"
        return None

    field("weight_exports", exports)
    field("train_history", history)
    field("updates_csv_and_stream_positions", lambda: _rows_equal(_csv(ref / "ms_updates.csv"), _csv(new / "ms_updates.csv"), upto, skip))
    field("check_rows", lambda: _rows_equal(_csv(ref / "ms_checks_stage2.csv"), _csv(new / "ms_checks_stage2.csv"), upto, skip))
    res["ALL"] = bool(all(res.values()))
    res["first_difference"] = msgs[0] if msgs else None
    u = (upto // export_every + 1) * export_every
    try:
        res["first_export_after"] = u
        res["first_export_after_differs"] = not _same_arrays(_npz(ref / "weights" / f"u{u:05d}.npz"),
                                                            _npz(new / "weights" / f"u{u:05d}.npz"))
    except Exception:  # noqa: BLE001
        res["first_export_after_differs"] = False
    return res


def scale_check(d: Path, arm: str, ramp: Tuple[int, int] = (RAMP_FIRST, RAMP_LAST)) -> Dict[str, Any]:
    """The applied scale against the D2 schedule at every update, the recorded schedule and the exported scales."""
    s = arm_scale(arm)
    out: Dict[str, Any] = {"s": s}
    errs: List[str] = []
    try:
        rows = _csv(d / "ms_updates.csv")
        bad = 0
        n2 = n1 = 0
        for r in rows:
            got = float(r["conc_scale"])
            if int(r["stage"]) == 2:
                n2 += 1
                want = schedule_value(int(r["local"]), s, *ramp)
            else:
                n1 += 1
                want = 1.0
            if not math.isclose(got, want, rel_tol=1e-12, abs_tol=0.0):
                bad += 1
        out.update(n_terminal_updates=n2, n_stage1_updates=n1, n_mismatches=bad)
        if bad or not n2 or not n1:
            errs.append(f"{bad} update(s) with a scale different from the schedule")
        cfg = _json(d / "run_config.json")
        man = _json(d / "manifest.json")
        want_sched = None if s == 1.0 else {**SCHEDULE_KEYS, "local_first": ramp[0], "local_last": ramp[1], "scale_last": s}
        if cfg.get("conc_scale_schedule") != want_sched or man.get("conc_scale_schedule") != want_sched:
            errs.append("recorded schedule differs from D2")
        if cfg.get("noise_report") is not True:
            errs.append("noise_report is not true")
        exp_bad = 0
        for p in sorted((d / "weights").glob("u*.npz")):
            u = int(p.stem[1:])
            z = _npz(p)
            got = float(z["conc_scale"]) if "conc_scale" in z else 1.0
            stage_of = 2 if u <= int(_json(d / "rule_log.json")["stages"]["2"]["exit_update"]) else 1
            want = schedule_value(u, s, *ramp) if stage_of == 2 else 1.0
            if not math.isclose(got, want, rel_tol=1e-12):
                exp_bad += 1
        out["n_export_mismatches"] = exp_bad
        if exp_bad:
            errs.append(f"{exp_bad} export(s) with a scale different from the schedule")
    except Exception as exc:  # noqa: BLE001
        errs.append(f"{type(exc).__name__}: {exc}")
    out["ok"] = not errs
    out["errors"] = errs
    return out


def first_polish_local(d: Path) -> Optional[int]:
    """First local update of the first polishing block of the terminal stage of an MS-R1 run (None if there is none)."""
    blocks = _json(d / "rule_log.json")["stages"]["2"]["blocks"]
    firsts = [int(b["first_local"]) for b in blocks if b.get("type") == "polish"]
    return min(firsts) if firsts else None


def check_root(root: Path, ms_r1_root: Path, qs: Sequence[int], seeds: Sequence[int], code_commit: Optional[str],
               ramp: Tuple[int, int] = (RAMP_FIRST, RAMP_LAST), export_every: int = EXPORT_EVERY) -> Dict[str, Any]:
    """All checks of a wave root (the arms of :data:`ARMS`)."""
    base = LC1.check_root(root, list(ARMS), qs, seeds, code_commit)
    upto = ramp[0]
    scales, c_nl, c_ms3, c_ms4 = [], [], [], []
    for q in qs:
        for s in seeds:
            for arm in ARMS:
                d = root / f"q{q}" / f"seed{s}" / arm
                r = scale_check(d, arm, ramp) if d.is_dir() else {"ok": False, "errors": ["missing run"]}
                scales.append({"arm": arm, "q": q, "seed": s, **r})
            for kind in ("bb", "st"):
                ref = root / f"q{q}" / f"seed{s}" / f"NL_{kind}_s1"
                for sc in (4, 16):
                    new = root / f"q{q}" / f"seed{s}" / f"NL_{kind}_s{sc}"
                    r = prefix_identity(ref, new, upto, export_every=export_every)
                    c_nl.append({"sampler": kind, "s": sc, "q": q, "seed": s, **r,
                                 "pass": bool(r["ALL"] and r["first_export_after_differs"])})
            nl_bb = root / f"q{q}" / f"seed{s}" / "NL_bb_s1"
            r = prefix_identity(ms_r1_root / f"q{q}" / f"seed{s}" / "MS_base2400", nl_bb, upto, export_every=export_every)
            c_ms3.append({"q": q, "seed": s, **r, "pass": bool(r["ALL"] and r["first_export_after_differs"])})
            ms_s35a5 = ms_r1_root / f"q{q}" / f"seed{s}" / "MS_s35a5"
            nl_st = root / f"q{q}" / f"seed{s}" / "NL_st_s1"
            try:
                fp = first_polish_local(ms_s35a5)
                bound = upto if fp is None else min(upto, fp - 1)
                # the check row at the block end before a polishing block writes the digest of the POLISHING setting into
                # p_digest_next (the next block's sampler); NL_st never polishes, so that one column cannot agree there
                # (the probabilities used by every update are still compared through p_digest of ms_updates.csv)
                skip = ("block_id", "block_type", "mode") + (("p_digest_next",) if fp is not None else ())
                r = prefix_identity(ms_s35a5, nl_st, bound, skip=skip, export_every=export_every)
                c_ms4.append({"q": q, "seed": s, "first_polish_local": fp, "identical_through_update": bound, **r,
                              "pass": bool(r["ALL"])})
            except Exception as exc:  # noqa: BLE001
                c_ms4.append({"q": q, "seed": s, "ALL": False, "pass": False, "first_difference": f"{type(exc).__name__}: {exc}"})
    n = lambda xs: sum(1 for x in xs if x.get("pass") or x.get("ok"))  # noqa: E731
    summary = {"scale_ok": n(scales), "scale_n": len(scales), "C_NL_pass": n(c_nl), "C_NL_n": len(c_nl),
               "C_MS3_pass": n(c_ms3), "C_MS3_n": len(c_ms3), "C_MS4_pass": n(c_ms4), "C_MS4_n": len(c_ms4)}
    all_ok = bool(base["all_ok"] and summary["scale_ok"] == summary["scale_n"] and summary["C_NL_pass"] == summary["C_NL_n"]
                  and summary["C_MS3_pass"] == summary["C_MS3_n"] and summary["C_MS4_pass"] == summary["C_MS4_n"])
    return {"tool": "tools/ms/r2_launch_checks.py", "root": str(root), "ms_r1_pilot_root": str(ms_r1_root),
            "code_commit": code_commit, "ramp": list(ramp), "export_every": export_every, "base": base, "summary": summary,
            "all_ok": all_ok, "scale": scales, "C_NL": c_nl, "C_MS3": c_ms3, "C_MS4": c_ms4}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI; exit code 0 iff every check passed."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True, help="the MS-R2 pilot root: q*/seed*/<arm>")
    p.add_argument("--ms-r1-pilot-root", required=True, help="the MS-R1 pilot root (MS_base2400, MS_s35a5)")
    p.add_argument("--code-commit", default=None)
    p.add_argument("--qs", type=int, nargs="+", default=[50, 60])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10501, 10511)))
    p.add_argument("--ramp-first", type=int, default=RAMP_FIRST, help="tests only")
    p.add_argument("--ramp-last", type=int, default=RAMP_LAST, help="tests only")
    p.add_argument("--export-every", type=int, default=EXPORT_EVERY, help="tests only")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    res = check_root(Path(a.root), Path(a.ms_r1_pilot_root), a.qs, a.seeds, a.code_commit, (a.ramp_first, a.ramp_last), a.export_every)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    print(json.dumps({"base_n_ok_per_check": res["base"]["n_ok_per_check"], "base_start_share_tests": res["base"]["start_share_tests"],
                      "summary": res["summary"], "all_ok": res["all_ok"]}, indent=1))
    return 0 if res["all_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
