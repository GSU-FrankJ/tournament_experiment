#!/usr/bin/env python3
"""Post-launch checks of MS-R4 (prompt sections 2.4 and 3.2).

``--mode pilot`` (wave ``r4``, ``<root>/launch_checks.json``): for the seven arms x q x seeds of the pilot root

  * the usual checks of ``tools/ms/launch_checks.py`` (status exit 0, manifest at the launch commit with a clean tree, files
    complete, process-global RNG assertions, the tail share lambda_T of every probability vector in force, the measured start
    shares per stratum within 3 binomial standard errors -- statistical flags are reported against their chance expectation);
  * **strata masses** (deterministic): at every terminal-stage update the probability mass of the tail stratum equals
    lambda_T in every arm, and in ``fb_a0``, ``fb_r`` and ``fb_rg`` the masses of the near-tie and middle strata equal
    lambda_P and lambda_M (to 1e-12); for ``fb_rx`` and the E1 arms the realised near and middle shares are reported;
  * **scale**: the applied concentration scale equals the schedule of the arm at every update (the E1 arms: 1 up to local
    update 2001, linear to 16 at 2200, 16 afterwards; the E2 arms and stage 1: 1.0 throughout; ``r2_launch_checks.scale_check``);
  * **C-INIT**: for every (q, seed) the manifests of ``t1_st_s16``, ``relu_st_s16`` and ``lrsm_st_s16`` carry one and the same
    ``init_state_sha256`` (reported as well: all seven arms);
  * **C-FB25**: ``fb_r``, ``fb_rg`` and ``fb_rx`` equal ``fb_a0`` bit for bit through local update 25 (the u25 export, the
    per-update series with the stream positions of updates 1-25 and the check row at local 25) except for the columns that
    record the configured alpha and the next-update digest, and their u50 exports differ;
  * **C-X2**: ``fb_rx`` equals ``t1_st_s16`` bit for bit through local update 1201 (exports u25 ... u1200, the per-update series
    and stream positions of updates 1-1201, the check rows' common columns through local 1200) and their u1225 exports differ;
  * **sampling weights**: for every E2 arm the probabilities in force after each check (``probs_next`` where recorded, else
    reconstructed from ``probs_first_update`` / ``probs_table``) are consistent with ``p_digest_next`` at every check of the
    stage except the last one, where the runner writes an empty digest and ``probs_next`` is NaN.

``--mode cms6`` (wave ``r4_cms6``): each of the 40 control runs (``t1_st_s16``, ``relu_st_s16`` x q x seeds 10501-10510) equals
MS-R3's run of the same name bit for bit over the whole run (``r3_launch_checks.whole_run_identity``: every array of every
weight export, the freeze arrays, the bin maps and the continuation table, the per-update history and series with the
stream positions after every update, the common columns of the check rows, the values of ``gates.json``).

Wall-clock columns (``update_wall_sec``, ``verifier_sec``, ``diag_sec``) and the run / arm labels are excluded from every
comparison. A missing run or file is a failure, never a skip; the expected number of comparisons of each check is derived from
the plan and recorded next to the number found (``expected``); ``all_ok`` needs them to agree.

Usage:
    python tools/ms/r4_launch_checks.py --mode pilot --root results/ms_r4/pilot --code-commit <sha> \
        --out results/ms_r4/pilot/launch_checks.json
    python tools/ms/r4_launch_checks.py --mode cms6 --root results/ms_r4/cms6 --ms-r3-pilot-root <MS-R3 pilot root> \
        --code-commit <sha> --out results/ms_r4/cms6_checks.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))

import launch_checks as LC1  # noqa: E402
import r2_launch_checks as RC2  # noqa: E402
import r3_launch_checks as R3C  # noqa: E402
import r4_configs as rc  # noqa: E402

EXPORT_EVERY = RC2.EXPORT_EVERY
WALL_COLS, LABEL_COLS = RC2.WALL_COLS, RC2.LABEL_COLS
RAMP = (RC2.RAMP_FIRST, RC2.RAMP_LAST)
FB25_ARMS = ("fb_r", "fb_rg", "fb_rx")
FB25_REF = "fb_a0"
FB25_UPTO = 25
FB25_SKIP = ("alpha", "p_digest_next")         # the columns that record the configured alpha and the next-update digest
X2_ARM, X2_REF, X2_UPTO = "fb_rx", "t1_st_s16", 1201
MASS_TOL = 1e-12
WITHIN_ARMS = ("fb_a0", "fb_r", "fb_rg")        # the arms whose three stratum masses are fixed at every update
E1_SCALE_REF = "NL_st_s16"                      # r2_launch_checks.scale_check names the schedule by the MS-R2 arm
E2_SCALE_REF = "NL_st_s1"


def run_dir(root: Path, q: int, seed: int, arm: str) -> Path:
    """``<root>/q<q>/seed<seed>/<arm>``."""
    return Path(root) / f"q{q}" / f"seed{seed}" / arm


# ---------------------------------------------------------------------------------------------- prefix identity (C-FB25, C-X2)
def prefix_identity(ref: Path, new: Path, upto: int, skip: Sequence[str] = (), export_every: int = EXPORT_EVERY
                    ) -> Dict[str, Any]:
    """Bit-for-bit equality of two runs through local update ``upto`` of the terminal stage, except for the columns in ``skip``.

    The same definition as ``r3_launch_checks.prefix_identity`` (weight exports, the per-update history of
    ``train_history.json``, ``ms_updates.csv`` with its five stream positions, the check rows of ``ms_checks_stage2.csv``;
    wall-clock columns and labels excluded) with an explicit list of columns that are not compared.

    Returns:
        ``{field: bool, ..., "ALL": bool, "first_difference": str | None, "first_export_after": int,
        "first_export_after_differs": bool}``.
    """
    res: Dict[str, Any] = {}
    msgs: List[str] = []
    field = R3C._field_runner(res, msgs)

    def exports() -> Optional[str]:
        ea, eb = RC2._exports(ref, upto), RC2._exports(new, upto)
        if set(ea) != set(eb) or not ea:
            return f"export sets differ ({len(ea)} against {len(eb)})"
        for u in sorted(ea):
            m = R3C._arrays_diff(RC2._npz(ea[u]), RC2._npz(eb[u]))
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
            if not all(R3C._same_json(x[k], y[k]) for k in keys):
                return f"update {x['update']}"
        return None

    field("weight_exports", exports)
    field("train_history", history)
    field("updates_csv_and_stream_positions",
          lambda: RC2._rows_equal(RC2._csv(ref / "ms_updates.csv"), RC2._csv(new / "ms_updates.csv"), upto, skip))
    field("check_rows",
          lambda: RC2._rows_equal(RC2._csv(ref / "ms_checks_stage2.csv"), RC2._csv(new / "ms_checks_stage2.csv"), upto, skip))
    res["ALL"] = bool(all(res.values()))
    res["first_difference"] = msgs[0] if msgs else None
    u = (upto // export_every + 1) * export_every
    res["first_export_after"] = u
    try:
        res["first_export_after_differs"] = R3C._weights_differ(RC2._npz(ref / "weights" / f"u{u:05d}.npz"),
                                                                RC2._npz(new / "weights" / f"u{u:05d}.npz"))
    except Exception:  # noqa: BLE001 - a missing export is not a difference
        res["first_export_after_differs"] = False
    return res


# ---------------------------------------------------------------------------------------------- strata masses
def _stage2_updates(d: Path) -> List[Dict[str, str]]:
    return [r for r in RC2._csv(d / "ms_updates.csv") if int(r["stage"]) == 2]


def strata_masses(d: Path) -> Dict[str, Any]:
    """Per terminal-stage update the probability mass of the tail, near-tie and middle strata (from the recorded
    probabilities in force), against lambda_T, lambda_P, lambda_M; and the realised start shares of the run.

    Returns:
        ``{"lambda": [T, P, M], "max_dev": [..3..], "realised_share": [..3..], "ok_tail": bool, "ok_within": bool, ...}``;
        ``ok_within`` is the exactness of all three masses (to 1e-12).
    """
    from envs.curriculum_env import StartSampler
    cfg = RC2._json(d / "run_config.json")
    spec = LC1._spec(cfg)
    sw = cfg["start_weights"]
    sampler = StartSampler(spec, float(cfg["record"]["protocol"]["es_bin_width"]))
    hw = float(sw["near_tie_half_width"])
    labels = sampler.stratum_labels(2, hw)
    n_bins = sampler.n_bins(2)
    lam_t = float(sampler.coverage_lambda_t(2))
    lam_p = float(sw["lambda_P"])
    lam = np.array([lam_t, lam_p, 1.0 - lam_p - lam_t])
    rows = _stage2_updates(d)
    upd = np.array([int(r["update"]) for r in rows])
    probs = LC1.per_update_probs(d, 2, n_bins, upd)
    shares = np.stack([probs[:, labels == k].sum(axis=1) for k in (0, 1, 2)], axis=1)
    dev = np.abs(shares - lam[None, :]).max(axis=0)
    counts = np.array([[int(r["n_start_tail"]), int(r["n_start_near"]), int(r["n_start_mid"])] for r in rows], dtype=float)
    return {"lambda": lam.tolist(), "max_dev": dev.tolist(), "n_updates": int(upd.size),
            "mean_probability_share": shares.mean(axis=0).tolist(),
            "realised_share": (counts.sum(axis=0) / counts.sum()).tolist(),
            "ok_tail": bool(dev[0] <= MASS_TOL), "ok_within": bool(dev.max() <= MASS_TOL)}


# ---------------------------------------------------------------------------------------------- sampling weights
def _digest(p: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(p, dtype="<f8").tobytes()).hexdigest()[:16]


def sampling_weights_check(d: Path) -> Dict[str, Any]:
    """The probabilities in force after every check against ``p_digest_next``.

    ``probs_next`` is used where the run records it (the within-stratum arms); otherwise the probabilities of the next update
    are reconstructed from ``probs_first_update`` / ``probs_table``. At the last check of the stage the runner writes an empty
    digest (and ``probs_next`` is NaN); every other check must match.
    """
    from envs.curriculum_env import StartSampler
    cfg = RC2._json(d / "run_config.json")
    spec = LC1._spec(cfg)
    sampler = StartSampler(spec, float(cfg["record"]["protocol"]["es_bin_width"]))
    n_bins = sampler.n_bins(2)
    z = np.load(d / "ms_binmaps_stage2.npz")
    checks = RC2._csv(d / "ms_checks_stage2.csv")
    upd = z["update"]
    out: Dict[str, Any] = {"n_checks": int(upd.size), "n_checked": 0, "n_bad": 0, "source": "", "errors": []}
    if upd.size != len(checks):
        out["errors"].append(f"{upd.size} bin-map rows against {len(checks)} check rows")
        out["ok"] = False
        return out
    have_next = "probs_next" in z.files
    out["source"] = "probs_next" if have_next else "probs_table"
    last_update = int(_stage2_updates(d)[-1]["update"])
    for i, row in enumerate(checks):
        u = int(upd[i])
        dg = row["p_digest_next"]
        if u >= last_update:                       # the last check of the stage: no next update
            if dg != "" or (have_next and not np.isnan(z["probs_next"][i]).all()):
                out["n_bad"] += 1
                out["errors"].append(f"check at update {u}: expected an empty digest and NaN probabilities")
            continue
        p = z["probs_next"][i] if have_next else LC1.per_update_probs(d, 2, n_bins, np.array([u + 1]))[0]
        out["n_checked"] += 1
        if _digest(p) != dg:
            out["n_bad"] += 1
            if len(out["errors"]) < 3:
                out["errors"].append(f"check at update {u}: digest of the probabilities {_digest(p)} != p_digest_next {dg!r}")
    out["ok"] = bool(out["n_bad"] == 0 and out["n_checked"] > 0 and not out["errors"])
    return out


# ---------------------------------------------------------------------------------------------- pilot checks
def check_pilot(root: Path, qs: Sequence[int], seeds: Sequence[int], code_commit: Optional[str],
                arms: Sequence[str] = rc.ARMS, ramp: Tuple[int, int] = RAMP,
                fb25_upto: int = FB25_UPTO, x2_upto: int = X2_UPTO, export_every: int = EXPORT_EVERY) -> Dict[str, Any]:
    """All checks of the pilot root (see the module docstring)."""
    arms = tuple(arms)
    base = LC1.check_root(root, list(arms), qs, seeds, code_commit)
    scales, masses, weights_chk, c_init, c_fb25, c_x2 = [], [], [], [], [], []
    for q in qs:
        for s in seeds:
            for arm in arms:
                d = run_dir(root, q, s, arm)
                if not d.is_dir():
                    for lst in (scales, masses):
                        lst.append({"arm": arm, "q": q, "seed": s, "ok": False, "errors": ["missing run"]})
                    continue
                ref = E1_SCALE_REF if arm in rc.E1_ARMS else E2_SCALE_REF
                scales.append({"arm": arm, "q": q, "seed": s, **RC2.scale_check(d, ref, ramp)})
                try:
                    m = strata_masses(d)
                    ok = m["ok_within"] if arm in WITHIN_ARMS else m["ok_tail"]
                    masses.append({"arm": arm, "q": q, "seed": s, **m, "ok": ok})
                except Exception as exc:  # noqa: BLE001 - a missing file is a failure
                    masses.append({"arm": arm, "q": q, "seed": s, "ok": False, "errors": [f"{type(exc).__name__}: {exc}"]})
                if arm in rc.E2_ARMS:
                    try:
                        weights_chk.append({"arm": arm, "q": q, "seed": s, **sampling_weights_check(d)})
                    except Exception as exc:  # noqa: BLE001
                        weights_chk.append({"arm": arm, "q": q, "seed": s, "ok": False, "errors": [f"{type(exc).__name__}: {exc}"]})
            e1 = [a for a in arms if a in rc.E1_ARMS]
            if len(e1) == len(rc.E1_ARMS):
                r = R3C.init_check(root, e1, q, s)
                r["all_arms"] = R3C.init_check(root, list(arms), q, s)["pass"]
                c_init.append(r)
            if FB25_REF in arms:
                for arm in (a for a in FB25_ARMS if a in arms):
                    r = prefix_identity(run_dir(root, q, s, FB25_REF), run_dir(root, q, s, arm), fb25_upto, FB25_SKIP, export_every)
                    c_fb25.append({"arm": arm, "reference": FB25_REF, "q": q, "seed": s, **r,
                                   "pass": bool(r["ALL"] and r["first_export_after_differs"])})
            if X2_ARM in arms and X2_REF in arms:
                r = prefix_identity(run_dir(root, q, s, X2_REF), run_dir(root, q, s, X2_ARM), x2_upto, (), export_every)
                c_x2.append({"arm": X2_ARM, "reference": X2_REF, "q": q, "seed": s, **r,
                             "pass": bool(r["ALL"] and r["first_export_after_differs"])})
    npass = lambda xs: sum(1 for x in xs if x.get("pass") or x.get("ok"))  # noqa: E731
    nrun = len(qs) * len(seeds)
    n_e1 = sum(1 for a in arms if a in rc.E1_ARMS)
    n_e2 = sum(1 for a in arms if a in rc.E2_ARMS)
    summary = {"scale_ok": npass(scales), "scale_n": len(scales), "strata_masses_ok": npass(masses), "strata_masses_n": len(masses),
               "sampling_weights_ok": npass(weights_chk), "sampling_weights_n": len(weights_chk),
               "C_INIT_pass": npass(c_init), "C_INIT_n": len(c_init), "C_FB25_pass": npass(c_fb25), "C_FB25_n": len(c_fb25),
               "C_X2_pass": npass(c_x2), "C_X2_n": len(c_x2)}
    expected = {"scale_n": len(arms) * nrun, "strata_masses_n": len(arms) * nrun, "sampling_weights_n": n_e2 * nrun,
                "C_INIT_n": nrun if n_e1 == len(rc.E1_ARMS) else 0,
                "C_FB25_n": (sum(1 for a in FB25_ARMS if a in arms) * nrun) if FB25_REF in arms else 0,
                "C_X2_n": nrun if (X2_ARM in arms and X2_REF in arms) else 0}
    pairs = (("scale_ok", "scale_n"), ("strata_masses_ok", "strata_masses_n"), ("sampling_weights_ok", "sampling_weights_n"),
             ("C_INIT_pass", "C_INIT_n"), ("C_FB25_pass", "C_FB25_n"), ("C_X2_pass", "C_X2_n"))
    all_ok = bool(base["all_ok"] and all(summary[ok] == summary[n] == expected[n] for ok, n in pairs))
    within_ok = [m for m in masses if m["arm"] in WITHIN_ARMS]
    reported = {a: {"realised_share_mean": np.mean([m["realised_share"] for m in masses if m["arm"] == a and "realised_share" in m],
                                                   axis=0).tolist(),
                    "mean_probability_share": np.mean([m["mean_probability_share"] for m in masses
                                                       if m["arm"] == a and "mean_probability_share" in m], axis=0).tolist()}
                for a in arms if any(m["arm"] == a and "realised_share" in m for m in masses)}
    ignored = {"csv_wall_clock_columns": sorted(WALL_COLS), "csv_label_columns": sorted(LABEL_COLS),
               "C_FB25_skipped_columns": list(FB25_SKIP)}
    return {"tool": "tools/ms/r4_launch_checks.py", "mode": "pilot", "root": str(root), "code_commit": code_commit,
            "arms": list(arms), "ramp": list(ramp), "base": base, "summary": summary, "expected": expected, "all_ok": all_ok,
            "strata_shares_reported": reported, "within_arms_masses_exact": bool(within_ok and all(m["ok"] for m in within_ok)),
            "ignored": ignored, "scale": scales, "strata_masses": masses, "sampling_weights": weights_chk, "C_INIT": c_init,
            "C_FB25": c_fb25, "C_X2": c_x2}


# ---------------------------------------------------------------------------------------------- C-MS6
def check_cms6(root: Path, ms_r3_root: Path, qs: Sequence[int], seeds: Sequence[int], code_commit: Optional[str],
               arms: Sequence[str] = rc.CMS6_ARMS) -> Dict[str, Any]:
    """C-MS6: each control run equals MS-R3's run of the same name over the whole run (``whole_run_identity``)."""
    base = LC1.check_root(root, list(arms), qs, seeds, code_commit)
    runs = []
    for q in qs:
        for s in seeds:
            for arm in arms:
                r = R3C.whole_run_identity(run_dir(ms_r3_root, q, s, arm), run_dir(root, q, s, arm))
                runs.append({"arm": arm, "q": q, "seed": s, **r, "pass": bool(r["ALL"])})
    n_pass = sum(1 for r in runs if r["pass"])
    expected = len(arms) * len(qs) * len(seeds)
    return {"tool": "tools/ms/r4_launch_checks.py", "mode": "cms6", "root": str(root), "ms_r3_pilot_root": str(ms_r3_root),
            "code_commit": code_commit, "arms": list(arms), "base": base,
            "summary": {"C_MS6_pass": n_pass, "C_MS6_n": len(runs)}, "expected": {"C_MS6_n": expected},
            "all_ok": bool(base["all_ok"] and n_pass == len(runs) == expected),
            "ignored": {"csv_wall_clock_columns": sorted(WALL_COLS), "csv_label_columns": sorted(LABEL_COLS),
                        "gates_json_paths": list(R3C.GATES_IGNORED),
                        "files_not_compared": list(R3C.NOT_COMPARED_FILES)},
            "C_MS6": runs}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI; exit code 0 iff every check passed."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    p.add_argument("--mode", choices=("pilot", "cms6"), required=True)
    p.add_argument("--root", required=True, help="the wave root: q*/seed*/<arm>")
    p.add_argument("--ms-r3-pilot-root", default=None, help="mode cms6: MS-R3's pilot root (the reference runs)")
    p.add_argument("--code-commit", required=True, help="the launch commit every manifest must carry")
    p.add_argument("--out", required=True, help="the JSON to write")
    p.add_argument("--qs", type=int, nargs="+", default=list(rc.QS))
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="default: the block of the wave")
    p.add_argument("--arms", nargs="+", default=None)
    p.add_argument("--ramp-first", type=int, default=RAMP[0], help="tests only")
    p.add_argument("--ramp-last", type=int, default=RAMP[1], help="tests only")
    p.add_argument("--export-every", type=int, default=EXPORT_EVERY, help="tests only")
    p.add_argument("--fb25-upto", type=int, default=FB25_UPTO, help="tests only")
    p.add_argument("--x2-upto", type=int, default=X2_UPTO, help="tests only")
    a = p.parse_args(argv)
    if a.mode == "cms6":
        if not a.ms_r3_pilot_root:
            p.error("--mode cms6 needs --ms-r3-pilot-root")
        res = check_cms6(Path(a.root), Path(a.ms_r3_pilot_root), a.qs, a.seeds or list(rc.SEEDS_CMS6), a.code_commit,
                         a.arms or rc.CMS6_ARMS)
    else:
        res = check_pilot(Path(a.root), a.qs, a.seeds or list(rc.SEEDS), a.code_commit, a.arms or rc.ARMS,
                          ramp=(a.ramp_first, a.ramp_last), fb25_upto=a.fb25_upto, x2_upto=a.x2_upto, export_every=a.export_every)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"mode": res["mode"], "base_n_ok_per_check": res["base"]["n_ok_per_check"],
                      "base_start_share_tests": res["base"]["start_share_tests"], "summary": res["summary"],
                      "expected": res["expected"], "all_ok": res["all_ok"]}, indent=1))
    keys = ("C_MS6",) if a.mode == "cms6" else ("scale", "strata_masses", "sampling_weights", "C_INIT", "C_FB25", "C_X2")
    for check in keys:
        for x in [x for x in res[check] if not (x.get("pass") or x.get("ok"))][:3]:
            print(f"FAILED {check} {x.get('arm', '')} q={x['q']} seed={x['seed']}: "
                  f"{x.get('first_difference') or x.get('errors') or 'differs'}")
    return 0 if res["all_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
