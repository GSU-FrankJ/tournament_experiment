#!/usr/bin/env python3
"""Automatic checks R1-R7 of the v2.0 re-rehearsal; writes results/v2_T2_locked/rehearsal_v2_0_checks.json.

Runs: q in {50, 60} x seeds 10501-10510 through the v2.0 entry point into
results/v2_T2_locked/rehearsal_v2_0/. Run this tool with OMP/MKL/OPENBLAS_NUM_THREADS=1 (it rebuilds
the continuation table in-process for R2 and refuses to start otherwise).

R1  Phase A unchanged. The end-of-A training-relevant state of every run equals ``rehearsal_v1_1``'s
    (canonical worktree): state_end_A.pt (actor, critic, opponent, frozen, both Adam states, minibatch
    stream, the four numpy streams, the torch generator), the weight exports of updates 1-1600, the
    Phase-A part of the histories, logs and CSVs, gateA_{final,development}.npz, and the end-of-A gate
    values (reported.end_of_A, the G-A metric values, the development-tier G-A values).
R2  Phase B equals the R1 pilot. The end-of-B training-relevant state equals
    ``results/v2_refine/stage1/q*/seed*/B_expcont/`` (the Check-2 relation: in-process Phase B against
    the branched Phase B): state_end_B.pt, the Phase-B part of the histories, logs, exports and CSVs,
    checkpoint_weights.npz, final_{final,development}.npz, the gate-metric values (Gmax, S1, the end-of-B
    reported scalars) and the continuation table (the pilot wrote no NPZ: rebuilt from the pilot run's
    frozen snapshot by utils.v2_continuation and compared bit for bit with continuation_table.npz; the
    rule record against the pilot's v2_run_summary.json). The snapshot-refresh counter is reported
    separately. drift_test.json is implied by R1 (end-of-A actor) and R2 (frozen snapshot in
    state_end_B.pt); induced_band.json and band_sweep.npz have no pilot counterpart and are not compared.
R3  Gates. All 20 pass G-A, G-F, G-N and G-S (gates.json). R3b (decision D4): >= 19/20 under G-S and the
    recomputed normal-model probability >= 0.90 at each q (confirmation_analysis_v2_0.py --rehearsal-safety).
R4  Global RNGs. No violation, exit code 0 (gates.json, status.json, launcher log), the table build moved
    no RNG; 20/20.
R5  Manifests. v2.0 protocol hash and version, the launch commit, clean_tree true, table SHA-256 present
    and equal to the file on disk; 20/20.
R6  tools/v2/confirmation_analysis_v2_0.py runs on the rehearsal root and its recomputed verdicts agree
    with gates.json in every run (it also writes the paired comparison with rehearsal_v1_1).
R7  At the launch commit: full test suite passes except the known test_registry_canonicalization failure
    (no other failure, no error); C7 bit-exact against C7_REF, the reference run in the canonical
    worktree.

Usage: python tools/v2/v2_0_rehearsal_checks.py --launch-commit <sha>
       [--root DIR] [--qs 50 60] [--seeds 10501 ...] [--out FILE] [--skip-r7]  (the last four: dry checks only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))

import cr1_compare as C  # noqa: E402

LK = ROOT / "results" / "v2_T2_locked"
REH = LK / "rehearsal_v2_0"
V11 = C.REF_DEFAULT                                       # canonical rehearsal_v1_1
# C7 reference run: its checkpoint.pt is gitignored, so it exists only in the canonical worktree
C7_REF = (C.CANONICAL_ROOT / "results" / "v2_pilots" / "phase1_regression" / "before"
          / "FINAL_A400_B25_C25" / "tel_q50_s10501")
PILOT = ROOT / "results" / "v2_refine" / "stage1"
OUT = LK / "rehearsal_v2_0_checks.json"
AN_OUT = LK / "rehearsal_v2_0_analysis"
PY = sys.executable
QS, SEEDS = (50, 60), tuple(range(10501, 10511))
PILOT_ARM = "B_expcont"
KNOWN_FAILURE = "tests/test_registry_canonicalization.py::test_registry_canonicalization"

C.MODES["locked_A"] = C.ModeSpec(
    "locked_A", 1, 1600, ("state_end_A.pt",), "A", (("v2_checkpoints_A.csv", ("v2_checkpoints_A.csv",)),),
    snap="before_B", npz_pairs=(("gateA_final.npz", "gateA_final.npz"),
                                ("gateA_development.npz", "gateA_development.npz")))
C.MODES["locked_B_vs_pilot"] = C.ModeSpec(
    "locked_B_vs_pilot", 1601, 2200, ("state_end_B.pt",), "B",
    (("v2_checkpoints.csv", ("v2_checkpoints_B.csv",)),), cross_config=True, snap="from_B",
    npz_pairs=(("final_final.npz", "final_final.npz"), ("final_development.npz", "final_development.npz"),
               ("checkpoint_weights.npz", "checkpoint_weights.npz")))


def _gates_a(ref: Path, new: Path) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    """End-of-A gate values: v2.0 gates.json against the rehearsal_v1_1 gates.json (a missing file fails)."""
    res = C.Result()
    keys = ("eta_final", "eta_dev", "rmse", "tail")

    def both() -> Tuple[Dict, Dict]:
        return C.load_json(ref / "gates.json"), C.load_json(new / "gates.json")

    def reported():
        r, n = both()
        return C.first_diff(r["reported"]["end_of_A"], n["reported"]["end_of_A"], "reported.end_of_A")

    def metrics():
        r, n = both()
        return C.first_diff({k: r["metric_values"][k] for k in keys}, {k: n["metric_values"][k] for k in keys},
                            "metric_values")

    def dev_values():
        r, n = both()
        return C.first_diff(r["dev_tier_values"]["G-A"], n["dev_tier_values"]["G-A"], "dev_tier_values.G-A")
    res.check("reported.end_of_A", reported)
    res.check("metric_values_A", metrics)
    res.check("dev_tier_values.G-A", dev_values)
    return dict(res.fields), {k: d.as_dict() for k, d in res.diffs.items()}


def r1_one(q: int, s: int, new_root: Path) -> Dict[str, Any]:
    """R1 for one run: v2.0 against rehearsal_v1_1, Phase A."""
    ref, new = C.run_dir(V11, q, s, None), C.run_dir(new_root, q, s, None)
    out = C.compare_run("locked_A", ref, new)
    fields, diffs = dict(out["fields"]), dict(out["differences"])
    f2, d2 = _gates_a(ref, new)
    fields.update(f2)
    diffs.update(d2)
    return {"q": q, "seed": s, "fields": fields, "ALL": bool(out["ALL"] and all(fields.values())),
            "differences": diffs, "info": out["info"]}


def _rebuild_pilot_table(pdir: Path, spec: Any) -> Any:
    """Table of the pilot run: rebuilt from the pilot's frozen stage-2 snapshot with the default rule."""
    from agents.ppo_curriculum import BetaActor, PPOConfig
    from utils import v2_continuation as vc
    state = C.load_state(pdir / "state_end_B.pt")
    cfg = PPOConfig()
    actor = BetaActor(cfg.hidden, cfg.c_min, cfg.mu_clamp, torch.Generator().manual_seed(0))
    actor.load_state_dict(state["agent"]["frozen"])
    actor.eval()
    return vc.build_continuation_table(actor, spec, stage=spec.T, step=0.05)


def r2_one(q: int, s: int, new_root: Path, proto: Dict) -> Dict[str, Any]:
    """R2 for one run: v2.0 against the B_expcont pilot, Phase B and the table."""
    pdir, new = C.run_dir(PILOT, q, s, PILOT_ARM), C.run_dir(new_root, q, s, None)
    out = C.compare_run("locked_B_vs_pilot", pdir, new)
    res = C.Result()
    res.fields, res.info = dict(out["fields"]), dict(out["info"])
    res._in_all = {k: True for k in res.fields}
    cache: Dict[str, Any] = {}

    def fv2() -> Dict[str, Any]:
        if "fv2" not in cache:
            cache["fv2"] = C.load_json(pdir / "final_v2.json")
        return cache["fv2"]

    def gates() -> Dict[str, Any]:
        if "gates" not in cache:
            cache["gates"] = C.load_json(new / "gates.json")
        return cache["gates"]
    for tier in ("final", "development"):
        res.check(f"end_of_B.{tier}_scalars", lambda tier=tier: C._scalars_diff(
            fv2()[tier], gates()["reported"]["end_of_B"][tier], f"end_of_B.{tier}", res.info))

    def metric_values():
        mv, dv, f = gates()["metric_values"], gates()["dev_tier_values"], fv2()
        new_v = {"gmax_final": mv["gmax_final"], "gmax_dev": mv["gmax_dev"], "s1": mv["s1"],
                 "s1_dev": dv["G-S"]["stage1_rel_err_abs"], "s1_dev_alias": dv["S1"]["stage1_rel_err_abs"]}
        ref_v = {"gmax_final": f["final"]["Gmax_full_over_dw"], "gmax_dev": f["development"]["Gmax_full_over_dw"],
                 "s1": f["final"]["stage1_rel_err_abs"], "s1_dev": f["development"]["stage1_rel_err_abs"],
                 "s1_dev_alias": f["development"]["stage1_rel_err_abs"]}
        return C.first_diff(ref_v, new_v, "gate_metric_values_B")
    res.check("gate_metric_values_B", metric_values)

    def table_arrays():
        from envs.curriculum_env import GameSpec
        spec = GameSpec(**proto["records"][str(q)]["game"])
        cache["pilot_table"] = _rebuild_pilot_table(pdir, spec)
        t = cache["pilot_table"]
        return C.first_diff({"y_grid": t.y_grid, "values": t.values}, C.load_npz(new / "continuation_table.npz"),
                            "continuation_table")
    res.check("continuation_table_bit_identical", table_arrays)

    def table_meta():
        mine = gates()["continuation_table"]["rule"]
        rec = C.load_json(pdir / "v2_run_summary.json").get("continuation_table") or {}
        keys = [k for k in mine if k in rec and k != "build_seconds"]
        res.info["table_rule_new"] = mine
        return C.first_diff({k: rec[k] for k in keys}, {k: mine[k] for k in keys}, "table_rule")
    res.check("continuation_table_rule_vs_pilot_record", table_meta)
    diffs = dict(out["differences"])
    diffs.update({k: d.as_dict() for k, d in res.diffs.items()})
    return {"q": q, "seed": s, "fields": res.fields, "ALL": bool(out["ALL"] and all(res.fields.values())),
            "differences": diffs, "info": res.info}


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """pass / counts / failing fields / first differences of a per-run check."""
    failing = sorted({k for r in rows for k, v in r["fields"].items() if not v})
    firsts = []
    for r in rows:
        if not r["ALL"]:
            k = next((k for k, v in r["fields"].items() if not v), None)
            firsts.append({"q": r["q"], "seed": r["seed"], "field": k, **(r["differences"].get(k, {}))})
    return {"pass": bool(rows and all(r["ALL"] for r in rows)), "n_identical": sum(1 for r in rows if r["ALL"]),
            "n": len(rows), "failing_fields": failing, "first_differences": firsts}


def run_checks_345(a: argparse.Namespace, new_root: Path, L: Any) -> Tuple[List[bool], List[bool], List[bool]]:
    """R3 (gates), R4 (RNGs, exit codes) and R5 (manifests) per run; one boolean per run and check."""
    launcher = (new_root / "launcher.out").read_text() if (new_root / "launcher.out").exists() else ""
    r3, r4, r5 = [], [], []
    for q in a.qs:
        for s in a.seeds:
            d = C.run_dir(new_root, q, s, None)
            b3 = b4 = b5 = False
            try:
                g, st, m = (C.load_json(d / f) for f in ("gates.json", "status.json", "manifest.json"))
                b3 = bool(g["G-A"]["pass"] and g["G-F"]["pass"] and g["G-N"]["pass"] and g["G-S"]["pass"])
                rc = re.search(rf"q{q} s{s} rc=(\d+)", launcher)
                tb = g["continuation_table"]
                b4 = bool(g["global_rng"]["status"] == "ok" and not g["global_rng"]["violations"]
                          and st.get("exit_code") == 0 and rc is not None and rc.group(1) == "0"
                          and all(tb["rng_unchanged_by_build"].values()))
                disk = hashlib.sha256((d / "continuation_table.npz").read_bytes()).hexdigest()
                b5 = bool(m["locked_protocol"]["sha256"] == L.PROTOCOL_SHA256
                          and str(m["locked_protocol"]["version"]) == "2.0" and str(g["protocol_version"]) == "2.0"
                          and m["git"]["commit"].startswith(a.launch_commit) and m["clean_tree"] is True
                          and g["clean_tree"] is True and tb.get("npz_sha256") == disk
                          and (m.get("continuation_table") or {}).get("npz_sha256") == disk)
            except Exception:  # noqa: BLE001 - a missing or unreadable record fails the checks it feeds
                pass
            r3.append(b3), r4.append(b4), r5.append(b5)
    return r3, r4, r5


def run_check_7(launch_commit: str) -> Dict[str, Any]:
    """R7: full suite (only the known registry failure, no error) and C7 bit-exact at the launch commit."""
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        t = subprocess.run([PY, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"], capture_output=True,
                           text=True, cwd=ROOT, env=env)
        (LK / "rehearsal_v2_0_pytest.txt").write_text(t.stdout[-20000:] + t.stderr[-5000:])
        failed = re.findall(r"^FAILED (\S+)", t.stdout, flags=re.M)
        errors = re.findall(r"^ERROR (\S+)", t.stdout, flags=re.M)
        summ = [ln for ln in t.stdout.splitlines() if re.search(r"\d+ (passed|failed)", ln)]
        tests_ok = failed == [KNOWN_FAILURE] and not errors
        c7dir = ROOT / "results" / "v2_pilots" / "phase2_regression" / f"v2_full_{launch_commit[:7]}"
        if not c7dir.exists():
            subprocess.run([PY, "-B", "run/run_v2_stagewise.py", "--config",
                            "results/v2_pilots/phase2_regression/run_config.json", "--out-dir", str(c7dir)],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        c = subprocess.run([PY, "tools/v2/compare_runs.py", str(C7_REF), str(c7dir)],
                           cwd=ROOT, capture_output=True, text=True)
        (c7dir.parent / f"v2_full_{launch_commit[:7]}.compare.txt").write_text(c.stdout)
        c7_ok = c.stdout.strip().endswith("IDENTICAL")
        man = json.load(open(c7dir / "manifest.json"))
        return {"pass": bool(tests_ok and c7_ok and man["git"]["commit"].startswith(launch_commit)
                             and man["git"]["dirty"] is False),
                "pytest_summary": summ[-1] if summ else None, "pytest_failed": failed, "pytest_errors": errors,
                "c7_identical": c7_ok, "c7_commit": man["git"]["short"], "c7_dirty": man["git"]["dirty"],
                "c7_reference": str(C7_REF),
                "pytest_log": "results/v2_T2_locked/rehearsal_v2_0_pytest.txt"}
    except Exception as exc:  # noqa: BLE001 - recorded as a failed check
        return {"pass": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    """CLI: run R1-R7 and write the checks file; exit code 0 iff every check (and R3b) passes."""
    p = argparse.ArgumentParser()
    p.add_argument("--launch-commit", required=True)
    p.add_argument("--root", default=str(REH))
    p.add_argument("--qs", type=int, nargs="+", default=list(QS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--skip-r7", action="store_true", help="dry check only; the recorded verdict requires R7")
    a = p.parse_args()
    bad_env = [k for k in ("OMP", "MKL", "OPENBLAS") if os.environ.get(f"{k}_NUM_THREADS") != "1"]
    if bad_env:
        print(f"refusing to run: set {', '.join(k + '_NUM_THREADS=1' for k in bad_env)} (R2 rebuilds the table in-process)")
        return 4
    torch.set_num_threads(1)
    new_root, out_path = Path(a.root), Path(a.out)
    import run.run_v2_T2_locked as L
    proto = L.load_protocol()
    n_expected = len(a.qs) * len(a.seeds)
    res: Dict[str, Any] = {"launch_commit": a.launch_commit, "v2_0_protocol_sha256": L.PROTOCOL_SHA256,
                           "root": str(new_root), "reference_v1_1": str(V11), "pilot": str(PILOT),
                           "n_expected": n_expected}
    # ---- R1, R2
    r1 = [r1_one(q, s, new_root) for q in a.qs for s in a.seeds]
    r2 = [r2_one(q, s, new_root, proto) for q in a.qs for s in a.seeds]
    for tag, rows in (("R1", r1), ("R2", r2)):
        pd.DataFrame([{"q": r["q"], "seed": r["seed"], **r["fields"], "ALL": r["ALL"]} for r in rows]).to_csv(
            out_path.with_name(f"{out_path.stem}_{tag}_details.csv"), index=False)
    res["R1"] = {**summarize(r1), "details": f"{out_path.stem}_R1_details.csv",
                 "snapshot_refreshes": [{"q": r["q"], "seed": r["seed"], **r["info"].get("state_end_A.pt:snapshot_refreshes", {})} for r in r1]}
    res["R2"] = {**summarize(r2), "details": f"{out_path.stem}_R2_details.csv",
                 "snapshot_refreshes_reported_separately": [
                     {"q": r["q"], "seed": r["seed"], **r["info"].get("state_end_B.pt:snapshot_refreshes", {})} for r in r2]}
    # ---- R6 first (R3b reads its output)
    an_dir = AN_OUT if str(new_root) == str(REH) else out_path.with_name(out_path.stem + "_analysis")
    an = subprocess.run([PY, str(ROOT / "tools" / "v2" / "confirmation_analysis_v2_0.py"), "--root", str(new_root),
                         "--seeds", str(min(a.seeds)), str(max(a.seeds)), "--rehearsal-safety", "--paired-root",
                         str(V11), "--paired-label", "rehearsal_v1_1", "--out", str(an_dir)],
                        capture_output=True, text=True, cwd=ROOT)
    ag = pd.read_csv(an_dir / "agreement.csv") if an.returncode == 0 else pd.DataFrame()
    res["R6"] = {"pass": bool(an.returncode == 0 and len(ag) == n_expected and ag.all_agree.all()),
                 "returncode": an.returncode, "n_agree": int(ag.all_agree.sum()) if len(ag) else 0, "n": len(ag),
                 "max_abs_value_diff": float(ag.filter(like="absdiff_").to_numpy().max()) if len(ag) else None,
                 "output": str(an_dir), "stderr_tail": an.stderr[-2000:] if an.returncode != 0 else ""}
    # ---- R3, R3b, R4, R5
    r3, r4, r5 = run_checks_345(a, new_root, L)
    safety_path = an_dir / "rehearsal_safety.json"
    rs = json.load(open(safety_path)) if an.returncode == 0 and safety_path.exists() else {}
    res["R3"] = {"pass": all(r3) and len(r3) == n_expected, "n_pass": sum(r3), "n": len(r3),
                 "R3b": {"pass": bool(rs.get("pass")), "pooled_n_G-S_pass": rs.get("pooled_n_G-S_pass"),
                         "needed": rs.get("pooled_needed"), "per_q": rs.get("per_q"), "source": str(safety_path)}}
    res["R4"] = {"pass": all(r4) and len(r4) == n_expected, "n_ok": sum(r4), "n": len(r4)}
    res["R5"] = {"pass": all(r5) and len(r5) == n_expected, "n_ok": sum(r5), "n": len(r5)}
    # ---- R7
    res["R7"] = {"pass": None, "note": "skipped (dry check)"} if a.skip_r7 else run_check_7(a.launch_commit)
    keys = ("R1", "R2", "R3", "R4", "R5", "R6", "R7")
    res["ALL_PASS"] = bool(all(res[k]["pass"] for k in keys) and res["R3"]["R3b"]["pass"])
    json.dump(res, open(out_path, "w"), indent=1)
    summary = {k: res[k]["pass"] for k in keys}
    summary["R3b"] = res["R3"]["R3b"]["pass"]
    summary["ALL_PASS"] = res["ALL_PASS"]
    print(json.dumps(summary, indent=1))
    return 0 if res["ALL_PASS"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
