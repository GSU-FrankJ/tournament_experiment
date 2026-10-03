#!/usr/bin/env python3
"""Automatic checks R1-R6 of the v1.1 re-rehearsal; writes results/v2_T2_locked/rehearsal_v1_1_checks.json.

R1  training-relevant state (decision D1 of the v1.1 round) of every (q, seed) of
    results/v2_T2_locked/rehearsal_v1_1/ equals the v1.0 rehearsal (results/v2_T2_locked/rehearsal/):
    state_end_A.pt / state_end_B.pt (actor, critic, opponent, frozen, both Adam states, minibatch RNG,
    numpy streams, torch generator); weights/u*.npz and checkpoint_weights.npz; train_history.json
    (history, stability, verifier_calls, curriculum, snapshots, weight_checkpoints) and v2_updates.csv
    without wall-clock fields; v2_checkpoints_{A,B}.csv without wall-clock fields; gate-metric values
    (v1.0 gates.json criteria vs v1.1 metric_values / dev_tier_values); gateA_*.npz, final_*.npz,
    band_sweep.npz, induced_band.json, drift_test.json. Excluded: verdicts, manifests, global RNG states.
R2  all runs pass G-A, G-F and G-N (gates.json).
R3  no global-RNG violation and exit code 0 in every run (gates.json, status.json, launcher log).
R4  every manifest: v1.1 protocol hash and version, the launch commit, clean_tree true.
R5  tools/v2/confirmation_analysis.py runs on the rehearsal root and its recomputed verdicts agree with
    gates.json in every run.
R6  at the launch commit: full test suite passes except the known test_registry_canonicalization
    failure; C7 bit-exact.

Usage: python tools/v2/v1_1_rehearsal_checks.py --launch-commit <sha>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LK = ROOT / "results" / "v2_T2_locked"
V10, V11 = LK / "rehearsal", LK / "rehearsal_v1_1"
OUT = LK / "rehearsal_v1_1_checks.json"
PY = sys.executable
QS, SEEDS = (50, 60), range(10501, 10511)
WALL_KEYS = {"time_sec", "update_wall_sec", "elapsed_phase_wall_sec", "verifier_sec"}
NPZ = ("gateA_final.npz", "gateA_development.npz", "final_final.npz", "final_development.npz", "band_sweep.npz",
       "checkpoint_weights.npz")


def teq(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(teq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(teq(x, y) for x, y in zip(a, b))
    return a == b


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in WALL_KEYS}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def npz_eq(a: Path, b: Path) -> bool:
    za, zb = np.load(a), np.load(b)
    return set(za.files) == set(zb.files) and all(np.array_equal(za[k], zb[k]) for k in za.files)


def r1_one(q: int, s: int) -> dict:
    a, b = V10 / f"q{q}" / f"seed{s}", V11 / f"q{q}" / f"seed{s}"
    r = {"q": q, "seed": s}
    for st in ("state_end_A.pt", "state_end_B.pt"):
        sa, sb = torch.load(a / st, weights_only=False), torch.load(b / st, weights_only=False)
        for k in ("actor", "critic", "opponent", "frozen", "opt_actor", "opt_critic", "rng_minibatch"):
            r[f"{st}:{k}"] = teq(sa["agent"].get(k), sb["agent"].get(k))
        r[f"{st}:rng_streams"] = teq(sa["rng"], sb["rng"])
        r[f"{st}:torch_generator"] = teq(sa["torch_generator_state"], sb["torch_generator_state"])
    ea = sorted(os.path.basename(x) for x in glob.glob(str(a / "weights" / "u*.npz")))
    eb = sorted(os.path.basename(x) for x in glob.glob(str(b / "weights" / "u*.npz")))
    r["weight_exports"] = ea == eb and len(ea) == 88 and all(npz_eq(a / "weights" / f, b / "weights" / f) for f in ea)
    for f in NPZ:
        r[f] = npz_eq(a / f, b / f)
    ha, hb = json.load(open(a / "train_history.json")), json.load(open(b / "train_history.json"))
    for k in ("history", "stability", "verifier_calls", "curriculum", "snapshots", "weight_checkpoints"):
        r[f"train_history:{k}"] = json.dumps(strip(ha[k]), sort_keys=True, default=str) == json.dumps(strip(hb[k]), sort_keys=True, default=str)
    for f in ("v2_updates.csv", "v2_checkpoints_A.csv", "v2_checkpoints_B.csv"):
        da, db = pd.read_csv(a / f), pd.read_csv(b / f)
        da, db = da.drop(columns=[c for c in da.columns if c in WALL_KEYS]), db.drop(columns=[c for c in db.columns if c in WALL_KEYS])
        r[f] = list(da.columns) == list(db.columns) and da.equals(db)
    for f in ("induced_band.json", "drift_test.json"):
        r[f] = json.load(open(a / f)) == json.load(open(b / f))
    ga, gb = json.load(open(a / "gates.json")), json.load(open(b / "gates.json"))
    v10 = {c["metric"]: (c["value_final"], c["value_dev"]) for blk in ("G-A", "G-F") for c in ga[blk]["criteria"]}
    mv, dv = gb["metric_values"], gb["dev_tier_values"]
    pairs = {"eta_T_over_dw": (mv["eta_final"], mv["eta_dev"]),
             "stage2_rmse_pos_over_g2_0": (mv["rmse"], dv["G-A"]["stage2_rmse_pos_over_g2_0"]),
             "stage2_tail_mean_over_g2_0": (mv["tail"], dv["G-A"]["stage2_tail_mean_over_g2_0"]),
             "Gmax_full_over_dw": (mv["gmax_final"], mv["gmax_dev"]),
             "stage1_rel_err_abs": (mv["s1"], dv["S1"]["stage1_rel_err_abs"])}
    r["gate_metric_values"] = all(tuple(map(float, v10[k])) == tuple(map(float, pairs[k])) for k in pairs)
    r["reported_metrics"] = json.dumps(ga["reported"], sort_keys=True) == json.dumps(gb["reported"], sort_keys=True)
    r["ALL"] = all(v for k, v in r.items() if k not in ("q", "seed"))
    return r


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--launch-commit", required=True)
    p.add_argument("--skip-r6", action="store_true", help="for a dry check only; the recorded verdict requires R6")
    a = p.parse_args()
    import run.run_v2_T2_locked as L
    res = {"launch_commit": a.launch_commit, "v1_1_protocol_sha256": L.PROTOCOL_SHA256}
    r1 = pd.DataFrame([r1_one(q, s) for q in QS for s in SEEDS])
    r1.to_csv(LK / "rehearsal_v1_1_R1_details.csv", index=False)
    res["R1"] = {"pass": bool(r1.ALL.all() and len(r1) == 20), "n_identical": int(r1.ALL.sum()), "n": len(r1),
                 "failing_fields": sorted({c for c in r1.columns if c not in ("q", "seed", "ALL") and not r1[c].all()}),
                 "details": str((LK / "rehearsal_v1_1_R1_details.csv").relative_to(ROOT))}
    r2, r3, r4 = [], [], []
    launcher = (V11 / "launcher.out").read_text() if (V11 / "launcher.out").exists() else ""
    for q in QS:
        for s in SEEDS:
            d = V11 / f"q{q}" / f"seed{s}"
            g, st, m = (json.load(open(d / f)) for f in ("gates.json", "status.json", "manifest.json"))
            r2.append(bool(g["G-A"]["pass"] and g["G-F"]["pass"] and g["G-N"]["pass"]))
            rc = re.search(rf"q{q} s{s} rc=(\d+)", launcher)
            r3.append(bool(g["global_rng"]["status"] == "ok" and not g["global_rng"]["violations"] and st.get("exit_code") == 0
                           and rc is not None and rc.group(1) == "0"))
            r4.append(bool(m["locked_protocol"]["sha256"] == L.PROTOCOL_SHA256 and str(m["locked_protocol"]["version"]) == "1.1"
                           and m["git"]["commit"].startswith(a.launch_commit) and m["clean_tree"] is True))
    res["R2"] = {"pass": all(r2) and len(r2) == 20, "n_pass": sum(r2), "n": len(r2)}
    res["R3"] = {"pass": all(r3) and len(r3) == 20, "n_ok": sum(r3), "n": len(r3)}
    res["R4"] = {"pass": all(r4) and len(r4) == 20, "n_ok": sum(r4), "n": len(r4)}
    an_out = LK / "rehearsal_v1_1_analysis"
    rr = subprocess.run([PY, str(ROOT / "tools" / "v2" / "confirmation_analysis.py"), "--root", str(V11), "--seeds", "10501", "10510",
                         "--out", str(an_out)],
                        capture_output=True, text=True, cwd=ROOT)
    ag = pd.read_csv(an_out / "agreement.csv") if rr.returncode == 0 else pd.DataFrame()
    res["R5"] = {"pass": bool(rr.returncode == 0 and len(ag) == 20 and ag.all_agree.all()), "returncode": rr.returncode,
                 "n_agree": int(ag.all_agree.sum()) if len(ag) else 0, "n": len(ag),
                 "max_abs_value_diff": float(ag.filter(like="absdiff_").to_numpy().max()) if len(ag) else None,
                 "output": str(an_out.relative_to(ROOT))}
    if a.skip_r6:
        res["R6"] = {"pass": None, "note": "skipped (dry check)"}
    else:
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        t = subprocess.run([PY, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"], capture_output=True, text=True, cwd=ROOT, env=env)
        (LK / "rehearsal_v1_1_pytest.txt").write_text(t.stdout[-20000:] + t.stderr[-5000:])
        failed = re.findall(r"^FAILED (\S+)", t.stdout, flags=re.M)
        summ = [ln for ln in t.stdout.splitlines() if re.search(r"\d+ (passed|failed)", ln)]
        tests_ok = failed == ["tests/test_registry_canonicalization.py::test_registry_canonicalization"]
        c7dir = ROOT / "results" / "v2_pilots" / "phase2_regression" / f"v2_full_{a.launch_commit[:7]}"
        if not c7dir.exists():
            subprocess.run([PY, "-B", "run/run_v2_stagewise.py", "--config", "results/v2_pilots/phase2_regression/run_config.json",
                            "--out-dir", str(c7dir)], cwd=ROOT, env=env, capture_output=True, text=True)
        c = subprocess.run([PY, "tools/v2/compare_runs.py", "results/v2_pilots/phase1_regression/before/FINAL_A400_B25_C25/tel_q50_s10501",
                            str(c7dir)], cwd=ROOT, capture_output=True, text=True)
        (c7dir.parent / f"v2_full_{a.launch_commit[:7]}.compare.txt").write_text(c.stdout)
        c7_ok = c.stdout.strip().endswith("IDENTICAL")
        man = json.load(open(c7dir / "manifest.json"))
        res["R6"] = {"pass": bool(tests_ok and c7_ok and man["git"]["commit"].startswith(a.launch_commit) and man["git"]["dirty"] is False),
                     "pytest_summary": summ[-1] if summ else None, "pytest_failed": failed, "c7_identical": c7_ok,
                     "c7_commit": man["git"]["short"], "c7_dirty": man["git"]["dirty"],
                     "pytest_log": str((LK / "rehearsal_v1_1_pytest.txt").relative_to(ROOT))}
    res["ALL_PASS"] = all(res[k]["pass"] for k in ("R1", "R2", "R3", "R4", "R5", "R6"))
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0 if res["ALL_PASS"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
