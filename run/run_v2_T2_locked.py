#!/usr/bin/env python3
"""Locked v2 T=2 pipeline: Phase A (1600) -> G-A -> freeze stage 2 -> Phase B (600) -> G-F.

The ONLY configuration source is ``protocols/v2_T2_locked.json``. The run refuses to start if
that file's SHA-256 differs from ``PROTOCOL_SHA256`` (the hash at the lock commit; also checked
against ``protocols/LOCK`` when that file exists) or if any command-line argument other than
``--q``, ``--seed`` and ``--out-dir`` is given.

Training is ``run.run_v2_stagewise.Run`` in mode ``locked`` (same loop, RNG streams and samplers
as every v2 pilot). Gates use the final verifier tier; the development tier is also reported.
The closed form enters only the evaluation (gates, reported metrics, decomposition).

Outputs in --out-dir: manifest.json (commit, clean-tree flag, protocol SHA-256, q, seed),
gates.json, v2_checkpoints_{A,B}.csv + checkpoints/, weights/, state_end_A.pt, state_end_B.pt,
gateA_{final,development}.npz, final_{final,development}.npz, induced_band.json + band_sweep.npz,
drift_test.json, train_history.json, v2_updates.csv, v2_run_summary.json, status.json.

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python run/run_v2_T2_locked.py --q 50 --seed 20501 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run.run_final_dp_br import write_json  # noqa: E402
from run.run_final_dp_br_round3_dense import ConfigError  # noqa: E402
from run.run_v2_stagewise import Run, git_state, write_manifest  # noqa: E402
from utils.dp_br_verifier import VerifierConfig  # noqa: E402
from utils.theory_multistage import f_xi, g1_two_stage, g2_two_stage  # noqa: E402
from utils.v2_metrics import evaluate, induced_band, save_npz, stage1_residual_sweep, sweep_grid  # noqa: E402

PROTOCOL_PATH = ROOT / "protocols" / "v2_T2_locked.json"
LOCK_PATH = ROOT / "protocols" / "LOCK"
PROTOCOL_SHA256 = "cf7b6929adfaef6b712eb01d1a731adc937d0b87fcc37a0a9e68eda9ae92d9f6"
ALLOWED_ARGS = ("--q", "--seed", "--out-dir")
SMOOTH_NODES = 400


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_protocol(path: Path = PROTOCOL_PATH, expected: str = PROTOCOL_SHA256) -> Dict:
    """Load the locked protocol; refuse on any hash mismatch (file vs code, file vs LOCK)."""
    actual = sha256_file(path)
    if actual != expected:
        raise ConfigError(f"protocol {path} sha256 {actual} != locked {expected}; refusing to run")
    if LOCK_PATH.exists():
        lock = json.load(open(LOCK_PATH))
        if lock.get("protocol_sha256") != actual:
            raise ConfigError(f"protocols/LOCK records protocol sha256 {lock.get('protocol_sha256')} != {actual}")
    return json.load(open(path))


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Accept exactly --q, --seed, --out-dir; refuse anything else."""
    p = argparse.ArgumentParser(description="locked v2 T=2 pipeline", allow_abbrev=False)
    p.add_argument("--q", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    a, extra = p.parse_known_args(argv)
    if extra:
        raise ConfigError(f"refusing overrides {extra}: only {ALLOWED_ARGS} are accepted")
    return a


def build_config(proto: Dict, q: int, seed: int, out_dir: str) -> Dict:
    """v2 run config for one (q, seed), taken entirely from the protocol."""
    if q not in proto["q_values"]:
        raise ConfigError(f"q={q} not in the protocol's q_values {proto['q_values']}")
    rec = copy.deepcopy(proto["records"][str(q)])
    run_name = f"v2T2locked_q{q}_s{seed}"
    rec.update(seed=int(seed), run=run_name, output_dir=out_dir)
    pl = proto["pipeline"]
    return {"schema": "v2_run_config/1", "base_commit": proto["base_commit"], "pilot": "v2_T2_locked", "arm": "locked",
            "run": run_name, "q": q, "seed": int(seed), "mode": pl["mode"], "fixed_budget": bool(pl["fixed_budget"]),
            "flags": dict(pl["flags"]), "parent_checkpoint": None, "parent_sha256": None, "record": rec,
            "threads_per_process": proto["threads_per_process"], "budget_overrides": {},
            "lr_decay": copy.deepcopy(pl["lr_decay"]), "full_state_at": []}


# ---------------------------------------------------------------------------- evaluation helpers
def smoothed_share(run: Run, ev, net=None) -> Dict[str, float]:
    """Share of the d=0 peak gap predicted by action-noise smoothing (Pilot-1 method)."""
    from scipy.stats import beta as beta_dist
    from run.run_final_dp_br import make_policy_fns
    spec = run.spec
    mf, bf = make_policy_fns(run.agent, spec, net=net)
    a, b = bf(2, np.zeros(1))
    u = (np.arange(SMOOTH_NODES) + 0.5) / SMOOTH_NODES
    x = spec.e_range * (beta_dist.ppf(u, a[0], b[0]) - a[0] / (a[0] + b[0]))
    pred0 = spec.dw / (2 * spec.k) * f_xi(x[:, None] - x[None, :], spec.q).mean()
    g20 = float(g2_two_stage(np.zeros(1), spec.q, spec.w_h, spec.w_l, spec.k, spec.e_max)[0])
    e0 = float(mf(2, np.zeros(1))[0])
    return {"smoothed_e_pred_0": float(pred0), "smoothed_e_learned_0": e0,
            "smoothed_share_peak_gap_d0": float((g20 - pred0) / (g20 - e0)) if g20 != e0 else float("nan")}


def stage2_extra(ev) -> Dict[str, float]:
    """Location-free peak error on the recovery grid."""
    D, e2 = ev.arrays["recovery_d_grid"], ev.arrays["recovery_e2"]
    g20 = float(ev.scalars["g2_at_0"])
    j = int(np.argmax(e2))
    return {"stage2_peak_locfree_rel_err": (float(e2[j]) - g20) / g20, "stage2_peak_locfree_argmax_d": float(D[j])}


def decomposition(run: Run, e1: float, proto: Dict, fin: VerifierConfig, step: Optional[float] = None) -> Dict[str, object]:
    """Residual-band decomposition of the stage-1 error against the frozen stage 2 (final tier)."""
    spec = run.spec
    ds = proto["stage1_decomposition"]
    step = float(ds["step"]) if step is None else step
    g1 = g1_two_stage(spec.q, spec.w_h, spec.w_l, spec.k)

    def eq(t, d):
        d = np.asarray(d, dtype=float)
        return np.full(d.shape, g1) if t == 1 else g2_two_stage(d, spec.q, spec.w_h, spec.w_l, spec.k, spec.e_max)
    cal = float(stage1_residual_sweep(eq, spec, fin, np.array([g1]))[0])
    floor = max(cal, 1e-12 * spec.dw)
    mean_fn, _ = run.policy_fns()
    lo, hi = min(0.4 * g1, e1 - 2.0), max(1.7 * g1, e1 + 2.0)
    E = sweep_grid(g1, lo, hi, step)
    D = stage1_residual_sweep(mean_fn, spec, fin, E)
    np.savez(os.path.join(run.out_dir, "band_sweep.npz"), e_sweep=E, delta1=D)
    b = induced_band(E, D, floor)
    et, blo, bhi = b["e_tilde"], b["band_lo"], b["band_hi"]
    out = {**b, "g1": g1, "e1_at_0": e1, "calibration_delta1": cal, "step": step,
           "learning_rel": (e1 - et) / g1, "learning_rel_lo": (e1 - bhi) / g1, "learning_rel_hi": (e1 - blo) / g1,
           "inherited_rel": (et - g1) / g1, "inherited_rel_lo": (blo - g1) / g1, "inherited_rel_hi": (bhi - g1) / g1,
           "e1_inside_sweep": bool(E[0] <= e1 <= E[-1])}
    out["learning_contains_0"] = bool(out["learning_rel_lo"] <= 0 <= out["learning_rel_hi"])
    out["inherited_contains_0"] = bool(out["inherited_rel_lo"] <= 0 <= out["inherited_rel_hi"])
    return out


def gate_block(name: str, crit: List[Dict], sc_fin: Dict, sc_dev: Dict) -> Dict[str, object]:
    """Gate verdict from the final-tier scalars (dev tier reported alongside)."""
    rows = []
    for c in crit:
        m = c["metric"]
        vf, vd = float(sc_fin[m]), float(sc_dev[m])
        rows.append({"metric": m, "threshold": c["threshold"], "value_final": vf, "value_dev": vd,
                     "dev_minus_final": vd - vf, "pass_final": bool(vf <= c["threshold"]),
                     "pass_dev": bool(vd <= c["threshold"])})
    return {"gate": name, "criteria": rows, "pass": all(r["pass_final"] for r in rows),
            "pass_dev_tier": all(r["pass_dev"] for r in rows)}


REPORT_KEYS_A = ("stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_sym_err_max", "stage2_tail_max",
                 "stage2_tail_max_over_g2_0", "stage2_tail_mean", "DeltaT_over_dw_on_max", "DeltaT_over_dw_off_max",
                 "DeltaT_over_dw_on_mean_cellmass_weighted", "sigma_effort_at_0_t2", "e2_at_0", "g2_at_0",
                 "stage2_rmse_pos_over_g2_0", "stage2_tail_mean_over_g2_0", "eta_T_over_dw", "valid")
REPORT_KEYS_F = ("Gmax_full_over_dw", "Gmax_full_t", "Gmax_full_d", "EXP_root_over_dw", "dReach_over_dw",
                 "Deltamax_all_over_dw", "dFull_over_dw", "stage1_rel_err_signed", "stage1_rel_err_abs", "e1_at_0", "g1",
                 "sigma_effort_at_0_t1", "eta_T_over_dw", "valid")


def pick(sc: Dict, keys) -> Dict[str, object]:
    return {k: sc[k] for k in keys if k in sc}


def diff(dev: Dict, fin: Dict) -> Dict[str, float]:
    return {k: float(dev[k]) - float(fin[k]) for k in fin
            if k in dev and isinstance(fin[k], (int, float)) and not isinstance(fin[k], bool)}


# ---------------------------------------------------------------------------- pipeline
def execute_locked(run: Run, cfg: Dict, proto: Dict, proto_sha: str, out_dir: str, cmd: str,
                   band_step: Optional[float] = None) -> int:
    """Phase A -> G-A -> freeze -> Phase B -> G-F -> final evaluation (one process)."""
    t0 = time.perf_counter()
    status = {"run": cfg["run"], "q": run.spec.q, "seed": run.seed, "mode": run.mode, "state": "running",
              "pid": os.getpid(), "cmd": cmd, "start_time": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_state()}
    write_json(os.path.join(out_dir, "status.json"), status)
    man = write_manifest(run, cfg, out_dir, cmd)
    g = man["git"]
    man.update({"locked_protocol": {"path": os.path.relpath(PROTOCOL_PATH, ROOT), "sha256": proto_sha,
                                    "version": proto["version"]},
                "commit": g["commit"], "clean_tree": g["dirty"] is False, "q": run.spec.q, "seed": run.seed})
    write_json(os.path.join(out_dir, "manifest.json"), man)
    gcfg = proto["gates"]
    try:
        # ---- Phase A
        run.run_phase("A")
        torch.save(run.full_state("A"), os.path.join(out_dir, "state_end_A.pt"))
        evA = {}
        for tier in (run.fin_cfg, run.dev_cfg):
            mf, bf = run.policy_fns()
            ev = evaluate(mf, run.spec, tier, beta_fn=bf, recovery_step=float(run.P["recovery_step"]))
            save_npz(ev, os.path.join(out_dir, f"gateA_{tier.name}.npz"))
            evA[tier.name] = ev
        scA = {t: {**e.scalars, **stage2_extra(e)} for t, e in evA.items()}
        GA = gate_block("G-A", gcfg["G-A"]["all_must_hold"], scA["final"], scA["development"])
        smooth = smoothed_share(run, evA["final"])
        # ---- freeze stage 2 (B2 machinery), as in run_v2_stagewise.execute for mode phase_B
        run.parent_ref = run.stage2_mapping()
        parent_actor = {k: v.clone() for k, v in run.agent.actor.state_dict().items()}
        run.agent.freeze_stage2_snapshot()
        # ---- Phase B
        run.run_phase("B")
        torch.save(run.full_state("B"), os.path.join(out_dir, "state_end_B.pt"))
        run.agent.export_weights_npz(os.path.join(out_dir, "checkpoint_weights.npz"))
        evF = {}
        for tier in (run.fin_cfg, run.dev_cfg):
            mf, bf = run.policy_fns()
            ev = evaluate(mf, run.spec, tier, beta_fn=bf, recovery_step=float(run.P["recovery_step"]))
            save_npz(ev, os.path.join(out_dir, f"final_{tier.name}.npz"))
            evF[tier.name] = ev
        scF = {t: {**e.scalars, **stage2_extra(e)} for t, e in evF.items()}
        GF = gate_block("G-F", gcfg["G-F"]["all_must_hold"], scF["final"], scF["development"])
        # ---- snapshot integrity (C5 drift test)
        now = run.stage2_mapping(run.agent.frozen)
        dt_ = {k: float(np.max(np.abs(now[k] - run.parent_ref[k]))) for k in ("mean", "alpha", "beta")}
        fz = run.agent.frozen.state_dict()
        drift = {"max_abs_diff_vs_freeze_time": dt_,
                 "snapshot_params_bit_identical_to_end_of_A_actor": bool(all(torch.equal(fz[k], parent_actor[k]) for k in parent_actor))}
        drift["pass"] = bool(all(v == 0.0 for v in dt_.values()) and drift["snapshot_params_bit_identical_to_end_of_A_actor"])
        write_json(os.path.join(out_dir, "drift_test.json"), drift)
        # ---- stage-1 decomposition (final tier, frozen stage 2)
        dec = decomposition(run, float(scF["final"]["e1_at_0"]), proto, run.fin_cfg, band_step)
        write_json(os.path.join(out_dir, "induced_band.json"), dec)
        run_pass = bool(GA["pass"] and GF["pass"])
        gates = {"q": run.spec.q, "seed": run.seed, "protocol_sha256": proto_sha, "commit": g["commit"],
                 "clean_tree": g["dirty"] is False, "G-A": GA, "G-F": GF, "run_pass": run_pass,
                 "outcome": "pass" if run_pass else ("stage2_failure" if not GA["pass"] else "fail_G-F"),
                 "reported": {
                     "end_of_A": {"final": pick(scA["final"], REPORT_KEYS_A + ("stage2_peak_locfree_rel_err", "stage2_peak_locfree_argmax_d")),
                                  "development": pick(scA["development"], REPORT_KEYS_A + ("stage2_peak_locfree_rel_err", "stage2_peak_locfree_argmax_d")),
                                  "dev_minus_final": diff(pick(scA["development"], REPORT_KEYS_A + ("stage2_peak_locfree_rel_err",)),
                                                          pick(scA["final"], REPORT_KEYS_A + ("stage2_peak_locfree_rel_err",))),
                                  "smoothed_game": smooth},
                     "end_of_B": {"final": pick(scF["final"], REPORT_KEYS_F), "development": pick(scF["development"], REPORT_KEYS_F),
                                  "dev_minus_final": diff(pick(scF["development"], REPORT_KEYS_F), pick(scF["final"], REPORT_KEYS_F)),
                                  "decomposition": dec},
                     "drift_test_pass": drift["pass"]},
                 "lr_last": {"A": run.lr_for("A", int(run.P["phase_caps"]["A"])), "B": run.lr_for("B", int(run.P["phase_caps"]["B"]))}}
        write_json(os.path.join(out_dir, "gates.json"), gates)
        write_json(os.path.join(out_dir, "train_history.json"),
                   {"run": cfg["run"], "group": cfg["arm"], "history": run.history, "stability": run.stability_log,
                    "verifier_calls": run.verifier_log, "curriculum": run.curriculum_log, "snapshots": run.snapshot_log,
                    "weight_checkpoints": run.weights_log, "stopping_record": None})
        write_json(os.path.join(out_dir, "v2_run_summary.json"),
                   {"phase_timing": run.phase_timing, "would_have_fired": run.would_fire, "fixed_budget": run.fixed,
                    "costs": run.costs.as_dict(), "costs_v2": run.costs_v2.as_dict(), "phases_done": run.phases_done,
                    "final_global_update": run.global_u, "total_wall_sec": time.perf_counter() - t0})
        with open(os.path.join(out_dir, "v2_updates.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(run.v2_history[0].keys()))
            w.writeheader()
            w.writerows(run.v2_history)
        status.update({"state": "done", "end_time": time.strftime("%Y-%m-%d %H:%M:%S"), "exit_code": 0,
                       "final_global_update": run.global_u, "run_pass": run_pass, "outcome": gates["outcome"],
                       "total_wall_sec": time.perf_counter() - t0})
        write_json(os.path.join(out_dir, "status.json"), status)
        print(f"[done] q={run.spec.q:g} seed={run.seed} G-A={GA['pass']} G-F={GF['pass']} outcome={gates['outcome']} "
              f"wall={time.perf_counter() - t0:.1f}s", flush=True)
        return 0
    except Exception:
        status.update({"state": "failed", "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "traceback": traceback.format_exc(), "updates_completed": run.global_u, "exit_code": 1})
        write_json(os.path.join(out_dir, "status.json"), status)
        print(traceback.format_exc(), flush=True)
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry (refuses any override other than q, seed, out-dir)."""
    a = parse_args(sys.argv[1:] if argv is None else argv)
    proto = load_protocol()
    out = a.out_dir
    if os.path.exists(os.path.join(out, "status.json")):
        raise ConfigError(f"{out} already holds status.json; not restarting")
    os.makedirs(out, exist_ok=True)
    cfg = build_config(proto, a.q, a.seed, out)
    json.dump(cfg, open(os.path.join(out, "run_config.json"), "w"), indent=1)
    run = Run(cfg, out)
    return execute_locked(run, cfg, proto, sha256_file(PROTOCOL_PATH), out, " ".join(sys.argv))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as exc:
        print(f"[config-error] {exc}", flush=True)
        sys.exit(4)
