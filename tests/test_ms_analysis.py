"""MS-R1 analysis tool tests (``tools/ms/r1_analysis.py`` and the independent ``tools/ms/blind_criterion.py``).

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_analysis.py -p no:cacheprovider -q

Fixtures. ``genuine`` runs two tiny T = 2 pipelines of the real runner (``MS_rule`` and the legacy ``MS_base``,
about 30 s) and the synthetic roots are built by copying and perturbing the genuine JSON / CSV files (so the
formats are the runner's); the synthetic ``parents_A`` / ``rehearsal_v2_0`` roots carry the key sets of the real
files (``final_v2.json`` / ``gates.json`` of the v2.0 reference runs). Sections: reader on genuine output, the
per-run columns on hand-computed cases, status handling (failed / missing / running / incomplete are rows),
bootstrap determinism, paired tables, criterion parts (a) and (b) on constructed cases, the blind recomputation
(agreement and failure on tampering), the CLI end to end, and a read-only smoke test on real reference runs.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import blind_criterion as BL  # noqa: E402
import ms_configs as mc  # noqa: E402
import r1_analysis as R  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402
from run.run_v2_T2_locked import verdicts  # noqa: E402

PROTO_PATH = ROOT / "protocols" / "v2_T2_locked_v2_0.json"
PROTO = json.loads(PROTO_PATH.read_text())
TH, _ = R.load_thresholds(PROTO_PATH)
SEEDS4 = (10501, 10502, 10503, 10504)
QS = (50, 60)
NB, LAND = 10, 5                      # synthetic block / landing lengths (updates)
EPU = 100                             # synthetic starts per update

# key sets of the real v2.0 reference files (parents_A final_v2.json tiers, rehearsal gates.json reported.*)
PARENT_KEYS = (
    "verifier_tier", "state_step", "effort_step", "gl_half", "valid", "invalid_reasons", "G_max_t1_over_dw",
    "G_argmax_d_t1", "Delta_max_t1_over_dw", "Delta_argmax_d_t1", "G_max_t2_over_dw", "G_argmax_d_t2",
    "Delta_max_t2_over_dw", "Delta_argmax_d_t2", "Gmax_full_over_dw", "Gmax_full_t", "Gmax_full_d",
    "EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw", "dFull_over_dw", "eta_T_over_dw",
    "pdl_residual_over_dw", "inv_GT_eq_DeltaT_absdiff_over_dw", "inv_GT_eq_DeltaT_at_d",
    "inv_Delta_le_G_pointwise_excess_over_dw", "inv_Delta_le_G_pointwise_at_t", "inv_Delta_le_G_pointwise_at_d",
    "inv_Deltamax_le_Gmax_excess_over_dw", "inv_Gmax_le_dFull_excess_over_dw", "inv_EXP_eq_G1_absdiff_over_dw",
    "inv_EXP_le_dReach_excess_over_dw", "inv_dReach_le_dFull_excess_over_dw",
    "inv_EXP_le_dReachPMF_excess_over_dw", "DeltaT_over_dw_n_on", "DeltaT_over_dw_n_off", "DeltaT_over_dw_on_mass",
    "DeltaT_over_dw_off_mass", "DeltaT_over_dw_on_max", "DeltaT_over_dw_on_argmax_d",
    "DeltaT_over_dw_on_mean_unweighted", "DeltaT_over_dw_off_max", "DeltaT_over_dw_off_argmax_d",
    "DeltaT_over_dw_off_mean_unweighted", "DeltaT_over_dw_on_mean_cellmass_weighted", "stage1_drift",
    "stage1_drift_is_zero", "cellmass_captured_total", "node_pmf_n_positive_T", "sigma_effort_at_0_t2",
    "sigma_effort_at_0_t1", "sigma2_effort_mean_pos", "g1", "g2_at_0", "e1_at_0", "e2_at_0",
    "stage1_rel_err_signed", "stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_rmse_pos",
    "stage2_rmse_pos_over_g2_0", "stage2_tail_mean", "stage2_tail_max", "stage2_tail_argmax_d",
    "stage2_tail_mean_over_g2_0", "stage2_tail_max_over_g2_0", "stage2_sym_err_max",
    "stage2_sym_err_max_over_g2_0", "stage2_sym_err_argmax_d", "recovery_grid_step", "recovery_n_pos",
    "recovery_n_tail", "stage1_rel_err_abs")
REH_A_KEYS = (
    "stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_sym_err_max", "stage2_tail_max",
    "stage2_tail_max_over_g2_0", "stage2_tail_mean", "DeltaT_over_dw_on_max", "DeltaT_over_dw_off_max",
    "DeltaT_over_dw_on_mean_cellmass_weighted", "sigma_effort_at_0_t2", "e2_at_0", "g2_at_0",
    "stage2_rmse_pos_over_g2_0", "stage2_tail_mean_over_g2_0", "eta_T_over_dw", "valid",
    "stage2_peak_locfree_rel_err", "stage2_peak_locfree_argmax_d")
REH_B_KEYS = (
    "Gmax_full_over_dw", "Gmax_full_t", "Gmax_full_d", "EXP_root_over_dw", "dReach_over_dw",
    "Deltamax_all_over_dw", "dFull_over_dw", "stage1_rel_err_signed", "stage1_rel_err_abs", "e1_at_0", "g1",
    "sigma_effort_at_0_t1", "eta_T_over_dw", "valid")
BAND_KEYS = (
    "e_tilde", "delta1_min", "band_lo", "band_hi", "band_n_points", "band_contiguous", "floor", "sweep_lo",
    "sweep_hi", "argmin_at_sweep_edge", "g1", "e1_at_0", "calibration_delta1", "step", "learning_rel",
    "learning_rel_lo", "learning_rel_hi", "inherited_rel", "inherited_rel_lo", "inherited_rel_hi",
    "e1_inside_sweep", "learning_contains_0", "inherited_contains_0")


# ====================================================================== the synthetic world
@dataclass
class Spec:
    """Target values of one synthetic run."""

    peak: float = -0.04               # signed peak error (final tier)
    rmse: float = 0.02
    tail: float = 0.005
    tail_max: float = 0.03
    eta: float = 0.003
    eta_dev: Optional[float] = None   # default eta + 1e-4
    sigma: float = 3.0
    smoothed: float = 0.30
    s1: float = 0.01                  # signed stage-1 error
    gmax: float = 0.003
    gmax_dev: Optional[float] = None
    learning: float = 0.004
    inherited: float = 0.006
    fire: bool = False                # the last block ends with a development stop
    forced: bool = False              # the last block ends at the cap (budget_forced)
    blocks: Tuple[Tuple[str, Optional[str], int, Tuple[int, int, int]], ...] = (
        ("global", None, 0, (50, 10, 40)),)
    landing_shares: Tuple[int, int, int] = (50, 10, 40)
    status: str = "done"              # done | failed | running | rng | nofiles | missing
    R: float = 0.02                   # R_t at the terminal freeze (final tier)
    R_dev: Optional[float] = None


def _pad(d: Dict[str, Any], keys: Tuple[str, ...]) -> Dict[str, Any]:
    out = {k: 0.0 for k in keys}
    out.update(d)
    return out


def terminal_scalars(sp: Spec, tier: str) -> Dict[str, Any]:
    """The terminal-stage scalars of one verifier tier (the keys of the runner's ``reported.end_of_stage2``)."""
    eta = sp.eta if tier == "final" else (sp.eta + 1e-4 if sp.eta_dev is None else sp.eta_dev)
    return {"stage2_peak_rel_err_signed": sp.peak, "stage2_peak_rel_err_abs": abs(sp.peak),
            "stage2_rmse_pos_over_g2_0": sp.rmse, "stage2_tail_mean_over_g2_0": sp.tail,
            "stage2_tail_max_over_g2_0": sp.tail_max, "sigma_effort_at_0_t2": sp.sigma, "e2_at_0": 66.0,
            "g2_at_0": 70.0, "eta_T_over_dw": eta, "valid": True}


def stage1_scalars(sp: Spec, tier: str) -> Dict[str, Any]:
    g = sp.gmax if tier == "final" else (sp.gmax + 1e-4 if sp.gmax_dev is None else sp.gmax_dev)
    return {"Gmax_full_over_dw": g, "stage1_rel_err_signed": sp.s1, "stage1_rel_err_abs": abs(sp.s1),
            "e1_at_0": 47.0, "g1": 46.666666666666664, "sigma_effort_at_0_t1": 3.3, "eta_T_over_dw": sp.eta,
            "valid": True}


def load_template(d: Path) -> Dict[str, Any]:
    """The genuine files that the synthetic runs copy."""
    t: Dict[str, Any] = {}
    for fn in ("status.json", "gates.json", "rule_log.json", "manifest.json", "run_config.json",
               "induced_band.json", "ms_run_summary.json"):
        t[fn] = json.loads((d / fn).read_text())
    t["checks_header"] = list(pd.read_csv(d / "ms_checks_stage2.csv", nrows=0).columns)
    t["updates_header"] = list(pd.read_csv(d / "ms_updates.csv", nrows=0).columns)
    return t


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=1))


def write_ms_run(run_dir: Path, tm: Dict[str, Any], arm: str, q: int, seed: int, sp: Spec) -> None:
    """One synthetic MS run directory in the runner's formats (copied from the genuine template)."""
    legacy = arm == "MS_base"
    run_dir.mkdir(parents=True, exist_ok=True)
    name = "msr1_q%d_s%d_%s" % (q, seed, arm)
    st = copy.deepcopy(tm["status.json"])
    st.update(run=name, arm=arm, q=q, seed=seed)
    if sp.status == "running":
        for k in ("end_time", "exit_code", "final_global_update", "total_wall_sec", "outcome"):
            st.pop(k, None)
        st["state"] = "running"
    elif sp.status == "failed":
        st.update(state="failed", exit_code=1, traceback="Traceback (most recent call last):\nRuntimeError: boom",
                  updates_completed=20)
        for k in ("final_global_update", "total_wall_sec", "outcome"):
            st.pop(k, None)
    elif sp.status == "rng":
        st["exit_code"] = 5
    _dump(run_dir / "status.json", st)
    cfg = copy.deepcopy(tm["run_config.json"])
    cfg.update(run=name, arm=arm, q=q, seed=seed)
    _dump(run_dir / "run_config.json", cfg)
    if sp.status == "running":
        return
    _dump(run_dir / "manifest.json", copy.deepcopy(tm["manifest.json"]))
    # ---- rule log, stage 2 (and stage 1 unless the run failed)
    rl = copy.deepcopy(tm["rule_log.json"])
    rl.update(arm=arm, run=name, q=q, seed=seed)
    s2 = rl["stages"]["2"]
    blocks: List[Dict[str, Any]] = []
    if legacy:
        n2 = NB * len(sp.blocks) + LAND
        blocks = [{"block_id": 1, "type": "legacy", "first_local": 1, "last_local": n2,
                   "exit_reason": "fixed_budget"}]
        s2.update(blocks=blocks, landing=None, fire_local=None, would_fire_local=None, budget_forced=False,
                  training_updates=None, total_updates=n2)
    else:
        first = 1
        for i, (typ, cls, n_s, _sh) in enumerate(sp.blocks, start=1):
            last = first + NB - 1
            is_last = i == len(sp.blocks)
            reason = "development_stop" if (is_last and sp.fire) else ("cap" if (is_last and sp.forced)
                                                                     else "block_end")
            blocks.append({"block_id": i, "type": typ, "first_local": first, "last_local": last,
                           "exit_reason": reason, "fire_local": last if (is_last and sp.fire) else None,
                           "S": list(range(n_s)), "n_S": n_s, "classification": cls})
            first = last + 1
        last_t = blocks[-1]
        landing = {"first_local": last_t["last_local"] + 1, "last_local": last_t["last_local"] + LAND,
                   "n_land": LAND, "followed_block_id": last_t["block_id"], "followed_block_type": last_t["type"],
                   "alpha": 0.0, "budget_forced": bool(sp.forced), "done": True}
        n2 = last_t["last_local"] + LAND
        s2.update(blocks=blocks, landing=landing, fire_local=last_t["last_local"] if sp.fire else None,
                  would_fire_local=None, budget_forced=bool(sp.forced), training_updates=last_t["last_local"],
                  total_updates=n2)
    s2.update(episodes=n2 * EPU, transitions=n2 * EPU, minibatch_steps=n2 * 10, wall_sec=1.5 + 0.01 * n2,
              n_checks=n2 // 5, entry_update=0, exit_update=n2)
    fz = s2["freeze"]
    fz["final"].update(terminal_scalars(sp, "final"))
    fz["development"].update(terminal_scalars(sp, "development"))
    fz["d3_final"].update(Delta=sp.eta, R=sp.R, R_tail=0.01, s=69.0, C=0.03, tail_term=True, R_argmax_d=-8.0)
    fz["d3_development"].update(Delta=terminal_scalars(sp, "development")["eta_T_over_dw"],
                                R=sp.R if sp.R_dev is None else sp.R_dev, R_tail=0.011, s=69.1, C=0.03,
                                tail_term=True, R_argmax_d=-8.0)
    fz["smoothed_game"] = {"smoothed_e_pred_0": 67.0, "smoothed_e_learned_0": 66.0,
                           "smoothed_share_peak_gap_d0": sp.smoothed}
    n_bins = len(fz["rho_bins_final"])
    rho = [None if i < 10 or i >= n_bins - 10 else 0.01 * (i % 5) for i in range(n_bins)]
    fz["rho_bins_final"], fz["rho_bins_development"] = rho, rho
    n1 = 6
    s1 = rl["stages"]["1"]
    s1.update(episodes=n1 * EPU, transitions=n1 * EPU, minibatch_steps=n1 * 10, total_updates=n1,
              wall_sec=0.7, n_checks=2, exit_update=n2 + n1, entry_update=n2)
    s1["freeze"]["final"].update(stage1_scalars(sp, "final"))
    s1["freeze"]["development"].update(stage1_scalars(sp, "development"))
    s1["freeze"]["d3_final"].update(Delta=1e-4, R=0.02, s=46.5, C=0.05)
    s1["freeze"]["d3_development"].update(Delta=1.1e-4, R=0.021, s=46.4, C=0.05)
    if sp.status == "failed":
        del rl["stages"]["1"]
    if sp.status != "nofiles":
        _dump(run_dir / "rule_log.json", rl)
    # ---- checks and updates
    chk = []
    n_chk = n2 // 5
    for k in range(1, n_chk + 1):
        row = {c: float("nan") for c in tm["checks_header"]}
        frac = k / n_chk
        row.update(run=name, arm=arm, q=q, seed=seed, stage=2, update=5 * k, local=5 * k,
                   mode="land" if (not legacy and 5 * k > n2 - LAND) else "train", block_id=1,
                   block_type="legacy" if legacy else "global", lr=3e-4, alpha=0.0, valid=True, Delta=sp.eta,
                   R=sp.R, stage2_peak_rel_err_signed=-0.3 + (0.3 + sp.peak) * frac, eligible=False,
                   consecutive=0)
        chk.append(row)
    pd.DataFrame(chk, columns=tm["checks_header"]).to_csv(run_dir / "ms_checks_stage2.csv", index=False)
    ups = []
    for loc in range(1, n2 + 1):
        row = {c: float("nan") for c in tm["updates_header"]}
        if legacy:
            bid, btype, shares = 1, "legacy", sp.blocks[0][3]
        elif loc > n2 - LAND:
            bid, btype, shares = len(sp.blocks) + 1, "landing", sp.landing_shares
        else:
            bi = (loc - 1) // NB
            bid, btype, shares = bi + 1, sp.blocks[bi][0], sp.blocks[bi][3]
        row.update(update=loc, stage=2, local=loc, block_id=bid, block_type=btype, lr=3e-4, alpha=0.0,
                   n_start_tail=shares[0], n_start_near=shares[1], n_start_mid=shares[2],
                   d1_L_s1_n=0, d1_L_s1_lo=0, d1_L_s1_hi=0, d1_O_s1_n=0, d1_O_s1_lo=0, d1_O_s1_hi=0,
                   d1_L_s2_n=EPU, d1_L_s2_lo=1, d1_L_s2_hi=2, d1_O_s2_n=EPU, d1_O_s2_lo=0, d1_O_s2_hi=1,
                   d1_L_s2_in_n=50, d1_L_s2_in_lo=1, d1_L_s2_in_hi=0, d1_L_s2_out_n=50, d1_L_s2_out_lo=0,
                   d1_L_s2_out_hi=2)
        ups.append(row)
    if sp.status != "failed":
        for loc in range(1, n1 + 1):
            row = {c: float("nan") for c in tm["updates_header"]}
            row.update(update=n2 + loc, stage=1, local=loc, block_id=1, block_type="legacy" if legacy else "global",
                       n_start_tail=0, n_start_near=0, n_start_mid=0, d1_L_s1_n=EPU, d1_L_s1_lo=3, d1_L_s1_hi=0,
                       d1_O_s1_n=EPU, d1_O_s1_lo=0, d1_O_s1_hi=0)
            ups.append(row)
    pd.DataFrame(ups, columns=tm["updates_header"]).to_csv(run_dir / "ms_updates.csv", index=False)
    np.savez(run_dir / "ms_binmaps_stage2.npz", update=np.arange(n_chk), local=np.arange(n_chk),
             rho=np.full((n_chk, n_bins), 0.01), rho_bar=np.full((n_chk, n_bins), 0.02),
             probs=np.full((n_chk, n_bins), 1.0 / n_bins))
    if sp.status == "failed":
        return
    # ---- gates, induced band, summary
    vals = {"eta_final": sp.eta, "eta_dev": terminal_scalars(sp, "development")["eta_T_over_dw"],
            "rmse": sp.rmse, "tail": sp.tail, "gmax_final": sp.gmax,
            "gmax_dev": stage1_scalars(sp, "development")["Gmax_full_over_dw"], "s1": abs(sp.s1)}
    V = verdicts(vals, PROTO)
    g = copy.deepcopy(tm["gates.json"])
    g.update(arm=arm, run=name, q=q, seed=seed, metric_values=vals, outcome=V["outcome"],
             v20_combination_pass=V["run_pass"])
    for k in ("G-A", "G-F", "G-N", "G-S", "S1", "v1_1_outcome", "v1_0_outcome"):
        g[k] = V[k]
    g["reported"]["end_of_stage2"]["final"] = _pad(terminal_scalars(sp, "final"), tuple(
        g["reported"]["end_of_stage2"]["final"]))
    g["reported"]["end_of_stage2"]["development"] = _pad(terminal_scalars(sp, "development"), tuple(
        g["reported"]["end_of_stage2"]["development"]))
    g["reported"]["end_of_stage2"]["smoothed_game"] = fz["smoothed_game"]
    g["reported"]["end_of_stage1"]["final"].update(stage1_scalars(sp, "final"))
    g["reported"]["end_of_stage1"]["development"].update(stage1_scalars(sp, "development"))
    g["budget_totals"] = {"total_updates": n2 + n1, "total_episodes": (n2 + n1) * EPU,
                          "total_transitions": (n2 + n1) * EPU, "minibatch_steps": (n2 + n1) * 10}
    g["global_rng"] = {"status": "violation" if sp.status == "rng" else "ok",
                       "violations": [{"rng": "numpy_global", "point": "end_of_run"}] if sp.status == "rng" else [],
                       "assertion_points": []}
    if sp.status != "nofiles":
        _dump(run_dir / "gates.json", g)
    ib = copy.deepcopy(tm["induced_band.json"])
    ib.update(learning_rel=sp.learning, learning_rel_lo=sp.learning - 0.002, learning_rel_hi=sp.learning,
              inherited_rel=sp.inherited, inherited_rel_lo=sp.inherited, inherited_rel_hi=sp.inherited + 0.004,
              learning_contains_0=False, inherited_contains_0=False)
    _dump(run_dir / "induced_band.json", ib)
    sm = copy.deepcopy(tm["ms_run_summary.json"])
    sm["phase_timing"]["stage2"].update(process_cpu_sec=1.4, updates=n2)
    sm["phase_timing"]["stage1"].update(process_cpu_sec=0.6, updates=n1)
    sm["total_episodes"] = (n2 + n1) * EPU
    _dump(run_dir / "ms_run_summary.json", sm)


def write_parents_run(run_dir: Path, q: int, seed: int, sp: Spec, status: str = "done") -> None:
    """One synthetic ``parents_A`` run (``phase_A`` outputs: final_v2.json with both tiers)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    st = {"run": "v2refine_parents_A_q%d_s%d_A_parent" % (q, seed), "pilot": "v2_refine_parents_A", "arm": "A_parent",
          "q": q, "seed": seed, "mode": "phase_A", "state": "done", "pid": 1, "host": "x", "cmd": "x",
          "start_time": "2026-10-03 19:24:11", "git": {"commit": "655b14c", "short": "655b14c", "dirty": False},
          "end_time": "2026-10-03 19:28:06", "total_wall_sec": 233.97, "exit_code": 0, "final_global_update": 1600}
    if status == "failed":
        st.update(state="failed", exit_code=1)
    _dump(run_dir / "status.json", st)
    if status == "missing_files":
        return
    fin = _pad({**terminal_scalars(sp, "final"), "verifier_tier": "final"}, PARENT_KEYS)
    dev = _pad({**terminal_scalars(sp, "development"), "verifier_tier": "development"}, PARENT_KEYS)
    _dump(run_dir / "final_v2.json", {"stage1_status": "stage1_untrained", "development": dev,
                                      "drift_vs_parent": {}, "final": fin})
    _dump(run_dir / "v2_run_summary.json", {
        "phase_timing": {"A": {"wall_sec": 233.19, "process_cpu_sec": 233.03, "updates": 1600, "global_entry": 0,
                               "global_exit": 1600, "by_category_sec": {"dev_verifier": 0.24}}},
        "costs": {"total_episodes": 819200, "minibatch_steps": 32000, "total_updates": 1600},
        "fixed_budget": True, "final_global_update": 1600})
    _dump(run_dir / "run_config.json", {"record": {"protocol": {"episodes_per_update": 512}}})


def write_rehearsal_run(run_dir: Path, q: int, seed: int, sp: Spec) -> None:
    """One synthetic ``rehearsal_v2_0`` run (locked pipeline: gates.json with end_of_A / end_of_B)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    _dump(run_dir / "status.json", {"run": "v2T2locked_q%d_s%d" % (q, seed), "q": q, "seed": seed, "mode": "locked",
                                    "state": "done", "git": {"commit": "f2d616c", "short": "f2d616c", "dirty": False},
                                    "exit_code": 0, "final_global_update": 2200, "run_pass": True, "outcome": "pass",
                                    "total_wall_sec": 331.35})
    vals = {"eta_final": sp.eta, "eta_dev": terminal_scalars(sp, "development")["eta_T_over_dw"], "rmse": sp.rmse,
            "tail": sp.tail, "gmax_final": sp.gmax, "gmax_dev": stage1_scalars(sp, "development")["Gmax_full_over_dw"],
            "s1": abs(sp.s1)}
    V = verdicts(vals, PROTO)
    a_f, a_d = (_pad(terminal_scalars(sp, t), REH_A_KEYS) for t in ("final", "development"))
    b_f, b_d = (_pad(stage1_scalars(sp, t), REH_B_KEYS) for t in ("final", "development"))
    dec = _pad({"learning_rel": sp.learning, "learning_rel_lo": sp.learning - 0.002, "learning_rel_hi": sp.learning,
                "inherited_rel": sp.inherited, "inherited_rel_lo": sp.inherited,
                "inherited_rel_hi": sp.inherited + 0.004,
                "band_contiguous": False, "e1_inside_sweep": True, "learning_contains_0": False,
                "inherited_contains_0": False, "e_tilde": 47.0, "band_lo": 47.0, "band_hi": 47.2}, BAND_KEYS)
    g = {"protocol_version": "2.0", "q": q, "seed": seed, "metric_values": vals, "G-A": V["G-A"], "G-F": V["G-F"],
         "G-N": V["G-N"], "G-S": V["G-S"], "S1": V["S1"], "v1_1_outcome": V["v1_1_outcome"],
         "v1_0_outcome": V["v1_0_outcome"], "global_rng": {"status": "ok", "violations": []},
         "gates_pass": V["run_pass"], "run_pass": V["run_pass"], "outcome": V["outcome"],
         "reported": {"end_of_A": {"final": a_f, "development": a_d, "dev_minus_final": a_f,
                                   "smoothed_game": {"smoothed_e_pred_0": 67.3, "smoothed_e_learned_0": 61.4,
                                                     "smoothed_share_peak_gap_d0": sp.smoothed}},
                      "end_of_B": {"final": b_f, "development": b_d, "dev_minus_final": b_f, "decomposition": dec},
                      "drift_test_pass": True}, "lr_last": {"A": 3e-5, "B": 3e-5}}
    _dump(run_dir / "gates.json", g)
    _dump(run_dir / "induced_band.json", dec)
    _dump(run_dir / "v2_run_summary.json", {
        "phase_timing": {"A": {"wall_sec": 148.3, "process_cpu_sec": 147.4, "updates": 1600,
                               "by_category_sec": {"dev_verifier": 0.15}},
                         "B": {"wall_sec": 112.9, "process_cpu_sec": 112.8, "updates": 600,
                               "by_category_sec": {"dev_verifier": 0.19}}},
        "fixed_budget": True, "final_global_update": 2200})
    _dump(run_dir / "run_config.json", {"record": {"protocol": {"episodes_per_update": 512}}})


def parents_spec(q: int, i: int) -> Spec:
    """parents_A: |peak| 0.060 + 0.002 i (q50) / 0.050 + 0.002 i (q60); q60 seed index 2 fails G-A (tail 0.03)."""
    ap = (0.060 if q == 50 else 0.050) + 0.002 * i
    return Spec(peak=-ap, rmse=0.03, tail=0.03 if (q == 60 and i == 2) else 0.010, tail_max=0.05, eta=0.004,
                eta_dev=0.0041, sigma=3.4, smoothed=0.31, s1=0.01 + 0.002 * i, gmax=0.004)


BLOCKS_RULE = (("global", "localized", 3, (50, 10, 40)), ("polish", "broad", 19, (50, 5, 45)),
               ("global", None, 0, (50, 10, 40)))
BLOCKS_CAP = (("global", "broad", 20, (50, 10, 40)), ("global", "broad", 20, (50, 10, 40)))


def arm_spec(arm: str, q: int, i: int, with_failures: bool = True) -> Spec:
    """The synthetic arms (see the module docstring of the test of the criterion for what each one shows)."""
    p = parents_spec(q, i)
    pa = -p.peak
    base = dict(rmse=0.03, tail=0.008, tail_max=0.04, eta=0.004, eta_dev=0.0041, sigma=3.2, smoothed=0.2 + 0.01 * i,
                s1=0.012 + 0.002 * i, gmax=0.004, fire=True)
    if arm == "MS_base":
        sp = Spec(**{**base, "peak": p.peak, "tail": p.tail, "smoothed": 0.31, "fire": False})
        sp.blocks = (("legacy", None, 0, (50, 10, 40)),) * 3
        return sp
    if arm == "MS_rule":
        return Spec(**{**base, "peak": p.peak, "tail": p.tail, "blocks": BLOCKS_RULE})
    if arm == "MS_s25a0":
        return Spec(**{**base, "peak": -(pa - 0.02 - 0.001 * i), "fire": False, "forced": True,
                       "blocks": BLOCKS_CAP})
    if arm == "MS_s25a5":
        peak = -(pa - 0.02 - 0.001 * i) if q == 50 else -(pa + 0.01)
        return Spec(**{**base, "peak": peak})
    if arm == "MS_s35a0":
        ap = pa - 0.02 - 0.001 * i
        sp = Spec(**{**base, "peak": -ap if i % 2 == 0 else ap})
        if q == 50 and i == 0:
            sp.tail = 0.03                 # breaks the G-A tail part that parents_A passes at q50 seed index 0
        return sp
    if arm == "MS_s35a5":
        sp = Spec(**{**base, "peak": -(pa - 0.02)})
        if with_failures and q == 60 and i == 1:
            sp.status = "failed"
        if with_failures and q == 50 and i == 3:
            sp.status = "missing"
        return sp
    raise KeyError(arm)


@pytest.fixture(scope="module")
def genuine(tmp_path_factory):
    """Two tiny genuine runs of the real runner (MS_rule and the legacy MS_base) -> their templates."""
    root = tmp_path_factory.mktemp("genuine")
    dirs = {}
    for arm in ("MS_rule", "MS_base"):
        d = root / arm
        d.mkdir()
        prm = copy.deepcopy(mc.DEFAULT_PARAMS)
        prm["K"] = 10
        one = dict(eps=0.005, rho=0.03, tau=0.02, n_block=20, u_cap=20, n_land=10)
        prm["stages"] = {"2": dict(one), "1": dict(one)}
        cfg = mc.build_config(PROTO, 50, 10501, arm, str(d), prm)
        cfg["record"]["protocol"]["episodes_per_update"] = 32
        if arm == "MS_base":
            cfg["pipeline"]["budgets"] = {"2": 20, "1": 10}
            cfg["pipeline"]["lr_windows"] = {"2": [{"first": 11, "last": 20, "start": 3e-4, "end": 3e-5}],
                                             "1": [{"first": 1, "last": 10, "start": 3e-4, "end": 3e-5}]}
        (d / "run_config.json").write_text(json.dumps(cfg, indent=1))
        assert rms.run_pipeline(cfg, str(d), "pytest", band_step=1.0) == 0
        dirs[arm] = d
    return dirs


@pytest.fixture(scope="module")
def templates(genuine):
    return {arm: load_template(d) for arm, d in genuine.items()}


def build_world(root: Path, templates: Dict[str, Any], with_failures: bool = True, seeds=SEEDS4, qs=QS
                ) -> Dict[str, str]:
    """Synthetic base / pilot / parents / rehearsal roots under ``root`` (returns the roots dict)."""
    roots = {k: str(root / k) for k in ("base", "pilot", "parents", "rehearsal")}
    for q in qs:
        for i, s in enumerate(seeds):
            write_parents_run(Path(roots["parents"]) / ("q%d" % q) / ("seed%d" % s), q, s, parents_spec(q, i))
            write_rehearsal_run(Path(roots["rehearsal"]) / ("q%d" % q) / ("seed%d" % s), q, s,
                                parents_spec(q, i))
            for arm in R.MS_ARMS:
                sp = arm_spec(arm, q, i, with_failures)
                if sp.status == "missing":
                    continue
                d, _ = R.arm_dir(arm, q, s, roots)
                write_ms_run(d, templates["MS_base" if arm == "MS_base" else "MS_rule"], arm, q, s, sp)
    return roots


@pytest.fixture(scope="module")
def world(tmp_path_factory, templates):
    root = tmp_path_factory.mktemp("world")
    return build_world(root, templates, with_failures=True)


@pytest.fixture(scope="module")
def world_ok(tmp_path_factory, templates):
    root = tmp_path_factory.mktemp("world_ok")
    return build_world(root, templates, with_failures=False)


@pytest.fixture(scope="module")
def extracted(world):
    df, blocks, rblocks = R.extract_all(world, R.MS_ARMS, QS, SEEDS4, TH)
    return df, blocks, rblocks


def isT(x: Any) -> bool:
    """True only for a real boolean True (not NaN, not an empty cell)."""
    return isinstance(x, (bool, np.bool_)) and bool(x)


def isF(x: Any) -> bool:
    """True only for a real boolean False."""
    return isinstance(x, (bool, np.bool_)) and not bool(x)


def row_of(df: pd.DataFrame, arm: str, q: int, seed: int) -> pd.Series:
    r = df[(df["arm"] == arm) & (df["q"] == q) & (df["seed"] == seed)]
    assert len(r) == 1, (arm, q, seed)
    return r.iloc[0]


# ====================================================================== 1. the reader on genuine runner output
def test_reader_on_genuine_runner_output(genuine):
    """Every column the spec asks for is read from the real files of both stages (and the legacy arm)."""
    for arm in ("MS_rule", "MS_base"):
        d = genuine[arm]
        row, blocks, rb = R.extract_ms_run(d, arm, 50, 10501, TH, "pilot" if arm == "MS_rule" else "base")
        assert row["status"] == "done" and row["complete"] is True and row["exit_code"] == 0
        gates = json.loads((d / "gates.json").read_text())
        rl = json.loads((d / "rule_log.json").read_text())["stages"]
        fin = gates["reported"]["end_of_stage2"]["final"]
        assert row["stage2_peak_rel_err_signed"] == fin["stage2_peak_rel_err_signed"]
        assert row["stage2_peak_rel_err_abs"] == abs(fin["stage2_peak_rel_err_signed"])
        assert row["stage2_rmse_pos_over_g2_0"] == fin["stage2_rmse_pos_over_g2_0"]
        assert row["stage2_tail_mean_over_g2_0"] == fin["stage2_tail_mean_over_g2_0"]
        assert row["stage2_tail_max_over_g2_0"] == fin["stage2_tail_max_over_g2_0"]
        assert row["sigma_effort_at_0_t2"] == fin["sigma_effort_at_0_t2"]
        assert row["eta_T_over_dw"] == fin["eta_T_over_dw"]
        assert row["eta_dev"] == gates["reported"]["end_of_stage2"]["development"]["eta_T_over_dw"]
        assert row["smoothed_share_peak_gap_d0"] == gates["reported"]["end_of_stage2"]["smoothed_game"][
            "smoothed_share_peak_gap_d0"]
        for t, p in ((2, "t2_"), (1, "t1_")):
            fz = rl[str(t)]["freeze"]
            for tier, name in (("d3_final", "final"), ("d3_development", "dev")):
                assert row[p + "Delta_" + name] == fz[tier]["Delta"]
                assert row[p + "R_" + name] == fz[tier]["R"]
                assert row[p + "s_" + name] == fz[tier]["s"]
                assert row[p + "C_" + name] == fz[tier]["C"]
            assert row[p + "updates"] == rl[str(t)]["total_updates"]
            assert row[p + "episodes"] == rl[str(t)]["episodes"]
            assert row[p + "minibatch_steps"] == rl[str(t)]["minibatch_steps"]
            assert row[p + "wall_sec"] == rl[str(t)]["wall_sec"]
            assert row[p + "n_blocks"] == len(rl[str(t)]["blocks"])
            assert np.isfinite(row[p + "cpu_sec"]) and np.isfinite(row[p + "dev_verifier_sec"])
        assert row["t2_Rtail_final"] == rl["2"]["freeze"]["d3_final"]["R_tail"]
        assert row["stage1_rel_err_abs"] == abs(gates["reported"]["end_of_stage1"]["final"]["stage1_rel_err_signed"])
        assert row["Gmax_full_over_dw"] == gates["reported"]["end_of_stage1"]["final"]["Gmax_full_over_dw"]
        ib = json.loads((d / "induced_band.json").read_text())
        assert row["learning_rel"] == ib["learning_rel"] and row["inherited_rel_hi"] == ib["inherited_rel_hi"]
        assert row["total_updates"] == rl["2"]["total_updates"] + rl["1"]["total_updates"]
        assert row["v20_combination_pass"] == gates["v20_combination_pass"]
        assert row["G_A_pass"] == gates["G-A"]["pass"] and row["G_S_pass"] == gates["G-S"]["pass"]
        assert row["S1_pass"] == gates["S1"]["pass"] and row["G_F_pass"] == gates["G-F"]["pass"]
        # start shares and clamp counts from ms_updates.csv, recomputed without the tool
        u = pd.read_csv(d / "ms_updates.csv")
        s2 = u[u["stage"] == 2]
        train = s2[s2["block_type"] != "landing"]
        tot = float(train[["n_start_tail", "n_start_near", "n_start_mid"]].to_numpy().sum())
        assert row["t2_share_tail_train"] == pytest.approx(train["n_start_tail"].sum() / tot, abs=1e-15)
        assert row["t2_share_near_train"] == pytest.approx(train["n_start_near"].sum() / tot, abs=1e-15)
        assert row["t2_d1_L_s2_n"] == s2["d1_L_s2_n"].sum() and row["t2_d1_O_s2_hi"] == s2["d1_O_s2_hi"].sum()
        assert row["t1_d1_L_s1_n"] == u[u["stage"] == 1]["d1_L_s1_n"].sum()
        assert not blocks.empty and set(blocks["block_type"]) <= {"global", "polish", "landing", "legacy"}
        assert len(rb) == len(rl["2"]["blocks"]) + len(rl["1"]["blocks"])
        assert row["flags"].replace("git tree dirty at run time", "").strip("; ") == "", row["flags"]
    rule = R.extract_ms_run(genuine["MS_rule"], "MS_rule", 50, 10501, TH, "pilot")[0]
    legacy = R.extract_ms_run(genuine["MS_base"], "MS_base", 50, 10501, TH, "base")[0]
    assert rule["rule_enabled"] is True and legacy["rule_enabled"] is False
    assert legacy["t2_block_types"] == "legacy" and np.isnan(legacy["t2_landing_first_local"])
    assert rule["t2_landing_first_local"] == 21 and rule["t2_budget_forced"] is True      # cap 20, landing next
    assert rule["design_lambda_T"] == 0.5 and rule["sw_scheme"] == "stratified_priority"


def test_arm_table_matches_the_launcher_configs():
    """The analysis arm names are the arms of tools/ms/ms_configs.py (D6)."""
    assert set(R.MS_ARMS) == set(mc.ARMS) and R.MS_ARMS[0] == "MS_base"
    assert tuple(a for a in R.MS_ARMS if a != "MS_base") == tuple(mc.PILOT_ARMS)


def test_per_run_columns_cover_the_spec(extracted):
    """The D6 'reported per arm and q' list and the rule record are columns; the file header equals COLUMNS."""
    df = extracted[0]
    assert list(df.columns) == R.COLUMNS and len(set(R.COLUMNS)) == len(R.COLUMNS)
    must = [
        "stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_rmse_pos_over_g2_0",
        "stage2_tail_mean_over_g2_0", "stage2_tail_max_over_g2_0", "G_A_eta_pass", "G_A_rmse_pass", "G_A_tail_pass",
        "G_A_pass", "eta_T_over_dw", "eta_dev", "G_N_eta_pass", "sigma_effort_at_0_t2",
        "smoothed_share_peak_gap_d0", "t2_Delta_final", "t2_Delta_dev", "t2_R_final", "t2_R_dev", "t2_Rtail_final",
        "t2_Rtail_dev", "t2_updates", "t2_episodes", "t2_minibatch_steps", "t2_wall_sec", "t2_fire_local",
        "t2_budget_forced", "t2_n_blocks", "t2_block_types", "t2_classifications", "t2_nS_at_classification",
        "t2_share_tail_train", "t2_share_near_train", "t2_share_mid_train", "t2_block_share_tail",
        "t2_d1_L_s2_n", "t2_d1_L_s2_lo", "t2_d1_L_s2_hi", "stage1_rel_err_abs", "stage1_rel_err_signed", "t1_R_final",
        "t1_R_dev", "learning_rel", "inherited_rel", "learning_rel_lo", "learning_rel_hi", "Gmax_full_over_dw",
        "G_F_pass", "G_N_gmax_pass", "G_S_pass", "S1_pass", "v20_combination_pass", "t1_updates", "t1_episodes",
        "t1_minibatch_steps", "t1_wall_sec", "t1_fire_local", "t1_budget_forced", "t1_n_blocks"]
    assert [c for c in must if c not in df.columns] == []
    assert set(df["arm"]) == set(R.MS_ARMS) | set(R.COMPARATORS)


# ====================================================================== 2. per-run values on hand-computed cases
def test_per_run_values_on_a_hand_computed_rule_run(extracted):
    """MS_rule q50 seed 10501: global (localized, |S| 3) > polish (broad, |S| 19) > global (development stop)."""
    df = extracted[0]
    r = row_of(df, "MS_rule", 50, 10501)
    p = parents_spec(50, 0)
    assert r["status"] == "done" and bool(r["complete"])
    assert r["stage2_peak_rel_err_signed"] == p.peak and r["stage2_peak_rel_err_abs"] == 0.06
    assert r["stage2_rmse_pos_over_g2_0"] == 0.03 and r["stage2_tail_mean_over_g2_0"] == 0.010
    assert r["eta_T_over_dw"] == 0.004 and r["eta_dev"] == 0.0041
    assert r["eta_dev_minus_final_abs"] == pytest.approx(1e-4, abs=1e-15)
    assert [bool(r[c]) for c in ("G_A_eta_pass", "G_A_rmse_pass", "G_A_tail_pass", "G_A_pass", "G_N_eta_pass",
                                 "gate_pass")] == [True] * 6
    # rule record
    assert r["t2_fire_local"] == 3 * NB and isF(r["t2_budget_forced"]) and r["t2_n_blocks"] == 3
    assert r["t2_block_types"] == "global>polish>global" and r["t2_n_polish_blocks"] == 1
    assert r["t2_n_global_blocks"] == 2
    assert r["t2_classifications"] == "localized,broad" and r["t2_nS_at_classification"] == "3,19"
    assert (r["t2_n_localized"], r["t2_n_broad"]) == (1, 1)
    assert r["t2_landing_first_local"] == 3 * NB + 1 and r["t2_landing_followed_type"] == "global"
    # budget: 3 blocks x 10 + 5 landing updates
    n2 = 3 * NB + LAND
    assert r["t2_updates"] == n2 and r["t2_training_updates"] == 3 * NB
    assert r["t2_episodes"] == n2 * EPU and r["t2_minibatch_steps"] == n2 * 10
    assert r["total_updates"] == n2 + 6 and r["total_episodes"] == (n2 + 6) * EPU
    # start shares: 10 updates per block, 100 starts per update
    assert r["t2_n_updates_train"] == 3 * NB and r["t2_n_updates_landing"] == LAND and r["t2_n_updates_polish"] == NB
    assert r["t2_share_tail_train"] == 0.5
    assert (r["t2_n_start_tail_train"], r["t2_n_start_near_train"], r["t2_n_start_mid_train"]) == (
        50 * 3 * NB, (10 + 5 + 10) * NB, (40 + 45 + 40) * NB)
    assert r["t2_share_near_train"] == pytest.approx((10 * NB + 5 * NB + 10 * NB) / (3.0 * NB * EPU), abs=1e-12)
    assert r["t2_share_mid_train"] == pytest.approx((40 * NB + 45 * NB + 40 * NB) / (3.0 * NB * EPU), abs=1e-12)
    assert r["t2_block_share_near"] == "0.1;0.05;0.1;0.1" and r["t2_block_share_tail"] == "0.5;0.5;0.5;0.5"
    assert r["t2_block_share_mid"] == "0.4;0.45;0.4;0.4"
    # clamp counts: per update L 1 low / 2 high of 100, O 0 / 1
    assert r["t2_d1_L_s2_n"] == n2 * EPU and r["t2_d1_L_s2_lo"] == n2 and r["t2_d1_L_s2_hi"] == 2 * n2
    assert r["t2_d1_O_s2_hi"] == n2 and r["t2_clamp_L_frac"] == pytest.approx(3 * n2 / (n2 * EPU))
    assert r["t1_d1_L_s1_n"] == 6 * EPU and r["t1_d1_L_s1_lo"] == 18
    # stage 1
    assert r["stage1_rel_err_signed"] == 0.012 and r["stage1_rel_err_abs"] == 0.012
    assert bool(r["G_S_pass"]) and bool(r["S1_pass"]) and bool(r["G_F_pass"]) and bool(r["G_N_gmax_pass"])
    assert r["learning_rel"] == 0.004 and r["inherited_rel"] == 0.006
    assert r["t2_R_final"] == 0.02 and r["t1_R_final"] == 0.02 and r["t1_Delta_dev"] == 1.1e-4
    assert bool(r["v20_combination_pass"]) and r["v20_combination_pass_recorded"] in (True, 1)
    assert r["smoothed_share_peak_gap_d0"] == 0.20
    assert r["design_lambda_T"] == 0.5 and r["design_lambda_P"] == 0.1


def test_block_shares_by_block_and_rule_blocks(extracted):
    """Per-block start shares (long table) and the flattened rule blocks of both stages."""
    _, blocks, rblocks = extracted
    b = blocks[(blocks["arm"] == "MS_rule") & (blocks["q"] == 50) & (blocks["seed"] == 10501)].sort_values("block_id")
    assert list(b["block_id"]) == [1, 2, 3, 4] and list(b["block_type"]) == ["global", "polish", "global", "landing"]
    assert list(b["n_updates"]) == [NB, NB, NB, LAND]
    assert list(b["share_near"]) == [0.1, 0.05, 0.1, 0.1]
    assert list(b["share_tail"]) == [0.5] * 4 and list(b["share_mid"]) == [0.4, 0.45, 0.4, 0.4]
    rb = rblocks[(rblocks["arm"] == "MS_rule") & (rblocks["q"] == 50) & (rblocks["seed"] == 10501)]
    t2 = rb[rb["stage"] == 2].sort_values("block_id")
    assert list(t2["classification"]) == ["localized", "broad", ""] and list(t2["n_S"]) == [3, 19, 0]
    assert list(t2["exit_reason"]) == ["block_end", "block_end", "development_stop"]
    assert t2["S"].iloc[0] == "0,1,2"
    leg = blocks[(blocks["arm"] == "MS_base") & (blocks["q"] == 50) & (blocks["seed"] == 10501)]
    assert list(leg["block_type"]) == ["legacy"]


def test_gate_verdicts_are_inclusive_and_use_both_tiers():
    """G-A on the final tier (<=, inclusive) and the eta part of G-N from the two tiers' eta."""
    base = {"stage2_peak_rel_err_signed": -0.05, "stage2_peak_rel_err_abs": 0.05, "stage2_rmse_pos_over_g2_0": 0.05,
            "stage2_tail_mean_over_g2_0": 0.02, "eta_T_over_dw": 0.005}
    on_edge = R.terminal_cols(base, {"eta_T_over_dw": 0.006}, TH)          # every limit exactly met; gap = 0.001
    assert on_edge["G_A_pass"] is True and on_edge["G_N_eta_pass"] is True and on_edge["gate_pass"] is True
    assert on_edge["eta_dev_minus_final_abs"] == pytest.approx(0.001, abs=1e-15)
    over = R.terminal_cols(dict(base, eta_T_over_dw=0.0050001), {"eta_T_over_dw": 0.005}, TH)
    assert over["G_A_eta_pass"] is False and over["G_A_pass"] is False and over["gate_pass"] is False
    gap = R.terminal_cols(dict(base, eta_T_over_dw=0.004), {"eta_T_over_dw": 0.0051}, TH)
    assert gap["G_A_pass"] is True and gap["G_N_eta_pass"] is False and gap["gate_pass"] is False
    for key, val, flag in (("stage2_rmse_pos_over_g2_0", 0.0500001, "G_A_rmse_pass"),
                           ("stage2_tail_mean_over_g2_0", 0.0200001, "G_A_tail_pass")):
        o = R.terminal_cols(dict(base, **{key: val}), {"eta_T_over_dw": 0.005}, TH)
        assert o[flag] is False and o["G_A_pass"] is False
    nan = R.terminal_cols({}, {}, TH)                                       # an unreadable run is never a pass
    assert nan["gate_pass"] is False and np.isnan(nan["stage2_peak_rel_err_abs"])
    s1 = R.stage1_cols({"stage1_rel_err_abs": 0.05, "stage1_rel_err_signed": -0.05, "Gmax_full_over_dw": 0.01},
                       {"Gmax_full_over_dw": 0.011}, TH)
    assert s1["G_S_pass"] is True and s1["G_F_pass"] is True and s1["G_N_gmax_pass"] is True and s1["S1_pass"] is True
    s1b = R.stage1_cols({"stage1_rel_err_abs": 0.0501, "Gmax_full_over_dw": 0.0100001}, {"Gmax_full_over_dw": 0.0},
                        TH)
    assert s1b["G_S_pass"] is False and s1b["S1_pass"] is True and s1b["G_F_pass"] is False
    assert s1b["G_N_gmax_pass"] is False


def test_rule_and_update_column_helpers_on_hand_made_inputs():
    rec = {"blocks": [{"type": "global", "classification": "broad", "n_S": 12},
                      {"type": "polish", "classification": "localized", "n_S": 4},
                      {"type": "polish", "classification": None, "n_S": 4}],
           "landing": {"first_local": 31, "followed_block_type": "polish"}, "fire_local": 30, "budget_forced": False,
           "total_updates": 45, "training_updates": 30, "episodes": 100, "minibatch_steps": 7, "wall_sec": 2.5,
           "n_checks": 3, "would_fire_local": None}
    c = R.rule_cols(rec, 2)
    assert c["t2_block_types"] == "global>polish>polish" and c["t2_n_polish_blocks"] == 2
    assert c["t2_classifications"] == "broad,localized" and c["t2_nS_at_classification"] == "12,4"
    assert c["t2_fire_local"] == 30 and c["t2_budget_forced"] is False and np.isnan(c["t2_would_fire_local"])
    assert c["t2_landing_followed_type"] == "polish" and c["t2_updates"] == 45
    u = pd.DataFrame({"stage": [2, 2, 2, 1], "block_id": [1, 1, 2, 1],
                      "block_type": ["global", "global", "landing", "global"],
                      "n_start_tail": [5, 3, 1, 0], "n_start_near": [1, 1, 2, 0], "n_start_mid": [4, 6, 7, 0],
                      "d1_L_s2_n": [10, 10, 10, np.nan], "d1_L_s2_lo": [1, 0, 0, np.nan],
                      "d1_L_s2_hi": [0, 2, 0, np.nan]})
    an: List[str] = []
    cols, bl = R.update_cols(u, an)
    assert cols["t2_share_tail_train"] == 8 / 20 and cols["t2_share_near_train"] == 2 / 20
    assert cols["t2_share_mid_all"] == pytest.approx(17 / 30) and cols["t2_n_updates_landing"] == 1
    assert list(bl["n_tail"]) == [8, 1] and list(bl["block_type"]) == ["global", "landing"]
    assert cols["t2_d1_L_s2_hi"] == 2 and cols["t2_clamp_L_frac"] == pytest.approx(3 / 30)
    assert any("d1_O_s2_n" in a for a in an)                 # a missing d1 column is reported, not skipped


# ====================================================================== 3. failed / missing / running / incomplete
def test_status_classification(tmp_path):
    for k, (state, code, want) in {"a": ("done", 0, "done"), "b": ("done", 5, "failed"),
                                   "c": ("failed", 1, "failed"), "d": ("running", None, "running"),
                                   "e": ("done", 3, "failed")}.items():
        d = tmp_path / k
        d.mkdir()
        st = {"state": state, "pid": 7}
        if code is not None:
            st["exit_code"] = code
        (d / "status.json").write_text(json.dumps(st))
        s, info, ec, _ = R.run_status(d)
        assert s == want, (k, s, info)
    assert "RNG" in R.run_status(tmp_path / "b")[1]
    assert R.run_status(tmp_path / "nope")[0] == "missing"
    (tmp_path / "empty").mkdir()
    assert R.run_status(tmp_path / "empty")[0] == "missing"
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "status.json").write_text("{not json")
    assert R.run_status(tmp_path / "bad")[0] == "failed"


def test_failed_missing_running_and_incomplete_runs_are_rows(templates, tmp_path):
    roots = build_world(tmp_path, templates, with_failures=False, seeds=SEEDS4[:3], qs=(50,))
    # one run of each kind in the MS_s25a5 arm
    arm = "MS_s25a5"
    kinds = {10501: "failed", 10502: "running", 10503: "nofiles"}
    for s, kind in kinds.items():
        shutil.rmtree(R.arm_dir(arm, 50, s, roots)[0])
        write_ms_run(R.arm_dir(arm, 50, s, roots)[0], templates["MS_rule"], arm, 50, s, Spec(status=kind))
    shutil.rmtree(R.arm_dir("MS_rule", 50, 10502, roots)[0])                         # a missing directory
    shutil.rmtree(R.arm_dir("MS_s35a0", 50, 10501, roots)[0])
    write_ms_run(R.arm_dir("MS_s35a0", 50, 10501, roots)[0], templates["MS_rule"], "MS_s35a0", 50, 10501,
                 Spec(status="rng"))
    df, _, _ = R.extract_all(roots, R.MS_ARMS, (50,), SEEDS4[:3], TH)
    assert len(df[(df["arm"] == arm)]) == 3 and len(df[(df["arm"] == "MS_rule")]) == 3      # no silent skip
    f = row_of(df, arm, 50, 10501)
    assert f["status"] == "failed" and not f["complete"] and f["exit_code"] == 1 and "boom" in f["status_info"]
    assert np.isfinite(f["stage2_peak_rel_err_abs"])                    # the stage-2 freeze record is kept
    assert np.isnan(f["stage1_rel_err_abs"])                             # stage 1 never ran
    r = row_of(df, arm, 50, 10502)
    assert r["status"] == "running" and not r["complete"]
    n = row_of(df, arm, 50, 10503)
    assert n["status"] == "incomplete" and not n["complete"] and "gates.json missing" in n["flags"]
    m = row_of(df, "MS_rule", 50, 10502)
    assert m["status"] == "missing" and m["status_info"] == "run directory absent" and not m["complete"]
    g = row_of(df, "MS_s35a0", 50, 10501)
    assert g["status"] == "failed" and "global-RNG violation" in g["status_info"] and g["exit_code"] == 5
    assert not g["complete"] and isF(g["global_rng_ok"])
    comp = R.completeness_table(df, R.MS_ARMS, (50,), SEEDS4[:3])
    c = comp[(comp["arm"] == arm)].iloc[0]
    assert (c["n_failed"], c["n_running"], c["n_incomplete"], c["n_done"]) == (1, 1, 1, 0)
    assert "seed10501: failed" in c["not_done_runs"]
    c2 = comp[(comp["arm"] == "MS_rule")].iloc[0]
    assert c2["n_missing"] == 1 and c2["n_done"] == 2
    # none of them enters a paired difference
    sv = R.paired_seed_values(df, arm, "parents_A", 50, R.PRIMARY, SEEDS4[:3])
    assert len(sv) == 0
    sv = R.paired_seed_values(df, "MS_rule", "parents_A", 50, R.PRIMARY, SEEDS4[:3])
    assert list(sv["seed"]) == [10501, 10503]


# ====================================================================== 4. bootstrap
def test_bootstrap_is_deterministic_and_uses_a_fresh_generator_per_call(monkeypatch):
    x = np.array([0.03, -0.01, 0.02, 0.005, -0.02, 0.01, 0.04, -0.03, 0.0, 0.015])
    # independent computation of the pre-registered scheme
    rng = np.random.default_rng(20261006)
    idx = rng.integers(0, x.size, size=(10000, x.size))
    want = (float(np.percentile(x[idx].mean(axis=1), 2.5)), float(np.percentile(x[idx].mean(axis=1), 97.5)))
    got = R.boot_ci(x)
    assert got == want and R.boot_ci(x) == got                              # deterministic
    seeds_seen: List[int] = []
    real = np.random.default_rng

    def spy(seed=None):
        seeds_seen.append(seed)
        return real(seed)
    monkeypatch.setattr(np.random, "default_rng", spy)
    R.boot_ci(x)
    R.boot_ci(x, "median")
    R.paired_summary(x, True)                                               # mean CI and median CI: two generators
    R.summary_stats(x)
    assert seeds_seen == [20261006] * 5                                     # a fresh generator per call, seed fixed
    # the interval of the median uses the same indices
    med = R.boot_ci(x, "median")
    assert med == (float(np.percentile(np.median(x[idx], axis=1), 2.5)),
                   float(np.percentile(np.median(x[idx], axis=1), 97.5)))
    assert R.N_BOOT == 10000 and R.BOOT_SEED == 20261006
    assert all(np.isnan(v) for v in R.boot_ci([]))
    # a generator shared across calls would give different second intervals: ours do not
    assert R.boot_ci(x[:7]) == R.boot_ci(x[:7])


def test_paired_table_draws_one_fresh_generator_per_q_and_statistic_in_table_order(extracted, monkeypatch):
    """Table order = arm, q, metric, (mean, median): every cell constructs its own default_rng(20261006), so a
    cell's interval does not depend on the cells computed before it."""
    df = extracted[0]
    seen: List[Tuple[Any, int]] = []
    real = np.random.default_rng

    def spy(seed=None):
        seen.append((seed, len(seen)))
        return real(seed)
    monkeypatch.setattr(np.random, "default_rng", spy)
    p, _ = R.paired_tables(df, [("MS_s25a0", "parents_A", "")], [(R.PRIMARY, True)], QS, SEEDS4, "vs parents_A")
    assert [x[0] for x in seen] == [20261006] * 4                      # q50 mean, q50 median, q60 mean, q60 median
    assert list(p["q"]) == [50, 60]
    monkeypatch.undo()
    alone, _ = R.paired_tables(df, [("MS_s25a0", "parents_A", "")], [(R.PRIMARY, True)], (60,), SEEDS4,
                               "vs parents_A")
    for col in ("ci_mean_lo", "ci_mean_hi", "ci_median_lo", "ci_median_hi", "mean"):
        assert alone[col].iloc[0] == p[p["q"] == 60][col].iloc[0]


def test_criterion_bootstrap_is_the_mean_of_the_paired_differences_with_a_fresh_generator(extracted):
    df = extracted[0]
    crit = R.criterion_table(df, R.paired_tables(df, [("MS_s25a0", "parents_A", "")], R.S2_METRICS, QS, SEEDS4,
                                                 "vs parents_A")[0], ["MS_s25a0"], QS, SEEDS4).iloc[0]
    for q in QS:
        d = []
        for i in range(4):
            d.append(-(0.02 + 0.001 * i))
        d = np.array(d)
        rng = np.random.default_rng(20261006)
        m = d[rng.integers(0, 4, size=(10000, 4))].mean(axis=1)
        assert crit["mean_q%d" % q] == pytest.approx(d.mean(), abs=1e-15)
        assert crit["ci_mean_lo_q%d" % q] == pytest.approx(np.percentile(m, 2.5), abs=1e-15)
        assert crit["ci_mean_hi_q%d" % q] == pytest.approx(np.percentile(m, 97.5), abs=1e-15)
        assert crit["n_pairs_q%d" % q] == 4


# ====================================================================== 5. paired tables
def test_paired_tables_signs_counts_and_exclusions(extracted):
    df = extracted[0]
    p, s = R.paired_tables(df, [("MS_s25a0", "parents_A", ""), ("MS_s35a5", "parents_A", "")], R.S2_METRICS, QS,
                           SEEDS4, "vs parents_A")
    r = p[(p["arm"] == "MS_s25a0") & (p["q"] == 50) & (p["metric"] == R.PRIMARY)].iloc[0]
    assert r["n_pairs"] == 4 and r["mean"] == pytest.approx(-0.0215, abs=1e-15)
    assert r["n_better"] == 4 and r["n_neg"] == 4 and r["n_pos"] == 0 and r["n_zero"] == 0
    assert r["median"] == pytest.approx(-0.0215, abs=1e-15)
    assert bool(r["ci_mean_below_0"]) and not bool(r["ci_mean_contains_0"])
    assert r["status"] == R.CRITERION_NOTE                                  # the only pre-registered row
    other = p[(p["arm"] == "MS_s25a0") & (p["q"] == 50) & (p["metric"] == "stage2_rmse_pos_over_g2_0")].iloc[0]
    assert other["status"] == "descriptive"
    # a failed (q60 seed idx 1) and a missing (q50 seed idx 3) run of MS_s35a5 do not pair
    a60 = p[(p["arm"] == "MS_s35a5") & (p["q"] == 60) & (p["metric"] == R.PRIMARY)].iloc[0]
    a50 = p[(p["arm"] == "MS_s35a5") & (p["q"] == 50) & (p["metric"] == R.PRIMARY)].iloc[0]
    assert (a60["n_pairs"], a50["n_pairs"]) == (3, 3)
    sl = s[(s["arm"] == "MS_s35a5") & (s["q"] == 60) & (s["metric"] == R.PRIMARY)]
    assert 10502 not in set(sl["seed"]) and len(sl) == 3
    assert np.allclose(sl["diff"], -0.02) and np.allclose(sl["base_value"] - sl["arm_value"], 0.02)
    # a metric that is NaN for the baseline (D3 columns of parents_A) is omitted, never a silent NaN row
    assert p[(p["arm"] == "MS_s25a0") & (p["metric"] == "t2_R_final")].empty
    # the control comparison keeps D3 columns
    pr, _ = R.paired_tables(df, [("MS_s25a0", "MS_rule", "")], R.S2_METRICS, QS, SEEDS4, "vs MS_rule")
    rr = pr[(pr["metric"] == "t2_R_final") & (pr["q"] == 50)].iloc[0]
    assert rr["n_pairs"] == 4 and rr["mean"] == pytest.approx(0.0, abs=1e-15)
    # stage 1 against rehearsal_v2_0: |error| 0.012 + 0.002 i against 0.010 + 0.002 i
    ps, _ = R.paired_tables(df, [("MS_s25a0", "rehearsal_v2_0", "")], R.S1_METRICS, QS, SEEDS4, "vs rehearsal_v2_0")
    s1 = ps[(ps["q"] == 60) & (ps["metric"] == R.PRIMARY_S1)].iloc[0]
    assert s1["n_pairs"] == 4 and s1["mean"] == pytest.approx(0.002, abs=1e-15) and s1["n_pos"] == 4
    assert s1["status"] == "descriptive" and s1["baseline"] == "rehearsal_v2_0"
    # every rule arm against MS_rule: the four other rule arms, both stages' metrics
    pm, _ = R.paired_tables(df, [(a, "MS_rule", "") for a in R.MS_ARMS if a not in ("MS_base", "MS_rule")],
                            R.S2_METRICS + R.S1_METRICS[:1], QS, SEEDS4, "vs MS_rule")
    assert set(pm["arm"]) == {"MS_s25a0", "MS_s25a5", "MS_s35a0", "MS_s35a5"}
    m = pm[(pm["arm"] == "MS_s25a0") & (pm["q"] == 50) & (pm["metric"] == R.PRIMARY)].iloc[0]
    assert m["mean"] == pytest.approx(-0.0215, abs=1e-15) and m["baseline"] == "MS_rule"
    # sign-less metrics carry no n_better
    sg = p[(p["arm"] == "MS_s25a0") & (p["q"] == 50) & (p["metric"] == R.SIGNED)].iloc[0]
    assert sg["n_better"] is None or pd.isna(sg["n_better"])


# ====================================================================== 6. criterion (a) and (b)
@pytest.fixture(scope="module")
def tables(world):
    df, _, _ = R.extract_all(world, R.MS_ARMS, QS, SEEDS4, TH)
    p, _ = R.paired_tables(df, [(a, "parents_A", "") for a in R.MS_ARMS], R.S2_METRICS, QS, SEEDS4, "vs parents_A")
    crit = R.criterion_table(df, p, R.MS_ARMS, QS, SEEDS4)
    return df, p, crit


def _c(crit: pd.DataFrame, arm: str) -> pd.Series:
    return crit[crit["arm"] == arm].iloc[0]


def test_criterion_arm_improving_at_both_q(tables):
    """MS_s25a0: |peak| lower by 0.02 + 0.001 i at both q, every G-A / G-N pass kept: (a) met, (b) holds."""
    c = _c(tables[2], "MS_s25a0")
    assert bool(c["a_q50"]) and bool(c["a_q60"]) and bool(c["a_met"]) and bool(c["a_complete"])
    assert c["ci_mean_hi_q50"] < 0 and c["ci_mean_hi_q60"] < 0
    assert c["b_status"] == "holds" and (c["b_violations"] == "" or pd.isna(c["b_violations"]))
    assert c["n_base_pass"] == 7                                    # parents_A: 8 runs, q60 seed idx 2 fails G-A
    assert c["overall"] == "met"


def test_criterion_arm_improving_at_one_q_only(tables):
    """MS_s25a5: improves at q50, worse at q60: (a) is a conjunction over both q."""
    c = _c(tables[2], "MS_s25a5")
    assert bool(c["a_q50"]) and not bool(c["a_q60"]) and not bool(c["a_met"])
    assert c["mean_q60"] == pytest.approx(0.01, abs=1e-15) and c["ci_mean_lo_q60"] > 0
    assert c["b_status"] == "holds" and c["overall"] == "not met"


def test_criterion_arm_that_breaks_a_G_A_pass(tables):
    """MS_s35a0: |peak| improves at both q but one run fails G-A (tail 0.03) where parents_A passes: (b) violated."""
    c = _c(tables[2], "MS_s35a0")
    assert bool(c["a_met"]) and c["b_status"] == "violated"
    assert c["b_violations"] == "q50/10501" and c["b_n_violations"] == 1
    assert c["overall"] == "not met"
    # the violation is only counted where the baseline passes: parents_A q60 seed idx 2 fails G-A itself
    df = tables[0]
    assert not bool(row_of(df, "parents_A", 60, 10503)["gate_pass"])
    assert bool(row_of(df, "parents_A", 50, 10501)["gate_pass"])
    assert not bool(row_of(df, "MS_s35a0", 50, 10501)["gate_pass"]) and not bool(
        row_of(df, "MS_s35a0", 50, 10501)["G_A_tail_pass"])


def test_criterion_failed_and_missing_runs(tables):
    """MS_s35a5: a failed run (baseline passes there) is a violation, a missing run is pending; both leave
    a pair out."""
    c = _c(tables[2], "MS_s35a5")
    assert c["b_status"] == "violated" and c["b_violations"] == "q60/10502" and c["b_pending"] == "q50/10504"
    assert (c["b_n_violations"], c["b_n_pending"]) == (1, 1)
    assert c["n_pairs_q50"] == 3 and c["n_pairs_q60"] == 3 and not bool(c["a_complete"])
    assert bool(c["a_met"]) and c["overall"] == "not met"


def test_criterion_identical_arms_have_a_zero_interval(tables):
    """MS_base and MS_rule have exactly parents_A's peak errors: difference 0, CI [0, 0], (a) not met."""
    for arm in ("MS_base", "MS_rule"):
        c = _c(tables[2], arm)
        assert c["mean_q50"] == 0.0 and c["ci_mean_lo_q50"] == 0.0 and c["ci_mean_hi_q50"] == 0.0
        assert not bool(c["a_met"]) and c["b_status"] == "holds" and c["overall"] == "not met"


def test_criterion_row_pending_violation_and_holds_on_constructed_frames():
    prim = pd.DataFrame({"q": [50, 60], "n_pairs": [2, 2], "mean": [-0.1, -0.1], "ci_mean_lo": [-0.2, -0.2],
                         "ci_mean_hi": [-0.05, -0.05]})

    def gates(rows):
        return pd.DataFrame(rows, columns=["q", "seed", "base_pass", "arm_status", "arm_pass"])
    both = [(50, 1, True, "done", True), (50, 2, True, "done", True), (60, 1, True, "done", True),
            (60, 2, False, "done", False)]
    ok = R.criterion_row(prim, gates(both), (50, 60), 2)
    assert ok["b_status"] == "holds" and ok["overall"] == "met" and ok["n_base_pass"] == 3
    pend = R.criterion_row(prim, gates(both[:3] + [(60, 2, True, "running", np.nan)]), (50, 60), 2)
    assert pend["b_status"] == "incomplete" and pend["b_pending"] == "q60/2" and pend["overall"] == "incomplete"
    viol = R.criterion_row(prim, gates([(50, 1, True, "done", False)] + both[1:]), (50, 60), 2)
    assert viol["b_status"] == "violated" and viol["overall"] == "not met"
    fail = R.criterion_row(prim, gates([(50, 1, True, "failed", np.nan)] + both[1:]), (50, 60), 2)
    assert fail["b_violations"] == "q50/1"
    edge = prim.copy()
    edge.loc[0, "ci_mean_hi"] = 0.0                                          # strict inequality: hi == 0 is not met
    assert not R.criterion_row(edge, gates(both), (50, 60), 2)["a_met"]
    short = R.criterion_row(prim.iloc[:1], gates(both), (50, 60), 2)         # q60 has no pairs
    assert not short["a_met"] and short["n_pairs_q60"] == 0 and not short["a_complete"]


# ====================================================================== 7. the other tables
def test_decision_inputs_budget_rule_and_tail_tables(world):
    df, _, _ = R.extract_all(world, R.MS_ARMS, QS, SEEDS4, TH)
    p, _ = R.paired_tables(df, [(a, "parents_A", "") for a in R.MS_ARMS], R.S2_METRICS, QS, SEEDS4, "vs parents_A")
    crit = R.criterion_table(df, p, R.MS_ARMS, QS, SEEDS4)
    dec = R.decision_inputs(df, crit, p, R.MS_ARMS, QS, SEEDS4)
    row = dec[(dec["arm"] == "MS_s25a0") & (dec["q"] == 50)].iloc[0]
    assert row["n_abs_peak_le_0.05"] == 4                       # |peak| = 0.040 + 0.001 i (i = 0..3), all <= 0.05
    assert row["crit_a_met"] and row["n_complete"] == 4 and row["n_t2_budget_forced"] == 4
    pr = dec[(dec["arm"] == "parents_A") & (dec["q"] == 50)].iloc[0]
    assert pr["n_abs_peak_le_0.05"] == 0 and pr["n_gate_pass"] == 4 and pr["role"] == "comparator"
    assert dec[(dec["arm"] == "parents_A") & (dec["q"] == "both")].iloc[0]["n_complete"] == 8
    # signed interval and whether it contains 0: MS_s35a0 alternates the sign of the peak (mixed), MS_s25a0 does not
    mixed = dec[(dec["arm"] == "MS_s35a0") & (dec["q"] == 60)].iloc[0]
    one = dec[(dec["arm"] == "MS_s25a0") & (dec["q"] == 60)].iloc[0]
    assert bool(mixed["signed_peak_ci_contains_0"]) and not bool(one["signed_peak_ci_contains_0"])
    assert one["n_signed_peak_negative"] == 4 and mixed["n_signed_peak_negative"] == 2
    v = np.array([-(0.03 + 0.001 * i) for i in range(4)])
    rng = np.random.default_rng(20261006)
    m = v[rng.integers(0, 4, size=(10000, 4))].mean(axis=1)
    assert one["signed_peak_ci_lo"] == pytest.approx(np.percentile(m, 2.5), abs=1e-15)
    # budget: both comparators present, rehearsal episodes derived from updates x 512
    b = R.budget_table(df, R.MS_ARMS, QS)
    e = b[(b["arm"] == "rehearsal_v2_0") & (b["q"] == 50) & (b["metric"] == "t2_episodes")].iloc[0]
    assert e["mean"] == 1600 * 512
    assert not b[(b["arm"] == "parents_A") & (b["metric"] == "t2_minibatch_steps")].empty
    assert b[(b["arm"] == "rehearsal_v2_0") & (b["metric"] == "t2_minibatch_steps")].empty
    # rule table
    rt = R.rule_table(df, R.MS_ARMS, QS)
    r = rt[(rt["arm"] == "MS_rule") & (rt["q"] == 50)].iloc[0]
    assert r["t2_n_fire"] == 4 and r["t2_n_budget_forced"] == 0 and r["t2_fire_local_median"] == 3 * NB
    assert r["t2_n_polish_blocks_total"] == 4 and r["t2_n_localized_total"] == 4 and r["t2_n_broad_total"] == 4
    assert r["t2_nS_at_classification_mean"] == 11.0 and r["t2_nS_at_classification_max"] == 19
    cap = rt[(rt["arm"] == "MS_s25a0") & (rt["q"] == 50)].iloc[0]
    assert cap["t2_n_budget_forced"] == 4 and cap["t2_n_fire"] == 0
    # tail table
    tt = R.tail_table(df, R.MS_ARMS, QS, TH)
    t = tt[(tt["arm"] == "parents_A") & (tt["q"] == 60)].iloc[0]
    assert t["n_tail_mean_over_limit"] == 1 and t["n_G_A_tail_pass"] == 3 and t["tail_mean_max"] == 0.03
    # decomposition table notes that parents_A does not store the smoothed share
    de = R.decomposition_table(df)
    assert np.isnan(de[de["arm"] == "parents_A"]["smoothed_share_peak_gap_d0"]).all()
    assert de[de["arm"] == "parents_A"]["note"].str.contains("not stored").all()
    assert (de[de["arm"] == "rehearsal_v2_0"]["smoothed_share_peak_gap_d0"] == 0.31).all()
    s1 = R.stage1_table(df, R.MS_ARMS, QS, TH)
    assert not s1[(s1["arm"] == "rehearsal_v2_0")].empty


# ====================================================================== 8. the blind recomputation
def _run_cli(roots, out, extra=()):
    argv = ["--base-root", roots["base"], "--pilot-root", roots["pilot"], "--parents-root", roots["parents"],
            "--rehearsal-root", roots["rehearsal"], "--out", str(out), *extra]
    return R.main(argv)


@pytest.fixture(scope="module")
def cli_out(world, tmp_path_factory):
    out = tmp_path_factory.mktemp("analysis")
    code = _run_cli(world, out, ["--seeds", "10501", "10502", "10503", "10504", "--no-figures"])
    return out, code


def test_blind_recomputation_agrees(cli_out):
    out, code = cli_out
    assert code == 3                                                      # the world has a failed and a missing run
    rc = BL.main(["--analysis-dir", str(out)])
    txt = (out / "blind_recomputation.txt").read_text()
    assert rc == 0 and "ALL " in txt and "DISAGREE" not in txt
    assert "numpy.random.default_rng(20261006)" in txt and "MS_s35a0" in txt
    assert "roots (analysis_info.json" in txt and "parents" in txt
    # every float agrees to 1e-12: recompute one by hand from the file
    per = pd.read_csv(out / "per_run.csv", float_precision="round_trip")
    crit = pd.read_csv(out / "criterion.csv", float_precision="round_trip")
    a = per[(per["arm"] == "MS_s25a0") & (per["q"] == 50)].set_index("seed")["stage2_peak_rel_err_abs"]
    b = per[(per["arm"] == "parents_A") & (per["q"] == 50)].set_index("seed")["stage2_peak_rel_err_abs"]
    d = (a - b).to_numpy()
    r = crit[crit["arm"] == "MS_s25a0"].iloc[0]
    assert abs(r["mean_q50"] - d.mean()) <= 1e-12


@pytest.mark.parametrize("tamper", ["ci_hi", "mean", "flag", "violation", "pairs", "drop_arm", "extra_arm",
                                    "overall"])
def test_blind_recomputation_fails_when_criterion_is_tampered_with(cli_out, tmp_path, tamper):
    out, _ = cli_out
    d = tmp_path / "analysis"
    shutil.copytree(out, d)
    crit = pd.read_csv(d / "criterion.csv", float_precision="round_trip")
    i = int(crit.index[crit["arm"] == "MS_s25a5"][0])
    if tamper == "ci_hi":
        crit.loc[i, "ci_mean_hi_q50"] += 1e-9                              # far below any tolerance of a sloppy check
    elif tamper == "mean":
        crit.loc[i, "mean_q60"] *= 1.000001
    elif tamper == "flag":
        crit.loc[i, "a_q60"] = True
    elif tamper == "violation":
        crit.loc[i, "b_violations"] = "q50/10501"
    elif tamper == "pairs":
        crit.loc[i, "n_pairs_q50"] = 3
    elif tamper == "drop_arm":
        crit = crit.drop(index=i)
    elif tamper == "extra_arm":
        crit = pd.concat([crit, crit.iloc[[i]].assign(arm="MS_ghost")], ignore_index=True)
    elif tamper == "overall":
        crit.loc[i, "overall"] = "met"
    crit.to_csv(d / "criterion.csv", index=False)
    assert BL.main(["--analysis-dir", str(d)]) == 1
    assert "DISAGREE" in (d / "blind_recomputation.txt").read_text()


def test_blind_recomputation_detects_a_changed_per_run_value_and_unreadable_inputs(cli_out, tmp_path):
    out, _ = cli_out
    d = tmp_path / "a2"
    shutil.copytree(out, d)
    per = pd.read_csv(d / "per_run.csv", float_precision="round_trip")
    j = per.index[(per["arm"] == "MS_s25a0") & (per["q"] == 50) & (per["seed"] == 10501)][0]
    per.loc[j, "stage2_peak_rel_err_abs"] += 1e-6
    per.to_csv(d / "per_run.csv", index=False)
    assert BL.main(["--analysis-dir", str(d)]) == 1
    (d / "per_run.csv").unlink()
    assert BL.main(["--analysis-dir", str(d)]) == 2
    # a flipped gate_pass column is caught by the recomputation from the component columns
    d2 = tmp_path / "a3"
    shutil.copytree(out, d2)
    per = pd.read_csv(d2 / "per_run.csv", float_precision="round_trip")
    j = per.index[(per["arm"] == "MS_s25a0") & (per["q"] == 50) & (per["seed"] == 10501)][0]
    per.loc[j, "gate_pass"] = False
    per.to_csv(d2 / "per_run.csv", index=False)
    assert BL.main(["--analysis-dir", str(d2)]) == 1


def test_blind_script_does_not_import_the_analysis_tool():
    """Plain numpy / pandas / stdlib only: no import of r1_analysis (or anything from tools/ms)."""
    tree = ast.parse((ROOT / "tools" / "ms" / "blind_criterion.py").read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"__future__", "argparse", "hashlib", "json", "math", "re", "sys", "pathlib", "typing", "numpy",
                    "pandas"}, mods
    src = (ROOT / "tools" / "ms" / "blind_criterion.py").read_text()
    assert "import r1_analysis" not in src and "from r1_analysis" not in src


# ====================================================================== 9. the CLI end to end
EXPECTED_FILES = ["per_run.csv", "paired_vs_parents_A.csv", "paired_vs_rehearsal_v2_0.csv", "paired_vs_MS_rule.csv",
                  "paired_seed_level.csv", "criterion.csv", "arm_summary.csv", "budget.csv", "rule.csv",
                  "rule_blocks.csv", "tail.csv", "decomposition.csv", "stage1.csv", "start_shares_by_block.csv",
                  "completeness.csv", "decision_inputs.csv", "analysis_info.json", "summary.txt"]
EXPECTED_FIGURES = ["paired_primary_abs_peak.png", "paired_signed_peak.png", "paired_primary_vs_MS_rule.png",
                    "paired_stage1_vs_rehearsal.png", "peak_error_vs_update.png", "residual_map_freeze_final.png",
                    "residual_map_freeze_development.png", "residual_map_ema_last_check.png",
                    "start_shares_by_block.png"]


def _tree_digest(root: Path) -> Dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def test_cli_end_to_end_complete_world_writes_every_output(world_ok, tmp_path, capsys):
    before = {k: _tree_digest(Path(v)) for k, v in world_ok.items()}
    out = tmp_path / "analysis"
    code = _run_cli(world_ok, out, ["--seeds", "10501-10504"])
    assert code == 0
    for f in EXPECTED_FILES:
        assert (out / f).stat().st_size > 0, f
    for f in EXPECTED_FIGURES:
        assert (out / "figures" / f).stat().st_size > 5000, f
    assert {k: _tree_digest(Path(v)) for k, v in world_ok.items()} == before          # inputs are read only
    info = json.loads((out / "analysis_info.json").read_text())
    assert info["boot_seed"] == 20261006 and info["n_boot"] == 10000 and info["all_done"] is True
    assert set(info["roots"]) == {"base", "pilot", "parents", "rehearsal"}
    assert all(v == "ok" for v in info["figures"].values()), info["figures"]
    assert info["roots"]["parents"] == str(Path(world_ok["parents"]).resolve())
    rid = info["roots_id"]
    for f in EXPECTED_FILES:
        if f.endswith(".csv"):
            assert pd.read_csv(out / f, nrows=1)["roots_id"].iloc[0] == rid, f      # the roots are in every CSV
    per = pd.read_csv(out / "per_run.csv")
    assert len(per) == (6 + 2) * 2 * 4 and (per["status"] == "done").all()
    assert list(per.columns) == ["roots_id"] + R.COLUMNS
    assert per["run_dir"].str.startswith("/").all()
    out_txt = capsys.readouterr().out
    assert "PRE-REGISTERED CRITERION" in out_txt and "DESCRIPTIVE" in out_txt
    assert R.NO_EFFECT_SENTENCE in out_txt                                       # MS_s35a0's signed CI contains 0
    assert "does not show that a mechanism has no effect" in (out / "summary.txt").read_text()
    # every non-criterion table is labelled descriptive
    for f in ("paired_vs_rehearsal_v2_0.csv", "paired_vs_MS_rule.csv", "arm_summary.csv", "budget.csv", "rule.csv",
              "tail.csv", "stage1.csv"):
        assert (pd.read_csv(out / f)["status"] == "descriptive").all(), f
    pp = pd.read_csv(out / "paired_vs_parents_A.csv")
    assert (pp[pp["metric"] == R.PRIMARY]["status"] == R.CRITERION_NOTE).all()
    assert (pp[pp["metric"] != R.PRIMARY]["status"] == "descriptive").all()
    # the blind recomputation of this very output agrees and exit code of a complete world is 0
    assert BL.main(["--analysis-dir", str(out)]) == 0
    crit = pd.read_csv(out / "criterion.csv")
    assert list(crit["arm"]) == list(R.MS_ARMS) and (crit["note"] == R.CRITERION_NOTE).all()
    assert (crit["boot_seed"] == 20261006).all() and (crit["n_boot"] == 10000).all()


def test_cli_incomplete_world_reports_the_rows_and_exits_3(cli_out, capsys):
    out, code = cli_out
    assert code == 3
    per = pd.read_csv(out / "per_run.csv")
    assert len(per) == (6 + 2) * 2 * 4                                          # no run is dropped
    assert sorted(per[per["status"] != "done"]["status"]) == ["failed", "missing"]
    comp = pd.read_csv(out / "completeness.csv")
    assert comp["n_failed"].sum() == 1 and comp["n_missing"].sum() == 1
    nd = " ".join(comp["not_done_runs"].dropna())
    assert "failed" in nd and "missing" in nd
    s = (out / "summary.txt").read_text()
    assert "MS_s35a5" in s and "planned 4, done 3" in s


def test_cli_before_any_ms_run_exists_reports_every_run_as_missing(templates, tmp_path):
    """No MS run yet (only the reference roots): every row is a missing run, every file and figure is still written,
    the criterion is incomplete (not met, not met by default), exit code 3, and the blind check agrees."""
    roots = build_world(tmp_path / "w", templates, with_failures=False)
    shutil.rmtree(roots["base"])
    shutil.rmtree(roots["pilot"])
    out = tmp_path / "analysis"
    assert _run_cli(roots, out, ["--seeds", "10501-10504"]) == 3
    per = pd.read_csv(out / "per_run.csv")
    ms = per[per["role"] == "ms_arm"]
    assert len(ms) == 6 * 2 * 4 and (ms["status"] == "missing").all() and not ms["complete"].any()
    assert (per[per["role"] == "comparator"]["status"] == "done").all()
    crit = pd.read_csv(out / "criterion.csv")
    assert (crit["overall"] == "incomplete").all() and (crit["n_pairs_q50"] == 0).all() and not crit["a_met"].any()
    assert (crit["b_n_pending"] == 7).all() and (crit["b_status"] == "incomplete").all()
    for f in EXPECTED_FILES:
        assert (out / f).stat().st_size > 0, f
    info = json.loads((out / "analysis_info.json").read_text())
    assert all(v == "ok" for v in info["figures"].values()), info["figures"]
    assert BL.main(["--analysis-dir", str(out)]) == 0


def test_cli_arm_and_q_subsets(world_ok, tmp_path):
    out = tmp_path / "sub"
    code = _run_cli(world_ok, out, ["--seeds", "10501-10504", "--arms", "MS_rule", "MS_s25a0", "--qs", "50",
                                    "--no-figures"])
    assert code == 0
    crit = pd.read_csv(out / "criterion.csv")
    assert list(crit["arm"]) == ["MS_rule", "MS_s25a0"] and "mean_q60" not in crit.columns
    per = pd.read_csv(out / "per_run.csv")
    assert set(per["arm"]) == {"MS_rule", "MS_s25a0", "parents_A", "rehearsal_v2_0"} and set(per["q"]) == {50}
    assert BL.main(["--analysis-dir", str(out)]) == 0
    assert "MS_s25a0 a50 -0.02150" in (out / "blind_recomputation.txt").read_text()


def test_cli_refuses_a_missing_reference_root_and_unknown_arms(world, tmp_path, capsys):
    argv = ["--base-root", world["base"], "--pilot-root", world["pilot"], "--parents-root", str(tmp_path / "nope"),
            "--rehearsal-root", world["rehearsal"], "--out", str(tmp_path / "o")]
    assert R.main(argv) == 2
    argv2 = ["--parents-root", world["parents"], "--rehearsal-root", world["rehearsal"], "--out", str(tmp_path / "o2"),
             "--arms", "MS_nope"]
    assert R.main(argv2) == 2


# ====================================================================== 10. read-only smoke test on real reference runs
V2 = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results")
REAL_PARENTS = V2 / "v2_refine" / "parents_A"
REAL_REHEARSAL = V2 / "v2_T2_locked" / "rehearsal_v2_0"


@pytest.mark.skipif(not (REAL_PARENTS.is_dir() and REAL_REHEARSAL.is_dir()), reason="reference roots not present")
def test_reader_on_real_reference_runs_read_only():
    """The comparator readers on three real directories: values equal the JSON; the same terminal candidate."""
    for q, seed in ((50, 10501), (60, 10503), (50, 10510)):
        pd_dir = REAL_PARENTS / ("q%d" % q) / ("seed%d" % seed)
        rd_dir = REAL_REHEARSAL / ("q%d" % q) / ("seed%d" % seed)
        a = R.extract_parents_run(pd_dir, q, seed, TH, "parents")
        b = R.extract_rehearsal_run(rd_dir, q, seed, TH, "rehearsal")
        fv = json.loads((pd_dir / "final_v2.json").read_text())
        assert a["status"] == "done" and bool(a["complete"]) and b["status"] == "done" and bool(b["complete"])
        assert a["stage2_peak_rel_err_signed"] == fv["final"]["stage2_peak_rel_err_signed"]
        assert a["stage2_peak_rel_err_abs"] == abs(a["stage2_peak_rel_err_signed"])
        assert a["eta_dev"] == fv["development"]["eta_T_over_dw"] and a["eta_T_over_dw"] == fv["final"]["eta_T_over_dw"]
        fin = fv["final"]
        assert a["G_A_pass"] == (fin["eta_T_over_dw"] <= 0.005 and fin["stage2_rmse_pos_over_g2_0"] <= 0.05
                                 and fin["stage2_tail_mean_over_g2_0"] <= 0.02)
        assert np.isnan(a["smoothed_share_peak_gap_d0"]) and np.isnan(a["stage1_rel_err_abs"])
        assert a["t2_updates"] == 1600 and a["t2_episodes"] == 819200 and a["t2_minibatch_steps"] == 32000
        g = json.loads((rd_dir / "gates.json").read_text())
        # parents_A is the end of Phase A of the rehearsal
        assert b["stage2_peak_rel_err_signed"] == a["stage2_peak_rel_err_signed"]
        sg = g["reported"]["end_of_A"]["smoothed_game"]
        assert b["smoothed_share_peak_gap_d0"] == sg["smoothed_share_peak_gap_d0"]
        assert b["stage1_rel_err_abs"] == g["reported"]["end_of_B"]["final"]["stage1_rel_err_abs"]
        assert b["learning_rel"] == g["reported"]["end_of_B"]["decomposition"]["learning_rel"]
        assert b["t1_updates"] == 600 and b["t2_updates"] == 1600 and b["t2_episodes"] == 1600 * 512
        assert bool(b["v20_combination_pass"]) == bool(g["run_pass"])


# ====================================================================== comparator D3 values from the calibration replay
def test_calibration_d3_join_reads_the_u1600_and_u2200_rows(tmp_path):
    """``calibration_d3`` takes the terminal stage at u1600 (both tiers) and stage 1 at u2200 (dev tier)."""
    import pandas as pd
    cal = tmp_path / "cal" / "rehearsal_v2_0" / "q50"
    cal.mkdir(parents=True)
    rows = []
    for u, st, tier, vals in ((1600, 2, "dev", (0.001, 68.0, 0.09, 0.009, 0.03)),
                              (1600, 2, "final", (0.0011, 68.1, 0.095, 0.0091, 0.031)),
                              (1575, 2, "dev", (9.0, 9.0, 9.0, 9.0, 9.0)),
                              (2200, 1, "dev", (3e-5, 60.0, 0.016, float("nan"), 0.027)),
                              (2175, 1, "dev", (9.0, 9.0, 9.0, 9.0, 9.0))):
        rows.append(dict(update=u, stage=st, tier=tier, Delta=vals[0], s=vals[1], R=vals[2], R_tail=vals[3], C=vals[4]))
    pd.DataFrame(rows).to_csv(cal / "seed10501.csv", index=False)
    out = R.calibration_d3(tmp_path / "cal", 50, 10501)
    assert out["t2_R_dev"] == 0.09 and out["t2_R_final"] == 0.095 and out["t2_Delta_final"] == 0.0011
    assert out["t2_Rtail_dev"] == 0.009 and out["t2_s_dev"] == 68.0 and out["t2_C_final"] == 0.031
    assert out["t1_R_dev"] == 0.016 and out["t1_Delta_dev"] == 3e-5 and "t1_R_final" not in out
    assert R.calibration_d3(tmp_path / "cal", 50, 10599) == {}
