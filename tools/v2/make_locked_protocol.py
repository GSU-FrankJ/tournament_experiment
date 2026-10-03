#!/usr/bin/env python3
"""Generate protocols/v2_T2_locked.json (the locked v2 T=2 protocol, version 1).

The per-q records are the archived as-run records already used by every v2 pilot
(tools/v2/launch_pilot.py:source_record), with only these changes: phase caps A 1600 / B 600, and
the seed / run / output_dir fields set to null (filled per run by run/run_v2_T2_locked.py). Every
other value (game, PPO, network and Beta parameterization, verifier tiers, cadence, RNG namespaces,
versions) is copied unchanged.

Usage: python tools/v2/make_locked_protocol.py   (writes protocols/v2_T2_locked.json)
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "v2"))

from launch_pilot import LINEAR_END_LR, source_record  # noqa: E402

OUT = ROOT / "protocols" / "v2_T2_locked.json"
GATES = {
    "G-A": {"stage": "end of Phase A", "candidate": "stage-2 last iterate (Beta mean of the live actor at global u1600)",
            "tier": "final", "all_must_hold": [
                {"metric": "eta_T_over_dw", "op": "<=", "threshold": 0.005,
                 "definition": "eta_2 / DW = max over the full final-tier D_2 grid of the one-step deviation gap Delta_2(d), "
                               "divided by DW (utils.dp_br_verifier.verify -> full_delta_max[2]; "
                               "utils.v2_metrics.evaluate scalar 'eta_T_over_dw')"},
                {"metric": "stage2_rmse_pos_over_g2_0", "op": "<=", "threshold": 0.05,
                 "definition": "sqrt(mean over the recovery grid nodes with |d| < 2q of (e_hat_2(d) - e2*(d))^2) / e2*(0); recovery "
                               "grid = symmetric grid on D_2 with step 'recovery_step' = 0.5 (utils.v2_metrics.recovery_metrics; "
                               "independent of the verifier tier)"},
                {"metric": "stage2_tail_mean_over_g2_0", "op": "<=", "threshold": 0.02,
                 "definition": "mean of e_hat_2(d) over the recovery grid nodes with |d| >= 2q, divided by e2*(0) "
                               "(utils.v2_metrics.recovery_metrics; independent of the verifier tier)"}]},
    "G-F": {"stage": "end of Phase B", "candidate": "full last iterate: live actor at t=1 (global u2200), frozen stage-2 snapshot at t=2",
            "tier": "final", "all_must_hold": [
                {"metric": "Gmax_full_over_dw", "op": "<=", "threshold": 0.01,
                 "definition": "max over stages t and final-tier grid nodes d of G_t(d) = V_t^BR(d) - V_t^mean(d), divided by DW "
                               "(utils.v2_metrics.evaluate scalar 'Gmax_full_over_dw')"},
                {"metric": "stage1_rel_err_abs", "op": "<=", "threshold": 0.10,
                 "definition": "|e_hat_1(0) - e1*(0)| / e1*(0) (utils.v2_metrics.recovery_metrics 'stage1_rel_err_signed', absolute value)"}]},
    "run_outcome": "pass iff G-A and G-F both pass; a run failing G-A is a stage-2 failure: its Phase B is still executed "
                   "and reported as a diagnostic, and the run counts as failed",
    "dev_tier": "every gate and reported metric is also computed on the development tier and reported; the gates use the final tier",
}
REPORTED = {
    "stage2": ["stage2_peak_rel_err_signed", "stage2_peak_locfree_rel_err (+ argmax d)", "stage2_sym_err_max", "stage2_tail_max",
               "DeltaT_over_dw_on_max", "DeltaT_over_dw_off_max", "sigma_effort_at_0_t2",
               "smoothed_share_peak_gap_d0 (Pilot-1 method: 400 equal-probability nodes per Beta at d=0)"],
    "stage1_and_full": ["EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw", "dFull_over_dw",
                        "stage-1 decomposition: learning (e_hat_1 - e~1)/e1*, inherited (e~1 - e1*)/e1*, with residual bands",
                        "sigma_effort_at_0_t1"],
    "numerics": "dev-tier minus final-tier difference of every gate and reported metric",
}


def record(q: int):
    rec, threads, src = source_record(q)
    rec = copy.deepcopy(rec)
    rec["protocol"]["phase_caps"] = {"A": 1600, "B": 600, "C": rec["protocol"]["phase_caps"]["C"]}
    rec.update(seed=None, run=None, output_dir=None)
    return rec, threads, src


def main() -> int:
    recs, srcs, threads = {}, {}, None
    for q in (50, 60):
        r, t, s = record(q)
        recs[str(q)] = r
        srcs[str(q)] = s
        threads = t
    ab = recs["50"]["lr_schedule"]["ab_lr"]
    proto = {
        "schema": "v2_T2_locked_protocol/1", "version": 1, "name": "v2 T=2 stagewise protocol (locked)",
        "date": "2026-10-02", "base_commit": "657f54a", "q_values": [50, 60],
        "pipeline": {
            "entry_point": "run/run_v2_T2_locked.py", "mode": "locked", "fixed_budget": True,
            "flags": {"reward_mode": "expected", "stage2_update_mode": "frozen", "adv_norm_scope": "stage1_rows",
                      "continuation_action_mode": "mean"},
            "flags_note": "stage2_update_mode=frozen, adv_norm_scope and continuation_action_mode take effect only in Phase B "
                          "(after the freeze); Phase A trains stage 2 alone with all rows normalized and stochastic actions",
            "phase_A": {"updates": 1600, "active_stages": [2], "starts": "bin-balanced exploring starts on D_2 (es_bin_width 10)",
                        "lr": f"constant {ab} for local updates 1-1200, then linear {ab} -> {LINEAR_END_LR} over local 1201-1600"},
            "freeze": "frozen stage-2 snapshot = deep copy of the actor at the end of Phase A (B2 machinery; no grad, no optimizer)",
            "phase_B": {"updates": 600, "active_stages": [1], "starts": "root (t=1, d=0)",
                        "adv_norm_scope": "stage1_rows", "continuation_action_mode": "mean",
                        "opponent": "lagged copy of the actor, refreshed every 20 global updates",
                        "lr": f"linear {ab} -> {LINEAR_END_LR} over local updates 1-600"},
            "lr_decay": [
                {"phase": "A", "start_lr": ab, "end_lr": LINEAR_END_LR, "local_first": 1201, "local_last": 1600},
                {"phase": "B", "start_lr": ab, "end_lr": LINEAR_END_LR, "local_first": 1, "local_last": 600}],
            "lr_form": "lr(j) = start + (end - start) * (j - local_first) / (local_last - local_first), applied to actor and "
                       "critic before each update, constant within an update, Adam state preserved "
                       "(run/run_final_dp_br_round3_dense.py:lr_at, kind 'linear'; run/run_v2_stagewise.py:Run.lr_for)",
            "not_used": ["joint training", "Phase C", "tail averaging"],
            "evaluated_candidate": "last iterate, Beta mean: stage 2 at the end of A (frozen), stage 1 at the end of B",
            "unchanged": ["RNG streams and samplers (numpy PCG64 streams from SeedSequence([seed, q, namespace]))",
                          "on-path rule: open interval |d - drift| < 2q with exact cell masses",
                          "e~1 residual-band method on the final tier"],
            "outputs": ["manifest.json (commit, clean-tree flag, protocol SHA-256, q, seed)", "gates.json",
                        "v2_checkpoints_A.csv, v2_checkpoints_B.csv + checkpoints/u*.npz (per verifier call)", "weights/u*.npz (every 25 updates)",
                        "state_end_A.pt, state_end_B.pt (full state)", "train_history.json, v2_updates.csv, v2_run_summary.json",
                        "gateA_{final,development}.npz, final_{final,development}.npz, drift_test.json, induced_band.json + sweep NPZ"],
        },
        "gates": GATES,
        "reported_not_gated": REPORTED,
        "peak_error": "not gated; reported with its decomposition (smoothed-game share; Pilot 4 section 1d floor)",
        "stage1_decomposition": {"method": "e~1 = argmin_e Delta_1(e; e_hat_2) on the final tier, band = {e: Delta_1 <= min + floor}",
                                 "floor": "max(Delta_1(e1*; e2*) on the final tier, 1e-12 DW)", "step": 0.01,
                                 "sweep_range": "anchored at e1*, covering [min(0.4 e1*, e_hat_1 - 2), max(1.7 e1*, e_hat_1 + 2)]",
                                 "functions": "utils.v2_metrics.stage1_residual_sweep / sweep_grid / induced_band"},
        "confirmation": {"status": "pre-registered; executed only in the next round after owner approval",
                         "q_values": [50, 60], "n_seeds": 20, "seeds_used_for_both_q": True,
                         "proposed_seed_block": list(range(20501, 20521)), "seed_block_status": "proposed; pending owner confirmation",
                         "disjointness_check": "tools/v2/seed_inventory.py -> results/v2_T2_locked/consolidation/seed_inventory.csv "
                                               "(0 collisions among 263 recorded seed values)",
                         "pass_rule": "for each q, at least 18 of 20 runs pass both gates (G-A and G-F)",
                         "reporting": "every run is reported, failures included; nothing changes after the lock"},
        "development_seeds": list(range(10501, 10511)),
        "records": recs,
        "record_sources": srcs,
        "threads_per_process": threads,
        "known_issues": ["tests/test_registry_canonicalization.py fails on main and on this branch (paper registry vs data on "
                         "disk; 'expected 15 Set-2 gradient runs, got 0'); pre-existing, unrelated to v2; test and data unchanged"],
        "change_log": [{"from": "0929 plan development targets (5% stage-1 error, 5% stage-2 peak error)",
                        "to": "gates G-A and G-F above; peak error reported, not gated",
                        "when": "2026-10-02, before any confirmation run",
                        "evidence": ["reports/v2/pilot4_stabilization.md section 6 (distribution tables)",
                                     "reports/v2/pilot4_stabilization.md section 2b (stability, decay vs constant)",
                                     "reports/v2/pilot4_stabilization.md section 1d (representation floor, peak-gap decomposition)"]}],
        "evidence": {
            "reward_mode=expected": "reports/v2/pilot1_reward_estimator.md",
            "frozen B2 (adv_norm_scope=stage1_rows)": "reports/v2/pilot2_freeze.md",
            "continuation_action_mode=mean": "reports/v2/pilot3_continuation_mode.md; decision D1 (Pilot 4 round)",
            "Phase A 1600 updates": "reports/v2/phaseA_ext.md; decision D2 (Pilot 4 round)",
            "Phase A end decay u1201-1600": "reports/v2/pilot4_stabilization.md section 2a",
            "Phase B decay": "reports/v2/pilot4_stabilization.md section 2b",
            "no joint training, no Phase C": "reports/v2/pilot2_freeze.md; decision D3 (Pilot 4 round)",
            "no sampler change": "reports/v2/phase2_infra.md section 2; decision D4 (Pilot 4 round)",
            "e~1 residual band": "reports/v2/pilot2_freeze.md section 6",
            "on-path rule": "reports/v2/dreach_reach_mask_check.md",
            "gate thresholds": "reports/v2/pilot4_stabilization.md section 6",
            "C7 regression": "reports/v2/phase2_infra.md section 3",
        },
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(proto, indent=1, sort_keys=False) + "\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
