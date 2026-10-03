#!/usr/bin/env python3
"""Generate protocols/v2_T2_locked_v2_0.json (and its Markdown) from the v1.1 JSON by applying ONLY
decisions D2-D5 of the v2.0 round, and write a machine diff v1.1 -> v2.0.

v1.1 (protocols/v2_T2_locked_v1_1.json, .md) is read, never written. The diff is checked against the
list of paths D2-D5 allow to change (``ALLOWED``); any other change stops the script.

Inputs read at generation time (so that no measured number is typed by hand):
  results/v2_T2_locked/v2_0/continuation_check_v2_0.json   check (ii) record (D3)
  results/v2_T2_locked/v2_0/v2_0_pass_probability.json     G-S pass probabilities (D4)
  results/v2_T2_locked/v2_0/seed_inventory_pre_lock.{csv,out}  seed inventory before the block appeared in any JSON/CSV seed record (D5)
  results/v2_refine/continuation_check.json                R1 record: the standard-tier gap quoted by D3
  utils/v2_continuation.py                                  SHA-256 at the lock, rule constants

Outputs:
  protocols/v2_T2_locked_v2_0.json
  protocols/v2_T2_locked_v2_0.md
  results/v2_T2_locked/v2_0/protocol_diff_v1_1_to_v2_0.json   (every added / removed / changed JSON path)

Usage: python tools/v2/make_locked_protocol_v2_0.py
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils import v2_continuation as vc  # noqa: E402

V11 = ROOT / "protocols" / "v2_T2_locked_v1_1.json"
V20 = ROOT / "protocols" / "v2_T2_locked_v2_0.json"
MD = ROOT / "protocols" / "v2_T2_locked_v2_0.md"
OUT = ROOT / "results" / "v2_T2_locked" / "v2_0"
DIFF = OUT / "protocol_diff_v1_1_to_v2_0.json"
CHECK = OUT / "continuation_check_v2_0.json"
R1_CHECK = ROOT / "results" / "v2_refine" / "continuation_check.json"
PASSP = OUT / "v2_0_pass_probability.json"
INV_CSV = OUT / "seed_inventory_pre_lock.csv"
INV_OUT = OUT / "seed_inventory_pre_lock.out"
MODULE = "utils/v2_continuation.py"
DATE = "2026-10-03"
SEED_BLOCK = list(range(30501, 30521))
BOOT_SEED = 20261002
DECIDED = "PI, after the R1 round and before any confirmation seed was run"
V11_SHA = "21d85983f2a2bebc0998e99fcf665c2c729f060996c529fa3162b3d62222a40f"
R1 = "reports/v2/refine"

# every JSON path D2-D5 allow to change (exact match; a list or scalar counts as one path)
ALLOWED = {
    "/schema", "/version", "/name", "/date",
    "/supersedes/version", "/supersedes/file", "/supersedes/lock_commit", "/supersedes/sha256",
    "/pipeline/flags/continuation_value_mode", "/pipeline/flags_note", "/pipeline/phase_B/continuation_value_mode",
    "/pipeline/continuation_table", "/pipeline/outputs", "/pipeline/entry_point_reads",
    "/gates/G-S", "/gates/run_outcome",
    "/confirmation/status", "/confirmation/seed_block", "/confirmation/seed_block_status",
    "/confirmation/disjointness_check", "/confirmation/pass_rule", "/confirmation/analysis_script",
    "/confirmation/crash_rule", "/confirmation/root", "/confirmation/rehearsal",
    "/v1_1_outcome_reported", "/training_relevant_state/compared",
    "/known_issues", "/change_log", "/evidence/v2.0 changes",
}


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_numbers() -> Dict[str, Any]:
    """Measured values the documents quote, read from their records."""
    chk, pp, r1 = json.load(open(CHECK)), json.load(open(PASSP)), json.load(open(R1_CHECK))
    gap = {q: float(r1["spec_requirement"]["final_tier_max_abs_diff_over_dw"][f"q{q}"]) for q in (50, 60)}
    for q in (50, 60):      # the v2.0 record re-measures the same gap on the same parents
        assert abs(gap[q] - chk["verifier_numerics"][f"q{q}"]["final"]["max_abs_gap_over_dw"]) <= 1e-12 * max(gap[q], 1e-30), q
    inv_txt = INV_OUT.read_text()
    m = re.search(r"scanned (\d+) json, (\d+) csv files; (\d+) distinct seed values", inv_txt)
    c = re.search(r"block 30501-30520: (\d+) collisions", inv_txt)
    assert m and c, "seed inventory output not in the expected form"
    with open(INV_CSV, newline="") as f:
        n_rows = sum(1 for _ in f) - 1
    assert n_rows == int(m.group(3))
    return {"chk": chk, "pp": pp, "gap": gap, "inv": {"n_json": int(m.group(1)), "n_csv": int(m.group(2)),
                                          "n_values": int(m.group(3)), "collisions": int(c.group(1))},
            "module_sha": sha256_file(ROOT / MODULE)}


def table_rule(q_list=(50, 60)) -> Dict[str, Any]:
    """Rule record built from the module constants (the entry point refuses anything else)."""
    from envs.curriculum_env import GameSpec
    n_panels, n_nodes = {}, {}
    for q in q_list:
        _, w, n = vc.shock_quadrature(q, vc.DEFAULT_PANEL_WIDTH, vc.DEFAULT_NODES_PER_PANEL)
        n_panels[str(q)], n_nodes[str(q)] = int(n), int(w.size)
    spec = GameSpec(w_h=6.0, w_l=2.0, k=1.0 / 3500.0, q=50.0, T=2, e_min=0.0, e_max=100.0)
    y = vc.y_grid_for(spec, vc.DEFAULT_STEP)
    return {
        "integration": "composite Gauss-Legendre in z against the triangular density (2q - |z|)/(4 q^2) on [-2q, 2q]; "
                       "weights include the density and sum to 1 to rounding",
        "panel_width": vc.DEFAULT_PANEL_WIDTH, "nodes_per_panel": vc.DEFAULT_NODES_PER_PANEL,
        "panel_alignment": "panels aligned at z = 0 and z = +-2q: each half interval is split into ceil(2q / panel_width) equal panels",
        "n_panels": n_panels, "n_nodes": n_nodes,
        "actor_evaluation": "the frozen actor is evaluated directly at the quadrature nodes d = y + z; g is never interpolated",
        "y_grid": {"range": "symmetric on [-e_range, e_range] (e_range = e_max - e_min = 100), contains 0",
                   "step": vc.DEFAULT_STEP, "n_nodes": int(y.size)},
        "dtype": "float64",
        "lookup": "linear interpolation; refuses y outside the grid by more than edge_tolerance x max(1, grid span)",
        "edge_tolerance": vc._EDGE_TOL,
        "max_points_per_chunk": vc._MAX_POINTS_PER_CHUNK,
        "numerical_constants_note": "edge_tolerance and max_points_per_chunk are checked by the entry point against utils/v2_continuation.py; "
                                    "the 1e-9 inside the ceil() of the panel count and of the y-grid size is a literal of the module (pinned by "
                                    "its SHA-256); the chunk size only batches the actor evaluation (measured effect below 2e-15 DW, "
                                    "results/v2_refine/continuation_check.json key batch_noise_probe)",
    }


def g(x: float, n: int = 4) -> str:
    """Compact number."""
    return f"{x:.{n}g}"


def change_log(N: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The five v2.0 entries (D1-D5; each decided by the PI after R1 and before any confirmation seed)."""
    chk, pp, inv = N["chk"], N["pp"], N["inv"]
    ia = {q: chk["ii_a"][f"q{q}"]["max_abs_change_over_dw"] for q in (50, 60)}
    ib = {q: chk["ii_b"][f"q{q}"]["max_abs_diff_over_dw"] for q in (50, 60)}
    ic = {q: chk["ii_c"][f"q{q}"] for q in (50, 60)}
    vn = {q: chk["verifier_numerics"][f"q{q}"]["final"] for q in (50, 60)}
    return [
        {"version": "2.0",
         "change": "Phase B runs with continuation_value_mode=expected: for the stage-1 rows only, the sampled continuation in the "
                   "stage-1 return is replaced by the table value V2~(y), y = e1 - e1_opp, built once at Phase-B entry from the frozen "
                   "stage-2 Beta mean by utils/v2_continuation.py with the R1 default rule (composite Gauss-Legendre, 6 nodes per panel, "
                   "panels of width 1 aligned at z = 0 and z = +-2q, y-grid step 0.05, float64, linear interpolation). Phase A is unchanged",
         "why": "the only R1 arm that met the pre-registered criterion at both q (method 6, arm B_expcont): paired difference "
                "(arm - B_base) of |relative stage-1 error| -0.02386 [-0.04061, -0.009831] (q=50) and -0.03783 [-0.06321, -0.01472] (q=60); "
                "across-seed SD of the signed error 0.3464x and 0.2687x the baseline's; mean per-pair ratio of the stage-1 advantage SD "
                "0.0656 and 0.0531; Phase-B wall time 1.084x (including the table build). No other "
                "R1 arm enters the protocol; target-KL changes cost, not accuracy, and is not adopted",
         "evidence": [f"{R1}/04_pilot_stage1.md", f"{R1}/06_decision_inputs.md section 1",
                      "results/v2_refine/analysis/stage1_criterion.csv", "results/v2_refine/analysis/stage1_dispersion.csv",
                      "results/v2_refine/analysis/stage1_adv_ratio.csv", "results/v2_refine/analysis/stage1_cost.csv",
                      f"{R1}/01_preregistration.md section 4 item 6 and Addendum 1"],
         "decided_by": DECIDED},
        {"version": "2.0",
         "change": "the pipeline code changed at 32a8c21 (R1 flags, every new key at its default behaves as before)",
         "why": "needed to run the expected continuation; v1.1 is unaffected: the unchanged v1.1 entry point reproduces rehearsal_v1_1 "
                "in 20/20 runs on the changed code (C-R1), and parents_A, B_base and A_base reproduce it in 20/20 each",
         "evidence": ["results/v2_refine/v11_reproduction_checks.json", "results/v2_refine/parents_A_checks.json",
                      "results/v2_refine/stage1_base_checks.json", "results/v2_refine/stage2_base_checks.json",
                      "results/v2_refine/c7/code_check.compare.txt"],
         "decided_by": DECIDED},
        {"version": "2.0",
         "change": "check (ii) of the continuation table settled: the literal R1 criterion (<= 1e-6 DW against the verifier's standard "
                   "final tier) is replaced by (ii-a) self-convergence <= 1e-8 DW, (ii-b) refined-verifier agreement <= 1e-6 DW "
                   "(state step 0.25, 64 Gauss-Legendre nodes) and (ii-c) training-side sensitivity: shift of the stage-1 optimum "
                   "<= 1e-3 e1* when the convergence difference profile is added",
         "why": f"the standard final tier interpolates V2 onto a state grid of step 2; the gap it shows ({g(N['gap'][50], 4)} DW at q=50, "
                f"{g(N['gap'][60], 4)} DW at q=60) is of the order of the dev - final differences G-N exists for, while the table is converged: "
                f"(ii-a) {g(ia[50], 3)} / {g(ia[60], 3)} DW, (ii-b) {g(ib[50], 3)} / {g(ib[60], 3)} DW, (ii-c) shift of the stage-1 "
                f"optimum {g(ic[50]['fixed_opponent']['shift_over_e1_star'], 3)} / {g(ic[60]['fixed_opponent']['shift_over_e1_star'], 3)} "
                f"e1* (q=50 / q=60; limit 1e-3). Verifier numerics item (reported, not gated): the standard-tier gap "
                f"{g(vn[50]['max_abs_gap_over_dw'], 4)} / {g(vn[60]['max_abs_gap_over_dw'], 4)} DW would imply a stage-1 optimum shift of "
                f"{g(vn[50]['implied_shift_over_e1_star'], 3)} / {g(vn[60]['implied_shift_over_e1_star'], 3)} e1* if it were a table error",
         "evidence": ["results/v2_T2_locked/v2_0/continuation_check_v2_0.json", "results/v2_T2_locked/v2_0/verifier_numerics_note.md",
                      "results/v2_refine/continuation_check.json", "tests/test_v2_refine_continuation.py"],
         "decided_by": DECIDED},
        {"version": "2.0",
         "change": "gate G-S added: |e_hat_1(0) - e1*(0)| / e1*(0) <= 0.05 on the end-of-B last iterate (the v1.1 secondary criterion S1 "
                   "with threshold 0.05, the 0929 target); run pass = G-A and G-F and G-N and G-S",
         "why": f"threshold evidence from the locked pipeline's own distribution before any confirmation seed: arm B_expcont on the "
                f"development seeds (the v2.0 rehearsal must reproduce it bit-identically: check R2): "
                f"{pp['per_q']['50']['n_within_threshold']}/{pp['per_q']['50']['n']} and "
                f"{pp['per_q']['60']['n_within_threshold']}/{pp['per_q']['60']['n']} within 0.05, max |error| "
                f"{g(pp['per_q']['50']['max_abs_err'])} (q=50) and {g(pp['per_q']['60']['max_abs_err'])} (q=60), signed error mean "
                f"{g(pp['per_q']['50']['mean_signed'])} / {g(pp['per_q']['60']['mean_signed'])}, SD (ddof=1) "
                f"{g(pp['per_q']['50']['sd_ddof1'])} / {g(pp['per_q']['60']['sd_ddof1'])}. Normal model (G-A, G-F, G-N treated as passing): "
                f"P(>= 18 of 20 fresh runs pass G-S) = {g(pp['per_q']['50']['P_ge18_of_20_normal_ddof1'])} (q=50) and "
                f"{g(pp['per_q']['60']['P_ge18_of_20_normal_ddof1'])} (q=60). S1 at 0.10, the v1.1 outcome and the v1.0 outcome are "
                f"reported for every run and decide nothing. Safety condition on the v2.0 rehearsal: >= 19/20 under G-S and a "
                f"recomputed normal-model probability >= 0.90 at each q",
         "evidence": ["results/v2_T2_locked/v2_0/v2_0_pass_probability.json (tools/v2/v2_0_pass_probability.py)",
                      "results/v2_refine/analysis/stage1_per_run.csv"],
         "decided_by": DECIDED},
        {"version": "2.0",
         "change": "confirmation seed block 30501-30520, used for both q (40 runs); pass rule per q >= 18 of 20 runs pass (G-A, G-F, G-N, G-S), "
                   "both q, exact Clopper-Pearson 95% CIs",
         "why": f"seeds reserved and unused; the inventory of every seed value recorded in the tournament_experiment tree "
                f"({inv['n_values']} distinct values in {inv['n_json']} json and {inv['n_csv']} csv files) found {inv['collisions']} collisions "
                f"with the block, taken before the block appeared in any JSON or CSV seed record (the block had been reserved and unused since the R1 "
                f"pre-flight: reports/v2/refine/00_preflight.md, results/v2_refine/preflight/seed_inventory_30501_30520.out)",
         "evidence": ["results/v2_T2_locked/v2_0/seed_inventory_pre_lock.csv", "results/v2_T2_locked/v2_0/seed_inventory_pre_lock.out",
                      "tools/v2/seed_inventory.py"],
         "decided_by": DECIDED},
    ]


def apply_changes(p: Dict[str, Any], N: Dict[str, Any]) -> Dict[str, Any]:
    """v1.1 -> v2.0: D2-D5 only."""
    p = copy.deepcopy(p)
    inv = N["inv"]
    # ---- identity
    p["schema"] = "v2_T2_locked_protocol/2.0"
    p["version"] = "2.0"
    p["name"] = "v2 T=2 stagewise protocol (locked, v2.0)"
    p["date"] = DATE
    p["supersedes"] = {"version": "1.1", "file": "protocols/v2_T2_locked_v1_1.json", "lock_commit": "431474d", "sha256": V11_SHA}
    # ---- D2: expected continuation in Phase B
    pl = p["pipeline"]
    pl["flags"]["continuation_value_mode"] = "expected"
    pl["flags_note"] = (pl["flags_note"] + "; continuation_value_mode=expected also takes effect only in Phase B: the stage-1 rows use the "
                        "shock-integrated table value of the frozen stage-2 policy instead of the sampled continuation "
                        "(requires frozen + mean + gamma = lambda = 1)")
    pl["phase_B"]["continuation_value_mode"] = "expected"
    pl["continuation_table"] = {
        "mode": "expected", "stage": 2,
        "applies_to": "Phase B, stage-1 rows only: stage-1 return = r_1 + V2~(y), y = e_1 - e_1^opp (both efforts as executed, float64); "
                      "stage-1 advantage = that return - V(s_1); stage-2 rows, their critic targets and every random draw are unchanged "
                      "(shocks are still drawn); the opponent's stage-1 action stays sampled",
        "implemented_at_commit": "32a8c21", "run_as_pilot_arm": f"B_expcont ({R1}/01_preregistration.md section 4 item 6)",
        "definition": "V2~(y) = E_z[g_2(y + z)], g_2(d) = w_l + DW F_xi(d + e_hat_2(d) - e_hat_2(-d)) - k e_hat_2(d)^2, e_hat_2 the frozen "
                      "stage-2 Beta mean on the float path of continuation_action_mode=mean, z = eps_L - eps_O with density "
                      "(2q - |z|)/(4 q^2) on [-2q, 2q]",
        "built": "once at Phase-B entry from the frozen stage-2 snapshot, by run/run_v2_stagewise.py:Run.run_phase calling "
                 "utils.v2_continuation.build_continuation_table (step = run_v2_stagewise.CONT_TABLE_STEP, default rule); "
                 "single-threaded, no RNG draw",
        "module": MODULE, "module_sha256_at_lock": N["module_sha"],
        "rule": table_rule(),
        "entry_point_refuses_other_values": "the JSON rule must equal run/run_v2_T2_locked.py:LOCKED_TABLE_RULE, utils/v2_continuation.py "
                                            "defaults and run_v2_stagewise.CONT_TABLE_STEP must equal it (numerical parameters compared: panel_width, nodes_per_panel, "
                                            "y step, stage, dtype, edge_tolerance, max_points_per_chunk), and the table that is built is "
                                            "checked against it at the first Phase-B update",
        "table_file": "continuation_table.npz in the run directory: arrays y_grid and values (float64), written as an NPZ with fixed zip "
                      "timestamps so that equal tables give equal file hashes",
        "recorded_in": "manifest.json (key continuation_table) and gates.json (key continuation_table): rule parameters of the table that "
                       "was built, build seconds, SHA-256 of continuation_table.npz",
        "rng": "the build draws from no RNG: the five training streams, the torch generator and the three process-global RNGs are "
               "compared between Phase-B entry and the first Phase-B update; a moved training stream aborts the run, a moved global RNG "
               "is a global-RNG violation (exit code 5)",
        "check_ii": {"settled_by": "D3 of the v2.0 round", "tests": "tests/test_v2_refine_continuation.py",
                     "record": "results/v2_T2_locked/v2_0/continuation_check_v2_0.json",
                     "parents": "end-of-A states of the v1.1 rehearsal, seed 10501, both q (SHA-256 in the record)",
                     "fixed_stage1_opponent_mean": {"50": 40.0, "60": 45.0},
                     "finest_rule": {"panel_width": 0.25, "nodes_per_panel": 24},
                     "ii_a": "halving the panel width and doubling the nodes per panel changes the table by <= 1e-8 DW",
                     "ii_b": "refined verifier (state step 0.25, 64 Gauss-Legendre nodes), both q: max over the effort grid of "
                             "|V2~(e - e1_hat(0)) - (Q1(0, e) + k e^2)| <= 1e-6 DW",
                     "ii_c": "adding the convergence difference profile (finest rule minus default rule) to the table moves "
                             "argmax_e [-k e^2 + V2~(e - e1_hat(0))] on a 0.001-effort grid by <= 1e-3 e1*"},
    }
    pl["outputs"] = pl["outputs"] + [
        "continuation_table.npz (Phase B table: y_grid, values)",
        "gates.json (v2.0; replaces the v1.1 entry above): G-A, G-F, G-N, G-S, run_pass (v2.0), S1 (0.10), v1_1_outcome, v1_0_outcome, continuation_table, global_rng, "
        "protocol_version 2.0",
        "manifest.json (v2.0; replaces the v1.1 entry above): continuation-table record (rule, build seconds, NPZ SHA-256)"]
    pl["entry_point_reads"] = "protocols/v2_T2_locked_v2_0.json"
    # ---- D4: gate G-S, run pass
    p["gates"]["G-S"] = {
        "stage": "end of Phase B", "candidate": "stage-1 last iterate: Beta mean of the live actor at t=1, d=0 (global u2200)",
        "tier": "final",
        "all_must_hold": [{"metric": "stage1_rel_err_abs", "op": "<=", "threshold": 0.05,
                           "definition": "|e_hat_1(0) - e1*(0)| / e1*(0) (utils.v2_metrics.recovery_metrics 'stage1_rel_err_signed', absolute "
                                         "value); the v1.1 secondary criterion S1 with the threshold 0.05 (the 0929 target); e1* is the closed "
                                         "form and enters the evaluation only"}]}
    p["gates"]["run_outcome"] = ("pass iff G-A, G-F, G-N and G-S all pass; a run failing G-A is a stage-2 failure: its Phase B is still "
                                 "executed and reported, and the run counts as failed; a run with a global-RNG violation (hardening) or any "
                                 "nonzero exit counts as failed; S1 at 0.10, the v1.1 outcome (G-A, G-F, G-N) and the v1.0 outcome are "
                                 "reported only")
    # ---- D4/D5: confirmation design
    c = p["confirmation"]
    c["status"] = "pre-registered; executed in the v2.0 round only if the re-rehearsal checks R1-R7 (results/v2_T2_locked/rehearsal_v2_0_checks.json) all pass"
    c["seed_block"] = list(SEED_BLOCK)
    c["seed_block_status"] = "set by the PI (v2.0 round, D5); used for both q"
    c["disjointness_check"] = (f"tools/v2/seed_inventory.py before the block appeared in any JSON/CSV seed record: results/v2_T2_locked/v2_0/"
                               f"seed_inventory_pre_lock.csv ({inv['collisions']} collisions among {inv['n_values']} recorded seed values); re-run at the "
                               f"lock commit with the files that declare the block excluded (--exclude): results/v2_T2_locked/v2_0/seed_inventory.csv; the stock scan at "
                               f"the lock commit finds only those declarations: results/v2_T2_locked/v2_0/seed_inventory_stock.out")
    c["pass_rule"] = ("for each q, at least 18 of 20 runs pass (G-A and G-F and G-N and G-S); the confirmation passes iff both q pass; "
                      "each q's pass rate is reported with an exact Clopper-Pearson 95% CI")
    c["analysis_script"] = "tools/v2/confirmation_analysis_v2_0.py"
    c["crash_rule"] = ("infrastructure kill (OOM, disk full, process killed): move the run directory to confirmation_v2_0/crashed/"
                       "q{q}/seed{s}_attempt1/ and re-run once into the original path; a second infrastructure failure stops "
                       "the round and that run is not counted either way; any other nonzero exit (pipeline exception, "
                       "global-RNG violation) counts as a failed run")
    c["root"] = "results/v2_T2_locked/confirmation_v2_0"
    c["rehearsal"] = {
        "root": "results/v2_T2_locked/rehearsal_v2_0", "q_values": [50, 60], "seeds": "development_seeds (10501-10510)",
        "safety_condition": {
            "G-S_pass_at_least_of_20": 19, "normal_model_probability_at_least_per_q": 0.90,
            "normal_model": "signed stage-1 error ~ N(mean, SD) fitted per q to that q's 10 rehearsal runs (SD with ddof = 1); a run passes "
                            "G-S iff |error| <= the G-S threshold; G-A, G-F and G-N treated as passing; probability = "
                            "P(Binomial(20, p_q) >= 18)"}}
    # ---- reported-only outcomes, training-relevant state
    p["v1_1_outcome_reported"] = {"definition": "G-A and G-F and G-N (the v1.1 run outcome)", "role": "reported only; decides nothing"}
    trs = p["training_relevant_state"]
    trs["compared"] = trs["compared"] + ["the continuation table (continuation_table.npz, Phase B): y_grid and values, bit for bit"]
    # ---- known issues, change log, evidence
    p["known_issues"] = p["known_issues"] + [
        "Phase A clamp flags (R1 diagnostic D1): M1 is exceeded in Phase A (median over runs of the clamp fraction among learner policy rows "
        "0.001385 pooled over both q, hits only on rows with |d| >= 2q of the final stage) and not in Phase B; M2 is exceeded in Phase A "
        "in the pooled reading (3 of 20 runs with a clamped-row gradient share above 1%, maximum 0.02353) and not in Phase B. "
        "Recorded, not fixed (Phase A is unchanged in v2.0): results/v2_refine/d1_clamp/d1_flags.csv, reports/v2/refine/02_d1_clamp.md",
        "verifier numerics (check (ii)): the verifier's standard final tier carries a stage-1 Q interpolation gap against the table of "
        f"{g(N['gap'][50], 4)} DW (q=50) and {g(N['gap'][60], 4)} DW (q=60); it bounds the precision of the verifier's own e~1 on that tier; "
        "reported, not gated: results/v2_T2_locked/v2_0/verifier_numerics_note.md",
        "utils/v2_continuation.py is byte-identical to its version at 32a8c21; its module docstring still calls check (ii) open "
        "(it predates decision D3) - protocols/v2_T2_locked_v2_0.md and results/v2_T2_locked/v2_0/continuation_check_v2_0.json supersede it"]
    p["change_log"] = p["change_log"] + change_log(N)
    p["evidence"]["v2.0 changes"] = (f"{R1}/06_decision_inputs.md; {R1}/04_pilot_stage1.md; {R1}/summary.md; "
                                     "reports/v2/protocol_v2_0_confirmation.md")
    return p


def diff(a: Any, b: Any, path: str = "") -> List[Dict[str, Any]]:
    """Every added / removed / changed JSON path (lists and scalars are leaves)."""
    out: List[Dict[str, Any]] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            pk = f"{path}/{k}"
            if k not in b:
                out.append({"op": "removed", "path": pk, "old": a[k]})
            elif k not in a:
                out.append({"op": "added", "path": pk, "new": b[k]})
            else:
                out += diff(a[k], b[k], pk)
    elif a != b:
        out.append({"op": "changed", "path": path, "old": a, "new": b})
    return out


def render_md(p: Dict[str, Any], N: Dict[str, Any]) -> str:
    """The Markdown of the protocol (the JSON is what runs)."""
    chk, pp, inv = N["chk"], N["pp"], N["inv"]
    ct = p["pipeline"]["continuation_table"]
    r = ct["rule"]
    ia = {q: chk["ii_a"][f"q{q}"]["max_abs_change_over_dw"] for q in (50, 60)}
    ib = {q: chk["ii_b"][f"q{q}"]["max_abs_diff_over_dw"] for q in (50, 60)}
    ic = {q: chk["ii_c"][f"q{q}"] for q in (50, 60)}
    vn = {q: chk["verifier_numerics"][f"q{q}"]["final"] for q in (50, 60)}
    L = [
        "# v2 T=2 protocol — locked, version 2.0", "",
        f"Date: {DATE}.", "",
        "- **Machine-readable source:** `protocols/v2_T2_locked_v2_0.json`. It is generated by `tools/v2/make_locked_protocol_v2_0.py` from "
        "`protocols/v2_T2_locked_v1_1.json` (v1.1) by applying only decisions D2–D5 of the v2.0 round.",
        "- **Machine diff v1.1 → v2.0:** `results/v2_T2_locked/v2_0/protocol_diff_v1_1_to_v2_0.json`.",
        "- **Entry point:** `run/run_v2_T2_locked.py`. It reads only the v2.0 JSON and embeds its hash. It refuses a modified protocol, a "
        "continuation-table rule that differs from the locked one, any argument other than `--q`, `--seed`, `--out-dir`, and any q ∉ {50, 60}.",
        "- **Pre-registered analysis:** `tools/v2/confirmation_analysis_v2_0.py`, locked with this version.",
        "- **Lock record:** `protocols/LOCK` (v2.0 record appended; the v1.0 and v1.1 records are kept verbatim).", "",
        "v1.1 (`protocols/v2_T2_locked_v1_1.json`, `protocols/v2_T2_locked_v1_1.md`) is unchanged and remains reproducible from its lock "
        "commit `431474d`; v1.0 from `4bd2214`.", "",
        "**Unchanged from v1.1** (see `protocols/v2_T2_locked_v1_1.md` and `protocols/v2_T2_locked.md` for the full settings):",
        "- Phase A in full (the end-of-A training-relevant state must be bit-identical to v1.1: re-rehearsal check R1);",
        "- the Phase-B flags `expected / frozen / stage1_rows / mean`, the budgets 1600 / 600 and the LR decay;",
        "- the gates G-A, G-F, G-N and their thresholds;",
        "- the verifier and its tiers; all metric definitions; the on-path rule; the ẽ₁ residual-band method;",
        "- the reported-not-gated list; the evaluated candidate (the last iterate);",
        "- the process-global RNG hardening; the bootstrap settings;",
        "- every per-q record.", "",
        "If this document and the JSON ever disagree, the JSON is what runs.", "", "---", "",
        "## 1. Gates (primary; decide the run)", "",
        "All values are on the **final tier** unless stated otherwise. Every \"≤\" is **inclusive**, compared at full float64 precision with "
        "no rounding. Metrics come from `utils.v2_metrics.evaluate`.", "",
        "| Gate | When / candidate | Criterion | Definition |", "|---|---|---|---|",
        "| **G-A** (unchanged) | End of Phase A; stage-2 last iterate | `eta_T_over_dw ≤ 0.005` | max over the full final-tier D₂ grid of Δ₂(d), /ΔW |",
        "| | | `stage2_rmse_pos_over_g2_0 ≤ 0.05` | RMSE of ê₂ − e₂* over recovery-grid nodes with \\|d\\| < 2q, /e₂*(0) |",
        "| | | `stage2_tail_mean_over_g2_0 ≤ 0.02` | mean of ê₂ over recovery-grid nodes with \\|d\\| ≥ 2q, /e₂*(0) |",
        "| **G-F** (unchanged) | End of Phase B; full last iterate (live t=1, frozen t=2) | `Gmax_full_over_dw ≤ 0.01` | max over t and the final-tier grid of V_t^BR − V_t^mean, /ΔW |",
        "| **G-N** (unchanged) | η₂ at the end of A; Ĝmax at the end of B | \\|η₂(dev) − η₂(final)\\|/ΔW ≤ 0.001 | the G-A η₂, development vs final tier |",
        "| | | \\|Ĝmax_full(dev) − Ĝmax_full(final)\\|/ΔW ≤ 0.001 | the G-F Ĝmax, development vs final tier |",
        "| **G-S** (new) | End of Phase B; stage-1 last iterate ê₁(0) | `stage1_rel_err_abs ≤ 0.05` | \\|ê₁(0) − e₁*(0)\\|/e₁*(0): the v1.1 secondary criterion S1 with the threshold 0.05 (the 0929 target) |", "",
        "**Run pass** = G-A ∧ G-F ∧ G-N ∧ G-S. In addition, a run with a global-RNG violation (§5) or any other nonzero exit counts as failed.",
        "- A G-A failure is recorded as a **stage-2 failure**. That run's Phase B still runs and is reported, and the run counts as failed.", "",
        "## 2. Reported only (decide nothing)", "",
        "- **S1 at 0.10:** |ê₁(0) − e₁*(0)|/e₁*(0) ≤ 0.10, value and pass or fail per run (the v1.1 secondary criterion, unchanged).",
        "- **v1.1 outcome:** G-A ∧ G-F ∧ G-N.",
        "- **v1.0 outcome:** G-A ∧ v1.0 G-F, where v1.0 G-F = Ĝmax_full/ΔW ≤ 0.01 ∧ S1 at 0.10.",
        "- **Per q (S1 at 0.05 and at 0.10):** the pass count with an exact Clopper–Pearson 95% CI; the mean signed stage-1 relative error with a 95% "
        "percentile bootstrap CI; the median and the SD (ddof = 1).", "",
        "## 3. Pass rule, confirmation design, bootstrap", "",
        "- q ∈ {50, 60}.",
        "- Seed block **30501–30520**, set by the PI; the same 20 seeds are used for both q (40 runs). The inventory of every recorded seed value "
        f"found {inv['collisions']} collisions with the block ({inv['n_values']} distinct values; `results/v2_T2_locked/v2_0/seed_inventory_pre_lock.csv`, "
        "taken before the block appeared in any JSON or CSV seed record — the block had been reserved and unused since the R1 pre-flight; re-run at the lock "
        "commit with the declaring files excluded (`--exclude`): `results/v2_T2_locked/v2_0/seed_inventory.csv`; the stock scan at the lock commit finds only "
        "those declarations: `results/v2_T2_locked/v2_0/seed_inventory_stock.out`).",
        "- **Pass rule:** for each q, at least **18 of 20** runs pass (§1). The confirmation passes if and only if both q pass. Each q's pass rate is "
        "reported with an exact Clopper–Pearson 95% CI.",
        f"- **Bootstrap:** 10,000 percentile resamples of the runs of one q; seed **{BOOT_SEED}**; one fresh `numpy.random.default_rng({BOOT_SEED})` per "
        "(q, statistic) in table order, q ascending.",
        "- **Rehearsal safety condition (before the confirmation):** the v2.0 re-rehearsal on the development seeds 10501–10510 must show ≥ 19 of 20 runs "
        "under G-S, and the normal-model probability that ≥ 18 of 20 fresh runs pass G-S (signed error ~ N(mean, SD, ddof = 1) fitted per q; "
        "G-A, G-F, G-N treated as passing) must be ≥ 0.90 at each q.",
        "- **Crashes:**",
        "  - Infrastructure kill (out of memory, disk full, process killed): move the run directory to `confirmation_v2_0/crashed/q{q}/seed{s}_attempt1/` and "
        "re-run once into the original path. If the re-run also fails for an infrastructure reason, the round stops and that run is not counted either way.",
        "  - Any other nonzero exit (a pipeline exception, or a global-RNG violation) counts as a failed run.",
        "- Every run is reported, failures included. Nothing changes after the lock.", "",
        "## 4. Expected continuation in Phase B (the one pipeline change)", "",
        "Phase B runs with `continuation_value_mode = expected`, exactly as implemented at `32a8c21` and run as the R1 arm `B_expcont`. For the "
        "stage-1 rows only, the sampled continuation in the stage-1 return is replaced by the table value Ṽ₂(y), y = e₁ − e₁^opp (both efforts as "
        "executed, float64); the stage-1 advantage is that return minus V(s₁). Stage-2 rows, their critic targets and every random draw are unchanged "
        "(shocks are still drawn); the opponent's stage-1 action stays sampled. Valid only with frozen stage 2, mean continuation actions and γ = λ = 1.", "",
        "**Table definition.** Ṽ₂(y) = E_z[g₂(y + z)], g₂(d) = w_L + ΔW·F_ξ(d + ê₂(d) − ê₂(−d)) − k·ê₂(d)², ê₂ the frozen stage-2 Beta mean on the "
        "float path of `continuation_action_mode = mean`, z = ε_L − ε_O with density (2q − |z|)/(4q²) on [−2q, 2q]. Built once at Phase-B entry from "
        "the frozen stage-2 snapshot by `utils/v2_continuation.py` (SHA-256 at the lock: "
        f"`{N['module_sha']}`), single-threaded, no RNG draw.", "",
        "**Integration rule (fixed; the entry point refuses any other value).**",
        f"- z-integral: {r['integration']};",
        f"- composite Gauss-Legendre with **{r['nodes_per_panel']} nodes per panel** on panels of **width {r['panel_width']:g}**, "
        f"aligned at z = 0 and z = ±2q ({r['n_panels']['50']} panels / {r['n_nodes']['50']} nodes at q = 50, {r['n_panels']['60']} panels / "
        f"{r['n_nodes']['60']} nodes at q = 60);",
        "- the frozen actor is evaluated directly at the nodes d = y + z; g₂ is never interpolated;",
        f"- y-grid symmetric on [−e_range, e_range] = [−100, 100], **step {r['y_grid']['step']:g}** ({r['y_grid']['n_nodes']} nodes), float64, linear "
        f"interpolation at lookup (refuses y outside the grid by more than {r['edge_tolerance']:g} × max(1, grid span));",
        f"- numerical constants of the module that the entry point also compares: edge tolerance {r['edge_tolerance']:g}, actor-evaluation chunk size "
        f"{r['max_points_per_chunk']} points (a batching detail, measured effect below 2e-15·ΔW). The panel width, nodes per panel, y step, stage and dtype are "
        "compared as well; the descriptive fields of the JSON rule (panel counts, wording) are not machine-checked.", "",
        "**Run record.** `continuation_table.npz` (arrays `y_grid`, `values`; fixed zip timestamps) in the run directory; the rule parameters of the table "
        "that was built, the build seconds and the NPZ's SHA-256 are in `manifest.json` and `gates.json`. The build must not move any RNG: the five training "
        "streams, the torch generator and the three global RNGs are compared between Phase-B entry and the first Phase-B update.", "",
        "**Check (ii), settled (decision D3).** The literal R1 criterion (≤ 1e-6·ΔW against the verifier's standard final tier) is withdrawn: that tier "
        f"interpolates V₂ onto a state grid of step 2 and its gap ({g(N['gap'][50], 4)}·ΔW at q = 50, {g(N['gap'][60], 4)}·ΔW at q = 60) is of the order of the dev − final "
        "differences G-N exists for. Replaced by three tests (`tests/test_v2_refine_continuation.py`; record "
        "`results/v2_T2_locked/v2_0/continuation_check_v2_0.json`):", "",
        "| test | criterion | q = 50 | q = 60 |", "|---|---|---|---|",
        f"| (ii-a) self-convergence | halving the panel width and doubling the nodes changes the table by ≤ 1e-8·ΔW | {g(ia[50], 3)} | {g(ia[60], 3)} |",
        f"| (ii-b) refined verifier (state step 0.25, 64 GL nodes) | max over the effort grid of \\|Ṽ₂(e − ê₁(0)) − (Q₁(0, e) + k e²)\\| ≤ 1e-6·ΔW | {g(ib[50], 3)} | {g(ib[60], 3)} |",
        f"| (ii-c) training-side sensitivity | shift of argmax_e[−k e² + Ṽ₂(e − ê₁(0))] on a 0.001 effort grid when the finest-minus-default profile is added ≤ 1e-3·e₁* | "
        f"{g(ic[50]['fixed_opponent']['shift'], 3)} = {g(ic[50]['fixed_opponent']['shift_over_e1_star'], 3)}·e₁* | "
        f"{g(ic[60]['fixed_opponent']['shift'], 3)} = {g(ic[60]['fixed_opponent']['shift_over_e1_star'], 3)}·e₁* |", "",
        "Verifier numerics item (reported, not gated; `results/v2_T2_locked/v2_0/verifier_numerics_note.md`): the standard-tier gap "
        f"{g(vn[50]['max_abs_gap_over_dw'], 4)}·ΔW (q = 50) / {g(vn[60]['max_abs_gap_over_dw'], 4)}·ΔW (q = 60) would, if it were a table error, shift the "
        f"stage-1 optimum by {g(vn[50]['implied_shift'], 3)} / {g(vn[60]['implied_shift'], 3)} effort units "
        f"({g(vn[50]['implied_shift_over_e1_star'], 3)}·e₁* / {g(vn[60]['implied_shift_over_e1_star'], 3)}·e₁*).", "",
        "## 5. Hardening of the process-global RNGs (unchanged; plus one digest at Phase-B entry and the table-build comparison of section 4)", "",
        "The three process-global RNGs are torch global, numpy legacy global and Python `random`.", "",
        "- **Seeding.** At the start of `run_v2_T2_locked.run_pipeline`, before any object is constructed: `torch.manual_seed(seed)`, `np.random.seed(seed)`, "
        "`random.seed(seed)`, with the run seed. The seeds are recorded in the manifest.",
        "- **Reference state.** Each RNG's state immediately before the first Phase A update (`Run.phase_start_hook` at the entry of Phase A, after the "
        "phase-entry snapshot refresh). The post-seed state and the state after `Run` construction are also recorded. Known draw: PyTorch's default "
        "`nn.Linear` initializer draws from the torch global RNG while `BetaActor` and `Critic` are constructed; their weights are then re-initialized from "
        "the explicit torch generator (`results/v2_T2_locked/v1_1/global_rng_trace.json`).",
        "- **Assertion.** Each state must equal its reference at the end of A, after G-A, at the end of B, and at the end of the run. On a violation, the RNG and "
        "the point are recorded in `gates.json`, the run finishes and writes all outputs, the entry point exits with code **5**, and the run counts as failed.",
        "- **Digests.** The manifest records the SHA-256 of each RNG state at seeding, after `Run` construction, at the reference point, at Phase-B entry, at the "
        "start of Phase B and at each assertion point.",
        "- **Training is unchanged by the hardening.**", "",
        "## 6. Training-relevant state (every bit-identity check; unchanged, now including the table)", "",
        "**Compared:**",
        "- actor, critic, lagged opponent, frozen stage-2 snapshot (Phase B), and both Adam states;",
        "- the five training RNG streams (env, learn, opp, start, minibatch) and the torch generator;",
        "- all weight exports and the final weights;",
        "- training histories and per-update logs, without their wall-clock fields;",
        "- checkpoint metrics, gate-metric values, and the evaluation outputs (`gateA_*.npz`, `final_*.npz`, `induced_band.json`, `band_sweep.npz`, `drift_test.json`);",
        "- the continuation table (`continuation_table.npz`, Phase B): `y_grid` and `values`, bit for bit.", "",
        "**Excluded:** verdict fields, manifests, and the three process-global RNG states.", "",
        "## 7. Change log, v1.1 → v2.0", "",
        "**The PI decided every change after the R1 round and before any confirmation seed was run.**", ""]
    for i, e in enumerate(p["change_log"][-5:], 1):
        L += [f"{i}. **{e['change']}**", f"   - Why: {e['why']}.", "   - Evidence: " + "; ".join(f"`{x}`" for x in e["evidence"]) + ".", ""]
    L += ["## 8. Known issues at the lock", ""] + [f"- {k}" for k in p["known_issues"]] + [""]
    L += ["## 9. How to run", "", "```bash",
          "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python "
          "run/run_v2_T2_locked.py --q 50 --seed 30501 --out-dir results/v2_T2_locked/confirmation_v2_0/q50/seed30501", "```", "",
          "```bash",
          "/home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/confirmation_analysis_v2_0.py --root results/v2_T2_locked/confirmation_v2_0 "
          "--out results/v2_T2_locked/confirmation_v2_0_analysis", "```", ""]
    return "\n".join(L)


def render_numerics_note(N: Dict[str, Any]) -> str:
    """results/v2_T2_locked/v2_0/verifier_numerics_note.md: the standard-tier gap item of decision D3."""
    chk = N["chk"]
    rows = []
    for q in (50, 60):
        for tier in ("final", "development"):
            v = chk["verifier_numerics"][f"q{q}"][tier]
            rows.append(f"| {tier} (state step {v['state_step']:g}, {v['gl_half']} GL nodes per half interval) | {q} | {g(v['max_abs_gap_over_dw'], 4)} | "
                        f"{v['argmax_gap_effort']:g} | {v['argmax_table']:.3f} | {v['argmax_if_gap_were_table_error']:.3f} | {v['implied_shift']:+.3f} | "
                        f"{g(v['implied_shift_over_e1_star'], 3)} |")
    ic = {q: chk["ii_c"][f"q{q}"] for q in (50, 60)}
    return "\n".join([
        "# Verifier numerics item (check (ii), decision D3)", "",
        "**Reported, not gated.** Source: `results/v2_T2_locked/v2_0/continuation_check_v2_0.json` (key `verifier_numerics`), written by "
        "`tests/test_v2_refine_continuation.py --write` (`verifier_numerics_item`). Parents: the end-of-A states of the v1.1 rehearsal, seed 10501, "
        "both q (`parents` in the record, with SHA-256).", "",
        "**What is measured.** With the stage-1 opponent fixed at 40 (q = 50) / 45 (q = 60), the verifier's stage-1 value Q₁(0, e) + k e² is compared "
        "with the table Ṽ₂(e − ê₁(0)) on the verifier's effort grid; δ(e) is their difference (units of ΔW = 4). If δ were an error of the table, the "
        "stage-1 objective −k e² + Ṽ₂(e − ê₁(0)) + δ(e) (δ linearly interpolated onto a 0.001 effort grid) would have its maximiser at the "
        "verifier's own stage-1 optimum instead of the table's. The shift of that maximiser is the precision with which the verifier, on that tier, "
        "can locate the stage-1 optimum ẽ₁ that the training objective defines.", "",
        "| verifier tier | q | max \\|δ\\| / ΔW | at effort | stage-1 optimum, table | optimum if δ were a table error | implied shift | shift / e₁* |",
        "|---|---|---|---|---|---|---|---|", *rows, "",
        f"**Reading.** On the standard final tier the gap is {g(N['gap'][50], 4)}·ΔW (q = 50) and {g(N['gap'][60], 4)}·ΔW (q = 60), and the stage-1 optimum it would imply "
        f"differs from the table's by {chk['verifier_numerics']['q50']['final']['implied_shift']:+.3f} and "
        f"{chk['verifier_numerics']['q60']['final']['implied_shift']:+.3f} effort units "
        f"({g(chk['verifier_numerics']['q50']['final']['implied_shift_over_e1_star'], 3)}·e₁* and "
        f"{g(chk['verifier_numerics']['q60']['final']['implied_shift_over_e1_star'], 3)}·e₁*). That is above the limit of 1e-3·e₁* that (ii-c) applies "
        "to the table's own convergence profile, whose measured shift is "
        f"{g(ic[50]['fixed_opponent']['shift_over_e1_star'], 3)}·e₁* (q = 50) and {g(ic[60]['fixed_opponent']['shift_over_e1_star'], 3)}·e₁* (q = 60). "
        "The gap sits on the verifier's side (its state-grid interpolation of V₂, step 2 on the final tier), falls by about 4× per halving of the state "
        f"step, and is {g(chk['ii_b']['q50']['max_abs_diff_over_dw'], 4)}·ΔW / {g(chk['ii_b']['q60']['max_abs_diff_over_dw'], 4)}·ΔW on the refined configuration ((ii-b)). It therefore bounds the precision of the verifier's own ẽ₁ on the "
        "standard tier; it is not evidence about the table.", ""])


def main() -> int:
    N = load_numbers()
    v11 = json.load(open(V11))
    assert sha256_file(V11) == V11_SHA, "v1.1 JSON is not the locked file"
    v20 = apply_changes(v11, N)
    V20.write_text(json.dumps(v20, indent=1) + "\n")
    MD.write_text(render_md(v20, N))
    (OUT / "verifier_numerics_note.md").write_text(render_numerics_note(N))
    d = diff(v11, v20)
    bad = [x["path"] for x in d if x["path"] not in ALLOWED]
    DIFF.parent.mkdir(parents=True, exist_ok=True)
    DIFF.write_text(json.dumps(d, indent=1) + "\n")
    print("wrote", V20, "sha256", sha256_file(V20))
    print("wrote", MD, "sha256", sha256_file(MD))
    for x in d:
        print(f"{x['op']:8s} {x['path']}")
    if bad:
        print("UNEXPECTED CHANGES (not allowed by D2-D5):", bad)
        return 3
    print(f"all {len(d)} changed paths are within D2-D5 (ALLOWED has {len(ALLOWED)} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
