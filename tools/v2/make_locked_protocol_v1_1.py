#!/usr/bin/env python3
"""Generate protocols/v2_T2_locked_v1_1.json from the v1.0 JSON by applying ONLY decisions D1-D5
of the v1.1 round, and write a machine diff v1.0 -> v1.1.

v1.0 (protocols/v2_T2_locked.json) is read, never written.

Outputs:
  protocols/v2_T2_locked_v1_1.json
  results/v2_T2_locked/v1_1/protocol_diff_v1_0_to_v1_1.json   (every added / removed / changed JSON path)

Usage: python tools/v2/make_locked_protocol_v1_1.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V10 = ROOT / "protocols" / "v2_T2_locked.json"
V11 = ROOT / "protocols" / "v2_T2_locked_v1_1.json"
DIFF = ROOT / "results" / "v2_T2_locked" / "v1_1" / "protocol_diff_v1_0_to_v1_1.json"
BOOT_SEED = 20261002
PRIOR = "reports/v2/protocol_lock_and_rehearsal.md"


def apply_changes(p: dict) -> dict:
    p = copy.deepcopy(p)
    # ---- identity
    p["schema"] = "v2_T2_locked_protocol/1.1"
    p["version"] = "1.1"
    p["name"] = "v2 T=2 stagewise protocol (locked, v1.1)"
    p["supersedes"] = {"version": 1, "file": "protocols/v2_T2_locked.json", "lock_commit": "4bd2214",
                       "sha256": "cf7b6929adfaef6b712eb01d1a731adc937d0b87fcc37a0a9e68eda9ae92d9f6"}
    # ---- D2: gates
    g = p["gates"]
    s1_def = copy.deepcopy(g["G-F"]["all_must_hold"][1])
    g["G-F"]["all_must_hold"] = [g["G-F"]["all_must_hold"][0]]
    g["G-N"] = {"stage": "end of Phase A (eta_2) and end of Phase B (Gmax_full)", "candidate": "as in G-A and G-F",
                "tier": "development vs final", "all_must_hold": [
                    {"metric": "eta_T_over_dw_dev_minus_final_abs", "op": "<=", "threshold": 0.001,
                     "definition": "|eta_T_over_dw(development) - eta_T_over_dw(final)| of the G-A candidate at the end of Phase A"},
                    {"metric": "Gmax_full_over_dw_dev_minus_final_abs", "op": "<=", "threshold": 0.001,
                     "definition": "|Gmax_full_over_dw(development) - Gmax_full_over_dw(final)| of the G-F candidate at the end of Phase B"}]}
    g["run_outcome"] = ("pass iff G-A, G-F and G-N all pass; a run failing G-A is a stage-2 failure: its Phase B is still "
                        "executed and reported, and the run counts as failed; a run with a global-RNG violation (hardening) "
                        "or any nonzero exit counts as failed")
    g["comparison"] = "every '<=' is inclusive; values compared at full float64 precision, no rounding"
    g["dev_tier"] = ("G-A and G-F use the final tier; G-N compares development and final tier; every gate and reported "
                     "metric is also reported on the development tier")
    # ---- D3: secondary criterion S1
    s1_def["metric"] = "stage1_rel_err_abs"
    p["secondary"] = {
        "S1": {"criterion": s1_def, "part_of_run_pass": False, "pass_rule": None,
               "report_per_run": "value and pass/fail",
               "report_per_q": ["S1 pass count out of the seeds, exact Clopper-Pearson 95% CI",
                                "mean signed stage-1 relative error with 95% percentile bootstrap CI",
                                "median and SD of the signed stage-1 relative error"]},
        "bootstrap": {"resamples": 10000, "seed": BOOT_SEED, "rng": "numpy.random.default_rng(seed)",
                      "method": "percentile 95% CI of the mean over seeds; one bootstrap stream per (q, statistic) in table order"},
        "binomial_ci": "exact Clopper-Pearson 95% (scipy.stats.beta quantiles)",
    }
    # ---- D4: pass rule, seeds, v1.0 outcome
    c = p["confirmation"]
    c["status"] = "pre-registered; executed in the v1.1 round only if the v1.1 re-rehearsal checks R1-R6 all pass"
    c["seed_block"] = c.pop("proposed_seed_block")
    c["seed_block_status"] = "confirmed by the owner (v1.1 round)"
    c["pass_rule"] = ("for each q, at least 18 of 20 runs pass (G-A and G-F and G-N); the confirmation passes iff both q pass; "
                      "each q's pass rate is reported with an exact Clopper-Pearson 95% CI")
    c["analysis_script"] = "tools/v2/confirmation_analysis.py"
    c["crash_rule"] = ("infrastructure kill (OOM, disk full, process killed): move the run directory to confirmation/crashed/"
                       "q{q}/seed{s}_attempt1/ and re-run once into the original path; a second infrastructure failure stops "
                       "the round and that run is not counted either way; any other nonzero exit (pipeline exception, "
                       "global-RNG violation) counts as a failed run")
    p["v1_0_outcome_reported"] = {"definition": "G-A and the v1.0 G-F (Gmax_full_over_dw <= 0.01 AND stage1_rel_err_abs <= 0.10)",
                                  "role": "reported only; decides nothing"}
    # ---- D1: training-relevant state
    p["training_relevant_state"] = {
        "compared": ["actor, critic, lagged opponent, frozen stage-2 snapshot (Phase B), both Adam states",
                     "the five training RNG streams (env, learn, opp, start, minibatch) and the torch generator",
                     "all weight exports and the final weights",
                     "training histories and per-update logs, without their wall-clock fields",
                     "checkpoint metrics, gate-metric values, and the evaluation outputs (gateA_*.npz, final_*.npz, "
                     "induced_band.json, band_sweep.npz, drift_test.json)"],
        "excluded": ["verdict fields", "manifests", "the process-global RNG states (torch global, numpy legacy global, Python random)"],
        "decided": f"2026-10-02, owner decision D1 of the v1.1 round ({PRIOR}, addendum)"}
    # ---- D5: hardening of the process-global RNGs
    p["global_rng_hardening"] = {
        "seeding": "at the start of the pipeline function (run_v2_T2_locked.run_pipeline), before any object is constructed: "
                   "torch.manual_seed(seed), np.random.seed(seed), random.seed(seed) with the run seed; seeds recorded in the manifest",
        "reference_state": "each RNG's state immediately before the first Phase A update (Run.phase_start_hook at phase A); "
                           "the post-seed state is also recorded, with whether anything between seeding and the reference drew from each RNG",
        "assertion_points": ["end_of_A", "after_G-A", "end_of_B", "end_of_run"],
        "assertion": "each of the three states equals its reference state at every assertion point",
        "on_violation": "record the RNG and the point in gates.json; the run finishes and writes all outputs, then the entry point "
                        "exits with a nonzero code (5); the run counts as failed",
        "digests": "SHA-256 of each RNG state at seeding, at the reference point and at each assertion point, in the manifest",
        "training": "no change to training-relevant state (checked by the re-rehearsal R1)"}
    # ---- outputs / entry point bookkeeping implied by D2-D5
    p["pipeline"]["entry_point_reads"] = "protocols/v2_T2_locked_v1_1.json"
    p["pipeline"]["outputs"] = p["pipeline"]["outputs"] + [
        "gates.json (v1.1): G-A, G-F, G-N, run_pass, S1, v1_0_outcome, global_rng, protocol_version 1.1",
        "manifest.json (v1.1): global-RNG seeds and digests"]
    # ---- change log v1.0 -> v1.1
    p["change_log"] = p["change_log"] + [
        {"version": "1.1", "change": "stage-1 criterion |e_hat_1(0) - e1*(0)|/e1*(0) <= 0.10 moved from G-F to the secondary criterion S1 (no pass rule)",
         "why": "all 3 v1.0 rehearsal failures were on this criterion while Gmax_full/DW was 0.0021-0.0023; the 0.10 threshold came from "
                "Pilot 4 distributions with a constant-LR Phase A parent; the locked pipeline's distribution has a heavier upper tail",
         "evidence": [f"{PRIOR} sections 4.3-4.5", "results/v2_T2_locked/rehearsal_analysis/gates_per_run.csv",
                      "results/v2_T2_locked/v1_1/v1_0_pass_probability.json (tools/v2/v1_0_pass_probability.py)"],
         "decided_by": "PI, after the development-seed rehearsal and before any confirmation seed was run"},
        {"version": "1.1", "change": "gate G-N added: |dev - final| of eta_2 and of Gmax_full, each <= 0.001 DW",
         "why": "numerical refinement check; in the v1.0 rehearsal the dev - final differences were at most 2.2e-4 DW",
         "evidence": [f"{PRIOR} section 4.5", "results/v2_T2_locked/rehearsal_analysis/gate_distributions.csv"],
         "decided_by": "PI, after the development-seed rehearsal and before any confirmation seed was run"},
        {"version": "1.1", "change": "hardening of the process-global RNGs (seeded with the run seed; reference state; assertions; digests)",
         "why": "rehearsal Check 1: three unseeded, never-consumed process-global RNG states differed between processes",
         "evidence": [f"{PRIOR} section 4.1 and addendum", "results/v2_T2_locked/rehearsal_analysis/check1_phaseA_vs_stitched.csv"],
         "decided_by": "PI, after the development-seed rehearsal and before any confirmation seed was run"},
        {"version": "1.1", "change": "training-relevant state defined for every bit-identity check; confirmation seed block confirmed; "
                                     "v1.0 outcome reported only; pass-rate CIs exact",
         "why": "owner decisions D1 and D4 of the v1.1 round", "evidence": [f"{PRIOR} addendum"],
         "decided_by": "PI, after the development-seed rehearsal and before any confirmation seed was run"}]
    p["evidence"]["v1.1 changes"] = f"{PRIOR} sections 4.1, 4.3-4.5 and addendum; reports/v2/protocol_v1_1_confirmation.md"
    return p


def diff(a, b, path=""):
    out = []
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


def main() -> int:
    v10 = json.load(open(V10))
    v11 = apply_changes(v10)
    V11.write_text(json.dumps(v11, indent=1) + "\n")
    d = diff(v10, v11)
    DIFF.parent.mkdir(parents=True, exist_ok=True)
    DIFF.write_text(json.dumps(d, indent=1) + "\n")
    print("wrote", V11)
    for x in d:
        print(f"{x['op']:8s} {x['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
