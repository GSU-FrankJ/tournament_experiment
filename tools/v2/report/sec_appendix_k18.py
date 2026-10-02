"""K18: the number of bit-identity checks passed, out of how many (from the reproducibility ledger T18)."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

import common as C
import studies as S

MOD = "sec_appendix_k18.py"
CHECK1 = "results/v2_T2_locked/rehearsal_analysis/check1_phaseA_vs_stitched.csv"
TRAINING_FIELDS = ["actor_identical", "critic_identical", "opponent_identical", "opt_actor_identical", "opt_critic_identical",
                   "rng_minibatch_identical", "rng_streams_identical", "torch_generator_identical", "exports_A_identical",
                   "stage2_metrics_u1600_identical"]
GLOBAL_FIELDS = ["torch_global_rng_identical", "numpy_global_rng_identical", "python_random_identical"]


def _row(sub: str, desc: str, value: Any, unit: str, sf: str, sel: str, comp: str) -> Dict[str, Any]:
    return {"sub": sub, "description": desc, "value": value, "unit": unit, "normalization": "none", "tier": "n/a", "q": "50, 60",
            "source_file": sf, "selector": sel, "computation": comp}


def build_k18(pack: C.Pack) -> None:
    """Count the ledger's checks (one row = one named check) and the Check 1 field results."""
    t18 = pack.pack_table("T18")
    t18_src = pack.pack_src("T18").path
    res = t18["result"].astype(str)
    passed = res == "pass"
    spec = t18["in_spec_list"].astype(bool)
    c1 = S.read_csv(CHECK1)
    train_ok = int(c1[TRAINING_FIELDS].astype(bool).all(axis=1).sum())
    glob_ok = int(c1[GLOBAL_FIELDS].astype(bool).all(axis=1).sum())
    rows: List[Dict[str, Any]] = [
        _row("ledger.n_named_checks", "Named bit-identity / reproducibility checks in the ledger T18 (one row = one named check)", int(len(t18)), "checks",
             t18_src, "all rows", "row count"),
        _row("ledger.n_pass", "Checks that pass on their stated criterion", int(passed.sum()), "checks of %d" % len(t18), t18_src, "result == pass",
             "count of rows with result == 'pass'"),
        _row("ledger.n_fail", "Checks that fail on their stated criterion", int((~passed).sum()), "checks", t18_src, "result != pass",
             "count of rows with result != 'pass'"),
        _row("ledger.failed_checks", "The failing check(s) and their ledger result", "; ".join(f"{i}: {r}" for i, r in zip(t18.loc[~passed, 'check_id'], res[~passed])),
             "text", t18_src, "result != pass", "listed"),
        _row("spec_list.n_named_checks", "Checks that the request names (Phase 2 branching test, Pilot 3 vs Pilot 2 B2, Pilot 4 2a vs extension, dirty-flag re-run, "
                                         "pre-lock insurance runs, v1.0 Check 1 and 2, v1.1 R1)", int(spec.sum()), "checks", t18_src, "in_spec_list == True",
             "count of rows flagged in_spec_list"),
        _row("spec_list.n_pass", "Of the checks named in the request, those that pass on their stated criterion", int((spec & passed).sum()), "checks", t18_src,
             "in_spec_list == True and result == pass", "count"),
        _row("check1.training_state_identical_runs", "v1.0 Check 1: runs (of 20) in which every training-relevant field is identical (actor, critic, opponent, both Adam "
                                                     "states, minibatch RNG, five RNG streams, torch generator, all exports, u1600 stage-2 metrics)", train_ok, "runs of 20",
             CHECK1, "fields " + ", ".join(TRAINING_FIELDS), "count of rows with all fields True"),
        _row("check1.global_rng_states_identical_runs", "v1.0 Check 1: runs (of 20) in which the three process-global RNG states (torch, numpy legacy, python random) are "
                                                        "identical", glob_ok, "runs of 20", CHECK1, "fields " + ", ".join(GLOBAL_FIELDS), "count of rows with all fields True"),
        _row("ledger.n_pass_with_D1_definition", "Checks passing if v1.0 Check 1 is judged on the training-relevant state (owner decision D1) instead of its literal "
                                                 "all-RNG-state criterion", int(passed.sum()) + (1 if train_ok == 20 else 0), "checks of %d" % len(t18), t18_src,
             "n_pass + 1 when the training state is identical in 20/20 (see check1.training_state_identical_runs)",
             "n_pass plus the one failing check re-judged under D1; D1 is recorded in reports/v2/protocol_lock_and_rehearsal.md (addendum)"),
    ]
    pack.numbers_item("K18", rows, status="derived", script=f"{MOD}:build_k18",
                      notes="counts are over named checks; the run-level counts of each check are in T18 (n_units, n_identical)")
