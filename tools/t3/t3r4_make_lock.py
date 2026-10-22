#!/usr/bin/env python3
"""Writes the lock of the T = 3 confirmation of round T3-R4: ``v2_T3_locked_v1_1.json`` and ``.md`` (protocol T3-1.1).

T3-1.1 is T3-1.0 (the locked ``bw``: ``protocols/v2_T3_locked_v1_0.json``, unchanged) plus the config key
``stage3_retry = {"delta_gt": 0.005, "max_retries": 1}`` (``run/ms_retry.py``, behind the key in ``run/run_ms_stagewise.py``). The files
contain the T3-1.0 settings with the key added, the reference config SHA-256 per q (normalised as ``t3c_launch`` does for T3-1.0, with the
key), the paper wording for A (verbatim), the confirmation rule fixed in T3-R3's pre-registration (D6), its operating characteristic, the
failure counting, the seed block, and the SHA-256 of ``run/ms_retry.py``, ``run/run_ms_stagewise.py`` and every tool the confirmation runs.
They hold no timestamp, no commit hash and no absolute path, so the drafts written in P1 and the lock written in P1.6 are byte-identical
when nothing they list has changed.

Usage:
    python tools/t3/t3r4_make_lock.py --out reports/t3/r4/lock_draft/      # the drafts
    python tools/t3/t3r4_make_lock.py --out protocols/                     # the lock (must equal the drafts)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import t3_configs as tc  # noqa: E402
import t3r2_make_lock as r2  # noqa: E402
import t3r4_launch as lt  # noqa: E402

NAME = "v2_T3_locked_v1_1"
PAPER_WORDING = ("a verifier-gated restart of the terminal stage: when the development-tier one-step deviation gain at the end of the "
                 "stage-3 phase exceeds the gate's limit 0.005, stage 3 is re-initialised once and retrained, and the second attempt is "
                 "kept; its cost is part of the method's cost.")
# the files of the lock besides ``tools/t3/t3r4_*``: the entry point and its code, and every tool module the t3r4 tools import
# (``tests/test_t3r4_lock.py`` checks that the import closure of the t3r4 tools stays inside this list)
FILES_FIXED = (
    "run/ms_retry.py", "run/run_ms_stagewise.py",
    "tools/ms/launch_ms_r1.py", "tools/ms/ms_configs.py", "tools/ms/r4_common.py",
    "tools/t3/t3_configs.py", "tools/t3/t3_eval.py", "tools/t3/t3_launch.py", "tools/t3/t3_launch_checks.py", "tools/t3/t3_metrics.py",
    "tools/t3/t3c_analysis.py", "tools/t3/t3c_blind.py", "tools/t3/t3c_eval.py", "tools/t3/t3c_launch.py",
    "tools/t3/t3r2_audit.py", "tools/t3/t3r2_br_calc.py", "tools/t3/t3r2_make_lock.py",
    "tools/t3/t3r3_analysis.py", "tools/t3/t3r3_checks.py", "tools/t3/t3r3_diag.py", "tools/t3/t3r3_last16.py", "tools/t3/t3r3_launch.py",
    "tools/t3/t3r3_pair.py", "tools/t3/compare_runs.py")
RULE = {"max_failures_total": 6, "max_failures_per_q": 4, "runs_per_q": 40, "n_runs": 80,
        "text": "the confirmation passes iff at most 6 of the 80 runs fail and at most 4 of the 40 runs fail at q = 50 and at most 4 of "
                "the 40 runs fail at q = 60; otherwise it fails",
        "rationale": "fixed before any development run in the pre-registration of round T3-R3 (reports/t3/r3/04_preregistration.md "
                     "section 9, D6) and binding here"}
RUN_PASS = r2.RUN_PASS
OC_FAILURE_PROBABILITIES = (0.01, 0.03, 0.05, 0.10, 0.15)
FAILURE_COUNTING = {
    "reporting": "every planned run is reported; no seed is replaced or re-drawn",
    "infrastructure_failure": ["a negative return code (a signal, e.g. SIGKILL)",
                               "an out-of-memory kill recorded in the kernel log or the run's log",
                               "OSError ENOSPC recorded in the run's status.json or traceback"],
    "ledger": "results/t3_r4/confirmation/infra_events.json; the analysis tool and the blind tool both read it",
    "rerun_after_infrastructure_failure": "the attempt is moved to results/t3_r4/confirmation/crashed/ and the run is repeated once into "
                                           "the original path; if the repeat also fails for an infrastructure reason the confirmation "
                                           "stops without a verdict (no pass or fail is declared)",
    "other_failures": "any other nonzero exit, and any global-RNG violation, is a failed run",
    "evaluation_problems": "an evaluation error, or an evaluation with valid = False, counts as a failed run unless a re-run of the locked "
                           "evaluator on the same weights succeeds; the re-run is reported; a completed run is never re-trained",
    "scheme_steps": "a re-initialisation of stage 3 inside a run (the retry of stage3_retry) is part of the run, never a re-run"}
REPORTED = ["pass counts per q and in total with exact Clopper-Pearson 95 % intervals",
            "the worst Gmax_full and its run; the medians of eta_3, rmse_pos_T_over_eT0, tail_mean_T_over_eT0, peak_T_signed, Gmax_full and EXP_root",
            "the fire rate of the trigger with the runs, and the first-attempt outcome (the pass count with every fired run counted as a "
            "failure of its first attempt)",
            "per failing run: the failing gates and values, the stage of the failing quantity, S3, the stage-2 stretch, the last-16-export "
            "range of the failing stage-3 quantity, and its class (F), (K), (V), (E) or (U) as in T3-R3's D2",
            "cost: updates, policy rows and wall time per run, and the share of fired runs",
            "descriptive comparisons: this confirmation pooled with bwA's 40 development runs of T3-R3; the three blocks of bw "
            "(T3-R1, T3-R2, T3-R3 development); the pass rate of T3-1.0 implied by the first-attempt outcome"]


def sha256_bytes(b: bytes) -> str:
    """SHA-256 of bytes."""
    return hashlib.sha256(b).hexdigest()


def tool_files() -> List[str]:
    """The files the lock lists, sorted (repository-relative POSIX paths): ``FILES_FIXED`` and every ``tools/t3/t3r4_*`` file."""
    own = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "tools" / "t3").glob("t3r4_*") if p.suffix in (".py", ".sh"))
    return sorted(set(own) | set(FILES_FIXED))


def pass_probability(f: float, n_per_q: int = 40, max_per_q: int = 4, max_total: int = 6) -> float:
    """Probability that the rule is met when every run fails independently with probability ``f`` (the same at both q): at most
    ``max_per_q`` failures among the ``n_per_q`` runs of each q and at most ``max_total`` in all."""
    pmf = [math.comb(n_per_q, k) * f ** k * (1.0 - f) ** (n_per_q - k) for k in range(max_per_q + 1)]
    return float(sum(pmf[a] * pmf[b] for a in range(max_per_q + 1) for b in range(max_per_q + 1) if a + b <= max_total))


def build_lock() -> Dict[str, Any]:
    """The lock document (a plain dict, no timestamp, no commit, no absolute path)."""
    proto = tc.load_protocol()
    params = tc.load_bw_params()
    base = r2.build_lock()                                  # the T3-1.0 settings (the same files, the same reference run)
    ref = {q: lt.reference_config(proto, params, q) for q in lt.QS}
    settings = dict(base["settings"])
    settings["stage3_retry"] = dict(lt.RETRY)
    settings["retry"] = {
        "code": "run/ms_retry.py, behind the config key stage3_retry in run/run_ms_stagewise.py",
        "trigger": "read at the development check at local update 1600 of the stage-3 phase: fires iff Delta > 0.005, or Delta, s or C is "
                   "not finite, or s <= 0",
        "retry": "one in-place re-initialisation of stage 3 with derived seeds and streams and one retraining; the second attempt is "
                 "frozen whatever its own check says",
        "records": "the artefacts of attempt 1 move to attempt1/ (nothing is overwritten or deleted) and retry.json records the trigger "
                   "values and the derived seeds; a run whose trigger does not fire is bit-identical to bw",
        "budget_of_a_fired_run": {"updates": 5400, "policy_rows": 2764800, "minibatch_steps": 108000, "development_checks": 216,
                                  "exports": "152 plus 64 in attempt1/"}}
    settings["arm"] = "bw"
    settings["code"] = ("agents/, envs/, utils/, run/, tools/ms/, tools/v2/ and the existing files of protocols/ as at the final commit of "
                        "round T3-R3 (git diff b565f0e8 HEAD over agents envs utils run tools/ms tools/v2 is empty); "
                        "run/run_ms_stagewise.py is the entry point")
    return {
        "schema": "t3_locked_protocol/1", "protocol_version": lt.PROTOCOL_VERSION, "round": lt.ROUND, "name": NAME,
        "note": "T = 3 backward-induction training with the verifier-gated restart of stage 3 (arm bwA of round T3-R3) and its fresh-seed "
                "confirmation. If this document and the JSON ever disagree, the JSON is what runs.",
        "builds_on": {"protocol": "protocols/v2_T3_locked_v1_0.json",
                      "protocol_sha256": sha256_bytes((ROOT / "protocols" / "v2_T3_locked_v1_0.json").read_bytes()),
                      "difference": "the config key stage3_retry = {\"delta_gt\": 0.005, \"max_retries\": 1}"},
        "protocol_file": tc.PROTOCOL_REL, "protocol_sha256": tc.sha256_file(tc.PROTOCOL_PATH),
        "params_file": tc.PARAMS_REL, "params_sha256": tc.sha256_file(tc.PARAMS_PATH), "params": params,
        "criteria_file": r2.CRITERIA_REL, "criteria_sha256": sha256_bytes((ROOT / r2.CRITERIA_REL).read_bytes()),
        "gates": base["gates"], "records": base["records"],
        "settings": settings,
        "paper_wording": PAPER_WORDING,
        "confirmation": {"qs": list(lt.QS), "seed_block": [lt.SEEDS[0], lt.SEEDS[-1]], "same_seeds_at_both_q": True,
                         "n_runs": len(lt.QS) * len(lt.SEEDS), "arm": lt.ARM, "wave": lt.WAVE,
                         "run_name": "t3r4_q{q}_s{seed}_bwA", "run_dir": "results/t3_r4/confirmation/q{q}/seed{seed}/bwA",
                         "config_fields_set_per_run": list(lt.ALLOWED_DIFF), "scheme_key": "stage3_retry",
                         "pilot": lt.ROUND, "reference_run": f"T3-R1 bw seed {lt.REFERENCE_SEED} with the key stage3_retry",
                         "seed_block_status": "reserved in round T3-R3 and never trained (reports/t3/r3/04_preregistration.md section 12)",
                         "run_pass": RUN_PASS, "rule": RULE,
                         "operating_characteristic": {
                             "assumption": "independent per-run failure probability f, the same at both q",
                             "pass_probability": {f"{f:g}": round(pass_probability(f), 6) for f in OC_FAILURE_PROBABILITIES}},
                         "failure_counting": FAILURE_COUNTING, "reported_not_deciding": REPORTED,
                         "on_failure": "nothing is changed or tried; the PI decides the next step"},
        "reference_config_sha256": {str(q): tc.config_sha256(ref[q]) for q in lt.QS},
        "tool_sha256": {f: sha256_bytes((ROOT / f).read_bytes()) for f in tool_files()},
        "rule_of_change": "nothing in protocols/ changes after this lock without a new version number and a stated reason"}


def render_md(lock: Dict[str, Any]) -> str:
    """Human-readable rendering of the lock (generated; the JSON is authoritative)."""
    s, c = lock["settings"], lock["confirmation"]
    r = s["retry"]
    lines = [
        "# T = 3 backward-induction training with a verifier-gated restart (`bwA`) — locked confirmation, version T3-1.1", "",
        f"- **Machine-readable source:** `protocols/{NAME}.json` (generated by `tools/t3/t3r4_make_lock.py`; if this document and the JSON "
        "ever disagree, the JSON is what runs).",
        "- **Entry:** `run/run_ms_stagewise.py` through `tools/t3/t3r4_launch.py` (wave `t3r4`).",
        "- **Lock record:** `protocols/LOCK` (the T3-1.1 record is appended in a separate commit; no existing record changes).",
        f"- **Builds on:** `{lock['builds_on']['protocol']}` (SHA-256 `{lock['builds_on']['protocol_sha256']}`), unchanged; "
        f"difference: {lock['builds_on']['difference']}.",
        f"- **Rule of change:** {lock['rule_of_change']}.", "",
        "## 1. Locked settings (T3-1.0's `bw`, plus the restart)", "",
        f"- Records: the v2.0 record of each q (`{lock['protocol_file']}`, SHA-256 `{lock['protocol_sha256']}`), `game.T` = 3.",
        f"- Parameters file `{lock['params_file']}`, SHA-256 `{lock['params_sha256']}`.",
        f"- Phases {s['phases']}; budgets {json.dumps(s['budgets'])}; {s['lr']['text']}.",
        f"- {s['episodes_per_update']} episodes per update; starts {json.dumps(s['start_weights'])}; rule disabled "
        f"(checks every {s['check_every']} local updates are recorded); `init_digest` {str(s['init_digest']).lower()}; "
        f"clamp likelihood `{s['clamp_likelihood']}`; snapshot refresh every {s['snapshot_every']} updates; exports every "
        f"{s['weights_every']} updates.",
        f"- Evaluated policy: {s['evaluated_policy']}.", f"- Code: {s['code']}.",
        f"- **Restart:** config key `stage3_retry = {json.dumps(s['stage3_retry'])}` ({r['code']}). Trigger: {r['trigger']}. Retry: {r['retry']}. "
        f"Records: {r['records']}. Budget of a fired run: {r['budget_of_a_fired_run']['updates']} updates, "
        f"{r['budget_of_a_fired_run']['policy_rows']} policy rows, {r['budget_of_a_fired_run']['minibatch_steps']} minibatch steps, "
        f"{r['budget_of_a_fired_run']['development_checks']} development checks, exports {r['budget_of_a_fired_run']['exports']}.", "",
        "**Paper wording for A (verbatim):**", "", f"> {lock['paper_wording']}", "",
        "## 2. Confirmation design", "",
        f"- q in {c['qs']}; seeds {c['seed_block'][0]}-{c['seed_block'][1]} (the same 40 seeds at both q): {c['n_runs']} runs, "
        f"arm `{c['arm']}` only; run names `{c['run_name']}`; directories `{c['run_dir']}`.",
        f"- Each config is T3-R1's `bw` config of the same q with `{c['scheme_key']}` added and only these fields set: "
        f"{', '.join(f'`{f}`' for f in c['config_fields_set_per_run'])} (`pilot` = \"{c['pilot']}\").",
        f"- Seed block: {c['seed_block_status']}.", "",
        "## 3. Gates (T3-R1's, inclusive `<=`, full float64 precision)", "",
        "| Gate | Criterion |", "|---|---|"]
    g = lock["gates"]
    lines += [f"| G-A | `eta_T_over_dw` <= {g['G-A']['eta_T_over_dw']}, `rmse_pos_T_over_eT0` <= {g['G-A']['rmse_pos_T_over_eT0']}, "
              f"`tail_mean_T_over_eT0` <= {g['G-A']['tail_mean_T_over_eT0']} (final tier) |",
              f"| G-F | `Gmax_full_over_dw` <= {g['G-F']['Gmax_full_over_dw']} (final tier) |",
              f"| G-N | development minus final: abs diff of `eta_T_over_dw` <= {g['G-N']['eta_T_over_dw_dev_minus_final_abs']} and "
              f"of `Gmax_full_over_dw` <= {g['G-N']['Gmax_full_over_dw_dev_minus_final_abs']} |", "",
              f"**Run pass:** {c['run_pass']}.", "", "## 4. Confirmation rule", "", f"- {c['rule']['text']}.",
              f"- Rationale: {c['rule']['rationale']}.",
              f"- Operating characteristic ({c['operating_characteristic']['assumption']}): pass probability "
              + ", ".join(f"{p:.4f} at f = {f}" for f, p in c["operating_characteristic"]["pass_probability"].items()) + ".", "",
              "## 5. Failure counting", "", f"- {c['failure_counting']['reporting']}.", "- Infrastructure failure, only if one of these holds:"]
    lines += [f"  - {x}" for x in c["failure_counting"]["infrastructure_failure"]]
    lines += [f"- Ledger: `{c['failure_counting']['ledger']}`.",
              f"- Re-run: {c['failure_counting']['rerun_after_infrastructure_failure']}.",
              f"- Other failures: {c['failure_counting']['other_failures']}.",
              f"- Evaluation problems: {c['failure_counting']['evaluation_problems']}.",
              f"- The scheme's own steps: {c['failure_counting']['scheme_steps']}.", "", "## 6. Reported, deciding nothing", ""]
    lines += [f"- {x}" for x in c["reported_not_deciding"]]
    lines += ["", f"If the confirmation fails: {c['on_failure']}.", "", "## 7. Reference configs and listed files", "",
              "SHA-256 of the config text of T3-R1's `bw` run (seed 50501) with `stage3_retry` per q (`record.output_dir` = REFERENCE); the "
              "launcher refuses a config whose normalised form hashes differently:", ""]
    lines += [f"- q = {q}: `{h}`" for q, h in lock["reference_config_sha256"].items()]
    lines += ["", "Files the confirmation uses (the launcher refuses if one differs):", ""]
    lines += [f"- `{f}`: `{h}`" for f, h in lock["tool_sha256"].items()]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R4 lock writer", allow_abbrev=False)
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = build_lock()
    js = json.dumps(lock, indent=1) + "\n"
    (out / f"{NAME}.json").write_text(js)
    (out / f"{NAME}.md").write_text(render_md(lock))
    print(f"wrote {out / (NAME + '.json')} sha256 {sha256_bytes(js.encode())}")
    print(f"wrote {out / (NAME + '.md')} sha256 {sha256_bytes((out / (NAME + '.md')).read_bytes())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
