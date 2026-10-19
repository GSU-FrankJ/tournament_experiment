#!/usr/bin/env python3
"""Writes the lock of the T = 3 confirmation of round T3-R2 (D4): ``v2_T3_locked_v1_0.json`` and ``.md``.

The files contain the settings of D4 (the v2.0 records of both q, the parameters file, the budgets and learning-rate
windows, the starts, the rule), the gates, the confirmation rule, the failure counting, the seed block, the reference
config hashes of T3-R1's ``bw`` runs, and the SHA-256 of every tool file the confirmation uses. They hold no timestamp, no
commit hash and no absolute path, so a draft written in P1 and the lock written in P2 are byte-identical when nothing they
list has changed.

Usage:
    python tools/t3/t3r2_make_lock.py --out reports/t3/r2/lock_draft/      # P1: the drafts
    python tools/t3/t3r2_make_lock.py --out protocols/                     # P2: the lock (must equal the drafts)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import t3_configs as tc  # noqa: E402
import t3c_launch as cl  # noqa: E402

NAME = "v2_T3_locked_v1_0"
TOOL_FILES_FIXED = ("tools/t3/t3_eval.py", "tools/t3/t3_metrics.py", "tools/t3/t3_configs.py", "tools/t3/t3_launch.py",
                    "tools/ms/ms_configs.py")
CRITERIA_REL = "reports/t3/r1/prereg_criteria.json"
RULE = {"min_total_passes": 36, "min_passes_per_q": 17, "runs_per_q": 20, "n_runs": 40,
        "text": "the confirmation passes iff at least 36 of the 40 runs pass, and at least 17 of 20 at q = 50 pass, and at "
                "least 17 of 20 at q = 60 pass; otherwise it fails",
        "rationale": "the confirmation must reproduce T3-R1's development pass rate (36 of 40), with neither q below 17 of 20"}
RUN_PASS = ("exit code 0; the process-global RNG assertion is ok; a valid evaluation (final and development tier); and "
            "G-A and G-F and G-N")
FAILURE_COUNTING = {
    "reporting": "every planned run is reported; no seed is replaced or re-drawn",
    "infrastructure_failure": ["a negative return code (a signal, e.g. SIGKILL)",
                               "an out-of-memory kill recorded in the kernel log or the run's log",
                               "OSError ENOSPC recorded in the run's status.json or traceback"],
    "ledger": "results/t3_r2/confirmation/infra_events.json; the analysis tool and the blind tool both read it",
    "rerun_after_infrastructure_failure": "the attempt is moved to results/t3_r2/crashed/ and the run is repeated once "
                                           "into the original path; if the repeat also fails for an infrastructure reason "
                                           "the confirmation stops without a verdict and that run is counted neither way",
    "other_failures": "any other nonzero exit, and any global-RNG violation, is a failed run",
    "evaluation_problems": "an evaluation error, or an evaluation with valid = False, counts as a failed run unless a re-run "
                           "of the locked evaluator on the same weights succeeds; the re-run is reported; a completed run is "
                           "never re-trained"}


def sha256_bytes(b: bytes) -> str:
    """SHA-256 of bytes."""
    return hashlib.sha256(b).hexdigest()


def tool_files() -> List[str]:
    """The tool files the confirmation uses, sorted (repository-relative POSIX paths)."""
    own = sorted(str(p.relative_to(ROOT).as_posix()) for p in (ROOT / "tools" / "t3").glob("t3c_*.py"))
    return sorted(set(own) | set(TOOL_FILES_FIXED))


def build_lock() -> Dict[str, Any]:
    """The lock document (a plain dict, no timestamp, no commit, no absolute path)."""
    proto = tc.load_protocol()
    params = tc.load_bw_params()
    crit = json.loads((ROOT / CRITERIA_REL).read_text())
    ref = {q: tc.build_bw_config(proto, q, cl.REFERENCE_SEED, "REFERENCE", params) for q in cl.QS}
    c50 = ref[50]
    settings = {
        "arm": "bw", "horizon": 3, "phases": c50["pipeline"]["phases"],
        "budgets": c50["pipeline"]["budgets"], "lr_windows": c50["pipeline"]["lr_windows"],
        "lr": {"start": tc.LR_START, "end": tc.LR_END, "text": "3e-4 constant, then linear 3e-4 -> 3e-5 over local updates "
                                                              "1201-1600 at stages 3 and 2 and over 1-600 at stage 1"},
        "episodes_per_update": c50["record"]["protocol"]["episodes_per_update"],
        "start_weights": c50["start_weights"], "rule": c50["rule"], "clamp_likelihood": c50["clamp_likelihood"],
        "init_digest": c50["init_digest"], "check_every": tc.CHECK_EVERY,
        "snapshot_every": c50["record"]["protocol"]["snapshot_every"], "weights_every": c50["record"]["protocol"]["weights_every"],
        "evaluated_policy": "the last iterate (Beta means) on the final and the development verifier tier",
        "code": "agents/, envs/, utils/, run/, tools/ms/, tools/v2/ and the existing files of protocols/ as at the final "
                "commit of round T3-R1 (unchanged since its launch); run/run_ms_stagewise.py is the entry point"}
    return {
        "schema": "t3_locked_protocol/1", "protocol_version": "T3-1.0", "round": cl.ROUND, "name": NAME,
        "note": "T = 3 backward-induction training (arm bw of round T3-R1, unchanged) and its fresh-seed confirmation; "
                "If this document and the JSON ever disagree, the JSON is what runs.",
        "protocol_file": tc.PROTOCOL_REL, "protocol_sha256": tc.sha256_file(tc.PROTOCOL_PATH),
        "params_file": tc.PARAMS_REL, "params_sha256": tc.sha256_file(tc.PARAMS_PATH), "params": params,
        "criteria_file": CRITERIA_REL, "criteria_sha256": sha256_bytes((ROOT / CRITERIA_REL).read_bytes()),
        "gates": crit["gates"], "records": {str(q): proto["records"][str(q)] for q in cl.QS},
        "settings": settings,
        "confirmation": {"qs": list(cl.QS), "seed_block": [cl.SEEDS[0], cl.SEEDS[-1]], "same_seeds_at_both_q": True,
                         "n_runs": len(cl.QS) * len(cl.SEEDS), "arm": cl.ARM, "wave": cl.WAVE,
                         "run_name": "t3r2_q{q}_s{seed}_bw", "run_dir": "results/t3_r2/confirmation/q{q}/seed{seed}/bw",
                         "config_fields_set_per_run": list(cl.ALLOWED_DIFF),
                         "pilot": cl.ROUND, "reference_run": f"T3-R1 bw seed {cl.REFERENCE_SEED}",
                         "never_trained_seeds": "40501-40520, 10501-10530, 30501-30520, 50500-50520 (except the four audit re-runs)",
                         "run_pass": RUN_PASS, "rule": RULE, "failure_counting": FAILURE_COUNTING,
                         "reported_not_deciding": ["pass counts per q with exact Clopper-Pearson 95 % intervals",
                                                   "per failing run the diagnosis of reports/t3/r1/04_t3_results.md section 4.1 plus the A7 analysis",
                                                   "the near-zero stretches of all 40 runs",
                                                   "the pass rate pooled with T3-R1's 40 development runs (descriptive)",
                                                   "the gate values set against T3-R1's"],
                         "on_failure": "nothing is changed or tried; the PI decides the next step"},
        "reference_config_sha256": {str(q): tc.config_sha256(ref[q]) for q in cl.QS},
        "tool_sha256": {f: sha256_bytes((ROOT / f).read_bytes()) for f in tool_files()},
        "rule_of_change": "nothing in protocols/ changes after this lock without a new version number and a stated reason"}


def render_md(lock: Dict[str, Any]) -> str:
    """Human-readable rendering of the lock (generated; the JSON is authoritative)."""
    s, c = lock["settings"], lock["confirmation"]
    lines = [
        "# T = 3 backward-induction training (`bw`) — locked confirmation, version T3-1.0", "",
        "- **Machine-readable source:** `protocols/v2_T3_locked_v1_0.json` (generated by `tools/t3/t3r2_make_lock.py`; if this "
        "document and the JSON ever disagree, the JSON is what runs).",
        "- **Entry:** `run/run_ms_stagewise.py` through `tools/t3/t3c_launch.py` (wave `t3c`).",
        "- **Lock record:** `protocols/LOCK` (the T3 record is appended in a separate commit; no existing record changes).",
        f"- **Rule of change:** {lock['rule_of_change']}.", "",
        "## 1. Locked settings (exactly T3-R1's `bw`)", "",
        f"- Records: the v2.0 record of each q (`{lock['protocol_file']}`, SHA-256 `{lock['protocol_sha256']}`), `game.T` = 3.",
        f"- Parameters file `{lock['params_file']}`, SHA-256 `{lock['params_sha256']}`.",
        f"- Phases {s['phases']}; budgets {json.dumps(s['budgets'])}; {s['lr']['text']}.",
        f"- {s['episodes_per_update']} episodes per update; starts {json.dumps(s['start_weights'])}; rule disabled "
        f"(checks every {s['check_every']} local updates are recorded); `init_digest` {str(s['init_digest']).lower()}; "
        f"clamp likelihood `{s['clamp_likelihood']}`; snapshot refresh every {s['snapshot_every']} updates; exports every "
        f"{s['weights_every']} updates.",
        f"- Evaluated policy: {s['evaluated_policy']}.", f"- Code: {s['code']}.", "",
        "## 2. Confirmation design", "",
        f"- q in {c['qs']}; seeds {c['seed_block'][0]}-{c['seed_block'][1]} (the same 20 seeds at both q): {c['n_runs']} runs, "
        f"arm `{c['arm']}` only; run names `{c['run_name']}`; directories `{c['run_dir']}`.",
        f"- Each config is T3-R1's `bw` config of the same q with only these fields set: {', '.join(f'`{f}`' for f in c['config_fields_set_per_run'])} "
        f"(`pilot` = \"{c['pilot']}\").",
        f"- Seeds never trained: {c['never_trained_seeds']}.", "",
        "## 3. Gates (T3-R1's, inclusive `<=`, full float64 precision)", "",
        "| Gate | Criterion |", "|---|---|"]
    g = lock["gates"]
    lines += [f"| G-A | `eta_T_over_dw` <= {g['G-A']['eta_T_over_dw']}, `rmse_pos_T_over_eT0` <= {g['G-A']['rmse_pos_T_over_eT0']}, "
              f"`tail_mean_T_over_eT0` <= {g['G-A']['tail_mean_T_over_eT0']} (final tier) |",
              f"| G-F | `Gmax_full_over_dw` <= {g['G-F']['Gmax_full_over_dw']} (final tier) |",
              f"| G-N | development minus final: abs diff of `eta_T_over_dw` <= {g['G-N']['eta_T_over_dw_dev_minus_final_abs']} and "
              f"of `Gmax_full_over_dw` <= {g['G-N']['Gmax_full_over_dw_dev_minus_final_abs']} |", "",
              f"**Run pass:** {c['run_pass']}.", "", "## 4. Confirmation rule", "", f"- {c['rule']['text']}.",
              f"- Rationale: {c['rule']['rationale']}.", "", "## 5. Failure counting", "",
              f"- {c['failure_counting']['reporting']}.", "- Infrastructure failure, only if one of these holds:"]
    lines += [f"  - {x}" for x in c["failure_counting"]["infrastructure_failure"]]
    lines += [f"- Ledger: `{c['failure_counting']['ledger']}`.",
              f"- Re-run: {c['failure_counting']['rerun_after_infrastructure_failure']}.",
              f"- Other failures: {c['failure_counting']['other_failures']}.",
              f"- Evaluation problems: {c['failure_counting']['evaluation_problems']}.", "",
              "## 6. Reported, deciding nothing", ""]
    lines += [f"- {x}" for x in c["reported_not_deciding"]]
    lines += ["", f"If the confirmation fails: {c['on_failure']}.", "", "## 7. Reference configs and tool files", "",
              "SHA-256 of the config text of T3-R1's `bw` run (seed 50501, `record.output_dir` = REFERENCE) per q; the launcher "
              "refuses a config whose normalised form hashes differently:", ""]
    lines += [f"- q = {q}: `{h}`" for q, h in lock["reference_config_sha256"].items()]
    lines += ["", "Tool files the confirmation uses (the launcher refuses if one differs):", ""]
    lines += [f"- `{f}`: `{h}`" for f, h in lock["tool_sha256"].items()]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R2 lock writer", allow_abbrev=False)
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
