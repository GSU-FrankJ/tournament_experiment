"""Front matter of the pack: T01 study inventory, T02 decision log, T03 plan vs execution."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List

import pandas as pd

import common as C
import studies as S

MOD = "sec_front.py"
RPT = "reports/v2"


# ---------------------------------------------------------------------------------------------
# T01
# ---------------------------------------------------------------------------------------------

def _run_stats(study: str) -> Dict[str, Any]:
    """Counts and time span of the runs of a registered study, from status.json files."""
    n_plan = n_done = 0
    starts: List[str] = []
    ends: List[str] = []
    walls: List[float] = []
    files = []
    for _, _, _, rd in S.iter_runs(study):
        n_plan += 1
        p = C.abspath(f"{rd}/status.json")
        if not p.is_file():
            continue
        st = S.status(rd)
        files.append(C.src(f"{rd}/status.json"))
        if st.get("state") == "done" and st.get("exit_code", 0) == 0:
            n_done += 1
        starts.append(st["start_time"])
        ends.append(st.get("end_time", ""))
        walls.append(float(st.get("total_wall_sec", float("nan"))))
    return {"planned": n_plan, "done": n_done, "start": min(starts) if starts else "", "end": max(ends) if ends else "",
            "files": files, "wall": walls}


def _seeds(study: str) -> str:
    sd = S.STUDIES[study]["seeds"]
    return f"{min(sd)}-{max(sd)}" if len(sd) > 1 else str(sd[0])


def build_t01(pack: C.Pack) -> None:
    """T01: one row per study, with counts computed from the run records."""
    sources: List[Any] = []
    rows: List[Dict[str, Any]] = []

    def run_row(label: str, study: str, report: str, report_commits: str, launch: str, notes: str = "",
                arms_override: str = "", runs_override: str = "", budget_override: str = "") -> None:
        st = _run_stats(study)
        sources.extend(st["files"])
        reg = S.STUDIES[study]
        arms = arms_override or ", ".join(reg["arms"])
        runs = runs_override or f"{st['done']} of {st['planned']} completed (exit code 0)"
        rows.append({
            "study": label, "report_path": report, "report_and_record_commits": report_commits, "launch_commits": launch,
            "date_utc": f"{st['start']} to {st['end']}", "q": ", ".join(str(q) for q in reg["qs"]), "seeds": _seeds(study),
            "arms": arms, "runs": runs, "phase_budgets": budget_override or reg["budget"], "parents": reg["parents"],
            "results_root": reg["root"], "notes": notes})

    def static_row(label: str, report: str, commits: str, launch: str, date: str, q: str, seeds: str, arms: str, runs: str,
                   budget: str, parents: str, root: str, notes: str = "") -> None:
        rows.append({"study": label, "report_path": report, "report_and_record_commits": commits, "launch_commits": launch,
                     "date_utc": date, "q": q, "seeds": seeds, "arms": arms, "runs": runs, "phase_budgets": budget,
                     "parents": parents, "results_root": root, "notes": notes})

    for rep in ("phase0_audit", "phase1_verifier", "phase2_infra", "phase2_opening_checks", "dreach_reach_mask_check",
                "pilot1_reward_estimator", "pilot2_freeze", "pilot3_continuation_mode", "phaseA_ext", "pilot4_stabilization",
                "protocol_lock_and_rehearsal", "protocol_v1_1_confirmation", "summary"):
        sources.append(C.src(f"{RPT}/{rep}.md"))
    for p in ("protocols/LOCK", "protocols/v2_T2_locked.json", "protocols/v2_T2_locked_v1_1.json"):
        sources.append(C.src(p))

    static_row("Phase 0 audit", f"{RPT}/phase0_audit.md", "1ad3805 (report)", "none (read-only audit)", "2026-10-01 (commit 16:06 UTC)",
               "50, 60 (existing cohorts)", "none run (proposed 10501-10503)", "none",
               "0 training runs; 80 existing final T=2 runs read (4 cohorts)", "none", "none",
               "none (reads RAW experiments/ of the main checkout and the base commit 657f54a)",
               "source: report text; base commit 657f54a")
    static_row("Phase 1 verifier", f"{RPT}/phase1_verifier.md", "b2bfec0 (code); da3d5f8 (data); 8ae6045 (report)",
               "regression runs: 1ad3805 (before), working tree with Phase 1 code (after)", "2026-10-01 (commits 16:44-16:46 UTC)",
               "50, 60", "10501 (regression only)", "none",
               "2 regression runs (smoke budget); 1,152 verifier evaluations (900 probe + 240 checkpoint + 12 calibration); "
               "756 Monte Carlo cells per q", "regression smoke caps A/B/C 40/40/40 (120 updates)", "none",
               "results/v2_pilots/phase1, results/v2_pilots/phase1_regression", "source: report text and CSV row counts")
    static_row("Phase 2 infrastructure and opening checks", f"{RPT}/phase2_infra.md; {RPT}/phase2_opening_checks.md",
               "8f2840e (code); f5347d1 (regression and smoke outputs); 56d934d (report); ca5b148, 53a2fd5, f681bfa (opening checks)",
               "8f2840e", "2026-10-01 (commits 17:00-17:18 UTC)", "50, 60 (opening checks); 50 (smoke)", "10501 (smoke)",
               "smoke: sampled, expected, A_joint, B1_frozen_allnorm, B2_frozen_s1norm, B1_frozen_allnorm+mean",
               "1 regression run + 6 smoke runs (flow checks, not pilot results); opening checks: 1,268 Delta_2 reference points, "
               "252 on-path cases", "regression caps 40/40/40; smoke Phase A 40, Phase B 20",
               "smoke Phase B parent: smoke Phase A state_end_A.pt",
               "results/v2_pilots/phase2_regression, results/v2_pilots/_smoke, results/v2_pilots/phase2_opening",
               "source: report text")
    static_row("dReach reach-mask check", f"{RPT}/dreach_reach_mask_check.md", "77f1fe6 (tool); aaaae3c (report and data)",
               "89cd600 (tool code base)", "2026-10-01 (commit 18:57 UTC)", "50, 60", "none (80 existing final checkpoints)",
               "analytic equilibrium, zero policy, 80 existing final checkpoints", "0 training runs; 168 (candidate, tier) evaluations",
               "none", "none", "results/v2_pilots/dreach_mask_check", "")
    run_row("Pilot 1: reward estimator", "pilot1", f"{RPT}/pilot1_reward_estimator.md",
            "77f1fe6 (tool); 246d550 (records); 08ca3bf (C7); 01704eb (report); e98e434/848b66b (smoothed-game side analysis)",
            "89cd600", "launched with tmux, 40 workers")
    run_row("Pilot 2: joint vs frozen stage 2", "pilot2", f"{RPT}/pilot2_freeze.md",
            "1791687 (code and analysis); 4db246f (records, C7); 18e5950 (report); cd760fd (section 6 recompute: residual band)",
            "1791687", "section 6 recompute at cd760fd uses saved arrays only (no new runs); 60 workers")
    run_row("Pilot 3: continuation mode", "pilot3", f"{RPT}/pilot3_continuation_mode.md",
            "cd760fd (tools); 4fa821a (records); 8b35066 (report)", "cd760fd",
            "stochastic arm is bit-identical to Pilot 2 B2 (20/20); 40 workers in parallel with the Phase A extension")
    run_row("Phase A extension", "phaseA_ext", f"{RPT}/phaseA_ext.md", "cd760fd (tools); 4fa821a (records); 8b35066 (report)",
            "cd760fd", "launched in parallel with Pilot 3 (20 + 40 processes)")
    # Pilot 4: three launches (2a, 2b) + analyses
    a, b = _run_stats("pilot4_A"), _run_stats("pilot4_B")
    sources.extend(a["files"] + b["files"])
    rows.append({
        "study": "Pilot 4: stabilization round (analyses 1a-1d, runs 2a and 2b)", "report_path": f"{RPT}/pilot4_stabilization.md",
        "report_and_record_commits": "c92ee74 (code); e1acf84 (analysis tools); c5d52b1 (records, C7); 0479aaf (report); 609a6f0 (manifests)",
        "launch_commits": "c92ee74", "date_utc": f"{min(a['start'], b['start'])} to {max(a['end'], b['end'])}", "q": "50, 60",
        "seeds": _seeds("pilot4_A"), "arms": "2a: constant, decay; 2b: B2_mean_constant, B2_mean_decay",
        "runs": f"2a {a['done']} of {a['planned']} + 2b {b['done']} of {b['planned']} completed (exit code 0); analyses 1a-1d on existing data",
        "phase_budgets": "2a: Phase A u1201-u1600 (400 updates, decay 3e-4 to 3e-5); 2b: Phase B u1601-u2200 (600 updates, decay 3e-4 to 3e-5)",
        "parents": "2a: phaseA_ext state_u01200.pt; 2b: phaseA_ext state_u01600.pt",
        "results_root": "results/v2_pilots/pilot4_A, results/v2_pilots/pilot4_B, results/v2_pilots/pilot4/analysis",
        "notes": "8 of the 2b manifests carry dirty=true (resolved by the clean re-run); 32 workers per launch"})
    run_row("Dirty-flag re-run (Pilot 4 section 2b q=60 seed 10507 B2_mean_constant)", "pilot4_B_rerun_clean",
            f"{RPT}/protocol_lock_and_rehearsal.md", "5b07293 (consolidation records); 5d50a9d (tool commit of the launch)", "5d50a9d",
            "bit-identical to the original dirty-flagged run")
    static_row("v1.0 lock and calibration", f"{RPT}/protocol_lock_and_rehearsal.md; protocols/v2_T2_locked.md",
               "4bd2214 (lock); 7412c41 (LOCK record); 344154a (calibration tool); 36060ca (records); 3098259 (report)", "4bd2214",
               "2026-10-02 (commits 02:53-03:09 UTC)", "50, 60", "none (calibration without runs)",
               "analytic equilibrium, zero policy (calibration); one pre-lock insurance run in scratch (not kept)",
               "0 kept training runs; 8 calibration evaluations", "protocol: Phase A 1,600 + Phase B 600 updates", "none",
               "protocols/, results/v2_T2_locked/calibration", "source: report text and CSV row counts")
    run_row("v1.0 rehearsal (development seeds)", "rehearsal", f"{RPT}/protocol_lock_and_rehearsal.md",
            "5b07293 (launch; code-identical to 4bd2214); 36060ca (records); 3098259 (report)", "5b07293",
            "dress rehearsal, not evidence for the protocol; 20 parallel processes")
    run_row("v1.0 Check 2 (Phase B code path)", "locked_check2_phaseB", f"{RPT}/protocol_lock_and_rehearsal.md",
            "5b07293 (launch); 36060ca (records)", "5b07293", "seed 10503 at both q; bit-identical to the rehearsal Phase B")
    static_row("Cusp diagnostic", f"{RPT}/protocol_lock_and_rehearsal.md", "344154a (tool); 36060ca (records)", "none (no RL runs)",
               "2026-10-02 (commits 03:09 UTC)", "50, 60", "init seeds 0-4", "supervised fit of the stage-2 actor (5 inits per q)",
               "10 supervised fits (300,000 full-batch steps each); no RL runs", "none", "none",
               "results/v2_T2_locked/cusp_diagnostic", "source: report text and CSV row counts")
    static_row("v1.1 lock", f"{RPT}/protocol_v1_1_confirmation.md; protocols/v2_T2_locked_v1_1.md",
               "431474d (lock); 95c000e (LOCK record, C7); 7c13aee (D1 addendum to the v1.0 report)", "431474d",
               "2026-10-02 (commits 04:24-04:36 UTC)", "50, 60", "none", "none", "0 runs (one pre-lock insurance run in scratch, not kept)",
               "protocol: Phase A 1,600 + Phase B 600 updates", "none", "protocols/", "source: report text")
    run_row("v1.1 re-rehearsal (development seeds)", "rehearsal_v1_1", f"{RPT}/protocol_v1_1_confirmation.md",
            "f6838ec (records and R1-R6 checks)", "95c000e", "checks R1-R6 all pass; 20 parallel processes")
    run_row("Confirmation (fresh seeds)", "confirmation", f"{RPT}/protocol_v1_1_confirmation.md",
            "0588705 (records and analysis); cb0b541 (report)", "f6838ec", "40 parallel processes; verdict PASS")
    df = pd.DataFrame(rows)
    docs = {c: {"definition": d, "units": "", "normalization": "none", "tier": "n/a", "source": "compiled: run records and reports"}
            for c, d in {
                "study": "Study (as listed in the request)", "report_path": "Report(s) of the study, repository-relative",
                "report_and_record_commits": "Commits that added the code, records and report of the study",
                "launch_commits": "Code commit that executed the runs (manifest git.short)",
                "date_utc": "Start of the first and end of the last run (status.json start_time/end_time, UTC), or the commit dates",
                "q": "q values covered", "seeds": "Seeds of the runs", "arms": "Arms of the study (run-directory names)",
                "runs": "Number of runs (completed with exit code 0, counted from status.json) or evaluations",
                "phase_budgets": "Phase budgets of the runs", "parents": "Parent states the runs branch from",
                "results_root": "Results directory of the study", "notes": "Remarks (source: report text unless stated)"}.items()}
    pack.table("T01", df, status="generated", sources=sources, script=f"{MOD}:build_t01", docs=docs, tier="n/a",
               notes="run counts, start/end times and q/seeds/arms are read from status.json and the study registry; commits, budgets and "
                     "parents of non-run studies are compiled from the reports (source: report text)")


# ---------------------------------------------------------------------------------------------
# T02
# ---------------------------------------------------------------------------------------------

def build_t02(pack: C.Pack) -> None:
    """T02: the 14 decisions of request section 5.1, with options, chosen value, evidence, round and first commit."""
    NR = "PI instruction, not recorded in repo"
    R: List[Dict[str, str]] = [
        dict(n=1, decision="Base commit 657f54a and the repository path (Phase 0)",
             options_compared="main (d1b8443), which lacks the T=2 curriculum code, vs 657f54a (branch codex/multistage-experiments-20260927); "
                              "whether a missing `survey` directory is a different checkout (Q0)",
             chosen_value="branch v2-stagewise-pilots created from 657f54a; repository /home/fjiang4/tournament_experiment",
             evidence="phase0_audit.md section 0 and Q0 (proposal and reason); every later report header records base commit 657f54a",
             recorded_in_repo=f"proposal and its use recorded; the approval itself: {NR}", round="Phase 0",
             first_commit="657f54a (base); first v2 commit 1ad3805",
             first_commit_basis="git log 657f54a..HEAD (1ad3805 is the first commit on top of the base)"),
        dict(n=2, decision="q in {50, 60}; development seeds 10501-10510 (three proposed in Phase 0, expanded to ten in Phase 2)",
             options_compared="q: only 50 and 60 were used in all final-protocol T=2 cohorts; seeds: three diagnostic seeds 10501-10503 proposed, "
                              "ten used",
             chosen_value="q = 50 and 60; seeds 10501-10510 at both q",
             evidence="phase0_audit.md section 8 (q and seed proposal, seed inventory); pilot1_reward_estimator.md section 1 (1b: seeds 10501-10510, "
                      "40 runs = 2 arms x 2 q x 10 seeds)",
             recorded_in_repo=f"proposal (3 seeds) and the ten-seed design recorded; the decision to expand: {NR}", round="Phase 0; Phase 2",
             first_commit="89cd600 (Pilot 1 launch with ten seeds); seeds 10501-10503 first used in Phase 1 (b2bfec0)",
             first_commit_basis="manifests of the 40 Pilot 1 runs (git.short = 89cd600)"),
        dict(n=3, decision="Stage-1 training settings: stage-1 training is Phase B only; stage-1 opponent = lagged copy refreshed every 20 updates; "
                           "in frozen mode advantages are normalized over stage-1 rows only; joint mode unchanged",
             options_compared="Q1: which phase is the stage-1 phase (B root-start, or C mixed starts); Q3: stage-1 opponent actions from the existing "
                              "lagged self-play snapshot (proposal: unchanged) while a new freeze snapshot supplies the stage-2 actions of both players; "
                              "Q2: advantage normalization pooled over all rows vs stage-1 rows (both implemented as arms B1 and B2)",
             chosen_value="Pilots 2 and 3 branch from the Phase A end state and run Phase B only (root starts, fixed budget 600); opponent "
                          "unchanged (lagged copy, refresh every 20 updates); frozen mode offers all-rows (B1) and stage-1-rows (B2) normalization; "
                          "joint mode bit-identical to the existing update",
             evidence="phase0_audit.md Q1-Q3 (proposals); phase2_infra.md section 1 (CurriculumPPOv2: both masks None calls the original update) and "
                      "section 2 (tests 3b.3, 3b.4: normalization scope; joint unchanged)",
             recorded_in_repo=f"proposals and implementation recorded; the approvals: {NR}", round="Phase 0; Phase 2", first_commit="8f2840e",
             first_commit_basis="git log -S'adv_norm_scope': first hit 8f2840e"),
        dict(n=4, decision="RNG alignment (A6): `expected` still draws the shocks; `mean` still draws the stage-2 actions and discards them",
             options_compared="draw and discard (streams stay aligned between the paired arms) vs skip the draws",
             chosen_value="draw in both cases (expected: shocks drawn, r_bar used; mean: Beta draws made and discarded, Beta mean executed)",
             evidence="phase0_audit.md Q4 (proposal); phase2_infra.md section 1 (collect_batch_v2) and section 2.1 (alignment table)",
             recorded_in_repo=f"proposal and implementation recorded; the approval: {NR}", round="Phase 0; Phase 2", first_commit="8f2840e",
             first_commit_basis="git log -S'continuation_action_mode': first hit 8f2840e"),
        dict(n=5, decision="On-path rule: the open interval with exact cell masses (Phase 2)",
             options_compared="(a) keep the node-based GL-PMF exact-positivity rule; (b) the continuous support {|d - drift| < 2q}; "
                              "(c) exact cell masses (support {|d - drift| < 2q + h})",
             chosen_value="on-path set = open interval |d - drift| < 2q, weighted with exact cell masses F_xi(b_{i+1} - drift) - F_xi(b_i - drift)",
             evidence="phase2_opening_checks.md section 1c (options a-c); pilot1_reward_estimator.md section 1a (rule as implemented); "
                      "summary.md (decisions taken so far)",
             recorded_in_repo=f"options and implementation recorded; the approval: {NR}", round="Phase 2 (decision before Pilot 1)",
             first_commit="89cd600", first_commit_basis="git log -S'cell_masses': first hit 89cd600 (Pilot 1 launch commit)"),
        dict(n=6, decision="Reward estimator `expected` (after Pilot 1)",
             options_compared="sampled terminal reward vs conditional expected reward w_l + dW F_xi(d + e_own - e_opp) - k e^2 (Pilot 1)",
             chosen_value="reward_mode = expected",
             evidence="pilot1_reward_estimator.md sections 3 and 5 (estimator not chosen in the report; expected lower error in 10/10 pairs); "
                      "pilot2_freeze.md (reward expected in all 60 runs); summary.md (decisions taken so far)",
             recorded_in_repo=f"outcome recorded (summary.md, Pilot 2 manifests); the PI's choice message: {NR}", round="after Pilot 1",
             first_commit="1791687", first_commit_basis="Pilot 2 launch commit (60 manifests, reward_mode = expected); the Pilot 1 'expected' end states are its parents"),
        dict(n=7, decision="Frozen variant B2 (after Pilot 2)",
             options_compared="A joint; B1 frozen with all-rows advantage normalization; B2 frozen with stage-1-rows normalization (Pilot 2)",
             chosen_value="B2 (adv_norm_scope = stage1_rows, stage2_update_mode = frozen)",
             evidence="pilot2_freeze.md closing line of section 5 ('Your decisions were B2 for Pilot 3 and residual minimization for the induced target'); "
                      "summary.md (decisions taken so far)",
             recorded_in_repo=f"outcome recorded; the PI's choice message: {NR}", round="after Pilot 2", first_commit="cd760fd",
             first_commit_basis="Pilot 3 launch commit (arms B2_frozen_s1norm and B2_frozen_s1norm_mean)"),
        dict(n=8, decision="e~1 by residual minimization on the final tier, with a band (after Pilot 2)",
             options_compared="keep the bracketing + Brent induced-target solver; restrict BR to the verifier's grid-plus-vertex candidates; evaluate e~1 on "
                              "the final tier (residual minimization with a band)",
             chosen_value="e~1 = argmin Delta_1(e; e_hat_2) on the final tier; band = {Delta_1 <= min + floor}; the earlier solver is superseded (Appendix S)",
             evidence="pilot2_freeze.md section 1.1, section 6 (decision D2), Appendix S.1 ('needs your decision on whether to keep this solver, restrict BR ..., "
                      "or evaluate e~1 on the final tier')",
             recorded_in_repo=f"options, decision label D2 and implementation recorded; the PI's message: {NR}", round="after Pilot 2",
             first_commit="cd760fd", first_commit_basis="git log -S'stage1_residual_sweep': first hit cd760fd"),
        dict(n=9, decision="The Phase A extension runs in parallel with Pilot 3",
             options_compared="not recorded in the repository",
             chosen_value="both launched together on 2026-10-01 at 23:30:12 (20 + 40 single-threaded processes)",
             evidence="phaseA_ext.md section 1 (launch: 'alongside Pilot 3: 20 + 40 single-threaded processes'); pilot3_continuation_mode.md section 1; "
                      "launch_20261001_233012.json in both result roots",
             recorded_in_repo=f"the parallel launch is recorded; the instruction: {NR}", round="after Pilot 2 (Pilot 3 round)",
             first_commit="cd760fd", first_commit_basis="launch commit of both studies (manifests git.short = cd760fd)"),
        dict(n=10, decision="Continuation mode `mean` (after Pilot 3)",
             options_compared="stochastic vs mean continuation, stage 2 frozen (Pilot 3: neither mode consistently more accurate; mean lower within-run SD at q=50)",
             chosen_value="continuation_action_mode = mean (decision D1, 2026-10-02, before Pilot 4)",
             evidence="summary.md (decisions taken so far, D1); pilot4_stabilization.md header ('Decisions applied: D1 continuation mean'); "
                      "pilot3_continuation_mode.md section 9",
             recorded_in_repo=f"decision D1 recorded; the PI's message: {NR}", round="after Pilot 3 (Pilot 4 round)", first_commit="c92ee74",
             first_commit_basis="Pilot 4 launch commit (2b arms B2_mean_constant / B2_mean_decay)"),
        dict(n=11, decision="After Pilot 3 and the extension: Phase A fixed at 1600 updates with an end-of-phase gate; joint training and Phase C dropped; "
                            "the sampler is not changed",
             options_compared="D2: Phase A budget 400 (Pilots 1-3) vs 1600 (extension); stop rule vs fixed budget with an end-of-phase pass/fail gate; "
                              "D3: keep or drop joint training and Phase C; D4: change the action sampler (inverse-CDF would align the streams but breaks C7) or not",
             chosen_value="D2 Phase A = fixed 1600 updates + end-of-phase gate (thresholds pending until the lock); D3 joint training and Phase C dropped from v2; "
                          "D4 no sampler change",
             evidence="summary.md (decisions taken so far, D2-D4); pilot4_stabilization.md header; phaseA_ext.md section 3; phase2_infra.md section 2.1 (sampler "
                      "alternative and C7)",
             recorded_in_repo=f"decisions D2-D4 recorded; the PI's message: {NR}", round="after Pilot 3 and the extension (Pilot 4 round)",
             first_commit="c92ee74 (Phase A parents at u1200/u1600, no joint/Phase C, sampler unchanged); gate thresholds first applied at 4bd2214",
             first_commit_basis="Pilot 4 launch commit; lock commit of the G-A gate"),
        dict(n=12, decision="After Pilot 4: LR decay in both phases, evaluate the last iterate, no tail averaging",
             options_compared="constant LR vs linear decay 3e-4 to 3e-5 (Pilot 4 sections 2a and 2b); last iterate vs tail averages K in {4, 8, 12, 16} (section 1c)",
             chosen_value="linear LR decay in Phase A (local updates 1201-1600) and Phase B (1-600); evaluated candidate = last iterate; tail averaging not used",
             evidence="pilot4_stabilization.md sections 1c, 2a, 2b, 7; protocols/v2_T2_locked.json (`pipeline.lr_decay`, `not_used` = tail averaging)",
             recorded_in_repo=f"outcome recorded in the protocol; the PI's message: {NR}", round="after Pilot 4 (lock round)", first_commit="4bd2214",
             first_commit_basis="lock commit (git log -S'G-A': first hit 4bd2214; lr_decay windows used in mode `locked`)"),
        dict(n=13, decision="Lock round: the v1.0 gates and pass rule; 20 fresh seeds; peak error reported only",
             options_compared="0929 development targets (5% stage-1 error, 5% stage-2 peak error) vs gates derived from the Pilot 4 distribution tables; "
                              "peak error as a gate vs reported with its decomposition",
             chosen_value="G-A (eta_2 <= 0.005 dW, RMSE/e2*(0) <= 0.05, tail mean/e2*(0) <= 0.02) and G-F (Gmax_full/dW <= 0.01 and |stage-1 error| <= 0.10); "
                          "pass rule >= 18 of 20 per q; fresh seed block 20501-20520 proposed; peak error reported, not gated",
             evidence="protocol_lock_and_rehearsal.md section 2.4 (change log versus the 0929 plan); protocols/v2_T2_locked.json (`gates`, `confirmation`, `change_log`)",
             recorded_in_repo=f"recorded in the protocol; the PI's message: {NR}", round="lock round", first_commit="4bd2214",
             first_commit_basis="lock commit; LOCK record 7412c41"),
        dict(n=14, decision="v1.1 round: Check 1 accepted with hardening; stage-1 criterion moved to S1; G-N added; seeds 20501-20520 confirmed; "
                            "automatic continuation to the confirmation",
             options_compared="Check 1 (literal all-RNG-state criterion failed on three never-consumed process-global states): accept vs change code; stage-1 criterion in G-F "
                              "vs secondary S1; add a numerical-refinement gate G-N; continue to the confirmation automatically if R1-R6 pass",
             chosen_value="D1 Check 1 accepted (training-relevant state defined) and process-global RNGs hardened (D5); D2/D3 stage-1 criterion moved from G-F to S1, "
                          "G-N added (|dev - final| of eta_2 and Gmax_full <= 0.001 dW); D4 seed block 20501-20520 confirmed, exact CIs, v1.0 outcome reported only; "
                          "confirmation runs automatically if R1-R6 all pass",
             evidence="protocol_lock_and_rehearsal.md addendum (D1); protocol_v1_1_confirmation.md sections 1.2-1.3; protocols/v2_T2_locked_v1_1.json "
                      "(`change_log`, `confirmation.status`)",
             recorded_in_repo=f"decisions D1-D5 recorded in the protocol and reports; the PI's message: {NR}", round="v1.1 round", first_commit="431474d",
             first_commit_basis="v1.1 lock commit (D1 addendum appended at 7c13aee; LOCK record 95c000e)"),
    ]
    df = pd.DataFrame(R)
    srcs = [C.src(f"{RPT}/{r}.md") for r in ("phase0_audit", "phase1_verifier", "phase2_infra", "phase2_opening_checks", "pilot1_reward_estimator",
                                             "pilot2_freeze", "pilot3_continuation_mode", "phaseA_ext", "pilot4_stabilization",
                                             "protocol_lock_and_rehearsal", "protocol_v1_1_confirmation", "summary")]
    srcs += [C.src("protocols/v2_T2_locked.json"), C.src("protocols/v2_T2_locked_v1_1.json"), C.src("protocols/LOCK")]
    docs = {"n": "Number of the decision in the request, section 5.1", "decision": "The decision (wording of the request)",
            "options_compared": "Options that were compared (from the reports; 'not recorded' if no file lists them)",
            "chosen_value": "The chosen value as applied", "evidence": "Reports and sections that record the options, outcome or implementation",
            "recorded_in_repo": "What the repository records; 'PI instruction, not recorded in repo' marks what no file records",
            "round": "Work round in which the decision was taken or applied",
            "first_commit": "First commit at which the decision applied (launch commit of the first run that used it, or the commit that implemented/locked it)",
            "first_commit_basis": "How the first commit was determined (git log -S pickaxe on the code, or launch commits recorded in run manifests)"}
    pack.table("T02", df, status="generated", sources=srcs, script=f"{MOD}:build_t02", tier="n/a",
               docs={k: {"definition": v, "units": "", "normalization": "none", "tier": "n/a", "source": "compiled: reports, protocols and git history (source: report text)"}
                     for k, v in docs.items()},
               notes="compiled from the reports and protocols (source: report text); first commits from `git log -S` on the v2 code and from run manifests; "
                     "where no file records the PI's decision message the cell says 'PI instruction, not recorded in repo'")


# ---------------------------------------------------------------------------------------------
# T03
# ---------------------------------------------------------------------------------------------

def build_t03(pack: C.Pack) -> None:
    """T03: the 0930 plan items against what was done, with the reason and source of every difference."""
    NR = "PI instruction, not recorded in repo"
    R = [
        dict(plan_item="Goal: improve T=2 accuracy in recovery and verification so that T=2 can serve as the algorithm-development and calibration "
                       "environment; then move to T=3",
             planned="improved recovery and verification at T=2; T=3 afterwards",
             executed="T=2 locked protocol v1.1; fresh-seed confirmation 20/20 and 20/20 primary passes (verdict PASS); T=3 not started",
             difference="T=3 not started in this body of work", reason="scope of the work packages: T=3 is out of scope of the T=2 reports",
             source="protocol_v1_1_confirmation.md (verdict); summary.md open question 7"),
        dict(plan_item="Method change (a): full-domain grid-based MPE metric Gmax_full, keeping EXP_root, dReach, Delta_max_all, dFull",
             planned="Gmax_full added; the four existing aggregates kept",
             executed="Phase 1: utils/v2_metrics.py wraps the unchanged verifier; Gmax_full with (t*, d*), eta_2, invariants, on/off-path split, "
                      "closed-form recovery metrics, sigma arrays, storage; calibration on the analytic equilibrium and the zero policy",
             difference="additions: on-path rule (open interval, exact cell masses), invariant residuals logged on every call, recovery metrics at every checkpoint, "
                        "tier comparison (development / dev_2x / final)",
             reason="on-path/off-path separation and the metric checks were part of the Phase 1 work (phase1_verifier.md sections 2-4); the approval of each addition: " + NR,
             source="phase1_verifier.md sections 1-4; phase2_opening_checks.md section 1c"),
        dict(plan_item="Method change (b): stagewise backward learning with frozen continuation",
             planned="Phase A trains stage 2; stage 2 is frozen; stage 1 trained against the frozen continuation",
             executed="Phase 2: run/run_v2_stagewise.py with the flags reward_mode, stage2_update_mode, adv_norm_scope, continuation_action_mode; full-state checkpoints; "
                      "locked pipeline: Phase A (1,600 updates) -> freeze -> Phase B (600 updates, stage 1 only)",
             difference="joint training and Phase C dropped from v2 (D3); continuation mode mean (D1); Phase A fixed at 1600 updates with an end-of-phase gate (D2); "
                        "LR decay in both phases; sampler unchanged (D4)",
             reason="Pilots 1-3, the Phase A extension and Pilot 4 results (stage-2 peak plateau from about u800; stage-1 oscillation; LR decay lowers late fluctuation); "
                    "decisions recorded as D1-D4 in summary.md; approvals: " + NR,
             source="summary.md (decisions taken so far); phaseA_ext.md section 3; pilot4_stabilization.md sections 2a, 2b, 7; protocols/v2_T2_locked_v1_1.json"),
        dict(plan_item="Pilot definitions: Delta_2(d), G_2 = Delta_2, eta_2",
             planned="Delta_2 one-step gap, G_2 = Delta_2 at the terminal stage, eta_2 = max Delta_2",
             executed="Delta_t, G_t = V_t^BR - V_t^mean, eta_T_over_dw implemented; G_2 = Delta_2 checked on every call (maximum violation 0)",
             difference="none", reason="not applicable", source="phase1_verifier.md sections 2.1 and 4.1"),
        dict(plan_item="Pilot 1: terminal-only; sampled vs conditional expected reward; compare peak error, RMSE, tail effort and eta_2; only the estimator changes",
             planned="Phase A only, two estimators, 2 q x 3 paired seeds (0929 plan)",
             executed="40 runs (2 arms x 2 q x 10 seeds 10501-10510), Phase A 400 updates, fixed budget; metrics: peak error (signed, absolute), RMSE, tail mean and max, "
                      "symmetry, eta_2, on/off-path Delta_2, sigma_2; stage-1 untrained",
             difference="ten seeds per q instead of three; pairing controls initialization, shocks, starts and minibatches but not action noise (learner/opponent Beta "
                        "streams desynchronize by update 73)",
             reason="seed expansion: " + NR + "; stream desynchronization: numpy's rejection-based Beta sampler (Phase 2 report section 2.1)",
             source="pilot1_reward_estimator.md sections 1, 3.5, 4; phase0_audit.md section 8; phase2_infra.md section 2.1"),
        dict(plan_item="Pilot 2: from the same stage-2 checkpoint, joint vs frozen stage 2; compare stage-2 drift, stage-1 recovery, Gmax_full, dReach, EXP_root "
                       "and the frozen stage's output drift",
             planned="joint vs frozen from one parent",
             executed="60 runs: 20 Pilot 1 'expected' end-of-A states x {A joint, B1 frozen all-rows normalization, B2 frozen stage-1-rows normalization}; Phase B 600 updates; "
                      "all listed metrics plus the stage-1 decomposition (superseded solver, then residual band)",
             difference="two frozen variants instead of one (normalization scope); decomposition of the stage-1 error added; drift of the live network reported for information",
             reason="advantage normalization in frozen mode is a choice inside the frozen arm (Phase 0 Q2); decomposition requested with Pilot 2 (section 6, D2); approvals: " + NR,
             source="pilot2_freeze.md sections 1, 2, 6; phase0_audit.md Q2, Q5"),
        dict(plan_item="Pilot 3: stochastic vs mean continuation, stage 2 frozen; judged by stage-1 accuracy and stability",
             planned="two continuation modes with stage 2 frozen",
             executed="40 runs on B2 frozen: stochastic (bit-identical to Pilot 2 B2 in 20/20) vs mean; stage-1 error, decomposition, within-run SD and across-seed SD",
             difference="the stochastic arm is the same configuration as Pilot 2 B2 (same parent, same code path) and was re-run: it is bit-identical to Pilot 2 B2 "
                        "in 20/20 runs; neither mode is consistently more accurate",
             reason="design of the pilot: the stochastic arm doubles as a reproducibility check of the branching", source="pilot3_continuation_mode.md sections 1, 2, 9"),
        dict(plan_item="If the pilots improve accuracy, run the formal two-stage experiment",
             planned="formal two-stage experiment after the pilots",
             executed="additional rounds before the formal experiment: Phase A extension to 1600 updates, Pilot 4 stabilization round (analyses 1a-1d, LR decay 2a/2b), "
                      "protocol v1.0 lock and dress rehearsal on the development seeds (Check 1 failed on its literal criterion), protocol v1.1 (S1, G-N, RNG hardening, "
                      "re-rehearsal R1-R6), then the confirmation on 20 fresh seeds per q",
             difference="extra rounds (extension, Pilot 4, two protocol versions, two rehearsals) between the pilots and the formal experiment; the confirmation seed block "
                        "was fixed in the v1.1 round",
             reason="at 400 Phase A updates the stage-2 peak error was -13% and -10% (median) and the stage-1 error still oscillated by a few percent (pilot reports, "
                    "summary.md open questions 1-3); the decision to run each extra round: " + NR,
             source="summary.md; phaseA_ext.md; pilot4_stabilization.md; protocol_lock_and_rehearsal.md; protocol_v1_1_confirmation.md"),
        dict(plan_item="The 0929 plan specified 2 q x 3 paired seeds for the pilots",
             planned="2 q x 3 paired seeds (0929 plan)", executed="2 q x 10 paired seeds in every pilot and in the Phase A extension (seeds 10501-10510)",
             difference="ten seeds instead of three", reason=NR + " (the 0929 plan itself is not stored in the repository)",
             source="phase0_audit.md section 8 (proposal of three seeds); pilot1_reward_estimator.md section 1; protocol_lock_and_rehearsal.md section 6 item 6"),
        dict(plan_item="(0929 development targets, 5% stage-1 and 5% stage-2 peak error; see T41)",
             planned="development targets of 5% stage-1 error and 5% stage-2 peak error",
             executed="replaced by the locked gates G-A and G-F (and S1 in v1.1); peak error reported, not gated",
             difference="targets not used as gates", reason="thresholds set from the Pilot 4 distribution tables; peak error decomposed instead (smoothing share and supervised floor)",
             source="protocol_lock_and_rehearsal.md section 2.4; protocols/v2_T2_locked_v1_1.json change_log"),
    ]
    df = pd.DataFrame(R)
    srcs = [C.src(f"{RPT}/{r}.md") for r in ("phase0_audit", "phase1_verifier", "phase2_infra", "phase2_opening_checks", "pilot1_reward_estimator", "pilot2_freeze",
                                             "pilot3_continuation_mode", "phaseA_ext", "pilot4_stabilization", "protocol_lock_and_rehearsal",
                                             "protocol_v1_1_confirmation", "summary")] + [C.src("protocols/v2_T2_locked_v1_1.json")]
    docs = {"plan_item": "Item of the 0930 plan (request section 5.2)", "planned": "What the plan specified",
            "executed": "What was done", "difference": "Difference between plan and execution ('none' if none)",
            "reason": "Reason recorded in the repository, or 'PI instruction, not recorded in repo'", "source": "Report sections or files that record the execution"}
    pack.table("T03", df, status="generated", sources=srcs, script=f"{MOD}:build_t03", tier="n/a",
               docs={k: {"definition": v, "units": "", "normalization": "none", "tier": "n/a", "source": "compiled from the reports (source: report text)"} for k, v in docs.items()},
               notes="compiled from the reports and protocols (source: report text); the 0929/0930 plans are not stored in the repository, so the 'planned' column quotes the request")


def build() -> None:
    pack = C.Pack("sec_front")
    build_t01(pack)
    build_t02(pack)
    build_t03(pack)
    pack.save_fragment()
