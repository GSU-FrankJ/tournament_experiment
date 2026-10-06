> **Provenance of this file (read first).** This is **not** the PI's delivered file. It is the assistant's transcription of the prompt text as it was pasted into the publication session on 2026-10-05 (the session that built this folder). The PI inbox (`.claude/pi_inbox/`) did not exist, and no delivered copy exists on the machine. Line wrapping and emphasis markers may differ from the delivered file; the section structure and decisions (D1-D4) are as pasted. If the PI supplies the delivered file, it should replace this one (the `SHA256SUMS` of `pi_record/` then changes).

# T=2 refinement work (R1, v2.0, R2b, R2c): publish one folder on `main` with a PI-facing summary report and an evidence pack

The four rounds that answered the PI's note of 2026-10-02 (`Multistage100226.docx`: six candidate methods and two diagnostics for the T=2 pipeline) are finished: R1 (`reports/v2/refine/`), protocol v2.0 with its confirmation (`reports/v2/protocol_v2_0_confirmation.md`), R2b (`reports/v2/refine_r2b/`) and R2c (`reports/v2/refine_r2c/`). This round publishes them the way `reports/v2/` publishes the T=2 v2 work: everything on `main`, one entry folder, a PI record, and an evidence pack whose every item carries its source path and SHA-256. Nothing is re-run, no protocol or locked file changes, no existing report is moved, renamed or edited (addenda only).

Steps:

1. P0 bring `main` up to date and start the publication branch (§1).
2. P1 build the folder `reports/t2_refine_100526/`: index, PI record, evidence pack, figures (§2).
3. P2 write `reports/t2_refine_100526/100526report.md` (§3), have it fact-checked (§4), then merge, tag, push (§5) and STOP (§6).

Repository `/home/fjiang4/tournament_experiment`; Python `/home/fjiang4/tournament_experiment/.venv/bin/python`; `pytest tests`. Standing rules: provenance discipline (every number in the report cites a path, and the evidence pack carries the SHA-256 of that file); no deletion or overwriting; no edits to `protocols/`, the locked entry point, `utils/v2_continuation.py`, `tools/v2/confirmation_analysis*.py`; no `.pt`, `train_history.json` or weight exports in git. Pushing is authorised only in §1 and §5.

If any step fails, or anything does not match this prompt, stop at that point and report. Do not work around it.

## 0. Decisions (binding)

* D1. The four rounds are closed as reported. R2c selected no arm under its pre-registered rule, so there is no v2.1; the locked T=2 solver is v2.0 (`protocols/v2_T2_locked_v2_0.json`, tags `t2-v2-lock-v2.0`, `t2-v2-confirmation-v2.0`, `t2-v2-main-v2.0`). Method 5 (pathwise fine-tuning) is closed as negative at matched budgets; the censored likelihood is not adopted; the peak-focused start distribution is not adopted at T=2 (it meets the criterion at q = 50 only, and the tail constraint binds at q = 60) and is carried as a design input for the T=3 terminal stage.
* D2. Principle of the folder. As in `reports/v2/README.md`: nothing is moved, renamed or copied except into the evidence pack, which holds copies of small source files with their original paths and SHA-256; the round reports stay where they are and are cited by path. Report-level corrections to earlier reports, if the fact-check finds any, are dated addenda in those reports, never edits.
* D3. The report has two readers. `100526report.md` is written in English, organised by the PI's note (six methods, two diagnostics), with a one-page Chinese executive summary (中文摘要) at the top for the author of the note. Numbers in the summary are the same numbers as in the body, with the same sources.
* D4. Inputs from the PI. Before you start, the PI places in `/home/fjiang4/tournament_experiment/.claude/pi_inbox/`: `Multistage100226.docx` (the note), and the four prompts as delivered: `12_t2_refine_round_prompt.md`, `13_v2_0_lock_confirmation_prompt.md`, `14_r2b_terminal_stage_followup_prompt.md`, `15_r2c_sampler_pilot_v2_1_prompt.md`. If a file is missing, use the copy of that prompt saved in the round's worktree or results (`PROMPT.md` or the review folders) and state the source; if neither exists, leave the row and say so. The PI's decisions at each gate are the §0 sections of those prompts plus the two in-round decisions already recorded in the reports (check (ii) kept open in R1 — `reports/v2/refine/01_preregistration.md` §4 item 6; D-R7 in the v2.0 round — `reports/v2/protocol_v2_0_confirmation.md` §6 deviation 1).

## 1. P0 — `main` and the publication branch

Record everything in `reports/t2_refine_100526/pi_record/00_publication_log.md` (commands, hashes, verbatim outputs).

1. State: `git ls-remote origin` for `main`, `v2-t2-refine`, `v2-t2-r2b`, `v2-t2-r2c` and the `t2-v2*` tags; the local heads of the three branches. Verify the chain is linear: `main` (`f02a256`) is an ancestor of `v2-t2-refine` (`b55d389`), which is an ancestor of the `v2-t2-r2b` head, which is an ancestor of the `v2-t2-r2c` head (`git merge-base --is-ancestor`). If it is not linear, stop and report.
2. If `v2-t2-r2c` is not on `origin`, push it (fast-forward, no force).
3. Fast-forward `main` in the primary checkout: record `git status --porcelain` verbatim. If the only tracked modification is `SESSION_STATE.md`, `git stash` it, do the merges, `git stash pop`, and record both; if any other tracked file is modified, stop and report. Then `git fetch origin`, `git merge --ff-only origin/main`, `git merge --ff-only v2-t2-r2c`; verify `git rev-parse main` equals the `v2-t2-r2c` head. Do not push `main` yet (§5).
4. Create `t2-refine-pack` from `main` in a new worktree `.claude/worktrees/t2-refine-pack`; all P1–P2 work happens there.

## 2. P1 — the folder `reports/t2_refine_100526/`

### 2.1 `README.md` (the index, modelled on `reports/v2/README.md`)

Scope and status; reading order (the summary report first, then the four round reports in order, then the PI record and the evidence pack); the branches, commits and tags of each round in one table (R1: code `32a8c21`, pre-registration `6c902db`, final record `155cdec`; v2.0: lock `1d6d4d0`, LOCK record `f2d616c`, rehearsal/launch `d2e377d`, confirmation records `85c294e`, report `6e216e2`, publication head `b55d389`; R2b: code `1ff99bd`, launch `d581b3c`, final `62ecc43`; R2c: code `58c2671`, results `3ad1b07`, final head as pushed — verify every hash with `git log` and correct any that differs, saying so); the paths outside the folder (protocols, entry point, continuation module, analysis scripts, results roots, and the worktrees that hold the untracked `.pt` and `train_history.json` files, as `reports/v2/README.md` does); the known caveats (§3 item 7).

### 2.2 `pi_record/`

* `plans/Multistage100226.docx` and a `.md` conversion (pandoc), with the note's six methods and two diagnostics numbered as the report numbers them.
* `prompts/12_…md` to `15_…md` as delivered (D4), plus a `README.md` table: prompt → round → the PI's decisions at the gate that followed (taken from the §0 of the next prompt and the two in-round decisions of D4) → the outcome.
* `SHA256SUMS` of everything in `pi_record/`.

### 2.3 `evidence/` (the pack)

A builder `tools/v2/report/build_t2_refine_pack.py` that copies each item from its source path, verifies that the copy's SHA-256 equals the source's at the current commit, and writes `evidence/manifest.csv` (item id, title, round, source path, SHA-256, source commit, size) and `evidence/SHA256SUMS`. Items, at minimum:

* Protocols and locks: `protocols/v2_T2_locked_v2_0.json` and `.md`, `protocols/LOCK`, `results/v2_T2_locked/v2_0/protocol_diff_v1_1_to_v2_0.json`, `results/v2_T2_locked/v2_0/continuation_check_v2_0.json`, `results/v2_T2_locked/v2_0/verifier_numerics_note.md`, `results/v2_refine/continuation_check.json`.
* v2.0 confirmation: `results/v2_T2_locked/confirmation_v2_0_analysis/` (`verdict.json`, `pass_counts.csv`, `per_run.csv`, `s1_summary.csv`, `distributions.csv`, `reported_metrics.csv`, `agreement.csv`), the rehearsal checks `results/v2_T2_locked/rehearsal_v2_0_checks.json` and `rehearsal_v2_0_checks/c7_canonical_reference.txt`, the v1.1 confirmation tables for comparison (`results/v2_T2_locked/confirmation_analysis/per_run.csv`, `s1_summary.csv`, `reported_metrics.csv`), and the gates.json of the failed run `results/v2_T2_locked/confirmation_v2_0/q50/seed30510/gates.json`.
* R1: `results/v2_refine/analysis/decision_inputs.csv`, `stage1_criterion.csv`, `stage1_dispersion.csv`, `stage1_adv_ratio.csv`, `stage1_per_run.csv`, `stage2_criterion.csv`, `stage2_annealing.csv`, `decision_method5_pairs.csv`, `v11_reproduction_checks.json`, `d1_clamp/d1_flags.csv`, `d2_verifier_sensitivity/detection_limits.csv` (and the per-candidate table), `parents_A_checks.json`, `stage1_base_checks.json`, `stage2_base_checks.json`.
* R2b: `results/v2_refine_r2b/analysis/decision_inputs.csv`, `criterion.csv`, `per_run.csv`, `tail.csv`, `trajectory_per_run.csv`, `v20_reproduction_checks.json`, `launch_checks.json`, the seed-30510 diagnostic tables under `results/v2_refine_r2b/diag_30510/`.
* R2c: the corresponding `results/v2_refine_r2c/analysis/` tables (`per_run`, `criterion`, `selection.json`, the tail table), `v20_reproduction_checks.json`, the launch checks.
* Round reports (copies, for completeness of the pack): `reports/v2/refine/{summary,06_decision_inputs}.md`, `reports/v2/protocol_v2_0_confirmation.md`, `reports/v2/refine_r2b/{summary,05_decision_inputs}.md`, `reports/v2/refine_r2c/{summary,03_selection}.md`. Large or untracked files are not copied; the manifest lists them with their location (worktree path) and SHA-256 where the file exists on disk.

### 2.4 `figures/`

Copies (with the manifest rows) of the figures the report uses: the stage-1 error distributions v1.1 vs v2.0 on fresh seeds; the stage-2 peak error per arm for R1, R2b and R2c; the D2 detection curves; the seed-30510 trajectory figure; the wave-P objective/FOC curves. If a needed figure does not exist, generate it from the pack's tables with a script under `tools/v2/report/` and say so.

## 3. P2 — `100526report.md`

Every number cites the evidence item (`evidence/manifest.csv` id) or a path; verdicts are stated exactly as computed in the round reports; nothing is estimated. Sections:

0. 中文摘要 (one page). 对应 PI 便签的六个方法和两项诊断，逐项一句话给出：做了什么、结果（配对差与 CI）、决定；v2.0 的 fresh-seed 结果（stage-1 误差前后对比）；stage-2 peak 的结论；建议（T=2 在 v2.0 收口，进入 T=3）。
1. Scope and status. What the note asked; the four rounds; the current locked solver (v2.0) and its tags; what is on `main` after this round.
2. The note's items and how each was tested. One table: item → mechanism as implemented → arms and rounds → pre-registered criterion → outcome → decision.
3. Protocol v1.1 → v2.0. The one pipeline change (expected continuation, with its definition and the fixed integration rule), check (ii) and its resolution, gate G-S and its admission rule, the seed block; the confirmation verdict table (per q passes, exact CIs, the failed run and its nature); stage-1 error on fresh seeds before (v1.1, 20501–20520) and after (v2.0, 30501–30520): median, max, SD per q; the stage-1 dispersion and advantage-SD ratios from R1.
4. Results by method (six subsections): paired tables per q with CIs, criterion parts (a)/(b), tail statistics where relevant, and the decision. For method 5 include the matched-budget comparison and the FOC/RMSE observation; for method 6 the full evidence chain (R1 pilot → v2.0 confirmation).
5. Diagnostics D1 and D2. Headline tables (M1/M2 by phase; detection limits per family and gate) and what they imply for the gates and for T=3.
6. The stage-2 peak. The evidence that the remaining gap is a cusp-representation / weighting limit (pathwise negative result; peak-focused starts; the q-asymmetry; the tail trade-off with the per-run tail values of `A_peak50`); the seed-30510 case; the R2c outcome and the selection rule's verdict; what is carried to T=3.
7. Known issues, limitations and deviations (consolidated from the four round reports): the D1 clamp flags in Phase A; the verifier numerics item; the stale `utils/v2_continuation.py` docstring (SHA-locked); the v1.1 report pack entries T05/T56/T57 and the waived refresh; the registry test; the `pytest tests` note; every accepted deviation (D-R7, the 17 R1 re-runs, the R2b worktree repair, the R2c tool-commit order, the renamed test file).
8. Implications for T=3 (short, descriptive): the terminal stage shares the closed form and the sampler question; stages 1–2 need an induced-target accuracy metric because the dynamic-deviation gates are second-order (D2); the decisions pending on the PI side are listed, not made.
9. Evidence pack and reproduction. How to read the manifest; the commands to rebuild the pack and to re-run each round's analysis script unchanged.

PI-side readings to verify, not to copy. The PI read these from the files; check each against the evidence item and report any disagreement in the fact-check (§4) rather than editing the number: R1 `B_expcont` S1 paired difference −0.02386 [−0.04061, −0.009831] (q=50) and −0.03783 [−0.06321, −0.01472] (q=60), dispersion ratios 0.3464 / 0.2687, advantage-SD ratios 0.0656 / 0.0531, wall 1.084×; v2.0 confirmation 19/20 [0.7513, 0.9987] and 20/20 [0.8316, 1], G-S 20/20 both q, stage-1 |error| median 0.0131 / 0.0117, max 0.0464 / 0.0324, SD 0.0211 / 0.0171, against v1.1's median 0.0378 / 0.0455 and max 0.0862 / 0.1530; the failed run q=50 seed 30510 with η₂ 0.00580334, peak error −0.1458, on-path 0.0058 against off-path 0.00082, smoothed share 0.279; check (ii) 3.229e-05 / 1.960e-05 ΔW on the standard final tier, 4.982e-07 / 4.100e-07 on the refined one, table convergence ≤ 4.76e-09 / 3.22e-09; D1 phase-A M1 median 0.001385 pooled and M2 3/20 with max share 0.02353, no hits at |d| < 2q; D2: G-F not reached by a 15% stage-1 error (max 0.00717 / 0.003247), stage-2 amplitude detected at 10% / 15%; R2b `A_peak50` −0.0235 [−0.0410, −0.0040] / −0.0118 [−0.0230, −0.0010], tail violations at q=60 seeds 10503 / 10504 / 10510 (0.0204 / 0.0212 / 0.0221), runs within 0.05: 2 → 8 and 5 → 8; pathwise vs matched controls −0.0026 / +0.0006 (LR 3e-5) and +0.0127 / +0.0029 (LR 3e-4), FOC residual falling 5.05e-04 → 3.6e-04 while RMSE falls 0.0201 → 0.0149 and 0.0198 → 0.0134; `A_ctrl200_lr3e-4` vs parent −0.019 (8/10) at q=50; R2c `A_peak35` −0.0308 [−0.0486, −0.0136] / −0.0054 [−0.0164, 0.0071], `A_peak40` −0.0235 [−0.0397, −0.0077] / −0.0065 [−0.0187, 0.0065], `A_peak50_late400` −0.0143 [−0.0303, 0.0005] / −0.0059 [−0.0179, 0.0058], `A_peak50_late800` −0.0178 [−0.0335, −0.0013] / +0.0002 [−0.0156, 0.0151], largest tail mean 0.0194, no arm selected.

## 4. Fact-check

Before the merge: an independent pass (a reviewer that did not write the report) traces every number in `100526report.md` and `README.md` to its evidence item, checks the SHA-256 in the manifest against the file at the current commit, and reruns `build_t2_refine_pack.py` to confirm the pack is reproduced byte for byte. Record the ledger under `reports/t2_refine_100526/pi_record/01_factcheck.md` (statements checked, disagreements found, how each was resolved). A disagreement with a round report is resolved by a dated addendum in that report, never by editing it; a disagreement with the PI-side readings above is reported as such.

## 5. Merge, tag, push

1. Commit the folder and the builder on `t2-refine-pack` (lightweight files only; the pack is small by construction).
2. `main`: fast-forward to the `t2-refine-pack` head (precondition and stash rule as in §1.3); verify.
3. Annotated tag `t2-refine-100526` on that head.
4. Push `main`, `t2-refine-pack` and the tag (fast-forward only, never force). Record `git ls-remote` for all of them in the publication log and commit the log as the last commit (then push `main` once more, fast-forward).
5. Append one paragraph to `reports/v2/summary.md` and one row to the reading-order table of `reports/v2/README.md` pointing to `reports/t2_refine_100526/100526report.md` (addenda, no edits of existing lines).

## 6. STOP

Stop after the final push. No experiments, no protocol changes, no T=3 work. The next round is mine to decide.
