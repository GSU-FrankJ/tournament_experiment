# T3-R3, P2: pre-registration of the development wave

Date: {{date}}. Branch `t3-r3`. **Generated** from `reports/t3/r3/report_scripts/04_preregistration.template.md` by `build_report.py`; the tables by `prereg_tables.py`. This file is committed and pushed **before any development run** and before the diagnostic runs; the wave is launched from the commit that holds it, on a clean tree. Nothing in it is tuned after the launch.

**Code version.** The final code commit of P2 is `{{code_commit}}` (`run/ms_retry.py`, `run/run_ms_stagewise.py`, `tools/t3/t3r3_*.py`); C-LEGACY ran at it (section 8). From that commit to the launch only `reports/`, `results/`, `docs/STATE.md` and `reports/t3/README.md` change. The binding texts are the prompt (`reports/t3/r3/pi_record/28_t3_r3_prompt.md`, D5-D7), the PI's reply at gate G1 (`pi_record/28b_g1_reply.md`) and the accepted proposal (`02_proposal.md` sections 2-3).

## 1. Arms, keys, seeds, names

{{table:prereg_arms}}

- **Seeds.** 70501-70520 at q = 50 and q = 60, the same seeds in every arm (40 runs per arm). The diagnostic runs are (q = 60, seed 60518), once for each candidate arm, into `results/t3_r3/diagnostic/q60/seed60518/{arm}/`; they are reported separately and **never counted**. Run names `t3r3_q{q}_s{seed}_{arm}`; run directories `results/t3_r3/dev/q{q}/seed{s}/{arm}/`. The launcher refuses any other seed (tested): 60501-60520 (except the diagnostic seed at q = 60), 80501-80540 (reserved for the later confirmation, never trained), 50500-50520, 40501-40520, 10501-10530, 30501-30520.
- **Control configs.** The locked T3-1.0 settings: `tools/t3/t3_configs.build_bw_config` with `seed`, `run`, `record.run`, `record.seed`, `record.output_dir` and `pilot` = "t3_r3" set; the config with the scheme's own keys removed and the D4 fields reset (`t3c_launch.normalise`) hashes to the lock's `reference_config_sha256` of its q (checked for all 122 runs by the launcher).
- **Candidate configs** differ from the control's config of the same (q, seed) only in the scheme's own keys and the run fields:

{{table:prereg_cfgdiff}}

- **The 122 planned runs** (120 development + 2 diagnostic), their config SHA-256 and planned init digests (attempt 1; attempt 2 for the candidate arms) are in `reports/t3/r3/dev_plan.csv` (SHA-256 of the file: `{{plan_sha}}`); the table lists them.

{{table:prereg_plan}}

## 2. Order before the launch, and the launch

1. The independent review of the P2 code, its fixes committed (`03_checks.md`).
2. C-LEGACY at the final code commit (section 8).
3. This pre-registration, committed and pushed.
4. The launch: one wave of 122 runs, at most 40 single-threaded workers, in tmux, `python tools/t3/t3r3_launch.py --workers 40 --code-commit <sha>` from the commit that holds this file, on a clean tree (the launcher refuses a dirty tree outside `results/`).

## 3. The schemes

**Candidate A (`bwA`)**, exactly as `02_proposal.md` section 2.1: `stage3_retry = {"delta_gt": 0.005, "max_retries": 1}`; the trigger is read at the local-1600 development check of the stage-3 phase and fires iff `Delta` > 0.005, or `Delta`, `s` or `C` is not finite, or `s` <= 0; one in-place re-initialisation of the actor and the critic and of both Adam states, from `SeedSequence([seed, q, ns_init, 2])` and the five streams `SeedSequence([seed, q, ns_name, 2])`; attempt 2 is frozen whatever its own check says; the artefacts of attempt 1 are moved to `attempt1/` (never overwritten, never deleted) and `retry.json` is written; a run whose trigger does not fire writes exactly the files, columns and rows of `bw`.

**Candidate B (`bwAB` = `bwA` + `terminal_clamp_likelihood = "censored"`)**, exactly as section 2.2: in the stage-3 phase only, for both attempts; it reuses `utils/beta_tail.py`, `collect_batch_v2(..., clamp_likelihood="censored")` and `CurriculumPPOv2.update(..., clamp_side=...)`; the wiring in `run/run_ms_stagewise.py` is all that was needed (no further change). Test: at T = 3, in the stage-3 phase, the stored log-prob of every flagged row equals the censored log-mass of `utils/beta_tail.py` and every unflagged row the density path (`tests/test_t3r3_retry.py`).

**Paper wording for A (verbatim, G1 reply 2.5):** "a verifier-gated restart of the terminal stage: when the development-tier one-step deviation gain at the end of the stage-3 phase exceeds the gate's limit 0.005, stage 3 is re-initialised once and retrained, and the second attempt is kept; its cost is part of the method's cost."

## 4. Budgets and cost (fixed in advance)

{{table:budgets}}

The measured wall time per arm and its ratio to `bw` is reported in addition to updates and policy rows (G1 reply 2.4).

## 5. Post-launch checks (`results/t3_r3/dev/launch_checks.json`)

Per run, by `tools/t3/t3r3_checks.py` (a deterministic failure is a stop-and-report):

- **Basics:** exit code 0; the files of a `bw` run, plus `retry.json` and `attempt1/` for a run that fired, neither for a run that did not; the manifest at the launch commit with a clean tree; the global-RNG assertion `ok`; the config equal to the plan (SHA-256 with the output directory made relative) and the lock/arm check.
- **Trigger:** recomputed from the check file it read; agrees with `retry.json`; `bw` has neither.
- **Budgets:** the frozen path has the `bw` budget (updates 1600 / 1600 / 600, policy rows 819,200 / 819,200 / 307,200, 76,000 minibatch steps); a run that fired also has 5,400 updates, 2,764,800 policy rows, 108,000 minibatch steps and 216 development checks in total and, in `attempt1/`, 64 exports, 1,600 rows and 64 checks.
- **Schedules:** the learning rate at every update (attempt 1 as well), the snapshot refreshes, 152 exports (`u00025` ... `u03800`), 64 / 64 / 24 checks.
- **C-INIT:** the recorded init digests equal the planned ones (attempt 1; attempt 2 for a run that fired); the attempt-1 digest is equal across the arms of one (q, seed); the attempt-2 digests of `bwA` and `bwAB` are equal.
- **C-REPLAY:** the evaluator reproduces the freeze arrays bit for bit.
- **C-PAIR:** `bwA` against `bw` of the same seed (`tools/t3/t3r3_pair.py`): where the trigger did not fire the run is bit-identical to `bw` with the scheme's records only in new files; where it fired, attempt 1 equals the `bw` run's stage 3 up to the check (the 64 exports, the stage-3 rows of `ms_updates.csv` and `ms_checks_stage3.csv` without wall-clock columns, the bin maps, the training state). The diagnostic runs (both arms) are compared with T3-R2's original run of seed 60518 in the same way (binding). `bwAB` against `bw` is computed and reported, deciding nothing.

## 6. Evaluation, the target and the selection

- **Evaluation.** `tools/t3/t3_eval.evaluate_run` on the final and development tiers through `tools/t3/t3r3_eval.py`; the evaluated policy is the one the scheme freezes (attempt 2 for a run that fired). Gates and run pass as in T3-1.0: exit 0, global-RNG assertion ok, a valid evaluation on both tiers, G-A (`eta_3` <= 0.005, `rmse_pos_T_over_eT0` <= 0.05, `tail_mean_T_over_eT0` <= 0.02), G-F (`Gmax_full_over_dw` <= 0.01), G-N (development minus final <= 0.001 for `eta_3` and `Gmax_full`), every `<=` inclusive at full float64 precision. S3 of a retrained run is computed on the frozen attempt.
- **Failure counting (D6).** Infrastructure failures (a negative return code, an out-of-memory kill, ENOSPC) are recorded in `infra_events.json` and the run is re-run once; a second one stops the arm without a verdict. Any other nonzero exit and any global-RNG violation is a failed run. An evaluation error or an invalid evaluation is a failed run unless a re-run of the locked evaluator succeeds. A re-initialisation inside a run is part of the run, never a re-run. Every planned run counts.
- **The target (the PI's).** A candidate arm reaches it iff at least 38 of its 40 runs pass and at least 18 of 20 at each q. The pass probabilities for comparison are in section 10.
- **Selection.** Only the candidate arms are eligible. In the order `bwA`, then `bwAB`, the first arm that reaches the target is the scheme proposed for the lock; if none reaches it, no scheme is proposed. Nothing is locked in this round. **If both arms reach the target, `bwA` is proposed. The PI may lock `bwAB` instead only with a stated reason taken from the items of §2.2 of the G1 reply. Nothing else may change the choice.**

## 7. Reported, deciding nothing

- Per arm and q: the pass counts with exact intervals; the worst `Gmax_full` and its run; the medians of `eta_3`, `rmse_pos_T_over_eT0`, `tail_mean_T_over_eT0`, `peak_T_signed`, `Gmax_full` and EXP_root.
- Against `bw`, per seed: the paired differences of `Gmax_full` and `eta_3` with paired bootstrap intervals (10,000 resamples, seed 20261021), and the pass / fail cross-table.
- S3 of every run, the stage-2 stretches, every trigger that fired with the run and the outcome, the compute cost per arm (updates, policy rows, minibatch steps, development checks, **measured wall time**, verifier seconds) and its ratio to `bw`.
- **First-attempt outcome per arm (G1 reply 2.1).** The pass count when every fired run is counted as a failure of its first attempt. For `bwA` this must equal the pass count of `bw`, as C-PAIR implies; **a mismatch is a stop.** Reading of the sentence: a `bw` run is counted as "fired" when the trigger rule, applied to its own stage-3 check at local update 1600, fires; the invariant is then `bwA` = `bw` seed by seed. The raw pass count of `bw`, and the number of `bw` runs that fire and nevertheless pass the gates (which the convention counts as first-attempt failures in both arms), are reported beside it.
- **The causal reading of B (G1 reply 2.2).** `bwAB` against `bw` and against `bwA`, paired by seed, on first attempts (for a fired run, `attempt1/`): S3 per run and the number of runs with S3 > 0.2, with an exact McNemar test on the paired regime indicator; at the first attempt's export `u01600` the share of out-of-band nodes at the actor's mean clamp and the minimum alpha on the D_3 grid; the development `Delta` at local 1600; for the seeds where the trigger fired in neither arm, the paired differences of `eta_3` and of `peak_T_signed` with paired bootstrap intervals (10,000 resamples, seed 20261021); the stage-2 stretches of each arm; the fire rate of each arm.
- **The reading that the proposal gives, stated before any run:** if S3, the stretches and the in-band bias fall in `bwAB`, the clamp regime was more than a marker; if they do not, it is a marker. For the report this is made operational with three indicators on the tables above, none of which decides anything: (i) the number of first-attempt runs with S3 > 0.2 is smaller in `bwAB` than in `bw` and in `bwA`, with the exact McNemar p < 0.05 for the pooled pairs; (ii) the number of runs with a stage-2 stretch is smaller in `bwAB` than in `bw` and in `bwA`; (iii) on the seeds where the trigger fired in neither arm, the paired bootstrap interval of the difference of `eta_3` (`bwAB` - `bw`, pooled over both q) lies entirely below 0. All three: "more than a marker". None: "a marker". Anything else: "mixed", reported as such with no causal claim.
- **The diagnostic runs (G1 reply 2.3).** Whether attempt 1 equals T3-R2's original stage 3 (the exports, the stage-3 rows and the checks); whether A fired; for attempt 2: the formation measure F(u) at the 64 exports, the formation export (threshold 26.78, half the median final F of the passing q = 60 runs of the 80 recorded runs, `results/t3_r3/diag/dg2_runs.csv`), the gap-sensitivity and the final gates.
- **Blind recomputation.** `tools/t3/t3r3_blind.py` (it imports no analysis tool) recomputes the counts, the intervals, the target outcome per arm and the selection from the evaluation table; it must agree with `headline.json` of the analysis.

## 8. C-LEGACY (at the final code commit)

{{table:prereg_legacy}}

## 9. The later confirmation (D6; executed only in a later round, after the PI's decision at G2)

- **What is confirmed:** the one scheme proposed under section 6, locked in that round as protocol T3-1.1.
- **Seeds:** 80501-80540, the same 40 seeds at q = 50 and at q = 60, for 80 runs (inventory below: 0 collisions, 0 hits).
- **The rule:** the confirmation passes iff at most 6 of the 80 runs fail and at most 4 of the 40 runs fail at each q.
- **Failure counting,** as in T3-R2's D4: infrastructure failures (negative return code, out-of-memory kill, ENOSPC) are recorded in an `infra_events.json` and the run is re-run once, a second one stops the confirmation without a verdict; any other nonzero exit and any global-RNG violation is a failed run; an evaluation error or an invalid evaluation is a failed run unless a re-run of the locked evaluator succeeds; a re-initialisation or retraining inside a run is part of the run, never a re-run.

## 10. Operating characteristics (computed by `proposal_tables.py`)

{{table:oc}}

## 11. Predictions (D7, the PI's, verbatim; expectations, not tests)

- **DP1.** 60518 is a formation failure, unrelated to the clamp.
  - At q = 60, runs pass through a level of development `Delta` of about 0.12-0.15 near local update 100. 60518 never left it.
  - Its stage-3 network is saturated: its tanh units are saturated over D_3, its gap-sensitivity is about 0, and its in-band surrogate gradient stays small, compared with runs that formed.
- **DP2.** The clamp regime is a mechanism, not only a correlate.
  - In the clamp-regime runs, the out-of-band samples and the draws below the clamp have a projection share of the surrogate gradient above their sample share.
  - The tie-point effort ê_3(0) falls, or stops rising, after the regime's onset.
- **DP3.** The following are one class, (K):
  - the q = 50 tie-point deficits (60501, 60506, 60513, 50504);
  - 60515's excess at d = -36;
  - 60516's shape failure;
  - the two T3-R1 stage-2 stretch failures. 50517's stretch was present at the phase entry, while 50514's formed in updates 2475-2750, but both runs are in the regime.

  T3-R1's 50517 at q = 50 is (V).
- **DP4.** No prediction of the development outcome.

(P1 found DP1a-b supported, DP1c-d partly or not, DP2 not supported, DP3 supported as a classification: `01_diagnosis.md` section 10.) **The predictions of this proposal that the wave can contradict:** A fires on about 9 % of the runs (7 of 80 in the recorded runs); `bwA` and `bw` are bit-identical wherever A does not fire; the diagnostic run of seed 60518 fires A in both arms; B is not expected to change the fire rate if the clamp regime is only a marker.

## 12. Seed inventories (P0, `00_housekeeping.md` section 3)

{{table:prereg_seeds}}

## 13. Known full-suite failures (G1 reply 3)

`tests/test_registry_canonicalization.py::test_registry_canonicalization` (the known failure) and `tests/test_ms_r4_launcher.py::test_the_existing_launcher_is_unchanged`, which has failed since the T3-1.0 lock changed `protocols/`. Both are recorded as known.
