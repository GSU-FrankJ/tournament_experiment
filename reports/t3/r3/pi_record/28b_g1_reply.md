Reply at gate G1 of round T3-R3
proceed P2
The diagnosis (`reports/t3/r3/01_diagnosis.md`) and the proposal (`reports/t3/r3/02_proposal.md`) at `a7a9e0c1` are accepted. Before anything else in P2, save this reply verbatim as `reports/t3/r3/pi_record/28b_g1_reply.md`.
1. Arms and schemes (approved as proposed)

* Arms: `bw` (the control, with the locked T3-1.0 settings; not eligible), `bwA` and `bwAB`.
* `bwA` is candidate A, exactly as `02_proposal.md` §2.1 specifies:
   * the config key `stage3_retry = {"delta_gt": 0.005, "max_retries": 1}`;
   * the trigger is read at the local-1600 development check of the stage-3 phase. It fires iff `Delta` > 0.005, or `Delta`, `s` or `C` is not finite, or `s` ≤ 0;
   * one in-place re-initialisation, with the derived seeds and streams of §2.1;
   * attempt 2 is frozen, whatever its own check says;
   * the artefacts of attempt 1 are moved to `attempt1/`, never overwritten or deleted, and `retry.json` is written;
   * a run whose trigger does not fire writes exactly the files, columns and rows of `bw`.
* `bwAB` is `bwA` plus candidate B, exactly as §2.2 specifies:
   * the key `terminal_clamp_likelihood = "censored"`, in the stage-3 phase only, for both attempts;
   * it reuses `utils/beta_tail.py`, `run/v2_rollout.collect_batch_v2(..., clamp_likelihood="censored")` and `CurriculumPPOv2.update(..., clamp_side=...)`;
   * if B needs anything beyond wiring these into `run/run_ms_stagewise.py` behind the key, stop and report;
   * add a test at T = 3, in the stage-3 phase: the stored log-prob of every flagged row equals the censored log-mass of `utils/beta_tail.py`, and every unflagged row equals the density path.
* Selection.
   * Only `bwA` and `bwAB` are eligible. In the order `bwA`, then `bwAB`, the first arm that reaches the target (at least 38 of 40, and at least 18 of 20 at each q) is proposed for the lock.
   * Pre-register one more sentence: "If both arms reach the target, `bwA` is proposed. The PI may lock `bwAB` instead only with a stated reason taken from the items of §2.2 of this reply. Nothing else may change the choice."

2. Additions to the pre-registration (reported, deciding nothing)

1. First-attempt outcome per arm. The pass count when every fired run is counted as a failure of its first attempt. For `bwA` this must equal the pass count of `bw`, as C-PAIR implies; a mismatch is a stop.
2. The causal reading of B. Compare `bwAB` with `bw` and with `bwA`, paired by seed, on first attempts. For a fired run, use `attempt1/`. Report:
   * S3 per run, and the number of runs with S3 > 0.2, with an exact McNemar test on the paired regime indicator;
   * at the first attempt's export `u01600`: the share of out-of-band nodes at the actor's mean clamp, and the minimum α on the D_3 grid;
   * the development `Delta` at local 1600;
   * for the seeds where the trigger fired in neither arm: the paired differences of `eta_3` and of `peak_T_signed`, with paired bootstrap intervals (10,000 resamples, seed 20261021);
   * the stage-2 stretches of each arm;
   * the fire rate of each arm.
State in the pre-registration, before any run, the reading that the proposal gives. If S3, the stretches and the in-band bias fall in `bwAB`, the clamp regime was more than a marker; if they do not, it is a marker.
3. The diagnostic runs at (q = 60, seed 60518). Report:
   * that attempt 1 equals T3-R2's original stage 3: the exports, the stage-3 rows and the checks;
   * whether A fired;
   * for attempt 2: the formation measure F(u), the formation export, the gap-sensitivity and the final gates.
4. Cost. Besides updates and policy rows, the measured wall time per arm and its ratio to `bw`.
5. Paper wording for A, verbatim in the pre-registration: "a verifier-gated restart of the terminal stage: when the development-tier one-step deviation gain at the end of the stage-3 phase exceeds the gate's limit 0.005, stage 3 is re-initialised once and retrained, and the second attempt is kept; its cost is part of the method's cost."

3. Everything else as in the prompt

* Known full-suite failures. They now include `tests/test_ms_r4_launcher.py::test_the_existing_launcher_is_unchanged`, which has failed since the T3-1.0 lock changed `protocols/`. Record it as known.
* Unchanged from `reports/t3/r3/pi_record/28_t3_r3_prompt.md`:
   * the order before the launch;
   * C-LEGACY at the final code commit;
   * the push points;
   * the post-launch checks, with the candidates' budgets of `02_proposal.md` §3;
   * the evaluation and the blind recomputation;
   * the reports;
   * the STOP at G2.
