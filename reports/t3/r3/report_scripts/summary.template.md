# T3-R3: summary (development wave, gate G2)

Date: {{date}}. Branch `t3-r3`. Generated from `reports/t3/r3/report_scripts/summary.template.md`; the full report is `05_development.md`, the diagnosis `01_diagnosis.md`, the proposal `02_proposal.md`, the pre-registration `04_preregistration.md`. **STOP at G2: nothing is locked, no confirmation seed was trained, and nothing was tuned after the pre-registration.**

## The result

One wave of 122 runs (seeds 70501-70520 at q = 50 and q = 60 in three arms, plus two diagnostic runs at q = 60, seed 60518) ran to completion: all exit codes 0, no infrastructure event.

{{table:res_target}}

- **`bwA` is proposed** (pre-registered order: both candidates reach the target, `bwA` comes first). `bwA` passes 39 of 40 runs (exact 95 % interval [0.868, 0.999]), `bwAB` 38 of 40 ([0.831, 0.994]), the control `bw` 37 of 40 ([0.796, 0.984]).
- The gain of `bwA` over `bw` is two seeds at q = 50 (70508, 70516), where the trigger fired and the second attempt passed; on the other 38 seeds the two arms are bit-identical (exact McNemar p = 0.5). The first-attempt outcome of `bwA` equals `bw`'s (37), as C-PAIR requires.
- **B** (`bwAB`): the pre-registered reading of the clamp-regime indicators is 'mixed' (S3 > 0.2 in 13 first attempts against 12; stage-2 stretches in 1 run against 3; the interval of the paired `eta_3` difference on the 36 seeds without a trigger is [-0.00008, 0.00064]); no causal claim.
- **Failures:** 6 failing runs on 4 seeds; 5 are class (K) (clamp regime), 1 is class (V); none is (F) (the formation measure is above threshold in all 120 development runs), and I assigned none to (E) or (U). `bwAB` reaches the target with no margin (38 of 40; exactly 18 of 20 at q = 50). The q = 60 stretch failure (seed 70511, fails in `bw` and `bwA`) is outside what the trigger can see.
- **Diagnostic runs (seed 60518):** attempt 1 equals T3-R2's original stage 3 in both arms; A fired (`Delta` 0.1447); both second attempts form (F(u01600) 54.1 and 50.2 against the threshold 26.78) and pass.

## The cost

Mean updates per run 3,800 (`bw`), 3,880 (`bwA`, 1.021 times), 3,920 (`bwAB`, 1.032 times); a run that fires trains stage 3 twice (5,400 updates, a factor 1.421). Measured wall time (one concurrent wave, the control last): `bwA` 1.081 and `bwAB` 1.201 times `bw`'s total; the 38 bit-identical computations of `bwA` took 1.065 times as long as `bw`'s, so most of the wall-time gap of `bwA` appears to be the environment (run order and load; inferred, not measured), not the scheme.

## Deviations (details in `05_development.md` section 1)

- **D-1.** 82 of 122 manifests record a dirty tree: I wrote an untracked report script into the worktree about 73 s after the launch. Checked on five clean-tree re-runs (two with a fired trigger), all bit-identical to their originals, plus the boundary argument (the flag follows the file's creation exactly); no `bw` run was re-run. The 122 runs remain the counted records.
- **D-2.** The first post-launch check used a relative `--root` (`config_equals_plan` 0 of 122); the absolute-root check is the record, the first file is kept. The tools now refuse a relative `--root`.
- **D-3.** The post-launch script ran the evaluation and analysis after the checks had exited 1; the replacement stops at the first failure.
- Full suite: {{suite_result}} (two known failures).

## For the PI at G2

1. Whether the proposed scheme (`bwA`, the paper wording of `04_preregistration.md` section 3) goes to the lock and the confirmation of D6 in the next round (seeds 80501-80540, at most 6 failures of 80 and at most 4 of 40 per q). The PI may lock `bwAB` instead only with a stated reason from section 2.2 of the G1 reply; the reading of those items is in `05_development.md` section 8.
2. The limits to keep in view: 40 runs per arm; the gain is two seeds; A repairs only what the stage-3 check shows; the cost of a fire is a second stage-3 phase.
