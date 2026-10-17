# T3-R1: summary

Date: 2026-10-17. Round T3-R1 (branch `t3-r1`, code commit `5535d610`, launch `a848114e`): backward induction (`bw`) against joint whole-game training (`jt`) and joint training with exploring starts (`jt_es`) at T = 3, and the same joint comparison at T = 2 (`jt2` against the recorded v2.0 runs), on 20 seeds per q for q in {50, 60}, at an equal budget of policy rows. The design, the criteria, the secondary statistics and the predictions were pre-registered (`02_preregistration.md`, pushed before the launch); the launch (160 runs, one launch of two waves) had no failed run, and the post-launch checks passed in 160 of 160 runs.

## Outcome

| hypothesis | pair (mean paired difference of `Gmax_full/dW`) | result |
|---|---|---|
| H1 | `bw` - `jt`, T = 3 | **supported** at both q: -0.18054 [-0.23237, -0.13464] (q = 50), -0.10019 [-0.12522, -0.07612] (q = 60) |
| H2 | `bw` - `jt_es`, T = 3 | **supported** at both q: -0.01115 [-0.01306, -0.00937], -0.00896 [-0.01317, -0.00252] |
| H3 | `jt_es` - `jt`, T = 3 | **supported** at both q: -0.16939 [-0.22104, -0.12330], -0.09122 [-0.11459, -0.06952] |
| H1-T2 | `v20` - `jt2`, T = 2 | **supported** at both q: -0.00923 [-0.01054, -0.00808], -0.01248 [-0.01442, -0.01088] |

(A hypothesis is supported iff the upper end of the 95 % bootstrap interval is below 0 at both q; 20 pairs per q.)

**H1 = H2 + H3:** over the same 20 pairs the mean differences add exactly; the exploring starts close 0.938 (q = 50) and 0.911 (q = 60) of the mean `bw` - `jt` difference (H3/H1) and 0.062 and 0.089 remain (H2/H1). With the D5 caveats, capacity and start density, H2 is not a test of the backward order alone.

**`bw` at T = 3:** passes G-A, G-F and G-N in 18 of 20 runs at q = 50 and 18 of 20 at q = 60 (Clopper-Pearson [0.683, 0.988]); the reading rule of D7 (at least 18 of 20 at both q) is met with no margin, so v2.0's design is a candidate for a T = 3 lock. The failures: two at q = 50 through `eta_3/dW` of G-A (0.00623, 0.00568 against 0.005), two at q = 60 through G-F at stage 2, in band (0.05628 and 0.01089 against 0.01). At the gap of the maximum the two q = 60 runs play a mean stage-2 effort of 0.52 and 0.16 against best responses of 34.44 and 14.71 (a stretch of near-zero effort inside the band; the other 18 runs of the q have `G_2(d)/dW` of at most 0.00062 and 0.00044 there); the two q = 50 runs have a smooth stage-3 policy that is 8.0 below (d = -4) and 16.3 above (d = -18) the best response (`04_t3_results.md` section 4.1; why is not tested). With 18 of 20 passes at each q (Clopper-Pearson [0.683, 0.988]), a fresh 20-seed block has at least 18 passes with probability 0.677, 0.925 and 0.405 at one q and 0.458, 0.855 and 0.164 at both q for a per-run pass probability of 0.90, 0.95 and 0.85 (arithmetic only); the confirmation rule of a T = 3 lock is for the PI to decide with the coworker, and nothing in this round decides it. `jt` passes G-A and G-F in no run at either q, `jt_es` G-A in none and G-F in 4 of 20 at each q.

**Where the gains are:** `jt`'s largest gain is out of band at stage 2 in 20 of 20 runs at both q (median G_out,2 0.158 and 0.093 against G_in,2 0.022 and 0.027), `jt2`'s in 34 of 40 runs; `jt_es`'s is in band in 33 of 40 runs; `bw`'s is in band in all 40. The root-path gain EXP_root orders the arms like `Gmax_full` but with 11 to 20 times smaller differences between `bw` and `jt` (prediction P2). The joint arm `jt` visits 24 of 40 stage-2 and 44 of 80 stage-3 bins at q = 50; the ordering of the arms' out-of-band gains follows their rows per out-of-band bin (observed co-variation, not a test of cause).

**Predictions** (D8): P1 holds for H1 and H1-T2 and for `jt` and `jt2`'s out-of-band gains, not for `jt_es`; P2 and P3 hold; P4 holds in its first sentence (18 against 19 and 20), and in its second only at q = 50 (at q = 60 both failures are G-F).

**Scope:** the arms as implemented, the pre-registered budget (policy rows), one game, 20 seeds per q. Not settled: capacity (three networks against one), a coverage-matched order, other budgets, the cause of the four `bw` failures. This round decides nothing; no lock, no confirmation, no tuning, no second wave.

## Where to read

`04_t3_results.md` (T = 3: checks, arms, per-seed tables, `bw` with the four failures in detail (section 4.1), the comparison with the H1 = H2 + H3 decomposition, where the gains are, EXP_root, best of run, curves, coverage, cost, predictions, caveats), `05_t2_comparison.md` (T = 2), `06_decision_inputs.md` (the T = 3 reading, what each hypothesis supports, what is open, options), `02_preregistration.md`, `03_checks.md`, `pi_record/01_factcheck.md` (the fact-check ledger), `pi_record/26_g1_reply.md` (the PI's reply), `handoff.md` (for the coworker, Chinese). Data: `results/t3_r1/{eval,analysis,blind}/`; figures: `results/t3_r1/analysis/figures/`.
