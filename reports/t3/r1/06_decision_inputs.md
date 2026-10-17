# T3-R1: inputs to the next decision

Date: 2026-10-17. **Nothing in this file decides.** It gives the T = 3 reading of D7 as the pre-registration states it, what each hypothesis supports and the exact scope of the statement, what is not settled, and options for the next round that no data of this round test. The decision belongs to the PI with the coworker. Numbers are those of `04_t3_results.md` and `05_t2_comparison.md` (tables generated from `results/t3_r1/`).

## 1. The T = 3 reading of D7

The rule (D7, `02_preregistration.md` section 7; this round decides nothing by it): *if `bw` passes G-A and G-F and G-N in at least 18 of 20 runs at both q, v2.0's design at T = 3 is a candidate for a T = 3 lock (the next round would lock it and confirm it on a fresh T = 3 seed block); otherwise the reports give the per-gate, per-stage diagnosis and no change is tried in this round.*

| arm | q | runs passing all gates (of 20) | 95 % Clopper-Pearson | at least 18 |
|---|---|---|---|---|
| `bw` T = 3 | 50 | 18 | [0.683, 0.988] | yes |
| `bw` T = 3 | 60 | 18 | [0.683, 0.988] | yes |
| `v20` T = 2 (recorded) | 50 | 19 | [0.751, 0.999] | yes |
| `v20` T = 2 (recorded) | 60 | 20 | [0.832, 1.000] | yes |

**`bw` passes in 18 of 20 runs at q = 50 and 18 of 20 at q = 60: the rule is met, with no margin.** One more failing run at either q would have put the round on the other side of the rule. By the rule's wording, v2.0's design at T = 3 (the parameter file `params_bw_t3.json`: 1,600 / 1,600 / 600 updates, the schedule of D3) is a candidate for a T = 3 lock, and the next round would lock it and confirm it on a fresh T = 3 seed block. For reference, `v20` at T = 2 passed in 19 of 20 and 20 of 20 runs; the Clopper-Pearson intervals of the T = 3 and T = 2 pass counts overlap widely, so this round does not show that T = 3 is harder or easier than T = 2 for the design.

What 18 of 20 at both q means as a rule (arithmetic on a binomial model with one true per-run pass probability p at both q; not an estimate of p):

| true per-run pass probability p (same at both q) | P(at least 18 of 20) at one q | P(at least 18 of 20) at both q |
|---|---|---|
| 0.70 | 0.035 | 0.001 |
| 0.80 | 0.206 | 0.042 |
| 0.85 | 0.405 | 0.164 |
| 0.90 | 0.677 | 0.458 |
| 0.95 | 0.925 | 0.855 |
| 0.98 | 0.993 | 0.986 |

The rule separates a pass probability of 0.70 from one of 0.98 well, and separates 0.90 from 0.95 poorly: the chance of meeting it at both q is 0.458 at p = 0.90 and 0.855 at p = 0.95. A confirmation round with the same rule on a fresh block would have the same limits.

**The diagnosis of the four failures** (`04_t3_results.md` section 4; the data identify where the gains are, not why):

- q = 50, seeds 50504 and 50517: G-A fails through `eta_3/dW` alone (0.00623 and 0.00568 against 0.005; the shape parts pass; seed 50517's RMSE_pos is 0.0492 against its limit 0.05). Across the 40 `bw` runs the eta part of G-A fails in 2 of 20 runs at q = 50 and in 0 of 20 at q = 60; 3 runs at q = 50 are above 0.004.
- q = 60, seeds 50514 and 50517: G-F fails at stage 2, in band (d = -14 and d = -66): 0.05628 and 0.01089 against 0.01. One of the two is far above the limit (5.6 times), one just above it (9 %). The stage-3 quantities of both pass.
- Neither the pattern of seeds nor an explanation of the failures is tested here; the failures are one run each of four.

## 2. What each hypothesis supports, and the scope of the statement

All four are statements about the mean paired difference of `Gmax_full/dW` (final tier, last iterate, Beta means), 20 seeds per q, the two q values 50 and 60 of the game of D2 (k = 1/3500, w_H = 6, w_L = 2), at the pre-registered equal-policy-row budget, for the arms as implemented, with the percentile bootstrap interval of the pairing by (q, seed); supported means the upper end of the 95 % interval is below 0 at both q.

| hypothesis | statement supported | not supported by this round |
|---|---|---|
| H1 (`bw` - `jt`, T = 3) | backward induction has a smaller full-domain gain than the joint scheme from root play (means 0.00249 / 0.00476 against 0.18303 / 0.10495); the difference is mostly out of band, at stage 2 and stage 3 | which ingredient causes it (training order, capacity, coverage); other budgets; other games |
| H2 (`bw` - `jt_es`, T = 3) | `bw` still has the smaller full-domain gain once the joint scheme has exploring starts with the pre-registered mix (0.00249 / 0.00476 against 0.01365 / 0.01373); the smallest of the three T = 3 differences and the one whose interval comes closest to 0 (the 98.75 % interval at q = 60 reaches -0.00003) | a coverage-matched test of the backward order: `bw` and `jt_es` differ in coverage density, capacity and order together; `jt_es` was still improving at the end of its budget |
| H3 (`jt_es` - `jt`, T = 3) | the coverage that exploring starts give helps the joint scheme (0.01365 / 0.01373 against 0.18303 / 0.10495; 20 of 20 seeds at both q) | whether a different mix, or more updates, would close the gap to `bw`; the share of the effect that comes from the 15 % more minibatch steps of `jt_es` |
| H1-T2 (`v20` - `jt2`, T = 2) | at T = 2 the recorded v2.0 runs have a smaller full-domain gain than the joint scheme (0.00188 / 0.00113 against 0.01111 / 0.01361), and a smaller tail and RMSE error of the terminal tent; the tent's apex and the root effort are not distinguishable at 20 seeds, while the stage-1 error against g1 is smaller for `v20` (at q = 50 only in the mean, not in the median) and `jt2` passes G-S in 15 and 14 of 20 runs against 20 and 20 | anything new about v2.0 (the runs are the recorded confirmation runs, re-evaluated); which of coverage, continuation, capacity (two networks against one) and schedule makes the gap; the best a tuned joint scheme can do at T = 2 |

The secondary results (`04_t3_results.md` sections 5 to 7) agree in direction with the criterion and are descriptive: the root-path gain EXP_root orders the arms the same way but by much smaller factors (the pre-registered prediction P2), the joint arms' gains are out of band for `jt` and `jt2` (15 to 20 of 20 runs) and mostly in band for `jt_es` (16 to 17 of 20), and the ordering of the arms in their out-of-band gains follows their coverage of out-of-band bins.

## 3. What is not settled

- **Capacity.** `bw` has three networks (one per stage), the joint arms one; the round cannot separate the training order from the capacity.
- **A coverage-matched order.** `jt_es` is the pre-registered mix, not a design matched in coverage to `bw`; `bw` and `jt_es` also differ in capacity. H2 is not a test of the backward order alone.
- **Other budgets.** The arms are matched by policy rows; at matched updates, minibatch steps or wall time the comparison could differ. The development Gmax_full of `jt_es` and `jt2` fell between 75 % and 100 % of the updates, which is the learning-rate decay window (medians over the seeds at q = 50: 0.0149 to 0.0111 for `jt_es`, 0.0118 to 0.0101 for `jt2`; at q = 60: 0.0164 to 0.0134 and 0.0140 to 0.0130); the round does not show whether longer budgets would lower them further.
- **The reason for the four `bw` failures** (section 1) and whether a fresh block would show the same rate.
- **Generality.** One game (the parameters of D2), two values of q, 20 seeds each, the expected terminal reward, the last iterate; the July 2026 runs used another reward.

## 4. Options for the next round (none tested by this round's data)

These are what the results suggest; each would be a new pre-registered round, and none is recommended here.

1. **Lock and confirm.** Lock v2.0's design at T = 3 (the rule says it is a candidate) and run it on a fresh T = 3 seed block with a pre-registered rule. What it answers: whether the 18 of 20 holds on fresh seeds. Its limits: the discrimination of the rule (section 1) and the first block's pass count of 18 of 20, which has no margin (three of its four failures are marginal, one is far above its limit).
2. **Diagnose the four failures first.** Look at the stage-2 and stage-3 phases of the four runs (check traces, in-band gains near d = -14 and d = -66, the stage-3 eta at q = 50) before locking; answers whether the failures have a common, remediable origin.
3. **Separate the ingredients of H1 and H2.** A capacity-matched joint arm (three stage-specific networks trained jointly), a `bw` run with the `jt_es` start mix, or a `jt_es` arm with a longer budget would each isolate one of the ingredients that this round varies together.
4. **T = 4.** Not suggested by these results; out of scope of this round (D1), and listed in the original plan as a later extension (`docs/Experiments Plan_Multi-stage.md`).
