# MS-R4: decision inputs

Date: 2026-10-14. Branch `ms-r4`. This file puts the outcomes of E1 and E2 next to the coworker's targets (Appendix A of `pi_record/23_ms_r4_prompt.md`, with the English rendering `pi_record/Multistage101126(1).md`) and says what the plan of Appendix A implies next for each outcome. **Nothing in this file decides.** The next round is the PI's decision with the coworker; the confirmation rule A2 of the PI's reply at gate G1 (`pi_record/24_g1_reply.md`, `02_preregistration.md` Addendum 2) is stated next to each outcome, and it is not used in this round. Evidence: `04_e1_actor.md`, `04_e2_feedback.md`, `results/ms_r4/analysis/`. Tables are printed by `report_scripts/pilot_tables.py`.

## 1. The outcomes against the coworker's targets

| | the coworker's target (Appendix A, as operationalised in D6) | outcome |
|---|---|---|
| **E1** (item 1) | at both q: mean abs peak error <= 0.04; the paired improvement over `t1_st_s16` holds in the 95 % interval; no G-A, G-F, G-N or G-S failure in the development run; "If the goal is not met, stop this line" | **not met.** q = 60: all three parts hold (0.0259; -0.0122 [-0.0219, -0.0016]; 0 of 20 failing). q = 50: (a) holds (0.0329), (b) fails (-0.0025 [-0.0175, +0.0196]), (c) fails (4 of 20 runs fail a gate; `t1_st_s16` 0 of 20, `relu_st_s16` 3 of 20) |
| **E2** (item 2) | against arm (1), at both q: the area under the stage-2 eta_2/DW curve at fixed checkpoints falls by at least 20 % with the paired 95 % interval supporting it, and the final gates and the stage-1 deviation gain do not get worse; if (2) meets the target (3) is not needed; if neither does, the DP-BR is kept as a verifier only | **neither arm (2) `fb_r` nor arm (3) `fb_rg` meets it.** rho = +0.020 [-0.014, +0.051] and -0.020 [-0.048, +0.008] (`fb_r`), +0.013 [-0.024, +0.049] and -0.009 [-0.038, +0.023] (`fb_rg`) at q = 50 and 60; every interval lies below 0.06. `fb_r` also violates (E2-b) at q = 50 (one run at eta_2/DW = 0.00516 against 0.005). The DP-BR stays a verifier; no training improvement is claimed |

Facts that bear on how to read the table (all in the two reports):

- **Part (a) of E1 does not discriminate.** The control `t1_st_s16` meets it by itself: 0.0354 at q = 50 and 0.0381 at q = 60 (`04_e1_actor.md` section 3). The discriminating parts are (b) and (c).
- **E1 at q = 60 is a small improvement supported by the 95 % interval (upper end -0.0016)**; at q = 50 the typical `lrsm` run is closer to the closed form than a `t1` run (15 of 20 pairs lower, median pair -0.0124), but four runs fail G-A and one of them has a peak error of 0.208. `relu_st_s16` has the same pattern with three failing runs; `lrsm_st_s16` is not detectably better than `relu_st_s16` (+0.0008 [-0.0212, +0.0263] and +0.0086 [-0.0000, +0.0174]).
- **The E2 area is not lowered by the within-stratum focus at this strength.** The weights moved by a median total variation of 0.05-0.06 from the alpha-0 weights (`fb_rx`: 0.122-0.125), and no arm-vs-`fb_a0` interval of the area contains 0.20 (`04_e2_feedback.md` sections 4 and 6).
- **The cross-stratum sampler `fb_rx` is descriptive and mixed:** a smaller area at q = 50 (rho +0.054), a larger one at q = 60 (rho -0.126), a smaller late-window area at both q. It also moves 0.12 of the sampling mass from the near-tie to the middle stratum, and no arm separates that from the feedback.
- **`fb_rg` has end-of-stage differences at q = 50** (eta_2/DW at update 1600 -0.00074 [-0.00124, -0.00029], |peak error| -0.0069 [-0.0127, -0.0013]) and none at q = 60; 15 of 60 descriptive secondary intervals exclude 0 against about 3 expected by chance, so these are leads, not findings.
- **E2-c as pre-registered measures stage-2 eta_2/DW, not the stage-1 deviation gain** (Gmax_full/DW equals eta_2/DW in all 280 runs); the stage-1 gain Delta_1/DW is at most 5.4e-4 in any run, about half of the margin of 0.001 (`04_e2_feedback.md` section 7). No verdict of this round depends on it.

For scale, side by side and with the caveat that the seeds and budgets differ (the arms of this round are on the development seeds 10511-10530; the locked v2.0 numbers are on the confirmation block 30501-30520 as reported by the status pack, `docs/STATE.md`, section "T=2 status pack"): v2.0 had a mean |peak error| of 0.0630 at q = 50 and 0.0678 at q = 60 with 5 of 20 and 4 of 20 runs within 0.05. On the new seeds the E2 reference arm `fb_a0` (v2.0's budgets and actor, stratified starts) has 0.0467 and 0.0490, and the E1 arms (2800 updates with the s = 16 landing) have the means of block `e1_overview`:

| q | arm | runs | mean abs peak error | median | mean gap (effort units) | median gap | mean RMSE_pos / e2*(0) | median eta_2/DW | runs failing a gate |
|---|---|---|---|---|---|---|---|---|---|
| 50 | `t1_st_s16` | 20 | 0.0354 | 0.0359 | 2.48 | 2.51 | 0.0155 | 0.000536 | 0 |
| 50 | `relu_st_s16` | 20 | 0.0321 | 0.0239 | 2.23 | 1.67 | 0.0346 | 0.000729 | 3 |
| 50 | `lrsm_st_s16` | 20 | 0.0329 | 0.0226 | 2.15 | 1.57 | 0.0487 | 0.000891 | 4 |
| 60 | `t1_st_s16` | 20 | 0.0381 | 0.0371 | 2.22 | 2.16 | 0.016 | 0.000374 | 0 |
| 60 | `relu_st_s16` | 20 | 0.0173 | 0.0142 | 0.90 | 0.83 | 0.0131 | 0.000306 | 0 |
| 60 | `lrsm_st_s16` | 20 | 0.0259 | 0.0260 | 1.47 | 1.52 | 0.0139 | 0.000443 | 0 |

| q | arm | mean A x 1e3 | median A x 1e3 | median eta_2/DW at u1600 | mean abs peak error | median gap | mean RMSE_pos / e2*(0) | mean tail mean / e2*(0) | mean stage-1 abs rel. error | runs passing all four gates |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | `fb_a0` | 6.862 | 6.733 | 0.00183 | 0.0467 | 3.29 | 0.0276 | 0.00947 | 0.0171 | 18 |
| 50 | `fb_r` | 6.724 | 6.605 | 0.00229 | 0.0458 | 3.26 | 0.0289 | 0.00991 | 0.0178 | 19 |
| 50 | `fb_rg` | 6.769 | 6.909 | 0.00135 | 0.0397 | 2.61 | 0.0228 | 0.00981 | 0.018 | 20 |
| 50 | `fb_rx` | 6.489 | 6.460 | 0.0013 | 0.0485 | 3.57 | 0.0227 | 0.00912 | 0.018 | 20 |
| 60 | `fb_a0` | 4.149 | 4.085 | 0.00069 | 0.0490 | 2.83 | 0.0221 | 0.0106 | 0.015 | 20 |
| 60 | `fb_r` | 4.232 | 4.136 | 0.00061 | 0.0453 | 2.55 | 0.0236 | 0.0111 | 0.0156 | 20 |
| 60 | `fb_rg` | 4.187 | 4.058 | 0.000664 | 0.0444 | 2.47 | 0.0224 | 0.0108 | 0.0135 | 20 |
| 60 | `fb_rx` | 4.669 | 4.725 | 0.000505 | 0.0452 | 2.67 | 0.0213 | 0.011 | 0.0118 | 20 |

## 2. What the plan of Appendix A implies next, outcome by outcome

The coworker's note (Appendix A) orders the steps: the two tries first; "after they are completed, finalize the T=2 protocol and start the T=3 experimental validation"; a development result that passes is followed by locking one complete scheme and a fresh-seed confirmation on 40501-40520; "even if the two items are each effective, one cannot directly assume that they are effective when used together; combined use must first be separately validated on development seeds"; and "If confirmation fails, retain v2.0, and report the two results as ablation and limitations" (quotations are from the English rendering, `pi_record/Multistage101126(1).md`). Read literally against the outcomes:

- **E1 (not met).** The coworker's rule is that the line stops. No `lrsm` scheme has passed a development criterion, so there is nothing to lock or confirm under the plan. *A2, if the coworker nevertheless chose to lock `lrsm_st_s16` as the E1-type scheme:* on fresh seeds 40501-40520 x q in {50, 60}, with `t1_st_s16` as the comparator in the same wave, it would pass only if at both q (i) at least 18 of 20 runs pass all four gates, (ii) the mean abs peak error is <= 0.04, and (iii) the 95 % interval of the mean paired difference of abs peak error, scheme - `t1_st_s16`, lies entirely below 0. For orientation, the development run had 16 of 20 passing runs at q = 50 and 20 of 20 at q = 60, and its q = 50 interval for (iii) contained 0 (post hoc, block `a2_rule_i`; the probabilities are for rule (i) alone and under assumed true rates, not an estimate of the true rate):

| arm | q | runs passing all four gates (development seeds) | Clopper-Pearson 95 % |
|---|---|---|---|
| `t1_st_s16` | 50 | 20 / 20 | [0.832, 1.000] |
| `t1_st_s16` | 60 | 20 / 20 | [0.832, 1.000] |
| `relu_st_s16` | 50 | 17 / 20 | [0.621, 0.968] |
| `relu_st_s16` | 60 | 20 / 20 | [0.832, 1.000] |
| `lrsm_st_s16` | 50 | 16 / 20 | [0.563, 0.943] |
| `lrsm_st_s16` | 60 | 20 / 20 | [0.832, 1.000] |

| assumed true per-run pass rate at one q | probability of at least 18 of 20 passing runs at that q |
|---|---|
| 0.80 | 0.206 |
| 0.85 | 0.405 |
| 0.90 | 0.677 |
| 0.95 | 0.925 |
| 0.99 | 0.999 |

- **E2 (neither arm met).** The coworker's rule is that the DP-BR stays a verifier and no training improvement is claimed; there is no E2-type scheme. *A2:* its E2 branch (the arm on which the E2 statement rests, comparator `fb_a0`; E2-a, E2-b and E2-c on the fresh seeds and at least 18 of 20 runs passing all four gates, at both q) has no arm to apply to.
- **The combination of E1 and E2.** Appendix A speaks of a combination only for the case in which each try is effective ("even if the two items are each effective, one cannot directly assume that they are effective when used together; combined use must first be separately validated on development seeds"); it is silent on a combination of tries that missed their targets, and neither try was effective. One structural fact for anyone who wants it anyway: E1's three arms already use the cross-stratum DP-BR feedback of `fb_rx` (`fixed_global_sampler`, alpha 0.5; adjustment 2), so E1's configuration is a combination of an actor change with that feedback, and E2 did not show that the feedback lowers the area at the within-stratum strength tested. A combined scheme would need its own development run and its own confirmation rule fixed before that run (A2).
- **Keeping v2.0.** With no scheme locked, v2.0 stays the T = 2 solver by default, and E1 and E2 are available as ablations and limitations: E1 (the combined change `lrsm` did not meet the target; the `relu` failures of MS-R3 recur at 3 of 20 q = 50 runs and are not removed by `lrsm`, 4 of 20); E2 (within-stratum DP-BR sampling feedback at alpha 0.5 did not lower the stage-2 learning-curve area by 20 % at v2.0's budget, and the verifier's checks cost 0.4 % of the stage-2 wall time).

## 3. The options and what the data say about each

The options of the prompt, as inputs and not as recommendations:

| option | what it requires under the plan | what this round's data say |
|---|---|---|
| **A. Lock one complete scheme and confirm it on 40501-40520** | a development result that passes | none passed. Locking `lrsm_st_s16` or `fb_rg` would be locking a scheme that missed its development criterion; A2 above is the confirmation rule that would bind it |
| **B. Run the combination on development seeds first** | Appendix A names it for the case that each try is effective on its own | neither was; E1's arms already contain the cross-stratum feedback; the within-stratum arms of E2 did not lower the area |
| **C. Keep v2.0 and report both tries as ablations and limitations** | nothing further | consistent with both outcomes; the numbers for the ablation and limitation text are in `summary.md` and the two reports |

## 4. What is not settled

- **Why `relu` and `lrsm` fail at q = 50.** In all seven failing runs RMSE_pos at update 2000 is 2.3-8.7 times its value at update 1200, against a median ratio of 1.0 in the 113 passing runs (post hoc, `04_e1_actor.md` section 5.1); in two of them RMSE_pos is above 0.1 already by update 1600. Which element of the training causes it was not tested (`04_e1_actor.md` section 5.1). The combined change of E1 does not separate the activation from the mean map.
- **Whether a stronger or different within-stratum focus helps.** The pilot tested alpha 0.5 inside strata with two focus maps; the lever was about half that of `fb_rx` in total variation. No other strength was run, and D6 allowed none.
- **Whether the cross-stratum sampler's late-window gain is feedback or the share change.** No arm has the cross-stratum shares without the feedback.
- **Power.** Twenty seeds per q: a zero-failure requirement over 20 runs has a 12 % chance of passing for a candidate with a 10 % failure rate (adjustment 6), and the E2 area intervals are wide (widths 0.06-0.08).
- **The E2-c definition.** It measures the stage-2 deviation gain (section 1); a later round that wants a stage-1 quantity should say which one (the one-step deviation gain Delta_1/DW or the stage-1 best-response gap) in its definition.
- **The `.docx` of the coworker's note** was never seen; the English rendering agrees with Appendix A (`02_preregistration.md` Addendum 1).

## 5. The confirmation rule A2, verbatim from the PI's reply at gate G1

(Printed from `pi_record/24_g1_reply.md` by block `a2_verbatim`.)

> A2. The confirmation rule of a possible lock (pre-registered now, before any pilot result exists)
> This rule is not used in this round. It binds a later lock round only if the coworker decides at G2 to lock a scheme. `05_decision_inputs.md` states it next to each outcome.
> Design. Fresh seeds 40501-40520 × q ∈ {50, 60}, 20 per q. The locked scheme and its comparator are run in the same wave on the same seeds, paired by (q, seed). The locked v2.0 entry point is run on the same seeds as a descriptive reference (it is not part of the pass rule). Bootstrap as in D6, with the seed of the experiment the scheme comes from (20261013 for E1, 20261014 for E2).
> E1-type scheme (`lrsm_st_s16`, as locked; comparator `t1_st_s16`). It passes iff all three hold at both q:
>
> * (i) at least 18 of 20 runs pass all four gates (the rule of v2.0's confirmation; exact Clopper-Pearson intervals reported);
> * (ii) the mean |peak error| over the 20 runs is <= 0.04;
> * (iii) the 95 % interval of the mean paired difference of |peak error|, scheme - `t1_st_s16`, lies entirely below 0.
>
> E2-type scheme (the arm on which the E2 statement rests, as locked; comparator `fb_a0`). It passes iff E2-a, E2-b and E2-c hold on the fresh seeds and at least 18 of 20 runs of the scheme pass all four gates, at both q.
> Combined scheme. It needs its own development run first (Appendix A). Its confirmation rule is fixed together with that run, before the run.
> If the confirmation fails, v2.0 stays the T = 2 solver, and E1 and E2 are reported as ablations and limitations (Appendix A).
