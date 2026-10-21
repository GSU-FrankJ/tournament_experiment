# T3-R3, P1: proposal for the development wave (gate G1)

Date: 2026-10-21. Branch `t3-r3`, based on `01_diagnosis.md` (same commit). **Generated** from `reports/t3/r3/report_scripts/02_proposal.template.md` by `build_report.py`; tables by `diagnosis_tables.py` and `proposal_tables.py`. Nothing here has been implemented, and no frozen file was touched in P1 (`git diff --stat c3289df9 HEAD -- agents envs utils run protocols tools` shows new files under `tools/t3/` only).

## 0. Summary for the decision at G1

- **Candidate A** (priority): at the end of the stage-3 phase, one re-initialisation and retraining of stage 3 if the development `Delta` (= Delta_3) of the last check exceeds 0.005; the retrained stage 3 is frozen. It is a **gate-level restart**: it gives a run that ends near or above the limit a second draw and does not change how stage 3 is learned. Expected cost +3.7 % updates (fire rate 7 of 80); worst case +42 %. In the 80 recorded runs the trigger fires on exactly the 7 stage-3 failures and on none of the 73 other runs. It cannot repair 3 of the 10 failures (two stage-2 stretches, one RMSE-only), and a model that assumes independent fresh draws gives it a probability of about 0.68 of meeting the development target (0.007 to 0.99 over the interval of every rate, §3).
- **Candidate B** (secondary, weak evidence): the censored likelihood of the draws clamped at 1e-6 in the stage-3 phase. It acts on the marker of the clamp regime (§1) and is the intervention that separates "the regime causes the in-band error" from "it only marks it". **The diagnosis does not support it as a mechanism** (DP2 failed, the clamped draws carry 0.3 % of the surrogate gradient) and the T = 2 precedent (R2b `A_censored`) found no effect at a ten times smaller clamp share. I propose it because the experiment is cheap and decisive, and say so plainly; dropping it is a legitimate decision at G1.
- **Default arms**: `bw` (control), `bwA`, `bwAB`, in the selection order `bwA`, then `bwAB`. Expected wave: 124 runs (120 + 2 diagnostic + 2 C-LEGACY), about 21 CPU-hours.
- **What the wave cannot show**: a formation failure occurs in 1 of 80 runs, so 40 runs contain 0.5 such runs in expectation; the effect of A on formation failure is tested by the diagnostic (60, 60518) runs, which are never counted.

## 1. The mechanisms that the evidence supports, by class

| class | runs | what the evidence shows | what it does not show |
|---|---|---|---|
| (F) formation failure | 60518 (1 of 80, exact 95 % interval [0.0003, 0.068]) | a stochastic escape from the early bottleneck (policy almost constant in d, sensitivity down to 5.5e-5 at its lowest in this run) that failed; the second hidden layer then saturates, the sensitivity decays to 1e-9, the in-band and out-of-band region gradients are exactly opposed (cosine -1.000) (`01_diagnosis.md` §3, §6) | why this seed did not escape: the depth of the bottleneck predicts the formation time (Spearman -0.92) but 60518 is only slightly deeper than runs that escaped |
| (K) clamp regime with a persistent in-band or stage-2 error | 8 runs (60501, 60506, 60513, 60515, 60516, 50504, 50514, 50517 at q = 60) | the regime (S3 > 0.2, 30 of 80 runs) goes with the Beta mean at the actor's mean clamp 1e-6 at the out-of-band nodes (21 of 30 against 0 of 50), alpha of order 1e-4, entropy of order -1000, 2.7 times larger pre-clip gradient norms, 1.5 times larger KL, a larger drawdown of the tie-point effort, and with every stage-2 stretch (10 of 30 against 0 of 50); the failing quantity is a persistent in-band bias (`eta_3` above 0.005 at 14-16 of the last 16 exports for the four q = 50 tie-point deficits, 11 for 60515, 9 for 60516; the two T3-R1 q = 60 members fail at stage 2) | that the regime causes the errors: the clamped and the out-of-band draws carry 0.3 % and 0.5-0.7 % of the surrogate gradient (DP2 failed); the route through noise or through the shared representation is untested |
| (V) last-iterate variability | 50517 at q = 50 | formed, outside the regime, `eta_3` above 0.005 only at the last export | nothing beyond a single run |

**What A and B act on.** A acts on the **outcome** of every class that shows in the stage-3 check (F, and the stage-3 members of K and V: 7 of the 10 failures), by drawing again. B acts on the **marker** of K (8 runs): it is meant for the 3 K failures that A cannot reach (two stage-2 stretches, one RMSE-only) and also acts on the 5 that A can reach.

## 2. The candidates

### 2.1 Candidate A: end-of-phase check and at most one re-initialisation of stage 3

**What it changes and why.** At the end of the stage-3 phase the development check already computed at local update 1600 (the last row of `ms_checks_stage3.csv`) is read; if it fails, stage 3 is trained once more from a fresh initialisation. The diagnosis supports this form: the formation failure is an escape that either happens at updates 150-700 (79 of 80 runs) or not; a formation failure is detected with a wide margin (Delta 0.1447 against at most 0.0114 in the other 79 at local update 1600, a factor of 13), and the persistent in-band errors of class K appear in the same column. **Published reference for the method (a restart with a cutoff for a randomised procedure with heavy-tailed run outcomes):** M. Luby, A. Sinclair and D. Zuckerman, *Optimal speedup of Las Vegas algorithms* (1993; Information Processing Letters 47, 173-180 as I remember it: the paper and its result on restart strategies with time limits were confirmed by a search, the journal volume and pages were not); C. P. Gomes, B. Selman, N. Crato and H. Kautz, *Heavy-tailed phenomena in satisfiability and constraint satisfaction problems*, Journal of Automated Reasoning 24 (2000) 67-100 (confirmed by search): randomised restarts remove the effect of a heavy tail. The seed sensitivity that motivates drawing again is the subject of P. Henderson et al., *Deep Reinforcement Learning that Matters*, AAAI 2018 (confirmed by search). Used in this project's terms: the "run time" is the stage-3 phase, the "cutoff" is its fixed budget of 1600 updates, and a restart replaces a draw that ended outside the target by a fresh one. The cited papers concern running time, not the final quality of a fixed budget; the transfer is by analogy.

**Exact specification.**

- **Config key** (optional, declared in `run/run_ms_stagewise.py`; absent = `bw`, bit for bit): `stage3_retry = {"delta_gt": 0.005, "max_retries": 1}`.
- **Trigger quantity.** The development check at the last local update of the stage-3 phase (local 1600): `Delta` (= Delta_3 = G_3, the largest one-step deviation gain at stage 3 over the development grid, divided by ΔW), `s` and `C`. **It fires iff `Delta` > 0.005, or `Delta`, `s` or `C` is not finite, or `s` <= 0** (strict, float64; the gate is the complement `eta_3` <= 0.005; the second clause is a degenerate check that could not be evaluated). The `valid` column is deliberately not read: it also depends on stages 1 and 2 of the composite candidate (played by the live stage-3 network), like the `Gmax_full` column; it is true at all 5,120 recorded stage-3 checks, so nothing changes in the recorded data. No closed-form quantity, no `rmse_pos`, `tail_mean`, peak error or `Gmax_full` column is read. No new verifier call: the check is the one that the phase already makes (event `fixed_budget_end`).
- **When and how often.** Once, at the end of the stage-3 phase. At most one retry (two attempts in total); the second attempt is not checked again.
- **Re-initialisation** (attempt 2), **in place** on the existing modules (a new `Linear` module draws from the process-global torch generator before its weights are overwritten, which would be recorded as a global-RNG violation at the next assertion, and a new optimiser object would trip the `optimizer object replaced` guard of `run_stage`): with `g2 = torch.Generator().manual_seed(SeedSequence([seed, q, ns_init, 2]).generate_state(1)[0])` (attempt 1 uses the same expression without the trailing 2), `nn.init.orthogonal_(..., gain, generator=g2)` and zero biases on the actor's `l1`, `l2` (gain sqrt 2), zeros on its output layer, then the critic's `l1`, `l2` (gain sqrt 2) and output (gain 1), in the order of the original construction; both Adam states are cleared in place and the learning rate is set by the schedule restarted at local update 1 (3e-4 to local 1200, linear decay to 3e-5 at 1600); the five streams (environment, learner action, opponent action, starts and roles, minibatch) are `default_rng(SeedSequence([seed, q, ns_name, 2]))`; the opponent snapshot is refreshed at the phase entry; `global_u` and the run counters restart for the attempt. A P2 test checks that the process-global RNG states are unchanged across the retry.
- **Which attempt is frozen.** If the trigger fires, **the second attempt, whatever its own check says** (no selection, no third attempt). There is therefore no choice between attempts that could read anything. A second attempt that also ends above 0.005 fails the run exactly as a first attempt would.
- **Artefacts.** Attempt 1 is written to the standard places while it runs. If the trigger fires, its stage-3 artefacts are **moved** (never overwritten, never deleted) to `attempt1/`: the 64 exports, `ms_checks_stage3.csv`, `ms_binmaps_stage3.npz`, its 1600 rows of `ms_updates.csv` as `ms_updates_stage3.csv` and its history as `train_history_stage3.json` (both are held in memory and written only at the end of a run, so they are dumped explicitly before the reset), and an explicit `torch.save(run.full_state(...))` as `state_attempt1_end.pt` (`state_end_stage3.pt` is written by `freeze_stage`, which attempt 1 never reaches: attempt 1 is not frozen, so it writes no `freeze_stage3_*.npz`, no continuation table and no `rule_log` entry, and costs no final-tier evaluation). The in-memory counters of the run (`rows`, `history`, `weights_log`, `snapshot_log`, `phase_timing` of stage 3, `costs`, `total_episodes`, `total_transitions`, `global_u`, `agent.snapshot_refreshes`) are reset after the dump, and the standard places then hold attempt 2. A new file `retry.json` records the trigger values, the fire decision, the derived seeds and the costs of both attempts. **A run whose trigger does not fire writes exactly the files, columns and rows of `bw`** (and no `retry.json` or `attempt1/`).
- **Budgets and cost.** See §3 (normal case = `bw`: 1600 / 1600 / 600 updates, 1,945,600 policy rows, 76,000 minibatch steps, 152 exports, 64 / 64 / 24 checks; fired: 5,400 updates, 2,764,800 policy rows, 108,000 steps, 216 checks, 152 + 64 exports; expected +3.7 %).
- **How the continuation is rebuilt.** `freeze_stage(3)` runs on the frozen network (attempt 2 if fired) and builds `continuation_table_stage2` from it as in `bw`; stage 2 then continues from that network by the original procedure (1600 updates, the same schedules), and stage 1 after it; the streams continue from the end of the frozen attempt. Nothing in stage 2 or stage 1 changes.
- **Pairing with `bw`** (C-PAIR in D5): where the trigger does not fire the run is bit-identical to the `bw` run of the same seed; where it fires, attempt 1 equals the `bw` run's stage 3 (exports, rows, checks), which `t3r3_pair.py` will check.

**What the retrospective table says about A** (`ms_checks_stage3.csv` at local update 1600 of the 80 recorded runs; `01_diagnosis.md` §11):

| trigger at local update 1600 | fires on | of which failed | failing runs not caught | extra updates | extra cost (share of the 80 x 3800 updates) |
|---|---|---|---|---|---|
| Delta>0.005 | 7 | 7 | 3 | 11200 | 0.03684 |
| R>0.4 | 1 | 1 | 9 | 1600 | 0.005263 |
| R>0.5 | 1 | 1 | 9 | 1600 | 0.005263 |
| R>0.7 | 1 | 1 | 9 | 1600 | 0.005263 |
| R>1 | 1 | 1 | 9 | 1600 | 0.005263 |
| R_tail>0.02 | 1 | 1 | 9 | 1600 | 0.005263 |
| R_tail>0.05 | 1 | 1 | 9 | 1600 | 0.005263 |
| R_tail>0.1 | 1 | 1 | 9 | 1600 | 0.005263 |
| degenerate | 0 | 0 | 10 | 0 | 0 |

| trigger | fired runs | failing runs not caught |
|---|---|---|
| Delta>0.005 | r1_q50_s50504;r1_q50_s50517;r2_q50_s60501;r2_q50_s60506;r2_q50_s60513;r2_q50_s60515;r2_q60_s60518 | r1_q60_s50514;r1_q60_s50517;r2_q60_s60516 |
| R>0.5 | r2_q60_s60518 | r1_q50_s50504;r1_q50_s50517;r1_q60_s50514;r1_q60_s50517;r2_q50_s60501;r2_q50_s60506;r2_q50_s60513;r2_q50_s60515;r2_q60_s60516 |
| R>1 | r2_q60_s60518 | r1_q50_s50504;r1_q50_s50517;r1_q60_s50514;r1_q60_s50517;r2_q50_s60501;r2_q50_s60506;r2_q50_s60513;r2_q50_s60515;r2_q60_s60516 |
| R_tail>0.05 | r2_q60_s60518 | r1_q50_s50504;r1_q50_s50517;r1_q60_s50514;r1_q60_s50517;r2_q50_s60501;r2_q50_s60506;r2_q50_s60513;r2_q50_s60515;r2_q60_s60516 |

| run | pass | Delta | R | R_tail | C |
|---|---|---|---|---|---|
| r2_q60_s60518 | no | 0.1447 | 1.18 | 0.1945 | 0.01477 |
| r2_q50_s60515 | no | 0.01138 | 0.3412 | 0.004153 | 0.04203 |
| r2_q50_s60506 | no | 0.007238 | 0.1956 | 0.003835 | 0.04369 |
| r1_q50_s50504 | no | 0.006227 | 0.1817 | 0.003606 | 0.04222 |
| r1_q50_s50517 | no | 0.005558 | 0.231 | 0.006474 | 0.03084 |
| r2_q50_s60513 | no | 0.005466 | 0.2202 | 0.004088 | 0.03904 |
| r2_q50_s60501 | no | 0.005241 | 0.193 | 0.002257 | 0.03918 |
| r2_q50_s60502 | yes | 0.004654 | 0.2127 | 0.005655 | 0.03153 |
| r1_q50_s50515 | yes | 0.004493 | 0.1851 | 0.003525 | 0.03742 |
| r2_q50_s60507 | yes | 0.004357 | 0.2146 | 0.002516 | 0.03869 |
| r2_q60_s60512 | yes | 0.004065 | 0.1474 | 0.003468 | 0.03862 |
| r2_q60_s60516 | no | 0.003765 | 0.1437 | 0.003072 | 0.03802 |
| r2_q50_s60517 | yes | 0.00337 | 0.1873 | 0.004236 | 0.04037 |
| r1_q50_s50506 | yes | 0.003147 | 0.1305 | 0.003026 | 0.03869 |

Largest values among the other 79 runs (without 60518): R 0.3412 (r2_q50_s60515), R_tail 0.0103 (r2_q60_s60510); 60518: R 1.1800, R_tail 0.1945.

- **A gate-level trigger (`Delta` > 0.005)** fires on exactly the 7 stage-3 failures (7 of 7 failed), on none of the 73 other runs, and misses 3 failures (the two stage-2 stretches of T3-R1 at q = 60 and 60516, whose `eta_3` is 0.0038). It costs 11,200 extra updates, 3.7 % of the 80 runs' compute. **It gives runs that end near or above the limit a second draw. It raises the pass rate without changing how a stage is learned, and a paper must describe it that way.** It is the development-tier version of the gate itself; a run at 0.0047 (60502) stays below the trigger, a run at 0.0052 (60501) is above it, and the difference between those two runs is within the noise of one seed.
- **A formation-level trigger (`R` or `R_tail`)** fires on 60518 alone for every threshold between the largest value of the other 79 runs (R 0.341, R_tail 0.0103) and 60518's (1.18, 0.195); it costs 0.5 % and **catches gross failures only**: it cannot change the pass rate of the 9 other failures. Its threshold is set from this single observation.

### 2.2 Candidate B: censored likelihood of clamped draws in the stage-3 phase

**What it changes and why.** The actor samples `a ~ Beta(alpha, beta)` and stores `clip(a, 1e-6, 1 - 1e-6)`. A draw that was clipped is *censored*: what is known is the event `A <= 1e-6`, not a density at 1e-6. The locked pipeline still evaluates the density at the clipped value, which for small alpha is not the probability of anything; the derivative of the log-density at a = 1e-6 with respect to alpha is of order +1/alpha (+9992 at alpha = 1e-4) and changes sign between alpha = 0.1 and 0.3, while the derivative of the log-mass stays between -8.2 and -10.0 for alpha from 1e-4 to 3:

| alpha | d log density at a = 1e-6 / d alpha | d log P(A <= 1e-6) / d alpha | P(A <= 1e-6) |
|---|---|---|---|
| 0.0001 | 9992 | -8.231 | 0.9992 |
| 0.001 | 991.8 | -8.233 | 0.9918 |
| 0.01 | 91.75 | -8.247 | 0.9209 |
| 0.05 | 11.69 | -8.31 | 0.6613 |
| 0.1 | 1.616 | -8.384 | 0.4356 |
| 0.3 | -5.304 | -8.637 | 0.07934 |
| 1 | -8.224 | -9.224 | 0.00015 |
| 3 | -9.711 | -10.04 | 5.74e-13 |

(`utils.beta_tail.log_betainc_small_x`, beta = 150, autograd; `proposal_tables.py`.) The regime of §1 is exactly the state in which this matters: the Beta at the out-of-band nodes sits at alpha of order 1e-4 with the mean at its clamp. **Published reference:** Y. Fujita and S.-i. Maeda, *Clipped Action Policy Gradient*, ICML 2018 (PMLR 80, 1597-1606; confirmed by search): policy-gradient methods clip out-of-bound actions before execution but optimise as if they were not clipped, and the clipped-action estimator uses the knowledge of the clipping (shown unbiased, with lower variance); the Beta policy for bounded actions is P.-W. Chou, D. Maturana and S. Scherer, *Improving stochastic policy gradients in continuous control with deep reinforcement learning using the Beta distribution*, ICML 2017 (PMLR 70, 834-843; confirmed by search). The code is in this repository since R2b (`utils/beta_tail.py`, tested against `scipy.special.betainc`; `collect_batch_v2(..., clamp_likelihood="censored")`; `CurriculumPPOv2.update(..., clamp_side=...)`).

**Evidence for and against.**

- *For*: the marker (Beta mean at the mean clamp in 21 of 30 regime runs, none of 50 others; alpha 2e-4 against 0.4; entropy -1000 against +0.3; pre-clip gradient norm 2.7 times larger and its maximum 3 times larger); every stage-2 stretch is in the regime.
- *Against*: the clamped draws carry 0.3 % of the projection of the surrogate gradient against 53-57 % of the samples, and their per-sample gradient norm (10) is *below* that of the in-band draws (102-105) (`01_diagnosis.md` §6); the tie-point effort does not fall after the onset (§7); the T = 2 precedent R2b `A_censored` (at a clamp share of the order 0.01-0.2, the T = 2 analogue of `01_diagnosis.md` §7) found paired differences of the peak error whose intervals contain 0 at both q (`reports/v2/refine_r2b/summary.md`). A dominant role of the clamped draws in the mean gradient is therefore not what the data show (the projection share is a first-order share of the raw gradient; with Adam and a global-norm clip a 0.3 % share does not exclude an effect on parameters where the in-band gradient is small); any effect would act through noise or through the representation that the logit swing of about 15 units across the band edge asks of the network.
- *The experiment*: `bwAB` against `bwA` on the same seeds is the intervention that separates cause from marker. If S3, the stretches and the in-band bias fall in `bwAB`, the regime was more than a marker; if not, the regime is a marker, and B should be dropped.

**Exact specification.**

- **Config key** (optional; absent = the density path, bit for bit): `terminal_clamp_likelihood = "censored"`. It acts in the phase of the terminal stage T only (stage 3): `collect_batch_v2(..., clamp_likelihood="censored")` and `agent.update(..., clamp_side=batch["clamp_side"])`. The key `clamp_likelihood` stays `"density"` (the validator refuses anything else for the other phases).
- **Unchanged**: the network, the optimiser, the schedules, the starts, stage 2 and stage 1 (they use `collect_stage_batch` and the density), the verifier, the evaluation. An update without any flagged row is dispatched as without the argument (bit-identical, `CurriculumPPOv2.update` docstring); a stage-3 phase with no draw below the clamp is bit-identical to `bw`.
- **Domain**: `x * beta <= 0.05`, i.e. beta <= 5e4 at c = 1e-6 (the concentration of the actor is `100 + softplus`, a few hundred at most in the recorded runs); outside it the function raises, nothing is silently wrong.
- **Cost**: at most 1.17 times the plain update on a batch with 466 of 512 flagged rows (measured), less for a typical regime run (about 57 % of the rows); a stage-3 phase is 39 % of a run, so at most about +7 % of a run, typically +2-3 %.
- **Interaction with A**: in `bwAB` the second attempt is also censored.
- **Not paired with `bw`**: where a flagged row exists the run differs from `bw` from the first such update on.

### 2.3 Considered and not proposed

- **A formation-level trigger as an arm** (`R` > 0.7): it fires on one of 80 runs; the model gives it 0.15 for the target (§3). It is a subset of A's trigger (60518 has `Delta` 0.1447), so A already contains it.
- **Peak-focused (in-band) exploring starts**, R2b mechanism 1: improved the T = 2 peak error at both q but crossed the G-A tail-mean limit in 3 of 10 q = 60 runs; at T = 3 the gradient is already 99 % in band and the starts are uniform over the bins (`01_diagnosis.md` §6), the evidence of benefit is T = 2 only, and it needs a new sampler.
- **An entropy bonus** against the alpha collapse (`entropy_coef` exists in `PPOConfig`): no evidence on its size, it would act on all stages, and the entropy of a Beta against its lower end is unbounded below.
- **A change of architecture or initialisation** against the saturation of 60518: one observation, and the saturation follows the loss of sensitivity.
- **An early cutoff** (check at local 800 instead of 1600): saves 0.26 % of the compute and would need a second check point.

## 3. The development arms, order and cost

| arm | what | order in the selection rule | expected updates per run | worst case |
|---|---|---|---|---|
| `bw` | the locked T3-1.0 solver (control; not eligible) | - | 3,800 | 3,800 |
| `bwA` | candidate A | 1 | 3,940 (+3.7 %) | 5,400 (+42 %) |
| `bwAB` | A and B | 2 | 3,940 plus the censored update (estimated +2-3 % wall) | 5,400 |

The first candidate arm in this order that reaches the target (at least 38 of 40 and at least 18 of 20 per q) is the scheme proposed for the lock. If `bwA` and `bwAB` both reach it, `bwA` (the simpler scheme) is proposed.

| case | updates | policy rows | minibatch steps | development checks | exports (frozen path) | relative to bw |
|---|---|---|---|---|---|---|
| bw, and bwA / bwAB when the trigger does not fire | 3800 | 1945600 | 76000 | 152 | 152 | 1 |
| bwA / bwAB when the trigger fires (stage 3 trained twice) | 5400 | 2764800 | 108000 | 216 | 152 (+ 64 in attempt1/) | 1.421 |
| bwA / bwAB expected, fire rate 7 / 80 = 0.0875 | 3940 | 2.02e+06 | 7.88e+04 | 157.6 | 152 | 1.037 |
| bwA / bwAB expected, lower limit of the 95 % interval = 0.0359 | 3857 | 1.98e+06 | 7.715e+04 | 154.3 | 152 | 1.015 |
| bwA / bwAB expected, upper limit = 0.1720 | 4075 | 2.09e+06 | 8.15e+04 | 163 | 152 | 1.072 |

Wall time of the 80 recorded `bw` runs (`status.json` `total_wall_sec`): median 600 s, range 503-640 s; the stage-3 phase (`ms_run_summary.json`) median 235 s, i.e. 39% of a run. One retraining therefore costs about 235 s; a wave of 40 `bw` runs costs about 6.7 CPU-hours, of 40 runs of an arm with the expected extra stage-3 work about 6.9. The censored update measured on a synthetic batch of 512 rows with 466 flagged rows (`agent.update` with `clamp_side`, mean of 3 calls, one thread) took 1.17 times the plain update (0.1355 against 0.1161 s); a regime run with about 57 % flagged rows costs less.

**The target as a function of the per-run pass probability, and of the later confirmation** (exact binomial, independent runs, the same probability at both q; the script reproduces the prompt's 0.882 / 0.677 / 0.223 / 0.108 and 0.982 / 0.865 / 0.271 / 0.030):

| per-run pass probability p (both q) | P(development target: >= 38 of 40 and >= 18 of 20 per q) |
|---|---|
| 0.99 | 0.9925 |
| 0.97 | 0.8822 |
| 0.95 | 0.6767 |
| 0.9 | 0.2228 |
| 0.875 | 0.1084 |

The later confirmation (D6):

| per-run failure probability f (both q) | P(confirmation: <= 6 of 80 and <= 4 of 40 per q) |
|---|---|
| 0.01 | 0.9999 |
| 0.03 | 0.9818 |
| 0.05 | 0.8648 |
| 0.1 | 0.2708 |
| 0.15 | 0.02989 |

**Model projection of the target for the arms** (a sizing aid, not a prediction; assumptions under the table):

| arm / trigger | assumed rates | fail q50 | fail q60 | expected failures in 40 runs | P(development target) |
|---|---|---|---|---|---|
| bw (control, observed rates) | central | 0.15 | 0.1 | 5 | 0.1077 |
| gate-level: Delta > 0.005 | central | 0.0225 | 0.0775 | 2 | 0.6768 |
| gate-level: Delta > 0.005 | pessimistic | 0.1827 | 0.2224 | 8.102 | 0.007143 |
| gate-level: Delta > 0.005 | optimistic | 0.003261 | 0.01615 | 0.3883 | 0.9933 |
| formation-level: R > 0.7 | central | 0.15 | 0.0775 | 4.55 | 0.1498 |

Rates from the 80 recorded runs (`retro_checks_local1600.csv`): failures q50 6 of 40, q60 4 of 40; the gate-level trigger fires on 6 (q50) and 1 (q60) runs and misses 0 and 3 failures; the formation-level trigger fires on 0 and 1. 'pessimistic' / 'optimistic' use the upper / lower limit of the exact 95 % interval of every rate. **This is a model, not a result**: it assumes independent runs, a re-drawn stage 3 distributed as a first attempt, a run that fires again failing, and rates estimated from 40 runs per q; it is shown to size the target, not to predict the wave.

The projection says: with A the target is at the margin, and its failures are the stage-2 members of class K (q = 60: 3 of 39 non-fired runs failed); the effect of A on a formation failure (60518) is tested by the diagnostic run, not by the 40 counted runs. If B removes the regime, the non-fired failure rate falls and `bwAB` would meet the target with a higher probability; if it does not, `bwAB` is `bwA` plus noise.

## 4. Draft of the P2 pre-registration (`04_preregistration.md`; D5 and D6 filled in)

- **Arms and keys.** `bw`: `tools/t3/t3_configs.build_bw_config` with `reports/t3/r1/params_bw_t3.json`, checked against the lock's `reference_config_sha256` (`tools/t3/t3c_launch.normalise`, `config_problems`). `bwA`: the same config plus `stage3_retry = {"delta_gt": 0.005, "max_retries": 1}`. `bwAB`: `bwA` plus `terminal_clamp_likelihood = "censored"`. A test checks with `tools/t3/t3_configs.config_diff` that a candidate config differs from the control's of the same (q, seed) only in these keys and the run fields (`seed`, `run`, `record.run`, `record.seed`, `record.output_dir`, `pilot` = "t3_r3").
- **Seeds and names.** 70501-70520 at q = 50 and q = 60, the same seeds in every arm; run names `t3r3_q{q}_s{seed}_{arm}`; directories `results/t3_r3/dev/q{q}/seed{s}/{arm}/`. Diagnostic: each candidate arm once at (q = 60, seed 60518) into `results/t3_r3/diagnostic/q60/seed60518/{arm}/`, never counted, compared with T3-R2's original run (C-INIT, C-PAIR). C-LEGACY: T3-R2's (q = 50, seed 60502) and (q = 60, seed 60518) re-run from their own `run_config.json` into `results/t3_r3/legacy/` and compared with `tools/t3/compare_runs.py bw` (both must be bit-identical; a mismatch is a stop). Reserved and never trained: 80501-80540 (inventory in `results/t3_r3/preflight/`: 0 collisions, 0 hits).
- **Order before the launch.** Independent review of the P2 code; C-LEGACY at the final code commit; the pre-registration pushed; the launch from the commit that holds it, clean tree, at most 40 single-threaded workers, in tmux.
- **Post-launch checks** (`results/t3_r3/dev/launch_checks.json`): basics (exit codes, files, manifests at the launch commit with a clean tree, global-RNG assertions), budgets (`bw`: 1600 / 1600 / 600, 819,200 / 819,200 / 307,200 rows, 76,000 steps; candidates: the budgets of §2 including a retraining), schedules (learning rate at every update, 152 exports, 64 / 64 / 24 checks), C-INIT (init digests equal the planned ones, equal across arms at the same seed unless the scheme changes the initial network; the attempt-2 digest equals the derived one), C-REPLAY (the evaluator reproduces the freeze arrays), C-PAIR (`bwA`: bit-identical to `bw` where the trigger did not fire; where it fired, `t3r3_pair.py` shows attempt 1 equals `bw`'s stage 3 up to the check). `bwAB` is compared with `bw` for the diagnostic and reported only.
- **Evaluation and target.** `tools/t3/t3_eval.evaluate_run` on both tiers through a new driver; the policy is the one the scheme freezes; G-A, G-F, G-N and the run pass as in T3-1.0; S3 on the frozen attempt. A candidate arm reaches the target iff at least 38 of its 40 runs pass and at least 18 of 20 at each q; every planned run counts; failures counted as in D6. Reported, deciding nothing: pass counts with exact intervals, worst `Gmax_full` and its run, medians of `eta_3`, `rmse_pos_T_over_eT0`, `tail_mean_T_over_eT0`, `peak_T_signed`, `Gmax_full`, EXP_root; paired differences of `Gmax_full` and `eta_3` against `bw` with 10,000-resample paired bootstrap intervals (seed 20261021) and the pass/fail cross-table; S3 of every run; the stage-2 stretches; every trigger that fired with its outcome; the compute cost per arm and its ratio to `bw`; a blind recomputation of the counts and the target outcome by an independent tool.
- **Selection.** Only candidate arms are eligible; the first in the order `bwA`, `bwAB` that reaches the target is the scheme proposed for the lock. Nothing is locked in this round.
- **The later confirmation (D6).** The one proposed scheme, locked as T3-1.1; seeds 80501-80540 at both q (80 runs); passes iff at most 6 of 80 fail and at most 4 of 40 at each q; the operating characteristic is in the table of §3 (0.982 / 0.865 / 0.271 / 0.030 at f = 0.03 / 0.05 / 0.10 / 0.15); failure counting as in T3-R2's D4 (infrastructure failures re-run once with an `infra_events.json`, other nonzero exits and global-RNG violations count as failed, an evaluation error counts as failed unless a re-run of the locked evaluator succeeds; **a re-initialisation inside a run is part of the run, never a re-run**).
- **Predictions D7**: verbatim in the pre-registration; DP4 (no prediction of the outcome) stands. The additions of this proposal that the wave can contradict: A will fire on about 9 % of the runs (7 of 80 in the recorded runs); `bwA` and `bw` are bit-identical where A does not fire; B is not expected to change the fire rate if the regime is only a marker.

## 5. Flags for G1

1. **A is a gate-level restart.** In a paper: "a run whose development one-step deviation gain at the end of the terminal-stage phase exceeded the gate's limit was re-initialised once"; the pass rate then includes a second draw for those runs, the cost is reported.
2. **B has the weakest evidence of the three things in this round.** DP2 failed; the T = 2 precedent was null. Dropping it (arms `bw`, `bwA`) is consistent with the diagnosis; keeping it buys the causal test.
3. **The wave cannot validate A on formation failures** (expected 0.5 events in 40 runs); the diagnostic runs are the only direct test and are not counted.
4. **The thresholds.** 0.005 is the gate's own number, fixed before this round. The `R` / `R_tail` thresholds of the formation-level alternative are set from one run and are not proposed for any arm.
5. **Implementation points for P2 that need care** (not blockers): the global-update counter and the exports of a retried stage 3; keeping `total_updates` / budget bookkeeping of `gates.json` for the frozen path while `retry.json` carries the total; resetting the run counters listed in §2.1 (and nothing else); a test that the process-global RNG states and the optimiser identity guard are unaffected by the retry; `validate_config` must declare both keys; every change to `run/run_ms_stagewise.py` additive behind its key, with C-LEGACY.
6. **Nothing was tuned on a seed of the development block or on the reserved block.** The recorded 80 runs were used for the retrospective table only.
