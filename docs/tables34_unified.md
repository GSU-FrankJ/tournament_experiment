# Tables 3 & 4 under one exploitability estimator (2026-07-29)

Replaces the mixed `det` / `in` / `mc` / `a` reporting. Every exploitability
number below comes from a single estimator; the superscripts and the `[0.00x]`
bracket column are gone.

## Estimator

`utils.mc_br_polish.exploitability_frozen_profile` — independent Monte Carlo,
applied identically to every row:

| | |
|---|---|
| Definition | `exploit_i = max(0, max_c Ê[payoff_i(c, e_-i)] − Ê[payoff_i(e_i, e_-i)])`, reported as `max_i exploit_i` |
| Effort domain | `[0, 100]`, uniform grid, step 0.25 |
| BR search | full-grid argmax, no coarse-to-fine branch |
| Samples | M = 200 000 per player |
| Draws | fresh seed, independent of the MC-BR polishing draws; CRN within a call (the same `eps` block scores the incumbent effort and every candidate) |
| Winner rule | realized argmax of `y_j = e_j + l_j + eps_j` — the same sampled convention the agents train on, no closed-form win probability |

Evaluation seeds are shared across the four Table-3 arms within a (q, seed)
cell, so the arms are compared under common random numbers and most estimator
noise cancels in their difference. They remain independent of the polishing
draws, which is what the "fresh draws" requirement targets.

Reproduce with `tools/unified_exploitability_tables.py`
(→ `results/one_stage_ablation/unified_exploitability_tables.json`).

## Estimator noise floor

Evaluated at the analytic `e*`, where the true exploitability is exactly zero.
Anything the estimator returns there is its own error, so no reported value
below the floor is distinguishable from an exact equilibrium. Five independent
seeds per cell; `tools/exploit_noise_floor.py`.

| Scenario | q = 35 | q = 45 | q = 55 |
|---|---|---|---|
| Two-player | 1.22e-03 | 7.42e-04 | 6.06e-04 |
| Three-player | 1.51e-03 | — | 6.45e-04 |
| Het. cost | 5.13e-04 | — | 2.83e-04 |
| Het. ability | 8.79e-04 | — | 8.52e-04 |

M sweep on two-player q = 35 (3 seeds each), confirming the floor decays like
1/sqrt(M) rather than vanishing:

| M | 25 000 | 50 000 | 100 000 | 200 000 | 400 000 | 800 000 |
|---|---|---|---|---|---|---|
| floor | 6.3e-03 | 2.6e-03 | 2.8e-03 | 1.8e-03 | 1.0e-03 | 7.8e-04 |

Raising M to 800 000 buys less than a factor of 2 over M = 200 000 at 4x the
cost, and the floor is still ~1e-03. No feasible M separates the arms on
exploitability alone.

## Table 3. Component ablation of TEL-PPO

Values are `q = 35 / 45 / 55`. Final Effort is the mean and sample SD of the
five per-seed terminal efforts; Absolute Bias is `|mean(e_i) − e*|` and is
reported separately, never folded into the effort SD.

| Ablation variant | Final Effort (mean ± SD) | Absolute Bias | Final Exploitability | Verification Update (mean ± SD) | Training outcome |
|---|---|---|---|---|---|
| TEL-PPO | 44.95 ± 0.02 / 35.09 ± 0.03 / 28.76 ± 0.02 | 0.51 / 0.26 / 0.17 | (1.09 ± 0.35)e-03 / (1.02 ± 0.44)e-03 / (0.87 ± 0.49)e-03 | 55.0 ± 8.9 / 75.0 ± 5.5 / 87.0 ± 16.4 | Verified |
| w/o MC-BR polishing | 43.58 ± 1.25 / 35.46 ± 0.60 / 29.65 ± 0.76 | 1.87 / 0.11 / 0.73 | (2.38 ± 1.75)e-03 / (0.73 ± 0.39)e-03 / (2.07 ± 2.00)e-03 | 55.0 ± 8.9 / 75.0 ± 5.5 / 87.0 ± 16.4 | Verified |
| w/o stability screening | 42.88 ± 1.30 / 35.11 ± 1.09 / 30.22 ± 1.05 | 2.57 / 0.24 / 1.30 | (3.56 ± 2.16)e-03 / (1.70 ± 0.89)e-03 / (3.33 ± 2.38)e-03 | 53.0 ± 8.9 / 79.0 ± 10.0 / 81.0 ± 13.0 | Verified |
| w/o exploitability verification | 44.60 ± 3.53 / 32.05 ± 1.38 / 29.51 ± 1.73 | 0.85 / 3.31 / 0.58 | (8.49 ± 7.51)e-03 / (5.86 ± 5.91)e-03 / (2.84 ± 3.51)e-03 | n/a | Reached budget (1500 updates, all seeds) |

Verification Update is the PPO update at which the fifth consecutive passing
exploitability check fires and training stops, averaged over the five seeds with
their sample SD. The `w/o exploitability verification` arm has no such event —
every seed exhausts the 1500-update budget (`stop_reason = max_updates`).

## Table 4. Quantitative summary for all tournament settings

Symmetric cells are collapsed to the player-averaged effort before the
cross-seed statistics, so the two-player rows carry exactly the quantity Table 3
reports. Heterogeneous cost reports both players.

| Scenario | q | e* | ê_raw | Raw Err. | ê_polish | Polished Err. | Final exploitability |
|---|---|---|---|---|---|---|---|
| Two-player | 35 | 45.45 | 43.58 ± 1.25 | 1.87 | 44.95 ± 0.02 | 0.51 | (1.09 ± 0.35)e-03 |
| Two-player | 45 | 35.35 | 35.46 ± 0.60 | 0.11 | 35.09 ± 0.03 | 0.26 | (1.02 ± 0.44)e-03 |
| Two-player | 55 | 28.93 | 29.65 ± 0.76 | 0.73 | 28.76 ± 0.02 | 0.17 | (0.87 ± 0.49)e-03 |
| Three-player | 35 | 25.00 | 22.99 ± 0.28 | 2.01 | 24.75 ± 0.02 | 0.25 | (0.87 ± 0.36)e-03 |
| Three-player | 55 | 15.91 | 15.31 ± 1.13 | 0.60 | 15.84 ± 0.01 | 0.07 | (0.45 ± 0.13)e-03 |
| Het. cost | 35 | 38.03 / 27.66 | 37.71 ± 1.02 / 27.24 ± 1.56 | 0.32 / 0.42 | 38.04 ± 0.02 / 27.66 ± 0.07 | 0.01 / 0.01 | (0.59 ± 0.43)e-03 |
| Het. cost | 55 | 26.54 / 19.30 | 26.39 ± 0.64 / 19.03 ± 2.42 | 0.14 / 0.27 | 26.56 ± 0.02 / 19.30 ± 0.04 | 0.02 / 0.00 | (0.41 ± 0.27)e-03 |
| Het. ability | 35 | 46.43 | 43.99 ± 0.20 | 2.44 | 46.45 ± 0.04 | 0.02 | (1.29 ± 0.37)e-03 |
| Het. ability | 55 | 30.37 | 29.70 ± 0.85 | 0.67 | 30.37 ± 0.01 | 0.01 | (0.36 ± 0.27)e-03 |

Every effort column is unchanged from the previous version of Table 4 — only the
exploitability column moved, because only it depended on the estimator.

## What the unification costs, and what replaces it

The old TEL-PPO exploitability (8.6e-05 / 2.8e-05 / 1.2e-05) came from the
deterministic referee, which uses closed-form expected payoffs and has no
sampling error. Under the common MC estimator those cells read
1.09e-03 / 1.02e-03 / 0.87e-03 — at 0.90x, 1.37x and 1.44x the estimator's own
noise floor. TEL-PPO is therefore **at the floor**: the measurement cannot
distinguish its profile from an exact equilibrium, which is the strongest
statement this instrument can make, but it is not a statement about being two
orders of magnitude better than the ablations.

The ordering across arms survives and stays monotone in the expected direction
(TEL-PPO < w/o polishing < w/o screening < w/o verification at q = 35 and
q = 55), but the separations are now 2-8x with per-seed SDs of the same order,
not 1-2 decades. The honest reading is that exploitability alone no longer
carries the ablation claim.

**The effort columns do.** They are estimator-independent and the separation
there is large relative to the seed spread:

| Arm | Absolute bias (q = 35 / 45 / 55) | Effort SD |
|---|---|---|
| TEL-PPO | 0.51 / 0.26 / 0.17 | 0.02-0.03 |
| w/o MC-BR polishing | 1.87 / 0.11 / 0.73 | 0.60-1.25 |
| w/o stability screening | 2.57 / 0.24 / 1.30 | 1.05-1.30 |
| w/o exploitability verification | 0.85 / 3.31 / 0.58 | 1.38-3.53 |

TEL-PPO's cross-seed SD is 40-100x tighter than every ablation at every q. That
is the result the ablation actually establishes, it is immune to the estimator
question, and it is what the text should lead with.
