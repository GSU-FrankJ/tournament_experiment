# D2 - verifier sensitivity to mild policy misspecification

Descriptive report (T57 issue 8, PROMPT.md section 3.2). It is a sensitivity study of the DP-BR verifier on closed-form perturbations of the equilibrium; it changes no threshold, no gate and no protocol file. Every number below is read from the CSV named under the table or in the sentence; verdict strings are copied from the CSV.

## 1. Provenance and reproduction

| item | value | source |
|---|---|---|
| code commit | `6c902dbd5943597088151c75919406ce0e030de5` | `results/v2_refine/d2_verifier_sensitivity/meta.json` |
| dirty flag (tracked changes or untracked files outside results/) | no | `results/v2_refine/d2_verifier_sensitivity/meta.json` |
| created | 2026-10-03 19:17:37 | `results/v2_refine/d2_verifier_sensitivity/meta.json` |
| protocol (read only) | `protocols/v2_T2_locked_v1_1.json` sha256 `21d85983f2a2bebc...` | `results/v2_refine/d2_verifier_sensitivity/meta.json` |
| q values / tiers | [50, 60] / ['development', 'final'] | `results/v2_refine/d2_verifier_sensitivity/meta.json` |
| families run | a b c d e | `results/v2_refine/d2_verifier_sensitivity/meta.json` |
| evaluations (rows) / errors | 152 / 0 | `results/v2_refine/d2_verifier_sensitivity/meta.json` |
| workers / total wall seconds | 4 / 7.7786 | `results/v2_refine/d2_verifier_sensitivity/meta.json` |

Commands (from the repository root; single-threaded processes):

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/v2/verifier_sensitivity.py --out results/v2_refine/d2_verifier_sensitivity --workers 4 --qs 50 60 --tiers development final --families a b c d e
/home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/verifier_sensitivity.py --out results/v2_refine/d2_verifier_sensitivity --report --report-path reports/v2/refine/03_d2_verifier_sensitivity.md
```

Per-row commit and dirty flag are also columns of `results/v2_refine/d2_verifier_sensitivity/evaluations.csv`; the figures are in `results/v2_refine/d2_verifier_sensitivity/figures/` (png and pdf).

## 2. What was evaluated

Candidates are deterministic `MeanPolicy` callables built from the closed-form T=2 equilibrium (`utils.theory_multistage.g1_two_stage`, `g2_two_stage`); stage 1 is queried at d = 0 only. Each candidate is evaluated with `utils.v2_metrics.evaluate` on the development and the final verifier tier (`utils.dp_br_verifier.DEV_CONFIG`, `FINAL_CONFIG`) with the locked recovery step and the game parameters of `protocols/v2_T2_locked_v1_1.json` (records[q].game, records[q].protocol.recovery_step). eta_2/DW is the scalar `eta_T_over_dw` (T = 2).

| family | perturbation | grid |
|---|---|---|
| (a) | e1 = (1 + d) e1*, stage 2 exact | d in [-0.15, -0.1, -0.05, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15] |
| (b) | e2(d) = (1 + d) e2*(d), stage 1 exact | d in [-0.15, -0.1, -0.05, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15] |
| (c) | e2* convolved with a uniform kernel of half-width h (d units, 0.01 grid, linear interpolation), stage 1 exact | h in [2.0, 5.0, 10.0, 12.0, 20.0] |
| (d) | e2(d) = e2*(d) + tau for abs(d) >= 2q, stage 1 exact | tau in [0.25, 0.5, 1.0, 2.0] (effort units) |
| (e) | (a) with the (c) kernel whose induced peak error is closest to -0.06 | d in [0.05, 0.1, 0.15], h chosen per q |

Source of the grids: `tools/v2/verifier_sensitivity.py` (constants DELTA_GRID, KERNEL_H, TAU_GRID, E_DELTAS, KERNEL_TARGET); the candidates actually run are the rows of `results/v2_refine/d2_verifier_sensitivity/evaluations.csv`.

| q | w_h | w_l | k | DW | e_min..e_max | e1* = DW/(6kq) | e2*(0) = DW/(4kq) | B | recovery step |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 6 | 2 | 0.00028571 | 4 | 0..100 | 46.667 | 70 | 200 | 0.5 |
| 60 | 6 | 2 | 0.00028571 | 4 | 0..100 | 38.889 | 58.333 | 220 | 0.5 |

Source: `results/v2_refine/d2_verifier_sensitivity/meta.json` (games).

Clipping. Efforts are clipped to [e_min, e_max] at the output of the policy, because the verifier rejects efforts outside that interval; the clip is inspected on the unclipped stage-1 value and on a dense 0.01 grid over the whole stage-2 domain. Rows in which the clip changes a value: 0 of 152; largest unclipped effort over all rows: 80.5, smallest: 0 (`results/v2_refine/d2_verifier_sensitivity/evaluations.csv`, columns clip_binds, clip_raw_min, clip_raw_max).

Family (e) kernel choice (the (c) kernel whose induced peak error is closest to -0.06, chosen per q from the (c) results of the same run):

| q | chosen h | peak error of that kernel | |distance to target| | peak error of every h (h: value) | source |
|---|---|---|---|---|---|
| 50 | 12 | -6.000e-02 | 7.702e-16 | 2.0: -0.01; 5.0: -0.025; 10.0: -0.05; 12.0: -0.06; 20.0: -0.1 | family (c) evaluation rows of this run |
| 60 | 12 | -5.000e-02 | 1.000e-02 | 2.0: -0.0083333; 5.0: -0.020833; 10.0: -0.041667; 12.0: -0.05; 20.0: -0.083333 | family (c) evaluation rows of this run |

Source: `results/v2_refine/d2_verifier_sensitivity/meta.json` (kernel_choice).

## 3. Unperturbed control (d = 0: the exact closed-form candidate)

| q | tier | valid | Gmax_full/DW | (t*, d*) | eta_2/DW | EXP_root/DW | dReach/DW | Delta_max_all/DW | dFull/DW |
|---|---|---|---|---|---|---|---|---|---|
| 50 | development | yes | 2.220e-16 | (2, -16) | 2.220e-16 | 0.000e+00 | 2.220e-16 | 2.220e-16 | 2.220e-16 |
| 50 | final | yes | 2.220e-16 | (2, -24) | 2.220e-16 | 0.000e+00 | 2.220e-16 | 2.220e-16 | 2.220e-16 |
| 60 | development | yes | 2.220e-16 | (2, 40) | 2.220e-16 | 0.000e+00 | 2.220e-16 | 2.220e-16 | 2.220e-16 |
| 60 | final | yes | 3.511e-07 | (1, 0) | 2.220e-16 | 3.511e-07 | 3.511e-07 | 3.511e-07 | 3.511e-07 |

Source: `results/v2_refine/d2_verifier_sensitivity/evaluations.csv` (family a, perturbation 0). These are the numerical floors of the verifier on the exact equilibrium; the responses below are read against them.

## 4. Family (a) stage-1 scalar

Perturbation parameter: `delta_stage1` (relative). Fit abscissa x: `delta_stage1`.

### q = 50: verifier quantities

| perturbation | x | G_dev | G_final | (t*, d*) final | G dev-final | eta_dev | eta_final | eta dev-final | EXP_root final | dReach final | Delta_max_all final | dFull final | valid dev/final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -0.15 | -0.15 | 5.876e-03 | 5.893e-03 | (1, 0) | -1.615e-05 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 5.893e-03 | 5.893e-03 | 5.893e-03 | 5.893e-03 | yes/yes |
| -0.1 | -0.1 | 2.800e-03 | 2.725e-03 | (1, 0) | 7.543e-05 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 2.725e-03 | 2.725e-03 | 2.725e-03 | 2.725e-03 | yes/yes |
| -0.05 | -0.05 | 7.891e-04 | 7.337e-04 | (1, 0) | 5.535e-05 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 7.337e-04 | 7.337e-04 | 7.337e-04 | 7.337e-04 | yes/yes |
| -0.02 | -0.02 | 1.411e-04 | 1.202e-04 | (1, 0) | 2.091e-05 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 1.202e-04 | 1.202e-04 | 1.202e-04 | 1.202e-04 | yes/yes |
| -0.01 | -0.01 | 4.899e-05 | 3.946e-05 | (1, 0) | 9.537e-06 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 3.946e-05 | 3.946e-05 | 3.946e-05 | 3.946e-05 | yes/yes |
| -0.005 | -0.005 | 9.060e-06 | 1.028e-05 | (1, 0) | -1.224e-06 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 1.028e-05 | 1.028e-05 | 1.028e-05 | 1.028e-05 | yes/yes |
| 0 | 0 | 2.220e-16 | 2.220e-16 | (2, -24) | 0.000e+00 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 0.000e+00 | 2.220e-16 | 2.220e-16 | 2.220e-16 | yes/yes |
| 0.005 | 0.005 | 2.220e-16 | 1.131e-05 | (1, 0) | -1.131e-05 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 1.131e-05 | 1.131e-05 | 1.131e-05 | 1.131e-05 | yes/yes |
| 0.01 | 0.01 | 2.486e-05 | 4.127e-05 | (1, 0) | -1.641e-05 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 4.127e-05 | 4.127e-05 | 4.127e-05 | 4.127e-05 | yes/yes |
| 0.02 | 0.02 | 1.128e-04 | 1.273e-04 | (1, 0) | -1.448e-05 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 1.273e-04 | 1.273e-04 | 1.273e-04 | 1.273e-04 | yes/yes |
| 0.05 | 0.05 | 8.560e-04 | 7.915e-04 | (1, 0) | 6.449e-05 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 7.915e-04 | 7.915e-04 | 7.915e-04 | 7.915e-04 | yes/yes |
| 0.1 | 0.1 | 3.283e-03 | 3.165e-03 | (1, 0) | 1.178e-04 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 3.165e-03 | 3.165e-03 | 3.165e-03 | 3.165e-03 | yes/yes |
| 0.15 | 0.15 | 7.361e-03 | 7.170e-03 | (1, 0) | 1.904e-04 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 7.170e-03 | 7.170e-03 | 7.170e-03 | 7.170e-03 | yes/yes |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family a, q 50); G = Gmax_full/DW, eta = eta_2/DW, all in units of DW; n/a = tier not run.

### q = 50: recovery metrics (tier independent) and clip

| perturbation | stage-1 error (signed) | stage-2 peak error (signed) | RMSE_pos / e2*(0) | tail mean / e2*(0) | clip binds |
|---|---|---|---|---|---|
| -0.15 | -1.500e-01 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| -0.1 | -1.000e-01 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| -0.05 | -5.000e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| -0.02 | -2.000e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| -0.01 | -1.000e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| -0.005 | -5.000e-03 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| 0 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| 0.005 | 5.000e-03 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| 0.01 | 1.000e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| 0.02 | 2.000e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| 0.05 | 5.000e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| 0.1 | 1.000e-01 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| 0.15 | 1.500e-01 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family a, q 50).

### q = 60: verifier quantities

| perturbation | x | G_dev | G_final | (t*, d*) final | G dev-final | eta_dev | eta_final | eta dev-final | EXP_root final | dReach final | Delta_max_all final | dFull final | valid dev/final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -0.15 | -0.15 | 3.034e-03 | 3.082e-03 | (1, 0) | -4.851e-05 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 3.082e-03 | 3.082e-03 | 3.082e-03 | 3.082e-03 | yes/yes |
| -0.1 | -0.1 | 1.440e-03 | 1.390e-03 | (1, 0) | 5.030e-05 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 1.390e-03 | 1.390e-03 | 1.390e-03 | 1.390e-03 | yes/yes |
| -0.05 | -0.05 | 3.444e-04 | 3.507e-04 | (1, 0) | -6.302e-06 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 3.507e-04 | 3.507e-04 | 3.507e-04 | 3.507e-04 | yes/yes |
| -0.02 | -0.02 | 7.794e-05 | 5.661e-05 | (1, 0) | 2.133e-05 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 5.661e-05 | 5.661e-05 | 5.661e-05 | 5.661e-05 | yes/yes |
| -0.01 | -0.01 | 1.267e-05 | 1.300e-05 | (1, 0) | -3.251e-07 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 1.300e-05 | 1.300e-05 | 1.300e-05 | 1.300e-05 | yes/yes |
| -0.005 | -0.005 | 2.582e-06 | 2.333e-06 | (1, 0) | 2.490e-07 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 2.333e-06 | 2.333e-06 | 2.333e-06 | 2.333e-06 | yes/yes |
| 0 | 0 | 2.220e-16 | 3.511e-07 | (1, 0) | -3.511e-07 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 3.511e-07 | 3.511e-07 | 3.511e-07 | 3.511e-07 | yes/yes |
| 0.005 | 0.005 | 1.565e-06 | 5.205e-06 | (1, 0) | -3.640e-06 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 5.205e-06 | 5.205e-06 | 5.205e-06 | 5.205e-06 | yes/yes |
| 0.01 | 0.01 | 6.484e-06 | 1.569e-05 | (1, 0) | -9.209e-06 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 1.569e-05 | 1.569e-05 | 1.569e-05 | 1.569e-05 | yes/yes |
| 0.02 | 0.02 | 5.875e-05 | 5.670e-05 | (1, 0) | 2.050e-06 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 5.670e-05 | 5.670e-05 | 5.670e-05 | 5.670e-05 | yes/yes |
| 0.05 | 0.05 | 3.649e-04 | 3.627e-04 | (1, 0) | 2.161e-06 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 3.627e-04 | 3.627e-04 | 3.627e-04 | 3.627e-04 | yes/yes |
| 0.1 | 0.1 | 1.486e-03 | 1.442e-03 | (1, 0) | 4.334e-05 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 1.442e-03 | 1.442e-03 | 1.442e-03 | 1.442e-03 | yes/yes |
| 0.15 | 0.15 | 3.192e-03 | 3.247e-03 | (1, 0) | -5.491e-05 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 3.247e-03 | 3.247e-03 | 3.247e-03 | 3.247e-03 | yes/yes |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family a, q 60); G = Gmax_full/DW, eta = eta_2/DW, all in units of DW; n/a = tier not run.

### q = 60: recovery metrics (tier independent) and clip

| perturbation | stage-1 error (signed) | stage-2 peak error (signed) | RMSE_pos / e2*(0) | tail mean / e2*(0) | clip binds |
|---|---|---|---|---|---|
| -0.15 | -1.500e-01 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| -0.1 | -1.000e-01 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| -0.05 | -5.000e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| -0.02 | -2.000e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| -0.01 | -1.000e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| -0.005 | -5.000e-03 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| 0 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| 0.005 | 5.000e-03 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| 0.01 | 1.000e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| 0.02 | 2.000e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| 0.05 | 5.000e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| 0.1 | 1.000e-01 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| 0.15 | 1.500e-01 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family a, q 60).

### Quadratic fit through the origin: metric ~ a x^2

| q | tier | metric | side | n | a | RMSE of residuals | max abs residual (at x) | max abs resid / max metric | R2 (uncentered) | x at threshold from fit | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | development | Gmax_full_over_dw | pooled | 12 | 2.961e-01 | 3.229e-04 | 7.869e-04 (-0.15) | 1.069e-01 | 9.885e-01 | G-F 0.01: 0.18376 |  |
| 50 | development | Gmax_full_over_dw | neg | 6 | 2.648e-01 | 8.915e-05 | 1.518e-04 (-0.1) | 2.583e-02 | 9.989e-01 | G-F 0.01: 0.19432 |  |
| 50 | development | Gmax_full_over_dw | pos | 6 | 3.275e-01 | 1.813e-05 | 3.729e-05 (0.05) | 5.066e-03 | 1.000e+00 | G-F 0.01: 0.17475 |  |
| 50 | development | eta_T_over_dw | pooled | 12 | 1.288e-14 | 1.799e-16 | 2.217e-16 (-0.005) | 9.986e-01 | 3.433e-01 | G-A 0.005: n/a | all fitted responses <= 1e-09 (numerical floor): no extrapolation; eta_2 does not depend on x in this family (stage 2 is unchanged along the perturbation): not a model |
| 50 | development | eta_T_over_dw | neg | 6 | 1.288e-14 | 1.799e-16 | 2.217e-16 (-0.005) | 9.986e-01 | 3.433e-01 | G-A 0.005: n/a | all fitted responses <= 1e-09 (numerical floor): no extrapolation; eta_2 does not depend on x in this family (stage 2 is unchanged along the perturbation): not a model |
| 50 | development | eta_T_over_dw | pos | 6 | 1.288e-14 | 1.799e-16 | 2.217e-16 (0.005) | 9.986e-01 | 3.433e-01 | G-A 0.005: n/a | all fitted responses <= 1e-09 (numerical floor): no extrapolation; eta_2 does not depend on x in this family (stage 2 is unchanged along the perturbation): not a model |
| 50 | final | Gmax_full_over_dw | pooled | 12 | 2.911e-01 | 2.770e-04 | 6.578e-04 (-0.15) | 9.174e-02 | 9.912e-01 | G-F 0.01: 0.18534 |  |
| 50 | final | Gmax_full_over_dw | neg | 6 | 2.640e-01 | 5.041e-05 | 8.513e-05 (-0.1) | 1.445e-02 | 9.996e-01 | G-F 0.01: 0.19464 |  |
| 50 | final | Gmax_full_over_dw | pos | 6 | 3.183e-01 | 9.233e-06 | 1.795e-05 (0.1) | 2.503e-03 | 1.000e+00 | G-F 0.01: 0.17725 |  |
| 50 | final | eta_T_over_dw | pooled | 12 | 1.288e-14 | 1.799e-16 | 2.217e-16 (-0.005) | 9.986e-01 | 3.433e-01 | G-A 0.005: n/a | all fitted responses <= 1e-09 (numerical floor): no extrapolation; eta_2 does not depend on x in this family (stage 2 is unchanged along the perturbation): not a model |
| 50 | final | eta_T_over_dw | neg | 6 | 1.288e-14 | 1.799e-16 | 2.217e-16 (-0.005) | 9.986e-01 | 3.433e-01 | G-A 0.005: n/a | all fitted responses <= 1e-09 (numerical floor): no extrapolation; eta_2 does not depend on x in this family (stage 2 is unchanged along the perturbation): not a model |
| 50 | final | eta_T_over_dw | pos | 6 | 1.288e-14 | 1.799e-16 | 2.217e-16 (0.005) | 9.986e-01 | 3.433e-01 | G-A 0.005: n/a | all fitted responses <= 1e-09 (numerical floor): no extrapolation; eta_2 does not depend on x in this family (stage 2 is unchanged along the perturbation): not a model |
| 60 | development | Gmax_full_over_dw | pooled | 12 | 1.397e-01 | 4.554e-05 | 1.092e-04 (-0.15) | 3.422e-02 | 9.990e-01 | G-F 0.01: 0.26755 |  |
| 60 | development | Gmax_full_over_dw | neg | 6 | 1.364e-01 | 3.558e-05 | 7.635e-05 (-0.1) | 2.517e-02 | 9.993e-01 | G-F 0.01: 0.27078 |  |
| 60 | development | Gmax_full_over_dw | pos | 6 | 1.430e-01 | 2.537e-05 | 5.559e-05 (0.1) | 1.741e-02 | 9.997e-01 | G-F 0.01: 0.26443 |  |
| 60 | development | eta_T_over_dw | pooled | 12 | 1.288e-14 | 1.799e-16 | 2.217e-16 (-0.005) | 9.986e-01 | 3.433e-01 | G-A 0.005: n/a | all fitted responses <= 1e-09 (numerical floor): no extrapolation; eta_2 does not depend on x in this family (stage 2 is unchanged along the perturbation): not a model |
| 60 | development | eta_T_over_dw | neg | 6 | 1.288e-14 | 1.799e-16 | 2.217e-16 (-0.005) | 9.986e-01 | 3.433e-01 | G-A 0.005: n/a | all fitted responses <= 1e-09 (numerical floor): no extrapolation; eta_2 does not depend on x in this family (stage 2 is unchanged along the perturbation): not a model |
| 60 | development | eta_T_over_dw | pos | 6 | 1.288e-14 | 1.799e-16 | 2.217e-16 (0.005) | 9.986e-01 | 3.433e-01 | G-A 0.005: n/a | all fitted responses <= 1e-09 (numerical floor): no extrapolation; eta_2 does not depend on x in this family (stage 2 is unchanged along the perturbation): not a model |
| 60 | final | Gmax_full_over_dw | pooled | 12 | 1.408e-01 | 3.559e-05 | 8.629e-05 (-0.15) | 2.657e-02 | 9.994e-01 | G-F 0.01: 0.26647 |  |
| 60 | final | Gmax_full_over_dw | neg | 6 | 1.374e-01 | 8.069e-06 | 1.632e-05 (-0.1) | 5.296e-03 | 1.000e+00 | G-F 0.01: 0.26982 |  |
| 60 | final | Gmax_full_over_dw | pos | 6 | 1.443e-01 | 1.261e-06 | 1.940e-06 (0.05) | 5.976e-04 | 1.000e+00 | G-F 0.01: 0.26324 |  |
| 60 | final | eta_T_over_dw | pooled | 12 | 1.288e-14 | 1.799e-16 | 2.217e-16 (-0.005) | 9.986e-01 | 3.433e-01 | G-A 0.005: n/a | all fitted responses <= 1e-09 (numerical floor): no extrapolation; eta_2 does not depend on x in this family (stage 2 is unchanged along the perturbation): not a model |
| 60 | final | eta_T_over_dw | neg | 6 | 1.288e-14 | 1.799e-16 | 2.217e-16 (-0.005) | 9.986e-01 | 3.433e-01 | G-A 0.005: n/a | all fitted responses <= 1e-09 (numerical floor): no extrapolation; eta_2 does not depend on x in this family (stage 2 is unchanged along the perturbation): not a model |
| 60 | final | eta_T_over_dw | pos | 6 | 1.288e-14 | 1.799e-16 | 2.217e-16 (0.005) | 9.986e-01 | 3.433e-01 | G-A 0.005: n/a | all fitted responses <= 1e-09 (numerical floor): no extrapolation; eta_2 does not depend on x in this family (stage 2 is unchanged along the perturbation): not a model |

Source: `results/v2_refine/d2_verifier_sensitivity/fits.csv` (family a); the full residual list is its `residuals` column (input order: ascending perturbation). Points with x = 0 carry no information in a fit through the origin and are excluded; 'x at threshold from fit' is the extrapolation sqrt(threshold / a) of the fitted law and is descriptive only.

### Detection limits

Smallest |perturbation| on the grid at which the metric exceeds the threshold (strict '>'): G-F Gmax_full/DW > 0.01, G-A eta_2/DW > 0.005, G-N |dev - final| of Gmax_full/DW > 0.001 or of eta_2/DW > 0.001 (units of DW). For the signed families (a), (b) the negative and positive sides are listed separately and 'any' is the smaller |perturbation| of the two. 'bracket lo' is the largest grid |perturbation| below the limit on that side.

| q | tier | criterion | side | limit (|p|) | reached | x at limit | value at limit | bracket lo | max |p| on grid | max value on grid |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | development | G-F | neg | not reached on the grid | no | n/a | n/a | none | 0.15 | 5.876e-03 |
| 50 | development | G-F | pos | not reached on the grid | no | n/a | n/a | none | 0.15 | 7.361e-03 |
| 50 | development | G-F | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 7.361e-03 |
| 50 | development | G-A | neg | not reached on the grid | no | n/a | n/a | none | 0.15 | 2.220e-16 |
| 50 | development | G-A | pos | not reached on the grid | no | n/a | n/a | none | 0.15 | 2.220e-16 |
| 50 | development | G-A | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 2.220e-16 |
| 50 | final | G-F | neg | not reached on the grid | no | n/a | n/a | none | 0.15 | 5.893e-03 |
| 50 | final | G-F | pos | not reached on the grid | no | n/a | n/a | none | 0.15 | 7.170e-03 |
| 50 | final | G-F | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 7.170e-03 |
| 50 | final | G-A | neg | not reached on the grid | no | n/a | n/a | none | 0.15 | 2.220e-16 |
| 50 | final | G-A | pos | not reached on the grid | no | n/a | n/a | none | 0.15 | 2.220e-16 |
| 50 | final | G-A | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 2.220e-16 |
| 50 | dev-final | G-N_Gmax | neg | not reached on the grid | no | n/a | n/a | none | 0.15 | 7.543e-05 |
| 50 | dev-final | G-N_Gmax | pos | not reached on the grid | no | n/a | n/a | none | 0.15 | 1.904e-04 |
| 50 | dev-final | G-N_Gmax | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 1.904e-04 |
| 50 | dev-final | G-N_eta | neg | not reached on the grid | no | n/a | n/a | none | 0.15 | 0.000e+00 |
| 50 | dev-final | G-N_eta | pos | not reached on the grid | no | n/a | n/a | none | 0.15 | 0.000e+00 |
| 50 | dev-final | G-N_eta | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 0.000e+00 |
| 60 | development | G-F | neg | not reached on the grid | no | n/a | n/a | none | 0.15 | 3.034e-03 |
| 60 | development | G-F | pos | not reached on the grid | no | n/a | n/a | none | 0.15 | 3.192e-03 |
| 60 | development | G-F | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 3.192e-03 |
| 60 | development | G-A | neg | not reached on the grid | no | n/a | n/a | none | 0.15 | 2.220e-16 |
| 60 | development | G-A | pos | not reached on the grid | no | n/a | n/a | none | 0.15 | 2.220e-16 |
| 60 | development | G-A | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 2.220e-16 |
| 60 | final | G-F | neg | not reached on the grid | no | n/a | n/a | none | 0.15 | 3.082e-03 |
| 60 | final | G-F | pos | not reached on the grid | no | n/a | n/a | none | 0.15 | 3.247e-03 |
| 60 | final | G-F | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 3.247e-03 |
| 60 | final | G-A | neg | not reached on the grid | no | n/a | n/a | none | 0.15 | 2.220e-16 |
| 60 | final | G-A | pos | not reached on the grid | no | n/a | n/a | none | 0.15 | 2.220e-16 |
| 60 | final | G-A | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 2.220e-16 |
| 60 | dev-final | G-N_Gmax | neg | not reached on the grid | no | n/a | n/a | none | 0.15 | 5.030e-05 |
| 60 | dev-final | G-N_Gmax | pos | not reached on the grid | no | n/a | n/a | none | 0.15 | 5.491e-05 |
| 60 | dev-final | G-N_Gmax | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 5.491e-05 |
| 60 | dev-final | G-N_eta | neg | not reached on the grid | no | n/a | n/a | none | 0.15 | 0.000e+00 |
| 60 | dev-final | G-N_eta | pos | not reached on the grid | no | n/a | n/a | none | 0.15 | 0.000e+00 |
| 60 | dev-final | G-N_eta | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 0.000e+00 |

Source: `results/v2_refine/d2_verifier_sensitivity/detection_limits.csv` (family a).

Sign handling (negative side vs positive side):

| q | tier | criterion | negative side | positive side | comparison |
|---|---|---|---|---|---|
| 50 | dev-final | G-N_Gmax | not reached on the grid | not reached on the grid | same on both sides |
| 50 | dev-final | G-N_eta | not reached on the grid | not reached on the grid | same on both sides |
| 50 | development | G-A | not reached on the grid | not reached on the grid | same on both sides |
| 50 | development | G-F | not reached on the grid | not reached on the grid | same on both sides |
| 50 | final | G-A | not reached on the grid | not reached on the grid | same on both sides |
| 50 | final | G-F | not reached on the grid | not reached on the grid | same on both sides |
| 60 | dev-final | G-N_Gmax | not reached on the grid | not reached on the grid | same on both sides |
| 60 | dev-final | G-N_eta | not reached on the grid | not reached on the grid | same on both sides |
| 60 | development | G-A | not reached on the grid | not reached on the grid | same on both sides |
| 60 | development | G-F | not reached on the grid | not reached on the grid | same on both sides |
| 60 | final | G-A | not reached on the grid | not reached on the grid | same on both sides |
| 60 | final | G-F | not reached on the grid | not reached on the grid | same on both sides |

Source: `results/v2_refine/d2_verifier_sensitivity/detection_limits.csv` (side = neg / pos); comparison is string equality of the two limits.

![Gmax_full/DW and eta_2/DW against |perturbation|, family a](../../../results/v2_refine/d2_verifier_sensitivity/figures/d2_family_a.png)

Figure: `results/v2_refine/d2_verifier_sensitivity/figures/d2_family_a.png` (and `.pdf`), log-log, both tiers, both q; data from `results/v2_refine/d2_verifier_sensitivity/paired.csv`.

## 5. Family (b) stage-2 amplitude

Perturbation parameter: `delta_stage2` (relative). Fit abscissa x: `delta_stage2`.

### q = 50: verifier quantities

| perturbation | x | G_dev | G_final | (t*, d*) final | G dev-final | eta_dev | eta_final | eta dev-final | EXP_root final | dReach final | Delta_max_all final | dFull final | valid dev/final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -0.15 | -0.15 | 1.567e-02 | 1.567e-02 | (2, -20) | 0.000e+00 | 1.567e-02 | 1.567e-02 | 0.000e+00 | 5.985e-03 | 1.567e-02 | 1.567e-02 | 1.567e-02 | yes/yes |
| -0.1 | -0.1 | 8.003e-03 | 8.003e-03 | (2, -16) | 0.000e+00 | 8.003e-03 | 8.003e-03 | 0.000e+00 | 2.858e-03 | 8.003e-03 | 8.003e-03 | 8.003e-03 | yes/yes |
| -0.05 | -0.05 | 2.337e-03 | 2.358e-03 | (2, -10) | -2.082e-05 | 2.337e-03 | 2.358e-03 | -2.082e-05 | 7.753e-04 | 2.358e-03 | 2.358e-03 | 2.358e-03 | yes/yes |
| -0.02 | -0.02 | 4.256e-04 | 4.256e-04 | (2, -4) | 0.000e+00 | 4.256e-04 | 4.256e-04 | 0.000e+00 | 1.310e-04 | 4.256e-04 | 4.256e-04 | 4.256e-04 | yes/yes |
| -0.01 | -0.01 | 1.075e-04 | 1.104e-04 | (2, -2) | -2.924e-06 | 1.075e-04 | 1.104e-04 | -2.924e-06 | 3.333e-05 | 1.104e-04 | 1.104e-04 | 1.104e-04 | yes/yes |
| -0.005 | -0.005 | 2.688e-05 | 2.801e-05 | (2, -2) | -1.132e-06 | 2.688e-05 | 2.801e-05 | -1.132e-06 | 8.341e-06 | 2.801e-05 | 2.801e-05 | 2.801e-05 | yes/yes |
| 0 | 0 | 2.220e-16 | 2.220e-16 | (2, -24) | 0.000e+00 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 0.000e+00 | 2.220e-16 | 2.220e-16 | 2.220e-16 | yes/yes |
| 0.005 | 0.005 | 2.917e-05 | 2.917e-05 | (2, 0) | 0.000e+00 | 2.917e-05 | 2.917e-05 | 0.000e+00 | 8.818e-06 | 2.917e-05 | 2.917e-05 | 2.917e-05 | yes/yes |
| 0.01 | 0.01 | 1.167e-04 | 1.167e-04 | (2, 0) | -1.000e-16 | 1.167e-04 | 1.167e-04 | -1.000e-16 | 3.527e-05 | 1.167e-04 | 1.167e-04 | 1.167e-04 | yes/yes |
| 0.02 | 0.02 | 4.667e-04 | 4.667e-04 | (2, 0) | -1.000e-16 | 4.667e-04 | 4.667e-04 | -1.000e-16 | 1.411e-04 | 4.667e-04 | 4.667e-04 | 4.667e-04 | yes/yes |
| 0.05 | 0.05 | 2.917e-03 | 2.917e-03 | (2, 0) | 0.000e+00 | 2.917e-03 | 2.917e-03 | 0.000e+00 | 8.825e-04 | 2.917e-03 | 2.917e-03 | 2.917e-03 | yes/yes |
| 0.1 | 0.1 | 1.167e-02 | 1.167e-02 | (2, 0) | 1.006e-16 | 1.167e-02 | 1.167e-02 | 1.006e-16 | 3.588e-03 | 1.167e-02 | 1.167e-02 | 1.167e-02 | yes/yes |
| 0.15 | 0.15 | 2.625e-02 | 2.625e-02 | (2, 0) | 0.000e+00 | 2.625e-02 | 2.625e-02 | 0.000e+00 | 8.204e-03 | 2.625e-02 | 2.625e-02 | 2.625e-02 | yes/yes |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family b, q 50); G = Gmax_full/DW, eta = eta_2/DW, all in units of DW; n/a = tier not run.

### q = 50: recovery metrics (tier independent) and clip

| perturbation | stage-1 error (signed) | stage-2 peak error (signed) | RMSE_pos / e2*(0) | tail mean / e2*(0) | clip binds |
|---|---|---|---|---|---|
| -0.15 | 0.000e+00 | -1.500e-01 | 8.671e-02 | 0.000e+00 | no |
| -0.1 | 0.000e+00 | -1.000e-01 | 5.781e-02 | 0.000e+00 | no |
| -0.05 | 0.000e+00 | -5.000e-02 | 2.890e-02 | 0.000e+00 | no |
| -0.02 | 0.000e+00 | -2.000e-02 | 1.156e-02 | 0.000e+00 | no |
| -0.01 | 0.000e+00 | -1.000e-02 | 5.781e-03 | 0.000e+00 | no |
| -0.005 | 0.000e+00 | -5.000e-03 | 2.890e-03 | 0.000e+00 | no |
| 0 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| 0.005 | 0.000e+00 | 5.000e-03 | 2.890e-03 | 0.000e+00 | no |
| 0.01 | 0.000e+00 | 1.000e-02 | 5.781e-03 | 0.000e+00 | no |
| 0.02 | 0.000e+00 | 2.000e-02 | 1.156e-02 | 0.000e+00 | no |
| 0.05 | 0.000e+00 | 5.000e-02 | 2.890e-02 | 0.000e+00 | no |
| 0.1 | 0.000e+00 | 1.000e-01 | 5.781e-02 | 0.000e+00 | no |
| 0.15 | 0.000e+00 | 1.500e-01 | 8.671e-02 | 0.000e+00 | no |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family b, q 50).

### q = 60: verifier quantities

| perturbation | x | G_dev | G_final | (t*, d*) final | G dev-final | eta_dev | eta_final | eta dev-final | EXP_root final | dReach final | Delta_max_all final | dFull final | valid dev/final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -0.15 | -0.15 | 8.355e-03 | 8.355e-03 | (2, -12) | 0.000e+00 | 8.355e-03 | 8.355e-03 | 0.000e+00 | 3.301e-03 | 8.355e-03 | 8.355e-03 | 8.355e-03 | yes/yes |
| -0.1 | -0.1 | 3.958e-03 | 3.970e-03 | (2, -10) | -1.187e-05 | 3.958e-03 | 3.970e-03 | -1.187e-05 | 1.504e-03 | 3.971e-03 | 3.970e-03 | 3.971e-03 | yes/yes |
| -0.05 | -0.05 | 1.052e-03 | 1.067e-03 | (2, -6) | -1.555e-05 | 1.052e-03 | 1.067e-03 | -1.555e-05 | 3.864e-04 | 1.068e-03 | 1.067e-03 | 1.068e-03 | yes/yes |
| -0.02 | -0.02 | 1.768e-04 | 1.815e-04 | (2, -2) | -4.762e-06 | 1.768e-04 | 1.815e-04 | -4.762e-06 | 6.302e-05 | 1.819e-04 | 1.815e-04 | 1.819e-04 | yes/yes |
| -0.01 | -0.01 | 4.420e-05 | 4.573e-05 | (2, -2) | -1.537e-06 | 4.420e-05 | 4.573e-05 | -1.537e-06 | 1.602e-05 | 4.608e-05 | 4.573e-05 | 4.608e-05 | yes/yes |
| -0.005 | -0.005 | 1.105e-05 | 1.143e-05 | (2, -2) | -3.843e-07 | 1.105e-05 | 1.143e-05 | -3.843e-07 | 4.269e-06 | 1.178e-05 | 1.143e-05 | 1.178e-05 | yes/yes |
| 0 | 0 | 2.220e-16 | 3.511e-07 | (1, 0) | -3.511e-07 | 2.220e-16 | 2.220e-16 | 0.000e+00 | 3.511e-07 | 3.511e-07 | 3.511e-07 | 3.511e-07 | yes/yes |
| 0.005 | 0.005 | 1.170e-05 | 1.182e-05 | (2, 0) | -1.211e-07 | 1.170e-05 | 1.182e-05 | -1.211e-07 | 4.392e-06 | 1.218e-05 | 1.182e-05 | 1.218e-05 | yes/yes |
| 0.01 | 0.01 | 4.730e-05 | 4.730e-05 | (2, 0) | -1.239e-10 | 4.730e-05 | 4.730e-05 | -1.239e-10 | 1.651e-05 | 4.765e-05 | 4.730e-05 | 4.765e-05 | yes/yes |
| 0.02 | 0.02 | 1.892e-04 | 1.892e-04 | (2, 0) | 3.000e-16 | 1.892e-04 | 1.892e-04 | 3.000e-16 | 6.499e-05 | 1.896e-04 | 1.892e-04 | 1.896e-04 | yes/yes |
| 0.05 | 0.05 | 1.182e-03 | 1.182e-03 | (2, 0) | 0.000e+00 | 1.182e-03 | 1.182e-03 | 0.000e+00 | 4.042e-04 | 1.183e-03 | 1.182e-03 | 1.183e-03 | yes/yes |
| 0.1 | 0.1 | 4.730e-03 | 4.730e-03 | (2, 0) | 0.000e+00 | 4.730e-03 | 4.730e-03 | 0.000e+00 | 1.627e-03 | 4.730e-03 | 4.730e-03 | 4.730e-03 | yes/yes |
| 0.15 | 0.15 | 1.064e-02 | 1.064e-02 | (2, 0) | 0.000e+00 | 1.064e-02 | 1.064e-02 | 0.000e+00 | 3.696e-03 | 1.064e-02 | 1.064e-02 | 1.064e-02 | yes/yes |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family b, q 60); G = Gmax_full/DW, eta = eta_2/DW, all in units of DW; n/a = tier not run.

### q = 60: recovery metrics (tier independent) and clip

| perturbation | stage-1 error (signed) | stage-2 peak error (signed) | RMSE_pos / e2*(0) | tail mean / e2*(0) | clip binds |
|---|---|---|---|---|---|
| -0.15 | 0.000e+00 | -1.500e-01 | 8.669e-02 | 0.000e+00 | no |
| -0.1 | 0.000e+00 | -1.000e-01 | 5.780e-02 | 0.000e+00 | no |
| -0.05 | 0.000e+00 | -5.000e-02 | 2.890e-02 | 0.000e+00 | no |
| -0.02 | 0.000e+00 | -2.000e-02 | 1.156e-02 | 0.000e+00 | no |
| -0.01 | 0.000e+00 | -1.000e-02 | 5.780e-03 | 0.000e+00 | no |
| -0.005 | 0.000e+00 | -5.000e-03 | 2.890e-03 | 0.000e+00 | no |
| 0 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | no |
| 0.005 | 0.000e+00 | 5.000e-03 | 2.890e-03 | 0.000e+00 | no |
| 0.01 | 0.000e+00 | 1.000e-02 | 5.780e-03 | 0.000e+00 | no |
| 0.02 | 0.000e+00 | 2.000e-02 | 1.156e-02 | 0.000e+00 | no |
| 0.05 | 0.000e+00 | 5.000e-02 | 2.890e-02 | 0.000e+00 | no |
| 0.1 | 0.000e+00 | 1.000e-01 | 5.780e-02 | 0.000e+00 | no |
| 0.15 | 0.000e+00 | 1.500e-01 | 8.669e-02 | 0.000e+00 | no |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family b, q 60).

### Quadratic fit through the origin: metric ~ a x^2

| q | tier | metric | side | n | a | RMSE of residuals | max abs residual (at x) | max abs resid / max metric | R2 (uncentered) | x at threshold from fit | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | development | Gmax_full_over_dw | pooled | 12 | 9.413e-01 | 2.300e-03 | 5.509e-03 (-0.15) | 2.099e-01 | 9.448e-01 | G-F 0.01: 0.10307 |  |
| 50 | development | Gmax_full_over_dw | neg | 6 | 7.160e-01 | 4.517e-04 | 8.437e-04 (-0.1) | 5.384e-02 | 9.961e-01 | G-F 0.01: 0.11818 |  |
| 50 | development | Gmax_full_over_dw | pos | 6 | 1.167e+00 | 5.807e-17 | 8.014e-17 (0.005) | 3.053e-15 | 1.000e+00 | G-F 0.01: 0.092582 |  |
| 50 | development | eta_T_over_dw | pooled | 12 | 9.413e-01 | 2.300e-03 | 5.509e-03 (-0.15) | 2.099e-01 | 9.448e-01 | G-A 0.005: 0.072882 |  |
| 50 | development | eta_T_over_dw | neg | 6 | 7.160e-01 | 4.517e-04 | 8.437e-04 (-0.1) | 5.384e-02 | 9.961e-01 | G-A 0.005: 0.083568 |  |
| 50 | development | eta_T_over_dw | pos | 6 | 1.167e+00 | 5.807e-17 | 8.014e-17 (0.005) | 3.053e-15 | 1.000e+00 | G-A 0.005: 0.065465 |  |
| 50 | final | Gmax_full_over_dw | pooled | 12 | 9.414e-01 | 2.300e-03 | 5.510e-03 (-0.15) | 2.099e-01 | 9.448e-01 | G-F 0.01: 0.10307 |  |
| 50 | final | Gmax_full_over_dw | neg | 6 | 7.160e-01 | 4.560e-04 | 8.429e-04 (-0.1) | 5.379e-02 | 9.960e-01 | G-F 0.01: 0.11818 |  |
| 50 | final | Gmax_full_over_dw | pos | 6 | 1.167e+00 | 7.005e-17 | 1.354e-16 (0.02) | 5.157e-15 | 1.000e+00 | G-F 0.01: 0.092582 |  |
| 50 | final | eta_T_over_dw | pooled | 12 | 9.414e-01 | 2.300e-03 | 5.510e-03 (-0.15) | 2.099e-01 | 9.448e-01 | G-A 0.005: 0.07288 |  |
| 50 | final | eta_T_over_dw | neg | 6 | 7.160e-01 | 4.560e-04 | 8.429e-04 (-0.1) | 5.379e-02 | 9.960e-01 | G-A 0.005: 0.083563 |  |
| 50 | final | eta_T_over_dw | pos | 6 | 1.167e+00 | 7.005e-17 | 1.354e-16 (0.02) | 5.157e-15 | 1.000e+00 | G-A 0.005: 0.065465 |  |
| 60 | development | Gmax_full_over_dw | pooled | 12 | 4.244e-01 | 4.961e-04 | 1.195e-03 (-0.15) | 1.123e-01 | 9.868e-01 | G-F 0.01: 0.1535 |  |
| 60 | development | Gmax_full_over_dw | neg | 6 | 3.758e-01 | 1.030e-04 | 2.001e-04 (-0.1) | 2.395e-02 | 9.993e-01 | G-F 0.01: 0.16312 |  |
| 60 | development | Gmax_full_over_dw | pos | 6 | 4.730e-01 | 4.943e-08 | 1.211e-07 (0.005) | 1.138e-05 | 1.000e+00 | G-F 0.01: 0.14541 |  |
| 60 | development | eta_T_over_dw | pooled | 12 | 4.244e-01 | 4.961e-04 | 1.195e-03 (-0.15) | 1.123e-01 | 9.868e-01 | G-A 0.005: 0.10854 |  |
| 60 | development | eta_T_over_dw | neg | 6 | 3.758e-01 | 1.030e-04 | 2.001e-04 (-0.1) | 2.395e-02 | 9.993e-01 | G-A 0.005: 0.11534 |  |
| 60 | development | eta_T_over_dw | pos | 6 | 4.730e-01 | 4.943e-08 | 1.211e-07 (0.005) | 1.138e-05 | 1.000e+00 | G-A 0.005: 0.10282 |  |
| 60 | final | Gmax_full_over_dw | pooled | 12 | 4.245e-01 | 4.956e-04 | 1.198e-03 (-0.15) | 1.125e-01 | 9.868e-01 | G-F 0.01: 0.15348 |  |
| 60 | final | Gmax_full_over_dw | neg | 6 | 3.761e-01 | 1.100e-04 | 2.093e-04 (-0.1) | 2.505e-02 | 9.992e-01 | G-F 0.01: 0.16306 |  |
| 60 | final | Gmax_full_over_dw | pos | 6 | 4.730e-01 | 1.146e-16 | 1.900e-16 (0.02) | 1.785e-14 | 1.000e+00 | G-F 0.01: 0.14541 |  |
| 60 | final | eta_T_over_dw | pooled | 12 | 4.245e-01 | 4.956e-04 | 1.198e-03 (-0.15) | 1.125e-01 | 9.868e-01 | G-A 0.005: 0.10852 |  |
| 60 | final | eta_T_over_dw | neg | 6 | 3.761e-01 | 1.100e-04 | 2.093e-04 (-0.1) | 2.505e-02 | 9.992e-01 | G-A 0.005: 0.1153 |  |
| 60 | final | eta_T_over_dw | pos | 6 | 4.730e-01 | 1.146e-16 | 1.900e-16 (0.02) | 1.785e-14 | 1.000e+00 | G-A 0.005: 0.10282 |  |

Source: `results/v2_refine/d2_verifier_sensitivity/fits.csv` (family b); the full residual list is its `residuals` column (input order: ascending perturbation). Points with x = 0 carry no information in a fit through the origin and are excluded; 'x at threshold from fit' is the extrapolation sqrt(threshold / a) of the fitted law and is descriptive only.

### Detection limits

Smallest |perturbation| on the grid at which the metric exceeds the threshold (strict '>'): G-F Gmax_full/DW > 0.01, G-A eta_2/DW > 0.005, G-N |dev - final| of Gmax_full/DW > 0.001 or of eta_2/DW > 0.001 (units of DW). For the signed families (a), (b) the negative and positive sides are listed separately and 'any' is the smaller |perturbation| of the two. 'bracket lo' is the largest grid |perturbation| below the limit on that side.

| q | tier | criterion | side | limit (|p|) | reached | x at limit | value at limit | bracket lo | max |p| on grid | max value on grid |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | development | G-F | neg | 0.15 | yes | -0.15 | 1.567e-02 | 0.1 | 0.15 | 1.567e-02 |
| 50 | development | G-F | pos | 0.1 | yes | 0.1 | 1.167e-02 | 0.05 | 0.15 | 2.625e-02 |
| 50 | development | G-F | any | 0.1 | yes | 0.1 | 1.167e-02 | 0.05 | 0.15 | 2.625e-02 |
| 50 | development | G-A | neg | 0.1 | yes | -0.1 | 8.003e-03 | 0.05 | 0.15 | 1.567e-02 |
| 50 | development | G-A | pos | 0.1 | yes | 0.1 | 1.167e-02 | 0.05 | 0.15 | 2.625e-02 |
| 50 | development | G-A | any | 0.1 | yes | -0.1 | 8.003e-03 | 0.05 | 0.15 | 2.625e-02 |
| 50 | final | G-F | neg | 0.15 | yes | -0.15 | 1.567e-02 | 0.1 | 0.15 | 1.567e-02 |
| 50 | final | G-F | pos | 0.1 | yes | 0.1 | 1.167e-02 | 0.05 | 0.15 | 2.625e-02 |
| 50 | final | G-F | any | 0.1 | yes | 0.1 | 1.167e-02 | 0.05 | 0.15 | 2.625e-02 |
| 50 | final | G-A | neg | 0.1 | yes | -0.1 | 8.003e-03 | 0.05 | 0.15 | 1.567e-02 |
| 50 | final | G-A | pos | 0.1 | yes | 0.1 | 1.167e-02 | 0.05 | 0.15 | 2.625e-02 |
| 50 | final | G-A | any | 0.1 | yes | -0.1 | 8.003e-03 | 0.05 | 0.15 | 2.625e-02 |
| 50 | dev-final | G-N_Gmax | neg | not reached on the grid | no | n/a | n/a | none | 0.15 | 2.082e-05 |
| 50 | dev-final | G-N_Gmax | pos | not reached on the grid | no | n/a | n/a | none | 0.15 | 1.006e-16 |
| 50 | dev-final | G-N_Gmax | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 2.082e-05 |
| 50 | dev-final | G-N_eta | neg | not reached on the grid | no | n/a | n/a | none | 0.15 | 2.082e-05 |
| 50 | dev-final | G-N_eta | pos | not reached on the grid | no | n/a | n/a | none | 0.15 | 1.006e-16 |
| 50 | dev-final | G-N_eta | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 2.082e-05 |
| 60 | development | G-F | neg | not reached on the grid | no | n/a | n/a | none | 0.15 | 8.355e-03 |
| 60 | development | G-F | pos | 0.15 | yes | 0.15 | 1.064e-02 | 0.1 | 0.15 | 1.064e-02 |
| 60 | development | G-F | any | 0.15 | yes | 0.15 | 1.064e-02 | 0.1 | 0.15 | 1.064e-02 |
| 60 | development | G-A | neg | 0.15 | yes | -0.15 | 8.355e-03 | 0.1 | 0.15 | 8.355e-03 |
| 60 | development | G-A | pos | 0.15 | yes | 0.15 | 1.064e-02 | 0.1 | 0.15 | 1.064e-02 |
| 60 | development | G-A | any | 0.15 | yes | -0.15 | 8.355e-03 | 0.1 | 0.15 | 1.064e-02 |
| 60 | final | G-F | neg | not reached on the grid | no | n/a | n/a | none | 0.15 | 8.355e-03 |
| 60 | final | G-F | pos | 0.15 | yes | 0.15 | 1.064e-02 | 0.1 | 0.15 | 1.064e-02 |
| 60 | final | G-F | any | 0.15 | yes | 0.15 | 1.064e-02 | 0.1 | 0.15 | 1.064e-02 |
| 60 | final | G-A | neg | 0.15 | yes | -0.15 | 8.355e-03 | 0.1 | 0.15 | 8.355e-03 |
| 60 | final | G-A | pos | 0.15 | yes | 0.15 | 1.064e-02 | 0.1 | 0.15 | 1.064e-02 |
| 60 | final | G-A | any | 0.15 | yes | -0.15 | 8.355e-03 | 0.1 | 0.15 | 1.064e-02 |
| 60 | dev-final | G-N_Gmax | neg | not reached on the grid | no | n/a | n/a | none | 0.15 | 1.555e-05 |
| 60 | dev-final | G-N_Gmax | pos | not reached on the grid | no | n/a | n/a | none | 0.15 | 3.511e-07 |
| 60 | dev-final | G-N_Gmax | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 1.555e-05 |
| 60 | dev-final | G-N_eta | neg | not reached on the grid | no | n/a | n/a | none | 0.15 | 1.555e-05 |
| 60 | dev-final | G-N_eta | pos | not reached on the grid | no | n/a | n/a | none | 0.15 | 1.211e-07 |
| 60 | dev-final | G-N_eta | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 1.555e-05 |

Source: `results/v2_refine/d2_verifier_sensitivity/detection_limits.csv` (family b).

Sign handling (negative side vs positive side):

| q | tier | criterion | negative side | positive side | comparison |
|---|---|---|---|---|---|
| 50 | dev-final | G-N_Gmax | not reached on the grid | not reached on the grid | same on both sides |
| 50 | dev-final | G-N_eta | not reached on the grid | not reached on the grid | same on both sides |
| 50 | development | G-A | 0.1 | 0.1 | same on both sides |
| 50 | development | G-F | 0.15 | 0.1 | differs between sides |
| 50 | final | G-A | 0.1 | 0.1 | same on both sides |
| 50 | final | G-F | 0.15 | 0.1 | differs between sides |
| 60 | dev-final | G-N_Gmax | not reached on the grid | not reached on the grid | same on both sides |
| 60 | dev-final | G-N_eta | not reached on the grid | not reached on the grid | same on both sides |
| 60 | development | G-A | 0.15 | 0.15 | same on both sides |
| 60 | development | G-F | not reached on the grid | 0.15 | differs between sides |
| 60 | final | G-A | 0.15 | 0.15 | same on both sides |
| 60 | final | G-F | not reached on the grid | 0.15 | differs between sides |

Source: `results/v2_refine/d2_verifier_sensitivity/detection_limits.csv` (side = neg / pos); comparison is string equality of the two limits.

![Gmax_full/DW and eta_2/DW against |perturbation|, family b](../../../results/v2_refine/d2_verifier_sensitivity/figures/d2_family_b.png)

Figure: `results/v2_refine/d2_verifier_sensitivity/figures/d2_family_b.png` (and `.pdf`), log-log, both tiers, both q; data from `results/v2_refine/d2_verifier_sensitivity/paired.csv`.

## 6. Family (c) stage-2 peak rounding (uniform kernel)

Perturbation parameter: `kernel_h` (d units (half-width)). Fit abscissa x: `induced_peak_error`.

### q = 50: verifier quantities

| perturbation | x | G_dev | G_final | (t*, d*) final | G dev-final | eta_dev | eta_final | eta dev-final | EXP_root final | dReach final | Delta_max_all final | dFull final | valid dev/final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | -0.01 | 2.048e-05 | 2.058e-05 | (2, 0) | -1.064e-07 | 2.048e-05 | 2.058e-05 | -1.064e-07 | 4.092e-07 | 2.058e-05 | 2.058e-05 | 2.058e-05 | yes/yes |
| 5 | -0.025 | 1.287e-04 | 1.287e-04 | (2, 0) | -4.123e-09 | 1.287e-04 | 1.287e-04 | -4.123e-09 | 4.851e-06 | 1.287e-04 | 1.287e-04 | 1.287e-04 | yes/yes |
| 10 | -0.05 | 5.147e-04 | 6.673e-04 | (2, -2) | -1.526e-04 | 5.147e-04 | 6.673e-04 | -1.526e-04 | 3.829e-05 | 6.673e-04 | 6.673e-04 | 6.673e-04 | yes/yes |
| 12 | -0.06 | 7.423e-04 | 9.731e-04 | (2, -2) | -2.308e-04 | 7.423e-04 | 9.731e-04 | -2.308e-04 | 6.552e-05 | 9.731e-04 | 9.731e-04 | 9.731e-04 | yes/yes |
| 20 | -0.1 | 2.669e-03 | 2.669e-03 | (2, -4) | -1.002e-16 | 2.669e-03 | 2.669e-03 | -1.002e-16 | 2.982e-04 | 2.669e-03 | 2.669e-03 | 2.669e-03 | yes/yes |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family c, q 50); G = Gmax_full/DW, eta = eta_2/DW, all in units of DW; n/a = tier not run.

### q = 50: recovery metrics (tier independent) and clip

| perturbation | stage-1 error (signed) | stage-2 peak error (signed) | RMSE_pos / e2*(0) | tail mean / e2*(0) | clip binds |
|---|---|---|---|---|---|
| 2 | 0.000e+00 | -1.000e-02 | 7.003e-04 | 4.664e-05 | no |
| 5 | 0.000e+00 | -2.500e-02 | 2.752e-03 | 2.394e-04 | no |
| 10 | 0.000e+00 | -5.000e-02 | 7.833e-03 | 8.924e-04 | no |
| 12 | 0.000e+00 | -6.000e-02 | 1.031e-02 | 1.270e-03 | no |
| 20 | 0.000e+00 | -1.000e-01 | 2.226e-02 | 3.442e-03 | no |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family c, q 50).

### q = 60: verifier quantities

| perturbation | x | G_dev | G_final | (t*, d*) final | G dev-final | eta_dev | eta_final | eta dev-final | EXP_root final | dReach final | Delta_max_all final | dFull final | valid dev/final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | -0.0083333 | 1.030e-05 | 1.134e-05 | (2, 0) | -1.039e-06 | 1.030e-05 | 1.134e-05 | -1.039e-06 | 1.773e-07 | 1.134e-05 | 1.134e-05 | 1.134e-05 | yes/yes |
| 5 | -0.020833 | 7.099e-05 | 7.099e-05 | (2, 0) | -7.087e-10 | 7.099e-05 | 7.099e-05 | -7.087e-10 | 3.709e-06 | 7.298e-05 | 7.099e-05 | 7.298e-05 | yes/yes |
| 10 | -0.041667 | 2.839e-04 | 3.110e-04 | (2, -2) | -2.705e-05 | 2.839e-04 | 3.110e-04 | -2.705e-05 | 1.450e-05 | 3.112e-04 | 3.110e-04 | 3.112e-04 | yes/yes |
| 12 | -0.05 | 4.089e-04 | 4.797e-04 | (2, -2) | -7.084e-05 | 4.089e-04 | 4.797e-04 | -7.084e-05 | 2.498e-05 | 4.799e-04 | 4.797e-04 | 4.799e-04 | yes/yes |
| 20 | -0.083333 | 1.244e-03 | 1.385e-03 | (2, -2) | -1.411e-04 | 1.244e-03 | 1.385e-03 | -1.411e-04 | 1.155e-04 | 1.386e-03 | 1.385e-03 | 1.386e-03 | yes/yes |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family c, q 60); G = Gmax_full/DW, eta = eta_2/DW, all in units of DW; n/a = tier not run.

### q = 60: recovery metrics (tier independent) and clip

| perturbation | stage-1 error (signed) | stage-2 peak error (signed) | RMSE_pos / e2*(0) | tail mean / e2*(0) | clip binds |
|---|---|---|---|---|---|
| 2 | 0.000e+00 | -8.333e-03 | 5.327e-04 | 3.887e-05 | no |
| 5 | 0.000e+00 | -2.083e-02 | 2.093e-03 | 1.995e-04 | no |
| 10 | 0.000e+00 | -4.167e-02 | 5.957e-03 | 7.437e-04 | no |
| 12 | 0.000e+00 | -5.000e-02 | 7.843e-03 | 1.058e-03 | no |
| 20 | 0.000e+00 | -8.333e-02 | 1.693e-02 | 2.868e-03 | no |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family c, q 60).

### Quadratic fit through the origin: metric ~ a x^2

| q | tier | metric | side | n | a | RMSE of residuals | max abs residual (at x) | max abs resid / max metric | R2 (uncentered) | x at threshold from fit | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | development | Gmax_full_over_dw | all | 5 | 2.569e-01 | 1.101e-04 | 1.827e-04 (-0.06) | 6.844e-02 | 9.924e-01 | G-F 0.01: 0.19728 |  |
| 50 | development | eta_T_over_dw | all | 5 | 2.569e-01 | 1.101e-04 | 1.827e-04 (-0.06) | 6.844e-02 | 9.924e-01 | G-A 0.005: 0.1395 |  |
| 50 | final | Gmax_full_over_dw | all | 5 | 2.671e-01 | 1.810e-05 | 3.825e-05 (-0.025) | 1.433e-02 | 9.998e-01 | G-F 0.01: 0.1935 |  |
| 50 | final | eta_T_over_dw | all | 5 | 2.671e-01 | 1.810e-05 | 3.825e-05 (-0.025) | 1.433e-02 | 9.998e-01 | G-A 0.005: 0.13682 |  |
| 60 | development | Gmax_full_over_dw | all | 5 | 1.766e-01 | 1.962e-05 | 3.257e-05 (-0.05) | 2.619e-02 | 9.989e-01 | G-F 0.01: 0.23797 |  |
| 60 | development | eta_T_over_dw | all | 5 | 1.766e-01 | 1.962e-05 | 3.257e-05 (-0.05) | 2.619e-02 | 9.989e-01 | G-A 0.005: 0.16827 |  |
| 60 | final | Gmax_full_over_dw | all | 5 | 1.975e-01 | 1.801e-05 | 3.181e-05 (-0.041667) | 2.297e-02 | 9.993e-01 | G-F 0.01: 0.22504 |  |
| 60 | final | eta_T_over_dw | all | 5 | 1.975e-01 | 1.801e-05 | 3.181e-05 (-0.041667) | 2.297e-02 | 9.993e-01 | G-A 0.005: 0.15913 |  |

Source: `results/v2_refine/d2_verifier_sensitivity/fits.csv` (family c); the full residual list is its `residuals` column (input order: ascending perturbation). Points with x = 0 carry no information in a fit through the origin and are excluded; 'x at threshold from fit' is the extrapolation sqrt(threshold / a) of the fitted law and is descriptive only.

### Detection limits

Smallest |perturbation| on the grid at which the metric exceeds the threshold (strict '>'): G-F Gmax_full/DW > 0.01, G-A eta_2/DW > 0.005, G-N |dev - final| of Gmax_full/DW > 0.001 or of eta_2/DW > 0.001 (units of DW). For the signed families (a), (b) the negative and positive sides are listed separately and 'any' is the smaller |perturbation| of the two. 'bracket lo' is the largest grid |perturbation| below the limit on that side.

| q | tier | criterion | side | limit (|p|) | reached | x at limit | value at limit | bracket lo | max |p| on grid | max value on grid |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | development | G-F | any | not reached on the grid | no | n/a | n/a | none | 20 | 2.669e-03 |
| 50 | development | G-A | any | not reached on the grid | no | n/a | n/a | none | 20 | 2.669e-03 |
| 50 | final | G-F | any | not reached on the grid | no | n/a | n/a | none | 20 | 2.669e-03 |
| 50 | final | G-A | any | not reached on the grid | no | n/a | n/a | none | 20 | 2.669e-03 |
| 50 | dev-final | G-N_Gmax | any | not reached on the grid | no | n/a | n/a | none | 20 | 2.308e-04 |
| 50 | dev-final | G-N_eta | any | not reached on the grid | no | n/a | n/a | none | 20 | 2.308e-04 |
| 60 | development | G-F | any | not reached on the grid | no | n/a | n/a | none | 20 | 1.244e-03 |
| 60 | development | G-A | any | not reached on the grid | no | n/a | n/a | none | 20 | 1.244e-03 |
| 60 | final | G-F | any | not reached on the grid | no | n/a | n/a | none | 20 | 1.385e-03 |
| 60 | final | G-A | any | not reached on the grid | no | n/a | n/a | none | 20 | 1.385e-03 |
| 60 | dev-final | G-N_Gmax | any | not reached on the grid | no | n/a | n/a | none | 20 | 1.411e-04 |
| 60 | dev-final | G-N_eta | any | not reached on the grid | no | n/a | n/a | none | 20 | 1.411e-04 |

Source: `results/v2_refine/d2_verifier_sensitivity/detection_limits.csv` (family c).

![Gmax_full/DW and eta_2/DW against |perturbation|, family c](../../../results/v2_refine/d2_verifier_sensitivity/figures/d2_family_c.png)

Figure: `results/v2_refine/d2_verifier_sensitivity/figures/d2_family_c.png` (and `.pdf`), log-log, both tiers, both q; data from `results/v2_refine/d2_verifier_sensitivity/paired.csv`.

## 7. Family (d) stage-2 tail offset

Perturbation parameter: `tau` (effort units). Fit abscissa x: `tau`.

### q = 50: verifier quantities

| perturbation | x | G_dev | G_final | (t*, d*) final | G dev-final | eta_dev | eta_final | eta dev-final | EXP_root final | dReach final | Delta_max_all final | dFull final | valid dev/final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.25 | 4.464e-06 | 4.464e-06 | (2, -200) | 0.000e+00 | 4.464e-06 | 4.464e-06 | 0.000e+00 | 3.981e-10 | 4.464e-06 | 4.464e-06 | 4.464e-06 | yes/yes |
| 0.5 | 0.5 | 1.786e-05 | 1.786e-05 | (2, -200) | 0.000e+00 | 1.786e-05 | 1.786e-05 | 0.000e+00 | 1.592e-09 | 1.786e-05 | 1.786e-05 | 1.786e-05 | yes/yes |
| 1 | 1 | 7.143e-05 | 7.143e-05 | (2, -200) | 0.000e+00 | 7.143e-05 | 7.143e-05 | 0.000e+00 | 7.781e-09 | 7.143e-05 | 7.143e-05 | 7.143e-05 | yes/yes |
| 2 | 2 | 2.857e-04 | 2.857e-04 | (2, -200) | 0.000e+00 | 2.857e-04 | 2.857e-04 | 0.000e+00 | 3.112e-08 | 2.857e-04 | 2.857e-04 | 2.857e-04 | yes/yes |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family d, q 50); G = Gmax_full/DW, eta = eta_2/DW, all in units of DW; n/a = tier not run.

### q = 50: recovery metrics (tier independent) and clip

| perturbation | stage-1 error (signed) | stage-2 peak error (signed) | RMSE_pos / e2*(0) | tail mean / e2*(0) | clip binds |
|---|---|---|---|---|---|
| 0.25 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 3.571e-03 | no |
| 0.5 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 7.143e-03 | no |
| 1 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.429e-02 | no |
| 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 2.857e-02 | no |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family d, q 50).

### q = 60: verifier quantities

| perturbation | x | G_dev | G_final | (t*, d*) final | G dev-final | eta_dev | eta_final | eta dev-final | EXP_root final | dReach final | Delta_max_all final | dFull final | valid dev/final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.25 | 4.464e-06 | 4.464e-06 | (2, -220) | 0.000e+00 | 4.464e-06 | 4.464e-06 | 0.000e+00 | 3.513e-07 | 4.815e-06 | 4.464e-06 | 4.815e-06 | yes/yes |
| 0.5 | 0.5 | 1.786e-05 | 1.786e-05 | (2, -220) | 0.000e+00 | 1.786e-05 | 1.786e-05 | 0.000e+00 | 3.521e-07 | 1.821e-05 | 1.786e-05 | 1.821e-05 | yes/yes |
| 1 | 1 | 7.143e-05 | 7.143e-05 | (2, -220) | 0.000e+00 | 7.143e-05 | 7.143e-05 | 0.000e+00 | 3.556e-07 | 7.178e-05 | 7.143e-05 | 7.178e-05 | yes/yes |
| 2 | 2 | 2.857e-04 | 2.857e-04 | (2, -220) | 0.000e+00 | 2.857e-04 | 2.857e-04 | 0.000e+00 | 3.693e-07 | 2.861e-04 | 2.857e-04 | 2.861e-04 | yes/yes |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family d, q 60); G = Gmax_full/DW, eta = eta_2/DW, all in units of DW; n/a = tier not run.

### q = 60: recovery metrics (tier independent) and clip

| perturbation | stage-1 error (signed) | stage-2 peak error (signed) | RMSE_pos / e2*(0) | tail mean / e2*(0) | clip binds |
|---|---|---|---|---|---|
| 0.25 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 4.286e-03 | no |
| 0.5 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 8.571e-03 | no |
| 1 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.714e-02 | no |
| 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 3.429e-02 | no |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family d, q 60).

### Quadratic fit through the origin: metric ~ a x^2

| q | tier | metric | side | n | a | RMSE of residuals | max abs residual (at x) | max abs resid / max metric | R2 (uncentered) | x at threshold from fit | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | development | Gmax_full_over_dw | all | 4 | 7.143e-05 | 1.091e-17 | 1.700e-17 (0.25) | 5.950e-14 | 1.000e+00 | G-F 0.01: 11.832 |  |
| 50 | development | eta_T_over_dw | all | 4 | 7.143e-05 | 1.091e-17 | 1.700e-17 (0.25) | 5.950e-14 | 1.000e+00 | G-A 0.005: 8.3666 |  |
| 50 | final | Gmax_full_over_dw | all | 4 | 7.143e-05 | 1.091e-17 | 1.700e-17 (0.25) | 5.950e-14 | 1.000e+00 | G-F 0.01: 11.832 |  |
| 50 | final | eta_T_over_dw | all | 4 | 7.143e-05 | 1.091e-17 | 1.700e-17 (0.25) | 5.950e-14 | 1.000e+00 | G-A 0.005: 8.3666 |  |
| 60 | development | Gmax_full_over_dw | all | 4 | 7.143e-05 | 1.091e-17 | 1.700e-17 (0.25) | 5.950e-14 | 1.000e+00 | G-F 0.01: 11.832 |  |
| 60 | development | eta_T_over_dw | all | 4 | 7.143e-05 | 1.091e-17 | 1.700e-17 (0.25) | 5.950e-14 | 1.000e+00 | G-A 0.005: 8.3666 |  |
| 60 | final | Gmax_full_over_dw | all | 4 | 7.143e-05 | 1.091e-17 | 1.700e-17 (0.25) | 5.950e-14 | 1.000e+00 | G-F 0.01: 11.832 |  |
| 60 | final | eta_T_over_dw | all | 4 | 7.143e-05 | 1.091e-17 | 1.700e-17 (0.25) | 5.950e-14 | 1.000e+00 | G-A 0.005: 8.3666 |  |

Source: `results/v2_refine/d2_verifier_sensitivity/fits.csv` (family d); the full residual list is its `residuals` column (input order: ascending perturbation). Points with x = 0 carry no information in a fit through the origin and are excluded; 'x at threshold from fit' is the extrapolation sqrt(threshold / a) of the fitted law and is descriptive only.

### Detection limits

Smallest |perturbation| on the grid at which the metric exceeds the threshold (strict '>'): G-F Gmax_full/DW > 0.01, G-A eta_2/DW > 0.005, G-N |dev - final| of Gmax_full/DW > 0.001 or of eta_2/DW > 0.001 (units of DW). For the signed families (a), (b) the negative and positive sides are listed separately and 'any' is the smaller |perturbation| of the two. 'bracket lo' is the largest grid |perturbation| below the limit on that side.

| q | tier | criterion | side | limit (|p|) | reached | x at limit | value at limit | bracket lo | max |p| on grid | max value on grid |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | development | G-F | any | not reached on the grid | no | n/a | n/a | none | 2 | 2.857e-04 |
| 50 | development | G-A | any | not reached on the grid | no | n/a | n/a | none | 2 | 2.857e-04 |
| 50 | final | G-F | any | not reached on the grid | no | n/a | n/a | none | 2 | 2.857e-04 |
| 50 | final | G-A | any | not reached on the grid | no | n/a | n/a | none | 2 | 2.857e-04 |
| 50 | dev-final | G-N_Gmax | any | not reached on the grid | no | n/a | n/a | none | 2 | 0.000e+00 |
| 50 | dev-final | G-N_eta | any | not reached on the grid | no | n/a | n/a | none | 2 | 0.000e+00 |
| 60 | development | G-F | any | not reached on the grid | no | n/a | n/a | none | 2 | 2.857e-04 |
| 60 | development | G-A | any | not reached on the grid | no | n/a | n/a | none | 2 | 2.857e-04 |
| 60 | final | G-F | any | not reached on the grid | no | n/a | n/a | none | 2 | 2.857e-04 |
| 60 | final | G-A | any | not reached on the grid | no | n/a | n/a | none | 2 | 2.857e-04 |
| 60 | dev-final | G-N_Gmax | any | not reached on the grid | no | n/a | n/a | none | 2 | 0.000e+00 |
| 60 | dev-final | G-N_eta | any | not reached on the grid | no | n/a | n/a | none | 2 | 0.000e+00 |

Source: `results/v2_refine/d2_verifier_sensitivity/detection_limits.csv` (family d).

![Gmax_full/DW and eta_2/DW against |perturbation|, family d](../../../results/v2_refine/d2_verifier_sensitivity/figures/d2_family_d.png)

Figure: `results/v2_refine/d2_verifier_sensitivity/figures/d2_family_d.png` (and `.pdf`), log-log, both tiers, both q; data from `results/v2_refine/d2_verifier_sensitivity/paired.csv`.

## 8. Family (e) RL-like: stage-1 scalar + peak-rounding kernel

Perturbation parameter: `delta_stage1` (relative). Fit abscissa x: `delta_stage1`.

### q = 50: verifier quantities

| perturbation | x | G_dev | G_final | (t*, d*) final | G dev-final | eta_dev | eta_final | eta dev-final | EXP_root final | dReach final | Delta_max_all final | dFull final | valid dev/final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 7.901e-04 | 9.731e-04 | (2, -2) | -1.830e-04 | 7.423e-04 | 9.731e-04 | -2.308e-04 | 7.769e-04 | 1.684e-03 | 9.731e-04 | 1.684e-03 | yes/yes |
| 0.1 | 0.1 | 3.018e-03 | 2.988e-03 | (1, 0) | 2.969e-05 | 7.423e-04 | 9.731e-04 | -2.308e-04 | 2.988e-03 | 3.897e-03 | 2.924e-03 | 3.897e-03 | yes/yes |
| 0.15 | 0.15 | 6.848e-03 | 6.813e-03 | (1, 0) | 3.482e-05 | 7.423e-04 | 9.731e-04 | -2.308e-04 | 6.813e-03 | 7.724e-03 | 6.751e-03 | 7.724e-03 | yes/yes |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family e, q 50); G = Gmax_full/DW, eta = eta_2/DW, all in units of DW; n/a = tier not run.

### q = 50: recovery metrics (tier independent) and clip

| perturbation | stage-1 error (signed) | stage-2 peak error (signed) | RMSE_pos / e2*(0) | tail mean / e2*(0) | clip binds |
|---|---|---|---|---|---|
| 0.05 | 5.000e-02 | -6.000e-02 | 1.031e-02 | 1.270e-03 | no |
| 0.1 | 1.000e-01 | -6.000e-02 | 1.031e-02 | 1.270e-03 | no |
| 0.15 | 1.500e-01 | -6.000e-02 | 1.031e-02 | 1.270e-03 | no |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family e, q 50).

### q = 60: verifier quantities

| perturbation | x | G_dev | G_final | (t*, d*) final | G dev-final | eta_dev | eta_final | eta dev-final | EXP_root final | dReach final | Delta_max_all final | dFull final | valid dev/final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 4.398e-04 | 4.797e-04 | (2, -2) | -3.989e-05 | 4.089e-04 | 4.797e-04 | -7.084e-05 | 3.787e-04 | 8.334e-04 | 4.797e-04 | 8.334e-04 | yes/yes |
| 0.1 | 0.1 | 1.469e-03 | 1.430e-03 | (1, 0) | 3.919e-05 | 4.089e-04 | 4.797e-04 | -7.084e-05 | 1.430e-03 | 1.885e-03 | 1.405e-03 | 1.885e-03 | yes/yes |
| 0.15 | 0.15 | 3.227e-03 | 3.196e-03 | (1, 0) | 3.013e-05 | 4.089e-04 | 4.797e-04 | -7.084e-05 | 3.196e-03 | 3.652e-03 | 3.172e-03 | 3.652e-03 | yes/yes |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family e, q 60); G = Gmax_full/DW, eta = eta_2/DW, all in units of DW; n/a = tier not run.

### q = 60: recovery metrics (tier independent) and clip

| perturbation | stage-1 error (signed) | stage-2 peak error (signed) | RMSE_pos / e2*(0) | tail mean / e2*(0) | clip binds |
|---|---|---|---|---|---|
| 0.05 | 5.000e-02 | -5.000e-02 | 7.843e-03 | 1.058e-03 | no |
| 0.1 | 1.000e-01 | -5.000e-02 | 7.843e-03 | 1.058e-03 | no |
| 0.15 | 1.500e-01 | -5.000e-02 | 7.843e-03 | 1.058e-03 | no |

Source: `results/v2_refine/d2_verifier_sensitivity/paired.csv` (family e, q 60).

### Quadratic fit through the origin: metric ~ a x^2

| q | tier | metric | side | n | a | RMSE of residuals | max abs residual (at x) | max abs resid / max metric | R2 (uncentered) | x at threshold from fit | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | development | Gmax_full_over_dw | all | 3 | 3.041e-01 | 2.211e-05 | 3.000e-05 (0.05) | 4.381e-03 | 1.000e+00 | G-F 0.01: 0.18135 |  |
| 50 | development | eta_T_over_dw | all | 3 | 4.242e-02 | 4.286e-04 | 6.363e-04 (0.05) | 8.571e-01 | 6.667e-01 | G-A 0.005: 0.34333 | eta_2 does not depend on x in this family (stage 2 is unchanged along the perturbation): not a model |
| 50 | final | Gmax_full_over_dw | all | 3 | 3.030e-01 | 1.268e-04 | 2.155e-04 (0.05) | 3.163e-02 | 9.991e-01 | G-F 0.01: 0.18166 |  |
| 50 | final | eta_T_over_dw | all | 3 | 5.561e-02 | 5.618e-04 | 8.341e-04 (0.05) | 8.571e-01 | 6.667e-01 | G-A 0.005: 0.29986 | eta_2 does not depend on x in this family (stage 2 is unchanged along the perturbation): not a model |
| 60 | development | Gmax_full_over_dw | all | 3 | 1.443e-01 | 4.943e-05 | 7.906e-05 (0.05) | 2.450e-02 | 9.994e-01 | G-F 0.01: 0.26324 |  |
| 60 | development | eta_T_over_dw | all | 3 | 2.336e-02 | 2.361e-04 | 3.505e-04 (0.05) | 8.571e-01 | 6.667e-01 | G-A 0.005: 0.4626 | eta_2 does not depend on x in this family (stage 2 is unchanged along the perturbation): not a model |
| 60 | final | Gmax_full_over_dw | all | 3 | 1.427e-01 | 7.149e-05 | 1.229e-04 (0.05) | 3.845e-02 | 9.988e-01 | G-F 0.01: 0.2647 |  |
| 60 | final | eta_T_over_dw | all | 3 | 2.741e-02 | 2.770e-04 | 4.112e-04 (0.05) | 8.571e-01 | 6.667e-01 | G-A 0.005: 0.42708 | eta_2 does not depend on x in this family (stage 2 is unchanged along the perturbation): not a model |

Source: `results/v2_refine/d2_verifier_sensitivity/fits.csv` (family e); the full residual list is its `residuals` column (input order: ascending perturbation). Points with x = 0 carry no information in a fit through the origin and are excluded; 'x at threshold from fit' is the extrapolation sqrt(threshold / a) of the fitted law and is descriptive only.

### Detection limits

Smallest |perturbation| on the grid at which the metric exceeds the threshold (strict '>'): G-F Gmax_full/DW > 0.01, G-A eta_2/DW > 0.005, G-N |dev - final| of Gmax_full/DW > 0.001 or of eta_2/DW > 0.001 (units of DW). For the signed families (a), (b) the negative and positive sides are listed separately and 'any' is the smaller |perturbation| of the two. 'bracket lo' is the largest grid |perturbation| below the limit on that side.

| q | tier | criterion | side | limit (|p|) | reached | x at limit | value at limit | bracket lo | max |p| on grid | max value on grid |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | development | G-F | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 6.848e-03 |
| 50 | development | G-A | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 7.423e-04 |
| 50 | final | G-F | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 6.813e-03 |
| 50 | final | G-A | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 9.731e-04 |
| 50 | dev-final | G-N_Gmax | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 1.830e-04 |
| 50 | dev-final | G-N_eta | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 2.308e-04 |
| 60 | development | G-F | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 3.227e-03 |
| 60 | development | G-A | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 4.089e-04 |
| 60 | final | G-F | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 3.196e-03 |
| 60 | final | G-A | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 4.797e-04 |
| 60 | dev-final | G-N_Gmax | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 3.989e-05 |
| 60 | dev-final | G-N_eta | any | not reached on the grid | no | n/a | n/a | none | 0.15 | 7.084e-05 |

Source: `results/v2_refine/d2_verifier_sensitivity/detection_limits.csv` (family e).

![Gmax_full/DW and eta_2/DW against |perturbation|, family e](../../../results/v2_refine/d2_verifier_sensitivity/figures/d2_family_e.png)

Figure: `results/v2_refine/d2_verifier_sensitivity/figures/d2_family_e.png` (and `.pdf`), log-log, both tiers, both q; data from `results/v2_refine/d2_verifier_sensitivity/paired.csv`.

### Family (e) predictions next to the confirmation runs with the largest stage-1 errors

Observed values are the confirmation runs' end-of-B candidates (their own, imperfect stage 2); predicted values are the family-(e) candidate at the grid delta nearest to the run's |stage-1 error| (positive deltas only) with the chosen kernel. Peak errors are shown because the (e) kernel and the run's own stage-2 shape differ.

| q | seed | stage-1 error | observed Gmax final | observed Gmax dev | observed peak error | (e) delta | (e) kernel h | (e) peak error | (e) predicted Gmax final | (e) predicted Gmax dev | observed / predicted (final) | kernel-only (c) Gmax final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 20504 | 8.619e-02 | 2.012e-03 | 1.994e-03 | -6.975e-02 | 0.1 | 12 | -6.000e-02 | 2.988e-03 | 3.018e-03 | 0.67324 | 9.731e-04 |
| 50 | 20511 | 7.774e-02 | 2.735e-03 | 2.765e-03 | -4.443e-02 | 0.1 | 12 | -6.000e-02 | 2.988e-03 | 3.018e-03 | 0.91535 | 9.731e-04 |
| 50 | 20509 | -7.694e-02 | 1.521e-03 | 1.522e-03 | -5.724e-02 | 0.1 | 12 | -6.000e-02 | 2.988e-03 | 3.018e-03 | 0.50897 | 9.731e-04 |
| 50 | 20508 | -6.961e-02 | 1.337e-03 | 1.349e-03 | -6.183e-02 | 0.05 | 12 | -6.000e-02 | 9.731e-04 | 7.901e-04 | 1.3739 | 9.731e-04 |
| 50 | 20513 | -6.098e-02 | 1.745e-03 | 1.728e-03 | -8.072e-02 | 0.05 | 12 | -6.000e-02 | 9.731e-04 | 7.901e-04 | 1.7931 | 9.731e-04 |
| 60 | 20510 | 1.530e-01 | 3.638e-03 | 3.649e-03 | -1.087e-01 | 0.15 | 12 | -5.000e-02 | 3.196e-03 | 3.227e-03 | 1.138 | 4.797e-04 |
| 60 | 20515 | -1.135e-01 | 1.810e-03 | 1.829e-03 | -8.249e-02 | 0.1 | 12 | -5.000e-02 | 1.430e-03 | 1.469e-03 | 1.2658 | 4.797e-04 |
| 60 | 20513 | 8.442e-02 | 1.171e-03 | 1.173e-03 | -6.038e-02 | 0.1 | 12 | -5.000e-02 | 1.430e-03 | 1.469e-03 | 0.81902 | 4.797e-04 |
| 60 | 20502 | 7.347e-02 | 9.342e-04 | 9.415e-04 | -5.858e-02 | 0.05 | 12 | -5.000e-02 | 4.797e-04 | 4.398e-04 | 1.9474 | 4.797e-04 |
| 60 | 20506 | 7.007e-02 | 9.109e-04 | 9.167e-04 | -6.582e-02 | 0.05 | 12 | -5.000e-02 | 4.797e-04 | 4.398e-04 | 1.8988 | 4.797e-04 |

Source: `results/v2_refine/d2_verifier_sensitivity/family_e_confirmation.csv`; observed columns from `/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/confirmation_analysis/per_run.csv` (sha256 `d4a77eb49963407a...`) and `/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/confirmation_analysis/reported_metrics.csv` (read only, canonical worktree); predicted columns from `results/v2_refine/d2_verifier_sensitivity/paired.csv`.

## 9. Detection limits, all families

| family | q | tier | criterion | limit (|p|, any side) | x at limit |
|---|---|---|---|---|---|
| a | 50 | development | G-F | not reached on the grid | n/a |
| a | 50 | development | G-A | not reached on the grid | n/a |
| a | 50 | final | G-F | not reached on the grid | n/a |
| a | 50 | final | G-A | not reached on the grid | n/a |
| a | 50 | dev-final | G-N_Gmax | not reached on the grid | n/a |
| a | 50 | dev-final | G-N_eta | not reached on the grid | n/a |
| a | 60 | development | G-F | not reached on the grid | n/a |
| a | 60 | development | G-A | not reached on the grid | n/a |
| a | 60 | final | G-F | not reached on the grid | n/a |
| a | 60 | final | G-A | not reached on the grid | n/a |
| a | 60 | dev-final | G-N_Gmax | not reached on the grid | n/a |
| a | 60 | dev-final | G-N_eta | not reached on the grid | n/a |
| b | 50 | development | G-F | 0.1 | 0.1 |
| b | 50 | development | G-A | 0.1 | -0.1 |
| b | 50 | final | G-F | 0.1 | 0.1 |
| b | 50 | final | G-A | 0.1 | -0.1 |
| b | 50 | dev-final | G-N_Gmax | not reached on the grid | n/a |
| b | 50 | dev-final | G-N_eta | not reached on the grid | n/a |
| b | 60 | development | G-F | 0.15 | 0.15 |
| b | 60 | development | G-A | 0.15 | -0.15 |
| b | 60 | final | G-F | 0.15 | 0.15 |
| b | 60 | final | G-A | 0.15 | -0.15 |
| b | 60 | dev-final | G-N_Gmax | not reached on the grid | n/a |
| b | 60 | dev-final | G-N_eta | not reached on the grid | n/a |
| c | 50 | development | G-F | not reached on the grid | n/a |
| c | 50 | development | G-A | not reached on the grid | n/a |
| c | 50 | final | G-F | not reached on the grid | n/a |
| c | 50 | final | G-A | not reached on the grid | n/a |
| c | 50 | dev-final | G-N_Gmax | not reached on the grid | n/a |
| c | 50 | dev-final | G-N_eta | not reached on the grid | n/a |
| c | 60 | development | G-F | not reached on the grid | n/a |
| c | 60 | development | G-A | not reached on the grid | n/a |
| c | 60 | final | G-F | not reached on the grid | n/a |
| c | 60 | final | G-A | not reached on the grid | n/a |
| c | 60 | dev-final | G-N_Gmax | not reached on the grid | n/a |
| c | 60 | dev-final | G-N_eta | not reached on the grid | n/a |
| d | 50 | development | G-F | not reached on the grid | n/a |
| d | 50 | development | G-A | not reached on the grid | n/a |
| d | 50 | final | G-F | not reached on the grid | n/a |
| d | 50 | final | G-A | not reached on the grid | n/a |
| d | 50 | dev-final | G-N_Gmax | not reached on the grid | n/a |
| d | 50 | dev-final | G-N_eta | not reached on the grid | n/a |
| d | 60 | development | G-F | not reached on the grid | n/a |
| d | 60 | development | G-A | not reached on the grid | n/a |
| d | 60 | final | G-F | not reached on the grid | n/a |
| d | 60 | final | G-A | not reached on the grid | n/a |
| d | 60 | dev-final | G-N_Gmax | not reached on the grid | n/a |
| d | 60 | dev-final | G-N_eta | not reached on the grid | n/a |
| e | 50 | development | G-F | not reached on the grid | n/a |
| e | 50 | development | G-A | not reached on the grid | n/a |
| e | 50 | final | G-F | not reached on the grid | n/a |
| e | 50 | final | G-A | not reached on the grid | n/a |
| e | 50 | dev-final | G-N_Gmax | not reached on the grid | n/a |
| e | 50 | dev-final | G-N_eta | not reached on the grid | n/a |
| e | 60 | development | G-F | not reached on the grid | n/a |
| e | 60 | development | G-A | not reached on the grid | n/a |
| e | 60 | final | G-F | not reached on the grid | n/a |
| e | 60 | final | G-A | not reached on the grid | n/a |
| e | 60 | dev-final | G-N_Gmax | not reached on the grid | n/a |
| e | 60 | dev-final | G-N_eta | not reached on the grid | n/a |

Source: `results/v2_refine/d2_verifier_sensitivity/detection_limits.csv` (side = any). Perturbation units: (a), (b), (e) relative delta; (c) kernel half-width h in d units; (d) tau in effort units.

## 10. Limitations

This is a sensitivity study of the verifier on closed-form perturbations of the exact equilibrium. It does not train anything, does not use Beta policies (the candidates are deterministic mean functions, so the distributional smoothing of a trained actor is absent), covers one perturbation family at a time (plus the single combination (e)), uses only the pre-registered grids (the detection limits are grid-resolution bounds, not continuous thresholds; 'not reached on the grid' says nothing beyond the largest perturbation tested), and relates to trained runs only through the descriptive comparison in the family-(e) section. The verifier tiers, the gates and every threshold of protocol v1.1 are used as locked and are not changed by this report.
