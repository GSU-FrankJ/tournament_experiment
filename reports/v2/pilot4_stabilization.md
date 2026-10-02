# Pilot 4 — Stabilization round (analyses on existing data + learning-rate decay)

Date: 2026-10-02. Branch `claude/pilot-4-stabilization-fb99a2`, on top of `v2-stagewise-pilots` (`e0e8327`).

| Item | Value |
|---|---|
| **Base commit** | `657f54a` |
| **Launch commit for all 80 runs (2a, 2b)** | **`c92ee74`** (`feat: add linear LR decay window and mid-run parent files for pilot 4`); recorded in every manifest (`A/run_records.csv`, column `commit`) |
| Dirty flag | `false` in 72 runs; `true` in 8 runs (2b, q=60, seeds 10507–10510, both arms); see §8.1 |
| Tests on `c92ee74` | `pytest tests/`: 77 passed, 2 xfailed, 1 failed. The failure is `tests/test_registry_canonicalization.py` (paper registry; "expected 15 Set-2 gradient runs, got 0"); it fails identically on `main` and does not touch v2 code |
| C7 regression on `c92ee74` | **IDENTICAL** (`results/v2_pilots/phase2_regression/v2_full_c92ee74.compare.txt`; manifest `dirty: false`) |
| Decisions applied | D1 continuation `mean`; D2 Phase A = 1600 updates; D3 no joint training, no Phase C; D4 no sampler change |

Path conventions:
- `A/` means `results/v2_pilots/pilot4/analysis/`.
- `P4A/` means `results/v2_pilots/pilot4_A/` (2a runs), `P4B/` means `results/v2_pilots/pilot4_B/` (2b runs).
- Figures are in `reports/v2/figures/pilot4/`.
- Every table in this report is rendered by `tools/v2/pilot4_tables.py` from the CSV named under it (also collected in `A/report_tables.md`).

ΔW = 4. e₁*(0) = 46.667 (q=50) and 38.889 (q=60); e₂*(0) = 70.0 and 58.333. All statistics are descriptive. **No gate thresholds are proposed, and the protocol is not locked.**

---

## 0. Code added in this round

| File | Role |
|---|---|
| `run/run_v2_stagewise.py` | Optional config field `lr_decay = {phase, start_lr, end_lr, local_first, local_last}`. Inside that phase, both optimizers follow the existing linear form `lr_at` (`run/run_final_dp_br_round3_dense.py:153-167`, branch `kind == "linear"`), mapped onto local updates `local_first..local_last`. New method `Run.lr_for` (`run/run_v2_stagewise.py:285-297`). Refused in mode `full`; `start_lr` must equal `ab_lr`; `local_last` must equal the phase cap. |
| `tools/v2/launch_pilot.py` | `--parent-file` (full-state file inside the parent run dir; default `state_end_A.pt`); arms `constant`, `decay` (Phase A continuation) and `B2_mean_constant`, `B2_mean_decay` (Phase B, B2 + `mean`); decay arms get `lr_decay` with `end_lr = 3e-5`. |
| `tests/test_v2_infra.py` | 3 refusal cases + `test_lr_decay_follows_existing_linear_form`. |
| `tools/v2/induced_band.py` | Sub-command `parents4` (bands of the 20 u1600 parents, adaptive sweep range). |
| `tools/v2/pilot4_*.py` | Analyses: `root_game` (1a), `fluctuation` (1b), `repr_floor` (1d), `repro` (2a check), `analysis` (1c, 2a, 2b, distribution tables), `common`, `tables`. |

**Decay schedule (both decay arms).** The existing linear schedule is the round-2 G3/G4 C-phase schedule: 3e−4 linearly to 3e−5 (`MultiStage/Discussion/ROUND2_T2_FOUR_GROUP_SETTINGS_20260909.md`, lines 24 and 38; record `lr_schedule` with `kind = linear`, `c_start_lr = 3e-4`, `c_end_lr = 3e-5`). Its end value is > 0, so it is kept as is; no new LR value is introduced.

lr(j) = 3e−4 + (3e−5 − 3e−4)·(j − 1)/(N − 1), applied to actor and critic before each update, constant within an update, Adam state preserved.

| Arm | Window | N | lr at j=1 | lr at j=N |
|---|---|---|---|---|
| 2a `decay` | u1201–u1600 (Phase A local 1–400) | 400 | 3e−4 | 3e−5 |
| 2b `decay` | u1601–u2200 (Phase B local 1–600) | 600 | 3e−4 | 3e−5 (logged 3.0000000000000024e−05) |

The actual per-update `actor_lr` and `critic_lr` are in each run's `train_history.json`. First and last values per arm are in `A/run_records.csv`.

---

## 1a. Root stage game: reference vs numeric

Method (`tools/v2/pilot4_root_game.py`):
- **Q₁.** The verifier's own Q₁(0, e | e_opp) is `utils.v2_metrics.stage1_Q`. It uses V₂^mean from `verify` on the closed-form candidate, with e₂* as the continuation for both players.
- **BR(e_opp).** Take the grid argmax on the verifier's effort grid. Then fit a least-squares quadratic to Q₁ at the effort-grid nodes within ±W of that argmax. BR is the vertex −b/(2a). W ∈ {1, 2, 4} effort units, which is 5/9/17 nodes on the final tier (step 0.5).
- **Slope.** (BR(e₁*+h) − BR(e₁*−h))/(2h) for h ∈ {0.25, 0.5, 1, 2, 4}.
- **Own curvature.** 2a of the same fit at e_opp = e₁*, expressed as 2a/(2k). The implied E[V₂″]/(2k) is 1 + 2a/(2k).
- **Tiers.** `final` is the requested tier. A finer tier `fine` (state 0.5, effort 0.25, 64 GL nodes per half) is shown only to separate fit and grid noise from a real discrepancy.

Source: `results/v2_pilots/pilot4/analysis/root_game/own_curvature.csv`

| q | tier | fit_half_width | fit_points | BR_minus_e1star | own_curv_over_2k | ev2pp_over_2k | ref_ev2pp_over_2k | slope_implied_by_curv |
|---|---|---|---|---|---|---|---|---|
| 50 | final | 1 | 5 | 0.03256 | -0.4869 | 0.5131 | 0.49 | -1.054 |
| 50 | final | 2 | 9 | 0.0025 | -0.528 | 0.472 | 0.49 | -0.894 |
| 50 | final | 4 | 17 | -0.04676 | -0.5239 | 0.4761 | 0.49 | -0.9088 |
| 50 | fine | 1 | 9 | -0.004796 | -0.5076 | 0.4924 | 0.49 | -0.9699 |
| 50 | fine | 2 | 17 | -0.01149 | -0.5205 | 0.4795 | 0.49 | -0.9212 |
| 50 | fine | 4 | 33 | -0.04506 | -0.5223 | 0.4777 | 0.49 | -0.9146 |
| 60 | final | 1 | 5 | 0.01227 | -0.7871 | 0.2129 | 0.236 | -0.2706 |
| 60 | final | 2 | 9 | -0.01177 | -0.7917 | 0.2083 | 0.236 | -0.2631 |
| 60 | final | 4 | 17 | -0.01861 | -0.7781 | 0.2219 | 0.236 | -0.2853 |
| 60 | fine | 1 | 9 | -0.001071 | -0.768 | 0.232 | 0.236 | -0.302 |
| 60 | fine | 2 | 17 | -0.005135 | -0.7663 | 0.2337 | 0.236 | -0.305 |
| 60 | fine | 4 | 33 | -0.01868 | -0.7693 | 0.2307 | 0.236 | -0.2999 |

Source: `results/v2_pilots/pilot4/analysis/root_game/br_slope.csv`

| q | tier | fit_half_width | h=0.25 | h=0.5 | h=1 | h=2 | h=4 | reference |
|---|---|---|---|---|---|---|---|---|
| 50 | final | 1 | -1.057 | -0.8924 | -0.9232 | -0.9269 | -0.8876 | -0.961 |
| 50 | final | 2 | -0.9741 | -0.9243 | -0.9288 | -0.8966 | -0.8585 | -0.961 |
| 50 | final | 4 | -0.9628 | -0.8741 | -0.9113 | -0.8825 | -0.8533 | -0.961 |
| 50 | fine | 1 | -0.9471 | -0.9226 | -0.9248 | -0.8993 | -0.8567 | -0.961 |
| 50 | fine | 2 | -0.9324 | -0.9288 | -0.9224 | -0.8981 | -0.8611 | -0.961 |
| 50 | fine | 4 | -0.9002 | -0.9079 | -0.9021 | -0.8876 | -0.8537 | -0.961 |
| 60 | final | 1 | -0.3322 | -0.332 | -0.2723 | -0.3061 | -0.2792 | -0.309 |
| 60 | final | 2 | -0.2644 | -0.2942 | -0.3138 | -0.3019 | -0.2958 | -0.309 |
| 60 | final | 4 | -0.2928 | -0.2922 | -0.3034 | -0.2944 | -0.2894 | -0.309 |
| 60 | fine | 1 | -0.3108 | -0.2968 | -0.2998 | -0.3041 | -0.2978 | -0.309 |
| 60 | fine | 2 | -0.3022 | -0.3019 | -0.3015 | -0.3004 | -0.2928 | -0.309 |
| 60 | fine | 4 | -0.2965 | -0.3001 | -0.298 | -0.2965 | -0.2926 | -0.309 |

**Comparison with the reference values.**

| q | quantity | reference | final tier (range over W, and over W×h for the slope) | fine tier |
|---|---|---|---|---|
| 50 | E[V₂″]/(2k) | 0.490 | 0.472–0.513 | 0.478–0.492 |
| 50 | own curvature /2k | −0.510 | −0.487 to −0.528 | −0.508 to −0.522 |
| 50 | dBR/de_opp | −0.961 | −0.853 to −1.057 (h ≤ 1: −0.874 to −1.057) | −0.854 to −0.947 (h ≤ 1: −0.900 to −0.947) |
| 60 | E[V₂″]/(2k) | 0.236 | 0.208–0.222 | 0.231–0.234 |
| 60 | own curvature /2k | −0.764 | −0.778 to −0.792 | −0.766 to −0.769 |
| 60 | dBR/de_opp | −0.309 | −0.264 to −0.332 | −0.292 to −0.311 |

- **q=60.** The numeric values bracket the reference on both tiers. On the fine tier, curvature is within 0.005 and the slope within 0.02 of the reference.
- **q=50.**
  - The curvature agrees within the fit variation (0.478–0.492 on the fine tier vs 0.490).
  - The final-tier slope range contains −0.961.
  - The fine-tier slopes, −0.90 to −0.95 at h ≤ 1, sit slightly below the reference in magnitude.
  - This is within the sensitivity of the slope to r = E[V₂″]/(2k): the slope is −r/(1−r), so r = 0.478 → −0.92 and r = 0.490 → −0.961.
  - The slope magnitude also falls as h grows (−0.85 at h = 4), so Q₁ is not quadratic over ±4.
- **No disagreement beyond the finite-difference and fit variation was found.**
- **Fixed point.** BR(e₁*) − e₁* is within ±0.05 effort units in every fit, consistent with e₁* being the fixed point.

---

## 1b. Anatomy of the stage-1 fluctuation (Pilot 3, both arms)

**Record used.** There is **no per-update record of the root policy mean.** `history[*].mean_effort_by_stage` is the batch mean of *sampled* efforts, not the Beta mean. The densest regular record is the stability log in `train_history.json`:
- it records ê₁(0), the Beta mean, every 20 updates, at global u420, 440, …, 1000;
- the verifier calls (every 25 from u500) and the weight exports (every 25) are sparser.

**Why the window regression is still possible.** The stability samples fall exactly on the opponent-refresh updates (global multiples of 20). After update u the lagged opponent is refreshed to the actor at u, so:
- the opponent for updates u+1…u+20 plays root mean ê₁(u);
- at the start of each window the learner and the opponent coincide;
- the learner's change over the window is ê₁(u+20) − ê₁(u).

The regression uses only the window endpoints, so per-update data would give exactly the same regression. I therefore ran it, with that caveat stated. Data: `A/fluctuation/stability_series_e1.csv`, `windows.csv`.

**ACF definitions.** Per-run sample ACF of x = ê₁(0) − e₁* on the segment:
- u ≥ 650 → u660–u1000, n = 18;
- u ≥ 700 → u700–u1000, n = 16.

Two versions are given: centred on the segment mean (standard), and about e₁* (x not demeaned). With n = 16–18, a per-run ACF carries a small-sample negative bias of order −1/n.

Source: `results/v2_pilots/pilot4/analysis/fluctuation/acf_summary_every20.csv`

| q | arm | lag 20 | lag 40 | lag 60 | lag 80 | lag 100 | lag 120 | lag 140 | lag 160 |
|---|---|---|---|---|---|---|---|---|---|
| 50 | mean | 0.59 [0.19, 0.72] | 0.30 [0.17, 0.48] | 0.05 [-0.07, 0.32] | 0.06 [0.02, 0.15] | 0.01 [-0.15, 0.09] | -0.11 [-0.19, 0.00] | -0.19 [-0.27, -0.06] | -0.23 [-0.27, -0.10] |
| 50 | stochastic | 0.48 [0.40, 0.69] | 0.29 [0.13, 0.41] | 0.10 [0.01, 0.26] | 0.02 [-0.02, 0.14] | -0.09 [-0.20, 0.02] | -0.19 [-0.26, -0.09] | -0.20 [-0.28, -0.06] | -0.25 [-0.34, -0.16] |
| 60 | mean | 0.51 [0.41, 0.56] | 0.23 [0.05, 0.27] | -0.06 [-0.21, 0.01] | -0.24 [-0.26, -0.11] | -0.10 [-0.29, -0.01] | -0.11 [-0.22, -0.09] | -0.11 [-0.25, -0.08] | -0.06 [-0.11, 0.00] |
| 60 | stochastic | 0.60 [0.56, 0.67] | 0.28 [0.04, 0.40] | -0.06 [-0.15, 0.24] | -0.08 [-0.40, 0.02] | -0.25 [-0.34, 0.01] | -0.20 [-0.31, -0.09] | -0.18 [-0.27, -0.06] | -0.07 [-0.20, 0.05] |

Source: `results/v2_pilots/pilot4/analysis/fluctuation/acf_summary_every20.csv`

| q | arm | lag 20 | lag 40 | lag 60 | lag 80 | lag 100 | lag 120 | lag 140 | lag 160 |
|---|---|---|---|---|---|---|---|---|---|
| 50 | mean | 0.55 [0.23, 0.70] | 0.22 [0.10, 0.39] | 0.07 [-0.15, 0.16] | 0.02 [-0.03, 0.12] | -0.07 [-0.16, 0.04] | -0.14 [-0.17, -0.07] | -0.17 [-0.29, -0.03] | -0.24 [-0.31, -0.01] |
| 50 | stochastic | 0.44 [0.39, 0.65] | 0.24 [0.14, 0.44] | 0.18 [-0.00, 0.21] | -0.01 [-0.06, 0.05] | -0.11 [-0.17, -0.01] | -0.14 [-0.21, -0.09] | -0.23 [-0.32, -0.08] | -0.27 [-0.40, -0.19] |
| 60 | mean | 0.51 [0.39, 0.58] | 0.23 [0.04, 0.26] | -0.07 [-0.23, 0.03] | -0.25 [-0.32, -0.11] | -0.07 [-0.29, -0.01] | -0.09 [-0.21, -0.02] | -0.07 [-0.18, -0.03] | -0.05 [-0.11, -0.02] |
| 60 | stochastic | 0.59 [0.48, 0.62] | 0.30 [0.13, 0.39] | -0.02 [-0.15, 0.28] | -0.05 [-0.33, 0.06] | -0.24 [-0.37, -0.07] | -0.21 [-0.33, -0.13] | -0.16 [-0.27, -0.06] | -0.12 [-0.23, 0.07] |

Source: `results/v2_pilots/pilot4/analysis/fluctuation/acf_summary_every20.csv`

| q | arm | lag 20 | lag 40 | lag 60 | lag 80 | lag 100 | lag 120 | lag 140 | lag 160 |
|---|---|---|---|---|---|---|---|---|---|
| 50 | mean | 0.77 [0.57, 0.87] | 0.54 [0.41, 0.78] | 0.44 [0.30, 0.66] | 0.37 [0.29, 0.56] | 0.33 [0.24, 0.47] | 0.29 [0.09, 0.42] | 0.22 [0.03, 0.30] | 0.13 [-0.00, 0.22] |
| 50 | stochastic | 0.82 [0.76, 0.87] | 0.65 [0.60, 0.72] | 0.51 [0.47, 0.57] | 0.38 [0.29, 0.54] | 0.29 [0.17, 0.48] | 0.15 [0.01, 0.42] | 0.12 [0.05, 0.34] | 0.11 [-0.04, 0.25] |
| 60 | mean | 0.71 [0.68, 0.77] | 0.55 [0.32, 0.63] | 0.46 [0.08, 0.57] | 0.29 [0.10, 0.56] | 0.29 [0.08, 0.51] | 0.30 [0.02, 0.46] | 0.24 [0.02, 0.40] | 0.27 [0.02, 0.40] |
| 60 | stochastic | 0.72 [0.65, 0.87] | 0.48 [0.27, 0.74] | 0.28 [-0.05, 0.62] | 0.15 [-0.06, 0.50] | 0.12 [-0.18, 0.44] | 0.06 [-0.17, 0.41] | 0.03 [-0.10, 0.38] | 0.14 [-0.02, 0.34] |

Source: `results/v2_pilots/pilot4/analysis/fluctuation/acf_summary_every20.csv`

| q | arm | lag 20 | lag 40 | lag 60 | lag 80 | lag 100 | lag 120 | lag 140 | lag 160 |
|---|---|---|---|---|---|---|---|---|---|
| 50 | mean | 0.76 [0.62, 0.87] | 0.58 [0.40, 0.75] | 0.43 [0.29, 0.62] | 0.33 [0.23, 0.55] | 0.28 [0.24, 0.40] | 0.25 [0.12, 0.35] | 0.19 [0.12, 0.31] | 0.14 [0.10, 0.24] |
| 50 | stochastic | 0.79 [0.74, 0.85] | 0.62 [0.60, 0.71] | 0.46 [0.40, 0.55] | 0.29 [0.24, 0.51] | 0.25 [0.12, 0.41] | 0.12 [0.01, 0.35] | 0.10 [0.03, 0.29] | 0.11 [-0.06, 0.25] |
| 60 | mean | 0.76 [0.70, 0.84] | 0.58 [0.41, 0.69] | 0.48 [0.11, 0.60] | 0.34 [0.09, 0.54] | 0.32 [0.07, 0.52] | 0.35 [-0.01, 0.47] | 0.26 [0.04, 0.42] | 0.26 [0.01, 0.37] |
| 60 | stochastic | 0.71 [0.60, 0.81] | 0.53 [0.20, 0.67] | 0.32 [0.05, 0.53] | 0.24 [0.01, 0.42] | 0.17 [-0.11, 0.41] | 0.08 [-0.13, 0.39] | 0.14 [-0.12, 0.33] | 0.19 [0.07, 0.24] |

Secondary, from the 25-update weight exports at lags 25, 50, 75 and 100 (the exports do not resolve the 20-update refresh cycle):

Source: `results/v2_pilots/pilot4/analysis/fluctuation/acf_summary_exports25.csv`

| segment_start | q | arm | lag 25 | lag 50 | lag 75 | lag 100 |
|---|---|---|---|---|---|---|
| 650 | 50 | mean | 0.50 [0.35, 0.63] | 0.30 [0.24, 0.36] | 0.00 [-0.14, 0.12] | -0.02 [-0.13, 0.13] |
| 650 | 50 | stochastic | 0.46 [0.30, 0.59] | 0.25 [-0.06, 0.32] | 0.04 [-0.11, 0.16] | -0.15 [-0.23, -0.03] |
| 650 | 60 | mean | 0.44 [0.32, 0.60] | 0.11 [0.00, 0.12] | -0.15 [-0.25, -0.08] | -0.13 [-0.16, -0.06] |
| 650 | 60 | stochastic | 0.52 [0.44, 0.56] | 0.11 [-0.03, 0.30] | -0.08 [-0.38, 0.06] | -0.22 [-0.32, -0.01] |
| 700 | 50 | mean | 0.42 [0.15, 0.62] | 0.19 [0.05, 0.30] | -0.08 [-0.25, 0.02] | -0.09 [-0.22, 0.03] |
| 700 | 50 | stochastic | 0.44 [0.35, 0.60] | 0.20 [0.06, 0.32] | 0.05 [-0.14, 0.12] | -0.12 [-0.28, -0.03] |
| 700 | 60 | mean | 0.38 [0.35, 0.63] | 0.09 [-0.03, 0.20] | -0.12 [-0.31, 0.02] | -0.11 [-0.16, -0.02] |
| 700 | 60 | stochastic | 0.51 [0.36, 0.53] | 0.13 [-0.02, 0.27] | -0.03 [-0.22, 0.07] | -0.19 [-0.34, -0.03] |

**Window regression.**
- y = ê₁(u+20) − ê₁(u) is regressed on x = e_opp − e₁* = ê₁(u) − e₁*, by OLS with intercept.
- Windows are pooled over seeds per (q, arm), and per q with both arms pooled.
- 95% CI: cluster bootstrap over runs, 10,000 resamples, numpy seed 20261001.
- Per-run OLS slopes (median and IQR) are also given.

Source: `results/v2_pilots/pilot4/analysis/fluctuation/window_regression.csv`

| segment_start | q | arm | n_windows | n_runs | slope | boot_ci95_lo | boot_ci95_hi | per_run_slope_median | per_run_slope_q25 | per_run_slope_q75 |
|---|---|---|---|---|---|---|---|---|---|---|
| 650 | 50 | mean | 170 | 10 | -0.1537 | -0.2937 | -0.07901 | -0.3597 | -0.7876 | -0.1581 |
| 650 | 50 | stochastic | 170 | 10 | -0.1457 | -0.236 | -0.1189 | -0.3964 | -0.5056 | -0.2987 |
| 650 | 60 | mean | 170 | 10 | -0.198 | -0.3171 | -0.133 | -0.4153 | -0.4345 | -0.3702 |
| 650 | 60 | stochastic | 170 | 10 | -0.1999 | -0.287 | -0.1617 | -0.3279 | -0.3691 | -0.2669 |
| 650 | 50 | both | 340 | 20 | -0.149 | -0.2196 | -0.1068 | -0.3964 | -0.6047 | -0.2102 |
| 650 | 60 | both | 340 | 20 | -0.1965 | -0.2638 | -0.1503 | -0.3686 | -0.4173 | -0.3081 |
| 700 | 50 | mean | 150 | 10 | -0.1358 | -0.3082 | -0.05948 | -0.4324 | -0.7476 | -0.1558 |
| 700 | 50 | stochastic | 150 | 10 | -0.1428 | -0.2261 | -0.09964 | -0.3681 | -0.5404 | -0.2423 |
| 700 | 60 | mean | 150 | 10 | -0.1675 | -0.2795 | -0.1059 | -0.3867 | -0.4098 | -0.3207 |
| 700 | 60 | stochastic | 150 | 10 | -0.2355 | -0.3224 | -0.1524 | -0.3283 | -0.3938 | -0.2541 |
| 700 | 50 | both | 300 | 20 | -0.1386 | -0.2143 | -0.09393 | -0.3833 | -0.5933 | -0.2372 |
| 700 | 60 | both | 300 | 20 | -0.2033 | -0.2707 | -0.1485 | -0.3754 | -0.4032 | -0.2585 |

**Comparison with the 1a BR slope (descriptive only).**

| q | pooled window slope (u ≥ 650, both arms) | per-run median slope | BR slope (1a reference / fine tier, h=1) |
|---|---|---|---|
| 50 | −0.149 [−0.220, −0.107] | −0.396 | −0.961 / −0.922 |
| 60 | −0.196 [−0.264, −0.150] | −0.369 | −0.309 / −0.302 |

The two are not expected to be equal: the learner moves only part of the way toward BR within a window, and its starting point equals the opponent's. The pooled and per-run slopes differ by a factor of about 2. Pooling mixes between-run level differences into the slope, and short per-run series (15–17 windows) bias per-run OLS slopes.

**Summary of 1b.**
- The fluctuation is positively autocorrelated over one to two refresh windows: median lag-20 ACF 0.48–0.60, lag-40 0.23–0.30.
- It is near 0 at lag 60.
- It is negative at lags 100–160: medians −0.06 to −0.25 at u ≥ 650.
- The pattern is the same at u ≥ 700 and in both arms.

---

## 1c. Tail-averaged candidates (post hoc)

**Construction** (`tools/v2/pilot4_common.py`).
- candidate_K(t, d) = (1/K) Σₖ μₖ(t, d), where μₖ is the Beta mean e_min + 100·αₖ/(αₖ+βₖ) of the k-th of the last K weight exports.
- Each network is evaluated at the exact query point d. The average is therefore exact at every point the verifier or the recovery grid queries; no interpolation is involved, the same as for a single network.
- Only trained stages are averaged; the frozen stage is the parent actor, unchanged.
- K = 1 is the last iterate.
- All metrics come from `evaluate` on the development tier, as in Pilots 1–3.
- σ is reported only for K = 1, because the average is not a Beta.

**Consistency check.** K = 1 reproduces the published medians exactly:
- Pilot 3 at u1000: stage-1 error −0.02835 / 0.01962, EXP 0.00234 / 0.00170;
- the extension at u1600: peak −0.06835 / −0.06003, RMSE 0.02605 / 0.02389.

### 1c.1 Pilot 3, stage 1 (K ∈ {1, 4, 8, 12}; K = 12 covers u725–u1000)

Decomposition against the parent band (`results/v2_pilots/induced_band/parent_bands.csv`). The inherited term is unchanged by construction: 0.00139 at q=50 and 0.00064 at q=60.

Source: `results/v2_pilots/pilot4/analysis/candidates_all.csv`

| q | arm | K | e1_cand | stage1_rel_err_signed | stage1_rel_err_abs | learning_rel | inherited_rel | Gmax_full_over_dw | EXP_root_over_dw | dReach_over_dw |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | mean | 1 | 45.34 | -0.02835 | 0.06763 | -0.02974 | 0.001393 | 0.00487 | 0.002336 | 0.006458 |
| 50 | mean | 4 | 45.28 | -0.02979 | 0.03462 | -0.02829 | 0.001393 | 0.004812 | 0.00197 | 0.005732 |
| 50 | mean | 8 | 45.2 | -0.03138 | 0.03943 | -0.03448 | 0.001393 | 0.004606 | 0.001781 | 0.005545 |
| 50 | mean | 12 | 45.23 | -0.03069 | 0.03434 | -0.03683 | 0.001393 | 0.004606 | 0.001635 | 0.005673 |
| 50 | stochastic | 1 | 47.58 | 0.01962 | 0.06479 | 0.01553 | 0.001393 | 0.004606 | 0.001705 | 0.005795 |
| 50 | stochastic | 4 | 47.18 | 0.01107 | 0.04979 | -0.001145 | 0.001393 | 0.004606 | 0.001726 | 0.006225 |
| 50 | stochastic | 8 | 46.7 | 0.0007315 | 0.04744 | 0.001482 | 0.001393 | 0.004606 | 0.001882 | 0.005691 |
| 50 | stochastic | 12 | 45.64 | -0.02197 | 0.0532 | -0.01993 | 0.001393 | 0.004606 | 0.001943 | 0.005769 |
| 60 | mean | 1 | 37.98 | -0.02346 | 0.08852 | -0.02655 | 0.0006429 | 0.002321 | 0.001452 | 0.003369 |
| 60 | mean | 4 | 40.06 | 0.03012 | 0.05756 | 0.03038 | 0.0006429 | 0.002043 | 0.0009127 | 0.00252 |
| 60 | mean | 8 | 40.9 | 0.05184 | 0.06137 | 0.05171 | 0.0006429 | 0.001833 | 0.0009411 | 0.002687 |
| 60 | mean | 12 | 40.52 | 0.04187 | 0.05933 | 0.04175 | 0.0006429 | 0.001833 | 0.0008846 | 0.002341 |
| 60 | stochastic | 1 | 39.27 | 0.009708 | 0.04948 | 0.007651 | 0.0006429 | 0.001833 | 0.0007316 | 0.002383 |
| 60 | stochastic | 4 | 39.95 | 0.0272 | 0.04061 | 0.02391 | 0.0006429 | 0.001833 | 0.0006055 | 0.002162 |
| 60 | stochastic | 8 | 40.53 | 0.0421 | 0.0421 | 0.04443 | 0.0006429 | 0.001833 | 0.0006545 | 0.002261 |
| 60 | stochastic | 12 | 40.12 | 0.03167 | 0.03167 | 0.02961 | 0.0006429 | 0.001833 | 0.0006495 | 0.002472 |

**Paired K − (K=1)** (from `A/paired_summary.csv`; "better" counts the pairs with K < K=1, out of 10):

| q | arm | K | Δ abs stage-1 err (median) | better | CI95 | Δ EXP_root/ΔW (median) | better | CI95 |
|---|---|---|---|---|---|---|---|---|
| 50 | stochastic | 4 | −0.0196 | 7 | [−0.032, 0.012] | −0.00051 | 6 | [−0.0011, 0.0006] |
| 50 | stochastic | 12 | −0.0125 | 6 | [−0.039, 0.016] | −0.00036 | 7 | [−0.0013, 0.0003] |
| 50 | mean | 4 | −0.0114 | 6 | [−0.025, 0.007] | −0.00023 | 6 | [−0.0011, 0.0001] |
| 50 | mean | 12 | −0.0191 | 6 | [−0.039, 0.013] | −0.00047 | 6 | [−0.0016, 0.00004] |
| 60 | stochastic | 4 | −0.0174 | 6 | [−0.044, 0.014] | −0.00017 | 6 | [−0.0009, 0.0001] |
| 60 | stochastic | 12 | +0.0100 | 4 | [−0.047, 0.028] | +0.00011 | 4 | [−0.0009, 0.0004] |
| 60 | mean | 4 | −0.0363 | 8 | [−0.052, −0.007] | −0.00058 | 8 | [−0.0012, −0.0001] |
| 60 | mean | 12 | −0.0372 | 8 | [−0.067, −0.005] | −0.00072 | 8 | [−0.0017, −0.0002] |

All K, all metrics: `A/paired_summary.csv` rows `family = p3`, `comparison = K_vs_K1`. Ĝmax_full is set by the frozen stage 2 in most runs and is unchanged in most pairs. Its location (t*, d*) is in the table in §2b.

### 1c.2 Phase A extension at u1600, stage 2 (K ∈ {1, 4, 8, 16}; K = 16 covers u1225–u1600)

Stage 1 is untrained in these runs, so no full-policy metric is reported.

Source: `results/v2_pilots/pilot4/analysis/candidates_all.csv`

| q | arm | K | stage2_peak_rel_err_signed | stage2_peak_rel_err_abs | stage2_peak_locfree_rel_err | stage2_rmse_pos_over_g2_0 | stage2_tail_mean | stage2_tail_max | stage2_tail_mean_over_g2_0 | stage2_tail_max_over_g2_0 | stage2_sym_err_max | eta_T_over_dw | DeltaT_over_dw_on_max | DeltaT_over_dw_off_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | ext | 1 | -0.06835 | 0.06835 | -0.06826 | 0.02605 | 0.515 | 2.22 | 0.007358 | 0.03172 | 2.638 | 0.001469 | 0.001469 | 0.0003395 |
| 50 | ext | 4 | -0.06294 | 0.06294 | -0.06247 | 0.0201 | 0.5552 | 2.334 | 0.007931 | 0.03334 | 1.809 | 0.001168 | 0.001168 | 0.0003489 |
| 50 | ext | 8 | -0.07022 | 0.07022 | -0.06972 | 0.01917 | 0.5619 | 2.34 | 0.008027 | 0.03343 | 1.444 | 0.001285 | 0.001285 | 0.0003247 |
| 50 | ext | 16 | -0.0667 | 0.0667 | -0.06667 | 0.01669 | 0.5778 | 2.436 | 0.008255 | 0.03479 | 1.426 | 0.0009973 | 0.0009973 | 0.0003602 |
| 60 | ext | 1 | -0.06003 | 0.06003 | -0.05585 | 0.02389 | 0.5312 | 1.901 | 0.009106 | 0.03259 | 2.39 | 0.0009012 | 0.0009012 | 0.0002439 |
| 60 | ext | 4 | -0.06243 | 0.06243 | -0.06217 | 0.0179 | 0.5404 | 2.064 | 0.009264 | 0.03538 | 1.165 | 0.0007207 | 0.0007207 | 0.0002679 |
| 60 | ext | 8 | -0.05595 | 0.05595 | -0.0551 | 0.01711 | 0.551 | 2.065 | 0.009446 | 0.0354 | 1.453 | 0.0006428 | 0.0006428 | 0.0002787 |
| 60 | ext | 16 | -0.05869 | 0.05869 | -0.05825 | 0.01635 | 0.5782 | 2.029 | 0.009911 | 0.03479 | 1.01 | 0.0005634 | 0.0005634 | 0.0002893 |

**Paired K − (K=1), stage 2** (`A/paired_summary.csv`, `family = ext`):

| q | K | Δ RMSE/e₂*(0) | better | CI95 | Δ sym. err | better | CI95 | Δ tail mean | better | CI95 | Δ η₂ | better | CI95 | Δ |peak| | better | CI95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 4 | −0.0030 | 8 | [−0.015, −0.002] | −0.50 | 7 | [−1.89, −0.10] | +0.018 | 1 | [0.004, 0.035] | −0.00032 | 8 | [−0.0022, 0.0002] | +0.0023 | 4 | [−0.020, 0.012] |
| 50 | 8 | −0.0045 | 8 | [−0.017, −0.002] | −0.78 | 10 | [−1.93, −0.61] | +0.035 | 2 | [0.007, 0.046] | −0.00027 | 6 | [−0.0022, 0.0003] | +0.0029 | 3 | [−0.021, 0.016] |
| 50 | 16 | −0.0079 | 8 | [−0.018, −0.004] | −1.14 | 9 | [−1.89, −0.63] | +0.058 | 2 | [0.030, 0.068] | −0.00071 | 7 | [−0.0023, 0.0001] | +0.0016 | 4 | [−0.018, 0.012] |
| 60 | 4 | −0.0051 | 9 | [−0.011, −0.004] | −0.89 | 9 | [−1.13, −0.44] | −0.005 | 5 | [−0.012, 0.030] | −0.00017 | 7 | [−0.0005, 0.00003] | +0.0076 | 4 | [−0.002, 0.019] |
| 60 | 8 | −0.0058 | 9 | [−0.012, −0.004] | −0.96 | 8 | [−1.27, −0.17] | +0.015 | 4 | [−0.0001, 0.042] | −0.00008 | 9 | [−0.0007, −0.0001] | −0.0005 | 5 | [−0.008, 0.017] |
| 60 | 16 | −0.0078 | 10 | [−0.013, −0.006] | −1.28 | 9 | [−1.47, −0.41] | +0.044 | 3 | [0.015, 0.071] | −0.00017 | 7 | [−0.0007, −0.00004] | +0.0045 | 4 | [−0.005, 0.020] |

All stage-2 metrics (signed and location-free peak, tail max, Δ₂ on/off path): `A/paired_summary.csv`.

---

## 1d. Representation floor of the stage-2 actor (diagnostic only)

This fit is never a training target, never an initialization, and never part of the method.

**Setup** (`tools/v2/pilot4_repr_floor.py`).
- **Network.** The training class `BetaActor`: 2 → 64 tanh → 64 tanh → 2; orthogonal √2 hidden init with bias 0; zero output head; μ = clamp(sigmoid, 1e−6, 1−1e−6); c = 100 + softplus. Input is `encode_obs(2, d)`; float32 throughout.
- **Objective.** Least squares of the Beta mean e_min + 100·α/(α+β) against e₂*(d), with uniform weights on the development D₂ grid: step 4, 101 nodes at q=50 and 111 at q=60.
- **Optimizer.** Full-batch Adam, lr 1e−3, betas (0.9, 0.999), eps 1e−8, no gradient clipping.
- **Plateau rule.** The loss is logged every 1,000 steps. Stop when the best loss has not improved by ≥ 0.1% over 20,000 steps, with a cap of 300,000 steps.
- **Inits.** 5 per q: `torch.Generator().manual_seed(s)`, s = 0…4.

**Deviation: the plateau rule never fired.** All 10 fits stopped at the 300,000-step cap. The loss was still falling, by about 22% over the last 20,000 steps (init 0: 1.38e−3 → 1.07e−3 at q=50, 8.7e−5 → 6.8e−5 at q=60). The numbers below are therefore **upper bounds on the floor** for this optimizer budget, not converged floors.

Source: `results/v2_pilots/pilot4/analysis/repr_floor/fits.csv`

| q | init_seed | steps | stop | final_loss_mse | stage2_peak_rel_err_signed | stage2_peak_locfree_rel_err | stage2_peak_locfree_argmax_d | stage2_rmse_pos_over_g2_0 | stage2_tail_mean | stage2_tail_max | stage2_sym_err_max | eta_T_over_dw | Gmax_full_over_dw | max_abs_resid | max_abs_resid_at_d | tail_min_fit | tail_max_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 0 | 300000 | max_steps | 0.001073 | -2.309e-05 | -2.309e-05 | 0 | 0.004416 | 0.003162 | 0.217 | 1.509 | 3.046e-06 | 3.046e-06 | 0.217 | -100 | 0.0001 | 0.217 |
| 50 | 1 | 300000 | max_steps | 0.000153 | -1.9e-05 | 0.001318 | 0.5 | 0.002146 | 0.001697 | 0.07285 | 0.6851 | 3.314e-07 | 3.314e-07 | 0.07285 | 100 | 0.0001 | 0.07285 |
| 50 | 2 | 300000 | max_steps | 0.0003907 | 1.453e-05 | 1.453e-05 | 0 | 0.004221 | 0.00211 | 0.1383 | 0.4177 | 8.892e-07 | 8.892e-07 | 0.1383 | 100 | 0.0001 | 0.1383 |
| 50 | 3 | 300000 | max_steps | 0.0002663 | -2.003e-05 | 8.849e-06 | 0.5 | 0.004034 | 0.001444 | 0.09482 | 0.3049 | 5.222e-07 | 5.222e-07 | 0.09482 | 100 | 0.0001 | 0.09482 |
| 50 | 4 | 300000 | max_steps | 0.005794 | 0.000759 | 0.000759 | 0 | 0.001855 | 0.007546 | 0.1142 | 0.126 | 8.245e-07 | 8.245e-07 | 0.1142 | 100 | 0.0002335 | 0.1142 |
| 60 | 0 | 300000 | max_steps | 6.847e-05 | -4.632e-06 | -4.632e-06 | 0 | 0.001311 | 0.003303 | 0.05792 | 0.1169 | 2.086e-07 | 2.086e-07 | 0.05792 | 120 | 0.001143 | 0.05792 |
| 60 | 1 | 300000 | max_steps | 0.000538 | -0.0007393 | -0.0007393 | 0 | 0.001623 | 0.003653 | 0.04235 | 0.3997 | 2.356e-07 | 2.356e-07 | 0.04757 | -40 | 0.001066 | 0.04235 |
| 60 | 2 | 300000 | max_steps | 5.414e-05 | 1.006e-05 | 1.006e-05 | 0 | 0.001275 | 0.002987 | 0.03844 | 0.1323 | 9.443e-08 | 9.443e-08 | 0.03844 | 120 | 0.001118 | 0.03844 |
| 60 | 3 | 300000 | max_steps | 0.0001314 | -0.000961 | -0.000961 | 0 | 0.001323 | 0.004828 | 0.05454 | 0.07127 | 4.834e-07 | 4.834e-07 | 0.07087 | 16 | 0.00175 | 0.05454 |
| 60 | 4 | 300000 | max_steps | 0.0002798 | -0.0005967 | -0.0005967 | 0 | 0.00132 | 0.004659 | 0.04804 | 0.0927 | 2.117e-07 | 2.117e-07 | 0.04804 | -120 | 0.002063 | 0.04804 |

Source: `results/v2_pilots/pilot4/analysis/repr_floor/fits.csv`

| q | stage2_peak_rel_err_signed | stage2_peak_locfree_rel_err | stage2_rmse_pos_over_g2_0 | stage2_tail_mean | stage2_tail_max | stage2_sym_err_max | eta_T_over_dw | Gmax_full_over_dw |
|---|---|---|---|---|---|---|---|---|
| 50 | -1.9e-05 | 1.453e-05 | 0.004034 | 0.00211 | 0.1142 | 0.4177 | 8.245e-07 | 8.245e-07 |
| 60 | -0.0005967 | -0.0005967 | 0.00132 | 0.003653 | 0.04804 | 0.1169 | 2.117e-07 | 2.117e-07 |

The same metrics for the RL stage-2 policies at u1600 (extension, last iterate):

Source: `results/v2_pilots/pilot4/analysis/repr_floor/rl_u1600.csv`

| q | stage2_peak_rel_err_signed | stage2_peak_locfree_rel_err | stage2_rmse_pos_over_g2_0 | stage2_tail_mean | stage2_tail_max | stage2_sym_err_max | eta_T_over_dw |
|---|---|---|---|---|---|---|---|
| 50 | -0.06835 | -0.06826 | 0.02605 | 0.515 | 2.22 | 2.638 | 0.001469 |
| 60 | -0.06003 | -0.05585 | 0.02389 | 0.5312 | 1.901 | 2.39 | 0.0009012 |

**Can the head reach the target?**
- In the tails (target 0), the fitted means are 0.001–0.22 effort units (`tail_min_fit`, `tail_max_fit` columns).
- The largest residuals sit at or near the domain edges, d = ±100 (q=50) and ±120 (q=60), and are ≤ 0.22 effort units.
- The head's lower limit is 100·1e−6 = 1e−4 effort units, so the clamp is not binding.
- No region was found where the target is unreachable.

**Three-way comparison of the d = 0 peak gap e₂*(0) − ê₂(0)** (effort units; medians):
- RL: 10 seeds at u1600.
- Smoothing-predicted: the Pilot-1 / extension method (`results/v2_pilots/phaseA_ext/analysis/curves_weights_every25.csv`, u1600).
- Floor: 5 inits.

Source: `results/v2_pilots/pilot4/analysis/repr_floor/three_way_peak_gap.csv`

| q | quantity | n | median | min | max | median_over_e2star0 |
|---|---|---|---|---|---|---|
| 50 | rl_peak_gap_d0 | 10 | 4.784 | 2.619 | 9.002 | 0.06835 |
| 50 | smoothing_predicted_gap_d0 | 10 | 2.159 | 2.05 | 2.59 | 0.03085 |
| 50 | supervised_floor_gap_d0 | 5 | 0.00133 | -0.05313 | 0.001616 | 1.9e-05 |
| 50 | rl_locfree_rel_err | 10 | -0.06826 | -0.1286 | -0.0368 | — |
| 50 | floor_locfree_rel_err | 5 | 1.453e-05 | -2.309e-05 | 0.001318 | — |
| 60 | rl_peak_gap_d0 | 10 | 3.502 | -0.2065 | 4.417 | 0.06003 |
| 60 | smoothing_predicted_gap_d0 | 10 | 1.581 | 1.464 | 1.723 | 0.0271 |
| 60 | supervised_floor_gap_d0 | 5 | 0.0348 | -0.0005871 | 0.05606 | 0.0005967 |
| 60 | rl_locfree_rel_err | 10 | -0.05585 | -0.07572 | 0.003559 | — |
| 60 | floor_locfree_rel_err | 5 | -0.0005967 | -0.000961 | 1.006e-05 | — |

| q | RL peak gap | smoothing-predicted part | supervised-fit floor |
|---|---|---|---|
| 50 | 4.78 (0.068·e₂*(0)) | 2.16 (0.031·e₂*(0)) | 0.0013 (0.00002·e₂*(0)) |
| 60 | 3.50 (0.060·e₂*(0)) | 1.58 (0.027·e₂*(0)) | 0.035 (0.0006·e₂*(0)) |

The location-free peak error of the RL policies (−0.068 and −0.056) is almost equal to the d = 0 error, so the RL peak is not displaced; it is low. For the fits the location-free error is ≤ 0.0013.

---

## 2. Runs

**Launch.**
- tmux sessions `v2_pilot4_A` and `v2_pilot4_B` ran in parallel, 32 workers each, single-threaded processes (OMP/MKL/OpenBLAS = 1).
- `nproc --all` = 64.
- Load average at launch 0.77 / 0.92 / 0.63; at the end of 2a 41.7 / 20.0 / 7.9 (`P4A/launch_20261002_012525.json`, `P4B/launch_20261002_012525.json`).
- 80/80 returncode 0.

Commands are in §9.

### 2a. Phase A end-of-phase decay (parents `state_u01200.pt`; u1201–u1600)

**Reproducibility check** (`tools/v2/pilot4_repro.py` → `A/repro_2a_constant_vs_ext.csv`).
- In **20/20** runs, the `constant` arm is **bit-identical** to the extension over u1201–u1600:
  - all 16 weight exports;
  - the final weights against the extension's `u01600.npz`;
  - `state_u01600.pt` (actor, critic, opponent, both Adam states, minibatch RNG);
  - every RNG state;
  - the full per-update training history (excluding timing);
  - the per-update RNG positions of all five streams.
- The verifier-call updates also coincide: u1300, 1400, 1500 and 1600 in every run. The only difference is the reason recorded at u1300 (`warmup_forced` on re-entry vs `timeout`). This is because both re-entry points (u400 for the extension, u1200 here) are multiples of 100.
- So this comparison alone cannot show that the verifier consumes no RNG. A **direct check** was run instead (`A/verifier_consumes_no_rng.csv`). For each of the 20 parents, the restored run's RNG states were recorded before and after six verifier calls (dev + final tier, three times). The states covered numpy env/learn/opp/start, the minibatch stream, the torch generator, and the torch/numpy/python global states. They were **unchanged in 20/20**.

**Run records.**

Source: `results/v2_pilots/pilot4/analysis/run_records.csv`

| q | arm | n | commit | dirty | lr_first | lr_last | n_would_fire | wf_median | wf_min | wf_max | wall_median | kl_median | clip_median | adv_s1_std | adv_used_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | constant | 10 | c92ee74 | False | 0.0003 | 0.0003 | 10 | 1500 | 1500 | 1500 | 69.4 | 0.006153 | 0.06694 | — | 0.04867 |
| 50 | decay | 10 | c92ee74 | False | 0.0003 | 3e-05 | 10 | 1500 | 1460 | 1500 | 70.58 | 0.004399 | 0.04634 | — | 0.04852 |
| 60 | constant | 10 | c92ee74 | False | 0.0003 | 0.0003 | 10 | 1500 | 1500 | 1500 | 69.23 | 0.006121 | 0.07466 | — | 0.04248 |
| 60 | decay | 10 | c92ee74 | False | 0.0003 | 3e-05 | 10 | 1500 | 1480 | 1500 | 69.13 | 0.004414 | 0.04883 | — | 0.043 |

`would_fire` is the existing Phase-A rule, with counters restarted at u1200.

**Final metrics (u1600, last iterate, all runs).**

Source: `results/v2_pilots/pilot4/analysis/candidates_all.csv`

| q | seed | arm | stage2_peak_rel_err_signed | stage2_peak_rel_err_abs | stage2_peak_locfree_rel_err | stage2_rmse_pos_over_g2_0 | stage2_tail_mean | stage2_tail_max | stage2_tail_mean_over_g2_0 | stage2_tail_max_over_g2_0 | stage2_sym_err_max | eta_T_over_dw | DeltaT_over_dw_on_max | DeltaT_over_dw_off_max | sigma_effort_at_0_t2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10501 | constant | -0.07134 | 0.07134 | -0.07134 | 0.03371 | 0.566 | 4.907 | 0.008086 | 0.07009 | 5.473 | 0.003373 | 0.003373 | 0.001545 | 3.28 |
| 50 | 10501 | decay | -0.1222 | 0.1222 | -0.1221 | 0.03966 | 0.5229 | 4.28 | 0.007471 | 0.06114 | 3.276 | 0.004014 | 0.004014 | 0.001155 | 3.446 |
| 50 | 10502 | constant | -0.05659 | 0.05659 | -0.05545 | 0.02527 | 0.8831 | 2.665 | 0.01262 | 0.03807 | 2.903 | 0.001985 | 0.001985 | 0.0003671 | 2.63 |
| 50 | 10502 | decay | -0.06803 | 0.06803 | -0.06719 | 0.01974 | 0.8309 | 3.004 | 0.01187 | 0.04291 | 1.813 | 0.0009844 | 0.0009844 | 0.000472 | 2.732 |
| 50 | 10503 | constant | -0.05913 | 0.05913 | -0.05883 | 0.01422 | 0.4423 | 2.085 | 0.006319 | 0.02978 | 1.469 | 0.0007199 | 0.0007199 | 0.0003052 | 2.781 |
| 50 | 10503 | decay | -0.02357 | 0.02357 | -0.02246 | 0.01997 | 0.4841 | 2.489 | 0.006916 | 0.03555 | 1.47 | 0.001639 | 0.001639 | 0.0004409 | 2.818 |
| 50 | 10504 | constant | -0.07236 | 0.07236 | -0.07113 | 0.02924 | 0.5181 | 2.173 | 0.007401 | 0.03104 | 2.628 | 0.001237 | 0.001237 | 0.0002502 | 2.674 |
| 50 | 10504 | decay | -0.07125 | 0.07125 | -0.07111 | 0.01909 | 0.602 | 2.402 | 0.0086 | 0.03432 | 2.359 | 0.001159 | 0.001159 | 0.000328 | 2.772 |
| 50 | 10505 | constant | -0.06799 | 0.06799 | -0.06799 | 0.02683 | 0.512 | 2.395 | 0.007314 | 0.03421 | 1.447 | 0.00258 | 0.00258 | 0.0003923 | 2.642 |
| 50 | 10505 | decay | -0.05432 | 0.05432 | -0.05385 | 0.02018 | 0.5821 | 2.218 | 0.008316 | 0.03168 | 2.471 | 0.0009195 | 0.0009195 | 0.0002605 | 2.703 |
| 50 | 10506 | constant | -0.0687 | 0.0687 | -0.06852 | 0.02068 | 0.4967 | 2.117 | 0.007095 | 0.03024 | 2.649 | 0.001048 | 0.001048 | 0.0003199 | 3.005 |
| 50 | 10506 | decay | -0.09843 | 0.09843 | -0.09831 | 0.02399 | 0.5423 | 2.762 | 0.007748 | 0.03946 | 1.787 | 0.002643 | 0.002643 | 0.0005444 | 3.145 |
| 50 | 10507 | constant | -0.06958 | 0.06958 | -0.06958 | 0.02189 | 0.6452 | 2.925 | 0.009217 | 0.04178 | 3.737 | 0.00108 | 0.00108 | 0.0005526 | 2.69 |
| 50 | 10507 | decay | -0.05776 | 0.05776 | -0.05668 | 0.0266 | 0.6903 | 2.652 | 0.009862 | 0.03788 | 4.862 | 0.001026 | 0.001026 | 0.0005013 | 2.748 |
| 50 | 10508 | constant | -0.1286 | 0.1286 | -0.1286 | 0.05976 | 0.3528 | 2.242 | 0.00504 | 0.03203 | 2.085 | 0.006971 | 0.006971 | 0.000359 | 3.085 |
| 50 | 10508 | decay | -0.06563 | 0.06563 | -0.0655 | 0.02341 | 0.3466 | 2.257 | 0.004951 | 0.03225 | 1.266 | 0.001002 | 0.001002 | 0.0003631 | 3.111 |
| 50 | 10509 | constant | -0.05643 | 0.05643 | -0.05575 | 0.02274 | 0.3906 | 2.198 | 0.005579 | 0.0314 | 1.627 | 0.001296 | 0.001296 | 0.0002461 | 2.915 |
| 50 | 10509 | decay | -0.03538 | 0.03538 | -0.03526 | 0.01742 | 0.4467 | 2.689 | 0.006381 | 0.03841 | 2.859 | 0.001138 | 0.001138 | 0.000385 | 2.994 |
| 50 | 10510 | constant | -0.03741 | 0.03741 | -0.0368 | 0.02713 | 0.5574 | 1.656 | 0.007962 | 0.02365 | 3.799 | 0.001642 | 0.001642 | 0.0001463 | 2.598 |
| 50 | 10510 | decay | -0.05937 | 0.05937 | -0.05937 | 0.01502 | 0.642 | 2.517 | 0.009171 | 0.03595 | 2.171 | 0.0007256 | 0.0007256 | 0.0004498 | 2.711 |
| 60 | 10501 | constant | -0.02907 | 0.02907 | -0.02775 | 0.03376 | 0.5004 | 2.32 | 0.008578 | 0.03977 | 2.558 | 0.001006 | 0.001006 | 0.0003005 | 2.985 |
| 60 | 10501 | decay | -0.04693 | 0.04693 | -0.04683 | 0.01333 | 0.4856 | 2.11 | 0.008325 | 0.03617 | 1.143 | 0.0003602 | 0.0003602 | 0.000314 | 3.078 |
| 60 | 10502 | constant | 0.003541 | 0.003541 | 0.003559 | 0.0318 | 0.5769 | 1.743 | 0.009889 | 0.02988 | 1.199 | 0.002156 | 0.002156 | 0.000217 | 2.721 |
| 60 | 10502 | decay | -0.03193 | 0.03193 | -0.03149 | 0.0182 | 0.6296 | 2.26 | 0.01079 | 0.03874 | 1.693 | 0.0005909 | 0.0005909 | 0.0003609 | 2.841 |
| 60 | 10503 | constant | -0.03993 | 0.03993 | -0.03967 | 0.03087 | 0.6248 | 2.556 | 0.01071 | 0.04382 | 2.411 | 0.0006834 | 0.0006834 | 0.000465 | 3.025 |
| 60 | 10503 | decay | -0.07216 | 0.07216 | -0.07202 | 0.02393 | 0.5382 | 2.134 | 0.009227 | 0.03658 | 3.357 | 0.0008517 | 0.0008517 | 0.0002514 | 3.159 |
| 60 | 10504 | constant | -0.07134 | 0.07134 | -0.07039 | 0.02505 | 0.5342 | 2.078 | 0.009157 | 0.03563 | 2.467 | 0.0009013 | 0.0009013 | 0.0002812 | 2.704 |
| 60 | 10504 | decay | -0.04855 | 0.04855 | -0.04838 | 0.02267 | 0.5996 | 2.113 | 0.01028 | 0.03621 | 1.373 | 0.0009172 | 0.0009172 | 0.0003167 | 2.793 |
| 60 | 10505 | constant | -0.07572 | 0.07572 | -0.07572 | 0.02274 | 0.4143 | 1.925 | 0.007102 | 0.033 | 1.238 | 0.0009715 | 0.0009715 | 0.0002393 | 3.135 |
| 60 | 10505 | decay | -0.07591 | 0.07591 | -0.07558 | 0.01628 | 0.5128 | 2.507 | 0.008791 | 0.04297 | 0.9295 | 0.00098 | 0.00098 | 0.0003399 | 3.23 |
| 60 | 10506 | constant | -0.07423 | 0.07423 | -0.07422 | 0.01864 | 0.6374 | 1.825 | 0.01093 | 0.03129 | 0.8313 | 0.0009011 | 0.0009011 | 0.0002377 | 2.671 |
| 60 | 10506 | decay | -0.04289 | 0.04289 | -0.04223 | 0.01728 | 0.7317 | 2.057 | 0.01254 | 0.03527 | 1.593 | 0.0003829 | 0.0003829 | 0.0003012 | 2.738 |
| 60 | 10507 | constant | -0.04872 | 0.04872 | -0.04815 | 0.02106 | 0.5345 | 1.708 | 0.009163 | 0.02927 | 2.651 | 0.0003882 | 0.0003882 | 0.0001492 | 2.76 |
| 60 | 10507 | decay | -0.05151 | 0.05151 | -0.05136 | 0.01729 | 0.5576 | 2.068 | 0.009559 | 0.03544 | 2.266 | 0.000434 | 0.000434 | 0.0002273 | 2.846 |
| 60 | 10508 | constant | -0.07179 | 0.07179 | -0.06356 | 0.03297 | 0.4881 | 1.865 | 0.008367 | 0.03197 | 3.963 | 0.0008429 | 0.0008429 | 0.0002485 | 3.142 |
| 60 | 10508 | decay | -0.09174 | 0.09174 | -0.09052 | 0.02454 | 0.61 | 2.318 | 0.01046 | 0.03974 | 2.833 | 0.001497 | 0.001497 | 0.0003812 | 3.242 |
| 60 | 10509 | constant | -0.07492 | 0.07492 | -0.07459 | 0.02083 | 0.5027 | 1.877 | 0.008618 | 0.03218 | 2.354 | 0.000919 | 0.000919 | 0.0001825 | 2.874 |
| 60 | 10509 | decay | -0.05088 | 0.05088 | -0.05048 | 0.02145 | 0.5067 | 1.879 | 0.008686 | 0.03222 | 1.952 | 0.0007645 | 0.0007645 | 0.000238 | 2.978 |
| 60 | 10510 | constant | -0.04158 | 0.04158 | -0.03975 | 0.01872 | 0.5282 | 2.12 | 0.009056 | 0.03634 | 2.369 | 0.0004406 | 0.0004406 | 0.0003207 | 2.892 |
| 60 | 10510 | decay | -0.02097 | 0.02097 | -0.02071 | 0.02651 | 0.5035 | 2.034 | 0.008632 | 0.03486 | 1.037 | 0.001453 | 0.001453 | 0.0002849 | 2.999 |

**Tail-averaged candidates** (K ∈ {1, 4, 8}):

Source: `results/v2_pilots/pilot4/analysis/candidates_all.csv`

| q | arm | K | stage2_peak_rel_err_signed | stage2_peak_rel_err_abs | stage2_peak_locfree_rel_err | stage2_rmse_pos_over_g2_0 | stage2_tail_mean | stage2_tail_max | stage2_tail_mean_over_g2_0 | stage2_tail_max_over_g2_0 | stage2_sym_err_max | eta_T_over_dw | DeltaT_over_dw_on_max | DeltaT_over_dw_off_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | constant | 1 | -0.06835 | 0.06835 | -0.06826 | 0.02605 | 0.515 | 2.22 | 0.007358 | 0.03172 | 2.638 | 0.001469 | 0.001469 | 0.0003395 |
| 50 | constant | 4 | -0.06294 | 0.06294 | -0.06247 | 0.0201 | 0.5552 | 2.334 | 0.007931 | 0.03334 | 1.809 | 0.001168 | 0.001168 | 0.0003489 |
| 50 | constant | 8 | -0.07022 | 0.07022 | -0.06972 | 0.01917 | 0.5619 | 2.34 | 0.008027 | 0.03343 | 1.444 | 0.001285 | 0.001285 | 0.0003247 |
| 50 | decay | 1 | -0.0625 | 0.0625 | -0.06243 | 0.02007 | 0.5622 | 2.584 | 0.008032 | 0.03692 | 2.265 | 0.001082 | 0.001082 | 0.0004454 |
| 50 | decay | 4 | -0.06911 | 0.06911 | -0.06831 | 0.01735 | 0.5905 | 2.616 | 0.008436 | 0.03737 | 1.622 | 0.001036 | 0.001036 | 0.0004245 |
| 50 | decay | 8 | -0.06673 | 0.06673 | -0.06634 | 0.01713 | 0.5991 | 2.501 | 0.008559 | 0.03572 | 1.583 | 0.001126 | 0.001126 | 0.0003807 |
| 60 | constant | 1 | -0.06003 | 0.06003 | -0.05585 | 0.02389 | 0.5312 | 1.901 | 0.009106 | 0.03259 | 2.39 | 0.0009012 | 0.0009012 | 0.0002439 |
| 60 | constant | 4 | -0.06243 | 0.06243 | -0.06217 | 0.0179 | 0.5404 | 2.064 | 0.009264 | 0.03538 | 1.165 | 0.0007207 | 0.0007207 | 0.0002679 |
| 60 | constant | 8 | -0.05595 | 0.05595 | -0.0551 | 0.01711 | 0.551 | 2.065 | 0.009446 | 0.0354 | 1.453 | 0.0006428 | 0.0006428 | 0.0002787 |
| 60 | decay | 1 | -0.04971 | 0.04971 | -0.04943 | 0.01982 | 0.5479 | 2.111 | 0.009393 | 0.03619 | 1.643 | 0.0008081 | 0.0008081 | 0.0003076 |
| 60 | decay | 4 | -0.05929 | 0.05929 | -0.059 | 0.01774 | 0.5726 | 2.122 | 0.009815 | 0.03637 | 1.455 | 0.0006526 | 0.0006526 | 0.0002998 |
| 60 | decay | 8 | -0.05884 | 0.05884 | -0.05858 | 0.01627 | 0.5719 | 2.122 | 0.009804 | 0.03638 | 1.557 | 0.0005716 | 0.0005716 | 0.0003144 |

**Paired differences decay − constant per (q, seed).** "Better" counts decay < constant for smaller-is-better metrics, out of 10. Signed errors have no preferred direction; their −/+ counts are given.

| q | metric | median | better | CI95 |
|---|---|---|---|---|
| 50 | |peak err| | −0.0065 | 6 | [−0.023, 0.016] |
| 50 | peak err (signed) | +0.0065 | (4−/6+) | [−0.016, 0.023] |
| 50 | RMSE/e₂*(0) | −0.0054 | 6 | [−0.014, 0.0009] |
| 50 | tail mean | +0.045 | 3 | [0.0016, 0.060] |
| 50 | tail max | +0.28 | 3 | [−0.077, 0.455] |
| 50 | symmetry err | −0.54 | 6 | [−1.03, 0.35] |
| 50 | η₂ | −0.00012 | 7 | [−0.0021, 0.0004] |
| 50 | Δ₂ off-path max | +0.00009 | 3 | [−0.00008, 0.00015] |
| 60 | |peak err| | +0.0015 | 4 | [−0.014, 0.014] |
| 60 | peak err (signed) | −0.0015 | (6−/4+) | [−0.015, 0.013] |
| 60 | RMSE/e₂*(0) | −0.0051 | 8 | [−0.010, −0.0011] |
| 60 | tail mean | +0.038 | 3 | [−0.006, 0.070] |
| 60 | tail max | +0.13 | 3 | [−0.053, 0.338] |
| 60 | symmetry err | −0.39 | 7 | [−0.88, 0.13] |
| 60 | η₂ | +0.00001 | 4 | [−0.0005, 0.0003] |
| 60 | Δ₂ off-path max | +0.00006 | 2 | [−0.00003, 0.00009] |

The same comparison at K = 4 and K = 8, and for location-free peak, tail/e₂*(0) and on-path Δ₂, is in `A/paired_summary.csv` (`family = 2a`, `comparison = decay_minus_constant`). Run-level paired differences:

Source: `results/v2_pilots/pilot4/analysis/paired_summary_run_records.csv`

| family | q | metric | median | n_better | n_neg | n_pos | boot_ci95_lo | boot_ci95_hi |
|---|---|---|---|---|---|---|---|---|
| 2a | 50 | kl_final_epoch | -0.006124 | — | 10 | 0 | -0.008637 | -0.004025 |
| 2a | 50 | clip_frac | -0.02871 | — | 7 | 3 | -0.0618 | -0.01422 |
| 2a | 50 | phase_wall_sec | 0.7803 | 2 | 2 | 8 | -3.388 | 2.966 |
| 2a | 60 | kl_final_epoch | -0.00175 | — | 7 | 3 | -0.008996 | 3.337e-05 |
| 2a | 60 | clip_frac | -0.0708 | — | 10 | 0 | -0.1251 | -0.05746 |
| 2a | 60 | phase_wall_sec | -0.04 | 6 | 6 | 4 | -0.3521 | 7.167 |
| 2b | 50 | kl_final_epoch | -0.003552 | — | 6 | 4 | -0.01787 | -0.00051 |
| 2b | 50 | clip_frac | -0.07348 | — | 9 | 1 | -0.1651 | -0.04704 |
| 2b | 50 | phase_wall_sec | 1.312 | 4 | 4 | 6 | -3.506 | 7.346 |
| 2b | 50 | within_run_sd_e1_last5 | -0.7811 | 7 | 7 | 3 | -1.215 | -0.2705 |
| 2b | 50 | within_run_range_e1_last5 | -2.115 | 7 | 7 | 3 | -2.957 | -0.4928 |
| 2b | 60 | kl_final_epoch | -0.007815 | — | 7 | 3 | -0.01483 | -0.002866 |
| 2b | 60 | clip_frac | -0.08662 | — | 9 | 1 | -0.1371 | -0.06401 |
| 2b | 60 | phase_wall_sec | -0.82 | 6 | 6 | 4 | -2.107 | 5.974 |
| 2b | 60 | within_run_sd_e1_last5 | -0.5096 | 6 | 6 | 4 | -1.198 | -0.07722 |
| 2b | 60 | within_run_range_e1_last5 | -1.284 | 6 | 6 | 4 | -2.984 | -0.2492 |

**Learning curves.** Median and IQR over 10 seeds, every export u1225–u1600: `reports/v2/figures/pilot4/curves_2a_q50.png`, `curves_2a_q60.png`. Data: `A/candidates_all.csv`, rows `family = 2a`, `kind = curve|tail`, `K = 1`.

### 2b. Phase B decay (parents `state_u01600.pt`; stage 2 frozen at u1600; u1601–u2200)

**Settings.** B2 normalization (`adv_norm_scope = stage1_rows`), `continuation_action_mode = mean`, `reward_mode = expected`, root starts, lagged opponent refreshed every 20 updates, 600 updates. Snapshot integrity: `drift_test` passes in 40/40, with mean/α/β drift exactly 0 (`A/run_records.csv`, columns `drift_test_pass`, `snapshot_drift_max`).

**Induced-target bands for the 20 new parents** (`results/v2_pilots/induced_band/parent4_bands.csv`; sweeps in `.../sweeps/parent4_q*_s*.npz`).
- Final tier, step 0.01, anchored at e₁*, same floor as before (`floors.json`).
- Range per parent: [min(0.4·e₁*, e₁,min − 2), max(1.7·e₁*, e₁,max + 2)]. Here e₁,min and e₁,max run over every training-time checkpoint and every weight export of both 2b arms of that (q, seed).
- Ranges used:
  - q=50: [18.67, 79.34–85.59], i.e. up to 1.834·e₁*;
  - q=60: [15.55, 66.12–76.17], i.e. up to 1.959·e₁*.
- **0 of 1,080 export rows** fall outside their sweep (`A/candidates_all.csv`, column `e1_inside_sweep`); the checkpoint rows are inside by construction of the range.
- ẽ₁ ranges over 46.12–47.30 (q=50) and 38.45–39.03 (q=60).
- 7 of the 10 q=50 bands are not contiguous (`band_contiguous`); the outermost points are used, as before.
- No argmin lies on a sweep edge.
- Median inherited term: −0.0030 (q=50), −0.0033 (q=60).

**Run records.**

Source: `results/v2_pilots/pilot4/analysis/run_records.csv`

| q | arm | n | commit | dirty | lr_first | lr_last | n_would_fire | wf_median | wf_min | wf_max | wall_median | kl_median | clip_median | adv_s1_std | adv_used_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | constant | 10 | c92ee74 | False | 0.0003 | 0.0003 | 10 | 1762 | 1750 | 1925 | 153.1 | 0.004271 | 0.08654 | 1.214 | 1.214 |
| 50 | decay | 10 | c92ee74 | False | 0.0003 | 3e-05 | 10 | 1775 | 1750 | 1925 | 155.9 | 0.003357 | 0.05102 | 1.217 | 1.217 |
| 60 | constant | 10 | c92ee74 | False,True | 0.0003 | 0.0003 | 10 | 1750 | 1750 | 1850 | 144.7 | 0.004525 | 0.09883 | 1.182 | 1.182 |
| 60 | decay | 10 | c92ee74 | False,True | 0.0003 | 3e-05 | 10 | 1750 | 1750 | 1850 | 151.4 | 0.003417 | 0.05912 | 1.182 | 1.182 |

**Final checkpoint u2200, all 40 runs.**
- e₁ cand = ê₁(0).
- learning and inherited terms are /e₁*, with bands.
- SD/range last5 is the within-run SD and range of ê₁(0) over the exports u2100, 2125, 2150, 2175 and 2200.
- Ĝmax, EXP, dReach, Δmax_all and dFull are /ΔW.
- KL and clip are from the last update.

Source: `results/v2_pilots/pilot4/analysis/candidates_all.csv + run_records.csv`

| q | seed | arm | e1_cand | stage1_rel_err_signed | learning_rel | learning_band | inherited_rel | inherited_band | sigma_effort_at_0_t1 | within_run_sd_e1_last5 | within_run_range_e1_last5 | Gmax_full_over_dw | Gmax_full_t | Gmax_full_d | EXP_root_over_dw | dReach_over_dw | Deltamax_all_over_dw | dFull_over_dw | kl_final_epoch | clip_frac | phase_wall_sec | would_fire_update |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10501 | constant | 47.92 | 0.02692 | 0.01342 | [0.0106, 0.0134] | 0.0135 | [0.0135, 0.0163] | 3.483 | 1.662 | 3.63 | 0.003373 | 2 | 32 | 0.001325 | 0.003378 | 0.003373 | 0.003378 | 0.008529 | 0.3595 | 153.4 | 1925 |
| 50 | 10501 | decay | 49.67 | 0.06427 | 0.05077 | [0.0480, 0.0508] | 0.0135 | [0.0135, 0.0163] | 3.565 | 1.615 | 4.134 | 0.003373 | 2 | 32 | 0.001972 | 0.004021 | 0.003373 | 0.004021 | 0.001559 | 0.006699 | 155.8 | 1925 |
| 50 | 10502 | constant | 50.27 | 0.07724 | 0.07532 | [0.0702, 0.0753] | 0.001929 | [0.0019, 0.0071] | 2.662 | 2.192 | 5.477 | 0.001985 | 2 | -60 | 0.001832 | 0.003451 | 0.001985 | 0.003451 | 0.007622 | 0.07101 | 147.4 | 1750 |
| 50 | 10502 | decay | 50.77 | 0.08802 | 0.08609 | [0.0809, 0.0861] | 0.001929 | [0.0019, 0.0071] | 2.732 | 1.024 | 2.67 | 0.002369 | 1 | 0 | 0.002369 | 0.003994 | 0.002009 | 0.003994 | 0.0001787 | 0 | 166 | 1750 |
| 50 | 10503 | constant | 44.7 | -0.04215 | -0.03443 | [-0.0385, -0.0344] | -0.007714 | [-0.0077, -0.0036] | 2.864 | 1.884 | 5.002 | 0.0007199 | 2 | 0 | 0.0004443 | 0.001049 | 0.0007199 | 0.001049 | 0.0005474 | 0.05013 | 158.3 | 1900 |
| 50 | 10503 | decay | 45.83 | -0.01801 | -0.0103 | [-0.0144, -0.0103] | -0.007714 | [-0.0077, -0.0036] | 2.961 | 0.6204 | 1.594 | 0.0007199 | 2 | 0 | 0.0001825 | 0.0007861 | 0.0007199 | 0.0007861 | 0.002811 | 0.001726 | 157.7 | 1875 |
| 50 | 10504 | constant | 50.24 | 0.07661 | 0.06611 | [0.0623, 0.0661] | 0.0105 | [0.0105, 0.0144] | 2.665 | 1.1 | 2.874 | 0.001423 | 1 | 0 | 0.001423 | 0.002307 | 0.001237 | 0.002307 | 0.002541 | 0.1099 | 145.3 | 1750 |
| 50 | 10504 | decay | 47.6 | 0.01998 | 0.00948 | [0.0056, 0.0095] | 0.0105 | [0.0105, 0.0144] | 2.736 | 1.136 | 2.556 | 0.001237 | 2 | -4 | 0.0003404 | 0.001237 | 0.001237 | 0.001237 | 0.008082 | 0.09091 | 142.5 | 1750 |
| 50 | 10505 | constant | 43.77 | -0.06216 | -0.05958 | [-0.0649, -0.0596] | -0.002571 | [-0.0026, 0.0028] | 2.605 | 1.444 | 3.587 | 0.00258 | 2 | -24 | 0.001497 | 0.003581 | 0.00258 | 0.003581 | 0.001812 | 0.0759 | 151 | 1750 |
| 50 | 10505 | decay | 44.16 | -0.05369 | -0.05111 | [-0.0565, -0.0511] | -0.002571 | [-0.0026, 0.0028] | 2.691 | 0.8607 | 2.017 | 0.00258 | 2 | -24 | 0.001226 | 0.003311 | 0.00258 | 0.003311 | 0.001678 | 0.006072 | 158 | 1750 |
| 50 | 10506 | constant | 54.96 | 0.1777 | 0.1895 | [0.1841, 0.1895] | -0.01179 | [-0.0118, -0.0064] | 3.119 | 2.787 | 5.743 | 0.01079 | 1 | 0 | 0.01079 | 0.01162 | 0.01058 | 0.01162 | 0.04471 | 0.207 | 156.4 | 1850 |
| 50 | 10506 | decay | 43.1 | -0.07646 | -0.06468 | [-0.0700, -0.0647] | -0.01179 | [-0.0118, -0.0064] | 3.187 | 1.007 | 2.436 | 0.001374 | 1 | 0 | 0.001374 | 0.002229 | 0.001181 | 0.002229 | 0.0005444 | 0.04531 | 143.2 | 1850 |
| 50 | 10507 | constant | 51.2 | 0.09724 | 0.1088 | [0.1052, 0.1088] | -0.01157 | [-0.0116, -0.0079] | 2.675 | 2.112 | 5.797 | 0.003423 | 1 | 0 | 0.003423 | 0.004305 | 0.003225 | 0.004305 | 0.0003759 | 0.1103 | 152.8 | 1750 |
| 50 | 10507 | decay | 47.59 | 0.0198 | 0.03137 | [0.0277, 0.0314] | -0.01157 | [-0.0116, -0.0079] | 2.783 | 0.1915 | 0.4713 | 0.00108 | 2 | -4 | 0.0004071 | 0.001284 | 0.00108 | 0.001284 | 0.001246 | 0 | 155.7 | 1750 |
| 50 | 10508 | constant | 49.63 | 0.06346 | 0.06689 | [0.0626, 0.0669] | -0.003429 | [-0.0034, 0.0009] | 3.129 | 1.281 | 3.456 | 0.006971 | 2 | -32 | 0.004072 | 0.008039 | 0.006971 | 0.008039 | 0.02404 | 0.1331 | 144.3 | 1775 |
| 50 | 10508 | decay | 44.71 | -0.04185 | -0.03842 | [-0.0427, -0.0384] | -0.003429 | [-0.0034, 0.0009] | 3.206 | 0.3024 | 0.7955 | 0.006971 | 2 | -32 | 0.003149 | 0.007311 | 0.006971 | 0.007311 | 0.003117 | 0.05569 | 156 | 1800 |
| 50 | 10509 | constant | 48.27 | 0.03433 | 0.04268 | [0.0380, 0.0427] | -0.008357 | [-0.0084, -0.0036] | 3.03 | 1.846 | 3.96 | 0.001296 | 2 | -40 | 0.0007628 | 0.00173 | 0.001296 | 0.00173 | 0.01332 | 0.0818 | 162.3 | 1825 |
| 50 | 10509 | decay | 48.74 | 0.04449 | 0.05285 | [0.0481, 0.0528] | -0.008357 | [-0.0084, -0.0036] | 3.094 | 2.01 | 5.286 | 0.001296 | 2 | -40 | 0.001035 | 0.002001 | 0.001296 | 0.002001 | 0.001154 | 0.005847 | 154.2 | 1825 |
| 50 | 10510 | constant | 46 | -0.01437 | -0.0253 | [-0.0311, -0.0253] | 0.01093 | [0.0109, 0.0167] | 2.553 | 1.052 | 2.913 | 0.001642 | 2 | -16 | 0.0006243 | 0.001842 | 0.001642 | 0.001842 | 0.001157 | 0.03785 | 157 | 1750 |
| 50 | 10510 | decay | 47.82 | 0.02479 | 0.01386 | [0.0081, 0.0139] | 0.01093 | [0.0109, 0.0167] | 2.677 | 1.274 | 3.236 | 0.001642 | 2 | -16 | 0.0004369 | 0.001649 | 0.001642 | 0.001649 | 0.003737 | 0.04741 | 157.3 | 1750 |
| 60 | 10501 | constant | 41.16 | 0.05846 | 0.06978 | [0.0677, 0.0703] | -0.01131 | [-0.0118, -0.0093] | 2.858 | 1.86 | 4.445 | 0.001006 | 2 | -20 | 0.0009515 | 0.001678 | 0.001006 | 0.001678 | 0.0001008 | 0.1202 | 152.6 | 1750 |
| 60 | 10501 | decay | 36.45 | -0.06274 | -0.05143 | [-0.0535, -0.0509] | -0.01131 | [-0.0118, -0.0093] | 2.933 | 0.3592 | 0.9205 | 0.001006 | 2 | -20 | 0.0006652 | 0.001409 | 0.001006 | 0.001409 | 0.004872 | 0.02707 | 154 | 1750 |
| 60 | 10502 | constant | 37.72 | -0.03005 | -0.02953 | [-0.0311, -0.0290] | -0.0005143 | [-0.0010, 0.0010] | 2.55 | 2.171 | 5.725 | 0.002156 | 2 | -16 | 0.0006679 | 0.002329 | 0.002156 | 0.002329 | 0.003401 | 0.07872 | 162.2 | 1750 |
| 60 | 10502 | decay | 40.26 | 0.03534 | 0.03585 | [0.0343, 0.0364] | -0.0005143 | [-0.0010, 0.0010] | 2.697 | 0.2731 | 0.6438 | 0.002156 | 2 | -16 | 0.0006981 | 0.002352 | 0.002156 | 0.002352 | 0.004684 | 0.01277 | 160.6 | 1750 |
| 60 | 10503 | constant | 39.02 | 0.003319 | 0.007433 | [0.0056, 0.0077] | -0.004114 | [-0.0044, -0.0023] | 2.939 | 2.403 | 5.675 | 0.0006834 | 2 | -20 | 0.0002893 | 0.0006955 | 0.0006834 | 0.0006955 | 0.01618 | 0.08706 | 154.1 | 1850 |
| 60 | 10503 | decay | 41.26 | 0.06087 | 0.06498 | [0.0632, 0.0652] | -0.004114 | [-0.0044, -0.0023] | 3.08 | 0.9357 | 2.462 | 0.0008746 | 1 | 0 | 0.0008746 | 0.001273 | 0.0006834 | 0.001273 | 0.001001 | 0.00961 | 150.3 | 1850 |
| 60 | 10504 | constant | 33.58 | -0.1366 | -0.1258 | [-0.1279, -0.1251] | -0.0108 | [-0.0116, -0.0087] | 2.445 | 1.045 | 2.798 | 0.002378 | 1 | 0 | 0.002378 | 0.003088 | 0.002186 | 0.003088 | 0.00949 | 0.07687 | 144.4 | 1750 |
| 60 | 10504 | decay | 35.75 | -0.08066 | -0.06986 | [-0.0719, -0.0691] | -0.0108 | [-0.0116, -0.0087] | 2.562 | 1.297 | 3.064 | 0.0009013 | 2 | -28 | 0.0008868 | 0.001587 | 0.0009013 | 0.001587 | 0.003683 | 0.01826 | 161.6 | 1750 |
| 60 | 10505 | constant | 44.43 | 0.1424 | 0.1442 | [0.1419, 0.1445] | -0.0018 | [-0.0021, 0.0005] | 3.15 | 1.428 | 3.742 | 0.00295 | 1 | 0 | 0.00295 | 0.003774 | 0.002802 | 0.003774 | 0.03449 | 0.2457 | 157.8 | 1825 |
| 60 | 10505 | decay | 39.38 | 0.01261 | 0.01441 | [0.0121, 0.0147] | -0.0018 | [-0.0021, 0.0005] | 3.157 | 0.5676 | 1.409 | 0.0009715 | 2 | -4 | 0.0001646 | 0.0009939 | 0.0009715 | 0.0009939 | 0.006137 | 0.009487 | 152.5 | 1750 |
| 60 | 10506 | constant | 39.71 | 0.02113 | 0.02293 | [0.0206, 0.0232] | -0.0018 | [-0.0021, 0.0005] | 2.487 | 2.813 | 6.654 | 0.0009011 | 2 | 0 | 0.0001762 | 0.0009632 | 0.0009011 | 0.0009632 | 0.005269 | 0.1505 | 144.9 | 1750 |
| 60 | 10506 | decay | 37.63 | -0.03228 | -0.03048 | [-0.0328, -0.0302] | -0.0018 | [-0.0021, 0.0005] | 2.541 | 1.102 | 2.292 | 0.0009011 | 2 | 0 | 0.0002842 | 0.001076 | 0.0009011 | 0.001076 | 0.003331 | 0.01396 | 154.7 | 1750 |
| 60 | 10507 | constant | 39 | 0.00297 | -0.0006299 | [-0.0024, -0.0004] | 0.0036 | [0.0033, 0.0054] | 2.64 | 0.6151 | 1.448 | 0.0003882 | 2 | 0 | 0.0001222 | 0.0004013 | 0.0003882 | 0.0004013 | 0.01147 | 0.1428 | 87.81 | 1800 |
| 60 | 10507 | decay | 37.41 | -0.03795 | -0.04155 | [-0.0434, -0.0413] | 0.0036 | [0.0033, 0.0054] | 2.687 | 0.8808 | 2.207 | 0.0003986 | 1 | 0 | 0.0003986 | 0.0006764 | 0.0003882 | 0.0006764 | 0.001647 | 0.01196 | 87.79 | 1800 |
| 60 | 10508 | constant | 40.88 | 0.05127 | 0.06207 | [0.0603, 0.0631] | -0.0108 | [-0.0118, -0.0090] | 3.03 | 1.209 | 2.72 | 0.0008816 | 1 | 0 | 0.0008816 | 0.00139 | 0.0008429 | 0.00139 | 0.001186 | 0.04108 | 87.8 | 1825 |
| 60 | 10508 | decay | 38.89 | 9.084e-05 | 0.01089 | [0.0091, 0.0119] | -0.0108 | [-0.0118, -0.0090] | 3.076 | 1.051 | 2.721 | 0.0008429 | 2 | 0 | 0.000354 | 0.0008637 | 0.0008429 | 0.0008637 | 0.001308 | 0.04205 | 88.24 | 1800 |
| 60 | 10509 | constant | 45.13 | 0.1604 | 0.1635 | [0.1612, 0.1637] | -0.003086 | [-0.0033, -0.0008] | 2.8 | 1.259 | 3.004 | 0.003753 | 1 | 0 | 0.003753 | 0.004542 | 0.003623 | 0.004542 | 0.02037 | 0.1419 | 88.46 | 1750 |
| 60 | 10509 | decay | 38.74 | -0.003852 | -0.000766 | [-0.0031, -0.0005] | -0.003086 | [-0.0033, -0.0008] | 2.818 | 1.346 | 2.768 | 0.000919 | 2 | -4 | 0.0001361 | 0.0009266 | 0.000919 | 0.0009266 | 0.004474 | 0.04437 | 86.18 | 1750 |
| 60 | 10510 | constant | 41.54 | 0.06826 | 0.07186 | [0.0698, 0.0721] | -0.0036 | [-0.0039, -0.0015] | 2.791 | 1.314 | 3.375 | 0.0008271 | 1 | 0 | 0.0008271 | 0.001158 | 0.0007178 | 0.001158 | 0.01787 | 0.107 | 87.81 | 1750 |
| 60 | 10510 | decay | 39.22 | 0.008496 | 0.0121 | [0.0100, 0.0124] | -0.0036 | [-0.0039, -0.0015] | 2.832 | 1.994 | 4.943 | 0.0004406 | 2 | -20 | 0.0001357 | 0.0004675 | 0.0004406 | 0.0004675 | 0.002812 | 0.02694 | 86.03 | 1750 |

**Stability.**

Source: `results/v2_pilots/pilot4/analysis/stability_2b.csv`

| q | arm | across_seed_sd_final_e1 | across_seed_iqr_final_e1 | median_sigma1_0 | median_within_run_sd_last5 | median_within_run_range_last5 |
|---|---|---|---|---|---|---|
| 50 | constant | 3.327 | 3.786 | 2.77 | 1.754 | 3.795 |
| 50 | decay | 2.487 | 3.521 | 2.872 | 1.016 | 2.496 |
| 60 | constant | 3.307 | 2.44 | 2.795 | 1.371 | 3.558 |
| 60 | decay | 1.696 | 1.871 | 2.825 | 0.9931 | 2.377 |

**Tail-averaged candidates** (K ∈ {1, 4, 8, 12}; K = 12 covers u1925–u2200):

Source: `results/v2_pilots/pilot4/analysis/candidates_all.csv`

| q | arm | K | e1_cand | stage1_rel_err_signed | stage1_rel_err_abs | learning_rel | inherited_rel | Gmax_full_over_dw | EXP_root_over_dw | dReach_over_dw |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | constant | 1 | 48.95 | 0.04889 | 0.06281 | 0.0544 | -0.003 | 0.002282 | 0.00146 | 0.003414 |
| 50 | constant | 4 | 49.17 | 0.05367 | 0.05367 | 0.05365 | -0.003 | 0.002805 | 0.001366 | 0.003098 |
| 50 | constant | 8 | 48.51 | 0.03946 | 0.04398 | 0.03715 | -0.003 | 0.002482 | 0.00114 | 0.00318 |
| 50 | constant | 12 | 49.39 | 0.05844 | 0.06485 | 0.05622 | -0.003 | 0.002623 | 0.001875 | 0.003559 |
| 50 | decay | 1 | 47.59 | 0.01989 | 0.04317 | 0.01167 | -0.003 | 0.001508 | 0.00113 | 0.002115 |
| 50 | decay | 4 | 46.78 | 0.002392 | 0.04553 | 0.01113 | -0.003 | 0.001933 | 0.0009912 | 0.0025 |
| 50 | decay | 8 | 46.13 | -0.01146 | 0.0356 | -0.001815 | -0.003 | 0.001874 | 0.0008426 | 0.002479 |
| 50 | decay | 12 | 46.61 | -0.00132 | 0.02476 | 0.008322 | -0.003 | 0.001564 | 0.0007759 | 0.002187 |
| 60 | constant | 1 | 40.3 | 0.0362 | 0.05486 | 0.0425 | -0.003343 | 0.0009536 | 0.0008544 | 0.001534 |
| 60 | constant | 4 | 40.48 | 0.04084 | 0.07151 | 0.0447 | -0.003343 | 0.001256 | 0.001142 | 0.001861 |
| 60 | constant | 8 | 41.31 | 0.06228 | 0.07859 | 0.06794 | -0.003343 | 0.001424 | 0.001339 | 0.002124 |
| 60 | constant | 12 | 41.43 | 0.06527 | 0.08221 | 0.07297 | -0.003343 | 0.001187 | 0.001049 | 0.001701 |
| 60 | decay | 1 | 38.82 | -0.00188 | 0.03381 | 0.005062 | -0.003343 | 0.0009012 | 0.0003763 | 0.001035 |
| 60 | decay | 4 | 38.94 | 0.001275 | 0.0374 | 0.003975 | -0.003343 | 0.0009012 | 0.0003822 | 0.001125 |
| 60 | decay | 8 | 39.07 | 0.004768 | 0.04758 | 0.007211 | -0.003343 | 0.0009101 | 0.0006352 | 0.001155 |
| 60 | decay | 12 | 39.25 | 0.009381 | 0.04318 | 0.01544 | -0.003343 | 0.000971 | 0.0006163 | 0.001288 |

**Location of Ĝmax_full** (t*, d*) for the stage-1 candidates (Pilot 3 and 2b):

Source: `results/v2_pilots/pilot4/analysis/candidates_all.csv`

| family | q | arm | K | t*=1 count | t*=2 count | d* values |
|---|---|---|---|---|---|---|
| 2b | 50 | constant | 1 | 3 | 7 | -60 -40 -32 -24 -16 0 32 |
| 2b | 50 | constant | 4 | 4 | 6 | -32 -24 -16 -4 0 32 |
| 2b | 50 | constant | 8 | 4 | 6 | -32 -24 -16 -4 0 32 |
| 2b | 50 | constant | 12 | 4 | 6 | -32 -24 -16 -4 0 32 |
| 2b | 50 | decay | 1 | 2 | 8 | -40 -32 -24 -16 -4 0 32 |
| 2b | 50 | decay | 4 | 2 | 8 | -40 -32 -24 -16 -4 0 32 |
| 2b | 50 | decay | 8 | 2 | 8 | -40 -32 -24 -16 -4 0 32 |
| 2b | 50 | decay | 12 | 3 | 7 | -40 -32 -24 -16 -4 0 |
| 2b | 60 | constant | 1 | 5 | 5 | -20 -16 0 |
| 2b | 60 | constant | 4 | 6 | 4 | -20 -16 0 |
| 2b | 60 | constant | 8 | 6 | 4 | -20 -16 0 |
| 2b | 60 | constant | 12 | 4 | 6 | -28 -20 -16 0 |
| 2b | 60 | decay | 1 | 2 | 8 | -28 -20 -16 -4 0 |
| 2b | 60 | decay | 4 | 1 | 9 | -28 -20 -16 -4 0 |
| 2b | 60 | decay | 8 | 2 | 8 | -28 -20 -16 -4 0 |
| 2b | 60 | decay | 12 | 4 | 6 | -28 -20 -16 -4 0 |
| p3 | 50 | mean | 1 | 1 | 9 | -4 0 |
| p3 | 50 | mean | 4 | 2 | 8 | -4 0 |
| p3 | 50 | mean | 8 | 1 | 9 | -4 0 |
| p3 | 50 | mean | 12 | 0 | 10 | -4 100 |
| p3 | 50 | stochastic | 1 | 1 | 9 | -4 0 |
| p3 | 50 | stochastic | 4 | 1 | 9 | -4 0 100 |
| p3 | 50 | stochastic | 8 | 0 | 10 | -4 100 |
| p3 | 50 | stochastic | 12 | 0 | 10 | -4 100 |
| p3 | 60 | mean | 1 | 5 | 5 | -4 0 |
| p3 | 60 | mean | 4 | 2 | 8 | -32 -4 0 40 |
| p3 | 60 | mean | 8 | 0 | 10 | -32 -4 40 |
| p3 | 60 | mean | 12 | 0 | 10 | -32 -4 40 |
| p3 | 60 | stochastic | 1 | 0 | 10 | -32 -4 40 |
| p3 | 60 | stochastic | 4 | 0 | 10 | -32 -4 40 |
| p3 | 60 | stochastic | 8 | 0 | 10 | -32 -4 40 |
| p3 | 60 | stochastic | 12 | 0 | 10 | -32 -4 40 |

**Paired differences decay − constant per (q, seed), stage 1 and full policy:**

| q | K | metric | median | better | CI95 |
|---|---|---|---|---|---|
| 50 | 1 | |stage-1 err| | −0.0150 | 6 | [−0.049, 0.003] |
| 50 | 1 | |learning term| | −0.0178 | 7 | [−0.057, −0.0009] |
| 50 | 1 | EXP_root/ΔW | −0.00027 | 7 | [−0.0034, −0.00001] |
| 50 | 1 | dReach/ΔW | −0.00027 | 7 | [−0.0034, −0.000003] |
| 50 | 1 | within-run SD last5 | −0.78 | 7 | [−1.22, −0.27] |
| 50 | 1 | within-run range last5 | −2.12 | 7 | [−2.96, −0.49] |
| 50 | 12 | |stage-1 err| | −0.0175 | 6 | [−0.043, 0.004] |
| 50 | 12 | EXP_root/ΔW | −0.00078 | 6 | [−0.0023, 0.00006] |
| 60 | 1 | |stage-1 err| | −0.0235 | 5 | [−0.076, 0.006] |
| 60 | 1 | |learning term| | −0.0348 | 6 | [−0.079, 0.004] |
| 60 | 1 | EXP_root/ΔW | −0.00041 | 6 | [−0.0017, −0.00008] |
| 60 | 1 | dReach/ΔW | −0.00040 | 6 | [−0.0017, −0.00009] |
| 60 | 1 | within-run SD last5 | −0.51 | 6 | [−1.20, −0.08] |
| 60 | 1 | within-run range last5 | −1.28 | 6 | [−2.98, −0.25] |
| 60 | 12 | |stage-1 err| | −0.0170 | 6 | [−0.069, 0.007] |
| 60 | 12 | EXP_root/ΔW | −0.00041 | 6 | [−0.0021, 0.00006] |

Signed stage-1 error, decay − constant, K = 1: q=50 median +0.009 (4−/6+, CI [−0.095, 0.010]); q=60 −0.052 (7−/3+, CI [−0.093, 0.003]). Every K and metric: `A/paired_summary.csv` (`family = 2b`) and `A/paired_summary_run_records.csv`.

**K − (K=1) within each arm, |stage-1 err|:**
- constant arm: q=50 −0.019 / −0.015 / −0.006 for K = 4 / 8 / 12; q=60 +0.008 / +0.021 / +0.023. All CIs include 0.
- decay arm: q=50 +0.004 / −0.001 / −0.003; q=60 −0.002 / +0.031 / +0.031. All CIs include 0.

Source: `A/paired_summary.csv` rows `family = 2b`, `comparison = K_vs_K1`.

**RNG divergence (decay vs constant, global update of the first difference).**

Source: `results/v2_pilots/pilot4/analysis/rng_divergence.csv`

| family | q | env | learn | opp | start | minibatch |
|---|---|---|---|---|---|---|
| 2a | 50 | never (10/10) | 10/10; min 1204, median 1209, max 1240 | 10/10; min 1221, median 1222, max 1236 | never (10/10) | never (10/10) |
| 2a | 60 | never (10/10) | 10/10; min 1205, median 1208, max 1229 | 10/10; min 1221, median 1224, max 1231 | never (10/10) | never (10/10) |
| 2b | 50 | never (10/10) | 10/10; min 1613, median 1672, max 1805 | 10/10; min 1619, median 1659, max 1909 | never (10/10) | never (10/10) |
| 2b | 60 | never (10/10) | 10/10; min 1607, median 1680, max 2168 | 10/10; min 1628, median 1714, max 1978 | never (10/10) | never (10/10) |

The learn/opp streams diverge in every pair, as in every earlier pilot (rejection-based Beta sampler, D4). In 2a this happens within 4–40 updates of u1200, because the LR differs from local update 2 on.

**Advantage statistics, KL and clip.** Medians over all updates: `A/run_records.csv`, columns `adv_*_median`, `kl_median`, `clip_median`; the 2b per-arm medians are in the run-records table above.
- The decay arms have lower KL and clip fraction: 2b clip median 0.051 vs 0.087 (q=50) and 0.059 vs 0.099 (q=60).
- The paired last-update clip fraction is lower in 9/10 pairs at both q.

**Learning curves.** Median and IQR over 10 seeds, exports u1625–u2200: `reports/v2/figures/pilot4/curves_2b_q50.png`, `curves_2b_q60.png`. Panels: signed stage-1 error, learning term, σ₁(0), Ĝmax_full/ΔW, EXP_root/ΔW.

**Wall clock.** Phase B wall time had a median of 145–156 s per run; 2a's 400 updates had a median of 69–71 s. The paired wall-time difference is not consistent (`A/paired_summary_run_records.csv`).

---

## 6. Distribution tables for setting gates (no threshold proposals)

Quantiles over the 10 seeds of each q for each end-of-phase candidate. Phase A = 2a at u1600 (`constant`, `decay`) × K ∈ {1, 4, 8}; Phase B = 2b at u2200 × K ∈ {1, 4, 8, 12}.
- Phase A has stage 1 untrained, so |stage-1 err|, Ĝmax_full and EXP_root are not defined there and are omitted.
- In Phase B the stage-2 metrics are those of the frozen u1600 parent, identical across arms and K. They are listed for completeness.
- Phase A `constant` K=1 is the extension's u1600 candidate (bit-identical, §2a).

Source: `results/v2_pilots/pilot4/analysis/gate_distribution_tables.csv`

| arm | K | q | metric | min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| constant | 1 | 50 | eta_T_over_dw | 0.0007199 | 0.001015 | 0.001119 | 0.001469 | 0.002431 | 0.003733 | 0.006971 |
| constant | 4 | 50 | eta_T_over_dw | 0.0004987 | 0.0006821 | 0.001018 | 0.001168 | 0.001855 | 0.00223 | 0.002563 |
| constant | 8 | 50 | eta_T_over_dw | 0.0005877 | 0.000866 | 0.001068 | 0.001285 | 0.001526 | 0.002608 | 0.00304 |
| decay | 1 | 50 | eta_T_over_dw | 0.0007256 | 0.0009001 | 0.0009888 | 0.001082 | 0.001519 | 0.00278 | 0.004014 |
| decay | 4 | 50 | eta_T_over_dw | 0.0006195 | 0.0006471 | 0.0006597 | 0.001036 | 0.001521 | 0.002795 | 0.003426 |
| decay | 8 | 50 | eta_T_over_dw | 0.0004457 | 0.0006706 | 0.0007642 | 0.001126 | 0.001284 | 0.002363 | 0.004195 |
| constant | 1 | 60 | eta_T_over_dw | 0.0003882 | 0.0004353 | 0.0007233 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| constant | 4 | 60 | eta_T_over_dw | 0.0004663 | 0.0005175 | 0.0005915 | 0.0007207 | 0.0007453 | 0.0009099 | 0.001142 |
| constant | 8 | 60 | eta_T_over_dw | 0.0003168 | 0.0003858 | 0.0004095 | 0.0006428 | 0.0007601 | 0.0008653 | 0.00097 |
| decay | 1 | 60 | eta_T_over_dw | 0.0003602 | 0.0003806 | 0.0004732 | 0.0008081 | 0.0009643 | 0.001458 | 0.001497 |
| decay | 4 | 60 | eta_T_over_dw | 0.000354 | 0.0003982 | 0.0004425 | 0.0006526 | 0.0008819 | 0.0009596 | 0.001131 |
| decay | 8 | 60 | eta_T_over_dw | 0.00037 | 0.0003752 | 0.0004112 | 0.0005716 | 0.0007766 | 0.0008276 | 0.0008496 |
| constant | 1 | 50 | stage2_peak_locfree_rel_err | -0.1286 | -0.07707 | -0.07074 | -0.06826 | -0.05652 | -0.05358 | -0.0368 |
| constant | 4 | 50 | stage2_peak_locfree_rel_err | -0.09311 | -0.09274 | -0.08075 | -0.06247 | -0.05435 | -0.04822 | -0.04065 |
| constant | 8 | 50 | stage2_peak_locfree_rel_err | -0.1069 | -0.09748 | -0.07593 | -0.06972 | -0.05426 | -0.03868 | -0.03396 |
| decay | 1 | 50 | stage2_peak_locfree_rel_err | -0.1221 | -0.1007 | -0.07013 | -0.06243 | -0.05456 | -0.03398 | -0.02246 |
| decay | 4 | 50 | stage2_peak_locfree_rel_err | -0.1115 | -0.09958 | -0.07868 | -0.06831 | -0.05189 | -0.03908 | -0.03662 |
| decay | 8 | 50 | stage2_peak_locfree_rel_err | -0.1205 | -0.09201 | -0.07369 | -0.06634 | -0.04799 | -0.034 | -0.02867 |
| constant | 1 | 60 | stage2_peak_locfree_rel_err | -0.07572 | -0.07471 | -0.07326 | -0.05585 | -0.03969 | -0.02462 | 0.003559 |
| constant | 4 | 60 | stage2_peak_locfree_rel_err | -0.08172 | -0.07425 | -0.06654 | -0.06217 | -0.05594 | -0.05117 | -0.03125 |
| constant | 8 | 60 | stage2_peak_locfree_rel_err | -0.0769 | -0.0681 | -0.0653 | -0.0551 | -0.04632 | -0.04383 | -0.04222 |
| decay | 1 | 60 | stage2_peak_locfree_rel_err | -0.09052 | -0.07707 | -0.06686 | -0.04943 | -0.04338 | -0.03041 | -0.02071 |
| decay | 4 | 60 | stage2_peak_locfree_rel_err | -0.08043 | -0.0752 | -0.07275 | -0.059 | -0.0498 | -0.04167 | -0.03799 |
| decay | 8 | 60 | stage2_peak_locfree_rel_err | -0.07081 | -0.06853 | -0.06799 | -0.05858 | -0.04811 | -0.0441 | -0.03961 |
| constant | 1 | 50 | stage2_peak_rel_err_abs | 0.03741 | 0.05453 | 0.05722 | 0.06835 | 0.0709 | 0.07798 | 0.1286 |
| constant | 4 | 50 | stage2_peak_rel_err_abs | 0.04077 | 0.04837 | 0.05453 | 0.06294 | 0.08174 | 0.09316 | 0.09494 |
| constant | 8 | 50 | stage2_peak_rel_err_abs | 0.03448 | 0.03956 | 0.05445 | 0.07022 | 0.07666 | 0.0978 | 0.107 |
| decay | 1 | 50 | stage2_peak_rel_err_abs | 0.02357 | 0.0342 | 0.05518 | 0.0625 | 0.07044 | 0.1008 | 0.1222 |
| decay | 4 | 50 | stage2_peak_rel_err_abs | 0.03757 | 0.03928 | 0.05234 | 0.06911 | 0.07868 | 0.1002 | 0.1116 |
| decay | 8 | 50 | stage2_peak_rel_err_abs | 0.02928 | 0.03464 | 0.04943 | 0.06673 | 0.07379 | 0.09278 | 0.1212 |
| constant | 1 | 60 | stage2_peak_rel_err_abs | 0.003541 | 0.02652 | 0.04034 | 0.06003 | 0.07362 | 0.075 | 0.07572 |
| constant | 4 | 60 | stage2_peak_rel_err_abs | 0.03148 | 0.0512 | 0.05679 | 0.06243 | 0.06688 | 0.07435 | 0.08176 |
| constant | 8 | 60 | stage2_peak_rel_err_abs | 0.04332 | 0.04394 | 0.04679 | 0.05595 | 0.06533 | 0.06988 | 0.07701 |
| decay | 1 | 60 | stage2_peak_rel_err_abs | 0.02097 | 0.03083 | 0.0439 | 0.04971 | 0.067 | 0.0775 | 0.09174 |
| decay | 4 | 60 | stage2_peak_rel_err_abs | 0.0385 | 0.04256 | 0.04982 | 0.05929 | 0.07342 | 0.07551 | 0.08114 |
| decay | 8 | 60 | stage2_peak_rel_err_abs | 0.0403 | 0.04479 | 0.0482 | 0.05884 | 0.06891 | 0.07113 | 0.07207 |
| constant | 1 | 50 | stage2_peak_rel_err_signed | -0.1286 | -0.07798 | -0.0709 | -0.06835 | -0.05722 | -0.05453 | -0.03741 |
| constant | 4 | 50 | stage2_peak_rel_err_signed | -0.09494 | -0.09316 | -0.08174 | -0.06294 | -0.05453 | -0.04837 | -0.04077 |
| constant | 8 | 50 | stage2_peak_rel_err_signed | -0.107 | -0.0978 | -0.07666 | -0.07022 | -0.05445 | -0.03956 | -0.03448 |
| decay | 1 | 50 | stage2_peak_rel_err_signed | -0.1222 | -0.1008 | -0.07044 | -0.0625 | -0.05518 | -0.0342 | -0.02357 |
| decay | 4 | 50 | stage2_peak_rel_err_signed | -0.1116 | -0.1002 | -0.07868 | -0.06911 | -0.05234 | -0.03928 | -0.03757 |
| decay | 8 | 50 | stage2_peak_rel_err_signed | -0.1212 | -0.09278 | -0.07379 | -0.06673 | -0.04943 | -0.03464 | -0.02928 |
| constant | 1 | 60 | stage2_peak_rel_err_signed | -0.07572 | -0.075 | -0.07362 | -0.06003 | -0.04034 | -0.02581 | 0.003541 |
| constant | 4 | 60 | stage2_peak_rel_err_signed | -0.08176 | -0.07435 | -0.06688 | -0.06243 | -0.05679 | -0.0512 | -0.03148 |
| constant | 8 | 60 | stage2_peak_rel_err_signed | -0.07701 | -0.06988 | -0.06533 | -0.05595 | -0.04679 | -0.04394 | -0.04332 |
| decay | 1 | 60 | stage2_peak_rel_err_signed | -0.09174 | -0.0775 | -0.067 | -0.04971 | -0.0439 | -0.03083 | -0.02097 |
| decay | 4 | 60 | stage2_peak_rel_err_signed | -0.08114 | -0.07551 | -0.07342 | -0.05929 | -0.04982 | -0.04256 | -0.0385 |
| decay | 8 | 60 | stage2_peak_rel_err_signed | -0.07207 | -0.07113 | -0.06891 | -0.05884 | -0.0482 | -0.04479 | -0.0403 |
| constant | 1 | 50 | stage2_rmse_pos_over_g2_0 | 0.01422 | 0.02004 | 0.0221 | 0.02605 | 0.02871 | 0.03632 | 0.05976 |
| constant | 4 | 50 | stage2_rmse_pos_over_g2_0 | 0.01642 | 0.01778 | 0.01888 | 0.0201 | 0.02321 | 0.02718 | 0.02746 |
| constant | 8 | 50 | stage2_rmse_pos_over_g2_0 | 0.01438 | 0.01691 | 0.01776 | 0.01917 | 0.02075 | 0.02488 | 0.02906 |
| decay | 1 | 50 | stage2_rmse_pos_over_g2_0 | 0.01502 | 0.01718 | 0.01925 | 0.02007 | 0.02384 | 0.0279 | 0.03966 |
| decay | 4 | 50 | stage2_rmse_pos_over_g2_0 | 0.01472 | 0.01511 | 0.01635 | 0.01735 | 0.02154 | 0.02582 | 0.03071 |
| decay | 8 | 50 | stage2_rmse_pos_over_g2_0 | 0.01416 | 0.0146 | 0.0159 | 0.01713 | 0.01899 | 0.02224 | 0.03369 |
| constant | 1 | 60 | stage2_rmse_pos_over_g2_0 | 0.01864 | 0.01871 | 0.02089 | 0.02389 | 0.03157 | 0.03305 | 0.03376 |
| constant | 4 | 60 | stage2_rmse_pos_over_g2_0 | 0.01626 | 0.01717 | 0.0173 | 0.0179 | 0.01925 | 0.02039 | 0.02177 |
| constant | 8 | 60 | stage2_rmse_pos_over_g2_0 | 0.01452 | 0.01515 | 0.01559 | 0.01711 | 0.01975 | 0.02153 | 0.02156 |
| decay | 1 | 60 | stage2_rmse_pos_over_g2_0 | 0.01333 | 0.01598 | 0.01729 | 0.01982 | 0.02362 | 0.02474 | 0.02651 |
| decay | 4 | 60 | stage2_rmse_pos_over_g2_0 | 0.01363 | 0.01423 | 0.01716 | 0.01774 | 0.01848 | 0.01885 | 0.02059 |
| decay | 8 | 60 | stage2_rmse_pos_over_g2_0 | 0.01388 | 0.01434 | 0.01549 | 0.01627 | 0.01794 | 0.02132 | 0.02282 |
| constant | 1 | 50 | stage2_tail_max_over_g2_0 | 0.02365 | 0.02917 | 0.03044 | 0.03172 | 0.03711 | 0.04461 | 0.07009 |
| constant | 4 | 50 | stage2_tail_max_over_g2_0 | 0.02726 | 0.0292 | 0.03218 | 0.03334 | 0.03433 | 0.03809 | 0.06256 |
| constant | 8 | 50 | stage2_tail_max_over_g2_0 | 0.03011 | 0.03043 | 0.03183 | 0.03343 | 0.03557 | 0.04449 | 0.06118 |
| decay | 1 | 50 | stage2_tail_max_over_g2_0 | 0.03168 | 0.03219 | 0.03463 | 0.03692 | 0.03919 | 0.04473 | 0.06114 |
| decay | 4 | 50 | stage2_tail_max_over_g2_0 | 0.03141 | 0.03368 | 0.03489 | 0.03737 | 0.03866 | 0.04335 | 0.06362 |
| decay | 8 | 50 | stage2_tail_max_over_g2_0 | 0.03114 | 0.03266 | 0.03447 | 0.03572 | 0.03713 | 0.04069 | 0.06423 |
| constant | 1 | 60 | stage2_tail_max_over_g2_0 | 0.02927 | 0.02982 | 0.03146 | 0.03259 | 0.03616 | 0.04017 | 0.04382 |
| constant | 4 | 60 | stage2_tail_max_over_g2_0 | 0.02999 | 0.03025 | 0.03212 | 0.03538 | 0.03753 | 0.03901 | 0.03912 |
| constant | 8 | 60 | stage2_tail_max_over_g2_0 | 0.03089 | 0.03222 | 0.03332 | 0.0354 | 0.03765 | 0.03827 | 0.03831 |
| decay | 1 | 60 | stage2_tail_max_over_g2_0 | 0.03222 | 0.0346 | 0.03531 | 0.03619 | 0.0382 | 0.04007 | 0.04297 |
| decay | 4 | 60 | stage2_tail_max_over_g2_0 | 0.03181 | 0.03409 | 0.03514 | 0.03637 | 0.038 | 0.03912 | 0.04044 |
| decay | 8 | 60 | stage2_tail_max_over_g2_0 | 0.03158 | 0.03254 | 0.03449 | 0.03638 | 0.03803 | 0.03936 | 0.04041 |
| constant | 1 | 50 | stage2_tail_mean_over_g2_0 | 0.00504 | 0.005525 | 0.006513 | 0.007358 | 0.008055 | 0.009557 | 0.01262 |
| constant | 4 | 50 | stage2_tail_mean_over_g2_0 | 0.005495 | 0.005756 | 0.006867 | 0.007931 | 0.008077 | 0.009736 | 0.0127 |
| constant | 8 | 50 | stage2_tail_mean_over_g2_0 | 0.005799 | 0.006122 | 0.006908 | 0.008027 | 0.008366 | 0.009555 | 0.0123 |
| decay | 1 | 50 | stage2_tail_mean_over_g2_0 | 0.004951 | 0.006238 | 0.007055 | 0.008032 | 0.009028 | 0.01006 | 0.01187 |
| decay | 4 | 50 | stage2_tail_mean_over_g2_0 | 0.00525 | 0.006395 | 0.006783 | 0.008436 | 0.009089 | 0.009798 | 0.01154 |
| decay | 8 | 50 | stage2_tail_mean_over_g2_0 | 0.005382 | 0.006385 | 0.006755 | 0.008559 | 0.009089 | 0.009868 | 0.01177 |
| constant | 1 | 60 | stage2_tail_mean_over_g2_0 | 0.007102 | 0.008241 | 0.008588 | 0.009106 | 0.009708 | 0.01073 | 0.01093 |
| constant | 4 | 60 | stage2_tail_mean_over_g2_0 | 0.007741 | 0.008168 | 0.008731 | 0.009264 | 0.00975 | 0.01038 | 0.01107 |
| constant | 8 | 60 | stage2_tail_mean_over_g2_0 | 0.008003 | 0.008152 | 0.008927 | 0.009446 | 0.009934 | 0.01057 | 0.01157 |
| decay | 1 | 60 | stage2_tail_mean_over_g2_0 | 0.008325 | 0.008601 | 0.008712 | 0.009393 | 0.01041 | 0.01097 | 0.01254 |
| decay | 4 | 60 | stage2_tail_mean_over_g2_0 | 0.008178 | 0.008306 | 0.008503 | 0.009815 | 0.01042 | 0.01093 | 0.01259 |
| decay | 8 | 60 | stage2_tail_mean_over_g2_0 | 0.00826 | 0.008264 | 0.008435 | 0.009804 | 0.0105 | 0.01101 | 0.01274 |

Source: `results/v2_pilots/pilot4/analysis/gate_distribution_tables.csv`

| arm | K | q | metric | min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| constant | 1 | 50 | EXP_root_over_dw | 0.0004443 | 0.0006063 | 0.0009034 | 0.00146 | 0.003025 | 0.004744 | 0.01079 |
| constant | 4 | 50 | EXP_root_over_dw | 0.0001632 | 0.0005439 | 0.0008332 | 0.001366 | 0.003712 | 0.00619 | 0.006218 |
| constant | 8 | 50 | EXP_root_over_dw | 0.0002234 | 0.0004095 | 0.0008106 | 0.00114 | 0.002876 | 0.004201 | 0.00978 |
| constant | 12 | 50 | EXP_root_over_dw | 0.0002052 | 0.0004748 | 0.000858 | 0.001875 | 0.003061 | 0.004138 | 0.008448 |
| decay | 1 | 50 | EXP_root_over_dw | 0.0001825 | 0.0003246 | 0.0004146 | 0.00113 | 0.001822 | 0.002447 | 0.003149 |
| decay | 4 | 50 | EXP_root_over_dw | 0.0001957 | 0.0004584 | 0.0006984 | 0.0009912 | 0.002293 | 0.003169 | 0.003208 |
| decay | 8 | 50 | EXP_root_over_dw | 0.000164 | 0.0002011 | 0.0004178 | 0.0008426 | 0.002665 | 0.003014 | 0.003042 |
| decay | 12 | 50 | EXP_root_over_dw | 0.0001345 | 0.0001996 | 0.0004992 | 0.0007759 | 0.002575 | 0.003082 | 0.004163 |
| constant | 1 | 60 | EXP_root_over_dw | 0.0001222 | 0.0001708 | 0.0003839 | 0.0008544 | 0.002021 | 0.00303 | 0.003753 |
| constant | 4 | 60 | EXP_root_over_dw | 0.0001466 | 0.0001827 | 0.0003352 | 0.001142 | 0.002641 | 0.003948 | 0.005093 |
| constant | 8 | 60 | EXP_root_over_dw | 0.0001447 | 0.0002217 | 0.0006332 | 0.001339 | 0.001967 | 0.002933 | 0.007975 |
| constant | 12 | 60 | EXP_root_over_dw | 0.0001129 | 0.0001392 | 0.0006298 | 0.001049 | 0.001947 | 0.003107 | 0.005886 |
| decay | 1 | 60 | EXP_root_over_dw | 0.0001357 | 0.000136 | 0.0001945 | 0.0003763 | 0.0006899 | 0.0008758 | 0.0008868 |
| decay | 4 | 60 | EXP_root_over_dw | 0.0001233 | 0.0001402 | 0.0001969 | 0.0003822 | 0.0007302 | 0.0007819 | 0.0008369 |
| decay | 8 | 60 | EXP_root_over_dw | 0.0002407 | 0.000273 | 0.0003133 | 0.0006352 | 0.0007736 | 0.001233 | 0.001683 |
| decay | 12 | 60 | EXP_root_over_dw | 0.0001868 | 0.0002048 | 0.0002897 | 0.0006163 | 0.001147 | 0.001595 | 0.001712 |
| constant | 1 | 50 | Gmax_full_over_dw | 0.0007199 | 0.001239 | 0.001478 | 0.002282 | 0.00341 | 0.007353 | 0.01079 |
| constant | 4 | 50 | Gmax_full_over_dw | 0.0007199 | 0.001176 | 0.001339 | 0.002805 | 0.005484 | 0.006293 | 0.006971 |
| constant | 8 | 50 | Gmax_full_over_dw | 0.0008063 | 0.001052 | 0.001339 | 0.002482 | 0.00329 | 0.007252 | 0.00978 |
| constant | 12 | 50 | Gmax_full_over_dw | 0.00108 | 0.001221 | 0.00158 | 0.002623 | 0.003328 | 0.007119 | 0.008448 |
| decay | 1 | 50 | Gmax_full_over_dw | 0.0007199 | 0.001044 | 0.001252 | 0.001508 | 0.002527 | 0.003733 | 0.006971 |
| decay | 4 | 50 | Gmax_full_over_dw | 0.0007199 | 0.001044 | 0.001252 | 0.001933 | 0.002514 | 0.003733 | 0.006971 |
| decay | 8 | 50 | Gmax_full_over_dw | 0.0007199 | 0.001044 | 0.001252 | 0.001874 | 0.002784 | 0.003733 | 0.006971 |
| decay | 12 | 50 | Gmax_full_over_dw | 0.0007199 | 0.001044 | 0.001252 | 0.001564 | 0.002866 | 0.004444 | 0.006971 |
| constant | 1 | 60 | Gmax_full_over_dw | 0.0003882 | 0.0006539 | 0.0008407 | 0.0009536 | 0.002322 | 0.00303 | 0.003753 |
| constant | 4 | 60 | Gmax_full_over_dw | 0.0003882 | 0.0004353 | 0.000939 | 0.001256 | 0.00286 | 0.003948 | 0.005093 |
| constant | 8 | 60 | Gmax_full_over_dw | 0.0003882 | 0.0004353 | 0.001008 | 0.001424 | 0.002148 | 0.002933 | 0.007975 |
| constant | 12 | 60 | Gmax_full_over_dw | 0.0003882 | 0.0004353 | 0.0009012 | 0.001187 | 0.002011 | 0.003107 | 0.005886 |
| decay | 1 | 60 | Gmax_full_over_dw | 0.0003986 | 0.0004364 | 0.0008508 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| decay | 4 | 60 | Gmax_full_over_dw | 0.0003882 | 0.0004353 | 0.0008384 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| decay | 8 | 60 | Gmax_full_over_dw | 0.0003882 | 0.0006556 | 0.0008574 | 0.0009101 | 0.0009974 | 0.00173 | 0.002156 |
| decay | 12 | 60 | Gmax_full_over_dw | 0.0003882 | 0.0008386 | 0.0009057 | 0.000971 | 0.001156 | 0.001757 | 0.002156 |
| constant | 1 | 50 | eta_T_over_dw | 0.0007199 | 0.001015 | 0.001119 | 0.001469 | 0.002431 | 0.003733 | 0.006971 |
| constant | 4 | 50 | eta_T_over_dw | 0.0007199 | 0.001015 | 0.001119 | 0.001469 | 0.002431 | 0.003733 | 0.006971 |
| constant | 8 | 50 | eta_T_over_dw | 0.0007199 | 0.001015 | 0.001119 | 0.001469 | 0.002431 | 0.003733 | 0.006971 |
| constant | 12 | 50 | eta_T_over_dw | 0.0007199 | 0.001015 | 0.001119 | 0.001469 | 0.002431 | 0.003733 | 0.006971 |
| decay | 1 | 50 | eta_T_over_dw | 0.0007199 | 0.001015 | 0.001119 | 0.001469 | 0.002431 | 0.003733 | 0.006971 |
| decay | 4 | 50 | eta_T_over_dw | 0.0007199 | 0.001015 | 0.001119 | 0.001469 | 0.002431 | 0.003733 | 0.006971 |
| decay | 8 | 50 | eta_T_over_dw | 0.0007199 | 0.001015 | 0.001119 | 0.001469 | 0.002431 | 0.003733 | 0.006971 |
| decay | 12 | 50 | eta_T_over_dw | 0.0007199 | 0.001015 | 0.001119 | 0.001469 | 0.002431 | 0.003733 | 0.006971 |
| constant | 1 | 60 | eta_T_over_dw | 0.0003882 | 0.0004353 | 0.0007233 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| constant | 4 | 60 | eta_T_over_dw | 0.0003882 | 0.0004353 | 0.0007233 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| constant | 8 | 60 | eta_T_over_dw | 0.0003882 | 0.0004353 | 0.0007233 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| constant | 12 | 60 | eta_T_over_dw | 0.0003882 | 0.0004353 | 0.0007233 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| decay | 1 | 60 | eta_T_over_dw | 0.0003882 | 0.0004353 | 0.0007233 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| decay | 4 | 60 | eta_T_over_dw | 0.0003882 | 0.0004353 | 0.0007233 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| decay | 8 | 60 | eta_T_over_dw | 0.0003882 | 0.0004353 | 0.0007233 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| decay | 12 | 60 | eta_T_over_dw | 0.0003882 | 0.0004353 | 0.0007233 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| constant | 1 | 50 | stage1_rel_err_abs | 0.01437 | 0.02567 | 0.03628 | 0.06281 | 0.07709 | 0.1053 | 0.1777 |
| constant | 4 | 50 | stage1_rel_err_abs | 0.01014 | 0.01354 | 0.018 | 0.05367 | 0.08246 | 0.1337 | 0.1436 |
| constant | 8 | 50 | stage1_rel_err_abs | 0.005005 | 0.01277 | 0.02768 | 0.04398 | 0.07093 | 0.09842 | 0.1791 |
| constant | 12 | 50 | stage1_rel_err_abs | 0.001757 | 0.003798 | 0.04771 | 0.06485 | 0.0804 | 0.1003 | 0.1674 |
| decay | 1 | 50 | stage1_rel_err_abs | 0.01801 | 0.01962 | 0.02118 | 0.04317 | 0.06162 | 0.07762 | 0.08802 |
| decay | 4 | 50 | stage1_rel_err_abs | 0.01994 | 0.02157 | 0.02749 | 0.04553 | 0.07779 | 0.09817 | 0.09909 |
| decay | 8 | 50 | stage1_rel_err_abs | 0.008304 | 0.009129 | 0.01734 | 0.0356 | 0.0819 | 0.0963 | 0.09713 |
| decay | 12 | 50 | stage1_rel_err_abs | 0.009957 | 0.01233 | 0.01734 | 0.02476 | 0.07281 | 0.1006 | 0.1155 |
| constant | 1 | 60 | stage1_rel_err_abs | 0.00297 | 0.003284 | 0.02336 | 0.05486 | 0.1195 | 0.1442 | 0.1604 |
| constant | 4 | 60 | stage1_rel_err_abs | 0.01084 | 0.01095 | 0.01963 | 0.07151 | 0.1219 | 0.1645 | 0.1882 |
| constant | 8 | 60 | stage1_rel_err_abs | 0.01045 | 0.01754 | 0.0513 | 0.07859 | 0.1079 | 0.1365 | 0.2366 |
| constant | 12 | 60 | stage1_rel_err_abs | 0.00774 | 0.009696 | 0.04406 | 0.08221 | 0.1015 | 0.1298 | 0.2027 |
| decay | 1 | 60 | stage1_rel_err_abs | 9.084e-05 | 0.003476 | 0.009526 | 0.03381 | 0.05514 | 0.06453 | 0.08066 |
| decay | 4 | 60 | stage1_rel_err_abs | 0.0009903 | 0.003286 | 0.01699 | 0.0374 | 0.04757 | 0.05894 | 0.06158 |
| decay | 8 | 60 | stage1_rel_err_abs | 0.02181 | 0.02449 | 0.03328 | 0.04758 | 0.06705 | 0.07335 | 0.09611 |
| decay | 12 | 60 | stage1_rel_err_abs | 0.01097 | 0.01255 | 0.02345 | 0.04318 | 0.08383 | 0.08799 | 0.09712 |
| constant | 1 | 50 | stage2_peak_locfree_rel_err | -0.1286 | -0.07707 | -0.07074 | -0.06826 | -0.05652 | -0.05358 | -0.0368 |
| constant | 4 | 50 | stage2_peak_locfree_rel_err | -0.1286 | -0.07707 | -0.07074 | -0.06826 | -0.05652 | -0.05358 | -0.0368 |
| constant | 8 | 50 | stage2_peak_locfree_rel_err | -0.1286 | -0.07707 | -0.07074 | -0.06826 | -0.05652 | -0.05358 | -0.0368 |
| constant | 12 | 50 | stage2_peak_locfree_rel_err | -0.1286 | -0.07707 | -0.07074 | -0.06826 | -0.05652 | -0.05358 | -0.0368 |
| decay | 1 | 50 | stage2_peak_locfree_rel_err | -0.1286 | -0.07707 | -0.07074 | -0.06826 | -0.05652 | -0.05358 | -0.0368 |
| decay | 4 | 50 | stage2_peak_locfree_rel_err | -0.1286 | -0.07707 | -0.07074 | -0.06826 | -0.05652 | -0.05358 | -0.0368 |
| decay | 8 | 50 | stage2_peak_locfree_rel_err | -0.1286 | -0.07707 | -0.07074 | -0.06826 | -0.05652 | -0.05358 | -0.0368 |
| decay | 12 | 50 | stage2_peak_locfree_rel_err | -0.1286 | -0.07707 | -0.07074 | -0.06826 | -0.05652 | -0.05358 | -0.0368 |
| constant | 1 | 60 | stage2_peak_locfree_rel_err | -0.07572 | -0.07471 | -0.07326 | -0.05585 | -0.03969 | -0.02462 | 0.003559 |
| constant | 4 | 60 | stage2_peak_locfree_rel_err | -0.07572 | -0.07471 | -0.07326 | -0.05585 | -0.03969 | -0.02462 | 0.003559 |
| constant | 8 | 60 | stage2_peak_locfree_rel_err | -0.07572 | -0.07471 | -0.07326 | -0.05585 | -0.03969 | -0.02462 | 0.003559 |
| constant | 12 | 60 | stage2_peak_locfree_rel_err | -0.07572 | -0.07471 | -0.07326 | -0.05585 | -0.03969 | -0.02462 | 0.003559 |
| decay | 1 | 60 | stage2_peak_locfree_rel_err | -0.07572 | -0.07471 | -0.07326 | -0.05585 | -0.03969 | -0.02462 | 0.003559 |
| decay | 4 | 60 | stage2_peak_locfree_rel_err | -0.07572 | -0.07471 | -0.07326 | -0.05585 | -0.03969 | -0.02462 | 0.003559 |
| decay | 8 | 60 | stage2_peak_locfree_rel_err | -0.07572 | -0.07471 | -0.07326 | -0.05585 | -0.03969 | -0.02462 | 0.003559 |
| decay | 12 | 60 | stage2_peak_locfree_rel_err | -0.07572 | -0.07471 | -0.07326 | -0.05585 | -0.03969 | -0.02462 | 0.003559 |
| constant | 1 | 50 | stage2_peak_rel_err_abs | 0.03741 | 0.05453 | 0.05722 | 0.06835 | 0.0709 | 0.07798 | 0.1286 |
| constant | 4 | 50 | stage2_peak_rel_err_abs | 0.03741 | 0.05453 | 0.05722 | 0.06835 | 0.0709 | 0.07798 | 0.1286 |
| constant | 8 | 50 | stage2_peak_rel_err_abs | 0.03741 | 0.05453 | 0.05722 | 0.06835 | 0.0709 | 0.07798 | 0.1286 |
| constant | 12 | 50 | stage2_peak_rel_err_abs | 0.03741 | 0.05453 | 0.05722 | 0.06835 | 0.0709 | 0.07798 | 0.1286 |
| decay | 1 | 50 | stage2_peak_rel_err_abs | 0.03741 | 0.05453 | 0.05722 | 0.06835 | 0.0709 | 0.07798 | 0.1286 |
| decay | 4 | 50 | stage2_peak_rel_err_abs | 0.03741 | 0.05453 | 0.05722 | 0.06835 | 0.0709 | 0.07798 | 0.1286 |
| decay | 8 | 50 | stage2_peak_rel_err_abs | 0.03741 | 0.05453 | 0.05722 | 0.06835 | 0.0709 | 0.07798 | 0.1286 |
| decay | 12 | 50 | stage2_peak_rel_err_abs | 0.03741 | 0.05453 | 0.05722 | 0.06835 | 0.0709 | 0.07798 | 0.1286 |
| constant | 1 | 60 | stage2_peak_rel_err_abs | 0.003541 | 0.02652 | 0.04034 | 0.06003 | 0.07362 | 0.075 | 0.07572 |
| constant | 4 | 60 | stage2_peak_rel_err_abs | 0.003541 | 0.02652 | 0.04034 | 0.06003 | 0.07362 | 0.075 | 0.07572 |
| constant | 8 | 60 | stage2_peak_rel_err_abs | 0.003541 | 0.02652 | 0.04034 | 0.06003 | 0.07362 | 0.075 | 0.07572 |
| constant | 12 | 60 | stage2_peak_rel_err_abs | 0.003541 | 0.02652 | 0.04034 | 0.06003 | 0.07362 | 0.075 | 0.07572 |
| decay | 1 | 60 | stage2_peak_rel_err_abs | 0.003541 | 0.02652 | 0.04034 | 0.06003 | 0.07362 | 0.075 | 0.07572 |
| decay | 4 | 60 | stage2_peak_rel_err_abs | 0.003541 | 0.02652 | 0.04034 | 0.06003 | 0.07362 | 0.075 | 0.07572 |
| decay | 8 | 60 | stage2_peak_rel_err_abs | 0.003541 | 0.02652 | 0.04034 | 0.06003 | 0.07362 | 0.075 | 0.07572 |
| decay | 12 | 60 | stage2_peak_rel_err_abs | 0.003541 | 0.02652 | 0.04034 | 0.06003 | 0.07362 | 0.075 | 0.07572 |
| constant | 1 | 50 | stage2_peak_rel_err_signed | -0.1286 | -0.07798 | -0.0709 | -0.06835 | -0.05722 | -0.05453 | -0.03741 |
| constant | 4 | 50 | stage2_peak_rel_err_signed | -0.1286 | -0.07798 | -0.0709 | -0.06835 | -0.05722 | -0.05453 | -0.03741 |
| constant | 8 | 50 | stage2_peak_rel_err_signed | -0.1286 | -0.07798 | -0.0709 | -0.06835 | -0.05722 | -0.05453 | -0.03741 |
| constant | 12 | 50 | stage2_peak_rel_err_signed | -0.1286 | -0.07798 | -0.0709 | -0.06835 | -0.05722 | -0.05453 | -0.03741 |
| decay | 1 | 50 | stage2_peak_rel_err_signed | -0.1286 | -0.07798 | -0.0709 | -0.06835 | -0.05722 | -0.05453 | -0.03741 |
| decay | 4 | 50 | stage2_peak_rel_err_signed | -0.1286 | -0.07798 | -0.0709 | -0.06835 | -0.05722 | -0.05453 | -0.03741 |
| decay | 8 | 50 | stage2_peak_rel_err_signed | -0.1286 | -0.07798 | -0.0709 | -0.06835 | -0.05722 | -0.05453 | -0.03741 |
| decay | 12 | 50 | stage2_peak_rel_err_signed | -0.1286 | -0.07798 | -0.0709 | -0.06835 | -0.05722 | -0.05453 | -0.03741 |
| constant | 1 | 60 | stage2_peak_rel_err_signed | -0.07572 | -0.075 | -0.07362 | -0.06003 | -0.04034 | -0.02581 | 0.003541 |
| constant | 4 | 60 | stage2_peak_rel_err_signed | -0.07572 | -0.075 | -0.07362 | -0.06003 | -0.04034 | -0.02581 | 0.003541 |
| constant | 8 | 60 | stage2_peak_rel_err_signed | -0.07572 | -0.075 | -0.07362 | -0.06003 | -0.04034 | -0.02581 | 0.003541 |
| constant | 12 | 60 | stage2_peak_rel_err_signed | -0.07572 | -0.075 | -0.07362 | -0.06003 | -0.04034 | -0.02581 | 0.003541 |
| decay | 1 | 60 | stage2_peak_rel_err_signed | -0.07572 | -0.075 | -0.07362 | -0.06003 | -0.04034 | -0.02581 | 0.003541 |
| decay | 4 | 60 | stage2_peak_rel_err_signed | -0.07572 | -0.075 | -0.07362 | -0.06003 | -0.04034 | -0.02581 | 0.003541 |
| decay | 8 | 60 | stage2_peak_rel_err_signed | -0.07572 | -0.075 | -0.07362 | -0.06003 | -0.04034 | -0.02581 | 0.003541 |
| decay | 12 | 60 | stage2_peak_rel_err_signed | -0.07572 | -0.075 | -0.07362 | -0.06003 | -0.04034 | -0.02581 | 0.003541 |
| constant | 1 | 50 | stage2_rmse_pos_over_g2_0 | 0.01422 | 0.02004 | 0.0221 | 0.02605 | 0.02871 | 0.03632 | 0.05976 |
| constant | 4 | 50 | stage2_rmse_pos_over_g2_0 | 0.01422 | 0.02004 | 0.0221 | 0.02605 | 0.02871 | 0.03632 | 0.05976 |
| constant | 8 | 50 | stage2_rmse_pos_over_g2_0 | 0.01422 | 0.02004 | 0.0221 | 0.02605 | 0.02871 | 0.03632 | 0.05976 |
| constant | 12 | 50 | stage2_rmse_pos_over_g2_0 | 0.01422 | 0.02004 | 0.0221 | 0.02605 | 0.02871 | 0.03632 | 0.05976 |
| decay | 1 | 50 | stage2_rmse_pos_over_g2_0 | 0.01422 | 0.02004 | 0.0221 | 0.02605 | 0.02871 | 0.03632 | 0.05976 |
| decay | 4 | 50 | stage2_rmse_pos_over_g2_0 | 0.01422 | 0.02004 | 0.0221 | 0.02605 | 0.02871 | 0.03632 | 0.05976 |
| decay | 8 | 50 | stage2_rmse_pos_over_g2_0 | 0.01422 | 0.02004 | 0.0221 | 0.02605 | 0.02871 | 0.03632 | 0.05976 |
| decay | 12 | 50 | stage2_rmse_pos_over_g2_0 | 0.01422 | 0.02004 | 0.0221 | 0.02605 | 0.02871 | 0.03632 | 0.05976 |
| constant | 1 | 60 | stage2_rmse_pos_over_g2_0 | 0.01864 | 0.01871 | 0.02089 | 0.02389 | 0.03157 | 0.03305 | 0.03376 |
| constant | 4 | 60 | stage2_rmse_pos_over_g2_0 | 0.01864 | 0.01871 | 0.02089 | 0.02389 | 0.03157 | 0.03305 | 0.03376 |
| constant | 8 | 60 | stage2_rmse_pos_over_g2_0 | 0.01864 | 0.01871 | 0.02089 | 0.02389 | 0.03157 | 0.03305 | 0.03376 |
| constant | 12 | 60 | stage2_rmse_pos_over_g2_0 | 0.01864 | 0.01871 | 0.02089 | 0.02389 | 0.03157 | 0.03305 | 0.03376 |
| decay | 1 | 60 | stage2_rmse_pos_over_g2_0 | 0.01864 | 0.01871 | 0.02089 | 0.02389 | 0.03157 | 0.03305 | 0.03376 |
| decay | 4 | 60 | stage2_rmse_pos_over_g2_0 | 0.01864 | 0.01871 | 0.02089 | 0.02389 | 0.03157 | 0.03305 | 0.03376 |
| decay | 8 | 60 | stage2_rmse_pos_over_g2_0 | 0.01864 | 0.01871 | 0.02089 | 0.02389 | 0.03157 | 0.03305 | 0.03376 |
| decay | 12 | 60 | stage2_rmse_pos_over_g2_0 | 0.01864 | 0.01871 | 0.02089 | 0.02389 | 0.03157 | 0.03305 | 0.03376 |
| constant | 1 | 50 | stage2_tail_max_over_g2_0 | 0.02365 | 0.02917 | 0.03044 | 0.03172 | 0.03711 | 0.04461 | 0.07009 |
| constant | 4 | 50 | stage2_tail_max_over_g2_0 | 0.02365 | 0.02917 | 0.03044 | 0.03172 | 0.03711 | 0.04461 | 0.07009 |
| constant | 8 | 50 | stage2_tail_max_over_g2_0 | 0.02365 | 0.02917 | 0.03044 | 0.03172 | 0.03711 | 0.04461 | 0.07009 |
| constant | 12 | 50 | stage2_tail_max_over_g2_0 | 0.02365 | 0.02917 | 0.03044 | 0.03172 | 0.03711 | 0.04461 | 0.07009 |
| decay | 1 | 50 | stage2_tail_max_over_g2_0 | 0.02365 | 0.02917 | 0.03044 | 0.03172 | 0.03711 | 0.04461 | 0.07009 |
| decay | 4 | 50 | stage2_tail_max_over_g2_0 | 0.02365 | 0.02917 | 0.03044 | 0.03172 | 0.03711 | 0.04461 | 0.07009 |
| decay | 8 | 50 | stage2_tail_max_over_g2_0 | 0.02365 | 0.02917 | 0.03044 | 0.03172 | 0.03711 | 0.04461 | 0.07009 |
| decay | 12 | 50 | stage2_tail_max_over_g2_0 | 0.02365 | 0.02917 | 0.03044 | 0.03172 | 0.03711 | 0.04461 | 0.07009 |
| constant | 1 | 60 | stage2_tail_max_over_g2_0 | 0.02927 | 0.02982 | 0.03146 | 0.03259 | 0.03616 | 0.04017 | 0.04382 |
| constant | 4 | 60 | stage2_tail_max_over_g2_0 | 0.02927 | 0.02982 | 0.03146 | 0.03259 | 0.03616 | 0.04017 | 0.04382 |
| constant | 8 | 60 | stage2_tail_max_over_g2_0 | 0.02927 | 0.02982 | 0.03146 | 0.03259 | 0.03616 | 0.04017 | 0.04382 |
| constant | 12 | 60 | stage2_tail_max_over_g2_0 | 0.02927 | 0.02982 | 0.03146 | 0.03259 | 0.03616 | 0.04017 | 0.04382 |
| decay | 1 | 60 | stage2_tail_max_over_g2_0 | 0.02927 | 0.02982 | 0.03146 | 0.03259 | 0.03616 | 0.04017 | 0.04382 |
| decay | 4 | 60 | stage2_tail_max_over_g2_0 | 0.02927 | 0.02982 | 0.03146 | 0.03259 | 0.03616 | 0.04017 | 0.04382 |
| decay | 8 | 60 | stage2_tail_max_over_g2_0 | 0.02927 | 0.02982 | 0.03146 | 0.03259 | 0.03616 | 0.04017 | 0.04382 |
| decay | 12 | 60 | stage2_tail_max_over_g2_0 | 0.02927 | 0.02982 | 0.03146 | 0.03259 | 0.03616 | 0.04017 | 0.04382 |
| constant | 1 | 50 | stage2_tail_mean_over_g2_0 | 0.00504 | 0.005525 | 0.006513 | 0.007358 | 0.008055 | 0.009557 | 0.01262 |
| constant | 4 | 50 | stage2_tail_mean_over_g2_0 | 0.00504 | 0.005525 | 0.006513 | 0.007358 | 0.008055 | 0.009557 | 0.01262 |
| constant | 8 | 50 | stage2_tail_mean_over_g2_0 | 0.00504 | 0.005525 | 0.006513 | 0.007358 | 0.008055 | 0.009557 | 0.01262 |
| constant | 12 | 50 | stage2_tail_mean_over_g2_0 | 0.00504 | 0.005525 | 0.006513 | 0.007358 | 0.008055 | 0.009557 | 0.01262 |
| decay | 1 | 50 | stage2_tail_mean_over_g2_0 | 0.00504 | 0.005525 | 0.006513 | 0.007358 | 0.008055 | 0.009557 | 0.01262 |
| decay | 4 | 50 | stage2_tail_mean_over_g2_0 | 0.00504 | 0.005525 | 0.006513 | 0.007358 | 0.008055 | 0.009557 | 0.01262 |
| decay | 8 | 50 | stage2_tail_mean_over_g2_0 | 0.00504 | 0.005525 | 0.006513 | 0.007358 | 0.008055 | 0.009557 | 0.01262 |
| decay | 12 | 50 | stage2_tail_mean_over_g2_0 | 0.00504 | 0.005525 | 0.006513 | 0.007358 | 0.008055 | 0.009557 | 0.01262 |
| constant | 1 | 60 | stage2_tail_mean_over_g2_0 | 0.007102 | 0.008241 | 0.008588 | 0.009106 | 0.009708 | 0.01073 | 0.01093 |
| constant | 4 | 60 | stage2_tail_mean_over_g2_0 | 0.007102 | 0.008241 | 0.008588 | 0.009106 | 0.009708 | 0.01073 | 0.01093 |
| constant | 8 | 60 | stage2_tail_mean_over_g2_0 | 0.007102 | 0.008241 | 0.008588 | 0.009106 | 0.009708 | 0.01073 | 0.01093 |
| constant | 12 | 60 | stage2_tail_mean_over_g2_0 | 0.007102 | 0.008241 | 0.008588 | 0.009106 | 0.009708 | 0.01073 | 0.01093 |
| decay | 1 | 60 | stage2_tail_mean_over_g2_0 | 0.007102 | 0.008241 | 0.008588 | 0.009106 | 0.009708 | 0.01073 | 0.01093 |
| decay | 4 | 60 | stage2_tail_mean_over_g2_0 | 0.007102 | 0.008241 | 0.008588 | 0.009106 | 0.009708 | 0.01073 | 0.01093 |
| decay | 8 | 60 | stage2_tail_mean_over_g2_0 | 0.007102 | 0.008241 | 0.008588 | 0.009106 | 0.009708 | 0.01073 | 0.01093 |
| decay | 12 | 60 | stage2_tail_mean_over_g2_0 | 0.007102 | 0.008241 | 0.008588 | 0.009106 | 0.009708 | 0.01073 | 0.01093 |

---

## 7. Interpretation (labelled; descriptive, no thresholds)

- **1a.**
  - The PI-side reference values are reproduced by the verifier's own Q₁ within the fit and step-size variation on both tiers.
  - At q=50 the root best response is steep (slope ≈ −0.9 to −0.96). The game is close to the boundary where |slope| = 1, and the slope is very sensitive to E[V₂″] there.
  - At q=60 the slope is ≈ −0.30.
- **1b.**
  - In the converged segment, ê₁(0) moves with a positive one-to-two-window autocorrelation and a negative autocorrelation at 100–160 updates. This is consistent with a slow oscillation of a few hundred updates rather than independent noise per window.
  - The window regression is negative at both q: the learner moves against the opponent's deviation.
  - Under a partial-adjustment reading y ≈ λ(s − 1)x, with x the common start point of learner and opponent, the pooled slopes correspond to λ ≈ 0.08 (q=50, s ≈ −0.96) and ≈ 0.15 (q=60, s ≈ −0.31). The per-run medians give λ ≈ 0.20 and ≈ 0.28.
  - This is a reading, not an estimate of a mechanism. No experiment here separates it from alternatives such as critic lag or sampling noise.
- **1c.**
  - Tail-averaging the stage-2 mapping consistently lowers RMSE (8–10/10) and the symmetry error (7–10/10). It does not move the d = 0 peak error, which stays at about −0.06, and it slightly raises the tail mean at q=50 (CI excludes 0).
  - For stage 1 in Pilot 3, averaging reduces |stage-1 error| clearly only in the q=60 `mean` arm (8/10, CI excludes 0); elsewhere the CIs include 0.
  - At q=60 the K=12 averages of both arms sit above e₁* (+3% to +4%). Averaging removes oscillation, but over u725–u1000 the average itself is offset.
- **1d.**
  - The stage-2 actor can represent e₂* to within 0.06% at the peak and ≤ 0.004 RMSE/e₂*(0); these are upper bounds, since the loss was still falling.
  - So the RL peak gap of 6–7% is not a representation floor of this architecture and head.
  - Action-noise smoothing accounts for about 45% of it (2.16 of 4.78 at q=50, 1.58 of 3.50 at q=60). The remainder is neither smoothing nor representation, as far as these diagnostics go.
- **2a.**
  - Decay over u1201–u1600 lowers RMSE/e₂*(0) (q=60: 8/10, CI excludes 0; q=50: 6/10, CI just includes 0) and lowers KL and clip.
  - It does not change the peak error or η₂.
  - It slightly raises the tail mean (q=50 CI excludes 0) and off-path Δ₂.
  - The constant arm's tail average at K=8 reaches a similar median RMSE: 0.019 vs decay K=1 0.020 at q=50; 0.017 vs 0.020 at q=60.
- **2b.**
  - Decay over Phase B reduces the late-phase fluctuation of ê₁(0):
    - within-run SD over the last 5 exports: 1.02 vs 1.75 (q=50) and 0.99 vs 1.37 (q=60); CIs exclude 0;
    - across-seed SD of the final ê₁(0): 2.49 vs 3.33 and 1.70 vs 3.31.
  - EXP_root and dReach are lower in 6–7/10 pairs, with CIs that exclude 0 at K=1.
  - |stage-1 error| is lower in median (0.043 vs 0.063 at q=50; 0.034 vs 0.055 at q=60), but the paired CIs include 0.
  - The constant arm ends with a positive signed error at both q (median +0.049 and +0.036; K=12: +0.058 and +0.065); the decay arm ends near 0.
  - The inherited term is small (−0.003) for the u1600 parents, so the learning term again carries most of the stage-1 error.
- **Comparison to Pilot 3 (different parents, so not a controlled comparison).** The 2b constant arm (u1600 parent) has a lower Ĝmax_full and EXP_root than Pilot 3 `mean` (u400 parent): 0.0023 vs 0.0049 and 0.0015 vs 0.0023 at q=50. This is consistent with the better stage 2 from the longer Phase A.

---

## 8. Anomalies and deviations

1. **Dirty flag in 8 runs.**
   - 2b q=60 seeds 10507–10510, both arms, started at 01:27:58–01:28:06 (from the queue of 32 workers). By then I had created two new, untracked analysis scripts: `tools/v2/pilot4_root_game.py` (01:26:53) and `tools/v2/pilot4_fluctuation.py`.
   - `git_state()` counts untracked files outside `results/` as dirty.
   - The tracked tree was still `c92ee74`, and the runner imports only `agents/`, `envs/`, `run/` and `utils/`, none of which changed. The first tracked edit after the launch commit (`tools/v2/induced_band.py`) was made after all 2b runs had finished.
   - The training code of these 8 runs is therefore `c92ee74`. I did not re-run them to prove this; a re-run of one of them in a clean tree would settle it.
2. **The 1d plateau rule never fired.** All fits stopped at 300,000 steps with the loss still falling (§1d). The reported floor is an upper bound.
3. **1b used the 20-update stability log, not a per-update log.** None exists. The window regression was run because it needs only the window endpoints (§1b).
4. **2a verifier cadence.** It coincided with the extension, so the "verifier consumes no RNG" confirmation comes from a direct RNG-state check instead (§2a).
5. **Non-contiguous bands.** 7 of the 10 q=50 parent4 bands are not contiguous; the outermost points are used, as in Pilots 2 and 3.
6. **Test suite.** `tests/test_registry_canonicalization.py` fails, identically on `main`. It is a check of the paper results registry against data on disk and is unrelated to v2.
7. **Data location.** The runs and parents live in this worktree's `results/v2_pilots/`. The pre-existing parent data were copied (rsync) from the `v2-stagewise-pilots` worktree, and the originals were not modified. Full-state checkpoints and NPZ arrays are not committed (size, gitignored `.pt`); the lightweight records are committed.

---

## 9. Commands to reproduce

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider
```

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python -B run/run_v2_stagewise.py --config results/v2_pilots/phase2_regression/run_config.json --out-dir results/v2_pilots/phase2_regression/v2_full_c92ee74
```

```bash
tmux new-session -d -s v2_pilot4_A "/home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/launch_pilot.py --pilot pilot4_A --phase Acont --parent-pilot phaseA_ext --parent-arm expected_ext --parent-file state_u01200.pt --reward-mode expected --budget-overrides '{\"phase_caps\": {\"A\": 400, \"B\": 600, \"C\": 1000}}' --qs 50 60 --seeds 10501 10502 10503 10504 10505 10506 10507 10508 10509 10510 --arms constant decay --workers 32"
```

```bash
tmux new-session -d -s v2_pilot4_B "/home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/launch_pilot.py --pilot pilot4_B --phase B --parent-pilot phaseA_ext --parent-arm expected_ext --parent-file state_u01600.pt --reward-mode expected --qs 50 60 --seeds 10501 10502 10503 10504 10505 10506 10507 10508 10509 10510 --arms B2_mean_constant B2_mean_decay --workers 32"
```

```bash
OMP_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/pilot4_root_game.py && OMP_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/pilot4_fluctuation.py && OMP_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/pilot4_repro.py && OMP_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/pilot4_repro.py --verifier-rng && OMP_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/induced_band.py parents4 --workers 60 && OMP_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/pilot4_repr_floor.py --workers 10 && OMP_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/pilot4_analysis.py --workers 50 && /home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/pilot4_tables.py
```
