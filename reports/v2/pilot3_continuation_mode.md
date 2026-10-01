# Pilot 3 — Continuation action mode: stochastic vs mean (frozen B2)

Date: 2026-10-01. Branch `v2-stagewise-pilots`.

| Item | Value |
|---|---|
| **Base commit** | `657f54a` |
| **Launch commit for all 40 runs** | **`cd760fd`** (all manifests `dirty = false`; `A/final_table.csv` columns `commit`, `dirty`) |
| Reward | `expected` |
| Mode | frozen B2 (`stage2_update_mode = frozen`, `adv_norm_scope = stage1_rows`) |

Path conventions:
- `P3/` means `results/v2_pilots/pilot3/`.
- `A/` means `P3/analysis/`, produced by `tools/v2/decomposition.py --pilot pilot3` and `tools/v2/pilot3_analysis.py`.
- Figures are in `reports/v2/figures/pilot3/`.

ΔW = 4. Arms: **stochastic** = `B2_frozen_s1norm`, **mean** = `B2_frozen_s1norm_mean`.

---

## 0. Note on the evaluated object

- **The verifier always evaluates the deterministic mapping** ê (the Beta mean): the live actor at t=1 and the frozen snapshot at t=2.
- **The `mean` arm** trains stage 1 against exactly that continuation: both players play the frozen Beta mean at stage 2.
- **The `stochastic` arm** trains stage 1 against a different continuation: actions sampled from the frozen Beta.
- **The induced target ẽ₁** is defined with the deterministic ê₂. So in the `stochastic` arm the learning term ê₁ − ẽ₁ also contains this mode mismatch. In the `mean` arm it does not.

---

## 1. What was done

### Design

- **Parents.** The same 20 Pilot-1 `expected` end-of-Phase-A states as Pilot 2 (q ∈ {50, 60} × seeds 10501–10510).
- **Phase B.** 600 updates (global 401–1000), fixed budget, root starts, stage-1 opponent = the lagged copy refreshed every 20 updates.
- **Arms.** `continuation_action_mode` ∈ {stochastic, mean}. In `mean`, both players' stage-2 Beta draws are still made and then discarded (A6).
- **Runs.** 40.
- **Launch.**
  - tmux session `v2_pilot3`, 40 workers, in parallel with the 20 Phase-A-extension runs.
  - `nproc --all` = 64. Load average 0.97 / 0.81 / 2.43 at launch and 61.19 / 31.62 / 14.37 at the end (`P3/launch_20261001_233012.json`).
  - 40/40 returncode 0; launcher wall 193.3–199.9 s per run.
- **Code.**
  - The superseded induced-target solver is now opt-in (`evaluate(..., legacy_induced=False)`), so Pilot 3 checkpoints do not carry it.
  - The decomposition uses the new residual-band method (D2, §1 of the request; details and calibration in `reports/v2/pilot2_freeze.md` §6).

### ẽ₁ for Pilot 3

Stage 2 is frozen at the parent, so ẽ₁ and its band are those of the parent:
- source: `results/v2_pilots/induced_band/parent_bands.csv`;
- final tier, sweep [0.4·e₁*, 1.7·e₁*], step 0.01, anchored at e₁*.

The learning term at a checkpoint is ê₁(0) − ẽ₁, with interval [ê₁ − band_hi, ê₁ − band_lo]. Decomposition rows: `A/decomposition_residual_band.csv` (every training-time checkpoint and every weight export; 1,800 rows).

---

## 2. Reproducibility check (stochastic arm vs Pilot 2 B2)

Source: `A/reproducibility_vs_pilot2_B2.csv`. Each of the 20 Pilot-3 `stochastic` runs was compared with the Pilot-2 `B2_frozen_s1norm` run of the same (q, seed).

| run | history_identical | verifier_calls_identical | stability_identical | final_weights_identical | weight_exports_identical | end_state_actor_critic_opt_identical | n_updates |
|---|---|---|---|---|---|---|---|
| q50/seed10501/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q50/seed10502/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q50/seed10503/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q50/seed10504/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q50/seed10505/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q50/seed10506/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q50/seed10507/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q50/seed10508/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q50/seed10509/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q50/seed10510/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q60/seed10501/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q60/seed10502/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q60/seed10503/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q60/seed10504/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q60/seed10505/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q60/seed10506/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q60/seed10507/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q60/seed10508/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q60/seed10509/B2_frozen_s1norm | True | True | True | True | True | True | 600 |
| q60/seed10510/B2_frozen_s1norm | True | True | True | True | True | True | 600 |

**All 20 runs are bit-identical:**
- training history (600 updates);
- verifier calls (excluding `time_sec`);
- stability log;
- final weights;
- every weight export;
- end-state actor, critic, lagged opponent and frozen snapshot.

The code commit differs (Pilot 2 ran at `1791687`, Pilot 3 at `cd760fd`); the training path is unchanged.

**Inherited-term identity within pairs.** Source: `A/inherited_identity.csv`.
- The frozen snapshot tensors are bit-identical between the two arms of every pair (20/20).
- The inherited term takes a single identical value in both arms for each (q, seed).

So arm differences in total stage-1 error equal arm differences in the learning term. This is confirmed numerically in §4.2: the paired differences of `stage1_rel_err_signed` and `learning_rel` agree to within the accuracy shown.

| q | seed | snapshots_bit_identical |
|---|---|---|
| 50 | 10501 | True |
| 50 | 10502 | True |
| 50 | 10503 | True |
| 50 | 10504 | True |
| 50 | 10505 | True |
| 50 | 10506 | True |
| 50 | 10507 | True |
| 50 | 10508 | True |
| 50 | 10509 | True |
| 50 | 10510 | True |
| 60 | 10501 | True |
| 60 | 10502 | True |
| 60 | 10503 | True |
| 60 | 10504 | True |
| 60 | 10505 | True |
| 60 | 10506 | True |
| 60 | 10507 | True |
| 60 | 10508 | True |
| 60 | 10509 | True |
| 60 | 10510 | True |

---

## 3. Final checkpoint (global update 1000), all 40 runs

Source: `A/final_table.csv`.

Units and definitions:
- e1 rel err = (ê₁(0) − e₁*)/e₁*.
- learning and inherited terms are /e₁*, with bands from the residual sweep.
- SD/range last5 = within-run SD and range of ê₁(0) (effort units) over the weight exports at global 900, 925, 950, 975 and 1000 (local Phase-B updates 500–600; n = 5).
- Ĝmax, EXP, dReach, Δmax_all and dFull are /ΔW.

| q | seed | arm | ê₁(0) | e1 rel err | learning | learning band | inherited | inherited band | σ₁(0) | SD last5 | range last5 | Ĝmax | t* | d* | EXP | dReach | Δmax_all | dFull | KL | clip | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10501 | stochastic | 51.7 | 0.1078 | 0.1059 | [0.1037, 0.1059] | 0.001929 | [0.0019, 0.0041] | 4.107 | 2.047 | 4.955 | 0.006638 | 2 | -4 | 0.004917 | 0.009355 | 0.006638 | 0.009355 | 0.006559 | 0.03635 | 189.6 |
| 50 | 10501 | mean | 45.82 | -0.01807 | -0.02 | [-0.0221, -0.0200] | 0.001929 | [0.0019, 0.0041] | 4.195 | 1.727 | 4.289 | 0.006638 | 2 | -4 | 0.002166 | 0.006768 | 0.006638 | 0.006768 | 0.0002185 | 0.03972 | 187.8 |
| 50 | 10502 | stochastic | 42.94 | -0.07982 | -0.09097 | [-0.0916, -0.0910] | 0.01114 | [0.0111, 0.0118] | 3.274 | 1.295 | 3.573 | 0.004646 | 2 | -4 | 0.00338 | 0.006663 | 0.004646 | 0.006663 | 0.01057 | 0.1208 | 188.7 |
| 50 | 10502 | mean | 43.02 | -0.07812 | -0.08926 | [-0.0899, -0.0893] | 0.01114 | [0.0111, 0.0118] | 3.298 | 0.7123 | 1.725 | 0.004646 | 2 | -4 | 0.00331 | 0.006592 | 0.004646 | 0.006592 | 0.01861 | 0.1056 | 189.2 |
| 50 | 10503 | stochastic | 48.07 | 0.03017 | 0.0141 | [0.0124, 0.0141] | 0.01607 | [0.0161, 0.0178] | 3.905 | 2.318 | 5.308 | 0.005119 | 2 | -4 | 0.001006 | 0.005124 | 0.005119 | 0.005124 | 0.008548 | 0.03052 | 187 |
| 50 | 10503 | mean | 43.13 | -0.0758 | -0.09187 | [-0.0936, -0.0919] | 0.01607 | [0.0161, 0.0178] | 3.715 | 2.36 | 6.279 | 0.005119 | 2 | -4 | 0.003001 | 0.007145 | 0.005119 | 0.007145 | 0.03002 | 0.1043 | 189.4 |
| 50 | 10504 | stochastic | 48.61 | 0.04166 | 0.05344 | [0.0534, 0.0534] | -0.01179 | [-0.0118, -0.0118] | 3.571 | 2.449 | 6.718 | 0.003827 | 2 | -4 | 0.001399 | 0.004565 | 0.003827 | 0.004565 | 0.01014 | 0.02709 | 187.2 |
| 50 | 10504 | mean | 49.44 | 0.05946 | 0.07125 | [0.0713, 0.0713] | -0.01179 | [-0.0118, -0.0118] | 3.585 | 2.126 | 5.59 | 0.003827 | 2 | -4 | 0.002001 | 0.005157 | 0.003827 | 0.005157 | 0.008031 | 0.08999 | 191.5 |
| 50 | 10505 | stochastic | 47.86 | 0.02551 | 0.03044 | [0.0294, 0.0304] | -0.004929 | [-0.0049, -0.0039] | 3.501 | 1.379 | 3.277 | 0.004565 | 2 | -4 | 0.0009865 | 0.004808 | 0.004565 | 0.004808 | 0.01426 | 0.1007 | 188.2 |
| 50 | 10505 | mean | 46.43 | -0.005032 | -0.0001035 | [-0.0012, -0.0001] | -0.004929 | [-0.0049, -0.0039] | 3.414 | 1.16 | 2.789 | 0.004565 | 2 | -4 | 0.0007219 | 0.004569 | 0.004565 | 0.004569 | 0.007925 | 0.06599 | 187.6 |
| 50 | 10506 | stochastic | 47.31 | 0.01374 | 0.01695 | [0.0161, 0.0170] | -0.003214 | [-0.0032, -0.0024] | 3.867 | 1.846 | 5.054 | 0.00635 | 2 | -4 | 0.001238 | 0.006405 | 0.00635 | 0.006405 | 0.02376 | 0.0847 | 192.2 |
| 50 | 10506 | mean | 42.95 | -0.07964 | -0.07643 | [-0.0773, -0.0764] | -0.003214 | [-0.0032, -0.0024] | 3.86 | 0.5107 | 1.321 | 0.00635 | 2 | -4 | 0.002506 | 0.007742 | 0.00635 | 0.007742 | 0.005007 | 0.07816 | 187.7 |
| 50 | 10507 | stochastic | 42.23 | -0.09501 | -0.09586 | [-0.0982, -0.0959] | 0.0008571 | [0.0009, 0.0032] | 3.357 | 1.602 | 4.085 | 0.005955 | 2 | -4 | 0.003461 | 0.008134 | 0.005955 | 0.008134 | 0.007934 | 0.07352 | 189.4 |
| 50 | 10507 | mean | 44.86 | -0.03863 | -0.03948 | [-0.0418, -0.0395] | 0.0008571 | [0.0009, 0.0032] | 3.503 | 0.5451 | 1.282 | 0.005955 | 2 | -4 | 0.001704 | 0.006323 | 0.005955 | 0.006323 | 0.007728 | 0.05026 | 189.9 |
| 50 | 10508 | stochastic | 42.95 | -0.07957 | -0.07614 | [-0.0809, -0.0761] | -0.003429 | [-0.0034, 0.0013] | 3.835 | 3.121 | 6.648 | 0.00185 | 1 | 0 | 0.00185 | 0.002998 | 0.001511 | 0.002998 | 6.381e-05 | 0.01594 | 190.4 |
| 50 | 10508 | mean | 40.06 | -0.1416 | -0.1382 | [-0.1429, -0.1382] | -0.003429 | [-0.0034, 0.0013] | 3.663 | 1.819 | 4.723 | 0.005093 | 1 | 0 | 0.005093 | 0.006246 | 0.004735 | 0.006246 | 0.003293 | 0.04634 | 190.3 |
| 50 | 10509 | stochastic | 44.33 | -0.05001 | -0.05836 | [-0.0629, -0.0584] | 0.008357 | [0.0084, 0.0129] | 3.72 | 2.727 | 6.108 | 0.004308 | 2 | -4 | 0.001559 | 0.005184 | 0.004308 | 0.005184 | 0.001829 | 0.1021 | 190 |
| 50 | 10509 | mean | 47.17 | 0.01069 | 0.002329 | [-0.0022, 0.0023] | 0.008357 | [0.0084, 0.0129] | 3.757 | 0.9287 | 1.966 | 0.004308 | 2 | -4 | 0.0006932 | 0.004308 | 0.004308 | 0.004308 | 0.001857 | 0.05623 | 189.7 |
| 50 | 10510 | stochastic | 51.67 | 0.1073 | 0.09765 | [0.0929, 0.0976] | 0.009643 | [0.0096, 0.0144] | 3.376 | 1.273 | 2.665 | 0.004566 | 2 | -4 | 0.003965 | 0.00694 | 0.004566 | 0.00694 | 0.01669 | 0.1042 | 190 |
| 50 | 10510 | mean | 51.57 | 0.105 | 0.09539 | [0.0907, 0.0954] | 0.009643 | [0.0096, 0.0144] | 3.395 | 0.8438 | 1.799 | 0.004566 | 2 | -4 | 0.003841 | 0.006816 | 0.004566 | 0.006816 | 0.01271 | 0.08209 | 187.2 |
| 60 | 10501 | stochastic | 36.17 | -0.07002 | -0.07079 | [-0.0731, -0.0703] | 0.0007714 | [0.0003, 0.0031] | 3.417 | 2.253 | 5.48 | 0.001831 | 2 | -4 | 0.001058 | 0.002537 | 0.001831 | 0.002537 | 0.002329 | 0.09293 | 187.2 |
| 60 | 10501 | mean | 32.1 | -0.1746 | -0.1754 | [-0.1777, -0.1749] | 0.0007714 | [0.0003, 0.0031] | 3.457 | 2.114 | 5.577 | 0.00445 | 1 | 0 | 0.00445 | 0.005943 | 0.004111 | 0.005943 | 0.003927 | 0.05771 | 187.3 |
| 60 | 10502 | stochastic | 36.98 | -0.04905 | -0.04416 | [-0.0462, -0.0436] | -0.004886 | [-0.0054, -0.0028] | 3.315 | 1.988 | 4.954 | 0.00225 | 2 | -4 | 0.0006408 | 0.002562 | 0.00225 | 0.002562 | 0.003429 | 0.0562 | 185.2 |
| 60 | 10502 | mean | 37.4 | -0.03833 | -0.03344 | [-0.0355, -0.0329] | -0.004886 | [-0.0054, -0.0028] | 3.369 | 1.223 | 2.706 | 0.00225 | 2 | -4 | 0.0005325 | 0.002454 | 0.00225 | 0.002454 | 0.02017 | 0.08094 | 187.9 |
| 60 | 10503 | stochastic | 39.54 | 0.01667 | 0.01153 | [0.0092, 0.0120] | 0.005143 | [0.0046, 0.0075] | 3.756 | 0.8926 | 2.329 | 0.001235 | 2 | -32 | 0.0005112 | 0.00125 | 0.001235 | 0.00125 | 0.004548 | 0.06375 | 187.4 |
| 60 | 10503 | mean | 43.13 | 0.1092 | 0.104 | [0.1017, 0.1045] | 0.005143 | [0.0046, 0.0075] | 3.849 | 1.861 | 4.288 | 0.001954 | 1 | 0 | 0.001954 | 0.002695 | 0.00146 | 0.002695 | 0.01152 | 0.1303 | 188.4 |
| 60 | 10504 | stochastic | 41.1 | 0.05691 | 0.06257 | [0.0603, 0.0631] | -0.005657 | [-0.0062, -0.0033] | 3.345 | 1.175 | 2.709 | 0.001181 | 2 | -4 | 0.0008224 | 0.001708 | 0.001181 | 0.001708 | 0.006533 | 0.0916 | 190.6 |
| 60 | 10504 | mean | 33.68 | -0.1338 | -0.1282 | [-0.1305, -0.1276] | -0.005657 | [-0.0062, -0.0033] | 3.208 | 2.381 | 6.232 | 0.002534 | 1 | 0 | 0.002534 | 0.00343 | 0.002249 | 0.00343 | 0.005022 | 0.04706 | 189.7 |
| 60 | 10505 | stochastic | 39 | 0.002742 | 0.00377 | [0.0015, 0.0043] | -0.001029 | [-0.0015, 0.0013] | 3.708 | 1.805 | 4.603 | 0.002252 | 2 | -4 | 0.0005495 | 0.002252 | 0.002252 | 0.002252 | 0.0007347 | 0.1014 | 192.2 |
| 60 | 10505 | mean | 38.07 | -0.02108 | -0.02005 | [-0.0224, -0.0195] | -0.001029 | [-0.0015, 0.0013] | 3.758 | 2.607 | 5.994 | 0.002252 | 2 | -4 | 0.0006437 | 0.00235 | 0.002252 | 0.00235 | 0.004474 | 0.05534 | 189.8 |
| 60 | 10506 | stochastic | 39.82 | 0.02381 | 0.02407 | [0.0220, 0.0246] | -0.0002571 | [-0.0008, 0.0018] | 3.318 | 1.501 | 3.803 | 0.00177 | 2 | -4 | 0.0003708 | 0.001844 | 0.00177 | 0.001844 | 0.004536 | 0.06715 | 186 |
| 60 | 10506 | mean | 41.53 | 0.06787 | 0.06812 | [0.0661, 0.0686] | -0.0002571 | [-0.0008, 0.0018] | 3.381 | 1.001 | 2.595 | 0.00177 | 2 | -4 | 0.0009201 | 0.002392 | 0.00177 | 0.002392 | 0.0005334 | 0.06927 | 189 |
| 60 | 10507 | stochastic | 36.95 | -0.04991 | -0.06148 | [-0.0638, -0.0612] | 0.01157 | [0.0113, 0.0139] | 3.394 | 3.488 | 7.904 | 0.001835 | 2 | 40 | 0.001118 | 0.002371 | 0.001835 | 0.002371 | 0.009554 | 0.05765 | 187.4 |
| 60 | 10507 | mean | 43.44 | 0.1171 | 0.1055 | [0.1032, 0.1058] | 0.01157 | [0.0113, 0.0139] | 3.534 | 1.155 | 3.128 | 0.002027 | 1 | 0 | 0.002027 | 0.003308 | 0.001835 | 0.003308 | 0.003227 | 0.05203 | 188.7 |
| 60 | 10508 | stochastic | 33.31 | -0.1434 | -0.1439 | [-0.1465, -0.1434] | 0.0005143 | [0.0000, 0.0031] | 3.676 | 2.513 | 6.575 | 0.003512 | 2 | -4 | 0.003287 | 0.006242 | 0.003512 | 0.006242 | 0.002133 | 0.124 | 189.4 |
| 60 | 10508 | mean | 36.9 | -0.05116 | -0.05168 | [-0.0542, -0.0512] | 0.0005143 | [0.0000, 0.0031] | 3.758 | 2.184 | 5.03 | 0.003512 | 2 | -4 | 0.0009508 | 0.0039 | 0.003512 | 0.0039 | 0.0002702 | 0.05101 | 186.2 |
| 60 | 10509 | stochastic | 39.93 | 0.02678 | 0.01958 | [0.0173, 0.0198] | 0.0072 | [0.0069, 0.0095] | 3.553 | 1.8 | 4.617 | 0.003374 | 2 | -4 | 0.0004758 | 0.003404 | 0.003374 | 0.003404 | 0.00379 | 0.07379 | 193 |
| 60 | 10509 | mean | 37.88 | -0.02584 | -0.03304 | [-0.0354, -0.0328] | 0.0072 | [0.0069, 0.0095] | 3.495 | 2.129 | 5.33 | 0.003374 | 2 | -4 | 0.0006397 | 0.003579 | 0.003374 | 0.003579 | 0.005143 | 0.0407 | 191.5 |
| 60 | 10510 | stochastic | 41.78 | 0.07444 | 0.06827 | [0.0660, 0.0688] | 0.006171 | [0.0057, 0.0085] | 3.589 | 0.6852 | 1.746 | 0.001766 | 2 | -4 | 0.0009607 | 0.002395 | 0.001766 | 0.002395 | 0.002803 | 0.01316 | 190.8 |
| 60 | 10510 | mean | 43.92 | 0.1293 | 0.1231 | [0.1208, 0.1236] | 0.006171 | [0.0057, 0.0085] | 3.501 | 1.641 | 3.661 | 0.002391 | 1 | 0 | 0.002391 | 0.00383 | 0.002064 | 0.00383 | 0.0033 | 0.04995 | 192.3 |

Medians per (q, arm), from the same file:

| q | arm | stage1_rel_err_signed | stage1_rel_err_abs | learning_rel | inherited_rel | sigma_effort_at_0_t1 | Gmax_full_over_dw | EXP_root_over_dw | dReach_over_dw |
|---|---|---|---|---|---|---|---|---|---|
| 50 | mean | -0.02835 | 0.06763 | -0.02974 | 0.001393 | 3.624 | 0.00487 | 0.002336 | 0.006458 |
| 50 | stochastic | 0.01962 | 0.06479 | 0.01553 | 0.001393 | 3.646 | 0.004606 | 0.001705 | 0.005795 |
| 60 | mean | -0.02346 | 0.08852 | -0.02655 | 0.0006429 | 3.498 | 0.002321 | 0.001452 | 0.003369 |
| 60 | stochastic | 0.009708 | 0.04948 | 0.007651 | 0.0006429 | 3.485 | 0.001833 | 0.0007316 | 0.002383 |

---

## 4. Paired differences (mean − stochastic) per (q, seed)

Source: `A/paired_differences.csv` and `A/paired_summary.csv`.

- **Bootstrap:** 95% percentile CI of the mean paired difference, 10,000 resamples, numpy seed 20261001.
- **"Better":** for metrics where smaller is better, the number of the 10 pairs with mean − stochastic < 0, i.e. pairs where `mean` is better. Signed errors, σ₁, KL and clip have no preferred direction, so their negative, positive and zero counts are given instead.

### 4.1 q = 50
| label | mean | sd | median | min | max | better | CI95 |
|---|---|---|---|---|---|---|---|
| stage-1 rel. err (signed) | -0.02835 | 0.06644 | -0.0164 | -0.1259 | 0.06069 | n/a (6−/4+/0=) | [-0.06798, 0.01123] |
| stage-1 abs rel. err | -0.00185 | 0.05139 | -0.001984 | -0.08975 | 0.0659 | 6/10 | [-0.03213, 0.02782] |
| learning term (e1 - e~1)/e1* (signed) | -0.02835 | 0.06644 | -0.0164 | -0.1259 | 0.06069 | n/a (6−/4+/0=) | [-0.06742, 0.01059] |
| abs learning term / e1* | -0.00155 | 0.05613 | -0.001984 | -0.08589 | 0.07777 | 6/10 | [-0.03484, 0.03169] |
| inherited term (e~1 - e1*)/e1* | 0 | 0 | 0 | 0 | 0 | n/a (0−/0+/10=) | [0, 0] |
| sigma_1(0) | -0.01306 | 0.107 | 0.01649 | -0.1901 | 0.1458 | n/a (4−/6+/0=) | [-0.0779, 0.04746] |
| Gmax_full/DW | 0.0003242 | 0.001025 | 0 | 0 | 0.003242 | 0/10 | [0, 0.0009727] |
| EXP_root/DW | 0.0001275 | 0.001757 | -9.664e-05 | -0.002751 | 0.003242 | 6/10 | [-0.0008869, 0.001156] |
| dReach/DW | 0.0001492 | 0.001744 | -9.698e-05 | -0.002587 | 0.003249 | 6/10 | [-0.0008788, 0.001157] |
| Delta_max_all/DW | 0.0003224 | 0.001019 | 0 | 0 | 0.003224 | 0/10 | [0, 0.0009671] |
| dFull/DW | 0.0001492 | 0.001744 | -9.698e-05 | -0.002587 | 0.003249 | 6/10 | [-0.0008363, 0.001191] |
| within-run SD of e1(0), last 5 exports | -0.7325 | 0.6002 | -0.5063 | -1.798 | 0.04201 | 9/10 | [-1.099, -0.4001] |
| within-run range of e1(0), last 5 exports | -1.663 | 1.566 | -1.488 | -4.141 | 0.9705 | 9/10 | [-2.61, -0.7656] |
| KL (last update) | -0.0004953 | 0.01047 | -0.001156 | -0.01876 | 0.02147 | n/a (6−/4+/0=) | [-0.006362, 0.005774] |
| clip fraction (last update) | 0.002275 | 0.04069 | -0.01086 | -0.04591 | 0.07377 | n/a (6−/4+/0=) | [-0.02024, 0.02747] |
| Phase B wall (s) | -0.2179 | 2.498 | -0.1814 | -4.484 | 4.305 | 6/10 | [-1.677, 1.277] |
### 4.2 q = 60
| label | mean | sd | median | min | max | better | CI95 |
|---|---|---|---|---|---|---|---|
| stage-1 rel. err (signed) | 0.008956 | 0.1053 | 0.02739 | -0.1907 | 0.167 | n/a (4−/6+/0=) | [-0.05426, 0.0682] |
| stage-1 abs rel. err | 0.03546 | 0.05894 | 0.04944 | -0.09222 | 0.1046 | 3/10 | [-0.0009213, 0.06754] |
| learning term (e1 - e~1)/e1* (signed) | 0.008956 | 0.1053 | 0.02739 | -0.1907 | 0.167 | n/a (4−/6+/0=) | [-0.05407, 0.06945] |
| abs learning term / e1* | 0.03325 | 0.05648 | 0.04406 | -0.09222 | 0.1046 | 2/10 | [-0.002521, 0.06367] |
| inherited term (e~1 - e1*)/e1* | 0 | 0 | 0 | 0 | 0 | n/a (0−/0+/10=) | [0, 0] |
| sigma_1(0) | 0.02405 | 0.08834 | 0.05218 | -0.1371 | 0.1403 | n/a (3−/7+/0=) | [-0.03091, 0.07312] |
| Gmax_full/DW | 0.0005506 | 0.0008562 | 9.555e-05 | 0 | 0.002618 | 0/10 | [0.0001198, 0.001119] |
| EXP_root/DW | 0.0007247 | 0.001489 | 0.0007288 | -0.002337 | 0.003392 | 2/10 | [-0.0001783, 0.001573] |
| dReach/DW | 0.0007316 | 0.001493 | 0.0007421 | -0.002342 | 0.003406 | 2/10 | [-0.0001899, 0.001589] |
| Delta_max_all/DW | 0.0003871 | 0.0007438 | 0 | 0 | 0.00228 | 0/10 | [2.981e-05, 0.0008954] |
| dFull/DW | 0.0007316 | 0.001493 | 0.0007421 | -0.002342 | 0.003406 | 2/10 | [-0.0001821, 0.001583] |
| within-run SD of e1(0), last 5 exports | 0.0197 | 1.078 | 0.09531 | -2.332 | 1.205 | 5/10 | [-0.6642, 0.5959] |
| within-run range of e1(0), last 5 exports | -0.01807 | 2.451 | 0.4047 | -4.777 | 3.523 | 4/10 | [-1.514, 1.415] |
| KL (last update) | 0.00172 | 0.006495 | 0.0009252 | -0.006327 | 0.01674 | n/a (4−/6+/0=) | [-0.001599, 0.005863] |
| clip fraction (last update) | -0.01074 | 0.04353 | -0.01935 | -0.07301 | 0.06654 | n/a (6−/4+/0=) | [-0.03501, 0.01562] |
| Phase B wall (s) | 0.187 | 2.096 | 0.5647 | -3.144 | 3.072 | 4/10 | [-1.048, 1.382] |

---

## 5. Learning curves

Figures:
- `reports/v2/figures/pilot3/curves_q50.png`
- `reports/v2/figures/pilot3/curves_q60.png`

Data: `A/curves_weights_every25.csv`.
- Each figure shows the median and IQR over 10 seeds per arm, at global updates 425–1000 (weight exports every 25 updates).
- Panels: stage-1 relative error (signed), learning term, σ₁(0), Ĝmax_full/ΔW, EXP_root/ΔW.

---

## 6. Stability

Source: `A/stability.csv`.
- Across-seed statistics are taken over the 10 final ê₁(0) values per (q, arm), in effort units.
- Within-run statistics are medians over runs of the last-5-export SD and range defined in §3.

| q | arm | across_seed_sd_final_e1 | across_seed_iqr_final_e1 | median_within_run_sd_last5 | median_within_run_range_last5 | median_sigma1_0 |
|---|---|---|---|---|---|---|
| 50 | stochastic | 3.506 | 5.178 | 1.947 | 5.004 | 3.646 |
| 50 | mean | 3.405 | 3.934 | 1.044 | 2.378 | 3.624 |
| 60 | stochastic | 2.576 | 2.945 | 1.802 | 4.61 | 3.485 |
| 60 | mean | 4.11 | 5.709 | 1.988 | 4.659 | 3.498 |

---

## 7. Records

### Snapshot integrity, stop rule and wall time

Source: `A/final_table.csv`, from each run's `drift_test.json` and `v2_run_summary.json`.
- The frozen snapshot's mean/α/β drift is exactly 0 in all 40 runs, and `drift_test` passes in 40/40.
- "Would-have-fired" is the existing Phase-B rule.

| q | arm | n_would_fire | median_update | min_update | max_update | wall_median | snapshot_drift_max | drift_test_pass |
|---|---|---|---|---|---|---|---|---|
| 50 | mean | 5 | 575 | 550 | 725 | 189.3 | 0 | 10 |
| 50 | stochastic | 5 | 575 | 550 | 675 | 189.5 | 0 | 10 |
| 60 | mean | 6 | 587.5 | 575 | 625 | 188.9 | 0 | 10 |
| 60 | stochastic | 6 | 600 | 550 | 650 | 188.4 | 0 | 10 |

### Advantage statistics, KL and clip fraction

Medians over all updates; `A/adv_stats_median.csv`.

| q | arm | adv_all_mean | adv_all_std | adv_s1_mean | adv_s1_std | adv_used_std | kl_final_epoch | clip_frac |
|---|---|---|---|---|---|---|---|---|
| 50 | B2_frozen_s1norm | 0.001373 | 0.8601 | 0.002309 | 1.21 | 1.21 | 0.003628 | 0.06245 |
| 50 | B2_frozen_s1norm_mean | 0.001531 | 0.8556 | 0.002784 | 1.209 | 1.209 | 0.00336 | 0.06079 |
| 60 | B2_frozen_s1norm | 0.0007164 | 0.8434 | 0.001256 | 1.188 | 1.188 | 0.003346 | 0.06121 |
| 60 | B2_frozen_s1norm_mean | 0.0003998 | 0.8404 | 0.001209 | 1.187 | 1.187 | 0.003531 | 0.061 |

### RNG stream divergence (stochastic vs mean)

Source: `A/rng_divergence.csv`; the entries are global update numbers.

| q | env | learn | opp | start | minibatch |
|---|---|---|---|---|---|
| 50 | never (10/10) | 10/10; min 404, median 426, max 517 | 10/10; min 406, median 441, max 484 | never (10/10) | never (10/10) |
| 60 | never (10/10) | 10/10; min 406, median 446, max 524 | 10/10; min 407, median 439, max 534 | never (10/10) | never (10/10) |

---

## 8. Anomalies and deviations

1. **Sweep coverage.** 6 of 1,800 decomposition rows, all at the earliest exports (u425/u450, q=60), have ê₁(0) up to 1.92·e₁*, outside the parent sweep's upper bound of 1.7·e₁* (`A/decomposition_residual_band.csv` column `e1_inside_sweep`). ẽ₁ and its band do not depend on ê₁, so these rows' decomposition values are still defined. But the request that the sweep "cover the current ê₁(0)" is not met for them. No final-checkpoint row is affected.
2. **Non-contiguous bands.** 4 of the 20 parent bands at q=50 (seeds 10505, 10508, 10509, 10510) are not contiguous: Δ₁ dips back to the minimum after rising (`results/v2_pilots/induced_band/parent_bands.csv` column `band_contiguous`). The interval endpoints used are the outermost band points.
3. **Action-noise RNG streams** (learn, opp) desynchronize between arms 4–134 updates after branching (global 400), as in every pilot. The env, start and minibatch streams never diverge (§7). The `mean` arm still makes and discards its stage-2 draws, but the arms' live stage-1 policies differ after the first update, so the rejection-based Beta sampler eventually consumes different numbers of draws.

---

## 9. Interpretation (labelled; descriptive)

- **Stage-1 accuracy.** Neither mode is consistently more accurate.
  - At q=50, `mean` has the smaller |stage-1 error| in 6/10 pairs, with a CI that includes 0.
  - At q=60, `stochastic` is better in 7/10 pairs; the CI is [−0.0009, 0.068], which just includes 0.
  - In both arms ê₁(0) keeps oscillating around e₁* by a few percent through update 1000 (curves). The final signed errors have opposite medians in the two arms, which is consistent with an oscillation phase difference rather than a bias.
- **Decomposition.** The inherited term is small: the parent bands put ẽ₁ within about 0.02·e₁* of e₁*. The learning term carries almost all of the remaining stage-1 error. Its band excludes 0 in 39 of 40 final rows.
- **Stability.**
  - At q=50, the `mean` arm has a smaller within-run SD of ê₁(0) over the last 5 exports in 9/10 pairs, with a CI that excludes 0 (median within-run SD 1.04 vs 1.95 effort units).
  - At q=60 there is no consistent difference (5/10).
  - Across-seed SD of the final ê₁(0): q=50 is similar (3.41 vs 3.51); q=60 is larger for `mean` (4.11 vs 2.58).
- **Strategic quality.**
  - EXP_root and dReach are higher with `mean` in 8/10 pairs at q=60, with CIs that just include 0.
  - Ĝmax_full is mostly set by the frozen stage 2 and is identical in most pairs.
- **Mode mismatch.** The `stochastic` arm's learning term includes the stochastic-vs-deterministic continuation mismatch, and its stage-1 error is not systematically larger than that of `mean`. Over this horizon, the mismatch is therefore not a dominant error source.
