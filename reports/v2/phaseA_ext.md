# Phase A extension — stage-2-only training continued from update 400 to update 1600

Date: 2026-10-01. Branch `v2-stagewise-pilots`.

| Item | Value |
|---|---|
| **Base commit** | `657f54a` |
| **Launch commit for all 20 runs** | **`cd760fd`** (all manifests `dirty = false`; `X/runs.csv`) |

`PX/` means `results/v2_pilots/phaseA_ext/` and `X/` means `PX/analysis/`, produced by `tools/v2/phaseA_ext_analysis.py`. Figures are in `reports/v2/figures/phaseA_ext/`.

These are descriptive results. **No precision criterion or Phase A budget is proposed.**

---

## 1. What was done

### Design

- **Parents.** The 20 Pilot-1 `expected` end-of-Phase-A full states (`results/v2_pilots/pilot1/q*/seed*/expected/state_end_A.pt`, update 400). They were verified before Pilot 2 (`results/v2_pilots/pilot2_parents_check.json`).
- **Continuation.** Phase A continues for 1,200 more updates, to global update 1600. It uses the same start-state distribution (stage 2, bin-balanced exploring starts on D₂), `reward_mode = expected`, the same hyperparameters, and `fixed_budget = true`.
- **Code.** New run mode `phase_A_continue` in `run/run_v2_stagewise.py` (commit `cd760fd`), plus the optional config field `full_state_at`.
- **Saved state.** Full-state checkpoints `state_u00800.pt`, `state_u01200.pt`, `state_u01600.pt` and `state_end_A.pt` (all present in every run; `X/runs.csv` column `full_states`). Weight exports every 25 updates.

### Schedules

These are the schedules that depend on the update counter or the Phase A budget:
- **Learning rate.** Phase A uses the constant `ab_lr` = 3e−4 for actor and critic (`run/run_final_dp_br_round3_dense.py:155-156`, `lr_at`). The C-phase linear schedule does not apply in A.
- **Entropy coefficient:** 0, constant (`PPOConfig.entropy_coef`; no schedule exists).
- **PPO clip range:** 0.2, constant (`PPOConfig.clip_eps`).
- **Other counter-dependent behaviour:**
  - the lagged-opponent refresh every 20 *global* updates;
  - the weight export every 25 global updates;
  - the verifier cadence, which is phase-local (warm-up at local 100, stability checks every 20, timeout 100);
  - the Phase A stop-rule counters (consecutive eligible calls).
- **Nothing had to be held.** No schedule changes after update 400, so no flag was added.
- **What the continuation does differently from an uninterrupted 1600-update Phase A**, because it re-enters phase A:
  1. One snapshot refresh at re-entry (u400). The periodic refresh had already run at u400, so this copies the same actor.
  2. The phase-local verifier and stop-rule counters restart at u400.
- **C7** is still bit-exact at `cd760fd` (`results/v2_pilots/phase2_regression/v2_full_cd760fd.compare.txt`).

### Launch

- Run in tmux session `v2_phaseA_ext` alongside Pilot 3: 20 + 40 single-threaded processes.
- `nproc --all` = 64. Load average 0.97 / 0.81 / 2.43 at launch and 44.26 / 30.06 / 14.31 at the end (`PX/launch_20261001_233012.json`).
- 20/20 returncode 0.

### Metrics

- `utils/v2_metrics.evaluate` on the development tier, with the Beta mean, re-evaluated at every weight export from u425 to u1600, plus u400 from the parent's export.
- The smoothed-game share at d = 0 uses the Pilot-1 method (400 equal-probability nodes per Beta).

---

## 2. Results

### 2.1 Medians at u400 / u800 / u1200 / u1600

Each cell is median [IQR] (min–max) across 10 seeds. Source: `X/table_400_800_1200_1600.csv`; each underlying value is in `X/curves_weights_every25.csv`.

Units:
- tail and symmetry are in effort units;
- `*_over_g2_0` is divided by e₂*(0);
- Δ₂ and η₂ are /ΔW;
- σ is in effort units;
- `share_peak_gap_explained` = (e₂*(0) − ê_pred(0)) / (e₂*(0) − ê_learned(0)).

#### q = 50
| metric | 400 | 800 | 1200 | 1600 |
|---|---|---|---|---|
| DeltaT_over_dw_off_max | 0.00144 [0.001238, 0.001757] (0.0005562–0.002626) | 0.0006315 [0.0005293, 0.0007409] (0.0003459–0.001735) | 0.0004991 [0.0003865, 0.0006277] (0.0003265–0.001006) | 0.0003395 [0.000264, 0.000386] (0.0001463–0.001545) |
| DeltaT_over_dw_on_max | 0.004606 [0.004373, 0.005746] (0.001451–0.006638) | 0.002664 [0.001909, 0.003183] (0.001172–0.003667) | 0.002556 [0.001709, 0.002941] (0.0005958–0.003502) | 0.001469 [0.001119, 0.002431] (0.0007199–0.006971) |
| DeltaT_over_dw_on_mean_cellmass_weighted | 0.001086 [0.0007081, 0.001434] (0.0003797–0.002074) | 0.0004923 [0.0003321, 0.0006156] (0.0002042–0.0006879) | 0.0005036 [0.000286, 0.0007265] (0.0002222–0.0008146) | 0.0003412 [0.0002337, 0.0005086] (0.0001176–0.002879) |
| e_learned_0 | 60.95 [60.02, 61.22] (59.06–64.43) | 64.47 [63.44, 65.1] (62.19–66.59) | 65.8 [63.97, 66.29] (62.15–67.68) | 65.22 [65.04, 65.99] (61–67.38) |
| e_pred_0 | 66.89 [66.82, 66.99] (66.56–67.05) | 67.34 [67.16, 67.45] (66.97–67.51) | 67.65 [67.45, 67.71] (67.22–67.78) | 67.84 [67.65, 67.91] (67.41–67.95) |
| eta_T_over_dw | 0.004606 [0.004373, 0.005746] (0.001511–0.006638) | 0.002664 [0.001909, 0.003183] (0.001172–0.003667) | 0.002556 [0.001709, 0.002941] (0.0005958–0.003502) | 0.001469 [0.001119, 0.002431] (0.0007199–0.006971) |
| share_peak_gap_explained | 0.3306 [0.3154, 0.3559] (0.2944–0.5718) | 0.4808 [0.4202, 0.5757] (0.3296–0.8083) | 0.5824 [0.3847, 0.686] (0.3377–0.9572) | 0.5059 [0.4365, 0.5288] (0.2706–0.783) |
| sigma2_effort_mean_pos | 3.476 [3.348, 3.556] (3.21–3.748) | 3.007 [2.857, 3.174] (2.797–3.489) | 2.644 [2.609, 2.894] (2.588–3.164) | 2.469 [2.377, 2.646] (2.324–2.955) |
| sigma_effort_at_0_t2 | 3.935 [3.811, 4.031] (3.741–4.352) | 3.365 [3.229, 3.591] (3.16–3.836) | 2.981 [2.897, 3.224] (2.814–3.524) | 2.735 [2.65, 2.982] (2.598–3.28) |
| stage2_peak_rel_err_abs | 0.1293 [0.1254, 0.1426] (0.07954–0.1562) | 0.07907 [0.06996, 0.09373] (0.04872–0.1115) | 0.06002 [0.053, 0.0861] (0.03316–0.1121) | 0.06835 [0.05722, 0.0709] (0.03741–0.1286) |
| stage2_peak_rel_err_signed | -0.1293 [-0.1426, -0.1254] (-0.1562–-0.07954) | -0.07907 [-0.09373, -0.06996] (-0.1115–-0.04872) | -0.06002 [-0.0861, -0.053] (-0.1121–-0.03316) | -0.06835 [-0.0709, -0.05722] (-0.1286–-0.03741) |
| stage2_rmse_pos_over_g2_0 | 0.04367 [0.03551, 0.04881] (0.03362–0.05957) | 0.03083 [0.02859, 0.03521] (0.01949–0.04158) | 0.02896 [0.02332, 0.0309] (0.02118–0.0411) | 0.02605 [0.0221, 0.02871] (0.01422–0.05976) |
| stage2_sym_err_max | 2.981 [2.351, 3.878] (1.128–6.462) | 3.467 [2.267, 4.268] (1.452–4.752) | 2.855 [2.233, 4.053] (0.8764–7.866) | 2.638 [1.742, 3.529] (1.447–5.473) |
| stage2_tail_max | 4.76 [4.264, 5.108] (3.193–6.134) | 3.144 [2.792, 3.444] (2.304–5.734) | 2.813 [2.585, 3.003] (2.309–4.23) | 2.22 [2.131, 2.597] (1.656–4.907) |
| stage2_tail_max_over_g2_0 | 0.068 [0.06092, 0.07297] (0.04562–0.08763) | 0.04492 [0.03988, 0.0492] (0.03292–0.08192) | 0.04019 [0.03693, 0.04289] (0.03298–0.06042) | 0.03172 [0.03044, 0.03711] (0.02365–0.07009) |
| stage2_tail_mean | 1.477 [1.352, 1.693] (1.008–2.092) | 0.8706 [0.727, 0.8805] (0.5781–1.254) | 0.6386 [0.5335, 0.6965] (0.4294–0.9571) | 0.515 [0.4559, 0.5639] (0.3528–0.8831) |
| stage2_tail_mean_over_g2_0 | 0.0211 [0.01932, 0.02419] (0.01439–0.02989) | 0.01244 [0.01039, 0.01258] (0.008258–0.01791) | 0.009123 [0.007621, 0.009951] (0.006134–0.01367) | 0.007358 [0.006513, 0.008055] (0.00504–0.01262) |
#### q = 60
| metric | 400 | 800 | 1200 | 1600 |
|---|---|---|---|---|
| DeltaT_over_dw_off_max | 0.0008603 [0.0008057, 0.0009269] (0.0006519–0.00105) | 0.0004195 [0.0003855, 0.0005562] (0.000242–0.0008193) | 0.000313 [0.0002443, 0.0004059] (0.0002324–0.0004535) | 0.0002439 [0.0002221, 0.0002957] (0.0001492–0.000465) |
| DeltaT_over_dw_on_max | 0.001833 [0.001767, 0.002251] (0.001181–0.003512) | 0.001113 [0.0008293, 0.002318] (0.0007333–0.006252) | 0.001059 [0.0006352, 0.001684] (0.0003918–0.0022) | 0.0009012 [0.0007233, 0.0009584] (0.0003882–0.002156) |
| DeltaT_over_dw_on_mean_cellmass_weighted | 0.000397 [0.0003308, 0.0005366] (0.0002937–0.0005735) | 0.0002832 [0.0001797, 0.0004761] (0.0001116–0.001451) | 0.0002222 [0.0001444, 0.0003065] (0.0001153–0.0006444) | 0.0001711 [0.0001165, 0.0002749] (0.0001092–0.0004971) |
| e_learned_0 | 52.57 [52.12, 53.08] (50.45–54.32) | 54.97 [53.93, 56.06] (48.47–57.38) | 55.16 [53.53, 55.39] (51.9–56.06) | 54.83 [54.04, 55.98] (53.92–58.54) |
| e_pred_0 | 56.15 [56.02, 56.18] (56–56.24) | 56.41 [56.31, 56.48] (56.25–56.57) | 56.6 [56.52, 56.69] (56.46–56.75) | 56.75 [56.68, 56.84] (56.61–56.87) |
| eta_T_over_dw | 0.001833 [0.001767, 0.002251] (0.001181–0.003512) | 0.001113 [0.0008293, 0.002318] (0.0007333–0.006252) | 0.001059 [0.0006352, 0.001684] (0.0004335–0.0022) | 0.0009012 [0.0007233, 0.0009584] (0.0003882–0.002156) |
| share_peak_gap_explained | 0.3759 [0.3547, 0.4143] (0.2959–0.5817) | 0.5791 [0.4306, 0.834] (0.2118–1.843) | 0.508 [0.3888, 0.5591] (0.2713–0.8043) | 0.4003 [0.3573, 0.6235] (-7.224–0.9654) |
| sigma2_effort_mean_pos | 3.31 [3.259, 3.461] (3.147–3.549) | 2.908 [2.795, 3.058] (2.703–3.119) | 2.596 [2.51, 2.764] (2.391–2.799) | 2.393 [2.283, 2.543] (2.206–2.576) |
| sigma_effort_at_0_t2 | 3.975 [3.929, 4.213] (3.823–4.257) | 3.499 [3.378, 3.693] (3.219–3.807) | 3.166 [2.991, 3.311] (2.895–3.422) | 2.883 [2.731, 3.015] (2.671–3.142) |
| stage2_peak_rel_err_abs | 0.09872 [0.09011, 0.1065] (0.06882–0.1351) | 0.0577 [0.03899, 0.07548] (0.01642–0.169) | 0.05439 [0.05049, 0.08235] (0.03897–0.1103) | 0.06003 [0.04034, 0.07362] (0.003541–0.07572) |
| stage2_peak_rel_err_signed | -0.09872 [-0.1065, -0.09011] (-0.1351–-0.06882) | -0.0577 [-0.07548, -0.03899] (-0.169–-0.01642) | -0.05439 [-0.08235, -0.05049] (-0.1103–-0.03897) | -0.06003 [-0.07362, -0.04034] (-0.07572–0.003541) |
| stage2_rmse_pos_over_g2_0 | 0.03254 [0.03051, 0.03613] (0.02905–0.03852) | 0.0309 [0.02391, 0.03802] (0.01857–0.06408) | 0.0261 [0.02242, 0.03097] (0.01894–0.03818) | 0.02389 [0.02089, 0.03157] (0.01864–0.03376) |
| stage2_sym_err_max | 2.887 [2.016, 3.47] (0.9355–5.179) | 3.079 [2.576, 3.801] (0.6723–4.763) | 2.893 [1.979, 3.052] (0.7366–7.713) | 2.39 [1.517, 2.535] (0.8313–3.963) |
| stage2_tail_max | 3.568 [3.435, 3.64] (3.061–3.894) | 2.56 [2.369, 2.886] (2.107–3.407) | 2.346 [2.095, 2.531] (1.894–2.709) | 1.901 [1.835, 2.109] (1.708–2.556) |
| stage2_tail_max_over_g2_0 | 0.06116 [0.05889, 0.06241] (0.05247–0.06675) | 0.04388 [0.04061, 0.04948] (0.03612–0.05841) | 0.04022 [0.03592, 0.04339] (0.03246–0.04645) | 0.03259 [0.03146, 0.03616] (0.02927–0.04382) |
| stage2_tail_mean | 1.339 [1.041, 1.443] (0.9137–1.878) | 0.7954 [0.7138, 0.8359] (0.646–1.106) | 0.6048 [0.5467, 0.6556] (0.4867–0.7983) | 0.5312 [0.501, 0.5663] (0.4143–0.6374) |
| stage2_tail_mean_over_g2_0 | 0.02295 [0.01784, 0.02473] (0.01566–0.03219) | 0.01363 [0.01224, 0.01433] (0.01107–0.01896) | 0.01037 [0.009372, 0.01124] (0.008343–0.01369) | 0.009106 [0.008588, 0.009708] (0.007102–0.01093) |

### 2.2 Learning curves

Figures:
- `reports/v2/figures/phaseA_ext/curves_q50.png`
- `reports/v2/figures/phaseA_ext/curves_q60.png`

Each shows the median and IQR across 10 seeds, u400–u1600 in steps of 25 updates. Panels: signed peak error, RMSE/e₂*(0), tail mean, tail max, symmetry error, η₂, off-path Δ₂, σ₂(0).

### 2.3 Run records

Source: `X/runs.csv`. KL and clip are medians over all updates of a run, then the median over runs.

| q | n | commit | dirty | n_would_fire | wf_min | wf_median | wf_max | wall_min | wall_median | wall_max | kl_median | clip_median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10 | cd760fd | False | 10 | 700 | 700 | 1100 | 198.6 | 200.2 | 214.6 | 0.005666 | 0.06304 |
| 60 | 10 | cd760fd | False | 10 | 700 | 700 | 900 | 198 | 200.1 | 201.3 | 0.005634 | 0.06846 |

The existing Phase A stop rule would have fired in all 20 continuations, at global updates 700–1100. Its counters restarted at u400, so "would have fired" here means "300–700 updates into the continuation". It never fired in the original 400-update Phase A of Pilot 1.

---

## 3. Where each metric plateaus (descriptive)

Read from the medians in §2.1 and the curves in §2.2.

- **Peak error (signed):**
  - q=50: −0.129 → −0.079 → −0.060 → −0.068.
  - q=60: −0.099 → −0.058 → −0.054 → −0.060.
  - Most of the gain is made by about u800. From about u800 to u1600 the median fluctuates between about −0.04 and −0.09 with no further trend: a plateau at roughly −0.06.
- **RMSE / e₂*(0):** 0.044 → 0.026 (q=50) and 0.033 → 0.024 (q=60). Still drifting down slowly at u1600, within a noisy band.
- **Tail mean and tail max:**
  - tail mean 1.48 → 0.52 (q=50) and 1.34 → 0.53 (q=60);
  - tail max 4.76 → 2.22 and 3.57 → 1.90.
  - Decreasing throughout, decelerating, with no plateau by u1600.
- **Symmetry error:** no clear trend; the median stays between 2.4 and 3.5 effort units.
- **η₂ = on-path max Δ₂:** 0.0046 → 0.0015 (q=50) and 0.0018 → 0.0009 (q=60). Noisy, with a downward drift and no clear plateau.
- **Off-path Δ₂ and the on-path cell-mass mean:** decrease steadily through u1600.
- **σ₂(0) and mean σ₂:** decrease almost linearly through u1600, from about 3.9–4.0 to 2.7–2.9. No plateau. The concentration floor (c ≥ 100, σ ≤ about 5) is not the binding constraint here.
- **Share of the peak gap explained by action-noise smoothing:**
  - q=50: 0.33 → 0.48 → 0.58 → 0.51.
  - q=60: 0.38 → 0.58 → 0.51 → 0.40.
  - It rises from u400 to u800 as the learned peak gap shrinks, then fluctuates.
  - The ratio is ill-conditioned once ê_learned(0) approaches e₂*(0): one q=60 seed at u1600 has ê_learned(0) slightly above e₂*(0), signed peak error +0.0035, which gives a share of −7.2 (`X/table_400_800_1200_1600.csv`, min). The medians are robust to this; the min/max are not.

---

## 4. Anomalies and deviations

1. **The stop rule would always have fired, but on restarted counters.** It fired in 20/20 continuations (§2.3), but this cannot be compared directly with an uninterrupted 1600-update Phase A, because the counters restarted at u400.
2. **The smoothed-game share is unstable near the target** (§3, last item).
3. **Not committed.** The full-state checkpoints and weight exports are on disk under `PX/` and not committed (gitignored `.pt`, size). The lightweight records are committed.

## 5. Commands to reproduce

```bash
tmux new-session -d -s v2_phaseA_ext "cd /home/fjiang4/tournament_experiment/.claude/worktrees/tournament-v2-stagewise-pilots-f97609 && /home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/launch_pilot.py --pilot phaseA_ext --phase Acont --parent-pilot pilot1 --parent-arm expected --reward-mode expected --qs 50 60 --seeds 10501 10502 10503 10504 10505 10506 10507 10508 10509 10510 --arms expected_ext --workers 20"
```

```bash
OMP_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/phaseA_ext_analysis.py --workers 24
```
