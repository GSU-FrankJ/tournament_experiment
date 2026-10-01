# Phase 1 — Verifier and metrics upgrade

Date: 2026-10-01. Repository: `/home/fjiang4/tournament_experiment`, worktree `.claude/worktrees/tournament-v2-stagewise-pilots-f97609`. Branch: `v2-stagewise-pilots`.

| Item | Value |
|---|---|
| **Base commit** | `657f54a` |
| Phase 1 code commit | `b2bfec0` |
| Phase 1 data commit | `da3d5f8` |

Path conventions:
- `P1/` means `results/v2_pilots/phase1/`.
- `REG/` means `results/v2_pilots/phase1_regression/`.
- `RAW/` means `/home/fjiang4/tournament_experiment/experiments/` (read only).
- ΔW = 4 for every number below (`game.w_h − game.w_l`; Phase 0 §2).

Phase 1 adds evaluation code only. Every file that the training runner imports is unchanged (§2, B7).

---

## 1. What was done

| File | Change | One-line reason |
|---|---|---|
| `utils/v2_metrics.py` | new | Wraps the unchanged `verify`. Adds Ĝmax_full and its argmax, η₂, the invariant residuals, the candidate's forward PMF, the on-/off-path split, the closed-form recovery metrics, the σ(d) arrays, and the NPZ/CSV/plot writers. |
| `tools/v2/common.py` | new | As-run game for each q; the analytic and zero candidates; a checkpoint → (mean, α/β) loader; the three verifier tiers. |
| `tools/v2/calibrate.py` | new | Item 7 calibration, plus one storage example on a trained checkpoint. |
| `tools/v2/benchmark_consistency.py` | new | Item 6: Monte Carlo of the environment's terminal step vs the closed-form CDF. |
| `tools/v2/probe_invariants.py` | new | Invariant stress test on 900 synthetic candidates. |
| `tools/v2/invariants_on_checkpoints.py` | new | Invariants on the 80 existing final T=2 checkpoints. |
| `tools/v2/compare_runs.py` | new | Field-by-field identity check of two runner outputs (B7). |
| `tests/test_v2_verifier.py` | new | The pytest suite (§3). |
| `results/v2_pilots/phase1/**`, `results/v2_pilots/phase1_regression/**` | new data | Outputs cited below. `checkpoint.pt` files are on disk but are not committed, because `.pt` is gitignored. |

**Not modified:**
- `utils/dp_br_verifier.py`
- `run/run_final_dp_br*.py`
- `agents/ppo_curriculum.py`
- `envs/curriculum_env.py`
- `utils/theory_multistage.py`

**Tooling:**
- `pytest 9.1.1` (plus `iniconfig 2.3.0`, `pluggy 1.6.0`, `pygments 2.21.0`) was installed into `/home/fjiang4/tournament_experiment/.venv` with `uv pip install --python …/.venv/bin/python pytest`, as you approved. The venv has no pip (Phase 1 stop report).
- numpy 2.5.0, torch 2.5.1+cu121 and Python 3.12.3 are unchanged; I checked them after the install.

**Seeds:** a fresh scan confirms that 10501–10503 are disjoint from the Phase 0 inventory. The scan script is `seeds.py` in the session scratchpad; its output ranges are listed in Phase 0 §8.

**Compute (A8):**
- Every Phase 1 computation ran as a single process with `OMP_NUM_THREADS=1`. The two regression runs also set `MKL_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`, which the runner requires.
- `nproc` = 64. Load average was 0.38 at the start and 0.83 at the end (`uptime`).
- No parallel runs were needed: the longest single job was the benchmark Monte Carlo, at about 17 s.
- Per-phase wall-clock logging belongs to the v2 training path and will be added in Phase 2.

---

## 2. Method documentation (B1, B2, item 1)

### 2.1 What the aggregated arrays are

All are in `utils/dp_br_verifier.py`, `verify`, lines 292–537, unchanged. The backward pass runs t = T, …, 1 on the stage grid G_t of D_t.

- **Opponent.**
  - `e_opp = policy(t, −G)` (line 386): the opponent plays ê_t(−d), i.e. its own perspective of the gap.
  - With `opponent_policy=None`, which is how v2 always calls it, the opponent is the same Beta-mean mapping as the candidate.
  - The opponent never deviates.
- **V_t^BR(d) (`v_br`).**
  - It is the max over the candidate set {effort grid E; valid concave-parabola vertices of Q^BR on E; the candidate's own mean action} of −k e² + W^BR_{t+1}(d − ê_t(−d) + e) (lines 393–402).
  - W^BR_{t+1} is built from **v_br at t+1** (lines 387 and 436). The deviator therefore re-optimizes at t and at every later stage.
  - At t = T the continuation is the closed form w_l + ΔW·F_ξ(y) (lines 356–359).
- **V_t^ê(d) (`v_mean`).** It is −k ê_t(d)² + W^mean_{t+1}(d − ê_t(−d) + ê_t(d)), where W^mean is built from v_mean at t+1 (line 406).
- **Δ_t(d) (`delta`).** It is the max over {E; vertices of Q^mean; ê_t(d); a^BR_t(d)} of Q^mean − V^ê (lines 405–418).
- **Action search.**
  - E is uniform on [0, 100] with step 1 (development tier) or 0.5 (final tier).
  - Vertices are added only where the grid triple is strictly concave, and Q is recomputed at each vertex.
  - Ties go to the smaller effort (`_select`, 237–256).
- **Expectation over the shock difference.**
  - Gauss–Legendre on [−2q, 0] ∪ [0, 2q], weighted by f_ξ (`gl_shock_rule`, 213–230). The tier sets 16 or 32 nodes per half.
  - Non-terminal continuation: linear `np.interp` of the next-stage **value** array (line 374).
  - A landing point outside D_{t+1} raises `DomainError`; there is no extrapolation (370–373).
- **The new metric.** G_t(d) = `v_br − v_mean`, computed in `utils/v2_metrics.py:evaluate`.
  - Ĝmax_full = max_t max_d G_t(d), reported together with (t*, d*).
- **Evaluated policy (B2).**
  - Trained checkpoints: the Beta mean, `make_policy_fns` (`run/run_final_dp_br.py:125-142`).
  - Calibration: the analytic and zero candidates are deterministic functions.

### 2.2 D_t construction and closure

- D_1 = {0}. D_2 = [−B, B] with B = 100 + 2q (200 at q=50, 220 at q=60). The grid is a symmetric `linspace` that includes both endpoints (`stage_grid`, 186–204).
- **Closure.** From the root, any action pair gives d + e_i − e_j ∈ [−100, 100], and ξ ∈ [−2q, 2q], so every reachable stage-2 state lies in [−B, B] = D_2. The GL nodes are strictly inside (−2q, 2q), so every landing point lies strictly inside D_2.
- **Interpolation.** Only stage 1 interpolates (V_2 values). Stage 2 uses the closed form, so it interpolates nothing.
- **Extrapolation.** None. `DomainError` was raised **0 times** in 1,152 verifier evaluations: all of them have `valid=True`. The count is 900 in `P1/invariants_probe.csv`, 240 in `P1/invariants_checkpoints.csv` and 12 in `P1/calibration/calibration.csv`. `evaluate` would propagate a `DomainError`; none occurred.
- **d = 0 is an exact grid node** on every grid used: the development tier (101 / 111 points), the 2× tier (201 / 221), the final tier (201 / 221) and the 0.5-step recovery grid (801 / 881).
  - `utils/v2_metrics.py` asserts this on every call (`zero_index` in `evaluate`; `symmetric_grid` for the recovery grid).
  - It is also covered by the test `test_zero_is_exact_grid_node`.

### 2.3 dReach, R_t and EXP_root

- R_1 = {0}. R_2 = the grid nodes covered by [a^BR_1(0) − ê_1(0) − 2q, a^BR_1(0) − ê_1(0) + 2q] ∩ D_2 (`dp_br_verifier.py:438-469`).
- For T=2 this set is the (closed) support of the stage-2 state distribution on the root BR path, which is the definition the prompt requires. dReach is unchanged.

### 2.4 On-path / off-path definition (B4)

- **On-path set.** `candidate_pmf` (`utils/v2_metrics.py`) is the forward PMF of the **candidate's own** chain: both players follow ê, and the drift is ê_1(0) − ê_1(−0).
  - It uses exactly the verifier's kernel: GL nodes plus the linear mass split onto the two neighbouring nodes (the code mirrors `dp_br_verifier.py:470-484`).
  - The verifier's own `pmf` field is the **BR** chain. That is a different object and is not used for this split.
- **"Positive probability" = exact positivity: `pmf > 0.0`, with no threshold.**
  - A node is on-path iff the linear split gave it strictly positive mass.
  - Because the GL nodes are strictly inside (−2q, 2q), the nodes nearest ±2q can receive zero mass. Examples:
    - analytic candidate at q=50, development tier: 47 on-path and 54 off-path nodes;
    - the same at the final tier: 95 on and 106 off (`P1/calibration/calibration.csv`, columns `DeltaT_over_dw_n_on/_n_off`).
  - Smallest positive on-path mass in the trained-checkpoint example: 3.27e−4 (`P1/calibration/example_checkpoint/metrics.csv`, `cand_pmf_min_positive_T`).
- **Reported for any per-state quantity** (`onoff_split`):
  - on-path max with its argmax d;
  - on-path unweighted mean;
  - on-path PMF-weighted mean (Σ p·v / Σ p);
  - off-path max with its argmax d;
  - off-path unweighted mean;
  - node counts.

  Off-path has zero PMF weight by definition, so there is no weighted off-path value.
- **Applied now** to Δ_2 (columns `DeltaT_over_dw_*`).
- **Phase 4** will apply the same function to the stage-2 drift |ê₂^child − ê₂^parent|. The function is generic, and `test_onoff_split` covers it.

---

## 3. Tests run and outcomes

Command:

```bash
OMP_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_v2_verifier.py -q -p no:cacheprovider
```

Result: **21 passed in 1.76 s.**

| Test | What it checks |
|---|---|
| `test_zero_is_exact_grid_node` ×2 q | 0.0 is an exact node of every tier and of the recovery grid |
| `test_invariants` ×2 q ×3 tiers | All invariants (§4.1) on 6 candidates: analytic, zero, constant 30, analytic stage 2 with low e₁, and two sinusoidal perturbations. Tolerance 1e−12·ΔW. |
| `test_analytic_floor_and_zero_policy_discriminates` ×2 | Ĝmax_full/ΔW is < 1e−5 for the analytic candidate and > 0.1 for zero effort |
| `test_candidate_pmf_mass_and_moments` ×2 | Candidate PMF has mass 1 and mean 0. Its variance lies in [2q²/3, 2q²/3 + h²/4]: the triangular variance plus at most the linear-split variance. The support is within (−2q − h, 2q + h). |
| `test_onoff_split` | Hand-computed example |
| `test_recovery_analytic_and_zero` ×2 | All recovery errors are exactly 0 for the analytic candidate. The zero policy gives −1 relative errors and the analytic RMSE. |
| `test_recovery_agrees_with_existing_helper` ×2 | The new recovery numbers equal the existing `run_final_dp_br.recovery_metrics` (RMSE, tail mean and max, symmetry, stage-1 error) to 1e−12 |
| `test_evaluate_consumes_no_rng` | numpy `Generator` states, the numpy global state and the torch RNG state are unchanged by `evaluate` |
| `test_terminal_win_probability_matches_cdf` ×2 | Small Monte Carlo spot check, \|z\| ≤ 5 |
| `test_storage_roundtrip` | NPZ arrays round-trip; a CSV header change raises an error |

**Runtime checks (item 5).** `evaluate` computes every invariant residual on **every call** and stores it in the CSV row as `inv_*` columns. The residuals are raw and never clamped; a positive excess is a violation. Nothing is gated on them.

---

## 4. Results

### 4.1 Invariant residuals (B3), in units of ΔW

**Definitions.**
- For an inequality lhs ≤ rhs, the "excess" is (lhs − rhs)/ΔW. Positive means violated.
- For an equality, the reported value is max |lhs − rhs|/ΔW.
- Δ ≤ G is reported both pointwise and in the requested aggregate form.
- I also report EXP_root ≤ dReach computed over the BR-PMF support, as a companion check.

| Relation | Max over 900 synthetic evals (`P1/invariants_probe.csv`) | Location of max | Max over 240 trained-checkpoint evals (`P1/invariants_checkpoints.csv`) | Analytic / zero calibration (`P1/calibration/calibration.csv`) |
|---|---|---|---|---|
| G₂ = Δ₂ (whole grid) | 0 | — | 0 | 0 |
| Δ_t(d) ≤ G_t(d), pointwise | **+5.55e−16** (38 of 900 rows > 0) | t=1, d=0 (q=60, candidate 87, final tier, stage 2 exactly closed-form) | 0 (no row > 0) | 0 |
| Δmax_all ≤ Ĝmax_full | **+5.55e−16** (38 rows > 0) | same as above | 0 | 0 |
| Ĝmax_full ≤ dFull | 0 | — | −4.41e−08 (max) | ≤ 0 |
| EXP_root = G₁(0) | 0 | — | 0 | 0 |
| EXP_root ≤ dReach | 0 | — | −4.27e−03 (max) | ≤ −2.2e−16 |
| dReach ≤ dFull | 0 | — | 0 | 0 |
| (companion) EXP_root ≤ dReach over the BR-PMF support | 0 | — | −4.27e−03 | ≤ −2.2e−16 |

How the Δ ≤ G residual changes with grid refinement (pointwise maximum per tier over the 300 probe rows each; `P1/invariants_probe.csv` grouped by `verifier_tier`):

| Tier | Max excess / ΔW | Rows > 0 |
|---|---|---|
| development (state 4, effort 1) | 3.33e−16 | 9 |
| dev_2x (state 2, effort 0.5) | 2.22e−16 | 12 |
| final (state 2, effort 0.5, GL 32) | 5.55e−16 | 17 |

- Every positive excess occurs at (t=1, d=0), and only for candidates whose stage 2 is **exactly** the closed form (`amp = 0`).
- The magnitude is 1–3 ulp of values of order ΔW, and it does not shrink on the 2× finer action grid.
- The 240 trained checkpoints are the 80 final checkpoints of `RAW/two_stage_{confirmation_T2_20260922,E1_q50_p_20260923,q50_restarts_20260924,q50_precision_20260924}/runs/*/*/checkpoint.pt`, each evaluated on 3 tiers. All are `valid=True`.

### 4.2 Verifier calibration (item 7)

Source: `P1/calibration/calibration.csv`. Arrays are in `P1/calibration/npz/q{q}_{policy}_{tier}.npz`. All rows are valid. Values are ×ΔW⁻¹.

| q | Candidate | Tier | Ĝmax_full | (t*, d*) | η₂ | EXP_root | dReach | Δmax_all | dFull |
|---|---|---|---|---|---|---|---|---|---|
| 50 | analytic | development | 2.2e−16 | (2, −16) | 2.2e−16 | 0 | 2.2e−16 | 2.2e−16 | 2.2e−16 |
| 50 | analytic | dev_2x | 2.2e−16 | (2, −24) | 2.2e−16 | 0 | 2.2e−16 | 2.2e−16 | 2.2e−16 |
| 50 | analytic | final | 2.2e−16 | (2, −24) | 2.2e−16 | 0 | 2.2e−16 | 2.2e−16 | 2.2e−16 |
| 60 | analytic | development | 2.2e−16 | (2, 40) | 2.2e−16 | 0 | 2.2e−16 | 2.2e−16 | 2.2e−16 |
| 60 | analytic | dev_2x | **2.640e−07** | (1, 0) | 2.2e−16 | 2.640e−07 | 2.640e−07 | 2.640e−07 | 2.640e−07 |
| 60 | analytic | final | **3.511e−07** | (1, 0) | 2.2e−16 | 3.511e−07 | 3.511e−07 | 3.511e−07 | 3.511e−07 |
| 50 | ê ≡ 0 | development | 0.25896 | (2, −24) | 0.25896 | 0.24275 | 0.39340 | 0.25896 | 0.39340 |
| 50 | ê ≡ 0 | dev_2x | 0.25926 | (2, −26) | 0.25926 | 0.24281 | 0.39376 | 0.25926 | 0.39376 |
| 50 | ê ≡ 0 | final | 0.25926 | (2, −26) | 0.25926 | 0.24281 | 0.39376 | 0.25926 | 0.39376 |
| 60 | ê ≡ 0 | development | 0.19551 | (2, −24) | 0.19551 | 0.19102 | 0.29506 | 0.19551 | 0.29506 |
| 60 | ê ≡ 0 | dev_2x | 0.19551 | (2, −24) | 0.19551 | 0.19106 | 0.29510 | 0.19551 | 0.29510 |
| 60 | ê ≡ 0 | final | 0.19551 | (2, −24) | 0.19551 | 0.19106 | 0.29509 | 0.19551 | 0.29509 |

**Numerical floor.** On the analytic equilibrium the floor is ≤ 3.6e−7·ΔW in every metric. It is non-zero only at q=60, at (t=1, d=0), on the two finer grids.

**Development → 2× tier differences:**
- Analytic: 0 at q=50; +2.64e−7 in Ĝmax_full, EXP_root and dReach at q=60.
- Zero effort, q=50: Ĝmax_full +2.94e−4, dReach +3.58e−4.
- Zero effort, q=60: Ĝmax_full 0, dReach +3.88e−5.

The argmax location of Ĝmax_full on the analytic candidate is not meaningful: every grid value is ≤ 2.2e−16·ΔW.

**PDL residual.** Max |`pdl_residual_over_dw`| over the 12 calibration rows is 4.7e−16.

### 4.3 Recovery metrics against the closed form (item 2)

Recovery is evaluated on the existing 0.5-step recovery grid (`recovery_step`), which has 0 as an exact node.
- **Analytic candidate:** every error is exactly 0 (test `test_recovery_analytic_and_zero`).
- **Storage example** on an existing trained checkpoint, q=50, `tel_q50_s10231` (source `P1/calibration/example_checkpoint/metrics.csv`, which evaluates `RAW/two_stage_q50_precision_20260924/runs/FINAL_A400_B25_C25/tel_q50_s10231/checkpoint.pt`). Values below are stated as numbers, not percentages:

| Metric | Value |
|---|---|
| Stage-1 relative error (signed) | +0.020911 (ê₁(0) 47.6425 vs e₁* 46.6667) |
| Stage-2 peak relative error at d=0 (signed) | −0.187463 (ê₂(0) 56.8776 vs e₂*(0) 70) |
| Stage-2 RMSE over \|d\| < 2q, raw / ÷ e₂*(0) | 5.1718 / 0.073883 (399 nodes) |
| Stage-2 tail mean / max effort (\|d\| ≥ 2q, 402 nodes) | 8.7910 / 12.2830 (max at d = 100) |
| Symmetry max\|ê₂(d) − ê₂(−d)\|, raw / ÷ e₂*(0) | 4.4264 / 0.063234 (at \|d\| = 139.5) |
| σ₂(0), σ₁(0) (effort units) | 3.4560, 3.7598 |
| σ₂ mean over \|d\| < 2q | 3.0602 |
| Ĝmax_full / ΔW (t*, d*) | 0.009852 (2, 108) |
| EXP_root / dReach / Δmax_all / dFull, all / ΔW | 0.002392 / 0.009684 / 0.009852 / 0.009852 |
| Δ₂ / ΔW, on-path max (d) / PMF-weighted mean | 0.009684 (−8) / 0.002392 |
| Δ₂ / ΔW, off-path max (d) / unweighted mean | 0.009852 (108) / 0.005450 |

The plot is `P1/calibration/example_checkpoint/stage2_final.png`. This checkpoint serves only as a format example; it is not a pilot result.

### 4.4 Storage format (items 3–4)

- **Per checkpoint:** one NPZ, `save_npz`, with 50 arrays in the example. Per stage it holds:
  - `v_t{t}_{d_grid, e_hat, e_opp, v_br, a_br, a_br_source, v_mean, delta, a_dev, a_dev_source, q_mean_at_abr, reach, pmf, q_br_grid, q_mean_grid, std_norm, alpha, beta}` (the existing `stage_result_arrays`);
  - plus `v_t{t}_G`, `v_t{t}_cand_pmf`, `v_t{t}_onpath`, `v_t{t}_sigma_effort` (σ in [0, 100] effort units);
  - plus `recovery_{d_grid, e2, g2}` and `v_{e_grid, gl_nodes, gl_weights}`.
- **Per run:** one CSV row per checkpoint, `append_csv`; the header is fixed, and a mismatch raises an error. The example has 69 columns.
- **Per run:** the final-checkpoint plot, `plot_stage2`, overlays ê₂(d) and e₂*(d) and shows σ₂(d) in a second panel.
- In Phase 2 these writers will be called at the existing verifier cadence. The arrays are on the development-tier grid that the training-time verifier uses.

### 4.5 Benchmark consistency (item 6)

Source: `P1/benchmark_consistency.csv` and `P1/benchmark_recheck.csv`.

- **Grid:** d ∈ 21 points on [−B, B] × e_i, e_j ∈ {0, 20, …, 100}, giving 756 cells per q. Each cell uses N = 200,000 draws, for both players.
- **Code path:** the environment's own `step_gap` and `GameSpec.terminal_reward`, with shocks U(−q, q) per player.

| q | Interior cells (0 < F < 1) | z-scores of interior cells: mean / sd | max \|z\| (cell) | max \|p̂ − F\| | Cells with F ∈ {0, 1}: max \|p̂ − F\| |
|---|---|---|---|---|---|
| 50 | 324 | 0.043 / 1.014 | 4.541 (d=60, e_i=60, e_j=60, player j) | 0.00314 | 0 (exact) |
| 60 | 394 | −0.036 / 1.024 | 3.830 (d=22, e_i=0, e_j=80, player i) | 0.00291 | 0 (exact) |

- **The two players are not independent tests.** Player j's z equals −z_i, because player j's outcome is the complement of player i's on the same draws.
- **Recheck of each max-\|z\| cell** with 40 fresh streams of 200k draws (`P1/benchmark_recheck.csv`):
  - q=50: z mean −0.007, sd 1.132, max \|z\| 2.38. The mean is −0.04 standard errors from 0.
  - q=60: z mean −0.234, sd 0.974, max \|z\| 2.12. The mean is −1.52 standard errors from 0.
- **Expected terminal reward** (r̄ = w_l + ΔW·F_ξ − k e², the Phase 2 `expected` formula):
  - Interior cells: max \|z\| 4.615 at q=50, the same cell as above, and 3.796 at q=60.
  - Cells where the reward is constant: max \|r̂ − r̄\| = 8.9e−16. The CSV keeps the raw `r_diff` and `r_const` for these and leaves their z as NaN.

---

## 5. B7 — No side effects on training

**Code argument.**
- Phase 1 added files only (`git show --stat b2bfec0`). No module imported by `run/run_final_dp_br_round3_dense.py` changed.
- Neither `utils/dp_br_verifier.py` nor `utils/v2_metrics.py` contains any RNG call (`grep -nE "random|default_rng|Generator|rng"` over both files: no hits).
- `test_evaluate_consumes_no_rng` confirms the RNG states are unchanged after a call.

**Before/after run.**
- Configuration: the existing runner on q=50, seed 10501, manifest `REG/manifest.json`. It is a copy of the precision record with only `seed`, `run` and `output_dir` changed.
- Smoke budget: caps A/B/C 40/40/40, warm-up 10, stability every 5, timeout 10, direct rollout 2000 × 1. That is 120 updates and 12 verifier calls.
- The "before" run was executed at `1ad3805`, before any Phase 1 file existed. The "after" run was executed with the Phase 1 code present.

`tools/v2/compare_runs.py` output (`REG/compare.txt`):

| Compared | Result |
|---|---|
| `train_history.json` (all leaves except `time_sec`) | 0 differing leaves |
| `final_eval.json` (all leaves except timing and path keys, listed in `REG/compare.txt`) | 0 differing leaves |
| `arrays.npz` (90 arrays), `checkpoint_weights.npz` (12), `phase_A_exit_arrays.npz` (41), `phase_B_exit_arrays.npz` (41) | 0 differ |
| `checkpoint.pt` actor/critic/opponent weights and both Adam states | 0 differing tensors |
| stdout logs (`REG/before.log`, `REG/after.log`), excluding the `[run]` out-dir line and the wall-time suffix | identical |

---

## 6. Anomalies, deviations and open questions

1. **Phase 0 Q6 did not materialize.** I had predicted a vertex-driven violation of Δmax_all ≤ Ĝmax_full. Across 1,152 evaluations the largest excess is 5.55e−16·ΔW, at (t=1, d=0). It appears only when stage 2 is exactly the closed form, it is rounding-level, and it does not shrink with the 2× finer action grid (§4.1).
2. **One 4.5σ Monte Carlo cell** (q=50; §4.5). The recheck with 40 fresh replicates showed no bias. I treat it as a sampling tail event and have not altered anything because of it.
3. **The calibration floor is non-zero only at q=60** on the finer tiers: 3.5e−7·ΔW at the root. The cause is not established. A candidate explanation is the stage-1 linear interpolation of V₂ combined with e₁* = 38.889 lying off the effort grid; I did not test it.
4. **Deviation: the "2× finer grid" calibration** is reported twice:
   - as `dev_2x`: state 2, effort 0.5, GL count unchanged at 16, which isolates state and action;
   - and as the existing final tier: also GL 32.
5. **The on-path set is determined by exact zeros** of a GL-node PMF. With 16 nodes per half on a step-4 grid, the outermost nodes near ±2q get exactly zero mass (§2.4). If you want a different numerical rule, for example a minimum-mass threshold, it has to be decided before Pilot 2.
6. **Benchmark clip.** As noted in Phase 0, the repo's g₂ clips to [0, 100]. The clip is inactive at q ∈ {50, 60}.

---

## 7. Commands to reproduce

All commands run from the worktree root. Each tool refuses to overwrite an existing output.

```bash
OMP_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_v2_verifier.py -q -p no:cacheprovider
```

```bash
cd tools/v2 && OMP_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python calibrate.py --out ../../results/v2_pilots/phase1/calibration --example-ckpt /home/fjiang4/tournament_experiment/experiments/two_stage_q50_precision_20260924/runs/FINAL_A400_B25_C25/tel_q50_s10231/checkpoint.pt
```

```bash
cd tools/v2 && OMP_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python benchmark_consistency.py --out ../../results/v2_pilots/phase1/benchmark_consistency.csv
```

```bash
cd tools/v2 && OMP_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python probe_invariants.py --out ../../results/v2_pilots/phase1/invariants_probe.csv
```

```bash
cd tools/v2 && OMP_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python invariants_on_checkpoints.py --raw /home/fjiang4/tournament_experiment/experiments --out ../../results/v2_pilots/phase1/invariants_checkpoints.csv
```

B7 runs: execute this once with `--smoke-root results/v2_pilots/phase1_regression/before` and once with `.../after`, then compare.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python -B run/run_final_dp_br_round3_dense.py --manifest results/v2_pilots/phase1_regression/manifest.json --group FINAL_A400_B25_C25 --q 50 --seed 10501 --smoke --smoke-phase-caps 40,40,40 --smoke-warmup 10 --smoke-stability-every 5 --smoke-timeout 10 --smoke-direct-rollout-episodes 2000 --smoke-direct-rollout-reps 1 --smoke-root results/v2_pilots/phase1_regression/after
```

```bash
/home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/compare_runs.py results/v2_pilots/phase1_regression/before/FINAL_A400_B25_C25/tel_q50_s10501 results/v2_pilots/phase1_regression/after/FINAL_A400_B25_C25/tel_q50_s10501
```

---

**STOP — Phase 1 complete. Waiting for approval before Phase 2.**
