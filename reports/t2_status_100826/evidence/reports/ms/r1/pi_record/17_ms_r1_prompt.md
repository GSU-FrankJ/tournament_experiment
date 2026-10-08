# Multistage round MS-R1: development stop rule with a first-order residual, coverage-constrained verifier-guided exploring starts, targeted polishing — T-generic code, T=2 pilot on the development seeds

The T=2 refinement work is closed (`reports/t2_refine_100526/100526report.md`; the PI's decisions in its section 1). The baseline is settled: **conditional expected reward + expected continuation + backward freeze**, i.e. protocol v2.0, which stays the locked T=2 solver and is not touched by this round. What v2.0 leaves open is in the report's sections 6 and 8: the stage-2 peak is below the closed form in every run (signed peak error negative in 20 of 20 development runs, medians −0.0625 / −0.0497 at q = 50 / 60; 20 of 20 at both q in the fresh-seed confirmation), the fixed budgets 1600 / 600 are a stand-in for a stop rule, and the earlier stages have no accuracy metric that does not use the closed form.

This round restores the original design — a **development DP-BR stop rule** (threshold or budget) with **stopping and targeted polishing** — and adds **verifier-guided, coverage-constrained stratified exploring starts**, applied **per stage**. Three facts from the repository shape the design and must be kept in mind (verify them yourself in P1c; the paths are given):

1. The original rule is still in `run/run_v2_stagewise.py` (`phase_thr_over_dw`, `k_phase`, `conc_thr`); v2.0 only disables it with `fixed_budget = true`. In v2.0 the development verifier runs every 100 updates in Phase A (`verifier_timeout`), and one development-tier call costs about 10 ms at T=2 (`verifier_sec` in `v2_checkpoints_A.csv`).
2. Replayed on the 60 v2.0 runs' Phase-A verifier logs (`v2_checkpoints_A.csv` of `rehearsal_v2_0` and `confirmation_v2_0`, calls every 100 updates), the original rule (3 consecutive eligible calls, threshold 0.02 or 0.005 on `max_D2dev_Delta2_over_dw`, concentration ≤ 0.04) fires at a median local update of 600–750, where the median |peak error| is 0.070–0.078 against 0.056–0.067 at update 1600, and the tail mean at the firing call reaches 0.0228–0.0235 (above the G-A limit 0.02) in some runs. **Restoring the second-order rule alone would freeze earlier and worse than the fixed budget.**
3. The one-step deviation gain is second order in the effort error. At the end of Phase A of `confirmation_v2_0` q = 50 seed 30501 (`gates.json`: ê₂(0) = 66.45 against e₂*(0) = 70, −5.1 %), the one-step deviation gain at d = 0 is 5.3e-4 ΔW, a tenth of the G-A limit, while the one-step best-response effort against the policy itself is 68.54, i.e. **2.09 effort units (3.0 % of e₂*(0)) above the policy — a first-order, detectable quantity** (computed with the repository's `F_xi`, k = 1/3500, ΔW = 4). The verifier already computes it per state (`StageResult.a_dev` against `StageResult.e_hat`). The D2 diagnostic says the same thing from the other side (a stage-2 amplitude error is detected by G-A only from 10 %).

Hence the stop rule of this round has a first-order part, the sampler is driven by the same first-order residual, and nothing in the rule or the sampler uses the closed form.

Steps:

1. **P0** housekeeping (§1).
2. **P1** the T-generic runner, the sampler, the rule, tests, the two reproduction checks, the offline calibration on the v2.0 weight exports, the pre-registration — then **STOP at gate G1** and wait for my "proceed P2" (§2).
3. **P2** the pilot wave on the development seeds and its pre-registered analysis (§3).
4. **P3** reports and the push, then **STOP** (§4, §5).

Repository `/home/fjiang4/tournament_experiment`; Python `/home/fjiang4/tournament_experiment/.venv/bin/python` only; `pytest tests` (never a bare `pytest`); every run single-threaded (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`) inside tmux; do not kill jobs you did not start. P1–P3 run on a new branch `ms-r1` taken from `origin/main` (the head of the `t2-refine-100526` publication), in a new worktree `.claude/worktrees/ms-r1`. Results root `results/ms_r1/`; reports `reports/ms/r1/`; new tools under `tools/ms/`; new tests `tests/test_ms_*.py`.

Everything from the earlier rounds still applies: provenance discipline (every number in a report cites a path); the closed-form equilibrium enters evaluation only, never training, never the rule, never the sampler; no deletion or overwriting of results or checkpoints; the paper-registry test and its data unchanged; no `.pt`, `train_history.json` or weight exports in git; `clamp_likelihood` stays `density`. Nothing in `protocols/`, `run/run_v2_T2_locked.py`, `utils/v2_continuation.py` or `tools/v2/confirmation_analysis*.py` changes. Pushing is authorised only where §2.7 and §4.3 say so. Save this prompt verbatim as `reports/ms/r1/pi_record/17_ms_r1_prompt.md` before anything else.

**If any step fails, or anything does not match this prompt, stop at that point and report. Do not work around it.**

---

## 0. Decisions (binding)

### D1. Scope

- T=2 only in the experiments: development seeds 10501–10510 × q ∈ {50, 60}, 20 runs per arm, paired with the v2.0 runs. No fresh seeds (40501–40520 stay reserved), no lock, no re-rehearsal, no confirmation, no T=3 experiment. The round ends with a pilot report and my decision.
- The code is **T-generic**: stage indices, domains, bins, the nested continuation tables and the rule work for `GameSpec.T ∈ {2, 3}`, and the test suite exercises T=3 on reduced budgets (D8). Protocol v2.0 stays the locked T=2 solver.

### D2. Pipeline structure: one stage per phase, exact expected continuation, new runner

- New entry point `run/run_ms_stagewise.py` and new module `utils/ms_continuation.py`. Phases run t = T, T−1, …, 1. In the phase of stage t the learner rolls out **stage t only**: exploring starts on D_t for t ≥ 2 (the root, d = 0, for t = 1), one learner action from the live actor, the opponent's stage-t action sampled from the lagged copy (snapshot refresh every 20 global updates and at phase entry, as now), the stage return
  `R_t = −k e_t² + Ṽ_{t+1}(d_t + e_t − e_t^opp)` (for t = T: the conditional expected terminal reward `w_L + ΔW F_ξ(d + e − e^opp) − k e²`, `reward_mode = expected`, exactly `run.v2_rollout.expected_terminal_reward`),
  advantage = return − V(s_t), critic target = the return; all rows of the batch are stage-t rows (advantage normalisation over all rows, which is v2.0's `stage1_rows` scope for t = 1). Later stages are frozen snapshots (deep copies taken at their freeze; no gradient, no optimiser); earlier stages are untrained until their phase.
- `Ṽ_{t+1}` is the shock-integrated table of the frozen suffix, nested: `Ṽ_{t+1}(y) = E_z[g_{t+1}(y + z)]`, `g_T(d) = w_L + ΔW F_ξ(d + ê_T(d) − ê_T(−d)) − k ê_T(d)²`, and for t+1 < T `g_{t+1}(d) = −k ê_{t+1}(d)² + Ṽ_{t+2}(d + ê_{t+1}(d) − ê_{t+1}(−d))`, with ê the frozen Beta means on the `mean` float path, z = ε_L − ε_O with density (2q − |z|)/(4q²). Integration rule = the locked v2.0 rule (composite Gauss–Legendre, 6 nodes per panel, panels of width 1 aligned at z = 0 and ±2q, y-grid step 0.05, float64, linear lookup); the y-grid of `Ṽ_{t+1}` covers every reachable `d_t + e_t − e_t^opp`, i.e. `[−(domain_half(t) + e_range), domain_half(t) + e_range]`, and the nodes `y + z` must lie in D_{t+1} (assert). At T=2 the new module's table must equal `utils.v2_continuation.build_continuation_table` on the same frozen actor to ≤ 1e-12 ΔW (test); the tables are built once at phase entry, single-threaded, without any RNG draw (the same stream comparison as v2.0's `continuation_table.rng`).
- Everything else is v2.0's: the game, the 2→64→64 tanh actor and critic, PPO settings (LR 3e-4 base, 10 epochs, minibatch 256, clip 0.2, grad clip 0.5, entropy 0, γ = λ = 1, Adam state preserved across phases), 512 episodes per update, the five RNG streams and the torch generator from `SeedSequence([seed, q, namespace])`, the process-global RNG hardening of the locked entry point (seeding, reference state, assertions, digests; a violation is recorded and the run counts as failed), ES bin width 10, `clamp_likelihood = density`, weight exports every 25 updates, the verifier tiers and every metric definition of `utils.v2_metrics`, concentration threshold 0.04.
- **Legacy settings must reproduce v2.0's Phase A bit for bit.** With `start_weights = {scheme: bin_balanced}`, `rule.enabled = false`, the fixed budget 1600 and v2.0's LR window (constant 3e-4 to local 1200, linear 3e-4 → 3e-5 over 1201–1600), the terminal-stage phase of the new runner must consume the streams exactly as `run/run_v2_stagewise.py` mode `phase_A` does and end in the training-relevant state of `parents_A` in 20 of 20 development runs (check C-MS1, §2.3). If the structure of D2 makes this impossible, stop and report; do not weaken the check. The stage-1 phase is **not** bit-identical to v2.0's Phase B (v2.0 still rolls out and critic-fits the stage-2 rows); its stage-1 return is the same table value, and its end-of-phase metrics are compared with `rehearsal_v2_0` descriptively.
- Shared modules (`envs/curriculum_env.py`, `run/v2_rollout.py`, `agents/*`, `utils/v2_metrics.py`, `utils/dp_br_verifier.py`) may gain code only behind new keys or new functions whose absence leaves every existing path byte-identical; C-R4 (§2.3) is the proof.

### D3. The two residual metrics (development tier, from the DP-BR verifier; no closed form)

At every development check of the phase of stage t, on the composite candidate (live actor at t, frozen snapshots above t, the live actor's untrained output below t — as v2.0's `policy_fns`), `utils.dp_br_verifier.verify` at `DEV_CONFIG` gives, per node d of the stage-t grid, the one-step deviation gain δ_t(d) (`StageResult.delta`), the policy mean ê_t(d) (`e_hat`) and the one-step best-response effort ẽ_t(d) = `a_dev` (argmax of the one-step deviation search, policy action and parabola-vertex candidates included). Define, all on the development tier:

- **Second-order:** `Δ_t = full_delta_max[t] / ΔW` (for t = T this is `eta_T_over_dw` on the development tier; for t = 1 the gain at the root).
- **First-order residual map:** `r_t(d) = |ê_t(d) − ẽ_t(d)|`, scale `s_t = ẽ_t(0)` (the best-response effort at d = 0; at t = 1 the only node). `R_t = max{ r_t(d) / s_t : |d| < 2q (T − t + 1) }` (the non-tail region; at t = 1 just `r_1(0)/s_1`, the first-order analogue of G-S). **Tail:** `R_t^tail = mean{ ê_t(d) : |d| ≥ 2q (T − t + 1) } / s_t` (the first-order analogue of G-A's tail mean; ẽ_t is 0 there under symmetric play, so this is the residual).
- **Per-bin map** for the sampler: for every ES bin b of D_t that is not a tail bin, `ρ_b = max{ r_t(d)/s_t : d ∈ b }` over the development nodes assigned to b by `envs.curriculum_env.gap_bin_index`; EMA across checks `ρ̄_b ← β ρ̄_b + (1 − β) ρ_b`, β = 0.5, initialised at the first check.
- **Concentration:** `C_t` = `concentration_stats(...)["max_std_norm"]` on the stage-t development grid, as now.

The closed-form quantities (peak error, RMSE_pos, tail mean over e₂*(0), the stage-1 error) are computed at the same checks **for reporting only** (`utils.v2_metrics.recovery_metrics`), and never read by the rule or the sampler. Tests must show that the rule and the sampler produce identical decisions when the closed-form functions are replaced by stubs.

### D4. The development stop rule, blocks, landing, freeze (per stage)

Parameters (defaults; D8 says which the calibration may move): cadence K = 25 local updates; M = 3 consecutive eligible checks; ε_t = 0.005 (both stages); ρ_t = 0.03; τ_t = 0.02; concentration limit 0.04; block length N_block = 400 (terminal stage) / 200 (stage 1); training cap U_cap = 2000 / 600; landing length N_land = 400 / 400; localized fraction 0.25.

- **Eligible check:** valid verifier ∧ `Δ_t ≤ ε_t` ∧ `R_t ≤ ρ_t` ∧ `R_t^tail ≤ τ_t` ∧ `C_t ≤ 0.04` (the tail term is void at t = 1 and in a stage without a tail region). All comparisons inclusive, float64.
- **Blocks.** Training runs in blocks of N_block updates at constant LR 3e-4; the development check runs every K updates and at every block end. **Development stop** = M consecutive eligible checks: the block ends at that check and the landing window starts. If a block ends without a development stop, classify the last check's residual: `S_t = { non-tail bins b : ρ̄_b > ρ_t }`; **localized** iff 1 ≤ |S_t| ≤ ⌈0.25 · n_nontail⌉ (q = 50: 5 of 20; q = 60: 6 of 24) **and** the first-order part is what fails (R_t > ρ_t); otherwise **broad** (including the cases where only Δ_t, C_t or the tail fail). Localized → the next block is a **targeted polishing block** (sampler per D5 with α = α_polish on S_t); broad → the next block is a **global block** (sampler per D5 with the arm's α_global). This is the PI's rule: Δ ≤ ε for M checks → freeze; residual broad → continue global training; residual localized → targeted polishing. Blocks continue until U_cap; reaching U_cap without a development stop starts the landing window with `budget_forced = true`.
- **Landing window:** N_land updates with the LR linear from 3e-4 to 3e-5 (the existing `lr_at` linear form, `Run.lr_for` semantics), the sampler settings (block type, α, focus) of the block it follows, development checks continuing at cadence K and reported but deciding nothing.
- **Freeze:** at the end of the landing window the candidate is evaluated on the **final tier** (and the development tier for G-N), the gates of D7 are written, the stage-t snapshot is frozen (deep copy), `Ṽ_t` is built, and the next phase starts. The stage is never frozen before its landing window; it is never trained after its freeze.
- For t = 1 there is no sampler and no polishing (D_1 = {0}); the rule degenerates to development stop / continue / forced landing and is recorded as such.
- Every check writes one row (`ms_checks_stage{t}.csv`: update, block id and type, all D3 quantities, the sampler's bin probabilities digest, eligible, consecutive count) and every decision one entry of `rule_log.json` (block type, first and last local update, exit reason, fire update, S_t, localized/broad, `budget_forced`, the landing window, the final-tier gate values at the freeze).

### D5. The sampler: coverage-constrained strata plus verifier-guided priority

For the phase of stage t ≥ 2, on the ES bins of D_t (width 10; n_t bins): **tail bins** = bins inside |d| ≥ 2q (T − t + 1) (q = 50 stage 2 of T=2: 20 of 40; q = 60: 20 of 44); **near-tie bins** = bins intersecting (−20, 20) (4 bins, R2b's half-width); **middle bins** = the rest (16 / 20). Shares λ_T, λ_P, λ_M with λ_T + λ_P + λ_M = 1:

- **Coverage constraint:** λ_T is fixed at the bin-balanced tail share n_tail / n_t (0.5000 at q = 50, 0.4545 at q = 60); the runner refuses any config with a smaller λ_T. λ_M = 1 − λ_P − λ_T must be positive. A stage with no tail bins (stage 2 of T=3 at q ∈ {50, 60}, where 4q ≥ B) has λ_T = 0 and the mixture runs over near-tie and middle bins only.
- `p_strat(b)` = λ_T / n_tail on tail bins, λ_P / n_P on near-tie bins, λ_M / n_M on middle bins (bin-balanced within a stratum, uniform within a bin).
- **Priority mixing:** `p(b) = λ_T · u_tail(b) + (1 − λ_T) · [ (1 − α) · p_PM(b) + α · f(b) ]`, where `u_tail` is uniform over the tail bins, `p_PM(b) = p_strat(b) / (1 − λ_T)` over the non-tail bins, and `f` is the focus distribution over the non-tail bins: in a global block `f(b) ∝ ρ̄_b` (the EMA residual map; before the first check, or if all ρ̄_b = 0, `f = p_PM`); in a targeted polishing block `f(b) ∝ ρ̄_b` restricted to S_t. α = α_global in global blocks (an arm parameter, 0 or 0.5), α_polish = 0.5 in polishing blocks. The tail share is λ_T whatever f and α are (test).
- **Draw mechanics** as `StartSampler.peak_focused`: one `rng.random(n)` through the bin CDF, one `rng.random(n)` within the bin, from the `start` stream, so the stream consumption per update equals the R2b sampler's; with `scheme = bin_balanced` the locked `StartSampler.balanced` call is used unchanged (the C-MS1 identity). Roles are drawn after the starts, as now. New scheme name `stratified_priority` with keys `{lambda_P, alpha_global, alpha_polish, ema_beta, near_tie_half_width}`; λ_T and λ_M are derived and written to `run_config.json` and `manifest.json`.

### D6. Arms, comparators, criterion

Six arms, each 20 runs (q ∈ {50, 60} × seeds 10501–10510), full pipeline (terminal stage, then stage 1) in one process:

| arm | sampler in global blocks | rule | notes |
|---|---|---|---|
| `MS_base` | bin_balanced | disabled; fixed 1600 / 600 with v2.0's LR windows; checks every 25 reported, would-fire recorded | the regression arm: its terminal-stage end state must equal `parents_A` in 20/20 (C-MS1) |
| `MS_rule` | stratified_priority with λ_P = the bin-balanced near-tie share (4/40, 4/44), α_global = 0 — bin-balanced in distribution (the draw mechanics of D5, not the locked `balanced` call) | D4 with polishing (α_polish = 0.5) | the stop rule and polishing alone |
| `MS_s25a0` | stratified_priority λ_P = 0.25, α_global = 0 | D4 | |
| `MS_s25a5` | stratified_priority λ_P = 0.25, α_global = 0.5 | D4 | |
| `MS_s35a0` | stratified_priority λ_P = 0.35, α_global = 0 | D4 | |
| `MS_s35a5` | stratified_priority λ_P = 0.35, α_global = 0.5 | D4 | |

- **Comparators**, paired by (q, seed): the terminal-stage candidate against `parents_A` (v2.0's end of Phase A, identical to `rehearsal_v2_0`'s); the stage-1 candidate against `rehearsal_v2_0`'s end of Phase B. Also report every rule arm against `MS_rule` (the sampler's effect net of the rule).
- **Pre-registered criterion** (descriptive, not a gate, as in R1/R2b/R2c): primary metric |peak error| (`stage2_peak_rel_err_abs`) of the frozen terminal-stage candidate; (a) the 95 % percentile bootstrap interval (10,000 resamples, `numpy.random.default_rng(20261006)`, one fresh generator per (q, statistic) in table order) of the mean paired difference arm − `parents_A` lies below 0 at **both** q; (b) no run that passes G-A with its G-N part under `parents_A` fails it under the arm. No selection rule and no protocol change follow from this round; the decision is mine.
- **Reported per arm and q** (paired where a comparator exists): the signed peak error with its interval (whether it contains 0), runs with |peak error| ≤ 0.05, RMSE_pos, tail mean and max with the G-A verdicts, η₂ on both tiers with G-N, σ₂(0) and the smoothed-game share of the d = 0 gap (`tools/v2/decomposition.py`, as in R2b), `R_t`, `R_t^tail` and Δ_t at the freeze on both tiers, the budget to the freeze (updates, episodes, optimiser steps, wall time), the rule record (fire update or `budget_forced`, number and types of blocks, localized/broad classifications, |S_t| at each classification), the measured start shares per stratum and the raw-draw clamp counts (D1 columns), and for stage 1: the G-S error against the closed form, the first-order residual r_1(0)/s_1 and the induced-target decomposition (`induced_band`-style learning vs inherited), G-F, the stage-1 budget and rule record.

### D7. Gates evaluated (reported; no run-pass rule in this round)

At the terminal-stage freeze on the final tier: G-A (η₂/ΔW ≤ 0.005, RMSE_pos/e₂*(0) ≤ 0.05, tail mean/e₂*(0) ≤ 0.02) and the η₂ part of G-N (|dev − final| ≤ 0.001). At the end of the stage-1 phase: G-F (Ĝmax_full/ΔW ≤ 0.01), the Ĝmax part of G-N, G-S (|ê₁(0) − e₁*(0)|/e₁*(0) ≤ 0.05), S1 at 0.10. `gates.json` carries every value and verdict, the v2.0 combination G-A ∧ G-F ∧ G-N ∧ G-S for information, the global-RNG record and the continuation-table records.

### D8. Calibration before pre-registration; what may move

- P1c replays the development verifier on the stored weight exports of the 60 v2.0 runs (§2.4) and reports the tables listed there. The pre-registration (§2.5) then fixes every parameter of D4/D5. **The defaults of D4/D5 apply unless a calibration table gives a reason to change them; each change is stated with its table.** What the calibration may move: ρ_t, τ_t, M, K, N_block, U_cap, N_land, α_polish, β, the localized fraction. What it may not move: ε_t = 0.005, the coverage constraint, the arms of D6, the criterion, the bootstrap seed, the comparators.
- **Gate G1:** P1 ends with the commit of the pre-registration and a short status message to me (the calibration headline numbers, the parameters chosen, anything that differs from the defaults). **Do not launch P2 until I reply "proceed P2".** If I reply with changes, apply them to the pre-registration first (dated addendum, no edits), then launch.
- T=3 in the test suite only: a reduced-budget smoke of the full pipeline (three phases, the rule exercised, nested tables), the nested-table self-convergence (halving the panel width and doubling the nodes changes `Ṽ_2` by ≤ 1e-8 ΔW on a random frozen suffix) and agreement of `Ṽ_2` with the verifier's stage-2 Q on a refined tier (state step 0.25, 64 GL nodes) to ≤ 1e-6 ΔW, the T=3 strata counts (D_2: 40 / 44 bins with no tail bins, D_3: 80 / 88 with the tail at |d| ≥ 2q).

### D9. Reference roots (untracked files live where they were produced; every tool takes its root as an explicit argument and records it; a missing reference file is a stop-and-report)

- `parents_A`, `rehearsal_v2_0`, `confirmation_v2_0` (with the weight exports `weights/u*.npz` every 25 updates, `state_end_A.pt`, `state_end_B.pt`, `train_history.json`): `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_refine/parents_A/` and `.../results/v2_T2_locked/`.
- The canonical worktree for C7: `/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2`.
- Comparison tools: `tools/v2/cr2_compare.py` (training-relevant state), `tools/v2/compare_runs.py`; the R2b diagnostic `tools/v2/diag_seed30510.py` shows how the weight exports are turned back into policies (`agents.ppo_curriculum.mean_effort_numpy` or the actor reload) and evaluated on the verifier.

---

## 1. P0 — housekeeping

Write `reports/ms/r1/00_housekeeping.md` as you go (commands, hashes, verbatim outputs).

1. State: `git ls-remote origin` for `main` and the tag `t2-refine-100526`; verify that `origin/main` carries the publication (`reports/t2_refine_100526/` exists at `origin/main`, `git log -1 origin/main`).
2. Fast-forward `main` in the primary checkout `/home/fjiang4/tournament_experiment`: record `git status --porcelain` verbatim. If the only tracked modification is `SESSION_STATE.md`, stash it, `git fetch origin`, `git merge --ff-only origin/main`, pop the stash, record both; if any other tracked file is modified, or the session's worktree guard refuses git in the primary checkout (as in the publication round), skip this item and report it. Never reset, rebase or force.
3. `git worktree add .claude/worktrees/ms-r1 -b ms-r1 origin/main`; all P1–P3 work happens there. Record `nproc`, free disk, the Python and package versions against `requirements.lock`.

---

## 2. P1 — code, tests, reproduction checks, calibration, pre-registration (on `ms-r1`)

### 2.1 Code

- `utils/ms_continuation.py` (D2): nested tables for a frozen suffix, T-generic, the locked integration rule, the record fields of v2.0's `continuation_table` (rule parameters, build seconds, NPZ SHA-256) per table; `continuation_table_stage{t}.npz` per phase.
- `envs/curriculum_env.py`: the strata, `stratified_priority` probabilities and draw (D5), coverage-constraint validation; the existing methods byte-identical.
- `run/run_ms_stagewise.py` (D2–D5, D7): config schema `ms_run_config/1` embedding the v2.0 record (`protocols/v2_T2_locked_v2_0.json` `records[q]`) plus the sections `pipeline` (T, phases, budgets, LR windows for the legacy arm), `start_weights`, `rule` (every D4 parameter; `enabled`), the global-RNG hardening of the locked entry point, outputs per run as in v2.0 plus `ms_checks_stage{t}.csv`, `rule_log.json`, `state_end_stage{t}.pt`, `gates.json` (D7), `manifest.json` (commit, clean-tree flag, record SHA-256, arm, every parameter written out). The verifier call at a check is one `utils.v2_metrics.evaluate` on the development tier (pure, no RNG), as now; the final tier only at the freeze and at the end.
- `tools/ms/launch_ms_r1.py` modelled on `tools/v2/launch_refine.py` (dry-run that writes and validates every config, launch record with nproc / load / disk / HEAD / `git diff --stat <code-commit> HEAD` / `git status --porcelain`, bounded pool ≤ 40 single-threaded workers, `--code-commit`): waves `base` (`MS_base`, also C-MS1), `v20_repro` (C-R4) and `pilot` (the five rule arms). Every config carries every new key explicitly.
- `tools/ms/replay_dev_rule.py` (§2.4) and `tools/ms/r1_analysis.py` (§3.2) with their tests.

### 2.2 Tests (`tests/test_ms_*.py`, plus the existing suites)

- Config validation and refusals (coverage constraint, λ_M ≤ 0, unknown keys, the legacy arm's fixed budget, T ∉ {2, 3}).
- Sampler: probabilities per stratum; the tail share equals λ_T for every f and α (property test over random residual maps); α = 0 gives `p_strat`; the polishing focus is supported on S_t only; empirical shares with 10⁶ draws within 3 binomial SE; the `bin_balanced` path and the stream positions equal the locked sampler's; the `stratified_priority` draw consumes the `start` stream exactly as `peak_focused` does.
- Residual metrics (D3): a candidate equal to the verifier's one-step best response gives R_t = 0; a candidate shifted by c effort units on a region gives r_t = c there; the node-to-bin aggregation and the EMA; the stubbed-closed-form invariance of the rule and sampler decisions.
- Rule (D4): the M-consecutive logic, block transitions on synthetic check sequences (stop inside a block, localized → polishing, broad → global, cap → `budget_forced`), the landing LR follows `lr_at`'s linear form and ends at 3e-5, no training after a freeze, the t = 1 degenerate case.
- Continuation (D2): T=2 equality with `utils.v2_continuation` (≤ 1e-12 ΔW); T=3 nested self-convergence and refined-verifier agreement (D8); the y-grid coverage assertion; no RNG movement during a build.
- Reduced-budget bit-identity of the new runner's terminal phase under legacy settings with `run/run_v2_stagewise.py` mode `phase_A` (the C7-style smoke, same seed and q), and of the C7 reference itself (unchanged).
- T=3 reduced-budget smoke of the full pipeline with the rule exercised (D8); launcher configs (the five pilot arms differ from `MS_rule` only in `start_weights`); the analysis tool on synthetic rows; the replay tool on one stored export.
- Full suite `pytest tests`: everything passes apart from the known `test_registry_canonicalization` failure; record the summary line.

### 2.3 Reproduction checks (real runs, development seeds)

- **C-R4:** the unchanged `run/run_v2_T2_locked.py` from `ms-r1`, q ∈ {50, 60} × seeds 10501–10510, into `results/ms_r1/v20_reproduction/`, compared with `rehearsal_v2_0` (root D9) by `tools/v2/cr2_compare.py`: 20/20 identical in the training-relevant state (including the continuation table), or stop and report.
- **C-MS1:** the 20 `MS_base` runs (wave `base`, into `results/ms_r1/base/q*/seed*/MS_base/`): the terminal-stage end state (`state_end_stage2.pt`: actor, critic, lagged opponent, both Adam states, the five streams and the torch generator, all weight exports) equals `parents_A`'s end of Phase A in 20/20, or stop and report. These runs continue into their stage-1 phase and are the `MS_base` arm of the pilot.
- Record both in `reports/ms/r1/03_checks.md` with the comparison outputs.

### 2.4 Offline calibration (`tools/ms/replay_dev_rule.py`, no training)

Replay the development verifier on every stored weight export of the 20 `rehearsal_v2_0` and the 40 `confirmation_v2_0` runs (root D9): Phase A exports u0025–u1600 (candidate = the export's stage-2 head), Phase B exports u1625–u2200 (candidate = the export at t = 1 with the run's u1600 export frozen at t = 2). At every export compute the D3 quantities (Δ_t, R_t, R_t^tail, the per-bin map, C_t) and, for reporting, the closed-form metrics (signed and absolute peak error, RMSE_pos, tail mean for Phase A; the signed stage-1 error for Phase B). Write one CSV per run under `results/ms_r1/calibration/` and the summary tables of `reports/ms/r1/01_calibration.md`:

1. Spearman correlation, over all (run, export) pairs with u ≥ 400, of |peak error| with R_t and with Δ_t (the point of the first-order residual); the same per q; a scatter figure.
2. For every candidate rule in the grid ε = 0.005 × ρ ∈ {0.02, 0.03, 0.05} × τ = 0.02 × M = 3 × K = 25: per run the first fire export (or none) and |peak error|, RMSE_pos, tail mean at the fire export against u1600; medians and ranges per q. Also the original second-order rule (ε ∈ {0.02, 0.005}, no first-order part) at K = 25 for comparison with fact 2 of the preamble.
3. The dev-tier against final-tier difference of R_t, R_t^tail and the per-bin map at u1600 in all 60 runs (the grid noise of the first-order residual; the final tier is computed for this table only).
4. The localized / broad classification at u1600 and at each candidate fire export under the 0.25 rule, with |S_t| and the bins in S_t (where is the residual: near-tie, middle?).
5. Phase B: r_1(0)/s_1 against the signed stage-1 error at every export; the fire export of the stage-1 rule for the same grid; how the landing window of D4 compares with v2.0's 600-update decay (descriptive).
6. Cost: verifier seconds per development call at T=2 (both tiers).

### 2.5 Pre-registration and the code commit

`reports/ms/r1/02_preregistration.md`: the pipeline structure (D2) and what differs from v2.0; the metrics (D3); every rule and sampler parameter with its value and the calibration table that supports it, and each departure from the D4/D5 defaults stated as such; the arms (D6); the criterion, the bootstrap seed and the comparators (D6); the gates reported (D7); the reference roots; the checks C-R4 and C-MS1 and the test-suite summary; the launch plan. Commit code, tests, the calibration outputs and the pre-registration (**code commit**); later launch commits touch only `results/` and the report folder.

### 2.6 Gate G1 — STOP

Send me the status message of D8 and stop. Launch P2 only after my "proceed P2". Nothing of P2 is started, not even a dry-run that writes into `results/ms_r1/pilot/`.

### 2.7 Push after P1

Push `ms-r1` (fast-forward, no force) once the pre-registration is committed, so that I can read `01_calibration.md` and `02_preregistration.md`; record `git ls-remote`.

---

## 3. P2 — the pilot wave and its analysis (after "proceed P2")

### 3.1 Launch

- The five rule arms × 20 = 100 runs through the launcher (wave `pilot`, ≤ 40 single-threaded workers in tmux; `nproc`, load and disk recorded; nothing changes after launch), output `results/ms_r1/pilot/q*/seed*/<arm>/`, one launch record. Crash rule as in v2.0: an infrastructure kill is re-run once into the original path after moving the first attempt to `pilot/crashed/`; any other nonzero exit (pipeline exception, global-RNG violation, verifier error) counts as a failed run and is reported, not re-run.
- Post-launch checks (`results/ms_r1/pilot/launch_checks.json`): every manifest at the code commit with `dirty: false`; every run has its rule record for both stages, its check tables and its weight exports; the measured start shares per stratum per run equal the design shares within 3 binomial SE over the block (the polishing blocks against their own S_t focus); the global-RNG assertions all pass; the tail share λ_T is met in every block of every run.

### 3.2 Analysis (`tools/ms/r1_analysis.py` → `results/ms_r1/analysis/`)

- Per-run table (`per_run.csv`: every D6 quantity for both stages, both tiers, plus the rule record flattened); paired tables against `parents_A` (terminal stage), `rehearsal_v2_0` (stage 1) and `MS_rule`; `criterion.csv` with parts (a) and (b) per arm; `budget.csv`; `rule.csv` (fire updates, blocks, classifications, `budget_forced` counts); `tail.csv`; `decomposition.csv` (smoothed-game share per run); `decision_inputs.csv`; figures (paired differences per arm with intervals; peak error against updates for every run with the fire and landing marked; the per-bin residual maps at the freeze; start shares per stratum over blocks).
- An independent recomputation of `criterion.csv` from `per_run.csv` by a script that does not import the analysis tool (as R2c's `blind_recomputation.txt`), agreeing to 1e-12.
- Everything that is not the pre-registered criterion is labelled descriptive; an interval that contains 0 with 10 seeds does not show that a mechanism has no effect, and the report says so where it applies.

---

## 4. P3 — reports and the push

### 4.1 Reports (`reports/ms/r1/`)

`00_housekeeping.md`, `01_calibration.md`, `02_preregistration.md` (plus dated addenda for anything I changed at G1), `03_checks.md` (C-R4, C-MS1, the test suite, the launch checks), `04_pilot.md` (one section per arm: paired tables with intervals and criterion parts, the rule statistics, the budget, the stage-1 results, figures, anomalies), `05_decision_inputs.md` (the arms side by side: criterion, |peak| ≤ 0.05 counts, signed-error intervals, tail, budget, the smoothed-game share — what sampling removed and what remains), `summary.md` (reading order, headline numbers with paths, deviations and open items, the commands to reproduce every table). Add `reports/ms/README.md` as the index of the multistage line (this round, its branch and commits, the reference roots, the tracking rules), and a dated section at the top of `docs/STATE.md`.

### 4.2 Tracking

Commit the lightweight records with the usual rules (per-run `manifest.json`, `run_config.json`, `status.json`, `gates.json`, `rule_log.json`, the check CSVs, the per-update CSV, the continuation tables, the analysis tables, the calibration CSVs; no `.pt`, no `train_history.json`, no weight exports).

### 4.3 Push

Push `ms-r1` (fast-forward, no force) once the reports are committed. Do not touch `main`, do not create tags; those follow my review.

---

## 5. STOP

Stop after the §4.3 push. No fresh seeds, no lock, no confirmation, no T=3 experiment, no second pilot, no change to any parameter after the pre-registration except through my reply at G1. The next round is mine to decide.
