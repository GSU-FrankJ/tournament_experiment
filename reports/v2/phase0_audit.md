# Phase 0 — Read-only audit of the T=2 pipeline

Date: 2026-10-01. Branch `v2-stagewise-pilots`. No code was changed in this phase.

Path conventions:
- Code paths are relative to the repo root of this branch.
- `RAW/` means `/home/fjiang4/tournament_experiment/experiments/`. This is the untracked raw run archive in the main checkout; it holds per-run `config.json`, `final_eval.json`, `train_history.json` and `checkpoint.pt`.
- `PUB/` means `experiments/` on this branch, the published compact tables.

---

## 0. Repository, branch and base (deviation: read first)

| Item | Finding | Source |
|---|---|---|
| Server | `vector2` | `hostname` |
| Repo top level | `/home/fjiang4/tournament_experiment` (this session works in its git worktree `.claude/worktrees/tournament-v2-stagewise-pilots-f97609`) | `git rev-parse --show-toplevel`, `git worktree list` |
| `survey` directory | **Not found.** A search of `/home/fjiang4` and `/` (depth 4) found no directory named `survey`. The repo is under `/home/fjiang4/tournament_experiment`. | `find / -maxdepth 4 -type d -name survey` → empty |
| Working tree at start | clean | `git status` |
| **Base of `v2-stagewise-pilots`** | Created from **`657f54a`** (branch `codex/multistage-experiments-20260927`, pushed to origin), **not from `main` (`d1b8443`)**. | `git checkout -b v2-stagewise-pilots 657f54a` |

**Why this base.** `main` does not contain the T=2 curriculum code at all. `agents/ppo_curriculum.py`, `envs/curriculum_env.py`, `utils/dp_br_verifier.py`, `run/run_final_dp_br.py` and `run/run_final_dp_br_round3_dense.py` are missing there. Commit `657f54a` ("Publish reproducible two-stage and three-stage experiment archive") is the only committed location of the code that produced the latest T=2 runs.

**Code identity check.** The six numerical files on this branch were compared with the files in the worktree that actually ran the latest cohorts (`.claude/worktrees/candidate-search-recovery-data-92d49a`; that worktree is named in each run's `status.json` `cmd`):
- Five files are byte-identical by sha256: `ppo_curriculum.py`, `curriculum_env.py`, `dp_br_verifier.py`, `theory_multistage.py` and `run_final_dp_br.py`.
- `run_final_dp_br_round3_dense.py` differs only in `MANIFEST_DEFAULT` (lines 94–95, a path constant). No training or verifier logic differs.

**→ Decision needed (Q0):** confirm the `657f54a` base. Also confirm that the missing `survey` directory is not a different checkout I should be using.

---

## 1. Code map

### 1.1 T=2 runner and entry points

| Role | Location |
|---|---|
| Runner of every final-protocol T=2 run | `run/run_final_dp_br_round3_dense.py`: `main()` lines 192–660. Reads one manifest record (`load_record`, lines 121–129) and applies strict config checks (lines 211–283). |
| Shared helpers (rollout, verifier wrapper, final evaluation) | `run/run_final_dp_br.py`: `collect_batch` 149–237, `run_verifier` 244–250, `recovery_metrics` 266–324, `final_evaluation` 355–459. Its own `main()` (466+) is the older round-1 runner and was not used for the latest runs. |
| Publication launcher | `MultiStage/two_stage/run_experiment.py`: spawns one runner subprocess per manifest record with OMP/MKL/OPENBLAS=1 (lines 202–229). |
| Protocol document | `MultiStage/two_stage/protocol/FINAL_T2_PROTOCOL_20260922.json` |

### 1.2 Environment, state encoding and noise

All of these are in `envs/curriculum_env.py`:
- **Game parameters:** `GameSpec`, lines 25–85.
- **Noise model:** eps ~ U(−q, q), i.i.d. per player and stage (docstring line 4). It is drawn in `collect_batch` (`run/run_final_dp_br.py:190-191`, `rng_env.uniform(-q, q)`). So the shock difference ξ = eps_i − eps_j has a triangular density on [−2q, 2q].
- **Transition:** d' = d + e_i − e_j + eps_i − eps_j (`step_gap`, lines 118–123). d is the player's own signed gap, kept in float64.
- **Observation:** `encode_obs`, lines 62–81, gives `[(t−1)/(T−1), d/((t−1)B)]`, cast to float32.
  - d is forced to 0 at t=1.
  - B = (e_max − e_min) + 2q (`GameSpec.B`, lines 47–50).
  - There is **no running observation normalizer**: the encoding is a fixed deterministic function.
- **Feasible domain:** D_t = [−(t−1)B, (t−1)B] (`domain_half`, lines 52–54). For T=2 this gives D_2 = [−200, 200] at q=50 and [−220, 220] at q=60.

### 1.3 Terminal reward and sign convention

- `GameSpec.terminal_reward`, lines 56–60: pays w_h if d_final > 0, w_l if d_final < 0, and (w_h+w_l)/2 on a tie.
- `stage_reward`, lines 126–131: r_t = −k e_t², plus the realized prize when t ≥ T.
- In `collect_batch` the reward is called with the **learner's own** post-shock gap (`run/run_final_dp_br.py:197-198`). The terminal reward is therefore a **sampled** outcome.
- The verifier's terminal expectation is W(y) = w_l + ΔW·F_ξ(y), with y = d + e_i − e_j (`utils/dp_br_verifier.py:356-359`). This is consistent with the env because P(y + ξ > 0) = F_ξ(y) for the symmetric ξ.

### 1.4 ActorCritic architecture

All in `agents/ppo_curriculum.py`:
- **Actor** (`BetaActor`, lines 65–87): 2 → 64 tanh → 64 tanh → 2.
  - μ = clamp(sigmoid(z_μ), 1e−6, 1−1e−6) and c = 100 + softplus(z_c).
  - α = μc, β = (1−μ)c.
- **Critic** (`Critic`, lines 90–106): a separate 2 → 64 → 64 → 1 network.
- **No shared trunk.** Actor and critic are separate networks with separate Adam optimizers (lines 116–122).
- **Stage input.** The single actor network serves both stages. Stage enters through the τ = (t−1)/(T−1) input feature. There are no per-stage heads.
- **Opponent snapshot.** `CurriculumPPO.opponent` (lines 124–135) is a deep copy of the actor with `requires_grad=False` and `eval()` set. It is refreshed every `snapshot_every=20` global updates and at every phase entry (runner lines 400, 447–450).
  - Rollouts are therefore against a **lagged snapshot, not the live actor**.
  - Note the name clash: "snapshot" in the existing code means this self-play opponent, not the §1.5 freeze snapshot.

### 1.5 PPO update

`CurriculumPPO.update`, `agents/ppo_curriculum.py:173-274`:
- **Advantages:** normalized per update over the **whole buffer, all stages pooled** (lines 194–196).
- **Epochs and minibatches:** 10 epochs over minibatches of 256 drawn from a single permutation of all transitions (`rng_mb.permutation`).
- **Losses:** actor loss = −clipped surrogate, with entropy coefficient 0. Critic loss = 0.5·MSE. Each has its own gradient-norm clip at 0.5.
- **How stage-specific samples enter.** `collect_batch` concatenates the transitions of every stage that the episode visited into one buffer (`run/run_final_dp_br.py:223-227`). Every transition, stage 1 or stage 2, enters both the policy loss and the value loss. There is no stage mask anywhere.

### 1.6 Backward curriculum and phase structure

Phases are defined in the runner at `run/run_final_dp_br_round3_dense.py:372`; start states at lines 415–427.

| Phase | Starts | Transitions/update | Active stages |
|---|---|---|---|
| A | 512 episodes at t=2, gaps drawn bin-balanced on D_2 | 512 | stage 2 only |
| B | 512 episodes at root (t=1, d=0) | 1024 | 1 and 2 (joint) |
| C | 256 root + 256 at t=2 bin-balanced, shuffled | 768 | 1 and 2 (joint) |

In every phase the learner role is drawn from 0/1 uniformly (line 427).

**Mapping to the prompt.** The prompt assumes two phases: a terminal phase and a stage-1 phase. The code has three:
- Phase A is the "stage-2-only (terminal) phase".
- Phase B is the root-start joint phase.
- Phase C is a second joint phase with mixed starts. The prompt does not mention it.

See Q1.

### 1.7 Checkpoint save and load

- `CurriculumPPO.state()`, `save()` and `load_weights()` are at `agents/ppo_curriculum.py:277-299`.
- `checkpoint.pt` is written **once, at the C stopping point** (runner lines 591–593).
- `weights/u%05d.npz` holds actor and critic weights only, exported every `weights_every=25` global updates (lines 451–455).
- No mid-run full-state checkpoint exists. See §7.

### 1.8 Verifier

All in `utils/dp_br_verifier.py`:
- **Grids:**
  - `stage_grid`, lines 186–204: D_1 = {0}. D_t is a symmetric `linspace` that contains 0 and both endpoints, with spacing ≤ step.
  - `effort_grid`, lines 207–210.
  - `gl_shock_rule`, lines 213–230: Gauss–Legendre on [−2q, 0] ∪ [0, 2q], weighted by f_ξ.
- **DP-BR:** `verify`, lines 292–537, does a backward pass. For each stage t and every grid point it computes:
  - `v_br` = V_t^BR(d) and `a_br`;
  - `v_mean` = V_t^ê(d);
  - `delta` = the one-step Δ_t(d) against the mean continuation.

  These are full-grid arrays over D_t (lines 382–436).
- **Action search** (lines 393–402 and 405–417): the effort grid, plus valid concave-parabola vertices (`_vertex_candidates`, 259–285), plus the candidate's own mean action. The Δ search additionally includes `a_br`. Ties go to the smaller effort (`_select`, 237–256).
- **Interpolation:**
  - Stage-1 continuation: linear `np.interp` of V_2 *values* on the D_2 grid (lines 364–375).
  - A landing point off-grid raises `DomainError`. There is no extrapolation (lines 370–373).
  - Terminal stage: closed form W(y) = w_l + ΔW·F_ξ(y) (lines 356–359).
- **Reach mask R_t** (lines 438–469):
  - R_1 = {0}.
  - R_{t+1} = the grid points covered by the union over d ∈ R_t of [d + a_BR(d) − ê(−d) − 2q, d + a_BR(d) − ê(−d) + 2q] ∩ D_{t+1}.
  - For T=2 this is exactly the closed support of the stage-2 state distribution on the root BR path: one interval, centred at a_BR(0) − ê(0).
- **Forward PMF** under the BR chain (GL nodes plus linear mass split): lines 470–484. It feeds the PDL residual check (lines 506 and 513).
- **Aggregates** (lines 486–513):
  - `exp_root` = V_1^BR(0) − V_1^mean(0);
  - `dreach` = Σ_t max over R_t of Δ_t;
  - `dfull` = Σ_t max over D_t of Δ_t;
  - `delta_max_all` = max_t max over D_t of Δ_t.
- **Validity reasons** (lines 516–526): non-finite values, empty reach, PMF mass error, PDL residual > 1e−10·ΔW, or GL weight sum ≠ 1.
- **Concentration:** `concentration_stats`, lines 552–599.
- **Array export:** `stage_result_arrays`, lines 602–616.
- **Relation to the new metric.** `v_br` and `v_mean` are already full-domain arrays. Ĝmax_full = max_t max_d (v_br − v_mean) can therefore be read from existing outputs with no new DP; it is simply never aggregated today.

### 1.9 Analytical benchmark

All in `utils/theory_multistage.py`:
- `f_xi` (lines 54–65) is the triangular density (2q − |x|)/(4q²) on |x| ≤ 2q. `F_xi` (lines 68–93) is its CDF.
- `g2_two_stage` (lines 100–122) is clip(ΔW·f_ξ(d)/(2k), 0, e_bar).
  - The **clip to [0, 100] is an addition relative to the prompt's formula.** It is inactive at q ∈ {50, 60}: g2(0) = 70 and 58.33 respectively (computed with these functions in this audit).
- `g1_two_stage` (lines 125–139) is ΔW/(6kq), giving 46.667 (q=50) and 38.889 (q=60).
- Recovery metrics against the closed form already exist: `run/run_final_dp_br.py:266-324`.
  - Positive region |d| < 2q, tail = its complement.
  - Peak at d = 0 (0 is on the grid).
  - Symmetry metric max|ê2(d) − ê2(−d)|.
  - Evaluated only in final evaluation, on a 0.5-step grid (`recovery_step`).

---

## 2. As-run configuration of the latest T=2 runs

"Latest" means the four cohorts run under the frozen final protocol (80 runs):
- `two_stage_confirmation_T2_20260922`
- `two_stage_E1_q50_p_20260923`
- `two_stage_q50_restarts_20260924`
- `two_stage_q50_precision_20260924` (newest, created 2026-09-24T14:31 UTC)

All values below come from the manifest record that the runner reads verbatim (`PUB/two_stage_q50_precision_20260924/manifest.json`, record `tel_q50_s10231`). I checked that they are identical in the q=50 restarts record and the q=60 confirmation record (`PUB/two_stage_confirmation_T2_20260922/manifest.json`), apart from q, B and the bin count.

| Setting | Value (q=50 / q=60) | Source |
|---|---|---|
| w_h, w_l, ΔW | 6, 2, 4 | manifest `game`, `dw` |
| k | 1/3500 = 0.000285714… | manifest `game.k` |
| q | 50 / 60 | manifest |
| Effort bounds | [0, 100] | manifest `game.e_min/e_max` |
| B, D_2 half-width | 200 / 220 | manifest `B`, `domain_half_stage2` |
| Episodes per update | 512 | `protocol.episodes_per_update` |
| Phase caps | A 400, B 600, C 1000 | `protocol.phase_caps` |
| Phase C mix | 256 root + 256 stage-2 exploring starts | `protocol.phase_c_root/phase_c_es` |
| Exploring-start bins | width 10 → 40 / 44 bins on D_2 | `protocol.es_bin_width`, `es_bins_stage2` |
| Verifier cadence | warm-up forced call at local 100; stability check every 20; stability-triggered call after 2 consecutive stable checks (drift ≤ 0.01 of range and KL ≤ 0.01); timeout A 100 / B 25 / C 25; a phase-end call if none occurred at the cap | `protocol.warmup`, `stability_every`, `stability_consecutive`, `drift_thr`, `kl_thr`, `verifier_timeout`; logic at runner lines 463–493 |
| Development verifier tier | state step 4, effort step 1, GL 16/half (D_2 grid 101 / 111 points; effort grid 101) | `verifier.development`, equal to `DEV_CONFIG` at `utils/dp_br_verifier.py:66`; point counts from `stage_grid`/`effort_grid` |
| Final verifier tier | state step 2, effort step 0.5, GL 32/half (D_2 grid 201 / 221 points; effort grid 201) | `verifier.final`, equal to `FINAL_CONFIG` at line 67 |
| PPO | lr 3e−4 constant (actor and critic), Adam (0.9, 0.999, 1e−8), wd 0, clip 0.2, value coef 0.5, grad-norm 0.5, 10 epochs, minibatch 256, γ = λ = 1, advantage-norm eps 1e−8 | `ppo`, `lr_schedule.kind=constant` |
| Entropy | coefficient 0, no schedule | `ppo.entropy_coef` |
| Beta | mean/concentration parameterization, c_min = 100, μ clamp 1e−6, action clamp [1e−6, 1−1e−6], sampled with numpy float64 and stored as float32 | `ppo.c_min/mu_clamp/action_clamp`, `action_sampling` |
| Opponent snapshot | refresh every 20 global updates and at phase entry | `protocol.snapshot_every` |
| Stop / advance | k_phase 3 (A, B), k_stop 1 (C); thresholds A 0.02 (max Δ_2 on the dev D_2 grid)/ΔW, B 0.02 (EXP_root)/ΔW, C 0.01 (dReach)/ΔW; concentration max std/range ≤ 0.04 | `protocol.k_phase/k_stop/phase_thr_over_dw/conc_thr` |
| Weights export | every 25 global updates | `protocol.weights_every` |
| RNG | numpy `SeedSequence([seed, q, namespace])` with namespaces init 0, env_noise 1, learner_action 2, opponent_action 3, starts_roles 4, minibatch 5. The torch generator is seeded from the init stream and used **only** for the weight init. | `protocol.rng_namespaces`; runner lines 307–313 |
| Device / threads | CPU, 1 thread (torch, OMP, MKL, OPENBLAS) | `device`, `threads_per_process` |
| Versions (enforced) | Python 3.12.3, torch 2.5.1+cu121, numpy 2.5.0. The runner refuses on mismatch (lines 212–215). `/home/fjiang4/tournament_experiment/.venv/bin/python` matches; there is no `python` on PATH. | manifest `versions`; checked in this audit |

**Config vs. code-default disagreements.** These matter only if the round-1 `main()` were used; the round-3 runner replaces `PROTOCOL` with the manifest values at lines 282–283.

| Key | Code default (`run/run_final_dp_br.py:62-76`) | As run (manifest) |
|---|---|---|
| `verifier_timeout` | 100 (scalar) | {A 100, B 25, C 25} |
| `k_stop` | 5 | 1 |
| `phase_thr_over_dw.C` | 0.015 | 0.01 |
| `weights_every` | absent | 25 |

`PPOConfig` defaults (`agents/ppo_curriculum.py:38-57`) equal the manifest `ppo` block. `DEV_CONFIG` and `FINAL_CONFIG` equal the manifest, and the runner enforces this at lines 231–233.

---

## 3. Stopping logic (all early-exit rules)

All rules are in `run/run_final_dp_br_round3_dense.py`.

1. **Phase cap.** The `while local < cap` loop (line 406) ends a phase with `exit_reason="budget_exhausted"`.
2. **A/B advance.** After each verifier call, a call is *eligible* when it is valid, the phase criterion is ≤ threshold, and concentration passes (lines 501–509). `eligible` counts consecutive eligible calls and resets to 0 on a non-eligible call. When it reaches `k_phase=3` in A or B, the phase exits with `"verifier_passed"` (lines 541–543). Unused budget is dropped.
   - Phase A criterion: `full_delta_max[2]/ΔW`, i.e. the max over the full dev D_2 grid of Δ_2 (line 379).
   - Phase B criterion: `exp_root/ΔW` (line 381).
3. **C stop.** When `eligible ≥ k_stop=1` in C, i.e. at the first eligible C call, training stops with `"k_stop_passes"` (lines 544–557). That iterate becomes the checkpoint ("first eligible").
4. **Verifier triggers.** These do not stop training, but they decide when rules 2–3 are evaluated (lines 484–493): a forced call at local 100; a stability call after local 100 when 2 consecutive stability checks pass; a timeout call when local − last_call ≥ the timeout; otherwise a phase-end call at the cap. When several triggers fire at once, the priority is warm-up, then stability, then timeout, then phase end.
5. **Invalid verifier.** A `DomainError`/`FloatingPointError`/`ValueError` is caught by `run_verifier` and makes the call ineligible (`run/run_final_dp_br.py:244-250`). It does not stop training.
6. **Exceptions.** Any other exception fails the run (`status.state="failed"`, lines 648–660).

**When each rule fired in the 80 latest runs** (each run's `final_eval.json` → `curriculum`; tallied in this audit from `RAW/<cohort>/runs/FINAL_A400_B25_C25/*/final_eval.json`):

| Cohort | A exit | B exit | C exit |
|---|---|---|---|
| confirmation q50 (n=10) | budget 10 | verifier_passed 8, budget 2 | k_stop 7, budget 3 |
| confirmation q60 (n=10) | budget 10 | verifier_passed 10 | k_stop 10 |
| E1 q50 (n=20) | budget 20 | verifier_passed 18, budget 2 | k_stop 19, budget 1 |
| restarts q50 (n=30) | budget 30 | verifier_passed 23, budget 7 | k_stop 25, budget 5 |
| precision q50 (n=10) | budget 10 | verifier_passed 7, budget 3 | budget 6, k_stop 4 |

The phase-A gate never fired in these 80 runs. It also never fired in the 40 runs of the 0915 formal cohort, by that cohort's own audit (`MultiStage/two_stage/protocol/FINAL_T2_PROTOCOL_20260922.json`, field `phase_A_status`). Phase A is therefore already de facto fixed at 400 updates.

---

## 4. Joint-training semantics today

- **Stage-2 samples in the stage-1 phases.** In phases B and C, stage-2 learner transitions **do** contribute to the policy loss. All visited stages are concatenated into one buffer (`run/run_final_dp_br.py:223-227`), and `update` applies the clipped surrogate to every row (`agents/ppo_curriculum.py:207-224`). There is no mask. Advantage normalization is pooled over stage 1 and stage 2 (lines 194–196).
- **Stage-2 action mode in rollouts.** It is **stochastic for both players**:
  - The learner samples from the live actor's Beta with `rng_learn` (`run/run_final_dp_br.py:186-187`).
  - The opponent samples from the lagged snapshot's Beta with `rng_opp` (lines 188–189).
  - No rollout uses the mean action.

---

## 5. Start-state distribution of the stage-2-only phase (A)

- **How starts are drawn:** `StartSampler.balanced(2, 512, rng_start)` (`envs/curriculum_env.py:105-110`). It picks one of the equal-width bins on D_2 uniformly, then draws uniformly inside that bin. The bins are 40 of width 10 on [−200, 200] at q=50, and 44 on [−220, 220] at q=60.
  - Gaps are stored as player 0's gap. The learner role is a fair coin, so the learner sees +d or −d (`run/run_final_dp_br.py:173`).
  - The opponent is queried at −d_learner.
- **Coverage.** The sampling support equals D_2 exactly, i.e. the full feasible set of §1.4. D_2 = [−(e_max−e_min) − 2q, (e_max−e_min) + 2q] is reached from d_1 = 0 under any action pair in [0, 100]² and any shock pair; this is the same interval the verifier uses (`stage_grid(2, B, ·)`). The density is uniform, so there is **no coverage gap**.
- **Caveat.** The density is uniform in d, while the root-path stage-2 distribution is concentrated near the centre. This audit only describes the distribution; it does not judge it.

---

## 6. Evaluation convention for ê

The evaluated policy is the **Beta mean**: ê_t(d) = e_min + (e_max − e_min)·α/(α+β), computed in float64 from the float32 α and β. The observation is built exactly as in training.
- Source: `make_policy_fns`, `run/run_final_dp_br.py:125-142`; manifest `mean_extraction`.
- The opponent in the verifier is the same function at −d (`utils/dp_br_verifier.py:343, 386`).
- There is a framework-free reimplementation, `mean_effort_numpy`, at `agents/ppo_curriculum.py:309-335`.

---

## 7. Checkpoint completeness

`checkpoint.pt` contains `actor`, `critic`, `opponent` (snapshot weights), `opt_actor`, `opt_critic` and `cfg` (`agents/ppo_curriculum.py:277-286`). It is saved only at the C stopping point.

| Required for exact branching | Present? |
|---|---|
| Actor / critic / opponent-snapshot weights | yes |
| Optimizer states (both Adams) | yes |
| numpy RNG states: env_noise, learner_action, opponent_action, starts_roles, minibatch | **missing** |
| torch RNG / CUDA RNG | **missing**. torch is used only for the init (orthogonal) and the device is CPU, so after init no torch RNG is consumed. Beta sampling is numpy (`ppo_curriculum.py:147-152`). |
| python `random` state | **missing**, and not used by these modules (no `import random` in the six files) |
| Observation-normalizer statistics | not applicable: there is no normalizer (§1.2). Advantage normalization is recomputed per batch. |
| LR / entropy schedule position | not stored. LR is constant 3e−4 and entropy coefficient 0, so there is nothing to restore. |
| Counters: global update, phase, local update, `eligible`, `stab_consec`, `prev_stab`, `last_call`, snapshot-refresh phase (`global_u % 20`) | **missing**, except `snapshot_refreshes` lives on the object but is not in `state()` |
| A-end (stage-2-only) checkpoint | **missing.** No full state is saved at the A→B boundary. Only `weights/u00400.npz` exists (actor and critic weights, no optimizer and no RNG), plus `phase_A_exit_arrays.npz` (verifier arrays, not weights). |

**Consequence.** No existing run can serve as an exact branching parent for Pilots 2 and 3. Pilot 1 has to produce the parents, as the prompt already plans.

---

## 8. Seeds and q values

**q values used for T=2.** Only 50 and 60 were used, in all T=2 cohorts under the final DP-BR protocol (manifests in `PUB/two_stage_*`; `PUB/two_stage_formal_T2_20260915/formal_results.csv` has q ∈ {50, 60}).

**Seeds already used anywhere in the repo** (T=2 and T=3, all worktrees, the main checkout and `/home/fjiang4/tournament_experiment_upload_20260927`). They were collected from run-directory names `*_s<seed>` and from `seed`/`seeds`/`held_out_training_seeds` fields in every `manifest*.json`, `formal_settings.json` and `study.json`:

- ≥ 10000: 10001–10020, 10101–10140, 10201–10240, 10300–10303, 10400–10403, 10410–10413, 10430–10433, 10440–10443, 10461–10463, 11001–11020, 11101–11120.
- < 10000: 47, 48, 49, 50, 51, 60, 80.
- The scan script is `seeds.py` in this session's scratchpad. It is not committed and can be added if you want it.

**Proposal (for your confirmation):**
- **q ∈ {50, 60}**: the two q values of every final-protocol T=2 cohort.
- **Diagnostic seeds 10501, 10502, 10503.** They appear in no scanned name or manifest; the nearest used seeds are 10463 and 11001.
- The same three seeds are used at both q. RNG streams are keyed by (seed, q, namespace), so the q=50 and q=60 streams differ.

---

## 9. Launch, monitoring and cost on vector2

- **Launch mechanism.** A Python launcher runs one subprocess per run through a `ThreadPoolExecutor`, at most 10 concurrent, single-threaded CPU (`RAW/two_stage_q50_precision_20260924/launch.py`, lines 24–47; publication equivalent `MultiStage/two_stage/run_experiment.py`).
  - The launcher PID was recorded (`launcher_pid.txt`).
  - How the launcher itself was detached (tmux, nohup or setsid) is not recorded → **UNKNOWN**.
  - The project `CLAUDE.md` requires tmux for long runs; I will use `tmux new-session -d`.
- **Monitoring.** Per run: `status.json` (`state` running/done/failed), `logs/<run>.log` (verifier lines `[u… X…] verifier(...)`), and the batch's `launch_status.json`.
- **GPU.** Runs are CPU-only (`device: "cpu"`, enforced at runner line 216), so GPU usage is 0. `nvidia-smi` at audit time showed 8× V100, 0 MiB used and 0 % utilization. There were 64 CPUs and the load average was 0.32.
- **Wall clock.** These are measured run totals from `final_eval.json` → `costs.total_wall_sec`, tallied from `RAW/<cohort>/runs/FINAL_A400_B25_C25/*/final_eval.json`:

| Cohort | n | total wall min / median / max (s) | median dev-verifier total (s) | median final eval (s) |
|---|---|---|---|---|
| confirmation q50 | 10 | 172.5 / 261.0 / 321.3 | 0.35 | 2.02 |
| confirmation q60 | 10 | 119.9 / 147.1 / 182.2 | 0.15 | 1.99 |
| E1 q50 | 20 | 164.7 / 204.7 / 299.6 | 0.24 | 2.02 |
| restarts q50 | 30 | 148.4 / 219.8 / 319.7 | 0.28 | 2.04 |
| precision q50 | 10 | 178.0 / 298.0 / 311.1 | 0.42 | 2.05 |

- **Per-phase wall clock is UNKNOWN.** It is not logged: `costs` has only run totals, and `history` rows have no timestamps.
- **Logged per phase:** minibatch steps per phase (`curriculum[*].minibatch_steps`). For example, `RAW/two_stage_q50_precision_20260924/runs/FINAL_A400_B25_C25/tel_q50_s10231/final_eval.json` shows A 8000, B 21000 and C 5250 minibatch steps, with `train_update_sec` 170.6 s out of 177.96 s total wall.
- **Derived figure (not a measurement):** the median `train_update_sec / total minibatch steps` is about 0.0049 s per step in every cohort (same tally). An A-only run (8000 steps) should therefore take on the order of a minute of CPU. Phase 3 will report the measured value.

---

## Report format items

### What was done

Files read: the files cited above, plus the four cohort manifests and the 80 `final_eval.json` files. Files changed:

| File | Change |
|---|---|
| `reports/v2/phase0_audit.md` | new (this report) |

No code changes.

### Tests run

None; this phase was read-only. Pre-existing T=2 tests are only `experiments/two_stage_q50_restarts_20260924/test_restarts.py`, a reporting regression; `tests/` has no T=2 tests. **pytest is not installed** in `/home/fjiang4/tournament_experiment/.venv` (`No module named pytest`). Phase 1 needs it (see Q8).

### Anomalies, deviations and open questions

- **Q0 — Base and path.** See §0. Branch `v2-stagewise-pilots` is based on `657f54a`, not `main`, and there is no `survey` directory.
- **Q1 — Which phase is "the stage-1 phase"?** The code has two joint phases.
  - B is root-start with cap 600. C is 256 root + 256 stage-2 exploring starts with cap 1000.
  - The prompt says Pilots 2 and 3 use "Root start (t=1, d=0), as the code does today", which matches **phase B**.
  - *Proposal:* Pilots 2 and 3 branch from the A-end parent and run phase B only, with a fixed budget of 600, and skip C.
  - In frozen mode, C's stage-2 exploring-start episodes would feed only the critic, so including C would mostly change the critic's data.
- **Q2 — Advantage normalization in frozen mode.** Today it is pooled over stage 1 and stage 2. If stage-2 rows are masked out of the policy loss but still normalized together with stage 1, the stage-2 advantages still set the scale of the stage-1 advantages.
  - *Proposal:* in `frozen` mode only, normalize over the stage-1 rows that enter the policy loss, keep the identical minibatch partition, and average the actor loss over the stage-1 rows of each minibatch. `joint` stays bit-identical.
  - This is a choice inside the frozen arm, so I need your call.
- **Q3 — Opponent at stage 1 in frozen mode.** §1.5 says the freeze snapshot generates both players' stage-2 actions. Stage-1 opponent actions still come from the existing lagged self-play snapshot, refreshed every 20 updates.
  - *Proposal:* keep that unchanged.
  - Confirm that "snapshot" in §1.5 means a *new* object (the freeze snapshot), distinct from `CurriculumPPO.opponent`.
- **Q4 — RNG alignment between arms.**
  - *Pilot 1:* in `expected` mode the shocks are still drawn from `rng_env`, with the same calls in the same order, so the paired arms consume identical streams; only the reward formula changes.
  - *Pilot 3:* in `mean` mode, proposal: still draw (and discard) the stage-2 Beta samples, so later learner and opponent draws stay aligned with the `stochastic` arm.
  - Please confirm both.
- **Q5 — Recovery drift in the frozen arm (Pilot 2).** The composite candidate is ê = (live actor at t=1, freeze snapshot at t=2), so its stage-2 drift is 0 by construction. I will also log the drift of the live network's own stage-2 output, as information.
- **Q6 — Expected numerical issue in a Phase 1 invariant.** Δ_t uses candidate set {grid, vertices of Q^mean, mean action, a_BR}. V^BR uses {grid, vertices of Q^BR, mean action} (`utils/dp_br_verifier.py:398-417`).
  - Because the sets differ, a Q^mean vertex can in principle exceed the BR search by a vertex-interpolation amount, so `Δmax_all ≤ Ĝmax_full` could be violated by a tiny amount.
  - Phase 1 will report the size of any such violation without clamping. It will not change the existing search.
- **Q7 — Versions.** The runner refuses to start unless it sees Python 3.12.3 / torch 2.5.1+cu121 / numpy 2.5.0.
  - The v2 entry point will run under `/home/fjiang4/tournament_experiment/.venv/bin/python`, which matches, and will keep that check.
- **Q8 — pytest.** Phase 1 needs pytest. *Proposal:* `/home/fjiang4/tournament_experiment/.venv/bin/python -m pip install pytest`, into the venv only, not system Python.
  - That venv is shared with the main checkout and other worktrees. Adding pytest does not touch numpy or torch.
  - Please approve the install.
- **Q9 — Output locations.** New outputs go to `results/v2_pilots/` and reports to `reports/v2/`, as you asked. In this repo, results are normally kept under `experiments/`, and prose reports under `docs/`. I will follow your paths unless you say otherwise.
- **Q10 — Benchmark clip.** The repo's g2 clips to [0, 100], which the prompt's formula does not. The clip is inactive at q ∈ {50, 60}.

### Commands to reproduce the audit numbers

```bash
git -C /home/fjiang4/tournament_experiment show --stat 657f54a | tail -1
```

```bash
/home/fjiang4/tournament_experiment/.venv/bin/python -c "import platform,torch,numpy;print(platform.python_version(),torch.__version__,numpy.__version__)"
```

The cohort cost/exit tally and the seed scan are the two scripts `costs.py` and `seeds.py` in this session's scratchpad. They read only the files cited above. I can commit them under `reports/v2/tools/` if you want them reproducible from the repo.

---

**STOP — waiting for approval.** Please confirm Q0 (base/path), the q and seed lists (§8), and Q1–Q4 and Q8 before Phase 1.
