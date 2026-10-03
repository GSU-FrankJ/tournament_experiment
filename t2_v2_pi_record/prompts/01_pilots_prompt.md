# Task: Implement the v2 two-stage (T=2) plan in `tournament-experiment` — verifier upgrade, stagewise frozen training, and three pilots

## 0. Ground rules (read first)

- **Repository.** `tournament-experiment` on the **vector2** server, under the `survey` directory. Before anything else, confirm the exact path (`git rev-parse --show-toplevel`), the current branch, and that the working tree is clean. Do all work on a new branch named `v2-stagewise-pilots`. Commit at the end of each phase with a descriptive message. Do not push unless asked.
- **Phased execution with hard stops.** This task has six phases (0–5). At the end of every phase, write the phase report, then **STOP and wait for my explicit approval** before starting the next phase. Do not start the next phase "to save time".
- **Provenance discipline.** Every number in any report must come from a file you produced or read (a log, CSV, NPZ, config, or code line), and you must cite that path next to the number. Never estimate, round-trip from memory, or fill gaps with plausible values. If a value cannot be determined, write `UNKNOWN` and explain why.
- **Do not change what I did not ask you to change.** Network architecture, learning rates, PPO hyperparameters, the action distribution (Beta), entropy settings, budgets, and verifier cadence all stay exactly as they are now. Each pilot changes exactly one variable.
- **Preserve old behavior.** Do not alter the behavior of the existing runner. Add the new behavior behind new flags and a separate, versioned v2 entry point. The flag defaults must reproduce the old behavior.
- **Never delete or overwrite existing results or checkpoints.** All new outputs go under `results/v2_pilots/…`, and all reports go under `reports/v2/…`.
- **Closed form is for evaluation only.** The analytical equilibrium may be used for recovery metrics and verifier calibration. It must never enter any training loss, reward, target, or stopping decision.
- **When something is unexpected, stop and ask.** This includes: the code does not match the description below, a test fails, or an invariant is violated. Do not silently patch around it.
- **Compute etiquette on vector2.** Check `nvidia-smi` before launching anything, and never kill jobs you did not start. Launch long runs so that they survive disconnects (use the repo's existing launch mechanism if there is one, otherwise `tmux`/`nohup`), log to files, and report how to monitor and resume them.

## 1. Background

### 1.1 Goal

Improve the accuracy of the two-stage (T=2) tournament experiment on two fronts:
- policy recovery against the closed-form equilibrium;
- strategic verification.

T=2 can then serve as the algorithm-development and calibration environment. The three-stage (T=3) experiments come later and are **out of scope** for this task.

### 1.2 Model and notation

- **State and policy.** The state is `s_t = (t, d_t)`: stage `t` and the current score gap `d_t` from the player's own perspective. Policies are shared and symmetric: both players use the same function, so at state `d` the opponent plays `ê_t(−d)`.
- **Effort range.** Effort lies in `[0, 100]`.
- **Candidate policy `ê`.** For evaluation, `ê` is whatever deterministic mapping the repo currently evaluates (presumably the Beta mean). Confirm this in Phase 0 and keep the existing convention.
- **Closed-form T=2 benchmark** (in the current parameterization):
  - `e1*(0) = ΔW / (6 k q)`
  - `e2*(d) = ΔW / (2 k) · f_ξ(d)`, where `f_ξ` is the density of the terminal performance-shock difference. Under the uniform shocks it is a triangular density, so `e2*` is a hump that peaks at `d = 0`, falls linearly on both sides, and is exactly zero in the theoretical tails.
  - Use the repo's own implementation of this benchmark if one exists. If not, implement it from the environment's actual noise model, and in Phase 1 verify numerically that it matches the environment.

### 1.3 Deviation quantities

For a frozen candidate `ê`:
- **One-step deviation gain:** `Δ_t(d) = max_{e∈[0,100]} Q_t^ê(d, e) − Q_t^ê(d, ê_t(d))`. Only stage `t`'s action changes; play then returns to `ê`.
- **Full dynamic deviation gain:** `G_t(d) = V_t^BR(d) − V_t^ê(d)`. The deviator re-optimizes at stage `t` and at every later stage, while the opponent keeps `ê`.
- At the terminal stage: `G_2(d) = Δ_2(d)`.
- **Stage-2 normalized residual:** `η_2 := max_{d∈D_2} Δ_2(d) / ΔW`.

### 1.4 Method change (a): full-domain, grid-based MPE metric — the new primary metric

```
Ĝmax_full = max_t max_{d∈D_t} [ V_t^BR(d) − V_t^ê(d) ]        reported as  Ĝmax_full / ΔW   (target: ≤ ε_MPE)
```

`D_t` is the full feasible state grid at stage `t`: every state reachable from the root under **any** action profile in `[0,100]²` together with the full shock support. It is not limited to states the candidate actually visits.

At every verifier checkpoint during training, keep logging all five metrics:

| Metric | State range | Deviation type | Aggregation | Role |
|---|---|---|---|---|
| EXP_root | root d₁ = 0 | full dynamic from root | single root value gap | supplementary main-path diagnostic |
| dReach | BR-reachable R_t | one-step at each state | sum over stages of per-stage max | retained reachable diagnostic |
| Δmax_all | full D_t | one-step | max over all stages/states | local full-domain diagnostic |
| Ĝmax_full | full D_t | full dynamic from that state | max over all stages/states | **primary MPE-oriented metric** |
| dFull | full D_t | one-step | sum over stages of per-stage full-domain max | conservative cumulative diagnostic |

### 1.5 Method change (b): joint training → stagewise backward learning with frozen continuation

`learn stage T → freeze → learn stage T−1 → freeze → … → learn stage 1`

- Once a later stage's policy is frozen, it still acts in rollouts and still determines the continuation payoff.
- The optimizer must no longer be able to change it.
- **Implementation decision (already made): frozen snapshot.** Keep the existing network. At freeze time, deep-copy the **complete** mapping that produces stage-2 actions:
  - all actor parameters, including any shared trunk;
  - any observation/feature normalization statistics;
  - anything else the stage-2 action depends on.
- **The snapshot** is set to eval mode, has `requires_grad=False`, and is referenced by no optimizer. In frozen mode, the snapshot generates both players' stage-2 actions.
- **The live network** is updated only through stage-1 policy-loss terms. Stage-2 transitions are masked out of the policy loss. The critic/value loss stays as it is now: it may use full trajectory returns and stage-2 states.

## 2. Phases

### Phase 0 — Read-only audit (no code changes)

Read the code and existing results, then write `reports/v2/phase0_audit.md`. Every item below must cite the file path and the function or line it comes from.

**1. Code map.** Locate:
- the T=2 runner(s) and entry point(s);
- the environment, its state/observation encoding, and the noise model (distribution and support);
- terminal reward computation and win/lose sign convention;
- the ActorCritic architecture (shared trunk? stage input?);
- the PPO update, including how stage-specific samples enter the policy loss;
- the backward curriculum / phase structure;
- checkpoint save/load;
- the verifier: DP-BR, dReach, EXP_root, grids, action search, interpolation, reach mask;
- the existing analytical-benchmark code, if any.

**2. As-run configuration.** List the values actually used by the latest T=2 runs, with the source of each value (config file, CLI, or code default): ΔW, W_L, W_H, k, q values, action bounds, grid definitions and resolutions, action-search resolution, verifier cadence (first check / check every), per-phase budgets, PPO hyperparameters, entropy schedule, Beta parameterization and clipping. If a config file and a code default disagree, report both.

**3. Stopping logic.** Describe every early-exit or stop rule in the current T=2 runner (gates, exploitability streaks, `k_stop`, and so on), and say when each one fires.

**4. Joint-training semantics.** During the stage-1 phase, do stage-2 samples contribute to the policy loss today? Which stage-2 action mode is used in rollouts: stochastic sample or mean?

**5. Start-state distribution.** For the stage-2-only (terminal) phase, how are starting states drawn? Does that distribution cover the full `D_2` as defined in §1.4? **Do not change it.** Just report it, and flag any coverage gap.

**6. Evaluation convention.** Which mapping is evaluated as `ê` (Beta mean, mode, or something else), and how is it computed?

**7. Checkpoint completeness.** Does a checkpoint contain the optimizer state, all RNG states (python / numpy / torch / CUDA), normalizer statistics, LR/entropy schedule positions, and update counters? List what is missing.

**8. Seeds and q values.** List the seeds and q values used in existing T=2 runs. The pilots use **2 q values × 3 paired seeds**:
- Propose the 2 q values, taken from those already used for T=2.
- Propose 3 new diagnostic seeds that are disjoint from all previously used seeds.

I will confirm both lists.

**9. Launch, monitoring, and cost.** How are runs launched and monitored on vector2? What wall-clock time and GPU usage does one T=2 run of each existing phase take, according to existing logs?

**STOP.** Wait for my approval.

### Phase 1 — Verifier and metrics upgrade, with tests and calibration

**1. Full-domain dynamic BR on the grid.**
- For every stage `t` and every grid point `d ∈ D_t`, compute `V_t^BR(d)`, `V_t^ê(d)`, `G_t(d)`, and `Δ_t(d)` against a frozen candidate.
- Save all of these arrays.
- Also save the per-stage maxima and their argmax locations `(t*, d*)`.
- Compute `Ĝmax_full/ΔW`, `Δmax_all`, `dFull`, `η_2`, and keep `dReach` and `EXP_root`.
- Reuse the existing verifier machinery where possible. Report exactly how `D_t` is constructed and confirm that it is closed under all feasible actions plus the shock support (no extrapolation off-grid; report any).

**2. Recovery metrics against the closed form** (evaluation only):
- Stage-1 relative error: `(ê1(0) − e1*(0)) / e1*(0)`, signed.
- Stage-2 peak relative error at `d = 0`, signed. Make sure 0 is a grid point.
- Stage-2 RMSE over the theoretical positive-effort region (the support of `f_ξ`). Report it both raw and divided by `e2*(0)`.
- Stage-2 tail mean effort and tail max effort, over the theoretical zero-effort region within `D_2`.
- Terminal symmetry error: `max_d |ê2(d) − ê2(−d)|`, reported both raw and divided by `e2*(0)`.

**3. Policy-distribution logging.** At every verifier checkpoint, for each stage and on the evaluation grid, save the Beta `α(d)`, `β(d)`, the mean, and the action standard deviation `σ(d)` in the `[0,100]` action units.

**4. Storage.**
- Per checkpoint: one NPZ with all arrays above.
- Per run: one CSV row per checkpoint with all scalar metrics.
- Per run: a final-checkpoint plot of `ê2(d)` and `e2*(d)` overlaid, plus `σ2(d)`.

**5. Invariant tests.** These are pytest tests, and the checks are also run inside the verifier on every call, with results logged:
- `G_2(d) == Δ_2(d)` on the whole grid;
- `Δmax_all ≤ Ĝmax_full ≤ dFull`;
- `EXP_root == G_1(0)`;
- `EXP_root ≤ dReach ≤ dFull`. The first inequality requires `R_t` to be the support of the root BR path. If the repo defines `R_t` differently, report the definition and the result. Do not change `dReach`.

Report the maximum violation of each relation in units of `ΔW`. Never clamp or hide a violation.

**6. Benchmark consistency.** Check numerically that the closed-form benchmark matches the environment:
- Monte Carlo the terminal win probability on a grid of `(d, e_i, e_j)` and compare it with the analytical CDF used by the benchmark.
- Report the maximum deviation relative to the MC standard error.

**7. Verifier calibration.**
- **Analytical equilibrium.** Run the verifier on `ê = (e1*, e2*)`. Report `Ĝmax_full/ΔW`, `η_2`, `EXP_root`, `dReach`, `Δmax_all`, and `dFull`; this is the numerical floor. Repeat on a grid that is 2× finer in both state and action, and report the differences.
- **Zero-effort policy.** Run the verifier on `ê ≡ 0` at both stages and report the same metrics. This checks that the metric has discriminating power.

**STOP.** Write `reports/v2/phase1_verifier.md`, then wait for approval.

### Phase 2 — Training infrastructure (no pilot runs yet)

All new behavior is controlled by flags on a new versioned v2 entry point. Each run writes a manifest JSON containing:
- the git commit and a dirty flag;
- the full resolved config;
- all seeds;
- `reward_mode`, `stage2_update_mode`, and `continuation_action_mode`;
- the parent checkpoint path and hash (for branches);
- the grid definitions;
- the verifier cadence.

The v2 entry point refuses to start if a required field is missing.

**1. `reward_mode ∈ {sampled, expected}`.**
- `sampled` is the existing terminal reward: draw the performance shocks and pay `W_H` or `W_L` according to the realized outcome.
- `expected` is the conditional expected terminal reward, given the state and both players' already-drawn efforts, under the known shock distribution:
  ```
  r̄_{i,T} = W_L + ΔW · F_ξ(d_T + e_{i,T} − e_{j,T}) − k · e_{i,T}²,    so that   E[r_{i,T} | state, actions] = r̄_{i,T}
  ```
- `F_ξ` and the sign convention must match the environment exactly. This only integrates out the prize-sampling noise of the final shock. Nothing else changes: non-terminal stage rewards, cost terms, and every other setting stay as they are.
- Unit test: for fixed `(d, e_i, e_j)` on a grid, the mean of many `sampled` rewards matches `r̄` within MC error, for both players.

**2. Stage-2-only training mode.** This uses the existing terminal phase as-is (same start-state distribution, budget, and so on), with stage 1 not trained.

**3. `stage2_update_mode ∈ {joint, frozen}`**, used while training stage 1.
- `joint` reproduces the current behavior exactly.
- `frozen` uses the frozen snapshot described in §1.5.
- Add an **output-drift test** on a fixed `D_2` grid: compare the snapshot's stage-2 mapping (mean, `α`, `β`) at freeze time with the same mapping after training, both the max-abs difference and the full array.
  - In `frozen` mode it must equal 0, up to floating point.
  - In `joint` mode, log the drift of the live network's stage-2 mapping.

**4. `continuation_action_mode ∈ {stochastic, mean}`.** This applies to the frozen stage-2 actions of **both** players during stage-1 training.
- `stochastic` samples from the frozen Beta policy.
- `mean` uses the Beta mean deterministically.
- Performance shocks are unaffected.

**5. Fixed-budget mode** (used by all pilots):
- No early exit: every arm runs the same number of updates, equal to the existing budget of that phase.
- Record the update at which each existing stop rule *would* have fired, but do not stop.
- The verifier runs at the existing cadence, and every checkpoint is logged.

**6. Exact branching.** A branch starts from a saved checkpoint and restores the complete state: model, optimizer, all RNG states, normalizer statistics, schedule positions, and counters. Both arms of a pair restore the same parent and use identical seeds from that point on.
- Test: two branches with identical flags produce identical metrics for N updates. If GPU nondeterminism makes this impossible, report the size of the discrepancy and which ops cause it.

**7. Regression test.** With all new flags at their defaults, the v2 entry point reproduces the old runner on a short run with the same seed: identical metrics, or a documented, explained tolerance.

**STOP.** Write `reports/v2/phase2_infra.md` (with test results), then wait for approval.

### Phase 3 — Pilot 1: terminal reward estimator (T=2, stage-2-only)

- **Question.** Does removing terminal prize-sampling noise improve the accuracy of the stage-2 effort policy?
- **Design.** 2 q × 3 paired seeds × 2 arms (`sampled` vs `expected`) = 12 stage-2-only runs. Fixed budget. Everything else is identical. "Paired" means that both arms of a (q, seed) pair use the same seed for initialization and environment RNG.
- **Per-checkpoint metrics.**
  - Stage-2 peak relative error, RMSE (raw and normalized), tail mean and max effort, `η_2`, and symmetry error.
  - A summary of `σ2(d)`: the value at `d = 0` and the mean over the positive region.
  - The existing KL and clip-fraction diagnostics, and wall-clock time.
  - The five metrics of §1.4 where they are defined. In this pilot stage 1 is untrained, so label every full-policy metric `stage1_untrained`.
- **Save each run's final Stage-2 checkpoint** with complete branching state, indexed by (q, seed, reward_mode). These checkpoints are the parents for Pilots 2 and 3.
- **Report.** Write `reports/v2/pilot1_reward_estimator.md` containing:
  - per-run tables at the final checkpoint;
  - paired differences (`expected − sampled`) per (q, seed);
  - learning curves of the key metrics.

  Describe what the data show. **Do not choose the estimator.**

**STOP.** I will choose the reward estimator used for development.

### Phase 4 — Pilot 2: joint update vs frozen Stage 2 (stage-1 training)

- **Question.** Does frozen continuation prevent stage-1 training from damaging the stage 2 that was already learned?
- **Design.** Take the 6 Pilot-1 Stage-2 checkpoints (2 q × 3 seeds) produced with the estimator I chose, and branch each one into two arms:
  - **Branch A (`joint`):** while training stage 1, both stage 1 and stage 2 keep updating. This is the existing behavior.
  - **Branch B (`frozen`):** stage 2 is a frozen snapshot, and only stage 1 is updated.
- **Fixed settings.** Fixed budget (the existing stage-1 phase budget). Root start `(t=1, d=0)`, as the code does today. The continuation action mode stays at the existing default. Everything else is identical.
- **Compare**, per checkpoint and at the end:
  - Stage-2 recovery drift relative to the parent checkpoint;
  - Stage-1 recovery (relative error against `e1*(0)`);
  - `Ĝmax_full/ΔW` and its argmax location;
  - `dReach` and `EXP_root`;
  - frozen-stage output drift (must be 0 in B).
- **On-path / off-path decomposition.** Split the Stage-2 drift and `Δ_2` into:
  - **On-path:** grid states with positive probability under the candidate's own stage-2 state distribution from the root (use the verifier's forward PMF). Report both the PMF-weighted and the unweighted values.
  - **Off-path:** the rest of `D_2`.

  State the exact definition in the report.
- **Report.** Write `reports/v2/pilot2_freeze.md` with paired comparisons (B − A) per (q, seed) and learning curves.

**STOP.** Wait for approval.

### Phase 5 — Pilot 3: continuation action mode (stochastic vs mean)

- **Question.** With Stage 2 already frozen, should stage-1 training use sampled stage-2 actions or the stage-2 policy mean, judged by Stage-1 accuracy and stability?
- **Design.** Use the same 6 parent Stage-2 checkpoints (chosen estimator). Stage 2 is frozen in both arms. Branch each checkpoint into `stochastic` (sample from the frozen Beta) vs `mean` (use the frozen Beta mean). Fixed budget, everything else identical.
- **Compare:**
  - Stage-1 relative error, final and over the course of training;
  - Stage-1 stability: across seeds, and the dispersion of `ê1(0)` over the last checkpoints, reporting how many checkpoints you used;
  - `Ĝmax_full/ΔW`;
  - `EXP_root` and `dReach`;
  - the stage-1 `σ1(0)`.
- **Evaluation object.** The verifier evaluates the deterministic mapping `ê` in both arms. Note this explicitly in the report.
- **Report.** Write `reports/v2/pilot3_continuation_mode.md`, plus a short `reports/v2/summary.md` that collects the three pilots' key tables with source paths.

**STOP.** Do not launch any formal T=2 or T=3 runs. I will decide the next step from the pilot results.

## 3. Report format (all phases)

Each report covers:
1. what was done (files changed, with a one-line reason each);
2. tests run and their outcomes;
3. results tables, with a source path for every number;
4. anomalies, deviations from this prompt, and open questions;
5. exact commands to reproduce.

Keep interpretation separate from results, and label it as interpretation.
