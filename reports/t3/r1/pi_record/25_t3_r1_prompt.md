# Round T3-R1: the first T = 3 experiment — backward-induction training (v2.0's design at T = 3) against joint whole-game training, budget-matched and pre-registered, with the same comparison at T = 2 on the v2.0 confirmation seeds

## Where the project stands

- **Model.** Two identical players, a T-stage rank-order tournament with terminal prizes (the dynamic Lazear-Rosen model of `docs/Experiments Plan_Multi-stage.md` and `MultiStage/METHODS.md`). D2 below has the numbers.
- **T = 2 is settled.** Solver v2.0 (`protocols/v2_T2_locked_v2_0.json`, entry point `run/run_v2_T2_locked.py`) passed its fresh-seed confirmation on seeds 30501-30520: 19 of 20 runs at q = 50 and 20 of 20 at q = 60.
- **Four later rounds** tried to lower v2.0's stage-2 tie bias (MS-R1 to MS-R4, branches `ms-r1` to `ms-r4`). None met its pre-registered target. MS-R4's final commit is `db7577ea5a85910ecfc43db142e872fc1950bf75` on `origin/ms-r4`.
- **Decision at MS-R4's gate G2** (PI, 2026-10-16): option (i).
  - v2.0 stays the T = 2 solver.
  - MS-R4's two tries are reported as ablations and limitations. That write-up is a separate task and not part of this round.
  - MS-R4's control recipe `t1_st_s16` is not taken forward.
  - The seed block 40501-40520 stays reserved and unused.
  - T = 3 starts with this round.

## What this round does

1. **BW at T = 3.** The first T = 3 runs of v2.0's design, backward-induction training (BW). The last stage is trained first, on a spread of start states, and frozen. Each earlier stage is then trained against the exact expected continuation of the frozen later stages. The existing T-generic pipeline `run/run_ms_stagewise.py` does this with its legacy settings.
2. **The comparison with JT.** A pre-registered, budget-matched comparison with joint whole-game training (JT): one network for all stages, every episode simulated from stage 1 (d = 0) to stage T, and all stages updated together.
   - JT is the training scheme of the project's original multi-stage plan, `docs/Experiments Plan_Multi-stage.md` §3.3-3.4: a shared policy pi_theta(e | t, d), a terminal reward, gamma = lambda = 1.
   - It is also the scheme of related work on multi-stage contests: Pieroth, Kohring and Bichler, "Deep Reinforcement Learning for Equilibrium Computation in Multistage Auctions and Contests", Management Science (arXiv:2312.11751). That paper trains one network for all stages by self-play on whole-game utility and evaluates it ex ante.
3. **JT-ES at T = 3.** A third arm adds exploring starts to JT. It separates the coverage of states from the backward order with a frozen, exactly integrated continuation.
4. **The same comparison at T = 2**, where the closed form is known. JT is run on the v2.0 confirmation seeds 30501-30520 and paired with the existing v2.0 confirmation runs.

**Why two families of metrics.** A learner trained only on the states its own play visits is not disciplined at the other states. In the learning-in-games literature the resulting outcome is a self-confirming equilibrium (Fudenberg and Levine, Econometrica 1993). Chen et al. (IJCAI 2021, §4.2) make the same point for RL with self-play, and train "in the reverse order of rounds" instead. The comparison therefore reports two things:

- **(a) Ex ante:** the gain from the best response at the root, EXP_root.
- **(b) At every state of every stage:** Gmax_full and the per-stage maxima, split into the band the shocks alone reach and the rest (D6).

The reports state what the data show and add no claim beyond them.

**Names.** In code, tables and reports the arms are `bw`, `jt`, `jt_es`, `jt2`, and the reference is `v20`. In prose they are "backward-induction training (BW)", "joint whole-game training (JT)" and "joint whole-game training with exploring starts (JT-ES)". Do not introduce other names for methods or quantities.

## Steps

1. **P0**: housekeeping (§1).
2. **P1**: code, tests, checks, the smoke wave, the independent review, the pre-registration and the push (§2). Then **STOP at gate G1**.
3. **P2**, after my reply "proceed P2": the launch of waves `t3` (120 runs) and `t2j` (40 runs), the post-launch checks, the evaluation and the analysis (§3).
4. **P3**: reports, fact-check and push, then **STOP** (§4, §5).

## Environment and standing rules

- **Repository** `/home/fjiang4/tournament_experiment`. Python `/home/fjiang4/tournament_experiment/.venv/bin/python` only. Run tests with `pytest tests`, never with a bare `pytest`.
- **Runs** are single-threaded (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`) and run in tmux. Do not kill jobs you did not start.
- **Branch and worktree.** Work in a new worktree on a new branch `t3-r1`, created from `origin/ms-r4` at `db7577ea5a85910ecfc43db142e872fc1950bf75`, and record the worktree path.
  - Do not touch MS-R4's launch worktree `.claude/worktrees/ms-r4-e1-e2-experiments-2da78b`. It holds untracked weight exports and freeze arrays.
  - Never run `git clean` there, and leave its local `ms-r4` ref where it is (`2f618531`).
- **Paths:** results root `results/t3_r1/`; reports `reports/t3/r1/`; new tools under `tools/t3/`; new tests `tests/test_t3_*.py`.
- **Provenance:** every number in a report cites a path.
- **Closed form.** The closed-form equilibrium (e_T*, g1) enters evaluation and reporting only. It never enters training, the start sampler, the schedule, or anything a run decides. The expected terminal reward of D2 uses the shock CDF F_ξ; that is part of the game, not of the equilibrium.
- **Results.**
  - No deletion or overwriting of results or checkpoints. The paper-registry test and its data stay unchanged. `clamp_likelihood` stays `density`.
  - **Tracking follows the rules of `reports/ms/README.md`.** For `bw` runs, track what those rules track for MS runs.
  - **Joint runs: tracked.** `run_config.json`, `manifest.json`, `status.json`, `joint_run_summary.json`, `joint_checks.csv`, `joint_updates.csv` and `visitation.npz`.
  - **Never tracked:** `.pt` files, `train_history.json`, weight exports, `freeze_stage*_*.npz`, `final_eval_*.npz`, the per-run NPZ arrays of `tools/t3/t3_eval.py`, and `run.log`.
- **Frozen files.**
  - No existing file changes in `protocols/`, `agents/`, `envs/`, `utils/`, `run/`, `tools/v2/` or `tools/ms/`. Their tests pin them.
  - The only new file under `run/` is `run/run_joint.py` (D4). Everything else is new code under `tools/t3/` and new tests.
  - If the existing pipeline needs any change to run this round, stop and report before changing it.
- **Pushing** is allowed only where §2.7 and §4.3 say so. `main` and tags are not touched, and no pull request is opened.
- **Save the prompt.** This prompt is saved verbatim as `reports/t3/r1/pi_record/25_t3_r1_prompt.md`, as the first file written in the new worktree (P0.2 creates it).

**If any step fails, or anything does not match this prompt, stop at that point and report. Do not work around it.**

---

## 0. Decisions (binding)

### D1. Scope and seeds

- q ∈ {50, 60}.
- **Wave `t3`** (T = 3): seeds 50501-50520, the same 20 seeds at both q. Arms `bw`, `jt`, `jt_es`: 120 runs, paired by (q, seed).
- **Wave `t2j`** (T = 2): seeds 30501-30520 at both q, arm `jt2`: 40 runs.
  - These runs are paired by (q, seed) with the 40 v2.0 confirmation runs (reference `v20`, D9).
  - The block is reused on purpose, so that the new arm pairs with the locked solver's recorded runs. It confirms nothing new, and the reports say so.
- **Wave `smoke`:** seed 50500 at both q, all four arms at full budget (8 runs). It is used for cost and correctness only and never analysed.
- **Excluded seeds:** 40501-40520 and the MS development seeds 10501-10530 appear in no config of this round.
- **Out of scope:** no lock, no confirmation, no tuning, no T = 4, and no change of the verifier, of v2.0 or of the existing behaviour of the MS pipeline.

### D2. The game and the settings every arm shares

**The game.**

- **Effort and cost:** two identical players and T stages. Each stage a player chooses effort e ∈ [0, 100] at cost k e², with k = 1/3500.
- **Shocks:** each player draws an independent shock ε ~ U[-q, q] in every stage.
- **Prizes:** the larger final cumulative score wins w_H = 6 and the other player gets w_L = 2, so ΔW = 4.
- **State:** the learner's score gap d, with d' = d + e - e_opp + ξ, where ξ = ε - ε_opp is triangular on [-2q, 2q].
- **Domains:** B = 100 + 2q. The stage-t domain is D_t = [-(t-1)B, (t-1)B], and D_1 = {0}.
- **Observation:** [(t-1)/(T-1), d/((t-1)B)]; the d component is 0 at the root.

**Shared settings.** All of them come from the v2.0 record of q (`protocols/v2_T2_locked_v2_0.json`, `records[str(q)]`), with `game.T` set to the horizon.

- **Networks** (`agents/ppo_curriculum.py`, `agents/ppo_curriculum_v2.py`), the same class `CurriculumPPOv2` in every arm:
  - a 2-64-64 tanh Beta actor, with mean clamp(sigmoid(z0), 1e-6, 1-1e-6) and concentration 100 + softplus(z1);
  - a separate 2-64-64 tanh critic.
- **PPO:**
  - Adam with LR 3e-4 outside the decay windows of D5;
  - 10 epochs, minibatch 256, clip 0.2;
  - value coefficient 0.5, entropy 0, grad-norm clip 0.5;
  - gamma = lambda = 1;
  - advantage normalisation over all rows of an update.
- **Episodes and opponent:**
  - 512 episodes per update;
  - the lagged opponent is refreshed at entry (in the MS runner, at every phase entry) and after every update whose global number is a multiple of 20;
  - weight exports every 25 global updates.
- **Randomness:**
  - the RNG streams are `SeedSequence([seed, q, namespace])` with init 0, env_noise 1, learner_action 2, opponent_action 3, starts_roles 4 and minibatch 5;
  - the torch generator is seeded from the init stream;
  - the process-global RNG hardening is v2.0's.
- **Compute:** CPU, one thread per process.
- **Verifier tiers** (`utils/dp_br_verifier.py`):

  | tier | constant | state step | effort step | GL nodes per half |
  |---|---|---|---|---|
  | development | `DEV_CONFIG` | 4 | 1 | 16 |
  | final | `FINAL_CONFIG` | 2 | 0.5 | 32 |

- **Clamps:** action clamp 1e-6, `clamp_likelihood` "density".
- **Terminal reward,** in every arm: the conditional expectation w_L + ΔW F_ξ(d + e - e_opp) - k e² at stage T. This is v2.0's `reward_mode = expected`; the shocks are still drawn.
  - The project's first multi-stage runs (July 2026, `run/run_multi_stage.py`) used sampled terminal rewards.
  - This round keeps the reward estimator equal across arms, so that the comparison is about the training scheme. The reports state this difference from the original runs.
- **Initial weights** depend only on (q, seed). They are identical across all arms of a (q, seed) and across T (check C-INIT).

### D3. The arms

**`bw`**, wave `t3`.
- Scheme: backward induction (stage 3, freeze, stage 2, freeze, stage 1).
- Starts per update: stage 3, 512 bin-balanced on D3; stage 2, 512 bin-balanced on D2; stage 1, 512 at the root.
- Stages updated: one stage per phase.
- Continuation: exact expected continuation of the frozen later stages (nested tables, `utils/ms_continuation.py`).
- Code: `run/run_ms_stagewise.py`, unchanged.

**`jt`**, wave `t3`.
- Scheme: joint whole-game.
- Starts per update: 512 at the root.
- Stages updated: every stage, every update.
- Continuation: sampled; the episode is simulated to stage T.
- Code: `run/run_joint.py` (new, D4).

**`jt_es`**, wave `t3`.
- Scheme: joint whole-game with exploring starts.
- Starts per update: 256 at the root, 85 stage-2 starts bin-balanced on D2, and 171 stage-3 starts bin-balanced on D3.
- Stages updated: every stage an episode reaches.
- Continuation: sampled.
- Code: `run/run_joint.py`.

**`jt2`**, wave `t2j`.
- Scheme: joint whole-game at T = 2.
- Starts per update: 512 at the root.
- Stages updated: both stages.
- Continuation: sampled.
- Code: `run/run_joint.py`.

**`v20`**, the reference.
- Scheme: v2.0 as locked, T = 2.
- Starts per update: Phase A, 512 bin-balanced on D2; Phase B, 512 at the root.
- Stages updated: one stage per phase.
- Continuation: exact expected continuation.
- Code: the existing runs, read only (D9).

**`bw` in detail.** `run/run_ms_stagewise.py` with its legacy settings:

- **Starts and rule.** `start_weights = {scheme: bin_balanced}` and `rule.enabled = false`.
  - The checks every K = 25 local updates run and are recorded.
  - Their would-fire entries are a record only. Stage 3 uses the terminal-stage values of `reports/ms/r1/prereg_parameters.json` (its stage "2": eps 0.005, rho 0.05, tau 0.02). Stages 2 and 1 use its stage "1" values (eps 0.005, rho 0.03, tau 0.02).
- **Budgets:** fixed at {3: 1600, 2: 1600, 1: 600}.
- **LR windows:** 3e-4 outside the windows; linear 3e-4 -> 3e-5 over local updates 1201-1600 at stages 3 and 2, and over local 1-600 at stage 1.
- **Optional keys:** `init_digest` is true. There is no `actor_variant`, `conc_scale_schedule`, `fixed_global_sampler`, `noise_report`, `focus_scope` or `focus_map`.
- **The three phases:**
  - the stage-3 phase is v2.0's Phase A with D3 in place of D2;
  - the stage-2 phase uses the return -k e² + Ṽ_3(d + e - e_opp) of the frozen stage 3;
  - the stage-1 phase uses Ṽ_2 of the frozen stages 2 and 3.
- **The config.** Build it with `tools/ms/ms_configs.build_config(proto, q, seed, arm, out_dir, params, T=3, arm_table=...)`. With T = 3 it embeds the v2.0 record with `game.T = 3` and removes the T = 2-only record fields.
  - **Arm-table entry:** `{scheme: bin_balanced, rule_enabled: False, definition: ...}`, with no `round` key. A new round tag would need a `RUN_PREFIX` entry, and a test pins `RUN_PREFIX`.
  - **Parameters file:** this round's `reports/t3/r1/params_bw_t3.json` holds:
    - K, M, `conc_limit`, `localized_fraction`, `alpha_polish`, `ema_beta` and `near_tie_half_width`, copied from `reports/ms/r1/prereg_parameters.json` (SHA-256 `0fbfc01857c5abe339ab1926361cd57d6e988da20e754fd6dea3ffa4e076b406`);
    - `stages` {"3", "2", "1"} with the eps, rho and tau above;
    - `legacy_pipeline` with the budgets and windows above.
  - **After `build_config`,** set `cfg["pilot"] = "t3_r1"`, set `cfg["run"]` and `cfg["record"]["run"]` to `t3r1_q{q}_s{seed}_bw`, and set `cfg["init_digest"] = true`. Neither `validate_config` nor `record_sha256` (the hash of the unmodified record) depends on these fields.
  - **`base_commit`:** `build_config` writes the v2.0 protocol's base commit (`657f54a`); the joint configs carry the same value (D4). The code commit of every run is in its `manifest.json`.
- **Known difference from T = 2** (recorded, not changed). D3 has 80 bins at q = 50 and 88 at q = 60 (bin width 10), twice D2's 40 and 44. A stage-3 bin therefore gets half the expected starts per update that a stage-2 bin got in v2.0's Phase A: 512/80 = 6.4 against 512/40 = 12.8 at q = 50.

**`jt_es` start mix.** The mix is that of the September 2026 T = 3 curriculum's Phase C (`MultiStage/three_stage/T3_IMPLEMENTATION_PLAN_20260924.md` §1.2). It gives every bin of D2 and of D3 about the same expected number of direct starts per update. Every bin gets direct starts, at a lower density than in `bw` (D6 reports the counts).

### D4. The JT runner `run/run_joint.py` (new file; T ∈ {2, 3})

The runner reuses the existing modules and changes none of them. It does not import `run.run_ms_stagewise`, which imports the stop-rule machinery. Where it needs a piece of `MSRun`, it reimplements it exactly, and a test compares the two.

**Config `joint_run_config/1`.** Every key is explicit, and an unknown key is refused with exit code 4.

- **Identity:** `schema`; `base_commit` (the v2.0 protocol's `base_commit`, as `build_config` writes it); `round` "t3_r1"; `arm` ∈ {`jt`, `jt_es`, `jt2`}; `run`; `q`; `seed`.
- **`record`:** the v2.0 record of q.
  - Set `seed`, `run` and `output_dir`, and set `game.T` to the horizon.
  - At T = 3, remove the T = 2-only fields `es_bins_stage2` and `domain_half_stage2`. This is exactly the transformation `tools/ms/ms_configs.build_config` makes.
  - The record's top-level `T` stays as it is: `build_config` does not change it either. `game.T` is authoritative, and `joint.T` must equal `game.T`.
- **Provenance:** `protocol_ref` {path, sha256}; `record_sha256` (of the unmodified record, as `build_config` computes it); `threads_per_process`.
- **`joint`:**
  - `T`;
  - `updates` U;
  - `starts` {`root`, `stage2`, `stage3`}, non-negative integers that sum to the record's `protocol.episodes_per_update`, with a stage key present only up to T;
  - `lr_windows`, a list of {first, last, start, end}, contiguous, with the last ending at U;
  - `check_every`.
- **Fixed values:** `reward_mode` "expected"; `clamp_likelihood` "density"; `init_digest` true.

**Validation.** The runner validates types and ranges only: T ∈ {2, 3}, U ≥ 1, `check_every` ≥ 1, the starts as above, the windows as above, and the fixed values. This lets the tests use small budgets and a lowered `record.protocol.episodes_per_update`, as `tests/test_ms_runner.py` does for the MS runner. The production values of D5 (512 episodes, U, starts, windows, `check_every` 25) are pinned by `tools/t3/t3_configs.py` and verified by the launch checks.

**Construction** (as `MSRun.__init__`):

- the environment versions and the thread variables are checked against the record;
- `GameSpec` comes from `record.game` and `PPOConfig` from `record.ppo`; the verifier tiers must equal `DEV_CONFIG` and `FINAL_CONFIG`;
- the four numpy streams env, learn, opp and start, the minibatch stream and the torch generator are built from the namespaces;
- the agent is `CurriculumPPOv2(ppo_cfg, torch_gen, rng_minibatch, device="cpu")`;
- the init digest is SHA-256 over the actor and the critic state dicts, exactly as `MSRun._init_digest` defines it (sorted names, prefixed `actor.` / `critic.`, float32 bytes). It is written to `manifest.json` before any update. `manifest.json` also records the git commit, the clean-tree flag, the versions and the config SHA-256.

**Loop**, for global update u = 1 ... U. Before u = 1, call `agent.refresh_snapshot()` and log it as "entry".

1. **LR.** Set the LR on both optimizers before the update (`run/run_final_dp_br_round3_dense.set_lr`). Outside a window it is 3e-4. Inside a window it is computed with the expression of `MSRun.lr_linear`, start + (end - start)(u - first)/(last - first), so the floats equal the MS runner's.
2. **Starts,** from the `starts_roles` stream, in this order:
   - t0 = [1]·n_root + [2]·n_2 + [3]·n_3;
   - d0 = zeros(n_root), then `StartSampler.balanced(2, n_2, rng)` if n_2 > 0, then `StartSampler.balanced(3, n_3, rng)` if n_3 > 0;
   - then roles = `rng.integers(0, 2, size=episodes_per_update)`.
3. **Rollout:** `run.v2_rollout.collect_batch_v2(spec, agent, t0, d0, roles, rng_env, rng_learn, rng_opp, gamma=1, lam=1, bin_width=10, reward_mode="expected", frozen=None, continuation_action_mode="stochastic", cont_table=None, clamp_likelihood="density")`.
4. **Update:** `agent.update(states, actions, logp, returns, advantages)` over all rows, with no mask.
5. **Refresh and export.** If u is a multiple of 20, call `agent.refresh_snapshot()` and log it. If u is a multiple of 25, write `weights/u{u:05d}.npz` with `agent.export_weights_npz`.
6. **Development check,** if u is a multiple of `check_every`. It is pure and moves no RNG.
   - Run `utils.v2_metrics.evaluate(mean_fn, spec, DEV_CONFIG, beta_fn=beta_fn, recovery_step=0.5)` on the live actor, with the functions from `run.run_final_dp_br.make_policy_fns(agent, spec)`.
   - Catch `DomainError`, `FloatingPointError` and `ValueError` as `MSRun.verify_candidate` does, recording NaN scalars and the error.
   - Run `concentration_stats` on the development grid of every stage t ≥ 2 (`dense_grid(spec.domain_half(t), 4)`) and on {0} at t = 1.
   - Write one row of `joint_checks.csv` with:
     - u and the cumulative policy rows per stage;
     - every scalar of `evaluate`'s `scalars` dict (its key set is fixed for a given T; at T = 2 it includes closed-form columns, which are evaluation only);
     - the maximum normalised Beta standard deviation per stage;
     - the verifier seconds and the error, if any.
7. **Per-update row** in `joint_updates.csv`:
   - u, the LR, the episodes and the policy rows per stage;
   - the mean episode return and the mean effort per stage;
   - the final-epoch KL, the clip fraction, both losses, and the raw advantage mean and SD;
   - whether the snapshot was refreshed;
   - the positions of the four streams and of the minibatch stream;
   - `update_wall_sec`.

**Visitation.** For every stage t ≥ 2, accumulate per-bin counts (bin width 10) of the learner-signed pre-action gap of the stage-t rows, by start stage.

- **Exact counts.** Rows from root episodes, and rows from direct starts at t, come from `collect_batch_v2`'s `visitation` keys `stage{t}_root_path` and `stage{t}_direct_es`.
- **Approximate counts.** For `jt_es` at stage 3, the rows from stage-2 starts are counted from the decoded observation d = obs_d · (t-1)B in float64 with `envs.curriculum_env.gap_bin_index`.
  - Within the stage-t block of the batch, the rows are the episodes with t0 ≤ t, in ascending order.
  - Label this source approximate: a gap on a bin edge can fall into the neighbouring bin through the float32 observation.
- Write `visitation.npz` at the end of the run.

**End of the run.**

- **Full state:** `state_end.pt`, format `joint_full_state/1`: the agent's full state, the stream states, the torch generator, the three global RNG states, the counters, the snapshot log and the LR.
- **Final evaluation** of the live actor on `FINAL_CONFIG` and `DEV_CONFIG`, with `recovery_step` 0.5. It writes `final_eval_final.npz` and `final_eval_development.npz` (`utils.v2_metrics.save_npz`), and puts their scalars in `joint_run_summary.json`.
- **`joint_run_summary.json`** also holds:
  - updates, episodes, policy rows per stage and minibatch steps;
  - wall and CPU time by category (`train_rollout`, `train_update`, `dev_verifier`, `final_eval`);
  - peak RSS;
  - the init digest.
- **`status.json`** and the exit code.

**Global-RNG hardening,** as `run/run_ms_stagewise.py` does it (`run.run_v2_T2_locked`: `seed_globals`, `global_states`, `digests`, `RNG_VIOLATION_EXIT`):

- seed at start;
- take the reference states before update 1, after the entry refresh;
- assert at the end of training, after the final evaluation and at the end of the run;
- on a violation, record it, finish the run and exit with code 5.

**Exit codes:**

- 0: done;
- 1: exception;
- 3: refusing to restart, when any `status.json` exists in the output directory, as the MS runner refuses;
- 4: config error;
- 5: global-RNG violation.

### D5. Budgets: matched by policy rows

**The rule.** Every T = 3 arm trains on the same number of policy rows as `bw`, to within one update; `jt2` trains on as many as v2.0 at T = 2. A policy row is a learner transition whose action enters the PPO policy loss.

- Equal policy rows give equal minibatch steps for `jt` and `bw`, since each update runs 10 epochs over its rows in minibatches of 256.
- `jt_es` has more steps, because 1109 rows leave a partial minibatch.
- **v2.0's Phase B updates also carry the frozen stage-2 rows.** Their minibatches run over 1024 rows: the critic trains on all of them, and the actor steps on the stage-1 rows of each minibatch.
  - Its step count is therefore larger than its policy rows imply: 1600 × 20 + 600 × 40 = 56,000 by design.
  - Verify this from each run's `v2_updates.csv`: steps per update = `n_epochs_run` × ceil(`n_rows` / 256), with `n_rows` 512 in Phase A and 1024 in Phase B, and `n_policy_rows` 512.

| arm | updates | policy rows per update (t = 1 / 2 / 3) | policy rows (t = 1 / 2 / 3; total) | minibatch steps | LR |
|---|---|---|---|---|---|
| `bw` | 3800 (1600 + 1600 + 600) | 512 in its stage | 307,200 / 819,200 / 819,200; 1,945,600 | 76,000 | per phase (D3) |
| `jt` | 1267 | 512 / 512 / 512 | 648,704 each; 1,946,112 | 76,020 | 3e-4 to 950; linear 3e-4 -> 3e-5 over 951-1267 |
| `jt_es` | 1754 | 256 / 341 / 512 | 449,024 / 598,114 / 898,048; 1,945,186 | 87,700 | 3e-4 to 1315; linear over 1316-1754 |
| `v20` (T = 2) | 2200 (1600 + 600) | 512 in its stage | 307,200 / 819,200; 1,126,400 | 56,000 | v2.0 |
| `jt2` | 1100 | 512 / 512 | 563,200 each; 1,126,400 | 44,000 | 3e-4 to 825; linear over 826-1100 |

The decay window of each joint arm is its last 25 % of updates, rounded up.

Two caveats are reported with every comparison. Neither is separated in this round.

- **Capacity.** The `bw` policy is three networks, one frozen copy per stage. A joint arm's policy is one network for all stages, as in the original plan and in Pieroth et al.
- **Coverage.** The `jt_es` start mix covers every bin, but at a lower density per bin than `bw`.

### D6. What is measured

**The evaluated policy** is always the last iterate, as Beta means:

- `bw`: the composite of the frozen stage-3 snapshot, the frozen stage-2 snapshot and the live actor at stage 1, from `state_end_stage1.pt` (keys `frozen_stages` and `agent`).
- `jt`, `jt_es`, `jt2`: the live actor at every stage, from `state_end.pt`.
- `v20`: the `u01600` export at t = 2 and the `u02200` export at t = 1. This is v2.0's evaluated candidate.

Exports are loaded with the loader of `tools/ms/replay_dev_rule.py` (`load_export`, `build_actor`, `candidate_fns`). That loader builds a float32 `BetaActor` with a forward pass bit-identical to training, and the policy functions come from `run.run_final_dp_br.make_policy_fns`.

**Per run, on the final and the development tier.** `tools/t3/t3_eval.py` computes the quantities below with `utils.v2_metrics.evaluate` (pure) plus `tools/t3/t3_metrics.py`. Every payoff quantity is divided by ΔW.

1. **`Gmax_full_over_dw`:** the maximum over t and over the stage-t grid of G_t(d) = V_t^BR(d) - V_t^mean(d), the gain of a dynamic best response.
2. **Per-stage maxima,** for t = 1 ... T, with their argmax d:
   - `G_max_t{t}_over_dw`;
   - `Delta_max_t{t}_over_dw` (η_t), the largest one-step deviation gain at stage t.
3. **Root and summary values:** `EXP_root_over_dw` = V_1^BR(0) - V_1^mean(0); `dReach_over_dw`; `Deltamax_all_over_dw`.
4. **Band split,** at every stage t ≥ 2.
   - The band is {d on the stage-t grid : |d| < 2q(t-1)}: the open support of the summed shock differences before stage t.
   - A node at exactly |d| = 2q(t-1) is out of band.
   - Report the in-band and out-of-band maxima of Δ_t and of G_t: `Delta_in_t{t}`, `Delta_out_t{t}`, `G_in_t{t}`, `G_out_t{t}`.
5. **Terminal-stage recovery** against the closed form e_T*(d) = `utils.theory_multistage.g2_two_stage(d, q, w_h, w_l, k, e_max)`. This final-stage tent holds at the last stage of any T. Use the grid of D_T with step 0.5, as `recovery_metrics` does at T = 2.
   - `rmse_pos_T_over_eT0`: the RMSE over |d| < 2q, divided by e_T*(0).
   - `tail_mean_T_over_eT0` and `tail_max_T_over_eT0`: over |d| ≥ 2q.
   - `peak_T_signed` and `peak_T_abs`: the relative error at d = 0.
   - e_T(0).
   - At T = 2 these must equal `evaluate`'s `stage2_rmse_pos_over_g2_0`, `stage2_tail_mean_over_g2_0`, `stage2_tail_max_over_g2_0` and `stage2_peak_rel_err_signed` bit for bit (a test).
   - At T = 3 the tail set is |d| ≥ 2q within D_3: 100 to 400 per side at q = 50, against 100 to 200 at T = 2. The threshold 0.02 is v2.0's, but the set is larger. The pre-registration notes this.
6. **Stage 1:** e_1(0). At T = 2, also `stage1_rel_err_signed` and its absolute value, against g1 = ΔW/(6kq).
7. **Concentration:** the maximum normalised Beta standard deviation per stage on the development grid.
8. **Gates,** with v2.0's thresholds. Every "≤" is inclusive at full float64 precision.
   - **G-A:** η_T (final) ≤ 0.005, `rmse_pos_T_over_eT0` ≤ 0.05 and `tail_mean_T_over_eT0` ≤ 0.02.
   - **G-F:** `Gmax_full_over_dw` (final) ≤ 0.01.
   - **G-N:** |η_T(dev) - η_T(final)| ≤ 0.001 and |Gmax_full(dev) - Gmax_full(final)| ≤ 0.001.
   - **G-S** (T = 2 only): |e_1(0) - g1|/g1 ≤ 0.05.
   - **Run pass:** G-A ∧ G-F ∧ G-N at T = 3; at T = 2, also G-S (v2.0's rule).
   - For `bw`, the gate values come from the final composite. This equals v2.0's timing, because η_T and the terminal recovery depend only on the frozen stage-T policy.
9. **Best of run** (`jt`, `jt_es`, `jt2`; descriptive only).
   - **Candidates:** the development checks plus the last iterate. The last iterate's development value comes from `final_eval_development.npz`; `jt` and `jt_es` have no check at their last update.
   - **Excluded:** a check whose evaluation is invalid or raised an error.
   - **Selection:** the candidate with the smallest development `Gmax_full_over_dw`; ties go to the earliest.
   - **Evaluation:** both tiers, items 1-8.
   - **Label:** "selected with the development verifier; not a pre-specified candidate".
10. **Cost:** updates, policy rows per stage, minibatch steps, and wall and CPU time per phase and per category.
11. **Coverage.**
    - Joint arms: the visitation per stage and bin, by start stage, and the share of each stage's rows that lie in band.
    - `bw`: the expected direct starts per bin (bin-balanced), and the realised per-stratum counts the MS runner logs (`n_start_tail`, `n_start_near`, `n_start_mid` in `ms_updates.csv`).

**Curves** (descriptive), all against cumulative policy rows:

- the joint arms' development checks every 25 updates: `Gmax_full`, EXP_root, and η_t and G_max_t per stage;
- `bw`'s per-phase checks (`ms_checks_stage{t}.csv`, the stage's own `Delta`).

### D7. Criteria and decision rules (pre-registered)

**Statistics.** All of this is written into `prereg_criteria.json`, and the blind tool follows it literally.

- **Pairing:** by (q, seed). n = the number of complete pairs of that (q, hypothesis).
- **Bootstrap indices:** `numpy.random.default_rng(S).integers(0, n, size=(10000, n))`, freshly created with the same S for every (hypothesis, q, statistic). S = 20261016 at T = 3 and 20261017 at T = 2.
- **Intervals:** percentiles of the 10,000 resampled means with `numpy.percentile` (default linear method):
  - 95 %: (2.5, 97.5);
  - 98.75 % (Bonferroni over four hypotheses; descriptive): (0.625, 99.375);
  - the median's 95 % interval: the same percentiles of the 10,000 resampled medians, from the same indices.
- **Helpers:** the analysis tool may import `paired_stats`, `boot_mean_ci` and `clopper_pearson` from `tools/ms/r4_common.py` where they match this definition. The blind tool reimplements everything.
- **Failed runs:** every planned run is reported. A run with a nonzero exit (after the crash rule of §3.1) counts as failed in the gate counts. Its metrics are missing and its pair is dropped from the paired statistics; the count of dropped pairs is reported.

**Primary statistic:** `Gmax_full_over_dw` on the final tier.

| hypothesis | horizon | paired difference |
|---|---|---|
| **H1** | T = 3 | `bw` - `jt` |
| **H2** | T = 3 | `bw` - `jt_es` |
| **H3** | T = 3 | `jt_es` - `jt` |
| **H1-T2** | T = 2 | `v20` - `jt2` |

- A hypothesis is **supported** iff the upper end of the 95 % interval of the mean paired difference is strictly below 0 at q = 50 and at q = 60.
- Otherwise it is **not supported**, and the report names the q at which it fails.
- Also report the 98.75 % interval, the median paired difference with its interval, and the number of seeds with a negative difference.

**What each hypothesis reads.**

- **H1:** BW against the original joint scheme.
- **H3:** whether the coverage that exploring starts give helps the joint scheme.
- **H2:** whether BW still has the lower full-domain gain once the joint scheme has exploring starts. As D5 says, H2 compares the two schemes as implemented. It is not a coverage-matched test of the backward order.

**Secondary statistics** (descriptive). The same paired statistics for each of the four pairs:

- `EXP_root_over_dw`;
- η_t and G_max_t for every t;
- the band split at every t ≥ 2;
- `dReach_over_dw`;
- `rmse_pos_T_over_eT0`, `tail_mean_T_over_eT0` and `peak_T_abs`;
- e_1(0), and at T = 2 the stage-1 error.

Also report the gate pass counts per arm and q, with exact Clopper-Pearson 95 % intervals.

**BW at T = 3: how to read it for the next round.** This round decides nothing.

- If `bw` passes G-A ∧ G-F ∧ G-N in at least 18 of 20 runs at both q, v2.0's design at T = 3 is a candidate for a T = 3 lock. The next round would lock it and confirm it on a fresh T = 3 seed block.
- Otherwise the reports give the per-gate, per-stage diagnosis. No change is tried in this round.

**Parameter file.** `reports/t3/r1/prereg_criteria.json` is written before the launch, and the analysis tools read it. It holds:

- the gate thresholds and the band definition;
- the hypotheses and their pairs;
- the bootstrap definition above;
- the reading rule (18 of 20).

### D8. Predictions (the PI's, written before the launch; expectations, not tests; nothing is selected by them)

- **P1.** H1 and H1-T2 are supported. The joint arms' largest gains lie out of band, at the stages and gaps their own play does not reach.
- **P2.** At both q and both horizons, the absolute mean paired difference BW - JT is smaller in EXP_root than in Gmax_full.
- **P3.** H3 and H2 are both supported: `jt_es` lies between `jt` and `bw` in Gmax_full.
- **P4.** At T = 3, `bw` passes all three gates in fewer runs than v2.0 did at T = 2 (19 of 20 and 20 of 20). Any failures are at G-A, the stage-3 recovery, where the expected starts per bin are half those at T = 2 (D3).

### D9. Reference roots

Every tool takes these as explicit arguments and records them. A missing file is a stop-and-report.

- **`v20`, the v2.0 confirmation runs:** `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/confirmation_v2_0/q{50,60}/seed{30501..30520}/`.
  - Each run has `weights/u00025.npz` ... `weights/u02200.npz` (88 exports), `state_end_A.pt`, `state_end_B.pt`, `gates.json`, `v2_run_summary.json` and `v2_updates.csv`.
  - The tracked copies of `gates.json` on the branch must be byte-identical to these.
- **The September T = 3 start mix:** `MultiStage/three_stage/T3_IMPLEMENTATION_PLAN_20260924.md`, read only.
- **The original plan:** `docs/Experiments Plan_Multi-stage.md` §3.3-3.4, read only.

---

## 1. P0 — housekeeping

Write `reports/t3/r1/00_housekeeping.md` as you go: commands, hashes and verbatim outputs.

1. **Refs.**
   - Run `git ls-remote origin` for `main`, `ms-r4`, `ms-r3`, `t2-status-pack` and the tags.
   - Verify `origin/ms-r4` = `db7577ea5a85910ecfc43db142e872fc1950bf75`.
   - Record, but do not fix, local refs that are behind.
2. **Branch and worktree.**
   - Create `t3-r1` from `origin/ms-r4` in a new worktree and record its path.
   - Save this prompt there first, verbatim, as `reports/t3/r1/pi_record/25_t3_r1_prompt.md`.
   - Do not touch `main`, MS-R4's launch worktree or any other worktree.
3. **Seeds.** Create `results/t3_r1/preflight/`, then run `python tools/v2/seed_inventory.py --block 50500 50520 --out results/t3_r1/preflight/seed_inventory_50500_50520.csv`, then the same for `--block 30501 30520` into `seed_inventory_30501_30520.csv`.
   - **Exit code.** The tool exits with code 1 when it finds hits; that is expected for 30501-30520.
   - **Full listing.** The tool prints at most five sources per seed and keeps one example per seed in its CSV. List every hit file with a small script of your own, using the tool's matching rules, into `results/t3_r1/preflight/seed_hits_{block}.csv`. Record the commands and outputs.
   - **Inventory outputs are not collisions.** Seed-inventory outputs (files named `seed_inventory*`) list every seed they found. Classify them by the files they point to.
   - **50500-50520: stop-and-report** if any training run of the project's tournament lines used one of these seeds. That is any `run_config.json`, `manifest.json`, `status.json`, `gates.json` or run directory under `results/`, `experiments/` or `MultiStage/`, in any worktree. Other hits are listed and classified.
   - **30501-30520: expected classes.** Assign every hit to one of these:
     - the 40 v2.0 confirmation runs, and copies, analyses or re-evaluations of them (for example `results/v2_T2_locked/confirmation_v2_0*`, `results/ms_r1/calibration/confirmation_v2_0/`, `reports/t2_refine_100526/evidence/`);
     - the v2.0 protocol and lock declarations (`protocols/`);
     - seed-inventory outputs.
   - **30501-30520: stop-and-report** if a training run other than v2.0's confirmation used one of these seeds.
4. **Reference roots.** Verify D9 and record:
   - the 40 `v20` run directories with their export counts and `status.json`;
   - the byte-identity of each tracked `gates.json` with its reference copy.

   Seed 30510 at q = 50 is v2.0's one failed confirmation run (it failed G-A). It stays in the comparison as recorded.
5. **Machine.** Record `nproc`, load, free disk, and the Python, torch and numpy versions against the record (3.12.3, 2.5.1+cu121, 2.5.0).

## 2. P1 — code, tests, checks, smoke, review, pre-registration (on `t3-r1`)

### 2.1 Code

- **`run/run_joint.py`** (D4).
- **`tools/t3/t3_configs.py`:**
  - the `bw` config, through `build_config(..., T=3, arm_table=...)` with `reports/t3/r1/params_bw_t3.json`, then the field settings of D3;
  - the `joint_run_config/1` configs of `jt`, `jt_es` and `jt2`, with the production values of D5 pinned;
  - the smoke configs.
- **`tools/t3/t3_launch.py`,** a new launcher.
  - **Reused from `tools/ms/launch_ms_r1.py` by import, unchanged:** `Job`, `run_job`, `child_env`, `write_record`, `host_record`, `git_record`, `check_no_previous_run`, `summarize_validation` and `MAX_WORKERS`.
  - **Its own pieces,** because the existing ones assume MS runs only:
    - the command builder: `run/run_ms_stagewise.py --config ... --out-dir ...` for `bw`, `run/run_joint.py --config ... --out-dir ...` for the joint arms;
    - the pool;
    - the validation: `run.run_ms_stagewise.validate_config` for `bw`, the joint runner's validator for the others;
    - the loader for `params_bw_t3.json`.
  - **Waves:** `smoke` (8 jobs), `t3` (120) and `t2j` (40).
  - **Seed guards:** `t3` accepts exactly 50501-50520, `t2j` exactly 30501-30520 and `smoke` exactly 50500. No wave accepts 40501-40520 or 10501-10530.
  - **`--plan-csv`** writes the run name, the output directory and the config SHA-256 without launching.
  - **Run directories:**
    - `results/t3_r1/runs/q{q}/seed{s}/{arm}/` for `t3`;
    - `results/t3_r1/runs_t2/q{q}/seed{s}/jt2/` for `t2j`;
    - `results/t3_r1/smoke/q{q}/{arm}/` for the smoke wave.

    Run names are `t3r1_q{q}_s{seed}_{arm}`.
- **`tools/t3/t3_metrics.py`:** the band split, the terminal-stage recovery for any T, and the gates (D6).
- **`tools/t3/t3_eval.py`:** the evaluation of D6, on both tiers, for the three run formats (`bw`, joint, `v20`), including the best-of-run of the joint arms.
  - It is pure and runs in parallel over runs with at most 40 workers.
  - It writes `per_run.csv` and per-run NPZ arrays to `results/t3_r1/eval/`.
- **`tools/t3/t3_launch_checks.py`:** the checks of §3.2.
- **`tools/t3/t3_analysis.py`:** D7 and the figures of §3.3.
- **`tools/t3/t3_blind.py`,** independent of the analysis: it recomputes the four hypothesis rows and the gate counts from `per_run.csv` without importing `t3_analysis.py`.

### 2.2 Tests (`tests/test_t3_*.py`, plus every existing suite)

**`run/run_joint.py`**

- The config refusals and the range checks.
- The init digest equals `MSRun`'s for the same (q, seed):
  - at T = 3, against the `bw` config;
  - at T = 2, against `build_config(proto, q, seed, "MS_base", out_dir, params=<reports/ms/r1/prereg_parameters.json>)` with `init_digest` set to true;
  - and the reimplemented digest equals `MSRun._init_digest` on the same agent.
- Reduced-budget runs at T = 2 and T = 3, with the root-only mix and with the `jt_es` mix. Use, for example, U = 30, `record.protocol.episodes_per_update` lowered to 48, `check_every` 10 and one decay window. They show:
  - the exact policy rows per stage at every update;
  - an LR at every update equal to the windows, and equal to `MSRun.lr_linear` for the same window;
  - refreshes at entry and at the multiples of 20;
  - the export, check and final-evaluation files;
  - the global RNG `ok`.
- The same config run twice gives bit-identical exports and CSVs, wall-clock columns excluded.
- On a root-only batch of `collect_batch_v2` with gamma = lambda = 1, take the stored returns of one episode's consecutive stages. The return at stage t minus the return at stage t+1 equals the stage cost -k e_t², with e_t from the stored stage-t action, for t < T. The tolerance is |difference| ≤ 1e-5 · max(1, |value|), because the returns are stored in float32.

**`bw` at T = 3**

- The built config has the budgets, windows and stage parameters of D3, and the MS runner accepts it.
- A reduced-budget legacy T = 3 run completes. Use budgets {3: 40, 2: 40, 1: 20}, windows over the last half at stages 3 and 2 and over all of stage 1, and 48 episodes. It has:
  - three phases and the two nested tables;
  - `gates.json` with outcome `not_evaluated_T3`;
  - the init digest in `manifest.json`.

  This path, with the rule disabled at T = 3, has no test yet; this is its test.

**`t3_metrics`**

- On T = 2 policies, the terminal-recovery fields equal `evaluate`'s stage-2 fields bit for bit.
- The band maxima on synthetic arrays with known answers, including nodes on the band edge.

**`t3_eval`,** on the reduced runs

- Its `bw` evaluation equals the runner's `freeze_stage1_{final,development}.npz` bit for bit.
- Its joint evaluation equals `final_eval_{final,development}.npz` bit for bit.
- Its `v20` loader works on a synthetic export pair.

**Launcher**

- The seed guards, and the job counts 8 / 120 / 40.
- **`jt` against `jt_es`** of one (q, seed): the configs differ only in `arm`, `run`, `record.run`, `record.output_dir` and `joint.{updates, starts, lr_windows}`.
- **`bw` against a joint arm** of one (q, seed):
  - their embedded records are equal apart from `run` and `output_dir`;
  - `base_commit`, `protocol_ref`, `record_sha256`, `threads_per_process`, `clamp_likelihood` and `init_digest` are equal.
- **Within an arm,** configs differ only in q, seed, run and output directory, and in the fields derived from them.

**Analysis and blind tools:** synthetic per-run tables with known answers for each hypothesis, including a failed run and a missing pair.

**Full suite:** `pytest tests` passes apart from the known `test_registry_canonicalization`. Record the summary line, and list the skipped tests with their reasons.

### 2.3 Checks before the launch (each is a stop-and-report if it fails)

- **C-DIFF.**
  - Stage the files of the code commit (§2.7), then run `git diff --cached --stat db7577ea -- agents envs utils run protocols tools/v2 tools/ms`. It must list only the new `run/run_joint.py`.
  - After the commit, run `git diff --stat db7577ea <code commit> -- <the same paths>` and record both outputs.
- **C-INIT (plan).** Construct every planned run without training and record its init digest. The digests must be equal:
  - within each (q, seed), across `bw`, `jt` and `jt_es`;
  - between `jt2` and `MSRun` at T = 2, built as in §2.2, for each (q, seed) of 30501-30520.
- **C-T2R.** `tools/t3/t3_eval.py` on the 40 `v20` runs must reproduce, bit for bit, each run's `gates.json`:
  - the values `eta_final`, `eta_dev`, `rmse`, `tail`, `gmax_final`, `gmax_dev` and `s1`;
  - the G-A, G-F, G-N and G-S verdicts.

### 2.4 The smoke wave

- Launch wave `smoke` (8 runs at full budget: seed 50500, both q, all four arms) through `tools/t3/t3_launch.py`, with at most 8 workers.
- All runs must exit 0. Run C-REPLAY (§3.2) on them.
- Record the wall and CPU time per run, per phase and per category, and the peak RSS. Estimate the wall time of waves `t3` and `t2j` at 40 workers.
- The smoke outputs serve cost and correctness only. They are never analysed or reported as results.

### 2.5 Independent review

One read-only review of the diff, by an agent that did not write it. It covers:

- conformance with D2-D7;
- the JT runner against the MS runner in everything D2 says is shared: streams, LR, refresh order, exports, checks, global-RNG hardening and init;
- the budget arithmetic of D5, and C-INIT;
- that no closed-form equilibrium enters training (the expected terminal reward's F_ξ is part of the game);
- the band split, the terminal recovery, the loaders and the analysis definitions.

Findings and dispositions go in `reports/t3/r1/03_checks.md`.

### 2.6 Pre-registration

`reports/t3/r1/02_preregistration.md` contains:

- D1-D9 as implemented, including the tail-set note of D6 item 5;
- `run_config_hashes.csv`, the SHA-256 of every planned `run_config.json` (168 rows);
- per arm, the keys in which its configs differ from those of its comparator;
- `prereg_criteria.json` and its SHA-256;
- the predictions and the reference roots;
- the test summary, and C-DIFF, C-INIT and C-T2R;
- the smoke results and their cost;
- the launch plan: waves `t3` and `t2j`, 160 runs, at most 40 workers.

### 2.7 Commits, push, STOP at G1

1. **The code commit.** Commit the code, the tests, the offline outputs and reports, and the pre-registration. Later commits touch only `results/t3_r1/`, `reports/t3/`, `reports/ms/README.md` and `docs/STATE.md`.
2. **Indexes.**
   - Add `reports/t3/README.md`, with a row for T3-R1.
   - Add this note to the MS-R4 row of `reports/ms/README.md`: "G2 (2026-10-16): option (i), see `reports/t3/r1/pi_record/25_t3_r1_prompt.md`".
   - Add a dated section at the top of `docs/STATE.md`: P1 done, gate G1.
3. **Push** `t3-r1` (fast-forward, no force).
4. **STOP.** P2 starts only after my reply "proceed P2" (gate G1). My reply may add an addendum to the pre-registration; nothing else changes after it.

## 3. P2 — the runs (after gate G1)

### 3.1 Launch

- **One launch** through `tools/t3/t3_launch.py`: waves `t3` and `t2j` (160 runs), at most 40 single-threaded workers, in tmux.
- **Recorded:** `nproc`, load, free disk, HEAD, `git diff --stat <code commit> HEAD` and `git status --porcelain`. Nothing changes after the launch.
- **Crash rule:**
  - an infrastructure kill (out of memory, full disk, killed process) is re-run once, after the attempt is moved to `results/t3_r1/crashed/`;
  - any other nonzero exit counts as a failed run and is reported, not re-run.

### 3.2 Post-launch checks

Output: `results/t3_r1/launch_checks.json`. A failed deterministic check is a stop-and-report. Wall-clock columns are excluded from every bit-identity comparison.

- **Basics:** manifests at the launch commit with a clean tree; exit codes; complete files; the global-RNG assertions.
- **Budgets:** per run, the updates, episodes, policy rows per stage and minibatch steps are exactly as in D5.
- **Schedules:**
  - the LR at every update equals the windows;
  - the snapshot refreshes are at the entries and at the multiples of 20;
  - the export counts are 152 for `bw` (`u00025` ... `u03800`), 50 for `jt` (`u00025` ... `u01250`), 70 for `jt_es` (`u00025` ... `u01750`) and 44 for `jt2` (`u00025` ... `u01100`);
  - the check counts are 64 / 64 / 24 per stage (3 / 2 / 1) for `bw`, 50 for `jt`, 70 for `jt_es` and 44 for `jt2`.
- **Starts.**
  - `jt_es` has exactly 256 / 85 / 171 direct starts per update.
  - Statistical, descriptive:
    - for `jt_es`, at each stage t ∈ {2, 3} and bin b, z = (n_b - N/m)/sqrt(N (1/m)(1 - 1/m)). Here n_b is the run's total of direct starts in bin b (exact, from `stage{t}_direct_es`), N is their sum and m is the number of bins of D_t;
    - for `bw`, the same z for the run's totals of `n_start_tail`, `n_start_near` and `n_start_mid` per stage, against the bin-balanced shares of those strata;
    - report the number of |z| > 3 next to its expectation under the null, 0.0027 per cell. A flag is not a failure by itself.
- **C-INIT:** the recorded init digests are equal within each (q, seed) of `t3`, and they equal the planned digests.
- **C-REPLAY:** `tools/t3/t3_eval.py` reproduces each run's own final evaluation arrays bit for bit: `freeze_stage1_{final,development}.npz` for `bw`, and `final_eval_{final,development}.npz` for the joint arms.

### 3.3 Evaluation and analysis

**Evaluation.** Run `tools/t3/t3_eval.py` on the 160 runs and the 40 `v20` runs, into `results/t3_r1/eval/`.

**Analysis.** `tools/t3/t3_analysis.py` writes to `results/t3_r1/analysis/`.

Tables:

- the per-run table;
- `criterion.csv`: H1, H2, H3 and H1-T2 per q, with the interval, the verdict, the 98.75 % interval, the median difference and its interval, the sign count and n;
- the secondary paired tables;
- the gate counts with Clopper-Pearson intervals;
- `bw`'s T = 3 gate table, with the per-gate, per-stage diagnosis of every failure;
- the best-of-run table;
- coverage;
- cost.

Figures:

- **Effort functions:** e_t(d) per arm and stage, as the seed median with an inter-quartile band. Add the closed form at the terminal stage, and at T = 2 also g1.
- **Deviation gains:** Δ_t(d)/ΔW per arm and stage, as the seed median and the seed maximum, with the band edges marked.
- **Paired differences** with their intervals: Gmax_full, EXP_root, and the in-band and out-of-band maxima per stage.
- **Learning curves** of D6, against cumulative policy rows.
- **Visitation** per stage of the joint arms (log counts per bin), with the band edges and `bw`'s design coverage.

**Blind recomputation.** `tools/t3/t3_blind.py` recomputes the criterion table and the gate counts; the two must agree to 1e-12.

## 4. P3 — reports, fact-check, push

### 4.1 Reports (`reports/t3/r1/`)

- **`00_housekeeping.md`, `02_preregistration.md`, `03_checks.md`.**
- **`04_t3_results.md`,** in this order:
  - the checks;
  - the arms, with their configuration and code version;
  - the per-seed tables;
  - **BW at T = 3:** the gates, the per-stage values, and the diagnosis of every failure;
  - **the comparison:** H1, H2 and H3, then the secondaries;
  - where the gains are, by stage and band;
  - EXP_root against Gmax_full;
  - the best-of-run, the learning curves, coverage and cost;
  - the predictions against the outcome;
  - the caveats:
    - the budget rule;
    - the capacity difference;
    - the `jt_es` start mix (D3, D5);
    - the expected terminal reward, which the July 2026 runs did not use;
    - the larger tail set at T = 3;
    - 20 seeds per q.
- **`05_t2_comparison.md`:** the same structure for `v20` against `jt2`, with the closed-form recovery (|peak|, RMSE_pos, tail, the stage-1 error). It states that the v2.0 runs are the recorded confirmation runs, re-evaluated (C-T2R), and that this block confirms nothing new.
- **`06_decision_inputs.md`:**
  - the T = 3 reading of D7, and what it implies for the next round: a T = 3 lock and fresh-seed confirmation, or the diagnosis and the options it suggests, untested;
  - for each hypothesis, what the data support and the exact scope of the statement;
  - what is not settled: capacity, a coverage-matched order, other budgets.

  Nothing in this file decides.
- **`summary.md`.**
- **`handoff.md`,** in Chinese only, for the coworker.
  - It gives the outcome and at most five GitHub links, placed in the text.
  - It is committed after the reports, and its links point at the commit that holds the reports.
  - Nothing is sent, and no name or contact detail is written.

Every table is generated by a script under `reports/t3/r1/report_scripts/` from the CSVs, and the key figures are embedded. Update the T3-R1 row in `reports/t3/README.md`, and add a dated section at the top of `docs/STATE.md`.

### 4.2 Fact-check and tracking

- An independent read-only fact-check checks every number in the reports against the CSVs. It also checks that every D6 item and every hypothesis of D7 is covered.
- Ledger: `reports/t3/r1/pi_record/01_factcheck.md`. Corrections go in before the push.
- Commit the lightweight records under the tracking rules of the standing rules above.

### 4.3 Push

Push `t3-r1` (fast-forward, no force) once the reports and `handoff.md` are committed. Do not touch `main`, do not create tags, and open no pull request.

## 5. STOP

Stop after the §4.3 push. No lock, no confirmation, no tuning, no second wave, and no change after the pre-registration. The next round is decided by the PI with the coworker.
