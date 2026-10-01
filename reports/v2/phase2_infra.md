# Phase 2 — Training infrastructure (no pilot runs)

Date: 2026-10-01. Repository `/home/fjiang4/tournament_experiment`, worktree `.claude/worktrees/tournament-v2-stagewise-pilots-f97609`, branch `v2-stagewise-pilots`.

| Item | Value |
|---|---|
| **Base commit** | `657f54a` |
| Phase 2 code commit | `8f2840e` |
| Regression and smoke outputs | `f5347d1` |
| Opening checks | `ca5b148`, `53a2fd5`, `f681bfa` (report: `reports/v2/phase2_opening_checks.md`) |

Path conventions:
- `SMK/` means `results/v2_pilots/_smoke/`.
- `REG2/` means `results/v2_pilots/phase2_regression/`.
- `REG/` means `results/v2_pilots/phase1_regression/`.

The smoke outputs are flow checks, not pilot results; they are not interpreted here.

---

## 1. What was done

| File | Change | Reason |
|---|---|---|
| `agents/ppo_curriculum_v2.py` | new | `CurriculumPPOv2(CurriculumPPO)` (details below) |
| `run/v2_rollout.py` | new | `collect_batch_v2`, a copy of `collect_batch` plus the flags (details below) |
| `run/run_v2_stagewise.py` | new | The v2 entry point (details below) |
| `tools/v2/launch_pilot.py` | new | C8 launcher (details below) |
| `tools/v2/rng_alignment.py` | new | Stream-by-stream RNG alignment table for the arm pairs (test 3b.5) |
| `tools/v2/smoke_checklist.py` | new | Artifact and log checklist over run directories |
| `tests/test_v2_infra.py` | new | 27 tests (§2) |

**`agents/ppo_curriculum_v2.py`** — `CurriculumPPOv2(CurriculumPPO)`:
- With both masks `None`, `update` calls the original `update` untouched (joint mode).
- The masked update adds:
  - advantage statistics over `norm_mask` rows;
  - the same per-epoch `rng_mb.permutation(n)` over **all** rows;
  - an actor loss over the policy rows of each minibatch, where the other rows never enter the graph;
  - an unchanged critic step over all rows.
- `freeze_stage2_snapshot()`, `full_state()` and `load_full_state()`.

**`run/v2_rollout.py`** — `collect_batch_v2`, a copy of `collect_batch` plus:
- `frozen`: the snapshot plays **both** players' stage-2 actions;
- `continuation_action_mode=mean`: the draws are made and discarded, and both players execute the Beta mean;
- `reward_mode=expected`: the shocks are still drawn, and r̄ = w_l + ΔW·F_ξ(d + e_own − e_opp) − k e_own² is used;
- a `stage` label on every row.

The diff against `collect_batch` consists only of `if` branches that do not execute with default flags (reproducible with `diff`).

**`run/run_v2_stagewise.py`** — the v2 entry point. It is a port of the `run_final_dp_br_round3_dense.py` loop with:
- modes `full`, `phase_A` and `phase_B`;
- `fixed_budget`;
- the 4 flags;
- full-state checkpoints;
- per-checkpoint v2 NPZ/CSV, `v2_updates.csv` (C3), per-phase timing and the would-have-fired records;
- the C5 drift test;
- a manifest that refuses missing fields.

**`tools/v2/launch_pilot.py`** — the C8 launcher:
- Phase A, or Phase B from a named parent;
- layout `results/v2_pilots/<pilot>/q<q>/seed<seed>/<arm>/`;
- bounded process pool, threads pinned to 1;
- a batch record per launch.

**Not modified:** any file that existed at `657f54a`.

### Design points needed to read the results

1. **Embedded record.** Each pilot run config embeds the archived as-run record of the final T=2 protocol, with only `run`, `seed` and `output_dir` replaced:
   - q=50: `experiments/two_stage_q50_precision_20260924/manifest.json`, record `tel_q50_s10231`;
   - q=60: `experiments/two_stage_confirmation_T2_20260922/manifest.json`, its first q=60 record.

   Together with the checks inherited from the existing runner, this guarantees the Phase 0 §2 settings: PPO, LR, Beta, verifier tiers and cadence.
2. **Manifest.** The runner refuses to start on:
   - a missing field among `schema, base_commit, pilot, arm, run, q, seed, mode, fixed_budget, flags, parent_checkpoint, parent_sha256, record, threads_per_process, budget_overrides`;
   - a flag set that is not exactly the four flags;
   - `joint + stage1_rows`, `joint + mean`, `frozen` outside `phase_B`, or `stage1_rows` in `phase_A`;
   - a non-default flag set or `fixed_budget` in `full` mode;
   - a missing parent, or a parent sha256 mismatch, in `phase_B`;
   - Python/torch/numpy version or thread-environment mismatches (inherited).

   It writes `manifest.json` with:
   - git commit and dirty flag, where dirty means tracked changes or untracked files outside `results/`;
   - base commit;
   - the resolved protocol and config;
   - seeds and their `SeedSequence` namespaces;
   - all 4 flags;
   - parent path and sha256;
   - grid definitions (point counts and steps for each tier, the recovery step);
   - verifier cadence.
3. **Full-state checkpoint (C1).** `state_end_<phase>.pt` holds:
   - actor, critic, lagged opponent and frozen snapshot (if any);
   - both Adam states;
   - `snapshot_refreshes`;
   - the four numpy streams (env, learn, opp, start) and the minibatch stream;
   - the torch generator state, the torch/numpy global states and the python `random` state;
   - counters: global update, episodes, transitions, and the next snapshot-refresh update;
   - the snapshot log;
   - schedule positions (constant LR, entropy 0, no schedule);
   - `normalizer_statistics: null`, because the observation encoding is a fixed function.
4. **Frozen candidate.** In frozen mode the evaluated ê is the live actor at t=1 and the snapshot at t=2. This applies to the verifier, the stability drift and the concentration checks.
5. **What differs in frozen mode beyond the policy loss:**
   - KL and clip fraction in the diagnostics are computed over stage-1 rows only.
   - The stability-triggered verifier calls use that KL, so verifier call positions can differ between the joint and frozen arms. This does not affect training, because the verifier is pure.
6. **Fixed budget.** `v2_run_summary.json -> would_have_fired[phase]` records `{global_update, local_update, rule}` for the first update at which the existing rule (k_phase=3 for A and B) would have stopped the phase, or `null`. The verifier keeps the existing cadence.

---

## 2. Test results

Command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_v2_verifier.py tests/test_v2_infra.py -q -p no:cacheprovider
```

Result at `8f2840e`: **48 passed, 2 xfailed, 67.1 s.** The 2 xfails are the strict on-path test from opening check 1c.

| Requirement | Test(s) in `tests/test_v2_infra.py` | Result |
|---|---|---|
| Manifest refuses missing / invalid fields | `test_config_refusals` (7 cases), `test_parent_hash_mismatch_refused` | pass |
| **C1** full-state restore | `test_continuous_equals_branched`: A then B in one process vs A, save, restore, B. B history, B verifier calls, final actor and every RNG stream are identical. | pass (bit-exact) |
| **3b.6 / C1** branch reproducibility, ≥ 20 updates | `test_branch_reproducibility[A_joint, B1_frozen_allnorm, B2_frozen_s1norm]`: two branches per arm from one parent, 20 B updates. `train_history` (history and verifier calls), `v2_updates.csv`, `v2_checkpoints.csv` and final weights are identical. | pass (bit-exact, CPU) |
| **3b.1** gradient isolation, both frozen arms | `test_gradient_isolation[B1, B2]`: ∂loss/∂adv is exactly 0 on stage-2 rows and nonzero on stage-1 rows. Perturbing stage-2 advantages leaves the parameter gradients bit-identical; perturbing stage-1 advantages changes them. | pass |
| Isolation of the full update (B2) | `test_b2_update_ignores_stage2_advantages_end_to_end`: perturbing raw stage-2 advantages leaves the updated actor bit-identical. In B1 the same perturbation does change the update, through the shared mean/std. This is by design and is documented for Pilot 2. | pass |
| **3b.2 / C5** snapshot immutability | `test_snapshot_immutable[B1, B2]`: after 20 updates, `drift_test.json` passes. Snapshot tensors are bit-identical to the parent actor. Stage-2 mean/α/β at the end equal their values at freeze time on the dev D₂ grid. The candidate's stage-2 drift is 0 at every checkpoint. | pass |
| C5 joint arm | `test_joint_logs_live_drift`: drift is logged, and the on/off-path split columns are present | pass |
| **3b.3** normalization scope | `test_norm_scope[B1, B2]`: the mean/std the update uses equal a direct recomputation over all rows (B1) or stage-1 rows (B2), exactly. Row counts are checked. | pass |
| **3b.4** joint unchanged | `test_joint_update_bit_identical_to_original`: v2 joint update vs `CurriculumPPO.update` on the same batch and minibatch-RNG state. Diagnostics (including policy loss) and resulting actor/critic are identical. | pass |
| **C2** expected reward | `test_expected_reward_matches_sampled_mean[q50, q60]`: 15 (d, e_i, e_j) points × 2 players, 200k draws each. The sampled mean equals r̄ within 5 SE; constant-reward cells match within 1e−12. | pass |
| C2 RNG after a batch | `test_expected_mode_preserves_rng_after_one_batch` | pass |
| **3b.5 / C4** RNG alignment | `test_rng_alignment_after_one_batch`; `test_rng_alignment_after_updates_fixed_count_streams` | pass (see §2.1) |
| Source of possible misalignment | `test_numpy_beta_consumption_depends_on_parameters` | pass |
| **C7** regression | Script check, §3 | bit-exact |

### 2.1 RNG alignment table

Source: `SMK/rng_alignment.json`, from `tools/v2/rng_alignment.py` on the smoke parent `SMK/smoke_A/q50/seed10501/sampled/state_end_A.pt`, with 20 updates.

| Pair | After 1 batch | After 20 updates |
|---|---|---|
| sampled vs expected (Phase A from init) | all 5 streams aligned | all 5 aligned |
| stochastic vs mean (frozen B1, Phase B) | all aligned | all aligned |
| A joint vs B1 | all aligned | all aligned |
| A joint vs B2 | all aligned | all aligned |
| B1 vs B2 | all aligned | all aligned |

**Limitation; the source is identified, not hypothesized.**
- The env, start and minibatch streams consume a fixed number of draws per update, so they stay aligned for any run length. The test asserts this.
- The learner and opponent action streams use `numpy.random.Generator.beta`, which is rejection-based (gamma variates). How many underlying draws it consumes depends on the Beta parameters. In 20 random parameter sets, at least one changed the stream state (test above); a session probe found 85 of 200.
- Once two arms' policies differ, a single different rejection decision desynchronizes `learn`/`opp` from that point on.
- No desynchronization occurred within 20 updates for any pair. Over 400–600 updates it is possible, and I cannot rule it out without running.
- Exact alignment for any horizon would require a different sampler, for example inverse-CDF on a fixed uniform per draw. That would break C7 against the existing runner, so I did not do it.
- Option, not implemented: log the four stream positions per update in `v2_updates.csv`, so the pilot reports can state when, if ever, a pair desynchronized.

---

## 3. C7 regression against the existing runner

**Configuration.**
- `REG2/run_config.json`: mode `full`, all flags at default, `fixed_budget=false`.
- Same record and smoke budget as the Phase 1 "before" run of `run/run_final_dp_br_round3_dense.py`: q=50, seed 10501; caps A/B/C 40/40/40; warm-up 10; stability every 5; timeout 10; direct rollout 2000 × 1.
- The run was executed at `8f2840e` with a clean tree (`REG2/v2_full/manifest.json` → `"dirty": false`).

`tools/v2/compare_runs.py REG/before/FINAL_A400_B25_C25/tel_q50_s10501 REG2/v2_full` → `REG2/compare_vs_existing_runner.txt`:

| Compared | Result |
|---|---|
| `train_history.json` (every leaf except `time_sec`) | 0 differences |
| `final_eval.json` (every leaf except timing and path keys) | 0 differences |
| `arrays.npz` (90), `checkpoint_weights.npz` (12), `phase_A_exit_arrays.npz` (41), `phase_B_exit_arrays.npz` (41) | 0 differ |
| `checkpoint.pt`: actor, critic, opponent and both Adam states | 0 differing tensors |
| stdout, except the `[run]`/`[cfg]`/`[done]` lines | identical |

**Overall: IDENTICAL, bit-exact on CPU.**

An earlier attempt differed only in two extra cost-counter keys (`v2_outputs`, `final_eval_v2`) in `final_eval.json`. They were moved to a separate `costs_v2` record before the commit.

---

## 4. Smoke runs (3c)

### Runs

- **Parent:** `SMK/smoke_A/q50/seed10501/sampled/state_end_A.pt` (Phase A, 40 updates, q=50, seed 10501).
- **Phase B:** 20 updates from that parent with `reward_mode=sampled`, for A_joint, B1_frozen_allnorm, B2_frozen_s1norm and B1_frozen_allnorm + mean.
- **Phase A:** also run with `reward_mode=expected`.
- **Overrides** (recorded in each manifest): caps A 40, B 20; warm-up 10; stability every 5; timeout 5.
- **Launch:** `tools/v2/launch_pilot.py` with 2 and 4 workers. All 6 runs ended with returncode 0 (`SMK/smoke_A/launch_*.json`, `SMK/smoke_B/launch_*.json`).

### Checklist

Source: `SMK/smoke_checklist.csv`, from `tools/v2/smoke_checklist.py`.

| Arm | Status | Commit / dirty | Updates | Checkpoint rows / NPZ | C3 adv stats logged | NPZ has G, PMF, σ, α, β | Plot | Full-state checkpoint | Would-have-fired | stage1 label | Drift test | Norm rows used |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sampled (A) | done | 8f2840e / false | 40 | 7 / 7 | yes | yes | yes | `state_end_A.pt` | null | stage1_untrained | — | — |
| expected (A) | done | 8f2840e / false | 40 | 7 / 7 | yes | yes | yes | `state_end_A.pt` | null | stage1_untrained | — | — |
| A_joint | done | 8f2840e / false | 20 | 3 / 3 | yes | yes | yes | `state_end_B.pt` | null | trained | logged (joint) | all rows |
| B1_frozen_allnorm | done | 8f2840e / false | 20 | 3 / 3 | yes | yes | yes | `state_end_B.pt` | null | trained | **pass** (mean/α/β Δ = 0) | all rows |
| B2_frozen_s1norm | done | 8f2840e / false | 20 | 3 / 3 | yes | yes | yes | `state_end_B.pt` | null | trained | **pass** | stage-1 rows |
| B1 + mean | done | 8f2840e / false | 20 | 3 / 3 | yes | yes | yes | `state_end_B.pt` | null | trained | **pass** | all rows |

### C3 logging example

This illustrates the format only; it is not a result. The first Phase B update is the same batch in A_joint, B1 and B2 (`SMK/smoke_B/q50/seed10501/<arm>/v2_updates.csv`, row 1):

| | Mean | Std |
|---|---|---|
| All rows | 1.46084 | 2.06666 |
| Stage-1 rows | 2.40594 | 1.99507 |

The statistics actually used are those of all rows in A_joint and B1, and those of stage-1 rows in B2. Policy rows: 1024 in A_joint (both stages), 512 in B1 and B2. Actor steps skipped for lack of policy rows: 0.

---

## 5. Measured per-phase wall-clock time

All runs are single process, single thread, CPU.

| Source | Phase | Updates | Wall (s) | CPU (s) | Of which `train_update` (s) |
|---|---|---|---|---|---|
| `REG2/v2_full/v2_run_summary.json` | A | 40 | 3.009 | 2.985 | 2.881 |
| same | B | 40 | 5.299 | 5.257 | 5.083 |
| same | C | 40 | 4.020 | 4.020 | 3.853 |
| `SMK/smoke_checklist.csv`, sampled | A | 40 | 3.319 | 3.118 | 3.148 |
| same, expected | A | 40 | 3.143 | 3.104 | 2.985 |
| same, A_joint | B | 20 | 3.265 | 3.202 | 3.141 |
| same, B1 | B | 20 | 2.952 | 2.740 | 2.838 |
| same, B2 | B | 20 | 3.220 | 3.074 | 3.089 |
| same, B1 + mean | B | 20 | 2.996 | 2.808 | 2.881 |

**Derived, not measured.** The per-update cost is about 0.075–0.083 s in A and about 0.13–0.16 s in B; that range is the measured smoke and regression wall time divided by updates. The verifier and v2 outputs add about 0.03 s per call. Full budgets would therefore take on the order of:
- **A400:** about 30–35 s plus verifier calls;
- **B600:** about 80–100 s.

The pilot reports will give the measured values from `v2_run_summary.json`.

---

## 6. Planned parallelism

- **Machine.** `nproc` = 64; 8× V100, all at 0 % (`nvidia-smi`), unused because the runs are CPU-only.
- **Load average.** 2.93 / 2.17 / 1.33 before the first smoke batch; 2.18 / 2.25 / 1.50 before the second; 2.04 / 2.25 / 1.57 after (`uptime`, `/proc/loadavg`). Other users hold about 2 cores.
- **Plan:** one single-threaded process per run (OMP/MKL/OPENBLAS = 1, torch threads 1, enforced), all runs of a pilot at once:

| Pilot | Runs | Workers |
|---|---|---|
| Pilot 1 | 12 | 12 |
| Pilot 2 | 18 | 18 |
| Pilot 3 | 12 | 12 |

  That is at most 18 of 64 cores. I will re-check the load before each launch and reduce `--workers` if it is above about 40.
- **Launch.** The launcher runs inside `tmux new-session -d -s v2_<pilot> "<launcher command>"`, as the project `CLAUDE.md` requires.
- **Monitoring:**
  - `results/v2_pilots/<pilot>/launch_*.json` lists the finished runs;
  - each run has `status.json` and `run.log`;
  - `tmux attach -t v2_<pilot>`.

---

## 7. Anomalies, deviations, open questions

1. **The dirty flag was wrong in the first smoke attempt.** The first dirty check ignored untracked files, and the first smoke batch ran with uncommitted new code yet recorded `dirty: false`. I fixed the check (untracked files outside `results/` now count), committed the code, deleted that batch and re-ran everything at `8f2840e`. Only the re-run is in the repository.
2. **RNG alignment over long horizons.** It is not guaranteed for the learner and opponent streams (§2.1). Decide whether to add per-update stream-position logging.
3. **Phase B reward mode for Pilots 2 and 3.** The launcher takes `--reward-mode` for Phase B and records it. I propose using the estimator you choose after Pilot 1 for Phase B as well, so the parent and the child use the same reward. Please confirm.
4. **On-path rule.** Still open; see `reports/v2/phase2_opening_checks.md` §1c. The NPZs keep the PMF, so the choice can be made later.
5. **KL and clip fraction differ in scope.** In frozen arms they cover stage-1 rows only, while joint covers all rows. Stability-triggered verifier timing can therefore differ between arms (§1, point 5).
6. **The would-have-fired rule in B** uses EXP_root of the candidate. In frozen arms the candidate includes the frozen stage 2.
7. **Smoke observation.** The Ĝmax_full of the last checkpoint is the same number, 0.0932, in B1, B2 and B1 + mean. This is consistent with Ĝmax_full sitting at stage 2, which is frozen in those arms, but this is not interpreted further.

---

## 8. Commands to reproduce

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_v2_verifier.py tests/test_v2_infra.py -q -p no:cacheprovider
```

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python -B run/run_v2_stagewise.py --config results/v2_pilots/phase2_regression/run_config.json --out-dir results/v2_pilots/phase2_regression/v2_full_repro
```

```bash
/home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/compare_runs.py results/v2_pilots/phase1_regression/before/FINAL_A400_B25_C25/tel_q50_s10501 results/v2_pilots/phase2_regression/v2_full_repro
```

```bash
/home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/launch_pilot.py --pilot smoke_A_repro --phase A --qs 50 --seeds 10501 --arms sampled expected --workers 2 --root results/v2_pilots/_smoke --budget-overrides '{"phase_caps":{"A":40,"B":20,"C":1},"warmup":10,"stability_every":5,"verifier_timeout":5}'
```

```bash
/home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/launch_pilot.py --pilot smoke_B_repro --phase B --parent-pilot smoke_A_repro --parent-arm sampled --reward-mode sampled --qs 50 --seeds 10501 --arms A_joint B1_frozen_allnorm B2_frozen_s1norm B1_frozen_allnorm_mean --workers 4 --root results/v2_pilots/_smoke --budget-overrides '{"phase_caps":{"A":40,"B":20,"C":1},"warmup":10,"stability_every":5,"verifier_timeout":5}'
```

Pilot 1, which is **not to be run until approved**, would be launched like this:

```bash
tmux new-session -d -s v2_pilot1 "cd /home/fjiang4/tournament_experiment/.claude/worktrees/tournament-v2-stagewise-pilots-f97609 && /home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/launch_pilot.py --pilot pilot1 --phase A --qs 50 60 --seeds 10501 10502 10503 --arms sampled expected --workers 12"
```

---

**STOP — Phase 2 complete. Waiting for approval before Pilot 1.**
