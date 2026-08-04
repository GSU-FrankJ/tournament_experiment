# Table 2 metadata audit (2026-07-29)

Answers to the four review questions on `docs/Figures&Tables07272026.docx`, Table 2.

## Which runs are "the four experiments"

`paper/generator/config.py:158 BASELINE_OVERRIDES` relabels the shipped arm to
`baseline` at load time, so figures, Claim-B and the polish all consume the same
runs:

| Scenario | Shipped arm | Convergence glob |
|---|---|---|
| Two-player | `r5_sampled` | `results/two_players/convergence/ppo_q{35,45,55}.0_seed{42..46}_r5_sampled_convergence.json` |
| Three-player | `r5_sampled` | `results/three_players/convergence/ppo_3p_q{35,55}.0_seed{42..46}_r5_sampled_convergence.json` |
| Het. cost | `r5_sampled` | `results/different_cost/convergence/different_cost_ppo_q{35,55}.0_seed{42..46}_r5_sampled_convergence.json` |
| Het. ability | `r5_sampled_std` | `results/different_ability/convergence/different_ability_ppo_q{35,55}.0_seed{42..46}_r5_sampled_std_convergence.json` |

Same list as `tools/one_stage_polish_per_seed.py:97-154`. Het-ability ships
`r5_sampled_std`, not `r5_sampled_v2`.

**Provenance gap:** only `two_players` stored `*_metadata.json` with a `cmdline`.
The other three have no recorded invocation; their settings below are recovered
from config defaults plus signatures in the logged trajectories (marked as
inferred where that applies).

## Q1 — Policy head, conc_min, conc_reg coefficient, variance-loss coefficient

| | Two-player | Three-player | Het. cost | Het. ability |
|---|---|---|---|---|
| `theory_align_v2` | True | True (inferred) | True (inferred) | **False** |
| Policy head | `ActorCriticMeanConc` | `ActorCriticMeanConc` | `ActorCriticMeanConc` | **`ActorCritic`** |
| Head parameterization | mean x concentration (sigmoid mean, softplus conc; a=uc, b=(1-u)c) | same | same | **independent softplus a, b heads (`+1.0` floor)** |
| `conc_min` | **100 for the whole run** | 100 -> 1000 | 100 -> 1000 | n/a (no floor) |
| `conc_scale` | **100 for the whole run** | 100 -> 10000 | 100 -> 10000 | n/a |
| `conc_max` | 100000 (never binding) | 100000 | 100000 | n/a |
| conc_reg coeff (`theory_align_conc_weight`) | **0** | **0** | **0** | **0** |
| var-loss coeff (`theory_align_v2_var_coef`) | **0 for the whole run** | 0 -> 0.05 | 0 -> 0.05 | 0 (no v2) |
| Ramp warmup / steps | 200 / 50 | 200 / 50 | 200 / 50 | n/a |
| Observed stop update | 49-109 | 300-309 | 300-309 | 1000-1219 |

Sources: `agents/ppo_two_players_clean.py:157-171` (head selection),
`:526-539` (conc_reg / var_loss terms), `run/run_two_players.py:1856-1870`
(v2 defaults block; the same block exists at `run_three_players.py:1454-1470`,
`run_different_cost.py:1109-1122`, `run_different_ability.py:1092-1105`),
`run/run_two_players.py:868-886` (ramp).

Three findings that Table 2 currently does not reflect:

1. **conc_reg is dead code in all four runs.** `conc_reg` is gated on the
   `theory_align` (v1) path, which is mutually exclusive with v2 and was never
   used; `theory_align_conc_weight` defaults to 0.0 regardless. The coefficient
   is 0 everywhere.
2. **The two-player ramp never fires.** Warm-up is 200 updates
   (`--override-conc-ramp-warmup 200`, confirmed in the stored cmdline) but every
   two-player run stops at update <= 109. `update_idx < warmup` holds throughout,
   so `ramp_t = 0` and the run trains at `conc_min=100`, `conc_scale=100`,
   `var_coef=0` from start to finish. "Ramped 100->1000" describes three-player
   and het-cost only.
3. **Het-ability uses a different policy head.** With `theory_align_v2=False` the
   agent builds the plain `ActorCritic`, which has independent softplus alpha/beta
   heads and no concentration floor. Table 2's Network row does not describe it.

Evidence for the inferred rows (no cmdline stored):

- Three-player `alpha_mean+beta_mean` is smooth at ~360 through update 191, then
  917.7 at update 201 and 28909 by update 251 — exactly `warmup=200`,
  `ramp_steps=50`.
- Het-cost `batch_entropy_agent1` is flat at ~-2.16 through update 191, drops
  sharply from update 201 and plateaus at ~-4.35 by update 251 — same signature.
- Het-ability `r5_sampled_std` starts at `batch_entropy = -0.09` (near-uniform
  Beta, so no concentration floor) and decays smoothly to -1.29 over 1200
  updates with no ramp discontinuity. The `r5_sampled_v2` arm on the same
  scenario starts at -1.83 and ends at -4.32, i.e. the v2 signature.

## Q2 — Resolved entropy coefficient

| Scenario | Run directory | Resolved c_ent at update 0 | Final c_ent |
|---|---|---|---|
| Two-player | `results/two_players/convergence/` (`*_r5_sampled_*`) | 0 | 0 |
| Three-player | `results/three_players/convergence/` (`*_r5_sampled_*`) | 0 | 0 |
| Het. cost | `results/different_cost/convergence/` (`*_r5_sampled_*`) | 0 | 0 |
| Het. ability | `results/different_ability/convergence/` (`*_r5_sampled_std_*`) | **0.03** | **0.0190-0.0300** |

Per-seed final `c_ent` for het-ability (schedule evaluated at each seed's stop):

| q | seed | stop update | final c_ent |
|---|---|---|---|
| 35 | 42 | 1219 | 0.01903 |
| 35 | 43 | 1192 | 0.02038 |
| 35 | 44 | 1171 | 0.02143 |
| 35 | 45 | 1023 | 0.02885 |
| 35 | 46 | 1160 | 0.02198 |
| 55 | 42 | 1007 | 0.02965 |
| 55 | 43 | 1000 | 0.03000 |
| 55 | 44 | 1001 | 0.02995 |
| 55 | 45 | 1002 | 0.02990 |
| 55 | 46 | 1000 | 0.03000 |

### Why the doc contradicts itself

Both statements in the doc are true, about different layers:

- **Config layer — all four identical.** `entropy_coef_start=0.03`,
  `entropy_coef_hold=0.03`, `entropy_coef_end=0.005`,
  `entropy_hold_fraction=2/3` in all four `config/one_stage_*.py`
  (the two configs that omit `entropy_hold_fraction` get 2/3 from the runner
  default).
- **CLI layer — v2 zeroes it.** The `--theory-align-v2` block sets
  `entropy_coef_start/hold/end = 0` (`run_two_players.py:1857-1859` and the
  equivalent line in each other runner), overriding the config. Two-player gets
  v2 automatically (`run_two_players.py:1766-1769` defaults it to True for
  `--method ppo`); three-player and het-cost got it explicitly. Het-ability's
  shipped arm did not, so its config schedule survives.

Schedule arithmetic for het-ability (`run_different_ability.py:617-621, 687-697`):
`total_updates = 6144000/4096 = 1500`, `hold_updates = ceil(1500 * 2/3) = 1000`,
`tail_updates = 500`. So `c_ent = 0.03` for updates 0-999, then linear to 0.005
across updates 1000-1499. Every seed stops inside the first half of that tail,
which is why the final values sit between 0.019 and 0.030 rather than near 0.005.

### Follow-up: was entropy in het-ability a preset choice or something the model picked?

**A preset choice. The model has no influence on the coefficient at all.**

`agent.cfg.entropy_coef` is written in exactly two places
(`run_different_ability.py:693` and `:697`), both branches of a schedule whose
only input is `update_idx`. The agent side merely reads it in the loss
(`agents/ppo_two_players_clean.py:525`). There is no KL-triggered adjustment, no
adaptive bump, and no gradient path into it. Same structure in all four runners.
(`run_three_players.py:999` does pin `entropy_coef` to `cont_entropy_coef`, but
only under the Claim-A continuation ladder — `cont_phase` is `"disabled"` unless
continuation is enabled, which the shipped `r5_sampled` arm does not use.)

But the commonly-given justification — *"it is hardcoded in config and identical
to the other three experiments: `one_stage_different_ability.py:59-62`"* — is
only half right, and the wrong half is load-bearing:

- **True about the config.** Lines 59-62 do read
  `start=0.03, hold=0.03, end=0.005, hold_fraction=2/3`, and all four configs
  agree (two-player and three-player omit `entropy_hold_fraction` and inherit
  2/3 from the runner default).
- **False about what ran.** The other three had those values overwritten to 0 by
  the `--theory-align-v2` block before training started. Effective c_ent is 0 for
  them and 0.03 for het-ability. Citing config identity therefore explains away
  precisely the fact that needs explaining: het-ability is the only scenario
  where entropy was live.

The actual causal chain, all of it human decisions:

1. Het-ability shipped the `r5_sampled_std` arm rather than `r5_sampled_v2`
   (the v2 / entropy=0 arm exists on disk and was rejected).
2. No v2 means the CLI never zeroes the entropy schedule.
3. So the config schedule survives and entropy is active during training.

The model participates at no step. What the model does produce is the resulting
policy entropy (`batch_entropy`, ending near -1.3 for het-ability vs -4.2 to -4.6
for the v2 scenarios) — that is a training outcome, not a setting that was chosen.

## Q3 — Effort-drift threshold definition

`CheapGateTracker.compute` (`run/run_two_players.py:178-194`):

```
drift_effort = | policy_mean_effort[t] - policy_mean_effort[t-W+1] |
```

**Last minus first of the rolling window, absolute value** — not the difference
of window means, and not the window max-min. `policy_mean_effort` is the Beta-mean
effort, one value per PPO update. The window is a `deque(maxlen=W)`; until it is
full `compute` returns `None` and the gate cannot pass (`fail_reasons` records
`drift:window`).

Resolved values, identical across all four scenarios (`relaxed` profile, which is
the PPO default):

| Parameter | Value |
|---|---|
| `window_size` W | 20 (same deque as the KL window) |
| `drift_effort_thresh` | 8.0 |
| `mean_kl_thresh` | 0.015 |
| `std_kl_thresh` | 0.012 |
| `patience_drift` | 1 |

The gate passes when `mean_kl <= 0.015` and `std_kl <= 0.012` and
`drift_effort <= 8.0`, sustained for `patience_drift` consecutive checks.

Note the `CHEAP_GATE_CONFIG` in `paper/generator/config.py:199` lists
`drift_effort_thresh: 2.0`, which is the strict-profile value and does not match
what the runs used. The runs used 8.0.

## Q4 — Final MC sample size vs training verifier

**No, they differ.** Three distinct sample sizes are in play:

| Stage | M | Grid | Seeds / CRN | Code |
|---|---|---|---|---|
| Training verifier (in-loop) | **8192** two-player, **16384** other three | coarse-to-fine 5.0 -> 1.0 -> 0.25 | CRN within a call | `run_*.eval_exploitability`, `exploit_config.exploit_M` in each convergence JSON |
| MC-BR polishing | **150000** | damped BR iteration, eta=0.4, 320 rounds, n_avg=200, bias-corrected | polish seed per run | `results/one_stage_ablation/polish_per_seed_all.json` -> `pol_config` |
| Final independent check | **200000** | uniform grid, step 0.25 | fresh seed `700000 + int(q)*1000 + si*7`, shared across the four Table-3 arms within a (q, seed-index) cell (CRN across arms and across the grid) | `utils/mc_br_polish.py:211 exploitability_frozen_profile`, called from `tools/unified_exploitability_tables.py:62 eval_seed` |

So the final check draws 24x the two-player training verifier and 12x the
three-player / heterogeneous one, and is not the same as the polish budget
either. Table 2's "8192 MC samples with common random numbers" is correct only
for the two-player scenario.

Note on the final-check seed: `exploitability_frozen_profile` declares
`seed: int = 99_991` as a parameter default, but every Table-3 and Table-4 row
is produced with that default OVERRIDDEN by `eval_seed(q, si)` above. 99991 is
never used for any reported number. The eval seeds are disjoint from the
MC-BR polishing seeds (2000/4000 + si), which is what the "fresh draws"
requirement asks for; sharing a seed across the four arms of a cell is
deliberate, so common random numbers cancel most of the estimator noise in
the arm-to-arm DIFFERENCE.

## Corrections Table 2 needs

1. **Entropy coefficient** — "0 for the entire run" holds for two-player,
   three-player and het-cost only. Het-ability runs at 0.03 annealing to
   0.019-0.030. Either split the row by scenario or footnote the exception.
2. **Payoff evaluation** — "8192 MC samples" is two-player only; the other three
   use 16384.
3. **Network row** — the mean x concentration head with a ramped floor describes
   three of four scenarios. Het-ability uses the plain independent-alpha/beta head
   with no floor. Also, the two-player ramp never engages (runs end at update
   <= 109 vs a 200-update warm-up), so its effective settings are conc_min=100,
   conc_scale=100, var_coef=0 throughout.
4. **Add** conc_reg coefficient = 0 (inactive in all four) and variance-loss
   coefficient = 0.05 for three-player / het-cost, 0 for two-player and
   het-ability, if these are to be reported at all.
5. **Final check** — state M=200000 explicitly so it is not read as the 8192
   training verifier.
