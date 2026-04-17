# Metric Consistency Diagnosis: sample mean vs policy mean

## Section 1: Field inventory

Source: `results/two_players/convergence/ppo_q55.0_seed42_convergence.json`

| field | shape | last_value | meaning |
|-------|-------|------------|---------|
| `algorithm` | str | `"PPO"` | algorithm name |
| `q` | float | `55.0` | discrimination parameter |
| `seed` | int | `42` | random seed |
| `ablation_name` | str | `"baseline"` | experiment variant name |
| `theoretical_effort` | float | `28.926` | e* = W/(4qk) |
| `theoretical` | dict(1) | `{"effort": 28.926}` | structured theoretical values |
| `final` | dict(2) | `{"effort": 29.288, "gap": 0.362}` | final-update policy mean + gap |
| `stop_reason` | str | `"exploitability"` | why training stopped |
| `stopped_at_update` | int | `99` | update index at stop |
| `joint_exploit_ok_streak` | int | `5` | consecutive exploit-OK evals |
| `final_exploit_1` | float | `0.006429` | last exploitability (player 1) |
| `final_exploit_2` | float | `0.006429` | last exploitability (player 2) |
| `final_exploit_max` | float | `0.006429` | max of exploit_1, exploit_2 |
| `final_br_effort_1` | float | `26.0` | best-response effort (player 1) |
| `final_br_effort_2` | float | `26.0` | best-response effort (player 2) |
| `steps` | list[99] | `401408` | cumulative env steps per update |
| `agent1_effort` | list[99] | `29.670` | **per-update sample mean of player 1 actions** |
| `agent2_effort` | list[99] | `29.580` | **per-update sample mean of player 2 actions** |
| `policy_mean_effort` | list[99] | `29.288` | **per-update policy distribution mean (deterministic)** |
| `approx_kl` | list[99] | `0.002377` | approximate KL divergence |
| `batch_entropy` | list[99] | `-2.1368` | batch entropy |
| `alpha_mean` | list[99] | `74.326` | Beta distribution alpha parameter |
| `beta_mean` | list[99] | `179.455` | Beta distribution beta parameter |
| `mean_kl_window` | list[99] | `0.000974` | windowed mean KL (sparse, most NaN) |
| `drift_effort` | list[99] | `1.027` | effort drift over sliding window (sparse, most NaN) |
| `exploitability` | list[99] | `0.006429` | exploitability (sparse, most NaN) |
| `exploitability_is_valid` | list[99] | `True` | whether exploit eval was done this update |
| `exploit_eval_steps` | list[10] | `405504` | steps at which exploit was evaluated |
| `rollout_mode` | str | `"selfplay"` | rollout mode |
| `total_episodes` | int | `6144000` | max episode budget |
| `disable_cheap_gate` | bool | `False` | cheap gate ablation flag |
| `disable_exploitability` | bool | `False` | exploit eval ablation flag |
| `exploit_every_updates` | int | `10` | exploit eval frequency |
| `exploit_config` | dict(5) | — | exploit evaluation config |

Fields containing "mean" or "effort": `agent1_effort`, `agent2_effort`, `policy_mean_effort`, `alpha_mean`, `beta_mean`, `theoretical_effort`, `final.effort`, `final_br_effort_{1,2}`.

No field called `effort_mean` or `effort_std` exists in the JSON.

## Section 2: True meaning of each field (from code)

| field | computation (file:line) | policy mean or sample mean | noise sources |
|-------|------------------------|---------------------------|---------------|
| `policy_mean_effort` | `run_two_players.py:1146-1151`: `dist.mean` (= alpha/(alpha+beta)) from `agent.dist(s_eval)`, clamped to [0,1], then scaled to effort range via `effort_bounds[0] + a_eval * (effort_bounds[1] - effort_bounds[0])` | **policy mean** (deterministic) | none — exact Beta distribution mean at fixed evaluation state `s_eval` |
| `agent1_effort` | `run_two_players.py:1067,1172`: Welford online mean of `e1_env` values collected during rollout, via `rollout_stats.update_effort(e1_env, player="p1")`; `e1_env` = sampled action mapped to effort space | **sample mean** (stochastic) | Monte Carlo sampling from Beta(alpha, beta); ~4096 samples per player per update |
| `agent2_effort` | `run_two_players.py:1071,1173`: same as agent1_effort but for player 2 | **sample mean** (stochastic) | same as agent1_effort |
| `final.effort` | `run_two_players.py:1685-1689,1702-1703`: `compute_policy_mean_effort(alpha, beta, low, high)` = `alpha/(alpha+beta) * (high-low) + low` | **policy mean** (deterministic) | none — identical computation as `policy_mean_effort[-1]` |
| `alpha_mean` | `run_two_players.py:1152`: `dist.concentration1.mean().item()` | — | direct parameter readout |
| `beta_mean` | `run_two_players.py:1153`: `dist.concentration0.mean().item()` | — | direct parameter readout |

### Key relationships confirmed from code

- `final.effort` ≡ `policy_mean_effort[-1]` (verified: max diff across all 15 runs < 3e-6)
- `sample_avg_effort` (used in logging at line 1162, not persisted in convergence JSON) = Welford mean of ALL sampled efforts (both players combined)
- `(agent1_effort[i] + agent2_effort[i]) / 2` ≈ `sample_avg_effort` for update i
- `mean_vs_sample_gap` (logged at line 1192, not persisted) = `policy_mean_effort - sample_avg_effort`

### What `round2_full_results.md` Section 1 actually used

Section 1 `final_effort` = `mean(policy_mean_effort[-50:])` — this is **metric C** (policy mean, last-50 average), not the sample mean. Both Section 1 and Section 4 used the same field (`policy_mean_effort`), differing only in windowing (last-50 average vs single last value).

The field name `effort_mean` referenced in the original task spec does not exist in the JSON. The implementation used `policy_mean_effort` as the closest match.

## Section 3: Three metrics comparison

Definitions:
- **A** = sample mean, last 50 updates avg: `mean([(agent1_effort[i] + agent2_effort[i])/2 for i in last_50])`
- **B** = policy mean, last update: `policy_mean_effort[-1]`
- **C** = policy mean, last 50 updates avg: `mean(policy_mean_effort[-50:])`

e* = 3.5 / (4 * q * 0.00055)

| q | seed | A_sample_mean | B_policy_last | C_policy_avg | A_gap | B_gap | C_gap |
|--:|-----:|--------------:|--------------:|-------------:|------:|------:|------:|
| 35 | 42 | 45.06 | 43.46 | 44.94 | 0.39 | 2.00 | 0.52 |
| 35 | 43 | 44.67 | 44.26 | 44.56 | 0.78 | 1.19 | 0.89 |
| 35 | 44 | 46.02 | 44.34 | 45.92 | 0.56 | 1.11 | 0.47 |
| 35 | 45 | 41.62 | 41.49 | 41.51 | 3.84 | 3.96 | 3.94 |
| 35 | 46 | 47.40 | 46.96 | 47.40 | 1.94 | 1.51 | 1.95 |
| 45 | 42 | 37.77 | 35.82 | 37.65 | 2.42 | 0.47 | 2.30 |
| 45 | 43 | 37.48 | 34.37 | 37.29 | 2.12 | 0.98 | 1.94 |
| 45 | 44 | 36.54 | 33.51 | 36.35 | 1.18 | 1.84 | 1.00 |
| 45 | 45 | 37.99 | 35.48 | 37.82 | 2.63 | 0.13 | 2.46 |
| 45 | 46 | 37.13 | 35.61 | 37.01 | 1.78 | 0.26 | 1.65 |
| 55 | 42 | 31.60 | 29.29 | 31.50 | 2.68 | 0.36 | 2.57 |
| 55 | 43 | 31.24 | 29.60 | 31.12 | 2.32 | 0.67 | 2.20 |
| 55 | 44 | 30.84 | 27.84 | 30.70 | 1.91 | 1.08 | 1.77 |
| 55 | 45 | 30.79 | 28.20 | 30.64 | 1.86 | 0.73 | 1.72 |
| 55 | 46 | 31.36 | 29.49 | 31.26 | 2.44 | 0.56 | 2.34 |

## Section 4: Systematic bias analysis

### A − B decomposition

`A − B = (A − C) + (C − B)`

where:
- `A − C` = per-update sample noise averaged over the last-50 window
- `C − B` = "window effect" — last-50 average includes still-converging (higher) values

| q | mean(A − B) | std(A − B) | mean(A − C) | mean(C − B) | direction |
|--:|------------:|------------|------------:|------------:|-----------|
| 35 | 0.8504 | 0.7315 | 0.1160 | 0.7644 | A > B (sample > policy_last) |
| 45 | 2.4209 | 0.6827 | 0.1573 | 2.2636 | A > B (sample > policy_last) |
| 55 | 2.2845 | 0.5420 | 0.1211 | 2.1634 | A > B (sample > policy_last) |

### Per-update sample-vs-policy gap (last 50 updates of each run)

| q | seed | mean(sample − policy) per update | std | conc_last (alpha+beta) |
|--:|-----:|---------------------------------:|----:|---:|
| 35 | 42 | +0.1258 | 0.2820 | 196 |
| 35 | 43 | +0.1108 | 0.2670 | 234 |
| 35 | 44 | +0.0975 | 0.2261 | 184 |
| 35 | 45 | +0.1058 | 0.3047 | 193 |
| 35 | 46 | -0.0055 | 0.2308 | 204 |
| 45 | 42 | +0.1206 | 0.2482 | 237 |
| 45 | 43 | +0.1838 | 0.2673 | 165 |
| 45 | 44 | +0.1824 | 0.2287 | 223 |
| 45 | 45 | +0.1715 | 0.2734 | 199 |
| 45 | 46 | +0.1282 | 0.2722 | 220 |
| 55 | 42 | +0.1036 | 0.2164 | 254 |
| 55 | 43 | +0.1202 | 0.2649 | 250 |
| 55 | 44 | +0.1391 | 0.2160 | 250 |
| 55 | 45 | +0.1410 | 0.2373 | 224 |
| 55 | 46 | +0.1016 | 0.2437 | 222 |

### Diagnosis

The A − B gap has **two distinct components**:

**Component 1: Window effect (C − B) — dominant, 0.02 to 2.92 units.**
The policy is still descending toward e* in the last 50 updates, so averaging over a window that extends into the approach phase inflates the reported effort above the final converged value. This effect is larger for q=45/55 because these runs take 59–109 updates (the last 50 spans a wide range of the converging tail) vs q=35 which stops at 49–69 updates (last 50 = almost all updates for 49-update runs, reducing the window effect only when the trajectory has flattened).

**Component 2: Sample noise (A − C) — small, +0.09 to +0.18 units per q.**
The per-update sample mean (from ~4096 actions per player) is consistently ~0.12 units above the policy distribution mean. Hypothesis (not verified): at high concentration (alpha+beta ≈ 200), the Beta distribution is nearly symmetric, but when alpha < beta (effort < midpoint of bounds), the right tail is slightly heavier than the left tail (right-skew when the mean is below the midpoint of [0,1]), pushing the sample mean above the distribution mean by a small margin. This effect is O(1/concentration) and consistent with the ~0.1 unit magnitude observed. Alternative hypothesis: boundary clipping effects from the `clamp(0,1)` on the normalized action could contribute, but at concentration ~200, this is negligible.

## Section 5: Per-q summary with three metrics

| q | n | mean_A_gap | mean_B_gap | mean_C_gap | mean_A_rel% | mean_B_rel% | mean_C_rel% |
|--:|--:|----------:|-----------:|-----------:|------------:|------------:|------------:|
| 35 | 5 | 1.50 | 1.96 | 1.55 | 3.31 | 4.30 | 3.42 |
| 45 | 5 | 2.03 | 0.74 | 1.87 | 5.74 | 2.08 | 5.29 |
| 55 | 5 | 2.24 | 0.68 | 2.12 | 7.75 | 2.36 | 7.33 |

Note: for q=35, metric B gives a LARGER gap than A/C. This is because the q=35 trajectories overshoot past e* (some seeds converge below 45.45), so the final value is further from e* than the last-50 average which includes higher values closer to e*.

For q=45/55, metric B gives a SMALLER gap than A/C. This is because the trajectories approach e* from above and are still above e* at termination. The last-50 average includes earlier, higher values that inflate the gap.

## Section 6: Answers

**Q1: Paper should report metric B — `policy_mean_effort[-1]`, the policy distribution mean at the final update.**

Reason: this is the deterministic, noise-free output of the learned policy at termination. It is what `final.effort` and `final.gap` in the JSON already store (confirmed: `final.effort` ≡ `policy_mean_effort[-1]` to <3e-6 precision). It is also what the training loop itself uses to compute the gap displayed in `[Update N]` log lines (line 1194: `gap = abs(final_e2_eval - e2_star_val)`). Reporting the policy mean is standard practice for policy-gradient methods: the policy parameters define the converged strategy, not any particular batch of samples.

Metric C (policy mean, last-50 avg) conflates "where the policy ended up" with "how fast it got there," systematically overstating the gap for runs that are still approaching equilibrium. Metric A adds sampling noise on top of the same window-inflation problem.

**Q2: Yes, Section 1 of `round2_full_results.md` is misleading for q=45 and q=55.**

Specifically:
- Section 1 reported `final_effort` = `mean(policy_mean_effort[-50:])` (metric C)
- Section 4 reported `t=final` = `policy_mean_effort[-1]` (metric B)
- For q=45: C overstates the gap by 1.13 units on average (mean_C_gap=1.87 vs mean_B_gap=0.74)
- For q=55: C overstates the gap by 1.44 units on average (mean_C_gap=2.12 vs mean_B_gap=0.68)
- The "convergence accuracy" of the Round 2 runs is substantially better than Section 1 suggests: mean relative gap is 2.08% (q=45) and 2.36% (q=55) by metric B, vs 5.29% and 7.33% by metric C.

Section 1 is misleading in the sense that it reports the gap as if the policy "settled at" 37.65 (q=45 seed=42) when in fact the policy's actual terminal state is 35.82 — much closer to e*=35.35. The last-50 average absorbs the approach trajectory into a metric that should reflect the endpoint.
