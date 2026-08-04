# Unified configuration — single source of truth for Table 2 (r9 generation)

> 2026-08-03 (final): certificate tightened to ε=0.01 / M=65,536 for the whole
> matrix (tag r9_cert001; 3P/da q55 reuse r8_sens_eps001 which IS this config;
> no_exploit ablation reuses r7_fig7_no_exploit — verifier disabled, ε/M never
> enter training). Post-training MC-BR polish row is RETIRED from the framework.

Owner directive (2026-08-01): one configuration for all four scenarios — one
policy head, one entropy/concentration/variance preset, one training-time
verifier, one stopping principle. No per-scenario selection.

The two design choices this fixes, and why they were resolved this way:

1. **Policy head = mean × concentration everywhere.** Three of four scenarios
   already trained on it; the shipped het-ability exception (independent α,β)
   is removed. The independent-α,β arm survives as an ablation
   (`r7_state4_std`), not as configuration.
2. **Stopping = verification-triggered, no minimum-update floor, for all.**
   The scenario-specific floors (3P/dc 300, da 1000) are removed. This is the
   only unification compatible with the two-player runner (which has no floor
   flag) and it eliminates the floor/ramp confound documented in
   docs/tasks/r7-state4-wave/STATE.md.

## Table 2 (unified; identical for all four scenarios)

| Category | Parameter | Value |
|---|---|---|
| Optimization | Optimizer | Adam |
| Optimization | Learning rate | 5×10⁻⁵ → 2×10⁻⁵ |
| Optimization | Batch size | 4096 episodes/update |
| Optimization | Minibatch size | 1024 |
| Optimization | Epochs/update | 1 |
| Optimization | Maximum PPO updates | 1500 (budget cap) |
| Optimization | Random seeds | 5 (42–46) |
| PPO / Policy | Clip range | 0.20 → 0.15 |
| PPO / Policy | Value-loss coefficient | 0.5 |
| PPO / Policy | Entropy coefficient | 0 for the entire run |
| PPO / Policy | Max gradient norm | 0.5 |
| PPO / Policy | Discount factor | 0.99 |
| PPO / Policy | GAE parameter | 0.95 |
| PPO / Policy | KL target | 0.06 |
| PPO / Policy | Ratio-stop threshold | 2.2 |
| Network | Architecture | Shared actor–critic trunk (two Tanh hidden layers, hidden 128); Beta policy head parameterized as mean × concentration (sigmoid mean, softplus concentration; α=μc, β=(1−μ)c); separate value head. |
| Network | Concentration / variance preset | ONE schedule for all: warm-up 200 updates, ramp 50; conc_min 100→1000, conc_scale 100→10000 (cap 100000), variance-loss coef 0→0.05 |
| State | Input variables | s_i = [q/60, k_i/10⁻³, (w_H−w_L)/10, (l_i−l̄₋ᵢ)/10], constant per run (state_dim = 4; 4th component 0 except het-ability ±0.5) |
| Diagnostics | KL window | 20 |
| Diagnostics | KL threshold | mean 0.015 + std 0.012 |
| Diagnostics | Effort-drift threshold | 8.0 (patience 1) |
| Verifier (in-training) | Payoff evaluation | 65,536 MC samples, common random numbers (r9; was 16,384 in r7/r8) |
| Verifier (in-training) | Search grid | coarse-to-fine 5.0 / 1.0 / 0.25 |
| Verifier (in-training) | Tolerance / patience / cadence | ε=0.01 / 5 consecutive checks / gate-triggered or every 10 updates (r9; was ε=0.03) |
| Stopping | Principle | Stop at the 5th consecutive verifier pass; NO minimum-update floor; 1500-update budget cap |
| Final check | Independent MC exploitability | M=200000, uniform grid 0.25, fresh seeds 700000+q·1000+si·7 (unchanged) |

**Honest footnote for the paper (realized vs preset).** Because stopping is
verification-triggered, scenarios that verify early never reach the late part
of the shared schedule. The preset is identical; the REALIZED late-schedule
values at stop depend only on when the verifier certifies. This replaces the
old per-scenario table of differing conc/var values.

## Realized values at stop (fill per generation)

| Scenario | Runs | Stop updates | Realized conc_min / var_coef at stop | Entropy |
|---|---|---|---|---|
| Two-player S1 | r7_state4 | 49–99 (pre-ramp) | 100 / 0 | 0 |
| Two-player S2 | wh8_wl4_r7_state4 | 49–119 (pre-ramp) | 100 / 0 | 0 |
| Three-player | r8_unified | 43–88 (pre-ramp) | 100 / 0 | 0 |
| Het. cost | r8_unified | 89–129 (pre-ramp) | 100 / 0 | 0 |
| Het. ability | r8_unified | 22–93 (pre-ramp) | 100 / 0 | 0 |

**Simplification discovered by r8:** under the unified no-floor rule, EVERY
scenario verifies before the 200-update ramp warm-up. The realized
configuration is therefore identical everywhere AND constant through training:
conc_min=100, conc_scale=100, var_coef=0, entropy=0. The late part of the
conc/var schedule is inert in this generation — Table 2 can state the realized
values directly instead of the schedule footnote.

## Where every reviewed figure/table gets its data (this generation)

| Artifact (review doc) | Source files |
|---|---|
| Table 2 (config) | this file; per-run `exploit_config` inside each convergence JSON; cmdlines in `results/r7_state4/manifest.csv`, `results/r8_unified/manifest.csv`; code snapshots `results/{r7_state4,r8_unified}/code_state.{txt,diff}` |
| Table 3 (ablation, dual endpoint) | raw: `results/two_players/convergence/ppo_q*_{r7_state4,r7_fig7_no_stability,r7_fig7_no_exploit}_convergence.json`; polished: `results/one_stage_ablation/polish_per_seed_r7.json` (CRN seed 4000+si shared across the three arms per (q,si)) |
| Table 4 (all scenarios) | raw: `*_r7_state4_*` (2P) + `*_r8_unified_*` (3P/dc/da) convergence JSONs; polished: `polish_per_seed_r7.json` stages A+B |
| Figure 2 (convergence) | same convergence JSONs (`policy_mean_effort`, `stopped_at_update`); Set-2 stars from polish rows `two_players_set2` |
| Head ablation (appendix) | `r7_state4_std` vs `r7_state4_v2` (same floor-1000 stopping within the pair, 4-dim, M=16384) — justifies the unified head choice without per-scenario tuning |
| Estimator floor | `results/one_stage_ablation/exploit_noise_floor.json` (training-independent; unchanged) |
| MC-BR-only baseline (what polish does without training) | `results/one_stage_ablation/mc_br_only.json`; pre-registered interpretation in `docs/ablation_narrative_preregistered.md` |
| 3-dim ↔ 4-dim accuracy check (red item 2) | r5 vs r7/r8 per-cell table in `docs/tasks/r7-state4-wave/STATE.md` |

Generator promotion (`BASELINE_OVERRIDES` → r7/r8 tags) is deliberately NOT
done yet — it happens after the owner reviews this generation's numbers.
