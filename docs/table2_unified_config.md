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
| Verifier (in-training) | Tolerance / patience | ε=0.01 / **5** consecutive checks (r9; was ε=0.03) |
| Verifier (in-training) | Evaluation checkpoint interval | at most every **10** PPO updates; the stability gate can trigger a check earlier (`exploit_every_updates=10`, recorded in every run's `exploit_config`) |
| Stopping | Principle | Stop at the 5th consecutive verifier pass; NO minimum-update floor; 1500-update budget cap |
| Final check | Independent MC exploitability | M=200000, uniform grid 0.25, fresh seeds 700000+q·1000+si·7 (unchanged) |

**Honest footnote for the paper (realized vs preset).** Because stopping is
verification-triggered, scenarios that verify early never reach the late part
of the shared schedule. The preset is identical; the REALIZED late-schedule
values at stop depend only on when the verifier certifies. This replaces the
old per-scenario table of differing conc/var values.

## Realized values at stop (fill per generation)

r9 generation (ε=0.01, M=65,536):

| Scenario | Runs | Stop updates | Realized conc_min / var_coef at stop | Entropy |
|---|---|---|---|---|
| Two-player S1 | r9_cert001 | 49–139 (pre-ramp) | 100 / 0 | 0 |
| Two-player S2 | wh8_wl4_r9_cert001 | 89–**209** | 100 / 0, except one seed (see below) | 0 |
| Three-player q35 | r9_cert001 | 44–52 (pre-ramp) | 100 / 0 | 0 |
| Three-player q55 | r8_sens_eps001 | 88–123 (pre-ramp) | 100 / 0 | 0 |
| Het. cost | r9_cert001 | 96–145 (pre-ramp) | 100 / 0 | 0 |
| Het. ability q35 | r9_cert001 | 100–160 (pre-ramp) | 100 / 0 | 0 |
| Het. ability q55 | r8_sens_eps001 | 81–120 (pre-ramp) | 100 / 0 | 0 |

**Realized configuration is constant in 74 of the 75 shipped runs:**
conc_min=100, conc_scale=100, var_coef=0, entropy=0 throughout, because
verification fires before the 200-update ramp warm-up. The single exception is
Set 2, q=35, seed 45, which stops at update 209 and therefore spends its last
9 updates inside the ramp (ramp_t=0.2 → conc_min≈280, var_coef≈0.01). Under
ε=0.03 (r7/r8) no run reached the ramp at all; the tighter certificate trains
long enough that the schedule is no longer strictly inert.

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
