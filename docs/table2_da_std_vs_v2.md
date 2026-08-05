# Table 2 (het-ability): `r5_sampled_std` vs `r5_sampled_v2`

Both arms are the same wave (2026-06-10), same 5 seeds (42–46), same q ∈ {35, 55},
same `--min-updates 1000`. The only CLI difference is `--theory-align-v2`.

```
std : run/run_different_ability.py --method ppo --q {35,55} --seed {42..46} \
      --min-updates 1000 --episodes 6144000 --ablation-name r5_sampled_std
v2  : run/run_different_ability.py --method ppo --q {35,55} --seed {42..46} \
      --theory-align-v2 --override-conc-ramp-warmup 200 \
      --min-updates 1000 --episodes 6144000 --ablation-name r5_sampled_v2
```

`r5_sampled_std` is the **shipped** arm (`paper/generator/config.py` `BASELINE_OVERRIDES`;
`tools/one_stage_polish_per_seed.py::cells_different_ability`). `r5_sampled_v2` exists on
disk and was rejected.

Sources: `config/one_stage_different_ability.py`, the `--theory-align-v2` override block
`run/run_different_ability.py:1092-1109`, `PPOConfig` defaults
`agents/ppo_two_players_clean.py:110-148`, `PPOConfig` construction
`run/run_different_ability.py:563-581`, and `exploit_config` recorded in each
convergence JSON.

---

## Table 2-A. Het-ability `r5_sampled_std` (shipped)

| Category | Parameter | Value |
|---|---|---|
| Optimization | Optimizer | Adam |
| Optimization | Learning rate | 3×10⁻⁴ → 2×10⁻⁴ |
| Optimization | Batch size | 4096 episodes/update |
| Optimization | Minibatch size | 1024 |
| Optimization | Epochs/update | 6 |
| Optimization | Maximum PPO updates | 1500 |
| Optimization | Minimum PPO updates | 1000 |
| Optimization | Random seeds | 5 (42–46) |
| PPO / Policy | Clip range | 0.50 → 0.35 |
| PPO / Policy | Value-loss coefficient | 0.5 |
| PPO / Policy | Entropy coefficient | 0.03 held for updates 0–999, then linear to 0.005 over 1000–1499 (**live**) |
| PPO / Policy | Max gradient norm | 0.5 |
| PPO / Policy | Discount factor | 0.99 |
| PPO / Policy | GAE parameter | 0.95 |
| PPO / Policy | KL target | 0.08 |
| PPO / Policy | Ratio-stop threshold | none |
| Network | Architecture | Shared actor–critic trunk (two Tanh hidden layers, hidden 128); Beta policy head as **two independent softplus heads** for α and β (each `+1.0` floor); no concentration floor; separate value head. |
| Network | Concentration floor | n/a |
| Network | Variance-loss coefficient | 0 |
| State | Input variables | Normalized (q, k, w_H−w_L), constant per run (state_dim = 3) |
| Diagnostics | KL window | 20 |
| Diagnostics | KL threshold | mean 0.015 + std 0.012 |
| Diagnostics | Effort-drift threshold | 8.0 |
| Diagnostics | Drift patience | 1 |
| Diagnostics | Exploitability tolerance | 0.03 |
| Diagnostics | Exploitability patience | 5 consecutive checks |
| Diagnostics | Exploitability check interval | every 10 updates |
| MC-BR / Verification | Payoff evaluation | 16384 MC samples with common random numbers |
| MC-BR / Verification | Search grid & coarse-to-fine steps | 5.0, 1.0, and 0.25 |
| MC-BR / Verification | Final check | Independent MC exploitability evaluation after polishing |

---

## Table 2-B. Het-ability `r5_sampled_v2` (unified configuration)

| Category | Parameter | Value |
|---|---|---|
| Optimization | Optimizer | Adam |
| Optimization | Learning rate | 5×10⁻⁵ → 2×10⁻⁵ |
| Optimization | Batch size | 4096 episodes/update |
| Optimization | Minibatch size | 1024 |
| Optimization | Epochs/update | 1 |
| Optimization | Maximum PPO updates | 1500 |
| Optimization | Minimum PPO updates | 1000 |
| Optimization | Random seeds | 5 (42–46) |
| PPO / Policy | Clip range | 0.20 → 0.15 |
| PPO / Policy | Value-loss coefficient | 0.5 |
| PPO / Policy | Entropy coefficient | 0 for the entire run |
| PPO / Policy | Max gradient norm | 0.5 |
| PPO / Policy | Discount factor | 0.99 |
| PPO / Policy | GAE parameter | 0.95 |
| PPO / Policy | KL target | 0.06 |
| PPO / Policy | Ratio-stop threshold | 2.2 |
| Network | Architecture | Shared actor–critic trunk (two Tanh hidden layers, hidden 128); Beta policy head parameterized as **mean × concentration** (sigmoid mean, softplus concentration; α = μc, β = (1−μ)c); separate value head. |
| Network | Concentration floor | ramped 100 → 1000 (scale 100 → 10000, cap 100000); 200-update warm-up, 50-update ramp — **fully engaged**, since every run reaches update ≥ 1000 |
| Network | Variance-loss coefficient | ramped 0 → 0.05 (same schedule) |
| State | Input variables | Normalized (q, k, w_H−w_L), constant per run (state_dim = 3) |
| Diagnostics | KL window | 20 |
| Diagnostics | KL threshold | mean 0.015 + std 0.012 |
| Diagnostics | Effort-drift threshold | 8.0 |
| Diagnostics | Drift patience | 1 |
| Diagnostics | Exploitability tolerance | 0.03 |
| Diagnostics | Exploitability patience | 5 consecutive checks |
| Diagnostics | Exploitability check interval | every 10 updates |
| MC-BR / Verification | Payoff evaluation | 16384 MC samples with common random numbers |
| MC-BR / Verification | Search grid & coarse-to-fine steps | 5.0, 1.0, and 0.25 |
| MC-BR / Verification | Final check | Independent MC exploitability evaluation after polishing |

---

## Differences only

`--theory-align-v2` changes 12 effective settings at once. A difference in outcome
cannot be attributed to any single one of them.

| Category | Parameter | `r5_sampled_std` | `r5_sampled_v2` |
|---|---|---|---|
| Optimization | Learning rate | 3×10⁻⁴ → 2×10⁻⁴ | 5×10⁻⁵ → 2×10⁻⁵ |
| Optimization | Epochs/update | 6 | 1 |
| PPO / Policy | Clip range | 0.50 → 0.35 | 0.20 → 0.15 |
| PPO / Policy | **Entropy coefficient** | 0.03 → 0.005 (live) | **0 for the entire run** |
| PPO / Policy | KL target | 0.08 | 0.06 |
| PPO / Policy | Ratio-stop threshold | none | 2.2 |
| Network | Policy head | independent softplus α, β | mean × concentration |
| Network | Concentration floor | n/a | 100 → 1000 |
| Network | Concentration scale | n/a | 100 → 10000 |
| Network | Concentration cap | n/a | 100000 |
| Network | Variance-loss coefficient | 0 | 0 → 0.05 |
| Network | Ramp warm-up / steps | n/a | 200 / 50 |

Everything else — optimizer, batch, minibatch, budget, min-updates, seeds, value
coefficient, max-grad-norm, discount, GAE, state encoding, all five cheap-gate
thresholds, and the whole exploitability verifier — is identical.

### Two settings in the v2 block that never take effect

`run/run_different_ability.py:1113-1114` writes `cfg["max_grad_norm"] = 0.25` and
`cfg["value_coef"] = 1.0`, but the `PPOConfig` constructor
(`run/run_different_ability.py:563-581`) never reads either key, so both arms train at
the dataclass defaults `max_grad_norm = 0.5`, `value_coef = 0.5`. Same dead path in
`run/run_two_players.py:1889-1890`, so Table 2's "Value-loss coefficient 0.5" is right
for all four scenarios — but by accident, not by configuration.

---

## Observed outcomes (5 seeds each)

| | q | stop update | final `batch_entropy` | ê_raw (mean ± SD) | per-seed mean \|err\| | \|mean − e*\| |
|---|---|---|---|---|---|---|
| std | 35 | 1023–1219 (verification) | −1.32 … −1.28 | 43.99 ± **0.20** | 2.44 | 2.44 |
| v2 | 35 | 1000–1048 (floor) | −4.42 … −4.32 | 44.93 ± 2.93 | 2.55 | **1.50** |
| std | 55 | 1000–1007 (floor) | −1.32 … −1.25 | 29.70 ± **0.85** | 0.99 | 0.67 |
| v2 | 55 | 1000–1008 (floor) | −4.59 … −4.50 | 30.27 ± 2.77 | 2.23 | **0.10** |

e* = 46.43 (q=35), 30.37 (q=55).

Reading:

- **std is high-bias / low-variance.** All five seeds land on the same side of e*
  (q=35: 43.79–44.31, all below), so per-seed mean |err| equals |mean − e*| exactly.
  That signature is pure bias.
- **v2 is low-bias / high-variance.** Seeds straddle e* (q=35: 41.15–49.07), so the
  bias largely cancels and what remains is scatter.
- Under the paper's own reporting convention (Table 4 reports ê_raw as a cross-seed
  mean and Raw Err. as |mean − e*|), **v2 is the more accurate arm at both q**
  (1.50 vs 2.44; 0.10 vs 0.67). The argument for std rests on cross-seed SD only.
- **Confound:** all 10 v2 runs and both q=55 std runs are floor-triggered — their
  endpoints are cut by `--min-updates 1000`, not by a verification event. How much of
  v2's spread is "stopped by a clock" cannot be separated from the current data.
- **Not yet measured:** `r5_sampled_v2` has never been MC-BR polished
  (`cells_different_ability` globs `*r5_sampled_std*` only). Since the paper reports
  polished values (std q=35 polishes 43.99 → 46.45 ± 0.04, err 0.02), whether the v2
  spread survives polishing is the question that actually decides the arm. Cost to
  answer: 10 polish rows, CPU only, ≈ 21 min.
