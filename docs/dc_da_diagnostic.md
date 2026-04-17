# dc/da Diagnostic Trajectory Analysis

Sources:
- dc: `results/different_cost/convergence/different_cost_ppo_q35.0_seed42_dc_diag_convergence.json`
- da: `results/different_ability/convergence/different_ability_ppo_q35.0_seed42_da_diag_convergence.json`

## Section 1: Final state

| scenario | agent | final_effort | e* | gap | rel% | stop_reason | n_updates |
|----------|-------|------------:|---:|----:|-----:|-------------|----------:|
| dc q=35 | agent1 | 36.93 | 38.03 | 1.10 | 2.9% | exploitability | 35 |
| dc q=35 | agent2 | 26.84 | 27.66 | 0.81 | 2.9% | exploitability | 35 |
| da q=35 | shared | 43.16 | 46.43 | 3.27 | 7.0% | exploitability | 32 |

## Section 2: Baseline comparison

From `paper/tables/final_summary.tex` (Metric B, current):
- Het. Cost q=35: RelErr = 4.44%
- Het. Ability q=35: RelErr = 6.83%

| scenario | CSV rel% | diagnostic rel% | diff | consistent? |
|----------|---------|----------------|------|-------------|
| dc q=35 | 4.44% | 2.9% (max gap) | -1.5% | close — CSV averages 5 seeds |
| da q=35 | 6.83% | 7.0% | +0.2% | yes |

dc diagnostic is better than the CSV mean (2.9% vs 4.44%). This is expected — a single seed can vary. The CSV mean includes 5 seeds with higher variance. No evidence of a silent bug.

## Section 3: Effort trajectory

### dc (agent1 / agent2)

| checkpoint | upd | agent1 | agent2 | note |
|-----------|----:|-------:|-------:|------|
| 0% | 0 | 45.36 | 48.05 | initial (both start high) |
| 11% | 4 | 24.70 | — | a1 minimum (overshoot) |
| 17% | 6 | — | 18.84 | a2 minimum (overshoot) |
| 25% | 9 | 33.80 | 24.16 | recovering |
| 50% | 18 | 35.52 | 25.97 | approaching |
| 75% | 27 | 36.46 | 26.30 | near e* |
| 100% | 35 | 36.93 | 26.84 | final |
| e* | — | 38.03 | 27.66 | theory |

Both agents overshoot significantly by update 4-6 (a1 drops to 24.70, a2 to 18.84), then recover. Agent1 recovers more (reaches 36.93 vs target 38.03) than agent2 (26.84 vs 27.66). The asymmetric cost (k1=0.0004, k2=0.00055) means different gradient magnitudes.

### da (shared policy)

| checkpoint | upd | effort | note |
|-----------|----:|-------:|------|
| 0% | 0 | 44.98 | initial |
| 9% | 3 | 33.62 | minimum (overshoot below e*) |
| 25% | 8 | 39.55 | recovering |
| 50% | 16 | 42.13 | approaching |
| 75% | 24 | 42.43 | plateau |
| 100% | 32 | 43.16 | final |
| e* | — | 46.43 | theory |

da overshoots to 33.62 at update 3, then recovers but plateaus around 42-43, stopping 3.27 below e*. The recovery stalls — effort at 50% (42.13) barely improves by 75% (42.43) and 100% (43.16).

## Section 4: Stopping analysis

### dc

- stop_reason: **exploitability** (streak=5)
- **exploit_eps = 0.05** (not 0.03 — config default is 0.05 for dc/da)
- Stopped at update 35

Last 5 exploit evals:

| upd | exploit_1 | exploit_2 | exploit_max | BR1 | BR2 | streak |
|----:|----------:|----------:|------------:|----:|----:|-------:|
| 29 | 0.0397 | 0.0275 | 0.0397 | 37.25 | 28.25 | 0 |
| 32 | 0.0377 | 0.0237 | 0.0377 | 37.50 | 28.25 | 1 |
| 33 | 0.0366 | 0.0274 | 0.0366 | 37.50 | 28.75 | 2 |
| 34 | 0.0336 | 0.0255 | 0.0336 | 37.00 | 27.00 | 3 |
| 35 | 0.0335 | 0.0248 | 0.0335 | 37.00 | 27.50 | 4 |

All exploit_max values are 0.034-0.040 — above 0.03 but below the 0.05 threshold. With a 0.03 threshold, this run would NOT have stopped (streak would reset every eval).

### da

- stop_reason: **exploitability** (streak=5)
- **exploit_eps = 0.05** (same as dc)
- Stopped at update 32

Last 5 exploit evals:

| upd | exploit_1 | exploit_2 | exploit_max | BR1 | BR2 | streak |
|----:|----------:|----------:|------------:|----:|----:|-------:|
| 19 | 0.0499 | 0.0253 | 0.0499 | 44.25 | 45.50 | 0 |
| 29 | 0.0408 | 0.0250 | 0.0408 | 44.25 | 45.75 | 1 |
| 30 | 0.0375 | 0.0230 | 0.0375 | 44.50 | 45.75 | 2 |
| 31 | 0.0362 | 0.0198 | 0.0362 | 45.00 | 45.25 | 3 |
| 32 | 0.0401 | 0.0219 | 0.0401 | 44.75 | 46.75 | 4 |

Same pattern — exploit_max 0.036-0.050, below 0.05 threshold but well above 0.03.

## Section 5: Concentration trajectory

**MISSING** for both dc and da. Neither runner records `alpha_mean`/`beta_mean` in convergence history (same issue as 3P before the logging fix).

## Section 6: Best response signal

### dc

| agent | policy | BR | diff |
|-------|-------:|---:|-----:|
| agent1 | 36.93 | 37.00 | 0.07 |
| agent2 | 26.84 | 27.50 | 0.66 |

BR-policy diff is small for both agents. The policy is near the best response.

### da

| agent | policy | BR | diff |
|-------|-------:|---:|-----:|
| player1 | 43.16 | 44.75 | 1.59 |
| player2 | 43.16 | 46.75 | 3.59 |

BR-policy diff is large, especially for player2 (3.59). The shared policy at 43.16 is suboptimal for player2 whose BR is 46.75 (close to e*=46.43). Player1's BR (44.75) is also above the current policy.

Both BRs point upward (toward e*), confirming the policy has undershot and stopped too early.

## Section 7: Candidate hypotheses

### Hypothesis 1: Loose exploit_eps threshold (0.05) causes premature stopping

dc and da both use `exploit_eps=0.05` (vs 0.03 for 2P/3P). This allows stopping when exploit_max is 0.034-0.050 — enough to declare "near-NE" when the policy is still 3-7% from e*. With 0.03 threshold, neither run would have stopped at update 32-35.

- Supporting evidence: exploit_max at stop is 0.034 (dc) and 0.040 (da) — both above 0.03
- Counter-evidence: none
- Verification: **can verify with existing data** — check if all 5-seed CSV runs also stopped with exploit 0.03-0.05

### Hypothesis 2: No concentration ramp → policy freezes too early

dc/da don't have the concentration ramp ported from 2P. Without the ramp, the policy's Beta distribution may lock into high concentration early (like the original 3P problem). This would prevent further movement toward e* even if more updates were allowed.

- Supporting evidence: alpha/beta MISSING so can't verify concentration levels. da's effort plateau (42.13 at 50% → 43.16 at 100% = +1.03 in 16 updates) suggests slowing.
- Counter-evidence: dc's gap is only 2.9%, suggesting the ramp may not be necessary for dc
- Verification: **needs new experiment** — port concentration ramp to dc/da and compare

### Hypothesis 3: da has asymmetric BR landscape that a shared policy can't navigate

da uses a shared policy for two players with different abilities (l1=10, l2=5). The BR efforts differ: player1 wants 44.75, player2 wants 46.75, theory says 46.43. The shared policy at 43.16 is a compromise that doesn't satisfy either player's BR well (especially player2). The gap may be inherent to the shared-policy architecture.

- Supporting evidence: BR1=44.75 vs BR2=46.75 (2.0 unit spread); policy at 43.16 is below both
- Counter-evidence: theory predicts a single symmetric equilibrium at 46.43, so both BRs should converge there if the policy reaches it
- Verification: **can verify with existing data** — check if BR1 and BR2 converge as policy approaches e*. If they stay 2+ units apart, the shared policy may be fundamentally limited.
