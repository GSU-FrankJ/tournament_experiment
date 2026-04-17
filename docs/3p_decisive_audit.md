# 3P Decisive Run Audit

Run: `results/three_players/convergence/ppo_3p_q35.0_seed42_baseline_convergence.json`
Config: warmup=200, min_updates=300, episodes=6144000

## A. Stop reason

1. `stop_reason: max_updates`, `stopped_at_update: 1500`
2. 1500 = 6,144,000 episodes / 4,096 steps_per_update. The run exhausted its episode budget without exploitability triggering a stop.
3. N/A (not exploitability-stopped). `joint_exploit_ok_streak: 0` at termination. Last 5 valid exploit evals: upd 1451=0.0056, 1461=0.0012, 1471=0.0061, 1481=0.0014, 1491=0.0002. Exploit values are low but the streak never reached 5/5 — the drift check kept resetting it.

## B. Policy stability (last 50 updates)

4. `policy_mean_effort` last 50: **mean=24.954, std=0.082, min=24.750, max=25.074**
5. `sample_mean` last 50: mean=24.954, std=0.082. **diff (policy - sample): 0.0000 ± 0.0000**

Note: policy_mean_effort and agent1_effort/agent2_effort are identical in 3P (shared policy, all three agents = same policy mean). The "sample mean" column records the same value.

## C. Numerical health

6. `approx_kl` full run (n=1500): mean=0.0129, std=0.0496. **Negative count: 504/1500 (33.6%)**. min=-0.179, max=0.160. Negatives spread across all phases: upd<500=111, 500-999=182, >=1000=211.
7. `alpha_mean`, `beta_mean`: **MISSING** (3P runner does not record these in convergence JSON).

## D. Best response check

8. `final_br_effort_1`: **None**, `final_br_effort_2`: **None**, `final_br_effort_3`: MISSING. `policy_mean_effort[-1]`: 24.958. BR-policy diff: **cannot compute** (BR fields are null because the final exploit eval did not form a complete streak).

## Verdict

**audit fail** (3 flags):
- neg KL = 33.6% >= 10%
- BR-policy diff = cannot compute (final BR is null)
- stop_reason = max_updates (not convergence-based)

The 0.04 gap (0.17%) is a snapshot at termination, not a convergence-verified result. Policy std=0.082 is clean, but the missing BR check and high negative KL rate are concerns. The negative KL likely stems from the concentration ramp pushing the policy distribution in ways that make the KL approximation unreliable, not from actual divergence — but this needs verification.
