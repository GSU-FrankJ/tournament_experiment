# 3P Fix Preflight — Code Analysis

## Q1: Drift check — what resets the exploit streak?

### What field resets exploit_ok_streak?

Two reset points:

1. **Exploitability eval fails** (`run_three_players.py:1101`): when `exploitability_val >= exploit_eps` (0.03), both `exploit_ok_streak = 0` and `drift_ok_streak = 0`.

2. **Exploit eval not run** (`run_three_players.py:1119`): when `run_exploit = False` (cheap gate didn't trigger AND periodic check not due), `exploit_ok_streak = 0` and `last_best_dev_effort = None`.

### Drift threshold

- `drift_effort_thresh`: **2.0** (default, line 1036)
- `patience_drift`: **2** consecutive passes needed (line 1037)
- Window size: **20** updates (line 772)

### How drift is computed

`CheapGateTracker.compute()` (lines 562-592):
```
drift = abs(policy_vals[-1] - policy_vals[0])
```
Absolute change in `policy_mean_effort` from the oldest to the newest entry in a rolling window of 20 updates.

### Cheap gate decision

`run_exploit` is True when (`run_three_players.py:1057`):
```
run_exploit = periodic_due OR (gate_triggered AND steps_since_last_exploit >= 1)
```

`gate_triggered` requires ALL of:
- `mean_kl_window <= 0.0045`
- `std_kl_window <= 0.0035`
- `drift_effort <= 2.0`
- `drift_ok_streak >= 2` (2 consecutive passes)

### 2P vs 3P comparison

| parameter | 3P | 2P | same? |
|-----------|----|----|-------|
| drift_effort_thresh | 2.0 | 2.0 | yes |
| patience_drift | 2 | 2 | yes |
| window_size | 20 | 20 | yes |
| mean_kl_thresh | 0.0045 | 0.0045 | yes |
| std_kl_thresh | 0.0035 | 0.0035 | yes |
| exploit_eps | 0.03 | 0.03 | yes |
| patience_exploit | 5 | 5 | yes |
| `--disable-cheap-gate` | NO | YES | **different** |

Key difference: 2P has `--disable-cheap-gate` CLI flag that forces `drift_pass = True`, bypassing the gate. 3P does not have this flag.

### Why the streak fails in the decisive run

With concentration ramp active (update 200+), the policy oscillates with KL alternating between positive and negative values (std_kl ~0.05, well above the 0.0035 threshold). The cheap gate blocks most exploit evaluations. On the rare occasions it passes, `exploit_ok_streak` gets 1/5 or 2/5 before the next eval is blocked by the gate, resetting the streak to 0 (line 1119).

### Next action

Add `--disable-cheap-gate` flag to `run_three_players.py` (copy from 2P runner). This would allow exploit evals every `exploit_every_updates` regardless of KL/drift status, enabling the streak to accumulate. Risk: minimal — this is an existing, tested mechanism from 2P. Does not affect existing results (opt-in flag).

---

## Q2: Negative KL root cause

### Formula

`agents/ppo_three_players.py:580`:
```python
approx_kl = (mb_old_logp - logp).mean()
```

This computes `E[log π_old(a|s) - log π_new(a|s)]` = `-E[log(ratio)]`.

Same formula in 2P (`agents/ppo_two_players_clean.py:507`).

### Why it can be negative

This is NOT the standard non-negative KL estimator `E[(ratio-1) - log(ratio)]`. It's the simpler `E[-log(ratio)]` which is negative whenever the new policy assigns higher probability to the sampled actions than the old policy (ratio > 1 on average).

This happens naturally when the concentration ramp increases `conc_min`/`conc_scale` between updates — the new policy's Beta distribution becomes sharper (higher concentration), increasing log-prob for actions near the mode while decreasing it for actions far from the mode. If the rollout actions were sampled from the old (less concentrated) policy, they may cluster near the mode, making the new policy assign higher log-probs on average.

### Numerical safeguards

| safeguard | present? | location |
|-----------|----------|----------|
| action clamping [1e-6, 1-1e-6] | yes | `ppo_three_players.py:360-362` |
| ratio clamping for loss | yes (1±clip_eps) | `:578` |
| ratio clamping for KL | **no** | — |
| KL value clamping | **no** | — |
| NaN/Inf guard | yes | `:558-571` (triggers early stop) |
| conc_max upper bound | optional | `:119-120` (default None = unbounded) |

### Beta distribution numerical stability

PyTorch's `Beta.log_prob` uses `lgamma` internally — numerically stable for high concentration. The action clamping to [1e-6, 1-1e-6] prevents boundary singularities. No known issues at α+β >> 1000.

### Next action

The 33.6% negative KL rate is a property of the estimator formula interacting with the concentration ramp, not a numerical bug. Two options:
1. **No change**: accept that `approx_kl` is a signed quantity with this estimator. The absolute value or the non-negative estimator `E[(ratio-1) - log(ratio)]` would eliminate negatives.
2. **Switch estimator**: change to `((ratio - 1) - torch.log(ratio)).mean()` which is always >= 0. Risk: changes KL-based early stopping thresholds; would affect ALL experiments if applied globally. Not recommended mid-project.

Recommendation: no code change. Document that negative `approx_kl` is expected with the concentration ramp and does not indicate divergence.

---

## Q3: Why BR and alpha/beta weren't recorded

### final_br_effort_* = None

**Root cause**: `last_best_dev_effort` is initialized to `None` (line 778) and only set when exploitability is evaluated (line 1086). It is reset to `None` whenever exploit eval is skipped (line 1121). When `stop_reason = "max_updates"`, there is no final exploitability evaluation — the training loop simply exits. If the last update's cheap gate blocked exploit eval, `last_best_dev_effort` is `None` at JSON write time (lines 1202-1203).

**Code path**:
- Init: `last_best_dev_effort = None` (line 778)
- Set: `last_best_dev_effort = best_dev_effort` only inside `if run_exploit:` (line 1086)
- Reset: `last_best_dev_effort = None` in `else:` block (line 1121)
- Write: `"final_br_effort_1": last_best_dev_effort` (line 1202) → `null`

### alpha_mean / beta_mean missing

**Root cause**: the 3P runner never initializes or populates these fields in `convergence_history`.

3P convergence_history init (lines 814-827): **no `alpha_mean` or `beta_mean` keys**.

3P per-update recording (lines 996-1006): **no alpha/beta extraction or append**.

Compare 2P: init includes `"alpha_mean": [], "beta_mean": []` (lines 868-869), and per-update recording extracts `dist.concentration1.mean()` and `dist.concentration0.mean()` (lines 1152-1153, appended at 1180-1181).

The 3P runner's training loop (around line 988) computes `policy_mean_effort` from `agent.mean_effort(test_state)` but never inspects the distribution's concentration parameters.

### Can we add a hook for unconditional final BR + concentration?

Yes. Two changes needed:

**1. Final BR eval regardless of stop reason** — after the training loop exits, add a forced exploitability evaluation:
```python
# After loop, before JSON assembly (~line 1145)
if last_best_dev_effort is None:
    # Force one final exploit eval to populate BR fields
    exploit_res = evaluate_exploitability(agent, env, ...)
    last_exploitability = exploit_res["exploitability"]
    last_best_dev_effort = exploit_res["best_dev_effort"]
```

**2. Record alpha/beta per update** — add to convergence_history init and per-update block:
```python
# In init: add "alpha_mean": [], "beta_mean": []
# Per update: extract from agent.net distribution
with torch.no_grad():
    dist, _ = agent.dist(s_eval)
    alpha_mean = float(dist.concentration1.mean().item())
    beta_mean = float(dist.concentration0.mean().item())
convergence_history["alpha_mean"].append(alpha_mean)
convergence_history["beta_mean"].append(beta_mean)
```

Risk: both changes are additive (new data fields, forced eval at end). They do not affect training dynamics or existing convergence JSONs. Old JSONs simply won't have the new fields, which the paper generator already handles via `data.get(field, [np.nan] * n)`.

### Next action

Add both (1) forced final BR eval and (2) alpha/beta recording to `run_three_players.py`. Minimal risk — purely additive data collection. Should be done before the full 10-seed batch.
