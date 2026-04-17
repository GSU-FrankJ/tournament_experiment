# 3P Stopping Drill-Down

## Data source
`results/three_players/convergence/ppo_3p_q35.0_seed42_baseline_convergence.json` (decisive run, 1500 updates)

## Exploit eval summary

- Total updates: 1500
- Valid exploit evals: 312
- All 309 consecutive evals from update 30 onward: **exploit < 0.03** (100% pass rate)
- Max streak in JSON data: 309
- Exploitability NEVER exceeded 0.03 after update 30

## Eval trigger pattern

| phase | updates | eval count | trigger | gap between evals |
|-------|---------|-----------|---------|-------------------|
| 0-29 | 30 | 3 | periodic (every 10) | 10 |
| 30-219 | 190 | 181 | cheap gate (every 1) | 1 |
| 220-1499 | 1280 | 128 | periodic only (every 10) | 10 |

The cheap gate triggered continuous per-update evals from update 30 to 219. After update 220 (concentration ramp active, KL unstable), the gate stopped passing and evals fell back to periodic-only (every 10 updates).

## Root cause of stopping failure

**Line 1119 resets `exploit_ok_streak = 0` on EVERY update where exploit is not evaluated.**

With periodic evals every 10 updates:
- Update 231: eval runs, passes → streak = 1
- Updates 232-240: no eval → streak = 0, 0, 0, 0, 0, 0, 0, 0, 0
- Update 241: eval runs, passes → streak = 1
- Updates 242-250: no eval → streak = 0, 0, ...

**The streak can never reach 5 with gap=10 between evals.** It hits 1, resets 9 times, hits 1, resets 9 times, forever.

During the gate phase (updates 30-219), evals happened every update (gap=1), so the streak accumulated continuously. The streak reached 181 during that phase. But `min_updates=300` blocked the stop. After the gate stopped triggering at update 220, the periodic-only pattern made it impossible to ever reach 5 again.

## Why the preflight analysis was wrong

The preflight (Q1) identified the correct mechanism (line 1119 resets on non-eval) but diagnosed the wrong failure mode ("gate completely blocks evals"). In reality:
- The gate didn't block evals — periodic evals still ran every 10 updates
- Exploitability was consistently below threshold (all 128 post-gate evals passed)
- The problem is that **the streak reset on non-eval updates makes it structurally impossible to stop with periodic-only evals**

## Fix

Change line 1119: remove the `exploit_ok_streak = 0` from the `else` (non-eval) branch. Only reset the streak when exploit IS evaluated and fails (line 1100-1101).

Current code:
```python
else:
    exploit_ok_streak = 0      # <-- THIS LINE IS THE BUG
    last_exploitability = None
    last_best_dev_effort = None
```

Fix: remove the streak reset, keep the other resets:
```python
else:
    # Don't reset exploit_ok_streak — only reset when eval actually fails.
    # Resetting on non-eval makes it impossible to accumulate streak with
    # periodic (gap=10) evaluations.
    last_exploitability = None
    last_best_dev_effort = None
```

Risk: this changes stopping behavior for ALL runs using periodic-only evals. For the 2P runner, the same line exists but 2P runs typically stop during the gate phase (where evals are every update), so the practical impact is minimal.
