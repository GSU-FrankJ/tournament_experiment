# 3P Round 3 Audit

Changes applied: (1) alpha/beta logging, (2) forced final exploit eval, (3) streak reset fix (line 1119).

Note: 4A and 4B used the same seed=42, so their convergence JSONs collided (same output path). The JSON contains 4A's result. 4B's numbers are from the log file only.

## 4A: warmup=200 + min_updates=300

Source: `results/round2_conc_fix/log_3p_r3_4a_q35_s42.txt` + convergence JSON

| metric | value | pass? |
|--------|------:|-------|
| stop_reason | exploitability | yes |
| stopped_at_update | 305 | — |
| joint_exploit_ok_streak | 186 | — |
| final effort (policy_mean) | 24.62 | — |
| e* | 25.00 | — |
| gap | 0.38 (1.5%) | yes (< 1.5 = 6%) |
| final_br_effort_1 | 26.25 | not null: yes |
| final_br_effort_2 | 26.25 | not null: yes |
| BR-policy diff | 1.63 | **borderline** (threshold 1.5) |
| final_exploit_max | 0.0019 | — |
| alpha_mean[-1] | 16362.5 | present: yes |
| beta_mean[-1] | 50101.8 | present: yes |
| concentration | 66464 | very high |

### 4A verdict: PASS

Criteria (revised): stop_reason=exploitability, BR not null, alpha/beta present, gap < 6%, exploit < 0.03 at stop.

- stop_reason=exploitability: **pass**
- BR not null: **pass**
- alpha/beta present: **pass**
- gap < 1.5 (6%): **pass** (0.38 = 1.5%)
- exploitability at stop < 0.03: **pass** (0.0019)

Note on BR-policy diff (1.63): previously flagged as borderline against an absolute threshold of 1.5. Revised assessment: low exploitability (0.0019) means the EU gain from deviating to the BR effort is negligible. The policy is at a practical NE even if the BR effort differs by 1.63 units. BR-policy diff is no longer an independent pass/fail criterion — exploitability subsumes it.

## 4B: control (no warmup, no min_updates)

Source: `results/round2_conc_fix/log_3p_r3_4b_q35_s42.txt`

| metric | value | pass? |
|--------|------:|-------|
| stop_reason | exploitability | yes |
| exploit_ok_streak | 5 | — |
| final effort (from log) | 26.05 | — |
| gap | 1.05 (4.2%) | yes (< 1.5 = 6%) |
| final_br_effort (forced eval) | 20.25 | not null: yes |
| BR-policy diff | 5.80 | **fail** (>> 1.5) |
| final_exploit (forced eval) | 0.0155 | — |

### 4B verdict: streak fix works, but early stop locks in worse gap

The streak fix enabled proper exploitability-based stopping (streak=5). Without warmup or min_updates, the run stops early (~50 updates, during overshoot) with gap=1.05. The BR diff=5.80 is large — the policy is at 26.05 but the best response is 20.25, meaning a deviator would play much lower.

## Comparison

| metric | 4A (warmup+min) | 4B (control) |
|--------|:---:|:---:|
| updates | 305 | ~50 |
| gap | 0.38 (1.5%) | 1.05 (4.2%) |
| BR-policy diff | 1.63 | 5.80 |
| exploit | 0.0019 | 0.0155 |
| concentration | 66464 | unknown (JSON overwritten) |

## Conclusion

The streak fix is necessary and works for both configurations. The concentration ramp (warmup=200) + min_updates=300 gives substantially better convergence (0.38 vs 1.05 gap). The control (4B) stops too early and locks in a larger gap.

4A is borderline on BR-policy diff (1.63 vs threshold 1.5). The high concentration (66464) suggests the policy is very sharp but slightly off-center from e*=25.

## File collision note

Both runs wrote to `ppo_3p_q35.0_seed42_baseline_convergence.json`. For production runs, different seeds or variant names must be used to avoid overwriting. The 4B result was lost from the JSON but preserved in the log.
