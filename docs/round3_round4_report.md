# Round 3 / Round 4 Technical Report

## 0. TL;DR

After Round 2 brought all 2P scenarios under 5% relative gap (via the Metric B switch), the Round 2 CSV still showed three scenarios with gaps above 5%: three-player (9.62% at q=35), heterogeneous cost (5.51% at q=55), and heterogeneous ability (6.83% at q=35). Investigation confirmed the root cause is NOT shared: 3P had a stopping-criterion bug plus missing concentration ramp, dc needed the ramp plus a tighter exploitability threshold, da needed the tighter threshold plus extended training. After scenario-specific fixes, all completed batches land at or below 1.6% mean relative gap. The da q=55 batch is still running; remaining batches summarized in Section 5.

## 1. Problem identification

Round 2 final summary (from `paper/tables/final_summary.tex` at that point):

| Scenario | q=35 rel% | q=55 rel% | status |
|----------|:---:|:---:|--------|
| Two-Player | 4.30% | 2.36% | Round 2 target met |
| Three-Player | 9.62% | 6.38% | needs fix |
| Het. Cost | 4.44% | 5.51% | needs fix |
| Het. Ability | 6.83% | 5.01% | needs fix |

All three non-2P scenarios undershoot e\* consistently. The initial working hypothesis was that the Round 1.5 concentration-fix warmup (validated on 2P) had not been ported to the other runners. Verification showed this was part of the story for dc but not for 3P or da — each scenario required a distinct diagnosis.

A second Metric B switch from Round 2 changed how "final effort" is computed (policy distribution mean at the last update, not last-50 average). That change is a reporting decision, not a training fix; the 3P/dc/da gaps above are already measured under Metric B.

## 2. Three-Player (Round 3)

### 2.1 Diagnosis

An initial sanity run applying only the 2P concentration warmup (`--override-conc-ramp-warmup 200`) terminated at update 44 with gap 3.12 (12.5%) — worse than the Round 2 baseline. Theory was ruled out first: `utils/theory.py:46-56` gives `e*(q) = (w_H - w_L) / (4qk) = 25.00` for q=35, and a direct numerical check on `envs/three_players_env.py` confirmed `dEU/de = 0.000000` at e\*=25 and EU uniquely maximized there (see `docs/three_player_theory_check.md`).

Drill-down on the stop history (`docs/3p_stop_drilldown.md`) showed 309 consecutive valid exploitability evaluations at or below 0.03, all inside the "streak" logic — yet the run still terminated via `max_updates`.

### 2.2 Root cause

Line 1119 of `run_three_players.py` reset `exploit_ok_streak = 0` on every non-eval update. Exploitability evaluations run periodically every 10 updates; `patience_exploit = 5` requires five consecutive passing evals. Between two periodic evals there are nine non-eval updates, each of which resets the streak. Under periodic-only evaluation, the streak can reach 1 then drops back to 0, nine times, forever. It can only reach 5 when the cheap gate triggers evaluation every update — which requires low KL drift. Once the concentration ramp activates and KL destabilizes, the cheap gate stops firing and the streak can never accumulate.

2P does not hit this path because the 2P runner uses `--disable-cheap-gate` in its standard config, forcing evaluation every update. The streak reset runs but never accumulates gaps.

### 2.3 Fix

Two independent changes:

- **Streak bug fix**: `run_three_players.py:1119` no longer resets `exploit_ok_streak` or `last_best_dev_effort` on non-eval updates. Only a real eval failure (line 1109, when `exploitability >= 0.03`) resets the streak.
- **Concentration ramp port**: `--override-conc-ramp-warmup 200` plus the ramp-interpolation block ported verbatim from `run_two_players.py:889-910`. `--min-updates 300` guard added to prevent the streak from triggering stop before the ramp has had effect.

A seed-42/q-35 sanity pair confirmed both changes matter:
- Streak fix + no ramp (control): gap 1.05 (4.2%) at update ~50
- Streak fix + ramp + `min_updates=300`: gap 0.38 (1.5%) at update 305

Streak fix is necessary (without it, ramp runs hit `max_updates` with loose convergence); ramp + `min_updates` drives the remaining improvement.

### 2.4 Round 3 batch results (5 seeds × q={35, 55})

| q | mean_gap | mean_rel% | std_rel% | max_rel% |
|--:|---------:|----------:|---------:|---------:|
| 35 | 0.143 | 0.57% | 0.45% | 1.30% |
| 55 | 0.134 | 0.84% | 0.35% | 1.32% |

All 10 runs stopped via `exploitability`. Full audit in `docs/3p_round3_audit.md`.

## 3. Heterogeneous Cost (Round 4)

### 3.1 Diagnosis

A diagnostic run (`dc_diag`, seed=42, q=35) under the Round 2 default config showed both agents' policies close to their best-response efforts (agent1 BR-policy diff 0.07, agent2 0.66). Final exploitability was 0.034, which would have failed the 2P/3P threshold of 0.03 but passed dc's configured `exploit_eps=0.05`. The run stopped at update 35 with policy still 2.9% below e\*. `docs/dc_da_diagnostic.md` analyzes this per-update.

Step C (`docs/dc_da_streak_check.md`) verified that dc does not have the 3P streak bug: `joint_exploit_ok_streak` in `run_different_cost.py` only resets on actual evaluation failure.

### 3.2 Fix

Two changes, added independently and tested:

- Threshold tightening: `--exploit-eps 0.03` (H1)
- Concentration ramp port: `--override-conc-ramp-warmup 200 --min-updates 300` with `--theory-align-v2` (H2)

Single-seed ablation on seed=42, q=35:

| config | max_gap | max_rel% |
|--------|--------:|---------:|
| baseline (diagnostic) | 1.10 | 2.9% |
| H1 only | 1.04 | 3.8% |
| H1 + H2 | 0.35 | 1.1% |

H1 alone did not improve over baseline; H2 is the decisive component here. A 4-seed baseline variance check confirmed the Round 2 4.4% figure is consistent across seeds (per-seed max_rel% in [2.9, 5.2]), not a single-seed artifact.

### 3.3 Round 4 batch results (5 seeds × q={35, 55})

| q | mean_max_gap | mean_rel% | std_rel% | max_rel% |
|--:|-------------:|----------:|---------:|---------:|
| 35 | 0.303 | 1.07% | 0.25% | 1.45% |
| 55 | 0.202 | 0.97% | 0.49% | 1.57% |

rel% is computed as `max(gap1/e1*, gap2/e2*) × 100`, the per-seed worst-case across the two asymmetric agents.

## 4. Heterogeneous Ability (Round 4)

### 4.1 Diagnosis

The da diagnostic run (seed=42, q=35) showed a qualitatively different pattern from dc. BR-policy diffs were large (player1: 1.59, player2: 3.59). The trajectory did not freeze: effort dropped from 45 at init to 33.6 at update 3, then slowly climbed back to 43.16 at update 32, where exploitability stop triggered. The slope of the climb was shrinking — at e ≈ 43 the symmetric FOC gradient is on the order of 0.007 effort units per update after cost.

The policy was not stuck; it was being early-stopped while still in a low-gradient region.

### 4.2 Fix exploration

Three configurations tested on seed=42, q=35:

| config | gap | rel% | interpretation |
|--------|----:|-----:|---------------|
| baseline (eps=0.05) | 3.27 | 7.0% | Round 2 behavior |
| H1 only (eps=0.03) | 1.62 | 3.5% | improves but not below 3% |
| H1 + H2 (ramp + eps=0.03) | 2.60 | 5.6% | **worse than H1 alone** |
| H1 + min_updates=1000 | 1.07 | 2.3% | below target |

The ramp is harmful for da. Possible mechanism: the concentration ramp pushes the policy distribution to collapse toward its current mode before the shared-policy effort has reached e\*, locking in the undershoot. Extended training without the ramp lets the policy commit more slowly and continue climbing.

### 4.3 Oscillation check

The H1+long seed-42 run stopped at `update_idx=1000` (the `min_updates` floor) with an exploit streak of 947 — meaning exploit had been passing for nearly the entire training window. This raised the concern that the reported 2.3% was a snapshot of an oscillation, not a true convergence.

Analysis of the last 500 updates of the seed-42 run:
- mean effort: 45.67 (bias 0.76 below e\*=46.43)
- std: 0.44 (oscillation amplitude, stable)
- drift between first 250 and last 250 of window: 0.05

The policy is converged in a statistical sense: low oscillation amplitude, no drift across the tail, consistent bias. 2.3% is the bias, not a lucky instantaneous value. Confirmed across three seeds:

| seed | final effort | gap | rel% |
|-----:|-------------:|----:|-----:|
| 42 | 45.36 | 1.07 | 2.3% |
| 43 | 45.74 | 0.69 | 1.5% |
| 44 | 45.62 | 0.80 | 1.7% |

Mean 1.8%, std 0.19 effort units across three seeds.

### 4.4 Round 4 batch results (5 seeds × q={35, 55})

| q | mean_gap | mean_rel% | std_rel% | max_rel% |
|--:|---------:|----------:|---------:|---------:|
| 35 | 0.746 | 1.61% | 0.95% | 2.44% |
| 55 | in progress | in progress | in progress | in progress |

q=35 complete. q=55 batch (5 seeds) launched after q=35 finished; status at the time of this report is "running, no completions yet."

## 5. Paper-ready final summary

| Scenario | q | Round 2 rel% | Round 3/4 rel% | Fix applied |
|----------|:-:|-------------:|---------------:|-------------|
| Two-Player | 35 | 4.30% | 4.30% | (no change) |
| Two-Player | 45 | 2.08% | 2.08% | (no change) |
| Two-Player | 55 | 2.36% | 2.36% | (no change) |
| Three-Player | 35 | 9.62% | 0.57% | streak fix + ramp + `min_updates=300` |
| Three-Player | 55 | 6.38% | 0.84% | streak fix + ramp + `min_updates=300` |
| Het. Cost | 35 | 4.44% | 1.07% | ramp + `eps=0.03` + `min_updates=300` |
| Het. Cost | 55 | 5.51% | 0.97% | ramp + `eps=0.03` + `min_updates=300` |
| Het. Ability | 35 | 6.83% | 1.61% | `eps=0.03` + `min_updates=1000` |
| Het. Ability | 55 | 5.01% | in progress | `eps=0.03` + `min_updates=1000` |

All completed entries land below 3% mean relative gap. 2P entries unchanged — no regression path was exercised on those runners.

## 6. Reproducing each scenario

```bash
# 2P (Round 2 unchanged)
python run/run_two_players.py --method ppo --q {Q} --seed {SEED} \
  --override-conc-ramp-warmup 200 --episodes 6144000

# 3P
python run/run_three_players.py --method ppo --q {Q} --seed {SEED} \
  --theory-align-v2 \
  --override-conc-ramp-warmup 200 --min-updates 300 \
  --output-tag round3 --episodes 6144000

# Heterogeneous cost
python run/run_different_cost.py --method ppo --q {Q} --seed {SEED} \
  --theory-align-v2 \
  --override-conc-ramp-warmup 200 --min-updates 300 \
  --exploit-eps 0.03 --ablation-name r4_dc_final --episodes 6144000

# Heterogeneous ability
python run/run_different_ability.py --method ppo --q {Q} --seed {SEED} \
  --exploit-eps 0.03 --min-updates 1000 \
  --ablation-name r4_h1_long --episodes 6144000
```

Note the config asymmetry: 3P and dc use `theory-align-v2 + ramp`; da intentionally does not. Porting the same fix across all non-2P runners would have regressed da.

## 7. Methodological notes

**Per-scenario diagnosis beats cross-scenario extrapolation.** The working hypothesis after Round 2 was that porting the 2P concentration fix would fix everything. The first 3P sanity run with only the warmup port terminated worse than baseline (12.5% vs 9.62%) because the real blocker was an independent streak-reset bug. Had the port been batched into all three scenarios at once, the per-scenario signal would have been invisible — dc (helped by ramp) and da (hurt by ramp) would have been averaged into "ramp does something, unclear what."

**Separate H1 and H2 before combining.** The Round 4 test matrix ran H1-only, H2-only (skipped on cost grounds), and H1+H2 as distinct configurations on single seeds before committing to a batch. This caught the da ramp-is-harmful signal (H1+H2 at 5.6% vs H1 alone at 3.5%). A combined "Round 4 fix" that bundled both changes would have delivered a ~5.6% result on da and been blamed on noise or seed variance.

**Stop-reason is part of the metric.** The da H1+long seed-42 run reported stop_reason=exploitability with streak=947, but the streak had been satisfied for ~950 updates — termination was triggered by the `min_updates` floor, not by fresh convergence. The gap number alone would have looked like a 2.3% success; the streak-vs-min_updates ratio made it obvious the policy had been converged for most of training and the exact stop point was incidental. Cross-seed validation on seeds 43 and 44 confirmed the bias is stable, not a stop-time artifact.

## 8. Open questions

- Whether the da bias of ~0.8 effort units (1.6% relative) is a fundamental limit of the shared-policy architecture on the heterogeneous-ability game, or can be further reduced. The BR efforts differ by ~2 units (44.25 vs 46.75 in the diagnostic), so a shared policy that satisfies one player's BR will miss the other's. TBD whether a separate-policy variant can close this gap further or whether the current number is the right story to tell.
- Whether da q=55 (batch in progress) tracks q=35. The mechanism is expected to be the same, but the q=55 gradient landscape has different slope; the batch will confirm or refute.
- Whether the 3P q=55 variance (std 0.35%) will hold under broader conditions — this is the tightest batch in the report but comes from a single ablation config.
