# Round 2 Metric Revision and Decision Options

## 1. TL;DR

The Round 2 report (`docs/round2_full_results.md`) used a wrong metric for "final effort": it averaged the policy mean over the last 50 updates instead of reporting the policy mean at the final update. Because the effort trajectory is still descending in those last 50 updates, this systematically inflated the reported gap vs theory. With the correct metric (`policy_mean_effort[-1]`), q=45 and q=55 both achieve < 3% relative gap (2.1% and 2.4%), which is substantially better than the 5.3% and 7.3% previously reported. q=35 worsens from 3.4% to 4.3% because some seeds overshoot below e* and the early stop locks in that error.

A decision is needed: accept the current numbers and re-report, or invest time improving q=35.

## 2. Corrected numbers

### Per-q summary

| q | n | mean_rel% (old, last-50 avg) | mean_rel% (corrected, final policy) | < 3%? |
|--:|--:|-----------------------------:|------------------------------------:|-------|
| 35 | 5 | 3.4% | 4.3% | no |
| 45 | 5 | 5.3% | 2.1% | yes |
| 55 | 5 | 7.3% | 2.4% | yes |

### Per-seed detail (corrected metric)

| q | seed | final_effort | e* | abs_gap | rel_gap% |
|--:|-----:|-------------:|-------:|--------:|---------:|
| 35 | 42 | 43.46 | 45.45 | 2.00 | 4.4% |
| 35 | 43 | 44.26 | 45.45 | 1.19 | 2.6% |
| 35 | 44 | 44.34 | 45.45 | 1.11 | 2.4% |
| 35 | 45 | 41.49 | 45.45 | 3.96 | 8.7% |
| 35 | 46 | 46.96 | 45.45 | 1.51 | 3.3% |
| 45 | 42 | 35.82 | 35.35 | 0.47 | 1.3% |
| 45 | 43 | 34.37 | 35.35 | 0.98 | 2.8% |
| 45 | 44 | 33.51 | 35.35 | 1.84 | 5.2% |
| 45 | 45 | 35.48 | 35.35 | 0.13 | 0.4% |
| 45 | 46 | 35.61 | 35.35 | 0.26 | 0.7% |
| 55 | 42 | 29.29 | 28.93 | 0.36 | 1.3% |
| 55 | 43 | 29.60 | 28.93 | 0.67 | 2.3% |
| 55 | 44 | 27.84 | 28.93 | 1.08 | 3.7% |
| 55 | 45 | 28.20 | 28.93 | 0.73 | 2.5% |
| 55 | 46 | 29.49 | 28.93 | 0.56 | 2.0% |

## 3. Why the old metric was wrong

The old metric averaged `policy_mean_effort` over the last 50 updates. Because all trajectories approach e\* from above (effort starts high and descends), this window includes updates where the policy was still well above its terminal value. The average gets pulled up by this converging tail.

Concrete example (q=55, seed=43, e\*=28.93):

- Policy mean at final update: **29.60** (gap 0.67, rel 2.3%)
- Mean of policy mean over last 50 updates: **31.12** (gap 2.20, rel 7.6%)
- The 1.52-unit difference is entirely because updates 40-88 had policy mean in the 30-33 range while the policy only reached 29.60 at update 89.

The diagnostic (`docs/metric_diagnosis.md`, Section 4) confirmed that the dominant error source is this window effect (mean C-B gap: 2.26 for q=45, 2.16 for q=55). The secondary source — sampling noise from rollout actions vs deterministic policy mean — contributes only ~0.12 units across all q values, which is negligible.

## 4. q=35 current state

The 5 seeds for q=35 show the following gaps under the corrected metric:

| seed | abs_gap | rel% | n_updates |
|-----:|--------:|-----:|----------:|
| 42 | 2.00 | 4.4% | 69 |
| 43 | 1.19 | 2.6% | 49 |
| 44 | 1.11 | 2.4% | 49 |
| 45 | 3.96 | 8.7% | 49 |
| 46 | 1.51 | 3.3% | 49 |

Mean gap: 1.96 (std 1.18). The outlier is seed 45 (8.7%). All 5 runs stopped via exploitability criterion (`exploit_eps=0.03`, `patience_exploit=5`, evaluated every 10 updates; from `exploit_config` in `results/two_players/convergence/ppo_q35.0_seed42_convergence.json`). q=35 runs stop at 49-69 updates, versus 59-109 for q=55.

Seed 45 trajectory:

| checkpoint | effort |
|-----------:|-------:|
| t=0 | 45.91 |
| t=25% | 40.81 |
| t=50% | 40.46 |
| t=75% | 42.30 |
| t=final | 41.49 |

The trajectory overshoots below e\*=45.45 by update 25% and never recovers before exploitability triggers a stop. Four seeds (42, 43, 44, 45) end below e\*; seed 45 is the farthest below (41.49, gap 3.96). Only seed 46 ends above e\* at 46.96.

## 5. Decision options

### Option A: Accept current results, re-report numbers

Update the Round 2 report and paper figure plan to use `policy_mean_effort[-1]` as the convergence metric. Report q=35 at 4.3% with a note that the gap is driven by early stopping (exploitability criterion met before the policy fully stabilized). No new experiments.

- Cost: ~30 min of documentation work.
- Expected outcome: q=45 (2.1%) and q=55 (2.4%) meet the < 3% target. q=35 (4.3%) does not, but 3 of 5 seeds are individually below 3.3%.
- Risk: a reviewer may question why the lowest-q (easiest) scenario has the worst accuracy. The mechanistic explanation (early stop) is defensible but not fully satisfying without a fix.

### Option B: Tighten q=35 stopping criterion, re-run

Lower `exploit_eps` from 0.03 to 0.01, or add a `min_updates=100` floor, then re-run the 5 q=35 seeds. This forces the policy to train longer before stopping, giving it time to stabilize near e\*.

- Cost: ~2 hours wall time for 5 q=35 runs. If the stopping change requires regression testing q=45/55, add another 4-6 hours.
- Expected outcome: q=35 rel% likely drops to ~2%, consistent with q=45/55.
- Risk: this is a q=35-specific tuning decision. The paper needs to justify why q=35 uses different stopping parameters, or the same parameters must be applied to all q values (triggering re-runs across the board).

### Option C: Expand to 10-20 seeds per q

Run additional seeds for all q values to determine whether q=35's high variance is a stable property of the problem or a 5-seed artifact.

- Cost: 15-45 additional runs, 1-2 days depending on GPU parallelism.
- Expected outcome: clearer picture of the variance structure. If q=35 variance is inherent, the mean gap may not improve, and the problem returns to Option B.
- Risk: highest time investment. If q=35 is genuinely unstable, the additional data confirms the problem but does not solve it.

## 6. Recommendation

Option A.

The paper's central claim is that PPO agents learn Nash equilibrium effort levels across a range of tournament parameters. q=45 and q=55 already demonstrate this at < 3% relative gap. q=35 at 4.3% is above the informal threshold, but 3 of 5 seeds are individually at 2.4-3.3%, and the mean is pulled up by a single outlier (seed 45, 8.7%). The outlier has a clear mechanistic explanation: exploitability-based early stopping terminates q=35 runs at 49 updates — roughly half the training time of q=55 — before the policy fully stabilizes. This is a stopping criterion artifact, not a failure of the learning algorithm.

Reporting the corrected numbers with this explanation is sufficient for an initial submission. If a reviewer specifically challenges q=35, Option B provides a clean fix that can be executed during rebuttal with ~2 hours of compute.

## 7. Appendix: data sources

- Old report: `docs/round2_full_results.md`
- Metric diagnostic: `docs/metric_diagnosis.md`
- Raw convergence JSONs: `results/two_players/convergence/ppo_q{35,45,55}.0_seed{42-46}_convergence.json`
