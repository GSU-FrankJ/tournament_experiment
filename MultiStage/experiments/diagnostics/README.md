# Archived T=3 diagnostic studies

These q=60 studies explain which ideas were tried after the three-stage implementation pilot, what failed, and why the investigation changed direction. They are **diagnostic evidence, not formal results or an estimate of a method's success rate**. This archive packages existing results; it introduces no experiment, threshold change, or new decision requirement.

## What belongs in this history

Keep a trial when it records a scientific question, the intervention and comparison, the actual seeds and budget, a measured outcome, or a stopping decision needed to interpret later work. Keep negative outcomes and unresolved limitations alongside improvements. Exclude material whose only purpose is operating an assistant or machine: conversations, prompts, memory, session notes, launch transcripts, and server-specific paths.

The archive therefore contains this English account, original machine-readable decisions and compact result tables, plus selected tables from the original reports in their original language. It omits raw trajectories, per-sample gradient tensors, model weights, optimizer checkpoints, figures, and study-specific runner code. **It is not a complete runnable reproduction package.** Historical run names and checkpoint filenames in data identify omitted source artifacts; they are not paths readers should expect to resolve here.

## Provenance and common definitions

The sampling studies were recorded on 2026-09-24 in source experiment three_stage_implementation_pilot_20260924. The noise audit and representation fit were recorded on 2026-09-25 in endgame_reward_noise_audit_20260925. The budget study is named q60_phase_a_budget_extension_20260925; its three runs completed on 2026-09-26 UTC. Publication curation occurred on 2026-09-27.

All use q=60, high/low prizes 6 and 2, prize difference ΔW=4, effort cost k=1/3500, effort range [0,100], B=220, and stage-3 domain D3=[−440,440]. The center is |d|≤220. The reported stage-3 deviation R3 is the maximum best-response gain divided by ΔW, against the same checkpoint's mean-action policy at the opponent's signed state −d. Representation-fit losses instead use a **fixed A400 mean opponent**. The full A/B/C study's dReach is the protocol's reach-based three-stage deviation, not R3.

The historical 0.01 deviation and 0.04 normalized-standard-deviation values retain their original meanings. A-only or supervised-fit diagnostic values are not candidate or certification decisions. ES denotes exploring starts; BR denotes best response. Final-tier stage-3 measurements use state spacing 2 and effort spacing 0.5; development-tier values are identified separately.

| Study | Actual study seeds | Design | Recorded disposition |
|---|---|---|---|
| [Phase A sampling](phase_a_sampling/REPORT_TABLES.md) | 10431, 10432, 10433; smoke 10430 | Three paired A400 runs per arm | 3/3 endpoint improvements; advance only to diagnostic A/B/C validation |
| [Full A/B/C validation](sampling_abc/REPORT_TABLES.md) | 10441, 10442, 10443; smoke 10440 | Fresh paired runs from initialization, A400/B600/C1800 | Stop the sampling change; neither arm has a candidate or certification |
| [Terminal-reward noise audit](reward_noise/REPORT_TABLES.md) | Checkpoints from baseline seeds 10441–10443 | Fixed weights, A400 and C minimum; 64×512 samples per checkpoint | Original decision: do not enter Step 2; **K8 PPO training never ran** |
| [Representation fit](representation_fit/REPORT_TABLES.md) | A400 checkpoints from baseline seeds 10441–10443 | Fixed-opponent supervised fits; 8,000 steps primary, 80,000 descriptive | Only full-network fitting succeeds at the primary budget |
| [Phase A budget extension](phase_a_budget/REPORT_TABLES.md) | 10461, 10462, 10463 | Three uninterrupted original-PPO A800 trajectories; each A400 is its own reference | Stop at A800; 2/3 improve, 1/3 remains flat, none reaches R3≤0.01 |

Seeds 10441–10443 are reused as fixed source checkpoints for the two offline diagnostics, not independent new training runs. The A800 seeds are fresh; no adverse seed was replaced or dropped.

## Phase A sampling: improvement did not establish state-dependent learning

Each A400 update used 512 stage-3 starts. The baseline sampled uniformly across D3; the center arm used 256 uniform D3 starts plus 256 uniform center starts, raising expected center exposure from 50% to 75% while retaining full-domain coverage. Each pair shared initialization and RNG stream seeds. PPO settings, A400 budget, check cadence, concentration settings, verifier tiers and thresholds stayed unchanged.

| Seed | Baseline final-tier R3 | Center final-tier R3 |
|---|---:|---:|
| 10431 | 0.1502 | 0.1204 |
| 10432 | 0.1421 | 0.0840 |
| 10433 | 0.1478 | 0.1300 |

The recorded rule called this consistent improvement (3/3), with no concentration flag. However, outer-region error and concentration increased in all pairs. In two pairs, the reduction mainly reflected an upward shift of a nearly constant effort policy, not learning the BR shape. The center arm still had R3=0.084–0.130, and maximum deviation usually increased between A100 and A400. This justified testing carryover, not promotion to formal evaluation.

Evidence: [decision](phase_a_sampling/decision.json), [per-run values](phase_a_sampling/runs.csv), [paired values](phase_a_sampling/pairs.csv), [development trajectory](phase_a_sampling/trajectory.csv), [policy shapes](phase_a_sampling/policy_shape_trajectory.csv). The latter includes explicitly labeled historical pilot-reference rows; those are not paired controls.

## Full A/B/C validation: the sampling change was stopped

The six runs began from fresh initialization. Only Phase A sampling differed; B/C mixtures and all protocol settings were inherited unchanged. Final certification was the primary endpoint. A stage-3 advantage counted as retained only when the center arm was lower in all three pairs at the specified checkpoint.

| Seed | Minimum valid C dReach/ΔW, baseline | Minimum valid C dReach/ΔW, center | Final dReach/ΔW, baseline / center |
|---|---:|---:|---:|
| 10441 | 0.0287 | 0.0396 | 0.0455 / 0.0660 |
| 10442 | 0.0389 | 0.0190 | 0.0472 / 0.0237 |
| 10443 | 0.0151 | 0.0218 | 0.0252 / 0.0250 |

Both arms had **0/3 candidates and 0/3 certifications**; certified/candidates is N/A, not zero percent. All six exited B at its budget cap, with zero eligible B calls, then reached C1800/global update 2800. Each made 2,800 updates and 2,815,400 environment steps. All final two-tier verifier calls were valid and passed refinement and dense concentration checks; all failed the main deviation test.

The center arm lowered stage-3 maximum deviation in 2/3 pairs at A400, 1/3 at B exit, 1/3 at C minimum, and 2/3 at the actual stop. Minimum C dReach improved in only 1/3 pairs. Thus the recorded rule stopped the intervention: no retained advantage, no candidate/certification gain, and no consistent diagnostic-only improvement. q50 and formal runs were not started for this change. Historical C minima remain diagnostics, never retrospectively selected candidates. Three pairs cannot resolve a small treatment effect or estimate a reliable success rate.

Evidence: [decision and original interpretation rules](sampling_abc/decision.json), [run outcomes](sampling_abc/runs.csv), [pairs](sampling_abc/pairs.csv), [checkpoint comparisons](sampling_abc/checkpoints.csv).

## Reward-noise audit: variance reduction observed; training benefit untested

This was an offline audit of the baseline A400 and C-minimum checkpoints. C minima occurred at global updates 1550, 2725 and 2500 for seeds 10441, 10442 and 10443. A400 opponents were reconstructed as copies of the actors at the saved snapshot; C-minimum opponents came from the saved checkpoints and were not always equal to the actors.

Each checkpoint used 64 batches of 512 uniform-D3 stage-3 starts, with sampled actions from both policies. K1 used one terminal prize draw; K8 averaged eight prize draws for the same state/actions, sharing the first draw with K1 and charging effort cost once. A closed-form reward was an offline reference. No optimizer update occurred. Audit RNG streams were independent of training and used common random numbers across a seed's two checkpoints.

At A400 in the center, terminal noise accounted for approximately 97–100% of raw-gradient variance (sampling estimates can slightly exceed 100%). K8 reduced raw-gradient variance to 7–15% of K1 across all three seeds; C-minimum ratios were 9–23%. This did not establish an improvement in actual PPO learning.

The original progression rule required every mathematical/implementation check to pass and, for every A400 seed, noise share φ≥0.25, variance ratio ρ≤0.80, and better normalized-surrogate leave-one-out directional alignment under K8. Two items failed:

- Seed 10442 A400: the K8 gradient-noise check gave z=−4.60 against the prespecified |z|≤4 criterion.
- Seed 10443 A400: directional alignment was 0.233 for K8 versus 0.266 for K1.

**The original decision remains enter_step2=false. The planned paired K1/K8 A-only PPO training, including its runner, training smoke and manifests, was never executed.** Offline K8 sample collection is not a K8 training run.

Post-hoc analysis, kept separate from the decision, found that the empirical-SE check was unstable for the nearly rank-one, heavy-tailed gradient statistic: using the exact conditional variance changed the first failure to z=−2.16. It also found an unresolved excess of large K8 batches at seed 10442 C minimum (Fisher p≈0.00086; approximately 0.01 after 12-cell Bonferroni correction). No extra audit batches were added to obtain a passing outcome; no code bug was established.

The naive SE for the directional comparison does not fully account for shared leave-one-out references and common states/actions. Failure of that directional condition is not evidence that K8 has no effect. C-minimum audit batches also differ from real C-phase mixed trajectories, minibatches and repeated Adam updates. Near-opposing center/outer expected gradients at A400 are a correlation, not proof of inadequate capacity or the cause of PPO failure.

Evidence: [study design](reward_noise/study.json), [unchanged decision values](reward_noise/decision.json), [mathematical checks](reward_noise/math_checks.json), [derived gradient metrics](reward_noise/derived.csv), [reward/advantage metrics](reward_noise/reward_advantage.csv). Files under [posthoc](reward_noise/posthoc/README.json) retain their post-hoc status.

## Representation fit: the target is expressible, but primary-budget failures remain

The three A400 actors were fit to the stage-3 BR against their own frozen A400 mean opponents. Training states were −439,−437,…,439 (440 points); evaluation states were −440,−438,…,440 (441 interleaved points). Deterministic full-batch effort MSE used Adam at 3e−4 with fresh optimizer state and no gradient clipping.

The primary comparison was an 8,000-step head-only fit (65 parameters) versus an 8,000-step full hidden-layer/mean-head fit (4,418 parameters), starting from A400. The concentration head remained fixed. Continuation to 80,000 steps and a full fit from the original PPO initialization were descriptive.

| Seed | Original max loss/ΔW | Head-only at 8,000 | Full at 8,000 | Full from initialization at 8,000 |
|---|---:|---:|---:|---:|
| 10441 | 0.1423 | 0.0288 | 0.0005 | 0.0850 |
| 10442 | 0.1465 | 0.1449 | 0.0002 | 0.1458 |
| 10443 | 0.0763 | 0.0153 | 0.0008 | 0.1464 |

Only full fitting from A400 passed the original ≤0.01 criterion at the primary budget, first doing so at 900/650/650 steps. This supports capacity to express this fixed-opponent stage-3 target. It does not prove adequacy for moving-opponent self-play or stages 1–2.

Head-only fits for 10441 and 10443 eventually passed in the descriptive extension but required large weights; 10442 still failed at 80,000 steps (0.1376). All three initialization fits failed at 8,000 steps despite noiseless labels; first success came at 9,500/11,500/14,500 steps. These plateaus suggest another possible mechanism, not a causal diagnosis of PPO. Supervised steps must not be converted into a PPO budget.

Fitted normalized policy standard deviations reached about 0.044–0.050 with the concentration head fixed. Passing payoff fit therefore did not establish protocol concentration compliance. These fitted policies are not equilibrium candidates, and BR labels were not introduced into PPO. The noise-audit decision and anomaly remained unchanged.

Evidence: [study](representation_fit/study.json), [decision](representation_fit/decision.json), [all primary and extension endpoints](representation_fit/summary.csv).

## Phase A budget: two trajectories developed structure, one did not

Only the original Phase A cap changed, from 400 to 800 updates. Each seed ran continuously from its own initialization, with no reset of optimizer, critic, RNG or opponent at A400. Sampling remained uniform D3, with 512 stage-3 starts/update and K1 sampled reward. The network remained 2-64-64, with original initialization, Adam 3e−4, 10 epochs, minibatch 256, advantage normalization and gradient clipping 0.5. No center sampling, K8, BR supervision or larger network was introduced.

| Seed | R3 at A400 | R3 at A800 | Difference | Interpretation |
|---|---:|---:|---:|---|
| 10461 | 0.1342 | 0.1423 | +0.0081 | Remained essentially flat |
| 10462 | 0.1264 | 0.0514 | −0.0750 | Developed state structure; improved through A800 |
| 10463 | 0.1480 | 0.0856 | −0.0624 | Developed state structure; best at A700 (0.0775), then worsened slightly |

All runs completed 800 updates, 409,600 stage-3 episodes/transitions and 16,000 actor optimizer steps. All 24 checkpoint evaluations were valid. The primary endpoint was A800; historical minima were descriptive.

Seeds 10462/10463 increased center effort ranges from 6.7/4.3 to 23.2/16.6, while seed 10461 remained flat (range ≤0.4). None reached R3≤0.01. Stage-3 concentration remained ≤0.04; all-stage concentration readings involve untrained stages and should not be interpreted as a complete-protocol result.

The study stopped at A800. B/C continuation, A1600, new initialization and K8 comparisons were not run. Candidate discovery, full certification and three-stage success were **not evaluated**, rather than being 0/3 outcomes. These three trajectories show possible escape and possible persistence of the plateau; they do not establish its cause, carryover to B/C, or K8 efficacy.

Evidence: [study](phase_a_budget/study.json), [paired A400/A800 metrics](phase_a_budget/paired_comparison.csv), [all checkpoint metrics](phase_a_budget/checkpoint_metrics.csv), [coverage](phase_a_budget/coverage_bins.csv), [training segments](phase_a_budget/training_segments.csv), [parameter movement](phase_a_budget/parameter_displacement.csv), [recorded completeness checks](phase_a_budget/completeness_checks.json).

## How the source evidence was curated

Numerical CSVs were copied without changing their values. Decision JSON retains original decisions, numerical outcomes and thresholds; study-file locations were changed to local archive filenames. Study JSON retains scientific design and interpretation rules while removing machine/session details and replacing server source locations with experiment identifiers. The budget study has no invented decision file; its fixed-stop design and measured endpoints are retained.

Each REPORT_TABLES.md contains selected scientific tables from the named original report, preserving their original wording and values, with a link back to this account for context and limitations. These are excerpts, not complete source reports. The original reports were A_SAMPLING_REPORT.md, ABC_VALIDATION_REPORT.md, NOISE_AUDIT_REPORT.md, REPRESENTATION_FIT_REPORT.md, and PHASE_A_BUDGET_REPORT.md. Operational sections and references to unavailable figures, logs and commands were not published.

Historical decision rules describe what happened in these studies. They do not add a new project-wide prerequisite for future work.
