# Two-stage experiments: reading and reproducing the results

Start with the cohort table below, then open the linked English preregistration where available, dated report, and all-seed CSV. This folder explains the T2 evidence behind the later T3 work. [Shared methods](../METHODS.md) describes the economic model, PPO and dynamic best-response verifier.

The selected T2 solver is a shared Beta-actor PPO with a backward curriculum and a **first-eligible checkpoint** rule. It can discover and certify useful approximate policies, but its success probability varies across batches. Final certification does not imply exact recovery of the closed-form equilibrium, and T2 success probabilities do not certify T3.

## Cohorts and their roles

"Discovery" means a development-eligible candidate was found. "Joint" means the same candidate passed final certification and dense concentration. All scheduled runs, including failures, remain in the discovery and end-to-end denominators.

| Cohort and role | q and seeds | Configuration | Discovery | Joint / all runs | Main artifacts |
|---|---|---|---:|---:|---|
| [Formal 0915](../../experiments/two_stage_formal_T2_20260915/REPORT.md), later used for development/selection | q50 and q60, paired seeds 10001–10020 | A400/B600/C1000; verifier cadence 100 in all phases; first eligible C | q50 10/20; q60 19/20 | q50 10/20; q60 18/20 | [All 40 rows](../../experiments/two_stage_formal_T2_20260915/formal_results.csv), [settings](../../experiments/two_stage_formal_T2_20260915/manifest.json) |
| [Confirmation](../../experiments/two_stage_confirmation_T2_20260922/REPORT.md), new fixed-N evaluation | q50 10101–10110; q60 10111–10120; unpaired | Final T2 protocol: cadence A100/B25/C25 | q50 7/10; q60 10/10 | q50 7/10; q60 10/10 | [All 20 rows](../../experiments/two_stage_confirmation_T2_20260922/formal_results.csv), [settings](../../experiments/two_stage_confirmation_T2_20260922/manifest.json), [preregistration](../../experiments/two_stage_confirmation_T2_20260922/PREREGISTRATION.md) |
| [E1](../../experiments/two_stage_E1_q50_p_20260923/REPORT.md), separate fixed-N extension | q50 10121–10140 | Same final T2 protocol | 19/20 | 17/20 | [All 20 rows](../../experiments/two_stage_E1_q50_p_20260923/formal_results.csv), [settings](../../experiments/two_stage_E1_q50_p_20260923/manifest.json), [preregistration](../../experiments/two_stage_E1_q50_p_20260923/PREREGISTRATION.md) |
| [Prospective restarts](../../experiments/two_stage_q50_restarts_20260924/REPORT.md), primary restart experiment | q50 10201–10230, ten triples fixed before outcomes | Same single-run protocol; compare fixed prefixes k=1,2,3 | 25/30 | 22/30 | [All 30 rows](../../experiments/two_stage_q50_restarts_20260924/runs.csv), [all 10 groups](../../experiments/two_stage_q50_restarts_20260924/groups.csv), [settings](../../experiments/two_stage_q50_restarts_20260924/manifest.json), [preregistration](../../experiments/two_stage_q50_restarts_20260924/PREREGISTRATION.md) |
| [Precision supplement](../../experiments/two_stage_q50_precision_20260924/REPORT.md), descriptive extension decided after the previous 30 outcomes | q50 10231–10240 | Same single-run protocol; no new restart groups | 4/10 | 3/10 | [All 10 rows](../../experiments/two_stage_q50_precision_20260924/runs.csv), [labeled pooled 40 rows](../../experiments/two_stage_q50_precision_20260924/pooled_runs.csv), [settings](../../experiments/two_stage_q50_precision_20260924/manifest.json), [preregistration](../../experiments/two_stage_q50_precision_20260924/PREREGISTRATION.md) |

Two different 30-run samples must stay separate:

- Historical q50 confirmation + E1: discovery **26/30**, joint **24/30**, conditional joint **24/26**. See [historical pooled results](../../experiments/two_stage_E1_q50_p_20260923/pooled_q50.json).
- Prospective restart sample: discovery **25/30**, joint **22/30**, conditional joint **22/25**. The ten fixed triples give joint group success **8/10, 9/10, 9/10** for k=1,2,3. One failed first attempt was rescued; k=3 did not add a success beyond k=2. All 30 runs were actually executed. Sequential stopping costs are reconstructed from those observations.

Adding the precision supplement to the restart sample gives descriptive discovery **29/40**, joint **25/40**, and conditional joint **25/29**. This extension followed inspection of earlier outcomes; its pooled Wilson intervals are descriptive summaries, not independent fixed-N confirmation. It does not replace the original ten restart groups. Results for q50 and q60 concern different games and are reported separately.

## Exact selected protocol

The [final protocol document](protocol/FINAL_T2_PROTOCOL_20260922.json) and each cohort's manifest contain the complete values. The primary manifests use the same game, PPO, learning-rate schedule and numerical verifier settings.

| Setting | Recorded value |
|---|---|
| Game | T=2; q=50 or 60; winner/loser prizes 6/2; k=1/3500; effort in [0,100] |
| Networks / optimizer | Separate 2→64→64 tanh actor and critic; Beta concentration floor 100; Adam; constant actor/critic LR 0.0003 |
| PPO | 10 epochs; minibatch 256; clip 0.2; value coefficient 0.5; gradient norm 0.5; entropy coefficient 0; gamma=lambda=1 |
| Phase caps / samples | A400, B600, C1000; 512 episodes/update |
| Phase A | Terminal-stage exploring starts; retained A transition condition; all recorded formal and confirmation runs used the full 400-update budget |
| Phase B | Root-start full episodes; three consecutive eligible development calls to advance, or the 600-update cap |
| Phase C | 256 root + 256 stage-2 exploring-start episodes; freeze the first eligible development call, including a call at C1000 |
| Checks | Warmup 100; stability every 20; maximum verifier interval A100/B25/C25; A/B normalized BR threshold 0.02; C threshold 0.01 |
| Concentration | Maximum normalized Beta standard deviation <=0.04 at the same checkpoint |
| Opponent | Hard snapshot every 20 global updates and on phase entry |
| Development / final grids | State step 4/2; effort step 1/0.5; Gauss–Legendre nodes per half 16/32 |
| Final joint certification | Development and final valid; final dReach/DeltaW <=0.01; both dReach and root-exploitability refinement differences <=0.002; dense concentration <=0.04 on step 0.05 |
| Execution | CPU, one Torch/OMP/MKL/OpenBLAS thread per process; at most 10 concurrent processes in original batches |
| Weight export | Every 25 global updates; diagnostic output, not a selection criterion |

A is described as fixed-budget pretraining in the protocol. The original runner retains its A verifier transition condition; it was inactive in the cited batches. The publication preserves that behavior.

Only development checks select the candidate. Final rejection is a failed run; training is not resumed and another checkpoint is not substituted. An endpoint from a run with no eligible candidate remains diagnostic even if some endpoint statistic looks favorable.

The closed-form benchmark is g1=DeltaW/(6kq) and g2(d)=DeltaW*f_xi(d)/(2k), where f_xi is the triangular difference-noise density. Thus g1 is 46.6667/38.8889 and g2(0) is 70/58.3333 for q50/q60. Recovery metrics measure distance to these functions; they have no additional acceptance threshold. The [confirmation recovery tables](../../experiments/two_stage_confirmation_T2_20260922/recovery_metrics.csv) include all 17 candidates.

## How the cadence change was chosen

The earlier 47–51 seed experiments compared A budgets, learning-rate schedules, checkpoint selection, opponent smoothing and verifier-triggered training changes. These were development comparisons with repeated seeds, not held-out reliability estimates. Offline evaluation found that earlier eligible policies could be lost by continuing training. This motivated delivery of the first eligible checkpoint.

The 0915 formal cohort then supplied 40 new runs under cadence 100. Its 11 no-candidate runs were investigated in two selected dense-C batches:

| Diagnostic batch | Selected runs | Change | Outcome |
|---|---|---|---|
| [Dense C first six](../../experiments/two_stage_dense_c25_20260922/REPORT.md) | q50 10008,10011,10012,10015,10020; q60 10017 | C cadence 100→25; export weights every 25; A/B unchanged | 4/6 produced jointly certified candidates |
| [Dense C remaining five](../../experiments/two_stage_dense_c25_rest_20260922/REPORT.md) | q50 10005,10006,10007,10010,10016 | Same change | 0/5 candidates |
| [Combined attribution](../../experiments/two_stage_dense_c25_rest_20260922/all11_summary.csv) | All 11 original no-candidate runs | Post-hoc selected failure cohort | 4 rescued; 7 still without candidates |

These are selected diagnostic runs. The 4/11 conversion count is not a cadence-25 protocol success rate and must not be spliced into the original 40-run table. The [A-stage audit](../../experiments/two_stage_formal_T2_20260915/A_stage_audit/README.md) and [B-cadence study](../../experiments/two_stage_formal_T2_20260915/B_cadence_feasibility/README.md) motivated the final A100/B25/C25 configuration. Changing B's cadence can change when B exits and thus the subsequent training trajectory; the confirmation used genuinely new runs.

## Run from any checkout location

Run commands from the repository root. The launcher uses the active Python interpreter and resolves repository files from its own location. The original exact environment check is retained: **Python 3.12.3, NumPy 2.5.0, Torch 2.5.1+cu121**. The recorded Torch build includes CUDA support, although these experiments run on CPU. A CPU-only Torch build with a different version string is rejected by the original runner. See [repository reproducibility instructions](../REPRODUCIBILITY.md) for environment setup.

First run a tiny flow smoke (2/2/3 updates, 1,000 direct-rollout episodes, one repetition):

```bash
python -B MultiStage/two_stage/run_experiment.py --cohort confirmation --q 50 --seed 10101 --smoke --workers 1 --out-root scratch/t2_smoke
```

The smoke exercises real PPO updates, verification and artifact writing. Its output is explicitly marked smoke and cannot be counted as a formal cohort.

To reproduce one historical configuration or a complete primary cohort:

```bash
python -B MultiStage/two_stage/run_experiment.py --cohort confirmation --q 50 --seed 10101 --workers 1 --out-root scratch/t2_one_run
python -B MultiStage/two_stage/run_experiment.py --cohort confirmation --out-root scratch/t2_confirmation
python -B MultiStage/two_stage/run_experiment.py --cohort e1 --out-root scratch/t2_e1
python -B MultiStage/two_stage/run_experiment.py --cohort restarts --out-root scratch/t2_restarts
python -B MultiStage/two_stage/run_experiment.py --cohort precision --out-root scratch/t2_precision
```

A complete command runs every prescribed seed, even after successful restart-group members. Omitting --out-root uses scratch/two_stage/<cohort>. Existing batch records are not overwritten; use a new output directory for a separate attempt. Per-run output includes config.json, train_history.json, final_eval.json, checkpoint.pt, arrays.npz and weight exports. A launcher failure is explicit in launch_status.json and retains the scheduled run.

Recount all published tables without training or writing:

```bash
python -B MultiStage/two_stage/summarize.py --archive all
```

Summarize a newly reproduced formal batch:

```bash
python -B MultiStage/two_stage/summarize.py --results-root scratch/t2_restarts
```

This writes new summary.json and runs.csv beside the new runtime manifest. The existing reporting logic rechecks candidate selection, stopping records and final numerical criteria. It reports all scheduled rows, including missing/failed operations. Fixed-group restart summaries are computed only when the complete predefined group schedule is present. The portable per-run summary uses a common schema for all cohorts; original cohort tables retain their historical schemas.

## Code and artifact map

| File(s) | Purpose |
|---|---|
| [run_experiment.py](run_experiment.py) | Relocates execution/output paths and launches archived settings; optional q/seed selection and original smoke mode |
| [summarize.py](summarize.py) | Recounts archived results and summarizes new formal outputs |
| [run/run_final_dp_br_round3_dense.py](../../run/run_final_dp_br_round3_dense.py) | Actual T2 curriculum, scheduling, stopping and weight export |
| [run/run_final_dp_br.py](../../run/run_final_dp_br.py) | Shared rollout and final-evaluation helpers |
| [agents/ppo_curriculum.py](../../agents/ppo_curriculum.py), [envs/curriculum_env.py](../../envs/curriculum_env.py) | PPO/Beta networks and sampled game/environment |
| [utils/dp_br_verifier.py](../../utils/dp_br_verifier.py), [utils/theory_multistage.py](../../utils/theory_multistage.py) | Numerical dynamic BR verification and theoretical benchmark |
| [summarize_restarts.py](../../experiments/two_stage_q50_restarts_20260924/summarize_restarts.py) | Existing result-classification, uncertainty, fixed-group and cost calculations reused by portable summarizer |
| [test_restarts.py](../../experiments/two_stage_q50_restarts_20260924/test_restarts.py) | Existing reporting regressions for rejection, missing runs, denominators and group assignment |

The solver's third-party dependencies are NumPy and Torch; the portable launcher/reporting entry points use the standard library. The six numerical source files came from the original candidate-search-recovery-data-92d49a worktree. Their mathematical and training logic is preserved. Machine-specific default paths and stale docstring references were adapted for publication; the existing module digests in protocol/manifest metadata describe the original research files and are not digests of those publication-only text edits.

This publication includes 80 primary scheduled rows (20 confirmation + 20 E1 + 30 restarts + 10 supplement), compact development tables, settings and reports. Historical pooled tables repeat some of those observations and add no independent runs. Full checkpoints, dense weight snapshots, training histories, replay bundles and duplicate Word/HTML reports are omitted to keep the repository compact. Archived output_dir fields in result tables identify original relative archive locations; those raw files are not bundled. Regenerate full artifacts in scratch/ with the launcher.

The dated reports preserve original wording and refer to some omitted full-archive files. Their old launch commands are historical; use the commands above. No original result was replaced by a new reproduction.

Run existing reporting regressions with:

```bash
python -B -m unittest discover -s experiments/two_stage_q50_restarts_20260924 -p 'test_restarts.py' -v
```
