# Three-stage experiment: portable compact archive

This directory contains the executed T3 curriculum experiment, its numerical tests, and complete outcomes for the formal 40-run and pilot 6-run cohorts. **Every supplied endpoint and minimum-development policy is diagnostic and uncertified.** No run found a candidate. There is no certified equilibrium in this archive.

| Cohort | q | Seeds | Candidates | Certified |
|---|---:|---|---:|---:|
| Formal | 50 | 11001–11020 | 0/20 | 0/20 |
| Formal | 60 | 11101–11120 | 0/20 | 0/20 |
| Pilot | 50 | 10411–10413 | 0/3 | 0/3 |
| Pilot | 60 | 10401–10403 | 0/3 | 0/3 |

The pilot is an implementation/debug cohort, not a reliability estimate. Formal rates use all 20 planned seeds per q. Conditional certification is N/A (zero candidates), not zero percent of candidates.

## Read the results

- [Formal report](reports/FORMAL_REPORT.md), [formal settings](reports/FORMAL_SETTINGS.md), and [per-seed tables](reports/formal/FORMAL_TABLES.md).
- [Pilot report](reports/PILOT_REPORT.md).
- [Chinese experiment record](../../MultiStage/three_stage/T3_FORMAL_EXPERIMENT_RECORD_20260926.md).
- For each cohort, reports/<cohort>/runs.csv maps run_id, q, seed, checkpoint identity, search outcome, verifier values, and economics.
- runs/<run_id>/ contains the recorded configuration, training and final-evaluation summaries, economics moments, status, and endpoint/minimum-development NumPy weights.
- reports/<cohort>/policy_curves_compact.csv preserves every seed and both checkpoints. Signed-gap curves retain a 1.0-unit grid from the original 0.05-unit grid, including zero and both boundaries. This is a plotting subset; it does not replace the original dense concentration check. Retained std_norm and effort curves are original samples, not interpolation.
- reports/<cohort>/figures/ contains original endpoint policy, asymmetry, visitation, and deviation PDFs. These depict diagnostic terminal policies.
- reports/<cohort>/state_visitation_compact.csv retains mean/stochastic root self-play visitation for the representative player, with every seed included.
- Economics are available as mean-policy and stochastic-policy effort/cost/payoff moments in economics.json and stage_metrics.csv. Formal economics_by_group.csv separates certified, uncertified candidate, and diagnostic groups; the first two groups are empty.

## Run from the repository root

Use the recorded stack and setup instructions in [reproducibility](../../MultiStage/REPRODUCIBILITY.md): Python 3.12.3, NumPy 2.5.0, PyTorch 2.5.1+cu121, CPU with one thread per worker. Newer compatible environments may execute the code but are not promised to reproduce training trajectories exactly. The five numerical core modules are the repository agents/ppo_curriculum.py, envs/curriculum_env.py, run/run_final_dp_br.py, utils/dp_br_verifier.py, and utils/theory_multistage.py. No private worktree or host path is required.

```sh
E=experiments/three_stage_implementation_pilot_20260924
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B -m unittest discover -s "$E/tests" -p 'test_t3.py' -v
python -B "$E/replay_publication.py" --run-id t3_formal_q50_s11001
python -B "$E/replay_publication.py" --cohort all
python -B "$E/formal_summary.py" --cohort formal
```

Replay loads actor weights from NumPy, recomputes endpoint development/final and minimum-development development verifiers, and compares six recorded metrics. It performs no training and writes no files. Defaults allow floating point differences of rtol=1e-7, atol=1e-9. This checks saved-policy evaluation, not bitwise training reproducibility or equilibrium certification.

formal_summary.py regenerates tables in regenerated/<cohort>/ by default, preserving archived reports. The original make_report.py needs full raw run artifacts; the compact archive intentionally omits them and its default output is regenerated/<cohort>/.

To execute a short smoke training run in a new directory:

```sh
python -B "$E/build_manifests.py" --cohort smoke --output-dir /tmp/t3-manifests --runs-prefix /tmp/t3-smoke-runs
python -B "$E/run_experiment.py" --manifest /tmp/t3-manifests/smoke_q50.json --seed 10410
python -B "$E/run_experiment.py" --manifest /tmp/t3-manifests/smoke_q60.json --seed 10400
python -B "$E/make_report.py" --cohort smoke --manifest /tmp/t3-manifests/smoke_q50.json --manifest /tmp/t3-manifests/smoke_q60.json --check-completeness --no-figures --out-dir /tmp/t3-smoke-report
```

The builder defaults to generated_manifests/ and reruns/. Existing differing manifests and existing run directories are never overwritten. launch.py uses the active Python interpreter and accepts --manifest (repeatable) and --max-workers. Explicit relative --runs-prefix paths resolve from the current working directory.

A new full formal rerun uses fresh manifests and output directories (40 runs; substantially longer than smoke):

```sh
python -B "$E/build_manifests.py" --cohort formal --settings "$E/reports/formal_settings.json" --seeds-per-q 20 --output-dir /tmp/t3-formal-manifests --runs-prefix /tmp/t3-formal-reruns
python -B "$E/launch.py" --manifest /tmp/t3-formal-manifests/formal_q50.json --manifest /tmp/t3-formal-manifests/formal_q60.json --max-workers 1
```

For a newly completed formal rerun, pass both new manifests to make_report.py with --cohort formal and --out-dir /tmp/t3-formal-report, following the smoke-report example. Omit --no-figures to produce figures.

Choose worker count according to CPU/RAM; the original formal experiment used ten workers, each with one thread.

## Archive limits

Full per-update histories (except the complete pilot q60 seed 10401 history retained for the baseline-reproduction test), dense policy/asymmetry CSVs, full action-value arrays, optimizer checkpoints (.pt), coverage arrays, and raw rollout aggregates are excluded to keep the Git repository small. Full original make_report.py output cannot be regenerated from this subset. The complete recorded cohort outcomes and economic moments remain available; saved weights allow verifier and policy-curve reevaluation. The A-sampling/ABC development studies remain in source code and tests, but their results are not included in these formal/pilot cohorts.

Archived machine paths have been rewritten to repository-relative paths; recorded numerical values, outcomes, seeds, and checkpoint identity fields are retained. Original .pt/raw-array file references inside historical metadata describe omitted artifacts. New reruns record their own runtime paths and provenance.
