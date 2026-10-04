# T=2 v2: index

Entry point for the T=2 v2 report. Everything a reader needs is reachable from this folder: the study reports, the PI record (plans, prompts, decisions) and the report pack. Material that lives elsewhere in the repository (protocols, code, results) is cited by path. Nothing was moved, renamed or copied to build this index.

## Scope and status

- **Scope.** P0 (code audit and regression) to P6 (fresh-seed confirmation) of the T=2 stagewise work, plus the report pack. T=3 and the paper are out of scope.
- **Status.** The pre-registered confirmation of protocol v1.1 (seeds 20501-20520, launched at `f6838ec`) **passed**: 20/20 primary passes at q = 50 and 20/20 at q = 60 (rule: at least 18 of 20 at each q). The secondary criterion S1 passed 20/20 at q = 50 and 18/20 at q = 60.
  - Sources: `results/v2_T2_locked/confirmation_analysis/verdict.json`, `pass_counts.csv` and `s1_summary.csv` (same folder).
  - Report: [protocol_v1_1_confirmation.md](protocol_v1_1_confirmation.md).
- **Freeze rule.** Nothing in the protocols, the entry point, the pipeline code or the analysis script changes after the v1.1 lock without a new version number and a stated reason (`protocols/LOCK`).

## Reading order

1. [pi_record/00_README.md](pi_record/00_README.md): the PI-side record. Plans, every prompt as delivered, and the PI's decisions at each review gate.
2. The study reports, in order:

   | # | report | what it covers |
   |---|---|---|
   | 1 | [phase0_audit.md](phase0_audit.md) | read-only audit of the T=2 pipeline before any v2 code |
   | 2 | [phase1_verifier.md](phase1_verifier.md) | full-domain verifier metrics, invariants, calibration |
   | 3 | [phase2_infra.md](phase2_infra.md), [phase2_opening_checks.md](phase2_opening_checks.md) | stagewise runner (frozen stage 2, reward modes, branching), C7 regression, smoke runs; opening checks of the on-path rule |
   | 4 | [dreach_reach_mask_check.md](dreach_reach_mask_check.md) | dReach reach-mask check |
   | 5 | [pilot1_reward_estimator.md](pilot1_reward_estimator.md) | Pilot 1: sampled vs expected terminal reward (40 runs) |
   | 6 | [pilot2_freeze.md](pilot2_freeze.md) | Pilot 2: joint vs frozen stage 2 (60 runs) |
   | 7 | [pilot3_continuation_mode.md](pilot3_continuation_mode.md) | Pilot 3: stochastic vs mean continuation (40 runs) |
   | 8 | [phaseA_ext.md](phaseA_ext.md) | Phase A extended to 1,600 updates (20 runs) |
   | 9 | [pilot4_stabilization.md](pilot4_stabilization.md) | Pilot 4: stabilization round, analyses 1a-1d and runs 2a and 2b (80 runs) |
   | 10 | [protocol_lock_and_rehearsal.md](protocol_lock_and_rehearsal.md) | v1.0 lock, development-seed rehearsal, cusp diagnostic, Check 1 addendum |
   | 11 | [protocol_v1_1_confirmation.md](protocol_v1_1_confirmation.md) | v1.1 lock, re-rehearsal, fresh-seed confirmation |
   | 12 | [protocol_v2_0_confirmation.md](protocol_v2_0_confirmation.md) | v2.0 lock (expected continuation in Phase B, gate G-S), re-rehearsal, fresh-seed confirmation (seeds 30501-30520), deviations |

3. [summary.md](summary.md): running summary across the studies.
4. [t2_report/README.md](t2_report/README.md): the report pack. 111 items (tables T01-T59, figures F01-F24, per-run tables D01-D09, key numbers K01-K19), each with source paths and SHA-256, the building script and its transformations. `manifest.csv` lists the items, `data_dictionary.csv` documents every column, `gaps.md`, `consistency.md` and `reevaluations.csv` list what is missing, what disagrees with the reports and what was recomputed. T05 and T57 count the files of `reports/v2/*.md` and T57 cites `docs/STATE.md` by SHA-256, so they and their entries in `manifest.csv` and `provenance/` were rebuilt once this index and the STATE.md section existed (the commit that follows the index in `git log -- reports/v2/t2_report`); the rest of the pack is as tagged `t2-v2-report-pack`.

## Outside this folder (paths only)

| what | where |
|---|---|
| Protocols | `protocols/v2_T2_locked.json` and `.md` (v1.0); `protocols/v2_T2_locked_v1_1.json` and `.md` (v1.1); `protocols/LOCK` (lock records with the SHA-256 of the protocol, the entry point and the analysis script) |
| Locked entry point | `run/run_v2_T2_locked.py` (refuses a modified protocol and any argument other than `--q`, `--seed`, `--out-dir`) |
| v2 pipeline code | `run/run_v2_stagewise.py`, `run/v2_rollout.py`, `agents/ppo_curriculum_v2.py`, `utils/v2_metrics.py`; built on the existing `agents/ppo_curriculum.py`, `envs/curriculum_env.py`, `run/run_final_dp_br_round3_dense.py` and `utils/dp_br_verifier.py` |
| Analysis and study tools | `tools/v2/confirmation_analysis.py` (pre-registered analysis); the other study tools are in `tools/v2/` |
| Tests | `tests/test_v2_verifier.py`, `tests/test_v2_infra.py`, `tests/test_v2_locked.py` |
| Pack builder | `tools/v2/report/`; `build_t2_report_pack.py` rebuilds `reports/v2/t2_report/` from `results/` |
| Results roots | `results/v2_pilots/` (Phase 1 and 2 outputs, Pilots 1-4, Phase A extension, regressions, smoke runs); `results/v2_T2_locked/` (calibration, rehearsals, consolidation, confirmation and their analyses) |

### Which result files are tracked

Tracked in `results/v2_pilots/` and `results/v2_T2_locked/`:
- per-run `manifest.json`, `run_config.json` and `status.json`;
- per-update logs (`v2_updates.csv`) and verifier checkpoint tables (`v2_checkpoints*.csv`);
- run summaries, gate and drift files (JSON) and stage-2 figures;
- every analysis CSV and Markdown file;
- 797 small `.npz` files (stage-1 induced-band sweeps, regression references, smoke runs, calibration grids), each at most 1.2 MB.

Not tracked, because of size: per-run weight exports (`weights/`), full-state checkpoints (`checkpoints/`, `state_end_*.pt`), per-run result arrays (`*.npz`: final, exit, gate, band-sweep and drift-test arrays, `checkpoint_weights.npz`), `train_history.json` and run logs.

### Large files that are not in git

They are on vector2, in the canonical worktree: `/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/`. The pack builder reads this root through `--results-root`.

| group | files | size |
|---|---|---|
| untracked arrays (`*.npz`) | 20,997 | 2,348.4 MiB |
| untracked `train_history.json` (one per run, 323 runs) | 323 | 522.6 MiB |
| gitignored checkpoints (`*.pt`) | 529 | 85.0 MiB |
| gitignored logs | 342 | 1.6 MiB |
| **total** | **22,191** | **2,957.6 MiB (2.89 GiB, 3.10 GB)** |

Integrity records: `t2_report/manifest.csv` and `t2_report/provenance/` give the SHA-256 of every source file of every item, including the untracked arrays the pack used. `results/v2_T2_locked/consolidation/results_root_checksums.csv` gives the SHA-256 of 10,034 files of the pilots root.

## Key commits and tags

Annotated tags (all on `main`'s history):

| tag | commit | what |
|---|---|---|
| `t2-v2-lock-v1.0` | `4bd2214` | lock of protocol v1.0 |
| `t2-v2-lock-v1.1` | `431474d` | lock of protocol v1.1 (gate G-N, secondary criterion S1, global-RNG hardening) |
| `t2-v2-confirmation` | `f6838ec` | launch commit of the confirmation (40 runs, seeds 20501-20520) |
| `t2-v2-report-pack` | `28e14b5` | the report pack as first built (builder at `7062b5b`); T05, T57 and their source entries were refreshed afterwards, see above |
| `t2-v2-main` | the last commit of the publication (index, PI record, pack refresh) | state of `main` after the T=2 v2 publication |

Not tagged and not on `main` (branch `v2-t2-refine`, not pushed): the lock of protocol v2.0 (expected continuation in Phase B, gate G-S) `1d6d4d0`; its LOCK record `f2d616c`; the launch commit of the v2.0 confirmation (40 runs, seeds 30501-30520) `d2e377d`; the checks-tool fix `3fedaa2`. Report: [protocol_v2_0_confirmation.md](protocol_v2_0_confirmation.md).

Other commits: the v2 work starts from `657f54a` (tip of PR #7); the v2 line was merged into `main` by `8667672`; the LOCK record of v1.1 is `95c000e`. Launch commits of the studies:

| study | launch commit | report |
|---|---|---|
| Pilot 1 | `89cd600` | [pilot1_reward_estimator.md](pilot1_reward_estimator.md) |
| Pilot 2 | `1791687` | [pilot2_freeze.md](pilot2_freeze.md) |
| Pilot 3 and Phase A extension | `cd760fd` | [pilot3_continuation_mode.md](pilot3_continuation_mode.md), [phaseA_ext.md](phaseA_ext.md) |
| Pilot 4 (2a and 2b) | `c92ee74` | [pilot4_stabilization.md](pilot4_stabilization.md) |
| dirty-flag re-run | `5d50a9d` | [protocol_lock_and_rehearsal.md](protocol_lock_and_rehearsal.md) |
| v1.0 rehearsal and Check 2 | `5b07293` | [protocol_lock_and_rehearsal.md](protocol_lock_and_rehearsal.md) |
| v1.1 re-rehearsal | `95c000e` | [protocol_v1_1_confirmation.md](protocol_v1_1_confirmation.md) |
| confirmation | `f6838ec` | [protocol_v1_1_confirmation.md](protocol_v1_1_confirmation.md) |
| v2.0 re-rehearsal | `f2d616c` | [protocol_v2_0_confirmation.md](protocol_v2_0_confirmation.md) |
| v2.0 confirmation | `d2e377d` | [protocol_v2_0_confirmation.md](protocol_v2_0_confirmation.md) |

## Known caveats

- **Pack against reports.** 42 pack values disagree with statements or table digits in the existing reports (22,186 cells compared). The old reports were not edited: [t2_report/consistency.md](t2_report/consistency.md).
- **Unknown values.** 10 values are `UNKNOWN`; nothing was estimated: [t2_report/gaps.md](t2_report/gaps.md).
- **Reproducibility ledger.** 26 of 27 checks pass. The one failure is v1.0 Check 1 judged literally (all RNG states); it was accepted under owner decision D1 because the training-relevant state is identical in 20/20 runs: [t2_report/tables/T18_reproducibility_ledger.md](t2_report/tables/T18_reproducibility_ledger.md).
- **Pilot 3 identity check.** The actor/critic identity check of Pilot 3 against Pilot 2 B2 does not compare optimizer states, so it does not support a claim of optimizer identity. Table: [t2_report/tables/T26_pilot_3_reproducibility_stochastic_arm_vs_pi.md](t2_report/tables/T26_pilot_3_reproducibility_stochastic_arm_vs_pi.md); column `end_state_actor_critic_opt_identical` in [t2_report/data_dictionary.csv](t2_report/data_dictionary.csv).
- **PI record.** It holds every prompt as delivered and the PI's decisions, not a verbatim transcript of the planning discussion (its own README says so). `prompts/11_github_main_prompt.md`, the prompt that asked for this publication, is covered by its `SHA256SUMS` but not listed in its README table.
