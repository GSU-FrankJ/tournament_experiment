# T=2 refinement work (R1, v2.0, R2b, R2c): index

Entry point for the publication of the four rounds that answered the PI's note of 2026-10-02 (six candidate methods and two diagnostics for the T=2 pipeline). Everything a reader needs is reachable from this folder: the summary report, the PI record, and the evidence pack. The round reports stay where they are (`reports/v2/refine/`, `reports/v2/protocol_v2_0_confirmation.md`, `reports/v2/refine_r2b/`, `reports/v2/refine_r2c/`) and are cited by path; material elsewhere in the repository (protocols, code, results) is cited by path. Nothing was moved, renamed or edited to build this folder (the only changes to existing files are the dated addenda of report section 7, item 24(d), and a dated section in `docs/STATE.md`): the only copies are those in `evidence/` and `figures/`, each with its original path (or figure name), source commit and SHA-256 in `evidence/manifest.csv`.

## Scope and status

- **Scope.** R1 (two diagnostics and six single-factor methods on the locked v1.1 pipeline), the lock and fresh-seed confirmation of protocol v2.0, R2b (stage-2 peak mechanisms) and R2c (start-distribution pilot with a pre-registered selection rule). T=3 and the paper are out of scope.
- **Status.** The four rounds are closed as reported (PI decision D1 of the publication prompt, `pi_record/prompts/16_t2_refine_publication_prompt.md`). **The locked T=2 solver is protocol v2.0** (`protocols/v2_T2_locked_v2_0.json`; tags `t2-v2-lock-v2.0`, `t2-v2-confirmation-v2.0`, `t2-v2-main-v2.0`). Its fresh-seed confirmation (seeds 30501-30520) passed the pre-registered rule: 19/20 at q = 50 (exact 95% CI [0.7513, 0.9987]) and 20/20 at q = 60 ([0.8316, 1.0000]) [evidence CF-02]. R2c selected no arm under its pre-registered rule, so there is no v2.1. Method 5 (pathwise fine-tuning) is closed as negative at matched budgets, the censored likelihood is not adopted, and the peak-focused start distribution is not adopted at T=2 and is carried as a design input for the T=3 terminal stage.
- **Freeze rule.** Nothing in `protocols/`, the locked entry point `run/run_v2_T2_locked.py`, `utils/v2_continuation.py` or `tools/v2/confirmation_analysis_v2_0.py` is changed by this publication.

## Reading order

1. [100526report.md](100526report.md): the summary report for the PI. A one-page Chinese executive summary (section 0), then scope, the note's eight items and how each was tested, protocol v1.1 → v2.0 with its confirmation, the results by method, the diagnostics, the stage-2 peak, known issues and deviations, implications for T=3, and how to reproduce.
2. The four round reports, in order:

   | # | report | what it covers |
   |---|---|---|
   | 1 | [../v2/refine/summary.md](../v2/refine/summary.md) (reading order of `00`-`06` in that folder) | R1: two diagnostics (D1 clamp, D2 verifier sensitivity) and six methods, development seeds |
   | 2 | [../v2/protocol_v2_0_confirmation.md](../v2/protocol_v2_0_confirmation.md) | v2.0 lock (expected continuation in Phase B, gate G-S), re-rehearsal, fresh-seed confirmation (seeds 30501-30520), deviations |
   | 3 | [../v2/refine_r2b/summary.md](../v2/refine_r2b/summary.md) | R2b: peak-focused starts, censored likelihood, pathwise epochs; the seed-30510 diagnostic |
   | 4 | [../v2/refine_r2c/summary.md](../v2/refine_r2c/summary.md) | R2c: four start-distribution arms, selection rule, no arm selected |

3. [pi_record/README.md](pi_record/README.md): the PI-side record: prompt → round → the PI's decisions at the gate that followed → outcome. [pi_record/00_publication_log.md](pi_record/00_publication_log.md) is the log of the git steps of this publication; [pi_record/01_factcheck.md](pi_record/01_factcheck.md) is the fact-check ledger.
4. [evidence/manifest.csv](evidence/manifest.csv): the evidence pack (items, source paths, SHA-256, source commits); the copies are under `evidence/<source path>`; [figures/](figures/) holds the pack's 19 figures `FG-01` to `FG-19` (the report cites 15; `FG-02` and `FG-04` are stage overviews and `FG-03` and `FG-14` per-arm paired-difference plots of the round reports, kept as manifest items); [report_scripts/](report_scripts/README.md) the scripts that computed the report's tables.

## Branches, commits and tags of each round

Hashes were checked with `git log` / `git rev-parse` (branch and tag hashes: `pi_record/00_publication_log.md` section 1.1; commit counts: section 1.6) and again in the fact-check (`pi_record/01_factcheck.md`). The seven-character forms in the PI's list are abbreviations of the hashes below.

| round | branch (head) | code | pre-registration / launch | records and final head |
|---|---|---|---|---|
| R1 | `v2-t2-refine` | `32a8c210` | pre-registration `6c902dbd` (the launch records name it); the manifests of the 380 stage-1 and stage-2 pilot runs record `655b14c6` (26 runs) and `6e99e01a` (354 runs), the 20 `parents_A` manifests record `655b14c6` and the 20 C-R1 manifests `6c902dbd`; at the commits named, the run code is identical to the code commit (`git diff --stat 32a8c21 <commit> -- run agents utils envs protocols` is empty) | final record `155cdece` |
| v2.0 | `v2-t2-refine` (head `b55d3890`) | lock `1d6d4d00` | LOCK record `f2d616cc`, also the commit at which the re-rehearsal was launched; confirmation launched at `d2e377d0` | confirmation records `85c294ea`; report `6e216e29`; publication head `b55d3890` |
| R2b | `v2-t2-r2b` (head `62ecc436`) | `1ff99bd4` | the pilots ran at `d581b3cc` (the commit in every pilot manifest) | final head `62ecc436` |
| R2c | `v2-t2-r2c` (head `6a8f4492`) | `58c26716` | C-R3 ran at the code commit; wave S ran at `3ad1b07d` (the commit in every wave-S manifest) | wave-S results, launch checks and analysis tables `3492cac4`; reports and head `6a8f4492` |

Where the PI's list and the repository differ (the prompt asks to say so): (1) `3ad1b07` is listed as the R2c "results" commit; its subject is "chore: record R2c C-R3 reproduction, tool validation and review", it is the commit the wave-S manifests record, and the commit that holds the wave-S runs, launch checks and analysis tables is `3492cac4`. (2) `d2e377d` is listed as "rehearsal/launch": it is the confirmation launch commit (tag `t2-v2-confirmation-v2.0` peels to it) and it holds the re-rehearsal records; the re-rehearsal itself was launched at `f2d616c`. (3) `d581b3c` is listed as the R2b "launch": it is the commit recorded in the pilot manifests; its subject is the addendum to the pre-registration. Every other hash in the PI's list exists and is what the list says.

Annotated tags of the earlier work, all on `main`'s history after this publication: `t2-v2-lock-v1.0` (`4bd22148`), `t2-v2-lock-v1.1` (`431474d1`), `t2-v2-confirmation` (v1.1 confirmation, `f6838ec2`), `t2-v2-main` (`f02a2560`), `t2-v2-report-pack` (`28e14b5c`), and from the v2.0 round `t2-v2-lock-v2.0` (`1d6d4d00`), `t2-v2-confirmation-v2.0` (`d2e377d0`), `t2-v2-main-v2.0` (`b55d3890`). This publication adds the annotated tag `t2-refine-100526` on its head; the result of the push is in `pi_record/00_publication_log.md` section 5.

## Outside this folder (paths only)

| what | where |
|---|---|
| Locked protocol and lock records | `protocols/v2_T2_locked_v2_0.json` and `.md`; `protocols/v2_T2_locked_v1_1.json`; `protocols/LOCK` (records for v1.0, v1.1 and v2.0; each carries the SHA-256 of the protocol and the entry point, v1.1 and v2.0 also that of the analysis script, v2.0 also that of the continuation module) |
| Locked entry point and module | `run/run_v2_T2_locked.py` (refuses a modified protocol and any argument other than `--q`, `--seed`, `--out-dir`); `utils/v2_continuation.py` (SHA-locked; its module docstring still calls check (ii) open, see report section 7) |
| v2 pipeline code | `run/run_v2_stagewise.py`, `run/v2_rollout.py`, `agents/ppo_curriculum_v2.py`, `utils/v2_metrics.py`; R1 additions (flags, phase P, continuation table), R2b additions (`start_weights`, `clamp_likelihood`, `pathwise_*`), R2c addition (`start_weights.local_first`) |
| Analysis scripts | `tools/v2/confirmation_analysis_v2_0.py` (v2.0 confirmation); `tools/v2/refine_analysis.py` (R1); `tools/v2/refine_r2b_analysis.py` (R2b); `tools/v2/refine_r2c_analysis.py` (R2c); `tools/v2/d1_clamp_analysis.py`, `tools/v2/verifier_sensitivity.py` (D1, D2); `tools/v2/diag_seed30510.py` |
| Launchers and checks | `tools/v2/launch_refine.py`; `tools/v2/cr2_compare.py` (C-R2, C-R3); `tools/v2/r2c_launch_checks.py`; `tools/v2/v2_0_rehearsal_checks.py` |
| Pack builder and figure script | `tools/v2/report/build_t2_refine_pack.py` (builds `evidence/` and `figures/`, verifies every copy against its source and its blob at `HEAD`, `--check-against` rebuilds and compares byte for byte, `--sums DIR` writes a `SHA256SUMS`); `tools/v2/report/plot_t2_refine_stage1_dist.py` (figure FG-01, drawn from the pack's own tables) |
| Results roots (tracked part) | `results/v2_refine/` (R1), `results/v2_T2_locked/` (rehearsals, v1.1 and v2.0 confirmations and analyses), `results/v2_refine_r2b/` (R2b), `results/v2_refine_r2c/` (R2c); the tracking rules are those of `reports/v2/README.md` (per-run `manifest.json`, `run_config.json`, `status.json`, per-update CSVs, gate and summary JSON, analysis tables; no `.pt`, `train_history.json` or weight exports; the 60 per-run `continuation_table.npz` files of the v2.0 re-rehearsal and confirmation are tracked) |

### Large files that are not in git

They are on vector2, in the worktrees below (sizes from `du -sh`, which prints GiB, tracked files included). The manifest lists the large and untracked per-run files that the round reports and the report refer to, with their location and SHA-256 (status `referenced (untracked)`; the two tracked files of 1 MiB or more are `referenced (tracked, large)`).

| worktree (`/home/fjiang4/tournament_experiment/.claude/worktrees/...`) | what it holds | size |
|---|---|---|
| `pilot-4-stabilization-fb99a2` (the canonical worktree) | `results/v2_T2_locked/`: `rehearsal_v1_1` and the v1.1 confirmation | 1.5 GiB |
| `pilot-4-stabilization-fb99a2` | `results/v2_pilots/phase1_regression/before/FINAL_A400_B25_C25/tel_q50_s10501`: the C7 reference run, with its `checkpoint.pt` | 2.2 MiB (`results/v2_pilots/` as a whole: 1.8 GiB) |
| `v2-t2-refine` | `results/v2_refine/` (`parents_A`, the R1 runs, D1 per-row tables and buffers), `results/v2_T2_locked/` (`rehearsal_v2_0`, the v2.0 confirmation including the failed run q = 50 seed 30510) | 3.0 GiB and 1.3 GiB |
| `v2-t2-r2b` | `results/v2_refine_r2b/` (waves A and P, the seed-30510 diagnostic) | 1.2 GiB |
| `v2-t2-r2c` | `results/v2_refine_r2c/` (wave S, C-R3) | 1.3 GiB |

## Known caveats

The full list, with sources and statuses, is section 7 of the report. The ones a reader should know first:

- **The one failed run of the v2.0 confirmation** (q = 50 seed 30510, G-A through eta_2) is part of the result (19/20); its read-only diagnostic is descriptive and the cause was not decided (report sections 3.5 and 6.4).
- **Decision D-R7:** check R7 of the v2.0 re-rehearsal is false as the checks tool computed it and was accepted as met on the canonical-reference comparison (report sections 3.4 and 7).
- **Known issues kept as recorded:** the Phase-A clamp flags (D1), the verifier numerics item, the stale docstring of `utils/v2_continuation.py`, the stale entries T05/T56/T57 of the v1.1 report pack (refresh waived), the registry test that fails as before, and "run `pytest tests`, not a bare `pytest`".
- **Inputs of this publication that were not available:** the PI's note `Multistage100226.docx` and the delivered prompts 12-14 (prompts 15 and 16 are transcriptions made by the assistant, labelled as such); see `pi_record/README.md` and `pi_record/plans/README.md`.
- **Numbers in the PI's list that differ from the evidence** are reported in `pi_record/01_factcheck.md`: five interval endpoints at the PI's four-decimal rounding (three of R2b `A_peak50`, one each of R2c `A_peak35` and `A_peak50_late400`; the largest difference is 0.0004), the shorthand "meets the criterion at q = 50 only" (exact for part (a) of three R2c arms, not for R2b `A_peak50`, which meets (a) at both q), and the commit labels listed above. The report uses the evidence values.
- **`main`:** this publication advances `origin/main` by pushing `t2-refine-pack` to it, because the session could not run git in the primary checkout; the primary checkout's local `main` was not touched and remains to be fast-forwarded by the owner (`pi_record/00_publication_log.md` sections 1.3 and 5).

## Integrity

`evidence/SHA256SUMS`, `figures/SHA256SUMS`, `pi_record/SHA256SUMS` and `report_scripts/SHA256SUMS` are in `sha256sum -c` format (run each from its own folder). `python tools/v2/report/build_t2_refine_pack.py --check-against reports/t2_refine_100526` rebuilds the pack in a temporary directory from the sources at the current commit and compares it with this folder byte for byte.
