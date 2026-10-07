# MS-R2 P0: housekeeping

Date: 2026-10-07. Prompt: `reports/ms/r2/pi_record/19_ms_r2_prompt.md` (saved verbatim before anything else). `§` and `D#` refer to it.

## 1.1 Remote state and the branch

Work tree (the session's own): `/home/fjiang4/tournament_experiment/.claude/worktrees/p2-gate-ms-base2400-e41857` (the worktree of MS-R1 P2; the session branch `claude/p2-gate-ms-base2400-e41857` is at `71c58904`; `ms-r2` is created here, so the untracked arrays of the MS-R1 pilot under `results/ms_r1/pilot/` stay in place, which D8 names as a reference root).

```
$ git fetch origin
$ git ls-remote origin refs/heads/main refs/heads/ms-r1 'refs/tags/*'
4dc604de9b2800d5dba99dfafe95696f630ab4b1	refs/heads/main
71c58904678badd5fdea28e8213567b328f52b5c	refs/heads/ms-r1
(tags: ipccc2026-pre-publication, t2-refine-100526, t2-v2-confirmation, t2-v2-confirmation-v2.0, t2-v2-lock-v1.0,
 t2-v2-lock-v1.1, t2-v2-lock-v2.0, t2-v2-main, t2-v2-main-v2.0; no ms tag)
$ git switch -c ms-r2 origin/ms-r1        # origin/ms-r1 = 71c58904, as required
$ git branch --unset-upstream ms-r2       # so that no plain `git push` can reach ms-r1; pushes name the target branch
```

Recorded, not fixed (the prompt forbids touching them): the local ref `ms-r1` of the MS-R1 P1 worktree (`.claude/worktrees/ms-r1-multistage-development-afebf2`) is at `21139215`, behind `origin/ms-r1` (`71c58904`); the local `main` of the primary checkout is at `d1b84437`, behind `origin/main` (`4dc604de`). `main` and the tags are not touched in this round.

## 1.2 Reference roots (D8): existence and content

Checked with a read-only script (every run directory against the files this round reads):

| root | run directories | content checked | result |
|---|---|---|---|
| MS-R1 pilot, `/home/fjiang4/tournament_experiment/.claude/worktrees/p2-gate-ms-base2400-e41857/results/ms_r1/pilot` (arms `MS_rule`, `MS_s25a0`, `MS_s25a5`, `MS_s35a0`, `MS_s35a5`, `MS_base2400`) | 120 of 120 (6 arms x 2 q x 10 seeds) | `ms_updates.csv`, `ms_checks_stage{1,2}.csv`, `freeze_stage2_{final,development}.npz`, `freeze_stage1_final.npz`, `train_history.json`, `rule_log.json`, `gates.json`, `run_config.json`, `manifest.json`, `status.json` | 0 files missing; weight exports per run 115-136 (u0025 ... , every 25 updates up to the run's last update) |
| MS-R1 base wave, `/home/fjiang4/tournament_experiment/.claude/worktrees/ms-r1-multistage-development-afebf2/results/ms_r1/base` (`MS_base`) | 20 of 20 | the same list | 0 missing; 88 weight exports per run (u0025-u2200) |
| `parents_A`, `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_refine/parents_A` | 20 of 20 | `status.json`, `run_config.json`, `final_v2.json`, `final_final.npz`, `final_development.npz` | 0 missing; 64 weight exports per run |
| `rehearsal_v2_0`, `.../v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0` | 20 of 20 | `status.json`, `run_config.json`, `gates.json`, `gateA_final.npz`, `induced_band.json` | 0 missing; 88 weight exports per run |
| canonical worktree for C7, `.../pilot-4-stabilization-fb99a2` | exists | | |

## 1.3 Host and environment

```
nproc = 64
 07:28:06 up 8 days, 15:08,  5 users,  load average: 11.19, 11.14, 11.61
/dev/md0  5.5T total, 4.3T used, 947G free (83 %)
Mem 754 GiB total, 716 GiB available
Python 3.12.3, torch 2.5.1+cu121, numpy 2.5.0, scipy 1.18.0, pandas 3.0.3 (/home/fjiang4/tournament_experiment/.venv/bin/python)
embedded record (protocols/v2_T2_locked_v2_0.json records[50].versions): python 3.12.3, torch 2.5.1+cu121, numpy 2.5.0  -> equal
```

No job of this session was running (`ps` shows no `run_ms*` / `run_v2*` / `pytest` process, no tmux server); the load average of 11 is other users' work. Nothing was killed.

## 2. P2 housekeeping addendum (2026-10-07)

- **Launch** (tmux session `ms_r2_pilot`): `python tools/ms/launch_ms_r1.py --wave r2 --params reports/ms/r1/prereg_parameters.json --workers 40 --code-commit 99857513daf88ee2813430da7390e19c985385a3` from HEAD `9b93b50c` (`results/ms_r2/pilot/launch_20261007_094801.json`: 64 cores, load average 14.8 / 13.7 / 13.6 at the start, 1.01e12 of 5.95e12 bytes free, `diff_stat_run_code_to_head` empty, `status_porcelain` 840 untracked files under `results/ms_r1/pilot`, state `done`, 120 planned and 120 finished, no non-zero exit). Each run took 555-711 s of wall time. Nothing was changed between the launch and the end of the pilot (a clean tree outside `results/`).
- **Roots used** (explicit arguments, recorded in `results/ms_r2/analysis/analysis_info.json`): MS-R1 pilot `results/ms_r1/pilot` of this worktree; `parents_A` and `rehearsal_v2_0` in `.claude/worktrees/v2-t2-refine/results/`; calibration `results/ms_r1/calibration`. The `MS_base` wave of MS-R1 is not needed by this round's analysis.
- **Results committed** in three commits after the pilot (rules of `reports/ms/README.md`): `75533f2e` (13 files per run of the 120 runs, the launch record and `launch_checks.json`: 1562 files, 235 MB; not tracked: `.pt`, `train_history.json`, weight exports, `freeze_stage*_*.npz`, `band_sweep.npz`, `run.log`), `a37cdd58` (`results/ms_r2/analysis/`), `2227fc3a` (`results/ms_r2/stop_calibration_pilot/`).
- **Departures from the prompt:** none beyond those recorded in `summary.md` (the P1 ones: the flag moved to a constructor argument, the C-MS4 `p_digest_next` rule, the transmission sign). Report-only blocks that D5 does not list are labelled in `04_pilot.md` (`sampler`, `window`) and `05_decision_inputs.md` (`tie_effort`, `directions`, `side_by_side`).
- **Not touched:** `main`, tags, the local refs of the other worktrees, the files of the constrained list of the prompt.
