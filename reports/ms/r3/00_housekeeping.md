# MS-R3 P0: housekeeping

Date: 2026-10-07. Spec: `reports/ms/r3/pi_record/20_ms_r3_prompt.md` (saved verbatim first) section 1.

## 1. Remote state and the branch

```
$ git fetch origin
   4dc604de..15e5b71a  main       -> origin/main
$ git ls-remote origin refs/heads/main refs/heads/ms-r2 refs/heads/ms-r1
15e5b71a5d87d85c891804f1b36fecb1b277e8ae	refs/heads/main
e8eb9a08041efc6a6021549294edabb65f7dc047	refs/heads/ms-r2
71c58904678badd5fdea28e8213567b328f52b5c	refs/heads/ms-r1
$ git ls-remote --tags origin           # 10 tags (each with its ^{} line)
ipccc2026-pre-publication, t2-refine-100526, t2-v2-confirmation, t2-v2-confirmation-v2.0, t2-v2-lock-v1.0, t2-v2-lock-v1.1,
t2-v2-lock-v2.0, t2-v2-main, t2-v2-main-v2.0, t2-v2-report-pack            (no ms tag)
$ git switch -c ms-r3 origin/ms-r2      # origin/ms-r2 = e8eb9a08, as required
$ git branch --unset-upstream ms-r3     # so that no plain `git push` can reach ms-r2; pushes name the target branch
```

`origin/main` moved from `4dc604de` (MS-R2 housekeeping) to `15e5b71a` ("Add files via upload", PI, 2026-10-07 15:13): it adds `reports/sandbox_fit_tip.py` and `reports/sandbox_fit_tip_results.jsonl`, the PI-side sandbox named in the prompt. They were read from `origin/main` with `git show` (the branch was not changed) and stored unchanged in `reports/ms/r3/pi_record/`; their SHA-256 equal the values in the prompt:

```
a02308287ba149c63678063fc995a5c7296d32150a392d6fb6134bba86269f36  reports/ms/r3/pi_record/sandbox_fit_tip.py
3038885bc3ff69b688425a1728a4bac21dcbec7fe0305c7a4f00cd5fa5257f8a  reports/ms/r3/pi_record/sandbox_fit_tip_results.jsonl
```

They are not re-run and are not evidence (prompt item 4).

Work tree (the session's own): `/home/fjiang4/tournament_experiment/.claude/worktrees/p2-gate-ms-base2400-e41857` (the worktree in which MS-R1 P2 and MS-R2 ran; its session branch `claude/p2-gate-ms-base2400-e41857` is at `71c58904`; `ms-r3` is created here from `origin/ms-r2`, so the untracked arrays of the MS-R1 and MS-R2 pilots under `results/` are present).

Recorded, not fixed (the prompt forbids touching them): the local `main` of the primary checkout is at `d1b84437`, behind `origin/main` (`15e5b71a`); the local ref `ms-r1` of the P1 worktree of MS-R1 is at `21139215`, behind `origin/ms-r1` (`71c58904`); the local `ms-r2` equals `origin/ms-r2`. Many other local refs of unrelated worktrees are at older commits.

## 2. Reference roots (D7): existence and what this round reads

| root | exists | content read in this round | counts |
|---|---|---|---|
| MS-R2 pilot, `/home/fjiang4/tournament_experiment/.claude/worktrees/p2-gate-ms-base2400-e41857/results/ms_r2/pilot` (this worktree; the MS-R2 fact-check ledger names it) | yes | weight exports, freeze arrays, per-update series, check tables, gates (C-MS5, section 2.3 (b)) | 120 runs; 16,320 weight exports (136 per run); 120 `freeze_stage2_final.npz`; 120 `ms_updates.csv`; 120 `ms_checks_stage2.csv` |
| MS-R1 pilot, `.../p2-gate-ms-base2400-e41857/results/ms_r1/pilot` | yes | weight exports (section 2.3 (b)) | 120 runs; 14,309 weight exports (rule arms stop early); 120 `freeze_stage2_final.npz` |
| MS-R1 base, `.../ms-r1-multistage-development-afebf2/results/ms_r1/base` | yes | weight exports (section 2.3 (b)) | 20 runs; 1,760 weight exports (88 per run) |
| `parents_A`, `.../v2-t2-refine/results/v2_refine/parents_A` | yes | `final_final.npz` and records for the MS-R1 criterion | 20 `status.json`, 20 `final_final.npz`; 1,280 `weights/u*.npz` (64 per run) |
| `rehearsal_v2_0`, `.../v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0` | yes | records for stage 1 and C-R6 | 20 `status.json`, 20 `gates.json`; 1,760 `weights/u*.npz` (88 per run) and 20 `checkpoint_weights.npz` |
| C7 canonical worktree, `.../pilot-4-stabilization-fb99a2` | yes | C7 reference | |

(Counts by `find <root> -path '*weights*' -name '*.npz' | wc -l` and `find <root> -name freeze_stage2_final.npz | wc -l`.)

## 3. Host and environment

```
nproc = 64
 19:15:49 up 9 days,  2:56,  5 users,  load average: 11.31, 10.88, 11.03
/dev/md0  5.5T total, 4.3T used, 935G free (83 %)
Mem 754 GiB total, 695 GiB available
Python 3.12.3, torch 2.5.1+cu121, numpy 2.5.0, scipy 1.18.0, pandas 3.0.3 (/home/fjiang4/tournament_experiment/.venv/bin/python)
embedded record (protocols/v2_T2_locked_v2_0.json records[50].versions): python 3.12.3, torch 2.5.1+cu121, numpy 2.5.0  -> equal
```

The load average of about 11 is other users' work; no job of this session was running at the time. Nothing was killed.
