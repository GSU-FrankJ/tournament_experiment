# T3-R3, P0: housekeeping

Date: 2026-10-21. Branch `t3-r3`. Commands and verbatim outputs; machine-readable copies are in `results/t3_r3/preflight/`.

## 1. Refs

`git ls-remote origin` (the lines the prompt asks for: `main`, `t3-r1`, `t3-r2` and the tags):

```
15e5b71a5d87d85c891804f1b36fecb1b277e8ae	refs/heads/main
be2b7e9fe3e0aad34f7481a1d8306af7cc2963e7	refs/heads/t3-r1
c3289df9afbcd69d995c2b8c6d5b497a9dddeebe	refs/heads/t3-r2
17c4a4366caed920287dcd6287a6ead7576d8f17	refs/tags/ipccc2026-pre-publication
c85dcd4590e16665b9dff303a59e12facc1dd95b	refs/tags/ipccc2026-pre-publication^{}
435978699aa8d2527f763181c987cef3972fac19	refs/tags/t2-refine-100526
978ca39d66b87ae9b8f21d27a2b419f61f26703b	refs/tags/t2-refine-100526^{}
289961e0b03ad8ce693969c2291c859628159d29	refs/tags/t2-v2-confirmation
f6838ec2550dbb7ce946a4085afd9189f03c8f5c	refs/tags/t2-v2-confirmation^{}
4724ca007e7ebcb0fe2d0e1a95ce11b9b483fb87	refs/tags/t2-v2-confirmation-v2.0
d2e377d0da702e8e162662f74576d1bff5b42a56	refs/tags/t2-v2-confirmation-v2.0^{}
e97b0d7216b067c0290bb9004a2a97d97423cbcd	refs/tags/t2-v2-lock-v1.0
4bd221484cd300aa89600cb01a636e35835af40d	refs/tags/t2-v2-lock-v1.0^{}
e4830b754c2ea0c1e053924d9e6fdc16fa26640d	refs/tags/t2-v2-lock-v1.1
431474d18259ff7534520b8a6d52616beb36c800	refs/tags/t2-v2-lock-v1.1^{}
32e68cfd117372b76edff70c2545da02434c539a	refs/tags/t2-v2-lock-v2.0
1d6d4d00736b265a18ae71b91ecb77b19e4915b9	refs/tags/t2-v2-lock-v2.0^{}
70fae9fbd7fb1a66d69cf7b9c50e9dac8663542e	refs/tags/t2-v2-main
f02a256095dcaa69dffbd011718d337f84183890	refs/tags/t2-v2-main^{}
53ae8c321a41b00d844c12696cecfedf95eb2ef4	refs/tags/t2-v2-main-v2.0
b55d38907b02106a6e874d3b60622e4e5b7c9d5d	refs/tags/t2-v2-main-v2.0^{}
909c7678c6b249d5e4f6e5ed5bcd1b14de5923b0	refs/tags/t2-v2-report-pack
28e14b5cca01ac2e0f2eab71005ff3bb6736a7d7	refs/tags/t2-v2-report-pack^{}
```

`git fetch origin` then `git rev-parse origin/t3-r2` = `c3289df9afbcd69d995c2b8c6d5b497a9dddeebe`: **as required**. The tags are unchanged against T3-R2's record.

Local refs that are behind or different (recorded, not fixed):

| local ref | local commit | origin | note |
|---|---|---|---|
| `main` (primary checkout) | `d1b84437` | `origin/main` `15e5b71a` | behind |
| `t3-r1` (in the T3-R1 worktree `.claude/worktrees/t3-bw-vs-jt-51e472`) | `ea727dcb` | `origin/t3-r1` `be2b7e9f` | behind, as in T3-R2's record; left where it is (`t3-r1-p3-finish` is at `be2b7e9f`) |
| `ms-r4` | `2f618531` | `origin/ms-r4` `db7577ea` | behind; left where it is |
| `ms-r1` | `21139215` | `origin/ms-r1` `71c58904` | behind |
| `t3-r2` (in the T3-R2 worktree `.claude/worktrees/t3-bw-failures-audit-6f3a38`) | `c3289df9` | `origin/t3-r2` `c3289df9` | equal |

## 2. Branch and worktree

- Worktree path (recorded): `/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-diagnosis-25cb13`. It is the worktree this session was started in (harness branch `claude/t3-bw-failures-diagnosis-25cb13`, at `15e5b71a` = `origin/main`, clean, no work of its own); it is the "new worktree" of the prompt, so no second worktree was created (the same arrangement as T3-R2's `00_housekeeping.md` §2).
- `git switch -c t3-r3 origin/t3-r2` in that worktree: HEAD `c3289df9afbcd69d995c2b8c6d5b497a9dddeebe`, branch `t3-r3`. The upstream that the switch set (`origin/t3-r2`) was removed with `git branch --unset-upstream`, so that a plain `git push` cannot go to `t3-r2`. The harness branch `claude/t3-bw-failures-diagnosis-25cb13` was left where it is.
- The prompt was saved verbatim as `reports/t3/r3/pi_record/28_t3_r3_prompt.md` before any other file was written in the worktree.
- Not touched: the worktrees of the earlier rounds (T3-R1, T3-R2, v2.0 refinement), any other worktree. No `git clean` was run anywhere.

## 3. Seeds (P0.3)

Run in tmux (session `t3r3_seeds`, `tools/t3/t3r3_preflight_seeds.sh`; the two commands in the prompt's order; outputs in `results/t3_r3/preflight/`).

```
$ python tools/v2/seed_inventory.py --block 70501 70520 --out results/t3_r3/preflight/seed_inventory_70501_70520.csv
start 2026-10-21T22:05:18Z
scanned 213844 json, 79067 csv files; 416 distinct seed values
block 70501-70520: 0 collisions
recorded seeds within +-1000 of the block: []
exit=0
end 2026-10-21T22:07:45Z

$ python tools/v2/seed_inventory.py --block 80501 80540 --out results/t3_r3/preflight/seed_inventory_80501_80540.csv
start 2026-10-21T22:07:45Z
scanned 213844 json, 79068 csv files; 416 distinct seed values
block 80501-80540: 0 collisions
recorded seeds within +-1000 of the block: []
exit=0
end 2026-10-21T22:10:09Z

$ python tools/t3/seed_hits.py --out-dir results/t3_r3/preflight --block 70501 70520 --block 80501 80540
start 2026-10-21T22:10:09Z
scanned 213844 json, 79069 csv files; 416 distinct seed values
block 70501-70520: 0 hit rows, 0 seeds -> results/t3_r3/preflight/seed_hits_70501_70520.csv
block 80501-80540: 0 hit rows, 0 seeds -> results/t3_r3/preflight/seed_hits_80501_80540.csv
exit=0
end 2026-10-21T22:12:35Z
```

(`seed_inventory_*.out.txt`, `seed_hits.out.txt`.) The inventory scans every `*.json` and `*.csv` of the whole tournament tree, all worktrees. Both blocks exit 0: **no collision**, no recorded seed within 1000 of either block. Both hit listings have **0 hit rows** (header only): no file of any class records a seed of 70501-70520 or 80501-80540, so there is nothing to classify and no training-run record to stop on. 70501-70520 is the development block, 80501-80540 is reserved for the later confirmation and is not trained in this round.

## 4. Reference roots (P0.4, D8)

`tools/t3/t3r3_check_roots.py --t3r1-root .../t3-bw-vs-jt-51e472/results/t3_r1 --t3r2-root .../t3-bw-failures-audit-6f3a38/results/t3_r2 --v20-root .../v2-t2-refine/results/v2_T2_locked/confirmation_v2_0 --out results/t3_r3/preflight/reference_roots.csv` (exit 0; `reference_roots.out.txt`):

```
runs ok: 120 / 120; bw with train_history.json: 80 / 80
tracked roots present in the branch: {'results/t3_r1': True, 'results/t3_r2': True, 'results/v2_T2_locked/confirmation_v2_0': True, 'reports/t3/r1': True, 'reports/t3/r2': True}
```

| runs | directories | exports per run | files checked per run |
|---|---|---|---|
| 40 T3-R1 `bw` (`runs/q{q}/seed{50501..50520}/bw`) | 40 of 40 | 152 | `state_end_stage{1,2,3}.pt`, `freeze_stage{1,2,3}_{final,development}.npz`, `train_history.json` (present in all 40) |
| 40 T3-R2 `bw` (`confirmation/q{q}/seed{60501..60520}/bw`) | 40 of 40 | 152 | the same (`train_history.json` present in all 40) |
| 40 v2.0 (`q{q}/seed{30501..30520}`) | 40 of 40 | 88 | `v2_updates.csv` (untracked root and tracked copy) |

For the 80 `bw` runs the tracked files (`ms_updates.csv`, `ms_checks_stage{1,2,3}.csv`, `ms_binmaps_stage{2,3}.npz`, `run_config.json`, `gates.json`) are present in the reference root and in the branch's index. Row-wise table: `results/t3_r3/preflight/reference_roots.csv`. (`run_stage` writes no `ms_binmaps_stage1.npz`: the bin maps are written for `t >= 2` only, stage 1 has the single state 0.)

## 5. Machine (P0.5)

| item | value |
|---|---|
| `nproc` | 64 |
| load average at the start | 11.1 / 11.0 / 10.7 (other users' jobs; ours are limited to the workers named per step) |
| free disk on `/home` | 799 GB of 5.5 TB (85 % used) |
| memory | 754 GB total, 705 GB available |
| Python | 3.12.3 |
| torch | 2.5.1+cu121 |
| numpy | 2.5.0 |
| scipy | 1.18.0 |

The three versions equal the v2.0 and T3-R2 records; `MSRun` refuses to run otherwise (`results/t3_r3/preflight/machine.txt`).

## 6. Files of P0

`reports/t3/r3/pi_record/28_t3_r3_prompt.md`, this file, `results/t3_r3/preflight/*` and the two preflight helpers `tools/t3/t3r3_preflight_seeds.sh` and `tools/t3/t3r3_check_roots.py` (new files, no existing file changed).
