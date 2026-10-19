# T3-R2, P0: housekeeping

Date: 2026-10-19. Branch `t3-r2`. Commands and verbatim outputs; machine-readable copies are in `results/t3_r2/preflight/`.

## 1. Refs

`git ls-remote origin` (the lines the prompt asks for: `main`, `t3-r1`, `ms-r4` and the tags):

```
15e5b71a5d87d85c891804f1b36fecb1b277e8ae	refs/heads/main
db7577ea5a85910ecfc43db142e872fc1950bf75	refs/heads/ms-r4
be2b7e9fe3e0aad34f7481a1d8306af7cc2963e7	refs/heads/t3-r1
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

`origin/t3-r1` = `be2b7e9fe3e0aad34f7481a1d8306af7cc2963e7` (`git fetch origin t3-r1` then `git rev-parse origin/t3-r1 FETCH_HEAD`: both `be2b7e9f...`): **as required**. The full `ls-remote` output also lists the other `origin` branches (`ms-r1`, `ms-r2`, `ms-r3`, `t2-refine-pack`, `t2-status-pack`, `v2-*`, ...), unchanged and not used.

Local refs that are behind or different (recorded, not fixed):

| local ref | local commit | origin | note |
|---|---|---|---|
| `main` | `d1b84437` | `origin/main` `15e5b71a` | behind |
| `ms-r4` | `2f618531` | `origin/ms-r4` `db7577ea` | behind; held by the MS-R4 launch worktree `.claude/worktrees/ms-r4-e1-e2-experiments-2da78b`, left where it is |
| `t3-r1` (in the T3-R1 worktree `.claude/worktrees/t3-bw-vs-jt-51e472`) | `ea727dcb` | `origin/t3-r1` `be2b7e9f` | behind, as the prompt says; left where it is. The branch `t3-r1-p3-finish` (worktree `t3-r1-p3-finish-df1175`) is at `be2b7e9f` |
| `ms-r1`, `ms-r2`, `ms-r3` | `21139215`, `e8eb9a08`, `be4fd202` | `ms-r1` `71c58904`, `ms-r2` `e8eb9a08`, `ms-r3` `be4fd202` | `ms-r1` is behind; the others equal |

## 2. Branch and worktree

- Worktree path (recorded): `/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-audit-6f3a38`. It is the worktree this session was started in (harness branch `claude/t3-bw-failures-audit-6f3a38`, at `15e5b71a` = `origin/main`, clean, no work of its own); it is the "new worktree" of the prompt, so no second worktree was created.
- `git fetch origin t3-r1`, then `git checkout -b t3-r2 origin/t3-r1` in that worktree: HEAD `be2b7e9fe3e0aad34f7481a1d8306af7cc2963e7`, branch `t3-r2`. The upstream that `checkout -b` set (`origin/t3-r1`) was removed with `git branch --unset-upstream`, so that a plain `git push` cannot go to `t3-r1`. The harness branch `claude/t3-bw-failures-audit-6f3a38` was left where it is.
- The prompt was saved verbatim as `reports/t3/r2/pi_record/27_t3_r2_prompt.md` before any other file was written in the worktree.
- Not touched: the T3-R1 worktree (read-only reference root), the MS-R4 launch worktree, any other worktree. No `git clean` was run anywhere.

## 3. Seeds (P0.3)

Run in tmux (session `t3r2_seeds`; the two commands in the prompt's order; outputs in `results/t3_r2/preflight/`).

```
$ python tools/v2/seed_inventory.py --block 60501 60520 --out results/t3_r2/preflight/seed_inventory_60501_60520.csv
start 2026-10-19T19:32:56Z
scanned 197552 json, 71755 csv files; 396 distinct seed values
block 60501-60520: 0 collisions
recorded seeds within +-1000 of the block: []
exit=0
end 2026-10-19T19:46:29Z

$ python tools/t3/seed_hits.py --out-dir results/t3_r2/preflight --block 60501 60520
start 2026-10-19T19:46:29Z
scanned 197552 json, 71756 csv files; 396 distinct seed values
block 60501-60520: 0 hit rows, 0 seeds -> results/t3_r2/preflight/seed_hits_60501_60520.csv
exit=0
end 2026-10-19T19:48:50Z
```

(`seed_inventory_60501_60520.out.txt`, `seed_hits.out.txt`.) The inventory scans every `*.json` and `*.csv` of the whole tournament tree, all worktrees; it exits 0: **no collision**. The listing has **0 hit rows** (header only): no file of any class records a seed of 60501-60520, so there is nothing to classify and no training-run record to stop on. There are no recorded seeds within +-1000 of the block.

## 4. Reference roots (P0.4, D6)

`tools/t3/t3r2_check_roots.py --t3r1-root .../t3-bw-vs-jt-51e472/results/t3_r1 --v20-root .../v2-t2-refine/results/v2_T2_locked/confirmation_v2_0 --out results/t3_r2/preflight/reference_roots.csv` (exit 0):

```
ok / total per arm: {'bw': (40, 40), 'jt': (40, 40), 'jt_es': (40, 40), 'jt2': (40, 40), 'v20': (40, 40)} ; eval/arrays files: 200
```

| arm | directories | exports per run | state files per run |
|---|---|---|---|
| `bw` | 40 of 40 | 152 | `state_end_stage1.pt`, `state_end_stage2.pt`, `state_end_stage3.pt` |
| `jt` | 40 of 40 | 50 | `state_end.pt` |
| `jt_es` | 40 of 40 | 70 | `state_end.pt` |
| `jt2` | 40 of 40 | 44 | `state_end.pt` |
| `v20` | 40 of 40 | 88 | `state_end_A.pt`, `state_end_B.pt` |

`eval/arrays/` holds 200 files. Every directory also holds `run_config.json` and `status.json` (row-wise table: `results/t3_r2/preflight/reference_roots.csv`).

## 5. Machine (P0.5)

| item | value |
|---|---|
| `nproc` | 64 |
| load average at the start | 12.54 / 12.19 / 11.97 (other users' jobs; ours are limited to the workers named per step) |
| free disk on `/home` | 820 GB of 5.5 TB (85 % used) |
| memory | 754 GB total, 704 GB available |
| Python | 3.12.3 (v2.0 record: 3.12.3) |
| torch | 2.5.1+cu121 (record: 2.5.1+cu121) |
| numpy | 2.5.0 (record: 2.5.0) |
| scipy | 1.18.0 |

The three versions equal the v2.0 record; `MSRun` refuses to run otherwise.
