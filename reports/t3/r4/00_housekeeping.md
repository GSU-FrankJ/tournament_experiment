# T3-R4, P0: housekeeping

Date: 2026-10-22. Branch `t3-r4`. Commands and verbatim outputs; machine-readable copies are in `results/t3_r4/preflight/`.

## 1. Refs

`git ls-remote origin main t3-r2 t3-r3 'refs/tags/*'` (the lines the prompt asks for):

```
15e5b71a5d87d85c891804f1b36fecb1b277e8ae	refs/heads/main
c3289df9afbcd69d995c2b8c6d5b497a9dddeebe	refs/heads/t3-r2
df2365d9278f0a0b7f3ab5efa5ec8cab04eaf1e5	refs/heads/t3-r3
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

- `git fetch origin t3-r3`, then `git rev-parse origin/t3-r3` = `df2365d9278f0a0b7f3ab5efa5ec8cab04eaf1e5`: **as required**. The tags are those of T3-R3's record (no tag for T3-R1..T3-R3).
- `git diff --stat b565f0e8 df2365d9 -- agents envs utils run tools/ms tools/v2` is **empty** (0 lines of output).

## 2. Branch and worktree

- Worktree path (recorded): `/home/fjiang4/tournament_experiment/.claude/worktrees/t3-r4-lock-confirm`, created for this round with `git worktree add -b t3-r4 <path> df2365d9278f0a0b7f3ab5efa5ec8cab04eaf1e5` (branch `t3-r4`, HEAD `df2365d9278f0a0b7f3ab5efa5ec8cab04eaf1e5`, no upstream set, so a plain `git push` cannot go to `t3-r3`).
- The prompt was saved verbatim as `reports/t3/r4/pi_record/29_t3_r4_prompt.md` before any other file was written in the worktree.
- Not touched: the worktrees of the earlier rounds (the T3-R3 worktree `.claude/worktrees/t3-bw-failures-diagnosis-25cb13`, branch `t3-r3`, is read only: it holds the weights of the C-REPRO originals), any other worktree. No `git clean` was run anywhere.

## 3. Seeds (P0.3)

Run in tmux (session `t3r4_seeds`, `tools/t3/t3r4_preflight_seeds.sh`; outputs in `results/t3_r4/preflight/`).

```
$ python tools/v2/seed_inventory.py --block 80501 80540 --out results/t3_r4/preflight/seed_inventory_80501_80540.csv
start 2026-10-22T05:29:06Z
scanned 231947 json, 87424 csv files; 437 distinct seed values
block 80501-80540: 0 collisions
recorded seeds within +-1000 of the block: []
exit=0
end 2026-10-22T05:31:40Z

$ python tools/t3/seed_hits.py --out-dir results/t3_r4/preflight --block 80501 80540
scanned 231947 json, 87425 csv files; 437 distinct seed values
block 80501-80540: 0 hit rows, 0 seeds -> results/t3_r4/preflight/seed_hits_80501_80540.csv
exit=0
```

**No hit at all**: not in a training-run record (`run_config.json`, `manifest.json`, `status.json`, `gates.json`, a run directory), and not in any other json or csv record of any worktree (the tool scans the json and csv files of all worktrees; T3-R3's pre-registration and plan name the block in markdown, which the tool does not read). Nothing to classify. The block 80501-80540 has never been trained.

## 4. Roots (P0.4)

Read-only, verified:

- T3-R3 root `/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-diagnosis-25cb13/results/t3_r3/`: exists.
- T3-R2 root `/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-audit-6f3a38/results/t3_r2/`: exists (`confirmation/` is present).
- The C-REPRO originals (T3-R3 `bwA` runs):

| run (original) | exports `weights/u*.npz` | `state_end_stage1.pt`, `state_end_stage2.pt`, `state_end_stage3.pt` | `retry.json` | `attempt1/` |
|---|---|---|---|---|
| `dev/q50/seed70508/bwA` (the trigger fired) | 152 | present | present | `attempt1_record.json`, `ms_binmaps_stage3.npz`, `ms_checks_stage3.csv`, `ms_updates_stage3.csv`, `state_attempt1_end.pt` and 64 exports in `attempt1/weights/` |
| `dev/q60/seed70501/bwA` (not fired) | 152 | present | absent (as it must be) | absent (as it must be) |

## 5. Machine (P0.5)

| item | value |
|---|---|
| `nproc` | 64 |
| load average at 05:36 UTC | 9.28 / 10.08 / 9.69 (other sessions are running on the machine) |
| free disk (`/home`, `df -h`) | 790 GB free of 5.5 TB (86 % used) |
| memory (`free -g`) | 754 GB total, 695 GB available |
| Python | 3.12.3 (`/home/fjiang4/tournament_experiment/.venv/bin/python`) |
| torch | 2.5.1+cu121 (CUDA available; the runs are CPU, single-threaded) |
| numpy | 2.5.0 |

## 6. A known erratum (recorded here only)

`reports/t3/r3/01_diagnosis.md` says that from update 400 on the out-of-band gradient share is 0.1-0.7 %. Its own table shows 1.06-1.7 % in the window 401-800. The conclusion (the shares are far below the out-of-band sample share of 0.75) is unchanged. T3-R3's files are not edited (this was also noted at T3-R3's fact-check, `reports/t3/r3/pi_record/01_factcheck.md`, C9).
