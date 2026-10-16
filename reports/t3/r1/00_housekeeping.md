# T3-R1, P0: housekeeping

Date: 2026-10-16. Branch `t3-r1`. Commands and verbatim outputs; machine-readable copies are in `results/t3_r1/preflight/`.

## 1. Refs

`git ls-remote origin` (full output kept outside the repository; the lines the prompt asks for):

```
15e5b71a5d87d85c891804f1b36fecb1b277e8ae	refs/heads/main
be4fd2021e5aee52625a994b3566dc18726e0586	refs/heads/ms-r3
db7577ea5a85910ecfc43db142e872fc1950bf75	refs/heads/ms-r4
be2881ff31474ce5147bef29f9ff0e785e1d2b95	refs/heads/t2-status-pack
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

`origin/ms-r4` = `db7577ea5a85910ecfc43db142e872fc1950bf75` (`git rev-parse origin/ms-r4`; object present locally): **as required**.

Local refs that are behind or ahead of their `origin` refs (recorded, not fixed): `main` d1b84437 (behind `origin/main` 15e5b71a by 107), `ms-r1` 21139215 (behind 71c58904 by 5), `ms-r4` 2f618531 (behind db7577ea by 3; held by the MS-R4 launch worktree, left where it is), `t2-status-pack` e4f93dc8 (behind be2881ff by 1), `feat/multistage-phase0` 73d04257 (ahead of ff9a8f28 by 2). Every other local branch that has a same-name `origin` branch is equal to it. Four stale local `claude/*` session branches track `origin/main` as their upstream and are behind it (`claude/pull-multistage-folder-290b5a` by 108, `claude/research-repo-reproducibility-641cd5` ahead 4 / behind 111, `claude/table-iii-effort-calculation-f8da7d` by 162, `claude/two-stage-experiment-results-069955` by 162); recorded, not fixed.

## 2. Branch and worktree

- Worktree path (recorded): `/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-vs-jt-51e472`. It is the worktree the session was started in for this task (name `t3-bw-vs-jt`); it was clean and held no work. It is the "new worktree" of the prompt: no second worktree was created.
- `git switch --no-track -c t3-r1 origin/ms-r4` in that worktree: HEAD `db7577ea5a85910ecfc43db142e872fc1950bf75`, branch `t3-r1`, no upstream. The harness branch `claude/t3-bw-vs-jt-51e472` (at `15e5b71a`, based on `main`, no commits of its own) was left where it is.
- The prompt was saved verbatim as `reports/t3/r1/pi_record/25_t3_r1_prompt.md` before any other file was written (commit `43d720c7`).
- Not touched: `main`, the MS-R4 launch worktree `.claude/worktrees/ms-r4-e1-e2-experiments-2da78b`, any other worktree. No `git clean` was run anywhere.

## 3. Seeds (P0.3)

Run in tmux (`tools/t3/preflight_seeds.sh` runs the two inventories in the prompt's order and then the listing; the listing was run a second time by hand after its class rules were extended, see the last paragraph). All outputs are in `results/t3_r1/preflight/`.

**Inventories** (`tools/v2/seed_inventory.py`; excerpts of `seed_inventory_50500_50520.out.txt` and `seed_inventory_30501_30520.out.txt`, which hold the full output with the five sources per seed, next to the CSVs):

```
$ python tools/v2/seed_inventory.py --block 50500 50520 --out results/t3_r1/preflight/seed_inventory_50500_50520.csv
start 2026-10-16T22:06:12Z
scanned 165540 json, 57503 csv files; 373 distinct seed values
block 50500-50520: 0 collisions
recorded seeds within +-1000 of the block: []
exit=0
end 2026-10-16T22:39:11Z

$ python tools/v2/seed_inventory.py --block 30501 30520 --out results/t3_r1/preflight/seed_inventory_30501_30520.csv
start 2026-10-16T22:39:11Z
scanned 165567 json, 57516 csv files; 394 distinct seed values
block 30501-30520: 20 collisions
recorded seeds within +-1000 of the block: [30001, 30501, 30502, ..., 30520]
exit=1
end 2026-10-16T23:07:35Z
```

(exit 1 is the tool's code for "hits found", expected for 30501-30520 per the prompt; the five sources it prints per seed are protocol declarations and files under `reports/t2_refine_100526/evidence/`.) A `seed_inventory*` output lists every seed it found, so each inventory output appears in the other block's listing below as class `seed_inventory_output`.

**Full listing** (`tools/t3/seed_hits.py`, the matching rules of `seed_inventory.scan`, one row per (seed, source file, key), the class from path rules; `seed_hits.out.txt`, `seed_hits_50500_50520.csv`, `seed_hits_30501_30520.csv`; 23:15:11 - 23:34:57 UTC):

```
$ python tools/t3/seed_hits.py --out-dir results/t3_r1/preflight --block 50500 50520 --block 30501 30520
scanned 165586 json, 57532 csv files; 394 distinct seed values
block 50500-50520: 105 hit rows, 21 seeds
block 30501-30520: 11506 hit rows, 20 seeds
exit=0
```

| block | class | rows | files | what the files are |
|---|---|---|---|---|
| 50500-50520 | `seed_inventory_output` | 21 | 1 | `seed_inventory_30501_30520.csv` of this round (it lists the seeds it found, among them the 50500-50520 ones of the first inventory) |
| 50500-50520 | `t3_r1_own_outputs` | 84 | 44 | the 8 smoke runs of this round (`results/t3_r1/smoke/`) and their records |
| 50500-50520 | any other | 0 | 0 | |
| 30501-30520 | `protocol` | 400 | 20 | `protocols/v2_T2_locked_v2_0.json` and `reports/t2_refine_100526/evidence/protocols/v2_T2_locked_v2_0.json` (the lock declaration of v2.0 and its evidence copy) |
| 30501-30520 | `seed_inventory_output` | 700 | 35 | `seed_inventory*` files (`seed_inventory_stock.csv` 20, `seed_inventory_40501_40520.csv` 10, `seed_inventory_10511_10530.csv` 3, `seed_inventory_50500_50520.csv` 1, `seed_inventory_30501_30520.csv` 1) |
| 30501-30520 | `t3_r1_own_outputs` | 100 | 5 | `c_init_plan.csv`, `c_t2r.csv`, `c_t2r_per_run.csv`, `v20_reference_runs.csv`, `run_config_hashes.csv` of this round (the planned `t2j` wave and the 40 recorded v2.0 runs) |
| 30501-30520 | `v20_confirmation_or_copy` | 10,306 | 3,548 | the 40 v2.0 confirmation runs and their copies, analyses and re-evaluations in 12 trees (`results/v2_T2_locked/confirmation_v2_0*`, `reports/t2_refine_100526/evidence/`, `results/ms_r1/calibration/`, `results/v2_refine_r2b/diag_30510/`) |
| 30501-30520 | any other | 0 | 0 | |

Among the hits, the files named `run_config.json`, `manifest.json`, `status.json` or `gates.json` are 26 for 50500-50520, all in `results/t3_r1/smoke` (this round's smoke runs), and 1,928 for 30501-30520, all under `results/v2_T2_locked/confirmation_v2_0` (the 40 v2.0 confirmation runs in 12 trees). **No training run of the project's tournament lines, in any worktree, used a seed of 50500-50520 before this round, and no training run other than v2.0's confirmation used a seed of 30501-30520.** Neither stop-and-report condition applies.

**Names** (supplement; `reports/t3/r1/report_scripts/seed_names.py`, `seed_names.out.txt`): the content scan does not see a run directory named `seed30501`, so the names of all files and directories in the tree were matched too. 50500-50520: 3 names, all in `results/t3_r1/preflight/` (this round's inventory and listing files). 30501-30520: 825 names: 480 in `results/v2_T2_locked/confirmation_v2_0` (the 40 run directories in 12 trees), 240 in `results/ms_r1/calibration`, 48 in `results/v2_refine/preflight` (the seed inventory of the v2.0 round), 24 in `reports/t2_refine_100526/evidence`, 10 in `reports/v2/refine_r2b`, 10 in `results/v2_refine_r2b/diag_30510`, 10 in `tools/v2/diag_seed30510.py`, 3 in this round's preflight directory.

**Reclassification** (stated openly): the first run of the listing (23:07:35 - 23:09:43 UTC; `first_listing/seed_hits.out.txt`) classified 3,410 rows of the 30501-30520 block as `other`: the files under `results/ms_r1/calibration/` (MS-R1's offline calibration: bit-identical replays of the recorded v2.0 exports, mentioned by the prompt as `results/ms_r1/calibration/confirmation_v2_0/`) and `results/v2_refine_r2b/diag_30510/` (R2b's diagnostic tables of seed 30510, computed from the recorded files). I opened them: they are tables computed from the 40 recorded confirmation runs, no training run. The rules of `tools/t3/seed_hits.py` were extended to put the two directories into the class `v20_confirmation_or_copy` (the prompt's "copies, analyses or re-evaluations") and the listing was run again, from scratch, on the same tree; the table above is the second listing (394 seed values, 165,586 json and 57,532 csv files; the first scanned the same counts), in which no row is `other`. The second run wrote the same file names (`seed_hits_{block}.csv`), so it replaced the CSVs of the first; copies made before the second run started are kept with the first run's output in `results/t3_r1/preflight/first_listing/` (the 11,506 + 105 hit rows; compared row by row with the second listing, the sets of (seed, source file, key) are identical and only the class of the 3,410 rows differs). The scan is the inventory tool's own (`seed_inventory.scan`); the file counts of the inventories and of the listings differ only by files this round created while they ran.

## 4. Reference roots (D9)

`python tools/t3/check_reference_roots.py --v20-root /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/confirmation_v2_0 --out results/t3_r1/preflight/v20_reference_runs.csv`:

```
40 runs; 40 directories; 40 with 88 exports; 40 done with exit code 0; 40 tracked gates.json byte-identical
runs that did not pass (q, seed, outcome): [(50, 30510, 'stage2_failure')]
problems: []
```

Per run (`results/t3_r1/preflight/v20_reference_runs.csv`): the directory, 88 exports `u00025.npz` ... `u02200.npz`, the files `state_end_A.pt`, `state_end_B.pt`, `gates.json`, `v2_run_summary.json`, `v2_updates.csv`, `run_config.json`, `status.json` state `done` with exit code 0, and the SHA-256 of `gates.json`; the tracked copy `results/v2_T2_locked/confirmation_v2_0/q{q}/seed{s}/gates.json` of this branch is byte-identical to the reference copy in all 40 cases. Seed 30510 at q = 50 is v2.0's one failed confirmation run (`stage2_failure`, it failed G-A); it stays in the comparison as recorded.

Also present and read-only: `MultiStage/three_stage/T3_IMPLEMENTATION_PLAN_20260924.md` (section 1.2: Phase C = 256 root + 85 ES2 + 171 ES3; its section 2: rows per update 256 / 341 / 512 = 1109) and `docs/Experiments Plan_Multi-stage.md`.

## 5. Machine

`results/t3_r1/preflight/machine.txt`: `nproc` 64; load average 11.20 / 11.52 / 10.86 at 22:39 (other users' jobs; none of them is touched); `/home` 850 GB free of 5.5 TB (84 % used); 754 GB memory; Python 3.12.3, torch 2.5.1+cu121, numpy 2.5.0, **equal to the record** (3.12.3, 2.5.1+cu121, 2.5.0).
