# MS-R4 P0: housekeeping

Date 2026-10-13. Prompt: `reports/ms/r4/pi_record/23_ms_r4_prompt.md` (the round's prompt, saved verbatim as the first action of the round; SHA-256 `aafe2567303e4815ab871682b4730165d9af73b89f3b305292b11be33c41c9cd`, 48,804 bytes). `§` and `D#` refer to it. Every command below was run from the session's worktree; outputs are verbatim unless marked.

## 1. Prompt and the coworker's document

- The prompt was written to `reports/ms/r4/pi_record/23_ms_r4_prompt.md` before any other step.
- **`Multistage1011261.docx` was not provided.** `find / -iname 'Multistage1011261*'` (whole filesystem, `/proc` excluded) and `find /home/fjiang4/tournament_experiment -iname '*1011261*'` find nothing; the `MultiStage/` folders of the primary checkout hold the earlier documents only (`Experiments Plan_Multi-stage081626.docx`, `Multi-stage discussion081826.docx`, `Multi-stage discussion082526.docx`). Appendix A of the prompt (the coworker's text, verbatim) is therefore the only source of the coworker's plan, and the SHA-256 `699f73f3...6e10e8` of the document **could not be checked**. This is not a stop condition (§1 item 1: "if the document is provided"); it is recorded here and in the pre-registration.

## 2. Refs and branch

`git ls-remote origin refs/heads/main refs/heads/ms-r3 refs/heads/t2-status-pack 'refs/tags/*'` (2026-10-13 15:2x):

```
15e5b71a5d87d85c891804f1b36fecb1b277e8ae	refs/heads/main
be4fd2021e5aee52625a994b3566dc18726e0586	refs/heads/ms-r3
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

- `origin/t2-status-pack` = `be2881ff31474ce5147bef29f9ff0e785e1d2b95` and `origin/ms-r3` = `be4fd2021e5aee52625a994b3566dc18726e0586`: **both as the prompt states** (`git fetch origin` then `git rev-parse origin/t2-status-pack origin/ms-r3 origin/main` prints `be2881ff...`, `be4fd202...`, `15e5b71a...`).
- The tags `t2-v2-lock-v2.0` (`32e68cfd...`, peeled `1d6d4d00...`) and `t2-v2-confirmation-v2.0` (`4724ca00...`, peeled `d2e377d0...`) exist and are untouched.
- The session's worktree was created by the app on `claude/ms-r4-e1-e2-experiments-2da78b` at `origin/main` (`15e5b71a`, a commit that only added `reports/sandbox_fit_tip.py` and its results). Branch created in this worktree: `git switch -c ms-r4 origin/t2-status-pack` (HEAD `be2881ff31474ce5147bef29f9ff0e785e1d2b95`). Git set `origin/t2-status-pack` as its upstream; `git branch --unset-upstream ms-r4` removed it so that no `git push` can reach `t2-status-pack` by accident (every push names `origin ms-r4` explicitly).
- **Worktree path:** `/home/fjiang4/tournament_experiment/.claude/worktrees/ms-r4-e1-e2-experiments-2da78b`. `main` and every other worktree are untouched.
- **Local refs that are behind** (recorded, not fixed): local `main` `d1b84437` is 107 commits behind `origin/main` `15e5b71a`; local `ms-r1` `21139215` is 5 behind `origin/ms-r1` `71c58904`; local `t2-status-pack` `e4f93dc8` (the worktree `t2-status-pack-0c452c`) is 1 behind `origin/t2-status-pack` `be2881ff` (the PI-reply commit). `ms-r2`, `ms-r3`, `t2-refine-pack` and `v2-t2-refine` equal their `origin` refs.

## 3. Seeds

```
python tools/v2/seed_inventory.py --block 10511 10530 --out results/ms_r4/preflight/seed_inventory_10511_10530.csv
```

(run in tmux, `results/ms_r4/preflight/seed_inventory_10511_10530.stdout`, exit 0). Output, verbatim:

```
scanned 132954 json, 43251 csv files; 353 distinct seed values
block 10511-10530: 0 collisions
recorded seeds within +-1000 of the block: [10001, ..., 10020, 10101, ..., 10130]
```

(the third line lists the seeds of other experiments within 1000 of the block; the CSV `results/ms_r4/preflight/seed_inventory_10511_10530.csv` holds the full inventory). **0 collisions**, so no record of the T = 2 v2 or MS lines (`results/v2_*`, `results/ms_r*`, `reports/v2/`, `reports/ms/`, `reports/t2_*`, `protocols/`, in any worktree) and no record of any other experiment holds a seed of 10511-10530. The scan covered every worktree of `/home/fjiang4/tournament_experiment` (`.git` and `.venv` skipped) and ran before any file of this round that mentions the block in a `seed` key existed.

## 4. Reference roots (D8)

| root | path | what this round reads | verified |
|---|---|---|---|
| MS-R3 runs (C-MS6, curve validation) | `/home/fjiang4/tournament_experiment/.claude/worktrees/p2-gate-ms-base2400-e41857/results/ms_r3/pilot` | the 20 `t1_st_s16` and 20 `relu_st_s16` runs of seeds 10501-10510 (q 50 and 60) | the worktree's HEAD is `be4fd2021e5aee52625a994b3566dc18726e0586` = `origin/ms-r3`; 40 runs found |
| `rehearsal_v2_0` (C-R7) | `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0` | the 20 reference runs, by `tools/v2/cr2_compare.py` | 20 run directories (`q{50,60}/seed10501-10510`); the worktree's HEAD is `b55d3890` (`v2-t2-refine`, the tag `t2-v2-main-v2.0`) |
| C7 canonical worktree | `/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2` | the C7 reference of `tests/test_v2_r2b.py` (`results/v2_pilots/phase2_regression/`), used by `tests/test_ms_runner.py::test_c7_reference_is_unchanged` | exists, HEAD `28e14b5c` (`v2-stagewise-pilots`) |
| tracked MS-R1 and MS-R3 records | `results/ms_r1/pilot/`, `results/ms_r1/analysis/per_run.csv`, `results/ms_r3/analysis/per_run.csv` of the branch | the preamble numbers (§2.3 (a)) | 120 `ms_checks_stage2.csv` files in `results/ms_r1/pilot/`, both `per_run.csv` present |

What C-MS6 and the curve validation read from the MS-R3 root, counted on the 40 runs: **5,440 weight exports** (136 per run), **160 freeze arrays** (`freeze_stage{1,2}_{final,development}.npz`, 4 per run), 40 per-update series (`ms_updates.csv`, which holds the five stream positions after every update), **80 check tables** (`ms_checks_stage{1,2}.csv`), 40 `gates.json`, 40 `ms_binmaps_stage2.npz`, 40 `continuation_table_stage1.npz`, 40 `train_history.json`. No reference file is missing.

## 5. Machine

- `nproc` 64; load average 17.88 / 14.91 / 13.72 at 15:27 (other sessions' jobs are running; none is touched); `/home` 5.5 T, 865 G free (84 % used); memory 754 GB, 696 GB available.
- Python 3.12.3, torch 2.5.1+cu121, numpy 2.5.0 (CPU, 1 torch thread): **equal to the versions embedded in the protocol's records** (`protocols/v2_T2_locked_v2_0.json`, `records.50.versions` and `records.60.versions`: python 3.12.3, torch 2.5.1+cu121, numpy 2.5.0; device cpu). Other packages of the analysis tools: pandas 3.0.3, scipy 1.18.0, matplotlib 3.11.0.
- Interpreter for everything: `/home/fjiang4/tournament_experiment/.venv/bin/python`; `pytest tests` only; every run single-threaded (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`) in tmux.
