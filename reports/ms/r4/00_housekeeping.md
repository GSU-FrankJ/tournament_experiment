# MS-R4 P0: housekeeping

Date 2026-10-13. Prompt: `reports/ms/r4/pi_record/23_ms_r4_prompt.md` (the round's prompt, saved verbatim as the first action of the round; SHA-256 `aafe2567303e4815ab871682b4730165d9af73b89f3b305292b11be33c41c9cd`, 48,804 bytes). `§` and `D#` refer to it. Every command below was run from the session's worktree; outputs are verbatim unless marked.

## 1. Prompt and the coworker's document

- The prompt was written to `reports/ms/r4/pi_record/23_ms_r4_prompt.md` before any other step.
- **`Multistage1011261.docx` was not provided.** `find / -iname 'Multistage1011261*'` (whole filesystem, `/proc` excluded) and `find /home/fjiang4/tournament_experiment -iname '*1011261*'` find nothing; the `MultiStage/` folders of the primary checkout hold the earlier documents only (`Experiments Plan_Multi-stage081626.docx`, `Multi-stage discussion081826.docx`, `Multi-stage discussion082526.docx`). Appendix A of the prompt (the coworker's text, verbatim) is therefore the only source of the coworker's plan, and the SHA-256 `699f73f3...6e10e8` of the document **could not be checked**. This is not a stop condition (§1 item 1: "if the document is provided"); it is recorded here and in the pre-registration. **Superseded in part by Addendum 1 at the end of this file:** an English rendering of the note was found later the same day under a different name and in another worktree.

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

## Addendum 1 (2026-10-13, after the P1 report): the coworker's note was found

The PI pointed to `reports/ms/r4/pi_record/` of the branch `origin/t2-status-pack` as the place where the coworker's document had been put. The remote branch does not hold it (`git fetch origin`; `git ls-remote origin refs/heads/t2-status-pack` is still `be2881ff31474ce5147bef29f9ff0e785e1d2b95`, and `git ls-tree -r origin/t2-status-pack -- reports/ms/r4` is empty; no other remote branch or pull-request ref holds a path with `1011261` or under `reports/ms/r4/pi_record` except `ms-r4`'s own prompt file). The file is an **untracked file of the worktree of the session "T=2 status pack PI reply"**:

- path `/home/fjiang4/tournament_experiment/.claude/worktrees/t2-status-pack-pi-reply-2ec71b/reports/ms/r4/pi_record/Multistage101126(1).md` (branch `claude/t2-status-pack-pi-reply-2ec71b` at `15e5b71a`, where `reports/ms/` is untracked); modified 2026-10-13 13:47 UTC, before the round started; 3,547 bytes, 18 lines, UTF-8; **SHA-256 `ea3e6198bb8fe08b750dda1232efe0d4c403fd4398c88ad36a742e248a5e3138`**.
- **Why the first search missed it:** that search (`find / -iname 'Multistage1011261*'` and `find <repo tree> -iname '*1011261*'`) keyed on the exact name of the docx; the file is named `Multistage101126(1).md`, and I did not look inside the other worktrees' `reports/ms/r4/`. "Not provided" in section 1 was therefore too strong.
- **What it is:** an **English rendering of the coworker's note** (page markers `===== Page 1 =====` ... `===== Page 4 =====`, the two highlighted headings as `{.mark}` spans), **not the `.docx`**. Its SHA-256 is that of the `.md`, so the SHA-256 `699f73f30010bc59777469c8cada4c8f81c2fa3bbde6fa39e28349b1126e10e8` of `Multistage1011261.docx` **still cannot be checked**; the docx itself has not been seen.
- **Stored unchanged** at `reports/ms/r4/pi_record/Multistage101126(1).md` (`cp -p`, `cmp` identical, same SHA-256).
- **Comparison with Appendix A** (the Chinese text of the prompt, "paragraphs in order"): read paragraph by paragraph, the note makes the same statements in the same order (the plan, the baseline v2.0, the seeds, item 1 with LeakyReLU and the smooth mean and the 0.04 target, item 2 with the three sampling arms and the 20 % target, the deliverables, the lock-then-confirm rule). Mechanical check (`python3 -I` on the two texts): all 43 numeric tokens of Appendix A occur in the note with the same multiplicity (10511-10530, 30501-30520, 40501-40520, 50, 60, 20, 0.04, 95, 1600, 600, 2800, 0.5, 20 %, ...); the note's only additional numeric tokens are the four page numbers; the identifiers `relu_st_s16`, `t1_st_s16`, `G-A`, `G-F`, `G-N`, `G-S`, `MS-R3`, `v2.0`, `s16`, `DP-BR`, the symbols `<= 0.04`, `alpha = 0`, `alpha = 0.5`, `eta_2/DW` and the circled numbers 1-3 occur in both. **No difference in any number, seed, threshold, arm or rule.** The design, the arms, the criteria and the parameters of this round are therefore unchanged by the find.

## Addendum 2 (2026-10-14): P2 and P3

- **Launch worktree:** `/home/fjiang4/tournament_experiment/.claude/worktrees/ms-r4-e1-e2-experiments-2da78b` (branch `ms-r4`, HEAD = `origin/ms-r4` = `2f618531` at the launch), where the config hashes were computed; the config gate of Addendum 2 of `02_preregistration.md` confirmed it (280 of 280 identical).
- **The launch:** tmux session `msr4_pilot`, started 2026-10-13 23:50:29 UTC (`results/ms_r4/pilot/launch_20261013_235029.json`, state `done`, record written 2026-10-14 00:54:02 UTC), 40 workers, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, interpreter `/home/fjiang4/tournament_experiment/.venv/bin/python` (`python` is not on the PATH of this machine). After the pool had finished (no run process was left) the idle tmux session, which was started from this session, was closed (`tmux kill-session -t msr4_pilot`); nothing of anyone else's was touched.
- **Machine at the launch:** `nproc` 64; load average 11.10 / 11.61 / 11.70; 924 GB of 5,952 GB free; other sessions' jobs were running and were not touched. Launch facts: block `launch` in `04_e1_actor.md` section 1.
- **P3 worktree:** the reports, `report_scripts/pilot_tables.py` and the commits of P3 were made in `/home/fjiang4/tournament_experiment/.claude/worktrees/gate-g1-decision-launch-55df61` on the local branch `claude/ms-r4-p3` (from `origin/ms-r4` `2f618531`), because the file tools of that session write only inside it. The tracked records (`results/ms_r4/pilot/` 13 records per run, `curves/`, `analysis/`) were copied from the launch worktree with `rsync` and compared by SHA-256 (3,643 pilot files and 330 analysis and curve files, 0 differences). `ms-r4` is pushed from `claude/ms-r4-p3` as a fast-forward.
- **Refs left behind (owner-side):** the local `ms-r4` of the launch worktree stays at `2f618531` until it is fast-forwarded to `origin/ms-r4`; it holds the untracked copies of the same result files, which have to be removed (or `git merge --ff-only` will refuse) before the fast-forward. The other stale local refs of section 2 are as before. Nothing was deleted or moved in the launch worktree; the untracked heavy files (weights, `train_history.json`, `state_end_*.pt`, freeze arrays, `run.log`) stay there only.
- **No result was changed, deleted or overwritten**; nothing under `results/*/convergence/` is touched by this line.
