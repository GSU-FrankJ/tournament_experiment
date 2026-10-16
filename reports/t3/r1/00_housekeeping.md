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

Local refs that are behind or ahead of their `origin` refs (recorded, not fixed): `main` d1b84437 (behind `origin/main` 15e5b71a by 107), `ms-r1` 21139215 (behind 71c58904 by 5), `ms-r4` 2f618531 (behind db7577ea by 3; held by the MS-R4 launch worktree, left where it is), `t2-status-pack` e4f93dc8 (behind be2881ff by 1), `feat/multistage-phase0` 73d04257 (ahead of ff9a8f28 by 2). Every other local branch that has an `origin` counterpart is equal to it.

## 2. Branch and worktree

- Worktree path (recorded): `/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-vs-jt-51e472`. It is the worktree the session was started in for this task (name `t3-bw-vs-jt`); it was clean and held no work. It is the "new worktree" of the prompt: no second worktree was created.
- `git switch --no-track -c t3-r1 origin/ms-r4` in that worktree: HEAD `db7577ea5a85910ecfc43db142e872fc1950bf75`, branch `t3-r1`, no upstream. The harness branch `claude/t3-bw-vs-jt-51e472` (at `15e5b71a`, based on `main`, no commits of its own) was left where it is.
- The prompt was saved verbatim as `reports/t3/r1/pi_record/25_t3_r1_prompt.md` before any other file was written (commit `43d720c7`).
- Not touched: `main`, the MS-R4 launch worktree `.claude/worktrees/ms-r4-e1-e2-experiments-2da78b`, any other worktree. No `git clean` was run anywhere.

## 3. Seeds (P0.3)

See the section "Seeds" below, written when the scans finished.

## 4. Reference roots (D9)

`python tools/t3/check_reference_roots.py --v20-root /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/confirmation_v2_0 --out results/t3_r1/preflight/v20_reference_runs.csv`:

```
40 runs; 40 directories; 40 with 88 exports; 40 done with exit code 0; 40 tracked gates.json byte-identical
runs that did not pass (q, seed, outcome): [(50, 30510, 'stage2_failure')]
problems: []
```

Per run (`results/t3_r1/preflight/v20_reference_runs.csv`): the directory, 88 exports `u00025.npz` ... `u02200.npz`, the files `state_end_A.pt`, `state_end_B.pt`, `gates.json`, `v2_run_summary.json`, `v2_updates.csv`, `run_config.json`, `status.json` state `done` with exit code 0, and the SHA-256 of `gates.json`; the tracked copy `results/v2_T2_locked/confirmation_v2_0/q{q}/seed{s}/gates.json` of this branch is byte-identical to the reference copy in all 40 cases. Seed 30510 at q = 50 is v2.0's one failed confirmation run (`stage2_failure`, it failed G-A); it stays in the comparison as recorded.

Also present and read-only: `MultiStage/three_stage/T3_IMPLEMENTATION_PLAN_20260924.md` (section 1.2: Phase C = 256 root + 85 ES2 + 171 ES3, rows per update 256 / 341 / 512 = 1109) and `docs/Experiments Plan_Multi-stage.md`.

## 5. Machine

`results/t3_r1/preflight/machine.txt`: `nproc` 64; load average 11.20 / 11.52 / 10.86 at 22:39 (other users' jobs; none of them is touched); `/home` 850 GB free of 5.5 TB (84 % used); 754 GB memory; Python 3.12.3, torch 2.5.1+cu121, numpy 2.5.0, **equal to the record** (3.12.3, 2.5.1+cu121, 2.5.0).
