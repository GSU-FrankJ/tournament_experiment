# Build log: T=2 status pack (reports/t2_status_100826)

Verbatim commands and outputs, in the order they ran. Date 2026-10-08. Nothing here runs training, a verifier evaluation, a launcher or a round analysis tool.

## P0. Housekeeping

### 0.1 Worktree and remote state

```
$ pwd
/home/fjiang4/tournament_experiment/.claude/worktrees/t2-status-pack-0c452c
$ git rev-parse --show-toplevel
/home/fjiang4/tournament_experiment/.claude/worktrees/t2-status-pack-0c452c
$ git ls-remote origin   # recorded lines: main, ms-r1, ms-r2, ms-r3, t2-refine-pack, tags (before the branch was created)
15e5b71a5d87d85c891804f1b36fecb1b277e8ae	refs/heads/main
71c58904678badd5fdea28e8213567b328f52b5c	refs/heads/ms-r1
e8eb9a08041efc6a6021549294edabb65f7dc047	refs/heads/ms-r2
be4fd2021e5aee52625a994b3566dc18726e0586	refs/heads/ms-r3
4dc604de9b2800d5dba99dfafe95696f630ab4b1	refs/heads/t2-refine-pack
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
$ git ls-remote origin t2-status-pack   # (empty output: the branch does not exist on origin)
```

Checks: `origin/ms-r3` = `be4fd2021e5aee52625a994b3566dc18726e0586` (matches `be4fd202`); `t2-status-pack` does not exist on origin. `origin/main` is `15e5b71a` (the session worktree started on `claude/t2-status-pack-0c452c` at that commit); the new branch is cut from `origin/ms-r3`, not from `main`.

```
$ git fetch origin ms-r3 ms-r2 ms-r1 t2-refine-pack
$ git checkout -b t2-status-pack be4fd2021e5aee52625a994b3566dc18726e0586
Switched to a new branch 't2-status-pack'
$ git log --oneline -1
be4fd202 docs: apply the MS-R3 fact-check corrections and add the ledger
```

Worktree path: `/home/fjiang4/tournament_experiment/.claude/worktrees/t2-status-pack-0c452c` (session worktree `t2-status-pack-0c452c`, branch `t2-status-pack`).

### 0.2 Prompt saved

`pi_record/21_t2_status_pack_prompt.md` was written right after the branch was created and before anything else. SHA-256 and size:
```
1cb3e793b70b040609d021ceee667a4f7d719e1086470846b3564e570e918962  reports/t2_status_100826/pi_record/21_t2_status_pack_prompt.md
39600 reports/t2_status_100826/pi_record/21_t2_status_pack_prompt.md
```

### 0.3 Inventory of the D1 sources at be4fd202

Present / tracked / size in bytes. Appendices A and B are part of the saved prompt (item 7 of D1). Every item is present and tracked: no stop-and-report.

```
reports/t2_refine_100526/README.md present=yes tracked_at_be4fd202=yes size=12999
reports/t2_refine_100526/100526report.md present=yes tracked_at_be4fd202=yes size=218110
reports/ms/r1/summary.md present=yes tracked_at_be4fd202=yes size=7573
reports/ms/r1/05_decision_inputs.md present=yes tracked_at_be4fd202=yes size=12491
reports/ms/r1/04_pilot.md present=yes tracked_at_be4fd202=yes size=59767
reports/ms/r1/02_preregistration.md present=yes tracked_at_be4fd202=yes size=43709
reports/ms/r2/summary.md present=yes tracked_at_be4fd202=yes size=6885
reports/ms/r2/05_decision_inputs.md present=yes tracked_at_be4fd202=yes size=30178
reports/ms/r2/04_pilot.md present=yes tracked_at_be4fd202=yes size=126827
reports/ms/r2/02_preregistration.md present=yes tracked_at_be4fd202=yes size=20984
reports/ms/r3/summary.md present=yes tracked_at_be4fd202=yes size=8494
reports/ms/r3/05_decision_inputs.md present=yes tracked_at_be4fd202=yes size=25205
reports/ms/r3/04_pilot.md present=yes tracked_at_be4fd202=yes size=142098
reports/ms/r3/02_preregistration.md present=yes tracked_at_be4fd202=yes size=23522
reports/ms/r1/pi_record/17_ms_r1_prompt.md present=yes tracked_at_be4fd202=yes size=34707
reports/ms/r1/pi_record/18_g1_reply.md present=yes tracked_at_be4fd202=yes size=5756
reports/ms/r2/pi_record/19_ms_r2_prompt.md present=yes tracked_at_be4fd202=yes size=24678
reports/ms/r3/pi_record/20_ms_r3_prompt.md present=yes tracked_at_be4fd202=yes size=28149
reports/ms/r2/pi_record/01_factcheck.md present=yes tracked_at_be4fd202=yes size=51105
reports/ms/r3/pi_record/01_factcheck.md present=yes tracked_at_be4fd202=yes size=106075
protocols/v2_T2_locked_v2_0.json present=yes tracked_at_be4fd202=yes size=36401
protocols/v2_T2_locked_v2_0.md present=yes tracked_at_be4fd202=yes size=18821
results/ms_r1 tracked files at be4fd202: 2177
results/ms_r2 tracked files at be4fd202: 1884
results/ms_r3 tracked files at be4fd202: 3556
```

## P1. Evidence pack

Sources read in the order of D1 before any item was selected: 100526 README and report (sections 0, 3, 6), the MS summaries / decision inputs / pilot reports / pre-registrations, the PI record (prompts 17, 19, 20, the G1 reply), the MS-R2 and MS-R3 fact-check ledgers (listed as evidence, read for their dispositions), the tracked per-round CSV/JSON records, protocol v2.0, and Appendices A and B of prompt 21.

```
$ python reports/t2_status_100826/report_scripts/build_pack.py
built reports/t2_status_100826: 89 evidence files, 14 figure files
$ python reports/t2_status_100826/report_scripts/build_pack.py --check
build_pack --check: PASS (0 differences)
$ (cd reports/t2_status_100826/evidence && sha256sum -c SHA256SUMS | tail -2)
results/ms_r3/supervised_screen/summary_median.csv: OK
results/ms_r3/v20_reproduction_checks.json: OK
$ (cd reports/t2_status_100826/figures && sha256sum -c SHA256SUMS)
FIG-01_tie_profile_runs.png: OK
FIG-02_trajectory_decomposition_bb.png: OK
FIG-03_trajectory_decomposition_st.png: OK
FIG-04_paired_abs_peak_vs_t1.png: OK
FIG-05_paired_noise_landing.png: OK
FIG-06_first_layer_d_weights.png: OK
FIG-07_trajectory_decomposition.png: OK
FIG-08_scatter_remainder_vs_smoothing_change.png: OK
FIG-09_cal_fig1_scatter_R_Delta_vs_peak.png: OK
FIG-10_FG-13_fig3_endA_profile.png: OK
FIG-11_FG-12_fig1_peak_trajectory_q50.png: OK
FIG-12_F1_abs_peak_per_run.png: OK
FIG-13_F2_gap_decomposition.png: OK
$ wc -l reports/t2_status_100826/evidence/manifest.csv
159 reports/t2_status_100826/evidence/manifest.csv
$ du -sh reports/t2_status_100826/evidence reports/t2_status_100826/figures
5.2M	reports/t2_status_100826/evidence
3.7M	reports/t2_status_100826/figures
```

Selection rule used: every tracked file smaller than 1 MiB that the report cites is copied under its original path; no cited file reached 1 MiB, so no `referenced (tracked, large)` row exists. Files not copied on purpose: run logs (`results/ms_r3/logs/*.log`, D8), weight exports, freeze arrays, per-run directories. Items of the 100526 pack are cited in place (status `cited in t2_refine_100526`); each was verified against the SHA-256 of that pack's manifest at build time. Extensions to the prompt's ID prefixes: `TBL-<name>` rows in the manifest for the tables produced by `tables.py` (status `generated (table text)`, SHA-256 of the rendered text), and `T2R:100526report` for the 100526 folder report.

## P2. Report, index, handoff

Written to D4/D6/D7; tables produced by `tables.py` and injected between markers; two figures by `figures.py`. Commits: `a279bb93` (pack extension), `44c9344f` (report, README, handoff, checkers), `5b67f487` (STATE pointer).

## P3. Checks and independent fact-check

Five read-only agents (slices: sections 0-6; 7.1-7.4; 7.5-7.7; 8-11 and coverage; Appendix B of the prompt against the records) ran on commit `5b67f487`. Their findings and the dispositions are in `01_factcheck.md`. One defect found by two of them: the table blocks of report.md were empty because the injector skipped empty blocks (so `tables.py --verify` and `check_numbers.py` had passed vacuously); fixed, and an empty block is now a difference.

```
$ tables.py --verify report.md
tables --verify: PASS
$ check_numbers.py
check_numbers: PASS (116 numeric sentences checked, 0 without a citation)
$ build_pack.py --check
build_pack --check: PASS (0 differences)
$ check_links.py
check_links: PASS (21 relative links checked)
$ sha256sum -c (evidence, figures)
0
0
```

## P4. Push and verification

```
$ git push origin t2-status-pack   (new branch, no force)
head: 58d720a3c9e7157ea3b4528313663a220ed5d2c0
$ git ls-remote origin t2-status-pack
58d720a3c9e7157ea3b4528313663a220ed5d2c0	refs/heads/t2-status-pack
$ check_links.py --ref origin/t2-status-pack
folder files in working tree: 119; missing from origin/t2-status-pack: 0
check_links: PASS (21 relative links checked)
$ git ls-remote origin main ms-r1 ms-r2 ms-r3 (unchanged)
15e5b71a5d87d85c891804f1b36fecb1b277e8ae	refs/heads/main
71c58904678badd5fdea28e8213567b328f52b5c	refs/heads/ms-r1
e8eb9a08041efc6a6021549294edabb65f7dc047	refs/heads/ms-r2
be4fd2021e5aee52625a994b3566dc18726e0586	refs/heads/ms-r3
```
