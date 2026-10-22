# T3-R4, P1: the lock drafts (protocol T3-1.1) and the plan of the confirmation

Date: 2026-10-22. Branch `t3-r4` (from `origin/t3-r3` `df2365d9`). **Generated** from `reports/t3/r4/report_scripts/01_lock.template.md` by `build_report.py`; the tables by `lock_tables.py`. This file is committed **before** the lock (section 9). The binding text is the prompt (`pi_record/29_t3_r4_prompt.md`); the decision to lock `bwA` is the PI's at gate G2 of round T3-R3 (`reports/t3/r3/summary.md`, `05_development.md`: `bw` 37 of 40, `bwA` 39 of 40, `bwAB` 38 of 40 on development seeds 70501-70520).

## 1. What is locked

**Protocol T3-1.1 = protocol T3-1.0 plus one config key.** T3-1.0 (`protocols/v2_T3_locked_v1_0.{json,md}`, unchanged) is the locked `bw`: backward-induction training of stage 3, then stage 2, then stage 1 through `run/run_ms_stagewise.py`. T3-1.1 adds `stage3_retry = {"delta_gt": 0.005, "max_retries": 1}` (`run/ms_retry.py`, `run/run_ms_stagewise.py`). The code is the code of round T3-R3: the diff `git diff --stat b565f0e8 df2365d9 -- agents envs utils run tools/ms tools/v2` is empty (`00_housekeeping.md` section 1), and the launcher checks the same diff against HEAD at every launch. Behaviour (`reports/t3/r3/02_proposal.md` section 2.1): the trigger is read at the development check at local update 1600 of the stage-3 phase and fires iff `Delta` > 0.005, or `Delta`, `s` or `C` is not finite, or `s` <= 0; a fired run re-initialises stage 3 once in place (derived seeds and streams) and retrains it, and the second attempt is frozen whatever its own check says; the artefacts of attempt 1 go to `attempt1/` and `retry.json` records the trigger values and the derived seeds; a run whose trigger does not fire is bit-identical to `bw`. Budget of a fired run: 5,400 updates, 2,764,800 policy rows, 108,000 minibatch steps, 216 development checks, 152 exports plus 64 in `attempt1/`.

**Paper wording for A (verbatim):** "a verifier-gated restart of the terminal stage: when the development-tier one-step deviation gain at the end of the stage-3 phase exceeds the gate's limit 0.005, stage 3 is re-initialised once and retrained, and the second attempt is kept; its cost is part of the method's cost."

The files (`reports/t3/r4/lock_draft/v2_T3_locked_v1_1.{json,md}`, written by `tools/t3/t3r4_make_lock.py`; SHA-256 of the JSON `97bb9146c379c957356706f463361c2ff48c2724fac31b9114e7aba5a2a7e38e`, of the MD `90a1047b745174209a0d05c8826cce868b8506d89cb1c1fc3239104ef0063c47`) hold no timestamp, no full commit hash (only the abbreviated base commit `b565f0e8` of the frozen-code diff) and no absolute path; the lock written into `protocols/` must equal them byte for byte.

| item | value |
|---|---|
| records | the v2.0 record of each q (`protocols/v2_T2_locked_v2_0.json`; the record itself has T = 2, the run configs set `game.T` = 3) |
| parameters file | `reports/t3/r1/params_bw_t3.json` |
| phases, budgets (updates) | [3, 2, 1], {"3": 1600, "2": 1600, "1": 600} |
| learning rate | 3e-4 constant, then linear 3e-4 -> 3e-5 over local updates 1201-1600 at stages 3 and 2 and over 1-600 at stage 1 |
| episodes per update, starts | 512, {"scheme": "bin_balanced"} |
| rule, init digest, clamp likelihood | disabled, True, `density` |
| checks, snapshots, exports | every 25, 20, 25 updates |
| restart (the only change against T3-1.0) | `stage3_retry = {"delta_gt": 0.005, "max_retries": 1}` |
| evaluated policy | the last iterate (Beta means) on the final and the development verifier tier |
| seed block, runs | 80501-80540 at q in [50, 60]: 80 runs, arm `bwA` |
| run names, directories | `t3r4_q{q}_s{seed}_bwA`, `results/t3_r4/confirmation/q{q}/seed{seed}/bwA` |
| fields set per run | `seed`, `run`, `record.run`, `record.seed`, `record.output_dir`, `pilot` |

## 2. The confirmation rule, the operating characteristic and the failure counting

The rule was fixed before any development run, in T3-R3's pre-registration (`reports/t3/r3/04_preregistration.md` section 9, D6), and is binding: **the confirmation passes iff at most 6 of the 80 runs fail and at most 4 of the 40 runs fail at each q** (seeds 80501-80540, the same 40 seeds at q = 50 and q = 60). A run passes iff it exits with code 0, its global-RNG assertion is ok, its evaluation is valid on both tiers, and G-A (`eta_3` <= 0.005, `rmse_pos_T_over_eT0` <= 0.05, `tail_mean_T_over_eT0` <= 0.02), G-F (`Gmax_full_over_dw` <= 0.01) and G-N (development minus final: `eta_3` and `Gmax_full` <= 0.001) hold, every `<=` inclusive at full float64 precision (`reports/t3/r1/prereg_criteria.json`).

Operating characteristic, recomputed here with `scipy.stats.binom` (not with `t3r4_make_lock.py`) and set against the values in the lock; independent per-run failure probability f, the same at both q:

| per-run failure probability f (both q) | P(at most 6 of 80 and at most 4 of 40 per q fail), recomputed | in the lock | agree to 1e-6 |
|---|---|---|---|
| 0.01 | 0.999893 | 0.999893 | yes |
| 0.03 | 0.981819 | 0.981819 | yes |
| 0.05 | 0.864763 | 0.864763 | yes |
| 0.1 | 0.270786 | 0.270786 | yes |
| 0.15 | 0.029886 | 0.029886 | yes |

(The prompt quotes 0.982 at f = 0.03, 0.865 at 0.05, 0.271 at 0.10 and 0.030 at 0.15.)

**Failure counting** (in the lock, section 5): every planned run is reported and no seed is replaced or re-drawn. Infrastructure failures (a negative return code, an out-of-memory kill, ENOSPC) are recorded with evidence in `results/t3_r4/confirmation/infra_events.json`; such a run is moved to `crashed/` and re-run once, and a second infrastructure failure stops the confirmation without a verdict. Any other nonzero exit and any global-RNG violation is a failed run. An evaluation error or an invalid evaluation is a failed run unless a re-run of the locked evaluator on the same weights succeeds. A re-initialisation inside a run is part of the run, never a re-run. If the confirmation fails, nothing is changed or tried in this round: the PI decides the next step.

## 3. The tools, and what the launcher refuses

`tools/t3/t3r4_launch.py` (arm `bwA` only; seeds exactly 80501-80540 at q in {50, 60}); `t3r4_checks.py` (the post-launch checks), `t3r4_eval.py`, `t3r4_analysis.py`, `t3r4_blind.py`, `t3r4_post_launch.sh` (checks, evaluation, analysis and blind recomputation, stopping at the first nonzero exit), `t3r4_make_lock.py`, `t3r4_repro.py` (C-REPRO). They reuse by import, unchanged, the `t3_*`, `t3c_*`, `t3r2_*` and `t3r3_*` tools; the lock lists the SHA-256 of every file the confirmation runs. At a launch the launcher refuses unless: the T3-1.1 record is the last record of `protocols/LOCK`; the lock JSON and MD match its SHA-256; every listed file matches; HEAD is the LOCK-record commit; `git status --porcelain` shows nothing outside `results/`; the diff of `agents envs utils run tools/ms tools/v2` since `b565f0e8` is empty; `--root` is absolute; each config equals the T3-1.1 reference but for the six run fields. Tests: `tests/test_t3r4_*.py` (see `02_checks.md`). A failed deterministic check is a stop-and-report: the post-launch script stops at it (exit code 1), and the evaluation, analysis and blind recomputation are then run by hand only if the report says so. If the analysis finishes with a failed reporting step (exit code 4), the verdict is already in `analysis/headline.json`, the script stops before the blind recomputation, and the blind tool is run by hand.

## 4. The plan of the 80 runs

`reports/t3/r4/confirmation_plan.csv` (written by `t3r4_launch.py --plan-csv`, which checks every config against the lock drafts; SHA-256 of the file `3e3d445689bcafa0afca21f5a6e4f363e7851fbcec3a84bce594de1dc86af633`): run name, output directory, config SHA-256 (as written and with the output directory made relative to the root), the planned init digest of attempt 1 and of attempt 2 (the run is constructed without training). The post-launch checks compare the recorded digests with these.

| run | config SHA-256 (16) | init digest, attempt 1 (16) | init digest, attempt 2 (16) |
|---|---|---|---|
| `t3r4_q50_s80501_bwA` | 83467c2ec9807706 | 986a8a848059f66e | 206cbc4c8b5f1a94 |
| `t3r4_q50_s80502_bwA` | f66b4959775a4ff7 | de77c5a1069bfb79 | 769a3efc03535eb7 |
| `t3r4_q50_s80503_bwA` | f275a5a8617c5721 | 2274851769f6ef66 | bc3cdeb867e54279 |
| `t3r4_q50_s80504_bwA` | ef6af99ac101c3f3 | c064226017365e24 | 50ffbaf35b1c9986 |
| `t3r4_q50_s80505_bwA` | 67bca0da0ef681bc | 62b779c85825d21c | 2052b94dd5857a23 |
| `t3r4_q50_s80506_bwA` | 80119d8cd7c3d091 | 23282058b8d5e2e0 | 0104e7670e323249 |
| `t3r4_q50_s80507_bwA` | b33ea65274506318 | 346f7506694842bf | c25b627c44db660b |
| `t3r4_q50_s80508_bwA` | cc564c8e35a6f6c7 | 144cc6caf2e33029 | a3564ffebaf0e1ce |
| `t3r4_q50_s80509_bwA` | 20bba649c3b2fd79 | ddcf25e0c0b6543f | 226fb7f82f22d987 |
| `t3r4_q50_s80510_bwA` | 4ecd5d4966a73689 | c391124e75140a9f | 589ba132bb3141b6 |
| `t3r4_q50_s80511_bwA` | cac43a036abefd15 | 298dfeeb4867fffa | 901d3125804e8788 |
| `t3r4_q50_s80512_bwA` | be2fde56030a0404 | fa13cc0966f564e7 | e8240c57d7a83a6e |
| `t3r4_q50_s80513_bwA` | 2bcd44d2ef21490d | 255a267ddfac8c0e | 7d75958a444055eb |
| `t3r4_q50_s80514_bwA` | 30c7894eb1265653 | d46ebda988a0f632 | 3f6230a713daf745 |
| `t3r4_q50_s80515_bwA` | 551a0fa2616420ad | 563b40e0cdf67cbe | 690c39dbd332b933 |
| `t3r4_q50_s80516_bwA` | 764c253fa250201f | 4025fee280492f47 | 4f6f7663357fae0e |
| `t3r4_q50_s80517_bwA` | 944529c59ae4a14b | 910ababeed196274 | 30c5fc9e4fb8b24c |
| `t3r4_q50_s80518_bwA` | 4f4c4629ff9fcda6 | e5c691355ff4508c | dd620044dfefd176 |
| `t3r4_q50_s80519_bwA` | 7f81769b6beb6755 | ec3efb6152aae52d | af208094f703dc4b |
| `t3r4_q50_s80520_bwA` | 63ec8e6f0d2c3cc5 | 64670261ef42ea5a | b1e51e8f84e808ca |
| `t3r4_q50_s80521_bwA` | 21f23ca304adef3c | fee353dd05acd4b3 | 143210afafe965b6 |
| `t3r4_q50_s80522_bwA` | 829ac4e218acec70 | 03df996d51a5a322 | 8c26b8286746ed16 |
| `t3r4_q50_s80523_bwA` | c43f977640845689 | 8421b50612c3bcda | ff4e9d901629b89b |
| `t3r4_q50_s80524_bwA` | 6b278532444d19a9 | d856f487ab4ed72b | b10b0e2270e32082 |
| `t3r4_q50_s80525_bwA` | 2b7b0004bef2d252 | fc5f707cc7cee316 | f1ba99950783aead |
| `t3r4_q50_s80526_bwA` | b0862378e1c03e5c | 06b5719e6cd083c9 | b924b9e4ef109664 |
| `t3r4_q50_s80527_bwA` | 38b3eda961d29a67 | b62221f11e16ad0f | fbde3f017474591b |
| `t3r4_q50_s80528_bwA` | e3a53b972322f0a3 | c5fe71aa60366d11 | e1f476486624ebcc |
| `t3r4_q50_s80529_bwA` | 34eab548616ddacc | a3b91ac07d8d282a | fa0c190a5073dd50 |
| `t3r4_q50_s80530_bwA` | 659bd82ae3cad3a8 | 41d44e102fe44b71 | 38ec971087af03f2 |
| `t3r4_q50_s80531_bwA` | 27d309aafd9a2cad | d92e5348b2557a92 | 1127f1c5b83db3b0 |
| `t3r4_q50_s80532_bwA` | f891294386d7cd13 | 1102661d25b39336 | fffaa636c0201b9e |
| `t3r4_q50_s80533_bwA` | 5ba55d752bf99b0a | bc64415a85596207 | 35ba2a78ca2490b9 |
| `t3r4_q50_s80534_bwA` | d1cac6b642695ea1 | cab719adb79ab434 | 7b51df09e79d1ed7 |
| `t3r4_q50_s80535_bwA` | 7ed9112e335ee847 | 83ddb5db07fca428 | ddcb7ca6c9528d8d |
| `t3r4_q50_s80536_bwA` | b4a0a7a094a6e8d2 | 10de2e108a41fe14 | 90f282ee848ee785 |
| `t3r4_q50_s80537_bwA` | ed4765a77d4f94c8 | 87182322e4d4b77f | 604f5c73ec9af3da |
| `t3r4_q50_s80538_bwA` | 773ea0405128a65f | 250b859ecfb00295 | 71e7f9db0a012821 |
| `t3r4_q50_s80539_bwA` | 9e8305ac683e7e90 | 44db0a6d3dd449b0 | d0d5a6f237ed89b1 |
| `t3r4_q50_s80540_bwA` | 239f5b0719efcce2 | 582246b72a55a2ac | 026665ee060bc4f4 |
| `t3r4_q60_s80501_bwA` | f779674633354121 | 9dba4fd58dffb99c | 79657f4bae2f80af |
| `t3r4_q60_s80502_bwA` | e175168c41be0b6f | 644c2ff01b4bc8ad | eec87e472a7b4c85 |
| `t3r4_q60_s80503_bwA` | 79673c629ee032b7 | bfab5f5d3ae85eaa | bdd28150defe7caa |
| `t3r4_q60_s80504_bwA` | 57dea338b5437d0e | b9b5d052824b53b7 | 6dac8e2da9859e1b |
| `t3r4_q60_s80505_bwA` | 2ca92597bfc961aa | a6a711eaa63bf4fa | 03f98a451bfefbfe |
| `t3r4_q60_s80506_bwA` | c72974fc0c61df98 | f144b254c8e4c52b | 4780c3b167207abf |
| `t3r4_q60_s80507_bwA` | 485df2df83102545 | f8f26a536c75d802 | 8be7caf8c7325a17 |
| `t3r4_q60_s80508_bwA` | 8a9072ccb3a4334b | c07aa93a1e100a9c | 7eaed29d04f89337 |
| `t3r4_q60_s80509_bwA` | cb257c6cbde26690 | 933cbfb5e78a44c0 | 1e74249cbf2f8fcd |
| `t3r4_q60_s80510_bwA` | dc35c40bf9c31bc5 | 5f336233c3b90563 | 45e9f22ff5bdb473 |
| `t3r4_q60_s80511_bwA` | b928a9f6739f04a1 | 2c034bcb42801630 | dfc7b9959281b74c |
| `t3r4_q60_s80512_bwA` | 64eaaebc241e3251 | 22991ba0e10f88e2 | cbfdb613dde33d20 |
| `t3r4_q60_s80513_bwA` | 9eb681214bf67191 | c061c6df86f7bbcf | 91ed114b5e9105b9 |
| `t3r4_q60_s80514_bwA` | 4a963411e29472d9 | 89b605377745ff50 | a39606283c89194d |
| `t3r4_q60_s80515_bwA` | a54076d1b5c4bc4f | 64a4757793b48d1c | f59e04fbdeee54ed |
| `t3r4_q60_s80516_bwA` | 22681cbae7290684 | 8a7b664c3296ec25 | c124e7eed0efa8d9 |
| `t3r4_q60_s80517_bwA` | 64f6d1ed7c6cf728 | 7f5979be2947f7dc | f3a3c241b47ba766 |
| `t3r4_q60_s80518_bwA` | 16b9d207d3b7c479 | 3db8d712c34272e7 | ed0e478bc375a2e7 |
| `t3r4_q60_s80519_bwA` | e162429775339bb1 | 220b5ed79ae394ef | f1438448ab68b652 |
| `t3r4_q60_s80520_bwA` | c269498b7965418f | e703a70460fdc8a2 | 5505faee4b286ead |
| `t3r4_q60_s80521_bwA` | 2ec6afa011d0c0da | 55d730d7266796ba | d29d0bda4b0bb7b6 |
| `t3r4_q60_s80522_bwA` | 18d6231772a26b40 | 914cef049b6562bf | 6f711682e3d60312 |
| `t3r4_q60_s80523_bwA` | f3da8653a1b31dd2 | 1c2364f8f629279c | 8477b8f3d4774f04 |
| `t3r4_q60_s80524_bwA` | 8ef323b7f5ee9183 | 28da411412e3cafa | 01fd515836691f99 |
| `t3r4_q60_s80525_bwA` | 41d6e5bef6e6dabc | d7080242fdcf7252 | 88ae6583c5e5db2c |
| `t3r4_q60_s80526_bwA` | 36f5d00f77f8f635 | 0b8c55aacc1ec1be | 04157b909c6ffb80 |
| `t3r4_q60_s80527_bwA` | b665564b4c71eaa2 | 200a9aec7057e630 | ebcf4b750df2de06 |
| `t3r4_q60_s80528_bwA` | 917798b9feff72dd | b141c069c73ea445 | e93f15b12f3a8b44 |
| `t3r4_q60_s80529_bwA` | cf64dc1110fd8418 | 0e841b4629299626 | 3a5f8f1e18208f3a |
| `t3r4_q60_s80530_bwA` | 98be101bb747f745 | 957839a2ece43239 | aa9ec6a2b77efc89 |
| `t3r4_q60_s80531_bwA` | a1119acafca47c99 | abcbd5665909da65 | e65215060981f584 |
| `t3r4_q60_s80532_bwA` | 00e5e5455aa7848f | 22b9bf1bb5c0768d | 3b83c69317c4d6e8 |
| `t3r4_q60_s80533_bwA` | 8f55a1ef1c62d5a9 | 66a5456aa1395544 | 4c646e707de33348 |
| `t3r4_q60_s80534_bwA` | 5f7594ad74425d20 | 0868dc9d1e1c2f84 | 96df63d72175b888 |
| `t3r4_q60_s80535_bwA` | aae1e5e4b6481626 | da1c4b23499a9d07 | b61fd4d154fe1e05 |
| `t3r4_q60_s80536_bwA` | c2e6d224e2794beb | a5c8f810d458b10c | e89c66e92936443b |
| `t3r4_q60_s80537_bwA` | 16684884cfead573 | 063bae5d262aef12 | db5d73e69196821a |
| `t3r4_q60_s80538_bwA` | 6ed782c7ca7b36f9 | b01677fc933f06be | d1c4989264afc7a2 |
| `t3r4_q60_s80539_bwA` | 291bcf4fc59e0a2e | 301b8df9076169c4 | cafdf816d85be0fd |
| `t3r4_q60_s80540_bwA` | edb1cc4e8c38c39a | f6db82d1bbaeefd6 | 5f48b3362cb8cd69 |

## 5. Config differences against T3-R3's `bwA` configs of the same q

A confirmation config differs from T3-R3's `bwA` development config of the same q in the six run fields only:

| q | T3-R3 `bwA` config (seed 70501) against the T3-R4 config (seed 80501): differing config paths |
|---|---|
| 50 | `pilot`, `record.output_dir`, `record.run`, `record.seed`, `run`, `seed` |
| 60 | `pilot`, `record.output_dir`, `record.run`, `record.seed`, `run`, `seed` |

(The launcher also tests this for all 80 configs against the T3-1.1 reference: `tests/test_t3r4_launch.py`.)

## 6. Seed inventory (P0, `00_housekeeping.md` section 3)

| block | scan | result | hit rows (`seed_hits.py`) |
|---|---|---|---|
| 80501-80540 | scanned 231947 json, 87424 csv files; 437 distinct seed values | block 80501-80540: 0 collisions | 0 |

## 7. C-REPRO (at the final code commit)

T3-R3's `bwA` development runs (q = 50, seed 70508), whose trigger fired, and (q = 60, seed 70501), whose trigger did not, re-run from their own `run_config.json` with `tools/t3/t3r4_repro.py` and compared with their originals in the T3-R3 untracked root (`tools/t3/compare_runs.py bw`, plus `attempt1/` and `retry.json` for 70508):

| run | trigger fired in the original | re-run | original (T3-R3 untracked root) | code commit of the re-run | `compare_runs.py bw` exit code | npz files compared | same set of files | stage-2 and stage-3 end states (bitwise) | gates, rule log, drift test, summary, manifest, train history | `attempt1/` and `retry.json` | all identical | wall time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t3r3_q50_s70508_bwA | yes | results/t3_r4/repro/q50/seed70508/bwA | results/t3_r3/dev/q50/seed70508/bwA | 631c19f1 | 0 | 162 | yes | yes | identical | identical | yes | 575.3 |
| t3r3_q60_s70501_bwA | no | results/t3_r4/repro/q60/seed70501/bwA | results/t3_r3/dev/q60/seed70501/bwA | 631c19f1 | 0 | 162 | yes | yes | identical | - | yes | 434.8 |

## 8. Rehearsal

Before the lock, the post-launch tools were run once on 80 run directories whose files are symlinks to T3-R3's 40 `bwA` development runs (`results/t3_r4/rehearsal/`, mapped to the seeds 80501-80540; not a lock input): the evaluation, the analysis, the blind recomputation and the report tables run end to end on production-size runs, including four fired runs; the budgets, schedules, exports, checks, retry budgets, trigger and C-REPLAY checks pass, and the config / plan / init / clean-tree verdicts fail by construction. The details are in `02_checks.md`.

## 9. The steps that follow

1. The lock: `t3r4_make_lock.py --out protocols/` (byte-identical to the drafts), committed alone.
2. The LOCK record appended to `protocols/LOCK` (protocol T3-1.1, the lock commit, the JSON and MD SHA-256, the date; entry `run/run_ms_stagewise.py` through `tools/t3/t3r4_launch.py`; reason "PI decision at gate G2 of round T3-R3: lock `bwA`"; the rule of change), committed separately; both commits pushed.
3. The wave of 80 runs from the LOCK-record commit (`results/t3_r4/confirmation/`), then `t3r4_post_launch.sh`.
