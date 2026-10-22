# T3-R3, P2: pre-registration of the development wave

Date: 2026-10-22. Branch `t3-r3`. **Generated** from `reports/t3/r3/report_scripts/04_preregistration.template.md` by `build_report.py`; the tables by `prereg_tables.py`. This file is committed and pushed **before any development run** and before the diagnostic runs; the wave is launched from the commit that holds it, on a clean tree. Nothing in it is tuned after the launch.

**Code version.** The final code commit of P2 is `727116a3` (`run/ms_retry.py`, `run/run_ms_stagewise.py`, `tools/t3/t3r3_*.py`); C-LEGACY ran at it (section 8). From that commit to the launch only `reports/`, `results/`, `docs/STATE.md` and `reports/t3/README.md` change. The binding texts are the prompt (`reports/t3/r3/pi_record/28_t3_r3_prompt.md`, D5-D7), the PI's reply at gate G1 (`pi_record/28b_g1_reply.md`) and the accepted proposal (`02_proposal.md` sections 2-3).

## 1. Arms, keys, seeds, names

| arm | role | scheme keys | runs | order in the selection |
|---|---|---|---|---|
| `bw` | control (locked T3-1.0 settings); not eligible | none | 40 (+ none diagnostic) | - |
| `bwA` | candidate A | `stage3_retry = {"delta_gt": 0.005, "max_retries": 1}` | 40 + 1 diagnostic | 1 |
| `bwAB` | candidate A and B | `stage3_retry` as above and `terminal_clamp_likelihood = "censored"` | 40 + 1 diagnostic | 2 |

- **Seeds.** 70501-70520 at q = 50 and q = 60, the same seeds in every arm (40 runs per arm). The diagnostic runs are (q = 60, seed 60518), once for each candidate arm, into `results/t3_r3/diagnostic/q60/seed60518/{arm}/`; they are reported separately and **never counted**. Run names `t3r3_q{q}_s{seed}_{arm}`; run directories `results/t3_r3/dev/q{q}/seed{s}/{arm}/`. The launcher refuses any other seed (tested): 60501-60520 (except the diagnostic seed at q = 60), 80501-80540 (reserved for the later confirmation, never trained), 50500-50520, 40501-40520, 10501-10530, 30501-30520.
- **Control configs.** The locked T3-1.0 settings: `tools/t3/t3_configs.build_bw_config` with `seed`, `run`, `record.run`, `record.seed`, `record.output_dir` and `pilot` = "t3_r3" set; the config with the scheme's own keys removed and the D4 fields reset (`t3c_launch.normalise`) hashes to the lock's `reference_config_sha256` of its q (checked for all 122 runs by the launcher).
- **Candidate configs** differ from the control's config of the same (q, seed) only in the scheme's own keys and the run fields:

| arm | differing config paths (q = 50, seed 70501; the same for every seed, tested) |
|---|---|
| `bwA` against `bw` | `record.run`, `run`, `stage3_retry.delta_gt`, `stage3_retry.max_retries` |
| `bwAB` against `bw` | `record.run`, `run`, `stage3_retry.delta_gt`, `stage3_retry.max_retries`, `terminal_clamp_likelihood` |

- **The 122 planned runs** (120 development + 2 diagnostic), their config SHA-256 and planned init digests (attempt 1; attempt 2 for the candidate arms) are in `reports/t3/r3/dev_plan.csv` (SHA-256 of the file: `c7c43a57f79bca67d439fcd2d4eaa557c4aa1de648e41b0f47284283e2d9af22`); the table lists them.

| run | kind | config SHA-256 (16) | init digest, attempt 1 (16) | init digest, attempt 2 (16) |
|---|---|---|---|---|
| `t3r3_q60_s60518_bwA` | diagnostic | e9eca4440f57cf50 | fb3fde188a7b348d | 0c5c5434c94f10b6 |
| `t3r3_q60_s60518_bwAB` | diagnostic | 199b472d1cdaf537 | fb3fde188a7b348d | 0c5c5434c94f10b6 |
| `t3r3_q50_s70501_bwAB` | dev | 8857b4be9753c1d2 | 570187541fd9e0a1 | 974ded085fa50157 |
| `t3r3_q50_s70502_bwAB` | dev | 8322a9078db86637 | 111280e3872b9dd5 | 3c9f4873641aee2e |
| `t3r3_q50_s70503_bwAB` | dev | f0019e58bf7e62f4 | 577e09e12be8fd74 | 419dec9fc91022b4 |
| `t3r3_q50_s70504_bwAB` | dev | a9bacdc8cf23568a | 9f75b75332d0345d | 5225116969c43b13 |
| `t3r3_q50_s70505_bwAB` | dev | ccbdc041a84ec6e1 | 3206fb1227374981 | 1e6c1f06c992f5b4 |
| `t3r3_q50_s70506_bwAB` | dev | 2a33d588157ce1e7 | a697d6a32cca7150 | 0740d9c690d20138 |
| `t3r3_q50_s70507_bwAB` | dev | 04e4b7418a8d1660 | ba797bb93088f7ae | ef0f16cb81f2464a |
| `t3r3_q50_s70508_bwAB` | dev | a477cb89caa4ff15 | 63497ccea04bf81f | ddeb5d0469c26ee9 |
| `t3r3_q50_s70509_bwAB` | dev | 2bc00ccf69f6ed00 | 14e34ce3a7364a03 | 425757cdadd5b1f9 |
| `t3r3_q50_s70510_bwAB` | dev | ca3fc91726cc0fdc | 89ca657bf293cd81 | d2797a29c98ab7ca |
| `t3r3_q50_s70511_bwAB` | dev | c813291d9fe3c4e3 | f12c3a9ea60111ef | b758a1e507f2a535 |
| `t3r3_q50_s70512_bwAB` | dev | 2e2d66a0880fc9e9 | 24d8215a2e85543d | b0802330ca74bbe7 |
| `t3r3_q50_s70513_bwAB` | dev | 5f543b4eeb95c7f7 | 460673ebef04e5b8 | ee79b7aca72e939f |
| `t3r3_q50_s70514_bwAB` | dev | 5835442177239644 | 2af59b7b67a0b82a | d4b679be92fc4258 |
| `t3r3_q50_s70515_bwAB` | dev | 624293cf835c21c1 | e4c5ae84628ced4f | de959e517d5ecb5d |
| `t3r3_q50_s70516_bwAB` | dev | 81b538f7f82e956f | 8747ef31dcfc8065 | 44634954463d38de |
| `t3r3_q50_s70517_bwAB` | dev | 08c82f5c30d1f728 | c63736453edafb9e | 92dcb4888d0e72c0 |
| `t3r3_q50_s70518_bwAB` | dev | 743004ed54eaa63a | 5cb1592aa562190b | 1db6a069355f7d69 |
| `t3r3_q50_s70519_bwAB` | dev | 5ce413caefca8919 | fcaad2d8c05b3f0e | 14fcf276752ace0e |
| `t3r3_q50_s70520_bwAB` | dev | e00cc0662482a6d7 | 0e82944fcbf2b0e8 | 254a399ddbc20895 |
| `t3r3_q60_s70501_bwAB` | dev | ac942af146223cd0 | 734feb77ac7563bb | b9f21fffdcc6fbc1 |
| `t3r3_q60_s70502_bwAB` | dev | 9afb459653df2e78 | b2abd48c76c522d6 | 5237ae37f1f2a853 |
| `t3r3_q60_s70503_bwAB` | dev | 7ba1c3e1624b054c | e721a60f6e8c13b2 | 5347500b5c8f857d |
| `t3r3_q60_s70504_bwAB` | dev | af5df99c73a03e40 | 632174ab212c8154 | 165456fb673f16b4 |
| `t3r3_q60_s70505_bwAB` | dev | aea81ce7d409b074 | fdc5dc526b01c951 | fef8f33124d8ab10 |
| `t3r3_q60_s70506_bwAB` | dev | cee356009cb4d9d7 | d65d8bcdea43b4c6 | bb750cad8a20b27c |
| `t3r3_q60_s70507_bwAB` | dev | c6d7565397dcd491 | 8f6d33d8f8f9718f | 527548459469c111 |
| `t3r3_q60_s70508_bwAB` | dev | 52fe406a2c606414 | c0ab4caccd009fb9 | 1de71c9e8f81ca0b |
| `t3r3_q60_s70509_bwAB` | dev | ae7f4891bb625f6d | 4b2a8a3f5262d965 | d29b139256277fef |
| `t3r3_q60_s70510_bwAB` | dev | 5dce78653e0c4c20 | 5bbd043c801f9be8 | 7f22f4fdfaa5ff41 |
| `t3r3_q60_s70511_bwAB` | dev | 8e8ec6a7ac94c658 | 099c4555ece33e31 | c5cb18882f284316 |
| `t3r3_q60_s70512_bwAB` | dev | d99f2300f9c24b4a | 4303daea94d5cfe5 | deb27895aa165e38 |
| `t3r3_q60_s70513_bwAB` | dev | 50c31a22ac472c32 | 27d5faa51ccbfc09 | 2afef6c7408d4d78 |
| `t3r3_q60_s70514_bwAB` | dev | e1127d1295cf35e9 | b69d981cfb45ae50 | 9c1cc2d2764243ea |
| `t3r3_q60_s70515_bwAB` | dev | a57f8540f46b06a8 | cfe559c930e5d6ff | 6c9d39847f7ce5b7 |
| `t3r3_q60_s70516_bwAB` | dev | 64d6df31b8b44c48 | 6dd073113007defd | a53bb7848fec63fc |
| `t3r3_q60_s70517_bwAB` | dev | 14c75d8f9e066d0c | 6a5cc1d89b53b9b9 | 1090e495447be355 |
| `t3r3_q60_s70518_bwAB` | dev | 66011bb57bd9e228 | 4b829caf2f8be1f0 | b8e61c223554489c |
| `t3r3_q60_s70519_bwAB` | dev | 75688e29beda6e59 | 521b7fddc948323b | 2cb752129b1b36b7 |
| `t3r3_q60_s70520_bwAB` | dev | f776b45756db666e | 6c3f82fb4d23dd6a | 25b8edd94d56a926 |
| `t3r3_q50_s70501_bwA` | dev | bfa54622634da19c | 570187541fd9e0a1 | 974ded085fa50157 |
| `t3r3_q50_s70502_bwA` | dev | cecaffa3933a1c87 | 111280e3872b9dd5 | 3c9f4873641aee2e |
| `t3r3_q50_s70503_bwA` | dev | 5465fa3edf3f9040 | 577e09e12be8fd74 | 419dec9fc91022b4 |
| `t3r3_q50_s70504_bwA` | dev | c8834a9878d9fc02 | 9f75b75332d0345d | 5225116969c43b13 |
| `t3r3_q50_s70505_bwA` | dev | 0a9c0e67c1a3ba8a | 3206fb1227374981 | 1e6c1f06c992f5b4 |
| `t3r3_q50_s70506_bwA` | dev | 9bfe404fedf888f0 | a697d6a32cca7150 | 0740d9c690d20138 |
| `t3r3_q50_s70507_bwA` | dev | 4f6720dbd5aaabea | ba797bb93088f7ae | ef0f16cb81f2464a |
| `t3r3_q50_s70508_bwA` | dev | 323d727e9f3bccf3 | 63497ccea04bf81f | ddeb5d0469c26ee9 |
| `t3r3_q50_s70509_bwA` | dev | 2aa95b3be8b5598b | 14e34ce3a7364a03 | 425757cdadd5b1f9 |
| `t3r3_q50_s70510_bwA` | dev | 0332be5cea7abd6a | 89ca657bf293cd81 | d2797a29c98ab7ca |
| `t3r3_q50_s70511_bwA` | dev | 7f32fccab2caee22 | f12c3a9ea60111ef | b758a1e507f2a535 |
| `t3r3_q50_s70512_bwA` | dev | a31853d035b1229b | 24d8215a2e85543d | b0802330ca74bbe7 |
| `t3r3_q50_s70513_bwA` | dev | 4cc8eb78486c9ce7 | 460673ebef04e5b8 | ee79b7aca72e939f |
| `t3r3_q50_s70514_bwA` | dev | 3c2037bb56a65e81 | 2af59b7b67a0b82a | d4b679be92fc4258 |
| `t3r3_q50_s70515_bwA` | dev | 931d662feaba5045 | e4c5ae84628ced4f | de959e517d5ecb5d |
| `t3r3_q50_s70516_bwA` | dev | 900adec0526d04a5 | 8747ef31dcfc8065 | 44634954463d38de |
| `t3r3_q50_s70517_bwA` | dev | 958d8d3d1faeef9e | c63736453edafb9e | 92dcb4888d0e72c0 |
| `t3r3_q50_s70518_bwA` | dev | 54afa3cd46a5cf9e | 5cb1592aa562190b | 1db6a069355f7d69 |
| `t3r3_q50_s70519_bwA` | dev | c2002c632cce9dec | fcaad2d8c05b3f0e | 14fcf276752ace0e |
| `t3r3_q50_s70520_bwA` | dev | 2327f6337ddeb3db | 0e82944fcbf2b0e8 | 254a399ddbc20895 |
| `t3r3_q60_s70501_bwA` | dev | 8788c9c3a9e60733 | 734feb77ac7563bb | b9f21fffdcc6fbc1 |
| `t3r3_q60_s70502_bwA` | dev | a053d743f7d99377 | b2abd48c76c522d6 | 5237ae37f1f2a853 |
| `t3r3_q60_s70503_bwA` | dev | de5c03aa54fceca6 | e721a60f6e8c13b2 | 5347500b5c8f857d |
| `t3r3_q60_s70504_bwA` | dev | f6f8886036cf89ca | 632174ab212c8154 | 165456fb673f16b4 |
| `t3r3_q60_s70505_bwA` | dev | 13b1f56656bf0eea | fdc5dc526b01c951 | fef8f33124d8ab10 |
| `t3r3_q60_s70506_bwA` | dev | 590760e3595e2453 | d65d8bcdea43b4c6 | bb750cad8a20b27c |
| `t3r3_q60_s70507_bwA` | dev | e8ca179b28d27326 | 8f6d33d8f8f9718f | 527548459469c111 |
| `t3r3_q60_s70508_bwA` | dev | 00203e171e81688a | c0ab4caccd009fb9 | 1de71c9e8f81ca0b |
| `t3r3_q60_s70509_bwA` | dev | 9be94d80595a52c5 | 4b2a8a3f5262d965 | d29b139256277fef |
| `t3r3_q60_s70510_bwA` | dev | e76ac18e4f46c1ad | 5bbd043c801f9be8 | 7f22f4fdfaa5ff41 |
| `t3r3_q60_s70511_bwA` | dev | 5abb959537d5556d | 099c4555ece33e31 | c5cb18882f284316 |
| `t3r3_q60_s70512_bwA` | dev | a6d963ee96418fd0 | 4303daea94d5cfe5 | deb27895aa165e38 |
| `t3r3_q60_s70513_bwA` | dev | d35b3b8df648e235 | 27d5faa51ccbfc09 | 2afef6c7408d4d78 |
| `t3r3_q60_s70514_bwA` | dev | 9f8bcc0f0400106e | b69d981cfb45ae50 | 9c1cc2d2764243ea |
| `t3r3_q60_s70515_bwA` | dev | a623123574c1d9c9 | cfe559c930e5d6ff | 6c9d39847f7ce5b7 |
| `t3r3_q60_s70516_bwA` | dev | 785ff226e7fa9879 | 6dd073113007defd | a53bb7848fec63fc |
| `t3r3_q60_s70517_bwA` | dev | 3e87c0149427ab6f | 6a5cc1d89b53b9b9 | 1090e495447be355 |
| `t3r3_q60_s70518_bwA` | dev | d5ee0c30d73efe35 | 4b829caf2f8be1f0 | b8e61c223554489c |
| `t3r3_q60_s70519_bwA` | dev | 85a388b9c3f3575e | 521b7fddc948323b | 2cb752129b1b36b7 |
| `t3r3_q60_s70520_bwA` | dev | 4e8468ee3dd8690c | 6c3f82fb4d23dd6a | 25b8edd94d56a926 |
| `t3r3_q50_s70501_bw` | dev | ea2e8f32d529f6f2 | 570187541fd9e0a1 | - |
| `t3r3_q50_s70502_bw` | dev | 10963c165621221f | 111280e3872b9dd5 | - |
| `t3r3_q50_s70503_bw` | dev | f28a0d331ecd19ad | 577e09e12be8fd74 | - |
| `t3r3_q50_s70504_bw` | dev | 420be282d2996c93 | 9f75b75332d0345d | - |
| `t3r3_q50_s70505_bw` | dev | 92e3751863ca8a56 | 3206fb1227374981 | - |
| `t3r3_q50_s70506_bw` | dev | f7e97904054fb4b9 | a697d6a32cca7150 | - |
| `t3r3_q50_s70507_bw` | dev | c7a112223ef6020f | ba797bb93088f7ae | - |
| `t3r3_q50_s70508_bw` | dev | c095c69493bf8f81 | 63497ccea04bf81f | - |
| `t3r3_q50_s70509_bw` | dev | 220ca3c99f412160 | 14e34ce3a7364a03 | - |
| `t3r3_q50_s70510_bw` | dev | 1b62502627340426 | 89ca657bf293cd81 | - |
| `t3r3_q50_s70511_bw` | dev | d6c669d3d752e733 | f12c3a9ea60111ef | - |
| `t3r3_q50_s70512_bw` | dev | c83f11b7f92573d2 | 24d8215a2e85543d | - |
| `t3r3_q50_s70513_bw` | dev | c2e6fafda7ca68f7 | 460673ebef04e5b8 | - |
| `t3r3_q50_s70514_bw` | dev | 072c449e26288c2f | 2af59b7b67a0b82a | - |
| `t3r3_q50_s70515_bw` | dev | 056cee0b42911799 | e4c5ae84628ced4f | - |
| `t3r3_q50_s70516_bw` | dev | ed8074fcd52426d5 | 8747ef31dcfc8065 | - |
| `t3r3_q50_s70517_bw` | dev | 7cbadfeb66e283c8 | c63736453edafb9e | - |
| `t3r3_q50_s70518_bw` | dev | f9097f192aa075dc | 5cb1592aa562190b | - |
| `t3r3_q50_s70519_bw` | dev | e6228ca37cc0beb9 | fcaad2d8c05b3f0e | - |
| `t3r3_q50_s70520_bw` | dev | 356c3545156a6d23 | 0e82944fcbf2b0e8 | - |
| `t3r3_q60_s70501_bw` | dev | 0b7723bae3f89974 | 734feb77ac7563bb | - |
| `t3r3_q60_s70502_bw` | dev | af0ca6898ee78d68 | b2abd48c76c522d6 | - |
| `t3r3_q60_s70503_bw` | dev | 55c2fd481068906f | e721a60f6e8c13b2 | - |
| `t3r3_q60_s70504_bw` | dev | db5fc0b1dd6ff9bd | 632174ab212c8154 | - |
| `t3r3_q60_s70505_bw` | dev | e0e71fe6b233edbf | fdc5dc526b01c951 | - |
| `t3r3_q60_s70506_bw` | dev | 1f99430c74299aa5 | d65d8bcdea43b4c6 | - |
| `t3r3_q60_s70507_bw` | dev | 0ef3599ab899df7a | 8f6d33d8f8f9718f | - |
| `t3r3_q60_s70508_bw` | dev | 75d22be2e61dae5d | c0ab4caccd009fb9 | - |
| `t3r3_q60_s70509_bw` | dev | b6771b00b78b450d | 4b2a8a3f5262d965 | - |
| `t3r3_q60_s70510_bw` | dev | 910634fdc37cb11a | 5bbd043c801f9be8 | - |
| `t3r3_q60_s70511_bw` | dev | d0e2230f2ccb752c | 099c4555ece33e31 | - |
| `t3r3_q60_s70512_bw` | dev | c01a1abdf740f9c3 | 4303daea94d5cfe5 | - |
| `t3r3_q60_s70513_bw` | dev | 7ba0ad076abe5a69 | 27d5faa51ccbfc09 | - |
| `t3r3_q60_s70514_bw` | dev | 1a623ab9725ba675 | b69d981cfb45ae50 | - |
| `t3r3_q60_s70515_bw` | dev | df2c4597a732b566 | cfe559c930e5d6ff | - |
| `t3r3_q60_s70516_bw` | dev | ac318fcec0874e8b | 6dd073113007defd | - |
| `t3r3_q60_s70517_bw` | dev | 46e03d48dd76acc3 | 6a5cc1d89b53b9b9 | - |
| `t3r3_q60_s70518_bw` | dev | 0e2709d9e212accc | 4b829caf2f8be1f0 | - |
| `t3r3_q60_s70519_bw` | dev | d6829967c900dee1 | 521b7fddc948323b | - |
| `t3r3_q60_s70520_bw` | dev | 15eeca9e0281218e | 6c3f82fb4d23dd6a | - |

## 2. Order before the launch, and the launch

1. The independent review of the P2 code, its fixes committed (`03_checks.md`).
2. C-LEGACY at the final code commit (section 8).
3. This pre-registration, committed and pushed.
4. The launch: one wave of 122 runs, at most 40 single-threaded workers, in tmux, `python tools/t3/t3r3_launch.py --workers 40 --code-commit <sha>` from the commit that holds this file, on a clean tree (the launcher refuses a dirty tree outside `results/`).

## 3. The schemes

**Candidate A (`bwA`)**, exactly as `02_proposal.md` section 2.1: `stage3_retry = {"delta_gt": 0.005, "max_retries": 1}`; the trigger is read at the local-1600 development check of the stage-3 phase and fires iff `Delta` > 0.005, or `Delta`, `s` or `C` is not finite, or `s` <= 0; one in-place re-initialisation of the actor and the critic and of both Adam states, from `SeedSequence([seed, q, ns_init, 2])` and the five streams `SeedSequence([seed, q, ns_name, 2])`; attempt 2 is frozen whatever its own check says; the artefacts of attempt 1 are moved to `attempt1/` (never overwritten, never deleted) and `retry.json` is written; a run whose trigger does not fire writes exactly the files, columns and rows of `bw`.

**Candidate B (`bwAB` = `bwA` + `terminal_clamp_likelihood = "censored"`)**, exactly as section 2.2: in the stage-3 phase only, for both attempts; it reuses `utils/beta_tail.py`, `collect_batch_v2(..., clamp_likelihood="censored")` and `CurriculumPPOv2.update(..., clamp_side=...)`; the wiring in `run/run_ms_stagewise.py` is all that was needed (no further change). Test: at T = 3, in the stage-3 phase, the stored log-prob of every flagged row equals the censored log-mass of `utils/beta_tail.py` and every unflagged row the density path (`tests/test_t3r3_retry.py`).

**Paper wording for A (verbatim, G1 reply 2.5):** "a verifier-gated restart of the terminal stage: when the development-tier one-step deviation gain at the end of the stage-3 phase exceeds the gate's limit 0.005, stage 3 is re-initialised once and retrained, and the second attempt is kept; its cost is part of the method's cost."

## 4. Budgets and cost (fixed in advance)

| case | updates | policy rows | minibatch steps | development checks | exports (frozen path) | relative to bw |
|---|---|---|---|---|---|---|
| bw, and bwA / bwAB when the trigger does not fire | 3800 | 1945600 | 76000 | 152 | 152 | 1 |
| bwA / bwAB when the trigger fires (stage 3 trained twice) | 5400 | 2764800 | 108000 | 216 | 152 (+ 64 in attempt1/) | 1.421 |
| bwA / bwAB expected, fire rate 7 / 80 = 0.0875 | 3940 | 2.02e+06 | 7.88e+04 | 157.6 | 152 | 1.037 |
| bwA / bwAB expected, lower limit of the 95 % interval = 0.0359 | 3857 | 1.98e+06 | 7.715e+04 | 154.3 | 152 | 1.015 |
| bwA / bwAB expected, upper limit = 0.1720 | 4075 | 2.09e+06 | 8.15e+04 | 163 | 152 | 1.072 |

Wall time of the 80 recorded `bw` runs (`status.json` `total_wall_sec`): median 600 s, range 503-640 s; the stage-3 phase (`ms_run_summary.json`) median 235 s, i.e. 39% of a run. One retraining therefore costs about 235 s; a wave of 40 `bw` runs costs about 6.7 CPU-hours, of 40 runs of an arm with the expected extra stage-3 work about 6.9. The censored update measured on a synthetic batch of 512 rows with 466 flagged rows (`agent.update` with `clamp_side`, mean of 3 calls, one thread) took 1.17 times the plain update (0.1355 against 0.1161 s); a regime run with about 57 % flagged rows costs less.

The measured wall time per arm and its ratio to `bw` is reported in addition to updates and policy rows (G1 reply 2.4).

## 5. Post-launch checks (`results/t3_r3/dev/launch_checks.json`)

Per run, by `tools/t3/t3r3_checks.py` (a deterministic failure is a stop-and-report):

- **Basics:** exit code 0; the files of a `bw` run, plus `retry.json` and `attempt1/` for a run that fired, neither for a run that did not; the manifest at the launch commit with a clean tree; the global-RNG assertion `ok`; the config equal to the plan (SHA-256 with the output directory made relative) and the lock/arm check.
- **Trigger:** recomputed from the check file it read; agrees with `retry.json`; `bw` has neither.
- **Budgets:** the frozen path has the `bw` budget (updates 1600 / 1600 / 600, policy rows 819,200 / 819,200 / 307,200, 76,000 minibatch steps); a run that fired also has 5,400 updates, 2,764,800 policy rows, 108,000 minibatch steps and 216 development checks in total and, in `attempt1/`, 64 exports, 1,600 rows and 64 checks.
- **Schedules:** the learning rate at every update (attempt 1 as well), the snapshot refreshes, 152 exports (`u00025` ... `u03800`), 64 / 64 / 24 checks.
- **C-INIT:** the recorded init digests equal the planned ones (attempt 1; attempt 2 for a run that fired); the attempt-1 digest is equal across the arms of one (q, seed); the attempt-2 digests of `bwA` and `bwAB` are equal.
- **C-REPLAY:** the evaluator reproduces the freeze arrays bit for bit.
- **C-PAIR:** `bwA` against `bw` of the same seed (`tools/t3/t3r3_pair.py`): where the trigger did not fire the run is bit-identical to `bw` with the scheme's records only in new files; where it fired, attempt 1 equals the `bw` run's stage 3 up to the check (the 64 exports, the stage-3 rows of `ms_updates.csv` and `ms_checks_stage3.csv` without wall-clock columns, the bin maps, the training state). The diagnostic runs (both arms) are compared with T3-R2's original run of seed 60518 in the same way (binding). `bwAB` against `bw` is computed and reported, deciding nothing.

## 6. Evaluation, the target and the selection

- **Evaluation.** `tools/t3/t3_eval.evaluate_run` on the final and development tiers through `tools/t3/t3r3_eval.py`; the evaluated policy is the one the scheme freezes (attempt 2 for a run that fired). Gates and run pass as in T3-1.0: exit 0, global-RNG assertion ok, a valid evaluation on both tiers, G-A (`eta_3` <= 0.005, `rmse_pos_T_over_eT0` <= 0.05, `tail_mean_T_over_eT0` <= 0.02), G-F (`Gmax_full_over_dw` <= 0.01), G-N (development minus final <= 0.001 for `eta_3` and `Gmax_full`), every `<=` inclusive at full float64 precision. S3 of a retrained run is computed on the frozen attempt.
- **Failure counting (D6).** Infrastructure failures (a negative return code, an out-of-memory kill, ENOSPC) are recorded in `infra_events.json` and the run is re-run once; a second one stops the arm without a verdict. Any other nonzero exit and any global-RNG violation is a failed run. An evaluation error or an invalid evaluation is a failed run unless a re-run of the locked evaluator succeeds. A re-initialisation inside a run is part of the run, never a re-run. Every planned run counts.
- **The target (the PI's).** A candidate arm reaches it iff at least 38 of its 40 runs pass and at least 18 of 20 at each q. The pass probabilities for comparison are in section 10.
- **Selection.** Only the candidate arms are eligible. In the order `bwA`, then `bwAB`, the first arm that reaches the target is the scheme proposed for the lock; if none reaches it, no scheme is proposed. Nothing is locked in this round. **If both arms reach the target, `bwA` is proposed. The PI may lock `bwAB` instead only with a stated reason taken from the items of §2.2 of the G1 reply. Nothing else may change the choice.**

## 7. Reported, deciding nothing

- Per arm and q: the pass counts with exact intervals; the worst `Gmax_full` and its run; the medians of `eta_3`, `rmse_pos_T_over_eT0`, `tail_mean_T_over_eT0`, `peak_T_signed`, `Gmax_full` and EXP_root.
- Against `bw`, per seed: the paired differences of `Gmax_full` and `eta_3` with paired bootstrap intervals (10,000 resamples, seed 20261021), and the pass / fail cross-table.
- S3 of every run, the stage-2 stretches, every trigger that fired with the run and the outcome, the compute cost per arm (updates, policy rows, minibatch steps, development checks, **measured wall time**, verifier seconds) and its ratio to `bw`.
- **First-attempt outcome per arm (G1 reply 2.1).** The pass count when every fired run is counted as a failure of its first attempt. For `bwA` this must equal the pass count of `bw`, as C-PAIR implies; **a mismatch is a stop.** Reading of the sentence: a `bw` run is counted as "fired" when the trigger rule, applied to its own stage-3 check at local update 1600, fires; the invariant is then `bwA` = `bw` seed by seed. The raw pass count of `bw`, and the number of `bw` runs that fire and nevertheless pass the gates (which the convention counts as first-attempt failures in both arms), are reported beside it.
- **The causal reading of B (G1 reply 2.2).** `bwAB` against `bw` and against `bwA`, paired by seed, on first attempts (for a fired run, `attempt1/`): S3 per run and the number of runs with S3 > 0.2, with an exact McNemar test on the paired regime indicator; at the first attempt's export `u01600` the share of out-of-band nodes at the actor's mean clamp and the minimum alpha on the D_3 grid; the development `Delta` at local 1600; for the seeds where the trigger fired in neither arm, the paired differences of `eta_3` and of `peak_T_signed` with paired bootstrap intervals (10,000 resamples, seed 20261021); the stage-2 stretches of each arm; the fire rate of each arm.
- **The reading that the proposal gives, stated before any run:** if S3, the stretches and the in-band bias fall in `bwAB`, the clamp regime was more than a marker; if they do not, it is a marker. For the report this is made operational with three indicators on the tables above, none of which decides anything: (i) the number of first-attempt runs with S3 > 0.2 is smaller in `bwAB` than in `bw` and in `bwA`, with the exact McNemar p < 0.05 for the pooled pairs; (ii) the number of runs with a stage-2 stretch is smaller in `bwAB` than in `bw` and in `bwA`; (iii) on the seeds where the trigger fired in neither arm, the paired bootstrap interval of the difference of `eta_3` (`bwAB` - `bw`, pooled over both q) lies entirely below 0. All three: "more than a marker". None: "a marker". Anything else: "mixed", reported as such with no causal claim.
- **The diagnostic runs (G1 reply 2.3).** Whether attempt 1 equals T3-R2's original stage 3 (the exports, the stage-3 rows and the checks); whether A fired; for attempt 2: the formation measure F(u) at the 64 exports, the formation export (threshold 26.78, half the median final F of the passing q = 60 runs of the 80 recorded runs, `results/t3_r3/diag/dg2_runs.csv`), the gap-sensitivity and the final gates.
- **Blind recomputation.** `tools/t3/t3r3_blind.py` (it imports no analysis tool) recomputes the counts, the intervals, the target outcome per arm and the selection from the evaluation table; it must agree with `headline.json` of the analysis.

## 8. C-LEGACY (at the final code commit)

| run | re-run | original | code commit of the re-run | npz files compared | all identical | wall time (s) |
|---|---|---|---|---|---|---|
| t3r2_q50_s60502_bw | /home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-diagnosis-25cb13/results/t3_r3/legacy/q50/seed60502/bw | /home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-audit-6f3a38/results/t3_r2/confirmation/q50/seed60502/bw | 727116a3 | 162 | yes | 423.5 |
| t3r2_q60_s60518_bw | /home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-diagnosis-25cb13/results/t3_r3/legacy/q60/seed60518/bw | /home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-audit-6f3a38/results/t3_r2/confirmation/q60/seed60518/bw | 727116a3 | 162 | yes | 436 |

## 9. The later confirmation (D6; executed only in a later round, after the PI's decision at G2)

- **What is confirmed:** the one scheme proposed under section 6, locked in that round as protocol T3-1.1.
- **Seeds:** 80501-80540, the same 40 seeds at q = 50 and at q = 60, for 80 runs (inventory below: 0 collisions, 0 hits).
- **The rule:** the confirmation passes iff at most 6 of the 80 runs fail and at most 4 of the 40 runs fail at each q.
- **Failure counting,** as in T3-R2's D4: infrastructure failures (negative return code, out-of-memory kill, ENOSPC) are recorded in an `infra_events.json` and the run is re-run once, a second one stops the confirmation without a verdict; any other nonzero exit and any global-RNG violation is a failed run; an evaluation error or an invalid evaluation is a failed run unless a re-run of the locked evaluator succeeds; a re-initialisation or retraining inside a run is part of the run, never a re-run.

## 10. Operating characteristics (computed by `proposal_tables.py`)

| per-run pass probability p (both q) | P(development target: >= 38 of 40 and >= 18 of 20 per q) |
|---|---|
| 0.99 | 0.9925 |
| 0.97 | 0.8822 |
| 0.95 | 0.6767 |
| 0.9 | 0.2228 |
| 0.875 | 0.1084 |

The later confirmation (D6):

| per-run failure probability f (both q) | P(confirmation: <= 6 of 80 and <= 4 of 40 per q) |
|---|---|
| 0.01 | 0.9999 |
| 0.03 | 0.9818 |
| 0.05 | 0.8648 |
| 0.1 | 0.2708 |
| 0.15 | 0.02989 |

## 11. Predictions (D7, the PI's, verbatim; expectations, not tests)

- **DP1.** 60518 is a formation failure, unrelated to the clamp.
  - At q = 60, runs pass through a level of development `Delta` of about 0.12-0.15 near local update 100. 60518 never left it.
  - Its stage-3 network is saturated: its tanh units are saturated over D_3, its gap-sensitivity is about 0, and its in-band surrogate gradient stays small, compared with runs that formed.
- **DP2.** The clamp regime is a mechanism, not only a correlate.
  - In the clamp-regime runs, the out-of-band samples and the draws below the clamp have a projection share of the surrogate gradient above their sample share.
  - The tie-point effort ê_3(0) falls, or stops rising, after the regime's onset.
- **DP3.** The following are one class, (K):
  - the q = 50 tie-point deficits (60501, 60506, 60513, 50504);
  - 60515's excess at d = -36;
  - 60516's shape failure;
  - the two T3-R1 stage-2 stretch failures. 50517's stretch was present at the phase entry, while 50514's formed in updates 2475-2750, but both runs are in the regime.

  T3-R1's 50517 at q = 50 is (V).
- **DP4.** No prediction of the development outcome.

(P1 found DP1a-b supported, DP1c-d partly or not, DP2 not supported, DP3 supported as a classification: `01_diagnosis.md` section 10.) **The predictions of this proposal that the wave can contradict:** A fires on about 9 % of the runs (7 of 80 in the recorded runs); `bwA` and `bw` are bit-identical wherever A does not fire; the diagnostic run of seed 60518 fires A in both arms; B is not expected to change the fire rate if the clamp regime is only a marker.

## 12. Seed inventories (P0, `00_housekeeping.md` section 3)

| block | role | scan | result | hit rows (`seed_hits.py`) |
|---|---|---|---|---|
| 70501-70520 | development block | scanned 213844 json, 79067 csv files; 416 distinct seed values | block 70501-70520: 0 collisions | 0 |
| 80501-80540 | reserved for the later confirmation | scanned 213844 json, 79068 csv files; 416 distinct seed values | block 80501-80540: 0 collisions | 0 |

## 13. Known full-suite failures (G1 reply 3)

`tests/test_registry_canonicalization.py::test_registry_canonicalization` (the known failure) and `tests/test_ms_r4_launcher.py::test_the_existing_launcher_is_unchanged`, which has failed since the T3-1.0 lock changed `protocols/`. Both are recorded as known.
