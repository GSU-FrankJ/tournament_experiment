Reply on the dirty-tree flag (T3-R3, P2)
Option 1, with the verification below. The 122 runs stay the counted records exactly as they are. No run is replaced or re-drawn. Save this reply verbatim as `reports/t3/r3/pi_record/28c_dirty_tree_reply.md`.
1. Verification before P3
Write everything under `results/t3_r3/dev/verify_clean_tree/`.

1. The boundary. From the 122 manifests and `status.json` start times, list which runs record a clean tree and which a dirty one. Set this against the creation time of `reports/t3/r3/report_scripts/results_tables.py` (02:21:15 UTC). Confirm that, in every dirty manifest, the only entry of `git status --porcelain` is that file.
2. Pairs across the boundary. From the C-PAIR results of `launch_checks_absroot.json`, count the `bwA`-`bw` pairs that are bit-identical while one run recorded a clean tree and the other a dirty one. Report the count.
3. Clean-tree re-runs, for verification only, never counted.
   * Move `results_tables.py` out of the worktree temporarily. Do not delete it.
   * Confirm that HEAD is `b565f0e8` and that `git status --porcelain` shows nothing outside `results/`.
   * Re-run, from their own `run_config.json`, into `verify_clean_tree/q{q}/seed{s}/{arm}/`:
      * every dirty-flagged run whose trigger fired, in `bwA`, in `bwAB` and among the diagnostic runs;
      * at each q, the lowest-numbered dirty-flagged `bwAB` run whose trigger did not fire.
   * Compare each re-run with its original bit for bit: `tools/t3/compare_runs.py bw`, plus `attempt1/` and `retry.json` where present.
   * Then restore the script.
   * If all are identical, record the flag as a deviation (§2) and continue.
   * If any differs, stop and report. Do not start P3.

2. Deviations to record in `05_development.md` and `03_checks.md`

* D-1. 82 manifests record a dirty tree, because a report script was written into the worktree 71 s after the launch. Give the evidence of §1.
* D-2. The first post-launch check used a relative `--root` and reported `config_equals_plan` as 0 of 122. The absolute-root check (`launch_checks_absroot.json`) is the record, and the first file is kept. This is the same defect as T3-R2's F10, now in `t3r3_checks.py`.
* D-3. `post_launch.sh` ran the evaluation, the analysis and the blind recomputation after the checks had exited 1.
* The `pair_ok` column. For `bwAB` it is not a check: only `bwA` and the diagnostic runs are paired with `bw`. Label the column that way in the tables, so it is not read as 40 failures.

3. From now on

* No writes during a wave. While the runs of a wave are starting or running, write nothing into the worktree outside `results/`.
* Stop on failure. A script that chains checks, evaluation and analysis must stop at the first nonzero exit.
* Absolute roots only. A tool that takes `--root` must refuse a relative path.

4. Then
The selection stands as pre-registered, and `bwA` is proposed. Continue with P3 as `28_t3_r3_prompt.md` and the G1 reply `28b_g1_reply.md` specify: reports, fact-check and push. Then STOP at G2.
