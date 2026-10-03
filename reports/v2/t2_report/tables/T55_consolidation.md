# T55: Consolidation

- priority: supp; status: generated; tier: n/a
- sources: `results/v2_T2_locked/consolidation/results_root_checksums.csv`, `results/v2_T2_locked/consolidation/dirty_rerun_compare.json`, `results/v2_T2_locked/consolidation/seed_inventory.csv`, `results/v2_pilots/pilot4/analysis/run_records.csv`, `results/v2_pilots/pilot4_B/q60/seed10507/B2_mean_constant/manifest.json`, `results/v2_pilots/pilot4_B_rerun_clean/q60/seed10507/B2_mean_constant/manifest.json`, `protocols/v2_T2_locked_v1_1.json`, `tools/v2/compare_results_roots.py`, `tools/v2/seed_inventory.py`, `reports/v2/protocol_lock_and_rehearsal.md`
- built by: `tools/v2/report/sec_stage1.py:build_t55`; base commit `cb0b541`
- transformation: Counts computed from results_root_checksums.csv (10,034 rows) and seed_inventory.csv (263 rows); every field of dirty_rerun_compare.json copied; the canonical-only file count and the scan scope of the seed inventory are not in any data file (source: report text).

Consolidation before the lock: one development results root (checksums), the clean re-run that resolves the dirty-flag runs, and the seed inventory behind the fresh seed block.

| block | quantity | value | unit | source |
|---|---|---|---|---|
| 1 results-root checksums (original results/v2_pilots vs the canonical copy) | files in the original copy | 10034 | count | results/v2_T2_locked/consolidation/results_root_checksums.csv |
| 1 results-root checksums (original results/v2_pilots vs the canonical copy) | present in the canonical copy | 10034 | count | results/v2_T2_locked/consolidation/results_root_checksums.csv |
| 1 results-root checksums (original results/v2_pilots vs the canonical copy) | byte-identical (SHA-256) | 10034 | count | results/v2_T2_locked/consolidation/results_root_checksums.csv |
| 1 results-root checksums (original results/v2_pilots vs the canonical copy) | missing in the canonical copy | 0 | count | results/v2_T2_locked/consolidation/results_root_checksums.csv |
| 1 results-root checksums (original results/v2_pilots vs the canonical copy) | present but differing | 0 | count | results/v2_T2_locked/consolidation/results_root_checksums.csv |
| 1 results-root checksums (original results/v2_pilots vs the canonical copy) | Pilot 4 parent files (phaseA_ext state_u01200.pt / state_u01600.pt) | 40 | count | results/v2_T2_locked/consolidation/results_root_checksums.csv |
| 1 results-root checksums (original results/v2_pilots vs the canonical copy) | Pilot 4 parent files byte-identical | 40 | count | results/v2_T2_locked/consolidation/results_root_checksums.csv |
| 1 results-root checksums (original results/v2_pilots vs the canonical copy) | files only in the canonical copy at comparison time | 4,027 (source: report text; the CSV lists only the original copy's files, the tool printed this count) | count | reports/v2/protocol_lock_and_rehearsal.md section 1.2 |
| 1 results-root checksums (original results/v2_pilots vs the canonical copy) | tool | tools/v2/compare_results_roots.py (SHA-256 of every file of the original root vs the same path under the canonical root) | text | tools/v2/compare_results_roots.py |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | runs launched with dirty = true (Pilot 4 2b) | 8: q60 s10507 constant; q60 s10507 decay; q60 s10508 constant; q60 s10508 decay; q60 s10509 constant; q60 s10509 decay; q60 s10510 constant; q60 s10510 decay | count | results/v2_pilots/pilot4/analysis/run_records.csv |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | original run: commit, dirty | c92ee74, True | text | results/v2_pilots/pilot4_B/q60/seed10507/B2_mean_constant/manifest.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | clean re-run: commit, dirty | 5d50a9d, False | text | results/v2_pilots/pilot4_B_rerun_clean/q60/seed10507/B2_mean_constant/manifest.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | dir_a | results/v2_pilots/pilot4_B/q60/seed10507/B2_mean_constant | text | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | dir_b | results/v2_pilots/pilot4_B_rerun_clean/q60/seed10507/B2_mean_constant | text | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | state_file | state_end_B.pt | text | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | history_identical | True | flag | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | stability_identical | True | flag | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | verifier_calls_identical | True | flag | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | n_updates | 600 | count | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | n_weight_exports | 24 | count | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | weight_exports_identical | True | flag | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | final_weights_identical | True | flag | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | rng_positions_identical | True | flag | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | end_state_actor_identical | True | flag | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | end_state_critic_identical | True | flag | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | end_state_opponent_identical | True | flag | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | end_state_frozen_identical | True | flag | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | end_state_opt_actor_identical | True | flag | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | end_state_opt_critic_identical | True | flag | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | end_state_rng_minibatch_identical | True | flag | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | end_state_rng_streams_identical | True | flag | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | end_state_torch_generator_identical | True | flag | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant) | ALL_IDENTICAL | True | flag | results/v2_T2_locked/consolidation/dirty_rerun_compare.json |
| 3 seed inventory (proposed fresh block 20501-20520) | seed block | 20501-20520 | seed | protocols/v2_T2_locked_v1_1.json confirmation.seed_block |
| 3 seed inventory (proposed fresh block 20501-20520) | distinct recorded seed values | 263 | count | results/v2_T2_locked/consolidation/seed_inventory.csv |
| 3 seed inventory (proposed fresh block 20501-20520) | recorded seeds inside the block (collisions) | 0 | count | results/v2_T2_locked/consolidation/seed_inventory.csv |
| 3 seed inventory (proposed fresh block 20501-20520) | recorded seeds within +-1000 of the block | 0 | count | results/v2_T2_locked/consolidation/seed_inventory.csv |
| 3 seed inventory (proposed fresh block 20501-20520) | nearest recorded seed below the block | 11120 | seed | results/v2_T2_locked/consolidation/seed_inventory.csv |
| 3 seed inventory (proposed fresh block 20501-20520) | nearest recorded seed above the block | 735000 | seed | results/v2_T2_locked/consolidation/seed_inventory.csv |
| 3 seed inventory (proposed fresh block 20501-20520) | smallest distance from the block | 9381 | seed values | results/v2_T2_locked/consolidation/seed_inventory.csv |
| 3 seed inventory (proposed fresh block 20501-20520) | smallest and largest recorded seed | -6, 29800895897491575 | seed | results/v2_T2_locked/consolidation/seed_inventory.csv |
| 3 seed inventory (proposed fresh block 20501-20520) | files scanned | 31,167 JSON and 3,070 CSV files in this repository (all worktrees), /home/fjiang4/tournament_experiment_upload_20260927, /home/fjiang4/TEL_PPO and the Phase 0 audit (source: report text; tools/v2/seed_inventory.py output) | text | reports/v2/protocol_lock_and_rehearsal.md section 2.1 |
