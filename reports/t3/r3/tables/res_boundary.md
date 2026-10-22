| manifests record | runs | `bw` | `bwA` | `bwAB` | first start (UTC) | last start (UTC) |
|---|---|---|---|---|---|---|
| a clean tree | 40 | 0 | 1 | 39 | 2026-10-22T02:20:08.000+00:00 | 2026-10-22T02:20:09.000+00:00 |
| a dirty tree | 82 | 40 | 40 | 2 | 2026-10-22T02:29:08.000+00:00 | 2026-10-22T02:48:43.000+00:00 |

`reports/t3/r3/report_scripts/results_tables.py` was created at 2026-10-22 02:21:15.679567227 +0000 (birth time, `stat`); the wave was launched at 02:20:02 UTC. Runs whose recorded flag disagrees with 'started after the file was created': 0. All 122 manifests carry the commit `b565f0e8`; the diagnostic runs are all in the clean group (0 dirty). Table: `results/t3_r3/dev/verify_clean_tree/boundary.csv`.
