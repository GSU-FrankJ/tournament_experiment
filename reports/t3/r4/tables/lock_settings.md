| item | value |
|---|---|
| records | the v2.0 record of each q (`protocols/v2_T2_locked_v2_0.json`), `game.T` = 3 |
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
