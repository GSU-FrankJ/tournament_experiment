| run | trigger fired in the original | re-run | original (T3-R3 untracked root) | code commit of the re-run | `compare_runs.py bw` exit code | npz files compared | same set of files | stage-2 and stage-3 end states (bitwise) | gates, rule log, drift test, summary, manifest, train history | `attempt1/` and `retry.json` | all identical | wall time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t3r3_q50_s70508_bwA | yes | results/t3_r4/repro/q50/seed70508/bwA | results/t3_r3/dev/q50/seed70508/bwA | 631c19f1 | 0 | 162 | yes | yes | identical | identical | yes | 575.3 |
| t3r3_q60_s70501_bwA | no | results/t3_r4/repro/q60/seed70501/bwA | results/t3_r3/dev/q60/seed70501/bwA | 631c19f1 | 0 | 162 | yes | yes | identical | - | yes | 434.8 |
