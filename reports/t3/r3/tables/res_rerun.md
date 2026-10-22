| run | why this run | flag, original | flag, re-run | `compare_runs.py bw` exit code | npz files identical | same file set | `attempt1/` (64 exports, CSVs, bin maps, state) and `retry.json` | gates / rule log / drift / summary JSON | re-run wall time (s, driver clock) |
|---|---|---|---|---|---|---|---|---|---|
| `t3r3_q50_s70508_bwA` | dirty-flagged bwA run whose trigger fired | dirty | clean | 0 | 162 | yes | identical | identical | 496 |
| `t3r3_q50_s70516_bwA` | dirty-flagged bwA run whose trigger fired | dirty | clean | 0 | 162 | yes | identical | identical | 494 |
| `t3r3_q60_s70519_bwAB` | lowest-numbered dirty-flagged bwAB run at q=60 whose trigger did not fire | dirty | clean | 0 | 162 | yes | no `attempt1/` | identical | 368 |
| `t3r3_q50_s70501_bwA` | (addition) lowest-numbered dirty-flagged bwA run at q=50 whose trigger did not fire | dirty | clean | 0 | 162 | yes | no `attempt1/` | identical | 377 |
| `t3r3_q60_s70501_bwA` | (addition) lowest-numbered dirty-flagged bwA run at q=60 whose trigger did not fire | dirty | clean | 0 | 162 | yes | no `attempt1/` | identical | 363 |
