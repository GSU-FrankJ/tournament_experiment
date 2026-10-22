| arm | role | scheme keys | runs | order in the selection |
|---|---|---|---|---|
| `bw` | control (locked T3-1.0 settings); not eligible | none | 40 (+ none diagnostic) | - |
| `bwA` | candidate A | `stage3_retry = {"delta_gt": 0.005, "max_retries": 1}` | 40 + 1 diagnostic | 1 |
| `bwAB` | candidate A and B | `stage3_retry` as above and `terminal_clamp_likelihood = "censored"` | 40 + 1 diagnostic | 2 |
