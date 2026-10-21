| case | updates | policy rows | minibatch steps | development checks | exports (frozen path) | relative to bw |
|---|---|---|---|---|---|---|
| bw, and bwA / bwAB when the trigger does not fire | 3800 | 1945600 | 76000 | 152 | 152 | 1 |
| bwA / bwAB when the trigger fires (stage 3 trained twice) | 5400 | 2764800 | 108000 | 216 | 152 (+ 64 in attempt1/) | 1.421 |
| bwA / bwAB expected, fire rate 7 / 80 = 0.0875 | 3940 | 2.02e+06 | 7.88e+04 | 157.6 | 152 | 1.037 |
| bwA / bwAB expected, lower limit of the 95 % interval = 0.0359 | 3857 | 1.98e+06 | 7.715e+04 | 154.3 | 152 | 1.015 |
| bwA / bwAB expected, upper limit = 0.1720 | 4075 | 2.09e+06 | 8.15e+04 | 163 | 152 | 1.072 |

Wall time of the 80 recorded `bw` runs (`status.json` `total_wall_sec`): median 600 s, range 503-640 s; the stage-3 phase (`ms_run_summary.json`) median 235 s, i.e. 39% of a run. One retraining therefore costs about 235 s; a wave of 40 `bw` runs costs about 6.7 CPU-hours, of 40 runs of an arm with the expected extra stage-3 work about 6.9. The censored update measured on a synthetic batch of 512 rows with 466 flagged rows (`agent.update` with `clamp_side`, mean of 3 calls, one thread) took 1.17 times the plain update (0.1355 against 0.1161 s); a regime run with about 57 % flagged rows costs less.
