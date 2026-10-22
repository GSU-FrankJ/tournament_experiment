| q | runs | pass | fail | limit on failures | exact 95 % interval of the pass rate | condition met |
|---|---|---|---|---|---|---|
| 50 | 40 | 40 | 0 | 4 | [0.912, 1.000] | yes |
| 60 | 40 | 38 | 2 | 4 | [0.831, 0.994] | yes |
| both | 80 | 78 | 2 | 6 | [0.913, 0.997] | yes |

**Verdict under the rule: PASS.** at most 6 of the 80 runs fail and at most 4 of the 40 at each q. Blind recomputation (`tools/t3/t3r4_blind.py`): verdict pass, failures 0 + 2 = 2; agrees with the analysis: True; differences: [].
