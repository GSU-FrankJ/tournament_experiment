| q | runs | pass | fail | limit on failures | exact 95 % interval of the pass rate | condition met |
|---|---|---|---|---|---|---|
| 50 | 40 | 40 | 0 | 4 | [0.912, 1.000] | yes |
| 60 | 40 | 39 | 1 | 4 | [0.868, 0.999] | yes |
| both | 80 | 79 | 1 | 6 | [0.932, 1.000] | yes |

**Verdict under the rule: PASS.** at most 6 of the 80 runs fail and at most 4 of the 40 at each q. Blind recomputation (`tools/t3/t3r4_blind.py`): verdict pass, failures 0 + 1 = 1; agrees with the analysis: True; differences: [].
