| check | pass | fail | not evaluated | first call (relative `--root`): pass / fail |
|---|---|---|---|---|
| exit_ok | 122 | 0 | 0 | 122 / 0 |
| files_complete | 122 | 0 | 0 | 122 / 0 |
| manifest_commit_ok | 122 | 0 | 0 | 122 / 0 |
| clean_tree | 40 | 82 | 0 | 40 / 82 |
| global_rng_ok | 122 | 0 | 0 | 122 / 0 |
| config_equals_plan | 122 | 0 | 0 | 0 / 122 |
| config_production_ok | 122 | 0 | 0 | 122 / 0 |
| trigger_consistent | 122 | 0 | 0 | 122 / 0 |
| budget_ok | 122 | 0 | 0 | 122 / 0 |
| lr_ok | 122 | 0 | 0 | 122 / 0 |
| refresh_ok | 122 | 0 | 0 | 122 / 0 |
| exports_ok | 122 | 0 | 0 | 122 / 0 |
| checks_ok | 122 | 0 | 0 | 122 / 0 |
| retry_budget_ok | 122 | 0 | 0 | 122 / 0 |
| init_equals_plan | 122 | 0 | 0 | 122 / 0 |
| replay_ok | 122 | 0 | 0 | 122 / 0 |
| pair_ok (binding for `bwA` and the diagnostic runs only) | 122 | 0 | 0 | 122 / 0 |

| arm | runs | trigger fired | trigger would fire (recomputed) | failed checks (other than `pair_ok`) | C-PAIR against `bw` |
|---|---|---|---|---|---|
| bw | 40 | 0 | 2 | clean_tree: 40 | - |
| bwA | 41 | 3 | 3 | clean_tree: 40 | 40 of 40 development runs as required (identical over the whole run where the trigger did not fire, same stage 3 up to the check where it fired); 1 of 1 diagnostic run equal to T3-R2's original up to the check |
| bwAB | 41 | 4 | 4 | clean_tree: 2 | 0 of 40 development runs identical to `bw` (not a check: `bwAB` differs from `bw` by construction); 1 of 1 diagnostic run equal to T3-R2's original up to the check |

Infrastructure ledger present: True; events: 0; stopped without verdict: []. Cross-arm C-INIT: attempt-1 digest differs across arms in 0 (q, seed); attempt-2 digests of `bwA` and `bwAB` differ in 0.
