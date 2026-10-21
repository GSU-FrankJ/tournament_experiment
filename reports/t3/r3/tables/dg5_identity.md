| q | seed | 64 exports | stage-3 rows of ms_updates.csv | ms_checks_stage3.csv | all identical | wall time (s) |
|---|---|---|---|---|---|---|
| 50 | 60501 | identical | identical | identical | yes | 321.8 |
| 50 | 60506 | identical | identical | identical | yes | 306.4 |
| 50 | 60513 | identical | identical | identical | yes | 312.5 |
| 50 | 60515 | identical | identical | identical | yes | 303.8 |
| 60 | 60516 | identical | identical | identical | yes | 319.3 |
| 60 | 60518 | identical | identical | identical | yes | 316 |
| 50 | 60505 | identical | identical | identical | yes | 312.6 |
| 50 | 60507 | identical | identical | identical | yes | 305.3 |
| 60 | 60503 | identical | identical | identical | yes | 319.1 |
| 60 | 60509 | identical | identical | identical | yes | 313.7 |
| 50 | 60502 | identical | identical | identical | yes | 306 |
| 50 | 60504 | identical | identical | identical | yes | 312.3 |
| 60 | 60501 | identical | identical | identical | yes | 317.8 |
| 60 | 60502 | identical | identical | identical | yes | 319.5 |

Freeze arrays `freeze_stage3_{final,development}.npz`, rollout clamp counters against the replay's stored-action counts and global-RNG assertions: 14 of 14 replays bit-identical in all items; global-RNG violations in total 0.
