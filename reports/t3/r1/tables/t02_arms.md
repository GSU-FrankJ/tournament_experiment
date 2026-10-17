| arm | T | scheme | policy / starts | updates | policy rows | minibatch steps | learning rate (actor, critic) |
|---|---|---|---|---|---|---|---|
| `bw` | 3 | backward induction (existing `run/run_ms_stagewise.py`, T = 3) | stage 3 alone, then stage 2 (stage 3 frozen), then stage 1 (stages 2, 3 frozen) | 3,800 | 1,945,600 | 76,000 | 3e-4, linear decay to 3e-5 over local updates 1201-1600 (stages 3 and 2) and 1-600 (stage 1) |
| `jt` | 3 | joint whole-game training (`run/run_joint.py`) | one network for all stages; starts at stage 1 only (root play) | 1,267 | 1,946,112 | 76,020 | 3e-4 up to 950, linear to 3e-5 over 951-1267 |
| `jt_es` | 3 | joint whole-game training with exploring starts (`run/run_joint.py`) | one network for all stages; starts 256 root + 85 stage-2 + 171 stage-3 per update | 1,754 | 1,945,186 | 87,700 | 3e-4 up to 1315, linear to 3e-5 over 1316-1754 |
| `v20` | 2 | recorded v2.0 confirmation runs, re-evaluated | two-phase backward induction of protocol v2.0 | 2,200 | 1,126,400 | 56,000 | as recorded by v2.0 |
| `jt2` | 2 | joint whole-game training at T = 2 (`run/run_joint.py`) | one network for both stages; root play only | 1,100 | 1,126,400 | 44,000 | 3e-4 up to 825, linear to 3e-5 over 826-1100 |
