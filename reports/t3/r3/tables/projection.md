| arm / trigger | assumed rates | fail q50 | fail q60 | expected failures in 40 runs | P(development target) |
|---|---|---|---|---|---|
| bw (control, observed rates) | central | 0.15 | 0.1 | 5 | 0.1077 |
| gate-level: Delta > 0.005 | central | 0.0225 | 0.0775 | 2 | 0.6768 |
| gate-level: Delta > 0.005 | pessimistic | 0.1827 | 0.2224 | 8.102 | 0.007143 |
| gate-level: Delta > 0.005 | optimistic | 0.003261 | 0.01615 | 0.3883 | 0.9933 |
| formation-level: R > 0.7 | central | 0.15 | 0.0775 | 4.55 | 0.1498 |

Rates from the 80 recorded runs (`retro_checks_local1600.csv`): failures q50 6 of 40, q60 4 of 40; the gate-level trigger fires on 6 (q50) and 1 (q60) runs and misses 0 and 3 failures; the formation-level trigger fires on 0 and 1. 'pessimistic' / 'optimistic' use the upper / lower limit of the exact 95 % interval of every rate. **This is a model, not a result**: it assumes independent runs, a re-drawn stage 3 distributed as a first attempt, a run that fires again failing, and rates estimated from 40 runs per q; it is shown to size the target, not to predict the wave.
