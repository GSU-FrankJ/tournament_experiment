# T11: Benchmark consistency

- priority: supp; status: generated; tier: tier-independent
- sources: `results/v2_pilots/phase1/benchmark_consistency.csv`, `results/v2_pilots/phase1/benchmark_recheck.csv`, `tools/v2/benchmark_consistency.py`
- built by: `tools/v2/report/sec_p0p1.py:build_t11`; base commit `cb0b541`
- transformation: Per q from the per-cell CSV (756 cells, 200,000 draws per cell, environment code path) and the 40-replicate recheck CSV; z SD with ddof = 1; maxima over both players.

Monte Carlo of the environment's terminal step against the closed-form CDF F_xi. The maximum deviation over the MC standard error is max |z|; the two players are not independent tests.

| q | block | quantity | value | detail | computation |
|---|---|---|---|---|---|
| 50 | MC terminal win probability | grid cells | 756 | d: 21 nodes on [-200, 200] x e_i, e_j in {0, 20, 40, 60, 80, 100} | rows of benchmark_consistency.csv |
| 50 | MC terminal win probability | draws per cell and player | 2e+05 | both players use the same draws | column n |
| 50 | MC terminal win probability | interior cells (0 < F < 1) | 324 | player i | count of 0 < F_i < 1 |
| 50 | MC terminal win probability | z mean over interior cells | 0.04328 | player i; z = (p_hat - F)/SE, SE = sqrt(F(1 - F)/n) | mean of z_i |
| 50 | MC terminal win probability | z SD over interior cells | 1.014 | player i; ddof = 1 | SD of z_i |
| 50 | MC terminal win probability | max \|z\| = max deviation over the MC standard error | 4.541 | cell d=60, e_i=60, e_j=60, player j | max over cells and players of \|z\| |
| 50 | MC terminal win probability | max \|p_hat - F\| over interior cells | 0.00314 | both players | max \|diff\| |
| 50 | MC terminal win probability | cells with F in {0, 1} | 432 | player i | count |
| 50 | MC terminal win probability | max \|p_hat - F\| over cells with F in {0, 1} | 0 | both players | max \|diff\| |
| 50 | MC terminal win probability | max \|z_i + z_j\| | 2.327e-13 | player j's outcome is the complement of player i's on the same draws (z_j = -z_i): not independent tests | max over cells |
| 50 | recheck of the max-\|z\| cell | F at the recheck cell | 0.08 | own gap -60, e_own 60, e_opp 60 (the max-\|z\| cell in the player's own perspective) | column F |
| 50 | recheck of the max-\|z\| cell | replicates | 40 | 200000 draws each (fresh streams) | column reps |
| 50 | recheck of the max-\|z\| cell | z mean | -0.006594 | 40 replicates | column z_mean |
| 50 | recheck of the max-\|z\| cell | z SD (ddof = 1) | 1.132 | 40 replicates | column z_sd |
| 50 | recheck of the max-\|z\| cell | z mean over its SE | -0.03683 | 40 replicates | column z_mean_over_se |
| 50 | recheck of the max-\|z\| cell | max \|z\| | 2.382 | 40 replicates | column max_abs_z |
| 50 | MC expected terminal reward | max \|z\| of the mean reward over cells with a random reward | 4.615 | cell d=60, e_i=60, e_j=60, player i; r_bar = w_L + DW F - k e^2 | max \|r_z\| |
| 50 | MC expected terminal reward | cells with a constant reward | 432 | player i (z undefined, NaN) | count of r_const_i |
| 50 | MC expected terminal reward | max \|r_hat - r_bar\| over constant-reward cells | 8.882e-16 | both players | max \|r_diff\| |
| 60 | MC terminal win probability | grid cells | 756 | d: 21 nodes on [-220, 220] x e_i, e_j in {0, 20, 40, 60, 80, 100} | rows of benchmark_consistency.csv |
| 60 | MC terminal win probability | draws per cell and player | 2e+05 | both players use the same draws | column n |
| 60 | MC terminal win probability | interior cells (0 < F < 1) | 394 | player i | count of 0 < F_i < 1 |
| 60 | MC terminal win probability | z mean over interior cells | -0.03576 | player i; z = (p_hat - F)/SE, SE = sqrt(F(1 - F)/n) | mean of z_i |
| 60 | MC terminal win probability | z SD over interior cells | 1.024 | player i; ddof = 1 | SD of z_i |
| 60 | MC terminal win probability | max \|z\| = max deviation over the MC standard error | 3.83 | cell d=22, e_i=0, e_j=80, player i | max over cells and players of \|z\| |
| 60 | MC terminal win probability | max \|p_hat - F\| over interior cells | 0.002913 | both players | max \|diff\| |
| 60 | MC terminal win probability | cells with F in {0, 1} | 362 | player i | count |
| 60 | MC terminal win probability | max \|p_hat - F\| over cells with F in {0, 1} | 0 | both players | max \|diff\| |
| 60 | MC terminal win probability | max \|z_i + z_j\| | 1.841e-12 | player j's outcome is the complement of player i's on the same draws (z_j = -z_i): not independent tests | max over cells |
| 60 | recheck of the max-\|z\| cell | F at the recheck cell | 0.1335 | own gap 22, e_own 0, e_opp 80 (the max-\|z\| cell in the player's own perspective) | column F |
| 60 | recheck of the max-\|z\| cell | replicates | 40 | 200000 draws each (fresh streams) | column reps |
| 60 | recheck of the max-\|z\| cell | z mean | -0.2339 | 40 replicates | column z_mean |
| 60 | recheck of the max-\|z\| cell | z SD (ddof = 1) | 0.974 | 40 replicates | column z_sd |
| 60 | recheck of the max-\|z\| cell | z mean over its SE | -1.519 | 40 replicates | column z_mean_over_se |
| 60 | recheck of the max-\|z\| cell | max \|z\| | 2.12 | 40 replicates | column max_abs_z |
| 60 | MC expected terminal reward | max \|z\| of the mean reward over cells with a random reward | 3.796 | cell d=22, e_i=0, e_j=80, player j; r_bar = w_L + DW F - k e^2 | max \|r_z\| |
| 60 | MC expected terminal reward | cells with a constant reward | 362 | player i (z undefined, NaN) | count of r_const_i |
| 60 | MC expected terminal reward | max \|r_hat - r_bar\| over constant-reward cells | 8.882e-16 | both players | max \|r_diff\| |
