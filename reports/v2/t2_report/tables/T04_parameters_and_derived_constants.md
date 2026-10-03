# T04: Parameters and derived constants

- priority: core; status: generated; tier: n/a
- sources: `utils/theory_multistage.py`, `envs/curriculum_env.py`, `utils/dp_br_verifier.py`, `utils/v2_metrics.py`, `protocols/v2_T2_locked_v1_1.json`, `results/v2_pilots/phase1/calibration/calibration.csv`, `results/v2_T2_locked/confirmation/q50/seed20501/manifest.json`, `results/v2_T2_locked/confirmation/q60/seed20501/manifest.json`
- built by: `tools/v2/report/sec_p0p1.py:build_t04`; base commit `cb0b541`
- transformation: Game parameters from the locked protocol records; every derived constant computed with the repository's functions (g1_two_stage, g2_two_stage, f_xi, q_crit, GameSpec.B/dw/domain_half, stage_grid, symmetric_grid; dev_2x tier = state step 2, effort step 0.5, GL 16 as in tools/v2/common.py); recorded_* columns cross-read the same numbers from the protocol, from the Phase 1 calibration rows and from the confirmation manifests (grids block).

Parameters and derived constants of the T=2 game, per q. Node counts come from the grid constructors (no typed numbers). dev_2x is the Phase 1 2x-finer check tier.

| quantity | symbol | q50 | q60 | unit | definition | computed_with | recorded_q50 | recorded_q60 | recorded_in | matches_record |
|---|---|---|---|---|---|---|---|---|---|---|
| Winner prize | W_H | 6 | 6 | payoff units | prize of the player with the positive final gap | GameSpec.w_h | 6 | 6 | protocols/v2_T2_locked_v1_1.json records[q].game.w_h | True |
| Loser prize | W_L | 2 | 2 | payoff units | prize of the player with the negative final gap | GameSpec.w_l | 2 | 2 | protocols/v2_T2_locked_v1_1.json records[q].game.w_l | True |
| Prize spread | DW = W_H - W_L | 4 | 4 | payoff units | normalizer of every deviation metric | envs/curriculum_env.py:GameSpec.dw | 4 | 4 | protocols/v2_T2_locked_v1_1.json records[q].dw | True |
| Effort cost coefficient | k | 0.0002857 | 0.0002857 | payoff units per effort unit squared | stage cost c(e) = k e^2 | GameSpec.k | 0.0002857 | 0.0002857 | protocols/v2_T2_locked_v1_1.json records[q].game.k | True |
| Effort lower bound | e_min | 0 | 0 | effort units | lower end of the effort interval | GameSpec.e_min | 0 | 0 | protocols/v2_T2_locked_v1_1.json records[q].game.e_min | True |
| Effort upper bound | e_max | 100 | 100 | effort units | upper end of the effort interval | GameSpec.e_max | 100 | 100 | protocols/v2_T2_locked_v1_1.json records[q].game.e_max | True |
| Horizon | T | 2 | 2 | stages | number of effort stages | GameSpec.T | 2 | 2 | protocols/v2_T2_locked_v1_1.json records[q].game.T | True |
| Shock half-width | q | 50 | 60 | effort units | eps ~ U(-q, q) | GameSpec.q | 50 | 60 | protocols/v2_T2_locked_v1_1.json records[q].game.q | True |
| Shock model | eps, xi | eps_i,t ~ U(-50, 50) i.i.d. per player and stage; xi = eps_i - eps_j ~ Triangular(-100, 100) with density f_xi(x) = (2q - \|x\|)/(4q^2); d_(t+1) = d_t + e_i - e_j + xi_t | eps_i,t ~ U(-60, 60) i.i.d. per player and stage; xi = eps_i - eps_j ~ Triangular(-120, 120) with density f_xi(x) = (2q - \|x\|)/(4q^2); d_(t+1) = d_t + e_i - e_j + xi_t |  | per-player uniform action shocks; the gap moves by the shock difference xi | envs/curriculum_env.py:step_gap; utils/theory_multistage.py:f_xi |  |  |  |  |
| Density of xi at 0 | f_xi(0) = 1/(2q) | 0.01 | 0.008333 | 1/effort units | peak of the triangular density | utils/theory_multistage.py:f_xi |  |  |  |  |
| Maximal one-stage gap change | B = (e_max - e_min) + 2q | 200 | 220 | effort units | half-width of D_2 | envs/curriculum_env.py:GameSpec.B | 200 | 220 | protocols/v2_T2_locked_v1_1.json records[q].B | True |
| Stage-1 closed-form effort | e1*(0) = DW/(6kq) | 46.67 | 38.89 | effort units | benchmark stage-1 effort at d = 0 | utils/theory_multistage.py:g1_two_stage | 46.67 | 38.89 | results/v2_pilots/phase1/calibration/calibration.csv: g1 (analytic_eq, final) | True |
| Stage-2 closed-form effort at d = 0 | e2*(0) = DW f_xi(0)/(2k) | 70 | 58.33 | effort units | peak of e2*(d) = clip(DW f_xi(d)/(2k), 0, e_max) | utils/theory_multistage.py:g2_two_stage | 70 | 58.33 | results/v2_pilots/phase1/calibration/calibration.csv: g2_at_0 (analytic_eq, final) | True |
| r | r = DW/(k q^2) | 5.6 | 3.889 | dimensionless | prize spread over the cost of a q-sized effort | computed from GameSpec fields |  |  |  |  |
| Positive-effort support of e2*(d) | \|d\| < 2q | (-100, 100) | (-120, 120) | effort units (gap d) | open interval on which e2*(d) > 0 (e2* = 0 at \|d\| = 2q; checked on the recovery grid) | utils/theory_multistage.py:g2_two_stage |  |  |  |  |
| Closed-form validity threshold | q_crit | 41.83 | 41.83 | effort units | max(q_soc, stage-2 and stage-1 effort bounds, participation); the closed form is valid for q > q_crit | utils/theory_multistage.py:q_crit |  |  |  |  |
| Stage-1 state domain | D_1 | {0} | {0} | effort units (gap d) | root only | utils/dp_br_verifier.py:stage_grid |  |  |  |  |
| D_1 nodes, development tier | \|D_1\| | 1 | 1 | nodes | stage-1 grid (every tier) | utils/dp_br_verifier.py:stage_grid | 1 | 1 | results/v2_T2_locked/confirmation/q*/seed20501/manifest.json grids.development.stage_grid_points.1 | True |
| D_1 nodes, dev_2x tier | \|D_1\| | 1 | 1 | nodes | stage-1 grid (every tier) | utils/dp_br_verifier.py:stage_grid |  |  |  |  |
| D_1 nodes, final tier | \|D_1\| | 1 | 1 | nodes | stage-1 grid (every tier) | utils/dp_br_verifier.py:stage_grid | 1 | 1 | results/v2_T2_locked/confirmation/q*/seed20501/manifest.json grids.final.stage_grid_points.1 | True |
| Stage-2 state domain | D_2 = [-B, B] | [-200, 200] | [-220, 220] | effort units (gap d) | feasible stage-2 gaps from the root | envs/curriculum_env.py:GameSpec.domain_half | [-200, 200] | [-220, 220] | protocols/v2_T2_locked_v1_1.json records[q].domain_half_stage2 | True |
| D_2 nodes, development tier (state step 4) | \|D_2\| | 101 | 111 | nodes | symmetric grid with 0 and both endpoints (training-time verifier calls) | utils/dp_br_verifier.py:stage_grid | 101 | 111 | results/v2_T2_locked/confirmation/q*/seed20501/manifest.json grids.development.stage_grid_points.2 | True |
| D_2 nodes, dev_2x tier (state step 2) | \|D_2\| | 201 | 221 | nodes | symmetric grid with 0 and both endpoints (Phase 1 calibration only) | utils/dp_br_verifier.py:stage_grid | 201 | 221 | results/v2_pilots/phase1/calibration/calibration.csv: DeltaT_over_dw_n_on + DeltaT_over_dw_n_off (analytic_eq, dev_2x) | True |
| D_2 nodes, final tier (state step 2) | \|D_2\| | 201 | 221 | nodes | symmetric grid with 0 and both endpoints (gates and end-of-run evaluation) | utils/dp_br_verifier.py:stage_grid | 201 | 221 | results/v2_T2_locked/confirmation/q*/seed20501/manifest.json grids.final.stage_grid_points.2 | True |
| Recovery grid nodes on D_2 (step 0.5; tier-independent) | \|D_2 recovery\| | 801 | 881 | nodes | grid of the closed-form recovery metrics (0 is an exact node) | utils/v2_metrics.py:symmetric_grid | 801 | 881 | results/v2_pilots/phase1/calibration/calibration.csv: recovery_n_pos + recovery_n_tail | True |
| Recovery grid nodes with \|d\| < 2q | n_pos | 399 | 479 | nodes | nodes of the RMSE region | utils/v2_metrics.py:recovery_metrics | 399 | 479 | results/v2_pilots/phase1/calibration/calibration.csv: recovery_n_pos | True |
| Recovery grid nodes with \|d\| >= 2q | n_tail | 402 | 402 | nodes | nodes of the tail region | utils/v2_metrics.py:recovery_metrics | 402 | 402 | results/v2_pilots/phase1/calibration/calibration.csv: recovery_n_tail | True |
