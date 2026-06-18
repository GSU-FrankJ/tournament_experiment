# r5_sampled re-run wave — launch manifest (2026-06-10)

Sampled-training re-runs replacing all closed-form-trained legacy results.
Code state: branch `fix/audit-remediation` @ `4446360` (sampled envs, MC-FD baselines,
eps_eq=0.03 unified, max_updates=1500, verified-stop reporting, overwrite guards).
All runs stop on the method's own stability+exploitability criterion.

Stage 1 (approved): 2P q=35 seed=42 → stop=exploitability@49, effort 43.36 vs e*=45.45
(4.62%), exploit 0.0070. Reused as A1's q35/s42 cell and the Fig-7 "full" arm member.

## Run matrix (159 new runs + 1 reused; seeds 42-46 everywhere)

| Group | Scenario / config | q | Tag | Output pattern (results/...) | Runs |
|---|---|---|---|---|---|
| A1 | 2P Set 1 (TEL-PPO, warmup 200) | 35,45,55 | r5_sampled | two_players/convergence/ppo_q{Q}.0_seed{S}_r5_sampled_convergence.json | 14 (+1 done) |
| A1b | 2P Set 2 (--k 6e-4 --w_h 8 --w_l 4, variant wh8_wl4) | 35,45,55 | r5_sampled | two_players/convergence/ppo_q{Q}.0_seed{S}_wh8_wl4_r5_sampled_convergence.json | 15 |
| A2 | 3P (TEL-PPO v2, warmup 200, min 300) | 35,55 | r5_sampled | three_players/convergence/ppo_3p_q{Q}.0_seed{S}_r5_sampled_convergence.json | 10 |
| A3 | het-cost (TEL-PPO v2, warmup 200, min 300) | 35,55 | r5_sampled | different_cost/convergence/different_cost_ppo_q{Q}.0_seed{S}_r5_sampled_convergence.json | 10 |
| A4a | het-ability STANDARD (min 1000) | 35,55 | r5_sampled_std | different_ability/convergence/different_ability_ppo_q{Q}.0_seed{S}_r5_sampled_std_convergence.json | 10 |
| A4b | het-ability THEORY-ALIGN-V2 (warmup 200, min 1000) | 35,55 | r5_sampled_v2 | different_ability/convergence/different_ability_ppo_q{Q}.0_seed{S}_r5_sampled_v2_convergence.json | 10 |
| B | MC-FD gradient, every scenario (2P Set1+Set2, 3P, dc, da) | as above | r5_sampled (Set2: r5_sampled_wh8_wl4) | …gradient…_q{Q}.0_seed{S}_{tag}_convergence.json | 60 |
| C1 | Fig-7 no-stability-screening (2P, --disable-cheap-gate) | 35,45,55 | r5_fig7_no_stability | two_players/convergence/ppo_q{Q}.0_seed{S}_r5_fig7_no_stability_convergence.json | 15 |
| C2 | Fig-7 no-exploitability-verification (2P, --disable-exploitability; runs to 1500-update budget) | 35,45,55 | r5_fig7_no_exploit | two_players/convergence/ppo_q{Q}.0_seed{S}_r5_fig7_no_exploit_convergence.json | 15 |
| C0 | Fig-7 full arm | 35,45,55 | r5_sampled | = A1 runs (reuse; no copies) | 0 |

Post-landing: Fig-7 JSONs + manifest collected into results/ablation/r5_fig7/.
da promotion (std vs v2) decided on sampled results; whole-wave promotion to baseline
(BASELINE_OVERRIDES -> r5_*) is a separate approved step after reconciliation.

## MC-FD solver parameters (standardized; FLAG FOR REVIEW)
`--grad-epsilon 0.5 --grad-samples 4096 --grad-lr 2.0 --grad-lr-decay 0.9999 --grad-steps 10000 --grad-tol 1e-6`
Rationale: config defaults (delta=0.01, 64 samples, lr 5.0 no decay for 2P) were tuned for
the REMOVED closed-form solver; under sampling, delta=0.01 gives FD-gradient SNR << 1.
With delta=0.5, N=4096 CRN: per-iter gradient SE ~= 0.007 vs signal 0.01-0.05.
LAUNCH CORRECTION (logged in manifest.csv): the first pass used decay 0.9995 x 4000 iters,
which truncates convergence (~7 contraction time-constants; trajectories still ascending at
the budget; q35 s42 landed 2.5/1.6 short of e*). Corrected to decay 0.9999 x 10000 iters
(~22 time-constants); verification run: e=45.137 vs e*=45.455 (0.7%), last-1000 hover
45.25+/-0.04. The 28 truncated first-pass files were overwritten via the --force escape
hatch (no deletions). 2P note: shared CRN draws synchronize the two players' trajectories
(e1 == e2), so 2P gradient symmetry gap is trivially ~0.

## Scheduling
- 8 GPUs; one run per GPU; tmux sessions r5w_gpu0..r5w_gpu7 run results/r5_sampled/worker.sh.
- GPUs 0-5: long_jobs.txt (55: 20 da -> 15 no_exploit -> 10 3P -> 10 dc, longest-first).
- GPUs 6-7: short_jobs.txt (44: 14 A1 + 15 A1b + 15 C1), then fall through to long_jobs.txt.
- tmux r5w_gradient: gradient_jobs.txt sequentially on CPU (CUDA_VISIBLE_DEVICES="").
- Live status: results/r5_sampled/manifest.csv (START/END/rc per job); per-job logs in
  results/r5_sampled/logs/.
- Expected wall-clock: ~2.5-3.5 days (dominated by 20 da runs @ ~12-24 h).

## Expected terminal state
160 r5-tagged convergence JSONs:
- two_players: 15 (A1) + 15 (A1b) + 15 (C1) + 15 (C2) + 30 gradient = 90
- three_players: 10 + 10 gradient = 20
- different_cost: 10 + 10 gradient = 20
- different_ability: 20 (std+v2) + 10 gradient = 30
