# PROVENANCE — paper tables and figures (r5_sampled, final)

Generated 2026-06-12 after the r5 promotion (commits 4db3bdb code, 693e74c artifacts).
Every TEL-PPO and Gradient number in the paper artifacts traces to the run files below
(all committed in d6a6e81), produced by the job queues in `results/r5_sampled/*.txt`
with per-job logs in `results/r5_sampled/logs/` and live status in
`results/r5_sampled/manifest.csv`. Code state for every run: branch
`fix/audit-remediation` @ `4446360` (sampled envs, sampled MC-FD baselines,
eps_eq=0.03, max_updates=1500). Theory rows are closed-form formulas
(`utils/theory.py`; no runs). Nothing in this file is estimated; UNVERIFIED items are
listed explicitly in section 4.

## 1. Table cells -> runs (final_summary.csv / .tex)

### Two-Player (Set 1) | q=35 | TEL-PPO  (tag `r5_sampled`, seeds [42, 43, 44, 45, 46])
- `results/two_players/convergence/ppo_q35.0_seed42_r5_sampled_convergence.json` (stop=exploitability@49)
- `results/two_players/convergence/ppo_q35.0_seed43_r5_sampled_convergence.json` (stop=exploitability@59)
- `results/two_players/convergence/ppo_q35.0_seed44_r5_sampled_convergence.json` (stop=exploitability@49)
- `results/two_players/convergence/ppo_q35.0_seed45_r5_sampled_convergence.json` (stop=exploitability@69)
- `results/two_players/convergence/ppo_q35.0_seed46_r5_sampled_convergence.json` (stop=exploitability@49)
- launch (one seed shown; others differ only in `--seed`): `python3 run/run_two_players.py --method ppo --q 35 --seed 43 --override-conc-ramp-warmup 200 --episodes 6144000 --ablation-name r5_sampled`

### Two-Player (Set 1) | q=45 | TEL-PPO  (tag `r5_sampled`, seeds [42, 43, 44, 45, 46])
- `results/two_players/convergence/ppo_q45.0_seed42_r5_sampled_convergence.json` (stop=exploitability@69)
- `results/two_players/convergence/ppo_q45.0_seed43_r5_sampled_convergence.json` (stop=exploitability@79)
- `results/two_players/convergence/ppo_q45.0_seed44_r5_sampled_convergence.json` (stop=exploitability@69)
- `results/two_players/convergence/ppo_q45.0_seed45_r5_sampled_convergence.json` (stop=exploitability@79)
- `results/two_players/convergence/ppo_q45.0_seed46_r5_sampled_convergence.json` (stop=exploitability@79)
- launch (one seed shown; others differ only in `--seed`): `python3 run/run_two_players.py --method ppo --q 45 --seed 42 --override-conc-ramp-warmup 200 --episodes 6144000 --ablation-name r5_sampled`

### Two-Player (Set 1) | q=55 | TEL-PPO  (tag `r5_sampled`, seeds [42, 43, 44, 45, 46])
- `results/two_players/convergence/ppo_q55.0_seed42_r5_sampled_convergence.json` (stop=exploitability@109)
- `results/two_players/convergence/ppo_q55.0_seed43_r5_sampled_convergence.json` (stop=exploitability@79)
- `results/two_players/convergence/ppo_q55.0_seed44_r5_sampled_convergence.json` (stop=exploitability@69)
- `results/two_players/convergence/ppo_q55.0_seed45_r5_sampled_convergence.json` (stop=exploitability@79)
- `results/two_players/convergence/ppo_q55.0_seed46_r5_sampled_convergence.json` (stop=exploitability@99)
- launch (one seed shown; others differ only in `--seed`): `python3 run/run_two_players.py --method ppo --q 55 --seed 42 --override-conc-ramp-warmup 200 --episodes 6144000 --ablation-name r5_sampled`

### Two-Player (Set 1) | q=35 | Gradient  (tag `r5_sampled`, seeds [42, 43, 44, 45, 46])
- `results/two_players/convergence/gradient_q35.0_seed42_r5_sampled_convergence.json` (iters=10000)
- `results/two_players/convergence/gradient_q35.0_seed43_r5_sampled_convergence.json` (iters=10000)
- `results/two_players/convergence/gradient_q35.0_seed44_r5_sampled_convergence.json` (iters=10000)
- `results/two_players/convergence/gradient_q35.0_seed45_r5_sampled_convergence.json` (iters=10000)
- `results/two_players/convergence/gradient_q35.0_seed46_r5_sampled_convergence.json` (iters=10000)
- launch (one seed shown; others differ only in `--seed`): `python3 run/run_two_players.py --method gradient --q 35 --seed 42 --grad-epsilon 0.5 --grad-samples 4096 --grad-lr 2.0 --grad-lr-decay 0.9999 --grad-steps 10000 --grad-tol 1e-6 --ablation-name r5_sampled --force`

### Two-Player (Set 1) | q=45 | Gradient  (tag `r5_sampled`, seeds [42, 43, 44, 45, 46])
- `results/two_players/convergence/gradient_q45.0_seed42_r5_sampled_convergence.json` (iters=10000)
- `results/two_players/convergence/gradient_q45.0_seed43_r5_sampled_convergence.json` (iters=10000)
- `results/two_players/convergence/gradient_q45.0_seed44_r5_sampled_convergence.json` (iters=10000)
- `results/two_players/convergence/gradient_q45.0_seed45_r5_sampled_convergence.json` (iters=10000)
- `results/two_players/convergence/gradient_q45.0_seed46_r5_sampled_convergence.json` (iters=10000)
- launch (one seed shown; others differ only in `--seed`): `python3 run/run_two_players.py --method gradient --q 45 --seed 42 --grad-epsilon 0.5 --grad-samples 4096 --grad-lr 2.0 --grad-lr-decay 0.9999 --grad-steps 10000 --grad-tol 1e-6 --ablation-name r5_sampled --force`

### Two-Player (Set 1) | q=55 | Gradient  (tag `r5_sampled`, seeds [42, 43, 44, 45, 46])
- `results/two_players/convergence/gradient_q55.0_seed42_r5_sampled_convergence.json` (iters=9059)
- `results/two_players/convergence/gradient_q55.0_seed43_r5_sampled_convergence.json` (iters=10000)
- `results/two_players/convergence/gradient_q55.0_seed44_r5_sampled_convergence.json` (iters=8458)
- `results/two_players/convergence/gradient_q55.0_seed45_r5_sampled_convergence.json` (iters=10000)
- `results/two_players/convergence/gradient_q55.0_seed46_r5_sampled_convergence.json` (iters=6562)
- launch (one seed shown; others differ only in `--seed`): `python3 run/run_two_players.py --method gradient --q 55 --seed 42 --grad-epsilon 0.5 --grad-samples 4096 --grad-lr 2.0 --grad-lr-decay 0.9999 --grad-steps 10000 --grad-tol 1e-6 --ablation-name r5_sampled --force`

### Three-Player | q=35 | TEL-PPO  (tag `r5_sampled`, seeds [42, 43, 44, 45, 46])
- `results/three_players/convergence/ppo_3p_q35.0_seed42_r5_sampled_convergence.json` (stop=exploitability@309)
- `results/three_players/convergence/ppo_3p_q35.0_seed43_r5_sampled_convergence.json` (stop=exploitability@302)
- `results/three_players/convergence/ppo_3p_q35.0_seed44_r5_sampled_convergence.json` (stop=exploitability@309)
- `results/three_players/convergence/ppo_3p_q35.0_seed45_r5_sampled_convergence.json` (stop=exploitability@304)
- `results/three_players/convergence/ppo_3p_q35.0_seed46_r5_sampled_convergence.json` (stop=exploitability@301)
- launch (one seed shown; others differ only in `--seed`): `python3 run/run_three_players.py --method ppo --q 35 --seed 42 --theory-align-v2 --override-conc-ramp-warmup 200 --min-updates 300 --episodes 6144000 --ablation-name r5_sampled`

### Three-Player | q=55 | TEL-PPO  (tag `r5_sampled`, seeds [42, 43, 44, 45, 46])
- `results/three_players/convergence/ppo_3p_q55.0_seed42_r5_sampled_convergence.json` (stop=exploitability@306)
- `results/three_players/convergence/ppo_3p_q55.0_seed43_r5_sampled_convergence.json` (stop=exploitability@303)
- `results/three_players/convergence/ppo_3p_q55.0_seed44_r5_sampled_convergence.json` (stop=exploitability@305)
- `results/three_players/convergence/ppo_3p_q55.0_seed45_r5_sampled_convergence.json` (stop=exploitability@300)
- `results/three_players/convergence/ppo_3p_q55.0_seed46_r5_sampled_convergence.json` (stop=exploitability@300)
- launch (one seed shown; others differ only in `--seed`): `python3 run/run_three_players.py --method ppo --q 55 --seed 42 --theory-align-v2 --override-conc-ramp-warmup 200 --min-updates 300 --episodes 6144000 --ablation-name r5_sampled`

### Three-Player | q=35 | Gradient  (tag `r5_sampled`, seeds [42, 43, 44, 45, 46])
- `results/three_players/convergence/gradient_3p_q35.0_seed42_r5_sampled_convergence.json` (iters=10000)
- `results/three_players/convergence/gradient_3p_q35.0_seed43_r5_sampled_convergence.json` (iters=10000)
- `results/three_players/convergence/gradient_3p_q35.0_seed44_r5_sampled_convergence.json` (iters=10000)
- `results/three_players/convergence/gradient_3p_q35.0_seed45_r5_sampled_convergence.json` (iters=10000)
- `results/three_players/convergence/gradient_3p_q35.0_seed46_r5_sampled_convergence.json` (iters=10000)
- launch (one seed shown; others differ only in `--seed`): `python3 run/run_three_players.py --method gradient --q 35 --seed 42 --grad-epsilon 0.5 --grad-samples 4096 --grad-lr 2.0 --grad-lr-decay 0.9999 --grad-steps 10000 --grad-tol 1e-6 --ablation-name r5_sampled --force`

### Three-Player | q=55 | Gradient  (tag `r5_sampled`, seeds [42, 43, 44, 45, 46])
- `results/three_players/convergence/gradient_3p_q55.0_seed42_r5_sampled_convergence.json` (iters=10000)
- `results/three_players/convergence/gradient_3p_q55.0_seed43_r5_sampled_convergence.json` (iters=10000)
- `results/three_players/convergence/gradient_3p_q55.0_seed44_r5_sampled_convergence.json` (iters=10000)
- `results/three_players/convergence/gradient_3p_q55.0_seed45_r5_sampled_convergence.json` (iters=10000)
- `results/three_players/convergence/gradient_3p_q55.0_seed46_r5_sampled_convergence.json` (iters=10000)
- launch (one seed shown; others differ only in `--seed`): `python3 run/run_three_players.py --method gradient --q 55 --seed 42 --grad-epsilon 0.5 --grad-samples 4096 --grad-lr 2.0 --grad-lr-decay 0.9999 --grad-steps 10000 --grad-tol 1e-6 --ablation-name r5_sampled --force`

### Het. Cost | q=35 | TEL-PPO  (tag `r5_sampled`, seeds [42, 43, 44, 45, 46])
- `results/different_cost/convergence/different_cost_ppo_q35.0_seed42_r5_sampled_convergence.json` (stop=exploitability@303)
- `results/different_cost/convergence/different_cost_ppo_q35.0_seed43_r5_sampled_convergence.json` (stop=exploitability@309)
- `results/different_cost/convergence/different_cost_ppo_q35.0_seed44_r5_sampled_convergence.json` (stop=exploitability@305)
- `results/different_cost/convergence/different_cost_ppo_q35.0_seed45_r5_sampled_convergence.json` (stop=exploitability@300)
- `results/different_cost/convergence/different_cost_ppo_q35.0_seed46_r5_sampled_convergence.json` (stop=exploitability@309)
- launch (one seed shown; others differ only in `--seed`): `python3 run/run_different_cost.py --method ppo --q 35 --seed 42 --theory-align-v2 --override-conc-ramp-warmup 200 --min-updates 300 --episodes 6144000 --ablation-name r5_sampled`

### Het. Cost | q=55 | TEL-PPO  (tag `r5_sampled`, seeds [42, 43, 44, 45, 46])
- `results/different_cost/convergence/different_cost_ppo_q55.0_seed42_r5_sampled_convergence.json` (stop=exploitability@302)
- `results/different_cost/convergence/different_cost_ppo_q55.0_seed43_r5_sampled_convergence.json` (stop=exploitability@301)
- `results/different_cost/convergence/different_cost_ppo_q55.0_seed44_r5_sampled_convergence.json` (stop=exploitability@307)
- `results/different_cost/convergence/different_cost_ppo_q55.0_seed45_r5_sampled_convergence.json` (stop=exploitability@301)
- `results/different_cost/convergence/different_cost_ppo_q55.0_seed46_r5_sampled_convergence.json` (stop=exploitability@304)
- launch (one seed shown; others differ only in `--seed`): `python3 run/run_different_cost.py --method ppo --q 55 --seed 42 --theory-align-v2 --override-conc-ramp-warmup 200 --min-updates 300 --episodes 6144000 --ablation-name r5_sampled`

### Het. Cost | q=35 | Gradient  (tag `r5_sampled`, seeds [42, 43, 44, 45, 46])
- `results/different_cost/convergence/different_cost_gradient_q35.0_seed42_r5_sampled_convergence.json` (iters=10000)
- `results/different_cost/convergence/different_cost_gradient_q35.0_seed43_r5_sampled_convergence.json` (iters=10000)
- `results/different_cost/convergence/different_cost_gradient_q35.0_seed44_r5_sampled_convergence.json` (iters=10000)
- `results/different_cost/convergence/different_cost_gradient_q35.0_seed45_r5_sampled_convergence.json` (iters=10000)
- `results/different_cost/convergence/different_cost_gradient_q35.0_seed46_r5_sampled_convergence.json` (iters=10000)
- launch (one seed shown; others differ only in `--seed`): `python3 run/run_different_cost.py --method gradient --q 35 --seed 42 --grad-epsilon 0.5 --grad-samples 4096 --grad-lr 2.0 --grad-lr-decay 0.9999 --grad-steps 10000 --grad-tol 1e-6 --ablation-name r5_sampled --force`

### Het. Cost | q=55 | Gradient  (tag `r5_sampled`, seeds [42, 43, 44, 45, 46])
- `results/different_cost/convergence/different_cost_gradient_q55.0_seed42_r5_sampled_convergence.json` (iters=10000)
- `results/different_cost/convergence/different_cost_gradient_q55.0_seed43_r5_sampled_convergence.json` (iters=10000)
- `results/different_cost/convergence/different_cost_gradient_q55.0_seed44_r5_sampled_convergence.json` (iters=10000)
- `results/different_cost/convergence/different_cost_gradient_q55.0_seed45_r5_sampled_convergence.json` (iters=10000)
- `results/different_cost/convergence/different_cost_gradient_q55.0_seed46_r5_sampled_convergence.json` (iters=10000)
- launch (one seed shown; others differ only in `--seed`): `python3 run/run_different_cost.py --method gradient --q 55 --seed 42 --grad-epsilon 0.5 --grad-samples 4096 --grad-lr 2.0 --grad-lr-decay 0.9999 --grad-steps 10000 --grad-tol 1e-6 --ablation-name r5_sampled --force`

### Het. Ability | q=35 | TEL-PPO  (tag `r5_sampled_std`, seeds [42, 43, 44, 45, 46])
- `results/different_ability/convergence/different_ability_ppo_q35.0_seed42_r5_sampled_std_convergence.json` (stop=exploitability@1219)
- `results/different_ability/convergence/different_ability_ppo_q35.0_seed43_r5_sampled_std_convergence.json` (stop=exploitability@1192)
- `results/different_ability/convergence/different_ability_ppo_q35.0_seed44_r5_sampled_std_convergence.json` (stop=exploitability@1171)
- `results/different_ability/convergence/different_ability_ppo_q35.0_seed45_r5_sampled_std_convergence.json` (stop=exploitability@1023)
- `results/different_ability/convergence/different_ability_ppo_q35.0_seed46_r5_sampled_std_convergence.json` (stop=exploitability@1160)
- launch (one seed shown; others differ only in `--seed`): `python3 run/run_different_ability.py --method ppo --q 35 --seed 42 --min-updates 1000 --episodes 6144000 --ablation-name r5_sampled_std`

### Het. Ability | q=55 | TEL-PPO  (tag `r5_sampled_std`, seeds [42, 43, 44, 45, 46])
- `results/different_ability/convergence/different_ability_ppo_q55.0_seed42_r5_sampled_std_convergence.json` (stop=exploitability@1007)
- `results/different_ability/convergence/different_ability_ppo_q55.0_seed43_r5_sampled_std_convergence.json` (stop=exploitability@1000)
- `results/different_ability/convergence/different_ability_ppo_q55.0_seed44_r5_sampled_std_convergence.json` (stop=exploitability@1001)
- `results/different_ability/convergence/different_ability_ppo_q55.0_seed45_r5_sampled_std_convergence.json` (stop=exploitability@1002)
- `results/different_ability/convergence/different_ability_ppo_q55.0_seed46_r5_sampled_std_convergence.json` (stop=exploitability@1000)
- launch (one seed shown; others differ only in `--seed`): `python3 run/run_different_ability.py --method ppo --q 55 --seed 42 --min-updates 1000 --episodes 6144000 --ablation-name r5_sampled_std`

### Het. Ability | q=35 | Gradient  (tag `r5_sampled`, seeds [42, 43, 44, 45, 46])
- `results/different_ability/convergence/different_ability_gradient_q35.0_seed42_r5_sampled_convergence.json` (iters=10000)
- `results/different_ability/convergence/different_ability_gradient_q35.0_seed43_r5_sampled_convergence.json` (iters=10000)
- `results/different_ability/convergence/different_ability_gradient_q35.0_seed44_r5_sampled_convergence.json` (iters=10000)
- `results/different_ability/convergence/different_ability_gradient_q35.0_seed45_r5_sampled_convergence.json` (iters=10000)
- `results/different_ability/convergence/different_ability_gradient_q35.0_seed46_r5_sampled_convergence.json` (iters=10000)
- launch (one seed shown; others differ only in `--seed`): `python3 run/run_different_ability.py --method gradient --q 35 --seed 42 --grad-epsilon 0.5 --grad-samples 4096 --grad-lr 2.0 --grad-lr-decay 0.9999 --grad-steps 10000 --grad-tol 1e-6 --ablation-name r5_sampled --force`

### Het. Ability | q=55 | Gradient  (tag `r5_sampled`, seeds [42, 43, 44, 45, 46])
- `results/different_ability/convergence/different_ability_gradient_q55.0_seed42_r5_sampled_convergence.json` (iters=10000)
- `results/different_ability/convergence/different_ability_gradient_q55.0_seed43_r5_sampled_convergence.json` (iters=10000)
- `results/different_ability/convergence/different_ability_gradient_q55.0_seed44_r5_sampled_convergence.json` (iters=10000)
- `results/different_ability/convergence/different_ability_gradient_q55.0_seed45_r5_sampled_convergence.json` (iters=10000)
- `results/different_ability/convergence/different_ability_gradient_q55.0_seed46_r5_sampled_convergence.json` (iters=10000)
- launch (one seed shown; others differ only in `--seed`): `python3 run/run_different_ability.py --method gradient --q 55 --seed 42 --grad-epsilon 0.5 --grad-samples 4096 --grad-lr 2.0 --grad-lr-decay 0.9999 --grad-steps 10000 --grad-tol 1e-6 --ablation-name r5_sampled --force`

### Two-Player Set 2 (wh8_wl4) | TEL-PPO + Gradient (Fig. 2 prize variant)
- TEL-PPO: 15 runs (q in {35,45,55} x seeds 42-46), files `results/two_players/convergence/*wh8_wl4_r5_sampled*_convergence.json`
- Gradient: 15 runs (q in {35,45,55} x seeds 42-46), files `results/two_players/convergence/*r5_sampled_wh8_wl4*_convergence.json`
- PPO launch form: `python3 run/run_two_players.py --method ppo --q 35 --seed 42 --k 0.0006 --w_h 8 --w_l 4 --variant-name wh8_wl4 --override-conc-ramp-warmup 200 --episodes 6144000 --ablation-name r5_sampled`
- gradient launch form: `python3 run/run_two_players.py --method gradient --q 35 --seed 42 --k 0.0006 --w_h 8 --w_l 4 --grad-epsilon 0.5 --grad-samples 4096 --grad-lr 2.0 --grad-lr-decay 0.9999 --grad-steps 10000 --grad-tol 1e-6 --ablation-name r5_sampled_wh8_wl4 --force`

### Fig-7 ablation arms (ablation_results.csv; 2P Set 1, q in {35,45,55}, seeds 42-46)
- TEL-PPO (full): = the Two-Player Set 1 TEL-PPO runs above (reused, no copies)
- No stability screening: `results/two_players/convergence/ppo_q{Q}.0_seed{S}_r5_fig7_no_stability_convergence.json` (15)
- No exploitability verification: `..._r5_fig7_no_exploit_convergence.json` (15; all stop_reason=max_updates@1500 by design)
- collection + arm summary: `results/ablation/r5_fig7/MANIFEST.md`

## 2. Figures -> data -> runs

All figures are produced by `python -m paper.generator make_all` (paper/generator/plots.py),
each writing its intermediate data to `paper/data/<name>.csv`. Run selection for every figure
is the promoted baseline described in section 1 (BASELINE_OVERRIDES in
paper/generator/config.py), so each figure traces to the same r5 files as its scenario's
table cells.

| Figure (paper/figures/) | Data (paper/data/) | Source runs |
|---|---|---|
| convergence_main | convergence_main.csv | baseline TEL-PPO + Gradient trajectories, all scenarios (section 1) |
| kl_dynamics | kl_dynamics.csv | 2P Set 1 TEL-PPO (approx_kl series) |
| exploitability_dynamics | exploitability_dynamics.csv | 2P Set 1 TEL-PPO (sparse exploit series + eval markers) |
| beta_evolution | beta_evolution.csv | 2P Set 1 TEL-PPO (alpha/beta series) |
| effort_drift | effort_drift.csv | 2P Set 1 TEL-PPO (drift series) |
| distance_to_equilibrium | distance_to_equilibrium.csv | baseline TEL-PPO |e-e*| trajectories |
| equilibrium_recovery_dotplot | equilibrium_recovery_dotplot.csv | per-seed final efforts, all scenarios |
| ablation_comparison | ablation_comparison.csv | 2P baseline arm (the Fig-7 TABLE carries all three arms; see section 1) |
| hyperparam_sensitivity | hyperparam_sensitivity.csv | 2P baseline + ε/patience verification sweeps (r5_sensitivity wave); see section 4 |

## 3. MC-FD gradient solver parameters (all gradient cells)

`--grad-epsilon 0.5 --grad-samples 4096 --grad-lr 2.0 --grad-lr-decay 0.9999
--grad-steps 10000 --grad-tol 1e-6` (sampled payoffs, common random numbers, central
differences, projected simultaneous ascent). Chosen because the legacy config defaults
(delta=0.01, 64 samples) were tuned for the removed closed-form solver and give FD-gradient
SNR << 1 under sampling; correction history in `results/r5_sampled/MANIFEST.md` and
`manifest.csv`. CRN couples the two 2P players' trajectories, so the 2P gradient symmetry
gap is ~0 by construction; 3P/dc gradient symmetry gaps are genuine measurements.

## 4. UNVERIFIED / caveated artifacts

1. **hyperparam_sensitivity.{pdf,png,csv} - VERIFIED (refilled 2026-06-18).** Baseline curve =
   promoted r5_sampled runs; the ε row (eps_001/003/010/020 = `--exploit-eps`
   0.01/0.03/0.10/0.20) and patience row (pat_01/03/10 = `--exploit-patience` 1/3/10) are the
   r5_sensitivity wave: 105 two-player PPO runs, q∈{35,45,55} × seeds 42-46, sampled training,
   `--override-conc-ramp-warmup 200`, all `stop_reason=exploitability` (data commit 5922b5d).
   eps_003 (ε=0.03) duplicates the canonical gate as a consistency arm. The CSV also carries
   the r5_fig7 arms (picked up by the registry glob; not in the figure's plot_order, so
   unplotted). Supersedes the prior degenerate closed-form-era placeholder.
2. **beta_snapshots / exploitability_q25 - RETIRED (2026-06-12).** Both code paths were
   dormant under canonical data (no alpha/beta snapshot series; q=25 dropped in the parameter
   overhaul) and have been removed from plots.py and the make_all pipeline. No files existed.
3. **convergence_comparison.{csv,tex} - RETIRED (2026-06-12).** It pooled scenarios per q
   (cross-experiment averaging), which is semantically misleading; the function, wiring, and
   generated files are removed. Per-scenario convergence lives in final_summary's
   "Conv. Update (verified)" column.
4. **environment_config.{csv,tex}** - static, traces to config/*.py (no runs).

## 5. Notes for manuscript section 5.3 (eps_eq gate semantics; owner-held framing decision)

Mechanism found in the r5 three-player results and held for the manuscript text (not a code
change): TEL-PPO 3P verifies at exploitability 0.004-0.020 (< eps_eq=0.03) while the effort
sits 6.6-8.0% from e* (22.99 vs 25.00 at q=35; 15.31 vs 15.91 at q=55). The two statements
are consistent because the verification criterion is denominated in UTILITY units while
recovery error is denominated in EFFORT units, and the 3P expected-utility landscape is
nearly flat around e*: with w_H - w_L = 3.5 and q = 35, a ~2-effort-unit unilateral
deviation changes expected utility by < 0.03. So the exploitability gate certifies
"no profitable deviation > 0.03 utils" - an approximate-Nash guarantee - well before it
certifies effort-level precision. The same effect, milder, appears in dc (verifies at
0.002 with 2.9-4.7% effort error) and da (0.026-0.029 at 3.3-5.3%). Options the text can
take: report eps_eq explicitly as a utility-scale tolerance alongside effort-scale error
(honest two-metric framing), and/or relate eps_eq to an effort-band via the local curvature
|U''| ~ 2k + (w_H-w_L)/(4q^2) per scenario.

## 6. RESOLVED (2026-06-18): ablation leak into convergence_main + ablation_results (fig7 / eps-pat)

Found 2026-06-18 while refilling hyperparam_sensitivity. Three builders select the "baseline"
result by `weight_variant` (or no filter) WITHOUT also constraining `ablation == "baseline"`,
so every two_players PPO run tagged `weight_variant=baseline` but a non-baseline ablation —
the dormant `r5_fig7_no_exploit`/`r5_fig7_no_stability` arms, and now the r5_sensitivity
`eps_*`/`pat_*` sweep — leaks into the baseline result:

- **plots.py `plot_convergence_main`** (per-weight-variant loop ~L181): mixes ablations into
  the top-row (Set-1) baseline curve + CI band. PROVEN already fig7-contaminated in the
  CURRENTLY-COMMITTED figure — baseline-only (fixed) bands are tighter than committed at every
  q: top-row effort floor q35 33.2 vs 30.9, q45 34.3 vs 29.6, q55 28.9 vs 24.8; committed
  overlays 3 mean-curves (baseline + 2 fig7), the fix draws 1. eps/pat would push it to 10.
- **tables.py `generate_summary_metrics_table`** (L79, no weight/ablation filter): emits one
  per-run row per ablation → +105 eps/pat rows on regen.
- **tables.py `generate_ablation_table`** (L192, `weight_variant=="baseline"` only): admits
  eps/pat as if they were mechanism ablations → +7 rows on regen.

NOT affected (verified): the other 8 figures and `final_summary` all constrain
`ablation=="baseline"` (final_summary also `weight_variant=="baseline"`); environment_config
is static; `aggregate_seeds`/`get_convergence_step`/metrics grouping all key on ablation.
Headline rel-err numbers trace to final_summary (e.g. 2P q35 TEL-PPO 43.58±1.25, 4.12%) — safe.

RESOLUTION (2026-06-18, owner-approved): fix applied to both code paths.
- `plot_convergence_main`: now constrains `ablation=="baseline"` within each weight-variant
  row. This is a fig7-LEAK CORRECTION, **NOT a data change** — same underlying r5 runs; the
  baseline panel simply stops averaging in the non-baseline arms. Effect on convergence_main
  (PNG md5 `a766b0f9` committed → `a884d748` fixed): the top-row x-axis collapses from the
  never-terminating `r5_fig7_no_exploit` arm (6.14M steps) back to the true baseline scale —
  q35 6.14M→279k (22×), q45 6.14M→319k (19×), q55 6.14M→442k (14×); shared y-floor 20.1→24.6
  as fig7's low drift leaves; top-row curves 3→1, CI bands tighten. Set-2 (bottom) row is
  byte-identical (only `ablation==baseline` runs exist there).
- `generate_ablation_table`: drops `eps_*`/`pat_*` sweeps → stays the curated mechanism table
  (TEL-PPO / No stability screening / No exploitability verification); regenerates identical to
  the prior committed file.
- `generate_summary_metrics_table`: left as ALL-RUNS by decision (eps/pat included, 200→305
  rows) — generator hygiene only, no manuscript impact.
MANUSCRIPT: the only contaminated artifact in the compiled paper was the Fig 2 image
(`convergence_main.pdf`); the corrected render was re-exported into `overleaf_export/figures/`.
No manuscript `.tex` was touched. SEPARATE / OUT OF SCOPE: Tables 3/4 carry stale hand-entered
numbers (e.g. 44.10/4.30%, pre-r5) — unrelated to this leak; owner handles the r5 number-swap
later.
