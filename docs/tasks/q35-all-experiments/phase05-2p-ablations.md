# Phase 05: 2-Player q=35 Ablation Experiments

## Objective

Run the two missing ablation conditions (`no_cheap_gate`, `no_exploitability`) for
q=35 two-player, so the `ablation_comparison` figure has all three panels populated
with green and orange lines (currently q=35 panel shows only the blue baseline).

## Background

Ablation experiments exist for q=25, q=40, q=55 (3 seeds each) but were never run
for q=35, which was added later. The paper figure `ablation_comparison.png` has a
visible gap in the q=35 panel.

## Runs (6 total)

All runs use `--episodes 2048000` to match existing ablation runs at other q values.

### no_cheap_gate (3 seeds)

```bash
tmux new-session -d -s abl35_ncg42 \
  "python run/run_two_players.py --method ppo --q 35 --seed 42 --episodes 2048000 \
   --disable-cheap-gate --ablation-name no_cheap_gate"

tmux new-session -d -s abl35_ncg123 \
  "python run/run_two_players.py --method ppo --q 35 --seed 123 --episodes 2048000 \
   --disable-cheap-gate --ablation-name no_cheap_gate"

tmux new-session -d -s abl35_ncg456 \
  "python run/run_two_players.py --method ppo --q 35 --seed 456 --episodes 2048000 \
   --disable-cheap-gate --ablation-name no_cheap_gate"
```

### no_exploitability (3 seeds)

```bash
tmux new-session -d -s abl35_nex42 \
  "python run/run_two_players.py --method ppo --q 35 --seed 42 --episodes 2048000 \
   --disable-exploitability --ablation-name no_exploitability"

tmux new-session -d -s abl35_nex123 \
  "python run/run_two_players.py --method ppo --q 35 --seed 123 --episodes 2048000 \
   --disable-exploitability --ablation-name no_exploitability"

tmux new-session -d -s abl35_nex456 \
  "python run/run_two_players.py --method ppo --q 35 --seed 456 --episodes 2048000 \
   --disable-exploitability --ablation-name no_exploitability"
```

## Expected output files

```
results/two_players/convergence/ppo_q35.0_seed42_no_cheap_gate_convergence.json
results/two_players/convergence/ppo_q35.0_seed42_no_cheap_gate_metadata.json
results/two_players/convergence/ppo_q35.0_seed123_no_cheap_gate_convergence.json
results/two_players/convergence/ppo_q35.0_seed123_no_cheap_gate_metadata.json
results/two_players/convergence/ppo_q35.0_seed456_no_cheap_gate_convergence.json
results/two_players/convergence/ppo_q35.0_seed456_no_cheap_gate_metadata.json
results/two_players/convergence/ppo_q35.0_seed42_no_exploitability_convergence.json
results/two_players/convergence/ppo_q35.0_seed42_no_exploitability_metadata.json
results/two_players/convergence/ppo_q35.0_seed123_no_exploitability_convergence.json
results/two_players/convergence/ppo_q35.0_seed123_no_exploitability_metadata.json
results/two_players/convergence/ppo_q35.0_seed456_no_exploitability_convergence.json
results/two_players/convergence/ppo_q35.0_seed456_no_exploitability_metadata.json
```

## Verification

1. All 12 output files exist
2. `python -m paper.generator make_all` regenerates `ablation_comparison.png`
3. q=35 panel now shows green (`no_exploitability`) and orange (`no_cheap_gate`) lines
4. Final efforts should be near e*(35) = 62.5 (theory-valid regime for 2-player)
