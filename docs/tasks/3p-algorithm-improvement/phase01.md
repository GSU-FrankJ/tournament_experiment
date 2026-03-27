# Phase 01: Implement and Test Pairwise Advantage Decomposition

## Objective

Implement Proposal 1 (pairwise reward decomposition) and test whether it reduces the 3-player convergence gap from ~5.3 to <2.0.

## Steps

### Step 1: Add reward mode to environment

File: `envs/three_players_env.py`

Add `reward_mode` parameter. When `reward_mode="pairwise"`:
- Compute 2-player win probs: `p_ij = p_from_diff(e_i - e_j, q)` for each pair
- Player i's reward: `w_L + 0.5*(p_ij + p_ik) * (w_H - w_L) - k * e_i^2`

Import `p_from_diff` from `utils/prob.py`.

### Step 2: Add CLI flag to runner

File: `run/run_three_players.py`

Add `--reward-mode {expected, pairwise}` flag (default: `expected` to preserve existing behavior).
Pass to environment constructor.

### Step 3: Verify equilibrium preservation

Run gradient solver with pairwise rewards:
```bash
python run/run_three_players.py --method gradient --q 35 --reward-mode pairwise
```
Must converge to gap < 0.5.

### Step 4: Run PPO experiment

```bash
tmux new-session -d -s 3p_pairwise \
  "python run/run_three_players.py --method ppo --q 35 --seed 42 \
   --reward-mode pairwise --episodes 6144000"
```

### Step 5: Evaluate

If gap < 2.0 and exploitability < 0.05:
- Run 4 more seeds (123, 456, 789, 1024)
- If 4/5 seeds converge: SUCCESS
- Update STATE.md

If gap >= 2.0: proceed to Proposal 3 (binary with control variate) or Proposal 2 (counterfactual baseline).

## Success criteria

- gap < 2.0 from e*(q=35) = 62.5
- exploitability < 0.05
- 4/5 seeds converge
