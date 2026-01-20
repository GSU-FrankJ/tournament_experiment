# Complete Command-Line Guide: VS_OPPONENT Mode Training

**Last Updated**: 2025-12-18  
**For**: One-Stage Two-Player PPO Experiment  
**Mode**: `vs_opponent` (learner vs lagged opponent as environment)

---

## Quick Reference: Best Default Command Template

```bash
python run/run_two_players.py \
    --method ppo \
    --rollout-mode vs_opponent \
    --episodes <TRAINING_BUDGET> \
    --q <NOISE_LEVEL> \
    [optional overrides...]
```

**Placeholders you MUST choose**:
- `<TRAINING_BUDGET>`: Total environment steps (e.g., 2048000 for full run, 20000 for quick test)
- `<NOISE_LEVEL>`: Noise parameter q (typically one of: 25.0, 40.0, 55.0)

---

## All Configurable Parameters (Grouped by Category)

### 🎯 REQUIRED PARAMETERS

#### 1. Training Method
```bash
--method ppo
```
- **Location**: `run/run_two_players.py:800`
- **Choices**: `["gradient", "ppo"]`
- **Default**: `"gradient"` (must override to `ppo`)
- **Why Required**: Must explicitly choose PPO method (default is gradient solver)

#### 2. Rollout Mode
```bash
--rollout-mode vs_opponent
```
- **Location**: `run/run_two_players.py:802-806`
- **Choices**: `["selfplay", "vs_opponent"]`
- **Default**: `"vs_opponent"`
- **Note**: Actually defaults to vs_opponent, but explicit is clearer
- **VS_OPPONENT Semantics**: Player1 always uses learner; Player2 may use lagged opponent based on lag schedule; only learner-generated transitions stored for PPO updates
- **Why It Matters**: Controls whether opponent lag mechanism is used and how data is stored

---

### 📊 CORE EXPERIMENTAL PARAMETERS (Highly Recommended to Set)

#### 3. Training Budget
```bash
--episodes <NUM_ENV_STEPS>
```
- **Location**: `run/run_two_players.py:809-813`
- **Type**: `int`
- **Default**: `1_800_000` (code default; config has `2_048_000`)
- **Source**: `config/one_stage_two_players.py:42`
- **Recommended Values**:
  - Quick sanity: `20_000` (5 updates at 4096 steps/update)
  - Short test: `100_000` (~24 updates)
  - Full experiment: `2_048_000` (500 updates) ← Config default
  - Extended run: `4_000_000` (~976 updates)
- **Why It Matters**: Total environment steps; effective PPO batch size = 1.5-2× steps_per_update depending on lag schedule in vs_opponent mode
- **Note**: `steps_per_update=4096` is hardcoded in PPOConfig, so 2_048_000 episodes = 500 PPO updates

#### 4. Noise Parameter
```bash
--q <NOISE_LEVEL>
```
- **Location**: `run/run_two_players.py:807`
- **Type**: `float`
- **Default**: Trains on all in `config["q_list"]` = `[25.0, 40.0, 55.0]` (if omitted)
- **Source**: `config/one_stage_two_players.py:12`
- **Recommended Values**:
  - Low noise: `25.0` (easier to learn, higher effort)
  - Medium noise: `40.0` (balanced)
  - High noise: `55.0` (harder to learn, lower effort)
- **Why It Matters**: Controls observation noise; affects theoretical optimal effort and learning difficulty
- **Note**: If omitted, trains on all q values in q_list simultaneously (multi-task)

---

### 🎮 GAME ENVIRONMENT PARAMETERS (Optional Overrides)

#### 5. Cost Parameter
```bash
--k <COST_COEFFICIENT>
```
- **Location**: `run/run_two_players.py:825`
- **Type**: `float`
- **Default**: `0.0004`
- **Source**: `config/one_stage_two_players.py:8`
- **Typical Range**: `0.0001` to `0.001`
- **Why It Matters**: Cost coefficient in utility function (cost = k × effort²); higher k → lower optimal efforts
- **When to Override**: Exploring different cost structures or reproducing specific experimental conditions

#### 6. Prize Values
```bash
--w_h <HIGH_PRIZE>
--w_l <LOW_PRIZE>
```
- **Location**: `run/run_two_players.py:826-827`
- **Type**: `float`
- **Defaults**: `w_h=6.5`, `w_l=3.0`
- **Source**: `config/one_stage_two_players.py:13-14`
- **Typical Values**: `w_h` in [5.0, 10.0], `w_l` in [2.0, 5.0]
- **Why It Matters**: Prize spread (w_h - w_l) affects effort incentives; larger spread → higher efforts
- **When to Override**: Testing different tournament structures

#### 7. Effort Bounds
```bash
--effort-range <LOW> <HIGH>
```
- **Location**: `run/run_two_players.py:828`
- **Type**: Two `float` values
- **Default**: `[0, 200]`
- **Source**: `config/one_stage_two_players.py:19`
- **Recommended Values**: 
  - Standard: `0 200`
  - Constrained: `0 100` (forces low-effort strategies)
  - Relaxed: `0 300` (allows higher efforts)
- **Why It Matters**: Bounds effort space for Beta policy; affects action normalization and exploration
- **When to Override**: Testing bounded rationality or different strategy spaces

#### 8. Random Seed
```bash
--seed <SEED_VALUE>
```
- **Location**: `run/run_two_players.py:829`
- **Type**: `int`
- **Default**: `42`
- **Source**: `config/one_stage_two_players.py:20`
- **Recommended Values**: Any integer (42, 123, 2025, etc.)
- **Why It Matters**: Controls RNG for reproducibility; set different seeds for replicate runs
- **When to Override**: Running multiple replicates or debugging specific behaviors

---

### 🔍 EVALUATION FLAGS (Optional, Control Post-Training Evaluation)

#### 9. Symmetric Self-Play Evaluation
```bash
--eval-symmetric          # Enable (default)
--no-eval-symmetric       # Disable
```
- **Location**: `run/run_two_players.py:822-824`
- **Default**: `True` (enabled)
- **Why It Matters**: Evaluates learned policy against itself (both players use learner)
- **VS_OPPONENT Note**: This is the "true" Nash evaluation after training; during training, player2 sometimes used opponent
- **When to Disable**: Only care about training metrics, not final symmetric evaluation

#### 10. Evaluate vs Opponent Policy
```bash
--eval-vs-opponent
```
- **Location**: `run/run_two_players.py:820`
- **Default**: `False` (disabled)
- **Why It Matters**: Evaluates learned policy against the final lagged opponent policy
- **VS_OPPONENT Note**: Shows how learner performs against the last opponent snapshot
- **When to Enable**: Analyzing asymmetric performance or opponent exploitation

#### 11. Evaluate vs Opponent History
```bash
--eval-vs-history
```
- **Location**: `run/run_two_players.py:821`
- **Default**: `False` (disabled)
- **Why It Matters**: Evaluates against all stored opponent snapshots, reports statistics
- **VS_OPPONENT Note**: Only relevant if `opponent_snapshot_keep > 0` in config
- **When to Enable**: Analyzing robustness to different opponent versions

---

### 🚫 GRADIENT-ONLY FLAGS (Not Used in PPO, Listed for Completeness)

These flags only apply when `--method gradient`:
- `--grad-lr`: Gradient descent learning rate (default: 0.08)
- `--grad-steps`: Max gradient iterations (default: 1500)
- `--grad-epsilon`: Finite-difference epsilon (default: 0.5)
- `--grad-tol`: Convergence tolerance (default: 1e-4)
- `--grad-samples`: Monte Carlo samples per gradient (default: 64)
- `--grad-init-perturb`: Initial asymmetry (default: 1.0)

**Location**: `run/run_two_players.py:814-819`  
**Note**: Ignored when `--method ppo`

---

### ⚙️ HARDWARE/INFRASTRUCTURE (Auto-Detected, No CLI Flags)

#### Device Selection
- **Code**: `agents/ppo_two_players_clean.py:77`
- **Behavior**: Auto-detects CUDA if available, else CPU
- **Logic**: `device = device or ("cuda" if torch.cuda.is_available() else "cpu")`
- **No CLI Flag**: Cannot override via command line (hardcoded auto-detection)
- **How to Force CPU**: Set environment variable: `CUDA_VISIBLE_DEVICES=""`
- **How to Force Specific GPU**: Set environment variable: `CUDA_VISIBLE_DEVICES="0"` (or "1", "2", etc.)

#### Logging
- **Auto-Generated**: All runs automatically log to `results/logs/one_stage_two_players_<method>_q<value>_ep<episodes>_<timestamp>.log`
- **Code**: `run/run_two_players.py:832-836`
- **No CLI Flag**: Logging is always enabled and mirrored to console

#### Results Output
- **Auto-Appended**: Results saved to `results/one_stage_two_players.csv`
- **Code**: `run/run_two_players.py:794`
- **No CLI Flag**: Output path is hardcoded

---

## Configuration Defaults Not Exposed via CLI

These are set in `config/one_stage_two_players.py` and used internally (cannot override via CLI):

**PPO Hyperparameters** (Source: `config/one_stage_two_players.py:38-58`):
- `steps_per_update`: 4096 (env steps per PPO update)
- `minibatch_size`: 1024 (PPO minibatch size)
- `update_epochs`: 6 (epochs per PPO update)
- `entropy_coef_start`: 0.03 (initial entropy coefficient)
- `entropy_coef_end`: 0.015 (final entropy coefficient)
- `lr_start`: 3e-4 (initial learning rate)
- `lr_end`: 2e-4 (final learning rate)
- `clip_range_start`: 0.50 (initial PPO clip range)
- `clip_range_end`: 0.35 (final PPO clip range)
- `target_kl`: 0.08 (target KL for adaptive scaling)

**Opponent Lag Settings** (Source: `config/one_stage_two_players.py:30-36`):
- `opponent_mode`: "periodic" (sync opponent every N updates)
- `opponent_sync_interval`: 2 (sync every 2 PPO updates)
- `opponent_ema_tau`: 0.20 (EMA coefficient, not used in periodic mode)
- `opponent_snapshot_keep`: 10 (keep 10 historical snapshots)
- `opponent_history_sample_p`: 0.0 (don't sample from history)
- `lag_warmup_updates`: 10 (use opponent for first 10 updates)
- `lag_fade_updates`: 10 (fade opponent usage over next 10 updates)

**Early Stopping** (Source: `config/one_stage_two_players.py:44-46`):
- `eval_every_updates`: 20 (evaluate every 20 updates)
- `early_stop_abs_err`: 0.8 (stop if error < 0.8)
- `early_stop_patience`: 6 (require 6 consecutive low-error evals)

**Note**: To modify these, edit `config/one_stage_two_players.py` directly.

---

## Ready-to-Run Examples

### Example A: Quick Sanity Check (Fast, ~2 minutes)

```bash
python run/run_two_players.py \
    --method ppo \
    --rollout-mode vs_opponent \
    --episodes 20000 \
    --q 25.0 \
    --seed 42
```

**What This Does**:
- Trains PPO in vs_opponent mode for 20,000 env steps (~5 PPO updates)
- Uses q=25.0 (low noise, easier to learn)
- Expected runtime: ~2 minutes on CPU, ~30 seconds on GPU
- Purpose: Verify installation, check for errors, see basic training loop

**Expected Output**:
```
[PPO] Rollout mode: VS_OPPONENT
[Update 1] q=25.0: e*=87.50, policy=..., gap=..., lag_prob=1.00, ...
[Update 2] q=25.0: e*=87.50, policy=..., gap=..., lag_prob=1.00, ...
...
Saved results to results/one_stage_two_players.csv
```

---

### Example B: Full Experiment (Production Run, ~1-2 hours)

```bash
python run/run_two_players.py \
    --method ppo \
    --rollout-mode vs_opponent \
    --episodes 2048000 \
    --q 40.0 \
    --seed 42 \
    --eval-vs-opponent
```

**What This Does**:
- Full training run: 2,048,000 env steps (500 PPO updates)
- Uses q=40.0 (medium noise, balanced difficulty)
- Includes opponent evaluation at the end
- Expected runtime: ~1-2 hours on CPU, ~20-40 minutes on GPU
- Purpose: Production-quality experiment for publication/analysis

**Expected Output**:
```
[PPO] Rollout mode: VS_OPPONENT
[Storage Stats] Update 20: stored_p1=4096, stored_p2=2047, skipped_p2=2049, effective_batch=6143 | Total: ...
[Storage Stats] Update 40: ...
...
[Update 500] q=40.0: e*=65.63, policy=65.12, gap=0.51, ...
Saved results to results/one_stage_two_players.csv
```

**Check Results**:
```bash
# View final performance
tail -1 results/one_stage_two_players.csv | tr ',' '\n' | grep -E "(rollout_mode|gap|effective_batch)"

# Expected to see:
# rollout_mode: vs_opponent
# stage2_gap_unweighted: <0.5-2.0 (good convergence)
# effective_batch_size_total: ~1,500,000 (1.5× episodes due to opponent skips)
```

---

### Example C: Multi-Q Training (Advanced, ~3-5 hours)

```bash
python run/run_two_players.py \
    --method ppo \
    --rollout-mode vs_opponent \
    --episodes 2048000 \
    --seed 42
```

**What This Does**:
- Trains on ALL q values in config simultaneously (q=25, 40, 55)
- Network learns to generalize across noise levels
- Expected runtime: ~3-5 hours on CPU, ~1-2 hours on GPU
- Purpose: Multi-task learning, robust policy

**Note**: Omitting `--q` triggers multi-q training (default behavior).

---

### Example D: Ablation Study (Compare Modes)

```bash
# Run 1: VS_OPPONENT mode
python run/run_two_players.py \
    --method ppo \
    --rollout-mode vs_opponent \
    --episodes 2048000 \
    --q 40.0 \
    --seed 42

# Run 2: SELFPLAY mode (for comparison)
python run/run_two_players.py \
    --method ppo \
    --rollout-mode selfplay \
    --episodes 2048000 \
    --q 40.0 \
    --seed 42
```

**Purpose**: Compare learning curves and final performance between modes.

**Important**: Note that effective_batch_size differs:
- Selfplay: ~4,096,000 samples (2× episodes)
- VS_OPPONENT: ~3,000,000 samples (1.5× episodes due to lag)
- Normalize by `effective_batch_size_total` when comparing metrics

---

## Decision Checklist (Priority Order)

Before running, decide on:

### 🔴 HIGH PRIORITY (Must Decide)
- [ ] **Training budget**: Quick test (20k) or full run (2M+)?
- [ ] **Noise level (q)**: Single value (25/40/55) or multi-q (omit flag)?
- [ ] **Rollout mode**: `vs_opponent` (with lag) or `selfplay` (pure symmetric)?

### 🟡 MEDIUM PRIORITY (Recommended to Set)
- [ ] **Seed**: Default (42) or custom for replicates?
- [ ] **Evaluation**: Just symmetric (default) or also vs_opponent/history?

### 🟢 LOW PRIORITY (Usually Keep Defaults)
- [ ] **Game parameters**: Override k, w_h, w_l, effort_range?
- [ ] **Device**: Use auto-detection or force CPU/GPU via env var?

### ⚪ OPTIONAL (Advanced Use Only)
- [ ] **Config modifications**: Edit `config/one_stage_two_players.py` for PPO hyperparameters?
- [ ] **Custom effort bounds**: Test bounded rationality scenarios?

---

## Common Pitfalls & Troubleshooting

### ❌ Pitfall 1: Forgot `--method ppo`
```bash
# WRONG (will run gradient solver):
python run/run_two_players.py --rollout-mode vs_opponent --episodes 2048000

# CORRECT:
python run/run_two_players.py --method ppo --rollout-mode vs_opponent --episodes 2048000
```

### ❌ Pitfall 2: Comparing modes without normalizing batch size
```bash
# Both runs have same episodes, but different effective batch sizes!
# Selfplay: 2× episodes stored
# VS_OPPONENT: ~1.5× episodes stored

# Solution: Check CSV field "effective_batch_size_total" and normalize metrics
```

### ❌ Pitfall 3: Not checking device utilization
```bash
# Check if GPU is being used:
nvidia-smi  # Should show python process using GPU memory

# If not using GPU when available:
# 1. Check PyTorch installation: python -c "import torch; print(torch.cuda.is_available())"
# 2. Check CUDA drivers
# 3. Verify GPU not blocked by other process
```

### ❌ Pitfall 4: Interpreting lag_prob during training
```
# Early training (updates 1-10): lag_prob=1.0 (always use opponent)
#   → ~50% of samples skipped (only P1 stored)
# Fade phase (updates 11-20): lag_prob decreases 1.0 → 0.0
# Late training (updates 21+): lag_prob=0.0 (both use learner, like selfplay)
#   → All samples stored (both P1 and P2)

# This is CORRECT behavior in vs_opponent mode!
```

---

## Where to Find More Information

### Code References
- **Main entrypoint**: `run/run_two_players.py:798-840` (argparse + main)
- **Config file**: `config/one_stage_two_players.py` (all defaults)
- **Agent implementation**: `agents/ppo_two_players_clean.py:73-108` (PPO init)
- **Rollout logic**: `run/run_two_players.py:465-520` (action generation + storage)

### Documentation
- **Rollout modes guide**: `docs/rollout_modes_ablation.md` (comprehensive)
- **Audit report**: `docs/AUDIT_REPORT_rollout_modes.md` (verification)
- **Implementation status**: `docs/IMPLEMENTATION_COMPLETE.md` (summary)

### Verification
```bash
# Run audit to verify modes work correctly:
python tools/audit_rollout_modes.py

# Run short verification test:
python tools/verify_rollout_modes.py
```

---

## Quick Command Builder

Copy and modify this template:

```bash
python run/run_two_players.py \
    --method ppo \
    --rollout-mode vs_opponent \
    --episodes 2048000 \
    --q 40.0 \
    --seed 42 \
    # [Add optional flags below]
    # --k 0.0005 \
    # --w_h 7.0 \
    # --w_l 2.5 \
    # --effort-range 0 150 \
    # --eval-vs-opponent \
    # --eval-vs-history
```

**Uncomment lines as needed for your experiment.**

---

**Last Updated**: 2025-12-18  
**Verified Against**: Code commit with rollout modes refactor + audit


