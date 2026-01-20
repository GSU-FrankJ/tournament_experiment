# Rollout Modes Refactor - Changes Summary

## Quick Reference

**Files Modified**: 1 primary file + 1 new verification tool + 1 documentation
**Lines Changed**: ~150 lines modified/added in `run/run_two_players.py`
**New Features**: Explicit rollout mode control with `--rollout-mode` flag
**Bug Fixed**: Opponent samples no longer mixed into learner's PPO buffer

---

## Modified Files

### 1. `run/run_two_players.py` (Primary Changes)

#### Change 1: Add rollout_mode parameter to run_ppo()
**Location**: Lines 248-268
```python
def run_ppo(
    cfg: Dict,
    episodes: Optional[int] = None,
    train_qs: Optional[List[float]] = None,
    eval_qs: Optional[List[float]] = None,
    *,
    rollout_mode: str = "vs_opponent",  # NEW PARAMETER
    eval_symmetric: bool = True,
    eval_vs_opponent: bool = False,
    eval_vs_history: bool = False,
) -> List[Dict]:
    """Train PPO via self-play with conditioning on (q, k, w_gap).
    
    - rollout_mode controls action generation and storage:
        * "selfplay": Both players always use learner policy; store both transitions
        * "vs_opponent": Player1 uses learner; Player2 may use opponent (with lag schedule);
                         store only learner-generated transitions
    """
    # Validate rollout_mode
    if rollout_mode not in ("selfplay", "vs_opponent"):
        raise ValueError(f"rollout_mode must be 'selfplay' or 'vs_opponent', got '{rollout_mode}'")
```

#### Change 2: Add mode announcement logging
**Location**: Lines 304-313
```python
    # Print rollout mode for clarity
    print(f"[PPO] Rollout mode: {rollout_mode.upper()}")
    if rollout_mode == "selfplay":
        print("[PPO]   - Both players use learner policy")
        print("[PPO]   - Store both transitions every step")
    else:  # vs_opponent
        print("[PPO]   - Player1 uses learner; Player2 may use opponent (lag schedule)")
        print("[PPO]   - Store only learner-generated transitions")
    print(flush=True)
```

#### Change 3: Initialize storage counters
**Location**: Lines 380-383
```python
    # Storage counters for debugging/verification
    stored_p1_total = 0
    stored_p2_total = 0
    skipped_p2_due_to_opponent_total = 0
```

#### Change 4: Refactor rollout loop (CRITICAL)
**Location**: Lines 445-497 (replaces old lines 437-459)

**OLD CODE** (buggy):
```python
        for _ in range(steps_this):
            q = float(rng.choice(train_qs))
            env.q = q
            
            s1 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
            s2 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
            a1_norm, e1, logp1, v1 = agent.act(s1)
            
            use_opponent = (lag_prob > 0.0) and (rng.random() < lag_prob)
            if use_opponent:
                a2_norm, e2, logp2, _ = agent.act_opponent(s2)
                v2 = agent.value_only(s2)
            else:
                a2_norm, e2, logp2, v2 = agent.act(s2)
            
            _, rewards, _, done, _ = env.step((torch.tensor([float(e1.item())]), torch.tensor([float(e2.item())])))
            
            # BUG: Always store both, even when p2 used opponent!
            agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
            agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
            history.append(float((e1.item() + e2.item()) / 2.0))
```

**NEW CODE** (fixed):
```python
        # Per-update storage counters
        stored_p1_this_update = 0
        stored_p2_this_update = 0
        skipped_p2_this_update = 0
        
        for _ in range(steps_this):
            q = float(rng.choice(train_qs))
            env.q = q
            
            s1 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
            s2 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
            
            # Player 1: ALWAYS uses learner policy (both modes)
            a1_norm, e1, logp1, v1 = agent.act(s1)
            
            # Player 2: Mode-dependent action generation
            if rollout_mode == "selfplay":
                # SELFPLAY MODE: Player2 always uses learner policy
                a2_norm, e2, logp2, v2 = agent.act(s2)
                use_opponent = False
            else:  # rollout_mode == "vs_opponent"
                # VS_OPPONENT MODE: Player2 may use opponent based on lag schedule
                use_opponent = (lag_prob > 0.0) and (rng.random() < lag_prob)
                if use_opponent:
                    a2_norm, e2, logp2, _ = agent.act_opponent(s2)
                    v2 = agent.value_only(s2)
                else:
                    a2_norm, e2, logp2, v2 = agent.act(s2)
            
            _, rewards, _, done, _ = env.step((torch.tensor([float(e1.item())]), torch.tensor([float(e2.item())])))
            
            # Storage: Mode-dependent logic (FIX!)
            # Player 1: ALWAYS store
            agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
            stored_p1_this_update += 1
            
            # Player 2: Mode-dependent storage
            if rollout_mode == "selfplay":
                agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
                stored_p2_this_update += 1
            else:  # vs_opponent
                if not use_opponent:
                    agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
                    stored_p2_this_update += 1
                else:
                    skipped_p2_this_update += 1
            
            history.append(float((e1.item() + e2.item()) / 2.0))
```

**Key Differences**:
1. ✅ Added per-update counters
2. ✅ Mode-specific Player 2 action generation
3. ✅ **Conditional storage for Player 2** (skips opponent-generated samples in vs_opponent mode)
4. ✅ Clear comments explaining each branch

#### Change 5: Add storage statistics logging
**Location**: Lines 499-514
```python
        last_update_metrics = agent.update()
        
        # Accumulate storage counters
        stored_p1_total += stored_p1_this_update
        stored_p2_total += stored_p2_this_update
        skipped_p2_due_to_opponent_total += skipped_p2_this_update
        
        # Periodic storage statistics logging (every 20 updates)
        if (update_idx + 1) % 20 == 0:
            total_stored = stored_p1_total + stored_p2_total
            if rollout_mode == "vs_opponent" and stored_p1_total > 0:
                skip_pct = 100.0 * skipped_p2_due_to_opponent_total / float(stored_p1_total)
            else:
                skip_pct = 0.0
            print(f"[Storage Stats] Update {update_idx + 1}: "
                  f"stored_p1={stored_p1_this_update}, stored_p2={stored_p2_this_update}, "
                  f"skipped_p2={skipped_p2_this_update} | "
                  f"Total: p1={stored_p1_total}, p2={stored_p2_total}, "
                  f"skipped={skipped_p2_due_to_opponent_total} ({skip_pct:.1f}%)", flush=True)
```

#### Change 6: Add rollout stats to CSV output
**Location**: Lines 635-640
```python
        row["rollout_mode"] = rollout_mode
        # ... existing fields ...
        row["stored_p1_total"] = stored_p1_total
        row["stored_p2_total"] = stored_p2_total
        row["skipped_p2_total"] = skipped_p2_due_to_opponent_total
```

#### Change 7: Add CLI argument
**Location**: Lines 721-727
```python
    parser.add_argument(
        "--rollout-mode",
        choices=["selfplay", "vs_opponent"],
        default="vs_opponent",
        help="Rollout mode for PPO: 'selfplay' (both use learner, store both) or 'vs_opponent' (p2 may use opponent, store only learner samples)",
    )
```

#### Change 8: Pass rollout_mode to run_ppo
**Location**: Line ~724 in _run_cli()
```python
        rows = run_ppo(
            cfg,
            episodes=args.episodes,
            train_qs=train_qs,
            eval_qs=eval_qs,
            rollout_mode=args.rollout_mode,  # NEW
            eval_symmetric=args.eval_symmetric,
            eval_vs_opponent=args.eval_vs_opponent,
            eval_vs_history=args.eval_vs_history,
        )
```

---

### 2. `tools/verify_rollout_modes.py` (New File)

**Purpose**: Automated verification tool to test both rollout modes

**Key Functions**:
- `test_rollout_mode()`: Runs short rollout and validates storage counts
- Checks for selfplay: exactly 2 transitions/step, no skips
- Checks for vs_opponent: 1-2 transitions/step, opponent samples skipped
- Validates buffer contains only expected samples

**Usage**:
```bash
python tools/verify_rollout_modes.py
```

**Verification Output**:
```
✅ SELFPLAY mode validation PASSED
   - Stored 2 transitions per step as expected

✅ VS_OPPONENT mode validation PASSED
   - Stored 1-2 transitions per step (opponent-dependent)
   - Skipped 53/100 Player 2 samples (opponent-generated)
   - Buffer contains ONLY learner-generated samples

✅ ALL CHECKS PASSED
```

---

### 3. `docs/rollout_modes_ablation.md` (New Documentation)

**Comprehensive documentation covering**:
- Executive summary of the bug and fix
- Detailed explanation of both modes
- Before/after code comparison
- File-by-file change list with line numbers
- Example command lines
- Verification procedures
- Testing & validation guide

---

## Summary of Changes by Impact

### Critical (Bug Fix)
✅ **Conditional storage for Player 2** (lines 445-497)
- In `vs_opponent` mode, skip storing opponent-generated transitions
- Fixes the core bug: no more mixing of opponent samples in PPO buffer

### High (New Feature)
✅ **Rollout mode parameter** (lines 248-268, 721-727)
- CLI flag `--rollout-mode` with two choices
- Explicit control over training semantics

### Medium (Observability)
✅ **Storage counters and logging** (lines 380-383, 499-514, 635-640)
- Track and report stored/skipped transitions
- Verify correct behavior at runtime
- Save stats to CSV for analysis

### Low (Code Quality)
✅ **Mode announcement** (lines 304-313)
- Clear logging of active mode
- Helps prevent confusion during experiments

✅ **Verification tool** (new file)
- Automated testing of both modes
- Sanity checks for storage counts

✅ **Documentation** (new file)
- Comprehensive guide for users
- Explains bug, fix, and usage

---

## Testing Checklist

- [x] Verification tool passes for both modes
- [x] Selfplay mode stores 2 transitions/step
- [x] VS_OPPONENT mode skips opponent samples
- [x] CLI arguments parse correctly
- [x] Mode announcement logs correctly
- [x] Storage stats log periodically
- [x] CSV includes new fields
- [x] Backward compatible (default = vs_opponent)
- [x] No changes to agent.update() needed
- [x] Documentation complete and accurate

---

## Migration Guide

### For Existing Experiments

**No changes required** if you want to keep the corrected "vs_opponent" behavior (default):
```bash
# These are equivalent:
python run/run_two_players.py --method ppo --episodes 1800000
python run/run_two_players.py --method ppo --rollout-mode vs_opponent --episodes 1800000
```

**To switch to pure selfplay**:
```bash
python run/run_two_players.py --method ppo --rollout-mode selfplay --episodes 1800000
```

### For Reproducibility

**Include rollout_mode in experiment tracking**:
- CLI logs now show active mode
- CSV results include `rollout_mode` column
- Storage stats logged every 20 updates

---

## Quick Start

### 1. Verify Installation
```bash
python tools/verify_rollout_modes.py
# Should output: ✅ ALL CHECKS PASSED
```

### 2. Run Selfplay
```bash
python run/run_two_players.py \
    --method ppo \
    --rollout-mode selfplay \
    --episodes 1800000 \
    --q 25.0
```

### 3. Run VS_OPPONENT (Default)
```bash
python run/run_two_players.py \
    --method ppo \
    --rollout-mode vs_opponent \
    --episodes 1800000 \
    --q 25.0
```

### 4. Compare Results
```python
import pandas as pd
df = pd.read_csv("results/one_stage_two_players.csv")
selfplay = df[df["rollout_mode"] == "selfplay"]
vs_opp = df[df["rollout_mode"] == "vs_opponent"]
```

---

## Contact / Questions

For questions about this refactor, refer to:
- **Full documentation**: `docs/rollout_modes_ablation.md`
- **Verification tool**: `tools/verify_rollout_modes.py`
- **Original investigation**: `docs/data_provenance_investigation.md`



