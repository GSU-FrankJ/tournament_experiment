# Audit Verification Commands

Quick reference for verifying all audit findings.

---

## 1. Run Complete Audit Tool

```bash
cd /home/fjiang4/tournament_experiment-4
python tools/audit_rollout_modes.py
```

**Expected Output**:
```
Risk Point 1 (Selfplay opponent disabling): ✅ PASS
Risk Point 2 (VS_OPPONENT no opponent logp): ✅ PASS
Risk Point 3 (Value handling consistency): ✅ PASS
Risk Point 4 (Counter/CSV semantics): ✅ PASS (verified by code inspection)
Risk Point 5 (Batch size comparability): ⚠️  NEEDS ATTENTION (not a bug, but needs documentation)

✅ AUDIT COMPLETE: All critical checks passed
```

---

## 2. Verify Selfplay Mode (Risk Point 1)

```bash
# Run short selfplay experiment
python run/run_two_players.py \
    --method ppo \
    --rollout-mode selfplay \
    --episodes 8192 \
    --q 25.0

# Check output for:
# [PPO] Rollout mode: SELFPLAY
# [PPO]   - Both players use learner policy
# [PPO]   - Store both transitions every step
```

**Manual Check**:
```bash
# Check CSV results
tail -1 results/one_stage_two_players.csv | python -c "
import sys, csv
row = next(csv.DictReader(sys.stdin))
print(f'rollout_mode: {row[\"rollout_mode\"]}')
print(f'skipped_p2_total: {row[\"skipped_p2_total\"]}')
assert row['rollout_mode'] == 'selfplay'
assert row['skipped_p2_total'] == '0'
print('✅ Selfplay verified')
"
```

---

## 3. Verify VS_OPPONENT Mode (Risk Point 2)

```bash
# Run short vs_opponent experiment
python run/run_two_players.py \
    --method ppo \
    --rollout-mode vs_opponent \
    --episodes 8192 \
    --q 25.0

# Check output for:
# [PPO] Rollout mode: VS_OPPONENT
# [PPO]   - Player1 uses learner; Player2 may use opponent (lag schedule)
# [PPO]   - Store only learner-generated transitions
```

**Manual Check**:
```bash
# Verify storage invariants in CSV
tail -1 results/one_stage_two_players.csv | python -c "
import sys, csv
row = next(csv.DictReader(sys.stdin))
p1 = int(row['stored_p1_total'])
p2 = int(row['stored_p2_total'])
skipped = int(row['skipped_p2_total'])
print(f'P1 stored: {p1}')
print(f'P2 stored: {p2}')
print(f'P2 skipped: {skipped}')
assert p2 + skipped == p1, f'{p2} + {skipped} != {p1}'
assert skipped > 0, 'Expected some skips in vs_opponent'
print('✅ VS_OPPONENT storage verified')
"
```

---

## 4. Verify Effective Batch Size Logging (Risk Point 5 Fix)

```bash
# Run experiment with enough updates to see logging
python run/run_two_players.py \
    --method ppo \
    --rollout-mode vs_opponent \
    --episodes 100000 \
    --q 25.0 \
    2>&1 | grep "Storage Stats" | head -3

# Expected output includes:
# [Storage Stats] Update 20: ... effective_batch=... |
```

**Check CSV Field Exists**:
```bash
# Verify new CSV field
tail -1 results/one_stage_two_players.csv | python -c "
import sys, csv
row = next(csv.DictReader(sys.stdin))
assert 'effective_batch_size_total' in row, 'Missing effective_batch_size_total field'
batch_size = int(row['effective_batch_size_total'])
p1 = int(row['stored_p1_total'])
p2 = int(row['stored_p2_total'])
assert batch_size == p1 + p2, f'{batch_size} != {p1} + {p2}'
print(f'✅ effective_batch_size_total = {batch_size} (correct)')
"
```

---

## 5. Compare Batch Sizes Between Modes

```bash
# Run both modes with same steps_per_update
python run/run_two_players.py --method ppo --rollout-mode selfplay --episodes 20000 --q 25.0
python run/run_two_players.py --method ppo --rollout-mode vs_opponent --episodes 20000 --q 25.0

# Analyze batch sizes
python -c "
import pandas as pd
df = pd.read_csv('results/one_stage_two_players.csv')
selfplay = df[df['rollout_mode'] == 'selfplay'].iloc[-1]
vs_opp = df[df['rollout_mode'] == 'vs_opponent'].iloc[-1]

sp_batch = selfplay['effective_batch_size_total']
vo_batch = vs_opp['effective_batch_size_total']
diff_pct = 100.0 * (sp_batch - vo_batch) / sp_batch

print(f'Selfplay effective_batch_size: {sp_batch}')
print(f'VS_OPPONENT effective_batch_size: {vo_batch}')
print(f'Difference: {diff_pct:.1f}%')
print(f'Expected: ~15-30% difference (depends on lag schedule)')
"
```

---

## 6. Verify No Opponent Logp in Buffer (Risk Point 2 Deep Check)

This is verified by the audit tool, but you can also manually inspect:

```bash
# Add debug flag to agent and check buffer contents
# (This requires code modification - see audit tool for implementation)
```

---

## 7. Check Documentation Updates

```bash
# Verify documentation includes batch size guidance
grep -A5 "Important for ablation comparisons" docs/rollout_modes_ablation.md

# Expected to see:
# - steps_per_update refers to **environment steps**, not stored transitions
# - Effective PPO batch size = ...
# - Selfplay mode: effective_batch_size = 2 × env_steps
# - VS_OPPONENT mode: effective_batch_size = 1.5 to 2 × env_steps
```

---

## 8. Full Integration Test

```bash
# Run both modes for longer to ensure stability
python run/run_two_players.py --method ppo --rollout-mode selfplay --episodes 200000 --q 25.0
python run/run_two_players.py --method ppo --rollout-mode vs_opponent --episodes 200000 --q 25.0

# Check final results
tail -2 results/one_stage_two_players.csv | python -c "
import sys, csv
rows = list(csv.DictReader(sys.stdin))
for row in rows:
    mode = row['rollout_mode']
    p1 = int(row['stored_p1_total'])
    p2 = int(row['stored_p2_total'])
    skipped = int(row['skipped_p2_total'])
    batch = int(row['effective_batch_size_total'])
    gap = float(row['stage2_gap_unweighted'])
    
    print(f'{mode}:')
    print(f'  P1={p1}, P2={p2}, skipped={skipped}')
    print(f'  effective_batch={batch}')
    print(f'  gap_from_theory={gap:.2f}')
    
    if mode == 'selfplay':
        assert p2 == p1, 'Selfplay should store both'
        assert skipped == 0, 'Selfplay should skip none'
    else:
        assert p2 + skipped == p1, 'VS_OPPONENT invariant'
        assert skipped > 0, 'VS_OPPONENT should skip some'
    
    assert batch == p1 + p2, 'Batch size formula'
    print('  ✅ Invariants verified')
"
```

---

## 9. Verify No Regressions

```bash
# Compare with old verification tool (should still pass)
python tools/verify_rollout_modes.py

# Expected:
# ✅ SELFPLAY mode validation PASSED
# ✅ VS_OPPONENT mode validation PASSED
# ✅ ALL CHECKS PASSED
```

---

## Quick Checklist

- [ ] Audit tool passes all 5 risk points
- [ ] Selfplay mode logs correctly and stores 2 transitions/step
- [ ] VS_OPPONENT mode logs correctly and skips opponent samples
- [ ] `effective_batch` appears in Storage Stats logging
- [ ] `effective_batch_size_total` field exists in CSV
- [ ] Documentation mentions batch size differences
- [ ] Both modes run without errors
- [ ] Storage invariants hold in CSV results

---

## Troubleshooting

### If audit tool fails:
1. Check Python environment: `python --version` (need 3.8+)
2. Check dependencies: `pip install -r requirements.txt`
3. Review error message for specific risk point
4. See full report: `docs/AUDIT_REPORT_rollout_modes.md`

### If logging doesn't show effective_batch:
1. Ensure you run enough updates (logging every 20 updates)
2. Check episodes count: need ~100k+ to see multiple log lines
3. Verify code changes applied: `grep effective_batch run/run_two_players.py`

### If CSV missing field:
1. Remove old CSV: `rm results/one_stage_two_players.csv`
2. Re-run experiment to generate fresh CSV
3. Check field exists: `head -1 results/one_stage_two_players.csv | tr ',' '\n' | grep effective_batch`

---

## Contact

For issues with audit verification:
1. Re-read `docs/AUDIT_SUMMARY.md` for quick overview
2. See `docs/AUDIT_REPORT_rollout_modes.md` for detailed analysis
3. Run `python tools/audit_rollout_modes.py` for automated checks



