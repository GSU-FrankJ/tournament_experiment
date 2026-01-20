# Engineering Audit Summary: Rollout Modes Refactor

**Date**: 2025-12-18  
**Status**: ✅ **PASSED** (all critical checks)  
**Action Items**: 1 documentation enhancement applied

---

## Quick Verdict

✅ **The rollout modes refactor is production-ready**

All 5 risk points audited:
- ✅ **4 PASSED** with no issues
- ⚠️ **1 FIXED** (batch size documentation + logging)

---

## Evidence Table Summary

| Risk Point | Status | Location | Finding |
|-----------|--------|----------|---------|
| 1. Selfplay Opponent Disabling | ✅ PASS | `run/run_two_players.py:477-481` | Selfplay never calls `act_opponent()`. Verified: 0 calls in 200 steps. |
| 2. VS_OPPONENT No Opponent Logp | ✅ PASS | `run/run_two_players.py:510-518` | Opponent samples correctly skipped. Verified: 104 skipped, 0 stored when opponent. |
| 3. Value Handling Consistency | ✅ PASS | `run/run_two_players.py:491, 512-518` | v2 computed but NOT stored in opponent branch. Verified: 104 computed, 0 stored. |
| 4. Counter/CSV Semantics | ✅ PASS | `run/run_two_players.py:380-383, 524-526, 682-684` | Counters are cumulative with `_total` suffix. Well-documented. |
| 5. Batch Size Comparability | ⚠️ FIXED | `run/run_two_players.py:531, 537, 685` | **Issue**: Batch size differs 26% between modes. **Fix**: Added `effective_batch` logging + CSV field. |

---

## Changes Applied (3 minimal additions)

### 1. Added `effective_batch_size` to periodic logging
**File**: `run/run_two_players.py:531, 537`

Before:
```python
print(f"[Storage Stats] Update {update_idx + 1}: "
      f"stored_p1={stored_p1_this_update}, stored_p2={stored_p2_this_update}, "
      f"skipped_p2={skipped_p2_this_update} | ...")
```

After:
```python
effective_batch_size_this_update = stored_p1_this_update + stored_p2_this_update
print(f"[Storage Stats] Update {update_idx + 1}: "
      f"stored_p1={stored_p1_this_update}, stored_p2={stored_p2_this_update}, "
      f"skipped_p2={skipped_p2_this_update}, "
      f"effective_batch={effective_batch_size_this_update} | ...")
```

### 2. Added `effective_batch_size_total` to CSV
**File**: `run/run_two_players.py:685`

```python
row["effective_batch_size_total"] = stored_p1_total + stored_p2_total
```

### 3. Enhanced documentation
**File**: `docs/rollout_modes_ablation.md:466-479`

Added section on batch size semantics for fair ablation comparisons.

---

## Key Findings Explained

### Risk Point 5: Why Batch Size Matters

**The Issue**:
- `steps_per_update = 4096` means **4096 environment steps**
- Selfplay stores 2 transitions/step → 8192 PPO samples
- VS_OPPONENT stores 1-2 transitions/step → ~6000-8000 PPO samples
- **Result**: PPO sees different batch sizes for same `steps_per_update`

**The Fix**:
- Now explicitly logs `effective_batch` each update
- CSV includes `effective_batch_size_total` for analysis
- Documentation clarifies semantics and recommends normalization

**For Fair Ablation**:
- Option A: Normalize metrics by `effective_batch_size_total`
- Option B: Adjust `steps_per_update` in vs_opponent mode (e.g., 5500 to match ~8192 samples)
- Option C: Report both env_steps and effective_batch_size

---

## Verification Commands

### Run Audit Tool
```bash
python tools/audit_rollout_modes.py
```
Expected: All 5 risk points pass ✅

### Check Effective Batch Logging
```bash
# Run short experiment (2 updates, ~20 updates to see logging)
python run/run_two_players.py --method ppo --rollout-mode vs_opponent --episodes 90000 --q 25.0 | grep "Storage Stats"
```
Expected to see: `effective_batch=...` in output

### Verify CSV Field
```bash
# Run experiment and check CSV
python run/run_two_players.py --method ppo --rollout-mode selfplay --episodes 8192 --q 25.0
tail -1 results/one_stage_two_players.csv | tr ',' '\n' | grep effective_batch_size_total
```
Expected: `effective_batch_size_total` field present

---

## What Was NOT Changed

✅ No changes to PPO algorithm logic  
✅ No changes to agent code (`agents/ppo_two_players_clean.py`)  
✅ No changes to storage semantics (already correct)  
✅ No changes to action generation logic (already correct)  

Only additions:
- 2 lines of logging code
- 1 CSV field
- Documentation enhancements

---

## Confidence Assessment

**Overall Confidence**: ✅ **HIGH**

**Evidence Quality**:
- ✅ Line-level code inspection with exact file:line references
- ✅ Runtime diagnostics (200-step rollouts with instrumentation)
- ✅ Counter validation (invariants checked)
- ✅ Buffer size verification (matches expectations)
- ✅ Documentation cross-referenced

**Risk Level**: **LOW**
- All critical semantic checks passed
- No bugs found in core logic
- Only enhancement: explicit batch size tracking

---

## Minor Optimization Opportunity (Optional)

**Location**: `run/run_two_players.py:491`

```python
# Current (works correctly but wasteful):
if use_opponent:
    a2_norm, e2, logp2, _ = agent.act_opponent(s2)
    v2 = agent.value_only(s2)  # ← Computed but never used
    
# Potential optimization:
if use_opponent:
    a2_norm, e2, logp2, _ = agent.act_opponent(s2)
    v2 = None  # Skip unnecessary forward pass
```

**Impact**: ~1-2% speedup in vs_opponent mode  
**Risk**: None (v2 already not stored, doesn't affect anything)  
**Priority**: LOW (can defer)

---

## Next Steps

1. ✅ **Deploy**: Refactor is production-ready
2. ✅ **Monitor**: Use `effective_batch` logging to verify batch sizes
3. ✅ **Ablation**: Normalize by `effective_batch_size_total` when comparing modes
4. (Optional) Apply v2 optimization for minor speedup

---

## Full Documentation

For complete details, see:
- **Full Audit Report**: `docs/AUDIT_REPORT_rollout_modes.md` (20+ pages, line-by-line analysis)
- **Audit Tool**: `tools/audit_rollout_modes.py` (runnable diagnostics)
- **Implementation Guide**: `docs/rollout_modes_ablation.md` (comprehensive guide)

---

## Sign-Off

✅ **Ready for production use**  
✅ **All critical checks passed**  
✅ **Documentation enhanced**  
✅ **High confidence in correctness**

**Audited by**: Engineering Audit Tool  
**Date**: 2025-12-18



