# docs/technical/

## Purpose

Technical documentation including audit reports, implementation details, and deep-dive analyses. These documents explain the "why" and "how" of complex implementations.

## Key Contents

| File | Description |
|------|-------------|
| `IMPLEMENTATION_COMPLETE.md` | Rollout modes refactor completion summary |
| `rollout_modes_ablation.md` | Detailed rollout modes documentation |
| `rollout_modes_changes_summary.md` | Quick reference for rollout mode changes |
| `AUDIT_REPORT_rollout_modes.md` | Full audit report for rollout modes |
| `AUDIT_SUMMARY.md` | Condensed audit findings |
| `audit_theory_align_v2.md` | Theory alignment v2 implementation audit |
| `data_provenance_investigation.md` | Investigation into data mixing bug |
| `data_mixing_summary.md` | Summary of data mixing issue and fix |
| `gradient_sweep_guide.md` | Guide for running gradient sweeps |
| `gradient_sweep_quickref.md` | Quick reference for gradient sweeps |
| `POLICY_SCALE_DIAGNOSTICS.md` | Policy scale diagnostic tools |
| `COMMAND_LINE_GUIDE_vs_opponent.md` | CLI guide for vs_opponent mode |
| `AUDIT_VERIFICATION_COMMANDS.md` | Commands for verifying audited behavior |

## Entry Points / How to Use

Reference these documents when:
- Debugging rollout mode behavior
- Understanding the data mixing fix
- Running gradient sweep experiments
- Verifying implementation correctness

```bash
# Understand rollout modes
cat docs/technical/rollout_modes_changes_summary.md

# Debug data provenance
cat docs/technical/data_provenance_investigation.md
```

## Dependencies & Contracts

**Depends on:** Nothing (documentation only)

**Provides to system:**
- Implementation audit trails
- Technical decision rationale
- Debugging guides and verification commands

## Document Categories

### Rollout Modes
- `rollout_modes_ablation.md` - Full documentation
- `rollout_modes_changes_summary.md` - Quick reference
- `AUDIT_REPORT_rollout_modes.md` - Formal audit

### Data Provenance
- `data_provenance_investigation.md` - Original investigation
- `data_mixing_summary.md` - Issue summary

### Gradient Sweeps
- `gradient_sweep_guide.md` - Full guide
- `gradient_sweep_quickref.md` - Quick reference
- `gradient_sweep_evaluation_update.md` - Evaluation updates

## Gotchas / Conventions

- Audit reports follow a formal structure with findings/recommendations
- Some documents reference line numbers that may be outdated after refactors
- `test.txt` is a placeholder file (can be removed)

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-02-03 | Moved from documents/docs/ to docs/technical/ |
