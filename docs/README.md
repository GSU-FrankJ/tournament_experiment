# docs/

## Purpose

Central documentation hub for the tournament experiment project. Contains user guides, technical documentation, and archived legacy materials.

## Key Contents

| Path | Description |
|------|-------------|
| `guides/` | User-facing guides for running experiments and using tools |
| `technical/` | Technical deep-dives, audit reports, and implementation details |
| `archive/` | Legacy code, Word documents, and historical materials |
| `CHANGES_SUMMARY.md` | Summary of major implementation changes |
| `CONVERGENCE_TRACKING.md` | Guide for recording and visualizing convergence |
| `experiment_plan.md` | Research methodology and experiment design |
| `sep 13 report.md` | Historical development notes (Chinese) |

## Entry Points / How to Use

Browse documentation by category:

```bash
# User guides (how to run experiments)
ls docs/guides/

# Technical documentation (implementation details)
ls docs/technical/

# Legacy materials
ls docs/archive/
```

## Dependencies & Contracts

**Depends on:** Nothing (documentation only)

**Provides to system:**
- Reference documentation for developers and users
- Context for AI agents exploring the codebase
- Historical record of design decisions

## Subdirectory Overview

### guides/
User guides for:
- PPO default configurations (`ppo_defaults.md`)
- Convergence plotting (`plot_convergence.md`)
- Asymmetric initialization (`asymmetric_init.md`)

### technical/
Implementation documentation:
- Rollout modes audit and implementation
- Data provenance investigation
- Gradient sweep guides
- Theory alignment documentation

### archive/
Legacy materials (not actively maintained):
- Original experiment plans (Word format)
- PowerPoint presentations
- Old Python scripts

## Gotchas / Conventions

- Active documentation is in markdown format
- Some documents are in Chinese (historical notes)
- `technical/` contains audit reports from code reviews
- `archive/` materials are preserved for reference but may be outdated

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-02-03 | Created docs/ by reorganizing documents/; added guides/, technical/, archive/ structure |
