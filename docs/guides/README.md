# docs/guides/

## Purpose

User-facing guides for running experiments and using project tools. These documents explain how to configure, run, and analyze tournament experiments.

## Key Contents

| File | Description |
|------|-------------|
| `ppo_defaults.md` | PPO default configuration guide - explains auto-enabled options for `--method ppo` |
| `plot_convergence.md` | Guide for generating convergence plots from JSON history files |
| `asymmetric_init.md` | Asymmetric initialization mechanism - two agents start from different effort points |

## Entry Points / How to Use

Read guides before running experiments:

```bash
# Understand PPO defaults
cat docs/guides/ppo_defaults.md

# Learn how to generate plots
cat docs/guides/plot_convergence.md
```

## Dependencies & Contracts

**Depends on:** Nothing (documentation only)

**Provides to system:**
- Quick-start instructions for new users
- Reference for CLI options and defaults
- Troubleshooting guides

## Gotchas / Conventions

- Documents may contain Chinese text (originally written for bilingual team)
- File paths in guides have been updated to reflect refactored structure
- PPO defaults guide reflects current default behavior (may change with code updates)

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-02-03 | Moved from root level (PPO_DEFAULTS_README.md, etc.) to docs/guides/ |
| 2026-02-03 | Updated file paths to reflect plot scripts moved to tools/ |
