# Task Pipeline

## Convention

Each non-trivial task gets a named directory under `docs/tasks/`:

```
docs/tasks/{task-name}/
├── CLAUDE.md       # Task-specific context, constraints, and decisions
├── STATE.md        # Current status, blockers, what's done / what's next
├── phase01.md      # First phase plan
├── phase02.md      # Subsequent phases...
└── ...
```

## Lifecycle

1. **Create** — `mkdir docs/tasks/{task-name}`, populate CLAUDE.md + STATE.md + phase01.md
2. **Execute** — Work through phases; update STATE.md after each phase
3. **Complete** — Mark STATE.md status as "complete", leave in place for reference

## Current Tasks

| Task | Status | Description |
|------|--------|-------------|
| `runner-refactor` | in-progress | Extract shared logic across the 4 experiment runners |
| `perfect-exploitability-figure` | in-progress | Fix exploitability_dynamics figure and related artifacts |
| `diagnose-all-experiments` | complete | Diagnostic analysis of three_players, different_cost, different_ability |
| `paper-figures-tables-revision` | planning | Revise paper figures/tables for publication (restyle, rename, add q=35) |
| `q35-all-experiments` | in-progress | Run q=35 PPO + gradient across all non-two-player experiment types |

## Naming

- Task names: lowercase, hyphenated (e.g., `runner-refactor`, `add-unit-tests`)
- Phase files: `phase{NN}.md` zero-padded (01–99)

## CLAUDE.md template

Task-specific context for Claude. Include:
- Goal (1-2 sentences)
- Scope boundaries (what to touch, what NOT to touch)
- Key files involved
- Constraints or decisions made

## STATE.md template

```markdown
# {Task Name}

Status: planning | in-progress | blocked | complete
Current phase: phase{NN}

## What's done
- ...

## What's next
- ...

## Blockers
- (if any)
```

## Phase file template

```markdown
# Phase {NN}: {Title}

## Objective
What this phase achieves.

## Steps
1. ...
2. ...

## Files to modify
- ...

## Verification
How to confirm the phase is done.
```
