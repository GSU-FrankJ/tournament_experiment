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
