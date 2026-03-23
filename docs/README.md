# docs/

Documentation for the tournament experiment project.

## Structure

```
docs/
├── README.md
├── STATE.md                    # Project-level state tracker
├── experiment_plan.md          # Research methodology and experiment design
├── guides/
│   ├── README.md
│   ├── ppo_defaults.md         # PPO auto-enabled flags
│   ├── ppo_flags.md            # Theory-align-v2, convergence-eval, cheap-gate
│   ├── plot_convergence.md     # Convergence plotting guide
│   ├── asymmetric_init.md      # Asymmetric agent initialization
│   └── results-folder-guide.md # Directory structure, JSON formats, naming
├── tasks/                      # Task pipeline (see tasks/README.md)
│   ├── README.md               # Conventions and templates
│   ├── runner-refactor/        # Extract shared runner logic
│   └── perfect-exploitability-figure/ # Fix exploitability_dynamics figure
└── technical/
    ├── README.md
    ├── rollout_modes.md        # Selfplay mode and historical data mixing bug
    ├── audit_theory_align_v2.md # Theory-align-v2 audit
    └── POLICY_SCALE_DIAGNOSTICS.md # Policy definition and scale metrics
```

## Entry Points

- **New to the project?** Start with `experiment_plan.md` for research context, then `guides/results-folder-guide.md` for data layout.
- **Running experiments?** See `guides/ppo_defaults.md` and `guides/ppo_flags.md`.
- **Understanding internals?** See `technical/rollout_modes.md` and `technical/POLICY_SCALE_DIAGNOSTICS.md`.
- **Starting a new task?** See `tasks/README.md` for the task pipeline convention.
