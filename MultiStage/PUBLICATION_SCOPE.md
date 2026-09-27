# Publication scope and provenance

## Inclusion criterion

Include an artifact if it is needed to **reproduce a reported experiment, identify its exact configuration and seeds, or understand a scientific conclusion or a decision to stop an approach**. Apply this criterion to successful and unsuccessful results equally.

| Material | Treatment |
|---|---|
| Shared dynamics, learner and verifier used in reported experiments | Executable source included |
| Primary T2 and T3 experiment settings, prescribed seeds, all-run outcomes | Included, with distinct cohort names |
| T3 pilot and formal endpoint/minimum weights and compact verification/economic evidence | Included as documented by the T3 archive |
| Earlier T2 protocols and failed design comparisons | Identified as historical/supporting evidence, separate from final-protocol confirmation |
| T3 sampling, noise, representation and budget diagnostics | Compact study descriptions, decisions and evidence in the diagnostics archive |
| Abandoned plans that never produced runs | Mentioned only when needed to prevent a misleading claim; never counted as completed data |
| Agent prompts, chat transcripts, memory, task handoffs and session-state files | Excluded from this upload |
| Large rollout tensors, duplicate dense CSVs, every intermediate checkpoint, temporary plots and shell logs | Retained on the research server; not bundled into Git |
| Existing one-stage repository contents | Preserved; linked as a separate research track |

Exploratory archives are **not promised to be complete executable replications**. Their README identifies which evidence is supplied. The primary T2/T3 code is packaged for execution. This distinction is deliberate: keeping a scientific record of a stopped approach does not require publishing internal coordination material.

## Provenance and portability

The original runs were performed in several server worktrees. Some experimental Python files were untracked at the time, so the original recorded base commit alone does not reconstruct them. This publication commit supplies those source files together with their experiment configurations and reported results.

The shared numerical implementation is copied from the worktree that supplied both T2 and T3 dependencies. The existing analytic utility file was checked against it and matched. The publication adapts machine-specific paths, interpreter selection and report links, and adds reproduction interfaces. It does not retune the solver, alter seeds, change certification thresholds or rerun the formal studies to obtain new outcomes.

Archived outcome numbers and checkpoints remain the original observations. Relocated manifests point to publication paths; use fresh output paths for reproduction. Runtime, host and path metadata from the original environment are not portable scientific parameters.

## Interpretation boundaries

- Pilot and exploratory seeds are not formal held-out observations.
- Earlier-protocol T2 formal results later used for design are not independent confirmation of the subsequently selected protocol.
- The T2 precision supplement contributes descriptive precision; it is not a new protocol-selection opportunity.
- T3 formal results establish failure to find candidates within the stated budgets on these seeds. They do not establish nonexistence of equilibrium.
- “Small root payoff deviation” does not imply that every state satisfies the strategic criterion.
- No claim is made that every seed converges.

The original server directories are preserved. This upload creates a reading and reproduction layer without deleting the full research record.
