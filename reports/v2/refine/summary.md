# R1 accuracy-refinement round: summary

Round scope (PI prompt, steps P0-P4): two open diagnostics of the T=2 report (T57 issue 7 clipped-Beta likelihood, issue 8 verifier sensitivity) and six candidate changes to the locked v1.1 pipeline tested one at a time, paired by (q, seed), on the development seeds 10501-10510 x q in {50, 60}. The protocol, the locked entry point and the confirmation seeds were not touched; no combination of factors was run; T=3 was not started. The round stops here: the combination, protocol v2.0 and the next confirmation are the PI's decision. Branch `v2-t2-refine` (from `f02a256`); code commit `32a8c21`, pre-registration `6c902db`.

## Reading order

| # | report | content |
|---|---|---|
| 00 | [00_preflight.md](00_preflight.md) | git state (one documented deviation), host, branch-point suite and C7, parents and seed inventory, baseline reference numbers, KL statistics |
| 01 | [01_preregistration.md](01_preregistration.md) | arms and exact settings, implementation of the six methods, metrics, criterion, bootstrap, D1/D2 definitions, verification state, Addendum 1 (after the launches) |
| 02 | [02_d1_clamp.md](02_d1_clamp.md) | D1: clamp hits of the raw Beta draws, censored log-mass, gradient share, flags M1/M2 |
| 03 | [03_d2_verifier_sensitivity.md](03_d2_verifier_sensitivity.md) | D2: 152 perturbed closed-form candidates on both tiers, fits, detection limits |
| 04 | [04_pilot_stage1.md](04_pilot_stage1.md) | stage-1 wave: 8 arms x 20 runs |
| 05 | [05_pilot_stage2.md](05_pilot_stage2.md) | stage-2 wave: 11 arms x 20 runs |
| 06 | [06_decision_inputs.md](06_decision_inputs.md) | one row per method, method-5 and method-6 detail, D1/D2 headlines, observations |

Data: `results/v2_refine/` (per-run records and analysis tables are tracked; checkpoints, weight exports, `train_history.json`, D1 buffers and the 35 MB per-update D1 table are not, see `reports/v2/README.md` for the tracking rules).

## What was verified before and after the runs

- Code: with every new key at its default the pipeline is bit-identical. C7 identical on the final code (`results/v2_refine/c7/code_check.compare.txt`); the unchanged locked entry point reproduces `rehearsal_v1_1` in **20/20** runs (C-R1, `results/v2_refine/v11_reproduction_checks.json`); suite on the final code: 328 passed, 1 failed (the known registry test), 4 xfailed (`results/v2_refine/code_pytest_full.txt`).
- Branching: `parents_A` end-of-A state 20/20, `B_base` 20/20, `A_base` 20/20 identical to `rehearsal_v1_1` (`parents_A_checks.json`, `stage1_base_checks.json`, `stage2_base_checks.json`).
- Runs: parents_A 20/20, stage 1 160/160, stage 2 220/220 exited 0; every manifest of the analysed runs shows `dirty: false`; 17 stage-1 runs had to be re-run for that reason and are bit-identical to their superseded originals (Addendum 1 item 3).
- Reviews: the module code was reviewed by independent reviewers (one pass per deliverable), the core diff by five reviewers with distinct lenses before the code commit, and the analysis tables and generated reports were recomputed from the raw files by three independent verifiers (about 2,900 report rows traced, no wrong number; the presentation points they raised were fixed before this summary).

## Headline numbers (all from the generated reports)

- **Stage 1.** `B_expcont` (method 6) is the only arm that meets the pre-registered criterion: mean paired difference of the S1 error -0.02386 [-0.04061, -0.009831] (q = 50) and -0.03783 [-0.06321, -0.01472] (q = 60); the signed error does not move; the across-seed SD of the signed error is 0.35x and 0.27x the baseline's; the stage-1 advantage SD is 0.066x and 0.053x; wall time 1.084x. The other six arms do not exclude 0 at either q. All 160 stage-1 runs pass G-F and G-N.
- **Stage 2.** No arm meets part (a) at both q; all 220 runs pass G-A and G-N. `A_batch_mb256` is below 0 at q = 50 only; `A_anneal4` is above 0 at q = 60. Method 5: `A_detmean` - `A_ctrl200` has an interval containing 0 at both q.
- **D1.** In the locked pipeline M1 is exceeded in phase A (median clamp fraction 0.001385 pooled; hits only for |d| >= 2q of the final stage) and not in phase B; M2 is exceeded in phase A in the pooled reading (3 of 20 runs; maximum clamped-row gradient share 0.02353) and not in phase B. Report-only; no fix applied.
- **D2.** The final-tier verifier does not reach G-F for a stage-1 scalar error up to 15% (largest Gmax/DW 0.00717 at q = 50); a stage-2 amplitude error violates G-F and G-A at 10% (q = 50) and 15% (q = 60); peak rounding up to h = 20, tail offsets up to tau = 2 and the RL-like combinations do not reach any gate.

## Decisions, deviations and open points

1. **Check (ii) of the continuation table is open as stated.** The final-tier verifier's stage-1 Q differs from the table by 3.229e-05 DW (q = 50) and 1.960e-05 DW (q = 60) against the required 1e-6 DW; with the verifier's state step refined to 0.25 and 64 quadrature nodes the gap is 4.982e-07 / 4.100e-07 DW. The table's quadrature is converged to at most 5e-9 DW. The PI decided to keep method 6 and to record both results (asked during the round); the literal test is a strict expected failure in the suite and no tolerance was loosened (`01_preregistration.md` section 4 item 6, `results/v2_refine/continuation_check.json`).
2. **Local `main` ref is stale (`d1b8443`, 52 commits behind) and its checkout holds other sessions' files;** the branch was taken from `f02a256` = `origin/main` = tag `t2-v2-main`, the commit the prompt names (`00_preflight.md` section 1).
3. **Concentration head of method 5.** The preserved Adam moments would move the head even with a zero gradient, so its weight and bias are restored after each step; the end-of-phase check (head, critic, critic Adam state bit-identical to the parent) passed in every `A_detmean` run (`phaseP_checks.json` per run).
4. **Launch hygiene.** The 17 re-runs and the concurrency (at most 40 processes) are described in Addendum 1 of the pre-registration. Two analysis-side commits touch `tools/v2/d1_clamp_analysis.py` after the launches began (the run code is unchanged since `32a8c21`).
5. **Report pack.** `docs/STATE.md` was edited and `reports/v2/summary.md` got an appended paragraph; the pack `reports/v2/t2_report` cites STATE.md by SHA-256 (T57), so that entry is stale until the documented pack rebuild (not done: out of scope).
6. **Stray files.** Four early copies of this round's files (`utils/v2_continuation.py`, `agents/ppo_pathwise.py`, `tests/test_v2_refine_pathwise.py`, `tools/v2/refine_preflight.py`) were written by workers into the original session worktree `.claude/worktrees/t2-accuracy-refinement-r1-32d472` before the session was switched to `v2-t2-refine`; they are untracked duplicates there and were not deleted.
7. **D1 flag reading.** "More than 2 of the 20 runs" for M2 is read as the 20 C-R1 runs pooled over both q, with per-q counts beside it (Addendum 1 item 2).

## Addendum (2026-10-03, v2.0 round): the PI accepted this round

Added by the v2.0 round; nothing above was changed. Source: decisions D1 and D3 of the PI's v2.0-round prompt (R1 accepted: protocol v2.0, re-rehearsal, then the confirmation).

1. **Accepted with its deviations.** The 17 stage-1 re-runs (manifests with `dirty: true`, originals kept under `results/v2_refine/stage1/dirty_rerun`, re-runs bit-identical 17/17), the two analysis-side commits after the launches (run code unchanged since `32a8c21`) and the pooled reading of the D1 flag M2 are accepted as recorded in `01_preregistration.md` Addendum 1.
2. **Only method 6 enters the protocol.** Only method 6 met the pre-registered criterion (`06_decision_inputs.md` section 1). No other R1 arm enters the protocol. Target-KL is not adopted either: it changes cost, not accuracy, and the protocol changes for accuracy only.
3. **Check (ii) of the continuation table is settled** (item 1 above, "open as stated", is closed): the literal criterion (<= 1e-6 DW against the verifier's standard final tier) is replaced by the three pre-registered tests (ii-a) self-convergence, (ii-b) refined-verifier agreement and (ii-c) training-side sensitivity. Definitions and measured values: `protocols/v2_T2_locked_v2_0.md` section 4 and `results/v2_T2_locked/v2_0/continuation_check_v2_0.json`; the standard-tier gap is kept as a verifier numerics item (`results/v2_T2_locked/v2_0/verifier_numerics_note.md`).

## Addendum (2026-10-04): pushed state

The "not pushed" wording in this summary (and in `docs/STATE.md`, R1 section) was true on 2026-10-03. The branch has since been pushed: `origin/v2-t2-refine` is at `155cdec` (`git rev-parse origin/v2-t2-refine`), the head of this round. The original lines are unchanged.
