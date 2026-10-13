proceed P2 — after the addendum below.
Gate G1 decision (PI, 2026-10-13). P1 is accepted as pushed: `origin/ms-r4` at `a4b2580cf6ef21cb9383bbdafc513b3c3e5d51dd`. This includes Addendum 1 (the English rendering of the coworker's note agrees with Appendix A; the `.docx` hash stays unchecked) and the deviations and clarifications of `02_preregistration.md` section 13, items 1-10, as written.
Also accepted:

* The six skipped tests. They read the untracked MS-R1 / MS-R2 pilot arrays, which are not in this worktree, and they test analysis tools this round does not touch. The default paths are covered by C-MS6, C-R7 and the `git archive` tests.
* The review dispositions R4-01 to R4-17 (`03_checks.md` section 3).
* The completeness rule of section 8. A missing run, a nonzero exit or a non-finite point of the 64-point curve makes the affected criterion part False with the status `incomplete`. A gate failure is not missing data: it enters E1-c and E2-b as a failure.

Nothing else changes. The arms, the configs (`reports/ms/r4/run_config_hashes.csv`), the criteria and their parameters (`reports/ms/r4/prereg_criteria.json`, SHA-256 `9b8b1a5558618aa297cad8bf99cb20d0e1df4602d6cba5a162248abdf3632393`), the decision order and the predictions stand.
A1. Before the launch
Each step is a stop-and-report if it fails.

1. Save this reply verbatim as `reports/ms/r4/pi_record/24_g1_reply.md`.
2. Append a section "Addendum 2 (2026-10-13): the PI's decision at gate G1" to `reports/ms/r4/02_preregistration.md`. Do not edit existing lines. The addendum records:
   * that P1 was accepted at `a4b2580c`;
   * the items accepted above;
   * A2 below, verbatim;
   * the result of step 3;
   * the launch commit, i.e. the commit that adds this addendum.
3. Config gate. Regenerate the plan of wave `r4` with `tools/ms/r4_launch.py --plan-csv`, written to a scratch path outside the repository, and compare it with the 280 `r4` rows of `reports/ms/r4/run_config_hashes.csv`. Every run name, output directory and `run_config.json` SHA-256 must be equal. This also confirms that the launch happens in the worktree the hashes were computed in (`.claude/worktrees/ms-r4-e1-e2-experiments-2da78b`).
4. Commit and push. Commit (`reports/ms/r4/` only, a `docs:` commit) and push `ms-r4` as a fast-forward. Then check that `git diff --stat 4c654d34 HEAD -- agents run utils envs protocols` is empty, i.e. the pilot runs the run code of C-MS6 and C-R7.
5. Launch from that commit, on a clean tree, in tmux, single-threaded:
`tools/ms/r4_launch.py --wave r4 --params reports/ms/r1/prereg_parameters.json --workers 40 --code-commit <the addendum commit>`

A2. The confirmation rule of a possible lock (pre-registered now, before any pilot result exists)
This rule is not used in this round. It binds a later lock round only if the coworker decides at G2 to lock a scheme. `05_decision_inputs.md` states it next to each outcome.
Design. Fresh seeds 40501-40520 × q ∈ {50, 60}, 20 per q. The locked scheme and its comparator are run in the same wave on the same seeds, paired by (q, seed). The locked v2.0 entry point is run on the same seeds as a descriptive reference (it is not part of the pass rule). Bootstrap as in D6, with the seed of the experiment the scheme comes from (20261013 for E1, 20261014 for E2).
E1-type scheme (`lrsm_st_s16`, as locked; comparator `t1_st_s16`). It passes iff all three hold at both q:

* (i) at least 18 of 20 runs pass all four gates (the rule of v2.0's confirmation; exact Clopper-Pearson intervals reported);
* (ii) the mean |peak error| over the 20 runs is <= 0.04;
* (iii) the 95 % interval of the mean paired difference of |peak error|, scheme - `t1_st_s16`, lies entirely below 0.

E2-type scheme (the arm on which the E2 statement rests, as locked; comparator `fb_a0`). It passes iff E2-a, E2-b and E2-c hold on the fresh seeds and at least 18 of 20 runs of the scheme pass all four gates, at both q.
Combined scheme. It needs its own development run first (Appendix A). Its confirmation rule is fixed together with that run, before the run.
If the confirmation fails, v2.0 stays the T = 2 solver, and E1 and E2 are reported as ablations and limitations (Appendix A).
Then
Continue with sections 3-4 of the prompt unchanged, including the STOP after 4.3.
