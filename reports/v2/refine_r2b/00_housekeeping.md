# R2b, P0: publication housekeeping for v2.0

Date: 2026-10-04. Worktree `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine`, branch `v2-t2-refine`, starting head `e89b61d` (`git rev-parse HEAD`: `e89b61d5e70ccadc65ddfd6fb2c8665bd4216904`).

## Status

| step | status |
|---|---|
| 1.1 dated addenda in four files | done |
| 1.2 report-pack refresh (T05, T56, T57) | **not done**: the builder does not run on the current code (two independent causes below); the pack is unchanged and my builder edit was reverted |
| 1.3 locked files untouched | done (nothing in `protocols/`, the entry point, `utils/v2_continuation.py`, the analysis script or any pipeline code changed) |
| 1.4 commit | done (the commit that adds this file) |
| 1.5 tags | created locally after that commit (commands below); not pushed |
| 1.6 fast-forward `main`, push, create `v2-t2-r2b` | **not done**: blocked, see below |

## 1.1 Addenda

Appended (original lines unchanged; in `docs/STATE.md` inserted as a paragraph before the R1 section) to `reports/v2/protocol_v2_0_confirmation.md`, `reports/v2/README.md`, `docs/STATE.md` and `reports/v2/summary.md`. Text: `origin/v2-t2-refine` is at `e89b61d`; the tags `t2-v2-lock-v2.0` -> `1d6d4d0`, `t2-v2-confirmation-v2.0` -> `d2e377d` and `t2-v2-main-v2.0` -> the commit that adds this file; the fast-forward of `main` and the push are recorded here. The addenda do not claim that `main` was fast-forwarded, because it was not (1.6).

`git ls-remote origin v2-t2-refine main`, before any push of this round:

```
f02a256095dcaa69dffbd011718d337f84183890	refs/heads/main
e89b61d5e70ccadc65ddfd6fb2c8665bd4216904	refs/heads/v2-t2-refine
```

## 1.2 Report pack: why it was not refreshed

Steps taken (nothing in `reports/v2/t2_report/` was written; the build ran in a scratch copy of the working tree without `results/` and `.git`, made with `rsync`, because the builder reads `results/` from `--results-root`):

1. Replaced every `tests/test_v2_locked.py` by `tests/test_v2_locked_v2_0.py` in `tools/v2/report/sec_stage1.py` (10 occurrences; the one prose mention `every test_v2_locked.py case` in a text about the v1.1 pytest log was left as it is).
2. `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PY tools/v2/report/build_t2_report_pack.py --results-root <canonical worktree>/results`, in tmux, in the scratch copy: **exit 1**.
   ```
   File "tools/v2/report/sec_locked_a.py", line 800, in _schedule
       rows = [("A", j, j, Run.lr_for(ns, "A", j)) for j in range(1, cap_a + 1)]
   File "run/run_v2_stagewise.py", line 402, in lr_for
       if int(w["local_first"]) <= local <= int(w["local_last"]):
   TypeError: string indices must be integers, not 'str'
   ```
   Cause 1: R1's commit `32a8c21` changed `Run.lr_windows` from one window per phase to a list of windows per phase; `_schedule` still builds `{phase: window}`. (`git show f02a256:run/run_v2_stagewise.py` has the old `lr_for`.)
3. To see whether the rename alone would otherwise be enough, a one-line shim (`{phase: [window]}`) was applied **in the scratch copy only** and the build repeated: **exit 1** again.
   ```
   File "tools/v2/report/sec_locked_a.py", line 816, in build_f11
       d_v = max(C.max_abs_diff(sch.lr, _schedule(L, Run, proto10, q)[0].lr) for q in QS)
   File "run/run_v2_T2_locked.py", line 179, in build_config
       "continuation_value_mode": pl["flags"]["continuation_value_mode"],
   KeyError: 'continuation_value_mode'
   ```
   Cause 2: the builder calls the live entry point's `build_config` with the v1.0 and v1.1 protocols; since the v2.0 lock the entry point accepts only v2.0-shaped protocols.
4. Conclusion: the pack builder needs more than the rename (at least a shim for cause 1 and a pinned v1.x entry point for cause 2; further modules may fail after that, since the build stops at the first error). That is outside "change nothing else in the builder". Per the P0 rule, the pack refresh is skipped: no rebuilt pack is committed, and the builder edit of step 1 was reverted (`git checkout -- tools/v2/report/sec_stage1.py`), because a builder change without a rebuild would leave `manifest.csv`'s module hashes stale. T05, T56 and T57 stay stale as before; the pack stays the v1.1 pack.

## 1.3 Nothing else in the locked files

`git diff --name-only e89b61d HEAD` after the commit lists only the four files of 1.1 and this file.

## 1.4 Commit

One commit: the four addenda and this record (message `docs: record v2.0 publication status and the R2b P0 housekeeping`).

## 1.5 Tags (annotated), created after that commit

```
git tag -a t2-v2-lock-v2.0 1d6d4d0 -m "T=2 protocol v2.0 lock commit"
git tag -a t2-v2-confirmation-v2.0 d2e377d -m "T=2 protocol v2.0 confirmation launch commit (seeds 30501-30520)"
git tag -a t2-v2-main-v2.0 <the commit that adds this file> -m "T=2 v2.0 publication head"
```

## 1.6 Fast-forward `main`, push, worktree `v2-t2-r2b`: blocked

This session is isolated to one worktree and its tooling refuses every git operation on the primary checkout `/home/fjiang4/tournament_experiment`:

- `EnterWorktree path=/home/fjiang4/tournament_experiment`: `Cannot enter worktree: /home/fjiang4/tournament_experiment is the main working tree, not a linked worktree.`
- `cd /home/fjiang4/tournament_experiment && git status --porcelain`: `Refusing to run it — a worktree-isolated session's git operations must target its own worktree.` (`git -C <primary>` is refused the same way.)

So the precondition (`git status --porcelain` in the primary checkout shows no tracked modification), `git merge --ff-only origin/main`, `git merge --ff-only v2-t2-refine` and the check `git rev-parse main` == head of `v2-t2-refine` could not be run here, and were not worked around (no script, no redirected git). Nothing was pushed in this step: not `main`, not the tags, not the new housekeeping commit. `origin/main` is still `f02a256`; local `main` in the primary checkout was not touched. Commands for the person who runs it in the primary checkout are in the hand-over message of this round. P1 to P4 need the branch `v2-t2-r2b` from `main` after this step and have not been started.
