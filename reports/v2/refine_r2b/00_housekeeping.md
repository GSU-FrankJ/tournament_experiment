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

## Addendum (2026-10-04, after the P0 stop): what section 1.6 actually produced, and the base of `v2-t2-r2b`

The person who owns the round ran section 1.6 in a terminal after the stop above. What the repository showed afterwards (`git ls-remote origin ...`, `git rev-parse`, `git worktree list`, run from the `v2-t2-refine` worktree):

```
b55d38907b02106a6e874d3b60622e4e5b7c9d5d	refs/heads/v2-t2-refine
f02a256095dcaa69dffbd011718d337f84183890	refs/heads/main
32e68cfd117372b76edff70c2545da02434c539a	refs/tags/t2-v2-lock-v2.0        (^{} = 1d6d4d00736b265a18ae71b91ecb77b19e4915b9)
4724ca007e7ebcb0fe2d0e1a95ce11b9b483fb87	refs/tags/t2-v2-confirmation-v2.0 (^{} = d2e377d0da702e8e162662f74576d1bff5b42a56)
53ae8c321a41b00d844c12696cecfedf95eb2ef4	refs/tags/t2-v2-main-v2.0        (^{} = b55d38907b02106a6e874d3b60622e4e5b7c9d5d)
```

- Pushed: `v2-t2-refine` (`b55d389`) and the three tags, each peeling to the intended commit.
- **Not done:** `main` was not fast-forwarded: `origin/main` is still `f02a256` and local `main` is still `d1b8443` (76 commits behind `b55d389`; `f02a256` is an ancestor of `b55d389`, so a fast-forward is possible).
- The worktree `v2-t2-r2b` had been created inside the `r7-canonical-reference-checks-bd0e75` worktree and on branch `v2-t2-r2b` = `d1b8443`, which contains none of the v2.0 code. The branch had no commits of its own (`git rev-list --count b55d389..v2-t2-r2b` = 0).

Decision of the owner (asked in the session): repair it from this session, without touching `main` (local or remote). Done:

```
git worktree remove /home/fjiang4/tournament_experiment/.claude/worktrees/r7-canonical-reference-checks-bd0e75/.claude/worktrees/v2-t2-r2b      (no --force: it was clean)
git worktree add -B v2-t2-r2b /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-r2b b55d389     (branch reset from d1b8443 to b55d389)
```

So `v2-t2-r2b` starts at `b55d389`, the commit `main` will have after the fast-forward (the tree is identical). `main` stays for the owner to fast-forward and push; nothing in R2b depends on it. The pack refresh of 1.2 remains undone.

## Addendum 2 (2026-10-04, after the R2b pilots, from the conformance audit against the prompt)

Nothing above was changed. This addendum closes three record gaps of section 1.4 / the preamble ("every command, hash and `ls-remote` output") and states what is still open for the owner.

1. **The 1.4 commit.** `b55d38907b02106a6e874d3b60622e4e5b7c9d5d` (`docs: record v2.0 publication status and the R2b P0 housekeeping`, 2026-10-04T06:17:13+00:00, parent `e89b61d`); it is the head of `v2-t2-refine` and `origin/v2-t2-refine`, and the target of the tag `t2-v2-main-v2.0` (the placeholder "the commit that adds this file" in the tag description resolves to it). Its `git diff --stat e89b61d b55d389`, verbatim:

```
 docs/STATE.md                            |  2 +
 reports/v2/README.md                     |  4 ++
 reports/v2/protocol_v2_0_confirmation.md |  4 ++
 reports/v2/refine_r2b/00_housekeeping.md | 75 ++++++++++++++++++++++++++++++++
 reports/v2/summary.md                    |  4 ++
 5 files changed, 89 insertions(+)
```

   There are no deletions: the four addenda are pure insertions. `git diff --stat f02a256 b55d389 -- reports/v2/t2_report tools/v2/report` is empty (the pack and its builder are the v1.1 originals).
2. **Commands that were not logged verbatim.** The scratch-copy command of the pack rebuild (an `rsync` into a scratch directory) and the tmux session of the rebuild were not recorded literally when they were run, and are not reconstructed here (a reconstructed command would be a false record). Their effect is recorded: the two tracebacks above, and that the tracked pack and builder are unchanged.
3. **State on 2026-10-04 (`git rev-parse`, `git rev-list --count`).** `main` = `d1b84437d3be32daf40a58d77e044201b7ba09d0` locally and `f02a256095dcaa69dffbd011718d337f84183890` on `origin`; `v2-t2-refine` = `b55d38907b02...`; `f02a256..b55d389` is 24 commits and `d1b8443..f02a256` 52.
4. **Still open, owner-side (no experiment and no locked file involved).**
   - **Section 1.6, `main`:** in `/home/fjiang4/tournament_experiment`: `git status --porcelain` shows no tracked modification; `git fetch origin`; `git merge --ff-only origin/main`; `git merge --ff-only v2-t2-refine`; `git rev-parse main` must print `b55d38907b02106a6e874d3b60622e4e5b7c9d5d`; then `git push origin main` (fast-forward `f02a256 -> b55d389`, no force) and `git ls-remote origin refs/heads/main`. `v2-t2-r2b` needs no rebase (its base is `b55d389`).
   - **Section 1.2, report-pack refresh:** either waived (the prompt itself keeps the pack the v1.1 pack and defers a v2.0 pack) or a builder repair authorised beyond "change nothing else": the rename in `sec_stage1.py`; the learning-rate windows as `{phase: [window]}` in `sec_locked_a._schedule`; the v1.0 / v1.1 protocols evaluated through a pinned v1.x entry point and `L.load_protocol()` pinned to the v1.1 protocol (the call at `sec_locked_a.py:811` now returns the v2.0 protocol); the allowed-diff rule widened to the module-hash cells of the 11 `manifest.csv` rows that carry the `sec_stage1.py` hash; rebuild at a settled head (T05 counts `reports/v2/*.md` and T57 hashes `docs/STATE.md`, both of which R2b changed).
