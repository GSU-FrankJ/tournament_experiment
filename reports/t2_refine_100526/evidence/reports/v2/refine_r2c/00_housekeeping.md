# R2c, P0: housekeeping, pushes and the new branch

Date: 2026-10-05. Written as the steps were run; commands and outputs are verbatim unless a line says otherwise. The round's prompt is cited as `§#` / `D#`. Branch `v2-t2-r2c`, worktree `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-r2c`, base `62ecc436b80d15e4740f310e2e6c72c708c1624c` (head of `v2-t2-r2b`).

## Status

| step | status |
|---|---|
| §1.1 push `v2-t2-r2b` | done (fast-forward not applicable: the branch did not exist on `origin`; plain push, no force) |
| §1.2 fast-forward `main` | **skipped by the rule of §1.2**: `git status --porcelain` in the primary checkout shows a tracked modification (` M SESSION_STATE.md`); nothing was stashed, reset, merged or pushed for `main` |
| §1.3 D1 waivers and acceptances | recorded below |
| §1.4 branch and worktree | done |

## 1.1 Push `v2-t2-r2b`

Before the push: `git rev-parse v2-t2-r2b` = `62ecc436b80d15e4740f310e2e6c72c708c1624c`; `git rev-list --count origin/v2-t2-refine..v2-t2-r2b` = 20 (the 20 R2b commits on top of `b55d389`); the R2b worktree has no tracked modification (`git status --porcelain --untracked-files=no` empty); no `.pt`, `.pth`, `.pyc`, `.npz`, `train_history` or weight export is added in the range (`git diff --name-only --diff-filter=A b55d3890 v2-t2-r2b` filtered for those patterns is empty); `git diff --stat b55d3890 v2-t2-r2b | tail -1`: `1413 files changed, 336758 insertions(+), 52 deletions(-)`.

```
$ git push origin v2-t2-r2b
remote: 
remote: Create a pull request for 'v2-t2-r2b' on GitHub by visiting:        
remote:      https://github.com/GSU-FrankJ/tournament_experiment/pull/new/v2-t2-r2b        
remote: 
To https://github.com/GSU-FrankJ/tournament_experiment.git
 * [new branch]        v2-t2-r2b -> v2-t2-r2b
$ git ls-remote origin v2-t2-r2b refs/heads/v2-t2-r2b
62ecc436b80d15e4740f310e2e6c72c708c1624c	refs/heads/v2-t2-r2b
```

## 1.2 `main` was not fast-forwarded

The precondition of §1.2, in the primary checkout `/home/fjiang4/tournament_experiment`, verbatim:

```
$ git status --porcelain
 M SESSION_STATE.md
?? MultiStage/Discussion/
?? MultiStage/Plan/
?? experiments/
?? results/different_ability/convergence/different_ability_ppo_q35.0_seed999_r7smoke_std_convergence.json
?? results/different_ability/convergence/different_ability_ppo_q35.0_seed999_r7smoke_v2_convergence.json
?? results/different_cost/convergence/different_cost_ppo_q35.0_seed999_r7smoke_convergence.json
?? results/three_players/convergence/ppo_3p_q35.0_seed999_r7smoke_convergence.json
?? results/two_players/convergence/ppo_q35.0_seed999_r7smoke_convergence.json
?? results/two_players/convergence/ppo_q35.0_seed999_r7smoke_metadata.json
?? results/two_players/convergence/ppo_q35.0_seed999_r9smoke_convergence.json
?? results/two_players/convergence/ppo_q35.0_seed999_r9smoke_metadata.json
$ git diff --stat
 SESSION_STATE.md | 15 +++++++++++++++
 1 file changed, 15 insertions(+)
$ git rev-parse main origin/main t2-v2-main-v2.0
d1b84437d3be32daf40a58d77e044201b7ba09d0
f02a256095dcaa69dffbd011718d337f84183890
53ae8c321a41b00d844c12696cecfedf95eb2ef4
```

` M SESSION_STATE.md` is a tracked modification (15 inserted lines; not made by this session, whose only action in the primary checkout was reading its status and pushing a branch), so §1.2 says to skip the item and report. I did not stash, reset, merge or push anything for `main`; `git fetch`, `git merge --ff-only` and `git push origin main` were not run. State: local `main` = `d1b84437` is an ancestor of `origin/main` = `f02a2560` (`git merge-base --is-ancestor main origin/main` succeeds) and the tag object `t2-v2-main-v2.0` (`53ae8c32...`) peels to `b55d38907b02106a6e874d3b60622e4e5b7c9d5d` (`git rev-parse t2-v2-main-v2.0^{commit}`). A fast-forward is therefore possible once the owner has committed or set aside `SESSION_STATE.md`. Nothing in R2c depends on `main` (the branch is taken from `v2-t2-r2b`).

For the owner, when `SESSION_STATE.md` is dealt with (commands of §1.2, unchanged): in `/home/fjiang4/tournament_experiment`: `git status --porcelain` shows no tracked modification; `git fetch origin`; `git merge --ff-only origin/main`; `git merge --ff-only t2-v2-main-v2.0`; `git rev-parse main` must print `b55d38907b02106a6e874d3b60622e4e5b7c9d5d`; `git push origin main`; `git ls-remote origin main`.

## 1.3 D1 acceptances and waivers (recorded, no action)

- **R2b is closed.** Method 5 (pathwise fine-tuning) is closed as a negative result at matched budgets; `clamp_likelihood` stays `density` (no measurable effect; the likelihood inconsistency stays a recorded known issue); the seed-30510 diagnostic stays descriptive. These three statements go into the v2.1 change log / known issues if a v2.1 is written (§4.2).
- **The report-pack refresh of the R2b prompt §1.2 is waived** (no repair). The builder's failure diagnosis, from `reports/v2/refine_r2b/00_housekeeping.md` section 1.2 and Addendum 2, is carried here as a known issue: `tools/v2/report/build_t2_report_pack.py` does not run on the current code, for two independent causes. Cause 1: R1's commit `32a8c21` changed `Run.lr_windows` from one window per phase to a list of windows per phase, and `tools/v2/report/sec_locked_a._schedule` (line 800) still builds `{phase: window}` (`TypeError: string indices must be integers, not 'str'` in `Run.lr_for`). Cause 2: the builder calls the live entry point's `build_config` with the v1.0 and v1.1 protocols (`sec_locked_a.build_f11`, line 816); since the v2.0 lock the entry point accepts only v2.0-shaped protocols (`KeyError: 'continuation_value_mode'`). Further modules may fail after these two. The pack under `reports/v2/t2_report/` is the v1.1 pack; T05, T56 and T57 are stale.
- **Over-long commit subjects stay** (six of the R2b commits `8db2786`, `1ff99bd`, `e63c2a9`, `4f24cec`, `d581b3c`, `c6c8da6` exceed 72 characters; they are cited by hash and are not reworded).
- **The R2b worktree repair is accepted** (`reports/v2/refine_r2b/00_housekeeping.md`, Addendum).

## 1.4 Branch and worktree

The session was started in a clean harness worktree (`/home/fjiang4/tournament_experiment/.claude/worktrees/r2c-sampler-protocol-v2-1-c4c0f1`, branch `claude/r2c-sampler-protocol-v2-1-c4c0f1` at `f02a2560`, untouched by this round). The branch and worktree of §1.4 were created from there with an absolute path (`git worktree add` takes the path as given), and the session then switched into the new worktree with the harness's `EnterWorktree path=...` (the harness refuses writes to another worktree's files from a session that is not in it):

```
$ git worktree add /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-r2c -b v2-t2-r2c v2-t2-r2b
Preparing worktree (new branch 'v2-t2-r2c')
Updating files: 100% (12876/12876), done.
HEAD is now at 62ecc436 docs: correct the R2b reports and addenda after the verification pass
$ git worktree list | grep r2c
/home/fjiang4/tournament_experiment/.claude/worktrees/r2c-sampler-protocol-v2-1-c4c0f1          f02a2560 [claude/r2c-sampler-protocol-v2-1-c4c0f1]
/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-r2c                                 62ecc436 [v2-t2-r2c]
```

(The progress lines of the checkout are shortened to the final one.)

## Machine at the start of the round

`nproc` = 64; `uptime`: load average 6.98, 6.68, 6.83 (system clock 17:53:34, up 7 days); `df -h /home/fjiang4`: `/dev/md0 5.5T size, 3.2T used, 2.1T available, 61%`; `tmux ls`: no server running. (The load average before the wave launches is recorded again in each launch record; no source is attributed to it.)

## Environment notes of this session (not results)

- A worktree-isolated session refuses some compound shell commands (one that combined a variable assignment, `cd` and a heredoc was refused); scripts are written with the file tool and run by path.
- `pytest tests` only (a bare `pytest` also collects `experiments/*/tests`).
