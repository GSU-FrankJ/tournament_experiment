# Publish the T=2 v2 work to GitHub `main`, with everything for the T=2 v2 report in one folder

The goal is to put the complete T=2 v2 work on GitHub `main`: every commit from Phase 0 to the report pack, plus the PI-side record (plans, prompts, decisions). Everything a reader needs for the T=2 v2 report should be reachable from one folder, `reports/v2/`.

Everything from earlier rounds still applies:
- no deletion or overwriting of results or checkpoints;
- no change to `protocols/`, the locked entry point, the pipeline code, or the analysis scripts;
- no training.

**This round is git and documentation work only.**

---

## 0. Ground rules

- **Never force-push.** Never rewrite published history. Never push large binaries.
- **Do not touch the original worktree's `main` checkout.** Do the integration on a new branch in your session worktree, then push that branch to `main` as a fast-forward (§3).
- **Stop and report**, without pushing, if any of these happens:
  - a merge conflict;
  - a test or regression result differs from what is stated below;
  - `origin/main` has protections that refuse the push;
  - authentication fails;
  - anything else does not match this prompt.
- **Folder layout.** Do not move or rename existing files under `reports/v2/`, `results/`, `protocols/` or code. Reports, manifests and hashes cite these paths. The single folder is created by adding an index and the PI record, not by moving files.

---

## 1. Pre-flight (read-only), reported before any change

1. **Remote and `main`.**
   - Report `git remote -v`, then run `git fetch origin`.
   - Report `origin/main`'s head and date, and how many commits it is ahead of or behind the merge base with `v2-stagewise-pilots`.
   - List any commits on `origin/main` that are not on `v2-stagewise-pilots`.
2. **Branches to publish.** Record the head of each:
   - `v2-stagewise-pilots`, currently `cb0b541`;
   - the report-pack branch `claude/t2-v2-report-pack-f51a78` (`7062b5b`, `28e14b5` on base `cb0b541`).
   - Confirm that the second fast-forwards from the first.
3. **Size and content audit** of everything that would be pushed (`git rev-list origin/main..<integration head>`):
   - the largest blobs, and any file > 10 MB (GitHub refuses > 100 MB);
   - any `.pt`, `.npz` or other binary arrays that are tracked;
   - a secret scan of the added files: tokens, keys, passwords, `.env` files, SSH material.
   - If anything large or sensitive is tracked, stop and report it. Do not rewrite history on your own.

---

## 2. Integrate (local only)

1. **Bring the report pack onto the v2 branch.** Fast-forward `v2-stagewise-pilots` to `claude/t2-v2-report-pack-f51a78`. Use the `--ff-only` command you gave earlier, or the equivalent in a worktree you are allowed to write to. If it is not a fast-forward, stop.
2. **Create the integration branch** `release/t2-v2`, starting from `origin/main`.
3. **Merge** `v2-stagewise-pilots` into it with a merge commit (`--no-ff`). The message summarizes the T=2 v2 work and lists:
   - the lock commits `4bd2214` (v1.0) and `431474d` (v1.1);
   - the confirmation launch `f6838ec`;
   - the report pack commits.

   On any conflict, abort the merge and stop.
4. **Add the PI record.**
   - The PI has supplied `t2_v2_pi_record.zip` at `<PATH_GIVEN_BY_PI>`.
   - Unzip it verbatim into `reports/v2/pi_record/`.
   - Run `sha256sum -c SHA256SUMS` inside that folder, and report the result. If any line fails, stop.
   - Do not edit its contents.
5. **Add the folder index `reports/v2/README.md`.** It is the entry point for the T=2 v2 report. Contents:
   - **Scope and status:** P0–P6, plus the report pack; the confirmation verdict PASS (20/20 at q=50 and q=60), with its source path.
   - **Reading order:**
     - `pi_record/00_README.md`, for the plans, prompts and decisions;
     - the study reports in order: `phase0_audit`, `phase1_verifier`, `phase2_infra` and `phase2_opening_checks`, `dreach_reach_mask_check`, `pilot1_reward_estimator`, `pilot2_freeze`, `pilot3_continuation_mode`, `phaseA_ext`, `pilot4_stabilization`, `protocol_lock_and_rehearsal`, `protocol_v1_1_confirmation`;
     - `summary.md`;
     - the report pack, `t2_report/README.md`.
   - **Outside this folder** (paths only, nothing copied):
     - the protocols: `protocols/v2_T2_locked*.json|md` and `protocols/LOCK`;
     - the locked entry point and the v2 pipeline code;
     - the pack builder `tools/v2/report/`;
     - the results roots `results/v2_pilots/` and `results/v2_T2_locked/`, saying which files are tracked;
     - the large arrays and checkpoints that are **not** in git, where they live on vector2, and their total size.
   - **Key commits and tags** (§2.7).
   - **Known caveats**, as one line each, pointing to the pack:
     - the `consistency.md` mismatches (42; the old reports were not edited);
     - the 10 `UNKNOWN` values in `gaps.md`;
     - reproducibility ledger: 26 of 27 checks pass. The one failure is v1.0 Check 1's literal criterion, accepted under D1.
     - Pilot 3's actor/critic identity check does not compare optimizer states.
6. **Update `docs/STATE.md`.** It was skipped in the pack round. Add a short T=2 v2 section that points to `reports/v2/README.md`. Change nothing else in it.
7. **Annotated tags** (local first):

   | Tag | Commit |
   |---|---|
   | `t2-v2-lock-v1.0` | `4bd2214` |
   | `t2-v2-lock-v1.1` | `431474d` |
   | `t2-v2-confirmation` | `f6838ec` |
   | `t2-v2-report-pack` | `28e14b5` |
   | `t2-v2-main` | the final integration commit |

   Each tag message states what the commit is.
8. **Checks on the integration head:**
   - **Full test suite.** The only failure allowed is the known `test_registry_canonicalization`, which also fails on `main`. Confirm this on `origin/main` as well.
   - **C7 regression:** must be bit-exact.
   - **Pack rebuild:** `tools/v2/report/build_t2_report_pack.py` must give byte-identical files, apart from the wall-time column of `reevaluations.csv`.
   - **Locked entry point:** it still accepts the v1.1 protocol hash. Run only its refusal and hash tests, not a training run.

   Commit the documentation changes (steps 4–6) as one commit on `release/t2-v2`.

---

## 3. Publish

1. **Final pre-push report.** Print:
   - `git log --oneline origin/main..release/t2-v2 | wc -l`, plus the first and last 10 commits;
   - `git diff --stat origin/main release/t2-v2 | tail -1`;
   - the size audit result.
2. **Push `main` as a fast-forward:** `git push origin release/t2-v2:main`. Without `--force`, this succeeds only as a fast-forward of `origin/main`. If it is refused, stop and report; do not retry with force.
3. **Push the history branches and the tags:** `v2-stagewise-pilots`, `claude/t2-v2-report-pack-f51a78`, and the five tags. Do not push `release/t2-v2` separately; it now equals `main`.
4. **Verify after the push:**
   - `git fetch origin`; `origin/main` equals the integration head;
   - `git ls-remote --tags origin` lists the five tags;
   - `reports/v2/README.md` and `reports/v2/pi_record/00_README.md` exist at `origin/main`.

---

## 4. Report back and STOP

Report:
- the pre-flight findings;
- the integration and merge commits;
- the test, C7 and rebuild results;
- the pushed refs and the verification output;
- the large files kept out of git, with their location on vector2 and total size, as a backup note.

Then **STOP**. Do not start any new experiment or report text.
