# T3-R2, summary

Branch `t3-r2` (from `origin/t3-r1` `be2b7e9f`). Prompt: `pi_record/27_t3_r2_prompt.md`. **STOP** after the push: the next round is for the PI to decide.

## Outcome

1. **The audit found no implementation error** (`01_audit.md`). The four failing T3-R1 `bw` runs re-run bit for bit; the 40 snapshots and the 8 continuation-table pairs are what the code builds; an independent best-response calculator (no verifier or metrics numerics) reproduces the stored gains of the 200 runs to at most 2.5e-5 ΔW (G), 5.5e-6 (Δ_T) and 1.0e-4 (EXP_root), with no gate flip and no flip of H1, H2, H3, H1-T2; the refined verifier tier changes no verdict. The four failures and the seven near-zero stage-2 stretches of T3-R1 are learning failures: at all 126 stretch nodes the one-step best-response effort is above 1, Q increases at the learned effort and has no local maximum near zero. An independent review of the code (0 MAJOR, 9 MINOR) led to a second code commit before the lock (`03_checks.md`).
2. **The confirmation fails the pre-registered rule** (`04_confirmation.md`). The locked `bw` (T3-R1's settings, unchanged; protocol `protocols/v2_T3_locked_v1_0.*`, LOCK record) on the fresh seeds 60501-60520 at q = 50 and 60 (40 runs): **34 of 40 pass** (q = 50: 16 of 20, exact 95 % interval [0.563, 0.943]; q = 60: 18 of 20, [0.683, 0.988]; both [0.702, 0.943]). The rule needed at least 36 of 40 and at least 17 of 20 at each q; the total and the q = 50 count are short. The independent recomputation agrees. All 40 runs finished (exit 0), no infrastructure event, every post-launch check passed (C-INIT, C-REPLAY, budgets, schedules, config identity; one invocation detail in `04_confirmation.md` section 1).
3. **The six failures are all at stage 3**: four at q = 50 (seeds 60501, 60506, 60513 exceed η_3 = 0.005 by 0.0002 to 0.002; 60515 has η_3 = Gmax_full = 0.0114 and RMSE 0.0667) and two at q = 60 (60516, RMSE only, 0.0538 against 0.05; 60518, a stage-3 policy that never formed: learned e_3(0) 8.2 against 58.3, Gmax_full = 0.145, the largest of the 80 `bw` runs of the two rounds). None is a stage-2 stretch; three q = 50 runs have a short stretch at the band edge, none at q = 60.
4. Descriptive comparison with T3-R1's 40 `bw` runs (36 of 40): pooled 70 of 80 pass (exact interval [0.782, 0.938]); the gate values are close in the median and worse in the tail in the confirmation; at q = 50, 17 of 20 runs have η_3 above 0.005 at one or more of the last 16 stage-3 exports (T3-R1: 14).

## Not done, by design

No tuning, no second wave, no other arm, no remedy, no change of a setting, gate or threshold; the mechanism of the stage-3 failures is not tested; T = 4 and the paper's figures are out of scope. No tag, no pull request, `main` untouched.

## Where things are

| what | path |
|---|---|
| the G1 audit report | `01_audit.md` (tables `tables/a*.md`, data `results/t3_r2/audit/`) |
| the pre-registration (D4, D5 verbatim, operating characteristic, 40 configs, lock drafts) | `02_preregistration.md`, `confirmation_plan.csv`, `lock_draft/` |
| tests, review and dispositions | `03_checks.md` |
| the confirmation | `04_confirmation.md` (tables `tables/c*.md`, data `results/t3_r2/confirmation/`) |
| handoff for the coworker (Chinese) | `handoff.md` |
| fact-check ledger | `pi_record/01_factcheck.md` |
| code | `tools/t3/t3r2_*.py`, `tools/t3/t3c_*.py`; tests `tests/test_t3r2_*.py` |

Commands (inside tmux, single-threaded: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, absolute paths for `--root`): `t3r2_audit.py <step>` (A1-A8), `t3r2_make_lock.py --out protocols/`, `t3c_launch.py --workers 40 --code-commit be2b7e9fe3e0aad34f7481a1d8306af7cc2963e7`, `t3c_checks.py --root <abs> --plan reports/t3/r2/confirmation_plan.csv --commit <launch HEAD> --out ...`, `t3c_eval.py`, `t3c_analysis.py --diagnose`, `t3c_blind.py --analysis-dir`; the tables by `report_scripts/audit_tables.py`, `confirmation_tables.py`, `assemble_audit.py`, `assemble.py`.
