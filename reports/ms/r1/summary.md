# MS-R1: summary of the round (P1-P2)

Date: 2026-10-07. Branch `ms-r1`. Status: the pilot is done and reported; **STOP** after the push of section 4.3 of the prompt (`pi_record/17_ms_r1_prompt.md`); the next round is the PI's decision. No fresh seed (40501-40520), lock, confirmation, second pilot or T=3 experiment was used or started, and no parameter changed after the pre-registration except through the PI's reply at gate G1 (`pi_record/18_g1_reply.md`, Addendum 1 of `02_preregistration.md`).

## Reading order

1. This file, then `05_decision_inputs.md` (the arms side by side, what the budget control explains, what remains, what the pilot does not settle).
2. `04_pilot.md` (every table with its regeneration command: the pre-registered criterion and the secondary table, the expectations of the addendum against the outcome, one section per arm, the A3 descriptive tables, stage 1, anomalies, figures).
3. `02_preregistration.md` (design, parameters, arms, criterion; Addendum 1 at the end: the PI's G1 decision, the arm `MS_base2400`, check C-MS2, expectations A2, analysis additions A3, the 120-run launch plan), `01_calibration.md` (what the v2.0 exports say about the rule), `03_checks.md` (C-R4, C-MS1, tests, launch checks, C-MS2, analysis), `00_housekeeping.md`.

## Headline numbers (paths: `results/ms_r1/analysis/`)

- **Runs.** 120 pilot runs (five rule arms and the budget-matched control `MS_base2400`, 20 each), all exit 0, launch checks 120 of 120, **C-MS2 20 of 20** (`MS_base2400` equals `parents_A` bit for bit through update 1201, its u1225 export differs), blind recomputation of the primary and the secondary criterion tables agrees (255 numbers, 1e-12). `pilot/launch_20261007_055045.json`, `pilot/launch_checks.json`, `analysis/blind_recomputation.txt`.
- **Pre-registered criterion** (`criterion.csv`): **no arm meets it.** The four sampler arms meet part (a) at q = 50 only (means -0.0260 / -0.0227 / -0.0277 / -0.0313 for `MS_s25a0` / `MS_s25a5` / `MS_s35a0` / `MS_s35a5`, intervals below 0); at q = 60 every interval contains 0 (means -0.0076 to -0.0162). `MS_rule` meets neither (-0.0098, +0.0019). `MS_s35a0` also violates part (b) in one run (q = 50 seed 10508: eta_2/DW 0.005054 against 0.005).
- **Budget control** (`MS_base2400` against `parents_A`): -0.0126 [-0.0285, +0.0014] (q = 50), -0.0085 [-0.0193, +0.0041] (q = 60). The control reproduces 40-55 % of the mean q = 50 improvement of the sampler arms and 52-112 % of the q = 60 one (a ratio of means; `05_decision_inputs.md` section 2).
- **Secondary table** (`criterion_vs_MS_base2400.csv`; rule arms minus the control): no arm meets part (a) at either q; mean differences -0.0101 to -0.0188 (sampler arms, q = 50), +0.0028 (`MS_rule`); -0.0077 to +0.0103 at q = 60; all intervals contain 0.
- **Stop rule.** 0 of 100 rule-arm terminal-stage runs stopped before the cap at rho_2 = 0.05 (2000 training updates + 400 landing in all 100); R_2 <= 0.05 held at 1 of 4000 checks (q = 50) and 91 of 4000 (q = 60), at most 2 consecutive. The localized branch (polishing) was reached at q = 60 (26-34 of 50 classifications per arm, polishing in 9-10 of 10 runs) and rarely at q = 50 (1-6 of 50). As expected in Addendum 1, A2.
- **Peak error.** |peak error| <= 0.05 in 7 / 12 / 10 / 14 / 14 / 15 / 16 of 20 runs for `MS_base` (= `parents_A`), the control, `MS_rule`, `MS_s25a0`, `MS_s25a5`, `MS_s35a0`, `MS_s35a5`; the signed peak error is negative in every arm and at both q (intervals exclude 0). R0 = r_2(0)/s_2 tracks the peak: |peak|/R0 has median 1.66-1.68 (q = 50) and 1.46-1.47 (q = 60) in every arm, the calibration's 1.65 / 1.45 (linearised factor 1.700 / 1.486).
- **Gates.** G-A and the eta part of G-N hold in 139 of 140 terminal-stage runs; all 140 stage-1 runs pass G-S, G-F and G-N(Gmax). The tail mean is below 0.02 in every run (largest 0.0125).

## Deviations and open items

1. **Branch / worktree.** P2 ran in the session worktree `p2-gate-ms-base2400-e41857` (branch fast-forwarded onto `ms-r1`, pushed with `git push origin HEAD:ms-r1`); the local ref `ms-r1` of the P1 worktree is behind `origin/ms-r1` (owner-side; `00_housekeeping.md` addendum 2.1). The primary checkout's `main` was not touched; no tag was created.
2. **Code commit `9b719715`** (authorised by A1; `tools/ms/` and tests only); the run code is that of `c95a2af4` (the launch record's `diff_stat_run_code_to_head` is empty).
3. **Analysis roots.** `MS_base` is read from the P1 worktree's `results/ms_r1/base` (the freeze arrays are untracked and live there); the roots are recorded in `analysis/analysis_info.json`.
4. **`MS_base`'s terminal-stage would-fire record** is at rho_2 = 0.03 (the P1 launch without `--params`; recomputed in `03_checks.md`); training unaffected. The `MS_base2400` and the rule arms were launched with `--params`.
5. **Reviewer's scratch run** (`MS_base2400` q = 50 seed 10501, outside `results/`) is not part of any result.
6. **Not tracked in git** (as pre-registered): `.pt` files, `train_history.json`, weight exports, freeze arrays (`freeze_stage*_*.npz`), `band_sweep.npz`, `run.log`; they stay in `results/ms_r1/pilot/` (and the P1 worktree for the base wave).
7. **Open for the PI** (`05_decision_inputs.md` section 6): whether any mechanism improves the peak at matched budget (no interval of the secondary table excludes 0 on |peak|; about 27-54 seeds at q = 50 would be needed at the observed effect sizes, a rough optimistic arithmetic); the effect of polishing at a given sampler is not separated (no arm has the sampler without the polishing branch; at q = 50 polishing blocks occurred in only 1-4 of 10 runs per sampler arm); the stop would need another rho to fire (untested); more budget than 2400 is untested; T = 3 untested.

## Commands that reproduce every table

```
# the runs (already done; do not repeat: output directories refuse to be reused)
python tools/ms/launch_ms_r1.py --wave pilot --params reports/ms/r1/prereg_parameters.json --workers 40 --code-commit c95a2af4
# checks (exit 0 iff every check and C-MS2 pass)
python tools/ms/launch_checks.py --root results/ms_r1/pilot --arms MS_rule MS_s25a0 MS_s25a5 MS_s35a0 MS_s35a5 MS_base2400 \
    --code-commit f969d55026b59edaaaac34be633b42b447f9af3c --out results/ms_r1/pilot/launch_checks.json \
    --cms2-ref <v2-t2-refine>/results/v2_refine/parents_A --cms2-arm MS_base2400
# analysis (writes results/ms_r1/analysis/*.csv, figures/, summary.txt, analysis_info.json) and the blind recomputation
python tools/ms/r1_analysis.py --base-root <P1 worktree>/results/ms_r1/base --pilot-root results/ms_r1/pilot \
    --parents-root <v2-t2-refine>/results/v2_refine/parents_A --rehearsal-root <v2-t2-refine>/results/v2_T2_locked/rehearsal_v2_0 \
    --calibration-root results/ms_r1/calibration --out results/ms_r1/analysis
python tools/ms/blind_criterion.py --analysis-dir results/ms_r1/analysis
# the tables of the reports (one block per table, `--list` names them; arm sections: --block arm:<arm>)
python reports/ms/r1/report_scripts/pilot_tables.py --list
```

`<v2-t2-refine>` is `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine`, `<P1 worktree>` is `/home/fjiang4/tournament_experiment/.claude/worktrees/ms-r1-multistage-development-afebf2`. The tables of `04_pilot.md` (sections 2, 4, 5.1-5.6, 6) and of `05_decision_inputs.md` (sections 1, 2, 6) are generated by the `pilot_tables.py` blocks named under them and need only the CSVs of `results/ms_r1/analysis/`; the prose and the two hand-written tables of `04_pilot.md` (sections 1 and 3) are not generated.
