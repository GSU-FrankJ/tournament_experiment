# R2c sampler pilot (wave S): summary

Branch `v2-t2-r2c` (local only, see "Deviations and open items"), development seeds 10501-10510 x q in {50, 60}, 2026-10-05. The round piloted four start-distribution arms (the share of the stage-2 exploring starts in the peak set and the update from which it applies), paired with R1's `parents_A`, and applied the pre-registered selection rule mechanically.

**Outcome: no arm is selected, so the round stops here, as D3 and §3 prescribe.** Every arm meets criterion part (b) (no run that passed G-A and its G-N part under the baseline fails it under the arm), and no arm meets part (a) at both q: at q = 60 the 95% bootstrap CI of the mean paired difference of the absolute signed peak error contains 0 for all four arms. The rule was not relaxed and no arm was added. **No protocol v2.1, no lock, no re-rehearsal, no confirmation was started; the seeds 40501-40520 were not used; `protocols/`, the v2.0 entry point, `utils/v2_continuation.py`, the v2.0 analysis script, `envs/` and `agents/` are unchanged** (`git diff --stat 62ecc436 HEAD` of those paths is empty). The next round is the owner's to decide.

## Reading order

| file | content |
|---|---|
| `00_housekeeping.md` | P0: the `v2-t2-r2b` push, why `main` was not fast-forwarded, the D1 waivers, the new branch |
| `01_preregistration.md` | arms, criterion and selection rule with the choices that make it mechanical, v2.1 / G-P / pass rule (not reached), the R2b design basis, the verification state at the code commit |
| `02_pilot_waveS.md` | checks first (launch checks, C-R3), provenance, completeness, one section per arm, cross-arm tables, anomalies, commands |
| `03_selection.md` | the rule, the inputs, the ranking (empty), the outcome |

Data: `results/v2_refine_r2c/` (`waveS/` runs and launch records, `v20_reproduction/` and its check, `analysis/` tables and `selection.json`, `launch_checks.json`, `review/` journals, `analysis_validation_r2b/`). Heavy files (checkpoints, weight exports, `train_history.json`, D1 buffers) are on this machine and untracked.

## What was checked

- **Code and tests before any run.** `start_weights.local_first` (draws and the `start`-stream consumption before it are those of the locked sampler); full suite on the code-commit tree 567 passed, 1 failed (the known `test_registry_canonicalization`), 2 xfailed; C7 bit-exact; the defaults equal the v2.0 lock commit's code and the R2b code commit's sampler (`01_preregistration.md` section 10). An independent review (five lenses, 21 agents) found no blocker; the minor findings were fixed before the code commit.
- **C-R3 (§2.3): the unchanged v2.0 entry point run from this branch reproduces `rehearsal_v2_0` in 20/20** (reference root passed explicitly, 20/20 exit 0, manifests at the code commit with `dirty: false`; `results/v2_refine_r2c/v20_reproduction_checks.json`).
- **Wave S: 80 of 80 runs exit 0** (wall time 179-266 s, median 208 s; 40 single-threaded workers on 64 logical CPUs; launch 2026-10-05 18:58 UTC, load average at the start 8.78 / 15.02 / 12.45 and at the end 27.25 / 34.71 / 24.2, free disk 2.23 TB; `results/v2_refine_r2c/waveS/launch_20261005_185758.json`). The wave ran at the results-only commit `3ad1b07d` that follows the code commit `58c26716`: all 80 manifests record `3ad1b07d`, `dirty: false`, and the `start_weights` of the arm table including `local_first`.
- **Post-launch checks 80/80** (`results/v2_refine_r2c/launch_checks.json`): manifest commit and dirty flag, config equal to the on-disk R1 `parents_A` config in `start_weights` only, manifest keys equal to the arm table, status. **The D2 prefix identities hold in 20/20 for each late arm:** for `A_peak50_late400` the training-relevant state in `state_u01200.pt` and every weight export up to u1200 are bit-identical to `parents_A` (the first differing export is u1225 in 20/20, and the 1200 per-update rows of `v2_updates.csv` are identical); for `A_peak50_late800` every export up to u0800 is identical (first differing export u0825 in 20/20, 800/800 rows). `A_peak35` and `A_peak40` differ from the first export (u25), as constructed. (The three process-global RNG states in `state_u01200.pt` differ from `parents_A`'s in 20/20: the stagewise runner does not seed them, and the training-relevant-state definition of v1.1 / v2.0 excludes them; the check reports them as informational.)
- **The selection was recomputed independently** from `per_run.csv` by a script that does not import the analysis tool (bootstrap, criterion, counts, rule): all four arms' means, CIs, counts and (b) verdicts agree with the tool to 1e-12, selected arm none (`results/v2_refine_r2c/analysis/blind_recomputation.txt`). Before the launch the tool had reproduced R2b's committed tables for `A_peak25` / `A_peak50` exactly (`analysis_validation_r2b/`).

## Result (arm - `parents_A`, absolute signed peak error; negative = better; source `results/v2_refine_r2c/analysis/selection_inputs.csv`, `criterion.csv`, `tail.csv`)

| arm | q = 50 mean [95% CI] | q = 60 mean [95% CI] | (a) | (b) | runs with \|peak error\| <= 0.05: q50 + q60 | mean \|peak error\| (20 runs) | largest tail mean / e2*(0), q50 / q60 (G-A limit 0.02) |
|---|---|---|---|---|---|---|---|
| `A_peak35` (0.35, from 1) | -0.03081 [-0.04864, -0.01355] | -0.00540 [-0.01639, 0.00714] | not met (q60) | holds | 8 + 5 = 13 | 0.04137 | 0.01601 / 0.01937 |
| `A_peak40` (0.40, from 1) | -0.02352 [-0.03973, -0.00771] | -0.00652 [-0.01872, 0.00654] | not met (q60) | holds | 7 + 5 = 12 | 0.04445 | 0.01670 / 0.01775 |
| `A_peak50_late400` (0.50, from 1201) | -0.01430 [-0.03025, 0.00053] | -0.00587 [-0.01786, 0.00583] | not met (q50, q60) | holds | 5 + 5 = 10 | 0.04939 | 0.01279 / 0.01442 |
| `A_peak50_late800` (0.50, from 801) | -0.01778 [-0.03352, -0.00133] | 0.00017 [-0.01559, 0.01509] | not met (q60) | holds | 7 + 1 = 8 | 0.05067 | 0.01914 / 0.01836 |
| baseline `parents_A` | | | | | 2 + 5 = 7 | 0.05947 | 0.01187 / 0.01254 |
| R2b `A_peak50` (0.50, from 1; reference, not eligible) | -0.02346 [-0.04166, -0.00369] | -0.01176 [-0.02269, -0.00119] | met | violated (3 runs, q60) | 8 + 8 = 16 | 0.04186 | 0.01645 / 0.02214 |

The R2b reference row is re-bootstrapped with this round's seed 20261005 (point estimates identical); R2b's own intervals, with its seed 20261004, are -0.0235 [-0.0414, -0.0038] (q = 50) and -0.0118 [-0.0227, -0.0010] (q = 60) (`results/v2_refine_r2b/analysis/criterion.csv`).

Observations (labelled as such; each restates a table, none is a pre-registered test):

- **q = 50: three of the four arms meet part (a)** (`A_peak35`, `A_peak40`, `A_peak50_late800`); `A_peak50_late400` misses it by 0.0005 on the upper bound. The number of runs within the 0.05 target rises from 2 (baseline) to 8, 7, 5, 7.
- **q = 60: no arm's interval excludes 0.** The point estimates are -0.0054, -0.0065, -0.0059 and +0.0002 (8, 6, 5 and 4 of the 10 seeds improve), and the number of runs within 0.05 stays at the baseline's 5 for the three arms that start at update 1 or 1201 and falls to 1 for `A_peak50_late800`. The baseline's q = 60 mean error (0.0534) is already smaller than at q = 50 (0.0656), and the q = 60 intervals are wide (half-widths 0.012-0.016 against point estimates of 0.0002-0.0065): an interval containing 0 does not show that an arm has no effect at q = 60.
- **Part (b) holds for all four arms.** No run that passed G-A with its G-N part under the baseline fails it under any arm. The largest G-A tail mean / e2*(0) per run is 0.0194 (`A_peak35`, q = 60) and 0.0191 (`A_peak50_late800`, q = 50) against the limit 0.02, i.e. 3.2% and 4.3% below it; `A_peak50_late400` stays at 0.0144 or below. For comparison R2b's `A_peak50` reached 0.0221 and the baseline 0.0125.
- **Share response at q = 60 (descriptive rows, R2b's `A_peak25` and `A_peak50` included):** the point estimate of the paired difference moves with the share (0.25: -0.0012, 0.35: -0.0054, 0.40: -0.0065, 0.50: -0.0118) while the largest tail mean over the seeds is 0.0170, 0.0194, 0.0178, 0.0221; at q = 50 the estimates are -0.0119, -0.0308, -0.0235, -0.0235. Whether this is a trade-off or noise at 10 seeds cannot be decided from these tables.
- **Timing:** restricting the share-0.50 draw to the last 400 or 800 updates does not reproduce the effect of the full schedule (q = 50: -0.0143 and -0.0178 against R2b's -0.0235; q = 60: -0.0059 and +0.0002 against -0.0118); the largest tail mean is 0.0128 / 0.0144 for `A_peak50_late400` (below R2b's `A_peak50` 0.0165 / 0.0221 at both q) and 0.0191 / 0.0184 for `A_peak50_late800` (above it at q = 50, below it at q = 60).
- **Anomalies** (`02_pilot_waveS.md` section 6): none for the baseline, `A_peak35`, `A_peak40`, `A_peak50_late400`; `A_peak50_late800` q60 seed 10506 has a positive signed peak error (+0.000938, i.e. 0.09% above e2*(0)) and a smoothed-game share that is not meaningful (observed d = 0 gap -0.0547).

What this round does not show: whether any share or timing would meet the criterion with more seeds, or under a different rule; those are not tested and were not tried (D3).

## Deviations and open items

1. **§1.2 (`main`) skipped, by the rule of §1.2.** The primary checkout shows ` M SESSION_STATE.md` (a tracked modification; the file was not touched by this round). `main` is `d1b84437` locally, `origin/main` `f02a2560`; nothing was fetched, merged or pushed for `main`. The commands to close it are in `00_housekeeping.md`.
2. **Not pushed.** `v2-t2-r2b` was pushed (§1.1). The prompt authorises a push of `v2-t2-r2c` only after the confirmation report (§4.6, which is part of P3); P3 was not reached, so `v2-t2-r2c` exists only locally (head: the commit that adds these reports). No tag was created. To publish it: `git push origin v2-t2-r2c` from this worktree.
3. **Order of commits (a choice recorded in the pre-registration).** The analysis tool and the launch-check tool are in the code commit `58c26716`, before any run, and were validated on R2b data; in R1 and R2b the analysis tool was committed after the runs. The wave-S manifests record the results-only commit `3ad1b07d`, not `58c26716`; the launch-check tool accepts a manifest commit whose diff from the code commit touches only `results/` and `reports/v2/refine_r2c/` (the launch record gives both hashes).
4. **Two existing R2b tests were edited** (the stub of `Run.draw_starts` gains the parameter `local`; the set of `WAVE_DIR` keys gains the R2c waves), both because this round changes those things on purpose.
5. **The session started in a different harness worktree** (`claude/r2c-sampler-protocol-v2-1-c4c0f1`, untouched) and switched into `.claude/worktrees/v2-t2-r2c` once it was created, because the harness refuses writes to another worktree's files.
6. **Load.** The 5-minute load average before the wave launch (15.0) is consistent with this round's own C-R3 wave and dry run that had just finished; no other source was identified or ruled out.
7. The known failure `tests/test_registry_canonicalization.py::test_registry_canonicalization` is unchanged; use `pytest tests`, not a bare `pytest`.
8. The R2b report-pack refresh stays waived (D1); the builder's failure diagnosis is carried as a known issue in `00_housekeeping.md`.

## Files

Code: `run/run_v2_stagewise.py` (`local_first`, `Run.draw_starts(stage, n, local)`), `tools/v2/launch_refine.py` (wave `r2c_waveS`, wave `r2c_v20_repro`), `tools/v2/r2c_launch_checks.py`, `tools/v2/refine_r2c_analysis.py`; tests `tests/test_v2_r2c.py` (41), `tests/test_v2_r2c_tools.py` (16), `tests/test_v2_r2c_analysis.py` (24). Reused unchanged: `tools/v2/cr2_compare.py`, `tools/v2/refine_analysis.py`, `tools/v2/refine_r2b_analysis.py`.

## Addendum (2026-10-06): one figure corrected by the fact-check of the publication

Added by the publication `reports/t2_refine_100526`; nothing above was changed. The independent fact-check of that publication (`reports/t2_refine_100526/pi_record/01_factcheck.md`) recomputed the figures of the observation "Part (b) holds for all four arms" (above) from `results/v2_refine_r2c/analysis/tail.csv`: the largest G-A tail mean / e2*(0) of `A_peak35` at q = 60 is 0.019374, which is 3.13% below the limit 0.02, and that of `A_peak50_late800` at q = 50 is 0.019143, which is 4.29% below it. The text above says "3.2% and 4.3%". The tail means themselves (0.0194, 0.0191) and the conclusion (below the limit at both q) are unchanged.
