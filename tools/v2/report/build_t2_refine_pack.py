#!/usr/bin/env python3
"""Build the evidence pack of ``reports/t2_refine_100526`` (R1, v2.0, R2b, R2c publication).

Every item is read from its source path at the current commit. Small files are copied into
``<out>/evidence/<source path>`` (original relative path kept) and figures into
``<out>/figures/<id>_<name>``; for each copy the SHA-256 of the copy is checked against the source on disk
and, for tracked files, against the blob at ``HEAD``. Large (>= 1 MiB) and untracked files are not copied: their
rows carry the location and, where the file exists, its SHA-256. One generated figure (FG-01) is produced from
the pack's own tables by ``tools/v2/report/plot_t2_refine_stage1_dist.py``.

Outputs (all deterministic: no timestamps, sorted rows):
  <out>/evidence/manifest.csv   item_id, title, round, source_path, sha256, source_commit, size_bytes, status,
                                copy_path, location
  <out>/evidence/SHA256SUMS     ``sha256sum -c`` format, paths relative to <out>/evidence
  <out>/figures/SHA256SUMS      same for the figures
Existing files are never overwritten with different content (a difference is an error); nothing is deleted.

Usage:
  python tools/v2/report/build_t2_refine_pack.py                      # build into reports/t2_refine_100526
  python tools/v2/report/build_t2_refine_pack.py --out <scratch>      # build elsewhere
  python tools/v2/report/build_t2_refine_pack.py --check-against reports/t2_refine_100526
      # build into a temporary directory and compare it byte for byte with an existing pack (exit 1 on a difference)
  python tools/v2/report/build_t2_refine_pack.py --sums reports/t2_refine_100526/pi_record
      # write pi_record/SHA256SUMS
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_OUT = ROOT / "reports" / "t2_refine_100526"
LARGE_BYTES = 1 << 20
WT = "/home/fjiang4/tournament_experiment/.claude/worktrees"
FIELDS = ["item_id", "title", "round", "source_path", "sha256", "source_commit", "size_bytes", "status",
          "copy_path", "location"]

R1, V20, R2B, R2C, PUB = "R1", "v2.0", "R2b", "R2c", "pack"


class Item(NamedTuple):
    """One pack item: id, title, round, source path (repo-relative or absolute), kind (copy | fig | ref | gen)."""

    id: str
    title: str
    round: str
    src: str
    kind: str = "copy"


def _c(i: str, t: str, r: str, s: str) -> Item:
    return Item(i, t, r, s, "copy")


def _f(i: str, t: str, r: str, s: str) -> Item:
    return Item(i, t, r, s, "fig")


def _r(i: str, t: str, r: str, s: str) -> Item:
    return Item(i, t, r, s, "ref")


V2L = "results/v2_T2_locked"
R1A = "results/v2_refine/analysis"
R2BA = "results/v2_refine_r2b/analysis"
R2BD = "results/v2_refine_r2b/diag_30510"
R2CA = "results/v2_refine_r2c/analysis"

ITEMS: List[Item] = [
    # ---- protocols and locks
    _c("PL-01", "Locked protocol v2.0 (JSON, what the entry point runs)", V20, "protocols/v2_T2_locked_v2_0.json"),
    _c("PL-02", "Locked protocol v2.0 (Markdown)", V20, "protocols/v2_T2_locked_v2_0.md"),
    _c("PL-03", "protocols/LOCK: lock records v1.0, v1.1, v2.0", V20, "protocols/LOCK"),
    _c("PL-04", "Machine diff of the protocol, v1.1 -> v2.0", V20, f"{V2L}/v2_0/protocol_diff_v1_1_to_v2_0.json"),
    _c("PL-05", "Continuation-table check (ii) as settled in v2.0: (ii-a), (ii-b), (ii-c)", V20, f"{V2L}/v2_0/continuation_check_v2_0.json"),
    _c("PL-06", "Verifier numerics note (standard-tier gap, reported not gated)", V20, f"{V2L}/v2_0/verifier_numerics_note.md"),
    _c("PL-07", "R1 continuation-table check (literal criterion, refined verifier, table convergence)", R1, "results/v2_refine/continuation_check.json"),
    _c("PL-08", "Locked protocol v1.1 (JSON), the baseline of the v2.0 change", V20, "protocols/v2_T2_locked_v1_1.json"),
    _c("PL-09", "G-S pass-probability record used for the G-S admission rule", V20, f"{V2L}/v2_0/v2_0_pass_probability.json"),
    _c("PL-10", "Seed inventory before the block 30501-30520 appeared in any record: summary", V20, f"{V2L}/v2_0/seed_inventory_pre_lock.out"),
    _c("PL-11", "Seed inventory before the block appeared in any record: all recorded seed values", V20, f"{V2L}/v2_0/seed_inventory_pre_lock.csv"),
    _c("PL-12", "Seed inventory at the lock commit, protocol declaration excluded: summary", V20, f"{V2L}/v2_0/seed_inventory.out"),
    _c("PL-13", "Seed inventory at the lock commit, stock scan: summary (hits = the protocol's own declaration)", V20, f"{V2L}/v2_0/seed_inventory_stock.out"),
    _c("PL-14", "Seed inventory at the lock commit, stock scan: all rows", V20, f"{V2L}/v2_0/seed_inventory_stock.csv"),
    _c("PL-15", "Continuation-table check (ii) as settled in v2.0: tool output", V20, f"{V2L}/v2_0/continuation_check_v2_0.out"),
    # ---- v2.0 confirmation and rehearsal
    _c("CF-01", "v2.0 confirmation verdict (seeds 30501-30520)", V20, f"{V2L}/confirmation_v2_0_analysis/verdict.json"),
    _c("CF-02", "v2.0 confirmation pass counts with exact Clopper-Pearson CIs", V20, f"{V2L}/confirmation_v2_0_analysis/pass_counts.csv"),
    _c("CF-03", "v2.0 confirmation: one row per run, every gate and metric", V20, f"{V2L}/confirmation_v2_0_analysis/per_run.csv"),
    _c("CF-04", "v2.0 confirmation: stage-1 error (S1 / G-S) summary per q", V20, f"{V2L}/confirmation_v2_0_analysis/s1_summary.csv"),
    _c("CF-05", "v2.0 confirmation: metric distributions", V20, f"{V2L}/confirmation_v2_0_analysis/distributions.csv"),
    _c("CF-06", "v2.0 confirmation: reported (non-gated) metrics", V20, f"{V2L}/confirmation_v2_0_analysis/reported_metrics.csv"),
    _c("CF-07", "v2.0 confirmation: agreement of the analysis script with gates.json", V20, f"{V2L}/confirmation_v2_0_analysis/agreement.csv"),
    _c("CF-08", "v2.0 re-rehearsal checks R1-R7 (development seeds)", V20, f"{V2L}/rehearsal_v2_0_checks.json"),
    _c("CF-09", "C7 against the canonical reference (decision D-R7)", V20, f"{V2L}/rehearsal_v2_0_checks/c7_canonical_reference.txt"),
    _c("CF-10", "v1.1 confirmation: one row per run (seeds 20501-20520)", V20, f"{V2L}/confirmation_analysis/per_run.csv"),
    _c("CF-11", "v1.1 confirmation: stage-1 error (S1) summary per q", V20, f"{V2L}/confirmation_analysis/s1_summary.csv"),
    _c("CF-12", "v1.1 confirmation: reported (non-gated) metrics", V20, f"{V2L}/confirmation_analysis/reported_metrics.csv"),
    _c("CF-13", "gates.json of the failed v2.0 run, q = 50 seed 30510", V20, f"{V2L}/confirmation_v2_0/q50/seed30510/gates.json"),
    _c("CF-14", "v1.1 confirmation verdict", V20, f"{V2L}/confirmation_analysis/verdict.json"),
    _c("CF-15", "v1.1 confirmation pass counts", V20, f"{V2L}/confirmation_analysis/pass_counts.csv"),
    _c("CF-16", "v2.0 confirmation: rendered tables", V20, f"{V2L}/confirmation_v2_0_analysis/tables.md"),
    _c("CF-17", "v1.1 confirmation: metric distributions", V20, f"{V2L}/confirmation_analysis/distributions.csv"),
    _c("CF-18", "v2.0 vs v1.1 on fresh seeds: distribution comparison", V20, f"{V2L}/confirmation_v2_0_analysis_vs_v1_1/compare_distributions.csv"),
    _c("CF-19", "v2.0 re-rehearsal: per-run gates and metrics", V20, f"{V2L}/rehearsal_v2_0_analysis/per_run.csv"),
    _c("CF-20", "Seed inventory at the v2.0 lock (disjointness of the seed block)", V20, f"{V2L}/v2_0/seed_inventory.csv"),
    _c("CF-21", "R7 re-run at the checks-tool fix commit 3fedaa2 (pass = true)", V20, f"{V2L}/rehearsal_v2_0_checks/tool_fix_rerun/r7_result.json"),
    _c("CF-22", "v2.0 re-rehearsal: G-S safety condition (normal model, probabilities per q)", V20, f"{V2L}/rehearsal_v2_0_analysis/rehearsal_safety.json"),
    _c("CF-23", "Record of the four stray R1 files (hashes and diffs) found in a sibling worktree", V20, f"{V2L}/v2_0/stray_files_record.json"),
    _c("CF-24", "Full-suite log at the v2.0 launch tree (391 passed, 1 failed, 2 xfailed)", V20, f"{V2L}/rehearsal_v2_0_pytest.txt"),
    _c("CF-25", "Full-suite log before the v2.0 lock", V20, f"{V2L}/v2_0/fullsuite_prelock.txt"),
    # ---- R1
    _c("R1-01", "R1 decision inputs: one row per (method, arm)", R1, f"{R1A}/decision_inputs.csv"),
    _c("R1-02", "R1 stage-1 criterion per arm", R1, f"{R1A}/stage1_criterion.csv"),
    _c("R1-03", "R1 stage-1 dispersion ratios (SD of the signed error, arm / baseline)", R1, f"{R1A}/stage1_dispersion.csv"),
    _c("R1-04", "R1 stage-1 advantage-SD ratios", R1, f"{R1A}/stage1_adv_ratio.csv"),
    _c("R1-05", "R1 stage-1: one row per run", R1, f"{R1A}/stage1_per_run.csv"),
    _c("R1-06", "R1 stage-2 criterion per arm", R1, f"{R1A}/stage2_criterion.csv"),
    _c("R1-07", "R1 stage-2 annealing: predicted vs observed smoothing", R1, f"{R1A}/stage2_annealing.csv"),
    _c("R1-08", "R1 method 5: pairs against the matched control and the parent", R1, f"{R1A}/decision_method5_pairs.csv"),
    _c("R1-09", "C-R1: the unchanged v1.1 entry point reproduces rehearsal_v1_1 (20/20)", R1, "results/v2_refine/v11_reproduction_checks.json"),
    _c("R1-10", "D1 clamp flags M1 / M2 per group, q and phase", R1, "results/v2_refine/d1_clamp/d1_flags.csv"),
    _c("R1-11", "D2 detection limits per family, gate and q", R1, "results/v2_refine/d2_verifier_sensitivity/detection_limits.csv"),
    _c("R1-12", "D2 per-candidate evaluations (152 rows: 38 closed-form candidates x 2 q x 2 tiers)", R1, "results/v2_refine/d2_verifier_sensitivity/evaluations.csv"),
    _c("R1-13", "D2 fits of the verifier response", R1, "results/v2_refine/d2_verifier_sensitivity/fits.csv"),
    _c("R1-14", "D2 family-e confirmation", R1, "results/v2_refine/d2_verifier_sensitivity/family_e_confirmation.csv"),
    _c("R1-15", "parents_A end-of-A state reproduces rehearsal_v1_1 (20/20)", R1, "results/v2_refine/parents_A_checks.json"),
    _c("R1-16", "B_base reproduces rehearsal_v1_1 (20/20)", R1, "results/v2_refine/stage1_base_checks.json"),
    _c("R1-17", "A_base reproduces rehearsal_v1_1 (20/20)", R1, "results/v2_refine/stage2_base_checks.json"),
    _c("R1-18", "R1 stage-1 paired differences, every arm and metric", R1, f"{R1A}/stage1_paired.csv"),
    _c("R1-19", "R1 stage-2 paired differences, every arm and metric", R1, f"{R1A}/stage2_paired.csv"),
    _c("R1-20", "R1 stage-1 overview", R1, f"{R1A}/stage1_overview.csv"),
    _c("R1-21", "R1 stage-2 overview", R1, f"{R1A}/stage2_overview.csv"),
    _c("R1-22", "R1 stage-1 cost per arm", R1, f"{R1A}/stage1_cost.csv"),
    _c("R1-23", "R1 stage-2 cost per arm", R1, f"{R1A}/stage2_cost.csv"),
    _c("R1-24", "R1 stage-1 gate counts", R1, f"{R1A}/stage1_gate_counts.csv"),
    _c("R1-25", "R1 stage-2 gate counts", R1, f"{R1A}/stage2_gate_counts.csv"),
    _c("R1-26", "R1 stage-1 frozen-snapshot and induced-target decomposition", R1, f"{R1A}/stage1_decomposition.csv"),
    _c("R1-27", "R1 stage-1 target-KL stop epochs", R1, f"{R1A}/stage1_stop_epoch.csv"),
    _c("R1-28", "R1 stage-2 target-KL stop epochs", R1, f"{R1A}/stage2_stop_epoch.csv"),
    _c("R1-29", "R1 D1 flags (headline rows, locked pipeline)", R1, f"{R1A}/decision_d1_flags.csv"),
    _c("R1-30", "R1 D1 flags per pilot arm", R1, f"{R1A}/decision_d1_flags_arms.csv"),
    _c("R1-31", "R1 D2 detection limits (decision table)", R1, f"{R1A}/decision_d2_detection_limits.csv"),
    _c("R1-32", "R1 method 6 check (ii) (decision table)", R1, f"{R1A}/decision_method6_check_ii.csv"),
    _c("R1-33", "R1 method 5: A_detmean summary", R1, f"{R1A}/stage2_detmean_summary.csv"),
    _c("R1-34", "R1 method 5: A_detmean trajectory (offline objective, FOC residual)", R1, f"{R1A}/stage2_detmean_trajectory.csv"),
    _c("R1-35", "R1 D1 summary", R1, "results/v2_refine/d1_clamp/d1_summary.json"),
    _c("R1-36", "R1 D1 by group (hits inside / outside |d| < 2q)", R1, "results/v2_refine/d1_clamp/d1_by_group.csv"),
    _c("R1-37", "R1 stage-2 per-run table", R1, f"{R1A}/stage2_per_run.csv"),
    _c("R1-38", "R1 stage-2 dispersion ratios", R1, f"{R1A}/stage2_dispersion.csv"),
    _c("R1-39", "R1 stage-1 arm summary", R1, f"{R1A}/stage1_arm_summary.csv"),
    _c("R1-40", "R1 stage-2 arm summary", R1, f"{R1A}/stage2_arm_summary.csv"),
    _c("R1-41", "D1: alpha < 1 / beta < 1 learner policy rows per group, q and phase", R1, "results/v2_refine/d1_clamp/d1_alpha_beta_lt1.csv"),
    _c("R1-42", "D1: saved-buffer statistics (clamped policy rows)", R1, "results/v2_refine/d1_clamp/d1_buffers.csv"),
    _c("R1-43", "D1: log density minus log censored mass at clamped rows, by group", R1, "results/v2_refine/d1_clamp/d1_logdiff_by_group.csv"),
    _c("R1-44", "D1: clamped-row gradient share per buffer (the basis of M2)", R1, "results/v2_refine/d1_clamp/d1_gradient_share.csv"),
    _c("R1-45", "Smoothed-game prediction e_pred(d) (definition used by the annealing analysis)", R1, "tools/v2/pilot1_smoothed_game.py"),
    _c("R1-46", "R1 re-runs of 17 stage-1 runs: bit-identical to the superseded originals", R1, "results/v2_refine/stage1_dirty_rerun_comparison.json"),
    _c("R1-47", "R1 full-suite log (328 passed, 1 failed, 4 xfailed)", R1, "results/v2_refine/code_pytest_full.txt"),
    # ---- R2b
    _c("R2B-01", "R2b decision inputs: one row per mechanism arm", R2B, f"{R2BA}/decision_inputs.csv"),
    _c("R2B-02", "R2b criterion parts (a) / (b) per arm", R2B, f"{R2BA}/criterion.csv"),
    _c("R2B-03", "R2b: one row per run (waves A and P, comparators)", R2B, f"{R2BA}/per_run.csv"),
    _c("R2B-04", "R2b tail statistics per arm and q", R2B, f"{R2BA}/tail.csv"),
    _c("R2B-05", "R2b wave-P trajectories per run (offline objective, FOC residual)", R2B, f"{R2BA}/trajectory_per_run.csv"),
    _c("R2B-06", "C-R2: the unchanged v2.0 entry point reproduces rehearsal_v2_0 (20/20)", R2B, "results/v2_refine_r2b/v20_reproduction_checks.json"),
    _c("R2B-07", "R2b launch checks", R2B, "results/v2_refine_r2b/launch_checks.json"),
    _c("R2B-08", "R2b paired differences, every arm and metric", R2B, f"{R2BA}/paired.csv"),
    _c("R2B-09", "R2b optimisation diagnostics", R2B, f"{R2BA}/optimisation.csv"),
    _c("R2B-10", "R2b cost, wave A", R2B, f"{R2BA}/cost_A.csv"),
    _c("R2B-11", "R2b cost, wave P", R2B, f"{R2BA}/cost_P.csv"),
    _c("R2B-12", "R2b gate counts", R2B, f"{R2BA}/gate_counts.csv"),
    _c("R2B-13", "R2b arm summary", R2B, f"{R2BA}/arm_summary.csv"),
    _c("R2B-14", "R2b location-free peak argmax summary", R2B, f"{R2BA}/locfree_argmax_summary.csv"),
    _c("R2B-15", "Seed-30510 diagnostic: all numbers (JSON)", R2B, f"{R2BD}/numbers.json"),
    _c("R2B-16", "Seed-30510 diagnostic: pre-registered hypothesis rules H1-H4", R2B, f"{R2BD}/tables/hypothesis_rules.csv"),
    _c("R2B-17", "Seed-30510 diagnostic: decomposition of the d = 0 gap, q = 50", R2B, f"{R2BD}/tables/tab_decomposition_q50.csv"),
    _c("R2B-18", "Seed-30510 diagnostic: decomposition, all runs", R2B, f"{R2BD}/tables/tab_decomposition_all_runs.csv"),
    _c("R2B-19", "Seed-30510 diagnostic: peak trajectory band, q = 50", R2B, f"{R2BD}/tables/tab_peak_trajectory_band_q50.csv"),
    _c("R2B-20", "Seed-30510 diagnostic: peak trajectory band, q = 60", R2B, f"{R2BD}/tables/tab_peak_trajectory_band_q60.csv"),
    _c("R2B-21", "Seed-30510 diagnostic: eta_2 at every export, q = 50", R2B, f"{R2BD}/tables/tab_eta2_every_export_q50.csv"),
    _c("R2B-22", "Seed-30510 diagnostic: eta_2 at every verifier call, all runs", R2B, f"{R2BD}/tables/tab_eta2_verifier_calls_all_runs.csv"),
    _c("R2B-23", "Seed-30510 diagnostic: end-of-A scalars, all runs", R2B, f"{R2BD}/tables/tab_endA_scalars_all_runs.csv"),
    _c("R2B-24", "Seed-30510 diagnostic: late-level means per seed, q = 50", R2B, f"{R2BD}/tables/tab_late_means_per_seed_q50.csv"),
    _c("R2B-25", "Seed-30510 diagnostic: per-seed trajectory statistics", R2B, f"{R2BD}/tables/tab_per_seed_trajectory_stats.csv"),
    _c("R2B-26", "Seed-30510 diagnostic: end-of-A profile (scalars)", R2B, f"{R2BD}/tables/tab_endA_profile_scalars.csv"),
    _c("R2B-27", "Seed-30510 diagnostic: D1 clamp counts, phase A", R2B, f"{R2BD}/tables/tab_d1_clamp_counts_phaseA_sum.csv"),
    _c("R2B-28", "Seed-30510 diagnostic: optimisation, trailing-25 band, q = 50", R2B, f"{R2BD}/tables/tab_opt_trailing25_band_q50.csv"),
    _c("R2B-29", "R2b analysis of the peak-set visitation and clamp counts (wave A)", R2B, f"{R2BA}/waveA_specifics.csv"),
    _c("R2B-30", "R2b full-suite log at the code commit (444 passed, 1 failed, 2 xfailed)", R2B, "results/v2_refine_r2b/code_pytest_full.txt"),
    _c("R2B-31", "R2b full-suite log after the audit (486 passed, 1 failed, 2 xfailed)", R2B, "results/v2_refine_r2b/post_audit_pytest_full.txt"),
    # ---- R2c
    _c("R2C-01", "R2c: one row per run (wave S, baseline, R2b reference arms)", R2C, f"{R2CA}/per_run.csv"),
    _c("R2C-02", "R2c criterion parts (a) / (b) per arm", R2C, f"{R2CA}/criterion.csv"),
    _c("R2C-03", "R2c selection record (rule, inputs, outcome)", R2C, f"{R2CA}/selection.json"),
    _c("R2C-04", "R2c tail statistics per arm and q", R2C, f"{R2CA}/tail.csv"),
    _c("R2C-05", "C-R3: the unchanged v2.0 entry point reproduces rehearsal_v2_0 (20/20)", R2C, "results/v2_refine_r2c/v20_reproduction_checks.json"),
    _c("R2C-06", "R2c launch checks (80/80, prefix identities)", R2C, "results/v2_refine_r2c/launch_checks.json"),
    _c("R2C-07", "R2c paired differences, every arm and metric", R2C, f"{R2CA}/paired.csv"),
    _c("R2C-08", "R2c selection inputs per arm", R2C, f"{R2CA}/selection_inputs.csv"),
    _c("R2C-09", "R2c share / local_first response (incl. R2b reference rows)", R2C, f"{R2CA}/waveS_response.csv"),
    _c("R2C-10", "R2c independent recomputation of the selection", R2C, f"{R2CA}/blind_recomputation.txt"),
    _c("R2C-11", "R2c gate counts", R2C, f"{R2CA}/gate_counts.csv"),
    _c("R2C-12", "R2c wave-S specifics, mean over seeds (peak-set visitation, clamp counts)", R2C, f"{R2CA}/waveS_specifics_mean.csv"),
    _c("R2C-13", "R2c wave-S optimisation diagnostics and cost", R2C, f"{R2CA}/waveS_optimisation.csv"),
    _c("R2C-14", "R2c full-suite log at the code commit (567 passed, 1 failed, 2 xfailed)", R2C, "results/v2_refine_r2c/code_pytest_full.txt"),
    # ---- round reports (copies)
    _c("RR-01", "R1 summary report", R1, "reports/v2/refine/summary.md"),
    _c("RR-02", "R1 decision inputs report", R1, "reports/v2/refine/06_decision_inputs.md"),
    _c("RR-03", "v2.0 lock, re-rehearsal and confirmation report", V20, "reports/v2/protocol_v2_0_confirmation.md"),
    _c("RR-04", "R2b summary report", R2B, "reports/v2/refine_r2b/summary.md"),
    _c("RR-05", "R2b decision inputs report", R2B, "reports/v2/refine_r2b/05_decision_inputs.md"),
    _c("RR-06", "R2c summary report", R2C, "reports/v2/refine_r2c/summary.md"),
    _c("RR-07", "R2c selection report", R2C, "reports/v2/refine_r2c/03_selection.md"),
    _c("RR-08", "R1 pre-registration", R1, "reports/v2/refine/01_preregistration.md"),
    _c("RR-09", "R2b pre-registration", R2B, "reports/v2/refine_r2b/01_preregistration.md"),
    _c("RR-10", "R2c pre-registration", R2C, "reports/v2/refine_r2c/01_preregistration.md"),
    _c("RR-11", "R2b seed-30510 diagnostic report", R2B, "reports/v2/refine_r2b/02_seed30510_diagnostic.md"),
    _c("RR-12", "R2c housekeeping record (main not fast-forwarded; D1 waivers)", R2C, "reports/v2/refine_r2c/00_housekeeping.md"),
    _c("RR-13", "R2b housekeeping record (report-pack refresh failure diagnosis)", R2B, "reports/v2/refine_r2b/00_housekeeping.md"),
    _c("RR-14", "R1 D1 report: clamp of the raw Beta draws", R1, "reports/v2/refine/02_d1_clamp.md"),
    _c("RR-15", "R1 D2 report: sensitivity of the DP-BR verifier", R1, "reports/v2/refine/03_d2_verifier_sensitivity.md"),
    _c("RR-16", "R2b wave-A report (peak-focused starts, censored likelihood)", R2B, "reports/v2/refine_r2b/03_pilot_waveA.md"),
    _c("RR-17", "R2b wave-P report (pathwise terminal fine-tuning, matched controls)", R2B, "reports/v2/refine_r2b/04_pilot_waveP.md"),
    # ---- large or untracked files: referenced, not copied
    _r("UT-01", "D1 per-row clamped draws (large, tracked)", R1, "results/v2_refine/d1_clamp/d1_clamped_rows.csv"),
    _r("UT-02", "D1 clamp fraction by local update (large, tracked)", R1, "results/v2_refine/d1_clamp/d1_clamp_fraction_by_local.csv"),
    _r("UT-03", "Failed run q=50 seed 30510: end-of-A state (untracked)", V20, f"{WT}/v2-t2-refine/{V2L}/confirmation_v2_0/q50/seed30510/state_end_A.pt"),
    _r("UT-04", "Failed run q=50 seed 30510: end-of-B state (untracked)", V20, f"{WT}/v2-t2-refine/{V2L}/confirmation_v2_0/q50/seed30510/state_end_B.pt"),
    _r("UT-05", "Failed run q=50 seed 30510: train history (untracked)", V20, f"{WT}/v2-t2-refine/{V2L}/confirmation_v2_0/q50/seed30510/train_history.json"),
    _r("UT-06", "Failed run q=50 seed 30510: final weights export (untracked)", V20, f"{WT}/v2-t2-refine/{V2L}/confirmation_v2_0/q50/seed30510/checkpoint_weights.npz"),
    _r("UT-07", "R2b A_peak50 q=60 seed 10503 (tail violation): final-tier evaluation (untracked)", R2B, f"{WT}/v2-t2-r2b/results/v2_refine_r2b/waveA/q60/seed10503/A_peak50/final_final.npz"),
    _r("UT-08", "R2b A_peak50 q=60 seed 10504 (tail violation): final-tier evaluation (untracked)", R2B, f"{WT}/v2-t2-r2b/results/v2_refine_r2b/waveA/q60/seed10504/A_peak50/final_final.npz"),
    _r("UT-09", "R2b A_peak50 q=60 seed 10510 (tail violation): final-tier evaluation (untracked)", R2B, f"{WT}/v2-t2-r2b/results/v2_refine_r2b/waveA/q60/seed10510/A_peak50/final_final.npz"),
    _r("UT-10", "R2b A_peak50 q=60 seed 10503: train history (untracked)", R2B, f"{WT}/v2-t2-r2b/results/v2_refine_r2b/waveA/q60/seed10503/A_peak50/train_history.json"),
    _r("UT-11", "R2c A_peak35 q=60 seed 10501: end-of-A state (untracked)", R2C, f"{WT}/v2-t2-r2c/results/v2_refine_r2c/waveS/q60/seed10501/A_peak35/state_end_A.pt"),
    _r("UT-12", "R1 parents_A q=50 seed 10501: full state at update 1200 (untracked)", R1, f"{WT}/v2-t2-refine/results/v2_refine/parents_A/q50/seed10501/state_u01200.pt"),
]

FIGURES: List[Item] = [
    Item("FG-01", "Stage-1 |error| on fresh seeds, v1.1 (20501-20520) vs v2.0 (30501-30520); generated from CF-03 and CF-10", V20,
         "tools/v2/report/plot_t2_refine_stage1_dist.py", "gen"),
    _f("FG-02", "R1 stage 1: primary metric per arm (overview)", R1, "reports/v2/refine/figures/stage1_overview.png"),
    _f("FG-03", "R1 stage 1: B_expcont paired primary (method 6)", R1, "reports/v2/refine/figures/stage1_B_expcont_paired_primary.png"),
    _f("FG-04", "R1 stage 2: peak error per arm (overview)", R1, "reports/v2/refine/figures/stage2_overview.png"),
    _f("FG-05", "R1 method 5: A_detmean first-order-condition residual", R1, "reports/v2/refine/figures/stage2_A_detmean_foc.png"),
    _f("FG-06", "R1 method 5: A_detmean offline objective", R1, "reports/v2/refine/figures/stage2_A_detmean_objective.png"),
    _f("FG-07", "D2 detection curves, family a (stage-1 scalar error)", R1, "results/v2_refine/d2_verifier_sensitivity/figures/d2_family_a.png"),
    _f("FG-08", "D2 detection curves, family b (stage-2 amplitude)", R1, "results/v2_refine/d2_verifier_sensitivity/figures/d2_family_b.png"),
    _f("FG-09", "D2 detection curves, family c (peak rounding)", R1, "results/v2_refine/d2_verifier_sensitivity/figures/d2_family_c.png"),
    _f("FG-10", "D2 detection curves, family d (tail offset)", R1, "results/v2_refine/d2_verifier_sensitivity/figures/d2_family_d.png"),
    _f("FG-11", "D2 detection curves, family e (RL-like combination)", R1, "results/v2_refine/d2_verifier_sensitivity/figures/d2_family_e.png"),
    _f("FG-12", "Seed 30510: stage-2 peak trajectory against the pack, q = 50", R2B, "reports/v2/refine_r2b/figures/fig1_peak_trajectory_q50.png"),
    _f("FG-13", "Seed 30510: end-of-A profile", R2B, "reports/v2/refine_r2b/figures/fig3_endA_profile.png"),
    _f("FG-14", "R2b A_peak25 minus A_base, paired |peak error|", R2B, "reports/v2/refine_r2b/figures/waveA_A_peak25_vs_A_base.png"),
    _f("FG-15", "R2b A_peak50 minus A_base, paired |peak error|", R2B, "reports/v2/refine_r2b/figures/waveA_A_peak50_vs_A_base.png"),
    _f("FG-16", "R2b A_censored minus A_base, paired |peak error|", R2B, "reports/v2/refine_r2b/figures/waveA_A_censored_vs_A_base.png"),
    _f("FG-17", "R2b wave P: offline objective / FOC residual at the weight exports", R2B, "reports/v2/refine_r2b/figures/waveP_offline_trajectory.png"),
    _f("FG-18", "R2b wave P: logged loss and FOC trajectories", R2B, "reports/v2/refine_r2b/figures/waveP_logged_trajectory.png"),
    _f("FG-19", "R2c wave S: paired |peak error| per arm and R2b reference arms", R2C, "reports/v2/refine_r2c/figures/waveS_overview.png"),
]


# --------------------------------------------------------------------------- helpers
def sha256_bytes(b: bytes) -> str:
    """SHA-256 hex digest of bytes."""
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    """SHA-256 hex digest of a file (streamed)."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> Tuple[int, bytes]:
    """Run git in the repository root; returns (returncode, stdout bytes)."""
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)
    return r.returncode, r.stdout


def is_tracked(rel: str) -> bool:
    """True when ``rel`` is tracked at HEAD."""
    rc, _ = git("cat-file", "-e", f"HEAD:{rel}")
    return rc == 0


def last_commit(rel: str) -> str:
    """Hash of the last commit that touched ``rel`` (40 hex), or ``untracked`` / ``uncommitted``."""
    if not is_tracked(rel):
        return "untracked"
    rc, out = git("log", "-1", "--format=%H", "--", rel)
    return out.decode().strip() if rc == 0 and out.strip() else "uncommitted"


def source_path(item: Item) -> Tuple[Path, str, bool]:
    """(absolute path, path as written in the manifest, repo-relative and inside the repo)."""
    p = Path(item.src)
    if p.is_absolute():
        return p, str(p), False
    return ROOT / p, item.src, True


def write_sums(base: Path, name: str = "SHA256SUMS") -> None:
    """``sha256sum -c`` file of every file under ``base`` (except the sums file), sorted by path."""
    rows = []
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name != name:
            rows.append(f"{sha256_file(p)}  {p.relative_to(base).as_posix()}\n")
    (base / name).write_text("".join(rows))


def put(dest: Path, src: Path) -> None:
    """Copy ``src`` to ``dest``; an existing different file is an error, an identical one is left alone."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha256_file(dest) != sha256_file(src):
            raise SystemExit(f"refusing to overwrite {dest}: its content differs from {src}")
        return
    shutil.copyfile(src, dest)


# --------------------------------------------------------------------------- build
def build(out: Path) -> List[Dict[str, str]]:
    """Copy / hash every item into ``out``; returns the manifest rows (in item order)."""
    ev, fg = out / "evidence", out / "figures"
    rows: List[Dict[str, str]] = []
    problems: List[str] = []
    for item in ITEMS + FIGURES:
        abs_p, shown, in_repo = source_path(item)
        row = {"item_id": item.id, "title": item.title, "round": item.round, "source_path": shown, "sha256": "",
               "source_commit": "", "size_bytes": "", "status": "", "copy_path": "", "location": ""}
        if item.kind == "gen":
            import plot_t2_refine_stage1_dist as P

            cf3, cf10 = ev / next(i for i in ITEMS if i.id == "CF-03").src, ev / next(i for i in ITEMS if i.id == "CF-10").src
            dest = fg / f"{item.id}_stage1_error_v1_1_vs_v2_0.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                tmp = Path(tempfile.mkdtemp()) / dest.name
                P.make(tmp, cf10, cf3)
                if sha256_file(tmp) != sha256_file(dest):
                    problems.append(f"{item.id}: regenerated figure differs from {dest}")
            else:
                P.make(dest, cf10, cf3)
            row.update(sha256=sha256_file(dest), source_commit=last_commit(item.src), size_bytes=str(dest.stat().st_size),
                       status="generated", copy_path=dest.relative_to(out).as_posix(),
                       location=f"generated by {item.src} from the pack copies of CF-10 and CF-03")
            rows.append(row)
            continue
        exists = abs_p.is_file()
        tracked = in_repo and is_tracked(item.src)
        if not exists:
            problems.append(f"{item.id}: source file missing: {shown}")
            row.update(status="MISSING", location=shown)
            rows.append(row)
            continue
        size = abs_p.stat().st_size
        sha_src = sha256_file(abs_p)
        row.update(sha256=sha_src, size_bytes=str(size), source_commit=last_commit(item.src) if in_repo else "untracked")
        if tracked:
            rc, blob = git("show", f"HEAD:{item.src}")
            if rc != 0 or sha256_bytes(blob) != sha_src:
                problems.append(f"{item.id}: {shown} on disk differs from its blob at HEAD")
        copy = item.kind in ("copy", "fig") and tracked and size < LARGE_BYTES
        if item.kind in ("copy", "fig") and not copy:
            problems.append(f"{item.id}: {shown} is large or untracked but was listed as a copy item")
        if copy:
            dest = (ev / item.src) if item.kind == "copy" else (fg / f"{item.id}_{Path(item.src).name}")
            put(dest, abs_p)
            if sha256_file(dest) != sha_src:
                problems.append(f"{item.id}: copy {dest} differs from its source")
            row.update(status="copied", copy_path=dest.relative_to(out).as_posix(), location=shown)
        else:
            row.update(status="referenced (tracked, large)" if tracked else "referenced (untracked)",
                       location=shown if not in_repo else f"{shown} (in the repository at HEAD)" if tracked else shown)
        rows.append(row)
    if problems:
        raise SystemExit("pack build failed:\n  " + "\n  ".join(problems))
    ev.mkdir(parents=True, exist_ok=True)
    with open(ev / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    write_sums(ev)
    write_sums(fg)
    return rows


def compare_trees(a: Path, b: Path) -> List[str]:
    """Differences between the pack directories (evidence and figures) of two builds, byte for byte."""
    diffs: List[str] = []
    for sub in ("evidence", "figures"):
        fa = {p.relative_to(a).as_posix() for p in (a / sub).rglob("*") if p.is_file()}
        fb = {p.relative_to(b).as_posix() for p in (b / sub).rglob("*") if p.is_file()}
        for p in sorted(fa - fb):
            diffs.append(f"only in the existing pack: {p}")
        for p in sorted(fb - fa):
            diffs.append(f"only in the rebuilt pack: {p}")
        for p in sorted(fa & fb):
            if not filecmp.cmp(a / p, b / p, shallow=False):
                diffs.append(f"differs: {p}")
    return diffs


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output folder (default reports/t2_refine_100526)")
    ap.add_argument("--check-against", default=None,
                    help="rebuild into a temporary directory and compare with this existing pack (exit 1 on a difference)")
    ap.add_argument("--sums", default=None, metavar="DIR",
                    help="only write DIR/SHA256SUMS (sha256sum -c format, every file under DIR except the sums file); "
                         "used for pi_record/")
    a = ap.parse_args(argv)
    if a.sums:
        write_sums(Path(a.sums))
        print(f"wrote {Path(a.sums) / 'SHA256SUMS'}")
        return 0
    if a.check_against:
        tmp = Path(tempfile.mkdtemp(prefix="t2_refine_pack_check_"))
        rows = build(tmp)
        diffs = compare_trees(Path(a.check_against), tmp)
        print(f"rebuilt {len(rows)} items into {tmp}")
        if diffs:
            print("\n".join(diffs))
            print(f"NOT reproduced: {len(diffs)} difference(s)")
            return 1
        print(f"reproduced byte for byte against {a.check_against}")
        shutil.rmtree(tmp)          # the tool's own temporary build; kept only when a difference is found
        return 0
    rows = build(Path(a.out))
    n = {}
    for r in rows:
        n[r["status"]] = n.get(r["status"], 0) + 1
    print(f"{len(rows)} items: {n}; manifest {Path(a.out) / 'evidence' / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
