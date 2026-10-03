"""Helpers of ``sec_ext.py``: column documentation, report-cell parsers, a cross-check recorder,
and the final-tier re-evaluation of saved weight exports.

Kept separate so that ``sec_ext.py`` holds one short function per pack item.
"""

from __future__ import annotations

import importlib.util
import math
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import common as C
import dictionary

# ----------------------------------------------------------------------------------------------
# column documentation (columns not covered by dictionary.py)
# ----------------------------------------------------------------------------------------------

PX_TOOL = "tools/v2/phaseA_ext_analysis.py"
P4_TOOL = "tools/v2/pilot4_analysis.py"
RF_TOOL = "tools/v2/pilot4_repr_floor.py"
CUSP_TOOL = "tools/v2/cusp_diagnostic.py"
LOCK_TOOL = "tools/v2/locked_rehearsal_analysis.py"
EFF = "effort units [0, 100], raw"


def _d(definition: str, units: str = "", normalization: str = "none", tier: str = "n/a",
       source: str = "pack builder (tools/v2/report/sec_ext.py)") -> Dict[str, str]:
    return {"definition": definition, "units": units, "normalization": normalization, "tier": tier, "source": source}


DOCS: Dict[str, Dict[str, str]] = {
    "block": _d("Block of the table: rows of different kinds share one table; the block names the kind of row "
                "(see the item notes); columns that do not apply to a block are empty", "label"),
    "first_export": _d("Global update of the first weight export averaged into the candidate", "updates",
                       source=P4_TOOL + ":_job"),
    "n_exports": _d("Number of weight exports averaged into the candidate (= K)", "count", source=P4_TOOL + ":_job"),
    "e_pred_0": _d("Smoothed-game prediction e_pred(0) = (Delta W / 2k) E[f_xi(a_i - a_j)] at d = 0, a_i and a_j the "
                   "mean-centred stage-2 Beta actions of the learned policy (400 equal-probability nodes per Beta)",
                   EFF, tier="tier-independent", source=PX_TOOL + ":_job (pilot1_smoothed_game.centred_nodes)"),
    "e_learned_0": _d("e_learned(0): stage-2 Beta-mean effort at d = 0 entering the smoothed-game share", EFF,
                      tier="tier-independent", source=PX_TOOL + ":_job"),
    "share_peak_gap_explained": _d("Share of the d = 0 peak gap explained by action-noise smoothing: "
                                   "(e2*(0) - e_pred(0)) / (e2*(0) - e_learned(0)); ill-conditioned when e_learned(0) "
                                   "approaches e2*(0) (share > 1 or < 0)", "dimensionless", tier="tier-independent",
                                   source=PX_TOOL + ":_job"),
    "would_have_fired_A": _d("JSON of the existing Phase A stop rule that would have fired (global and local update, "
                             "rule); its counters restarted at the re-entry u400", "JSON text", source=PX_TOOL),
    "phase_A_ext_wall_sec": _d("Wall-clock seconds of the continuation u401-u1600", "seconds", source=PX_TOOL),
    "updates": _d("Number of PPO updates run by the continuation (u401-u1600)", "updates", source=PX_TOOL),
    "kl_median": _d("Median over all updates of the run of kl_final_epoch (last-epoch approximate KL)", "nats",
                    source=PX_TOOL + " / " + P4_TOOL + ":run_records"),
    "clip_median": _d("Median over all updates of the run of clip_frac", "fraction",
                      source=PX_TOOL + " / " + P4_TOOL + ":run_records"),
    "full_states": _d("Full-state checkpoint files present in the run directory", "file names", source=PX_TOOL),
    "final_tier_source": _d("Where the final-tier values of the row come from: 'saved: <file>' (final_v2.json['final'], "
                            "the run's own final-tier evaluation) or 're-evaluated: <weight export>' (verifier "
                            "evaluation by this builder, logged in reevaluations.csv)", "text"),
    "e1_cand": _d("e_hat_1(0) of the candidate: stage-1 Beta mean at d = 0 (pointwise K-average for K > 1)", EFF,
                  tier="tier-independent", source=P4_TOOL + ":_job"),
    "e1_band_input": _d("Stage-1 effort used as input of the band decomposition (equals e1_cand)", EFF,
                        tier="tier-independent", source=P4_TOOL + ":add_decomposition"),
    "learning_rel_abs": _d("|learning_rel|", "fraction of e_1*(0)", "divided by e_1*(0)", "tier-independent",
                           P4_TOOL + ":add_decomposition"),
    "stage2_peak_locfree_rel_err_abs": _d("|location-free peak error|", "fraction of e_2*(0)", "divided by e_2*(0)",
                                          "tier-independent", P4_TOOL + ":add_decomposition"),
    "parent": _d("Full-state file the run started from (repo-relative)", "path", source=P4_TOOL + ":run_records"),
    "lr_decay": _d("JSON of the lr_decay config of the run (null = constant LR 3e-4)", "JSON text",
                   source=P4_TOOL + ":run_records (manifest input_config)"),
    "actor_lr_first": _d("Actor learning rate of the first update of the run (train_history)", "learning rate",
                         source=P4_TOOL + ":run_records"),
    "actor_lr_last": _d("Actor learning rate of the last update of the run (train_history)", "learning rate",
                        source=P4_TOOL + ":run_records"),
    "critic_lr_last": _d("Critic learning rate of the last update of the run", "learning rate",
                         source=P4_TOOL + ":run_records"),
    "n_updates": _d("Number of updates in the run's phase (train_history entries)", "updates",
                    source=P4_TOOL + ":run_records"),
    "would_have_fired": _d("JSON of the existing stop rule that would have fired in the phase (counters restarted at "
                           "the re-entry update)", "JSON text", source=P4_TOOL + ":run_records"),
    "would_fire_update": _d("Global update at which the existing stop rule would have fired", "updates",
                            source=P4_TOOL + ":run_records"),
    "phase_wall_sec": _d("Wall-clock seconds of the phase", "seconds", source=P4_TOOL + ":run_records"),
    "adv_all_mean_median": _d("Median over updates of the mean advantage over all rows", "advantage units",
                              source=P4_TOOL + ":run_records"),
    "adv_all_std_median": _d("Median over updates of the advantage SD over all rows", "advantage units",
                             source=P4_TOOL + ":run_records"),
    "adv_s1_mean_median": _d("Median over updates of the mean advantage over stage-1 rows", "advantage units",
                             source=P4_TOOL + ":run_records"),
    "adv_s1_std_median": _d("Median over updates of the advantage SD over stage-1 rows", "advantage units",
                            source=P4_TOOL + ":run_records"),
    "adv_used_std_median": _d("Median over updates of the SD used for advantage normalization", "advantage units",
                              source=P4_TOOL + ":run_records"),
    "drift_test_pass": _d("Snapshot drift test (C5): frozen stage-2 mapping (mean, alpha, beta) unchanged since the "
                          "freeze and snapshot parameters bit-identical", "bool", source="drift_test.json"),
    "snapshot_drift_max": _d("Largest |difference| of the frozen stage-2 mean/alpha/beta against freeze time",
                             "effort units / Beta parameter units", source="drift_test.json"),
    "ckpt_e1_at_0": _d("e_hat_1(0) at the last training-time verifier checkpoint (development tier call)", EFF,
                       tier="tier-independent", source="v2_checkpoints.csv"),
    "ckpt_Gmax_full_over_dw": _d("Gmax_full / Delta W at the last training-time checkpoint", "Delta W (dimensionless)",
                                 "divided by Delta W", "development", "v2_checkpoints.csv"),
    "ckpt_EXP_root_over_dw": _d("EXP_root / Delta W at the last training-time checkpoint", "Delta W (dimensionless)",
                                "divided by Delta W", "development", "v2_checkpoints.csv"),
    "ckpt_dReach_over_dw": _d("dReach / Delta W at the last training-time checkpoint", "Delta W (dimensionless)",
                              "divided by Delta W", "development", "v2_checkpoints.csv"),
    "ckpt_sigma_effort_at_0_t1": _d("sigma_1(0) at the last training-time checkpoint", "effort units [0, 100]",
                                    tier="tier-independent", source="v2_checkpoints.csv"),
    "ckpt_update": _d("Global update of the last training-time checkpoint", "updates", source="v2_checkpoints.csv"),
    "n_last5": _d("Number of weight exports in the last-5 window (u2100-u2200)", "count", source=P4_TOOL + ":main"),
    "within_run_sd_e1_last5": _d("Within-run SD (ddof 1) of e_hat_1(0) over the exports u2100, 2125, 2150, 2175, "
                                 "2200", EFF, tier="tier-independent", source=P4_TOOL + ":main"),
    "within_run_range_e1_last5": _d("Within-run range (max - min) of e_hat_1(0) over the exports u2100-u2200", EFF,
                                    tier="tier-independent", source=P4_TOOL + ":main"),
    "n_better": _d("Pairs in the better direction (difference < 0 for smaller-is-better metrics); empty when the "
                   "metric has no preferred direction", "count", source="tools/v2/pilot4_common.py:paired_summary"),
    "better_if": _d("Direction counted as better ('diff < 0') or 'no preferred direction' (signed metrics, KL, clip)",
                    "text", source="tools/v2/pilot4_common.py:paired_summary"),
    "quantity": _d("Name of the quantity in the row", "text"),
    "median_over_e2star0": _d("median divided by e_2*(0) (gap rows only)", "fraction of e_2*(0)", "divided by e_2*(0)",
                              "tier-independent", RF_TOOL + ":main"),
    "steps": _d("Optimizer steps of the supervised fit when it stopped", "Adam steps", source=RF_TOOL + ":fit"),
    "final_loss_mse": _d("Last logged full-batch loss: mean squared error of the Beta mean against e_2*(d) on the "
                         "development D_2 grid", "effort units squared", source=RF_TOOL + ":fit"),
    "best_loss_mse": _d("Smallest logged loss of the fit", "effort units squared", source=RF_TOOL + ":fit"),
    "stop": _d("Stop reason: plateau (plateau rule fired) or max_steps (300,000-step cap)", "label",
               source=RF_TOOL + ":fit"),
    "rmse_fit_grid": _d("RMSE of the fitted mean against e_2*(d) on the fit grid (development D_2, all nodes)",
                        "effort units", source=RF_TOOL + ":fit"),
    "max_abs_resid": _d("Largest |fitted mean - e_2*(d)| on the fit grid", "effort units", source=RF_TOOL + ":fit"),
    "max_abs_resid_at_d": _d("Gap d of the largest residual", "effort units (gap d)", source=RF_TOOL + ":fit"),
    "tail_min_fit": _d("Smallest fitted mean over the tail nodes |d| >= 2q of the fit grid (head floor = 100 * 1e-6 "
                       "= 1e-4)", EFF, source=RF_TOOL + ":fit"),
    "tail_max_fit": _d("Largest fitted mean over the tail nodes |d| >= 2q of the fit grid", EFF,
                       source=RF_TOOL + ":fit"),
    "n_grid": _d("Nodes of the fit grid (development D_2, step 4)", "count", source=RF_TOOL + ":fit"),
    "loss_at_step_280000": _d("Logged loss at step 280,000 (from the fit's loss log, every 1,000 steps; the logged "
                              "loss oscillates, see best_loss_* columns)", "effort units squared",
                              source="repr_floor/fit_q*_init*.npz:loss_log"),
    "best_loss_up_to_step_280000": _d("Smallest logged loss up to step 280,000", "effort units squared",
                                      source="repr_floor/fit_q*_init*.npz:loss_log"),
    "best_loss_rel_drop_last_20000": _d("Relative decrease of the running best (smallest logged) loss over the last "
                                        "20,000 steps: (best up to 280k - best up to 300k) / best up to 280k; > 0 means "
                                        "the best loss was still falling at the cap (the plateau rule's quantity)",
                                        "fraction", source="repr_floor/fit_q*_init*.npz:loss_log"),
    "best_loss_step": _d("Step of the smallest logged loss", "Adam steps",
                         source="repr_floor/fit_q*_init*.npz:loss_log"),
    "final_over_best_loss": _d("final_loss_mse / best_loss_mse (the fit metrics are those of the network at the last "
                               "step, not of the best logged step)", "ratio",
                               source="repr_floor/fit_q*_init*.npz:loss_log"),
    "n_reached": _d("Number of inits whose |error| fell below the threshold within 300,000 steps", "count",
                    source=CUSP_TOOL + ":main"),
    "init_or_seed": _d("Seed of the run (RL rows) or init seed of the supervised fit (floor rows)", "integer"),
    "G-A": _d("v1.0 gate G-A at the end of Phase A (final tier): eta_2 <= 0.005 Delta W and RMSE/e_2*(0) <= 0.05 "
              "and tail mean/e_2*(0) <= 0.02", "bool", tier="final", source=LOCK_TOOL),
    "G-F": _d("v1.0 gate G-F at the end of Phase B (final tier): Gmax_full <= 0.01 Delta W and |stage-1 error| <= 0.10",
              "bool", tier="final", source=LOCK_TOOL),
    "G-A_dev": _d("G-A evaluated with the development-tier values", "bool", tier="development", source=LOCK_TOOL),
    "G-F_dev": _d("G-F evaluated with the development-tier values", "bool", tier="development", source=LOCK_TOOL),
    "run_pass": _d("Run passes (v1.0): G-A and G-F", "bool", tier="final", source=LOCK_TOOL),
    "outcome": _d("Run outcome label (pass / fail_G-A / fail_G-F)", "label", tier="final", source=LOCK_TOOL),
    "eta_T_over_dw_pass": _d("eta_2 criterion of G-A passed (final tier)", "bool", tier="final", source=LOCK_TOOL),
    "stage2_rmse_pos_over_g2_0_pass": _d("RMSE criterion of G-A passed", "bool", tier="tier-independent",
                                         source=LOCK_TOOL),
    "stage2_tail_mean_over_g2_0_pass": _d("Tail-mean criterion of G-A passed", "bool", tier="tier-independent",
                                          source=LOCK_TOOL),
    "Gmax_full_over_dw_pass": _d("Gmax_full criterion of G-F passed (final tier)", "bool", tier="final",
                                 source=LOCK_TOOL),
    "stage1_rel_err_abs_pass": _d("|stage-1 error| criterion of G-F passed", "bool", tier="tier-independent",
                                  source=LOCK_TOOL),
    "failing_criteria": _d("Criteria of G-A / G-F that the run failed (empty = none)", "text", tier="final"),
    "n_run_pass": _d("Runs passing the v1.0 run rule G-A and G-F (pass_counts.csv column run_pass)", "count",
                     tier="final", source=LOCK_TOOL),
    "G_A_pass": _d("Runs passing G-A (final tier)", "count", tier="final", source=LOCK_TOOL),
    "G_F_pass": _d("Runs passing G-F (final tier)", "count", tier="final", source=LOCK_TOOL),
    "G_A_pass_dev": _d("Runs passing G-A with development-tier values", "count", tier="development", source=LOCK_TOOL),
    "G_F_pass_dev": _d("Runs passing G-F with development-tier values", "count", tier="development", source=LOCK_TOOL),
    "n_fail_eta_T_over_dw": _d("Runs failing the eta_2 criterion", "count", tier="final"),
    "n_fail_stage2_rmse_pos_over_g2_0": _d("Runs failing the RMSE criterion", "count", tier="tier-independent"),
    "n_fail_stage2_tail_mean_over_g2_0": _d("Runs failing the tail-mean criterion", "count", tier="tier-independent"),
    "n_fail_Gmax_full_over_dw": _d("Runs failing the Gmax_full criterion", "count", tier="final"),
    "n_fail_stage1_rel_err_abs": _d("Runs failing the |stage-1 error| criterion", "count", tier="tier-independent"),
    "gate": _d("Gate of the v1.0 protocol (G-A or G-F)", "label", source="protocols/v2_T2_locked.json:gates"),
    "op": _d("Comparison operator of the criterion", "text", source="protocols/v2_T2_locked.json:gates"),
    "threshold": _d("Threshold of the criterion, in the units of the metric", "as the metric",
                    source="protocols/v2_T2_locked.json:gates"),
    "study": _d("Study key (pilot1, phaseA_ext, rehearsal = v1.0 rehearsal, rehearsal_v1_1 = v1.1 re-rehearsal, "
                "confirmation)", "label"),
    "checkpoint": _d("Checkpoint of the row (u = global update; end of Phase A = u1600 of the locked runs)", "label"),
    "seeds": _d("Seeds behind the row", "text"),
    "e2_star_0": _d("e_2*(0) = Delta W f_xi(0) / (2k), closed form", EFF, tier="tier-independent",
                    source="utils/theory_multistage.py:g2_two_stage"),
    "share_median": _d("Median over runs of share_peak_gap_explained", "dimensionless", tier="tier-independent"),
    "share_q25": _d("25th percentile over runs of the share (numpy linear)", "dimensionless", tier="tier-independent"),
    "share_q75": _d("75th percentile over runs of the share (numpy linear)", "dimensionless", tier="tier-independent"),
    "share_min": _d("Minimum over runs of the share", "dimensionless", tier="tier-independent"),
    "share_max": _d("Maximum over runs of the share", "dimensionless", tier="tier-independent"),
    "n_share_gt_1": _d("Runs with share > 1 (learned peak gap smaller than the smoothing prediction)", "count",
                       tier="tier-independent"),
    "n_share_lt_0": _d("Runs with share < 0 (e_learned(0) above e_2*(0))", "count", tier="tier-independent"),
    "runs_share_outside_0_1": _d("Seeds with share outside [0, 1] and their share", "text", tier="tier-independent"),
    "e_pred_0_median": _d("Median over runs of e_pred(0)", EFF, tier="tier-independent"),
    "e_learned_0_median": _d("Median over runs of e_learned(0)", EFF, tier="tier-independent"),
    "gap_pred_median": _d("Median over runs of the smoothing-predicted gap e_2*(0) - e_pred(0)", "effort units",
                          tier="tier-independent"),
    "gap_learned_median": _d("Median over runs of the learned gap e_2*(0) - e_learned(0)", "effort units",
                             tier="tier-independent"),
    "ratio_of_median_gaps": _d("gap_pred_median / gap_learned_median (ratio of medians; robust to runs with "
                               "e_learned(0) near e_2*(0), unlike the median of the per-run ratios)", "dimensionless",
                               tier="tier-independent"),
    "n_nodes_per_beta": _d("Equal-probability quadrature nodes per Beta in the smoothed-game prediction", "count",
                           source="tools/v2/pilot1_smoothed_game.py / run/run_v2_T2_locked.py:SMOOTH_NODES"),
    "panel": _d("Figure panel (metric plotted)", "label"),
    "series": _d("Plotted series", "label"),
    "stat": _d("Statistic of the plotted point (median, min, max) or 'run' for a per-run value", "label"),
    "x_value": _d("x coordinate of the plotted reference line", "as the axis"),
    "y_value": _d("y coordinate of the plotted reference line", "as the axis"),
    "value_effort_units": _d("Plotted value in effort units (d = 0 peak gap e_2*(0) - e_hat_2(0))", "effort units",
                             tier="tier-independent"),
    "value_over_e2star0": _d("The same value divided by e_2*(0)", "fraction of e_2*(0)", "divided by e_2*(0)",
                             "tier-independent"),
    "abs_peak_rel_err": _d("|peak error at d = 0| = |peak_rel_err_signed| of the supervised fit", "fraction of e_2*(0)",
                           "divided by e_2*(0)", "tier-independent", CUSP_TOOL + ":fit"),
    "step": _d("Optimizer step of the supervised fit", "Adam steps", source=CUSP_TOOL + ":fit"),
    "loss_mse": _d("Full-batch loss at the step (float32)", "effort units squared", source=CUSP_TOOL + ":fit"),
    "peak_rel_err_signed": _d("(e_hat_2(0) - e_2*(0)) / e_2*(0) of the supervised fit on the recovery grid",
                              "fraction of e_2*(0)", "divided by e_2*(0), signed", "tier-independent",
                              CUSP_TOOL + ":fit"),
    "rmse_pos_over_g2_0": _d("RMSE over |d| < 2q of the supervised fit on the recovery grid, / e_2*(0)",
                             "fraction of e_2*(0)", "divided by e_2*(0)", "tier-independent", CUSP_TOOL + ":fit"),
    "check": _d("Name of the reproducibility check", "text"),
    "n_runs": _d("Runs checked", "count"),
}

# cusp thresholds and first-crossing quantities (thresholds_*.csv)
for _thr in ("0.05", "0.03", "0.01"):
    DOCS[f"step_abs_peak_lt_{_thr}"] = _d(f"First logged step (every 500) with |peak error at d = 0| < {_thr}",
                                          "Adam steps", source=CUSP_TOOL + ":first_below")
    DOCS[f"step_abs_locfree_lt_{_thr}"] = _d(f"First logged step with |location-free peak error| < {_thr}",
                                             "Adam steps", source=CUSP_TOOL + ":first_below")
DOCS["rmse_at_peak_lt_0.05"] = _d("RMSE/e_2*(0) at the first step with |peak error| < 0.05", "fraction of e_2*(0)",
                                  "divided by e_2*(0)", "tier-independent", CUSP_TOOL + ":main")
DOCS["locfree_at_peak_lt_0.05"] = _d("Location-free peak error at the first step with |peak error| < 0.05",
                                     "fraction of e_2*(0)", "divided by e_2*(0), signed", "tier-independent",
                                     CUSP_TOOL + ":main")


def docs_for(cols: Iterable[str], table_tier: str = "", extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Docs for every column not covered by the global dictionary (``final_tier__*`` generated from it).

    Args:
        cols: Column names of the table.
        table_tier: Tier of the table (passed to the dictionary lookup).
        extra: Item-specific overrides (take precedence).

    Returns:
        ``{column: doc}`` for ``Pack.table(docs=...)``.
    """
    out: Dict[str, Any] = {}
    for c in cols:
        if c.startswith("final_tier__"):
            base = dictionary.lookup(c[len("final_tier__"):], "final")
            if base:
                d = dict(base)
                d["definition"] = "Final tier (state step 2, effort step 0.5, GL 32 per half): " + d["definition"]
                d["tier"] = "final"
                out[c] = d
        elif c in DOCS and dictionary.lookup(c, table_tier) is None:
            out[c] = DOCS[c]
    for k, v in (extra or {}).items():
        base = out.get(k) or DOCS.get(k)
        out[k] = {**base, "definition": v} if (isinstance(v, str) and base) else v
    return out


# ----------------------------------------------------------------------------------------------
# report-cell parsing and the cross-check recorder
# ----------------------------------------------------------------------------------------------

def clean(cell: Any) -> str:
    """Report cell normalised for ``C.consistent``: unicode minus, thousands commas and spaces removed."""
    s = str(cell).strip().replace("−", "-").replace(",", "").replace(" ", "").replace("`", "")
    return s.replace("·", "").strip()


_MID = re.compile(r"^\s*(\S+)\s*\[\s*([^,\]]+?)\s*,\s*([^\]]+?)\s*\]\s*\(\s*(.+?)\s*\)\s*$")
_CI = re.compile(r"^\s*\[\s*([^,\]]+?)\s*,\s*([^\]]+?)\s*\]\s*$")
_MED_RANGE = re.compile(r"^\s*(\S+)\s*\[\s*(.+?)\s*–\s*(.+?)\s*\]\s*$")
_MED_CI = re.compile(r"^\s*(\S+)\s*\[\s*([^,\]]+?)\s*,\s*([^\]]+?)\s*\]\s*$")
_LEAD_PAREN = re.compile(r"^\s*(\S+)\s*\(\s*([0-9.eE+\-−]+)")


def _sig(txt: str) -> int:
    """Significant digits shown in a cleaned numeric cell text."""
    mant = re.split(r"[eE]", txt.lstrip("+-"))[0].replace(".", "").lstrip("0")
    return max(len(mant), 1)


def _round_sig(x: float, nsig: int) -> float:
    """``x`` rounded half-up (away from zero on ties) to ``nsig`` significant digits, on its shortest repr."""
    from decimal import ROUND_HALF_UP, Decimal
    d = Decimal(repr(float(x)))
    if d == 0:
        return 0.0
    return float(d.quantize(Decimal(1).scaleb(d.adjusted() - nsig + 1), rounding=ROUND_HALF_UP))


def double_rounding_explains(x: float, cell: Any) -> bool:
    """Whether the report cell equals ``x`` rounded first to one more significant digit, then to the shown digits."""
    txt = clean(cell)
    v = C.parse_num(txt)
    if v is None or v == 0.0 or x is None or not math.isfinite(float(x)):
        return False
    n = _sig(txt)
    return math.isclose(_round_sig(_round_sig(float(x), n + 1), n), v, rel_tol=1e-9, abs_tol=0.0)


def parse_mid(cell: str) -> Optional[Dict[str, str]]:
    """'median [q25, q75] (min–max)' -> dict of the five cell texts (None if the pattern does not match)."""
    m = _MID.match(str(cell))
    if not m:
        return None
    med, lo, hi, rng = m.groups()
    parts = rng.split("–")
    if len(parts) != 2:
        return None
    return {"median": med, "q25": lo, "q75": hi, "min": parts[0], "max": parts[1]}


def parse_ci(cell: str) -> Optional[Tuple[str, str]]:
    """'[lo, hi]' -> (lo, hi) texts."""
    m = _CI.match(str(cell))
    return (m.group(1), m.group(2)) if m else None


def parse_med_range(cell: str) -> Optional[Tuple[str, str, str]]:
    """'8,500 [7,500–9,000]' -> (median, min, max) texts."""
    m = _MED_RANGE.match(str(cell))
    return (m.group(1), m.group(2), m.group(3)) if m else None


def parse_med_ci(cell: str) -> Optional[Tuple[str, str, str]]:
    """'0.331 [0.294, 0.572]' -> (value, lo, hi) texts."""
    m = _MED_CI.match(str(cell))
    return (m.group(1), m.group(2), m.group(3)) if m else None


def parse_lead_paren(cell: str) -> Optional[Tuple[str, str]]:
    """'4.78 (0.068·e2*(0))' -> ('4.78', '0.068')."""
    m = _LEAD_PAREN.match(str(cell))
    return (m.group(1), m.group(2)) if m else None


def raw_rows(report: str, table: Dict[str, Any]) -> List[Tuple[int, List[str]]]:
    """Raw data rows of a parsed report table as (line number, cells split on every '|', stripped).

    Used for tables whose cells contain unescaped pipes (e.g. '|peak err|').
    """
    lines = C.abspath(report).read_text(encoding="utf-8").splitlines()
    out = []
    for i in range(len(table["rows"])):
        ln = table["line"] + 2 + i  # 1-based line number of the data row
        s = lines[ln - 1].strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        out.append((ln, [c.strip() for c in s.split("|")]))
    return out


def find_tables(report: str, header_first: Sequence[str], heading_has: str = "") -> List[Dict[str, Any]]:
    """Report tables whose header starts with ``header_first`` and whose heading contains ``heading_has``."""
    out = []
    for t in C.parse_md_tables(report):
        if heading_has.lower() not in t["heading"].lower():
            continue
        if list(t["header"][:len(header_first)]) == list(header_first):
            out.append(t)
    return out


class Checker:
    """Collects cell comparisons for one (item, report) and records mismatches and a summary row.

    Cells are compared with ``C.consistent`` at the report's displayed precision; booleans compare
    'yes'/'no'/'True'/'False'; text compares exactly.
    """

    def __init__(self, pack: "C.Pack", item: str, report: str, label: str):
        self.pack, self.item, self.report, self.label = pack, item, report, label
        self.n = self.bad = self.unmatched = 0
        self.tables: set = set()

    def num(self, quantity: str, value: Any, cell: Any, where: str = "") -> None:
        """Compare a numeric pack value with a report cell (non-numeric cells are skipped)."""
        txt = clean(cell)
        if C.parse_num(txt) is None:
            return
        self.n += 1
        ok = value is not None and isinstance(value, (int, float, np.integer, np.floating)) and \
            math.isfinite(float(value)) and C.consistent(float(value), txt)
        if not ok:
            self.bad += 1
            comment = ""
            if value is not None and double_rounding_explains(float(value), cell):
                comment = (f"report digit is one off; it equals the pack value rounded first to {_sig(txt) + 1} and "
                           f"then to {_sig(txt)} significant digits (double rounding in the report's rendering)")
            self.pack.mismatch(self.item, quantity, value, f"{self.report} ({where})" if where else self.report,
                               str(cell).strip(), comment)

    def boolean(self, quantity: str, value: Any, cell: Any, where: str = "") -> None:
        """Compare a boolean pack value with a 'yes'/'no' (or True/False) report cell."""
        t = str(cell).strip().lower()
        if t not in ("yes", "no", "true", "false"):
            return
        self.n += 1
        if bool(value) != (t in ("yes", "true")):
            self.bad += 1
            self.pack.mismatch(self.item, quantity, bool(value), f"{self.report} ({where})", str(cell).strip())

    def text(self, quantity: str, value: Any, cell: Any, where: str = "") -> None:
        """Compare a text value with a report cell (exact, after stripping)."""
        self.n += 1
        if str(value).strip() != str(cell).strip():
            self.bad += 1
            self.pack.mismatch(self.item, quantity, value, f"{self.report} ({where})", str(cell).strip())

    def claim(self, quantity: str, ok: bool, pack_value: Any, report_text: str, where: str = "",
              comment: str = "") -> None:
        """Record a prose claim checked by the builder (mismatch when ``ok`` is False)."""
        self.n += 1
        if not ok:
            self.bad += 1
            self.pack.mismatch(self.item, quantity, pack_value, f"{self.report} ({where})" if where else self.report,
                               report_text, comment)

    def miss(self, n: int = 1) -> None:
        """Count report rows that could not be matched to a pack row."""
        self.unmatched += n

    def close(self) -> Dict[str, Any]:
        """Append the summary row to the pack's cross-check list."""
        res = {"item": self.item, "report": self.report, "label": self.label, "n_tables": len(self.tables),
               "n_compared": self.n, "n_mismatch": self.bad, "n_unmatched_rows": self.unmatched}
        self.pack.crosschecks.append(res)
        return res


# ----------------------------------------------------------------------------------------------
# final-tier re-evaluation of saved weight exports
# ----------------------------------------------------------------------------------------------

_V2C = None


def _v2common():
    """tools/v2/common.py loaded under a private name (its module name clashes with the pack's common.py)."""
    global _V2C
    if _V2C is None:
        sp = importlib.util.spec_from_file_location("v2_tools_common_for_pack", str(C.REPO / "tools" / "v2" / "common.py"))
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        _V2C = mod
    return _V2C


def final_tier_scalars(rel_npz: str, q: int) -> Dict[str, Any]:
    """Final-tier ``utils.v2_metrics.evaluate`` scalars of the actor in a weight-export NPZ (Beta mean).

    The policy is built exactly as ``tools/v2/pilot2_analysis._actor_fns`` (CurriculumPPO with the
    exported actor weights, ``run.run_final_dp_br.make_policy_fns``); the game is
    ``tools/v2/common.spec_for(q)``.
    """
    import torch
    torch.set_num_threads(1)
    v2c = _v2common()
    from agents.ppo_curriculum import CurriculumPPO, PPOConfig
    from run.run_final_dp_br import make_policy_fns
    from utils.dp_br_verifier import FINAL_CONFIG
    from utils.v2_metrics import evaluate
    spec = v2c.spec_for(q)
    agent = CurriculumPPO(PPOConfig(), torch.Generator().manual_seed(0), np.random.default_rng(0))
    w = np.load(C.abspath(rel_npz))
    agent.actor.load_state_dict({k[len("actor."):]: torch.as_tensor(w[k]) for k in w.files if k.startswith("actor.")})
    mf, bf = make_policy_fns(agent, spec)
    return dict(evaluate(mf, spec, FINAL_CONFIG, beta_fn=bf).scalars)


def g2_star_0(q: int) -> float:
    """Closed-form e_2*(0) of the as-run game (``utils.theory_multistage.g2_two_stage``)."""
    from utils.theory_multistage import g2_two_stage
    v2c = _v2common()
    s = v2c.spec_for(q)
    return float(g2_two_stage(np.zeros(1), s.q, s.w_h, s.w_l, s.k, s.e_max)[0])


class Timer:
    """Wall-clock timer (context manager)."""

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.wall = time.perf_counter() - self.t0


def stats_row(values: Sequence[float]) -> Dict[str, float]:
    """``C.median_iqr`` of the finite values (median, q25, q75, min, max, n)."""
    return C.median_iqr(values)


def fmt_seeds(seeds: Iterable[int]) -> str:
    """'10501-10510' for a contiguous block, else a space-separated list."""
    s = sorted(int(x) for x in seeds)
    if s and s == list(range(s[0], s[-1] + 1)):
        return f"{s[0]}-{s[-1]}"
    return " ".join(str(x) for x in s)


def max_abs(a: Sequence[float], b: Sequence[float]) -> float:
    """``C.max_abs_diff`` on float arrays."""
    return C.max_abs_diff(np.asarray(a, float), np.asarray(b, float))


INT_COLS = {"q", "seed", "init_seed", "init_or_seed", "step", "K", "update", "steps", "best_loss_step",
            "first_export", "G_A_pass", "G_F_pass", "G_A_pass_dev", "G_F_pass_dev", "B_Gmax_full_t"}


def intify(df: pd.DataFrame) -> pd.DataFrame:
    """Integer-valued float columns (ids and counts left empty by some blocks) as nullable integers."""
    out = df.copy()
    for c in out.columns:
        if (c in INT_COLS or c.startswith("n_") or c == "n") and pd.api.types.is_float_dtype(out[c]):
            v = out[c].dropna()
            if len(v) and bool((v == np.round(v)).all()):
                out[c] = out[c].astype("Int64")
    return out


def union_frame(blocks: List[pd.DataFrame], first: Sequence[str]) -> pd.DataFrame:
    """Concatenate block frames (union of columns) with ``first`` columns first, then the rest in order of appearance."""
    cols: List[str] = list(first)
    for b in blocks:
        for c in b.columns:
            if c not in cols:
                cols.append(c)
    out = pd.concat([b.reindex(columns=cols) for b in blocks], ignore_index=True)
    return intify(out[cols])
