"""Global column dictionary of the T=2 v2 report pack.

``lookup(column, table_tier)`` returns ``dict(definition, units, normalization, tier, source)``
for a pack-table column, from (1) exact entries, (2) prefix / suffix rules, (3) family rules.
Table-specific meanings are passed by the builder as ``docs=`` and take precedence.

Tier labels: ``development`` = state step 4 / effort step 1 / GL 16 per half;
``final`` = state step 2 / effort step 0.5 / GL 32 per half (``utils/dp_br_verifier.py``
``DEV_CONFIG`` / ``FINAL_CONFIG``); ``tier-independent`` = computed on the 0.5-step recovery grid
or by a direct policy query, so it does not depend on the verifier tier.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

V2M = "utils/v2_metrics.py"
DPB = "utils/dp_br_verifier.py"

# key -> (definition, units, normalization, tier_dependent, source function)
# tier_dependent: True -> tier comes from the column suffix / the table's tier; False -> independent
E: Dict[str, Tuple[str, str, str, Optional[bool], str]] = {
    # ---- identifiers --------------------------------------------------------------------------
    "q": ("Half-width q of the uniform action-shock eps ~ U(-q, q); the two values studied are 50 and 60",
          "effort units", "none", None, "run config (game.q)"),
    "seed": ("Run seed; numpy SeedSequence([seed, q, namespace]) keys every RNG stream", "integer", "none", None,
             "run manifest (seed)"),
    "init_seed": ("Seed of the torch generator used to initialise the supervised-fit network (0-4)", "integer",
                  "none", None, "tools/v2/pilot4_repr_floor.py"),
    "arm": ("Study arm (the compared configuration)", "label", "none", None, "run manifest (arm)"),
    "family": ("Analysis family in Pilot 4: ext = Phase A extension, 2a = Phase A LR decay, 2b = Phase B LR decay, "
               "p3 = Pilot 3", "label", "none", None, "tools/v2/pilot4_analysis.py"),
    "K": ("Number of last weight exports averaged pointwise (K = 1 is the last iterate)", "count", "none", None,
          "tools/v2/pilot4_common.py:average"),
    "kind": ("Row kind in candidates_all.csv: curve = per-export value, tail = tail-averaged candidate", "label", "none",
             None, "tools/v2/pilot4_analysis.py"),
    "update": ("Global update index at which the value was taken (u = number of PPO updates done)", "updates", "none",
               None, "run records"),
    "phase": ("Training phase of the row (A = stage 2 only, B = stage 1 with frozen stage 2)", "label", "none", None,
              "run/run_v2_stagewise.py"),
    "local": ("Update index within the phase", "updates", "none", None, "run/run_v2_stagewise.py"),
    "run": ("Run identifier string", "label", "none", None, "run manifest (run)"),
    "run_dir": ("Run directory, relative to the repository root", "path", "none", None, "run records"),
    "commit": ("Git commit of the code that executed the run (manifest git.short or commit)", "hash", "none", None,
               "run manifest (git)"),
    "dirty": ("Whether the working tree had tracked changes or untracked non-results files at launch", "bool", "none",
              None, "run/run_v2_stagewise.py:git_state"),
    "clean_tree": ("Locked entry point: no tracked changes and no untracked files outside results/ at launch", "bool",
                   "none", None, "run/run_v2_T2_locked.py"),
    "protocol_sha256": ("SHA-256 of the locked protocol JSON that the run read", "hash", "none", None,
                        "run/run_v2_T2_locked.py"),
    "tier": ("Verifier tier of the row", "label", "none", None, "utils/dp_br_verifier.py"),
    "verifier_tier": ("Verifier tier used for the evaluation (development or final)", "label", "none", None,
                      "utils/dp_br_verifier.py"),
    "state_step": ("State-grid spacing of the verifier tier", "effort units (gap d)", "none", None,
                   "utils/dp_br_verifier.py:VerifierConfig"),
    "effort_step": ("Effort-grid spacing of the verifier tier", "effort units", "none", None,
                    "utils/dp_br_verifier.py:VerifierConfig"),
    "gl_half": ("Gauss-Legendre nodes per half of the shock-difference support", "count", "none", None,
                "utils/dp_br_verifier.py:gl_shock_rule"),
    "valid": ("Verifier validity flag (finite values, PMF mass, PDL residual, GL weights)", "bool", "none", None,
              "utils/dp_br_verifier.py:verify"),
    "invalid_reasons": ("Semicolon-separated validity failures (empty when valid)", "text", "none", None,
                        "utils/dp_br_verifier.py:verify"),
    "stage1_status": ("stage1_untrained: Phase-A-only run, the stage-1 policy is the untrained initial network, so "
                      "full-policy metrics (EXP_root, dReach, Gmax_full) are not evidence about stage 2", "label",
                      "none", None, "run/run_v2_stagewise.py"),
    # ---- full-domain deviation metrics ------------------------------------------------------------
    "Gmax_full_over_dw": ("Gmax_full: max over stages t and over every node d of D_t of G_t(d) = V_t^BR(d) - V_t^mean(d), "
                          "the full dynamic deviation gain", "Delta W (dimensionless)", "divided by Delta W = 4", True,
                          V2M + ":evaluate"),
    "Gmax_full_t": ("Stage t* at which Gmax_full is attained", "stage index", "none", True, V2M + ":evaluate"),
    "Gmax_full_d": ("Gap d* at which Gmax_full is attained", "effort units (gap d)", "none", True, V2M + ":evaluate"),
    "EXP_root_over_dw": ("EXP_root = V_1^BR(0) - V_1^mean(0): the root deviation gain at (t=1, d=0)",
                         "Delta W (dimensionless)", "divided by Delta W", True, DPB + ":verify"),
    "dReach_over_dw": ("dReach = sum over stages of the max of Delta_t over the reach mask R_t (BR chain support)",
                       "Delta W (dimensionless)", "divided by Delta W", True, DPB + ":verify"),
    "Deltamax_all_over_dw": ("Delta_max_all = max over stages and nodes of the one-step gap Delta_t(d)",
                             "Delta W (dimensionless)", "divided by Delta W", True, DPB + ":verify"),
    "dFull_over_dw": ("dFull = sum over stages of the max of Delta_t over the whole grid D_t",
                      "Delta W (dimensionless)", "divided by Delta W", True, DPB + ":verify"),
    "eta_T_over_dw": ("eta_2 (T = 2): max over the full stage-2 grid D_2 of the one-step deviation gap Delta_2(d); "
                      "equals G_2 on the whole grid", "Delta W (dimensionless)", "divided by Delta W", True,
                      V2M + ":evaluate"),
    "pdl_residual_over_dw": ("Performance-difference-lemma residual of the BR chain", "Delta W (dimensionless)",
                             "divided by Delta W", True, DPB + ":verify"),
    "G_max_t1_over_dw": ("max over D_1 of G_1(d) (the root, d = 0)", "Delta W (dimensionless)", "divided by Delta W",
                         True, V2M + ":evaluate"),
    "G_max_t2_over_dw": ("max over D_2 of G_2(d)", "Delta W (dimensionless)", "divided by Delta W", True,
                         V2M + ":evaluate"),
    "G_argmax_d_t1": ("argmax d of G_1", "effort units (gap d)", "none", True, V2M + ":evaluate"),
    "G_argmax_d_t2": ("argmax d of G_2", "effort units (gap d)", "none", True, V2M + ":evaluate"),
    "Delta_max_t1_over_dw": ("max over D_1 of Delta_1(d)", "Delta W (dimensionless)", "divided by Delta W", True,
                             DPB + ":verify"),
    "Delta_max_t2_over_dw": ("max over D_2 of Delta_2(d)", "Delta W (dimensionless)", "divided by Delta W", True,
                             DPB + ":verify"),
    "Delta_argmax_d_t1": ("argmax d of Delta_1", "effort units (gap d)", "none", True, DPB + ":verify"),
    "Delta_argmax_d_t2": ("argmax d of Delta_2", "effort units (gap d)", "none", True, DPB + ":verify"),
    # ---- on/off path split of Delta_2 -----------------------------------------------------------------------
    "DeltaT_over_dw_n_on": ("Number of on-path stage-2 nodes (|d - drift| < 2q)", "count", "none", True,
                            V2M + ":onpath_mask"),
    "DeltaT_over_dw_n_off": ("Number of off-path stage-2 nodes", "count", "none", True, V2M + ":onpath_mask"),
    "DeltaT_over_dw_on_mass": ("Normalised exact cell mass on path (cell masses of the stage-2 grid renormalised to sum to 1, summed over the on-path nodes; the raw captured total is cellmass_captured_total)", "probability", "none", True,
                               V2M + ":cell_masses"),
    "DeltaT_over_dw_off_mass": ("Normalised cell mass off path (1 - on-path mass)", "probability", "none", True, V2M + ":cell_masses"),
    "DeltaT_over_dw_on_max": ("max of Delta_2 over the on-path nodes (open set |d - drift| < 2q)",
                              "Delta W (dimensionless)", "divided by Delta W", True, V2M + ":onoff_split"),
    "DeltaT_over_dw_on_argmax_d": ("argmax d of the on-path max of Delta_2", "effort units (gap d)", "none", True,
                                   V2M + ":onoff_split"),
    "DeltaT_over_dw_on_mean_unweighted": ("unweighted mean of Delta_2 over the on-path nodes", "Delta W (dimensionless)",
                                          "divided by Delta W", True, V2M + ":onoff_split"),
    "DeltaT_over_dw_on_mean_cellmass_weighted": ("mean of Delta_2 over the on-path nodes weighted by exact cell masses "
                                                 "F_xi(b_{i+1}-drift) - F_xi(b_i-drift)", "Delta W (dimensionless)",
                                                 "divided by Delta W", True, V2M + ":onoff_split"),
    "DeltaT_over_dw_on_mean_pmf_weighted": ("mean of Delta_2 on path weighted by the node-based GL PMF (Phase 1 rule, "
                                            "superseded by cell masses)", "Delta W (dimensionless)",
                                            "divided by Delta W", True, V2M + ":onoff_split"),
    "DeltaT_over_dw_off_max": ("max of Delta_2 over the off-path nodes", "Delta W (dimensionless)", "divided by Delta W",
                               True, V2M + ":onoff_split"),
    "DeltaT_over_dw_off_argmax_d": ("argmax d of the off-path max of Delta_2", "effort units (gap d)", "none", True,
                                    V2M + ":onoff_split"),
    "DeltaT_over_dw_off_mean_unweighted": ("unweighted mean of Delta_2 over the off-path nodes", "Delta W (dimensionless)",
                                           "divided by Delta W", True, V2M + ":onoff_split"),
    "stage1_drift": ("Root drift e_hat_1(0) - e_hat_1(-0); the stage-2 on-path set is centred at it", "effort units",
                     "none", False, V2M + ":evaluate"),
    "stage1_drift_is_zero": ("Whether the root drift is exactly 0", "bool", "none", False, V2M + ":evaluate"),
    "cellmass_captured_total": ("Total exact cell mass captured by the stage-2 grid before normalisation (1.0 = all)",
                                "probability", "none", True, V2M + ":cell_masses"),
    "node_pmf_n_positive_T": ("Stage-2 nodes with positive node-GL PMF (diagnostic of the superseded rule)", "count",
                              "none", True, V2M + ":candidate_pmf"),
    "cand_pmf_mass_T": ("Total mass of the candidate's node-based stage-2 PMF", "probability", "none", True,
                        V2M + ":candidate_pmf"),
    "cand_pmf_min_positive_T": ("Smallest positive node-PMF mass at stage 2", "probability", "none", True,
                                V2M + ":candidate_pmf"),
    # ---- spread -------------------------------------------------------------------------------------------------
    "sigma_effort_at_0_t1": ("sigma_1(0): SD of the stage-1 Beta action at d = 0, in effort units",
                             "effort units [0, 100]", "none", False, V2M + ":evaluate (std_norm * e_range)"),
    "sigma_effort_at_0_t2": ("sigma_2(0): SD of the stage-2 Beta action at d = 0", "effort units [0, 100]", "none", False,
                             V2M + ":evaluate (std_norm * e_range)"),
    "sigma2_effort_mean_pos": ("mean of sigma_2(d) over the stage-2 nodes with |d| < 2q", "effort units [0, 100]", "none",
                               True, V2M + ":evaluate"),
    # ---- recovery metrics (closed form; tier independent) -------------------------------------------------------------
    "g1": ("e_1*(0) = Delta W / (6 k q): stage-1 closed-form effort", "effort units", "none", False,
           "utils/theory_multistage.py:g1_two_stage"),
    "g2_at_0": ("e_2*(0) = Delta W f_xi(0) / (2k): stage-2 closed-form effort at d = 0", "effort units", "none", False,
                "utils/theory_multistage.py:g2_two_stage"),
    "e1_at_0": ("e_hat_1(0): stage-1 Beta-mean effort at d = 0", "effort units [0, 100]", "none", False,
                V2M + ":recovery_metrics"),
    "e2_at_0": ("e_hat_2(0): stage-2 Beta-mean effort at d = 0", "effort units [0, 100]", "none", False,
                V2M + ":recovery_metrics"),
    "stage1_rel_err_signed": ("(e_hat_1(0) - e_1*(0)) / e_1*(0)", "fraction of e_1*(0)", "divided by e_1*(0), signed",
                              False, V2M + ":recovery_metrics"),
    "stage1_rel_err_abs": ("|e_hat_1(0) - e_1*(0)| / e_1*(0)", "fraction of e_1*(0)", "divided by e_1*(0)", False,
                           V2M + ":evaluate"),
    "stage2_peak_rel_err_signed": ("(e_hat_2(0) - e_2*(0)) / e_2*(0): peak error at d = 0, signed", "fraction of e_2*(0)",
                                   "divided by e_2*(0), signed", False, V2M + ":recovery_metrics"),
    "stage2_peak_rel_err_abs": ("|e_hat_2(0) - e_2*(0)| / e_2*(0)", "fraction of e_2*(0)", "divided by e_2*(0)", False,
                                V2M + ":recovery_metrics"),
    "stage2_peak_locfree_rel_err": ("Location-free peak error (max_d e_hat_2(d) - e_2*(0)) / e_2*(0) on the recovery "
                                    "grid", "fraction of e_2*(0)", "divided by e_2*(0), signed", False,
                                    "tools/v2/pilot4_common.py:location_free"),
    "stage2_peak_locfree_argmax_d": ("Gap d at which e_hat_2 is maximal on the recovery grid", "effort units (gap d)",
                                     "none", False, "tools/v2/pilot4_common.py:location_free"),
    "stage2_max_e2": ("max over the recovery grid of e_hat_2(d)", "effort units [0, 100]", "none", False,
                      "tools/v2/pilot4_common.py:location_free"),
    "stage2_rmse_pos": ("RMSE of e_hat_2 - e_2* over the recovery nodes with |d| < 2q", "effort units", "none", False,
                        V2M + ":recovery_metrics"),
    "stage2_rmse_pos_over_g2_0": ("RMSE over |d| < 2q divided by e_2*(0) (G-A criterion)", "fraction of e_2*(0)",
                                  "divided by e_2*(0)", False, V2M + ":recovery_metrics"),
    "stage2_tail_mean": ("mean of e_hat_2 over the recovery nodes with |d| >= 2q (theoretical zero-effort region)",
                         "effort units [0, 100]", "none", False, V2M + ":recovery_metrics"),
    "stage2_tail_max": ("max of e_hat_2 over the tail nodes |d| >= 2q", "effort units [0, 100]", "none", False,
                        V2M + ":recovery_metrics"),
    "stage2_tail_argmax_d": ("Gap d at which the tail maximum occurs", "effort units (gap d)", "none", False,
                             V2M + ":recovery_metrics"),
    "stage2_tail_mean_over_g2_0": ("tail mean divided by e_2*(0) (G-A criterion)", "fraction of e_2*(0)",
                                   "divided by e_2*(0)", False, V2M + ":recovery_metrics"),
    "stage2_tail_max_over_g2_0": ("tail max divided by e_2*(0)", "fraction of e_2*(0)", "divided by e_2*(0)", False,
                                  V2M + ":recovery_metrics"),
    "stage2_sym_err_max": ("Symmetry error: max over the recovery grid of |e_hat_2(d) - e_hat_2(-d)|",
                           "effort units", "none", False, V2M + ":recovery_metrics"),
    "stage2_sym_err_max_over_g2_0": ("Symmetry error divided by e_2*(0)", "fraction of e_2*(0)", "divided by e_2*(0)",
                                     False, V2M + ":recovery_metrics"),
    "stage2_sym_err_argmax_d": ("|d| at which the symmetry error is maximal", "effort units (gap d)", "none", False,
                                V2M + ":recovery_metrics"),
    "recovery_grid_step": ("Spacing of the recovery grid on D_2", "effort units (gap d)", "none", False,
                           V2M + ":recovery_metrics"),
    "recovery_n_pos": ("Recovery-grid nodes with |d| < 2q", "count", "none", False, V2M + ":recovery_metrics"),
    "recovery_n_tail": ("Recovery-grid nodes with |d| >= 2q", "count", "none", False, V2M + ":recovery_metrics"),
    # ---- decomposition of the stage-1 error (residual band, final tier) ----------------------------------------------------
    "e_tilde": ("Induced stage-1 target: argmin_e Delta_1(e; e_hat_2) on the final tier (residual minimisation)",
                "effort units", "none", "final", V2M + ":stage1_residual_sweep"),
    "band_lo": ("Lower end of the band {e: Delta_1(e) <= Delta_1,min + floor}", "effort units", "none", "final",
                V2M + ":induced_band"),
    "band_hi": ("Upper end of the band", "effort units", "none", "final", V2M + ":induced_band"),
    "learning_rel": ("Learning term (e_hat_1 - e_tilde) / e_1*", "fraction of e_1*(0)", "divided by e_1*(0), signed",
                     "final", "tools/v2/decomposition.py"),
    "learning_rel_lo": ("Lower end of the learning-term interval [e_hat_1 - band_hi, e_hat_1 - band_lo] / e_1*",
                        "fraction of e_1*(0)", "divided by e_1*(0)", "final", "tools/v2/decomposition.py"),
    "learning_rel_hi": ("Upper end of the learning-term interval", "fraction of e_1*(0)", "divided by e_1*(0)", "final",
                        "tools/v2/decomposition.py"),
    "learning_contains_0": ("Whether the learning-term interval contains 0", "bool", "none", "final",
                            "tools/v2/decomposition.py"),
    "inherited_rel": ("Inherited term (e_tilde - e_1*) / e_1*", "fraction of e_1*(0)", "divided by e_1*(0), signed",
                      "final", "tools/v2/decomposition.py"),
    "inherited_rel_lo": ("Lower end of the inherited-term interval [band_lo - e_1*, band_hi - e_1*] / e_1*",
                         "fraction of e_1*(0)", "divided by e_1*(0)", "final", "tools/v2/decomposition.py"),
    "inherited_rel_hi": ("Upper end of the inherited-term interval", "fraction of e_1*(0)", "divided by e_1*(0)", "final",
                         "tools/v2/decomposition.py"),
    "inherited_contains_0": ("Whether the inherited-term interval contains 0", "bool", "none", "final",
                             "tools/v2/decomposition.py"),
    "total_rel": ("Total stage-1 error (e_hat_1 - e_1*) / e_1* = learning + inherited", "fraction of e_1*(0)",
                  "divided by e_1*(0), signed", "final", "tools/v2/decomposition.py"),
    "e1_inside_sweep": ("Whether e_hat_1(0) lies inside the sweep range of the induced-band computation", "bool", "none",
                        "final", "tools/v2/induced_band.py"),
    "stage1_learning_err_rel": ("Learning term /e_1* from the superseded bracketing+Brent solver (Pilot 2 Appendix S)",
                                "fraction of e_1*(0)", "divided by e_1*(0)", True, V2M + ":induced_stage1_target"),
    "stage1_inherited_err_rel": ("Inherited term /e_1* from the superseded solver", "fraction of e_1*(0)",
                                 "divided by e_1*(0)", True, V2M + ":induced_stage1_target"),
    # ---- training diagnostics --------------------------------------------------------------------------------------
    "kl_final_epoch": ("Approximate KL of the actor update in the last PPO epoch, last update", "nats", "none", None,
                       "agents/ppo_curriculum.py:update"),
    "clip_frac": ("Fraction of minibatch rows whose PPO ratio was clipped, last update", "fraction", "none", None,
                  "agents/ppo_curriculum.py:update"),
    "wall_sec": ("Wall-clock seconds of the whole run (or of the named phase)", "seconds", "none", None,
                 "run records (status.json / v2_run_summary.json)"),
    "returncode": ("Process exit code", "integer", "none", None, "launch_*.json"),
    # ---- paired / summary statistics ------------------------------------------------------------------------------------
    "n": ("Number of observations (runs or pairs) behind the row", "count", "none", None, "pack builder"),
    "n_pairs": ("Number of paired runs (same q and seed)", "count", "none", None, "pack builder"),
    "mean": ("Mean over the observations", "as the metric", "none", None, "pack builder"),
    "sd": ("Sample SD (ddof = 1) over the observations", "as the metric", "none", None, "pack builder"),
    "median": ("Median over seeds", "as the metric", "none", None, "numpy.median"),
    "q25": ("25th percentile over seeds (numpy linear interpolation)", "as the metric", "none", None,
            "numpy.percentile"),
    "q75": ("75th percentile over seeds (numpy linear interpolation)", "as the metric", "none", None,
            "numpy.percentile"),
    "p10": ("10th percentile over seeds (numpy linear interpolation)", "as the metric", "none", None,
            "numpy.percentile"),
    "p25": ("25th percentile over seeds (numpy linear interpolation)", "as the metric", "none", None,
            "numpy.percentile"),
    "p75": ("75th percentile over seeds (numpy linear interpolation)", "as the metric", "none", None,
            "numpy.percentile"),
    "p90": ("90th percentile over seeds (numpy linear interpolation)", "as the metric", "none", None,
            "numpy.percentile"),
    "min": ("Minimum over the observations", "as the metric", "none", None, "numpy.min"),
    "max": ("Maximum over the observations", "as the metric", "none", None, "numpy.max"),
    "n_neg": ("Pairs with a negative difference", "count", "none", None, "pack builder"),
    "n_pos": ("Pairs with a positive difference", "count", "none", None, "pack builder"),
    "n_zero": ("Pairs with a zero difference", "count", "none", None, "pack builder"),
    "boot_ci95_lo": ("Lower end of the 95% percentile-bootstrap CI of the mean paired difference "
                     "(10,000 resamples of the pairs, numpy seed 20261001)", "as the metric", "none", None,
                     "tools/v2/pilot4_common.py:paired_summary"),
    "boot_ci95_hi": ("Upper end of the 95% percentile-bootstrap CI of the mean paired difference", "as the metric",
                     "none", None, "tools/v2/pilot4_common.py:paired_summary"),
    "cp95_lo": ("Lower end of the exact (Clopper-Pearson) 95% CI of a pass rate", "probability", "none", None,
                "tools/v2/confirmation_analysis.py"),
    "cp95_hi": ("Upper end of the exact (Clopper-Pearson) 95% CI of a pass rate", "probability", "none", None,
                "tools/v2/confirmation_analysis.py"),
    "label": ("Human-readable name of the metric in the paired table", "text", "none", None, "pack builder"),
    "metric": ("Name of the metric (column name of the per-run table, or a label)", "text", "none", None,
               "pack builder"),
    "comparison": ("Compared arms, first minus second", "label", "none", None, "pack builder"),
    "better": ("Direction that counts as better for the metric", "text", "none", None, "pack builder"),
    "note": ("Free-text note", "text", "none", None, "pack builder"),
    "notes": ("Free-text notes", "text", "none", None, "pack builder"),
    "source": ("File or report the value was read from", "path", "none", None, "pack builder"),
    "value": ("Value of the quantity named in the row", "as stated", "none", None, "pack builder"),
    "status": ("Status of the row (completed / found / ...)", "label", "none", None, "pack builder"),
}

# ----- rule tables ---------------------------------------------------------------------------------------------------
_PREFIX = [
    ("final_tier__", "Final-tier value (last checkpoint of the run) of "),
    ("A_dmf_", "dev - final difference of the end-of-Phase-A value of "),
    ("B_dmf_", "dev - final difference of the end-of-Phase-B value of "),
    ("A_", "End of Phase A (stage-2 last iterate, u1600), final tier unless the column says otherwise: "),
    ("B_", "End of Phase B (full last iterate, u2200), final tier unless the column says otherwise: "),
    ("dec_", "Stage-1 decomposition at the end of Phase B (residual band, final tier): "),
    ("phase1_", "Phase 1 value of "),
    ("diff_vs_phase1_", "Difference to the Phase 1 value of "),
]
_SUFFIX = [
    ("_dev_minus_final", "development tier minus final tier", "dev - final"),
    ("_final", "final tier", "final"),
    ("_dev", "development tier", "development"),
    ("_development", "development tier", "development"),
]


def _tier_label(tier_dep: Any, suffix_tier: Optional[str], table_tier: str) -> str:
    if suffix_tier:
        return suffix_tier
    if tier_dep == "final":
        return "final"
    if tier_dep is False:
        return "tier-independent"
    if tier_dep is True:
        return table_tier or "unspecified"
    return "n/a"


def lookup(col: str, table_tier: str = "") -> Optional[Dict[str, str]]:
    """Dictionary entry of a column, or ``None`` if no rule covers it.

    Args:
        col: Column name as it appears in the pack table.
        table_tier: Tier of the table (``development``, ``final`` or a mix) for unsuffixed
            tier-dependent columns.

    Returns:
        ``{definition, units, normalization, tier, source}`` or ``None``.
    """
    def pack(ent, prefix_txt="", suffix_tier=None, suffix_note=""):
        d, u, n, td, s = ent
        if suffix_note:
            d = f"{d} [{suffix_note}]"
        return {"definition": prefix_txt + d if prefix_txt else d, "units": u, "normalization": n,
                "tier": _tier_label(td, suffix_tier, table_tier), "source": s}

    if col in E:
        return pack(E[col])
    # prefix rules (peel one prefix, then retry exact/suffix)
    for pre, txt in _PREFIX:
        if col.startswith(pre) and len(col) > len(pre):
            inner = lookup(col[len(pre):], "final" if pre in ("A_", "B_", "dec_", "final_tier__") else table_tier)
            if inner is not None:
                inner = dict(inner)
                inner["definition"] = txt + inner["definition"]
                return inner
    # suffix rules
    for suf, note, tlabel in _SUFFIX:
        if col.endswith(suf) and len(col) > len(suf):
            base = col[: -len(suf)]
            for cand in (base, base + "_over_dw", base.replace("eta", "eta_T")):
                if cand in E:
                    return pack(E[cand], suffix_tier=tlabel, suffix_note=note)
    # families
    m = re.match(r"^inv_(.+)$", col)
    if m:
        return {"definition": "Invariant residual '" + m.group(1) + "' (raw, never clamped): *_excess_over_dw = (lhs - rhs)/Delta W "
                "of the inequality (positive = violated); *_absdiff_over_dw = max |lhs - rhs|/Delta W of the equality; "
                "*_at_* = location of the maximum", "units": "Delta W (dimensionless) or location",
                "normalization": "divided by Delta W", "tier": _tier_label(True, None, table_tier),
                "source": V2M + ":_invariants"}
    m = re.match(r"^stage2_drift_(live|cand)_(.+)$", col)
    if m:
        who = "live stage-2 network output" if m.group(1) == "live" else "candidate's stage-2 mapping (frozen snapshot in B arms)"
        return {"definition": f"Drift |e_hat_2 - e_hat_2^parent| of the {who}, statistic '{m.group(2)}' "
                "(on / off = on-path / off-path set of the open rule |d - drift| < 2q)", "units": "effort units "
                "[0, 100] (alpha/beta in Beta parameter units)", "normalization": "none",
                "tier": "development", "source": "run/run_v2_stagewise.py (drift_vs_parent) with " + V2M + ":onoff_split"}
    return None
