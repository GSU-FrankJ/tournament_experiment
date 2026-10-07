#!/usr/bin/env python3
"""MS-R1 pre-registered analysis of the pilot (prompt section 3.2): per-run table, paired tables, criterion.

Inputs (all read only; every root is an explicit argument and is recorded in ``analysis_info.json``,
``summary.txt``, the figure footers, a ``roots_id`` column of every CSV and the ``run_dir`` of every row of
``per_run.csv``):

  * ``--base-root``      ``q*/seed*/MS_base`` (runs of ``run/run_ms_stagewise.py``, arm ``MS_base``)
  * ``--pilot-root``     ``q*/seed*/<arm>``    (the five rule arms ``MS_rule``, ``MS_s25a0``, ``MS_s25a5``,
                         ``MS_s35a0``, ``MS_s35a5``)
  * ``--parents-root``   ``q*/seed*``          (v2.0 ``parents_A``: ``phase_A`` runs, ``final_v2.json`` with both
                         verifier tiers, ``status.json``, ``v2_run_summary.json``)
  * ``--rehearsal-root`` ``q*/seed*``          (``rehearsal_v2_0``: locked v2.0 runs, ``gates.json`` with
                         ``reported.end_of_A`` / ``reported.end_of_B``, ``induced_band.json``)

One row per (arm, q, seed) of the six MS arms plus the two comparators as pseudo-arms ``parents_A`` and
``rehearsal_v2_0``. A missing, running, failed (``status.json`` not ``done`` / exit code 0, which includes a
global-RNG violation, exit code 5) or unreadable run is a row with its ``status`` and ``status_info``, never a
silent skip; it never enters a paired difference.

Pre-registered criterion (``criterion.csv``; descriptive, not a gate, no selection follows; the decision is the
PI's). Primary metric ``stage2_peak_rel_err_abs`` (|signed peak error| of the frozen terminal-stage candidate,
final tier). (a) The 95% percentile bootstrap interval of the mean paired difference ``arm - parents_A`` lies
below 0 (``ci_mean_hi < 0``) at BOTH q. (b) No run that passes G-A with its G-N eta part under ``parents_A`` fails
it under the arm; "passes" = G-A on the final tier (eta_2/DW <= 0.005, RMSE_pos/e2*(0) <= 0.05, tail mean /
e2*(0) <= 0.02) AND |eta_2/DW(development) - eta_2/DW(final)| <= 0.001 (the eta part of G-N), computed from the
two tiers' scalars of ``final_v2.json`` for ``parents_A`` and of ``gates.json`` for the MS arms (thresholds from
``protocols/v2_T2_locked_v2_0.json``). A baseline-passing run whose arm run is ``done`` and fails, or is
``failed``, is a violation; one whose arm run is missing / running / incomplete is pending. Bootstrap: 10,000
resamples, ``numpy.random.default_rng(20261006)``, ONE FRESH generator per (q, statistic) in table order, the
resampled indices are ``rng.integers(0, n, size=(10000, n))`` and the interval is the 2.5 / 97.5 percentiles of
the resampled statistic. Everything that is not this criterion is descriptive; an interval that contains 0 with
10 seeds does not show that a mechanism has no effect.

Outputs under ``--out``: ``per_run.csv`` (columns in :data:`COLUMNS`), ``paired_vs_parents_A.csv`` (terminal
stage), ``paired_vs_rehearsal_v2_0.csv`` (stage 1), ``paired_vs_MS_rule.csv``, ``paired_seed_level.csv``,
``criterion.csv``, ``arm_summary.csv``, ``budget.csv``, ``rule.csv``, ``rule_blocks.csv``, ``tail.csv``,
``decomposition.csv``, ``stage1.csv``, ``start_shares_by_block.csv``, ``completeness.csv``,
``decision_inputs.csv``, ``analysis_info.json``, ``summary.txt``, ``figures/*.png``. No timestamps are written:
the tool is deterministic given the same inputs.

Exit code: 0 when every planned run is done; 3 when a run is missing / failed / incomplete (reported in the
tables and on stderr); 2 on an argument error.

Usage:
    python tools/ms/r1_analysis.py --base-root results/ms_r1/base --pilot-root results/ms_r1/pilot \
        --parents-root <v2-t2-refine>/results/v2_refine/parents_A \
        --rehearsal-root <v2-t2-refine>/results/v2_T2_locked/rehearsal_v2_0 --out results/ms_r1/analysis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "protocols" / "v2_T2_locked_v2_0.json"

# --------------------------------------------------------------------------------------------- constants
N_BOOT = 10000
BOOT_SEED = 20261006
QS: Tuple[int, ...] = (50, 60)
SEEDS: Tuple[int, ...] = tuple(range(10501, 10511))
MS_ARMS: Tuple[str, ...] = ("MS_base", "MS_rule", "MS_s25a0", "MS_s25a5", "MS_s35a0", "MS_s35a5")
BASE_ARM = "MS_base"
CONTROL_ARM = "MS_rule"
PARENTS = "parents_A"
REHEARSAL = "rehearsal_v2_0"
COMPARATORS: Tuple[str, ...] = (PARENTS, REHEARSAL)
PRIMARY = "stage2_peak_rel_err_abs"
PRIMARY_S1 = "stage1_rel_err_abs"
SIGNED = "stage2_peak_rel_err_signed"
TAIL_PEAK = 0.05                                    # |peak error| <= 0.05, inclusive
NO_EFFECT_SENTENCE = ("An interval that contains 0 with 10 seeds does not show that a mechanism has no effect "
                      "(descriptive: the criterion is the pre-registered statement, nothing else is a test).")
CRITERION_NOTE = ("pre-registered criterion (descriptive, not a gate); no selection rule and no protocol change "
                  "follow from it")
STATUSES = ("done", "failed", "running", "incomplete", "missing")
D1_STAGE2 = ("L_s2", "O_s2", "L_s2_in", "L_s2_out")
D1_STAGE1 = ("L_s1", "O_s1")
D1_PARTS = ("n", "lo", "hi")
START_COLS = ("n_start_tail", "n_start_near", "n_start_mid")
REQUIRED_FILES = ("status.json", "gates.json", "rule_log.json", "manifest.json", "ms_checks_stage2.csv",
                  "ms_updates.csv", "induced_band.json")


# --------------------------------------------------------------------------------------------- columns
ID_COLS = ["arm", "q", "seed", "role", "source_root", "status", "status_info", "exit_code", "complete",
           "run_dir", "flags", "git_commit", "git_dirty", "outcome", "global_rng_ok"]
DESIGN_COLS = ["rule_enabled", "sw_scheme", "sw_lambda_P", "sw_alpha_global", "sw_alpha_polish",
               "design_lambda_T", "design_lambda_P", "design_lambda_M"]
TERMINAL_COLS = [
    "stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_rmse_pos_over_g2_0",
    "stage2_tail_mean_over_g2_0", "stage2_tail_max_over_g2_0", "e2_at_0", "g2_at_0", "sigma_effort_at_0_t2",
    "eta_T_over_dw", "eta_dev", "eta_dev_minus_final_abs", "G_A_eta_pass", "G_A_rmse_pass", "G_A_tail_pass",
    "G_A_pass", "G_N_eta_pass", "gate_pass", "smoothed_share_peak_gap_d0", "smoothed_e_pred_0",
    "smoothed_e_learned_0"]
D3_STAGE2 = ["t2_Delta_final", "t2_Delta_dev", "t2_s_final", "t2_s_dev", "t2_R_final", "t2_R_dev",
             "t2_Rtail_final", "t2_Rtail_dev", "t2_C_final", "t2_C_dev", "t2_tail_term", "t2_R_argmax_d_final"]
BUDGET2 = ["t2_updates", "t2_training_updates", "t2_episodes", "t2_minibatch_steps", "t2_wall_sec",
           "t2_cpu_sec", "t2_dev_verifier_sec"]
RULE2 = ["t2_fire_local", "t2_would_fire_local", "t2_budget_forced", "t2_n_checks", "t2_n_blocks",
         "t2_n_global_blocks", "t2_n_polish_blocks", "t2_block_types", "t2_classifications",
         "t2_n_localized", "t2_n_broad", "t2_nS_at_classification", "t2_landing_first_local",
         "t2_landing_followed_type"]
SHARES2 = ["t2_n_updates_train", "t2_n_updates_landing", "t2_n_updates_polish", "t2_n_start_tail_train",
           "t2_n_start_near_train", "t2_n_start_mid_train", "t2_share_tail_train",
           "t2_share_near_train", "t2_share_mid_train", "t2_share_tail_all", "t2_share_near_all",
           "t2_share_mid_all", "t2_block_share_tail", "t2_block_share_near", "t2_block_share_mid"]
CLAMP2 = ["t2_d1_%s_%s" % (s, p) for s in D1_STAGE2 for p in D1_PARTS] + ["t2_clamp_L_frac", "t2_clamp_O_frac"]
STAGE1_COLS = [
    "stage1_rel_err_signed", "stage1_rel_err_abs", "e1_at_0", "g1", "sigma_effort_at_0_t1",
    "Gmax_full_over_dw", "gmax_dev", "gmax_dev_minus_final_abs", "G_F_pass", "G_N_gmax_pass", "G_S_pass",
    "S1_pass", "v20_combination_pass", "v20_combination_pass_recorded"]
D3_STAGE1 = ["t1_Delta_final", "t1_Delta_dev", "t1_s_final", "t1_s_dev", "t1_R_final", "t1_R_dev",
             "t1_C_final", "t1_C_dev"]
BAND_COLS = ["band_e_tilde", "band_lo", "band_hi", "learning_rel", "learning_rel_lo", "learning_rel_hi",
             "learning_contains_0", "inherited_rel", "inherited_rel_lo", "inherited_rel_hi",
             "inherited_contains_0", "band_contiguous", "e1_inside_sweep"]
BUDGET1 = ["t1_updates", "t1_training_updates", "t1_episodes", "t1_minibatch_steps", "t1_wall_sec",
           "t1_cpu_sec", "t1_dev_verifier_sec"]
RULE1 = ["t1_fire_local", "t1_would_fire_local", "t1_budget_forced", "t1_n_checks", "t1_n_blocks",
         "t1_block_types", "t1_landing_first_local"]
CLAMP1 = ["t1_d1_%s_%s" % (s, p) for s in D1_STAGE1 for p in D1_PARTS] + ["t1_clamp_L_frac"]
TOTAL_COLS = ["total_updates", "total_episodes", "total_minibatch_steps", "total_wall_sec"]
COLUMNS: List[str] = (ID_COLS + DESIGN_COLS + TERMINAL_COLS + D3_STAGE2 + BUDGET2 + RULE2 + SHARES2 + CLAMP2
                      + STAGE1_COLS + D3_STAGE1 + BAND_COLS + BUDGET1 + RULE1 + CLAMP1 + TOTAL_COLS)
STR_COLS = {"arm", "role", "source_root", "status", "status_info", "run_dir", "flags", "git_commit", "outcome",
            "sw_scheme", "t2_block_types", "t2_classifications", "t2_nS_at_classification",
            "t2_landing_followed_type", "t2_block_share_tail", "t2_block_share_near", "t2_block_share_mid",
            "t1_block_types"}
BOOL_COLS = {"complete", "git_dirty", "global_rng_ok", "rule_enabled", "G_A_eta_pass", "G_A_rmse_pass",
             "G_A_tail_pass", "G_A_pass", "G_N_eta_pass", "gate_pass", "t2_tail_term", "t2_budget_forced",
             "G_F_pass", "G_N_gmax_pass", "G_S_pass", "S1_pass", "v20_combination_pass",
             "v20_combination_pass_recorded", "learning_contains_0", "inherited_contains_0", "band_contiguous",
             "e1_inside_sweep", "t1_budget_forced"}

#: (column, direction): True = smaller is better, None = no direction
S2_METRICS: List[Tuple[str, Optional[bool]]] = [
    (PRIMARY, True), (SIGNED, None), ("stage2_rmse_pos_over_g2_0", True), ("stage2_tail_mean_over_g2_0", True),
    ("stage2_tail_max_over_g2_0", True), ("eta_T_over_dw", True), ("eta_dev", True),
    ("eta_dev_minus_final_abs", True), ("sigma_effort_at_0_t2", None), ("e2_at_0", None),
    ("smoothed_share_peak_gap_d0", None), ("t2_Delta_final", True), ("t2_R_final", True),
    ("t2_Rtail_final", True), ("t2_updates", True), ("t2_episodes", True), ("t2_minibatch_steps", True),
    ("t2_wall_sec", True)]
S1_METRICS: List[Tuple[str, Optional[bool]]] = [
    (PRIMARY_S1, True), ("stage1_rel_err_signed", None), ("e1_at_0", None), ("Gmax_full_over_dw", True),
    ("gmax_dev_minus_final_abs", True), ("learning_rel", None), ("inherited_rel", None),
    ("sigma_effort_at_0_t1", None), ("t1_Delta_final", True), ("t1_R_final", True), ("t1_updates", True),
    ("t1_wall_sec", True), ("total_updates", True), ("total_wall_sec", True)]
DIRECTION: Dict[str, Optional[bool]] = {m: d for m, d in S2_METRICS + S1_METRICS}


# --------------------------------------------------------------------------------------------- small helpers
def _f(x: Any) -> float:
    """Float of ``x``; NaN for None / non-numeric."""
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _fin(x: Any) -> bool:
    """True if ``x`` is a finite number."""
    return bool(np.isfinite(_f(x)))


def _truthy(v: Any) -> bool:
    """Bool of a value read back from a CSV / JSON (NaN and empty are False)."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() == "true"
    if v is None:
        return False
    try:
        return bool(v) and not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _bool_or_nan(v: Any) -> Any:
    """``bool(v)`` or NaN when ``v`` is None / NaN (a quantity that is not defined)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return float("nan")
    return bool(v)


def _jload(path: Path, anomalies: List[str], required: bool = True) -> Dict[str, Any]:
    """JSON file as a dict; ``{}`` (and an anomaly, if ``required``) when missing or unreadable."""
    try:
        with open(path) as f:
            out = json.load(f)
        return out if isinstance(out, dict) else {}
    except FileNotFoundError:
        if required:
            anomalies.append("%s missing" % path.name)
    except Exception as exc:  # noqa: BLE001
        anomalies.append("%s unreadable: %s: %s" % (path.name, type(exc).__name__, exc))
    return {}


def _get(d: Mapping[str, Any], *keys: str) -> Any:
    """Nested ``d[k0][k1]...``; None when any level is missing."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _join(items: Iterable[Any], sep: str = ";") -> str:
    """Separator-joined string of the items (floats in ``%.6g``)."""
    return sep.join(("%.6g" % x) if isinstance(x, float) else str(x) for x in items)


def roots_id(roots: Mapping[str, str]) -> str:
    """12-hex digest of the four resolved roots (a ``roots_id`` column in every CSV; the roots are in
    ``analysis_info.json``)."""
    return hashlib.sha256(json.dumps(dict(sorted(roots.items())), sort_keys=True).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------------------------- thresholds
@dataclass(frozen=True)
class Thresholds:
    """The gate thresholds of protocol v2.0 (inclusive ``<=``)."""

    eta: float
    rmse: float
    tail: float
    n_eta: float
    n_gmax: float
    g_f: float
    g_s: float
    s1: float


def thresholds_from_protocol(proto: Mapping[str, Any]) -> Thresholds:
    """Thresholds from a protocol dict (``gates`` and ``secondary``; also the ``protocol_gates`` of a run config).

    Args:
        proto: Mapping with ``gates`` (G-A, G-F, G-N, G-S ``all_must_hold`` lists) and ``secondary.S1``.
    """
    g = proto["gates"]
    ta = {c["metric"]: c["threshold"] for c in g["G-A"]["all_must_hold"]}
    tf = {c["metric"]: c["threshold"] for c in g["G-F"]["all_must_hold"]}
    tn = {c["metric"]: c["threshold"] for c in g["G-N"]["all_must_hold"]}
    ts = {c["metric"]: c["threshold"] for c in g["G-S"]["all_must_hold"]}
    return Thresholds(
        eta=float(ta["eta_T_over_dw"]), rmse=float(ta["stage2_rmse_pos_over_g2_0"]),
        tail=float(ta["stage2_tail_mean_over_g2_0"]), n_eta=float(tn["eta_T_over_dw_dev_minus_final_abs"]),
        n_gmax=float(tn["Gmax_full_over_dw_dev_minus_final_abs"]), g_f=float(tf["Gmax_full_over_dw"]),
        g_s=float(ts["stage1_rel_err_abs"]), s1=float(proto["secondary"]["S1"]["criterion"]["threshold"]))


def load_thresholds(path: Path) -> Tuple[Thresholds, str]:
    """Thresholds and the SHA-256 of the protocol file."""
    raw = Path(path).read_bytes()
    return thresholds_from_protocol(json.loads(raw.decode())), hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------------------------- run status
def run_status(d: Path) -> Tuple[str, str, float, Dict[str, Any]]:
    """Status of a run directory from ``status.json``.

    Returns:
        ``(status, info, exit_code, status_json)`` with ``status`` one of ``done`` (state done and exit code
        0), ``failed`` (state failed, or a non-zero exit code incl. the global-RNG violation 5), ``running``,
        ``missing`` (no directory or no ``status.json``). ``incomplete`` is assigned later by the extractors.
    """
    if not d.is_dir():
        return "missing", "run directory absent", float("nan"), {}
    sp = d / "status.json"
    if not sp.exists():
        return "missing", "no status.json", float("nan"), {}
    try:
        with open(sp) as f:
            st = json.load(f)
    except Exception as exc:  # noqa: BLE001
        return "failed", "status.json unreadable: %s" % exc, float("nan"), {}
    state, code = st.get("state"), st.get("exit_code")
    if state == "running":
        return "running", "state running (pid %s)" % st.get("pid"), float("nan"), st
    if state == "done" and code == 0:
        return "done", "", 0.0, st
    if state == "done":
        extra = " (global-RNG violation)" if code == 5 else ""
        return "failed", "state done with exit_code %s%s" % (code, extra), _f(code), st
    if state == "failed":
        tb = str(st.get("traceback", "")).strip().splitlines()
        return "failed", "state failed: %s" % (tb[-1] if tb else "no traceback"), _f(code), st
    return "failed", "unknown state %r" % (state,), _f(code), st


# --------------------------------------------------------------------------------------------- row building
def new_row(arm: str, q: int, seed: int, role: str, run_dir: Path, source_root: str) -> Dict[str, Any]:
    """A row with every column of :data:`COLUMNS` at its undefined value (NaN, or "" for strings)."""
    row: Dict[str, Any] = {c: ("" if c in STR_COLS else float("nan")) for c in COLUMNS}
    row.update(arm=arm, q=int(q), seed=int(seed), role=role, source_root=source_root, run_dir=str(run_dir),
               complete=False)
    return row


def terminal_cols(fin: Mapping[str, Any], dev: Mapping[str, Any], th: Thresholds) -> Dict[str, Any]:
    """Terminal-stage closed-form columns and the G-A / G-N eta verdicts from the two tiers' scalars."""
    out: Dict[str, Any] = {k: _f(fin.get(k)) for k in (
        "stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_rmse_pos_over_g2_0",
        "stage2_tail_mean_over_g2_0", "stage2_tail_max_over_g2_0", "e2_at_0", "g2_at_0",
        "sigma_effort_at_0_t2", "eta_T_over_dw")}
    eta, rmse, tail = out["eta_T_over_dw"], out["stage2_rmse_pos_over_g2_0"], out["stage2_tail_mean_over_g2_0"]
    out["eta_dev"] = _f(dev.get("eta_T_over_dw"))
    out["eta_dev_minus_final_abs"] = abs(out["eta_dev"] - eta)
    out["G_A_eta_pass"] = bool(eta <= th.eta)
    out["G_A_rmse_pass"] = bool(rmse <= th.rmse)
    out["G_A_tail_pass"] = bool(tail <= th.tail)
    out["G_A_pass"] = bool(out["G_A_eta_pass"] and out["G_A_rmse_pass"] and out["G_A_tail_pass"])
    out["G_N_eta_pass"] = bool(out["eta_dev_minus_final_abs"] <= th.n_eta)
    out["gate_pass"] = bool(out["G_A_pass"] and out["G_N_eta_pass"])
    return out


def stage1_cols(fin: Mapping[str, Any], dev: Mapping[str, Any], th: Thresholds) -> Dict[str, Any]:
    """Stage-1 columns (G-S error, G-F, the Gmax part of G-N, S1) from the stage-1 freeze scalars."""
    out: Dict[str, Any] = {k: _f(fin.get(k)) for k in (
        "stage1_rel_err_signed", "stage1_rel_err_abs", "e1_at_0", "g1", "sigma_effort_at_0_t1",
        "Gmax_full_over_dw")}
    out["gmax_dev"] = _f(dev.get("Gmax_full_over_dw"))
    out["gmax_dev_minus_final_abs"] = abs(out["gmax_dev"] - out["Gmax_full_over_dw"])
    out["G_F_pass"] = bool(out["Gmax_full_over_dw"] <= th.g_f)
    out["G_N_gmax_pass"] = bool(out["gmax_dev_minus_final_abs"] <= th.n_gmax)
    out["G_S_pass"] = bool(out["stage1_rel_err_abs"] <= th.g_s)
    out["S1_pass"] = bool(out["stage1_rel_err_abs"] <= th.s1)
    return out


def d3_cols(freeze: Mapping[str, Any], t: int) -> Dict[str, Any]:
    """D3 quantities at the freeze of stage ``t`` on both tiers (``rule_log.json`` ``freeze.d3_*``)."""
    p = "t%d_" % t
    out: Dict[str, Any] = {}
    for tier, name in (("d3_final", "final"), ("d3_development", "dev")):
        d3 = freeze.get(tier) or {}
        out[p + "Delta_" + name] = _f(d3.get("Delta"))
        out[p + "s_" + name] = _f(d3.get("s"))
        out[p + "R_" + name] = _f(d3.get("R"))
        out[p + "C_" + name] = _f(d3.get("C"))
        if t == 2:
            out[p + "Rtail_" + name] = _f(d3.get("R_tail"))
    if t == 2:
        d3f = freeze.get("d3_final") or {}
        out["t2_tail_term"] = _bool_or_nan(d3f.get("tail_term"))
        out["t2_R_argmax_d_final"] = _f(d3f.get("R_argmax_d"))
    return out


def rule_cols(rec: Mapping[str, Any], t: int) -> Dict[str, Any]:
    """Budget and rule-record columns of stage ``t`` from its ``rule_log.json`` entry."""
    p = "t%d_" % t
    blocks = list(rec.get("blocks") or [])
    types = [str(b.get("type")) for b in blocks]
    classified = [b for b in blocks if b.get("classification")]
    landing = rec.get("landing") or {}
    out: Dict[str, Any] = {
        p + "updates": _f(rec.get("total_updates")), p + "training_updates": _f(rec.get("training_updates")),
        p + "episodes": _f(rec.get("episodes")), p + "minibatch_steps": _f(rec.get("minibatch_steps")),
        p + "wall_sec": _f(rec.get("wall_sec")), p + "fire_local": _f(rec.get("fire_local")),
        p + "would_fire_local": _f(rec.get("would_fire_local")),
        p + "budget_forced": _bool_or_nan(rec.get("budget_forced")), p + "n_checks": _f(rec.get("n_checks")),
        p + "n_blocks": float(len(blocks)), p + "block_types": ">".join(types),
        p + "landing_first_local": _f(landing.get("first_local"))}
    if t == 2:
        out.update({"t2_n_global_blocks": float(sum(1 for x in types if x == "global")),
                    "t2_n_polish_blocks": float(sum(1 for x in types if x == "polish")),
                    "t2_classifications": ",".join(str(b["classification"]) for b in classified),
                    "t2_n_localized": float(sum(1 for b in classified if b["classification"] == "localized")),
                    "t2_n_broad": float(sum(1 for b in classified if b["classification"] == "broad")),
                    "t2_nS_at_classification": ",".join(str(int(b.get("n_S") or 0)) for b in classified),
                    "t2_landing_followed_type": str(landing.get("followed_block_type") or "")})
    return out


def timing_cols(summary: Mapping[str, Any], t: int) -> Dict[str, Any]:
    """Process CPU seconds and development-verifier seconds of stage ``t`` (``ms_run_summary.json``)."""
    ph = _get(summary, "phase_timing", "stage%d" % t) or {}
    p = "t%d_" % t
    return {p + "cpu_sec": _f(ph.get("process_cpu_sec")),
            p + "dev_verifier_sec": _f(_get(ph, "by_category_sec", "dev_verifier"))}


def band_cols(ib: Mapping[str, Any]) -> Dict[str, Any]:
    """Induced-target decomposition columns (``induced_band.json``)."""
    out: Dict[str, Any] = {"band_e_tilde": _f(ib.get("e_tilde")), "band_lo": _f(ib.get("band_lo")),
                           "band_hi": _f(ib.get("band_hi"))}
    for k in ("learning_rel", "learning_rel_lo", "learning_rel_hi", "inherited_rel", "inherited_rel_lo",
              "inherited_rel_hi"):
        out[k] = _f(ib.get(k))
    for k_out, k_in in (("learning_contains_0", "learning_contains_0"),
                        ("inherited_contains_0", "inherited_contains_0"),
                        ("band_contiguous", "band_contiguous"), ("e1_inside_sweep", "e1_inside_sweep")):
        out[k_out] = _bool_or_nan(ib.get(k_in))
    return out


def update_cols(u: pd.DataFrame, anomalies: List[str]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Start shares and raw-draw clamp counts from ``ms_updates.csv``.

    Returns:
        ``(columns, blocks)``: the per-run columns and the long table of per-block start shares of the
        terminal stage (one row per (block id, block type)).
    """
    out: Dict[str, Any] = {}
    s2 = u[u["stage"] == 2]
    tr = s2[s2["block_type"] != "landing"]
    for scope, sub in (("train", tr), ("all", s2)):
        tot = {c: float(sub[c].sum()) for c in START_COLS}
        n = sum(tot.values())
        for name, c in (("tail", "n_start_tail"), ("near", "n_start_near"), ("mid", "n_start_mid")):
            out["t2_share_%s_%s" % (name, scope)] = tot[c] / n if n > 0 else float("nan")
            if scope == "train":
                out["t2_%s_train" % c] = tot[c]
    out["t2_n_updates_train"] = float(len(tr))
    out["t2_n_updates_landing"] = float(len(s2) - len(tr))
    out["t2_n_updates_polish"] = float((s2["block_type"] == "polish").sum())
    blocks = []
    for (bid, btype), g in s2.groupby(["block_id", "block_type"], sort=True):
        tot = {c: float(g[c].sum()) for c in START_COLS}
        n = sum(tot.values())
        blocks.append({"stage": 2, "block_id": int(bid), "block_type": str(btype), "n_updates": int(len(g)),
                       "n_tail": tot["n_start_tail"], "n_near": tot["n_start_near"], "n_mid": tot["n_start_mid"],
                       "share_tail": tot["n_start_tail"] / n if n > 0 else float("nan"),
                       "share_near": tot["n_start_near"] / n if n > 0 else float("nan"),
                       "share_mid": tot["n_start_mid"] / n if n > 0 else float("nan")})
    bt = pd.DataFrame(blocks, columns=["stage", "block_id", "block_type", "n_updates", "n_tail", "n_near",
                                       "n_mid", "share_tail", "share_near", "share_mid"])
    out["t2_block_share_tail"] = _join(bt["share_tail"].tolist())
    out["t2_block_share_near"] = _join(bt["share_near"].tolist())
    out["t2_block_share_mid"] = _join(bt["share_mid"].tolist())
    for t, suffixes, prefix in ((2, D1_STAGE2, "t2_d1_"), (1, D1_STAGE1, "t1_d1_")):
        sub = u[u["stage"] == t]
        for s in suffixes:
            for part in D1_PARTS:
                col = "d1_%s_%s" % (s, part)
                if col not in u.columns:
                    anomalies.append("ms_updates.csv lacks column %s" % col)
                    continue
                out["%s%s_%s" % (prefix, s, part)] = float(sub[col].sum())
    for t, who, prefix in ((2, "L_s2", "t2_clamp_L_frac"), (2, "O_s2", "t2_clamp_O_frac"),
                           (1, "L_s1", "t1_clamp_L_frac")):
        key = "%sd1_%s_" % ("t%d_" % t, who)
        n = out.get(key + "n", float("nan"))
        if n and np.isfinite(n):
            out[prefix] = (out.get(key + "lo", float("nan")) + out.get(key + "hi", float("nan"))) / n
    return out, bt


def finish_row(row: Dict[str, Any], anomalies: List[str], need_stage1: bool) -> None:
    """Set ``complete``, demote a ``done`` run without readable metrics to ``incomplete``, join the flags.

    Raises:
        KeyError: If the row carries a key that is not a column (a typo would otherwise drop data silently).
    """
    extra = sorted(set(row) - set(COLUMNS))
    if extra:
        raise KeyError("row keys that are not per_run columns: %s" % extra)
    if row["status"] == "done":
        bad = []
        if not _fin(row[PRIMARY]):
            bad.append("terminal-stage metrics not readable")
        if need_stage1 and not _fin(row[PRIMARY_S1]):
            bad.append("stage-1 metrics not readable")
        if bad:
            row["status"] = "incomplete"
            row["status_info"] = "; ".join(bad)
    row["complete"] = bool(row["status"] == "done")
    row["flags"] = "; ".join(anomalies)


# --------------------------------------------------------------------------------------------- extractors
def extract_ms_run(d: Path, arm: str, q: int, seed: int, th: Thresholds, source: str
                   ) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """One MS run (``run/run_ms_stagewise.py`` outputs) as a per_run row.

    Returns:
        ``(row, block_shares, rule_blocks)``: the row, the terminal-stage per-block start shares and the
        flattened ``rule_log`` blocks of both stages (empty frames for a run without those files).
    """
    row = new_row(arm, q, seed, "ms_arm", d, source)
    status, info, code, stj = run_status(d)
    row.update(status=status, status_info=info, exit_code=code)
    an: List[str] = []
    empty_b = pd.DataFrame(columns=["stage", "block_id", "block_type", "n_updates", "n_tail", "n_near", "n_mid",
                                    "share_tail", "share_near", "share_mid"])
    empty_r: List[Dict[str, Any]] = []
    if status == "missing":
        finish_row(row, an, True)
        return row, empty_b, pd.DataFrame(empty_r)
    must = status == "done"
    gates = _jload(d / "gates.json", an, must)
    rl = _jload(d / "rule_log.json", an, must)
    cfg = _jload(d / "run_config.json", an, must)
    ib = _jload(d / "induced_band.json", an, must)
    summ = _jload(d / "ms_run_summary.json", an, must)
    if must:
        _jload(d / "manifest.json", an, True)
        for fn in ("ms_checks_stage2.csv", "ms_updates.csv"):
            if not (d / fn).exists():
                an.append("%s missing" % fn)
    git = stj.get("git") or {}
    row["git_commit"] = str(git.get("commit", ""))
    row["git_dirty"] = _bool_or_nan(git.get("dirty"))
    if git.get("dirty") is True:
        an.append("git tree dirty at run time")
    row["outcome"] = str(gates.get("outcome", ""))
    row["global_rng_ok"] = _bool_or_nan(None if not gates.get("global_rng") else
                                        gates["global_rng"].get("status") == "ok")
    T = _get(cfg, "record", "game", "T") or _get(cfg, "pipeline", "T")
    if T is not None and int(T) != 2:
        an.append("T = %s: the closed-form columns are defined for T = 2 only" % T)
    sw = cfg.get("start_weights") or {}
    row["rule_enabled"] = _bool_or_nan(_get(cfg, "rule", "enabled"))
    row["sw_scheme"] = str(sw.get("scheme", ""))
    row["sw_lambda_P"] = _f(sw.get("lambda_P"))
    row["sw_alpha_global"] = _f(sw.get("alpha_global"))
    row["sw_alpha_polish"] = _f(sw.get("alpha_polish"))
    der = _get(rl, "derived", "2") or _get(cfg, "derived", "2") or {}
    row["design_lambda_T"] = _f(der.get("lambda_T"))
    row["design_lambda_P"] = _f(der.get("lambda_P"))
    row["design_lambda_M"] = _f(der.get("lambda_M"))
    # ---- thresholds of the run's own config must equal the analysis thresholds
    pg = cfg.get("protocol_gates")
    if pg:
        try:
            if thresholds_from_protocol(pg) != th:
                an.append("run_config protocol_gates thresholds differ from the analysis thresholds")
        except Exception as exc:  # noqa: BLE001
            an.append("run_config protocol_gates unreadable: %s" % exc)
    stages = rl.get("stages") or {}
    fz2 = _get(stages, "2", "freeze") or {}
    fz1 = _get(stages, "1", "freeze") or {}
    rep = gates.get("reported") or {}
    a_fin = _get(rep, "end_of_stage2", "final") or fz2.get("final") or {}
    a_dev = _get(rep, "end_of_stage2", "development") or fz2.get("development") or {}
    if a_fin:
        row.update(terminal_cols(a_fin, a_dev, th))
        if gates.get("G-A") and "pass" in gates["G-A"] and bool(gates["G-A"]["pass"]) != row["G_A_pass"]:
            an.append("recomputed G-A verdict differs from gates.json")
    sg = _get(rep, "end_of_stage2", "smoothed_game") or fz2.get("smoothed_game") or {}
    row["smoothed_share_peak_gap_d0"] = _f(sg.get("smoothed_share_peak_gap_d0"))
    row["smoothed_e_pred_0"] = _f(sg.get("smoothed_e_pred_0"))
    row["smoothed_e_learned_0"] = _f(sg.get("smoothed_e_learned_0"))
    s_fin = _get(rep, "end_of_stage1", "final") or fz1.get("final") or {}
    s_dev = _get(rep, "end_of_stage1", "development") or fz1.get("development") or {}
    if s_fin:
        row.update(stage1_cols(s_fin, s_dev, th))
        if a_fin:
            row["v20_combination_pass"] = bool(row["G_A_pass"] and row["G_N_eta_pass"] and row["G_F_pass"]
                                               and row["G_N_gmax_pass"] and row["G_S_pass"])
        if "v20_combination_pass" in gates:
            row["v20_combination_pass_recorded"] = _bool_or_nan(gates["v20_combination_pass"])
            if a_fin and bool(gates["v20_combination_pass"]) != row["v20_combination_pass"]:
                an.append("recomputed v2.0 combination verdict differs from gates.json")
    # ---- D3 at the freeze, rule record, budget
    for t in (2, 1):
        st = stages.get(str(t))
        if not st:
            if must:
                an.append("rule_log.json has no stage %d" % t)
            continue
        row.update(d3_cols(st.get("freeze") or {}, t))
        row.update(rule_cols(st, t))
        row.update(timing_cols(summ, t))
    row.update(band_cols(ib or (_get(rep, "end_of_stage1", "decomposition") or {})))
    bt = gates.get("budget_totals") or {}
    row["total_updates"] = _f(bt.get("total_updates", stj.get("final_global_update")))
    row["total_episodes"] = _f(bt.get("total_episodes", summ.get("total_episodes")))
    row["total_minibatch_steps"] = _f(bt.get("minibatch_steps"))
    row["total_wall_sec"] = _f(stj.get("total_wall_sec"))
    # ---- start shares and clamp counts
    blocks = empty_b
    try:
        u = pd.read_csv(d / "ms_updates.csv")
        cols, blocks = update_cols(u, an)
        row.update(cols)
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        an.append("ms_updates.csv unreadable: %s: %s" % (type(exc).__name__, exc))
    # ---- flattened rule blocks of both stages
    rb: List[Dict[str, Any]] = []
    for t in (2, 1):
        for b in (_get(stages, str(t)) or {}).get("blocks") or []:
            rb.append({"arm": arm, "q": q, "seed": seed, "stage": t, "block_id": b.get("block_id"),
                       "block_type": b.get("type"), "first_local": b.get("first_local"),
                       "last_local": b.get("last_local"), "exit_reason": b.get("exit_reason"),
                       "classification": b.get("classification") or "", "n_S": b.get("n_S"),
                       "S": _join(b.get("S") or [], ","), "fire_local": b.get("fire_local")})
    finish_row(row, an, True)
    if not blocks.empty:
        blocks.insert(0, "seed", seed)
        blocks.insert(0, "q", q)
        blocks.insert(0, "arm", arm)
    return row, blocks, pd.DataFrame(rb)


def calibration_d3(cal_root: Path, q: int, seed: int) -> Dict[str, Any]:
    """D3 quantities of the v2.0 candidate from the calibration replay (``rehearsal_v2_0/q*/seed*.csv``).

    Terminal stage at u1600 on both tiers (``t2_*``), stage 1 at u2200 on the development tier (``t1_*_dev``).
    ``parents_A`` has the same candidate as ``rehearsal_v2_0`` at u1600 (D6), so its terminal-stage values come
    from the same file. Empty if the file is missing.
    """
    f = Path(cal_root) / REHEARSAL / ("q%d" % q) / ("seed%d.csv" % seed)
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    out: Dict[str, Any] = {}
    a = df[(df["update"] == 1600) & (df["stage"] == 2)]
    for tier, name in (("final", "final"), ("dev", "dev")):
        r = a[a["tier"] == tier]
        if len(r) == 1:
            r = r.iloc[0]
            for k, col in (("Delta", "Delta"), ("s", "s"), ("R", "R"), ("Rtail", "R_tail"), ("C", "C")):
                out["t2_%s_%s" % (k, name)] = _f(r[col])
    b = df[(df["update"] == 2200) & (df["stage"] == 1) & (df["tier"] == "dev")]
    if len(b) == 1:
        r = b.iloc[0]
        for k, col in (("Delta", "Delta"), ("s", "s"), ("R", "R"), ("C", "C")):
            out["t1_%s_dev" % k] = _f(r[col])
    return out


def extract_parents_run(d: Path, q: int, seed: int, th: Thresholds, source: str) -> Dict[str, Any]:
    """One ``parents_A`` run (v2.0 ``phase_A``; stage 1 untrained) as a per_run row of the pseudo-arm."""
    row = new_row(PARENTS, q, seed, "comparator", d, source)
    status, info, code, stj = run_status(d)
    row.update(status=status, status_info=info, exit_code=code)
    an: List[str] = []
    if status == "missing":
        finish_row(row, an, False)
        return row
    must = status == "done"
    fv = _jload(d / "final_v2.json", an, must)
    summ = _jload(d / "v2_run_summary.json", an, False)
    cfg = _jload(d / "run_config.json", an, False)
    git = stj.get("git") or {}
    row["git_commit"] = str(git.get("commit", ""))
    row["git_dirty"] = _bool_or_nan(git.get("dirty"))
    fin, dev = fv.get("final") or {}, fv.get("development") or {}
    if fin:
        row.update(terminal_cols(fin, dev, th))
    ph = _get(summ, "phase_timing", "A") or {}
    row["t2_updates"] = _f(ph.get("updates", stj.get("final_global_update")))
    row["t2_wall_sec"] = _f(ph.get("wall_sec"))
    row["t2_cpu_sec"] = _f(ph.get("process_cpu_sec"))
    row["t2_dev_verifier_sec"] = _f(_get(ph, "by_category_sec", "dev_verifier"))
    row["t2_episodes"] = _f(_get(summ, "costs", "total_episodes"))
    row["t2_minibatch_steps"] = _f(_get(summ, "costs", "minibatch_steps"))
    row["total_updates"] = row["t2_updates"]
    row["total_episodes"] = row["t2_episodes"]
    row["total_minibatch_steps"] = row["t2_minibatch_steps"]
    row["total_wall_sec"] = _f(stj.get("total_wall_sec"))
    finish_row(row, an, False)
    return row


def extract_rehearsal_run(d: Path, q: int, seed: int, th: Thresholds, source: str) -> Dict[str, Any]:
    """One ``rehearsal_v2_0`` run (locked v2.0, both stages) as a per_run row of the pseudo-arm."""
    row = new_row(REHEARSAL, q, seed, "comparator", d, source)
    status, info, code, stj = run_status(d)
    row.update(status=status, status_info=info, exit_code=code)
    an: List[str] = []
    if status == "missing":
        finish_row(row, an, True)
        return row
    must = status == "done"
    gates = _jload(d / "gates.json", an, must)
    ib = _jload(d / "induced_band.json", an, must)
    summ = _jload(d / "v2_run_summary.json", an, False)
    cfg = _jload(d / "run_config.json", an, False)
    git = stj.get("git") or {}
    row["git_commit"] = str(git.get("commit", ""))
    row["git_dirty"] = _bool_or_nan(git.get("dirty"))
    row["outcome"] = str(gates.get("outcome", ""))
    row["global_rng_ok"] = _bool_or_nan(None if not gates.get("global_rng") else
                                        gates["global_rng"].get("status") == "ok")
    rep = gates.get("reported") or {}
    a_fin, a_dev = _get(rep, "end_of_A", "final") or {}, _get(rep, "end_of_A", "development") or {}
    if a_fin:
        row.update(terminal_cols(a_fin, a_dev, th))
    sg = _get(rep, "end_of_A", "smoothed_game") or {}
    row["smoothed_share_peak_gap_d0"] = _f(sg.get("smoothed_share_peak_gap_d0"))
    row["smoothed_e_pred_0"] = _f(sg.get("smoothed_e_pred_0"))
    row["smoothed_e_learned_0"] = _f(sg.get("smoothed_e_learned_0"))
    s_fin, s_dev = _get(rep, "end_of_B", "final") or {}, _get(rep, "end_of_B", "development") or {}
    if s_fin:
        row.update(stage1_cols(s_fin, s_dev, th))
        if a_fin:
            row["v20_combination_pass"] = bool(row["G_A_pass"] and row["G_N_eta_pass"] and row["G_F_pass"]
                                               and row["G_N_gmax_pass"] and row["G_S_pass"])
        if "run_pass" in gates:
            row["v20_combination_pass_recorded"] = _bool_or_nan(gates["run_pass"])
    row.update(band_cols(ib or (_get(rep, "end_of_B", "decomposition") or {})))
    epu = _f(_get(cfg, "record", "protocol", "episodes_per_update"))
    for t, key in ((2, "A"), (1, "B")):
        ph = _get(summ, "phase_timing", key) or {}
        p = "t%d_" % t
        row[p + "updates"] = _f(ph.get("updates"))
        row[p + "wall_sec"] = _f(ph.get("wall_sec"))
        row[p + "cpu_sec"] = _f(ph.get("process_cpu_sec"))
        row[p + "dev_verifier_sec"] = _f(_get(ph, "by_category_sec", "dev_verifier"))
        row[p + "episodes"] = row[p + "updates"] * epu          # derived: updates x episodes_per_update
    row["total_updates"] = _f(stj.get("final_global_update"))
    row["total_episodes"] = row["t2_episodes"] + row["t1_episodes"]
    row["total_wall_sec"] = _f(stj.get("total_wall_sec"))
    finish_row(row, an, True)
    return row


def arm_dir(arm: str, q: int, seed: int, roots: Mapping[str, str]) -> Tuple[Path, str]:
    """Run directory of (arm, q, seed) and the name of the root it lives under."""
    if arm == BASE_ARM:
        return Path(roots["base"]) / ("q%d" % q) / ("seed%d" % seed) / arm, "base"
    if arm == PARENTS:
        return Path(roots["parents"]) / ("q%d" % q) / ("seed%d" % seed), "parents"
    if arm == REHEARSAL:
        return Path(roots["rehearsal"]) / ("q%d" % q) / ("seed%d" % seed), "rehearsal"
    return Path(roots["pilot"]) / ("q%d" % q) / ("seed%d" % seed) / arm, "pilot"


def extract_all(roots: Mapping[str, str], arms: Sequence[str], qs: Sequence[int], seeds: Sequence[int],
                th: Thresholds) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The per-run table (arms, then the comparators), the per-block start shares and the rule blocks."""
    rows: List[Dict[str, Any]] = []
    blocks: List[pd.DataFrame] = []
    rblocks: List[pd.DataFrame] = []
    for arm in list(arms) + list(COMPARATORS):
        for q in qs:
            for s in seeds:
                d, src = arm_dir(arm, q, s, roots)
                if arm in (PARENTS, REHEARSAL):
                    r = (extract_parents_run if arm == PARENTS else extract_rehearsal_run)(d, q, s, th, src)
                    if roots.get("calibration"):
                        cal = calibration_d3(Path(roots["calibration"]), q, s)
                        if arm == PARENTS:
                            cal = {k: v for k, v in cal.items() if k.startswith("t2_")}
                        r.update(cal)
                    rows.append(r)
                else:
                    r, b, rb = extract_ms_run(d, arm, q, s, th, src)
                    rows.append(r)
                    if not b.empty:
                        blocks.append(b)
                    if not rb.empty:
                        rblocks.append(rb)
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in COLUMNS:
        if c in STR_COLS or c in BOOL_COLS:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bcols = ["arm", "q", "seed", "stage", "block_id", "block_type", "n_updates", "n_tail", "n_near", "n_mid",
             "share_tail", "share_near", "share_mid"]
    rcols = ["arm", "q", "seed", "stage", "block_id", "block_type", "first_local", "last_local", "exit_reason",
             "classification", "n_S", "S", "fire_local"]
    bdf = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=bcols)
    rdf = pd.concat(rblocks, ignore_index=True) if rblocks else pd.DataFrame(columns=rcols)
    return df, bdf, rdf


# --------------------------------------------------------------------------------------------- statistics
def boot_ci(x: Sequence[float], stat: str = "mean", n_boot: int = N_BOOT, seed: int = BOOT_SEED
            ) -> Tuple[float, float]:
    """95% percentile bootstrap interval of the mean (or median) of ``x`` with a FRESH generator.

    A new ``numpy.random.default_rng(seed)`` is created at every call; the resampled indices are
    ``rng.integers(0, n, size=(n_boot, n))`` and the interval is the 2.5 / 97.5 percentiles of the resampled
    statistic.

    Args:
        x: Values (1-D).
        stat: ``"mean"`` or ``"median"``.
        n_boot: Number of resamples.
        seed: Generator seed (the pre-registered 20261006).

    Returns:
        ``(lo, hi)``; ``(nan, nan)`` for an empty input.
    """
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    r = a[idx]
    s = r.mean(axis=1) if stat == "mean" else np.median(r, axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def paired_summary(diff: Sequence[float], lower_better: Optional[bool]) -> Dict[str, Any]:
    """Mean / median, sign counts and the two bootstrap intervals of paired differences ``arm - baseline``.

    Args:
        diff: Paired differences (one per seed).
        lower_better: True if a decrease is an improvement; None for a metric without direction.
    """
    d = np.asarray(diff, dtype=float)
    lo_m, hi_m = boot_ci(d, "mean")
    lo_d, hi_d = boot_ci(d, "median")
    return {"n_pairs": int(d.size), "mean": float(d.mean()) if d.size else float("nan"),
            "median": float(np.median(d)) if d.size else float("nan"),
            "n_better": int((d < 0).sum()) if lower_better else None,
            "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "n_zero": int((d == 0).sum()),
            "ci_mean_lo": lo_m, "ci_mean_hi": hi_m, "ci_median_lo": lo_d, "ci_median_hi": hi_d,
            "ci_mean_contains_0": bool(d.size > 0 and lo_m <= 0.0 <= hi_m),
            "ci_mean_below_0": bool(d.size > 0 and hi_m < 0.0)}


def summary_stats(x: Sequence[float]) -> Dict[str, Any]:
    """Descriptive statistics of one sample with the bootstrap interval of its mean."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    lo, hi = boot_ci(a, "mean")
    return {"n": int(a.size), "mean": float(a.mean()) if a.size else float("nan"),
            "median": float(np.median(a)) if a.size else float("nan"),
            "sd": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
            "min": float(a.min()) if a.size else float("nan"), "max": float(a.max()) if a.size else float("nan"),
            "ci_mean_lo": lo, "ci_mean_hi": hi,
            "ci_mean_contains_0": bool(a.size > 0 and lo <= 0.0 <= hi)}


def paired_seed_values(df: pd.DataFrame, arm: str, base: str, q: int, metric: str, seeds: Sequence[int]
                       ) -> pd.DataFrame:
    """Seed-level arm / baseline values of the seeds where both runs are complete and the metric is finite."""
    cols = ["seed", "arm_value", "base_value", "diff"]
    if metric not in df.columns:
        return pd.DataFrame(columns=cols)
    a = df[(df["arm"] == arm) & (df["q"] == q) & df["complete"].astype(bool)].set_index("seed")[metric]
    b = df[(df["arm"] == base) & (df["q"] == q) & df["complete"].astype(bool)].set_index("seed")[metric]
    rows = []
    for s in seeds:
        if s in a.index and s in b.index:
            x, y = _f(a.loc[s]), _f(b.loc[s])
            if np.isfinite(x) and np.isfinite(y):
                rows.append({"seed": int(s), "arm_value": x, "base_value": y, "diff": x - y})
    return pd.DataFrame(rows, columns=cols)


def paired_tables(df: pd.DataFrame, comparisons: Sequence[Tuple[str, str, str]],
                  metrics: Sequence[Tuple[str, Optional[bool]]], qs: Sequence[int], seeds: Sequence[int],
                  label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Paired-difference table (arm - baseline) and its seed-level long form.

    Args:
        df: The per-run table.
        comparisons: ``(arm, baseline, kind)`` in table order.
        metrics: ``(column, direction)`` list.
        qs: q values.
        seeds: Development seeds (the pairing keys).
        label: Name of the comparison (``vs parents_A`` ...).

    Returns:
        ``(summary, seed_level)``. Metrics without any finite pair are omitted except the first one.
    """
    rows, long_rows = [], []
    for arm, base, _ in comparisons:
        for q in qs:
            for i, (m, direction) in enumerate(metrics):
                sv = paired_seed_values(df, arm, base, q, m, seeds)
                if sv.empty and i > 0:
                    continue
                crit = bool(label == "vs parents_A" and m == PRIMARY)
                rows.append({"comparison": label, "arm": arm, "baseline": base, "q": q, "metric": m,
                             "direction": "lower is better" if direction else "none",
                             "n_expected": len(seeds),
                             "status": CRITERION_NOTE if crit else "descriptive",
                             **paired_summary(sv["diff"].to_numpy(), direction)})
                for r in sv.itertuples(index=False):
                    long_rows.append({"comparison": label, "arm": arm, "baseline": base, "q": q, "metric": m,
                                      "seed": r.seed, "arm_value": r.arm_value, "base_value": r.base_value,
                                      "diff": r.diff})
    cols = ["comparison", "arm", "baseline", "q", "metric", "seed", "arm_value", "base_value", "diff"]
    long = pd.DataFrame(long_rows, columns=cols)
    long["status"] = np.where((long["comparison"] == "vs parents_A") & (long["metric"] == PRIMARY),
                              CRITERION_NOTE, "descriptive")
    return pd.DataFrame(rows), long


# --------------------------------------------------------------------------------------------- criterion
def gate_pairs(df: pd.DataFrame, arm: str, base: str, qs: Sequence[int], seeds: Sequence[int]) -> pd.DataFrame:
    """Per (q, seed): did the baseline run pass (G-A and the eta part of G-N), and the arm run's status / verdict."""
    rows = []
    for q in qs:
        for s in seeds:
            b = df[(df["arm"] == base) & (df["q"] == q) & (df["seed"] == s)]
            a = df[(df["arm"] == arm) & (df["q"] == q) & (df["seed"] == s)]
            bp = bool(len(b) and _truthy(b["complete"].iloc[0]) and _truthy(b["gate_pass"].iloc[0]))
            st = str(a["status"].iloc[0]) if len(a) else "missing"
            ap = float("nan")
            if len(a) and st == "done":
                ap = _truthy(a["gate_pass"].iloc[0])
            rows.append({"q": q, "seed": s, "base_pass": bp, "arm_status": st, "arm_pass": ap})
    return pd.DataFrame(rows)


def criterion_row(prim: pd.DataFrame, gates: pd.DataFrame, qs: Sequence[int], n_expected: int
                  ) -> Dict[str, Any]:
    """Both parts of the pre-registered criterion of one arm (descriptive; not a gate).

    (a) ``ci_mean_hi < 0`` of the primary paired difference at EVERY q (strict). (b) no baseline-passing run
    fails under the arm: an arm run that is ``done`` and fails, or ``failed``, is a violation; a ``missing`` /
    ``running`` / ``incomplete`` arm run is pending.
    """
    out: Dict[str, Any] = {}
    flags, complete = [], []
    for q in qs:
        r = prim[prim["q"] == q]
        n = int(r["n_pairs"].iloc[0]) if len(r) else 0
        hi = float(r["ci_mean_hi"].iloc[0]) if n else float("nan")
        out["n_pairs_q%d" % q] = n
        out["mean_q%d" % q] = float(r["mean"].iloc[0]) if n else float("nan")
        out["ci_mean_lo_q%d" % q] = float(r["ci_mean_lo"].iloc[0]) if n else float("nan")
        out["ci_mean_hi_q%d" % q] = hi
        flag = bool(n > 0 and hi < 0.0)
        out["a_q%d" % q] = flag
        flags.append(flag)
        complete.append(n == n_expected)
    out["a_met"] = bool(all(flags))
    out["a_complete"] = bool(all(complete))
    g = gates[gates["base_pass"].astype(bool)]
    arm_pass = g["arm_pass"].astype("object").map(_truthy)
    viol = g[((g["arm_status"] == "done") & (~arm_pass)) | (g["arm_status"] == "failed")]
    pend = g[g["arm_status"].isin(["missing", "incomplete", "running"])]
    out["n_base_pass"] = int(len(g))
    out["b_violations"] = " ".join("q%d/%d" % (int(r.q), int(r.seed)) for r in viol.itertuples())
    out["b_pending"] = " ".join("q%d/%d" % (int(r.q), int(r.seed)) for r in pend.itertuples())
    out["b_n_violations"], out["b_n_pending"] = int(len(viol)), int(len(pend))
    out["b_status"] = "violated" if len(viol) else ("incomplete" if len(pend) else "holds")
    if out["b_status"] == "violated":
        out["overall"] = "not met"
    elif out["a_complete"] and out["b_status"] == "holds":
        out["overall"] = "met" if out["a_met"] else "not met"
    else:
        out["overall"] = "incomplete"
    return out


def criterion_table(df: pd.DataFrame, paired: pd.DataFrame, arms: Sequence[str], qs: Sequence[int],
                    seeds: Sequence[int]) -> pd.DataFrame:
    """``criterion.csv``: one row per arm (parts (a) and (b)), arms in table order, q in the given order."""
    rows = []
    for arm in arms:
        prim = paired[(paired["arm"] == arm) & (paired["baseline"] == PARENTS) & (paired["metric"] == PRIMARY)
                      & (paired["comparison"] == "vs parents_A")]
        gp = gate_pairs(df, arm, PARENTS, qs, seeds)
        rows.append({"arm": arm, "baseline": PARENTS, "comparison": "vs parents_A", "primary_metric": PRIMARY,
                     **criterion_row(prim, gp, qs, len(seeds)), "boot_seed": BOOT_SEED, "n_boot": N_BOOT,
                     "note": CRITERION_NOTE})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- other tables
def _sel(df: pd.DataFrame, arm: str, q: Any = None, complete_only: bool = True) -> pd.DataFrame:
    """Rows of ``arm`` (at ``q`` unless ``q`` is None / 'both'), complete runs only by default."""
    m = df["arm"] == arm
    if q is not None and q != "both":
        m &= df["q"] == q
    if complete_only:
        m &= df["complete"].astype(bool)
    return df[m]


def _count(s: pd.Series) -> int:
    """Number of True values (NaN and empty count as False)."""
    return int(sum(_truthy(v) for v in s))


def completeness_table(df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int], seeds: Sequence[int]
                       ) -> pd.DataFrame:
    """Per (arm, q): planned / done / failed / running / incomplete / missing runs and the runs not done."""
    rows = []
    for arm in list(arms) + list(COMPARATORS):
        for q in qs:
            g = df[(df["arm"] == arm) & (df["q"] == q)]
            r: Dict[str, Any] = {"arm": arm, "q": q, "n_planned": len(seeds)}
            for st in STATUSES:
                r["n_" + st] = int((g["status"] == st).sum())
            nd = g[g["status"] != "done"].sort_values("seed")
            r["not_done_runs"] = "; ".join("seed%d: %s (%s)" % (int(x.seed), x.status, x.status_info)
                                           for x in nd.itertuples())
            r["n_flagged"] = int((g["flags"].astype(str) != "").sum())
            rows.append(r)
    return pd.DataFrame(rows)


def arm_summary_table(df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int]) -> pd.DataFrame:
    """Descriptive statistics (with the bootstrap interval of the mean and whether it contains 0)."""
    rows = []
    metrics = [m for m, _ in S2_METRICS] + [m for m, _ in S1_METRICS if m not in dict(S2_METRICS)]
    for arm in list(arms) + list(COMPARATORS):
        for q in qs:
            g = _sel(df, arm, q)
            for m in metrics:
                if m not in df.columns:
                    continue
                v = pd.to_numeric(g[m], errors="coerce").dropna()
                if v.empty and m not in (PRIMARY, SIGNED):
                    continue
                rows.append({"arm": arm, "q": q, "metric": m, "status": "descriptive",
                             **summary_stats(v.to_numpy())})
    return pd.DataFrame(rows)


BUDGET_METRICS = ["t2_updates", "t2_training_updates", "t2_episodes", "t2_minibatch_steps", "t2_wall_sec",
                  "t2_cpu_sec", "t1_updates", "t1_training_updates", "t1_episodes", "t1_minibatch_steps",
                  "t1_wall_sec", "t1_cpu_sec", "total_updates", "total_episodes", "total_minibatch_steps",
                  "total_wall_sec"]


def budget_table(df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int]) -> pd.DataFrame:
    """Budget to the freeze per arm and q (updates, episodes, minibatch steps, wall seconds): long format."""
    rows = []
    for arm in list(arms) + list(COMPARATORS):
        for q in qs:
            g = _sel(df, arm, q)
            for m in BUDGET_METRICS:
                v = pd.to_numeric(g[m], errors="coerce").dropna()
                if v.empty:
                    continue
                rows.append({"arm": arm, "q": q, "metric": m, "n": int(v.size), "mean": float(v.mean()),
                             "median": float(v.median()), "min": float(v.min()), "max": float(v.max()),
                             "status": "descriptive"})
    return pd.DataFrame(rows)


def rule_table(df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int]) -> pd.DataFrame:
    """Rule record per arm and q: fire updates, block counts, classifications, ``budget_forced`` counts."""
    rows = []
    for arm in arms:
        for q in qs:
            g = _sel(df, arm, q)
            r: Dict[str, Any] = {"arm": arm, "q": q, "n_runs": len(g), "status": "descriptive"}
            for t in (2, 1):
                p = "t%d_" % t
                fire = pd.to_numeric(g[p + "fire_local"], errors="coerce").dropna()
                r[p + "n_fire"] = int(fire.size)
                r[p + "n_budget_forced"] = _count(g[p + "budget_forced"])
                r[p + "n_would_fire"] = int(pd.to_numeric(g[p + "would_fire_local"], errors="coerce").notna().sum())
                r[p + "fire_local_min"] = float(fire.min()) if fire.size else float("nan")
                r[p + "fire_local_median"] = float(fire.median()) if fire.size else float("nan")
                r[p + "fire_local_max"] = float(fire.max()) if fire.size else float("nan")
                nb = pd.to_numeric(g[p + "n_blocks"], errors="coerce").dropna()
                r[p + "n_blocks_mean"] = float(nb.mean()) if nb.size else float("nan")
                r[p + "n_blocks_max"] = float(nb.max()) if nb.size else float("nan")
                r[p + "updates_mean"] = float(pd.to_numeric(g[p + "updates"], errors="coerce").mean())
                r[p + "n_checks_mean"] = float(pd.to_numeric(g[p + "n_checks"], errors="coerce").mean())
            r["t2_n_polish_blocks_total"] = float(pd.to_numeric(g["t2_n_polish_blocks"], errors="coerce").sum())
            r["t2_n_runs_with_polish"] = int((pd.to_numeric(g["t2_n_polish_blocks"], errors="coerce") > 0).sum())
            r["t2_n_localized_total"] = float(pd.to_numeric(g["t2_n_localized"], errors="coerce").sum())
            r["t2_n_broad_total"] = float(pd.to_numeric(g["t2_n_broad"], errors="coerce").sum())
            nS = [int(x) for s in g["t2_nS_at_classification"].astype(str) for x in s.split(",") if x != ""]
            r["t2_nS_at_classification_mean"] = float(np.mean(nS)) if nS else float("nan")
            r["t2_nS_at_classification_max"] = float(max(nS)) if nS else float("nan")
            r["t2_updates_in_polish_mean"] = float(pd.to_numeric(g["t2_n_updates_polish"], errors="coerce").mean())
            rows.append(r)
    return pd.DataFrame(rows)


def tail_table(df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int], th: Thresholds) -> pd.DataFrame:
    """Tail mean and max over e2*(0) with the G-A tail verdict, and the first-order tail residual R_t^tail."""
    rows = []
    for arm in list(arms) + list(COMPARATORS):
        for q in qs:
            g = _sel(df, arm, q)
            r: Dict[str, Any] = {"arm": arm, "q": q, "n_runs": len(g), "status": "descriptive",
                                 "G_A_tail_limit": th.tail}
            for col, name in (("stage2_tail_mean_over_g2_0", "tail_mean"), ("stage2_tail_max_over_g2_0", "tail_max"),
                              ("t2_Rtail_final", "Rtail_final"), ("t2_Rtail_dev", "Rtail_dev")):
                v = pd.to_numeric(g[col], errors="coerce").dropna()
                r[name + "_mean"] = float(v.mean()) if v.size else float("nan")
                r[name + "_median"] = float(v.median()) if v.size else float("nan")
                r[name + "_max"] = float(v.max()) if v.size else float("nan")
            r["n_tail_mean_over_limit"] = int((pd.to_numeric(g["stage2_tail_mean_over_g2_0"], errors="coerce")
                                               > th.tail).sum())
            r["n_G_A_tail_pass"] = _count(g["G_A_tail_pass"])
            rows.append(r)
    return pd.DataFrame(rows)


def decomposition_table(df: pd.DataFrame) -> pd.DataFrame:
    """Smoothed-game share of the d = 0 gap per run (``parents_A``: not stored, NaN; stated in ``note``)."""
    cols = ["arm", "q", "seed", "status", "complete", "smoothed_share_peak_gap_d0", "smoothed_e_pred_0",
            "smoothed_e_learned_0", "e2_at_0", "g2_at_0", "stage2_peak_rel_err_signed", "sigma_effort_at_0_t2"]
    out = df[cols].copy()
    out["note"] = ""
    out.loc[out["arm"] == PARENTS, "note"] = ("not stored in parents_A (final_v2.json carries no smoothed_game); "
                                              "the end-of-A value of rehearsal_v2_0 is the v2.0 reference")
    out["status_label"] = "descriptive"
    return out


def stage1_table(df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int], th: Thresholds) -> pd.DataFrame:
    """Stage-1 summary per arm and q: G-S error, residual r_1(0)/s_1, induced-target decomposition, G-F, S1."""
    rows = []
    for arm in list(arms) + [REHEARSAL]:
        for q in qs:
            g = _sel(df, arm, q)
            r: Dict[str, Any] = {"arm": arm, "q": q, "n_runs": len(g), "status": "descriptive"}
            for col in ("stage1_rel_err_signed", PRIMARY_S1, "t1_R_final", "t1_R_dev", "t1_Delta_final",
                        "learning_rel", "inherited_rel", "Gmax_full_over_dw"):
                v = pd.to_numeric(g[col], errors="coerce").dropna()
                s = summary_stats(v.to_numpy())
                r[col + "_mean"] = s["mean"]
                r[col + "_median"] = s["median"]
                r[col + "_ci_lo"] = s["ci_mean_lo"]
                r[col + "_ci_hi"] = s["ci_mean_hi"]
                r[col + "_ci_contains_0"] = s["ci_mean_contains_0"]
            for col in ("G_S_pass", "S1_pass", "G_F_pass", "G_N_gmax_pass", "v20_combination_pass",
                        "learning_contains_0", "inherited_contains_0"):
                r["n_" + col] = _count(g[col])
            r["n_stage1_abs_le_%g" % th.g_s] = int((pd.to_numeric(g[PRIMARY_S1], errors="coerce") <= th.g_s).sum())
            rows.append(r)
    return pd.DataFrame(rows)


def decision_inputs(df: pd.DataFrame, crit: pd.DataFrame, paired: pd.DataFrame, arms: Sequence[str],
                    qs: Sequence[int], seeds: Sequence[int]) -> pd.DataFrame:
    """The arms side by side for the PI. Columns ``crit_*`` are the pre-registered criterion; all others are
    descriptive. Rows: (arm, q) for each q and ``both`` (pooled over q); comparators as reference rows."""
    rows = []
    for arm in list(arms) + list(COMPARATORS):
        cr = crit[crit["arm"] == arm]
        cr = cr.iloc[0] if len(cr) else None
        for q in list(qs) + ["both"]:
            g = _sel(df, arm, q)
            n_plan = len(seeds) * (len(qs) if q == "both" else 1)
            r: Dict[str, Any] = {"arm": arm, "q": q, "role": "comparator" if arm in COMPARATORS else "ms_arm",
                                 "n_complete": len(g), "n_planned": n_plan}
            if cr is not None:
                if q == "both":
                    r.update(crit_a_met=bool(cr["a_met"]), crit_b_status=cr["b_status"],
                             crit_overall=cr["overall"], crit_b_violations=cr["b_violations"])
                else:
                    r.update(crit_a_mean=cr["mean_q%d" % q], crit_a_ci_lo=cr["ci_mean_lo_q%d" % q],
                             crit_a_ci_hi=cr["ci_mean_hi_q%d" % q], crit_a_met=bool(cr["a_q%d" % q]))
            pk = pd.to_numeric(g[PRIMARY], errors="coerce").dropna()
            r["n_abs_peak_le_0.05"] = int((pk <= TAIL_PEAK).sum())
            r["abs_peak_mean"] = float(pk.mean()) if pk.size else float("nan")
            r["abs_peak_median"] = float(pk.median()) if pk.size else float("nan")
            sg = pd.to_numeric(g[SIGNED], errors="coerce").dropna()
            s = summary_stats(sg.to_numpy())
            r.update(signed_peak_mean=s["mean"], signed_peak_ci_lo=s["ci_mean_lo"], signed_peak_ci_hi=s["ci_mean_hi"],
                     signed_peak_ci_contains_0=s["ci_mean_contains_0"], n_signed_peak_negative=int((sg < 0).sum()))
            if arm not in COMPARATORS and q != "both":
                pr = paired[(paired["comparison"] == "vs parents_A") & (paired["arm"] == arm)
                            & (paired["q"] == q) & (paired["metric"] == SIGNED)]
                if len(pr):
                    r.update(signed_peak_diff_vs_parents_mean=float(pr["mean"].iloc[0]),
                             signed_peak_diff_vs_parents_ci_lo=float(pr["ci_mean_lo"].iloc[0]),
                             signed_peak_diff_vs_parents_ci_hi=float(pr["ci_mean_hi"].iloc[0]),
                             signed_peak_diff_vs_parents_ci_contains_0=bool(pr["ci_mean_contains_0"].iloc[0]))
            for col, name in (("stage2_rmse_pos_over_g2_0", "rmse"), ("stage2_tail_mean_over_g2_0", "tail_mean"),
                              ("stage2_tail_max_over_g2_0", "tail_max"), ("eta_T_over_dw", "eta2"),
                              ("smoothed_share_peak_gap_d0", "smoothed_share"), ("t2_R_final", "R2_final"),
                              ("t2_Rtail_final", "Rtail2_final"), ("t2_updates", "t2_updates"),
                              ("t2_episodes", "t2_episodes"), ("t2_wall_sec", "t2_wall_sec"),
                              ("total_wall_sec", "total_wall_sec"), (PRIMARY_S1, "stage1_abs"),
                              ("t1_R_final", "R1_final")):
                v = pd.to_numeric(g[col], errors="coerce").dropna()
                r[name + "_mean"] = float(v.mean()) if v.size else float("nan")
            for col, name in (("G_A_pass", "n_G_A_pass"), ("G_A_tail_pass", "n_G_A_tail_pass"),
                              ("G_A_rmse_pass", "n_G_A_rmse_pass"), ("G_N_eta_pass", "n_G_N_eta_pass"),
                              ("gate_pass", "n_gate_pass"), ("G_S_pass", "n_G_S_pass"), ("S1_pass", "n_S1_pass"),
                              ("G_F_pass", "n_G_F_pass"), ("v20_combination_pass", "n_v20_combination_pass"),
                              ("t2_budget_forced", "n_t2_budget_forced")):
                r[name] = _count(g[col])
            r["n_t2_fire"] = int(pd.to_numeric(g["t2_fire_local"], errors="coerce").notna().sum())
            r["n_t2_polish_runs"] = int((pd.to_numeric(g["t2_n_polish_blocks"], errors="coerce") > 0).sum())
            r["status_label"] = ("criterion columns (crit_*) pre-registered; every other column descriptive"
                                 if arm not in COMPARATORS else "reference row (descriptive)")
            rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- figures
def _plt() -> Any:
    """matplotlib.pyplot with the Agg backend (deterministic PDF dates are not needed: PNG only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _footer(fig: Any, roots: Mapping[str, str]) -> None:
    """Provenance footer: the four roots and the bootstrap setting."""
    txt = ("roots: base=%s | pilot=%s | parents_A=%s | rehearsal_v2_0=%s | bootstrap %d resamples, "
           "default_rng(%d)" % (roots["base"], roots["pilot"], roots["parents"], roots["rehearsal"], N_BOOT,
                               BOOT_SEED))
    fig.text(0.005, 0.003, txt, fontsize=5, ha="left", va="bottom", wrap=True)


def fig_paired(path: Path, seed_df: pd.DataFrame, summ: pd.DataFrame, comparison: str, metric: str,
               arms: Sequence[str], qs: Sequence[int], title: str, roots: Mapping[str, str]) -> None:
    """Paired differences per arm (seed dots, mean and 95% bootstrap interval) at each q."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(qs), figsize=(5.4 * len(qs), 0.5 * len(arms) + 2.6), sharey=True,
                             squeeze=False)
    for ax, q in zip(axes[0], qs):
        for i, arm in enumerate(arms):
            sv = seed_df[(seed_df["comparison"] == comparison) & (seed_df["arm"] == arm) & (seed_df["q"] == q)
                         & (seed_df["metric"] == metric)]
            if len(sv):
                jit = np.linspace(-0.18, 0.18, len(sv))
                ax.scatter(sv["diff"], np.full(len(sv), float(i)) + jit, s=12, alpha=0.55, color="tab:blue")
            rw = summ[(summ["comparison"] == comparison) & (summ["arm"] == arm) & (summ["q"] == q)
                      & (summ["metric"] == metric)]
            if len(rw) and np.isfinite(rw["mean"].iloc[0]) and np.isfinite(rw["ci_mean_lo"].iloc[0]):
                m, lo, hi = float(rw["mean"].iloc[0]), float(rw["ci_mean_lo"].iloc[0]), float(rw["ci_mean_hi"].iloc[0])
                ax.errorbar([m], [i], xerr=[[m - lo], [hi - m]], fmt="D", color="k", capsize=3, ms=5, lw=1.4)
        ax.axvline(0.0, color="tab:red", lw=0.8, ls="--")
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels(list(arms), fontsize=8)
        ax.set_title("q = %d" % q, fontsize=9)
        ax.set_xlabel("paired difference (arm - baseline)", fontsize=8)
        ax.invert_yaxis()
        ax.grid(alpha=0.25)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    _footer(fig, roots)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


def fig_peak_vs_update(path: Path, df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int],
                       roots: Mapping[str, str]) -> int:
    """Signed peak error against the update for every run (checks of the terminal stage) with the development
    stop (fire), the start of the landing, the forced landing and the legacy would-fire marked.

    Returns:
        Number of runs plotted.
    """
    from matplotlib.lines import Line2D
    plt = _plt()
    fig, axes = plt.subplots(len(arms), len(qs), figsize=(5.6 * len(qs), 2.3 * len(arms)), sharex=False,
                             sharey=True, squeeze=False)
    cmap = plt.get_cmap("tab10")
    n_plot = 0
    for i, arm in enumerate(arms):
        for j, q in enumerate(qs):
            ax = axes[i][j]
            g = df[(df["arm"] == arm) & (df["q"] == q)].sort_values("seed")
            for k, r in enumerate(g.itertuples()):
                ck = _read_csv_safe(Path(r.run_dir) / "ms_checks_stage2.csv")
                if ck is None or ck.empty or "stage2_peak_rel_err_signed" not in ck.columns:
                    continue
                col = cmap(k % 10)
                ax.plot(ck["update"], ck["stage2_peak_rel_err_signed"], color=col, lw=0.8, alpha=0.8)
                n_plot += 1
                entry = float((ck["update"] - ck["local"]).iloc[0])

                def at(local: float) -> Optional[float]:
                    m = ck[ck["local"] == int(local)]
                    return float(m["stage2_peak_rel_err_signed"].iloc[0]) if len(m) else None
                if np.isfinite(r.t2_fire_local):
                    y = at(r.t2_fire_local)
                    if y is not None:
                        ax.plot([entry + r.t2_fire_local], [y], marker="v", color=col, mec="k", ms=6, ls="none")
                if np.isfinite(r.t2_landing_first_local):
                    x = entry + r.t2_landing_first_local
                    ax.axvline(x, color=col, lw=0.4, ls=":", alpha=0.7)
                if _truthy(r.t2_budget_forced) and np.isfinite(r.t2_landing_first_local):
                    x = entry + r.t2_landing_first_local - 1
                    k_near = int(np.argmin(np.abs(ck["update"].to_numpy() - x)))
                    ax.plot([float(ck["update"].iloc[k_near])], [float(ck["stage2_peak_rel_err_signed"].iloc[k_near])],
                            marker="X", color=col, mec="k", ms=6, ls="none")
                if np.isfinite(r.t2_would_fire_local):
                    y = at(r.t2_would_fire_local)
                    if y is not None:
                        ax.plot([entry + r.t2_would_fire_local], [y], marker="^", mfc="none", color=col, ms=6,
                                ls="none")
            ax.axhline(0.0, color="k", lw=0.6)
            for v in (-TAIL_PEAK, TAIL_PEAK):
                ax.axhline(v, color="gray", lw=0.5, ls="--")
            ax.set_title("%s, q = %d" % (arm, q), fontsize=8)
            ax.grid(alpha=0.2)
            if j == 0:
                ax.set_ylabel("signed peak error", fontsize=7)
            if i == len(arms) - 1:
                ax.set_xlabel("global update", fontsize=7)
    handles = [Line2D([], [], marker="v", color="gray", mec="k", ls="none", label="development stop (fire)"),
               Line2D([], [], color="gray", ls=":", label="landing start"),
               Line2D([], [], marker="X", color="gray", mec="k", ls="none", label="budget forced (cap)"),
               Line2D([], [], marker="^", mfc="none", color="gray", ls="none", label="would fire (legacy arm)")]
    fig.legend(handles=handles, loc="upper right", fontsize=6, ncol=4)
    fig.suptitle("Descriptive: signed peak error of the terminal stage at every development check, one line per "
                 "seed (colour = seed); dashed = +-0.05", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    _footer(fig, roots)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return n_plot


def bin_centres(n_bins: int, half: float) -> np.ndarray:
    """Centres of the equal-width ES bins of D_2 (``linspace(-half, half, n_bins + 1)``)."""
    e = np.linspace(-half, half, n_bins + 1)
    return 0.5 * (e[:-1] + e[1:])


def fig_residual_maps(path: Path, df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int], kind: str,
                      roots: Mapping[str, str]) -> int:
    """Per-bin first-order residual maps at the freeze, one line per seed.

    ``kind`` is ``final`` / ``development`` (``rule_log.json`` ``freeze.rho_bins_<kind>``) or ``ema`` (the last
    row of ``rho_bar`` in ``ms_binmaps_stage2.npz``).

    Returns:
        Number of runs plotted.
    """
    plt = _plt()
    fig, axes = plt.subplots(len(arms), len(qs), figsize=(5.6 * len(qs), 2.2 * len(arms)), sharey=True,
                             squeeze=False)
    cmap = plt.get_cmap("tab10")
    n_plot = 0
    for i, arm in enumerate(arms):
        for j, q in enumerate(qs):
            ax = axes[i][j]
            g = df[(df["arm"] == arm) & (df["q"] == q)].sort_values("seed")
            shaded = False
            for k, r in enumerate(g.itertuples()):
                d = Path(r.run_dir)
                rho: Optional[np.ndarray] = None
                half = float("nan")
                try:
                    cfg = json.load(open(d / "run_config.json"))
                    half = float(cfg["record"]["domain_half_stage2"])
                    if kind == "ema":
                        z = np.load(d / "ms_binmaps_stage2.npz")
                        rho = np.asarray(z["rho_bar"], dtype=float)[-1]
                    else:
                        rl = json.load(open(d / "rule_log.json"))
                        rho = np.array([np.nan if x is None else x for x in
                                        rl["stages"]["2"]["freeze"]["rho_bins_%s" % kind]], dtype=float)
                except Exception:  # noqa: BLE001
                    continue
                c = bin_centres(rho.size, half)
                if not shaded:
                    ax.axvspan(-half, -2 * q, color="0.9")
                    ax.axvspan(2 * q, half, color="0.9")
                    shaded = True
                ax.plot(c, rho, color=cmap(k % 10), lw=0.9, marker=".", ms=3, alpha=0.85)
                n_plot += 1
            ax.axhline(0.03, color="k", lw=0.5, ls="--")
            ax.set_title("%s, q = %d" % (arm, q), fontsize=8)
            ax.grid(alpha=0.2)
            if j == 0:
                ax.set_ylabel("rho_b" if kind != "ema" else "rho_bar_b", fontsize=7)
            if i == len(arms) - 1:
                ax.set_xlabel("bin centre d (grey: tail bins, void)", fontsize=7)
    what = {"final": "final-tier per-bin map at the freeze", "development": "development-tier per-bin map at the "
            "freeze", "ema": "EMA map rho_bar at the last check"}[kind]
    fig.suptitle("Descriptive: first-order residual %s (dashed = rho_t 0.03), one line per seed" % what, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    _footer(fig, roots)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return n_plot


def fig_start_shares(path: Path, blocks: pd.DataFrame, df: pd.DataFrame, arms: Sequence[str],
                     qs: Sequence[int], roots: Mapping[str, str]) -> None:
    """Measured start shares per stratum over the blocks of the terminal stage (mean over seeds, min-max band),
    with the design shares of the arm as dotted lines."""
    from matplotlib.ticker import MaxNLocator
    plt = _plt()
    fig, axes = plt.subplots(len(arms), len(qs), figsize=(5.6 * len(qs), 2.2 * len(arms)), sharey=True,
                             squeeze=False)
    colors = {"tail": "tab:blue", "near": "tab:red", "mid": "tab:green"}
    for i, arm in enumerate(arms):
        for j, q in enumerate(qs):
            ax = axes[i][j]
            b = blocks[(blocks["arm"] == arm) & (blocks["q"] == q)] if len(blocks) else blocks
            for name, c in colors.items():
                if len(b):
                    gg = b.groupby("block_id")["share_" + name]
                    ids = sorted(b["block_id"].unique())
                    ax.plot(ids, [gg.mean().loc[k] for k in ids], marker="o", ms=3, color=c, label=name)
                    ax.fill_between(ids, [gg.min().loc[k] for k in ids], [gg.max().loc[k] for k in ids], color=c,
                                    alpha=0.15)
                dv = df[(df["arm"] == arm) & (df["q"] == q)]["design_lambda_%s" % {"tail": "T", "near": "P",
                                                                                   "mid": "M"}[name]]
                dv = pd.to_numeric(dv, errors="coerce").dropna()
                if len(dv):
                    ax.axhline(float(dv.iloc[0]), color=c, lw=0.7, ls=":")
            ax.set_title("%s, q = %d" % (arm, q), fontsize=8)
            ax.set_ylim(0, 1)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.2)
            if i == 0 and j == 0 and len(b):
                ax.legend(fontsize=6, loc="center right")
            if j == 0:
                ax.set_ylabel("start share", fontsize=7)
            if i == len(arms) - 1:
                ax.set_xlabel("block id (the last block of a rule arm is the landing window)", fontsize=7)
    fig.suptitle("Descriptive: measured start shares per stratum over the blocks of the terminal stage (mean and "
                 "min-max over seeds; dotted = design share lambda_T / lambda_P / lambda_M)", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    _footer(fig, roots)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- summary text
def _g(x: Any) -> str:
    """Compact signed number."""
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else "%+.5f" % float(x)


def summary_text(roots: Mapping[str, str], crit: pd.DataFrame, comp: pd.DataFrame, dec: pd.DataFrame,
                 qs: Sequence[int], seeds: Sequence[int], command: Optional[Sequence[str]] = None) -> str:
    """The CLI summary (also ``summary.txt``): provenance, criterion, run status, descriptive inputs."""
    L: List[str] = []
    L.append("MS-R1 analysis (prompt section 3.2)")
    if command:
        L.append("command: " + " ".join(str(c) for c in command))
    L.append("independent check: python tools/ms/blind_criterion.py --analysis-dir <the --out directory>")
    L.append("roots: base=%s | pilot=%s | parents_A=%s | rehearsal_v2_0=%s" % (
        roots["base"], roots["pilot"], roots["parents"], roots["rehearsal"]))
    L.append("bootstrap: %d resamples, numpy.random.default_rng(%d), one fresh generator per (q, statistic)" %
             (N_BOOT, BOOT_SEED))
    L.append("")
    L.append("== PRE-REGISTERED CRITERION: %s ==" % CRITERION_NOTE)
    L.append("primary metric %s of the frozen terminal-stage candidate; (a) CI of the mean paired difference "
             "arm - parents_A below 0 at BOTH q; (b) no run passing G-A with its G-N eta part under parents_A "
             "fails it under the arm" % PRIMARY)
    any_ci0 = False
    for r in crit.itertuples():
        parts = []
        for q in qs:
            lo, hi = getattr(r, "ci_mean_lo_q%d" % q), getattr(r, "ci_mean_hi_q%d" % q)
            met = getattr(r, "a_q%d" % q)
            parts.append("a%d %s [%s,%s] n=%d/%d %s" % (q, _g(getattr(r, "mean_q%d" % q)), _g(lo), _g(hi),
                                                        getattr(r, "n_pairs_q%d" % q), len(seeds),
                                                        "met" if met else "not"))
            if np.isfinite(lo) and np.isfinite(hi) and lo <= 0.0 <= hi:
                any_ci0 = True
        L.append("%-9s %s | b: %s%s | overall: %s" % (
            r.arm, " | ".join(parts), r.b_status, (" (%s)" % r.b_violations) if r.b_violations else "", r.overall))
    if any_ci0:
        L.append("NOTE: " + NO_EFFECT_SENTENCE)
    L.append("")
    L.append("== RUN STATUS ==")
    bad = comp[(comp["n_done"] != comp["n_planned"])]
    if bad.empty:
        L.append("every planned run of every arm and comparator is done")
    for r in bad.itertuples():
        L.append("%s q=%d: planned %d, done %d, failed %d, running %d, incomplete %d, missing %d -> %s" % (
            r.arm, r.q, r.n_planned, r.n_done, r.n_failed, r.n_running, r.n_incomplete, r.n_missing,
            r.not_done_runs))
    L.append("")
    L.append("== DESCRIPTIVE (not pre-registered; per arm and q) ==")
    L.append("%-15s %-5s %3s %6s %10s %24s %7s %9s %9s %8s %7s %6s" % (
        "arm", "q", "n", "<=0.05", "signed", "signed CI (contains 0?)", "tailmean", "gate_pass", "t2_upd", "budg_f",
        "fire", "smooth"))
    any_ci0 = False
    for _, r in dec.iterrows():
        ci0 = _truthy(r["signed_peak_ci_contains_0"])
        any_ci0 = any_ci0 or ci0
        L.append("%-15s %-5s %3d %6d %10s [%s,%s] %-3s %8.4f %9d %9.0f %8d %7d %6.3f" % (
            r["arm"], r["q"], r["n_complete"], r["n_abs_peak_le_0.05"], _g(r["signed_peak_mean"]),
            _g(r["signed_peak_ci_lo"]), _g(r["signed_peak_ci_hi"]), "yes" if ci0 else "no",
            _f(r["tail_mean_mean"]), r["n_gate_pass"], _f(r["t2_updates_mean"]), r["n_t2_budget_forced"],
            r["n_t2_fire"], _f(r["smoothed_share_mean"])))
    if any_ci0:
        L.append("NOTE: " + NO_EFFECT_SENTENCE)
    L.append("(columns after the arm are descriptive; the paired tables, rule, budget, tail, decomposition and "
             "stage-1 CSVs are descriptive as well)")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------------- driver
def write_csv(df: pd.DataFrame, path: Path, rid: str) -> None:
    """Write a CSV with a leading ``roots_id`` column."""
    out = df.copy()
    if "roots_id" in out.columns:
        out = out.drop(columns=["roots_id"])
    out.insert(0, "roots_id", rid)
    out.to_csv(path, index=False)


def run_analysis(roots: Mapping[str, str], out_dir: Path, qs: Sequence[int] = QS, seeds: Sequence[int] = SEEDS,
                 arms: Sequence[str] = MS_ARMS, protocol: Path = DEFAULT_PROTOCOL, figures: bool = True,
                 argv: Optional[Sequence[str]] = None) -> Tuple[Dict[str, pd.DataFrame], bool]:
    """Extract, tabulate and write every output.

    Args:
        roots: ``{base, pilot, parents, rehearsal}`` run roots.
        out_dir: Output directory.
        qs: q values.
        seeds: Development seeds.
        arms: MS arms to analyse (a subset of :data:`MS_ARMS`).
        protocol: ``protocols/v2_T2_locked_v2_0.json`` (the gate thresholds).
        figures: Whether to draw the figures.
        argv: Command line recorded in ``analysis_info.json``.

    Returns:
        ``(tables, all_done)`` with ``all_done`` True when every planned run of every arm and comparator is done.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    th, proto_sha = load_thresholds(protocol)
    rid = roots_id(roots)
    df, blocks, rblocks = extract_all(roots, arms, qs, seeds, th)
    cmp_parents = [(a, PARENTS, "") for a in arms]
    cmp_rehearsal = [(a, REHEARSAL, "") for a in arms]
    cmp_rule = [(a, CONTROL_ARM, "") for a in arms if a not in (BASE_ARM, CONTROL_ARM)]
    p_par, s_par = paired_tables(df, cmp_parents, S2_METRICS, qs, seeds, "vs parents_A")
    p_reh, s_reh = paired_tables(df, cmp_rehearsal, S1_METRICS, qs, seeds, "vs rehearsal_v2_0")
    p_rule2, s_rule2 = paired_tables(df, cmp_rule, S2_METRICS, qs, seeds, "vs MS_rule")
    p_rule1, s_rule1 = paired_tables(df, cmp_rule, S1_METRICS, qs, seeds, "vs MS_rule")
    p_rule = pd.concat([p_rule2, p_rule1], ignore_index=True)
    seed_level = pd.concat([s_par, s_reh, s_rule2, s_rule1], ignore_index=True)
    crit = criterion_table(df, p_par, arms, qs, seeds)
    comp = completeness_table(df, arms, qs, seeds)
    for t_ in (blocks, rblocks):
        t_["status"] = "descriptive"
    tables: Dict[str, pd.DataFrame] = {
        "per_run": df, "paired_vs_parents_A": p_par, "paired_vs_rehearsal_v2_0": p_reh,
        "paired_vs_MS_rule": p_rule, "paired_seed_level": seed_level, "criterion": crit,
        "arm_summary": arm_summary_table(df, arms, qs), "budget": budget_table(df, arms, qs),
        "rule": rule_table(df, arms, qs), "rule_blocks": rblocks, "tail": tail_table(df, arms, qs, th),
        "decomposition": decomposition_table(df), "stage1": stage1_table(df, arms, qs, th),
        "start_shares_by_block": blocks, "completeness": comp}
    tables["decision_inputs"] = decision_inputs(df, crit, p_par, arms, qs, seeds)
    for name, t in tables.items():
        write_csv(t, out_dir / ("%s.csv" % name), rid)
    fig_status: Dict[str, str] = {}
    if figures:
        fdir = out_dir / "figures"
        fdir.mkdir(exist_ok=True)
        jobs = [
            ("paired_primary_abs_peak.png", lambda p: fig_paired(
                p, s_par, p_par, "vs parents_A", PRIMARY, arms, qs,
                "Criterion part (a) (pre-registered statistic; the figure is descriptive): paired difference of "
                "|peak error|, arm - parents_A; dots = seeds, diamond = mean, bar = 95% percentile bootstrap "
                "interval", roots)),
            ("paired_signed_peak.png", lambda p: fig_paired(
                p, s_par, p_par, "vs parents_A", SIGNED, arms, qs,
                "Descriptive: paired difference of the signed peak error, arm - parents_A (interval containing 0 "
                "does not show that a mechanism has no effect, 10 seeds)", roots)),
            ("paired_primary_vs_MS_rule.png", lambda p: fig_paired(
                p, seed_level, p_rule, "vs MS_rule", PRIMARY, [a for a in arms if a not in (BASE_ARM, CONTROL_ARM)],
                qs, "Descriptive: paired difference of |peak error|, arm - MS_rule (the sampler's effect net of "
                "the rule)", roots)),
            ("paired_stage1_vs_rehearsal.png", lambda p: fig_paired(
                p, s_reh, p_reh, "vs rehearsal_v2_0", PRIMARY_S1, arms, qs,
                "Descriptive: paired difference of the stage-1 |error| (G-S metric), arm - rehearsal_v2_0", roots)),
            ("peak_error_vs_update.png", lambda p: fig_peak_vs_update(p, df[df["arm"].isin(arms)], arms, qs, roots)),
            ("residual_map_freeze_final.png", lambda p: fig_residual_maps(
                p, df[df["arm"].isin(arms)], arms, qs, "final", roots)),
            ("residual_map_freeze_development.png", lambda p: fig_residual_maps(
                p, df[df["arm"].isin(arms)], arms, qs, "development", roots)),
            ("residual_map_ema_last_check.png", lambda p: fig_residual_maps(
                p, df[df["arm"].isin(arms)], arms, qs, "ema", roots)),
            ("start_shares_by_block.png", lambda p: fig_start_shares(p, blocks, df, arms, qs, roots))]
        for fname, fn in jobs:
            try:
                fn(fdir / fname)
                fig_status[fname] = "ok"
            except Exception as exc:  # noqa: BLE001
                fig_status[fname] = "FAILED: %s: %s" % (type(exc).__name__, exc)
                print("[warn] figure %s failed: %s: %s" % (fname, type(exc).__name__, exc), file=sys.stderr)
    all_done = bool((comp["n_done"] == comp["n_planned"]).all())
    info = {
        "tool": "tools/ms/r1_analysis.py", "roots": dict(roots), "roots_id": rid, "protocol": str(protocol),
        "protocol_sha256": proto_sha, "thresholds": th.__dict__, "boot_seed": BOOT_SEED, "n_boot": N_BOOT,
        "bootstrap_scheme": "fresh numpy.random.default_rng(seed) per (q, statistic); idx = rng.integers(0, n, "
                            "size=(n_boot, n)); 2.5 / 97.5 percentiles of the resampled mean (or median)",
        "qs": list(qs), "seeds": list(seeds), "arms": list(arms), "comparators": list(COMPARATORS),
        "criterion": {"primary_metric": PRIMARY, "a": "ci_mean_hi < 0 of the mean paired difference arm - "
                      "parents_A at BOTH q", "b": "no run passing G-A (final tier) with the eta part of G-N "
                      "(|eta_dev - eta_final| <= 0.001) under parents_A fails it under the arm",
                      "note": CRITERION_NOTE},
        "status_counts": {a: {s: int(((df["arm"] == a) & (df["status"] == s)).sum()) for s in STATUSES}
                          for a in list(arms) + list(COMPARATORS)},
        "all_done": all_done, "figures": fig_status, "command": list(argv) if argv is not None else None,
        "numpy": np.__version__, "pandas": pd.__version__,
        "not_stored_by_design": {
            PARENTS: ["smoothed_share_peak_gap_d0 (final_v2.json carries no smoothed_game; NaN)", "D3 columns "
                      "t2_R_*, t2_Rtail_*, t2_C_*, t2_s_* (no one-step best-response effort stored; NaN)",
                      "every stage-1 column (stage 1 untrained)", "rule record, start shares, clamp counts (NaN)"],
            REHEARSAL: ["D3 columns t*_R_*, t*_C_*, t*_s_* (NaN)", "rule record, start shares, clamp counts (NaN)",
                        "t*_minibatch_steps (not stored in v2_run_summary.json; NaN)",
                        "t*_episodes = updates x episodes_per_update (derived)"]}}
    with open(out_dir / "analysis_info.json", "w") as f:
        json.dump(info, f, indent=1, sort_keys=True)
    text = summary_text(roots, crit, comp, tables["decision_inputs"], qs, seeds, argv)
    (out_dir / "summary.txt").write_text(text)
    return tables, all_done


def _parse_seeds(vals: Sequence[str]) -> Tuple[int, ...]:
    """Seeds as a list of ints or a single ``a-b`` range."""
    if len(vals) == 1 and "-" in vals[0]:
        a, b = vals[0].split("-")
        return tuple(range(int(a), int(b) + 1))
    return tuple(int(v) for v in vals)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point (see the module docstring)."""
    p = argparse.ArgumentParser(description="MS-R1 analysis (section 3.2)")
    p.add_argument("--base-root", default=str(ROOT / "results" / "ms_r1" / "base"))
    p.add_argument("--pilot-root", default=str(ROOT / "results" / "ms_r1" / "pilot"))
    p.add_argument("--parents-root", required=True)
    p.add_argument("--rehearsal-root", required=True)
    p.add_argument("--calibration-root", default=None,
                   help="results/ms_r1/calibration: fills the D3 quantities of the two comparators from the replay")
    p.add_argument("--out", default=str(ROOT / "results" / "ms_r1" / "analysis"))
    p.add_argument("--qs", type=int, nargs="+", default=list(QS))
    p.add_argument("--seeds", nargs="+", default=["%d-%d" % (SEEDS[0], SEEDS[-1])])
    p.add_argument("--arms", nargs="+", default=list(MS_ARMS))
    p.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)
    bad = [x for x in a.arms if x not in MS_ARMS]
    if bad:
        print("[error] unknown arms %s; arms are %s" % (bad, list(MS_ARMS)), file=sys.stderr)
        return 2
    roots = {"base": str(Path(a.base_root).resolve()), "pilot": str(Path(a.pilot_root).resolve()),
             "parents": str(Path(a.parents_root).resolve()), "rehearsal": str(Path(a.rehearsal_root).resolve())}
    if a.calibration_root:
        roots["calibration"] = str(Path(a.calibration_root).resolve())
    for k in ("parents", "rehearsal"):
        if not Path(roots[k]).is_dir():
            print("[error] %s root %s is not a directory (a missing reference root is a stop-and-report)" %
                  (k, roots[k]), file=sys.stderr)
            return 2
    tables, all_done = run_analysis(roots, Path(a.out), tuple(a.qs), _parse_seeds(a.seeds), tuple(a.arms),
                                    Path(a.protocol), not a.no_figures, list(argv) if argv is not None
                                    else sys.argv)
    sys.stdout.write((Path(a.out) / "summary.txt").read_text())
    if not all_done:
        print("[incomplete] at least one planned run is missing / failed / running / incomplete "
              "(see completeness.csv)", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
