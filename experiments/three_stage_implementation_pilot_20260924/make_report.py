#!/usr/bin/env python3
"""Tables, figures, completeness checks and rates for one cohort of T=3 runs.

Runs are enumerated from the manifests (never from whatever final_eval.json
files happen to exist): a manifest run without a directory is ``pending``;
partial runs get a row. Per-run tables go to ``runs/<run_id>/tables/``;
cohort tables, figures, ``completeness.json`` and ``AUTO_SUMMARY.md`` go to
``reports/<cohort>[_q<q>]/``.

    .venv/bin/python -B experiments/three_stage_implementation_pilot_20260924/make_report.py \
        --cohort pilot --q 60 --check-completeness
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import E, now_iso, read_jsonl, write_json_atomic  # noqa: E402
from metrics import (STAGE_VARS, EPISODE_VARS, center_bin_mask, certification_flags, count_stats,  # noqa: E402
                     diagnose_minimum_C, strict_gap_bins, summarize_moments, tail_masks, wilson)

from envs.curriculum_env import GameSpec  # noqa: E402

KEYS = ("run_id", "cohort", "q", "seed", "checkpoint_kind", "checkpoint_global_update")
STARTED_STATES = ("running", "done", "failed", "interrupted")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _reject_constant(name: str) -> None:
    raise ValueError(f"non-strict JSON constant {name}")


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Strict JSON (NaN/Infinity rejected); None if the file is missing."""
    if not path.exists():
        return None
    return json.loads(path.read_text(), parse_constant=_reject_constant)


def load_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """All arrays of an npz (None if missing)."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def fmt(v: Any) -> Any:
    """CSV cell: full-precision floats, empty for None/non-finite."""
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return repr(f) if math.isfinite(f) else ""
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v)
    return v


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows (union of keys, first-seen order)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in cols})


def over(v: Any, dw: float) -> Optional[float]:
    return None if v is None else float(v) / dw


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------

def load_run(manifest: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    """Everything persisted for one manifest run (missing pieces are None)."""
    d = E / run["output_dir"]
    R: Dict[str, Any] = {"run": run, "dir": d, "manifest": manifest, "cfg": manifest["resolved"],
                         "exists": d.exists(), "load_errors": []}
    if not d.exists():
        R["state"] = "pending"
        return R
    for name in ("status", "config", "training_summary", "final_eval", "economics"):
        try:
            R[name] = load_json(d / f"{name}.json")
        except Exception as exc:
            R[name] = None
            R["load_errors"].append(f"{name}.json: {exc}")
    for name in ("endpoint_record", "min_dev_record", "min_dev_profiles"):
        try:
            R[name] = load_json(d / "checkpoints" / f"{name}.json")
        except Exception as exc:
            R[name] = None
            R["load_errors"].append(f"{name}.json: {exc}")
    for name in ("history", "verifier_calls", "events", "resources"):
        try:
            R[name], R[name + "_info"] = read_jsonl(d / f"{name}.jsonl")
        except Exception as exc:
            R[name], R[name + "_info"] = [], {"error": str(exc)}
            R["load_errors"].append(f"{name}.jsonl: {exc}")
    R["coverage"] = load_npz(d / "coverage.npz")
    R["arrays"] = load_npz(d / "arrays.npz")
    R["econ_arrays"] = load_npz(d / "economics_arrays.npz")
    R["min_arrays"] = load_npz(d / "checkpoints" / "min_dev_arrays.npz")
    R["min_prof"] = load_npz(d / "checkpoints" / "min_dev_profiles.npz")
    st = R.get("status") or {}
    R["state"] = st.get("state", "started_without_status")
    return R


def persisted_has_candidate(R: Dict[str, Any]) -> Optional[bool]:
    """Candidate discovery from any persisted source (status, endpoint record, events)."""
    if not R["exists"]:
        return None
    st = R.get("status") or {}
    if st.get("has_candidate") is not None:
        return bool(st["has_candidate"])
    er = R.get("endpoint_record") or {}
    if er.get("identity", {}).get("checkpoint_kind") == "candidate":
        return True
    for ev in R.get("events", []):
        if ev.get("event") == "checkpoint_saved" and ev.get("kind") == "candidate":
            return True
    return False


def run_keys(R: Dict[str, Any], kind: Optional[str] = None, g: Optional[int] = None) -> Dict[str, Any]:
    r = R["run"]
    return {"run_id": r["run_id"], "cohort": r["cohort"], "q": r["q"], "seed": r["seed"],
            "checkpoint_kind": kind, "checkpoint_global_update": g}


def endpoint_keys(R: Dict[str, Any]) -> Dict[str, Any]:
    ident = ((R.get("endpoint_record") or {}).get("identity") or {})
    return run_keys(R, ident.get("checkpoint_kind"), ident.get("global_update"))


def min_keys(R: Dict[str, Any]) -> Dict[str, Any]:
    ident = ((R.get("min_dev_record") or {}).get("identity") or {})
    return run_keys(R, "minimum_development", ident.get("global_update"))


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def phase_rows(R: Dict[str, Any]) -> List[Dict[str, Any]]:
    """phases.csv rows (not-reached phases included)."""
    out = []
    calls = R.get("verifier_calls", [])
    done = {p["phase"]: p for p in ((R.get("status") or {}).get("phases") or [])}
    for ph in R["cfg"]["phases"]:
        name = ph["name"]
        row = {**run_keys(R), "phase": name, "cap": ph["cap"], "check_every": ph["check_every"],
               "k_consecutive": ph["k_consecutive"]}
        pc = [c for c in calls if c["phase"] == name]
        hist = [h for h in R.get("history", []) if h["phase"] == name]
        p = done.get(name)
        if p is None and not hist:
            row.update(exit_reason="not_reached", local_updates=0)
            out.append(row)
            continue
        if p is None:
            row["exit_reason"] = "unavailable_due_to_interruption"
            row["local_updates"] = len(hist)
        else:
            row.update(exit_reason=p["exit_reason"], local_updates=p["local_updates"],
                       entry_global_update=p["entry_global_update"], exit_global_update=p["exit_global_update"])
        elig = [c for c in pc if c.get("eligible")]
        longest = cur = 0
        third = None
        for c in pc:
            cur = cur + 1 if c.get("eligible") else 0
            longest = max(longest, cur)
            if cur == 3 and third is None:
                third = c["local_update"]
        types: Dict[str, int] = {}
        for c in pc:
            types[c.get("failure_type")] = types.get(c.get("failure_type"), 0) + 1
        crit = [c.get("criterion_value_over_dw") for c in pc if c.get("criterion_value_over_dw") is not None]
        conc = [c["concentration"].get("max_std_norm") for c in pc
                if c.get("concentration") and c["concentration"].get("max_std_norm") is not None]
        thr = ph["threshold_over_dw"]
        row.update({
            "n_checks": len(pc), "eligible_count": len(elig),
            "first_eligible_local": elig[0]["local_update"] if elig else None,
            "longest_consecutive_eligible": longest if name != "A" else None,
            "third_consecutive_eligible_local": third if name == "B" else None,
            "n_invalid": types.get("invalid", 0), "n_strategic_only_fail": types.get("strategic_only", 0),
            "n_conc_only_fail": types.get("conc_only", 0), "n_both_fail": types.get("both", 0),
            "criterion_min_over_dw": min(crit) if crit else None,
            "criterion_last_over_dw": crit[-1] if crit else None,
            "criterion_n_below_threshold": (sum(1 for v in crit if v <= thr) if thr is not None else None),
            "criterion_threshold_crossings": (sum(1 for a, b in zip(crit, crit[1:])
                                                  if (a <= thr) != (b <= thr)) if thr is not None else None),
            "conc_min": min(conc) if conc else None, "conc_last": conc[-1] if conc else None,
            "conc_n_below_threshold": sum(1 for v in conc if v <= 0.04),
            "episodes": sum(h["n_episodes"] for h in hist),
            "environment_steps": sum(h["n_environment_steps"] for h in hist),
            "physical_actions": sum(h["n_physical_actions"] for h in hist),
        })
        out.append(row)
    return out


def call_rows(R: Dict[str, Any]) -> List[Dict[str, Any]]:
    """verifier_calls.csv: one row per development call."""
    dw = R["cfg"]["derived"]["dw"]
    out = []
    for c in R.get("verifier_calls", []):
        s = c.get("summary") or {}
        conc = c.get("concentration") or {}
        cg = c.get("continuation_gain_by_stage") or {}
        row = {**run_keys(R), "phase": c["phase"], "local_update": c["local_update"],
               "global_update": c["global_update"], "call_index": c["call_index"], "tier": "development",
               "valid": c["valid"], "error": c.get("error"), "criterion": c["criterion"],
               "criterion_value_raw": c.get("criterion_value_raw"),
               "criterion_value_over_dw": c.get("criterion_value_over_dw"),
               "threshold_over_dw": c.get("threshold_over_dw"), "strategic_pass": c.get("strategic_pass"),
               "conc_max_std_norm": conc.get("max_std_norm"), "conc_stage": conc.get("stage"),
               "conc_d": conc.get("d"), "conc_valid": c.get("conc_valid"), "conc_pass": c.get("conc_pass"),
               "eligible": c.get("eligible"), "consecutive_eligible": c.get("consecutive_eligible"),
               "failure_type": c.get("failure_type"),
               "exp_root_over_dw": s.get("exp_root_over_dw"), "dreach_over_dw": s.get("dreach_over_dw"),
               "delta_max_all_over_dw": s.get("delta_max_all_over_dw"), "dfull_over_dw": s.get("dfull_over_dw"),
               "pdl_residual_over_dw": s.get("pdl_residual_over_dw"), "pmf_mass_err_max": s.get("pmf_mass_err_max"),
               "invalid_reasons": c.get("invalid_reasons"),
               "wall_sec": (c.get("cost") or {}).get("wall_sec"),
               "cpu_sec": (c.get("cost") or {}).get("cpu_sec")}
        for t in ("1", "2", "3"):
            row[f"reach_contribution_t{t}_over_dw"] = over((s.get("reach_delta_max") or {}).get(t), dw)
            row[f"reach_argmax_d_t{t}"] = (s.get("reach_argmax_d") or {}).get(t)
            row[f"full_contribution_t{t}_over_dw"] = over((s.get("full_delta_max") or {}).get(t), dw)
            row[f"continuation_gain_max_t{t}_over_dw"] = (cg.get(t) or {}).get("max_over_dw")
            row[f"continuation_gain_argmax_d_t{t}"] = (cg.get(t) or {}).get("argmax_d")
        out.append(row)
    return out


def tier_rows(R: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """verifier_summary.csv and verifier_stage_metrics.csv rows."""
    dw = R["cfg"]["derived"]["dw"]
    items: List[Tuple[Dict[str, Any], str, Dict[str, Any]]] = []
    fe = R.get("final_eval") or {}
    for tier in ("development", "final"):
        if fe.get("tiers", {}).get(tier) is not None:
            items.append((endpoint_keys(R), tier, fe["tiers"][tier]))
    er = R.get("endpoint_record") or {}
    if er.get("development_call") and er["development_call"].get("summary"):
        c = er["development_call"]
        items.append((endpoint_keys(R), "development_call_at_endpoint",
                      {**c["summary"], "continuation_gain_by_stage": c.get("continuation_gain_by_stage")}))
    mr = R.get("min_dev_record")
    if mr and mr.get("summary"):
        items.append((min_keys(R), "minimum_development",
                      {**mr["summary"], "continuation_gain_by_stage": mr.get("continuation_gain_by_stage")}))
    rows, srows = [], []
    for keys, tier, s in items:
        row = {**keys, "tier": tier, "valid": s.get("valid"), "invalid_reasons": s.get("invalid_reasons"),
               "error": s.get("error"), "exp_root": s.get("exp_root"), "exp_root_over_dw": s.get("exp_root_over_dw"),
               "dreach": s.get("dreach"), "dreach_over_dw": s.get("dreach_over_dw"),
               "delta_max_all": s.get("delta_max_all"), "delta_max_all_over_dw": s.get("delta_max_all_over_dw"),
               "dfull": s.get("dfull"), "dfull_over_dw": s.get("dfull_over_dw"),
               "dreach_pmf_support_over_dw": s.get("dreach_pmf_support_over_dw"),
               "v_br_root": s.get("v_br_root"), "v_mean_root": s.get("v_mean_root"),
               "pdl_residual_over_dw": s.get("pdl_residual_over_dw"), "pmf_mass_err_max": s.get("pmf_mass_err_max"),
               "gl_weight_sum": s.get("gl_weight_sum"), "effort_grid_points": s.get("effort_grid_points")}
        for t in ("1", "2", "3"):
            row[f"grid_points_t{t}"] = (s.get("grid_points") or {}).get(t)
        rows.append(row)
        cg = s.get("continuation_gain_by_stage") or {}
        for t in ("1", "2", "3"):
            rd = (s.get("reach_delta_max") or {}).get(t)
            fd = (s.get("full_delta_max") or {}).get(t)
            srows.append({**keys, "tier": tier, "stage": int(t),
                          "reach_contribution_raw": rd, "reach_contribution_over_dw": over(rd, dw),
                          "reach_argmax_d": (s.get("reach_argmax_d") or {}).get(t),
                          "reach_count": (s.get("reach_count") or {}).get(t),
                          "full_contribution_raw": fd, "full_contribution_over_dw": over(fd, dw),
                          "full_argmax_d": (s.get("full_argmax_d") or {}).get(t),
                          "pmf_support_contribution_raw": (s.get("pmf_support_delta_max") or {}).get(t),
                          "pmf_mass": (s.get("pmf_mass") or {}).get(t),
                          "continuation_gain_max_raw": (cg.get(t) or {}).get("max_raw"),
                          "continuation_gain_max_over_dw": (cg.get(t) or {}).get("max_over_dw"),
                          "continuation_gain_argmax_d": (cg.get(t) or {}).get("argmax_d"),
                          "grid_points": (s.get("grid_points") or {}).get(t)})
    return rows, srows


def deviation_rows(R: Dict[str, Any]) -> List[Dict[str, Any]]:
    """deviations.csv from arrays.npz (development/final) and min_dev_arrays.npz."""
    dw = R["cfg"]["derived"]["dw"]
    out = []
    srcs = []
    if R.get("arrays") is not None:
        srcs += [(endpoint_keys(R), "development", R["arrays"]), (endpoint_keys(R), "final", R["arrays"])]
    if R.get("min_arrays") is not None:
        srcs.append((min_keys(R), "minimum_development", R["min_arrays"]))
    for keys, tier, A in srcs:
        for t in (1, 2, 3):
            p = f"{tier}_t{t}_"
            if p + "d_grid" not in A:
                continue
            g = A[p + "v_br"] - A[p + "v_mean"]
            for i, d in enumerate(A[p + "d_grid"]):
                out.append({**keys, "tier": tier, "stage": t, "d": float(d),
                            "delta_raw": float(A[p + "delta"][i]), "delta_over_dw": float(A[p + "delta"][i]) / dw,
                            "continuation_gain_raw": float(g[i]), "continuation_gain_over_dw": float(g[i]) / dw,
                            "a_dev": float(A[p + "a_dev"][i]), "a_br": float(A[p + "a_br"][i]),
                            "e_hat": float(A[p + "e_hat"][i]), "e_opp": float(A[p + "e_opp"][i]),
                            "reach_mask": bool(A[p + "reach"][i]), "br_pmf": float(A[p + "pmf"][i]),
                            "v_br": float(A[p + "v_br"][i]), "v_mean": float(A[p + "v_mean"][i]),
                            "std_norm": float(A[p + "std_norm"][i]) if p + "std_norm" in A else None})
    return out


def profile_rows(R: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """policy_profiles.csv and the curve part of policy_asymmetry.csv."""
    srcs = []
    if R.get("arrays") is not None:
        srcs.append((endpoint_keys(R), R["arrays"]))
    if R.get("min_prof") is not None:
        srcs.append((min_keys(R), R["min_prof"]))
    prof, asym = [], []
    for keys, A in srcs:
        for t in (1, 2, 3):
            p = f"dense_t{t}_"
            if p + "d" not in A:
                continue
            cols = ("d", "d_normalized", "alpha", "beta", "mean_effort", "effort_variance", "std_effort",
                    "std_norm", "opponent_mean_effort")
            arrs = [A[p + c] for c in cols]
            for vals in zip(*arrs):
                prof.append({**keys, "stage": t, **{c: float(v) for c, v in zip(cols, vals)}})
            if t >= 2:
                for x, lv, fv, dv in zip(A[p + "asym_x"], A[p + "leader_mean_effort"],
                                         A[p + "follower_mean_effort"], A[p + "lead_follow_policy_difference"]):
                    asym.append({**keys, "kind": "curve", "stage": t, "x": float(x),
                                 "leader_mean_effort": float(lv), "follower_mean_effort": float(fv),
                                 "lead_follow_policy_difference": float(dv)})
    return prof, asym


VAR_LABEL = {
    "eff_p0": ("effort", "player0"), "eff_p1": ("effort", "player1"),
    "cost_p0": ("cost", "player0"), "cost_p1": ("cost", "player1"),
    "X": ("effort", "representative"), "Y": ("effort_sq", "representative"),
    "leader": ("effort", "leader"), "follower": ("effort", "follower"),
    "lf_diff": ("effort_difference", "leader_minus_follower_paired"),
    "phys_diff": ("effort_difference", "player0_minus_player1"),
    "gap_p0": ("pre_action_gap", "player0"),
    "theo_mean_p0": ("conditional_theoretical_mean_effort", "player0"),
    "theo_mean_p1": ("conditional_theoretical_mean_effort", "player1"),
    "theo_m2_p0": ("conditional_theoretical_E_e2", "player0"),
    "theo_m2_p1": ("conditional_theoretical_E_e2", "player1"),
}


def econ_rows(R: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """stage_metrics.csv, occupancy asymmetry rows and self-play state visitation, from npz moments."""
    A = R.get("econ_arrays")
    ec = R.get("economics") or {}
    if A is None or not ec.get("modes"):
        return [], [], []
    k = float(R["cfg"]["game"]["k"])
    keys = endpoint_keys(R)
    reps = int(ec["config"]["replicates"])
    rows, asym, vis = [], [], []
    for mode_id, mode in ((0, "mean"), (1, "stochastic")):
        for rep in list(range(reps)) + ["pooled"]:
            pre = f"m{mode_id}_{'pooled' if rep == 'pooled' else 'r' + str(rep)}"
            for v in STAGE_VARS:
                arr = A[f"{pre}_stage_{v}"]
                for t in range(1, arr.shape[0] + 1):
                    m = summarize_moments(*arr[t - 1])
                    var, persp = VAR_LABEL[v]
                    row = {**keys, "mode": mode, "replicate": rep, "stage": t, "variable": var,
                           "perspective": persp, "source_key": v, "n": m["n"], "sum": m["sum"],
                           "sum_sq": m["sum_sq"], "mean": m["mean"], "variance": m["variance"], "mcse": m["mcse"],
                           "unavailable_reason": m.get("mean_reason") or m.get("variance_reason")}
                    if v in ("eff_p0", "eff_p1"):
                        row["E_e2"] = m["sum_sq"] / m["n"] if m["n"] else None
                        row["E_cost"] = k * row["E_e2"] if row["E_e2"] is not None else None
                    if v == "X":
                        y = summarize_moments(*A[f"{pre}_stage_Y"][t - 1])
                        row["E_e2"] = y["mean"]
                        row["E_cost"] = k * y["mean"] if y["mean"] is not None else None
                    rows.append(row)
                    if v in ("leader", "follower", "lf_diff") and rep == "pooled":
                        asym.append({**keys, "kind": "occupancy", "mode": mode, "stage": t,
                                     "quantity": v, "n": m["n"], "mean": m["mean"], "mcse": m["mcse"],
                                     "ties_excluded": int(A[f"{pre}_lf_ties"][t - 1]),
                                     "unavailable_reason": m.get("mean_reason")})
            for v in EPISODE_VARS:
                m = summarize_moments(*A[f"{pre}_episode_{v}"])
                rows.append({**keys, "mode": mode, "replicate": rep, "stage": "episode", "variable": v,
                             "perspective": {"payoff0": "player0", "payoff1": "player1", "U": "representative",
                                             "prize0": "player0", "prize1": "player1"}[v],
                             "source_key": v, "n": m["n"], "sum": m["sum"], "sum_sq": m["sum_sq"],
                             "mean": m["mean"], "variance": m["variance"], "mcse": m["mcse"]})
            if rep != "pooled":
                continue
            for t in (1, 2, 3):
                edges = A[f"edges_t{t}"]
                c0 = A[f"{pre}_bins_p0_count_t{t}"]
                c1 = A[f"{pre}_bins_p1_count_t{t}"]
                crep = 0.5 * (c0 + c1)
                for persp, cnt in (("player0", c0), ("player1", c1), ("representative", crep)):
                    tot = float(cnt.sum())
                    for b in range(cnt.size):
                        row = {**keys, "origin": f"{mode}_selfplay", "phase": None, "start_stage": None,
                               "stage": t, "perspective": persp, "grid_kind": "width10_bins",
                               "bin_index": b, "bin_left": float(edges[b]), "bin_right": float(edges[b + 1]),
                               "count": float(cnt[b]), "probability_mass": float(cnt[b]) / tot if tot else None}
                        if persp != "representative":
                            p = persp[-1]
                            es = A[f"{pre}_bins_p{p}_eff_sum_t{t}"][b]
                            row.update(effort_sum=float(es),
                                       effort_sq_sum=float(A[f"{pre}_bins_p{p}_eff_sq_sum_t{t}"][b]),
                                       cost_sum=float(A[f"{pre}_bins_p{p}_cost_sum_t{t}"][b]),
                                       bin_mean_effort=float(es) / float(cnt[b]) if cnt[b] else None)
                        vis.append(row)
    return rows, asym, vis


def coverage_tables(R: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """coverage.csv (bins), coverage_summary.csv and training state_visitation rows."""
    C = R.get("coverage")
    if C is None:
        return [], [], []
    spec = GameSpec(**R["cfg"]["game"])
    frac = R["cfg"]["coverage"]["tail_fraction_of_domain_half"]
    layout = json.loads(str(C["layout_json"]))
    phases = [str(p) for p in C["phases"]]
    by_phase = {p: ph for p in phases for ph in R["cfg"]["phases"] if ph["name"] == p}
    keys = run_keys(R)
    bins_rows, summ_rows, vis = [], [], []
    for pi, p in enumerate(phases):
        upd = int(C["phase_updates"][pi])
        if upd == 0:
            continue
        sc = by_phase[p]["start_counts"]
        center_es = by_phase[p].get("center_es") or {}
        tot = C["phase_totals"][pi]
        for seg in layout:
            s, t, kind = seg["start_stage"], seg["current_stage"], seg["kind"]
            counts = tot[seg["offset"]:seg["offset"] + seg["nbins"]]
            n_start = int(sc.get(str(s), 0)) * upd
            if n_start == 0:
                continue
            edges = C[f"edges_t{t}"]
            half = spec.domain_half(t)
            direct = kind != "visits" or s == t
            exp_bins = None
            nc = int(center_es.get(str(s), 0)) * upd
            if t == 1:
                expected, reason = float(n_start), None
            elif direct and nc:
                cmask = center_bin_mask(spec, t, float(C["bin_width"]), spec.B)
                exp_bins = (n_start - nc) / seg["nbins"] + np.where(cmask, nc / cmask.sum(), 0.0)
                expected, reason = None, "non-uniform by design (center mixture); per-bin values in coverage.csv"
            elif direct:
                expected, reason = n_start / seg["nbins"], None
            else:
                expected, reason = None, "continuation visits are not uniform by design"
            pos, neg = tail_masks(edges, half, frac)
            label = {"visits": "learner_signed_pre_action", "start_raw": "player0_start_before_role_flip",
                     "start_learner": "learner_signed_start"}[kind]
            st = count_stats(counts)
            summ_rows.append({**keys, "phase": p, "phase_updates": upd, "kind": kind, "perspective": label,
                              "start_stage": s, "current_stage": t,
                              "source": "direct_start" if direct else "continuation", **st,
                              "expected_per_bin": expected, "expected_reason": reason,
                              "expected_center_bin": float(exp_bins.max()) if exp_bins is not None else None,
                              "expected_outer_bin": float(exp_bins.min()) if exp_bins is not None else None,
                              "center_bins_count": (int(counts[center_bin_mask(spec, t, float(C["bin_width"]), spec.B)].sum())
                                                    if t == 3 else None),
                              "tail_pos_count": int(counts[pos].sum()), "tail_neg_count": int(counts[neg].sum()),
                              "tail_pos_bins": int(pos.sum()), "tail_neg_bins": int(neg.sum()),
                              "tail_pos_min_bin": int(counts[pos].min()) if pos.any() else None,
                              "tail_neg_min_bin": int(counts[neg].min()) if neg.any() else None,
                              "tail_definition": f"|bin midpoint| >= {frac} x domain_half"})
            for b in range(seg["nbins"]):
                mid = 0.5 * (edges[b] + edges[b + 1])
                bins_rows.append({**keys, "phase": p, "kind": kind, "perspective": label, "start_stage": s,
                                  "current_stage": t, "bin_index": b, "bin_left": float(edges[b]),
                                  "bin_right": float(edges[b + 1]), "bin_midpoint": float(mid),
                                  "count": int(counts[b]),
                                  "expected": float(exp_bins[b]) if exp_bins is not None else expected,
                                  "expected_reason": None if exp_bins is not None else reason,
                                  "tail_pos": bool(pos[b]), "tail_neg": bool(neg[b])})
                vis.append({**keys, "origin": "training", "phase": p, "start_stage": s, "stage": t,
                            "perspective": label, "grid_kind": "width10_bins" if t > 1 else "root_point",
                            "bin_index": b, "bin_left": float(edges[b]), "bin_right": float(edges[b + 1]),
                            "count": float(counts[b]),
                            "probability_mass": float(counts[b]) / float(counts.sum()) if counts.sum() else None})
    return bins_rows, summ_rows, vis


def br_visitation(R: Dict[str, Any]) -> List[Dict[str, Any]]:
    """BR-chain pmf on the final verifier nodes (not width-10 bins)."""
    A = R.get("arrays")
    if A is None:
        return []
    out = []
    for t in (1, 2, 3):
        p = f"final_t{t}_"
        if p + "pmf" not in A:
            continue
        for d, m, r in zip(A[p + "d_grid"], A[p + "pmf"], A[p + "reach"]):
            out.append({**endpoint_keys(R), "origin": "br_chain", "tier": "final", "stage": t,
                        "perspective": "deviator", "grid_kind": "verifier_nodes", "d": float(d),
                        "bin_left": float(d) if t == 1 else None, "bin_right": float(d) if t == 1 else None,
                        "probability_mass": float(m), "reach_mask": bool(r)})
    return out


def resource_rows(R: Dict[str, Any]) -> List[Dict[str, Any]]:
    """resources.csv rows (one per measured segment)."""
    out = []
    for r in R.get("resources", []):
        out.append({**run_keys(R), "segment": r["segment"], "phase": r.get("phase"),
                    "call_index": r.get("call_index"), "global_update": r.get("global_update"),
                    "wall_sec": r.get("wall_sec"), "cpu_sec": r.get("cpu_sec"),
                    "rss_before_bytes": r.get("rss_before_bytes"), "rss_after_bytes": r.get("rss_after_bytes"),
                    "rss_peak_bytes": r.get("rss_peak_bytes"), "peak_method": r.get("peak_method")})
    return out


def exposure_at(R: Dict[str, Any], stage: int, d: float, snap_index: Optional[int]) -> Dict[str, Any]:
    """Training exposure of the bin containing (stage, d): C snapshot and all phases up to it."""
    C = R.get("coverage")
    if C is None or snap_index is None or stage is None or d is None:
        return {"exposure_reason": "coverage snapshot unavailable"}
    spec = GameSpec(**R["cfg"]["game"])
    layout = json.loads(str(C["layout_json"]))
    b = int(strict_gap_bins(spec, stage, np.array([d]), float(C["bin_width"]))[0])
    snap = C["snapshot_counts"][snap_index]
    phases = [str(p) for p in C["phases"]]
    pre = sum(C["phase_totals"][phases.index(p)] for p in ("A", "B"))
    out = {"exposure_bin_index": b, "exposure_C_direct": 0, "exposure_C_continuation": 0,
           "exposure_AB_direct": 0, "exposure_AB_continuation": 0}
    for seg in layout:
        if seg["kind"] != "visits" or seg["current_stage"] != stage:
            continue
        key = "direct" if seg["start_stage"] == stage else "continuation"
        out[f"exposure_C_{key}"] += int(snap[seg["offset"] + b])
        out[f"exposure_AB_{key}"] += int(pre[seg["offset"] + b])
    out["exposure_all_phases_to_call"] = sum(out[k] for k in ("exposure_C_direct", "exposure_C_continuation",
                                                              "exposure_AB_direct", "exposure_AB_continuation"))
    return out


def failure_row(R: Dict[str, Any]) -> Dict[str, Any]:
    """failure_diagnostics.csv row (minimum valid C development call)."""
    dw = R["cfg"]["derived"]["dw"]
    c_reached = any(h["phase"] == "C" for h in R.get("history", []))
    dg = diagnose_minimum_C(R.get("verifier_calls", []), R.get("min_dev_record"), c_reached, dw)
    row = {**min_keys(R), "reason": dg.get("reason")}
    for k, v in dg.items():
        if isinstance(v, dict) and k.startswith(("reach_", "full_")):
            for t, x in v.items():
                row[f"min_C_{k}_t{t}"] = x
        elif k in ("max_reach_state", "max_all_state"):
            v = v or {}
            row[f"min_C_{k}_stage"] = v.get("stage")
            row[f"min_C_{k}_d"] = v.get("d")
            row[f"min_C_{k}_delta_raw"] = v.get("delta")
            row[f"min_C_{k}_delta_over_dw"] = over(v.get("delta"), dw)
            ex = exposure_at(R, v.get("stage"), v.get("d"), dg.get("coverage_snapshot_index"))
            row.update({f"min_C_{k}_{kk}": vv for kk, vv in ex.items()})
        else:
            row[k] = v
    return row


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------

def completeness(R: Dict[str, Any], stage_metric_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-file checks for one run; returns failures, notes and unavailable checks."""
    fails: List[str] = []
    notes: List[str] = []
    unavailable: List[Dict[str, Any]] = []
    st = R.get("status") or {}
    state = R["state"]
    done = state == "done"
    if R["load_errors"]:
        fails += R["load_errors"]
    cfg = R["cfg"]
    phases = {p["name"]: p for p in cfg["phases"]}
    hist = R.get("history", [])
    for name in ("history", "verifier_calls", "events", "resources"):
        info = R.get(name + "_info") or {}
        if info.get("truncated_last_line"):
            notes.append(f"{name}.jsonl: truncated final line ignored")
    # 1. updates vs history
    if [h["global_update"] for h in hist] != list(range(1, len(hist) + 1)):
        fails.append("history global_update sequence is not 1..N")
    if st.get("last_completed_update") is not None and st["last_completed_update"] != len(hist):
        fails.append(f"status last_completed_update {st['last_completed_update']} != history rows {len(hist)}")
    ts = R.get("training_summary")
    if ts and ts["updates_completed"] != len(hist):
        fails.append("training_summary updates_completed != history rows")
    # 2-4. phase counts, per-update counts, scheduled calls
    calls = R.get("verifier_calls", [])
    for pname, ph in phases.items():
        ph_hist = [h for h in hist if h["phase"] == pname]
        pu = ph["per_update"]
        for h in ph_hist:
            if h["n_episodes"] != pu["episodes"] or h["n_environment_steps"] != pu["joint_environment_steps"]:
                fails.append(f"{pname} u{h['global_update']}: episodes/steps {h['n_episodes']}/"
                             f"{h['n_environment_steps']} != {pu['episodes']}/{pu['joint_environment_steps']}")
                break
            if h["stage_transition_counts"] != pu["transitions_by_stage"]:
                fails.append(f"{pname} u{h['global_update']}: stage transition counts differ")
                break
            if {k: v for k, v in h["start_counts"].items() if v} != {k: int(v) for k, v in ph["start_counts"].items()}:
                fails.append(f"{pname} u{h['global_update']}: start counts differ")
                break
        if [h["local_update"] for h in ph_hist] != list(range(1, len(ph_hist) + 1)):
            fails.append(f"{pname}: local_update sequence broken")
        expected = [lu for lu in range(ph["check_every"], len(ph_hist) + 1, ph["check_every"])]
        got = [c["local_update"] for c in calls if c["phase"] == pname]
        missing = [lu for lu in expected if lu not in got]
        extra = [lu for lu in got if lu not in expected]
        if extra:
            fails.append(f"{pname}: unscheduled verifier calls at {extra}")
        for lu in missing:
            if done:
                fails.append(f"{pname}: scheduled call at local {lu} missing")
            else:
                unavailable.append({"phase": pname, "local_update": lu,
                                    "status": "unavailable_due_to_interruption"})
        if not done and ph_hist and len(ph_hist) < ph["cap"] and pname == hist[-1]["phase"]:
            nxt = (len(ph_hist) // ph["check_every"] + 1) * ph["check_every"]
            if nxt <= ph["cap"]:
                unavailable.append({"phase": pname, "local_update": nxt,
                                    "status": "unavailable_due_to_interruption"})
        if pname == "A" and done and len(ph_hist) != ph["cap"]:
            fails.append("phase A did not run its full fixed budget")
    # 5. eligibility/consecutive recomputation
    for pname, ph in phases.items():
        cur = 0
        for c in [c for c in calls if c["phase"] == pname]:
            conc = c.get("concentration") or {}
            cv = conc.get("max_std_norm")
            conc_pass = bool(conc.get("valid") and cv is not None and cv <= cfg["concentration_threshold"])
            if conc_pass != c.get("conc_pass"):
                fails.append(f"{pname} call {c['call_index']}: conc_pass mismatch")
            if pname == "A":
                if c.get("eligible") is not None:
                    fails.append("A call carries eligibility")
                continue
            v = c.get("criterion_value_over_dw")
            sp = bool(c["valid"] and v is not None and v <= ph["threshold_over_dw"])
            el = sp and conc_pass
            if el != c.get("eligible"):
                fails.append(f"{pname} call {c['call_index']}: eligible mismatch")
            cur = cur + 1 if el else 0
            if cur != c.get("consecutive_eligible"):
                fails.append(f"{pname} call {c['call_index']}: consecutive mismatch")
            if pname == "B" and c.get("summary"):
                cg = (c.get("continuation_gain_by_stage") or {}).get("2", {}).get("max_raw")
                if cg is None or abs(cg - c["criterion_value_raw"]) > 0:
                    fails.append("B criterion is not the stage-2 continuation gain")
            if pname == "C" and c.get("summary"):
                if abs(c["criterion_value_raw"] - c["summary"]["dreach"]) > 0:
                    fails.append("C criterion is not dReach")
    # 6. candidate = first eligible C call; totals
    has_cand = persisted_has_candidate(R)
    c_calls = [c for c in calls if c["phase"] == "C"]
    c_elig = [c for c in c_calls if c.get("eligible")]
    er = R.get("endpoint_record")
    if ts:
        if has_cand:
            if not c_elig or c_elig[0]["global_update"] != ts["candidate_update"]:
                fails.append("candidate update is not the first eligible C call")
            if c_calls[-1]["global_update"] != ts["candidate_update"] or len(c_elig) != 1:
                fails.append("training continued after the first eligible C call")
        elif ts["search_outcome"] == "no_candidate_budget_exhausted":
            if c_elig:
                fails.append("eligible C call without candidate")
            if ts["stop_C_local_update"] != phases["C"]["cap"]:
                fails.append("no-candidate run stopped before the C cap")
        tot = ts["totals"]
        if tot["episodes"] != sum(h["n_episodes"] for h in hist) or \
                tot["environment_steps"] != sum(h["n_environment_steps"] for h in hist) or \
                tot["physical_actions"] != 2 * tot["environment_steps"]:
            fails.append("episode/step totals differ from history sums")
    # 7. coverage mass conservation
    C = R.get("coverage")
    if C is not None:
        layout = json.loads(str(C["layout_json"]))
        cph = [str(p) for p in C["phases"]]
        for pi, p in enumerate(cph):
            upd = int(C["phase_updates"][pi])
            if upd != len([h for h in hist if h["phase"] == p]):
                fails.append(f"coverage phase_updates {p} != history")
            sc = phases[p]["start_counts"]
            for seg in layout:
                n = int(sc.get(str(seg["start_stage"]), 0)) * upd
                got = int(C["phase_totals"][pi][seg["offset"]:seg["offset"] + seg["nbins"]].sum())
                if got != n:
                    fails.append(f"coverage {p} {seg['kind']} s{seg['start_stage']} t{seg['current_stage']}: "
                                 f"{got} != {n}")
            for seg in layout:
                t = seg["current_stage"]
                if seg["nbins"] != int(cfg["derived"]["bins"][str(t)]):
                    fails.append(f"coverage bins at stage {t} differ from derived bins")
        if C["snapshot_counts"].shape[0] != len(calls):
            if done:
                fails.append("coverage snapshots != number of verifier calls")
            else:
                notes.append("coverage snapshots != verifier calls (interrupted run)")
    elif R["exists"] and hist:
        fails.append("coverage.npz missing")
    # 8. endpoint identity, final evaluation, grids, profiles
    fe = R.get("final_eval")
    if done:
        if er is None or fe is None:
            fails.append("done run without endpoint record/final_eval")
        else:
            ident = er["identity"]
            if fe.get("checkpoint_identity") != ident:
                fails.append("final_eval checkpoint identity != endpoint record")
            if (st.get("endpoint") or {}).get("identity") != ident:
                fails.append("status endpoint identity != endpoint record")
            if ident["global_update"] != len(hist):
                fails.append("endpoint update != last completed update")
            npz = load_npz(R["dir"] / "checkpoints" / "endpoint_weights.npz") or {}
            if "identity_json" not in npz or json.loads(str(npz["identity_json"])) != ident:
                fails.append("endpoint_weights.npz identity mismatch")
            if not (fe.get("reload_identity") or {}).get("identical"):
                fails.append("reloaded endpoint differs from the in-memory actor")
            drc = fe.get("dev_recompute_vs_call") or {}
            if drc.get("available"):
                worst = max(v for k, v in drc.items() if k.endswith("_over_dw") and v is not None)
                if worst > 1e-12:
                    fails.append(f"recomputed development differs from the original call by {worst:g}")
            for tier, key in (("development", "dev_grid_points"), ("final", "final_grid_points")):
                gp = (fe["tiers"][tier].get("grid_points") or {})
                if gp and gp != {k: int(v) for k, v in cfg["derived"][key].items()}:
                    fails.append(f"{tier} grid points {gp} != derived {cfg['derived'][key]}")
            if fe.get("dense_points_by_stage") != {k: int(v) for k, v in cfg["derived"]["dense_points"].items()}:
                fails.append("dense profile point counts differ from derived")
            A = R.get("arrays") or {}
            for t in (1, 2, 3):
                if f"dense_t{t}_mean_effort" not in A:
                    fails.append(f"dense profile stage {t} missing")
            if "dense_t1_mean_effort" in A and "final_t1_e_hat" in A:
                if not (A["dense_t1_mean_effort"][0] == fe["e1_mean_effort"] == A["final_t1_e_hat"][0]):
                    fails.append("root profile value differs across files")
            if fe.get("pass_flags"):
                re_flags = certification_flags(
                    fe["tiers"]["development"] if "exp_root" in fe["tiers"]["development"] else None,
                    fe["tiers"]["final"] if "exp_root" in fe["tiers"]["final"] else None,
                    fe["dense_concentration"], fe["checkpoint_kind"] == "candidate",
                    cfg["final_certification"], cfg["derived"]["dw"])
                for k in ("numeric_thresholds_pass", "final_joint_pass", "certification", "main_pass",
                          "refine_dreach_pass", "refine_exp_pass", "dense_conc_pass"):
                    if re_flags[k] != fe["pass_flags"][k]:
                        fails.append(f"recomputed flag {k} differs")
                if fe["checkpoint_kind"] != "candidate" and fe["final_joint_pass"]:
                    fails.append("diagnostic terminal marked as final_joint_pass")
            for tier in ("development", "final"):
                pm = fe["tiers"][tier].get("pmf_mass") or {}
                if pm and max(abs(float(v) - 1.0) for v in pm.values()) > 1e-10:
                    notes.append(f"{tier} BR pmf mass error above 1e-10 (verifier marks invalid)")
    # 9. minimum development same-call identity
    mr = R.get("min_dev_record")
    if mr is not None:
        dg = diagnose_minimum_C(calls, mr, True, cfg["derived"]["dw"])
        if not dg.get("min_record_matches_selected_call"):
            fails.append("min_dev_record is not the minimum valid C call")
        if not dg.get("contribution_sum_matches_dreach"):
            fails.append("minimum stage contributions do not sum to dReach")
        ma = R.get("min_arrays") or {}
        if "identity_json" not in ma or json.loads(str(ma["identity_json"])) != mr["identity"]:
            fails.append("min_dev_arrays identity mismatch")
        mw = load_npz(R["dir"] / "checkpoints" / "min_dev_weights.npz") or {}
        if "identity_json" not in mw or json.loads(str(mw["identity_json"])) != mr["identity"]:
            fails.append("min_dev_weights identity mismatch")
        if C is not None and "coverage_snapshot" in ma:
            if not np.array_equal(ma["coverage_snapshot"], C["snapshot_counts"][mr["coverage_snapshot_index"]]):
                fails.append("min_dev coverage snapshot differs from coverage.npz")
        mp = R.get("min_dev_profiles")
        if done:
            if mp is None:
                fails.append("min_dev profiles missing")
            elif mp.get("identity") != mr["identity"] or (mp.get("dev_replay_abs_diff") or 0) > 1e-12:
                fails.append("min_dev profiles/replay do not match the recorded call")
        for t in ("1", "2", "3"):
            if ma and f"minimum_development_t{t}_delta" in ma:
                arr_max = float(ma[f"minimum_development_t{t}_delta"][ma[f"minimum_development_t{t}_reach"]].max())
                if abs(arr_max - float(mr["summary"]["reach_delta_max"][t])) > 1e-12:
                    fails.append(f"min_dev arrays stage {t} reach max differs from record")
    # 10. economics recomputation and mass conservation
    ec = R.get("economics")
    if done and (ec is None or ec.get("error")):
        fails.append("economics missing or errored")
    if ec and ec.get("modes"):
        eps = int(ec["config"]["episodes_per_replicate"]) * int(ec["config"]["replicates"])
        for mode in ("mean", "stochastic"):
            pooled = ec["modes"][mode]["pooled"]
            for t in ("1", "2", "3"):
                stored = pooled["stages"][t]["X"]["mean"]
                recomputed = [r for r in stage_metric_rows if r["mode"] == mode and r["replicate"] == "pooled"
                              and str(r["stage"]) == t and r["source_key"] == "X"]
                if not recomputed or recomputed[0]["mean"] != stored:
                    fails.append(f"economics {mode} stage {t}: recomputed mean differs from JSON")
                if pooled["stages"][t]["eff_p0"]["n"] != eps:
                    fails.append(f"economics {mode} stage {t}: n != episodes")
                if abs(pooled["stages"][t]["hist_mass_player0"] - 1.0) > 1e-12:
                    fails.append(f"economics {mode} stage {t}: histogram mass != 1")
            if not pooled["accounting_ok"]:
                fails.append(f"economics {mode}: payoff accounting residual")
    return {"run_id": R["run"]["run_id"], "state": state, "ok": not fails, "failures": fails,
            "notes": notes, "unavailable_checks": unavailable}


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------

def rate_groups(agg_rows: List[Dict[str, Any]], label: str) -> Dict[Any, Dict[str, Any]]:
    """Aggregate per q, or per (q, arm) when runs carry an arm (arms are never pooled)."""
    keys = sorted({(a["q"], a.get("arm")) for a in agg_rows}, key=lambda k: (k[0], str(k[1])))
    out: Dict[Any, Dict[str, Any]] = {}
    for q, arm in keys:
        rows = [a for a in agg_rows if a["q"] == q and a.get("arm") == arm]
        out[q if arm is None else f"{q}_{arm}"] = aggregate(rows, label)
    return out


def aggregate(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """Counts, three rates with Wilson intervals, and a completed-only sensitivity.

    Args:
        rows: One dict per manifest run with ``state``, ``has_candidate``, ``certification``.
        label: Wording for the rates (debug outcome vs formal reliability).

    Returns:
        Aggregate dict.
    """
    planned = len(rows)
    started = [r for r in rows if r["state"] not in ("pending",)]
    completed = [r for r in rows if r["state"] == "done"]
    op_failed = [r for r in started if r["state"] in ("failed", "interrupted", "started_without_status")]
    running = [r for r in started if r["state"] == "running"]
    attempted = started

    def rates(pop: List[Dict[str, Any]]) -> Dict[str, Any]:
        cand = [r for r in pop if r.get("has_candidate")]
        cert = [r for r in pop if r.get("certification") == "certified"]
        return {"n_attempted": len(pop), "n_candidate": len(cand), "n_certified": len(cert),
                "n_candidate_certification_error": sum(1 for r in cand if r.get("certification") == "error"),
                "n_candidate_not_yet_certified": sum(1 for r in cand if r.get("certification") in (None, "")),
                "candidate_discovery": wilson(len(cand), len(pop)),
                "conditional_certification": (wilson(len(cert), len(cand)) if cand else
                                              {"k": 0, "n": 0, "p": None, "lo": None, "hi": None,
                                               "reason": "no candidates (N/A)"}),
                "end_to_end": wilson(len(cert), len(pop))}

    return {"label": label, "N_planned": planned, "N_started": len(started), "N_completed": len(completed),
            "N_operational_failed": len(op_failed), "N_running": len(running),
            "N_pending": planned - len(started),
            "all_attempted": len(started) == planned,
            "primary": rates(attempted),
            "completed_only_sensitivity": rates(completed),
            "denominator_rule": "primary = all started runs incl. operational failures; "
                                "conditional certification denominator = candidates"}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figures(out_dir: Path, runs: List[Dict[str, Any]], tables: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """Plain matplotlib figures with the plotted data as CSV next to them."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: List[str] = []

    def save(fig: Any, name: str, data: List[Dict[str, Any]]) -> None:
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"{name}.{ext}", dpi=130)
        plt.close(fig)
        write_csv(fig_dir / f"{name}.csv", data)
        made.append(name)

    def short(R: Dict[str, Any]) -> str:
        rid = R["run"]["run_id"]
        return rid.split(f"_q{R['run']['q']}_", 1)[-1]

    qs = sorted({r["run"]["q"] for r in runs if r["exists"]})
    for q in qs:
        rq = [r for r in runs if r["run"]["q"] == q and r["exists"]]
        prof = [p for p in tables["policy_profiles"] if p["q"] == q and p["checkpoint_kind"] != "minimum_development"
                and p["stage"] in (2, 3)]
        if prof:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for ax, t in zip(axes, (2, 3)):
                for r in rq:
                    rows = [p for p in prof if p["run_id"] == r["run"]["run_id"] and p["stage"] == t][::20]
                    if rows:
                        ax.plot([p["d"] for p in rows], [p["mean_effort"] for p in rows],
                                label=f"{short(r)} ({rows[0]['checkpoint_kind']})")
                ax.set_xlabel(f"signed gap d (stage {t})")
                ax.set_ylabel("mean effort m_t(d) [effort]")
                ax.set_title(f"q={q}, stage {t}: endpoint mean policy")
                ax.legend(fontsize=7)
            save(fig, f"policy_curves_q{q}", [p for p in prof][::20])
        asym = [a for a in tables["policy_asymmetry"] if a.get("kind") == "curve" and a["q"] == q
                and a["checkpoint_kind"] != "minimum_development"]
        if asym:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for ax, t in zip(axes, (2, 3)):
                for r in rq:
                    rows = [a for a in asym if a["run_id"] == r["run"]["run_id"] and a["stage"] == t][::10]
                    if rows:
                        ax.plot([a["x"] for a in rows], [a["lead_follow_policy_difference"] for a in rows],
                                label=short(r))
                ax.axhline(0, color="k", lw=0.5)
                ax.set_xlabel(f"lead x > 0 (stage {t})")
                ax.set_ylabel("m_t(x) - m_t(-x) [effort]")
                ax.set_title(f"q={q}, stage {t}: leader - follower policy")
                ax.legend(fontsize=7)
            save(fig, f"asymmetry_q{q}", asym[::10])
        vis = [v for v in tables["state_visitation"] if v["q"] == q and v["origin"] in ("mean_selfplay",
               "stochastic_selfplay") and v["perspective"] == "representative" and v["stage"] in (2, 3)]
        if vis:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for ax, t in zip(axes, (2, 3)):
                for r in rq:
                    for origin, ls in (("mean_selfplay", "-"), ("stochastic_selfplay", "--")):
                        rows = [v for v in vis if v["run_id"] == r["run"]["run_id"] and v["stage"] == t
                                and v["origin"] == origin]
                        if rows:
                            ax.plot([0.5 * (v["bin_left"] + v["bin_right"]) for v in rows],
                                    [v["probability_mass"] for v in rows], ls,
                                    label=f"{short(r)} {origin.split('_')[0]}")
                ax.set_xlabel(f"signed gap bin midpoint (stage {t})")
                ax.set_ylabel("probability mass (representative)")
                ax.set_title(f"q={q}, stage {t}: root self-play visitation")
                ax.legend(fontsize=6)
            save(fig, f"state_histogram_q{q}", vis)
        dev = [d for d in tables["deviations"] if d["q"] == q and d["tier"] == "final" and d["stage"] in (2, 3)]
        if dev:
            fig, axes = plt.subplots(len(rq), 2, figsize=(11, 2.8 * len(rq)), squeeze=False)
            for i, r in enumerate(rq):
                for j, t in enumerate((2, 3)):
                    ax = axes[i][j]
                    rows = [d for d in dev if d["run_id"] == r["run"]["run_id"] and d["stage"] == t]
                    if not rows:
                        continue
                    ax.plot([d["d"] for d in rows], [d["delta_over_dw"] for d in rows], lw=0.8, label="delta_t/DW")
                    ax.plot([d["d"] for d in rows], [d["continuation_gain_over_dw"] for d in rows], lw=0.8,
                            label="(V_BR-V_mean)/DW")
                    rr = [d for d in rows if d["reach_mask"]]
                    if rr:
                        ax.axvspan(min(d["d"] for d in rr), max(d["d"] for d in rr), color="0.9")
                    ax.set_title(f"q={q} {short(r)} stage {t} (final tier; shaded = BR-reachable)",
                                 fontsize=8)
                    ax.set_xlabel("d")
                    ax.set_ylabel("gain / DW")
                    ax.legend(fontsize=6)
            save(fig, f"deviation_profile_q{q}", dev[::3])
        calls = [c for c in tables["verifier_calls"] if c["q"] == q]
        if calls:
            fig, axes = plt.subplots(2, 2, figsize=(12, 7))
            for r in rq:
                rc = [c for c in calls if c["run_id"] == r["run"]["run_id"]]
                b = [c for c in rc if c["phase"] == "B"]
                cc = [c for c in rc if c["phase"] == "C"]
                lab = short(r)
                axes[0][0].plot([c["global_update"] for c in b], [c["criterion_value_over_dw"] for c in b], ".-", label=lab)
                axes[0][1].plot([c["global_update"] for c in b], [c["conc_max_std_norm"] for c in b], ".-", label=lab)
                axes[1][0].plot([c["global_update"] for c in cc], [c["dreach_over_dw"] for c in cc], ".-", label=lab)
                for t, ls in (("1", ":"), ("2", "--"), ("3", "-.")):
                    axes[1][1].plot([c["global_update"] for c in cc],
                                    [c[f"reach_contribution_t{t}_over_dw"] for c in cc], ls, label=f"{lab} t{t}")
            axes[0][0].axhline(0.02, color="k", lw=0.6)
            axes[0][0].set_title(f"q={q} B: max_D2 (V2_BR-V2_mean)/DW")
            axes[0][1].axhline(0.04, color="k", lw=0.6)
            axes[0][1].set_title(f"q={q} B: max std_norm (stages 2,3)")
            axes[1][0].axhline(0.01, color="k", lw=0.6)
            axes[1][0].set_title(f"q={q} C: dReach/DW (development)")
            axes[1][1].set_title(f"q={q} C: stage reach contributions / DW")
            for ax in axes.flat:
                ax.set_xlabel("global update")
                ax.legend(fontsize=6)
            save(fig, f"verifier_curves_q{q}", calls)
        cov = [c for c in tables["coverage"] if c["q"] == q and c["phase"] == "A" and c["kind"] == "start_learner"
               and c["current_stage"] == 3]
        if cov:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            for r in rq:
                rows = [c for c in cov if c["run_id"] == r["run"]["run_id"]]
                ax.plot([c["bin_midpoint"] for c in rows], [c["count"] for c in rows], ".", label=short(r))
                if rows and len({c["expected"] for c in rows}) > 1:
                    ax.plot([c["bin_midpoint"] for c in rows], [c["expected"] for c in rows], "-", lw=0.6,
                            label=f"expected {short(r)}")
            if cov[0]["expected"] is not None and len({c["expected"] for c in cov}) == 1:
                ax.axhline(cov[0]["expected"], color="k", lw=0.8, label="expected")
            ax.set_xlabel("stage-3 learner-signed start gap bin midpoint")
            ax.set_ylabel("count (phase A)")
            ax.set_title(f"q={q}: phase A ES3 exposure per bin")
            ax.legend(fontsize=7)
            save(fig, f"coverage_A_q{q}", cov)
    return made


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def run_row(R: Dict[str, Any], ph_rows: List[Dict[str, Any]], comp: Dict[str, Any],
            fail_row: Dict[str, Any]) -> Dict[str, Any]:
    """runs.csv row (partial and pending runs included)."""
    st = R.get("status") or {}
    ts = R.get("training_summary") or {}
    fe = R.get("final_eval") or {}
    ec = R.get("economics") or {}
    dw = R["cfg"]["derived"]["dw"]
    hist = R.get("history", [])
    fl = fe.get("pass_flags") or {}
    tiers = fe.get("tiers") or {}
    row = {**endpoint_keys(R), "state": R["state"], "stage": st.get("stage"),
           "failure_reason": st.get("failure_reason"),
           "has_candidate": persisted_has_candidate(R), "candidate_update": ts.get("candidate_update"),
           "search_outcome": ts.get("search_outcome") or st.get("search_outcome"),
           "stop_global_update": ts.get("stop_global_update"), "stop_C_local_update": ts.get("stop_C_local_update"),
           "updates_completed": len(hist),
           "total_train_episodes": sum(h["n_episodes"] for h in hist),
           "total_environment_steps": sum(h["n_environment_steps"] for h in hist),
           "physical_action_count": sum(h["n_physical_actions"] for h in hist)}
    for p in ph_rows:
        n = p["phase"]
        row[f"phase_{n}_exit_reason"] = p.get("exit_reason")
        row[f"phase_{n}_local_updates"] = p.get("local_updates")
        row[f"{n}_n_checks"] = p.get("n_checks")
        row[f"{n}_first_eligible_local"] = p.get("first_eligible_local")
        if n == "B":
            row["B_third_consecutive_eligible_local"] = p.get("third_consecutive_eligible_local")
            row["B_longest_consecutive_eligible"] = p.get("longest_consecutive_eligible")
            row["B_eligible_count"] = p.get("eligible_count")
    for tier in ("development", "final"):
        s = tiers.get(tier) or {}
        pre = "dev" if tier == "development" else "final"
        row[f"{pre}_valid"] = s.get("valid")
        row[f"{pre}_exp_root"] = s.get("exp_root")
        row[f"{pre}_exp_root_over_dw"] = s.get("exp_root_over_dw")
        row[f"{pre}_dreach"] = s.get("dreach")
        row[f"{pre}_dreach_over_dw"] = s.get("dreach_over_dw")
        row[f"{pre}_delta_max_all"] = s.get("delta_max_all")
        row[f"{pre}_delta_max_all_over_dw"] = s.get("delta_max_all_over_dw")
        row[f"{pre}_dfull_over_dw"] = s.get("dfull_over_dw")
        row[f"{pre}_v_mean_root"] = s.get("v_mean_root")
    for k in ("main_pass", "refine_dreach_diff_over_dw", "refine_dreach_pass", "refine_exp_diff_over_dw",
              "refine_exp_pass", "refine_delta_max_all_diff_over_dw", "dense_conc_max_std_norm",
              "dense_conc_pass", "numeric_thresholds_pass"):
        row[k] = fl.get(k)
    row["certification"] = fe.get("certification") if fe else (st.get("certification"))
    row["final_joint_pass"] = fe.get("final_joint_pass") if fe else None
    row["final_eval_error"] = fe.get("error")
    A = R.get("arrays") or {}
    row["e1_mean_effort"] = fe.get("e1_mean_effort")
    for t in (2, 3):
        if f"dense_t{t}_d" in A:
            j = int(np.argmin(np.abs(A[f"dense_t{t}_d"])))
            row[f"e{t}_at_0"] = float(A[f"dense_t{t}_mean_effort"][j])
    row["reload_identical"] = (fe.get("reload_identity") or {}).get("identical")
    for k in ("min_valid_C_dreach_over_dw", "min_C_global_update", "min_C_local_update", "min_C_conc", "reason"):
        row["min_" + k if k == "reason" else k] = fail_row.get(k)
    row["min_checkpoint_ref"] = json.dumps(fail_row.get("min_checkpoint_ref")) if fail_row.get("min_checkpoint_ref") else None
    if ec.get("modes"):
        mu = ec["modes"]["mean"]["pooled"]["episode"]["U"]
        su = ec["modes"]["stochastic"]["pooled"]["episode"]["U"]
        cmp_ = ec.get("mean_payoff_vs_dp") or {}
        row.update(mean_selfplay_U=mu["mean"], mean_selfplay_U_mcse=mu["mcse"],
                   dp_v1_mean_root_final=cmp_.get("v1_mean_root_final"), mc_minus_dp=cmp_.get("difference"),
                   mc_minus_dp_over_mcse=cmp_.get("z"), stochastic_selfplay_U=su["mean"],
                   stochastic_selfplay_U_mcse=su["mcse"])
        for mode in ("mean", "stochastic"):
            for t in ("1", "2", "3"):
                row[f"{mode}_E_e_t{t}"] = ec["modes"][mode]["pooled"]["stages"][t]["X"]["mean"]
    row["economics_error"] = ec.get("error")
    res = R.get("resources", [])
    row["process_peak_rss_bytes"] = st.get("process_peak_rss_bytes")
    row["max_segment_peak_rss_bytes"] = max((r.get("rss_peak_bytes") or 0 for r in res), default=None) or None
    for seg in ("training", "final_eval_total", "economics_total"):
        rr = [r for r in res if r["segment"] == seg]
        row[f"{seg}_wall_sec"] = rr[-1]["wall_sec"] if rr else None
        row[f"{seg}_cpu_sec"] = rr[-1]["cpu_sec"] if rr else None
    dv = [r for r in res if r["segment"] == "dev_verifier"]
    row["dev_verifier_calls"] = len(dv)
    row["dev_verifier_wall_sec_total"] = sum(r["wall_sec"] for r in dv) if dv else None
    row["dev_verifier_cpu_sec_total"] = sum(r["cpu_sec"] for r in dv) if dv else None
    row["train_update_wall_sec_total"] = sum(h["wall_sec"] for h in hist) if hist else None
    row["completeness_ok"] = comp["ok"] if R["exists"] else None
    row["completeness_failures"] = comp["failures"] if R["exists"] else None
    row["unavailable_reason"] = "pending (not started)" if not R["exists"] else None
    return row


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort", required=True, choices=["smoke", "pilot", "formal", "asamp_smoke", "asamp",
                                                       "asamp_abc_smoke", "asamp_abc"])
    p.add_argument("--q", type=int, default=None)
    p.add_argument("--manifest", action="append", type=Path, help="explicit run manifest; repeat for multiple q values")
    p.add_argument("--check-completeness", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None, help="default: regenerated/<cohort>")
    args = p.parse_args()
    mpaths = args.manifest or sorted((E / "manifests").glob(f"{args.cohort}_q*.json"))
    if args.q is not None:
        mpaths = [mp for mp in mpaths if int(json.loads(mp.read_text())["q"]) == args.q]
    if not mpaths:
        raise SystemExit("no manifests found")
    out_dir = args.out_dir or E / "regenerated" / (args.cohort if args.q is None else f"{args.cohort}_q{args.q}")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, Any]] = []
    for mp in mpaths:
        m = json.loads(mp.read_text())
        if m["cohort"] != args.cohort:
            p.error(f"{mp}: cohort {m['cohort']!r} differs from --cohort {args.cohort!r}")
        for r in m["runs"]:
            runs.append(load_run(m, r))
    names = ("runs", "phases", "verifier_calls", "verifier_summary", "verifier_stage_metrics", "deviations",
             "policy_profiles", "policy_asymmetry", "stage_metrics", "state_visitation", "coverage",
             "coverage_summary", "failure_diagnostics", "resources")
    tables: Dict[str, List[Dict[str, Any]]] = {n: [] for n in names}
    comps = []
    for R in runs:
        per: Dict[str, List[Dict[str, Any]]] = {n: [] for n in names}
        if R["exists"]:
            per["phases"] = phase_rows(R)
            per["verifier_calls"] = call_rows(R)
            per["verifier_summary"], per["verifier_stage_metrics"] = tier_rows(R)
            per["deviations"] = deviation_rows(R)
            per["policy_profiles"], asym_curve = profile_rows(R)
            per["stage_metrics"], asym_occ, vis_self = econ_rows(R)
            per["policy_asymmetry"] = asym_curve + asym_occ
            per["coverage"], per["coverage_summary"], vis_train = coverage_tables(R)
            per["state_visitation"] = vis_train + vis_self + br_visitation(R)
            per["failure_diagnostics"] = [failure_row(R)]
            per["resources"] = resource_rows(R)
            comp = completeness(R, per["stage_metrics"])
        else:
            comp = {"run_id": R["run"]["run_id"], "state": "pending", "ok": None, "failures": [],
                    "notes": ["pending: manifest run not started"], "unavailable_checks": []}
        comps.append(comp)
        per["runs"] = [run_row(R, per["phases"] or [{"phase": n} for n in ("A", "B", "C")], comp,
                               per["failure_diagnostics"][0] if per["failure_diagnostics"] else {})]
        if R["exists"]:
            for n, rows in per.items():
                write_csv(R["dir"] / "tables" / f"{n}.csv", rows)
        for n in names:
            tables[n] += per[n]
    for n in names:
        write_csv(out_dir / f"{n}.csv", tables[n])
    label = ("formal reliability" if args.cohort == "formal" else "implementation/debug outcome (not a reliability estimate)")
    agg_rows = [{"q": R["run"]["q"], "arm": R["run"].get("arm"), "state": R["state"],
                 "has_candidate": persisted_has_candidate(R),
                 "certification": (R.get("final_eval") or {}).get("certification")} for R in runs]
    by_q = rate_groups(agg_rows, label)
    if any([p["name"] for p in R["cfg"]["phases"]] == ["A"] for R in runs):
        by_q = {}          # A-only study: no C phase, candidate rates do not apply
    rate_rows = [] if by_q else [{"cohort": args.cohort, "reason": "A-only study: no C phase; candidate "
                                  "discovery/certification rates not applicable"}]
    for q, agg in by_q.items():
        for pop in ("primary", "completed_only_sensitivity"):
            a = agg[pop]
            for rate in ("candidate_discovery", "conditional_certification", "end_to_end"):
                w = a[rate]
                rate_rows.append({"cohort": args.cohort, "q": q, "population": pop, "rate": rate, "k": w["k"],
                                  "n": w["n"], "p": w["p"], "wilson95_lo": w["lo"], "wilson95_hi": w["hi"],
                                  "reason": w.get("reason"), "label": label})
    write_csv(out_dir / "rates.csv", rate_rows)
    made = [] if args.no_figures else figures(out_dir, runs, tables)
    comp_out = {"cohort": args.cohort, "q": args.q, "generated": now_iso(), "runs": comps,
                "aggregate_by_q": by_q, "figures": made}
    write_json_atomic(out_dir / "completeness.json", comp_out)
    write_summary(out_dir, args.cohort, runs, tables, by_q, comps)
    bad = [c for c in comps if c["ok"] is False]
    print(f"[report] {out_dir}: {len(runs)} manifest runs; states "
          f"{ {s: sum(1 for R in runs if R['state'] == s) for s in sorted({R['state'] for R in runs})} }")
    for c in comps:
        print(f"  {c['run_id']}: state={c['state']} complete={c['ok']} failures={len(c['failures'])} "
              f"notes={len(c['notes'])} unavailable={len(c['unavailable_checks'])}")
        for f in c["failures"][:20]:
            print(f"     FAIL {f}")
    if args.check_completeness and bad:
        return 1
    return 0


def write_summary(out_dir: Path, cohort: str, runs: List[Dict[str, Any]], tables: Dict[str, List[Dict[str, Any]]],
                  by_q: Dict[int, Dict[str, Any]], comps: List[Dict[str, Any]]) -> None:
    """Machine-written overview; interpretation lives in the hand-written reports."""
    def f(v: Any, nd: int = 5) -> str:
        if v is None or v == "":
            return "—"
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            return f"{v:.{nd}g}"
        return str(v)

    L = [f"# AUTO_SUMMARY — cohort `{cohort}`", "", f"Generated {now_iso()} by make_report.py. "
         "All values come from the CSVs in this directory; interpretation is in PILOT_REPORT.md.", ""]
    for q, agg in by_q.items():
        a = agg["primary"]
        L += [f"## q={q}: counts ({agg['label']})", "",
              f"N_planned={agg['N_planned']}, N_started={agg['N_started']}, N_completed={agg['N_completed']}, "
              f"N_operational_failed={agg['N_operational_failed']}, N_pending={agg['N_pending']}", ""]
        for rate in ("candidate_discovery", "conditional_certification", "end_to_end"):
            w = a[rate]
            L.append(f"- {rate}: {w['k']}/{w['n']}  p={f(w['p'])}  Wilson95=[{f(w['lo'])}, {f(w['hi'])}] "
                     f"{w.get('reason') or ''}")
        L.append("")
    L += ["## Runs", "", "| run | state | outcome | B exit (local) | C local | final dReach/DW | refine dReach | "
          "refine EXP | dense conc | certification | min C dReach/DW (u) | complete |", "|" + "---|" * 12]
    for r in tables["runs"]:
        L.append(f"| {r['run_id']} | {r['state']} | {f(r.get('search_outcome'))} | "
                 f"{f(r.get('phase_B_exit_reason'))} ({f(r.get('phase_B_local_updates'))}) | "
                 f"{f(r.get('phase_C_local_updates'))} | {f(r.get('final_dreach_over_dw'))} | "
                 f"{f(r.get('refine_dreach_diff_over_dw'))} | {f(r.get('refine_exp_diff_over_dw'))} | "
                 f"{f(r.get('dense_conc_max_std_norm'))} | {f(r.get('certification'))} | "
                 f"{f(r.get('min_valid_C_dreach_over_dw'))} ({f(r.get('min_C_global_update'))}) | "
                 f"{f(r.get('completeness_ok'))} |")
    L += ["", "## Completeness", ""]
    for c in comps:
        L.append(f"- {c['run_id']}: ok={c['ok']}; failures={c['failures'] or 'none'}; notes={c['notes'] or 'none'}; "
                 f"unavailable={c['unavailable_checks'] or 'none'}")
    L += ["", "## Files", "", "runs.csv, phases.csv, verifier_calls.csv, verifier_summary.csv, verifier_stage_metrics.csv, "
          "deviations.csv, policy_profiles.csv, policy_asymmetry.csv, stage_metrics.csv, state_visitation.csv, "
          "coverage.csv, coverage_summary.csv, failure_diagnostics.csv, resources.csv, rates.csv, completeness.json, "
          "figures/*.png|pdf|csv. Per-run copies: runs/<run_id>/tables/."]
    (out_dir / "AUTO_SUMMARY.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
