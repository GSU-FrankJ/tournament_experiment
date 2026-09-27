#!/usr/bin/env python3
"""Per-seed tables and endpoint-group economics for one finished cohort.

Reads only what ``make_report.py`` and the runs already wrote (``reports/<cohort>/*.csv``,
per-run ``economics.json``, ``launch_logs/*.json``); it never touches run data. Writes to
``reports/<cohort>/``:

- ``verification_by_seed.csv``: per run and tier (development/final of the endpoint, plus the
  minimum-development call): validity, EXP_root, dReach, Delta_max_all, dfull and per-stage
  reach/full contributions, each raw and /DW, plus dev-final refinement differences raw and /DW.
- ``economics_by_group.csv``: endpoint groups (certified candidate, uncertified candidate,
  diagnostic terminal) per q, mode and stage; across-seed mean/SD/min/max kept apart from the
  within-run rollout MCSE.
- ``FORMAL_TABLES.md``: the per-seed markdown tables quoted in the cohort report.

    .venv/bin/python -B experiments/three_stage_implementation_pilot_20260924/formal_summary.py --cohort formal
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import E, now_iso  # noqa: E402

GROUPS = ("certified_candidate", "uncertified_candidate", "diagnostic_terminal")


def read_csv(path: Path) -> List[Dict[str, str]]:
    """All rows of a CSV as dicts of strings."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(v: Any) -> Optional[float]:
    """CSV cell -> float (None for empty)."""
    if v is None or v == "":
        return None
    return float(v)


def f(v: Any, nd: int = 4) -> str:
    """Markdown cell: significant digits, em dash for missing."""
    if v is None or v == "":
        return "—"
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            return v
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e7:
            return str(int(v))
        return f"{v:.{nd}g}"
    return str(v)


def endpoint_group(r: Dict[str, str]) -> Optional[str]:
    """Economics group of a run's endpoint (None when no endpoint was saved)."""
    kind = r.get("checkpoint_kind")
    if kind == "candidate":
        return "certified_candidate" if r.get("certification") == "certified" else "uncertified_candidate"
    if kind == "diagnostic_terminal":
        return "diagnostic_terminal"
    return None


def outcome_class(r: Dict[str, str]) -> str:
    """Search outcome vs operational state, kept separate."""
    if r["state"] == "pending":
        return "pending"
    if r.get("search_outcome") == "candidate_found":
        base = "candidate"
    elif r.get("search_outcome") == "no_candidate_budget_exhausted":
        base = "no_candidate"
    else:
        return f"execution_error_before_search_end ({r['state']})"
    if r["state"] == "done":
        return base
    return f"{base}; post-search error ({r['state']})"


def launch_records(cohort: str) -> Dict[str, List[Dict[str, Any]]]:
    """Launcher entries per run_id from every launch log that loaded this cohort's manifests."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for p in sorted((E / "launch_logs").glob("launch_*.json")):
        rec = json.loads(p.read_text())
        if not any(Path(m).name.startswith(f"{cohort}_q") for m in rec.get("manifests", [])):
            continue
        for r in rec.get("runs", []):
            out.setdefault(r["run_id"], []).append({**r, "launch_log": p.name})
    return out


def verification_rows(runs: List[Dict[str, str]], vsum: List[Dict[str, str]],
                      vstage: List[Dict[str, str]], fail: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    """verification_by_seed.csv rows (raw and /DW side by side)."""
    out: List[Dict[str, Any]] = []
    for r in runs:
        dw = 4.0
        base = {k: r[k] for k in ("run_id", "q", "seed", "checkpoint_kind", "checkpoint_global_update")}
        stages = {(s["tier"], s["stage"]): s for s in vstage if s["run_id"] == r["run_id"]}
        for s in [s for s in vsum if s["run_id"] == r["run_id"]]:
            row: Dict[str, Any] = {**base, "tier": s["tier"], "tier_checkpoint_kind": s["checkpoint_kind"],
                                   "tier_checkpoint_global_update": s["checkpoint_global_update"],
                                   "valid": s["valid"], "invalid_reasons": s["invalid_reasons"], "error": s["error"]}
            for k in ("exp_root", "dreach", "delta_max_all", "dfull"):
                row[f"{k}_raw"] = num(s[k])
                row[f"{k}_over_dw"] = num(s[f"{k}_over_dw"])
            for t in ("1", "2", "3"):
                st = stages.get((s["tier"], t), {})
                for k in ("reach_contribution", "full_contribution", "continuation_gain_max"):
                    row[f"{k}_t{t}_raw"] = num(st.get(f"{k}_raw"))
                    row[f"{k}_t{t}_over_dw"] = num(st.get(f"{k}_over_dw"))
                row[f"reach_argmax_d_t{t}"] = num(st.get("reach_argmax_d"))
                row[f"full_argmax_d_t{t}"] = num(st.get("full_argmax_d"))
            row["pdl_residual_over_dw"] = num(s["pdl_residual_over_dw"])
            row["pmf_mass_err_max"] = num(s["pmf_mass_err_max"])
            out.append(row)
        dev = next((x for x in out if x["run_id"] == r["run_id"] and x["tier"] == "development"), None)
        fin = next((x for x in out if x["run_id"] == r["run_id"] and x["tier"] == "final"), None)
        if dev and fin:
            ref: Dict[str, Any] = {**base, "tier": "final_minus_development"}
            for k in ["exp_root", "dreach", "delta_max_all", "dfull"] + [
                    f"{c}_t{t}" for c in ("reach_contribution", "full_contribution", "continuation_gain_max")
                    for t in ("1", "2", "3")]:
                a, b = dev.get(f"{k}_raw"), fin.get(f"{k}_raw")
                ref[f"{k}_raw"] = None if a is None or b is None else b - a
                ref[f"{k}_over_dw"] = None if ref[f"{k}_raw"] is None else ref[f"{k}_raw"] / dw
            out.append(ref)
    return out


def econ_by_run(runs: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    """Pooled economics per run from economics.json (None when missing/errored)."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in runs:
        p = E / "runs" / r["run_id"] / "economics.json"
        if not p.exists():
            continue
        ec = json.loads(p.read_text())
        if not ec.get("modes"):
            continue
        out[r["run_id"]] = ec
    return out


def stage_value(ec: Dict[str, Any], mode: str, t: str, key: str) -> Dict[str, Any]:
    return ec["modes"][mode]["pooled"]["stages"][t][key]


def group_rows(runs: List[Dict[str, str]], econ: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """economics_by_group.csv rows; every group appears per q, empty groups with n_runs=0."""
    rows: List[Dict[str, Any]] = []
    qs = sorted({int(r["q"]) for r in runs})

    def spread(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {"mean_across_runs": None, "sd_across_runs": None, "min": None, "max": None}
        return {"mean_across_runs": statistics.fmean(vals),
                "sd_across_runs": statistics.stdev(vals) if len(vals) > 1 else None,
                "min": min(vals), "max": max(vals)}

    for q in qs:
        for g in GROUPS:
            members = [r for r in runs if int(r["q"]) == q and endpoint_group(r) == g]
            ids = [r["run_id"] for r in members]
            with_econ = [i for i in ids if i in econ]
            head = {"q": q, "group": g, "n_runs": len(members), "n_runs_with_economics": len(with_econ),
                    "seeds": " ".join(r["seed"] for r in members) or None,
                    "note": None if members else "no run in this group"}
            for name, key in (("e1_at_0", "e1_mean_effort"), ("e2_at_0", "e2_at_0"), ("e3_at_0", "e3_at_0")):
                vals = [num(r[key]) for r in members if num(r[key]) is not None]
                rows.append({**head, "mode": "policy_curve", "stage": name[1], "quantity": name,
                             **spread(vals), "max_rollout_mcse": None})
            for mode in ("mean", "stochastic"):
                for t in ("1", "2", "3"):
                    for quantity, key, field in (("E_effort_representative", "X", "mean"),
                                                 ("E_effort_sq_representative", "Y", "mean"),
                                                 ("leader_minus_follower_paired", "lf_diff", "mean")):
                        vals, mcses = [], []
                        for i in with_econ:
                            v = stage_value(econ[i], mode, t, key)
                            if v.get(field) is not None:
                                vals.append(float(v[field]))
                                if v.get("mcse") is not None:
                                    mcses.append(float(v["mcse"]))
                        rows.append({**head, "mode": mode, "stage": t, "quantity": quantity, **spread(vals),
                                     "max_rollout_mcse": max(mcses) if mcses else None})
                    vals = [float(stage_value(econ[i], mode, t, "representative_expected_cost"))
                            for i in with_econ]
                    rows.append({**head, "mode": mode, "stage": t, "quantity": "expected_cost_representative",
                                 **spread(vals), "max_rollout_mcse": None})
                vals = [float(econ[i]["modes"][mode]["pooled"]["episode"]["U"]["mean"]) for i in with_econ]
                mc = [float(econ[i]["modes"][mode]["pooled"]["episode"]["U"]["mcse"]) for i in with_econ]
                rows.append({**head, "mode": mode, "stage": "episode", "quantity": "U_mean_payoff",
                             **spread(vals), "max_rollout_mcse": max(mc) if mc else None})
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else (repr(r[k]) if isinstance(r[k], float) else r[k]))
                        for k in cols})


def reasons(*tiers: Dict[str, str]) -> str:
    """Union of the tiers' invalid_reasons (JSON lists in the CSV); 'none' when all are empty."""
    out: List[str] = []
    for t in tiers:
        raw = t.get("invalid_reasons")
        if not raw:
            continue
        try:
            vals = json.loads(raw)
        except json.JSONDecodeError:
            vals = [raw]
        out += [str(v) for v in vals if str(v) not in out]
    return "; ".join(out).replace("|", "/") or "none"


def md_table(header: List[str], rows: List[List[Any]]) -> List[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(f(c) if not isinstance(c, str) else c for c in row) + " |" for row in rows]
    return out


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort", default="formal")
    p.add_argument("--out-dir", default=None, help="default: regenerated/<cohort>")
    args = p.parse_args()
    src = E / "reports" / args.cohort
    out_dir = Path(args.out_dir) if args.out_dir else E / "regenerated" / args.cohort
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = read_csv(src / "runs.csv")
    runs.sort(key=lambda r: (int(r["q"]), int(r["seed"])))
    phases = read_csv(src / "phases.csv")
    vsum = read_csv(src / "verifier_summary.csv")
    vstage = read_csv(src / "verifier_stage_metrics.csv")
    calls = read_csv(src / "verifier_calls.csv")
    fail = {r["run_id"]: r for r in read_csv(src / "failure_diagnostics.csv")}
    rates = read_csv(src / "rates.csv")
    comp = json.loads((src / "completeness.json").read_text())
    launches = launch_records(args.cohort)
    econ = econ_by_run(runs)

    ver = verification_rows(runs, vsum, vstage, fail)
    write_csv(out_dir / "verification_by_seed.csv", ver)
    grp = group_rows(runs, econ)
    write_csv(out_dir / "economics_by_group.csv", grp)

    ph = {(r["run_id"], r["phase"]): r for r in phases}
    L: List[str] = [f"# Per-seed tables — cohort `{args.cohort}`", "",
                    f"Generated {now_iso()} by formal_summary.py from `reports/{args.cohort}/*.csv`, "
                    "per-run `economics.json` and `launch_logs/`. /DW = divided by DW = w_h − w_l = 4; "
                    "raw values are in `runs.csv`, `verification_by_seed.csv` and `verifier_stage_metrics.csv`.", ""]

    # --- rates
    L += ["## R. Rates per q (primary population = all started runs)", ""]
    rr = []
    for q in sorted({r["q"] for r in rates}):
        for pop in ("primary", "completed_only_sensitivity"):
            for rate in ("candidate_discovery", "conditional_certification", "end_to_end"):
                x = next(y for y in rates if y["q"] == q and y["population"] == pop and y["rate"] == rate)
                val = "N/A" if x["reason"] else f"{x['k']}/{x['n']} = {f(x['p'])}"
                ci = "N/A" if x["reason"] else f"[{f(x['wilson95_lo'])}, {f(x['wilson95_hi'])}]"
                rr.append([q, pop, rate, val, ci, x["reason"] or ""])
    L += md_table(["q", "population", "rate", "k/n = p", "Wilson 95%", "note"], rr) + [""]

    # --- search
    L += ["## S. Search (per seed)", ""]
    rows = []
    for r in runs:
        lr = launches.get(r["run_id"], [])
        lwall = lr[-1].get("wall_sec") if lr else None
        lrc = ",".join(str(x.get("returncode")) for x in lr) if lr else "—"
        b = ph.get((r["run_id"], "B"), {})
        rows.append([r["q"], r["seed"], r["state"], outcome_class(r),
                     f"{r['phase_A_exit_reason'] or '—'} ({f(r['phase_A_local_updates'])})",
                     f"{r['phase_B_exit_reason'] or '—'} ({f(r['phase_B_local_updates'])})",
                     f"{f(b.get('eligible_count'))}/{f(b.get('n_checks'))}; longest {f(b.get('longest_consecutive_eligible'))}",
                     f"{r['phase_C_exit_reason'] or '—'} ({f(r['phase_C_local_updates'])})",
                     f(r["candidate_update"]), f(r["stop_global_update"]), f(r["total_train_episodes"]),
                     f(r["total_environment_steps"]), f(r["physical_action_count"]),
                     f(num(lwall), 5) if lwall is not None else "—", lrc,
                     f(num(r["training_wall_sec"]), 5), f(num(r["training_cpu_sec"]), 5),
                     f(num(r["final_eval_total_wall_sec"]), 3), f(num(r["economics_total_wall_sec"]), 3),
                     f(None if not r["process_peak_rss_bytes"] else num(r["process_peak_rss_bytes"]) / 2 ** 20, 4)])
    L += md_table(["q", "seed", "state", "outcome", "A exit (local)", "B exit (local)", "B eligible/checks",
                   "C exit (local)", "candidate update", "stop update", "episodes", "joint env steps",
                   "physical actions", "run wall s (launcher)", "exit code", "train wall s", "train CPU s",
                   "final wall s", "econ wall s", "peak RSS MiB"], rows) + [""]

    L += ["## P. Phase B and C development checks (per seed)", "",
          "Failure types per check: S = strategic only, K = concentration only, SK = both, I = invalid. "
          "B strategic value = max_D2 (V2_BR − V2_mean)/DW (threshold 0.02); C = dReach/DW (threshold 0.01); "
          "concentration = max std_norm on the dev grids of the phase's stages (threshold 0.04).", ""]
    rows = []
    for r in runs:
        cells = [r["q"], r["seed"]]
        for name in ("B", "C"):
            x = ph.get((r["run_id"], name), {})
            if not x or not x.get("n_checks"):
                cells += ["not reached", "—", "—", "—", "—"]
                continue
            first = x.get("first_eligible_local") or "none"
            third = x.get("third_consecutive_eligible_local") if name == "B" else None
            cells += [f"{x['exit_reason']} ({x['local_updates']})",
                      f"{x['eligible_count']}/{x['n_checks']}; first {first}; longest {x['longest_consecutive_eligible']}"
                      + (f"; 3rd consec. {third or 'none'}" if name == "B" else ""),
                      f"{x['n_strategic_only_fail']}/{x['n_conc_only_fail']}/{x['n_both_fail']}/{x['n_invalid']}",
                      f"{f(x['criterion_min_over_dw'])} / {f(x['criterion_last_over_dw'])} "
                      f"(≤thr {x['criterion_n_below_threshold']}×)",
                      f"{f(x['conc_min'])} / {f(x['conc_last'])}"]
        rows.append(cells)
    L += md_table(["q", "seed", "B exit (local)", "B eligible", "B S/K/SK/I", "B strategic min / last",
                   "B conc min / last", "C exit (local)", "C eligible", "C S/K/SK/I", "C dReach min / last",
                   "C conc min / last"], rows) + [""]

    # --- verification: endpoint tiers
    L += ["## V1. Verification of the saved endpoint (development and final tiers re-run on the reloaded weights)",
          "", "Values /DW. `kind` = candidate or diagnostic_terminal (C-cap weights). Raw = 4 × /DW.", ""]
    rows = []
    for r in runs:
        v = {x["tier"]: x for x in vsum if x["run_id"] == r["run_id"]}
        dv, fv = v.get("development", {}), v.get("final", {})
        rows.append([r["q"], r["seed"], f"{r['checkpoint_kind'] or '—'} ({f(r['checkpoint_global_update'])})",
                     f"{dv.get('valid', '—')}/{fv.get('valid', '—')}",
                     reasons(dv, fv),
                     f"{f(r['dev_exp_root_over_dw'])} / {f(r['final_exp_root_over_dw'])}",
                     f"{f(r['dev_dreach_over_dw'])} / {f(r['final_dreach_over_dw'])}",
                     f"{f(r['dev_delta_max_all_over_dw'])} / {f(r['final_delta_max_all_over_dw'])}",
                     f(r["final_dfull_over_dw"]),
                     f(r["refine_dreach_diff_over_dw"], 2), f(r["refine_exp_diff_over_dw"], 2),
                     f(r["refine_delta_max_all_diff_over_dw"], 2), f(r["dense_conc_max_std_norm"])])
    L += md_table(["q", "seed", "kind (update)", "valid dev/final", "invalid reasons", "EXP_root dev / final",
                   "dReach dev / final", "Delta_max_all dev / final", "dfull final", "abs ΔdReach dev−final",
                   "abs ΔEXP dev−final", "abs ΔDelta_max_all dev−final", "dense conc"], rows) + [""]

    L += ["## V2. Certification components (thresholds: main dReach_final ≤ 0.01; refine ≤ 0.002; dense conc ≤ 0.04)",
          ""]
    rows = []
    for r in runs:
        rows.append([r["q"], r["seed"], r["checkpoint_kind"] or "—", f"{r['dev_valid'] or '—'}/{r['final_valid'] or '—'}",
                     r["main_pass"] or "—", r["refine_dreach_pass"] or "—", r["refine_exp_pass"] or "—",
                     r["dense_conc_pass"] or "—", r["numeric_thresholds_pass"] or "—", r["has_candidate"] or "—",
                     r["certification"] or "—", r["final_joint_pass"] or "—"])
    L += md_table(["q", "seed", "kind", "valid dev/final", "main", "refine dReach", "refine EXP", "dense conc",
                   "numeric (1-5)", "has_candidate", "certification", "final_joint_pass"], rows) + [""]

    L += ["## V3. Stage-wise final-tier deviations at the endpoint (/DW)", "",
          "reach_t = max over the BR-reachable mask R_t of the one-step gain δ_t; dReach = Σ_t reach_t. "
          "full_t = max over D_t of δ_t; Delta_max_all = max_t full_t (a maximum, not a sum). "
          "cg_t = max over D_t of the continuation gain V_t^BR − V_t^mean.", ""]
    rows = []
    for r in runs:
        st = {s["stage"]: s for s in vstage if s["run_id"] == r["run_id"] and s["tier"] == "final"}
        cells = [r["q"], r["seed"]]
        for t in ("1", "2", "3"):
            s = st.get(t, {})
            cells.append(f"{f(s.get('reach_contribution_over_dw'))} @ {f(s.get('reach_argmax_d'))}")
        for t in ("2", "3"):
            s = st.get(t, {})
            cells.append(f"{f(s.get('full_contribution_over_dw'))} @ {f(s.get('full_argmax_d'))}")
        for t in ("2", "3"):
            s = st.get(t, {})
            cells.append(f"{f(s.get('continuation_gain_max_over_dw'))} @ {f(s.get('continuation_gain_argmax_d'))}")
        rows.append(cells)
    L += md_table(["q", "seed", "reach t1 @ d", "reach t2 @ d", "reach t3 @ d", "full t2 @ d", "full t3 @ d",
                   "cg t2 @ d", "cg t3 @ d"], rows) + [""]

    L += ["## V4. Minimum valid C development call vs the actual endpoint (reported separately)", "",
          "The minimum is diagnostic only: it is never promoted and gets no final evaluation. "
          "`endpoint dev call` is the development call made during training at the endpoint update "
          "(for a diagnostic terminal, the C1800 check).", ""]
    rows = []
    for r in runs:
        d = fail.get(r["run_id"], {})
        at = next((x for x in vsum if x["run_id"] == r["run_id"] and x["tier"] == "development_call_at_endpoint"), {})
        rows.append([r["q"], r["seed"], r["checkpoint_kind"] or "—",
                     f"{f(d.get('min_valid_C_dreach_over_dw'))} @ {f(d.get('min_C_global_update'))}"
                     + (f" ({d.get('reason')})" if d.get("reason") else ""),
                     f(d.get("min_C_exp_root_over_dw")), f(d.get("min_C_delta_max_all_over_dw")), f(d.get("min_C_conc")),
                     f"{f(at.get('dreach_over_dw'))} @ {f(r['checkpoint_global_update'])}",
                     f(r["dev_dreach_over_dw"]), f(r["final_dreach_over_dw"])])
    L += md_table(["q", "seed", "endpoint kind", "min valid C dev dReach/DW @ update", "EXP_root/DW at min",
                   "Delta_max_all/DW at min", "conc at min", "endpoint dev call dReach/DW @ update",
                   "endpoint re-run dev dReach/DW", "endpoint final dReach/DW"], rows) + [""]

    # --- economics
    L += ["## E. Economic policy of the actual endpoint (per seed)", "",
          "E[e_t]: representative X=(e0+e1)/2 under root self-play from d=0; pooled 3 × 200000 episodes per mode; "
          "(MCSE, episode unit). LF = paired leader − follower effort at stage t (ties excluded). "
          "sd(gap) = SD of the pre-action physical gap under mean self-play.", ""]
    rows = []
    for r in runs:
        ec = econ.get(r["run_id"])
        g = endpoint_group(r) or "no endpoint"
        if ec is None:
            rows.append([r["q"], r["seed"], g, f(r["e1_mean_effort"]), f(r["e2_at_0"]), f(r["e3_at_0"])]
                        + ["missing"] * 7)
            continue
        cells = [r["q"], r["seed"], g, f(r["e1_mean_effort"]), f(r["e2_at_0"]), f(r["e3_at_0"])]
        for mode in ("mean", "stochastic"):
            for t in ("2", "3"):
                x = stage_value(ec, mode, t, "X")
                cells.append(f"{x['mean']:.3f} ({x['mcse']:.2g})")
        for t in ("2", "3"):
            x = stage_value(ec, "mean", t, "lf_diff")
            cells.append("—" if x.get("mean") is None else f"{x['mean']:.2f} ({x['mcse']:.2g})")
        gp = [stage_value(ec, "mean", t, "gap_p0") for t in ("2", "3")]
        cells.append(" / ".join("—" if g_["variance"] is None else f"{math.sqrt(g_['variance']):.1f}" for g_ in gp))
        cmp_ = ec.get("mean_payoff_vs_dp") or {}
        cells.append("—" if cmp_.get("z") is None else f"{cmp_['difference']:.2g} ({cmp_['z']:.2f})")
        rows.append(cells)
    L += md_table(["q", "seed", "group", "e1(0)", "e2(0)", "e3(0)", "mean E[e2]", "mean E[e3]", "stoch E[e2]",
                   "stoch E[e3]", "LF t2 (mean mode)", "LF t3 (mean mode)", "sd(gap) t2/t3", "MC U − DP V1 (z)"],
                   rows) + [""]

    L += ["## G. Economics by endpoint group (across seeds; rollout MCSE kept separate)", ""]
    rows = []
    for g in grp:
        if g["quantity"] in ("e1_at_0", "e2_at_0", "e3_at_0") or (g["quantity"] == "E_effort_representative") \
                or g["quantity"] == "U_mean_payoff":
            if g["n_runs"] == 0 and g["quantity"] != "e1_at_0":
                continue
            rows.append([g["q"], g["group"], g["n_runs"], g["mode"], g["stage"], g["quantity"],
                         f(g["mean_across_runs"]), f(g["sd_across_runs"]), f(g["min"]), f(g["max"]),
                         f(g["max_rollout_mcse"], 2), g["note"] or ""])
    L += md_table(["q", "group", "n runs", "mode", "stage", "quantity", "mean across runs", "SD across runs",
                   "min", "max", "max rollout MCSE", "note"], rows) + [""]

    # --- failure diagnosis
    L += ["## F. Failure diagnosis of no-candidate runs (minimum valid C development call; not the endpoint)", "",
          "Exposure = training visits in the 10-wide bin of the state, cumulative to that call: "
          "C direct / C continuation / A+B direct / A+B continuation.", ""]
    rows = []
    for r in runs:
        if r.get("search_outcome") != "no_candidate_budget_exhausted":
            continue
        d = fail.get(r["run_id"], {})
        c = ph.get((r["run_id"], "C"), {})
        cc = [num(x["dreach_over_dw"]) for x in calls if x["run_id"] == r["run_id"] and x["phase"] == "C"
              and x["valid"] == "true"]
        reach_in = "yes" if (d.get("min_C_max_all_state_stage") == d.get("min_C_max_reach_state_stage")
                             and d.get("min_C_max_all_state_d") == d.get("min_C_max_reach_state_d")) else "no"
        rows.append([
            r["q"], r["seed"], r["phase_B_exit_reason"] or "—",
            f"{f(d.get('min_valid_C_dreach_over_dw'))} ({f(d.get('min_C_global_update'))}/{f(d.get('min_C_local_update'))})",
            f(d.get("min_C_conc")),
            " / ".join(f(d.get(f"min_C_reach_contribution_over_dw_t{t}")) for t in ("1", "2", "3")),
            f"t{f(d.get('min_C_max_reach_state_stage'))} d={f(d.get('min_C_max_reach_state_d'))}: "
            f"{f(d.get('min_C_max_reach_state_delta_over_dw'))}",
            " / ".join(f(d.get(f"min_C_max_reach_state_exposure_{k}")) for k in
                       ("C_direct", "C_continuation", "AB_direct", "AB_continuation")),
            f"t{f(d.get('min_C_max_all_state_stage'))} d={f(d.get('min_C_max_all_state_d'))}: "
            f"{f(d.get('min_C_max_all_state_delta_over_dw'))} (same as reach: {reach_in})",
            " / ".join(f(d.get(f"min_C_max_all_state_exposure_{k}")) for k in
                       ("C_direct", "C_continuation", "AB_direct", "AB_continuation")),
            f"{f(c.get('criterion_min_over_dw'))} / {f(c.get('criterion_last_over_dw'))}",
            f"{sum(1 for v in cc if v is not None and v <= 0.02)}/{sum(1 for v in cc if v is not None and v <= 0.03)}/{len(cc)}",
            f(r["final_dreach_over_dw"]), f(r["final_delta_max_all_over_dw"])])
    L += md_table(["q", "seed", "B exit", "min dReach/DW (global/C local)", "conc at min",
                   "reach contrib t1/t2/t3", "max reach state", "its exposure", "max all-domain state",
                   "its exposure", "C dReach min / last", "C calls ≤0.02 / ≤0.03 / valid", "endpoint final dReach",
                   "endpoint final Delta_max_all"], rows) + [""]

    # --- completeness
    L += ["## C. Completeness (make_report.py --check-completeness)", ""]
    rows = []
    for c in comp["runs"]:
        rows.append([c["run_id"], c["state"], str(c["ok"]), "; ".join(c["failures"]) or "none",
                     "; ".join(c["notes"]) or "none",
                     "; ".join(json.dumps(u) for u in c["unavailable_checks"]) or "none"])
    L += md_table(["run", "state", "complete", "failures", "notes", "unavailable checks"], rows) + [""]
    (out_dir / "FORMAL_TABLES.md").write_text("\n".join(L) + "\n")
    print(f"[formal_summary] {len(runs)} runs -> {out_dir}/verification_by_seed.csv, economics_by_group.csv, "
          f"FORMAL_TABLES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
