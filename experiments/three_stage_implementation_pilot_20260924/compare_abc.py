#!/usr/bin/env python3
"""Paired full A/B/C comparison: baseline vs center-weighted phase-A sampling.

Reads saved run files of the two arm manifests. For each run it reports the
final outcome (candidate / certification), the minimum valid C development
call, actual check counts / updates / environment steps, and a retention
check at four checkpoints: A400, B exit, C minimum and the actual stop.
At every checkpoint the development verifier is replayed from the saved
weights (periodic weights, min_dev_weights.npz, endpoint_weights.npz); the
replay must reproduce the recorded call's dReach exactly, then stage-3 and
stage-2 policy-vs-BR shape and deviations are taken from it. The rules applied
are the ones pre-registered in ``resolved.study``.

    .venv/bin/python -B experiments/three_stage_implementation_pilot_20260924/compare_abc.py --cohort asamp_abc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import E, now_iso, read_jsonl, write_json_atomic  # noqa: E402
from make_report import write_csv  # noqa: E402
from metrics import diagnose_minimum_C, stage_region_metrics  # noqa: E402
from run_experiment import fresh_agent  # noqa: E402

from envs.curriculum_env import GameSpec  # noqa: E402
from run.run_final_dp_br import make_policy_fns, run_verifier  # noqa: E402
from utils.dp_br_verifier import VerifierConfig  # noqa: E402

ARMS = ("baseline", "center")
CHECKPOINTS = ("A400", "B_exit", "C_minimum", "actual_stop")


def load_weights(cfg: Dict[str, Any], npz: Path) -> Any:
    """Fresh agent with actor weights from an exported npz (torch forward, same as training)."""
    agent = fresh_agent(cfg)
    w = np.load(npz)
    agent.actor.load_state_dict({k[len("actor."):]: torch.as_tensor(w[k]) for k in w.files if k.startswith("actor.")})
    return agent


def checkpoint_metrics(cfg: Dict[str, Any], spec: GameSpec, npz: Path, call: Optional[Dict[str, Any]]
                       ) -> Dict[str, Any]:
    """Dev-verifier replay at saved weights; stage-3/stage-2 deviations and policy-vs-BR shape."""
    agent = load_weights(cfg, npz)
    mean_fn, beta_fn = make_policy_fns(agent, spec)
    res, err = run_verifier(mean_fn, beta_fn, spec, VerifierConfig(**cfg["verifier"]["development"]))
    out: Dict[str, Any] = {"weights": str(npz.relative_to(E)), "replay_error": err}
    if res is None:
        return out
    dw = spec.dw
    out["replay_dreach_over_dw"] = res.dreach / dw
    if call is not None and call.get("summary"):
        out["recorded_dreach_over_dw"] = call["summary"]["dreach_over_dw"]
        out["replay_matches_recorded"] = bool(res.dreach / dw == call["summary"]["dreach_over_dw"])
    s3, s2 = res.stages[3], res.stages[2]
    r3 = stage_region_metrics(s3.d_grid, s3.v_br - s3.v_mean, s3.std_norm, spec.B, dw)
    out.update({f"s3_{k}": r3[k] for k in ("all_max_over_dw", "all_argmax_d", "center_max_over_dw",
                                           "outer_max_over_dw", "center_mean_over_dw", "outer_mean_over_dw")})
    g2 = s2.v_br - s2.v_mean
    j = int(np.argmax(g2))
    out.update({"s2_cont_max_over_dw": float(g2[j]) / dw, "s2_cont_argmax_d": float(s2.d_grid[j]),
                "s2_delta_full_max_over_dw": float(s2.delta.max()) / dw,
                "s2_delta_reach_max_over_dw": float(s2.delta[s2.reach].max()) / dw,
                "dreach_over_dw": res.dreach / dw, "exp_root_over_dw": res.exp_root / dw,
                "reach_t1_over_dw": res.reach_delta_max[1] / dw, "reach_t2_over_dw": res.reach_delta_max[2] / dw,
                "reach_t3_over_dw": res.reach_delta_max[3] / dw})
    for t, s in ((3, s3), (2, s2)):
        act = s.a_br > 0
        contest = np.abs(s.d_grid) <= 2 * spec.q
        out[f"e{t}_at_0"] = float(s.e_hat[np.argmin(np.abs(s.d_grid))])
        out[f"e{t}_rise_within_2q"] = float(s.e_hat[contest].max() - s.e_hat[contest].min())
        out[f"br{t}_rise_within_2q"] = float(s.a_br[contest].max() - s.a_br[contest].min())
        out[f"e{t}_mean_abs_minus_br_within_2q"] = float(np.mean(np.abs(s.e_hat[contest] - s.a_br[contest])))
        out[f"s{t}_br_active_points"] = int(act.sum())
        out[f"arrays_t{t}"] = {"d": s.d_grid, "e_hat": s.e_hat, "a_br": s.a_br, "gain": (s.v_br - s.v_mean) / dw}
    return out


def run_summary(run: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Outcome, minimum diagnosis, counts and retention checkpoints for one run."""
    d = E / run["output_dir"]
    out: Dict[str, Any] = {"run_id": run["run_id"], "seed": run["seed"], "arm": run["arm"]}
    if not d.exists():
        out["state"] = "pending"
        return out
    spec = GameSpec(**cfg["game"])
    st = json.loads((d / "status.json").read_text())
    out["state"] = st["state"]
    ts = json.loads((d / "training_summary.json").read_text()) if (d / "training_summary.json").exists() else {}
    fe = json.loads((d / "final_eval.json").read_text()) if (d / "final_eval.json").exists() else {}
    ec = json.loads((d / "economics.json").read_text()) if (d / "economics.json").exists() else {}
    calls, _ = read_jsonl(d / "verifier_calls.jsonl")
    hist, _ = read_jsonl(d / "history.jsonl")
    phases = {p["phase"]: p for p in ts.get("phases", [])}
    out.update({
        "has_candidate": ts.get("has_candidate"), "candidate_global_update": ts.get("candidate_update"),
        "candidate_C_local": ts.get("candidate_C_local"), "search_outcome": ts.get("search_outcome"),
        "certification": fe.get("certification"), "final_joint_pass": fe.get("final_joint_pass"),
        "stop_global_update": ts.get("stop_global_update"), "stop_C_local_update": ts.get("stop_C_local_update"),
        "endpoint_kind": (ts.get("endpoint") or {}).get("identity", {}).get("checkpoint_kind"),
        "B_exit_reason": phases.get("B", {}).get("exit_reason"), "B_local_updates": phases.get("B", {}).get("local_updates"),
        "C_exit_reason": phases.get("C", {}).get("exit_reason"), "C_local_updates": phases.get("C", {}).get("local_updates"),
        "updates": len(hist), "episodes": sum(h["n_episodes"] for h in hist),
        "environment_steps": sum(h["n_environment_steps"] for h in hist),
        "physical_actions": sum(h["n_physical_actions"] for h in hist),
        "n_checks_A": sum(1 for c in calls if c["phase"] == "A"),
        "n_checks_B": sum(1 for c in calls if c["phase"] == "B"),
        "n_checks_C": sum(1 for c in calls if c["phase"] == "C"),
        "B_eligible_calls": sum(1 for c in calls if c["phase"] == "B" and c.get("eligible")),
    })
    if out["endpoint_kind"] == "candidate":
        out["stop_label"] = f"actual stop: candidate at C local {out['stop_C_local_update']} (global {out['stop_global_update']})"
    elif out["endpoint_kind"] == "diagnostic_terminal":
        out["stop_label"] = f"C cap terminal (C local {out['stop_C_local_update']}, global {out['stop_global_update']})"
    pf = fe.get("pass_flags") or {}
    for k in ("dreach_final_over_dw", "main_pass", "refine_dreach_diff_over_dw", "refine_dreach_pass",
              "refine_exp_diff_over_dw", "refine_exp_pass", "dense_conc_max_std_norm", "dense_conc_pass",
              "numeric_thresholds_pass"):
        out[f"final_{k}"] = pf.get(k)
    out["final_exp_root_over_dw"] = ((fe.get("tiers") or {}).get("final") or {}).get("exp_root_over_dw")
    out["economics_ok"] = bool(ec.get("modes")) and not ec.get("error")
    if ec.get("modes"):
        out["mean_selfplay_U"] = ec["modes"]["mean"]["pooled"]["episode"]["U"]["mean"]
        out["mc_minus_dp_over_mcse"] = (ec.get("mean_payoff_vs_dp") or {}).get("z")
    mr = json.loads((d / "checkpoints" / "min_dev_record.json").read_text()) \
        if (d / "checkpoints" / "min_dev_record.json").exists() else None
    dg = diagnose_minimum_C(calls, mr, any(h["phase"] == "C" for h in hist), spec.dw)
    for k in ("min_valid_C_dreach_over_dw", "min_C_global_update", "min_C_local_update", "min_C_conc",
              "min_C_conc_stage", "min_C_conc_d", "reason", "min_record_matches_selected_call",
              "contribution_sum_matches_dreach", "n_valid_C_calls"):
        out[k if k != "reason" else "min_reason"] = dg.get(k)
    for t in ("1", "2", "3"):
        out[f"min_C_reach_contribution_t{t}_over_dw"] = (dg.get("reach_contribution_over_dw") or {}).get(t)
        out[f"min_C_reach_argmax_d_t{t}"] = (dg.get("reach_argmax_d") or {}).get(t)
        out[f"min_C_full_contribution_t{t}_over_dw"] = (dg.get("full_contribution_over_dw") or {}).get(t)
    for key in ("max_reach_state", "max_all_state"):
        v = dg.get(key) or {}
        out[f"min_C_{key}"] = f"t{v.get('stage')} d={v.get('d')} {v.get('delta', 0) / spec.dw:.5f}" if v else None
    # retention checkpoints
    by_g = {c["global_update"]: c for c in calls}
    b_calls = [c for c in calls if c["phase"] == "B"]
    cps: Dict[str, Tuple[Path, Optional[Dict[str, Any]]]] = {}
    a400 = [c for c in calls if c["phase"] == "A" and c["local_update"] == 400] or \
        [c for c in calls if c["phase"] == "A"][-1:]
    if a400:
        cps["A400"] = (d / "weights" / f"update_{a400[0]['global_update']:05d}.npz", a400[0])
    if b_calls:
        cps["B_exit"] = (d / "weights" / f"update_{b_calls[-1]['global_update']:05d}.npz", b_calls[-1])
    if mr is not None:
        cps["C_minimum"] = (d / "checkpoints" / "min_dev_weights.npz", by_g.get(mr["identity"]["global_update"]))
    if (d / "checkpoints" / "endpoint_weights.npz").exists():
        cps["actual_stop"] = (d / "checkpoints" / "endpoint_weights.npz", by_g.get(out["stop_global_update"]))
    out["checkpoints"] = {}
    for name, (npz, call) in cps.items():
        if not npz.exists():
            out["checkpoints"][name] = {"missing": str(npz)}
            continue
        m = checkpoint_metrics(cfg, spec, npz, call)
        m["global_update"] = call["global_update"] if call else None
        out["checkpoints"][name] = m
    return out


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort", default="asamp_abc", choices=["asamp_abc", "asamp_abc_smoke"])
    p.add_argument("--q", type=int, default=60)
    args = p.parse_args()
    torch.set_num_threads(1)
    mans = {arm: json.loads((E / "manifests" / f"{args.cohort}_q{args.q}_{arm}.json").read_text()) for arm in ARMS}
    assert mans["baseline"]["seeds"] == mans["center"]["seeds"], "arms must share seeds"
    study = mans["center"]["resolved"]["study"]
    seeds = mans["center"]["seeds"]
    spec = GameSpec(**mans["center"]["resolved"]["game"])
    out_dir = E / "reports" / f"{args.cohort}_q{args.q}_comparison"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    R = {arm: {r["seed"]: run_summary(r, mans[arm]["resolved"]) for r in mans[arm]["runs"]} for arm in ARMS}
    scalar = lambda m: {k: v for k, v in m.items() if k != "checkpoints"}
    write_csv(out_dir / "runs.csv", [scalar(R[a][s]) for s in seeds for a in ARMS])
    cp_rows = []
    for s in seeds:
        for a in ARMS:
            for name in CHECKPOINTS:
                m = (R[a][s].get("checkpoints") or {}).get(name)
                if m is None:
                    continue
                cp_rows.append({"seed": s, "arm": a, "checkpoint": name,
                                **{k: v for k, v in m.items() if not k.startswith("arrays_")}})
    write_csv(out_dir / "checkpoints.csv", cp_rows)
    # counts per arm (diagnostic only)
    counts = {}
    for a in ARMS:
        rs = [R[a][s] for s in seeds]
        n = len(rs)
        cand = sum(1 for r in rs if r.get("has_candidate"))
        cert = sum(1 for r in rs if r.get("certification") == "certified")
        counts[a] = {"n_runs": n, "n_done": sum(1 for r in rs if r.get("state") == "done"),
                     "candidate": f"{cand}/{n}", "certified": f"{cert}/{n}",
                     "certified_given_candidate": f"{cert}/{cand}" if cand else "N/A (0 candidates)",
                     "n_candidate": cand, "n_certified": cert}
    # paired comparisons
    pairs = []
    for s in seeds:
        b, c = R["baseline"][s], R["center"][s]
        row: Dict[str, Any] = {"seed": s}
        for k in ("has_candidate", "certification", "stop_label", "B_exit_reason", "B_local_updates",
                  "min_valid_C_dreach_over_dw", "min_C_global_update", "final_dreach_final_over_dw",
                  "updates", "environment_steps", "n_checks_B", "n_checks_C"):
            row[f"baseline_{k}"], row[f"center_{k}"] = b.get(k), c.get(k)
        mb, mc = b.get("min_valid_C_dreach_over_dw"), c.get("min_valid_C_dreach_over_dw")
        row["diff_min_valid_C_dreach_over_dw"] = (mc - mb) if mb is not None and mc is not None else None
        for name in CHECKPOINTS:
            for k in ("s3_all_max_over_dw", "s2_cont_max_over_dw", "s2_delta_full_max_over_dw", "e3_rise_within_2q"):
                vb = ((b.get("checkpoints") or {}).get(name) or {}).get(k)
                vc = ((c.get("checkpoints") or {}).get(name) or {}).get(k)
                row[f"{name}_{k}_baseline"], row[f"{name}_{k}_center"] = vb, vc
                row[f"{name}_{k}_diff"] = (vc - vb) if vb is not None and vc is not None else None
        pairs.append(row)
    write_csv(out_dir / "pairs.csv", pairs)
    retained = {name: sum(1 for r in pairs if (r.get(f"{name}_s3_all_max_over_dw_diff") or 0) < 0) for name in CHECKPOINTS}
    s2_lower = {name: sum(1 for r in pairs if (r.get(f"{name}_s2_cont_max_over_dw_diff") or 0) < 0) for name in CHECKPOINTS}
    n = len(seeds)
    min_lower = sum(1 for r in pairs if (r.get("diff_min_valid_C_dreach_over_dw") or 0) < 0)
    disappeared = retained.get("C_minimum", 0) < n and retained.get("actual_stop", 0) < n
    cand_gain = counts["center"]["n_candidate"] > counts["baseline"]["n_candidate"]
    cert_gain = counts["center"]["n_certified"] > counts["baseline"]["n_certified"]
    no_cand = counts["center"]["n_candidate"] == 0 and counts["baseline"]["n_candidate"] == 0
    replay_ok = all(m.get("replay_matches_recorded", True) for s in seeds for a in ARMS
                    for m in (R[a][s].get("checkpoints") or {}).values())
    decision = {
        "cohort": args.cohort, "q": args.q, "generated": now_iso(), "study": study, "counts": counts,
        "pairs_center_lower_stage3_full_max": {k: f"{v}/{n}" for k, v in retained.items()},
        "pairs_center_lower_stage2_cont_max": {k: f"{v}/{n}" for k, v in s2_lower.items()},
        "pairs_center_lower_min_valid_C_dreach": f"{min_lower}/{n}",
        "A_advantage_retained_at": [k for k, v in retained.items() if v == n],
        "A_advantage_disappeared_in_BC": bool(disappeared),
        "candidate_gain": bool(cand_gain), "certification_gain": bool(cert_gain),
        "rule_stop_sampling_change": bool(disappeared and not cand_gain and not cert_gain),
        "rule_diagnostic_improvement_only": bool(no_cand and min_lower == n),
        "checkpoint_replays_match_recorded_calls": bool(replay_ok),
        "all_runs_done": all(R[a][s].get("state") == "done" for a in ARMS for s in seeds),
    }
    write_json_atomic(out_dir / "decision.json", decision)
    figures(out_dir, R, seeds, spec)
    write_md(out_dir, decision, R, seeds, pairs)
    print(json.dumps({k: v for k, v in decision.items() if k != "study"}, indent=1))
    return 0


def figures(out_dir: Path, R: Dict[str, Dict[int, Dict[str, Any]]], seeds: List[int], spec: GameSpec) -> None:
    """Policy vs BR at four checkpoints (stages 3 and 2) and retention summaries."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"baseline": "tab:blue", "center": "tab:orange"}
    done = [s for s in seeds if all(R[a][s].get("state") == "done" for a in ARMS)]
    for t in (3, 2):
        if not done:
            break
        fig, axes = plt.subplots(len(done), len(CHECKPOINTS), figsize=(16, 2.9 * len(done)), squeeze=False)
        rows = []
        for i, s in enumerate(done):
            for j, name in enumerate(CHECKPOINTS):
                ax = axes[i][j]
                for a in ARMS:
                    m = (R[a][s].get("checkpoints") or {}).get(name) or {}
                    arr = m.get(f"arrays_t{t}")
                    if arr is None:
                        continue
                    ax.plot(arr["d"], arr["e_hat"], color=col[a], lw=1.0, label=f"{a} policy")
                    ax.plot(arr["d"], arr["a_br"], color=col[a], lw=0.8, ls="--", label=f"{a} BR")
                    rows += [{"seed": s, "arm": a, "checkpoint": name, "stage": t, "d": float(x), "e_hat": float(e),
                              "a_br": float(b), "gain_over_dw": float(g)}
                             for x, e, b, g in zip(arr["d"], arr["e_hat"], arr["a_br"], arr["gain"])]
                ax.set_title(f"s{s} {name}: stage {t}", fontsize=8)
                ax.set_xlabel("d")
                ax.set_ylabel("effort")
                if i == 0 and j == 0:
                    ax.legend(fontsize=6)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / "figures" / f"stage{t}_policy_vs_br_checkpoints.{ext}", dpi=120)
        plt.close(fig)
        write_csv(out_dir / "figures" / f"stage{t}_policy_vs_br_checkpoints.csv", rows)
    if done:
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
        x = np.arange(len(CHECKPOINTS))
        rows = []
        for s, ls in zip(done, ("-", "--", ":")):
            for a in ARMS:
                cp = R[a][s].get("checkpoints") or {}
                for ax, k in zip(axes, ("s3_all_max_over_dw", "s2_cont_max_over_dw", "dreach_over_dw")):
                    y = [(cp.get(n) or {}).get(k) for n in CHECKPOINTS]
                    ax.plot(x, [np.nan if v is None else v for v in y], ls, marker="o", color=col[a], label=f"{a} s{s}")
                    rows += [{"seed": s, "arm": a, "metric": k, "checkpoint": n, "value": v} for n, v in zip(CHECKPOINTS, y)]
        for ax, k in zip(axes, ("stage-3 full max gain / DW", "stage-2 max continuation gain / DW", "dReach / DW")):
            ax.set_xticks(x)
            ax.set_xticklabels(CHECKPOINTS, fontsize=8)
            ax.set_title(f"q={spec.q:g}: {k} (dev tier replay)")
        axes[0].legend(fontsize=6)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / "figures" / f"retention_summary.{ext}", dpi=120)
        plt.close(fig)
        write_csv(out_dir / "figures" / "retention_summary.csv", rows)


def write_md(out_dir: Path, decision: Dict[str, Any], R: Dict[str, Dict[int, Dict[str, Any]]], seeds: List[int],
             pairs: List[Dict[str, Any]]) -> None:
    """Machine-written tables; interpretation is in the hand-written report."""
    def f(v: Any, nd: int = 4) -> str:
        return "—" if v is None else (f"{v:.{nd}g}" if isinstance(v, float) else str(v))

    L = [f"# Paired A/B/C comparison — {decision['cohort']} q{decision['q']}", "",
         f"Generated {decision['generated']}. Pre-registered rules: `resolved.study` in the arm manifests "
         "(also copied into decision.json).", "", "## Counts per arm (3 seeds: diagnostic only)", "",
         "| arm | runs done | candidate / all | certified / all | certified / candidates |", "|---|---|---|---|---|"]
    for a, c in decision["counts"].items():
        L.append(f"| {a} | {c['n_done']}/{c['n_runs']} | {c['candidate']} | {c['certified']} | {c['certified_given_candidate']} |")
    L += ["", "## Rule application", "",
          f"- pairs with center lower, stage-3 full max (dev replay): {decision['pairs_center_lower_stage3_full_max']}",
          f"- pairs with center lower, stage-2 max continuation gain: {decision['pairs_center_lower_stage2_cont_max']}",
          f"- pairs with center lower min valid C dReach/DW: {decision['pairs_center_lower_min_valid_C_dreach']}",
          f"- A advantage retained at: {decision['A_advantage_retained_at'] or 'none'}; disappeared in B/C: "
          f"{decision['A_advantage_disappeared_in_BC']}",
          f"- candidate gain: {decision['candidate_gain']}; certification gain: {decision['certification_gain']}",
          f"- rule 'stop this sampling change': **{decision['rule_stop_sampling_change']}**",
          f"- rule 'diagnostic improvement only': **{decision['rule_diagnostic_improvement_only']}**",
          f"- checkpoint replays reproduce recorded calls: {decision['checkpoint_replays_match_recorded_calls']}", "",
          "## Per run", "",
          "| seed | arm | outcome / stop | certification | B exit (local, eligible calls) | checks A/B/C | updates | env steps | "
          "min valid C dReach/DW (u, C local) | contributions t1/t2/t3 | max reach state | conc at min | final dReach/DW |",
          "|" + "---|" * 13]
    for s in seeds:
        for a in ARMS:
            r = R[a][s]
            if r.get("state") != "done":
                L.append(f"| {s} | {a} | {r.get('state')} |" + " |" * 10)
                continue
            L.append(f"| {s} | {a} | {r.get('stop_label')} | {r.get('certification')} | {r.get('B_exit_reason')} "
                     f"({r.get('B_local_updates')}, {r.get('B_eligible_calls')}) | {r['n_checks_A']}/{r['n_checks_B']}/"
                     f"{r['n_checks_C']} | {r['updates']} | {r['environment_steps']} | "
                     f"{f(r.get('min_valid_C_dreach_over_dw'))} ({r.get('min_C_global_update')}, {r.get('min_C_local_update')}) | "
                     f"{f(r.get('min_C_reach_contribution_t1_over_dw'))}/{f(r.get('min_C_reach_contribution_t2_over_dw'))}/"
                     f"{f(r.get('min_C_reach_contribution_t3_over_dw'))} | {r.get('min_C_max_reach_state')} | "
                     f"{f(r.get('min_C_conc'))} | {f(r.get('final_dreach_final_over_dw'))} |")
    L += ["", "## Retention checkpoints (dev-tier replay from saved weights)", "",
          "| seed | arm | checkpoint | global u | stage-3 full max (d) | center / outer max | stage-2 cont max (d) | "
          "stage-2 one-step max (all / reach) | dReach | e3(0) / rise 2q (BR rise) | e2(0) / rise 2q (BR rise) | "
          "|e3−BR| / |e2−BR| mean within 2q |", "|" + "---|" * 12]
    for s in seeds:
        for a in ARMS:
            for name in CHECKPOINTS:
                m = (R[a][s].get("checkpoints") or {}).get(name)
                if not m or "s3_all_max_over_dw" not in m:
                    continue
                L.append(f"| {s} | {a} | {name} | {m.get('global_update')} | {f(m['s3_all_max_over_dw'])} ({f(m['s3_all_argmax_d'])}) | "
                         f"{f(m['s3_center_max_over_dw'])} / {f(m['s3_outer_max_over_dw'])} | "
                         f"{f(m['s2_cont_max_over_dw'])} ({f(m['s2_cont_argmax_d'])}) | "
                         f"{f(m['s2_delta_full_max_over_dw'])} / {f(m['s2_delta_reach_max_over_dw'])} | {f(m['dreach_over_dw'])} | "
                         f"{f(m['e3_at_0'])} / {f(m['e3_rise_within_2q'])} ({f(m['br3_rise_within_2q'])}) | "
                         f"{f(m['e2_at_0'])} / {f(m['e2_rise_within_2q'])} ({f(m['br2_rise_within_2q'])}) | "
                         f"{f(m['e3_mean_abs_minus_br_within_2q'])} / {f(m['e2_mean_abs_minus_br_within_2q'])} |")
    L += ["", "Files: runs.csv, pairs.csv, checkpoints.csv, decision.json, figures/ "
          "(stage3/stage2_policy_vs_br_checkpoints, retention_summary). Full per-run tables: reports/<cohort>/ (make_report)."]
    (out_dir / "COMPARISON.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
