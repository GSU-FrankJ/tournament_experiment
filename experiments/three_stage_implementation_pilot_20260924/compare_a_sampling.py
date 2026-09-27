#!/usr/bin/env python3
"""Paired comparison for the Phase-A state-sampling study (baseline vs center-weighted ES3).

Reads only saved run files (final_eval.json, arrays.npz, verifier_calls.jsonl,
coverage.npz, history.jsonl) of the two arm manifests and applies the
pre-registered rule stored in the manifests' ``resolved.study`` block:
primary = final-tier max_{d in D3}(V3_BR - V3_mean)/DW at A400; consistent
improvement iff the center arm is lower in every seed pair.

    .venv/bin/python -B experiments/three_stage_implementation_pilot_20260924/compare_a_sampling.py --cohort asamp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import E, now_iso, read_jsonl, write_json_atomic  # noqa: E402
from make_report import write_csv  # noqa: E402
from metrics import center_bin_mask, stage_region_metrics  # noqa: E402

from agents.ppo_curriculum import mean_effort_numpy  # noqa: E402
from envs.curriculum_env import GameSpec  # noqa: E402

ARMS = ("baseline", "center")
REGIONS = ("all", "center", "outer")


def run_metrics(run: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """All comparison quantities of one run (None + reason when missing)."""
    d = E / run["output_dir"]
    out: Dict[str, Any] = {"run_id": run["run_id"], "seed": run["seed"], "arm": run["arm"]}
    if not d.exists():
        out["state"] = "pending"
        return out
    st = json.loads((d / "status.json").read_text())
    out["state"] = st["state"]
    fe_path = d / "final_eval.json"
    if st["state"] != "done" or not fe_path.exists():
        out["reason"] = st.get("failure_reason") or "final_eval missing"
        return out
    spec = GameSpec(**cfg["game"])
    fe = json.loads(fe_path.read_text())
    out["checkpoint_global_update"] = fe["checkpoint_identity"]["global_update"]
    for tier in ("final", "development"):
        r = fe["tiers"][tier].get("stage3_regions") or {}
        out[f"{tier}_valid"] = fe["tiers"][tier].get("valid")
        for reg in REGIONS:
            for k in ("max_over_dw", "argmax_d", "mean_over_dw"):
                out[f"{tier}_{reg}_{k}"] = r.get(f"{reg}_{k}")
    dense = fe.get("dense_stage3_conc_regions") or {}
    for reg in REGIONS:
        out[f"dense_conc_{reg}_max_std_norm"] = dense.get(f"{reg}_max_std_norm")
        out[f"dense_conc_{reg}_argmax_d"] = dense.get(f"{reg}_max_std_norm_d")
    calls, _ = read_jsonl(d / "verifier_calls.jsonl")
    a400 = [c for c in calls if c["phase"] == "A" and c["global_update"] == out["checkpoint_global_update"]]
    if a400:
        c = a400[-1]
        out["dev_call_conc_stage3_max_std_norm"] = (c.get("concentration") or {}).get("max_std_norm")
        out["dev_call_conc_stage3_argmax_d"] = (c.get("concentration") or {}).get("d")
        out["dev_call_conc_pass_0p04"] = c.get("conc_pass")
    out["trajectory"] = [{"local_update": c["local_update"], "global_update": c["global_update"],
                          **{f"{reg}_max_over_dw": (c.get("stage3_regions") or {}).get(f"{reg}_max_over_dw")
                             for reg in REGIONS},
                          **{f"{reg}_mean_over_dw": (c.get("stage3_regions") or {}).get(f"{reg}_mean_over_dw")
                             for reg in REGIONS},
                          "conc_stage3_max_std_norm": (c.get("concentration") or {}).get("max_std_norm")}
                         for c in calls if c["phase"] == "A"]
    cov = np.load(d / "coverage.npz")
    layout = json.loads(str(cov["layout_json"]))
    seg = next(s for s in layout if s["kind"] == "start_learner" and s["start_stage"] == 3)
    counts = cov["phase_totals"][0][seg["offset"]:seg["offset"] + seg["nbins"]]
    cm = center_bin_mask(spec, 3, float(cov["bin_width"]), spec.B)
    out["exposure_center_total"] = int(counts[cm].sum())
    out["exposure_outer_total"] = int(counts[~cm].sum())
    out["exposure_center_share"] = counts[cm].sum() / counts.sum()
    out["exposure_center_bin_min"] = int(counts[cm].min())
    out["exposure_outer_bin_min"] = int(counts[~cm].min())
    out["exposure_counts"] = counts
    hist, _ = read_jsonl(d / "history.jsonl")
    out["updates"] = len(hist)
    tail = hist[-25:]
    out["train_mean_episode_return_last25"] = float(np.mean([h["mean_episode_return"] for h in tail]))
    out["train_kl_last25"] = float(np.mean([h["kl_final_epoch"] for h in tail]))
    A = np.load(d / "arrays.npz")
    out["arrays"] = {k: A[f"final_t3_{k}"] for k in ("d_grid", "v_br", "v_mean", "e_hat", "a_br", "std_norm")}
    # recompute the primary from the saved arrays (same definition as the stored summary)
    re = stage_region_metrics(A["final_t3_d_grid"], A["final_t3_v_br"] - A["final_t3_v_mean"], None, spec.B, spec.dw)
    out["primary_recomputed_matches"] = bool(re["all_max_over_dw"] == out["final_all_max_over_dw"])
    out.update(policy_shape_at_endpoint(A, spec))
    out["shape_trajectory"] = shape_trajectory(d, spec, [100, 200, 300, 400])
    return out


def policy_shape_at_endpoint(A: Any, spec: GameSpec) -> Dict[str, Any]:
    """Stage-3 policy vs best response on the BR-active set {d: a_BR(d) > 0} (final tier)."""
    dg, e, br = A["final_t3_d_grid"], A["final_t3_e_hat"], A["final_t3_a_br"]
    act = br > 0
    g = A["final_t3_v_br"] - A["final_t3_v_mean"]
    j = int(np.argmax(g))
    return {"br_active_points": int(act.sum()),
            "br_active_d_min": float(dg[act].min()) if act.any() else None,
            "br_active_d_max": float(dg[act].max()) if act.any() else None,
            "policy_range_on_br_active": float(e[act].max() - e[act].min()) if act.any() else None,
            "br_range_on_br_active": float(br[act].max() - br[act].min()) if act.any() else None,
            "mean_abs_policy_minus_br_on_br_active": float(np.mean(np.abs(e[act] - br[act]))) if act.any() else None,
            "policy_e3_at_0": float(e[np.argmin(np.abs(dg))]),
            "policy_e3_at_primary_argmax": float(e[j]), "br_at_primary_argmax": float(br[j]),
            "policy_e3_min_D3": float(e.min()), "policy_e3_max_D3": float(e.max())}


def shape_trajectory(run_dir: Path, spec: GameSpec, updates: List[int]) -> List[Dict[str, Any]]:
    """Stage-3 policy e3(0) and its rise within |d| <= 2q from saved periodic weights (framework-free)."""
    d = np.linspace(-spec.domain_half(3), spec.domain_half(3), 441)
    obs = spec.encode_obs(3, d)
    contest = np.abs(d) <= 2 * spec.q
    out = []
    for u in updates:
        w = run_dir / "weights" / f"update_{u:05d}.npz"
        if not w.exists():
            continue
        e, _, _ = mean_effort_numpy(dict(np.load(w)), obs, e_min=spec.e_min, e_max=spec.e_max)
        out.append({"global_update": u, "e3_at_0": float(e[np.argmin(np.abs(d))]),
                    "e3_rise_within_2q": float(e[contest].max() - e[contest].min()),
                    "e3_min_D3": float(e.min()), "e3_max_D3": float(e.max())})
    return out


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort", default="asamp", choices=["asamp", "asamp_smoke"])
    p.add_argument("--q", type=int, default=60)
    args = p.parse_args()
    mans = {arm: json.loads((E / "manifests" / f"{args.cohort}_q{args.q}_{arm}.json").read_text()) for arm in ARMS}
    study = mans["center"]["resolved"]["study"]
    assert mans["baseline"]["seeds"] == mans["center"]["seeds"], "arms must share seeds"
    cfg = mans["center"]["resolved"]
    spec = GameSpec(**cfg["game"])
    out_dir = E / "reports" / f"{args.cohort}_q{args.q}_comparison"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    R: Dict[str, Dict[int, Dict[str, Any]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for run in mans[arm]["runs"]:
            R[arm][run["seed"]] = run_metrics(run, mans[arm]["resolved"])
    seeds = mans["center"]["seeds"]
    scalar = lambda m: {k: v for k, v in m.items() if k not in ("trajectory", "arrays", "exposure_counts",
                                                                "shape_trajectory")}
    run_rows = [scalar(R[arm][s]) for s in seeds for arm in ARMS]
    write_csv(out_dir / "runs.csv", run_rows)
    traj_rows = [{"run_id": R[arm][s]["run_id"], "seed": s, "arm": arm, **t}
                 for s in seeds for arm in ARMS for t in R[arm][s].get("trajectory", [])]
    write_csv(out_dir / "trajectory.csv", traj_rows)
    shape_rows = [{"run_id": R[arm][s]["run_id"], "seed": s, "arm": arm, **t}
                  for s in seeds for arm in ARMS for t in R[arm][s].get("shape_trajectory", [])]
    ref = []
    for ps in (10401, 10402, 10403):          # pilot q60 runs: same code as the baseline arm (other seeds)
        pdir = E / "runs" / f"t3_pilot_q{args.q}_s{ps}"
        if pdir.exists():
            ref += [{"run_id": pdir.name, "seed": ps, "arm": "pilot_reference", **t}
                    for t in shape_trajectory(pdir, spec, [100, 200, 300, 400, 1000, 2800])]
    write_csv(out_dir / "policy_shape_trajectory.csv", shape_rows + ref)
    keys = ([f"final_{r}_max_over_dw" for r in REGIONS] + [f"final_{r}_mean_over_dw" for r in REGIONS]
            + [f"development_{r}_max_over_dw" for r in REGIONS]
            + ["dev_call_conc_stage3_max_std_norm"] + [f"dense_conc_{r}_max_std_norm" for r in REGIONS]
            + ["exposure_center_share", "train_mean_episode_return_last25"])
    pair_rows = []
    complete = True
    for s in seeds:
        b, c = R["baseline"][s], R["center"][s]
        row: Dict[str, Any] = {"seed": s, "baseline_state": b.get("state"), "center_state": c.get("state")}
        if b.get("state") != "done" or c.get("state") != "done":
            complete = False
            row["reason"] = "pair incomplete"
            pair_rows.append(row)
            continue
        for k in keys:
            row[f"baseline_{k}"], row[f"center_{k}"] = b.get(k), c.get(k)
            row[f"diff_{k}"] = (c[k] - b[k]) if b.get(k) is not None and c.get(k) is not None else None
        row["baseline_final_all_argmax_d"] = b.get("final_all_argmax_d")
        row["center_final_all_argmax_d"] = c.get("final_all_argmax_d")
        row["primary_improved"] = row["diff_final_all_max_over_dw"] < 0
        row["conc_flag_center_exceeds_where_baseline_ok"] = bool(
            (c.get("dev_call_conc_stage3_max_std_norm") or 0) > 0.04 >= (b.get("dev_call_conc_stage3_max_std_norm") or 0))
        pair_rows.append(row)
    write_csv(out_dir / "pairs.csv", pair_rows)
    n_imp = sum(1 for r in pair_rows if r.get("primary_improved"))
    decision = {
        "cohort": args.cohort, "q": args.q, "generated": now_iso(), "study": study,
        "pairs_planned": len(seeds), "pairs_complete": sum(1 for r in pair_rows if "reason" not in r),
        "pairs_primary_improved": n_imp,
        "consistent_improvement": bool(complete and n_imp == len(seeds)),
        "conc_flags": [r["seed"] for r in pair_rows if r.get("conc_flag_center_exceeds_where_baseline_ok")],
        "primary_recomputed_from_arrays": all(R[a][s].get("primary_recomputed_matches") for a in ARMS for s in seeds
                                              if R[a][s].get("state") == "done"),
    }
    write_json_atomic(out_dir / "decision.json", decision)
    figures(out_dir, R, seeds, spec)
    write_md(out_dir, decision, pair_rows, R, seeds, shape_rows + ref)
    print(json.dumps({k: v for k, v in decision.items() if k != "study"}, indent=1))
    return 0


def figures(out_dir: Path, R: Dict[str, Dict[int, Dict[str, Any]]], seeds: List[int], spec: GameSpec) -> None:
    """Gain profiles, policy vs BR, trajectories and exposure per arm."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"baseline": "tab:blue", "center": "tab:orange"}
    done = [s for s in seeds if all(R[a][s].get("state") == "done" for a in ARMS)]
    if not done:
        return
    fig, axes = plt.subplots(len(done), 2, figsize=(12, 3.0 * len(done)), squeeze=False)
    data = []
    for i, s in enumerate(done):
        for arm in ARMS:
            A = R[arm][s]["arrays"]
            g = (A["v_br"] - A["v_mean"]) / spec.dw
            axes[i][0].plot(A["d_grid"], g, color=col[arm], lw=0.9, label=arm)
            axes[i][1].plot(A["d_grid"], A["e_hat"], color=col[arm], lw=0.9, label=f"{arm} policy mean")
            axes[i][1].plot(A["d_grid"], A["a_br"], color=col[arm], lw=0.8, ls="--", label=f"{arm} BR")
            data += [{"seed": s, "arm": arm, "d": float(x), "gain_over_dw": float(y), "e_hat": float(e),
                      "a_br": float(a)} for x, y, e, a in zip(A["d_grid"], g, A["e_hat"], A["a_br"])]
        for ax in axes[i]:
            ax.axvspan(-spec.B, spec.B, color="0.93", zorder=0)
            ax.set_xlabel("stage-3 signed gap d (shaded: |d| <= B)")
        axes[i][0].set_ylabel("(V3_BR - V3_mean)/DW")
        axes[i][0].set_title(f"q={spec.q:g} s{s}: stage-3 gain at A400 (final tier)")
        axes[i][1].set_ylabel("effort")
        axes[i][1].set_title(f"q={spec.q:g} s{s}: policy mean vs best response")
        axes[i][0].legend(fontsize=7)
        axes[i][1].legend(fontsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / "figures" / f"stage3_gain_and_policy.{ext}", dpi=130)
    plt.close(fig)
    write_csv(out_dir / "figures" / "stage3_gain_and_policy.csv", data)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for s, ls in zip(done, ("-", "--", ":")):
        for arm in ARMS:
            tr = R[arm][s]["trajectory"]
            for ax, reg in zip(axes, REGIONS):
                ax.plot([t["local_update"] for t in tr], [t[f"{reg}_max_over_dw"] for t in tr], ls, marker=".",
                        color=col[arm], label=f"{arm} s{s}")
    for ax, reg in zip(axes, REGIONS):
        ax.set_title(f"dev tier: {reg} max stage-3 gain / DW")
        ax.set_xlabel("A local update")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / "figures" / f"trajectory.{ext}", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    edges = np.linspace(-spec.domain_half(3), spec.domain_half(3), R["baseline"][done[0]]["exposure_counts"].size + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for s, mk in zip(done, ("o", "s", "^")):
        for arm in ARMS:
            cnt = R[arm][s]["exposure_counts"]
            ax.plot(mid, cnt, mk, ms=3, color=col[arm], label=f"{arm} s{s}")
            rows += [{"seed": s, "arm": arm, "bin_midpoint": float(m), "count": int(c)} for m, c in zip(mid, cnt)]
    ax.set_xlabel("stage-3 learner-signed start gap (bin midpoint)")
    ax.set_ylabel("phase-A starts per bin")
    ax.set_title(f"q={spec.q:g}: actual A exposure by arm")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / "figures" / f"exposure.{ext}", dpi=130)
    plt.close(fig)
    write_csv(out_dir / "figures" / "exposure.csv", rows)


def write_md(out_dir: Path, decision: Dict[str, Any], pair_rows: List[Dict[str, Any]],
             R: Dict[str, Dict[int, Dict[str, Any]]], seeds: List[int], shape_rows: List[Dict[str, Any]]) -> None:
    """Machine-written comparison tables (interpretation lives in the hand-written report)."""
    def f(v: Any, nd: int = 4) -> str:
        return "—" if v is None else (f"{v:.{nd}g}" if isinstance(v, float) else str(v))

    L = [f"# A-sampling comparison — {decision['cohort']} q{decision['q']}", "",
         f"Generated {decision['generated']}. Rule (pre-registered in the manifests): {decision['study']['decision_rule']}", "",
         f"- pairs complete: {decision['pairs_complete']}/{decision['pairs_planned']}",
         f"- primary improved (center < baseline): {decision['pairs_primary_improved']}/{decision['pairs_planned']}",
         f"- consistent improvement: **{decision['consistent_improvement']}**",
         f"- concentration flags: {decision['conc_flags'] or 'none'}", "",
         "## Primary and region metrics at A400 (final tier, gain/DW)", "",
         "| seed | arm | all max (d) | center max (d) | outer max (d) | center mean | outer mean | dev-tier all max | "
         "conc dev-grid A400 | dense conc center / outer | center share | train return (last 25) |",
         "|" + "---|" * 12]
    for s in seeds:
        for arm in ARMS:
            m = R[arm][s]
            if m.get("state") != "done":
                L.append(f"| {s} | {arm} | {m.get('state')} |" + " |" * 10)
                continue
            L.append(f"| {s} | {arm} | {f(m['final_all_max_over_dw'])} ({f(m['final_all_argmax_d'])}) | "
                     f"{f(m['final_center_max_over_dw'])} ({f(m['final_center_argmax_d'])}) | "
                     f"{f(m['final_outer_max_over_dw'])} ({f(m['final_outer_argmax_d'])}) | "
                     f"{f(m['final_center_mean_over_dw'])} | {f(m['final_outer_mean_over_dw'])} | "
                     f"{f(m['development_all_max_over_dw'])} | {f(m.get('dev_call_conc_stage3_max_std_norm'))} | "
                     f"{f(m['dense_conc_center_max_std_norm'])} / {f(m['dense_conc_outer_max_std_norm'])} | "
                     f"{f(m['exposure_center_share'])} | {f(m['train_mean_episode_return_last25'])} |")
    L += ["", "## Paired differences (center − baseline)", "",
          "| seed | Δ all max | Δ center max | Δ outer max | Δ center mean | Δ outer mean | Δ dev conc | improved |",
          "|" + "---|" * 8]
    for r in pair_rows:
        if "reason" in r:
            L.append(f"| {r['seed']} | {r['reason']} |" + " |" * 6)
            continue
        L.append(f"| {r['seed']} | {f(r['diff_final_all_max_over_dw'])} | {f(r['diff_final_center_max_over_dw'])} | "
                 f"{f(r['diff_final_outer_max_over_dw'])} | {f(r['diff_final_center_mean_over_dw'])} | "
                 f"{f(r['diff_final_outer_mean_over_dw'])} | {f(r['diff_dev_call_conc_stage3_max_std_norm'])} | "
                 f"{r['primary_improved']} |")
    L += ["", "## A100–A400 trajectory (dev tier, gain/DW)", "", "| seed | arm | local | all max | center max | outer max | conc |",
          "|" + "---|" * 7]
    for s in seeds:
        for arm in ARMS:
            for t in R[arm][s].get("trajectory", []):
                L.append(f"| {s} | {arm} | {t['local_update']} | {f(t['all_max_over_dw'])} | {f(t['center_max_over_dw'])} | "
                         f"{f(t['outer_max_over_dw'])} | {f(t['conc_stage3_max_std_norm'])} |")
    L += ["", "## Stage-3 policy shape at A400 (final tier; BR-active set = {d: a_BR(d) > 0})", "",
          "| seed | arm | BR-active d range | policy range there | BR range there | mean abs(policy − BR) there | "
          "e3(0) | e3 / BR at primary argmax | e3 min/max on D3 |", "|" + "---|" * 9]
    for s in seeds:
        for arm in ARMS:
            m = R[arm][s]
            if m.get("state") != "done":
                continue
            L.append(f"| {s} | {arm} | [{f(m['br_active_d_min'])}, {f(m['br_active_d_max'])}] | "
                     f"{f(m['policy_range_on_br_active'])} | {f(m['br_range_on_br_active'])} | "
                     f"{f(m['mean_abs_policy_minus_br_on_br_active'])} | {f(m['policy_e3_at_0'])} | "
                     f"{f(m['policy_e3_at_primary_argmax'])} / {f(m['br_at_primary_argmax'])} | "
                     f"{f(m['policy_e3_min_D3'])} / {f(m['policy_e3_max_D3'])} |")
    L += ["", "## Stage-3 policy shape along training (saved periodic weights; rise = max − min of e3 on |d| <= 2q)", "",
          "| run | update | e3(0) | rise within 2q | e3 min/max on D3 |", "|" + "---|" * 5]
    for r in shape_rows:
        L.append(f"| {r['run_id']} | {r['global_update']} | {f(r['e3_at_0'])} | {f(r['e3_rise_within_2q'])} | "
                 f"{f(r['e3_min_D3'])} / {f(r['e3_max_D3'])} |")
    L += ["", "pilot_reference rows: earlier pilot q60 runs (same code as the baseline arm, different seeds); "
          "update 1000 = end of B, 2800 = end of C.", "",
          "Files: runs.csv, pairs.csv, trajectory.csv, policy_shape_trajectory.csv, decision.json, "
          "figures/ (stage3_gain_and_policy, trajectory, exposure)."]
    (out_dir / "COMPARISON.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
