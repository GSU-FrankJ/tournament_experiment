"""Report the preregistered fixed-triplet restart evaluation; never launches runs."""
from pathlib import Path
import csv
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
COSTS = ("total_updates", "total_episodes", "total_transitions", "dev_calls",
         "training_seconds", "final_eval_seconds", "total_wall_seconds", "process_cpu_seconds")


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def le(x, threshold):
    return finite(x) and x <= threshold


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rate(k, n):
    if not n:
        return {"k": k, "n": n, "rate": None, "wilson95": None}
    z = 1.959963984540054
    p, den = k / n, 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return {"k": k, "n": n, "rate": p, "wilson95": [max(0, center-half), min(1, center+half)]}


def stats(values):
    a = [v for v in values if finite(v)]
    return {"n": len(a), "missing": len(values)-len(a), "sum": sum(a) if a else None,
            "mean": statistics.mean(a) if a else None, "median": statistics.median(a) if a else None,
            "min": min(a) if a else None, "max": max(a) if a else None}


def flatten(d, prefix=""):
    result = {}
    for key, value in d.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(flatten(value, name + "."))
        elif finite(value):
            result[name] = value
    return result


def read_run(rec, launch):
    row = {k: rec.get(k) for k in ("run", "seed", "q", "output_dir")}
    row.update(operation="done", issue="", candidate=False, joint_certified=False,
               certification="not_applicable_no_candidate", first_C_update=None, first_global_update=None,
               terminal_final_overall_pass=None, final_concentration_pass=None,
               final_dreach=None, final_exp=None, dense_concentration=None,
               refine_dreach_diff=None, refine_exp_diff=None, recovery={})
    row.update({k: None for k in COSTS})
    row["launcher_wall_seconds"] = launch.get("wall_sec") if launch else None
    d = Path(rec["output_dir"])
    try:
        cfg = json.loads((d/"config.json").read_text())
        require(cfg["record"] == rec and not cfg["smoke"], "configuration does not match formal manifest")
        for key in ("game", "ppo", "protocol", "lr_schedule", "verifier"):
            if key in cfg:
                require(cfg[key] == rec[key], "effective configuration mismatch: " + key)
        if "versions_actual" in cfg:
            require(cfg["versions_actual"] == rec["versions"], "software versions differ from manifest")
        h = json.loads((d/"train_history.json").read_text())
        c = [v for v in h["verifier_calls"] if v["phase"] == "C"]
        eligible = []
        for v in c:
            bp = bool(v["valid"] and le(v["criterion_value_over_dw"], .01))
            cp = bool(v["concentration"]["valid"] and le(v["concentration"]["max_std_norm"], .04))
            ep = bp and cp
            require((bp, cp, ep) == (v["br_pass"], v["conc_pass"], v["eligible"]),
                    "development flags disagree with numerical criteria")
            if ep:
                eligible.append(v)
        require(len(eligible) <= 1, "training continued after first eligible candidate")
        if eligible:
            require(h["history"][-1]["update"] == eligible[0]["update"],
                    "first eligible candidate is not the stopping checkpoint")
        row.update(candidate=bool(eligible), total_updates=len(h["history"]), dev_calls=len(h["verifier_calls"]))
        if eligible:
            row.update(first_C_update=eligible[0]["local"], first_global_update=eligible[0]["update"])
        f = json.loads((d/"final_eval.json").read_text())
        costs, stop = f["costs"], f["stopping_record"]
        row.update(total_updates=costs["total_updates"], total_episodes=stop["total_episodes"],
                   total_transitions=stop["total_transitions"], training_seconds=costs["training_wall_sec"],
                   final_eval_seconds=costs["final_eval_sec"], total_wall_seconds=costs["total_wall_sec"],
                   process_cpu_seconds=costs.get("total_process_cpu_sec"), recovery=flatten(f.get("recovery", {})))
        require(not f["smoke"], "smoke result is not a formal run")
        require(bool(eligible) == stop["development_stopping_criterion_satisfied"], "candidate/stop mismatch")
        require(len(h["history"]) == stop["stop_update"] == costs["total_updates"], "update counts disagree")
        if eligible:
            require(eligible[0]["update"] == stop["stop_update"], "stopped after first eligible checkpoint")
            require(eligible[0]["local"] == stop["phase_C_local_at_stop"], "C stopping update disagrees")
        else:
            require(stop["reason"] == "budget_exhausted" and
                    stop["phase_C_local_at_stop"] == rec["protocol"]["phase_caps"]["C"],
                    "no-candidate run did not exhaust the C budget")
        dev, final = f["tiers"]["development"], f["tiers"]["final"]
        if eligible and dev["valid"]:
            require(dev.get("dreach_over_dw") == eligible[0]["criterion_value_over_dw"],
                    "final development check does not match selected candidate")
        dr, de = None, None
        if all(finite(t.get(k)) for t in (dev, final) for k in ("dreach_over_dw", "exp_root_over_dw")):
            dr = abs(final["dreach_over_dw"] - dev["dreach_over_dw"])
            de = abs(final["exp_root_over_dw"] - dev["exp_root_over_dw"])
        final_pass = bool(dev["valid"] and final["valid"] and le(final.get("dreach_over_dw"), .01)
                          and le(dr, .002) and le(de, .002))
        flags = f["pass_flags"]
        require(final_pass == flags["overall_pass"], "final flag disagrees with numerical criteria")
        for key, value in (("refine_dreach_diff_over_dw", dr), ("refine_exp_diff_over_dw", de)):
            if finite(value):
                require(finite(flags[key]) and math.isclose(flags[key], value, rel_tol=1e-12, abs_tol=1e-15),
                        "reported refinement difference disagrees: " + key)
        conc = f["concentration"]["C_all"]
        conc_pass = bool(conc["valid"] and le(conc["max_std_norm"], .04))
        row.update(terminal_final_overall_pass=final_pass, final_concentration_pass=conc_pass,
                   final_dreach=final.get("dreach_over_dw"), final_exp=final.get("exp_root_over_dw"),
                   dense_concentration=conc["max_std_norm"], refine_dreach_diff=dr, refine_exp_diff=de)
        status = json.loads((d/"status.json").read_text())
        require(status["state"] == "done", "run status is not done")
        require(launch is not None and launch.get("returncode") == 0, "missing or failed launch record")
        row["joint_certified"] = bool(row["candidate"] and final_pass and conc_pass)
        if row["candidate"]:
            row["certification"] = "passed" if row["joint_certified"] else "failed"
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        row.update(operation="error", issue=f"{type(exc).__name__}: {exc}", joint_certified=False)
        if row["candidate"]:
            row["certification"] = "unavailable_operation_error"
    return {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in row.items()}


def summarize(manifest, rows):
    by_seed = {r["seed"]: r for r in rows}
    planned = [r["seed"] for r in manifest["runs"]]
    require(len(by_seed) == len(rows) == len(planned) and set(by_seed) == set(planned),
            "report must contain exactly every planned run")
    assignments = manifest["restart_evaluation"]["groups"]
    require([s for g in assignments for s in g["seeds"]] == planned,
            "triplet assignment disagrees with predetermined manifest order")
    groups, by_k, selected = [], {}, []
    for g in assignments:
        a = [by_seed[s] for s in g["seeds"]]
        require(len(a) == 3, "each fixed group must contain three runs")
        first = next((i for i, r in enumerate(a, 1) if r["joint_certified"]), None)
        group = {"group_id": g["group_id"], "seeds": g["seeds"],
                 "first_joint_success_position": first,
                 "first_joint_success_seed": a[first-1]["seed"] if first else None}
        if first:
            selected.append((first, a[first-1]))
        for k in (1, 2, 3):
            attempts = min(first, k) if first else k
            group.update({f"k{k}_candidate": any(r["candidate"] for r in a[:k]),
                          f"k{k}_joint": any(r["joint_certified"] for r in a[:k]),
                          f"k{k}_simulated_attempts": attempts})
            for cost in COSTS:
                values = [r[cost] for r in a[:attempts]]
                group[f"k{k}_simulated_{cost}"] = sum(values) if all(finite(v) for v in values) else None
        group["rescued"] = bool(first and first > 1)
        groups.append(group)
    for k in (1, 2, 3):
        by_k[str(k)] = {"candidate_discovery": rate(sum(g[f"k{k}_candidate"] for g in groups), len(groups)),
                       "joint_success": rate(sum(g[f"k{k}_joint"] for g in groups), len(groups)),
                       "simulated_attempts": sum(g[f"k{k}_simulated_attempts"] for g in groups),
                       "simulated_costs": {c: stats([g[f"k{k}_simulated_{c}"] for g in groups]) for c in COSTS}}
    n, candidates, certified = len(rows), sum(r["candidate"] for r in rows), sum(r["joint_certified"] for r in rows)
    errors, rescues = sum(r["operation"] != "done" for r in rows), sum(g["rescued"] for g in groups)
    useful = by_k["3"]["joint_success"]["k"] > by_k["1"]["joint_success"]["k"] and rescues >= 1
    recovery = {}
    keys = sorted({key for _, r in selected for key in r["recovery"]})
    for position in (1, 2, 3):
        a = [r for p, r in selected if p == position]
        recovery[str(position)] = {"n": len(a), "seeds": [r["seed"] for r in a],
                                  "metrics": {key: stats([r["recovery"].get(key) for r in a]) for key in keys}}
    return {"single_run": {"candidate_discovery": rate(candidates, n),
                          "conditional_certification": rate(certified, candidates),
                          "end_to_end": rate(certified, n)}, "rows": [by_seed[s] for s in planned],
            "groups": groups, "by_k": by_k, "actual_runs": n, "operation_errors": errors,
            "actual_evaluation_costs": {c: stats([r[c] for r in rows]) for c in COSTS},
            "launcher_process_wall_seconds": stats([r["launcher_wall_seconds"] for r in rows]),
            "rescued_groups": rescues, "recovery_by_first_success_position": recovery,
            "decision": "inconclusive_operation_errors" if errors else
                        ("retain_useful_T2_pilot" if useful else "terminate_method")}


def format_rate(r):
    if r["rate"] is None:
        return "N/A (0 candidates)"
    low, high = r["wilson95"]
    return f'{r["k"]}/{r["n"]} ({r["rate"]:.1%}; Wilson 95% {low:.1%}–{high:.1%})'


def write_outputs(root, result):
    (root/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, values in (("runs.csv", result["rows"]), ("groups.csv", result["groups"])):
        with (root/name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in values)
    lines = ["# q=50, T=2 prospective random-restart evaluation", "",
             "All 30 scheduled seeds were assigned to ten fixed non-overlapping triples before outcomes. "
             "k=1,2,3 use their fixed prefixes. All runs are evaluated even after an earlier success.", "",
             "Candidate = first valid development C call with dReach/Δw ≤0.01 and concentration ≤0.04. "
             "Joint certification additionally requires valid development and final tiers, final dReach/Δw ≤0.01, "
             "both dReach and exploitability refinement differences ≤0.002, and dense concentration ≤0.04. "
             "No-candidate terminal evaluations are diagnostic only.", ""]
    for label, r in result["single_run"].items():
        lines.append(f'- {label}: {format_rate(r)}')
    lines += ["", "| Fixed budget | Candidate discovery | Joint certification | Reconstructed attempts |",
              "|---|---|---|---|"]
    for k, data in result["by_k"].items():
        lines.append(f'| k={k} | {format_rate(data["candidate_discovery"])} | {format_rate(data["joint_success"])} | {data["simulated_attempts"]} |')
    lines += ["", f'Rescued groups: {result["rescued_groups"]}. Operation errors: {result["operation_errors"]}.',
              f'Prespecified decision: **{result["decision"]}**.', "",
              "This is a descriptive engineering decision with ten paired group replications of one configuration. "
              "Prefix results are dependent; Wilson intervals are marginal intervals, not an interval for the gain. "
              "There is no significance or reliability-guarantee claim. The per-run success rate is unchanged by regrouping. "
              "T2 results cannot certify T3; any T3 pilot requires a separate preregistration.", "",
              "## Resource costs", "", "Actual evaluation costs sum all scheduled runs. Reconstructed sequential costs stop "
              "after the first jointly certified candidate, or exhaust k attempts. They are counterfactual costs from "
              "the observed runs; sums of process wall time are not parallel batch elapsed time. Missing costs are explicit.",
              "", "| Metric | Actual all runs | Reconstructed k1 | Reconstructed k2 | Reconstructed k3 |",
              "|---|---:|---:|---:|---:|"]
    def cost_cell(x):
        return f'{x["sum"]:.6g} (missing {x["missing"]})' if x["sum"] is not None else f'N/A (missing {x["missing"]})'
    for cost in COSTS:
        cells = [result["actual_evaluation_costs"][cost]] + [result["by_k"][str(k)]["simulated_costs"][cost] for k in (1, 2, 3)]
        lines.append("| " + cost + " | " + " | ".join(cost_cell(x) for x in cells) + " |")
    if "batch_wall_seconds" in result:
        lines += ["", f'Actual parallel batch elapsed seconds: {result["batch_wall_seconds"]}.']
    lines += ["", "## Fixed groups", "", "| Group | Seeds in fixed order | First certified position | k1 | k2 | k3 |",
              "|---|---|---:|---|---|---|"]
    for g in result["groups"]:
        lines.append(f'| {g["group_id"]} | {g["seeds"]} | {g["first_joint_success_position"] or "none"} | {g["k1_joint"]} | {g["k2_joint"]} | {g["k3_joint"]} |')
    lines += ["", "## Every scheduled run", "", "| Seed | Operation | Candidate | Certification | C update | Final dReach | Dense concentration | Issue |",
              "|---|---|---|---|---:|---:|---:|---|"]
    for r in result["rows"]:
        lines.append(f'| {r["seed"]} | {r["operation"]} | {r["candidate"]} | {r["certification"]} | {r["first_C_update"]} | {r["final_dreach"]} | {r["dense_concentration"]} | {r["issue"]} |')
    lines += ["", "## Recovery diagnostics by first certified position", "",
              "These describe the returned candidates only and have no acceptance threshold. All available numerical "
              "recovery fields, counts and seed identities are included in summary.json.", "",
              "| Position | Candidates | Stage1 absolute error mean | Stage2 positive-region RMSE mean | On-path RMSE mean |",
              "|---|---:|---:|---:|---:|"]
    for pos, data in result["recovery_by_first_success_position"].items():
        means = [data["metrics"].get(key, {}).get("mean") for key in
                 ("stage1.abs_error", "stage2_positive_region.pooled.rmse", "onpath.rmse")]
        lines.append(f'| {pos} | {data["n"]} | ' + " | ".join("N/A" if v is None else f"{v:.6g}" for v in means) + " |")
    (root/"REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    manifest = json.loads((ROOT/"manifest.json").read_text())
    launch = json.loads((ROOT/"launch_status.json").read_text())
    if launch.get("state") not in ("done", "failed"):
        print("Batch is still running; final report and decision are pending.")
        return
    runs = manifest["runs"]
    require([r["seed"] for r in runs] == list(range(10201, 10231)), "unexpected formal seed schedule")
    normalized = lambda r: {k: v for k, v in r.items() if k not in ("run", "seed", "output_dir")}
    source = json.loads(Path(manifest["source_manifest"]).read_text())["runs"][0]
    require(all(normalized(r) == normalized(source) for r in runs), "run settings differ from unchanged source protocol")
    launch_by_run = {r["run"]: r for r in launch["runs"]}
    result = summarize(manifest, [read_run(r, launch_by_run.get(r["run"])) for r in runs])
    result.update(manifest=str(ROOT/"manifest.json"), batch_wall_seconds=launch.get("wall_sec"),
                  configuration_check="All 30 planned configurations equal source except run, seed and output_dir")
    write_outputs(ROOT, result)
    print(json.dumps({k: result[k] for k in ("single_run", "rescued_groups", "operation_errors", "decision")}, indent=2))


if __name__ == "__main__":
    main()
