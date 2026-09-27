"""Replay archived endpoint and minimum-development weights; never trains or writes."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from common import E
from run_experiment import fresh_agent
from envs.curriculum_env import GameSpec
from run.run_final_dp_br import make_policy_fns, run_verifier
from utils.dp_br_verifier import VerifierConfig

def replay(run_id: str):
    directory = E / "runs" / run_id
    cfg = json.loads((directory / "config.json").read_text())["resolved"]
    spec = GameSpec(**cfg["game"])
    final = json.loads((directory / "final_eval.json").read_text())
    minimum = json.loads((directory / "checkpoints/min_dev_record.json").read_text())
    results = []
    for kind, filename, expected_tiers in (
        ("diagnostic_terminal", "endpoint_weights.npz", final["tiers"]),
        ("minimum_development", "min_dev_weights.npz", {"development": minimum["summary"]}),
    ):
        agent = fresh_agent(cfg)
        with np.load(directory / "checkpoints" / filename, allow_pickle=False) as weights:
            identity = json.loads(str(weights["identity_json"]))
            if identity["run_id"] != run_id or identity["checkpoint_kind"] != kind:
                raise ValueError(f"Checkpoint identity mismatch: {run_id}, {kind}")
            agent.actor.load_state_dict({
                key.removeprefix("actor."): torch.as_tensor(weights[key])
                for key in weights.files if key.startswith("actor.")
            })
        mean_fn, beta_fn = make_policy_fns(agent, spec)
        for tier, expected in expected_tiers.items():
            result, error = run_verifier(mean_fn, beta_fn, spec, VerifierConfig(**cfg["verifier"][tier]))
            if result is None:
                raise RuntimeError(f"{run_id} {kind} {tier}: {error}")
            names = ("exp_root", "dreach", "dfull", "delta_max_all", "v_mean_root", "v_br_root")
            errors = {name: abs(float(getattr(result, name)) - float(expected[name])) for name in names}
            matches = bool(result.valid == expected["valid"] and all(
                np.isclose(getattr(result, name), expected[name], rtol=1e-7, atol=1e-9) for name in names))
            if not matches:
                raise AssertionError(f"Verifier replay differs: {run_id} {kind} {tier}: {errors}")
            results.append({"run_id": run_id, "checkpoint_kind": kind, "tier": tier,
                            "max_abs_error": max(errors.values()), "matches": matches})
    return results

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", choices=("formal", "pilot", "all"), default="all")
    p.add_argument("--run-id", action="append", help="restrict replay to named run(s)")
    args = p.parse_args()
    torch.set_num_threads(1)
    cohorts = ("formal", "pilot") if args.cohort == "all" else (args.cohort,)
    rows = []
    for cohort in cohorts:
        with (E / "reports" / cohort / "runs.csv").open() as source:
            rows.extend(csv.DictReader(source))
    selected = [r for r in rows if not args.run_id or r["run_id"] in args.run_id]
    if not selected or (args.run_id and set(args.run_id) != {r["run_id"] for r in selected}):
        p.error("Requested run is outside the selected archived cohorts")
    results = [item for row in selected for item in replay(row["run_id"])]
    print(json.dumps({"runs": len(selected), "verifier_replays": len(results),
                      "max_abs_error": max(r["max_abs_error"] for r in results),
                      "all_match": True, "policy_status": "diagnostic, not certified equilibrium",
                      "results": results}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
