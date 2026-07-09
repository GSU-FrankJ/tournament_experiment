"""Aggregate per-seed multi-stage runs and apply the pre-registered T=2 gate.

Reads the convergence JSONs written by ``run/run_multi_stage.py`` for a set
of seeds, prints a per-seed table, and evaluates the frozen acceptance gate
from ``docs/tasks/multistage-tel-ppo/preregistration_T2.md``
(implemented in ``utils/multi_stage_metrics.py``).

Run:
    python tools/evaluate_gate.py --glob "results/multi_stage/convergence/ms_T2_q50_seed*_gateT2_convergence.json"
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.multi_stage_metrics import (  # noqa: E402
    GATE_DREACH_OVER_DW,
    SeedGateInput,
    evaluate_gate,
)


def main() -> int:
    """Load seed JSONs, print the table, and apply the gate; return 0 on PASS."""
    p = argparse.ArgumentParser(description="Evaluate the pre-registered T=2 gate")
    p.add_argument("--glob", type=str,
                   default="results/multi_stage/convergence/ms_T2_q50_seed*_gateT2_convergence.json")
    args = p.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"no files match: {args.glob}")
        return 2

    seeds: List[SeedGateInput] = []
    dw = None
    print(f"{'seed':>5} {'EXP':>9} {'EXP/DW':>8} {'EXP^UCB/DW':>11} {'dReach/DW':>10} "
          f"{'cert?':>6} {'RE_1':>7} {'RPE2core':>9} {'PL_2/DW':>8}")
    for path in paths:
        with open(path) as f:
            d = json.load(f)
        params = d["params"]
        dw = params["w_h"] - params["w_l"]
        fe = d["final_eval"]
        gr = d.get("grid_refinement", {})
        rm = d.get("recovery_metrics") or {}
        dreach_over_dw = fe["delta_sum_reachable"] / dw
        exp_over_dw = fe["exp_over_dw"]
        ucb_over_dw = gr.get("exp_ucb_over_dw", float("nan"))
        certified = dreach_over_dw <= GATE_DREACH_OVER_DW
        seeds.append(SeedGateInput(
            seed=params["seed"],
            dreach_over_dw=dreach_over_dw,
            exp_over_dw=exp_over_dw,
            certified=certified,
            re_1=rm.get("re_1", float("nan")),
            rpe_2_core=rm.get("rpe_2_core", float("nan")),
        ))
        print(f"{params['seed']:>5} {fe['exp']:>9.4f} {exp_over_dw:>8.4f} "
              f"{ucb_over_dw:>11.4f} {dreach_over_dw:>10.4f} {str(certified):>6} "
              f"{rm.get('re_1', float('nan')):>7.3f} {rm.get('rpe_2_core', float('nan')):>9.3f} "
              f"{rm.get('pl_2_over_dw', float('nan')):>8.3f}")

    verdict = evaluate_gate(seeds)
    print(f"\nGATE ({verdict.n_seeds} seeds): "
          f"{'PASS' if verdict.passed else 'FAIL'}")
    print(f"  certified {verdict.n_certified}/{verdict.n_seeds} "
          f"(frac {verdict.cert_fraction:.2f}); "
          f"dReach/DW mean {verdict.dreach_mean:.4f} std {verdict.dreach_std:.4f} "
          f"max {verdict.dreach_max:.4f}")
    print(f"  EXP/DW mean {verdict.exp_mean:.4f} std {verdict.exp_std:.4f}; "
          f"RE_1 mean {verdict.re1_mean:.3f}; RPE2core mean {verdict.rpe2_core_mean:.3f}")
    for r in verdict.reasons:
        print(f"  {r}")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
