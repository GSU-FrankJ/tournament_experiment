#!/usr/bin/env python3
"""
Diagnose whether TwoPlayersEnv noise becomes deterministic due to re-creating the
environment (and re-seeding its RNG) every rollout step.

This script reproduces the *runner's* pattern from `run/run_two_players.py`
and compares it against a control that instantiates the env once and reuses it.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import torch

# Ensure repo root is on sys.path (tools/ is one level under root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.two_players_env import TwoPlayersEnv


@dataclass(frozen=True)
class Params:
    w_h: float = 6.5
    w_l: float = 3.0
    k: float = 0.0004
    q: float = 40.0
    seed: int = 42
    effort_bounds: Tuple[float, float] = (0.0, 200.0)


def _step_once(env: TwoPlayersEnv, e1: float, e2: float) -> Tuple[Tuple[float, float], Tuple[float, float], int]:
    _, _, _, _, info = env.step(
        (
            torch.tensor([e1], dtype=torch.float32),
            torch.tensor([e2], dtype=torch.float32),
        )
    )
    eps1, eps2 = info["noises"]
    y1, y2 = info["outputs"]
    winner = int(info["winner"])
    return (float(eps1), float(eps2)), (float(y1), float(y2)), winner


def scenario_recreate_env_each_step(
    params: Params, *, e1: float, e2: float, n_steps: int
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[int]]:
    noises: List[Tuple[float, float]] = []
    outputs: List[Tuple[float, float]] = []
    winners: List[int] = []
    for _ in range(n_steps):
        # This matches the pattern in run/run_two_players.py:427
        env = TwoPlayersEnv(
            w_h=params.w_h,
            w_l=params.w_l,
            k=params.k,
            q=params.q,
            effort_bounds=params.effort_bounds,
            seed=params.seed,
        )
        eps_pair, y_pair, winner = _step_once(env, e1, e2)
        noises.append(eps_pair)
        outputs.append(y_pair)
        winners.append(winner)
    return noises, outputs, winners


def scenario_reuse_single_env(
    params: Params, *, e1: float, e2: float, n_steps: int
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[int]]:
    env = TwoPlayersEnv(
        w_h=params.w_h,
        w_l=params.w_l,
        k=params.k,
        q=params.q,
        effort_bounds=params.effort_bounds,
        seed=params.seed,
    )
    noises: List[Tuple[float, float]] = []
    outputs: List[Tuple[float, float]] = []
    winners: List[int] = []
    for _ in range(n_steps):
        eps_pair, y_pair, winner = _step_once(env, e1, e2)
        noises.append(eps_pair)
        outputs.append(y_pair)
        winners.append(winner)
    return noises, outputs, winners


def _all_equal(seq) -> bool:
    if not seq:
        return True
    first = seq[0]
    return all(x == first for x in seq[1:])


def _print_summary(label: str, noises: List[Tuple[float, float]], winners: List[int], show: int = 10) -> None:
    print(f"\n[{label}]")
    print(f"first_{show}_noises={noises[:show]}")
    print(f"first_{show}_winners={winners[:show]}")
    print(f"all_noises_equal_within_run={_all_equal(noises)}")
    print(f"all_winners_equal_within_run={_all_equal(winners)}")

def _check(condition: bool, message: str) -> bool:
    if condition:
        print(f"PASS: {message}")
        return True
    print(f"FAIL: {message}")
    return False


def main() -> int:
    torch.set_num_threads(1)

    params = Params()
    n_steps = 50
    # Fixed efforts to isolate env noise behavior
    e1, e2 = 100.0, 110.0

    # Scenario A: re-create env each step (matches the old runner behavior)
    noises_a1, _outputs_a1, winners_a1 = scenario_recreate_env_each_step(params, e1=e1, e2=e2, n_steps=n_steps)
    noises_a2, _outputs_a2, winners_a2 = scenario_recreate_env_each_step(params, e1=e1, e2=e2, n_steps=n_steps)

    # Scenario B: reuse a single env instance (RNG state should advance)
    noises_b1, _outputs_b1, winners_b1 = scenario_reuse_single_env(params, e1=e1, e2=e2, n_steps=n_steps)
    noises_b2, _outputs_b2, winners_b2 = scenario_reuse_single_env(params, e1=e1, e2=e2, n_steps=n_steps)

    _print_summary("A (recreate env each step)", noises_a1, winners_a1)
    print(f"A_repeated_run_identical_noises={noises_a1 == noises_a2}")
    print(f"A_repeated_run_identical_winners={winners_a1 == winners_a2}")

    _print_summary("B (reuse single env)", noises_b1, winners_b1)
    print(f"B_repeated_run_identical_noises={noises_b1 == noises_b2}")
    print(f"B_repeated_run_identical_winners={winners_b1 == winners_b2}")

    checks = [
        _check(_all_equal(noises_a1), "Scenario A noises are constant within run"),
        _check(noises_a1 == noises_a2, "Scenario A noises are identical across runs with same seed"),
        _check(not _all_equal(noises_b1), "Scenario B noises vary within run"),
        _check(noises_b1 == noises_b2, "Scenario B noises are reproducible across runs with same seed"),
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
