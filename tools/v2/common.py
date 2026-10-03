"""Shared helpers for the v2 tools: game specs, analytic candidates, checkpoint policies."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.ppo_curriculum import CurriculumPPO, PPOConfig  # noqa: E402
from envs.curriculum_env import GameSpec  # noqa: E402
from run.run_final_dp_br import make_policy_fns  # noqa: E402
from utils.dp_br_verifier import DEV_CONFIG, FINAL_CONFIG, VerifierConfig  # noqa: E402
from utils.theory_multistage import g1_two_stage, g2_two_stage  # noqa: E402

BASE_COMMIT = "657f54a"
W_H, W_L, K = 6.0, 2.0, 1.0 / 3500.0   # as-run values, reports/v2/phase0_audit.md section 2
DEV_2X = VerifierConfig("dev_2x", state_step=DEV_CONFIG.state_step / 2,
                        effort_step=DEV_CONFIG.effort_step / 2, gl_half=DEV_CONFIG.gl_half)
TIERS = (DEV_CONFIG, DEV_2X, FINAL_CONFIG)


def spec_for(q: float) -> GameSpec:
    """As-run T=2 game for one q."""
    return GameSpec(w_h=W_H, w_l=W_L, k=K, q=float(q), T=2, e_min=0.0, e_max=100.0)


def analytic_policy(spec: GameSpec) -> Callable[[int, np.ndarray], np.ndarray]:
    """Closed-form T=2 equilibrium (e1*, e2*) as a policy (evaluation only)."""
    g1 = g1_two_stage(spec.q, spec.w_h, spec.w_l, spec.k)

    def pol(t: int, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        if t == 1:
            return np.full(d.shape, g1)
        return g2_two_stage(d, spec.q, spec.w_h, spec.w_l, spec.k, spec.e_max)
    return pol


def zero_policy(t: int, d: np.ndarray) -> np.ndarray:
    """e_hat == 0 at every stage."""
    return np.zeros(np.asarray(d).shape)


def checkpoint_policy(path: str, spec: GameSpec) -> Tuple[Callable, Callable]:
    """(mean_fn, beta_fn) of a saved CurriculumPPO checkpoint (actor weights only used)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = PPOConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items()})
    agent = CurriculumPPO(cfg, torch.Generator().manual_seed(0), np.random.default_rng(0))
    agent.actor.load_state_dict(ck["actor"])
    return make_policy_fns(agent, spec)
