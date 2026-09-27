#!/usr/bin/env python3
"""Resolved, q-specific manifests for the T=3 implementation pilot (A -> B -> C).

Every manifest is a complete, independent dict: ``resolved.game.q`` is the only
q source (the top-level ``q`` is asserted equal), and every derived quantity
(B, domains, bins, grid sizes, dense points, expected exposures) is computed
from GameSpec / the verifier grid functions, never copied from a template.

Usage (from the checkout root):
    python -B experiments/three_stage_implementation_pilot_20260924/build_manifests.py --cohort smoke
    python -B experiments/three_stage_implementation_pilot_20260924/build_manifests.py --cohort pilot
    python -B .../build_manifests.py --cohort formal --settings .../reports/formal_settings.json --seeds-per-q 10
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import E, PLAN, PYTHON, ROOT, W, write_json_atomic  # noqa: E402

from agents.ppo_curriculum import PPOConfig  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402
from utils.dp_br_verifier import DEV_CONFIG, FINAL_CONFIG, stage_grid  # noqa: E402

EXPERIMENT = "three_stage_implementation_pilot_20260924"
Q_VALUES = (60, 50)
GAME_BASE: Dict[str, Any] = {"w_h": 6.0, "w_l": 2.0, "k": 1.0 / 3500.0, "T": 3,
                             "e_min": 0.0, "e_max": 100.0}

SEEDS: Dict[str, Dict[int, List[int]]] = {
    "smoke": {60: [10400], 50: [10410]},
    "pilot": {60: [10401, 10402, 10403], 50: [10411, 10412, 10413]},
    "formal": {50: list(range(11001, 11021)), 60: list(range(11101, 11121))},
    # Phase-A state-sampling study (A-only debug; paired arms share each seed)
    "asamp_smoke": {60: [10430]},
    "asamp": {60: [10431, 10432, 10433]},
    # Paired full A/B/C validation of the Phase-A sampling change (A differs only)
    "asamp_abc_smoke": {60: [10440]},
    "asamp_abc": {60: [10441, 10442, 10443]},
}
ABC_COHORTS = ("asamp_abc_smoke", "asamp_abc")
ABC_STUDY: Dict[str, Any] = {
    "name": "phase_A_state_sampling_full_ABC_20260924",
    "phases_run": ["A", "B", "C"],
    "question": "Does the center-weighted phase-A sampling carry over to B/C and to final certification?",
    "arms": {"baseline": "pilot configuration (A: 512 ES3 uniform on D3)",
             "center": "pilot configuration except phase A: 256 ES3 uniform on D3 + 256 ES3 uniform on [-B, B]"},
    "only_difference": "phase-A start sampling; A400/B600/C1800, B/C start mixtures, PPO, cadence, first-eligible "
                       "rule, final verifier, economics and all thresholds are the pilot values; the center "
                       "sampling is not applied to B or C",
    "pairing": "each fresh seed runs both arms from initialization (no continuation of earlier A400 runs; "
               "earlier pilot runs are not used as the paired control)",
    "primary_endpoint": "final certification success (certification == certified) per run",
    "reported_counts": "per arm: certified / all runs, candidate / all runs, certified / candidates (N/A when "
                       "there is no candidate); per-seed paired outcomes; 3 seeds are diagnostic only and no "
                       "reliability rate is claimed",
    "continuous_diagnostic": "minimum valid C development dReach/DW with the same call's stage contributions, "
                             "maximum-deviation states, concentration, plus actual check counts, updates and "
                             "environment steps; an early stop is reported as the actual stop, never as a C1800 "
                             "endpoint, and the historical minimum is never promoted to a candidate",
    "retention_check": "stage-3 policy and stage-3 deviation (full/center/outer) plus stage-2 deviation "
                       "(continuation gain and one-step delta) at A400, B exit, C minimum and the actual stop; "
                       "descriptive paired signs per checkpoint",
    "interpretation_rules": "if the A advantage disappears in B/C and there is no candidate or certification "
                            "gain, stop this sampling change; if only the minimum decreases and both arms still "
                            "have no candidate, report a diagnostic improvement and do not advance to formal; "
                            "q50 and formal are not run in this round",
    "retention_operationalization": "the A advantage is retained at a checkpoint if the center arm's dev-tier "
                                    "stage-3 full-domain max (V3_BR-V3_mean)/DW is lower in all 3 pairs; it has "
                                    "disappeared in B/C if it is retained at neither the C minimum nor the actual "
                                    "stop",
    "gain_operationalization": "candidate gain = more candidate runs in the center arm than in baseline; "
                               "certification gain = more certified runs in the center arm than in baseline",
    "diagnostic_improvement_operationalization": "minimum valid C dReach/DW lower in the center arm in all 3 pairs "
                                                 "while both arms have 0 candidates",
}
A_SAMPLING_COHORTS = ("asamp_smoke", "asamp")
A_SAMPLING_ARMS = ("baseline", "center")
A_SAMPLING_STUDY: Dict[str, Any] = {
    "name": "phase_A_state_sampling_20260924",
    "question": "Does moving A's ES3 weight toward the center interval [-B, B] reduce the A400 full-domain "
                "stage-3 deviation, given a fixed A budget and a shared network?",
    "arms": {"baseline": "512 ES3/update uniform on D3 (pilot phase A)",
             "center": "256 ES3/update uniform on D3 + 256 ES3/update uniform on [-B, B] (its width-10 bins)"},
    "rationale": "At stage 3, |d| > B cannot change the terminal winner for any feasible actions/noise; "
                 "uniform D3=[-2B,2B] puts half of the starts there. The mixture raises the expected center "
                 "share from 50% to 75% while keeping full-domain coverage. This tests allocation of a fixed "
                 "training amount across state regions; it is not a fix for missing bins and may not help.",
    "unchanged": "A cap 400, check cadence 100, PPO parameters, concentration threshold, verifier tiers, "
                 "all thresholds; B and C are not run",
    "pairing": "each seed runs both arms (same initial weights and RNG stream seeds)",
    "primary_metric": "final-tier max_{d in D3} (V3_BR(d) - V3_mean(d))/DW at the A400 endpoint",
    "secondary_metrics": "center (|d| <= B) and outer (|d| > B) max and grid-mean of the same gain (final and "
                         "dev tiers), argmax location, stage-3 concentration (dev grid max std_norm at A400 and "
                         "dense D3 by region), A100-A400 dev-tier trajectory, actual per-bin exposure",
    "decision_rule": "consistent improvement iff the center arm's primary metric is lower than the baseline's in "
                     "all 3 seed pairs; center/outer values and concentration are reported with it, and a center "
                     "arm that exceeds concentration 0.04 where baseline does not is flagged; training mean "
                     "reward is descriptive only and never a criterion",
    "next_if_consistent": "validate full B/C at q60, then q50; an A improvement alone is not sufficient for formal "
                          "(the C-end stage-2 deviations are also large)",
}
# Seeds exposed by the earlier exploratory T3 debug batch (smoke 10300, runs 10301-10303).
HISTORICAL_T3_SEEDS = [10300, 10301, 10302, 10303]

PPO: Dict[str, Any] = {
    "hidden": 64, "lr": 3e-4, "adam_betas": [0.9, 0.999], "adam_eps": 1e-8, "weight_decay": 0.0,
    "clip_eps": 0.2, "value_coef": 0.5, "entropy_coef": 0.0, "max_grad_norm": 0.5, "epochs": 10,
    "minibatch": 256, "gamma": 1.0, "gae_lambda": 1.0, "c_min": 100.0, "mu_clamp": 1e-6,
    "action_clamp": 1e-6, "adv_norm_eps": 1e-8,
}

PHASES: List[Dict[str, Any]] = [
    {"name": "A", "active_stages": [3], "start_counts": {"3": 512}, "cap": 400,
     "check_every": 100, "check_role": "diagnostic_only", "k_consecutive": None,
     "criterion": "stage3_continuation_gain_over_dw", "threshold_over_dw": None,
     "concentration_stages": [3], "exit_rule": "fixed_budget",
     "cap_exit_reason": "fixed_budget_completed"},
    {"name": "B", "active_stages": [2, 3], "start_counts": {"2": 512}, "cap": 600,
     "check_every": 25, "check_role": "eligibility", "k_consecutive": 3,
     "criterion": "max_D2_continuation_gain_over_dw", "threshold_over_dw": 0.02,
     "concentration_stages": [2, 3], "exit_rule": "k_consecutive_eligible",
     "pass_exit_reason": "verifier_passed", "cap_exit_reason": "budget_forced"},
    {"name": "C", "active_stages": [1, 2, 3], "start_counts": {"1": 256, "2": 85, "3": 171},
     "cap": 1800, "check_every": 25, "check_role": "eligibility", "k_consecutive": 1,
     "criterion": "dreach_over_dw", "threshold_over_dw": 0.01,
     "concentration_stages": [1, 2, 3], "exit_rule": "first_eligible_candidate",
     "pass_exit_reason": "candidate_found", "cap_exit_reason": "no_candidate_budget_exhausted"},
]

SMOKE_OVERRIDES: Dict[str, Any] = {
    "phases": {
        "A": {"cap": 2, "check_every": 1, "start_counts": {"3": 32}},
        "B": {"cap": 2, "check_every": 1, "k_consecutive": 3, "start_counts": {"2": 32}},
        "C": {"cap": 2, "check_every": 1, "k_consecutive": 1,
              "start_counts": {"1": 16, "2": 5, "3": 11}},
    },
    "economics": {"replicates": 2, "episodes_per_replicate": 200},
    "note": "smoke only: caps 2/2/2, cadence 1, B K=3 (budget-forced after 2), C K=1, "
            "32 episodes/update (C 16 root + 5 ES2 + 11 ES3 = 69 transitions), "
            "economics 2 replicates x 200 episodes per mode",
}

ECONOMICS: Dict[str, Any] = {
    "modes": {"0": "mean_policy_selfplay", "1": "stochastic_policy_selfplay"},
    "replicates": 3, "episodes_per_replicate": 200000, "chunk_size": 10000,
    "seed_base": 9005000, "seed_tag": 100,
    "seed_components": ["seed_base", "seed", "q", "seed_tag", "mode_id", "rep_id", "stream_id"],
    "streams": {"0": "environment", "1": "player0_actions", "2": "player1_actions"},
    "gap_quantile_method": "histogram_linear_width10",
}


def derived(spec: GameSpec, bin_width: float, dense_step: float) -> Dict[str, Any]:
    """Quantities derived from GameSpec and the verifier grid functions.

    Args:
        spec: Game specification.
        bin_width: Training/visitation bin width.
        dense_step: Dense profile / concentration step.

    Returns:
        Dict of domains, bins, grid sizes, dense point counts.
    """
    sampler = StartSampler(spec, bin_width)
    T = spec.T
    dense = {}
    for t in range(1, T + 1):
        half = spec.domain_half(t)
        dense[str(t)] = 1 if t == 1 else 2 * int(round(half / dense_step)) + 1
    return {
        "dw": spec.dw, "B": spec.B, "e_range": spec.e_range,
        "domains": {str(t): [-spec.domain_half(t), spec.domain_half(t)] for t in range(1, T + 1)},
        "bin_width": bin_width,
        "bins": {str(t): (1 if t == 1 else sampler.n_bins(t)) for t in range(1, T + 1)},
        "dev_grid_points": {str(t): int(stage_grid(t, spec.B, DEV_CONFIG.state_step).size)
                            for t in range(1, T + 1)},
        "final_grid_points": {str(t): int(stage_grid(t, spec.B, FINAL_CONFIG.state_step).size)
                              for t in range(1, T + 1)},
        "dense_step": dense_step,
        "dense_points": dense,
        "dense_points_total": int(sum(dense.values())),
    }


def per_update_counts(phase: Dict[str, Any], T: int) -> Dict[str, Any]:
    """Episodes, per-stage learner transitions and joint steps per update."""
    sc = {int(s): int(n) for s, n in phase["start_counts"].items()}
    by_stage = {t: sum(n for s, n in sc.items() if s <= t) for t in range(1, T + 1)}
    return {"episodes": sum(sc.values()), "transitions_by_stage": {str(t): v for t, v in by_stage.items()},
            "joint_environment_steps": sum(by_stage.values()),
            "physical_actions": 2 * sum(by_stage.values())}


def expected_exposure(phase: Dict[str, Any], bins: Dict[str, int]) -> Dict[str, float]:
    """Expected direct-start count per bin per update (ES starts only)."""
    return {s: n / bins[s] for s, n in phase["start_counts"].items() if int(s) > 1}


def resolved_config(q: int, cohort: str, formal_settings: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, Any]:
    """Complete resolved training/evaluation config for one q and cohort.

    Args:
        q: Noise half-width (the only economic parameter that varies).
        cohort: ``smoke``, ``pilot`` or ``formal``.
        formal_settings: Selected formal settings (formal cohort only).

    Returns:
        Resolved config dict.
    """
    phases = copy.deepcopy(PHASES)
    econ = copy.deepcopy(ECONOMICS)
    overrides: Dict[str, Any] = {}
    fs: Dict[str, Any] = {}
    if cohort == "smoke":
        overrides = copy.deepcopy(SMOKE_OVERRIDES)
        for ph in phases:
            ph.update(copy.deepcopy(overrides["phases"][ph["name"]]))
        econ.update(overrides["economics"])
    elif cohort == "formal":
        if formal_settings is None:
            raise ValueError("formal cohort requires formal settings")
        fs = copy.deepcopy(formal_settings["resolved_non_q"])
        if fs.get("smoke_overrides"):
            raise ValueError("formal settings must not carry smoke overrides")
        if "game" in fs or "derived" in fs:
            raise ValueError("formal settings must not set game/derived values (q is the only game input)")
    elif cohort != "pilot":
        raise ValueError(f"unknown cohort {cohort!r}")

    game = dict(GAME_BASE, q=float(q))
    spec = GameSpec(**game)
    bin_width = 10.0
    dense_step = 0.05
    der = derived(spec, bin_width, dense_step)
    cfg = {
        "game": game,
        "derived": der,
        "ppo": copy.deepcopy(PPO),
        "phases": phases,
        "concentration_threshold": 0.04,
        "snapshot_every": 20,
        "weights_every": 25,
        "es_bin_width": bin_width,
        "rng": {"namespaces": {"init": 0, "env_noise": 1, "learner_action": 2, "opponent_action": 3,
                               "starts_roles": 4, "minibatch": 5},
                "seed_components": ["seed", "q", "namespace"]},
        "verifier": {
            "development": {"name": "development", "state_step": 4.0, "effort_step": 1.0, "gl_half": 16,
                            "grid_tol": 1e-9, "mass_tol": 1e-10, "pdl_tol_over_dw": 1e-10},
            "final": {"name": "final", "state_step": 2.0, "effort_step": 0.5, "gl_half": 32,
                      "grid_tol": 1e-9, "mass_tol": 1e-10, "pdl_tol_over_dw": 1e-10},
        },
        "final_certification": {
            "main_dreach_final_thr_over_dw": 0.01, "refine_dreach_thr_over_dw": 0.002,
            "refine_exp_root_thr_over_dw": 0.002, "dense_conc_step": dense_step,
            "dense_conc_thr": 0.04, "requires_candidate": True,
            "endpoint_rule": "first eligible C development call (candidate) else C-cap terminal "
                             "(diagnostic_terminal); no promotion of other checkpoints",
        },
        "economics": econ,
        "coverage": {"bin_width": bin_width, "tail_fraction_of_domain_half": 0.8, "tolerance": 1e-9},
        "runtime": {"device": "cpu", "torch_threads": 1,
                    "thread_env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                   "OPENBLAS_NUM_THREADS": "1"}},
        "smoke_overrides": overrides,
        "obs_encoding": "tau=(t-1)/(T-1); dnorm_1=0; dnorm_t=d/((t-1)B); float64 -> float32",
        "dtypes": {"network": "float32", "gaps_rewards_verifier_statistics": "float64"},
    }
    for key, val in fs.items():          # formal: every selected non-game setting is applied
        cfg[key] = copy.deepcopy(val)
    for ph in cfg["phases"]:
        ph["per_update"] = per_update_counts(ph, spec.T)
        ph["expected_direct_es_per_bin_per_update"] = expected_exposure(ph, der["bins"])
    return cfg


def excluded_seeds(cohort: str) -> Dict[str, List[int]]:
    """Seeds reserved or exposed outside ``cohort``."""
    out: Dict[str, List[int]] = {"historical_t3_debug": list(HISTORICAL_T3_SEEDS)}
    for name, by_q in SEEDS.items():
        if name != cohort:
            out[name] = sorted(s for seeds in by_q.values() for s in seeds)
    return out


def build_manifest(cohort: str, q: int, seeds: List[int],
                   formal_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Independent, complete manifest for one cohort and q.

    Args:
        cohort: ``smoke``, ``pilot`` or ``formal``.
        q: Noise half-width.
        seeds: Pre-specified seeds for this cohort and q.
        formal_settings: Selected formal settings (formal only).

    Returns:
        Manifest dict.

    Raises:
        ValueError: On a seed that is reserved/exposed elsewhere, or duplicates.
    """
    seeds = [int(s) for s in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds")
    excl = excluded_seeds(cohort)
    clash = {name: sorted(set(seeds) & set(v)) for name, v in excl.items() if set(seeds) & set(v)}
    if clash:
        raise ValueError(f"seeds used/reserved elsewhere: {clash}")
    cfg = resolved_config(q, cohort, formal_settings)
    name = f"{cohort}_q{q}"
    runs = [{"run_id": f"t3_{cohort}_q{q}_s{s}", "cohort": cohort, "q": int(q), "seed": s,
             "output_dir": f"runs/t3_{cohort}_q{q}_s{s}"} for s in seeds]
    m = {
        "experiment": EXPERIMENT,
        "manifest_name": name,
        "cohort": cohort,
        "q": int(q),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan": str(PLAN),
        "paths": {"root": str(ROOT), "worktree_W": str(W), "experiment_E": str(E),
                  "python": str(PYTHON), "runner": "run_experiment.py (relative to E)",
                  "runs_dir": "runs (relative to E)"},
        "seeds": seeds,
        "excluded_seeds": excl,
        "runs": runs,
        "max_workers_default": 3 if cohort != "formal" else None,
        "resolved": cfg,
        "formal_settings_source": (formal_settings or {}).get("source_path"),
    }
    check_manifest(m)
    return m


def check_manifest(m: Dict[str, Any]) -> None:
    """Consistency assertions on a manifest (q single source, counts, no legacy fields)."""
    cfg = m["resolved"]
    assert float(m["q"]) == float(cfg["game"]["q"]), "top-level q differs from game.q"
    for r in m["runs"]:
        assert int(r["q"]) == int(m["q"])
    names = [p["name"] for p in cfg["phases"]]
    expected = cfg["study"].get("phases_run", ["A"]) if cfg.get("study") else ["A", "B", "C"]
    assert names == expected, names
    text = json.dumps(cfg)
    for legacy in ("warmup", "stability_every", "B1", "A3", "B2", "verifier_timeout", "k_stop"):
        assert f'"{legacy}"' not in text, f"legacy field {legacy} present"
    for p in cfg["phases"]:
        assert p["cap"] % p["check_every"] == 0, "cap must be a multiple of the cadence"


def build_a_sampling_manifest(cohort: str, q: int, seeds: List[int], arm: str) -> Dict[str, Any]:
    """A-only manifest for one arm of the Phase-A state-sampling study.

    Everything except the A start mixture is the pilot configuration (A400,
    cadence 100, PPO, verifier, thresholds). Economics is disabled because
    stages 1-2 are untrained in an A-only run.
    """
    if cohort not in A_SAMPLING_COHORTS or arm not in A_SAMPLING_ARMS:
        raise ValueError(f"unknown A-sampling cohort/arm {cohort!r}/{arm!r}")
    seeds = [int(s) for s in seeds]
    excl = excluded_seeds(cohort)
    clash = {name: sorted(set(seeds) & set(v)) for name, v in excl.items() if set(seeds) & set(v)}
    if clash or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds used/reserved elsewhere or duplicated: {clash}")
    cfg = resolved_config(q, "pilot")
    spec = GameSpec(**cfg["game"])
    A = copy.deepcopy(next(p for p in cfg["phases"] if p["name"] == "A"))
    overrides: Dict[str, Any] = {}
    if cohort == "asamp_smoke":
        overrides = {"A": {"cap": 2, "check_every": 1, "start_counts": {"3": 32}},
                     "note": "smoke only: A cap 2, cadence 1, 32 episodes/update (center arm 16 + 16)"}
        A.update(copy.deepcopy(overrides["A"]))
    n = int(A["start_counts"]["3"])
    if arm == "center":
        A["center_es"] = {"3": n // 2}
    nb = cfg["derived"]["bins"]["3"]
    n_center_bins = int(round(2 * spec.B / cfg["es_bin_width"]))
    nc = int(A.get("center_es", {}).get("3", 0))
    A["per_update"] = per_update_counts(A, spec.T)
    A["expected_direct_es_per_bin_per_update"] = {"3": {
        "outer_bin": (n - nc) / nb, "center_bin": (n - nc) / nb + (nc / n_center_bins if nc else 0.0),
        "n_center_bins": n_center_bins, "n_bins": nb,
        "expected_center_share": ((n - nc) * n_center_bins / nb + nc) / n}}
    cfg["phases"] = [A]
    cfg["economics"] = dict(cfg["economics"], enabled=False,
                            disabled_reason="A-only study: stages 1-2 untrained; root self-play not meaningful")
    cfg["smoke_overrides"] = overrides
    cfg["study"] = dict(copy.deepcopy(A_SAMPLING_STUDY), arm=arm, center_es=A.get("center_es", {}),
                        center_half_width=spec.B)
    name = f"{cohort}_q{q}_{arm}"
    runs = [{"run_id": f"t3_{cohort}_q{q}_{arm}_s{s}", "cohort": cohort, "q": int(q), "seed": s, "arm": arm,
             "output_dir": f"runs/t3_{cohort}_q{q}_{arm}_s{s}"} for s in seeds]
    m = {
        "experiment": EXPERIMENT, "manifest_name": name, "cohort": cohort, "q": int(q), "arm": arm,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "plan": str(PLAN),
        "paths": {"root": str(ROOT), "worktree_W": str(W), "experiment_E": str(E), "python": str(PYTHON),
                  "runner": "run_experiment.py (relative to E)", "runs_dir": "runs (relative to E)"},
        "seeds": seeds, "excluded_seeds": excl, "runs": runs, "max_workers_default": 3, "resolved": cfg,
    }
    check_manifest(m)
    return m


def build_abc_arm_manifest(cohort: str, q: int, seeds: List[int], arm: str) -> Dict[str, Any]:
    """Full A/B/C manifest for one arm of the paired validation (phase A sampling differs only)."""
    if cohort not in ABC_COHORTS or arm not in A_SAMPLING_ARMS:
        raise ValueError(f"unknown A/B/C cohort/arm {cohort!r}/{arm!r}")
    seeds = [int(s) for s in seeds]
    excl = excluded_seeds(cohort)
    clash = {name: sorted(set(seeds) & set(v)) for name, v in excl.items() if set(seeds) & set(v)}
    if clash or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds used/reserved elsewhere or duplicated: {clash}")
    cfg = resolved_config(q, "smoke" if cohort == "asamp_abc_smoke" else "pilot")
    spec = GameSpec(**cfg["game"])
    A = next(p for p in cfg["phases"] if p["name"] == "A")
    n = int(A["start_counts"]["3"])
    if arm == "center":
        A["center_es"] = {"3": n // 2}
    nc = int(A.get("center_es", {}).get("3", 0))
    nb = cfg["derived"]["bins"]["3"]
    n_center_bins = int(round(2 * spec.B / cfg["es_bin_width"]))
    A["expected_direct_es_per_bin_per_update"] = {"3": {
        "outer_bin": (n - nc) / nb, "center_bin": (n - nc) / nb + (nc / n_center_bins if nc else 0.0),
        "n_center_bins": n_center_bins, "n_bins": nb,
        "expected_center_share": ((n - nc) * n_center_bins / nb + nc) / n}}
    cfg["study"] = dict(copy.deepcopy(ABC_STUDY), arm=arm, center_es_phase_A=A.get("center_es", {}),
                        center_half_width=spec.B)
    name = f"{cohort}_q{q}_{arm}"
    runs = [{"run_id": f"t3_{cohort}_q{q}_{arm}_s{s}", "cohort": cohort, "q": int(q), "seed": s, "arm": arm,
             "output_dir": f"runs/t3_{cohort}_q{q}_{arm}_s{s}"} for s in seeds]
    m = {
        "experiment": EXPERIMENT, "manifest_name": name, "cohort": cohort, "q": int(q), "arm": arm,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "plan": str(PLAN),
        "paths": {"root": str(ROOT), "worktree_W": str(W), "experiment_E": str(E), "python": str(PYTHON),
                  "runner": "run_experiment.py (relative to E)", "runs_dir": "runs (relative to E)"},
        "seeds": seeds, "excluded_seeds": excl, "runs": runs, "max_workers_default": 3, "resolved": cfg,
    }
    check_manifest(m)
    return m


def manifest_path(cohort: str, q: int) -> Path:
    """Default manifest location."""
    return E / "manifests" / f"{cohort}_q{q}.json"


def write_manifest(m: Dict[str, Any], path: Path) -> str:
    """Write a manifest; an existing different manifest is never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text())
        a = {k: v for k, v in old.items() if k != "created_utc"}
        b = json.loads(json.dumps({k: v for k, v in m.items() if k != "created_utc"}))
        if a == b:
            return "unchanged"
        raise FileExistsError(f"{path} exists with different content; not overwritten")
    write_json_atomic(path, m)
    return "written"


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort", required=True,
                   choices=["smoke", "pilot", "formal"] + list(A_SAMPLING_COHORTS) + list(ABC_COHORTS))
    p.add_argument("--settings", default=None, help="formal only: selected formal_settings.json")
    p.add_argument("--seeds-per-q", type=int, default=None, help="formal only: 10 or 20")
    p.add_argument("--output-dir", type=Path, default=E / "generated_manifests", help="directory for newly generated manifests")
    p.add_argument("--runs-prefix", default=str(E / "reruns"), help="new run directories under this path (relative to working directory, or absolute)")
    args = p.parse_args()
    def output_manifest(m):
        for run in m["runs"]:
            run["output_dir"] = str(Path(args.runs_prefix).resolve() / run["run_id"])
        path = args.output_dir / (m["manifest_name"] + ".json")
        print(f"{path}: {write_manifest(m, path)} seeds={m['seeds']}")
    fs = None
    if args.cohort in ABC_COHORTS:
        for q in SEEDS[args.cohort]:
            for arm in A_SAMPLING_ARMS:
                m = build_abc_arm_manifest(args.cohort, q, SEEDS[args.cohort][q], arm)
                output_manifest(m)
        return 0
    if args.cohort in A_SAMPLING_COHORTS:
        for q in SEEDS[args.cohort]:
            for arm in A_SAMPLING_ARMS:
                m = build_a_sampling_manifest(args.cohort, q, SEEDS[args.cohort][q], arm)
                output_manifest(m)
        return 0
    if args.cohort == "formal":
        if not args.settings or args.seeds_per_q not in (10, 20):
            raise SystemExit("formal requires --settings and --seeds-per-q {10,20}")
        fs = json.loads(Path(args.settings).read_text())
        fs["source_path"] = str(Path(args.settings).resolve())
        if int(fs.get("seeds_per_q", -1)) != args.seeds_per_q:
            raise SystemExit("--seeds-per-q differs from the N recorded in formal settings")
    for q in Q_VALUES:
        seeds = SEEDS[args.cohort][q]
        if args.cohort == "formal":
            seeds = seeds[:args.seeds_per_q]
        m = build_manifest(args.cohort, q, seeds, fs)
        output_manifest(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
