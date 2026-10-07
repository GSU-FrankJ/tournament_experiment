#!/usr/bin/env python3
"""MS-R1 pipeline: one stage per phase, exact expected continuation, development stop rule,
verifier-guided coverage-constrained exploring starts, targeted polishing (T-generic, T in {2, 3}).

Phases run t = T, T-1, ..., 1. In the phase of stage t the learner rolls out stage t only
(``run.v2_rollout.collect_batch_v2`` for t = T, exactly as v2.0's Phase A; ``run.ms_rollout`` for
t < T with the return -k e^2 + V~_{t+1}(d + e - e_opp) from the nested table of the frozen suffix,
``utils.ms_continuation``). Later stages are frozen snapshots (deep copies taken at their freeze).
Each phase is run by ``utils.ms_rule.StageController``: blocks at constant LR, a development check every
K local updates and at every block end, development stop after M consecutive eligible checks, targeted
polishing of a localized residual, landing window with the LR decaying linearly, freeze. The rule and the
sampler read only the D3 quantities of ``utils.ms_residual`` (verifier arrays); the closed form enters
the evaluation and reporting only.

Legacy settings (``rule.enabled = false``, ``start_weights = {scheme: bin_balanced}``, fixed budget and
v2.0's LR windows) make the terminal-stage phase consume every stream exactly as v2.0's Phase A
(``run/run_v2_stagewise.py`` mode phase_A): check C-MS1.

MS-R2 optional config keys (all absent == the MS-R1 behaviour, bit for bit):
  conc_scale_schedule  {stage, local_first, local_last, scale_first, scale_last}: the concentration scale of the
                       Beta actor (``BetaActor.conc_scale``; the mean does not depend on it) is, before local
                       update j of ``stage``, scale_first for j <= local_first, linear in j up to scale_last at
                       local_last, scale_last afterwards (the form of ``run_v2_stagewise.Run.conc_scale_for``). It
                       is set on the live actor and the lagged opponent before every update with j > local_first
                       (constant within the update) and reset to 1.0 at the entry of every other phase (before
                       the phase-entry snapshot refresh); the snapshot refresh and the frozen snapshot carry it.
  fixed_global_sampler true: with ``rule.enabled = false`` and ``stratified_priority`` starts, every terminal-stage
                       update draws from the sampler of an MS-R1 global block (alpha = alpha_global, focus = the
                       EMA of the per-bin residual map, updated at every valid check); no classification, no
                       polishing, the fixed budget and LR windows of the legacy arm.
  noise_report         true: reporting columns (closed form allowed, never read by anything): ``conc_scale`` per
                       update in ms_updates.csv; ``conc_scale, sigma_0, e_sigma_0, g2_0, smoothing, remainder,
                       gap, R0`` at every check of the terminal stage (T = 2); the decomposition and R0 on both
                       tiers in the freeze record.

MS-R3 optional config keys (absent == the MS-R2 behaviour, bit for bit):
  actor_variant        "relu" (ReLU hidden units) or "t10" (tanh units on 10 * d / ((t - 1) B); the stage feature is
                       not scaled): the actor of the whole run (``BetaActor.variant``; the initial weights are drawn
                       by the same generator calls as the locked actor, so they are identical across variants).
                       Absent == "t1", the locked tanh actor on d / ((t - 1) B); "t1" is not accepted as a value so
                       that a t1 run writes no variant field anywhere. The variant is set on the live actor and the
                       lagged opponent before the first update and carried by every snapshot, weight export
                       (``actor_variant`` entry) and full state.
  init_digest          true: ``manifest.json`` records ``init_state_sha256``, the SHA-256 of the initial actor and
                       critic weights (before any update; check C-INIT).

Outputs in --out-dir: run_config.json, manifest.json, status.json, ms_checks_stage{t}.csv,
ms_binmaps_stage{t}.npz, ms_updates.csv, rule_log.json, gates.json, state_end_stage{t}.pt,
continuation_table_stage{t}.npz (t = T-1 ... 1), freeze_stage{t}_{final,development}.npz, weights/,
train_history.json, drift_test.json, induced_band.json + band_sweep.npz (T = 2), ms_run_summary.json.

Exit codes: 0 done, 1 pipeline exception, 3 refusing to restart, 4 config error, 5 process-global RNG
violation (the run finished; it counts as failed).

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python run/run_ms_stagewise.py --config <run_config.json> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import random
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.ppo_curriculum import PPOConfig  # noqa: E402
from agents.ppo_curriculum_v2 import CurriculumPPOv2  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402
from run.ms_rollout import collect_stage_batch  # noqa: E402
from run.run_final_dp_br import Costs, dense_grid, make_policy_fns, write_json  # noqa: E402
from run.run_final_dp_br_round3_dense import (  # noqa: E402
    LR_SCHEDULE_REQUIRED_KEYS, PROTOCOL_REQUIRED_KEYS, ConfigError, lr_at, read_lr, set_lr,
    strict_dataclass, strict_keys)
from run.run_v2_stagewise import git_state, rng_position  # noqa: E402
from run.run_v2_T2_locked import (  # noqa: E402
    GLOBAL_RNGS, REPORT_KEYS_A, REPORT_KEYS_F, RNG_VIOLATION_EXIT, decomposition, diff, digests,
    global_states, pick, rng_snapshot, seed_globals, smoothed_share, stage2_extra, verdicts,
    write_table_npz)
from run.v2_rollout import collect_batch_v2  # noqa: E402
from utils.dp_br_verifier import (  # noqa: E402
    DEV_CONFIG, FINAL_CONFIG, DomainError, VerifierConfig, concentration_stats)
from utils.ms_continuation import build_table  # noqa: E402
from utils.ms_noise import tie_noise_report, tie_residual_R0  # noqa: E402
from utils.ms_residual import StageDiag, invalid_diag, stage_diag  # noqa: E402
from utils.ms_rule import StageController, StageRule  # noqa: E402
from utils.theory_multistage import validate_two_stage_params  # noqa: E402
from utils.v2_metrics import V2Eval, append_csv, evaluate, save_npz  # noqa: E402

SCHEMA = "ms_run_config/1"
REQUIRED = ("schema", "base_commit", "pilot", "arm", "run", "q", "seed", "record", "protocol_ref",
            "record_sha256", "protocol_gates", "threads_per_process", "pipeline", "start_weights", "rule",
            "clamp_likelihood", "full_state_at")
OPTIONAL = ("derived", "conc_scale_schedule", "fixed_global_sampler", "noise_report", "actor_variant", "init_digest")
CONC_SCHEDULE_KEYS = ("stage", "local_first", "local_last", "scale_first", "scale_last")
CHECK_NOISE = ("conc_scale", "sigma_0", "e_sigma_0", "g2_0", "smoothing", "remainder", "gap", "R0")
START_SCHEMES = ("bin_balanced", "stratified_priority")
STRAT_KEYS = ("scheme", "lambda_P", "alpha_global", "alpha_polish", "ema_beta", "near_tie_half_width")
STRAT_OPTIONAL = ("lambda_T",)
RULE_KEYS = ("enabled", "K", "M", "conc_limit", "localized_fraction", "stages")
STAGE_KEYS_ENABLED = ("eps", "rho", "tau", "n_block", "u_cap", "n_land")
STAGE_KEYS_LEGACY = ("eps", "rho", "tau")      # the legacy arm records the would-fire update only
PIPELINE_KEYS = ("T", "phases", "budgets", "lr_windows")
LR_WIN_KEYS = ("first", "last", "start", "end")
GATE_SECTIONS = ("gates", "secondary", "stage1_decomposition")
CONT_TABLE_STEP = 0.05
LANDING_LR_END = 3e-5                  # D4: the landing window decays the LR from ab_lr to this value
REPORT_NEAR_TIE_HALF_WIDTH = 20.0      # strata used to MEASURE start shares when the scheme is bin_balanced
CHECK_META = ("run", "arm", "q", "seed", "stage", "update", "local", "mode", "block_id", "block_type",
              "lr", "alpha")
CHECK_D3 = ("valid", "Delta", "s", "R", "R_tail", "tail_term", "C", "R_argmax_d", "eligible",
            "consecutive", "would_fire_local")
CHECK_SAMPLER = ("p_digest", "p_digest_next")
CHECK_REPORT = ("stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_rmse_pos_over_g2_0",
                "stage2_tail_mean_over_g2_0", "stage2_tail_max_over_g2_0", "e1_at_0", "e2_at_0",
                "stage1_rel_err_signed", "eta_T_over_dw", "Gmax_full_over_dw")
CHECK_TAIL = ("verifier_sec", "error")
CHECK_COLS = CHECK_META + CHECK_D3 + CHECK_SAMPLER + CHECK_REPORT + CHECK_TAIL
FREEZE_KEYS = tuple(dict.fromkeys(REPORT_KEYS_A + REPORT_KEYS_F + (
    "stage2_peak_locfree_rel_err", "stage1_rel_err_signed", "e1_at_0", "e2_at_0", "g1", "g2_at_0",
    "stage2_tail_max_over_g2_0", "stage2_rmse_pos", "stage2_tail_mean", "sigma_effort_at_0_t2",
    "sigma_effort_at_0_t1", "G_max_t1_over_dw", "G_max_t2_over_dw", "Delta_max_t1_over_dw",
    "Delta_max_t2_over_dw", "invalid_reasons")))


# ---------------------------------------------------------------------------------- config
def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def derive_start_info(spec: GameSpec, sw: Dict[str, Any], bin_width: float) -> Dict[str, Dict[str, Any]]:
    """Strata counts and shares per stage t >= 2 for the stratified scheme (D5).

    Returns ``{str(t): {n_bins, n_tail, n_near, n_mid, lambda_T, lambda_P, lambda_M}}``.

    Raises:
        ValueError: If the coverage constraint or lambda_M > 0 is violated, or a stratum is empty.
    """
    sampler = StartSampler(spec, bin_width)
    out: Dict[str, Dict[str, Any]] = {}
    for t in range(2, spec.T + 1):
        sampler.stratified_bin_probs(t, float(sw["lambda_P"]), float(sw["near_tie_half_width"]))
        st = sampler.strata(t, float(sw["near_tie_half_width"]))
        n = sampler.n_bins(t)
        n_tail = int(st["tail"].sum())
        lam_t = n_tail / n
        out[str(t)] = {"n_bins": n, "n_tail": n_tail, "n_near": int(st["near"].sum()),
                       "n_mid": int(st["mid"].sum()), "lambda_T": lam_t,
                       "lambda_P": float(sw["lambda_P"]), "lambda_M": 1.0 - float(sw["lambda_P"]) - lam_t}
    return out


def validate_config(cfg: Dict[str, Any]) -> None:
    """Refuse a config with a missing / unknown key or an invalid parameter (ConfigError)."""
    missing = [k for k in REQUIRED if k not in cfg]
    unknown = [k for k in cfg if k not in REQUIRED + OPTIONAL]
    if missing or unknown:
        raise ConfigError(f"run config: missing {missing}, unknown {unknown}")
    if cfg["schema"] != SCHEMA:
        raise ConfigError(f"schema {cfg['schema']!r} != {SCHEMA!r}")
    rec = cfg["record"]
    pl = cfg["pipeline"]
    if not isinstance(pl, dict) or set(pl) != set(PIPELINE_KEYS):
        raise ConfigError(f"pipeline must have exactly the keys {PIPELINE_KEYS}")
    T = pl["T"]
    if isinstance(T, bool) or T not in (2, 3):
        raise ConfigError(f"pipeline.T must be 2 or 3; got {T!r}")
    if pl["phases"] != list(range(T, 0, -1)):
        raise ConfigError(f"pipeline.phases must be {list(range(T, 0, -1))} (the phases run t = T, ..., 1)")
    if int(rec["game"]["T"]) != T:
        raise ConfigError(f"record game.T {rec['game']['T']} != pipeline.T {T}")
    if int(cfg["seed"]) != int(rec["seed"]) or float(cfg["q"]) != float(rec["q"]):
        raise ConfigError("q/seed differ from the embedded record")
    if cfg["clamp_likelihood"] != "density":
        raise ConfigError("clamp_likelihood must stay 'density'")
    if not isinstance(cfg["full_state_at"], list):
        raise ConfigError("full_state_at must be a list of global updates")
    pg = cfg["protocol_gates"]
    if not isinstance(pg, dict) or set(pg) != set(GATE_SECTIONS):
        raise ConfigError(f"protocol_gates must have exactly {GATE_SECTIONS}")
    spec = strict_dataclass(GameSpec, rec["game"], "game")
    bin_w = float(rec["protocol"]["es_bin_width"])
    # ---- MS-R2 optional keys (absent == the MS-R1 behaviour)
    cs = cfg.get("conc_scale_schedule")
    if cs is not None:
        if not isinstance(cs, dict) or set(cs) != set(CONC_SCHEDULE_KEYS):
            raise ConfigError(f"conc_scale_schedule must have exactly {CONC_SCHEDULE_KEYS}")
        if isinstance(cs["stage"], bool) or not isinstance(cs["stage"], int) or not 1 <= cs["stage"] <= T:
            raise ConfigError(f"conc_scale_schedule.stage must be an int in 1..{T}")
        for k_ in ("local_first", "local_last"):
            if isinstance(cs[k_], bool) or not isinstance(cs[k_], int):
                raise ConfigError(f"conc_scale_schedule.{k_} must be an int")
        if not 1 <= cs["local_first"] < cs["local_last"]:
            raise ConfigError("conc_scale_schedule needs 1 <= local_first < local_last")
        if not (_num(cs["scale_first"]) and _num(cs["scale_last"]) and cs["scale_first"] > 0 and cs["scale_last"] > 0):
            raise ConfigError("conc_scale_schedule scales must be positive numbers")
    if "noise_report" in cfg:
        if not isinstance(cfg["noise_report"], bool):
            raise ConfigError("noise_report must be a bool")
        if cfg["noise_report"] and T != 2:
            raise ConfigError("noise_report is defined for T = 2 (the closed-form tie value) only")
    if "actor_variant" in cfg and cfg["actor_variant"] not in ("relu", "t10"):
        raise ConfigError("actor_variant, if present, must be 'relu' or 't10' (absent = the locked actor 't1')")
    if "init_digest" in cfg and cfg["init_digest"] is not True:
        raise ConfigError("init_digest, if present, must be true")
    if "fixed_global_sampler" in cfg:
        if cfg["fixed_global_sampler"] is not True:
            raise ConfigError("fixed_global_sampler, if present, must be true")
        if cfg["rule"]["enabled"] is not False or cfg["start_weights"].get("scheme") != "stratified_priority":
            raise ConfigError("fixed_global_sampler needs rule.enabled = false and stratified_priority starts")
    # ---- start weights
    sw = cfg["start_weights"]
    if not isinstance(sw, dict) or sw.get("scheme") not in START_SCHEMES:
        raise ConfigError(f"start_weights.scheme must be one of {START_SCHEMES}")
    if sw["scheme"] == "bin_balanced":
        if set(sw) != {"scheme"}:
            raise ConfigError("start_weights scheme bin_balanced takes no other key")
    else:
        if not set(STRAT_KEYS) <= set(sw) <= set(STRAT_KEYS) | set(STRAT_OPTIONAL):
            raise ConfigError(f"start_weights stratified_priority must have exactly {STRAT_KEYS} "
                              f"(optionally {STRAT_OPTIONAL}); got {sorted(sw)}")
        for k in STRAT_KEYS[1:]:
            if not _num(sw[k]):
                raise ConfigError(f"start_weights.{k} must be a number")
        if not 0.0 < sw["lambda_P"] < 1.0:
            raise ConfigError("start_weights.lambda_P must lie in (0, 1)")
        if not (0.0 <= sw["alpha_global"] <= 1.0 and 0.0 <= sw["alpha_polish"] <= 1.0):
            raise ConfigError("start_weights alphas must lie in [0, 1]")
        if not 0.0 <= sw["ema_beta"] < 1.0:
            raise ConfigError("start_weights.ema_beta must lie in [0, 1)")
        if not sw["near_tie_half_width"] > 0.0:
            raise ConfigError("start_weights.near_tie_half_width must be positive")
        try:
            derived = derive_start_info(spec, sw, bin_w)
        except ValueError as exc:
            raise ConfigError(f"start_weights: {exc}") from exc
        if "lambda_T" in sw:
            if T != 2:
                raise ConfigError("start_weights.lambda_T is accepted for T = 2 only (it is derived per stage)")
            want = derived["2"]["lambda_T"]
            if not _num(sw["lambda_T"]) or abs(float(sw["lambda_T"]) - want) > 1e-12:
                raise ConfigError(f"coverage constraint: lambda_T {sw['lambda_T']} != the bin-balanced "
                                  f"tail share {want}")
        if "derived" in cfg and cfg["derived"] != derived:
            raise ConfigError("derived start-weights record differs from the re-derived one")
    # ---- rule
    rl = cfg["rule"]
    if not isinstance(rl, dict) or set(rl) != set(RULE_KEYS):
        raise ConfigError(f"rule must have exactly {RULE_KEYS}")
    if not isinstance(rl["enabled"], bool):
        raise ConfigError("rule.enabled must be a bool")
    for k in ("K", "M"):
        if isinstance(rl[k], bool) or not isinstance(rl[k], int) or rl[k] < 1:
            raise ConfigError(f"rule.{k} must be a positive int")
    if not _num(rl["conc_limit"]) or not 0.0 < rl["conc_limit"] <= 1.0:
        raise ConfigError("rule.conc_limit must lie in (0, 1]")
    if not _num(rl["localized_fraction"]) or not 0.0 < rl["localized_fraction"] <= 1.0:
        raise ConfigError("rule.localized_fraction must lie in (0, 1]")
    stages = rl["stages"]
    if not isinstance(stages, dict) or set(stages) != {str(t) for t in range(1, T + 1)}:
        raise ConfigError(f"rule.stages must have exactly the keys {[str(t) for t in range(1, T + 1)]}")
    for t, st in stages.items():
        want_keys = STAGE_KEYS_ENABLED if rl["enabled"] else STAGE_KEYS_LEGACY
        if not isinstance(st, dict) or set(st) != set(want_keys):
            raise ConfigError(f"rule.stages[{t}] must have exactly {want_keys} "
                              f"(rule.enabled={rl['enabled']}: a fixed budget is the legacy arm's, an "
                              f"enabled rule takes none)")
        for k in ("eps", "rho", "tau"):
            if not _num(st[k]) or st[k] < 0.0:
                raise ConfigError(f"rule.stages[{t}].{k} must be a non-negative number")
        if rl["enabled"]:
            for k in ("n_block", "u_cap", "n_land"):
                if isinstance(st[k], bool) or not isinstance(st[k], int) or st[k] < 1:
                    raise ConfigError(f"rule.stages[{t}].{k} must be a positive int")
            if st["n_land"] < 2:
                raise ConfigError(f"rule.stages[{t}].n_land must be >= 2")
    if rl["enabled"]:
        if pl["budgets"] is not None or pl["lr_windows"] is not None:
            raise ConfigError("an enabled rule takes no fixed budget: pipeline.budgets and pipeline.lr_windows "
                              "must be null")
    else:   # the legacy arm: fixed budgets and v2.0's LR windows, explicit for every stage
        want = {str(t) for t in range(1, T + 1)}
        if not isinstance(pl["budgets"], dict) or set(pl["budgets"]) != want \
                or not isinstance(pl["lr_windows"], dict) or set(pl["lr_windows"]) != want:
            raise ConfigError(f"the legacy arm (rule.enabled false) needs pipeline.budgets and "
                              f"pipeline.lr_windows for the stages {sorted(want)}")
        for t in want:
            fb, wins = pl["budgets"][t], pl["lr_windows"][t]
            if isinstance(fb, bool) or not isinstance(fb, int) or fb < 1:
                raise ConfigError(f"pipeline.budgets[{t}] (fixed budget) must be a positive int")
            if not isinstance(wins, list):
                raise ConfigError(f"pipeline.lr_windows[{t}] must be a list")
            prev_last = None
            for w in sorted(wins, key=lambda w: int(w["first"]) if isinstance(w, dict) and "first" in w else 0):
                if not isinstance(w, dict) or set(w) != set(LR_WIN_KEYS):
                    raise ConfigError(f"lr window must have exactly {LR_WIN_KEYS}")
                if not 1 <= int(w["first"]) < int(w["last"]):
                    raise ConfigError("lr window needs 1 <= first < last")
                if prev_last is not None and int(w["first"]) != prev_last + 1:
                    raise ConfigError("lr windows must be contiguous and non-overlapping")
                prev_last = int(w["last"])
            if wins and prev_last != fb:
                raise ConfigError("the last lr window must end at the fixed budget")
    if rl["enabled"] and sw["scheme"] != "stratified_priority":
        raise ConfigError("an enabled rule (targeted polishing) needs start_weights.scheme = "
                          "stratified_priority")


# ---------------------------------------------------------------------------------- run
class MSRun:
    """One MS-R1 run: state, phase loop, freezes (see the module docstring)."""

    def __init__(self, cfg: Dict[str, Any], out_dir: str):
        validate_config(cfg)
        self.cfg = cfg
        self.out_dir = out_dir
        rec = cfg["record"]
        self.rec = rec
        self.seed = int(cfg["seed"])
        versions_actual = {"python": platform.python_version(), "torch": torch.__version__,
                           "numpy": np.__version__}
        if versions_actual != rec["versions"]:
            raise ConfigError(f"environment versions {versions_actual} != record {rec['versions']}")
        if rec["device"] != "cpu":
            raise ConfigError(f"record device {rec['device']} is not cpu")
        torch.set_num_threads(int(rec["torch_threads"]))
        self.thread_env = {k: os.environ.get(f"{k}_NUM_THREADS") for k in ("OMP", "MKL", "OPENBLAS")}
        for k, v in cfg["threads_per_process"].items():
            if k != "torch" and self.thread_env.get(k) != str(v):
                raise ConfigError(f"{k}_NUM_THREADS={self.thread_env.get(k)} but config requires {v}")
        self.versions_actual = versions_actual
        self.spec = strict_dataclass(GameSpec, rec["game"], "game")
        spec = self.spec
        self.T = int(spec.T)
        self.ppo_cfg = strict_dataclass(PPOConfig, rec["ppo"], "ppo")
        self.dev_cfg = strict_dataclass(VerifierConfig, rec["verifier"]["development"], "verifier.development")
        self.fin_cfg = strict_dataclass(VerifierConfig, rec["verifier"]["final"], "verifier.final")
        if self.dev_cfg != DEV_CONFIG or self.fin_cfg != FINAL_CONFIG:
            raise ConfigError("record verifier tiers differ from utils.dp_br_verifier DEV/FINAL constants")
        P = dict(rec["protocol"])
        strict_keys(P, PROTOCOL_REQUIRED_KEYS, "protocol")
        sched = dict(rec["lr_schedule"])
        strict_keys(sched, LR_SCHEDULE_REQUIRED_KEYS, "lr_schedule")
        if not sched["actor_and_critic"] or not sched["apply_before_each_update"] \
                or not sched["constant_within_update"] or not sched["preserve_adam_state"]:
            raise ConfigError(f"lr_schedule flags not all true: {sched}")
        if float(self.ppo_cfg.lr) != float(sched["ab_lr"]):
            raise ConfigError("ppo.lr != lr_schedule.ab_lr")
        for k_ in ("w_h", "w_l", "k", "e_min", "e_max"):
            if float(P[k_]) != float(getattr(spec, k_)):
                raise ConfigError(f"protocol.{k_} != game.{k_}")
        if float(rec["dw"]) != spec.dw or float(rec["B"]) != spec.B:
            raise ConfigError("record dw/B inconsistent with game")
        if rec["smoke_overrides"]:
            raise ConfigError("embedded record carries smoke_overrides")
        self.P = P
        self.sched = sched
        if self.T == 2:
            if float(rec["domain_half_stage2"]) != spec.domain_half(2):
                raise ConfigError("record domain_half_stage2 inconsistent with game")
            val = validate_two_stage_params(q=spec.q, w_h=spec.w_h, w_l=spec.w_l, k=spec.k, e_bar=spec.e_max)
            if not val.ok:
                raise ConfigError(f"q={spec.q} fails closed-form validity: {val.messages}")
        # ---- RNG streams: identical construction to run/run_v2_stagewise.py
        ns = P["rng_namespaces"]
        self.seed_namespaces = {name: [self.seed, int(spec.q), int(v)] for name, v in ns.items()}

        def _rng(name: str) -> np.random.Generator:
            return np.random.default_rng(np.random.SeedSequence([self.seed, int(spec.q), int(ns[name])]))
        init_state = np.random.SeedSequence([self.seed, int(spec.q), int(ns["init"])]).generate_state(1)[0]
        self.torch_gen = torch.Generator().manual_seed(int(init_state))
        self.rngs = {"env": _rng("env_noise"), "learn": _rng("learner_action"),
                     "opp": _rng("opponent_action"), "start": _rng("starts_roles")}
        rng_mb = _rng("minibatch")
        self.agent = CurriculumPPOv2(self.ppo_cfg, self.torch_gen, rng_mb, device=rec["device"])
        self.actor_variant = str(cfg.get("actor_variant", "t1"))
        self.init_state_sha256 = self._init_digest(self.agent) if cfg.get("init_digest") else None
        if self.actor_variant != "t1":
            self.agent.set_actor_variant(self.actor_variant)
        self.opt_ids = {"actor": id(self.agent.opt_actor), "critic": id(self.agent.opt_critic)}
        set_lr(self.agent, lr_at(sched, "A", 1))
        self.sampler = StartSampler(spec, P["es_bin_width"])
        if self.T == 2 and self.sampler.n_bins(2) != int(rec["es_bins_stage2"]):
            raise ConfigError("es bins differ from record")
        # ---- sampler / rule configuration
        self.start_weights = copy.deepcopy(cfg["start_weights"])
        self.stratified = self.start_weights["scheme"] == "stratified_priority"
        self.hw = float(self.start_weights["near_tie_half_width"]) if self.stratified \
            else REPORT_NEAR_TIE_HALF_WIDTH
        self.derived = derive_start_info(spec, self.start_weights, P["es_bin_width"]) if self.stratified else {}
        self.rule_cfg = copy.deepcopy(cfg["rule"])
        self.conc_sched = copy.deepcopy(cfg.get("conc_scale_schedule"))
        self.fixed_global = bool(cfg.get("fixed_global_sampler", False))
        self.noise_report = bool(cfg.get("noise_report", False))
        self.stage_rules = {t: self._stage_rule(t) for t in range(1, self.T + 1)}
        # ---- run state
        self.costs = Costs()
        self.history: List[Dict[str, Any]] = []
        self.rows: List[Dict[str, Any]] = []
        self.snapshot_log: List[Dict[str, Any]] = [{"update": 0, "reason": "init"}]
        self.weights_log: List[Dict[str, Any]] = []
        self.phase_timing: Dict[str, Dict[str, Any]] = {}
        self.rule_log: Dict[str, Any] = {}
        self.freeze_records: Dict[str, Any] = {}
        self.scal_freeze: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.table_records: Dict[str, Any] = {}
        self.tables: Dict[int, Any] = {}
        self.frozen_nets: Dict[int, Any] = {}
        self.frozen_digest: Dict[int, str] = {}
        self.freeze_evals: Dict[int, Dict[str, V2Eval]] = {}
        self.total_episodes = 0
        self.total_transitions = 0
        self.global_u = 0
        self.phases_done: List[str] = []
        self.weights_every = int(P["weights_every"])
        self.weights_dir = os.path.join(out_dir, "weights")
        self.full_state_at = {int(u) for u in cfg.get("full_state_at", [])}
        self.phase_start_hook: Optional[Callable[[int], None]] = None
        self.table_hook: Optional[Callable[[int, str], None]] = None
        self.violations: List[Dict[str, str]] = []

    # ------------------------------------------------------------------ rule / lr
    def lr_linear(self, start: float, end: float, first: int, last: int, local: int) -> float:
        """The linear branch of ``lr_at`` (the form of ``Run.lr_for``): start + (end - start) (j - first)/den."""
        lin = dict(self.sched, kind="linear", c_start_lr=float(start), c_end_lr=float(end),
                   c_local_first=int(first), linear_denominator=int(last) - int(first))
        return lr_at(lin, "C", local)

    def _legacy_lr(self, t: int) -> Callable[[int], float]:
        wins = sorted(self.cfg["pipeline"]["lr_windows"][str(t)], key=lambda w: int(w["first"]))

        def fn(j: int) -> float:
            dec = None
            for w in wins:
                if int(w["first"]) <= j <= int(w["last"]):
                    dec = w
            if dec is None:
                return lr_at(self.sched, "A", j)
            return self.lr_linear(dec["start"], dec["end"], dec["first"], dec["last"], j)
        return fn

    def conc_scale_for(self, local: int) -> float:
        """Concentration scale before local update ``local`` of the scheduled stage (the form of
        ``run_v2_stagewise.Run.conc_scale_for``: ``scale_first`` up to ``local_first``, linear, ``scale_last`` from
        ``local_last``)."""
        cs = self.conc_sched
        j0, j1 = int(cs["local_first"]), int(cs["local_last"])
        s0, s1 = float(cs["scale_first"]), float(cs["scale_last"])
        if local <= j0:
            return s0
        if local >= j1:
            return s1
        return s0 + (s1 - s0) * (local - j0) / (j1 - j0)

    def _stage_rule(self, t: int) -> StageRule:
        rc = self.rule_cfg
        st = rc["stages"][str(t)]
        sw = self.start_weights
        n_nontail = 0
        if t >= 2:
            n_nontail = self.sampler.n_bins(t) - int(self.sampler.tail_mask(t).sum())
        common = dict(stage=t, enabled=bool(rc["enabled"]), K=int(rc["K"]), M=int(rc["M"]),
                      eps=float(st["eps"]), rho=float(st["rho"]), tau=float(st["tau"]),
                      conc_limit=float(rc["conc_limit"]), loc_frac=float(rc["localized_fraction"]),
                      alpha_polish=float(sw.get("alpha_polish", 0.5)),
                      alpha_global=float(sw.get("alpha_global", 0.0)),
                      ema_beta=float(sw.get("ema_beta", 0.5)), lr_base=float(self.sched["ab_lr"]),
                      lr_end=LANDING_LR_END, n_nontail_bins=n_nontail)
        if rc["enabled"]:
            return StageRule(n_block=int(st["n_block"]), u_cap=int(st["u_cap"]), n_land=int(st["n_land"]),
                             **common)
        return StageRule(fixed_budget=int(self.cfg["pipeline"]["budgets"][str(t)]), **common)

    # ------------------------------------------------------------------ policies / verifier
    def policy_fns(self) -> Tuple[Callable, Callable]:
        """(mean_fn, beta_fn) of the composite candidate: frozen snapshots for the frozen stages, the live
        actor elsewhere (the untrained output below the stage being trained)."""
        live_mean, live_beta = make_policy_fns(self.agent, self.spec)
        if not self.frozen_nets:
            return live_mean, live_beta
        fz = {s: make_policy_fns(self.agent, self.spec, net=net) for s, net in self.frozen_nets.items()}

        def mean_fn(t: int, d: np.ndarray) -> np.ndarray:
            return fz[t][0](t, d) if t in fz else live_mean(t, d)

        def beta_fn(t: int, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return fz[t][1](t, d) if t in fz else live_beta(t, d)
        return mean_fn, beta_fn

    def verify_candidate(self, cfg: VerifierConfig) -> Tuple[Optional[V2Eval], Optional[str]]:
        """``utils.v2_metrics.evaluate`` on the current composite candidate (pure, no RNG)."""
        mean_fn, beta_fn = self.policy_fns()
        try:
            return evaluate(mean_fn, self.spec, cfg, beta_fn=beta_fn,
                            recovery_step=float(self.P["recovery_step"])), None
        except (DomainError, FloatingPointError, ValueError) as exc:
            return None, f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------ full state
    def full_state(self, phase_done: str) -> Dict[str, object]:
        """Complete branching state at a phase boundary (the keys of ``Run.full_state`` plus the snapshots)."""
        return {
            "format": "ms_full_state/1", "phase_done": phase_done, "phases_done": list(self.phases_done),
            "agent": self.agent.full_state(),
            "rng": {k: copy.deepcopy(g.bit_generator.state) for k, g in self.rngs.items()},
            "torch_generator_state": self.torch_gen.get_state(),
            "torch_global_rng_state": torch.get_rng_state(),
            "numpy_global_rng_state": np.random.get_state(), "python_random_state": random.getstate(),
            "counters": {"global_u": self.global_u, "total_episodes": self.total_episodes,
                         "total_transitions": self.total_transitions,
                         "snapshot_refreshes": self.agent.snapshot_refreshes},
            "snapshot_log": copy.deepcopy(self.snapshot_log),
            "schedule_positions": {"actor_lr": read_lr(self.agent)[0], "critic_lr": read_lr(self.agent)[1],
                                   "phase_done": phase_done},
            "frozen_stages": {str(s): {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                              for s, net in self.frozen_nets.items()},
            "seed": self.seed, "q": self.spec.q, "seed_namespaces": self.seed_namespaces,
            **({"actor_variant": self.actor_variant} if self.actor_variant != "t1" else {}),
        }

    # ------------------------------------------------------------------ one phase
    def _digest(self, probs: Optional[np.ndarray]) -> str:
        if probs is None:
            return ""
        return hashlib.sha256(np.ascontiguousarray(probs, dtype="<f8").tobytes()).hexdigest()[:16]

    def _probs(self, t: int, setting: Any) -> Optional[np.ndarray]:
        if t < 2 or not self.stratified:
            return None
        sw = self.start_weights
        return self.sampler.stratified_bin_probs(t, float(sw["lambda_P"]), self.hw, alpha=setting.alpha,
                                                 focus=setting.focus)

    def run_stage(self, t: int) -> Dict[str, Any]:
        """The phase of stage ``t``: rule-driven training, the freeze, the next table."""
        spec, P, agent = self.spec, self.P, self.agent
        T = self.T
        rule = self.stage_rules[t]
        ctl = StageController(rule, self.lr_linear,
                              None if rule.enabled else self._legacy_lr(t),
                              legacy_global_sampler=bool(self.fixed_global and t >= 2))
        n_ep = int(P["episodes_per_update"])
        bin_w = float(P["es_bin_width"])
        table_next = self.tables.get(t) if t < T else None
        if t < T and table_next is None:
            raise RuntimeError(f"no continuation table for the phase of stage {t}")
        dev_grid = dense_grid(spec.domain_half(t), self.dev_cfg.state_step)
        n_bins_t = self.sampler.n_bins(t) if t >= 2 else 0
        labels = self.sampler.stratum_labels(t, self.hw) if t >= 2 else None
        edges = self.sampler.bin_edges(t) if t >= 2 else None
        entry = self.global_u
        t_ph, c_ph = time.perf_counter(), time.process_time()
        cost_before = dict(self.costs.t)
        if self.conc_sched is not None and int(self.conc_sched["stage"]) != t:
            agent.actor.conc_scale = 1.0          # MS-R2: the scale of the scheduled stage ends with its phase
        agent.refresh_snapshot()
        self.snapshot_log.append({"update": self.global_u, "reason": f"phase_stage{t}_entry"})
        csv_path = os.path.join(self.out_dir, f"ms_checks_stage{t}.csv")
        print(f"[stage {t}] entry at global update {self.global_u}, rule enabled={rule.enabled}, "
              f"scheme={self.start_weights['scheme']}", flush=True)
        if self.phase_start_hook is not None:
            self.phase_start_hook(t)
        counts = {"episodes": 0, "transitions": 0, "minibatch_steps": 0}
        maps: Dict[str, List[Any]] = {"update": [], "local": [], "rho": [], "rho_bar": [], "probs": []}
        last: Dict[str, Any] = {}
        local = 0
        probs: Optional[np.ndarray] = None
        prob_changes: List[Tuple[int, np.ndarray]] = []   # (first global update, bin probabilities in force)
        last_digest = ""

        def diag_fn() -> StageDiag:
            t_v = time.perf_counter()
            ev, err = self.verify_candidate(self.dev_cfg)
            if ev is None:
                d = invalid_diag(t, n_bins_t)
            else:
                _, beta_fn = self.policy_fns()
                conc = concentration_stats(beta_fn, {t: dev_grid}, spec.e_range)
                d = stage_diag(ev.res, spec, t, conc, bin_w)
            dt = time.perf_counter() - t_v
            self.costs.add("dev_verifier", dt)
            last.update(ev=ev, err=err, dt=dt, diag=d)
            return d

        while not ctl.finished:
            local += 1
            self.global_u += 1
            t_upd = time.perf_counter()
            lr_now = ctl.lr(local)
            set_lr(agent, lr_now)
            actor_lr, critic_lr = read_lr(agent)
            if id(agent.opt_actor) != self.opt_ids["actor"] or id(agent.opt_critic) != self.opt_ids["critic"]:
                raise RuntimeError("optimizer object replaced")
            if self.conc_sched is not None and int(self.conc_sched["stage"]) == t \
                    and local > int(self.conc_sched["local_first"]):
                sc_ = self.conc_scale_for(local)
                agent.actor.conc_scale = sc_      # live actor and lagged opponent; constant within the update
                agent.opponent.conc_scale = sc_
            setting = ctl.sampler_setting()
            bid, btype = ctl.block_label()
            rng_start = self.rngs["start"]
            probs = None
            if t == 1:
                d0 = np.zeros(n_ep)
            elif not self.stratified:
                d0 = self.sampler.balanced(t, n_ep, rng_start)
            else:
                probs = self._probs(t, setting)
                dg = self._digest(probs)
                if dg != last_digest:
                    prob_changes.append((self.global_u, probs.copy()))
                    last_digest = dg
                d0 = self.sampler.stratified_priority(t, n_ep, rng_start, probs)
            roles = rng_start.integers(0, 2, size=n_ep)
            n_start = {"n_start_tail": 0, "n_start_near": 0, "n_start_mid": 0}
            if t >= 2:
                b_idx = np.clip(np.searchsorted(edges, d0, side="right") - 1, 0, n_bins_t - 1)
                lab = labels[b_idx]
                n_start = {"n_start_tail": int((lab == 0).sum()), "n_start_near": int((lab == 1).sum()),
                           "n_start_mid": int((lab == 2).sum())}
            t_r = time.perf_counter()
            if t == T:
                t0 = np.full(n_ep, T)
                batch = collect_batch_v2(spec, agent, t0, d0, roles, self.rngs["env"], self.rngs["learn"],
                                         self.rngs["opp"], self.ppo_cfg.gamma, self.ppo_cfg.gae_lambda,
                                         bin_w, reward_mode="expected", frozen=None,
                                         continuation_action_mode="stochastic", cont_table=None,
                                         clamp_likelihood="density")
            else:
                batch = collect_stage_batch(spec, agent, t, d0, roles, self.rngs["learn"], self.rngs["opp"],
                                            table_next)
            self.costs.add("train_rollout", time.perf_counter() - t_r)
            t_u = time.perf_counter()
            diag = agent.update(batch["states"], batch["actions"], batch["logp"], batch["returns"],
                                batch["advantages"])
            self.costs.add("train_update", time.perf_counter() - t_u)
            self.total_episodes += batch["n_episodes"]
            self.total_transitions += batch["n_transitions"]
            counts["episodes"] += batch["n_episodes"]
            counts["transitions"] += batch["n_transitions"]
            counts["minibatch_steps"] += diag["n_minibatch_steps"]
            snap = False
            if self.global_u % int(P["snapshot_every"]) == 0:
                agent.refresh_snapshot()
                self.snapshot_log.append({"update": self.global_u, "reason": f"every_{P['snapshot_every']}"})
                snap = True
            if self.global_u in self.full_state_at:
                torch.save(self.full_state(f"stage{t}"), os.path.join(self.out_dir, f"state_u{self.global_u:05d}.pt"))
            if self.weights_every and self.global_u % self.weights_every == 0:
                os.makedirs(self.weights_dir, exist_ok=True)
                w_path = os.path.join(self.weights_dir, f"u{self.global_u:05d}.npz")
                agent.export_weights_npz(w_path)
                self.weights_log.append({"update": self.global_u, "stage": t, "local": local,
                                         "file": os.path.relpath(w_path, self.out_dir)})
            self.history.append({"update": self.global_u, "stage": t, "local": local, "actor_lr": actor_lr,
                                 "critic_lr": critic_lr, "n_episodes": batch["n_episodes"],
                                 "mean_episode_return": batch["mean_episode_return"],
                                 "mean_effort": float(batch["mean_effort_by_stage"][t]),
                                 "snapshot_refreshed": snap,
                                 **{k_: v_ for k_, v_ in diag.items() if k_ != "kl_epochs"},
                                 "kl_epochs": diag["kl_epochs"]})
            self.rows.append({"update": self.global_u, "stage": t, "local": local, "block_id": bid,
                              "block_type": btype, "lr": actor_lr, "alpha": setting.alpha,
                              "mean_episode_return": batch["mean_episode_return"],
                              "mean_effort": float(batch["mean_effort_by_stage"][t]),
                              "kl_final_epoch": diag["kl_final_epoch"], "clip_frac": diag["clip_frac"],
                              "policy_loss": diag["policy_loss"], "value_loss": diag["value_loss"],
                              "adv_raw_mean": diag["adv_raw_mean"], "adv_raw_std": diag["adv_raw_std"],
                              **n_start, **batch["d1"], "p_digest": self._digest(probs),
                              "update_wall_sec": time.perf_counter() - t_upd,
                              **{f"rngpos_{k_}": rng_position(g_) for k_, g_ in self.rngs.items()},
                              "rngpos_minibatch": rng_position(agent.rng_mb),
                              **({"conc_scale": float(agent.actor.conc_scale)} if self.noise_report else {})})
            p_used = probs
            step = ctl.after_update(local, diag_fn)
            if step.check is not None:
                ev, err = last["ev"], last["err"]
                row = {c: float("nan") for c in CHECK_COLS}
                row.update({"run": self.cfg["run"], "arm": self.cfg["arm"], "q": spec.q, "seed": self.seed,
                            "stage": t, "update": self.global_u, "lr": actor_lr, "alpha": setting.alpha,
                            "would_fire_local": ctl.would_fire if ctl.would_fire is not None else "",
                            "p_digest": self._digest(p_used),
                            "p_digest_next": "" if ctl.finished else self._digest(self._probs(t, ctl.sampler_setting())),
                            "verifier_sec": last["dt"], "error": err or ""})
                row.update({k: v for k, v in step.check.items() if k in CHECK_COLS})
                if ev is not None:
                    row.update({k: ev.scalars[k] for k in CHECK_REPORT if k in ev.scalars})
                cols = CHECK_COLS
                if self.noise_report and t == T:
                    cols = CHECK_COLS + CHECK_NOISE
                    row.update({c: float("nan") for c in CHECK_NOISE})
                    row["conc_scale"] = float(agent.actor.conc_scale)
                    if ev is not None:
                        mean_fn_, beta_fn_ = self.policy_fns()
                        rep_ = tie_noise_report(mean_fn_, beta_fn_, spec, t, g2_0=float(ev.scalars["g2_at_0"]),
                                                e_hat_0=float(ev.scalars["e2_at_0"]))
                        row.update({"sigma_0": rep_["sigma_0"], "e_sigma_0": rep_["e_sigma_0"], "g2_0": rep_["g2_0"],
                                    "smoothing": rep_["smoothing"], "remainder": rep_["remainder"], "gap": rep_["gap"],
                                    "R0": tie_residual_R0(ev.res, t)})
                append_csv(csv_path, {c: row[c] for c in cols})
                if t >= 2:
                    maps["update"].append(self.global_u)
                    maps["local"].append(local)
                    maps["rho"].append(last["diag"].rho_bins.copy())
                    maps["rho_bar"].append(np.full(n_bins_t, np.nan) if ctl.rho_bar is None else ctl.rho_bar.copy())
                    maps["probs"].append(np.full(n_bins_t, np.nan) if p_used is None else p_used.copy())
                print(f"[u{self.global_u:>5} s{t} {local:>4}] check {step.check['block_type']} "
                      f"Delta={step.check['Delta']:.5f} R={step.check['R']:.4f} Rt={step.check['R_tail']:.4f} "
                      f"C={step.check['C']:.4f} elig={step.check['eligible']} consec={step.check['consecutive']} "
                      f"events={step.events}", flush=True)
            elif local % 50 == 0:
                print(f"[u{self.global_u:>5} s{t} {local:>4}] ret={batch['mean_episode_return']:.4f} "
                      f"kl={diag['kl_final_epoch']:.4f} lr={actor_lr:.3g} {btype}", flush=True)
        if t >= 2 and maps["update"]:
            extra = {}
            if prob_changes:   # the sampler probabilities change only at checks: piecewise constant in the update
                extra = {"probs_first_update": np.array([u for u, _ in prob_changes]),
                         "probs_table": np.array([p_ for _, p_ in prob_changes])}
            np.savez(os.path.join(self.out_dir, f"ms_binmaps_stage{t}.npz"), update=np.array(maps["update"]),
                     local=np.array(maps["local"]), rho=np.array(maps["rho"]),
                     rho_bar=np.array(maps["rho_bar"]), probs=np.array(maps["probs"]), **extra)
        rec = ctl.record()
        rec["entry_update"] = entry
        rec["exit_update"] = self.global_u
        rec["episodes"] = counts["episodes"]
        rec["transitions"] = counts["transitions"]
        rec["minibatch_steps"] = counts["minibatch_steps"]
        self.phase_timing[f"stage{t}"] = {
            "wall_sec": time.perf_counter() - t_ph, "process_cpu_sec": time.process_time() - c_ph,
            "updates": local, "global_entry": entry, "global_exit": self.global_u,
            "by_category_sec": {k_: v_ - cost_before.get(k_, 0.0) for k_, v_ in self.costs.t.items()}}
        rec["wall_sec"] = self.phase_timing[f"stage{t}"]["wall_sec"]
        print(f"[stage {t}] done at global update {self.global_u} after {local} local updates "
              f"(fire={rec['fire_local']}, would_fire={rec['would_fire_local']}, "
              f"budget_forced={rec['budget_forced']})", flush=True)
        return rec

    # ------------------------------------------------------------------ freeze
    def evaluate_tiers(self, t: int) -> Dict[str, Tuple[V2Eval, StageDiag]]:
        """Final- and development-tier evaluation of the candidate with the D3 diag at stage ``t`` (pure)."""
        out: Dict[str, Tuple[V2Eval, StageDiag]] = {}
        bin_w = float(self.P["es_bin_width"])
        for tier in (self.fin_cfg, self.dev_cfg):
            mf, bf = self.policy_fns()
            ev = evaluate(mf, self.spec, tier, beta_fn=bf, recovery_step=float(self.P["recovery_step"]))
            grid = dense_grid(self.spec.domain_half(t), tier.state_step)
            conc = concentration_stats(bf, {t: grid}, self.spec.e_range)
            out[tier.name] = (ev, stage_diag(ev.res, self.spec, t, conc, bin_w))
        return out

    def freeze_stage(self, t: int, rec: Dict[str, Any], check: Callable[[str], None]) -> None:
        """End of the landing window: state, final/dev evaluation, snapshot, next table, rule-log entry."""
        spec = self.spec
        self.phases_done.append(f"stage{t}")
        torch.save(self.full_state(f"stage{t}"), os.path.join(self.out_dir, f"state_end_stage{t}.pt"))
        check(f"end_of_stage{t}")
        tiers = self.evaluate_tiers(t)
        scal: Dict[str, Dict[str, Any]] = {}
        d3: Dict[str, Dict[str, Any]] = {}
        for name, (ev, dg) in tiers.items():
            save_npz(ev, os.path.join(self.out_dir, f"freeze_stage{t}_{name}.npz"))
            extra = stage2_extra(ev) if spec.T == 2 else {}
            scal[name] = {**ev.scalars, **extra}
            d3[name] = dg.scalars()
        self.freeze_evals[t] = {n: v[0] for n, v in tiers.items()}
        fz: Dict[str, Any] = {"final": pick(scal["final"], FREEZE_KEYS), "development": pick(scal["development"], FREEZE_KEYS),
                              "d3_final": d3["final"], "d3_development": d3["development"],
                              "rho_bins_final": [None if not np.isfinite(x) else float(x) for x in tiers["final"][1].rho_bins],
                              "rho_bins_development": [None if not np.isfinite(x) else float(x) for x in tiers["development"][1].rho_bins]}
        if spec.T == 2 and t == spec.T:
            fz["smoothed_game"] = smoothed_share(self, tiers["final"][0])
            if self.noise_report:
                mean_fn_, beta_fn_ = self.policy_fns()
                nz = tie_noise_report(mean_fn_, beta_fn_, spec, t, g2_0=float(scal["final"]["g2_at_0"]),
                                         e_hat_0=float(scal["final"]["e2_at_0"]))
                nz["smoothing_over_gaussian_formula"] = nz["smoothing"] / (
                    nz["g2_0"] * nz["sigma_0"] / (np.sqrt(np.pi) * spec.q))
                nz["conc_scale"] = float(self.agent.actor.conc_scale)
                nz["R0_final"] = tie_residual_R0(tiers["final"][0].res, t)
                nz["R0_development"] = tie_residual_R0(tiers["development"][0].res, t)
                fz["noise"] = nz
        check(f"after_freeze_eval_stage{t}")
        self.scal_freeze[t] = scal
        # ---- freeze the snapshot (deep copy: no grad, eval, referenced by no optimizer)
        snap = copy.deepcopy(self.agent.actor)
        for p in snap.parameters():
            p.requires_grad_(False)
        snap.eval()
        self.frozen_nets[t] = snap
        self.frozen_digest[t] = self._net_digest(snap)
        # ---- the table used by the phase of stage t - 1 (pure: must not move any RNG)
        if t >= 2:
            before = rng_snapshot(self)
            tbl = build_table(snap, spec, t - 1, self.tables.get(t), step=CONT_TABLE_STEP)
            after = rng_snapshot(self)
            if after["training"] != before["training"]:
                raise RuntimeError("a training RNG stream moved during the continuation-table build")
            for k in GLOBAL_RNGS:
                if after[k] != before[k]:
                    self.violations.append({"rng": k, "point": f"table_build_stage{t - 1}"})
            fname = f"continuation_table_stage{t - 1}.npz"
            self.tables[t - 1] = tbl
            self.table_records[str(t - 1)] = {
                "file": fname, "npz_sha256": write_table_npz(os.path.join(self.out_dir, fname), tbl),
                "meta": tbl.meta, "build_seconds": float(tbl.meta["build_seconds"]),
                "rng_unchanged_by_build": {"training": True, **{k: after[k] == before[k] for k in GLOBAL_RNGS}}}
        rec["freeze"] = fz
        self.freeze_records[str(t)] = fz
        self.rule_log[str(t)] = rec
        write_json(os.path.join(self.out_dir, "rule_log.json"), self._rule_log_doc())

    @staticmethod
    def _init_digest(agent: Any) -> str:
        """SHA-256 of the actor and critic weights (sorted names, float32 bytes); pure, no RNG."""
        h = hashlib.sha256()
        for pre, net in (("actor", agent.actor), ("critic", agent.critic)):
            sd = net.state_dict()
            for k in sorted(sd):
                h.update(f"{pre}.{k}".encode())
                h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
        return h.hexdigest()

    @staticmethod
    def _net_digest(net: Any) -> str:
        h = hashlib.sha256()
        for k, v in net.state_dict().items():
            h.update(k.encode() + v.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def _rule_log_doc(self) -> Dict[str, Any]:
        return {"schema": "ms_rule_log/1", "arm": self.cfg["arm"], "run": self.cfg["run"], "q": self.spec.q,
                "seed": self.seed, "rule": self.rule_cfg, "start_weights": self.start_weights,
                "derived": self.derived, "stages": self.rule_log}



# ---------------------------------------------------------------------------------- pipeline
def run_pipeline(cfg: Dict[str, Any], out_dir: str, cmd: str, band_step: Optional[float] = None) -> int:
    """Seed globals -> build the run -> phases T..1 -> gates -> outputs. Returns the exit code.

    ``band_step`` overrides the stage-1 decomposition step of the protocol (tests only)."""
    seed = int(cfg["seed"])
    os.makedirs(out_dir, exist_ok=True)
    cfg_copy = os.path.join(out_dir, "run_config.json")
    if not os.path.exists(cfg_copy):
        with open(cfg_copy, "w") as f:
            json.dump(cfg, f, indent=1)
    seed_globals(seed)
    rng_log: Dict[str, Any] = {"seeding": {"seeds": {k: seed for k in GLOBAL_RNGS},
                                           "digests": digests(global_states())}}
    run = MSRun(cfg, out_dir)
    rng_log["after_run_construction"] = {"digests": digests(global_states())}
    ref: Dict[str, object] = {}

    def hook(t: int) -> None:
        if not ref:
            ref.update(global_states())
            rng_log["reference_before_first_update"] = {"digests": digests(ref)}
    run.phase_start_hook = hook

    def check(point: str) -> None:
        dg = digests(global_states())
        rng_log[point] = {"digests": dg}
        rd = rng_log["reference_before_first_update"]["digests"]
        for k in GLOBAL_RNGS:
            if dg[k] != rd[k]:
                run.violations.append({"rng": k, "point": point})

    t0 = time.perf_counter()
    git = git_state()
    status = {"run": cfg["run"], "arm": cfg["arm"], "q": run.spec.q, "seed": run.seed, "state": "running",
              "pid": os.getpid(), "host": socket.gethostname(), "cmd": cmd,
              "start_time": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git}
    write_json(os.path.join(out_dir, "status.json"), status)
    man = write_manifest(run, cfg, out_dir, cmd, git, rng_log)
    try:
        T = run.T
        for t in range(T, 0, -1):
            rec = run.run_stage(t)
            run.freeze_stage(t, rec, check)
        gates = build_gates(run, cfg, git, band_step)
        write_drift_test(run, out_dir)
        check("end_of_run")                  # after the decomposition and the writers, as in the locked entry
        rng_ok = not run.violations
        gates["global_rng"] = {"status": "ok" if rng_ok else "violation", "violations": run.violations,
                               "assertion_points": [k for k in rng_log if k not in ("seeding",)]}
        gates["run_pass_v20_combination_and_rng"] = bool(gates.get("v20_combination_pass") and rng_ok)
        if not rng_ok:                       # the gate verdicts stay readable, the outcome label is the violation
            gates["outcome_gates"] = gates.get("outcome")
            gates["outcome"] = "global_rng_violation"
        write_json(os.path.join(out_dir, "gates.json"), gates)
        write_json(os.path.join(out_dir, "rule_log.json"), run._rule_log_doc())
        write_histories(run, out_dir)
        man["global_rng"] = rng_log
        man["continuation_tables"] = run.table_records
        write_json(os.path.join(out_dir, "manifest.json"), man)
        code = 0 if rng_ok else RNG_VIOLATION_EXIT
        status.update({"state": "done", "end_time": time.strftime("%Y-%m-%d %H:%M:%S"), "exit_code": code,
                       "final_global_update": run.global_u, "total_wall_sec": time.perf_counter() - t0,
                       "outcome": gates.get("outcome")})
        write_json(os.path.join(out_dir, "status.json"), status)
        print(f"[done] arm={cfg['arm']} q={run.spec.q:g} seed={run.seed} u{run.global_u} "
              f"outcome={gates.get('outcome')} global_rng={'ok' if rng_ok else run.violations} "
              f"wall={time.perf_counter() - t0:.1f}s", flush=True)
        return code
    except Exception:
        man["global_rng"] = rng_log
        man["continuation_tables"] = run.table_records
        write_json(os.path.join(out_dir, "manifest.json"), man)
        status.update({"state": "failed", "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "traceback": traceback.format_exc(), "updates_completed": run.global_u, "exit_code": 1})
        write_json(os.path.join(out_dir, "status.json"), status)
        print(traceback.format_exc(), flush=True)
        return 1


def write_manifest(run: MSRun, cfg: Dict[str, Any], out_dir: str, cmd: str, git: Dict[str, Any],
                   rng_log: Dict[str, Any]) -> Dict[str, Any]:
    """Resolved run manifest (commit, clean-tree flag, record SHA-256, arm, every parameter)."""
    spec, P = run.spec, run.P
    man = {
        "schema": "ms_run_manifest/1", "commit": git["commit"], "clean_tree": git["dirty"] is False,
        "git": git, "base_commit": cfg["base_commit"], "pilot": cfg["pilot"], "arm": cfg["arm"],
        "run": cfg["run"], "q": spec.q, "seed": run.seed, "T": run.T, "seed_namespaces": run.seed_namespaces,
        "record_sha256": cfg["record_sha256"], "protocol_ref": cfg["protocol_ref"],
        "start_weights": run.start_weights, "derived": run.derived, "rule": run.rule_cfg,
        "clamp_likelihood": cfg["clamp_likelihood"], "pipeline": cfg["pipeline"],
        "resolved_protocol": P, "lr_schedule": run.sched,
        "conc_scale_schedule": run.conc_sched, "fixed_global_sampler": run.fixed_global,
        "noise_report": run.noise_report,
        **({"actor_variant": run.actor_variant} if run.actor_variant != "t1" else {}),
        **({"init_state_sha256": run.init_state_sha256} if run.init_state_sha256 is not None else {}),
        "resolved_config": {"game": {k: getattr(spec, k) for k in ("w_h", "w_l", "k", "q", "T", "e_min", "e_max")},
                            "ppo": run.rec["ppo"], "verifier": run.rec["verifier"]},
        "versions": run.versions_actual, "thread_env": run.thread_env, "torch_threads": torch.get_num_threads(),
        "host": socket.gethostname(), "cmd": cmd, "global_rng": rng_log, "input_config": cfg,
    }
    write_json(os.path.join(out_dir, "manifest.json"), man)
    return man


def build_gates(run: MSRun, cfg: Dict[str, Any], git: Dict[str, Any],
                band_step: Optional[float] = None) -> Dict[str, Any]:
    """``gates.json`` (D7): every value and verdict (T = 2), the D3 values at both freezes, the budgets."""
    spec = run.spec
    T = run.T
    out: Dict[str, Any] = {
        "schema": "ms_gates/1", "arm": cfg["arm"], "run": cfg["run"], "q": spec.q, "seed": run.seed,
        "commit": git["commit"], "clean_tree": git["dirty"] is False, "protocol_ref": cfg["protocol_ref"],
        "continuation_tables": run.table_records,
        "budget": {k: {"updates": v["updates"], "wall_sec": v["wall_sec"],
                       "process_cpu_sec": v["process_cpu_sec"]} for k, v in run.phase_timing.items()},
        "budget_totals": {"total_updates": run.global_u, "total_episodes": run.total_episodes,
                          "total_transitions": run.total_transitions,
                          "minibatch_steps": int(sum(h["n_minibatch_steps"] for h in run.history))},
        "d3_at_freeze": {str(t): {"final": run.freeze_records[str(t)]["d3_final"],
                                  "development": run.freeze_records[str(t)]["d3_development"]}
                         for t in range(T, 0, -1)},
        "rule": {str(t): {k: run.rule_log[str(t)].get(k) for k in
                          ("fire_local", "would_fire_local", "budget_forced", "training_updates", "total_updates")}
                 for t in range(T, 0, -1)},
    }
    if T != 2:
        out.update({"closed_form_gates": None, "v20_combination_pass": None, "outcome": "not_evaluated_T3"})
        return out
    scA = run.scal_freeze[T]
    scF = run.scal_freeze[1]
    vals = {"eta_final": float(scA["final"]["eta_T_over_dw"]), "eta_dev": float(scA["development"]["eta_T_over_dw"]),
            "rmse": float(scA["final"]["stage2_rmse_pos_over_g2_0"]),
            "tail": float(scA["final"]["stage2_tail_mean_over_g2_0"]),
            "gmax_final": float(scF["final"]["Gmax_full_over_dw"]),
            "gmax_dev": float(scF["development"]["Gmax_full_over_dw"]),
            "s1": float(scF["final"]["stage1_rel_err_abs"])}
    V = verdicts(vals, cfg["protocol_gates"])
    dec = decomposition(run, float(scF["final"]["e1_at_0"]), cfg["protocol_gates"], run.fin_cfg, band_step)
    write_json(os.path.join(run.out_dir, "induced_band.json"), dec)
    out.update({
        "metric_values": vals,
        "dev_tier_values": {"G-A": {"eta_T_over_dw": vals["eta_dev"],
                                    "stage2_rmse_pos_over_g2_0": float(scA["development"]["stage2_rmse_pos_over_g2_0"]),
                                    "stage2_tail_mean_over_g2_0": float(scA["development"]["stage2_tail_mean_over_g2_0"])},
                            "G-F": {"Gmax_full_over_dw": vals["gmax_dev"]},
                            "G-S": {"stage1_rel_err_abs": float(scF["development"]["stage1_rel_err_abs"])}},
        "G-A": V["G-A"], "G-F": V["G-F"], "G-N": V["G-N"], "G-S": V["G-S"], "S1": V["S1"],
        "v1_1_outcome": V["v1_1_outcome"], "v1_0_outcome": V["v1_0_outcome"],
        "v20_combination_pass": V["run_pass"], "outcome": V["outcome"],
        "reported": {
            "end_of_stage2": {"final": pick(scA["final"], REPORT_KEYS_A + ("stage2_peak_locfree_rel_err", "stage2_peak_locfree_argmax_d")),
                              "development": pick(scA["development"], REPORT_KEYS_A + ("stage2_peak_locfree_rel_err", "stage2_peak_locfree_argmax_d")),
                              "dev_minus_final": diff(pick(scA["development"], REPORT_KEYS_A), pick(scA["final"], REPORT_KEYS_A)),
                              "smoothed_game": run.freeze_records[str(T)].get("smoothed_game")},
            "end_of_stage1": {"final": pick(scF["final"], REPORT_KEYS_F), "development": pick(scF["development"], REPORT_KEYS_F),
                              "dev_minus_final": diff(pick(scF["development"], REPORT_KEYS_F), pick(scF["final"], REPORT_KEYS_F)),
                              "decomposition": dec}}})
    return out




def write_drift_test(run: MSRun, out_dir: str) -> None:
    """C5-style integrity test: every frozen snapshot is bit-identical to its freeze-time weights, carries no
    grad and is referenced by no optimizer."""
    opt_ids = {id(p) for g in run.agent.opt_actor.param_groups for p in g["params"]} | \
              {id(p) for g in run.agent.opt_critic.param_groups for p in g["params"]}
    test: Dict[str, Any] = {}
    for s, net in sorted(run.frozen_nets.items()):
        test[str(s)] = {"bit_identical_to_freeze": run._net_digest(net) == run.frozen_digest[s],
                        "requires_grad_any": bool(any(p.requires_grad for p in net.parameters())),
                        "in_any_optimizer": bool(any(id(p) in opt_ids for p in net.parameters())),
                        "training_mode": bool(net.training)}
    test["pass"] = bool(all(v["bit_identical_to_freeze"] and not v["requires_grad_any"]
                            and not v["in_any_optimizer"] and not v["training_mode"] for v in test.values()))
    write_json(os.path.join(out_dir, "drift_test.json"), test)


def write_histories(run: MSRun, out_dir: str) -> None:
    """Per-update CSV and train_history.json (untracked, large)."""
    if run.rows:
        keys: List[str] = []
        for r in run.rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(out_dir, "ms_updates.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(run.rows)
    write_json(os.path.join(out_dir, "train_history.json"),
               {"run": run.cfg["run"], "arm": run.cfg["arm"], "history": run.history,
                "snapshots": run.snapshot_log, "weight_checkpoints": run.weights_log,
                "phase_timing": run.phase_timing})
    write_json(os.path.join(out_dir, "ms_run_summary.json"),
               {"phase_timing": run.phase_timing, "costs": run.costs.as_dict(), "phases_done": run.phases_done,
                "final_global_update": run.global_u, "total_episodes": run.total_episodes,
                "total_transitions": run.total_transitions})


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry."""
    p = argparse.ArgumentParser(description="MS-R1 multistage runner")
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    cfg = json.load(open(a.config))
    out_dir = a.out_dir
    if os.path.exists(os.path.join(out_dir, "status.json")):
        print(f"[refuse] {out_dir} already holds status.json; not restarting", flush=True)
        return 3
    validate_config(cfg)
    return run_pipeline(cfg, out_dir, " ".join(sys.argv))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as exc:
        print(f"[config-error] {exc}", flush=True)
        sys.exit(4)
