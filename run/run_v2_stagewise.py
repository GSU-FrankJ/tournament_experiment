#!/usr/bin/env python3
"""v2 T=2 entry point: stagewise (frozen) training flags, fixed budget, exact branching.

The training loop is a port of ``run/run_final_dp_br_round3_dense.py`` (base commit 657f54a).
With ``mode="full"``, ``fixed_budget=false`` and every flag at its default it executes the same
operations in the same order and writes the same ``train_history.json`` / ``final_eval.json`` /
``arrays.npz`` / ``checkpoint_weights.npz`` / ``phase_{A,B}_exit_arrays.npz`` (regression test
C7). New behaviour, all behind flags recorded in ``manifest.json``:

  modes          full (A -> B -> C, existing rules) | phase_A (A only, then a full-state
                 checkpoint) | phase_B (B only, restored from a full-state parent checkpoint) |
                 phase_A_continue (restore an end-of-A parent and continue phase A; the phase-entry
                 snapshot refresh and the phase-local verifier/stop-rule counters restart)
  full_state_at  optional list of global updates at which a full-state checkpoint
                 ``state_u<update>.pt`` is also written (default: none)
  lr_decay       optional window {phase, start_lr, end_lr, local_first, local_last}, or a list
                 of windows (at most one per phase): from local update local_first to local_last
                 both optimizers follow the existing phase-C linear form ``lr_at`` (round-3
                 runner); before local_first the record's schedule applies (default: none, i.e.
                 the record's schedule everywhere; refused in mode full)
  mode phase_P   (R1 refinement, method 5) restore an end-of-A parent and run the pathwise
                 terminal fine-tuning phase P: exact-gradient ascent on the conditional expected
                 terminal payoff, both players executing the Beta mean (agents/ppo_pathwise.py)
  mode locked    the locked v2 T=2 pipeline (run/run_v2_T2_locked.py): phase A then, after the
                 caller freezes the stage-2 snapshot, phase B in the same process; flags must be
                 expected / frozen / stage1_rows / mean (frozen only takes effect in phase B)
  R1 optional keys (all absent == the locked behaviour, bit-identical):
    lr_decay          several contiguous windows per phase (piecewise linear); start_lr != ab_lr
                      allowed for a window starting at local 1 of a continued phase (and for a
                      later window of a multi-window phase)
    budget_overrides  also episodes_per_update; phase_caps / verifier_timeout may carry key "P"
    ppo_overrides     {"minibatch": int, "target_kl": float | null}
    conc_anneal       {phase, local_first, local_last, scale_first, scale_last}: multiplicative
                      concentration factor on the live actor AND the lagged opponent (phase_A_continue)
    continuation_value_mode  "sampled" (default) | "expected" (stage-1 rows use the shock-integrated
                      table value of the frozen stage-2 policy; frozen + mean only)
  R2b optional keys (all absent == the locked behaviour, bit-identical):
    start_weights     {"scheme": "bin_balanced"} (default) | {"scheme": "peak_focused",
                      "peak_half_width": h, "peak_share": s}: a share s of the exploring starts of the
                      final stage in the bins that intersect (-h, h) (modes phase_A, phase_A_continue,
                      phase_P); one rng.random(n) mapped through the bin CDF, then the usual within-bin
                      draw, so the ``start`` stream desynchronises from bin_balanced by construction
    clamp_likelihood  "density" (default) | "censored": the log-prob of a learner row whose raw Beta draw
                      was clamped is the censored log-mass of the clamped event, in the rollout and in
                      every log-prob of the update (utils/beta_tail.py; modes phase_A, phase_A_continue,
                      phase_B)
    pathwise_epochs, pathwise_minibatch   mode phase_P: E passes over the rows, minibatches of M rows,
                      one exact-gradient step per minibatch (defaults 1 and null = the R1 single
                      full-batch step; with E = 10, M = 256 and 512 rows: 20 steps, the minibatch
                      stream drawn as the PPO update draws it)
  fixed_budget   no early exit; the update at which the existing phase rule would have fired is
                 recorded; the verifier keeps the existing cadence
  flags          reward_mode {sampled, expected}; stage2_update_mode {joint, frozen};
                 adv_norm_scope {all_rows, stage1_rows}; continuation_action_mode
                 {stochastic, mean}. Valid: joint => all_rows and stochastic; frozen only in
                 phase_B; phase_A => all_rows.

Every verifier call also runs ``utils.v2_metrics.evaluate`` on the same candidate (one ``verify``
call, pure) and writes one NPZ and one CSV row. The candidate in frozen mode is
(live actor at t=1, frozen snapshot at t=2). The closed form enters only the evaluation metrics.

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python run/run_v2_stagewise.py --config <run_config.json> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.ppo_curriculum import PPOConfig  # noqa: E402
from agents.ppo_curriculum_v2 import CurriculumPPOv2  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402
from run.run_final_dp_br import PROTOCOL, Costs, dense_grid, final_evaluation, make_policy_fns, write_json  # noqa: E402
from run.run_final_dp_br_round3_dense import (  # noqa: E402
    LR_SCHEDULE_REQUIRED_KEYS, PROTOCOL_REQUIRED_KEYS, ConfigError, lr_at, read_lr, set_lr,
    strict_dataclass, strict_keys)
from run.v2_rollout import CLAMP_LIKELIHOODS, CONT_MODES, REWARD_MODES, collect_batch_v2  # noqa: E402
from utils.dp_br_verifier import (  # noqa: E402
    DEV_CONFIG, FINAL_CONFIG, DomainError, VerifierConfig, VerifierResult, beta_std_norm,
    concentration_stats, effort_grid, stage_grid, stage_result_arrays)
from utils.theory_multistage import validate_two_stage_params  # noqa: E402
from utils.v2_metrics import V2Eval, append_csv, evaluate, onoff_split, plot_stage2, save_npz  # noqa: E402

SCHEMA = "v2_run_config/1"
REQUIRED = ("schema", "base_commit", "pilot", "arm", "run", "q", "seed", "mode", "fixed_budget",
            "flags", "parent_checkpoint", "parent_sha256", "record", "threads_per_process",
            "budget_overrides")
FLAG_KEYS = ("reward_mode", "stage2_update_mode", "adv_norm_scope", "continuation_action_mode")
DEFAULT_FLAGS = {"reward_mode": "sampled", "stage2_update_mode": "joint",
                 "adv_norm_scope": "all_rows", "continuation_action_mode": "stochastic"}
MODES = ("full", "phase_A", "phase_B", "phase_A_continue", "locked", "phase_P")
LOCKED_FLAGS = {"reward_mode": "expected", "stage2_update_mode": "frozen",
                "adv_norm_scope": "stage1_rows", "continuation_action_mode": "mean"}
MODE_PHASES = {"full": ("A", "B", "C"), "phase_A": ("A",), "phase_A_continue": ("A",), "phase_B": ("B",),
               "locked": ("A", "B"), "phase_P": ("P",)}
OVERRIDE_KEYS = ("phase_caps", "warmup", "stability_every", "verifier_timeout",
                 "direct_rollout_episodes", "direct_rollout_reps", "episodes_per_update")
PARENT_MODES = ("phase_B", "phase_A_continue", "phase_P")   # modes that restore a full-state parent
CONTINUED_MODES = ("phase_A_continue", "phase_P")           # a window may start at local 1 with start_lr != ab_lr
PPO_OVERRIDE_KEYS = ("minibatch", "target_kl")
CONC_ANNEAL_KEYS = ("phase", "local_first", "local_last", "scale_first", "scale_last")
CONT_VALUE_MODES = ("sampled", "expected")
CONT_TABLE_STEP = 0.05
RNG_NAMES = ("env", "learn", "opp", "start")   # + the minibatch stream held by the agent
LR_DECAY_KEYS = ("phase", "start_lr", "end_lr", "local_first", "local_last")
# R2b optional keys (absent == the locked behaviour, bit-identical)
START_SCHEMES = ("bin_balanced", "peak_focused")
PEAK_START_KEYS = ("scheme", "peak_half_width", "peak_share")
START_MODES = ("phase_A", "phase_A_continue", "phase_P")        # modes whose phases draw exploring starts
CLAMP_MODES = ("phase_A", "phase_A_continue", "phase_B")        # modes whose phases sample learner actions


def sha256_file(path: str) -> str:
    """sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> Dict[str, object]:
    """Commit and dirty flag of the repository this file lives in."""
    root = str(Path(__file__).resolve().parent.parent)
    try:
        commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
        # tracked changes OR untracked files anywhere except results/ (run outputs live there)
        dirty = bool(subprocess.check_output(["git", "-C", root, "status", "--porcelain", "--", ".",
                                              ":(exclude)results"],
                                             stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        commit, dirty = "unknown", None
    return {"commit": commit, "short": commit[:7], "dirty": dirty}


def _tree_equal(a: object, b: object) -> bool:
    """Exact (bitwise for tensors) equality of nested dict / list / tuple / tensor structures."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def rng_position(g: np.random.Generator) -> str:
    """Exact position of a numpy PCG64 stream: 128-bit state, cached-uint32 flag and value.

    Diagnostic only (reading the state does not advance the stream). ``inc`` is fixed per
    stream and omitted.
    """
    st = g.bit_generator.state
    return f"{st['state']['state']:032x}:{st['has_uint32']}:{st['uinteger']}"


def validate_config(cfg: Dict) -> None:
    """Refuse a config with a missing field or an invalid flag combination."""
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ConfigError(f"run config missing required fields {missing}")
    if cfg["schema"] != SCHEMA:
        raise ConfigError(f"schema {cfg['schema']!r} != {SCHEMA!r}")
    fl = cfg["flags"]
    miss_f = [k for k in FLAG_KEYS if k not in fl]
    if miss_f or set(fl) - set(FLAG_KEYS):
        raise ConfigError(f"flags must have exactly {FLAG_KEYS}; got {sorted(fl)}")
    if fl["reward_mode"] not in REWARD_MODES or fl["continuation_action_mode"] not in CONT_MODES \
            or fl["stage2_update_mode"] not in ("joint", "frozen") \
            or fl["adv_norm_scope"] not in ("all_rows", "stage1_rows"):
        raise ConfigError(f"invalid flag value in {fl}")
    mode = cfg["mode"]
    if mode not in MODES:
        raise ConfigError(f"mode {mode!r} not in {MODES}")
    if fl["stage2_update_mode"] == "joint" and fl["adv_norm_scope"] != "all_rows":
        raise ConfigError("joint requires adv_norm_scope=all_rows")
    if fl["stage2_update_mode"] == "joint" and fl["continuation_action_mode"] != "stochastic":
        raise ConfigError("continuation_action_mode=mean requires stage2_update_mode=frozen")
    if fl["stage2_update_mode"] == "frozen" and mode not in ("phase_B", "locked"):
        raise ConfigError("stage2_update_mode=frozen is only defined in mode phase_B")
    if mode == "locked" and (fl != LOCKED_FLAGS or not cfg["fixed_budget"]):
        raise ConfigError(f"mode locked requires flags {LOCKED_FLAGS} and fixed_budget true")
    if mode in ("phase_A", "phase_A_continue", "phase_P") and fl["adv_norm_scope"] != "all_rows":
        raise ConfigError("phase_A has no stage-1 rows; adv_norm_scope must be all_rows")
    if not isinstance(cfg.get("full_state_at", []), list):
        raise ConfigError("full_state_at must be a list of global updates")
    if mode == "full" and (fl != DEFAULT_FLAGS or cfg["fixed_budget"]):
        raise ConfigError("mode full is the regression mode: default flags, fixed_budget false")
    dec = cfg.get("lr_decay")
    if dec is not None:
        if mode == "full":
            raise ConfigError("lr_decay is not allowed in mode full (regression mode)")
        wins = dec if isinstance(dec, list) else [dec]
        for w in wins:
            if not isinstance(w, dict) or set(w) != set(LR_DECAY_KEYS):
                raise ConfigError(f"lr_decay must have exactly {LR_DECAY_KEYS}; got {sorted(w) if isinstance(w, dict) else w}")
            if w["phase"] not in MODE_PHASES[mode]:
                raise ConfigError(f"lr_decay.phase {w['phase']!r} is not the phase run by mode {mode}")
            if not 1 <= int(w["local_first"]) < int(w["local_last"]):
                raise ConfigError("lr_decay needs 1 <= local_first < local_last")
        for ph in {w["phase"] for w in wins}:
            seq = sorted((w for w in wins if w["phase"] == ph), key=lambda w: int(w["local_first"]))
            for a_, b_ in zip(seq[:-1], seq[1:]):
                if int(b_["local_first"]) != int(a_["local_last"]) + 1:
                    raise ConfigError(f"lr_decay: the windows of phase {ph} must be contiguous and "
                                      f"non-overlapping (local_first of the next = local_last + 1)")
    if mode in PARENT_MODES:
        if not cfg["parent_checkpoint"] or not cfg["parent_sha256"]:
            raise ConfigError(f"{mode} needs parent_checkpoint and parent_sha256")
    elif cfg["parent_checkpoint"] is not None or cfg["parent_sha256"] is not None:
        raise ConfigError("parent_checkpoint/parent_sha256 must be null unless mode=phase_B, "
                          "phase_A_continue or phase_P")
    # ---- R1 optional keys (absent == the locked behaviour)
    ppo_ov = cfg.get("ppo_overrides")
    if ppo_ov is not None:
        if not isinstance(ppo_ov, dict) or set(ppo_ov) - set(PPO_OVERRIDE_KEYS):
            raise ConfigError(f"ppo_overrides keys must be within {PPO_OVERRIDE_KEYS}")
        mb, tk = ppo_ov.get("minibatch"), ppo_ov.get("target_kl")
        if mb is not None and (not isinstance(mb, int) or isinstance(mb, bool) or mb < 1):
            raise ConfigError("ppo_overrides.minibatch must be a positive int")
        if tk is not None and (isinstance(tk, bool) or not isinstance(tk, (int, float)) or not tk > 0):
            raise ConfigError("ppo_overrides.target_kl must be null or a positive number")
        if mode == "full" and (mb is not None or tk is not None):
            raise ConfigError("ppo_overrides are not allowed in mode full (regression mode)")
    ca = cfg.get("conc_anneal")
    if ca is not None:
        if not isinstance(ca, dict) or set(ca) != set(CONC_ANNEAL_KEYS):
            raise ConfigError(f"conc_anneal must have exactly {CONC_ANNEAL_KEYS}")
        if mode != "phase_A_continue" or ca["phase"] != "A":
            raise ConfigError("conc_anneal is defined for phase A of mode phase_A_continue only")
        if not 1 <= int(ca["local_first"]) < int(ca["local_last"]):
            raise ConfigError("conc_anneal needs 1 <= local_first < local_last")
        if not (float(ca["scale_first"]) > 0 and float(ca["scale_last"]) > 0):
            raise ConfigError("conc_anneal scales must be positive")
    cvm = cfg.get("continuation_value_mode", "sampled")
    if cvm not in CONT_VALUE_MODES:
        raise ConfigError(f"continuation_value_mode {cvm!r} not in {CONT_VALUE_MODES}")
    if cvm == "expected":
        if fl["stage2_update_mode"] != "frozen" or fl["continuation_action_mode"] != "mean":
            raise ConfigError("continuation_value_mode=expected requires stage2_update_mode=frozen "
                              "and continuation_action_mode=mean")
        if float(cfg["record"]["ppo"]["gamma"]) != 1.0 or float(cfg["record"]["ppo"]["gae_lambda"]) != 1.0:
            raise ConfigError("continuation_value_mode=expected requires gamma = lambda = 1")
    # ---- R2b optional keys (absent == the locked behaviour)
    sw = cfg.get("start_weights")
    if sw is not None:
        if not isinstance(sw, dict) or sw.get("scheme") not in START_SCHEMES:
            raise ConfigError(f"start_weights.scheme must be one of {START_SCHEMES}")
        if mode not in START_MODES:
            raise ConfigError(f"start_weights is defined for modes {START_MODES} only (not {mode})")
        if sw["scheme"] == "bin_balanced":
            if set(sw) != {"scheme"}:
                raise ConfigError("start_weights scheme bin_balanced takes no other key")
        else:
            if set(sw) != set(PEAK_START_KEYS):
                raise ConfigError(f"start_weights peak_focused must have exactly {PEAK_START_KEYS}")
            hw, sh = sw["peak_half_width"], sw["peak_share"]
            if isinstance(hw, bool) or not isinstance(hw, (int, float)) or not hw > 0:
                raise ConfigError("start_weights.peak_half_width must be a positive number")
            if isinstance(sh, bool) or not isinstance(sh, (int, float)) or not 0.0 < sh < 1.0:
                raise ConfigError("start_weights.peak_share must lie in (0, 1)")
    cl = cfg.get("clamp_likelihood", "density")
    if cl not in CLAMP_LIKELIHOODS:
        raise ConfigError(f"clamp_likelihood {cl!r} not in {CLAMP_LIKELIHOODS}")
    if cl != "density" and mode not in CLAMP_MODES:
        raise ConfigError(f"clamp_likelihood={cl} is defined for modes {CLAMP_MODES} only (not {mode})")
    pe, pm = cfg.get("pathwise_epochs", 1), cfg.get("pathwise_minibatch")
    if isinstance(pe, bool) or not isinstance(pe, int) or pe < 1:
        raise ConfigError("pathwise_epochs must be a positive int")
    if pm is not None and (isinstance(pm, bool) or not isinstance(pm, int) or pm < 1):
        raise ConfigError("pathwise_minibatch must be null or a positive int")
    if (pe != 1 or pm is not None) and mode != "phase_P":
        raise ConfigError("pathwise_epochs / pathwise_minibatch are defined for mode phase_P only")
    bad = set(cfg["budget_overrides"]) - set(OVERRIDE_KEYS)
    if bad:
        raise ConfigError(f"unknown budget_overrides {sorted(bad)}")
    if not isinstance(cfg["fixed_budget"], bool):
        raise ConfigError("fixed_budget must be a bool")
    if int(cfg["seed"]) != int(cfg["record"]["seed"]) or float(cfg["q"]) != float(cfg["record"]["q"]):
        raise ConfigError("q/seed differ from the embedded record")


class Run:
    """One v2 run (state + phase loop)."""

    def __init__(self, cfg: Dict, out_dir: str):
        validate_config(cfg)
        self.cfg = cfg
        self.out_dir = out_dir
        rec = cfg["record"]
        self.rec = rec
        self.flags = dict(cfg["flags"])
        self.mode = cfg["mode"]
        self.fixed = bool(cfg["fixed_budget"])
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
        self.ppo_cfg = strict_dataclass(PPOConfig, rec["ppo"], "ppo")
        self.ppo_overrides = copy.deepcopy(cfg.get("ppo_overrides") or {})
        if self.ppo_overrides.get("minibatch") is not None:
            self.ppo_cfg = dataclasses.replace(self.ppo_cfg, minibatch=int(self.ppo_overrides["minibatch"]))
        self.target_kl = self.ppo_overrides.get("target_kl")
        self.conc_anneal = copy.deepcopy(cfg.get("conc_anneal"))
        self.cont_mode = cfg.get("continuation_value_mode", "sampled")
        self.cont_table = None
        # R2b: peak-focused starts, censored likelihood of clamped draws, phase-P optimiser budget
        self.start_weights = copy.deepcopy(cfg.get("start_weights"))
        self.clamp_likelihood = cfg.get("clamp_likelihood", "density")
        self.pathwise_epochs = int(cfg.get("pathwise_epochs", 1))
        self.pathwise_minibatch = cfg.get("pathwise_minibatch")
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
        spec = self.spec
        for k_ in ("w_h", "w_l", "k", "e_min", "e_max"):
            if float(P[k_]) != float(getattr(spec, k_)):
                raise ConfigError(f"protocol.{k_} != game.{k_}")
        if float(rec["dw"]) != spec.dw or float(rec["B"]) != spec.B or float(rec["domain_half_stage2"]) != spec.domain_half(2):
            raise ConfigError("record dw/B/domain_half_stage2 inconsistent with game")
        if int(P["phase_c_root"]) + int(P["phase_c_es"]) != int(P["episodes_per_update"]):   # the record itself
            raise ConfigError("phase_c_root + phase_c_es != episodes_per_update")
        if rec["smoke_overrides"]:
            raise ConfigError("embedded record carries smoke_overrides")
        ov = copy.deepcopy(cfg["budget_overrides"])
        P.update(ov)
        if "C" in MODE_PHASES[self.mode] and int(P["phase_c_root"]) + int(P["phase_c_es"]) != int(P["episodes_per_update"]):
            raise ConfigError("phase_c_root + phase_c_es != episodes_per_update")   # after an episodes_per_update override
        if "P" in MODE_PHASES[self.mode] and isinstance(P["verifier_timeout"], dict) and "P" not in P["verifier_timeout"]:
            raise ConfigError("verifier_timeout has no entry for phase P")
        for ph in MODE_PHASES[self.mode]:
            if ph not in P["phase_caps"]:
                raise ConfigError(f"budget phase_caps has no entry for phase {ph}")
        self.overrides = ov
        self.P = P
        self.sched = sched
        val = validate_two_stage_params(q=spec.q, w_h=spec.w_h, w_l=spec.w_l, k=spec.k, e_bar=spec.e_max)
        if not val.ok:
            raise ConfigError(f"q={spec.q} fails closed-form validity: {val.messages}")
        # ---- RNG streams: identical construction to the existing runner
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
        self.agent.target_kl = None if self.target_kl is None else float(self.target_kl)
        self.opt_ids = {"actor": id(self.agent.opt_actor), "critic": id(self.agent.opt_critic)}
        set_lr(self.agent, lr_at(sched, "A", 1))
        self.sampler = StartSampler(spec, P["es_bin_width"])
        if self.sampler.n_bins(2) != int(rec["es_bins_stage2"]):
            raise ConfigError("es bins differ from record")
        if self.start_weights is not None and self.start_weights["scheme"] == "peak_focused":
            try:   # the peak set must be a non-empty proper subset of the bins of the final stage
                self.sampler.peak_bin_probs(spec.T, float(self.start_weights["peak_half_width"]),
                                            float(self.start_weights["peak_share"]))
            except ValueError as exc:
                raise ConfigError(f"start_weights: {exc}") from exc
        # ---- run state
        self.costs = Costs()
        self.costs_v2 = Costs()   # v2-only bookkeeping, kept out of final_eval.json
        self.history: List[Dict] = []
        self.stability_log: List[Dict] = []
        self.verifier_log: List[Dict] = []
        self.curriculum_log: List[Dict] = []
        self.snapshot_log: List[Dict] = [{"update": 0, "reason": "init"}]
        self.weights_log: List[Dict] = []
        self.v2_history: List[Dict] = []
        self.phase_timing: Dict[str, Dict] = {}
        self.would_fire: Dict[str, Optional[Dict]] = {}
        self.visitation_by_phase: Dict[str, Dict[str, np.ndarray]] = {}
        self.total_episodes = 0
        self.total_transitions = 0
        self.global_u = 0
        self.phases_done: List[str] = []
        self.stop_record: Optional[Dict] = None
        self.five_window: Optional[Dict] = None
        self.c_elig_stats = {"max_consecutive_eligible": 0, "consecutive_eligible_at_end": 0,
                             "n_calls": 0, "n_eligible": 0}
        self.dev_grid2 = dense_grid(spec.domain_half(2), self.dev_cfg.state_step)
        self.parent_ref: Optional[Dict[str, np.ndarray]] = None
        self.weights_every = int(P["weights_every"])
        self.weights_dir = os.path.join(out_dir, "weights")
        self.full_state_at = {int(u) for u in cfg.get("full_state_at", [])}
        # optional callable(phase) run immediately before the first update of each phase (after the
        # phase-entry snapshot refresh); used by the locked entry point to take its global-RNG reference
        self.phase_start_hook: Optional[Callable[[str], None]] = None
        self.lr_decay = cfg.get("lr_decay")
        dec = self.lr_decay
        self.lr_windows: Dict[str, List[Dict]] = {}
        for w in ([] if dec is None else (dec if isinstance(dec, list) else [dec])):
            self.lr_windows.setdefault(w["phase"], []).append(w)
        for ph, seq in self.lr_windows.items():
            seq.sort(key=lambda w: int(w["local_first"]))
            if int(seq[-1]["local_last"]) != int(P["phase_caps"][ph]):
                raise ConfigError("lr_decay.local_last must equal the phase cap")
            first = seq[0]
            if float(first["start_lr"]) != float(sched["ab_lr"]) and not (
                    self.mode in CONTINUED_MODES and int(first["local_first"]) == 1):
                raise ConfigError("lr_decay.start_lr must equal lr_schedule.ab_lr (a window starting at "
                                  "local 1 of a continued phase may differ)")
            if ph == "P" and int(first["local_first"]) != 1:
                raise ConfigError("phase P has no record schedule: its lr window must start at local 1")
        if "P" in MODE_PHASES[self.mode] and "P" not in self.lr_windows:
            raise ConfigError("mode phase_P needs an lr_decay window for phase P")

    # ------------------------------------------------------------------ learning rate
    def lr_for(self, phase: str, local: int) -> float:
        """LR before local update ``local``: the record's ``lr_at``, or the linear decay window.

        Inside the window (local_first <= local) the decay reuses ``lr_at``'s linear branch
        unchanged: lr(j) = start + (end - start) * (j - local_first) / (local_last - local_first).
        """
        dec = None
        for w in self.lr_windows.get(phase, ()):
            if int(w["local_first"]) <= local <= int(w["local_last"]):
                dec = w
        if dec is None:
            return lr_at(self.sched, phase, local)
        lin = dict(self.sched, kind="linear", c_start_lr=float(dec["start_lr"]),
                   c_end_lr=float(dec["end_lr"]), c_local_first=int(dec["local_first"]),
                   linear_denominator=int(dec["local_last"]) - int(dec["local_first"]))
        return lr_at(lin, "C", local)

    # ------------------------------------------------------------------ R2b helpers
    def draw_starts(self, stage: int, n: int) -> np.ndarray:
        """Exploring starts on D_stage from the ``start`` stream: bin-balanced (the locked scheme, the
        default) or, with ``start_weights = {scheme: peak_focused, ...}``, peak-focused."""
        sw, rng = self.start_weights, self.rngs["start"]
        if sw is None or sw["scheme"] == "bin_balanced":
            return self.sampler.balanced(stage, n, rng)
        return self.sampler.peak_focused(stage, n, rng, float(sw["peak_half_width"]), float(sw["peak_share"]))

    # ------------------------------------------------------------------ R1 helpers
    def conc_scale_for(self, local: int) -> float:
        """Concentration factor before local update ``local`` (linear between the anneal endpoints)."""
        ca = self.conc_anneal
        j0, j1 = int(ca["local_first"]), int(ca["local_last"])
        s0, s1 = float(ca["scale_first"]), float(ca["scale_last"])
        if local <= j0:
            return s0
        if local >= j1:
            return s1
        return s0 + (s1 - s0) * (local - j0) / (j1 - j0)

    @staticmethod
    def d1_policy_row_stats(batch: Dict, pol_rows: np.ndarray) -> Dict[str, object]:
        """D1 columns over the learner's policy rows: min alpha / beta and counts below 1."""
        a, b = batch["d1_buf"]["alpha"][pol_rows], batch["d1_buf"]["beta"][pol_rows]
        n = int(a.size)
        return {"d1_pol_n_rows": n,
                "d1_pol_alpha_min": float(a.min()) if n else float("nan"),
                "d1_pol_beta_min": float(b.min()) if n else float("nan"),
                "d1_pol_n_alpha_lt1": int((a < 1.0).sum()), "d1_pol_n_beta_lt1": int((b < 1.0).sum())}

    # ------------------------------------------------------------------ policy functions
    def policy_fns(self) -> Tuple[Callable, Callable]:
        """(mean_fn, beta_fn) of the current candidate (composite in frozen mode)."""
        live_mean, live_beta = make_policy_fns(self.agent, self.spec)
        if self.agent.frozen is None:
            return live_mean, live_beta
        fz_mean, fz_beta = make_policy_fns(self.agent, self.spec, net=self.agent.frozen)
        T = self.spec.T

        def mean_fn(t, d):
            return fz_mean(t, d) if t == T else live_mean(t, d)

        def beta_fn(t, d):
            return fz_beta(t, d) if t == T else live_beta(t, d)
        return mean_fn, beta_fn

    def stage2_mapping(self, net=None) -> Dict[str, np.ndarray]:
        """Stage-2 mean / alpha / beta of ``net`` (default live actor) on the dev D_2 grid."""
        mf, bf = make_policy_fns(self.agent, self.spec, net=net)
        a, b = bf(2, self.dev_grid2)
        return {"mean": mf(2, self.dev_grid2), "alpha": a, "beta": b}

    # ------------------------------------------------------------------ full state
    def full_state(self, phase_done: str) -> Dict[str, object]:
        """Complete branching state at a phase boundary."""
        return {
            "format": "v2_full_state/1", "phase_done": phase_done, "phases_done": list(self.phases_done),
            "agent": self.agent.full_state(),
            "rng": {k: copy.deepcopy(g.bit_generator.state) for k, g in self.rngs.items()},
            "torch_generator_state": self.torch_gen.get_state(),
            "torch_global_rng_state": torch.get_rng_state(),
            "numpy_global_rng_state": np.random.get_state(),
            "python_random_state": random.getstate(),
            "counters": {"global_u": self.global_u, "total_episodes": self.total_episodes,
                         "total_transitions": self.total_transitions,
                         "snapshot_refreshes": self.agent.snapshot_refreshes,
                         "next_snapshot_refresh_update": (self.global_u // int(self.P["snapshot_every"]) + 1)
                         * int(self.P["snapshot_every"])},
            "snapshot_log": copy.deepcopy(self.snapshot_log),
            "schedule_positions": {"lr_kind": self.sched["kind"], "phase_done": phase_done,
                                   "actor_lr": read_lr(self.agent)[0], "critic_lr": read_lr(self.agent)[1],
                                   "entropy_coef": self.ppo_cfg.entropy_coef, "entropy_schedule": None},
            "normalizer_statistics": None,   # obs encoding is a fixed function; no running normalizer
            "flags": dict(self.flags), "seed": self.seed, "q": self.spec.q,
            "seed_namespaces": self.seed_namespaces,
        }

    def restore(self, path: str) -> None:
        """Restore a full-state checkpoint written by :meth:`full_state`."""
        s = torch.load(path, map_location="cpu", weights_only=False)
        if s.get("format") != "v2_full_state/1":
            raise ConfigError(f"{path} is not a v2 full-state checkpoint")
        if float(s["q"]) != float(self.spec.q) or int(s["seed"]) != self.seed:
            raise ConfigError("parent checkpoint q/seed differ from this run")
        self.agent.load_full_state(s["agent"])
        for k, g in self.rngs.items():
            g.bit_generator.state = copy.deepcopy(s["rng"][k])
        self.torch_gen.set_state(s["torch_generator_state"])
        torch.set_rng_state(s["torch_global_rng_state"])
        np.random.set_state(s["numpy_global_rng_state"])
        random.setstate(s["python_random_state"])
        c = s["counters"]
        self.global_u = int(c["global_u"])
        self.total_episodes = int(c["total_episodes"])
        self.total_transitions = int(c["total_transitions"])
        self.snapshot_log = copy.deepcopy(s["snapshot_log"])
        self.phases_done = list(s["phases_done"])
        self.parent_state = s

    # ------------------------------------------------------------------ verifier call
    def verify_candidate(self, cfg: VerifierConfig) -> Tuple[Optional[V2Eval], Optional[str]]:
        """``utils.v2_metrics.evaluate`` on the current candidate (errors as in run_verifier)."""
        mean_fn, beta_fn = self.policy_fns()
        try:
            return evaluate(mean_fn, self.spec, cfg, beta_fn=beta_fn,
                            recovery_step=float(self.P["recovery_step"])), None
        except (DomainError, FloatingPointError, ValueError) as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def drift_scalars(self, ev: Optional[V2Eval]) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
        """Stage-2 drift of the live net and of the candidate against the phase-B parent mapping."""
        if self.parent_ref is None or ev is None:
            return {}, {}
        live = self.stage2_mapping()
        mean_fn, _ = self.policy_fns()
        cand = np.asarray(mean_fn(2, self.dev_grid2), dtype=float)
        on, w = ev.arrays["v_t2_onpath"], ev.arrays["v_t2_cell_mass"]
        out: Dict[str, object] = {}
        arr: Dict[str, np.ndarray] = {"drift_parent_mean": self.parent_ref["mean"],
                                      "drift_live_mean": live["mean"], "drift_cand_mean": cand}
        for name, x in (("live", live["mean"]), ("cand", cand)):
            dd = np.abs(x - self.parent_ref["mean"])
            out[f"stage2_drift_{name}_maxabs"] = float(dd.max())
            for k_, v_ in onoff_split(dd, on, w, self.dev_grid2).items():
                out[f"stage2_drift_{name}_{k_}"] = v_
        out["stage2_drift_live_alpha_maxabs"] = float(np.max(np.abs(live["alpha"] - self.parent_ref["alpha"])))
        out["stage2_drift_live_beta_maxabs"] = float(np.max(np.abs(live["beta"] - self.parent_ref["beta"])))
        return out, arr

    # ------------------------------------------------------------------ phase loop
    def run_phase(self, phase: str) -> str:
        """One curriculum phase (port of the existing loop; see module docstring)."""
        spec, P, sched, agent, ppo_cfg = self.spec, self.P, self.sched, self.agent, self.ppo_cfg
        active = [spec.T] if phase == "A" else list(range(1, spec.T + 1))
        cap = int(P["phase_caps"][phase])
        entry = self.global_u
        local = 0
        eligible = 0
        stab_consec = 0
        prev_stab: Optional[Dict[int, np.ndarray]] = None
        last_call: Optional[int] = None
        last_result: Optional[VerifierResult] = None
        phase_updates = {"episodes": 0, "transitions": 0, "minibatch_steps": 0}
        t_ph, c_ph = time.perf_counter(), time.process_time()
        cost_before = dict(self.costs.t)
        agent.refresh_snapshot()
        self.snapshot_log.append({"update": self.global_u, "reason": f"phase_{phase}_entry"})
        vis_phase: Dict[str, np.ndarray] = {}
        exit_reason = "budget_exhausted"
        would_fire: Optional[Dict] = None
        frozen = agent.frozen if (phase == "B" and self.flags["stage2_update_mode"] == "frozen") else None
        masked = frozen is not None
        if frozen is not None and self.cont_mode == "expected":
            from utils.v2_continuation import build_continuation_table   # R1 method 6 (lazy: default path never imports it)
            t_ct = time.perf_counter()
            self.cont_table = build_continuation_table(frozen, spec, stage=spec.T, step=CONT_TABLE_STEP)
            self.costs_v2.add("continuation_table", time.perf_counter() - t_ct)
        cont_table = self.cont_table if (frozen is not None and self.cont_mode == "expected") else None
        save_locals = {u for u in (25, int(round(cap / 2 / 25.0)) * 25, cap) if 1 <= u <= cap}   # D1 buffers
        # mode locked runs A and B in one process; B rows carry extra drift columns -> one CSV per phase
        csv_path = os.path.join(self.out_dir, f"v2_checkpoints_{phase}.csv" if self.mode == "locked" else "v2_checkpoints.csv")
        os.makedirs(os.path.join(self.out_dir, "checkpoints"), exist_ok=True)
        print(f"[phase {phase}] entry at global update {self.global_u}, cap {cap}, active stages {active}, "
              f"lr(first)={self.lr_for(phase, 1):.6g}", flush=True)

        def stability_points() -> Dict[int, np.ndarray]:
            return {2: self.dev_grid2} if phase == "A" else {1: np.zeros(1), 2: self.dev_grid2}

        def phase_criterion(res: VerifierResult) -> Tuple[str, float]:
            if phase == "A":
                return "max_D2dev_Delta2_over_dw", res.full_delta_max[spec.T] / spec.dw
            if phase == "B":
                return "exp_root_over_dw", res.exp_root / spec.dw
            return "dreach_over_dw", res.dreach / spec.dw

        def timeout_for() -> int:
            v = P["verifier_timeout"]
            return int(v[phase]) if isinstance(v, dict) else int(v)

        if self.phase_start_hook is not None:
            self.phase_start_hook(phase)
        while local < cap:
            local += 1
            self.global_u += 1
            t_upd = time.perf_counter()
            lr_now = self.lr_for(phase, local)
            set_lr(agent, lr_now)
            actor_lr, critic_lr = read_lr(agent)
            if id(agent.opt_actor) != self.opt_ids["actor"] or id(agent.opt_critic) != self.opt_ids["critic"]:
                raise RuntimeError("optimizer object replaced")
            if self.conc_anneal is not None and self.conc_anneal["phase"] == phase:
                sc_ = self.conc_scale_for(local)
                agent.actor.conc_scale = sc_      # live actor and lagged opponent; constant within the update
                agent.opponent.conc_scale = sc_
            n_ep = int(P["episodes_per_update"])
            rng_start = self.rngs["start"]
            if phase == "A":
                t0 = np.full(n_ep, spec.T)
                d0 = self.draw_starts(spec.T, n_ep)
            elif phase == "B":
                t0 = np.ones(n_ep, dtype=int)
                d0 = np.zeros(n_ep)
            else:
                nr, ne = int(P["phase_c_root"]), int(P["phase_c_es"])
                t0 = np.concatenate([np.ones(nr, dtype=int), np.full(ne, spec.T)])
                d0 = np.concatenate([np.zeros(nr), self.draw_starts(spec.T, ne)])
                perm = rng_start.permutation(n_ep)
                t0, d0 = t0[perm], d0[perm]
            roles = rng_start.integers(0, 2, size=n_ep)
            t_r = time.perf_counter()
            batch = collect_batch_v2(spec, agent, t0, d0, roles, self.rngs["env"], self.rngs["learn"],
                                     self.rngs["opp"], ppo_cfg.gamma, ppo_cfg.gae_lambda, P["es_bin_width"],
                                     reward_mode=self.flags["reward_mode"], frozen=frozen,
                                     # 'mean' is defined only with a frozen stage-2 actor; without
                                     # one (phase A of mode locked) the default 'stochastic' applies,
                                     # which is what every other mode already validates to
                                     continuation_action_mode=(self.flags["continuation_action_mode"]
                                                               if frozen is not None else "stochastic"),
                                     cont_table=cont_table, clamp_likelihood=self.clamp_likelihood)
            self.costs.add("train_rollout", time.perf_counter() - t_r)
            s1 = batch["stage"] == 1
            pol_rows = s1 if masked else np.ones(s1.size, dtype=bool)
            adv_t = torch.as_tensor(batch["advantages"])
            adv_stats = {"adv_all_mean": float(adv_t.mean()), "adv_all_std": float(adv_t.std(unbiased=False)),
                         "adv_s1_mean": float(adv_t[torch.as_tensor(s1)].mean()) if s1.any() else float("nan"),
                         "adv_s1_std": float(adv_t[torch.as_tensor(s1)].std(unbiased=False)) if s1.any() else float("nan"),
                         "n_rows": int(s1.size), "n_s1_rows": int(s1.sum())}
            pre_actor = ({f"actor_pre.{k_}": v_.detach().cpu().numpy().copy() for k_, v_ in agent.actor.state_dict().items()}
                         if local in save_locals else None)   # the policy that generated this buffer (D1)
            pre_scale = float(agent.actor.conc_scale)
            t_u = time.perf_counter()
            side = batch["clamp_side"] if self.clamp_likelihood == "censored" else None
            if masked:
                norm_mask = s1 if self.flags["adv_norm_scope"] == "stage1_rows" else None
                diag = agent.update(batch["states"], batch["actions"], batch["logp"], batch["returns"],
                                    batch["advantages"], policy_mask=s1, norm_mask=norm_mask, clamp_side=side)
            elif side is None:
                diag = agent.update(batch["states"], batch["actions"], batch["logp"], batch["returns"],
                                    batch["advantages"])
            else:   # R2b: censored likelihood (same update; the flagged rows use the censored log-mass)
                diag = agent.update(batch["states"], batch["actions"], batch["logp"], batch["returns"],
                                    batch["advantages"], clamp_side=side)
            self.costs.add("train_update", time.perf_counter() - t_u)
            if local in save_locals:   # D1 buffer of THIS update (collected before it); bookkeeping only
                os.makedirs(os.path.join(self.out_dir, "d1_buffers"), exist_ok=True)
                np.savez(os.path.join(self.out_dir, "d1_buffers", f"u{self.global_u:05d}.npz"),
                         states=batch["states"], raw=batch["d1_buf"]["raw"], actions=batch["actions"],
                         alpha=batch["d1_buf"]["alpha"], beta=batch["d1_buf"]["beta"], stage=batch["stage"],
                         old_logp=batch["logp"], adv_raw=batch["advantages"], policy_mask=pol_rows,
                         adv_norm_mean=np.float64(diag["adv_raw_mean"]), adv_norm_std=np.float64(diag["adv_raw_std"]),
                         local=np.int64(local), global_update=np.int64(self.global_u),
                         conc_scale_pre=np.float64(pre_scale), **pre_actor)
            actor_lr_after, critic_lr_after = read_lr(agent)
            if actor_lr_after != actor_lr or critic_lr_after != critic_lr:
                raise RuntimeError("learning rate changed inside an update")
            self.total_episodes += batch["n_episodes"]
            self.total_transitions += batch["n_transitions"]
            phase_updates["episodes"] += batch["n_episodes"]
            phase_updates["transitions"] += batch["n_transitions"]
            phase_updates["minibatch_steps"] += diag["n_minibatch_steps"]
            for k_, v_ in batch["visitation"].items():
                vis_phase[k_] = vis_phase.get(k_, 0) + v_
            snap = False
            if self.global_u % int(P["snapshot_every"]) == 0:
                agent.refresh_snapshot()
                self.snapshot_log.append({"update": self.global_u, "reason": f"every_{P['snapshot_every']}"})
                snap = True
            if self.global_u in self.full_state_at:
                torch.save(self.full_state(phase), os.path.join(self.out_dir, f"state_u{self.global_u:05d}.pt"))
            if self.weights_every and self.global_u % self.weights_every == 0:
                os.makedirs(self.weights_dir, exist_ok=True)
                w_path = os.path.join(self.weights_dir, f"u{self.global_u:05d}.npz")
                agent.export_weights_npz(w_path)
                self.weights_log.append({"update": self.global_u, "phase": phase, "local": local,
                                         "file": os.path.relpath(w_path, self.out_dir)})
            rec_h = {"update": self.global_u, "phase": phase, "local": local, "actor_lr": actor_lr,
                     "critic_lr": critic_lr, "n_episodes": batch["n_episodes"],
                     "n_transitions": batch["n_transitions"],
                     "mean_episode_return": batch["mean_episode_return"],
                     "mean_effort_by_stage": {str(t): v for t, v in batch["mean_effort_by_stage"].items()},
                     "snapshot_refreshed": snap, **{k_: v_ for k_, v_ in diag.items() if k_ != "kl_epochs"},
                     "kl_epochs": diag["kl_epochs"]}
            self.history.append(rec_h)
            mean_fn, beta_fn = self.policy_fns()
            # ---- stability checkpoint
            stab_now = False
            if local % int(P["stability_every"]) == 0:
                t_s = time.perf_counter()
                pts = stability_points()
                cur = {t: np.asarray(mean_fn(t, d), dtype=float) for t, d in pts.items()}
                drift = None
                if prev_stab is not None:
                    drift = max(float(np.max(np.abs(cur[t] - prev_stab[t]))) / spec.e_range for t in cur)
                kl = diag["kl_final_epoch"]
                stable = drift is not None and drift <= P["drift_thr"] and kl <= P["kl_thr"]
                stab_consec = stab_consec + 1 if stable else 0
                conc_s = concentration_stats(beta_fn, pts, spec.e_range)
                self.stability_log.append({"update": self.global_u, "phase": phase, "local": local,
                                           "drift": drift, "kl": kl, "stable": bool(stable),
                                           "consecutive": stab_consec,
                                           "max_std_norm": conc_s.get("max_std_norm"),
                                           "e_hat": {str(t): cur[t] for t in cur}})
                prev_stab = cur
                stab_now = True
                self.costs.add("stability_check", time.perf_counter() - t_s)
            # ---- verifier trigger (priority: warm-up, stability, timeout, phase end)
            reason = None
            if local == int(P["warmup"]):
                reason = "warmup_forced"
            elif local > int(P["warmup"]):
                if stab_now and stab_consec >= int(P["stability_consecutive"]):
                    reason = "stability"
                elif last_call is None or local - last_call >= timeout_for():
                    reason = "timeout"
            if reason is None and local == cap and last_call != local:
                reason = "phase_end"
            v2row = {"update": self.global_u, "phase": phase, "local": local, **adv_stats,
                     "adv_used_mean": diag["adv_raw_mean"], "adv_used_std": diag["adv_raw_std"],
                     "adv_norm_scope": self.flags["adv_norm_scope"] if masked else "all_rows",
                     "n_policy_rows": diag.get("n_policy_rows", batch["n_transitions"]),
                     "n_actor_steps_skipped": diag.get("n_actor_steps_skipped_no_policy_rows", 0),
                     "kl_final_epoch": diag["kl_final_epoch"], "clip_frac": diag["clip_frac"],
                     "policy_loss": diag["policy_loss"], "value_loss": diag["value_loss"],
                     "update_wall_sec": time.perf_counter() - t_upd,
                     **{f"rngpos_{k_}": rng_position(g_) for k_, g_ in self.rngs.items()},
                     "rngpos_minibatch": rng_position(agent.rng_mb),
                     # ---- R1 columns (new; excluded from bit-identity comparisons)
                     "n_epochs_run": len(diag["kl_epochs"]), "conc_scale": float(agent.actor.conc_scale),
                     **batch["d1"], **self.d1_policy_row_stats(batch, pol_rows)}
            if self.clamp_likelihood == "censored":   # R2b columns (absent in every other run)
                v2row["n_censored_rows"] = int(diag.get("n_censored_rows", 0))
                v2row["n_clamped_rows_learner"] = int((batch["clamp_side"] != 0).sum())
            self.v2_history.append(v2row)
            if reason is not None:
                t_v = time.perf_counter()
                ev, err = self.verify_candidate(self.dev_cfg)
                res = ev.res if ev is not None else None
                dt = time.perf_counter() - t_v
                self.costs.add("dev_verifier", dt)
                pts = stability_points()
                conc_s = concentration_stats(beta_fn, pts, spec.e_range)
                conc_pass = bool(conc_s.get("valid") and conc_s["max_std_norm"] <= P["conc_thr"])
                if res is None:
                    valid, br_pass, crit_name, crit_val, summ = False, False, None, None, None
                else:
                    valid = bool(res.valid)
                    crit_name, crit_val = phase_criterion(res)
                    br_pass = bool(valid and crit_val <= P["phase_thr_over_dw"][phase])
                    summ = res.summary()
                elig = bool(valid and br_pass and conc_pass)
                eligible = eligible + 1 if elig else 0
                stab_consec = 0
                last_call = local
                last_result = res
                entry_log = {"update": self.global_u, "phase": phase, "local": local, "reason": reason,
                             "actor_lr": actor_lr, "critic_lr": critic_lr,
                             "active_stages": active, "valid": valid, "error": err,
                             "criterion": crit_name, "criterion_value_over_dw": crit_val,
                             "criterion_threshold_over_dw": P["phase_thr_over_dw"][phase],
                             "br_pass": br_pass, "concentration": conc_s, "conc_pass": conc_pass,
                             "eligible": elig, "consecutive_eligible": eligible, "time_sec": dt,
                             "summary": summ,
                             "e_hat_dev": {str(t): np.asarray(mean_fn(t, d), dtype=float) for t, d in pts.items()},
                             "std_norm_dev": {str(t): beta_std_norm(*beta_fn(t, d)) for t, d in pts.items()},
                             "visitation_cumulative_phase": {k_: np.asarray(v_) for k_, v_ in vis_phase.items()}}
                self.verifier_log.append(entry_log)
                # ---- v2 per-checkpoint outputs (NPZ + CSV row)
                t_w = time.perf_counter()
                if ev is not None:
                    dsc, darr = self.drift_scalars(ev)
                    row = {"run": self.cfg["run"], "pilot": self.cfg["pilot"], "arm": self.cfg["arm"],
                           "q": spec.q, "seed": self.seed, "mode": self.mode, **self.flags,
                           "stage1_status": "stage1_untrained" if "B" not in self.phases_done + [phase] else "trained",
                           "update": self.global_u, "phase": phase, "local": local, "reason": reason,
                           "eligible": elig, "consecutive_eligible": eligible, "conc_max_std_norm": conc_s.get("max_std_norm"),
                           "phase_criterion": crit_name, "phase_criterion_value_over_dw": crit_val,
                           "kl_final_epoch": diag["kl_final_epoch"], "clip_frac": diag["clip_frac"],
                           **{k_: v_ for k_, v_ in adv_stats.items()},
                           "adv_used_mean": diag["adv_raw_mean"], "adv_used_std": diag["adv_raw_std"],
                           "elapsed_phase_wall_sec": time.perf_counter() - t_ph, "verifier_sec": dt}
                    row.update(ev.scalars)
                    row.update(dsc)
                    append_csv(csv_path, row)
                    save_npz(ev, os.path.join(self.out_dir, "checkpoints", f"u{self.global_u:05d}.npz"), **darr)
                self.costs_v2.add("v2_outputs", time.perf_counter() - t_w)
                if phase == "C":
                    self.c_elig_stats["n_calls"] += 1
                    self.c_elig_stats["n_eligible"] += int(elig)
                    self.c_elig_stats["max_consecutive_eligible"] = max(self.c_elig_stats["max_consecutive_eligible"], eligible)
                    self.c_elig_stats["consecutive_eligible_at_end"] = eligible
                if summ is not None:
                    print(f"[u{self.global_u:>5} {phase}{local:>4}] verifier({reason}) valid={valid} "
                          f"{crit_name}={round(crit_val, 5)} EXP/DW={summ['exp_root_over_dw']:.5f} "
                          f"dReach/DW={summ['dreach_over_dw']:.5f} dFull/DW={summ['dfull_over_dw']:.4f} "
                          f"C={conc_s.get('max_std_norm', float('nan')):.4f} elig={elig} consec={eligible} "
                          f"| lr={actor_lr:.3g} kl={diag['kl_final_epoch']:.4f} "
                          f"e1(0)={float(mean_fn(1, np.zeros(1))[0]):.2f} e2(0)={float(mean_fn(2, np.zeros(1))[0]):.2f}",
                          flush=True)
                else:
                    print(f"[u{self.global_u:>5} {phase}{local:>4}] verifier({reason}) INVALID: {err}", flush=True)
                if phase != "C" and eligible >= int(P["k_phase"]):
                    if would_fire is None:
                        would_fire = {"global_update": self.global_u, "local_update": local, "rule": "k_phase"}
                    if not self.fixed:
                        exit_reason = "verifier_passed"
                        break
                if phase == "C" and eligible >= int(P["k_stop"]):
                    if would_fire is None:
                        would_fire = {"global_update": self.global_u, "local_update": local, "rule": "k_stop"}
                    if not self.fixed:
                        exit_reason = "k_stop_passes"
                        c_calls = [v for v in self.verifier_log if v["phase"] == "C"]
                        win = c_calls[-int(P["k_stop"]):]
                        us = [w["update"] for w in win]
                        self.five_window = {"local_updates": [w["local"] for w in win], "global_updates": us,
                                            "adjacent_diffs": [us[i + 1] - us[i] for i in range(len(us) - 1)],
                                            "span_updates": us[-1] - us[0],
                                            "actor_lr": [w["actor_lr"] for w in win], "critic_lr": [w["critic_lr"] for w in win],
                                            "dreach_over_dw": [w["criterion_value_over_dw"] for w in win],
                                            "concentration_max_std_norm": [w["concentration"]["max_std_norm"] for w in win],
                                            "reasons": [w["reason"] for w in win],
                                            "strictly_before_cap": bool(local < cap)}
                        break
            elif local % 50 == 0:
                print(f"[u{self.global_u:>5} {phase}{local:>4}] ret={batch['mean_episode_return']:.4f} "
                      f"kl={diag['kl_final_epoch']:.4f} pl={diag['policy_loss']:.4f} vl={diag['value_loss']:.4f} "
                      f"ent={diag['entropy_post_update']:.3f} lr={actor_lr:.3g} "
                      f"e1(0)={float(mean_fn(1, np.zeros(1))[0]):.2f} e2(0)={float(mean_fn(2, np.zeros(1))[0]):.2f}",
                      flush=True)
        self.visitation_by_phase[phase] = vis_phase
        self.would_fire[phase] = would_fire
        self.curriculum_log.append({"phase": phase, "entry_update": entry, "exit_update": self.global_u,
                                    "local_updates": local, "cap": cap, "exit_reason": exit_reason,
                                    "active_stages": active, "consecutive_eligible_at_exit": eligible,
                                    "n_verifier_calls": sum(1 for v in self.verifier_log if v["phase"] == phase),
                                    "lr_first": self.lr_for(phase, 1), "lr_last": self.lr_for(phase, local),
                                    "episodes": phase_updates["episodes"], "transitions": phase_updates["transitions"],
                                    "minibatch_steps": phase_updates["minibatch_steps"],
                                    "exit_call_update": (self.verifier_log[-1]["update"] if self.verifier_log and self.verifier_log[-1]["phase"] == phase else None)})
        print(f"[phase {phase}] exit at global update {self.global_u} after {local} local updates: {exit_reason} "
              f"(lr last {self.lr_for(phase, local):.6g})", flush=True)
        if phase in ("A", "B") and last_result is not None:
            np.savez(os.path.join(self.out_dir, f"phase_{phase}_exit_arrays.npz"),
                     **stage_result_arrays(last_result, f"dev_phase{phase}_exit"),
                     exit_global_update=np.array(self.global_u), exit_local_update=np.array(local))
        if phase == "C":
            self.stop_record = {"stop_update": self.global_u, "reason": exit_reason, "phase_C_local_at_stop": local,
                                "phase_C_cap": cap, "total_episodes": self.total_episodes,
                                "total_transitions": self.total_transitions, "n_verifier_calls": len(self.verifier_log),
                                "development_stopping_criterion_satisfied": exit_reason == "k_stop_passes",
                                "development_stop_strictly_before_cap": bool(exit_reason == "k_stop_passes" and local < cap),
                                "actor_lr_at_stop": read_lr(agent)[0], "critic_lr_at_stop": read_lr(agent)[1],
                                "five_eligible_window": self.five_window, "c_eligible_stats": self.c_elig_stats}
        self.phase_timing[phase] = {
            "wall_sec": time.perf_counter() - t_ph, "process_cpu_sec": time.process_time() - c_ph,
            "updates": local, "global_entry": entry, "global_exit": self.global_u,
            "by_category_sec": {k_: v_ - cost_before.get(k_, 0.0) for k_, v_ in self.costs.t.items()}}
        if not (self.phases_done and self.phases_done[-1] == phase):   # continuation: no duplicate
            self.phases_done.append(phase)
        return exit_reason

    # ------------------------------------------------------------------ phase P (R1 method 5)
    def run_pathwise_phase(self, phase: str = "P") -> str:
        """Phase P: exact-gradient ascent on the conditional expected terminal payoff.

        Both players execute the Beta mean (no action draw, no shock draw); the learner's gradient
        flows through its mean into the actor (``agents/ppo_pathwise.py``). The ``start`` stream is
        consumed with the same calls as phase A (exploring starts on D_T through :meth:`draw_starts`,
        bin-balanced unless ``start_weights`` says otherwise, and random roles). The minibatch stream
        is untouched in the R1 form (``pathwise_epochs == 1``, no ``pathwise_minibatch``); with
        epochs x minibatches it advances by one permutation per epoch, as the PPO update does, and
        ``n_minibatch_steps`` of a history row is the number of exact-gradient steps of that update.
        The lagged opponent is refreshed at the phase entry and every
        ``snapshot_every`` global updates. The concentration head and the critic must be
        bit-identical to the parent's at the end of the phase (checked, recorded in
        ``phaseP_checks.json``; a violation raises).
        """
        from agents.ppo_pathwise import pathwise_step, pathwise_update   # lazy: the default path never imports it
        spec, P, agent, ppo_cfg = self.spec, self.P, self.agent, self.ppo_cfg
        stage = spec.T
        cap = int(P["phase_caps"][phase])
        vt = P["verifier_timeout"]
        if isinstance(vt, dict) and phase not in vt:
            raise ConfigError(f"verifier_timeout has no entry for phase {phase}")
        timeout = int(vt[phase]) if isinstance(vt, dict) else int(vt)
        entry = self.global_u
        local = 0
        t_ph, c_ph = time.perf_counter(), time.process_time()
        cost_before = dict(self.costs.t)
        agent.refresh_snapshot()
        self.snapshot_log.append({"update": self.global_u, "reason": f"phase_{phase}_entry"})
        head0 = (agent.actor.out.weight[1].detach().clone(), agent.actor.out.bias[1].detach().clone())
        critic0 = {k: v.detach().clone() for k, v in agent.critic.state_dict().items()}
        opt_critic0 = copy.deepcopy(agent.opt_critic.state_dict())
        csv_path = os.path.join(self.out_dir, f"v2_checkpoints_{phase}.csv")
        os.makedirs(os.path.join(self.out_dir, "checkpoints"), exist_ok=True)
        print(f"[phase {phase}] pathwise phase at global update {self.global_u}, cap {cap}, stage {stage}, "
              f"lr(first)={self.lr_for(phase, 1):.6g}", flush=True)
        if self.phase_start_hook is not None:
            self.phase_start_hook(phase)
        while local < cap:
            local += 1
            self.global_u += 1
            t_upd = time.perf_counter()
            lr_now = self.lr_for(phase, local)
            set_lr(agent, lr_now)
            actor_lr, critic_lr = read_lr(agent)
            n_ep = int(P["episodes_per_update"])
            rng_start = self.rngs["start"]
            d0 = self.draw_starts(stage, n_ep)
            roles = rng_start.integers(0, 2, size=n_ep)
            d_learner = np.where(roles == 0, d0, -d0).astype(float)
            t_u = time.perf_counter()
            if self.pathwise_epochs == 1 and self.pathwise_minibatch is None:
                diag = pathwise_step(agent, spec, d_learner, stage, max_grad_norm=ppo_cfg.max_grad_norm)   # R1
            else:   # R2b: matched optimiser budget (epochs x minibatches of exact-gradient steps)
                n_steps_batch = n_ep if self.pathwise_minibatch is None else int(self.pathwise_minibatch)
                diag = pathwise_update(agent, spec, d_learner, stage, self.pathwise_epochs, n_steps_batch,
                                       max_grad_norm=ppo_cfg.max_grad_norm)
            self.costs.add("train_update", time.perf_counter() - t_u)
            self.total_episodes += n_ep
            self.total_transitions += n_ep
            snap = False
            if self.global_u % int(P["snapshot_every"]) == 0:
                agent.refresh_snapshot()
                self.snapshot_log.append({"update": self.global_u, "reason": f"every_{P['snapshot_every']}"})
                snap = True
            if self.global_u in self.full_state_at:
                torch.save(self.full_state(phase), os.path.join(self.out_dir, f"state_u{self.global_u:05d}.pt"))
            if self.weights_every and self.global_u % self.weights_every == 0:
                os.makedirs(self.weights_dir, exist_ok=True)
                w_path = os.path.join(self.weights_dir, f"u{self.global_u:05d}.npz")
                agent.export_weights_npz(w_path)
                self.weights_log.append({"update": self.global_u, "phase": phase, "local": local,
                                         "file": os.path.relpath(w_path, self.out_dir)})
            self.history.append({"update": self.global_u, "phase": phase, "local": local, "actor_lr": actor_lr,
                                 "critic_lr": critic_lr, "n_episodes": n_ep, "n_transitions": n_ep,
                                 "n_minibatch_steps": int(diag.get("n_steps", 1)), "snapshot_refreshed": snap,
                                 **diag})
            self.v2_history.append({"update": self.global_u, "phase": phase, "local": local, "actor_lr": actor_lr,
                                    **diag, "update_wall_sec": time.perf_counter() - t_upd,
                                    **{f"rngpos_{k_}": rng_position(g_) for k_, g_ in self.rngs.items()},
                                    "rngpos_minibatch": rng_position(agent.rng_mb)})
            reason = "timeout" if local % timeout == 0 else ("phase_end" if local == cap else None)
            if reason is not None:
                t_v = time.perf_counter()
                ev, err = self.verify_candidate(self.dev_cfg)
                dt = time.perf_counter() - t_v
                self.costs.add("dev_verifier", dt)
                res = ev.res if ev is not None else None
                crit = None if res is None else res.full_delta_max[spec.T] / spec.dw
                self.verifier_log.append({"update": self.global_u, "phase": phase, "local": local, "reason": reason,
                                          "actor_lr": actor_lr, "valid": bool(res.valid) if res is not None else False,
                                          "error": err, "criterion": "max_D2dev_Delta2_over_dw",
                                          "criterion_value_over_dw": crit, "time_sec": dt,
                                          "summary": None if res is None else res.summary()})
                if ev is not None:
                    t_w = time.perf_counter()
                    row = {"run": self.cfg["run"], "pilot": self.cfg["pilot"], "arm": self.cfg["arm"], "q": spec.q,
                           "seed": self.seed, "mode": self.mode, **self.flags, "stage1_status": "stage1_untrained",
                           "update": self.global_u, "phase": phase, "local": local, "reason": reason,
                           "phase_criterion": "max_D2dev_Delta2_over_dw", "phase_criterion_value_over_dw": crit,
                           "loss": diag["loss"], "foc_abs_mean": diag["foc_abs_mean"], "foc_abs_max": diag["foc_abs_max"],
                           "elapsed_phase_wall_sec": time.perf_counter() - t_ph, "verifier_sec": dt}
                    row.update(ev.scalars)
                    append_csv(csv_path, row)
                    save_npz(ev, os.path.join(self.out_dir, "checkpoints", f"u{self.global_u:05d}.npz"))
                    self.costs_v2.add("v2_outputs", time.perf_counter() - t_w)
                    print(f"[u{self.global_u:>5} {phase}{local:>4}] verifier({reason}) valid={res.valid} "
                          f"eta2/DW={crit:.5f} loss={diag['loss']:.5f} foc_mean={diag['foc_abs_mean']:.3e} "
                          f"e2(0)={diag['e0']:.3f} lr={actor_lr:.3g}", flush=True)
                else:
                    print(f"[u{self.global_u:>5} {phase}{local:>4}] verifier({reason}) INVALID: {err}", flush=True)
            elif local % 50 == 0:
                print(f"[u{self.global_u:>5} {phase}{local:>4}] loss={diag['loss']:.5f} "
                      f"foc_mean={diag['foc_abs_mean']:.3e} gn={diag['grad_norm_pre_clip']:.3e} "
                      f"e2(0)={diag['e0']:.3f} lr={actor_lr:.3g}", flush=True)
        # ---- end-of-phase assertions: concentration head and critic untouched (D3)
        head_ok = bool(torch.equal(agent.actor.out.weight[1], head0[0]) and torch.equal(agent.actor.out.bias[1], head0[1]))
        critic_ok = bool(all(torch.equal(v, critic0[k]) for k, v in agent.critic.state_dict().items()))
        opt_ok = _tree_equal(opt_critic0["state"], agent.opt_critic.state_dict()["state"])   # moments and step counts (the lr is set per update)
        self.phase_checks = {"concentration_head_bit_identical": head_ok, "critic_bit_identical": critic_ok,
                             "critic_adam_state_bit_identical": opt_ok}
        write_json(os.path.join(self.out_dir, f"phase{phase}_checks.json"), self.phase_checks)
        if not head_ok:
            raise RuntimeError("concentration head (actor.out row 1) changed during phase P")
        self.visitation_by_phase[phase] = {}
        self.would_fire[phase] = None
        self.curriculum_log.append({"phase": phase, "entry_update": entry, "exit_update": self.global_u,
                                    "local_updates": local, "cap": cap, "exit_reason": "budget_exhausted",
                                    "active_stages": [stage], "lr_first": self.lr_for(phase, 1),
                                    "lr_last": self.lr_for(phase, local), "episodes": local * int(P["episodes_per_update"]),
                                    "transitions": local * int(P["episodes_per_update"]),
                                    "minibatch_steps": sum(int(h["n_minibatch_steps"]) for h in self.history
                                                           if h["phase"] == phase),
                                    "n_verifier_calls": sum(1 for v in self.verifier_log if v["phase"] == phase)})
        print(f"[phase {phase}] exit at global update {self.global_u} after {local} local updates "
              f"(head bit-identical={head_ok}, critic bit-identical={critic_ok})", flush=True)
        self.phase_timing[phase] = {
            "wall_sec": time.perf_counter() - t_ph, "process_cpu_sec": time.process_time() - c_ph,
            "updates": local, "global_entry": entry, "global_exit": self.global_u,
            "by_category_sec": {k_: v_ - cost_before.get(k_, 0.0) for k_, v_ in self.costs.t.items()}}
        if not (self.phases_done and self.phases_done[-1] == phase):
            self.phases_done.append(phase)
        return "budget_exhausted"


def record_restored_scale(run: Run, out_dir: str) -> None:
    """Overwrite manifest ``conc_scale_initial`` with the scales carried by the restored parent state."""
    man_path = os.path.join(out_dir, "manifest.json")
    man = json.load(open(man_path))
    man["conc_scale_initial"] = {"actor": float(run.agent.actor.conc_scale),
                                 "opponent": float(run.agent.opponent.conc_scale),
                                 "note": "after restoring the parent state, before the first update"}
    write_json(man_path, man)


def write_manifest(run: Run, cfg: Dict, out_dir: str, cmd: str) -> Dict:
    """Resolved run manifest (refusal of missing fields already happened in validate_config)."""
    spec, P = run.spec, run.P
    grids = {tier.name: {"stage_grid_points": {str(t): int(stage_grid(t, spec.B, tier.state_step).size)
                                               for t in range(1, spec.T + 1)},
                         "state_step": tier.state_step, "effort_step": tier.effort_step,
                         "effort_grid_points": int(effort_grid(spec.e_min, spec.e_max, tier.effort_step).size),
                         "gl_half": tier.gl_half, "D2_half": spec.domain_half(2)}
             for tier in (run.dev_cfg, run.fin_cfg)}
    grids["recovery"] = {"step": float(P["recovery_step"]), "D2_half": spec.domain_half(2)}
    man = {
        "schema": "v2_run_manifest/1", "base_commit": cfg["base_commit"], "git": git_state(),
        "pilot": cfg["pilot"], "arm": cfg["arm"], "run": cfg["run"], "q": spec.q, "seed": run.seed,
        "seed_namespaces": run.seed_namespaces, "mode": run.mode, "fixed_budget": run.fixed,
        "flags": run.flags, "reward_mode": run.flags["reward_mode"],
        "stage2_update_mode": run.flags["stage2_update_mode"], "adv_norm_scope": run.flags["adv_norm_scope"],
        "continuation_action_mode": run.flags["continuation_action_mode"],
        "continuation_value_mode": run.cont_mode, "ppo_overrides": run.ppo_overrides,
        "target_kl": run.target_kl, "conc_anneal": run.conc_anneal,
        "start_weights": run.start_weights, "clamp_likelihood": run.clamp_likelihood,
        "pathwise_epochs": run.pathwise_epochs, "pathwise_minibatch": run.pathwise_minibatch,
        "conc_scale_initial": {"actor": float(run.agent.actor.conc_scale), "opponent": float(run.agent.opponent.conc_scale)},
        "parent_checkpoint": cfg["parent_checkpoint"], "parent_sha256": cfg["parent_sha256"],
        "budget_overrides": run.overrides, "resolved_protocol": P,
        "resolved_config": {"game": dataclasses.asdict(spec), "ppo": dataclasses.asdict(run.ppo_cfg),
                            "verifier": {"development": dataclasses.asdict(run.dev_cfg),
                                         "final": dataclasses.asdict(run.fin_cfg)},
                            "lr_schedule": run.sched, "lr_decay": run.lr_decay},
        "grids": grids,
        "verifier_cadence": {"warmup": P["warmup"], "stability_every": P["stability_every"],
                             "stability_consecutive": P["stability_consecutive"],
                             "verifier_timeout": P["verifier_timeout"], "drift_thr": P["drift_thr"],
                             "kl_thr": P["kl_thr"]},
        "versions": run.versions_actual, "thread_env": run.thread_env,
        "torch_threads": torch.get_num_threads(), "host": socket.gethostname(), "cmd": cmd,
        "record_source_run": cfg["record"].get("run"), "input_config": cfg,
    }
    write_json(os.path.join(out_dir, "manifest.json"), man)
    return man


def main() -> int:
    """CLI entry."""
    p = argparse.ArgumentParser(description="v2 stagewise T=2 runner")
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    cfg = json.load(open(args.config))
    out_dir = args.out_dir
    status_path = os.path.join(out_dir, "status.json")
    if os.path.exists(status_path):
        print(f"[refuse] {out_dir} already holds status.json; not restarting", flush=True)
        return 3
    if cfg.get("mode") in PARENT_MODES and cfg.get("parent_checkpoint"):
        if not os.path.exists(cfg["parent_checkpoint"]):
            raise ConfigError(f"parent checkpoint {cfg['parent_checkpoint']} not found")
        actual = sha256_file(cfg["parent_checkpoint"])
        if actual != cfg.get("parent_sha256"):
            raise ConfigError(f"parent sha256 {actual} != config {cfg.get('parent_sha256')}")
    run = Run(cfg, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    return execute(run, cfg, out_dir, " ".join(sys.argv))


def execute(run: Run, cfg: Dict, out_dir: str, cmd: str) -> int:
    """Run the configured mode end to end (also used in-process by the tests)."""
    t_wall0 = time.perf_counter()
    t_cpu0 = time.process_time()
    status_path = os.path.join(out_dir, "status.json")
    status = {"run": cfg["run"], "pilot": cfg["pilot"], "arm": cfg["arm"], "q": run.spec.q, "seed": run.seed,
              "mode": run.mode, "state": "running", "pid": os.getpid(), "host": socket.gethostname(),
              "cmd": cmd, "start_time": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_state()}
    write_json(status_path, status)
    write_manifest(run, cfg, out_dir, cmd)
    spec = run.spec
    try:
        if run.mode == "phase_B":
            run.restore(cfg["parent_checkpoint"])
            record_restored_scale(run, out_dir)
            if run.phases_done != ["A"]:
                raise ConfigError(f"parent phases_done {run.phases_done} != ['A']")
            run.parent_ref = run.stage2_mapping()
            parent_actor = {k: v.clone() for k, v in run.agent.actor.state_dict().items()}
            if run.flags["stage2_update_mode"] == "frozen":
                run.agent.freeze_stage2_snapshot()
            phases = ["B"]
        elif run.mode == "phase_A":
            phases = ["A"]
        elif run.mode == "phase_A_continue":
            run.restore(cfg["parent_checkpoint"])
            record_restored_scale(run, out_dir)
            # an end-of-A state, or a mid-phase-A state (full_state_at) of an uninterrupted phase A
            mid_phase_a = run.phases_done == [] and run.parent_state.get("phase_done") == "A"
            if run.phases_done != ["A"] and not mid_phase_a:
                raise ConfigError(f"parent phases_done {run.phases_done} != ['A']")
            phases = ["A"]
        elif run.mode == "phase_P":
            run.restore(cfg["parent_checkpoint"])
            record_restored_scale(run, out_dir)
            if run.phases_done != ["A"]:
                raise ConfigError(f"parent phases_done {run.phases_done} != ['A']")
            phases = ["P"]
        else:
            phases = ["A", "B", "C"]
        if run.mode == "full":
            PROTOCOL.clear()
            PROTOCOL.update(run.P)
        for ph in phases:
            if ph == "P":
                run.run_pathwise_phase(ph)
            else:
                run.run_phase(ph)
            if run.mode != "full":
                torch.save(run.full_state(ph), os.path.join(out_dir, f"state_end_{ph}.pt"))
        train_wall = time.perf_counter() - t_wall0
        mean_fn, beta_fn = run.policy_fns()
        if run.mode == "full":
            # identical tail to the existing runner
            ckpt_path = os.path.join(out_dir, "checkpoint.pt")
            torch.save(run.agent.full_state(), ckpt_path)
            run.agent.export_weights_npz(os.path.join(out_dir, "checkpoint_weights.npz"))
            t_f = time.perf_counter()
            final, arrays = final_evaluation(run.agent, spec, run.seed, run.costs)
            run.costs.add("final_eval_total", time.perf_counter() - t_f)
            for ph, vis in run.visitation_by_phase.items():
                for k_, v_ in vis.items():
                    arrays[f"visitation_{ph}_{k_}"] = np.asarray(v_)
            np.savez(os.path.join(out_dir, "arrays.npz"), **arrays)
        else:
            run.agent.export_weights_npz(os.path.join(out_dir, "checkpoint_weights.npz"))
            final, arrays = {}, {}
        # ---- v2 final evaluation (dev + final tiers) and the stage-2 plot
        t_f2 = time.perf_counter()
        final_v2: Dict[str, object] = {"stage1_status": "stage1_untrained" if "B" not in run.phases_done else "trained"}
        for tier in (run.dev_cfg, run.fin_cfg):
            ev, err = run.verify_candidate(tier)
            if ev is None:
                final_v2[tier.name] = {"error": err}
                continue
            final_v2[tier.name] = ev.scalars
            save_npz(ev, os.path.join(out_dir, f"final_{tier.name}.npz"))
            if tier is run.dev_cfg:
                plot_stage2(ev, spec, os.path.join(out_dir, "stage2_final.png"),
                            title=f"{cfg['run']} {cfg['arm']} (q={spec.q:g}, u{run.global_u})")
                final_v2["drift_vs_parent"] = run.drift_scalars(ev)[0]
        # ---- C5 output-drift test at the end of phase B
        if run.mode == "phase_B":
            now = run.stage2_mapping(run.agent.frozen) if run.agent.frozen is not None else run.stage2_mapping()
            dt_ = {k: float(np.max(np.abs(now[k] - run.parent_ref[k]))) for k in ("mean", "alpha", "beta")}
            test = {"mode": run.flags["stage2_update_mode"], "grid": "dev D_2", "n_grid": int(run.dev_grid2.size),
                    "max_abs_diff_vs_freeze_time": dt_}
            if run.agent.frozen is not None:
                fz = run.agent.frozen.state_dict()
                test["snapshot_params_bit_identical_to_parent_actor"] = bool(
                    all(torch.equal(fz[k], parent_actor[k]) for k in parent_actor))
                test["snapshot_requires_grad_any"] = bool(any(p_.requires_grad for p_ in run.agent.frozen.parameters()))
                opt_ids = {id(p_) for g in run.agent.opt_actor.param_groups for p_ in g["params"]} | \
                          {id(p_) for g in run.agent.opt_critic.param_groups for p_ in g["params"]}
                test["snapshot_params_in_any_optimizer"] = bool(any(id(p_) in opt_ids for p_ in run.agent.frozen.parameters()))
                test["snapshot_training_mode"] = bool(run.agent.frozen.training)
                test["pass"] = bool(all(v == 0.0 for v in dt_.values()) and test["snapshot_params_bit_identical_to_parent_actor"]
                                    and not test["snapshot_requires_grad_any"] and not test["snapshot_params_in_any_optimizer"])
                live = run.stage2_mapping()
                test["live_network_stage2_mean_maxabs_drift"] = float(np.max(np.abs(live["mean"] - run.parent_ref["mean"])))
            np.savez(os.path.join(out_dir, "drift_test_arrays.npz"), d_grid=run.dev_grid2,
                     **{f"parent_{k}": v for k, v in run.parent_ref.items()},
                     **{f"end_{k}": v for k, v in now.items()})
            write_json(os.path.join(out_dir, "drift_test.json"), test)
        run.costs_v2.add("final_eval_v2", time.perf_counter() - t_f2)
        wall = time.perf_counter() - t_wall0
        cpu = time.process_time() - t_cpu0
        cs = run.costs.as_dict()
        cs.update({"training_wall_sec": train_wall,
                   "training_sec": run.costs.t.get("train_rollout", 0.0) + run.costs.t.get("train_update", 0.0),
                   "train_rollout_sec": run.costs.t.get("train_rollout", 0.0), "train_update_sec": run.costs.t.get("train_update", 0.0),
                   "stability_sec": run.costs.t.get("stability_check", 0.0),
                   "dev_verifier_sec": run.costs.t.get("dev_verifier", 0.0), "dev_verifier_calls": run.costs.n.get("dev_verifier", 0),
                   "final_eval_sec": run.costs.t.get("final_eval_total", 0.0), "total_wall_sec": wall,
                   "total_process_cpu_sec": cpu, "total_episodes": run.total_episodes, "total_transitions": run.total_transitions,
                   "total_updates": run.global_u, "minibatch_steps": int(sum(h["n_minibatch_steps"] for h in run.history))})
        if run.mode == "full":
            stop = run.stop_record
            final.update({"run": cfg["record"]["run"], "group": cfg["record"]["group"], "smoke": bool(run.overrides),
                          "stopping_record": stop, "curriculum": run.curriculum_log, "costs": cs,
                          "checkpoint": {"path": os.path.join(out_dir, "checkpoint.pt"),
                                         "weights_npz": os.path.join(out_dir, "checkpoint_weights.npz"),
                                         "selection_rule": "C-phase stopping point (k_stop consecutive eligible passes) "
                                                           "or C budget exhaustion; no best-checkpoint restore",
                                         "stop_update": stop["stop_update"] if stop else None},
                          "config_ref": os.path.join(out_dir, "config.json"),
                          "status_flags": {"program_completed": True,
                                           "numerically_valid": bool(final["pass_flags"].get("valid_final") and final["pass_flags"].get("valid_dev")),
                                           "development_stopping_criterion_satisfied": bool(stop and stop["development_stopping_criterion_satisfied"]),
                                           "development_stop_strictly_before_cap": bool(stop and stop["development_stop_strictly_before_cap"]),
                                           "final_overall_pass": bool(final["pass_flags"].get("overall_pass"))}})
            write_json(os.path.join(out_dir, "final_eval.json"), final)
        write_json(os.path.join(out_dir, "train_history.json"),
                   {"run": cfg["record"]["run"] if run.mode == "full" else cfg["run"],
                    "group": cfg["record"]["group"] if run.mode == "full" else cfg["arm"],
                    "history": run.history, "stability": run.stability_log,
                    "verifier_calls": run.verifier_log, "curriculum": run.curriculum_log, "snapshots": run.snapshot_log,
                    "weight_checkpoints": run.weights_log, "stopping_record": run.stop_record})
        write_json(os.path.join(out_dir, "final_v2.json"), final_v2)
        write_json(os.path.join(out_dir, "v2_run_summary.json"),
                   {"phase_timing": run.phase_timing, "would_have_fired": run.would_fire,
                    "fixed_budget": run.fixed, "costs": cs, "costs_v2": run.costs_v2.as_dict(),
                    "phases_done": run.phases_done,
                    "final_global_update": run.global_u,
                    "conc_scale_final": {"actor": float(run.agent.actor.conc_scale),
                                         "opponent": float(run.agent.opponent.conc_scale)},
                    "continuation_table": None if run.cont_table is None else run.cont_table.meta,
                    "phase_checks": getattr(run, "phase_checks", None)})
        man_path = os.path.join(out_dir, "manifest.json")
        man_now = json.load(open(man_path))
        man_now["conc_scale_final"] = {"actor": float(run.agent.actor.conc_scale),
                                       "opponent": float(run.agent.opponent.conc_scale)}
        write_json(man_path, man_now)
        if run.v2_history:
            keys = list(run.v2_history[0].keys())
            import csv as _csv
            with open(os.path.join(out_dir, "v2_updates.csv"), "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(run.v2_history)
        status.update({"state": "done", "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "total_wall_sec": wall, "exit_code": 0, "final_global_update": run.global_u})
        write_json(status_path, status)
        print(f"[done] mode={run.mode} u{run.global_u} wall={wall:.1f}s phases={run.phase_timing and {k: round(v['wall_sec'], 1) for k, v in run.phase_timing.items()}}",
              flush=True)
        return 0
    except Exception:
        tb = traceback.format_exc()
        status.update({"state": "failed", "end_time": time.strftime("%Y-%m-%d %H:%M:%S"), "traceback": tb,
                       "updates_completed": run.global_u, "exit_code": 1})
        write_json(status_path, status)
        print(tb, flush=True)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as exc:
        print(f"[config-error] {exc}", flush=True)
        sys.exit(4)
