#!/usr/bin/env python3
"""Dense-C re-runner (2026-09-22): round-2 protocol with a per-phase verifier cadence.

Verbatim copy of ``run/run_final_dp_br_round2.py`` from worktree
``tournament-dp-br-verify-4df573`` (sha256 c2a503dd06b689d13cb8ede2e76582c248bb7c25555c93ef212b546713cfdad7)
with exactly two additions, both recorded in the manifest record:

  - ``protocol.verifier_timeout`` may be a per-phase mapping (e.g. {"A": 100, "B": 100, "C": 25})
    instead of a scalar, so only phase C is densified;
  - ``protocol.weights_every``: actor/critic weights are exported every N global updates to
    ``weights/u<update>.npz`` and indexed in train_history under ``weight_checkpoints``.

Neither addition touches training. ``verify`` and ``concentration_stats`` are pure functions of
the policy network and consume no RNG, and the weight export is read-only, so a run with a denser
cadence follows the SAME trajectory as the original until the C stopping rule fires.

Original round-2 docstring follows.

Round-2 T=2 four-group runner (2026-09-09): explicit per-run configuration + C-phase LR schedule.

Protocol source: ``MultiStage/Discussion/ROUND2_T2_FOUR_GROUP_SETTINGS_20260909.md`` and the
40-run manifest ``ROUND2_T2_FOUR_GROUP_RUNS_20260909.json``. Every experimental value is taken
from the manifest record selected by (group, q, seed); nothing is filled from code defaults.
Differences from the round-1 runner (``run/run_final_dp_br.py``):

  - C-phase development threshold 0.01 (from the record), A cap 400 or 600 (from the record);
  - actor AND critic learning rate written into every optimizer parameter group before each
    PPO update: A/B = ab_lr; C = constant, or linear
        lr_C(j) = c_start + (c_end - c_start) * (j - c_local_first) / linear_denominator,
    j = local C update (1..1000); constant inside an update; Adam objects/states preserved;
  - actual actor_lr / critic_lr recorded per update and per verifier call;
  - phase A / B exit verifier arrays saved (phase_A_exit_arrays.npz, phase_B_exit_arrays.npz)
    from the last development call of that phase (no extra call);
  - the five-consecutive-eligible window (updates, spans, LRs, dReach, concentration, reasons)
    and C-phase eligible statistics recorded;
  - refuses to start if the output directory already holds a status.json.

Training rollout / PPO update / verifier / final evaluation reuse the round-1 implementation
(``agents/ppo_curriculum.py``, ``envs/curriculum_env.py``, ``utils/dp_br_verifier.py`` and the
helpers of ``run/run_final_dp_br.py``).

Formal run:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run/run_final_dp_br_round2.py \
        --group G3_A400_linear --q 50 --seed 47
Smoke (flow only; overrides recorded in config.json smoke_overrides):
    python run/run_final_dp_br_round2.py --group G1_A400_const --q 50 --seed 47 --smoke \
        --smoke-phase-caps 2,2,3 --smoke-warmup 1 --smoke-stability-every 1 --smoke-timeout 1 \
        --smoke-direct-rollout-episodes 1000 --smoke-direct-rollout-reps 1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.ppo_curriculum import CurriculumPPO, PPOConfig  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402
from run.run_final_dp_br import (  # noqa: E402
    PROTOCOL,
    Costs,
    _git_commit,
    collect_batch,
    dense_grid,
    final_evaluation,
    make_policy_fns,
    run_verifier,
    write_json,
)
from utils.dp_br_verifier import (  # noqa: E402
    DEV_CONFIG,
    FINAL_CONFIG,
    VerifierConfig,
    VerifierResult,
    beta_std_norm,
    concentration_stats,
    stage_result_arrays,
)
from utils.theory_multistage import validate_two_stage_params  # noqa: E402

MANIFEST_DEFAULT = str(Path(__file__).resolve().parents[1] / "experiments" /
                       "two_stage_confirmation_T2_20260922" / "manifest.json")
SMOKE_OVERRIDE_KEYS = ("phase_caps", "warmup", "stability_every", "verifier_timeout",
                       "direct_rollout_episodes", "direct_rollout_reps")
PROTOCOL_REQUIRED_KEYS = {
    "w_h", "w_l", "k", "e_min", "e_max", "episodes_per_update", "phase_caps", "warmup",
    "stability_every", "drift_thr", "kl_thr", "stability_consecutive", "verifier_timeout",
    "k_phase", "k_stop", "phase_thr_over_dw", "conc_thr", "snapshot_every", "es_bin_width",
    "phase_c_root", "phase_c_es", "final_thr_over_dw", "refine_thr_over_dw",
    "sensitivity_thr_over_dw", "dense_conc_step", "recovery_step", "direct_rollout_episodes",
    "direct_rollout_reps", "direct_rollout_seed_base", "rng_namespaces", "weights_every",
}
LR_SCHEDULE_REQUIRED_KEYS = {
    "kind", "actor_and_critic", "ab_lr", "c_start_lr", "c_end_lr", "c_budget_updates",
    "c_local_first", "c_local_last", "linear_denominator", "apply_before_each_update",
    "constant_within_update", "preserve_adam_state",
}


class ConfigError(RuntimeError):
    """A manifest/record inconsistency that must not be resolved by defaults."""


# ---------------------------------------------------------------------------
# Manifest loading (strict)
# ---------------------------------------------------------------------------

def load_record(manifest_path: str, group: str, q: float, seed: int) -> Tuple[Dict, Dict]:
    """Return (manifest_meta, record) for exactly one (group, q, seed)."""
    man = json.load(open(manifest_path))
    hits = [r for r in man["runs"] if r["group"] == group and float(r["q"]) == float(q)
            and int(r["seed"]) == int(seed)]
    if len(hits) != 1:
        raise ConfigError(f"manifest has {len(hits)} records for group={group} q={q} seed={seed}")
    meta = {k: v for k, v in man.items() if k != "runs"}
    return meta, hits[0]


def strict_dataclass(cls, d: Dict, what: str):
    """Instantiate a dataclass only when the key set matches its fields exactly."""
    fields = {f.name for f in dataclasses.fields(cls)}
    keys = set(d.keys())
    if keys != fields:
        raise ConfigError(f"{what}: keys {sorted(keys ^ fields)} differ from {cls.__name__} fields")
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
    return cls(**kw)


def strict_keys(d: Dict, required: set, what: str) -> None:
    missing = required - set(d.keys())
    extra = set(d.keys()) - required
    if missing or extra:
        raise ConfigError(f"{what}: missing {sorted(missing)}, unexpected {sorted(extra)}")


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def lr_at(sched: Dict, phase: str, local_j: int) -> float:
    """Learning rate for both optimizers before local update ``local_j`` of ``phase``."""
    if phase in ("A", "B"):
        return float(sched["ab_lr"])
    if phase != "C":
        raise ConfigError(f"unknown phase {phase}")
    if sched["kind"] == "constant":
        if float(sched["c_start_lr"]) != float(sched["c_end_lr"]):
            raise ConfigError("constant schedule with c_start_lr != c_end_lr")
        return float(sched["c_start_lr"])
    if sched["kind"] == "linear":
        s, e = float(sched["c_start_lr"]), float(sched["c_end_lr"])
        j0, den = int(sched["c_local_first"]), int(sched["linear_denominator"])
        return s + (e - s) * (local_j - j0) / den
    raise ConfigError(f"unknown lr_schedule.kind {sched['kind']!r}")


def set_lr(agent: CurriculumPPO, lr: float) -> None:
    """Write ``lr`` into every parameter group of both existing Adam optimizers."""
    for opt in (agent.opt_actor, agent.opt_critic):
        for g in opt.param_groups:
            g["lr"] = float(lr)


def read_lr(agent: CurriculumPPO) -> Tuple[float, float]:
    """Actual (actor_lr, critic_lr); raises if an optimizer has mixed group LRs."""
    out = []
    for opt in (agent.opt_actor, agent.opt_critic):
        lrs = {float(g["lr"]) for g in opt.param_groups}
        if len(lrs) != 1:
            raise RuntimeError(f"mixed learning rates in one optimizer: {lrs}")
        out.append(lrs.pop())
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Round-2 four-group T=2 runner")
    p.add_argument("--manifest", default=MANIFEST_DEFAULT)
    p.add_argument("--group", required=True)
    p.add_argument("--q", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--smoke", action="store_true", help="flow smoke: all six overrides required")
    p.add_argument("--smoke-phase-caps", type=str, default=None)
    p.add_argument("--smoke-warmup", type=int, default=None)
    p.add_argument("--smoke-stability-every", type=int, default=None)
    p.add_argument("--smoke-timeout", type=int, default=None)
    p.add_argument("--smoke-direct-rollout-episodes", type=int, default=None)
    p.add_argument("--smoke-direct-rollout-reps", type=int, default=None)
    p.add_argument("--smoke-root", type=str, default=None, help="default: <result_root>/smoke")
    args = p.parse_args()

    meta, rec = load_record(args.manifest, args.group, args.q, args.seed)
    run_name = rec["run"]

    # ---- environment vs record (record, do not adapt)
    versions_actual = {"python": platform.python_version(), "torch": torch.__version__,
                       "numpy": np.__version__}
    if versions_actual != rec["versions"]:
        raise ConfigError(f"environment versions {versions_actual} != record {rec['versions']}")
    if rec["device"] != "cpu":
        raise ConfigError(f"record device {rec['device']} is not cpu")
    torch.set_num_threads(int(rec["torch_threads"]))
    thread_env = {k: os.environ.get(f"{k}_NUM_THREADS") for k in ("OMP", "MKL", "OPENBLAS")}
    for k, v in meta["threads_per_process"].items():
        if k == "torch":
            continue
        if thread_env.get(k) != str(v):
            raise ConfigError(f"{k}_NUM_THREADS={thread_env.get(k)} but manifest requires {v}")

    # ---- strict configuration objects
    spec = strict_dataclass(GameSpec, rec["game"], "game")
    ppo_cfg = strict_dataclass(PPOConfig, rec["ppo"], "ppo")
    dev_cfg = strict_dataclass(VerifierConfig, rec["verifier"]["development"], "verifier.development")
    fin_cfg = strict_dataclass(VerifierConfig, rec["verifier"]["final"], "verifier.final")
    if dev_cfg != DEV_CONFIG or fin_cfg != FINAL_CONFIG:
        raise ConfigError("record verifier tiers differ from utils.dp_br_verifier DEV/FINAL constants "
                          f"(record {dev_cfg}, {fin_cfg}; code {DEV_CONFIG}, {FINAL_CONFIG})")
    P = dict(rec["protocol"])
    strict_keys(P, PROTOCOL_REQUIRED_KEYS, "protocol")
    sched = dict(rec["lr_schedule"])
    strict_keys(sched, LR_SCHEDULE_REQUIRED_KEYS, "lr_schedule")
    if not sched["actor_and_critic"] or not sched["apply_before_each_update"] \
            or not sched["constant_within_update"] or not sched["preserve_adam_state"]:
        raise ConfigError(f"lr_schedule flags not all true: {sched}")
    if float(ppo_cfg.lr) != float(sched["ab_lr"]):
        raise ConfigError(f"ppo.lr {ppo_cfg.lr} != lr_schedule.ab_lr {sched['ab_lr']}")
    if int(sched["c_budget_updates"]) != int(P["phase_caps"]["C"]) or int(sched["c_local_last"]) != int(P["phase_caps"]["C"]):
        raise ConfigError("lr_schedule C budget differs from protocol phase_caps.C")
    for k_ in ("w_h", "w_l", "k", "e_min", "e_max"):
        if float(P[k_]) != float(getattr(spec, k_)):
            raise ConfigError(f"protocol.{k_} != game.{k_}")
    if float(rec["dw"]) != spec.dw or float(rec["B"]) != spec.B or float(rec["domain_half_stage2"]) != spec.domain_half(2):
        raise ConfigError("record dw/B/domain_half_stage2 inconsistent with game")
    if int(P["phase_c_root"]) + int(P["phase_c_es"]) != int(P["episodes_per_update"]):
        raise ConfigError("phase_c_root + phase_c_es != episodes_per_update")
    if rec["smoke_overrides"]:
        raise ConfigError(f"formal record carries smoke_overrides {rec['smoke_overrides']}")

    # ---- smoke overrides (flow test only; every override recorded)
    overrides: Dict[str, object] = {}
    if args.smoke:
        vals = {"phase_caps": args.smoke_phase_caps, "warmup": args.smoke_warmup,
                "stability_every": args.smoke_stability_every, "verifier_timeout": args.smoke_timeout,
                "direct_rollout_episodes": args.smoke_direct_rollout_episodes,
                "direct_rollout_reps": args.smoke_direct_rollout_reps}
        missing = [k for k, v in vals.items() if v is None]
        if missing:
            raise ConfigError(f"--smoke requires explicit overrides for {missing}")
        caps = [int(x) for x in vals["phase_caps"].split(",")]
        if len(caps) != 3:
            raise ConfigError("--smoke-phase-caps needs A,B,C")
        overrides = {"phase_caps": {"A": caps[0], "B": caps[1], "C": caps[2]}, "warmup": vals["warmup"],
                     "stability_every": vals["stability_every"], "verifier_timeout": vals["verifier_timeout"],
                     "direct_rollout_episodes": vals["direct_rollout_episodes"],
                     "direct_rollout_reps": vals["direct_rollout_reps"]}
        P.update(overrides)
        smoke_root = args.smoke_root or os.path.join(meta["result_root"], "smoke")
        out_dir = os.path.join(smoke_root, args.group, run_name)
    else:
        for k in ("smoke_phase_caps", "smoke_warmup", "smoke_stability_every", "smoke_timeout",
                  "smoke_direct_rollout_episodes", "smoke_direct_rollout_reps"):
            if getattr(args, k) is not None:
                raise ConfigError("smoke overrides given without --smoke")
        out_dir = rec["output_dir"]
    # module-level protocol used by the reused final_evaluation
    PROTOCOL.clear()
    PROTOCOL.update(P)

    # ---- output directory: never overwrite an existing attempt
    status_path = os.path.join(out_dir, "status.json")
    if os.path.exists(status_path):
        prev = json.load(open(status_path))
        print(f"[refuse] {out_dir} already holds status.json (state={prev.get('state')}); not restarting",
              flush=True)
        return 3
    os.makedirs(out_dir, exist_ok=True)

    val = validate_two_stage_params(q=spec.q, w_h=spec.w_h, w_l=spec.w_l, k=spec.k, e_bar=spec.e_max)
    if not val.ok:
        raise ConfigError(f"q={spec.q} fails closed-form validity: {val.messages}")

    t_wall0 = time.perf_counter()
    t_cpu0 = time.process_time()
    status = {"run": run_name, "group": args.group, "q": spec.q, "seed": args.seed, "state": "running",
              "smoke": bool(args.smoke), "pid": os.getpid(), "host": socket.gethostname(),
              "cmd": " ".join(sys.argv), "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "git_commit": _git_commit(), "thread_env": thread_env, "torch_threads": torch.get_num_threads()}
    write_json(status_path, status)

    # ---- RNG streams (seed, q, namespace); no group / phase id
    ns = P["rng_namespaces"]
    def _rng(name: str) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([args.seed, int(spec.q), int(ns[name])]))
    init_state = np.random.SeedSequence([args.seed, int(spec.q), int(ns["init"])]).generate_state(1)[0]
    torch_gen = torch.Generator().manual_seed(int(init_state))
    rng_env, rng_learn, rng_opp = _rng("env_noise"), _rng("learner_action"), _rng("opponent_action")
    rng_start, rng_mb = _rng("starts_roles"), _rng("minibatch")

    agent = CurriculumPPO(ppo_cfg, torch_gen, rng_mb, device=rec["device"])
    opt_ids = {"actor": id(agent.opt_actor), "critic": id(agent.opt_critic)}
    set_lr(agent, lr_at(sched, "A", 1))
    sampler = StartSampler(spec, P["es_bin_width"])
    if sampler.n_bins(2) != int(rec["es_bins_stage2"]):
        raise ConfigError(f"es bins {sampler.n_bins(2)} != record {rec['es_bins_stage2']}")

    config = {
        "run": run_name, "group": args.group, "q": spec.q, "seed": args.seed, "T": spec.T,
        "manifest": args.manifest, "manifest_meta": meta, "record": rec,
        "game": dataclasses.asdict(spec), "dw": spec.dw, "B": spec.B, "domain_half_stage2": spec.domain_half(2),
        "ppo": dataclasses.asdict(ppo_cfg), "protocol": P, "smoke_overrides": overrides, "smoke": bool(args.smoke),
        "lr_schedule": sched,
        "lr_schedule_formula": ("A/B: ab_lr; C constant: c_start_lr; C linear: "
                                "c_start_lr + (c_end_lr - c_start_lr) * (j - c_local_first) / linear_denominator, "
                                "j = local C update; written to all parameter groups of both Adam optimizers "
                                "before each update; Adam objects and states preserved"),
        "lr_schedule_samples": {ph: {str(j): lr_at(sched, ph, j) for j in ([1] if ph != "C" else [1, 2, 500, 501, 999, 1000])}
                                for ph in ("A", "B", "C")},
        "obs_encoding": rec["obs_encoding"], "action_mapping": rec["action_mapping"],
        "action_sampling": rec["action_sampling"], "mean_extraction": rec["mean_extraction"],
        "es_bins_stage2": sampler.n_bins(2),
        "verifier": {"development": dataclasses.asdict(dev_cfg), "final": dataclasses.asdict(fin_cfg)},
        "dtypes": rec["dtypes"], "device": rec["device"], "torch_threads": torch.get_num_threads(),
        "thread_env": thread_env, "versions_expected": rec["versions"], "versions_actual": versions_actual,
        "git_commit": status["git_commit"], "cmd": status["cmd"], "host": status["host"],
        "output_dir": out_dir, "trainable_params": {"actor": sum(p_.numel() for p_ in agent.actor.parameters()),
                                                     "critic": sum(p_.numel() for p_ in agent.critic.parameters())},
    }
    write_json(os.path.join(out_dir, "config.json"), config)
    print(f"[run] {args.group}/{run_name} q={spec.q} seed={args.seed} smoke={args.smoke} out={out_dir}", flush=True)
    print(f"[cfg] caps={P['phase_caps']} C thr={P['phase_thr_over_dw']['C']} lr={sched['kind']} "
          f"({sched['c_start_lr']}->{sched['c_end_lr']}) episodes/update={P['episodes_per_update']} "
          f"ES bins={sampler.n_bins(2)}", flush=True)

    costs = Costs()
    history: List[Dict] = []
    stability_log: List[Dict] = []
    verifier_log: List[Dict] = []
    curriculum_log: List[Dict] = []
    snapshot_log: List[Dict] = [{"update": 0, "reason": "init"}]
    weights_every = int(P["weights_every"])
    weights_log: List[Dict] = []
    weights_dir = os.path.join(out_dir, "weights")
    if weights_every:
        os.makedirs(weights_dir, exist_ok=True)
    visitation_by_phase: Dict[str, Dict[str, np.ndarray]] = {}
    phase_exit_results: Dict[str, Tuple[int, Optional[VerifierResult]]] = {}
    mean_fn, beta_fn = make_policy_fns(agent, spec)
    dev_grid2 = dense_grid(spec.domain_half(2), dev_cfg.state_step)

    total_episodes = 0
    total_transitions = 0
    global_u = 0
    stop_record: Optional[Dict] = None
    five_window: Optional[Dict] = None
    c_elig_stats = {"max_consecutive_eligible": 0, "consecutive_eligible_at_end": 0, "n_calls": 0, "n_eligible": 0}
    phases = [("A", [spec.T]), ("B", list(range(1, spec.T + 1))), ("C", list(range(1, spec.T + 1)))]

    def stability_points(phase: str) -> Dict[int, np.ndarray]:
        return {2: dev_grid2} if phase == "A" else {1: np.zeros(1), 2: dev_grid2}

    def phase_criterion(phase: str, res: VerifierResult) -> Tuple[str, float]:
        if phase == "A":
            return "max_D2dev_Delta2_over_dw", res.full_delta_max[spec.T] / spec.dw
        if phase == "B":
            return "exp_root_over_dw", res.exp_root / spec.dw
        return "dreach_over_dw", res.dreach / spec.dw

    def timeout_for(phase: str) -> int:
        """Verifier timeout for one phase; accepts a scalar or a per-phase mapping."""
        v = P["verifier_timeout"]
        return int(v[phase]) if isinstance(v, dict) else int(v)

    try:
        for phase, active in phases:
            cap = int(P["phase_caps"][phase])
            entry = global_u
            local = 0
            eligible = 0
            stab_consec = 0
            prev_stab: Optional[Dict[int, np.ndarray]] = None
            last_call: Optional[int] = None
            last_result: Optional[VerifierResult] = None
            phase_updates = {"episodes": 0, "transitions": 0, "minibatch_steps": 0}
            agent.refresh_snapshot()
            snapshot_log.append({"update": global_u, "reason": f"phase_{phase}_entry"})
            vis_phase: Dict[str, np.ndarray] = {}
            exit_reason = "budget_exhausted"
            print(f"[phase {phase}] entry at global update {global_u}, cap {cap}, active stages {active}, "
                  f"lr(first)={lr_at(sched, phase, 1):.6g}", flush=True)
            while local < cap:
                local += 1
                global_u += 1
                lr_now = lr_at(sched, phase, local)
                set_lr(agent, lr_now)
                actor_lr, critic_lr = read_lr(agent)
                if id(agent.opt_actor) != opt_ids["actor"] or id(agent.opt_critic) != opt_ids["critic"]:
                    raise RuntimeError("optimizer object replaced")
                n_ep = int(P["episodes_per_update"])
                if phase == "A":
                    t0 = np.full(n_ep, spec.T)
                    d0 = sampler.balanced(spec.T, n_ep, rng_start)
                elif phase == "B":
                    t0 = np.ones(n_ep, dtype=int)
                    d0 = np.zeros(n_ep)
                else:
                    nr, ne = int(P["phase_c_root"]), int(P["phase_c_es"])
                    t0 = np.concatenate([np.ones(nr, dtype=int), np.full(ne, spec.T)])
                    d0 = np.concatenate([np.zeros(nr), sampler.balanced(spec.T, ne, rng_start)])
                    perm = rng_start.permutation(n_ep)
                    t0, d0 = t0[perm], d0[perm]
                roles = rng_start.integers(0, 2, size=n_ep)
                t_r = time.perf_counter()
                batch = collect_batch(spec, agent, t0, d0, roles, rng_env, rng_learn, rng_opp,
                                      ppo_cfg.gamma, ppo_cfg.gae_lambda, P["es_bin_width"])
                costs.add("train_rollout", time.perf_counter() - t_r)
                t_u = time.perf_counter()
                diag = agent.update(batch["states"], batch["actions"], batch["logp"],
                                    batch["returns"], batch["advantages"])
                costs.add("train_update", time.perf_counter() - t_u)
                actor_lr_after, critic_lr_after = read_lr(agent)
                if actor_lr_after != actor_lr or critic_lr_after != critic_lr:
                    raise RuntimeError("learning rate changed inside an update")
                total_episodes += batch["n_episodes"]
                total_transitions += batch["n_transitions"]
                phase_updates["episodes"] += batch["n_episodes"]
                phase_updates["transitions"] += batch["n_transitions"]
                phase_updates["minibatch_steps"] += diag["n_minibatch_steps"]
                for k_, v_ in batch["visitation"].items():
                    vis_phase[k_] = vis_phase.get(k_, 0) + v_
                snap = False
                if global_u % int(P["snapshot_every"]) == 0:
                    agent.refresh_snapshot()
                    snapshot_log.append({"update": global_u, "reason": f"every_{P['snapshot_every']}"})
                    snap = True
                if weights_every and global_u % weights_every == 0:
                    w_path = os.path.join(weights_dir, f"u{global_u:05d}.npz")
                    agent.export_weights_npz(w_path)
                    weights_log.append({"update": global_u, "phase": phase, "local": local,
                                        "file": os.path.relpath(w_path, out_dir)})
                rec_h = {"update": global_u, "phase": phase, "local": local, "actor_lr": actor_lr, "critic_lr": critic_lr,
                         "n_episodes": batch["n_episodes"], "n_transitions": batch["n_transitions"],
                         "mean_episode_return": batch["mean_episode_return"],
                         "mean_effort_by_stage": {str(t): v for t, v in batch["mean_effort_by_stage"].items()},
                         "snapshot_refreshed": snap, **{k_: v_ for k_, v_ in diag.items() if k_ != "kl_epochs"},
                         "kl_epochs": diag["kl_epochs"]}
                history.append(rec_h)
                # ---- stability checkpoint
                stab_now = False
                if local % int(P["stability_every"]) == 0:
                    t_s = time.perf_counter()
                    pts = stability_points(phase)
                    cur = {t: np.asarray(mean_fn(t, d), dtype=float) for t, d in pts.items()}
                    drift = None
                    if prev_stab is not None:
                        drift = max(float(np.max(np.abs(cur[t] - prev_stab[t]))) / spec.e_range for t in cur)
                    kl = diag["kl_final_epoch"]
                    stable = drift is not None and drift <= P["drift_thr"] and kl <= P["kl_thr"]
                    stab_consec = stab_consec + 1 if stable else 0
                    conc_s = concentration_stats(beta_fn, pts, spec.e_range)
                    stability_log.append({"update": global_u, "phase": phase, "local": local, "drift": drift, "kl": kl,
                                          "stable": bool(stable), "consecutive": stab_consec,
                                          "max_std_norm": conc_s.get("max_std_norm"),
                                          "e_hat": {str(t): cur[t] for t in cur}})
                    prev_stab = cur
                    stab_now = True
                    costs.add("stability_check", time.perf_counter() - t_s)
                # ---- verifier trigger (priority: warm-up, stability, timeout, phase end)
                reason = None
                if local == int(P["warmup"]):
                    reason = "warmup_forced"
                elif local > int(P["warmup"]):
                    if stab_now and stab_consec >= int(P["stability_consecutive"]):
                        reason = "stability"
                    elif last_call is None or local - last_call >= timeout_for(phase):
                        reason = "timeout"
                if reason is None and local == cap and last_call != local:
                    reason = "phase_end"
                if reason is not None:
                    t_v = time.perf_counter()
                    res, err = run_verifier(mean_fn, beta_fn, spec, dev_cfg)
                    dt = time.perf_counter() - t_v
                    costs.add("dev_verifier", dt)
                    pts = stability_points(phase)
                    conc_s = concentration_stats(beta_fn, pts, spec.e_range)
                    conc_pass = bool(conc_s.get("valid") and conc_s["max_std_norm"] <= P["conc_thr"])
                    if res is None:
                        valid, br_pass, crit_name, crit_val, summ = False, False, None, None, None
                    else:
                        valid = bool(res.valid)
                        crit_name, crit_val = phase_criterion(phase, res)
                        br_pass = bool(valid and crit_val <= P["phase_thr_over_dw"][phase])
                        summ = res.summary()
                    elig = bool(valid and br_pass and conc_pass)
                    eligible = eligible + 1 if elig else 0
                    stab_consec = 0
                    last_call = local
                    last_result = res
                    entry_log = {"update": global_u, "phase": phase, "local": local, "reason": reason,
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
                    verifier_log.append(entry_log)
                    if phase == "C":
                        c_elig_stats["n_calls"] += 1
                        c_elig_stats["n_eligible"] += int(elig)
                        c_elig_stats["max_consecutive_eligible"] = max(c_elig_stats["max_consecutive_eligible"], eligible)
                        c_elig_stats["consecutive_eligible_at_end"] = eligible
                    if summ is not None:
                        print(f"[u{global_u:>5} {phase}{local:>4}] verifier({reason}) valid={valid} "
                              f"{crit_name}={round(crit_val, 5)} EXP/DW={summ['exp_root_over_dw']:.5f} "
                              f"dReach/DW={summ['dreach_over_dw']:.5f} dFull/DW={summ['dfull_over_dw']:.4f} "
                              f"C={conc_s.get('max_std_norm', float('nan')):.4f} elig={elig} consec={eligible} "
                              f"| lr={actor_lr:.3g} kl={diag['kl_final_epoch']:.4f} "
                              f"e1(0)={float(mean_fn(1, np.zeros(1))[0]):.2f} e2(0)={float(mean_fn(2, np.zeros(1))[0]):.2f}",
                              flush=True)
                    else:
                        print(f"[u{global_u:>5} {phase}{local:>4}] verifier({reason}) INVALID: {err}", flush=True)
                    if phase != "C" and eligible >= int(P["k_phase"]):
                        exit_reason = "verifier_passed"
                        break
                    if phase == "C" and eligible >= int(P["k_stop"]):
                        exit_reason = "k_stop_passes"
                        c_calls = [v for v in verifier_log if v["phase"] == "C"]
                        win = c_calls[-int(P["k_stop"]):]
                        us = [w["update"] for w in win]
                        five_window = {"local_updates": [w["local"] for w in win], "global_updates": us,
                                       "adjacent_diffs": [us[i + 1] - us[i] for i in range(len(us) - 1)],
                                       "span_updates": us[-1] - us[0],
                                       "actor_lr": [w["actor_lr"] for w in win], "critic_lr": [w["critic_lr"] for w in win],
                                       "dreach_over_dw": [w["criterion_value_over_dw"] for w in win],
                                       "concentration_max_std_norm": [w["concentration"]["max_std_norm"] for w in win],
                                       "reasons": [w["reason"] for w in win],
                                       "strictly_before_cap": bool(local < cap)}
                        break
                elif local % 50 == 0:
                    print(f"[u{global_u:>5} {phase}{local:>4}] ret={batch['mean_episode_return']:.4f} "
                          f"kl={diag['kl_final_epoch']:.4f} pl={diag['policy_loss']:.4f} vl={diag['value_loss']:.4f} "
                          f"ent={diag['entropy_post_update']:.3f} lr={actor_lr:.3g} "
                          f"e1(0)={float(mean_fn(1, np.zeros(1))[0]):.2f} e2(0)={float(mean_fn(2, np.zeros(1))[0]):.2f}",
                          flush=True)
            visitation_by_phase[phase] = vis_phase
            phase_exit_results[phase] = (global_u, last_result)
            curriculum_log.append({"phase": phase, "entry_update": entry, "exit_update": global_u,
                                   "local_updates": local, "cap": cap, "exit_reason": exit_reason,
                                   "active_stages": active, "consecutive_eligible_at_exit": eligible,
                                   "n_verifier_calls": sum(1 for v in verifier_log if v["phase"] == phase),
                                   "lr_first": lr_at(sched, phase, 1), "lr_last": lr_at(sched, phase, local),
                                   "episodes": phase_updates["episodes"], "transitions": phase_updates["transitions"],
                                   "minibatch_steps": phase_updates["minibatch_steps"],
                                   "exit_call_update": (verifier_log[-1]["update"] if verifier_log and verifier_log[-1]["phase"] == phase else None)})
            print(f"[phase {phase}] exit at global update {global_u} after {local} local updates: {exit_reason} "
                  f"(lr last {lr_at(sched, phase, local):.6g})", flush=True)
            if phase in ("A", "B") and last_result is not None:
                np.savez(os.path.join(out_dir, f"phase_{phase}_exit_arrays.npz"),
                         **stage_result_arrays(last_result, f"dev_phase{phase}_exit"),
                         exit_global_update=np.array(global_u), exit_local_update=np.array(local))
            if phase == "C":
                stop_record = {"stop_update": global_u, "reason": exit_reason, "phase_C_local_at_stop": local,
                               "phase_C_cap": cap, "total_episodes": total_episodes,
                               "total_transitions": total_transitions, "n_verifier_calls": len(verifier_log),
                               "development_stopping_criterion_satisfied": exit_reason == "k_stop_passes",
                               "development_stop_strictly_before_cap": bool(exit_reason == "k_stop_passes" and local < cap),
                               "actor_lr_at_stop": read_lr(agent)[0], "critic_lr_at_stop": read_lr(agent)[1],
                               "five_eligible_window": five_window, "c_eligible_stats": c_elig_stats}
        train_wall = time.perf_counter() - t_wall0

        # ---- freeze + persist the main checkpoint BEFORE evaluation
        ckpt_path = os.path.join(out_dir, "checkpoint.pt")
        agent.save(ckpt_path)
        agent.export_weights_npz(os.path.join(out_dir, "checkpoint_weights.npz"))

        t_f = time.perf_counter()
        final, arrays = final_evaluation(agent, spec, args.seed, costs)
        costs.add("final_eval_total", time.perf_counter() - t_f)
        for ph, vis in visitation_by_phase.items():
            for k_, v_ in vis.items():
                arrays[f"visitation_{ph}_{k_}"] = np.asarray(v_)
        np.savez(os.path.join(out_dir, "arrays.npz"), **arrays)

        wall = time.perf_counter() - t_wall0
        cpu = time.process_time() - t_cpu0
        cs = costs.as_dict()
        cs.update({"training_wall_sec": train_wall,
                   "training_sec": costs.t.get("train_rollout", 0.0) + costs.t.get("train_update", 0.0),
                   "train_rollout_sec": costs.t.get("train_rollout", 0.0), "train_update_sec": costs.t.get("train_update", 0.0),
                   "stability_sec": costs.t.get("stability_check", 0.0),
                   "dev_verifier_sec": costs.t.get("dev_verifier", 0.0), "dev_verifier_calls": costs.n.get("dev_verifier", 0),
                   "final_eval_sec": costs.t.get("final_eval_total", 0.0), "total_wall_sec": wall,
                   "total_process_cpu_sec": cpu, "total_episodes": total_episodes, "total_transitions": total_transitions,
                   "total_updates": global_u, "minibatch_steps": int(sum(h["n_minibatch_steps"] for h in history))})
        final.update({"run": run_name, "group": args.group, "smoke": bool(args.smoke), "stopping_record": stop_record,
                      "curriculum": curriculum_log, "costs": cs,
                      "checkpoint": {"path": ckpt_path, "weights_npz": os.path.join(out_dir, "checkpoint_weights.npz"),
                                     "selection_rule": "C-phase stopping point (k_stop consecutive eligible passes) "
                                                       "or C budget exhaustion; no best-checkpoint restore",
                                     "stop_update": stop_record["stop_update"] if stop_record else None},
                      "config_ref": os.path.join(out_dir, "config.json"),
                      "status_flags": {"program_completed": True,
                                       "numerically_valid": bool(final["pass_flags"].get("valid_final") and final["pass_flags"].get("valid_dev")),
                                       "development_stopping_criterion_satisfied": bool(stop_record and stop_record["development_stopping_criterion_satisfied"]),
                                       "development_stop_strictly_before_cap": bool(stop_record and stop_record["development_stop_strictly_before_cap"]),
                                       "final_overall_pass": bool(final["pass_flags"].get("overall_pass"))}})
        write_json(os.path.join(out_dir, "final_eval.json"), final)
        write_json(os.path.join(out_dir, "train_history.json"),
                   {"run": run_name, "group": args.group, "history": history, "stability": stability_log,
                    "verifier_calls": verifier_log, "curriculum": curriculum_log, "snapshots": snapshot_log,
                    "weight_checkpoints": weights_log, "stopping_record": stop_record})
        status.update({"state": "done", "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "stop_update": stop_record["stop_update"] if stop_record else None,
                       "stop_reason": stop_record["reason"] if stop_record else None,
                       "status_flags": final["status_flags"],
                       "dreach_final_over_dw": final["pass_flags"].get("dreach_final_over_dw"),
                       "total_wall_sec": wall, "exit_code": 0})
        write_json(status_path, status)
        fl = final["pass_flags"]
        tf = final["tiers"]["final"]
        print(f"[done] stop@u{stop_record['stop_update']} ({stop_record['reason']}) | final valid={tf['valid']} "
              f"EXP/DW={tf.get('exp_root_over_dw', float('nan')):.5f} dReach/DW={tf.get('dreach_over_dw', float('nan')):.5f} "
              f"dFull/DW={tf.get('dfull_over_dw', float('nan')):.4f} | δdReach={fl.get('refine_dreach_diff_over_dw', float('nan')):.5f} "
              f"δEXP={fl.get('refine_exp_diff_over_dw', float('nan')):.5f} | main={fl.get('main_pass')} overall={fl.get('overall_pass')} "
              f"| C_all={final['concentration']['C_all'].get('max_std_norm', float('nan')):.4f} "
              f"| lr_stop={stop_record['actor_lr_at_stop']:.3g} | wall={wall:.0f}s train={train_wall:.0f}s "
              f"dev={cs['dev_verifier_sec']:.1f}s final={cs['final_eval_sec']:.1f}s", flush=True)
        return 0
    except Exception:
        tb = traceback.format_exc()
        status.update({"state": "failed", "end_time": time.strftime("%Y-%m-%d %H:%M:%S"), "traceback": tb,
                       "updates_completed": global_u, "exit_code": 1})
        write_json(status_path, status)
        try:
            write_json(os.path.join(out_dir, "train_history_partial.json"),
                       {"run": run_name, "group": args.group, "history": history, "stability": stability_log,
                        "verifier_calls": verifier_log, "curriculum": curriculum_log})
        except Exception:
            pass
        print(tb, flush=True)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as exc:
        print(f"[config-error] {exc}", flush=True)
        sys.exit(4)
