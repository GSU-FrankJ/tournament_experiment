#!/usr/bin/env python3
"""T3-R1 joint whole-game runner (T in {2, 3}): one network for all stages, every episode simulated from
its start stage to stage T, all stages updated together.

``jt`` / ``jt2`` start every episode at the root (stage 1, d = 0); ``jt_es`` adds bin-balanced exploring
starts at stages 2 and 3 (256 root + 85 stage-2 + 171 stage-3 starts of the 512 episodes). The terminal
reward is v2.0's conditional expectation (``reward_mode = expected``), the continuation of the earlier
stages is sampled (``collect_batch_v2`` with ``frozen=None``), gamma = lambda = 1. The runner reuses the
existing modules and changes none of them; it does not import ``run.run_ms_stagewise`` (stop-rule
machinery). Where it needs a piece of ``MSRun`` it re-implements it exactly: the initial-weight digest
(``MSRun._init_digest``), the learning-rate expression (``MSRun.lr_linear``), the stream construction, the
global-RNG hardening, the development check (``MSRun.verify_candidate``). The closed-form equilibrium
enters the reporting columns of ``utils.v2_metrics.evaluate`` (T = 2) only, never a training quantity.

Config ``joint_run_config/1`` (every key explicit; an unknown key is refused with exit code 4): ``schema``,
``base_commit``, ``round`` ("t3_r1"), ``arm`` (jt, jt_es, jt2), ``run``, ``q``, ``seed``, ``record`` (the
v2.0 record of q with ``game.T`` set), ``protocol_ref``, ``record_sha256``, ``threads_per_process``,
``joint`` {T, updates, starts {root, stage2[, stage3]}, lr_windows [{first, last, start, end}], check_every},
``reward_mode`` ("expected"), ``clamp_likelihood`` ("density"), ``init_digest`` (true).

Outputs in --out-dir: run_config.json, manifest.json, status.json, joint_updates.csv, joint_checks.csv,
visitation.npz, weights/u{u:05d}.npz (every 25 updates), state_end.pt, final_eval_{final,development}.npz,
joint_run_summary.json.

Exit codes: 0 done, 1 exception, 3 refusing to restart, 4 config error, 5 process-global RNG violation
(the run finished; it counts as failed).

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python run/run_joint.py --config <run_config.json> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import os
import platform
import random
import resource
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.ppo_curriculum import PPOConfig  # noqa: E402
from agents.ppo_curriculum_v2 import CurriculumPPOv2  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler, gap_bin_index  # noqa: E402
from run.run_final_dp_br import Costs, dense_grid, make_policy_fns, write_json  # noqa: E402
from run.run_final_dp_br_round3_dense import (  # noqa: E402
    LR_SCHEDULE_REQUIRED_KEYS, PROTOCOL_REQUIRED_KEYS, ConfigError, lr_at, read_lr, set_lr,
    strict_dataclass, strict_keys)
from run.run_v2_stagewise import git_state, rng_position  # noqa: E402
from run.run_v2_T2_locked import (  # noqa: E402
    GLOBAL_RNGS, RNG_VIOLATION_EXIT, digests, global_states, seed_globals)
from run.v2_rollout import collect_batch_v2  # noqa: E402
from utils.dp_br_verifier import (  # noqa: E402
    DEV_CONFIG, FINAL_CONFIG, DomainError, VerifierConfig, concentration_stats)
from utils.theory_multistage import validate_two_stage_params  # noqa: E402
from utils.v2_metrics import V2Eval, append_csv, evaluate, save_npz  # noqa: E402

SCHEMA = "joint_run_config/1"
ROUND = "t3_r1"
ARMS = ("jt", "jt_es", "jt2")
REQUIRED = ("schema", "base_commit", "round", "arm", "run", "q", "seed", "record", "protocol_ref",
            "record_sha256", "threads_per_process", "joint", "reward_mode", "clamp_likelihood",
            "init_digest")
JOINT_KEYS = ("T", "updates", "starts", "lr_windows", "check_every")
STARTS_KEYS = {2: ("root", "stage2"), 3: ("root", "stage2", "stage3")}
LR_WIN_KEYS = ("first", "last", "start", "end")
RECORD_KEYS = ("game", "ppo", "protocol", "verifier", "lr_schedule", "versions", "device",
               "torch_threads", "dw", "B", "smoke_overrides", "seed", "q")
PROTOCOL_REF_KEYS = ("path", "sha256")


# ---------------------------------------------------------------------------------- config
def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def policy_rows_per_update(starts: Mapping[str, int], T: int) -> Dict[int, int]:
    """Learner transitions per stage in one update: an episode started at stage s contributes one
    policy row to every stage t >= s.

    Args:
        starts: Start counts ``{root, stage2[, stage3]}``.
        T: Horizon.

    Returns:
        ``{t: rows}`` for t = 1 ... T.
    """
    n = [int(starts["root"]), int(starts.get("stage2", 0)), int(starts.get("stage3", 0))]
    return {t: int(sum(n[:t])) for t in range(1, T + 1)}


def validate_config(cfg: Dict[str, Any]) -> None:
    """Refuse a config with a missing / unknown key or a type / range violation (``ConfigError``).

    Types and ranges only: the production values (512 episodes, the budgets, the start mixes, the
    windows, ``check_every`` 25) are pinned by ``tools/t3/t3_configs.py`` and verified by the launch checks.
    """
    if not isinstance(cfg, dict):
        raise ConfigError("run config must be a JSON object")
    missing = [k for k in REQUIRED if k not in cfg]
    unknown = [k for k in cfg if k not in REQUIRED]
    if missing or unknown:
        raise ConfigError(f"run config: missing {missing}, unknown {unknown}")
    if cfg["schema"] != SCHEMA:
        raise ConfigError(f"schema {cfg['schema']!r} != {SCHEMA!r}")
    if cfg["round"] != ROUND:
        raise ConfigError(f"round {cfg['round']!r} != {ROUND!r}")
    if cfg["arm"] not in ARMS:
        raise ConfigError(f"arm {cfg['arm']!r} not in {ARMS}")
    for k in ("base_commit", "run", "record_sha256"):
        if not isinstance(cfg[k], str) or not cfg[k]:
            raise ConfigError(f"{k} must be a non-empty string")
    if not _int(cfg["q"]) or not _int(cfg["seed"]):
        raise ConfigError("q and seed must be integers")
    rec = cfg["record"]
    if not isinstance(rec, dict) or any(k not in rec for k in RECORD_KEYS):
        raise ConfigError(f"record must be an object with the keys {RECORD_KEYS}")
    if int(cfg["seed"]) != int(rec["seed"]) or float(cfg["q"]) != float(rec["q"]):
        raise ConfigError("q/seed differ from the embedded record")
    pr = cfg["protocol_ref"]
    if not isinstance(pr, dict) or set(pr) != set(PROTOCOL_REF_KEYS):
        raise ConfigError(f"protocol_ref must have exactly {PROTOCOL_REF_KEYS}")
    if not isinstance(cfg["threads_per_process"], dict):
        raise ConfigError("threads_per_process must be an object")
    if cfg["reward_mode"] != "expected":
        raise ConfigError("reward_mode must be 'expected'")
    if cfg["clamp_likelihood"] != "density":
        raise ConfigError("clamp_likelihood must stay 'density'")
    if cfg["init_digest"] is not True:
        raise ConfigError("init_digest must be true")
    ppo = rec["ppo"]
    if float(ppo["gamma"]) != 1.0 or float(ppo["gae_lambda"]) != 1.0:
        raise ConfigError("gamma and gae_lambda must be 1")
    j = cfg["joint"]
    if not isinstance(j, dict) or set(j) != set(JOINT_KEYS):
        raise ConfigError(f"joint must have exactly the keys {JOINT_KEYS}")
    T = j["T"]
    if not _int(T) or T not in (2, 3):
        raise ConfigError(f"joint.T must be 2 or 3; got {T!r}")
    if int(rec["game"]["T"]) != T:
        raise ConfigError(f"record game.T {rec['game']['T']} != joint.T {T}")
    U = j["updates"]
    if not _int(U) or U < 1:
        raise ConfigError(f"joint.updates must be an int >= 1; got {U!r}")
    if not _int(j["check_every"]) or j["check_every"] < 1:
        raise ConfigError(f"joint.check_every must be an int >= 1; got {j['check_every']!r}")
    n_ep = rec["protocol"]["episodes_per_update"]
    if not _int(n_ep) or n_ep < 1:
        raise ConfigError("record.protocol.episodes_per_update must be an int >= 1")
    st = j["starts"]
    if not isinstance(st, dict) or set(st) != set(STARTS_KEYS[T]):
        raise ConfigError(f"joint.starts must have exactly the keys {STARTS_KEYS[T]} at T = {T}")
    if any(not _int(v) or v < 0 for v in st.values()):
        raise ConfigError("joint.starts values must be non-negative ints")
    if sum(st.values()) != n_ep:
        raise ConfigError(f"joint.starts sum {sum(st.values())} != episodes_per_update {n_ep}")
    wins = j["lr_windows"]
    if not isinstance(wins, list):
        raise ConfigError("joint.lr_windows must be a list")
    prev_last: Optional[int] = None
    for w in sorted(wins, key=lambda w: int(w["first"]) if isinstance(w, dict) and _int(w.get("first")) else 0):
        if not isinstance(w, dict) or set(w) != set(LR_WIN_KEYS):
            raise ConfigError(f"lr window must have exactly {LR_WIN_KEYS}")
        if not (_int(w["first"]) and _int(w["last"]) and _num(w["start"]) and _num(w["end"])):
            raise ConfigError("lr window first/last must be ints and start/end numbers")
        if not 1 <= w["first"] < w["last"]:
            raise ConfigError("lr window needs 1 <= first < last")
        if prev_last is not None and w["first"] != prev_last + 1:
            raise ConfigError("lr windows must be contiguous and non-overlapping")
        prev_last = w["last"]
    if wins and prev_last != U:
        raise ConfigError("the last lr window must end at joint.updates")


# ---------------------------------------------------------------------------------- run
class JointRun:
    """One joint run: state, the update loop, the development checks and the final evaluation."""

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
        j = cfg["joint"]
        self.U = int(j["updates"])
        self.check_every = int(j["check_every"])
        self.starts = {k: int(v) for k, v in j["starts"].items()}
        self.windows = sorted(copy.deepcopy(j["lr_windows"]), key=lambda w: int(w["first"]))
        self.rows_per_update = policy_rows_per_update(self.starts, self.T)
        self.ppo_cfg = strict_dataclass(PPOConfig, rec["ppo"], "ppo")
        self.dev_cfg = strict_dataclass(VerifierConfig, rec["verifier"]["development"],
                                        "verifier.development")
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
        # ---- RNG streams: identical construction to run/run_ms_stagewise.py
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
        self.init_state_sha256 = self._init_digest(self.agent)
        self.opt_ids = {"actor": id(self.agent.opt_actor), "critic": id(self.agent.opt_critic)}
        set_lr(self.agent, lr_at(sched, "A", 1))
        self.sampler = StartSampler(spec, P["es_bin_width"])
        if self.T == 2 and self.sampler.n_bins(2) != int(rec["es_bins_stage2"]):
            raise ConfigError("es bins differ from record")
        # ---- run state
        self.costs = Costs()
        self.cpu_costs = Costs()
        self.rows: List[Dict[str, Any]] = []
        self.check_rows: List[Dict[str, Any]] = []
        self._check_keys: Optional[List[str]] = None
        self._pending_checks: List[Dict[str, Any]] = []
        self.snapshot_log: List[Dict[str, Any]] = [{"update": 0, "reason": "init"}]
        self.weights_log: List[Dict[str, Any]] = []
        self.global_u = 0
        self.total_episodes = 0
        self.total_transitions = 0
        self.total_minibatch_steps = 0
        self.rows_by_stage = {t: 0 for t in range(1, self.T + 1)}
        self.vis = self._empty_visitation()
        self.weights_every = int(P["weights_every"])
        self.snapshot_every = int(P["snapshot_every"])
        self.weights_dir = os.path.join(out_dir, "weights")
        self.checks_csv = os.path.join(out_dir, "joint_checks.csv")
        self.entry_hook: Optional[Callable[[], None]] = None
        self.violations: List[Dict[str, str]] = []

    # ------------------------------------------------------------------ MSRun pieces, re-implemented
    @staticmethod
    def _init_digest(agent: Any) -> str:
        """SHA-256 of the actor and critic weights (sorted names, float32 bytes); pure, no RNG.

        Identical to ``run.run_ms_stagewise.MSRun._init_digest``.
        """
        h = hashlib.sha256()
        for pre, net in (("actor", agent.actor), ("critic", agent.critic)):
            sd = net.state_dict()
            for k in sorted(sd):
                h.update(f"{pre}.{k}".encode())
                h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
        return h.hexdigest()

    def lr_linear(self, start: float, end: float, first: int, last: int, local: int) -> float:
        """The linear branch of ``lr_at``: start + (end - start) (j - first) / (last - first).

        Identical to ``run.run_ms_stagewise.MSRun.lr_linear`` (the same ``lr_at`` call).
        """
        lin = dict(self.sched, kind="linear", c_start_lr=float(start), c_end_lr=float(end),
                   c_local_first=int(first), linear_denominator=int(last) - int(first))
        return lr_at(lin, "C", local)

    def lr_for(self, u: int) -> float:
        """Learning rate before global update ``u``: 3e-4 outside the windows, linear inside one."""
        dec = None
        for w in self.windows:
            if int(w["first"]) <= u <= int(w["last"]):
                dec = w
        if dec is None:
            return lr_at(self.sched, "A", u)
        return self.lr_linear(dec["start"], dec["end"], dec["first"], dec["last"], u)

    @contextlib.contextmanager
    def timed(self, key: str) -> Iterator[None]:
        """Add the wall and the process-CPU time of the block to the category ``key``."""
        w, c = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            self.costs.add(key, time.perf_counter() - w)
            self.cpu_costs.add(key, time.process_time() - c)

    # ------------------------------------------------------------------ starts
    def draw_starts(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(t0, d0, roles) of one update from the ``start`` stream, in the order of D4: the root starts
        (no draw), the stage-2 balanced starts, the stage-3 balanced starts, then the roles."""
        rng = self.rngs["start"]
        n_root, n_2, n_3 = (self.starts.get(k, 0) for k in ("root", "stage2", "stage3"))
        t0 = np.concatenate([np.full(n_root, 1), np.full(n_2, 2), np.full(n_3, 3)]).astype(np.int64)
        parts = [np.zeros(n_root)]
        if n_2 > 0:
            parts.append(self.sampler.balanced(2, n_2, rng))
        if n_3 > 0:
            parts.append(self.sampler.balanced(3, n_3, rng))
        d0 = np.concatenate(parts)
        roles = rng.integers(0, 2, size=int(self.P["episodes_per_update"]))
        return t0, d0, roles

    # ------------------------------------------------------------------ visitation
    def _empty_visitation(self) -> Dict[str, np.ndarray]:
        vis: Dict[str, np.ndarray] = {}
        for t in range(2, self.T + 1):
            nb = self.sampler.n_bins(t)
            vis[f"stage{t}_root_path"] = np.zeros(nb, dtype=np.int64)
            vis[f"stage{t}_direct_es"] = np.zeros(nb, dtype=np.int64)
            for s in range(2, t):
                vis[f"stage{t}_from_stage{s}_start_approx"] = np.zeros(nb, dtype=np.int64)
        return vis

    def _accumulate_visitation(self, batch: Dict[str, Any], t0: np.ndarray) -> None:
        """Exact counts from ``collect_batch_v2`` (root-path and direct-start rows); approximate counts
        (decoded float32 observation, float64 bin index) for rows of stage t that come from a direct
        start at an earlier stage s >= 2. Within the stage-t block the rows are the episodes with
        t0 <= t in ascending order."""
        bw = float(self.P["es_bin_width"])
        for key, cnt in batch["visitation"].items():
            self.vis[key] += np.asarray(cnt, dtype=np.int64)
        stage = np.asarray(batch["stage"])
        for t in range(3, self.T + 1):
            idx = np.nonzero(t0 <= t)[0]
            block0 = int(np.searchsorted(stage, t))
            for s in range(2, t):
                pos = np.nonzero(t0[idx] == s)[0]
                if pos.size == 0:
                    continue
                obs_d = np.asarray(batch["states"])[block0 + pos, 1].astype(np.float64)
                d = obs_d * ((t - 1) * self.spec.B)
                b, nb = gap_bin_index(self.spec, t, d, bw)
                self.vis[f"stage{t}_from_stage{s}_start_approx"] += np.bincount(b, minlength=nb)

    # ------------------------------------------------------------------ development check
    def development_check(self, u: int) -> None:
        """Pure development-tier evaluation of the live actor (moves no RNG); one row of the CSV."""
        spec = self.spec
        t_v = time.perf_counter()
        ev: Optional[V2Eval] = None
        err = ""
        conc: Dict[str, float] = {f"conc_max_std_norm_t{t}": float("nan") for t in range(1, self.T + 1)}
        with self.timed("dev_verifier"):
            mean_fn, beta_fn = make_policy_fns(self.agent, spec)
            try:
                ev = evaluate(mean_fn, spec, self.dev_cfg, beta_fn=beta_fn,
                              recovery_step=float(self.P["recovery_step"]))
            except (DomainError, FloatingPointError, ValueError) as exc:
                err = f"{type(exc).__name__}: {exc}"
            if ev is not None:
                for t in range(1, self.T + 1):
                    grid = np.zeros(1) if t == 1 else dense_grid(spec.domain_half(t), self.dev_cfg.state_step)
                    cs = concentration_stats(beta_fn, {t: grid}, spec.e_range)
                    conc[f"conc_max_std_norm_t{t}"] = float(cs.get("max_std_norm", float("nan")))
        dt = time.perf_counter() - t_v
        row: Dict[str, Any] = {"update": u, **{f"rows_t{t}": self.rows_by_stage[t]
                                               for t in range(1, self.T + 1)}}
        if ev is not None:
            row.update(ev.scalars)
        row.update(conc)
        row.update({"verifier_sec": dt, "error": err})
        self._write_check_row(row, ev is not None)
        self.check_rows.append(row)

    def _write_check_row(self, row: Dict[str, Any], ok: bool) -> None:
        """Append a check row. The key set of ``evaluate``'s scalars is fixed for a given T: the header
        is the first successful row; rows of failed evaluations before it are held and written with NaN."""
        if self._check_keys is None:
            if not ok:
                self._pending_checks.append(row)
                return
            self._check_keys = list(row)
            pending, self._pending_checks = self._pending_checks, []
            for p in pending:
                append_csv(self.checks_csv, {k: p.get(k, float("nan")) for k in self._check_keys})
        elif ok and set(row) != set(self._check_keys):
            raise RuntimeError("the key set of the evaluation scalars changed between checks")
        append_csv(self.checks_csv, {k: row.get(k, float("nan")) for k in self._check_keys})

    def flush_pending_checks(self) -> None:
        """Write the rows of failed evaluations when no evaluation ever succeeded (base columns only)."""
        for p in self._pending_checks:
            append_csv(self.checks_csv, p)
        self._pending_checks = []

    # ------------------------------------------------------------------ training
    def train(self) -> None:
        """The update loop of D4 (global updates 1 ... U)."""
        spec, P, agent = self.spec, self.P, self.agent
        T = self.T
        gamma, lam = float(self.ppo_cfg.gamma), float(self.ppo_cfg.gae_lambda)
        bin_w = float(P["es_bin_width"])
        agent.refresh_snapshot()
        self.snapshot_log.append({"update": 0, "reason": "entry"})
        if self.entry_hook is not None:
            self.entry_hook()
        print(f"[joint] entry: T={T} U={self.U} starts={self.starts} "
              f"policy rows/update={self.rows_per_update}", flush=True)
        for u in range(1, self.U + 1):
            self.global_u = u
            t_upd = time.perf_counter()
            lr_now = self.lr_for(u)
            set_lr(agent, lr_now)
            actor_lr, critic_lr = read_lr(agent)
            if id(agent.opt_actor) != self.opt_ids["actor"] or id(agent.opt_critic) != self.opt_ids["critic"]:
                raise RuntimeError("optimizer object replaced")
            t0, d0, roles = self.draw_starts()
            with self.timed("train_rollout"):
                batch = collect_batch_v2(spec, agent, t0, d0, roles, self.rngs["env"], self.rngs["learn"],
                                         self.rngs["opp"], gamma, lam, bin_w, reward_mode="expected",
                                         frozen=None, continuation_action_mode="stochastic",
                                         cont_table=None, clamp_likelihood="density")
            counts = np.bincount(np.asarray(batch["stage"]), minlength=T + 1)
            for t in range(1, T + 1):
                self.rows_by_stage[t] += int(counts[t])
            self._accumulate_visitation(batch, t0)
            with self.timed("train_update"):
                diag = agent.update(batch["states"], batch["actions"], batch["logp"], batch["returns"],
                                    batch["advantages"])
            self.total_episodes += int(batch["n_episodes"])
            self.total_transitions += int(batch["n_transitions"])
            self.total_minibatch_steps += int(diag["n_minibatch_steps"])
            snap = False
            if u % self.snapshot_every == 0:
                agent.refresh_snapshot()
                self.snapshot_log.append({"update": u, "reason": f"every_{self.snapshot_every}"})
                snap = True
            if self.weights_every and u % self.weights_every == 0:
                os.makedirs(self.weights_dir, exist_ok=True)
                w_path = os.path.join(self.weights_dir, f"u{u:05d}.npz")
                agent.export_weights_npz(w_path)
                self.weights_log.append({"update": u, "file": os.path.relpath(w_path, self.out_dir)})
            if u % self.check_every == 0:
                self.development_check(u)
            eff = batch["mean_effort_by_stage"]
            self.rows.append({
                "update": u, "lr": actor_lr, "n_episodes": int(batch["n_episodes"]),
                "n_transitions": int(batch["n_transitions"]), "n_minibatch_steps": int(diag["n_minibatch_steps"]),
                **{f"rows_t{t}": int(counts[t]) for t in range(1, T + 1)},
                "mean_episode_return": batch["mean_episode_return"],
                **{f"mean_effort_t{t}": float(eff.get(t, float("nan"))) for t in range(1, T + 1)},
                "kl_final_epoch": diag["kl_final_epoch"], "clip_frac": diag["clip_frac"],
                "policy_loss": diag["policy_loss"], "value_loss": diag["value_loss"],
                "adv_raw_mean": diag["adv_raw_mean"], "adv_raw_std": diag["adv_raw_std"],
                "snapshot_refreshed": snap, "critic_lr": critic_lr,
                **{f"rngpos_{k_}": rng_position(g_) for k_, g_ in self.rngs.items()},
                "rngpos_minibatch": rng_position(agent.rng_mb),
                "update_wall_sec": time.perf_counter() - t_upd})
            if u % 50 == 0 or u == self.U:
                print(f"[u{u:>5}] ret={batch['mean_episode_return']:.4f} kl={diag['kl_final_epoch']:.4f} "
                      f"lr={actor_lr:.3g}", flush=True)

    # ------------------------------------------------------------------ evaluation / state
    def evaluate_tier(self, cfg: VerifierConfig) -> V2Eval:
        """``utils.v2_metrics.evaluate`` of the live actor on one verifier tier (pure, no RNG)."""
        mean_fn, beta_fn = make_policy_fns(self.agent, self.spec)
        return evaluate(mean_fn, self.spec, cfg, beta_fn=beta_fn,
                        recovery_step=float(self.P["recovery_step"]))

    def full_state(self) -> Dict[str, object]:
        """Complete end-of-run state (format ``joint_full_state/1``)."""
        return {
            "format": "joint_full_state/1", "agent": self.agent.full_state(),
            "rng": {k: copy.deepcopy(g.bit_generator.state) for k, g in self.rngs.items()},
            "torch_generator_state": self.torch_gen.get_state(),
            "torch_global_rng_state": torch.get_rng_state(),
            "numpy_global_rng_state": np.random.get_state(), "python_random_state": random.getstate(),
            "counters": {"global_u": self.global_u, "total_episodes": self.total_episodes,
                         "total_transitions": self.total_transitions,
                         "total_minibatch_steps": self.total_minibatch_steps,
                         "rows_by_stage": dict(self.rows_by_stage),
                         "snapshot_refreshes": self.agent.snapshot_refreshes},
            "snapshot_log": copy.deepcopy(self.snapshot_log),
            "schedule_positions": {"actor_lr": read_lr(self.agent)[0], "critic_lr": read_lr(self.agent)[1],
                                   "global_u": self.global_u},
            "seed": self.seed, "q": self.spec.q, "seed_namespaces": self.seed_namespaces}

    def write_visitation(self) -> None:
        """``visitation.npz``: per-bin counts of the learner-signed pre-action gap of the stage-t rows
        (t >= 2) by start stage, with the bin edges. Arrays whose name ends in ``_approx`` are approximate."""
        meta = {"bin_width": float(self.P["es_bin_width"]), "updates": self.global_u,
                "rows_by_stage": {str(t): v for t, v in self.rows_by_stage.items()},
                "exact": [k for k in self.vis if not k.endswith("_approx")],
                "approximate": [k for k in self.vis if k.endswith("_approx")]}
        edges = {f"stage{t}_bin_edges": self.sampler.bin_edges(t) for t in range(2, self.T + 1)}
        np.savez(os.path.join(self.out_dir, "visitation.npz"), **self.vis, **edges,
                 meta=np.array(json.dumps(meta)))


# ---------------------------------------------------------------------------------- pipeline
def write_manifest(run: JointRun, cfg: Dict[str, Any], out_dir: str, cmd: str, git: Dict[str, Any],
                   rng_log: Dict[str, Any], config_file_sha256: Optional[str]) -> Dict[str, Any]:
    """Resolved run manifest (commit, clean-tree flag, config / record SHA-256, versions, init digest)."""
    spec = run.spec
    man = {
        "schema": "joint_run_manifest/1", "commit": git["commit"], "clean_tree": git["dirty"] is False,
        "git": git, "base_commit": cfg["base_commit"], "round": cfg["round"], "arm": cfg["arm"],
        "run": cfg["run"], "q": spec.q, "seed": run.seed, "T": run.T, "seed_namespaces": run.seed_namespaces,
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "config_file_sha256": config_file_sha256,
        "record_sha256": cfg["record_sha256"], "protocol_ref": cfg["protocol_ref"],
        "joint": cfg["joint"], "policy_rows_per_update": {str(t): v for t, v in run.rows_per_update.items()},
        "reward_mode": cfg["reward_mode"], "clamp_likelihood": cfg["clamp_likelihood"],
        "init_state_sha256": run.init_state_sha256,
        "resolved_protocol": run.P, "lr_schedule": run.sched,
        "resolved_config": {"game": {k: getattr(spec, k) for k in ("w_h", "w_l", "k", "q", "T", "e_min", "e_max")},
                            "ppo": run.rec["ppo"], "verifier": run.rec["verifier"]},
        "versions": run.versions_actual, "thread_env": run.thread_env, "torch_threads": torch.get_num_threads(),
        "host": socket.gethostname(), "cmd": cmd, "global_rng": rng_log, "input_config": cfg}
    write_json(os.path.join(out_dir, "manifest.json"), man)
    return man


def write_histories(run: JointRun, out_dir: str) -> None:
    """Per-update CSV (``joint_updates.csv``)."""
    if not run.rows:
        return
    keys: List[str] = []
    for r in run.rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(out_dir, "joint_updates.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(run.rows)


def scalars_json(ev: V2Eval) -> Dict[str, Any]:
    """The scalars of an evaluation as plain JSON values (NaN stays a float)."""
    return dict(ev.scalars)


def run_pipeline(cfg: Dict[str, Any], out_dir: str, cmd: str, config_file_sha256: Optional[str] = None) -> int:
    """Seed globals -> build the run -> train -> final evaluation -> outputs. Returns the exit code."""
    seed = int(cfg["seed"])
    os.makedirs(out_dir, exist_ok=True)
    cfg_copy = os.path.join(out_dir, "run_config.json")
    if not os.path.exists(cfg_copy):
        with open(cfg_copy, "w") as f:
            json.dump(cfg, f, indent=1)
    seed_globals(seed)
    rng_log: Dict[str, Any] = {"seeding": {"seeds": {k: seed for k in GLOBAL_RNGS},
                                           "digests": digests(global_states())}}
    run = JointRun(cfg, out_dir)
    rng_log["after_run_construction"] = {"digests": digests(global_states())}
    ref: Dict[str, object] = {}

    def hook() -> None:
        if not ref:
            ref.update(global_states())
            rng_log["reference_before_first_update"] = {"digests": digests(ref)}
    run.entry_hook = hook

    def check(point: str) -> None:
        dg = digests(global_states())
        rng_log[point] = {"digests": dg}
        rd = rng_log["reference_before_first_update"]["digests"]
        for k in GLOBAL_RNGS:
            if dg[k] != rd[k]:
                run.violations.append({"rng": k, "point": point})

    t_start, c_start = time.perf_counter(), time.process_time()
    git = git_state()
    status = {"run": cfg["run"], "arm": cfg["arm"], "q": run.spec.q, "seed": run.seed, "state": "running",
              "pid": os.getpid(), "host": socket.gethostname(), "cmd": cmd,
              "start_time": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git}
    write_json(os.path.join(out_dir, "status.json"), status)
    man = write_manifest(run, cfg, out_dir, cmd, git, rng_log, config_file_sha256)
    try:
        run.train()
        check("end_of_training")
        with run.timed("final_eval"):
            evs = {tier.name: run.evaluate_tier(tier) for tier in (run.fin_cfg, run.dev_cfg)}
        for name, ev in evs.items():
            save_npz(ev, os.path.join(out_dir, f"final_eval_{name}.npz"))
        check("after_final_eval")
        torch.save(run.full_state(), os.path.join(out_dir, "state_end.pt"))
        run.write_visitation()
        run.flush_pending_checks()
        write_histories(run, out_dir)
        check("end_of_run")
        rng_ok = not run.violations
        wall, cpu = time.perf_counter() - t_start, time.process_time() - c_start
        summary = {
            "schema": "joint_run_summary/1", "run": cfg["run"], "arm": cfg["arm"], "q": run.spec.q,
            "seed": run.seed, "T": run.T, "updates": run.global_u, "episodes": run.total_episodes,
            "transitions": run.total_transitions, "minibatch_steps": run.total_minibatch_steps,
            "policy_rows_by_stage": {str(t): v for t, v in run.rows_by_stage.items()},
            "policy_rows_total": int(sum(run.rows_by_stage.values())),
            "policy_rows_per_update": {str(t): v for t, v in run.rows_per_update.items()},
            "wall_sec": wall, "process_cpu_sec": cpu,
            "costs_wall_sec": dict(run.costs.t), "costs_cpu_sec": dict(run.cpu_costs.t),
            "costs_count": dict(run.costs.n),
            "peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "init_state_sha256": run.init_state_sha256, "n_checks": len(run.check_rows),
            "n_weight_exports": len(run.weights_log),
            "global_rng": {"status": "ok" if rng_ok else "violation", "violations": run.violations},
            "final_eval": {name: scalars_json(ev) for name, ev in evs.items()}}
        write_json(os.path.join(out_dir, "joint_run_summary.json"), summary)
        man["global_rng"] = rng_log
        write_json(os.path.join(out_dir, "manifest.json"), man)
        code = 0 if rng_ok else RNG_VIOLATION_EXIT
        status.update({"state": "done", "end_time": time.strftime("%Y-%m-%d %H:%M:%S"), "exit_code": code,
                       "final_global_update": run.global_u, "total_wall_sec": wall,
                       "outcome": "done" if rng_ok else "global_rng_violation"})
        write_json(os.path.join(out_dir, "status.json"), status)
        print(f"[done] arm={cfg['arm']} q={run.spec.q:g} seed={run.seed} u{run.global_u} "
              f"global_rng={'ok' if rng_ok else run.violations} wall={wall:.1f}s", flush=True)
        return code
    except Exception:
        man["global_rng"] = rng_log
        write_json(os.path.join(out_dir, "manifest.json"), man)
        with contextlib.suppress(Exception):
            write_histories(run, out_dir)
        status.update({"state": "failed", "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "traceback": traceback.format_exc(), "updates_completed": run.global_u, "exit_code": 1})
        write_json(os.path.join(out_dir, "status.json"), status)
        print(traceback.format_exc(), flush=True)
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry."""
    p = argparse.ArgumentParser(description="T3-R1 joint whole-game runner", allow_abbrev=False)
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    with open(a.config, "rb") as f:
        raw = f.read()
    cfg = json.loads(raw.decode())
    out_dir = a.out_dir
    if os.path.exists(os.path.join(out_dir, "status.json")):
        print(f"[refuse] {out_dir} already holds status.json; not restarting", flush=True)
        return 3
    validate_config(cfg)
    return run_pipeline(cfg, out_dir, " ".join(sys.argv), hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as exc:
        print(f"[config-error] {exc}", flush=True)
        sys.exit(4)
