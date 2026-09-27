#!/usr/bin/env python3
"""T=3 three-phase TEL-PPO run: A [3] -> B [2,3] -> C [1,2,3], then endpoint evaluation.

Phases (resolved manifest; pilot defaults)
    A  512 ES3 starts/update, cap 400, development diagnostics every 100 local updates,
       never exits early (fixed_budget_completed).
    B  512 ES2 starts/update (all continue to t3), cap 600, check every 25; exits after
       K=3 consecutive eligible calls (verifier_passed), else budget_forced.
       Eligible = valid verifier AND max_{d in D2}[V2_BR - V2_mean]/DW <= .02 AND
       max std_norm on dev grids of stages 2,3 <= .04.
    C  256 root + 85 ES2 + 171 ES3 starts/update, cap 1800, check every 25; the first
       eligible call (valid, dReach/DW <= .01, concentration on stages 1,2,3 <= .04)
       is saved as the candidate endpoint and training stops; otherwise the C-cap
       weights become a diagnostic_terminal endpoint.
Final evaluation re-reads the saved endpoint file and recomputes the development
and final verifier tiers, dense profiles/concentration and self-play economics.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B \
        experiments/three_stage_implementation_pilot_20260924/run_experiment.py \
        --manifest experiments/three_stage_implementation_pilot_20260924/manifests/smoke_q60.json --seed 10400
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (E, ResourceTracker, append_jsonl, now_iso, proc_mem, provenance,  # noqa: E402
                    thread_settings, write_json_atomic)

from agents.ppo_curriculum import CurriculumPPO, PPOConfig  # noqa: E402
from collect_t3 import collect_batch  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402
from metrics import (CoverageAccumulator, center_balanced, certification_flags,  # noqa: E402
                     evaluate_economics, policy_profiles, stage_continuation, stage_region_metrics)
from run.run_final_dp_br import make_policy_fns, run_verifier  # noqa: E402
from utils.dp_br_verifier import (VerifierConfig, VerifierResult, concentration_stats,  # noqa: E402
                                  stage_grid, stage_result_arrays)


class RunInterrupted(RuntimeError):
    """Raised by the SIGTERM handler."""


class _SkipEconomics(Exception):
    """Internal: economics disabled by the resolved config (A-only study)."""


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def ppo_config(cfg: Dict[str, Any]) -> PPOConfig:
    """PPOConfig from the resolved manifest dict."""
    d = dict(cfg["ppo"])
    d["adam_betas"] = tuple(d["adam_betas"])
    return PPOConfig(**d)


def make_streams(seed: int, q: float, namespaces: Dict[str, int]
                 ) -> Tuple[torch.Generator, Dict[str, np.random.Generator]]:
    """Training RNG streams SeedSequence([seed, q, namespace]) (init -> torch generator)."""
    gens = {name: np.random.default_rng(np.random.SeedSequence([int(seed), int(q), int(ns)]))
            for name, ns in namespaces.items() if name != "init"}
    init = np.random.SeedSequence([int(seed), int(q), int(namespaces["init"])]).generate_state(1)[0]
    return torch.Generator().manual_seed(int(init)), gens


def fresh_agent(cfg: Dict[str, Any]) -> CurriculumPPO:
    """Agent shell for loading saved weights (its own init draws are discarded)."""
    return CurriculumPPO(ppo_config(cfg), torch.Generator().manual_seed(0), np.random.default_rng(0))


def sample_starts(spec: GameSpec, sampler: StartSampler, start_counts: Dict[str, int],
                  rng: np.random.Generator, center_es: Optional[Dict[str, int]] = None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Start stages, player-0 start gaps and learner roles (starts_roles stream only).

    Root starts are exactly 0 (the interval ES sampler is never called for D1).
    ES starts use StartSampler.balanced over all of D_s; with ``center_es[s] = c``
    the last c of the n stage-s starts are instead uniform on [-B, B] (its
    width-10 bins, same draw pattern). A mixture is permuted; roles are drawn last.
    Without ``center_es`` the draws are exactly those of the pilot runner.
    """
    center_es = center_es or {}
    starts, gaps = [], []
    for s in sorted(start_counts, key=int):
        n = int(start_counts[s])
        stage = int(s)
        nc = int(center_es.get(s, 0))
        starts.append(np.full(n, stage, dtype=int))
        if stage == 1:
            if nc:
                raise ValueError("center ES is undefined for the root stage")
            gaps.append(np.zeros(n))
        elif nc == 0:
            gaps.append(sampler.balanced(stage, n, rng))
        else:
            if not 0 < nc <= n:
                raise ValueError(f"center ES count {nc} outside (0, {n}]")
            gaps.append(np.concatenate([sampler.balanced(stage, n - nc, rng),
                                        center_balanced(spec, stage, nc, rng, sampler.bin_width, spec.B)]))
    t0, d0 = np.concatenate(starts), np.concatenate(gaps)
    if len(starts) > 1 or any(int(v) for v in center_es.values()):
        perm = rng.permutation(t0.size)
        t0, d0 = t0[perm], d0[perm]
    roles = rng.integers(0, 2, size=t0.size)
    return t0, d0, roles


def dev_points(spec: GameSpec, dev_cfg: VerifierConfig, stages: List[int]) -> Dict[int, np.ndarray]:
    """Development state grids (same grids as the verifier) for the given stages."""
    return {t: stage_grid(t, spec.B, dev_cfg.state_step) for t in stages}


# ---------------------------------------------------------------------------
# Development-call metric adapter
# ---------------------------------------------------------------------------

def _finite(x: Any) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))


def development_record(phase: Dict[str, Any], res: Optional[VerifierResult], err: Optional[str],
                       conc: Dict[str, Any], conc_thr: float, dw: float,
                       center_half: Optional[float] = None) -> Dict[str, Any]:
    """Phase criterion, validity and eligibility of one development call.

    B uses the dynamic continuation gain max_{d in D2}[V2_BR(d) - V2_mean(d)]
    (never the one-step delta_2); C uses dReach; A records the stage-3 gain as a
    diagnostic without eligibility.
    """
    name = phase["name"]
    rec: Dict[str, Any] = {"criterion": phase["criterion"], "threshold_over_dw": phase["threshold_over_dw"],
                           "check_role": phase["check_role"], "error": err}
    valid = bool(res is not None and res.valid)
    rec["valid"] = valid
    rec["invalid_reasons"] = (list(res.invalid_reasons) if res is not None
                              else [f"exception: {err}"])
    rec["summary"] = res.summary() if res is not None else None
    rec["continuation_gain_by_stage"] = stage_continuation(res) if res is not None else None
    if res is not None and center_half is not None:
        s = res.stages[max(res.stages)]
        rec["stage3_regions"] = stage_region_metrics(s.d_grid, s.v_br - s.v_mean, s.std_norm, center_half, dw)
    value = None
    if res is not None:
        if name == "A":
            value = rec["continuation_gain_by_stage"]["3"]["max_raw"]
        elif name == "B":
            value = rec["continuation_gain_by_stage"]["2"]["max_raw"]
        else:
            value = float(res.dreach)
    rec["criterion_value_raw"] = value
    rec["criterion_value_over_dw"] = value / dw if _finite(value) else None
    conc_valid = bool(conc.get("valid") and _finite(conc.get("max_std_norm")))
    rec["concentration"] = conc
    rec["conc_valid"] = conc_valid
    rec["conc_pass"] = bool(conc_valid and conc["max_std_norm"] <= conc_thr)
    if name == "A":
        rec["strategic_pass"] = None
        rec["eligible"] = None
        rec["failure_type"] = "diagnostic_only"
        return rec
    sp = bool(valid and rec["criterion_value_over_dw"] is not None
              and rec["criterion_value_over_dw"] <= phase["threshold_over_dw"])
    rec["strategic_pass"] = sp
    rec["eligible"] = bool(valid and sp and rec["conc_pass"])
    if rec["eligible"]:
        rec["failure_type"] = "eligible"
    elif not valid or not conc_valid:
        rec["failure_type"] = "invalid"
    elif not sp and not rec["conc_pass"]:
        rec["failure_type"] = "both"
    elif not sp:
        rec["failure_type"] = "strategic_only"
    else:
        rec["failure_type"] = "conc_only"
    return rec


# ---------------------------------------------------------------------------
# Phase state machine (pure control flow; I/O through callbacks)
# ---------------------------------------------------------------------------

def run_phases(phases: List[Dict[str, Any]], *,
               train_step: Callable[[Dict[str, Any], int, int], None],
               check: Callable[[Dict[str, Any], int, int], Tuple[Dict[str, Any], Any]],
               on_phase_entry: Callable[[Dict[str, Any], int], None],
               on_phase_exit: Callable[[Dict[str, Any]], None],
               log_verifier: Callable[[Dict[str, Any]], None],
               update_minimum: Callable[[Dict[str, Any], Any], None],
               save_candidate: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
    """Run A, B, C with the plan's exact control flow.

    Args:
        phases: Resolved phase dicts (name, cap, check_every, k_consecutive, exit reasons).
        train_step: ``(phase, local, global)`` collect + PPO update + logs + refresh + weights.
        check: ``(phase, local, global) -> (record, verifier_result)``.
        on_phase_entry: ``(phase, global)`` snapshot refresh + event.
        on_phase_exit: ``(phase_row)`` event/persistence.
        log_verifier: Append the finished record (with ``consecutive_eligible``).
        update_minimum: C only: keep the minimum valid call (every C call is offered).
        save_candidate: C only: persist the endpoint before any final evaluation.

    Returns:
        Summary with per-phase rows and the search outcome.
    """
    g = 0
    rows: List[Dict[str, Any]] = []
    out: Dict[str, Any] = {"has_candidate": False, "candidate_update": None, "candidate_C_local": None,
                           "search_outcome": None}
    for ph in phases:
        on_phase_entry(ph, g)
        entry = g
        consecutive = 0
        seq: List[Dict[str, Any]] = []
        exit_reason = None
        local = 0
        stop = False
        for local in range(1, int(ph["cap"]) + 1):
            train_step(ph, local, g + 1)
            g += 1
            if local % int(ph["check_every"]) != 0:
                continue
            record, result = check(ph, local, g)
            if ph["name"] != "A":
                consecutive = consecutive + 1 if record["eligible"] else 0
            record["consecutive_eligible"] = consecutive
            log_verifier(record)
            seq.append({"local": local, "global": g, "eligible": record.get("eligible"),
                        "consecutive": consecutive})
            if ph["name"] == "A":
                continue
            if ph["name"] == "C":
                update_minimum(record, result)
            if consecutive >= int(ph["k_consecutive"]):
                exit_reason = ph["pass_exit_reason"]
                if ph["name"] == "C":
                    save_candidate(record)
                    out.update(has_candidate=True, candidate_update=g, candidate_C_local=local)
                    stop = True
                break
        if exit_reason is None:
            exit_reason = ph["cap_exit_reason"]
        row = {"phase": ph["name"], "entry_global_update": entry, "exit_global_update": g,
               "local_updates": local, "cap": int(ph["cap"]), "exit_reason": exit_reason,
               "n_checks": len(seq), "eligible_count": sum(1 for s in seq if s["eligible"]),
               "consecutive_at_exit": consecutive, "check_sequence": seq}
        rows.append(row)
        on_phase_exit(row)
        if stop:
            break
    out["phases"] = rows
    out["stop_global_update"] = g
    c_rows = [r for r in rows if r["phase"] == "C"]
    out["stop_C_local_update"] = c_rows[0]["local_updates"] if c_rows else None
    out["search_outcome"] = "candidate_found" if out["has_candidate"] else (
        "no_candidate_budget_exhausted" if c_rows else "C_not_reached")
    return out


# ---------------------------------------------------------------------------
# Checkpoint files
# ---------------------------------------------------------------------------

def save_checkpoint_files(agent: CurriculumPPO, ckpt_dir: Path, stem: str,
                          identity: Dict[str, Any]) -> Dict[str, str]:
    """Write ``{stem}.pt`` and ``{stem}_weights.npz`` via temp files + os.replace."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pt = ckpt_dir / f"{stem}.pt"
    npz = ckpt_dir / f"{stem}_weights.npz"
    state = agent.state()
    state["identity"] = dict(identity)
    tmp_pt = ckpt_dir / f".{stem}.pt.tmp"
    torch.save(state, tmp_pt)
    os.replace(tmp_pt, pt)
    arrs = {f"actor.{k}": v.detach().cpu().numpy() for k, v in agent.actor.state_dict().items()}
    arrs.update({f"critic.{k}": v.detach().cpu().numpy() for k, v in agent.critic.state_dict().items()})
    arrs["identity_json"] = np.array(json.dumps(identity))
    tmp_npz = ckpt_dir / f".{stem}_weights.tmp.npz"
    np.savez(tmp_npz, **arrs)
    os.replace(tmp_npz, npz)
    return {"pt": str(pt.relative_to(ckpt_dir.parent)), "weights_npz": str(npz.relative_to(ckpt_dir.parent))}


def save_npz_atomic(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    """np.savez through a temporary ``.npz`` name and os.replace."""
    tmp = path.with_name(f".{path.stem}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Endpoint evaluation (reads the checkpoint file only; no optimizer updates)
# ---------------------------------------------------------------------------

def evaluate_checkpoint(checkpoint_path: str, resolved_config: Dict[str, Any], checkpoint_kind: str,
                        tracker: Optional[ResourceTracker] = None
                        ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], CurriculumPPO]:
    """Recompute development and final tiers, dense profiles and certification flags.

    Args:
        checkpoint_path: ``endpoint.pt``.
        resolved_config: ``manifest["resolved"]``.
        checkpoint_kind: ``candidate`` or ``diagnostic_terminal``.
        tracker: Optional ResourceTracker.

    Returns:
        ``(final_json, arrays, reloaded_agent)``.
    """
    cfg = resolved_config
    spec = GameSpec(**cfg["game"])
    agent = fresh_agent(cfg)
    ck = agent.load_weights(checkpoint_path)
    mean_fn, beta_fn = make_policy_fns(agent, spec)
    arrays: Dict[str, np.ndarray] = {}
    tiers: Dict[str, Any] = {}
    results: Dict[str, Optional[VerifierResult]] = {}
    for name in ("development", "final"):
        vcfg = VerifierConfig(**cfg["verifier"][name])
        ctx = tracker.segment(f"final_eval_verifier_{name}") if tracker else None
        seg = ctx.__enter__() if ctx else {}
        try:
            res, err = run_verifier(mean_fn, beta_fn, spec, vcfg)
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
        results[name] = res
        if res is None:
            tiers[name] = {"valid": False, "error": err, "invalid_reasons": [f"exception: {err}"]}
        else:
            s = res.summary()
            s["error"] = None
            s["continuation_gain_by_stage"] = stage_continuation(res)
            s3 = res.stages[max(res.stages)]
            s["stage3_regions"] = stage_region_metrics(s3.d_grid, s3.v_br - s3.v_mean, s3.std_norm, spec.B,
                                                       spec.dw)
            tiers[name] = s
            arrays.update(stage_result_arrays(res, name))
        tiers[name]["resources"] = dict(seg)
    ctx = tracker.segment("dense_profiles") if tracker else None
    seg = ctx.__enter__() if ctx else {}
    try:
        prof = policy_profiles(mean_fn, beta_fn, spec, cfg["final_certification"]["dense_conc_step"])
    finally:
        if ctx:
            ctx.__exit__(None, None, None)
    arrays.update({f"dense_{k}": v for k, v in prof["arrays"].items()})
    has_candidate = checkpoint_kind == "candidate"
    flags = certification_flags(tiers["development"] if results["development"] is not None else None,
                                tiers["final"] if results["final"] is not None else None,
                                prof["concentration"], has_candidate, cfg["final_certification"], spec.dw)
    e1 = float(prof["arrays"]["t1_mean_effort"][0])
    out = {
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_identity": ck.get("identity"),
        "tiers": tiers,
        "dense_concentration": prof["concentration"],
        "dense_stage3_conc_regions": stage_region_metrics(
            prof["arrays"][f"t{spec.T}_d"], np.zeros(prof["arrays"][f"t{spec.T}_d"].size),
            prof["arrays"][f"t{spec.T}_std_norm"], spec.B, spec.dw),
        "dense_points_by_stage": {str(t): int(prof["arrays"][f"t{t}_d"].size) for t in range(1, spec.T + 1)},
        "profile_mean_fn_consistency_max_abs": prof["mean_fn_consistency_max_abs"],
        "e1_mean_effort": e1,
        "pass_flags": flags,
        "certification": flags["certification"],
        "final_joint_pass": flags["final_joint_pass"],
        "dense_profile_resources": dict(seg),
        "notes": "No global MPE certification is claimed; Delta_max_all shows deviations "
                 "outside the BR-reachable criterion.",
    }
    return out, arrays, agent


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------

class Run:
    """One manifest run: training, endpoint persistence, evaluation, logs."""

    def __init__(self, manifest_path: Path, seed: int):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        runs = [r for r in self.manifest["runs"] if int(r["seed"]) == int(seed)]
        if len(runs) != 1:
            raise SystemExit(f"seed {seed} is not listed exactly once in {self.manifest_path}")
        self.run = runs[0]
        self.cfg = self.manifest["resolved"]
        assert float(self.cfg["game"]["q"]) == float(self.manifest["q"]) == float(self.run["q"])
        self.out = (E / self.run["output_dir"]).resolve()
        self.ident = {"run_id": self.run["run_id"], "cohort": self.run["cohort"],
                      "q": int(self.run["q"]), "seed": int(self.run["seed"])}
        self.spec = GameSpec(**self.cfg["game"])
        self.tracker = ResourceTracker()
        self.status: Dict[str, Any] = {}
        self.updates_completed = 0
        self.totals = {"episodes": 0, "environment_steps": 0, "physical_actions": 0}
        self.call_index = 0
        self.min_best: Optional[float] = None
        self.min_identity: Optional[Dict[str, Any]] = None
        self.endpoint: Optional[Dict[str, Any]] = None
        self.dev_records: List[Dict[str, Any]] = []
        self.phase_rows: List[Dict[str, Any]] = []
        self.current_phase: Optional[str] = None

    # ------------------------------------------------------------------ logging
    def path(self, name: str) -> Path:
        return self.out / name

    def event(self, event_name: str, **fields: Any) -> None:
        append_jsonl(self.path("events.jsonl"), {**self.ident, "event": event_name, "time": now_iso(),
                                                 "updates_completed": self.updates_completed, **fields})

    def write_status(self, **fields: Any) -> None:
        self.status.update(fields)
        self.status["last_completed_update"] = self.updates_completed
        self.status["updated_time"] = now_iso()
        write_json_atomic(self.path("status.json"), self.status)

    def resource(self, seg: Dict[str, Any]) -> None:
        append_jsonl(self.path("resources.jsonl"), {**self.ident, "time": now_iso(), **seg})

    def available_checkpoints(self) -> List[str]:
        out = []
        for sub in ("checkpoints", "weights"):
            d = self.out / sub
            if d.exists():
                out += sorted(str(p.relative_to(self.out)) for p in d.iterdir() if not p.name.startswith("."))
        return out

    def save_coverage(self) -> None:
        save_npz_atomic(self.path("coverage.npz"), self.coverage.arrays())

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / "checkpoints").mkdir()
        (self.out / "weights").mkdir()
        torch.set_num_threads(int(self.cfg["runtime"]["torch_threads"]))
        self.status = {**self.ident, "state": "running", "stage": "setup", "pid": os.getpid(),
                       "host": socket.gethostname(), "start_time": now_iso(), "end_time": None,
                       "cmd": " ".join(sys.argv), "failure_reason": None, "search_outcome": None,
                       "has_candidate": None, "endpoint": None, "available_checkpoints": []}
        self.write_status()
        config = {**self.ident, "manifest": str(self.manifest_path),
                  "manifest_name": self.manifest["manifest_name"], "output_dir": str(self.out),
                  "resolved": self.cfg, "provenance": provenance(), "threads": thread_settings(),
                  "cmd": " ".join(sys.argv), "pid": os.getpid(), "host": socket.gethostname(),
                  "start_time": self.status["start_time"],
                  "resource_peak_method": self.tracker.peak_method}
        write_json_atomic(self.path("config.json"), config)
        ns = self.cfg["rng"]["namespaces"]
        gen, self.rng = make_streams(self.ident["seed"], self.spec.q, ns)
        self.ppo_cfg = ppo_config(self.cfg)
        self.agent = CurriculumPPO(self.ppo_cfg, gen, self.rng["minibatch"], self.cfg["runtime"]["device"])
        self.sampler = StartSampler(self.spec, float(self.cfg["es_bin_width"]))
        self.mean_fn, self.beta_fn = make_policy_fns(self.agent, self.spec)
        self.dev_cfg = VerifierConfig(**self.cfg["verifier"]["development"])
        self.coverage = CoverageAccumulator(self.spec, float(self.cfg["es_bin_width"]),
                                            [p["name"] for p in self.cfg["phases"]])
        self.event("run_start", pid=os.getpid(), host=socket.gethostname())
        self.event("snapshot_refresh", reason="init", global_update=0)

    # ------------------------------------------------------------------ callbacks
    def on_phase_entry(self, ph: Dict[str, Any], g: int) -> None:
        self.current_phase = ph["name"]
        self.agent.refresh_snapshot()
        self.event("snapshot_refresh", reason=f"phase_{ph['name']}_entry", global_update=g)
        self.event("phase_entry", phase=ph["name"], global_update=g, cap=ph["cap"],
                   active_stages=ph["active_stages"], start_counts=ph["start_counts"])
        self.write_status(stage=f"training_{ph['name']}")

    def on_phase_exit(self, row: Dict[str, Any]) -> None:
        row = dict(row)
        ph_hist = self.phase_totals.get(row["phase"], {})
        row.update({f"phase_{k}": v for k, v in ph_hist.items()})
        self.phase_rows.append(row)
        self.event("phase_exit", **row)
        self.save_coverage()
        self.write_status(phases=self.phase_rows)

    def train_step(self, ph: Dict[str, Any], local: int, g: int) -> None:
        t_w, t_c = time.perf_counter(), time.process_time()
        t0, d0, roles = sample_starts(self.spec, self.sampler, ph["start_counts"], self.rng["starts_roles"],
                                      ph.get("center_es"))
        cfg = self.ppo_cfg
        batch = collect_batch(self.spec, self.agent, t0, d0, roles, self.rng["env_noise"],
                              self.rng["learner_action"], self.rng["opponent_action"], cfg.gamma,
                              cfg.gae_lambda, float(self.cfg["es_bin_width"]))
        diag = self.agent.update(batch["states"], batch["actions"], batch["logp"], batch["returns"],
                                 batch["advantages"])
        self.updates_completed = g
        self.coverage.add(ph["name"], batch)
        steps = batch["n_transitions"]
        self.totals["episodes"] += batch["n_episodes"]
        self.totals["environment_steps"] += steps
        self.totals["physical_actions"] += 2 * steps
        pt = self.phase_totals.setdefault(ph["name"], {"episodes": 0, "environment_steps": 0,
                                                      "physical_actions": 0, "updates": 0})
        pt["episodes"] += batch["n_episodes"]
        pt["environment_steps"] += steps
        pt["physical_actions"] += 2 * steps
        pt["updates"] += 1
        mem = proc_mem()
        row = {**self.ident, "phase": ph["name"], "local_update": local, "global_update": g,
               "time": now_iso(), "wall_sec": time.perf_counter() - t_w,
               "cpu_sec": time.process_time() - t_c, "rss_bytes": mem.get("rss_bytes"),
               "n_episodes": batch["n_episodes"], "n_environment_steps": steps,
               "n_physical_actions": 2 * steps, "start_counts": batch["start_counts_actual"],
               "stage_transition_counts": batch["stage_transition_counts"],
               "cum_episodes": self.totals["episodes"], "cum_environment_steps": self.totals["environment_steps"],
               "cum_physical_actions": self.totals["physical_actions"],
               "training_mean_effort_by_stage": {str(t): v for t, v in batch["mean_effort_by_stage"].items()},
               "mean_episode_return": batch["mean_episode_return"], **diag}
        append_jsonl(self.path("history.jsonl"), row)
        if g % int(self.cfg["snapshot_every"]) == 0:
            self.agent.refresh_snapshot()
            self.event("snapshot_refresh", reason="periodic", global_update=g)
        if g % int(self.cfg["weights_every"]) == 0:
            p = self.out / "weights" / f"update_{g:05d}.npz"
            self.agent.export_weights_npz(str(p))
            self.event("checkpoint_saved", kind="periodic_weights", global_update=g,
                       path=str(p.relative_to(self.out)))
        self.write_status()

    def check(self, ph: Dict[str, Any], local: int, g: int) -> Tuple[Dict[str, Any], Any]:
        self.call_index += 1
        with self.tracker.segment("dev_verifier", phase=ph["name"], global_update=g,
                                  call_index=self.call_index) as seg:
            res, err = run_verifier(self.mean_fn, self.beta_fn, self.spec, self.dev_cfg)
            conc = concentration_stats(self.beta_fn,
                                       dev_points(self.spec, self.dev_cfg, ph["concentration_stages"]),
                                       self.spec.e_range)
        rec = development_record(ph, res, err, conc, float(self.cfg["concentration_threshold"]), self.spec.dw,
                                 self.spec.B)
        snap = self.coverage.snapshot(ph["name"], local, g, self.call_index)
        rec = {**self.ident, "phase": ph["name"], "local_update": local, "global_update": g,
               "call_index": self.call_index, "tier": "development", "time": now_iso(),
               "concentration_stages": ph["concentration_stages"], **rec,
               "coverage_snapshot_index": snap, "cost": dict(seg)}
        self.resource(seg)
        return rec, res

    def log_verifier(self, rec: Dict[str, Any]) -> None:
        append_jsonl(self.path("verifier_calls.jsonl"), rec)
        self.dev_records.append(rec)
        self.save_coverage()

    def update_minimum(self, rec: Dict[str, Any], res: Optional[VerifierResult]) -> None:
        v = rec.get("criterion_value_over_dw")
        if res is None or not rec["valid"] or not _finite(v):
            return
        if self.min_best is not None and not v < self.min_best:
            return
        identity = {**self.ident, "checkpoint_kind": "minimum_development", "phase": "C",
                    "global_update": rec["global_update"], "local_update": rec["local_update"],
                    "call_index": rec["call_index"]}
        ck = self.out / "checkpoints"
        files = save_checkpoint_files(self.agent, ck, "min_dev", identity)
        arrs = stage_result_arrays(res, "minimum_development")
        arrs["identity_json"] = np.array(json.dumps(identity))
        arrs["coverage_snapshot"] = self.coverage.snap_vectors[rec["coverage_snapshot_index"]]
        arrs["coverage_phase_totals_AB"] = np.stack([self.coverage.totals[p] for p in ("A", "B")])
        save_npz_atomic(ck / "min_dev_arrays.npz", arrs)
        files["arrays_npz"] = "checkpoints/min_dev_arrays.npz"
        meta = {"identity": identity, "files": files, "summary": rec["summary"],
                "concentration": rec["concentration"],
                "continuation_gain_by_stage": rec["continuation_gain_by_stage"],
                "dreach_over_dw": v, "coverage_snapshot_index": rec["coverage_snapshot_index"],
                "saved_time": now_iso()}
        write_json_atomic(ck / "min_dev_record.json", meta)       # metadata last
        self.min_best = v
        self.min_identity = identity
        self.event("checkpoint_saved", kind="minimum_development", global_update=rec["global_update"],
                   dreach_over_dw=v, files=files)

    def save_endpoint(self, kind: str, rec: Optional[Dict[str, Any]]) -> None:
        g = self.updates_completed
        identity = {**self.ident, "checkpoint_kind": kind, "global_update": g,
                    "phase": self.current_phase,
                    "call_index": rec["call_index"] if rec else None,
                    "local_update": rec["local_update"] if rec else None}
        ck = self.out / "checkpoints"
        files = save_checkpoint_files(self.agent, ck, "endpoint", identity)
        write_json_atomic(ck / "endpoint_record.json", {"identity": identity, "files": files,
                                                        "development_call": rec, "saved_time": now_iso()})
        self.endpoint = {"identity": identity, "files": files}
        self.event("checkpoint_saved", kind=kind, global_update=g, files=files)

    def save_candidate(self, rec: Dict[str, Any]) -> None:
        self.save_endpoint("candidate", rec)

    # ------------------------------------------------------------------ main
    def execute(self) -> int:
        self.setup()
        self.phase_totals: Dict[str, Dict[str, int]] = {}
        try:
            with self.tracker.segment("training") as seg_train:
                summary = run_phases(self.cfg["phases"], train_step=self.train_step, check=self.check,
                                     on_phase_entry=self.on_phase_entry, on_phase_exit=self.on_phase_exit,
                                     log_verifier=self.log_verifier, update_minimum=self.update_minimum,
                                     save_candidate=self.save_candidate)
            self.resource(seg_train)
            if not summary["has_candidate"]:
                last = self.dev_records[-1] if self.dev_records and \
                    self.dev_records[-1]["global_update"] == self.updates_completed else None
                self.save_endpoint("diagnostic_terminal", last)
            summary["phases"] = self.phase_rows
            summary["totals"] = dict(self.totals)
            summary["updates_completed"] = self.updates_completed
            summary["n_dev_calls"] = len(self.dev_records)
            summary["min_dev_identity"] = self.min_identity
            summary["endpoint"] = self.endpoint
            write_json_atomic(self.path("training_summary.json"), summary)
            self.event("search_outcome", search_outcome=summary["search_outcome"],
                       has_candidate=summary["has_candidate"], endpoint=self.endpoint)
            self.write_status(stage="search_complete", search_outcome=summary["search_outcome"],
                              has_candidate=summary["has_candidate"], endpoint=self.endpoint,
                              stop_global_update=summary["stop_global_update"],
                              stop_C_local_update=summary["stop_C_local_update"],
                              totals=self.totals, available_checkpoints=self.available_checkpoints())
        except BaseException as exc:
            return self.fail(exc, "training")
        return self.evaluate(summary)

    def evaluate(self, summary: Dict[str, Any]) -> int:
        """Final evaluation, reload identity, min_dev profiles and economics."""
        errors: List[str] = []
        kind = self.endpoint["identity"]["checkpoint_kind"]
        final_ok = False
        final_json: Dict[str, Any] = {}
        reloaded: Optional[CurriculumPPO] = None
        try:
            self.write_status(stage="final_eval")
            self.event("final_eval_start", checkpoint_kind=kind)
            with self.tracker.segment("final_eval_total") as seg:
                final_json, arrays, reloaded = evaluate_checkpoint(
                    str(self.out / self.endpoint["files"]["pt"]), self.cfg, kind, self.tracker)
                final_json.update(self.ident)
                final_json["checkpoint_global_update"] = self.endpoint["identity"]["global_update"]
                final_json["search_outcome"] = summary["search_outcome"]
                final_json["has_candidate"] = summary["has_candidate"]
                final_json["candidate_update"] = summary["candidate_update"]
                final_json["reload_identity"] = self.reload_identity(reloaded)
                final_json["dev_recompute_vs_call"] = self.dev_recompute_diff(final_json)
            final_json["resources"] = dict(seg)
            self.resource(seg)
            save_npz_atomic(self.path("arrays.npz"), arrays)
            write_json_atomic(self.path("final_eval.json"), final_json)
            final_ok = True
            self.event("final_eval_end", certification=final_json["certification"],
                       final_joint_pass=final_json["final_joint_pass"])
            self.write_status(certification=final_json["certification"],
                              final_joint_pass=final_json["final_joint_pass"])
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, RunInterrupted)):
                return self.fail(exc, "final_eval")
            errors.append(f"final_eval: {type(exc).__name__}: {exc}")
            self.write_traceback("final_eval")
            err = {**self.ident, "checkpoint_kind": kind, "error": errors[-1],
                   "certification": "error" if kind == "candidate" else "not_applicable_no_candidate",
                   "final_joint_pass": False, "has_candidate": summary["has_candidate"],
                   "search_outcome": summary["search_outcome"], "partial": final_json or None}
            write_json_atomic(self.path("final_eval.json"), err)
            self.event("error", where="final_eval", error=errors[-1])
            self.write_status(certification=err["certification"], final_joint_pass=False)
        try:
            self.min_dev_profiles()
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, RunInterrupted)):
                return self.fail(exc, "min_dev_profiles")
            errors.append(f"min_dev_profiles: {type(exc).__name__}: {exc}")
            self.write_traceback("min_dev_profiles")
            self.event("error", where="min_dev_profiles", error=errors[-1])
        econ_cfg = self.cfg["economics"]
        if econ_cfg.get("enabled", True) is False:
            write_json_atomic(self.path("economics.json"), {**self.ident, "skipped": True,
                                                            "reason": econ_cfg.get("disabled_reason"),
                                                            "checkpoint_kind": kind})
            self.event("economics_skipped", reason=econ_cfg.get("disabled_reason"))
        try:
            if econ_cfg.get("enabled", True) is False:
                raise _SkipEconomics()
            self.write_status(stage="economics")
            self.event("economics_start")
            if reloaded is None:
                reloaded = fresh_agent(self.cfg)
                reloaded.load_weights(str(self.out / self.endpoint["files"]["pt"]))
            vm = None
            if final_ok and final_json["tiers"]["final"].get("valid") is not None:
                vm = final_json["tiers"]["final"].get("v_mean_root")
            with self.tracker.segment("economics_total") as seg:
                econ, econ_arrays = evaluate_economics(reloaded, self.spec, self.ident["seed"],
                                                       self.cfg["economics"], float(self.cfg["es_bin_width"]),
                                                       vm, self.tracker)
            econ.update(self.ident)
            econ["checkpoint_kind"] = kind
            econ["checkpoint_global_update"] = self.endpoint["identity"]["global_update"]
            econ["resources"] = dict(seg)
            self.resource(seg)
            save_npz_atomic(self.path("economics_arrays.npz"), econ_arrays)
            write_json_atomic(self.path("economics.json"), econ)
            self.event("economics_end")
        except _SkipEconomics:
            pass
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, RunInterrupted)):
                return self.fail(exc, "economics")
            errors.append(f"economics: {type(exc).__name__}: {exc}")
            self.write_traceback("economics")
            write_json_atomic(self.path("economics.json"), {**self.ident, "error": errors[-1],
                                                            "checkpoint_kind": kind})
            self.event("error", where="economics", error=errors[-1])
        peak = self.tracker.current_process_peak()
        state = "done" if not errors else "failed"
        self.event("run_end", state=state, errors=errors, process_peak_rss_bytes=peak)
        self.write_status(state=state, stage="finished", end_time=now_iso(),
                          failure_reason="; ".join(errors) or None, process_peak_rss_bytes=peak,
                          available_checkpoints=self.available_checkpoints())
        return 0 if not errors else 1

    def reload_identity(self, reloaded: CurriculumPPO) -> Dict[str, Any]:
        """Compare the reloaded endpoint with the in-memory actor on dev and dense grids."""
        _, b_mem = self.mean_fn, self.beta_fn
        _, b_rel = make_policy_fns(reloaded, self.spec)
        worst = 0.0
        for t in range(1, self.spec.T + 1):
            for step in (self.dev_cfg.state_step, 0.05):
                d = stage_grid(t, self.spec.B, step)
                a1, c1 = b_mem(t, d)
                a2, c2 = b_rel(t, d)
                worst = max(worst, float(np.max(np.abs(a1 - a2))), float(np.max(np.abs(c1 - c2))))
        same_params = all(torch.equal(v, reloaded.actor.state_dict()[k])
                          for k, v in self.agent.actor.state_dict().items())
        return {"alpha_beta_max_abs_diff": worst, "actor_state_dict_equal": bool(same_params),
                "identical": bool(worst == 0.0 and same_params)}

    def dev_recompute_diff(self, final_json: Dict[str, Any]) -> Dict[str, Any]:
        """Original development call at the endpoint update vs the recomputed dev tier."""
        g = self.endpoint["identity"]["global_update"]
        calls = [r for r in self.dev_records if r["global_update"] == g]
        if not calls or calls[-1]["summary"] is None or final_json["tiers"]["development"].get("valid") is None:
            return {"available": False, "reason": "no development call at the endpoint update"}
        a, b = calls[-1]["summary"], final_json["tiers"]["development"]
        out = {"available": True, "call_index": calls[-1]["call_index"]}
        for key in ("dreach_over_dw", "exp_root_over_dw", "delta_max_all_over_dw", "dfull_over_dw"):
            if a.get(key) is None or b.get(key) is None:
                out[key] = None
            else:
                out[key] = abs(float(a[key]) - float(b[key]))
        return out

    def min_dev_profiles(self) -> None:
        """Dense profiles from the saved minimum-development weights, plus a dev replay."""
        rec_path = self.out / "checkpoints" / "min_dev_record.json"
        if not rec_path.exists():
            return
        meta = json.loads(rec_path.read_text())
        with self.tracker.segment("min_dev_profiles") as seg:
            agent = fresh_agent(self.cfg)
            ck = agent.load_weights(str(self.out / meta["files"]["pt"]))
            mean_fn, beta_fn = make_policy_fns(agent, self.spec)
            prof = policy_profiles(mean_fn, beta_fn, self.spec, self.cfg["final_certification"]["dense_conc_step"])
            res, err = run_verifier(mean_fn, beta_fn, self.spec, self.dev_cfg)
        self.resource(seg)
        replay = None if res is None else float(res.dreach / self.spec.dw)
        arrs = {f"dense_{k}": v for k, v in prof["arrays"].items()}
        arrs["identity_json"] = np.array(json.dumps(ck.get("identity")))
        save_npz_atomic(self.out / "checkpoints" / "min_dev_profiles.npz", arrs)
        write_json_atomic(self.out / "checkpoints" / "min_dev_profiles.json", {
            "identity": ck.get("identity"), "dense_concentration": prof["concentration"],
            "dev_replay_dreach_over_dw": replay, "dev_replay_error": err,
            "recorded_dreach_over_dw": meta["dreach_over_dw"],
            "dev_replay_abs_diff": None if replay is None else abs(replay - meta["dreach_over_dw"]),
            "resources": dict(seg)})

    def write_traceback(self, where: str) -> None:
        with open(self.path("traceback.txt"), "a") as f:
            f.write(f"=== {now_iso()} ({where}) ===\n{traceback.format_exc()}\n")

    def fail(self, exc: BaseException, where: str) -> int:
        interrupted = isinstance(exc, (KeyboardInterrupt, RunInterrupted))
        try:
            self.write_traceback(where)
            if hasattr(self, "coverage"):
                self.save_coverage()
            self.event("error", where=where, error=f"{type(exc).__name__}: {exc}")
        finally:
            self.write_status(state="interrupted" if interrupted else "failed", stage=where,
                              end_time=now_iso(), failure_reason=f"{where}: {type(exc).__name__}: {exc}",
                              phases=self.phase_rows, totals=self.totals,
                              available_checkpoints=self.available_checkpoints())
        print(traceback.format_exc(), flush=True)
        return 130 if interrupted else 1


def _sigterm(signum: int, frame: Any) -> None:
    raise RunInterrupted(f"signal {signum}")


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--seed", required=True, type=int)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, _sigterm)
    run = Run(Path(args.manifest), args.seed)
    if run.out.exists():
        print(f"refusing to overwrite existing run directory {run.out}", file=sys.stderr)
        return 3
    print(f"[run] {run.ident['run_id']} host={socket.gethostname()} out={run.out}", flush=True)
    rc = run.execute()
    print(f"[end] {run.ident['run_id']} state={run.status.get('state')} "
          f"outcome={run.status.get('search_outcome')} certification={run.status.get('certification')} "
          f"updates={run.updates_completed}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
