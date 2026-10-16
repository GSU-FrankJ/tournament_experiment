#!/usr/bin/env python3
"""Evaluation of the runs of round T3-R1 (D6): last iterate of every run, on the final and the development
verifier tier, from the run's own files.

Three run formats are read (the policy is always the last iterate, as Beta means):
  * ``bw``    composite of the frozen stage-3 and stage-2 snapshots and the live actor at stage 1, from
              ``state_end_stage1.pt`` (keys ``frozen_stages`` and ``agent``);
  * ``joint`` (``jt``, ``jt_es``, ``jt2``) the live actor at every stage, from ``state_end.pt``; the best-of-run
              candidate (descriptive) comes from the exports ``weights/u{u:05d}.npz``;
  * ``v20``   the ``u01600`` export at t = 2 and the ``u02200`` export at t = 1 of a v2.0 confirmation run.
Exports are loaded with the loader of ``tools/ms/replay_dev_rule.py`` (``load_export``, ``build_actor``,
``candidate_fns``): a float32 ``BetaActor`` with the forward pass of training, the policy functions from
``run.run_final_dp_br.make_policy_fns``. Every metric is ``utils.v2_metrics.evaluate`` (pure) plus
``tools/t3/t3_metrics.py``. The evaluation consumes no random number and changes no run file.

Outputs in --out-dir: ``per_run.csv`` (one row per run), ``best_of_run.csv`` (joint arms, descriptive) and
``arrays/{run}.npz`` (per-stage final-tier arrays for the figures).

Usage:
    python tools/t3/t3_eval.py --root results/t3_r1 --waves t3 t2j --v20-root <D9 root> \
        --out-dir results/t3_r1/eval --workers 40
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import t3_configs as tc  # noqa: E402
import t3_metrics as tm  # noqa: E402
from agents.ppo_curriculum import BetaActor  # noqa: E402
from envs.curriculum_env import GameSpec  # noqa: E402
from replay_dev_rule import build_actor, candidate_fns, load_export  # noqa: E402
from utils.dp_br_verifier import DEV_CONFIG, FINAL_CONFIG, DomainError, VerifierConfig  # noqa: E402
from utils.v2_metrics import V2Eval, evaluate, save_npz  # noqa: E402

TIERS = {"final": FINAL_CONFIG, "development": DEV_CONFIG}
V20_EXPORTS = {2: "u01600", 1: "u02200"}
LABEL_BEST = "selected with the development verifier; not a pre-specified candidate"
ID_COLS = ["wave", "arm", "q", "seed", "T", "run", "run_dir", "kind", "status", "exit_code", "global_rng_ok",
           "eval_error"]


@dataclass
class Candidate:
    """A composite candidate policy: one network per stage."""

    spec: GameSpec
    nets: Dict[int, BetaActor]
    recovery_step: float
    label: str = "last_iterate"


# ---------------------------------------------------------------------------------- loading
def read_json(path: Path) -> Dict[str, Any]:
    """A JSON file as a dict."""
    with open(path) as f:
        return json.load(f)


def spec_of(cfg: Mapping[str, Any]) -> GameSpec:
    """``GameSpec`` of the record embedded in a run config."""
    return GameSpec(**cfg["record"]["game"])


def net_from_state_dict(sd: Mapping[str, torch.Tensor], cfg: Mapping[str, Any]) -> BetaActor:
    """A ``BetaActor`` carrying exactly the float32 weights of an actor ``state_dict``."""
    ppo = cfg["record"]["ppo"]
    arrays = {f"actor.{k}": v.detach().cpu().numpy() for k, v in sd.items()}
    return build_actor(arrays, hidden=int(ppo["hidden"]), c_min=float(ppo["c_min"]), mu_clamp=float(ppo["mu_clamp"]))


def net_from_export(path: Path, cfg: Mapping[str, Any]) -> BetaActor:
    """A ``BetaActor`` from one ``weights/u*.npz`` export."""
    ppo = cfg["record"]["ppo"]
    return build_actor(load_export(str(path)), hidden=int(ppo["hidden"]), c_min=float(ppo["c_min"]),
                       mu_clamp=float(ppo["mu_clamp"]))


def load_bw(run_dir: Path, cfg: Mapping[str, Any]) -> Candidate:
    """Composite of ``bw``: frozen stage 3, frozen stage 2, live actor at stage 1 (``state_end_stage1.pt``)."""
    st = torch.load(run_dir / "state_end_stage1.pt", map_location="cpu", weights_only=False)
    nets = {int(s): net_from_state_dict(sd, cfg) for s, sd in st["frozen_stages"].items()}
    nets[1] = net_from_state_dict(st["agent"]["actor"], cfg)
    spec = spec_of(cfg)
    if sorted(nets) != list(range(1, spec.T + 1)):
        raise ValueError(f"{run_dir}: bw state holds the stages {sorted(nets)}")
    return Candidate(spec, nets, float(cfg["record"]["protocol"]["recovery_step"]))


def load_joint(run_dir: Path, cfg: Mapping[str, Any]) -> Candidate:
    """The live actor of a joint run at every stage (``state_end.pt``)."""
    st = torch.load(run_dir / "state_end.pt", map_location="cpu", weights_only=False)
    net = net_from_state_dict(st["agent"]["actor"], cfg)
    spec = spec_of(cfg)
    return Candidate(spec, {t: net for t in range(1, spec.T + 1)}, float(cfg["record"]["protocol"]["recovery_step"]))


def load_joint_export(run_dir: Path, cfg: Mapping[str, Any], u: int) -> Candidate:
    """A joint run's candidate at update ``u`` from ``weights/u{u:05d}.npz`` (one network for all stages)."""
    net = net_from_export(run_dir / "weights" / f"u{u:05d}.npz", cfg)
    spec = spec_of(cfg)
    return Candidate(spec, {t: net for t in range(1, spec.T + 1)}, float(cfg["record"]["protocol"]["recovery_step"]),
                     label=f"export_u{u:05d}")


def load_v20(run_dir: Path, cfg: Mapping[str, Any]) -> Candidate:
    """v2.0's evaluated candidate: stage 2 from ``u01600``, stage 1 from ``u02200``."""
    nets = {t: net_from_export(run_dir / "weights" / f"{name}.npz", cfg) for t, name in V20_EXPORTS.items()}
    return Candidate(spec_of(cfg), nets, float(cfg["record"]["protocol"]["recovery_step"]))


def load_candidate(kind: str, run_dir: Path, cfg: Mapping[str, Any]) -> Candidate:
    """The last-iterate candidate of a run of the given format (``bw``, ``joint``, ``v20``)."""
    return {"bw": load_bw, "joint": load_joint, "v20": load_v20}[kind](run_dir, cfg)


# ---------------------------------------------------------------------------------- evaluation
def evaluate_tier(cand: Candidate, tier: VerifierConfig) -> V2Eval:
    """``utils.v2_metrics.evaluate`` of the candidate on one tier (pure)."""
    mean_fn, beta_fn = candidate_fns(cand.spec, cand.nets)
    return evaluate(mean_fn, cand.spec, tier, beta_fn=beta_fn, recovery_step=cand.recovery_step)


def evaluate_candidate(cand: Candidate, criteria: Mapping[str, Any]
                       ) -> Tuple[Dict[str, V2Eval], Dict[str, Any]]:
    """Both tiers and the per-run metric row body of one candidate.

    Returns:
        ``({tier: V2Eval}, row)`` with the final-tier columns unprefixed and the development-tier columns
        prefixed ``dev_`` (:func:`t3_metrics.run_columns`).
    """
    mean_fn, _ = candidate_fns(cand.spec, cand.nets)
    evs = {name: evaluate_tier(cand, cfg) for name, cfg in TIERS.items()}
    tm_ = {name: tm.tier_metrics(ev, mean_fn, cand.spec, cand.recovery_step) for name, ev in evs.items()}
    return evs, tm.run_columns(tm_["final"], tm_["development"], int(cand.spec.T), criteria)


def arrays_for_figures(evs: Mapping[str, V2Eval], cand: Candidate) -> Dict[str, np.ndarray]:
    """Final-tier per-stage arrays (gap grid, mean effort, one-step gain, dynamic gain) and the
    terminal-stage recovery grid, for the figures."""
    ev = evs["final"]
    out: Dict[str, np.ndarray] = {}
    for t in range(1, cand.spec.T + 1):
        for name, key in (("d_grid", "d_grid"), ("e_hat", "e_hat"), ("delta", "delta"), ("G", "G")):
            out[f"fin_t{t}_{name}"] = np.asarray(ev.arrays[f"v_t{t}_{key}"])
    mean_fn, _ = candidate_fns(cand.spec, cand.nets)
    _, rarr = tm.terminal_recovery(mean_fn, cand.spec, cand.recovery_step)
    out.update(rarr)
    return out


def status_of(run_dir: Path) -> Dict[str, Any]:
    """``status.json`` of a run (``{}`` if absent)."""
    p = run_dir / "status.json"
    return read_json(p) if p.exists() else {}


def global_rng_ok(run_dir: Path, kind: str) -> Optional[bool]:
    """The run's own global-RNG verdict (``gates.json`` for ``bw`` and ``v20``, the summary for joint runs)."""
    p = run_dir / ("joint_run_summary.json" if kind == "joint" else "gates.json")
    if not p.exists():
        return None
    return (read_json(p).get("global_rng") or {}).get("status") == "ok"


def evaluate_run(job: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate one run (a dict with ``kind``, ``wave``, ``arm``, ``q``, ``seed``, ``run_dir``, ``arrays_dir``).

    A run whose status is not ``done`` with exit code 0 (or whose files are missing) is a row with NaN metrics
    and its status: it counts as failed downstream.
    """
    torch.set_num_threads(1)
    criteria, _ = tm.load_criteria()
    run_dir = Path(job["run_dir"])
    st = status_of(run_dir)
    row: Dict[str, Any] = {
        "wave": job["wave"], "arm": job["arm"], "q": int(job["q"]), "seed": int(job["seed"]),
        "T": int(job["T"]), "run": job["run"], "run_dir": str(run_dir), "kind": job["kind"],
        "status": st.get("state", "missing"), "exit_code": st.get("exit_code"),
        "global_rng_ok": global_rng_ok(run_dir, job["kind"]), "eval_error": ""}
    if st.get("state") != "done" or st.get("exit_code") != 0:
        return row
    try:
        cfg = read_json(run_dir / "run_config.json")
        cand = load_candidate(job["kind"], run_dir, cfg)
        evs, body = evaluate_candidate(cand, criteria)
        row.update(body)
        if job.get("arrays_dir"):
            os.makedirs(job["arrays_dir"], exist_ok=True)
            np.savez(os.path.join(job["arrays_dir"], f"{job['run']}.npz"), **arrays_for_figures(evs, cand))
    except (DomainError, FloatingPointError, ValueError) as exc:
        row["eval_error"] = f"{type(exc).__name__}: {exc}"
    return row


# ---------------------------------------------------------------------------------- best of run
def read_checks(run_dir: Path) -> List[Dict[str, Any]]:
    """The development checks of a joint run (``joint_checks.csv``) with the fields best-of-run needs."""
    out = []
    with open(run_dir / "joint_checks.csv", newline="") as f:
        for r in csv.DictReader(f):
            ok = r["valid"] == "True" and r["error"] == ""
            out.append({"update": int(r["update"]), "valid": ok,
                        "gmax_dev": float(r["Gmax_full_over_dw"]) if ok else float("nan")})
    return out


def best_candidates(run_dir: Path) -> List[Tuple[float, int, str]]:
    """Candidates of the best-of-run selection: the valid development checks plus the last iterate, as
    ``(development Gmax_full_over_dw, update, source)``; a check whose evaluation is invalid or raised an error
    is excluded; the last iterate's development value is the one of ``final_eval_development.npz``
    (``joint_run_summary.json``)."""
    summ = read_json(run_dir / "joint_run_summary.json")
    cands = [(c["gmax_dev"], c["update"], "check") for c in read_checks(run_dir) if c["valid"]]
    fe = summ["final_eval"]["development"]
    U = int(summ["updates"])
    if bool(fe["valid"]) and U not in {u for _, u, _ in cands}:
        cands.append((float(fe["Gmax_full_over_dw"]), U, "last_iterate"))
    return cands


def select_best(cands: Sequence[Tuple[float, int, str]]) -> Optional[Tuple[float, int, str]]:
    """Smallest development ``Gmax_full_over_dw``; ties go to the earliest update."""
    return min(cands, key=lambda c: (c[0], c[1])) if cands else None


def evaluate_best(job: Mapping[str, Any]) -> Dict[str, Any]:
    """Best-of-run row of one joint run (descriptive; selected with the development verifier)."""
    torch.set_num_threads(1)
    criteria, _ = tm.load_criteria()
    run_dir = Path(job["run_dir"])
    row: Dict[str, Any] = {"wave": job["wave"], "arm": job["arm"], "q": int(job["q"]), "seed": int(job["seed"]),
                           "T": int(job["T"]), "run": job["run"], "label": LABEL_BEST, "eval_error": ""}
    st = status_of(run_dir)
    if st.get("state") != "done" or st.get("exit_code") != 0:
        row["status"] = st.get("state", "missing")
        return row
    row["status"] = "done"
    cfg = read_json(run_dir / "run_config.json")
    cands = best_candidates(run_dir)
    best = select_best(cands)
    row["n_candidates"] = len(cands)
    if best is None:
        row["eval_error"] = "no valid candidate"
        return row
    _, u, src = best
    try:
        cand = load_joint(run_dir, cfg) if src == "last_iterate" else load_joint_export(run_dir, cfg, u)
        _, body = evaluate_candidate(cand, criteria)
        row.update(body)
    except (DomainError, FloatingPointError, ValueError) as exc:
        row["eval_error"] = f"{type(exc).__name__}: {exc}"
    row.update({"best_update": u, "best_source": src, "best_dev_gmax_selection": best[0]})
    return row


# ---------------------------------------------------------------------------------- C-REPLAY helpers
REPLAY_FILES = {"bw": {"final": "freeze_stage1_final.npz", "development": "freeze_stage1_development.npz"},
                "joint": {"final": "final_eval_final.npz", "development": "final_eval_development.npz"}}


def replay_mismatches(run_dir: Path, kind: str) -> List[str]:
    """Keys in which a re-evaluation of the run's last iterate differs from the run's own final-evaluation
    arrays (bit for bit); empty list: identical. ``kind`` is ``bw`` or ``joint``."""
    cfg = read_json(run_dir / "run_config.json")
    cand = load_candidate(kind, run_dir, cfg)
    bad: List[str] = []
    for name, tier in TIERS.items():
        ev = evaluate_tier(cand, tier)
        with np.load(run_dir / REPLAY_FILES[kind][name]) as z:
            if sorted(z.files) != sorted(ev.arrays):
                bad.append(f"{name}: keys differ")
                continue
            for k in z.files:
                a, b = np.asarray(ev.arrays[k]), z[k]
                if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=True):
                    bad.append(f"{name}:{k}")
    return bad


# ---------------------------------------------------------------------------------- driver
def plan_jobs(root: Path, waves: Sequence[str], v20_root: Optional[Path], arrays_dir: Optional[Path]
              ) -> List[Dict[str, Any]]:
    """The run directories to evaluate: the planned runs of the waves (``t3``, ``t2j``, ``smoke``) and, with
    ``v20_root``, the 40 v2.0 confirmation runs."""
    jobs: List[Dict[str, Any]] = []
    for wave in waves:
        for q in tc.QS:
            for seed in tc.WAVE_SEEDS[wave]:
                for arm in tc.WAVE_ARMS[wave]:
                    d = tc.run_dir(root, wave, q, seed, arm)
                    jobs.append({"wave": wave, "arm": arm, "q": q, "seed": seed, "T": tc.T_OF_ARM[arm],
                                 "run": tc.run_name(q, seed, arm), "run_dir": str(d),
                                 "kind": "bw" if arm == "bw" else "joint", "arrays_dir": str(arrays_dir) if arrays_dir else None})
    if v20_root is not None:
        for q in tc.QS:
            for seed in tc.WAVE_SEEDS["t2j"]:
                jobs.append({"wave": "v20", "arm": "v20", "q": q, "seed": seed, "T": 2,
                             "run": f"v2T2locked_q{q}_s{seed}", "run_dir": str(Path(v20_root) / f"q{q}" / f"seed{seed}"),
                             "kind": "v20", "arrays_dir": str(arrays_dir) if arrays_dir else None})
    return jobs


def run_parallel(fn: Any, jobs: Sequence[Mapping[str, Any]], workers: int) -> List[Dict[str, Any]]:
    """``fn`` over the jobs in a pool of single-threaded spawned processes (input order kept)."""
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=ctx) as pool:
        return list(pool.map(fn, jobs))


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write dict rows to a CSV with the union of the keys (identifier columns first)."""
    keys: List[str] = [c for c in ID_COLS if any(c in r for r in rows)]
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R1 evaluation of the runs", allow_abbrev=False)
    ap.add_argument("--root", default=str(tc.DEFAULT_ROOT))
    ap.add_argument("--waves", nargs="+", default=["t3", "t2j"], choices=list(tc.WAVES))
    ap.add_argument("--v20-root", default=None, help="the v2.0 confirmation root (D9); omit to skip the v20 runs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8, help="at most 40")
    ap.add_argument("--no-best", action="store_true", help="skip the best-of-run evaluation of the joint arms")
    a = ap.parse_args(argv)
    if not 1 <= a.workers <= 40:
        ap.error("--workers must be in 1..40")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = plan_jobs(Path(a.root), a.waves, Path(a.v20_root) if a.v20_root else None, out / "arrays")
    rows = run_parallel(evaluate_run, jobs, a.workers)
    write_rows(out / "per_run.csv", rows)
    print(f"{len(rows)} runs -> {out / 'per_run.csv'}", flush=True)
    if not a.no_best:
        bj = [j for j in jobs if j["kind"] == "joint"]
        brows = run_parallel(evaluate_best, bj, a.workers)
        write_rows(out / "best_of_run.csv", brows)
        print(f"{len(brows)} best-of-run rows -> {out / 'best_of_run.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
