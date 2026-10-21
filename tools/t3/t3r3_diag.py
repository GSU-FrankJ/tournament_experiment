#!/usr/bin/env python3
"""Diagnosis tool of round T3-R3 (prompt D3): DG1-DG4, DG6-DG8 and the retrospective trigger table of D4.

Read-only: it consumes no random number, trains nothing and changes no file outside ``--out-dir``. It reads the 80 recorded ``bw``
runs (T3-R1 seeds 50501-50520, T3-R2 confirmation seeds 60501-60520; tracked logs in the branch, untracked weights and train
histories in the two reference roots, both recorded in the outputs) and the 40 v2.0 confirmation runs. The closed-form tent enters
the reporting columns only (``e*(0)``, ``rmse_pos_T_over_eT0`` ... are read from the stored evaluation tables); none of it is used
by a quantity that could trigger anything (``F(u)``, ``S3``, ``R``, ``R_tail``, ``Delta``, the entry stretch are closed-form-free).

Subcommands:
  dg1   records inventory of the 80 + 40 runs
  dg2   the stage-3 policy at every one of the 64 stage-3 exports of the 80 ``bw`` runs, on the final-tier grid of D_3
  dg3   training signal, coverage, development checks and gradient-norm series (stage-3 phase)
  dg4   deviation-gain curves of the DG5 set at ``u00100, u00400, u00800, u01600`` (independent calculator)
  dg5   the summary of the instrumented replays (reads ``results/t3_r3/replays``; produced by ``tools/t3/t3r3_replay.py``)
  dg6   the clamp regime S3 of the 80 ``bw`` and the T = 2 analogue of the 40 v2.0 runs, its relations and its timing
  dg8   the stage-3 network ``u01600`` at the stage-2 input (phase entry) and the stage-2 stretches of all 80 runs
  retro the retrospective trigger table of D4 (``ms_checks_stage3.csv`` at local update 1600)
  dg7   the per-failure evidence table (reads the outputs of dg2, dg3, dg6, dg8 and retro)

Usage:
    python tools/t3/t3r3_diag.py <step> [--t3r1-root <results/t3_r1>] [--t3r2-root <results/t3_r2>] [--v20-root <dir>] \
        [--out-dir results/t3_r3/diag] [--workers 8]
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import csv
import json
import math
import multiprocessing as mp
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import t3_eval as te  # noqa: E402
import t3_metrics as tm  # noqa: E402
import t3r2_audit as au  # noqa: E402
import t3r2_br_calc as bc  # noqa: E402
from t3c_blind import run_passes  # noqa: E402

DEFAULT_R1 = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-vs-jt-51e472/results/t3_r1")
DEFAULT_R2 = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-audit-6f3a38/results/t3_r2")
DEFAULT_V20 = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/confirmation_v2_0")
DEFAULT_OUT = ROOT / "results" / "t3_r3" / "diag"
TRACKED_V20 = ROOT / "results" / "v2_T2_locked" / "confirmation_v2_0"
PER_RUN = {"t3_r1": ROOT / "results" / "t3_r1" / "eval" / "per_run.csv",
           "t3_r2": ROOT / "results" / "t3_r2" / "confirmation" / "eval" / "per_run.csv"}
QS = (50, 60)
SEEDS = {"t3_r1": tuple(range(50501, 50521)), "t3_r2": tuple(range(60501, 60521))}
EXPORTS3 = tuple(range(25, 1601, 25))          # the 64 stage-3 exports
DG4_EXPORTS = (100, 400, 800, 1600)
DG4_D = (0.0, 4.0, -4.0, 6.0, -6.0, 10.0, -10.0, 20.0, -20.0, 36.0, -36.0, 40.0, -40.0, 80.0, -80.0)
SAT_THR = 3.0                                   # |pre-activation| above which a tanh unit counts as saturated
ONSET_SHARE = 0.2                               # out-of-band clamp share that starts the clamp regime window
REGIME_SHARE = 0.2                              # S3 above this: the run is in the stage-3 clamp regime
HIGH_SHARE = 0.4
LOW_SHARE = 0.05
BAND_TOL = 1e-9
MU_CLAMP = 1e-6                                 # ``BetaActor`` mean clamp: mu = clamp(sigmoid(z0), 1e-6, 1 - 1e-6)
WINDOWS3 = ((1, 100), (101, 400), (401, 800), (801, 1200), (1201, 1600))
CHECK_LOCALS = (25, 50, 100, 200, 400, 800, 1200, 1600)
# the ten failures of the 80 bw runs (prompt table): (round, q, seed)
FAILURES = (("t3_r1", 50, 50504), ("t3_r1", 50, 50517), ("t3_r1", 60, 50514), ("t3_r1", 60, 50517),
            ("t3_r2", 50, 60501), ("t3_r2", 50, 60506), ("t3_r2", 50, 60513), ("t3_r2", 50, 60515),
            ("t3_r2", 60, 60516), ("t3_r2", 60, 60518))
# the DG5 set (T3-R2 confirmation runs)
REPLAY_SET = ((50, 60501), (50, 60506), (50, 60513), (50, 60515), (60, 60516), (60, 60518),
              (50, 60505), (50, 60507), (60, 60503), (60, 60509), (50, 60502), (50, 60504), (60, 60501), (60, 60502))
TRIGGER_THRESHOLDS = {"Delta": 0.005, "R": (0.4, 0.5, 0.7, 1.0), "R_tail": (0.02, 0.05, 0.1)}


# ---------------------------------------------------------------------------------- small helpers
def read_csv(path: Path) -> List[Dict[str, str]]:
    """Rows of a CSV as dicts of strings."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x: Any) -> float:
    """CSV cell as float (NaN if empty or not a number)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Dict rows to a CSV (union of the keys, first-seen order)."""
    au.write_csv_rows(path, rows)


def write_json(path: Path, obj: Any) -> None:
    """JSON file with indent 1."""
    au.write_json(path, obj)


def parallel(fn: Callable[[Any], Any], items: Sequence[Any], workers: int) -> List[Any]:
    """``fn`` over ``items`` in spawned single-threaded processes (input order kept)."""
    return au.parallel(fn, items, workers)


def git_head() -> Dict[str, Any]:
    """HEAD and the porcelain status of the repository (recorded with every output)."""
    return {"head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "status_porcelain": subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True,
                                               text=True).stdout.splitlines()}


def bw_runs(r1_root: Path, r2_root: Path) -> List[Dict[str, Any]]:
    """The 80 recorded ``bw`` runs: ``{round, q, seed, run, dir, tracked, row}`` (``row`` = the stored ``per_run.csv`` row)."""
    roots = {"t3_r1": (Path(r1_root) / "runs", "runs"), "t3_r2": (Path(r2_root) / "confirmation", "confirmation")}
    out = []
    for rnd in ("t3_r1", "t3_r2"):
        table = {(int(float(r["q"])), int(float(r["seed"]))): r for r in read_csv(PER_RUN[rnd]) if r["arm"] == "bw"}
        base, sub = roots[rnd]
        for q in QS:
            for s in SEEDS[rnd]:
                row = table[(q, s)]
                out.append({"round": rnd, "q": q, "seed": s, "run": row["run"], "dir": base / f"q{q}" / f"seed{s}" / "bw",
                            "tracked": ROOT / "results" / rnd / sub / f"q{q}" / f"seed{s}" / "bw", "row": row})
    return out


def run_name(r: Mapping[str, Any]) -> str:
    """Short run label ``r1_q50_s50504``."""
    return f"{r['round'][-2:]}_q{r['q']}_s{r['seed']}"


def gates() -> Dict[str, Any]:
    """The gate thresholds of ``reports/t3/r1/prereg_criteria.json``."""
    return tm.load_criteria()[0]["gates"]


def passes(r: Mapping[str, Any], g: Optional[Mapping[str, Any]] = None) -> bool:
    """Run pass of the stored ``per_run.csv`` row (the rule of ``t3c_blind.run_passes``)."""
    return bool(run_passes(r["row"], g or gates()))


# ---------------------------------------------------------------------------------- S3 (clamp share)
def clamp_share(rows: Sequence[Mapping[str, str]], lo_col: str, n_col: str, first: int, last: int,
                keep: Optional[Callable[[Mapping[str, str]], bool]] = None) -> Tuple[float, int, int]:
    """``(sum lo / sum n, sum lo, sum n)`` of two counter columns over the rows with local update in ``[first, last]``.

    Args:
        rows: Rows of ``ms_updates.csv`` / ``v2_updates.csv`` (strings).
        lo_col: Column of the draws below the action clamp.
        n_col: Column of the draws.
        first: First local update.
        last: Last local update.
        keep: Extra row filter (the stage of ``ms_updates.csv``, the phase of ``v2_updates.csv``).
    """
    lo = n = 0
    for r in rows:
        if keep is not None and not keep(r):
            continue
        if first <= int(r["local"]) <= last:
            lo += int(r[lo_col])
            n += int(r[n_col])
    return (lo / n if n else float("nan")), lo, n


def s3_of_rows(rows: Sequence[Mapping[str, str]]) -> Tuple[float, int, int]:
    """S3 of a run: out-of-band stage-3 learner draws below the clamp, local updates 1201-1600 of the stage-3 phase."""
    return clamp_share(rows, "d1_L_s3_out_lo", "d1_L_s3_out_n", 1201, 1600, lambda r: int(r["stage"]) == 3)


def s2_t2_of_rows(rows: Sequence[Mapping[str, str]]) -> Tuple[float, int, int]:
    """The T = 2 analogue: stage-2 learner draws at out-of-band starts, phase A local updates 1201-1600 of a v2.0 run."""
    return clamp_share(rows, "d1_L_s2_out_lo", "d1_L_s2_out_n", 1201, 1600, lambda r: r["phase"] == "A")


def onset_window(rows: Sequence[Mapping[str, str]], window: int = 25, share: float = ONSET_SHARE) -> Optional[int]:
    """First local update ``last`` of a ``window``-update window (local ``window*(w-1)+1 .. window*w``) of the stage-3
    phase whose out-of-band clamp share exceeds ``share`` (``None`` if there is none)."""
    for w in range(1, 1600 // window + 1):
        s, _, n = clamp_share(rows, "d1_L_s3_out_lo", "d1_L_s3_out_n", window * (w - 1) + 1, window * w,
                              lambda r: int(r["stage"]) == 3)
        if n and s > share:
            return window * w
    return None


# ---------------------------------------------------------------------------------- the stage-3 network on a grid (DG2)
def forward_parts(net: Any, x: Any) -> Dict[str, Any]:
    """Pre-activations and outputs of a ``t1`` ``BetaActor`` with the operations of ``BetaActor.forward`` (float32).

    Returns:
        Dict of tensors: ``pre1, pre2, z, mu, c, alpha, beta`` (``alpha, beta`` equal ``net(x)`` exactly).
    """
    import torch
    import torch.nn.functional as F

    pre1 = net.l1(x)
    h1 = torch.tanh(pre1)
    pre2 = net.l2(h1)
    h2 = torch.tanh(pre2)
    z = net.out(h2)
    mu = torch.clamp(torch.sigmoid(z[:, 0]), net.mu_clamp, 1.0 - net.mu_clamp)
    c = net.c_min + F.softplus(z[:, 1])
    return {"pre1": pre1, "pre2": pre2, "z": z, "mu": mu, "c": c, "alpha": mu * c, "beta": (1.0 - mu) * c}


def critic_from_arrays(arrays: Mapping[str, np.ndarray], hidden: int = 64) -> Any:
    """The critic of an export (``critic.*`` keys) as a float32 ``Critic`` module."""
    import torch
    from agents.ppo_curriculum import Critic

    gen = torch.Generator()
    gen.manual_seed(0)
    net = Critic(hidden, gen)
    net.load_state_dict({k[len("critic."):]: torch.from_numpy(np.array(v, dtype=np.float32))
                         for k, v in arrays.items() if k.startswith("critic.")})
    net.eval()
    return net


def stage_grid(spec: Any, stage: int, step: float = 2.0) -> np.ndarray:
    """The final-tier symmetric grid of D_stage (state step 2, endpoints and 0 included)."""
    from run.run_final_dp_br import dense_grid

    return dense_grid(spec.domain_half(stage), step)


def network_on_grid(arrays: Mapping[str, np.ndarray], spec: Any, grid: np.ndarray, stage: int, hidden: int = 64,
                    c_min: float = 100.0, mu_clamp: float = 1e-6, with_critic: bool = True) -> Dict[str, np.ndarray]:
    """The network of an export at the nodes ``grid`` of stage ``stage``'s input.

    The input is built exactly as in training (float64 gap -> ``spec.encode_obs`` float32). Returns float64 arrays: ``e_hat`` (mean
    effort formed from the float32 alpha, beta as in ``make_policy_fns``), ``alpha``, ``beta``, ``sd_norm`` (Beta standard
    deviation on [0, 1]), ``sens`` (|d mu / d input 2| by autograd), ``sat1`` / ``sat2`` (per node, the share of the first- /
    second-layer tanh units with |pre-activation| > 3) and, with ``with_critic``, ``V`` (the critic).
    """
    import torch
    from replay_dev_rule import build_actor

    net = build_actor(arrays, hidden=hidden, c_min=c_min, mu_clamp=mu_clamp)
    x = torch.as_tensor(spec.encode_obs(stage, grid)).clone().requires_grad_(True)
    p = forward_parts(net, x)
    sens = torch.autograd.grad(p["mu"].sum(), x)[0][:, 1].abs().detach().numpy().astype(float)
    a = p["alpha"].detach().numpy().astype(float)
    b = p["beta"].detach().numpy().astype(float)
    out = {"e_hat": spec.e_min + spec.e_range * a / (a + b), "alpha": a, "beta": b, "sd_norm": au.beta_sd_norm(a, b),
           "sens": sens,
           "sat1": (p["pre1"].detach().abs() > SAT_THR).float().mean(dim=1).numpy().astype(float),
           "sat2": (p["pre2"].detach().abs() > SAT_THR).float().mean(dim=1).numpy().astype(float)}
    if with_critic:
        crit = critic_from_arrays(arrays, hidden)
        with torch.no_grad():
            out["V"] = crit(x.detach()).numpy().astype(float)
    return out


def export_summary(grid: np.ndarray, net: Mapping[str, np.ndarray], q: float) -> Dict[str, float]:
    """The per-export summaries of DG2 (band ``|d| < 2q``; every quantity is closed-form-free).

    ``slope_neg`` / ``slope_pos``: mean slope of ``e_hat`` on the in-band nodes ``(-2q, 0]`` and ``[0, 2q)``, i.e. the chord
    ``(e(0) - e(d_min)) / (0 - d_min)`` and ``(e(d_max) - e(0)) / d_max`` over the outermost in-band nodes (the mean of the
    derivative). ``F`` = e_hat(0) - mean of e_hat over ``|d| >= 2q``.
    """
    d = np.asarray(grid, dtype=float)
    e = np.asarray(net["e_hat"], dtype=float)
    inb = np.abs(d) < 2.0 * float(q) - BAND_TOL
    out = ~inb
    z = int(np.argmin(np.abs(d)))
    left = np.where(inb & (d <= 0.0))[0]
    right = np.where(inb & (d >= 0.0))[0]
    e0 = float(e[z])
    out_mean = float(e[out].mean())
    mu = np.asarray(net["alpha"], dtype=float) / (np.asarray(net["alpha"], dtype=float) + np.asarray(net["beta"], dtype=float))
    return {
        "e0": e0, "in_min": float(e[inb].min()), "in_max": float(e[inb].max()),
        "in_range": float(e[inb].max() - e[inb].min()),
        "slope_neg": float((e0 - e[left[0]]) / (0.0 - d[left[0]])) if left.size > 1 else float("nan"),
        "slope_pos": float((e[right[-1]] - e0) / d[right[-1]]) if right.size > 1 else float("nan"),
        "out_mean": out_mean, "out_max": float(e[out].max()),
        "out_alpha_lt1": float((np.asarray(net["alpha"])[out] < 1.0).mean()), "F": e0 - out_mean,
        "mu_min": float(mu.min()), "out_mu_at_clamp": float((mu[out] <= MU_CLAMP * (1.0 + 1e-3)).mean()),
        "out_mu_lt_1e4": float((mu[out] < 1e-4).mean()),
        "sens_in_mean": float(np.asarray(net["sens"])[inb].mean()), "sens_in_max": float(np.asarray(net["sens"])[inb].max()),
        "sens_at_0": float(np.asarray(net["sens"])[z]),
        "sat1_in": float(np.asarray(net["sat1"])[inb].mean()), "sat2_in": float(np.asarray(net["sat2"])[inb].mean()),
        "sd_norm_max": float(np.asarray(net["sd_norm"]).max()),
        "alpha_min": float(np.asarray(net["alpha"]).min()),
        "V0": float(np.asarray(net["V"])[z]) if "V" in net else float("nan"),
        "V_in_range": float(np.ptp(np.asarray(net["V"])[inb])) if "V" in net else float("nan"),
    }


def formation_export(F: np.ndarray, updates: np.ndarray, threshold: float) -> Optional[int]:
    """First export (update) at which ``F(u)`` exceeds ``threshold``; ``None`` if it never does."""
    idx = np.where(np.asarray(F) > threshold)[0]
    return int(updates[idx[0]]) if idx.size else None


def formation_threshold(final_F: Sequence[float]) -> float:
    """Half of the median final ``F`` of a set of runs (the passing runs of one q)."""
    return 0.5 * float(np.median(np.asarray(final_F, dtype=float)))


# ---------------------------------------------------------------------------------- DG1
def dg1_one(r: Mapping[str, Any]) -> Dict[str, Any]:
    """Records of one ``bw`` run (existence of the tracked and untracked files, export count, the bin map claim)."""
    d, t = Path(r["dir"]), Path(r["tracked"])
    row: Dict[str, Any] = {"round": r["round"], "q": r["q"], "seed": r["seed"], "run": r["run"], "dir": str(d)}
    for name in ("ms_updates.csv", "ms_checks_stage1.csv", "ms_checks_stage2.csv", "ms_checks_stage3.csv", "ms_binmaps_stage2.npz",
                 "ms_binmaps_stage3.npz", "run_config.json", "gates.json"):
        row[f"tracked_{name}"] = bool((t / name).is_file())
    for name in ("state_end_stage1.pt", "state_end_stage2.pt", "state_end_stage3.pt", "train_history.json",
                 "freeze_stage3_final.npz", "freeze_stage3_development.npz", "freeze_stage2_final.npz",
                 "freeze_stage1_final.npz"):
        row[f"untracked_{name}"] = bool((d / name).is_file())
    row["n_exports"] = len(list((d / "weights").glob("u*.npz")))
    row["n_stage3_exports"] = len([u for u in EXPORTS3 if (d / "weights" / f"u{u:05d}.npz").is_file()])
    bm = t / "ms_binmaps_stage3.npz"
    if bm.is_file():
        z = np.load(bm)
        row["binmaps3_keys"] = ";".join(z.files)
        row["binmaps3_probs_all_nan"] = bool(np.isnan(z["probs"]).all()) if "probs" in z.files else None
        row["binmaps3_n_checks"] = int(z["update"].size)
    return row


def v20_inventory(v20_root: Path) -> List[Dict[str, Any]]:
    """Export count and ``v2_updates.csv`` of the 40 v2.0 confirmation runs."""
    out = []
    for q in QS:
        for s in range(30501, 30521):
            d = Path(v20_root) / f"q{q}" / f"seed{s}"
            out.append({"round": "v2_0", "q": q, "seed": s, "dir": str(d),
                        "n_exports": len(list((d / "weights").glob("u*.npz"))),
                        "untracked_v2_updates.csv": (d / "v2_updates.csv").is_file(),
                        "tracked_v2_updates.csv": (TRACKED_V20 / f"q{q}" / f"seed{s}" / "v2_updates.csv").is_file()})
    return out


# ---------------------------------------------------------------------------------- DG2
def dg2_one(r: Mapping[str, Any]) -> Dict[str, Any]:
    """DG2 of one run: the 64 stage-3 exports on the final-tier grid of D_3 (rows of summaries and the arrays)."""
    import torch

    torch.set_num_threads(1)
    d = Path(r["dir"])
    cfg = te.read_json(d / "run_config.json")
    spec = te.spec_of(cfg)
    ppo = cfg["record"]["ppo"]
    grid = stage_grid(spec, spec.T)
    rows: List[Dict[str, Any]] = []
    mats: Dict[str, List[np.ndarray]] = {k: [] for k in ("e_hat", "alpha", "beta", "sd_norm", "sens", "V")}
    for u in EXPORTS3:
        arrays = {k: v for k, v in np.load(d / "weights" / f"u{u:05d}.npz").items()}
        net = network_on_grid(arrays, spec, grid, spec.T, hidden=int(ppo["hidden"]), c_min=float(ppo["c_min"]),
                              mu_clamp=float(ppo["mu_clamp"]))
        rows.append({"round": r["round"], "q": r["q"], "seed": r["seed"], "update": u, **export_summary(grid, net, spec.q)})
        for k in mats:
            mats[k].append(net[k])
    arrs = {"updates": np.array(EXPORTS3), "grid": grid, **{k: np.stack(v) for k, v in mats.items()}}
    return {"rows": rows, "arrays": arrs, "key": run_name(r)}


def dg2(runs: Sequence[Mapping[str, Any]], out: Path, workers: int) -> None:
    """Run DG2 for the 80 runs; write ``dg2_exports.csv``, ``dg2_runs.csv`` and the (untracked) arrays."""
    t0 = time.perf_counter()
    res = parallel(dg2_one, list(runs), workers)
    adir = out / "arrays"
    adir.mkdir(parents=True, exist_ok=True)
    rows = [row for x in res for row in x["rows"]]
    for x in res:
        np.savez(adir / f"dg2_{x['key']}.npz", **x["arrays"])
    write_rows(out / "dg2_exports.csv", rows)
    g = gates()
    final_F: Dict[int, List[float]] = {50: [], 60: []}
    for r in runs:
        if passes(r, g):
            final_F[r["q"]].append(next(x for x in rows if (x["round"], x["q"], x["seed"], x["update"]) ==
                                        (r["round"], r["q"], r["seed"], 1600))["F"])
    thr = {q: formation_threshold(v) for q, v in final_F.items()}
    summ = []
    for r in runs:
        mine = [x for x in rows if (x["round"], x["q"], x["seed"]) == (r["round"], r["q"], r["seed"])]
        F = np.array([x["F"] for x in mine])
        ups = np.array([x["update"] for x in mine])
        fe = formation_export(F, ups, thr[r["q"]])
        last = mine[-1]
        summ.append({"round": r["round"], "q": r["q"], "seed": r["seed"], "run": r["run"], "pass": passes(r, g),
                     "formation_threshold": thr[r["q"]], "formation_export": fe, "F_final": last["F"], "e0_final": last["e0"],
                     "in_range_final": last["in_range"], "slope_neg_final": last["slope_neg"], "slope_pos_final": last["slope_pos"],
                     "out_mean_final": last["out_mean"], "out_alpha_lt1_final": last["out_alpha_lt1"],
                     "sens_in_mean_final": last["sens_in_mean"], "sat1_in_final": last["sat1_in"],
                     "sat2_in_final": last["sat2_in"], "sd_norm_max_final": last["sd_norm_max"],
                     "mu_min_final": last["mu_min"], "out_mu_at_clamp_final": last["out_mu_at_clamp"],
                     "out_mu_lt_1e4_final": last["out_mu_lt_1e4"],
                     "first_export_tail_mu_at_clamp": next((x["update"] for x in mine if x["out_mu_at_clamp"] > 0.0), None),
                     "F_max": float(F.max()), "F_argmax_update": int(ups[int(np.argmax(F))]),
                     "e0_max": float(max(x["e0"] for x in mine))})
    write_rows(out / "dg2_runs.csv", summ)
    write_json(out / "dg2_record.json", {"step": "dg2", "formation_threshold_by_q": thr,
                                         "formation_threshold_rule": "half of the median F(u01600) of the passing runs of the q (both rounds pooled)",
                                         "sec": time.perf_counter() - t0, "workers": workers, **git_head()})


# ---------------------------------------------------------------------------------- DG3
def window_stats(rows: Sequence[Mapping[str, str]], first: int, last: int) -> Dict[str, float]:
    """Means of the training-signal columns and the sums of the coverage and clamp counters of the stage-3 rows with local
    update in ``[first, last]``."""
    sel = [r for r in rows if int(r["stage"]) == 3 and first <= int(r["local"]) <= last]
    out: Dict[str, float] = {"n_updates": float(len(sel))}
    for c in ("mean_effort", "kl_final_epoch", "clip_frac", "policy_loss", "value_loss", "adv_raw_mean", "adv_raw_std"):
        out[c] = float(np.mean([float(r[c]) for r in sel])) if sel else float("nan")
    for c in ("n_start_tail", "n_start_near", "n_start_mid", "d1_L_s3_in_n", "d1_L_s3_in_lo", "d1_L_s3_in_hi", "d1_L_s3_out_n",
              "d1_L_s3_out_lo", "d1_L_s3_out_hi", "d1_O_s3_lo", "d1_O_s3_n"):
        out[c] = float(sum(int(r[c]) for r in sel))
    out["share_out_lo"] = out["d1_L_s3_out_lo"] / out["d1_L_s3_out_n"] if out["d1_L_s3_out_n"] else float("nan")
    out["share_in_lo"] = out["d1_L_s3_in_lo"] / out["d1_L_s3_in_n"] if out["d1_L_s3_in_n"] else float("nan")
    return out


def history_stats(hist: Sequence[Mapping[str, Any]], first: int, last: int) -> Dict[str, float]:
    """Gradient-norm, entropy and concentration series of the stage-3 phase over local updates ``[first, last]``."""
    sel = [h for h in hist if int(h["stage"]) == 3 and first <= int(h["local"]) <= last]
    out: Dict[str, float] = {}
    for c in ("grad_norm_actor_mean", "grad_norm_critic_mean", "entropy_post_update_effort_scale", "kl_final_epoch"):
        out[c] = float(np.mean([float(h[c]) for h in sel])) if sel else float("nan")
    out["grad_norm_actor_max"] = float(max(float(h["grad_norm_actor_max"]) for h in sel)) if sel else float("nan")
    out["grad_norm_critic_max"] = float(max(float(h["grad_norm_critic_max"]) for h in sel)) if sel else float("nan")
    out["conc_min"] = float(min(float(h["conc_buffer_min_mean_max"][0]) for h in sel)) if sel else float("nan")
    out["conc_mean"] = float(np.mean([float(h["conc_buffer_min_mean_max"][1]) for h in sel])) if sel else float("nan")
    return out


def check_summary(checks: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    """Summary of the 64 stage-3 development checks: the values at selected local updates, the first check with
    ``Delta`` below 0.10 and 0.05, and the number of checks with ``Delta`` in [0.12, 0.15]."""
    by_local = {int(r["local"]): r for r in checks}
    out: Dict[str, Any] = {}
    for loc in CHECK_LOCALS:
        r = by_local.get(loc)
        for c in ("Delta", "s", "R", "R_tail", "C"):
            out[f"{c}_L{loc}"] = fnum(r[c]) if r else float("nan")
    delta = [(int(r["local"]), fnum(r["Delta"])) for r in checks]
    out["first_local_Delta_lt_0.10"] = next((l for l, v in delta if v < 0.10), None)
    out["first_local_Delta_lt_0.05"] = next((l for l, v in delta if v < 0.05), None)
    out["n_checks_Delta_in_0.12_0.15"] = int(sum(1 for _, v in delta if 0.12 <= v <= 0.15))
    out["Delta_min"] = float(min(v for _, v in delta))
    out["Delta_max_after_L400"] = float(max(v for l, v in delta if l > 400))
    return out


def dg3_one(r: Mapping[str, Any]) -> Dict[str, Any]:
    """DG3 of one run: window statistics of ``ms_updates.csv``, the checks, and ``train_history.json`` where present."""
    t, d = Path(r["tracked"]), Path(r["dir"])
    rows = read_csv(t / "ms_updates.csv")
    checks = [c for c in read_csv(t / "ms_checks_stage3.csv")]
    out_w = []
    for a, b in WINDOWS3:
        out_w.append({"round": r["round"], "q": r["q"], "seed": r["seed"], "first": a, "last": b, **window_stats(rows, a, b)})
    hist_rows = []
    hp = d / "train_history.json"
    if hp.is_file():
        hist = json.load(open(hp))["history"]
        for a, b in WINDOWS3:
            hist_rows.append({"round": r["round"], "q": r["q"], "seed": r["seed"], "first": a, "last": b, **history_stats(hist, a, b)})
    return {"windows": out_w, "history": hist_rows,
            "checks": {"round": r["round"], "q": r["q"], "seed": r["seed"], **check_summary(checks)}}


def dg3(runs: Sequence[Mapping[str, Any]], out: Path, workers: int) -> None:
    """Run DG3 for the 80 runs."""
    t0 = time.perf_counter()
    res = parallel(dg3_one, list(runs), workers)
    write_rows(out / "dg3_windows.csv", [w for x in res for w in x["windows"]])
    write_rows(out / "dg3_history_windows.csv", [w for x in res for w in x["history"]])
    write_rows(out / "dg3_checks.csv", [x["checks"] for x in res])
    write_json(out / "dg3_record.json", {"step": "dg3", "sec": time.perf_counter() - t0, "workers": workers, **git_head()})


# ---------------------------------------------------------------------------------- DG4
def stage3_policy(net_mean: Callable[[int, np.ndarray], np.ndarray]) -> Callable[[int, np.ndarray], np.ndarray]:
    """Calculator policy: the export's mean effort at stage 3 (the stages below are never queried by stage-3 values)."""
    def pol(t: int, d: np.ndarray) -> np.ndarray:
        if int(t) != 3:
            raise ValueError("the stage-3 diagnosis queries stage 3 only")
        return net_mean(3, d)
    return pol


def curve_summary(calc: bc.BRCalculator, d: float, e_grid: np.ndarray) -> Dict[str, Any]:
    """Learned effort, best response, slope and local maxima of ``Q_3(d, .)`` at one gap."""
    dw = calc.game.dw
    sv = calc.stage_values(3, [d], brent=False)
    curve = calc.q_curve(3, d, e_grid, "mean")
    lm = au.local_maxima(e_grid, curve)
    q_hat = float(sv["V_mean"][0])
    return {"d": float(d), "e_hat": float(sv["e_hat"][0]), "e_opp": float(sv["e_opp"][0]), "slope_at_e_hat": float(sv["slope"][0]),
            "a_dev": float(sv["a_dev"][0]), "Delta_over_dw": float(sv["Delta"][0]) / dw,
            "n_local_maxima": len(lm), "local_maxima_efforts": ";".join(f"{m[0]:.2f}" for m in lm),
            "Q_at_e_hat": q_hat, "Q_at_grid_max_minus_Q_hat_over_dw": (float(curve.max()) - q_hat) / dw,
            "grid_argmax_effort": float(e_grid[int(np.argmax(curve))])}


def dg4_one(args: Tuple[Dict[str, Any], int]) -> Dict[str, Any]:
    """DG4 of one export: Delta_3(d) over D_3 and the objective at the nodes (and at the stored maximiser)."""
    r, u = args
    import torch

    torch.set_num_threads(1)
    d = Path(r["dir"])
    cfg = te.read_json(d / "run_config.json")
    spec = te.spec_of(cfg)
    from replay_dev_rule import candidate_fns

    net = te.net_from_export(d / "weights" / f"u{u:05d}.npz", cfg)
    mean_fn, _ = candidate_fns(spec, {3: net})
    game = au.game_of(spec)
    calc = bc.BRCalculator(game, stage3_policy(mean_fn))
    grid = stage_grid(spec, 3)
    sv = calc.stage_values(3, grid, brent=False)
    dw = game.dw
    delta = sv["Delta"] / dw
    inb = np.abs(grid) < 2.0 * spec.q - BAND_TOL
    stored = fnum(r["row"].get("Delta_argmax_d_t3"))
    j = int(np.argmax(delta))
    e_grid = np.linspace(spec.e_min, spec.e_max, int(round(spec.e_range / 0.01)) + 1)
    pts = list(DG4_D)
    labels = [f"node {x:g}" for x in pts]
    if np.isfinite(stored):
        pts.append(stored)
        labels.append("stored maximiser (evaluated iterate, final tier)")
    pts.append(float(grid[j]))
    labels.append("maximiser of Delta_3 at this export")
    curves = [{"run": run_name(r), "round": r["round"], "q": r["q"], "seed": r["seed"], "update": u, "label": lab,
               **curve_summary(calc, x, e_grid)} for x, lab in zip(pts, labels)]
    prof = {"round": r["round"], "q": r["q"], "seed": r["seed"], "update": u, "Delta_max_over_dw": float(delta.max()),
            "Delta_argmax_d": float(grid[j]), "Delta_in_max": float(delta[inb].max()), "Delta_in_argmax_d": float(grid[inb][np.argmax(delta[inb])]),
            "Delta_out_max": float(delta[~inb].max()), "e_hat_at_argmax": float(sv["e_hat"][j]),
            "a_dev_at_argmax": float(sv["a_dev"][j]), "e_hat_at_0": float(sv["e_hat"][int(np.argmin(np.abs(grid)))]),
            "a_dev_at_0": float(sv["a_dev"][int(np.argmin(np.abs(grid)))])}
    return {"curves": curves, "profile": prof, "arrays": {"grid": grid, "Delta_over_dw": delta, "a_dev": sv["a_dev"],
                                                          "e_hat": sv["e_hat"], "slope": sv["slope"]},
            "key": f"{run_name(r)}_u{u:05d}"}


def dg4(runs: Sequence[Mapping[str, Any]], out: Path, workers: int) -> None:
    """DG4 for the DG5 set at the four exports."""
    t0 = time.perf_counter()
    chosen = [r for r in runs if r["round"] == "t3_r2" and (r["q"], r["seed"]) in REPLAY_SET]
    items = [(r, u) for r in chosen for u in DG4_EXPORTS]
    res = parallel(dg4_one, items, workers)
    adir = out / "arrays"
    adir.mkdir(parents=True, exist_ok=True)
    for x in res:
        np.savez(adir / f"dg4_{x['key']}.npz", **x["arrays"])
    write_rows(out / "dg4_curves.csv", [c for x in res for c in x["curves"]])
    write_rows(out / "dg4_profiles.csv", [x["profile"] for x in res])
    write_json(out / "dg4_record.json", {"step": "dg4", "sec": time.perf_counter() - t0, "workers": workers, "n_runs": len(chosen),
                                         "calculator": "tools/t3/t3r2_br_calc.py (default CalcConfig)", **git_head()})


# ---------------------------------------------------------------------------------- DG5 (summary of the replays)
REPLAY_GROUPS = {"failure": ((50, 60501), (50, 60506), (50, 60513), (50, 60515), (60, 60516), (60, 60518)),
                 "pass_regime": ((50, 60505), (50, 60507), (60, 60503), (60, 60509)),
                 "pass_low": ((50, 60502), (50, 60504), (60, 60501), (60, 60502))}


def replay_group(q: int, seed: int) -> str:
    """The DG5 group of a replayed run (``failure``, ``pass_regime``, ``pass_low``)."""
    for g, members in REPLAY_GROUPS.items():
        if (q, seed) in members:
            return g
    raise KeyError((q, seed))


def pooled_share(share: np.ndarray, gnorm_total: np.ndarray) -> float:
    """``sum_u <g_r, g> / sum_u |g|^2`` over a window of updates: ``<g_r, g> = share * |g|^2`` (the share is per update)."""
    w = np.asarray(gnorm_total, dtype=float) ** 2
    sh = np.asarray(share, dtype=float)
    ok = np.isfinite(sh) & np.isfinite(w)
    return float((sh[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() and w[ok].sum() > 0 else float("nan")


def replay_windows(rows: Sequence[Mapping[str, str]], windows: Sequence[Tuple[int, int]] = WINDOWS3) -> List[Dict[str, float]]:
    """Window summaries of a replay's ``replay_updates.csv``: sample shares, gradient shares (mean over updates and pooled), the
    ratio share / sample share, the gradient norms and the actor's sensitivity and saturation."""
    out = []
    ups = np.array([int(float(r["update"])) for r in rows])
    f = lambda c: np.array([fnum(r[c]) for r in rows], dtype=float)  # noqa: E731
    N = f("N")
    cols = {k: f(k) for k in ("n_out", "n_in", "share_out", "share_in", "n_clamp_lo", "share_clamp_lo", "n_clamp_hi", "share_clamp_hi",
                              "n_clamp_free", "share_clamp_free", "gnorm_total", "gnorm_in", "gnorm_out", "sens_in_mean", "sat1_in",
                              "sat2_in", "sample_norm_mean_clamp_lo", "sample_norm_mean_clamp_free", "sample_norm_mean_in",
                              "sample_norm_mean_out", "adv_norm_mean_clamp_lo", "adv_norm_mean_out", "adv_norm_mean_in")}
    for a, b in windows:
        m = (ups >= a) & (ups <= b)
        row: Dict[str, float] = {"first": a, "last": b, "updates": int(m.sum())}
        for g in ("out", "in", "clamp_lo", "clamp_hi", "clamp_free"):
            frac = cols[f"n_{g}"][m] / N[m]
            row[f"sample_share_{g}"] = float(frac.mean())
            sh_ = cols[f"share_{g}"][m]
            row[f"grad_share_{g}_mean"] = float(np.nanmean(sh_)) if np.isfinite(sh_).any() else float("nan")
            row[f"grad_share_{g}_pooled"] = pooled_share(cols[f"share_{g}"][m], cols["gnorm_total"][m])
            row[f"ratio_{g}"] = row[f"grad_share_{g}_pooled"] / row["sample_share_" + g] if row["sample_share_" + g] > 0 else float("nan")
        gt, gi, go = cols["gnorm_total"][m], cols["gnorm_in"][m], cols["gnorm_out"][m]
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = (gt ** 2 - gi ** 2 - go ** 2) / (2.0 * gi * go)          # cos(g_in, g_out) from |g|^2 = |g_in|^2 + |g_out|^2 + 2 <g_in, g_out>
        row["cos_in_out_mean"] = float(np.nanmean(cos)) if np.isfinite(cos).any() else float("nan")
        for k in ("gnorm_total", "gnorm_in", "gnorm_out", "sens_in_mean", "sat1_in", "sat2_in", "sample_norm_mean_clamp_lo",
                  "sample_norm_mean_clamp_free", "sample_norm_mean_in", "sample_norm_mean_out", "adv_norm_mean_clamp_lo",
                  "adv_norm_mean_out", "adv_norm_mean_in"):
            row[k] = float(np.nanmean(cols[k][m])) if np.isfinite(cols[k][m]).any() else float("nan")
        out.append(row)
    return out


def replay_bins_summary(z: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, str]], q: float, half: float,
                        windows: Sequence[Tuple[int, int]] = WINDOWS3, width: float = 20.0) -> List[Dict[str, Any]]:
    """Per window: the starts per 20-unit bin (as a share of the window's samples) and the pooled gradient share per bin, as one
    row per (window, bin)."""
    ups = np.asarray(z["update"], dtype=int)
    g2 = np.array([fnum(r["gnorm_total"]) for r in rows], dtype=float) ** 2
    nb = z["n"].shape[1]
    edges = np.linspace(-half, half, nb + 1)
    centre = 0.5 * (edges[:-1] + edges[1:])
    out = []
    for a, b in windows:
        m = (ups >= a) & (ups <= b)
        n = np.asarray(z["n"])[m].sum(axis=0)
        w = g2[m][:, None]
        sh = np.asarray(z["share"])[m]
        pooled = np.where(np.isfinite(sh), sh * w, 0.0).sum(axis=0) / w.sum()
        for j in range(nb):
            out.append({"first": a, "last": b, "bin": j, "bin_centre": float(centre[j]), "in_band": bool(abs(centre[j]) < 2.0 * q),
                        "starts_share": float(n[j] / n.sum()), "grad_share_pooled": float(pooled[j])})
    return out


def dg5(runs: Sequence[Mapping[str, Any]], replay_dir: Path, out: Path) -> None:
    """Summaries of the 14 replays: window tables per run, the group medians and the per-bin coverage."""
    win_rows, bin_rows, ident = [], [], []
    for q, s in REPLAY_SET:
        d = Path(replay_dir) / f"q{q}" / f"seed{s}"
        rows = read_csv(d / "replay_updates.csv")
        grp = replay_group(q, s)
        for w in replay_windows(rows):
            win_rows.append({"q": q, "seed": s, "group": grp, **w})
        z = np.load(d / "replay_bins.npz")
        spec = te.spec_of(te.read_json(d / "run_config.json"))
        for b in replay_bins_summary(z, rows, spec.q, spec.domain_half(spec.T)):
            bin_rows.append({"q": q, "seed": s, "group": grp, **b})
        ident.append({"q": q, "seed": s, **{k: v for k, v in te.read_json(d / "replay_identity.json").items()
                                            if k in ("all_identical", "exports_identical", "updates_identical", "checks_identical", "wall_sec")}})
    write_rows(out / "dg5_windows.csv", win_rows)
    write_rows(out / "dg5_bins.csv", bin_rows)
    write_rows(out / "dg5_identity.csv", ident)
    med = []
    for grp in REPLAY_GROUPS:
        for first in sorted({w["first"] for w in win_rows}):
            sub = [w for w in win_rows if w["group"] == grp and w["first"] == first]
            row: Dict[str, Any] = {"group": grp, "first": first, "last": sub[0]["last"], "runs": len(sub)}
            for k in ("sample_share_out", "grad_share_out_pooled", "ratio_out", "sample_share_clamp_lo", "grad_share_clamp_lo_pooled",
                      "ratio_clamp_lo", "grad_share_in_pooled", "ratio_in", "sens_in_mean", "sat2_in", "gnorm_total", "gnorm_in",
                      "gnorm_out", "cos_in_out_mean"):
                v = [w[k] for w in sub if np.isfinite(w[k])]
                row[k + "_median"] = float(np.median(v)) if v else float("nan")
            med.append(row)
    write_rows(out / "dg5_group_medians.csv", med)
    write_json(out / "dg5_record.json", {"step": "dg5", "replay_dir": str(replay_dir),
                                         "share_definition": "pooled share = sum_u <g_r, g> / sum_u |g|^2 over the window; baseline = the sample share n_r / N",
                                         **git_head()})


# ---------------------------------------------------------------------------------- DG6
def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (average ranks for ties)."""
    from scipy.stats import spearmanr

    return float(spearmanr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))[0])


def share_class(s: float) -> str:
    """``regime`` (S3 > 0.2), ``mid`` (0.05 < S3 <= 0.2) or ``low`` (S3 <= 0.05)."""
    return "regime" if s > REGIME_SHARE else ("mid" if s > LOW_SHARE else "low")


def dg6(runs: Sequence[Mapping[str, Any]], v20_root: Path, out: Path, dg2_dir: Path) -> None:
    """S3 of the 80 runs, the T = 2 analogue of the 40 v2.0 runs, the relations and the timing."""
    g = gates()
    s3_rows, win_rows = [], []
    e0 = {}
    if (dg2_dir / "dg2_exports.csv").is_file():
        for x in read_csv(dg2_dir / "dg2_exports.csv"):
            e0[(x["round"], int(x["q"]), int(x["seed"]), int(x["update"]))] = fnum(x["e0"])
    stretch = {}
    for rnd, p in (("t3_r1", ROOT / "results" / "t3_r1" / "analysis" / "bw_failures" / "zero_effort.csv"),
                   ("t3_r2", ROOT / "results" / "t3_r2" / "confirmation" / "analysis" / "stretches.csv")):
        for x in read_csv(p):
            if int(float(x["nodes"])) > 0:               # ``stretches.csv`` lists every run, with 0 nodes where there is none
                stretch[(rnd, int(float(x["q"])), int(float(x["seed"])))] = x
    for r in runs:
        rows = read_csv(Path(r["tracked"]) / "ms_updates.csv")
        s3, lo, n = s3_of_rows(rows)
        onset = onset_window(rows)
        pr = r["row"]
        key = (r["round"], r["q"], r["seed"])
        rec = {"round": r["round"], "q": r["q"], "seed": r["seed"], "run": r["run"], "S3": s3, "lo": lo, "n": n,
               "class": share_class(s3), "onset_local": onset, "pass": passes(r, g),
               "eta_3": fnum(pr["eta_T_over_dw"]), "peak_T_signed": fnum(pr["peak_T_signed"]),
               "rmse_pos_T_over_eT0": fnum(pr["rmse_pos_T_over_eT0"]), "tail_mean_T_over_eT0": fnum(pr["tail_mean_T_over_eT0"]),
               "Gmax_full_over_dw": fnum(pr["Gmax_full_over_dw"]), "stretch_listed": key in stretch}
        for u in (100, 200, 400, 800, 1200, 1600):
            rec[f"e0_u{u}"] = e0.get((r["round"], r["q"], r["seed"], u), float("nan"))
        if onset is not None and e0:
            ups = [u for u in EXPORTS3]
            ser = np.array([e0[(r["round"], r["q"], r["seed"], u)] for u in ups])
            i_on = ups.index(onset)
            rec["e0_at_onset"] = float(ser[i_on])
            rec["e0_max_before_onset"] = float(ser[:i_on + 1].max())
            rec["e0_max_after_onset"] = float(ser[i_on:].max()) if i_on < len(ups) else float("nan")
            rec["e0_change_onset_to_final"] = float(ser[-1] - ser[i_on])
            rec["e0_change_onset_to_plus_10_exports"] = float(ser[min(i_on + 10, len(ups) - 1)] - ser[i_on])
            rec["e0_final_minus_max_before"] = float(ser[-1] - ser[:i_on + 1].max())
        s3_rows.append(rec)
        for a in range(1, 65):
            sh, lo_, n_ = clamp_share(rows, "d1_L_s3_out_lo", "d1_L_s3_out_n", 25 * (a - 1) + 1, 25 * a, lambda x: int(x["stage"]) == 3)
            win_rows.append({"round": r["round"], "q": r["q"], "seed": r["seed"], "window_last_local": 25 * a, "share_out": sh,
                             "lo": lo_, "n": n_})
    # timing: the tie-point effort after the regime's onset against the same quantity in the other runs, at a reference update
    # (the run's own onset, or the median onset of the regime runs of its q for a run without one)
    if e0:
        ups = list(EXPORTS3)
        med_on = {q: float(np.median([x["onset_local"] for x in s3_rows
                                      if x["q"] == q and x["S3"] > REGIME_SHARE and x["onset_local"] is not None])) for q in QS}
        for rec in s3_rows:
            ref = rec["onset_local"] if rec["onset_local"] is not None else med_on[rec["q"]]
            ser = np.array([e0[(rec["round"], rec["q"], rec["seed"], u)] for u in ups])
            i = min(max(int(ref // 25) - 1, 0), len(ups) - 1)
            rec.update({"ref_local": float(ref), "e0_at_ref": float(ser[i]), "e0_change_ref_plus_10_exports": float(ser[min(i + 10, 63)] - ser[i]),
                        "e0_change_ref_to_final": float(ser[-1] - ser[i]), "e0_drawdown_from_max_after_ref": float(ser[i:].max() - ser[-1])})
    write_rows(out / "dg6_s3_runs.csv", s3_rows)
    write_rows(out / "dg6_s3_windows.csv", win_rows)
    # ---- the T = 2 analogue of the 40 v2.0 runs
    t2 = []
    for q in QS:
        for s in range(30501, 30521):
            p = TRACKED_V20 / f"q{q}" / f"seed{s}" / "v2_updates.csv"
            sh, lo, n = s2_t2_of_rows(read_csv(p))
            t2.append({"q": q, "seed": s, "S2_analogue": sh, "lo": lo, "n": n})
    write_rows(out / "dg6_t2_analogue.csv", t2)
    # ---- relations
    rel = []
    for scope, sel in (("all", s3_rows), ("q50", [x for x in s3_rows if x["q"] == 50]), ("q60", [x for x in s3_rows if x["q"] == 60])):
        for col in ("eta_3", "peak_T_signed", "rmse_pos_T_over_eT0", "Gmax_full_over_dw"):
            rel.append({"scope": scope, "quantity": col, "n": len(sel), "spearman_with_S3": spearman([x["S3"] for x in sel], [x[col] for x in sel])})
        rel.append({"scope": scope, "quantity": "fail (1/0)", "n": len(sel),
                    "spearman_with_S3": spearman([x["S3"] for x in sel], [0.0 if x["pass"] else 1.0 for x in sel])})
    if e0:
        from scipy.stats import mannwhitneyu

        for q in QS:
            a = [x for x in s3_rows if x["q"] == q and x["S3"] > REGIME_SHARE]
            b = [x for x in s3_rows if x["q"] == q and x["S3"] <= REGIME_SHARE]
            for col in ("e0_change_ref_plus_10_exports", "e0_change_ref_to_final", "e0_drawdown_from_max_after_ref"):
                rel.append({"scope": f"timing q{q}", "quantity": col, "n": len(a) + len(b), "regime_median": float(np.median([x[col] for x in a])),
                            "other_median": float(np.median([x[col] for x in b])),
                            "mannwhitney_p": float(mannwhitneyu([x[col] for x in a], [x[col] for x in b]).pvalue)})
    write_rows(out / "dg6_relations.csv", rel)
    tab = []
    for scope, sel in (("all", s3_rows), ("q50", [x for x in s3_rows if x["q"] == 50]), ("q60", [x for x in s3_rows if x["q"] == 60])):
        for cls, f in (("S3 > 0.2", lambda x: x["S3"] > 0.2), ("S3 > 0.4", lambda x: x["S3"] > 0.4), ("S3 <= 0.2", lambda x: x["S3"] <= 0.2),
                       ("S3 < 0.05", lambda x: x["S3"] < 0.05)):
            sub = [x for x in sel if f(x)]
            tab.append({"scope": scope, "group": cls, "runs": len(sub), "failures": sum(1 for x in sub if not x["pass"]),
                        "with_stretch": sum(1 for x in sub if x["stretch_listed"]),
                        "median_eta_3": float(np.median([x["eta_3"] for x in sub])) if sub else float("nan"),
                        "median_peak_T_signed": float(np.median([x["peak_T_signed"] for x in sub])) if sub else float("nan"),
                        "median_rmse_pos_T_over_eT0": float(np.median([x["rmse_pos_T_over_eT0"] for x in sub])) if sub else float("nan"),
                        "median_Gmax_full": float(np.median([x["Gmax_full_over_dw"] for x in sub])) if sub else float("nan")})
    write_rows(out / "dg6_groups.csv", tab)
    write_json(out / "dg6_record.json", {"step": "dg6", **git_head()})


# ---------------------------------------------------------------------------------- DG8
def entry_stretch(grid: np.ndarray, e_hat: np.ndarray, q: float) -> Dict[str, Any]:
    """In-band stage-2 nodes with mean effort below 1: their number, the maximal runs of consecutive nodes and the longest."""
    idx = au.stretch_nodes(grid, e_hat, q)
    runs = au.runs_of_indices(idx)
    longest = max(runs, key=lambda ab: ab[1] - ab[0]) if runs else None
    return {"n_nodes": int(idx.size), "n_runs": len(runs), "longest_len": int(longest[1] - longest[0] + 1) if longest else 0,
            "longest_d": f"[{grid[longest[0]]:g}, {grid[longest[1]]:g}]" if longest else "", "idx": idx}


def dg8_one(r: Mapping[str, Any]) -> Dict[str, Any]:
    """DG8 of one run: ``u01600`` at the stage-2 input (phase entry) and the stage-2 export ``u03200`` (= the frozen stage 2)."""
    import torch

    torch.set_num_threads(1)
    d = Path(r["dir"])
    cfg = te.read_json(d / "run_config.json")
    spec = te.spec_of(cfg)
    ppo = cfg["record"]["ppo"]
    grid = stage_grid(spec, 2)
    kw = dict(hidden=int(ppo["hidden"]), c_min=float(ppo["c_min"]), mu_clamp=float(ppo["mu_clamp"]), with_critic=False)
    a_in = {k: v for k, v in np.load(d / "weights" / "u01600.npz").items()}
    a_fin = {k: v for k, v in np.load(d / "weights" / "u03200.npz").items()}
    e_in = network_on_grid(a_in, spec, grid, 2, **kw)["e_hat"]
    e_fin = network_on_grid(a_fin, spec, grid, 2, **kw)["e_hat"]
    ent = entry_stretch(grid, e_in, spec.q)
    fin = entry_stretch(grid, e_fin, spec.q)
    inb = np.abs(grid) < 2.0 * spec.q - BAND_TOL
    fin_idx = fin["idx"]
    return {"round": r["round"], "q": r["q"], "seed": r["seed"], "run": r["run"],
            "entry_nodes_lt1": ent["n_nodes"], "entry_runs": ent["n_runs"], "entry_longest_len": ent["longest_len"],
            "entry_longest_d": ent["longest_d"], "entry_median_effort_in_band": float(np.median(e_in[inb])),
            "entry_min_effort_in_band": float(e_in[inb].min()), "entry_max_effort_in_band": float(e_in[inb].max()),
            "final_stretch_nodes": fin["n_nodes"], "final_longest_len": fin["longest_len"], "final_longest_d": fin["longest_d"],
            "final_stretch_nodes_lt1_at_entry": int((e_in[fin_idx] < 1.0).sum()) if fin_idx.size else 0,
            "final_stretch_entry_median_effort": float(np.median(e_in[fin_idx])) if fin_idx.size else float("nan"),
            "final_stretch_entry_max_effort": float(e_in[fin_idx].max()) if fin_idx.size else float("nan")}


def dg8(runs: Sequence[Mapping[str, Any]], out: Path, dg6_dir: Path, workers: int) -> None:
    """DG8 for the 80 runs; joined with S3 from ``dg6_s3_runs.csv``."""
    t0 = time.perf_counter()
    res = parallel(dg8_one, list(runs), workers)
    s3 = {(x["round"], int(x["q"]), int(x["seed"])): x for x in read_csv(dg6_dir / "dg6_s3_runs.csv")}
    for x in res:
        x["S3"] = fnum(s3[(x["round"], x["q"], x["seed"])]["S3"])
        x["class"] = share_class(x["S3"])
        x["has_final_stretch"] = x["final_stretch_nodes"] > 0
    write_rows(out / "dg8_phase_entry.csv", res)
    tab = []
    for cls in ("regime", "mid", "low"):
        sub = [x for x in res if x["class"] == cls]
        tab.append({"class": cls, "runs": len(sub), "with_final_stretch": sum(1 for x in sub if x["has_final_stretch"]),
                    "with_entry_stretch_ge_1_node": sum(1 for x in sub if x["entry_nodes_lt1"] > 0),
                    "median_entry_nodes_lt1": float(np.median([x["entry_nodes_lt1"] for x in sub])) if sub else float("nan"),
                    "max_entry_nodes_lt1": max((x["entry_nodes_lt1"] for x in sub), default=0)})
    write_rows(out / "dg8_by_class.csv", tab)
    write_json(out / "dg8_record.json", {"step": "dg8", "sec": time.perf_counter() - t0, "workers": workers, **git_head()})


# ---------------------------------------------------------------------------------- retrospective trigger table
def trigger_flags(check: Mapping[str, Any]) -> Dict[str, bool]:
    """The trigger conditions of D4 on one stage-3 development check (the columns ``Delta``, ``s``, ``C``, ``R``, ``R_tail`` only; none
    of them reads the closed form, and ``valid`` is not used because it also depends on stages 1 and 2 of the composite candidate).
    ``degenerate``: ``Delta``, ``s`` or ``C`` not finite, or ``s <= 0`` (the check could not be evaluated)."""
    out = {"Delta>0.005": fnum(check["Delta"]) > TRIGGER_THRESHOLDS["Delta"]}
    for x in TRIGGER_THRESHOLDS["R"]:
        out[f"R>{x:g}"] = fnum(check["R"]) > x
    for x in TRIGGER_THRESHOLDS["R_tail"]:
        out[f"R_tail>{x:g}"] = fnum(check["R_tail"]) > x
    vals = [fnum(check["Delta"]), fnum(check["s"]), fnum(check["C"])]
    out["degenerate"] = bool((not all(np.isfinite(vals))) or fnum(check["s"]) <= 0.0)
    return out


def retro_table(rows: Sequence[Mapping[str, Any]], stage3_updates: int = 1600, total_updates: int = 3800) -> List[Dict[str, Any]]:
    """The retrospective table of one set of trigger conditions.

    Args:
        rows: One dict per run with ``pass`` (bool) and the flags of :func:`trigger_flags` (``flags``).
        stage3_updates: Updates of the stage-3 phase (the cost of one retraining).
        total_updates: Updates of one ``bw`` run.

    Returns:
        One row per trigger: the runs it fires on, how many of those failed, the failing runs it would miss, and the extra
        compute (fired runs times one stage-3 phase, over the total updates of all runs).
    """
    out = []
    names = list(rows[0]["flags"]) if rows else []
    n = len(rows)
    for nm in names:
        fired = [x for x in rows if x["flags"][nm]]
        failed = [x for x in rows if not x["pass"]]
        out.append({"trigger": nm, "n_runs": n, "n_fired": len(fired), "fired_runs": ";".join(x["label"] for x in fired),
                    "fired_and_failed": sum(1 for x in fired if not x["pass"]),
                    "failing_runs_total": len(failed),
                    "failing_missed": sum(1 for x in failed if not x["flags"][nm]),
                    "missed_runs": ";".join(x["label"] for x in failed if not x["flags"][nm]),
                    "extra_updates": len(fired) * stage3_updates, "extra_cost_fraction": len(fired) * stage3_updates / (n * total_updates)})
    return out


def retro(runs: Sequence[Mapping[str, Any]], out: Path) -> None:
    """The retrospective table on the 80 recorded runs (stage-3 check at local update 1600)."""
    g = gates()
    per_run = []
    for r in runs:
        ch = [c for c in read_csv(Path(r["tracked"]) / "ms_checks_stage3.csv") if int(c["local"]) == 1600][0]
        per_run.append({"label": f"{r['round'][-2:]}_q{r['q']}_s{r['seed']}", "round": r["round"], "q": r["q"], "seed": r["seed"],
                        "pass": passes(r, g), "Delta": fnum(ch["Delta"]), "s": fnum(ch["s"]), "R": fnum(ch["R"]),
                        "R_tail": fnum(ch["R_tail"]), "C": fnum(ch["C"]), "valid": ch["valid"], "flags": trigger_flags(ch)})
    write_rows(out / "retro_checks_local1600.csv", [{k: v for k, v in x.items() if k != "flags"} for x in per_run])
    tab = retro_table(per_run)
    write_rows(out / "retro_triggers.csv", tab)
    others = [x for x in per_run if x["label"] != "r2_q60_s60518"]
    write_json(out / "retro_record.json", {"step": "retro", "largest_R_other_than_60518": max(x["R"] for x in others),
                                           "largest_R_tail_other_than_60518": max(x["R_tail"] for x in others),
                                           "R_60518": next((x["R"] for x in per_run if x["label"] == "r2_q60_s60518"), None),
                                           "R_tail_60518": next((x["R_tail"] for x in per_run if x["label"] == "r2_q60_s60518"), None),
                                           "thresholds": {k: list(v) if isinstance(v, tuple) else v for k, v in TRIGGER_THRESHOLDS.items()},
                                           **git_head()})


# ---------------------------------------------------------------------------------- DG7
def rank_from_top(values: Sequence[float], final: float) -> int:
    """Rank of ``final`` among ``values`` counted from the largest (1 = the largest; ties rank above)."""
    return 1 + int(sum(1 for v in values if v > final))


def dg7(runs: Sequence[Mapping[str, Any]], out: Path) -> None:
    """The per-failure evidence table: gate values, S3, formation, last-16 statements, stage-2 stretch, development check."""
    g = gates()
    d2r = {(x["round"], int(x["q"]), int(x["seed"])): x for x in read_csv(out / "dg2_runs.csv")}
    s3 = {(x["round"], int(x["q"]), int(x["seed"])): x for x in read_csv(out / "dg6_s3_runs.csv")}
    d8 = {(x["round"], int(x["q"]), int(x["seed"])): x for x in read_csv(out / "dg8_phase_entry.csv")}
    ck = {(x["round"], int(x["q"]), int(x["seed"])): x for x in read_csv(out / "dg3_checks.csv")}
    a8 = {}
    for rnd, p in (("t3_r1", ROOT / "results" / "t3_r2" / "audit" / "a8_last16.csv"),
                   ("t3_r2", ROOT / "results" / "t3_r2" / "confirmation" / "analysis" / "a8_last16_confirmation.csv")):
        for x in read_csv(p):
            a8.setdefault((rnd, int(float(x["q"])), int(float(x["seed"]))), []).append(x)
    ret = {(x["round"], int(x["q"]), int(x["seed"])): x for x in read_csv(out / "retro_checks_local1600.csv")}
    rows = []
    for rnd, q, s in FAILURES:
        r = next(x for x in runs if (x["round"], x["q"], x["seed"]) == (rnd, q, s))
        k = (rnd, q, s)
        pr = r["row"]
        eta, rmse, tail, gm = (fnum(pr["eta_T_over_dw"]), fnum(pr["rmse_pos_T_over_eT0"]), fnum(pr["tail_mean_T_over_eT0"]),
                               fnum(pr["Gmax_full_over_dw"]))
        failing = []
        if not eta <= g["G-A"]["eta_T_over_dw"]:
            failing.append("G-A eta_3")
        if not rmse <= g["G-A"]["rmse_pos_T_over_eT0"]:
            failing.append("G-A rmse")
        if not tail <= g["G-A"]["tail_mean_T_over_eT0"]:
            failing.append("G-A tail")
        if not gm <= g["G-F"]["Gmax_full_over_dw"]:
            failing.append("G-F")
        last16 = a8[k]
        eta16 = [fnum(x["eta_3_over_dw"]) for x in last16]
        rm16 = [fnum(x["rmse_pos_T_over_eT0"]) for x in last16]
        rows.append({
            "round": rnd, "q": q, "seed": s, "failing": ";".join(failing), "eta_3": eta, "rmse_pos": rmse, "tail_mean": tail, "Gmax_full": gm,
            "S3": fnum(s3[k]["S3"]), "S3_class": s3[k]["class"], "onset_local": s3[k]["onset_local"],
            "F_final": fnum(d2r[k]["F_final"]), "formation_threshold": fnum(d2r[k]["formation_threshold"]),
            "formation_export": d2r[k]["formation_export"], "e0_final": fnum(d2r[k]["e0_final"]),
            "sens_in_mean_final": fnum(d2r[k]["sens_in_mean_final"]), "sat1_in_final": fnum(d2r[k]["sat1_in_final"]),
            "sat2_in_final": fnum(d2r[k]["sat2_in_final"]),
            "eta3_last16_min": min(eta16), "eta3_last16_median": float(np.median(eta16)), "eta3_last16_max": max(eta16),
            "eta3_last16_n_above_0.005": sum(1 for v in eta16 if v > 0.005), "eta3_final_rank_from_top": rank_from_top(eta16, eta16[-1]),
            "rmse_last16_max": max(rm16), "rmse_final_rank_from_top": rank_from_top(rm16, rm16[-1]),
            "dev_Delta_L1600": fnum(ret[k]["Delta"]), "dev_R_L1600": fnum(ret[k]["R"]), "dev_R_tail_L1600": fnum(ret[k]["R_tail"]),
            "dev_C_L1600": fnum(ret[k]["C"]), "dev_Delta_L100": fnum(ck[k]["Delta_L100"]),
            "stage2_final_stretch_nodes": int(d8[k]["final_stretch_nodes"]), "stage2_entry_nodes_lt1": int(d8[k]["entry_nodes_lt1"]),
            "stage2_stretch_nodes_lt1_at_entry": int(d8[k]["final_stretch_nodes_lt1_at_entry"])})
    write_rows(out / "dg7_failures.csv", rows)
    write_json(out / "dg7_record.json", {"step": "dg7", "rule": "rank_from_top counts from the largest; 1 = the failing value is the largest of the 16 exports",
                                         **git_head()})


# ---------------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R3 diagnosis tool", allow_abbrev=False)
    ap.add_argument("step", choices=["dg1", "dg2", "dg3", "dg4", "dg5", "dg6", "dg8", "retro", "dg7"])
    ap.add_argument("--t3r1-root", default=str(DEFAULT_R1))
    ap.add_argument("--t3r2-root", default=str(DEFAULT_R2))
    ap.add_argument("--v20-root", default=str(DEFAULT_V20))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--replay-dir", default=str(ROOT / "results" / "t3_r3" / "replays"))
    ap.add_argument("--workers", type=int, default=8, help="at most 40")
    a = ap.parse_args(argv)
    if not 1 <= a.workers <= 40:
        ap.error("--workers must be in 1..40")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = bw_runs(Path(a.t3r1_root), Path(a.t3r2_root))
    t0 = time.perf_counter()
    if a.step == "dg1":
        rows = parallel(dg1_one, runs, a.workers)
        write_rows(out / "dg1_inventory.csv", rows)
        write_rows(out / "dg1_inventory_v20.csv", v20_inventory(Path(a.v20_root)))
    elif a.step == "dg2":
        dg2(runs, out, a.workers)
    elif a.step == "dg3":
        dg3(runs, out, a.workers)
    elif a.step == "dg4":
        dg4(runs, out, a.workers)
    elif a.step == "dg5":
        dg5(runs, Path(a.replay_dir), out)
    elif a.step == "dg6":
        dg6(runs, Path(a.v20_root), out, out)
    elif a.step == "dg8":
        dg8(runs, out, out, a.workers)
    elif a.step == "retro":
        retro(runs, out)
    elif a.step == "dg7":
        dg7(runs, out)
    rec = {"step": a.step, "t3r1_root": a.t3r1_root, "t3r2_root": a.t3r2_root, "v20_root": a.v20_root, "workers": a.workers,
           "wall_sec": time.perf_counter() - t0, **git_head()}
    write_json(out / f"record_{a.step}.json", rec)
    au.log_cost(out, a.step, time.perf_counter() - t0, a.workers, len(runs))
    print(f"[{a.step}] done in {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
