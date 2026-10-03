#!/usr/bin/env python3
"""Pre-registered analysis of the two R1 pilot waves and the three data-driven reports.

Implements ``reports/v2/refine/01_preregistration.md`` section 5 (PI prompt sections 5 and 6):
final tier for every gate metric, recovery metrics are tier independent, every difference is
``arm - baseline`` paired by (q, seed); per q the median, the mean, ``n_better`` out of 10 and 95%
percentile bootstrap CIs of the mean and of the median. Bootstrap (D5): 10,000 resamples of the
paired seeds, ``numpy.random.default_rng(20261003)``, one FRESH generator per (q, statistic) in
table order (``idx = rng.integers(0, n, size=(10000, n))``), CI = 2.5 / 97.5 percentiles.

Inputs (read only): ``<root>/stage1/q*/seed*/<arm>/``, ``<root>/stage2/q*/seed*/<arm>/``, the
rehearsal reference ``--ref-root`` (``q*/seed*/{gates.json, induced_band.json, state_end_A.pt}``),
``<root>/{parents_A_checks,stage1_base_checks,stage2_base_checks}.json`` and, for the decision
report, ``<root>/d1_clamp/d1_flags.csv``, ``<root>/d2_verifier_sensitivity/
detection_limits.csv`` and ``<root>/continuation_check.json``.

Outputs: CSVs under ``--out``, figures under ``--figures`` (png + pdf, fonts embedded as
type 42) and the reports ``04_pilot_stage1.md``, ``05_pilot_stage2.md``, ``06_decision_inputs.md``
under ``--reports``. Every number of a report is read from a CSV that the report cites; the tool
is deterministic given the same inputs. A run counts as complete iff ``status.json`` has
``state == done`` and ``exit_code == 0`` and ``final_v2.json`` holds a final-tier evaluation;
every other planned run is listed explicitly (failed / incomplete / missing).

This tool is an evaluation tool: the closed-form equilibrium (``g1``, ``g2``) is used here only
to measure errors; nothing here touches training.

Usage:
  python tools/v2/refine_analysis.py {stage1|stage2|decision|all} --root results/v2_refine \
      --out results/v2_refine/analysis --reports reports/v2/refine \
      [--figures reports/v2/refine/figures]
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_REPO = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine")
CAN_ROOT = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/"
                "pilot-4-stabilization-fb99a2")


def _find_repo(argv: Optional[Sequence[str]] = None) -> Path:
    """Repository root: ``--repo-root``, env ``REFINE_REPO_ROOT``, this file's repo, the default."""
    argv = list(sys.argv[1:] if argv is None else argv)
    cands: List[Optional[str]] = []
    for i, a in enumerate(argv):
        if a == "--repo-root" and i + 1 < len(argv):
            cands.append(argv[i + 1])
        elif a.startswith("--repo-root="):
            cands.append(a.split("=", 1)[1])
    cands += [os.environ.get("REFINE_REPO_ROOT"), str(Path(__file__).resolve().parents[2]),
              str(DEFAULT_REPO)]
    for c in cands:
        if c and (Path(c) / "utils" / "v2_metrics.py").exists():
            return Path(c).resolve()
    raise RuntimeError("cannot locate the repository (utils/v2_metrics.py); use --repo-root")


REPO = _find_repo()
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "v2"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from agents.ppo_curriculum import (CurriculumPPO, PPOConfig,  # noqa: E402
                                   mean_effort_numpy)
from agents.ppo_pathwise import effort_mean, expected_payoff, foc_residual  # noqa: E402
from decomposition import rows_for  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402
from launch_refine import (BASE_ARM, METHOD_ARMS, REHEARSAL, STAGE1_ARMS,  # noqa: E402
                           STAGE2_ARMS)
from pilot1_smoothed_game import centred_nodes  # noqa: E402
from rng_divergence import STREAMS, first_divergence  # noqa: E402
from run.run_v2_T2_locked import SMOOTH_NODES, stage2_extra, verdicts  # noqa: E402
from utils.theory_multistage import f_xi, g2_two_stage  # noqa: E402

# --------------------------------------------------------------------------- constants
N_BOOT = 10000
BOOT_SEED = 20261003
QS: Tuple[int, ...] = (50, 60)
SEEDS: Tuple[int, ...] = tuple(range(10501, 10511))
PROTOCOL = REPO / "protocols" / "v2_T2_locked_v1_1.json"
TARGET_0929 = 0.05                       # PI target |stage-1 error| <= 0.05 (reported only)
LAST5 = (2100, 2125, 2150, 2175, 2200)   # global updates of the last 5 stage-1 weight exports
PRIMARY = {"stage1": "stage1_rel_err_abs", "stage2": "stage2_peak_rel_err_abs"}
PHASE = {"stage1": "B", "stage2": "A"}
TIER_FINAL, TIER_DEV = "final", "development"
LOGGED_WINDOW = 20                       # rolling window of the logged A_detmean loss
STOP_EPOCHS = tuple(range(1, 11))


def _rel(p: Any) -> str:
    """Repo-relative path string if ``p`` lies inside the repository, else the absolute path."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


# metric catalogues: (column, direction) with True = smaller is better, None = no direction
S1_METRICS: List[Tuple[str, Optional[bool]]] = [
    ("stage1_rel_err_abs", True), ("stage1_rel_err_signed", None), ("e1_at_0", None),
    ("learning_rel", None), ("learning_rel_abs", True), ("Gmax_full_over_dw", True),
    ("EXP_root_over_dw", True), ("dReach_over_dw", True), ("Deltamax_all_over_dw", True),
    ("dFull_over_dw", True), ("gmax_dev_minus_final_abs", True),
    ("within_run_sd_e1_last5", True), ("within_run_range_e1_last5", True),
    ("kl_mean", None), ("clip_frac_mean", None), ("gn_actor_mean", None), ("gn_actor_max", None),
    ("n_epochs_run_mean", None), ("adv_s1_std_mean", None), ("phase_wall_sec", True)]
S2_METRICS: List[Tuple[str, Optional[bool]]] = [
    ("stage2_peak_rel_err_abs", True), ("stage2_peak_rel_err_signed", None),
    ("stage2_peak_locfree_rel_err_abs", True), ("stage2_peak_locfree_rel_err", None),
    ("stage2_rmse_pos_over_g2_0", True), ("stage2_tail_mean_over_g2_0", True),
    ("stage2_tail_max_over_g2_0", True), ("eta_T_over_dw", True),
    ("eta_dev_minus_final_abs", True), ("stage2_sym_err_max", True),
    ("sigma_effort_at_0_t2", None), ("e2_at_0", None), ("smoothed_pred_gap_d0", None),
    ("smoothed_share_peak_gap_d0", None), ("kl_mean", None), ("clip_frac_mean", None),
    ("gn_actor_mean", None), ("gn_actor_max", None), ("n_epochs_run_mean", None),
    ("phase_wall_sec", True)]
METRICS = {"stage1": S1_METRICS, "stage2": S2_METRICS}
DIRECTION = {m: d for ms in METRICS.values() for m, d in ms}
STREAM_COLS = [f"rng_div_{s}" for s in STREAMS]


# --------------------------------------------------------------------------- statistics
def boot_ci(x: np.ndarray, stat: str = "mean", n_boot: Optional[int] = None) -> Tuple[float, float]:
    """95% percentile bootstrap CI of the mean or median of ``x`` (pre-registered scheme).

    A fresh ``default_rng(20261003)`` draws ``idx = integers(0, n, size=(n_boot, n))``; the CI
    is the 2.5 / 97.5 percentiles of the resampled statistic.

    Args:
        x: Paired values (1-D).
        stat: ``"mean"`` or ``"median"``.
        n_boot: Number of resamples (default :data:`N_BOOT`).

    Returns:
        ``(lo, hi)``; ``(nan, nan)`` for an empty input.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOT_SEED)
    nb = N_BOOT if n_boot is None else int(n_boot)
    idx = rng.integers(0, x.size, size=(nb, x.size))
    r = x[idx]
    s = r.mean(axis=1) if stat == "mean" else np.median(r, axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def paired_summary(diff: np.ndarray, lower_better: Optional[bool]) -> Dict[str, Any]:
    """Median, mean, sign counts, ``n_better`` and the two bootstrap CIs of paired differences.

    Args:
        diff: ``arm - baseline`` per paired seed.
        lower_better: True if a decrease is an improvement, None if the metric has no direction
            (then ``n_better`` is None).

    Returns:
        Flat dict (n_pairs, mean, median, n_better, n_pos, n_neg, n_zero, ci_mean_lo/hi,
        ci_median_lo/hi, excl0_improving).
    """
    d = np.asarray(diff, dtype=float)
    lo_m, hi_m = boot_ci(d, "mean")
    lo_d, hi_d = boot_ci(d, "median")
    out: Dict[str, Any] = {
        "n_pairs": int(d.size), "mean": float(d.mean()) if d.size else float("nan"),
        "median": float(np.median(d)) if d.size else float("nan"),
        "n_better": int((d < 0).sum()) if lower_better else None,
        "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "n_zero": int((d == 0).sum()),
        "ci_mean_lo": lo_m, "ci_mean_hi": hi_m, "ci_median_lo": lo_d, "ci_median_hi": hi_d}
    out["excl0_improving"] = (None if lower_better is None
                              else bool(d.size > 0 and hi_m < 0.0))
    return out


def sd_ratio(arm: np.ndarray, base: np.ndarray) -> Dict[str, Any]:
    """Across-seed SD ratio arm/baseline (ddof=1) with a paired-resampling bootstrap CI.

    The same resampled seed indices are used for both arms (``default_rng(20261003)``, one
    index matrix of shape (10000, n)). Resamples with a zero baseline SD are excluded from the
    percentiles and counted.
    """
    a, b = np.asarray(arm, float), np.asarray(base, float)
    n = a.size
    out: Dict[str, Any] = {"n": int(n), "sd_arm": float("nan"), "sd_base": float("nan"),
                           "ratio": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                           "n_nonfinite_resamples": 0}
    if n < 2:
        return out
    out["sd_arm"], out["sd_base"] = float(a.std(ddof=1)), float(b.std(ddof=1))
    out["ratio"] = out["sd_arm"] / out["sd_base"] if out["sd_base"] > 0 else float("nan")
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a[idx].std(axis=1, ddof=1) / b[idx].std(axis=1, ddof=1)
    fin = np.isfinite(r)
    out["n_nonfinite_resamples"] = int((~fin).sum())
    if fin.any():
        out["ci_lo"], out["ci_hi"] = (float(v) for v in np.percentile(r[fin], [2.5, 97.5]))
    return out


# --------------------------------------------------------------------------- actor loading
def load_actor(source: Any, conc_scale: Optional[float] = None) -> torch.nn.Module:
    """Beta actor of a weight-export NPZ or a v2 full-state checkpoint, ``conc_scale`` applied.

    The loaders of the earlier pilots (``pilot1_analysis.load_actor_policy``,
    ``pilot2_analysis._actor_fns``, ``induced_band.actor_policy``) ignore the concentration
    factor of an annealed actor; this one reads it from the NPZ array ``conc_scale`` or from
    ``state['agent']['conc_scale']['actor']`` (absent = 1.0) unless ``conc_scale`` is given.

    Args:
        source: Path to ``weights/u*.npz`` (``actor.*`` arrays) or to ``state_end_*.pt``.
        conc_scale: Override of the factor (None = read it from the source).

    Returns:
        A ``BetaActor`` in eval mode with ``conc_scale`` set.
    """
    path = str(source)
    if path.endswith(".npz"):
        with np.load(path) as w:
            sd = {k[len("actor."):]: torch.as_tensor(w[k]) for k in w.files
                  if k.startswith("actor.")}
            scale = float(w["conc_scale"]) if "conc_scale" in w.files else 1.0
    else:
        st = torch.load(path, map_location="cpu", weights_only=False)["agent"]
        sd = st["actor"]
        scale = float((st.get("conc_scale") or {}).get("actor", 1.0))
    agent = CurriculumPPO(PPOConfig(), torch.Generator().manual_seed(0), np.random.default_rng(0))
    agent.actor.load_state_dict(sd)
    agent.actor.conc_scale = scale if conc_scale is None else float(conc_scale)
    agent.actor.eval()
    return agent.actor


@torch.no_grad()
def actor_alpha_beta(actor: torch.nn.Module, spec: GameSpec, t: int,
                     d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(alpha, beta) as float64 of the float32 network output (the run's own ``beta_fn``)."""
    obs = torch.as_tensor(spec.encode_obs(t, np.asarray(d, dtype=float)))
    a, b = actor(obs)
    return a.numpy().astype(float), b.numpy().astype(float)


def sigma_effort(alpha: float, beta: float, spec: GameSpec) -> float:
    """Standard deviation of the Beta(alpha, beta) effort in effort units."""
    s = alpha + beta
    return spec.e_range * math.sqrt(alpha * beta / (s * s * (s + 1.0)))


# --------------------------------------------------------------------------- smoothed game
def g2_peak(spec: GameSpec) -> float:
    """Closed-form stage-2 peak e2*(0) (evaluation only)."""
    return float(g2_two_stage(np.zeros(1), spec.q, spec.w_h, spec.w_l, spec.k, spec.e_max)[0])


def smoothed_pred_d0(alpha: float, beta: float, spec: GameSpec,
                     n_nodes: int = SMOOTH_NODES) -> float:
    """Smoothed-game prediction of the d = 0 stage-2 effort (Pilot-1 method).

    ``e_pred(0) = DW / (2k) E[f_xi(x_i - x_j)]`` with ``x_i, x_j`` the mean-centred effort noise
    of the learned Beta at d = 0, ``n_nodes`` equal-probability nodes per Beta and the tensor
    product of the two (the same construction as ``run_v2_T2_locked.smoothed_share``).
    """
    x = centred_nodes(alpha, beta, n_nodes, spec.e_range)
    return float(spec.dw / (2.0 * spec.k) * f_xi(x[:, None] - x[None, :], spec.q).mean())


def smoothed_record(alpha: float, beta: float, spec: GameSpec, e_hat_0: float) -> Dict[str, float]:
    """Predicted value, predicted / observed peak gap and the share of the gap explained."""
    g20 = g2_peak(spec)
    pred0 = smoothed_pred_d0(alpha, beta, spec)
    obs_gap = g20 - e_hat_0
    return {"smoothed_e_pred_0": pred0, "smoothed_pred_gap_d0": g20 - pred0,
            "observed_gap_d0": obs_gap,
            "smoothed_share_peak_gap_d0": (g20 - pred0) / obs_gap if obs_gap != 0 else float("nan")}


# --------------------------------------------------------------------------- A_detmean offline
def bin_centres(spec: GameSpec, bin_width: float, stage: Optional[int] = None) -> np.ndarray:
    """Bin centres of the exploring-start bins on D_stage (default: the final stage)."""
    edges = StartSampler(spec, bin_width).bin_edges(spec.T if stage is None else stage)
    return 0.5 * (edges[:-1] + edges[1:])


def offline_objective(actor: torch.nn.Module, spec: GameSpec, centres: np.ndarray
                      ) -> Dict[str, float]:
    """Fixed-grid pathwise objective and first-order-condition residual of an actor.

    Definition (stated in the report): on the bin centres ``d`` of D_2 (``es_bin_width`` 10),
    ``e(d)`` is the actor's Beta-mean effort at d and the opponent is the SAME actor at ``-d``;
    ``J = mean_d R(d, e(d), e(-d))`` with ``R = w_l + DW F_xi(d + e - e_opp) - k e^2``
    (``agents.ppo_pathwise.expected_payoff`` on ``effort_mean``; the training loss is ``-J`` on
    random exploring starts) and ``foc = |dR/de| = |DW f_xi(d + e - e_opp) - 2 k e|``
    (``agents.ppo_pathwise.foc_residual``), reported as its mean and max over the bin centres.
    """
    c = np.asarray(centres, dtype=float)
    d = torch.as_tensor(c, dtype=torch.float64)
    with torch.no_grad():
        e = effort_mean(actor, torch.as_tensor(spec.encode_obs(spec.T, c)), spec)
        eo = effort_mean(actor, torch.as_tensor(spec.encode_obs(spec.T, -c)), spec)
        r = expected_payoff(spec, d, e, eo)
        foc = foc_residual(spec, d, e, eo).abs()
    return {"J": float(r.mean()), "foc_mean": float(foc.mean()), "foc_max": float(foc.max())}


# --------------------------------------------------------------------------- run discovery
def classify_run(d: Path) -> Dict[str, Any]:
    """Status of one run directory from ``status.json`` (and ``final_v2.json``).

    ``done`` = state done and exit code 0 and a final-tier evaluation; ``failed`` = state failed
    or a nonzero exit code or a final-tier evaluation error; ``incomplete`` = anything that has
    started but is not done (running, killed); ``missing`` = no ``status.json``.
    """
    st_path = Path(d) / "status.json"
    if not st_path.exists():
        return {"status": "missing", "exit_code": None, "status_info": "no status.json"}
    st = json.loads(st_path.read_text())
    state, code = st.get("state"), st.get("exit_code")
    if state == "failed" or (state == "done" and code not in (0, None)):
        tb = (st.get("traceback") or "").strip().splitlines()
        return {"status": "failed", "exit_code": code,
                "status_info": tb[-1] if tb else f"state={state} exit_code={code}"}
    if state != "done" or code != 0:
        return {"status": "incomplete", "exit_code": code, "status_info": f"state={state}"}
    fv = Path(d) / "final_v2.json"
    if not fv.exists():
        return {"status": "incomplete", "exit_code": code,
                "status_info": "state done but final_v2.json missing"}
    j = json.loads(fv.read_text())
    if "error" in (j.get(TIER_FINAL) or {"error": "no final tier"}):
        return {"status": "failed", "exit_code": code,
                "status_info": f"final-tier evaluation error: {(j.get(TIER_FINAL) or {})}"}
    return {"status": "done", "exit_code": code, "status_info": ""}


@dataclass(frozen=True)
class Ctx:
    """Paths of one analysis (picklable, passed to the workers)."""

    root: str
    ref_root: str
    protocol: str = str(PROTOCOL)

    def run_dir(self, wave: str, q: int, seed: int, arm: str) -> Path:
        """``<root>/<wave>/q<q>/seed<seed>/<arm>``."""
        return Path(self.root) / wave / f"q{q}" / f"seed{seed}" / arm

    def ref_dir(self, q: int, seed: int) -> Path:
        """``<ref_root>/q<q>/seed<seed>`` (the rehearsal_v1_1 parent run)."""
        return Path(self.ref_root) / f"q{q}" / f"seed{seed}"


@functools.lru_cache(maxsize=None)
def _protocol(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def spec_for_q(ctx: Ctx, q: int) -> GameSpec:
    """Game of the locked protocol record for ``q``."""
    return GameSpec(**_protocol(ctx.protocol)["records"][str(q)]["game"])


def _nan() -> float:
    return float("nan")


def _f(x: Any) -> float:
    return float(x) if x is not None else _nan()


def _jload(p: Path) -> Any:
    return json.loads(Path(p).read_text())


@functools.lru_cache(maxsize=64)
def _parent_actor_sd(path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=False)["agent"]["actor"]


def base_row(wave: str, q: int, seed: int, arm: str, d: Path) -> Dict[str, Any]:
    """Identity and status columns of a per-run row (metrics are filled by the extractors)."""
    row: Dict[str, Any] = {"wave": wave, "q": q, "seed": seed, "arm": arm,
                           "run_dir": _rel(d)}
    row.update(classify_run(d))
    row["complete"] = row["status"] == "done"
    mp = Path(d) / "manifest.json"
    if mp.exists():                       # provenance: the commit and tree state the run recorded
        g = json.loads(mp.read_text()).get("git") or {}
        row["manifest_commit"], row["manifest_dirty"] = g.get("short"), g.get("dirty")
    return row


def opt_diagnostics(d: Path, phase: str, anomalies: List[str]) -> Dict[str, Any]:
    """Optimisation diagnostics of one run from ``v2_updates.csv`` and ``train_history.json``.

    Columns that the run schema does not have (``A_detmean``: no KL, clip, epochs, advantages)
    are skipped, not treated as errors.
    """
    out: Dict[str, Any] = {}
    up = pd.read_csv(Path(d) / "v2_updates.csv")
    out["n_updates"] = int(len(up))
    out["phase_first_update"] = int(up["update"].iloc[0])
    for col, name in (("kl_final_epoch", "kl"), ("clip_frac", "clip_frac")):
        if col in up:
            out[f"{name}_mean"] = float(up[col].mean())
            out[f"{name}_median"] = float(up[col].median())
    for col, name in (("adv_s1_std", "adv_s1_std_mean"), ("adv_all_std", "adv_all_std_mean")):
        if col in up:
            out[name] = float(up[col].mean())
    if "n_epochs_run" in up:
        ne = up["n_epochs_run"].astype(int)
        out["n_epochs_run_mean"] = float(ne.mean())
        for k in STOP_EPOCHS:
            out[f"n_epochs_cnt_{k}"] = int((ne == k).sum())
    th = _jload(Path(d) / "train_history.json")
    hist = th["history"]
    for key, name, fn in (("grad_norm_actor_mean", "gn_actor_mean", np.mean),
                          ("grad_norm_actor_max", "gn_actor_max", np.max)):
        vals = [h[key] for h in hist if key in h]
        out[name] = float(fn(vals)) if vals else _nan()
    for key, name in (("n_minibatch_steps", "n_minibatch_steps_total"),
                      ("n_actor_steps", "n_actor_steps_total")):
        vals = [h[key] for h in hist if key in h]
        out[name] = int(sum(vals)) if vals else None
    cur = [c for c in th.get("curriculum", []) if c.get("phase") == phase]
    if cur:
        out["phase_episodes"] = int(cur[-1].get("episodes", 0))
        out["phase_transitions"] = int(cur[-1].get("transitions", 0))
        out["phase_local_updates"] = int(cur[-1].get("local_updates", 0))
    else:
        anomalies.append(f"no curriculum entry for phase {phase} in train_history.json")
    rs = _jload(Path(d) / "v2_run_summary.json")
    pt = (rs.get("phase_timing") or {}).get(phase) or {}
    out["phase_wall_sec"] = _f(pt.get("wall_sec"))
    cs = rs.get("costs") or {}
    out["train_update_sec"] = _f(cs.get("train_update_sec"))
    out["train_rollout_sec"] = _f(cs.get("train_rollout_sec"))
    out["dev_verifier_sec"] = _f(cs.get("dev_verifier_sec"))
    out["conc_scale_final"] = _f((rs.get("conc_scale_final") or {}).get("actor"))
    st = _jload(Path(d) / "status.json")
    out["total_wall_sec"] = _f(st.get("total_wall_sec"))
    return out


def rng_columns(d: Path, base_dir: Optional[Path], same: bool) -> Dict[str, Any]:
    """First update at which each RNG stream differs from the baseline run's (or ``never``)."""
    cols: Dict[str, Any] = {c: "n/a" for c in STREAM_COLS}
    if same or base_dir is None:
        return cols
    try:
        fd = first_divergence(str(d), str(base_dir))
    except Exception as exc:  # noqa: BLE001 - reported in the table, not raised
        cols = {c: f"n/a ({type(exc).__name__})" for c in STREAM_COLS}
        return cols
    for s in STREAMS:
        cols[f"rng_div_{s}"] = fd[s]
    return cols


S1_FINAL_KEYS = ("e1_at_0", "g1", "stage1_rel_err_signed", "stage1_rel_err_abs",
                 "sigma_effort_at_0_t1", "Gmax_full_over_dw", "Gmax_full_t", "Gmax_full_d",
                 "EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw", "dFull_over_dw",
                 "eta_T_over_dw", "valid")
S2_FINAL_KEYS = ("stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs",
                 "stage2_rmse_pos_over_g2_0", "stage2_tail_mean_over_g2_0",
                 "stage2_tail_max_over_g2_0", "stage2_tail_mean", "stage2_tail_max",
                 "stage2_sym_err_max", "eta_T_over_dw", "sigma_effort_at_0_t2", "g2_at_0",
                 "e2_at_0", "valid")


def extract_stage1(ctx: Ctx, q: int, seed: int, arm: str) -> Dict[str, Any]:
    """Per-run metrics of one stage-1 (Phase-B) run; empty metrics for an incomplete run."""
    d = ctx.run_dir("stage1", q, seed, arm)
    row = base_row("stage1", q, seed, arm, d)
    anomalies: List[str] = []
    if not row["complete"]:
        row["anomalies"] = f"run {row['status']}: {row['status_info']}"
        return row
    fv = _jload(d / "final_v2.json")
    fin, dev = fv[TIER_FINAL], fv[TIER_DEV]
    if row.get("manifest_dirty") is True:
        anomalies.append("manifest.json records dirty = true")
    row.update({k: fin.get(k) for k in S1_FINAL_KEYS if k != "valid"})
    row["valid_final"], row["valid_dev"] = bool(fin.get("valid")), bool(dev.get("valid"))
    row["gmax_dev"] = _f(dev.get("Gmax_full_over_dw"))
    row["gmax_dev_minus_final"] = row["gmax_dev"] - _f(fin.get("Gmax_full_over_dw"))
    row["gmax_dev_minus_final_abs"] = abs(row["gmax_dev_minus_final"])
    spec = spec_for_q(ctx, q)
    proto = _protocol(ctx.protocol)
    ref = ctx.ref_dir(q, seed)
    # ---- parent (shared stage 2): G-A, G-N eta part, band
    try:
        gates = _jload(ref / "gates.json")
        mv = gates["metric_values"]
        vals = {"eta_final": mv["eta_final"], "eta_dev": mv["eta_dev"], "rmse": mv["rmse"],
                "tail": mv["tail"], "gmax_final": _f(fin.get("Gmax_full_over_dw")),
                "gmax_dev": row["gmax_dev"], "s1": _f(fin.get("stage1_rel_err_abs"))}
        v = verdicts(vals, proto)
        row.update({"parent_eta_final": mv["eta_final"], "parent_eta_dev": mv["eta_dev"],
                    "parent_rmse": mv["rmse"], "parent_tail": mv["tail"],
                    "parent_outcome": gates.get("outcome"),
                    "G_A_parent": v["G-A"]["pass"],
                    "G_N_eta_parent": v["G-N"]["criteria"][0]["pass"],
                    "G_F": v["G-F"]["pass"], "G_N_gmax": v["G-N"]["criteria"][1]["pass"],
                    "run_pass": v["run_pass"], "outcome": v["outcome"], "S1_pass": v["S1"]["pass"]})
        # a verdict computed on a non-finite metric is not a verdict
        if not np.isfinite(vals["gmax_final"]):
            anomalies.append("Gmax_full_over_dw not finite")
    except Exception as exc:  # noqa: BLE001
        anomalies.append(f"parent gates unavailable: {type(exc).__name__}: {exc}")
    row["target0929_pass"] = bool(abs(_f(fin.get("stage1_rel_err_signed"))) <= TARGET_0929)
    try:
        band = _jload(ref / "induced_band.json")
        dec = rows_for(_f(fin.get("e1_at_0")), band, float(band["g1"]))
        row.update({k: dec[k] for k in ("e_tilde", "band_lo", "band_hi", "learning_rel",
                                        "learning_rel_lo", "learning_rel_hi",
                                        "learning_contains_0", "inherited_rel",
                                        "inherited_rel_lo", "inherited_rel_hi",
                                        "inherited_contains_0", "e1_inside_sweep")})
        row["learning_rel_abs"] = abs(dec["learning_rel"])
    except Exception as exc:  # noqa: BLE001
        anomalies.append(f"induced band unavailable: {type(exc).__name__}: {exc}")
    # ---- the shared stage 2: frozen snapshot bit-identical to the parent's actor
    try:
        man = _jload(d / "manifest.json")
        parent = man.get("parent_checkpoint") or str(ref / "state_end_A.pt")
        fz = torch.load(d / "state_end_B.pt", map_location="cpu",
                        weights_only=False)["agent"].get("frozen")
        par = _parent_actor_sd(parent)
        same = fz is not None and set(fz) == set(par) and all(
            torch.equal(fz[k], par[k]) for k in par)
        row["frozen_bit_identical_to_parent"] = bool(same)
        row["parent_checkpoint"] = parent
        if not same:
            anomalies.append("frozen stage 2 differs from the parent actor (e~1 not shared)")
    except Exception as exc:  # noqa: BLE001
        row["frozen_bit_identical_to_parent"] = None
        anomalies.append(f"frozen-identity check failed: {type(exc).__name__}: {exc}")
    try:
        dt = _jload(d / "drift_test.json")
        row["drift_test_pass"] = bool(dt.get("pass"))
        if not row["drift_test_pass"]:
            anomalies.append("drift_test.json pass = false")
    except Exception as exc:  # noqa: BLE001
        row["drift_test_pass"] = None
        anomalies.append(f"drift_test.json unavailable: {type(exc).__name__}")
    # ---- within-run stability of e1_hat(0) over the last five exports
    try:
        e1s = []
        for u in LAST5:
            with np.load(d / "weights" / f"u{u:05d}.npz") as z:
                w = {k: z[k] for k in z.files}
            e1s.append(float(mean_effort_numpy(w, np.zeros((1, 2), dtype=np.float32),
                                               e_min=spec.e_min, e_max=spec.e_max)[0][0]))
        row["within_run_sd_e1_last5"] = float(np.std(e1s, ddof=1))
        row["within_run_range_e1_last5"] = float(max(e1s) - min(e1s))
        row["e1_last5_n"] = len(e1s)
        row["e1_reload_rel_diff"] = abs(e1s[-1] - _f(fin.get("e1_at_0"))) / _f(fin.get("e1_at_0"))
    except Exception as exc:  # noqa: BLE001
        anomalies.append(f"last-5 exports unavailable: {type(exc).__name__}: {exc}")
    # ---- optimisation, cost, rng
    try:
        row.update(opt_diagnostics(d, PHASE["stage1"], anomalies))
    except Exception as exc:  # noqa: BLE001
        anomalies.append(f"optimisation diagnostics unavailable: {type(exc).__name__}: {exc}")
    base = BASE_ARM["stage1"]
    row.update(rng_columns(d, ctx.run_dir("stage1", q, seed, base), arm == base))
    for k in ("e1_at_0", "stage1_rel_err_abs", "Gmax_full_over_dw"):
        if row.get(k) is None or not np.isfinite(_f(row.get(k))):
            anomalies.append(f"{k} missing or not finite")
    row["anomalies"] = "; ".join(anomalies)
    return row


def stage2_phase_state(arm: str) -> Tuple[str, str]:
    """(phase letter, end-of-phase state file) of a stage-2 arm."""
    return ("P", "state_end_P.pt") if arm == "A_detmean" else ("A", "state_end_A.pt")


def detmean_logged(d: Path) -> pd.DataFrame:
    """Per-update logged loss / FOC of an A_detmean run with rolling mean and SD (window 20).

    The logged ``loss`` and ``foc_abs_*`` are evaluated on that update's 512 fresh random
    exploring-start rows, so they are noisy; the rolling statistics are what the report shows.
    """
    up = pd.read_csv(Path(d) / "v2_updates.csv")
    cols = [c for c in ("update", "local", "loss", "grad_norm_pre_clip", "foc_abs_mean",
                        "foc_abs_max", "e0", "actor_lr") if c in up]
    df = up[cols].copy()
    for c in ("loss", "foc_abs_mean", "foc_abs_max"):
        df[f"{c}_roll_mean"] = df[c].rolling(LOGGED_WINDOW, min_periods=LOGGED_WINDOW).mean()
        df[f"{c}_roll_sd"] = df[c].rolling(LOGGED_WINDOW, min_periods=LOGGED_WINDOW).std()
    return df


def pathwise_trajectory(ctx: Ctx, q: int, seed: int, arm: str, spec: GameSpec, d: Path,
                        parent_path: Optional[str]) -> List[Dict[str, Any]]:
    """Offline fixed-grid objective at the parent state and at every weight export of a run."""
    es_bw = float(_protocol(ctx.protocol)["records"][str(q)]["protocol"]["es_bin_width"])
    cen = bin_centres(spec, es_bw)
    rows: List[Dict[str, Any]] = []
    if parent_path:
        rows.append({"q": q, "seed": seed, "arm": arm, "update": 1600, "local": 0,
                     **offline_objective(load_actor(parent_path), spec, cen)})
    for f in sorted((d / "weights").glob("u*.npz")):
        u = int(re.fullmatch(r"u(\d+)\.npz", f.name).group(1))
        if u <= 1600:
            continue
        rows.append({"q": q, "seed": seed, "arm": arm, "update": u, "local": u - 1600,
                     **offline_objective(load_actor(f), spec, cen)})
    return rows


def extract_stage2(ctx: Ctx, q: int, seed: int, arm: str) -> Dict[str, Any]:
    """Per-run metrics of one stage-2 run (Phase-A continuation, control or A_detmean)."""
    d = ctx.run_dir("stage2", q, seed, arm)
    row = base_row("stage2", q, seed, arm, d)
    anomalies: List[str] = []
    if not row["complete"]:
        row["anomalies"] = f"run {row['status']}: {row['status_info']}"
        return row
    phase, state_file = stage2_phase_state(arm)
    fv = _jload(d / "final_v2.json")
    fin, dev = fv[TIER_FINAL], fv[TIER_DEV]
    if row.get("manifest_dirty") is True:
        anomalies.append("manifest.json records dirty = true")
    row.update({k: fin.get(k) for k in S2_FINAL_KEYS if k != "valid"})
    row["valid_final"], row["valid_dev"] = bool(fin.get("valid")), bool(dev.get("valid"))
    row["eta_dev"] = _f(dev.get("eta_T_over_dw"))
    row["eta_dev_minus_final"] = row["eta_dev"] - _f(fin.get("eta_T_over_dw"))
    row["eta_dev_minus_final_abs"] = abs(row["eta_dev_minus_final"])
    spec = spec_for_q(ctx, q)
    proto = _protocol(ctx.protocol)
    vals = {"eta_final": _f(fin.get("eta_T_over_dw")), "eta_dev": row["eta_dev"],
            "rmse": _f(fin.get("stage2_rmse_pos_over_g2_0")),
            "tail": _f(fin.get("stage2_tail_mean_over_g2_0")),
            "gmax_final": 0.0, "gmax_dev": 0.0, "s1": 0.0}   # stage-1 parts are not evaluated
    v = verdicts(vals, proto)
    crit = {c["metric"]: c for c in v["G-A"]["criteria"]}
    row.update({"G_A": v["G-A"]["pass"], "G_N_eta": v["G-N"]["criteria"][0]["pass"],
                "gate_pass": bool(v["G-A"]["pass"] and v["G-N"]["criteria"][0]["pass"]),
                "G_A_eta_pass": crit["eta_T_over_dw"]["pass"],
                "G_A_rmse_pass": crit["stage2_rmse_pos_over_g2_0"]["pass"],
                "G_A_tail_pass": crit["stage2_tail_mean_over_g2_0"]["pass"]})
    # ---- location-free peak error (stage2_extra logic on the recovery arrays)
    try:
        with np.load(d / "final_final.npz") as z:
            arrays = {k: z[k] for k in z.files}
        ex = stage2_extra(SimpleNamespace(arrays=arrays, scalars=fin))
        row["stage2_peak_locfree_rel_err"] = ex["stage2_peak_locfree_rel_err"]
        row["stage2_peak_locfree_rel_err_abs"] = abs(ex["stage2_peak_locfree_rel_err"])
        row["stage2_peak_locfree_argmax_d"] = ex["stage2_peak_locfree_argmax_d"]
        z0 = int(np.nonzero(arrays["v_t2_d_grid"] == 0.0)[0][0])
        row["npz_alpha_d0"], row["npz_beta_d0"] = (float(arrays["v_t2_alpha"][z0]),
                                                   float(arrays["v_t2_beta"][z0]))
    except Exception as exc:  # noqa: BLE001
        arrays = {}
        anomalies.append(f"final_final.npz unavailable: {type(exc).__name__}: {exc}")
    # ---- smoothed game from the saved actor (conc_scale applied), cross-checked with the npz
    try:
        actor = load_actor(d / state_file)
        row["conc_scale_actor"] = float(actor.conc_scale)
        a, b = actor_alpha_beta(actor, spec, 2, np.zeros(1))
        sm = smoothed_record(float(a[0]), float(b[0]), spec, _f(fin.get("e2_at_0")))
        row.update(sm)
        row["alpha_d0"], row["beta_d0"] = float(a[0]), float(b[0])
        row["sigma2_0_reload"] = sigma_effort(float(a[0]), float(b[0]), spec)
        sg = _f(fin.get("sigma_effort_at_0_t2"))
        row["sigma2_0_reload_rel_diff"] = abs(row["sigma2_0_reload"] - sg) / sg
        if "npz_alpha_d0" in row:
            row["alpha_reload_rel_diff"] = abs(a[0] - row["npz_alpha_d0"]) / row["npz_alpha_d0"]
        if row["sigma2_0_reload_rel_diff"] > 1e-4:
            anomalies.append("reloaded sigma_2(0) differs from the run's own beta_fn by "
                             f"{row['sigma2_0_reload_rel_diff']:.3g} (relative)")
    except Exception as exc:  # noqa: BLE001
        anomalies.append(f"actor reload unavailable: {type(exc).__name__}: {exc}")
    # ---- optimisation, cost, rng
    try:
        row.update(opt_diagnostics(d, phase, anomalies))
    except Exception as exc:  # noqa: BLE001
        anomalies.append(f"optimisation diagnostics unavailable: {type(exc).__name__}: {exc}")
    base = BASE_ARM["stage2"] if arm != "A_detmean" else "A_ctrl200"
    row.update(rng_columns(d, ctx.run_dir("stage2", q, seed, base),
                           arm in (BASE_ARM["stage2"], "A_ctrl200")))
    # ---- method-5 pair: pathwise trajectory (offline objective), FOC, logged loss
    traj: List[Dict[str, Any]] = []
    if arm in ("A_detmean", "A_ctrl200"):
        try:
            man = _jload(d / "manifest.json")
            parent = man.get("parent_checkpoint")
            traj = pathwise_trajectory(ctx, q, seed, arm, spec, d, parent)
            row["J_offline_start"], row["J_offline_end"] = traj[0]["J"], traj[-1]["J"]
            row["foc_offline_end_mean"], row["foc_offline_end_max"] = (traj[-1]["foc_mean"],
                                                                        traj[-1]["foc_max"])
        except Exception as exc:  # noqa: BLE001
            anomalies.append(f"offline objective unavailable: {type(exc).__name__}: {exc}")
    if arm == "A_detmean":
        try:
            lg = detmean_logged(d)
            w = LOGGED_WINDOW
            row["loss_first20_mean"] = float(lg["loss"].iloc[:w].mean())
            row["loss_last20_mean"] = float(lg["loss"].iloc[-w:].mean())
            row["loss_last20_sd"] = float(lg["loss"].iloc[-w:].std())
            row["foc_logged_mean_last20"] = float(lg["foc_abs_mean"].iloc[-w:].mean())
            row["foc_logged_max_last20"] = float(lg["foc_abs_max"].iloc[-w:].max())
            row["foc_logged_mean_last1"] = float(lg["foc_abs_mean"].iloc[-1])
            row["foc_logged_max_last1"] = float(lg["foc_abs_max"].iloc[-1])
            row["gn_pre_clip_last20_mean"] = float(lg["grad_norm_pre_clip"].iloc[-w:].mean())
            pc = _jload(d / "phaseP_checks.json")
            row["phaseP_head_bit_identical"] = bool(pc.get("concentration_head_bit_identical"))
            row["phaseP_critic_bit_identical"] = bool(pc.get("critic_bit_identical"))
            row["phaseP_critic_adam_bit_identical"] = bool(
                pc.get("critic_adam_state_bit_identical"))
            if not all(pc.values()):
                anomalies.append(f"phaseP_checks.json has a false entry: {pc}")
            ck = pd.read_csv(d / "v2_checkpoints_P.csv")
            row["verifier_calls"] = int(len(ck))
            last = ck[ck["local"] == ck["local"].max()].iloc[-1]
            row["verifier_last_reason"] = str(last["reason"])
            row["verifier_last_local"] = int(last["local"])
        except Exception as exc:  # noqa: BLE001
            anomalies.append("A_detmean logged diagnostics unavailable: "
                             f"{type(exc).__name__}: {exc}")
    for k in ("stage2_peak_rel_err_signed", "eta_T_over_dw", "stage2_rmse_pos_over_g2_0"):
        if row.get(k) is None or not np.isfinite(_f(row.get(k))):
            anomalies.append(f"{k} missing or not finite")
    if not row["valid_final"] or not row["valid_dev"]:
        anomalies.append("verifier reported valid = false")
    row["anomalies"] = "; ".join(anomalies)
    row["_traj"] = traj
    return row


def parent_u1600_rows(ctx: Ctx, qs: Sequence[int], seeds: Sequence[int]) -> List[Dict[str, Any]]:
    """Pseudo-arm ``parent_u1600``: end-of-A values of the rehearsal (the baseline u1600 state)."""
    rows = []
    for q in qs:
        for s in seeds:
            ref = ctx.ref_dir(q, s)
            row: Dict[str, Any] = {"wave": "stage2", "q": q, "seed": s, "arm": "parent_u1600",
                                   "run_dir": _rel(ref), "status": "reference",
                                   "exit_code": 0, "status_info": "", "complete": False,
                                   "reference": True}
            try:
                g = _jload(ref / "gates.json")
                ea = g["reported"]["end_of_A"]
                fin, dev = ea["final"], ea["development"]
                row.update({k: fin.get(k) for k in S2_FINAL_KEYS if k != "valid"})
                row["stage2_peak_locfree_rel_err"] = fin.get("stage2_peak_locfree_rel_err")
                row["stage2_peak_locfree_rel_err_abs"] = abs(
                    _f(fin.get("stage2_peak_locfree_rel_err")))
                row["stage2_peak_locfree_argmax_d"] = fin.get("stage2_peak_locfree_argmax_d")
                row["eta_dev"] = _f(dev.get("eta_T_over_dw"))
                row["eta_dev_minus_final_abs"] = abs(row["eta_dev"] - _f(fin.get("eta_T_over_dw")))
                sm = ea.get("smoothed_game") or {}
                g20 = _f(fin.get("g2_at_0"))
                row["smoothed_e_pred_0"] = _f(sm.get("smoothed_e_pred_0"))
                row["smoothed_pred_gap_d0"] = g20 - row["smoothed_e_pred_0"]
                row["smoothed_share_peak_gap_d0"] = _f(sm.get("smoothed_share_peak_gap_d0"))
                row["G_A"] = bool(g["G-A"]["pass"])
                row["G_N_eta"] = bool(g["G-N"]["criteria"][0]["pass"])
                row["gate_pass"] = row["G_A"] and row["G_N_eta"]
                row["complete"] = True
                row["anomalies"] = ""
            except Exception as exc:  # noqa: BLE001
                row["anomalies"] = f"rehearsal gates unavailable: {type(exc).__name__}: {exc}"
            rows.append(row)
    return rows


def _extract_task(args: Tuple[Ctx, str, int, int, str]) -> Dict[str, Any]:
    ctx, wave, q, seed, arm = args
    torch.set_num_threads(1)
    return (extract_stage1 if wave == "stage1" else extract_stage2)(ctx, q, seed, arm)


def plan(wave: str, qs: Sequence[int], seeds: Sequence[int], arms: Sequence[str]
         ) -> List[Tuple[str, int, int, str]]:
    """All planned runs of a wave, ordered q, seed, arm."""
    return [(wave, q, s, a) for q in qs for s in seeds for a in arms]


def extract_wave(ctx: Ctx, wave: str, qs: Sequence[int], seeds: Sequence[int],
                 arms: Sequence[str], workers: int = 1
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-run table of a wave (every planned run, in plan order) and the pathwise trajectories."""
    tasks = [(ctx, w, q, s, a) for (w, q, s, a) in plan(wave, qs, seeds, arms)]
    if workers > 1:
        with Pool(workers) as pool:
            rows = pool.map(_extract_task, tasks, chunksize=2)
    else:
        rows = [_extract_task(t) for t in tasks]
    traj: List[Dict[str, Any]] = []
    for r in rows:
        traj += r.pop("_traj", [])
    df = pd.DataFrame(rows)
    if wave == "stage2":
        ref = pd.DataFrame(parent_u1600_rows(ctx, qs, seeds))
        df = pd.concat([df, ref], ignore_index=True, sort=False)
    return df, pd.DataFrame(traj)



# --------------------------------------------------------------------------- tables
GATE_COL = {"stage1": "run_pass", "stage2": "gate_pass"}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _truthy(v: Any) -> bool:
    """True only for a real boolean True (NaN, None and strings are not)."""
    return isinstance(v, (bool, np.bool_)) and bool(v)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes of a per-run table (``complete`` as bool, rows in plan order)."""
    df = df.copy()
    df["complete"] = df["complete"].fillna(False).astype(bool)
    return df


SET_ASIDE_DIRS = ("crashed", "dirty_rerun")   # moved-aside originals of re-run runs (not analysed)


def set_aside_counts(root: Path, wave: str) -> Dict[Tuple[str, int], int]:
    """Moved-aside original runs per (arm, q): ``<wave>/<aside>/q<q>/seed<s>/<arm>/status.json``.

    These are originals that were moved away when a run was re-run into its original path; they
    are counted here and never analysed.
    """
    out: Dict[Tuple[str, int], int] = {}
    for aside in SET_ASIDE_DIRS:
        base = Path(root) / wave / aside
        if not base.exists():
            continue
        for st in sorted(base.rglob("status.json")):
            rel = st.relative_to(base).parts          # (q50, seed10504, B_batch, status.json)
            if len(rel) != 4 or not rel[0].startswith("q"):
                continue
            key = (rel[2], int(rel[0][1:]))
            out[key] = out.get(key, 0) + 1
    return out


def completeness_table(df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int],
                       seeds: Sequence[int], root: Path, wave: str) -> pd.DataFrame:
    """Runs done / failed / incomplete / missing per arm and q, plus the set-aside originals."""
    aside = set_aside_counts(root, wave)
    rows = []
    for arm in arms:
        for q in qs:
            g = df[(df.arm == arm) & (df.q == q)]
            st = g["status"].value_counts()
            rows.append({"wave": wave, "arm": arm, "q": q, "planned": len(seeds),
                         "done": int(st.get("done", 0)), "failed": int(st.get("failed", 0)),
                         "incomplete": int(st.get("incomplete", 0)),
                         "missing": int(len(seeds) - len(g) + st.get("missing", 0)),
                         "failed_seeds": " ".join(str(int(s)) for s in
                                                  g[g.status == "failed"]["seed"]),
                         "not_done_seeds": " ".join(str(int(s)) for s in
                                                    g[g.status != "done"]["seed"]),
                         "set_aside_originals": aside.get((arm, int(q)), 0)})
    return pd.DataFrame(rows)


def manifest_commit_table(df: pd.DataFrame) -> pd.DataFrame:
    """Runs per (commit, dirty flag) recorded in their ``manifest.json`` (analysed runs only)."""
    cols = ["manifest_commit", "manifest_dirty", "n_runs"]
    if "manifest_commit" not in df.columns:
        return pd.DataFrame(columns=cols)
    g = df[df["complete"] & df["manifest_commit"].notna()
           & (df["arm"] != "parent_u1600")]
    if g.empty:
        return pd.DataFrame(columns=cols)
    out = (g.groupby(["manifest_commit", "manifest_dirty"], dropna=False).size()
           .reset_index(name="n_runs"))
    return out.sort_values(["manifest_commit", "manifest_dirty"]).reset_index(drop=True)


def paired_seed_values(df: pd.DataFrame, arm: str, base: str, q: int, metric: str,
                       seeds: Sequence[int]) -> pd.DataFrame:
    """Seed-level arm / baseline values of the seeds with finite values in both complete runs."""
    if metric not in df.columns:
        return pd.DataFrame(columns=["seed", "arm_value", "base_value", "diff"])
    a = df[(df.arm == arm) & (df.q == q) & df.complete].set_index("seed")[metric]
    b = df[(df.arm == base) & (df.q == q) & df.complete].set_index("seed")[metric]
    rows = []
    for s in seeds:
        if s in a.index and s in b.index:
            x, y = pd.to_numeric(pd.Series([a.loc[s], b.loc[s]]), errors="coerce")
            if np.isfinite(x) and np.isfinite(y):
                rows.append({"seed": s, "arm_value": float(x), "base_value": float(y),
                             "diff": float(x - y)})
    return pd.DataFrame(rows, columns=["seed", "arm_value", "base_value", "diff"])


def paired_tables(df: pd.DataFrame, wave: str, comparisons: Sequence[Tuple[str, str, str]],
                  qs: Sequence[int], seeds: Sequence[int]
                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Paired-difference table (arm - baseline) and its seed-level long form.

    Args:
        df: Per-run table of the wave.
        wave: ``stage1`` or ``stage2``.
        comparisons: ``(arm, baseline, label)`` in table order.
        qs: q values.
        seeds: Development seeds.

    Returns:
        ``(summary, seed_level)``. The summary has one row per (comparison, q, metric) with
        n_pairs, mean, median, n_better, n_pos / n_neg / n_zero and the four CI bounds; metrics
        without any finite pair are omitted except the primary one.
    """
    rows, long_rows = [], []
    for arm, base, label in comparisons:
        for q in qs:
            for m, direction in METRICS[wave]:
                sv = paired_seed_values(df, arm, base, q, m, seeds)
                if sv.empty and m != PRIMARY[wave]:
                    continue
                rows.append({"wave": wave, "comparison": label, "arm": arm, "baseline": base,
                             "q": q, "metric": m,
                             "direction": "lower is better" if direction else "none",
                             "n_expected": len(seeds),
                             **paired_summary(sv["diff"].to_numpy(), direction)})
                for r in sv.itertuples(index=False):
                    long_rows.append({"wave": wave, "comparison": label, "arm": arm,
                                      "baseline": base, "q": q, "metric": m, "seed": r.seed,
                                      "arm_value": r.arm_value, "base_value": r.base_value,
                                      "diff": r.diff})
    return pd.DataFrame(rows), pd.DataFrame(long_rows)


def criterion(prim: pd.DataFrame, gates: pd.DataFrame, qs: Sequence[int],
              n_expected: int) -> Dict[str, Any]:
    """Pre-registered per-arm criterion (descriptive, not a gate); both parts separately.

    (a) The bootstrap CI of the mean paired difference of the primary metric lies entirely
    below 0 (improvement = decrease) at EVERY q; the test is strict (``ci_mean_hi < 0``).
    (b) No run that passed its gate under the baseline fails it under the arm. A baseline-passing
    pair whose arm run failed (crashed, evaluation error, gate not computable) counts as a
    violation; one whose arm run is missing or still running is ``pending``.

    Args:
        prim: Rows of the primary metric (columns ``q, n_pairs, mean, ci_mean_lo, ci_mean_hi``).
        gates: One row per (q, seed) with ``base_pass`` (bool), ``arm_status`` (``done`` /
            ``failed`` / ``incomplete`` / ``missing``) and ``arm_pass`` (bool or NaN).
        qs: q values.
        n_expected: Pairs per q when all seeds are complete.

    Returns:
        Flat dict with ``a_met``, ``a_complete``, ``b_status`` (``holds`` / ``violated`` /
        ``incomplete``), ``b_violations``, ``b_pending`` and ``overall`` (``met`` / ``not met`` /
        ``incomplete``).
    """
    out: Dict[str, Any] = {}
    flags, complete = [], []
    for q in qs:
        r = prim[prim.q == q]
        n = int(r["n_pairs"].iloc[0]) if len(r) else 0
        hi = float(r["ci_mean_hi"].iloc[0]) if n else float("nan")
        out[f"n_pairs_q{q}"] = n
        out[f"mean_q{q}"] = float(r["mean"].iloc[0]) if n else float("nan")
        out[f"ci_mean_lo_q{q}"] = float(r["ci_mean_lo"].iloc[0]) if n else float("nan")
        out[f"ci_mean_hi_q{q}"] = hi
        flag = bool(n > 0 and hi < 0.0)
        out[f"a_q{q}"] = flag
        flags.append(flag)
        complete.append(n == n_expected)
    out["a_met"] = bool(all(flags))
    out["a_complete"] = bool(all(complete))
    g = gates[gates["base_pass"].astype(bool)]
    arm_pass = g["arm_pass"].astype("object").map(_truthy)
    viol = g[((g["arm_status"] == "done") & (~arm_pass)) | (g["arm_status"] == "failed")]
    pend = g[g["arm_status"].isin(["missing", "incomplete"])]
    out["n_base_pass"] = int(len(g))
    out["b_violations"] = " ".join(f"q{int(r.q)}/{int(r.seed)}" for r in viol.itertuples())
    out["b_pending"] = " ".join(f"q{int(r.q)}/{int(r.seed)}" for r in pend.itertuples())
    out["b_n_violations"], out["b_n_pending"] = int(len(viol)), int(len(pend))
    out["b_status"] = "violated" if len(viol) else ("incomplete" if len(pend) else "holds")
    if out["b_status"] == "violated":
        out["overall"] = "not met"
    elif out["a_complete"] and out["b_status"] == "holds":
        out["overall"] = "met" if out["a_met"] else "not met"
    else:
        out["overall"] = "incomplete"
    return out


def gate_pairs(df: pd.DataFrame, wave: str, arm: str, base: str, qs: Sequence[int],
               seeds: Sequence[int]) -> pd.DataFrame:
    """Per (q, seed): did the baseline run pass its gate, and the arm run's status and verdict."""
    col = GATE_COL[wave]
    rows = []
    for q in qs:
        for s in seeds:
            b = df[(df.arm == base) & (df.q == q) & (df.seed == s)]
            a = df[(df.arm == arm) & (df.q == q) & (df.seed == s)]
            bp = bool(len(b) and bool(b["complete"].iloc[0]) and _truthy(b[col].iloc[0]))
            st = str(a["status"].iloc[0]) if len(a) else "missing"
            ap = np.nan
            if len(a) and st == "done":
                v = a[col].iloc[0]
                ap = bool(v) if isinstance(v, (bool, np.bool_)) else np.nan
            rows.append({"q": q, "seed": s, "base_pass": bp, "arm_status": st, "arm_pass": ap})
    return pd.DataFrame(rows)


def criterion_table(df: pd.DataFrame, wave: str, paired: pd.DataFrame,
                    comparisons: Sequence[Tuple[str, str, str]], qs: Sequence[int],
                    seeds: Sequence[int]) -> pd.DataFrame:
    """Per-arm criterion (both parts) for the comparisons ``vs baseline`` and, for the method-5
    ablation, ``ablation vs matched control`` (``A_detmean`` against ``A_ctrl200``: part (b) is
    then about the runs that passed their gate under the matched control).

    The ``comparison`` column tells the two kinds of rows apart.
    """
    rows = []
    for arm, base, label in comparisons:
        if label not in CRITERION_LABELS:
            continue
        prim = paired[(paired.arm == arm) & (paired.baseline == base)
                      & (paired.metric == PRIMARY[wave]) & (paired.comparison == label)]
        gp = gate_pairs(df, wave, arm, base, qs, seeds)
        rows.append({"wave": wave, "arm": arm, "baseline": base, "comparison": label,
                     "primary_metric": PRIMARY[wave], **criterion(prim, gp, qs, len(seeds))})
    return pd.DataFrame(rows)


CRITERION_LABELS = ("vs baseline", "ablation vs matched control")


def dispersion_table(df: pd.DataFrame, wave: str, arms: Sequence[str], base: str,
                     qs: Sequence[int], seeds: Sequence[int],
                     extra_pairs: Sequence[Tuple[str, str]] = ()) -> pd.DataFrame:
    """Across-seed SD of the signed error and of the effort at 0, ratio arm/baseline + CI.

    Stage 1: ``stage1_rel_err_signed`` and ``e1_at_0`` (pre-registered). Stage 2: the signed
    peak error and ``e2_at_0`` (descriptive extra). Pairs = seeds complete in both runs.
    ``extra_pairs`` are further (arm, baseline) comparisons (the method-5 ablation against its
    matched control); every row has its ``baseline`` column.
    """
    mets = (("stage1_rel_err_signed", "e1_at_0") if wave == "stage1"
            else ("stage2_peak_rel_err_signed", "e2_at_0"))
    pairs = [(a, base) for a in arms if a != base] + list(extra_pairs)
    rows = []
    for arm, bs in pairs:
        for q in qs:
            for m in mets:
                sv = paired_seed_values(df, arm, bs, q, m, seeds)
                rows.append({"wave": wave, "arm": arm, "baseline": bs, "q": q, "metric": m,
                             **sd_ratio(sv["arm_value"].to_numpy(), sv["base_value"].to_numpy())})
    return pd.DataFrame(rows)


OPT_COLS = ["kl_mean", "clip_frac_mean", "gn_actor_mean", "gn_actor_max", "n_actor_steps_total",
            "n_minibatch_steps_total", "n_epochs_run_mean", "adv_s1_std_mean", "adv_all_std_mean",
            "phase_wall_sec"]


def arm_summary(df: pd.DataFrame, wave: str, arms: Sequence[str], qs: Sequence[int]
                ) -> pd.DataFrame:
    """Absolute per-arm statistics (complete runs): medians / means and gate counts per q."""
    stage1 = wave == "stage1"
    cols = (["stage1_rel_err_abs", "stage1_rel_err_signed", "e1_at_0", "learning_rel",
             "Gmax_full_over_dw", "EXP_root_over_dw", "dReach_over_dw", "within_run_sd_e1_last5",
             "within_run_range_e1_last5"]
            if stage1 else
            ["stage2_peak_rel_err_abs", "stage2_peak_rel_err_signed",
             "stage2_peak_locfree_rel_err", "stage2_rmse_pos_over_g2_0",
             "stage2_tail_mean_over_g2_0", "eta_T_over_dw", "stage2_sym_err_max",
             "sigma_effort_at_0_t2", "e2_at_0", "smoothed_share_peak_gap_d0"])
    flags = (["S1_pass", "G_F", "G_N_gmax", "run_pass", "target0929_pass"] if stage1
             else ["G_A", "G_N_eta", "gate_pass"])
    rows = []
    for arm in arms:
        for q in qs:
            g = df[(df.arm == arm) & (df.q == q) & df.complete]
            r: Dict[str, Any] = {"wave": wave, "arm": arm, "q": q, "n_complete": int(len(g))}
            for c in cols:
                if c in g and len(g):
                    x = _num(g[c]).dropna()
                    r[f"median_{c}"] = float(x.median()) if len(x) else float("nan")
                    r[f"mean_{c}"] = float(x.mean()) if len(x) else float("nan")
                    r[f"sd_{c}"] = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
            for f_ in flags:
                if f_ in g and len(g):
                    r[f"n_{f_}"] = int(sum(1 for v in g[f_] if _truthy(v)))
            rows.append(r)
    return pd.DataFrame(rows)


def gate_counts_table(df: pd.DataFrame, wave: str, arms: Sequence[str], qs: Sequence[int]
                      ) -> pd.DataFrame:
    """Verdict counts per arm and q over the complete runs, with the location of the maxima.

    Stage 1: S1, G-F, G-N (Gmax part), the parent's G-A and G-N (eta part), the run outcome,
    the 0929 target, and (t*, d*) of Gmax_full (count of t* = 1 / 2, median d*). Stage 2: G-A and
    its three criteria, G-N (eta part), the gate of the criterion, the argmax d of the
    location-free peak error.
    """
    flags = (["S1_pass", "G_F", "G_N_gmax", "G_A_parent", "G_N_eta_parent", "run_pass",
              "target0929_pass", "valid_final", "valid_dev"] if wave == "stage1" else
             ["G_A", "G_A_eta_pass", "G_A_rmse_pass", "G_A_tail_pass", "G_N_eta", "gate_pass",
              "valid_final", "valid_dev"])
    rows = []
    for arm in arms:
        for q in qs:
            g = df[(df.arm == arm) & (df.q == q) & df.complete]
            r: Dict[str, Any] = {"wave": wave, "arm": arm, "q": q, "n_complete": int(len(g))}
            for f_ in flags:
                if f_ in g:
                    r[f"n_{f_}"] = int(sum(1 for v in g[f_] if _truthy(v)))
            if wave == "stage1" and len(g) and "Gmax_full_t" in g:
                t_ = _num(g["Gmax_full_t"])
                d_ = _num(g["Gmax_full_d"])
                r.update({"n_Gmax_at_t1": int((t_ == 1).sum()),
                          "n_Gmax_at_t2": int((t_ == 2).sum()),
                          "median_Gmax_d": float(d_.median()),
                          "min_Gmax_d": float(d_.min()), "max_Gmax_d": float(d_.max()),
                          "max_Gmax_full_over_dw": float(_num(g["Gmax_full_over_dw"]).max())})
            if wave == "stage2" and len(g) and "stage2_peak_locfree_argmax_d" in g:
                d_ = _num(g["stage2_peak_locfree_argmax_d"])
                r.update({"median_abs_locfree_argmax_d": float(d_.abs().median()),
                          "n_locfree_argmax_at_0": int((d_ == 0).sum()),
                          "max_eta_T_over_dw": float(_num(g["eta_T_over_dw"]).max())})
            rows.append(r)
    return pd.DataFrame(rows)


def opt_table(df: pd.DataFrame, wave: str, arms: Sequence[str], qs: Sequence[int]) -> pd.DataFrame:
    """Median over runs of the per-run optimisation diagnostics, per arm and q."""
    rows = []
    for arm in arms:
        for q in qs:
            g = df[(df.arm == arm) & (df.q == q) & df.complete]
            r: Dict[str, Any] = {"wave": wave, "arm": arm, "q": q, "n_complete": int(len(g))}
            for c in OPT_COLS:
                if c in g and len(g) and _num(g[c]).notna().any():
                    r[f"median_{c}"] = float(_num(g[c]).median())
            rows.append(r)
    return pd.DataFrame(rows)


def adv_ratio_table(df: pd.DataFrame, arms: Sequence[str], base: str, qs: Sequence[int],
                    seeds: Sequence[int]) -> pd.DataFrame:
    """Stage-1 advantage SD ratio arm / baseline per (q, seed) and its per-q summary.

    ``adv_s1_std`` is the population SD of the raw stage-1 advantages of an update; the table
    uses its mean over the 600 updates of a run. For ``B_expcont`` this is the pre-registered
    variance-reduction measurement.
    """
    rows = []
    for arm in arms:
        if arm == base:
            continue
        for q in qs:
            sv = paired_seed_values(df, arm, base, q, "adv_s1_std_mean", seeds)
            ratio = (sv["arm_value"] / sv["base_value"]).to_numpy()
            rows.append({"arm": arm, "baseline": base, "q": q, "n_pairs": int(ratio.size),
                         "mean_ratio": float(ratio.mean()) if ratio.size else float("nan"),
                         "median_ratio": float(np.median(ratio)) if ratio.size else float("nan"),
                         "min_ratio": float(ratio.min()) if ratio.size else float("nan"),
                         "max_ratio": float(ratio.max()) if ratio.size else float("nan"),
                         "mean_adv_s1_std_arm": float(sv["arm_value"].mean()) if len(sv)
                         else float("nan"),
                         "mean_adv_s1_std_base": float(sv["base_value"].mean()) if len(sv)
                         else float("nan")})
    return pd.DataFrame(rows)


def rng_table(df: pd.DataFrame, wave: str, arms: Sequence[str], base: str,
              qs: Sequence[int]) -> pd.DataFrame:
    """Per arm, q and stream: how many runs diverge at the first update, never, and when."""
    rows = []
    for arm in arms:
        if arm == base:
            continue
        for q in qs:
            g = df[(df.arm == arm) & (df.q == q) & df.complete]
            for s in STREAMS:
                col = f"rng_div_{s}"
                if col not in g or not len(g):
                    continue
                num = pd.to_numeric(g[col], errors="coerce")
                first = pd.to_numeric(g["phase_first_update"], errors="coerce")
                comparable = num.notna() | (g[col].astype(str) == "never")
                rows.append({"wave": wave, "arm": arm, "q": q, "stream": s,
                             "n_runs": int(comparable.sum()),
                             "n_at_first_update": int((num == first).sum()),
                             "n_never": int((g[col].astype(str) == "never").sum()),
                             "first_update_of_phase": (int(first.min()) if first.notna().any()
                                                       else None),
                             "median_first_divergence": float(num.median()) if num.notna().any()
                             else float("nan"),
                             "min_first_divergence": float(num.min()) if num.notna().any()
                             else float("nan"),
                             "max_first_divergence": float(num.max()) if num.notna().any()
                             else float("nan"),
                             "n_not_comparable": int(len(g) - comparable.sum())})
    return pd.DataFrame(rows)


def cost_table(df: pd.DataFrame, wave: str, arms: Sequence[str], base: str) -> pd.DataFrame:
    """Cost per run of every arm (mean over complete runs of both q): wall time, episodes, steps.

    Besides the phase wall time relative to the baseline (``phase_wall_ratio_vs_base``) the table
    has the wall time per update (``mean_wall_sec_per_update``) with its ratio to the baseline's
    (``wall_per_update_ratio_vs_base``: arms with a different number of updates are comparable
    only per update) and, for the method-5 pair, the ratio to the matched control ``A_ctrl200``
    (``wall_ratio_vs_matched_control``, same 200 updates). ``phase_episodes`` of ``A_detmean``
    are exploring-start rows (phase P draws no action and no shock: there are no rollouts).
    """
    rows = []
    base_wall = base_per_upd = ctrl_wall = None
    for arm in [base] + [a for a in arms if a != base]:
        g = df[(df.arm == arm) & df.complete]
        r: Dict[str, Any] = {"wave": wave, "arm": arm, "n_runs": int(len(g))}
        for c in ("phase_wall_sec", "total_wall_sec", "train_update_sec", "train_rollout_sec",
                  "phase_episodes", "phase_local_updates", "n_minibatch_steps_total",
                  "n_actor_steps_total"):
            r[f"mean_{c}"] = float(_num(g[c]).mean()) if c in g and len(g) else float("nan")
        r["mean_wall_sec_per_update"] = (r["mean_phase_wall_sec"] / r["mean_phase_local_updates"]
                                         if r["mean_phase_local_updates"] else float("nan"))
        if arm == base:
            base_wall, base_per_upd = r["mean_phase_wall_sec"], r["mean_wall_sec_per_update"]
        r["phase_wall_ratio_vs_base"] = (r["mean_phase_wall_sec"] / base_wall
                                         if base_wall else float("nan"))
        r["wall_per_update_ratio_vs_base"] = (r["mean_wall_sec_per_update"] / base_per_upd
                                              if base_per_upd else float("nan"))
        rows.append(r)
    out = pd.DataFrame(rows)
    order = {a: i for i, a in enumerate(arms)}
    out = out.sort_values("arm", key=lambda s: s.map(order)).reset_index(drop=True)
    if "A_ctrl200" in set(out["arm"]):
        ctrl_wall = float(out.loc[out["arm"] == "A_ctrl200", "mean_phase_wall_sec"].iloc[0])
    out["wall_ratio_vs_matched_control"] = [
        (w / ctrl_wall if (a in ("A_ctrl200", "A_detmean") and ctrl_wall) else float("nan"))
        for a, w in zip(out["arm"], out["mean_phase_wall_sec"])]
    return out


def stop_epoch_table(df: pd.DataFrame, wave: str, arms: Sequence[str], qs: Sequence[int]
                     ) -> pd.DataFrame:
    """Share of updates per stopping epoch (``n_epochs_run`` = 1..10) pooled over runs."""
    rows = []
    for arm in arms:
        for q in qs:
            g = df[(df.arm == arm) & (df.q == q) & df.complete]
            cols = [f"n_epochs_cnt_{k}" for k in STOP_EPOCHS]
            if not len(g) or any(c not in g for c in cols):
                continue
            if any(not _num(g[c]).notna().any() for c in cols):
                continue                  # the arm's runs have no n_epochs_run column (A_detmean)
            cnt = np.array([int(_num(g[c]).sum()) for c in cols], dtype=float)
            tot = cnt.sum()
            if tot == 0:
                continue
            for k, c in zip(STOP_EPOCHS, cnt):
                rows.append({"wave": wave, "arm": arm, "q": q, "n_runs": int(len(g)),
                             "n_updates": int(tot), "epochs_run": k, "count": int(c),
                             "share": float(c / tot) if tot else float("nan"),
                             "share_fewer_than_10": float(cnt[:-1].sum() / tot) if tot
                             else float("nan")})
    return pd.DataFrame(rows, columns=["wave", "arm", "q", "n_runs", "n_updates", "epochs_run",
                                       "count", "share", "share_fewer_than_10"])


def decomposition_table(df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int]) -> pd.DataFrame:
    """Stage-1 learning / inherited decomposition per arm and q (shared e~1 and band)."""
    rows = []
    for arm in arms:
        for q in qs:
            g = df[(df.arm == arm) & (df.q == q) & df.complete]
            if not len(g):
                continue
            def tr(c: str) -> int:
                return int(sum(1 for v in g[c] if _truthy(v)))
            rows.append({"arm": arm, "q": q, "n_complete": int(len(g)),
                         "median_learning_rel": float(_num(g["learning_rel"]).median()),
                         "median_learning_rel_abs": float(_num(g["learning_rel_abs"]).median()),
                         "n_learning_band_contains_0": tr("learning_contains_0"),
                         "median_inherited_rel": float(_num(g["inherited_rel"]).median()),
                         "n_inherited_band_contains_0": tr("inherited_contains_0"),
                         "n_e1_inside_sweep": tr("e1_inside_sweep"),
                         "median_e_tilde": float(_num(g["e_tilde"]).median()),
                         "median_band_width": float(
                             (_num(g["band_hi"]) - _num(g["band_lo"])).median()),
                         "n_frozen_bit_identical": tr("frozen_bit_identical_to_parent"),
                         "n_drift_test_pass": tr("drift_test_pass")})
    return pd.DataFrame(rows)


def annealing_table(df: pd.DataFrame, arms: Sequence[str], base: str, qs: Sequence[int],
                    seeds: Sequence[int]) -> pd.DataFrame:
    """Smoothing-predicted peak gap before (A_base) and after annealing next to the observed change.

    Per arm, q and quantity: paired median before / after and the paired difference
    (after - before: arm - A_base) with mean, median and the bootstrap CI of the mean. The
    Pearson correlation over seeds between the change of the predicted gap and the change of the
    observed gap is in :func:`annealing_corr_table` (one row per arm and q).
    """
    rows = []
    quantities = [("smoothed_pred_gap_d0", "predicted peak gap g2(0) - e_pred(0)"),
                  ("observed_gap_d0", "observed peak gap g2(0) - e2_hat(0)"),
                  ("e2_at_0", "observed e2_hat(0)"),
                  ("smoothed_e_pred_0", "predicted e_pred(0)"),
                  ("sigma_effort_at_0_t2", "sigma_2(0)"),
                  ("smoothed_share_peak_gap_d0", "share of the gap predicted")]
    for arm in arms:
        for q in qs:
            for col, label in quantities:
                sv = paired_seed_values(df, arm, base, q, col, seeds)
                lo, hi = boot_ci(sv["diff"].to_numpy(), "mean")
                rows.append({"arm": arm, "baseline": base, "q": q, "quantity": col,
                             "description": label, "n_pairs": int(len(sv)),
                             "median_before": float(sv["base_value"].median()) if len(sv)
                             else float("nan"),
                             "median_after": float(sv["arm_value"].median()) if len(sv)
                             else float("nan"),
                             "mean_change": float(sv["diff"].mean()) if len(sv) else float("nan"),
                             "median_change": float(sv["diff"].median()) if len(sv)
                             else float("nan"),
                             "ci_mean_lo": lo, "ci_mean_hi": hi})
    return pd.DataFrame(rows)


def annealing_corr_table(df: pd.DataFrame, arms: Sequence[str], base: str, qs: Sequence[int],
                         seeds: Sequence[int]) -> pd.DataFrame:
    """Pearson correlation over seeds of the change of the predicted and of the observed gap.

    One row per (arm, q): the changes are arm - baseline of ``smoothed_pred_gap_d0`` and of
    ``observed_gap_d0`` (descriptive; n = number of seeds complete in both runs; NaN for n < 3).
    """
    rows = []
    for arm in arms:
        for q in qs:
            dp = paired_seed_values(df, arm, base, q, "smoothed_pred_gap_d0", seeds)
            do = paired_seed_values(df, arm, base, q, "observed_gap_d0", seeds)
            both = dp.merge(do, on="seed", suffixes=("_pred", "_obs"))
            corr = (float(np.corrcoef(both["diff_pred"], both["diff_obs"])[0, 1])
                    if len(both) > 2 else float("nan"))
            rows.append({"arm": arm, "baseline": base, "q": q, "n_pairs": int(len(both)),
                         "corr_dchange_pred_gap_vs_dchange_obs_gap": corr})
    return pd.DataFrame(rows)


def detmean_summary(df: pd.DataFrame, traj: pd.DataFrame, qs: Sequence[int],
                    seeds: Sequence[int]) -> pd.DataFrame:
    """A_detmean vs its matched control: offline objective and FOC at the end, logged loss."""
    rows = []
    for q in qs:
        for arm in ("A_detmean", "A_ctrl200"):
            g = df[(df.arm == arm) & (df.q == q) & df.complete]
            r: Dict[str, Any] = {"q": q, "arm": arm, "n_complete": int(len(g))}
            for c in ("J_offline_start", "J_offline_end", "foc_offline_end_mean",
                      "foc_offline_end_max", "loss_first20_mean", "loss_last20_mean",
                      "loss_last20_sd", "foc_logged_mean_last20", "foc_logged_max_last20",
                      "foc_logged_mean_last1", "foc_logged_max_last1", "gn_pre_clip_last20_mean",
                      "stage2_peak_rel_err_signed", "stage2_peak_locfree_argmax_d"):
                if c in g and _num(g[c]).notna().any():
                    r[f"median_{c}"] = float(_num(g[c]).median())
            rows.append(r)
        for m in ("J_offline_end", "foc_offline_end_mean", "foc_offline_end_max"):
            sv = paired_seed_values(df, "A_detmean", "A_ctrl200", q, m, seeds)
            if len(sv):
                rows.append({"q": q, "arm": "A_detmean - A_ctrl200", "metric": m,
                             "n_complete": int(len(sv)),
                             **{k: v for k, v in paired_summary(sv["diff"].to_numpy(), None).items()
                                if k in ("mean", "median", "ci_mean_lo", "ci_mean_hi", "n_pos",
                                         "n_neg")}})
    return pd.DataFrame(rows)


def trajectory_summary(traj: pd.DataFrame) -> pd.DataFrame:
    """Median and IQR over seeds of the offline objective and FOC per (q, arm, update)."""
    if traj.empty:
        return pd.DataFrame()
    g = traj.groupby(["q", "arm", "update", "local"])
    out = g.agg(n=("J", "size"), J_median=("J", "median"),
                J_p25=("J", lambda x: float(np.percentile(x, 25))),
                J_p75=("J", lambda x: float(np.percentile(x, 75))),
                foc_mean_median=("foc_mean", "median"), foc_max_median=("foc_max", "median"),
                foc_mean_p25=("foc_mean", lambda x: float(np.percentile(x, 25))),
                foc_mean_p75=("foc_mean", lambda x: float(np.percentile(x, 75)))).reset_index()
    return out


def profile_table(ctx: Ctx, qs: Sequence[int], seeds: Sequence[int],
                  arms: Sequence[str] = ("A_detmean", "A_ctrl200", "A_base")) -> pd.DataFrame:
    """Profile |e2_hat(d) - e2*(d)| on the recovery grid (median / IQR over seeds); diagnostic."""
    rows = []
    for q in qs:
        for arm in arms:
            errs, e2s, g2 = [], [], None
            for s in seeds:
                d = ctx.run_dir("stage2", q, s, arm)
                if classify_run(d)["status"] != "done":
                    continue
                with np.load(d / "final_final.npz") as z:
                    D, e2, gg = z["recovery_d_grid"], z["recovery_e2"], z["recovery_g2"]
                errs.append(np.abs(e2 - gg))
                e2s.append(e2)
                g2 = (D, gg)
            if not errs:
                continue
            E, V = np.array(errs), np.array(e2s)
            for i, dd in enumerate(g2[0]):
                rows.append({"q": q, "arm": arm, "d": float(dd), "n": int(E.shape[0]),
                             "e2_star": float(g2[1][i]), "e2_hat_median": float(np.median(V[:, i])),
                             "abs_err_median": float(np.median(E[:, i])),
                             "abs_err_p25": float(np.percentile(E[:, i], 25)),
                             "abs_err_p75": float(np.percentile(E[:, i], 75))})
    return pd.DataFrame(rows)


def logged_table(ctx: Ctx, qs: Sequence[int], seeds: Sequence[int]) -> pd.DataFrame:
    """Logged per-update loss / FOC of the A_detmean runs with rolling mean and SD (window 20)."""
    frames = []
    for q in qs:
        for s in seeds:
            d = ctx.run_dir("stage2", q, s, "A_detmean")
            if classify_run(d)["status"] != "done":
                continue
            lg = detmean_logged(d)
            lg.insert(0, "seed", s)
            lg.insert(0, "q", q)
            frames.append(lg)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()



# --------------------------------------------------------------------------- figures
ARM_STYLE = {"marker": ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*"],
             "color": ["#4c4c4c", "#1b6ca8", "#d1495b", "#2e8b57", "#9467bd", "#c7861a",
                       "#17a2b8", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]}


def _mpl():
    """matplotlib (Agg, type-42 fonts, plain style) imported lazily."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
                                "axes.spines.top": False, "axes.spines.right": False,
                                "axes.titlesize": 9, "figure.dpi": 100})
    return plt


def save_fig(fig: Any, stem: Path) -> List[str]:
    """Write ``<stem>.png`` and ``<stem>.pdf``; returns the two file names."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(stem) + ".png", dpi=130, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return [stem.name + ".png", stem.name + ".pdf"]


def _g(x: float) -> str:
    """Compact number format used on figures and in the reports (4 significant digits)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.4g}"


def fig_paired_dots(stem: Path, seed_df: pd.DataFrame, summ: pd.DataFrame, qs: Sequence[int],
                    title: str, ylabel: str) -> List[str]:
    """Seed-level paired differences per q with the mean and its bootstrap CI."""
    plt = _mpl()
    fig, axes = plt.subplots(1, len(qs), figsize=(3.8 * len(qs), 3.3), squeeze=False)
    for ax, q in zip(axes[0], qs):
        sv = seed_df[seed_df.q == q].sort_values("seed")
        r = summ[summ.q == q]
        x = np.arange(len(sv))
        ax.axhline(0.0, color="0.5", lw=0.8)
        if len(sv):
            ax.plot(x, sv["diff"], "o", color="#1b6ca8", ms=5, label="seed pair")
        xm = len(sv) + 0.8
        ax.set_xticks(list(x) + [xm])
        ax.set_xticklabels([str(int(s) - 10500) for s in sv["seed"]] + ["mean"], fontsize=7)
        if len(r) and int(r["n_pairs"].iloc[0]):
            m, lo, hi = (float(r["mean"].iloc[0]), float(r["ci_mean_lo"].iloc[0]),
                         float(r["ci_mean_hi"].iloc[0]))
            ax.errorbar([xm], [m], yerr=[[m - lo], [hi - m]], fmt="D", color="#d1495b", ms=6,
                        capsize=4, label="mean, 95% CI")
            ax.plot([xm - 0.3, xm + 0.3], [float(r["median"].iloc[0])] * 2, "-", color="k", lw=2,
                    label="median")
            ax.set_title(f"q = {q}: mean {_g(m)} [{_g(lo)}, {_g(hi)}]\n"
                         f"n = {int(r['n_pairs'].iloc[0])}", fontsize=8)
        else:
            ax.set_title(f"q = {q}: no complete pair", fontsize=8)
        ax.set_xlabel("seed (10500 + n)")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return save_fig(fig, stem)


def fig_overview(stem: Path, prim: pd.DataFrame, arms: Sequence[str], qs: Sequence[int],
                 title: str, xlabel: str) -> List[str]:
    """Primary-metric paired difference per arm and q with the bootstrap CI of the mean."""
    plt = _mpl()
    fig, axes = plt.subplots(1, len(qs), figsize=(4.2 * len(qs), 0.45 * len(arms) + 1.6),
                             sharey=True, squeeze=False)
    for ax, q in zip(axes[0], qs):
        ax.axvline(0.0, color="0.5", lw=0.8)
        for i, arm in enumerate(arms):
            r = prim[(prim.arm == arm) & (prim.q == q)]
            if not len(r) or int(r["n_pairs"].iloc[0]) == 0:
                continue
            m, lo, hi = (float(r["mean"].iloc[0]), float(r["ci_mean_lo"].iloc[0]),
                         float(r["ci_mean_hi"].iloc[0]))
            ax.errorbar([m], [i], xerr=[[m - lo], [hi - m]], fmt="D", color="#1b6ca8", ms=5,
                        capsize=3)
            ax.plot([float(r["median"].iloc[0])], [i], "|", color="k", ms=10, mew=2)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels(arms)
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_title(f"q = {q}")
        ax.set_xlabel(xlabel)
    fig.suptitle(title + "  (diamond + bar: mean, 95% CI; tick: median)", fontsize=9)
    fig.tight_layout()
    return save_fig(fig, stem)


def fig_stop_epochs(stem: Path, stop: pd.DataFrame, arm: str, qs: Sequence[int], title: str
                    ) -> List[str]:
    """Histogram of the stopping epoch (epochs run per update) of a target-KL arm, per q."""
    plt = _mpl()
    fig, axes = plt.subplots(1, len(qs), figsize=(3.8 * len(qs), 3.0), sharey=True,
                             squeeze=False)
    for ax, q in zip(axes[0], qs):
        r = stop[(stop.arm == arm) & (stop.q == q)].sort_values("epochs_run")
        if len(r):
            ax.bar(r["epochs_run"], r["share"], color="#1b6ca8", edgecolor="k", lw=0.5)
            ax.set_title(f"q = {q}: {100 * float(r['share_fewer_than_10'].iloc[0]):.1f}% of "
                         f"{int(r['n_updates'].iloc[0])} updates ran < 10 epochs", fontsize=8)
        ax.set_xticks(list(STOP_EPOCHS))
        ax.set_xlabel("epochs run in the update")
    axes[0][0].set_ylabel("share of updates")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return save_fig(fig, stem)


def fig_anneal(stem: Path, df: pd.DataFrame, arms: Sequence[str], base: str, qs: Sequence[int],
               seeds: Sequence[int]) -> List[str]:
    """Smoothing-predicted vs observed peak gap (levels) and their changes under annealing."""
    plt = _mpl()
    fig, axes = plt.subplots(len(qs), 2, figsize=(7.6, 3.4 * len(qs)), squeeze=False)
    names = [base] + list(arms)
    for i, q in enumerate(qs):
        ax1, ax2 = axes[i]
        lim = []
        for j, arm in enumerate(names):
            g = df[(df.arm == arm) & (df.q == q) & df.complete]
            x, y = _num(g["smoothed_pred_gap_d0"]), _num(g["observed_gap_d0"])
            ax1.plot(x, y, ARM_STYLE["marker"][j], color=ARM_STYLE["color"][j], ms=5,
                     mfc="none" if arm == base else ARM_STYLE["color"][j], label=arm)
            lim += list(x.dropna()) + list(y.dropna())
        if lim:
            hi = max(lim) * 1.05
            ax1.plot([0, hi], [0, hi], color="0.6", lw=0.8, ls="--")
            ax1.text(hi * 0.98, hi * 0.9, "y = x", ha="right", color="0.4", fontsize=7)
        ax1.set_xlabel("predicted gap g2(0) - e_pred(0)")
        ax1.set_ylabel("observed gap g2(0) - e2_hat(0)")
        ax1.set_title(f"q = {q}: levels at the end of the phase", fontsize=8)
        ax1.legend(frameon=False, fontsize=7)
        for j, arm in enumerate(arms, start=1):
            dp = paired_seed_values(df, arm, base, q, "smoothed_pred_gap_d0", seeds)
            do = paired_seed_values(df, arm, base, q, "observed_gap_d0", seeds)
            m = dp.merge(do, on="seed", suffixes=("_p", "_o"))
            ax2.plot(m["diff_p"], m["diff_o"], ARM_STYLE["marker"][j],
                     color=ARM_STYLE["color"][j], ms=5, label=f"{arm} - {base}")
        ax2.axhline(0, color="0.6", lw=0.8)
        ax2.axvline(0, color="0.6", lw=0.8)
        ax2.set_xlabel("change of predicted gap (arm - A_base)")
        ax2.set_ylabel("change of observed gap (arm - A_base)")
        ax2.set_title(f"q = {q}: paired changes, one point per seed", fontsize=8)
        ax2.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return save_fig(fig, stem)


def fig_traj(stem: Path, traj: pd.DataFrame, value: str, ylabel: str, qs: Sequence[int],
             title: str, logy: bool = False) -> List[str]:
    """Median over seeds (band: IQR) of a per-export quantity vs the global update, per arm."""
    plt = _mpl()
    fig, axes = plt.subplots(1, len(qs), figsize=(4.2 * len(qs), 3.3), squeeze=False)
    sty = {"A_detmean": ("#d1495b", "-", "A_detmean"), "A_ctrl200": ("#1b6ca8", "--", "A_ctrl200")}
    lo_c, hi_c = {"J_median": ("J_p25", "J_p75"),
                  "foc_mean_median": ("foc_mean_p25", "foc_mean_p75")}.get(value, (None, None))
    for ax, q in zip(axes[0], qs):
        for arm, (c, ls, lab) in sty.items():
            r = traj[(traj.q == q) & (traj.arm == arm)].sort_values("update")
            if not len(r):
                continue
            ax.plot(r["update"], r[value], ls, color=c, lw=1.6, marker="o", ms=3, label=lab)
            if lo_c:
                ax.fill_between(r["update"], r[lo_c], r[hi_c], color=c, alpha=0.15, lw=0)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("global update")
        ax.set_ylabel(ylabel)
        ax.set_title(f"q = {q}")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return save_fig(fig, stem)


def fig_logged(stem: Path, logged: pd.DataFrame, qs: Sequence[int], title: str) -> List[str]:
    """Logged A_detmean loss (noisy) and its rolling mean / SD (window 20), per q."""
    plt = _mpl()
    fig, axes = plt.subplots(1, len(qs), figsize=(4.4 * len(qs), 3.3), squeeze=False)
    for ax, q in zip(axes[0], qs):
        g = logged[logged.q == q]
        for s, gs in g.groupby("seed"):
            ax.plot(gs["update"], gs["loss"], color="0.8", lw=0.5)
        if len(g):
            med = g.groupby("update")["loss_roll_mean"].median()
            sd = g.groupby("update")["loss_roll_sd"].median()
            ax.plot(med.index, med.values, color="#d1495b", lw=1.8,
                    label="rolling mean (median of seeds)")
            ax.fill_between(med.index, med.values - sd.values, med.values + sd.values,
                            color="#d1495b", alpha=0.15, lw=0, label="+/- rolling SD (median)")
        ax.set_xlabel("global update")
        ax.set_ylabel("logged loss = -mean R (512 fresh rows)")
        ax.set_title(f"q = {q}: grey = single logged losses")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return save_fig(fig, stem)


def fig_profile(stem: Path, prof: pd.DataFrame, qs: Sequence[int], title: str) -> List[str]:
    """e2_hat(d) against e2*(d) (top) and |e2_hat - e2*| (bottom) on the recovery grid."""
    plt = _mpl()
    fig, axes = plt.subplots(2, len(qs), figsize=(4.6 * len(qs), 5.4), squeeze=False,
                             sharex="col")
    sty = {"A_base": ("#4c4c4c", ":"), "A_ctrl200": ("#1b6ca8", "--"),
           "A_detmean": ("#d1495b", "-")}
    for k, q in enumerate(qs):
        top, bot = axes[0][k], axes[1][k]
        p = prof[prof.q == q]
        if len(p):
            star = p[p.arm == p.arm.iloc[0]].sort_values("d")
            top.plot(star["d"], star["e2_star"], color="k", lw=1.0, label="e2*(d)")
        for arm, (c, ls) in sty.items():
            r = p[p.arm == arm].sort_values("d")
            if not len(r):
                continue
            top.plot(r["d"], r["e2_hat_median"], ls, color=c, lw=1.5, label=f"{arm} (median)")
            bot.plot(r["d"], r["abs_err_median"], ls, color=c, lw=1.5, label=arm)
            bot.fill_between(r["d"], r["abs_err_p25"], r["abs_err_p75"], color=c, alpha=0.1, lw=0)
        top.set_ylabel("stage-2 effort")
        top.set_title(f"q = {q}")
        top.legend(frameon=False, fontsize=7)
        bot.set_ylabel("|e2_hat(d) - e2*(d)|")
        bot.set_xlabel("d (stage-2 gap)")
        if len(p):
            top.set_xlim(-2.6 * q, 2.6 * q)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return save_fig(fig, stem)



# --------------------------------------------------------------------------- checks / records
CHECK_FILES = {"parents_A": "parents_A_checks.json", "stage1_base": "stage1_base_checks.json",
               "stage2_base": "stage2_base_checks.json", "C-R1": "v11_reproduction_checks.json"}


def checks_tables(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summary of the reproducibility checks (one row per file) and their per-run results.

    A check file that does not exist yet is listed as ``not present``.
    """
    srows, rrows = [], []
    for name, fn in CHECK_FILES.items():
        p = Path(root) / fn
        if not p.exists():
            srows.append({"check": name, "file": _rel(p), "present": False})
            continue
        j = json.loads(p.read_text())
        fd = j.get("first_differences") or []
        snap = [(r["q"], r["seed"], v) for r in j.get("runs", [])
                for k, v in (r.get("info") or {}).items() if k.endswith("snapshot_refreshes")]
        srows.append({
            "check": name, "file": _rel(p), "present": True, "tool": j.get("tool"),
            "mode": j.get("mode"), "arm": j.get("arm"), "n": j.get("n"),
            "n_identical": j.get("n_identical"), "ALL": j.get("ALL"),
            "failing_fields": " ".join(j.get("failing_fields") or []),
            "first_difference": (f"q{fd[0]['q']}/{fd[0]['seed']} {fd[0]['field']}: "
                                 f"{fd[0]['path']} ref={fd[0]['ref']} new={fd[0]['new']}"
                                 if fd else ""),
            "n_snapshot_refresh_counter_differs": int(sum(1 for _, _, v in snap
                                                           if not v.get("equal", True))),
            "window": " ".join(str(x) for x in (j.get("window") or []))})
        for r in j.get("runs", []):
            rrows.append({"check": name, "q": r["q"], "seed": r["seed"], "ALL": r.get("ALL"),
                          "first_difference_field": (r.get("first_difference")
                                                     or {}).get("field", ""),
                          "failing_fields": " ".join(k for k, v in (r.get("fields") or {}).items()
                                                     if not v)})
    return (pd.DataFrame(srows),
            pd.DataFrame(rrows, columns=["check", "q", "seed", "ALL", "first_difference_field",
                                         "failing_fields"]))


def launch_record_table(root: Path, waves: Sequence[str]) -> pd.DataFrame:
    """Every launch record of each wave: planned / finished / return codes (read only)."""
    rows = []
    for w in waves:
        recs = sorted((Path(root) / w).glob("launch_*.json"))
        if not recs:
            rows.append({"wave": w, "record": "none"})
        for rec in recs:
            rows.append(_launch_row(w, rec))
    return pd.DataFrame(rows)


def _launch_row(w: str, rec: Path) -> Dict[str, Any]:
    """One row of the launch-record table."""
    j = json.loads(rec.read_text())
    rc = [r.get("returncode") for r in j.get("runs", [])]
    return {"wave": w, "record": _rel(rec), "state": j.get("state"),
            "n_planned": j.get("n_planned"), "n_finished": len(rc),
            "n_returncode_0": int(sum(1 for c in rc if c == 0)),
            "n_returncode_nonzero": int(sum(1 for c in rc if c != 0)),
            "workers": j.get("workers"), "head": str(j.get("head", ""))[:7],
            "code_commit": str(j.get("code_commit") or "")[:7], "nproc": j.get("nproc"),
            "loadavg_at_start": " ".join(f"{x:.1f}" for x in j.get("loadavg_at_start", []))}


def protocol_thresholds(path: Path = PROTOCOL) -> pd.DataFrame:
    """Gate thresholds read from the locked protocol (the values the verdicts use)."""
    proto = json.loads(Path(path).read_text())
    rows = [{"gate": blk, "metric": c["metric"], "threshold": c["threshold"]}
            for blk in ("G-A", "G-F", "G-N") for c in proto["gates"][blk]["all_must_hold"]]
    rows.append({"gate": "S1", "metric": "stage1_rel_err_abs",
                 "threshold": proto["secondary"]["S1"]["criterion"]["threshold"]})
    rows.append({"gate": "0929 target (reported only)", "metric": "|stage1_rel_err_signed|",
                 "threshold": TARGET_0929})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- markdown
def fmt_cell(v: Any) -> str:
    """Cell text: numbers to 4 significant digits, booleans / strings as computed."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return _g(float(v))
    return str(v).replace("|", "\\|").replace("\n", " ")


def md_table(df: pd.DataFrame) -> str:
    """GitHub-markdown table of a display frame."""
    if df.empty:
        return "_(no rows)_"
    head = "| " + " | ".join(str(c).replace("|", "\\|") for c in df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt_cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def ci_str(lo: Any, hi: Any) -> str:
    """``[lo, hi]`` with 4 significant digits."""
    return f"[{_g(lo)}, {_g(hi)}]"


class Doc:
    """Markdown report builder; every table is saved as a CSV that the report cites."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.lines: List[str] = []

    def h(self, n: int, text: str) -> None:
        """Heading of level ``n``."""
        self.lines += ["", "#" * n + " " + text, ""]

    def p(self, text: str) -> None:
        """Paragraph."""
        self.lines += [text, ""]

    def table(self, csv: pd.DataFrame, name: str, disp: Optional[pd.DataFrame] = None,
              note: str = "") -> None:
        """Save ``csv`` as ``<out>/<name>.csv`` and render ``disp`` (default: ``csv``)."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{name}.csv"
        csv.to_csv(path, index=False)
        self.lines += [md_table(csv if disp is None else disp), ""]
        self.lines += [f"Source: `{_rel(path)}`" + (f" ({note})" if note else "") + ".", ""]

    def figures(self, items: Sequence[Tuple[str, Sequence[str]]], fig_dir: Path,
                reports_dir: Path) -> None:
        """Image links (png) with a pdf link, relative to the report directory."""
        for cap, files in items:
            rel = os.path.relpath(str(fig_dir), str(reports_dir))
            png = [f for f in files if f.endswith(".png")]
            pdf = [f for f in files if f.endswith(".pdf")]
            if png:
                self.lines += [f"![{cap}]({rel}/{png[0]})", ""]
                self.lines += [f"{cap} (`{_rel(Path(fig_dir) / png[0])}`"
                               + (f", `{_rel(Path(fig_dir) / pdf[0])}`" if pdf else "") + ").", ""]

    def text(self) -> str:
        """The assembled markdown."""
        return "\n".join(self.lines).strip() + "\n"


def method_of(arm: str) -> str:
    """Method label (``1 polish`` ... ``6 expected continuation``) of an arm, or ``baseline``."""
    for m, d in METHOD_ARMS.items():
        if arm in d["stage1"] or arm in d["stage2"]:
            return m
    return "baseline" if arm in BASE_ARM.values() else ""


def definition_of(wave: str, arm: str) -> str:
    """One-line definition of an arm from the launcher's arm table."""
    return (STAGE1_ARMS if wave == "stage1" else STAGE2_ARMS)[arm]["definition"]


def repro_cmd(sub: str, args: argparse.Namespace) -> str:
    """The exact command that regenerates the CSVs, figures and reports of a sub-command."""
    parts = ["python tools/v2/refine_analysis.py", sub, f"--root {_rel(args.root)}",
             f"--out {_rel(args.out)}", f"--reports {_rel(args.reports)}",
             f"--figures {_rel(args.figures)}"]
    if Path(args.ref_root).resolve() != Path(REHEARSAL).resolve():
        parts.append(f"--ref-root {args.ref_root}")
    if getattr(args, "qs", None) and tuple(args.qs) != QS:
        parts.append("--qs " + " ".join(str(q) for q in args.qs))
    if getattr(args, "seeds", None) and tuple(args.seeds) != SEEDS:
        parts.append("--seeds " + " ".join(str(s) for s in args.seeds))
    if getattr(args, "arms", None):
        parts.append("--arms " + " ".join(args.arms))
    return " ".join(parts)



# --------------------------------------------------------------------------- wave computation
@dataclass
class WaveResult:
    """All tables of one wave (the per-run table, the paired tables, the criterion, ...)."""

    wave: str
    arms: List[str]
    base: str
    qs: Tuple[int, ...]
    seeds: Tuple[int, ...]
    df: pd.DataFrame
    comps: List[Tuple[str, str, str]]
    tabs: Dict[str, pd.DataFrame]
    figs: Dict[str, List[Tuple[str, List[str]]]]


def wave_arms(wave: str, subset: Optional[Sequence[str]]) -> List[str]:
    """Arms of a wave in table order (optionally a subset; the baseline is always included)."""
    table = list(STAGE1_ARMS if wave == "stage1" else STAGE2_ARMS)
    if not subset:
        return table
    keep = set(subset) | {BASE_ARM[wave]}
    return [a for a in table if a in keep]


def wave_comparisons(wave: str, arms: Sequence[str]) -> List[Tuple[str, str, str]]:
    """(arm, baseline, label) in table order; stage 2 adds the method-5 comparisons."""
    base = BASE_ARM[wave]
    comps = [(a, base, "vs baseline") for a in arms if a != base]
    if wave == "stage2":
        if base in arms:
            comps.append((base, "parent_u1600",
                          "sanity: A_base end state vs rehearsal end-of-A"))
        for a in ("A_ctrl200", "A_detmean"):
            if a in arms:
                comps.append((a, "parent_u1600", "vs parent u1600 candidate"))
        if "A_detmean" in arms and "A_ctrl200" in arms:
            comps.append(("A_detmean", "A_ctrl200", "ablation vs matched control"))
    return comps


def compute_wave(wave: str, args: argparse.Namespace) -> WaveResult:
    """Extract every planned run of a wave and compute all pre-registered tables.

    All CSVs are written under ``args.out``; figures are made by :func:`make_figures`.
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    qs, seeds = tuple(args.qs), tuple(args.seeds)
    arms = wave_arms(wave, args.arms)
    base = BASE_ARM[wave]
    ctx = Ctx(root=str(args.root), ref_root=str(args.ref_root))
    df, traj = extract_wave(ctx, wave, qs, seeds, arms, workers=args.workers)
    df = prepare(df)
    comps = wave_comparisons(wave, arms)
    paired, seed_level = paired_tables(df, wave, comps, qs, seeds)
    tabs: Dict[str, pd.DataFrame] = {
        "per_run": df, "paired": paired, "paired_seed_level": seed_level,
        "completeness": completeness_table(df, arms, qs, seeds, Path(args.root), wave),
        "criterion": criterion_table(df, wave, paired, comps, qs, seeds),
        "dispersion": dispersion_table(
            df, wave, arms, base, qs, seeds,
            extra_pairs=[(a, b) for a, b, lab in comps if lab == "ablation vs matched control"]),
        "manifest_commits": manifest_commit_table(df),
        "arm_summary": arm_summary(df, wave, arms + (["parent_u1600"] if wave == "stage2" else []),
                                   qs),
        "gate_counts": gate_counts_table(df, wave, arms, qs),
        "optimisation": opt_table(df, wave, arms, qs),
        "rng_divergence": rng_table(df, wave, arms, base, qs),
        "cost": cost_table(df, wave, arms, base),
        "stop_epoch": stop_epoch_table(df, wave, arms, qs),
    }
    if wave == "stage1":
        tabs["decomposition"] = decomposition_table(df, arms, qs)
        tabs["adv_ratio"] = adv_ratio_table(df, arms, base, qs, seeds)
    else:
        anneal_arms = [a for a in ("A_anneal2", "A_anneal4") if a in arms]
        tabs["annealing"] = annealing_table(df, anneal_arms, base, qs, seeds)
        tabs["annealing_corr"] = annealing_corr_table(df, anneal_arms, base, qs, seeds)
        if "A_detmean" in arms:
            tabs["detmean_summary"] = detmean_summary(df, traj, qs, seeds)
            tabs["detmean_trajectory_per_run"] = traj
            tabs["detmean_trajectory"] = trajectory_summary(traj)
            tabs["detmean_logged"] = logged_table(ctx, qs, seeds)
            tabs["detmean_profile"] = profile_table(ctx, qs, seeds)
    for name, t in tabs.items():
        t.to_csv(out / f"{wave}_{name}.csv", index=False)
    return WaveResult(wave, arms, base, qs, seeds, df, comps, tabs, {})


def make_figures(wr: WaveResult, fig_dir: Path) -> None:
    """All figures of a wave (png + pdf); the registry ``wr.figs`` maps an arm to its figures."""
    wave, base, qs = wr.wave, wr.base, wr.qs
    t = wr.tabs
    prim = PRIMARY[wave]
    ylab = ("paired diff. of |stage-1 error|" if wave == "stage1"
            else "paired diff. of |signed peak error|")
    for arm, bs, label in wr.comps:
        if label not in ("vs baseline", "ablation vs matched control"):
            continue
        sl = t["paired_seed_level"]
        sl = sl[(sl.arm == arm) & (sl.baseline == bs) & (sl.metric == prim)
                & (sl.comparison == label)]
        sm = t["paired"]
        sm = sm[(sm.arm == arm) & (sm.baseline == bs) & (sm.metric == prim)
                & (sm.comparison == label)]
        tag = "" if label == "vs baseline" else f"_vs_{bs}"
        files = fig_paired_dots(Path(fig_dir) / f"{wave}_{arm}{tag}_paired_primary", sl, sm, qs,
                                f"{arm} - {bs}: {prim}", ylab)
        wr.figs.setdefault(arm, []).append(
            (f"Paired difference of the primary metric `{prim}`, {arm} - {bs}, one point per "
             f"seed, per q", files))
    overview = t["paired"][(t["paired"].metric == prim) & (t["paired"].comparison == "vs baseline")]
    arms_ov = [a for a in wr.arms if a != base]
    wr.figs["_overview"] = [(f"Overview: paired difference of `{prim}` (arm - {base}) per arm "
                             "and q",
                             fig_overview(Path(fig_dir) / f"{wave}_overview", overview, arms_ov,
                                          qs, f"{wave}: {prim}, arm - {base}",
                                          "paired difference (negative = improvement)"))]
    stop = t["stop_epoch"]
    for arm in wr.arms:
        if arm.endswith(("kl005", "kl010")) and not stop[stop.arm == arm].empty:
            wr.figs.setdefault(arm, []).append(
                (f"Stopping epoch of {arm}: share of updates by epochs run (pooled over runs)",
                 fig_stop_epochs(Path(fig_dir) / f"{wave}_{arm}_stop_epoch", stop, arm, qs,
                                 f"{arm}: epochs run per update")))
    if wave == "stage2":
        for arm in ("A_anneal2", "A_anneal4"):
            if arm in wr.arms:
                wr.figs.setdefault(arm, []).append(
                    (f"Smoothing-predicted vs observed d = 0 peak gap, {base} and {arm}",
                     fig_anneal(Path(fig_dir) / f"{wave}_{arm}_pred_vs_obs", wr.df, [arm], base,
                                qs, wr.seeds)))
        if all(a in wr.arms for a in ("A_anneal2", "A_anneal4")):
            wr.figs["_annealing"] = [(
                "Smoothing-predicted vs observed d = 0 peak gap, both annealing arms",
                fig_anneal(Path(fig_dir) / f"{wave}_annealing_pred_vs_obs", wr.df,
                           ["A_anneal2", "A_anneal4"], base, qs, wr.seeds))]
        if "A_detmean" in wr.arms and len(t.get("detmean_trajectory", [])):
            tr, lg, pf = t["detmean_trajectory"], t["detmean_logged"], t["detmean_profile"]
            wr.figs["A_detmean"] = wr.figs.get("A_detmean", []) + [
                ("Offline fixed-grid objective J (actor vs itself on the D_2 bin centres) at the "
                 "weight exports, median and IQR over seeds",
                 fig_traj(Path(fig_dir) / "stage2_A_detmean_objective", tr, "J_median",
                          "offline J = mean R(d, e(d), e(-d))", qs,
                          "A_detmean vs A_ctrl200: offline objective")),
                ("Offline first-order-condition residual |dR/de| (mean over the bin centres) at "
                 "the weight exports, median and IQR over seeds",
                 fig_traj(Path(fig_dir) / "stage2_A_detmean_foc", tr, "foc_mean_median",
                          "offline mean |dR/de|", qs, "A_detmean vs A_ctrl200: FOC residual",
                          logy=True))]
            if len(lg):
                wr.figs["A_detmean"].append(
                    ("Logged per-update loss of A_detmean with its rolling mean and SD (window "
                     f"{LOGGED_WINDOW})",
                     fig_logged(Path(fig_dir) / "stage2_A_detmean_logged_loss", lg, qs,
                                "A_detmean: logged loss (noisy, fresh rows each update)")))
            if len(pf):
                wr.figs["A_detmean"].append(
                    ("Stage-2 profile e2_hat(d) vs e2*(d) and |e2_hat - e2*| on the recovery "
                     "grid (median over seeds; diagnostic only)",
                     fig_profile(Path(fig_dir) / "stage2_A_detmean_profile", pf, qs,
                                 "stage-2 profile at the end of the phase")))



# --------------------------------------------------------------------------- wave reports
OUTCOME = {"stage1": [m for m, _ in S1_METRICS[:13]],
           "stage2": [m for m, _ in S2_METRICS[:14]]}


def paired_disp(p: pd.DataFrame, metrics: Sequence[str], with_arm: bool = False,
                arm_order: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Display frame of a paired table: mean / median with CIs, n_better, sign counts."""
    rows = []
    arms: List[Optional[str]] = ([None] if not with_arm else
                                 list(arm_order or dict.fromkeys(p["arm"])))
    for arm in arms:
        pa = p if arm is None else p[p.arm == arm]
        for m in metrics:
            for r in pa[pa.metric == m].sort_values("q").to_dict("records"):
                nb = r["n_better"]
                row: Dict[str, Any] = {} if arm is None else {"arm": arm}
                has_nb = nb is not None and not (isinstance(nb, float) and np.isnan(nb))
                row.update({
                    "metric": m, "q": r["q"], "n pairs": r["n_pairs"],
                    "mean [95% CI]": f"{_g(r['mean'])} "
                                     f"{ci_str(r['ci_mean_lo'], r['ci_mean_hi'])}",
                    "median [95% CI]": f"{_g(r['median'])} "
                                       f"{ci_str(r['ci_median_lo'], r['ci_median_hi'])}",
                    "n_better": f"{int(nb)}/{r['n_pairs']}" if has_nb else "n/a (no direction)",
                    "n(diff>0)/n(diff<0)/n(diff=0)":
                        f"{r['n_pos']}/{r['n_neg']}/{r['n_zero']}"})
                rows.append(row)
    return pd.DataFrame(rows)


def criterion_disp(c: pd.DataFrame, qs: Sequence[int]) -> pd.DataFrame:
    """Display frame of the criterion table (both parts reported separately)."""
    rows = []
    for r in c.to_dict("records"):
        d: Dict[str, Any] = {"arm": r["arm"], "baseline": r["baseline"],
                             "comparison": r.get("comparison", "vs baseline")}
        for q in qs:
            ci = ci_str(r[f"ci_mean_lo_q{q}"], r[f"ci_mean_hi_q{q}"])
            d[f"q={q}: mean diff [95% CI]"] = f"{_g(r[f'mean_q{q}'])} {ci} (n={r[f'n_pairs_q{q}']})"
            d[f"q={q}: CI < 0"] = r[f"a_q{q}"]
        d.update({"(a) met at both q": r["a_met"], "(a) all pairs present": r["a_complete"],
                  "(b) gate part": r["b_status"],
                  "(b) baseline-passing runs": r["n_base_pass"],
                  "(b) violations": r["b_violations"] or "none",
                  "(b) pending": r["b_pending"] or "none", "overall": r["overall"]})
        rows.append(d)
    return pd.DataFrame(rows)


def dispersion_disp(d: pd.DataFrame) -> pd.DataFrame:
    """Display frame of the dispersion table."""
    return pd.DataFrame([{"arm": r["arm"], "baseline": r["baseline"], "metric": r["metric"],
                          "q": r["q"], "n": r["n"],
                          "SD arm": r["sd_arm"], "SD baseline": r["sd_base"],
                          "ratio arm/baseline [95% CI]": f"{_g(r['ratio'])} "
                                                         f"{ci_str(r['ci_lo'], r['ci_hi'])}"}
                         for r in d.to_dict("records")])


def opt_disp(o: pd.DataFrame, wave: str) -> pd.DataFrame:
    """Display frame of the optimisation diagnostics (medians over runs)."""
    cols = [("median_kl_mean", "KL (mean over updates)"), ("median_clip_frac_mean", "clip frac"),
            ("median_gn_actor_mean", "actor grad norm (mean)"),
            ("median_gn_actor_max", "actor grad norm (max)"),
            ("median_n_actor_steps_total", "actor steps"),
            ("median_n_minibatch_steps_total", "optimizer steps"),
            ("median_n_epochs_run_mean", "epochs run (mean)"),
            ("median_adv_s1_std_mean" if wave == "stage1" else "median_adv_all_std_mean",
             "advantage SD (stage-1 rows)" if wave == "stage1" else "advantage SD (all rows)"),
            ("median_phase_wall_sec", "phase wall s")]
    rows = []
    for r in o.to_dict("records"):
        d = {"arm": r["arm"], "q": r["q"], "n complete": r["n_complete"]}
        d.update({lab: r.get(c) for c, lab in cols})
        rows.append(d)
    return pd.DataFrame(rows)


def summary_disp(s: pd.DataFrame, wave: str) -> pd.DataFrame:
    """Compact display of the absolute per-arm statistics (baseline sections)."""
    cols = {"stage1": [("arm", "arm"), ("q", "q"), ("n_complete", "n complete"),
                       ("median_stage1_rel_err_abs", "median |err|"),
                       ("mean_stage1_rel_err_abs", "mean |err|"),
                       ("median_stage1_rel_err_signed", "median signed err"),
                       ("sd_stage1_rel_err_signed", "SD signed err"),
                       ("median_Gmax_full_over_dw", "median Gmax/DW"),
                       ("median_EXP_root_over_dw", "median EXP_root/DW"),
                       ("median_within_run_sd_e1_last5", "median within-run SD e1"),
                       ("n_S1_pass", "n S1 pass"), ("n_G_F", "n G-F pass"),
                       ("n_run_pass", "n run pass"), ("n_target0929_pass", "n |err|<=0.05")],
            "stage2": [("arm", "arm"), ("q", "q"), ("n_complete", "n complete"),
                       ("median_stage2_peak_rel_err_abs", "median |peak err|"),
                       ("median_stage2_peak_rel_err_signed", "median signed peak err"),
                       ("sd_stage2_peak_rel_err_signed", "SD signed peak err"),
                       ("median_stage2_rmse_pos_over_g2_0", "median RMSE_pos/e2*(0)"),
                       ("median_stage2_tail_mean_over_g2_0", "median tail mean/e2*(0)"),
                       ("median_eta_T_over_dw", "median eta_2/DW"),
                       ("median_sigma_effort_at_0_t2", "median sigma_2(0)"),
                       ("n_G_A", "n G-A pass"), ("n_G_N_eta", "n G-N(eta) pass"),
                       ("n_gate_pass", "n gate pass")]}[wave]
    keep = [(c, lab) for c, lab in cols if c in s.columns]
    return s[[c for c, _ in keep]].rename(columns=dict(keep))


def completeness_sentence(comp: pd.DataFrame, arm: str) -> str:
    """Run counts of an arm over both q from the completeness table."""
    c = comp[comp.arm == arm]
    planned, done = int(c["planned"].sum()), int(c["done"].sum())
    return (f"Runs complete: {done} of {planned} (failed {int(c['failed'].sum())}, incomplete "
            f"{int(c['incomplete'].sum())}, missing {int(c['missing'].sum())}).")


def anomalies_disp(df: pd.DataFrame, arm: Optional[str] = None) -> pd.DataFrame:
    """Runs with a status other than done or a non-empty anomaly string."""
    g = df
    if "reference" in g.columns:
        g = g[~g["reference"].map(_truthy)]
    if arm is not None:
        g = g[g.arm == arm]
    g = g[(g.status != "done") | (g["anomalies"].fillna("") != "")]
    return pd.DataFrame([{"arm": r["arm"], "q": r["q"], "seed": r["seed"], "status": r["status"],
                          "detail": (r.get("anomalies") or r.get("status_info") or "")}
                         for r in g.to_dict("records")])


def set_aside_sentence(root: Path, wave: str, comp: pd.DataFrame) -> str:
    """One sentence on the original runs that were moved aside and re-run (not analysed).

    The counts are those of the ``set_aside_originals`` column of the completeness table
    (directory listing of ``<wave>/dirty_rerun`` and ``<wave>/crashed``). For the dirty re-runs
    the sentence cites ``<root>/stage1_dirty_rerun_comparison.json`` (read here: how many of the
    re-runs compare bit-identical to the originals).
    """
    n = int(comp["set_aside_originals"].sum()) if "set_aside_originals" in comp else 0
    if n == 0:
        return "No original run was set aside: every analysed run is the first run of its path."
    qs_ = sorted(int(q) for q in comp[comp["set_aside_originals"] > 0]["q"].unique())
    qtxt = f"all q = {qs_[0]}" if len(qs_) == 1 else "q = " + ", ".join(str(q) for q in qs_)
    droot = Path(root) / wave / "dirty_rerun"
    if not droot.exists():
        return (f"{n} original run directories ({qtxt}) were moved aside to "
                f"`{_rel(Path(root) / wave / 'crashed')}/` when their runs were re-run into "
                "their original paths; they are not analysed.")
    cmp_path = Path(root) / f"{wave}_dirty_rerun_comparison.json"
    cite = f"`{_rel(cmp_path)}`"
    ident = ""
    if cmp_path.exists():
        cj = json.loads(cmp_path.read_text())
        n_id = int(sum(1 for v in cj.values() if isinstance(v, dict) and v.get("all_identical")))
        ident = (", and compare bit-identical to the re-runs" if n_id == len(cj) == n else
                 f", and {n_id} of {len(cj)} compare bit-identical to the re-runs")
    else:
        cite = f"`{_rel(cmp_path)}` (not present)"
    moved = droot / "moved.json"
    mtxt = f"; `{_rel(moved)}` lists the moved directories" if moved.exists() else ""
    return (f"{n} originals ({qtxt}) were set aside because their manifests recorded dirty = "
            "true (an analysis-tool edit was uncommitted when they started); they were re-run "
            "once from a clean tree into the original paths, are superseded, not analysed"
            f"{ident} ({cite}{mtxt}; prereg Addendum 1 item 3).")


def commit_note(mc: pd.DataFrame, lr: pd.DataFrame) -> str:
    """Note on the commits in the run manifests against the heads of the launch records.

    Args:
        mc: :func:`manifest_commit_table` (commit, dirty flag, number of runs).
        lr: :func:`launch_record_table` (columns ``head`` and ``code_commit``).
    """
    if mc.empty:
        return "No manifest commit is available."
    per = mc.groupby("manifest_commit")["n_runs"].sum()
    n_all = int(mc["n_runs"].sum())
    n_dirty = int(mc[mc["manifest_dirty"].map(_truthy)]["n_runs"].sum())
    mtxt = ", ".join(f"`{c}` ({int(n)} runs)" for c, n in per.items())
    heads = sorted({str(h) for h in lr["head"].dropna()}) if "head" in lr else []
    codes = sorted({str(h) for h in lr["code_commit"].dropna() if str(h)}) \
        if "code_commit" in lr else []
    dtxt = ("`dirty` = false in all of them" if n_dirty == 0
            else f"`dirty` = true in {n_dirty} of them")
    txt = (f"The `manifest.json` of the {n_all} analysed runs record the commit(s) {mtxt}, "
           f"{dtxt}; the launch records name the head(s) {', '.join(f'`{h}`' for h in heads)} "
           f"and the code commit(s) {', '.join(f'`{h}`' for h in codes)}.")
    if set(per.index) != set(heads) or not set(codes) <= set(per.index):
        txt += (" The commits differ because HEAD advanced during the wave (documentation and "
                "analysis-tool commits only); the run code is identical (prereg Addendum 1 "
                "item 1).")
    return txt


def checks_section(doc: Doc, wr: Optional[WaveResult], root: Path, which: Sequence[str]) -> None:
    """The reproducibility checks of the wave (they come first).

    Shows the summary of every check file of the wave (and C-R1), the per-run results of the
    runs that are NOT identical (the full per-run grid is in the cited CSV), and, for stage 1,
    the verification that stage 2 is shared (frozen snapshot bit-identical to the parent actor).
    """
    summ, runs = checks_tables(root)
    doc.h(2, "1. Checks (they come first)")
    doc.p("The paired design needs the branch points to reproduce the locked pipeline. The "
          "check files are written by `tools/v2/cr1_compare.py` (read only here). A missing "
          "file means the check has not been produced yet.")
    cols = ["check", "file", "present", "mode", "n", "n_identical", "ALL", "failing_fields",
            "first_difference", "n_snapshot_refresh_counter_differs"]
    sel = summ[summ.check.isin(list(which) + ["C-R1"])]
    doc.table(summ, "checks_summary", disp=sel[[c for c in cols if c in sel]])
    runs.to_csv(doc.out_dir / "checks_per_run.csv", index=False)     # always written
    if len(runs):
        mine = runs[runs.check.isin(list(which) + ["C-R1"])]
        bad = mine[~mine["ALL"].map(_truthy)]
        if bad.empty:
            doc.p(f"Per-run results: all {len(mine)} compared (check, q, seed) results are "
                  f"identical (`{_rel(doc.out_dir / 'checks_per_run.csv')}` lists every run).")
        else:
            doc.p("Per-run results that are NOT identical:")
            doc.table(runs, "checks_per_run", disp=bad[["check", "q", "seed", "ALL",
                                                        "first_difference_field",
                                                        "failing_fields"]])
    sb = summ[(summ.check == "stage2_base") & summ["present"].map(_truthy)]
    if wr is not None and wr.wave == "stage2" and len(sb):
        doc.p("The snapshot-refresh counter of `A_base` may differ from the rehearsal's by the "
              "extra phase-entry refresh of a continued phase (the check reports it separately; "
              "it is not training-relevant): it differs in "
              f"{int(sb['n_snapshot_refresh_counter_differs'].iloc[0])} of "
              f"{int(sb['n'].iloc[0])} runs "
              f"(`{sb['file'].iloc[0]}`, `info` of each run).")
    if wr is not None and wr.wave == "stage1":
        t = wr.tabs
        ident = (t["decomposition"][["arm", "q", "n_complete", "n_frozen_bit_identical",
                                     "n_drift_test_pass"]]
                 if len(t["decomposition"]) else pd.DataFrame())
        doc.p("Shared stage 2: the induced target e~1 and its residual band are the same for all "
              "arms of a (q, seed) because stage 2 is frozen from the same parent. Verification "
              "per run: the frozen snapshot in each arm's `state_end_B.pt` is compared tensor by "
              "tensor (`torch.equal`) with the parent's `state_end_A.pt` actor; the bands come "
              "from the rehearsal `induced_band.json` of that (q, seed). Counts of complete runs "
              "with a bit-identical snapshot and a passing `drift_test.json`:")
        doc.table(t["decomposition"], "stage1_decomposition", disp=ident,
                  note="per-run record: columns `frozen_bit_identical_to_parent`, "
                       "`drift_test_pass` of `" + _rel(doc.out_dir / "stage1_per_run.csv") + "`")
        doc.h(3, "Check (ii) of the continuation table (method 6, `B_expcont`)")
        check_ii_block(doc, root, wr.tabs["criterion"], "stage1_continuation_check_ii")


REFINED_COL = "refined_verifier (state step 0.25, 64 GL nodes) max abs diff / DW"


def check_ii_sentence(ct: pd.DataFrame) -> str:
    """Plain statement of the check-(ii) outcome from the continuation-check table.

    The table is :func:`continuation_table` (one row per tier and q, read from
    ``results/v2_refine/continuation_check.json``). The literal final-tier criterion is
    ``max |V~2 - verifier| <= spec_tolerance_over_dw``.
    """
    if "tier" not in ct.columns:
        return ("The continuation check file `results/v2_refine/continuation_check.json` is not "
                "present, so check (ii) cannot be stated.")
    fin = ct[ct["tier"] == "final"]
    dev = ct[ct["tier"] == "development"]
    tol = float(fin["spec_tolerance_over_dw"].iloc[0])
    met = bool(len(fin) and all(_truthy(v) for v in fin["meets_spec_tolerance"]))
    def by_q(d: pd.DataFrame, col: str) -> str:
        return " / ".join(f"{float(v):.3e}" for v in d[col])

    qtxt = " / ".join(f"q = {q}" for q in fin["q"])
    tol_txt = f"{tol:.0e}".replace("e-0", "e-")
    txt = (f"the literal final-tier criterion (maximum absolute difference between the table "
           f"value and the verifier's stage-1 Q + k e^2 on every effort-grid node <= {tol_txt} "
           f"DW) is {'met' if met else 'NOT met'}: {by_q(fin, 'max_abs_diff_over_dw')} DW at "
           f"{qtxt}")
    if len(dev):
        txt += f" (development tier {by_q(dev, 'max_abs_diff_over_dw')} DW)"
    if REFINED_COL in fin and fin[REFINED_COL].notna().all():
        txt += (f"; the refined verifier configuration (state step 0.25, 64 Gauss-Legendre nodes) "
                f"gives {by_q(fin, REFINED_COL)} DW")
    return txt + "."


def check_ii_block(doc: Doc, root: Path, crit: pd.DataFrame, name: str) -> None:
    """Check (ii) of the continuation table: table, outcome sentence and the criterion statement.

    The statement names the arms that meet the pre-registered criterion; when ``B_expcont`` is
    the only one it says so and says that check (ii) is open as stated.
    """
    ct = continuation_table(root)
    doc.p("Outcome of check (ii) of PI prompt section 2.3 (the table V~2(y) against the "
          "verifier's stage-1 Q + k e^2; produced by `tests/test_v2_refine_continuation.py "
          "--write`; the full record is `results/v2_refine/continuation_check.json`): "
          + check_ii_sentence(ct)
          + " The PI decided to keep method 6 in the round and to record both results "
            "(`reports/v2/refine/01_preregistration.md` section 4 item 6).")
    doc.table(ct, name, disp=ct.drop(columns=["file"], errors="ignore"),
              note="rows: `results/v2_refine/continuation_check.json`, key `table_vs_verifier`")
    doc.p(criterion_statement(crit, check_ii_met(ct)))


def check_ii_met(ct: pd.DataFrame) -> Optional[bool]:
    """True iff every final-tier row of the check-(ii) table meets the tolerance (None: no data)."""
    fin = ct[ct["tier"] == "final"] if "tier" in ct else ct.iloc[0:0]
    return None if fin.empty else bool(all(_truthy(v) for v in fin["meets_spec_tolerance"]))


def criterion_statement(crit: pd.DataFrame, check_met: Optional[bool]) -> str:
    """Which stage-1 arms meet the criterion, and whether check (ii) of method 6 is open.

    Args:
        crit: The stage-1 criterion table.
        check_met: Whether the literal final-tier check (ii) is met (None = file not present).
    """
    cv = crit[crit["comparison"] == "vs baseline"] if "comparison" in crit else crit
    met = [a for a, o in zip(cv["arm"], cv["overall"]) if o == "met"]
    if met == ["B_expcont"]:
        who = ("`B_expcont` is the only arm that meets the pre-registered criterion (part (a) at "
               "both q and part (b));")
    elif "B_expcont" in met:
        who = ("`B_expcont` meets the pre-registered criterion, as do "
               + ", ".join(f"`{a}`" for a in met if a != "B_expcont") + ";")
    else:
        who = ("`B_expcont` does not meet the pre-registered criterion (arms that do: "
               + (", ".join(f"`{a}`" for a in met) or "none") + ");")
    if check_met is None:
        tail = "check (ii) of its continuation table cannot be stated (file not present)."
    elif check_met:
        tail = "check (ii) of its continuation table is met as stated."
    else:
        tail = ("check (ii) of its continuation table is open as stated: the literal final-tier "
                "criterion of the check is not met (see the outcome above).")
    return who + " " + ("at the same time, " if met else "in addition, ") + tail


def write_wave_report(wr: WaveResult, args: argparse.Namespace, sub: str) -> Path:
    """Generate ``04_pilot_stage1.md`` or ``05_pilot_stage2.md`` from the tables of a wave."""
    wave, base, qs = wr.wave, wr.base, wr.qs
    t, df = wr.tabs, wr.df
    stage1 = wave == "stage1"
    doc = Doc(Path(args.out))
    fig_dir, rep_dir = Path(args.figures), Path(args.reports)
    prim = PRIMARY[wave]
    if stage1:
        doc.h(1, "R1 pilot wave 1: stage-1 (Phase B) arms")
        doc.p("Candidate: the end-of-B last iterate. Baseline arm: `B_base` (the locked Phase B "
              "from the rehearsal end-of-A state). Method numbering and arm definitions are those "
              "of `tools/v2/launch_refine.py` (`METHOD_ARMS`, `STAGE1_ARMS`). Method 6 "
              "(`B_expcont`) has a pre-launch check, check (ii) of its continuation table: "
              "its outcome is stated in section 1 (subsection \"Check (ii)\"), in section "
              "4 and in the `B_expcont` section.")
    else:
        doc.h(1, "R1 pilot wave 2: stage-2 (Phase A continuation) arms and the method-5 pair")
        doc.p("Candidate: the stage-2 last iterate at the end of the continued phase. Baseline "
              "arm: `A_base` (the locked global updates 1201-1600 from `parents_A` u1200). The "
              "method-5 pair `A_ctrl200` / `A_detmean` starts from the baseline u1600 state "
              "(`rehearsal_v1_1` `state_end_A.pt`) and is also compared with that parent "
              "candidate (pseudo-arm `parent_u1600`, read from the rehearsal `gates.json`) and "
              "with each other (`A_detmean` is an ablation against its matched control).")
    doc.p("Generated by `tools/v2/refine_analysis.py " + sub + "`; deterministic given the same "
          "inputs. Every number below is read from a CSV that is cited under its table; nothing is "
          "typed by hand. Section 1 holds the checks, section 2 the completeness of the wave, "
          "section 3 the definitions, section 4 the stage-wide overview, section 5 one section per "
          "arm, section 6 the cost per run, section 7 the anomalies index, section 8 the command.")
    checks_section(doc, wr, Path(args.root),
                   ["parents_A", "stage1_base"] if stage1 else ["parents_A", "stage2_base"])
    # ---- completeness
    doc.h(2, "2. Completeness: runs done / failed / incomplete / missing")
    doc.table(t["completeness"], f"{wave}_completeness",
              disp=t["completeness"][["arm", "q", "planned", "done", "failed", "incomplete",
                                      "missing", "not_done_seeds", "set_aside_originals"]])
    doc.p(set_aside_sentence(Path(args.root), wave, t["completeness"]))
    lr = launch_record_table(Path(args.root), [wave])
    doc.p("Launch records of the wave (read only; re-runs have their own record):")
    doc.table(lr, f"{wave}_launch_record")
    doc.p(commit_note(t["manifest_commits"], lr))
    doc.table(t["manifest_commits"], f"{wave}_manifest_commits",
              note="commit and dirty flag recorded in `manifest.json` of the analysed runs")
    # ---- definitions
    doc.h(2, "3. Definitions")
    doc.p("The analysis is the one of `reports/v2/refine/01_preregistration.md` section 5: final "
          "tier for every gate metric (development tier only for the G-N part), recovery metrics "
          "are tier independent; every difference is arm - baseline paired by (q, seed); "
          "`n_better` counts the seeds whose primary metric is smaller than the baseline's; the "
          "confidence intervals are 95% percentile bootstrap intervals of the mean and of the "
          f"median of the paired differences ({N_BOOT} resamples of the paired seeds, "
          f"`numpy.random.default_rng({BOOT_SEED})`, one fresh generator per (q, statistic)). "
          "A run is complete iff `status.json` says done with exit code 0 and `final_v2.json` "
          "holds a final-tier evaluation; every other planned run is listed.")
    doc.p(f"Primary metric: `{prim}` " + (
        "= |e1_hat(0) - e1*| / e1* (the S1 value); improvement = a decrease." if stage1 else
        "= |signed peak error| = |e2_hat(0) - e2*(0)| / e2*(0); improvement = a decrease of the "
        "absolute value (the baseline peak errors are negative, so this equals an increase of the "
        "signed error); the signed difference is reported as well."))
    doc.p("Thresholds of the verdicts (read from `protocols/v2_T2_locked_v1_1.json`):")
    doc.table(protocol_thresholds(), f"{wave}_thresholds")
    if stage1:
        doc.p("Stage-1 verdict definitions: G-F = `Gmax_full_over_dw` <= its threshold (final "
              "tier); G-N (Gmax part) = |dev - final| of `Gmax_full_over_dw` <= its threshold; "
              "the G-A and the eta part of G-N of a stage-1 run are those of the shared parent "
              "(`rehearsal_v1_1/.../gates.json`); run pass = parent G-A and parent G-N (eta part) "
              "and arm G-F and arm G-N (Gmax part) (the global-RNG assertion of the locked entry "
              "point does not exist in the stage runner); S1 = `stage1_rel_err_abs` <= its "
              "threshold; 0929 target = |signed stage-1 error| <= 0.05. Verdict strings "
              "(`outcome`) are those of `run.run_v2_T2_locked.verdicts`. The learning / inherited "
              "decomposition uses the residual band of the shared parent; because e~1 is shared, "
              "the paired difference of `learning_rel` equals that of `stage1_rel_err_signed`.")
        doc.p("Within-run stability: `within_run_sd_e1_last5` and `within_run_range_e1_last5` are "
              "computed per run from the five weight exports at global updates "
              f"{', '.join(str(u) for u in LAST5)} (`weights/u*.npz`): e1_hat(0) of each export "
              "(`agents.ppo_curriculum.mean_effort_numpy` at the observation [0, 0], effort "
              "units, e_max - e_min = 100), then the sample SD (ddof = 1) and the range "
              "(max - min) of those five values. The across-seed SD of the dispersion table is "
              "the sample SD (ddof = 1) over the seeds of one arm and q.")
    else:
        doc.p("Stage-2 verdict definitions: G-A = `eta_T_over_dw`, `stage2_rmse_pos_over_g2_0` and "
              "`stage2_tail_mean_over_g2_0` each <= their threshold (final tier); G-N (eta part) = "
              "|dev - final| of `eta_T_over_dw` <= its threshold; the gate of the criterion is G-A "
              "with its G-N (eta) part (`gate_pass`). The stage-1 parts of the protocol are not "
              "evaluated for stage-2 runs (stage 1 is untrained). sigma_2(0) of the annealing arms "
              "is the annealed value (the run's own `beta_fn`); every reload in this analysis "
              "applies the exported `conc_scale` (`load_actor`), cross-checked against the run's "
              "own value in the per-run table (`sigma2_0_reload_rel_diff`). The location-free peak "
              "error is the `stage2_extra` logic of `run/run_v2_T2_locked.py` on the recovery "
              "arrays of `final_final.npz`.")
        doc.p("The method-5 ablation `A_detmean` has two criterion rows and two dispersion "
              "ratios: against `A_base` like every arm, and against its matched control "
              "`A_ctrl200` (`comparison` = `ablation vs matched control`; the pre-registered "
              "comparison of the ablation: same parent state, same 200 updates). For the "
              "control row, part (b) concerns the runs that passed their gate under "
              "`A_ctrl200`. The `A_base` end state is compared with the rehearsal end-of-A values "
              "in the sanity table of the `A_base` section.")
    # ---- overview
    doc.h(2, "4. Stage-wide overview of the primary metric")
    ov = t["paired"][(t["paired"].metric == prim) & (t["paired"].comparison == "vs baseline")]
    ovd = paired_disp(ov, [prim], with_arm=True, arm_order=wr.arms).drop(columns=["metric"])
    doc.table(ov, f"{wave}_overview", disp=ovd, note=f"rows: metric == {prim}, arm - {base}")
    doc.figures(wr.figs.get("_overview", []), fig_dir, rep_dir)
    doc.p("Per-arm criterion (descriptive, not a gate; part (a) = CI of the mean paired "
          "difference of the primary metric below 0 at both q, part (b) = no baseline-passing run "
          "fails under the arm):")
    cr = t["criterion"]
    doc.table(cr, f"{wave}_criterion", disp=criterion_disp(cr, qs))
    if stage1:
        doc.p(criterion_statement(cr, check_ii_met(continuation_table(Path(args.root)))))
    # ---- arm sections
    doc.h(2, "5. Arms (in the order of the arm table)")
    for arm in wr.arms:
        arm_section(doc, wr, arm, args, sub)
    if not stage1 and wr.figs.get("_annealing"):
        doc.h(3, "Both annealing arms together")
        doc.figures(wr.figs["_annealing"], fig_dir, rep_dir)
    # ---- cost
    doc.h(2, "6. Cost per run")
    doc.p("Mean over the complete runs of both q; `phase_wall_sec` is the wall time of the phase "
          "from `v2_run_summary.json`, `total_wall_sec` the run's wall time from `status.json` "
          "(includes the final evaluation), episodes / optimizer steps are the phase totals from "
          "`train_history.json`. The runs shared the machine with other jobs, so wall times are "
          "indicative only. `phase_wall_ratio_vs_base` is the ratio of phase wall times; "
          "`wall_per_update_ratio_vs_base` divides each wall time by the number of updates of "
          "the phase first, which is the comparable ratio when the number of updates differs."
          + ("" if stage1 else
             " In this wave `A_ctrl200` and `A_detmean` run 200 updates against 400 for the "
             "other arms; `wall_ratio_vs_matched_control` is their phase wall time relative to "
             "`A_ctrl200` (same 200 updates). The `episodes` of `A_detmean` are exploring-start "
             "rows of the pathwise updates (phase P draws no action and no shock, there are no "
             "rollouts and no simulated episodes)."))
    cost = t["cost"]
    cost_cols = ["arm", "n_runs", "mean_phase_local_updates", "mean_phase_wall_sec",
                 "mean_wall_sec_per_update", "phase_wall_ratio_vs_base",
                 "wall_per_update_ratio_vs_base"]
    if not stage1:
        cost_cols.append("wall_ratio_vs_matched_control")
    cost_cols += ["mean_phase_episodes", "mean_n_minibatch_steps_total", "mean_total_wall_sec"]
    doc.table(cost, f"{wave}_cost", disp=cost[[c for c in cost_cols if c in cost]],
              note="the CSV has more columns (optimizer and actor steps, update and rollout "
                   "seconds)")
    # ---- anomalies index
    doc.h(2, "7. Anomalies index (all arms)")
    an = anomalies_disp(df)
    doc.table(df, f"{wave}_per_run", disp=an if len(an) else
              pd.DataFrame([{"result": "no failed, incomplete or anomalous planned run"}]),
              note="columns `status`, `anomalies`; one row per planned run")
    doc.h(2, "8. Reproduce")
    doc.p("```\n" + repro_cmd(sub, args) + "\n```")
    path = Path(args.reports) / ("04_pilot_stage1.md" if stage1 else "05_pilot_stage2.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc.text())
    return path


def arm_section(doc: Doc, wr: WaveResult, arm: str, args: argparse.Namespace, sub: str) -> None:
    """One arm: definition, paired table, criterion (both parts), diagnostics, figures."""
    wave, base, qs = wr.wave, wr.base, wr.qs
    t, df = wr.tabs, wr.df
    stage1 = wave == "stage1"
    fig_dir, rep_dir = Path(args.figures), Path(args.reports)
    doc.h(3, f"`{arm}`" + (f"  (method {method_of(arm)})" if method_of(arm) not in ("", "baseline")
                          else "  (baseline arm)" if arm == base else ""))
    doc.p(f"Single change: {definition_of(wave, arm)}. "
          + ("The outcome of check (ii) of this table is stated below. "
             if stage1 and arm == "B_expcont" else "")
          + completeness_sentence(t["completeness"], arm))
    if arm == base:
        s = t["arm_summary"]
        sel = s[s.arm.isin([arm] + (["parent_u1600"] if not stage1 else []))]
        doc.p("Baseline values (complete runs; medians, means and across-seed SDs; the CSV has "
              "more columns):")
        doc.table(s, f"{wave}_arm_summary", disp=summary_disp(sel, wave))
        if not stage1:
            sane = t["paired"][(t["paired"].comparison.str.startswith("sanity"))]
            doc.p("Sanity: the end state of `A_base` against the rehearsal end-of-A values "
                  "(`A_base - parent_u1600`). When the reproduction check holds the differences "
                  "of the verifier and recovery metrics are exactly 0; the row "
                  "`smoothed_share_peak_gap_d0` is about 1e-6 instead of 0 because the parent "
                  "share (from `gates.json`) was computed from a single-row evaluation of the "
                  "effort at d = 0, while this tool takes it from the batched final evaluation "
                  "(`final_v2.json`, float32 batch paths differ in the last bits); "
                  "`smoothed_pred_gap_d0` uses the single-row (alpha, beta) in both and is "
                  "exactly 0.")
            doc.table(t["paired"], f"{wave}_paired", disp=paired_disp(sane, OUTCOME[wave]),
                      note="rows: comparison == sanity")
    else:
        comps = [c for c in wr.comps if c[0] == arm]
        for a_, b_, label in comps:
            p = t["paired"][(t["paired"].arm == a_) & (t["paired"].baseline == b_)
                            & (t["paired"].comparison == label)]
            doc.p(f"**Paired differences {a_} - {b_}** ({label}); outcome metrics. "
                  f"`n_better` = seeds with a smaller value than the {b_} run.")
            doc.table(t["paired"], f"{wave}_paired",
                      disp=paired_disp(p, OUTCOME[wave]),
                      note=f"rows: arm == {a_}, baseline == {b_}; seed-level values in "
                           f"`{_rel(Path(args.out) / (wave + '_paired_seed_level.csv'))}`")
            if label in CRITERION_LABELS:
                doc.p("**Pre-registered criterion** (descriptive; both parts separately"
                      + ("" if label == "vs baseline" else
                         f"; here against the matched control `{b_}`, part (b) = no run that "
                         f"passed its gate under `{b_}` fails under `{a_}`") + "):")
                crit = t["criterion"]
                c = crit[(crit.arm == a_) & (crit.baseline == b_) & (crit.comparison == label)]
                doc.table(crit, f"{wave}_criterion", disp=criterion_disp(c, qs),
                          note=f"row: arm == {a_}, baseline == {b_}")
        if arm in set(t["dispersion"]["arm"]):
            dd = t["dispersion"][t["dispersion"].arm == arm]
            doc.p("**Dispersion** across seeds (SD with ddof = 1) of the signed error and of the "
                  "effort at d = 0; ratio arm/baseline with a paired-resampling bootstrap CI "
                  "(same resampled seeds for both). The two metrics give the same ratio because "
                  "the signed error is an affine function of the effort with the same target"
                  + ("" if stage1 else " (descriptive extra: the pre-registration lists "
                                       "dispersion for stage 1 only)")
                  + ("; `A_detmean` is shown against `A_base` and against its matched control "
                     "`A_ctrl200`" if arm == "A_detmean" else "") + ".")
            doc.table(t["dispersion"], f"{wave}_dispersion", disp=dispersion_disp(dd),
                      note=f"rows: arm == {arm}")
        if stage1:
            dec = t["decomposition"][t["decomposition"].arm == arm]
            doc.p("**Learning / inherited decomposition** (e~1 and its band from the shared "
                  "parent; learning = (e1_hat - e~1)/e1*, inherited = (e~1 - e1*)/e1*):")
            doc.table(t["decomposition"], "stage1_decomposition", disp=dec,
                      note=f"rows: arm == {arm}")
        if not stage1 and arm in ("A_anneal2", "A_anneal4"):
            an = t["annealing"][t["annealing"].arm == arm]
            doc.p("**Smoothing explanation test.** Predicted peak gap g2(0) - e_pred(0) from the "
                  "smoothed-game prediction (Pilot-1 method, 400 equal-probability nodes per Beta, "
                  "reconstructed from the saved actor including its `conc_scale`) BEFORE "
                  "annealing (the `A_base` end state, same seed) and AFTER (this arm's end state), "
                  "next to the observed change of e2_hat(0). Change = arm - A_base.")
            show = an[["q", "quantity", "n_pairs", "median_before", "median_after", "mean_change",
                       "median_change"]].copy()
            show["95% CI of mean change"] = [ci_str(r["ci_mean_lo"], r["ci_mean_hi"])
                                             for r in an.to_dict("records")]
            doc.table(t["annealing"], "stage2_annealing", disp=show, note=f"rows: arm == {arm}")
            cc = t["annealing_corr"]
            cc = cc[cc.arm == arm]
            doc.p("Pearson correlation over seeds between the change of the predicted gap and "
                  "the change of the observed gap (one value per q; changes = arm - A_base):")
            doc.table(t["annealing_corr"], "stage2_annealing_corr",
                      disp=cc[["q", "n_pairs", "corr_dchange_pred_gap_vs_dchange_obs_gap"]],
                      note=f"rows: arm == {arm}")
        if not stage1 and arm == "A_detmean":
            detmean_tables(doc, wr)
        if stage1 and arm == "B_expcont":
            ct = continuation_table(Path(args.root))
            ii_csv = _rel(Path(args.out) / "stage1_continuation_check_ii.csv")
            doc.p("**Check (ii) of the continuation table"
                  + (": open as stated" if check_ii_met(ct) is False else "") + ".** Source "
                  "`results/v2_refine/continuation_check.json` (key `table_vs_verifier`; the "
                  f"same numbers are in `{ii_csv}` and in section 1): " + check_ii_sentence(ct)
                  + " The PI decided to keep method 6 in the round and to record both results "
                    "(`reports/v2/refine/01_preregistration.md` section 4 item 6).")
            doc.p(criterion_statement(t["criterion"], check_ii_met(ct)))
    gc = t["gate_counts"]
    gsel = gc[gc.arm.isin([arm, base])]
    doc.p("**Gate metrics** (verdict counts over the complete runs, final tier; "
          + ("`n_Gmax_at_t*` and the d* columns locate the maximum deviation gain" if stage1 else
             "the location-free peak error's argmax d is its location") + "):")
    doc.table(gc, f"{wave}_gate_counts", disp=gsel, note=f"rows: arm in ({arm}, {base})")
    o = t["optimisation"]
    op = o[o.arm.isin([arm, base])]
    doc.p("**Optimisation diagnostics** (medians over the complete runs of per-run values: mean "
          "over the updates of the phase; grad norms are the actor's global pre-clip norms)"
          + ("; `A_detmean` has no KL, clip, epoch or advantage columns by construction."
             if arm == "A_detmean" else "."))
    doc.table(o, f"{wave}_optimisation", disp=opt_disp(op, wave),
              note=f"rows: arm in ({arm}, {base})")
    if stage1 and arm == "B_expcont":
        ar = t["adv_ratio"][t["adv_ratio"].arm == arm]
        doc.p("**Variance-reduction measurement** (`B_expcont`): the SD of the raw stage-1 "
              "advantages (mean over the 600 updates of a run) relative to the baseline's, per "
              "(q, seed) pair:")
        doc.table(t["adv_ratio"], "stage1_adv_ratio", disp=ar, note=f"rows: arm == {arm}")
    rg = t["rng_divergence"]
    rg = rg[(rg.arm == arm) & (rg.n_runs > 0)] if len(rg) else rg
    if len(rg):
        cmp_with = "A_ctrl200 (its matched control)" if arm == "A_detmean" else base
        txt = ("First update of the phase at which each RNG stream's position differs from "
               f"that of {cmp_with} (`never` = identical to the end). ")
        if "batch" in arm:
            txt += ("The batch arms diverge at the first update BY CONSTRUCTION: a different "
                    "number of episodes per update draws a different number of values from every "
                    "stream; this is not an anomaly.")
        elif arm == "A_detmean":
            txt += ("`A_detmean` draws no action and no shock, so every stream except `start` "
                    "stays where it was at the parent state, while the control advances them.")
        elif "kl" in arm:
            txt += ("Rule A6 (the permutations of skipped epochs are still drawn) predicts that "
                    "the minibatch stream never diverges; the other streams diverge once the "
                    "policies differ.")
        doc.p("**RNG divergence.** " + txt)
        doc.table(t["rng_divergence"], f"{wave}_rng_divergence",
                  disp=rg[["q", "stream", "n_runs", "n_at_first_update", "n_never",
                           "first_update_of_phase", "median_first_divergence",
                           "min_first_divergence", "max_first_divergence", "n_not_comparable"]],
                  note=f"rows: arm == {arm}")
    se = t["stop_epoch"]
    if arm.endswith(("kl005", "kl010")) and len(se[se.arm == arm]):
        sp = se[se.arm == arm].pivot_table(index="q", columns="epochs_run", values="share")
        sp.columns = [f"ep {c}" for c in sp.columns]
        sp = sp.reset_index()
        sp["share fewer than 10 epochs"] = [float(se[(se.arm == arm) & (se.q == q)]
                                                  ["share_fewer_than_10"].iloc[0])
                                            for q in sp["q"]]
        doc.p("**Stopping epoch** (epochs run per update, share of all updates pooled over the "
              "complete runs):")
        doc.table(se, f"{wave}_stop_epoch", disp=sp, note=f"rows: arm == {arm}")
    figs = wr.figs.get(arm, [])
    if figs:
        doc.p("**Figures.**")
        doc.figures(figs, fig_dir, rep_dir)
    an_ = anomalies_disp(df, arm)
    doc.p("**Anomalies.**" + (" None among the planned runs of this arm." if an_.empty else ""))
    if len(an_):
        doc.table(df, f"{wave}_per_run", disp=an_,
                  note=f"rows: arm == {arm}, columns `status`, `anomalies`")
    doc.p("**Reproduce:** `" + repro_cmd(sub, args) + "`")


def detmean_tables(doc: Doc, wr: WaveResult) -> None:
    """A_detmean: loss trajectory, FOC residual at the end, profile (the pre-registered items)."""
    t = wr.tabs
    doc.p("**Objective and first-order condition.** Definition of the offline objective: on the "
          "bin centres d of D_2 (`es_bin_width` 10) the learner's effort is the actor's Beta-mean "
          "e(d) and the opponent is the SAME actor at -d, `J = mean_d R(d, e(d), e(-d))` with "
          "`R = w_l + DW F_xi(d + e - e_opp) - k e^2` (`agents.ppo_pathwise.expected_payoff` on "
          "`effort_mean`), computed from the weight exports every 25 updates and from the parent "
          "state (local 0). The training loss is `-J` on random exploring starts with the LAGGED "
          "opponent held fixed, so `J` of the actor against itself is a fixed-grid reading, not "
          "the quantity the update ascends; the first-order-condition residual |dR/de| is. The "
          "logged per-update `loss` and `foc_abs_*` are evaluated on that update's 512 fresh rows "
          "and are noisy; they are reported as rolling statistics (window "
          f"{LOGGED_WINDOW}) and as the last-20-update mean (max for the maximum) next to the "
          "offline fixed-grid value of the final weights (mean and max over the bin centres).")
    s = t["detmean_summary"]
    lab = [("q", "q"), ("arm", "arm"), ("n_complete", "n"),
           ("median_J_offline_start", "J start (u1600)"), ("median_J_offline_end", "J end (u1800)"),
           ("median_foc_offline_end_mean", "offline FOC mean at end"),
           ("median_foc_offline_end_max", "offline FOC max at end"),
           ("median_loss_first20_mean", "logged loss, first 20 updates"),
           ("median_loss_last20_mean", "logged loss, last 20 updates"),
           ("median_loss_last20_sd", "logged loss SD, last 20"),
           ("median_foc_logged_mean_last20", "logged FOC mean, last 20"),
           ("median_foc_logged_max_last20", "logged FOC max (max of last 20)"),
           ("median_stage2_peak_rel_err_signed", "signed peak error"),
           ("median_stage2_peak_locfree_argmax_d", "location-free argmax d")]
    doc.p("Medians over seeds of the per-run values (the last verifier call of `A_detmean` is "
          "labelled `timeout` because the cap 200 is a multiple of the timeout 50; the "
          "end-of-phase row is the one with `local = cap`, and the end metrics here come from "
          "`final_v2.json`).")
    doc.table(s, "stage2_detmean_summary", disp=s[s["arm"].isin(["A_detmean", "A_ctrl200"])][
        [c for c, _ in lab if c in s.columns]].rename(columns=dict(lab)))
    dv = s[~s["arm"].isin(["A_detmean", "A_ctrl200"])]
    doc.p("Paired difference A_detmean - A_ctrl200 of the offline end values (mean, median, CI of "
          "the mean):")
    doc.table(s, "stage2_detmean_summary", disp=dv[["q", "metric", "n_complete", "mean", "median",
                                                    "ci_mean_lo", "ci_mean_hi", "n_pos",
                                                    "n_neg"]],
              note="rows: arm == A_detmean - A_ctrl200")
    doc.p("Trajectory of the offline objective and FOC over the exports (median and IQR over "
          "seeds): see the two trajectory figures below; table:")
    tr = t["detmean_trajectory"]
    doc.table(tr, "stage2_detmean_trajectory",
              disp=tr[["q", "arm", "update", "local", "n", "J_median", "J_p25", "J_p75",
                       "foc_mean_median", "foc_max_median"]])
    doc.p("Per-run values (columns `J_offline_*`, `foc_offline_end_*`, `loss_*`, `foc_logged_*`) "
          f"are in `{_rel(Path(doc.out_dir) / 'stage2_per_run.csv')}`; the profile "
          f"|e2_hat(d) - e2*(d)| on the recovery grid (diagnostic only) is in "
          f"`{_rel(Path(doc.out_dir) / 'stage2_detmean_profile.csv')}` and shown below.")



# --------------------------------------------------------------------------- decision inputs
OBS_HEADING = "## Observations (what the data show; not recommendations)"
OBS_PLACEHOLDER = "<!-- OBSERVATIONS-PLACEHOLDER -->"


def _read(out: Path, name: str) -> pd.DataFrame:
    p = Path(out) / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _one(df: pd.DataFrame, **kw: Any) -> Optional[Dict[str, Any]]:
    """The single row of ``df`` matching ``kw`` as a dict, or None."""
    if df.empty:
        return None
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        if k not in df:
            return None
        m &= df[k] == v
    r = df[m]
    return r.iloc[0].to_dict() if len(r) else None


def decision_long(out: Path, qs: Sequence[int]) -> pd.DataFrame:
    """One row per (method, arm): primary difference per q, criterion, dispersion, cost.

    Everything is read from the wave CSVs written by the ``stage1`` / ``stage2`` sub-commands.
    """
    names = ("paired", "criterion", "dispersion", "cost")
    data = {w: {n: _read(out, f"{w}_{n}.csv") for n in names} for w in ("stage1", "stage2")}
    rows = []
    for method, d in METHOD_ARMS.items():
        for wave in ("stage1", "stage2"):
            for arm in d[wave]:
                base = BASE_ARM[wave]
                prim = PRIMARY[wave]
                row: Dict[str, Any] = {"method": method, "stage": wave, "arm": arm,
                                       "baseline": base, "primary_metric": prim}
                cr = _one(data[wave]["criterion"], arm=arm, baseline=base,
                          comparison="vs baseline")
                for q in qs:
                    pr = _one(data[wave]["paired"], arm=arm, baseline=base, q=q, metric=prim,
                              comparison="vs baseline")
                    for k in ("n_pairs", "mean", "ci_mean_lo", "ci_mean_hi", "median",
                              "ci_median_lo", "ci_median_hi", "n_better"):
                        row[f"{k}_q{q}"] = pr[k] if pr else np.nan
                    if wave == "stage2":
                        sg = _one(data[wave]["paired"], arm=arm, baseline=base, q=q,
                                  metric="stage2_peak_rel_err_signed", comparison="vs baseline")
                        row[f"signed_mean_q{q}"] = sg["mean"] if sg else np.nan
                        row[f"signed_ci_lo_q{q}"] = sg["ci_mean_lo"] if sg else np.nan
                        row[f"signed_ci_hi_q{q}"] = sg["ci_mean_hi"] if sg else np.nan
                    row[f"criterion_a_q{q}"] = cr[f"a_q{q}"] if cr else np.nan
                    mm = ("stage1_rel_err_signed" if wave == "stage1"
                          else "stage2_peak_rel_err_signed")
                    dp = _one(data[wave]["dispersion"], arm=arm, baseline=base, q=q, metric=mm)
                    for k in ("sd_arm", "sd_base", "ratio", "ci_lo", "ci_hi"):
                        row[f"disp_{k}_q{q}"] = dp[k] if dp else np.nan
                row["criterion_a_met"] = cr["a_met"] if cr else np.nan
                row["criterion_a_complete"] = cr["a_complete"] if cr else np.nan
                row["criterion_b_status"] = cr["b_status"] if cr else np.nan
                row["criterion_b_n_violations"] = cr["b_n_violations"] if cr else np.nan
                row["criterion_b_violations"] = cr["b_violations"] if cr else np.nan
                row["criterion_b_n_pending"] = cr["b_n_pending"] if cr else np.nan
                row["criterion_overall"] = cr["overall"] if cr else np.nan
                co = _one(data[wave]["cost"], arm=arm)
                cb = _one(data[wave]["cost"], arm=base)
                for k in ("mean_phase_wall_sec", "phase_wall_ratio_vs_base",
                          "mean_phase_local_updates", "wall_per_update_ratio_vs_base",
                          "wall_ratio_vs_matched_control", "mean_phase_episodes",
                          "mean_n_minibatch_steps_total", "mean_total_wall_sec", "n_runs"):
                    row[f"cost_{k}"] = co[k] if co else np.nan
                row["cost_base_mean_phase_wall_sec"] = cb["mean_phase_wall_sec"] if cb else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


D1_COLS = ("file", "group", "q", "phase", "n_runs", "M1_median_over_runs", "M1_threshold",
           "M1_outcome", "M2_runs_share_above_threshold", "M2_runs_needed_more_than",
           "M2_max_share", "M2_outcome")


def d1_tables(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """D1 flags: the locked-pipeline headline rows and the pooled (q == all) rows of every arm.

    Reads ``<root>/d1_clamp/d1_flags.csv`` (the final D1 output; ``report: 02_d1_clamp.md``).
    The headline is ``group == v11_reproduction`` (q = 50, 60 and the pooled q = all); the
    per-arm rows are ``group == stage1/<arm>`` or ``stage2/<arm>`` with q = all.
    """
    p = Path(root) / "d1_clamp" / "d1_flags.csv"
    if not p.exists():
        miss = pd.DataFrame([{"file": _rel(p), "present": False}])
        return miss, miss
    d = pd.read_csv(p)
    d.insert(0, "file", _rel(p))
    d = d[[c for c in D1_COLS if c in d.columns]]
    head = d[d["group"] == "v11_reproduction"]
    arms = d[(d["group"] != "v11_reproduction") & (d["q"].astype(str) == "all")]
    return head.reset_index(drop=True), arms.reset_index(drop=True)


def d1_arm_text(arms: pd.DataFrame, stage: str, arm: str) -> str:
    """``arm: M1 .. (median ..), M2 ..`` from the pooled D1 row of one arm (or ``no D1 row``)."""
    if "group" not in arms:
        return f"`{arm}`: D1 file not present"
    r = arms[arms["group"] == f"{stage}/{arm}"]
    if r.empty:
        return f"`{arm}`: no D1 row"
    r = r.iloc[0]
    return (f"`{arm}` (phase {r['phase']}): M1 {r['M1_outcome']} "
            f"(median {_g(r['M1_median_over_runs'])}), M2 {r['M2_outcome']} "
            f"({_g(r['M2_runs_share_above_threshold'])} runs above)")


def d2_table(root: Path) -> pd.DataFrame:
    """Detection limits (final tier for G-F / G-A, dev-final for G-N; side ``any``), per family."""
    p = Path(root) / "d2_verifier_sensitivity" / "detection_limits.csv"
    if not p.exists():
        return pd.DataFrame([{"file": _rel(p), "present": False}])
    d = pd.read_csv(p)
    d = d[(d["side"] == "any") & (((d["tier"] == "final") & d["criterion"].isin(["G-F", "G-A"]))
                                  | ((d["tier"] == "dev-final")
                                     & d["criterion"].isin(["G-N_Gmax", "G-N_eta"])))]
    rows = []
    for (fam, crit, tier), g in d.groupby(["family", "criterion", "tier"], sort=False):
        r: Dict[str, Any] = {"family": fam, "criterion": crit, "tier": tier,
                             "perturbation": g["perturbation_name"].iloc[0]}
        for q in sorted(g["q"].unique()):
            gq = g[g.q == q].iloc[0]
            r[f"q{q} detection limit"] = gq["limit_str"]
            r[f"q{q} max |perturbation| on grid"] = gq["max_abs_pert_on_grid"]
            r[f"q{q} max value on grid"] = gq["max_value_on_grid"]
        rows.append(r)
    out = pd.DataFrame(rows)
    out.insert(0, "file", _rel(p))
    return out


def continuation_table(root: Path) -> pd.DataFrame:
    """Check (ii) of the continuation table against the verifier (``continuation_check.json``)."""
    p = Path(root) / "continuation_check.json"
    if not p.exists():
        return pd.DataFrame([{"file": _rel(p), "present": False}])
    j = json.loads(p.read_text())
    rows = []
    for tier in ("final", "development"):
        for qk, v in j["table_vs_verifier"][tier].items():
            rows.append({"file": _rel(p), "tier": tier, "q": qk[1:],
                         "max_abs_diff_over_dw": v["max_abs_diff_over_dw"],
                         "spec_tolerance_over_dw": v.get("spec_tolerance_over_dw", 1e-6),
                         "meets_spec_tolerance": v.get("meets_spec_tolerance"),
                         "refined_verifier (state step 0.25, 64 GL nodes) max abs diff / DW":
                             (j.get("diagnosis", {}).get("verifier_grid_sweep_max_abs_diff_over_dw",
                                                         {}).get(qk, {}).get("state_step_0.25", {})
                              .get("gl_half_64") if tier == "final" else np.nan)})
    return pd.DataFrame(rows)


def write_decision_report(args: argparse.Namespace) -> Path:
    """Generate ``06_decision_inputs.md`` from the wave CSVs (plus D1, D2, check-(ii) files)."""
    out, root, qs = Path(args.out), Path(args.root), tuple(args.qs)
    doc = Doc(out)
    long = decision_long(out, qs)
    long.to_csv(out / "decision_inputs.csv", index=False)
    doc.h(1, "R1 decision inputs: the six methods side by side")
    doc.p("Generated by `tools/v2/refine_analysis.py decision` from the wave tables written by the "
          "`stage1` and `stage2` sub-commands (`" + _rel(out) + "/stage1_*.csv`, `stage2_*.csv`); "
          "no number is typed by hand and none is recomputed here. All differences are "
          "arm - baseline paired by (q, seed) (stage 1: arm - `B_base`; stage 2: arm - `A_base`); "
          "the primary metric is `|e1_hat(0) - e1*|/e1*` for stage-1 arms and the absolute signed "
          "peak error for stage-2 arms (improvement = negative difference); the confidence "
          "intervals are the pre-registered 95% percentile bootstrap intervals "
          f"({N_BOOT} resamples, `default_rng({BOOT_SEED})`). Criterion part (a): the CI of the "
          "mean difference is below 0 at both q; part (b) (the gate part): no run that passed its "
          "gate under the baseline fails it under the arm. The criterion is descriptive, not a "
          "gate. Dispersion ratio = across-seed SD of the signed error of the arm over the "
          "baseline's with a paired-resampling CI.")
    d1, d1_arms = d1_tables(root)
    d2 = d2_table(root)
    d1_path = _rel(root / "d1_clamp" / "d1_flags.csv")
    d1_head = "D1 file not present"
    if "M1_outcome" in d1:
        allr = d1[d1["q"].astype(str) == "all"]
        d1_head = "locked pipeline (v11_reproduction, q = all): " + "; ".join(
            f"phase {r['phase']}: M1 {r['M1_outcome']}, M2 {r['M2_outcome']}"
            for r in allr.to_dict("records"))
    d2_txt = ("D2 (`" + _rel(root / 'd2_verifier_sensitivity' / 'detection_limits.csv')
              + "`): see section 4") if "family" in d2 else "D2 file not present"
    # ---- one row per method
    doc.h(2, "1. One row per method")
    rows = []
    for method, d in METHOD_ARMS.items():
        sub = long[long.method == method]
        r: Dict[str, Any] = {"method": method}
        r["arms"] = "<br>".join(
            f"`{a['arm']}` (" + ("stage 1" if a["stage"] == "stage1" else "stage 2") + ")"
            for a in sub.to_dict("records"))
        for q in qs:
            cells = []
            for a in sub.to_dict("records"):
                cim = ci_str(a[f"ci_mean_lo_q{q}"], a[f"ci_mean_hi_q{q}"])
                cid = ci_str(a[f"ci_median_lo_q{q}"], a[f"ci_median_hi_q{q}"])
                s = (f"`{a['arm']}`: mean {_g(a[f'mean_q{q}'])} {cim}; median "
                     f"{_g(a[f'median_q{q}'])} {cid}"
                     f"; n_better {_g(a[f'n_better_q{q}'])}/{_g(a[f'n_pairs_q{q}'])}")
                if a["stage"] == "stage2":
                    s += f"; signed mean {_g(a[f'signed_mean_q{q}'])}"
                cells.append(s)
            r[f"primary paired diff, q={q}"] = "<br>".join(cells)
        r["criterion (a): CI < 0 at both q"] = "<br>".join(
            f"`{a['arm']}`: {a['criterion_a_met']} ("
            + ", ".join(f"q{q} {a[f'criterion_a_q{q}']}" for q in qs) + ")"
            for a in sub.to_dict("records"))
        r["gate part (b)"] = "<br>".join(
            f"`{a['arm']}`: {a['criterion_b_status']} ({_g(a['criterion_b_n_violations'])} "
            f"violations)" for a in sub.to_dict("records"))
        r["dispersion ratio SD(signed error) arm/baseline [95% CI]"] = "<br>".join(
            f"`{a['arm']}`: " + "; ".join(
                f"q{q} {_g(a[f'disp_ratio_q{q}'])} "
                f"{ci_str(a[f'disp_ci_lo_q{q}'], a[f'disp_ci_hi_q{q}'])}" for q in qs)
            for a in sub.to_dict("records"))
        r["compute cost per run"] = "<br>".join(
            f"`{a['arm']}`: phase wall {_g(a['cost_mean_phase_wall_sec'])} s over "
            f"{_g(a['cost_mean_phase_local_updates'])} updates "
            f"(x{_g(a['cost_phase_wall_ratio_vs_base'])} of `{a['baseline']}`, per update "
            f"x{_g(a['cost_wall_per_update_ratio_vs_base'])}), episodes "
            f"{_g(a['cost_mean_phase_episodes'])}, optimizer steps "
            f"{_g(a['cost_mean_n_minibatch_steps_total'])}" for a in sub.to_dict("records"))
        d1_cells = [d1_arm_text(d1_arms, a["stage"], a["arm"]) for a in sub.to_dict("records")]
        r["D1 / D2 outcomes"] = ("D1 (`" + d1_path + "`), pooled over both q: "
                                 + "<br>".join(d1_cells) + f"<br>{d1_head}.<br>{d2_txt}.")
        rows.append(r)
    wide = pd.DataFrame(rows)
    doc.table(long, "decision_inputs", disp=wide,
              note="one row per (method, arm) in the CSV; the table cells list the arms of a "
                   "method")
    doc.h(2, "2. Method 5: the pair against its matched control and the parent candidate")
    doc.p("Method 5: `A_ctrl200` and `A_detmean` are compared with `A_base` in this table like "
          "every arm, but `A_detmean` is an ablation (pathwise, model-based, not eligible for the "
          "locked protocol) and its matched control is `A_ctrl200` (200 further PPO updates at "
          "constant LR 3e-5 from the same state); the paired comparison with the control follows.")
    s2p = _read(out, "stage2_paired.csv")
    ab = s2p[(s2p.get("comparison", pd.Series(dtype=str)).isin(
        ["ablation vs matched control", "vs parent u1600 candidate"]))] if len(s2p) else s2p
    abd = (paired_disp(ab[ab.metric.isin([PRIMARY["stage2"], "stage2_peak_rel_err_signed",
                                          "stage2_peak_locfree_rel_err_abs"])]
                       .assign(arm=lambda x: x.arm + " - " + x.baseline), [PRIMARY["stage2"],
                       "stage2_peak_rel_err_signed", "stage2_peak_locfree_rel_err_abs"],
                       with_arm=True) if len(ab) else pd.DataFrame())
    doc.table(ab, "decision_method5_pairs", disp=abd,
              note="rows: comparison in (ablation vs matched control, vs parent u1600 candidate)")
    doc.h(2, "3. Method 6: check (ii) of the continuation table")
    doc.p("Method 6 (`B_expcont`): check (ii) of the continuation table against the verifier "
          "(`tests/test_v2_refine_continuation.py --write`). The literal final-tier criterion "
          "(maximum absolute difference <= 1e-6 DW) is NOT met on the verifier's standard final "
          "grid; on the refined verifier configuration listed in the last column it is. The PI "
          "decided to keep method 6 in the round (`01_preregistration.md` section 4 item 6).")
    ct = continuation_table(root)
    doc.table(ct, "decision_method6_check_ii", disp=ct)
    doc.h(2, "4. D1 / D2 headline values")
    doc.p("D1 materiality flags (M1 = median over runs of the per-phase clamp fraction among "
          "learner policy rows above its threshold; M2 = more than the stated number of runs "
          "with a clamped-row gradient share above 1%; the final D1 report is "
          "`reports/v2/refine/02_d1_clamp.md`). Headline: the locked pipeline (20 C-R1 runs, "
          "`group == v11_reproduction`; q = 50, 60 and pooled):")
    doc.table(d1, "decision_d1_flags", disp=d1)
    doc.p("Pooled rows (q = all) of every pilot arm of this round, as cited in the method "
          "table above:")
    doc.table(d1_arms, "decision_d1_flags_arms", disp=d1_arms)
    doc.p("D2 detection limits (smallest perturbation at which G-F / G-A / G-N is violated; "
          "`not reached on the grid` otherwise):")
    doc.table(d2, "decision_d2_detection_limits", disp=d2)
    doc.lines += ["", OBS_HEADING, "", OBS_PLACEHOLDER, ""]
    path = Path(args.reports) / "06_decision_inputs.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc.text())
    return path


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    """Command-line parser."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    p.add_argument("cmd", choices=("stage1", "stage2", "decision", "all"))
    p.add_argument("--root", default=str(REPO / "results" / "v2_refine"))
    p.add_argument("--out", default=str(REPO / "results" / "v2_refine" / "analysis"))
    p.add_argument("--reports", default=str(REPO / "reports" / "v2" / "refine"))
    p.add_argument("--figures", default=None, help="default: <reports>/figures")
    p.add_argument("--ref-root", default=str(REHEARSAL),
                   help="rehearsal_v1_1 reference (gates.json, induced_band.json, state_end_A.pt)")
    p.add_argument("--repo-root", default=None, help="repository to import the study code from")
    p.add_argument("--qs", type=int, nargs="+", default=list(QS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--arms", nargs="+", default=None,
                   help="restrict the arms (the baseline arm is always kept); for tests")
    p.add_argument("--workers", type=int, default=4, help="processes for the per-run extraction")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run ``stage1`` / ``stage2`` / ``decision`` / ``all``; returns 0."""
    a = build_parser().parse_args(argv)
    a.figures = a.figures or str(Path(a.reports) / "figures")
    for k in ("root", "out", "reports", "figures", "ref_root"):
        setattr(a, k, str(Path(getattr(a, k)).resolve()))
    waves = {"stage1": ["stage1"], "stage2": ["stage2"], "decision": [],
             "all": ["stage1", "stage2"]}[a.cmd]
    for w in waves:
        wr = compute_wave(w, a)
        make_figures(wr, Path(a.figures))
        print(f"wrote {write_wave_report(wr, a, w)}", flush=True)
    if a.cmd in ("decision", "all"):
        print(f"wrote {write_decision_report(a)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
