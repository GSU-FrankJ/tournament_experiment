#!/usr/bin/env python3
"""D2: sensitivity of the DP-BR verifier to mild policy misspecification (T57 issue 8).

Pure evaluation, no training. Candidates are deterministic ``MeanPolicy`` callables built from the
closed-form T=2 equilibrium (``utils.theory_multistage.g1_two_stage`` / ``g2_two_stage``; the
closed form is allowed here because this is the evaluation side) with one controlled perturbation,
evaluated with ``utils.v2_metrics.evaluate`` on BOTH verifier tiers (development, final) and both q.

Families (grids exactly as in PROMPT.md section 3.2):
  (a) stage-1 scalar     e1 = (1 + d) e1*, stage 2 exact;                d in DELTA_GRID
  (b) stage-2 amplitude  e2(d) = (1 + d) e2*(d), stage 1 exact;          d in DELTA_GRID
  (c) stage-2 rounding   e2 = e2* convolved with a uniform kernel of half-width h (d units),
                         evaluated on a 0.01 grid and linearly interpolated; stage 1 exact;
                         h in KERNEL_H
  (d) stage-2 tail       e2(d) = e2*(d) + tau for |d| >= 2q, stage 1 exact; tau in TAU_GRID
  (e) RL-like            (a) with d in E_DELTAS combined with the (c) kernel whose induced peak
                         error is closest to KERNEL_TARGET, chosen per q from the (c) results of
                         the same run (the chosen h is recorded)

Clipping: every returned effort is passed through ``np.clip(e, e_min, e_max)`` because the verifier
rejects efforts outside [e_min, e_max]. The clip is applied at the output of the policy only. For
every candidate the UNCLIPPED stage-1 value and the unclipped stage-2 function on a dense 0.01 grid
over the whole stage-2 domain [-B, B] are inspected and the columns ``clip_binds`` /
``clip_raw_min`` / ``clip_raw_max`` record whether the clip changes any value (expected not to
bind on the pre-registered grids: the largest effort is 1.15 * 70 = 80.5 < 100; verify in the
output).

Per candidate x tier x q the evaluation writes (evaluations.csv) the verifier quantities, the
recovery metrics, the perturbation parameters, the git commit and dirty flag and the wall time.
Derived files: paired.csv (dev and final side by side, dev - final differences), fits.csv
(least squares G ~ a x^2 through the origin), detection_limits.csv (G-F / G-A / G-N detection
limits), family_e_confirmation.csv, meta.json, figures/. ``--report`` turns these into the
markdown report. This tool is a sensitivity study of the verifier; it changes no threshold.

Usage:
  python tools/v2/verifier_sensitivity.py --out results/v2_refine/d2_verifier_sensitivity \
      --workers 8
  python tools/v2/verifier_sensitivity.py --out <same dir> --report \
      [--report-path reports/v2/refine/03_d2_verifier_sensitivity.md]
  python tools/v2/verifier_sensitivity.py --out results/v2_refine/d2_smoke --smoke
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from envs.curriculum_env import GameSpec  # noqa: E402
from utils.dp_br_verifier import (  # noqa: E402
    DEV_CONFIG,
    FINAL_CONFIG,
    DomainError,
    VerifierConfig,
)
from utils.theory_multistage import g1_two_stage, g2_two_stage  # noqa: E402
from utils.v2_metrics import evaluate  # noqa: E402

PROTOCOL = ROOT / "protocols" / "v2_T2_locked_v1_1.json"
CANONICAL = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/"
                 "pilot-4-stabilization-fb99a2")
CONFIRMATION_DIR = CANONICAL / "results" / "v2_T2_locked" / "confirmation_analysis"
DEFAULT_REPORT = ROOT / "reports" / "v2" / "refine" / "03_d2_verifier_sensitivity.md"

FAMILIES = ("a", "b", "c", "d", "e")
_D = (0.005, 0.01, 0.02, 0.05, 0.10, 0.15)
DELTA_GRID: Tuple[float, ...] = tuple(sorted([0.0] + [s * v for v in _D for s in (1.0, -1.0)]))
KERNEL_H: Tuple[float, ...] = (2.0, 5.0, 10.0, 12.0, 20.0)
TAU_GRID: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
E_DELTAS: Tuple[float, ...] = (0.05, 0.10, 0.15)
KERNEL_TARGET = -0.06            # confirmation median of the stage-2 peak error (both q)
KERNEL_GRID_STEP = 0.01          # d units
TIERS = {"development": DEV_CONFIG, "final": FINAL_CONFIG}
TIER_SUFFIX = {"development": "dev", "final": "final"}
NOT_REACHED = "not reached on the grid"
FAMILY_TITLE = {
    "a": "(a) stage-1 scalar",
    "b": "(b) stage-2 amplitude",
    "c": "(c) stage-2 peak rounding (uniform kernel)",
    "d": "(d) stage-2 tail offset",
    "e": "(e) RL-like: stage-1 scalar + peak-rounding kernel",
}
PERT_NAME = {"a": "delta_stage1", "b": "delta_stage2", "c": "kernel_h", "d": "tau",
             "e": "delta_stage1"}
PERT_UNIT = {"a": "relative", "b": "relative", "c": "d units (half-width)",
             "d": "effort units", "e": "relative"}
SIGNED_FAMILIES = ("a", "b")
METRICS = (("Gmax_full_over_dw", "G-F"), ("eta_T_over_dw", "G-A"))
# eta_2 depends on the stage-2 policy only: families (a) and (e) leave it unchanged along their
# perturbation (for (e) it equals the kernel-only value), so a ~ x^2 fit of eta_2 is not a model
# of anything there. The fit is still written (nothing is hidden) but flagged.
ETA_INDEPENDENT_OF_X = ("a", "e")
# Presentation rule for the descriptive extrapolation sqrt(threshold / a): when every fitted
# response is at or below this value (in units of DW) the response is the verifier's numerical
# floor (the d = 0 control rows of evaluations.csv give the measured floors) and the extrapolated
# perturbation would be meaningless.
FLOOR_FIT = 1e-9


# ---------------------------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """One perturbed candidate; absent perturbations are 0.

    Attributes:
        family: One of ``a`` .. ``e``.
        delta1: Relative stage-1 scale error (families a, e).
        delta2: Relative stage-2 amplitude error (family b).
        h: Uniform-kernel half-width in d units (families c, e); 0 = no kernel.
        tau: Tail offset in effort units for ``|d| >= 2q`` (family d).
    """

    family: str
    delta1: float = 0.0
    delta2: float = 0.0
    h: float = 0.0
    tau: float = 0.0

    @property
    def pert_name(self) -> str:
        """Name of the primary perturbation parameter of the family."""
        return PERT_NAME[self.family]

    @property
    def pert(self) -> float:
        """Primary perturbation value (signed)."""
        return {"a": self.delta1, "b": self.delta2, "c": self.h, "d": self.tau,
                "e": self.delta1}[self.family]

    @property
    def cand_id(self) -> str:
        """Stable identifier, e.g. ``a_d+0.05``, ``c_h12``, ``e_d+0.1_h12``."""
        f = self.family
        if f in ("a", "b"):
            return f"{f}_d{self.pert:+g}"
        if f == "c":
            return f"c_h{self.h:g}"
        if f == "d":
            return f"d_tau{self.tau:g}"
        return f"e_d{self.delta1:+g}_h{self.h:g}"


def family_candidates(family: str, kernel_h: Optional[float] = None) -> List[Candidate]:
    """Candidates of one family on the pre-registered grid, in ascending perturbation order.

    Args:
        family: ``a`` .. ``e``.
        kernel_h: Required for family ``e``: the chosen kernel half-width.

    Returns:
        List of candidates.
    """
    if family == "a":
        return [Candidate("a", delta1=d) for d in DELTA_GRID]
    if family == "b":
        return [Candidate("b", delta2=d) for d in DELTA_GRID]
    if family == "c":
        return [Candidate("c", h=h) for h in KERNEL_H]
    if family == "d":
        return [Candidate("d", tau=t) for t in TAU_GRID]
    if family == "e":
        if kernel_h is None:
            raise ValueError("family e needs the chosen kernel half-width")
        return [Candidate("e", delta1=d, h=float(kernel_h)) for d in E_DELTAS]
    raise ValueError(f"unknown family {family!r}")


def box_average(values: np.ndarray, n_h: int, step: float) -> np.ndarray:
    """Centered moving average over a window of half-width ``n_h`` samples (uniform kernel).

    The window integral is the trapezoid rule on the sample grid, so it is exact for a function
    that is piecewise linear with kinks on the grid. A constant maps to itself.

    Args:
        values: Samples on a uniform grid, length ``L >= 2 n_h + 1``.
        n_h: Half-window in samples (0 returns a copy).
        step: Grid spacing.

    Returns:
        Array of length ``L - 2 n_h``; element m is the mean over the window centred at sample
        ``m + n_h``.
    """
    v = np.asarray(values, dtype=float)
    if n_h == 0:
        return v.copy()
    if v.size < 2 * n_h + 1:
        raise ValueError("series shorter than the window")
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (v[:-1] + v[1:])) * step])
    return (cum[2 * n_h:] - cum[:-2 * n_h]) / (2.0 * n_h * step)


def kernel_table(spec: GameSpec, h: float, step: float = KERNEL_GRID_STEP
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """g2* convolved with a uniform kernel of half-width ``h``, tabulated on a ``step`` grid.

    The table covers the whole stage-2 domain [-B, B]; g2* is sampled on the grid extended by
    ``h`` on each side (it vanishes there). The table is symmetrised (0.5 (v + v[::-1])) so that
    it is exactly even, as the kernel and g2* are.

    Args:
        spec: Game parameters.
        h: Kernel half-width in d units; must be a multiple of ``step``.
        step: Grid spacing.

    Returns:
        ``(x, values)`` with x = i * step, |i| <= N, N = ceil(B / step).
    """
    n_h = int(round(h / step))
    if n_h < 1 or abs(h - n_h * step) > 1e-9:
        raise ValueError(f"kernel half-width {h} is not a positive multiple of {step}")
    big_n = int(math.ceil(spec.domain_half(2) / step - 1e-9))
    idx = np.arange(-(big_n + n_h), big_n + n_h + 1)
    g = g2_two_stage(idx * step, spec.q, spec.w_h, spec.w_l, spec.k, spec.e_max)
    vals = box_average(g, n_h, step)
    vals = 0.5 * (vals + vals[::-1])
    return np.arange(-big_n, big_n + 1) * step, vals


class PerturbedPolicy:
    """Deterministic ``policy(t, d) -> float64 effort`` of one perturbed candidate.

    Stage 1 is queried at the single point d = 0 (the output is constant in d); stage 2 at the
    verifier / recovery grid nodes. Outputs are clipped to [e_min, e_max] (see module docstring);
    ``clip_report`` states whether the clip ever changes a value.
    """

    def __init__(self, spec: GameSpec, cand: Candidate, step: float = KERNEL_GRID_STEP) -> None:
        """Build the candidate.

        Args:
            spec: Game parameters (T = 2).
            cand: Perturbation.
            step: Kernel table spacing.
        """
        if spec.T != 2:
            raise ValueError("D2 candidates are defined for T = 2")
        self.spec, self.cand = spec, cand
        self.g1 = float(g1_two_stage(spec.q, spec.w_h, spec.w_l, spec.k))
        self._tab = kernel_table(spec, cand.h, step) if cand.h > 0.0 else None

    def _base2(self, d: np.ndarray) -> np.ndarray:
        """Unperturbed (or kernel-smoothed) stage-2 effort."""
        s = self.spec
        if self._tab is None:
            return g2_two_stage(d, s.q, s.w_h, s.w_l, s.k, s.e_max)
        x, v = self._tab
        if d.size and float(np.max(np.abs(d))) > x[-1] + 1e-9:
            raise ValueError("stage-2 query outside the tabulated domain")
        return np.interp(d, x, v)

    def raw(self, t: int, d: np.ndarray) -> np.ndarray:
        """Candidate effort before the [e_min, e_max] clip."""
        d = np.asarray(d, dtype=float)
        if t == 1:
            return np.full(d.shape, (1.0 + self.cand.delta1) * self.g1)
        if t != 2:
            raise ValueError(f"stage {t} is not defined for a T = 2 candidate")
        e = (1.0 + self.cand.delta2) * self._base2(d)
        if self.cand.tau != 0.0:
            e = e + self.cand.tau * (np.abs(d) >= 2.0 * self.spec.q)
        return e

    def __call__(self, t: int, d: np.ndarray) -> np.ndarray:
        """Clipped effort, float64, same shape as ``d``."""
        return np.clip(self.raw(t, d), self.spec.e_min, self.spec.e_max)

    def clip_report(self, step: float = KERNEL_GRID_STEP) -> Dict[str, object]:
        """Whether the clip binds anywhere (dense grid over the whole stage-2 domain + stage 1)."""
        half = self.spec.domain_half(2)
        n = int(round(half / step))
        raw = np.concatenate([self.raw(2, np.arange(-n, n + 1) * step), self.raw(1, np.zeros(1))])
        lo, hi = float(raw.min()), float(raw.max())
        return {"clip_binds": bool(lo < self.spec.e_min or hi > self.spec.e_max),
                "clip_raw_min": lo, "clip_raw_max": hi}


# ---------------------------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------------------------
def git_state() -> Dict[str, object]:
    """Commit and dirty flag; dirty = tracked changes or untracked files outside results/."""
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--", ".", ":(exclude)results"],
            stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        commit, dirty = "unknown", None
    return {"commit": commit, "short": commit[:7], "dirty": dirty}


def sha256_of(path: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: os.PathLike) -> str:
    """Path relative to the repository root when inside it, else absolute."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------
EVAL_SCALARS = (
    "valid", "invalid_reasons",
    "Gmax_full_over_dw", "Gmax_full_t", "Gmax_full_d", "eta_T_over_dw", "EXP_root_over_dw",
    "dReach_over_dw", "Deltamax_all_over_dw", "dFull_over_dw",
    "stage1_rel_err_signed", "stage2_peak_rel_err_signed", "stage2_rmse_pos_over_g2_0",
    "stage2_tail_mean_over_g2_0", "e1_at_0", "e2_at_0", "g1", "g2_at_0",
)


def evaluate_task(task: Dict[str, object]) -> Dict[str, object]:
    """Evaluate one (candidate, q, tier); module-level so it can run in a process pool.

    Args:
        task: Dict with ``cand`` (Candidate), ``game`` (GameSpec fields), ``tier`` (name),
            ``recovery_step``, ``commit``, ``dirty``.

    Returns:
        One flat row for evaluations.csv. Verifier errors (DomainError, FloatingPointError,
        ValueError) are recorded with ``status`` = ``error: ...`` and no metric columns.
    """
    t_task = time.perf_counter()
    cand: Candidate = task["cand"]
    spec = GameSpec(**task["game"])
    tier: VerifierConfig = TIERS[task["tier"]]
    pol = PerturbedPolicy(spec, cand)
    row: Dict[str, object] = {
        "family": cand.family, "cand_id": cand.cand_id, "perturbation_name": cand.pert_name,
        "perturbation": cand.pert, "delta_stage1": cand.delta1, "delta_stage2": cand.delta2,
        "kernel_h": cand.h, "tau": cand.tau, "q": int(spec.q), "tier": tier.name,
        "state_step": tier.state_step, "effort_step": tier.effort_step, "gl_half": tier.gl_half,
    }
    row.update(pol.clip_report())
    t0 = time.perf_counter()
    try:
        ev = evaluate(pol, spec, tier, beta_fn=None, recovery_step=float(task["recovery_step"]))
        row["status"] = "ok"
        sc = ev.scalars
        row.update({k: sc[k] for k in EVAL_SCALARS})
    except (DomainError, FloatingPointError, ValueError) as exc:
        row["status"] = f"error: {type(exc).__name__}: {exc}"
    row["eval_wall_sec"] = time.perf_counter() - t0
    row["task_wall_sec"] = time.perf_counter() - t_task
    row["commit"], row["dirty"] = task["commit"], task["dirty"]
    return row


def select_kernel(c_rows: Sequence[Dict[str, object]], target: float = KERNEL_TARGET
                  ) -> Dict[str, object]:
    """Kernel half-width whose induced peak error is closest to ``target``.

    Args:
        c_rows: Rows (or dicts) of the family-(c) evaluations of ONE q with ``kernel_h`` and
            ``stage2_peak_rel_err_signed``.
        target: Target signed peak error.

    Returns:
        Dict with ``h``, ``peak_err``, ``distance`` and the full ``table`` {h: peak_err}; ties go
        to the smaller h.
    """
    tab: Dict[float, float] = {}
    for r in c_rows:
        tab.setdefault(float(r["kernel_h"]), float(r["stage2_peak_rel_err_signed"]))
    if not tab:
        raise ValueError("no family-(c) results to choose the kernel from")
    h = min(sorted(tab), key=lambda x: abs(tab[x] - target))
    return {"h": h, "peak_err": tab[h], "distance": abs(tab[h] - target), "table": tab,
            "target": target}


def _run_tasks(tasks: List[Dict[str, object]], workers: int) -> List[Dict[str, object]]:
    """Run tasks inline (workers <= 1) or in a process pool (each process single-threaded)."""
    if workers <= 1 or len(tasks) <= 1:
        return [evaluate_task(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
        return list(ex.map(evaluate_task, tasks, chunksize=1))


def run_evaluations(qs: Sequence[int], tiers: Sequence[str], families: Sequence[str],
                    workers: int, proto: Dict, git: Dict[str, object],
                    only: Optional[Sequence[str]] = None
                    ) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    """Evaluate every requested candidate on every q and tier.

    Families a-d are evaluated first; family e then uses, per q, the (c) kernel whose peak error
    is closest to KERNEL_TARGET (from the (c) rows of this run; if (c) is not part of the run the
    peak errors are computed directly from the kernel candidates without the verifier).

    Args:
        qs: q values.
        tiers: Tier names.
        families: Families to run.
        workers: Process-pool size.
        proto: Locked protocol dict (games and recovery step).
        git: Output of :func:`git_state`.
        only: Optional list of candidate ids to keep (smoke).

    Returns:
        ``(rows, kernel_choice)``; ``kernel_choice[str(q)]`` is the :func:`select_kernel` dict
        plus ``source``.
    """

    def tasks_for(cands: Sequence[Candidate], q: int) -> List[Dict[str, object]]:
        rec = proto["records"][str(q)]
        return [{"cand": c, "game": rec["game"], "tier": t,
                 "recovery_step": rec["protocol"]["recovery_step"],
                 "commit": git["commit"], "dirty": git["dirty"]}
                for c in cands for t in tiers]

    keep_ids = None if only is None else set(only)

    def keep(c: Candidate) -> bool:
        return keep_ids is None or c.cand_id in keep_ids

    tasks: List[Dict[str, object]] = []
    for q in qs:
        for f in families:
            if f != "e":
                tasks += tasks_for([c for c in family_candidates(f) if keep(c)], q)
    rows = _run_tasks(tasks, workers)
    choice: Dict[str, Dict[str, object]] = {}
    if "e" in families:
        tasks_e: List[Dict[str, object]] = []
        for q in qs:
            c_rows = [r for r in rows if r["family"] == "c" and r["q"] == q
                      and r["status"] == "ok"]
            if c_rows:
                ch = select_kernel(c_rows)
                ch["source"] = "family (c) evaluation rows of this run"
            else:
                spec = GameSpec(**proto["records"][str(q)]["game"])
                direct = []
                for h in KERNEL_H:
                    pol = PerturbedPolicy(spec, Candidate("c", h=h))
                    g20 = float(g2_two_stage(np.zeros(1), spec.q, spec.w_h, spec.w_l, spec.k,
                                             spec.e_max)[0])
                    direct.append({"kernel_h": h, "stage2_peak_rel_err_signed":
                                   (float(pol(2, np.zeros(1))[0]) - g20) / g20})
                ch = select_kernel(direct)
                ch["source"] = "peak error computed directly (family (c) not in this run)"
            choice[str(q)] = ch
            tasks_e += tasks_for([c for c in family_candidates("e", ch["h"]) if keep(c)], q)
        rows += _run_tasks(tasks_e, workers)
    order = {f: i for i, f in enumerate(FAMILIES)}
    tier_order = {t: i for i, t in enumerate(TIERS)}
    rows.sort(key=lambda r: (order[r["family"]], r["perturbation"], r["q"],
                             tier_order[r["tier"]]))
    return rows, choice


# ---------------------------------------------------------------------------------------------
# Analysis helpers: paired table, fits, detection limits
# ---------------------------------------------------------------------------------------------
PAIR_COLS = ("valid", "Gmax_full_over_dw", "Gmax_full_t", "Gmax_full_d", "eta_T_over_dw",
             "EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw", "dFull_over_dw")
RECOVERY_COLS = ("stage1_rel_err_signed", "stage2_peak_rel_err_signed",
                 "stage2_rmse_pos_over_g2_0", "stage2_tail_mean_over_g2_0")


def abscissa(row: pd.Series) -> Tuple[str, float]:
    """Fit abscissa of a paired row: delta (a, b, e), induced peak error (c), tau (d)."""
    f = row["family"]
    if f == "c":
        return "induced_peak_error", float(row["stage2_peak_rel_err_signed"])
    if f == "d":
        return "tau", float(row["tau"])
    return "delta_stage1" if f in ("a", "e") else "delta_stage2", float(row["perturbation"])


def build_paired(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per (family, candidate, q): dev and final quantities side by side.

    Adds ``dev_minus_final_*`` (signed) and ``absdiff_*`` of Gmax_full/DW and eta_2/DW and the fit
    abscissa. Recovery metrics are tier independent and are taken from the first available tier.

    Args:
        ev: evaluations.csv as a DataFrame.

    Returns:
        Paired DataFrame (error rows are excluded here and counted in meta.json).
    """
    ok = ev[ev["status"] == "ok"]
    out = []
    for (fam, cid, q), g in ok.groupby(["family", "cand_id", "q"], sort=False):
        first = g.iloc[0]
        r: Dict[str, object] = {k: first[k] for k in (
            "family", "cand_id", "perturbation_name", "perturbation", "delta_stage1",
            "delta_stage2", "kernel_h", "tau", "q", "clip_binds")}
        for k in RECOVERY_COLS:
            r[k] = first[k]
        for tier, sfx in TIER_SUFFIX.items():
            gt = g[g["tier"] == tier]
            for k in PAIR_COLS:
                r[f"{k}_{sfx}"] = gt.iloc[0][k] if len(gt) else None
        if len(g["tier"].unique()) == 2:
            for k, nm in (("Gmax_full_over_dw", "Gmax"), ("eta_T_over_dw", "eta")):
                diff = float(r[f"{k}_dev"]) - float(r[f"{k}_final"])
                r[f"dev_minus_final_{nm}"] = diff
                r[f"absdiff_{nm}"] = abs(diff)
        name, x = abscissa(pd.Series(r))
        r["x_name"], r["x"] = name, x
        out.append(r)
    return pd.DataFrame(out)


def fit_through_origin(x: Sequence[float], y: Sequence[float]) -> Dict[str, object]:
    """Least squares y ~ a x^2 through the origin (points with x = 0 carry no information).

    Args:
        x: Abscissae.
        y: Responses.

    Returns:
        Dict: ``a``, ``n``, ``rmse_resid``, ``max_abs_resid`` (+ ``x_at_max_abs_resid``),
        ``max_abs_resid_over_ymax``, ``r2_uncentered`` = 1 - SSE / sum(y^2) and ``residuals``
        (y - a x^2 in input order, semicolon separated).

    Raises:
        ValueError: If no point has x != 0.
    """
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = xa != 0.0
    xa, ya = xa[m], ya[m]
    if xa.size == 0:
        raise ValueError("no point with x != 0")
    x2 = xa * xa
    a = float(np.sum(x2 * ya) / np.sum(x2 * x2))
    res = ya - a * x2
    j = int(np.argmax(np.abs(res)))
    ymax = float(np.max(np.abs(ya)))
    ss_y = float(np.sum(ya * ya))
    return {"a": a, "n": int(xa.size), "rmse_resid": float(np.sqrt(np.mean(res ** 2))),
            "max_abs_resid": float(abs(res[j])), "x_at_max_abs_resid": float(xa[j]),
            "max_abs_resid_over_ymax": float(abs(res[j]) / ymax) if ymax > 0 else float("nan"),
            "r2_uncentered": (1.0 - float(np.sum(res ** 2)) / ss_y) if ss_y > 0 else float("nan"),
            "residuals": ";".join(f"{v:.6e}" for v in res)}


def detection_limit(pert: Sequence[float], values: Sequence[float], threshold: float,
                    side: str = "any") -> Dict[str, object]:
    """Smallest |perturbation| on the grid at which ``value > threshold`` (strict).

    The unperturbed point (p = 0) belongs to every side. The limit is the first exceedance in
    ascending |p|; no monotonicity is assumed. ``reached`` and ``limit_str`` state explicitly when
    the threshold is not exceeded anywhere on the (side of the) grid; ``limit`` is then blank.

    Args:
        pert: Signed perturbations of the grid points.
        values: Metric value at each point.
        threshold: Detection threshold.
        side: ``pos`` (p >= 0), ``neg`` (p <= 0) or ``any``.

    Returns:
        Dict with ``reached``, ``limit`` (|p| or None), ``limit_str``, ``p_at_limit`` (signed p),
        ``value_at_limit``, ``bracket_lo`` (largest grid |p| below the limit on this side, or
        "none"), ``n_points_side``, ``max_abs_pert_on_grid``, ``max_value_on_grid``.
    """
    p, v = np.asarray(pert, dtype=float), np.asarray(values, dtype=float)
    keep = {"pos": p >= 0.0, "neg": p <= 0.0, "any": np.ones(p.shape, dtype=bool)}[side]
    p, v = p[keep], v[keep]
    order = np.argsort(np.abs(p), kind="stable")
    p, v = p[order], v[order]
    out: Dict[str, object] = {
        "n_points_side": int(p.size),
        "max_abs_pert_on_grid": float(np.max(np.abs(p))) if p.size else float("nan"),
        "max_value_on_grid": float(np.max(v)) if p.size else float("nan"),
    }
    hit = np.nonzero(v > threshold)[0]
    if hit.size == 0:
        out.update(reached=False, limit=None, limit_str=NOT_REACHED, p_at_limit=None,
                   value_at_limit=None, bracket_lo="none")
        return out
    j = int(hit[0])
    lo = [abs(x) for x in p[:j] if abs(x) < abs(p[j])]
    out.update(reached=True, limit=float(abs(p[j])), limit_str=f"{abs(p[j]):.6g}",
               p_at_limit=float(p[j]), value_at_limit=float(v[j]),
               bracket_lo=f"{max(lo):.6g}" if lo else "none")
    return out


def thresholds(proto: Dict) -> Dict[str, float]:
    """Gate thresholds from the locked protocol (read only): G-F, G-A, G-N (Gmax and eta)."""
    g = proto["gates"]
    th = {c["metric"]: float(c["threshold"]) for blk in ("G-A", "G-F", "G-N")
          for c in g[blk]["all_must_hold"]}
    return {"G-F": th["Gmax_full_over_dw"], "G-A": th["eta_T_over_dw"],
            "G-N_Gmax": th["Gmax_full_over_dw_dev_minus_final_abs"],
            "G-N_eta": th["eta_T_over_dw_dev_minus_final_abs"]}


def sides_for(family: str) -> Tuple[str, ...]:
    """Perturbation sides reported separately for a family."""
    return ("neg", "pos", "any") if family in SIGNED_FAMILIES else ("any",)


def build_fits(paired: pd.DataFrame, th: Dict[str, float]) -> pd.DataFrame:
    """Per family, q, tier, metric and side: fit y ~ a x^2 through the origin.

    ``x`` is delta for (a), (b), (e), the induced peak error for (c) and tau for (d). Signed
    families get a pooled fit and separate negative / positive side fits. ``x_at_threshold_fit``
    is sqrt(threshold / a) of the fitted law (descriptive extrapolation; threshold G-F for the
    Gmax metric and G-A for eta); it is NaN, with the reason in ``fit_note``, when every fitted
    response is at the numerical floor (``FLOOR_FIT``) or ``a <= 0``. ``fit_note`` also flags the
    eta_2 fits of families (a) and (e), where eta_2 does not depend on x.
    """
    rows = []
    for (fam, q), g in paired.groupby(["family", "q"], sort=False):
        for tier, sfx in TIER_SUFFIX.items():
            for metric, crit in METRICS:
                col = f"{metric}_{sfx}"
                if g[col].isna().all():
                    continue
                for side in (("pooled", "neg", "pos") if fam in SIGNED_FAMILIES else ("all",)):
                    gg = g
                    if side == "neg":
                        gg = g[g["x"] < 0]
                    elif side == "pos":
                        gg = g[g["x"] > 0]
                    gg = gg[gg["x"] != 0]
                    if len(gg) == 0:
                        continue
                    f = fit_through_origin(gg["x"], gg[col].astype(float))
                    a = f["a"]
                    ymax = float(gg[col].astype(float).abs().max())
                    notes = []
                    f["x_at_threshold_fit"] = float("nan")
                    if ymax <= FLOOR_FIT:
                        notes.append(f"all fitted responses <= {FLOOR_FIT:g} (numerical floor): "
                                     "no extrapolation")
                    elif a > 0:
                        f["x_at_threshold_fit"] = math.sqrt(th[crit] / a)
                    else:
                        notes.append("a <= 0: no extrapolation")
                    if metric == "eta_T_over_dw" and fam in ETA_INDEPENDENT_OF_X:
                        notes.append("eta_2 does not depend on x in this family (stage 2 is "
                                     "unchanged along the perturbation): not a model")
                    f["fit_note"] = "; ".join(notes)
                    rows.append({"family": fam, "q": q, "tier": tier, "metric": metric,
                                 "fit_threshold": crit, "threshold": th[crit], "side": side,
                                 "x_name": g["x_name"].iloc[0], **f})
    return pd.DataFrame(rows)


def build_detection(paired: pd.DataFrame, th: Dict[str, float]) -> pd.DataFrame:
    """Detection limits per family, q, tier and criterion (G-F, G-A per tier; G-N on dev-final).

    Criteria: G-F (Gmax_full/DW > 0.01), G-A (eta_2/DW > 0.005), G-N_Gmax and G-N_eta
    (|dev - final| > 0.001, in units of DW). Signed families report the negative and positive
    side separately and ``any`` (smallest |p| over both sides).
    """
    rows = []
    for (fam, q), g in paired.groupby(["family", "q"], sort=False):
        specs = []
        for tier, sfx in TIER_SUFFIX.items():
            for metric, crit in METRICS:
                col = f"{metric}_{sfx}"
                if not g[col].isna().all():
                    specs.append((crit, tier, metric, g[col].astype(float).to_numpy()))
        if "absdiff_Gmax" in g and not g["absdiff_Gmax"].isna().all():
            specs.append(("G-N_Gmax", "dev-final", "|dev - final| of Gmax_full_over_dw",
                          g["absdiff_Gmax"].astype(float).to_numpy()))
            specs.append(("G-N_eta", "dev-final", "|dev - final| of eta_T_over_dw",
                          g["absdiff_eta"].astype(float).to_numpy()))
        for crit, tier, metric, vals in specs:
            for side in sides_for(fam):
                d = detection_limit(g["perturbation"].to_numpy(float), vals, th[crit], side)
                x_at = None
                if d["reached"]:
                    j = int(np.argmin(np.abs(g["perturbation"].to_numpy(float) - d["p_at_limit"])))
                    x_at = float(g["x"].iloc[j])
                rows.append({"family": fam, "q": q, "tier": tier, "criterion": crit,
                             "metric": metric, "threshold": th[crit], "side": side,
                             "perturbation_name": g["perturbation_name"].iloc[0],
                             "x_name": g["x_name"].iloc[0], "x_at_limit": x_at, **d})
    return pd.DataFrame(rows)


def _ratio(num: object, den: object) -> Optional[float]:
    """num / den, or None when either is missing or den is 0."""
    if num is None or den is None or pd.isna(num) or pd.isna(den) or float(den) == 0.0:
        return None
    return float(num) / float(den)


def build_family_e_confirmation(paired: pd.DataFrame, conf_dir: Path, n_top: int = 5
                                ) -> pd.DataFrame:
    """Family-(e) predictions next to the confirmation runs with the largest stage-1 errors.

    For each q the ``n_top`` confirmation runs with the largest |stage-1 error| (per_run.csv) are
    listed with their observed Gmax_full/DW (final and dev tier, from the end-of-B candidate) and
    the family-(e) prediction at the grid delta nearest to |error| (positive deltas only), the
    kernel-only (c) prediction and the peak error of the chosen kernel next to the run's own
    stage-2 peak error (reported_metrics.csv).

    Args:
        paired: paired.csv as a DataFrame (must contain family e).
        conf_dir: Directory with confirmation per_run.csv and reported_metrics.csv.
        n_top: Number of runs per q.

    Returns:
        DataFrame (empty when family e is absent).
    """
    pr = pd.read_csv(conf_dir / "per_run.csv")
    rm = pd.read_csv(conf_dir / "reported_metrics.csv")[
        ["q", "seed", "A_stage2_peak_rel_err_signed"]]
    pr = pr.merge(rm, on=["q", "seed"], how="left")
    rows = []
    for q, ge in paired[paired["family"] == "e"].groupby("q"):
        gc = paired[(paired["family"] == "c") & (paired["q"] == q)
                    & (paired["kernel_h"] == ge["kernel_h"].iloc[0])]
        top = pr[pr["q"] == q].assign(abs_err=lambda d: d["stage1_rel_err_signed"].abs())
        top = top.sort_values("abs_err", ascending=False).head(n_top)
        for _, r in top.iterrows():
            j = int(np.argmin(np.abs(ge["delta_stage1"].to_numpy(float) - r["abs_err"])))
            e = ge.iloc[j]
            rows.append({
                "q": int(q), "seed": int(r["seed"]),
                "stage1_rel_err_signed": r["stage1_rel_err_signed"],
                "observed_gmax_final": r["gmax_final"], "observed_gmax_dev": r["gmax_dev"],
                "observed_stage2_peak_rel_err_signed": r["A_stage2_peak_rel_err_signed"],
                "e_nearest_delta": e["delta_stage1"], "e_kernel_h": e["kernel_h"],
                "e_peak_rel_err_signed": e["stage2_peak_rel_err_signed"],
                "e_pred_gmax_final": e["Gmax_full_over_dw_final"],
                "e_pred_gmax_dev": e["Gmax_full_over_dw_dev"],
                "ratio_observed_over_pred_final": _ratio(r["gmax_final"],
                                                         e["Gmax_full_over_dw_final"]),
                "kernel_only_pred_gmax_final": (gc["Gmax_full_over_dw_final"].iloc[0]
                                                if len(gc) else None),
                "kernel_only_pred_gmax_dev": (gc["Gmax_full_over_dw_dev"].iloc[0]
                                              if len(gc) else None),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
def make_figures(paired: pd.DataFrame, th: Dict[str, float], fig_dir: Path) -> List[str]:
    """Log-log Gmax_full/DW and eta_2/DW against |perturbation|, one figure per family.

    Rows: metric; columns: q. Colour = tier (development blue, final orange), marker = side
    (up triangle positive, down triangle negative, circle one-sided families). The G-F / G-A
    thresholds are drawn as dotted lines. Values below 1e-12 are drawn at 1e-12 (the unperturbed
    values, including numerical floors, are in paired.csv / the report, not on a log axis).

    Returns:
        List of written file names (png and pdf per family).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
                                "axes.spines.top": False, "axes.spines.right": False,
                                "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
    colour = {"development": "#2a78d6", "final": "#eb6834"}
    marker = {"pos": "^", "neg": "v", "all": "o"}
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for fam in FAMILIES:
        g = paired[paired["family"] == fam]
        if g.empty:
            continue
        qs = sorted(g["q"].unique())
        fig, axes = plt.subplots(2, len(qs), figsize=(3.6 * len(qs) + 0.6, 6.0), squeeze=False)
        for ci, q in enumerate(qs):
            gq = g[g["q"] == q]
            for ri, (metric, crit) in enumerate(METRICS):
                ax = axes[ri][ci]
                for tier, sfx in TIER_SUFFIX.items():
                    col = f"{metric}_{sfx}"
                    if gq[col].isna().all():
                        continue
                    for side in (("pos", "neg") if fam in SIGNED_FAMILIES else ("all",)):
                        if side == "neg":
                            m = gq["perturbation"] < 0
                        else:
                            m = gq["perturbation"] > 0
                        gs = gq[m].sort_values("perturbation", key=lambda s: s.abs())
                        if gs.empty:
                            continue
                        ax.plot(gs["perturbation"].abs(), np.maximum(gs[col].astype(float), 1e-12),
                                marker=marker[side], ms=4.5, lw=1.2, color=colour[tier],
                                ls="--" if side == "neg" else "-",
                                label=f"{tier}" + ("" if side == "all" else f", {side}"))
                ax.axhline(th[crit], color="0.35", lw=0.9, ls=":")
                ax.text(0.02, th[crit], f" {crit} threshold {th[crit]:g}",
                        transform=ax.get_yaxis_transform(), va="bottom", fontsize=7, color="0.35")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.set_ylim(bottom=1e-12)
                ax.set_title(f"q = {q}", fontsize=9)
                ax.set_ylabel("Gmax_full / DW" if ri == 0 else "eta_2 / DW")
                ax.set_xlabel(f"|{PERT_NAME[fam]}| ({PERT_UNIT[fam]})")
                if ri == 0 and ci == 0:
                    ax.legend(frameon=False, fontsize=7)
        fig.suptitle(f"D2 family {FAMILY_TITLE[fam]}", fontsize=10)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.text(0.5, 0.008, "values below 1e-12 are drawn at 1e-12 (numerical floor); exact "
                 "values are in paired.csv", ha="center", fontsize=7, color="0.35")
        for ext in ("png", "pdf"):
            p = fig_dir / f"d2_family_{fam}.{ext}"
            fig.savefig(p, dpi=150)
            written.append(p.name)
        plt.close(fig)
    return written


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------
def fnum(x: object, kind: str = "e") -> str:
    """Format a number for the report ('e': 3 significant digits scientific, 'g': 5 significant)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, str):
        return x
    return f"{float(x):.3e}" if kind == "e" else f"{float(x):.5g}"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Markdown pipe table."""
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _load(out_dir: Path, name: str) -> Optional[pd.DataFrame]:
    p = out_dir / name
    return pd.read_csv(p) if p.exists() else None


def render_report(out_dir: Path, report_path: Path) -> str:
    """Render the markdown report from the CSVs in ``out_dir`` (descriptive; no thresholds change).

    Every table and every sentence with numbers names the CSV it was computed from; verdict
    strings (detection limits, "not reached on the grid") are copied from detection_limits.csv.

    Args:
        out_dir: Directory written by the evaluation run.
        report_path: Markdown destination (used for relative figure links).

    Returns:
        The markdown text (also written to ``report_path``).
    """
    out_dir, report_path = Path(out_dir), Path(report_path)
    meta = json.loads((out_dir / "meta.json").read_text())
    ev = pd.read_csv(out_dir / "evaluations.csv")
    paired = pd.read_csv(out_dir / "paired.csv")
    fits = _load(out_dir, "fits.csv")
    det = _load(out_dir, "detection_limits.csv")
    fe = _load(out_dir, "family_e_confirmation.csv")
    c_ev, c_pa = rel(out_dir / "evaluations.csv"), rel(out_dir / "paired.csv")
    c_fi, c_de = rel(out_dir / "fits.csv"), rel(out_dir / "detection_limits.csv")
    c_fe, c_me = rel(out_dir / "family_e_confirmation.csv"), rel(out_dir / "meta.json")
    th = meta["thresholds"]
    L: List[str] = []
    a = L.append
    qs, tiers = meta["qs"], meta["tiers"]
    a("# D2 - verifier sensitivity to mild policy misspecification\n")
    a("Descriptive report (T57 issue 8, PROMPT.md section 3.2). It is a sensitivity study of the "
      "DP-BR verifier on closed-form perturbations of the equilibrium; it changes no threshold, "
      "no gate and no protocol file. Every number below is read from the CSV named under the "
      "table or in the sentence; verdict strings are copied from the CSV.\n")
    a("## 1. Provenance and reproduction\n")
    a(md_table(["item", "value", "source"], [
        ["code commit", f"`{meta['commit']}`", f"`{c_me}`"],
        ["dirty flag (tracked changes or untracked files outside results/)", fnum(meta["dirty"]),
         f"`{c_me}`"],
        ["created", meta["created"], f"`{c_me}`"],
        ["protocol (read only)", f"`{meta['protocol']}` sha256 `{meta['protocol_sha256'][:16]}...`",
         f"`{c_me}`"],
        ["q values / tiers", f"{qs} / {tiers}", f"`{c_me}`"],
        ["families run", " ".join(meta["families"]), f"`{c_me}`"],
        ["evaluations (rows) / errors", f"{meta['n_evaluations']} / {meta['n_errors']}",
         f"`{c_me}`"],
        ["workers / total wall seconds", f"{meta['workers']} / {fnum(meta['total_wall_sec'], 'g')}",
         f"`{c_me}`"],
    ]))
    a("")
    a("Commands (from the repository root; single-threaded processes):\n")
    a("```bash")
    a("export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1")
    a(meta["command"])
    a(f"{meta['python']} tools/v2/verifier_sensitivity.py --out {rel(out_dir)} --report "
      f"--report-path {rel(report_path)}")
    a("```\n")
    a("Per-row commit and dirty flag are also columns of "
      f"`{c_ev}`; the figures are in `{rel(out_dir / 'figures')}/` (png and pdf).\n")

    a("## 2. What was evaluated\n")
    a("Candidates are deterministic `MeanPolicy` callables built from the closed-form T=2 "
      "equilibrium (`utils.theory_multistage.g1_two_stage`, `g2_two_stage`); stage 1 is queried "
      "at d = 0 only. Each candidate is evaluated with `utils.v2_metrics.evaluate` on the "
      "development and the final verifier tier (`utils.dp_br_verifier.DEV_CONFIG`, "
      "`FINAL_CONFIG`) with the locked recovery step and the game parameters of "
      "`protocols/v2_T2_locked_v1_1.json` (records[q].game, records[q].protocol.recovery_step). "
      "eta_2/DW is the scalar `eta_T_over_dw` (T = 2).\n")
    a(md_table(["family", "perturbation", "grid"], [
        ["(a)", "e1 = (1 + d) e1*, stage 2 exact", f"d in {list(DELTA_GRID)}"],
        ["(b)", "e2(d) = (1 + d) e2*(d), stage 1 exact", f"d in {list(DELTA_GRID)}"],
        ["(c)", "e2* convolved with a uniform kernel of half-width h (d units, 0.01 grid, linear "
                "interpolation), stage 1 exact", f"h in {list(KERNEL_H)}"],
        ["(d)", "e2(d) = e2*(d) + tau for abs(d) >= 2q, stage 1 exact",
         f"tau in {list(TAU_GRID)} (effort units)"],
        ["(e)", "(a) with the (c) kernel whose induced peak error is closest to "
                f"{KERNEL_TARGET}", f"d in {list(E_DELTAS)}, h chosen per q"],
    ]))
    a("\nSource of the grids: `tools/v2/verifier_sensitivity.py` (constants DELTA_GRID, KERNEL_H, "
      "TAU_GRID, E_DELTAS, KERNEL_TARGET); the candidates actually run are the rows of "
      f"`{c_ev}`.\n")
    gm = meta["games"]
    a(md_table(["q", "w_h", "w_l", "k", "DW", "e_min..e_max", "e1* = DW/(6kq)", "e2*(0) = DW/(4kq)",
                "B", "recovery step"],
               [[q, gm[str(q)]["w_h"], gm[str(q)]["w_l"], fnum(gm[str(q)]["k"], "g"),
                 fnum(gm[str(q)]["dw"], "g"), f"{gm[str(q)]['e_min']}..{gm[str(q)]['e_max']}",
                 fnum(gm[str(q)]["g1"], "g"), fnum(gm[str(q)]["g2_0"], "g"),
                 fnum(gm[str(q)]["B"], "g"), gm[str(q)]["recovery_step"]] for q in qs]))
    a(f"\nSource: `{c_me}` (games).\n")
    nb = int(ev["clip_binds"].sum()) if "clip_binds" in ev else 0
    a("Clipping. Efforts are clipped to [e_min, e_max] at the output of the policy, because the "
      "verifier rejects efforts outside that interval; the clip is inspected on the unclipped "
      "stage-1 value and on a dense 0.01 grid over the whole stage-2 domain. "
      f"Rows in which the clip changes a value: {nb} of {len(ev)}; largest unclipped effort over "
      f"all rows: {fnum(ev['clip_raw_max'].max(), 'g')}, smallest: "
      f"{fnum(ev['clip_raw_min'].min(), 'g')} (`{c_ev}`, columns clip_binds, clip_raw_min, "
      "clip_raw_max).\n")
    kc = meta.get("kernel_choice", {})
    if kc:
        a("Family (e) kernel choice (the (c) kernel whose induced peak error is closest to "
          f"{KERNEL_TARGET}, chosen per q from the (c) results of the same run):\n")
        a(md_table(["q", "chosen h", "peak error of that kernel", "|distance to target|",
                    "peak error of every h (h: value)", "source"],
                   [[q, fnum(k["h"], "g"), fnum(k["peak_err"]), fnum(k["distance"]),
                     "; ".join(f"{h}: {fnum(v, 'g')}" for h, v in k["table"].items()),
                     k["source"]] for q, k in kc.items()]))
        a(f"\nSource: `{c_me}` (kernel_choice).\n")
    bad = ev[ev["status"] != "ok"]
    if len(bad):
        a("Evaluations that raised a verifier error (no metrics recorded):\n")
        a(md_table(["family", "cand_id", "q", "tier", "status"],
                   [[r.family, r.cand_id, r.q, r.tier, r.status] for r in bad.itertuples()]))
        a(f"\nSource: `{c_ev}`.\n")

    ctrl = ev[(ev["family"] == "a") & (ev["perturbation"] == 0.0) & (ev["status"] == "ok")]
    a("## 3. Unperturbed control (d = 0: the exact closed-form candidate)\n")
    if len(ctrl):
        a(md_table(["q", "tier", "valid", "Gmax_full/DW", "(t*, d*)", "eta_2/DW", "EXP_root/DW",
                    "dReach/DW", "Delta_max_all/DW", "dFull/DW"],
                   [[r.q, r.tier, fnum(r.valid), fnum(r.Gmax_full_over_dw),
                     f"({int(r.Gmax_full_t)}, {fnum(r.Gmax_full_d, 'g')})", fnum(r.eta_T_over_dw),
                     fnum(r.EXP_root_over_dw), fnum(r.dReach_over_dw),
                     fnum(r.Deltamax_all_over_dw), fnum(r.dFull_over_dw)]
                    for r in ctrl.itertuples()]))
        a(f"\nSource: `{c_ev}` (family a, perturbation 0). These are the numerical floors of the "
          "verifier on the exact equilibrium; the responses below are read against them.\n")
    else:
        a("The d = 0 candidate (family a) was not part of this run.\n")

    for fam in FAMILIES:
        g = paired[paired["family"] == fam]
        a(f"## {4 + FAMILIES.index(fam)}. Family {FAMILY_TITLE[fam]}\n")
        if g.empty:
            a("Not computed in this run (family not selected).\n")
            continue
        a(f"Perturbation parameter: `{PERT_NAME[fam]}` ({PERT_UNIT[fam]}). Fit abscissa x: "
          f"`{g['x_name'].iloc[0]}`.\n")
        for q in sorted(g["q"].unique()):
            gq = g[g["q"] == q].sort_values("perturbation")
            has_f = ("Gmax_full_over_dw_final" in gq
                     and not gq["Gmax_full_over_dw_final"].isna().all())
            a(f"### q = {q}: verifier quantities\n")
            hdr = ["perturbation", "x", "G_dev", "G_final", "(t*, d*) final", "G dev-final",
                   "eta_dev", "eta_final", "eta dev-final", "EXP_root final", "dReach final",
                   "Delta_max_all final", "dFull final", "valid dev/final"]
            rows = []
            for r in gq.itertuples():
                tstar = (f"({int(r.Gmax_full_t_final)}, {fnum(r.Gmax_full_d_final, 'g')})"
                         if has_f and not pd.isna(r.Gmax_full_t_final) else "n/a")
                rows.append([fnum(r.perturbation, "g"), fnum(r.x, "g"),
                             fnum(r.Gmax_full_over_dw_dev), fnum(r.Gmax_full_over_dw_final),
                             tstar, fnum(getattr(r, "dev_minus_final_Gmax", None)),
                             fnum(r.eta_T_over_dw_dev), fnum(r.eta_T_over_dw_final),
                             fnum(getattr(r, "dev_minus_final_eta", None)),
                             fnum(r.EXP_root_over_dw_final), fnum(r.dReach_over_dw_final),
                             fnum(r.Deltamax_all_over_dw_final), fnum(r.dFull_over_dw_final),
                             f"{fnum(r.valid_dev)}/{fnum(r.valid_final)}"])
            a(md_table(hdr, rows))
            a(f"\nSource: `{c_pa}` (family {fam}, q {q}); G = Gmax_full/DW, eta = eta_2/DW, "
              "all in units of DW; n/a = tier not run.\n")
            a(f"### q = {q}: recovery metrics (tier independent) and clip\n")
            a(md_table(["perturbation", "stage-1 error (signed)", "stage-2 peak error (signed)",
                        "RMSE_pos / e2*(0)", "tail mean / e2*(0)", "clip binds"],
                       [[fnum(r.perturbation, "g"), fnum(r.stage1_rel_err_signed),
                         fnum(r.stage2_peak_rel_err_signed), fnum(r.stage2_rmse_pos_over_g2_0),
                         fnum(r.stage2_tail_mean_over_g2_0), fnum(r.clip_binds)]
                        for r in gq.itertuples()]))
            a(f"\nSource: `{c_pa}` (family {fam}, q {q}).\n")
        if fits is not None and len(fits[fits["family"] == fam]):
            a("### Quadratic fit through the origin: metric ~ a x^2\n")
            ff = fits[fits["family"] == fam]
            a(md_table(["q", "tier", "metric", "side", "n", "a", "RMSE of residuals",
                        "max abs residual (at x)", "max abs resid / max metric", "R2 (uncentered)",
                        "x at threshold from fit", "note"],
                       [[r.q, r.tier, r.metric, r.side, r.n, fnum(r.a), fnum(r.rmse_resid),
                         f"{fnum(r.max_abs_resid)} ({fnum(r.x_at_max_abs_resid, 'g')})",
                         fnum(r.max_abs_resid_over_ymax), fnum(r.r2_uncentered),
                         f"{r.fit_threshold} {fnum(r.threshold, 'g')}: "
                         f"{fnum(r.x_at_threshold_fit, 'g')}",
                         "" if pd.isna(r.fit_note) else r.fit_note] for r in ff.itertuples()]))
            a(f"\nSource: `{c_fi}` (family {fam}); the full residual list is its `residuals` "
              "column (input order: ascending perturbation). Points with x = 0 carry no "
              "information in a fit through the origin and are excluded; 'x at threshold from "
              "fit' is the extrapolation sqrt(threshold / a) of the fitted law and is "
              "descriptive only.\n")
        if det is not None and len(det[det["family"] == fam]):
            a("### Detection limits\n")
            a("Smallest |perturbation| on the grid at which the metric exceeds the threshold "
              f"(strict '>'): G-F Gmax_full/DW > {th['G-F']:g}, G-A eta_2/DW > {th['G-A']:g}, "
              f"G-N |dev - final| of Gmax_full/DW > {th['G-N_Gmax']:g} or of eta_2/DW > "
              f"{th['G-N_eta']:g} (units of DW). For the signed families (a), (b) "
              "the negative and positive sides are listed separately and 'any' is the smaller "
              "|perturbation| of the two. 'bracket lo' is the largest grid |perturbation| below "
              "the limit on that side.\n")
            dd = det[det["family"] == fam]
            a(md_table(["q", "tier", "criterion", "side", "limit (|p|)", "reached",
                        "x at limit", "value at limit", "bracket lo", "max |p| on grid",
                        "max value on grid"],
                       [[r.q, r.tier, r.criterion, r.side, r.limit_str, fnum(r.reached),
                         fnum(r.x_at_limit, "g"), fnum(r.value_at_limit), r.bracket_lo,
                         fnum(r.max_abs_pert_on_grid, "g"), fnum(r.max_value_on_grid)]
                        for r in dd.itertuples()]))
            a(f"\nSource: `{c_de}` (family {fam}).\n")
            if fam in SIGNED_FAMILIES:
                notes = []
                for (q, tier, crit), gg in dd.groupby(["q", "tier", "criterion"]):
                    n_, p_ = gg[gg["side"] == "neg"].iloc[0], gg[gg["side"] == "pos"].iloc[0]
                    same = n_["limit_str"] == p_["limit_str"]
                    notes.append([q, tier, crit, n_["limit_str"], p_["limit_str"],
                                  "same on both sides" if same else "differs between sides"])
                a("Sign handling (negative side vs positive side):\n")
                a(md_table(["q", "tier", "criterion", "negative side", "positive side",
                            "comparison"], notes))
                a(f"\nSource: `{c_de}` (side = neg / pos); comparison is string equality of the "
                  "two limits.\n")
        fig = out_dir / "figures" / f"d2_family_{fam}.png"
        if fig.exists():
            a(f"![Gmax_full/DW and eta_2/DW against |perturbation|, family {fam}]"
              f"({os.path.relpath(fig, report_path.parent)})\n")
            a(f"Figure: `{rel(fig)}` (and `.pdf`), log-log, both tiers, both q; data from "
              f"`{c_pa}`.\n")
        if fam == "e":
            a("### Family (e) predictions next to the confirmation runs with the largest "
              "stage-1 errors\n")
            if fe is None or fe.empty:
                a("No comparison table in this run (family e absent or confirmation files not "
                  "found).\n")
            else:
                a("Observed values are the confirmation runs' end-of-B candidates (their own, "
                  "imperfect stage 2); predicted values are the family-(e) candidate at the grid "
                  "delta nearest to the run's |stage-1 error| (positive deltas only) with the "
                  "chosen kernel. Peak errors are shown because the (e) kernel and the run's own "
                  "stage-2 shape differ.\n")
                a(md_table(["q", "seed", "stage-1 error", "observed Gmax final",
                            "observed Gmax dev", "observed peak error", "(e) delta",
                            "(e) kernel h", "(e) peak error",
                            "(e) predicted Gmax final", "(e) predicted Gmax dev",
                            "observed / predicted (final)", "kernel-only (c) Gmax final"],
                           [[r.q, r.seed, fnum(r.stage1_rel_err_signed),
                             fnum(r.observed_gmax_final), fnum(r.observed_gmax_dev),
                             fnum(r.observed_stage2_peak_rel_err_signed),
                             fnum(r.e_nearest_delta, "g"), fnum(r.e_kernel_h, "g"),
                             fnum(r.e_peak_rel_err_signed), fnum(r.e_pred_gmax_final),
                             fnum(r.e_pred_gmax_dev), fnum(r.ratio_observed_over_pred_final, "g"),
                             fnum(r.kernel_only_pred_gmax_final)] for r in fe.itertuples()]))
                a(f"\nSource: `{c_fe}`; observed columns from `{meta['confirmation_per_run']}` "
                  f"(sha256 `{meta['confirmation_per_run_sha256'][:16]}...`) and "
                  f"`{meta['confirmation_reported_metrics']}` (read only, canonical worktree); "
                  f"predicted columns from `{c_pa}`.\n")

    a(f"## {4 + len(FAMILIES)}. Detection limits, all families\n")
    if det is None or det.empty:
        a("No detection limits in this run.\n")
    else:
        dd = det[det["side"] == "any"]
        a(md_table(["family", "q", "tier", "criterion", "limit (|p|, any side)", "x at limit"],
                   [[r.family, r.q, r.tier, r.criterion, r.limit_str, fnum(r.x_at_limit, "g")]
                    for r in dd.itertuples()]))
        a(f"\nSource: `{c_de}` (side = any). Perturbation units: (a), (b), (e) relative delta; "
          "(c) kernel half-width h in d units; (d) tau in effort units.\n")
    a(f"## {5 + len(FAMILIES)}. Limitations\n")
    a("This is a sensitivity study of the verifier on closed-form perturbations of the exact "
      "equilibrium. It does not train anything, does not use Beta policies (the candidates are "
      "deterministic mean functions, so the distributional smoothing of a trained actor is "
      "absent), covers one perturbation family at a time (plus the single combination (e)), uses "
      "only the pre-registered grids (the detection limits are grid-resolution bounds, not "
      "continuous thresholds; 'not reached on the grid' says nothing beyond the largest "
      "perturbation tested), and relates to trained runs only through the descriptive comparison "
      "in the family-(e) section. The verifier tiers, the gates and every threshold of "
      "protocol v1.1 are used as locked and are not changed by this report.\n")
    text = "\n".join(L)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text)
    return text


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
SMOKE_IDS = ("a_d+0.05", "c_h12", "e_d+0.05_h12")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write dict rows with the union of keys (first-seen order); missing values are blank."""
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


FLOOR_KEYS = ("Gmax_full_over_dw", "Gmax_full_t", "Gmax_full_d", "eta_T_over_dw",
              "EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw", "dFull_over_dw",
              "stage1_rel_err_signed", "stage2_peak_rel_err_signed",
              "stage2_rmse_pos_over_g2_0", "stage2_tail_mean_over_g2_0")


def floors_delta0(ev: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """Verifier outputs for the exact closed-form candidate (family a, delta = 0) per q and tier.

    These are the numerical floors against which every response is read; the exact candidate
    has zero true deviation gain.

    Args:
        ev: evaluations.csv as a DataFrame.

    Returns:
        ``{"q50_development": {...}, ...}``; empty when the control is not part of the run.
    """
    ctrl = ev[(ev["family"] == "a") & (ev["perturbation"] == 0.0) & (ev["status"] == "ok")]
    return {f"q{int(r['q'])}_{r['tier']}": {k: (None if pd.isna(r[k]) else float(r[k]))
                                             for k in FLOOR_KEYS}
            for _, r in ctrl.iterrows()}


def run_job(args: argparse.Namespace) -> int:
    """Evaluate, analyse, plot and write meta.json."""
    t_start = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    proto = json.loads(Path(args.protocol).read_text())
    qs, tiers, families = list(args.qs), list(args.tiers), list(args.families)
    only = None
    if args.smoke:
        qs, tiers, families, only = [50], ["development"], ["a", "c", "e"], list(SMOKE_IDS)
    git = git_state()
    rows, choice = run_evaluations(qs, tiers, families, int(args.workers), proto, git, only)
    write_csv(out / "evaluations.csv", rows)
    ev = pd.read_csv(out / "evaluations.csv")
    th = thresholds(proto)
    paired = build_paired(ev)
    paired.to_csv(out / "paired.csv", index=False)
    build_fits(paired, th).to_csv(out / "fits.csv", index=False)
    build_detection(paired, th).to_csv(out / "detection_limits.csv", index=False)
    conf_dir = Path(args.confirmation_dir)
    conf_ok = (conf_dir / "per_run.csv").exists() and (conf_dir / "reported_metrics.csv").exists()
    if "e" in families and conf_ok:
        build_family_e_confirmation(paired, conf_dir).to_csv(
            out / "family_e_confirmation.csv", index=False)
    figs = make_figures(paired, th, out / "figures")
    games = {}
    for q in qs:
        g = proto["records"][str(q)]["game"]
        spec = GameSpec(**g)
        games[str(q)] = {**g, "dw": spec.dw, "B": spec.B,
                         "recovery_step": proto["records"][str(q)]["protocol"]["recovery_step"],
                         "g1": float(g1_two_stage(spec.q, spec.w_h, spec.w_l, spec.k)),
                         "g2_0": float(g2_two_stage(np.zeros(1), spec.q, spec.w_h, spec.w_l,
                                                    spec.k, spec.e_max)[0])}
    cmd = ("python tools/v2/verifier_sensitivity.py --out " + rel(out)
           + (" --smoke" if args.smoke else
              f" --workers {args.workers} --qs {' '.join(map(str, qs))}"
              f" --tiers {' '.join(tiers)} --families {' '.join(families)}"))
    meta = {
        "schema": "d2_verifier_sensitivity/1", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv, "command": cmd, "python": sys.executable, "smoke": bool(args.smoke),
        **git, "qs": qs, "tiers": tiers, "families": families, "workers": int(args.workers),
        "protocol": rel(args.protocol), "protocol_sha256": sha256_of(Path(args.protocol)),
        "thresholds": th, "games": games, "kernel_choice": choice,
        "kernel_grid_step": KERNEL_GRID_STEP, "kernel_target": KERNEL_TARGET,
        "tiers_config": {k: asdict(v) for k, v in TIERS.items()},
        "tiers_equal_protocol_record": {
            k: VerifierConfig(**proto["records"][str(qs[0])]["verifier"][k]) == v
            for k, v in TIERS.items()},
        "n_evaluations": len(rows), "n_errors": int((ev["status"] != "ok").sum()),
        "n_clip_binds": int(ev["clip_binds"].sum()),
        "figures": figs,
        "confirmation_per_run": str(conf_dir / "per_run.csv") if conf_ok else None,
        "confirmation_per_run_sha256": sha256_of(conf_dir / "per_run.csv") if conf_ok else None,
        "confirmation_reported_metrics": (str(conf_dir / "reported_metrics.csv")
                                          if conf_ok else None),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "python": platform.python_version()},
        "eval_wall_sec_sum": float(ev["eval_wall_sec"].sum()),
        "timing_by_tier": {
            tier: {"n": int(len(g)), "eval_sec_mean": float(g["eval_wall_sec"].mean()),
                   "eval_sec_max": float(g["eval_wall_sec"].max()),
                   "eval_sec_sum": float(g["eval_wall_sec"].sum())}
            for tier, g in ev.groupby("tier")},
        "numerical_floors_delta0": floors_delta0(ev),
        "total_wall_sec": time.perf_counter() - t_start,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"wrote {len(rows)} evaluations ({meta['n_errors']} errors) to {rel(out)}; "
          f"total {meta['total_wall_sec']:.1f}s")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--qs", type=int, nargs="+", default=[50, 60])
    p.add_argument("--tiers", nargs="+", default=["development", "final"], choices=list(TIERS))
    p.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    p.add_argument("--workers", type=int, default=1,
                   help="process pool size (each single-threaded)")
    p.add_argument("--smoke", action="store_true",
                   help="q=50, development tier, 3 candidates (a +0.05, c h=12, e +0.05)")
    p.add_argument("--report", action="store_true",
                   help="render the markdown report from the CSVs in --out (no evaluation)")
    p.add_argument("--report-path", default=str(DEFAULT_REPORT))
    p.add_argument("--protocol", default=str(PROTOCOL))
    p.add_argument("--confirmation-dir", default=str(CONFIRMATION_DIR))
    args = p.parse_args(argv)
    if args.report:
        render_report(Path(args.out), Path(args.report_path))
        print(f"wrote {args.report_path}")
        return 0
    return run_job(args)


if __name__ == "__main__":
    raise SystemExit(main())
