#!/usr/bin/env python
"""MS-R2 D6: three closed-form-free stop-criterion candidates for the terminal stage (report only).

No training, nothing in any run reads this tool. For every weight export ``weights/u{u:05d}.npz`` of the
terminal stage (stage 2 of T = 2; the exports u <= ``rule_log.json["stages"]["2"]["exit_update"]``, every 25
updates) the development verifier is replayed on the composite candidate {stage 1: the export's network,
stage 2: the export's network} (``replay_dev_rule``: ``build_actor``, ``candidate_fns``, ``replay_candidate``;
the float32 forward pass is the training one, a ``conc_scale`` array of the export is applied) and three
candidates are computed from the verifier arrays and the policy alone:

* C1 = R0 = r_2(0) / s_2 = |e_hat_2(0) - a_dev_2(0)| / a_dev_2(0) at the node d = 0 (MS-R1 A3(a)).
* C2 = (e_sigma(0) - e_hat_2(0)) / e_sigma(0), the training residual at the tie, with e_sigma(0) the tie best
  response of the game played with the learned noise (``utils.ms_noise``), reported with the noise floor
  sigma_2(0) / (sqrt(pi) q). The signed value is the column ``c2``; the tables and the fire rule use |c2|.
* C3 = R_defl = max over the non-tail nodes (|d| < 2q) of r_2(d) / (s_2 |1 - J(d)|), r_2 and s_2 the
  development-tier values of D3, J(d) = d e_tilde_2(d) / d e_opp a central finite difference (h = 0.5 effort
  units) of the ACCURATE one-step best response e_tilde at d against the opponent action e_opp(d) = e_hat_2(-d)
  shifted by +-h. The terminal one-step objective of the verifier is Q(e; d, o) = w_L + DW F_xi(d + e - o) -
  k e^2, Q' = DW f_xi(d + e - o) - 2 k e, strictly decreasing when 2k > a = DW / (4 q^2): the exact best response
  is the root of Q' (vectorised bisection, clipped to [e_min, e_max]); for 2k <= a a dense effort grid (step
  0.01) with parabolic refinement is used. Nodes with |1 - J| < 0.1 are excluded and counted. J is reported
  against the linearisation J - 1 = -2k / (2k - a) (d < 0) and -2k / (2k + a) (d > 0).

Diagnostics next to the candidates: ``s_exact`` (the exact best response at d = 0) and ``R0_exact`` (the R0
the exact best response would give: the error of the development tier's effort grid, step 1, is visible in
``s - s_exact``). Closed-form errors (``utils.v2_metrics.evaluate`` scalars) are reporting columns only.

Every root is an explicit argument and is recorded in ``manifest.json``; a missing file stops before any
computation. ``--source NAME=ROOT:ARM1,ARM2`` names a root with layout ``<ROOT>/q{q}/seed{seed}/<ARM>/``.

Commands::

    python tools/ms/r2_stop_candidates.py replay --out results/ms_r2/stop_calibration \
        --source base=<MS-R1 base root>:MS_base \
        --source pilot=<MS-R1 pilot root>:MS_rule,MS_s25a0,MS_s25a5,MS_s35a0,MS_s35a5,MS_base2400 \
        --qs 50 60 --seeds 10501-10510 --workers 30
    python tools/ms/r2_stop_candidates.py tables --out results/ms_r2/stop_calibration --md \
        > results/ms_r2/stop_calibration/tables_nofire.md
    python tools/ms/r2_stop_candidates.py tables --out results/ms_r2/stop_calibration --fire \
        --grids grids.json --md > fire.md       # grids.json = {"C1": [theta...], "C2": [...], "C3": [...]}

``replay`` writes ``<out>/<source>/<arm>/q{q}/seed{seed}.csv`` (one row per terminal-stage export),
``manifest.json`` and ``validation_summary.csv`` (the replay against the runs' own ``ms_checks_stage2.csv``
rows at the same updates). ``tables`` writes ``<out>/tables/*.csv`` from those CSVs (nothing is recomputed);
without ``--fire``: the Spearman tables (a), the freeze distributions (b), the J check (c) and the diagnostics;
with ``--fire`` (which needs ``--grids``) only the fire tables (d) and ``fire_meta.json``. The rule is
``candidate <= theta`` at M = 3 consecutive terminal-stage exports (K = 25) with C1 = R0, C2 = |c2|, C3 = R_defl;
an optional key ``"C2s"`` of the grids file applies the same rule to the signed c2. Both commands refuse
to overwrite an existing output file. Exit code 0 on success, 2 on a stop-and-report (missing input, existing
output), 3 when the replay does not reproduce the runs' own check rows.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "tools" / "ms")):
    if _p not in sys.path:
        sys.path.append(_p)

from envs.curriculum_env import GameSpec  # noqa: E402
from replay_dev_rule import (  # noqa: E402
    Table, build_actor, candidate_fns, fnum, git_head, load_export, med_rng, replay_candidate,
    sha256_file, spearman)
from utils.dp_br_verifier import DEV_CONFIG  # noqa: E402
from utils.ms_noise import tie_noise_report  # noqa: E402
from utils.ms_residual import NODE_TOL  # noqa: E402
from utils.theory_multistage import F_xi, f_xi  # noqa: E402

torch.set_num_threads(1)

# --------------------------------------------------------------------------------------- constants
STAGE = 2                        # the terminal stage of T = 2
H_FD = 0.5                       # finite-difference step of J, effort units
EXCL_ABS = 0.1                   # nodes with |1 - J| below this are excluded from R_defl
J_AWAY = 8.0                     # |d| beyond which J is compared with the linearisation (away from the kink)
BISECT_ITERS = 64
DENSE_STEP = 0.01
CONST_LR = 3e-4
SPEARMAN_FROM_U = 400
FIRE_M = 3
FIRE_K = 25
PEAK_TOL = 0.05
MAX_WORKERS = 30
CHECK_TOL = 1e-9                 # replay vs the run's own check rows (expected 0)
CHECK_FILE = "ms_checks_stage2.csv"
REQUIRED = ("run_config.json", "rule_log.json", "train_history.json", CHECK_FILE)

ID_COLS = ["source", "arm", "q", "seed", "update", "actor_lr", "conc_scale", "valid"]
D3_COLS = ["Delta", "s", "R", "R_tail", "C"]
C1_COLS = ["R0"]
C2_COLS = ["e_hat0", "sigma0", "e_sigma0", "c2", "floor"]
C3_COLS = ["R_defl", "n_defl_excluded", "n_defl_nodes", "J_mean_neg", "J_mean_pos", "n_J_neg", "n_J_pos",
           "J_lin_neg", "J_lin_pos", "s_exact", "R0_exact"]
CLOSED_COLS = ["stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_rmse_pos_over_g2_0",
               "stage2_tail_mean_over_g2_0", "stage2_tail_max_over_g2_0", "g2_at_0", "e2_at_0"]
CSV_COLS = ID_COLS + D3_COLS + C1_COLS + C2_COLS + C3_COLS + CLOSED_COLS + ["error"]
STR_COLS = {"source", "arm", "error"}
INT_COLS = {"q", "seed", "update", "n_defl_excluded", "n_defl_nodes", "n_J_neg", "n_J_pos"}
BOOL_COLS = {"valid"}
CHECK_CLOSED = ("stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_rmse_pos_over_g2_0",
                "stage2_tail_mean_over_g2_0", "stage2_tail_max_over_g2_0", "e2_at_0")

# candidate name -> per-export column (the C2 column is |c2|, added when the CSVs are loaded)
CANDIDATES: Tuple[Tuple[str, str], ...] = (("C1", "R0"), ("C2", "c2_abs"), ("C3", "R_defl"))
SPEARMAN_COLS: Tuple[Tuple[str, str], ...] = CANDIDATES + (("R", "R"), ("Delta", "Delta"),
                                                           ("C2 signed", "c2"), ("C1 exact", "R0_exact"))
# the fire rule's candidates: C2 uses |c2|; the signed residual is an optional variant (key "C2s" of the grids file)
FIRE_CANDIDATES: Tuple[Tuple[str, str], ...] = CANDIDATES + (("C2s", "c2"),)
TARGETS: Tuple[Tuple[str, str], ...] = (("|peak|", "stage2_peak_rel_err_abs"),
                                        ("RMSE_pos", "stage2_rmse_pos_over_g2_0"))
FREEZE_COLS: Tuple[Tuple[str, str], ...] = (
    ("C1", "R0"), ("C2 (|c2|)", "c2_abs"), ("C2 signed", "c2"), ("C2 floor", "floor"), ("C3", "R_defl"),
    ("R", "R"), ("Delta", "Delta"), ("C1 exact", "R0_exact"), ("|peak|", "stage2_peak_rel_err_abs"),
    ("RMSE_pos", "stage2_rmse_pos_over_g2_0"))
FIRE_METRICS: Tuple[Tuple[str, str], ...] = (
    ("abs_peak", "stage2_peak_rel_err_abs"), ("signed_peak", "stage2_peak_rel_err_signed"),
    ("rmse_pos", "stage2_rmse_pos_over_g2_0"), ("tail_mean", "stage2_tail_mean_over_g2_0"))

Row = Dict[str, object]
RunKey = Tuple[str, str, int, int]          # (source, arm, q, seed)


# ====================================================================== accurate terminal best response
def terminal_dq(e: np.ndarray, d: np.ndarray, o: np.ndarray, dw: float, k: float, q: float) -> np.ndarray:
    """Derivative ``Q'(e; d, o) = DW f_xi(d + e - o) - 2 k e`` of the terminal one-step objective.

    Args:
        e: Own effort.
        d: Score gap at the start of the terminal stage.
        o: Opponent's (deterministic) effort.
        dw: Prize spread w_H - w_L.
        k: Cost coefficient.
        q: Noise half-width.

    Returns:
        ``Q'`` (broadcast shape of the inputs).
    """
    e = np.asarray(e, dtype=float)
    return dw * f_xi(np.asarray(d, dtype=float) + e - np.asarray(o, dtype=float), q) - 2.0 * k * e


def terminal_q(e: np.ndarray, d: np.ndarray, o: np.ndarray, dw: float, k: float, q: float) -> np.ndarray:
    """Terminal one-step objective without the constant w_L: ``DW F_xi(d + e - o) - k e^2``."""
    e = np.asarray(e, dtype=float)
    x = np.asarray(d, dtype=float) + e - np.asarray(o, dtype=float)
    return dw * F_xi(x, q) - k * e * e


def br_is_monotone(dw: float, k: float, q: float) -> bool:
    """True iff ``Q'`` is strictly decreasing in e, i.e. ``2k > a = DW / (4 q^2)``."""
    return bool(2.0 * k > dw / (4.0 * q * q))


def exact_best_response(d: Any, o: Any, dw: float, k: float, q: float, e_min: float, e_max: float,
                        iters: int = BISECT_ITERS) -> np.ndarray:
    """Exact best response: the root of the strictly decreasing ``Q'``, clipped to ``[e_min, e_max]``.

    Vectorised bisection (``iters`` = 64 halvings of the effort range: below the float resolution).

    Args:
        d: Gaps (broadcast with ``o``).
        o: Opponent efforts.
        dw: Prize spread.
        k: Cost coefficient (requires ``2k > DW / (4 q^2)``).
        q: Noise half-width.
        e_min: Lower effort bound.
        e_max: Upper effort bound.
        iters: Number of bisection steps.

    Returns:
        The maximiser of ``Q(.; d, o)`` over ``[e_min, e_max]``, shape of the broadcast inputs (at least 1-D).
    """
    d_a, o_a = np.broadcast_arrays(np.atleast_1d(np.asarray(d, dtype=float)),
                                   np.atleast_1d(np.asarray(o, dtype=float)))
    lo = np.full(d_a.shape, float(e_min))
    hi = np.full(d_a.shape, float(e_max))
    at_min = terminal_dq(lo, d_a, o_a, dw, k, q) <= 0.0
    at_max = terminal_dq(hi, d_a, o_a, dw, k, q) >= 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = terminal_dq(mid, d_a, o_a, dw, k, q) > 0.0
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return np.where(at_min, float(e_min), np.where(at_max, float(e_max), 0.5 * (lo + hi)))


def dense_best_response(d: Any, o: Any, dw: float, k: float, q: float, e_min: float, e_max: float,
                        step: float = DENSE_STEP) -> np.ndarray:
    """Global maximiser of ``Q`` on a dense effort grid with a parabolic refinement of the best cell.

    Used to verify :func:`exact_best_response` and as the fallback when ``2k <= DW / (4 q^2)``.

    Args:
        d: Gaps (broadcast with ``o``).
        o: Opponent efforts.
        dw: Prize spread.
        k: Cost coefficient.
        q: Noise half-width.
        e_min: Lower effort bound.
        e_max: Upper effort bound.
        step: Effort grid spacing (at most).

    Returns:
        The refined maximiser, shape of the broadcast inputs (at least 1-D).
    """
    d_a, o_a = np.broadcast_arrays(np.atleast_1d(np.asarray(d, dtype=float)),
                                   np.atleast_1d(np.asarray(o, dtype=float)))
    shape = d_a.shape
    d_f, o_f = d_a.reshape(-1), o_a.reshape(-1)
    n = int(np.ceil((e_max - e_min) / step - 1e-9))
    grid = np.linspace(e_min, e_max, n + 1)
    h = float(grid[1] - grid[0])
    qm = terminal_q(grid[None, :], d_f[:, None], o_f[:, None], dw, k, q)
    j = np.argmax(qm, axis=1)
    jj = np.clip(j, 1, n - 1)
    ar = np.arange(d_f.size)
    y0, y1, y2 = qm[ar, jj - 1], qm[ar, jj], qm[ar, jj + 1]
    den = y0 - 2.0 * y1 + y2
    ok = (j > 0) & (j < n) & (den < 0.0)
    off = np.clip(np.where(ok, 0.5 * (y0 - y2) / np.where(ok, den, -1.0), 0.0), -1.0, 1.0)
    ev = np.clip(grid[jj] + off * h, e_min, e_max)
    better = ok & (terminal_q(ev, d_f, o_f, dw, k, q) >= qm[ar, j])
    return np.where(better, ev, grid[j]).reshape(shape)


def best_response(d: Any, o: Any, dw: float, k: float, q: float, e_min: float, e_max: float) -> np.ndarray:
    """Accurate one-step best response: exact root when ``2k > DW / (4 q^2)``, else the dense-grid method."""
    if br_is_monotone(dw, k, q):
        return exact_best_response(d, o, dw, k, q, e_min, e_max)
    return dense_best_response(d, o, dw, k, q, e_min, e_max)


def br_jacobian(d: Any, o: Any, dw: float, k: float, q: float, e_min: float, e_max: float,
                h: float = H_FD) -> np.ndarray:
    """``J = d e_tilde / d o``: central difference of the best response with ``o`` shifted by +-``h``."""
    o_a = np.asarray(o, dtype=float)
    plus = best_response(d, o_a + h, dw, k, q, e_min, e_max)
    minus = best_response(d, o_a - h, dw, k, q, e_min, e_max)
    return (plus - minus) / (2.0 * h)


def linearised_j(dw: float, k: float, q: float) -> Tuple[float, float]:
    """``(J_lin_neg, J_lin_pos)``: ``J - 1 = -2k / (2k - a)`` for d < 0 and ``-2k / (2k + a)`` for d > 0."""
    a = dw / (4.0 * q * q)
    return 1.0 - 2.0 * k / (2.0 * k - a), 1.0 - 2.0 * k / (2.0 * k + a)


def deflate(r: np.ndarray, s: float, jac: np.ndarray, nontail: np.ndarray, tol: float = EXCL_ABS
            ) -> Tuple[float, int, int]:
    """``R_defl = max r / (s |1 - J|)`` over the non-tail nodes with ``|1 - J| >= tol``.

    Returns:
        ``(R_defl, n_excluded, n_nodes)`` (``R_defl`` is NaN when every node is excluded).
    """
    fac = np.abs(1.0 - jac)
    excl = nontail & (fac < tol)
    use = nontail & ~excl
    n_nodes, n_excl = int(nontail.sum()), int(excl.sum())
    if not use.any():
        return float("nan"), n_excl, n_nodes
    return float(np.max(r[use] / (s * fac[use]))), n_excl, n_nodes


@dataclass
class TerminalCandidates:
    """C1 and C3 of one verifier result plus the diagnostics (NaN where undefined)."""

    R0: float = float("nan")
    R_defl: float = float("nan")
    n_defl_excluded: int = 0
    n_defl_nodes: int = 0
    J_mean_neg: float = float("nan")
    J_mean_pos: float = float("nan")
    n_J_neg: int = 0
    n_J_pos: int = 0
    s_exact: float = float("nan")
    R0_exact: float = float("nan")
    J: Optional[np.ndarray] = None


def terminal_candidates(d_grid: np.ndarray, e_hat: np.ndarray, e_opp: np.ndarray, a_dev: np.ndarray,
                        spec: GameSpec, stage: int = STAGE, h: float = H_FD) -> TerminalCandidates:
    """C1 (R0), C3 (R_defl) and the J / exact-best-response diagnostics from the stage arrays.

    The node set and ``s``, ``r`` are those of ``utils.ms_residual.stage_diag`` (non-tail: ``|d| < 2q (T -
    stage + 1)``; ``s = a_dev(0)``; ``r = |e_hat - a_dev|``).

    Args:
        d_grid: Stage grid (contains the node 0).
        e_hat: Candidate mean action per node.
        e_opp: Opponent action per node (``e_hat(-d)``).
        a_dev: Development-tier one-step best response per node.
        spec: Game specification.
        stage: The terminal stage.
        h: Finite-difference step of J.

    Returns:
        The :class:`TerminalCandidates`.
    """
    d = np.asarray(d_grid, dtype=float)
    e_hat, e_opp, a_dev = (np.asarray(x, dtype=float) for x in (e_hat, e_opp, a_dev))
    j0 = int(np.argmin(np.abs(d)))
    if abs(d[j0]) > NODE_TOL:
        raise ValueError(f"stage {stage} grid has no zero node")
    thr = 2.0 * float(spec.q) * (spec.T - stage + 1)
    nontail = np.abs(d) < thr - NODE_TOL
    out = TerminalCandidates(n_defl_nodes=int(nontail.sum()))
    s = float(a_dev[j0])
    if not (np.isfinite(s) and s > 0.0):
        return out
    out.R0 = abs(float(e_hat[j0]) - s) / s
    dw, k, q = float(spec.dw), float(spec.k), float(spec.q)
    jac = br_jacobian(d, e_opp, dw, k, q, spec.e_min, spec.e_max, h)
    out.J = jac
    out.R_defl, out.n_defl_excluded, out.n_defl_nodes = deflate(np.abs(e_hat - a_dev), s, jac, nontail)
    neg, pos = nontail & (d < -J_AWAY), nontail & (d > J_AWAY)
    out.n_J_neg, out.n_J_pos = int(neg.sum()), int(pos.sum())
    out.J_mean_neg = float(jac[neg].mean()) if neg.any() else float("nan")
    out.J_mean_pos = float(jac[pos].mean()) if pos.any() else float("nan")
    out.s_exact = float(best_response(0.0, e_opp[j0], dw, k, q, spec.e_min, spec.e_max)[0])
    out.R0_exact = abs(float(e_hat[j0]) - out.s_exact) / out.s_exact if out.s_exact > 0.0 else float("nan")
    return out


# ============================================================================================= replay
def export_row(spec: GameSpec, net: Any) -> Row:
    """The candidate columns of one export (every numeric column NaN and ``valid`` False if the replay fails).

    Args:
        spec: Game specification.
        net: ``BetaActor`` carrying the export's weights (``build_actor``).

    Returns:
        A dict with the keys ``D3_COLS + C1_COLS + C2_COLS + C3_COLS + CLOSED_COLS``, ``valid`` and ``error``.
    """
    nan = float("nan")
    row: Row = {c: nan for c in D3_COLS + C1_COLS + C2_COLS + C3_COLS + CLOSED_COLS}
    row.update(valid=False, error="", n_defl_excluded=0, n_defl_nodes=0, n_J_neg=0, n_J_pos=0)
    nets = {1: net, 2: net}
    rp = replay_candidate(spec, nets, STAGE, DEV_CONFIG, 0)
    if rp.ev is None:
        row["error"] = rp.error
        return row
    dg = rp.diag
    row.update(valid=bool(dg.valid), Delta=dg.delta_over_dw, s=dg.s, R=dg.R, R_tail=dg.R_tail, C=dg.C)
    sc = rp.ev.scalars
    for c in CLOSED_COLS:
        row[c] = float(sc[c]) if c in sc else nan
    sr = rp.ev.res.stages[STAGE]
    cand = terminal_candidates(sr.d_grid, sr.e_hat, sr.e_opp, sr.a_dev, spec)
    j_lin = linearised_j(spec.dw, spec.k, spec.q)
    row.update(R0=cand.R0, R_defl=cand.R_defl, n_defl_excluded=cand.n_defl_excluded,
               n_defl_nodes=cand.n_defl_nodes, J_mean_neg=cand.J_mean_neg, J_mean_pos=cand.J_mean_pos,
               n_J_neg=cand.n_J_neg, n_J_pos=cand.n_J_pos, J_lin_neg=j_lin[0], J_lin_pos=j_lin[1],
               s_exact=cand.s_exact, R0_exact=cand.R0_exact)
    try:
        mean_fn, beta_fn = candidate_fns(spec, nets)
        tie = tie_noise_report(mean_fn, beta_fn, spec, STAGE)
        e_sig = float(tie["e_sigma_0"])
        row.update(e_hat0=tie["e_hat_0"], sigma0=tie["sigma_0"], e_sigma0=e_sig,
                   c2=(e_sig - tie["e_hat_0"]) / e_sig if e_sig != 0.0 else nan,
                   floor=tie["sigma_0"] / (math.sqrt(math.pi) * float(spec.q)))
    except (ValueError, FloatingPointError, ZeroDivisionError) as exc:
        row["error"] = f"tie: {type(exc).__name__}: {exc}"
    return row


def conc_scale_of(arrays: Mapping[str, np.ndarray]) -> float:
    """The export's ``conc_scale`` (1.0 when the export carries none)."""
    return float(np.asarray(arrays["conc_scale"]).reshape(-1)[0]) if "conc_scale" in arrays else 1.0


# ------------------------------------------------------------------------------------------ run refs
@dataclass(frozen=True)
class RunRef:
    """One run: ``<root>/q{q}/seed{seed}/<arm>``."""

    source: str
    root: str
    arm: str
    q: int
    seed: int

    @property
    def run_dir(self) -> Path:
        """Directory of the run."""
        return Path(self.root) / f"q{self.q}" / f"seed{self.seed}" / self.arm

    @property
    def key(self) -> str:
        """``<source>/<arm>/q<q>/seed<seed>``."""
        return f"{self.source}/{self.arm}/q{self.q}/seed{self.seed}"


def parse_source(text: str) -> Tuple[str, str, List[str]]:
    """``NAME=ROOT:ARM1,ARM2`` -> ``(name, root, [arms])``."""
    if "=" not in text or ":" not in text.split("=", 1)[1]:
        raise ValueError(f"--source expects NAME=ROOT:ARM1,ARM2 (got {text!r})")
    name, rest = text.split("=", 1)
    root, arms = rest.rsplit(":", 1)
    arm_list = [a for a in arms.split(",") if a]
    if not name or not root or not arm_list:
        raise ValueError(f"--source expects NAME=ROOT:ARM1,ARM2 (got {text!r})")
    return name, root, arm_list


def parse_seeds(items: Sequence[str]) -> List[int]:
    """Seeds from ``10501-10510`` ranges and / or ``10501,10503`` lists (several items allowed)."""
    out: List[int] = []
    for item in items:
        for part in item.split(","):
            if "-" in part:
                lo, hi = part.split("-", 1)
                out += list(range(int(lo), int(hi) + 1))
            elif part:
                out.append(int(part))
    return out


def discover(sources: Sequence[Tuple[str, str, List[str]]], qs: Sequence[int], seeds: Sequence[int]
             ) -> List[RunRef]:
    """Every (source, arm, q, seed) combination, in the order of the arguments."""
    return [RunRef(name, root, arm, int(q), int(seed)) for name, root, arms in sources for arm in arms
            for q in qs for seed in seeds]


def terminal_exports(run_dir: Path, only: Optional[Sequence[int]] = None) -> Tuple[List[int], List[str]]:
    """Terminal-stage export updates of a run and the files of it that are missing.

    The exports are those ``entry < u <= exit`` (``u`` a multiple of 25) of ``rule_log.json["stages"]["2"]``;
    ``only`` restricts them (debug / tests).

    Returns:
        ``(updates, missing)``; ``updates`` is empty when ``rule_log.json`` is unreadable (then ``missing``
        says why).
    """
    miss = [str(run_dir / f) for f in REQUIRED if not (run_dir / f).is_file()]
    if miss:
        return [], miss
    try:
        with open(run_dir / "rule_log.json") as f:
            st = json.load(f)["stages"][str(STAGE)]
        entry, exit_u = int(st["entry_update"]), int(st["exit_update"])
    except (KeyError, ValueError, TypeError) as exc:
        return [], [f"{run_dir / 'rule_log.json'}: no stage-{STAGE} entry/exit ({type(exc).__name__}: {exc})"]
    us = [u for u in range(25, exit_u + 1, 25) if u > entry]
    if only is not None:
        bad = [u for u in only if u not in us]
        if bad:
            return [], [f"{run_dir}: requested updates {bad} are not terminal-stage exports"]
        us = [u for u in us if u in set(only)]
    miss = [str(run_dir / "weights" / f"u{u:05d}.npz") for u in us
            if not (run_dir / "weights" / f"u{u:05d}.npz").is_file()]
    return us, miss


# ------------------------------------------------------------------------------------------- CSV io
def _fmt(v: object) -> object:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, (int, np.integer)):
        return int(v)
    return v


def write_new_csv(path: str, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    """Write ``rows`` (floats as ``repr``); refuses to replace an existing file."""
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists; this tool never overwrites an output file")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r[k]) for k in fields})
    os.replace(tmp, path)


def write_new_json(path: str, obj: object) -> None:
    """Write ``obj`` as JSON; refuses to replace an existing file."""
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists; this tool never overwrites an output file")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


def _parse(col: str, v: str) -> object:
    if col in STR_COLS:
        return v
    if col in INT_COLS:
        return int(v)
    if col in BOOL_COLS:
        return v == "True"
    return float(v) if v != "" else float("nan")


def read_run_csv(path: str) -> List[Row]:
    """A per-run CSV written by ``replay`` (types restored, ``c2_abs`` = |c2| added)."""
    with open(path, newline="") as f:
        rows: List[Row] = [{k: _parse(k, v) for k, v in r.items()} for r in csv.DictReader(f)]
    for r in rows:
        r["c2_abs"] = abs(float(r["c2"]))
    return rows


def run_csv_path(out_dir: str, ref: RunRef) -> str:
    """``<out>/<source>/<arm>/q<q>/seed<seed>.csv``."""
    return os.path.join(out_dir, ref.source, ref.arm, f"q{ref.q}", f"seed{ref.seed}.csv")


# ---------------------------------------------------------------------------------------- validation
def _read_checks(path: Path) -> Dict[int, Dict[str, str]]:
    with open(path, newline="") as f:
        return {int(r["update"]): r for r in csv.DictReader(f)}


def _absdiff(a: float, b: float) -> float:
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if math.isnan(a) or math.isnan(b):
        return float("inf")
    return abs(a - b)


def validate_rows(rows: Sequence[Row], checks: Mapping[int, Mapping[str, str]]) -> Dict[str, object]:
    """Replay against the run's own check rows at the same updates (D3 columns, closed-form columns, valid).

    Returns:
        ``n_exports``, ``n_matched`` (exports with a check row), ``max_abs_d3``, ``max_abs_closed`` (inf when
        exactly one side is NaN) and ``n_valid_mismatch``.
    """
    n_match, max_d3, max_cf, n_valid = 0, 0.0, 0.0, 0
    for r in rows:
        lg = checks.get(int(r["update"]))
        if lg is None:
            continue
        n_match += 1
        for c in D3_COLS:
            max_d3 = max(max_d3, _absdiff(float(r[c]), float(lg[c]) if lg[c] != "" else float("nan")))
        for c in CHECK_CLOSED:
            max_cf = max(max_cf, _absdiff(float(r[c]), float(lg[c]) if lg[c] != "" else float("nan")))
        n_valid += int(bool(r["valid"]) != (lg["valid"] == "True"))
    return {"n_exports": len(rows), "n_matched": n_match, "max_abs_d3": max_d3, "max_abs_closed": max_cf,
            "n_valid_mismatch": n_valid}


def _worker(job: Dict[str, Any]) -> Dict[str, Any]:
    ref = RunRef(**job["ref"])
    with open(ref.run_dir / "run_config.json") as f:
        rec = json.load(f)["record"]
    spec = GameSpec(**rec["game"])
    ppo = rec["ppo"]
    with open(ref.run_dir / "train_history.json") as f:
        lr_of = {int(h["update"]): float(h["actor_lr"]) for h in json.load(f)["history"] if "actor_lr" in h}
    t0 = time.perf_counter()
    rows: List[Row] = []
    for u in job["updates"]:
        arrays = load_export(str(ref.run_dir / "weights" / f"u{u:05d}.npz"))
        net = build_actor(arrays, int(ppo["hidden"]), float(ppo["c_min"]), float(ppo["mu_clamp"]))
        row: Row = {"source": ref.source, "arm": ref.arm, "q": ref.q, "seed": ref.seed, "update": u,
                    "actor_lr": lr_of.get(u, float("nan")), "conc_scale": conc_scale_of(arrays)}
        row.update(export_row(spec, net))
        rows.append(row)
    write_new_csv(job["csv"], rows, CSV_COLS)
    summ: Dict[str, object] = {"key": ref.key, "source": ref.source, "arm": ref.arm, "q": ref.q,
                               "seed": ref.seed, "last_update": job["updates"][-1] if job["updates"] else -1,
                               "n_invalid": int(sum(not r["valid"] for r in rows)),
                               "n_errors": int(sum(bool(r["error"]) for r in rows))}
    summ.update(validate_rows(rows, _read_checks(ref.run_dir / CHECK_FILE)))
    summ["replay_wall_sec"] = time.perf_counter() - t0
    return summ


def cmd_replay(args: argparse.Namespace) -> int:
    """``replay``: preflight, one CSV per run, ``validation_summary.csv``, ``manifest.json``."""
    sources = [parse_source(s) for s in args.source]
    qs, seeds = [int(q) for q in args.qs], parse_seeds(args.seeds)
    runs = discover(sources, qs, seeds)
    updates: Dict[str, List[int]] = {}
    missing: List[str] = []
    for ref in runs:
        us, miss = terminal_exports(ref.run_dir, args.updates)
        updates[ref.key] = us
        missing += miss
    if missing or not runs:
        print(f"STOP: {len(missing)} required input file(s) are missing (nothing was computed).")
        for m in missing[:20]:
            print("  missing:", m)
        return 2
    out_dir = os.path.abspath(args.out)
    for _, root, _ in sources:
        rt = os.path.abspath(root)
        if os.path.commonpath([out_dir, rt]) == rt:
            print(f"STOP: --out lies inside the (read-only) source root {rt}")
            return 2
    man_path = os.path.join(out_dir, "manifest.json")
    targets = [run_csv_path(out_dir, r) for r in runs] + [man_path, os.path.join(out_dir, "validation_summary.csv")]
    existing = [p for p in targets if os.path.exists(p)]
    if existing:
        print(f"STOP: {len(existing)} output file(s) already exist under {out_dir} (never overwritten), "
              f"first: {existing[0]}")
        return 2
    workers = max(1, min(int(args.workers), MAX_WORKERS))
    jobs = [{"ref": dataclasses.asdict(r), "updates": updates[r.key], "csv": run_csv_path(out_dir, r)}
            for r in runs]
    t0 = time.perf_counter()
    load_start = os.getloadavg()
    if workers == 1:
        results = [_worker(j) for j in jobs]
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(_worker, jobs))
    wall = time.perf_counter() - t0
    write_new_csv(os.path.join(out_dir, "validation_summary.csv"), results, list(results[0].keys()))
    n_rows = int(sum(int(r["n_exports"]) for r in results))
    max_d3 = max(float(r["max_abs_d3"]) for r in results)
    max_cf = max(float(r["max_abs_closed"]) for r in results)
    ok = bool(max_d3 <= CHECK_TOL and max_cf <= CHECK_TOL and all(r["n_valid_mismatch"] == 0 for r in results))
    import scipy
    manifest = {
        "tool": "tools/ms/r2_stop_candidates.py", "tool_sha256": sha256_file(Path(__file__).resolve()),
        "command": " ".join(sys.argv), "git_head": git_head(),
        "sources": {name: {"root": root, "arms": arms} for name, root, arms in sources},
        "qs": qs, "seeds": seeds, "restricted_updates": args.updates, "n_runs": len(runs), "n_rows": n_rows,
        "verifier_tier": dataclasses.asdict(DEV_CONFIG),
        "settings": {"h_fd": H_FD, "excl_abs": EXCL_ABS, "j_away": J_AWAY, "bisect_iters": BISECT_ITERS,
                     "dense_step": DENSE_STEP, "stage": STAGE, "const_lr": CONST_LR},
        "workers": workers, "loadavg_at_start": list(load_start), "wall_sec": wall,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "validation": {"n_runs": len(results), "n_exports_matched": int(sum(int(r["n_matched"]) for r in results)),
                       "max_abs_d3": max_d3, "max_abs_closed": max_cf, "ok": ok,
                       "tolerance": CHECK_TOL},
        "note": "numeric columns deterministic; replay_wall_sec is wall-clock"}
    write_new_json(man_path, manifest)
    print(f"replayed {len(runs)} runs ({n_rows} exports) in {wall:.1f}s; vs the runs' own check rows: "
          f"{manifest['validation']['n_exports_matched']} exports matched, max |diff| D3 {max_d3:.3g}, "
          f"closed-form {max_cf:.3g}, valid-flag mismatches {sum(int(r['n_valid_mismatch']) for r in results)}")
    if not ok:
        print("STOP: the replay does not reproduce the runs' own check rows (see validation_summary.csv).")
        return 3
    return 0


# ================================================================================================ tables
def load_runs(out_dir: str) -> Dict[RunKey, List[Row]]:
    """All per-run CSVs under ``out_dir`` (rows sorted by update), keyed by (source, arm, q, seed)."""
    runs: Dict[RunKey, List[Row]] = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "*", "*", "q*", "seed*.csv"))):
        rows = sorted(read_run_csv(p), key=lambda r: int(r["update"]))
        if rows:
            r0 = rows[0]
            runs[(str(r0["source"]), str(r0["arm"]), int(r0["q"]), int(r0["seed"]))] = rows
    return runs


def groups_of(runs: Mapping[RunKey, Any]) -> List[Tuple[str, Optional[Tuple[str, str]]]]:
    """``('all', None)`` followed by one ``('source/arm', (source, arm))`` per source/arm present."""
    arms = sorted({(k[0], k[1]) for k in runs})
    return [("all", None)] + [(f"{s}/{a}", (s, a)) for s, a in arms]


def select_runs(runs: Mapping[RunKey, List[Row]], grp: Optional[Tuple[str, str]], q: Optional[int]
                ) -> List[List[Row]]:
    """Run row lists of a group (``None`` = all arms) and q (``None`` = both), in key order."""
    return [rows for k, rows in sorted(runs.items())
            if (grp is None or (k[0], k[1]) == grp) and (q is None or k[2] == q)]


def _sel(rows: Sequence[Row], const_lr: bool) -> List[Row]:
    return [r for r in rows if r["valid"] and int(r["update"]) >= SPEARMAN_FROM_U
            and (not const_lr or r["actor_lr"] == CONST_LR)]


def spearman_cell(runs: Sequence[Sequence[Row]], cand: str, target: str, const_lr: bool
                  ) -> Tuple[int, float, float]:
    """``(n_pairs, pooled Spearman, median within-run Spearman)`` over valid exports with u >= 400."""
    xs: List[float] = []
    ys: List[float] = []
    per: List[float] = []
    for rows in runs:
        sel = _sel(rows, const_lr)
        x, y = [float(r[cand]) for r in sel], [float(r[target]) for r in sel]
        xs += x
        ys += y
        per.append(spearman(x, y))
    n = int(sum(1 for a, b in zip(xs, ys) if math.isfinite(a) and math.isfinite(b)))
    return n, spearman(xs, ys), med_rng(per)[0]


def table_spearman(runs: Mapping[RunKey, List[Row]]) -> List[Table]:
    """Tables (a): Spearman of the candidates (and R, Delta) with |peak| and RMSE_pos, per q and per arm."""
    qs = sorted({k[2] for k in runs})
    subsets = (("all", False), ("constLR", True))
    cr: List[Dict[str, object]] = []
    for gname, grp in groups_of(runs):
        for q in qs:
            sel_runs = select_runs(runs, grp, q)
            if not sel_runs:
                continue
            for sname, clr in subsets:
                for cname, ccol in SPEARMAN_COLS:
                    for tname, tcol in TARGETS:
                        n, rho, wr = spearman_cell(sel_runs, ccol, tcol, clr)
                        cr.append({"group": gname, "q": q, "subset": sname, "candidate": cname,
                                   "target": tname, "n_runs": len(sel_runs), "n_pairs": n,
                                   "spearman_pooled": rho, "within_run_median": wr})
    cols = ["group", "q", "subset", "candidate", "target", "n_runs", "n_pairs", "spearman_pooled",
            "within_run_median"]
    full = Table("a_spearman", "", cols, cr, [], [])
    look = {(r["group"], r["q"], r["subset"], r["candidate"], r["target"]): r for r in cr}

    def cell(g: str, q: int, s: str, c: str, t: str) -> str:
        r = look.get((g, q, s, c, t))
        return "-" if r is None else f"{fnum(r['spearman_pooled'], 3)} ({fnum(r['within_run_median'], 2)})"

    md_rows: List[List[object]] = []
    for q in qs:
        for cname, _ in SPEARMAN_COLS:
            r_all = look.get(("all", q, "all", cname, "|peak|"))
            r_clr = look.get(("all", q, "constLR", cname, "|peak|"))
            md_rows.append([q, cname] + [cell("all", q, s, cname, t) for s, _ in subsets for t, _ in TARGETS]
                           + [f"{r_all['n_pairs'] if r_all else '-'} / {r_clr['n_pairs'] if r_clr else '-'}"])
    md_cols = ["q", "candidate", "all u>=400: |peak|", "all u>=400: RMSE_pos", "const LR: |peak|",
               "const LR: RMSE_pos", "n pairs (all / constLR)"]
    pooled = Table("a_spearman_pooled_md", "Table (a). Spearman correlation of the candidates with the closed-form "
                   "errors, pooled over all runs and arms, per q: pooled value (median of the within-run "
                   "values); valid exports with u >= 400; C2 = |c2|", cols, [], md_cols, md_rows,
                   "n pairs counts exports with finite candidate and error.")
    tabs: List[Table] = [full, pooled]
    names = [g for g, _ in groups_of(runs)][1:]
    for sname, _ in subsets:
        for tname, _ in TARGETS:
            rr: List[List[object]] = []
            for g in names:
                for q in qs:
                    if (g, q, sname, "C1", tname) in look:
                        rr.append([g, q] + [cell(g, q, sname, c, tname) for c, _ in SPEARMAN_COLS[:5]])
            tabs.append(Table(f"a_by_arm_{sname}_{tname}_md", f"Table (a), per source/arm: Spearman with {tname} "
                              f"({'constant-LR exports' if sname == 'constLR' else 'all exports'} with u >= 400)"
                              f", pooled (within-run median)", cols, [], ["source/arm", "q"]
                              + [c for c, _ in SPEARMAN_COLS[:5]], rr))
    return tabs


def freeze_rows(runs: Mapping[RunKey, List[Row]], grp: Optional[Tuple[str, str]], q: int) -> List[Row]:
    """The last terminal-stage export (the freeze candidate) of every valid run of a group and q."""
    return [rows[-1] for rows in select_runs(runs, grp, q) if rows[-1]["valid"]]


def table_freeze(runs: Mapping[RunKey, List[Row]]) -> List[Table]:
    """Tables (b): min / median / max of the candidates at the freeze, per q (pooled and per arm)."""
    qs = sorted({k[2] for k in runs})
    cr: List[Dict[str, object]] = []
    for gname, grp in groups_of(runs):
        for q in qs:
            fr = freeze_rows(runs, grp, q)
            if not fr:
                continue
            for cname, ccol in FREEZE_COLS:
                med, lo, hi = med_rng([float(r[ccol]) for r in fr])
                cr.append({"group": gname, "q": q, "candidate": cname, "n": len(fr), "min": lo, "median": med,
                           "max": hi})
    cols = ["group", "q", "candidate", "n", "min", "median", "max"]
    md_rows = [[r["q"], r["candidate"], r["n"], fnum(r["min"], 4), fnum(r["median"], 4), fnum(r["max"], 4)]
               for r in cr if r["group"] == "all"]
    pooled = Table("b_freeze", "Table (b). Distribution at the freeze (the last terminal-stage export of every "
                   "valid run), per q, all runs and arms pooled; C2 floor = sigma_2(0) / (sqrt(pi) q)", cols, cr,
                   ["q", "candidate", "n runs", "min", "median", "max"], md_rows)
    by_arm: List[List[object]] = []
    for gname, _ in groups_of(runs)[1:]:
        for q in qs:
            row: List[object] = [gname, q]
            have = False
            for cname in ("C1", "C2 (|c2|)", "C2 floor", "C3", "|peak|"):
                m = [r for r in cr if r["group"] == gname and r["q"] == q and r["candidate"] == cname]
                have = have or bool(m)
                row.append("-" if not m else f"{fnum(m[0]['median'], 4)} [{fnum(m[0]['min'], 4)}, "
                                              f"{fnum(m[0]['max'], 4)}]")
            if have:
                by_arm.append(row)
    arm_t = Table("b_freeze_by_arm_md", "Table (b), per source/arm: median [min, max] at the freeze", cols, [],
                  ["source/arm", "q", "C1", "C2 (|c2|)", "C2 floor", "C3", "|peak|"], by_arm)
    return [pooled, arm_t]


def table_jcheck(runs: Mapping[RunKey, List[Row]]) -> Table:
    """Table (c): the finite-difference J against the linearisation, over all valid exports, per q."""
    qs = sorted({k[2] for k in runs})
    cr: List[Dict[str, object]] = []
    for q in qs:
        rows = [r for rs in select_runs(runs, None, q) for r in rs if r["valid"]]
        for side in ("neg", "pos"):
            m = np.array([float(r[f"J_mean_{side}"]) for r in rows])
            lin = np.array([float(r[f"J_lin_{side}"]) for r in rows])
            ok = np.isfinite(m) & np.isfinite(lin)
            dev = m[ok] - lin[ok]
            cr.append({"q": q, "side": "d < -8" if side == "neg" else "d > +8", "n_exports": int(ok.sum()),
                       "mean_J": float(m[ok].mean()) if ok.any() else float("nan"),
                       "J_lin": float(lin[ok].mean()) if ok.any() else float("nan"),
                       "mean_abs_dev": float(np.abs(dev).mean()) if ok.any() else float("nan"),
                       "max_abs_dev": float(np.abs(dev).max()) if ok.any() else float("nan")})
    cols = ["q", "side", "n_exports", "mean_J", "J_lin", "mean_abs_dev", "max_abs_dev"]
    md = [[r["q"], r["side"], r["n_exports"], fnum(r["mean_J"], 4), fnum(r["J_lin"], 4),
           fnum(r["mean_abs_dev"], 4), fnum(r["max_abs_dev"], 4)] for r in cr]
    return Table("c_jcheck", "Table (c). J check: per export the mean of the finite-difference J over the "
                 "non-tail nodes with d < -8 (d > +8) against the linearisation J - 1 = -2k/(2k -+ a); mean over "
                 "all valid exports and mean / max of the absolute deviation of the per-export mean", cols, cr,
                 ["q", "side", "n exports", "mean J", "linearised J", "mean |dev|", "max |dev|"], md)


def table_diagnostics(runs: Mapping[RunKey, List[Row]]) -> Table:
    """Extra diagnostics: dev-tier best-response grid error (s vs s_exact) and the nodes excluded from C3."""
    qs = sorted({k[2] for k in runs})
    cr: List[Dict[str, object]] = []
    for q in qs:
        rows = [r for rs in select_runs(runs, None, q) for r in rs if r["valid"]]
        ds = np.array([abs(float(r["s"]) - float(r["s_exact"])) for r in rows])
        dr = np.array([abs(float(r["R0"]) - float(r["R0_exact"])) for r in rows])
        ex = np.array([int(r["n_defl_excluded"]) for r in rows])
        cr.append({"q": q, "n_exports": len(rows), "abs_s_minus_s_exact_median": float(np.nanmedian(ds)),
                   "abs_s_minus_s_exact_max": float(np.nanmax(ds)),
                   "abs_R0_minus_R0_exact_median": float(np.nanmedian(dr)),
                   "abs_R0_minus_R0_exact_max": float(np.nanmax(dr)),
                   "n_defl_nodes_median": float(np.median([int(r["n_defl_nodes"]) for r in rows])),
                   "n_defl_excluded_total": int(ex.sum()), "n_exports_with_exclusion": int((ex > 0).sum()),
                   "n_defl_excluded_max": int(ex.max())})
    cols = list(cr[0].keys())
    md = [[r["q"], r["n_exports"], fnum(r["abs_s_minus_s_exact_median"], 4), fnum(r["abs_s_minus_s_exact_max"], 4),
           fnum(r["abs_R0_minus_R0_exact_median"], 4), fnum(r["abs_R0_minus_R0_exact_max"], 4),
           fnum(r["n_defl_nodes_median"], 0), r["n_defl_excluded_total"], r["n_exports_with_exclusion"],
           r["n_defl_excluded_max"]] for r in cr]
    return Table("x_diagnostics", "Diagnostics. Development-tier best-response grid error (|s - s_exact| in effort "
                 "units, |R0 - R0_exact|) and the nodes excluded from C3 (|1 - J| < 0.1), valid exports", cols, cr,
                 ["q", "n exports", "|s - s_exact| median", "|s - s_exact| max", "|R0 - R0_exact| median",
                  "|R0 - R0_exact| max", "non-tail nodes", "excluded nodes (total)", "exports with exclusion",
                  "excluded max per export"], md)


# ------------------------------------------------------------------------------------------------ fire
def fire_export(updates: Sequence[int], values: Sequence[float], valid: Sequence[bool], theta: float,
                m: int = FIRE_M, cadence: int = FIRE_K) -> Optional[int]:
    """The export at which ``values <= theta`` holds for the ``m``-th consecutive check (None if never).

    The exports are the checks: an export at a multiple of ``cadence`` is eligible iff it is valid and its
    value is finite and ``<= theta``; the streak is reset by a failing (or invalid, or non-finite) check and
    by nothing else (an export off the cadence is not a check and does not interrupt the streak).

    Args:
        updates: Terminal-stage export updates in increasing order.
        values: The candidate at each export.
        valid: Validity of each export.
        theta: Threshold.
        m: Consecutive eligible checks required.
        cadence: Check cadence in updates.
    """
    streak = 0
    for u, v, ok in zip(updates, values, valid):
        if int(u) % cadence:
            continue
        streak = streak + 1 if (bool(ok) and math.isfinite(v) and v <= theta) else 0
        if streak >= m:
            return int(u)
    return None


def load_grids(path: str) -> Dict[str, List[float]]:
    """``{"C1": [...], "C2": [...], "C3": [...]}`` from a JSON file (unknown keys are an error)."""
    with open(path) as f:
        g = json.load(f)
    names = {c for c, _ in FIRE_CANDIDATES}
    if not isinstance(g, dict) or not g or set(g) - names:
        raise ValueError(f"grids file must be a JSON object with keys among {sorted(names)}")
    return {c: [float(x) for x in g[c]] for c, _ in FIRE_CANDIDATES if c in g}


def git_commit_of(path: str) -> str:
    """``git log -1 --format=%H -- <path>`` ('' if the file is untracked or not in a repository)."""
    p = Path(path).resolve()
    try:
        return subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(p)], cwd=str(p.parent),
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def table_fire(runs: Mapping[RunKey, List[Row]], grids: Mapping[str, Sequence[float]]) -> List[Table]:
    """Tables (d): for each candidate and theta the first fire export and the errors at the fire."""
    qs = sorted({k[2] for k in runs})
    col_of = dict(FIRE_CANDIDATES)
    cr: List[Dict[str, object]] = []
    for cname in (c for c, _ in FIRE_CANDIDATES if c in grids):
        for theta in grids[cname]:
            for gname, grp in groups_of(runs):
                for q in qs:
                    sel = select_runs(runs, grp, q)
                    if not sel:
                        continue
                    fires: List[Tuple[int, Row, Row]] = []
                    for rows in sel:
                        fu = fire_export([int(r["update"]) for r in rows], [float(r[col_of[cname]]) for r in rows],
                                         [bool(r["valid"]) for r in rows], theta)
                        if fu is not None:
                            at = next(r for r in rows if int(r["update"]) == fu)
                            fires.append((fu, at, rows[-1]))
                    row: Dict[str, object] = {"candidate": cname, "theta": theta, "group": gname, "q": q,
                                              "n_runs": len(sel), "n_fired": len(fires)}
                    fm, flo, fhi = med_rng([float(f[0]) for f in fires])
                    row.update(fire_u_median=fm, fire_u_min=flo, fire_u_max=fhi)
                    for mname, mcol in FIRE_METRICS:
                        for tag, idx in (("at_fire", 1), ("at_end", 2)):
                            mm, lo, hi = med_rng([float(f[idx][mcol]) for f in fires])
                            row.update({f"{mname}_{tag}_median": mm, f"{mname}_{tag}_min": lo,
                                        f"{mname}_{tag}_max": hi})
                    pk = "stage2_peak_rel_err_abs"
                    row.update(n_peak_le_0p05_at_fire=int(sum(float(f[1][pk]) <= PEAK_TOL for f in fires)),
                               n_peak_le_0p05_at_end_of_fired=int(sum(float(f[2][pk]) <= PEAK_TOL for f in fires)),
                               n_peak_le_0p05_at_end_all_runs=int(sum(float(rows[-1][pk]) <= PEAK_TOL
                                                                      for rows in sel)))
                    cr.append(row)
    cols = list(cr[0].keys()) if cr else []
    pooled_md: List[List[object]] = []
    arm_md: List[List[object]] = []
    for r in cr:
        fu = f"{fnum(r['fire_u_median'], 0)} [{fnum(r['fire_u_min'], 0)}, {fnum(r['fire_u_max'], 0)}]"
        if r["group"] == "all":
            pooled_md.append([r["candidate"], fnum(r["theta"], 4), r["q"], f"{r['n_fired']}/{r['n_runs']}", fu,
                              f"{fnum(r['abs_peak_at_fire_median'], 4)} [{fnum(r['abs_peak_at_fire_min'], 4)}, "
                              f"{fnum(r['abs_peak_at_fire_max'], 4)}]",
                              f"{fnum(r['abs_peak_at_end_median'], 4)} [{fnum(r['abs_peak_at_end_min'], 4)}, "
                              f"{fnum(r['abs_peak_at_end_max'], 4)}]",
                              fnum(r["signed_peak_at_fire_median"], 4), fnum(r["signed_peak_at_end_median"], 4),
                              fnum(r["rmse_pos_at_fire_median"], 4), fnum(r["rmse_pos_at_end_median"], 4),
                              fnum(r["tail_mean_at_fire_median"], 4), fnum(r["tail_mean_at_end_median"], 4),
                              f"{r['n_peak_le_0p05_at_fire']}/{r['n_fired']}"])
        else:
            arm_md.append([r["group"], r["candidate"], fnum(r["theta"], 4), r["q"], f"{r['n_fired']}/{r['n_runs']}",
                           fnum(r["fire_u_median"], 0), fnum(r["abs_peak_at_fire_median"], 4),
                           fnum(r["abs_peak_at_end_median"], 4), fnum(r["rmse_pos_at_fire_median"], 4),
                           fnum(r["rmse_pos_at_end_median"], 4),
                           f"{r['n_peak_le_0p05_at_fire']}/{r['n_fired']}"])
    pooled = Table("fire", "Table (d). Fire of the rule 'candidate <= theta at M = 3 consecutive terminal-stage "
                   "exports (K = 25)', all runs and arms pooled, per q: fire update median [min, max] and the "
                   "closed-form errors at the fire against the same (fired) runs' values at the end of the run, "
                   "median [min, max] or median", cols, cr,
                   ["candidate", "theta", "q", "fired", "fire u", "|peak| at fire", "|peak| at end", "signed at "
                    "fire", "signed at end", "RMSE_pos at fire", "RMSE_pos at end", "tail mean at fire",
                    "tail mean at end", "|peak| <= 0.05 at fire"], pooled_md)
    by_arm = Table("fire_by_arm_md", "Table (d), per source/arm (medians over the fired runs)", cols, [],
                   ["source/arm", "candidate", "theta", "q", "fired", "fire u median", "|peak| at fire",
                    "|peak| at end", "RMSE_pos at fire", "RMSE_pos at end", "|peak| <= 0.05 at fire"], arm_md)
    return [pooled, by_arm]


# -------------------------------------------------------------------------------------------- command
def _emit(tabs: Sequence[Table], out_dir: str) -> List[str]:
    """Write the CSV of every table that has CSV rows; returns the paths."""
    paths = []
    for t in tabs:
        if t.csv_rows:
            p = os.path.join(out_dir, "tables", f"{t.name}.csv")
            write_new_csv(p, t.csv_rows, t.csv_cols)
            paths.append(p)
    return paths


def cmd_tables(args: argparse.Namespace) -> int:
    """``tables``: Spearman / freeze / J-check / diagnostics tables, or (``--fire --grids``) the fire tables."""
    out_dir = os.path.abspath(args.out)
    if args.grids and not args.fire:
        print("STOP: --grids is only used with --fire", file=sys.stderr)
        return 2
    if args.fire and not args.grids:
        print("STOP: --fire needs --grids <grids.json> (the threshold grids are recorded before the fire "
              "tables are computed); nothing was computed", file=sys.stderr)
        return 2
    runs = load_runs(out_dir)
    if not runs:
        print(f"STOP: no per-run CSVs under {out_dir}; run `replay` first", file=sys.stderr)
        return 2
    names = (["fire.csv", "fire_meta.json"] if args.fire
             else ["a_spearman.csv", "b_freeze.csv", "c_jcheck.csv", "x_diagnostics.csv"])
    existing = [n for n in names if os.path.exists(os.path.join(out_dir, "tables", n))]
    if existing:
        print(f"STOP: {existing} already exist under {out_dir}/tables (never overwritten)", file=sys.stderr)
        return 2
    if args.fire:
        grids = load_grids(args.grids)
        tabs = table_fire(runs, grids)
        write_new_json(os.path.join(out_dir, "tables", "fire_meta.json"), {
            "grids_file": os.path.abspath(args.grids), "grids_sha256": sha256_file(Path(args.grids)),
            "grids_git_commit": git_commit_of(args.grids), "grids": grids,
            "rule": {"M": FIRE_M, "K": FIRE_K, "columns": dict(FIRE_CANDIDATES), "valid_only": True,
                     "peak_tol": PEAK_TOL}, "n_runs": len(runs), "tool_sha256": sha256_file(Path(__file__).resolve()),
            "command": " ".join(sys.argv)})
    else:
        tabs = table_spearman(runs) + table_freeze(runs) + [table_jcheck(runs), table_diagnostics(runs)]
    paths = _emit(tabs, out_dir)
    print(f"wrote {len(paths)} table CSV(s) under {out_dir}/tables from {len(runs)} runs", file=sys.stderr)
    if args.md:
        print("\n".join(t.md() for t in tabs if t.md_rows))
    return 0


# ----------------------------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the tool."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("replay", help="replay the development verifier on every terminal-stage export")
    p.add_argument("--source", action="append", required=True, metavar="NAME=ROOT:ARM1,ARM2",
                   help="a run root with layout ROOT/q{q}/seed{seed}/<ARM>/ (repeatable; read only)")
    p.add_argument("--qs", nargs="+", type=int, required=True, help="q values (50 60)")
    p.add_argument("--seeds", nargs="+", required=True, help="seeds: 10501-10510 and / or 10501,10503")
    p.add_argument("--out", required=True, help="output directory (results/ms_r2/stop_calibration)")
    p.add_argument("--workers", type=int, default=8, help=f"processes (single-threaded each; max {MAX_WORKERS})")
    p.add_argument("--updates", nargs="+", type=int, default=None, help="restrict to these export updates (debug)")
    p.set_defaults(func=cmd_replay)
    t = sub.add_parser("tables", help="tables (a)-(c) from the per-run CSVs, or (d) with --fire --grids")
    t.add_argument("--out", required=True, help="the --out of the replay; tables/ goes here")
    t.add_argument("--fire", action="store_true", help="compute the fire tables (needs --grids)")
    t.add_argument("--grids", default=None, help='{"C1": [theta...], "C2": [...], "C3": [...]} for --fire')
    t.add_argument("--md", action="store_true", help="print the tables as markdown to stdout")
    t.set_defaults(func=cmd_tables)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
