#!/usr/bin/env python3
"""MS-R3 prompt 2.3 (b): RL-actor diagnostics on the weight exports of MS-R1 and MS-R2 runs.

Per export (``<run>/weights/u?????.npz``, every 25 updates; ``u`` is the global update index):

* the first-layer weights on the d input in units of d / B, ``actor.l1.weight[:, 1]`` (at T = 2
  stage 2 the d component of the actor input is d / B). The ``t10`` actor feeds ``10 d / B`` to
  the stored weight, so for a ``t10`` export the reported weight is ten times the stored one.
  Reported: max |w|, the quantiles 0, 10, 25, 50, 75, 90, 100 of |w| over the 64 units (linear
  interpolation, ``numpy.quantile``) and the number of units with |w| strictly above 1, 2, 3, 5;
* ``e_hat_2_0``: the Beta-mean effort at the stage-2 observation of d = 0,
  ``GameSpec.encode_obs(2, [0.0])``, from ``agents.ppo_curriculum.mean_effort_numpy`` with the
  variant read from the export (an export of another variant is never read as tanh d / B);
* ``e_star_2_0``: the closed-form tent peak ``utils.theory_multistage.g2_two_stage(0, ...)``,
  which ``utils.v2_metrics.recovery_metrics`` stores as ``g2_at_0`` (per_run.csv ``g2_at_0``);
* ``gap = e_star_2_0 - e_hat_2_0`` and ``w_eff = gap / (e_star_2_0 / (2 q))``, the effective
  rounding width in units of d (the tent slope is ``e_star_2_0 / (2 q)``). ``w_eff`` exists for
  terminal-stage exports only (``stage`` 2 in ``ms_updates.csv``; that stage is trained first, so
  its local update equals the global one): in the stage-1 phase the live actor is trained on the
  root game and its output at the stage-2 observation is no longer the terminal policy (NaN
  there). ``e_hat_2_0`` and ``gap`` are computed for every export. ``local`` is the
  ``ms_updates.csv`` column of that name.

Inputs are explicit and every root is recorded in ``manifest.json``. Layout
``<root>/q<q>/seed<seed>/<arm>/`` with ``weights/u*.npz``, ``ms_updates.csv`` (update, stage,
local) and ``run_config.json`` (``record.game``, ``record.ppo``). A declared root that does not
exist, a run without weights, an export that ``ms_updates.csv`` does not list and an export of an
unknown variant are errors. The tool writes only into ``--out`` and refuses to overwrite.

Outputs (``--out``):

* ``per_export.csv``: one row per export, all stages with the stage tag, ``w_eff`` NaN in stage 1.
* ``per_arm.csv``: per source / arm / q, medians (and means) over the seeds of max |w| and w_eff
  at the terminal-stage export nearest to and not after each update of ``TARGET_UPDATES``
  (``which`` = ``u<target>``; ``update_used_*`` show the export actually used, the run's last
  terminal-stage export when its budget is shorter than the target) and at the final export
  (``which`` = ``final``: the last terminal-stage export of each run, i.e. the terminal freeze).
* ``relation.csv``: per source / arm / q, Spearman and Pearson correlation of max |w| with w_eff
  over all terminal-stage exports pooled over the seeds, the median over the runs of the same
  two correlations within a run and, supplementary, the two across the seeds at the final export.
* ``regime.csv`` (needs ``--screen-dir`` with ``summary_by_cell.csv``): the share of RL exports
  with max |w| at or below the reference, the median over the seeds of the supervised screen's
  ``max_abs_w_d`` for actor ``t1``, bin-balanced starts, 56,000 steps (main-grid cells: the
  screen's ``extended`` cell is left out), per q. Shares over all exports, over terminal-stage
  exports and over final exports. Descriptive; no gate. ``--screen-filter COL=VALUE`` adds
  equality filters on the screen rows (not needed for the schema of ``r3_supervised_screen``).
* ``manifest.json`` and ``summary.txt`` (the MS-R2 w_eff at the freeze next to the PI's preamble).

Usage:
    python tools/ms/r3_actor_diagnostics.py \\
        --ms-r2-pilot-root <worktree>/results/ms_r2/pilot \\
        --ms-r1-pilot-root <worktree>/results/ms_r1/pilot \\
        --ms-r1-base-root <ms-r1 worktree>/results/ms_r1/base \\
        [--screen-dir results/ms_r3/supervised_screen] --workers 4 \\
        --out results/ms_r3/rl_actor_diagnostics
"""

from __future__ import annotations

import os

for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"       # tiny numpy work; every worker single-threaded

import sys  # noqa: E402

if __name__ == "__main__":
    sys.dont_write_bytecode = True              # as a script, the only files written are in --out

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402
import subprocess  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.ppo_curriculum import (  # noqa: E402
    ACTOR_VARIANTS, D_FEATURE_SCALE_T10, mean_effort_numpy)
from envs.curriculum_env import GameSpec  # noqa: E402
from utils.theory_multistage import g2_two_stage  # noqa: E402

#: Source labels in output order and the flag that declares each root.
SOURCE_ORDER: Tuple[str, ...] = ("ms_r2_pilot", "ms_r1_pilot", "ms_r1_base")
ROOT_FLAGS: Dict[str, str] = {
    "ms_r2_pilot": "--ms-r2-pilot-root", "ms_r1_pilot": "--ms-r1-pilot-root",
    "ms_r1_base": "--ms-r1-base-root"}
TERMINAL_STAGE = 2          # T = 2: the stage trained first (global update == local update)
QUANTILES: Tuple[int, ...] = (0, 10, 25, 50, 75, 90, 100)
COUNT_THRESHOLDS: Tuple[int, ...] = (1, 2, 3, 5)
TARGET_UPDATES: Tuple[int, ...] = (400, 1200, 1600, 2000, 2400, 2800)
#: The PI's preamble figure (``20_ms_r3_prompt.md``, preamble item 2): gap / tent slope averaged
#: over the six MS-R2 arms, from the arm means of ``results/ms_r2/analysis/per_run.csv``.
#: Quoted in the summary next to the tool's own number; never used in a computation.
PREAMBLE_W_EFF: Dict[int, float] = {50: 4.85, 60: 5.33}
SCREEN_CSV = "summary_by_cell.csv"
SCREEN_STEPS = 56000
SCREEN_BB_LABELS = ("bb", "binbalanced", "balanced")   # normalised labels of bin-balanced starts
OUTPUT_FILES: Tuple[str, ...] = (
    "per_export.csv", "per_arm.csv", "relation.csv", "regime.csv", "manifest.json", "summary.txt")
EXPORT_COLUMNS: Tuple[str, ...] = (
    "source", "arm", "q", "seed", "update", "stage", "local", "variant", "max_abs_w_d",
    *[f"absw_q{p}" for p in QUANTILES], *[f"n_absw_gt{t}" for t in COUNT_THRESHOLDS],
    "e_hat_2_0", "e_star_2_0", "gap", "w_eff")
PER_ARM_COLUMNS: Tuple[str, ...] = (
    "source", "arm", "q", "which", "target_update", "n_runs", "update_used_min",
    "update_used_median", "update_used_max", "median_max_abs_w_d", "mean_max_abs_w_d",
    "median_w_eff", "mean_w_eff")
RELATION_COLUMNS: Tuple[str, ...] = (
    "source", "arm", "q", "n_runs", "n_pooled", "pooled_spearman", "pooled_pearson",
    "n_runs_within", "within_run_median_spearman", "within_run_median_pearson",
    "n_final", "final_cross_seed_spearman", "final_cross_seed_pearson")

_EXPORT_RE = re.compile(r"^u(\d+)\.npz$")
RunKey = Tuple[str, str, int, int]


class DiagnosticsError(RuntimeError):
    """A declared input is missing or malformed (the tool never skips an input silently)."""


@dataclass(frozen=True)
class RunJob:
    """One run directory ``<root>/q<q>/seed<seed>/<arm>``."""

    source: str
    arm: str
    q: int
    seed: int
    run_dir: str


# ---------------------------------------------------------------------------- one export
def export_variant(weights: Mapping[str, np.ndarray]) -> str:
    """Actor variant of an export: its ``actor_variant`` entry, else ``"t1"``.

    Every MS-R1 and MS-R2 export is ``t1`` (no entry).

    Args:
        weights: Arrays of one weight export.

    Returns:
        ``"t1"``, ``"relu"`` or ``"t10"``.

    Raises:
        ValueError: The export names a variant this code does not know.
    """
    v = str(np.asarray(weights["actor_variant"])) if "actor_variant" in weights else "t1"
    if v not in ACTOR_VARIANTS:
        raise ValueError(f"unknown actor variant {v!r}; known: {ACTOR_VARIANTS}")
    return v


def d_weights(weights: Mapping[str, np.ndarray], variant: str) -> np.ndarray:
    """First-layer weights on the d input in units of d / B (signed, float64, per hidden unit).

    Column 1 of ``actor.l1.weight`` multiplies the d component of the actor input. The ``t10``
    actor feeds ``10 d / B`` to it, so its weight in units of d / B is ten times the stored one.

    Args:
        weights: Arrays of one weight export.
        variant: Variant of the export (:func:`export_variant`).

    Returns:
        Array of shape (hidden,).
    """
    w = np.asarray(weights["actor.l1.weight"], dtype=np.float64)[:, 1]
    return w * D_FEATURE_SCALE_T10 if variant == "t10" else w


def abs_weight_stats(w: np.ndarray) -> Dict[str, float]:
    """Max |w|, the quantiles of |w| and the counts of units with |w| strictly above thresholds.

    Args:
        w: Signed weights (any shape).

    Returns:
        Dict with ``max_abs_w_d``, ``absw_q<p>`` for p in ``QUANTILES`` (percent, linear
        interpolation) and ``n_absw_gt<t>`` for t in ``COUNT_THRESHOLDS``.
    """
    a = np.abs(np.asarray(w, dtype=np.float64)).reshape(-1)
    out: Dict[str, float] = {"max_abs_w_d": float(a.max())}
    for p, v in zip(QUANTILES, np.quantile(a, [p / 100.0 for p in QUANTILES])):
        out[f"absw_q{p}"] = float(v)
    for t in COUNT_THRESHOLDS:
        out[f"n_absw_gt{t}"] = int((a > t).sum())
    return out


def e_hat_stage2(weights: Mapping[str, np.ndarray], spec: GameSpec, d: float = 0.0,
                 c_min: float = 100.0, mu_clamp: float = 1e-6) -> float:
    """Beta-mean effort of an export at the stage-2 observation of gap ``d`` (float32 reload).

    Args:
        weights: Arrays of one weight export; the variant is read from it.
        spec: Game (``encode_obs`` input, effort bounds).
        d: Stage-2 gap.
        c_min: Concentration floor of the run (``record.ppo.c_min``).
        mu_clamp: Mean clamp of the run (``record.ppo.mu_clamp``).

    Returns:
        ``e_min + (e_max - e_min) alpha / (alpha + beta)``.
    """
    obs = spec.encode_obs(TERMINAL_STAGE, np.asarray([d], dtype=float))
    e, _, _ = mean_effort_numpy(
        dict(weights), obs, c_min=c_min, mu_clamp=mu_clamp, e_min=spec.e_min, e_max=spec.e_max,
        variant=export_variant(weights))
    return float(e[0])


def tent_peak(spec: GameSpec) -> float:
    """``e2*(0)``: the closed-form stage-2 effort at d = 0 (``g2_two_stage``, g2_at_0)."""
    return float(g2_two_stage(np.asarray(0.0), spec.q, spec.w_h, spec.w_l, spec.k, spec.e_max))


def w_eff_from_gap(gap: float, e_star_0: float, q: float) -> float:
    """Effective rounding width in units of d: ``gap / (e2*(0) / (2 q))``, the tent slope."""
    return float(gap) / (float(e_star_0) / (2.0 * float(q)))


def export_stats(weights: Mapping[str, np.ndarray], spec: GameSpec, e_star_0: float,
                 terminal: bool, c_min: float = 100.0, mu_clamp: float = 1e-6) -> Dict[str, Any]:
    """The per-export columns of one export except the run keys.

    Args:
        weights: Arrays of one weight export.
        spec: Game of the run.
        e_star_0: ``tent_peak(spec)``.
        terminal: Whether the export belongs to the terminal stage (``w_eff`` is NaN otherwise).
        c_min: Concentration floor of the run.
        mu_clamp: Mean clamp of the run.

    Returns:
        Dict with ``variant``, the |w| statistics, ``e_hat_2_0``, ``e_star_2_0``, ``gap`` and
        ``w_eff``.
    """
    variant = export_variant(weights)
    out: Dict[str, Any] = {"variant": variant, **abs_weight_stats(d_weights(weights, variant))}
    e_hat = e_hat_stage2(weights, spec, 0.0, c_min, mu_clamp)
    gap = e_star_0 - e_hat
    out.update(e_hat_2_0=e_hat, e_star_2_0=e_star_0, gap=gap,
               w_eff=w_eff_from_gap(gap, e_star_0, spec.q) if terminal else float("nan"))
    return out


# ---------------------------------------------------------------------------- one run
def load_run_config(path: Path) -> Tuple[GameSpec, float, float]:
    """Game, ``c_min`` and ``mu_clamp`` of a run (``record.game`` / ``record.ppo``)."""
    rec = json.loads(Path(path).read_text())["record"]
    g, ppo = rec["game"], rec["ppo"]
    spec = GameSpec(**{k: g[k] for k in ("w_h", "w_l", "k", "q", "T", "e_min", "e_max")})
    if spec.T != 2:
        raise DiagnosticsError(f"{path}: T = {spec.T}; the tent peak is defined for T = 2 only")
    return spec, float(ppo["c_min"]), float(ppo["mu_clamp"])


def read_update_table(path: Path) -> Dict[int, Tuple[int, Optional[int]]]:
    """``update -> (stage, local)`` from ``ms_updates.csv`` (``local`` None if column absent)."""
    t = pd.read_csv(path, usecols=lambda c: c in ("update", "stage", "local"))
    loc = t["local"].tolist() if "local" in t.columns else [None] * len(t)
    return {int(u): (int(s), None if v is None else int(v))
            for u, s, v in zip(t["update"], t["stage"], loc)}


def list_exports(weights_dir: Path) -> List[Tuple[int, Path]]:
    """``(update, path)`` of every ``u<update>.npz`` in a weights directory, ascending."""
    out: List[Tuple[int, Path]] = []
    for p in Path(weights_dir).glob("u*.npz"):
        m = _EXPORT_RE.match(p.name)
        if m is None:
            raise DiagnosticsError(f"{p}: not an export file name (expected u<update>.npz)")
        out.append((int(m.group(1)), p))
    return sorted(out)


def _load_actor_arrays(path: Path) -> Dict[str, np.ndarray]:
    """Actor arrays of an export (``actor*``, ``conc_scale``); the critic is not read."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files if k.startswith("actor") or k == "conc_scale"}


def process_run(job: RunJob) -> List[Dict[str, Any]]:
    """All export rows of one run (a worker task); a failure is re-raised naming the run."""
    try:
        return _process_run(job)
    except DiagnosticsError:
        raise
    except Exception as exc:                                   # noqa: BLE001 - name the run
        raise DiagnosticsError(f"{job.run_dir}: {type(exc).__name__}: {exc}") from exc


def _process_run(job: RunJob) -> List[Dict[str, Any]]:
    run = Path(job.run_dir)
    spec, c_min, mu_clamp = load_run_config(run / "run_config.json")
    table = read_update_table(run / "ms_updates.csv")
    e_star_0 = tent_peak(spec)
    rows: List[Dict[str, Any]] = []
    for update, path in list_exports(run / "weights"):
        if update not in table:
            raise DiagnosticsError(f"{path}: update {update} is not listed in ms_updates.csv")
        stage, local = table[update]
        stats = export_stats(_load_actor_arrays(path), spec, e_star_0, stage == TERMINAL_STAGE,
                             c_min, mu_clamp)
        rows.append({"source": job.source, "arm": job.arm, "q": job.q, "seed": job.seed,
                     "update": update, "stage": stage,
                     "local": float("nan") if local is None else local, **stats})
    return rows


def discover_runs(source: str, root: Path) -> List[Tuple[RunJob, Optional[str]]]:
    """Run directories ``<root>/q<q>/seed<seed>/<arm>`` and what each is missing (None = nothing).

    Raises:
        DiagnosticsError: ``root`` is not a directory or holds no run.
    """
    root = Path(root)
    if not root.is_dir():
        raise DiagnosticsError(f"{source}: root {root} does not exist or is not a directory")
    found: List[Tuple[RunJob, Optional[str]]] = []

    def subdirs(parent: Path, pattern: str) -> List[Path]:
        return sorted(p for p in parent.iterdir() if p.is_dir() and re.fullmatch(pattern, p.name))

    for qd in subdirs(root, r"q\d+"):
        for sd in subdirs(qd, r"seed\d+"):
            for ad in subdirs(sd, r".*"):
                job = RunJob(source, ad.name, int(qd.name[1:]), int(sd.name[4:]), str(ad))
                missing = [n for n in ("run_config.json", "ms_updates.csv")
                           if not (ad / n).is_file()]
                if not (ad / "weights").is_dir() or not any((ad / "weights").glob("u*.npz")):
                    missing.append("weights/u*.npz")
                found.append((job, ("missing " + ", ".join(missing)) if missing else None))
    if not found:
        raise DiagnosticsError(f"{source}: no run directory q<q>/seed<seed>/<arm> under {root}")
    return found


def _rank(source: str) -> int:
    return SOURCE_ORDER.index(source) if source in SOURCE_ORDER else len(SOURCE_ORDER)


def _natural(s: str) -> List[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def collect_jobs(sources: Mapping[str, Path]) -> List[RunJob]:
    """Every run of every declared root, ordered by source, arm, q, seed; incomplete = error."""
    jobs: List[RunJob] = []
    bad: List[str] = []
    for source, root in sources.items():
        for job, note in discover_runs(source, Path(root)):
            if note is not None:
                bad.append(f"{job.run_dir}: {note}")
            jobs.append(job)
    if bad:
        raise DiagnosticsError(
            f"{len(bad)} run(s) incomplete (a run without weights is an error):\n  "
            + "\n  ".join(bad[:10]) + ("\n  ..." if len(bad) > 10 else ""))
    return sorted(jobs, key=lambda j: (_rank(j.source), j.source, _natural(j.arm), j.q, j.seed))


def process_all(jobs: Sequence[RunJob], workers: int) -> List[List[Dict[str, Any]]]:
    """:func:`process_run` over all jobs (in-process for 1 worker, else a pool), in job order."""
    results: List[List[Dict[str, Any]]] = []
    step = max(1, len(jobs) // 10)
    pool_cm = mp.Pool(processes=min(workers, len(jobs))) if workers > 1 \
        else contextlib.nullcontext()
    with pool_cm as pool:
        stream = map(process_run, jobs) if pool is None \
            else pool.imap(process_run, jobs, chunksize=1)
        for i, rows in enumerate(stream, 1):
            results.append(rows)
            if i % step == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] runs read", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------- statistics
def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation; NaN for fewer than 3 points or a constant series."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.size < 3 or np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return float("nan")
    am, bm = a - a.mean(), b - b.mean()
    return float((am * bm).sum() / np.sqrt((am * am).sum() * (bm * bm).sum()))


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman correlation (Pearson of the average ranks); NaN as :func:`pearson`."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.size < 3:
        return float("nan")
    return pearson(rankdata(a), rankdata(b))


def _median(values: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def _mean(values: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    return float(v.mean()) if v.size else float("nan")


# ---------------------------------------------------------------------------- tables
def terminal_runs(df: pd.DataFrame) -> Dict[RunKey, Dict[str, np.ndarray]]:
    """Terminal-stage export series per run (source, arm, q, seed), ascending in update."""
    term = df[df["stage"] == TERMINAL_STAGE].sort_values(
        ["source", "arm", "q", "seed", "update"], kind="stable")
    out: Dict[RunKey, Dict[str, np.ndarray]] = {}
    for key, g in term.groupby(["source", "arm", "q", "seed"], sort=False):
        out[(str(key[0]), str(key[1]), int(key[2]), int(key[3]))] = {
            "update": g["update"].to_numpy(dtype=np.int64),
            "max_abs_w_d": g["max_abs_w_d"].to_numpy(dtype=float),
            "w_eff": g["w_eff"].to_numpy(dtype=float)}
    return out


def _cells(runs: Mapping[RunKey, Any]) -> List[Tuple[str, str, int]]:
    cells = {(k[0], k[1], k[2]) for k in runs}
    return sorted(cells, key=lambda c: (_rank(c[0]), c[0], _natural(c[1]), c[2]))


def build_per_arm(df: pd.DataFrame, targets: Sequence[int] = TARGET_UPDATES) -> pd.DataFrame:
    """Medians / means over the seeds of max |w| and w_eff at the targets and at the final export.

    For each run and target update the export used is the terminal-stage export with the
    greatest update not after the target (a run with none before the target does not enter that
    row); ``final`` is the last terminal-stage export of each run.

    Args:
        df: The per-export table.
        targets: Global update indices.

    Returns:
        One row per source / arm / q / ``which`` (``u<target>`` ..., ``final``).
    """
    runs = terminal_runs(df)
    rows: List[Dict[str, Any]] = []
    for source, arm, q in _cells(runs):
        series = [v for k, v in runs.items() if k[:3] == (source, arm, q)]
        for which, target in [(f"u{t}", t) for t in targets] + [("final", None)]:
            used: List[int] = []
            mw: List[float] = []
            we: List[float] = []
            for s in series:
                i = (len(s["update"]) - 1 if target is None
                     else int(np.searchsorted(s["update"], target, "right")) - 1)
                if i >= 0:
                    used.append(int(s["update"][i]))
                    mw.append(float(s["max_abs_w_d"][i]))
                    we.append(float(s["w_eff"][i]))
            if not used:
                continue
            rows.append({
                "source": source, "arm": arm, "q": q, "which": which,
                "target_update": pd.NA if target is None else int(target), "n_runs": len(used),
                "update_used_min": min(used), "update_used_median": float(np.median(used)),
                "update_used_max": max(used), "median_max_abs_w_d": _median(mw),
                "mean_max_abs_w_d": _mean(mw), "median_w_eff": _median(we),
                "mean_w_eff": _mean(we)})
    out = pd.DataFrame(rows, columns=list(PER_ARM_COLUMNS))
    out["target_update"] = out["target_update"].astype("Int64")   # integers, <NA> for 'final'
    return out


def build_relation(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation of max |w| with w_eff per source / arm / q over the terminal-stage exports.

    Pooled over all exports of all seeds; the median over the runs of the within-run correlation
    (a run whose series is constant or shorter than 3 points has none and is not counted in
    ``n_runs_within``); supplementary, across the seeds at the final export.
    """
    runs = terminal_runs(df)
    rows: List[Dict[str, Any]] = []
    for source, arm, q in _cells(runs):
        series = [v for k, v in runs.items() if k[:3] == (source, arm, q)]
        x = np.concatenate([s["max_abs_w_d"] for s in series])
        y = np.concatenate([s["w_eff"] for s in series])
        within_s = np.array([spearman(s["max_abs_w_d"], s["w_eff"]) for s in series])
        within_p = np.array([pearson(s["max_abs_w_d"], s["w_eff"]) for s in series])
        fx = np.array([s["max_abs_w_d"][-1] for s in series])
        fy = np.array([s["w_eff"][-1] for s in series])
        rows.append({
            "source": source, "arm": arm, "q": q, "n_runs": len(series), "n_pooled": int(x.size),
            "pooled_spearman": spearman(x, y), "pooled_pearson": pearson(x, y),
            "n_runs_within": int(np.isfinite(within_s).sum()),
            "within_run_median_spearman": _median(within_s),
            "within_run_median_pearson": _median(within_p), "n_final": int(fx.size),
            "final_cross_seed_spearman": spearman(fx, fy),
            "final_cross_seed_pearson": pearson(fx, fy)})
    return pd.DataFrame(rows, columns=list(RELATION_COLUMNS))


def build_regime(df: pd.DataFrame, reference: Mapping[int, float]) -> pd.DataFrame:
    """Share of RL exports with max |w| at or below the reference of their q (descriptive).

    Args:
        df: The per-export table.
        reference: ``q -> reference max |w|`` (the screen's t1 bin-balanced 56,000-step median).

    Returns:
        One row per source / arm / q with the shares over all exports, over terminal-stage
        exports and over the final (last terminal-stage) export of each run.

    Raises:
        DiagnosticsError: A q of the RL runs has no reference.
    """
    runs = terminal_runs(df)
    rows: List[Dict[str, Any]] = []
    for source, arm, q in _cells(runs):
        if q not in reference:
            raise DiagnosticsError(
                f"no screen reference for q = {q}; the screen has q = {sorted(reference)}")
        ref = float(reference[q])
        sel = df[(df["source"] == source) & (df["arm"] == arm) & (df["q"] == q)]
        term = [v for k, v in runs.items() if k[:3] == (source, arm, q)]
        t_all = np.concatenate([s["max_abs_w_d"] for s in term])
        fin = np.array([s["max_abs_w_d"][-1] for s in term])
        a_all = sel["max_abs_w_d"].to_numpy(dtype=float)
        rows.append({
            "source": source, "arm": arm, "q": q, "reference_max_abs_w_d": ref,
            "n_exports_all": int(a_all.size), "frac_all_at_or_below": float((a_all <= ref).mean()),
            "n_exports_terminal": int(t_all.size),
            "frac_terminal_at_or_below": float((t_all <= ref).mean()),
            "n_final": int(fin.size), "frac_final_at_or_below": float((fin <= ref).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- the screen
def _norm(x: Any) -> str:
    """Comparison form of a CSV cell or filter value: numbers as floats, labels lower-cased
    without punctuation (``Bin-Balanced`` -> ``binbalanced``, ``1`` and ``1.0`` -> ``1.0``)."""
    s = str(x).strip().lower()
    try:
        return repr(float(s))
    except ValueError:
        return re.sub(r"[^a-z0-9]", "", s)


def _truthy(x: Any) -> bool:
    return _norm(x) not in ("0.0", "false", "no", "nan", "")


def parse_screen_filters(items: Sequence[str]) -> Dict[str, str]:
    """``["col=value", ...]`` -> ``{col: value}`` (the first ``=`` splits)."""
    out: Dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise DiagnosticsError(f"--screen-filter expects COL=VALUE, got {it!r}")
        col, val = it.split("=", 1)
        out[col.strip()] = val.strip()
    return out


def read_screen_reference(screen_dir: Optional[Path], filters: Sequence[str] = ()
                          ) -> Tuple[Optional[Dict[int, float]], Dict[str, Any]]:
    """The small-weight-regime reference from the supervised screen, and a record of the step.

    Reads ``<screen_dir>/summary_by_cell.csv`` (column names matched case-insensitively): the
    rows with ``actor`` = ``t1``, ``steps`` = 56,000 and bin-balanced ``starts`` (``bb`` /
    ``bin_balanced`` / ``balanced`` in any spelling, or the label of a ``starts=`` filter) and
    every further ``COL=VALUE`` filter; if the file has an ``extended`` column its truthy rows
    (the screen's 224,000-step cell, whose 56,000-step checkpoint is at constant LR) are left
    out unless an ``extended=`` filter is given. The reference is the median of ``max_abs_w_d``
    over those rows per q.

    Args:
        screen_dir: Directory of the screen, or ``None``.
        filters: Extra equality filters ``COL=VALUE`` (compared after lower-casing and dropping
            punctuation).

    Returns:
        ``(reference_by_q, info)``. ``reference_by_q`` is ``None`` (``info["status"]`` =
        ``"skipped"`` with the reason) if the directory was not given or the file is absent.

    Raises:
        DiagnosticsError: The file exists but lacks a needed column, selects no row, or selects
            several rows for one (q, seed).
    """
    info: Dict[str, Any] = {
        "status": "skipped", "screen_dir": None if screen_dir is None else str(screen_dir),
        "filters": parse_screen_filters(filters)}
    if screen_dir is None:
        info["reason"] = "--screen-dir not given"
        return None, info
    path = Path(screen_dir) / SCREEN_CSV
    info["screen_csv"] = str(path)
    if not path.is_file():
        info["reason"] = f"{path} not found"
        return None, info
    raw = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in raw.columns}
    missing = [c for c in ("actor", "starts", "q", "steps", "max_abs_w_d") if c not in cols]
    if missing:
        raise DiagnosticsError(f"{path}: missing column(s) {missing}; found {list(raw.columns)}")
    user = {k.strip().lower(): v for k, v in info["filters"].items()}
    unknown = [k for k in user if k not in cols]
    if unknown:
        raise DiagnosticsError(
            f"{path}: --screen-filter column(s) {unknown} not in {list(raw.columns)}")
    keep = raw[cols["actor"]].map(_norm) == "t1"
    keep &= pd.to_numeric(raw[cols["steps"]], errors="coerce") == SCREEN_STEPS
    if "starts" not in user:
        keep &= raw[cols["starts"]].map(_norm).isin(SCREEN_BB_LABELS)
    if "extended" in cols and "extended" not in user:
        # the screen's extended cell also passes 56,000 steps, but at constant LR: not the
        # budget match of the RL terminal stage
        keep &= ~raw[cols["extended"]].map(_truthy)
        info["default_filters"] = ["extended is false (main-grid cells only)"]
    for k, v in user.items():
        keep &= raw[cols[k]].map(_norm) == _norm(v)
    sel = raw[keep]
    if sel.empty:
        seen = raw[[cols["actor"], cols["starts"], cols["steps"]]].drop_duplicates()
        raise DiagnosticsError(
            f"{path}: no row with actor t1, steps {SCREEN_STEPS}, bin-balanced starts and "
            f"filters {info['filters']}; (actor, starts, steps) present: "
            f"{seen.to_dict('records')[:12]}")
    qs = pd.to_numeric(sel[cols["q"]]).astype(int).to_numpy()
    if "seed" in cols and pd.DataFrame(
            {"q": qs, "seed": sel[cols["seed"]].to_numpy()}).duplicated().any():
        raise DiagnosticsError(
            f"{path}: several selected rows for one (q, seed); restrict them with "
            f"--screen-filter COL=VALUE (columns: {list(raw.columns)})")
    vals = pd.to_numeric(sel[cols["max_abs_w_d"]]).to_numpy(dtype=float)
    ref = {int(q): float(np.median(vals[qs == q])) for q in sorted(set(qs.tolist()))}
    info.update(
        status="computed", reference_max_abs_w_d_by_q={str(q): v for q, v in ref.items()},
        n_rows_by_q={str(q): int((qs == q).sum()) for q in ref},
        definition=("share of RL exports with max |w| <= the median over the screen rows (actor "
                    f"t1, bin-balanced starts, {SCREEN_STEPS} steps) of max_abs_w_d, per q"))
    return ref, info


# ---------------------------------------------------------------------------- outputs
def check_outputs_free(out: Path) -> None:
    """Refuse to overwrite: no output file may exist in ``out`` (and ``out`` may not be a file)."""
    if os.path.lexists(out) and not Path(out).is_dir():
        raise DiagnosticsError(f"--out {out} exists and is not a directory")
    existing = [n for n in OUTPUT_FILES if os.path.lexists(Path(out) / n)]
    if existing:
        raise DiagnosticsError(
            f"refusing to overwrite existing output file(s) in {out}: {', '.join(existing)}")


def _write(path: Path, text: str) -> None:
    try:
        with open(path, "x") as fh:
            fh.write(text)
    except FileExistsError as exc:
        raise DiagnosticsError(f"refusing to overwrite {path}") from exc


def _csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False, na_rep="nan")


def git_head() -> Optional[str]:
    """``git rev-parse HEAD`` of the repository containing this tool, ``None`` if git fails."""
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def build_summary(manifest: Mapping[str, Any], per_arm: pd.DataFrame, relation: pd.DataFrame,
                  regime: Optional[pd.DataFrame]) -> str:
    """The human-readable ``summary.txt``."""
    lines: List[str] = [
        "r3_actor_diagnostics: RL-actor diagnostics on existing weight exports "
        "(MS-R3 prompt 2.3 (b))",
        f"git HEAD: {manifest['git_head']}", f"command: {manifest['command_line']}",
        f"workers: {manifest['workers']}", "roots (runs / exports / terminal-stage exports):"]
    for s, c in manifest["counts"]["by_source"].items():
        lines.append(f"  {s}: {manifest['roots'][s]}  "
                     f"({c['runs']} / {c['exports']} / {c['terminal_stage_exports']})")
    t = manifest["counts"]["total"]
    lines += [
        f"  total: {t['runs']} runs, {t['exports']} exports "
        f"({t['terminal_stage_exports']} terminal-stage)", "",
        "Definitions: w_d = first-layer weights on the d input in units of d / B (x10 stored for",
        "  variant t10); w_eff = gap / (e2*(0) / (2 q)) in units of d, gap = e2*(0) - e_hat_2(0)",
        "  (export, variant-aware numpy reload at the stage-2 observation of d = 0; e2*(0) =",
        "  utils.theory_multistage.g2_two_stage). w_eff is NaN for stage-1 exports (the live actor",
        "  is then trained on the root game). 'final' = the last terminal-stage export of a run",
        "  (the terminal freeze); u<target> = the terminal-stage export nearest to and not after",
        "  that global update (update_used_* in per_arm.csv show which was used).", ""]
    fin = per_arm[(per_arm["source"] == "ms_r2_pilot") & (per_arm["which"] == "final")]
    if fin.empty:
        lines.append("MS-R2 w_eff at the freeze: no MS-R2 root given, preamble comparison "
                     "not computed.")
    else:
        lines += ["MS-R2 w_eff at the freeze (final terminal-stage export), over the seeds, "
                  "units of d:",
                  "  q  arm            n   mean_w_eff  median_w_eff  median_max_abs_w_d"]
        for r in sorted(fin.itertuples(), key=lambda r: (r.q, _natural(r.arm))):
            lines.append(f"  {r.q}  {r.arm:<12} {r.n_runs:>3}  {r.mean_w_eff:>10.3f}  "
                         f"{r.median_w_eff:>12.3f}  {r.median_max_abs_w_d:>18.3f}")
        lines += ["", "  six-arm mean of the arm means of w_eff at the freeze, next to the PI's "
                  "preamble figure", "  (gap / tent slope averaged over the six MS-R2 arms, "
                  "quoted from the round prompt):",
                  "  q   arms   mean w_eff (this tool)   PI preamble"]
        for q, g in fin.groupby("q"):
            pre = PREAMBLE_W_EFF.get(int(q))
            lines.append(f"  {int(q)}   {len(g):>4}   {g['mean_w_eff'].mean():>22.3f}   "
                         f"{'n/a' if pre is None else format(pre, '.2f'):>11}")
    rel = relation.set_index(["source", "arm", "q"])
    lines += ["", "Final terminal-stage export (median over seeds) and the Spearman correlation",
              "of max |w| with w_eff over the terminal-stage exports (pooled over seeds / median",
              "within a run); relation.csv also has Pearson:",
              "  source        arm           q   n  max_abs_w_d    w_eff  pooled_rho  within_rho"]
    for r in per_arm[per_arm["which"] == "final"].itertuples():
        c = rel.loc[(r.source, r.arm, r.q)]
        lines.append(f"  {r.source:<13} {r.arm:<13} {r.q}  {r.n_runs:>2}  "
                     f"{r.median_max_abs_w_d:>10.3f}  {r.median_w_eff:>7.3f}  "
                     f"{c['pooled_spearman']:>10.3f}  {c['within_run_median_spearman']:>10.3f}")
    lines.append("")
    rg = manifest["regime"]
    if regime is None:
        lines.append(f"Small-weight regime table (regime.csv): skipped; {rg.get('reason')}.")
    else:
        refs = ", ".join(f"q{q} {v:.3f}" for q, v in rg["reference_max_abs_w_d_by_q"].items())
        lines += [f"Share of exports with max |w| <= the screen reference (t1, bin-balanced "
                  f"starts, {SCREEN_STEPS} steps, median over seeds): {refs}",
                  "  source        arm           q  frac_all  frac_terminal  frac_final"]
        for r in regime.itertuples():
            lines.append(f"  {r.source:<13} {r.arm:<13} {r.q}  {r.frac_all_at_or_below:>8.3f}  "
                         f"{r.frac_terminal_at_or_below:>13.3f}  "
                         f"{r.frac_final_at_or_below:>10.3f}")
    return "\n".join(lines) + "\n"


def run_diagnostics(sources: Mapping[str, Path], out: Path, workers: int = 4,
                    screen_dir: Optional[Path] = None, screen_filters: Sequence[str] = (),
                    command: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Read every export of every declared root; write the tables, manifest and summary.

    Args:
        sources: ``source label -> root`` (labels in ``SOURCE_ORDER`` sort first).
        out: Output directory (created if absent; no output file may exist in it).
        workers: Worker processes (1 = in-process).
        screen_dir: Optional directory of the supervised screen (``summary_by_cell.csv``).
        screen_filters: Extra ``COL=VALUE`` filters for the screen rows.
        command: Command line to record.

    Returns:
        The manifest dict.

    Raises:
        DiagnosticsError: Any missing or malformed input, or an existing output file.
    """
    out = Path(out)
    if not sources:
        raise DiagnosticsError(
            "no root given: pass at least one of " + ", ".join(ROOT_FLAGS.values()))
    if workers < 1:
        raise DiagnosticsError("--workers must be at least 1")
    check_outputs_free(out)
    jobs = collect_jobs(sources)
    reference, screen_info = read_screen_reference(screen_dir, screen_filters)
    if reference is not None:
        lacking = sorted({j.q for j in jobs} - set(reference))
        if lacking:
            raise DiagnosticsError(f"the screen reference has no row for q = {lacking}")
    per_run = process_all(jobs, workers)
    df = pd.DataFrame([r for rows in per_run for r in rows], columns=list(EXPORT_COLUMNS))
    per_arm, relation = build_per_arm(df), build_relation(df)
    regime = build_regime(df, reference) if reference is not None else None

    by_source: Dict[str, Any] = {}
    for s in sources:
        d = df[df["source"] == s]
        arms = Counter(j.arm for j in jobs if j.source == s)
        by_source[s] = {
            "runs": sum(arms.values()), "exports": int(len(d)),
            "terminal_stage_exports": int((d["stage"] == TERMINAL_STAGE).sum()),
            "arms": {a: arms[a] for a in sorted(arms, key=_natural)},
            "variants": sorted(d["variant"].unique().tolist())}
    manifest: Dict[str, Any] = {
        "tool": str(Path(__file__).resolve()), "git_head": git_head(),
        "command_line": shlex.join(command) if command is not None else None,
        "workers": workers, "roots": {s: os.path.abspath(r) for s, r in sources.items()},
        "counts": {"by_source": by_source,
                   "total": {"runs": len(jobs), "exports": int(len(df)),
                             "terminal_stage_exports": int((df["stage"] == TERMINAL_STAGE).sum())}},
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "target_updates": list(TARGET_UPDATES),
        "regime": screen_info, "preamble_w_eff": {str(q): v for q, v in PREAMBLE_W_EFF.items()}}
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "per_export.csv", _csv(df))
    _write(out / "per_arm.csv", _csv(per_arm))
    _write(out / "relation.csv", _csv(relation))
    if regime is not None:
        _write(out / "regime.csv", _csv(regime))
    _write(out / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    _write(out / "summary.txt", build_summary(manifest, per_arm, relation, regime))
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI (see the module docstring). Returns 0, or 2 after printing a DiagnosticsError."""
    args_in = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for source, flag in ROOT_FLAGS.items():
        p.add_argument(flag, default=None, help=f"root of the {source} runs "
                       "(<root>/q<q>/seed<seed>/<arm>/); default: not included")
    p.add_argument("--screen-dir", default=None, help="supervised-screen directory with "
                   "summary_by_cell.csv (optional; without it regime.csv is skipped)")
    p.add_argument("--screen-filter", action="append", default=[], metavar="COL=VALUE",
                   help="extra equality filter on the screen rows (repeatable)")
    p.add_argument("--out", required=True,
                   help="output directory; existing output files are never overwritten")
    p.add_argument("--workers", type=int, default=4, help="worker processes (default 4)")
    a = p.parse_args(args_in)
    dest = {s: flag[2:].replace("-", "_") for s, flag in ROOT_FLAGS.items()}
    sources = {s: Path(getattr(a, d)) for s, d in dest.items() if getattr(a, d) is not None}
    try:
        run_diagnostics(sources, Path(a.out), a.workers,
                        None if a.screen_dir is None else Path(a.screen_dir), a.screen_filter,
                        [sys.executable, str(Path(__file__).resolve())] + args_in)
    except DiagnosticsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
