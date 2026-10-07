#!/usr/bin/env python
"""Offline calibration of the MS-R1 development rule on stored v2.0 weight exports (spec 2.4).

No training. For every run of the two reference roots (``rehearsal_v2_0``: 20 runs,
``confirmation_v2_0``: 40 runs; both live under ROOT_V2 and are READ ONLY) and every weight export
``weights/u{u:05d}.npz`` (u = 25, 50, ..., 2200) the development verifier is replayed on the candidate
the export defines and the D3 quantities of ``utils.ms_residual`` are computed:

* Phase A export u (u <= 1600): the export's network at every stage (v2.0's Phase A has no frozen
  stage); the D3 stage is the terminal stage t = 2.
* Phase B export u (u >= 1625): the export's network at t = 1 and the u1600 export FROZEN at t = 2
  (as ``run_v2_stagewise.Run.policy_fns`` in Phase B); the D3 stage is t = 1.

The float32 forward pass is the training one: the arrays are loaded into an
``agents.ppo_curriculum.BetaActor`` and the verifier-facing functions come from
``run.run_final_dp_br.make_policy_fns``. The replay is validated against the runs' own
``v2_checkpoints_A.csv`` / ``v2_checkpoints_B.csv`` (bit for bit; the u1600 comparison is the gate).

The closed-form equilibrium enters ONLY the reporting columns (``recovery_metrics``); the rule logic
(``fire_export``) reads nothing but a ``utils.ms_residual.StageDiag`` and calls the pipeline's own
``utils.ms_rule.is_eligible``.

Commands (each takes every root explicitly and records it)::

    python tools/ms/replay_dev_rule.py replay  --root-v2 ROOT_V2 --out results/ms_r1/calibration
    python tools/ms/replay_dev_rule.py analyze --root-v2 ROOT_V2 --out results/ms_r1/calibration \
        --fig-dir reports/ms/r1/figures

``replay`` writes one CSV per run (``<out>/<root>/q<q>/seed<seed>.csv``), ``manifest.json``,
``validation_summary.csv``. ``analyze`` writes ``<out>/tables/*.csv``, the figures and
``<out>/facts.json`` (the three facts of the preamble of the round prompt); with ``--md-dir`` it also
renders every table as markdown into that directory. Both refuse to overwrite an
existing output file unless ``--overwrite-own-output`` is given (it only ever touches files this tool
writes). The numeric columns are deterministic (single-threaded float32 / float64 arithmetic); only
``verifier_sec`` / ``diag_sec`` are wall-clock measurements. A missing reference file stops the replay
before anything is computed and is listed.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.ppo_curriculum import ACTOR_VARIANTS, BetaActor  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402
from run.run_final_dp_br import dense_grid, make_policy_fns  # noqa: E402
from utils.dp_br_verifier import (  # noqa: E402
    DEV_CONFIG, FINAL_CONFIG, DomainError, VerifierConfig, concentration_stats)
from utils.ms_residual import StageDiag, ema_update, invalid_diag, stage_diag  # noqa: E402
from utils.ms_rule import StageRule, classify_block_end, is_eligible  # noqa: E402
from utils.v2_metrics import V2Eval, evaluate  # noqa: E402

torch.set_num_threads(1)

MeanFn = Callable[[int, np.ndarray], np.ndarray]
BetaFn = Callable[[int, np.ndarray], Tuple[np.ndarray, np.ndarray]]

# --------------------------------------------------------------------------------------- constants
EXPORT_US: Tuple[int, ...] = tuple(range(25, 2201, 25))      # 88 exports: A u25..u1600, B u1625..u2200
PHASE_A_END = 1600
DEFAULT_LABELS: Tuple[str, ...] = ("rehearsal_v2_0", "confirmation_v2_0")
EXPECTED_SEEDS: Dict[str, Tuple[int, ...]] = {
    "rehearsal_v2_0": tuple(range(10501, 10511)), "confirmation_v2_0": tuple(range(30501, 30521))}
QS: Tuple[int, ...] = (50, 60)
RHO_GRID: Tuple[float, ...] = (0.02, 0.03, 0.05)
EPS, TAU, M_DEFAULT, K_DEFAULT, CONC_LIMIT = 0.005, 0.02, 3, 25, 0.04
LOC_FRAC, EMA_BETA = 0.25, 0.5
NEAR_TIE_HALF_WIDTH, BIN_WIDTH = 20.0, 10.0
GA_TAIL, GA_RMSE = 0.02, 0.05                       # G-A limits on the tail mean and RMSE_pos
RECOVERY_STEP = 0.5
SPEARMAN_FROM_U = 400
PROTOCOL = REPO / "protocols" / "v2_T2_locked_v2_0.json"
REQUIRED_FILES = ("v2_checkpoints_A.csv", "v2_checkpoints_B.csv", "state_end_A.pt", "gates.json",
                  "train_history.json")

ID_COLS = ["root", "q", "seed", "update", "phase", "stage", "tier", "valid", "res_valid", "Delta", "s",
           "R", "R_tail", "C", "R_argmax_d", "tail_term"]
CLOSED_FORM_COLS = ["stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs",
                    "stage2_rmse_pos_over_g2_0", "stage2_tail_mean_over_g2_0",
                    "stage2_tail_max_over_g2_0", "stage1_rel_err_signed", "e1_at_0", "e2_at_0",
                    "eta_T_over_dw", "Gmax_full_over_dw"]
TIME_COLS = ["verifier_sec", "diag_sec", "error"]
STR_COLS = {"root", "phase", "tier", "error"}
INT_COLS = {"q", "seed", "update", "stage"}
BOOL_COLS = {"valid", "res_valid", "tail_term"}
# fields compared with the runs' own v2_checkpoints_{A,B}.csv (name in the log == name in the replay)
VALIDATION_FIELDS = ("eta_T_over_dw", "stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs",
                     "stage2_rmse_pos_over_g2_0", "stage2_tail_mean_over_g2_0",
                     "stage2_tail_max_over_g2_0", "e1_at_0", "e2_at_0", "stage1_rel_err_signed",
                     "Gmax_full_over_dw", "EXP_root_over_dw", "conc_max_std_norm")
GATE_FIELDS = ("eta_T_over_dw", "stage2_peak_rel_err_signed", "stage2_rmse_pos_over_g2_0")
GATE_TOL = 1e-12


# ------------------------------------------------------------------------------------ run discovery
@dataclass(frozen=True)
class RunRef:
    """One reference run: ``<root_dir>/q<q>/seed<seed>``."""

    root_label: str
    root_dir: str
    q: int
    seed: int

    @property
    def run_dir(self) -> Path:
        """Directory of the run."""
        return Path(self.root_dir) / f"q{self.q}" / f"seed{self.seed}"

    @property
    def key(self) -> str:
        """``<root>/q<q>/seed<seed>``."""
        return f"{self.root_label}/q{self.q}/seed{self.seed}"


def discover_runs(root_v2: str, labels: Sequence[str] = DEFAULT_LABELS) -> List[RunRef]:
    """All ``<root_v2>/<label>/q*/seed*`` run directories, sorted by (label order, q, seed)."""
    runs: List[RunRef] = []
    for label in labels:
        base = Path(root_v2) / label
        for qd in sorted(base.glob("q*"), key=lambda p: int(p.name[1:])):
            for sd in sorted(qd.glob("seed*"), key=lambda p: int(p.name[4:])):
                runs.append(RunRef(label, str(base), int(qd.name[1:]), int(sd.name[4:])))
    return runs


def layout_problems(runs: Sequence[RunRef], labels: Sequence[str] = DEFAULT_LABELS) -> List[str]:
    """Differences between the discovered runs and the expected layout (10 / 20 seeds per q)."""
    out: List[str] = []
    for label in labels:
        for q in QS:
            got = sorted(r.seed for r in runs if r.root_label == label and r.q == q)
            exp = list(EXPECTED_SEEDS.get(label, ()))
            if exp and got != exp:
                out.append(f"{label}/q{q}: seeds {got} != expected {exp}")
    return out


def missing_files(ref: RunRef) -> List[str]:
    """Reference files of one run that do not exist (88 weight exports and ``REQUIRED_FILES``)."""
    d = ref.run_dir
    need = [d / f for f in REQUIRED_FILES] + [d / "weights" / f"u{u:05d}.npz" for u in EXPORT_US]
    return [str(p) for p in need if not p.is_file()]


# ------------------------------------------------------------------------------------ policy / replay
class _Shim:
    """Exposes ``beta_params(obs, net)`` as ``make_policy_fns`` needs (the training float32 pass)."""

    @staticmethod
    def beta_params(obs: np.ndarray, net: Optional[BetaActor] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """(alpha, beta) float32 arrays of ``net`` on a float32 observation batch (no grad)."""
        if net is None:
            raise ValueError("a network is required")
        with torch.no_grad():
            a, b = net(torch.as_tensor(np.asarray(obs, dtype=np.float32)))
        return a.cpu().numpy(), b.cpu().numpy()


def load_export(path: str) -> Dict[str, np.ndarray]:
    """The arrays of one ``weights/u*.npz`` export."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def build_actor(arrays: Mapping[str, np.ndarray], hidden: int = 64, c_min: float = 100.0,
                mu_clamp: float = 1e-6) -> BetaActor:
    """``BetaActor`` carrying exactly the exported float32 weights (and ``conc_scale`` / ``actor_variant`` if exported).

    The forward pass of the returned module is the one of training (torch float32), hence bit-identical.
    """
    gen = torch.Generator()
    gen.manual_seed(0)
    net = BetaActor(hidden, c_min, mu_clamp, gen)
    sd = {k[len("actor."):]: torch.from_numpy(np.array(v, dtype=np.float32))
          for k, v in arrays.items() if k.startswith("actor.")}
    net.load_state_dict(sd)
    if "conc_scale" in arrays:
        net.conc_scale = float(arrays["conc_scale"])
    if "actor_variant" in arrays:                       # MS-R3: never rebuild a relu / t10 export as the tanh d / B actor
        variant = str(np.asarray(arrays["actor_variant"]))
        if variant not in ACTOR_VARIANTS:
            raise ValueError(f"unknown actor variant {variant!r} in the export; known: {ACTOR_VARIANTS}")
        net.variant = variant
    net.eval()
    return net


def candidate_fns(spec: GameSpec, nets: Mapping[int, BetaActor]
                  ) -> Tuple[MeanFn, BetaFn]:
    """``(mean_fn, beta_fn)`` of the composite candidate ``{stage: network}`` (one network per stage)."""
    shim = _Shim()
    fns = {t: make_policy_fns(shim, spec, net=n) for t, n in nets.items()}   # type: ignore[arg-type]

    def mean_fn(t: int, d: np.ndarray) -> np.ndarray:
        """Mean effort of the stage-``t`` network."""
        return fns[t][0](t, d)

    def beta_fn(t: int, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Beta parameters of the stage-``t`` network."""
        return fns[t][1](t, d)
    return mean_fn, beta_fn


@dataclass
class Replayed:
    """One replayed candidate."""

    ev: Optional[V2Eval]
    diag: StageDiag
    conc: Optional[Dict[str, object]]
    sec: float
    diag_sec: float
    error: str = ""


def stage_points(spec: GameSpec, stage: int, cfg: VerifierConfig) -> np.ndarray:
    """The development grid of the D3 stage (``{0}`` at stage 1)."""
    if stage <= 1:
        return np.zeros(1)
    return dense_grid(spec.domain_half(stage), cfg.state_step)


def replay_candidate(spec: GameSpec, nets: Mapping[int, BetaActor], stage: int, cfg: VerifierConfig,
                     n_bins: int = 0) -> Replayed:
    """One ``utils.v2_metrics.evaluate`` call (timed like v2.0's ``verify_candidate``) and the D3 diag."""
    mean_fn, beta_fn = candidate_fns(spec, nets)
    t0 = time.perf_counter()
    try:
        ev = evaluate(mean_fn, spec, cfg, beta_fn=beta_fn, recovery_step=RECOVERY_STEP)
    except (DomainError, FloatingPointError, ValueError) as exc:
        return Replayed(None, invalid_diag(stage, n_bins), None, time.perf_counter() - t0, 0.0,
                        f"{type(exc).__name__}: {exc}")
    sec = time.perf_counter() - t0
    t1 = time.perf_counter()
    conc = concentration_stats(beta_fn, {stage: stage_points(spec, stage, cfg)}, spec.e_range)
    diag = stage_diag(ev.res, spec, stage, conc, BIN_WIDTH)
    return Replayed(ev, diag, conc, sec, time.perf_counter() - t1)


def make_row(ref: RunRef, update: int, phase: str, stage: int, tier: str, rp: Replayed, n_bins: int
             ) -> Dict[str, object]:
    """One CSV row (``ID_COLS`` + per-bin map + closed-form reporting columns + timing)."""
    d = rp.diag
    row: Dict[str, object] = {
        "root": ref.root_dir, "q": ref.q, "seed": ref.seed, "update": update, "phase": phase,
        "stage": stage, "tier": tier, "valid": bool(d.valid),
        "res_valid": bool(rp.ev.scalars["valid"]) if rp.ev is not None else False,
        "Delta": d.delta_over_dw, "s": d.s, "R": d.R, "R_tail": d.R_tail, "C": d.C,
        "R_argmax_d": d.argmax_d, "tail_term": bool(d.tail_term)}
    rho = np.asarray(d.rho_bins, dtype=float)
    for i in range(n_bins):
        row[f"rho_bin_{i}"] = float(rho[i]) if (stage >= 2 and i < rho.size) else float("nan")
    for c in CLOSED_FORM_COLS:
        row[c] = float(rp.ev.scalars[c]) if (rp.ev is not None and c in rp.ev.scalars) else float("nan")
    row.update(verifier_sec=rp.sec, diag_sec=rp.diag_sec, error=rp.error)
    return row


def csv_fields(n_bins: int) -> List[str]:
    """Column order of the per-run CSVs."""
    return ID_COLS + [f"rho_bin_{i}" for i in range(n_bins)] + CLOSED_FORM_COLS + TIME_COLS


@dataclass
class ReplayOut:
    """Rows of one run plus the extra scalars needed to validate against the logs."""

    rows: List[Dict[str, object]]
    extras: Dict[Tuple[str, int], Dict[str, float]]


def _v2_conc(spec: GameSpec, nets: Mapping[int, BetaActor], phase: str, rp: Replayed) -> float:
    """``conc_max_std_norm`` of the v2.0 log: stage-2 dev grid (A) or {root} U stage-2 dev grid (B)."""
    if phase == "A" and rp.conc is not None and rp.conc.get("valid"):
        return float(rp.conc["max_std_norm"])
    _, beta_fn = candidate_fns(spec, nets)
    pts = {1: np.zeros(1), 2: stage_points(spec, 2, DEV_CONFIG)}
    c = concentration_stats(beta_fn, pts if phase == "B" else {2: pts[2]}, spec.e_range)
    return float(c["max_std_norm"]) if c.get("valid") else float("nan")


def replay_exports(ref: RunRef, spec: GameSpec, ppo: Mapping[str, Any]) -> ReplayOut:
    """Replay every export of one run (dev tier) and the final tier at u1600 (Phase A, stage 2)."""
    n_bins = StartSampler(spec, BIN_WIDTH).n_bins(2)
    hidden, c_min, mu_clamp = int(ppo["hidden"]), float(ppo["c_min"]), float(ppo["mu_clamp"])
    rows: List[Dict[str, object]] = []
    extras: Dict[Tuple[str, int], Dict[str, float]] = {}
    net_1600: Optional[BetaActor] = None
    for u in EXPORT_US:
        net = build_actor(load_export(str(ref.run_dir / "weights" / f"u{u:05d}.npz")), hidden, c_min,
                          mu_clamp)
        if u == PHASE_A_END:
            net_1600 = net
        if u <= PHASE_A_END:
            phase, stage, nets = "A", 2, {1: net, 2: net}
        else:
            assert net_1600 is not None
            phase, stage, nets = "B", 1, {1: net, 2: net_1600}
        rp = replay_candidate(spec, nets, stage, DEV_CONFIG, n_bins)
        rows.append(make_row(ref, u, phase, stage, "dev", rp, n_bins))
        if rp.ev is not None:
            ex = {f: float(rp.ev.scalars[f]) for f in VALIDATION_FIELDS
                  if f in rp.ev.scalars and f != "conc_max_std_norm"}
            ex["conc_max_std_norm"] = _v2_conc(spec, nets, phase, rp)
            extras[(phase, u)] = ex
        if u == PHASE_A_END:
            rpf = replay_candidate(spec, nets, 2, FINAL_CONFIG, n_bins)
            rows.append(make_row(ref, u, "A", 2, "final", rpf, n_bins))
    return ReplayOut(rows, extras)


# -------------------------------------------------------------------------------------------- CSV io
def _fmt(v: object) -> object:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, (int, np.integer)):
        return int(v)
    return v


def write_csv(path: str, rows: Sequence[Mapping[str, object]], fields: Sequence[str],
              overwrite: bool = False) -> None:
    """Write ``rows`` (floats as ``repr``); refuses to replace an existing file unless ``overwrite``."""
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite-own-output to replace this tool's output")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r[k]) for k in fields})
    os.replace(tmp, path)


def _parse(col: str, v: str) -> object:
    if col in STR_COLS:
        return v
    if col in INT_COLS:
        return int(v)
    if col in BOOL_COLS:
        return v == "True"
    return float(v) if v != "" else float("nan")


def read_csv_rows(path: str) -> List[Dict[str, object]]:
    """Read a per-run CSV written by :func:`write_csv` (types restored; ``float('nan')`` for blanks)."""
    with open(path, newline="") as f:
        return [{k: _parse(k, v) for k, v in r.items()} for r in csv.DictReader(f)]


def read_log_rows(path: str) -> List[Dict[str, float]]:
    """A v2 ``v2_checkpoints_{A,B}.csv`` as float dicts (non-numeric cells stay strings)."""
    out: List[Dict[str, float]] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d: Dict[str, Any] = {}
            for k, v in r.items():
                try:
                    d[k] = float(v)
                except (TypeError, ValueError):
                    d[k] = v
            out.append(d)
    return out


def _same(a: float, b: float) -> bool:
    return bool(a == b or (math.isnan(a) and math.isnan(b)))


def validate_run(ref: RunRef, out: ReplayOut) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Compare the replay with the run's own ``v2_checkpoints_A/B.csv`` (all logged calls).

    Returns:
        ``(summary, mismatches)``; ``summary['gate_ok']`` is True iff the three gate fields of the u1600
        call agree to ``GATE_TOL`` (v2.0's Phase-A end call is a dev-tier call on the same candidate).
    """
    n = n_eq = 0
    max_abs = 0.0
    mism: List[Dict[str, object]] = []
    gate = {f: float("nan") for f in GATE_FIELDS}
    missing_calls: List[str] = []
    for phase in ("A", "B"):
        for lg in read_log_rows(str(ref.run_dir / f"v2_checkpoints_{phase}.csv")):
            u = int(lg["update"])
            ex = out.extras.get((phase, u))
            if ex is None:
                missing_calls.append(f"{phase}{u}")
                continue
            for f in VALIDATION_FIELDS:
                if f not in lg or f not in ex or not isinstance(lg[f], float):
                    continue
                a, b = float(ex[f]), float(lg[f])
                n += 1
                eq = _same(a, b)
                n_eq += int(eq)
                diff = 0.0 if eq else abs(a - b)
                max_abs = max(max_abs, diff) if not math.isnan(diff) else float("nan")
                if phase == "A" and u == PHASE_A_END and f in GATE_FIELDS:
                    gate[f] = diff
                if not eq:
                    mism.append({"key": ref.key, "phase": phase, "update": u, "field": f,
                                 "replay": a, "logged": b, "abs_diff": diff})
    gate_ok = all(math.isfinite(v) and v <= GATE_TOL for v in gate.values())
    summ = {"key": ref.key, "root": ref.root_label, "q": ref.q, "seed": ref.seed, "n_compared": n,
            "n_bitwise_equal": n_eq, "max_abs_diff": max_abs,
            "gate_eta_absdiff": gate["eta_T_over_dw"], "gate_peak_absdiff": gate["stage2_peak_rel_err_signed"],
            "gate_rmse_absdiff": gate["stage2_rmse_pos_over_g2_0"], "gate_ok": bool(gate_ok),
            "calls_without_replay": ";".join(missing_calls)}
    return summ, mism


# ------------------------------------------------------------------------------------ replay driver
def load_protocol(path: Path = PROTOCOL) -> Dict[str, Any]:
    """The locked v2.0 protocol record (``records[str(q)]`` = game + ppo)."""
    with open(path) as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_csv_path(out_dir: str, ref: RunRef) -> str:
    """``<out>/<root>/q<q>/seed<seed>.csv``."""
    return os.path.join(out_dir, ref.root_label, f"q{ref.q}", f"seed{ref.seed}.csv")


def _worker(job: Dict[str, Any]) -> Dict[str, Any]:
    ref = RunRef(**job["ref"])
    proto = load_protocol(Path(job["protocol"]))
    rec = proto["records"][str(ref.q)]
    spec = GameSpec(**rec["game"])
    t0 = time.perf_counter()
    out = replay_exports(ref, spec, rec["ppo"])
    n_bins = StartSampler(spec, BIN_WIDTH).n_bins(2)
    write_csv(job["csv"], out.rows, csv_fields(n_bins), overwrite=job["overwrite"])
    summ, mism = validate_run(ref, out)
    summ["replay_wall_sec"] = time.perf_counter() - t0
    return {"summary": summ, "mismatches": mism}


def git_head() -> str:
    """HEAD commit of the repo (``unknown`` if git fails)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                                       text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def cmd_replay(args: argparse.Namespace) -> int:
    """``replay``: preflight, replay every run, validate, write CSVs + manifest + validation summary."""
    labels = tuple(args.labels)
    runs = discover_runs(args.root_v2, labels)
    if args.only:
        runs = [r for r in runs if r.key in set(args.only)]
    problems = layout_problems(runs, labels) if not args.only else []
    missing = {r.key: m for r in runs for m in [missing_files(r)] if m}
    if problems or missing or not runs:
        print("STOP: reference layout / files do not match the brief (nothing was computed).")
        for p in problems:
            print("  layout:", p)
        for k, m in missing.items():
            print(f"  {k}: {len(m)} missing, first: {m[0]}")
        if not runs:
            print("  no runs discovered under", args.root_v2)
        return 2
    out_dir = os.path.abspath(args.out)
    if os.path.commonpath([out_dir, os.path.abspath(args.root_v2)]) == os.path.abspath(args.root_v2):
        print("STOP: --out lies inside the (read-only) reference root")
        return 2
    existing = [run_csv_path(out_dir, r) for r in runs if os.path.exists(run_csv_path(out_dir, r))]
    man_path = os.path.join(out_dir, "manifest.json")
    if (existing or os.path.exists(man_path)) and not args.overwrite_own_output:
        print(f"STOP: {len(existing)} per-run CSV(s) / manifest already exist under {out_dir}; "
              "pass --overwrite-own-output to replace this tool's own output")
        return 2
    workers = max(1, min(int(args.workers), 8))
    jobs = [{"ref": dataclasses.asdict(r), "protocol": str(args.protocol), "csv": run_csv_path(out_dir, r),
             "overwrite": bool(args.overwrite_own_output)} for r in runs]
    t0 = time.perf_counter()
    load_start = os.getloadavg()
    if workers == 1:
        results = [_worker(j) for j in jobs]
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(_worker, jobs))
    wall = time.perf_counter() - t0
    summ = [r["summary"] for r in results]
    mism = [m for r in results for m in r["mismatches"]]
    fields = list(summ[0].keys())
    write_csv(os.path.join(out_dir, "validation_summary.csv"), summ, fields, overwrite=True)
    if mism:
        write_csv(os.path.join(out_dir, "validation_mismatches.csv"), mism, list(mism[0].keys()),
                  overwrite=True)
    manifest = {
        "tool": "tools/ms/replay_dev_rule.py", "command": " ".join(sys.argv), "git_head": git_head(),
        "reference_roots": {lab: str(Path(args.root_v2) / lab) for lab in labels},
        "root_v2": str(args.root_v2), "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(Path(args.protocol)), "n_runs": len(runs),
        "n_exports_per_run": len(EXPORT_US), "export_us": [EXPORT_US[0], EXPORT_US[-1], 25],
        "verifier_tiers": {"dev": dataclasses.asdict(DEV_CONFIG), "final": dataclasses.asdict(FINAL_CONFIG)},
        "recovery_step": RECOVERY_STEP, "workers": workers, "loadavg_at_start": list(load_start),
        "wall_sec": wall, "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "validation": {"n_runs": len(summ), "n_gate_ok": int(sum(s["gate_ok"] for s in summ)),
                       "n_mismatching_fields": len(mism),
                       "max_abs_diff": max(s["max_abs_diff"] for s in summ)},
        "note": "numeric columns deterministic; verifier_sec/diag_sec are wall-clock"}
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=1)
    ok = all(s["gate_ok"] for s in summ)
    print(f"replayed {len(runs)} runs in {wall:.1f}s; gate (u1600 eta/peak/rmse within {GATE_TOL}): "
          f"{sum(s['gate_ok'] for s in summ)}/{len(summ)}; non-identical logged fields: {len(mism)}; "
          f"max |diff| {manifest['validation']['max_abs_diff']:.3g}")
    if not ok:
        print("STOP: the replay does not reproduce the runs' own u1600 verifier call "
              "(see validation_summary.csv); do not use these CSVs.")
        return 3
    return 0


# ===================================================================================== analysis
GROUPS: Tuple[Tuple[str, Optional[int]], ...] = (("pooled", None), ("q50", 50), ("q60", 60))


@dataclass
class Geometry:
    """Stage-2 exploring-start bins of one q (ES bin width 10)."""

    q: int
    n_bins: int
    edges: np.ndarray
    centers: np.ndarray
    labels: np.ndarray          # 0 tail, 1 near-tie (intersects (-20, 20)), 2 middle
    n_nontail: int
    outer: Tuple[int, int]      # the two outermost non-tail bins (adjacent to the support boundary)
    cap: int                    # ceil(loc_frac * n_nontail)


def geometry(spec: GameSpec) -> Geometry:
    """Bin geometry and strata of D_2 for ``spec`` (the sampler's own ``stratum_labels``)."""
    sm = StartSampler(spec, BIN_WIDTH)
    edges = sm.bin_edges(2)
    lab = sm.stratum_labels(2, NEAR_TIE_HALF_WIDTH)
    nt = np.flatnonzero(lab != 0)
    return Geometry(q=int(spec.q), n_bins=sm.n_bins(2), edges=edges, centers=0.5 * (edges[:-1] + edges[1:]),
                    labels=lab, n_nontail=int(nt.size), outer=(int(nt[0]), int(nt[-1])),
                    cap=int(math.ceil(LOC_FRAC * nt.size - 1e-12)))


class RunData:
    """The replay rows of one run, split by phase and tier."""

    def __init__(self, label: str, q: int, seed: int, rows: List[Dict[str, object]]) -> None:
        self.label, self.q, self.seed, self.rows = label, int(q), int(seed), rows
        self.a_dev = sorted((r for r in rows if r["phase"] == "A" and r["tier"] == "dev"),
                            key=lambda r: r["update"])
        self.b_dev = sorted((r for r in rows if r["phase"] == "B" and r["tier"] == "dev"),
                            key=lambda r: r["update"])
        fin = [r for r in rows if r["tier"] == "final"]
        self.a_final: Optional[Dict[str, object]] = fin[0] if fin else None
        self.a_by_u = {int(r["update"]): r for r in self.a_dev}
        self.b_by_u = {int(r["update"]): r for r in self.b_dev}

    @property
    def key(self) -> str:
        """``<root>/q<q>/seed<seed>``."""
        return f"{self.label}/q{self.q}/seed{self.seed}"


def load_runs(calib_dir: str) -> List[RunData]:
    """All per-run CSVs under ``calib_dir`` (reference-root order, q, seed)."""
    import glob
    runs: List[RunData] = []
    for label in DEFAULT_LABELS:
        for q in QS:
            paths = glob.glob(os.path.join(calib_dir, label, f"q{q}", "seed*.csv"))
            for p in sorted(paths, key=lambda s: int(os.path.basename(s)[4:-4])):
                runs.append(RunData(label, q, int(os.path.basename(p)[4:-4]), read_csv_rows(p)))
    return runs


def in_group(run: RunData, q: Optional[int]) -> bool:
    """Membership of a run in the pooled (``None``) or the per-q group."""
    return q is None or run.q == q


def diag_from_row(row: Mapping[str, object]) -> StageDiag:
    """The :class:`StageDiag` a stored replay row stands for (what the rule reads, nothing else)."""
    stage = int(row["stage"])
    keys = sorted((k for k in row if k.startswith("rho_bin_")), key=lambda k: int(k[len("rho_bin_"):]))
    rho = np.array([row[k] for k in keys], dtype=float) if stage >= 2 else np.zeros(0)
    return StageDiag(stage=stage, valid=bool(row["valid"]), delta_over_dw=float(row["Delta"]),
                     s=float(row["s"]), R=float(row["R"]), R_tail=float(row["R_tail"]),
                     C=float(row["C"]), tail_term=bool(row["tail_term"]), rho_bins=rho,
                     argmax_d=float(row["R_argmax_d"]))


# ------------------------------------------------------------------------------------- rule logic
def new_rule(rho: float, eps: float = EPS, tau: float = TAU, M: int = M_DEFAULT,
             K: int = K_DEFAULT, stage: int = 2) -> StageRule:
    """The first-order development rule of D4 (all other ``StageRule`` fields at their defaults)."""
    return StageRule(stage=stage, eps=eps, rho=rho, tau=tau, M=M, K=K)


def original_rule(eps: float, M: int = M_DEFAULT, K: int = K_DEFAULT, stage: int = 2) -> StageRule:
    """The original second-order rule: ``Delta <= eps`` and the concentration limit, no first-order part."""
    return StageRule(stage=stage, eps=eps, rho=math.inf, tau=math.inf, M=M, K=K)


def eligible(rule: StageRule, diag: StageDiag) -> bool:
    """``utils.ms_rule.is_eligible`` (the tail term is void for a rule with ``tau = inf``)."""
    if math.isinf(rule.tau):
        diag = dataclasses.replace(diag, tail_term=False)
    return is_eligible(rule, diag)


def fire_export(diags: Sequence[Tuple[int, StageDiag]], rule: StageRule, offset: int = 0
                ) -> Optional[int]:
    """The export at which the ``rule.M``-th consecutive eligible check occurs (None if it never does).

    Args:
        diags: ``(update, diag)`` of one phase in increasing update order, one per export.
        rule: The rule; a check is made only at exports whose local update ``update - offset`` is a
            multiple of ``rule.K`` (the export cadence is 25).
        offset: Global update at which the phase starts (0 for Phase A, 1600 for Phase B).
    """
    consec = 0
    for u, dg in diags:
        if (u - offset) % rule.K:
            continue
        consec = consec + 1 if eligible(rule, dg) else 0
        if consec >= rule.M:
            return int(u)
    return None


def run_diags(rows: Sequence[Mapping[str, object]]) -> List[Tuple[int, StageDiag]]:
    """``[(update, StageDiag)]`` of replay rows."""
    return [(int(r["update"]), diag_from_row(r)) for r in rows]


def rho_bar_series(rows: Sequence[Mapping[str, object]], beta: float = EMA_BETA
                   ) -> Dict[int, Optional[np.ndarray]]:
    """EMA ``rho_bar`` after every export (as the pipeline controller builds it: valid checks only)."""
    out: Dict[int, Optional[np.ndarray]] = {}
    cur: Optional[np.ndarray] = None
    for u, dg in run_diags(rows):
        if dg.valid and dg.rho_bins.size and np.isfinite(dg.rho_bins).any():
            cur = ema_update(cur, dg.rho_bins, beta)
        out[u] = None if cur is None else cur.copy()
    return out


# ----------------------------------------------------------------------------------- statistics
def rankdata(x: Sequence[float]) -> np.ndarray:
    """Average ranks (1-based; ties share the mean rank)."""
    a = np.asarray(x, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size)
    s = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation over the pairs where both are finite (NaN if < 3 pairs or constant)."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra, rb = rankdata(a[m]), rankdata(b[m])
    if ra.std() == 0.0 or rb.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation over the pairs where both are finite."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0.0 or b[m].std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def med_rng(v: Iterable[float]) -> Tuple[float, float, float]:
    """(median, min, max) of the finite values (NaNs if none)."""
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(a)), float(a.min()), float(a.max())


def fnum(x: object, nd: int = 4) -> str:
    """Compact number format for the markdown tables (``-`` for NaN, exponent for tiny values)."""
    if x is None:
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    xf = float(x)  # type: ignore[arg-type]
    if math.isnan(xf):
        return "-"
    if xf != 0.0 and abs(xf) < 10.0 ** (-nd):
        return f"{xf:.2e}"
    return f"{xf:.{nd}f}"


def mr(v: Iterable[float], nd: int = 4) -> str:
    """``median [min, max]`` as text."""
    m, lo, hi = med_rng(v)
    return "-" if math.isnan(m) else f"{fnum(m, nd)} [{fnum(lo, nd)}, {fnum(hi, nd)}]"


def md_table(columns: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """A GitHub markdown table."""
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


@dataclass
class Table:
    """One output table: raw rows for the CSV, formatted rows for the markdown fragment."""

    name: str
    title: str
    csv_cols: List[str]
    csv_rows: List[Dict[str, object]]
    md_cols: List[str]
    md_rows: List[List[object]]
    note: str = ""

    def md(self) -> str:
        """Markdown fragment (title, table, note)."""
        s = f"**{self.title}**\n\n" + md_table(self.md_cols, self.md_rows)
        return s + (f"\n{self.note}\n" if self.note else "")


# ---------------------------------------------------------------------------------- table 1
def metric_value(name: str, row: Mapping[str, object], geo: Optional[Geometry]) -> float:
    """A table-1 metric of one replay row (``R0`` = r_2(0)/s_2, ``R_near`` = max of rho_b over the near-tie
    bins)."""
    cols = {"|peak|": "stage2_peak_rel_err_abs", "RMSE_pos": "stage2_rmse_pos_over_g2_0",
            "tail mean": "stage2_tail_mean_over_g2_0", "R": "R", "R_tail": "R_tail", "Delta": "Delta"}
    if name in cols:
        return float(row[cols[name]])
    if name == "R0":
        return abs(float(row["e2_at_0"]) - float(row["s"])) / float(row["s"])
    if name == "R_near":
        assert geo is not None
        rho = np.array([row[f"rho_bin_{i}"] for i in range(geo.n_bins)], dtype=float)
        return float(np.nanmax(rho[geo.labels == 1]))
    raise KeyError(name)


def table1(runs: Sequence[RunData], geos: Mapping[int, Geometry]) -> Table:
    """Spearman correlation of the closed-form errors with R and Delta (all (run, export) pairs)."""
    pairs = [("|peak|", "R"), ("|peak|", "Delta"), ("|peak|", "R0"), ("|peak|", "R_near"),
             ("RMSE_pos", "R"), ("RMSE_pos", "Delta"), ("tail mean", "R_tail"), ("tail mean", "Delta"),
             ("R", "Delta")]
    subsets = (("u>=400", 400, PHASE_A_END), ("400<=u<=1200 (constant LR)", 400, 1200))
    cr: List[Dict[str, object]] = []
    mdr: List[List[object]] = []
    for sname, lo, hi in subsets:
        for gname, gq in GROUPS:
            for a, b in pairs:
                xs: List[float] = []
                ys: List[float] = []
                per: List[float] = []
                n_inv = 0
                for run in runs:
                    if not in_group(run, gq):
                        continue
                    sel = [r for r in run.a_dev if lo <= r["update"] <= hi]
                    n_inv += sum(1 for r in sel if not r["valid"])
                    sel = [r for r in sel if r["valid"]]
                    x = [metric_value(a, r, geos[run.q]) for r in sel]
                    y = [metric_value(b, r, geos[run.q]) for r in sel]
                    xs += x
                    ys += y
                    per.append(spearman(x, y))
                rho = spearman(xs, ys)
                m, mn, mx = med_rng(per)
                primary = bool(sname == "u>=400" and a == "|peak|" and b in ("R", "Delta"))
                suppl = b in ("R0", "R_near")
                cr.append({"subset": sname, "group": gname, "x": a, "y": b, "n_pairs": len(xs),
                           "n_invalid_dropped": n_inv, "spearman_pooled": rho, "within_run_median": m,
                           "within_run_min": mn, "within_run_max": mx, "primary": primary,
                           "supplementary_metric": suppl})
                mdr.append([("**" + sname + "**") if primary else sname, gname, f"{a} vs {b}", len(xs),
                            ("**" + fnum(rho, 3) + "**") if primary else fnum(rho, 3),
                            f"{fnum(m, 3)} [{fnum(mn, 3)}, {fnum(mx, 3)}]"])
    cols = ["subset", "group", "x vs y", "n pairs", "Spearman (pooled)", "within-run median [min, max]"]
    return Table("table1_spearman", "Table 1. Spearman correlation of the closed-form errors with the "
                 "residuals (Phase-A exports, dev tier; bold = the pair asked for in 2.4 item 1; R0 = "
                 "r_2(0)/s_2 at the node d = 0 and R_near = max of rho_b over the four near-tie bins are "
                 "supplementary, not rule quantities)", list(cr[0].keys()), cr, cols, mdr)


def table1b(runs: Sequence[RunData]) -> Table:
    """Supplementary: |peak error| against the node residual R0 = r_2(0)/s_2 (ratio distribution, u >= 400)."""
    rows: List[Dict[str, object]] = []
    mdr: List[List[object]] = []
    for gname, gq in GROUPS:
        sel = [r for run in runs if in_group(run, gq) for r in run.a_dev if r["update"] >= SPEARMAN_FROM_U]
        r0 = np.array([metric_value("R0", r, None) for r in sel])
        pk = np.array([float(r["stage2_peak_rel_err_abs"]) for r in sel])
        ratio = pk[r0 > 0.0] / r0[r0 > 0.0]                      # R0 = 0 exactly: a_dev(0) is the policy action
        q10, q50, q90 = (float(np.quantile(ratio, p)) for p in (0.1, 0.5, 0.9))
        rows.append({"group": gname, "n": len(sel), "n_R0_exactly_zero": int((r0 == 0.0).sum()),
                     "ratio_q10": q10, "ratio_q50": q50, "ratio_q90": q90,
                     "R0_median": float(np.median(r0)), "R0_q10": float(np.quantile(r0, 0.1)),
                     "R0_q90": float(np.quantile(r0, 0.9)), "R0_at_peak_0.05_from_median_ratio": 0.05 / q50,
                     "frac_R0_le_0.03": float(np.mean(r0 <= 0.03)),
                     "frac_peak_le_0.05": float(np.mean(pk <= 0.05))})
        mdr.append([gname, len(sel), f"{q10:.3f} / {q50:.3f} / {q90:.3f}",
                    " / ".join(fnum(rows[-1][k], 3) for k in ("R0_q10", "R0_median", "R0_q90")),
                    fnum(0.05 / q50, 4), fnum(rows[-1]["frac_R0_le_0.03"], 3),
                    fnum(rows[-1]["frac_peak_le_0.05"], 3)])
    return Table("table1b_peak_vs_R0", "Table 1b (supplementary). |peak error| / R0 with R0 = r_2(0)/s_2, "
                 "Phase-A exports u >= 400", list(rows[0].keys()), rows,
                 ["group", "n", "|peak|/R0 q10 / q50 / q90", "R0 q10 / q50 / q90",
                  "R0 where |peak| = 0.05 (median ratio)",
                  "exports with R0 <= 0.03", "exports with |peak| <= 0.05"], mdr)


def table_u1600(runs: Sequence[RunData]) -> Table:
    """The D3 quantities and the closed-form errors at u1600 (the end of v2.0's Phase A), dev tier."""
    cols = [("R", "R"), ("R_tail", "R_tail"), ("Delta (= eta_2/DW)", "Delta"), ("C", "C"), ("s_2 = a_dev(0)", "s"),
            ("|peak error|", "stage2_peak_rel_err_abs"), ("signed peak error", "stage2_peak_rel_err_signed"),
            ("RMSE_pos", "stage2_rmse_pos_over_g2_0"), ("tail mean", "stage2_tail_mean_over_g2_0"),
            ("tail max", "stage2_tail_max_over_g2_0")]
    out: List[Dict[str, object]] = []
    mdr: List[List[object]] = []
    for gname, gq in GROUPS:
        sel = [run.a_by_u[PHASE_A_END] for run in runs if in_group(run, gq)]
        for name, c in cols:
            v = [float(r[c]) for r in sel]
            m, lo, hi = med_rng(v)
            out.append({"group": gname, "quantity": name, "n": len(v), "median": m, "min": lo, "max": hi,
                        "q10": float(np.quantile(v, 0.1)), "q90": float(np.quantile(v, 0.9)),
                        "mean": float(np.mean(v))})
            mdr.append([gname, name, len(v), fnum(m, 4), fnum(lo, 4), fnum(hi, 4),
                        f"{fnum(np.quantile(v, 0.1), 4)} - {fnum(np.quantile(v, 0.9), 4)}"])
    return Table("table0_u1600", "Table 0. D3 quantities and closed-form errors at u1600 (end of v2.0's Phase A, "
                 "dev tier, one row per quantity)", list(out[0].keys()), out,
                 ["group", "quantity", "runs", "median", "min", "max", "q10 - q90"], mdr)


# ---------------------------------------------------------------------------------- table 2
def rule_outcome(run: RunData, rule: StageRule, name: str) -> Dict[str, object]:
    """Fire export of ``rule`` on the Phase-A exports of one run and the closed-form errors there."""
    fire = fire_export(run_diags(run.a_dev), rule, 0)
    ref = run.a_by_u[PHASE_A_END]
    mid = run.a_by_u[1200]
    nan = float("nan")
    out: Dict[str, object] = {
        "rule": name, "root": run.label, "q": run.q, "seed": run.seed, "fired": fire is not None,
        "fire_update": float(fire) if fire is not None else nan,
        "peak_u1600": ref["stage2_peak_rel_err_abs"], "peak_signed_u1600": ref["stage2_peak_rel_err_signed"],
        "rmse_u1600": ref["stage2_rmse_pos_over_g2_0"], "tail_u1600": ref["stage2_tail_mean_over_g2_0"],
        "peak_u1200": mid["stage2_peak_rel_err_abs"], "rmse_u1200": mid["stage2_rmse_pos_over_g2_0"],
        "tail_u1200": mid["stage2_tail_mean_over_g2_0"]}
    keys = ("peak", "peak_signed", "rmse", "tail", "tail_max", "R", "R_tail", "Delta", "C")
    src = ("stage2_peak_rel_err_abs", "stage2_peak_rel_err_signed", "stage2_rmse_pos_over_g2_0",
           "stage2_tail_mean_over_g2_0", "stage2_tail_max_over_g2_0", "R", "R_tail", "Delta", "C")
    if fire is None:
        for k in keys:
            out[f"{k}_fire"] = nan
        out["ok_g_a_parts"] = False
        out["peak_le_005"] = False
    else:
        fr = run.a_by_u[fire]
        for k, s in zip(keys, src):
            out[f"{k}_fire"] = fr[s]
        out["ok_g_a_parts"] = bool(fr["stage2_tail_mean_over_g2_0"] <= GA_TAIL
                                   and fr["stage2_rmse_pos_over_g2_0"] <= GA_RMSE)
        out["peak_le_005"] = bool(fr["stage2_peak_rel_err_abs"] <= 0.05)
    return out


def rule_grid() -> List[Tuple[str, StageRule]]:
    """The candidate rules of table 2: the original second-order rules (K = 25, and K = 100 = the v2.0
    log cadence) and the rho grid of the new rule (K = 25)."""
    g: List[Tuple[str, StageRule]] = [("orig eps=0.02", original_rule(0.02)),
                                      ("orig eps=0.005", original_rule(0.005))]
    for rho in RHO_GRID:
        g.append((f"new rho={rho:g}", new_rule(rho)))
    g += [("orig eps=0.02 K=100", original_rule(0.02, K=100)),
          ("orig eps=0.005 K=100", original_rule(0.005, K=100))]
    return g


def summarize_outcomes(outs: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Medians / ranges / counts of the per-run outcomes of ONE rule in ONE group."""
    n = len(outs)
    fired = [o for o in outs if o["fired"]]
    fu = [float(o["fire_update"]) for o in fired]
    s: Dict[str, object] = {"n_runs": n, "n_fire": len(fired), "n_never": n - len(fired)}
    for k in ("fire_update",):
        s[f"{k}_med"], s[f"{k}_min"], s[f"{k}_max"] = med_rng(fu)
    for k in ("peak", "rmse", "tail", "peak_signed", "R", "R_tail", "Delta"):
        s[f"{k}_fire_med"], s[f"{k}_fire_min"], s[f"{k}_fire_max"] = med_rng(o[f"{k}_fire"] for o in fired)
    for k in ("peak", "rmse", "tail"):
        s[f"{k}_u1600_med_fired"] = med_rng(o[f"{k}_u1600"] for o in fired)[0]
        s[f"{k}_u1600_med_all"] = med_rng(o[f"{k}_u1600"] for o in outs)[0]
        s[f"{k}_u1200_med_fired"] = med_rng(o[f"{k}_u1200"] for o in fired)[0]
    s["paired_peak_fire_minus_u1600_med"] = med_rng(float(o["peak_fire"]) - float(o["peak_u1600"])
                                                    for o in fired)[0]
    s["n_fire_after_1200"] = int(sum(1 for u in fu if u > 1200))
    s["n_ok_ga_parts"] = int(sum(1 for o in fired if o["ok_g_a_parts"]))
    s["n_peak_le_005"] = int(sum(1 for o in fired if o["peak_le_005"]))
    s["frac_ok_of_fired"] = (s["n_ok_ga_parts"] / len(fired)) if fired else float("nan")  # type: ignore
    s["frac_ok_of_all"] = s["n_ok_ga_parts"] / n if n else float("nan")  # type: ignore
    s["tail_fire_max"] = max((float(o["tail_fire"]) for o in fired), default=float("nan"))
    return s


def table2(runs: Sequence[RunData]) -> Tuple[Table, Table]:
    """Fire exports of every candidate rule: per-run detail table and the per-group summary."""
    per_run: List[Dict[str, object]] = []
    for name, rule in rule_grid():
        for run in runs:
            per_run.append(rule_outcome(run, rule, name))
    summ: List[Dict[str, object]] = []
    mdr: List[List[object]] = []
    for name, _ in rule_grid():
        for gname, gq in GROUPS:
            sel = [o for o in per_run if o["rule"] == name and (gq is None or o["q"] == gq)]
            s = summarize_outcomes(sel)
            summ.append({"rule": name, "group": gname, **s})
            mdr.append([name, gname, f"{s['n_fire']}/{s['n_runs']}", s["n_never"],
                        mr([o["fire_update"] for o in sel if o["fired"]], 0),
                        mr([o["peak_fire"] for o in sel if o["fired"]], 4),
                        fnum(s["peak_u1600_med_fired"]),
                        mr([o["rmse_fire"] for o in sel if o["fired"]], 4),
                        fnum(s["rmse_u1600_med_fired"]),
                        mr([o["tail_fire"] for o in sel if o["fired"]], 4),
                        fnum(s["tail_u1600_med_fired"]),
                        f"{s['n_ok_ga_parts']}/{s['n_fire']}" if s["n_fire"] else "-",
                        fnum(s["frac_ok_of_all"], 3)])
    cols = ["rule", "group", "fired", "never", "fire update med [min,max]", "|peak| at fire",
            "|peak| u1600 (same runs)", "RMSE_pos at fire", "RMSE_pos u1600", "tail mean at fire",
            "tail mean u1600", "fire with tail<=0.02 & RMSE<=0.05", "that share of ALL runs"]
    t_sum = Table("table2_rule_grid", "Table 2. Fire export of every candidate rule (M = 3, K = 25, tau = 0.02, "
                  "eps = 0.005 for the new rules; 'orig' = second-order only), closed-form errors at the fire "
                  "export against u1600 (medians over the runs that fire)",
                  list(summ[0].keys()), summ, cols, mdr)
    t_run = Table("table2_per_run", "Table 2 (per run)", list(per_run[0].keys()), per_run, [], [])
    return t_sum, t_run


# ---------------------------------------------------------------------------------- table 2c
EXT_RHOS: Tuple[float, ...] = (0.02, 0.03, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20)


def table2c(runs: Sequence[RunData]) -> Table:
    """Supplementary: the new rule (eps 0.005, tau 0.02, M 3, K 25) for a wider rho range (attainability)."""
    rows: List[Dict[str, object]] = []
    mdr: List[List[object]] = []
    for rho in EXT_RHOS:
        rule = new_rule(rho)
        outs = [rule_outcome(run, rule, f"rho={rho:g}") for run in runs]
        for gname, gq in GROUPS:
            sel = [o for o in outs if gq is None or o["q"] == gq]
            s = summarize_outcomes(sel)
            rows.append({"rho": rho, "group": gname, **s})
            mdr.append([f"{rho:g}", gname, f"{s['n_fire']}/{s['n_runs']}",
                        mr([o["fire_update"] for o in sel if o["fired"]], 0),
                        mr([o["peak_fire"] for o in sel if o["fired"]], 4), fnum(s["peak_u1600_med_fired"]),
                        mr([o["rmse_fire"] for o in sel if o["fired"]], 4),
                        mr([o["tail_fire"] for o in sel if o["fired"]], 4),
                        f"{s['n_ok_ga_parts']}/{s['n_fire']}" if s["n_fire"] else "-", s["n_fire_after_1200"]])
    return Table("table2c_extended_rho", "Table 2c (supplementary). The new rule for a wider rho range "
                 "(eps = 0.005, tau = 0.02, M = 3, K = 25): how often and when it fires on the v2.0 trajectories",
                 list(rows[0].keys()), rows,
                 ["rho", "group", "fired", "fire update med [min,max]", "|peak| at fire", "|peak| u1600 (same runs)",
                  "RMSE_pos at fire", "tail mean at fire", "fire with tail<=0.02 & RMSE<=0.05",
                  "fires after u1200 (inside v2.0's LR decay)"], mdr)


# ---------------------------------------------------------------------------------- table 2b
def table2b(runs: Sequence[RunData], geos: Mapping[int, Geometry]) -> Table:
    """Pass rates of every eligibility component (u >= 400 and at u1600) and R attainability."""
    cr: List[Dict[str, object]] = []
    mdr: List[List[object]] = []
    for scope, lo in (("u>=400", SPEARMAN_FROM_U), ("u1600 only", PHASE_A_END)):
        for gname, gq in GROUPS:
            sel = [(run, r) for run in runs if in_group(run, gq) for r in run.a_dev if r["update"] >= lo]
            n = len(sel)
            if not n:
                continue
            comp = {"valid": np.mean([bool(r["valid"]) for _, r in sel]),
                    "Delta<=0.005": np.mean([r["Delta"] <= EPS for _, r in sel]),
                    "R_tail<=0.02": np.mean([r["R_tail"] <= TAU for _, r in sel]),
                    "C<=0.04": np.mean([r["C"] <= CONC_LIMIT for _, r in sel])}
            for rho in RHO_GRID:
                comp[f"R<={rho:g}"] = np.mean([r["R"] <= rho for _, r in sel])
            for rho in RHO_GRID:
                comp[f"R_int<={rho:g}"] = np.mean([_r_interior(r, geos[run.q]) <= rho for run, r in sel])
            for rho in RHO_GRID:
                rule = new_rule(rho)
                comp[f"all eligible (rho={rho:g})"] = np.mean(
                    [eligible(rule, diag_from_row(r)) for _, r in sel])
            q10, q50_, q90 = (float(np.quantile([r["R"] for _, r in sel], p)) for p in (0.1, 0.5, 0.9))
            qt = [float(np.quantile([r["R_tail"] for _, r in sel], p)) for p in (0.1, 0.5, 0.9)]
            best = [min(r["R"] for r in run.a_dev if r["update"] >= lo) for run in runs if in_group(run, gq)]
            row: Dict[str, object] = {"scope": scope, "group": gname, "n": n,
                                      "R_q10": q10, "R_q50": q50_, "R_q90": q90,
                                      "R_tail_q10": qt[0], "R_tail_q50": qt[1], "R_tail_q90": qt[2],
                                      "R_best_per_run_median": med_rng(best)[0],
                                      "R_best_per_run_min": med_rng(best)[1],
                                      "R_best_per_run_max": med_rng(best)[2]}
            for rho in RHO_GRID:
                row[f"runs_with_any_R<={rho:g}"] = float(np.mean([b <= rho for b in best]))
            row.update({k: float(v) for k, v in comp.items()})
            cr.append(row)
            mdr.append([scope, gname, n] + [fnum(comp[k], 3) for k in
                                           ("valid", "Delta<=0.005", "R_tail<=0.02", "C<=0.04")]
                       + [fnum(comp[f"R<={r:g}"], 3) for r in RHO_GRID]
                       + [fnum(comp[f"R_int<={r:g}"], 3) for r in RHO_GRID]
                       + [fnum(comp[f"all eligible (rho={r:g})"], 3) for r in RHO_GRID]
                       + [f"{fnum(q10, 3)} / {fnum(q50_, 3)} / {fnum(q90, 3)}",
                          mr(best, 3)])
    cols = (["scope", "group", "n", "valid", "Delta<=0.005", "R_tail<=0.02", "C<=0.04"]
            + [f"R<={r:g}" for r in RHO_GRID] + [f"R_int<={r:g}" for r in RHO_GRID]
            + [f"all (rho={r:g})" for r in RHO_GRID] + ["R q10 / q50 / q90", "best R per run"])
    return Table("table2b_components", "Table 2b. Fraction of Phase-A exports passing each eligibility "
                 "component (R_int = R without the two outermost non-tail bins; 'best R per run' = the minimum "
                 "of R over the exports in scope, median [min, max] over runs)",
                 list(cr[0].keys()), cr, cols, mdr)


def _r_interior(row: Mapping[str, object], geo: Geometry) -> float:
    """R without the two outermost non-tail bins (those adjacent to the support boundary |d| = 2q)."""
    rho = np.array([row[f"rho_bin_{i}"] for i in range(geo.n_bins)], dtype=float)
    keep = (geo.labels != 0)
    keep[list(geo.outer)] = False
    return float(np.nanmax(rho[keep])) if np.isfinite(rho[keep]).any() else float("nan")


# ---------------------------------------------------------------------------------- table 3
def table3(runs: Sequence[RunData], geos: Mapping[int, Geometry]) -> Tuple[Table, Table]:
    """Dev-tier against final-tier D3 quantities at u1600 (grid noise of the first-order residual)."""
    per: List[Dict[str, object]] = []
    for run in runs:
        d, f = run.a_by_u[PHASE_A_END], run.a_final
        assert f is not None
        n_bins = geos[run.q].n_bins
        rd = np.array([d[f"rho_bin_{i}"] for i in range(n_bins)], dtype=float)
        rf = np.array([f[f"rho_bin_{i}"] for i in range(n_bins)], dtype=float)
        diff = np.abs(rd - rf)
        jm = int(np.nanargmax(diff))
        row: Dict[str, object] = {
            "root": run.label, "q": run.q, "seed": run.seed,
            "R_dev": d["R"], "R_final": f["R"], "dR": f["R"] - d["R"],
            "Rtail_dev": d["R_tail"], "Rtail_final": f["R_tail"], "dRtail": f["R_tail"] - d["R_tail"],
            "bin_maxabs_diff": float(diff[jm]), "bin_maxabs_diff_bin": jm,
            "bin_maxabs_diff_center": float(geos[run.q].centers[jm]),
            "bin_maxabs_diff_region": ("outermost" if jm in geos[run.q].outer else
                                       "near-tie" if geos[run.q].labels[jm] == 1 else "middle"),
            "bin_maxabs_diff_rel_to_R_dev": float(diff[jm]) / float(d["R"]),
            "Delta_dev": d["Delta"], "Delta_final": f["Delta"], "dDelta": f["Delta"] - d["Delta"],
            "s_dev": d["s"], "s_final": f["s"], "C_dev": d["C"], "C_final": f["C"],
            "argmax_d_dev": d["R_argmax_d"], "argmax_d_final": f["R_argmax_d"]}
        for rho in RHO_GRID:
            row[f"flip_R<={rho:g}"] = bool((d["R"] <= rho) != (f["R"] <= rho))
        per.append(row)
    summ: List[Dict[str, object]] = []
    mdr: List[List[object]] = []
    for gname, gq in GROUPS:
        sel = [p for p in per if gq is None or p["q"] == gq]
        if not sel:
            continue
        row = {"group": gname, "n_runs": len(sel)}
        for k in ("dR", "dRtail", "dDelta"):
            vals = [abs(float(p[k])) for p in sel]
            row[f"|{k}|_median"], row[f"|{k}|_min"], row[f"|{k}|_max"] = med_rng(vals)
        vals = [float(p["bin_maxabs_diff"]) for p in sel]
        row["bin_maxabs_median"], row["bin_maxabs_min"], row["bin_maxabs_max"] = med_rng(vals)
        for reg in ("outermost", "near-tie", "middle"):
            row[f"maxdiff_bin_{reg}"] = int(sum(p["bin_maxabs_diff_region"] == reg for p in sel))
        row["R_dev_median"] = med_rng(p["R_dev"] for p in sel)[0]
        row["R_final_median"] = med_rng(p["R_final"] for p in sel)[0]
        row["signed_dR_median"] = float(np.median([p["dR"] for p in sel]))
        for rho in RHO_GRID:
            row[f"n_flip_R<={rho:g}"] = int(sum(p[f"flip_R<={rho:g}"] for p in sel))
        summ.append(row)
        mdr.append([gname, len(sel), mr([abs(p["dR"]) for p in sel], 4), mr([abs(p["dRtail"]) for p in sel], 5),
                    mr(vals, 4), "/".join(str(row[f"maxdiff_bin_{g}"]) for g in ("outermost", "near-tie", "middle")),
                    fnum(row["signed_dR_median"], 4), mr([abs(p["dDelta"]) for p in sel], 5),
                    "/".join(str(row[f"n_flip_R<={r:g}"]) for r in RHO_GRID)])
    cols = ["group", "runs", "|R_final - R_dev| med [min,max]", "|R_tail diff| med [min,max]",
            "max over bins |rho_final,b - rho_dev,b|: med [min,max] over runs",
            "that bin is outermost / near-tie / other middle", "median signed R diff",
            "|Delta diff| med [min,max]", "R<=rho flips dev->final (rho .02/.03/.05)"]
    t = Table("table3_dev_vs_final", "Table 3. Dev tier vs final tier at u1600 (state step 4 vs 2, effort step 1 vs "
              "0.5, GL 16 vs 32 nodes per half): grid noise of the first-order residual",
              list(summ[0].keys()), summ, cols, mdr)
    tp = Table("table3_per_run", "Table 3 (per run)", list(per[0].keys()), per, [], [])
    return t, tp


# ---------------------------------------------------------------------------------- table 4
def classify_export(rho_bar: Optional[np.ndarray], R: float, geo: Geometry, rho: float
                    ) -> Dict[str, object]:
    """Localized / broad classification (pipeline rule) and where the bins of S sit."""
    cls, S = classify_block_end(R, rho_bar, rho, geo.n_nontail, LOC_FRAC)
    lab = geo.labels[S] if S.size else np.zeros(0, dtype=int)
    n_near, n_mid = int((lab == 1).sum()), int((lab == 2).sum())
    outer = int(sum(1 for b in S if int(b) in geo.outer))
    n_neg = int(sum(1 for b in S if geo.centers[int(b)] < 0.0))
    return {"class": cls, "n_S": int(S.size), "S_bins": ";".join(str(int(b)) for b in S),
            "S_centers": ";".join(f"{geo.centers[int(b)]:g}" for b in S), "n_S_near": n_near,
            "n_S_mid": n_mid, "n_S_outer": outer, "n_S_neg": n_neg, "n_S_pos": int(S.size) - n_neg,
            "S_within_cap_ignoring_R": bool(1 <= S.size <= geo.cap), "cap": geo.cap}


def argmax_region(d: float, geo: Geometry) -> str:
    """Region of the node d at which R is attained: near-tie, middle, or the boundary band."""
    ad = abs(d)
    if ad < NEAR_TIE_HALF_WIDTH:
        return "near-tie"
    if ad >= 2.0 * geo.q - BIN_WIDTH:
        return "boundary"
    return "middle"


def table4(runs: Sequence[RunData], geos: Mapping[int, Geometry]) -> Tuple[Table, Table, Table, Table]:
    """Classification at u1600 / block ends / fire exports, plus the residual-location tables."""
    det: List[Dict[str, object]] = []
    series = {run.key: rho_bar_series(run.a_dev) for run in runs}
    for run in runs:
        geo = geos[run.q]
        for rho in RHO_GRID:
            fire = fire_export(run_diags(run.a_dev), new_rule(rho), 0)
            locs: List[Tuple[str, int]] = [(f"u{u}", u) for u in (400, 800, 1200, 1600)]
            if fire is not None:
                locs.append(("fire", fire))
            for loc, u in locs:
                r = run.a_by_u[u]
                c = classify_export(series[run.key][u], float(r["R"]), geo, rho)
                det.append({"rho": rho, "location": loc, "root": run.label, "q": run.q, "seed": run.seed,
                            "update": u, "R": r["R"], "R_argmax_d": r["R_argmax_d"],
                            "argmax_region": argmax_region(float(r["R_argmax_d"]), geo), **c})
    summ: List[Dict[str, object]] = []
    mdr: List[List[object]] = []
    for rho in RHO_GRID:
        for loc in ("u1600", "fire", "u400", "u800", "u1200"):
            for gname, gq in GROUPS:
                sel = [d for d in det if d["rho"] == rho and d["location"] == loc
                       and (gq is None or d["q"] == gq)]
                if not sel:
                    continue
                n_s = [d["n_S"] for d in sel]
                tot_s = int(sum(n_s))
                near_slots = int(sum((geos[d["q"]].labels == 1).sum() for d in sel))
                mid_slots = int(sum((geos[d["q"]].labels == 2).sum() for d in sel))
                row = {"rho": rho, "location": loc, "group": gname, "n": len(sel),
                       "n_localized": int(sum(d["class"] == "localized" for d in sel)),
                       "n_broad": int(sum(d["class"] == "broad" for d in sel)),
                       "n_S_median": float(np.median(n_s)), "n_S_min": int(min(n_s)), "n_S_max": int(max(n_s)),
                       "n_S_zero": int(sum(x == 0 for x in n_s)),
                       "n_S_over_cap": int(sum(d["n_S"] > d["cap"] for d in sel)),
                       "n_within_cap_ignoring_R": int(sum(d["S_within_cap_ignoring_R"] for d in sel)),
                       "S_bins_total": tot_s, "S_near_total": int(sum(d["n_S_near"] for d in sel)),
                       "S_mid_total": int(sum(d["n_S_mid"] for d in sel)),
                       "S_outer_total": int(sum(d["n_S_outer"] for d in sel)),
                       "S_neg_total": int(sum(d["n_S_neg"] for d in sel)),
                       "S_pos_total": int(sum(d["n_S_pos"] for d in sel)),
                       "near_slots": near_slots, "mid_slots": mid_slots,
                       "near_frac_in_S": (int(sum(d["n_S_near"] for d in sel)) / near_slots
                                          if near_slots else float("nan")),
                       "mid_frac_in_S": (int(sum(d["n_S_mid"] for d in sel)) / mid_slots
                                         if mid_slots else float("nan")),
                       "argmax_near": int(sum(d["argmax_region"] == "near-tie" for d in sel)),
                       "argmax_mid": int(sum(d["argmax_region"] == "middle" for d in sel)),
                       "argmax_boundary": int(sum(d["argmax_region"] == "boundary" for d in sel))}
                summ.append(row)
                mdr.append([f"{rho:g}", loc, gname, len(sel), f"{row['n_localized']}/{row['n_broad']}",
                            f"{fnum(row['n_S_median'], 1)} [{row['n_S_min']}, {row['n_S_max']}]",
                            row["n_S_zero"], row["n_S_over_cap"], row["n_within_cap_ignoring_R"],
                            f"{row['S_near_total']}/{row['S_mid_total']}",
                            f"{fnum(row['near_frac_in_S'], 2)} / {fnum(row['mid_frac_in_S'], 2)}",
                            f"{row['S_neg_total']}/{row['S_pos_total']}",
                            f"{row['argmax_near']}/{row['argmax_mid']}/{row['argmax_boundary']}"])
    cols = ["rho", "where", "group", "n", "localized / broad", "|S| med [min,max]", "|S|=0", "|S|>cap",
            "|S| in 1..cap (R clause ignored)", "S bins near-tie / middle (total)",
            "share of near-tie / of middle bins in S", "S bins d<0 / d>0 (total)",
            "R argmax near-tie / middle / boundary"]
    t4 = Table("table4_classification", "Table 4. Localized / broad classification (loc_frac 0.25, EMA beta 0.5 "
               "from u0025) at the block ends of N_block = 400, at u1600 and at each rho's fire export; at the "
               "fire export R <= rho by construction, so the pipeline classification is 'broad' there",
               list(summ[0].keys()), summ, cols, mdr)
    t4d = Table("table4_detail", "Table 4 (per run)", list(det[0].keys()), det, [], [])
    # residual map at u1600 per bin
    bm: List[Dict[str, object]] = []
    for q in QS:
        geo = geos[q]
        sel = [run for run in runs if run.q == q]
        mats = np.array([series[run.key][PHASE_A_END] for run in sel], dtype=float)
        raw = np.array([[run.a_by_u[PHASE_A_END][f"rho_bin_{i}"] for i in range(geo.n_bins)] for run in sel],
                       dtype=float)
        for i in range(geo.n_bins):
            if geo.labels[i] == 0:
                continue
            row = {"q": q, "bin": i, "center": float(geo.centers[i]),
                   "stratum": {1: "near-tie", 2: "middle"}[int(geo.labels[i])],
                   "outer": bool(i in geo.outer), "rho_u1600_median": float(np.median(raw[:, i])),
                   "rho_u1600_q10": float(np.quantile(raw[:, i], 0.1)),
                   "rho_u1600_q90": float(np.quantile(raw[:, i], 0.9)),
                   "rho_u1600_max": float(raw[:, i].max()),
                   "rhobar_u1600_median": float(np.median(mats[:, i]))}
            for rho in RHO_GRID:
                row[f"frac_runs_rhobar>{rho:g}"] = float(np.mean(mats[:, i] > rho))
                row[f"frac_runs_rho>{rho:g}"] = float(np.mean(raw[:, i] > rho))
            bm.append(row)
    t4b = Table("table4_bin_map", "Table 4 (per-bin map at u1600)", list(bm[0].keys()), bm, [], [])
    # location of the argmax of R over all exports u >= 400
    am: List[Dict[str, object]] = []
    mda: List[List[object]] = []
    for scope, lo in (("u>=400", SPEARMAN_FROM_U), ("u1600", PHASE_A_END)):
        for gname, gq in GROUPS:
            sel = [(run, r) for run in runs if in_group(run, gq) for r in run.a_dev if r["update"] >= lo]
            reg = [argmax_region(float(r["R_argmax_d"]), geos[run.q]) for run, r in sel]
            rel = [abs(float(r["R_argmax_d"])) / (2.0 * run.q) for run, r in sel]
            rint = [_r_interior(r, geos[run.q]) for run, r in sel]
            rr = [float(r["R"]) for _, r in sel]
            row = {"scope": scope, "group": gname, "n": len(sel),
                   "frac_near_tie": reg.count("near-tie") / len(reg),
                   "frac_middle": reg.count("middle") / len(reg),
                   "frac_boundary": reg.count("boundary") / len(reg),
                   "abs_argmax_over_2q_median": float(np.median(rel)),
                   "frac_argmax_d_negative": float(np.mean([float(r["R_argmax_d"]) < 0.0 for _, r in sel])),
                   "R_median": float(np.median(rr)), "R_int_median": float(np.median(rint)),
                   "frac_R_int_lt_R": float(np.mean([a < b - 1e-15 for a, b in zip(rint, rr)]))}
            am.append(row)
            mda.append([scope, gname, len(sel), fnum(row["frac_near_tie"], 3), fnum(row["frac_middle"], 3),
                        fnum(row["frac_boundary"], 3), fnum(row["frac_argmax_d_negative"], 3),
                        fnum(row["abs_argmax_over_2q_median"], 3),
                        fnum(row["R_median"], 4), fnum(row["R_int_median"], 4)])
    t4c = Table("table4_argmax", "Table 4 (where the maximum of R sits: near-tie |d|<20, boundary band "
                "|d| >= 2q-10, middle = the rest)", list(am[0].keys()), am,
                ["scope", "group", "n", "near-tie", "middle", "boundary band", "argmax d < 0",
                 "median |argmax d|/2q", "median R", "median R_int"], mda)
    return t4, t4d, t4b, t4c


def table4_ema(runs: Sequence[RunData], geos: Mapping[int, Geometry]) -> Table:
    """Descriptive: export-to-export noise of the per-bin map and the lag of the EMA, for a few betas."""
    rows: List[Dict[str, object]] = []
    mdr: List[List[object]] = []
    for beta in (0.0, 0.5, 0.8):
        consec: List[float] = []
        lag: List[float] = []
        for run in runs:
            nt = geos[run.q].labels != 0
            ser = rho_bar_series(run.a_dev, beta)
            raw = {int(r["update"]): np.array([r[f"rho_bin_{i}"] for i in range(geos[run.q].n_bins)], dtype=float)
                   for r in run.a_dev}
            for u in sorted(raw):
                if u < SPEARMAN_FROM_U:
                    continue
                consec += list(np.abs(ser[u][nt] - ser[u - 25][nt]))
                lag += list(np.abs(ser[u][nt] - raw[u][nt]))
        rows.append({"beta": beta, "n": len(consec), "consecutive_change_median": float(np.median(consec)),
                     "consecutive_change_q90": float(np.quantile(consec, 0.9)),
                     "gap_to_raw_median": float(np.median(lag)), "gap_to_raw_q90": float(np.quantile(lag, 0.9))})
        mdr.append([f"{beta:g}", len(consec), fnum(rows[-1]["consecutive_change_median"], 4),
                    fnum(rows[-1]["consecutive_change_q90"], 4), fnum(rows[-1]["gap_to_raw_median"], 4),
                    fnum(rows[-1]["gap_to_raw_q90"], 4)])
    return Table("table4_ema_beta", "Table 4c (descriptive). Per-bin map, exports u >= 400: change of the EMA "
                 "between consecutive exports (25 updates) and its gap to the raw map rho_b, by EMA weight beta "
                 "(beta = 0 is the raw map)", list(rows[0].keys()), rows,
                 ["beta", "(run, bin, export) triples", "median |change| / 25 updates", "q90 |change|",
                  "median |EMA - raw|", "q90 |EMA - raw|"], mdr)


def table4_locfrac(runs: Sequence[RunData], geos: Mapping[int, Geometry]) -> Table:
    """Descriptive: how many runs are 'localized' at the block ends of 400 for other localized fractions."""
    series = {run.key: rho_bar_series(run.a_dev) for run in runs}
    rows: List[Dict[str, object]] = []
    mdr: List[List[object]] = []
    for rho in RHO_GRID:
        for frac in (0.15, 0.25, 0.35, 0.50):
            cells: List[object] = [f"{rho:g}", f"{frac:g}"]
            row: Dict[str, object] = {"rho": rho, "loc_frac": frac}
            for u in (400, 800, 1200, 1600):
                n_loc = 0
                for run in runs:
                    geo = geos[run.q]
                    cls, _ = classify_block_end(float(run.a_by_u[u]["R"]), series[run.key][u], rho,
                                                geo.n_nontail, frac)
                    n_loc += int(cls == "localized")
                row[f"n_localized_u{u}"] = n_loc
                cells.append(f"{n_loc}/{len(runs)}")
            rows.append(row)
            mdr.append(cells)
    return Table("table4_localized_fraction", "Table 4d (descriptive). Runs classified 'localized' at the "
                 "block ends u400 / u800 / u1200 / u1600 for other localized fractions (pipeline value 0.25)",
                 list(rows[0].keys()), rows,
                 ["rho", "localized fraction", "u400", "u800", "u1200", "u1600"], mdr)


# ---------------------------------------------------------------------------------- table 5
def _stage1_decomp(row: Mapping[str, object]) -> Dict[str, float]:
    """Total / inherited / learning error of the stage-1 effort, relative to the closed form g1.

    ``g1 = e1_hat / (1 + rel_err)`` is the closed-form value (reporting only); ``s = a_dev_1(0)`` is the
    one-step best-response effort against the candidate's own (frozen) continuation.
    """
    e1, rel, s = float(row["e1_at_0"]), float(row["stage1_rel_err_signed"]), float(row["s"])
    g1 = e1 / (1.0 + rel)
    return {"g1": g1, "total": (e1 - g1) / g1, "inherited_proxy": (s - g1) / g1,
            "learning_proxy": (e1 - s) / g1}


def table5(runs: Sequence[RunData]) -> Tuple[Table, Table, Table, Table]:
    """Phase B: the stage-1 residual against the stage-1 error, fire exports, LR bands, u2200 numbers."""
    corr: List[Dict[str, object]] = []
    mdc: List[List[object]] = []
    for gname, gq in GROUPS:
        sel = [r for run in runs if in_group(run, gq) for r in run.b_dev]
        if not sel:
            continue
        R = [float(r["R"]) for r in sel]
        err = [float(r["stage1_rel_err_signed"]) for r in sel]
        rs = [(float(r["e1_at_0"]) - float(r["s"])) / float(r["s"]) for r in sel]
        learn = [_stage1_decomp(r)["learning_proxy"] for r in sel]
        per_abs = [spearman([r["R"] for r in run.b_dev], [abs(r["stage1_rel_err_signed"]) for r in run.b_dev])
                   for run in runs if in_group(run, gq)]
        per_sgn = [spearman([(r["e1_at_0"] - r["s"]) / r["s"] for r in run.b_dev],
                            [r["stage1_rel_err_signed"] for r in run.b_dev]) for run in runs if in_group(run, gq)]
        row = {"group": gname, "n_pairs": len(sel), "n_R1_exactly_zero": int(sum(1 for x in R if x == 0.0)),
               "frac_C1_le_0.04": float(np.mean([r["C"] <= CONC_LIMIT for r in sel])),
               "spearman_R_vs_abs_err": spearman(R, [abs(e) for e in err]),
               "pearson_R_vs_abs_err": pearson(R, [abs(e) for e in err]),
               "spearman_signedres_vs_signed_err": spearman(rs, err),
               "pearson_signedres_vs_signed_err": pearson(rs, err),
               "pearson_learning_proxy_vs_signed_err": pearson(learn, err),
               "within_run_median_spearman_R_vs_abs_err": med_rng(per_abs)[0],
               "within_run_median_spearman_signed": med_rng(per_sgn)[0]}
        corr.append(row)
        mdc.append([gname, len(sel), row["n_R1_exactly_zero"], fnum(row["frac_C1_le_0.04"], 3),
                    fnum(row["spearman_R_vs_abs_err"], 3), fnum(row["pearson_R_vs_abs_err"], 3),
                    fnum(row["spearman_signedres_vs_signed_err"], 3), fnum(row["pearson_signedres_vs_signed_err"], 3),
                    fnum(row["pearson_learning_proxy_vs_signed_err"], 3),
                    fnum(row["within_run_median_spearman_R_vs_abs_err"], 3),
                    fnum(row["within_run_median_spearman_signed"], 3)])
    t_corr = Table("table5_corr", "Table 5a. Phase B (stage 1, u1625..u2200, every export): stage-1 residual "
                   "r_1(0)/s_1 against the closed-form stage-1 error (signed residual = (e_hat_1 - s_1)/s_1; "
                   "learning proxy = (e_hat_1 - s_1)/g1)", list(corr[0].keys()), corr,
                   ["group", "n pairs", "exports with R_1 = 0 exactly", "exports with C_1 <= 0.04",
                    "Spearman R vs |err|", "Pearson R vs |err|",
                    "Spearman signed res vs err",
                    "Pearson signed res vs err", "Pearson learning proxy vs err",
                    "within-run median Spearman R vs |err|", "within-run median Spearman signed"], mdc)
    # u2200 numbers (per run) and summary
    u22: List[Dict[str, object]] = []
    for run in runs:
        r = run.b_by_u[2200]
        dec = _stage1_decomp(r)
        u22.append({"root": run.label, "q": run.q, "seed": run.seed, "R_1": r["R"], "Delta_1": r["Delta"],
                    "C_1": r["C"], "s_1": r["s"], "e1_at_0": r["e1_at_0"],
                    "stage1_rel_err_signed": r["stage1_rel_err_signed"],
                    "signed_residual": (r["e1_at_0"] - r["s"]) / r["s"], "inherited_proxy": dec["inherited_proxy"],
                    "learning_proxy": dec["learning_proxy"], "valid": r["valid"]})
    t_u22 = Table("table5_u2200_per_run", "Table 5 (u2200 per run)", list(u22[0].keys()), u22, [], [])
    sm: List[Dict[str, object]] = []
    mds: List[List[object]] = []
    for gname, gq in GROUPS:
        sel = [u for u in u22 if gq is None or u["q"] == gq]
        if not sel:
            continue
        row = {"group": gname, "n_runs": len(sel)}
        for k in ("R_1", "Delta_1", "C_1", "stage1_rel_err_signed", "inherited_proxy", "learning_proxy"):
            row[f"{k}_median"], row[f"{k}_min"], row[f"{k}_max"] = med_rng(u[k] for u in sel)
        row["abs_err_median"] = med_rng(abs(u["stage1_rel_err_signed"]) for u in sel)[0]
        sm.append(row)
        mds.append([gname, len(sel), mr([u["R_1"] for u in sel], 4), mr([u["Delta_1"] for u in sel], 5),
                    mr([u["C_1"] for u in sel], 4),
                    mr([u["stage1_rel_err_signed"] for u in sel], 4), fnum(row["abs_err_median"], 4),
                    mr([u["inherited_proxy"] for u in sel], 4), mr([u["learning_proxy"] for u in sel], 4)])
    t_sm = Table("table5_u2200", "Table 5b. Stage 1 at u2200 (the end of v2.0's 600-update decay), median "
                 "[min, max] over runs", list(sm[0].keys()), sm,
                 ["group", "runs", "R_1 = r_1(0)/s_1", "Delta_1", "C_1", "signed stage-1 error", "median |error|",
                  "inherited proxy (s_1 - g1)/g1", "learning proxy (e_hat_1 - s_1)/g1"], mds)
    # fire exports of the stage-1 rule
    fr: List[Dict[str, object]] = []
    mdf: List[List[object]] = []
    for rho in RHO_GRID:
        rule = new_rule(rho, tau=math.inf, stage=1)
        for gname, gq in GROUPS:
            sel = [run for run in runs if in_group(run, gq)]
            if not sel:
                continue
            fires = [fire_export(run_diags(run.b_dev), rule, PHASE_A_END) for run in sel]
            loc = [f - PHASE_A_END for f in fires if f is not None]
            err_fire = [abs(run.b_by_u[f]["stage1_rel_err_signed"]) for run, f in zip(sel, fires) if f is not None]
            err_2200 = [abs(run.b_by_u[2200]["stage1_rel_err_signed"]) for run, f in zip(sel, fires) if f is not None]
            row = {"rho": rho, "group": gname, "n_runs": len(sel), "n_fire": len(loc),
                   "n_never": len(sel) - len(loc)}
            row["fire_local_median"], row["fire_local_min"], row["fire_local_max"] = med_rng(loc)
            sv = [600 - x for x in loc]
            row["saved_training_median"], row["saved_training_min"], row["saved_training_max"] = med_rng(sv)
            tot = [x + 400 for x in loc]
            row["total_with_land400_median"] = med_rng(tot)[0]
            row["err_at_fire_abs_median"], row["err_at_fire_abs_min"], row["err_at_fire_abs_max"] = med_rng(err_fire)
            row["err_u2200_abs_median_same_runs"] = med_rng(err_2200)[0]
            row["n_err_fire_gt_0p05"] = int(sum(e > 0.05 for e in err_fire))
            row["n_fire_with_R1_zero_in_firing_checks"] = int(sum(
                1 for run, f in zip(sel, fires) if f is not None
                and any(run.b_by_u[f - 25 * i]["R"] == 0.0 for i in range(rule.M))))
            fr.append(row)
            mdf.append([f"{rho:g}", gname, f"{len(loc)}/{len(sel)}", mr(loc, 0), mr(sv, 0),
                        fnum(row["total_with_land400_median"], 0), mr(err_fire, 4),
                        fnum(row["err_u2200_abs_median_same_runs"], 4), row["n_err_fire_gt_0p05"],
                        row["n_fire_with_R1_zero_in_firing_checks"]])
    t_fire = Table("table5_fire", "Table 5c. Stage-1 rule (eps = 0.005, M = 3, K = 25, no tail term): fire export in "
                   "local updates of Phase B, training updates saved against v2.0's fixed 600, total updates with a "
                   "400-update landing window, and the closed-form |stage-1 error| at the fire export",
                   list(fr[0].keys()), fr,
                   ["rho", "group", "fired", "fire local update med [min,max]", "600 - fire (training updates saved)",
                    "fire + 400 (median)", "|err_1| at fire", "|err_1| at u2200 (same runs)", "runs with |err_1|>0.05",
                    "fires with R_1 = 0 in a firing check"],
                   mdf)
    return t_corr, t_sm, t_u22, t_fire


def lr_v20_phase_b(local: int) -> float:
    """v2.0's Phase-B learning rate before local update ``local`` (linear 3e-4 -> 3e-5 over 1..600)."""
    return 3e-4 + (3e-5 - 3e-4) * (local - 1) / (600 - 1)


def table5_lr(runs: Sequence[RunData]) -> Table:
    """Descriptive: the stage-1 error noise by v2.0 learning-rate band (the 600-update decay)."""
    bands = ((1, 200), (201, 400), (401, 600))
    rows: List[Dict[str, object]] = []
    mdr: List[List[object]] = []
    for lo, hi in bands:
        for gname, gq in GROUPS:
            sel = [(run, [r for r in run.b_dev if lo <= r["update"] - PHASE_A_END <= hi]) for run in runs
                   if in_group(run, gq)]
            errs = [abs(r["stage1_rel_err_signed"]) for _, rs in sel for r in rs]
            swing = []
            for _, rs in sel:
                e = [r["stage1_rel_err_signed"] for r in rs]
                swing += [abs(b - a) for a, b in zip(e[:-1], e[1:])]
            Rv = [r["R"] for _, rs in sel for r in rs]
            row = {"local_from": lo, "local_to": hi, "group": gname, "lr_at_from": lr_v20_phase_b(lo),
                   "lr_at_to": lr_v20_phase_b(hi), "n_exports": len(errs), "abs_err_median": float(np.median(errs)),
                   "abs_err_q90": float(np.quantile(errs, 0.9)), "consecutive_swing_median": float(np.median(swing)),
                   "consecutive_swing_q90": float(np.quantile(swing, 0.9)), "R_median": float(np.median(Rv)),
                   "R_q90": float(np.quantile(Rv, 0.9))}
            rows.append(row)
            mdr.append([f"{lo}-{hi}", f"{row['lr_at_from']:.2e} -> {row['lr_at_to']:.2e}", gname, len(errs),
                        fnum(row["abs_err_median"], 4), fnum(row["abs_err_q90"], 4),
                        fnum(row["consecutive_swing_median"], 4), fnum(row["consecutive_swing_q90"], 4),
                        fnum(row["R_median"], 4), fnum(row["R_q90"], 4)])
    return Table("table5_lr_bands", "Table 5d. Stage-1 error noise by v2.0 learning-rate band (descriptive; v2.0 "
                 "decays 3e-4 -> 3e-5 over the 600 local updates of Phase B; the MS stage-1 phase keeps 3e-4 until "
                 "the stop and then decays over N_land = 400)", list(rows[0].keys()), rows,
                 ["local updates", "v2.0 LR", "group", "exports", "median |err_1|", "q90 |err_1|",
                  "median |err_1(u) - err_1(u-25)|", "q90 swing", "median R_1", "q90 R_1"], mdr)


def table4_landing(runs: Sequence[RunData]) -> Table:
    """Descriptive: the closed-form and D3 quantities at u1200 (before v2.0's 400-update decay) and u1600."""
    rows: List[Dict[str, object]] = []
    mdr: List[List[object]] = []
    for gname, gq in GROUPS:
        sel = [run for run in runs if in_group(run, gq)]
        row: Dict[str, object] = {"group": gname, "n_runs": len(sel)}
        cells: List[object] = [gname, len(sel)]
        for name, col in (("|peak|", "stage2_peak_rel_err_abs"), ("RMSE_pos", "stage2_rmse_pos_over_g2_0"),
                          ("tail mean", "stage2_tail_mean_over_g2_0"), ("R", "R"), ("R_tail", "R_tail"),
                          ("Delta", "Delta")):
            a = [run.a_by_u[1200][col] for run in sel]
            b = [run.a_by_u[1600][col] for run in sel]
            row[f"{name}_u1200_median"], row[f"{name}_u1600_median"] = float(np.median(a)), float(np.median(b))
            row[f"{name}_paired_change_median"] = float(np.median([y - x for x, y in zip(a, b)]))
            row[f"{name}_n_decrease"] = int(sum(y < x for x, y in zip(a, b)))
            cells.append(f"{fnum(np.median(a), 4)} -> {fnum(np.median(b), 4)}")
        rows.append(row)
        mdr.append(cells)
    return Table("table4_landing", "Table 4b. Effect of v2.0's 400-update LR decay on the Phase-A candidate "
                 "(u1200 = last constant-LR export, u1600 = end of the decay): medians over runs",
                 list(rows[0].keys()), rows, ["group", "runs", "|peak|", "RMSE_pos", "tail mean", "R", "R_tail",
                                              "Delta"], mdr)


# ---------------------------------------------------------------------------------- table 6
def table6(runs: Sequence[RunData], root_v2: str) -> Table:
    """Cost of one development / final verifier call (replay and the runs' own logs)."""
    def stats(v: Sequence[float]) -> Dict[str, float]:
        a = np.asarray(v, dtype=float)
        return {"n": int(a.size), "median": float(np.median(a)), "mean": float(a.mean()),
                "q10": float(np.quantile(a, 0.1)), "q90": float(np.quantile(a, 0.9)),
                "min": float(a.min()), "max": float(a.max())}
    sources: List[Tuple[str, List[float]]] = []
    rows_all = [r for run in runs for r in run.rows]
    sources.append(("replay dev tier, Phase A (stage 2)", [r["verifier_sec"] for r in rows_all
                                                           if r["tier"] == "dev" and r["phase"] == "A"]))
    sources.append(("replay dev tier, Phase B (stage 1)", [r["verifier_sec"] for r in rows_all
                                                           if r["tier"] == "dev" and r["phase"] == "B"]))
    sources.append(("replay final tier (u1600)", [r["verifier_sec"] for r in rows_all if r["tier"] == "final"]))
    sources.append(("replay D3 diag + concentration (dev, Phase A)", [r["diag_sec"] for r in rows_all
                                                                      if r["tier"] == "dev" and r["phase"] == "A"]))
    for ph in ("A", "B"):
        v: List[float] = []
        for run in runs:
            p = Path(root_v2) / run.label / f"q{run.q}" / f"seed{run.seed}" / f"v2_checkpoints_{ph}.csv"
            v += [float(x["verifier_sec"]) for x in read_log_rows(str(p))]
        sources.append((f"v2.0 own log v2_checkpoints_{ph}.csv (dev tier)", v))
    out: List[Dict[str, object]] = []
    mdr: List[List[object]] = []
    for name, v in sources:
        s = stats(v)
        out.append({"source": name, **s})
        mdr.append([name, s["n"], f"{1e3 * s['median']:.1f}", f"{1e3 * s['mean']:.1f}",
                    f"{1e3 * s['q10']:.1f} - {1e3 * s['q90']:.1f}", f"{1e3 * s['min']:.1f} - {1e3 * s['max']:.1f}"])
    return Table("table6_cost", "Table 6. Wall-clock of one verifier call at T = 2 (milliseconds)",
                 list(out[0].keys()), out, ["source", "calls", "median ms", "mean ms", "q10 - q90 ms", "min - max ms"],
                 mdr)


# ===================================================================================== facts (section 0)
def grep_lines(path: Path, needles: Sequence[str]) -> Dict[str, List[Tuple[int, str]]]:
    """Line numbers and text of every line of ``path`` containing each needle (for fact 1)."""
    out: Dict[str, List[Tuple[int, str]]] = {n: [] for n in needles}
    with open(path) as f:
        for i, line in enumerate(f, 1):
            for n in needles:
                if n in line:
                    out[n].append((i, line.rstrip("\n").strip()))
    return out


def fact1(root_v2: str, runs: Sequence[RunData]) -> Dict[str, object]:
    """Fact 1: the original rule is still in ``run_v2_stagewise.py``; v2.0 disables it; cost per call."""
    src = REPO / "run" / "run_v2_stagewise.py"
    lines = grep_lines(src, ['P["phase_thr_over_dw"]', 'P["k_phase"]', 'P["conc_thr"]', "self.fixed = ",
                             "if not self.fixed:", '"verifier_timeout"', "def timeout_for"])
    cfgs: Dict[str, set] = {k: set() for k in ("mode", "fixed_budget", "verifier_timeout_A", "phase_thr_over_dw_A",
                                               "k_phase", "conc_thr", "phase_cap_A")}
    for run in runs:
        with open(Path(root_v2) / run.label / f"q{run.q}" / f"seed{run.seed}" / "run_config.json") as f:
            c = json.load(f)
        p = c["record"]["protocol"]
        cfgs["mode"].add(c["mode"])
        cfgs["fixed_budget"].add(bool(c["fixed_budget"]))
        cfgs["verifier_timeout_A"].add(p["verifier_timeout"]["A"])
        cfgs["phase_thr_over_dw_A"].add(p["phase_thr_over_dw"]["A"])
        cfgs["k_phase"].add(p["k_phase"])
        cfgs["conc_thr"].add(p["conc_thr"])
        cfgs["phase_cap_A"].add(p["phase_caps"]["A"])
    spacing: Dict[int, int] = {}
    n_irregular_runs = 0
    secs: List[float] = []
    for run in runs:
        p = Path(root_v2) / run.label / f"q{run.q}" / f"seed{run.seed}" / "v2_checkpoints_A.csv"
        rows = read_log_rows(str(p))
        ups = [int(r["update"]) for r in rows]
        secs += [float(r["verifier_sec"]) for r in rows]
        gaps = [b - a for a, b in zip(ups[:-1], ups[1:])]
        for g in gaps:
            spacing[g] = spacing.get(g, 0) + 1
        n_irregular_runs += int(any(r["reason"] == "stability" for r in rows))
    return {"source_lines": {k: [[i, t] for i, t in v] for k, v in lines.items()},
            "run_config_values": {k: sorted(v, key=str) for k, v in cfgs.items()},
            "phase_A_call_spacing_counts": {str(k): v for k, v in sorted(spacing.items())},
            "n_runs_with_a_stability_triggered_call_in_A": n_irregular_runs,
            "verifier_sec_A": {"n": len(secs), "median": float(np.median(secs)), "mean": float(np.mean(secs)),
                               "q10": float(np.quantile(secs, 0.1)), "q90": float(np.quantile(secs, 0.9)),
                               "min": float(min(secs)), "max": float(max(secs))}}


def original_rule_on_log(log_rows: Sequence[Mapping[str, object]], thr: float, M: int = M_DEFAULT,
                         conc_limit: float = CONC_LIMIT) -> Optional[int]:
    """The original rule on a v2 Phase-A verifier log: M consecutive calls with a valid verifier,
    ``phase_criterion_value_over_dw <= thr`` and ``conc_max_std_norm <= conc_limit``; returns the update of
    the M-th consecutive eligible call (None if it never fires)."""
    consec = 0
    for r in sorted(log_rows, key=lambda x: float(x["update"])):  # type: ignore[arg-type]
        ok = (str(r["valid"]) == "True" and float(r["phase_criterion_value_over_dw"]) <= thr  # type: ignore
              and float(r["conc_max_std_norm"]) <= conc_limit)  # type: ignore[arg-type]
        consec = consec + 1 if ok else 0
        if consec >= M:
            return int(float(r["update"]))  # type: ignore[arg-type]
    return None


def fact2(root_v2: str, runs: Sequence[RunData]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]],
                                                           Dict[str, object]]:
    """Fact 2: the original rule replayed on the 60 runs' own Phase-A verifier logs."""
    per: List[Dict[str, object]] = []
    n_elig_mismatch = n_fire_mismatch = n_calls = 0
    for run in runs:
        d = Path(root_v2) / run.label / f"q{run.q}" / f"seed{run.seed}"
        log = [r for r in read_log_rows(str(d / "v2_checkpoints_A.csv"))]
        by = {int(r["update"]): r for r in log}  # type: ignore[arg-type]
        with open(d / "v2_run_summary.json") as f:
            wf = json.load(f)["would_have_fired"].get("A")
        # eligibility recomputed at 0.02 against the logged ``eligible`` column
        for r in log:
            n_calls += 1
            mine = (str(r["valid"]) == "True" and float(r["phase_criterion_value_over_dw"]) <= 0.02  # type: ignore
                    and float(r["conc_max_std_norm"]) <= CONC_LIMIT)  # type: ignore[arg-type]
            n_elig_mismatch += int(mine != (str(r["eligible"]) == "True"))
        for thr in (0.02, 0.005):
            fire = original_rule_on_log(log, thr)
            if thr == 0.02:
                n_fire_mismatch += int((fire is None) != (wf is None) or
                                       (fire is not None and wf is not None and fire != int(wf["global_update"])))
            ref = by[PHASE_A_END]
            row: Dict[str, object] = {"thr": thr, "root": run.label, "q": run.q, "seed": run.seed,
                                      "fired": fire is not None,
                                      "fire_update": float(fire) if fire is not None else float("nan"),
                                      "peak_u1600": ref["stage2_peak_rel_err_abs"],
                                      "rmse_u1600": ref["stage2_rmse_pos_over_g2_0"],
                                      "tail_u1600": ref["stage2_tail_mean_over_g2_0"]}
            for k, s in (("peak", "stage2_peak_rel_err_abs"), ("rmse", "stage2_rmse_pos_over_g2_0"),
                         ("tail", "stage2_tail_mean_over_g2_0")):
                row[f"{k}_fire"] = by[fire][s] if fire is not None else float("nan")
            per.append(row)
    cells: List[Dict[str, object]] = []
    for thr in (0.02, 0.005):
        defs: List[Tuple[str, Optional[str], Optional[int]]] = [("pooled", None, None)]
        defs += [(f"q{q}", None, q) for q in QS]
        defs += [(f"{lab}/q{q}", lab, q) for lab in DEFAULT_LABELS for q in QS]
        for name, lab, q in defs:
            sel = [p for p in per if p["thr"] == thr and (lab is None or p["root"] == lab)
                   and (q is None or p["q"] == q)]
            fired = [p for p in sel if p["fired"]]
            cell: Dict[str, object] = {"thr": thr, "cell": name, "n_runs": len(sel), "n_fire": len(fired)}
            for k in ("fire_update", "peak_fire", "rmse_fire", "tail_fire"):
                cell[f"{k}_med"], cell[f"{k}_min"], cell[f"{k}_max"] = med_rng(p[k] for p in fired)
            cell["peak_u1600_med_all"] = med_rng(p["peak_u1600"] for p in sel)[0]
            cell["peak_u1600_med_fired"] = med_rng(p["peak_u1600"] for p in fired)[0]
            cell["peak_u1600_min"], cell["peak_u1600_max"] = med_rng(p["peak_u1600"] for p in sel)[1:]
            cell["n_tail_fire_gt_0p02"] = int(sum(float(p["tail_fire"]) > GA_TAIL for p in fired))
            cell["n_peak_fire_gt_peak_u1600"] = int(sum(float(p["peak_fire"]) > float(p["peak_u1600"])
                                                        for p in fired))
            cell["paired_peak_fire_minus_u1600_med"] = med_rng(float(p["peak_fire"]) - float(p["peak_u1600"])
                                                               for p in fired)[0]
            cells.append(cell)
    # consistency: the same rule on the EXPORTS at K = 100 (local updates 100, 200, ...) vs the logged calls
    n_cmp = n_eq = 0
    differ: List[str] = []
    for run in runs:
        for thr in (0.02, 0.005):
            lg = [p for p in per if p["thr"] == thr and p["root"] == run.label and p["q"] == run.q
                  and p["seed"] == run.seed][0]
            f_exp = fire_export(run_diags(run.a_dev), original_rule(thr, K=100), 0)
            f_log = int(lg["fire_update"]) if lg["fired"] else None
            n_cmp += 1
            n_eq += int(f_exp == f_log)
            if f_exp != f_log:
                differ.append(f"{run.key} thr={thr}: exports K=100 {f_exp} vs log {f_log}")
    return per, cells, {"n_calls_checked": n_calls, "n_eligible_mismatch_vs_logged_column": n_elig_mismatch,
                        "n_fire_mismatch_vs_would_have_fired": n_fire_mismatch,
                        "n_run_thr_pairs_export_K100_vs_log": n_cmp, "n_equal": n_eq, "differing": differ}


def fact3(root_v2: str, protocol: Mapping[str, Any]) -> Dict[str, object]:
    """Fact 3: confirmation_v2_0 q=50 seed 30501, end of Phase A: policy vs one-step best response at d = 0."""
    from scipy.optimize import minimize_scalar
    from utils.theory_multistage import F_xi
    ref = RunRef("confirmation_v2_0", str(Path(root_v2) / "confirmation_v2_0"), 50, 30501)
    rec = protocol["records"]["50"]
    spec = GameSpec(**rec["game"])
    with open(ref.run_dir / "gates.json") as f:
        g = json.load(f)
    rep = g["reported"]["end_of_A"]["final"]
    net = build_actor(load_export(str(ref.run_dir / "weights" / f"u{PHASE_A_END:05d}.npz")),
                      int(rec["ppo"]["hidden"]), float(rec["ppo"]["c_min"]), float(rec["ppo"]["mu_clamp"]))
    out: Dict[str, object] = {"run": ref.key, "gates_json_e2_at_0": rep["e2_at_0"],
                              "gates_json_g2_at_0": rep["g2_at_0"],
                              "gates_json_peak_rel_err_signed": rep["stage2_peak_rel_err_signed"],
                              "gates_json_eta_final": g["metric_values"]["eta_final"],
                              "gates_json_eta_dev": g["metric_values"]["eta_dev"], "dw": spec.dw, "k": spec.k}
    for tier, cfg in (("dev", DEV_CONFIG), ("final", FINAL_CONFIG)):
        rp = replay_candidate(spec, {1: net, 2: net}, 2, cfg, StartSampler(spec, BIN_WIDTH).n_bins(2))
        assert rp.ev is not None
        sr = rp.ev.res.stages[2]
        z = int(np.argmin(np.abs(sr.d_grid)))
        out[f"verifier_{tier}"] = {"e_hat_2_0": float(sr.e_hat[z]), "a_dev_2_0": float(sr.a_dev[z]),
                                   "a_dev_minus_e_hat": float(sr.a_dev[z] - sr.e_hat[z]),
                                   "delta_2_0_over_dw": float(sr.delta[z] / spec.dw),
                                   "eta_T_over_dw": float(rp.ev.scalars["eta_T_over_dw"])}
    e_hat = out["verifier_final"]["e_hat_2_0"]  # type: ignore[index]

    def q_of(a: float) -> float:
        return float(-spec.k * a * a + spec.dw * float(F_xi(np.asarray(a - e_hat), spec.q)[0]))

    grid = np.linspace(spec.e_min, spec.e_max, 100001)
    qv = -spec.k * grid ** 2 + spec.dw * F_xi(grid - e_hat, spec.q)
    j = int(np.argmax(qv))
    r = minimize_scalar(lambda a: -q_of(a), bounds=(max(spec.e_min, grid[j] - 0.01), min(spec.e_max, grid[j] + 0.01)),
                        method="bounded", options={"xatol": 1e-12})
    a_star = float(r.x)
    gain = q_of(a_star) - q_of(e_hat)
    g2 = float(rep["g2_at_0"])
    out["independent_F_xi"] = {"e_hat_2_0": e_hat, "best_response": a_star,
                               "best_response_minus_e_hat": a_star - e_hat,
                               "best_response_minus_e_hat_over_g2_0": (a_star - e_hat) / g2,
                               "gain_at_0_over_dw": gain / spec.dw,
                               "gain_over_GA_limit_0p005": gain / spec.dw / 0.005,
                               "e_hat_rel_err": (e_hat - g2) / g2}
    return out


# =================================================================================================== figures
COLORS = {50: "#0072B2", 60: "#D55E00"}


def _plt() -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100})
    return plt


def _log_ticks(ax: Any, axis: str, ticks: Sequence[float]) -> None:
    """Log axis with explicit, readable tick labels (no minor labels)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(list(ticks)))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def _save(fig: Any, path: str, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite-own-output to replace this tool's output")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def fig_scatter(runs: Sequence[RunData], t1: Table, path: str, overwrite: bool) -> None:
    """Figure 1: R, Delta and (supplementary) R0 = r_2(0)/s_2 against |peak error| (u >= 400, colour by q)."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    panels = (("R", r"$R_2$ (first-order residual, max over non-tail nodes)", 0.03,
               [0.03, 0.05, 0.1, 0.2, 0.5]),
              ("Delta", r"$\Delta_2$ (second-order, $\eta_2/\Delta W$)", 0.005, [5e-4, 1e-3, 5e-3, 1e-2]),
              ("R0", r"$R_2^{0} = r_2(0)/s_2$ (supplementary: the node $d=0$ only)", None,
               [0.01, 0.02, 0.05, 0.1]))
    for ax, (key, lab, vl, ticks) in zip(axes, panels):
        for q in QS:
            sel = [r for run in runs if run.q == q for r in run.a_dev if r["update"] >= SPEARMAN_FROM_U]
            xs = [metric_value(key, r, None) if key == "R0" else r[key] for r in sel]
            ys = [r["stage2_peak_rel_err_abs"] for r in sel]
            ax.scatter(xs, ys, s=5, alpha=0.3, color=COLORS[q], label=f"q = {q}", linewidths=0)
        if vl is not None:
            ax.axvline(vl, color="0.3", ls="--", lw=0.8)
        ax.axhline(0.05, color="0.3", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _log_ticks(ax, "x", ticks)
        _log_ticks(ax, "y", [1e-3, 1e-2, 0.05, 0.1, 0.2])
        ax.set_xlabel(lab, fontsize=7.5)
        ax.set_ylabel(r"$|$peak error$|$ at $d=0$ (closed form, reporting only)")
        rows = [r for r in t1.csv_rows if r["subset"] == "u>=400" and r["x"] == "|peak|" and r["y"] == key]
        txt = "\n".join(f"{r['group']}: Spearman {r['spearman_pooled']:.2f}" for r in rows)
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=7.5)
    axes[0].legend(frameon=False, loc="lower right", markerscale=3)
    fig.suptitle("Phase-A exports u >= 400, 60 runs (dev tier); dashed = rho 0.03 / eps 0.005, dotted = 0.05",
                 fontsize=8)
    fig.tight_layout()
    _save(fig, path, overwrite)
    plt.close(fig)


def fig_trajectories(runs: Sequence[RunData], path: str, overwrite: bool) -> None:
    """Figure 2: |peak|, R, R_tail, Delta against the update, every run (thin) and the median (thick)."""
    plt = _plt()
    spec = [("stage2_peak_rel_err_abs", r"$|$peak error$|$", [0.05]), ("R", r"$R_2$", list(RHO_GRID)),
            ("R_tail", r"$R_2^{tail}$", [TAU]), ("Delta", r"$\Delta_2$", [EPS])]
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.8), sharex=True)
    for i, q in enumerate(QS):
        sel = [run for run in runs if run.q == q]
        us = [r["update"] for r in sel[0].a_dev]
        for j, (col, lab, hl) in enumerate(spec):
            ax = axes[i][j]
            mat = np.array([[r[col] for r in run.a_dev] for run in sel], dtype=float)
            for row in mat:
                ax.plot(us, row, color=COLORS[q], alpha=0.18, lw=0.6)
            ax.plot(us, np.median(mat, axis=0), color="k", lw=1.4)
            for h in hl:
                ax.axhline(h, color="0.4", ls="--", lw=0.7)
            ax.axvspan(1200, 1600, color="0.9", zorder=0)
            ax.set_yscale("log")
            ax.set_title(f"q = {q}: {lab}", fontsize=8)
            if i == 1:
                ax.set_xlabel("global update (shaded: v2.0 LR decay)")
    fig.tight_layout()
    _save(fig, path, overwrite)
    plt.close(fig)


def fig_residual_map(t4b: Table, path: str, overwrite: bool) -> None:
    """Figure 3: per-bin first-order residual at u1600 (median and 10-90 % over runs)."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, q in zip(axes, QS):
        rows = [r for r in t4b.csv_rows if r["q"] == q]
        x = [r["center"] for r in rows]
        ax.fill_between(x, [r["rho_u1600_q10"] for r in rows], [r["rho_u1600_q90"] for r in rows],
                        color=COLORS[q], alpha=0.25, lw=0)
        ax.plot(x, [r["rho_u1600_median"] for r in rows], color=COLORS[q], lw=1.5,
                label=r"median $\rho_b$ (dev, u1600)")
        ax.plot(x, [r["rhobar_u1600_median"] for r in rows], color="k", lw=1, ls="-.",
                label=r"median EMA $\bar\rho_b$")
        for h in RHO_GRID:
            ax.axhline(h, color="0.4", ls="--", lw=0.7)
        ax.axvspan(-NEAR_TIE_HALF_WIDTH, NEAR_TIE_HALF_WIDTH, color="0.88", zorder=0)
        ax.set_xlabel("bin centre d (stage-2 gap); shaded: near-tie bins")
        ax.set_title(f"q = {q} (non-tail bins; support boundary at |d| = {2 * q})", fontsize=8)
        ax.set_yscale("log")
    axes[0].set_ylabel(r"per-bin residual $\rho_b = \max_d r_2(d)/s_2$")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, path, overwrite)
    plt.close(fig)


def fig_phase_b(runs: Sequence[RunData], path: str, overwrite: bool) -> None:
    """Figure 4: stage-1 residual against the stage-1 error (every Phase-B export; u2200 outlined)."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for q in QS:
        rows = [r for run in runs if run.q == q for r in run.b_dev]
        e = [r["stage1_rel_err_signed"] for r in rows]
        sr = [(r["e1_at_0"] - r["s"]) / r["s"] for r in rows]
        axes[0].scatter(e, sr, s=5, alpha=0.3, color=COLORS[q], linewidths=0, label=f"q = {q}")
        last = [r for r in rows if r["update"] == 2200]
        axes[0].scatter([r["stage1_rel_err_signed"] for r in last], [(r["e1_at_0"] - r["s"]) / r["s"] for r in last],
                        s=22, facecolor="none", edgecolor=COLORS[q], lw=1.0)
        axes[1].scatter([abs(x) for x in e], [r["R"] for r in rows], s=5, alpha=0.3, color=COLORS[q], linewidths=0)
        axes[1].scatter([abs(r["stage1_rel_err_signed"]) for r in last], [r["R"] for r in last], s=22,
                        facecolor="none", edgecolor=COLORS[q], lw=1.0)
    lim = 0.12
    axes[0].plot([-lim, lim], [-lim, lim], color="0.4", lw=0.7, ls="--")
    axes[0].set_xlim(-lim, lim)
    axes[0].set_ylim(-lim, lim)
    axes[0].set_xlabel("signed stage-1 error (e1 - e1*)/e1* (closed form, reporting only)")
    axes[0].set_ylabel(r"signed residual $(\hat e_1(0) - s_1)/s_1$")
    axes[0].legend(frameon=False, markerscale=3)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    _log_ticks(axes[1], "x", [1e-3, 1e-2, 0.1])
    _log_ticks(axes[1], "y", [1e-4, 1e-3, 1e-2, 0.1, 1.0])
    axes[1].set_xlabel("|stage-1 error|")
    axes[1].set_ylabel(r"$R_1 = r_1(0)/s_1$")
    fig.suptitle("Phase-B exports u1625..u2200, 60 runs (dev tier); circles = u2200", fontsize=8)
    fig.tight_layout()
    _save(fig, path, overwrite)
    plt.close(fig)


def fig_fire(t2_run: Table, path: str, overwrite: bool) -> None:
    """Figure 5: fire export of every rule (one row per rule), and |peak| at the fire export vs u1600."""
    plt = _plt()
    rng = np.random.default_rng(0)       # jitter only (display)
    names = [n for n, _ in rule_grid()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
    for i, name in enumerate(names):
        for q in QS:
            sel = [o for o in t2_run.csv_rows if o["rule"] == name and o["q"] == q]
            xs = [o["fire_update"] if o["fired"] else 1650 for o in sel]
            axes[0].scatter(xs, i + rng.uniform(-0.18, 0.18, len(xs)) + (0.2 if q == 60 else -0.2), s=9,
                            color=COLORS[q], alpha=0.75, linewidths=0, label=f"q = {q}" if i == 0 else None)
    axes[0].axvline(1200, color="0.5", ls=":", lw=0.8)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("fire export (= local update); 1650 = never fires within u1600")
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2)
    shown = [("orig eps=0.02", "o"), ("orig eps=0.005", "s"), ("new rho=0.05", "^")]
    for name, mk in shown:
        for q in QS:
            sel = [o for o in t2_run.csv_rows if o["rule"] == name and o["q"] == q and o["fired"]]
            axes[1].scatter([o["peak_u1600"] for o in sel], [o["peak_fire"] for o in sel], s=14, marker=mk,
                            facecolor="none", edgecolor=COLORS[q], lw=0.8, label=f"{name}, q={q}")
    axes[1].plot([0.02, 0.4], [0.02, 0.4], color="0.4", ls="--", lw=0.7)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    _log_ticks(axes[1], "x", [0.03, 0.05, 0.1, 0.2])
    _log_ticks(axes[1], "y", [0.03, 0.05, 0.1, 0.2])
    axes[1].set_xlabel(r"$|$peak error$|$ at u1600 (same run)")
    axes[1].set_ylabel(r"$|$peak error$|$ at the fire export")
    axes[1].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    _save(fig, path, overwrite)
    plt.close(fig)


# ============================================================================================ analyze
def emit(table: Table, out_dir: str, overwrite: bool, md_dir: Optional[str] = None) -> None:
    """Write ``<out>/tables/<name>.csv`` (results stay data-only); the markdown rendering of a table is
    written to ``<md_dir>/<name>.md`` only when ``md_dir`` is given."""
    write_csv(os.path.join(out_dir, "tables", f"{table.name}.csv"), table.csv_rows, table.csv_cols, overwrite)
    if table.md_cols and md_dir:
        path = os.path.join(md_dir, f"{table.name}.md")
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite-own-output")
        os.makedirs(md_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write(table.md())


def write_json(path: str, obj: object, overwrite: bool) -> None:
    """JSON output with the overwrite rule."""
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite-own-output")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


def fact_tables(per2: Sequence[Mapping[str, object]], cells2: Sequence[Mapping[str, object]]) -> Table:
    """Fact-2 summary as a table (original rule on the runs' own logs, calls every 100 updates)."""
    mdr = []
    for c in cells2:
        mdr.append([c["thr"], c["cell"], f"{c['n_fire']}/{c['n_runs']}",
                    f"{fnum(c['fire_update_med'], 0)} "
                    f"[{fnum(c['fire_update_min'], 0)}, {fnum(c['fire_update_max'], 0)}]",
                    f"{fnum(c['peak_fire_med'])} [{fnum(c['peak_fire_min'])}, {fnum(c['peak_fire_max'])}]",
                    f"{fnum(c['peak_u1600_med_fired'])} ({fnum(c['peak_u1600_med_all'])} all)",
                    f"{fnum(c['tail_fire_med'])} [{fnum(c['tail_fire_min'])}, {fnum(c['tail_fire_max'])}]",
                    c["n_tail_fire_gt_0p02"], f"{c['n_peak_fire_gt_peak_u1600']}/{c['n_fire']}"])
    return Table("fact2_original_rule_on_logs", "Fact 2. The original rule (3 consecutive eligible calls, calls of "
                 "the v2.0 log every 100 updates) replayed on the runs' own v2_checkpoints_A.csv",
                 list(cells2[0].keys()), list(cells2),
                 ["threshold", "cell", "fired", "fire local update med [min,max]", "|peak error| at the firing call",
                  "|peak error| at u1600 (fired runs)", "tail mean at the firing call", "runs with tail mean > 0.02",
                  "runs with |peak| at fire > |peak| at u1600"],
                 mdr)


def cmd_analyze(args: argparse.Namespace) -> int:
    """``analyze``: facts, tables 1-6, figures from the per-run CSVs written by ``replay``."""
    out = os.path.abspath(args.out)
    ov = bool(args.overwrite_own_output)
    vs_path = os.path.join(out, "validation_summary.csv")
    if not os.path.exists(vs_path):
        print("STOP: no validation_summary.csv; run `replay` first")
        return 2
    with open(vs_path, newline="") as f:
        vs = list(csv.DictReader(f))
    if not vs or any(v["gate_ok"] != "True" for v in vs):
        print("STOP: the replay did not reproduce the runs' own u1600 verifier call in every run")
        return 3
    runs = load_runs(out)
    if len(runs) != 60:
        print(f"STOP: expected 60 per-run CSVs, found {len(runs)}")
        return 2
    proto = load_protocol(Path(args.protocol))
    geos = {q: geometry(GameSpec(**proto["records"][str(q)]["game"])) for q in QS}
    tabs: List[Table] = []
    t1 = table1(runs, geos)
    t1b = table1b(runs)
    t0 = table_u1600(runs)
    t2, t2r = table2(runs)
    t2b = table2b(runs, geos)
    t2c = table2c(runs)
    t3, t3r = table3(runs, geos)
    t4, t4d, t4b, t4c = table4(runs, geos)
    t4l = table4_landing(runs)
    t4e = table4_ema(runs, geos)
    t4f = table4_locfrac(runs, geos)
    t5a, t5b, t5u, t5f = table5(runs)
    t5l = table5_lr(runs)
    t6 = table6(runs, args.root_v2)
    per2, cells2, chk2 = fact2(args.root_v2, runs)
    t_f2 = fact_tables(per2, cells2)
    t_f2r = Table("fact2_per_run", "Fact 2 (per run)", list(per2[0].keys()), per2, [], [])
    tabs += [t0, t1, t1b, t2, t2r, t2b, t2c, t3, t3r, t4, t4d, t4b, t4c, t4l, t4e, t4f, t5a, t5b, t5u, t5f,
             t5l, t6, t_f2, t_f2r]
    for t in tabs:
        emit(t, out, ov, args.md_dir)
    f1 = fact1(args.root_v2, runs)
    f3 = fact3(args.root_v2, proto)
    write_json(os.path.join(out, "facts.json"), {"fact1": f1, "fact2_checks": chk2, "fact3": f3,
                                                  "root_v2": args.root_v2}, ov)
    fd = args.fig_dir
    fig_scatter(runs, t1, os.path.join(fd, "cal_fig1_scatter_R_Delta_vs_peak.png"), ov)
    fig_trajectories(runs, os.path.join(fd, "cal_fig2_trajectories.png"), ov)
    fig_residual_map(t4b, os.path.join(fd, "cal_fig3_residual_map_u1600.png"), ov)
    fig_phase_b(runs, os.path.join(fd, "cal_fig4_phaseB_stage1.png"), ov)
    fig_fire(t2r, os.path.join(fd, "cal_fig5_fire_exports.png"), ov)
    print(f"wrote {len(tabs)} tables, facts.json and 5 figures under {out} / {fd}")
    return 0


# ------------------------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the tool."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("replay", help="replay the development verifier on every stored export")
    p.add_argument("--root-v2", required=True, help="ROOT_V2 (contains the reference roots; read only)")
    p.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS),
                   help="reference roots under ROOT_V2")
    p.add_argument("--out", required=True, help="output directory (results/ms_r1/calibration)")
    p.add_argument("--protocol", default=str(PROTOCOL))
    p.add_argument("--workers", type=int, default=8, help="processes (single-threaded each; max 8)")
    p.add_argument("--only", nargs="*", default=None, help="run keys <root>/q<q>/seed<seed> (debug)")
    p.add_argument("--overwrite-own-output", action="store_true",
                   help="replace this tool's own existing outputs under --out")
    p.set_defaults(func=cmd_replay)
    a = sub.add_parser("analyze", help="tables 1-6, the three facts and the figures from the per-run CSVs")
    a.add_argument("--root-v2", required=True, help="ROOT_V2 (read only; logs, run_config, gates)")
    a.add_argument("--out", required=True,
                   help="the --out of the replay command; tables/ and facts.json go here")
    a.add_argument("--fig-dir", required=True, help="figure directory (reports/ms/r1/figures)")
    a.add_argument("--protocol", default=str(PROTOCOL))
    a.add_argument("--md-dir", default=None,
                   help="optional: also write the markdown rendering of every table to this directory")
    a.add_argument("--overwrite-own-output", action="store_true",
                   help="replace this tool's own existing outputs")
    a.set_defaults(func=cmd_analyze)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
