#!/usr/bin/env python3
"""DG5 of round T3-R3: instrumented replay of the stage-3 phase of recorded ``bw`` runs.

The driver (:func:`replay_stage3`) does the set-up of ``run.run_ms_stagewise.run_pipeline`` (output directory, config copy,
``seed_globals``, ``MSRun``, the phase-start hook and the global-RNG check), runs ``rec = run.run_stage(3)`` and
``run.freeze_stage(3, rec, check)``, writes the histories and stops. The same driver without the instrumentation is the reference
of the passivity test.

The instrumentation (:class:`Stage3Instrument`) wraps ``agent.update`` of the live run for the stage-3 phase and the module-level
``collect_batch_v2`` of the runner. It draws no random number, changes no tensor, optimizer state or generator of the training,
and computes on detached deep copies of the actor (``torch.func`` per-sample gradients on a copy of the parameters, so no ``.grad``
of the training network is touched). Per update, on the batch at the start of the update (the network theta_old):

* regions ``in`` (|d_L| < 2q) and ``out`` (|d_L| >= 2q) of the learner's start gap, and 20-unit bins of d_L;
* per region and bin: the number of samples n_r, mean / SD of the raw advantage and of the normalised advantage A_hat (as
  ``CurriculumPPO.update`` forms it), the region gradient ``g_r = (1/N) sum_{i in r} A_hat_i grad log pi(a_i | s_i)`` over the
  actor parameters, its norm and its projection share ``<g_r, g> / |g|^2`` with ``g = sum_r g_r`` (the shares of any partition
  sum to 1; the baseline for "disproportionate" is n_r / N);
* the same for the partition of the draws by the action clamp (below 1e-6, above 1 - 1e-6, free), and the mean norm of
  ``grad log pi`` of each group;
* the actor's sensitivity ``|d mu / d input 2|`` and the tanh saturation shares over the in-band samples.

Outputs per run (``<out>/q{q}/seed{s}/``): the run files of the stage-3 phase (``weights``, ``ms_updates.csv``,
``ms_checks_stage3.csv``, ``freeze_stage3_*.npz`` ...), ``replay_updates.csv`` (one row per update), ``replay_bins.npz`` (per update
and bin arrays) and ``replay_identity.json`` (the bit-identity comparison with the original run).

Usage:
    python tools/t3/t3r3_replay.py run --orig-run <dir of the original bw run> --out <dir> [--no-instrument]
    python tools/t3/t3r3_replay.py all --t3r2-root <results/t3_r2> --out results/t3_r3/replays --workers 14
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import copy
import csv
import importlib.util
import json
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

from run import run_ms_stagewise as rms  # noqa: E402

# ``tools/t3/compare_runs.py`` loaded by path: ``tools/v2/compare_runs.py`` has the same module name and a full test session may import it first
_spec = importlib.util.spec_from_file_location("t3_compare_runs", Path(__file__).with_name("compare_runs.py"))
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)  # type: ignore[union-attr]
import t3r3_diag as dg  # noqa: E402

DEFAULT_R2 = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-audit-6f3a38/results/t3_r2")
DEFAULT_OUT = ROOT / "results" / "t3_r3" / "replays"
BIN_WIDTH = 20.0
ACTION_CLAMP = 1e-6


# ---------------------------------------------------------------------------------- the pure per-update computation
def per_sample_grads(actor: Any, states: Any, actions: Any) -> Any:
    """``(N, P)`` matrix of ``grad_theta log pi(a_i | s_i)`` (float32, the parameters in ``named_parameters`` order).

    ``actor`` is used through ``torch.func.functional_call`` on a detached copy of its parameters: no ``.grad`` is touched and no
    random number is drawn.
    """
    import torch
    from torch.func import functional_call, grad, vmap

    params = {k: v.detach() for k, v in actor.named_parameters()}

    def logp(p: Dict[str, Any], x: Any, a: Any) -> Any:
        al, be = functional_call(actor, p, (x[None, :],))
        return torch.distributions.Beta(al, be, validate_args=False).log_prob(a[None])[0]

    jac = vmap(grad(logp), in_dims=(None, 0, 0))(params, states, actions)
    return torch.cat([jac[k].reshape(states.shape[0], -1) for k in params], dim=1)


def normalised_advantage(adv_raw: Any, eps: float = 1e-8) -> Any:
    """The normalisation of ``CurriculumPPO.update`` (float32, population SD)."""
    return (adv_raw - adv_raw.mean()) / (adv_raw.std(unbiased=False) + eps)


def group_stats(labels: np.ndarray, n_groups: int, J: np.ndarray, adv_norm: np.ndarray, adv_raw: np.ndarray,
                norms: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Per-group quantities of a partition of the batch.

    Args:
        labels: Group index of every sample (``0 .. n_groups - 1``).
        n_groups: Number of groups.
        J: ``(N, P)`` per-sample gradients of ``log pi`` (float64).
        adv_norm: Normalised advantages ``A_hat`` (N,).
        adv_raw: Raw advantages (N,).
        norms: Per-sample ``|J_i|`` (computed if ``None``).

    Returns:
        Arrays over the groups: ``n``, ``adv_raw_mean``, ``adv_raw_std``, ``adv_norm_mean``, ``adv_norm_std``, ``gnorm`` (norm of the
        group gradient ``g_r = (1/N) sum A_hat_i J_i``), ``share`` (projection share ``<g_r, g> / |g|^2``), ``sample_norm_mean`` (mean
        per-sample ``|J_i|``). Empty groups give NaN moments and zero gradient.
    """
    N = J.shape[0]
    if norms is None:
        norms = np.linalg.norm(J, axis=1)
    onehot = (labels[None, :] == np.arange(n_groups)[:, None]).astype(np.float64)       # (n_groups, N)
    G = onehot @ ((adv_norm[:, None] * J) / N)                                          # (n_groups, P)
    g = G.sum(axis=0)
    gg = float(g @ g)
    out = {k: np.full(n_groups, np.nan) for k in ("adv_raw_mean", "adv_raw_std", "adv_norm_mean", "adv_norm_std", "sample_norm_mean")}
    out["n"] = np.zeros(n_groups)
    for r in range(n_groups):
        m = labels == r
        out["n"][r] = int(m.sum())
        if m.any():
            out["adv_raw_mean"][r], out["adv_raw_std"][r] = adv_raw[m].mean(), adv_raw[m].std()
            out["adv_norm_mean"][r], out["adv_norm_std"][r] = adv_norm[m].mean(), adv_norm[m].std()
            out["sample_norm_mean"][r] = norms[m].mean()
    out["gnorm"] = np.linalg.norm(G, axis=1)
    out["share"] = (G @ g) / gg if gg > 0.0 else np.full(n_groups, np.nan)
    return out


def region_labels(d_L: np.ndarray, q: float) -> np.ndarray:
    """0 = in band (``|d| < 2q``), 1 = out of band."""
    return (np.abs(np.asarray(d_L, dtype=float)) >= 2.0 * float(q)).astype(int)


def bin_labels(d_L: np.ndarray, half: float, width: float = BIN_WIDTH) -> Tuple[np.ndarray, int]:
    """Index of the ``width``-unit bin of D_3 = [-half, half] of every gap (and the number of bins)."""
    n = int(np.ceil(2.0 * half / width - 1e-9))
    edges = np.linspace(-half, half, n + 1)
    return np.clip(np.searchsorted(edges, np.asarray(d_L, dtype=float), side="right") - 1, 0, n - 1), n


def clamp_labels(actions: np.ndarray, c: float = ACTION_CLAMP) -> np.ndarray:
    """0 = stored action at the lower clamp, 1 = at the upper clamp, 2 = free (compared in float32 as stored)."""
    a = np.asarray(actions, dtype=np.float32)
    lo = a <= np.float32(c)
    hi = a >= np.float32(1.0 - c)
    return np.where(lo, 0, np.where(hi, 1, 2)).astype(int)


def update_diag(actor: Any, spec: Any, states: np.ndarray, actions: np.ndarray, advs_raw: np.ndarray) -> Dict[str, Any]:
    """Every logged quantity of one update, on the batch at theta_old (``actor`` is not modified).

    Returns:
        ``{"scalars": {...}, "bins": {name: (n_bins,) array}}``.
    """
    import torch

    st = torch.as_tensor(np.asarray(states, dtype=np.float32))
    ac = torch.as_tensor(np.asarray(actions, dtype=np.float32))
    adv_raw = torch.as_tensor(np.asarray(advs_raw, dtype=np.float32))
    adv = normalised_advantage(adv_raw)
    J = per_sample_grads(actor, st, ac).numpy().astype(np.float64)
    norms = np.linalg.norm(J, axis=1)
    advn, advr = adv.numpy().astype(np.float64), adv_raw.numpy().astype(np.float64)
    T = int(spec.T)
    d_L = st[:, 1].numpy().astype(np.float64) * ((T - 1) * spec.B)         # the update sees the float32 observation
    q = float(spec.q)
    N = int(st.shape[0])
    reg = region_labels(d_L, q)
    clamp = clamp_labels(actions)
    bins, nb = bin_labels(d_L, spec.domain_half(T))
    R = group_stats(reg, 2, J, advn, advr, norms)
    C = group_stats(clamp, 3, J, advn, advr, norms)
    Bn = group_stats(bins, nb, J, advn, advr, norms)
    g_all = (advn[:, None] * J / N).sum(axis=0)
    inb = reg == 0
    sc: Dict[str, Any] = {"N": N, "gnorm_total": float(np.linalg.norm(g_all))}
    for name, i in (("in", 0), ("out", 1)):
        for k in ("n", "adv_raw_mean", "adv_raw_std", "adv_norm_mean", "adv_norm_std", "gnorm", "share", "sample_norm_mean"):
            sc[f"{k}_{name}"] = float(R[k][i])
    for name, i in (("lo", 0), ("hi", 1), ("free", 2)):
        for k in ("n", "adv_raw_mean", "adv_raw_std", "adv_norm_mean", "adv_norm_std", "gnorm", "share", "sample_norm_mean"):
            sc[f"{k}_clamp_{name}"] = float(C[k][i])
    sc["n_clamp_lo_in"] = int(((clamp == 0) & inb).sum())
    sc["n_clamp_lo_out"] = int(((clamp == 0) & ~inb).sum())
    sc["n_clamp_hi_in"] = int(((clamp == 1) & inb).sum())
    sc["n_clamp_hi_out"] = int(((clamp == 1) & ~inb).sum())
    # actor sensitivity and saturation over the in-band samples (the actor is used as it is: theta_old)
    if inb.any():
        x = st[torch.as_tensor(inb)].clone().requires_grad_(True)
        p = dg.forward_parts(actor, x)
        sens = torch.autograd.grad(p["mu"].sum(), x)[0][:, 1].abs().detach().numpy()
        sc["sens_in_mean"] = float(sens.mean())
        sc["sens_in_max"] = float(sens.max())
        sc["sat1_in"] = float((p["pre1"].detach().abs() > dg.SAT_THR).float().mean())
        sc["sat2_in"] = float((p["pre2"].detach().abs() > dg.SAT_THR).float().mean())
    else:
        sc.update({"sens_in_mean": float("nan"), "sens_in_max": float("nan"), "sat1_in": float("nan"), "sat2_in": float("nan")})
    return {"scalars": sc, "bins": {"n": Bn["n"], "adv_raw_mean": Bn["adv_raw_mean"], "adv_raw_std": Bn["adv_raw_std"],
                                    "adv_norm_mean": Bn["adv_norm_mean"], "adv_norm_std": Bn["adv_norm_std"],
                                    "gnorm": Bn["gnorm"], "share": Bn["share"], "sample_norm_mean": Bn["sample_norm_mean"]}}


# ---------------------------------------------------------------------------------- the passive instrumentation
class Stage3Instrument:
    """Passive per-update logging of the stage-3 phase of an :class:`rms.MSRun` (see the module docstring)."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.bins: List[Dict[str, np.ndarray]] = []
        self._last_batch: Optional[Dict[str, Any]] = None
        self._orig_collect: Optional[Callable] = None
        self._orig_update: Optional[Callable] = None
        self._run: Optional[Any] = None

    def attach(self, run: Any) -> None:
        """Wrap ``run.agent.update`` and ``rms.collect_batch_v2`` (restored by :meth:`detach`)."""
        self._run = run
        self._orig_collect = rms.collect_batch_v2
        self._orig_update = run.agent.update
        inst = self

        def collect(*a: Any, **k: Any) -> Any:
            batch = inst._orig_collect(*a, **k)         # type: ignore[misc]
            inst._last_batch = batch                    # a reference only: nothing is changed
            return batch

        def update(states: Any, actions: Any, logp: Any, returns: Any, advs: Any, *a: Any, **k: Any) -> Any:
            actor_copy = copy.deepcopy(run.agent.actor)         # theta_old; a deep copy draws no random number
            actor_copy.eval()
            diag = update_diag(actor_copy, run.spec, states, actions, advs)
            row = {"update": run.global_u, "local": run.global_u, **diag["scalars"]}
            b = inst._last_batch
            if b is not None:                                   # the rollout's own clamp counters (raw draws before the clip)
                d1 = b["d1"]
                row.update({"d1_in_n": d1["d1_L_s3_in_n"], "d1_out_n": d1["d1_L_s3_out_n"], "d1_in_lo": d1["d1_L_s3_in_lo"],
                            "d1_out_lo": d1["d1_L_s3_out_lo"], "d1_in_hi": d1["d1_L_s3_in_hi"], "d1_out_hi": d1["d1_L_s3_out_hi"]})
            inst.rows.append(row)
            inst.bins.append(diag["bins"])
            return inst._orig_update(states, actions, logp, returns, advs, *a, **k)     # type: ignore[misc]

        rms.collect_batch_v2 = collect
        run.agent.update = update

    def detach(self) -> None:
        """Undo :meth:`attach`."""
        rms.collect_batch_v2 = self._orig_collect
        if self._run is not None:
            try:
                del self._run.agent.update              # the instance attribute: the class method is used again
            except AttributeError:
                pass

    def write(self, out_dir: str) -> None:
        """``replay_updates.csv`` and ``replay_bins.npz``."""
        dg.write_rows(Path(out_dir) / "replay_updates.csv", self.rows)
        arr = {k: np.stack([b[k] for b in self.bins]) for k in self.bins[0]} if self.bins else {}
        np.savez(Path(out_dir) / "replay_bins.npz", update=np.array([r["update"] for r in self.rows]), **arr)


# ---------------------------------------------------------------------------------- the driver
def replay_stage3(cfg: Dict[str, Any], out_dir: str, instrument: bool, cmd: str = "t3r3_replay") -> Dict[str, Any]:
    """Set-up of ``run_pipeline``, ``run_stage(3)``, ``freeze_stage(3)``, ``write_histories``, stop.

    Args:
        cfg: The run config (the original ``run_config.json`` of the recorded run, unchanged).
        out_dir: Output directory (created; it must not hold a previous replay).
        instrument: Attach :class:`Stage3Instrument` and write its records.
        cmd: Command string (recorded).

    Returns:
        ``{"violations": [...], "wall_sec": float, "instrument": Stage3Instrument or None}``.
    """
    seed = int(cfg["seed"])
    os.makedirs(out_dir, exist_ok=True)
    cfg_copy = os.path.join(out_dir, "run_config.json")
    if os.path.exists(os.path.join(out_dir, "ms_updates.csv")):
        raise RuntimeError(f"{out_dir} already holds a replay; refusing to overwrite")
    if not os.path.exists(cfg_copy):
        with open(cfg_copy, "w") as f:
            json.dump(cfg, f, indent=1)
    rms.seed_globals(seed)
    rng_log: Dict[str, Any] = {"seeding": {"seeds": {k: seed for k in rms.GLOBAL_RNGS}, "digests": rms.digests(rms.global_states())}}
    run = rms.MSRun(cfg, out_dir)
    ref: Dict[str, object] = {}

    def hook(t: int) -> None:
        if not ref:
            ref.update(rms.global_states())
            rng_log["reference_before_first_update"] = {"digests": rms.digests(ref)}
    run.phase_start_hook = hook

    def check(point: str) -> None:
        dgs = rms.digests(rms.global_states())
        rng_log[point] = {"digests": dgs}
        rd = rng_log["reference_before_first_update"]["digests"]
        for k in rms.GLOBAL_RNGS:
            if dgs[k] != rd[k]:
                run.violations.append({"rng": k, "point": point})

    inst = Stage3Instrument() if instrument else None
    if inst is not None:
        inst.attach(run)
    t0 = time.perf_counter()
    try:
        rec = run.run_stage(3)
        run.freeze_stage(3, rec, check)
    finally:
        if inst is not None:
            inst.detach()
    rms.write_histories(run, out_dir)
    if inst is not None:
        inst.write(out_dir)
    return {"violations": run.violations, "wall_sec": time.perf_counter() - t0, "instrument": inst}


# ---------------------------------------------------------------------------------- bit identity
def compare_with_original(orig: Path, rep: Path) -> Dict[str, Any]:
    """Bit-for-bit comparison of a replay with its original run: the stage-3 exports, the stage-3 rows of ``ms_updates.csv``
    (without ``update_wall_sec``), ``ms_checks_stage3.csv`` (without ``verifier_sec``) and ``freeze_stage3_{final,development}.npz``."""
    out: Dict[str, Any] = {"orig": str(orig), "replay": str(rep)}
    cfg = json.load(open(rep / "run_config.json"))
    every = int(cfg["record"]["protocol"]["weights_every"])
    n3 = int(cfg["pipeline"]["budgets"]["3"])
    exports = list(range(every, n3 + 1, every)) if every else []
    ok_w = True
    n_w = 0
    bad_w: List[str] = []
    for u in exports:
        fa, fb = orig / "weights" / f"u{u:05d}.npz", rep / "weights" / f"u{u:05d}.npz"
        if not (fa.is_file() and fb.is_file()):
            ok_w = False
            bad_w.append(f"u{u:05d}: missing")
            continue
        za, zb = np.load(fa), np.load(fb)
        same = za.files == zb.files and all(cr.same_array(za[k], zb[k]) for k in za.files)
        n_w += 1
        if not same:
            ok_w = False
            bad_w.append(f"u{u:05d}")
    out["exports_compared"] = n_w
    out["exports_identical"] = ok_w and n_w == len(exports)
    out["exports_bad"] = bad_w
    ra = [{k: v for k, v in r.items() if k != "update_wall_sec"} for r in cr.rows(orig / "ms_updates.csv", set()) if r["stage"] == "3"]
    rb = [{k: v for k, v in r.items() if k != "update_wall_sec"} for r in cr.rows(rep / "ms_updates.csv", set())]
    out["updates_rows"] = [len(ra), len(rb)]
    out["updates_identical"] = ra == rb
    out["checks_identical"] = cr.rows(orig / "ms_checks_stage3.csv", {"verifier_sec"}) == cr.rows(rep / "ms_checks_stage3.csv", {"verifier_sec"})
    frz = {}
    for name in ("final", "development"):
        za, zb = np.load(orig / f"freeze_stage3_{name}.npz"), np.load(rep / f"freeze_stage3_{name}.npz")
        frz[name] = bool(za.files == zb.files and all(cr.same_array(za[k], zb[k]) for k in za.files))
    out["freeze_identical"] = frz
    out["all_identical"] = bool(out["exports_identical"] and out["updates_identical"] and out["checks_identical"] and all(frz.values()))
    return out


def counts_match(inst_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Cross-check of the replay's clamp counts (stored action at the clamp) against the rollout's counters
    (``d1_L_s3_{in,out}_{lo,hi,n}``): the number of updates and samples where they differ."""
    keys = (("in", "lo"), ("out", "lo"), ("in", "hi"), ("out", "hi"))
    res = {"n_updates": len(inst_rows)}
    for reg, side in keys:
        res[f"updates_differ_{reg}_{side}"] = int(sum(1 for r in inst_rows if int(r[f"n_clamp_{side}_{reg}"]) != int(r[f"d1_{reg}_{side}"])))
        res[f"samples_differ_{reg}_{side}"] = int(sum(abs(int(r[f"n_clamp_{side}_{reg}"]) - int(r[f"d1_{reg}_{side}"])) for r in inst_rows))
    res["updates_differ_n_in"] = int(sum(1 for r in inst_rows if int(r["n_in"]) != int(r["d1_in_n"])))
    return res


# ---------------------------------------------------------------------------------- runs
def replay_one(args: Tuple[str, str, bool]) -> Dict[str, Any]:
    """Replay one recorded run (``orig`` dir) into ``out`` and compare; returns the identity record."""
    orig_s, out_s, instrument = args
    import torch

    torch.set_num_threads(1)
    orig, out = Path(orig_s), Path(out_s)
    cfg = json.load(open(orig / "run_config.json"))
    t0 = time.perf_counter()
    res = replay_stage3(cfg, str(out), instrument, cmd=f"t3r3_replay run --orig-run {orig}")
    rec = compare_with_original(orig, out)
    rec.update({"instrument": instrument, "wall_sec": time.perf_counter() - t0, "global_rng_violations": res["violations"],
                "q": int(cfg["q"]), "seed": int(cfg["seed"])})
    if instrument and res["instrument"] is not None:
        rec["clamp_count_crosscheck"] = counts_match(res["instrument"].rows)
    dg.write_json(out / "replay_identity.json", rec)
    return rec


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R3 DG5 replay", allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("run")
    p1.add_argument("--orig-run", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--no-instrument", action="store_true")
    p2 = sub.add_parser("all")
    p2.add_argument("--t3r2-root", default=str(DEFAULT_R2))
    p2.add_argument("--out", default=str(DEFAULT_OUT))
    p2.add_argument("--workers", type=int, default=14, help="at most 14")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        rec = replay_one((a.orig_run, a.out, not a.no_instrument))
        print(json.dumps({k: rec[k] for k in ("all_identical", "wall_sec")}))
        return 0 if rec["all_identical"] else 1
    if not 1 <= a.workers <= 14:
        ap.error("--workers must be in 1..14")
    root, out = Path(a.t3r2_root), Path(a.out)
    items = [(str(root / "confirmation" / f"q{q}" / f"seed{s}" / "bw"), str(out / f"q{q}" / f"seed{s}"), True)
             for q, s in dg.REPLAY_SET]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(a.workers, len(items)), mp_context=mp.get_context("spawn")) as pool:
        recs = list(pool.map(replay_one, items))
    rows = [{"q": r["q"], "seed": r["seed"], "all_identical": r["all_identical"], "exports_identical": r["exports_identical"],
             "updates_identical": r["updates_identical"], "checks_identical": r["checks_identical"],
             "freeze_final": r["freeze_identical"]["final"], "freeze_development": r["freeze_identical"]["development"],
             "wall_sec": r["wall_sec"], "rng_violations": len(r["global_rng_violations"])} for r in recs]
    dg.write_rows(out / "replay_identity.csv", rows)
    dg.write_json(out / "record_replay_all.json", {"t3r2_root": str(root), "workers": a.workers, "wall_sec": time.perf_counter() - t0,
                                                   **dg.git_head()})
    print(f"[replay all] {sum(1 for r in recs if r['all_identical'])} of {len(recs)} identical in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
