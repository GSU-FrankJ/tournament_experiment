#!/usr/bin/env python3
"""Locked v2 T=2 pipeline (protocol v2.0): Phase A (1600) -> G-A -> freeze -> Phase B (600, expected
continuation) -> G-F, G-N, G-S.

The ONLY configuration source is ``protocols/v2_T2_locked_v2_0.json``. The run refuses to start if
that file's SHA-256 differs from ``PROTOCOL_SHA256`` (the hash at the v2.0 lock commit; also checked
against the v2.0 record in ``protocols/LOCK`` once that record exists), if the continuation-table
rule recorded in the file differs from the rule implemented here and in ``utils/v2_continuation.py``,
if any command-line argument other than ``--q``, ``--seed`` and ``--out-dir`` is given, or if q is
not in the protocol's q values. Protocol v1.1 (``protocols/v2_T2_locked_v1_1.json``) is run by this
file as of its lock commit 431474d, protocol v1.0 as of 4bd2214.

Training is ``run.run_v2_stagewise.Run`` in mode ``locked`` (same loop, RNG streams and samplers as
every v2 pilot and as v1.0/v1.1). v2.0 differs from v1.1 in exactly one pipeline setting: Phase B
runs with ``continuation_value_mode=expected`` (the stage-1 return uses the table value of the
frozen stage-2 policy instead of the sampled continuation; ``utils/v2_continuation.py``, built once
at Phase-B entry by ``Run.run_phase``). Phase A is unchanged. This entry point additionally:
  - checks, at the first Phase-B update, that the table that was built has the locked rule, writes
    it to ``continuation_table.npz`` (deterministic zip) and records the rule, the build time and the
    NPZ's SHA-256 in manifest.json and gates.json; the build must not move any training RNG stream
    or global RNG (asserted: states at Phase-B entry equal the states at the first Phase-B update);
  - gates G-A (unchanged), G-F (Gmax_full), G-N (dev vs final refinement) and G-S (stage-1 error
    <= 0.05); run pass = all four. S1 at 0.10, the v1.1 outcome and the v1.0 outcome are reported
    only;
  - hardening of the process-global RNGs (torch global, numpy legacy global, Python random): seeded
    with the run seed at the start of ``run_pipeline``; reference state immediately before the first
    Phase A update; equality asserted at the end of A, after G-A, at the end of B and at the end of the
    run; a violation is recorded in gates.json, the run finishes, and the exit code is 5.
The closed form enters only the evaluation (gates, reported metrics, decomposition).

Outputs in --out-dir: manifest.json (commit, clean-tree flag, protocol SHA-256 and version, q, seed,
global-RNG seeds and digests, continuation-table record), gates.json, continuation_table.npz,
v2_checkpoints_{A,B}.csv + checkpoints/, weights/, state_end_A.pt, state_end_B.pt,
gateA_{final,development}.npz, final_{final,development}.npz, induced_band.json + band_sweep.npz,
drift_test.json, train_history.json, v2_updates.csv, v2_run_summary.json, status.json, run_config.json.

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python run/run_v2_T2_locked.py --q 50 --seed 30501 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import random
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run.run_final_dp_br import write_json  # noqa: E402
from run.run_final_dp_br_round3_dense import ConfigError  # noqa: E402
from envs.curriculum_env import GameSpec  # noqa: E402
from run import run_v2_stagewise as stagewise  # noqa: E402
from run.run_v2_stagewise import FLAG_KEYS, Run, git_state, rng_position, write_manifest  # noqa: E402
from utils import v2_continuation as vc  # noqa: E402
from utils.dp_br_verifier import VerifierConfig  # noqa: E402
from utils.theory_multistage import f_xi, g1_two_stage, g2_two_stage  # noqa: E402
from utils.v2_metrics import evaluate, induced_band, save_npz, stage1_residual_sweep, sweep_grid  # noqa: E402

PROTOCOL_PATH = ROOT / "protocols" / "v2_T2_locked_v2_0.json"
PROTOCOL_REL = "protocols/v2_T2_locked_v2_0.json"
LOCK_PATH = ROOT / "protocols" / "LOCK"
PROTOCOL_SHA256 = "6ca0a6f95ee8109aeca7c787186fb40f5fec80a09c113b880fe80d20f1d47fd3"
TABLE_FILE = "continuation_table.npz"
# the continuation-table rule that is locked (the JSON must state exactly this; utils/v2_continuation.py
# defaults and run_v2_stagewise.CONT_TABLE_STEP must equal it; the table that is built is checked against it)
LOCKED_TABLE_RULE = {"panel_width": 1.0, "nodes_per_panel": 6, "y_step": 0.05, "stage": 2, "dtype": "float64",
                     "edge_tolerance": 1e-9, "max_points_per_chunk": 1 << 19}
ALLOWED_ARGS = ("--q", "--seed", "--out-dir")
SMOOTH_NODES = 400
RNG_VIOLATION_EXIT = 5
GLOBAL_RNGS = ("torch_global", "numpy_global", "python_random")
ASSERT_POINTS = ("end_of_A", "after_G-A", "end_of_B", "end_of_run")


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lock_records(path: Path = LOCK_PATH) -> List[Dict]:
    """All JSON records in protocols/LOCK (concatenated JSON objects, v1.0 record first)."""
    if not path.exists():
        return []
    txt, dec, out, i = path.read_text(), json.JSONDecoder(), [], 0
    while True:
        while i < len(txt) and txt[i].isspace():
            i += 1
        if i >= len(txt):
            return out
        obj, i = dec.raw_decode(txt, i)
        out.append(obj)


def check_table_rule(proto: Dict) -> None:
    """Refuse a protocol whose continuation-table rule is not the locked one, or the code's.

    The rule in the JSON must equal ``LOCKED_TABLE_RULE``; ``utils/v2_continuation.py`` (the
    defaults the pipeline builds with) and ``run_v2_stagewise.CONT_TABLE_STEP`` must equal it too.
    """
    pl = proto["pipeline"]
    ct = pl.get("continuation_table")
    if not isinstance(ct, dict) or "rule" not in ct:
        raise ConfigError("protocol has no pipeline.continuation_table.rule; refusing to run")
    if pl["flags"].get("continuation_value_mode") != "expected":
        raise ConfigError(f"pipeline.flags.continuation_value_mode must be 'expected', got "
                          f"{pl['flags'].get('continuation_value_mode')!r}")
    r = ct["rule"]
    got = {"panel_width": r.get("panel_width"), "nodes_per_panel": r.get("nodes_per_panel"),
           "y_step": (r.get("y_grid") or {}).get("step"), "stage": ct.get("stage"), "dtype": r.get("dtype"),
           "edge_tolerance": r.get("edge_tolerance"), "max_points_per_chunk": r.get("max_points_per_chunk")}
    if got != LOCKED_TABLE_RULE:
        raise ConfigError(f"continuation-table rule {got} != locked {LOCKED_TABLE_RULE}; refusing to run")
    impl = {"panel_width": vc.DEFAULT_PANEL_WIDTH, "nodes_per_panel": vc.DEFAULT_NODES_PER_PANEL,
            "y_step": vc.DEFAULT_STEP, "stage": 2, "dtype": "float64",
            "edge_tolerance": vc._EDGE_TOL, "max_points_per_chunk": vc._MAX_POINTS_PER_CHUNK}
    if impl != LOCKED_TABLE_RULE or float(stagewise.CONT_TABLE_STEP) != LOCKED_TABLE_RULE["y_step"]:
        raise ConfigError(f"implemented continuation-table rule {impl} (CONT_TABLE_STEP "
                          f"{stagewise.CONT_TABLE_STEP}) != locked {LOCKED_TABLE_RULE}; refusing to run")


def load_protocol(path: Path = PROTOCOL_PATH, expected: str = PROTOCOL_SHA256, lock_path: Path = LOCK_PATH) -> Dict:
    """Load the locked protocol; refuse on any hash mismatch (file vs code, file vs its LOCK record)
    or on a continuation-table rule that is not the locked one."""
    actual = sha256_file(path)
    if actual != expected:
        raise ConfigError(f"protocol {path} sha256 {actual} != locked {expected}; refusing to run")
    mine = [r for r in lock_records(lock_path) if r.get("protocol") == PROTOCOL_REL]
    if mine and mine[-1].get("protocol_sha256") != actual:
        raise ConfigError(f"protocols/LOCK records sha256 {mine[-1].get('protocol_sha256')} for {PROTOCOL_REL} != {actual}")
    proto = json.load(open(path))
    check_table_rule(proto)
    return proto


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Accept exactly --q, --seed, --out-dir; refuse anything else."""
    p = argparse.ArgumentParser(description="locked v2 T=2 pipeline (v2.0)", allow_abbrev=False)
    p.add_argument("--q", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    a, extra = p.parse_known_args(argv)
    if extra:
        raise ConfigError(f"refusing overrides {extra}: only {ALLOWED_ARGS} are accepted")
    return a


def build_config(proto: Dict, q: int, seed: int, out_dir: str) -> Dict:
    """v2 run config for one (q, seed), taken entirely from the protocol."""
    if q not in proto["q_values"]:
        raise ConfigError(f"q={q} not in the protocol's q_values {proto['q_values']}")
    rec = copy.deepcopy(proto["records"][str(q)])
    run_name = f"v2T2locked_q{q}_s{seed}"
    rec.update(seed=int(seed), run=run_name, output_dir=out_dir)
    pl = proto["pipeline"]
    # the run config keeps the four stagewise flags in "flags"; continuation_value_mode is its own key
    return {"schema": "v2_run_config/1", "base_commit": proto["base_commit"], "pilot": "v2_T2_locked", "arm": "locked",
            "run": run_name, "q": q, "seed": int(seed), "mode": pl["mode"], "fixed_budget": bool(pl["fixed_budget"]),
            "flags": {k: pl["flags"][k] for k in FLAG_KEYS},
            "continuation_value_mode": pl["flags"]["continuation_value_mode"],
            "parent_checkpoint": None, "parent_sha256": None, "record": rec,
            "threads_per_process": proto["threads_per_process"], "budget_overrides": {},
            "lr_decay": copy.deepcopy(pl["lr_decay"]), "full_state_at": []}


# ---------------------------------------------------------------------------- process-global RNGs (D5)
def seed_globals(seed: int) -> None:
    """Seed the three process-global RNGs with the run seed."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))


def global_states() -> Dict[str, object]:
    """Current states of the three process-global RNGs."""
    return {"torch_global": torch.get_rng_state().clone(), "numpy_global": np.random.get_state(),
            "python_random": random.getstate()}


def _state_bytes(name: str, st) -> bytes:
    if name == "torch_global":
        return st.numpy().tobytes()
    if name == "numpy_global":
        kind, key, pos, has_gauss, cached = st
        return repr((kind, pos, has_gauss, float(cached))).encode() + np.asarray(key).tobytes()
    return repr(st).encode()


def digests(states: Dict[str, object]) -> Dict[str, str]:
    """SHA-256 of each global RNG state."""
    return {k: hashlib.sha256(_state_bytes(k, v)).hexdigest() for k, v in states.items()}


# ---------------------------------------------------------------------------- verdicts (D2, D3, D4)
def verdicts(v: Dict[str, float], proto: Dict) -> Dict[str, object]:
    """Gate verdicts from metric values (inclusive <=, full precision).

    ``v`` keys: eta_final, eta_dev, rmse, tail, gmax_final, gmax_dev, s1 (all /DW or relative).
    v2.0 (the protocol has a gate ``G-S``): run pass = G-A and G-F and G-N and G-S, where G-S is
    the stage-1 error ``s1`` against its own threshold. S1 (at the secondary threshold 0.10), the
    v1.1 outcome (G-A and G-F and G-N) and the v1.0 outcome are reported only. A protocol without
    ``G-S`` (v1.1, used by ``tools/v2/refine_analysis.py``) gets the v1.1 verdicts unchanged.
    """
    g = proto["gates"]
    ta = {c["metric"]: c["threshold"] for c in g["G-A"]["all_must_hold"]}
    tf = {c["metric"]: c["threshold"] for c in g["G-F"]["all_must_hold"]}
    tn = {c["metric"]: c["threshold"] for c in g["G-N"]["all_must_hold"]}
    t1 = proto["secondary"]["S1"]["criterion"]["threshold"]
    GA = [("eta_T_over_dw", v["eta_final"], ta["eta_T_over_dw"]),
          ("stage2_rmse_pos_over_g2_0", v["rmse"], ta["stage2_rmse_pos_over_g2_0"]),
          ("stage2_tail_mean_over_g2_0", v["tail"], ta["stage2_tail_mean_over_g2_0"])]
    GF = [("Gmax_full_over_dw", v["gmax_final"], tf["Gmax_full_over_dw"])]
    GN = [("eta_T_over_dw_dev_minus_final_abs", abs(v["eta_dev"] - v["eta_final"]), tn["eta_T_over_dw_dev_minus_final_abs"]),
          ("Gmax_full_over_dw_dev_minus_final_abs", abs(v["gmax_dev"] - v["gmax_final"]), tn["Gmax_full_over_dw_dev_minus_final_abs"])]

    def block(rows):
        crit = [{"metric": m, "value": float(x), "threshold": t, "pass": bool(x <= t)} for m, x, t in rows]
        return {"criteria": crit, "pass": all(c["pass"] for c in crit)}
    out = {"G-A": block(GA), "G-F": block(GF), "G-N": block(GN)}
    p_v11 = bool(out["G-A"]["pass"] and out["G-F"]["pass"] and out["G-N"]["pass"])
    out["v1_1_outcome"] = {"G-A": out["G-A"]["pass"], "G-F": out["G-F"]["pass"], "G-N": out["G-N"]["pass"],
                           "run_pass_v1_1": p_v11}
    if "G-S" in g:
        ts = {c["metric"]: c["threshold"] for c in g["G-S"]["all_must_hold"]}
        out["G-S"] = block([("stage1_rel_err_abs", v["s1"], ts["stage1_rel_err_abs"])])
        out["run_pass"] = bool(p_v11 and out["G-S"]["pass"])
    else:
        out["run_pass"] = p_v11
    s1 = bool(v["s1"] <= t1)
    out["S1"] = {"metric": "stage1_rel_err_abs", "value": float(v["s1"]), "threshold": t1, "pass": s1}
    gf10 = bool(out["G-F"]["pass"] and s1)
    out["v1_0_outcome"] = {"G-A": out["G-A"]["pass"], "G-F_v1_0": gf10, "run_pass_v1_0": bool(out["G-A"]["pass"] and gf10)}
    out["outcome"] = ("pass" if out["run_pass"] else "stage2_failure" if not out["G-A"]["pass"]
                      else "fail_G-F" if not out["G-F"]["pass"] else "fail_G-N" if not out["G-N"]["pass"]
                      else "fail_G-S")
    return out


# ---------------------------------------------------------------------------- continuation table (v2.0)
def training_rng_digest(run: Run) -> str:
    """SHA-256 over the five training streams (env, learn, opp, start, minibatch) and the torch generator."""
    h = hashlib.sha256()
    for k in sorted(run.rngs):
        h.update(k.encode() + json.dumps(run.rngs[k].bit_generator.state, sort_keys=True).encode())
    h.update(b"minibatch" + json.dumps(run.agent.rng_mb.bit_generator.state, sort_keys=True).encode())
    h.update(b"torch_generator" + run.torch_gen.get_state().numpy().tobytes())
    return h.hexdigest()


def rng_snapshot(run: Run) -> Dict[str, str]:
    """Digest of every RNG a table build must not touch: the training RNGs and the three globals."""
    return {"training": training_rng_digest(run), **digests(global_states())}


def write_table_npz(path: str, table: vc.ContinuationTable) -> str:
    """Write ``y_grid`` and ``values`` as an NPZ with fixed zip timestamps; return its SHA-256.

    The zip members carry a fixed date, so the same table always yields the same file bytes and
    the file hash is a function of the table alone, whatever numpy's own zip writer does.
    ``np.load`` reads the file normally.
    """
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        for name, arr in (("y_grid", table.y_grid), ("values", table.values)):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arr), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            z.writestr(info, buf.getvalue())
    return sha256_file(Path(path))


def check_built_table(table: vc.ContinuationTable, spec: GameSpec) -> Dict[str, object]:
    """Refuse a table that was not built with the locked rule; return its rule record."""
    m = table.meta
    n_half = int(np.ceil(spec.e_range / LOCKED_TABLE_RULE["y_step"] - 1e-9))
    want = {"stage": LOCKED_TABLE_RULE["stage"], "step_requested": LOCKED_TABLE_RULE["y_step"],
            "panel_width": LOCKED_TABLE_RULE["panel_width"], "nodes_per_panel": LOCKED_TABLE_RULE["nodes_per_panel"],
            "n_y": 2 * n_half + 1, "q": float(spec.q), "conc_scale": 1.0}
    got = {k: m.get(k) for k in want}
    if got != want or table.values.dtype != np.float64 or table.y_grid.dtype != np.float64:
        raise ConfigError(f"built continuation table {got} ({table.values.dtype}) != locked rule {want}")
    if abs(float(m["weight_sum"]) - 1.0) > 1e-13:
        raise ConfigError(f"continuation-table quadrature weights sum to {m['weight_sum']}")
    keys = ("stage", "step_requested", "step_actual", "n_y", "panel_width", "nodes_per_panel", "n_panels", "n_nodes",
            "weight_sum", "conc_scale", "q", "e_min", "e_max")
    return {k: m[k] for k in keys}


# ---------------------------------------------------------------------------- evaluation helpers
def smoothed_share(run: Run, ev, net=None) -> Dict[str, float]:
    """Share of the d=0 peak gap predicted by action-noise smoothing (Pilot-1 method)."""
    from scipy.stats import beta as beta_dist
    from run.run_final_dp_br import make_policy_fns
    spec = run.spec
    mf, bf = make_policy_fns(run.agent, spec, net=net)
    a, b = bf(2, np.zeros(1))
    u = (np.arange(SMOOTH_NODES) + 0.5) / SMOOTH_NODES
    x = spec.e_range * (beta_dist.ppf(u, a[0], b[0]) - a[0] / (a[0] + b[0]))
    pred0 = spec.dw / (2 * spec.k) * f_xi(x[:, None] - x[None, :], spec.q).mean()
    g20 = float(g2_two_stage(np.zeros(1), spec.q, spec.w_h, spec.w_l, spec.k, spec.e_max)[0])
    e0 = float(mf(2, np.zeros(1))[0])
    return {"smoothed_e_pred_0": float(pred0), "smoothed_e_learned_0": e0,
            "smoothed_share_peak_gap_d0": float((g20 - pred0) / (g20 - e0)) if g20 != e0 else float("nan")}


def stage2_extra(ev) -> Dict[str, float]:
    """Location-free peak error on the recovery grid."""
    D, e2 = ev.arrays["recovery_d_grid"], ev.arrays["recovery_e2"]
    g20 = float(ev.scalars["g2_at_0"])
    j = int(np.argmax(e2))
    return {"stage2_peak_locfree_rel_err": (float(e2[j]) - g20) / g20, "stage2_peak_locfree_argmax_d": float(D[j])}


def decomposition(run: Run, e1: float, proto: Dict, fin: VerifierConfig, step: Optional[float] = None) -> Dict[str, object]:
    """Residual-band decomposition of the stage-1 error against the frozen stage 2 (final tier)."""
    spec = run.spec
    ds = proto["stage1_decomposition"]
    step = float(ds["step"]) if step is None else step
    g1 = g1_two_stage(spec.q, spec.w_h, spec.w_l, spec.k)

    def eq(t, d):
        d = np.asarray(d, dtype=float)
        return np.full(d.shape, g1) if t == 1 else g2_two_stage(d, spec.q, spec.w_h, spec.w_l, spec.k, spec.e_max)
    cal = float(stage1_residual_sweep(eq, spec, fin, np.array([g1]))[0])
    floor = max(cal, 1e-12 * spec.dw)
    mean_fn, _ = run.policy_fns()
    lo, hi = min(0.4 * g1, e1 - 2.0), max(1.7 * g1, e1 + 2.0)
    E = sweep_grid(g1, lo, hi, step)
    D = stage1_residual_sweep(mean_fn, spec, fin, E)
    np.savez(os.path.join(run.out_dir, "band_sweep.npz"), e_sweep=E, delta1=D)
    b = induced_band(E, D, floor)
    et, blo, bhi = b["e_tilde"], b["band_lo"], b["band_hi"]
    out = {**b, "g1": g1, "e1_at_0": e1, "calibration_delta1": cal, "step": step,
           "learning_rel": (e1 - et) / g1, "learning_rel_lo": (e1 - bhi) / g1, "learning_rel_hi": (e1 - blo) / g1,
           "inherited_rel": (et - g1) / g1, "inherited_rel_lo": (blo - g1) / g1, "inherited_rel_hi": (bhi - g1) / g1,
           "e1_inside_sweep": bool(E[0] <= e1 <= E[-1])}
    out["learning_contains_0"] = bool(out["learning_rel_lo"] <= 0 <= out["learning_rel_hi"])
    out["inherited_contains_0"] = bool(out["inherited_rel_lo"] <= 0 <= out["inherited_rel_hi"])
    return out


REPORT_KEYS_A = ("stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_sym_err_max", "stage2_tail_max",
                 "stage2_tail_max_over_g2_0", "stage2_tail_mean", "DeltaT_over_dw_on_max", "DeltaT_over_dw_off_max",
                 "DeltaT_over_dw_on_mean_cellmass_weighted", "sigma_effort_at_0_t2", "e2_at_0", "g2_at_0",
                 "stage2_rmse_pos_over_g2_0", "stage2_tail_mean_over_g2_0", "eta_T_over_dw", "valid")
REPORT_KEYS_F = ("Gmax_full_over_dw", "Gmax_full_t", "Gmax_full_d", "EXP_root_over_dw", "dReach_over_dw",
                 "Deltamax_all_over_dw", "dFull_over_dw", "stage1_rel_err_signed", "stage1_rel_err_abs", "e1_at_0", "g1",
                 "sigma_effort_at_0_t1", "eta_T_over_dw", "valid")


def pick(sc: Dict, keys) -> Dict[str, object]:
    return {k: sc[k] for k in keys if k in sc}


def diff(dev: Dict, fin: Dict) -> Dict[str, float]:
    return {k: float(dev[k]) - float(fin[k]) for k in fin
            if k in dev and isinstance(fin[k], (int, float)) and not isinstance(fin[k], bool)}


# ---------------------------------------------------------------------------- pipeline
def run_pipeline(cfg: Dict, proto: Dict, proto_sha: str, out_dir: str, cmd: str,
                 band_step: Optional[float] = None) -> int:
    """Seed globals -> build Run -> Phase A -> G-A -> freeze -> Phase B -> G-F, G-N -> final evaluation.

    Returns 0, 1 (pipeline exception) or RNG_VIOLATION_EXIT (global-RNG assertion violated).
    """
    seed = int(cfg["seed"])
    seed_globals(seed)                                   # before any object is constructed (D5)
    rng_log: Dict[str, Dict] = {"seeding": {"seeds": {k: seed for k in GLOBAL_RNGS}, "digests": digests(global_states())}}
    run = Run(cfg, out_dir)
    rng_log["after_run_construction"] = {"digests": digests(global_states())}
    ref: Dict[str, object] = {}
    violations: List[Dict[str, str]] = []
    box: Dict[str, Dict[str, str]] = {}                  # RNG snapshot taken at Phase-B entry
    tbl_rec: Dict[str, object] = {}                      # continuation-table record (manifest, gates)

    def hook(phase: str) -> None:
        if phase == "A" and not ref:
            ref.update(global_states())
            rng_log["reference_before_first_A_update"] = {"digests": digests(ref)}
        elif phase == "B":
            rng_log["before_first_B_update"] = {"digests": digests(global_states())}
            # Run.run_phase built the continuation table between Phase-B entry and this call: it must
            # not have drawn from any RNG; then check it has the locked rule and write it out
            before, after = box["before_B"], rng_snapshot(run)
            if after["training"] != before["training"]:
                raise RuntimeError("a training RNG stream moved between Phase-B entry and the first Phase-B "
                                   "update (the continuation-table build must not draw)")
            for k in GLOBAL_RNGS:
                if after[k] != before[k]:
                    violations.append({"rng": k, "point": "table_build"})
            tbl = run.cont_table
            if tbl is None:
                raise RuntimeError("continuation_value_mode=expected but no table was built at Phase-B entry")
            rule = check_built_table(tbl, run.spec)
            tbl_rec.update({"file": TABLE_FILE, "npz_sha256": write_table_npz(os.path.join(out_dir, TABLE_FILE), tbl),
                            "rule": rule, "build_seconds": float(tbl.meta["build_seconds"]),
                            "training_rng_digest_at_phase_B_entry": before["training"],
                            "rng_unchanged_by_build": {"training": True, **{k: after[k] == before[k] for k in GLOBAL_RNGS}}})
            man["continuation_table"] = tbl_rec
            write_json(os.path.join(out_dir, "manifest.json"), man)
    run.phase_start_hook = hook

    def check(point: str) -> None:
        now = global_states()
        dg = digests(now)
        rng_log[point] = {"digests": dg}
        rd = rng_log["reference_before_first_A_update"]["digests"]
        for k in GLOBAL_RNGS:
            if dg[k] != rd[k]:
                violations.append({"rng": k, "point": point})

    t0 = time.perf_counter()
    status = {"run": cfg["run"], "q": run.spec.q, "seed": run.seed, "mode": run.mode, "state": "running",
              "pid": os.getpid(), "cmd": cmd, "start_time": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_state()}
    write_json(os.path.join(out_dir, "status.json"), status)
    man = write_manifest(run, cfg, out_dir, cmd)
    g = man["git"]
    man.update({"locked_protocol": {"path": PROTOCOL_REL, "sha256": proto_sha, "version": proto["version"]},
                "protocol_version": proto["version"], "commit": g["commit"], "clean_tree": g["dirty"] is False,
                "q": run.spec.q, "seed": run.seed, "global_rng": rng_log})
    write_json(os.path.join(out_dir, "manifest.json"), man)
    try:
        # ---- Phase A
        run.run_phase("A")
        torch.save(run.full_state("A"), os.path.join(out_dir, "state_end_A.pt"))
        check("end_of_A")
        evA = {}
        for tier in (run.fin_cfg, run.dev_cfg):
            mf, bf = run.policy_fns()
            ev = evaluate(mf, run.spec, tier, beta_fn=bf, recovery_step=float(run.P["recovery_step"]))
            save_npz(ev, os.path.join(out_dir, f"gateA_{tier.name}.npz"))
            evA[tier.name] = ev
        scA = {t: {**e.scalars, **stage2_extra(e)} for t, e in evA.items()}
        smooth = smoothed_share(run, evA["final"])
        check("after_G-A")
        # ---- freeze stage 2 (B2 machinery)
        run.parent_ref = run.stage2_mapping()
        parent_actor = {k: v.clone() for k, v in run.agent.actor.state_dict().items()}
        run.agent.freeze_stage2_snapshot()
        # ---- Phase B (expected continuation: the table is built inside run_phase at Phase-B entry)
        box["before_B"] = rng_snapshot(run)
        rng_log["before_B_entry"] = {"digests": digests(global_states())}
        run.run_phase("B")
        torch.save(run.full_state("B"), os.path.join(out_dir, "state_end_B.pt"))
        check("end_of_B")
        run.agent.export_weights_npz(os.path.join(out_dir, "checkpoint_weights.npz"))
        evF = {}
        for tier in (run.fin_cfg, run.dev_cfg):
            mf, bf = run.policy_fns()
            ev = evaluate(mf, run.spec, tier, beta_fn=bf, recovery_step=float(run.P["recovery_step"]))
            save_npz(ev, os.path.join(out_dir, f"final_{tier.name}.npz"))
            evF[tier.name] = ev
        scF = {t: {**e.scalars, **stage2_extra(e)} for t, e in evF.items()}
        # ---- snapshot integrity (C5 drift test)
        now = run.stage2_mapping(run.agent.frozen)
        dt_ = {k: float(np.max(np.abs(now[k] - run.parent_ref[k]))) for k in ("mean", "alpha", "beta")}
        fz = run.agent.frozen.state_dict()
        drift = {"max_abs_diff_vs_freeze_time": dt_,
                 "snapshot_params_bit_identical_to_end_of_A_actor": bool(all(torch.equal(fz[k], parent_actor[k]) for k in parent_actor))}
        drift["pass"] = bool(all(v == 0.0 for v in dt_.values()) and drift["snapshot_params_bit_identical_to_end_of_A_actor"])
        write_json(os.path.join(out_dir, "drift_test.json"), drift)
        # ---- stage-1 decomposition (final tier, frozen stage 2)
        dec = decomposition(run, float(scF["final"]["e1_at_0"]), proto, run.fin_cfg, band_step)
        write_json(os.path.join(out_dir, "induced_band.json"), dec)
        check("end_of_run")
        vals = {"eta_final": float(scA["final"]["eta_T_over_dw"]), "eta_dev": float(scA["development"]["eta_T_over_dw"]),
                "rmse": float(scA["final"]["stage2_rmse_pos_over_g2_0"]), "tail": float(scA["final"]["stage2_tail_mean_over_g2_0"]),
                "gmax_final": float(scF["final"]["Gmax_full_over_dw"]), "gmax_dev": float(scF["development"]["Gmax_full_over_dw"]),
                "s1": float(scF["final"]["stage1_rel_err_abs"])}
        V = verdicts(vals, proto)
        rng_ok = not violations
        run_pass = bool(V["run_pass"] and rng_ok)
        gates = {"protocol_version": proto["version"], "q": run.spec.q, "seed": run.seed, "protocol_sha256": proto_sha,
                 "commit": g["commit"], "clean_tree": g["dirty"] is False, "metric_values": vals,
                 "dev_tier_values": {"G-A": {"eta_T_over_dw": vals["eta_dev"],
                                             "stage2_rmse_pos_over_g2_0": float(scA["development"]["stage2_rmse_pos_over_g2_0"]),
                                             "stage2_tail_mean_over_g2_0": float(scA["development"]["stage2_tail_mean_over_g2_0"])},
                                     "G-F": {"Gmax_full_over_dw": vals["gmax_dev"]},
                                     "G-S": {"stage1_rel_err_abs": float(scF["development"]["stage1_rel_err_abs"])},
                                     "S1": {"stage1_rel_err_abs": float(scF["development"]["stage1_rel_err_abs"])}},
                 "G-A": V["G-A"], "G-F": V["G-F"], "G-N": V["G-N"], "G-S": V["G-S"], "S1": V["S1"],
                 "v1_1_outcome": V["v1_1_outcome"], "v1_0_outcome": V["v1_0_outcome"],
                 "continuation_table": tbl_rec,
                 "global_rng": {"status": "ok" if rng_ok else "violation", "violations": violations,
                                "assertion_points": list(ASSERT_POINTS) + ["table_build"]},
                 "gates_pass": V["run_pass"], "run_pass": run_pass,
                 "outcome": V["outcome"] if rng_ok else "global_rng_violation",
                 "reported": {
                     "end_of_A": {"final": pick(scA["final"], REPORT_KEYS_A + ("stage2_peak_locfree_rel_err", "stage2_peak_locfree_argmax_d")),
                                  "development": pick(scA["development"], REPORT_KEYS_A + ("stage2_peak_locfree_rel_err", "stage2_peak_locfree_argmax_d")),
                                  "dev_minus_final": diff(pick(scA["development"], REPORT_KEYS_A + ("stage2_peak_locfree_rel_err",)),
                                                          pick(scA["final"], REPORT_KEYS_A + ("stage2_peak_locfree_rel_err",))),
                                  "smoothed_game": smooth},
                     "end_of_B": {"final": pick(scF["final"], REPORT_KEYS_F), "development": pick(scF["development"], REPORT_KEYS_F),
                                  "dev_minus_final": diff(pick(scF["development"], REPORT_KEYS_F), pick(scF["final"], REPORT_KEYS_F)),
                                  "decomposition": dec},
                     "drift_test_pass": drift["pass"]},
                 "lr_last": {"A": run.lr_for("A", int(run.P["phase_caps"]["A"])), "B": run.lr_for("B", int(run.P["phase_caps"]["B"]))}}
        write_json(os.path.join(out_dir, "gates.json"), gates)
        write_json(os.path.join(out_dir, "train_history.json"),
                   {"run": cfg["run"], "group": cfg["arm"], "history": run.history, "stability": run.stability_log,
                    "verifier_calls": run.verifier_log, "curriculum": run.curriculum_log, "snapshots": run.snapshot_log,
                    "weight_checkpoints": run.weights_log, "stopping_record": None})
        write_json(os.path.join(out_dir, "v2_run_summary.json"),
                   {"phase_timing": run.phase_timing, "would_have_fired": run.would_fire, "fixed_budget": run.fixed,
                    "costs": run.costs.as_dict(), "costs_v2": run.costs_v2.as_dict(), "phases_done": run.phases_done,
                    "final_global_update": run.global_u, "total_wall_sec": time.perf_counter() - t0})
        with open(os.path.join(out_dir, "v2_updates.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(run.v2_history[0].keys()))
            w.writeheader()
            w.writerows(run.v2_history)
        man["global_rng"] = rng_log
        man["global_rng_status"] = gates["global_rng"]
        write_json(os.path.join(out_dir, "manifest.json"), man)
        code = 0 if rng_ok else RNG_VIOLATION_EXIT
        status.update({"state": "done", "end_time": time.strftime("%Y-%m-%d %H:%M:%S"), "exit_code": code,
                       "final_global_update": run.global_u, "run_pass": run_pass, "outcome": gates["outcome"],
                       "total_wall_sec": time.perf_counter() - t0})
        write_json(os.path.join(out_dir, "status.json"), status)
        print(f"[done] q={run.spec.q:g} seed={run.seed} G-A={V['G-A']['pass']} G-F={V['G-F']['pass']} G-N={V['G-N']['pass']} "
              f"G-S={V['G-S']['pass']} global_rng={'ok' if rng_ok else violations} outcome={gates['outcome']} "
              f"table_sha256={str(tbl_rec.get('npz_sha256'))[:12]} wall={time.perf_counter() - t0:.1f}s",
              flush=True)
        return code
    except Exception:
        man["global_rng"] = rng_log
        write_json(os.path.join(out_dir, "manifest.json"), man)
        status.update({"state": "failed", "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "traceback": traceback.format_exc(), "updates_completed": run.global_u, "exit_code": 1})
        write_json(os.path.join(out_dir, "status.json"), status)
        print(traceback.format_exc(), flush=True)
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry (refuses any override other than q, seed, out-dir)."""
    a = parse_args(sys.argv[1:] if argv is None else argv)
    proto = load_protocol()
    out = a.out_dir
    if os.path.exists(os.path.join(out, "status.json")):
        raise ConfigError(f"{out} already holds status.json; not restarting")
    cfg = build_config(proto, a.q, a.seed, out)
    os.makedirs(out, exist_ok=True)
    json.dump(cfg, open(os.path.join(out, "run_config.json"), "w"), indent=1)
    return run_pipeline(cfg, proto, sha256_file(PROTOCOL_PATH), out, " ".join(sys.argv))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as exc:
        print(f"[config-error] {exc}", flush=True)
        sys.exit(4)
