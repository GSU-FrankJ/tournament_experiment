"""Shared helpers of the T3-R1 tests (reduced-budget configs, run caches)."""

from __future__ import annotations

import atexit
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "ms", ROOT / "tools" / "t3", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import ms_configs as mc  # noqa: E402
import t3_configs as tc  # noqa: E402

PROTO = tc.load_protocol()
BW_PARAMS = tc.load_bw_params()
MS_PARAMS_PATH = ROOT / "reports" / "ms" / "r1" / "prereg_parameters.json"

ROOT_ONLY_T3 = {"root": 48, "stage2": 0, "stage3": 0}
ES_MIX_T3 = {"root": 24, "stage2": 8, "stage3": 16}
ROOT_ONLY_T2 = {"root": 48, "stage2": 0}
ES_MIX_T2 = {"root": 32, "stage2": 16}


def joint_cfg(out: str, arm: str = "jt", q: int = 50, seed: int = 50500, U: int = 30, epu: int = 48,
              starts: Optional[Dict[str, int]] = None, check_every: int = 10, window: bool = True) -> Dict[str, Any]:
    """A reduced-budget ``joint_run_config/1`` (the production record with a lowered episode count; the
    exports are written at the check updates)."""
    cfg = tc.build_joint_config(PROTO, q, seed, arm, out)
    T = cfg["joint"]["T"]
    cfg["record"]["protocol"]["episodes_per_update"] = epu
    cfg["record"]["protocol"]["weights_every"] = check_every
    if starts is None:
        starts = {"root": epu, "stage2": 0, "stage3": 0}
    keys = ("root", "stage2") if T == 2 else ("root", "stage2", "stage3")
    cfg["joint"]["starts"] = {k: int(starts.get(k, 0)) for k in keys}
    cfg["joint"]["updates"] = U
    cfg["joint"]["check_every"] = check_every
    cfg["joint"]["lr_windows"] = ([{"first": U // 2 + 1, "last": U, "start": 3e-4, "end": 3e-5}]
                                  if window else [])
    return cfg


def bw_cfg(out: str, q: int = 50, seed: int = 50500, budgets=(40, 40, 20), epu: int = 48) -> Dict[str, Any]:
    """A reduced-budget ``bw`` config: budgets {3, 2, 1}, windows over the last half of stages 3 and 2 and over
    all of stage 1, ``epu`` episodes per update."""
    cfg = tc.build_bw_config(PROTO, q, seed, out, BW_PARAMS)
    b3, b2, b1 = budgets
    cfg["pipeline"]["budgets"] = {"3": b3, "2": b2, "1": b1}
    cfg["pipeline"]["lr_windows"] = {
        "3": [{"first": b3 // 2 + 1, "last": b3, "start": 3e-4, "end": 3e-5}],
        "2": [{"first": b2 // 2 + 1, "last": b2, "start": 3e-4, "end": 3e-5}],
        "1": [{"first": 1, "last": b1, "start": 3e-4, "end": 3e-5}]}
    cfg["record"]["protocol"]["episodes_per_update"] = epu
    return cfg


def write_cfg(cfg: Dict[str, Any], out: str) -> str:
    """Create the run directory and write ``run_config.json`` there; return the config path."""
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "run_config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=1)
    return path


def tree_equal(a: Any, b: Any) -> bool:
    """Exact (bitwise for tensors / arrays) equality of nested dict / list / tuple / tensor structures."""
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


# ---------------------------------------------------------------------------------- session-wide run caches
SPECS = {   # name -> (arm, T, starts)
    "t3_root": ("jt", 3, ROOT_ONLY_T3), "t3_es": ("jt_es", 3, ES_MIX_T3),
    "t2_root": ("jt2", 2, ROOT_ONLY_T2), "t2_es": ("jt2", 2, ES_MIX_T2)}
REDUCED_U, REDUCED_CHECK = 30, 10
_CACHE: Dict[str, Any] = {"dir": None, "joint": {}, "bw": None}


def _tmp() -> str:
    if _CACHE["dir"] is None:
        _CACHE["dir"] = tempfile.mkdtemp(prefix="t3_tests_")
        atexit.register(shutil.rmtree, _CACHE["dir"], True)
    return _CACHE["dir"]


def reduced_runs() -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """The four reduced joint runs (U = 30, 48 episodes, checks and exports every 10 updates, one decay window),
    run once per test session: ``{name: (run_dir, config)}``."""
    from run import run_joint as rj
    if not _CACHE["joint"]:
        for name, (arm, T, starts) in SPECS.items():
            d = os.path.join(_tmp(), name)
            os.makedirs(d)
            cfg = joint_cfg(d, arm, 50, 50500, U=REDUCED_U, starts=starts, check_every=REDUCED_CHECK)
            assert rj.run_pipeline(cfg, d, "pytest") == 0, name
            _CACHE["joint"][name] = (d, cfg)
    return _CACHE["joint"]


def reduced_bw_run() -> Tuple[str, Dict[str, Any]]:
    """The reduced legacy ``bw`` run at T = 3 (budgets 40 / 40 / 20, 48 episodes), run once per test session."""
    from run import run_ms_stagewise as rms
    if _CACHE["bw"] is None:
        d = os.path.join(_tmp(), "bw")
        os.makedirs(d)
        cfg = bw_cfg(d)
        assert rms.run_pipeline(cfg, d, "pytest") == 0
        _CACHE["bw"] = (d, cfg)
    return _CACHE["bw"]
