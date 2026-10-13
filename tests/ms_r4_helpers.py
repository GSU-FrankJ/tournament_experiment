"""Shared helpers of the MS-R4 tests (not a test module): reduced-budget configs of the seven arms, a worker for genuine
reduced runs and the tree of the base commit.

The reduced budgets keep the structure of the real arms: the E2 arms run 60 + 20 updates with the learning rate constant to
update 40 and decaying over 41-60; the E1 arms run 100 + 20 updates with the concentration ramp over 61-80, the learning rate
constant to update 80 and decaying over 81-100. Checks every K = 10 updates, exports every 5, 64 episodes per update.
"""

from __future__ import annotations

import copy
import io
import json
import multiprocessing
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import ms_configs as mc  # noqa: E402
import r4_configs as rc  # noqa: E402

PROTO = mc.load_protocol()
PARAMS = json.load(open(ROOT / "reports" / "ms" / "r1" / "prereg_parameters.json"))
BASE_COMMIT = "be2881ff"          # the head of t2-status-pack: the code the defaults must reproduce bit for bit
FAKE_GIT = {"commit": "abc1234", "short": "abc1234", "dirty": False}
EPU = 64
E2_N2, E2_CONST_TO = 60, 40                       # E2 arms: terminal stage 60, LR constant to 40, decay 41-60
E1_N2, E1_CONST_TO, E1_RAMP = 100, 80, (61, 80)   # E1 arms: terminal stage 100, ramp 61-80, LR constant to 80, decay 81-100
N1 = 20


def reduced_r4(arm: str, out: str, q: int = 50, seed: int = 10511, K: int = 10, weights_every: int = 5,
               n1: int = N1, epu: int = EPU) -> Dict[str, Any]:
    """The config of an MS-R4 arm with the budgets scaled down (see the module docstring)."""
    prm = copy.deepcopy(PARAMS)
    prm["K"] = K
    cfg = rc.build_config_r4(PROTO, q, seed, arm, out, prm)
    cfg["record"]["protocol"]["episodes_per_update"] = epu
    cfg["record"]["protocol"]["weights_every"] = weights_every
    n2, const_to = (E1_N2, E1_CONST_TO) if arm in rc.E1_ARMS else (E2_N2, E2_CONST_TO)
    cfg["pipeline"]["budgets"] = {"2": n2, "1": n1}
    cfg["pipeline"]["lr_windows"] = {"2": [{"first": const_to + 1, "last": n2, "start": 3e-4, "end": 3e-5}],
                                     "1": [{"first": 1, "last": n1, "start": 3e-4, "end": 3e-5}]}
    if "conc_scale_schedule" in cfg:
        cfg["conc_scale_schedule"].update(local_first=E1_RAMP[0], local_last=E1_RAMP[1])
    return cfg


def run_one(job: Tuple[Dict[str, Any], str]) -> int:
    """One genuine reduced run in a worker process (a module-level function: the pool pickles it by name)."""
    from run import run_ms_stagewise as rms
    cfg, out = job
    rms.git_state = lambda: dict(FAKE_GIT)
    os.makedirs(out, exist_ok=True)
    return rms.run_pipeline(cfg, out, "pytest", band_step=2.0)


def base_tree(tmp: Path) -> str:
    """The run code of the base commit (``git archive`` of run utils envs agents protocols) in ``tmp``; raises
    ``subprocess.CalledProcessError`` if the commit is not available."""
    tar = subprocess.check_output(["git", "-C", str(ROOT), "archive", BASE_COMMIT, "--", "run", "utils", "envs", "agents",
                                   "protocols"], stderr=subprocess.DEVNULL)
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(tmp)
    return str(tmp)


def run_dir(root: Path, arm: str, q: int = 50, seed: int = 10511) -> Path:
    return Path(root) / f"q{q}" / f"seed{seed}" / arm


_WAVE: Dict[Tuple[str, ...], Path] = {}


def get_wave(tmp_path_factory: Any, arms: Sequence[str] = rc.ARMS, q: int = 50, seed: int = 10511, workers: int = 4) -> Path:
    """The pilot root of genuine reduced runs of ``arms`` at one (q, seed), built once per test session (spawned workers).

    Returns:
        The root holding ``q<q>/seed<seed>/<arm>``.
    """
    key = (tuple(arms), q, seed)
    if key not in _WAVE:
        root = Path(tmp_path_factory.mktemp("r4wave")) / "pilot"
        jobs = [(reduced_r4(a, str(run_dir(root, a, q, seed)), q=q, seed=seed), str(run_dir(root, a, q, seed))) for a in arms]
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            codes = list(pool.map(run_one, jobs))
        assert codes == [0] * len(jobs), codes
        _WAVE[key] = root
    return _WAVE[key]
