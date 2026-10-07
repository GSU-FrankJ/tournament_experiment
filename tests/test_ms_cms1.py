"""C-MS1 comparison tool (tools/ms/cms1_compare.py) on a reduced legacy run and on tampered copies."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import cms1_compare as M  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402
from run.run_v2_stagewise import Run, execute  # noqa: E402
from test_ms_runner import legacy_cfg, v2_phase_a_cfg  # noqa: E402

N, EPU = 60, 96


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """A reduced v2.0 phase_A run and the MS_base run (terminal stage only; the stage-1 phase is not needed)."""
    root = tmp_path_factory.mktemp("cms1")
    ref, new = root / "ref", root / "new"
    ref.mkdir()
    new.mkdir()
    cfg_ref = v2_phase_a_cfg(50, 10501, N, EPU)
    assert execute(Run(cfg_ref, str(ref)), cfg_ref, str(ref), "pytest") == 0
    ms = rms.MSRun(legacy_cfg(str(new), 50, 10501, N, 20, EPU), str(new))
    rec = ms.run_stage(2)
    ms.freeze_stage(2, rec, lambda point: None)          # state_end_stage2.pt, freeze_stage2_{tier}.npz
    rms.write_histories(ms, str(new))                     # train_history.json, ms_updates.csv
    return ref, new


def _compare(ref, new):
    return M.compare_run(ref, new, "parentsA", 2, N)


def test_identical_runs_pass_every_field(pair):
    r = _compare(*pair)
    assert r["ALL"], r["first_difference"]
    assert all(r["fields"].values()) and len(r["fields"]) >= 17


def _copy(pair, tmp_path):
    ref, new = pair
    n2 = tmp_path / "n"
    shutil.copytree(new, n2)
    return ref, n2


def test_a_changed_weight_export_is_found(pair, tmp_path):
    ref, new = _copy(pair, tmp_path)
    f = new / "weights" / "u00025.npz"
    z = dict(np.load(f))
    z["actor.l1.bias"] = z["actor.l1.bias"].copy()
    z["actor.l1.bias"][0] += np.float32(1e-7)
    np.savez(f, **z)
    r = _compare(ref, new)
    assert not r["ALL"] and r["fields"]["weight_exports"] is False and r["first_difference"]["field"] == "weight_exports"


def test_a_changed_stream_position_or_state_is_found(pair, tmp_path):
    ref, new = _copy(pair, tmp_path)
    st = torch.load(new / "state_end_stage2.pt", weights_only=False)
    st["rng"]["start"]["state"]["state"] += 1
    torch.save(st, new / "state_end_stage2.pt")
    r = _compare(ref, new)
    assert not r["ALL"] and r["fields"]["state:rng_streams"] is False
    ref2, new2 = _copy(pair, tmp_path / "b")
    st = torch.load(new2 / "state_end_stage2.pt", weights_only=False)
    st["agent"]["actor"]["l1.bias"][0] += 1e-7
    torch.save(st, new2 / "state_end_stage2.pt")
    assert _compare(ref2, new2)["fields"]["state:actor"] is False


def test_a_missing_file_or_run_is_a_difference_not_a_skip(pair, tmp_path):
    ref, new = _copy(pair, tmp_path)
    os.remove(new / "freeze_stage2_final.npz")
    r = _compare(ref, new)
    assert not r["ALL"] and r["fields"]["eval:freeze_stage2_final.npz"] is False
    assert M.compare_run(ref, tmp_path / "absent", "parentsA", 2, N)["ALL"] is False


def test_a_tampered_series_is_found(pair, tmp_path):
    import json
    ref, new = _copy(pair, tmp_path)
    th = json.load(open(new / "train_history.json"))
    th["history"][5]["policy_loss"] += 1e-9
    json.dump(th, open(new / "train_history.json", "w"))
    r = _compare(ref, new)
    assert not r["ALL"] and r["fields"]["train_history:series"] is False
