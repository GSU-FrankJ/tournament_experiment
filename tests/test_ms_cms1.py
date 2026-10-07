"""C-MS1 comparison tool (tools/ms/cms1_compare.py) and the C-MS2 check of tools/ms/launch_checks.py on reduced
legacy runs and on tampered copies."""

from __future__ import annotations

import csv
import json
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


# ====================================================================== C-MS2 (addendum A1)
# MS_base2400 against parents_A through update 1201 (the first update of parents_A's LR window), reduced:
# reference = v2.0 phase_A with budget N2_REF (window N2_REF//2+1 ... N2_REF, so THROUGH = 41), new = the legacy
# arm with the terminal stage fixed at 2 * N2_REF (window 3/4 of the way to the end, as 2001-2400 of 2400).
import launch_checks as LC  # noqa: E402
from test_ms_runner import make_cfg  # noqa: E402

N2_REF, THROUGH = 80, 41


def cfg_2400_like(out: str, q: int = 50, seed: int = 10501, epu: int = EPU) -> dict:
    """MS_base2400 with budgets scaled down: terminal stage 2 * N2_REF, window = the last quarter."""
    n = 2 * N2_REF
    cfg = make_cfg(out, "MS_base2400", q, seed, epu=epu)
    cfg["pipeline"]["budgets"] = {"2": n, "1": 40}
    cfg["pipeline"]["lr_windows"] = {"2": [{"first": 3 * n // 4 + 1, "last": n, "start": 3e-4, "end": 3e-5}],
                                     "1": [{"first": 1, "last": 40, "start": 3e-4, "end": 3e-5}]}
    return cfg


@pytest.fixture(scope="module")
def pair2400(tmp_path_factory):
    """A reduced v2.0 phase_A run and the reduced MS_base2400 run (terminal stage only)."""
    root = tmp_path_factory.mktemp("cms2")
    ref, new = root / "ref" / "q50" / "seed10501", root / "new" / "q50" / "seed10501" / "MS_base2400"
    ref.mkdir(parents=True)
    new.mkdir(parents=True)
    cfg_ref = v2_phase_a_cfg(50, 10501, N2_REF, EPU)
    assert execute(Run(cfg_ref, str(ref)), cfg_ref, str(ref), "pytest") == 0
    ms = rms.MSRun(cfg_2400_like(str(new)), str(new))
    rec = ms.run_stage(2)
    assert rec["total_updates"] == 2 * N2_REF
    ms.freeze_stage(2, rec, lambda point: None)
    rms.write_histories(ms, str(new))
    return ref, new


def test_ms_base2400_equals_parents_a_through_the_first_window_update(pair2400):
    ref, new = pair2400
    r = LC.cms2_run(ref, new, THROUGH)
    assert r["ALL"], r["first_difference"]
    assert set(r["fields"]) == {"weight_exports_through", "train_history:series",
                                "updates_csv:stream_positions_and_losses", "first_export_after_differs"}
    assert r["info"]["n_weight_exports_compared"] == 1                      # u25 (u50 is after the window start)
    # the first update of the reference's window runs at the base rate, the next one does not
    ra = {h["update"]: h for h in json.load(open(ref / "train_history.json"))["history"]}
    assert ra[THROUGH]["lr"] == 3e-4 and ra[THROUGH + 1]["lr"] < 3e-4
    rb = {h["update"]: h for h in json.load(open(new / "train_history.json"))["history"]}
    assert rb[THROUGH + 1]["lr"] == 3e-4


def _copy2400(pair2400, tmp_path):
    ref, new = pair2400
    n2 = tmp_path / "n"
    shutil.copytree(new, n2)
    return ref, n2


def test_cms2_finds_a_changed_export_series_value_or_stream_position(pair2400, tmp_path):
    ref, new = _copy2400(pair2400, tmp_path)
    f = new / "weights" / "u00025.npz"
    z = dict(np.load(f))
    z["actor.l1.bias"] = z["actor.l1.bias"].copy()
    z["actor.l1.bias"][0] += np.float32(1e-7)
    np.savez(f, **z)
    r = LC.cms2_run(ref, new, THROUGH)
    assert not r["ALL"] and r["fields"]["weight_exports_through"] is False
    ref2, new2 = _copy2400(pair2400, tmp_path / "b")
    th = json.load(open(new2 / "train_history.json"))
    th["history"][THROUGH - 1]["policy_loss"] += 1e-9                       # update 41: compared
    json.dump(th, open(new2 / "train_history.json", "w"))
    r = LC.cms2_run(ref2, new2, THROUGH)
    assert not r["ALL"] and r["fields"]["train_history:series"] is False
    ref3, new3 = _copy2400(pair2400, tmp_path / "c")
    th = json.load(open(new3 / "train_history.json"))
    th["history"][THROUGH]["policy_loss"] += 1e-9                           # update 42: not compared
    json.dump(th, open(new3 / "train_history.json", "w"))
    assert LC.cms2_run(ref3, new3, THROUGH)["ALL"]
    ref4, new4 = _copy2400(pair2400, tmp_path / "d")
    rows = list(csv.DictReader(open(new4 / "ms_updates.csv")))
    rows[THROUGH - 1]["rngpos_start"] = "0:0:0"
    with open(new4 / "ms_updates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    r = LC.cms2_run(ref4, new4, THROUGH)
    assert not r["ALL"] and r["fields"]["updates_csv:stream_positions_and_losses"] is False


def test_cms2_requires_the_export_after_the_window_start_to_differ(pair2400, tmp_path):
    ref, new = _copy2400(pair2400, tmp_path)
    shutil.copy(ref / "weights" / "u00050.npz", new / "weights" / "u00050.npz")
    r = LC.cms2_run(ref, new, THROUGH)
    assert not r["ALL"] and r["fields"]["first_export_after_differs"] is False
    assert r["fields"]["weight_exports_through"] and r["fields"]["train_history:series"]
    ref2, new2 = _copy2400(pair2400, tmp_path / "b")
    os.remove(new2 / "weights" / "u00050.npz")
    assert LC.cms2_run(ref2, new2, THROUGH)["fields"]["first_export_after_differs"] is False   # missing = failure


def test_cms2_missing_run_is_a_difference_and_the_root_summary(pair2400, tmp_path):
    ref, new = pair2400
    assert LC.cms2_run(ref, tmp_path / "absent", THROUGH)["ALL"] is False
    root_ref, root_new = ref.parents[1], new.parents[2]
    s = LC.cms2_root(root_ref, root_new, "MS_base2400", [50], [10501], THROUGH)
    assert s["ALL"] and s["n"] == 1 and s["n_identical"] == 1 and s["through_update"] == THROUGH
    s = LC.cms2_root(root_ref, root_new, "MS_base2400", [50, 60], [10501], THROUGH)
    assert not s["ALL"] and s["n_identical"] == 1 and s["first_differences"][0]["q"] == 60
    assert LC.CMS2_THROUGH == 1201


def test_the_real_schedule_of_ms_base2400_holds_the_base_rate_through_update_1201(tmp_path):
    """The claim behind C-MS2 on the real budgets: 3e-4 up to local 2000 (so also at 1201, the first update of
    parents_A's window), linear 3e-4 -> 3e-5 over 2001-2400; parents_A's own window starts at 3e-4 at 1201."""
    cfg = make_cfg(str(tmp_path), "MS_base2400")
    cfg["pipeline"]["budgets"] = {"2": 2400, "1": 600}
    cfg["pipeline"]["lr_windows"] = {"2": [{"first": 2001, "last": 2400, "start": 3e-4, "end": 3e-5}],
                                     "1": [{"first": 1, "last": 600, "start": 3e-4, "end": 3e-5}]}
    ms = rms.MSRun(cfg, str(tmp_path))
    fn = ms._legacy_lr(2)
    parents = {"kind": "linear", "c_start_lr": 3e-4, "c_end_lr": 3e-5, "c_local_first": 1201, "linear_denominator": 399}
    assert all(fn(j) == 3e-4 for j in range(1, 2002))
    assert rms.lr_at(parents, "C", 1201) == 3e-4 and rms.lr_at(parents, "C", 1202) < 3e-4
    assert fn(2002) < 3e-4 and abs(fn(2400) - 3e-5) < 1e-18
    assert all(fn(j) > fn(j + 1) for j in range(2001, 2400))


def test_the_cli_runs_cms2_and_writes_it_into_the_launch_checks_json(pair2400, tmp_path, capsys):
    ref, new = pair2400
    out = tmp_path / "lc.json"
    argv = ["--root", str(new.parents[2]), "--arms", "MS_base2400", "--qs", "50", "--seeds", "10501",
            "--out", str(out), "--cms2-ref", str(ref.parents[1]), "--cms2-through", str(THROUGH)]
    rc = LC.main(argv)                 # the per-run launch checks fail (no status.json: a bare reduced run)
    printed = capsys.readouterr().out
    rec = json.load(open(out))
    assert rc == 2 and rec["all_ok"] is False
    assert rec["c_ms2"]["ALL"] is True and rec["c_ms2"]["arm"] == "MS_base2400" and rec["c_ms2"]["through_update"] == THROUGH
    assert f"C-MS2 (MS_base2400 against parents_A through update {THROUGH}) identical 1/1 ALL=True" in printed
    rc = LC.main(argv[:-2] + ["--cms2-through", "9999"])             # beyond the reference's last update: a failure
    assert rc == 2 and "ALL=False" in capsys.readouterr().out


def test_a_cms2_failure_alone_makes_the_exit_code_nonzero(pair2400, tmp_path, monkeypatch, capsys):
    ref, new = pair2400
    monkeypatch.setattr(LC, "check_root", lambda *a, **k: {
        "n_runs": 1, "n_ok_per_check": {}, "all_ok": True, "start_share_tests": {}})   # every per-run check passes
    argv = ["--root", str(new.parents[2]), "--arms", "MS_base2400", "--qs", "50", "--seeds", "10501",
            "--cms2-ref", str(ref.parents[1]), "--cms2-through", str(THROUGH)]
    assert LC.main(argv) == 0
    assert LC.main(argv[:-2] + ["--cms2-through", "9999"]) == 2          # only C-MS2 differs between the two calls
    capsys.readouterr()
