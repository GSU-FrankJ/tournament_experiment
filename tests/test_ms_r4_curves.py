"""MS-R4 offline curve tool (tools/ms/r4_curves.py): on genuine reduced runs the development-tier values equal the online checks
bit for bit, the last stage-2 point equals the final-tier record of the freeze, the tool is pure (no RNG moved, nothing written
under the run), it is variant-aware and it refuses what it cannot read.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_r4_curves.py -p no:cacheprovider -q
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ms_r4_helpers as H  # noqa: E402,I001
from ms_r4_helpers import run_dir  # noqa: E402

import r4_configs as rc  # noqa: E402
import r4_curves as CU  # noqa: E402

Q, SEED = 50, 10511


@pytest.fixture(scope="module")
def wave(tmp_path_factory) -> Path:
    return H.get_wave(tmp_path_factory)


def _tree_digest(d: Path) -> Dict[str, str]:
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("arm", ["fb_a0", "fb_rg", "t1_st_s16", "relu_st_s16", "lrsm_st_s16"])
def test_the_development_tier_equals_the_online_checks_bit_for_bit_and_the_last_point_the_freeze(wave, arm):
    rows, rec = CU.curve_run(run_dir(wave, arm), arm, Q, SEED, c_off=True, strict_online=False)
    n2 = 100 if arm in rc.E1_ARMS else 60
    assert len(rows) == n2 // 5 and rec["status"] == "ok"
    assert rec["n_mismatch"] == 0 and rec["max_abs_diff"] == 0.0 and rec["n_compared"] == (n2 // 10) * len(CU.C_OFF_PAIRS)
    assert rec["n_no_online_row"] == n2 // 10                          # the exports at 5, 15, ... have no check row (K = 10 here)
    assert rec["last_export_update"] == n2
    assert rec["last_final_eta_equals_freeze"] and rec["last_dev_eta_equals_freeze"] and rec["last_final_peak_equals_freeze"]
    assert rec["ok"] is True
    online = {int(r["local"]): r for r in csv.DictReader(open(run_dir(wave, arm) / "ms_checks_stage2.csv"))}
    for r in rows:
        if r["local"] in online:
            assert r["dev_eta_T_over_dw"] == float(online[r["local"]]["Delta"]) == float(online[r["local"]]["eta_T_over_dw"])
    g = json.load(open(run_dir(wave, arm) / "gates.json"))
    assert rows[-1]["final_eta_T_over_dw"] == g["metric_values"]["eta_final"] and rows[-1]["dev_eta_T_over_dw"] == g["metric_values"]["eta_dev"]
    assert rows[-1]["final_stage2_peak_rel_err_abs"] == g["reported"]["end_of_stage2"]["final"]["stage2_peak_rel_err_abs"]
    assert {r["actor_variant"] for r in rows} == {{"relu_st_s16": "relu", "lrsm_st_s16": "lrsm"}.get(arm, "t1")}
    assert all(np.isfinite(r["final_eta_T_over_dw"]) and r["final_valid"] and r["dev_valid"] for r in rows)


def test_the_conc_scale_of_the_export_is_used(wave):
    rows, _ = CU.curve_run(run_dir(wave, "lrsm_st_s16"), "lrsm_st_s16", Q, SEED, c_off=False)
    cs = {r["update"]: r["conc_scale"] for r in rows}
    assert cs[60] == 1.0 and cs[65] < 16.0 and cs[65] > 1.0 and cs[80] == 16.0 and cs[100] == 16.0       # the ramp 61-80 of the reduced arm


def test_the_tool_is_pure_it_moves_no_rng_and_writes_nothing_under_the_run(wave):
    d = run_dir(wave, "fb_rg")
    before_files = _tree_digest(d)
    np_state, py_state, torch_state = np.random.get_state(), random.getstate(), torch.get_rng_state().clone()
    np.random.seed(123)
    random.seed(123)
    ref = (np.random.get_state()[1].copy(), random.getstate(), torch.get_rng_state().clone())
    CU.curve_run(d, "fb_rg", Q, SEED, c_off=True, strict_online=False)
    after = (np.random.get_state()[1], random.getstate(), torch.get_rng_state())
    assert np.array_equal(ref[0], after[0]) and ref[1] == after[1] and torch.equal(ref[2], after[2])
    assert _tree_digest(d) == before_files
    np.random.set_state(np_state)
    random.setstate(py_state)
    torch.set_rng_state(torch_state)


def test_a_missing_or_unreadable_run_is_reported_not_skipped(wave, tmp_path):
    rows, rec = CU.curve_run(tmp_path / "nope", "fb_r", Q, SEED)
    assert rows == [] and rec["status"] == "missing" and rec["ok"] is False and rec["first_mismatch"]
    w = tmp_path / "w"
    shutil.copytree(run_dir(wave, "fb_r"), w)
    (w / "weights" / "u00030.npz").unlink()
    rows, rec = CU.curve_run(w, "fb_r", Q, SEED, strict_online=False)
    # a failed or damaged run: the exports that exist are evaluated, the gap is reported and the run is not ok
    assert rec["status"] == "incomplete" and "1 of 12 stage-2 exports are missing" in rec["first_mismatch"] and rec["ok"] is False
    assert [r["update"] for r in rows] == [u for u in range(5, 61, 5) if u != 30]
    assert CU.export_paths(w, 60, 5, strict=False)[5][0] == 35
    with pytest.raises(FileNotFoundError, match="u00030"):
        CU.export_paths(w, 60, 5)
    shutil.rmtree(w / "weights")
    rows, rec = CU.curve_run(w, "fb_r", Q, SEED, strict_online=False)
    assert rows == [] and rec["status"] == "missing" and rec["ok"] is False and "no stage-2 export" in rec["first_mismatch"]


def test_a_changed_online_value_or_freeze_record_is_a_mismatch(wave, tmp_path):
    w = tmp_path / "w"
    shutil.copytree(run_dir(wave, "fb_r"), w)
    rows = list(csv.DictReader(open(w / "ms_checks_stage2.csv")))
    cols = list(rows[0])
    rows[2]["Delta"] = repr(float(rows[2]["Delta"]) * (1 + 1e-9))
    with open(w / "ms_checks_stage2.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    _, rec = CU.curve_run(w, "fb_r", Q, SEED, strict_online=False)
    assert rec["ok"] is False and rec["n_mismatch"] >= 1 and "Delta" in rec["first_mismatch"] and rec["max_abs_diff"] > 0
    w2 = tmp_path / "w2"
    shutil.copytree(run_dir(wave, "fb_r"), w2)
    g = json.load(open(w2 / "gates.json"))
    g["metric_values"]["eta_final"] *= 1.001
    json.dump(g, open(w2 / "gates.json", "w"))
    _, rec2 = CU.curve_run(w2, "fb_r", Q, SEED, strict_online=False)
    assert rec2["ok"] is False and rec2["last_final_eta_equals_freeze"] is False


def test_strict_online_flags_an_export_without_a_check_row(wave):
    _, rec = CU.curve_run(run_dir(wave, "fb_a0"), "fb_a0", Q, SEED, c_off=True, strict_online=True)
    assert rec["ok"] is False and "no online check row" in rec["first_mismatch"]          # reduced runs export more often than they check


def test_run_all_writes_per_run_files_the_combined_table_and_the_info(wave, tmp_path):
    out = tmp_path / "curves"
    info = CU.run_all(wave, out, ["fb_a0", "lrsm_st_s16"], [Q], [SEED], workers=2, c_off=False)
    assert info["n_runs"] == 2 and info["n_ok"] == 2 and info["n_points"] == 12 + 20
    for arm in ("fb_a0", "lrsm_st_s16"):
        assert (out / f"q{Q}" / f"seed{SEED}" / f"{arm}.csv").exists()
    allrows = list(csv.DictReader(open(out / "curves_all.csv")))
    assert len(allrows) == 32 and list(allrows[0]) == CU.CURVE_COLS
    assert json.load(open(out / "curves_info.json"))["c_off"] is False
    with pytest.raises(FileExistsError):                                                          # never overwrites
        CU.write_csv(out / "curves_all.csv", CU.CURVE_COLS, [])


def test_main_refuses_a_non_empty_output_directory_and_too_many_workers(wave, tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "x").write_text("x")
    with pytest.raises(SystemExit):
        CU.main(["--pilot-root", str(wave), "--out", str(out), "--workers", "4"])
    with pytest.raises(SystemExit):
        CU.main(["--pilot-root", str(wave), "--out", str(tmp_path / "o2"), "--workers", "41"])
    assert CU.MAX_WORKERS == 40


def test_export_paths_names_the_missing_file(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "u00025.npz").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="u00050"):
        CU.export_paths(tmp_path, 75)
    assert [u for u, _ in CU.export_paths(tmp_path, 25)] == [25]
