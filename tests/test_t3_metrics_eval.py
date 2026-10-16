"""T3-R1: tests of ``tools/t3/t3_metrics.py`` (band split, terminal-stage recovery, gates) and
``tools/t3/t3_eval.py`` (the three run formats, the replay of the runs' own evaluation arrays, best of run).

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_t3_metrics_eval.py -p no:cacheprovider -q
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil

import numpy as np
import pytest

import t3_helpers as H
import t3_eval as te  # noqa: E402
import t3_metrics as tm  # noqa: E402
from envs.curriculum_env import GameSpec  # noqa: E402
from replay_dev_rule import candidate_fns  # noqa: E402
from utils.dp_br_verifier import DEV_CONFIG, FINAL_CONFIG  # noqa: E402
from utils.theory_multistage import g2_two_stage  # noqa: E402
from utils.v2_metrics import evaluate  # noqa: E402

CRIT, CRIT_SHA = tm.load_criteria()


@pytest.fixture(scope="module")
def runs():
    return H.reduced_runs()


def spec_of(T: int, q: int = 50) -> GameSpec:
    return GameSpec(w_h=6, w_l=2, k=1.0 / 3500.0, q=q, T=T)


# ====================================================================== t3_metrics
def test_criteria_file_has_the_pre_registered_values():
    g = CRIT["gates"]
    assert g["G-A"] == {"eta_T_over_dw": 0.005, "rmse_pos_T_over_eT0": 0.05, "tail_mean_T_over_eT0": 0.02}
    assert g["G-F"] == {"Gmax_full_over_dw": 0.01} and g["G-S"]["stage1_rel_err_abs"] == 0.05
    assert g["G-N"] == {"eta_T_over_dw_dev_minus_final_abs": 0.001, "Gmax_full_over_dw_dev_minus_final_abs": 0.001}
    s = CRIT["statistics"]
    assert s["n_boot"] == 10000 and s["bootstrap_seed"] == {"3": 20261016, "2": 20261017}
    assert s["percentiles"] == {"ci95": [2.5, 97.5], "ci98_75": [0.625, 99.375]}
    assert CRIT["reading_rule"]["min_passes_of_20_at_each_q"] == 18
    assert set(CRIT["hypotheses"]) == {"H1", "H2", "H3", "H1-T2"}
    assert len(CRIT_SHA) == 64
    # v2.0's thresholds
    proto = H.PROTO["gates"]
    assert proto["G-A"]["all_must_hold"][0]["threshold"] == g["G-A"]["eta_T_over_dw"]
    assert proto["G-F"]["all_must_hold"][0]["threshold"] == g["G-F"]["Gmax_full_over_dw"]
    assert proto["G-S"]["all_must_hold"][0]["threshold"] == g["G-S"]["stage1_rel_err_abs"]


def test_the_code_constants_equal_the_criteria_file():
    assert tm.BAND_TOL == CRIT["band"]["node_tolerance"] == 1e-9
    assert CRIT["band"]["in_band"].startswith("|d| < 2 q (t - 1)") and CRIT["recovery"]["pos"] == "|d| < 2 q"
    assert CRIT["statistics"]["percentiles"] == {"ci95": [2.5, 97.5], "ci98_75": [0.625, 99.375]}


@pytest.mark.parametrize("q", [50, 60])
def test_terminal_recovery_equals_evaluate_bit_for_bit_at_t2(q):
    """On T = 2 policies the terminal-stage recovery fields equal ``evaluate``'s stage-2 fields exactly (two
    policies: a smooth synthetic one and the tent shifted and clipped)."""
    spec = spec_of(2, q)
    pols = {
        "smooth": lambda t, d: np.full(np.shape(d), 40.0) if t == 1 else 60.0 * np.exp(-(np.asarray(d, float) / 70.0) ** 2),
        "tent": lambda t, d: (np.full(np.shape(d), 29.0) if t == 1 else
                              np.clip(g2_two_stage(np.asarray(d, float) + 1.3, q, 6, 2, spec.k, 100.0) * 0.93 + 0.4, 0, 100)),
    }
    for name, pol in pols.items():
        for tier in (FINAL_CONFIG, DEV_CONFIG):
            ev = evaluate(pol, spec, tier, recovery_step=0.5)
            rec, _ = tm.terminal_recovery(pol, spec, 0.5)
            assert tm.terminal_recovery_matches_evaluate(ev, rec), (name, tier.name)
            tmx = tm.tier_metrics(ev, pol, spec, 0.5)
            assert tmx["stage1_rel_err_signed"] == ev.scalars["stage1_rel_err_signed"]
            assert tmx["stage1_rel_err_abs"] == ev.scalars["stage1_rel_err_abs"]
            assert tmx["e1_at_0"] == ev.scalars["e1_at_0"] and tmx["g1"] == ev.scalars["g1"]
            assert tmx["recovery_n_pos"] == ev.scalars["recovery_n_pos"] and tmx["recovery_n_tail"] == ev.scalars["recovery_n_tail"]


def test_terminal_recovery_at_t3_uses_the_grid_of_d3():
    spec = spec_of(3, 50)
    pol = lambda t, d: np.clip(g2_two_stage(np.asarray(d, float), 50, 6, 2, spec.k, 100.0), 0, 100)   # noqa: E731
    rec, arr = tm.terminal_recovery(pol, spec, 0.5)
    D = arr["recovery_d_grid"]
    assert D[0] == -400.0 and D[-1] == 400.0 and D.size == 1601
    assert rec["recovery_n_pos"] == int((np.abs(D) < 100).sum()) == 399
    assert rec["recovery_n_tail"] == 1601 - 399               # |d| >= 2q: 100 to 400 per side
    assert rec["rmse_pos_T_over_eT0"] == 0.0 and rec["tail_mean_T_over_eT0"] == 0.0
    assert rec["peak_T_signed"] == 0.0 and rec["peak_T_abs"] == 0.0
    assert rec["eT_at_0"] == rec["eT_star_at_0"]
    pol2 = lambda t, d: np.full(np.shape(d), 2.0)     # noqa: E731
    rec2, _ = tm.terminal_recovery(pol2, spec, 0.5)
    assert rec2["tail_mean_T_over_eT0"] == pytest.approx(2.0 / rec2["eT_star_at_0"], rel=1e-12)
    assert rec2["tail_max_T_over_eT0"] == pytest.approx(2.0 / rec2["eT_star_at_0"], rel=1e-12)


def test_band_split_on_synthetic_arrays_with_known_answers():
    q, t, dw = 2.0, 3, 4.0                                   # band: |d| < 2 q (t - 1) = 8
    d = np.arange(-10.0, 11.0)
    delta = np.zeros_like(d)
    gain = np.zeros_like(d)
    delta[np.abs(d) == 8] = 5.0                              # exactly on the edge: out of band
    delta[d == 3] = 2.0                                      # in band
    gain[d == -7] = 1.0                                      # in band, next to the edge
    gain[d == 10] = 3.0                                      # out of band, domain end
    c = tm.band_columns(d, delta, gain, q, t, dw)
    assert c["Delta_in_t3"] == 2.0 / dw and c["Delta_in_argmax_d_t3"] == 3.0
    assert c["Delta_out_t3"] == 5.0 / dw and abs(c["Delta_out_argmax_d_t3"]) == 8.0
    assert c["G_in_t3"] == 1.0 / dw and c["G_in_argmax_d_t3"] == -7.0
    assert c["G_out_t3"] == 3.0 / dw and c["G_out_argmax_d_t3"] == 10.0
    assert c["n_in_t3"] == 15 and c["n_out_t3"] == 6          # |d| <= 7 -> 15 nodes
    # a node an epsilon inside or outside the edge counts as on the edge (out of band); 7.9999 is inside
    d2 = np.array([-8.0 + 1e-12, 8.0 - 1e-12, 8.0 + 1e-12, 7.9999, 0.0])
    m = tm.in_band_mask(d2, q, t)
    assert m.tolist() == [False, False, False, True, True]
    # an empty side gives NaN
    c0 = tm.band_columns(np.array([-20.0, 20.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0]), q, t, dw)
    assert math.isnan(c0["Delta_in_t3"]) and c0["Delta_out_t3"] == 2.0 / dw and c0["n_in_t3"] == 0


@pytest.mark.parametrize("name", ["t3_root", "t3_es"])
def test_band_maxima_cover_the_full_maximum_of_a_real_evaluation(runs, name):
    d, cfg = runs[name]
    cand = te.load_joint(__import__("pathlib").Path(d), cfg)
    mean_fn, _ = candidate_fns(cand.spec, cand.nets)
    for tier in (FINAL_CONFIG, DEV_CONFIG):
        ev = te.evaluate_tier(cand, tier)
        m = tm.tier_metrics(ev, mean_fn, cand.spec, 0.5)
        for t in (2, 3):
            assert max(m[f"Delta_in_t{t}"], m[f"Delta_out_t{t}"]) == m[f"Delta_max_t{t}_over_dw"]
            assert max(m[f"G_in_t{t}"], m[f"G_out_t{t}"]) == m[f"G_max_t{t}_over_dw"]
            assert m[f"n_in_t{t}"] + m[f"n_out_t{t}"] == ev.arrays[f"v_t{t}_d_grid"].size
        assert m["n_in_t2"] == int((np.abs(ev.arrays["v_t2_d_grid"]) < 100.0).sum())
        assert m["n_in_t3"] == int((np.abs(ev.arrays["v_t3_d_grid"]) < 200.0).sum())
        assert m["eta_T_over_dw"] == m["Delta_max_t3_over_dw"] and m["Gmax_full_over_dw"] >= m["G_max_t3_over_dw"]


def test_gates_are_inclusive_and_v20_combines_s1_only_at_t2():
    fin = {"eta_T_over_dw": 0.005, "rmse_pos_T_over_eT0": 0.05, "tail_mean_T_over_eT0": 0.02,
           "Gmax_full_over_dw": 0.01, "stage1_rel_err_abs": 0.05}
    dev = {"eta_T_over_dw": 0.005 + 0.001, "Gmax_full_over_dw": 0.01 - 0.001}
    g3 = tm.gate_columns(fin, dev, 3, CRIT)
    assert g3["gate_G_A"] and g3["gate_G_F"] and g3["gate_G_N"] and g3["gate_G_S"] is None and g3["gates_pass"]
    g2 = tm.gate_columns(fin, dev, 2, CRIT)
    assert g2["gate_G_S"] is True and g2["gates_pass"]
    for key, bump in (("eta_T_over_dw", 1e-12), ("rmse_pos_T_over_eT0", 1e-12), ("tail_mean_T_over_eT0", 1e-12),
                      ("Gmax_full_over_dw", 1e-12)):
        bad = dict(fin, **{key: fin[key] + bump})
        r = tm.gate_columns(bad, dev, 3, CRIT)
        assert not r["gates_pass"], key
    assert not tm.gate_columns(dict(fin, stage1_rel_err_abs=0.05 + 1e-12), dev, 2, CRIT)["gates_pass"]
    assert tm.gate_columns(dict(fin, stage1_rel_err_abs=0.05 + 1e-12), dev, 3, CRIT)["gates_pass"]   # G-S: T = 2 only
    assert not tm.gate_columns(fin, dict(dev, eta_T_over_dw=0.005 + 0.0010000001), 3, CRIT)["gate_G_N"]
    assert not tm.gate_columns(fin, dict(dev, Gmax_full_over_dw=0.01 - 0.0010000001), 3, CRIT)["gate_G_N"]
    nan = dict(fin, eta_T_over_dw=float("nan"))
    assert not tm.gate_columns(nan, dev, 3, CRIT)["gate_G_A"]


# ====================================================================== t3_eval
@pytest.mark.parametrize("name", ["t3_root", "t3_es", "t2_root", "t2_es"])
def test_joint_evaluation_equals_the_runs_own_final_evaluation_bit_for_bit(runs, name):
    d, _ = runs[name]
    from pathlib import Path
    assert te.replay_mismatches(Path(d), "joint") == []


def test_bw_evaluation_equals_the_runners_freeze_arrays_bit_for_bit():
    d, _ = H.reduced_bw_run()
    from pathlib import Path
    assert te.replay_mismatches(Path(d), "bw") == []


def test_evaluate_run_columns_gates_and_arrays(runs, tmp_path):
    d, cfg = runs["t3_es"]
    job = {"wave": "t3", "arm": "jt_es", "q": 50, "seed": 50500, "T": 3, "run": cfg["run"], "run_dir": d,
           "kind": "joint", "arrays_dir": str(tmp_path / "arrays")}
    row = te.evaluate_run(job)
    assert row["status"] == "done" and row["exit_code"] == 0 and row["global_rng_ok"] is True and row["eval_error"] == ""
    for c in ("Gmax_full_over_dw", "EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw", "eta_T_over_dw",
              "G_max_t1_over_dw", "G_max_t2_over_dw", "G_max_t3_over_dw", "Delta_max_t1_over_dw", "Delta_in_t2",
              "Delta_out_t3", "G_in_t2", "G_out_t3", "rmse_pos_T_over_eT0", "tail_mean_T_over_eT0", "tail_max_T_over_eT0",
              "peak_T_signed", "peak_T_abs", "eT_at_0", "e1_at_0", "conc_max_std_norm_t1", "conc_max_std_norm_t3",
              "dev_Gmax_full_over_dw", "dev_eta_T_over_dw", "gate_G_A", "gate_G_F", "gate_G_N", "gates_pass"):
        assert c in row, c
    assert row["gate_G_S"] is None and "stage1_rel_err_abs" not in row
    z = np.load(tmp_path / "arrays" / f"{cfg['run']}.npz")
    assert {"fin_t1_d_grid", "fin_t3_e_hat", "fin_t2_delta", "fin_t3_G", "recovery_d_grid", "recovery_eT"} <= set(z.files)
    # the dev-tier value of the evaluation equals the in-run check at the last update bit for bit
    checks = list(csv.DictReader(open(os.path.join(d, "joint_checks.csv"))))
    assert float(checks[-1]["Gmax_full_over_dw"]) == row["dev_Gmax_full_over_dw"]
    assert float(checks[-1]["EXP_root_over_dw"]) == row["dev_EXP_root_over_dw"]


def test_evaluate_run_t2_has_the_stage1_error_and_g_s(runs):
    d, cfg = runs["t2_root"]
    job = {"wave": "t2j", "arm": "jt2", "q": 50, "seed": 50500, "T": 2, "run": cfg["run"], "run_dir": d,
           "kind": "joint", "arrays_dir": None}
    row = te.evaluate_run(job)
    assert row["gate_G_S"] in (True, False) and 0 <= row["stage1_rel_err_abs"]
    assert row["stage1_rel_err_abs"] == abs(row["stage1_rel_err_signed"])
    from pathlib import Path
    cand = te.load_joint(Path(d), cfg)
    ev = te.evaluate_tier(cand, FINAL_CONFIG)
    assert row["stage1_rel_err_abs"] == ev.scalars["stage1_rel_err_abs"]


def test_a_failed_or_missing_run_is_a_row_without_metrics(tmp_path):
    d = tmp_path / "failed"
    d.mkdir()
    json.dump({"state": "failed", "exit_code": 1}, open(d / "status.json", "w"))
    job = {"wave": "t3", "arm": "jt", "q": 50, "seed": 50501, "T": 3, "run": "r", "run_dir": str(d), "kind": "joint", "arrays_dir": None}
    row = te.evaluate_run(job)
    assert row["status"] == "failed" and row["exit_code"] == 1 and "Gmax_full_over_dw" not in row
    job["run_dir"] = str(tmp_path / "nothing")
    row = te.evaluate_run(job)
    assert row["status"] == "missing" and row["exit_code"] is None and "Gmax_full_over_dw" not in row


def test_v20_loader_works_on_a_synthetic_export_pair(runs, tmp_path):
    """stage 2 from u01600, stage 1 from u02200 (two exports of the reduced T = 2 run stand in)."""
    from pathlib import Path
    src, cfg = runs["t2_root"]
    d = tmp_path / "v20"
    (d / "weights").mkdir(parents=True)
    shutil.copy(os.path.join(src, "weights", "u00010.npz"), d / "weights" / "u01600.npz")
    shutil.copy(os.path.join(src, "weights", "u00030.npz"), d / "weights" / "u02200.npz")
    json.dump(cfg, open(d / "run_config.json", "w"))
    json.dump({"state": "done", "exit_code": 0}, open(d / "status.json", "w"))
    json.dump({"global_rng": {"status": "ok"}}, open(d / "gates.json", "w"))
    cand = te.load_v20(d, cfg)
    assert sorted(cand.nets) == [1, 2] and cand.nets[1] is not cand.nets[2]
    mean_fn, _ = candidate_fns(cand.spec, cand.nets)
    ref10 = te.load_joint_export(Path(src), cfg, 10)
    ref30 = te.load_joint_export(Path(src), cfg, 30)
    m10, _ = candidate_fns(ref10.spec, ref10.nets)
    m30, _ = candidate_fns(ref30.spec, ref30.nets)
    dd = np.linspace(-200, 200, 41)
    assert np.array_equal(mean_fn(2, dd), m10(2, dd)) and np.array_equal(mean_fn(1, np.zeros(1)), m30(1, np.zeros(1)))
    row = te.evaluate_run({"wave": "v20", "arm": "v20", "q": 50, "seed": 30501, "T": 2, "run": "v", "run_dir": str(d),
                           "kind": "v20", "arrays_dir": None})
    assert row["status"] == "done" and row["gate_G_S"] in (True, False)
    assert row["eT_at_0"] == float(m10(2, np.zeros(1))[0]) and row["e1_at_0"] == float(m30(1, np.zeros(1))[0])


def test_plan_jobs_counts_and_directories(tmp_path):
    jobs = te.plan_jobs(tmp_path, ["t3", "t2j"], tmp_path / "v20", None)
    kinds = [j["kind"] for j in jobs]
    assert len(jobs) == 200 and kinds.count("bw") == 40 and kinds.count("joint") == 120 and kinds.count("v20") == 40
    j = [j for j in jobs if j["arm"] == "jt2" and j["q"] == 60 and j["seed"] == 30520][0]
    assert j["run_dir"] == str(tmp_path / "runs_t2" / "q60" / "seed30520" / "jt2")
    assert [j for j in jobs if j["arm"] == "v20" and j["q"] == 50][0]["run_dir"] == str(tmp_path / "v20" / "q50" / "seed30501")
    sm = te.plan_jobs(tmp_path, ["smoke"], None, None)
    assert len(sm) == 8 and sm[0]["run_dir"] == str(tmp_path / "smoke" / "q50" / "bw")


# ---------------------------------------------------------------------------------- best of run
def _fake_run(d, checks, final_dev, U, valid_final=True):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "joint_checks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["update", "valid", "Gmax_full_over_dw", "error"])
        for u, g, ok, err in checks:
            w.writerow([u, ok, g, err])
    json.dump({"updates": U, "final_eval": {"development": {"valid": valid_final, "Gmax_full_over_dw": final_dev}}},
              open(d / "joint_run_summary.json", "w"))


def test_best_of_run_selection_rules(tmp_path):
    # smallest development Gmax wins; ties go to the earliest update
    _fake_run(tmp_path / "a", [(25, 0.04, True, ""), (50, 0.02, True, ""), (75, 0.02, True, "")], 0.03, 80)
    c = te.best_candidates(tmp_path / "a")
    assert sorted(u for _, u, _ in c) == [25, 50, 75, 80]
    assert te.select_best(c) == (0.02, 50, "check")
    # the last iterate is a candidate (here the smallest)
    _fake_run(tmp_path / "b", [(25, 0.04, True, ""), (50, 0.02, True, "")], 0.01, 60)
    assert te.select_best(te.best_candidates(tmp_path / "b")) == (0.01, 60, "last_iterate")
    # a check whose evaluation is invalid or raised an error is excluded, even with the smallest value
    _fake_run(tmp_path / "c", [(25, 0.001, False, ""), (50, 0.002, True, "DomainError: x"), (75, 0.05, True, "")], 0.06, 80)
    assert te.select_best(te.best_candidates(tmp_path / "c")) == (0.05, 75, "check")
    # the last iterate at a check update is not counted twice
    _fake_run(tmp_path / "d", [(25, 0.04, True, ""), (50, 0.03, True, "")], 0.03, 50)
    c = te.best_candidates(tmp_path / "d")
    assert sorted(u for _, u, _ in c) == [25, 50] and te.select_best(c) == (0.03, 50, "check")
    # an invalid last iterate is excluded
    _fake_run(tmp_path / "e", [(25, 0.04, True, "")], 0.001, 30, valid_final=False)
    assert te.select_best(te.best_candidates(tmp_path / "e")) == (0.04, 25, "check")
    assert te.select_best([]) is None


def test_a_check_file_without_any_successful_evaluation_has_no_valid_candidate(tmp_path):
    """When every development evaluation failed the CSV holds the base columns only: no candidate, no crash."""
    d = tmp_path / "bad"
    d.mkdir()
    with open(d / "joint_checks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["update", "rows_t1", "verifier_sec", "error"])
        w.writerow([25, 100, 0.1, "DomainError: x"])
        w.writerow([50, 200, 0.1, "ValueError: y"])
    assert [c["valid"] for c in te.read_checks(d)] == [False, False]
    json.dump({"updates": 60, "final_eval": {"development": {"valid": False, "Gmax_full_over_dw": float("nan")}}},
              open(d / "joint_run_summary.json", "w"))
    assert te.best_candidates(d) == [] and te.select_best([]) is None


def test_a_missing_reference_file_stops_the_evaluation(tmp_path):
    from pathlib import Path
    miss = te.missing_reference_files(tmp_path)
    assert len(miss) == 2 * 20 * len(te.V20_FILES)
    # one v20 directory complete, the rest missing
    d = tmp_path / "q50" / "seed30501"
    (d / "weights").mkdir(parents=True)
    for f in te.V20_FILES:
        (d / f).write_text("x")
    assert len(te.missing_reference_files(tmp_path)) == len(miss) - len(te.V20_FILES)
    with pytest.raises(SystemExit):
        te.main(["--root", str(tmp_path), "--waves", "t2j", "--out-dir", str(tmp_path / "o")])          # no --v20-root
    with pytest.raises(SystemExit):
        te.main(["--root", str(tmp_path), "--waves", "t3", "--v20-root", str(tmp_path), "--out-dir", str(tmp_path / "o")])
    assert not Path(tmp_path / "o").exists()


@pytest.mark.parametrize("name", ["t3_es", "t2_root"])
def test_best_of_run_on_a_reduced_run_is_the_smallest_development_check(runs, name):
    d, cfg = runs[name]
    arm, T, _ = H.SPECS[name]
    checks = list(csv.DictReader(open(os.path.join(d, "joint_checks.csv"))))
    want_u, want_g = min(((int(c["update"]), float(c["Gmax_full_over_dw"])) for c in checks), key=lambda x: (x[1], x[0]))[0], \
        min(float(c["Gmax_full_over_dw"]) for c in checks)
    row = te.evaluate_best({"wave": "t3", "arm": arm, "q": 50, "seed": 50500, "T": T, "run": cfg["run"], "run_dir": d,
                            "kind": "joint", "arrays_dir": None})
    assert row["best_update"] == want_u and row["n_candidates"] == 3
    assert row["dev_Gmax_full_over_dw"] == want_g           # re-evaluation of the export equals the in-run check
    assert row["label"] == "selected with the development verifier; not a pre-specified candidate"
    assert "gates_pass" in row
