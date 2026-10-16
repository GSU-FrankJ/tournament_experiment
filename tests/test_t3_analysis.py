"""T3-R1: tests of ``tools/t3/t3_analysis.py`` and ``tools/t3/t3_blind.py`` on synthetic per-run tables with known answers
(including a failed run and a missing pair), the blind recomputation against the analysis, and the run-file readers
(coverage, cost, curves, figures) on the reduced runs.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_t3_analysis.py -p no:cacheprovider -q
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import t3_helpers as H
import t3_analysis as ta  # noqa: E402
import t3_blind as tb  # noqa: E402
import t3_eval as te  # noqa: E402
import t3_metrics as tm  # noqa: E402

CRIT, _ = tm.load_criteria()
SEEDS3, SEEDS2 = list(range(50501, 50521)), list(range(30501, 30521))
FIN_KEYS = ("eta_T_over_dw", "rmse_pos_T_over_eT0", "tail_mean_T_over_eT0", "Gmax_full_over_dw", "stage1_rel_err_abs")


def make_row(arm, q, seed, T, gmax, eta=0.001, rmse=0.02, tail=0.005, s1=0.01, dev_shift=1e-4, exp_root=0.001, **extra):
    fin = {"eta_T_over_dw": eta, "rmse_pos_T_over_eT0": rmse, "tail_mean_T_over_eT0": tail, "Gmax_full_over_dw": gmax,
           "stage1_rel_err_abs": s1}
    dev = {"eta_T_over_dw": eta + dev_shift, "Gmax_full_over_dw": gmax + dev_shift}
    row = {"wave": "t3" if T == 3 else ("v20" if arm == "v20" else "t2j"), "arm": arm, "q": q, "seed": seed, "T": T,
           "run": f"{arm}_{q}_{seed}", "run_dir": "", "kind": "x", "status": "done", "exit_code": 0, "global_rng_ok": True,
           "eval_error": "", **fin, **{f"dev_{k}": v for k, v in dev.items()},
           "EXP_root_over_dw": exp_root, "dReach_over_dw": gmax / 2, "peak_T_abs": 0.03, "peak_T_signed": -0.03, "e1_at_0": 40.0,
           "Gmax_full_t": 3, "Gmax_full_d": 0.0}
    for t in range(1, T + 1):
        row[f"G_max_t{t}_over_dw"] = gmax / t
        row[f"Delta_max_t{t}_over_dw"] = gmax / (2 * t)
    for t in range(2, T + 1):
        row.update({f"G_in_t{t}": gmax / 3, f"G_out_t{t}": gmax, f"Delta_in_t{t}": gmax / 4, f"Delta_out_t{t}": gmax / 2})
    row.update(extra)
    row.update(tm.gate_columns(fin, dev, T, CRIT))
    return row


def synth_table(rng, bw_scale=0.004, jt_scale=0.03, jes_scale=0.02, fail=None, drop=None, constant=False):
    rows = []
    scale = {"bw": bw_scale, "jt": jt_scale, "jt_es": jes_scale, "v20": bw_scale, "jt2": jt_scale}
    for arm in ("bw", "jt", "jt_es", "v20", "jt2"):
        T = 3 if arm in ("bw", "jt", "jt_es") else 2
        for q in (50, 60):
            for s in (SEEDS3 if T == 3 else SEEDS2):
                g = scale[arm] * (1.0 if constant else (1.0 + 0.3 * rng.standard_normal()))
                rows.append(make_row(arm, q, s, T, abs(g)))
    for (arm, q, s) in (fail or []):
        for r in rows:
            if (r["arm"], r["q"], r["seed"]) == (arm, q, s):
                for k in list(r):
                    if k not in ("wave", "arm", "q", "seed", "T", "run", "run_dir", "kind", "global_rng_ok", "eval_error"):
                        r[k] = float("nan")
                r["status"], r["exit_code"] = "failed", 1
    for key in (drop or []):
        rows = [r for r in rows if (r["arm"], r["q"], r["seed"]) != key]
    return rows


def write_table(rows, path):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


# ====================================================================== criterion table, known answers
def test_hypothesis_rows_with_constant_differences_have_exact_answers(tmp_path):
    rng = np.random.default_rng(0)
    rows = synth_table(rng, bw_scale=0.01, jt_scale=0.03, jes_scale=0.03, constant=True)
    for r in rows:      # q = 60: bw equals jt (difference 0)
        if r["arm"] == "bw" and r["q"] == 60:
            r["Gmax_full_over_dw"] = 0.03
    p = write_table(rows, tmp_path / "per_run.csv")
    df = ta.load_per_run(p)
    c = ta.criterion_table(df, CRIT).set_index(["hypothesis", "q"])
    h1_50, h1_60 = c.loc[("H1", 50)], c.loc[("H1", 60)]
    assert h1_50["n_pairs"] == 20 and h1_50["dropped_pairs"] == 0
    for k in ("mean_diff", "ci95_lo", "ci95_hi", "ci98_75_lo", "ci98_75_hi", "median_diff", "median_ci95_lo", "median_ci95_hi"):
        assert h1_50[k] == pytest.approx(-0.02, abs=1e-12), k
    assert h1_50["n_neg"] == 20 and bool(h1_50["ci_hi_below_0"]) and h1_50["bootstrap_seed"] == 20261016
    assert h1_60["mean_diff"] == pytest.approx(0.0, abs=1e-12) and not bool(h1_60["ci_hi_below_0"])
    assert not bool(h1_50["hypothesis_supported"]) and h1_50["fails_at_q"] == "60"
    h3 = c.loc[("H3", 50)]                      # jt_es - jt = 0
    assert h3["mean_diff"] == pytest.approx(0.0, abs=1e-12) and not bool(h3["ci_hi_below_0"])
    h2_50 = c.loc[("H2", 50)]                   # bw - jt_es = -0.02
    assert h2_50["mean_diff"] == pytest.approx(-0.02, abs=1e-12) and bool(h2_50["ci_hi_below_0"])
    t2 = c.loc[("H1-T2", 50)]                   # v20 - jt2 = 0.01 - 0.03, seed 20261017
    assert t2["bootstrap_seed"] == 20261017 and t2["mean_diff"] == pytest.approx(-0.02, abs=1e-12)
    assert bool(t2["hypothesis_supported"]) and t2["fails_at_q"] == ""
    assert c.loc[("H1", 50)]["horizon"] == 3 and t2["horizon"] == 2


def test_a_failed_run_and_a_missing_pair_are_dropped_and_counted():
    rng = np.random.default_rng(1)
    rows = synth_table(rng, fail=[("jt", 50, 50503), ("v20", 60, 30502)], drop=[("bw", 60, 50505, ), ])
    df = ta.load_per_run(write_table(rows, Path(H._tmp()) / "fail_per_run.csv"))
    c = ta.criterion_table(df, CRIT).set_index(["hypothesis", "q"])
    assert c.loc[("H1", 50)]["n_pairs"] == 19 and c.loc[("H1", 50)]["dropped_pairs"] == 1       # jt failed
    assert c.loc[("H1", 60)]["n_pairs"] == 19 and c.loc[("H1", 60)]["dropped_pairs"] == 1       # bw row missing
    assert c.loc[("H2", 60)]["n_pairs"] == 19 and c.loc[("H3", 50)]["n_pairs"] == 19
    assert c.loc[("H1-T2", 60)]["n_pairs"] == 19 and c.loc[("H1-T2", 50)]["n_pairs"] == 20
    counts = ta.gate_counts(df).set_index(["arm", "q"])
    assert counts.loc[("jt", 50)]["n_exit_ok"] == 19 and counts.loc[("bw", 60)]["n_exit_ok"] == 19
    assert counts.loc[("bw", 60)]["n_planned"] == 20
    assert counts.loc[("bw", 60)]["run_pass"] == 19       # the missing run counts as not passed
    # the pair of the failed run is dropped, the other 19 are used
    x = ta.series(df, "jt", 50, "Gmax_full_over_dw")
    assert math.isnan(x[2]) and np.isfinite(np.delete(x, 2)).all()


def test_blind_recomputation_agrees_with_the_analysis_on_random_data(tmp_path):
    rng = np.random.default_rng(2)
    rows = synth_table(rng, fail=[("jt_es", 60, 50511), ("jt2", 50, 30519)], drop=[("v20", 50, 30504)])
    # make a few runs fail gates on purpose
    for r in rows:
        if r["arm"] == "bw" and r["seed"] in (50502, 50509):
            r["rmse_pos_T_over_eT0"] = 0.06
            r.update(tm.gate_columns({k: r[k] for k in ("eta_T_over_dw", "rmse_pos_T_over_eT0", "tail_mean_T_over_eT0", "Gmax_full_over_dw", "stage1_rel_err_abs")},
                                     {"eta_T_over_dw": r["dev_eta_T_over_dw"], "Gmax_full_over_dw": r["dev_Gmax_full_over_dw"]}, 3, CRIT))
    p = write_table(rows, tmp_path / "per_run.csv")
    df = ta.load_per_run(p)
    a_dir = tmp_path / "analysis"
    a_dir.mkdir()
    ta.criterion_table(df, CRIT).to_csv(a_dir / "criterion.csv", index=False)
    ta.gate_counts(df).to_csv(a_dir / "gate_counts.csv", index=False)
    out = tmp_path / "blind"
    assert tb.main(["--per-run", str(p), "--out-dir", str(out), "--analysis-dir", str(a_dir)]) == 0
    brows = tb.criterion_rows(tb.read_table(p), CRIT)
    assert len(brows) == 8 and {r["hypothesis"] for r in brows} == {"H1", "H2", "H3", "H1-T2"}
    worst, n_cmp, missing = tb.compare(brows, a_dir / "criterion.csv", ("hypothesis", "q"))
    assert n_cmp > 80 and worst <= 1e-12 and not [m for m in missing if m.startswith("row ")]
    g = list(csv.DictReader(open(out / "blind_gate_counts.csv")))
    bw50 = [r for r in g if r["arm"] == "bw" and r["q"] == "50"][0]
    assert int(bw50["G-A_pass"]) == 18 and int(bw50["run_pass"]) == 18      # seeds 50502, 50509 fail G-A (rmse 0.06)
    gc = ta.gate_counts(df).set_index(["arm", "q"])
    for r in g:
        a = gc.loc[(r["arm"], int(r["q"]))]
        for k in ("n_exit_ok", "G-A_pass", "G-F_pass", "G-N_pass", "run_pass"):
            assert int(r[k]) == int(a[k]), (r["arm"], r["q"], k)
        for k in ("run_pass_lo", "run_pass_hi"):
            assert abs(float(r[k]) - float(a[k])) <= 1e-12


def test_gate_counts_and_the_clopper_pearson_intervals_and_the_reading_rule(tmp_path):
    rng = np.random.default_rng(3)
    rows = synth_table(rng, bw_scale=0.003, constant=True)
    # bw at q = 50: two runs fail (one G-A, one G-F) -> 18 of 20 pass; at q = 60: 17 pass
    def fail_ga(r):
        r["rmse_pos_T_over_eT0"] = 0.06
    def fail_gf(r):
        r["Gmax_full_over_dw"] = 0.02
        r["dev_Gmax_full_over_dw"] = 0.02
    for r in rows:
        if r["arm"] == "bw" and r["q"] == 50 and r["seed"] == 50501:
            fail_ga(r)
        if r["arm"] == "bw" and r["q"] == 50 and r["seed"] == 50502:
            fail_gf(r)
        if r["arm"] == "bw" and r["q"] == 60 and r["seed"] in (50501, 50502, 50503):
            fail_gf(r)
        if r["arm"] == "bw" and r["seed"] in (50501, 50502, 50503):
            fin = {k: r[k] for k in FIN_KEYS}
            r.update(tm.gate_columns(fin, {"eta_T_over_dw": r["dev_eta_T_over_dw"], "Gmax_full_over_dw": r["dev_Gmax_full_over_dw"]}, 3, CRIT))
    df = ta.load_per_run(write_table(rows, tmp_path / "p.csv"))
    gc = ta.gate_counts(df).set_index(["arm", "q"])
    b50, b60 = gc.loc[("bw", 50)], gc.loc[("bw", 60)]
    assert (b50["G-A_pass"], b50["G-F_pass"], b50["G-N_pass"], b50["run_pass"]) == (19, 19, 20, 18)
    assert b50["run_pass_lo"] == pytest.approx(0.6830172859809177, abs=1e-12)
    assert b50["run_pass_hi"] == pytest.approx(0.9876514728297052, abs=1e-12)
    assert b60["run_pass"] == 17 and gc.loc[("v20", 50)]["G-S_pass"] == 20 and pd.isna(gc.loc[("bw", 50)]["G-S_pass"])
    rr = ta.reading_rule(ta.gate_counts(df), CRIT)
    assert rr == {"min_passes": 18, "passes_by_q": {50: 18, 60: 17}, "candidate_for_a_lock": False}
    for r in rows:
        if r["arm"] == "bw" and r["q"] == 60:
            r["Gmax_full_over_dw"] = 0.003
            r["dev_Gmax_full_over_dw"] = 0.003
            r.update(tm.gate_columns({k: r[k] for k in FIN_KEYS}, {"eta_T_over_dw": r["dev_eta_T_over_dw"], "Gmax_full_over_dw": 0.003}, 3, CRIT))
    df = ta.load_per_run(write_table(rows, tmp_path / "p2.csv"))
    assert ta.reading_rule(ta.gate_counts(df), CRIT)["candidate_for_a_lock"] is True
    # a run with a global-RNG violation (exit 5) is failed and has no metrics
    rows2 = [dict(r) for r in rows]
    for r in rows2:
        if r["arm"] == "bw" and r["q"] == 60 and r["seed"] == 50510:
            r["status"], r["exit_code"], r["global_rng_ok"], r["Gmax_full_over_dw"] = "done", 5, False, float("nan")
    df2 = ta.load_per_run(write_table(rows2, tmp_path / "p3.csv"))
    assert ta.gate_counts(df2).set_index(["arm", "q"]).loc[("bw", 60)]["run_pass"] == 19


def test_bw_gate_table_diagnoses_every_failure(tmp_path):
    rng = np.random.default_rng(4)
    rows = synth_table(rng, bw_scale=0.003, constant=True)
    for r in rows:
        if r["arm"] == "bw" and r["q"] == 50 and r["seed"] == 50501:
            r["tail_mean_T_over_eT0"] = 0.03
        if r["arm"] == "bw" and r["q"] == 50 and r["seed"] == 50502:
            r.update(Gmax_full_over_dw=0.02, dev_Gmax_full_over_dw=0.02, Gmax_full_t=2, Gmax_full_d=-150.0, G_out_t2=0.02, G_in_t2=0.001)
        if r["arm"] == "bw" and r["q"] == 50 and r["seed"] == 50503:
            r["dev_Gmax_full_over_dw"] = r["Gmax_full_over_dw"] + 0.002
        if r["arm"] == "bw" and r["seed"] in (50501, 50502, 50503):
            fin = {k: r[k] for k in FIN_KEYS}
            r.update(tm.gate_columns(fin, {"eta_T_over_dw": r["dev_eta_T_over_dw"], "Gmax_full_over_dw": r["dev_Gmax_full_over_dw"]}, 3, CRIT))
    rows = [r for r in rows if not (r["arm"] == "bw" and r["q"] == 60 and r["seed"] == 50504)] + [
        dict(make_row("bw", 60, 50504, 3, 0.003), status="failed", exit_code=1)]
    df = ta.load_per_run(write_table(rows, tmp_path / "p.csv"))
    t = ta.bw_gate_table(df).set_index(["q", "seed"])
    assert not t.loc[(50, 50501)]["passed"] and "G-A tail_mean 0.0300 > 0.02" in t.loc[(50, 50501)]["diagnosis"]
    d2 = t.loc[(50, 50502)]["diagnosis"]
    assert "G-F Gmax 0.02000 > 0.01 at stage 2, d = -150 (out of band)" in d2 and "G-N" not in d2
    assert "G-N |Gmax(dev) - Gmax(final)| = 0.00200 > 0.001" in t.loc[(50, 50503)]["diagnosis"]
    assert t.loc[(50, 50510)]["passed"] and t.loc[(50, 50510)]["diagnosis"] == ""
    assert "no result (status failed, exit code 1)" in t.loc[(60, 50504)]["diagnosis"]


def test_secondary_table_covers_the_pairs_and_statistics():
    rng = np.random.default_rng(5)
    rows = synth_table(rng)
    p = Path(H._tmp()) / "sec.csv"
    write_table(rows, p)
    df = ta.load_per_run(p)
    sec = ta.secondary_table(df, CRIT)
    stats3 = ta.secondary_stats(3)
    assert {"EXP_root_over_dw", "Delta_max_t3_over_dw", "G_out_t2", "Delta_in_t3", "dReach_over_dw", "rmse_pos_T_over_eT0",
            "tail_mean_T_over_eT0", "peak_T_abs", "e1_at_0"} <= set(stats3) and "stage1_rel_err_abs" not in stats3
    assert "stage1_rel_err_abs" in ta.secondary_stats(2) and "G_in_t3" not in ta.secondary_stats(2)
    assert len(sec) == (len(stats3) * 3 + len(ta.secondary_stats(2))) * 2
    assert set(sec["hypothesis"]) == {"H1", "H2", "H3", "H1-T2"}
    ex = sec[(sec["hypothesis"] == "H1") & (sec["statistic"] == "EXP_root_over_dw") & (sec["q"] == 50)].iloc[0]
    assert ex["mean_diff"] == pytest.approx(0.0, abs=1e-15)         # EXP_root is 0.001 in both arms


def test_the_secondary_statistics_are_those_of_the_criteria_file():
    sec = CRIT["secondary"]
    for T in (3, 2):
        want = list(sec["scalar"])
        for t in range(1, T + 1):
            want += [x.format(t=t) for x in sec["per_stage"]]
        for t in range(2, T + 1):
            want += [x.format(t=t) for x in sec["per_stage_from_2"]]
        if T == 2:
            want += sec["horizon_2_only"]
        assert sorted(ta.secondary_stats(T)) == sorted(want), T


def test_best_of_run_table():
    best = pd.DataFrame([
        {"arm": "jt", "q": 50, "seed": 50501, "status": "done", "Gmax_full_over_dw": 0.02, "EXP_root_over_dw": 0.001, "best_update": 1000,
         "best_source": "check", "gates_pass": True},
        {"arm": "jt", "q": 50, "seed": 50502, "status": "done", "Gmax_full_over_dw": 0.03, "EXP_root_over_dw": 0.002, "best_update": 1267,
         "best_source": "last_iterate", "gates_pass": False}])
    last = pd.DataFrame([
        {"arm": "jt", "q": 50, "seed": 50501, "exit_ok": True, "Gmax_full_over_dw": 0.04, "EXP_root_over_dw": 0.003},
        {"arm": "jt", "q": 50, "seed": 50502, "exit_ok": True, "Gmax_full_over_dw": 0.03, "EXP_root_over_dw": 0.002}])
    t = ta.best_of_run_table(last, best).iloc[0]
    assert t["n"] == 2 and t["Gmax_full_over_dw_last_mean"] == pytest.approx(0.035) and t["Gmax_full_over_dw_best_mean"] == pytest.approx(0.025)
    assert t["n_last_iterate_selected"] == 1 and t["best_update_mean"] == pytest.approx(1133.5) and t["n_gates_pass_best"] == 1


# ====================================================================== run-file readers on the reduced runs
@pytest.fixture(scope="module")
def reduced_df(tmp_path_factory):
    runs = H.reduced_runs()
    bw_dir, bw_cfg = H.reduced_bw_run()
    arrays = tmp_path_factory.mktemp("arrays")
    rows = []
    for name, (d, cfg) in {"t3_root": runs["t3_root"], "t3_es": runs["t3_es"], "bw": (bw_dir, bw_cfg), "t2_root": runs["t2_root"]}.items():
        arm = cfg["arm"]
        T = cfg["record"]["game"]["T"]
        rows.append(te.evaluate_run({"wave": "t3" if T == 3 else "t2j", "arm": arm, "q": 50, "seed": 50500, "T": T, "run": cfg["run"],
                                     "run_dir": d, "kind": "bw" if arm == "bw" else "joint", "arrays_dir": str(arrays)}))
    df = pd.DataFrame(rows)
    df["exit_ok"] = (df["status"] == "done") & (df["exit_code"] == 0)
    return df, arrays


def test_coverage_cost_and_curves_on_the_reduced_runs(reduced_df):
    df, _ = reduced_df
    cov, arrays = ta.coverage_arrays(Path("."), df)
    jt = cov[(cov["arm"] == "jt_es") & (cov["stage"] == 3)].iloc[0]
    assert jt["rows"] == 48 * 30 and jt["n_bins"] == 80 and 0 < jt["share_in_band"] < 1
    assert jt["rows_direct_es"] == 16 * 30 and jt["rows_root_path"] == 24 * 30 and jt["rows_from_stage2_start_approx"] == 8 * 30
    rj = cov[(cov["arm"] == "jt") & (cov["stage"] == 2)].iloc[0]
    assert rj["rows"] == 48 * 30 and rj["rows_direct_es"] == 0
    bw = cov[(cov["arm"] == "bw") & (cov["stage"] == 3)].iloc[0]
    assert bw["rows"] == 40 * 48 and bw["mean_rows_per_bin_in_band"] == pytest.approx(40 * 48 / 80)
    assert bw["n_start_tail"] + bw["n_start_near"] + bw["n_start_mid"] == 40 * 48
    assert "bw_q50_stage3_direct_design" in arrays and "jt_es_q50_stage3_direct_es" in arrays
    cost = ta.cost_table(df).set_index("arm")
    assert cost.loc["bw", "updates"] == 100 and cost.loc["bw", "policy_rows_total"] == 100 * 48
    assert cost.loc["jt_es", "updates"] == 30 and cost.loc["jt_es", "policy_rows_total"] == (24 + 32 + 48) * 30
    assert cost.loc["jt", "minibatch_steps"] == 30 * 10 * math.ceil(144 / 256)
    assert cost.loc["jt", "train_update_sec"] > 0 and cost.loc["bw", "wall_stage3_sec"] > 0
    jc = ta.joint_curve(Path(df[df["arm"] == "jt"].iloc[0]["run_dir"]))
    assert list(jc["cum_rows"]) == [144 * 10, 144 * 20, 144 * 30]
    bc = ta.bw_curve(Path(df[df["arm"] == "bw"].iloc[0]["run_dir"]))
    assert set(bc["stage"]) == {1, 2, 3} and (bc["cum_rows"] == bc["update"] * 48).all()


def _labels(ax):
    return [h.get_label() for h in ax.get_lines() + [c for c in ax.collections]] + [t.get_text() for t in (ax.get_legend().get_texts() if ax.get_legend() else [])]


def _vlines(ax):
    return sorted(float(l.get_xdata()[0]) for l in ax.get_lines() if len(set(l.get_xdata())) == 1 and l.get_linestyle() == "--")


def test_figures_have_the_pre_registered_content(reduced_df):
    """Effort functions with the closed form (terminal stage) and g1 (stage 1 at T = 2), deviation gains with the band edges,
    the paired differences of Gmax_full, EXP_root and the in / out-of-band maxima, the learning curves of every stage, and the
    visitation with bw's design; the T = 2 row / figure is there. Checked on the reduced runs (q = 50; no v20)."""
    import matplotlib.pyplot as plt
    df, arrays = reduced_df
    fe = ta.figure_effort(df, arrays, 50)
    assert len(fe.axes) == 6 and fe.axes[2].get_title() == "T = 3, stage 3, q = 50" and fe.axes[4].get_title() == "T = 2, stage 2, q = 50"
    txt3 = _labels(fe.axes[2])
    assert "closed form e_T*(d)" in txt3 and {"bw", "jt", "jt_es"} <= set(txt3)          # the closed form at the T = 3 terminal stage
    assert "closed form e_T*(d)" in _labels(fe.axes[4]) and "jt2" in _labels(fe.axes[4])  # and at the T = 2 terminal stage
    assert "g1" in _labels(fe.axes[3]) and "g1" not in _labels(fe.axes[0])               # g1 at stage 1 of the T = 2 row only
    assert not fe.axes[5].axison                                                        # no stage 3 at T = 2
    x, y = [l for l in fe.axes[2].get_lines() if l.get_label() == "closed form e_T*(d)"][0].get_data()
    assert x[0] == -400.0 and x[-1] == 400.0 and abs(float(np.max(y)) - 70.0) < 1e-9     # the tent of q = 50: 70 at d = 0
    fg = ta.figure_gains(df, arrays, 50)
    assert len(fg.axes) == 6 and _vlines(fg.axes[1]) == [-100.0, 100.0] and _vlines(fg.axes[2]) == [-200.0, 200.0]
    assert _vlines(fg.axes[0]) == [] and _vlines(fg.axes[4]) == [-100.0, 100.0]          # stage 1 has no band; T = 2 stage 2 does
    assert any("max" in t for t in _labels(fg.axes[2])) and any("median" in t for t in _labels(fg.axes[2]))
    p = Path(H._tmp()) / "fig_per_run.csv"
    write_table(synth_table(np.random.default_rng(6)), p)
    dfs = ta.load_per_run(p)
    fp = ta.figure_paired(dfs, CRIT)
    titles = [ax.get_title() for ax in fp.axes if ax.axison]
    assert titles == [s for s in ta.PAIRED_STATS], titles
    n_rows = {ax.get_title(): len(ax.get_yticks()) for ax in fp.axes if ax.axison}
    assert n_rows["Gmax_full_over_dw"] == 8 and n_rows["EXP_root_over_dw"] == 8 and n_rows["G_out_t3"] == 6 and n_rows["G_in_t2"] == 8
    for T, ncols_ in ((3, 8), (2, 6)):
        fc = ta.figure_curves(df, 50, T)
        shown = [ax for ax in fc.axes if ax.axison]
        assert len(shown) == ncols_ and shown[0].get_title().startswith("Gmax_full_over_dw") and shown[1].get_title().startswith("EXP_root")
        assert all(ax.get_yscale() == "log" for ax in shown)
        eta3 = [ax for ax in shown if ax.get_title().startswith("Delta_max_t3")]
        if T == 3:
            assert "bw (phase of stage 3)" in _labels(eta3[0]) and "jt_es" in _labels(shown[0]) and "jt" in _labels(shown[0])
            assert not any(l.startswith("bw") for l in _labels(shown[0]))                # bw has only its own Delta
        else:
            assert not eta3 and "jt2" in _labels(shown[0])
    _, arr = ta.coverage_arrays(Path("."), df)
    fv = ta.figure_visitation(arr, 50)
    assert len(fv.axes) == 3 and "bw design (direct starts)" in _labels(fv.axes[0]) and "bw design (direct starts)" in _labels(fv.axes[1])
    assert "jt2 (all sources)" in _labels(fv.axes[2]) and "bw design (direct starts)" not in _labels(fv.axes[2])
    assert _vlines(fv.axes[0]) == [-100.0, 100.0] and _vlines(fv.axes[1]) == [-200.0, 200.0]
    assert any("jt_es: direct_es" in t for t in _labels(fv.axes[1]))
    for f in (fe, fg, fp, fc, fv):
        plt.close(f)


def test_the_intervals_follow_the_percentiles_of_the_criteria_file():
    rng = np.random.default_rng(11)
    x, y = rng.normal(0.02, 0.01, 20), rng.normal(0.03, 0.01, 20)
    base = ta.paired_row(x, y, 20261016, 10000, [2.5, 97.5], [0.625, 99.375])
    wide = ta.paired_row(x, y, 20261016, 10000, [10.0, 90.0], [0.625, 99.375])
    idx = np.random.default_rng(20261016).integers(0, 20, size=(10000, 20))
    means = (x - y)[idx].mean(axis=1)
    assert base["ci95_lo"] == np.percentile(means, 2.5) and base["ci95_hi"] == np.percentile(means, 97.5)
    assert wide["ci95_lo"] == np.percentile(means, 10.0) and wide["ci95_hi"] == np.percentile(means, 90.0)
    assert wide["ci95_lo"] > base["ci95_lo"] and wide["ci95_hi"] < base["ci95_hi"] and base["ci98_75_lo"] == np.percentile(means, 0.625)
    med = np.median((x - y)[idx], axis=1)
    assert base["median_ci95_lo"] == np.percentile(med, 2.5) and base["median_ci95_hi"] == np.percentile(med, 97.5)
    assert base["mean_diff"] == pytest.approx(float(np.mean(x - y)), abs=1e-15) and base["n_neg"] == int(np.sum(x - y < 0))


def test_a_run_with_a_clean_exit_but_no_metrics_is_reported_not_a_crash(tmp_path):
    rng = np.random.default_rng(12)
    rows = synth_table(rng, bw_scale=0.003)
    for r in rows:
        if r["arm"] == "bw" and r["q"] == 50 and r["seed"] == 50502:      # an evaluation error: done, exit 0, no metrics
            for k in list(r):
                if k not in ("wave", "arm", "q", "seed", "T", "run", "run_dir", "kind", "global_rng_ok", "status", "exit_code"):
                    r[k] = float("nan")
            r["eval_error"] = "DomainError: landing outside D_3"
    df = ta.load_per_run(write_table(rows, tmp_path / "p.csv"))
    t = ta.bw_gate_table(df).set_index(["q", "seed"])
    assert t.loc[(50, 50502)]["diagnosis"] == "no result (status done, exit code 0, evaluation error: DomainError: landing outside D_3)"
    assert not t.loc[(50, 50502)]["passed"]
    c = ta.criterion_table(df, CRIT).set_index(["hypothesis", "q"])
    assert c.loc[("H1", 50)]["n_pairs"] == 19 and c.loc[("H1", 50)]["dropped_pairs"] == 1


def test_budget_check_compares_every_run_with_d5():
    tcm = ta.tc
    rows = []
    for arm in ("bw", "jt", "jt_es", "jt2", "v20"):
        e = tcm.expected_budget(arm)
        for seed in (1, 2):
            r = {"arm": arm, "q": 50, "seed": seed, "updates": e["updates"], "policy_rows_total": e["rows_total"],
                 "minibatch_steps": e["minibatch_steps"]}
            r.update({f"rows_t{t}": n for t, n in e["rows_by_stage"].items()})
            rows.append(r)
    cost = pd.DataFrame(rows)
    b = ta.budget_check(cost).set_index("arm")
    assert (b["n_runs"] == 2).all() and (b["updates_equal"] == 2).all() and (b["rows_total_equal"] == 2).all()
    assert (b["rows_by_stage_equal"] == 2).all() and (b["minibatch_steps_equal"] == 2).all()
    assert b.loc["v20", "minibatch_steps_expected"] == 56000 and b.loc["jt_es", "minibatch_steps_expected"] == 87700
    cost.loc[(cost.arm == "v20") & (cost.seed == 2), "minibatch_steps"] = 55999
    cost.loc[(cost.arm == "jt") & (cost.seed == 1), "rows_t2"] -= 1
    b = ta.budget_check(cost).set_index("arm")
    assert b.loc["v20", "minibatch_steps_equal"] == 1 and b.loc["jt", "rows_by_stage_equal"] == 1 and b.loc["bw", "rows_by_stage_equal"] == 2


# ====================================================================== the driver end to end
def _fake_v20_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"phase": "A", "n_epochs_run": 10, "n_rows": 512, "n_policy_rows": 512}] * 1600 + \
           [{"phase": "B", "n_epochs_run": 10, "n_rows": 1024, "n_policy_rows": 512}] * 600
    pd.DataFrame(rows).to_csv(d / "v2_updates.csv", index=False)
    json.dump({"final_global_update": 2200, "total_wall_sec": 350.0,
               "phase_timing": {"A": {"wall_sec": 200.0}, "B": {"wall_sec": 150.0}},
               "costs": {"seconds": {"train_update": 300.0, "train_rollout": 7.0}}}, open(d / "v2_run_summary.json", "w"))


def test_run_analysis_end_to_end_on_a_synthetic_table(tmp_path):
    """Every table and figure of the driver is produced; the run-file readers see the reduced runs (every planned
    row of an arm points at the same reduced run); the headline reads the gate counts."""
    runs = H.reduced_runs()
    bw_dir, _ = H.reduced_bw_run()
    v20_dir = tmp_path / "v20_fake"
    _fake_v20_dir(v20_dir)
    dirs = {"bw": bw_dir, "jt": runs["t3_root"][0], "jt_es": runs["t3_es"][0], "jt2": runs["t2_root"][0], "v20": str(v20_dir)}
    rng = np.random.default_rng(7)
    rows = synth_table(rng, bw_scale=0.003)
    for r in rows:
        r["run_dir"] = dirs[r["arm"]]
        r["kind"] = "bw" if r["arm"] == "bw" else ("v20" if r["arm"] == "v20" else "joint")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    write_table(rows, eval_dir / "per_run.csv")
    best = [{"wave": "t3", "arm": a, "q": q, "seed": s, "T": 3 if a != "jt2" else 2, "run": f"{a}_{q}_{s}", "status": "done",
             "Gmax_full_over_dw": 0.01, "EXP_root_over_dw": 0.001, "best_update": 1000, "best_source": "check", "gates_pass": True}
            for a in ("jt", "jt_es", "jt2") for q in (50, 60) for s in (SEEDS3 if a != "jt2" else SEEDS2)]
    pd.DataFrame(best).to_csv(eval_dir / "best_of_run.csv", index=False)
    out = tmp_path / "analysis"
    head = ta.run_analysis(tmp_path, eval_dir, out, CRIT, figures=True)
    files = ["per_run_table.csv", "criterion.csv", "secondary.csv", "gate_counts.csv", "bw_t3_gates.csv", "best_of_run.csv",
             "coverage.csv", "coverage_arrays.npz", "cost.csv", "budget_check.csv", "headline.json", "figures/paired_differences.png"]
    files += [f"figures/{n}_q{q}.png" for q in (50, 60) for n in ("effort_functions", "deviation_gains", "learning_curves_t3",
                                                                    "learning_curves_t2", "visitation")]
    for f in files:
        assert (out / f).exists() and (f.endswith(".png") is False or (out / f).stat().st_size > 1000), f
    bud = pd.read_csv(out / "budget_check.csv").set_index("arm")
    assert bud.loc["v20", "n_runs"] == 40 and bud.loc["v20", "minibatch_steps_equal"] == 40 and bud.loc["v20", "updates_equal"] == 40
    assert bud.loc["bw", "updates_equal"] == 0 and head["budget_check_all_equal"] is False      # the reduced runs are not D5's
    assert head["reading_rule"]["candidate_for_a_lock"] is True and set(head["hypotheses"]) == {"H1", "H2", "H3", "H1-T2"}
    cost = pd.read_csv(out / "cost.csv")
    v20 = cost[cost["arm"] == "v20"]
    assert len(v20) == 40 and (v20["minibatch_steps"] == 56000).all() and (v20["policy_rows_total"] == 1126400).all()
    assert (v20["rows_t2"] == 819200).all() and (v20["rows_t1"] == 307200).all()
    assert json.load(open(out / "headline.json")) == json.loads(json.dumps(head))
    # the blind tool agrees with the driver's tables
    assert tb.main(["--per-run", str(eval_dir / "per_run.csv"), "--out-dir", str(out / "blind"), "--analysis-dir", str(out)]) == 0
