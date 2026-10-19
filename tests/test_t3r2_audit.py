"""T3-R2: tests of the audit tool ``tools/t3/t3r2_audit.py`` (A1-A3 comparisons, the A7 analysis on a planted zero-effort
stretch, the helpers) and of ``tools/t3/t3r2_make_lock.py``.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_t3r2_audit.py -p no:cacheprovider -q
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "t3", ROOT / "tools" / "ms", ROOT / "tests"):
    sys.path.insert(0, str(_p))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import t3_helpers as H  # noqa: E402
import t3_eval as te  # noqa: E402
import t3r2_audit as au  # noqa: E402
import t3r2_br_calc as bc  # noqa: E402
from utils.theory_multistage import g2_two_stage  # noqa: E402

W_H, W_L, K, Q = 6.0, 2.0, 1.0 / 3500.0, 50.0
GAME = bc.Game(W_H, W_L, K, Q, 3)
COARSE = bc.CalcConfig(table_step=0.1, effort_step=0.05, panel_width=1.0, nodes_per_panel=4)
C_BETA = 100.7


# ====================================================================== pure helpers
def test_stretch_nodes_are_the_in_band_nodes_below_effort_one():
    grid = np.arange(-120.0, 121.0, 2.0)
    e = np.full(grid.size, 20.0)
    e[(grid >= -30) & (grid <= -10)] = 0.4
    e[grid == 100.0] = 0.2                         # exactly on the band edge |d| = 2q: out of band
    e[grid == -110.0] = 0.2                        # outside the band
    idx = au.stretch_nodes(grid, e, Q)
    assert grid[idx].tolist() == list(np.arange(-30.0, -9.0, 2.0))
    assert au.runs_of_indices(idx) == [(int(idx[0]), int(idx[-1]))]
    assert au.runs_of_indices(np.array([2, 3, 7, 8, 9, 12])) == [(2, 3), (7, 9), (12, 12)]


def test_first_below_and_the_export_from_which_a_node_stays_below():
    e = np.array([[5.0, 5.0, 0.5, 5.0],
                  [5.0, 0.5, 0.4, 5.0],
                  [0.9, 5.0, 0.3, 5.0],
                  [0.8, 0.7, 0.2, 5.0],
                  [0.7, 0.6, 0.1, 0.9]])
    first, stay = au.first_below(e)
    assert first.tolist() == [2, 1, 0, 4]
    assert stay.tolist() == [2, 3, 0, 4]            # node 1 dips at export 1, rises at 2, is below again from 3
    never, stay2 = au.first_below(np.full((4, 2), 3.0))
    assert never.tolist() == [-1, -1] and stay2.tolist() == [-1, -1]


def test_local_maxima_include_the_end_points():
    e = np.linspace(0, 10, 11)
    assert [m[0] for m in au.local_maxima(e, np.array([5.0, 4, 3, 4, 5, 6, 5, 4, 3, 2, 1]))] == [0.0, 5.0]
    assert [m[0] for m in au.local_maxima(e, np.arange(11.0))] == [10.0]
    assert [m[0] for m in au.local_maxima(e, -np.arange(11.0))] == [0.0]


def test_beta_sd_and_ms_windows():
    assert au.beta_sd_norm(np.array([50.0]), np.array([50.0]))[0] == pytest.approx(np.sqrt(0.25 / 101.0))
    rows = [{"stage": "2", "local": str(i), "kl_final_epoch": "0.01", "clip_frac": "0.1", "adv_raw_mean": "0.0",
             "adv_raw_std": "2.0", "n_start_tail": "100", "n_start_near": "12", "n_start_mid": "400", "d1_L_s2_n": "512",
             "d1_L_s2_lo": str(i)} for i in range(1, 11)] + [{"stage": "3", "local": "1"}]
    w = au.ms_windows(rows, 1, 5)
    assert w["updates"] == 5 and w["learner_draws"] == 5 * 512 and w["learner_draws_below_clamp"] == 15
    assert w["share_below_clamp"] == pytest.approx(15 / 2560) and w["starts_tail"] == 500
    assert au.ms_windows(rows, 20, 30) == {}


def test_compare_text_flags_a_single_differing_cell():
    row = {"a": 1.5, "b": True, "c": None, "d": float("nan")}
    stored = {"a": "1.5", "b": "True", "c": "", "d": "nan"}
    assert au.compare_text(row, stored, ["a", "b", "c", "d"]) == []
    stored["a"] = "1.5000000000000002"
    assert au.compare_text(row, stored, ["a", "b", "c", "d"]) == ["a"]


# ====================================================================== the A7 core on a planted stretch
def planted(j0: int, n_exports: int = 8):
    """Synthetic stage-3 policy (the closed-form tent) and stage-2 exports with a zero-effort stretch on d in [-40, -20]
    (and on d <= -180) from export ``j0`` on; before it the same nodes have the ordinary effort."""
    def mean3(d):
        return g2_two_stage(np.asarray(d, dtype=float), Q, W_H, W_L, K, 100.0)

    def base(d):
        d = np.asarray(d, dtype=float)
        return 25.0 + 15.0 * np.exp(-(d / 60.0) ** 2)

    exports = []
    for j in range(n_exports):
        def mean2(d, j=j):
            d = np.asarray(d, dtype=float)
            out = base(d)
            planted_mask = ((d >= -40.0) & (d <= -20.0)) | (d <= -180.0)
            return np.where(planted_mask & (j >= j0), 0.3, np.where(planted_mask, max(1.5, 5.0 - 0.5 * (j0 - j)), out))

        def beta2(d, mean2=mean2):
            m = np.asarray(mean2(d), dtype=float) / 100.0
            return m * C_BETA, (1.0 - m) * C_BETA
        exports.append((1625 + 25 * j, mean2, beta2))
    return mean3, exports


def test_the_planted_zero_effort_stretch_is_detected_with_its_formation_slope_and_class():
    mean3, exports = planted(j0=4)
    grid = bc.sym_grid(GAME.B, 2.0)
    res = au.analyze_nodes(GAME, mean3, exports, grid, None, COARSE)
    nodes = res["nodes"]
    assert nodes.tolist() == list(np.arange(-40.0, -19.0, 2.0))          # the nodes d <= -180 are out of band
    assert res["summary"]["stretches"] == [(-40.0, -20.0)]
    assert res["summary"]["n_nodes"] == 11
    assert all(f["first_below1_update"] == 1625 + 25 * 4 and f["stays_below1_from_update"] == 1625 + 25 * 4 for f in res["formation"])
    assert res["summary"]["n_below1_by_export"] == [0, 0, 0, 0, 11, 11, 11, 11]
    # series: one row per (export, node); alpha < 1 where the effort is 0.3 (a mean 0.003 with concentration 100.7)
    ser = [s for s in res["series"] if s["update"] == exports[-1][0]]
    assert len(res["series"]) == 8 * 11 and all(s["alpha"] < 1.0 and s["e_hat"] == 0.3 for s in ser)
    # objective: at the final export the opponent plays base(-d); the calculator's own report must agree
    calc = bc.BRCalculator(GAME, lambda t, d: mean3(d) if t == 3 else exports[-1][1](d), COARSE)
    sv = calc.stage_values(2, nodes, brent=False)
    fin = {r["d"]: r for r in res["objective"] if r["export"] == "final"}
    assert len(fin) == 11
    for i, d in enumerate(nodes):
        r = fin[float(d)]
        assert r["slope_at_e_hat"] == pytest.approx(float(sv["slope"][i]), rel=1e-9)
        assert r["a_dev"] == pytest.approx(float(sv["a_dev"][i]), abs=1e-9)
        assert r["slope_at_e_hat"] > 0.0 and not r["local_max_near_zero"]
        assert r["class"] == ("equilibrium_like" if r["a_dev"] < 1.0 else "policy_error")
    assert res["summary"]["n_policy_error"] + res["summary"]["n_equilibrium_like"] == 11
    assert res["summary"]["n_positive_slope_final"] == 11 and res["summary"]["n_local_max_near_zero_final"] == 0
    # the first-below-1 export is analysed too: the opponent there is the earlier export's policy
    first = [r for r in res["objective"] if r["export"] == "first_below1"]
    assert len(first) == 0 or all(r["update"] == 1625 + 25 * 4 for r in first)       # j0 = 4 is not the last export


def test_a_node_far_behind_with_zero_effort_is_equilibrium_like():
    mean3, exports = planted(j0=2, n_exports=5)
    grid = bc.sym_grid(GAME.B, 2.0)
    res = au.analyze_nodes(GAME, mean3, exports, grid, nodes=[-30.0, -190.0], cfg=COARSE)
    by_d = {r["d"]: r for r in res["objective"] if r["export"] == "final"}
    assert by_d[-190.0]["class"] == "equilibrium_like" and by_d[-190.0]["a_dev"] < 1.0
    assert by_d[-30.0]["class"] == "policy_error" and by_d[-30.0]["a_dev"] > 1.0
    assert res["summary"]["n_equilibrium_like"] == 1 and res["summary"]["n_policy_error"] == 1


def test_a_node_that_is_never_below_one_has_no_formation_and_no_first_export_row():
    mean3, exports = planted(j0=99, n_exports=4)               # the stretch never forms; the last export has base effort
    grid = bc.sym_grid(GAME.B, 2.0)
    res = au.analyze_nodes(GAME, mean3, exports, grid, nodes=[-30.0], cfg=COARSE)
    assert res["formation"][0]["first_below1_update"] is None and res["formation"][0]["n_exports_below1"] == 0
    assert [r["export"] for r in res["objective"]] == ["final"]


# ====================================================================== A2, A3 on a reduced bw run
@pytest.fixture(scope="module")
def reduced_run(tmp_path_factory):
    """A reduced legacy ``bw`` run (budgets 40 / 40 / 20, exports every 20 updates)."""
    from run import run_ms_stagewise as rms

    d = tmp_path_factory.mktemp("t3r2_audit_bw")
    cfg = H.bw_cfg(str(d))
    cfg["record"]["protocol"]["weights_every"] = 20
    assert rms.run_pipeline(cfg, str(d), "pytest") == 0
    return d, cfg


def test_snapshot_identity_a2_on_a_reduced_run(reduced_run):
    d, cfg = reduced_run
    job = {"q": 50, "seed": 50500, "run_dir": str(d), "exports": ("u00040", "u00080", "u00100")}
    r = au.a2_one(job)
    assert r["all_identical"] and r["frozen3_eq_u01600"] and r["frozen2_eq_u03200"] and r["live_eq_u03800"]
    # the wrong export is detected
    bad = au.a2_one({**job, "exports": ("u00020", "u00080", "u00100")})
    assert not bad["frozen3_eq_u01600"] and not bad["all_identical"]


def test_continuation_tables_a3_on_a_reduced_run(reduced_run):
    d, cfg = reduced_run
    r = au.a3_one({"q": 50, "seed": 50500, "run_dir": str(d)})
    assert r["table2_bit_identical"] and r["table1_bit_identical"]
    assert r["table2_max_diff_over_dw"] < 1e-5 and r["table1_max_diff_over_dw"] < 1e-5
    assert not r["table2_flag"] and not r["table1_flag"]
    # a table that differs in the last bit is not bit-identical
    z = dict(np.load(d / "continuation_table_stage2.npz"))
    z["values"] = z["values"].copy()
    z["values"][0] = np.nextafter(z["values"][0], np.inf)
    assert not au.same_array(z["values"], np.load(d / "continuation_table_stage2.npz")["values"])


def test_t3_evaluation_of_a_run_is_reproduced_by_the_calculator_within_resolution(reduced_run):
    d, cfg = reduced_run
    job = {"wave": "t3", "arm": "bw", "q": 50, "seed": 50500, "T": 3, "run": "r", "run_dir": str(d), "kind": "bw",
           "arrays_dir": None}
    row = te.evaluate_run(job)
    cfgd, cand, game, mean_fn = au.load_run(job)
    calc = bc.BRCalculator(game, mean_fn)
    sv = calc.stage_values(int(row["Gmax_full_t"]), [float(row["Gmax_full_d"])])
    assert abs(float(sv["G"][0]) / game.dw - float(row["Gmax_full_over_dw"])) < 1e-3
    assert abs(calc.root()["EXP_root"] / game.dw - float(row["EXP_root_over_dw"])) < 1e-3


# ====================================================================== the lock
def test_the_lock_is_deterministic_and_carries_no_timestamp_commit_or_absolute_path(tmp_path):
    import t3r2_make_lock as ml

    a, b = ml.build_lock(), ml.build_lock()
    assert json.dumps(a, indent=1) == json.dumps(b, indent=1)
    text = json.dumps(a, indent=1) + ml.render_md(a)
    assert not re.search(r"\b[0-9a-f]{40}\b", text), "a 40-hex commit hash"
    assert not re.search(r"\b20\d\d-\d\d-\d\d\b", text), "a date"
    assert "/home/" not in text and str(ROOT) not in text
    assert a["confirmation"]["rule"]["min_total_passes"] == 36 and a["confirmation"]["rule"]["min_passes_per_q"] == 17
    assert a["confirmation"]["seed_block"] == [60501, 60520] and a["confirmation"]["n_runs"] == 40
    assert set(a["reference_config_sha256"]) == {"50", "60"}
    assert all(f in a["tool_sha256"] for f in ("tools/t3/t3_eval.py", "tools/t3/t3_metrics.py", "tools/t3/t3_configs.py",
                                               "tools/t3/t3_launch.py", "tools/ms/ms_configs.py", "tools/t3/t3c_launch.py"))
    out = tmp_path / "lock"
    assert ml.main(["--out", str(out)]) == 0
    assert (out / "v2_T3_locked_v1_0.json").read_text() == json.dumps(a, indent=1) + "\n"
    assert (out / "v2_T3_locked_v1_0.md").read_text() == ml.render_md(a)
