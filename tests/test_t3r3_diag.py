"""Tests of ``tools/t3/t3r3_diag.py`` (T3-R3 P1): S3, F(u) and the formation export, the saturation share, the retrospective trigger
table, on synthetic inputs."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "ms", ROOT / "tools" / "t3"):
    sys.path.insert(0, str(_p))

import t3r3_diag as dg  # noqa: E402
from envs.curriculum_env import GameSpec  # noqa: E402

SPEC = GameSpec(w_h=6.0, w_l=2.0, k=1.0 / 3500.0, q=50.0, T=3)


# ---------------------------------------------------------------------------------- S3 and the onset window
def _rows(shares):
    """Stage-3 rows for local 1..1600 whose out-of-band clamp counters follow ``shares(local)``, plus stage-2 rows that must be
    ignored."""
    rows = []
    for loc in range(1, 1601):
        n = 100
        lo = int(round(shares(loc) * n))
        rows.append({"stage": "3", "local": str(loc), "d1_L_s3_out_n": str(n), "d1_L_s3_out_lo": str(lo)})
        rows.append({"stage": "2", "local": str(loc), "d1_L_s3_out_n": "100", "d1_L_s3_out_lo": "100"})
    return rows


def test_s3_uses_stage3_rows_of_local_1201_1600_only():
    rows = _rows(lambda loc: 0.5 if loc >= 1201 else 0.0)
    s3, lo, n = dg.s3_of_rows(rows)
    assert (lo, n) == (400 * 50, 400 * 100)
    assert s3 == 0.5
    s3b, lo_b, n_b = dg.s3_of_rows(_rows(lambda loc: 0.3 if loc < 1201 else 0.1))
    assert (lo_b, n_b) == (400 * 10, 400 * 100) and s3b == 0.1


def test_s3_ratio_of_sums_not_mean_of_ratios():
    rows = []
    for loc, (n, lo) in zip(range(1201, 1601), [(10, 10), (990, 0)] * 200):
        rows.append({"stage": "3", "local": str(loc), "d1_L_s3_out_n": str(n), "d1_L_s3_out_lo": str(lo)})
    s3, lo, n = dg.s3_of_rows(rows)
    assert s3 == pytest.approx(10 * 200 / (1000 * 200))


def test_t2_analogue_filters_phase_a():
    rows = []
    for loc in range(1, 1601):
        rows.append({"phase": "A", "local": str(loc), "d1_L_s2_out_n": "10", "d1_L_s2_out_lo": "1" if loc >= 1201 else "0"})
        rows.append({"phase": "B", "local": str(loc), "d1_L_s2_out_n": "10", "d1_L_s2_out_lo": "10"})
    s, lo, n = dg.s2_t2_of_rows(rows)
    assert (lo, n) == (400, 4000) and s == pytest.approx(0.1)


def test_onset_window_is_first_25_update_window_above_0_2():
    rows = _rows(lambda loc: 0.0 if loc <= 100 else 0.3)
    assert dg.onset_window(rows) == 125
    assert dg.onset_window(_rows(lambda loc: 0.2)) is None          # not strictly above 0.2
    assert dg.share_class(0.21) == "regime" and dg.share_class(0.2) == "mid" and dg.share_class(0.05) == "low"


# ---------------------------------------------------------------------------------- synthetic networks
def _zeros_arrays(hidden: int = 64):
    return {"actor.l1.weight": np.zeros((hidden, 2), np.float32), "actor.l1.bias": np.zeros(hidden, np.float32),
            "actor.l2.weight": np.zeros((hidden, hidden), np.float32), "actor.l2.bias": np.zeros(hidden, np.float32),
            "actor.out.weight": np.zeros((2, hidden), np.float32), "actor.out.bias": np.zeros(2, np.float32)}


def _bump_arrays(k: float = 30.0, w: float = 0.2, a: float = 4.0, b0: float = -3.0):
    """A network that is high on a band |d / (2B)| < w and low outside: z0 = a * tanh(tanh(k(x + w)) - tanh(k(x - w))) + b0."""
    arr = _zeros_arrays()
    arr["actor.l1.weight"][0, 1] = k
    arr["actor.l1.bias"][0] = k * w
    arr["actor.l1.weight"][1, 1] = k
    arr["actor.l1.bias"][1] = -k * w
    arr["actor.l2.weight"][0, 0] = 1.0
    arr["actor.l2.weight"][0, 1] = -1.0
    arr["actor.out.weight"][0, 0] = a
    arr["actor.out.bias"][0] = b0
    return arr


def test_forward_parts_reproduces_the_network_exactly():
    gen = torch.Generator().manual_seed(3)
    from agents.ppo_curriculum import BetaActor

    net = BetaActor(64, 100.0, 1e-6, gen)
    with torch.no_grad():
        for p in net.parameters():
            p += 0.3 * torch.randn(p.shape, generator=gen)
    x = torch.as_tensor(SPEC.encode_obs(3, np.linspace(-400, 400, 41)))
    p = dg.forward_parts(net, x)
    a, b = net(x)
    assert torch.equal(p["alpha"], a) and torch.equal(p["beta"], b)


def test_constant_network_has_no_formation_and_no_sensitivity():
    grid = dg.stage_grid(SPEC, 3)
    net = dg.network_on_grid(_zeros_arrays(), SPEC, grid, 3, with_critic=False)
    s = dg.export_summary(grid, net, SPEC.q)
    assert s["in_range"] == 0.0 and s["F"] == 0.0 and s["slope_neg"] == 0.0 and s["slope_pos"] == 0.0
    assert s["sens_in_max"] == 0.0 and s["sat1_in"] == 0.0 and s["sat2_in"] == 0.0
    assert s["e0"] == pytest.approx(50.0)


def test_saturated_second_layer_is_counted_and_kills_the_sensitivity():
    arr = _bump_arrays()
    arr["actor.l2.bias"][:] = 5.0                       # every second-layer unit: pre-activation >= 3 for every input
    arr["actor.l2.weight"][2:, :] = 0.0
    grid = dg.stage_grid(SPEC, 3)
    net = dg.network_on_grid(arr, SPEC, grid, 3, with_critic=False)
    s = dg.export_summary(grid, net, SPEC.q)
    assert s["sat2_in"] == pytest.approx(1.0)
    assert s["sat1_in"] < 0.1
    free = dg.export_summary(grid, dg.network_on_grid(_bump_arrays(), SPEC, grid, 3, with_critic=False), SPEC.q)
    assert s["sens_in_max"] < 0.01 * free["sens_in_max"] and s["in_range"] < 0.01 * free["in_range"]


def test_bump_network_forms_and_f_is_positive():
    grid = dg.stage_grid(SPEC, 3)                       # B = 200: D_3 = [-400, 400]; the bump covers |d| < 0.2 * 400 = 80 < 2q = 100
    net = dg.network_on_grid(_bump_arrays(w=0.25), SPEC, grid, 3, with_critic=False)     # |d| < 100 = 2q
    s = dg.export_summary(grid, net, SPEC.q)
    assert s["e0"] > 60.0 and s["out_mean"] < 10.0 and s["F"] > 50.0
    assert s["in_range"] > 10.0 and s["sens_in_max"] > 0.0
    assert s["slope_pos"] <= 0.0 <= s["slope_neg"]                   # the chords of a bump: rising to the left of 0, falling to the right
    assert s["out_alpha_lt1"] == 0.0                                 # mu = sigmoid(-3) = 0.047 and c >= 100: alpha >= 4.7


def test_formation_export_and_threshold():
    ups = np.arange(25, 201, 25)
    F = np.array([-0.2, -0.1, 0.0, 0.3, 6.0, 20.0, 40.0, 50.0])
    thr = dg.formation_threshold([40.0, 60.0, 50.0])          # half of the median 50 = 25
    assert thr == 25.0
    assert dg.formation_export(F, ups, thr) == 175
    assert dg.formation_export(F, ups, 100.0) is None
    assert dg.formation_export(np.full(8, 0.0), ups, thr) is None        # a constant network never forms


def test_stretch_entry_summary():
    grid = np.arange(-150.0, 151.0, 2.0)
    e = np.full(grid.shape, 30.0)
    e[(grid > -60) & (grid < -20)] = 0.2
    e[(grid > 40) & (grid < 50)] = 0.1
    s = dg.entry_stretch(grid, e, 50.0)
    assert s["n_runs"] == 2 and s["n_nodes"] == int(((grid > -60) & (grid < -20)).sum() + ((grid > 40) & (grid < 50)).sum())
    assert s["longest_d"] == "[-58, -22]" and s["longest_len"] == 19


# ---------------------------------------------------------------------------------- the retrospective table
def _check(delta, R, Rt, valid="True"):
    return {"Delta": delta, "R": R, "R_tail": Rt, "valid": valid}


def test_trigger_flags_read_only_stage3_check_columns():
    f = dg.trigger_flags(_check(0.006, 0.2, 0.01))
    assert f["Delta>0.005"] and not f["R>0.5"] and not f["R_tail>0.05"] and not f["invalid"]
    src = inspect.getsource(dg.trigger_flags) + inspect.getsource(dg.retro_table)
    for banned in ("rmse", "tail_mean", "peak", "Gmax", "eT_", "g2_", "theory"):
        assert banned not in src


def test_retro_table_counts():
    rows = []
    spec = [("a", True, 0.001, 0.1, 0.00), ("b", False, 0.009, 0.2, 0.01), ("c", False, 0.004, 1.2, 0.20), ("d", True, 0.002, 0.3, 0.00),
            ("e", False, 0.003, 0.2, 0.00)]           # e fails for a reason the checks do not see
    for lab, ok, de, R, Rt in spec:
        rows.append({"label": lab, "pass": ok, "flags": dg.trigger_flags(_check(de, R, Rt))})
    tab = {t["trigger"]: t for t in dg.retro_table(rows)}
    g = tab["Delta>0.005"]
    assert g["n_fired"] == 1 and g["fired_runs"] == "b" and g["fired_and_failed"] == 1
    assert g["failing_runs_total"] == 3 and g["failing_missed"] == 2 and set(g["missed_runs"].split(";")) == {"c", "e"}
    assert g["extra_updates"] == 1600 and g["extra_cost_fraction"] == pytest.approx(1600 / (5 * 3800))
    r = tab["R>0.7"]
    assert r["n_fired"] == 1 and r["fired_runs"] == "c" and r["fired_and_failed"] == 1 and r["failing_missed"] == 2
    assert tab["R_tail>0.05"]["fired_runs"] == "c"
    assert tab["invalid"]["n_fired"] == 0


def test_rank_from_top():
    assert dg.rank_from_top([1.0, 2.0, 3.0], 3.0) == 1
    assert dg.rank_from_top([1.0, 2.0, 3.0], 1.0) == 3
    assert dg.rank_from_top([3.0, 3.0, 1.0], 3.0) == 1


# ---------------------------------------------------------------------------------- the replay summaries (DG5)
def test_pooled_share_weights_by_the_squared_gradient_norm():
    assert dg.pooled_share(np.array([1.0, 0.0]), np.array([1.0, 3.0])) == pytest.approx(0.1)
    assert dg.pooled_share(np.array([0.5, np.nan]), np.array([2.0, 1.0])) == pytest.approx(0.5)


def _replay_rows(n_up: int = 200):
    rows = []
    for u in range(1, n_up + 1):
        r = {"update": str(u), "N": "512", "n_out": "384", "n_in": "128", "share_out": "0.9", "share_in": "0.1",
             "n_clamp_lo": "300", "share_clamp_lo": "0.6", "n_clamp_hi": "0", "share_clamp_hi": "nan", "n_clamp_free": "212",
             "share_clamp_free": "0.4", "gnorm_total": "2.0", "gnorm_in": "0.5", "gnorm_out": "1.9", "sens_in_mean": "0.01",
             "sat1_in": "0.0", "sat2_in": "0.5", "sample_norm_mean_clamp_lo": "3.0", "sample_norm_mean_clamp_free": "1.0",
             "sample_norm_mean_in": "1.5", "sample_norm_mean_out": "2.0", "adv_norm_mean_clamp_lo": "0.3",
             "adv_norm_mean_out": "0.1", "adv_norm_mean_in": "-0.3"}
        rows.append(r)
    return rows


def test_replay_windows_compare_gradient_share_with_sample_share():
    w = dg.replay_windows(_replay_rows(), windows=((1, 100), (101, 200)))
    assert [x["updates"] for x in w] == [100, 100]
    a = w[0]
    assert a["sample_share_out"] == pytest.approx(0.75) and a["grad_share_out_pooled"] == pytest.approx(0.9)
    assert a["ratio_out"] == pytest.approx(0.9 / 0.75)
    assert a["sample_share_clamp_lo"] == pytest.approx(300 / 512) and a["ratio_clamp_lo"] == pytest.approx(0.6 / (300 / 512))
    assert a["sample_share_in"] + a["sample_share_out"] == pytest.approx(1.0)
    assert a["cos_in_out_mean"] == pytest.approx((4.0 - 0.25 - 1.9 ** 2) / (2 * 0.5 * 1.9))
    assert a["sample_share_clamp_hi"] == 0.0 and np.isnan(a["ratio_clamp_hi"])               # no hi draws: no ratio is formed


def test_replay_bins_summary_coverage_and_pooled_share():
    nb = 40
    n = np.zeros((10, nb))
    n[:, 10] = 5
    n[:, 25] = 5
    share = np.full((10, nb), np.nan)
    share[:, 10], share[:, 25] = 0.25, 0.75
    z = {"update": np.arange(1, 11), "n": n, "share": share}
    rows = [{"gnorm_total": "1.0"} for _ in range(10)]
    out = dg.replay_bins_summary(z, rows, 50.0, 400.0, windows=((1, 10),))
    assert len(out) == nb and sum(x["starts_share"] for x in out) == pytest.approx(1.0)
    assert out[10]["starts_share"] == pytest.approx(0.5) and out[25]["grad_share_pooled"] == pytest.approx(0.75)
    assert out[10]["bin_centre"] == pytest.approx(-400 + 10 * 20 + 10) and out[20]["in_band"] and not out[10]["in_band"]
