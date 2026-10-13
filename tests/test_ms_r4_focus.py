"""MS-R4 sampler tests (E2): the within-stratum focus keeps the three stratum masses at lambda_T, lambda_P, lambda_M for any
focus and alpha, the normalisation and the fallbacks follow D4, the probabilities before the first valid check are the alpha-0
floats bit for bit, the gain map is built from the same verifier call (no extra call), the sampler decisions do not depend on the
closed form, and the configuration refusals hold.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_r4_focus.py -p no:cacheprovider -q
"""

from __future__ import annotations

import copy
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ms_r4_helpers as H  # noqa: E402,I001
from ms_r4_helpers import reduced_r4  # noqa: E402

import ms_configs as mc  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402
from run.run_final_dp_br import dense_grid, make_policy_fns  # noqa: E402
from run.run_final_dp_br_round3_dense import ConfigError  # noqa: E402
from test_ms_r2_runner import run_cfg  # noqa: E402
from test_ms_runner import _t3_params, make_cfg, tree_equal  # noqa: E402
from utils import ms_focus as F  # noqa: E402
from utils.dp_br_verifier import DEV_CONFIG, concentration_stats  # noqa: E402
from utils.ms_residual import StageDiag, ema_update, invalid_diag, stage_diag  # noqa: E402
from utils.ms_rule import StageController, StageRule  # noqa: E402
from utils.v2_metrics import evaluate  # noqa: E402


def _sampler(q: int = 50, T: int = 2) -> StartSampler:
    return StartSampler(GameSpec(w_h=6.5, w_l=3.0, k=0.001, q=float(q), T=T, e_min=0.0, e_max=100.0), 10.0)


def _masses(p: np.ndarray, sp: StartSampler, t: int = 2, hw: float = 20.0) -> List[float]:
    lab = sp.stratum_labels(t, hw)
    return [float(p[lab == k].sum()) for k in (0, 1, 2)]


def _lams(sp: StartSampler, t: int, lam_p: float = 0.35) -> List[float]:
    lt = sp.coverage_lambda_t(t)
    return [lt, lam_p, 1.0 - lam_p - lt]


# ---------------------------------------------------------------------------------------------- 1. the probability function
@pytest.mark.parametrize("q", [50, 60])
def test_the_stratum_masses_are_exactly_lambda_for_any_focus_and_alpha(q):
    sp = _sampler(q)
    st = sp.strata(2, 20.0)
    n = sp.n_bins(2)
    rng = np.random.default_rng(7)
    lam = _lams(sp, 2)
    maps = [rng.random(n), rng.random(n) ** 8, np.zeros(n), np.where(np.arange(n) == 17, 3.0, 0.0), np.full(n, np.nan)]
    for rho in maps:
        for gain in (None, rng.random(n), np.zeros(n)):
            f = F.within_stratum_focus(st, rho, gain)
            for alpha in (0.0, 0.25, 0.5, 1.0):
                p = F.within_strata_probs(sp, 2, 0.35, 20.0, alpha, f)
                assert abs(p.sum() - 1.0) < 1e-12 and (p >= 0).all()
                np.testing.assert_allclose(_masses(p, sp), lam, rtol=0, atol=1e-12)
                tail = p[st["tail"]]
                assert np.allclose(tail, tail[0])            # the tail stays uniform within its stratum


def test_the_existing_cross_stratum_sampler_does_move_the_stratum_masses():
    """The contrast that motivates adjustment 1: at alpha 0.5 the cross-stratum focus changes the near-tie / middle shares."""
    sp = _sampler(50)
    n = sp.n_bins(2)
    focus = np.zeros(n)
    focus[sp.strata(2, 20.0)["mid"]] = 1.0                   # all the focus on the middle stratum
    p = sp.stratified_bin_probs(2, 0.35, 20.0, alpha=0.5, focus=focus)
    m = _masses(p, sp)
    assert abs(m[0] - 0.5) < 1e-12 and abs(m[1] - 0.35) > 0.05
    f = F.within_stratum_focus(sp.strata(2, 20.0), focus)
    assert np.allclose(_masses(F.within_strata_probs(sp, 2, 0.35, 20.0, 0.5, f), sp), _lams(sp, 2), atol=1e-12)


def test_the_focus_is_normalised_inside_every_stratum_and_zero_on_the_tail():
    sp = _sampler(50)
    st = sp.strata(2, 20.0)
    rng = np.random.default_rng(1)
    rho, gain = rng.random(sp.n_bins(2)), rng.random(sp.n_bins(2))
    f = F.within_stratum_focus(st, rho, gain)
    assert f[st["tail"]].sum() == 0.0
    assert abs(f[st["near"]].sum() - 1.0) < 1e-12 and abs(f[st["mid"]].sum() - 1.0) < 1e-12
    for key in ("near", "mid"):
        m = st[key]
        want = 0.5 * rho[m] / rho[m].sum() + 0.5 * gain[m] / gain[m].sum()
        np.testing.assert_allclose(f[m], want, rtol=0, atol=1e-15)
    fr = F.within_stratum_focus(st, rho)                     # the residual focus alone
    for key in ("near", "mid"):
        np.testing.assert_allclose(fr[st[key]], rho[st[key]] / rho[st[key]].sum(), rtol=0, atol=1e-15)


def test_the_fallbacks_of_d4():
    sp = _sampler(50)
    st = sp.strata(2, 20.0)
    n = sp.n_bins(2)
    rng = np.random.default_rng(2)
    rho, gain = rng.random(n), rng.random(n)
    zeros = np.zeros(n)
    norm = lambda w, m: np.where(m, w, 0.0) / w[m].sum()  # noqa: E731
    for key in ("near", "mid"):
        m = st[key]
        # the g_bar sum over S is 0 -> the normalised rho_bar term alone; the rho_bar sum is 0 -> the g_bar term alone
        np.testing.assert_allclose(F.within_stratum_focus(st, rho, zeros)[m], norm(rho, m)[m], atol=1e-15)
        np.testing.assert_allclose(F.within_stratum_focus(st, zeros, gain)[m], norm(gain, m)[m], atol=1e-15)
        # both zero -> uniform on S
        np.testing.assert_allclose(F.within_stratum_focus(st, zeros, zeros)[m], np.full(int(m.sum()), 1.0 / m.sum()), atol=1e-15)
    # the fallback is per stratum: a map that is zero on the near-tie stratum only
    rho2 = rho.copy()
    rho2[st["near"]] = 0.0
    f = F.within_stratum_focus(st, rho2)
    np.testing.assert_allclose(f[st["near"]], np.full(int(st["near"].sum()), 1.0 / st["near"].sum()), atol=1e-15)
    np.testing.assert_allclose(f[st["mid"]], norm(rho2, st["mid"])[st["mid"]], atol=1e-15)
    # NaN is read as 0; a non-finite sum falls back (an infinite entry makes the sum not finite)
    rho3 = rho.copy()
    rho3[np.flatnonzero(st["mid"])[0]] = np.nan
    f3 = F.within_stratum_focus(st, rho3)
    assert f3[np.flatnonzero(st["mid"])[0]] == 0.0 and abs(f3[st["mid"]].sum() - 1.0) < 1e-12
    rho4 = rho.copy()
    rho4[np.flatnonzero(st["mid"])[0]] = np.inf
    f4 = F.within_stratum_focus(st, rho4)
    np.testing.assert_allclose(f4[st["mid"]], np.full(int(st["mid"].sum()), 1.0 / st["mid"].sum()), atol=1e-15)


def test_invalid_maps_are_refused():
    sp = _sampler(50)
    st = sp.strata(2, 20.0)
    n = sp.n_bins(2)
    with pytest.raises(ValueError):
        F.within_stratum_focus(st, np.full(n, -1.0))
    with pytest.raises(ValueError):
        F.within_stratum_focus(st, np.ones(n + 1))
    with pytest.raises(ValueError):
        F.within_strata_probs(sp, 2, 0.35, 20.0, 1.5, np.ones(n) / n)
    with pytest.raises(ValueError):
        F.within_strata_probs(sp, 2, 1.2, 20.0, 0.5, np.ones(n) / n)
    with pytest.raises(ValueError):
        F.within_strata_probs(sp, 2, 0.35, 20.0, 0.5, np.ones(n + 3))


def test_before_the_first_valid_check_the_probabilities_are_the_alpha_zero_floats_bit_for_bit():
    for q in (50, 60):
        sp = _sampler(q)
        want = sp.stratified_bin_probs(2, 0.35, 20.0, alpha=0.0)
        got = F.within_strata_probs(sp, 2, 0.35, 20.0, 0.5, None)
        assert np.array_equal(want, got) and want.tobytes() == got.tobytes()
        # alpha 0 with the existing sampler: a focus leaves the floats unchanged too (the premise of fb_a0's configuration)
        focus = np.random.default_rng(3).random(sp.n_bins(2))
        assert np.array_equal(sp.stratified_bin_probs(2, 0.35, 20.0, alpha=0.0, focus=focus), want)
        # alpha 0.5 without a focus: 0.5 p + 0.5 p == p exactly (the premise of fb_rx's identity with fb_a0 through update 25)
        assert np.array_equal(sp.stratified_bin_probs(2, 0.35, 20.0, alpha=0.5, focus=None), want)


def test_the_probability_function_is_defined_for_every_stage_of_t3():
    """Stage 3 of T = 3 has a tail; stage 2 of T = 3 has none (n_T = 0): the tail term is absent and lambda_T = 0."""
    sp = _sampler(50, T=3)
    lam_p = mc.bin_balanced_near_tie_share(sp.spec, 10.0, 20.0)         # 0.35 leaves no middle mass at stage 3 of T = 3
    for t in (2, 3):
        st = sp.strata(t, 20.0)
        n = sp.n_bins(t)
        f = F.within_stratum_focus(st, np.random.default_rng(t).random(n), np.random.default_rng(t + 9).random(n))
        for alpha in (0.0, 0.5, 1.0):
            p = F.within_strata_probs(sp, t, lam_p, 20.0, alpha, f)
            lam = _lams(sp, t, lam_p)
            assert abs(p.sum() - 1.0) < 1e-12
            np.testing.assert_allclose(_masses(p, sp, t), lam, atol=1e-12)
    assert int(sp.strata(2, 20.0)["tail"].sum()) == 0 and sp.coverage_lambda_t(2) == 0.0
    assert int(sp.strata(3, 20.0)["tail"].sum()) > 0


# ---------------------------------------------------------------------------------------------- 2. the gain map
def _verify_result(q: int = 50):
    """A real development-tier verifier result of a smooth, non-equilibrium policy (the closed form is not used)."""
    sp_game = GameSpec(w_h=6.5, w_l=3.0, k=0.001, q=float(q), T=2, e_min=0.0, e_max=100.0)

    def mean_fn(t: int, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        return 45.0 + 15.0 * np.exp(-(d / 40.0) ** 2) * (1 if t == 2 else 0) + 0.02 * d * (t == 2)

    ev = evaluate(mean_fn, sp_game, DEV_CONFIG, beta_fn=None, recovery_step=0.5)
    return sp_game, ev


def test_stage_diag_with_gain_adds_only_the_gain_map():
    spec, ev = _verify_result()
    conc = {"valid": True, "max_std_norm": 0.01}
    a = stage_diag(ev.res, spec, 2, conc, 10.0)
    b = stage_diag(ev.res, spec, 2, conc, 10.0, with_gain=True)
    assert a.gain_bins.size == 0                              # the default: an empty map
    for f in ("valid", "delta_over_dw", "s", "R", "R_tail", "C", "tail_term", "n_nodes_nontail", "n_nodes_tail", "argmax_d"):
        x, y = getattr(a, f), getattr(b, f)
        assert (x == y) or (isinstance(x, float) and np.isnan(x) and np.isnan(y)), f
    assert np.array_equal(a.rho_bins, b.rho_bins, equal_nan=True) and a.scalars() == b.scalars()
    sr = ev.res.stages[2]
    d = np.asarray(sr.d_grid, float)
    sampler = StartSampler(spec, 10.0)
    from envs.curriculum_env import gap_bin_index
    idx, nb = gap_bin_index(spec, 2, d, 10.0)
    nontail = np.abs(d) < 2.0 * spec.q * (spec.T - 2 + 1) - 1e-9
    use = nontail & ~sampler.tail_mask(2)[idx]
    want = np.full(nb, np.nan)
    g = np.asarray(sr.delta, float) / ev.res.dw
    for k in range(nb):
        m = use & (idx == k)
        if m.any():
            want[k] = g[m].max()
    assert np.array_equal(b.gain_bins, want, equal_nan=True)
    assert np.isnan(b.gain_bins[sampler.tail_mask(2)]).all() and np.isfinite(b.gain_bins[~sampler.tail_mask(2)]).all()
    assert (b.gain_bins[np.isfinite(b.gain_bins)] >= 0).all()    # Delta >= 0 by construction
    assert np.nanmax(b.gain_bins) <= b.delta_over_dw + 1e-15    # the maximum over a subset of the grid


# ---------------------------------------------------------------------------------------------- 3. the controller
def _rule(n_nontail: int, budget: int = 40) -> StageRule:
    return StageRule(stage=2, enabled=False, K=10, eps=0.005, rho=0.03, tau=0.02, ema_beta=0.5, alpha_global=0.5,
                     n_nontail_bins=n_nontail, fixed_budget=budget)


def _diag(rho: np.ndarray, gain: np.ndarray, valid: bool = True) -> StageDiag:
    return StageDiag(stage=2, valid=valid, delta_over_dw=0.01, s=50.0, R=0.1, R_tail=0.0, C=0.01, tail_term=True,
                     rho_bins=rho, gain_bins=gain)


def _ctl(focus_map=None, with_scope=True):
    sp = _sampler(50)
    st = sp.strata(2, 20.0)
    kw = dict(focus_scope="within_strata", focus_map=focus_map, strata=st) if with_scope else {}
    return StageController(_rule(int((~st["tail"]).sum())), lambda *a: 3e-4, lambda j: 3e-4, legacy_global_sampler=True, **kw), sp


def test_the_controller_averages_the_gain_map_with_the_same_ema_as_rho_bar():
    ctl, sp = _ctl("residual_gain")
    n = sp.n_bins(2)
    tail = sp.tail_mask(2)
    r1, g1 = np.where(tail, np.nan, 1.0), np.where(tail, np.nan, 0.2)
    r2, g2 = np.where(tail, np.nan, 3.0), np.where(tail, np.nan, 0.6)
    assert ctl.focus_within() is None and ctl.sampler_setting().focus_within is None and ctl.gain_bar is None
    ctl.after_update(1, lambda: invalid_diag(2, n))              # an invalid check makes no map
    ctl.after_update(10, lambda: _diag(r1, g1, valid=False))
    assert ctl.rho_bar is None and ctl.gain_bar is None and ctl.focus_within() is None
    ctl.after_update(20, lambda: _diag(r1, g1))
    np.testing.assert_array_equal(ctl.rho_bar, ema_update(None, r1, 0.5))
    np.testing.assert_array_equal(ctl.gain_bar, ema_update(None, g1, 0.5))
    ctl.after_update(30, lambda: _diag(r2, g2))
    np.testing.assert_allclose(ctl.gain_bar[~tail], 0.5 * 0.2 + 0.5 * 0.6)
    np.testing.assert_allclose(ctl.rho_bar[~tail], 0.5 * 1.0 + 0.5 * 3.0)
    f = ctl.focus_within()
    assert f is not None and f[tail].sum() == 0 and ctl.sampler_setting().focus_within is not None


def test_a_residual_only_controller_keeps_no_gain_state_and_a_plain_one_no_within_focus():
    ctl, sp = _ctl(None)
    n = sp.n_bins(2)
    tail = sp.tail_mask(2)
    ctl.after_update(10, lambda: _diag(np.where(tail, np.nan, 1.0), np.where(tail, np.nan, 0.2)))
    assert ctl.gain_bar is None and ctl.focus_within() is not None
    plain, _ = _ctl(with_scope=False)
    plain.after_update(10, lambda: _diag(np.where(tail, np.nan, 1.0), np.where(tail, np.nan, 0.2)))
    assert plain.focus_within() is None and plain.sampler_setting().focus_within is None and plain.gain_bar is None


@pytest.mark.parametrize("kw", [dict(focus_scope="across", strata={}), dict(focus_scope="within_strata"),
                                dict(focus_map="residual_gain"), dict(focus_scope="within_strata", focus_map="gain", strata={})])
def test_the_controller_refuses_inconsistent_focus_arguments(kw):
    sp = _sampler(50)
    st = sp.strata(2, 20.0)
    if "strata" in kw and kw["strata"] == {}:
        kw = {**kw, "strata": st}
    with pytest.raises(ValueError):
        StageController(_rule(20), lambda *a: 3e-4, lambda j: 3e-4, legacy_global_sampler=True, **kw)
    with pytest.raises(ValueError):                      # focus_scope needs the fixed-budget global sampler
        StageController(_rule(20), lambda *a: 3e-4, lambda j: 3e-4, legacy_global_sampler=False, focus_scope="within_strata",
                        strata=st)


# ---------------------------------------------------------------------------------------------- 4. configuration validation
def _e2(arm: str = "fb_rg") -> Dict[str, Any]:
    return reduced_r4(arm, "/unused")


def test_the_four_e2_configs_differ_only_in_the_sampler_keys():
    cfgs = {a: _e2(a) for a in ("fb_a0", "fb_r", "fb_rg", "fb_rx")}
    base = copy.deepcopy(cfgs["fb_a0"])
    for a, c in cfgs.items():
        rms.validate_config(c)
        diff = {k for k in set(c) | set(base) if c.get(k) != base.get(k)}
        want = set() if a == "fb_a0" else {"arm", "run", "record", "start_weights"}
        if a in ("fb_r", "fb_rg"):
            want |= {"focus_scope"}
        if a == "fb_rg":
            want |= {"focus_map"}
        assert diff == want, (a, diff)
        assert {k: v for k, v in c["start_weights"].items() if k != "alpha_global"} == \
               {k: v for k, v in base["start_weights"].items() if k != "alpha_global"}
        assert c["fixed_global_sampler"] is True and c["rule"]["enabled"] is False and "actor_variant" not in c
        assert c["start_weights"]["alpha_global"] == (0.0 if a == "fb_a0" else 0.5)
    assert cfgs["fb_rg"]["focus_map"] == "residual_gain" and cfgs["fb_r"].get("focus_map") is None
    assert "focus_scope" not in cfgs["fb_a0"] and "focus_scope" not in cfgs["fb_rx"]


@pytest.mark.parametrize("mutate", [
    lambda c: c.pop("focus_scope"),                                            # focus_map without focus_scope
    lambda c: c.__setitem__("focus_scope", "across"),
    lambda c: c.__setitem__("focus_scope", True),
    lambda c: c.__setitem__("focus_map", "residual"),
    lambda c: c.pop("fixed_global_sampler"),
    lambda c: c.__setitem__("fixed_global_sampler", False),
    lambda c: c["start_weights"].__setitem__("scheme", "bin_balanced"),
    lambda c: c["rule"].__setitem__("enabled", True),
])
def test_configuration_refusals_of_the_new_keys(mutate):
    cfg = _e2("fb_rg")
    mutate(cfg)
    with pytest.raises(ConfigError):
        rms.validate_config(cfg)


def test_focus_scope_alone_and_without_its_preconditions():
    cfg = _e2("fb_r")
    rms.validate_config(cfg)
    bad = copy.deepcopy(cfg)
    bad["rule"]["enabled"] = True
    with pytest.raises(ConfigError):
        rms.validate_config(bad)
    bad = _e2("fb_rx")
    bad["focus_scope"] = "within_strata"
    bad.pop("fixed_global_sampler")
    with pytest.raises(ConfigError, match="focus_scope needs"):
        rms.validate_config(bad)


# ---------------------------------------------------------------------------------------------- 5. runs
@pytest.fixture(scope="module")
def e2_wave(tmp_path_factory) -> Path:
    return H.get_wave(tmp_path_factory, arms=("fb_a0", "fb_r", "fb_rg", "fb_rx"))


def _binmaps(root: Path, arm: str):
    return np.load(H.run_dir(root, arm) / "ms_binmaps_stage2.npz")


def test_the_within_stratum_arms_keep_every_stratum_mass_fixed_at_every_update(e2_wave):
    sp = _sampler(50)
    lab = sp.stratum_labels(2, 20.0)
    lam = np.array(_lams(sp, 2))
    for arm in ("fb_a0", "fb_r", "fb_rg"):
        pt = _binmaps(e2_wave, arm)["probs_table"]
        sh = np.stack([pt[:, lab == k].sum(axis=1) for k in (0, 1, 2)], axis=1)
        assert np.abs(sh - lam).max() <= 1e-12, arm
    pt = _binmaps(e2_wave, "fb_rx")["probs_table"]
    sh = np.stack([pt[:, lab == k].sum(axis=1) for k in (0, 1, 2)], axis=1)
    assert abs(sh[:, 0] - lam[0]).max() <= 1e-12 and np.abs(sh[:, 1] - lam[1]).max() > 1e-3     # fb_rx moves the near-tie share


def test_the_new_records_exist_only_for_the_new_keys(e2_wave):
    old = {"update", "local", "rho", "rho_bar", "probs", "probs_first_update", "probs_table"}
    for arm in ("fb_a0", "fb_rx"):
        assert set(_binmaps(e2_wave, arm).files) == old
    assert set(_binmaps(e2_wave, "fb_r").files) == old | {"focus_within", "probs_next"}
    z = _binmaps(e2_wave, "fb_rg")
    assert set(z.files) == old | {"focus_within", "probs_next", "gain", "gain_bar"}
    assert np.isnan(z["probs_next"][-1]).all() and np.isfinite(z["probs_next"][:-1]).all()
    sp = _sampler(50)
    st = sp.strata(2, 20.0)
    assert (z["focus_within"][:, st["tail"]] == 0).all()
    for key in ("near", "mid"):
        np.testing.assert_allclose(z["focus_within"][:, st[key]].sum(axis=1), 1.0, atol=1e-12)
    assert np.isnan(z["gain"][:, st["tail"]]).all() and (z["gain"][:, ~st["tail"]] >= 0).all()


def test_the_first_twenty_updates_up_to_the_first_check_are_identical_in_all_four_arms(e2_wave):
    """C-FB25 in miniature (K = 10): the per-update series of updates 1-10 and the u5 / u10 exports are the same, but for
    the column that records the configured alpha."""
    rows = {a: list(csv.DictReader(open(H.run_dir(e2_wave, a) / "ms_updates.csv"))) for a in ("fb_a0", "fb_r", "fb_rg", "fb_rx")}
    skip = {"update_wall_sec", "alpha", "run", "arm"}
    for a in ("fb_r", "fb_rg", "fb_rx"):
        assert all(all(rows["fb_a0"][i][c] == rows[a][i][c] for c in rows["fb_a0"][0] if c not in skip) for i in range(10)), a
        assert any(rows["fb_a0"][i]["p_digest"] != rows[a][i]["p_digest"] for i in range(10, 60)), a
    s2 = lambda a: {r["alpha"] for r in rows[a] if r["stage"] == "2"}  # noqa: E731   (stage 1 has no sampler: alpha 0.0)
    assert s2("fb_a0") == {"0.0"} and s2("fb_r") == s2("fb_rg") == s2("fb_rx") == {"0.5"}


def test_fb_rg_reads_the_gain_from_the_same_verifier_call(tmp_path, monkeypatch):
    """No extra verifier call: ``fb_r`` and ``fb_rg`` call ``utils.v2_metrics.evaluate`` the same number of times (one per
    check), with identical arguments."""
    calls: Dict[str, List[Any]] = {}
    orig = rms.evaluate
    for arm in ("fb_r", "fb_rg"):
        calls[arm] = []

        def spy(*a: Any, _arm=arm, **k: Any) -> Any:
            calls[_arm].append((a[2].name, k.get("recovery_step")))
            return orig(*a, **k)
        monkeypatch.setattr(rms, "evaluate", spy)
        out = str(tmp_path / arm)
        os.makedirs(out)
        ms = rms.MSRun(reduced_r4(arm, out), out)
        ms.run_stage(2)
    assert calls["fb_r"] == calls["fb_rg"] and len(calls["fb_r"]) == 6                # one development-tier call per check (60 / 10)
    assert {c[0] for c in calls["fb_rg"]} == {"development"}


@pytest.mark.parametrize("arm", ["fb_r", "fb_rg"])
def test_sampler_decisions_do_not_depend_on_the_closed_form_with_the_new_keys(tmp_path, monkeypatch, arm):
    """The repetition of ``test_ms_runner.test_decisions_and_sampler_do_not_depend_on_the_closed_form`` with both new keys:
    replacing the closed-form functions by stubs changes neither the check decisions, nor the probabilities in force, nor the
    focus and gain maps, nor the training state."""
    import utils.v2_metrics as vm
    states, rows, maps = [], [], []
    for tag in ("real", "stub"):
        d = str(tmp_path / tag)
        os.makedirs(d)
        if tag == "stub":
            def junk(policy, spec, step):
                D = np.linspace(-spec.domain_half(2), spec.domain_half(2), 9)
                sc = {"g1": 1.0, "g2_at_0": 1.0, "e1_at_0": 7.0, "e2_at_0": 3.0, "stage1_rel_err_signed": 123.0,
                      "stage2_peak_rel_err_signed": -9.0, "stage2_peak_rel_err_abs": 9.0}
                return sc, {"recovery_d_grid": D, "recovery_e2": D * 0, "recovery_g2": D * 0 + 5.0}
            monkeypatch.setattr(vm, "recovery_metrics", junk)
        run = rms.MSRun(reduced_r4(arm, d), d)
        run.run_stage(2)
        states.append(run.full_state("stage2"))
        rows.append(list(csv.DictReader(open(os.path.join(d, "ms_checks_stage2.csv")))))
        maps.append(dict(np.load(os.path.join(d, "ms_binmaps_stage2.npz"))))
    cols = ("update", "local", "mode", "block_id", "block_type", "valid", "Delta", "s", "R", "R_tail", "C", "eligible",
            "consecutive", "p_digest", "p_digest_next")
    assert [{c: r[c] for c in cols} for r in rows[0]] == [{c: r[c] for c in cols} for r in rows[1]]
    assert rows[0][0]["stage2_peak_rel_err_signed"] != rows[1][0]["stage2_peak_rel_err_signed"]          # the stub was used
    assert maps[0].keys() == maps[1].keys()
    for k in maps[0]:
        assert np.array_equal(maps[0][k], maps[1][k], equal_nan=True), k
    for k in ("agent", "rng", "torch_generator_state"):
        assert tree_equal(states[0][k], states[1][k]), k


# ---------------------------------------------------------------------------------------------- 6. T = 3 smoke
def _t3_legacy(out: str, focus_map) -> Dict[str, Any]:
    cfg = make_cfg(out, arm="MS_rule", T=3, params=_t3_params(False), epu=48)   # lambda_P = the bin-balanced near-tie share
    cfg["start_weights"]["alpha_global"] = 0.5
    cfg["rule"]["enabled"] = False
    cfg["rule"]["stages"] = {t: {k: v for k, v in e.items() if k in ("eps", "rho", "tau")} for t, e in cfg["rule"]["stages"].items()}
    cfg["pipeline"]["budgets"] = {"3": 30, "2": 30, "1": 10}
    cfg["pipeline"]["lr_windows"] = {"3": [], "2": [], "1": []}
    cfg["fixed_global_sampler"] = True
    cfg["focus_scope"] = "within_strata"
    if focus_map:
        cfg["focus_map"] = focus_map
    return cfg


@pytest.mark.parametrize("focus_map", [None, "residual_gain"])
def test_t3_smoke_with_the_within_stratum_focus(tmp_path, focus_map):
    """The reduced T = 3 pipeline (phases 3, 2, 1; stage 2 has no tail bin) runs with the new keys; the stratum masses of every
    update are lambda_T, lambda_P, lambda_M of its stage."""
    d = str(tmp_path / "t3")
    os.makedirs(d)
    cfg = _t3_legacy(d, focus_map)
    assert rms.run_pipeline(cfg, d, "pytest") == 0
    assert np.load(os.path.join(d, "ms_binmaps_stage3.npz")).files.count("probs_next") == 1
    spec = GameSpec(**{k: cfg["record"]["game"][k] for k in ("w_h", "w_l", "k", "q", "T", "e_min", "e_max")})
    sp = StartSampler(spec, 10.0)
    lam_p = cfg["start_weights"]["lambda_P"]
    for t in (3, 2):
        z = np.load(os.path.join(d, f"ms_binmaps_stage{t}.npz"))
        lam = np.array(_lams(sp, t, lam_p))
        pt = z["probs_table"]
        lab = sp.stratum_labels(t, 20.0)
        sh = np.stack([pt[:, lab == k].sum(axis=1) for k in (0, 1, 2)], axis=1)
        assert np.abs(sh - lam).max() <= 1e-12, t
        assert ("gain" in z.files) == bool(focus_map)
    assert np.load(os.path.join(d, "ms_binmaps_stage2.npz"))["probs_next"].shape[1] == sp.n_bins(2)
    assert not os.path.exists(os.path.join(d, "ms_binmaps_stage1.npz"))              # stage 1 has no sampler
    assert __import__("json").load(open(os.path.join(d, "gates.json")))["global_rng"]["status"] == "ok"
