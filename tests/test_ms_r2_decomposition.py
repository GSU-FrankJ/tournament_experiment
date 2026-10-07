"""MS-R2 premise check (tools/ms/r2_decomposition.py) and the noise helpers (utils/ms_noise.py)."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import r2_decomposition as D  # noqa: E402
from agents.ppo_curriculum import BetaActor  # noqa: E402
from envs.curriculum_env import GameSpec  # noqa: E402
from run.run_final_dp_br import make_policy_fns  # noqa: E402
from test_ms_runner import tree_equal  # noqa: E402,F401
from utils import ms_noise as N  # noqa: E402

PILOT = ROOT / "results" / "ms_r1" / "pilot"
PER_RUN = ROOT / "results" / "ms_r1" / "analysis" / "per_run.csv"
PROTO = json.load(open(ROOT / "protocols" / "v2_T2_locked_v2_0.json"))


def _spec(q: int) -> GameSpec:
    g = PROTO["records"][str(q)]["game"]
    return GameSpec(**{k: g[k] for k in ("w_h", "w_l", "k", "q", "T", "e_min", "e_max")})


# ------------------------------------------------------------------------------------------------ formulas
@pytest.mark.parametrize("q", [50, 60])
def test_the_smoothing_part_equals_the_gaussian_formula_for_a_gaussian_like_beta(q):
    spec = _spec(q)
    g2 = spec.dw / (4.0 * spec.k * spec.q)
    for sigma in (0.6, 1.25, 2.5, 4.0):
        c = 0.7 * 0.3 * spec.e_range ** 2 / sigma ** 2 - 1.0            # Beta with mean 0.7 and the effort sd sigma
        a, b = 0.7 * c, 0.3 * c
        e_sig, sd = N.smoothed_tie_prediction(a, b, spec.dw, spec.k, spec.q, spec.e_range)
        assert sd == pytest.approx(sigma, rel=2e-3)
        smooth = g2 - e_sig
        assert smooth / N.smoothing_formula(g2, sd, spec.q) == pytest.approx(1.0, abs=1.5e-3)
        assert N.beta_std_effort(a, b, spec.e_range) == pytest.approx(sigma, rel=1e-12)


def test_the_decomposition_identity_and_the_formula_ratio():
    d = N.decompose_gap(70.0, 68.0, 66.5, 2.5, 50.0)
    assert d["gap"] == pytest.approx(3.5) and d["smoothing"] == pytest.approx(2.0) and d["remainder"] == pytest.approx(1.5)
    assert d["gap"] == pytest.approx(d["smoothing"] + d["remainder"])
    assert d["formula"] == pytest.approx(70.0 * 2.5 / (math.sqrt(math.pi) * 50.0)) and d["ratio"] == pytest.approx(2.0 / d["formula"])
    assert math.isnan(N.decompose_gap(70.0, 68.0, 66.5, 0.0, 50.0)["ratio"])


def test_beta_moments_equal_scipy():
    for a, b in ((140.0, 60.0), (210.0, 90.0), (50.0, 150.0)):
        skew, exk = N.beta_noise_moments(a, b)
        m, v, s, k = beta_dist.stats(a, b, moments="mvsk")
        assert skew == pytest.approx(float(s), rel=1e-9) and exk == pytest.approx(float(k), rel=1e-9)


def test_the_noise_report_of_a_policy_equals_smoothed_share_and_the_gap_identity():
    """``tie_noise_report`` on a network equals ``run.run_v2_T2_locked.smoothed_share``'s own computation."""
    from run import run_v2_T2_locked as lk
    spec = _spec(50)
    g = torch.Generator().manual_seed(5)
    net = BetaActor(64, 100.0, 1e-6, g)
    with torch.no_grad():
        net.out.weight.copy_(torch.randn(2, 64, generator=g) * 0.05)
        net.out.bias.copy_(torch.tensor([0.9, 0.4]))
    net.eval()

    class Shim:
        @staticmethod
        def beta_params(obs, net=None):
            with torch.no_grad():
                a, b = net(torch.as_tensor(np.asarray(obs, dtype=np.float32)))
            return a.numpy(), b.numpy()

    mean_fn, beta_fn = make_policy_fns(Shim(), spec, net=net)

    class RunLike:
        agent = Shim()

    RunLike.spec = spec
    sm = lk.smoothed_share(RunLike(), None, net=net)                    # the locked implementation, on the same network
    rep = N.tie_noise_report(mean_fn, beta_fn, spec, 2)
    assert rep["e_sigma_0"] == pytest.approx(sm["smoothed_e_pred_0"], abs=1e-12)
    assert rep["e_hat_0"] == pytest.approx(sm["smoothed_e_learned_0"], abs=1e-12)
    assert rep["gap"] == pytest.approx(rep["smoothing"] + rep["remainder"], abs=1e-12)
    assert rep["g2_0"] == pytest.approx(70.0) and rep["e_hat_0"] == pytest.approx(float(mean_fn(2, np.zeros(1))[0]))
    assert N.tie_noise_report(mean_fn, beta_fn, spec, 2, g2_0=70.5, e_hat_0=60.0)["gap"] == pytest.approx(10.5)


# ------------------------------------------------------------------------------------------------ the tool on synthetic rows
def _frame(ratio_shift: float = 0.0) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(1)
    for arm in ("MS_base", "MS_base2400", "MS_rule", "MS_s25a0", "MS_s25a5", "MS_s35a0", "MS_s35a5"):
        for q in (50, 60):
            g2 = 70.0 if q == 50 else 350.0 / 6.0
            for seed in D.SEEDS:
                sigma = 2.5 + 0.3 * rng.standard_normal()
                smooth = (g2 * sigma / (math.sqrt(math.pi) * q)) * (1.0 + ratio_shift)
                rem = abs(rng.normal(1.0, 1.0))
                rows.append({"arm": arm, "q": q, "seed": seed, "role": "ms_arm", "status": "done", "g2_at_0": g2,
                             "smoothed_e_pred_0": g2 - smooth, "e2_at_0": g2 - smooth - rem, "sigma_effort_at_0_t2": sigma})
    return pd.DataFrame(rows)


def test_load_per_run_decomposes_every_row(tmp_path):
    f = tmp_path / "per_run.csv"
    df = _frame()
    df = pd.concat([df, df.assign(arm="parents_A", role="comparator")])          # comparators are not analysed
    df.to_csv(f, index=False)
    out = D.load_per_run(f)
    assert len(out) == 140 and set(out["role"]) == {"ms_arm"}
    assert np.allclose(out["gap"], out["smoothing"] + out["remainder"], atol=1e-12)
    assert out["ratio"].between(0.9999, 1.0001).all()
    assert np.allclose(out["smoothing_rel"], out["smoothing"] / out["g2_at_0"])


def test_checks_pass_on_formula_consistent_data_and_stop_when_the_ratio_is_off(tmp_path):
    for shift, stop in ((0.0, False), (0.05, True)):
        f = tmp_path / f"p{shift}.csv"
        _frame(shift).to_csv(f, index=False)
        out = tmp_path / f"o{shift}"
        rc = D.main(["--per-run", str(f), "--out", str(out)])
        c = json.load(open(out / "decomposition_checks.json"))
        assert c["ratio_check_pass"] is (not stop) and c["stop_condition"] is True       # the preamble table is not reproduced by synthetic data
        assert rc == 3
        assert bool(c["ratio_outside_limits"]) is stop


def test_the_paired_table_has_budget_and_sampler_rows_with_fresh_bootstrap_generators():
    tmp = _frame()
    rows = [D.decompose(float(r.g2_at_0), float(r.smoothed_e_pred_0), float(r.e2_at_0), float(r.sigma_effort_at_0_t2), float(r.q))
            for r in tmp.itertuples()]
    full = pd.concat([tmp.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    pair = D.paired_table(full)
    assert set(pair.comparison) == {"budget", "sampler"} and len(pair) == 2 * 4 + 2 * 4 * 3
    a = pair[(pair.comparison == "budget") & (pair.q == 50) & (pair.quantity == "smoothing")].iloc[0]
    x = full[(full.arm == "MS_base2400") & (full.q == 50)].set_index("seed").smoothing
    y = full[(full.arm == "MS_base") & (full.q == 50)].set_index("seed").smoothing
    d = (x - y).to_numpy()
    lo, hi = D.boot_ci(d)
    assert (a["mean"], a.ci_lo, a.ci_hi) == (float(d.mean()), lo, hi)
    assert D.boot_ci(d) == D.boot_ci(d)                                           # a fresh generator per call: repeatable


# ------------------------------------------------------------------------------------------------ stored MS-R1 results
@pytest.mark.skipif(not (PILOT / "q50" / "seed10501" / "MS_s35a5" / "freeze_stage2_final.npz").exists() or not PER_RUN.exists(),
                    reason="the MS-R1 pilot arrays / per_run.csv are not in this clone")
def test_e_sigma_recomputed_from_the_stored_beta_parameters_equals_the_logged_value():
    df = D.load_per_run(PER_RUN)
    sub = df[(df.arm == "MS_s35a5") & (df.q == 50) & (df.seed == 10501)]
    rec = D.recompute_from_arrays(sub, PILOT, None)
    assert len(rec) == 1 and abs(rec["diff_e_sigma"].iloc[0]) < 1e-6 and rec["d_node"].iloc[0] == 0.0
    assert abs(rec["skewness"].iloc[0]) < 0.11 and abs(rec["excess_kurtosis"].iloc[0]) < 0.03


@pytest.mark.skipif(not PER_RUN.exists(), reason="results/ms_r1/analysis/per_run.csv is not in this clone")
def test_the_ms_r1_table_of_the_preamble_is_reproduced():
    df = D.load_per_run(PER_RUN)
    tab = D.arm_table(df)
    pair = D.paired_table(df)
    c = D.run_checks(df, tab, pair, pd.DataFrame())
    assert c["n_runs"] == 140 and c["ratio_check_pass"] and c["table_reproduced"] and c["stop_condition"] is False
    assert c["learned_peak_below_closed_form"] == 139
    assert 0.9992 < c["ratio_min"] < c["ratio_max"] < 0.9998
