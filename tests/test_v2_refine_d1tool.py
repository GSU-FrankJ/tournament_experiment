"""Tests of tools/v2/d1_clamp_analysis.py (D1, T57 issue 7).

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest \
     tests/test_v2_refine_d1tool.py -q

Covers: the censored log-mass and the log-density difference against scipy on buffers with known
clamped rows (alpha = beta = 0.3); the gradient share on a tiny actor against a brute-force
per-row autograd computation; the zero-clamp case; the M1 / M2 flag logic (strict inequalities,
exactly at threshold); discovery with arms; end to end on a mocked multi-seed root and on small
real runs (phase A and a frozen phase B of the repo's smoke configuration).
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import special
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import d1_clamp_analysis as D  # noqa: E402
from agents.ppo_curriculum import BetaActor  # noqa: E402
from rng_alignment import base_config  # noqa: E402
from run.run_v2_stagewise import Run, execute  # noqa: E402

C = D.CLAMP
B2_MEAN = {"reward_mode": "expected", "stage2_update_mode": "frozen",
           "adv_norm_scope": "stage1_rows", "continuation_action_mode": "mean"}


# ====================================================================== censored log-mass vs scipy
def _known_buffer(n: int = 1000, a: float = 0.3, b: float = 0.3, seed: int = 0):
    """Buffer whose alpha = beta = 0.3 and whose clamped rows are known exactly."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.01, 0.99, n)
    lo_rows, hi_rows = [3, 10, 11, 500], [7, 600]
    raw[lo_rows] = [0.0, 5e-7, 1e-9, 9.99999e-7]
    raw[hi_rows] = [1.0, 1.0 - 5e-7]
    raw[20], raw[21] = C, 1.0 - C                  # exactly at the clamp: not "below" / "above"
    raw[40], raw[41] = 0.0, 1.0                    # clamped but NOT policy rows
    pm = np.ones(n, dtype=bool)
    pm[[40, 41]] = False
    act = np.clip(raw, C, 1.0 - C).astype(np.float32)
    al = np.full(n, a, dtype=np.float32)
    be = np.full(n, b, dtype=np.float32)
    olp = torch.distributions.Beta(torch.as_tensor(al), torch.as_tensor(be)).log_prob(
        torch.as_tensor(act)).numpy()
    buf = {"raw": raw, "actions": act, "alpha": al, "beta": be, "old_logp": olp,
           "policy_mask": pm, "stage": np.full(n, 2)}
    return buf, lo_rows, hi_rows


def test_clamped_rows_counts_and_censored_mass_match_scipy():
    buf, lo_rows, hi_rows = _known_buffer()
    tab = D.clamped_row_table(buf)
    assert sorted(tab.loc[tab["side"] == "lo", "row"]) == sorted(lo_rows)
    assert sorted(tab.loc[tab["side"] == "hi", "row"]) == sorted(hi_rows)
    assert len(tab) == 6                           # rows 20, 21 (at the clamp), 40, 41 excluded
    a64 = float(np.float32(0.3))
    mass = np.log(special.betainc(a64, a64, C))    # P(A <= c); alpha = beta so the hi tail is equal
    assert np.allclose(tab["log_mass"], mass, rtol=0, atol=1e-9)
    ref = np.where(tab["side"] == "lo", beta_dist.logcdf(C, a64, a64),
                   beta_dist.logsf(1 - C, a64, a64))
    assert np.allclose(tab["log_mass"], ref, rtol=0, atol=1e-12)
    # the training log-prob is the stored one and torch (float32) reproduces it
    assert np.array_equal(tab["logp_stored"], buf["old_logp"][tab["row"]].astype(np.float64))
    assert np.abs(tab["logp_torch"] - tab["logp_stored"]).max() <= 1e-5
    assert np.allclose(tab["diff_stored"], tab["logp_stored"] - mass, rtol=0, atol=1e-9)
    # density at the clipped value against the scipy density (float32 rounding of the action)
    pdf = beta_dist.logpdf(tab["action"].to_numpy(), a64, a64)
    assert np.allclose(tab["logp_stored"], pdf, rtol=0, atol=1e-3)


def test_hi_side_mass_is_the_mirrored_lo_tail_and_series_matches_scipy():
    a = np.array([0.3, 2.0, 5.0, 40.0])
    b = np.array([0.7, 3.0, 5.0, 90.0])
    hi = D.log_censored_mass(np.array(["hi"] * 4), a, b)
    ref = beta_dist.logsf(1 - C, a, b)
    fin = np.isfinite(ref)
    assert fin[:3].all() and np.allclose(hi[fin], ref[fin], rtol=0, atol=1e-12)
    assert np.isfinite(hi).all()                   # the underflowed entry comes from the series
    lo_mirror = D.log_censored_mass(np.array(["lo"] * 4), b, a)
    assert np.allclose(hi, lo_mirror, rtol=0, atol=1e-6)
    # the underflow fallback (leading series of the incomplete beta) against scipy
    x, aa, bb = 1e-3, np.array([3.0, 0.5, 12.0]), np.array([4.0, 2.5, 30.0])
    got = D._log_cdf_small_x(x, aa, bb)
    assert np.allclose(got, beta_dist.logcdf(x, aa, bb), rtol=0, atol=1e-11)
    # trained-policy scale: finite, equal to the leading terms with the first-order correction
    big = D.log_censored_mass(np.array(["lo"]), np.array([100.0]), np.array([100.0]))[0]
    lead = 100 * np.log(C) - np.log(100.0) - special.betaln(100.0, 100.0)
    corr = np.log1p(-(100.0 / 101.0) * 99.0 * C)
    assert np.isfinite(big) and abs(big - (lead + corr)) < 1e-6


def test_zero_clamped_rows_is_reported_not_nan():
    buf, _, _ = _known_buffer()
    buf["raw"] = np.clip(buf["raw"], 0.01, 0.99)
    tab = D.clamped_row_table(buf)
    assert len(tab) == 0 and list(tab.columns)[:2] == ["row", "side"]
    row = {"group": "g", "q": 50, "phase": "A", "n_clamped_policy": 0, "n_clamped_policy_lo": 0,
           "n_clamped_policy_hi": 0, "n_clamped_nonpolicy": 0}
    out = D.logdiff_by_group(pd.DataFrame([row]), tab)
    assert out.loc[0, "n_clamped_policy_rows"] == 0 and out.loc[0, "note"] == "no clamped rows"


# ====================================================================== gradient share
def _tiny_actor(seed: int = 0, hidden: int = 6) -> BetaActor:
    g = torch.Generator().manual_seed(seed)
    actor = BetaActor(hidden, 100.0, 1e-6, g)
    with torch.no_grad():
        for p in actor.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.6)
    return actor


def _tiny_buffer(actor: BetaActor, n: int = 14, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    st = torch.randn(n, 2, generator=g)
    with torch.no_grad():
        a, b = actor(st)
        act = torch.distributions.Beta(a, b).sample().clamp(1e-3, 1 - 1e-3)
        a2 = a + 1.0 * torch.randn(n, generator=g)       # ratio != 1, some rows clipped
        olp = torch.distributions.Beta(a2, b).log_prob(act)
    pm = np.ones(n, dtype=bool)
    pm[[2, 9]] = False
    return {"states": st.numpy(), "actions": act.numpy(), "old_logp": olp.numpy(),
            "adv_raw": torch.randn(n, generator=g).numpy() * 1.7 + 0.2, "policy_mask": pm,
            "adv_norm_mean": np.float64(0.15), "adv_norm_std": np.float64(1.6)}


def _brute(actor: BetaActor, buf, sets):
    """Per-row autograd gradients of -min(rA, clip(r)A) / n_policy, summed over each row set."""
    st = torch.as_tensor(buf["states"])
    ac = torch.as_tensor(buf["actions"])
    olp = torch.as_tensor(buf["old_logp"])
    adv = (torch.as_tensor(buf["adv_raw"]) - torch.tensor(float(buf["adv_norm_mean"]))) / (
        torch.tensor(float(buf["adv_norm_std"])) + 1e-8)
    n_pol = int(buf["policy_mask"].sum())
    params = list(actor.parameters())
    per_row = {}
    for i in np.nonzero(buf["policy_mask"])[0]:
        a, b = actor(st[i:i + 1])
        r = torch.exp(torch.distributions.Beta(a, b).log_prob(ac[i:i + 1]) - olp[i:i + 1])
        loss = -torch.min(r * adv[i], torch.clamp(r, 0.8, 1.2) * adv[i]).sum() / n_pol
        gs = torch.autograd.grad(loss, params)
        per_row[int(i)] = torch.cat([x.reshape(-1).double() for x in gs])
    zero = torch.zeros_like(next(iter(per_row.values())))
    return [sum((per_row[i] for i in s), zero) for s in sets], n_pol


def test_gradient_share_equals_brute_force_autograd():
    actor = _tiny_actor()
    buf = _tiny_buffer(actor)
    clamped = np.zeros(14, dtype=bool)
    clamped[[0, 5, 6, 9]] = True                   # row 9 is not a policy row: must be ignored
    pol = np.nonzero(buf["policy_mask"])[0]
    cl_rows = [i for i in pol if clamped[i]]
    wo_rows = [i for i in pol if not clamped[i]]
    (g_all, g_wo, g_cl), n_pol = _brute(actor, buf, [list(pol), wo_rows, cl_rows])
    res = D.gradient_share(actor, buf, clamped)
    na, nw, nc = (float(g.norm()) for g in (g_all, g_wo, g_cl))
    assert res["n_policy_rows"] == n_pol == 12 and res["n_clamped"] == 3
    for key, want in (("grad_norm_all", na), ("grad_norm_without", nw),
                      ("grad_norm_clamped", nc), ("share", 1 - nw / na),
                      ("clamped_over_all", nc / na)):
        assert res[key] == pytest.approx(want, rel=2e-4), key
    # the literal "mean over the remaining rows" variant rescales g_without by n / (n - m)
    renorm = nw * n_pol / len(wo_rows)
    assert res["grad_norm_without_renorm"] == pytest.approx(renorm, rel=2e-4)
    assert res["share_renorm"] == pytest.approx(1 - renorm / na, rel=2e-4, abs=1e-6)
    assert float((g_all - g_wo - g_cl).norm()) <= 1e-5 * na      # g_all = g_without + g_clamped
    # ratio deviation: mean |r - 1| over policy rows
    with torch.no_grad():
        a, b = actor(torch.as_tensor(buf["states"][pol]))
        lp = torch.distributions.Beta(a, b).log_prob(torch.as_tensor(buf["actions"][pol]))
        r = torch.exp(lp - torch.as_tensor(buf["old_logp"][pol]))
    assert res["ratio_dev_mean"] == pytest.approx(float((r - 1).abs().mean()), rel=1e-5)
    assert res["ratio_dev_max"] == pytest.approx(float((r - 1).abs().max()), rel=1e-5)
    assert not np.isnan(res["ratio_dev_clamped_mean"])


def test_gradient_share_without_clamped_rows_is_exactly_zero():
    actor = _tiny_actor(3)
    buf = _tiny_buffer(actor)
    res = D.gradient_share(actor, buf, np.zeros(14, dtype=bool))
    assert res["n_clamped"] == 0 and res["share"] == 0.0 and res["clamped_over_all"] == 0.0
    assert res["grad_norm_clamped"] == 0.0 and res["grad_norm_all"] > 0
    assert res["share_renorm"] == 0.0 and np.isnan(res["ratio_dev_clamped_mean"])


def test_actor_loaded_from_export_reproduces_the_agent_with_conc_scale(tmp_path):
    actor = _tiny_actor(5)
    actor.conc_scale = 2.5
    arrs = {f"actor.{k}": v.detach().numpy() for k, v in actor.state_dict().items()}
    arrs["conc_scale"] = np.asarray(2.5, dtype=np.float64)
    np.savez(tmp_path / "w.npz", **arrs)
    loaded = D.load_actor(tmp_path / "w.npz")
    x = torch.randn(9, 2, generator=torch.Generator().manual_seed(4))
    for u, v in zip(actor(x), loaded(x)):
        assert torch.equal(u, v)


# ====================================================================== M1 / M2 flag logic
def _prp(fracs, group="g", q=50, phase="A") -> pd.DataFrame:
    return pd.DataFrame({"group": group, "arm": "", "q": q, "seed": range(len(fracs)),
                         "phase": phase, "pol_frac": fracs})


def _grad(shares, group="g", q=50, phase="A", n_buf=3) -> pd.DataFrame:
    """Gradient-share rows: a scalar share applies to the last buffer, a list to all buffers."""
    rows = []
    for seed, s in enumerate(shares):
        for j in range(n_buf):
            v = s[j] if isinstance(s, (list, tuple)) else (s if j == n_buf - 1 else 0.0)
            rows.append({"group": group, "arm": "", "q": q, "seed": seed, "phase": phase,
                         "share": v})
    return pd.DataFrame(rows)


def _flag(prp, grad):
    return D.evaluate_flags(prp, grad).iloc[0]


def test_m1_is_strict_and_uses_the_median_over_runs():
    zero = _grad([0.0] * 20)
    assert _flag(_prp([0.001] * 20), zero)["M1_outcome"] == "not exceeded"   # exactly at threshold
    assert _flag(_prp([0.0011] * 20), zero)["M1_outcome"] == "exceeded"
    # ten runs at 0 and ten at 0.002: median (0 + 0.002) / 2 = 0.001 exactly -> not exceeded
    f = _flag(_prp([0.0] * 10 + [0.002] * 10), zero)
    assert f["M1_median_over_runs"] == 0.001 and f["M1_outcome"] == "not exceeded"
    assert f["M1_runs_above_threshold"] == 10
    # a single huge run does not move the median
    assert _flag(_prp([0.0] * 19 + [0.5]), zero)["M1_outcome"] == "not exceeded"
    # eleven of twenty runs above the threshold: median above
    assert _flag(_prp([0.0] * 9 + [0.002] * 11), zero)["M1_outcome"] == "exceeded"


def test_m2_counts_runs_with_any_saved_update_above_one_percent_strictly():
    ok = _prp([0.0] * 20)
    two = _flag(ok, _grad([0.0101] * 2 + [0.0] * 18))
    assert two["M2_outcome"] == "not exceeded"                     # 2 runs: not "more than 2"
    f = _flag(ok, _grad([0.0101] * 3 + [0.0] * 17))
    assert f["M2_outcome"] == "exceeded" and f["M2_runs_share_above_threshold"] == 3
    assert _flag(ok, _grad([0.01] * 20))["M2_outcome"] == "not exceeded"    # exactly 1%: not >
    # 'any saved update': the max over a run's buffers is what counts, once per run
    multi = _grad([[0.0, 0.5, 0.5]] * 2 + [[0.0, 0.0, 0.0]] * 18)
    f = _flag(ok, multi)
    assert f["M2_runs_share_above_threshold"] == 2 and f["M2_outcome"] == "not exceeded"
    assert f["M2_max_share"] == 0.5


def test_flags_with_fewer_than_20_runs_missing_gradient_and_per_phase_per_q():
    f = _flag(_prp([0.0] * 4), _grad([0.5] * 4))
    assert f["M2_outcome"] == "exceeded" and "4 runs (< 20)" in f["note"]   # literal reading
    nograd = D.evaluate_flags(_prp([0.0] * 3), pd.DataFrame())
    assert nograd.loc[0, "M2_outcome"] == "not evaluable (no gradient data)"
    gap = _grad([0.0] * 3)
    gap["share"] = np.nan
    assert D.evaluate_flags(_prp([0.0] * 3), gap).loc[0, "M2_n_runs_with_gradient_data"] == 0
    prp = pd.concat([_prp([0.0] * 5, phase="A"), _prp([0.01] * 5, phase="B"),
                     _prp([0.0] * 5, q=60, phase="A")], ignore_index=True)
    out = D.evaluate_flags(prp, pd.DataFrame())
    got = {(r.q, r.phase): r.M1_outcome for r in out.itertuples() if r.q != "all"}
    assert got == {(50, "A"): "not exceeded", (50, "B"): "exceeded", (60, "A"): "not exceeded"}
    pooled = {r.phase: (r.n_runs, r.M1_outcome) for r in out.itertuples() if r.q == "all"}
    assert pooled == {"A": (10, "not exceeded"), "B": (5, "exceeded")}      # pooled over q ("the 20 runs")
    # M2 pooled over q: 2 + 1 runs above 1% in different q make 3 > 2, while each q alone does not
    prp2 = pd.concat([_prp([0.0] * 10, q=50, phase="A"), _prp([0.0] * 10, q=60, phase="A")], ignore_index=True)
    g50, g60 = _grad([0.02] * 2 + [0.0] * 8, q=50, phase="A"), _grad([0.02] * 1 + [0.0] * 9, q=60, phase="A")
    o2 = D.evaluate_flags(prp2, pd.concat([g50, g60], ignore_index=True))
    m2 = {r.q: r.M2_outcome for r in o2.itertuples()}
    assert m2 == {50: "not exceeded", 60: "not exceeded", "all": "exceeded"}


# ====================================================================== synthetic run roots
def _row(update, phase, local, n=512, pol_lo=0, pol_hi=0, s2_lo=0, s2_hi=0, opp_lo=0, a_lt1=0):
    """One v2_updates.csv row with d1 columns (A: stage-2 rows; B: stage-1 policy + stage-2)."""
    r = {"update": update, "phase": phase, "local": local}
    for t in (1, 2):
        for who in ("L", "O"):
            for s in ("n", "lo", "hi"):
                r[f"d1_{who}_s{t}_{s}"] = 0
    if phase == "A":
        r.update(d1_L_s2_n=n, d1_L_s2_lo=pol_lo, d1_L_s2_hi=pol_hi, d1_O_s2_n=n,
                 d1_O_s2_lo=opp_lo)
    else:
        r.update(d1_L_s1_n=n, d1_L_s1_lo=pol_lo, d1_L_s1_hi=pol_hi, d1_O_s1_n=n,
                 d1_O_s1_lo=opp_lo, d1_L_s2_n=n, d1_L_s2_lo=s2_lo, d1_L_s2_hi=s2_hi, d1_O_s2_n=n)
    half = n // 2
    r.update(d1_L_s2_in_n=half, d1_L_s2_in_lo=r["d1_L_s2_lo"], d1_L_s2_in_hi=r["d1_L_s2_hi"],
             d1_L_s2_out_n=n - half, d1_L_s2_out_lo=0, d1_L_s2_out_hi=0)
    r.update(d1_pol_n_rows=n, d1_pol_alpha_min=50.0, d1_pol_beta_min=40.0,
             d1_pol_n_alpha_lt1=a_lt1, d1_pol_n_beta_lt1=0)
    return r


def _synth_buffer(actor: BetaActor, gu: int, local: int, n_lo: int, n_hi: int, n_np: int,
                  n: int = 64, seed: int = 0):
    """Consistent buffer from ``actor`` with known clamped policy rows (ratio = 1 at the actor)."""
    g = torch.Generator().manual_seed(seed + gu)
    st = torch.randn(n, 2, generator=g).numpy().astype(np.float32)
    with torch.no_grad():
        a, b = actor(torch.as_tensor(st))
        raw = torch.distributions.Beta(a, b).sample().clamp(0.02, 0.98).double().numpy()
        pm = np.ones(n, dtype=bool)
        pm[n - n_np:] = False                      # non-policy rows at the end
        raw[:n_lo] = 0.0
        raw[n_lo:n_lo + n_hi] = 1.0
        raw[n - n_np:] = 0.0                       # clamped, non-policy
        act = np.clip(raw, C, 1 - C).astype(np.float32)
        olp = torch.distributions.Beta(a, b).log_prob(torch.as_tensor(act)).numpy()
    adv = np.random.default_rng(seed + gu).normal(0.3, 1.5, n).astype(np.float32)
    return dict(states=st, raw=raw, actions=act, alpha=a.numpy(), beta=b.numpy(),
                stage=np.full(n, 2), old_logp=olp, adv_raw=adv, policy_mask=pm,
                adv_norm_mean=np.float64(adv[pm].mean()), adv_norm_std=np.float64(adv[pm].std()),
                local=np.int64(local), global_update=np.int64(gu))


def _make_synth_run(path: Path, hits_a=(0, 0), hits_b=(0, 0), with_weights=True, seed=0,
                    s2_nonpolicy_b=0):
    """Phases A (u1..30) and B (u31..60) with clamp hits (lo, hi) per update from update 1."""
    path.mkdir(parents=True, exist_ok=True)
    rows = [_row(u, "A", u, pol_lo=hits_a[0], pol_hi=hits_a[1], opp_lo=1) for u in range(1, 31)]
    rows += [_row(30 + u, "B", u, pol_lo=hits_b[0], pol_hi=hits_b[1], s2_lo=s2_nonpolicy_b,
                  a_lt1=2 if u == 1 else 0) for u in range(1, 31)]
    pd.DataFrame(rows).to_csv(path / "v2_updates.csv", index=False)
    actor = _tiny_actor(11 + seed, hidden=8)
    (path / "d1_buffers").mkdir(exist_ok=True)
    (path / "weights").mkdir(exist_ok=True)
    for gu, loc, hits in ((25, 25, hits_a), (55, 25, hits_b)):
        buf = _synth_buffer(actor, gu, loc, hits[0], hits[1], 1, seed=seed)
        np.savez(path / "d1_buffers" / f"u{gu:05d}.npz", **buf)
        if with_weights:
            arrs = {f"actor.{k}": v.detach().numpy() for k, v in actor.state_dict().items()}
            np.savez(path / "weights" / f"u{gu:05d}.npz", **arrs)


def test_discovery_layouts_arms_filter_and_labels(tmp_path):
    for p in ("rootA/q50/seed1", "rootA/q60/seed2", "rootB/q50/seed1/armX",
              "rootB/q50/seed1/armY"):
        _make_synth_run(tmp_path / p)
    (tmp_path / "rootA/q50/seed3").mkdir()
    (tmp_path / "rootA/q50/seed3/status.json").write_text("{}")        # unfinished run
    roots = [D.parse_root_spec(str(tmp_path / "rootA")),
             D.parse_root_spec(f"lab={tmp_path / 'rootB'}")]
    runs, skipped = D.discover_runs(roots)
    assert sorted((r.group, r.q, r.seed, r.arm) for r in runs) == [
        ("lab/armX", 50, 1, "armX"), ("lab/armY", 50, 1, "armY"), ("rootA", 50, 1, ""),
        ("rootA", 60, 2, "")]
    assert len(skipped) == 1 and skipped[0]["path"].endswith("seed3")
    only, sk = D.discover_runs(roots, arms=["armY"])
    assert [r.group for r in only] == ["lab/armY"] and sk == []


def test_policy_scope_mismatch_is_refused():
    df = pd.DataFrame([_row(1, "B", 1)])
    df["d1_pol_n_rows"] = 77
    with pytest.raises(ValueError, match="policy-row count"):
        D.derive_update_table(df)


def test_end_to_end_on_a_mocked_two_seed_root(tmp_path):
    root = tmp_path / "root"
    # q50 seed 1: no clamps; seed 2: 3 lo + 2 hi per phase-A update (n=512) from known buffers
    _make_synth_run(root / "q50/seed1", seed=1)
    _make_synth_run(root / "q50/seed2", hits_a=(3, 2), hits_b=(0, 0), seed=2, s2_nonpolicy_b=4)
    _make_synth_run(root / "q60/seed1", seed=3, with_weights=False)
    out, rep = tmp_path / "out", tmp_path / "rep" / "02.md"
    t = D.run_analysis([("root", root)], out, command="cmd", report_path=rep)
    prp = t["d1_per_run_phase"].set_index(["q", "seed", "phase"])
    r = prp.loc[(50, 2, "A")]
    assert (r["policy_scope"], r["pol_n"], r["pol_lo"], r["pol_hi"], r["n_updates"]) == (
        "all_stages", 30 * 512, 90, 60, 30)
    assert r["pol_frac"] == 150 / (30 * 512) and r["upd_frac_max"] == 5 / 512
    assert r["learner_s2_n"] == 30 * 512 and r["opponent_s2_lo"] == 30
    rb = prp.loc[(50, 2, "B")]
    assert rb["policy_scope"] == "stage1_only" and rb["pol_hit"] == 0
    assert rb["learner_s2_lo"] == 4 * 30 and rb["learner_all_n"] == 2 * 30 * 512
    r1b = prp.loc[(50, 1, "B")]
    assert r1b["pol_n_alpha_lt1"] == 2 and r1b["n_updates_alpha_lt1"] == 1
    bg = t["d1_by_group"].set_index(["q", "phase", "category"])
    s2b = bg.loc[(50, "B", "learner_s2")]            # reported, labelled non-policy
    assert s2b["policy_status"] == "non-policy (masked)" and s2b["lo_sum"] == 120
    assert bg.loc[(50, "B", "learner_s1"), "policy_status"] == "policy"
    pa = bg.loc[(50, "A", "learner_policy")]
    assert pa["n_runs"] == 2 and pa["hit_sum"] == 150 and pa["run_frac_max"] == 150 / (30 * 512)
    assert pa["run_frac_min"] == 0.0 and pa["upd_frac_min"] == 0.0
    assert pa["upd_frac_p90"] == 5 / 512
    # saved buffers: 3 lo + 2 hi clamped policy rows and 1 clamped non-policy row per buffer
    bufs = t["d1_buffers"].set_index(["q", "seed", "phase"])
    b = bufs.loc[(50, 2, "A")]
    assert (b["n_clamped_policy_lo"], b["n_clamped_policy_hi"], b["n_clamped_nonpolicy"]) == (
        3, 2, 1)
    assert b["max_abs_logp_recompute_diff_clamped"] < 1e-2 and not np.isnan(b["diff_median"])
    assert bufs.loc[(50, 1, "A"), "note"] == "no clamped rows"
    cl = t["d1_clamped_rows"]
    assert len(cl) == 5 and np.isfinite(cl["log_mass"]).all()
    assert np.isfinite(cl["diff_stored"]).all() and (cl["diff_stored"] > 0).all()
    # analytic check: pdf(c) / P(A <= c) = alpha / c (up to O(c beta)); the float32 action is c to
    # 3e-8 for the low clamp, but 1 - c only to 1.3% in (1 - a) for the high clamp (wider tol.)
    want = np.log(np.where(cl["side"] == "lo", cl["alpha"], cl["beta"]) / C)
    tol = np.where(cl["side"] == "lo", 1e-2, 1.0)
    assert (np.abs(cl["diff_stored"] - want) < tol).all()
    ld = t["d1_logdiff_by_group"].set_index(["q", "phase"])
    assert ld.loc[(50, "A"), "n_clamped_policy_rows"] == 5
    assert ld.loc[(50, "B"), "note"] == "no clamped rows"
    # gradient share: clamped rows exist only in seed 2 phase A; seed 1 buffers have zero share
    gs = t["d1_gradient_share"].set_index(["q", "seed", "phase"])
    g2 = gs.loc[(50, 2, "A")]
    assert g2["n_clamped"] == 5 and 0.0 < g2["share"] <= 1.0 and g2["grad_norm_clamped"] > 0
    assert gs.loc[(50, 1, "A"), "share"] == 0.0
    assert "no clamped rows" in gs.loc[(50, 1, "A"), "note"]
    assert "weights file missing" in gs.loc[(60, 1, "A"), "note"]
    assert np.isnan(gs.loc[(60, 1, "A"), "share"])
    # flags
    fl = t["d1_flags"].set_index(["q", "phase"])
    assert fl.loc[(50, "A"), "n_runs"] == 2
    assert fl.loc[(50, "A"), "M1_median_over_runs"] == (0.0 + 150 / (30 * 512)) / 2
    assert fl.loc[(50, "A"), "M1_outcome"] == "exceeded"             # median 0.0049 > 1e-3
    assert fl.loc[(50, "B"), "M1_outcome"] == "not exceeded"
    assert fl.loc[(60, "A"), "M1_outcome"] == "not exceeded"
    assert fl.loc[(60, "A"), "M2_outcome"] == "not evaluable (no gradient data)"
    assert fl.loc[(50, "A"), "M2_runs_share_above_threshold"] <= 1   # 2 runs: never "more than 2"
    assert fl.loc[(50, "A"), "M2_outcome"] == "not exceeded"
    # outputs on disk and the report
    for name in D.D1_FILES:
        assert (out / f"{name}.csv").exists()
    assert (out / "figures" / "d1_clamp_fraction_root.png").exists()
    assert (out / "figures" / "d1_clamp_fraction_root.pdf").exists()
    txt = rep.read_text()
    assert "# D1: likelihood consistency" in txt and "no clamped rows" in txt and "cmd" in txt
    assert "M1 exceeded" in txt and "M1 not exceeded" in txt
    assert "M2 not evaluable (no gradient data)" in txt and "Limitations" in txt


def test_m1_exceeded_on_a_mocked_root_with_frequent_clamps(tmp_path):
    root = tmp_path / "root"
    for s in (1, 2):
        _make_synth_run(root / f"q50/seed{s}", hits_a=(4, 4), seed=s)   # 8 / 512 per update
    t = D.run_analysis([("root", root)], tmp_path / "out")
    fl = t["d1_flags"].set_index(["q", "phase"])
    assert fl.loc[(50, "A"), "M1_median_over_runs"] == 8 / 512
    assert fl.loc[(50, "A"), "M1_outcome"] == "exceeded"
    assert fl.loc[(50, "B"), "M1_outcome"] == "not exceeded"


# ====================================================================== real runs (smoke config)
def _sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@pytest.fixture(scope="module")
def real_root(tmp_path_factory) -> Path:
    """q=50 seed 10501: a 25-update phase A, then a 25-update frozen phase B from its end state."""
    root = tmp_path_factory.mktemp("d1real") / "root"
    da = root / "q50/seed10501/A_real"
    da.mkdir(parents=True)
    cfg = base_config()
    cfg["flags"]["reward_mode"] = "expected"
    cfg["budget_overrides"]["phase_caps"]["A"] = 25
    cfg["budget_overrides"]["warmup"] = 1000
    assert execute(Run(cfg, str(da)), cfg, str(da), "pytest") == 0
    db = root / "q50/seed10501/B_real"
    db.mkdir(parents=True)
    cfg = base_config()
    parent = str(da / "state_end_A.pt")
    cfg.update(mode="phase_B", parent_checkpoint=parent, parent_sha256=_sha(parent))
    cfg["flags"].update(B2_MEAN)
    cfg["budget_overrides"]["phase_caps"]["B"] = 25
    cfg["budget_overrides"]["warmup"] = 1000
    assert execute(Run(cfg, str(db)), cfg, str(db), "pytest") == 0
    return root


def test_end_to_end_on_real_runs_phase_A_and_frozen_phase_B(real_root, tmp_path):
    out, rep = tmp_path / "out", tmp_path / "rep.md"
    t = D.run_analysis([("real", real_root)], out, command="pytest", report_path=rep)
    prp = t["d1_per_run_phase"].set_index(["group", "phase"])
    a, b = prp.loc[("real/A_real", "A")], prp.loc[("real/B_real", "B")]
    assert (a["policy_scope"], a["pol_n"], a["n_updates"]) == ("all_stages", 25 * 512, 25)
    assert (b["policy_scope"], b["pol_n"], b["n_updates"]) == ("stage1_only", 25 * 512, 25)
    assert b["learner_all_n"] == 2 * 25 * 512 and b["learner_s2_n"] == 25 * 512
    assert a["pol_hit"] == 0 and b["pol_hit"] == 0
    assert a["pol_n_alpha_lt1"] == 0                       # trained alpha, beta ~ 50
    bufs = t["d1_buffers"]
    assert len(bufs) == 2 and bufs["buffer_equals_csv"].all()
    assert (bufs["n_clamped_policy"] == 0).all()
    assert bufs["max_abs_logp_recompute_diff_policy_rows"].max() < 1e-5   # stored = torch's
    assert sorted(bufs["update"]) == [25, 50] and sorted(bufs["phase"]) == ["A", "B"]
    gs = t["d1_gradient_share"]
    assert len(gs) == 2 and gs["share"].eq(0.0).all() and (gs["grad_norm_all"] > 0).all()
    assert (gs["ratio_dev_mean"] < 1e-6).all() and gs["ratio_dev_clamped_mean"].isna().all()   # pre-update actor: ratio 1
    assert gs["weights_source"].str.contains("pre-update actor stored in the buffer").all()
    assert (gs["postexport_ratio_dev_mean"] > 0).all()                      # the post-update export is off-policy
    assert (gs["n_policy_rows"] == 512).all()
    fl = t["d1_flags"]
    assert (fl["M1_outcome"] == "not exceeded").all()
    assert (fl["M2_outcome"] == "not exceeded").all()
    assert (out / "figures").exists() and "no clamped rows" in rep.read_text()
    # zero clamped rows are reported as zeros, never NaN, in the tables the report is built from
    cols = ["rows_sum", "hit_sum", "frac_sum", "upd_frac_max"]
    assert not t["d1_by_group"][cols].isna().any().any()


def test_cli_with_arms_filter_on_real_root(real_root, tmp_path):
    out = tmp_path / "cli"
    rc = D.main(["--roots", f"x={real_root}", "--out", str(out), "--arms", "B_real",
                 "--report-path", str(tmp_path / "r.md")])
    assert rc == 0
    runs = pd.read_csv(out / "d1_runs.csv")
    assert list(runs["group"]) == ["x/B_real"] and list(runs["phases"]) == ["B"]
    assert (tmp_path / "r.md").exists()
