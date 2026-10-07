"""Tests of the MS-R1 start sampler (spec 2.2 "Sampler"; D5; D8 T=3 strata counts).

Run:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_sampler.py -q

Module under test: the MS-R1 additions of ``envs/curriculum_env.py::StartSampler``
(``tail_threshold``, ``tail_mask``, ``strata``, ``stratum_labels``, ``coverage_lambda_t``,
``stratified_bin_probs``, ``stratified_priority``).

What "polishing focus" means here: ``stratified_bin_probs(..., alpha, focus)`` with ``focus`` equal
to the EMA residual map restricted to a set S of non-tail bins (zero elsewhere), which is what
``utils.ms_rule.StageController`` passes in a polishing block. Support statement asserted below:
for alpha = 1 the probability vanishes outside tail U S; for 0 < alpha < 1 a non-tail bin outside S
keeps exactly its ``(1 - alpha) * p_strat`` part (the ``p_PM`` mixture component), the tail keeps
``lambda_T`` in total and the extra mass ``alpha * (1 - lambda_T)`` goes to S only.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Dict, Tuple

for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ.setdefault(f"{_k}_NUM_THREADS", "1")

import numpy as np
import pytest
from scipy.stats import kstest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.curriculum_env import GameSpec, StartSampler, gap_bin_index  # noqa: E402

# the base of ms-r1: curriculum_env.py before the MS-R1 additions
LOCKED_COMMIT = "4dc604de"
RECORDS = json.loads((ROOT / "protocols" / "v2_T2_locked_v2_0.json").read_text())["records"]
NEAR = 20.0                          # near-tie half-width (R2b's value)
N_DRAWS = 10 ** 6
SEED = 20261007
EXISTING_METHODS = ("n_bins", "bin_edges", "balanced", "peak_set", "peak_bin_probs", "peak_focused",
                    "root")


def spec_for(q: int, T: int = 2) -> GameSpec:
    """The v2.0 game of this q with horizon T."""
    return GameSpec(**{**RECORDS[str(q)]["game"], "q": float(q), "T": int(T)})


def sampler(q: int, T: int = 2) -> StartSampler:
    """Sampler with the ES bin width 10."""
    return StartSampler(spec_for(q, T), 10.0)


def p_strat_reference(sp: StartSampler, t: int, lam_p: float) -> np.ndarray:
    """Independent construction of p_strat from explicit bin counts (D5)."""
    n = sp.n_bins(t)
    edges = sp.bin_edges(t)
    lo, hi = edges[:-1], edges[1:]
    thr = sp.tail_threshold(t)
    tail = (lo >= thr - 1e-9) | (hi <= -thr + 1e-9)
    near = (lo < NEAR) & (hi > -NEAR)
    mid = ~tail & ~near
    lam_t = tail.sum() / n
    p = np.zeros(n)
    if tail.any():
        p[tail] = lam_t / tail.sum()
    p[near] = lam_p / near.sum()
    p[mid] = (1.0 - lam_p - lam_t) / mid.sum()
    return p


def bin_of(sp: StartSampler, t: int, d: np.ndarray) -> np.ndarray:
    """Bin index of draws (right-open bins, last bin closed)."""
    edges = sp.bin_edges(t)
    return np.clip(np.searchsorted(edges, d, side="right") - 1, 0, edges.size - 2)


# ----------------------------------------------------------------------------------------------
# strata counts and probabilities
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("q,n,n_tail,n_mid,lam_t",
                         [(50, 40, 20, 16, 0.5), (60, 44, 20, 20, 20 / 44)])
def test_strata_counts_t2(q, n, n_tail, n_mid, lam_t):
    sp = sampler(q)
    assert sp.n_bins(2) == n
    st = sp.strata(2, NEAR)
    assert set(st) == {"tail", "near", "mid"}
    counts = (int(st["tail"].sum()), int(st["near"].sum()), int(st["mid"].sum()))
    assert counts == (n_tail, 4, n_mid)
    assert np.array_equal(st["tail"] | st["near"] | st["mid"], np.ones(n, dtype=bool))
    assert not (st["tail"] & st["near"]).any() and not (st["tail"] & st["mid"]).any()
    assert not (st["near"] & st["mid"]).any()
    edges = sp.bin_edges(2)
    assert list(edges[:-1][st["near"]]) == [-20.0, -10.0, 0.0, 10.0]
    thr = 2.0 * q
    assert sp.tail_threshold(2) == thr
    # |bin| entirely in the tail
    assert np.all(np.abs(edges[:-1][st["tail"]] + 5.0) >= thr)
    assert np.all(np.abs(edges[:-1][~st["tail"]] + 5.0) < thr)
    assert sp.coverage_lambda_t(2) == pytest.approx(lam_t, abs=1e-15)
    lab = sp.stratum_labels(2, NEAR)
    assert lab.dtype == np.int8
    assert np.array_equal(lab == 0, st["tail"]) and np.array_equal(lab == 1, st["near"])
    assert np.array_equal(lab == 2, st["mid"])


@pytest.mark.parametrize("q", [50, 60])
@pytest.mark.parametrize("lam_p", [0.10, 0.25, 0.35])
def test_stratum_probabilities_t2(q, lam_p):
    sp = sampler(q)
    n = sp.n_bins(2)
    n_tail, n_near, n_mid = 20, 4, n - 24
    lam_t = n_tail / n
    lam_m = 1.0 - lam_p - lam_t
    if lam_m <= 0.0:
        pytest.skip("refused configuration (covered by the refusal tests)")
    p = sp.stratified_bin_probs(2, lam_p, NEAR)
    st = sp.strata(2, NEAR)
    assert p.sum() == pytest.approx(1.0, abs=1e-14) and p.min() > 0.0
    assert np.allclose(p[st["tail"]], lam_t / n_tail, rtol=1e-14, atol=0)
    assert np.allclose(p[st["near"]], lam_p / n_near, rtol=1e-14, atol=0)
    assert np.allclose(p[st["mid"]], lam_m / n_mid, rtol=1e-14, atol=0)
    assert p[st["tail"]].sum() == pytest.approx(lam_t, abs=1e-14)
    assert p[st["near"]].sum() == pytest.approx(lam_p, abs=1e-14)
    assert p[st["mid"]].sum() == pytest.approx(lam_m, abs=1e-14)
    np.testing.assert_allclose(p, p_strat_reference(sp, 2, lam_p), rtol=1e-14, atol=0)


def test_hard_coded_probabilities_q50_and_q60():
    """The numbers of D5: q = 50, lambda_P = 0.25: 0.025 / 0.0625 / 0.015625 per bin."""
    p = sampler(50).stratified_bin_probs(2, 0.25, NEAR)
    st = sampler(50).strata(2, NEAR)
    assert np.allclose(p[st["tail"]], 0.025, rtol=1e-14, atol=0)
    assert np.allclose(p[st["near"]], 0.0625, rtol=1e-14, atol=0)
    assert np.allclose(p[st["mid"]], 0.015625, rtol=1e-14, atol=0)
    p = sampler(60).stratified_bin_probs(2, 0.35, NEAR)
    st = sampler(60).strata(2, NEAR)
    assert np.allclose(p[st["tail"]], 1.0 / 44.0, rtol=1e-14, atol=0)
    assert np.allclose(p[st["near"]], 0.0875, rtol=1e-14, atol=0)
    assert np.allclose(p[st["mid"]], (1.0 - 0.35 - 20 / 44) / 20, rtol=1e-14, atol=0)


@pytest.mark.parametrize("q", [50, 60])
def test_ms_rule_arm_is_bin_balanced_in_distribution(q):
    """MS_rule: lambda_P = the bin-balanced near-tie share (4/n) gives 1/n on every bin."""
    sp = sampler(q)
    n = sp.n_bins(2)
    p = sp.stratified_bin_probs(2, 4.0 / n, NEAR)
    np.testing.assert_allclose(p, np.full(n, 1.0 / n), rtol=1e-13, atol=0)


# ----------------------------------------------------------------------------------------------
# alpha = 0, tail share property, focus supports
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("q", [50, 60])
def test_alpha_zero_gives_p_strat_exactly(q):
    """alpha = 0 ignores the focus: p equals p_strat to <= 1 ulp-level (rtol 1e-14)."""
    sp = sampler(q)
    n = sp.n_bins(2)
    ref = p_strat_reference(sp, 2, 0.25)
    rng = np.random.default_rng(3)
    worst = 0.0
    sparse = rng.random(n) * (rng.random(n) < 0.2)
    for focus in (None, np.zeros(n), np.ones(n), rng.random(n), sparse):
        p = sp.stratified_bin_probs(2, 0.25, NEAR, alpha=0.0, focus=focus)
        np.testing.assert_allclose(p, ref, rtol=1e-14, atol=0)
        worst = max(worst, float(np.abs(p - ref).max()))
    assert worst <= 1e-16


@pytest.mark.parametrize("q,t_big,lam_cap",
                         [(50, 2, 0.45), (60, 2, 0.5), (50, 3, 0.2), (60, 3, 0.2)])
def test_tail_share_equals_lambda_T_for_every_focus_and_alpha(q, t_big, lam_cap):
    """Property test: random residual maps, alphas in [0, 1], lambda_P; tail share = lambda_T."""
    T = 3 if t_big == 3 else 2
    sp = sampler(q, T)
    n = sp.n_bins(t_big)
    tail = sp.strata(t_big, NEAR)["tail"]
    lam_t = sp.coverage_lambda_t(t_big)
    n_tail = int(tail.sum())
    n_cases = 0
    for seed in range(12):
        rng = np.random.default_rng(1000 + seed)
        for alpha in (0.0, 1.0, float(rng.random()), float(rng.random()), 0.5):
            lam_p = float(rng.uniform(0.02, lam_cap))
            kind = seed % 4
            if kind == 0:
                focus = rng.random(n)
            elif kind == 1:
                focus = rng.random(n) * (rng.random(n) < 0.15)          # sparse residual map
            elif kind == 2:
                focus = rng.random(n) ** 6 * 1e3                         # heavy-tailed scale
            else:
                focus = np.where(tail, 50.0, 0.0) + rng.random(n) * 1e-3   # garbage on the tail
            p = sp.stratified_bin_probs(t_big, lam_p, NEAR, alpha=alpha, focus=focus)
            assert p.min() >= 0.0 and p.sum() == pytest.approx(1.0, abs=1e-13)
            assert p[tail].sum() == pytest.approx(lam_t, abs=1e-12), (q, seed, alpha)
            if n_tail:
                assert np.allclose(p[tail], lam_t / n_tail, rtol=1e-13, atol=0)
            assert p[~tail].sum() == pytest.approx(1.0 - lam_t, abs=1e-12)
            n_cases += 1
    assert n_cases == 60


def test_focus_proportional_in_a_global_block_and_all_zero_focus_is_p_pm():
    sp = sampler(50)
    n = sp.n_bins(2)
    lam_t = 0.5
    tail = sp.strata(2, NEAR)["tail"]
    rng = np.random.default_rng(8)
    focus = np.where(tail, 0.0, rng.random(n) + 0.1)
    p1 = sp.stratified_bin_probs(2, 0.25, NEAR, alpha=1.0, focus=focus)
    np.testing.assert_allclose(p1[~tail], (1.0 - lam_t) * focus[~tail] / focus.sum(), rtol=1e-13)
    # alpha = 0.5: a convex mixture of p_PM and f on the non-tail bins
    ref = p_strat_reference(sp, 2, 0.25)
    p5 = sp.stratified_bin_probs(2, 0.25, NEAR, alpha=0.5, focus=focus)
    expect = np.where(tail, ref, 0.5 * ref + 0.5 * (1.0 - lam_t) * focus / focus.sum())
    np.testing.assert_allclose(p5, expect, rtol=1e-13, atol=1e-18)
    # an all-zero focus on the non-tail bins (positive garbage on the tail) and None give p_PM
    junk = np.where(tail, 7.0, 0.0)
    for f in (None, np.zeros(n), junk):
        np.testing.assert_allclose(sp.stratified_bin_probs(2, 0.25, NEAR, alpha=0.7, focus=f), ref,
                                   rtol=1e-13, atol=0)


@pytest.mark.parametrize("q", [50, 60])
def test_polishing_focus_is_supported_on_S_plus_tail_plus_the_p_pm_part(q):
    """Focus = rho_bar restricted to S: support is tail U S at alpha = 1; at alpha < 1 the bins
    outside S keep exactly (1 - alpha) p_strat, S gets the whole extra mass alpha (1 - lambda_T)."""
    sp = sampler(q)
    n = sp.n_bins(2)
    st = sp.strata(2, NEAR)
    lam_t = sp.coverage_lambda_t(2)
    rng = np.random.default_rng(31)
    nontail = np.flatnonzero(~st["tail"])
    S = np.sort(rng.choice(nontail, size=4, replace=False))
    rho_bar = rng.random(n) * 0.2 + 0.01
    focus = np.zeros(n)
    focus[S] = rho_bar[S]
    in_s = np.zeros(n, dtype=bool)
    in_s[S] = True
    ref = p_strat_reference(sp, 2, 0.30)
    # alpha = 1: support is tail U S, no mass on the other non-tail bins
    p1 = sp.stratified_bin_probs(2, 0.30, NEAR, alpha=1.0, focus=focus)
    other = ~st["tail"] & ~in_s
    assert np.all(p1[other] == 0.0) and np.all(p1[S] > 0.0)
    assert p1[st["tail"]].sum() == pytest.approx(lam_t, abs=1e-13)
    np.testing.assert_allclose(p1[S], (1.0 - lam_t) * rho_bar[S] / rho_bar[S].sum(), rtol=1e-13)
    # the polishing setting of D5: alpha = 0.5
    a = 0.5
    p = sp.stratified_bin_probs(2, 0.30, NEAR, alpha=a, focus=focus)
    np.testing.assert_allclose(p[other], (1.0 - a) * ref[other], rtol=1e-13, atol=0)
    np.testing.assert_allclose(p[st["tail"]], ref[st["tail"]], rtol=1e-13, atol=0)
    extra = p[S] - (1.0 - a) * ref[S]
    np.testing.assert_allclose(extra, a * (1.0 - lam_t) * rho_bar[S] / rho_bar[S].sum(), rtol=1e-12)
    assert extra.sum() == pytest.approx(a * (1.0 - lam_t), abs=1e-13)
    assert p.sum() == pytest.approx(1.0, abs=1e-13)


# ----------------------------------------------------------------------------------------------
# empirical shares of 10^6 draws
# ----------------------------------------------------------------------------------------------

def _configs() -> Dict[str, Tuple[int, float, float, str]]:
    return {
        "q50_global_a0": (50, 0.25, 0.0, "none"),
        "q50_global_a5": (50, 0.25, 0.5, "global"),
        "q60_global_a5": (60, 0.35, 0.5, "global"),
        "q60_polish": (60, 0.35, 0.5, "polish"),
        "q50_polish": (50, 0.25, 0.5, "polish"),
    }


@pytest.mark.parametrize("name", list(_configs()))
def test_empirical_shares_of_a_million_draws(name):
    """Per stratum within 3 binomial SE; per bin within 4.5 SE (40-44 simultaneous comparisons);
    the start inside a bin is uniform (KS)."""
    q, lam_p, alpha, kind = _configs()[name]
    sp = sampler(q)
    n = sp.n_bins(2)
    st = sp.strata(2, NEAR)
    rng = np.random.default_rng(SEED + q)
    focus = None
    if kind == "global":
        focus = rng.random(n) + 0.05
    elif kind == "polish":
        S = np.sort(rng.choice(np.flatnonzero(~st["tail"]), size=5, replace=False))
        focus = np.zeros(n)
        focus[S] = rng.random(S.size) + 0.2
    p = sp.stratified_bin_probs(2, lam_p, NEAR, alpha=alpha, focus=focus)
    d = sp.stratified_priority(2, N_DRAWS, rng, p)
    edges = sp.bin_edges(2)
    assert d.min() >= edges[0] and d.max() <= edges[-1]
    b = bin_of(sp, 2, d)
    cnt = np.bincount(b, minlength=n)
    for key in ("tail", "near", "mid"):
        ps = float(p[st[key]].sum())
        share = cnt[st[key]].sum() / N_DRAWS
        assert abs(share - ps) <= 3.0 * np.sqrt(ps * (1.0 - ps) / N_DRAWS), (name, key, share, ps)
    se = np.sqrt(N_DRAWS * p * (1.0 - p))
    z = np.abs(cnt - N_DRAWS * p)[p > 0] / se[p > 0]
    assert z.max() <= 4.5, (name, z.max())
    assert np.all(cnt[p == 0.0] == 0)
    for i in list(np.flatnonzero(st["near"])[:1]) + list(np.flatnonzero(st["mid"])[:1]):
        u = (d[b == i] - edges[i]) / (edges[i + 1] - edges[i])
        assert kstest(u, "uniform").pvalue > 1e-3, (name, i)


def test_draws_on_a_stage_without_tail_t3():
    sp = sampler(50, T=3)
    p = sp.stratified_bin_probs(2, 0.25, NEAR)
    rng = np.random.default_rng(SEED)
    d = sp.stratified_priority(2, 200_000, rng, p)
    cnt = np.bincount(bin_of(sp, 2, d), minlength=40)
    near = sp.strata(2, NEAR)["near"]
    share = cnt[near].sum() / 200_000
    assert abs(share - 0.25) <= 3.0 * np.sqrt(0.25 * 0.75 / 200_000)
    assert np.abs(d).max() <= sp.spec.domain_half(2)


# ----------------------------------------------------------------------------------------------
# the locked paths are untouched; stream consumption
# ----------------------------------------------------------------------------------------------

def _locked_source() -> str:
    try:
        return subprocess.run(["git", "show", f"{LOCKED_COMMIT}:envs/curriculum_env.py"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout
    except Exception as exc:                                         # pragma: no cover
        pytest.skip(f"locked source unavailable: {exc}")


def _locked_module() -> types.ModuleType:
    mod = types.ModuleType("curriculum_env_locked_4dc604de")
    sys.modules[mod.__name__] = mod
    exec(compile(_locked_source(), "curriculum_env_locked", "exec"), mod.__dict__)
    return mod


def test_existing_methods_are_textually_unchanged():
    """The pre-existing methods of StartSampler (and the module functions) are byte-identical to the
    base commit: only methods are added."""
    old_src, new_src = _locked_source(), (ROOT / "envs" / "curriculum_env.py").read_text()
    old_t, new_t = ast.parse(old_src), ast.parse(new_src)

    def segments(tree: ast.Module, src: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name != "StartSampler":
                out[node.name] = ast.get_source_segment(src, node)
            if isinstance(node, ast.ClassDef) and node.name == "StartSampler":
                for f in node.body:
                    if isinstance(f, ast.FunctionDef):
                        out[f"StartSampler.{f.name}"] = ast.get_source_segment(src, f)
        return out

    old_s, new_s = segments(old_t, old_src), segments(new_t, new_src)
    old_methods = {k.split(".")[1] for k in old_s if k.startswith("StartSampler.")}
    assert set(EXISTING_METHODS) <= old_methods
    for name, text in old_s.items():
        assert new_s.get(name) == text, name
    added = set(new_s) - set(old_s)
    assert added == {f"StartSampler.{m}" for m in (
        "tail_threshold", "tail_mask", "strata", "stratum_labels", "coverage_lambda_t",
        "stratified_bin_probs", "stratified_priority")}


@pytest.mark.parametrize("q,t,n", [(50, 2, 512), (60, 2, 512), (50, 2, 1), (60, 2, 1000)])
def test_balanced_equals_the_locked_sampler_values_and_stream_position(q, t, n):
    locked = _locked_module()
    old = locked.StartSampler(locked.GameSpec(**{**RECORDS[str(q)]["game"], "q": float(q), "T": 2}),
                              10.0)
    new = sampler(q)
    r_old, r_new, r_ref = (np.random.default_rng(77) for _ in range(3))
    for _ in range(3):                                   # three consecutive updates
        a, b = old.balanced(t, n, r_old), new.balanced(t, n, r_new)
        assert np.array_equal(a, b)
        assert r_old.bit_generator.state == r_new.bit_generator.state
        edges = new.bin_edges(t)                         # explicit replay of the documented calls
        bb = r_ref.integers(0, edges.size - 1, size=n)
        uu = r_ref.random(n)
        assert np.array_equal(b, edges[bb] + uu * (edges[bb + 1] - edges[bb]))
        assert r_ref.bit_generator.state == r_new.bit_generator.state
    assert np.array_equal(old.peak_focused(t, n, np.random.default_rng(5), 20.0, 0.5),
                          new.peak_focused(t, n, np.random.default_rng(5), 20.0, 0.5))


@pytest.mark.parametrize("q", [50, 60])
@pytest.mark.parametrize("n", [1, 512, 1000])
@pytest.mark.parametrize("share", [0.25, 0.5])
def test_stratified_priority_consumes_the_start_stream_like_peak_focused(q, n, share):
    sp = sampler(q)
    probs = sp.peak_bin_probs(2, 20.0, share)
    r1, r2, r3, r4 = (np.random.default_rng(12) for _ in range(4))
    for _ in range(3):                                         # stream position after k updates
        a = sp.peak_focused(2, n, r1, 20.0, share)
        b = sp.stratified_priority(2, n, r2, probs)
        assert np.array_equal(a, b)                            # identical output for equal probs
        assert r1.bit_generator.state == r2.bit_generator.state
        r3.random(n)
        r3.random(n)                                           # exactly two random(n) calls
        assert r2.bit_generator.state == r3.bit_generator.state
    # a stratified draw advances the stream the same way whatever the probabilities are
    p_s = sp.stratified_bin_probs(2, 0.25, NEAR, alpha=0.5, focus=np.random.default_rng(1).random(
        sp.n_bins(2)))
    sp.stratified_priority(2, n, r4, p_s)
    sp.stratified_priority(2, n, r4, p_s)
    sp.stratified_priority(2, n, r4, p_s)
    assert r4.bit_generator.state == r1.bit_generator.state
    # and it desynchronises from the locked bin-balanced scheme (which draws integers)
    r5 = np.random.default_rng(12)
    sp.balanced(2, n, r5)
    r6 = np.random.default_rng(12)
    sp.stratified_priority(2, n, r6, probs)
    assert r5.bit_generator.state != r6.bit_generator.state


def test_stratified_priority_replays_the_bin_draw_then_the_position_draw():
    sp = sampler(50)
    probs = sp.stratified_bin_probs(2, 0.25, NEAR, alpha=0.5,
                                    focus=np.random.default_rng(2).random(sp.n_bins(2)))
    got = sp.stratified_priority(2, 257, np.random.default_rng(21), probs)
    r = np.random.default_rng(21)
    ub, up = r.random(257), r.random(257)
    edges = sp.bin_edges(2)
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0
    b = np.minimum(np.searchsorted(cdf, ub, side="right"), edges.size - 2)
    assert np.array_equal(got, edges[b] + up * (edges[b + 1] - edges[b]))
    # every draw falls in the bin chosen by the first call
    assert np.array_equal(gap_bin_index(sp.spec, 2, got)[0], b)


def test_stratified_priority_refuses_malformed_probabilities():
    sp = sampler(50)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sp.stratified_priority(2, 5, rng, np.full(39, 1.0 / 39))              # wrong length
    with pytest.raises(ValueError):
        sp.stratified_priority(2, 5, rng, np.full(40, 1.0 / 40) * 1.001)      # does not sum to 1
    with pytest.raises(ValueError):
        sp.stratified_priority(2, 5, rng, np.full(40, 1.0 / 40) * 0.999)


# ----------------------------------------------------------------------------------------------
# refusals
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lam_p", [0.0, 1.0, -0.1, 1.5])
def test_lambda_P_outside_open_unit_interval_is_refused(lam_p):
    with pytest.raises(ValueError, match="lambda_P"):
        sampler(50).stratified_bin_probs(2, lam_p, NEAR)


@pytest.mark.parametrize("q,lam_p,ok", [(50, 0.6, False), (50, 0.5, False), (50, 0.499, True),
                                        (60, 0.55, False), (60, 0.5, True), (50, 0.45, True)])
def test_lambda_M_must_be_positive(q, lam_p, ok):
    """lambda_M = 1 - lambda_P - lambda_T <= 0 is refused (q = 50: lambda_T = 0.5)."""
    sp = sampler(q)
    if ok:
        p = sp.stratified_bin_probs(2, lam_p, NEAR)
        assert p.min() > 0 and p.sum() == pytest.approx(1.0)
    else:
        with pytest.raises(ValueError, match="lambda_M"):
            sp.stratified_bin_probs(2, lam_p, NEAR)


@pytest.mark.parametrize("alpha", [-0.01, 1.01])
def test_alpha_outside_unit_interval_is_refused(alpha):
    with pytest.raises(ValueError, match="alpha"):
        sampler(50).stratified_bin_probs(2, 0.25, NEAR, alpha=alpha)


def test_malformed_focus_is_refused():
    sp = sampler(50)
    for bad in (np.ones(39), -np.ones(40), np.full(40, np.nan), np.full(40, np.inf)):
        with pytest.raises(ValueError, match="focus"):
            sp.stratified_bin_probs(2, 0.25, NEAR, alpha=0.5, focus=bad)


def test_near_tie_half_width_that_reaches_the_tail_or_empties_the_middle_is_refused():
    sp = sampler(50)
    # the tail starts at |d| = 100: half-width 100 makes all 20 non-tail bins near-tie (the window
    # (-100, 100) just misses the first tail bin) -> the middle stratum is empty
    with pytest.raises(ValueError, match="0 middle"):
        sp.strata(2, 100.0)
    for h in (100.5, 120.0, 400.0):
        with pytest.raises(ValueError, match="reaches the tail"):
            sp.strata(2, h)
    with pytest.raises(ValueError):
        sp.strata(2, 0.0)                       # no bin intersects (0, 0): near-tie empty
    for fn in (lambda: sp.stratified_bin_probs(2, 0.25, 100.5),
               lambda: sp.stratum_labels(2, 100.5)):
        with pytest.raises(ValueError):
            fn()
    # a stage without tail: half-width covers D_2
    with pytest.raises(ValueError, match="middle"):
        sampler(50, T=3).strata(2, 400.0)


# ----------------------------------------------------------------------------------------------
# T = 3 strata (D8)
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("q,n2,n3,tail3", [(50, 40, 80, 60), (60, 44, 88, 64)])
def test_t3_strata_counts(q, n2, n3, tail3):
    """D_2: n bins and no tail bin (4 q >= B); D_3: 2n bins with the tail at |d| >= 2 q."""
    sp = sampler(q, T=3)
    assert sp.n_bins(2) == n2 and sp.n_bins(3) == n3
    assert sp.tail_threshold(2) == 4.0 * q and sp.tail_threshold(3) == 2.0 * q
    s2, s3 = sp.strata(2, NEAR), sp.strata(3, NEAR)
    assert int(s2["tail"].sum()) == 0 and int(s2["near"].sum()) == 4
    assert int(s2["mid"].sum()) == n2 - 4
    assert int(s3["tail"].sum()) == tail3 and int(s3["near"].sum()) == 4
    assert int(s3["mid"].sum()) == n3 - tail3 - 4
    assert sp.coverage_lambda_t(2) == 0.0
    assert sp.coverage_lambda_t(3) == pytest.approx(tail3 / n3, abs=1e-15)
    edges3 = sp.bin_edges(3)
    assert np.all(np.abs(edges3[:-1][s3["tail"]] + 5.0) >= 2 * q)
    assert np.all(np.abs(edges3[:-1][~s3["tail"]] + 5.0) < 2 * q)


@pytest.mark.parametrize("q", [50, 60])
def test_stage_without_tail_has_lambda_T_zero_and_a_valid_distribution(q):
    sp = sampler(q, T=3)
    n = sp.n_bins(2)
    assert sp.coverage_lambda_t(2) == 0.0
    rng = np.random.default_rng(4)
    for lam_p, alpha, focus in ((0.25, 0.0, None), (0.35, 0.5, rng.random(n)), (0.9, 1.0,
                                                                                 rng.random(n))):
        p = sp.stratified_bin_probs(2, lam_p, NEAR, alpha=alpha, focus=focus)
        assert p.shape == (n,) and p.min() >= 0.0 and p.sum() == pytest.approx(1.0, abs=1e-14)
    p0 = sp.stratified_bin_probs(2, 0.25, NEAR)
    near = sp.strata(2, NEAR)["near"]
    assert np.allclose(p0[near], 0.25 / 4, rtol=1e-14) and np.allclose(p0[~near], 0.75 / (n - 4),
                                                                      rtol=1e-14)
    # lambda_P up to (but excluding) 1 is allowed when lambda_T = 0
    assert sp.stratified_bin_probs(2, 0.99, NEAR).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("q,lam_t3", [(50, 0.75), (60, 64 / 88)])
def test_t3_terminal_stage_tail_share_and_refusals(q, lam_t3):
    sp = sampler(q, T=3)
    p = sp.stratified_bin_probs(3, 0.10, NEAR)
    tail = sp.strata(3, NEAR)["tail"]
    assert p[tail].sum() == pytest.approx(lam_t3, abs=1e-14)
    # lambda_P = 0.25 gives lambda_M = 0 at q = 50 (D_3 tail share 0.75): refused; 0.35 refused too
    with pytest.raises(ValueError, match="lambda_M"):
        sp.stratified_bin_probs(3, 0.35, NEAR)
    if q == 50:
        with pytest.raises(ValueError, match="lambda_M"):
            sp.stratified_bin_probs(3, 0.25, NEAR)
