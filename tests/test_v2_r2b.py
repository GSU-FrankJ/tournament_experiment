"""Round R2b tests: peak-focused starts, censored likelihood of clamped draws, phase-P optimiser budget.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_v2_r2b.py -p no:cacheprovider -q

Sections: defaults are bit-identical (C7 reference; the v2.0 lock commit's own code on a reduced-budget
locked run), the start sampler, the censored likelihood (series vs scipy, gradient, update, rollout),
phase P with E epochs x M minibatches, config refusals, the launcher arms.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
from scipy.special import betainc
from scipy.stats import kstest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import launch_refine as L  # noqa: E402
from agents.ppo_curriculum import PPOConfig  # noqa: E402
from agents.ppo_curriculum_v2 import CurriculumPPOv2, masked_actor_loss  # noqa: E402
from agents.ppo_pathwise import pathwise_step, pathwise_update  # noqa: E402
from compare_runs import _walk  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402
from rng_alignment import base_config, make_run  # noqa: E402
from run.run_final_dp_br_round3_dense import ConfigError  # noqa: E402
from run.run_v2_stagewise import Run, execute, validate_config  # noqa: E402
from run.v2_rollout import collect_batch_v2  # noqa: E402
from utils.beta_tail import DOMAIN_X_B, log_betainc_small_x, row_log_prob  # noqa: E402

C7_CONFIG = ROOT / "results/v2_pilots/phase2_regression/run_config.json"
C7_BEFORE = ROOT / "results/v2_pilots/phase1_regression/before/FINAL_A400_B25_C25/tel_q50_s10501"
LOCK_COMMIT = "1d6d4d0"        # the v2.0 lock commit (code identical to the head of v2-t2-refine)
CANON = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2")
R1_ROOT = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_refine")
RECORDS = json.loads((ROOT / "protocols" / "v2_T2_locked_v1_1.json").read_text())["records"]
NEW_KEYS_AT_DEFAULT = {"start_weights": None, "clamp_likelihood": "density", "pathwise_epochs": 1,
                       "pathwise_minibatch": None}


def spec_for(q: int) -> GameSpec:
    """The locked-protocol game of this q."""
    return GameSpec(**{**RECORDS[str(q)]["game"], "q": float(q)})


def tree_equal(a, b) -> bool:
    """Exact (bitwise for tensors / arrays) equality of nested structures."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


# ====================================================================== 1. defaults are bit-identical
def _c7_equal(d: str) -> None:
    for f in ("train_history.json", "final_eval.json"):
        diffs, skipped = [], set()
        _walk(json.load(open(C7_BEFORE / f)), json.load(open(os.path.join(d, f))), f, diffs, skipped)
        assert diffs == [], diffs[:3]
    for f in ("arrays.npz", "checkpoint_weights.npz", "phase_A_exit_arrays.npz", "phase_B_exit_arrays.npz"):
        za, zb = np.load(C7_BEFORE / f), np.load(os.path.join(d, f))
        assert set(za.files) == set(zb.files)
        assert all(np.array_equal(za[k], zb[k]) for k in za.files), f


def test_new_keys_at_default_equal_the_c7_reference(tmp_path):
    """The four R2b keys written at their defaults: the regression pipeline equals the C7 reference."""
    cfg = json.load(open(C7_CONFIG))
    cfg.update(NEW_KEYS_AT_DEFAULT)
    d = str(tmp_path / "c7")
    os.makedirs(d)
    assert execute(Run(cfg, d), cfg, d, "pytest") == 0
    _c7_equal(d)


_V20_RUNNER = textwrap.dedent('''
    import json, os, sys
    sys.path.insert(0, sys.argv[1])
    for k in ("OMP", "MKL", "OPENBLAS"):
        os.environ[k + "_NUM_THREADS"] = "1"
    import run.run_v2_T2_locked as L
    proto = L.load_protocol()
    out = sys.argv[2]
    cfg = L.build_config(proto, 50, 10501, out)
    cfg["budget_overrides"] = {"phase_caps": {"A": 6, "B": 4, "C": 1}, "warmup": 3, "stability_every": 2,
                               "verifier_timeout": 3}
    cfg["lr_decay"] = [{"phase": "A", "start_lr": 3e-4, "end_lr": 3e-5, "local_first": 4, "local_last": 6},
                       {"phase": "B", "start_lr": 3e-4, "end_lr": 3e-5, "local_first": 1, "local_last": 4}]
    os.makedirs(out, exist_ok=True)
    sys.exit(L.run_pipeline(cfg, proto, L.PROTOCOL_SHA256, out, "pytest", band_step=2.0))
''')


def _run_locked_small(code_root: str, out: str) -> None:
    """Reduced-budget locked pipeline of the code tree at ``code_root`` in a fresh interpreter."""
    r = subprocess.run([sys.executable, "-B", "-c", _V20_RUNNER, code_root, out], cwd=code_root,
                       capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


@pytest.fixture(scope="module")
def v20_reference_tree(tmp_path_factory):
    """The code of the v2.0 lock commit, extracted with ``git archive`` (skipped without the commit)."""
    ok = subprocess.run(["git", "cat-file", "-e", f"{LOCK_COMMIT}^{{commit}}"], cwd=ROOT, capture_output=True)
    if ok.returncode != 0:
        pytest.skip(f"commit {LOCK_COMMIT} is not in this clone")
    d = tmp_path_factory.mktemp("v20_code")
    arch = subprocess.run(["git", "archive", LOCK_COMMIT, "agents", "envs", "run", "utils", "protocols", "tools/v2"],
                          cwd=ROOT, capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(d)], input=arch.stdout, check=True)
    for extra in ("config", "experiments"):          # read-only inputs of the record (symlinks, not copies)
        if (ROOT / extra).exists():
            os.symlink(ROOT / extra, d / extra)
    return str(d)


def test_defaults_equal_the_v20_lock_commit_on_a_reduced_locked_run(v20_reference_tree, tmp_path):
    """Reduced-budget locked pipeline (A then B with the expected-continuation table): the current code
    and the code of the v2.0 lock commit end in bit-identical training-relevant state."""
    ref_out, new_out = str(tmp_path / "ref"), str(tmp_path / "new")
    _run_locked_small(v20_reference_tree, ref_out)
    _run_locked_small(str(ROOT), new_out)
    for st in ("state_end_A.pt", "state_end_B.pt"):
        a = torch.load(os.path.join(ref_out, st), weights_only=False)
        b = torch.load(os.path.join(new_out, st), weights_only=False)
        for k in ("agent", "rng", "torch_generator_state", "counters"):
            assert tree_equal(a[k], b[k]), (st, k)
    za, zb = np.load(os.path.join(ref_out, "continuation_table.npz")), np.load(os.path.join(new_out, "continuation_table.npz"))
    assert all(np.array_equal(za[k], zb[k]) for k in ("y_grid", "values"))
    ha = json.load(open(os.path.join(ref_out, "train_history.json")))["history"]
    hb = json.load(open(os.path.join(new_out, "train_history.json")))["history"]
    drop = {"time_sec", "update_wall_sec"}
    assert [{k: v for k, v in r.items() if k not in drop} for r in ha] == [{k: v for k, v in r.items() if k not in drop} for r in hb]


# ====================================================================== 2. the start sampler
@pytest.mark.parametrize("q", [50, 60])
def test_peak_set_is_four_bins_and_bin_probabilities(q):
    sp = StartSampler(spec_for(q), 10.0)
    nb = sp.n_bins(2)
    assert nb == {50: 40, 60: 44}[q]
    m = sp.peak_set(2, 20.0)
    edges = sp.bin_edges(2)
    assert m.sum() == 4 and list(edges[:-1][m]) == [-20.0, -10.0, 0.0, 10.0]
    for share in (0.25, 0.50):
        p = sp.peak_bin_probs(2, 20.0, share)
        assert abs(p.sum() - 1.0) < 1e-12
        assert np.allclose(p[m], share / 4) and np.allclose(p[~m], (1 - share) / (nb - 4))
    # the bin-balanced scheme gives the peak set 4/40 = 0.10 and 4/44 = 0.0909
    assert abs(4 / nb - {50: 0.10, 60: 4 / 44}[q]) < 1e-15


@pytest.mark.parametrize("q", [50, 60])
@pytest.mark.parametrize("share", [0.25, 0.50])
def test_peak_focused_masses_and_within_bin_uniformity(q, share):
    sp = StartSampler(spec_for(q), 10.0)
    n = 1_000_000
    d = sp.peak_focused(2, n, np.random.default_rng(11), 20.0, share)
    edges = sp.bin_edges(2)
    m = sp.peak_set(2, 20.0)
    assert d.min() >= edges[0] and d.max() <= edges[-1]
    b = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, edges.size - 2)
    cnt = np.bincount(b, minlength=edges.size - 1)
    p = sp.peak_bin_probs(2, 20.0, share)
    # empirical peak share within 3 binomial SE of s
    ps = cnt[m].sum() / n
    assert abs(ps - share) <= 3 * np.sqrt(share * (1 - share) / n)
    # every bin of the peak set (and every bin outside it) receives its equal mass
    se = np.sqrt(n * p * (1 - p))
    assert np.all(np.abs(cnt - n * p) <= 4.5 * se)
    # uniform inside a bin (Kolmogorov-Smirnov on the position within three bins)
    for i in (np.flatnonzero(m)[1], np.flatnonzero(~m)[3], np.flatnonzero(~m)[-1]):
        u = (d[b == i] - edges[i]) / (edges[i + 1] - edges[i])
        assert kstest(u, "uniform").pvalue > 1e-3, i


def test_peak_focused_start_stream_use_and_desync():
    """One rng.random(n) for the bin and one for the position; ``balanced`` draws integers, so the streams differ."""
    sp = StartSampler(spec_for(50), 10.0)
    r1, r2, r3 = (np.random.default_rng(5) for _ in range(3))
    sp.peak_focused(2, 512, r1, 20.0, 0.5)
    r2.random(512)
    r2.random(512)
    assert r1.bit_generator.state == r2.bit_generator.state          # exactly two random(n) calls
    sp.balanced(2, 512, r3)
    assert r1.bit_generator.state != r3.bit_generator.state          # desynchronised from bin_balanced


def test_bin_balanced_is_the_original_balanced(tmp_path):
    """Default start_weights (absent, None or {scheme: bin_balanced}): Run.draw_starts is the original draw."""
    ref = make_run("phase_A", {}, str(tmp_path / "a"))
    want = ref.sampler.balanced(2, 512, ref.rngs["start"])
    for sw in (None, {"scheme": "bin_balanced"}):
        cfg = base_config()
        if sw is not None:
            cfg["start_weights"] = sw
        os.makedirs(tmp_path / f"b{sw}", exist_ok=True)
        r = Run(cfg, str(tmp_path / f"b{sw}"))
        got = r.draw_starts(2, 512)
        assert np.array_equal(got, want)
        assert r.rngs["start"].bit_generator.state == ref.rngs["start"].bit_generator.state


def test_run_peak_focused_draw_starts_uses_the_peak_sampler(tmp_path):
    cfg = base_config()
    cfg["start_weights"] = {"scheme": "peak_focused", "peak_half_width": 20, "peak_share": 0.5}
    os.makedirs(tmp_path / "p")
    r = Run(cfg, str(tmp_path / "p"))
    d = r.draw_starts(2, 200_000)
    assert abs(float(np.mean(np.abs(d) < 20.0)) - 0.5) < 0.01
    ref = StartSampler(r.spec, 10.0)
    assert np.array_equal(d[:50], ref.peak_focused(2, 200_000, np.random.default_rng(
        np.random.SeedSequence([10501, 50, r.P["rng_namespaces"]["starts_roles"]])), 20.0, 0.5)[:50])


# ====================================================================== 3. censored likelihood
def _grid(n=400, seed=0):
    rng = np.random.default_rng(seed)
    a = np.concatenate([np.geomspace(1e-4, 10.0, n), rng.uniform(1e-4, 10.0, n)])
    b = np.concatenate([rng.uniform(1.0, 1000.0, n), np.geomspace(1.0, 1000.0, n)])
    return a, b


def test_series_matches_scipy_betainc():
    a, b = _grid()
    for x in (1e-6,):
        la = log_betainc_small_x(x, torch.tensor(a), torch.tensor(b)).numpy()
        ref = betainc(a, b, x)
        assert np.all(ref > 0)
        assert np.max(np.abs(np.exp(la) / ref - 1.0)) <= 1e-6
    # float32 inputs (what the actor produces)
    a32, b32 = torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
    la = log_betainc_small_x(1e-6, a32, b32).numpy()
    assert np.max(np.abs(np.exp(la) / betainc(a32.double().numpy(), b32.double().numpy(), 1e-6) - 1.0)) <= 1e-6


def test_series_domain_is_refused():
    with pytest.raises(ValueError, match="series domain"):
        log_betainc_small_x(1e-6, torch.tensor([1.0]), torch.tensor([DOMAIN_X_B / 1e-6 * 1.01]))
    with pytest.raises(ValueError):
        log_betainc_small_x(0.0, torch.tensor([1.0]), torch.tensor([10.0]))


def test_series_gradient_matches_central_finite_differences():
    a = torch.tensor([1e-3, 0.05, 0.7, 3.0, 9.0], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([2.0, 30.0, 150.0, 400.0, 900.0], dtype=torch.float64, requires_grad=True)
    ga, gb = torch.autograd.grad(log_betainc_small_x(1e-6, a, b).sum(), (a, b))
    h = 1e-6
    for i in range(5):
        da = torch.zeros(5, dtype=torch.float64)
        da[i] = h
        fa = (log_betainc_small_x(1e-6, a.detach() + da, b.detach()) - log_betainc_small_x(1e-6, a.detach() - da, b.detach()))[i] / (2 * h)
        fb = (log_betainc_small_x(1e-6, a.detach(), b.detach() + da * 10) - log_betainc_small_x(1e-6, a.detach(), b.detach() - da * 10))[i] / (2 * h * 10)
        assert abs(float(ga[i]) - float(fa)) <= 1e-6 * max(1.0, abs(float(fa))), (i, float(ga[i]), float(fa))
        assert abs(float(gb[i]) - float(fb)) <= 1e-6 * max(1.0, abs(float(fb))), (i, float(gb[i]), float(fb))


def test_row_log_prob_sides_and_unflagged_rows():
    alpha = torch.tensor([0.002, 5.0, 300.0, 4.0], dtype=torch.float32)
    beta = torch.tensor([200.0, 0.001, 250.0, 120.0], dtype=torch.float32)
    c = 1e-6
    ac = torch.tensor([c, 1 - c, 0.5, 0.03], dtype=torch.float32)
    side = torch.tensor([-1, 1, 0, 0], dtype=torch.int8)
    lp = row_log_prob(alpha, beta, ac, side, c)
    dens = torch.distributions.Beta(alpha, beta).log_prob(ac)
    assert torch.equal(lp[2:], dens[2:])                                     # unflagged: exactly the density
    assert abs(float(lp[0]) - np.log(betainc(0.002, 200.0, c))) < 1e-4       # lower side: I_c(alpha, beta)
    assert abs(float(lp[1]) - np.log(betainc(0.001, 5.0, c))) < 1e-4         # upper side: I_c(beta, alpha)
    assert torch.equal(row_log_prob(alpha, beta, ac, None, c), dens)         # no side array: the density


def _agent(seed=0, lr=3e-4):
    agent = CurriculumPPOv2(PPOConfig(), torch.Generator().manual_seed(seed), np.random.default_rng(seed + 1))
    for g in agent.opt_actor.param_groups:
        g["lr"] = lr
    return agent


def _buffer(agent, spec, n=512, seed=3, stage=2):
    rng = np.random.default_rng(seed)
    d = rng.uniform(-spec.domain_half(stage), spec.domain_half(stage), n)
    st = spec.encode_obs(stage, d)
    a, b = agent.beta_params(st)
    act = agent.sample_actions(a, b, rng)
    return (st, act, agent.log_prob(a, b, act), rng.normal(size=n).astype(np.float32), rng.normal(size=n).astype(np.float32))


def _full(agent):
    return {"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict(),
            "opt_actor": agent.opt_actor.state_dict(), "opt_critic": agent.opt_critic.state_dict(),
            "rng_mb": agent.rng_mb.bit_generator.state}


def test_update_without_clamped_rows_is_bit_identical_under_censored():
    spec = spec_for(50)
    a1, a2, a3 = _agent(), _agent(), _agent()
    buf = _buffer(a1, spec)
    zeros = np.zeros(512, dtype=np.int8)
    d1 = a1.update(*buf)
    d2 = a2.update(*buf, clamp_side=zeros)
    d3 = a3.update(*buf, clamp_side=None)
    assert tree_equal(_full(a1), _full(a2)) and tree_equal(_full(a1), _full(a3))
    assert d1["policy_loss"] == d2["policy_loss"] == d3["policy_loss"] and d1["kl_epochs"] == d2["kl_epochs"]
    assert "n_censored_rows" not in d2


def test_stored_log_prob_and_exact_ratio_for_one_synthetic_clamped_row():
    spec = spec_for(50)
    agent = _agent()
    d = np.array([-170.0])                                    # one row; the raw draw is below c
    st = spec.encode_obs(2, d)
    a, b = agent.beta_params(st)
    c = agent.cfg.action_clamp
    side = np.array([-1], dtype=np.int8)
    act = np.array([c], dtype=np.float32)
    olp = agent.log_prob(a, b, act, clamp_side=side)
    assert abs(float(olp[0]) - np.log(betainc(float(a[0]), float(b[0]), c))) < 1e-3    # the censored log-mass (float32 value ~ -620)
    assert float(olp[0]) != float(agent.log_prob(a, b, act)[0])                          # not the density at c
    # the update's own log-prob at the pre-update parameters gives the ratio exactly 1
    t_st, t_ac, t_olp = (torch.as_tensor(x) for x in (st, act, olp))
    alpha, beta = agent.actor(t_st)
    lp = row_log_prob(alpha, beta, t_ac, torch.as_tensor(side), c)
    assert float(torch.exp(lp - t_olp)[0]) == 1.0
    loss, ratio, _ = masked_actor_loss(agent.actor, t_st, t_ac, t_olp, torch.ones(1), torch.arange(1), 0.2,
                                       torch.as_tensor(side), c)
    assert float(ratio[0]) == 1.0 and np.isfinite(float(loss))


def test_update_with_clamped_rows_runs_and_differs_from_density():
    spec = spec_for(50)
    a1, a2 = _agent(), _agent()
    st, act, olp, ret, adv = _buffer(a1, spec)
    side = np.zeros(512, dtype=np.int8)
    side[[3, 40]] = -1
    side[200] = 1
    act = act.copy()
    c = a1.cfg.action_clamp
    act[[3, 40]] = c
    act[200] = 1.0 - c
    alpha, beta = a1.beta_params(st)
    olp_c = a1.log_prob(alpha, beta, act, clamp_side=side)
    d_cens = a1.update(st, act, olp_c, ret, adv, clamp_side=side)
    d_dens = a2.update(st, act, a2.log_prob(alpha, beta, act), ret, adv)
    assert d_cens["n_censored_rows"] == 3 and np.isfinite(d_cens["policy_loss"]) and np.isfinite(d_cens["kl_final_epoch"])
    assert not tree_equal(a1.actor.state_dict(), a2.actor.state_dict())


def test_kl_diagnostics_of_flagged_rows_use_the_censored_log_mass():
    """At lr = 0 the policy cannot move, so every ratio is exactly 1 and every KL epoch is exactly 0 -- but only
    if the KL block uses the same (censored) log-prob as the stored one; the density at the clipped value of a
    clamped row differs from it by hundreds of nats."""
    spec = spec_for(50)
    agent = _agent(lr=0.0)
    st, act, _, ret, adv = _buffer(agent, spec)
    side = np.zeros(512, dtype=np.int8)
    side[[3, 40]] = -1
    side[200] = 1
    act = act.copy()
    c = agent.cfg.action_clamp
    act[[3, 40]] = c
    act[200] = 1.0 - c
    alpha, beta = agent.beta_params(st)
    olp = agent.log_prob(alpha, beta, act, clamp_side=side)
    d = agent.update(st, act, olp, ret, adv, clamp_side=side)
    assert d["n_censored_rows"] == 3 and d["kl_epochs"] == [0.0] * len(d["kl_epochs"])
    assert d["clip_frac"] == 0.0           # the actor loss's ratio of the flagged rows is exactly 1, too


class _ClampingRng:
    """Learner stream wrapper: the first ``k`` Beta draws of every call are forced below / above the clamp."""

    def __init__(self, rng, k_lo=2, k_hi=1):
        self.rng, self.k_lo, self.k_hi = rng, k_lo, k_hi

    @property
    def bit_generator(self):      # the runner logs and checkpoints the stream positions
        return self.rng.bit_generator

    def beta(self, a, b):
        x = self.rng.beta(a, b)
        x[: self.k_lo] = 1e-9
        if self.k_hi:
            x[self.k_lo: self.k_lo + self.k_hi] = 1.0 - 1e-9
        return x


@pytest.mark.parametrize("mode", ["density", "censored"])
def test_rollout_flags_and_stored_log_prob(mode, tmp_path):
    r = make_run("phase_A", {"reward_mode": "expected"}, str(tmp_path / mode))
    spec, agent, n = r.spec, r.agent, 64
    t0 = np.full(n, spec.T)
    d0 = r.sampler.balanced(spec.T, n, r.rngs["start"])
    roles = r.rngs["start"].integers(0, 2, size=n)
    b = collect_batch_v2(spec, agent, t0, d0, roles, r.rngs["env"], _ClampingRng(r.rngs["learn"]), r.rngs["opp"],
                         1.0, 1.0, 10.0, reward_mode="expected", clamp_likelihood=mode)
    side = b["clamp_side"]
    assert side.dtype == np.int8 and side.shape == b["actions"].shape
    assert (side == -1).sum() == 2 and (side == 1).sum() == 1 and (side == 0).sum() == n - 3
    a, bb = b["d1_buf"]["alpha"], b["d1_buf"]["beta"]
    c = agent.cfg.action_clamp
    dens = agent.log_prob(a, bb, b["actions"])
    cens = agent.log_prob(a, bb, b["actions"], clamp_side=side)
    flagged = side != 0
    assert np.array_equal(b["logp"][~flagged], dens[~flagged])
    want = cens if mode == "censored" else dens
    assert np.array_equal(b["logp"], want.astype(np.float32))
    if mode == "censored":
        assert np.all(b["logp"][flagged] != dens[flagged])
        i = int(np.flatnonzero(side == -1)[0])
        assert abs(float(b["logp"][i]) - np.log(betainc(float(a[i]), float(bb[i]), c))) < 1e-3


def test_mean_mode_rows_are_never_flagged(tmp_path):
    """Phase B stage-T rows carry the frozen Beta mean (the draw is discarded): not censored."""
    r = make_run("phase_A", {"reward_mode": "expected"}, str(tmp_path / "a"))
    spec, agent, n = r.spec, r.agent, 32
    frozen = copy.deepcopy(agent.actor)
    t0 = np.ones(n, dtype=int)
    b = collect_batch_v2(spec, agent, t0, np.zeros(n), np.zeros(n, dtype=int), r.rngs["env"], _ClampingRng(r.rngs["learn"], 3, 0),
                         r.rngs["opp"], 1.0, 1.0, 10.0, reward_mode="expected", frozen=frozen,
                         continuation_action_mode="mean", clamp_likelihood="censored")
    stage = b["stage"]
    assert np.all(b["clamp_side"][stage == spec.T] == 0)
    assert np.all(b["clamp_side"][stage == 1][:3] == -1)             # stage-1 rows are sampled draws


def test_censored_run_end_to_end(tmp_path):
    """A reduced phase-A run with forced clamped draws completes and records the censored counts."""
    cfg = base_config()
    cfg["clamp_likelihood"] = "censored"
    cfg["flags"]["reward_mode"] = "expected"
    d = str(tmp_path / "c")
    os.makedirs(d)
    run = Run(cfg, d)
    run.rngs["learn"] = _ClampingRng(run.rngs["learn"], 2, 1)
    assert execute(run, cfg, d, "pytest") == 0
    rows = run.v2_history
    assert all(r["n_clamped_rows_learner"] == 3 and r["n_censored_rows"] == 3 for r in rows)
    man = json.load(open(os.path.join(d, "manifest.json")))
    assert man["clamp_likelihood"] == "censored" and man["start_weights"] is None
    assert man["pathwise_epochs"] == 1 and man["pathwise_minibatch"] is None


# ====================================================================== 4. phase P with E epochs x M minibatches
def _agent_with_adam(seed=0):
    spec = spec_for(50)
    a = _agent(seed, lr=3e-5)
    a.update(*_buffer(a, spec, seed=seed + 5))
    a.refresh_snapshot()
    for g in a.opt_actor.param_groups:
        g["lr"] = 3e-5
    return spec, a


def _adam_steps(agent) -> float:
    return float(next(iter(agent.opt_actor.state.values()))["step"])


def test_pathwise_update_makes_twenty_steps_and_aligns_the_minibatch_stream():
    spec, a = _agent_with_adam()
    _, b = _agent_with_adam()                          # identical twin: same weights, same minibatch stream position
    assert a.rng_mb.bit_generator.state == b.rng_mb.bit_generator.state
    s0 = _adam_steps(a)
    out = pathwise_update(a, spec, np.random.default_rng(1).uniform(-200, 200, 512), 2, epochs=10, minibatch=256)
    assert out["n_steps"] == 20 and _adam_steps(a) - s0 == 20
    # the PPO control: one update on 512 rows draws one permutation(512) per epoch (10); the pathwise update
    # draws the same permutations, so the two streams are at the same position afterwards
    b.update(*_buffer(b, spec, n=512, seed=9))
    assert a.rng_mb.bit_generator.state == b.rng_mb.bit_generator.state
    assert set(out) >= {"loss", "grad_norm_pre_clip", "grad_norm_pre_clip_max", "foc_abs_mean", "foc_abs_max", "e0", "n_steps"}
    assert out["grad_norm_pre_clip_max"] >= out["grad_norm_pre_clip"] and np.isfinite(out["loss"])


def test_pathwise_update_minibatches_are_a_fresh_permutation_in_every_epoch(monkeypatch):
    import agents.ppo_pathwise as pw
    spec, a = _agent_with_adam()
    _, twin = _agent_with_adam()                       # same minibatch stream position
    d = np.random.default_rng(4).uniform(-200, 200, 512)
    seen = []

    def record(agent, spec_, rows, stage, max_grad_norm=0.5):
        seen.append(np.array(rows))
        return {"loss": 0.0, "grad_norm_pre_clip": 0.0}

    monkeypatch.setattr(pw, "pathwise_step", record)
    pathwise_update(a, spec, d, 2, epochs=3, minibatch=256)
    got = [np.concatenate(seen[2 * i: 2 * i + 2]) for i in range(3)]
    for g in got:                                      # each epoch: the rows in the order of the stream's permutation
        assert np.array_equal(g, d[twin.rng_mb.permutation(512)])
    assert not np.array_equal(got[0], got[1]) and len(seen) == 6


def test_pathwise_update_leaves_head_critic_and_adam_of_the_critic_untouched():
    spec, a = _agent_with_adam()
    head = (a.actor.out.weight[1].detach().clone(), a.actor.out.bias[1].detach().clone())
    critic = {k: v.clone() for k, v in a.critic.state_dict().items()}
    opt_c = copy.deepcopy(a.opt_critic.state_dict())
    w0 = a.actor.l1.weight.detach().clone()
    pathwise_update(a, spec, np.random.default_rng(2).uniform(-200, 200, 512), 2, epochs=3, minibatch=128)
    assert torch.equal(a.actor.out.weight[1], head[0]) and torch.equal(a.actor.out.bias[1], head[1])
    assert tree_equal(critic, a.critic.state_dict()) and tree_equal(opt_c["state"], a.opt_critic.state_dict()["state"])
    assert not torch.equal(a.actor.l1.weight, w0)                          # the actor did move


def test_pathwise_update_rejects_bad_budget():
    spec, a = _agent_with_adam()
    with pytest.raises(ValueError):
        pathwise_update(a, spec, np.zeros(8), 2, epochs=0, minibatch=4)


def _p_config(epochs=None, minibatch=None, cap=3):
    """In-process phase_P config on the reduced base config (parent restored by the caller)."""
    cfg = base_config()
    cfg["mode"] = "phase_P"
    cfg["flags"]["reward_mode"] = "expected"
    cfg["budget_overrides"]["phase_caps"]["P"] = cap
    cfg["budget_overrides"]["verifier_timeout"] = {"A": 5, "B": 5, "C": 5, "P": 2}
    cfg["lr_decay"] = [{"phase": "P", "start_lr": 3e-5, "end_lr": 3e-5, "local_first": 1, "local_last": cap}]
    if epochs is not None:
        cfg["pathwise_epochs"] = epochs
    if minibatch is not None:
        cfg["pathwise_minibatch"] = minibatch
    return cfg


@pytest.fixture(scope="module")
def p_parent(tmp_path_factory) -> str:
    d = str(tmp_path_factory.mktemp("pparent"))
    cfg = base_config()
    cfg["flags"]["reward_mode"] = "expected"
    assert execute(Run(cfg, d), cfg, d, "pytest") == 0
    return os.path.join(d, "state_end_A.pt")


def _run_p(cfg, parent, out) -> Run:
    cfg["parent_checkpoint"], cfg["parent_sha256"] = parent, "in-process"
    os.makedirs(out, exist_ok=True)
    run = Run(cfg, out)
    assert execute(run, cfg, out, "pytest") == 0
    return run


def test_phase_p_epochs_end_to_end(p_parent, tmp_path):
    run = _run_p(_p_config(10, 256), p_parent, str(tmp_path / "p"))
    assert all(h["n_steps"] == 20 for h in run.history) and run.phase_checks == {
        "concentration_head_bit_identical": True, "critic_bit_identical": True, "critic_adam_state_bit_identical": True}
    cap = len(run.history)
    assert all(h["n_minibatch_steps"] == 20 for h in run.history)       # the recorded step count is the real one
    assert run.curriculum_log[-1]["minibatch_steps"] == 20 * cap
    summary = json.load(open(tmp_path / "p" / "v2_run_summary.json"))
    assert summary["costs"]["minibatch_steps"] == 20 * cap
    man = json.load(open(tmp_path / "p" / "manifest.json"))
    assert man["pathwise_epochs"] == 10 and man["pathwise_minibatch"] == 256


def test_phase_p_defaults_are_the_r1_single_step(p_parent, tmp_path):
    """E = 1 and no minibatch: one pathwise_step per update, no use of the minibatch stream."""
    run = _run_p(_p_config(), p_parent, str(tmp_path / "p"))
    assert all("n_steps" not in h and h["n_minibatch_steps"] == 1 for h in run.history)
    assert run.curriculum_log[-1]["minibatch_steps"] == len(run.history)          # one step per update, as in R1
    parent_state = torch.load(p_parent, weights_only=False)
    assert tree_equal(run.agent.rng_mb.bit_generator.state, parent_state["agent"]["rng_minibatch"])


def test_phase_p_default_reproduces_r1_a_detmean_first_25_updates(tmp_path):
    """From the real development-seed parent: the first 25 per-update rows of R1's A_detmean, bit for bit."""
    parent = CANON / "results/v2_T2_locked/rehearsal_v1_1/q50/seed10501/state_end_A.pt"
    ref_csv = R1_ROOT / "stage2/q50/seed10501/A_detmean/v2_updates.csv"
    if not parent.exists() or not ref_csv.exists():
        pytest.skip("parent state or the R1 A_detmean log is not on this machine")
    import csv
    ref = list(csv.DictReader(open(ref_csv)))[:25]
    job = L.build_stage2("A_detmean", 50, 10501, tmp_path, True)
    cfg = copy.deepcopy(job.cfg)
    cfg["budget_overrides"]["phase_caps"]["P"] = 25
    cfg["lr_decay"] = [{"phase": "P", "start_lr": 3e-5, "end_lr": 3e-5, "local_first": 1, "local_last": 25}]
    cfg.update(NEW_KEYS_AT_DEFAULT)
    d = str(tmp_path / "run")
    os.makedirs(d)
    run = Run(cfg, d)
    assert execute(run, cfg, d, "pytest") == 0
    for k in ("loss", "grad_norm_pre_clip", "foc_abs_mean", "foc_abs_max", "e0"):
        assert [float(r[k]) for r in ref] == [float(r[k]) for r in run.v2_history], k


# ====================================================================== 5. config refusals and manifest keys
@pytest.mark.parametrize("key, value, mode, msg", [
    ("start_weights", {"scheme": "nope"}, "phase_A", "scheme"),
    ("start_weights", {"scheme": "peak_focused", "peak_half_width": 20}, "phase_A", "exactly"),
    ("start_weights", {"scheme": "peak_focused", "peak_half_width": 20, "peak_share": 1.0}, "phase_A", "share"),
    ("start_weights", {"scheme": "peak_focused", "peak_half_width": 20, "peak_share": 0.0}, "phase_A", "share"),
    ("start_weights", {"scheme": "peak_focused", "peak_half_width": -1, "peak_share": 0.5}, "phase_A", "half_width"),
    ("start_weights", {"scheme": "peak_focused", "peak_half_width": 20, "peak_share": 0.5}, "phase_B", "defined for modes"),
    ("start_weights", {"scheme": "bin_balanced", "peak_share": 0.5}, "phase_A", "no other key"),
    ("clamp_likelihood", "bogus", "phase_A", "clamp_likelihood"),
    ("clamp_likelihood", "censored", "phase_P", "defined for modes"),
    ("pathwise_epochs", 0, "phase_P", "positive int"),
    ("pathwise_epochs", 10, "phase_A", "phase_P only"),
    ("pathwise_minibatch", 256, "phase_A", "phase_P only"),
    ("pathwise_minibatch", 0, "phase_P", "positive int"),
])
def test_config_refusals(key, value, mode, msg):
    cfg = base_config()
    cfg["mode"] = mode
    cfg[key] = value
    if mode in ("phase_B", "phase_P"):
        cfg["parent_checkpoint"], cfg["parent_sha256"] = "x", "y"
    if mode == "phase_B":
        cfg["flags"].update(stage2_update_mode="frozen", adv_norm_scope="stage1_rows", continuation_action_mode="mean")
    if mode == "phase_P":
        cfg["lr_decay"] = [{"phase": "P", "start_lr": 3e-5, "end_lr": 3e-5, "local_first": 1, "local_last": 6}]
    with pytest.raises(ConfigError, match=msg):
        validate_config(cfg)


def test_mode_full_refuses_the_new_non_default_keys():
    cfg = json.load(open(C7_CONFIG))
    for k, v in (("start_weights", {"scheme": "peak_focused", "peak_half_width": 20, "peak_share": 0.5}),
                 ("clamp_likelihood", "censored")):
        c = copy.deepcopy(cfg)
        c[k] = v
        with pytest.raises(ConfigError):
            validate_config(c)
    validate_config({**cfg, **NEW_KEYS_AT_DEFAULT})


def test_every_peak_set_must_be_a_proper_subset(tmp_path):
    cfg = base_config()
    cfg["start_weights"] = {"scheme": "peak_focused", "peak_half_width": 1000, "peak_share": 0.5}   # every bin
    os.makedirs(tmp_path / "x")
    with pytest.raises(ConfigError, match="start_weights"):
        Run(cfg, str(tmp_path / "x"))


# ====================================================================== 6. the launcher arms
@pytest.fixture(scope="module")
def planned(tmp_path_factory):
    root = tmp_path_factory.mktemp("r2b_root")
    return root, {w: L.build_configs(w, (50, 60), L.DEFAULT_SEEDS, None, root, require_parents=False) for w in L.R2B_WAVES}


def test_r2b_arm_tables_and_job_counts(planned):
    _, jobs = planned
    assert list(L.R2B_WAVEA_ARMS) == ["A_peak25", "A_peak50", "A_censored"]
    assert list(L.R2B_WAVEP_ARMS) == ["P20_lr3e-5", "P20_lr3e-4", "A_ctrl200_lr3e-4"]
    for w in L.R2B_WAVES:
        assert len(jobs[w]) == 60 and len({j.out_dir for j in jobs[w]}) == 60
    assert {Path(j.out_dir).parts[-4] for j in jobs["r2b_waveA"]} == {"waveA"}
    assert {Path(j.out_dir).parts[-4] for j in jobs["r2b_waveP"]} == {"waveP"}


@pytest.mark.parametrize("wave", L.R2B_WAVES)
def test_r2b_arms_differ_from_their_comparator_in_exactly_the_preregistered_keys(planned, wave):
    root, jobs = planned
    refs = {(q, s): L.r2b_comparator_config(wave, q, s, root) for q in (50, 60) for s in L.DEFAULT_SEEDS}
    for j in jobs[wave]:
        c = j.cfg
        ref_arm, keys = L.R2B_EXPECTED_DIFFS[wave][c["arm"]]
        assert set(L.r2b_arm_diff(c, refs[(c["q"], c["seed"])][ref_arm])) == keys, (c["arm"], c["q"], c["seed"])


def test_r2b_arm_values_and_explicit_defaults(planned):
    _, jobs = planned
    cfgs = {(j.cfg["arm"], j.cfg["q"], j.cfg["seed"]): j.cfg for w in jobs for j in jobs[w]}
    a = cfgs[("A_peak25", 50, 10501)]
    assert a["mode"] == "phase_A" and a["start_weights"] == {"scheme": "peak_focused", "peak_half_width": 20, "peak_share": 0.25}
    assert a["full_state_at"] == [1200] and a["clamp_likelihood"] == "density" and a["pathwise_epochs"] == 1
    assert cfgs[("A_peak50", 60, 10510)]["start_weights"]["peak_share"] == 0.50
    assert cfgs[("A_censored", 50, 10501)]["clamp_likelihood"] == "censored" and cfgs[("A_censored", 50, 10501)]["start_weights"] is None
    p = cfgs[("P20_lr3e-4", 60, 10503)]
    assert p["mode"] == "phase_P" and p["pathwise_epochs"] == 10 and p["pathwise_minibatch"] == 256
    assert p["lr_decay"] == [{"phase": "P", "start_lr": 3e-4, "end_lr": 3e-4, "local_first": 1, "local_last": 200}]
    assert p["parent_checkpoint"] == str(L.REHEARSAL / "q60" / "seed10503" / "state_end_A.pt")
    ctrl = cfgs[("A_ctrl200_lr3e-4", 50, 10501)]
    assert ctrl["mode"] == "phase_A_continue" and ctrl["budget_overrides"]["phase_caps"]["A"] == 200
    assert ctrl["lr_decay"] == [{"phase": "A", "start_lr": 3e-4, "end_lr": 3e-4, "local_first": 1, "local_last": 200}]
    assert ctrl["pathwise_epochs"] == 1 and ctrl["pathwise_minibatch"] is None


def test_r2b_configs_validate(planned):
    _, jobs = planned
    for w in jobs:
        rows = L.validate_jobs(jobs[w])
        assert [r for r in rows if not r["ok"]] == []
        assert all(not r["window_problems"] for r in rows)


def test_r2b_waves_do_not_change_the_r1_launcher_tables():
    assert L.WAVES == ("v11_repro", "parents_A", "stage1", "stage2")
    assert set(L.WAVE_DIR) == set(L.WAVES) | set(L.R2B_WAVES) | {L.R2B_REPRO}


def test_c_r2_wave_is_the_unchanged_locked_entry_point(tmp_path):
    jobs = L.build_configs(L.R2B_REPRO, (50, 60), L.DEFAULT_SEEDS, None, tmp_path)
    assert len(jobs) == 20 and len({j.out_dir for j in jobs}) == 20
    assert all(L.is_locked_entry(j.cfg) and j.cfg["wave"] == L.R2B_REPRO for j in jobs)
    assert Path(jobs[0].out_dir).relative_to(tmp_path).parts == ("v20_reproduction", "q50", "seed10501")
    cmd = L.job_command(*jobs[0])
    assert cmd[3].endswith("run/run_v2_T2_locked.py") and "--config" not in cmd
    assert cmd[cmd.index("--q") + 1] == "50" and cmd[cmd.index("--seed") + 1] == "10501"
    with pytest.raises(ValueError, match="no arms"):
        L.build_configs(L.R2B_REPRO, (50,), (10501,), ["x"], tmp_path)


# ====================================================================== 7. the post-launch checks (section 4.3)
def _fake_launched_waves(tmp_path):
    """A scratch results root with one (q, seed) of every arm, manifests as the runs would write them, and the
    R1 comparator configs as on-disk files (R1 configs do not have the four R2b keys)."""
    import r2b_launch_checks as K

    root, r1 = tmp_path / "r2b", tmp_path / "r1"
    for wave in L.R2B_WAVES:
        for ref_arm, cfg in L.r2b_comparator_config(wave, 50, 10501, root).items():
            ref_dir = (r1 / "parents_A" / "q50" / "seed10501" if wave == "r2b_waveA"
                       else r1 / "stage2" / "q50" / "seed10501" / ref_arm)
            os.makedirs(ref_dir, exist_ok=True)
            json.dump({k: v for k, v in cfg.items() if k not in L.R2B_DEFAULTS}, open(ref_dir / "run_config.json", "w"))
        for job in L.build_configs(wave, (50,), (10501,), None, root, require_parents=False):
            c = job.cfg
            os.makedirs(job.out_dir, exist_ok=True)
            man = {"git": {"commit": "HEAD", "dirty": False}, "input_config": c, "parent_sha256": "abc",
                   "parent_checkpoint": c.get("parent_checkpoint"), **{k: c[k] for k in L.R2B_DEFAULTS}}
            json.dump(man, open(Path(job.out_dir) / "manifest.json", "w"))
    parents = {(50, 10501): {"waveP_parent_sha256": "abc",
                             "waveP_parent_path": str(L.REHEARSAL / "q50" / "seed10501" / "state_end_A.pt")}}
    return K, root, r1, parents


def test_launch_checks_pass_on_matching_runs_and_fail_on_each_violation(tmp_path, monkeypatch):
    K, root, r1, parents = _fake_launched_waves(tmp_path)
    monkeypatch.setattr(K, "sha256_file", lambda p: "abc")
    cache = {}

    def run(wave, arm, seed=10501):
        return K.check_run(wave, arm, 50, seed, root, r1, parents, "HEAD", cache)

    arms = [("r2b_waveA", a) for a in L.R2B_WAVEA_ARMS] + [("r2b_waveP", a) for a in L.R2B_WAVEP_ARMS]
    assert all(run(w, a)["ok"] for w, a in arms), [run(w, a)["problems"] for w, a in arms]

    def edit_manifest(wave, arm, fn):
        p = root / ("waveA" if wave == "r2b_waveA" else "waveP") / "q50" / "seed10501" / arm / "manifest.json"
        m = json.load(open(p))
        fn(m)
        json.dump(m, open(p, "w"))

    edit_manifest("r2b_waveA", "A_peak25", lambda m: m["git"].update(dirty=True))
    assert any("dirty" in s for s in run("r2b_waveA", "A_peak25")["problems"])
    edit_manifest("r2b_waveA", "A_peak50", lambda m: m["input_config"]["ppo_overrides"].update(minibatch=64))
    assert any("config differs" in s for s in run("r2b_waveA", "A_peak50")["problems"])
    edit_manifest("r2b_waveA", "A_censored", lambda m: m.update(clamp_likelihood="density"))
    assert any("clamp_likelihood" in s for s in run("r2b_waveA", "A_censored")["problems"])
    edit_manifest("r2b_waveP", "P20_lr3e-5", lambda m: m.update(parent_sha256="zzz"))
    assert any("parent_sha256" in s for s in run("r2b_waveP", "P20_lr3e-5")["problems"])
    edit_manifest("r2b_waveP", "P20_lr3e-4", lambda m: m["input_config"]["lr_decay"][0].update(start_lr=3e-5, end_lr=3e-5))
    assert any("config differs" in s for s in run("r2b_waveP", "P20_lr3e-4")["problems"])
    assert run("r2b_waveP", "A_ctrl200_lr3e-4", seed=10502)["ok"] is False          # no manifest: failed, not skipped
