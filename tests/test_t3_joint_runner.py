"""T3-R1: tests of ``run/run_joint.py`` (the joint whole-game runner) and of the ``bw`` path at T = 3.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_t3_joint_runner.py -p no:cacheprovider -q

Sections: config refusals and range checks; the init digest equals MSRun's (bw config at T = 3, MS_base at T = 2) and
the reimplemented digest equals ``MSRun._init_digest``; reduced-budget runs at T = 2 and T = 3 with the root-only mix
and the exploring-starts mix (policy rows per stage, learning rate, refreshes, files, global RNG); determinism; the
return identity of a root-only batch; the closed form does not enter; ``bw`` at T = 3 (production values, a reduced
legacy run with the rule disabled).
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import t3_helpers as H
import t3_configs as tc  # noqa: E402
from run import run_joint as rj  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402
from run.run_final_dp_br_round3_dense import ConfigError  # noqa: E402
from run.v2_rollout import collect_batch_v2  # noqa: E402

PY = sys.executable
SPECS = H.SPECS
U, CHECK = H.REDUCED_U, H.REDUCED_CHECK


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def runs():
    """The four reduced runs (U = 30, 48 episodes, checks and exports every 10 updates, one decay window)."""
    return H.reduced_runs()


# ====================================================================== 1. configuration
def _mut(path, value):
    def f(cfg):
        node = cfg
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
    return f


def _drop(path):
    def f(cfg):
        node = cfg
        for k in path[:-1]:
            node = node[k]
        del node[path[-1]]
    return f


REFUSALS = {
    "unknown top-level key": lambda c: c.update(extra=1),
    "missing key": _drop(["round"]),
    "wrong schema": _mut(["schema"], "joint_run_config/2"),
    "wrong round": _mut(["round"], "ms_r1"),
    "unknown arm": _mut(["arm"], "bw"),
    "seed differs from the record": _mut(["seed"], 1),
    "unknown joint key": lambda c: c["joint"].update(extra=1),
    "missing joint key": _drop(["joint", "check_every"]),
    "T = 4": _mut(["joint", "T"], 4),
    "T differs from game.T": _mut(["joint", "T"], 2),
    "updates 0": _mut(["joint", "updates"], 0),
    "updates bool": _mut(["joint", "updates"], True),
    "check_every 0": _mut(["joint", "check_every"], 0),
    "starts sum differs": _mut(["joint", "starts", "root"], 40),
    "negative start": lambda c: c["joint"]["starts"].update(root=60, stage2=-12),
    "stage-3 key missing at T = 3": _drop(["joint", "starts", "stage3"]),
    "window first 0": _mut(["joint", "lr_windows"], [{"first": 0, "last": 30, "start": 3e-4, "end": 3e-5}]),
    "window first >= last": _mut(["joint", "lr_windows"], [{"first": 30, "last": 30, "start": 3e-4, "end": 3e-5}]),
    "windows not contiguous": _mut(["joint", "lr_windows"], [{"first": 1, "last": 10, "start": 3e-4, "end": 2e-4},
                                                             {"first": 12, "last": 30, "start": 2e-4, "end": 3e-5}]),
    "last window short of U": _mut(["joint", "lr_windows"], [{"first": 16, "last": 29, "start": 3e-4, "end": 3e-5}]),
    "window with an extra key": _mut(["joint", "lr_windows"],
                                     [{"first": 16, "last": 30, "start": 3e-4, "end": 3e-5, "x": 1}]),
    "sampled reward": _mut(["reward_mode"], "sampled"),
    "censored likelihood": _mut(["clamp_likelihood"], "censored"),
    "init_digest false": _mut(["init_digest"], False),
    "gamma 0.99": _mut(["record", "ppo", "gamma"], 0.99),
    "episodes_per_update 0": _mut(["record", "protocol", "episodes_per_update"], 0),
}


@pytest.mark.parametrize("name", list(REFUSALS))
def test_config_refusals(tmp_path, name):
    base = H.joint_cfg(str(tmp_path / "x"), "jt", 50, 50500, U=30)
    rj.validate_config(base)                       # the base config is accepted
    cfg = copy.deepcopy(base)
    REFUSALS[name](cfg)
    with pytest.raises(ConfigError):
        rj.validate_config(cfg)


def test_stage3_key_is_refused_at_t2(tmp_path):
    cfg = H.joint_cfg(str(tmp_path / "x"), "jt2", 50, 50500)
    rj.validate_config(cfg)
    cfg["joint"]["starts"]["stage3"] = 0
    with pytest.raises(ConfigError):
        rj.validate_config(cfg)


def test_production_ranges_are_accepted_and_small_budgets_too(tmp_path):
    for arm in ("jt", "jt_es", "jt2"):
        rj.validate_config(tc.build_joint_config(H.PROTO, 60, 50501, arm, str(tmp_path)))
    cfg = H.joint_cfg(str(tmp_path), "jt_es", starts=H.ES_MIX_T3, U=1, check_every=1, window=False)
    rj.validate_config(cfg)


def test_cli_exit_codes(tmp_path):
    """4 for a config error, 3 for a directory that already holds a status.json, 0 for a complete run."""
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    bad = H.joint_cfg(str(tmp_path / "bad"), "jt", U=4, check_every=2)
    bad["extra"] = 1
    p_bad = H.write_cfg(bad, str(tmp_path / "bad"))
    r = subprocess.run([PY, "-B", str(H.ROOT / "run" / "run_joint.py"), "--config", p_bad,
                        "--out-dir", str(tmp_path / "bad")], env=env, capture_output=True, text=True)
    assert r.returncode == 4 and "[config-error]" in r.stdout
    good = H.joint_cfg(str(tmp_path / "good"), "jt2", U=4, check_every=2, window=False)
    p_good = H.write_cfg(good, str(tmp_path / "good"))
    r = subprocess.run([PY, "-B", str(H.ROOT / "run" / "run_joint.py"), "--config", p_good,
                        "--out-dir", str(tmp_path / "good")], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    r = subprocess.run([PY, "-B", str(H.ROOT / "run" / "run_joint.py"), "--config", p_good,
                        "--out-dir", str(tmp_path / "good")], env=env, capture_output=True, text=True)
    assert r.returncode == 3 and "[refuse]" in r.stdout
    man = json.load(open(tmp_path / "good" / "manifest.json"))
    assert man["config_file_sha256"] and man["init_state_sha256"] and man["commit"]


# ====================================================================== 2. init digest
@pytest.mark.parametrize("q,seed", [(50, 50501), (60, 50502)])
def test_init_digest_equals_msrun_at_t3_and_across_arms(tmp_path, q, seed):
    """Initial weights depend on (q, seed) only: bw (MSRun), jt and jt_es have the same digest, and the
    reimplemented digest equals ``MSRun._init_digest`` on the same agent."""
    d_b = str(tmp_path / "bw")
    os.makedirs(d_b)
    ms = rms.MSRun(H.bw_cfg(d_b, q, seed), d_b)
    assert ms.init_state_sha256 and len(ms.init_state_sha256) == 64
    for arm in ("jt", "jt_es"):
        d = str(tmp_path / arm)
        os.makedirs(d)
        jr = rj.JointRun(H.joint_cfg(d, arm, q, seed, starts=H.ES_MIX_T3 if arm == "jt_es" else None), d)
        assert jr.init_state_sha256 == ms.init_state_sha256
    assert rj.JointRun._init_digest(ms.agent) == rms.MSRun._init_digest(ms.agent) == ms.init_state_sha256


@pytest.mark.parametrize("q,seed", [(50, 30501), (60, 30520)])
def test_init_digest_of_jt2_equals_msrun_at_t2(tmp_path, q, seed):
    params = json.load(open(H.MS_PARAMS_PATH))
    d_m, d_j = str(tmp_path / "ms"), str(tmp_path / "jt2")
    os.makedirs(d_m)
    os.makedirs(d_j)
    cfg = H.mc.build_config(H.PROTO, q, seed, "MS_base", d_m, params)
    cfg["init_digest"] = True
    ms = rms.MSRun(cfg, d_m)
    jr = rj.JointRun(H.joint_cfg(d_j, "jt2", q, seed), d_j)
    assert jr.init_state_sha256 == ms.init_state_sha256


def test_lr_expression_equals_msrun_lr_linear(tmp_path):
    d = str(tmp_path / "x")
    os.makedirs(d)
    ms = rms.MSRun(H.bw_cfg(d), d)
    jr = rj.JointRun(H.joint_cfg(d, "jt", U=30), d)
    for u in range(1, 31):
        want = ms.lr_linear(3e-4, 3e-5, 16, 30, u) if u >= 16 else 3e-4
        assert jr.lr_for(u) == want, u


# ====================================================================== 3. reduced runs
@pytest.mark.parametrize("name", list(SPECS))
def test_policy_rows_lr_refreshes_files_and_global_rng(runs, name):
    d, cfg = runs[name]
    arm, T, starts = SPECS[name]
    rows = read_csv(os.path.join(d, "joint_updates.csv"))
    assert len(rows) == U
    per_update = rj.policy_rows_per_update(cfg["joint"]["starts"], T)
    n_rows = sum(per_update.values())
    ms_dir = os.path.join(d, "ref_ms")
    os.makedirs(ms_dir, exist_ok=True)
    ms = rms.MSRun(H.bw_cfg(ms_dir), ms_dir)
    for r in rows:
        u = int(r["update"])
        assert [int(r[f"rows_t{t}"]) for t in range(1, T + 1)] == [per_update[t] for t in range(1, T + 1)], u
        assert int(r["n_transitions"]) == n_rows and int(r["n_episodes"]) == 48
        assert int(r["n_minibatch_steps"]) == 10 * math.ceil(n_rows / 256), u
        want = ms.lr_linear(3e-4, 3e-5, U // 2 + 1, U, u) if u > U // 2 else 3e-4
        assert float(r["lr"]) == want and float(r["critic_lr"]) == want, u      # equal to MSRun.lr_linear
        assert (r["snapshot_refreshed"] == "True") == (u % 20 == 0), u
    st = torch.load(os.path.join(d, "state_end.pt"), weights_only=False)
    assert st["format"] == "joint_full_state/1"
    log = st["snapshot_log"]
    assert log[0] == {"update": 0, "reason": "init"} and log[1] == {"update": 0, "reason": "entry"}
    assert [e["update"] for e in log[2:]] == [20] and st["counters"]["snapshot_refreshes"] == 3     # init, entry, update 20
    assert st["counters"]["rows_by_stage"] == {t: per_update[t] * U for t in range(1, T + 1)}
    for u in (10, 20, 30):
        assert os.path.exists(os.path.join(d, "weights", f"u{u:05d}.npz"))
    assert not os.path.exists(os.path.join(d, "weights", "u00025.npz"))
    checks = read_csv(os.path.join(d, "joint_checks.csv"))
    assert [int(c["update"]) for c in checks] == [10, 20, 30]
    assert [int(checks[0][f"rows_t{t}"]) for t in range(1, T + 1)] == [per_update[t] * 10 for t in range(1, T + 1)]
    assert all(c["valid"] == "True" and c["error"] == "" for c in checks)
    assert all(float(checks[0][f"conc_max_std_norm_t{t}"]) > 0 for t in range(1, T + 1))
    for f in ("final_eval_final.npz", "final_eval_development.npz", "visitation.npz", "state_end.pt", "manifest.json",
              "status.json", "joint_run_summary.json", "run_config.json"):
        assert os.path.exists(os.path.join(d, f)), f
    summ = json.load(open(os.path.join(d, "joint_run_summary.json")))
    assert summ["global_rng"] == {"status": "ok", "violations": []}
    assert summ["updates"] == U and summ["episodes"] == 48 * U and summ["transitions"] == n_rows * U
    assert summ["minibatch_steps"] == U * 10 * math.ceil(n_rows / 256)
    assert summ["policy_rows_by_stage"] == {str(t): per_update[t] * U for t in range(1, T + 1)}
    assert set(summ["costs_wall_sec"]) == {"train_rollout", "train_update", "dev_verifier", "final_eval"}
    assert summ["peak_rss_kb"] > 0 and len(summ["init_state_sha256"]) == 64
    assert json.load(open(os.path.join(d, "status.json")))["exit_code"] == 0
    man = json.load(open(os.path.join(d, "manifest.json")))
    assert set(man["global_rng"]) >= {"seeding", "reference_before_first_update", "end_of_training",
                                      "after_final_eval", "end_of_run"}


@pytest.mark.parametrize("name", ["t3_es", "t2_es"])
def test_visitation_counts_add_up(runs, name):
    d, cfg = runs[name]
    arm, T, starts = SPECS[name]
    v = np.load(os.path.join(d, "visitation.npz"))
    meta = json.loads(str(v["meta"]))
    n_2, n_3 = cfg["joint"]["starts"].get("stage2", 0), cfg["joint"]["starts"].get("stage3", 0)
    n_root = cfg["joint"]["starts"]["root"]
    assert int(v["stage2_root_path"].sum()) == n_root * U and int(v["stage2_direct_es"].sum()) == n_2 * U
    if T == 3:
        assert int(v["stage3_root_path"].sum()) == n_root * U and int(v["stage3_direct_es"].sum()) == n_3 * U
        assert int(v["stage3_from_stage2_start_approx"].sum()) == n_2 * U
        assert "stage3_from_stage2_start_approx" in meta["approximate"]
        assert "stage3_from_stage2_start_approx" not in meta["exact"]
    assert v["stage2_root_path"].shape == (40,) and v["stage2_bin_edges"].shape == (41,)


def test_root_only_runs_have_no_direct_start_counts(runs):
    v = np.load(os.path.join(runs["t3_root"][0], "visitation.npz"))
    for k in ("stage2_direct_es", "stage3_direct_es", "stage3_from_stage2_start_approx"):
        assert int(v[k].sum()) == 0


def test_visitation_sources_equal_the_bins_of_the_true_gaps(tmp_path, monkeypatch):
    """On one jt_es batch: the exact counts (root path, direct starts) and the approximate counts (stage-3 rows from stage-2
    starts, decoded from the float32 observation) equal the bins of the true float64 learner gaps recorded at the
    ``encode_obs`` calls; the rows of each source are those of its start stage within the stage-3 block."""
    from envs.curriculum_env import GameSpec, gap_bin_index
    d = str(tmp_path / "x")
    os.makedirs(d)
    jr = rj.JointRun(H.joint_cfg(d, "jt_es", 60, 50500, U=2, starts=H.ES_MIX_T3, check_every=2, window=False), d)
    recorded = []
    orig = GameSpec.encode_obs

    def spy(self, t, dd):
        recorded.append((int(t), np.array(dd, dtype=float)))
        return orig(self, t, dd)
    monkeypatch.setattr(GameSpec, "encode_obs", spy)
    t0, d0, roles = jr.draw_starts()
    batch = collect_batch_v2(jr.spec, jr.agent, t0, d0, roles, jr.rngs["env"], jr.rngs["learn"], jr.rngs["opp"], 1.0, 1.0, 10.0,
                             reward_mode="expected", frozen=None, continuation_action_mode="stochastic", cont_table=None,
                             clamp_likelihood="density")
    jr._accumulate_visitation(batch, t0)
    true3 = [dd for t, dd in recorded if t == 3][0]          # the first stage-3 call is the learner's (obs_L)
    true2 = [dd for t, dd in recorded if t == 2][0]
    assert true3.size == t0.size == 48
    for key, rows_ in (("stage3_root_path", t0 == 1), ("stage3_direct_es", t0 == 3), ("stage3_from_stage2_start_approx", t0 == 2)):
        b, nb = gap_bin_index(jr.spec, 3, true3[rows_], 10.0)
        assert nb == 88 and np.array_equal(jr.vis[key], np.bincount(b, minlength=nb)), key
    t02 = t0[t0 <= 2]                                        # the stage-2 block holds the episodes with t0 <= 2
    assert true2.size == t02.size == 32
    for key, rows_ in (("stage2_root_path", t02 == 1), ("stage2_direct_es", t02 == 2)):
        b, nb = gap_bin_index(jr.spec, 2, true2[rows_], 10.0)
        assert nb == 44 and np.array_equal(jr.vis[key], np.bincount(b, minlength=nb)), key


def test_a_stage3_only_joint_run_equals_the_ms_stage_3_phase_bit_for_bit(tmp_path):
    """JointRun with every episode started at stage 3 (and legacy windows) consumes every stream, sets every learning rate and
    ends in the state of ``MSRun``'s terminal-stage phase of the legacy bw arm: the training loop is the MS runner's."""
    n_ep, n = 48, 40
    d_ms = str(tmp_path / "ms")
    os.makedirs(d_ms)
    ms = rms.MSRun(H.bw_cfg(d_ms, 50, 50500, budgets=(n, 40, 20), epu=n_ep), d_ms)
    ms.run_stage(3)
    d_j = str(tmp_path / "j")
    os.makedirs(d_j)
    jr = rj.JointRun(H.joint_cfg(d_j, "jt", 50, 50500, U=n, epu=n_ep, starts={"root": 0, "stage2": 0, "stage3": n_ep},
                                 check_every=25), d_j)
    jr.train()
    fs_ms, fs_j = ms.agent.full_state(), jr.agent.full_state()
    for k in ("actor", "critic", "opponent", "opt_actor", "opt_critic", "rng_minibatch", "snapshot_refreshes"):
        assert H.tree_equal(fs_ms[k], fs_j[k]), k
    assert H.tree_equal({k: g.bit_generator.state for k, g in ms.rngs.items()}, {k: g.bit_generator.state for k, g in jr.rngs.items()})
    assert len(ms.rows) == len(jr.rows) == n
    for a, b in zip(ms.rows, jr.rows):
        for k in ("lr", "kl_final_epoch", "clip_frac", "policy_loss", "value_loss", "adv_raw_mean", "adv_raw_std",
                  "mean_episode_return", "rngpos_env", "rngpos_learn", "rngpos_opp", "rngpos_start", "rngpos_minibatch"):
            assert a[k] == b[k], (a["update"], k)
        assert a["mean_effort"] == b["mean_effort_t3"]
    wa, wb = np.load(os.path.join(d_ms, "weights", "u00025.npz")), np.load(os.path.join(d_j, "weights", "u00025.npz"))
    assert wa.files == wb.files and all(np.array_equal(wa[k], wb[k]) for k in wa.files)
    assert jr.init_state_sha256 == ms.init_state_sha256
    # the checks (pure) agree too: the development evaluation of the same live actor at update 25
    ca = read_csv(os.path.join(d_ms, "ms_checks_stage3.csv"))[0]
    cb = read_csv(os.path.join(d_j, "joint_checks.csv"))[0]
    assert ca["update"] == cb["update"] == "25" and float(ca["eta_T_over_dw"]) == float(cb["eta_T_over_dw"])
    assert float(ca["Gmax_full_over_dw"]) == float(cb["Gmax_full_over_dw"])


@pytest.mark.parametrize("name", ["t3_es", "t2_root"])
def test_same_config_twice_is_bit_identical(runs, tmp_path, name):
    d1, cfg1 = runs[name]
    arm, T, starts = SPECS[name]
    d2 = str(tmp_path / "again")
    os.makedirs(d2)
    cfg2 = copy.deepcopy(cfg1)
    cfg2["record"]["output_dir"] = d2
    assert rj.run_pipeline(cfg2, d2, "pytest") == 0
    for u in (10, 20, 30):
        a, b = np.load(os.path.join(d1, "weights", f"u{u:05d}.npz")), np.load(os.path.join(d2, "weights", f"u{u:05d}.npz"))
        assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files), u
    for f in ("final_eval_final.npz", "final_eval_development.npz", "visitation.npz"):
        a, b = np.load(os.path.join(d1, f)), np.load(os.path.join(d2, f))
        assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files), f
    for f, wall in (("joint_updates.csv", {"update_wall_sec"}), ("joint_checks.csv", {"verifier_sec"})):
        ra, rb = read_csv(os.path.join(d1, f)), read_csv(os.path.join(d2, f))
        strip = lambda rows: [{k: v for k, v in r.items() if k not in wall} for r in rows]   # noqa: E731
        assert strip(ra) == strip(rb), f
    sa = torch.load(os.path.join(d1, "state_end.pt"), weights_only=False)
    sb = torch.load(os.path.join(d2, "state_end.pt"), weights_only=False)
    assert H.tree_equal(sa["agent"], sb["agent"]) and H.tree_equal(sa["rng"], sb["rng"])
    assert H.tree_equal(sa["torch_generator_state"], sb["torch_generator_state"])


def test_the_last_iterate_equals_the_state_end_actor(runs):
    d, _ = runs["t3_root"]
    st = torch.load(os.path.join(d, "state_end.pt"), weights_only=False)
    w = np.load(os.path.join(d, "weights", "u00030.npz"))
    for k, v in st["agent"]["actor"].items():
        assert np.array_equal(w[f"actor.{k}"], v.numpy()), k


# ====================================================================== 4. returns and the closed form
@pytest.mark.parametrize("T", [2, 3])
def test_root_batch_returns_differ_by_the_stage_cost(tmp_path, T):
    """gamma = lambda = 1: return_t - return_{t+1} = -k e_t^2 (e_t from the stored stage-t action)."""
    d = str(tmp_path / "x")
    os.makedirs(d)
    cfg = H.joint_cfg(d, "jt" if T == 3 else "jt2", U=2, epu=64, check_every=1, window=False)
    jr = rj.JointRun(cfg, d)
    spec = jr.spec
    n = 64
    rng = np.random.default_rng(7)
    batch = collect_batch_v2(spec, jr.agent, np.ones(n, dtype=np.int64), np.zeros(n), rng.integers(0, 2, size=n),
                             jr.rngs["env"], jr.rngs["learn"], jr.rngs["opp"], 1.0, 1.0, 10.0,
                             reward_mode="expected", frozen=None, continuation_action_mode="stochastic",
                             cont_table=None, clamp_likelihood="density")
    stage = np.asarray(batch["stage"])
    ret = {t: np.asarray(batch["returns"], dtype=np.float64)[stage == t] for t in range(1, T + 1)}
    act = {t: np.asarray(batch["actions"], dtype=np.float64)[stage == t] for t in range(1, T + 1)}
    for t in range(1, T):
        e_t = spec.effort_from_action(act[t])
        diff = ret[t] - ret[t + 1]
        want = -spec.k * e_t ** 2
        tol = 1e-5 * np.maximum(1.0, np.abs(want))
        assert np.all(np.abs(diff - want) <= tol), (t, float(np.abs(diff - want).max()))
    assert [int((stage == t).sum()) for t in range(1, T + 1)] == [n] * T


def test_no_closed_form_equilibrium_in_the_runner():
    """The runner imports the closed-form validity check only; no equilibrium function is read."""
    src = open(os.path.join(str(H.ROOT), "run", "run_joint.py")).read()
    for name in ("g2_two_stage", "g1_two_stage", "v2_star", "eq_utility_two_stage", "recovery_metrics"):
        assert name not in src, name
    assert "from run.run_ms_stagewise" not in src and "import run_ms_stagewise" not in src


# ====================================================================== 5. bw at T = 3
@pytest.mark.parametrize("q", [50, 60])
def test_bw_production_config_has_the_d3_values_and_the_ms_runner_accepts_it(q):
    cfg = tc.build_bw_config(H.PROTO, q, 50501, "/x/y", H.BW_PARAMS)
    rms.validate_config(cfg)
    assert tc.production_problems(cfg) == []
    assert cfg["pipeline"] == {"T": 3, "phases": [3, 2, 1], "budgets": {"3": 1600, "2": 1600, "1": 600},
                               "lr_windows": tc.BW_WINDOWS}
    assert cfg["start_weights"] == {"scheme": "bin_balanced"} and cfg["rule"]["enabled"] is False
    assert cfg["rule"]["stages"] == {"3": {"eps": 0.005, "rho": 0.05, "tau": 0.02},
                                     "2": {"eps": 0.005, "rho": 0.03, "tau": 0.02},
                                     "1": {"eps": 0.005, "rho": 0.03, "tau": 0.02}}
    assert cfg["pilot"] == "t3_r1" and cfg["run"] == cfg["record"]["run"] == f"t3r1_q{q}_s50501_bw"
    assert cfg["init_digest"] is True and cfg["record"]["game"]["T"] == 3
    assert "es_bins_stage2" not in cfg["record"] and "domain_half_stage2" not in cfg["record"]
    for k in ("actor_variant", "conc_scale_schedule", "fixed_global_sampler", "noise_report", "focus_scope",
              "focus_map"):
        assert k not in cfg
    assert cfg["base_commit"] == "657f54a"
    n_bins = {3: {50: 80, 60: 88}[q], 2: {50: 40, 60: 44}[q]}
    from envs.curriculum_env import GameSpec, StartSampler
    smp = StartSampler(GameSpec(**cfg["record"]["game"]), 10.0)
    assert (smp.n_bins(3), smp.n_bins(2)) == (n_bins[3], n_bins[2])


def test_bw_reduced_legacy_t3_run_completes():
    """The legacy T = 3 path with the rule disabled: three phases, the two nested tables, gates.json with outcome
    not_evaluated_T3, the init digest in the manifest (the first test of this path)."""
    d, cfg = H.reduced_bw_run()
    for t in (3, 2, 1):
        for f in (f"ms_checks_stage{t}.csv", f"state_end_stage{t}.pt", f"freeze_stage{t}_final.npz",
                  f"freeze_stage{t}_development.npz"):
            assert os.path.exists(os.path.join(d, f)), f
    for t in (2, 1):
        assert os.path.exists(os.path.join(d, f"continuation_table_stage{t}.npz"))
    gates = json.load(open(os.path.join(d, "gates.json")))
    assert gates["outcome"] == "not_evaluated_T3" and gates["global_rng"]["status"] == "ok"
    rows = read_csv(os.path.join(d, "ms_updates.csv"))
    assert [int(r["stage"]) for r in rows] == [3] * 40 + [2] * 40 + [1] * 20
    log = json.load(open(os.path.join(d, "rule_log.json")))["stages"]
    assert all(log[t]["fire_local"] is None and log[t]["budget_forced"] is False for t in ("3", "2", "1"))
    assert [log[t]["total_updates"] for t in ("3", "2", "1")] == [40, 40, 20]
    man = json.load(open(os.path.join(d, "manifest.json")))
    assert len(man["init_state_sha256"]) == 64
    assert set(man["continuation_tables"]) == {"2", "1"}
    # the LR of the windows: 3e-4 up to local 20, linear over 21-40 (stages 3 and 2), linear over 1-20 (stage 1)
    lrs = [float(r["lr"]) for r in rows]
    assert lrs[19] == lrs[59] == lrs[80] == 3e-4                       # local 20 of stages 3 and 2; local 1 of stage 1
    assert lrs[39] == pytest.approx(3e-5, rel=1e-12) and lrs[79] == pytest.approx(3e-5, rel=1e-12)
    assert lrs[99] == pytest.approx(3e-5, rel=1e-12) and 3e-5 < lrs[30] < 3e-4
