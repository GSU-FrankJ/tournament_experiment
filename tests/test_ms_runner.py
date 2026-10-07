"""MS-R1 runner tests (``run/run_ms_stagewise.py``).

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_runner.py -p no:cacheprovider -q

Sections: legacy settings reproduce v2.0's Phase A bit for bit (reduced budget, both q); the C7 reference and the
v2.0 lock commit are unchanged; the table of stage 1 equals the v2.0 builder and is built without any RNG draw;
the T = 3 pipeline (three phases, nested tables, development stop and forced landing); the closed form does not
enter the rule or the sampler; configuration refusals.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "v2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import ms_configs as mc  # noqa: E402
import test_v2_r2b as T2B  # noqa: E402
from test_v2_r2b import (NEW_KEYS_AT_DEFAULT, _c7_equal, _run_locked_small, tree_equal,  # noqa: E402,F401
                         v20_reference_tree)
from agents.ppo_curriculum import BetaActor  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402
from run.run_final_dp_br_round3_dense import ConfigError  # noqa: E402
from run.run_v2_stagewise import Run, execute  # noqa: E402
from utils import ms_continuation as mcont  # noqa: E402
from utils import v2_continuation as vc  # noqa: E402

PROTO = mc.load_protocol()


def small_params(T: int = 2, K: int = 10, block: int = 20, cap: int = 40, land: int = 10, **stage) -> dict:
    """Reduced-budget rule parameters (every stage the same budgets, thresholds of D4)."""
    prm = copy.deepcopy(mc.DEFAULT_PARAMS)
    prm["K"] = K
    one = dict(eps=0.005, rho=0.03, tau=0.02, n_block=block, u_cap=cap, n_land=land)
    one.update(stage)
    prm["stages"] = {str(t): dict(one) for t in range(1, T + 1)}
    return prm


def make_cfg(out: str, arm: str = "MS_rule", q: int = 50, seed: int = 10501, T: int = 2, params=None,
             epu: int = 128) -> dict:
    """A reduced-budget ms_run_config/1 (the v2.0 record, episodes per update lowered)."""
    cfg = mc.build_config(PROTO, q, seed, arm, out, params or small_params(T), T=T)
    cfg["record"]["protocol"]["episodes_per_update"] = epu
    return cfg


def legacy_cfg(out: str, q: int = 50, seed: int = 10501, n2: int = 80, n1: int = 40, epu: int = 128) -> dict:
    """MS_base with the budgets and LR windows of v2.0 scaled down (window = the last half)."""
    cfg = make_cfg(out, "MS_base", q, seed, epu=epu)
    cfg["pipeline"]["budgets"] = {"2": n2, "1": n1}
    cfg["pipeline"]["lr_windows"] = {"2": [{"first": n2 // 2 + 1, "last": n2, "start": 3e-4, "end": 3e-5}],
                                     "1": [{"first": 1, "last": n1, "start": 3e-4, "end": 3e-5}]}
    return cfg


def v2_phase_a_cfg(q: int, seed: int, n: int, epu: int) -> dict:
    """The v2.0 ``phase_A`` run config with the same reduced settings as :func:`legacy_cfg`."""
    rec = copy.deepcopy(PROTO["records"][str(q)])
    rec.update(seed=seed, run="v2ref", output_dir="unused")
    return {"schema": "v2_run_config/1", "base_commit": PROTO["base_commit"], "pilot": "test", "arm": "x",
            "run": "v2ref", "q": q, "seed": seed, "mode": "phase_A", "fixed_budget": True,
            "flags": {"reward_mode": "expected", "stage2_update_mode": "joint", "adv_norm_scope": "all_rows",
                      "continuation_action_mode": "stochastic"},
            "parent_checkpoint": None, "parent_sha256": None, "record": rec,
            "threads_per_process": PROTO["threads_per_process"],
            "budget_overrides": {"phase_caps": {"A": n, "B": 10, "C": 1}, "warmup": 5, "stability_every": 5,
                                 "verifier_timeout": 40, "episodes_per_update": epu},
            "lr_decay": [{"phase": "A", "start_lr": 3e-4, "end_lr": 3e-5, "local_first": n // 2 + 1,
                          "local_last": n}], "full_state_at": []}


# ====================================================================== 1. legacy == v2.0 Phase A
@pytest.mark.parametrize("q,seed", [(50, 10501), (60, 10503)])
def test_legacy_terminal_phase_equals_v20_phase_a_bit_for_bit(tmp_path, q, seed):
    """MS_base settings: the terminal-stage phase ends in the training-relevant state of v2.0's phase_A and
    writes the same weight exports and the same per-update series (C-MS1 on a reduced budget)."""
    n, epu = 80, 128
    d_ref, d_ms = str(tmp_path / "v2"), str(tmp_path / "ms")
    os.makedirs(d_ref)
    os.makedirs(d_ms)
    cfg_ref = v2_phase_a_cfg(q, seed, n, epu)
    ref = Run(cfg_ref, d_ref)
    assert execute(ref, cfg_ref, d_ref, "pytest") == 0
    ms = rms.MSRun(legacy_cfg(d_ms, q, seed, n, 40, epu), d_ms)
    rec = ms.run_stage(2)
    assert rec["total_updates"] == n and rec["fire_local"] is None and rec["budget_forced"] is False
    st_ref = torch.load(os.path.join(d_ref, "state_end_A.pt"), weights_only=False)
    st_ms = ms.full_state("stage2")
    for k in ("actor", "critic", "opponent", "frozen", "opt_actor", "opt_critic", "rng_minibatch", "snapshot_refreshes"):
        assert tree_equal(st_ref["agent"][k], st_ms["agent"][k]), k
    assert tree_equal(st_ref["rng"], st_ms["rng"])
    assert tree_equal(st_ref["torch_generator_state"], st_ms["torch_generator_state"])
    assert (st_ref["counters"]["global_u"], st_ref["counters"]["total_episodes"]) == \
           (st_ms["counters"]["global_u"], st_ms["counters"]["total_episodes"])
    for u in range(25, n + 1, 25):
        a = np.load(os.path.join(d_ref, "weights", f"u{u:05d}.npz"))
        b = np.load(os.path.join(d_ms, "weights", f"u{u:05d}.npz"))
        assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files), u
    h_ref, h_ms = ref.history, ms.history
    assert len(h_ref) == len(h_ms) == n
    for x, y in zip(h_ref, h_ms):
        keys = sorted((set(x) & set(y)) - {"phase", "stage", "mean_effort_by_stage", "mean_effort"})
        assert tree_equal({k: x[k] for k in keys}, {k: y[k] for k in keys}), x["update"]
        assert x["mean_effort_by_stage"]["2"] == y["mean_effort"]
    for x, y in zip(ref.v2_history, ms.rows):   # the five stream positions after every update
        for k in ("rngpos_env", "rngpos_learn", "rngpos_opp", "rngpos_start", "rngpos_minibatch"):
            assert x[k] == y[k], (x["update"], k)


def test_c7_reference_is_unchanged(tmp_path):
    """The C7 regression pipeline (mode full) of the current tree equals the C7 reference."""
    cfg = json.load(open(T2B.C7_CONFIG))
    cfg.update(NEW_KEYS_AT_DEFAULT)
    d = str(tmp_path / "c7")
    os.makedirs(d)
    assert execute(Run(cfg, d), cfg, d, "pytest") == 0
    _c7_equal(d)


def test_the_locked_v20_pipeline_equals_the_lock_commit_on_a_reduced_run(v20_reference_tree, tmp_path):
    """C-R4 in miniature: the reduced locked run of the current tree ends in the state of the v2.0 lock commit."""
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


# ====================================================================== 2. the table of stage 1
def test_the_stage_1_table_equals_the_v20_builder_and_is_built_without_rng(tmp_path):
    """T = 2: V~_2 of the frozen stage-2 snapshot equals utils.v2_continuation.build_continuation_table bit for
    bit; the build moved no training stream and no process-global RNG; the written NPZ holds the table."""
    d = str(tmp_path / "t2")
    os.makedirs(d)
    cfg = make_cfg(d, params=small_params(2, cap=20, block=20, land=6))
    run = rms.MSRun(cfg, d)
    rec = run.run_stage(2)
    run.freeze_stage(2, rec, lambda point: None)
    ref = vc.build_continuation_table(run.frozen_nets[2], run.spec, stage=2, step=0.05)
    got = run.tables[1]
    assert np.array_equal(ref.y_grid, got.y_grid) and np.array_equal(ref.values, got.values)
    tr = run.table_records["1"]
    assert tr["file"] == "continuation_table_stage1.npz"
    assert tr["rng_unchanged_by_build"] == {"training": True, "torch_global": True, "numpy_global": True,
                                            "python_random": True}
    z = np.load(os.path.join(d, tr["file"]))
    assert np.array_equal(z["values"], ref.values) and np.array_equal(z["y_grid"], ref.y_grid)
    assert run.violations == []


# ====================================================================== 3. T = 3 pipeline
def _t3_params(loose: bool) -> dict:
    prm = small_params(3, K=5, block=10, cap=20 if not loose else 30, land=6)
    if loose:   # every check eligible: the development stop fires after M = 3 checks
        for t in prm["stages"]:
            prm["stages"][t].update(eps=1e9, rho=1e9, tau=1e9)
        prm["conc_limit"] = 1.0
    return prm


@pytest.mark.parametrize("loose", [True, False])
def test_t3_pipeline_three_phases_nested_tables_and_the_rule(tmp_path, loose):
    """Reduced-budget T = 3 pipeline: phases 3, 2, 1; the stop path (every check eligible) and the forced-landing
    path (cap reached); nested tables equal an independent rebuild from the frozen snapshots; frozen snapshots
    are never trained."""
    d = str(tmp_path / "t3")
    os.makedirs(d)
    cfg = make_cfg(d, T=3, params=_t3_params(loose), epu=48)
    assert rms.run_pipeline(cfg, d, "pytest") == 0
    for t in (3, 2, 1):
        assert os.path.exists(os.path.join(d, f"ms_checks_stage{t}.csv"))
        assert os.path.exists(os.path.join(d, f"state_end_stage{t}.pt"))
    assert os.path.exists(os.path.join(d, "continuation_table_stage2.npz"))
    assert os.path.exists(os.path.join(d, "continuation_table_stage1.npz"))
    log = json.load(open(os.path.join(d, "rule_log.json")))["stages"]
    for t in ("3", "2", "1"):
        st = log[t]
        assert st["landing"]["n_land"] == 6 and st["landing"]["done"]
        if loose:
            assert st["fire_local"] == 15 and st["budget_forced"] is False       # 3 checks of K = 5
            assert st["blocks"][-1]["exit_reason"] == "development_stop"
            assert st["total_updates"] == 15 + 6
        else:
            assert st["fire_local"] is None and st["budget_forced"] is True
            assert st["training_updates"] == 20 and st["total_updates"] == 26
            assert [b["exit_reason"] for b in st["blocks"]] == ["block_end", "cap"]
    # the tail term is void where there is no tail bin (stage 2 of T = 3 at q = 50) and applies at stage 3
    rows = {t: list(csv.DictReader(open(os.path.join(d, f"ms_checks_stage{t}.csv")))) for t in (3, 2, 1)}
    assert all(r["tail_term"] == "True" for r in rows[3]) and all(r["tail_term"] == "False" for r in rows[2])
    assert all(r["tail_term"] == "False" for r in rows[1])
    # nested tables: independent rebuild from the frozen snapshots saved in state_end_stage1.pt
    spec = rms.strict_dataclass(rms.GameSpec, cfg["record"]["game"], "game")
    fz = torch.load(os.path.join(d, "state_end_stage1.pt"), weights_only=False)["frozen_stages"]
    nets = {}
    for s in (2, 3):
        net = BetaActor(64, 100.0, 1e-6, torch.Generator().manual_seed(0))
        net.load_state_dict(fz[str(s)])
        net.eval()
        nets[s] = net
    tables = mcont.build_nested_tables(nets, spec, 1)
    for t in (2, 1):
        z = np.load(os.path.join(d, f"continuation_table_stage{t}.npz"))
        assert np.array_equal(z["y_grid"], tables[t].y_grid) and np.array_equal(z["values"], tables[t].values), t
    man = json.load(open(os.path.join(d, "manifest.json")))
    assert [man["continuation_tables"][k]["meta"]["nested"] for k in ("2", "1")] == [False, True]
    assert json.load(open(os.path.join(d, "drift_test.json")))["pass"] is True
    g = json.load(open(os.path.join(d, "gates.json")))
    assert g["global_rng"]["status"] == "ok" and g["outcome"] == "not_evaluated_T3"


# ====================================================================== 4. no closed form in the rule / sampler
def test_decisions_and_sampler_do_not_depend_on_the_closed_form(tmp_path, monkeypatch):
    """Replacing the closed-form functions by stubs changes neither the checks' D3 columns, nor the blocks, nor the
    sampler probabilities, nor the training state."""
    import utils.v2_metrics as vm
    recs, states, rows = [], [], []
    for tag in ("real", "stub"):
        d = str(tmp_path / tag)
        os.makedirs(d)
        cfg = make_cfg(d, params=small_params(2, K=10, block=20, cap=40, land=10), epu=64)
        if tag == "stub":
            def junk(policy, spec, step):
                D = np.linspace(-spec.domain_half(2), spec.domain_half(2), 9)
                sc = {"g1": 1.0, "g2_at_0": 1.0, "e1_at_0": 7.0, "e2_at_0": 3.0, "stage1_rel_err_signed": 123.0,
                      "stage2_peak_rel_err_signed": -9.0, "stage2_peak_rel_err_abs": 9.0}
                return sc, {"recovery_d_grid": D, "recovery_e2": D * 0, "recovery_g2": D * 0 + 5.0}
            monkeypatch.setattr(vm, "recovery_metrics", junk)
        run = rms.MSRun(cfg, d)
        recs.append(run.run_stage(2))
        states.append(run.full_state("stage2"))
        rows.append(list(csv.DictReader(open(os.path.join(d, "ms_checks_stage2.csv")))))
    cols = ("update", "local", "mode", "block_id", "block_type", "valid", "Delta", "s", "R", "R_tail", "C",
            "eligible", "consecutive", "p_digest", "p_digest_next")
    assert [{c: r[c] for c in cols} for r in rows[0]] == [{c: r[c] for c in cols} for r in rows[1]]
    assert rows[0][0]["stage2_peak_rel_err_signed"] != rows[1][0]["stage2_peak_rel_err_signed"]   # the stub was used
    keep = ("blocks", "landing", "fire_local", "would_fire_local", "budget_forced", "training_updates")
    assert {k: recs[0][k] for k in keep} == {k: recs[1][k] for k in keep}
    for k in ("agent", "rng", "torch_generator_state"):
        assert tree_equal(states[0][k], states[1][k]), k


# ====================================================================== 5. start shares
def test_measured_start_shares_follow_the_design(tmp_path):
    """MS_s25a0 at q = 50: per update the start counts add up to the batch and the pooled tail share is lambda_T."""
    d = str(tmp_path / "s")
    os.makedirs(d)
    cfg = make_cfg(d, arm="MS_s25a0", params=small_params(2, K=10, block=20, cap=40, land=10), epu=512)
    run = rms.MSRun(cfg, d)
    run.run_stage(2)
    rows = [r for r in run.rows if r["stage"] == 2]
    n_ep = 512
    assert all(r["n_start_tail"] + r["n_start_near"] + r["n_start_mid"] == n_ep for r in rows)
    tot = len(rows) * n_ep
    lam_t, lam_p = 0.5, 0.25
    for key, p in (("n_start_tail", lam_t), ("n_start_near", lam_p), ("n_start_mid", 1.0 - lam_t - lam_p)):
        share = sum(r[key] for r in rows) / tot
        assert abs(share - p) < 4.0 * np.sqrt(p * (1 - p) / tot), (key, share, p)


# ====================================================================== 6. configuration refusals
def _ok(**kw) -> dict:
    return make_cfg("unused", **kw)


def test_valid_configs_pass():
    for arm in mc.ARMS:
        rms.validate_config(_ok(arm=arm, q=50))
        rms.validate_config(_ok(arm=arm, q=60))


@pytest.mark.parametrize("mutate,msg", [
    (lambda c: c.__setitem__("schema", "ms_run_config/2"), "schema"),
    (lambda c: c.__setitem__("surprise", 1), "unknown"),
    (lambda c: c.pop("rule"), "missing"),
    (lambda c: c["pipeline"].__setitem__("T", 4), "pipeline.T"),
    (lambda c: c["pipeline"].__setitem__("T", 1), "pipeline.T"),
    (lambda c: c["pipeline"].__setitem__("extra", 1), "pipeline"),
    (lambda c: c["pipeline"].__setitem__("phases", [1, 2]), "pipeline.phases"),
    (lambda c: c["pipeline"].__setitem__("budgets", {"2": 1600, "1": 600}), "fixed budget"),   # enabled with a budget
    (lambda c: c.__setitem__("clamp_likelihood", "censored"), "clamp_likelihood"),
    (lambda c: c["start_weights"].__setitem__("unknown_key", 1), "start_weights"),
    (lambda c: c["start_weights"].__setitem__("lambda_P", 0.6), "start_weights"),            # lambda_M <= 0
    (lambda c: c["start_weights"].__setitem__("lambda_P", 0.5), "start_weights"),            # lambda_M = 0
    (lambda c: c["start_weights"].__setitem__("lambda_P", 0.0), "lambda_P"),
    (lambda c: c["start_weights"].__setitem__("alpha_global", 1.5), "alpha"),
    (lambda c: c["start_weights"].__setitem__("ema_beta", 1.0), "ema_beta"),
    (lambda c: c["start_weights"].__setitem__("lambda_T", 0.4), "coverage"),                  # smaller than n_tail/n
    (lambda c: c["start_weights"].__setitem__("lambda_T", 0.6), "coverage"),
    (lambda c: c["start_weights"].__setitem__("near_tie_half_width", 150.0), "start_weights"),  # reaches the tail
    (lambda c: c["derived"]["2"].__setitem__("n_tail", 19), "derived"),
    (lambda c: c["rule"].__setitem__("enabled", False), "rule.stages"),                       # legacy keys mismatch
    (lambda c: c["rule"]["stages"]["2"].__setitem__("fixed_budget", 1600), "rule.stages"),   # budget key in the rule
    (lambda c: c["rule"]["stages"]["2"].__setitem__("n_land", 1), "n_land"),
    (lambda c: c["rule"]["stages"]["2"].__setitem__("u_cap", 0), "u_cap"),
    (lambda c: c["rule"]["stages"]["2"].__setitem__("rho", -0.1), "rho"),
    (lambda c: c["rule"]["stages"]["2"].pop("tau"), "rule.stages"),
    (lambda c: c["rule"]["stages"].pop("1"), "rule.stages"),
    (lambda c: c["rule"].__setitem__("K", 0), "rule.K"),
    (lambda c: c["rule"].__setitem__("conc_limit", 0.0), "conc_limit"),
    (lambda c: c.__setitem__("start_weights", {"scheme": "bin_balanced"}), "stratified_priority"),  # enabled rule
    (lambda c: c["record"].__setitem__("seed", 1), "seed"),
    (lambda c: c["record"]["game"].__setitem__("T", 3), "game.T"),
    (lambda c: c.__setitem__("protocol_gates", {}), "protocol_gates"),
])
def test_config_refusals(mutate, msg):
    cfg = _ok(arm="MS_rule")
    mutate(cfg)
    with pytest.raises(ConfigError) as ei:
        rms.validate_config(cfg)
    assert msg in str(ei.value)


def test_the_legacy_arm_needs_its_fixed_budget_and_windows():
    cfg = _ok(arm="MS_base")
    rms.validate_config(cfg)
    assert cfg["pipeline"]["budgets"] == {"2": 1600, "1": 600} and cfg["rule"]["enabled"] is False
    for mutate, msg in [
        (lambda c: c["pipeline"].__setitem__("budgets", None), "legacy arm"),
        (lambda c: c["pipeline"].__setitem__("lr_windows", None), "legacy arm"),
        (lambda c: c["pipeline"]["budgets"].pop("1"), "legacy arm"),
        (lambda c: c["rule"]["stages"]["2"].__setitem__("n_block", 400), "rule.stages"),
        (lambda c: c["pipeline"]["budgets"].__setitem__("2", 0), "fixed budget"),
        (lambda c: c["pipeline"]["budgets"].__setitem__("2", 1500), "fixed budget"),   # windows end at 1600
        (lambda c: c["pipeline"]["lr_windows"]["2"][0].__setitem__("first", 0), "lr window"),
        (lambda c: c["pipeline"]["lr_windows"]["2"].append(
            {"first": 1500, "last": 1550, "start": 3e-4, "end": 3e-5}), "contiguous"),
    ]:
        c = copy.deepcopy(cfg)
        mutate(c)
        with pytest.raises(ConfigError) as ei:
            rms.validate_config(c)
        assert msg in str(ei.value), (msg, str(ei.value))


def test_the_derived_start_weights_are_written_and_cover_the_tail():
    cfg = _ok(arm="MS_s35a5", q=50)
    d = cfg["derived"]["2"]
    assert (d["n_bins"], d["n_tail"], d["n_near"], d["n_mid"]) == (40, 20, 4, 16)
    assert d["lambda_T"] == 0.5 and d["lambda_P"] == 0.35 and abs(d["lambda_M"] - 0.15) < 1e-15
    cfg60 = _ok(arm="MS_s35a5", q=60)
    d = cfg60["derived"]["2"]
    assert (d["n_bins"], d["n_tail"], d["n_near"], d["n_mid"]) == (44, 20, 4, 20)
    assert abs(d["lambda_T"] - 20 / 44) < 1e-15
    rule = _ok(arm="MS_rule", q=50)["start_weights"]
    assert rule["lambda_P"] == 4 / 40 and rule["alpha_global"] == 0.0
    assert _ok(arm="MS_rule", q=60)["start_weights"]["lambda_P"] == 4 / 44


def test_pilot_arms_differ_from_ms_rule_only_in_start_weights():
    base = _ok(arm="MS_rule", q=50)
    for arm in mc.PILOT_ARMS[1:]:
        c = _ok(arm=arm, q=50)
        diff = {k for k in set(base) | set(c) if base.get(k) != c.get(k)}
        assert diff <= {"arm", "run", "start_weights", "derived", "record"}, (arm, diff)
        rd = {k for k in set(base["record"]) | set(c["record"]) if base["record"].get(k) != c["record"].get(k)}
        assert rd <= {"run", "output_dir"}, rd
        assert base["rule"] == c["rule"]


# ====================================================================== 7. process-global RNG hardening
def test_a_global_rng_draw_after_the_stage_1_phase_is_caught_and_labelled(tmp_path, monkeypatch):
    """A draw from a process-global RNG inside the stage-1 decomposition (after the last training update) is caught by
    the final assertion: exit code 5, the violation recorded, the outcome label is the violation."""
    import random
    real = rms.decomposition

    def drawing(*a, **k):
        random.random()
        return real(*a, **k)
    monkeypatch.setattr(rms, "decomposition", drawing)
    d = str(tmp_path / "v")
    os.makedirs(d)
    cfg = make_cfg(d, params=small_params(2, K=10, block=20, cap=20, land=6), epu=64)
    assert rms.run_pipeline(cfg, d, "pytest", band_step=2.0) == rms.RNG_VIOLATION_EXIT
    g = json.load(open(os.path.join(d, "gates.json")))
    assert g["global_rng"]["status"] == "violation"
    assert {v["rng"] for v in g["global_rng"]["violations"]} == {"python_random"}
    assert g["global_rng"]["violations"][0]["point"] == "end_of_run"
    assert g["outcome"] == "global_rng_violation" and g["outcome_gates"] in ("pass", "stage2_failure", "fail_G-F",
                                                                           "fail_G-N", "fail_G-S")
    assert g["run_pass_v20_combination_and_rng"] is False
    assert json.load(open(os.path.join(d, "status.json")))["exit_code"] == rms.RNG_VIOLATION_EXIT
