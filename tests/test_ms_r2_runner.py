"""MS-R2 runner tests: the concentration-scale schedule, the fixed-budget global sampler, the reporting columns and
the prefix identities C-NL / C-MS3 / C-MS4 on reduced budgets.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_r2_runner.py -p no:cacheprovider -q

Sections: (1) without the new keys the runner is bit-identical to commit 71c58904 (two reduced configs, run in a
checkout of that commit); (2) the scale before each update equals the D2 table and is applied to the live actor and
the lagged opponent, carried by the snapshot refresh and the frozen snapshot, reset at stage-1 entry; (3) the Beta mean
and the action spread under the scale; the stage-1 table from a scaled snapshot; (4) the prefix identities; (5) the
reporting columns; (6) configuration refusals.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import ms_configs as mc  # noqa: E402
from agents.ppo_curriculum import BetaActor  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402
from run.run_final_dp_br_round3_dense import ConfigError  # noqa: E402
from test_ms_runner import make_cfg, small_params, tree_equal  # noqa: E402
from utils import ms_continuation as mcont  # noqa: E402

PROTO = mc.load_protocol()
PARAMS = json.load(open(ROOT / "reports" / "ms" / "r1" / "prereg_parameters.json"))
BASE_COMMIT = "71c58904"
WALL = {"update_wall_sec", "verifier_sec", "diag_sec", "run", "arm"}      # wall clock and the run / arm labels
EPU = 64


# ---------------------------------------------------------------------------------------------- reduced configs
def reduced_nl(arm: str, out: str, q: int = 50, seed: int = 10501, n2: int = 60, n1: int = 20, ramp: Tuple[int, int] = (21, 40),
               const_to: int = 48, K: int = 10, weights_every: int = 5) -> Dict[str, Any]:
    """An NL arm with budgets scaled down: ramp ``ramp``, LR constant to ``const_to`` then linear to ``n2``."""
    prm = copy.deepcopy(PARAMS)
    prm["K"] = K
    cfg = mc.build_config(PROTO, q, seed, arm, out, prm)
    cfg["record"]["protocol"]["episodes_per_update"] = EPU
    cfg["record"]["protocol"]["weights_every"] = weights_every
    cfg["pipeline"]["budgets"] = {"2": n2, "1": n1}
    cfg["pipeline"]["lr_windows"] = {"2": [{"first": const_to + 1, "last": n2, "start": 3e-4, "end": 3e-5}],
                                     "1": [{"first": 1, "last": n1, "start": 3e-4, "end": 3e-5}]}
    if "conc_scale_schedule" in cfg:
        cfg["conc_scale_schedule"].update(local_first=ramp[0], local_last=ramp[1])
    return cfg


def run_cfg(cfg: Dict[str, Any], out: str) -> None:
    os.makedirs(out, exist_ok=True)
    assert rms.run_pipeline(cfg, out, "pytest", band_step=2.0) == 0


def read_csv(path: str) -> List[Dict[str, str]]:
    return list(csv.DictReader(open(path)))


def weights(d: str, u: int) -> Dict[str, np.ndarray]:
    z = np.load(os.path.join(d, "weights", f"u{u:05d}.npz"))
    return {k: z[k] for k in z.files}


def same_weights(a: str, b: str, u: int) -> bool:
    x, y = weights(a, u), weights(b, u)
    return x.keys() == y.keys() and all(np.array_equal(x[k], y[k]) for k in x)


def equal_rows(a: List[Dict[str, str]], b: List[Dict[str, str]], upto: int, skip: Tuple[str, ...] = (),
               only: Optional[Tuple[str, ...]] = None) -> bool:
    """Rows with ``update <= upto`` equal on the columns common to both (minus wall clock and ``skip``)."""
    ra = [r for r in a if int(r["update"]) <= upto]
    rb = [r for r in b if int(r["update"]) <= upto]
    if len(ra) != len(rb) or not ra:
        return False
    cols = (set(ra[0]) & set(rb[0])) - WALL - set(skip)
    if only is not None:
        cols &= set(only)
    return all(x[c] == y[c] for x, y in zip(ra, rb) for c in cols) and len(cols) >= 3


def same_history(a: str, b: str, upto: int) -> bool:
    ha = json.load(open(os.path.join(a, "train_history.json")))["history"]
    hb = json.load(open(os.path.join(b, "train_history.json")))["history"]
    ha, hb = [h for h in ha if h["update"] <= upto], [h for h in hb if h["update"] <= upto]
    keys = sorted((set(ha[0]) & set(hb[0])) - {"stage", "local"})
    return len(ha) == len(hb) == upto and all(tree_equal({k: x[k] for k in keys}, {k: y[k] for k in keys})
                                              for x, y in zip(ha, hb))


# ---------------------------------------------------------------------------------------------- 1. defaults == 71c58904
@pytest.fixture(scope="module")
def base_tree(tmp_path_factory) -> str:
    """The run code of commit 71c58904 (``git archive`` of run utils envs agents protocols) in a temp directory."""
    try:
        tar = subprocess.check_output(["git", "-C", str(ROOT), "archive", BASE_COMMIT, "--", "run", "utils", "envs",
                                       "agents", "protocols"], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pytest.skip(f"commit {BASE_COMMIT} is not available in this clone")
    d = tmp_path_factory.mktemp("tree_71c58904")
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(d)
    return str(d)


def run_in_tree(tree: str, cfg: Dict[str, Any], out: str) -> None:
    os.makedirs(out, exist_ok=True)
    cfg_path = os.path.join(out, "run_config.json")
    json.dump(cfg, open(cfg_path, "w"))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-B", os.path.join(tree, "run", "run_ms_stagewise.py"), "--config", cfg_path,
                        "--out-dir", out], cwd=tree, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def _same_run_outputs(a: str, b: str) -> None:
    for st in ("stage2", "stage1"):
        sa = torch.load(os.path.join(a, f"state_end_{st}.pt"), weights_only=False)
        sb = torch.load(os.path.join(b, f"state_end_{st}.pt"), weights_only=False)
        for k in ("agent", "rng", "torch_generator_state", "counters"):
            assert tree_equal(sa[k], sb[k]), (st, k)
    assert sorted(os.listdir(os.path.join(a, "weights"))) == sorted(os.listdir(os.path.join(b, "weights")))
    for f in sorted(os.listdir(os.path.join(a, "weights"))):
        assert same_weights(a, b, int(f[1:6])), f
    ra, rb = read_csv(os.path.join(a, "ms_updates.csv")), read_csv(os.path.join(b, "ms_updates.csv"))
    assert len(ra) == len(rb) and equal_rows(ra, rb, 10 ** 9)
    assert set(ra[0]) == set(rb[0])                       # no new column when the keys are absent
    for st in (2, 1):
        ca, cb = read_csv(os.path.join(a, f"ms_checks_stage{st}.csv")), read_csv(os.path.join(b, f"ms_checks_stage{st}.csv"))
        assert set(ca[0]) == set(cb[0]) and equal_rows(ca, cb, 10 ** 9)


def test_without_the_new_keys_a_legacy_run_equals_71c58904(base_tree, tmp_path):
    """MS_base2400's configuration (reduced budgets): the current tree and the 71c58904 tree end in the same state."""
    prm = copy.deepcopy(PARAMS)
    prm["K"] = 10
    cfg = mc.build_config(PROTO, 50, 10501, "MS_base2400", str(tmp_path / "x"), prm)
    cfg["record"]["protocol"]["episodes_per_update"] = EPU
    cfg["pipeline"]["budgets"] = {"2": 60, "1": 20}
    cfg["pipeline"]["lr_windows"] = {"2": [{"first": 41, "last": 60, "start": 3e-4, "end": 3e-5}],
                                     "1": [{"first": 1, "last": 20, "start": 3e-4, "end": 3e-5}]}
    for k in ("conc_scale_schedule", "fixed_global_sampler", "noise_report"):
        assert k not in cfg
    a, b = str(tmp_path / "now"), str(tmp_path / "old")
    cfg_a, cfg_b = copy.deepcopy(cfg), copy.deepcopy(cfg)
    run_cfg(cfg_a, a)
    run_in_tree(base_tree, cfg_b, b)
    _same_run_outputs(a, b)


def test_without_the_new_keys_a_rule_run_equals_71c58904(base_tree, tmp_path):
    """The MS_s35a5 rule configuration (reduced budgets, a polishing-capable rule): same state in both trees."""
    prm = small_params(2, K=10, block=20, cap=40, land=20, rho=0.05)
    cfg = make_cfg(str(tmp_path / "x"), "MS_s35a5", params=prm, epu=EPU)
    cfg["record"]["protocol"]["weights_every"] = 5
    a, b = str(tmp_path / "now"), str(tmp_path / "old")
    cfg_a, cfg_b = copy.deepcopy(cfg), copy.deepcopy(cfg)
    run_cfg(cfg_a, a)
    run_in_tree(base_tree, cfg_b, b)
    _same_run_outputs(a, b)
    la, lb = json.load(open(os.path.join(a, "rule_log.json"))), json.load(open(os.path.join(b, "rule_log.json")))
    assert la["stages"]["2"]["params"] == lb["stages"]["2"]["params"]       # the new StageRule field is not logged at its default


# ---------------------------------------------------------------------------------------------- 2. the schedule
def test_scale_function_follows_the_d2_table():
    """The real schedule (ramp 2001-2200): 1 up to update 2001, linear to s at 2200, s afterwards."""
    for s in (4.0, 16.0):
        cfg = mc.build_config(PROTO, 50, 10501, f"NL_bb_s{int(s)}", "/tmp/unused", PARAMS)
        ms = rms.MSRun(cfg, "/tmp/unused")
        f = ms.conc_scale_for
        assert f(1) == f(2000) == f(2001) == 1.0
        assert f(2002) == pytest.approx(1.0 + (s - 1.0) / 199.0, rel=1e-15)
        assert f(2100) == pytest.approx(1.0 + (s - 1.0) * 99.0 / 199.0, rel=1e-15)
        assert f(2199) == pytest.approx(s - (s - 1.0) / 199.0, rel=1e-12)
        assert f(2200) == f(2201) == f(2400) == f(2800) == s
        assert all(f(j) <= f(j + 1) for j in range(1, 2800))


def test_applied_to_live_actor_and_opponent_carried_by_refresh_and_reset_at_stage_1(tmp_path):
    out = str(tmp_path / "s4")
    cfg = reduced_nl("NL_bb_s4", out)
    os.makedirs(out)
    ms = rms.MSRun(cfg, out)
    seen: List[Tuple[int, float, float]] = []
    orig_update = ms.agent.update
    refreshes: List[Tuple[float, float]] = []
    orig_refresh = ms.agent.refresh_snapshot

    def spy_update(*a: Any, **k: Any) -> Any:
        seen.append((ms.global_u, float(ms.agent.actor.conc_scale), float(ms.agent.opponent.conc_scale)))
        return orig_update(*a, **k)

    def spy_refresh() -> None:
        orig_refresh()
        refreshes.append((float(ms.agent.actor.conc_scale), float(ms.agent.opponent.conc_scale)))
    ms.agent.update, ms.agent.refresh_snapshot = spy_update, spy_refresh
    rec = ms.run_stage(2)
    for u, sa, so in seen:                                  # the table of D2 on the reduced ramp, both players
        assert sa == so == ms.conc_scale_for(u), u
    assert [u for u, sa, _ in seen if sa != 1.0][0] == 22   # 1.0 through the first ramp update, then it moves
    assert all(a == o for a, o in refreshes)                # the snapshot refresh carries the scale to the opponent
    ms.freeze_stage(2, rec, lambda point: None)
    assert ms.frozen_nets[2].conc_scale == 4.0              # the frozen snapshot keeps s
    seen.clear()
    refreshes.clear()
    rms.MSRun.run_stage(ms, 1)
    assert refreshes[0] == (1.0, 1.0)                       # reset before the phase-entry refresh
    assert all(sa == so == 1.0 for _, sa, so in seen)
    assert ms.frozen_nets[2].conc_scale == 4.0


def test_the_recorded_scale_column_and_the_s1_arms_never_set_a_scale(tmp_path):
    d1, d4 = str(tmp_path / "s1"), str(tmp_path / "s4")
    cfg1, cfg4 = reduced_nl("NL_bb_s1", d1), reduced_nl("NL_bb_s4", d4)
    assert "conc_scale_schedule" not in cfg1 and cfg1["noise_report"] is True
    run_cfg(cfg1, d1)
    run_cfg(cfg4, d4)
    r1, r4 = read_csv(os.path.join(d1, "ms_updates.csv")), read_csv(os.path.join(d4, "ms_updates.csv"))
    ms = rms.MSRun(cfg4, d4)
    for r in r4:
        want = ms.conc_scale_for(int(r["local"])) if r["stage"] == "2" else 1.0
        assert float(r["conc_scale"]) == want, r["update"]
    assert all(float(r["conc_scale"]) == 1.0 for r in r1)
    assert "conc_scale" not in weights(d1, 50)              # an unscaled export carries no scale array
    assert float(weights(d4, 50)["conc_scale"]) == 4.0
    assert "conc_scale" not in weights(d4, 70)              # stage-1 exports are unscaled again


# ---------------------------------------------------------------------------------------------- 3. mean, spread, table
def _trained_actor(q: int = 50, seed: int = 3) -> BetaActor:
    g = torch.Generator().manual_seed(seed)
    net = BetaActor(64, 100.0, 1e-6, g)
    with torch.no_grad():                                   # a non-trivial head: mean 0.7, varying concentration
        net.out.weight.copy_(torch.randn(2, 64, generator=g) * 0.05)
        net.out.bias.copy_(torch.tensor([0.847, 0.3]))
    return net.eval()


def test_the_beta_mean_does_not_depend_on_the_scale():
    net = _trained_actor()
    x = torch.tensor(np.stack([np.linspace(-150, 150, 301), np.zeros(301)], axis=1), dtype=torch.float32) / 100.0
    with torch.no_grad():
        a1, b1 = net(x)
        worst = 0.0
        for s in (1.5, 4.0, 16.0, 100.0):
            net.conc_scale = s
            a, b = net(x)
            worst = max(worst, float((100.0 * (a / (a + b) - a1 / (a1 + b1))).abs().max()))
            assert torch.allclose(a + b, (a1 + b1) * s, rtol=1e-6)
        net.conc_scale = 1.0
    print(f"\nmax |mean effort(s) - mean effort(1)| = {worst:.3e} effort units")
    assert worst <= 1e-4


def test_the_action_spread_scales_as_one_over_sqrt_of_concentration():
    net = _trained_actor()
    x = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    rng = np.random.default_rng(20261007)
    n = 10 ** 6
    with torch.no_grad():
        a1, b1 = net(x)
    c1 = float(a1 + b1)
    mu = float(a1 / (a1 + b1))
    for s in (1.0, 4.0, 16.0):
        net.conc_scale = s
        with torch.no_grad():
            a, b = net(x)
        e = 100.0 * rng.beta(float(a), float(b), size=n)
        sd_theory = 100.0 * math.sqrt(mu * (1.0 - mu) / (c1 * s + 1.0))
        se = sd_theory / math.sqrt(2.0 * n)                 # standard error of a sample SD (Gaussian approximation)
        assert abs(e.std() - sd_theory) <= 3.0 * se, (s, e.std(), sd_theory)
    net.conc_scale = 1.0


def test_the_stage_1_table_of_a_scaled_snapshot_equals_the_unscaled_one(tmp_path):
    """Max |V~_1(scaled) - V~_1(scale 1)| in units of DW (reported, limit 1e-6 DW) for s = 4 and s = 16."""
    cfg = mc.build_config(PROTO, 50, 10501, "NL_bb_s4", str(tmp_path), PARAMS)
    spec = rms.MSRun(cfg, str(tmp_path)).spec
    net = _trained_actor()
    base = mcont.build_table(copy.deepcopy(net), spec, 1, None, step=rms.CONT_TABLE_STEP)
    for s in (4.0, 16.0):
        sn = copy.deepcopy(net)
        sn.conc_scale = s
        tb = mcont.build_table(sn, spec, 1, None, step=rms.CONT_TABLE_STEP)
        diff = float(np.abs(np.asarray(tb.values) - np.asarray(base.values)).max()) / spec.dw
        print(f"\nmax |table(s={s:g}) - table(1)| / DW = {diff:.3e}")
        assert diff <= 1e-6


# ---------------------------------------------------------------------------------------------- 4. prefix identities
def test_c_nl_scale_runs_equal_the_s1_run_through_the_first_ramp_update(tmp_path):
    for kind in ("bb", "st"):
        d1 = str(tmp_path / f"{kind}1")
        run_cfg(reduced_nl(f"NL_{kind}_s1", d1), d1)
        for s in (4, 16):
            d = str(tmp_path / f"{kind}{s}")
            run_cfg(reduced_nl(f"NL_{kind}_s{s}", d), d)
            for u in (5, 10, 15, 20):
                assert same_weights(d1, d, u), (kind, s, u)
            assert not same_weights(d1, d, 25), (kind, s)             # the scale has moved from update 22
            assert same_history(d1, d, 21) and not same_history(d1, d, 22)
            assert equal_rows(read_csv(os.path.join(d1, "ms_updates.csv")), read_csv(os.path.join(d, "ms_updates.csv")), 21)
            assert equal_rows(read_csv(os.path.join(d1, "ms_checks_stage2.csv")),
                              read_csv(os.path.join(d, "ms_checks_stage2.csv")), 20)


def test_c_ms3_bin_balanced_nl_equals_the_budget_control_through_its_last_constant_lr_update(tmp_path):
    """NL_bb_s1 (constant LR to 48) against MS_base2400-like (window 41-48 at 3e-4 -> 3e-5): equal through update 41."""
    nl = str(tmp_path / "nl")
    run_cfg(reduced_nl("NL_bb_s1", nl), nl)
    prm = copy.deepcopy(PARAMS)
    prm["K"] = 10
    cfg = mc.build_config(PROTO, 50, 10501, "MS_base2400", str(tmp_path / "b"), prm)
    cfg["record"]["protocol"]["episodes_per_update"] = EPU
    cfg["record"]["protocol"]["weights_every"] = 5
    cfg["pipeline"]["budgets"] = {"2": 48, "1": 20}
    cfg["pipeline"]["lr_windows"] = {"2": [{"first": 41, "last": 48, "start": 3e-4, "end": 3e-5}],
                                     "1": [{"first": 1, "last": 20, "start": 3e-4, "end": 3e-5}]}
    b = str(tmp_path / "b")
    run_cfg(cfg, b)
    for u in range(5, 41, 5):
        assert same_weights(nl, b, u), u
    assert not same_weights(nl, b, 45)
    assert same_history(nl, b, 41) and not same_history(nl, b, 42)
    assert equal_rows(read_csv(os.path.join(nl, "ms_updates.csv")), read_csv(os.path.join(b, "ms_updates.csv")), 41)
    assert equal_rows(read_csv(os.path.join(nl, "ms_checks_stage2.csv")), read_csv(os.path.join(b, "ms_checks_stage2.csv")), 40)


def test_c_ms4_fixed_budget_stratified_equals_the_rule_mode_global_blocks_through_the_cap(tmp_path):
    """NL_st_s1 against MS_s35a5-like in rule mode with stop and polishing impossible: equal through the first
    landing update (the cap is 40, so update 41 still runs at the base LR)."""
    nl = str(tmp_path / "nl")
    run_cfg(reduced_nl("NL_st_s1", nl, n2=60, const_to=48), nl)
    prm = small_params(2, K=10, block=20, cap=40, land=20, rho=0.0)
    prm["localized_fraction"] = 1e-9
    prm["stages"]["2"].update(eps=PARAMS["stages"]["2"]["eps"], tau=PARAMS["stages"]["2"]["tau"])
    cfg = make_cfg(str(tmp_path / "r"), "MS_s35a5", params=prm, epu=EPU)
    cfg["record"]["protocol"]["weights_every"] = 5
    r = str(tmp_path / "r")
    run_cfg(cfg, r)
    rl = json.load(open(os.path.join(r, "rule_log.json")))["stages"]["2"]
    assert [b["type"] for b in rl["blocks"]] == ["global", "global"] and rl["budget_forced"] and rl["landing"]["first_local"] == 41
    for u in range(5, 41, 5):
        assert same_weights(nl, r, u), u
    assert not same_weights(nl, r, 45)
    assert same_history(nl, r, 41) and not same_history(nl, r, 42)
    skip = ("block_id", "block_type")
    assert equal_rows(read_csv(os.path.join(nl, "ms_updates.csv")), read_csv(os.path.join(r, "ms_updates.csv")), 41, skip=skip)
    assert equal_rows(read_csv(os.path.join(nl, "ms_checks_stage2.csv")), read_csv(os.path.join(r, "ms_checks_stage2.csv")), 40,
                      skip=skip + ("mode",))
    nl_rows = read_csv(os.path.join(nl, "ms_updates.csv"))
    assert any(float(x["alpha"]) == 0.5 for x in nl_rows) and nl_rows[0]["block_type"] == "legacy"


# ---------------------------------------------------------------------------------------------- 5. reporting columns
def test_check_and_freeze_reporting_columns(tmp_path):
    d = str(tmp_path / "st4")
    run_cfg(reduced_nl("NL_st_s4", d), d)
    rows = read_csv(os.path.join(d, "ms_checks_stage2.csv"))
    ms = rms.MSRun(reduced_nl("NL_st_s4", d), d)
    for r in rows:
        assert float(r["g2_0"]) == pytest.approx(70.0)
        assert float(r["gap"]) == pytest.approx(float(r["smoothing"]) + float(r["remainder"]), abs=1e-9)
        assert float(r["gap"]) == pytest.approx(70.0 - float(r["e2_at_0"]), abs=1e-9)
        assert float(r["conc_scale"]) == ms.conc_scale_for(int(r["update"]))
        assert 0.0 <= float(r["R0"]) and float(r["sigma_0"]) > 0.0
    s1 = read_csv(os.path.join(d, "ms_checks_stage1.csv"))
    assert "e_sigma_0" not in s1[0] and "conc_scale" not in s1[0]
    nz = json.load(open(os.path.join(d, "rule_log.json")))["stages"]["2"]["freeze"]["noise"]
    assert nz["conc_scale"] == 4.0 and 0.99 < nz["smoothing_over_gaussian_formula"] < 1.01
    sg = json.load(open(os.path.join(d, "rule_log.json")))["stages"]["2"]["freeze"]["smoothed_game"]
    assert nz["e_sigma_0"] == pytest.approx(sg["smoothed_e_pred_0"], abs=1e-9)    # the same computation as smoothed_share
    man = json.load(open(os.path.join(d, "manifest.json")))
    assert man["conc_scale_schedule"]["scale_last"] == 4.0 and man["fixed_global_sampler"] is True and man["noise_report"] is True


# ---------------------------------------------------------------------------------------------- 6. refusals
def _nl(tmp_path: Path, arm: str = "NL_st_s4") -> Dict[str, Any]:
    return mc.build_config(PROTO, 50, 10501, arm, str(tmp_path), PARAMS)


@pytest.mark.parametrize("mutate", [
    lambda c: c["conc_scale_schedule"].pop("scale_last"),
    lambda c: c["conc_scale_schedule"].update(extra=1),
    lambda c: c["conc_scale_schedule"].update(local_first=300, local_last=200),
    lambda c: c["conc_scale_schedule"].update(local_first=0),
    lambda c: c["conc_scale_schedule"].update(scale_last=0.0),
    lambda c: c["conc_scale_schedule"].update(scale_last=-1.0),
    lambda c: c["conc_scale_schedule"].update(stage=3),
    lambda c: c["conc_scale_schedule"].update(stage=True),
    lambda c: c.update(noise_report="yes"),
    lambda c: c.update(fixed_global_sampler=False),
    lambda c: c.update(fixed_global_sampler=1),
    lambda c: c["rule"].update(enabled=True),
])
def test_configuration_refusals(tmp_path, mutate):
    cfg = _nl(tmp_path)
    rms.validate_config(cfg)                                  # the unmodified config is valid
    mutate(cfg)
    with pytest.raises(ConfigError):
        rms.validate_config(cfg)


def test_fixed_global_sampler_needs_stratified_starts(tmp_path):
    cfg = _nl(tmp_path, "NL_bb_s4")
    cfg["fixed_global_sampler"] = True
    with pytest.raises(ConfigError):
        rms.validate_config(cfg)


# ---------------------------------------------------------------------------------------------- 7. the controller flag
def test_the_legacy_global_sampler_flag_is_a_controller_argument_not_a_rule_field():
    from utils.ms_rule import StageController, StageRule
    assert "legacy_global_sampler" not in StageRule.__dataclass_fields__          # the logged rule parameters of MS-R1 are unchanged
    lin = lambda s, e, f, l, j: s + (e - s) * (j - f) / (l - f)                   # noqa: E731
    leg = StageRule(stage=2, enabled=False, fixed_budget=30, alpha_global=0.5)
    plain, glob = StageController(leg, lin, lambda j: 3e-4), StageController(leg, lin, lambda j: 3e-4, legacy_global_sampler=True)
    assert plain.sampler_setting().block_type == "legacy" and plain.sampler_setting().alpha == 0.0
    st = glob.sampler_setting()
    assert st.block_type == "global" and st.alpha == 0.5 and st.focus is None      # p_PM before the first check
    assert glob.block_label() == (1, "legacy")                                     # the label of the legacy block is kept
    with pytest.raises(ValueError):
        StageController(StageRule(stage=2, enabled=True), lin, legacy_global_sampler=True)
