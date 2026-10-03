"""Tests of the locked v2 T=2 entry point, protocol v1.1 (run/run_v2_T2_locked.py).

Refusals (modified protocol, extra overrides, q not in the protocol), the locked LR schedule at every
update of A and B (against lr_at), the v1.1 gate logic on synthetic values (incl. values exactly at
each threshold), the process-global RNG hardening (seeding, no violation in a reduced-budget run,
detection of a single injected draw from each RNG), and an in-process reduced-budget pipeline whose
output is analysed by tools/v2/confirmation_analysis.py. The reduced-budget runs bypass the CLI on
purpose; the CLI accepts no budget change.
"""

from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run.run_final_dp_br_round3_dense import ConfigError, lr_at  # noqa: E402
from run import run_v2_T2_locked as L  # noqa: E402
from run.run_v2_stagewise import Run  # noqa: E402


@pytest.fixture(scope="module")
def proto():
    return L.load_protocol()


def test_protocol_hash_and_version(proto):
    assert L.sha256_file(L.PROTOCOL_PATH) == L.PROTOCOL_SHA256
    assert proto["version"] == "1.1" and proto["pipeline"]["mode"] == "locked"
    assert L.PROTOCOL_PATH.name == "v2_T2_locked_v1_1.json"


def test_v1_0_protocol_untouched():
    assert L.sha256_file(ROOT / "protocols" / "v2_T2_locked.json") == \
        "cf7b6929adfaef6b712eb01d1a731adc937d0b87fcc37a0a9e68eda9ae92d9f6"


def test_refuses_modified_protocol(tmp_path):
    p = tmp_path / "v2_T2_locked_v1_1.json"
    d = json.load(open(L.PROTOCOL_PATH))
    d["gates"]["G-N"]["all_must_hold"][0]["threshold"] = 0.002
    p.write_text(json.dumps(d, indent=1) + "\n")
    with pytest.raises(ConfigError, match="sha256"):
        L.load_protocol(p, L.PROTOCOL_SHA256)
    p.write_bytes(open(L.PROTOCOL_PATH, "rb").read() + b" ")
    with pytest.raises(ConfigError, match="sha256"):
        L.load_protocol(p, L.PROTOCOL_SHA256)


def test_refuses_mismatching_lock_record(tmp_path):
    lock = tmp_path / "LOCK"
    lock.write_text(json.dumps({"protocol": "protocols/v2_T2_locked.json", "protocol_sha256": "x"}) + "\n"
                    + json.dumps({"protocol": L.PROTOCOL_REL, "protocol_sha256": "0" * 64}) + "\n")
    with pytest.raises(ConfigError, match="LOCK"):
        L.load_protocol(L.PROTOCOL_PATH, L.PROTOCOL_SHA256, lock)
    lock.write_text(json.dumps({"protocol": L.PROTOCOL_REL, "protocol_sha256": L.PROTOCOL_SHA256}) + "\n")
    assert L.load_protocol(L.PROTOCOL_PATH, L.PROTOCOL_SHA256, lock)["version"] == "1.1"


@pytest.mark.parametrize("extra", [["--phase-caps", "3"], ["--lr", "1e-3"], ["--protocol", "x.json"], ["stray"]])
def test_refuses_extra_overrides(extra, tmp_path):
    argv = ["--q", "50", "--seed", "1", "--out-dir", str(tmp_path / "o")] + extra
    with pytest.raises(ConfigError, match="refusing overrides"):
        L.parse_args(argv)
    with pytest.raises(ConfigError, match="refusing overrides"):
        L.main(argv)
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("q", [55, 40])
def test_refuses_unknown_q(proto, tmp_path, q):
    with pytest.raises(ConfigError, match="q_values"):
        L.build_config(proto, q, 1, str(tmp_path))
    with pytest.raises(ConfigError, match="q_values"):
        L.main(["--q", str(q), "--seed", "1", "--out-dir", str(tmp_path / "o")])
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("q", [50, 60])
def test_locked_lr_schedule_every_update(proto, tmp_path, q):
    """Per-update LR of the locked run == the locked schedule, via lr_at, for every update of A and B."""
    run = Run(L.build_config(proto, q, 10501, str(tmp_path)), str(tmp_path))
    sched = run.sched
    assert sched["ab_lr"] == 3e-4 and run.P["phase_caps"]["A"] == 1600 and run.P["phase_caps"]["B"] == 600
    linA = dict(sched, kind="linear", c_start_lr=3e-4, c_end_lr=3e-5, c_local_first=1201, linear_denominator=399)
    linB = dict(sched, kind="linear", c_start_lr=3e-4, c_end_lr=3e-5, c_local_first=1, linear_denominator=599)
    for j in range(1, 1601):
        assert run.lr_for("A", j) == (lr_at(sched, "A", j) if j <= 1200 else lr_at(linA, "C", j))
    for j in range(1, 601):
        assert run.lr_for("B", j) == lr_at(linB, "C", j)


# ------------------------------------------------------------------ gate logic (synthetic values)
BASE = {"eta_final": 0.001, "eta_dev": 0.001, "rmse": 0.02, "tail": 0.01, "gmax_final": 0.002, "gmax_dev": 0.002, "s1": 0.05}


def test_gates_at_exact_thresholds_pass(proto):
    """Every '<=' is inclusive: values exactly at each threshold pass (G-N gaps chosen exactly representable)."""
    v = dict(BASE, eta_final=0.005, eta_dev=0.005, rmse=0.05, tail=0.02, gmax_final=0.01, gmax_dev=0.01, s1=0.10)
    V = L.verdicts(v, proto)
    assert V["G-A"]["pass"] and V["G-F"]["pass"] and V["G-N"]["pass"] and V["run_pass"] and V["S1"]["pass"]
    assert abs(0.001 - 0.0) == 0.001
    V = L.verdicts(dict(BASE, eta_final=0.0, eta_dev=0.001, gmax_final=0.0, gmax_dev=0.001), proto)
    assert V["G-N"]["pass"] and V["run_pass"]


@pytest.mark.parametrize("key, bad, gate", [("eta_final", 0.0050000001, "G-A"), ("rmse", 0.0500001, "G-A"),
                                            ("tail", 0.0200001, "G-A"), ("gmax_final", 0.0100001, "G-F")])
def test_gates_just_above_threshold_fail(proto, key, bad, gate):
    v = dict(BASE, **{key: bad})
    if key == "eta_final":
        v["eta_dev"] = bad
    if key == "gmax_final":
        v["gmax_dev"] = bad
    V = L.verdicts(v, proto)
    assert not V[gate]["pass"] and not V["run_pass"]


@pytest.mark.parametrize("which", ["eta", "gmax"])
def test_gate_N_fails_on_refinement_gap(proto, which):
    v = dict(BASE, **{f"{which}_dev": BASE[f"{which}_final"] + 0.0011})
    V = L.verdicts(v, proto)
    assert not V["G-N"]["pass"] and not V["run_pass"] and V["G-A"]["pass"] and V["G-F"]["pass"]
    assert V["outcome"] == "fail_G-N"


def test_run_pass_is_GA_GF_GN_only(proto):
    V = L.verdicts(dict(BASE, s1=0.5), proto)               # S1 fails -> run pass unaffected; v1.0 outcome fails
    assert V["run_pass"] and not V["S1"]["pass"] and not V["v1_0_outcome"]["run_pass_v1_0"] and V["outcome"] == "pass"
    V = L.verdicts(dict(BASE, rmse=0.06), proto)            # G-A fails -> stage-2 failure
    assert not V["run_pass"] and V["outcome"] == "stage2_failure" and V["S1"]["pass"]
    V = L.verdicts(dict(BASE, gmax_final=0.02, gmax_dev=0.02), proto)
    assert not V["run_pass"] and V["outcome"] == "fail_G-F"
    V = L.verdicts(BASE, proto)
    assert V["run_pass"] and V["v1_0_outcome"]["run_pass_v1_0"]


# ------------------------------------------------------------------ global RNG hardening
def test_seed_globals_matches_fresh_seeding():
    torch.rand(3), np.random.rand(3), random.random()
    L.seed_globals(10501)
    a = L.digests(L.global_states())
    torch.manual_seed(10501)
    np.random.seed(10501)
    random.seed(10501)
    assert L.digests(L.global_states()) == a


def _small_cfg(proto, out):
    cfg = L.build_config(proto, 50, 10501, out)
    cfg["budget_overrides"] = {"phase_caps": {"A": 6, "B": 4, "C": 1}, "warmup": 3, "stability_every": 2, "verifier_timeout": 3}
    cfg["lr_decay"] = [{"phase": "A", "start_lr": 3e-4, "end_lr": 3e-5, "local_first": 4, "local_last": 6},
                       {"phase": "B", "start_lr": 3e-4, "end_lr": 3e-5, "local_first": 1, "local_last": 4}]
    return cfg


@pytest.fixture(scope="module")
def smoke(proto, tmp_path_factory):
    """Reduced-budget in-process pipeline laid out as <root>/q50/seed10501."""
    root = tmp_path_factory.mktemp("root")
    out = str(root / "q50" / "seed10501")
    os.makedirs(out)
    code = L.run_pipeline(_small_cfg(proto, out), proto, L.PROTOCOL_SHA256, out, "pytest", band_step=2.0)
    return root, out, code


def test_pipeline_smoke_no_violation(smoke):
    root, out, code = smoke
    assert code == 0
    for f in ("manifest.json", "gates.json", "state_end_A.pt", "state_end_B.pt", "v2_checkpoints_A.csv", "v2_checkpoints_B.csv",
              "drift_test.json", "induced_band.json", "band_sweep.npz", "train_history.json", "v2_updates.csv",
              "gateA_final.npz", "gateA_development.npz", "final_final.npz", "final_development.npz"):
        assert os.path.exists(os.path.join(out, f)), f
    g = json.load(open(os.path.join(out, "gates.json")))
    assert g["protocol_version"] == "1.1" and g["global_rng"]["status"] == "ok" and g["global_rng"]["violations"] == []
    assert set(g) >= {"G-A", "G-F", "G-N", "S1", "v1_0_outcome", "run_pass", "outcome", "metric_values"}
    assert [c["metric"] for c in g["G-F"]["criteria"]] == ["Gmax_full_over_dw"]
    man = json.load(open(os.path.join(out, "manifest.json")))
    gr = man["global_rng"]
    assert man["protocol_version"] == "1.1" and gr["seeding"]["seeds"] == {k: 10501 for k in L.GLOBAL_RNGS}
    for pt in ("seeding", "after_run_construction", "reference_before_first_A_update") + L.ASSERT_POINTS:
        assert set(gr[pt]["digests"]) == set(L.GLOBAL_RNGS), pt
    for pt in L.ASSERT_POINTS:
        assert gr[pt]["digests"] == gr["reference_before_first_A_update"]["digests"], pt
    h = json.load(open(os.path.join(out, "train_history.json")))["history"]
    assert [x["phase"] for x in h] == ["A"] * 6 + ["B"] * 4


def test_confirmation_analysis_on_smoke(smoke, tmp_path):
    root, out, code = smoke
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "v2" / "confirmation_analysis.py"), "--root", str(root),
                        "--seeds", "10501", "10501", "--out", str(tmp_path / "an")], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    import pandas as pd
    ag = pd.read_csv(tmp_path / "an" / "agreement.csv")
    assert len(ag) == 1 and bool(ag.all_agree.iloc[0])
    assert float(ag.filter(like="absdiff_").to_numpy().max()) == 0.0
    pr = pd.read_csv(tmp_path / "an" / "per_run.csv")
    assert set(pr.status) == {"completed", "missing"}          # q60 seed10501 is absent in the smoke root


@pytest.mark.parametrize("rng_name", ["torch_global", "numpy_global", "python_random"])
def test_injected_global_draw_detected(proto, tmp_path, monkeypatch, rng_name):
    """One draw from one global RNG after the reference point -> recorded in gates.json, exit code 5."""
    draw = {"torch_global": lambda: torch.rand(1), "numpy_global": lambda: np.random.rand(),
            "python_random": lambda: random.random()}[rng_name]
    orig = L.smoothed_share
    calls = {"n": 0}

    def injected(*a, **k):
        if calls["n"] == 0:
            draw()
        calls["n"] += 1
        return orig(*a, **k)
    monkeypatch.setattr(L, "smoothed_share", injected)       # runs between the end of A and the after-G-A check
    out = str(tmp_path / "run")
    os.makedirs(out)
    code = L.run_pipeline(_small_cfg(proto, out), proto, L.PROTOCOL_SHA256, out, "pytest", band_step=2.0)
    assert code == L.RNG_VIOLATION_EXIT
    g = json.load(open(os.path.join(out, "gates.json")))
    assert g["global_rng"]["status"] == "violation" and g["run_pass"] is False and g["outcome"] == "global_rng_violation"
    assert {v["rng"] for v in g["global_rng"]["violations"]} == {rng_name}
    assert g["global_rng"]["violations"][0]["point"] == "after_G-A"
    st = json.load(open(os.path.join(out, "status.json")))
    assert st["state"] == "done" and st["exit_code"] == L.RNG_VIOLATION_EXIT
    assert os.path.exists(os.path.join(out, "state_end_B.pt"))   # the run finished and wrote all outputs
