"""Tests of the locked v2 T=2 entry point (run/run_v2_T2_locked.py).

Refusals (modified protocol, extra overrides), the locked LR schedule at every update of A and B
(against lr_at), and an in-process smoke of the whole pipeline with reduced budgets (the smoke
bypasses the CLI on purpose; the CLI itself never accepts budget changes).
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run.run_final_dp_br_round3_dense import ConfigError, lr_at  # noqa: E402
from run import run_v2_T2_locked as L  # noqa: E402
from run.run_v2_stagewise import Run  # noqa: E402


@pytest.fixture(scope="module")
def proto():
    return L.load_protocol()


def test_protocol_hash_matches_lock(proto):
    assert L.sha256_file(L.PROTOCOL_PATH) == L.PROTOCOL_SHA256
    assert proto["pipeline"]["mode"] == "locked"


def test_refuses_modified_protocol(tmp_path):
    p = tmp_path / "v2_T2_locked.json"
    d = json.load(open(L.PROTOCOL_PATH))
    d["gates"]["G-A"]["all_must_hold"][0]["threshold"] = 0.006
    p.write_text(json.dumps(d, indent=1) + "\n")
    with pytest.raises(ConfigError, match="sha256"):
        L.load_protocol(p, L.PROTOCOL_SHA256)
    p.write_bytes(open(L.PROTOCOL_PATH, "rb").read() + b" ")      # one extra byte
    with pytest.raises(ConfigError, match="sha256"):
        L.load_protocol(p, L.PROTOCOL_SHA256)


@pytest.mark.parametrize("extra", [["--phase-caps", "3"], ["--lr", "1e-3"], ["--protocol", "x.json"], ["stray"]])
def test_refuses_extra_overrides(extra, tmp_path):
    argv = ["--q", "50", "--seed", "1", "--out-dir", str(tmp_path / "o")] + extra
    with pytest.raises(ConfigError, match="refusing overrides"):
        L.parse_args(argv)
    with pytest.raises(ConfigError, match="refusing overrides"):
        L.main(argv)
    assert not (tmp_path / "o").exists()


def test_refuses_unknown_q(proto, tmp_path):
    with pytest.raises(ConfigError, match="q_values"):
        L.build_config(proto, 55, 1, str(tmp_path))


@pytest.mark.parametrize("q", [50, 60])
def test_locked_lr_schedule_every_update(proto, tmp_path, q):
    """Per-update LR of the locked run == the locked schedule, via lr_at, for every update of A and B."""
    cfg = L.build_config(proto, q, 10501, str(tmp_path))
    run = Run(cfg, str(tmp_path))
    sched = run.sched
    assert sched["ab_lr"] == 3e-4 and run.P["phase_caps"]["A"] == 1600 and run.P["phase_caps"]["B"] == 600
    linA = dict(sched, kind="linear", c_start_lr=3e-4, c_end_lr=3e-5, c_local_first=1201, linear_denominator=399)
    linB = dict(sched, kind="linear", c_start_lr=3e-4, c_end_lr=3e-5, c_local_first=1, linear_denominator=599)
    for j in range(1, 1601):
        want = lr_at(sched, "A", j) if j <= 1200 else lr_at(linA, "C", j)
        assert run.lr_for("A", j) == want
    for j in range(1, 601):
        assert run.lr_for("B", j) == lr_at(linB, "C", j)
    assert run.lr_for("A", 1200) == 3e-4 and run.lr_for("A", 1201) == 3e-4 and run.lr_for("B", 1) == 3e-4
    assert abs(run.lr_for("A", 1600) - 3e-5) < 1e-15 and abs(run.lr_for("B", 600) - 3e-5) < 1e-15


def test_pipeline_smoke_in_process(proto, tmp_path):
    """Whole pipeline with reduced budgets: outputs, gates, frozen snapshot, LR at every update."""
    out = str(tmp_path / "run")
    os.makedirs(out)
    cfg = L.build_config(proto, 50, 10501, out)
    cfg["budget_overrides"] = {"phase_caps": {"A": 6, "B": 4, "C": 1}, "warmup": 3, "stability_every": 2,
                               "verifier_timeout": 3}
    cfg["lr_decay"] = [{"phase": "A", "start_lr": 3e-4, "end_lr": 3e-5, "local_first": 4, "local_last": 6},
                       {"phase": "B", "start_lr": 3e-4, "end_lr": 3e-5, "local_first": 1, "local_last": 4}]
    run = Run(cfg, out)
    assert L.execute_locked(run, cfg, proto, L.PROTOCOL_SHA256, out, "pytest", band_step=2.0) == 0
    for f in ("manifest.json", "gates.json", "state_end_A.pt", "state_end_B.pt", "v2_checkpoints_A.csv", "v2_checkpoints_B.csv", "drift_test.json",
              "induced_band.json", "train_history.json", "v2_updates.csv", "gateA_final.npz", "final_final.npz"):
        assert os.path.exists(os.path.join(out, f)), f
    g = json.load(open(os.path.join(out, "gates.json")))
    assert set(g) >= {"G-A", "G-F", "run_pass", "outcome", "reported"}
    assert [c["metric"] for c in g["G-A"]["criteria"]] == ["eta_T_over_dw", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean_over_g2_0"]
    assert [c["metric"] for c in g["G-F"]["criteria"]] == ["Gmax_full_over_dw", "stage1_rel_err_abs"]
    assert g["reported"]["drift_test_pass"] is True
    man = json.load(open(os.path.join(out, "manifest.json")))
    assert man["locked_protocol"]["sha256"] == L.PROTOCOL_SHA256 and "clean_tree" in man
    h = json.load(open(os.path.join(out, "train_history.json")))["history"]
    assert [x["phase"] for x in h] == ["A"] * 6 + ["B"] * 4
    for x in h:
        assert x["actor_lr"] == run.lr_for(x["phase"], x["local"]) == x["critic_lr"]
