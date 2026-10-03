"""Tests of the locked v2 T=2 entry point, protocol v2.0 (run/run_v2_T2_locked.py).

Ported from the v1.1 suite (tests/test_v2_locked.py at 431474d; every test kept): refusals (modified
protocol, extra overrides, q not in the protocol), the locked LR schedule at every update of A and B
(against lr_at), the global-RNG hardening (seeding, no violation in a reduced-budget run, detection of a
single injected draw from each RNG). New for v2.0: gate G-S and the run-pass definition on synthetic
values (including values exactly at each threshold), refusal of a changed continuation-table rule, the
table written by the entry point against ``utils.v2_continuation`` called directly on the frozen
snapshot, no RNG movement during the build, the v1.1 files and analysis script byte-identical, the
protocol generator reproducing the committed JSON, and an in-process reduced-budget pipeline whose output
is analysed by tools/v2/confirmation_analysis_v2_0.py. The reduced-budget runs bypass the CLI on purpose;
the CLI accepts no budget change.
"""

from __future__ import annotations

import hashlib
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
sys.path.insert(0, str(ROOT / "tools" / "v2"))

from agents.ppo_curriculum import BetaActor, PPOConfig  # noqa: E402
from envs.curriculum_env import GameSpec  # noqa: E402
from run.run_final_dp_br_round3_dense import ConfigError, lr_at  # noqa: E402
from run import run_v2_T2_locked as L  # noqa: E402
from run import run_v2_stagewise as stagewise  # noqa: E402
from run.run_v2_stagewise import Run  # noqa: E402
from utils import v2_continuation as vc  # noqa: E402

import confirmation_analysis_v2_0 as A  # noqa: E402
import make_locked_protocol_v2_0 as G  # noqa: E402
import v2_0_pass_probability as PP  # noqa: E402

V10_SHA = "cf7b6929adfaef6b712eb01d1a731adc937d0b87fcc37a0a9e68eda9ae92d9f6"
V11_SHA = "21d85983f2a2bebc0998e99fcf665c2c729f060996c529fa3162b3d62222a40f"
V11_MD_SHA = "2ac714107d8242d3356fc6b49753f0d75a6871b2e7fd172106df2d6db53cfa3a"
V11_ANALYSIS_SHA = "f03ab182cfe1301ffdce0180794dc23e8a3f0de77d3ee48a2ba840109a92f025"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def proto():
    return L.load_protocol()


# ------------------------------------------------------------------ hash, version, refusals
def test_protocol_hash_and_version(proto):
    assert L.sha256_file(L.PROTOCOL_PATH) == L.PROTOCOL_SHA256
    assert proto["version"] == "2.0" and proto["pipeline"]["mode"] == "locked"
    assert L.PROTOCOL_PATH.name == "v2_T2_locked_v2_0.json" and L.PROTOCOL_REL == "protocols/v2_T2_locked_v2_0.json"
    assert proto["pipeline"]["flags"]["continuation_value_mode"] == "expected"
    assert proto["confirmation"]["seed_block"] == list(range(30501, 30521))


def test_earlier_protocol_files_untouched():
    """v1.0 and v1.1 (JSON and Markdown) stay byte-identical; so does the v1.1 analysis script."""
    assert _sha(ROOT / "protocols" / "v2_T2_locked.json") == V10_SHA
    assert _sha(ROOT / "protocols" / "v2_T2_locked_v1_1.json") == V11_SHA
    assert _sha(ROOT / "protocols" / "v2_T2_locked_v1_1.md") == V11_MD_SHA
    assert _sha(ROOT / "tools" / "v2" / "confirmation_analysis.py") == V11_ANALYSIS_SHA


def test_lock_record_matches_when_present(proto):
    """Once the v2.0 record is appended to protocols/LOCK its hashes must be the files' hashes."""
    mine = [r for r in L.lock_records() if r.get("protocol") == L.PROTOCOL_REL]
    if not mine:
        pytest.skip("v2.0 LOCK record not appended yet")
    r = mine[-1]
    assert r["protocol_sha256"] == L.sha256_file(L.PROTOCOL_PATH)
    assert r["protocol_md_sha256"] == _sha(ROOT / "protocols" / "v2_T2_locked_v2_0.md")
    assert r["entry_point_sha256_at_lock"] == _sha(ROOT / "run" / "run_v2_T2_locked.py")
    assert r["analysis_script_sha256"] == _sha(ROOT / "tools" / "v2" / "confirmation_analysis_v2_0.py")
    assert r["continuation_module_sha256"] == _sha(ROOT / "utils" / "v2_continuation.py")
    assert proto["pipeline"]["continuation_table"]["module_sha256_at_lock"] == r["continuation_module_sha256"]
    assert str(r["protocol_version"]) == "2.0" and r["supersedes"]["protocol_version"] == "1.1"


def test_refuses_modified_protocol(tmp_path):
    p = tmp_path / "v2_T2_locked_v2_0.json"
    d = json.load(open(L.PROTOCOL_PATH))
    d["gates"]["G-N"]["all_must_hold"][0]["threshold"] = 0.002
    p.write_text(json.dumps(d, indent=1) + "\n")
    with pytest.raises(ConfigError, match="sha256"):
        L.load_protocol(p, L.PROTOCOL_SHA256)
    d = json.load(open(L.PROTOCOL_PATH))
    d["gates"]["G-S"]["all_must_hold"][0]["threshold"] = 0.10
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
    assert L.load_protocol(L.PROTOCOL_PATH, L.PROTOCOL_SHA256, lock)["version"] == "2.0"


@pytest.mark.parametrize("edit", [
    lambda d: d["pipeline"]["continuation_table"]["rule"].update(panel_width=0.5),
    lambda d: d["pipeline"]["continuation_table"]["rule"].update(nodes_per_panel=12),
    lambda d: d["pipeline"]["continuation_table"]["rule"]["y_grid"].update(step=0.1),
    lambda d: d["pipeline"]["continuation_table"]["rule"].update(dtype="float32"),
    lambda d: d["pipeline"]["continuation_table"]["rule"].update(edge_tolerance=1e-6),
    lambda d: d["pipeline"]["continuation_table"]["rule"].update(max_points_per_chunk=1 << 18),
    lambda d: d["pipeline"]["continuation_table"].update(stage=1),
    lambda d: d["pipeline"]["flags"].update(continuation_value_mode="sampled"),
    lambda d: d["pipeline"].pop("continuation_table"),
], ids=["panel_width", "nodes_per_panel", "y_step", "dtype", "edge_tolerance", "max_points_per_chunk", "stage",
       "value_mode_sampled", "table_missing"])
def test_refuses_changed_table_rule(edit, tmp_path):
    """A changed rule is refused even when the file's hash is (re)locked to the changed file: the rule
    check is independent of the hash check."""
    d = json.load(open(L.PROTOCOL_PATH))
    edit(d)
    p = tmp_path / "v2_T2_locked_v2_0.json"
    p.write_text(json.dumps(d, indent=1) + "\n")
    with pytest.raises(ConfigError, match="continuation"):
        L.load_protocol(p, L.sha256_file(p), tmp_path / "no_LOCK")


@pytest.mark.parametrize("target, name, value", [(vc, "DEFAULT_NODES_PER_PANEL", 12), (vc, "DEFAULT_PANEL_WIDTH", 0.5),
                                                 (vc, "DEFAULT_STEP", 0.1), (stagewise, "CONT_TABLE_STEP", 0.1),
                                                 (vc, "_EDGE_TOL", 1e-6), (vc, "_MAX_POINTS_PER_CHUNK", 1 << 18)])
def test_refuses_code_with_other_table_rule(monkeypatch, target, name, value):
    """The code's own rule (module defaults, pipeline constant) must equal the locked one."""
    monkeypatch.setattr(target, name, value)
    with pytest.raises(ConfigError, match="implemented continuation-table rule"):
        L.load_protocol()


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
    assert run.cont_mode == "expected" and run.flags == {k: proto["pipeline"]["flags"][k] for k in stagewise.FLAG_KEYS}
    linA = dict(sched, kind="linear", c_start_lr=3e-4, c_end_lr=3e-5, c_local_first=1201, linear_denominator=399)
    linB = dict(sched, kind="linear", c_start_lr=3e-4, c_end_lr=3e-5, c_local_first=1, linear_denominator=599)
    for j in range(1, 1601):
        assert run.lr_for("A", j) == (lr_at(sched, "A", j) if j <= 1200 else lr_at(linA, "C", j))
    for j in range(1, 601):
        assert run.lr_for("B", j) == lr_at(linB, "C", j)


# ------------------------------------------------------------------ gate logic (synthetic values)
BASE = {"eta_final": 0.001, "eta_dev": 0.001, "rmse": 0.02, "tail": 0.01, "gmax_final": 0.002, "gmax_dev": 0.002, "s1": 0.03}


def test_gates_at_exact_thresholds_pass(proto):
    """Every '<=' is inclusive: values exactly at each threshold pass (G-N gaps chosen exactly representable)."""
    v = dict(BASE, eta_final=0.005, eta_dev=0.005, rmse=0.05, tail=0.02, gmax_final=0.01, gmax_dev=0.01, s1=0.05)
    V = L.verdicts(v, proto)
    assert V["G-A"]["pass"] and V["G-F"]["pass"] and V["G-N"]["pass"] and V["G-S"]["pass"] and V["run_pass"]
    assert V["S1"]["pass"] and V["v1_1_outcome"]["run_pass_v1_1"] and V["v1_0_outcome"]["run_pass_v1_0"]
    V = L.verdicts(dict(BASE, eta_final=0.0, eta_dev=0.001, gmax_final=0.0, gmax_dev=0.001), proto)
    assert V["G-N"]["pass"] and V["run_pass"]
    # S1 at its own threshold 0.10 (inclusive) and just above it; G-S fails at both values, S1 and the v1.0 outcome decide on 0.10
    V = L.verdicts(dict(BASE, s1=0.10), proto)
    assert V["S1"]["pass"] and V["v1_0_outcome"]["run_pass_v1_0"] and not V["G-S"]["pass"] and not V["run_pass"]
    V = L.verdicts(dict(BASE, s1=float(np.nextafter(0.10, 1.0))), proto)
    assert not V["S1"]["pass"] and not V["v1_0_outcome"]["run_pass_v1_0"] and V["v1_1_outcome"]["run_pass_v1_1"]


@pytest.mark.parametrize("key, bad, gate", [("eta_final", 0.0050000001, "G-A"), ("rmse", 0.0500001, "G-A"),
                                            ("tail", 0.0200001, "G-A"), ("gmax_final", 0.0100001, "G-F"),
                                            ("s1", 0.0500001, "G-S")])
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
    assert not V["G-N"]["pass"] and not V["run_pass"] and V["G-A"]["pass"] and V["G-F"]["pass"] and V["G-S"]["pass"]
    assert V["outcome"] == "fail_G-N"


def test_run_pass_is_GA_GF_GN_GS_only(proto):
    """Run pass = G-A and G-F and G-N and G-S; S1 at 0.10, the v1.1 and the v1.0 outcomes do not enter it."""
    V = L.verdicts(BASE, proto)
    assert V["run_pass"] and V["outcome"] == "pass" and V["v1_1_outcome"]["run_pass_v1_1"] and V["v1_0_outcome"]["run_pass_v1_0"]
    # G-S fails alone (0.07 passes S1 at 0.10, the v1.1 outcome and the v1.0 outcome): the run fails
    V = L.verdicts(dict(BASE, s1=0.07), proto)
    assert not V["run_pass"] and V["outcome"] == "fail_G-S" and not V["G-S"]["pass"]
    assert V["S1"]["pass"] and V["v1_1_outcome"]["run_pass_v1_1"] and V["v1_0_outcome"]["run_pass_v1_0"]
    # S1 at 0.10 also fails: run fails, v1.1 outcome still passes, v1.0 outcome fails
    V = L.verdicts(dict(BASE, s1=0.5), proto)
    assert not V["run_pass"] and not V["S1"]["pass"] and V["v1_1_outcome"]["run_pass_v1_1"] and not V["v1_0_outcome"]["run_pass_v1_0"]
    V = L.verdicts(dict(BASE, rmse=0.06), proto)            # G-A fails -> stage-2 failure
    assert not V["run_pass"] and V["outcome"] == "stage2_failure" and V["S1"]["pass"] and not V["v1_1_outcome"]["run_pass_v1_1"]
    V = L.verdicts(dict(BASE, gmax_final=0.02, gmax_dev=0.02), proto)
    assert not V["run_pass"] and V["outcome"] == "fail_G-F"
    # several gates fail: outcome names the first in the order G-A, G-F, G-N, G-S
    V = L.verdicts(dict(BASE, gmax_final=0.02, gmax_dev=0.02, s1=0.5), proto)
    assert V["outcome"] == "fail_G-F"
    assert [c["metric"] for c in V["G-S"]["criteria"]] == ["stage1_rel_err_abs"] and V["G-S"]["criteria"][0]["threshold"] == 0.05


def test_verdicts_of_the_v1_1_protocol_are_unchanged():
    """tools/v2/refine_analysis.py calls verdicts() with the v1.1 protocol: no G-S, run pass = G-A, G-F, G-N."""
    p11 = json.load(open(ROOT / "protocols" / "v2_T2_locked_v1_1.json"))
    V = L.verdicts(dict(BASE, s1=0.5), p11)
    assert "G-S" not in V and V["run_pass"] and V["outcome"] == "pass" and not V["S1"]["pass"]
    assert not V["v1_0_outcome"]["run_pass_v1_0"]


def test_analysis_decide_agrees_with_entry_point_verdicts(proto):
    """The analysis script's independent decide() equals the entry point's verdicts() over edge and random values."""
    rng = np.random.default_rng(5)
    cases = [dict(BASE)]
    for key, edge in (("eta_final", 0.005), ("rmse", 0.05), ("tail", 0.02), ("gmax_final", 0.01), ("s1", 0.05), ("s1", 0.10)):
        for x in (edge, np.nextafter(edge, 1.0), np.nextafter(edge, 0.0)):
            cases.append(dict(BASE, **{key: float(x)}))
    for _ in range(300):
        cases.append({"eta_final": float(rng.uniform(0, 0.008)), "eta_dev": float(rng.uniform(0, 0.008)), "rmse": float(rng.uniform(0, 0.08)),
                      "tail": float(rng.uniform(0, 0.03)), "gmax_final": float(rng.uniform(0, 0.015)), "gmax_dev": float(rng.uniform(0, 0.015)),
                      "s1": float(rng.uniform(0, 0.15))})
    for v in cases:
        V, D = L.verdicts(v, proto), A.decide(v, proto)
        assert (V["G-A"]["pass"], V["G-F"]["pass"], V["G-N"]["pass"], V["G-S"]["pass"], V["S1"]["pass"]) == \
            (D["G-A"], D["G-F"], D["G-N"], D["G-S"], D["S1"])
        assert V["run_pass"] == D["gates_pass"] and V["v1_1_outcome"]["run_pass_v1_1"] == D["v1_1_run_pass"]
        assert V["v1_0_outcome"]["run_pass_v1_0"] == D["run_pass_v1_0"] and V["v1_0_outcome"]["G-F_v1_0"] == D["G-F_v1_0"]
        assert A.outcome_of(D, "completed") == V["outcome"]


def test_normal_model_agrees_between_the_two_tools():
    """Analysis-script normal model == tools/v2/v2_0_pass_probability.py on the R1 B_expcont data (0.9959, 0.9995)."""
    import pandas as pd
    d = pd.read_csv(ROOT / "results" / "v2_refine" / "analysis" / "stage1_per_run.csv")
    d = d[d.arm == "B_expcont"]
    want = {50: 0.99587, 60: 0.99953}
    for q in (50, 60):
        x = d[d.q == q].stage1_rel_err_signed.astype(float).to_numpy()
        nm = A.normal_model(x, 0.05)
        assert nm["P_ge_k_of_n"] == pytest.approx(PP.p_rule(PP.p_pass(float(x.mean()), float(x.std(ddof=1)))), rel=0, abs=1e-15)
        assert nm["P_ge_k_of_n"] == pytest.approx(want[q], abs=5e-5)


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


def test_pipeline_smoke_no_violation(smoke, proto):
    root, out, code = smoke
    assert code == 0
    for f in ("manifest.json", "gates.json", "state_end_A.pt", "state_end_B.pt", "v2_checkpoints_A.csv", "v2_checkpoints_B.csv",
              "drift_test.json", "induced_band.json", "band_sweep.npz", "train_history.json", "v2_updates.csv",
              "gateA_final.npz", "gateA_development.npz", "final_final.npz", "final_development.npz", "continuation_table.npz"):
        assert os.path.exists(os.path.join(out, f)), f
    g = json.load(open(os.path.join(out, "gates.json")))
    assert g["protocol_version"] == "2.0" and g["global_rng"]["status"] == "ok" and g["global_rng"]["violations"] == []
    assert set(g) >= {"G-A", "G-F", "G-N", "G-S", "S1", "v1_1_outcome", "v1_0_outcome", "run_pass", "outcome", "metric_values",
                      "continuation_table"}
    assert [c["metric"] for c in g["G-F"]["criteria"]] == ["Gmax_full_over_dw"]
    assert [c["metric"] for c in g["G-S"]["criteria"]] == ["stage1_rel_err_abs"] and g["G-S"]["criteria"][0]["threshold"] == 0.05
    assert g["S1"]["threshold"] == 0.10 and "G-S" in g["dev_tier_values"] and "S1" in g["dev_tier_values"]
    assert g["global_rng"]["assertion_points"] == list(L.ASSERT_POINTS) + ["table_build"]
    assert g["run_pass"] == (g["G-A"]["pass"] and g["G-F"]["pass"] and g["G-N"]["pass"] and g["G-S"]["pass"])
    man = json.load(open(os.path.join(out, "manifest.json")))
    gr = man["global_rng"]
    assert man["protocol_version"] == "2.0" and gr["seeding"]["seeds"] == {k: 10501 for k in L.GLOBAL_RNGS}
    assert man["continuation_value_mode"] == "expected" and man["flags"] == {k: proto["pipeline"]["flags"][k] for k in stagewise.FLAG_KEYS}
    for pt in ("seeding", "after_run_construction", "reference_before_first_A_update", "before_B_entry") + L.ASSERT_POINTS:
        assert set(gr[pt]["digests"]) == set(L.GLOBAL_RNGS), pt
    for pt in ("before_B_entry", "before_first_B_update") + L.ASSERT_POINTS:
        assert gr[pt]["digests"] == gr["reference_before_first_A_update"]["digests"], pt
    h = json.load(open(os.path.join(out, "train_history.json")))["history"]
    assert [x["phase"] for x in h] == ["A"] * 6 + ["B"] * 4
    cfg = json.load(open(os.path.join(root, "q50", "seed10501", "manifest.json")))["input_config"]
    assert cfg["continuation_value_mode"] == "expected" and set(cfg["flags"]) == set(stagewise.FLAG_KEYS)


def test_table_record_and_file(smoke, proto):
    """continuation_table.npz: arrays y_grid and values; SHA-256, rule and build time recorded in manifest and gates."""
    root, out, code = smoke
    g = json.load(open(os.path.join(out, "gates.json")))
    man = json.load(open(os.path.join(out, "manifest.json")))
    rec = g["continuation_table"]
    assert man["continuation_table"] == rec
    assert rec["file"] == "continuation_table.npz" and rec["npz_sha256"] == _sha(Path(out) / "continuation_table.npz")
    assert rec["build_seconds"] > 0 and rec["rng_unchanged_by_build"] == {"training": True, **{k: True for k in L.GLOBAL_RNGS}}
    rule = proto["pipeline"]["continuation_table"]["rule"]
    assert rec["rule"]["panel_width"] == rule["panel_width"] and rec["rule"]["nodes_per_panel"] == rule["nodes_per_panel"]
    assert rec["rule"]["step_requested"] == rule["y_grid"]["step"] and rec["rule"]["n_y"] == rule["y_grid"]["n_nodes"]
    assert rec["rule"]["n_panels"] == rule["n_panels"]["50"] and rec["rule"]["n_nodes"] == rule["n_nodes"]["50"]
    z = np.load(os.path.join(out, "continuation_table.npz"))
    assert set(z.files) == {"y_grid", "values"} and z["y_grid"].dtype == np.float64 and z["values"].dtype == np.float64
    assert z["y_grid"].shape == z["values"].shape == (rule["y_grid"]["n_nodes"],)


def test_table_equals_direct_build_on_the_frozen_snapshot(smoke, proto):
    """The table the entry point wrote is bit-identical to utils.v2_continuation called directly on the frozen snapshot."""
    root, out, code = smoke
    state = torch.load(os.path.join(out, "state_end_B.pt"), map_location="cpu", weights_only=False)
    cfg = PPOConfig()
    actor = BetaActor(cfg.hidden, cfg.c_min, cfg.mu_clamp, torch.Generator().manual_seed(0))
    actor.load_state_dict(state["agent"]["frozen"])
    actor.eval()
    spec = GameSpec(**proto["records"]["50"]["game"])
    direct = vc.build_continuation_table(actor, spec, stage=2, step=0.05)
    z = np.load(os.path.join(out, "continuation_table.npz"))
    assert np.array_equal(z["y_grid"], direct.y_grid) and np.array_equal(z["values"], direct.values)


def test_table_npz_is_deterministic(tmp_path):
    """Equal tables give equal file bytes (fixed zip member dates), different tables different bytes, and
    np.load reads the arrays back exactly."""
    y = np.linspace(-100.0, 100.0, 4001)
    tab = vc.ContinuationTable(y, np.sin(y / 17.0) * 3.0 + 2.0)
    s1 = L.write_table_npz(str(tmp_path / "a.npz"), tab)
    s2 = L.write_table_npz(str(tmp_path / "b.npz"), tab)
    assert s1 == s2 and (tmp_path / "a.npz").read_bytes() == (tmp_path / "b.npz").read_bytes()
    import zipfile
    with zipfile.ZipFile(tmp_path / "a.npz") as z_:
        assert {i.date_time for i in z_.infolist()} == {(1980, 1, 1, 0, 0, 0)}
    s3 = L.write_table_npz(str(tmp_path / "c.npz"), vc.ContinuationTable(y, tab.values + 1e-12))
    assert s3 != s1
    z = np.load(tmp_path / "a.npz")
    assert np.array_equal(z["y_grid"], tab.y_grid) and np.array_equal(z["values"], tab.values)


def test_table_build_moves_no_rng(proto, tmp_path):
    """Building the table from the frozen snapshot draws from no RNG, and the digests can see a draw."""
    run = Run(L.build_config(proto, 50, 10501, str(tmp_path)), str(tmp_path))
    run.agent.freeze_stage2_snapshot()
    before = L.rng_snapshot(run)
    vc.build_continuation_table(run.agent.frozen, run.spec, stage=2, step=stagewise.CONT_TABLE_STEP)
    assert L.rng_snapshot(run) == before
    run.rngs["env"].random()
    assert L.rng_snapshot(run)["training"] != before["training"]
    before = L.rng_snapshot(run)
    run.agent.rng_mb.random()
    assert L.rng_snapshot(run)["training"] != before["training"]
    before = L.rng_snapshot(run)
    run.torch_gen.manual_seed(1)
    assert L.rng_snapshot(run)["training"] != before["training"]
    before = L.rng_snapshot(run)
    torch.rand(1)
    assert L.rng_snapshot(run)["torch_global"] != before["torch_global"]
    before = L.rng_snapshot(run)
    np.random.rand()
    assert L.rng_snapshot(run)["numpy_global"] != before["numpy_global"]
    before = L.rng_snapshot(run)
    random.random()
    assert L.rng_snapshot(run)["python_random"] != before["python_random"]


@pytest.mark.parametrize("bad", [{"panel_width": 0.5}, {"nodes_per_panel": 12}, {"step_requested": 0.1}, {"conc_scale": 2.0},
                                 {"stage": 1}, {"n_y": 4000}, {"q": 55.0}, {"weight_sum": 1.001}])
def test_built_table_is_checked_against_the_locked_rule(proto, bad):
    """check_built_table refuses a table whose meta differs from the locked rule."""
    spec = GameSpec(**proto["records"]["50"]["game"])
    actor = BetaActor(PPOConfig().hidden, PPOConfig().c_min, PPOConfig().mu_clamp, torch.Generator().manual_seed(0))
    tab = vc.build_continuation_table(actor, spec, stage=2, step=0.25)       # coarse step: only its meta is used below
    meta = dict(tab.meta)
    meta.update(step_requested=0.05, n_y=4001, panel_width=1.0, nodes_per_panel=6, conc_scale=1.0)
    ok = vc.ContinuationTable(np.linspace(-100, 100, 4001), np.zeros(4001), dict(meta))
    assert L.check_built_table(ok, spec)["n_y"] == 4001
    meta.update(bad)
    with pytest.raises(ConfigError, match="locked rule|quadrature weights"):
        L.check_built_table(vc.ContinuationTable(np.linspace(-100, 100, 4001), np.zeros(4001), meta), spec)


def test_built_table_dtype_is_checked(proto):
    """A float32 table is refused (the locked rule is float64)."""
    spec = GameSpec(**proto["records"]["50"]["game"])
    actor = BetaActor(PPOConfig().hidden, PPOConfig().c_min, PPOConfig().mu_clamp, torch.Generator().manual_seed(0))
    meta = dict(vc.build_continuation_table(actor, spec, stage=2, step=0.25).meta)
    meta.update(step_requested=0.05, n_y=4001, panel_width=1.0, nodes_per_panel=6, conc_scale=1.0)
    with pytest.raises(ConfigError, match="locked rule"):
        L.check_built_table(vc.ContinuationTable(np.linspace(-100, 100, 4001), np.zeros(4001, dtype=np.float32), meta), spec)


def test_build_that_draws_a_training_stream_aborts_the_run(proto, tmp_path, monkeypatch):
    """A table build that moves a training stream aborts the run (exception -> status failed, exit code 1)."""
    spy = []

    class SpyRun(Run):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            spy.append(self)
    real = vc.build_continuation_table

    def drawing(*a, **k):
        spy[0].rngs["env"].random()
        return real(*a, **k)
    monkeypatch.setattr(L, "Run", SpyRun)
    monkeypatch.setattr(vc, "build_continuation_table", drawing)
    out = str(tmp_path / "run")
    os.makedirs(out)
    code = L.run_pipeline(_small_cfg(proto, out), proto, L.PROTOCOL_SHA256, out, "pytest", band_step=2.0)
    assert code == 1
    st = json.load(open(os.path.join(out, "status.json")))
    assert st["state"] == "failed" and "training RNG stream moved" in st["traceback"]


def test_build_that_draws_a_global_rng_is_a_violation(proto, tmp_path, monkeypatch):
    """A table build that draws from a global RNG is recorded at point 'table_build'; the run exits with code 5."""
    real = vc.build_continuation_table

    def drawing(*a, **k):
        np.random.rand()
        return real(*a, **k)
    monkeypatch.setattr(vc, "build_continuation_table", drawing)
    out = str(tmp_path / "run")
    os.makedirs(out)
    code = L.run_pipeline(_small_cfg(proto, out), proto, L.PROTOCOL_SHA256, out, "pytest", band_step=2.0)
    assert code == L.RNG_VIOLATION_EXIT
    g = json.load(open(os.path.join(out, "gates.json")))
    pts = [v["point"] for v in g["global_rng"]["violations"] if v["rng"] == "numpy_global"]
    assert pts[0] == "table_build" and g["run_pass"] is False and g["outcome"] == "global_rng_violation"
    assert g["continuation_table"]["rng_unchanged_by_build"]["numpy_global"] is False


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


# ------------------------------------------------------------------ analysis script on the smoke run
def test_confirmation_analysis_on_smoke(smoke, tmp_path):
    root, out, code = smoke
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "v2" / "confirmation_analysis_v2_0.py"), "--root", str(root),
                        "--seeds", "10501", "10501", "--rehearsal-safety", "--paired-root", str(root), "--compare-root", str(root),
                        "--compare-seeds", "10501", "10501", "--out", str(tmp_path / "an")], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    import pandas as pd
    ag = pd.read_csv(tmp_path / "an" / "agreement.csv")
    assert len(ag) == 1 and bool(ag.all_agree.iloc[0])
    assert float(ag.filter(like="absdiff_").to_numpy().max()) == 0.0
    for c in ("G-S", "v1_1_run_pass", "run_pass", "outcome", "protocol_version", "protocol_sha256", "table_sha256", "table_rule"):
        assert bool(ag[c].iloc[0]), c
    pr = pd.read_csv(tmp_path / "an" / "per_run.csv")
    assert set(pr.status) == {"completed", "missing"}          # q60 seed10501 is absent in the smoke root
    done = pr[pr.status == "completed"].iloc[0]
    assert done["table_sha256"] == done["table_sha256_disk"]
    assert bool(done["run_pass"]) == bool(done["G-A_pass"] and done["G-F_pass"] and done["G-N_pass"] and done["G-S_pass"])
    rs = json.load(open(tmp_path / "an" / "rehearsal_safety.json"))
    assert rs["pass"] is False and rs["n_completed"] == 1                       # one run only: the 20-run condition cannot hold
    ps = pd.read_csv(tmp_path / "an" / "paired_summary.csv")
    assert (ps.mean_diff == 0).all() and (ps.n_equal == 1).all()
    v = json.load(open(tmp_path / "an" / "verdict.json"))
    assert v["overall"].startswith("not applicable") and v["n_all_agree"] == 1


def test_rehearsal_safety_condition_on_synthetic_runs(proto):
    """19 of 20 under G-S with a good fit passes; 18 of 20 fails; a bad fit (probability < 0.90) fails."""
    import pandas as pd
    rng = np.random.default_rng(1)

    def runs(errs50, errs60):
        rows = []
        for q, errs in ((50, errs50), (60, errs60)):
            for i, e in enumerate(errs):
                rows.append({"q": q, "seed": 10501 + i, "status": "completed", "stage1_rel_err_signed": float(e), "G-S_pass": abs(e) <= 0.05})
        return pd.DataFrame(rows)
    good = rng.normal(0.0, 0.015, 10)
    assert A.rehearsal_safety(runs(good, good), proto, range(10501, 10511))["pass"]
    bad = good.copy()
    bad[0] = 0.06
    r19 = A.rehearsal_safety(runs(bad, good), proto, range(10501, 10511))
    assert r19["pooled_n_G-S_pass"] == 19
    bad2 = bad.copy()
    bad2[1] = -0.06
    assert A.rehearsal_safety(runs(bad2, good), proto, range(10501, 10511))["pooled_n_G-S_pass"] == 18
    assert not A.rehearsal_safety(runs(bad2, good), proto, range(10501, 10511))["pass"]
    wide = np.array([0.045, -0.045, 0.04, -0.04, 0.045, -0.044, 0.043, -0.045, 0.044, -0.043])      # all within 0.05, SD ~ 0.044
    rw = A.rehearsal_safety(runs(wide, good), proto, range(10501, 10511))
    assert rw["pooled_n_G-S_pass"] == 20 and rw["per_q"]["50"]["P_ge_k_of_n"] < 0.90 and not rw["pass"]


# ------------------------------------------------------------------ protocol generator
def test_generator_reproduces_the_committed_protocol():
    """apply_changes(v1.1) + the records reproduces protocols/v2_T2_locked_v2_0.json and .md exactly; D2-D5 paths only."""
    N = G.load_numbers()
    v11 = json.load(open(G.V11))
    p = G.apply_changes(v11, N)
    assert json.dumps(p, indent=1) + "\n" == G.V20.read_text()
    assert G.render_md(p, N) == G.MD.read_text()
    assert G.render_numerics_note(N) == (G.OUT / "verifier_numerics_note.md").read_text()
    d = G.diff(v11, p)
    assert [x["path"] for x in d if x["path"] not in G.ALLOWED] == []
    assert not [x for x in d if x["op"] == "removed"]
    assert json.loads(G.DIFF.read_text()) == json.loads(json.dumps(d))
    assert G.sha256_file(G.V11) == V11_SHA


# ------------------------------------------------------------------ analysis script on degenerate roots
def _run_analysis(root, out, *extra, seeds=("10501", "10502")):
    return subprocess.run([sys.executable, str(ROOT / "tools" / "v2" / "confirmation_analysis_v2_0.py"), "--root", str(root), "--seeds",
                           *seeds, "--out", str(out), *extra], capture_output=True, text=True, cwd=ROOT)


def test_analysis_on_a_root_without_completed_runs(tmp_path):
    """Every run missing (or failed): the analysis still writes verdict.json and tables.md with a FAIL/not-applicable verdict."""
    (tmp_path / "root" / "q50" / "seed10501").mkdir(parents=True)
    (tmp_path / "root" / "q50" / "seed10501" / "status.json").write_text(json.dumps({"state": "failed", "exit_code": 1, "traceback": "x\nBoom"}))
    r = _run_analysis(tmp_path / "root", tmp_path / "an", "--rehearsal-safety")
    assert r.returncode == 0, r.stderr
    v = json.load(open(tmp_path / "an" / "verdict.json"))
    assert v["n_completed"] == 0 and v["n_failed_exception"] == 1 and v["n_missing"] == 3 and v["n_expected"] == 4
    assert v["overall"].startswith("not applicable") and (tmp_path / "an" / "tables.md").exists()
    assert json.load(open(tmp_path / "an" / "rehearsal_safety.json"))["pass"] is False


def test_analysis_with_empty_paired_and_compare_roots(smoke, tmp_path):
    """A paired / compare root without runs gives a warning and empty tables, not a crash."""
    root, out, code = smoke
    (tmp_path / "empty").mkdir()
    r = _run_analysis(root, tmp_path / "an", "--paired-root", str(tmp_path / "empty"), "--compare-root", str(tmp_path / "empty"),
                      seeds=("10501", "10501"))
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("WARNING") == 2
    v = json.load(open(tmp_path / "an" / "verdict.json"))
    assert v["paired_root"]["n_pairs"] == 0 and v["compare_root"]["n_completed"] == 0
    assert v["n_completed"] == 1 and v["n_clean_tree"] in (0, 1) and len(v["commits"]) == 1


def test_analysis_paired_gate_flips_on_the_smoke_root(smoke, tmp_path):
    """Pairing the smoke run with itself: no flips, every gate flag identical."""
    import pandas as pd
    root, out, code = smoke
    r = _run_analysis(root, tmp_path / "an", "--paired-root", str(root), seeds=("10501", "10501"))
    assert r.returncode == 0, r.stderr
    pf = pd.read_csv(tmp_path / "an" / "paired_gate_flips.csv")
    assert (pf.n_ref_pass_new_fail == 0).all() and (pf.n_ref_fail_new_pass == 0).all() and (pf.n_ref_pass == pf.n_new_pass).all()
    assert set(pf["flag"]) == set(A.GATE_FLAGS)
