"""Round R2c post-launch checks (tools/v2/r2c_launch_checks.py): pure functions, real-data controls, end to end.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_v2_r2c_tools.py -p no:cacheprovider -q

Sections: 1. pure functions on synthetic directories; 2. positive and negative controls on real result
files (skipped when the reference directories are absent); 3. the whole tool on a tiny synthetic root.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import launch_refine as L  # noqa: E402
import r2c_launch_checks as K  # noqa: E402

WT = Path("/home/fjiang4/tournament_experiment/.claude/worktrees")
R1 = WT / "v2-t2-refine" / "results" / "v2_refine"
R2B = WT / "v2-t2-r2b" / "results" / "v2_refine_r2b"


# ====================================================================== 1. pure functions
def _write_run(d: Path, updates: Dict[int, float], state_value: Optional[float] = None) -> None:
    """A fake run directory: ``weights/uNNNNN.npz`` filled with one value each, optionally a full state."""
    os.makedirs(d / "weights", exist_ok=True)
    for u, v in updates.items():
        np.savez(d / "weights" / f"u{u:05d}.npz", w=np.full((3, 2), v, dtype=np.float32),
                 b=np.full(2, v, dtype=np.float32))
    if state_value is not None:
        torch.save(_fake_state(state_value), d / f"state_u{K.STATE_AT:05d}.pt")


def _fake_state(v: float) -> Dict[str, Any]:
    """The structure of ``Run.full_state`` with small tensors."""
    t = torch.full((4,), v, dtype=torch.float32)
    mb = {"bit_generator": "PCG64", "state": {"state": 12345, "inc": 67}, "has_uint32": 0, "uinteger": 0}
    return {
        "format": "v2_full_state/1", "phase_done": "A", "phases_done": [],
        "agent": {"actor": {"w": t.clone()}, "critic": {"w": t.clone()}, "opponent": {"w": t.clone()},
                  "frozen": None, "opt_actor": {"state": {0: {"exp_avg": t.clone(), "step": torch.tensor(3.0)}},
                                                "param_groups": [{"lr": 3e-4}]},
                  "opt_critic": {"state": {}, "param_groups": [{"lr": 3e-4}]}, "snapshot_refreshes": 2,
                  "rng_minibatch": mb},
        "rng": {k: copy.deepcopy(mb) for k in ("env", "learn", "opp", "start")},
        "torch_generator_state": torch.arange(8, dtype=torch.uint8),
        "torch_global_rng_state": torch.arange(8, dtype=torch.uint8),
        "numpy_global_rng_state": ("MT19937", np.arange(5, dtype=np.uint32), 1, 0, 0.0),
        "python_random_state": (3, (1, 2, 3), None),
        "counters": {"global_u": 1200, "total_episodes": 10, "total_transitions": 10},
        "snapshot_log": [{"update": 0, "reason": "init"}],
        "schedule_positions": {"actor_lr": 3e-4}, "flags": {"reward_mode": "expected"}, "seed": 1, "q": 50,
    }


def _ulp_up(v: float) -> float:
    return float(np.nextafter(np.float32(v), np.float32(np.inf)))


def test_compare_exports_identical_and_one_ulp_and_missing_and_extra_and_beyond_bound(tmp_path):
    ups = {25: 0.1, 50: 0.2, 75: 0.3, 100: 0.4}
    _write_run(tmp_path / "ref", ups)
    _write_run(tmp_path / "same", ups)
    ex = K.compare_exports(tmp_path / "ref", tmp_path / "same", 75)
    assert ex["prefix_ok"] and ex["n_prefix"] == 3 and ex["first_diff_update"] is None

    _write_run(tmp_path / "ulp", {**ups, 50: _ulp_up(0.2)})                      # one ulp in one export
    ex = K.compare_exports(tmp_path / "ref", tmp_path / "ulp", 75)
    assert not ex["prefix_ok"] and ex["first_diff_update"] == 50 and ex["n_differing_in_prefix"] == 1
    assert ex["differing"][0]["update"] == 50

    _write_run(tmp_path / "miss", {u: v for u, v in ups.items() if u != 50})     # missing export
    ex = K.compare_exports(tmp_path / "ref", tmp_path / "miss", 75)
    assert not ex["prefix_ok"] and ex["missing"] == [50] and ex["extra"] == []

    _write_run(tmp_path / "extra", {**ups, 60: 0.25})                            # extra export inside the bound
    ex = K.compare_exports(tmp_path / "ref", tmp_path / "extra", 75)
    assert not ex["prefix_ok"] and ex["extra"] == [60]

    _write_run(tmp_path / "late", {**ups, 100: 9.0, 125: 1.0})                   # difference and extra after the bound
    ex = K.compare_exports(tmp_path / "ref", tmp_path / "late", 75)
    assert ex["prefix_ok"] and ex["first_diff_update"] == 100 and ex["extra"] == []

    _write_run(tmp_path / "dtype", ups)                                          # dtype differs, values equal
    np.savez(tmp_path / "dtype" / "weights" / "u00025.npz", w=np.full((3, 2), 0.1, dtype=np.float64),
             b=np.full(2, 0.1, dtype=np.float32))
    assert not K.compare_exports(tmp_path / "ref", tmp_path / "dtype", 75)["prefix_ok"]

    (tmp_path / "empty" / "weights").mkdir(parents=True)                         # nothing to compare is not a pass
    assert not K.compare_exports(tmp_path / "empty", tmp_path / "empty", 75)["prefix_ok"]
    assert not K.compare_exports(tmp_path / "nowhere", tmp_path / "nowhere", 75)["prefix_ok"]


def test_compare_states_identical_and_each_kind_of_difference(tmp_path):
    def save(name: str, s: Dict[str, Any]) -> Path:
        torch.save(s, tmp_path / name)
        return tmp_path / name

    ref = save("ref.pt", _fake_state(0.5))
    assert K.compare_states(ref, save("same.pt", _fake_state(0.5)))["identical"]

    s = _fake_state(0.5)
    s["agent"]["actor"]["w"][2] = torch.nextafter(s["agent"]["actor"]["w"][2], torch.tensor(1.0))
    r = K.compare_states(ref, save("ulp.pt", s))
    assert not r["identical"] and r["differing_fields"] == ["agent.actor"]

    s = _fake_state(0.5)
    s["agent"]["opt_actor"]["state"][0]["exp_avg"] += 1e-7
    assert K.compare_states(ref, save("adam.pt", s))["differing_fields"] == ["agent.opt_actor"]

    s = _fake_state(0.5)
    s["rng"]["start"]["state"]["state"] += 1
    assert K.compare_states(ref, save("rng.pt", s))["differing_fields"] == ["rng"]

    s = _fake_state(0.5)
    s["agent"]["rng_minibatch"]["state"]["inc"] += 1
    assert K.compare_states(ref, save("mb.pt", s))["differing_fields"] == ["agent.rng_minibatch"]

    s = _fake_state(0.5)
    s["torch_generator_state"][0] += 1
    assert K.compare_states(ref, save("gen.pt", s))["differing_fields"] == ["torch_generator_state"]

    s = _fake_state(0.5)
    s["counters"]["total_episodes"] += 1
    assert K.compare_states(ref, save("cnt.pt", s))["differing_fields"] == ["counters"]

    s = _fake_state(0.5)
    s["agent"]["actor"]["w"] = s["agent"]["actor"]["w"].double()                 # dtype
    assert not K.compare_states(ref, save("dt.pt", s))["identical"]

    s = _fake_state(0.5)                                                         # process-global RNG: reported only
    s["torch_global_rng_state"][0] += 1
    r = K.compare_states(ref, save("glob.pt", s))
    assert r["identical"] and r["global_rng_identical"] is False

    with pytest.raises(FileNotFoundError):
        K.compare_states(ref, tmp_path / "absent.pt")


def test_compare_update_rows_ignores_wall_clock_columns_and_counts_identical_rows(tmp_path):
    def write(name: str, wall: float, bump_row: Optional[int] = None) -> Path:
        with open(tmp_path / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["update", "kl", "update_wall_sec", "time_sec"])
            for u in range(1, 11):
                w.writerow([u, f"0.{u}" if u != bump_row else "0.99", wall + u, wall])
        return tmp_path / name

    ref = write("a.csv", 1.0)
    r = K.compare_update_rows(ref, write("b.csv", 50.0), 8)
    assert (r["n_rows_ref"], r["n_identical"]) == (8, 8)
    r = K.compare_update_rows(ref, write("c.csv", 50.0, bump_row=5), 8)
    assert r["n_identical"] == 7
    assert K.compare_update_rows(ref, write("d.csv", 50.0, bump_row=9), 8)["n_identical"] == 8   # beyond the bound


def test_paths_outside_and_commit_logic_on_synthetic_path_lists(monkeypatch):
    allowed = K.ALLOWED_AFTER_CODE
    assert allowed == ("results/", "reports/v2/refine_r2c/")
    paths = ["results/v2_refine_r2c/waveS/q50/x.json", "reports/v2/refine_r2c/01.md"]
    assert K.paths_outside(paths, allowed) == []
    assert K.paths_outside(paths + ["tools/v2/launch_refine.py", "reports/v2/refine_r2b/x.md",
                                    "resultsX/a", "run/run_v2_stagewise.py"], allowed) == [
        "tools/v2/launch_refine.py", "reports/v2/refine_r2b/x.md", "resultsX/a", "run/run_v2_stagewise.py"]
    assert K.paths_outside([], allowed) == []
    assert K.paths_outside(["docs/a.md"], ("docs/",)) == []                     # --allowed-after-code replaces the default

    K._diff_names.cache_clear()
    calls = []

    def fake_diff(code: str, commit: str):
        calls.append((code, commit))
        return {"c_ok": ("results/a.csv",), "c_bad": ("results/a.csv", "run/run_v2_stagewise.py"),
                "c_git_fails": None}[commit]

    monkeypatch.setattr(K, "_diff_names", fake_diff)
    assert K.commit_touches_only_allowed("code", "c_ok") == []
    assert K.commit_touches_only_allowed("code", "c_bad") == ["run/run_v2_stagewise.py"]
    assert K.commit_touches_only_allowed("code", "c_git_fails") is None
    man = {"git": {"commit": "c_ok", "dirty": False}}
    assert K.check_manifest_commit(man, "code", K.ALLOWED_AFTER_CODE) == []
    assert any("dirty" in p for p in K.check_manifest_commit({"git": {"commit": "c_ok", "dirty": True}}, "code", allowed))
    assert any("dirty" in p for p in K.check_manifest_commit({"git": {"commit": "c_ok"}}, "code", allowed))
    assert any("outside" in p for p in K.check_manifest_commit({"git": {"commit": "c_bad", "dirty": False}}, "code", allowed))
    assert any("cannot compare" in p for p in K.check_manifest_commit({"git": {"commit": "c_git_fails", "dirty": False}}, "code", allowed))
    assert any("cannot compare" in p for p in K.check_manifest_commit({"git": {"dirty": False}}, "code", allowed))
    assert any("cannot compare" in p for p in K.check_manifest_commit({}, "code", allowed))


def _synthetic_config() -> Dict[str, Any]:
    return {"arm": "A_parent", "run": "r1", "pilot": "p", "q": 50, "mode": "phase_A",
            "record": {"run": "x", "output_dir": "/a", "seed": 1}, "ppo_overrides": {"minibatch": 128},
            "lr_decay": [{"phase": "A", "start_lr": 3e-4, "end_lr": 3e-4}]}


def test_config_diff_problems_with_a_synthetic_config():
    expected = frozenset({"start_weights"})
    ref = _synthetic_config()                                                  # R1 style: no R2b keys
    cfg = copy.deepcopy(ref)
    cfg.update(copy.deepcopy(L.R2B_DEFAULTS))
    cfg.update(arm="A_peak35", run="r2c", pilot="q", start_weights=L.R2C_WAVE_S_ARMS["A_peak35"]["r2b"]["start_weights"])
    cfg["record"].update(run="y", output_dir="/b")
    assert K.config_diff_problems(cfg, ref, expected) == ([], ["start_weights"])   # identity keys are ignored
    assert "start_weights" not in ref and "clamp_likelihood" not in ref           # the reference is not modified

    cfg2 = copy.deepcopy(cfg)
    cfg2["ppo_overrides"]["minibatch"] = 64
    problems, diff = K.config_diff_problems(cfg2, ref, expected)
    assert problems and diff == ["ppo_overrides.minibatch", "start_weights"]

    cfg3 = copy.deepcopy(cfg)
    cfg3["start_weights"] = None                                               # the one expected change is absent
    assert K.config_diff_problems(cfg3, ref, expected)[0]

    cfg4 = copy.deepcopy(cfg)
    cfg4["clamp_likelihood"] = "censored"
    assert K.config_diff_problems(cfg4, ref, expected)[1] == ["clamp_likelihood", "start_weights"]


def test_manifest_value_and_status_problems():
    want = L.R2C_WAVE_S_ARMS["A_peak50_late400"]["r2b"]["start_weights"]
    assert want["local_first"] == 1201
    man = {"start_weights": copy.deepcopy(want), "clamp_likelihood": "density", "pathwise_epochs": 1,
           "pathwise_minibatch": None}
    assert K.manifest_value_problems(man, want) == []
    for edit in ({"start_weights": {**want, "local_first": 1}}, {"start_weights": {k: v for k, v in want.items()
                                                                                    if k != "local_first"}},
                 {"clamp_likelihood": "censored"}, {"pathwise_epochs": 10}, {"pathwise_minibatch": 256}):
        assert K.manifest_value_problems({**man, **edit}, want), edit
    assert K.manifest_value_problems({k: v for k, v in man.items() if k != "pathwise_minibatch"}, want)   # absent != None

    ok = {"state": "done", "exit_code": 0, "final_global_update": 1600}
    assert K.status_problems(ok) == []
    for edit in ({"state": "running"}, {"state": "failed", "exit_code": 1}, {"exit_code": None},
                 {"final_global_update": 1200}):
        assert K.status_problems({**ok, **edit}), edit
    assert K.status_problems({})


# ====================================================================== 2. controls on real data
def _need(*paths: Path) -> None:
    for p in paths:
        if not p.is_dir():
            pytest.skip(f"reference directory absent: {p}")


def test_real_parents_a_against_itself_is_identical_and_against_another_seed_differs():
    a, b = R1 / "parents_A" / "q50" / "seed10501", R1 / "parents_A" / "q50" / "seed10502"
    _need(a, b)
    r = K.prefix_check(a, a, 1200, with_state=True)
    assert r["exports"]["prefix_ok"] and r["exports"]["n_prefix"] == 48 and r["state"]["identical"]
    assert r["v2_updates"]["n_identical"] == r["v2_updates"]["n_rows_ref"] == 1200
    assert not r["ok"] and r["exports"]["first_diff_update"] is None        # the divergence sanity check fails
    r = K.prefix_check(a, b, 1200, with_state=True)
    assert not r["exports"]["prefix_ok"] and r["exports"]["first_diff_update"] == 25
    assert not r["state"]["identical"] and not r["ok"]


@pytest.mark.parametrize("q,seed", [(50, 10503), (50, 10504), (60, 10504)])
def test_real_censored_run_is_identical_to_its_baseline_before_the_first_flagged_update(q, seed):
    """A_censored differs from parents_A only through censored log-masses, which need a clamped draw; before the
    first flagged update (``per_run.csv`` d1_first_flagged_update) the exports are bit-identical."""
    run, base = R2B / "waveA" / f"q{q}" / f"seed{seed}" / "A_censored", R1 / "parents_A" / f"q{q}" / f"seed{seed}"
    per_run = R2B / "analysis" / "per_run.csv"
    _need(run, base)
    if not per_run.exists():
        pytest.skip(f"missing {per_run}")
    with open(per_run) as f:
        flagged = [float(r["d1_first_flagged_update"]) for r in csv.DictReader(f)
                   if (r["arm"], int(r["q"]), int(r["seed"])) == ("A_censored", q, seed)
                   and r["d1_first_flagged_update"] != ""]
    assert len(flagged) == 1 and flagged[0] > 300
    last_before = int(flagged[0] // 25 * 25) if flagged[0] % 25 else int(flagged[0] - 25)
    ex = K.compare_exports(base, run, last_before)
    assert ex["prefix_ok"] and ex["n_prefix"] == last_before // 25
    assert ex["first_diff_update"] is not None and ex["first_diff_update"] >= last_before + 25
    assert ex["first_diff_update"] > flagged[0] - 1                             # no export before the flag differs
    csv_rows = K.compare_update_rows(base / "v2_updates.csv", run / "v2_updates.csv", last_before)
    assert csv_rows["n_identical"] == last_before


def test_real_peak50_from_update_1_fails_the_late400_prefix_check():
    run, base = R2B / "waveA" / "q50" / "seed10501" / "A_peak50", R1 / "parents_A" / "q50" / "seed10501"
    _need(run, base)
    r = K.prefix_check(base, run, 1200, with_state=True)
    assert not r["ok"] and r["exports"]["first_diff_update"] == 25 and not r["state"]["identical"]
    assert K.compare_exports(base, run, 0)["first_diff_update"] == 25
    assert K.descriptive_first_diff(base, run) == {"first_diff_update": 25, "ok": True, "problems": []}


def test_real_r1_config_differs_from_a_built_wave_s_config_in_start_weights_only(tmp_path):
    """C2 on real data: the on-disk R1 parents_A config against what the launcher writes for the four arms."""
    ref_path = R1 / "parents_A" / "q50" / "seed10501" / "run_config.json"
    if not ref_path.exists():
        pytest.skip(f"missing {ref_path}")
    ref = json.load(open(ref_path))
    jobs = L.build_configs(L.R2C_WAVES[0], (50,), (10501,), None, tmp_path)
    assert [j.cfg["arm"] for j in jobs] == list(L.R2C_WAVE_S_ARMS)
    for j in jobs:
        assert K.config_diff_problems(j.cfg, ref, K.arm_tables(j.cfg["arm"])[1]) == ([], ["start_weights"])


# ====================================================================== 3. the whole tool on a synthetic root
BOUNDS = K.PREFIX_BOUNDS
UPS = [25, 50, 800, 825, 1200, 1225, 1600]


def _tiny_root(tmp_path: Path):
    """One (q, seed): R1 parents_A (config, exports, state, csv) and the four arm runs as the runner writes them."""
    root, r1 = tmp_path / "r2c", tmp_path / "r1"
    q, seed = 50, 10501
    ref = r1 / "parents_A" / f"q{q}" / f"seed{seed}"
    base_vals = {u: 0.001 * u for u in UPS}
    _write_run(ref, base_vals, state_value=0.5)
    comparator = L.r2b_comparator_config(L.R2C_WAVES[0], q, seed, root)["A_parent"]
    json.dump({k: v for k, v in comparator.items() if k not in L.R2B_DEFAULTS}, open(ref / "run_config.json", "w"))
    _write_csv(ref / "v2_updates.csv", 1600, {})
    for job in L.build_configs(L.R2C_WAVES[0], (q,), (seed,), None, root):
        arm, d = job.cfg["arm"], Path(job.out_dir)
        vals = dict(base_vals)
        bound = BOUNDS.get(arm, 0)
        for u in UPS:
            if u > bound:
                vals[u] = base_vals[u] + 1.0                                     # diverges right after the bound
        _write_run(d, vals, state_value=0.5 if bound == 1200 else None)
        _write_csv(d / "v2_updates.csv", 1600, {} if bound == 0 else {bound + 1: "9"})
        man = {"git": {"commit": "HEAD", "dirty": False}, "input_config": job.cfg,
               **{k: job.cfg[k] for k in L.R2B_DEFAULTS}}
        json.dump(man, open(d / "manifest.json", "w"))
        json.dump({"state": "done", "exit_code": 0, "final_global_update": 1600}, open(d / "status.json", "w"))
    return root, r1


def _write_csv(p: Path, n: int, bump: Dict[int, str]) -> None:
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["update", "kl", "update_wall_sec"])
        for u in range(1, n + 1):
            w.writerow([u, bump.get(u, "0.1"), u * 0.5 + len(bump)])


def _edit(path: Path, fn) -> None:
    with open(path) as f:
        j = json.load(f)
    fn(j)
    with open(path, "w") as f:
        json.dump(j, f)


def test_check_run_passes_on_a_consistent_tiny_root_and_fails_on_each_violation(tmp_path):
    root, r1 = _tiny_root(tmp_path)

    def run(arm: str) -> Dict[str, Any]:
        return K.check_run(arm, 50, 10501, root, r1, "HEAD")

    def rdir(arm: str) -> Path:
        return root / "waveS" / "q50" / "seed10501" / arm

    arms = list(L.R2C_WAVE_S_ARMS)
    results = {a: run(a) for a in arms}
    assert all(r["ok"] for r in results.values()), {a: r["checks"] for a, r in results.items() if not r["ok"]}
    assert results["A_peak50_late400"]["checks"]["C5"]["exports"]["first_diff_update"] == 1225
    assert results["A_peak50_late800"]["checks"]["C5"]["exports"]["first_diff_update"] == 825
    assert results["A_peak35"]["checks"]["C5"] == {"required": False, "first_diff_update": 25, "ok": True,
                                                   "problems": []}
    assert results["A_peak50_late400"]["checks"]["C5"]["v2_updates"]["n_identical"] == 1200

    _edit(rdir("A_peak35") / "manifest.json", lambda m: m["git"].update(dirty=True))      # dirty manifest
    r = run("A_peak35")
    assert not r["ok"] and not r["checks"]["C1"]["ok"] and any("dirty" in p for p in r["checks"]["C1"]["problems"])
    assert r["checks"]["C2"]["ok"] and r["checks"]["C3"]["ok"] and r["checks"]["C4"]["ok"]

    _edit(rdir("A_peak40") / "manifest.json", lambda m: m.update(start_weights={**m["start_weights"], "local_first": 5}))
    r = run("A_peak40")                                                                   # wrong start_weights
    assert not r["checks"]["C3"]["ok"] and "local_first" in r["checks"]["C3"]["problems"][0]
    assert r["checks"]["C1"]["ok"] and r["checks"]["C2"]["ok"]

    _edit(rdir("A_peak50_late400") / "manifest.json", lambda m: m["input_config"]["ppo_overrides"].update(minibatch=64))
    assert not run("A_peak50_late400")["checks"]["C2"]["ok"]

    _edit(rdir("A_peak50_late800") / "status.json", lambda s: s.update(state="failed", exit_code=1))
    r = run("A_peak50_late800")
    assert not r["checks"]["C4"]["ok"] and r["checks"]["C5"]["ok"]

    np.savez(rdir("A_peak50_late400") / "weights" / "u00800.npz", w=np.full((3, 2), 7, dtype=np.float32),
             b=np.full(2, 7, dtype=np.float32))                                             # prefix export differs
    assert not run("A_peak50_late400")["checks"]["C5"]["ok"]

    (rdir("A_peak50_late800") / "status.json").unlink()                                     # missing file: failed
    assert not run("A_peak50_late800")["checks"]["C4"]["ok"]
    (rdir("A_peak35") / "manifest.json").unlink()
    r = run("A_peak35")
    assert [r["checks"][c]["ok"] for c in ("C1", "C2", "C3")] == [False] * 3
    assert not K.check_run("A_peak35", 50, 10502, root, r1, "HEAD")["ok"]                   # no run directory at all


def test_c3_pins_the_config_the_run_read_and_a_corrupt_file_is_a_failed_check_not_a_crash(tmp_path):
    root, r1 = _tiny_root(tmp_path)
    d = root / "waveS" / "q50" / "seed10501" / "A_peak35"

    def run() -> Dict[str, Any]:
        return K.check_run("A_peak35", 50, 10501, root, r1, "HEAD")

    assert run()["ok"]
    # the top-level value is right, the config the run actually read is not
    _edit(d / "manifest.json", lambda m: m["input_config"].update(
        start_weights={**m["input_config"]["start_weights"], "peak_share": 0.9}))
    r = run()
    assert not r["checks"]["C3"]["ok"] and any("input_config.start_weights" in p for p in r["checks"]["C3"]["problems"])
    # a manifest cut off mid-write (a worker killed while rewriting it): C1-C3 fail, nothing is raised
    (d / "manifest.json").write_text('{"git": {"commit": "HE')
    r = run()
    assert [r["checks"][c]["ok"] for c in ("C1", "C2", "C3")] == [False] * 3
    assert "unreadable" in r["checks"]["C1"]["problems"][0]
    (d / "status.json").write_text("not json at all")
    r = run()
    assert not r["checks"]["C4"]["ok"] and "unreadable" in r["checks"]["C4"]["problems"][0] and not r["ok"]
    # a manifest of the wrong shape
    json.dump({"git": "abc"}, open(d / "manifest.json", "w"))
    r = run()
    assert not r["checks"]["C1"]["ok"] and r["checks"]["C2"]["problems"] == ["manifest has no input_config"]


def test_late400_state_difference_and_a_run_that_never_diverges_fail_c5(tmp_path):
    root, r1 = _tiny_root(tmp_path)
    d = root / "waveS" / "q50" / "seed10501" / "A_peak50_late400"
    torch.save(_fake_state(0.5000001), d / "state_u01200.pt")                               # state differs only
    c5 = K.check_run("A_peak50_late400", 50, 10501, root, r1, "HEAD")["checks"]["C5"]
    assert not c5["ok"] and c5["exports"]["prefix_ok"] and not c5["state"]["identical"]
    torch.save(_fake_state(0.5), d / "state_u01200.pt")
    (d / "state_u01200.pt").unlink()                                                        # missing state: failed
    c5 = K.check_run("A_peak50_late400", 50, 10501, root, r1, "HEAD")["checks"]["C5"]
    assert not c5["ok"] and c5["state"]["first_difference"]["field"] == "<missing file>"
    _write_run(d, {u: 0.001 * u for u in UPS})                                              # identical at every export
    c5 = K.check_run("A_peak50_late400", 50, 10501, root, r1, "HEAD")["checks"]["C5"]
    assert not c5["ok"] and any("must diverge" in p for p in c5["problems"])


def test_main_writes_json_summary_and_exit_status(tmp_path, monkeypatch, capsys):
    root, r1 = _tiny_root(tmp_path)
    monkeypatch.setattr(L, "DEFAULT_QS", (50,))
    monkeypatch.setattr(L, "DEFAULT_SEEDS", (10501,))
    out = tmp_path / "out" / "checks.json"
    argv = ["--code-commit", "HEAD", "--root", str(root), "--r1-root", str(r1), "--out", str(out), "--workers", "1"]
    assert K.main(argv) == 0
    j = json.load(open(out))
    s = j["summary"]
    assert s["all_pass"] and s["n_runs"] == 4 and len(j["runs"]) == 4
    assert s["checks"]["C5"] == {"n": 2, "n_pass": 2} and s["checks"]["C1"] == {"n": 4, "n_pass": 4}
    assert s["code_commit"] == "HEAD" and len(s["head"]) == 40 and s["roots"]["root"] == str(root)
    assert "all_pass=True" in capsys.readouterr().out

    _edit(root / "waveS" / "q50" / "seed10501" / "A_peak35" / "manifest.json", lambda m: m["git"].update(dirty=True))
    assert K.main(argv) == 2
    s = json.load(open(out))["summary"]
    assert not s["all_pass"] and s["checks"]["C1"] == {"n": 4, "n_pass": 3}
    assert "FAIL" in capsys.readouterr().out
    assert K.main(argv + ["--allowed-after-code", "docs/"]) == 2
    assert json.load(open(out))["summary"]["allowed_after_code"] == ["docs/"]

    # hidden validation options: another wave directory, an arm list and an explicit prefix bound
    assert K.main(["--code-commit", "HEAD", "--root", str(root), "--r1-root", str(r1), "--out", str(out),
                   "--workers", "1", "--wave-dirname", "nowhere", "--arms", "A_peak35"]) == 2
    assert json.load(open(out))["summary"]["n_runs"] == 1
