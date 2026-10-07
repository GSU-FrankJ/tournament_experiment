"""MS-R3 post-launch checks (tools/ms/r3_launch_checks.py) on a reduced genuine wave and on tampered copies.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_r3_launch_checks.py -p no:cacheprovider -q

The fixture runs the real runner (``run/run_ms_stagewise.py``, reduced budgets of ``test_ms_r2_runner.reduced_nl``: 60 + 20
updates, ramp 21-40, exports every 5) on all twelve MS-R3 arms of one (q, seed) and on the four MS-R2 ``NL_*`` arms that
the ``t1`` arms are compared with (C-MS5), four runs at a time. The checks run on that real output; the tests then break
copies of it one way at a time.
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import launch_checks as LC1  # noqa: E402
import ms_configs as mc  # noqa: E402
import r3_launch_checks as RC  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402
from test_ms_r2_runner import reduced_nl  # noqa: E402

FAKE_GIT = {"commit": "abc1234", "short": "abc1234", "dirty": False}
RAMP = (21, 40)
Q, SEED = 50, 10501
WORKERS = 4
PARTS = ("weight_exports", "train_history", "updates_csv", "checks_stage1", "checks_stage2", "freeze_arrays",
         "binmaps_stage2", "continuation_table_stage1", "gates")


def _run_one(job: Tuple[Dict[str, Any], str]) -> int:
    """One reduced run in a worker process (a module-level function: the pool pickles it by name)."""
    cfg, out = job
    rms.git_state = lambda: dict(FAKE_GIT)
    os.makedirs(out, exist_ok=True)
    return rms.run_pipeline(cfg, out, "pytest", band_step=2.0)


@pytest.fixture(scope="module")
def wave(tmp_path_factory):
    """``(root of the 12 MS-R3 runs, root of the 4 MS-R2 reference runs)`` of one (q, seed), genuine reduced runs."""
    base = tmp_path_factory.mktemp("r3wave")
    new, ref = base / "r3" / "pilot", base / "r2" / "pilot"
    jobs = [(reduced_nl(a, str(_run(new, a)), q=Q, seed=SEED), str(_run(new, a))) for a in mc.R3_ARMS]
    jobs += [(reduced_nl(a, str(_run(ref, a)), q=Q, seed=SEED), str(_run(ref, a)))
             for a in sorted(set(mc.R3_MS_R2_REFERENCE.values()))]
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        codes = list(pool.map(_run_one, jobs))
    assert codes == [0] * len(jobs) == [0] * 16
    return new, ref


def _run(root: Path, arm: str) -> Path:
    return root / f"q{Q}" / f"seed{SEED}" / arm


def _check(new: Path, ref: Path, actors=None):
    mp = pytest.MonkeyPatch()
    mp.setattr(LC1, "WEIGHTS_EVERY", 5)
    try:
        return RC.check_root(new, ref, [Q], [SEED], "abc1234", actors, ramp=RAMP, export_every=5)
    finally:
        mp.undo()


def _copy(wave, tmp_path) -> Tuple[Path, Path]:
    new, ref = wave
    n2, r2 = tmp_path / "n", tmp_path / "r"
    shutil.copytree(new, n2)
    shutil.copytree(ref, r2)
    return n2, r2


def _edit_npz(path: Path, fn: Callable[[Dict[str, np.ndarray]], None]) -> None:
    z = dict(np.load(path))
    fn(z)
    np.savez(path, **z)


def _edit_json(path: Path, fn: Callable[[Any], None]) -> None:
    d = json.load(open(path))
    fn(d)
    json.dump(d, open(path, "w"))


def _edit_csv(path: Path, fn: Callable[[list], None], drop: Tuple[str, ...] = ()) -> None:
    rows = list(csv.DictReader(open(path)))
    fn(rows)
    cols = [c for c in rows[0] if c not in drop]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _ulp(key: str) -> Callable[[Dict[str, np.ndarray]], None]:
    """Change the first finite entry of ``key`` by one unit in the last place (an integer array: by one)."""
    def fn(z: Dict[str, np.ndarray]) -> None:
        a = z[key].copy()
        if a.dtype.kind == "f":
            i = int(np.flatnonzero(np.isfinite(a.ravel()))[0])
            a.flat[i] = np.nextafter(a.flat[i], np.array(np.inf, dtype=a.dtype))
        else:
            a.flat[0] += 1
        z[key] = a
    return fn


def _nth(i: int, col: str, value: str) -> Callable[[list], None]:
    def fn(rows: list) -> None:
        rows[i][col] = value
    return fn


def _ok(res: Dict[str, Any]) -> bool:
    return all(res[p] for p in PARTS) and res["ALL"] is True


# ---------------------------------------------------------------------------------------------- the genuine wave
def test_the_defaults_are_the_d3_schedule_of_the_real_pilot():
    assert (RC.RAMP_FIRST, RC.RAMP_LAST, RC.EXPORT_EVERY) == (2001, 2200, 25) == (mc.CONC_RAMP_FIRST, mc.CONC_RAMP_LAST, 25)
    assert LC1.WEIGHTS_EVERY == 25 and RC.KINDS == ("bb", "st")
    sched = mc.build_config(mc.load_protocol(), 50, 10501, "relu_st_s16", "/unused")["conc_scale_schedule"]
    assert (sched["local_first"], sched["local_last"], sched["scale_first"], sched["scale_last"]) == (2001, 2200, 1.0, 16.0)


def test_a_genuine_wave_passes_every_check(wave):
    new, ref = wave
    r = _check(new, ref)
    s = r["summary"]
    bad = [x for k in ("scale", "C_INIT", "C_NL", "C_MS5") for x in r[k] if not (x.get("pass") or x.get("ok"))]
    assert r["all_ok"], json.dumps(s) + json.dumps(bad[:2], indent=1)[:2500]
    assert s == {"scale_ok": 12, "scale_n": 12, "C_INIT_pass": 1, "C_INIT_n": 1, "C_NL_pass": 6, "C_NL_n": 6,
                 "C_MS5_pass": 4, "C_MS5_n": 4}
    assert r["expected"] == {"scale_n": 12, "C_INIT_n": 1, "C_NL_n": 6, "C_MS5_n": 4}
    assert r["actors"] == ["t1", "relu", "t10"] and r["arms"] == list(mc.R3_ARMS)
    assert r["base"]["n_ok_per_check"] == {"status": 12, "manifest": 12, "files": 12, "global_rng": 12, "tail_share_coverage": 12}
    assert {(x["actor"], x["sampler"]) for x in r["C_NL"]} == {(a, k) for a in mc.R3_ACTORS for k in ("bb", "st")}
    assert all(x["first_export_after"] == 25 and x["first_export_after_differs"] for x in r["C_NL"])
    assert [(x["arm"], x["reference"]) for x in r["C_MS5"]] == list(mc.R3_MS_R2_REFERENCE.items())
    for x in r["C_MS5"]:                                           # the whole run: 80 updates exported every 5, every row, every array
        assert x["compared"]["exports"] == 16 and x["compared"]["update_rows"] == 80 and x["compared"]["gate_values"] > 100
        assert all(x[p] for p in PARTS) and x["first_difference"] is None
    ci = r["C_INIT"][0]
    assert ci["n_arms"] == ci["n_with_digest"] == 12 and ci["n_distinct"] == 1 and len(ci["digest"]) == 64
    assert r["ignored"]["gates_json_paths"] == list(RC.GATES_IGNORED) and "update_wall_sec" in r["ignored"]["csv_wall_clock_columns"]


def test_the_wave_really_ran_the_three_actors(wave):
    """The comparisons are made on three different actors: the exports, manifests and configs carry the variant."""
    new, _ = wave
    for arm in mc.R3_ARMS:
        actor = mc.r3_arm_parts(arm)[0]
        z = np.load(_run(new, arm) / "weights" / "u00050.npz")
        man = json.load(open(_run(new, arm) / "manifest.json"))
        cfg = json.load(open(_run(new, arm) / "run_config.json"))
        if actor == "t1":
            assert "actor_variant" not in z.files and "actor_variant" not in man and "actor_variant" not in cfg
        else:
            assert str(z["actor_variant"]) == actor and man["actor_variant"] == actor and cfg["actor_variant"] == actor
        assert len(man["init_state_sha256"]) == 64
    w = {a: np.load(_run(new, f"{a}_bb_s1") / "weights" / "u00005.npz")["actor.l2.weight"] for a in mc.R3_ACTORS}
    assert not np.array_equal(w["t1"], w["relu"]) and not np.array_equal(w["relu"], w["t10"])       # trained differently
    rows = {a: list(csv.DictReader(open(_run(new, f"{a}_bb_s1") / "ms_updates.csv"))) for a in mc.R3_ACTORS}
    assert len({r[0]["mean_effort"] for r in rows.values()}) == 1       # zero output layer: the same initial policy, same first rollout
    assert len({r[9]["mean_effort"] for r in rows.values()}) == 3       # then three different forwards, three different rollouts


def test_the_wall_clock_columns_differ_between_the_runs_and_are_not_compared(wave):
    new, ref = wave
    a = list(csv.DictReader(open(_run(new, "t1_bb_s1") / "ms_updates.csv")))
    b = list(csv.DictReader(open(_run(ref, "NL_bb_s1") / "ms_updates.csv")))
    assert [r["update_wall_sec"] for r in a] != [r["update_wall_sec"] for r in b]
    assert all(x["policy_loss"] == y["policy_loss"] and x["rngpos_minibatch"] == y["rngpos_minibatch"] for x, y in zip(a, b))
    assert _ok(RC.whole_run_identity(_run(ref, "NL_bb_s1"), _run(new, "t1_bb_s1")))


# ---------------------------------------------------------------------------------------------- the scale record
def test_a_tampered_applied_scale_or_exported_scale_fails(wave, tmp_path):
    new, ref = _copy(wave, tmp_path)
    _edit_csv(_run(new, "relu_bb_s16") / "ms_updates.csv", _nth(45, "conc_scale", "3.9"))       # after the ramp: 16 expected
    r = _check(new, ref)
    assert not r["all_ok"] and r["summary"]["scale_ok"] == 11
    assert [x["arm"] for x in r["scale"] if not x["ok"]] == ["relu_bb_s16"]
    new2, ref2 = _copy(wave, tmp_path / "b")
    _edit_npz(_run(new2, "t10_st_s16") / "weights" / "u00050.npz", lambda z: z.update(conc_scale=np.asarray(8.0)))
    s16 = [x for x in _check(new2, ref2)["scale"] if x["arm"] == "t10_st_s16"][0]
    assert not s16["ok"] and s16["n_export_mismatches"] == 1
    new3, ref3 = _copy(wave, tmp_path / "c")                                                   # stage 1 is unscaled (D3)
    _edit_csv(_run(new3, "t1_st_s16") / "ms_updates.csv", _nth(70, "conc_scale", "16.0"))
    r3 = _check(new3, ref3)
    assert r3["summary"]["scale_ok"] == 11 and [x for x in r3["scale"] if not x["ok"]][0]["arm"] == "t1_st_s16"


def test_a_scale_where_none_is_expected_or_none_where_one_is_fails(wave, tmp_path):
    new, ref = _copy(wave, tmp_path)
    sched = {"stage": 2, "local_first": 21, "local_last": 40, "scale_first": 1.0, "scale_last": 2.0}
    _edit_json(_run(new, "relu_st_s1") / "run_config.json", lambda c: c.update(conc_scale_schedule=sched))
    r = _check(new, ref)
    assert not r["all_ok"] and [x["arm"] for x in r["scale"] if not x["ok"]] == ["relu_st_s1"]
    new2, ref2 = _copy(wave, tmp_path / "b")
    _edit_json(_run(new2, "t10_bb_s16") / "manifest.json", lambda m: m.update(conc_scale_schedule=None))
    r = _check(new2, ref2)
    assert not r["all_ok"] and [x["arm"] for x in r["scale"] if not x["ok"]] == ["t10_bb_s16"]


# ---------------------------------------------------------------------------------------------- C-INIT
def test_c_init_finds_a_changed_or_missing_init_digest(wave, tmp_path):
    new, ref = _copy(wave, tmp_path)
    _edit_json(_run(new, "t10_st_s1") / "manifest.json", lambda m: m.update(init_state_sha256="0" * 64))
    r = _check(new, ref)
    c = r["C_INIT"][0]
    assert not r["all_ok"] and r["summary"]["C_INIT_pass"] == 0 and c["n_distinct"] == 2 and c["digest"] is None
    assert c["digests"]["t10_st_s1"] == "0" * 64 and r["summary"]["C_NL_pass"] == 6 and r["summary"]["C_MS5_pass"] == 4
    new2, ref2 = _copy(wave, tmp_path / "b")                                  # one arm of another variant, starts and s
    _edit_json(_run(new2, "relu_bb_s16") / "manifest.json", lambda m: m.pop("init_state_sha256"))
    r = _check(new2, ref2)
    c = r["C_INIT"][0]
    assert r["summary"]["C_INIT_pass"] == 0 and c["n_with_digest"] == 11 and "relu_bb_s16" in c["errors"][0]
    new3, ref3 = _copy(wave, tmp_path / "c")
    _edit_json(_run(new3, "t1_st_s16") / "manifest.json", lambda m: m.update(init_state_sha256=""))
    assert _check(new3, ref3)["summary"]["C_INIT_pass"] == 0
    new4, ref4 = _copy(wave, tmp_path / "d")
    os.remove(_run(new4, "t10_bb_s1") / "manifest.json")
    r = _check(new4, ref4)
    assert r["summary"]["C_INIT_pass"] == 0 and not r["all_ok"]


# ---------------------------------------------------------------------------------------------- C-NL
def test_c_nl_finds_a_changed_export_series_value_or_stream_position(wave, tmp_path):
    new, ref = _copy(wave, tmp_path)
    _edit_npz(_run(new, "t10_st_s16") / "weights" / "u00010.npz", _ulp("actor.l1.weight"))
    r = _check(new, ref)
    bad = [x for x in r["C_NL"] if not x["pass"]]
    assert not r["all_ok"] and [(x["actor"], x["sampler"]) for x in bad] == [("t10", "st")] and bad[0]["weight_exports"] is False
    assert r["summary"]["C_NL_pass"] == 5 and r["summary"]["C_MS5_pass"] == 4
    new2, ref2 = _copy(wave, tmp_path / "b")
    _edit_json(_run(new2, "relu_bb_s16") / "train_history.json", lambda t: t["history"][10].update(policy_loss=1.0))
    bad = [x for x in _check(new2, ref2)["C_NL"] if not x["pass"]]
    assert [(x["actor"], x["sampler"], x["train_history"]) for x in bad] == [("relu", "bb", False)]
    new3, ref3 = _copy(wave, tmp_path / "c")
    _edit_csv(_run(new3, "t1_st_s16") / "ms_updates.csv", _nth(5, "rngpos_start", "0:0:0"))
    bad = [x for x in _check(new3, ref3)["C_NL"] if not x["pass"]]
    assert [(x["actor"], x["sampler"], x["updates_csv_and_stream_positions"]) for x in bad] == [("t1", "st", False)]
    new4, ref4 = _copy(wave, tmp_path / "d")
    _edit_csv(_run(new4, "t10_bb_s1") / "ms_checks_stage2.csv", _nth(0, "R", "0.5"))
    bad = [x for x in _check(new4, ref4)["C_NL"] if not x["pass"]]
    assert [(x["actor"], x["sampler"], x["check_rows"]) for x in bad] == [("t10", "bb", False)]


def test_c_nl_requires_the_first_export_after_the_ramp_start_to_differ(wave, tmp_path):
    new, ref = _copy(wave, tmp_path)
    shutil.copy(_run(new, "relu_bb_s1") / "weights" / "u00025.npz", _run(new, "relu_bb_s16") / "weights" / "u00025.npz")
    r = _check(new, ref)
    x = [x for x in r["C_NL"] if x["actor"] == "relu" and x["sampler"] == "bb"][0]
    assert x["ALL"] and not x["first_export_after_differs"] and not x["pass"] and not r["all_ok"]
    assert r["summary"]["C_NL_pass"] == 5


def test_c_nl_export_differs_half_compares_the_network_arrays_not_the_labels(wave, tmp_path):
    """The s = 16 export after the ramp start has a ``conc_scale`` entry that the s = 1 export lacks. With the network arrays
    of the s = 1 export under that label the two exports are the same network: C-NL must fail (review finding M1)."""
    new, _ = _copy(wave, tmp_path)
    s1, s16 = _run(new, "relu_bb_s1") / "weights" / "u00025.npz", _run(new, "relu_bb_s16") / "weights" / "u00025.npz"
    assert "conc_scale" in np.load(s16).files and "conc_scale" not in np.load(s1).files          # the label that made the old test pass
    nets = {k: v for k, v in dict(np.load(s1)).items() if k.startswith(("actor.", "critic."))}
    _edit_npz(s16, lambda z: z.update(nets))
    assert RC._weights_differ(dict(np.load(s1)), dict(np.load(s16))) is False
    x = RC.prefix_identity(_run(new, "relu_bb_s1"), _run(new, "relu_bb_s16"), RAMP[0], 5)
    assert x["ALL"] and x["first_export_after_differs"] is False                                  # identical networks: not "differs"
    assert RC._weights_differ({"actor.l1.weight": np.zeros(2, np.float32)}, {"actor.l1.weight": np.zeros(2, np.float32), "conc_scale": np.asarray(4.0)}) is False
    assert RC._weights_differ({"actor.l1.weight": np.zeros(2, np.float32)}, {"actor.l1.weight": np.ones(2, np.float32)}) is True


def test_c_nl_compares_the_variant_entry_of_the_exports(wave, tmp_path):
    """A relu / t10 export carries the string entry ``actor_variant``; the comparison handles it (``np.array_equal`` with
    ``equal_nan=True``, as ``r2_launch_checks`` uses it, raises on a string array in the installed numpy) and sees a change."""
    new, _ = wave
    assert RC.prefix_identity(_run(new, "relu_bb_s1"), _run(new, "relu_bb_s16"), RAMP[0], 5)["ALL"] is True
    assert RC.prefix_identity(_run(new, "t10_st_s1"), _run(new, "t10_st_s16"), RAMP[0], 5)["ALL"] is True
    new2, _ = _copy(wave, tmp_path)
    _edit_npz(_run(new2, "relu_bb_s16") / "weights" / "u00015.npz", lambda z: z.update(actor_variant=np.asarray("t10")))
    r = RC.prefix_identity(_run(new2, "relu_bb_s1"), _run(new2, "relu_bb_s16"), RAMP[0], 5)
    assert r["weight_exports"] is False and "actor_variant" in r["first_difference"] and not r["ALL"]


# ---------------------------------------------------------------------------------------------- C-MS5
def _pair(wave, tmp_path, arm: str = "t1_bb_s1") -> Tuple[Path, Path]:
    """Copies of one t1 run and of its MS-R2 reference run."""
    new, ref = wave
    a, b = tmp_path / "ref", tmp_path / "new"
    shutil.copytree(_run(ref, mc.R3_MS_R2_REFERENCE[arm]), a)
    shutil.copytree(_run(new, arm), b)
    return a, b


def _gates(fn: Callable[[Dict[str, Any]], None]) -> Callable[[Path], None]:
    return lambda d: _edit_json(d / "gates.json", fn)


def _hist(fn: Callable[[Dict[str, Any]], None]) -> Callable[[Path], None]:
    return lambda d: _edit_json(d / "train_history.json", fn)


def _wipe_gate_leaf(g: Dict[str, Any]) -> None:
    g["reported"]["end_of_stage2"]["final"]["e2_at_0"] += 1e-9


TAMPERS = {
    "an early export, one ulp": ("weight_exports", lambda d: _edit_npz(d / "weights" / "u00010.npz", _ulp("actor.l1.weight"))),
    "the last export (stage 1), one ulp": ("weight_exports",
                                           lambda d: _edit_npz(d / "weights" / "u00080.npz", _ulp("critic.out.bias"))),
    "an export array in float64": ("weight_exports", lambda d: _edit_npz(
        d / "weights" / "u00045.npz", lambda z: z.update({"actor.out.bias": z["actor.out.bias"].astype(np.float64)}))),
    "an export array with another shape": ("weight_exports", lambda d: _edit_npz(
        d / "weights" / "u00045.npz", lambda z: z.update({"actor.out.bias": z["actor.out.bias"].reshape(1, 2)}))),
    "a variant entry on a t1 export": ("weight_exports", lambda d: _edit_npz(
        d / "weights" / "u00035.npz", lambda z: z.update(actor_variant=np.asarray("relu")))),
    "a missing export": ("weight_exports", lambda d: os.remove(d / "weights" / "u00045.npz")),
    "an extra export": ("weight_exports", lambda d: shutil.copy(d / "weights" / "u00080.npz", d / "weights" / "u00085.npz")),
    "a history value": ("train_history", _hist(lambda t: t["history"][70].update(value_loss=t["history"][70]["value_loss"] + 1e-12))),
    "the snapshot log": ("train_history", _hist(lambda t: t["snapshots"].pop())),
    "a late updates value": ("updates_csv", lambda d: _edit_csv(d / "ms_updates.csv", _nth(75, "value_loss", "0.123"))),
    "the last stream position": ("updates_csv", lambda d: _edit_csv(d / "ms_updates.csv", _nth(79, "rngpos_minibatch", "0:0:0"))),
    "a dropped updates column": ("updates_csv", lambda d: _edit_csv(d / "ms_updates.csv", lambda rows: None, drop=("conc_scale",))),
    "a dropped updates row": ("updates_csv", lambda d: _edit_csv(d / "ms_updates.csv", lambda rows: rows.pop())),
    "a stage-1 check value": ("checks_stage1", lambda d: _edit_csv(d / "ms_checks_stage1.csv", _nth(1, "R", "0.25"))),
    "a stage-2 check value": ("checks_stage2", lambda d: _edit_csv(d / "ms_checks_stage2.csv", _nth(2, "stage2_rmse_pos_over_g2_0", "9"))),
    "a stage-2 check row count": ("checks_stage2", lambda d: _edit_csv(d / "ms_checks_stage2.csv", lambda rows: rows.pop())),
    "a freeze array": ("freeze_arrays", lambda d: _edit_npz(d / "freeze_stage1_development.npz", _ulp("v_t2_e_hat"))),
    "the final-tier freeze array": ("freeze_arrays", lambda d: _edit_npz(d / "freeze_stage2_final.npz", _ulp("v_t2_delta"))),
    "the bin maps (integer array)": ("binmaps_stage2", lambda d: _edit_npz(d / "ms_binmaps_stage2.npz", _ulp("local"))),
    "the bin maps (probability table of a stratified run)": ("binmaps_stage2", lambda d: _edit_npz(
        d / "ms_binmaps_stage2.npz", _ulp("probs_table")), "t1_st_s16"),
    "the continuation table": ("continuation_table_stage1", lambda d: _edit_npz(d / "continuation_table_stage1.npz", _ulp("values"))),
    "a gate value": ("gates", _gates(_wipe_gate_leaf)),
    "a gate verdict": ("gates", _gates(lambda g: g["G-A"].update({"pass": not g["G-A"]["pass"]}))),
    "a missing gate section": ("gates", _gates(lambda g: g.pop("S1"))),
}

#: changes the comparison must not see: wall clock, labels, git state and the files that are not part of the check
IGNORED = {
    "wall-clock column of the updates": lambda d: _edit_csv(d / "ms_updates.csv", _nth(3, "update_wall_sec", "99.0")),
    "wall-clock and label columns of the check tables": lambda d: (
        _edit_csv(d / "ms_checks_stage2.csv", lambda rows: [r.update(verifier_sec="9.9", run="x", arm="y") for r in rows]),
        _edit_csv(d / "ms_checks_stage1.csv", lambda rows: [r.update(verifier_sec="9.9", run="x", arm="y") for r in rows])),
    "labels, git state and timings of the gates": _gates(lambda g: (
        g.update(arm="x", run="y", commit="0" * 40, clean_tree=False),
        [g["budget"][k].update(wall_sec=1e9, process_cpu_sec=1e9) for k in g["budget"]],
        [g["continuation_tables"][k].update(build_seconds=1e9) for k in g["continuation_tables"]],
        [g["continuation_tables"][k]["meta"].update(build_seconds=1e9) for k in g["continuation_tables"]])),
    "the phase timing and labels of the history": _hist(lambda t: (t.update(run="x", arm="y"), t["phase_timing"].clear())),
    "manifest, status and config": lambda d: (
        _edit_json(d / "manifest.json", lambda m: m.update(commit="1" * 40, run="z", init_state_sha256="2" * 64)),
        _edit_json(d / "status.json", lambda s: s.update(run="z")),
        _edit_json(d / "run_config.json", lambda c: c.update(run="z", pilot="ms_r9"))),
}


@pytest.mark.parametrize("name", list(TAMPERS))
def test_c_ms5_finds_every_kind_of_difference_in_the_whole_run(wave, tmp_path, name):
    part, tamper, *arm = TAMPERS[name]
    arm = arm[0] if arm else "t1_bb_s1"
    ref, new = _pair(wave, tmp_path, arm)
    assert _ok(RC.whole_run_identity(ref, new))
    tamper(new)
    res = RC.whole_run_identity(ref, new)
    assert res["ALL"] is False and res[part] is False and res["first_difference"].startswith(part)
    assert all(res[p] for p in PARTS if p != part), {p: res[p] for p in PARTS}              # only that part fails
    ref2, new2 = _pair(wave, tmp_path / "again", arm)                                     # the same change on the reference side
    tamper(ref2)
    assert RC.whole_run_identity(ref2, new2)[part] is False


@pytest.mark.parametrize("name", list(IGNORED))
def test_c_ms5_does_not_look_at_what_it_lists_as_ignored(wave, tmp_path, name):
    ref, new = _pair(wave, tmp_path)
    IGNORED[name](new)
    res = RC.whole_run_identity(ref, new)
    assert _ok(res), res["first_difference"]


def test_c_ms5_fails_for_a_missing_reference_file_or_run(wave, tmp_path):
    ref, new = _pair(wave, tmp_path)
    os.remove(ref / "freeze_stage2_final.npz")
    res = RC.whole_run_identity(ref, new)
    assert not res["ALL"] and res["freeze_arrays"] is False and "FileNotFoundError" in res["first_difference"]
    assert all(res[p] for p in PARTS if p != "freeze_arrays")
    for f in ("gates.json", "ms_updates.csv", "train_history.json", "ms_checks_stage1.csv", "continuation_table_stage1.npz"):
        ref2, new2 = _pair(wave, tmp_path / f)
        os.remove(ref2 / f)
        assert not RC.whole_run_identity(ref2, new2)["ALL"], f
    shutil.rmtree(ref)
    res = RC.whole_run_identity(ref, new)
    assert not res["ALL"] and not any(res[p] for p in PARTS)
    assert not RC.whole_run_identity(tmp_path / "nowhere", tmp_path / "nowhere2")["ALL"]


def test_c_ms5_does_not_accept_a_run_of_another_arm(wave, tmp_path):
    new, ref = wave
    for arm, other in (("t1_bb_s1", "NL_st_s1"), ("t1_bb_s1", "NL_bb_s16"), ("t1_st_s16", "NL_bb_s16")):
        assert not RC.whole_run_identity(_run(ref, other), _run(new, arm))["ALL"], (arm, other)
    assert not RC.whole_run_identity(_run(ref, "NL_bb_s1"), _run(new, "relu_bb_s1"))["ALL"]      # another actor


def test_a_changed_export_value_in_a_t1_arm_fails_c_ms5_and_only_c_ms5(wave, tmp_path):
    new, ref = _copy(wave, tmp_path)
    _edit_npz(_run(new, "t1_st_s16") / "weights" / "u00070.npz", _ulp("actor.l2.bias"))        # a stage-1 export: C-NL does not see it
    r = _check(new, ref)
    bad = [x for x in r["C_MS5"] if not x["pass"]]
    assert not r["all_ok"] and [x["arm"] for x in bad] == ["t1_st_s16"] and bad[0]["weight_exports"] is False
    assert r["summary"]["C_MS5_pass"] == 3 and r["summary"]["C_NL_pass"] == 6 and r["summary"]["scale_ok"] == 12
    assert r["summary"]["C_INIT_pass"] == 1
    new2, ref2 = _copy(wave, tmp_path / "b")                                                    # the same on the reference side
    _edit_npz(_run(ref2, "NL_bb_s16") / "weights" / "u00010.npz", _ulp("critic.l1.bias"))
    r = _check(new2, ref2)
    assert [x["arm"] for x in r["C_MS5"] if not x["pass"]] == ["t1_bb_s16"] and r["summary"]["C_MS5_pass"] == 3


# ---------------------------------------------------------------------------------------------- missing runs, --actors
def test_a_missing_planned_run_is_a_failure(wave, tmp_path):
    new, ref = _copy(wave, tmp_path)
    shutil.rmtree(_run(new, "t10_bb_s16"))
    r = _check(new, ref)
    s = r["summary"]
    assert not r["all_ok"] and r["base"]["n_ok_per_check"]["status"] == 11
    assert (s["scale_ok"], s["scale_n"], s["C_INIT_pass"], s["C_NL_pass"], s["C_NL_n"], s["C_MS5_pass"]) == (11, 12, 0, 5, 6, 4)
    new2, ref2 = _copy(wave, tmp_path / "b")                                                    # a missing reference run
    shutil.rmtree(_run(ref2, "NL_st_s1"))
    r = _check(new2, ref2)
    assert not r["all_ok"] and r["summary"]["C_MS5_pass"] == 3 and r["summary"]["C_MS5_n"] == 4
    assert [x["arm"] for x in r["C_MS5"] if not x["pass"]] == ["t1_st_s1"]
    new3, ref3 = _copy(wave, tmp_path / "c")                                                    # a missing reference root
    shutil.rmtree(ref3)
    r = _check(new3, ref3)
    assert not r["all_ok"] and r["summary"]["C_MS5_pass"] == 0 and r["summary"]["C_MS5_n"] == 4
    new4, ref4 = _copy(wave, tmp_path / "d")                                                    # every run of the root missing
    shutil.rmtree(new4)
    r = _check(new4, ref4)
    s = r["summary"]
    assert not r["all_ok"] and (s["scale_n"], s["C_NL_n"], s["C_MS5_n"], s["C_INIT_n"]) == (12, 6, 4, 1)
    assert (s["scale_ok"], s["C_INIT_pass"], s["C_NL_pass"], s["C_MS5_pass"]) == (0, 0, 0, 0)


def test_the_actors_filter_scopes_the_checks_and_the_expected_counts(wave, tmp_path):
    new, ref = _copy(wave, tmp_path)
    for arm in mc.R3_ARMS:
        if arm.startswith("t10_"):
            shutil.rmtree(_run(new, arm))                                  # the premise check dropped t10: its arms were not run
    r = _check(new, ref, ("t1", "relu"))
    assert r["all_ok"] and r["actors"] == ["t1", "relu"] and r["arms"] == [a for a in mc.R3_ARMS if not a.startswith("t10_")]
    assert r["summary"] == {"scale_ok": 8, "scale_n": 8, "C_INIT_pass": 1, "C_INIT_n": 1, "C_NL_pass": 4, "C_NL_n": 4,
                            "C_MS5_pass": 4, "C_MS5_n": 4}
    assert r["expected"] == {"scale_n": 8, "C_INIT_n": 1, "C_NL_n": 4, "C_MS5_n": 4}
    assert r["C_INIT"][0]["n_arms"] == 8 and {x["actor"] for x in r["C_NL"]} == {"t1", "relu"}
    assert _check(new, ref, ("relu", "t1"))["summary"] == r["summary"]                          # the order does not matter
    r = _check(new, ref)                                                                       # not filtered: t10 is planned, so missing
    assert not r["all_ok"] and r["summary"]["scale_ok"] == 8 and r["summary"]["scale_n"] == 12
    assert r["summary"]["C_INIT_pass"] == 0 and r["summary"]["C_NL_pass"] == 4 and r["summary"]["C_NL_n"] == 6
    r = _check(new, ref, ("t1", "relu", "t10"))
    assert not r["all_ok"]
    r = _check(new, ref, ("relu", "t10"))                                                       # t10 is planned but missing
    assert not r["all_ok"] and r["summary"]["C_NL_pass"] == 2 and r["summary"]["C_NL_n"] == 4
    assert r["expected"]["C_MS5_n"] == 0 and r["summary"]["C_MS5_n"] == 0                       # no t1 arm planned: no reference
    with pytest.raises(ValueError):
        _check(new, ref, ("t1", "t2"))


def test_the_filter_checks_the_listed_actors_in_full_and_ignores_the_others(wave, tmp_path):
    new, ref = _copy(wave, tmp_path)
    _edit_npz(_run(new, "t10_bb_s16") / "weights" / "u00010.npz", _ulp("actor.l1.weight"))     # breaks t10 only
    assert not _check(new, ref)["all_ok"]
    r = _check(new, ref, ("t1", "relu"))
    assert r["all_ok"] and r["summary"]["C_NL_n"] == 4                                          # t10 is not planned: not looked at
    assert not _check(new, ref, ("t1", "t10"))["all_ok"]
    new2, ref2 = _copy(wave, tmp_path / "b")
    _edit_npz(_run(new2, "relu_bb_s16") / "weights" / "u00010.npz", _ulp("actor.l1.weight"))
    r = _check(new2, ref2, ("t1", "relu"))
    assert not r["all_ok"] and r["summary"]["C_NL_pass"] == 3
    r = _check(new2, ref2, ("relu",))                                                           # one actor alone: C-INIT over its 4 arms
    assert not r["all_ok"] and r["summary"]["C_INIT_pass"] == 1 and r["summary"]["scale_n"] == 4 and r["summary"]["C_MS5_n"] == 0
    assert _check(new2, ref2, ("t10",))["all_ok"]


def test_c_init_reads_the_manifests_of_the_planned_arms_only(wave, tmp_path):
    new, ref = _copy(wave, tmp_path)
    _edit_json(_run(new, "t1_bb_s1") / "manifest.json", lambda m: m.update(init_state_sha256="f" * 64))
    r = _check(new, ref, ("t1", "relu", "t10"))
    assert r["summary"]["C_INIT_pass"] == 0
    r = _check(new, ref, ("relu", "t10"))                                                       # t1 not run: its manifest is not read
    assert r["summary"]["C_INIT_pass"] == 1 and r["all_ok"]


# ---------------------------------------------------------------------------------------------- CLI
def _argv(new: Path, ref: Path, out: Path, *extra: str):
    return ["--root", str(new), "--ms-r2-pilot-root", str(ref), "--code-commit", "abc1234", "--qs", str(Q), "--seeds", str(SEED),
            "--ramp-first", "21", "--ramp-last", "40", "--export-every", "5", "--out", str(out), *extra]


def test_the_cli_exit_code_and_the_record(wave, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(LC1, "WEIGHTS_EVERY", 5)
    new, ref = wave
    out = tmp_path / "sub" / "lc.json"
    assert RC.main(_argv(new, ref, out)) == 0
    assert "FAILED" not in capsys.readouterr().out
    rec = json.load(open(out))
    assert rec["all_ok"] is True and rec["summary"]["C_MS5_pass"] == 4 and rec["tool"] == "tools/ms/r3_launch_checks.py"
    assert rec["code_commit"] == "abc1234" and rec["ms_r2_pilot_root"] == str(ref) and rec["actors"] == ["t1", "relu", "t10"]
    assert rec["ignored"]["files_not_compared_by_C_MS5"] and rec["expected"]["C_NL_n"] == 6
    n2, r2 = _copy(wave, tmp_path / "x")                                                        # a difference that C-MS5 sees
    _edit_npz(_run(n2, "t1_bb_s1") / "weights" / "u00070.npz", _ulp("actor.l2.bias"))
    assert RC.main(_argv(n2, r2, tmp_path / "bad.json")) == 2
    assert json.load(open(tmp_path / "bad.json"))["all_ok"] is False
    shown = capsys.readouterr().out
    assert "FAILED C_MS5 t1_bb_s1" in shown and "weight_exports: u00070.npz differs" in shown
    argv = _argv(new, ref, tmp_path / "commit.json")
    argv[argv.index("--code-commit") + 1] = "def5678"                                           # the manifests are at another commit
    assert RC.main(argv) == 2


def test_the_cli_requires_its_arguments_and_the_actors_flag_takes_a_comma_list(wave, tmp_path, monkeypatch):
    monkeypatch.setattr(LC1, "WEIGHTS_EVERY", 5)
    new, ref = _copy(wave, tmp_path)
    for arm in mc.R3_ARMS:
        if arm.startswith("t10_"):
            shutil.rmtree(_run(new, arm))                                                      # the t10 arms were not run
    out = tmp_path / "part.json"
    assert RC.main(_argv(new, ref, out)) == 2                                                  # planned by default, so missing
    assert RC.main(_argv(new, ref, out, "--actors", "t1,relu")) == 0
    assert json.load(open(out))["summary"]["C_NL_n"] == 4 and json.load(open(out))["actors"] == ["t1", "relu"]
    assert RC.main(_argv(new, ref, out, "--actors", "relu,t1")) == 0
    assert RC.main(_argv(new, ref, out, "--actors", "relu")) == 0
    assert RC.main(_argv(new, ref, out, "--actors", "t1,relu,t10")) == 2
    for bogus in ("t1,bogus", "", "t2", "t1,,relu"):
        with pytest.raises(SystemExit):
            RC.main(_argv(new, ref, out, "--actors", bogus))
    argv = _argv(new, ref, out)
    for missing in ("--root", "--ms-r2-pilot-root", "--code-commit", "--out"):
        args = [*argv]
        del args[args.index(missing):args.index(missing) + 2]
        with pytest.raises(SystemExit):
            RC.main(args)


# ---------------------------------------------------------------------------------------------- the exact comparisons
def test_the_array_comparison_is_exact_and_handles_strings_and_nan():
    z = {"a": np.array([1.0, np.nan], dtype=np.float32), "s": np.asarray("relu"), "i": np.arange(3), "b": np.array([True, False])}
    same = {k: v.copy() for k, v in z.items()}
    assert RC._arrays_diff(z, same) is None                                                     # NaN equals NaN, strings compare
    assert "keys differ" in RC._arrays_diff(z, {k: v for k, v in z.items() if k != "s"})
    assert "values differ" in RC._arrays_diff(z, {**same, "s": np.asarray("tanh")})            # same dtype, other string
    assert "<U3" in RC._arrays_diff(z, {**same, "s": np.asarray("t10")})                       # other string length: other dtype
    assert "float32" in RC._arrays_diff(z, {**same, "a": z["a"].astype(np.float64)})            # dtype
    assert "[1, 2]" in RC._arrays_diff(z, {**same, "a": np.array([[1.0, np.nan]], dtype=np.float32)})   # shape
    assert "values differ" in RC._arrays_diff(z, {**same, "i": np.array([0, 1, 3])})
    assert "values differ" in RC._arrays_diff(z, {**same, "b": np.array([True, True])})
    assert "values differ" in RC._arrays_diff(z, {**same, "a": np.array([1.0, 2.0], dtype=np.float32)})


def test_the_json_comparison_is_exact_and_the_flattening_keeps_empty_containers():
    nan = float("nan")
    assert RC._same_json({"a": [1, 2.5, nan, None, "x", True]}, {"a": [1, 2.5, float("nan"), None, "x", True]})
    assert not RC._same_json({"a": 1}, {"a": 1.0}) and not RC._same_json({"a": True}, {"a": 1})
    assert not RC._same_json({"a": [1]}, {"a": [1, 2]}) and not RC._same_json({"a": 1}, {"b": 1})
    assert not RC._same_json({"a": nan}, {"a": 1.0})
    flat = RC._flatten({"x": {"y": [1, {"z": 2}], "e": [], "f": {}}, "w": None})
    assert flat == {"x.y[0]": 1, "x.y[1].z": 2, "x.e": [], "x.f": {}, "w": None}
