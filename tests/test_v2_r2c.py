"""Round R2c tests: the scheduled peak-focused start sampler (``start_weights.local_first``) and the wave-S arms.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_v2_r2c.py -p no:cacheprovider -q

Sections: defaults are bit-identical (the C7 reference, the reduced locked run of the v2.0 lock commit, and
R2b's peak-focused arm under the R2b code), the window (draws and stream positions before ``local_first``
equal the bin-balanced sampler's exactly; the peak share from ``local_first`` on), the runner wiring and the
manifest, configuration refusals, the launcher arms and the C-R3 wave.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import launch_refine as L  # noqa: E402
import test_v2_r2b as T  # noqa: E402
from test_v2_r2b import (NEW_KEYS_AT_DEFAULT, _c7_equal, _p_config, _run_locked_small, _run_p,  # noqa: E402,F401
                         p_parent, tree_equal, v20_reference_tree)
from test_v2_r2b_gaps import _cfg_for_mode  # noqa: E402
from rng_alignment import base_config  # noqa: E402
from run.run_final_dp_br_round3_dense import ConfigError  # noqa: E402
from run.run_v2_stagewise import Run, execute, validate_config  # noqa: E402

R2B_CODE_COMMIT = "1ff99bd4"      # the R2b code commit: peak_focused as the owner accepted it
PEAK = {"scheme": "peak_focused", "peak_half_width": 20, "peak_share": 0.5}


def _peak(share: float = 0.5, local_first=None) -> dict:
    sw = {"scheme": "peak_focused", "peak_half_width": 20, "peak_share": share}
    if local_first is not None:
        sw["local_first"] = local_first
    return sw


def _run_dir(tmp_path, tag, cfg):
    d = str(tmp_path / tag)
    os.makedirs(d)
    return Run(cfg, d), d


def _stream_state(run: Run):
    return copy.deepcopy(run.rngs["start"].bit_generator.state)


# ====================================================================== 1. defaults are bit-identical
def test_start_weights_absent_equals_the_c7_reference(tmp_path):
    """The C7 regression pipeline (mode full: start_weights absent) is bit-identical to the C7 reference."""
    cfg = json.load(open(T.C7_CONFIG))
    cfg.update(NEW_KEYS_AT_DEFAULT)
    d = str(tmp_path / "c7")
    os.makedirs(d)
    assert execute(Run(cfg, d), cfg, d, "pytest") == 0
    _c7_equal(d)


def test_defaults_equal_the_v20_lock_commit_on_a_reduced_locked_run(v20_reference_tree, tmp_path):
    """The reduced-budget locked pipeline of the current tree ends in the training-relevant state of the v2.0
    lock commit's own code (state_end_A / state_end_B, all streams, the continuation table, the histories)."""
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


def test_bin_balanced_and_absent_are_the_same_phase_a_run(tmp_path):
    """start_weights absent, null and {scheme: bin_balanced}: the reduced phase-A run ends bit-identical."""
    ends = []
    for tag, sw in (("absent", "ABSENT"), ("none", None), ("bb", {"scheme": "bin_balanced"})):
        cfg = base_config()
        cfg["flags"]["reward_mode"] = "expected"
        if sw != "ABSENT":
            cfg["start_weights"] = sw
        run, d = _run_dir(tmp_path, tag, cfg)
        assert execute(run, cfg, d, "pytest") == 0
        ends.append(torch.load(os.path.join(d, "state_end_A.pt"), weights_only=False))
    for e in ends[1:]:
        for k in ("agent", "rng", "torch_generator_state", "counters"):
            assert tree_equal(ends[0][k], e[k]), k


_R2B_PHASE_A_RUNNER = textwrap.dedent('''
    import json, os, sys
    sys.path.insert(0, sys.argv[1])
    sys.path.insert(0, os.path.join(sys.argv[1], "tools", "v2"))
    for k in ("OMP", "MKL", "OPENBLAS"):
        os.environ[k + "_NUM_THREADS"] = "1"
    from rng_alignment import base_config
    from run.run_v2_stagewise import Run, execute
    cfg = base_config()
    cfg["flags"]["reward_mode"] = "expected"
    cfg["start_weights"] = json.loads(sys.argv[3])
    out = sys.argv[2]
    os.makedirs(out)
    sys.exit(execute(Run(cfg, out), cfg, out, "pytest"))
''')


def _phase_a_in_tree(code_root: str, out: str, sw: dict) -> dict:
    r = subprocess.run([sys.executable, "-B", "-c", _R2B_PHASE_A_RUNNER, code_root, out, json.dumps(sw)],
                       cwd=code_root, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return torch.load(os.path.join(out, "state_end_A.pt"), weights_only=False)


@pytest.fixture(scope="module")
def r2b_code_tree(tmp_path_factory):
    """The code of the R2b code commit, extracted with ``git archive`` (skipped without the commit)."""
    ok = subprocess.run(["git", "cat-file", "-e", f"{R2B_CODE_COMMIT}^{{commit}}"], cwd=ROOT, capture_output=True)
    if ok.returncode != 0:
        pytest.skip(f"commit {R2B_CODE_COMMIT} is not in this clone")
    d = tmp_path_factory.mktemp("r2b_code")
    arch = subprocess.run(["git", "archive", R2B_CODE_COMMIT, "agents", "envs", "run", "utils", "protocols", "tools/v2"],
                          cwd=ROOT, capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(d)], input=arch.stdout, check=True)
    for extra in ("config", "experiments"):
        if (ROOT / extra).exists():
            os.symlink(ROOT / extra, d / extra)
    return str(d)


def test_peak_focused_from_update_1_equals_the_r2b_implementation(r2b_code_tree, tmp_path):
    """peak_focused with local_first absent or 1 is R2b's A_peak50 sampler: the reduced phase-A run of the current
    tree ends in the training-relevant state of the same run under the R2b code commit."""
    ref = _phase_a_in_tree(r2b_code_tree, str(tmp_path / "r2b"), _peak())
    for tag, sw in (("absent", _peak()), ("one", _peak(local_first=1))):
        got = _phase_a_in_tree(str(ROOT), str(tmp_path / tag), sw)
        for k in ("agent", "rng", "torch_generator_state", "counters"):
            assert tree_equal(ref[k], got[k]), (tag, k)


# ====================================================================== 2. the window: draws and stream positions
def _chain(run: Run, n_updates: int, n: int = 512, balanced_only: bool = False):
    """The per-update ``start``-stream use of the phase loop (starts, then the role draw), starts returned."""
    out = []
    for local in range(1, n_updates + 1):
        if balanced_only:
            d = run.sampler.balanced(run.spec.T, n, run.rngs["start"])
        else:
            d = run.draw_starts(run.spec.T, n, local)
        run.rngs["start"].integers(0, 2, size=n)
        out.append(d)
    return out


@pytest.mark.parametrize("first", [2, 5, 801, 1201])
def test_updates_before_local_first_equal_the_bin_balanced_sampler_exactly(first, tmp_path):
    """Draws and the stream position of ALL local updates 1..first-1 (800 and 1200 for the two late arms) equal
    those of the locked sampler."""
    n_before = first - 1
    cfg = base_config()
    cfg["start_weights"] = _peak(local_first=first)
    run, _ = _run_dir(tmp_path, "w", cfg)
    ref, _ = _run_dir(tmp_path, "r", base_config())
    got = _chain(run, n_before)
    want = _chain(ref, n_before, balanced_only=True)
    assert all(np.array_equal(a, b) for a, b in zip(got, want))
    assert _stream_state(run) == _stream_state(ref)


def test_the_first_peak_focused_update_continues_from_the_stream_position_of_the_balanced_ones(tmp_path):
    """At local = first the draw is exactly ``peak_focused`` from the current position of the stream."""
    first = 4
    cfg = base_config()
    cfg["start_weights"] = _peak(0.4, first)
    run, _ = _run_dir(tmp_path, "w", cfg)
    ref, _ = _run_dir(tmp_path, "r", base_config())
    _chain(run, first - 1)
    _chain(ref, first - 1, balanced_only=True)
    alt = np.random.default_rng()                     # what bin-balanced would draw from the same position
    alt.bit_generator.state = _stream_state(ref)
    balanced = ref.sampler.balanced(ref.spec.T, 512, alt)
    got = run.draw_starts(run.spec.T, 512, first)
    want = ref.sampler.peak_focused(ref.spec.T, 512, ref.rngs["start"], 20.0, 0.4)
    assert np.array_equal(got, want) and not np.array_equal(got, balanced)
    assert _stream_state(run) == _stream_state(ref)


def test_an_absent_local_first_is_one(tmp_path):
    """R2b's configs carry no ``local_first``: absent, 1 and the plain peak-focused sampler are the same draws from
    the first update, on the same stream positions."""
    runs = []
    for lf in (None, 1):
        cfg = base_config()
        cfg["start_weights"] = _peak(0.4, lf)
        runs.append(_run_dir(tmp_path, f"w{lf}", cfg)[0])
    ref, _ = _run_dir(tmp_path, "r", base_config())
    want = []
    for _ in range(3):
        want.append(ref.sampler.peak_focused(ref.spec.T, 512, ref.rngs["start"], 20.0, 0.4))
        ref.rngs["start"].integers(0, 2, size=512)
    for run in runs:
        got = _chain(run, 3)
        assert all(np.array_equal(a, b) for a, b in zip(got, want))
        assert _stream_state(run) == _stream_state(ref)


def test_two_windows_draw_identical_starts_until_the_earlier_one_opens(tmp_path):
    """L=3 and L=6: updates 1-2 are bin-balanced in both and identical; update 3 is peak-focused only for L=3."""
    runs = []
    for first in (3, 6):
        cfg = base_config()
        cfg["start_weights"] = _peak(local_first=first)
        runs.append(_run_dir(tmp_path, f"w{first}", cfg)[0])
    a, b = _chain(runs[0], 5), _chain(runs[1], 5)
    assert all(np.array_equal(x, y) for x, y in zip(a[:2], b[:2]))
    assert not np.array_equal(a[2], b[2])


@pytest.mark.parametrize("share, first", [(0.35, 1), (0.40, 1), (0.50, 1201), (0.50, 801)])
def test_peak_share_from_local_first_on_and_the_locked_share_before(share, first, tmp_path):
    """10^6 starts: before local_first the peak set (four bins of 40) has 0.10, from local_first on `share`,
    each within 3 binomial standard errors."""
    cfg = base_config()
    cfg["start_weights"] = _peak(share, first)
    run, _ = _run_dir(tmp_path, "s", cfg)
    n = 1_000_000
    m = run.sampler.peak_set(run.spec.T, 20.0)
    edges = run.sampler.bin_edges(run.spec.T)

    def peak_share(d):
        b = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, edges.size - 2)
        return float(m[b].mean())

    if first > 1:
        p0 = m.sum() / m.size
        assert abs(p0 - 0.10) < 1e-15
        assert abs(peak_share(run.draw_starts(run.spec.T, n, first - 1)) - p0) <= 3 * np.sqrt(p0 * (1 - p0) / n)
    assert abs(peak_share(run.draw_starts(run.spec.T, n, first)) - share) <= 3 * np.sqrt(share * (1 - share) / n)
    assert abs(peak_share(run.draw_starts(run.spec.T, n, first + 7)) - share) <= 3 * np.sqrt(share * (1 - share) / n)


# ====================================================================== 3. runner wiring and the manifest
def test_phase_a_loop_passes_the_local_update_and_the_state_at_local_first_minus_one_is_the_baseline(
        monkeypatch, tmp_path):
    """Reduced phase-A run: draw_starts is called once per update with local = 1..6; the full state after
    update 3 of a run with local_first = 4 equals that of the bin-balanced run, the end states differ, and the
    manifest records start_weights with local_first."""
    calls = []
    orig = Run.draw_starts

    def rec(self, stage, n, local=1):
        calls.append(local)
        return orig(self, stage, n, local)

    monkeypatch.setattr(Run, "draw_starts", rec)
    out = {}
    for tag, sw in (("base", None), ("late", _peak(local_first=4))):
        cfg = base_config()
        cfg["flags"]["reward_mode"] = "expected"
        cfg["full_state_at"] = [3]
        if sw is not None:
            cfg["start_weights"] = sw
        calls.clear()
        run, d = _run_dir(tmp_path, tag, cfg)
        assert execute(run, cfg, d, "pytest") == 0
        assert calls == [1, 2, 3, 4, 5, 6], tag
        out[tag] = d
    a = torch.load(os.path.join(out["base"], "state_u00003.pt"), weights_only=False)
    b = torch.load(os.path.join(out["late"], "state_u00003.pt"), weights_only=False)
    for k in ("agent", "rng", "torch_generator_state", "counters"):
        assert tree_equal(a[k], b[k]), k
    ea = torch.load(os.path.join(out["base"], "state_end_A.pt"), weights_only=False)
    eb = torch.load(os.path.join(out["late"], "state_end_A.pt"), weights_only=False)
    assert not tree_equal(ea["agent"], eb["agent"])
    man = json.load(open(os.path.join(out["late"], "manifest.json")))
    assert man["start_weights"] == _peak(local_first=4)
    assert json.load(open(os.path.join(out["base"], "manifest.json")))["start_weights"] is None


def test_phase_p_loop_passes_the_local_update(monkeypatch, p_parent, tmp_path):
    calls = []
    orig = Run.draw_starts

    def rec(self, stage, n, local=1):
        calls.append((stage, local))
        return orig(self, stage, n, local)

    monkeypatch.setattr(Run, "draw_starts", rec)
    cfg = _p_config(cap=3)
    cfg["start_weights"] = _peak(local_first=2)
    run = _run_p(cfg, p_parent, str(tmp_path / "p"))
    assert calls == [(run.spec.T, 1), (run.spec.T, 2), (run.spec.T, 3)]


# ====================================================================== 4. configuration
@pytest.mark.parametrize("mode", ["phase_A", "phase_A_continue", "phase_P"])
def test_local_first_is_accepted_in_every_documented_mode(mode):
    cfg = _cfg_for_mode(mode)
    cfg["start_weights"] = _peak(local_first=801)
    validate_config(cfg)


@pytest.mark.parametrize("bad", [0, -3, 1.5, True, "4", None])
def test_local_first_must_be_a_positive_int(bad):
    cfg = base_config()
    cfg["start_weights"] = {**PEAK, "local_first": bad}
    with pytest.raises(ConfigError, match="local_first"):
        validate_config(cfg)


def test_bin_balanced_takes_no_local_first_and_unknown_keys_stay_refused():
    cfg = base_config()
    cfg["start_weights"] = {"scheme": "bin_balanced", "local_first": 5}
    with pytest.raises(ConfigError, match="no other key"):
        validate_config(cfg)
    cfg["start_weights"] = {**PEAK, "local_last": 9}
    with pytest.raises(ConfigError, match="exactly"):
        validate_config(cfg)


def test_mode_full_still_refuses_start_weights_with_local_first():
    cfg = json.load(open(T.C7_CONFIG))
    cfg.update({k: v for k, v in NEW_KEYS_AT_DEFAULT.items() if k != "start_weights"})
    cfg["start_weights"] = _peak(local_first=5)
    with pytest.raises(ConfigError, match="defined for modes"):
        validate_config(cfg)


# ====================================================================== 5. the launcher
@pytest.fixture(scope="module")
def planned(tmp_path_factory):
    root = tmp_path_factory.mktemp("r2c_root")
    return root, L.build_configs("r2c_waveS", (50, 60), L.DEFAULT_SEEDS, None, root, require_parents=False)


def test_wave_s_arm_table_and_job_counts(planned):
    root, jobs = planned
    assert list(L.R2C_WAVE_S_ARMS) == ["A_peak35", "A_peak40", "A_peak50_late400", "A_peak50_late800"]
    want = {"A_peak35": (0.35, 1), "A_peak40": (0.40, 1), "A_peak50_late400": (0.50, 1201), "A_peak50_late800": (0.50, 801)}
    for arm, (share, first) in want.items():
        assert L.R2C_WAVE_S_ARMS[arm]["r2b"]["start_weights"] == {
            "scheme": "peak_focused", "peak_half_width": 20, "peak_share": share, "local_first": first}
    assert len(jobs) == 80 and len({j.out_dir for j in jobs}) == 80
    assert {Path(j.out_dir).relative_to(root).parts[0] for j in jobs} == {"waveS"}
    assert Path(jobs[0].out_dir).relative_to(root).parts == ("waveS", "q50", "seed10501", "A_peak35")
    assert [j.cfg["arm"] for j in jobs[:4]] == list(L.R2C_WAVE_S_ARMS)          # q, seed, arm order
    assert L.R2C_ROOT == ROOT / "results" / "v2_refine_r2c"


def test_wave_s_configs_differ_from_the_parents_a_config_only_in_start_weights(planned):
    """The four arms against the R1 builder's own parents_A config (R2b keys at their defaults)."""
    root, jobs = planned
    refs = {(q, s): L.r2b_comparator_config("r2c_waveS", q, s, root)["A_parent"]
            for q in (50, 60) for s in L.DEFAULT_SEEDS}
    for j in jobs:
        c = j.cfg
        ref_arm, keys = L.R2C_EXPECTED_DIFFS["r2c_waveS"][c["arm"]]
        assert ref_arm == "A_parent" and keys == frozenset({"start_weights"})
        assert set(L.r2b_arm_diff(c, refs[(c["q"], c["seed"])])) == {"start_weights"}, (c["arm"], c["q"], c["seed"])
        assert c["start_weights"] == L.R2C_WAVE_S_ARMS[c["arm"]]["r2b"]["start_weights"]


def test_wave_s_is_a_full_phase_a_from_scratch_with_the_locked_settings(planned):
    _, jobs = planned
    for j in jobs:
        c = j.cfg
        assert c["mode"] == "phase_A" and c["fixed_budget"] is True and c["full_state_at"] == [1200]
        assert c["parent_checkpoint"] is None and c["parent_sha256"] is None
        assert c["flags"] == L.PHASE_A_FLAGS and c["budget_overrides"] == {"episodes_per_update": 512}
        assert [(w["phase"], w["start_lr"], w["end_lr"], w["local_first"], w["local_last"]) for w in c["lr_decay"]] == [
            ("A", 3e-4, 3e-5, 1201, 1600)]
        assert c["clamp_likelihood"] == "density" and c["pathwise_epochs"] == 1 and c["pathwise_minibatch"] is None
        assert c["pilot"] == "v2_refine_r2c_waveS"
    assert {j.cfg["seed"] for j in jobs} == set(L.DEFAULT_SEEDS) and {j.cfg["q"] for j in jobs} == {50, 60}


def test_wave_s_configs_validate(planned):
    _, jobs = planned
    rows = L.validate_jobs(jobs)
    assert len(rows) == 80 and [r for r in rows if not r["ok"]] == []
    assert all(not r["window_problems"] for r in rows)


def test_wave_s_launcher_refuses_unknown_arms_and_non_development_seeds(tmp_path):
    with pytest.raises(ValueError, match="not valid for wave"):
        L.build_configs("r2c_waveS", (50,), (10501,), ["A_peak25"], tmp_path)
    with pytest.raises(SystemExit):
        L.main(["--wave", "r2c_waveS", "--seeds", "40501", "--root", str(tmp_path), "--dry-run"])


def test_wave_s_dry_run_cli_writes_the_configs_and_validates(tmp_path):
    assert L.main(["--wave", "r2c_waveS", "--root", str(tmp_path), "--dry-run", "--code-commit", "HEAD"]) == 0
    cfgs = sorted((tmp_path / "waveS").glob("q*/seed*/*/run_config.json"))
    assert len(cfgs) == 80
    c = json.load(open(tmp_path / "waveS" / "q60" / "seed10510" / "A_peak50_late800" / "run_config.json"))
    assert c["start_weights"]["local_first"] == 801 and c["mode"] == "phase_A"
    rec = json.load(open(next((tmp_path / "waveS").glob("dryrun_*.json"))))
    assert rec["n_planned"] == 80 and rec["validation"]["n_invalid"] == 0


def test_c_r3_wave_is_the_unchanged_locked_entry_point_into_the_r2c_root(tmp_path):
    jobs = L.build_configs(L.R2C_REPRO, (50, 60), L.DEFAULT_SEEDS, None, tmp_path)
    assert len(jobs) == 20 and len({j.out_dir for j in jobs}) == 20
    assert all(L.is_locked_entry(j.cfg) and j.cfg["wave"] == "r2c_v20_repro" for j in jobs)
    assert Path(jobs[0].out_dir).relative_to(tmp_path).parts == ("v20_reproduction", "q50", "seed10501")
    cmd = L.job_command(*jobs[0])
    assert cmd[3].endswith("run/run_v2_T2_locked.py") and "--config" not in cmd
    assert cmd[cmd.index("--q") + 1] == "50" and cmd[cmd.index("--seed") + 1] == "10501"
    with pytest.raises(ValueError, match="no arms"):
        L.build_configs(L.R2C_REPRO, (50,), (10501,), ["x"], tmp_path)


@pytest.mark.parametrize("wave, root", [("r2c_waveS", L.R2C_ROOT), ("r2c_v20_repro", L.R2C_ROOT),
                                        ("r2b_waveA", L.R2B_ROOT), ("r2b_v20_repro", L.R2B_ROOT),
                                        ("parents_A", L.DEFAULT_ROOT)])
def test_the_launcher_defaults_the_results_root_by_round(monkeypatch, wave, root):
    """Without ``--root`` the R2c waves write to results/v2_refine_r2c, the R2b waves to results/v2_refine_r2b."""
    seen = []

    def stop(w, qs, seeds, arms, r, require_parents=True):
        seen.append((w, Path(r)))
        raise ValueError("stop here")

    monkeypatch.setattr(L, "build_configs", stop)
    with pytest.raises(SystemExit):
        L.main(["--wave", wave, "--dry-run"])
    assert seen == [(wave, root.resolve())]
    assert L.R2C_ROOT.name == "v2_refine_r2c" and L.R2B_ROOT.name == "v2_refine_r2b"


def test_r2c_does_not_change_the_r2b_launcher_tables(tmp_path):
    assert list(L.R2B_WAVEA_ARMS) == ["A_peak25", "A_peak50", "A_censored"]
    assert L.R2B_WAVES == ("r2b_waveA", "r2b_waveP") and set(L.R2B_EXPECTED_DIFFS) == set(L.R2B_WAVES)
    assert L.R2B_WAVEA_ARMS["A_peak50"]["r2b"]["start_weights"] == {
        "scheme": "peak_focused", "peak_half_width": 20, "peak_share": 0.50}      # no local_first written in R2b
    jobs = L.build_configs("r2b_waveA", (50,), (10501,), None, tmp_path, require_parents=False)
    assert Path(jobs[0].out_dir).relative_to(tmp_path).parts[0] == "waveA"
    assert "local_first" not in jobs[1].cfg["start_weights"]
