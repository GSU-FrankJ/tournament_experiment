"""MS-R3 RL-actor diagnostics (tools/ms/r3_actor_diagnostics.py).

First-layer d-weights, w_eff, the variant-aware reload, the tables, the input / output rules, a
genuine reduced-budget runner smoke per new variant (relu, t10) and a smoke on one real MS-R2 run.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest \
     tests/test_ms_r3_actor_diagnostics.py -p no:cacheprovider -q

Sections: (1) one export: variants, the effective d-weight, quantile and count arithmetic, w_eff by
hand, the variant-aware reload against the torch forward; (2) the tables built from per-export
rows (per-arm nearest-export rule, correlations, regime shares); (3) the tool on synthetic run
directories: layout errors, overwrite refusal, the screen, manifest, serial == pool, nothing
written outside --out; (4) a genuine mini-run of the runner per new variant; (5) one real MS-R2
run, read-only.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import r3_actor_diagnostics as D  # noqa: E402
from agents.ppo_curriculum import (  # noqa: E402
    BetaActor, CurriculumPPO, PPOConfig, mean_effort_numpy)
from envs.curriculum_env import GameSpec  # noqa: E402

GAME = {"w_h": 6, "w_l": 2, "k": 1.0 / 3500.0, "q": 50, "T": 2, "e_min": 0, "e_max": 100}
#: Where a real MS-R2 pilot may live: ``results/ms_r2`` of the repository this file belongs to (ROOT is derived from the
#: file location, no machine-specific path) and, if the environment names one, ``$MS_R2_RESULTS_DIR``. The real-run tests
#: below skip when neither holds the run.
MS_R2_BASES = tuple(
    [ROOT / "results" / "ms_r2"] + ([Path(os.environ["MS_R2_RESULTS_DIR"])] if os.environ.get("MS_R2_RESULTS_DIR") else []))
Row = Tuple[str, str, int, int, int, int, float, float]


def spec_for(q: int = 50) -> GameSpec:
    return GameSpec(**{**GAME, "q": q})


def e_star_by_hand(q: int) -> float:
    """DW / (4 k q) with DW = 4 and k = 1/3500: the tent peak, 70 at q = 50 and 58.33 at q = 60."""
    return 3500.0 / q


# ------------------------------------------------------------------------ synthetic exports
def make_agent(variant: str, seed: int = 3, d_col: Optional[Sequence[float]] = None,
               out_w_scale: float = 0.3, out_bias: Tuple[float, float] = (0.4, 0.1)
               ) -> CurriculumPPO:
    """A real ``CurriculumPPO`` with random (non-trivial) actor weights and the variant set."""
    agent = CurriculumPPO(PPOConfig(), torch.Generator().manual_seed(seed),
                          np.random.default_rng(seed))
    g = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        agent.actor.out.weight.copy_(torch.randn(2, 64, generator=g) * out_w_scale)
        agent.actor.out.bias.copy_(torch.tensor(out_bias))
        agent.actor.l1.bias.copy_(torch.randn(64, generator=g) * 0.2)
        if d_col is not None:
            agent.actor.l1.weight[:, 1] = torch.as_tensor(np.asarray(d_col, dtype=np.float32))
    agent.set_actor_variant(variant)
    return agent


def export(agent: CurriculumPPO, path: Path) -> Dict[str, np.ndarray]:
    """Write the agent's weights with the production exporter and read them back as a dict."""
    agent.export_weights_npz(str(path))
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def torch_e_hat(agent: CurriculumPPO, spec: GameSpec, d: float) -> float:
    """Beta-mean effort of the agent's torch actor (variant included) at gap d, stage 2."""
    a, b = agent.beta_params(spec.encode_obs(2, np.asarray([d])))
    a, b = float(a[0]), float(b[0])
    return spec.e_min + spec.e_range * a / (a + b)


def spike_d_col(m: float, j: int = 7) -> np.ndarray:
    """A d-weight column whose only nonzero entry is m (so max |w| = m)."""
    c = np.zeros(64)
    c[j] = m
    return c


def write_run(root: Path, q: int, seed: int, arm: str,
              plan: Sequence[Tuple[int, int, int, CurriculumPPO]], local_column: bool = True,
              listed: bool = True) -> Path:
    """``<root>/q<q>/seed<seed>/<arm>`` with run_config.json, ms_updates.csv and the exports.

    ``plan`` rows are ``(update, stage, local, agent)``; the CSV lists the updates of the plan
    (``listed=False`` leaves the last one out of the CSV).
    """
    d = Path(root) / f"q{q}" / f"seed{seed}" / arm
    (d / "weights").mkdir(parents=True)
    cfg = {"q": q, "seed": seed, "arm": arm,
           "record": {"game": {**GAME, "q": q}, "ppo": {"c_min": 100.0, "mu_clamp": 1e-06}}}
    (d / "run_config.json").write_text(json.dumps(cfg))
    listed_rows = list(plan) if listed else list(plan)[:-1]
    with open(d / "ms_updates.csv", "w") as fh:
        fh.write("update,stage,local,lr\n" if local_column else "update,stage,lr\n")
        for u, s, loc, _ in listed_rows:
            fh.write(f"{u},{s},{loc},0.0003\n" if local_column else f"{u},{s},0.0003\n")
    for u, _, _, agent in plan:
        agent.export_weights_npz(str(d / "weights" / f"u{u:05d}.npz"))
    return d


def small_plan(seed: int, n_terminal: int = 6, n_stage1: int = 2, variant: str = "t1",
               m0: float = 0.5) -> List[Tuple[int, int, int, CurriculumPPO]]:
    """Exports every 25 updates: ``n_terminal`` of stage 2 then ``n_stage1`` of stage 1.

    The max |w| of export i (0-based) is ``m0 * (i + 1)``.
    """
    plan = []
    for i in range(n_terminal + n_stage1):
        u = 25 * (i + 1)
        stage, loc = (2, u) if i < n_terminal else (1, u - 25 * n_terminal)
        agent = make_agent(variant, seed + i, d_col=spike_d_col(m0 * (i + 1)),
                           out_bias=(0.1 * i, 0.0))
        plan.append((u, stage, loc, agent))
    return plan


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory) -> Path:
    """8 runs (arms A_s1, A_s16 x q 50, 60 x seeds 1, 2): 6 terminal-stage + 2 stage-1 exports."""
    root = tmp_path_factory.mktemp("synth_root")
    s = 0
    for arm in ("A_s1", "A_s16"):
        for q in (50, 60):
            for seed in (1, 2):
                s += 11
                write_run(root, q, seed, arm, small_plan(s))
    return root


def run_tool(roots: Dict[str, Path], out: Path, **kw: Any) -> Dict[str, Any]:
    return D.run_diagnostics(roots, out, workers=kw.pop("workers", 1), **kw)


# ------------------------------------------------------------------------ 1. one export
def test_all_three_variants_export_and_t10_weights_are_reported_ten_fold(tmp_path):
    """t1 / relu / t10 exports with identical stored d-weights: t10 reports 10x, the others 1x."""
    col = np.linspace(-0.6, 0.9, 64)
    spec, e_star = spec_for(), e_star_by_hand(50)
    out = {}
    for v in D.ACTOR_VARIANTS:
        w = export(make_agent(v, d_col=col), tmp_path / f"{v}.npz")
        assert ("actor_variant" in w) == (v != "t1")          # the exporter names relu / t10 only
        assert D.export_variant(w) == v
        out[v] = D.export_stats(w, spec, e_star, terminal=True)
        assert out[v]["variant"] == v
    stored = np.abs(col.astype(np.float32).astype(np.float64))
    for key in ["max_abs_w_d"] + [f"absw_q{p}" for p in D.QUANTILES]:
        assert out["t1"][key] == out["relu"][key]
        assert out["t10"][key] == pytest.approx(10.0 * out["t1"][key], rel=1e-12)
    assert out["t1"]["max_abs_w_d"] == pytest.approx(stored.max())      # column 1, not column 0
    assert out["t10"]["max_abs_w_d"] == pytest.approx(10.0 * stored.max(), rel=1e-6)
    for t in D.COUNT_THRESHOLDS:                                  # counts use the effective weights
        assert out["t1"][f"n_absw_gt{t}"] == int((stored > t).sum())
        assert out["t10"][f"n_absw_gt{t}"] == int((10.0 * stored > t).sum())
    assert out["t10"]["n_absw_gt5"] > out["t1"]["n_absw_gt5"] == 0


def test_quantile_and_count_arithmetic_by_hand(tmp_path):
    """|w| = 0.0, 0.1, ..., 6.3 (shuffled, random signs): linear quantiles, strict counts."""
    rng = np.random.default_rng(0)
    absw = np.arange(64) / 10.0                                   # 1.0, 2.0, 3.0, 5.0 exact in f32
    col = rng.permutation(absw) * rng.choice([-1.0, 1.0], size=64)
    spec, e_star = spec_for(), e_star_by_hand(50)
    for variant in ("t1", "relu"):
        w = export(make_agent(variant, d_col=col), tmp_path / f"{variant}.npz")
        s = D.export_stats(w, spec, e_star, True)
        # position p * 63 on the sorted values v_i = i / 10 -> v_lo + frac * 0.1
        want = {0: 0.0, 10: 0.63, 25: 1.575, 50: 3.15, 75: 4.725, 90: 5.67, 100: 6.3}
        for p, v in want.items():
            assert s[f"absw_q{p}"] == pytest.approx(v, rel=1e-6, abs=1e-9), p
        assert s["max_abs_w_d"] == pytest.approx(6.3, rel=1e-6)
        # strictly above: the values equal to 1, 2, 3, 5 are not counted
        counts = (s["n_absw_gt1"], s["n_absw_gt2"], s["n_absw_gt3"], s["n_absw_gt5"])
        assert counts == (53, 43, 33, 13)


@pytest.mark.parametrize("q", [50, 60])
def test_w_eff_by_hand_on_a_constructed_actor(tmp_path, q):
    """Zero output layer, bias b0: the Beta mean is sigmoid(b0) everywhere, so e_hat is known."""
    spec = spec_for(q)
    mean = 0.66 if q == 50 else 0.5
    agent = make_agent("t1")
    with torch.no_grad():
        agent.actor.out.weight.zero_()
        agent.actor.out.bias.copy_(torch.tensor([math.log(mean / (1.0 - mean)), 0.0]))
    w = export(agent, tmp_path / "w.npz")
    e_hat = 100.0 * mean                                          # 66.0 (q 50) or 50.0 (q 60)
    e_star = e_star_by_hand(q)                                    # 70.0 or 58.333...
    s = D.export_stats(w, spec, D.tent_peak(spec), terminal=True)
    assert s["e_star_2_0"] == pytest.approx(e_star, rel=1e-12)    # the repo's DW / (4 k q)
    assert s["e_hat_2_0"] == pytest.approx(e_hat, abs=1e-4)
    gap = e_star - e_hat
    slope = e_star / (2.0 * q)                                    # the tent falls to 0 over 2q
    assert s["gap"] == pytest.approx(gap, abs=1e-4)
    assert s["w_eff"] == pytest.approx(gap / slope, abs=2e-4)
    assert s["w_eff"] == pytest.approx(2.0 * q * (1.0 - e_hat / e_star), abs=2e-4)
    assert s["w_eff"] == pytest.approx({50: 4.0 / 0.7, 60: 120.0 * (1.0 - 50.0 / e_star)}[q],
                                       abs=2e-4)
    # a non-terminal export has the same e_hat and gap but no w_eff
    n = D.export_stats(w, spec, D.tent_peak(spec), terminal=False)
    assert math.isnan(n["w_eff"]) and n["gap"] == s["gap"] and n["e_hat_2_0"] == s["e_hat_2_0"]


@pytest.mark.parametrize("variant", ["relu", "t10"])
def test_variant_aware_reload_is_the_torch_forward_not_the_tanh_forward(tmp_path, variant):
    """An export of relu / t10 is read as that variant: the torch forward, not tanh d / B."""
    spec = spec_for()
    agent = make_agent(variant, seed=9, d_col=np.random.default_rng(5).normal(0.0, 0.6, 64))
    w = export(agent, tmp_path / "w.npz")
    tanh_w = {k: v for k, v in w.items() if k != "actor_variant"}   # the pre-MS-R3 reload
    assert "actor_variant" in w and D.export_variant(tanh_w) == "t1"
    for d in (-60.0, -12.0, 0.0, 7.0, 40.0, 150.0):
        tool = D.e_hat_stage2(w, spec, d)
        assert tool == pytest.approx(torch_e_hat(agent, spec, d), abs=1e-4), (variant, d)
        e, _, _ = mean_effort_numpy(w, spec.encode_obs(2, np.asarray([d])), variant=variant)
        assert tool == pytest.approx(float(e[0]), abs=1e-9)
    for d in (-60.0, 7.0, 40.0, 150.0):                           # off d = 0: not the tanh forward
        assert abs(D.e_hat_stage2(w, spec, d) - D.e_hat_stage2(tanh_w, spec, d)) > 1e-2, d
    # at d = 0 the 10x input scaling is invisible (0 * 10 = 0): t10 equals the tanh reload there,
    # relu does not
    gap0 = abs(D.e_hat_stage2(w, spec, 0.0) - D.e_hat_stage2(tanh_w, spec, 0.0))
    assert (gap0 == 0.0) if variant == "t10" else (gap0 > 1e-2)
    s = D.export_stats(w, spec, D.tent_peak(spec), True)          # and the per-export columns
    assert s["variant"] == variant
    assert s["e_hat_2_0"] == pytest.approx(torch_e_hat(agent, spec, 0.0), abs=1e-4)


def test_export_of_an_unknown_variant_is_an_error_naming_the_run(tmp_path):
    w = export(make_agent("t1"), tmp_path / "w.npz")
    with pytest.raises(ValueError, match="unknown actor variant"):
        D.export_variant({**w, "actor_variant": np.asarray("gelu")})
    root = tmp_path / "root"
    d = write_run(root, 50, 1, "A", small_plan(1, 2, 0))
    z = dict(np.load(d / "weights" / "u00050.npz"))
    z["actor_variant"] = np.asarray("gelu")
    np.savez(d / "weights" / "u00050.npz", **z)
    with pytest.raises(D.DiagnosticsError, match=r"seed1/A.*unknown actor variant"):
        run_tool({"ms_r2_pilot": root}, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_the_loader_keeps_variant_and_concentration_scale_and_skips_the_critic(tmp_path):
    spec = spec_for()
    agent = make_agent("relu", seed=4)
    agent.actor.conc_scale = 4.0                                  # an annealed actor exports it
    path = tmp_path / "w.npz"
    export(agent, path)
    w = D._load_actor_arrays(path)
    assert "actor_variant" in w and "conc_scale" in w
    assert not any(k.startswith("critic") for k in w) and "actor.l1.weight" in w
    assert D.e_hat_stage2(w, spec, 25.0) == pytest.approx(torch_e_hat(agent, spec, 25.0), abs=1e-4)


# ------------------------------------------------------------------------ 2. tables
def rows_df(rows: Sequence[Row]) -> pd.DataFrame:
    """Per-export rows with the columns the tables read."""
    return pd.DataFrame(rows, columns=["source", "arm", "q", "seed", "update", "stage",
                                       "max_abs_w_d", "w_eff"])


def series_rows(source: str, arm: str, q: int, seed: int, terminal: Sequence[int],
                stage1: Sequence[int] = ()) -> List[Row]:
    """max |w| = update / 100 and w_eff = 1000 / update (terminal); stage-1 rows get NaN w_eff."""
    rows = [(source, arm, q, seed, u, 2, u / 100.0, 1000.0 / u) for u in terminal]
    return rows + [(source, arm, q, seed, u, 1, 9.0, float("nan")) for u in stage1]


def test_per_arm_uses_the_nearest_terminal_stage_export_not_after_the_target():
    rows = (series_rows("s", "A", 50, 1, [25, 50, 75, 100], stage1=[125, 150])    # budget 100
            + series_rows("s", "A", 50, 2, list(range(25, 301, 25)), stage1=[325, 350])
            + series_rows("s", "A", 50, 3, [75, 100, 125, 150]))        # first export after 50
    t = D.build_per_arm(rows_df(rows), targets=(50, 100, 200, 400))
    g = t.set_index("which")
    assert list(t["which"]) == ["u50", "u100", "u200", "u400", "final"]
    # target 50: runs 1 and 2 (exports 50, 50); run 3 has none <= 50 and does not enter the row
    assert g.loc["u50", "n_runs"] == 2
    assert (g.loc["u50", "update_used_min"], g.loc["u50", "update_used_max"]) == (50, 50)
    assert g.loc["u100", "n_runs"] == 3 and g.loc["u100", "update_used_median"] == 100
    # target 200: run 1 stops at its last terminal export 100 (the stage-1 exports 125, 150 do
    # not count), run 2 -> 200, run 3 -> 150
    r = g.loc["u200"]
    assert (r["update_used_min"], r["update_used_median"], r["update_used_max"]) == (100, 150, 200)
    assert r["median_max_abs_w_d"] == pytest.approx(np.median([1.0, 2.0, 1.5]))
    assert r["mean_max_abs_w_d"] == pytest.approx(np.mean([1.0, 2.0, 1.5]))
    assert r["median_w_eff"] == pytest.approx(np.median([10.0, 5.0, 1000.0 / 150]))
    assert r["mean_w_eff"] == pytest.approx(np.mean([10.0, 5.0, 1000.0 / 150]))
    # target 400: the last terminal export of each run (100, 300, 150)
    assert g.loc["u400", "update_used_max"] == 300
    assert g.loc["u400", "median_max_abs_w_d"] == pytest.approx(1.5)
    # final = the last terminal-stage export of each run; no target
    f = g.loc["final"]
    assert pd.isna(f["target_update"])
    assert (f["update_used_min"], f["update_used_max"]) == (100, 300)
    assert f["median_max_abs_w_d"] == pytest.approx(1.5)


def test_pearson_and_spearman_equal_scipy_and_are_nan_without_variation():
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    y = 0.5 * x + rng.normal(size=40)
    x[:5] = x[5]                                                  # ties
    assert D.pearson(x, y) == pytest.approx(stats.pearsonr(x, y)[0], rel=1e-12)
    assert D.spearman(x, y) == pytest.approx(stats.spearmanr(x, y)[0], rel=1e-12)
    assert math.isnan(D.pearson(np.ones(10), y[:10]))
    assert math.isnan(D.spearman(x[:10], np.full(10, 3.0)))
    assert math.isnan(D.pearson([1.0, 2.0], [3.0, 4.0]))          # fewer than 3 points


def test_relation_pooled_within_run_and_final_cross_seed():
    inc = np.arange(1.0, 11.0)
    series = {1: (inc, 100.0 / inc),              # monotone, nonlinear: Spearman -1, Pearson > -1
              2: (2.0 * inc, 10.0 - inc),         # linear: both -1
              3: (np.full(10, 4.0), inc)}         # constant max |w|: no correlation
    rows: List[Row] = []
    for seed, (mw, we) in series.items():
        rows += [("s", "A", 50, seed, 25 * (i + 1), 2, float(a), float(b))
                 for i, (a, b) in enumerate(zip(mw, we))]
    rows += [("s", "A", 50, 1, 275, 1, 5.0, float("nan"))]        # a stage-1 export is not used
    r = D.build_relation(rows_df(rows)).iloc[0]
    x = np.concatenate([inc, 2.0 * inc, np.full(10, 4.0)])
    y = np.concatenate([100.0 / inc, 10.0 - inc, inc])
    assert r["n_runs"] == 3 and r["n_pooled"] == 30
    assert r["pooled_spearman"] == pytest.approx(stats.spearmanr(x, y)[0], rel=1e-12)
    assert r["pooled_pearson"] == pytest.approx(stats.pearsonr(x, y)[0], rel=1e-12)
    p1 = stats.pearsonr(inc, 100.0 / inc)[0]
    assert r["n_runs_within"] == 2                                # run 3 has none
    assert r["within_run_median_spearman"] == pytest.approx(-1.0)
    assert r["within_run_median_pearson"] == pytest.approx((p1 - 1.0) / 2.0)   # median of two
    fx, fy = [10.0, 20.0, 4.0], [100.0 / 10.0, 10.0 - 10.0, 10.0]  # last terminal export, runs 1-3
    assert r["n_final"] == 3
    assert r["final_cross_seed_spearman"] == pytest.approx(stats.spearmanr(fx, fy)[0], rel=1e-12)
    assert r["final_cross_seed_pearson"] == pytest.approx(stats.pearsonr(fx, fy)[0], rel=1e-12)


def test_regime_shares_by_hand():
    # q 50, reference 2.0: terminal max |w| 0.5, 1.0, 2.0 (equal: at or below), 3.0; stage-1 1.5
    rows: List[Row] = [("s", "A", 50, 1, 25 * (i + 1), 2, m, 1.0)
                       for i, m in enumerate([0.5, 1.0, 2.0, 3.0])]
    rows += [("s", "A", 50, 1, 125, 1, 1.5, float("nan"))]
    rows += [("s", "A", 50, 2, 25 * (i + 1), 2, m, 1.0) for i, m in enumerate([4.0, 1.9])]
    rows += [("s", "A", 60, 1, 25, 2, 0.7, 1.0), ("s", "A", 60, 1, 50, 2, 0.71, 1.0)]
    t = D.build_regime(rows_df(rows), {50: 2.0, 60: 0.7}).set_index("q")
    a, b = t.loc[50], t.loc[60]
    assert (a["n_exports_all"], a["n_exports_terminal"], a["n_final"]) == (7, 6, 2)
    assert a["frac_terminal_at_or_below"] == pytest.approx(4 / 6)  # 0.5, 1.0, 2.0, 1.9 of 6
    assert a["frac_all_at_or_below"] == pytest.approx(5 / 7)       # plus the stage-1 1.5
    assert a["frac_final_at_or_below"] == pytest.approx(0.5)       # run 1 ends at 3.0, run 2 at 1.9
    assert a["reference_max_abs_w_d"] == 2.0
    assert b["frac_terminal_at_or_below"] == pytest.approx(0.5)
    assert b["frac_final_at_or_below"] == 0.0
    with pytest.raises(D.DiagnosticsError, match="no screen reference for q = 60"):
        D.build_regime(rows_df(rows), {50: 2.0})


# ------------------------------------------------------------------------ 3. the tool on runs
def test_outputs_and_columns_of_a_synthetic_root(synth_root, tmp_path):
    out = tmp_path / "out"
    m = run_tool({"ms_r2_pilot": synth_root}, out)
    assert sorted(os.listdir(out)) == [
        "manifest.json", "per_arm.csv", "per_export.csv", "relation.csv", "summary.txt"]
    df = pd.read_csv(out / "per_export.csv")
    assert tuple(df.columns) == D.EXPORT_COLUMNS
    assert len(df) == 8 * 8
    assert m["counts"]["total"] == {"runs": 8, "exports": 64, "terminal_stage_exports": 48}
    assert set(df["source"]) == {"ms_r2_pilot"} and set(df["variant"]) == {"t1"}
    s2, s1 = df[df.stage == 2], df[df.stage == 1]
    assert (s2.groupby(["arm", "q", "seed"]).size() == 6).all()
    assert np.isfinite(s2["w_eff"]).all() and s1["w_eff"].isna().all()
    assert np.isfinite(s1["e_hat_2_0"]).all() and np.isfinite(s1["gap"]).all()   # every export
    assert (df["local"] == np.where(df["stage"] == 2, df["update"], df["update"] - 150)).all()
    # max |w| of the synthetic plan is 0.5 * (export index + 1)
    run1 = df[(df.arm == "A_s1") & (df.q == 50) & (df.seed == 1)]
    assert run1["max_abs_w_d"].tolist() == pytest.approx([0.5 * (i + 1) for i in range(8)])
    # e* = 3500 / q, and w_eff = gap / (e* / 2q) row by row
    assert df["e_star_2_0"].to_numpy() == pytest.approx((3500.0 / df["q"]).to_numpy(), rel=1e-12)
    want = (s2["gap"] / (s2["e_star_2_0"] / (2.0 * s2["q"]))).to_numpy()
    assert s2["w_eff"].to_numpy() == pytest.approx(want, rel=1e-12)
    pa = pd.read_csv(out / "per_arm.csv")
    assert set(pa["which"]) == {"u400", "u1200", "u1600", "u2000", "u2400", "u2800", "final"}
    fin = pa[pa.which == "final"]
    assert (fin["update_used_max"] == 150).all() and (fin["n_runs"] == 2).all() and len(fin) == 4
    rel = pd.read_csv(out / "relation.csv")
    assert len(rel) == 4 and (rel["n_pooled"] == 12).all()
    assert not (out / "regime.csv").exists()
    text = (out / "summary.txt").read_text()
    assert "skipped" in text and "MS-R2 w_eff at the freeze" in text
    assert "PI preamble" in text and "4.85" in text


def test_missing_root_no_weights_and_incomplete_runs_are_errors(synth_root, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(D.DiagnosticsError, match="does not exist"):
        run_tool({"ms_r2_pilot": synth_root, "ms_r1_pilot": tmp_path / "nope"}, out)
    (tmp_path / "empty").mkdir()
    with pytest.raises(D.DiagnosticsError, match="no run directory"):
        run_tool({"ms_r1_base": tmp_path / "empty"}, out)
    with pytest.raises(D.DiagnosticsError, match="no root given"):
        run_tool({}, out)
    # a run without weights: config and CSV exist but there are no exports
    root = tmp_path / "r1"
    write_run(root, 50, 1, "A", small_plan(1, 2, 0))
    d = write_run(root, 50, 2, "A", small_plan(2, 2, 0))
    for f in (d / "weights").iterdir():
        f.unlink()
    with pytest.raises(D.DiagnosticsError, match=r"(?s)incomplete.*seed2/A: missing weights/u\*"):
        run_tool({"ms_r2_pilot": root}, out)
    # no ms_updates.csv; an empty run directory
    root2 = tmp_path / "r2"
    d2 = write_run(root2, 50, 1, "A", small_plan(1, 2, 0))
    (d2 / "ms_updates.csv").unlink()
    with pytest.raises(D.DiagnosticsError, match="missing ms_updates.csv"):
        run_tool({"ms_r2_pilot": root2}, out)
    root3 = tmp_path / "r3"
    (root3 / "q50" / "seed1" / "A").mkdir(parents=True)
    with pytest.raises(D.DiagnosticsError, match="missing run_config.json, ms_updates.csv, wei"):
        run_tool({"ms_r2_pilot": root3}, out)
    # an export that ms_updates.csv does not list
    root4 = tmp_path / "r4"
    write_run(root4, 50, 1, "A", small_plan(1, 3, 0), listed=False)
    with pytest.raises(D.DiagnosticsError, match="update 75 is not listed in ms_updates.csv"):
        run_tool({"ms_r2_pilot": root4}, out)
    assert not out.exists()                                       # no failure created anything


def test_cli_errors_return_2_and_write_nothing(synth_root, tmp_path, capsys):
    out = tmp_path / "out"
    assert D.main(["--ms-r1-pilot-root", str(tmp_path / "missing"), "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "error:" in err and "does not exist" in err
    assert D.main(["--out", str(out)]) == 2                       # no root at all
    assert "no root given" in capsys.readouterr().err
    assert D.main(["--ms-r2-pilot-root", str(synth_root), "--out", str(out),
                   "--workers", "0"]) == 2
    assert not out.exists()


def test_a_local_column_is_optional(tmp_path):
    root = tmp_path / "root"
    write_run(root, 50, 1, "A", small_plan(1, 3, 1), local_column=False)
    run_tool({"ms_r2_pilot": root}, tmp_path / "out")
    df = pd.read_csv(tmp_path / "out" / "per_export.csv")
    assert df["local"].isna().all() and df["stage"].tolist() == [2, 2, 2, 1]


def test_overwrite_is_refused_and_nothing_else_is_written(synth_root, tmp_path):
    out = tmp_path / "out"
    run_tool({"ms_r2_pilot": synth_root}, out)
    before = {n: (out / n).read_bytes() for n in os.listdir(out)}
    with pytest.raises(D.DiagnosticsError, match="refusing to overwrite"):
        run_tool({"ms_r2_pilot": synth_root}, out)
    assert {n: (out / n).read_bytes() for n in os.listdir(out)} == before
    # one pre-existing output file blocks the whole run before anything is written
    out2 = tmp_path / "out2"
    out2.mkdir()
    (out2 / "per_arm.csv").write_text("keep me")
    with pytest.raises(D.DiagnosticsError, match="per_arm.csv"):
        run_tool({"ms_r2_pilot": synth_root}, out2)
    assert os.listdir(out2) == ["per_arm.csv"] and (out2 / "per_arm.csv").read_text() == "keep me"
    # a regime.csv left by an earlier run also blocks (it would sit next to new tables)
    out3 = tmp_path / "out3"
    out3.mkdir()
    (out3 / "regime.csv").write_text("old")
    with pytest.raises(D.DiagnosticsError, match="regime.csv"):
        run_tool({"ms_r2_pilot": synth_root}, out3)
    f = tmp_path / "a_file"                                       # --out that is a file
    f.write_text("x")
    with pytest.raises(D.DiagnosticsError, match="not a directory"):
        run_tool({"ms_r2_pilot": synth_root}, f)
    assert D.main(["--ms-r2-pilot-root", str(synth_root), "--out", str(out)]) == 2


def test_nothing_is_written_to_the_roots(synth_root, tmp_path):
    def listing() -> List[Tuple[str, int, int]]:
        return sorted((str(p), p.stat().st_size, p.stat().st_mtime_ns)
                      for p in synth_root.rglob("*"))

    before = listing()
    run_tool({"ms_r2_pilot": synth_root}, tmp_path / "out", workers=2)
    assert listing() == before


def test_the_pool_gives_the_serial_result_byte_for_byte(synth_root, tmp_path):
    run_tool({"ms_r2_pilot": synth_root}, tmp_path / "w1", workers=1)
    run_tool({"ms_r2_pilot": synth_root}, tmp_path / "w3", workers=3)
    for name in ("per_export.csv", "per_arm.csv", "relation.csv"):
        assert (tmp_path / "w1" / name).read_bytes() == (tmp_path / "w3" / name).read_bytes()
    m1, m3 = (json.loads((tmp_path / d / "manifest.json").read_text()) for d in ("w1", "w3"))
    assert (m1["workers"], m3["workers"]) == (1, 3) and m1["counts"] == m3["counts"]
    one = tmp_path / "one"                                        # more workers than runs
    write_run(one, 50, 1, "A", small_plan(1, 2, 0))
    run_tool({"ms_r2_pilot": one}, tmp_path / "w8", workers=8)
    assert len(pd.read_csv(tmp_path / "w8" / "per_export.csv")) == 2


def test_manifest_records_roots_counts_and_environment(synth_root, tmp_path, capsys):
    other = tmp_path / "other"
    write_run(other, 50, 1, "MS_base", small_plan(3, 4, 2))
    out = tmp_path / "out"
    rc = D.main(["--ms-r2-pilot-root", str(synth_root), "--ms-r1-base-root", str(other),
                 "--out", str(out), "--workers", "2"])
    assert rc == 0 and "runs read" in capsys.readouterr().err
    m = json.loads((out / "manifest.json").read_text())
    assert m["roots"] == {"ms_r2_pilot": str(synth_root), "ms_r1_base": str(other)}
    assert m["counts"]["by_source"]["ms_r2_pilot"] == {
        "runs": 8, "exports": 64, "terminal_stage_exports": 48,
        "arms": {"A_s1": 4, "A_s16": 4}, "variants": ["t1"]}
    assert m["counts"]["by_source"]["ms_r1_base"]["runs"] == 1
    assert m["counts"]["total"]["runs"] == 9
    assert m["workers"] == 2 and m["python"] == platform.python_version()
    assert m["numpy"] == np.__version__
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True)
    assert m["git_head"] == head.strip()
    assert m["tool"] == str(Path(D.__file__).resolve()) and "--ms-r1-base-root" in m["command_line"]
    assert m["command_line"].endswith("--workers 2") and m["regime"]["status"] == "skipped"
    df = pd.read_csv(out / "per_export.csv")
    assert list(df["source"].unique()) == ["ms_r2_pilot", "ms_r1_base"]    # output source order


def _write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path / "summary_by_cell.csv", index=False)
    return path


def screen_csv(path: Path) -> Path:
    """The long-form schema of ``r3_supervised_screen`` (extended, cell_steps): the t1 / bb /
    56000 main-grid rows are the reference rows, among decoys of another step, start, actor and
    the extended cell's own 56,000-step checkpoint."""
    rows = []
    for q, vals in ((50, [1.0, 2.0, 3.0]), (60, [0.5, 0.7, 0.9])):
        for seed, v in enumerate(vals, 1):
            def row(actor: str, starts: str, steps: int, val: float,
                    ext: int = 0) -> Dict[str, Any]:
                return {"actor": actor, "starts": starts, "q": q, "seed": seed, "extended": ext,
                        "cell_steps": 224000 if ext else 56000, "steps": steps, "lr": 3e-4,
                        "max_abs_w_d": val}

            rows += [row("t1", "bb", 56000, v), row("t1", "bb", 48000, 9.0),
                     row("t1", "st", 56000, 9.0), row("relu", "bb", 56000, 9.0),
                     row("t1", "bb", 56000, 9.0, ext=1), row("t1", "bb", 224000, 9.0, ext=1)]
    return _write_rows(path, rows)


def reference_root(tmp_path: Path) -> Path:
    """One run per q: terminal max |w| 0.5, 1.0, 2.0, 3.0 (q 50) and 0.5, 0.7, 0.8 (q 60)."""
    root = tmp_path / "ref_root"
    for q, ms in ((50, [0.5, 1.0, 2.0, 3.0]), (60, [0.5, 0.7, 0.8])):
        plan = [(25 * (i + 1), 2, 25 * (i + 1),
                 make_agent("t1", 20 + i, d_col=spike_d_col(m), out_bias=(0.1 * i, 0.0)))
                for i, m in enumerate(ms)]
        plan.append((25 * (len(ms) + 1), 1, 1, make_agent("t1", 40, d_col=spike_d_col(1.5))))
        write_run(root, q, 1, "A", plan)
    return root


def test_screen_dir_absent_or_without_the_file_skips_the_table_and_says_so(synth_root, tmp_path):
    m = run_tool({"ms_r2_pilot": synth_root}, tmp_path / "a")
    assert m["regime"]["status"] == "skipped" and "--screen-dir not given" in m["regime"]["reason"]
    assert not (tmp_path / "a" / "regime.csv").exists()
    assert "skipped" in (tmp_path / "a" / "summary.txt").read_text()
    empty = tmp_path / "screen_empty"
    empty.mkdir()
    m = run_tool({"ms_r2_pilot": synth_root}, tmp_path / "b", screen_dir=empty)
    assert m["regime"]["status"] == "skipped"
    assert "summary_by_cell.csv not found" in m["regime"]["reason"]
    on_disk = json.loads((tmp_path / "b" / "manifest.json").read_text())
    assert on_disk["regime"]["status"] == "skipped"
    assert not (tmp_path / "b" / "regime.csv").exists()
    m = run_tool({"ms_r2_pilot": synth_root}, tmp_path / "c", screen_dir=tmp_path / "no_such_dir")
    assert m["regime"]["status"] == "skipped"


def test_screen_dir_present_gives_the_regime_table(tmp_path):
    root, screen = reference_root(tmp_path), screen_csv(tmp_path / "screen")
    out = tmp_path / "out"
    m = run_tool({"ms_r2_pilot": root}, out, screen_dir=screen)
    assert m["regime"]["status"] == "computed"
    assert m["regime"]["reference_max_abs_w_d_by_q"] == {"50": 2.0, "60": 0.7}   # t1 / bb / 56000
    assert m["regime"]["n_rows_by_q"] == {"50": 3, "60": 3}
    t = pd.read_csv(out / "regime.csv").set_index("q")
    a, b = t.loc[50], t.loc[60]
    assert (a["reference_max_abs_w_d"], b["reference_max_abs_w_d"]) == (2.0, 0.7)
    assert (a["n_exports_all"], a["n_exports_terminal"], a["n_final"]) == (5, 4, 1)
    assert a["frac_terminal_at_or_below"] == pytest.approx(0.75)  # 0.5, 1.0, 2.0 (equal) of 4
    assert a["frac_all_at_or_below"] == pytest.approx(0.8)        # plus the stage-1 export at 1.5
    assert a["frac_final_at_or_below"] == 0.0                     # final 3.0 > 2.0
    assert b["frac_terminal_at_or_below"] == pytest.approx(2 / 3)  # 0.5, 0.7 (equal), not 0.8
    assert b["frac_final_at_or_below"] == 0.0                     # final 0.8 > 0.7
    assert b["frac_all_at_or_below"] == pytest.approx(2 / 4)      # 0.5, 0.7 of 0.5, 0.7, 0.8, 1.5
    text = (out / "summary.txt").read_text()
    assert "Share of exports with max |w| <= the screen reference" in text


@pytest.mark.parametrize("label", ["bb", "BB", "bin_balanced", "Bin-Balanced", "balanced"])
def test_screen_bin_balanced_labels_are_matched_loosely(tmp_path, label):
    rows = [{"actor": "T1", "starts": label, "q": 50, "seed": s, "steps": 56000.0,
             "max_abs_w_d": v} for s, v in enumerate([1.0, 2.0, 4.0], 1)]
    ref, info = D.read_screen_reference(_write_rows(tmp_path / "s", rows))
    assert ref == {50: 2.0} and info["status"] == "computed"


def test_screen_reading_is_defensive_about_labels_columns_and_duplicates(tmp_path):
    base = {"actor": "t1", "starts": "bb", "q": 50, "steps": 56000}
    with pytest.raises(D.DiagnosticsError, match="no row with actor t1"):    # lists what is there
        D.read_screen_reference(_write_rows(
            tmp_path / "a", [{**base, "starts": "xx", "seed": 1, "max_abs_w_d": 1.0}]))
    with pytest.raises(D.DiagnosticsError, match="missing column"):
        D.read_screen_reference(_write_rows(tmp_path / "b", [base]))
    ref, _ = D.read_screen_reference(_write_rows(          # column names are case-insensitive
        tmp_path / "c", [{"Actor": "t1", "STARTS": "bb", "Q": 50, "Steps": 56000,
                          "Max_Abs_W_D": 1.25}]))
    assert ref == {50: 1.25}
    # two cells (main and extended) with rows at 56000 steps for the same (q, seed): ambiguous
    rows = [{**base, "seed": s, "cell": c, "max_abs_w_d": v}
            for s in (1, 2) for c, v in (("main", 1.0 * s), ("ext", 9.0))]
    d = _write_rows(tmp_path / "d", rows)
    with pytest.raises(D.DiagnosticsError, match="several selected rows"):
        D.read_screen_reference(d)
    ref, info = D.read_screen_reference(d, ["cell=main"])
    assert ref == {50: 1.5} and info["filters"] == {"cell": "main"}
    with pytest.raises(D.DiagnosticsError, match="not in"):
        D.read_screen_reference(d, ["nope=1"])
    with pytest.raises(D.DiagnosticsError, match="COL=VALUE"):
        D.parse_screen_filters(["cell"])
    rows = [{**base, "starts": "stratified", "seed": 1, "max_abs_w_d": 7.0}]   # starts= filter
    ref, _ = D.read_screen_reference(_write_rows(tmp_path / "e", rows), ["starts=stratified"])
    assert ref == {50: 7.0}


def test_the_extended_cell_is_left_out_by_default_and_selectable(tmp_path):
    base = {"actor": "t1", "starts": "bb", "q": 50, "steps": 56000}
    rows = [{**base, "seed": s, "extended": e, "cell_steps": 224000 if e else 56000,
             "max_abs_w_d": 9.0 if e else float(s)} for s in (1, 2, 3) for e in (0, 1)]
    d = _write_rows(tmp_path / "s", rows)
    ref, info = D.read_screen_reference(d)
    assert ref == {50: 2.0} and info["n_rows_by_q"] == {"50": 3} and info["default_filters"]
    ref, info = D.read_screen_reference(d, ["extended=1"])       # an explicit filter wins
    assert ref == {50: 9.0} and "default_filters" not in info
    rows = [{**r, "extended": bool(r["extended"])} for r in rows]    # True / False spelling
    ref, _ = D.read_screen_reference(_write_rows(tmp_path / "t", rows))
    assert ref == {50: 2.0}


def test_the_reader_matches_the_row_schema_of_the_supervised_screen_tool():
    """The columns and labels this tool reads are the ones r3_supervised_screen writes."""
    S = pytest.importorskip("r3_supervised_screen")
    need = {"actor", "starts", "q", "seed", "steps", "max_abs_w_d", "extended", "cell_steps"}
    assert need <= set(S.ROW_COLS)
    assert "t1" in S.ACTOR_VARIANTS and "bb" in S.STARTS and "bb" in D.SCREEN_BB_LABELS
    assert S.STEPS_MAIN == D.SCREEN_STEPS


def test_a_screen_without_a_q_of_the_rl_runs_is_an_error_before_any_work(tmp_path):
    root = reference_root(tmp_path)
    only50 = _write_rows(tmp_path / "s", [{"actor": "t1", "starts": "bb", "q": 50, "seed": 1,
                                          "steps": 56000, "max_abs_w_d": 1.0}])
    with pytest.raises(D.DiagnosticsError, match=r"no row for q = \[60\]"):
        run_tool({"ms_r2_pilot": root}, tmp_path / "out", screen_dir=only50)
    assert not (tmp_path / "out").exists()


def test_summary_gives_the_six_arm_mean_next_to_the_preamble_figure(tmp_path, capsys):
    """Six-arm mean of w_eff at the freeze per q, next to 4.85 / 5.33 (numbers only)."""
    root = tmp_path / "root"
    expected: Dict[int, List[float]] = {50: [], 60: []}
    arms = ["NL_bb_s1", "NL_bb_s4", "NL_bb_s16", "NL_st_s1", "NL_st_s4", "NL_st_s16"]
    for arm_i, arm in enumerate(arms):
        for q in (50, 60):
            for seed in (1, 2):
                # final terminal export: effort fraction frac < e*/100 at both q -> w_eff by hand
                frac = 0.55 - 0.01 * arm_i - 0.002 * seed
                agent = make_agent("t1", 3 + arm_i + seed, d_col=spike_d_col(1.0))
                with torch.no_grad():
                    agent.actor.out.weight.zero_()
                    agent.actor.out.bias.copy_(torch.tensor([math.log(frac / (1.0 - frac)), 0.0]))
                plan = [(25, 2, 25, agent), (50, 1, 25, make_agent("t1", 90))]
                write_run(root, q, seed, arm, plan)
                expected[q].append(2.0 * q * (1.0 - 100.0 * frac / e_star_by_hand(q)))
    out = tmp_path / "out"
    assert D.main(["--ms-r2-pilot-root", str(root), "--out", str(out)]) == 0
    capsys.readouterr()
    fin = pd.read_csv(out / "per_arm.csv").query("which == 'final'")
    for q in (50, 60):
        arm_means = fin[fin.q == q]["mean_w_eff"].to_numpy()
        assert len(arm_means) == 6
        assert arm_means.mean() == pytest.approx(np.mean(expected[q]), abs=2e-3)
    text = (out / "summary.txt").read_text()
    pat = r"\s*(50|60)\s+(\d+)\s+(-?[\d.]+)\s+([\d.]+)\s*"
    six = [m.groups() for m in (re.fullmatch(pat, ln) for ln in text.splitlines()) if m]
    assert [(g[0], g[1], g[3]) for g in six] == [("50", "6", "4.85"), ("60", "6", "5.33")]
    for g in six:                                    # q, number of arms, mean w_eff, preamble
        assert float(g[2]) == pytest.approx(np.mean(expected[int(g[0])]), abs=2e-3)


# ------------------------------------------------------------------------ 4. mini-run per variant
@pytest.fixture(scope="module", params=["relu", "t10"])
def mini_run(request, tmp_path_factory) -> Tuple[str, Path]:
    """Reduced-budget run of the real runner: 20 terminal + 5 stage-1 updates, export every 5."""
    from test_ms_r2_runner import reduced_nl, run_cfg
    variant = request.param
    root = tmp_path_factory.mktemp(f"mini_{variant}")
    d = root / "q50" / "seed10501" / "NL_bb_s1"
    cfg = reduced_nl("NL_bb_s1", str(d), n2=20, n1=5, const_to=15, K=5, weights_every=5)
    cfg["actor_variant"] = variant
    run_cfg(cfg, str(d))
    return variant, root


def torch_actor_from_export(w: Dict[str, np.ndarray], variant: str) -> BetaActor:
    net = BetaActor(64, 100.0, 1e-6, torch.Generator().manual_seed(0))
    net.load_state_dict({k[len("actor."):]: torch.from_numpy(v)
                         for k, v in w.items() if k.startswith("actor.")})
    net.variant = variant
    return net


def torch_mean(net: BetaActor, spec: GameSpec, d: float) -> float:
    with torch.no_grad():
        a, b = net(torch.as_tensor(spec.encode_obs(2, np.asarray([d]))))
    return float(100.0 * a[0] / (a[0] + b[0]))


def test_mini_run_exports_are_read_as_their_variant(mini_run, tmp_path):
    variant, root = mini_run
    run = root / "q50" / "seed10501" / "NL_bb_s1"
    spec = spec_for()
    out = tmp_path / "out"
    m = run_tool({"ms_r2_pilot": root}, out)
    df = pd.read_csv(out / "per_export.csv")
    assert df["update"].tolist() == [5, 10, 15, 20, 25]
    assert df["stage"].tolist() == [2, 2, 2, 2, 1] and df["local"].tolist() == [5, 10, 15, 20, 5]
    assert set(df["variant"]) == {variant}
    assert m["counts"]["by_source"]["ms_r2_pilot"]["variants"] == [variant]
    assert df["w_eff"].iloc[:4].notna().all() and np.isnan(df["w_eff"].iloc[4])
    for row in df.itertuples():
        with np.load(run / "weights" / f"u{row.update:05d}.npz") as z:
            w = {k: z[k] for k in z.files}
        assert str(w["actor_variant"]) == variant
        raw = np.abs(w["actor.l1.weight"][:, 1].astype(np.float64))
        scale = 10.0 if variant == "t10" else 1.0
        assert row.max_abs_w_d == pytest.approx(scale * raw.max(), rel=1e-12)
        assert row.n_absw_gt1 == int((scale * raw > 1.0).sum())
        # the tool's e_hat is the torch forward of the exported actor with the variant ...
        net, tanh_net = torch_actor_from_export(w, variant), torch_actor_from_export(w, "t1")
        assert row.e_hat_2_0 == pytest.approx(torch_mean(net, spec, 0.0), abs=1e-4)
        # ... and not the tanh d / B forward (relu differs at d = 0; t10 only off d = 0)
        if variant == "relu":
            assert abs(row.e_hat_2_0 - torch_mean(tanh_net, spec, 0.0)) > 1e-3
        for d in (-30.0, 12.0, 80.0):
            tool = D.e_hat_stage2(w, spec, d)
            assert tool == pytest.approx(torch_mean(net, spec, d), abs=1e-4)
            assert abs(tool - torch_mean(tanh_net, spec, d)) > 1e-3


def test_mini_run_terminal_export_equals_the_verifiers_freeze_policy(mini_run):
    """The last terminal-stage export is the verifier's frozen actor: reload == verifier e_hat_2."""
    variant, root = mini_run
    run = root / "q50" / "seed10501" / "NL_bb_s1"
    spec = spec_for()
    with np.load(run / "weights" / "u00020.npz") as z:
        w = {k: z[k] for k in z.files}
    fz = np.load(run / "freeze_stage2_final.npz")
    grid, e2 = fz["recovery_d_grid"], fz["recovery_e2"]
    for d in (-60.0, -20.0, 0.0, 5.0, 40.0, 120.0):
        i = int(np.argmin(np.abs(grid - d)))
        assert grid[i] == pytest.approx(d, abs=1e-9)
        assert D.e_hat_stage2(w, spec, d) == pytest.approx(float(e2[i]), abs=1e-4), (variant, d)


def test_mini_run_against_the_g2_at_0_of_the_freeze(mini_run, tmp_path):
    variant, root = mini_run
    run = root / "q50" / "seed10501" / "NL_bb_s1"
    fz = np.load(run / "freeze_stage2_final.npz")
    i0 = int(np.argmin(np.abs(fz["recovery_d_grid"])))
    g2, e2 = float(fz["recovery_g2"][i0]), float(fz["recovery_e2"][i0])
    out = tmp_path / "out"
    run_tool({"ms_r2_pilot": root}, out)
    last = pd.read_csv(out / "per_export.csv").query("stage == 2").iloc[-1]
    assert last["e_star_2_0"] == pytest.approx(g2, rel=1e-12)     # the verifier's g2_at_0
    assert last["e_hat_2_0"] == pytest.approx(e2, abs=1e-4)
    assert last["w_eff"] == pytest.approx((g2 - e2) / (g2 / 100.0), abs=2e-4)


# ------------------------------------------------------------------------ 5. one real MS-R2 run
def _real_ms_r2() -> Optional[Path]:
    for base in MS_R2_BASES:
        run = base / "pilot" / "q50" / "seed10501" / "NL_bb_s1"
        has_per_run = (base / "analysis" / "per_run.csv").is_file()
        if (run / "weights" / "u02800.npz").is_file() and has_per_run:
            return base
    return None


@pytest.fixture(scope="module")
def real_run_output(tmp_path_factory) -> Tuple[pd.DataFrame, Path, Dict[str, Any]]:
    base = _real_ms_r2()
    if base is None:
        pytest.skip("MS-R2 run q50/seed10501/NL_bb_s1 or results/ms_r2/analysis/per_run.csv absent")
    root = tmp_path_factory.mktemp("real_root")
    (root / "q50" / "seed10501").mkdir(parents=True)
    run = base / "pilot" / "q50" / "seed10501" / "NL_bb_s1"
    os.symlink(run, root / "q50" / "seed10501" / "NL_bb_s1")      # the tool reads the run read-only

    def listing() -> List[Tuple[str, int, int]]:
        return sorted((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in run.rglob("*"))

    before = listing()
    out = tmp_path_factory.mktemp("real_out") / "out"
    m = run_tool({"ms_r2_pilot": root}, out)
    assert listing() == before                                    # nothing in the run changed
    return pd.read_csv(out / "per_export.csv"), base, m


def test_real_ms_r2_run_structure(real_run_output):
    df, _, m = real_run_output
    assert len(df) == 136 and m["counts"]["total"]["terminal_stage_exports"] == 112
    assert df["update"].tolist() == list(range(25, 3401, 25)) and set(df["variant"]) == {"t1"}
    assert (df[df["update"] <= 2800]["stage"] == 2).all()
    assert (df[df["update"] > 2800]["stage"] == 1).all()
    assert df[df.stage == 2]["w_eff"].notna().all() and df[df.stage == 1]["w_eff"].isna().all()
    assert (df["e_star_2_0"] == 70.0).all()
    # the live actor's output at the stage-2 observation is no longer the terminal policy once
    # stage 1 trains (the reason w_eff is NaN there)
    e_end = df[df["update"] == 3400]["e_hat_2_0"].iloc[0]
    assert abs(e_end - df[df["update"] == 2800]["e_hat_2_0"].iloc[0]) > 1.0


def test_real_ms_r2_w_eff_at_the_freeze_equals_the_verifiers_per_run_values(real_run_output):
    """Last w_eff == gap / (e*(0) / 2q) from per_run.csv e2_at_0 / g2_at_0.

    e_hat is a float32 reload of the u2800 export against the verifier's float64 evaluation of
    the frozen actor, so the tolerance is 1e-4 effort units; the observed difference is printed.
    """
    df, base, _ = real_run_output
    pr = pd.read_csv(base / "analysis" / "per_run.csv", float_precision="round_trip")
    sel = (pr.arm == "NL_bb_s1") & (pr.q == 50) & (pr.seed == 10501) & (pr.role == "ms_arm")
    row = pr[sel].iloc[0]
    last = df[df.stage == 2].iloc[-1]
    assert int(last["update"]) == 2800
    g2, e2 = float(row["g2_at_0"]), float(row["e2_at_0"])
    ref_w_eff = (g2 - e2) / (g2 / (2.0 * 50))
    d_e = abs(float(last["e_hat_2_0"]) - e2)
    d_w = abs(float(last["w_eff"]) - ref_w_eff)
    print(f"\nreal run q50/seed10501/NL_bb_s1 @u2800: |e_hat(export) - e2_at_0(per_run)| = "
          f"{d_e:.3e} effort units; |w_eff - reference| = {d_w:.3e}; w_eff = {last['w_eff']:.6f}")
    assert last["e_star_2_0"] == pytest.approx(g2, rel=1e-12)
    assert d_e < 1e-4 and d_w < 2e-4, (d_e, d_w)
    fz = np.load(base / "pilot" / "q50" / "seed10501" / "NL_bb_s1" / "freeze_stage2_final.npz")
    i0 = int(np.argmin(np.abs(fz["recovery_d_grid"])))            # the verifier's own freeze array
    assert abs(float(last["e_hat_2_0"]) - float(fz["recovery_e2"][i0])) < 1e-4


def test_the_closed_form_tent_peak_is_the_stored_g2_at_0_for_both_q():
    """tent_peak (g2_two_stage) equals g2_at_0 of per_run.csv and of a q = 60 freeze record."""
    base = _real_ms_r2()
    if base is None:
        pytest.skip("results/ms_r2 absent")
    pr = pd.read_csv(base / "analysis" / "per_run.csv", float_precision="round_trip")
    pr = pr[pr.role == "ms_arm"]
    for q in (50, 60):
        assert (pr[pr.q == q]["g2_at_0"] == D.tent_peak(spec_for(q))).all(), q
    run60 = base / "pilot" / "q60" / "seed10501" / "NL_bb_s1"
    if run60.is_dir():
        fz = np.load(run60 / "freeze_stage2_final.npz")
        i0 = int(np.argmin(np.abs(fz["recovery_d_grid"])))
        assert float(fz["recovery_g2"][i0]) == D.tent_peak(spec_for(60))
