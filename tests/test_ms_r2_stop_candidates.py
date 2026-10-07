"""MS-R2 D6 tool (tools/ms/r2_stop_candidates.py): exact terminal best response, J, C1/C2/C3, fire logic, tables, CLI.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_r2_stop_candidates.py -p no:cacheprovider -q

Sections: 1. exact best response against a dense grid; 2. J against the linearisation, the exclusion; 3. conc_scale;
4. fire logic, io, tables and refusals on synthetic data; 5. one stored MS-R1 pilot run (skipped when the root is
absent): replay against the run's own check rows, C1 against the freeze arrays, C2 against ``per_run.csv``, the CLI.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import r2_stop_candidates as SC  # noqa: E402
from agents.ppo_curriculum import BetaActor  # noqa: E402
from envs.curriculum_env import GameSpec  # noqa: E402
from replay_dev_rule import build_actor, candidate_fns, dense_grid  # noqa: E402
from utils.theory_multistage import f_xi  # noqa: E402

PROTOCOL = ROOT / "protocols" / "v2_T2_locked_v2_0.json"
PILOT_ROOT = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/p2-gate-ms-base2400-e41857/results/ms_r1/pilot")
REF_RUN = PILOT_ROOT / "q50" / "seed10501" / "MS_rule"
PER_RUN = ROOT / "results" / "ms_r1" / "analysis" / "per_run.csv"
needs_pilot = pytest.mark.skipif(not (REF_RUN / "weights" / "u02400.npz").is_file(),
                                 reason=f"MS-R1 pilot root {PILOT_ROOT} (read-only input) is absent")


def _spec(q: int) -> GameSpec:
    with open(PROTOCOL) as f:
        return GameSpec(**json.load(f)["records"][str(q)]["game"])


def _g2(d: np.ndarray, spec: GameSpec) -> np.ndarray:
    """The closed-form terminal effort DW f_xi(d) / (2k) (the verifier's exact equilibrium)."""
    return spec.dw * f_xi(d, spec.q) / (2.0 * spec.k)


# ====================================================================== 1. exact best response
@pytest.mark.parametrize("spec", [_spec(50), _spec(60),
                                  GameSpec(w_h=6, w_l=2, k=0.0003, q=50, T=2, e_min=5.0, e_max=20.0)])
def test_exact_best_response_matches_dense_grid_with_parabolic_refinement(spec: GameSpec) -> None:
    rng = np.random.default_rng(11)
    n = 400
    d = rng.uniform(-(spec.e_range + 2 * spec.q), spec.e_range + 2 * spec.q, n)
    o = rng.uniform(spec.e_min, spec.e_max, n)
    d[:40] = rng.uniform(-8.0, 8.0, 40)                                  # near the kink x = 0
    exact = SC.exact_best_response(d, o, spec.dw, spec.k, spec.q, spec.e_min, spec.e_max)
    dense = np.concatenate([SC.dense_best_response(d[i:i + 100], o[i:i + 100], spec.dw, spec.k, spec.q,
                                                   spec.e_min, spec.e_max) for i in range(0, n, 100)])
    assert np.max(np.abs(exact - dense)) < 1e-6
    q_ex = SC.terminal_q(exact, d, o, spec.dw, spec.k, spec.q)
    q_de = SC.terminal_q(dense, d, o, spec.dw, spec.k, spec.q)
    assert np.all(q_ex >= q_de - 1e-12)                                  # the exact root is the maximiser
    inside = (exact > spec.e_min + 1e-9) & (exact < spec.e_max - 1e-9)
    assert np.max(np.abs(SC.terminal_dq(exact[inside], d[inside], o[inside], spec.dw, spec.k, spec.q))) < 1e-12
    assert exact.min() >= spec.e_min and exact.max() <= spec.e_max
    if spec.e_max == 20.0:                                               # the bound-clipped game really clips
        assert (exact == spec.e_max).sum() > 5 and (exact == spec.e_min).sum() > 5


def test_exact_best_response_is_accurate_to_1e_10_on_a_known_piece() -> None:
    spec = _spec(50)
    a = spec.dw / (4 * spec.q ** 2)
    d, o = -30.0, 60.0
    # d + e - o < 0: dw (2q + x)/(4q^2) = 2k e  ->  e = a (2q + d - o) / (2k - a)
    e = a * (2 * spec.q + d - o) / (2 * spec.k - a)
    assert d + e - o < 0 and 0 < e < spec.e_max
    got = SC.exact_best_response(d, o, spec.dw, spec.k, spec.q, spec.e_min, spec.e_max)[0]
    assert abs(got - e) < 1e-10


@pytest.mark.parametrize("q", [50, 60])
def test_the_repo_games_satisfy_the_monotone_assumption_and_the_linearisation_numbers(q: int) -> None:
    spec = _spec(q)
    a = spec.dw / (4 * q * q)
    assert 2 * spec.k > a and SC.br_is_monotone(spec.dw, spec.k, q)
    neg, pos = SC.linearised_j(spec.dw, spec.k, q)
    assert neg - 1 == pytest.approx(-2 * spec.k / (2 * spec.k - a), rel=1e-14)
    assert pos - 1 == pytest.approx(-2 * spec.k / (2 * spec.k + a), rel=1e-14)
    assert neg - 1 == pytest.approx({50: -3.3333333, 60: -1.9459459}[q], abs=1e-6)   # MS-R1's factors 3.33 / 1.95


def test_best_response_falls_back_to_the_dense_grid_when_q_prime_is_not_monotone() -> None:
    spec = GameSpec(w_h=6, w_l=2, k=1e-4, q=50, T=2)                     # 2k = 2e-4 < a = 4e-4
    assert not SC.br_is_monotone(spec.dw, spec.k, spec.q)
    d, o = np.array([-20.0, 0.0, 15.0]), np.array([40.0, 50.0, 30.0])
    got = SC.best_response(d, o, spec.dw, spec.k, spec.q, 0.0, 100.0)
    assert np.array_equal(got, SC.dense_best_response(d, o, spec.dw, spec.k, spec.q, 0.0, 100.0))
    e_grid = np.linspace(0.0, 100.0, 100001)
    for i in range(3):                                                   # the global maximum, not a local root
        qv = SC.terminal_q(e_grid, d[i], o[i], spec.dw, spec.k, spec.q)
        assert SC.terminal_q(got[i], d[i], o[i], spec.dw, spec.k, spec.q)[0] >= qv.max() - 1e-9


# ====================================================================== 2. J and the exclusion
@pytest.mark.parametrize("q", [50, 60])
def test_j_equals_the_linearisation_on_the_equilibrium_policy(q: int) -> None:
    spec = _spec(q)
    d = dense_grid(spec.domain_half(2), 4.0)
    e = _g2(d, spec)                                                     # the verifier's own exact equilibrium
    nontail = np.abs(d) < 2 * q - SC.NODE_TOL
    jac = SC.br_jacobian(d, e, spec.dw, spec.k, q, spec.e_min, spec.e_max)
    neg, pos = SC.linearised_j(spec.dw, spec.k, q)
    away = nontail & (np.abs(d) >= 4.0)
    assert np.max(np.abs(jac[away & (d < 0)] - neg)) < 1e-6
    assert np.max(np.abs(jac[away & (d > 0)] - pos)) < 1e-6
    # at the kink node itself the central difference straddles the two branches: between the two slopes
    j0 = int(np.argmin(np.abs(d)))
    assert neg - 1e-9 <= jac[j0] <= pos + 1e-9
    cand = SC.terminal_candidates(d, e, e[::-1], e, spec)                # e_opp(d) = e_hat(-d) = e_hat(d), a_dev = e_hat
    assert cand.R0 == 0.0 and cand.R_defl == 0.0 and cand.n_defl_excluded == 0
    assert cand.n_defl_nodes == int(nontail.sum())
    assert cand.J_mean_neg == pytest.approx(neg, abs=1e-6) and cand.J_mean_pos == pytest.approx(pos, abs=1e-6)
    assert cand.s_exact == pytest.approx(e[j0], abs=1e-6) and cand.R0_exact == pytest.approx(0.0, abs=1e-8)


def test_j_equals_the_linearisation_on_a_constant_effort_policy_where_the_branch_is_clear() -> None:
    spec = _spec(50)
    c = 40.0
    d = np.arange(-98.0, 98.1, 2.0)
    o = np.full(d.shape, c)
    h = SC.H_FD
    jac = SC.br_jacobian(d, o, spec.dw, spec.k, spec.q, spec.e_min, spec.e_max, h)
    neg, pos = SC.linearised_j(spec.dw, spec.k, spec.q)
    checked = {"neg": 0, "pos": 0}
    for sgn, name, want in ((-1, "neg", neg), (1, "pos", pos)):
        for dd, jj, oo in zip(d, jac, o):
            if sgn * dd < SC.J_AWAY:
                continue
            ex = [SC.exact_best_response(dd, oo + s * h, spec.dw, spec.k, spec.q, spec.e_min, spec.e_max)[0]
                  for s in (-1, 1)]
            xs = [dd + e - oo - s * h for e, s in zip(ex, (-1, 1))]   # the gap argument of the two solves
            same_piece = all(sgn * x > 1.0 and abs(x) < 2 * spec.q - 1.0 for x in xs) and min(ex) > 0.0
            if same_piece:
                assert abs(jj - want) < 1e-6
                checked[name] += 1
    assert checked["neg"] >= 3 and checked["pos"] >= 3                   # the check is not vacuous


def test_deflate_excludes_nodes_with_j_near_one_and_counts_them() -> None:
    r = np.array([1.0, 2.0, 3.0, 4.0, 50.0])
    jac = np.array([-2.0, 0.95, 1.05, 0.5, 0.0])                         # |1 - J| = 3, 0.05, 0.05, 0.5, 1
    nontail = np.array([True, True, True, True, False])                  # the last node is a tail node
    val, n_ex, n_nodes = SC.deflate(r, 10.0, jac, nontail)
    assert (n_ex, n_nodes) == (2, 4)
    assert val == pytest.approx(max(1.0 / (10 * 3.0), 4.0 / (10 * 0.5)))
    val, n_ex, _ = SC.deflate(r, 10.0, np.array([0.95, 1.0, 1.05, 0.99, 0.0]), nontail)
    assert math.isnan(val) and n_ex == 4                                 # everything excluded -> NaN, not 0
    # the threshold is 0.1 on |1 - J|: 0.11 is kept, 0.09 is excluded
    assert SC.deflate(np.array([1.0]), 1.0, np.array([0.89]), np.array([True]))[1] == 0
    assert SC.deflate(np.array([1.0]), 1.0, np.array([0.91]), np.array([True]))[1] == 1


def test_terminal_candidates_are_nan_when_s_is_not_positive_and_use_the_stage_node_set() -> None:
    spec = _spec(50)
    d = dense_grid(spec.domain_half(2), 4.0)
    e = _g2(d, spec)
    bad = SC.terminal_candidates(d, e, e, np.zeros_like(e), spec)
    assert math.isnan(bad.R0) and math.isnan(bad.R_defl) and bad.n_defl_nodes == 49
    # q = 50: |d| < 100 on the step-4 grid is 49 nodes; the tail nodes (|d| >= 100) never enter
    e_hat = e + 1.0
    cand = SC.terminal_candidates(d, e_hat, e_hat[::-1], e, spec)
    s = e[int(np.argmin(np.abs(d)))]
    assert cand.R0 == pytest.approx(1.0 / s)
    assert cand.R_defl >= cand.R0 / 3.4                                  # |1 - J| <= 3.34 on the repo's game


# ====================================================================== 3. conc_scale
def _synthetic_export(seed: int = 0, scale: Optional[float] = None) -> Dict[str, np.ndarray]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    net = BetaActor(64, 100.0, 1e-6, gen)
    with torch.no_grad():
        net.out.weight.normal_(0.0, 0.2, generator=gen)
        net.out.bias.copy_(torch.tensor([0.3, 0.5]))
    arrays = {f"actor.{k}": v.numpy().copy() for k, v in net.state_dict().items()}
    if scale is not None:
        arrays["conc_scale"] = np.array(scale)
    return arrays


def test_conc_scale_multiplies_alpha_plus_beta_and_keeps_the_mean() -> None:
    spec = _spec(50)
    base = candidate_fns(spec, {2: build_actor(_synthetic_export(), 64, 100.0, 1e-6)})
    scaled = candidate_fns(spec, {2: build_actor(_synthetic_export(scale=4.0), 64, 100.0, 1e-6)})
    d = np.linspace(-150.0, 150.0, 31)
    a0, b0 = base[1](2, d)
    a4, b4 = scaled[1](2, d)
    assert np.allclose(a4 + b4, 4.0 * (a0 + b0), rtol=1e-6)              # alpha + beta = 4 (c_min + softplus)
    assert np.allclose(a4 / (a4 + b4), a0 / (a0 + b0), rtol=1e-6)        # same mean
    assert np.allclose(base[0](2, d), scaled[0](2, d), rtol=1e-6)
    assert SC.conc_scale_of(_synthetic_export(scale=4.0)) == 4.0 and SC.conc_scale_of(_synthetic_export()) == 1.0
    row = SC.export_row(spec, build_actor(_synthetic_export(scale=4.0), 64, 100.0, 1e-6))
    row0 = SC.export_row(spec, build_actor(_synthetic_export(), 64, 100.0, 1e-6))
    assert row["sigma0"] == pytest.approx(row0["sigma0"] / 2.0, rel=0.02)    # sd ~ 1 / sqrt(alpha + beta + 1)
    assert row["e_hat0"] == pytest.approx(row0["e_hat0"], rel=1e-6)


# ====================================================================== 4. fire logic, io, tables
def test_fire_export_needs_m_consecutive_and_resets_on_a_failing_or_invalid_export() -> None:
    us = list(range(25, 300, 25))                                        # 11 exports
    ok = [True] * 11
    f = SC.fire_export
    assert f(us, [0.1] * 11, ok, 0.2) == 75                              # third export
    assert f(us, [0.1, 0.1, 0.5, 0.1, 0.1, 0.1] + [0.5] * 5, ok, 0.2) == 150          # reset by the failing export
    assert f(us, [0.5] * 11, ok, 0.2) is None                            # never fires
    assert f(us, [0.1] * 11, [True, True, False] + [True] * 8, 0.2) == 150           # an invalid export resets
    assert f(us, [0.1, 0.1, float("nan")] + [0.1] * 8, ok, 0.2) == 150  # a non-finite value is a failing check
    assert f(us, [0.2] * 11, ok, 0.2) == 75                              # <= theta: equality counts
    assert f(us, [0.1] * 11, ok, 0.2, m=1) == 25
    assert f([25, 30, 50, 75], [0.1] * 4, [True] * 4, 0.2) == 75         # u = 30 is off the cadence: not a check
    assert f([25, 30, 50, 75], [0.1, 0.9, 0.1, 0.1], [True] * 4, 0.2) == 75


def test_parse_source_and_seeds_and_discover() -> None:
    assert SC.parse_source("pilot=/a/b:MS_rule,MS_base2400") == ("pilot", "/a/b", ["MS_rule", "MS_base2400"])
    with pytest.raises(ValueError):
        SC.parse_source("no-equals:MS_rule")
    with pytest.raises(ValueError):
        SC.parse_source("name=/only/root")
    assert SC.parse_seeds(["10501-10503", "10510,10512"]) == [10501, 10502, 10503, 10510, 10512]
    runs = SC.discover([("pilot", "/r", ["A", "B"])], [50, 60], [1, 2])
    assert len(runs) == 8 and runs[0].run_dir == Path("/r/q50/seed1/A") and runs[0].key == "pilot/A/q50/seed1"


def test_write_new_csv_and_json_refuse_to_overwrite(tmp_path: Path) -> None:
    p = str(tmp_path / "x" / "t.csv")
    SC.write_new_csv(p, [{"a": 1, "b": 0.1}], ["a", "b"])
    with pytest.raises(FileExistsError):
        SC.write_new_csv(p, [{"a": 2, "b": 0.2}], ["a", "b"])
    assert open(p).read().splitlines() == ["a,b", "1,0.1"]
    j = str(tmp_path / "m.json")
    SC.write_new_json(j, {"a": 1})
    with pytest.raises(FileExistsError):
        SC.write_new_json(j, {"a": 2})


def _fake_tree(root: Path, n_exports: int = 30) -> None:
    """2 sources x 2 q x 3 seeds, exports every 25 updates; the exports of u <= 600 are constant LR.

    C1 = 0.5 |peak| (Spearman 1), C3 = 1 - |peak| (Spearman -1), c2 falls with the export index; export index 5
    (u = 150) is invalid in every run and index 20 (u = 525) in the seed-1 runs.
    """
    rng = np.random.default_rng(5)
    for src, arm in (("base", "MS_base"), ("pilot", "MS_rule")):
        for q in (50, 60):
            for seed in (1, 2, 3):
                peak = np.sort(rng.uniform(0.01, 0.5, n_exports))[::-1] * (1.0 + 0.1 * seed)
                rows = []
                for i in range(n_exports):
                    u = 25 * (i + 1)
                    row = {c: float("nan") for c in SC.CSV_COLS}
                    row.update(source=src, arm=arm, q=q, seed=seed, update=u, actor_lr=3e-4 if u <= 600 else 1e-4,
                               conc_scale=1.0, valid=bool(i != 5 and not (seed == 1 and i == 20)), error="",
                               n_defl_excluded=0, n_defl_nodes=49, n_J_neg=22, n_J_pos=22)
                    row.update(Delta=0.1 / (i + 1), s=60.0, R=peak[i] * 3, R_tail=0.01, C=0.03, R0=0.5 * peak[i],
                               e_hat0=60.0, sigma0=3.0, e_sigma0=66.0, c2=0.1 - 0.001 * i, floor=0.03,
                               R_defl=1.0 - peak[i], J_mean_neg=-2.2, J_mean_pos=0.4, J_lin_neg=-2.3333,
                               J_lin_pos=0.4118, s_exact=60.5, R0_exact=0.4 * peak[i],
                               stage2_peak_rel_err_signed=-peak[i], stage2_peak_rel_err_abs=peak[i],
                               stage2_rmse_pos_over_g2_0=peak[i] / 2, stage2_tail_mean_over_g2_0=peak[i] / 5,
                               stage2_tail_max_over_g2_0=peak[i], g2_at_0=70.0, e2_at_0=60.0)
                    rows.append(row)
                SC.write_new_csv(SC.run_csv_path(str(root), SC.RunRef(src, "/unused", arm, q, seed)), rows,
                                 SC.CSV_COLS)


def test_tables_command_on_a_synthetic_tree(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _fake_tree(tmp_path)
    assert SC.main(["tables", "--out", str(tmp_path), "--md"]) == 0
    out = capsys.readouterr().out
    t = tmp_path / "tables"
    assert {p.name for p in t.iterdir()} == {"a_spearman.csv", "b_freeze.csv", "c_jcheck.csv", "x_diagnostics.csv"}
    cell = {(r["group"], r["q"], r["subset"], r["candidate"], r["target"]): r
            for r in csv.DictReader(open(t / "a_spearman.csv"))}
    c1 = cell[("all", "50", "all", "C1", "|peak|")]
    assert float(c1["spearman_pooled"]) == pytest.approx(1.0, abs=1e-12) and int(c1["n_pairs"]) == 88
    assert float(cell[("all", "50", "all", "C3", "|peak|")]["spearman_pooled"]) == pytest.approx(-1.0, abs=1e-12)
    assert float(cell[("all", "50", "all", "C2", "RMSE_pos")]["within_run_median"]) == pytest.approx(1.0)
    assert float(cell[("base/MS_base", "60", "constLR", "C1", "RMSE_pos")]["spearman_pooled"]) == pytest.approx(1.0)
    assert int(cell[("all", "50", "constLR", "C1", "|peak|")]["n_pairs"]) == 52
    assert "Table (a)" in out and "Table (b)" in out and "Table (c)" in out and "| q | candidate |" in out
    fr = {(r["group"], r["q"], r["candidate"]): r for r in csv.DictReader(open(t / "b_freeze.csv"))}
    assert int(fr[("all", "50", "C1")]["n"]) == 6                      # 2 sources x 3 seeds, the last export is valid
    assert float(fr[("all", "50", "C2 floor")]["median"]) == pytest.approx(0.03)
    assert float(fr[("all", "50", "C2 (|c2|)")]["max"]) == pytest.approx(0.1 - 0.001 * 29)
    jc = list(csv.DictReader(open(t / "c_jcheck.csv")))
    assert [r["side"] for r in jc[:2]] == ["d < -8", "d > +8"]
    assert float(jc[0]["mean_J"]) == pytest.approx(-2.2) and float(jc[0]["max_abs_dev"]) == pytest.approx(0.1333, abs=1e-4)
    assert float(jc[1]["mean_abs_dev"]) == pytest.approx(0.0118, abs=1e-4)
    dg = list(csv.DictReader(open(t / "x_diagnostics.csv")))
    assert float(dg[0]["abs_s_minus_s_exact_max"]) == pytest.approx(0.5) and int(dg[0]["n_defl_excluded_total"]) == 0
    assert SC.main(["tables", "--out", str(tmp_path)]) == 2             # a second call would overwrite: refuse


def test_spearman_cell_uses_valid_exports_with_u_ge_400_and_the_constant_lr_subset(tmp_path: Path) -> None:
    _fake_tree(tmp_path)
    rows = SC.select_runs(SC.load_runs(str(tmp_path)), None, 50)
    assert len(rows) == 6
    n, rho, wr = SC.spearman_cell(rows, "R0", "stage2_peak_rel_err_abs", False)
    assert (n, rho, wr) == (6 * 15 - 2, pytest.approx(1.0), pytest.approx(1.0))   # u >= 400: 15 exports; 2 invalid
    n_c, rho_c, _ = SC.spearman_cell(rows, "R0", "stage2_peak_rel_err_abs", True)
    assert n_c == 6 * 9 - 2 and rho_c == pytest.approx(1.0)                       # 400 <= u <= 600 only
    early = [[r for r in rs if int(r["update"]) < 400] for rs in rows]
    assert SC.spearman_cell(early, "R0", "stage2_peak_rel_err_abs", False)[0] == 0


def test_fire_tables_need_grids_and_record_the_grid_file(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _fake_tree(tmp_path)
    assert SC.main(["tables", "--out", str(tmp_path), "--fire"]) == 2          # refuses without --grids
    assert not (tmp_path / "tables").exists()
    assert SC.main(["tables", "--out", str(tmp_path), "--grids", "g.json"]) == 2   # --grids alone does not compute
    assert not (tmp_path / "tables").exists()
    g = tmp_path / "grids.json"
    g.write_text(json.dumps({"C1": [0.05, 0.2], "C3": [0.5], "C2s": [0.0895]}))
    assert SC.main(["tables", "--out", str(tmp_path), "--fire", "--grids", str(g), "--md"]) == 0
    out = capsys.readouterr().out
    meta = json.load(open(tmp_path / "tables" / "fire_meta.json"))
    import hashlib
    assert meta["grids_sha256"] == hashlib.sha256(g.read_bytes()).hexdigest()
    assert meta["grids_git_commit"] == "" and meta["grids"] == {"C1": [0.05, 0.2], "C3": [0.5], "C2s": [0.0895]}
    assert meta["rule"]["M"] == 3 and meta["rule"]["K"] == 25
    fire = list(csv.DictReader(open(tmp_path / "tables" / "fire.csv")))
    # C1 = 0.5 |peak| decreases along the run; theta = 0.2 fires where |peak| <= 0.4, hence in every run; q = 50, all
    c = {(r["candidate"], r["theta"], r["group"], r["q"]): r for r in fire}
    r = c[("C1", "0.2", "all", "50")]
    assert int(r["n_runs"]) == 6 and int(r["n_fired"]) == 6
    assert float(r["abs_peak_at_fire_median"]) >= float(r["abs_peak_at_end_median"]) > 0     # the peak falls on
    # c2 = 0.1 - 0.001 i <= 0.0895 from export index 11 on: the third consecutive one is index 13 (u = 350)
    s = c[("C2s", "0.0895", "all", "50")]
    assert int(s["n_fired"]) == 6 and float(s["fire_u_median"]) == 350
    assert "Table (d)" in out
    assert SC.main(["tables", "--out", str(tmp_path), "--fire", "--grids", str(g)]) == 2    # never overwrites
    g2 = tmp_path / "bad.json"
    g2.write_text(json.dumps({"C9": [0.1]}))
    with pytest.raises(ValueError):
        SC.load_grids(str(g2))


# ====================================================================== 5. one stored pilot run
def _row_at(u: int) -> Dict[str, object]:
    with open(REF_RUN / "run_config.json") as f:
        rec = json.load(f)["record"]
    spec = GameSpec(**rec["game"])
    arrays = SC.load_export(str(REF_RUN / "weights" / f"u{u:05d}.npz"))
    net = build_actor(arrays, int(rec["ppo"]["hidden"]), float(rec["ppo"]["c_min"]), float(rec["ppo"]["mu_clamp"]))
    row: Dict[str, object] = {"update": u}
    row.update(SC.export_row(spec, net))
    return row


@needs_pilot
def test_replayed_d3_columns_equal_the_runs_own_check_rows() -> None:
    rows = [_row_at(u) for u in (25, 400, 1600, 2000, 2400)]
    checks = SC._read_checks(REF_RUN / SC.CHECK_FILE)
    v = SC.validate_rows(rows, checks)
    assert v["n_matched"] == 5 and v["n_valid_mismatch"] == 0
    assert v["max_abs_d3"] <= 1e-12 and v["max_abs_closed"] <= 1e-12


@needs_pilot
def test_c1_equals_r0_of_the_freeze_arrays_and_c2_equals_smoothed_e_pred_0() -> None:
    row = _row_at(2400)
    z = np.load(REF_RUN / "freeze_stage2_development.npz")
    d, e_hat, a_dev = z["v_t2_d_grid"], z["v_t2_e_hat"], z["v_t2_a_dev"]
    j0 = int(np.argmin(np.abs(d)))
    r0_freeze = abs(e_hat[j0] - a_dev[j0]) / a_dev[j0]
    assert row["R0"] == pytest.approx(float(r0_freeze), abs=1e-12)
    if PER_RUN.is_file():
        pr = [r for r in csv.DictReader(open(PER_RUN)) if (r["arm"], r["q"], r["seed"]) == ("MS_rule", "50", "10501")]
        assert len(pr) == 1
        assert abs(float(row["e_sigma0"]) - float(pr[0]["smoothed_e_pred_0"])) < 1e-9
        assert float(row["R0"]) == pytest.approx(float(pr[0]["t2_R0_dev"]), abs=1e-12)
    # C2 is the residual of the policy at the tie, the floor is sigma_0 / (sqrt(pi) q)
    assert row["c2"] == pytest.approx((row["e_sigma0"] - row["e_hat0"]) / row["e_sigma0"], rel=1e-14)
    assert row["floor"] == pytest.approx(row["sigma0"] / (math.sqrt(math.pi) * 50), rel=1e-14)
    # C3 sits on the repo's game: no node is excluded, J is near the linearisation away from the kink
    assert row["n_defl_excluded"] == 0 and row["n_defl_nodes"] == 49
    assert abs(row["J_mean_neg"] - row["J_lin_neg"]) < 0.2 and abs(row["J_mean_pos"] - row["J_lin_pos"]) < 0.02
    assert row["R_defl"] > 0 and row["s_exact"] == pytest.approx(row["s"], abs=0.5)


@needs_pilot
def test_cli_end_to_end_on_two_exports_of_one_real_run(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "stop_calibration"
    argv = ["replay", "--source", f"pilot={PILOT_ROOT}:MS_rule", "--qs", "50", "--seeds", "10501", "--out", str(out),
            "--workers", "1", "--updates", "400", "2400"]
    assert SC.main(argv) == 0
    csv_path = out / "pilot" / "MS_rule" / "q50" / "seed10501.csv"
    rows = list(csv.DictReader(open(csv_path)))
    assert [r["update"] for r in rows] == ["400", "2400"] and [r["actor_lr"] for r in rows][0] == "0.0003"
    assert set(rows[0]) == set(SC.CSV_COLS)
    man = json.load(open(out / "manifest.json"))
    assert man["n_runs"] == 1 and man["n_rows"] == 2 and man["validation"]["ok"] is True
    assert man["validation"]["max_abs_d3"] <= 1e-12 and man["restricted_updates"] == [400, 2400]
    assert man["sources"]["pilot"] == {"root": str(PILOT_ROOT), "arms": ["MS_rule"]}
    assert len(man["tool_sha256"]) == 64 and man["git_head"]
    assert SC.main(argv) == 2                                              # refuses to overwrite
    assert "already exist" in capsys.readouterr().out
    assert SC.main(["tables", "--out", str(out), "--md"]) == 0             # n_pairs small, but the command runs
    assert (out / "tables" / "b_freeze.csv").is_file()
    freeze = {r["candidate"]: r for r in csv.DictReader(open(out / "tables" / "b_freeze.csv")) if r["group"] == "all"}
    assert float(freeze["C1"]["median"]) == pytest.approx(float(rows[-1]["R0"]))


def test_replay_stops_and_reports_a_missing_input(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    argv = ["replay", "--source", f"x={tmp_path}:NOPE", "--qs", "50", "--seeds", "1", "--out", str(tmp_path / "o")]
    assert SC.main(argv) == 2
    assert "missing" in capsys.readouterr().out and not (tmp_path / "o").exists()
